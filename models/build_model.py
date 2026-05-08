from __future__ import annotations

import math
import os
import pickle as pkl
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from spikingjelly.activation_based import functional

from .common.registry import get_model_cls
from .common.spike_ops import expand_static_to_temporal
from utils import accuracy_at_k

# trigger model registration
from . import spikformer  # noqa: F401
from . import spike_driven_transformer  # noqa: F401
from . import spiking_cnn  # noqa: F401
from . import billeh_v1  # noqa: F401
from . import transformer  # noqa: F401


def init_weights(model: nn.Module) -> None:
    fn = getattr(model, "init_weights", None)
    if callable(fn):
        fn()


def make_optimizer_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    no_decay_types = (nn.Embedding, nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if isinstance(module, no_decay_types) or param_name == "bias" or param.ndim < 2:
                no_decay.append(param)
            else:
                decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class LitVisionSNN(L.LightningModule):
    def __init__(self, model: nn.Module, optimizer_config: Any, train_config: Any, data_config: Any, model_config: Any | None = None):
        super().__init__()
        self.model = model
        self.optimizer_config = optimizer_config
        self.train_config = train_config
        self.data_config = data_config
        self.model_config = model_config
        self.save_hyperparameters(ignore=["model"])

        self.T = int(getattr(model, "T", getattr(optimizer_config, "T", 4)))
        self._is_event_input = bool(getattr(data_config, "is_event", False))
        self._step_start_time = None

        # Strict-reproduction loss weights and target firing-rate buffer.
        self.rate_cost = float(getattr(optimizer_config, "rate_cost", 0.0))
        self.voltage_cost = float(getattr(optimizer_config, "voltage_cost", 0.0))
        self._huber_kappa = float(getattr(optimizer_config, "huber_kappa", 0.002))
        if self.rate_cost > 0.0:
            self._init_target_firing_rates()
        else:
            self.register_buffer("target_firing_rates", torch.zeros(0), persistent=False)

    # ---------- target firing rates ----------

    def _init_target_firing_rates(self) -> None:
        mcfg = self.model_config
        ocfg = self.optimizer_config
        n_neurons = int(getattr(mcfg, "n_neurons", 0)) if mcfg is not None else 0
        if n_neurons <= 0:
            n_neurons = int(getattr(self.model, "n_neurons", 0))
        if n_neurons <= 0:
            raise ValueError("Cannot determine n_neurons for target firing rates")
        seed = int(getattr(mcfg, "seed", 3000)) if mcfg is not None else 3000

        user_path = str(getattr(ocfg, "target_rates_path", "") or "").strip()
        candidate = user_path
        if not candidate and mcfg is not None:
            data_dir = str(getattr(mcfg, "billeh_data_dir", "") or "").strip()
            if data_dir:
                candidate = os.path.join(data_dir, "garrett_firing_rates.pkl")
        if not candidate or not os.path.isfile(candidate):
            raise FileNotFoundError(
                "garrett_firing_rates.pkl not found. Set optimizer_config.target_rates_path "
                f"or place the file under model_config.billeh_data_dir. Looked at: {candidate!r}"
            )

        with open(candidate, "rb") as f:
            rates = pkl.load(f)
        rates = np.asarray(rates, dtype=np.float32).reshape(-1)
        if rates.size == 0:
            raise ValueError(f"Empty firing-rate file: {candidate}")

        sorted_r = np.sort(rates)
        pct = (np.arange(sorted_r.size) + 1) / sorted_r.size
        rng = np.random.RandomState(seed)
        x_rand = rng.uniform(size=n_neurons)
        target = np.sort(np.interp(x_rand, pct, sorted_r)).astype(np.float32)  # in Hz
        self.register_buffer("target_firing_rates", torch.from_numpy(target))
        rank_zero_info(
            f"Loaded target firing rates from {candidate}: "
            f"N={target.size}, range=[{target.min():.3f}, {target.max():.3f}] Hz"
        )

    # ---------- forward + losses ----------

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_event_input:
            return x  # already [T, B, C, H, W]
        # LRA-style tasks may provide non-image tensors (e.g., [B, L] token ids).
        # Only expand static images.
        if x.ndim == 4:
            return expand_static_to_temporal(x, self.T)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model.__class__.__name__ != "SpikeDrivenTransformerV3":
            x = self._prepare_input(x)
        logits = self.model(x)
        functional.reset_net(self.model)
        return logits

    def _huber_quantile_rate_loss(self, spikes: torch.Tensor) -> torch.Tensor:
        # Match upstream TF: rate = mean(spikes, (0,1)) is per-neuron spikes per
        # timestep, compared directly against target_firing_rates from
        # garrett_firing_rates.pkl which is also in spikes/timestep units (max
        # ≈ 0.066, i.e. ~66 Hz at dt=1 ms). DO NOT convert to Hz here.
        rate = spikes.float().mean(dim=(0, 1))
        N = rate.numel()
        target = self.target_firing_rates.to(rate.device)
        if target.numel() != N:
            raise ValueError(
                f"target_firing_rates size {target.numel()} != N={N}"
            )
        perm = torch.randperm(N, device=rate.device)
        sorted_rate, _ = torch.sort(rate.index_select(0, perm))
        u = sorted_rate - target
        tau = (torch.arange(N, device=u.device, dtype=u.dtype) + 1.0) / N
        kappa = self._huber_kappa
        huber = torch.where(
            u.abs() <= kappa,
            0.5 * u.pow(2),
            kappa * (u.abs() - 0.5 * kappa),
        )
        quantile = torch.where(u >= 0, tau * huber / kappa, (1.0 - tau) * huber / kappa)
        return quantile.sum()

    def _shared_step(self, batch, split: str):
        x, y = batch
        logits_inf = self(x)
        m = self.model

        # Base classification loss; if pool readout exposes chunked logits + mask, use those.
        logits_chunks = getattr(m, "last_logits_chunks", None)
        resp_mask = getattr(m, "_resp_chunk_mask", None)
        if logits_chunks is not None and resp_mask is not None and resp_mask.numel() == logits_chunks.shape[1]:
            n_chunks = logits_chunks.shape[1]
            log_p = F.log_softmax(logits_chunks, dim=-1)
            y_btc = y.view(-1, 1).expand(-1, n_chunks)
            n_classes = logits_chunks.shape[-1]
            nll = F.nll_loss(
                log_p.reshape(-1, n_classes),
                y_btc.reshape(-1),
                reduction="none",
            ).view(-1, n_chunks)
            mask = resp_mask.to(nll.device).view(1, -1)
            cls_loss = ((nll * mask).sum(dim=1) / mask.sum().clamp_min(1)).mean()
        else:
            cls_loss = F.cross_entropy(logits_inf, y)

        loss = cls_loss
        voltage_loss = logits_inf.new_zeros(())
        rate_loss = logits_inf.new_zeros(())

        if self.voltage_cost > 0.0:
            v_norm = getattr(m, "last_v_norm", None)
            if v_norm is not None:
                v_pos = F.relu(v_norm - 1.0).pow(2)
                v_neg = F.relu(-v_norm + 1.0).pow(2)
                voltage_loss = (v_pos + v_neg).sum(dim=-1).mean() * self.voltage_cost
                loss = loss + voltage_loss

        if self.rate_cost > 0.0:
            spikes = getattr(m, "last_spikes", None)
            if spikes is not None and self.target_firing_rates.numel() > 0:
                rate_loss = self._huber_quantile_rate_loss(spikes) * self.rate_cost
                loss = loss + rate_loss

        # Logging.
        num_classes = logits_inf.size(1)
        topk = (1, 5) if num_classes >= 5 else (1,)
        accs = accuracy_at_k(logits_inf, y, topk=topk)

        sync = split != "train"
        on_step = split == "train"
        self.log(f"{split}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=True, sync_dist=sync)
        self.log(f"{split}/cls_loss", cls_loss, on_step=on_step, on_epoch=True, sync_dist=sync)
        if self.voltage_cost > 0.0:
            self.log(f"{split}/voltage_loss", voltage_loss, on_step=on_step, on_epoch=True, sync_dist=sync)
        if self.rate_cost > 0.0:
            self.log(f"{split}/rate_loss", rate_loss, on_step=on_step, on_epoch=True, sync_dist=sync)
        self.log(f"{split}/top1", accs["top1"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=sync)
        if num_classes >= 5:
            self.log(f"{split}/top5", accs["top5"], on_step=False, on_epoch=True, sync_dist=sync)

        spike_rate_hz = getattr(self.model, "latest_spike_rate_hz", None)
        if spike_rate_hz is not None:
            self.log(
                f"{split}/spike_rate_hz",
                spike_rate_hz,
                prog_bar=(split != "train"),
                on_step=on_step,
                on_epoch=True,
                sync_dist=sync,
            )

        # Free large stashed tensors so they don't pin memory across optimizer step.
        for attr in ("last_spikes", "last_v_norm", "last_logits_chunks"):
            if hasattr(self.model, attr):
                setattr(self.model, attr, None)

        return loss

    def training_step(self, batch, batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int):
        self._shared_step(batch, "test")

    def on_train_batch_start(self, batch, batch_idx):
        if self.trainer.is_global_zero:
            self._step_start_time = time.perf_counter()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self._step_start_time is not None and self.trainer.is_global_zero:
            elapsed = time.perf_counter() - self._step_start_time
            x, _ = batch
            imgs = x.shape[0] if x.ndim == 4 else x.shape[1]
            self.log("train/imgs_per_sec", imgs / elapsed, on_step=True, sync_dist=False)

    def configure_optimizers(self):
        lr = float(getattr(self.optimizer_config, "lr", 1e-3))
        b1 = float(getattr(self.optimizer_config, "beta1", 0.9))
        b2 = float(getattr(self.optimizer_config, "beta2", 0.999))
        weight_decay = float(getattr(self.optimizer_config, "weight_decay", 0.0))

        optim_groups = make_optimizer_groups(self.model, weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(b1, b2))

        sched_name = str(getattr(self.optimizer_config, "scheduler", "cosine")).lower()
        if sched_name != "cosine":
            rank_zero_info(f"Unknown scheduler={sched_name}; using optimizer only.")
            return optimizer

        min_lr_ratio = float(getattr(self.optimizer_config, "min_lr_ratio", 0.1))
        max_epochs = int(getattr(self.train_config, "trainer", None) and
                         getattr(self.train_config.trainer, "max_epochs", 0) or 0)
        warmup_epochs = int(getattr(self.optimizer_config, "warmup_epochs", 0) or 0)

        if max_epochs <= 0:
            rank_zero_info("scheduler=cosine requires trainer.max_epochs; using optimizer only.")
            return optimizer

        def lr_lambda(epoch: int):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))
            progress = (epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def build_model(model_config: Any, optimizer_config: Any, train_config: Any, data_config: Any) -> LitVisionSNN:
    name = str(model_config.name).lower()
    model_cls = get_model_cls(name)
    model = model_cls(model_config)
    init_weights(model)

    total_params = sum(p.numel() for p in model.parameters())
    rank_zero_info(f"Model: {model.__class__.__name__}  params={total_params:,}")

    return LitVisionSNN(
        model=model,
        optimizer_config=optimizer_config,
        train_config=train_config,
        data_config=data_config,
        model_config=model_config,
    )
