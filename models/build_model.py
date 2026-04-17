from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data.mixup import Mixup
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
from . import mem_gated_attention  # noqa: F401


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
    def __init__(self, model: nn.Module, optimizer_config: Any, train_config: Any, data_config: Any):
        super().__init__()
        self.model = model
        # self.model = torch.compile(model, mode="reduce-overhead")
        self.optimizer_config = optimizer_config
        self.train_config = train_config
        self.data_config = data_config
        self.save_hyperparameters(ignore=["model"])

        self.T = int(getattr(model, "T", getattr(optimizer_config, "T", 4)))
        self._is_event_input = bool(getattr(data_config, "is_event", False))
        self._step_start_time = None

        mixup_alpha = float(getattr(data_config, "mixup_alpha", 0.0))
        cutmix_alpha = float(getattr(data_config, "cutmix_alpha", 0.0))
        label_smoothing = float(getattr(data_config, "label_smoothing", 0.0))
        num_classes = int(getattr(data_config, "num_classes", 10))

        if mixup_alpha > 0 or cutmix_alpha > 0:
            self.mixup_fn = Mixup(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                label_smoothing=label_smoothing,
                num_classes=num_classes,
            )
        else:
            self.mixup_fn = None
        self.label_smoothing = label_smoothing
        self.mixup_off_epoch = getattr(data_config, 'mixup_off_epoch', 0)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_event_input:
            # already [T, B, C, H, W]
            return x
        # static image [B, C, H, W] -> [T, B, C, H, W]
        return expand_static_to_temporal(x, self.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self.model)
        if self.model.__class__.__name__ != "SpikeDrivenTransformerV3":
            x = self._prepare_input(x)
        logits = self.model(x)
        # functional.reset_net(self.model)
        return logits

    def _shared_step(self, batch, split: str):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        num_classes = logits.size(1)
        topk = (1, 5) if num_classes >= 5 else (1,)
        accs = accuracy_at_k(logits, y, topk=topk)

        sync = split != "train"
        self.log(f"{split}/loss", loss, prog_bar=True, on_step=(split == "train"),
                 on_epoch=True, sync_dist=sync)
        self.log(f"{split}/top1", accs["top1"], prog_bar=True, on_step=False,
                 on_epoch=True, sync_dist=sync)
        if num_classes >= 5:
            self.log(f"{split}/top5", accs["top5"], on_step=False, on_epoch=True, sync_dist=sync)
        return loss

    # def training_step(self, batch, batch_idx: int):
    #     return self._shared_step(batch, "train")
    def training_step(self, batch, batch_idx: int):
        # functional.reset_net(self.model)
        x, y = batch
        if self.mixup_fn is not None:
            x, y = self.mixup_fn(x, y)  # y 变成 soft label
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing if self.mixup_fn is None else 0.0)
        # mixup_fn 内部已处理 label_smoothing，所以 mixup 开启时不重复
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        # mixup 后 y 是 soft label，不能算 top-k accuracy，跳过或用 argmax
        if y.ndim == 1:
            accs = accuracy_at_k(logits, y, topk=(1,))
            self.log("train/top1", accs["top1"], prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int):
        self._shared_step(batch, "test")

    def on_train_batch_start(self, batch, batch_idx):
        if self.trainer.is_global_zero:
            self._step_start_time = time.perf_counter()

        if self.mixup_off_epoch and self.mixup_fn is not None:
            self.mixup_fn.mixup_enabled = self.current_epoch < self.mixup_off_epoch
            
        for m in self.modules():
            if hasattr(m, 'reset'):
                m.reset()


    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self._step_start_time is not None and self.trainer.is_global_zero:
            elapsed = time.perf_counter() - self._step_start_time
            x, _ = batch
            imgs = x.shape[0] if x.ndim == 4 else x.shape[1]
            self.log("train/imgs_per_sec", imgs / elapsed, on_step=True, sync_dist=False)
        
        # Log gate firing rates
        if hasattr(self.model, "blocks"):
            for i, blk in enumerate(self.model.blocks):
                if hasattr(blk.attn, "decay_gate"):
                    dr = blk.attn.decay_gate.last_firing_rate
                    if dr is not None:
                        self.log(f"gate/block{i}_decay_fr", dr, on_step=True, on_epoch=False)
                        
        for m in self.modules():
            if hasattr(m, 'reset'):
                m.reset()


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
    )
