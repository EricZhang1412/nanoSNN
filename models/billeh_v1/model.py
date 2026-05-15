from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from ..common.registry import register_model
from .billeh_column import BillehColumnTorch
from .dvs_encoder import DVSDirectEncoder
from .lgn_torch import TorchLGN
from .load_sparse_torch import load_billeh_torch
from .pool_readout import LocalizedPoolReadout


@register_model("billeh_v1")
class BillehV1Classifier(nn.Module):
    """
    V1 GLIF backbone + localized L5 pool readout for Chen-Maass-style classification.
    Input supports:
      - static expanded temporal: [T, B, C, H, W]
      - direct temporal current/spike-like: [B, T, n_input]
    """

    def __init__(self, model_config):
        super().__init__()
        self.T = int(getattr(model_config, "T", 600))
        self.auto_n_input_from_in_channels = bool(
            getattr(model_config, "auto_n_input_from_in_channels", False)
        )
        in_channels = int(getattr(model_config, "in_channels", 1))
        if self.auto_n_input_from_in_channels:
            self.n_input = in_channels
        else:
            self.n_input = int(getattr(model_config, "n_input", 17400))
        self.n_neurons = int(getattr(model_config, "n_neurons", 1000))
        self.num_classes = int(getattr(model_config, "num_classes", 10))
        self.seed = int(getattr(model_config, "seed", 3000))
        self.full_core = bool(getattr(model_config, "full_core", False))
        self.train_v1 = bool(getattr(model_config, "train_v1", True))
        self.encoding = str(getattr(model_config, "encoding", "identity")).lower()
        self.encoding_gain = float(getattr(model_config, "encoding_gain", 1.0))
        # If true, keep temporal length from input instead of forcing model_config.T.
        self.use_input_t = bool(getattr(model_config, "use_input_t", False))
        self.use_lgn = bool(getattr(model_config, "use_lgn", False))
        self.use_dvs_direct = bool(getattr(model_config, "use_dvs_direct", False))
        if self.use_dvs_direct and self.use_lgn:
            raise ValueError("use_dvs_direct and use_lgn are mutually exclusive")
        self.lgn_input_height = int(getattr(model_config, "lgn_input_height", 0) or 0)
        self.lgn_input_width = int(getattr(model_config, "lgn_input_width", 0) or 0)
        # DVS direct encoder knobs (only used when use_dvs_direct=True).
        self.dvs_image_size = int(getattr(model_config, "dvs_image_size", 0) or 0)
        self.dvs_in_channels = int(getattr(model_config, "dvs_in_channels", 1))
        self.dvs_encoder_hidden = int(getattr(model_config, "dvs_encoder_hidden", 16))
        self.dvs_encoder_output_scale = float(getattr(model_config, "dvs_encoder_output_scale", 50.0))

        # Strict-reproduction trial timing.
        self.pre_delay = int(getattr(model_config, "pre_delay", 0))
        self.post_delay = int(getattr(model_config, "post_delay", 0))
        self.response_window_len = int(getattr(model_config, "response_window_len", 0))
        self.down_sample = int(getattr(model_config, "down_sample", 0))
        self.dampening_factor = float(getattr(model_config, "dampening_factor", 0.3))
        self.gauss_std = float(getattr(model_config, "gauss_std", 0.5))
        self.chunk_size = int(getattr(model_config, "chunk_size", 0))
        self.neurons_per_output = int(getattr(model_config, "neurons_per_output", 30))
        self.localized_readout = bool(getattr(model_config, "localized_readout", True))
        # LGN cells output firing rates in Hz; V1 input weights are calibrated for
        # spike-per-timestep currents. With dt=1 ms, dividing by 1000 turns Hz into
        # spikes/ms ("Poisson-rate-equivalent" continuous current).
        self.lgn_input_scale = float(getattr(model_config, "lgn_input_scale", 1.0))

        self.lgn = None
        if self.use_lgn:
            lgn_data_path = str(getattr(model_config, "lgn_data_path", "")).strip()
            lgn_temporal_kernels_path = str(
                getattr(model_config, "lgn_temporal_kernels_path", "")
            ).strip() or None
            self.lgn = TorchLGN(
                lgn_data_path=lgn_data_path,
                temporal_kernels_path=lgn_temporal_kernels_path,
            )
            auto_n_input_from_lgn = bool(
                getattr(model_config, "auto_n_input_from_lgn", True)
            )
            if auto_n_input_from_lgn:
                self.n_input = int(self.lgn.n_cells)

        data_dir = str(getattr(model_config, "billeh_data_dir", "")).strip()
        if not data_dir:
            raise ValueError("model_config.billeh_data_dir is required for billeh_v1")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"billeh_data_dir not found: {data_dir}")

        try:
            loaded = load_billeh_torch(
                n_input=self.n_input,
                n_neurons=self.n_neurons,
                core_only=not self.full_core,
                data_dir=data_dir,
                seed=self.seed,
                connected_selection=self.full_core,
                localized_readout=self.localized_readout,
                neurons_per_output=self.neurons_per_output,
                device="cpu",
            )
        except ValueError as exc:
            # Some reduced cores cannot populate all localized pools.
            msg = str(exc)
            if self.localized_readout and "No readout population" in msg:
                print(
                    "Warning: localized_readout failed due to small neuronal volume; "
                    "falling back to global random readout."
                )
                loaded = load_billeh_torch(
                    n_input=self.n_input,
                    n_neurons=self.n_neurons,
                    core_only=not self.full_core,
                    data_dir=data_dir,
                    seed=self.seed,
                    connected_selection=self.full_core,
                    localized_readout=False,
                    neurons_per_output=self.neurons_per_output,
                    device="cpu",
                )
            else:
                raise

        bkg_weights = loaded["bkg_weights"]
        if isinstance(bkg_weights, torch.Tensor):
            bkg_weights = bkg_weights.detach().cpu().numpy()

        self.v1 = BillehColumnTorch(
            loaded["network"],
            loaded["input_population"],
            bkg_weights,
            gauss_std=self.gauss_std,
            dampening_factor=self.dampening_factor,
            chunk_size=self.chunk_size,
            device="cpu",
        )

        self.dvs_encoder = None
        if self.use_dvs_direct:
            if self.dvs_image_size <= 0:
                raise ValueError("use_dvs_direct=True requires model_config.dvs_image_size > 0")
            n_current = int(self.v1.n_receptors) * int(self.v1.n_neurons)
            self.dvs_encoder = DVSDirectEncoder(
                in_channels=self.dvs_in_channels,
                image_size=self.dvs_image_size,
                n_out=n_current,
                T_model=self.T,
                hidden=self.dvs_encoder_hidden,
                output_scale=self.dvs_encoder_output_scale,
            )
            # input_population sparse weights are bypassed in this path; freeze
            # them so they neither train nor receive (zero) gradient updates.
            self.v1.input_weight_values.requires_grad_(False)

        # Pool ids: upstream uses pools 5..14 for the 10-class image task
        # (pools 0..4 reserved for garrett 2-class). For other class counts
        # (e.g. 11-class DVS128Gesture) we use the first num_classes pools so
        # all classes get disjoint readouts. Cap at 15 (the available pool count).
        network = loaded["network"]
        if self.num_classes > 15:
            raise ValueError(
                f"num_classes={self.num_classes} exceeds the 15 localized readout pools"
            )
        if self.num_classes == 10 and "localized_readout_neuron_ids_5" in network:
            pool_offset = 5
        else:
            pool_offset = 0
        pool_ids = []
        for i in range(self.num_classes):
            key = f"localized_readout_neuron_ids_{i + pool_offset}"
            if key not in network:
                raise KeyError(f"missing localized readout pool {i + pool_offset}")
            pool_ids.append(np.asarray(network[key]).reshape(-1))

        self.readout = LocalizedPoolReadout(
            pool_ids=pool_ids,
            dampening_factor=self.dampening_factor,
            down_sample=self.down_sample if self.down_sample > 0 else self.T,
            n_classes=self.num_classes,
        )

        # Response-window chunk mask.
        if self.down_sample > 0 and self.T % self.down_sample == 0:
            n_chunks = self.T // self.down_sample
            im_slice = self.T - self.pre_delay - self.post_delay
            if im_slice <= 0:
                # No timing structure → use all chunks for inference.
                mask = torch.ones(n_chunks, dtype=torch.float32)
            else:
                resp_start_bin = self.pre_delay // self.down_sample
                resp_end_step = self.pre_delay + im_slice + self.response_window_len
                resp_end_bin = (resp_end_step + self.down_sample - 1) // self.down_sample
                resp_end_bin = min(resp_end_bin, n_chunks)
                mask = torch.zeros(n_chunks, dtype=torch.float32)
                mask[resp_start_bin:resp_end_bin] = 1.0
                if mask.sum() == 0:
                    mask[:] = 1.0
        else:
            mask = torch.ones(1, dtype=torch.float32)
        self.register_buffer("_resp_chunk_mask", mask)

        # Stash slots filled by forward() for the loss step.
        self.last_spikes: torch.Tensor | None = None
        self.last_v_norm: torch.Tensor | None = None
        self.last_logits_chunks: torch.Tensor | None = None
        self.latest_spike_rate_hz: torch.Tensor | None = None

    def _to_b_t_n(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_lgn:
            return self._to_b_t_n_via_lgn(x)

        # [T, B, C, H, W] (static expanded temporal) -> [B, T, n_input]
        if x.ndim == 5:
            if (not self.use_input_t) and x.shape[0] != self.T:
                t_keep = min(self.T, x.shape[0])
                x = x[:t_keep]
            t, b = x.shape[0], x.shape[1]
            x = x.permute(1, 0, 2, 3, 4).reshape(b, t, -1)
        elif x.ndim == 3:
            b, t = x.shape[:2]
            x = x.reshape(b, t, -1)
            if (not self.use_input_t) and t != self.T:
                t_keep = min(self.T, t)
                x = x[:, :t_keep]
        else:
            raise ValueError(f"Unsupported input shape for billeh_v1: {tuple(x.shape)}")

        if x.shape[-1] != self.n_input:
            raise ValueError(
                f"billeh_v1 expects input last dim == n_input ({self.n_input}), "
                f"got {x.shape[-1]}. Align the data feature dim with model_config.in_channels "
                f"(or n_input)."
            )

        if not self.use_input_t:
            if x.shape[1] < self.T:
                repeat_n = self.T - x.shape[1]
                x = torch.cat([x, x[:, -1:, :].repeat(1, repeat_n, 1)], dim=1)
            elif x.shape[1] > self.T:
                x = x[:, : self.T]
        return x

    def _to_b_t_n_via_lgn(self, x: torch.Tensor) -> torch.Tensor:
        if self.lgn is None:
            raise RuntimeError("use_lgn=True but LGN module is not initialized")

        if x.ndim == 5:
            # [T, B, C, H, W] -> [B, T, H, W]
            x = x.permute(1, 0, 2, 3, 4)
            if x.shape[2] > 1:
                movie = x.mean(dim=2)
            else:
                movie = x[:, :, 0]
            # K-replay: short event-frame inputs (e.g. DVS T_dvs=16) get each
            # frame held for self.T // T_in LGN steps. Lets datasets follow
            # spikingjelly conventions while LGN/V1 retain ms-resolution.
            t_in = movie.shape[1]
            t_lgn = int(self.T)
            if t_lgn != t_in:
                if t_lgn <= 0 or t_lgn % t_in != 0:
                    raise ValueError(
                        f"K-replay requires model.T ({t_lgn}) to be a positive multiple of input T ({t_in}); "
                        f"adjust model_config.T or the dataset's frames_number so model.T % T_input == 0"
                    )
                k = t_lgn // t_in
                movie = movie.repeat_interleave(k, dim=1)
        elif x.ndim == 4:
            # [B, C, H, W] -> [B, T=1, H, W]
            movie = x.mean(dim=1, keepdim=False).unsqueeze(1)
        elif x.ndim == 3:
            h = int(getattr(self, "lgn_input_height", 0) or 0)
            w = int(getattr(self, "lgn_input_width", 0) or 0)
            if h <= 0 or w <= 0:
                raise ValueError(
                    "use_lgn with 3D input requires model_config.lgn_input_height and lgn_input_width"
                )
            b, t_seq, c_last = x.shape
            if c_last == h * w:
                # Flat-spatial: last dim already encodes the (H, W) frame.
                movie = x.reshape(b, t_seq, h, w)
            elif t_seq == h * w:
                # Sequential pixel-scan layout (e.g. seq-CIFAR/MNIST):
                # the sequence dim is the row-major spatial flatten and the
                # last dim is C. Collapse to grayscale, fold time into (H, W),
                # then replay as a static movie of length self.T so the LGN
                # has a usable temporal axis.
                gray = x.mean(dim=-1) if c_last > 1 else x[..., 0]
                frame = gray.reshape(b, h, w).unsqueeze(1)
                t_movie = int(self.T) if self.T > 0 else 1
                movie = frame.expand(b, t_movie, h, w).contiguous()
            else:
                raise ValueError(
                    f"Unsupported 3D input shape for LGN path: {tuple(x.shape)}; "
                    f"expected last dim={h*w} or seq dim={h*w} (got {c_last}, {t_seq})"
                )
        else:
            raise ValueError(f"Unsupported input shape for LGN path: {tuple(x.shape)}")

        x_btn = self.lgn(movie)
        # Strict reproduction: dataset is responsible for assembling the full
        # [pre + im_slice + post] movie; do NOT pad or truncate here.
        if x_btn.shape[-1] != self.n_input:
            raise ValueError(
                f"LGN output last dim ({x_btn.shape[-1]}) != n_input ({self.n_input}). "
                f"Set model_config.n_input (or auto_n_input_from_lgn=true) to match the LGN cell count."
            )
        return x_btn

    def _encode(self, x_btn: torch.Tensor) -> torch.Tensor:
        # On the LGN path, x_btn carries firing rates in Hz; convert to
        # spikes/timestep via lgn_input_scale (default 1.0 = pass-through;
        # set to 1e-3 for strict reproduction with dt=1 ms).
        if self.use_lgn:
            return x_btn.float() * self.lgn_input_scale
        if self.encoding == "identity":
            return x_btn.float()

        x_btn = x_btn.float()
        norm_dim = 1 if x_btn.shape[-1] == 1 else -1
        x_btn = x_btn - x_btn.amin(dim=norm_dim, keepdim=True)
        denom = x_btn.amax(dim=norm_dim, keepdim=True).clamp_min(1e-6)
        x_btn = x_btn / denom

        if self.encoding == "repeat":
            return x_btn
        if self.encoding == "poisson":
            p = (x_btn * self.encoding_gain).clamp(0.0, 1.0)
            return (torch.rand_like(p) < p).float()
        raise ValueError(f"Unknown encoding: {self.encoding}")

    def _collapse_chunks_for_inference(self, logits_chunks: torch.Tensor) -> torch.Tensor:
        # Average response-window chunks only.
        if self._resp_chunk_mask is None or self._resp_chunk_mask.numel() != logits_chunks.shape[1]:
            return logits_chunks.mean(dim=1)
        m = self._resp_chunk_mask.to(logits_chunks.device).view(1, -1, 1)
        return (logits_chunks * m).sum(dim=1) / m.sum().clamp_min(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_dvs_direct:
            current = self.dvs_encoder(x)            # [B, T, n_receptors*n_neurons]
            self.v1.device = current.device
            spikes, v_mV, _ = self.v1(current, input_is_current=True)
        else:
            x_btn = self._to_b_t_n(x)
            x_btn = self._encode(x_btn)
            self.v1.device = x_btn.device
            spikes, v_mV, _ = self.v1(x_btn)

        # Convert v to normalized voltage for downstream regularization.
        vs_scale = self.v1.voltage_scale[None, None, :]
        vo_scale = self.v1.voltage_offset[None, None, :]
        v_norm = (v_mV - vo_scale) / vs_scale.clamp_min(1e-6)

        logits_chunks = self.readout(spikes)

        # Stash for the Lightning loss step.
        self.last_spikes = spikes
        self.last_v_norm = v_norm
        self.last_logits_chunks = logits_chunks
        # Population mean rate Hz for logging (detached).
        self.latest_spike_rate_hz = spikes.detach().float().mean() * 1000.0

        return self._collapse_chunks_for_inference(logits_chunks)
