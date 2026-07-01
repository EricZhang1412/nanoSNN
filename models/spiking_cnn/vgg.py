from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional

from ..common.layers import ConvBNLIF
from ..common.spike_ops import build_neuron, temporal_mean
from ..common.registry import register_model


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "none", ""}:
        return False
    return default


def _parse_stage_ids(value, n_stages: int, *, default_all: bool = False) -> set[int]:
    """Parse 1-indexed stage ids from yaml-friendly values."""
    if value is None:
        return set(range(1, n_stages + 1)) if default_all else set()
    if isinstance(value, bool):
        return set(range(1, n_stages + 1)) if value else set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {int(v) for v in value}

    text = str(value).strip().lower().replace("_", "").replace("-", "")
    if text in {"", "none", "false", "0", "off"}:
        return set()
    if text in {"all", "allstages", "true", "1", "on"}:
        return set(range(1, n_stages + 1))

    stage_ids: set[int] = set()
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        token = token.replace("stage", "").replace("only", "")
        if token:
            stage_ids.add(int(token))
    return stage_ids


class HubSpokeModulator(nn.Module):
    """Stage-level hub that gates membrane drive before the main LIF."""

    def __init__(self, channels: int, model_config):
        super().__init__()
        ratio = float(getattr(model_config, "hub_ratio", 1.0 / 16.0))
        hub_channels = max(1, int(round(channels * ratio)))
        self.pool = layer.AdaptiveAvgPool2d((1, 1), step_mode="m")
        self.to_hub = layer.Conv2d(channels, hub_channels, 1, bias=False, step_mode="m")
        self.hub_bn = layer.BatchNorm2d(hub_channels, step_mode="m")
        self.hub_lif = build_neuron(model_config)
        self.from_hub = layer.Conv2d(hub_channels, channels, 1, bias=False, step_mode="m")
        self.feedback_bn = layer.BatchNorm2d(channels, step_mode="m")

        beta = float(getattr(model_config, "hub_beta", 0.5))
        if _as_bool(getattr(model_config, "hub_learnable_beta", False)):
            self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        else:
            self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32), persistent=False)
        self.latest_stats: dict[str, torch.Tensor | int] = {}

    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        hub = self.pool(drive)
        hub = self.hub_bn(self.to_hub(hub))
        hub_spikes = self.hub_lif(hub)
        feedback = self.feedback_bn(self.from_hub(hub_spikes))

        beta = self.beta.to(device=drive.device, dtype=drive.dtype)
        out = drive + beta * feedback

        with torch.no_grad():
            hub_float = hub_spikes.detach().float()
            self.latest_stats = {
                "hub_spike_rate": hub_float.mean(),
                "hub_total_spikes": hub_float.sum(),
                "hub_numel": hub_float.numel(),
            }
        return out


class MembraneShortcutConvBNLIF(nn.Module):
    """Conv-BN-LIF with optional pre-LIF membrane shortcut and hub modulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_config,
        *,
        membrane_shortcut: bool,
        hub_and_spoke: bool,
    ):
        super().__init__()
        self.conv = layer.Conv2d(
            in_channels, out_channels, 3, stride=1, padding=1, bias=False, step_mode="m"
        )
        self.bn = layer.BatchNorm2d(out_channels, step_mode="m")
        self.lif = build_neuron(model_config)
        self.use_shortcut = bool(membrane_shortcut)

        if self.use_shortcut:
            if in_channels == out_channels:
                self.shortcut_proj = nn.Identity()
            else:
                self.shortcut_proj = nn.Sequential(
                    layer.Conv2d(in_channels, out_channels, 1, bias=False, step_mode="m"),
                    layer.BatchNorm2d(out_channels, step_mode="m"),
                )
            alpha = float(getattr(model_config, "shortcut_alpha", 0.5))
            if _as_bool(getattr(model_config, "shortcut_learnable_alpha", False)):
                self.shortcut_alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
            else:
                self.register_buffer(
                    "shortcut_alpha", torch.tensor(alpha, dtype=torch.float32), persistent=False
                )
        else:
            self.shortcut_proj = None
            self.register_buffer(
                "shortcut_alpha", torch.tensor(0.0, dtype=torch.float32), persistent=False
            )

        self.hub = HubSpokeModulator(out_channels, model_config) if hub_and_spoke else None
        self.latest_stats: dict[str, torch.Tensor | int] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        drive = self.bn(self.conv(x))
        if self.use_shortcut:
            alpha = self.shortcut_alpha.to(device=drive.device, dtype=drive.dtype)
            drive = drive + alpha * self.shortcut_proj(x)
        if self.hub is not None:
            drive = self.hub(drive)

        spikes = self.lif(drive)
        with torch.no_grad():
            spikes_float = spikes.detach().float()
            self.latest_stats = {
                "spike_rate": spikes_float.mean(),
                "total_spikes": spikes_float.sum(),
                "numel": spikes_float.numel(),
            }
            if self.hub is not None:
                self.latest_stats.update(self.hub.latest_stats)
        return spikes


class SpikingVGGBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_convs: int,
        model_config,
        *,
        membrane_shortcut: bool = False,
        hub_and_spoke: bool = False,
    ):
        super().__init__()
        convs = []
        for i in range(num_convs):
            conv_in = in_ch if i == 0 else out_ch
            use_hub = hub_and_spoke and i == num_convs - 1
            if membrane_shortcut or use_hub:
                convs.append(
                    MembraneShortcutConvBNLIF(
                        conv_in,
                        out_ch,
                        model_config,
                        membrane_shortcut=membrane_shortcut,
                        hub_and_spoke=use_hub,
                    )
                )
            else:
                convs.append(ConvBNLIF(conv_in, out_ch, 3, 1, 1, model_config))
        self.convs = nn.ModuleList(convs)
        self.pool = layer.MaxPool2d(2, 2, step_mode="m")
        self.latest_layer_stats: dict[str, dict[str, torch.Tensor | int]] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.latest_layer_stats = {}
        for i, conv in enumerate(self.convs, start=1):
            x = conv(x)
            stats = getattr(conv, "latest_stats", None)
            if not stats:
                with torch.no_grad():
                    spikes_float = x.detach().float()
                    stats = {
                        "spike_rate": spikes_float.mean(),
                        "total_spikes": spikes_float.sum(),
                        "numel": spikes_float.numel(),
                    }
            self.latest_layer_stats[f"conv{i}"] = stats
        return self.pool(x)


_VGG_CONFIGS = {
    "vgg11": [1, 1, 2, 2, 2],
    "vgg13": [2, 2, 2, 2, 2],
    "vgg16": [2, 2, 3, 3, 3],
    "vgg19": [2, 2, 4, 4, 4],
}

_VGG_CHANNELS = [64, 128, 256, 512, 512]


@register_model("spiking_vgg")
class SpikingVGG(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        variant = str(getattr(model_config, "variant", getattr(model_config, "arch", "vgg11"))).lower()
        num_classes = int(getattr(model_config, "num_classes", 10))
        self.T = int(getattr(model_config, "T", 4))
        dropout = float(getattr(model_config, "dropout", 0.0))
        self.spike_dt_ms = float(getattr(model_config, "spike_dt_ms", 1.0))
        self.membrane_shortcut = _as_bool(getattr(model_config, "membrane_shortcut", False))
        self.hub_and_spoke = _as_bool(getattr(model_config, "hub_and_spoke", False))

        if variant not in _VGG_CONFIGS:
            raise ValueError(f"Unsupported VGG variant: {variant}")
        num_convs_per_stage = _VGG_CONFIGS[variant]
        channels = _VGG_CHANNELS
        n_stages = len(num_convs_per_stage)
        shortcut_stages = _parse_stage_ids(
            getattr(model_config, "membrane_shortcut_stages", None),
            n_stages,
            default_all=self.membrane_shortcut,
        )
        hub_stages = _parse_stage_ids(
            getattr(model_config, "hub_spoke_stages", None),
            n_stages,
            default_all=False,
        )
        if self.hub_and_spoke and not hub_stages:
            hub_stages = {3}

        stages = []
        in_ch = int(getattr(model_config, "in_channels", 3))
        for stage_id, (num_convs, out_ch) in enumerate(zip(num_convs_per_stage, channels), start=1):
            stages.append(
                SpikingVGGBlock(
                    in_ch,
                    out_ch,
                    num_convs,
                    model_config,
                    membrane_shortcut=stage_id in shortcut_stages,
                    hub_and_spoke=self.hub_and_spoke and stage_id in hub_stages,
                )
            )
            in_ch = out_ch
        self.features = nn.ModuleList(stages)

        self.pool = layer.AdaptiveAvgPool2d((1, 1), step_mode="m")

        hidden_dim = int(getattr(model_config, "classifier_hidden", 4096))
        if hidden_dim <= 0:
            raise ValueError(f"classifier_hidden must be positive, got {hidden_dim}")
        classifier = [layer.Linear(512, hidden_dim, step_mode="m"),
                      build_neuron(model_config)]
        if dropout > 0:
            classifier.append(layer.Dropout(dropout, step_mode="m"))
        classifier += [layer.Linear(hidden_dim, hidden_dim, step_mode="m"),
                       build_neuron(model_config)]
        if dropout > 0:
            classifier.append(layer.Dropout(dropout, step_mode="m"))
        classifier.append(layer.Linear(hidden_dim, num_classes, step_mode="m"))
        self.classifier = nn.Sequential(*classifier)
        self._classifier_spike_module_ids = {
            i for i, module in enumerate(self.classifier)
            if module.__class__.__name__.lower().endswith("node")
        }
        self.latest_spike_rate_per_timestep = None
        self.latest_spike_rate_hz = None
        self.latest_hub_spike_rate_hz = None
        self.latest_spikes_per_sample = None
        self.latest_synops_proxy = None
        self.latest_layerwise_spike_rate: dict[str, torch.Tensor] = {}

    @staticmethod
    def _stat_to_device(value, ref: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=ref.device)
        return ref.new_tensor(float(value))

    def _finalize_spike_stats(
        self,
        ref: torch.Tensor,
        batch_size: int,
        extra_layer_stats: dict[str, dict[str, torch.Tensor | int]] | None = None,
    ) -> None:
        layerwise: dict[str, torch.Tensor] = {}
        total_spikes = ref.new_zeros(())
        total_numel = 0
        total_hub_spikes = ref.new_zeros(())
        total_hub_numel = 0

        for stage_id, stage in enumerate(self.features, start=1):
            for layer_name, stats in stage.latest_layer_stats.items():
                prefix = f"stage{stage_id}.{layer_name}"
                rate = stats.get("spike_rate")
                total = stats.get("total_spikes")
                numel = int(stats.get("numel", 0))
                if rate is not None:
                    layerwise[prefix] = self._stat_to_device(rate, ref).detach()
                if total is not None and numel > 0:
                    total_spikes = total_spikes + self._stat_to_device(total, ref)
                    total_numel += numel
                hub_rate = stats.get("hub_spike_rate")
                hub_total = stats.get("hub_total_spikes")
                hub_numel = int(stats.get("hub_numel", 0))
                if hub_rate is not None:
                    layerwise[f"{prefix}.hub"] = self._stat_to_device(hub_rate, ref).detach()
                if hub_total is not None and hub_numel > 0:
                    total_hub_spikes = total_hub_spikes + self._stat_to_device(hub_total, ref)
                    total_hub_numel += hub_numel

        if extra_layer_stats:
            for prefix, stats in extra_layer_stats.items():
                rate = stats.get("spike_rate")
                total = stats.get("total_spikes")
                numel = int(stats.get("numel", 0))
                if rate is not None:
                    layerwise[prefix] = self._stat_to_device(rate, ref).detach()
                if total is not None and numel > 0:
                    total_spikes = total_spikes + self._stat_to_device(total, ref)
                    total_numel += numel

        if total_numel > 0:
            rate = total_spikes / float(total_numel)
            self.latest_spike_rate_per_timestep = rate.detach()
            self.latest_spike_rate_hz = (rate * (1000.0 / max(self.spike_dt_ms, 1e-6))).detach()
            self.latest_spikes_per_sample = (total_spikes / float(max(batch_size, 1))).detach()
            self.latest_synops_proxy = self.latest_spikes_per_sample
        else:
            self.latest_spike_rate_per_timestep = None
            self.latest_spike_rate_hz = None
            self.latest_spikes_per_sample = None
            self.latest_synops_proxy = None

        if total_hub_numel > 0:
            hub_rate = total_hub_spikes / float(total_hub_numel)
            self.latest_hub_spike_rate_hz = (
                hub_rate * (1000.0 / max(self.spike_dt_ms, 1e-6))
            ).detach()
        else:
            self.latest_hub_spike_rate_hz = None
        self.latest_layerwise_spike_rate = layerwise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, C, H, W]
        batch_size = x.shape[1]
        for stage in self.features:
            x = stage(x)
        x = self.pool(x)
        x = x.flatten(2)
        classifier_spike_stats = {}
        for i, module in enumerate(self.classifier):
            x = module(x)
            if i in self._classifier_spike_module_ids:
                with torch.no_grad():
                    spikes_float = x.detach().float()
                    classifier_spike_stats[f"classifier.lif{i}"] = {
                        "spike_rate": spikes_float.mean(),
                        "total_spikes": spikes_float.sum(),
                        "numel": spikes_float.numel(),
                    }
        self._finalize_spike_stats(x, batch_size, classifier_spike_stats)
        return temporal_mean(x)
