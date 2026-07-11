from __future__ import annotations

import os
from typing import Any

import torch


def import_torch_npu(required: bool = False) -> bool:
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:
        if required:
            raise RuntimeError(
                "torch-npu is required for Ascend NPU training. Install the "
                "CANN 9.0.0 environment and run `uv sync --python python3.11`."
            ) from exc
        return False
    return True


def is_npu_available() -> bool:
    return (
        import_torch_npu(required=False)
        and hasattr(torch, "npu")
        and bool(torch.npu.is_available())
    )


def npu_device_count() -> int:
    if not is_npu_available():
        return 0
    return int(torch.npu.device_count())


def _parse_device_list(devices: Any, count: int) -> list[int]:
    if devices in (None, "auto", -1, "-1"):
        return list(range(count))

    if isinstance(devices, int):
        if devices <= 0:
            raise ValueError(f"devices must be a positive integer, got {devices}")
        if devices > count:
            raise ValueError(f"requested {devices} NPU devices, but only {count} are available")
        return list(range(devices))

    if isinstance(devices, str):
        devices = devices.strip()
        if not devices:
            raise ValueError("devices cannot be an empty string")
        if devices.isdigit():
            return _parse_device_list(int(devices), count)
        parsed = [int(part.strip()) for part in devices.split(",") if part.strip()]
    else:
        parsed = [int(device) for device in devices]

    if not parsed:
        raise ValueError("no NPU devices selected")
    invalid = [idx for idx in parsed if idx < 0 or idx >= count]
    if invalid:
        raise ValueError(f"NPU device ids out of range: {invalid}; available ids: 0..{count - 1}")
    return parsed


class NPUAccelerator:
    """Lightning Accelerator implementation for torch-npu devices."""

    def __new__(cls):
        from lightning.pytorch.accelerators.accelerator import Accelerator

        class _NPUAccelerator(Accelerator):
            def setup_device(self, device: torch.device) -> None:
                import_torch_npu(required=True)
                if device.type != "npu":
                    raise ValueError(f"Device should be NPU, got {device} instead.")
                torch.npu.set_device(device)

            def setup(self, trainer) -> None:  # noqa: ANN001
                local_rank = int(getattr(trainer, "local_rank", 0))
                visible = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "all")
                print(f"LOCAL_RANK: {local_rank} - ASCEND_RT_VISIBLE_DEVICES: [{visible}]")
                if hasattr(torch.npu, "empty_cache"):
                    torch.npu.empty_cache()

            def get_device_stats(self, device: Any) -> dict[str, Any]:
                if hasattr(torch.npu, "memory_stats"):
                    return torch.npu.memory_stats(device)
                return {}

            def teardown(self) -> None:
                if import_torch_npu(required=False) and hasattr(torch.npu, "empty_cache"):
                    torch.npu.empty_cache()

            @staticmethod
            def parse_devices(devices: Any) -> list[int]:
                count = npu_device_count()
                if count <= 0:
                    raise ValueError("No available Ascend NPU devices found.")
                return _parse_device_list(devices, count)

            @staticmethod
            def get_parallel_devices(devices: list[int]) -> list[torch.device]:
                return [torch.device("npu", idx) for idx in devices]

            @staticmethod
            def auto_device_count() -> int:
                return npu_device_count()

            @staticmethod
            def is_available() -> bool:
                return is_npu_available()

            @staticmethod
            def name() -> str:
                return "npu"

        _NPUAccelerator.__name__ = "NPUAccelerator"
        return _NPUAccelerator()


def make_npu_ddp_strategy(parallel_devices: list[torch.device], accelerator: Any):
    from lightning.fabric.utilities.distributed import _init_dist_connection
    from lightning.fabric.utilities.seed import reset_seed
    from lightning.pytorch.strategies import DDPStrategy
    from lightning.pytorch.utilities.rank_zero import rank_zero_only
    from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
    from torch.nn.parallel.distributed import DistributedDataParallel

    class NPUDDPStrategy(DDPStrategy):
        def __init__(self) -> None:
            super().__init__(
                accelerator=accelerator,
                parallel_devices=parallel_devices,
                process_group_backend="hccl",
            )

        def setup_distributed(self) -> None:
            reset_seed()
            self.set_world_ranks()
            self._process_group_backend = self._get_process_group_backend()
            assert self.cluster_environment is not None
            _init_dist_connection(
                self.cluster_environment,
                self._process_group_backend,
                timeout=self._timeout,
            )

        def _setup_model(self, model):
            return DistributedDataParallel(
                module=model,
                device_ids=self.determine_ddp_device_ids(),
                **self._ddp_kwargs,
            )

        def set_world_ranks(self) -> None:
            super().set_world_ranks()
            rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank

    return NPUDDPStrategy()


def resolve_accelerator(requested: str) -> str:
    requested = requested.lower()
    if requested not in {"auto", "npu", "cuda", "cpu"}:
        raise ValueError(f"Unsupported accelerator: {requested}")
    if requested == "auto":
        if is_npu_available():
            return "npu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "npu" and not is_npu_available():
        import_torch_npu(required=True)
        raise RuntimeError("Ascend NPU was requested but torch.npu.is_available() is False.")
    return requested
