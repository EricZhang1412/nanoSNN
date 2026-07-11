#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.ascend import NPUAccelerator, make_npu_ddp_strategy, npu_device_count  # noqa: E402


class TinyLitModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
        )

    def training_step(self, batch, batch_idx):  # noqa: ANN001
        x, y = batch
        logits = self.net(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def _make_loader(batch_size: int) -> DataLoader:
    # Fixed random data keeps the smoke independent of external datasets.
    gen = torch.Generator().manual_seed(20240616)
    x = torch.randn(16, 8, generator=gen)
    y = torch.randint(0, 4, (16,), generator=gen)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False, num_workers=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightning HCCL/DDP smoke test for nanoSNN's custom Ascend strategy")
    parser.add_argument("--devices", type=int, default=None, help="Visible NPU count to hand to Lightning. Defaults to WORLD_SIZE or 1.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--require_multi", action="store_true", help="Fail unless at least two NPUs are visible.")
    args = parser.parse_args()

    try:
        import torch_npu  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on host runtime
        raise RuntimeError("torch_npu is required for Lightning HCCL/DDP smoke") from exc

    count = npu_device_count()
    if count <= 0:
        raise RuntimeError("No available Ascend NPU devices found")
    if args.require_multi and count < 2:
        raise RuntimeError(f"--require_multi needs >=2 visible NPUs, got {count}")

    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)
    devices = int(args.devices or world_size or 1)
    if devices > count:
        raise RuntimeError(f"Requested devices={devices}, but only {count} visible NPUs are available")

    accelerator = NPUAccelerator()
    device_ids = type(accelerator).parse_devices(devices)
    parallel_devices = type(accelerator).get_parallel_devices(device_ids)
    strategy = make_npu_ddp_strategy(parallel_devices, accelerator)

    if local_rank == 0:
        print(
            f"[lightning-hccl] torch={torch.__version__} npu_count={count} "
            f"devices={device_ids} world_size={world_size} visible={os.getenv('ASCEND_RT_VISIBLE_DEVICES', 'all')}",
            flush=True,
        )

    trainer = L.Trainer(
        accelerator="auto",
        devices=device_ids,
        strategy=strategy,
        max_epochs=1,
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(TinyLitModule(), train_dataloaders=_make_loader(args.batch_size))
    torch.npu.synchronize()

    if trainer.global_rank == 0:
        print("[PASS] Lightning HCCL/DDP smoke completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
