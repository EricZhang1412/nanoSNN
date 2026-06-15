#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
import traceback

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.build_model import build_model  # noqa: E402
from scripts.smoke_ascend_npu import _dummy_input  # noqa: E402
from utils.load_config import load_config  # noqa: E402


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None and value != "" else default


def _rank_print(message: str) -> None:
    rank = _env_int("RANK", 0)
    print(f"[rank {rank}] {message}", flush=True)


def _setup_npu(require_multi: bool) -> tuple[torch.device, int, int, int]:
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on host runtime
        raise RuntimeError("torch_npu is required for Ascend HCCL/DDP smoke") from exc

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is False")

    count = int(torch.npu.device_count())
    local_rank = _env_int("LOCAL_RANK", 0)
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)

    if require_multi and count < 2:
        raise RuntimeError(f"--require_multi needs >=2 visible NPUs, got {count}")
    if local_rank >= count:
        raise RuntimeError(f"LOCAL_RANK={local_rank} but only {count} NPUs are visible")

    device = torch.device("npu", local_rank)
    torch.npu.set_device(device)
    _rank_print(
        f"torch={torch.__version__} npu_count={count} local_rank={local_rank} "
        f"world_size={world_size} device={device} visible={os.getenv('ASCEND_RT_VISIBLE_DEVICES', 'all')}"
    )
    return device, rank, local_rank, world_size


def _setup_dist(world_size: int) -> bool:
    launched = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not launched:
        _rank_print("not launched by torchrun; skipping process-group init")
        return False
    dist.init_process_group(backend="hccl", init_method="env://")
    _rank_print(f"process group initialized: backend={dist.get_backend()}")
    return True


def _run_collective(device: torch.device, rank: int, world_size: int, distributed: bool) -> None:
    x = torch.tensor([rank + 1.0], device=device)
    if distributed:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2
    torch.npu.synchronize()
    got = float(x.item())
    if abs(got - expected) > 1e-5:
        raise AssertionError(f"all_reduce mismatch: got={got}, expected={expected}")
    _rank_print(f"[PASS] HCCL all_reduce sum={got:.1f}")


def _run_tiny_ddp(device: torch.device, local_rank: int, distributed: bool) -> None:
    torch.manual_seed(1234)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 8, device=device)
    y = torch.randint(0, 4, (4,), device=device)
    opt.zero_grad(set_to_none=True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    opt.step()
    torch.npu.synchronize()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if not grad_ok:
        raise AssertionError("tiny DDP did not produce finite gradients")
    _rank_print(f"[PASS] tiny DDP step loss={loss.item():.4f}")


def _run_nanosnn_step(args, device: torch.device, local_rank: int, distributed: bool) -> None:
    mcfg = load_config(args.model_config)
    dcfg = load_config(args.data_config)
    tcfg = load_config(args.train_config)
    ocfg = load_config(args.optimizer_config)

    lit = build_model(mcfg, ocfg, tcfg, dcfg).to(device)
    lit.train()
    opt_cfg = lit.configure_optimizers()
    optimizer = opt_cfg["optimizer"] if isinstance(opt_cfg, dict) else opt_cfg
    model = DistributedDataParallel(lit, device_ids=[local_rank], find_unused_parameters=False) if distributed else lit

    x = _dummy_input(lit.model, mcfg, dcfg, args.batch_size, device)
    n_classes = int(getattr(mcfg, "num_classes", getattr(dcfg, "num_classes", 10)))
    y = torch.randint(0, n_classes, (args.batch_size,), device=device)

    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    torch.npu.synchronize()

    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in lit.parameters())
    if not grad_ok:
        raise AssertionError("nanoSNN DDP step did not produce finite gradients")
    _rank_print(f"[PASS] nanoSNN DDP step {lit.model.__class__.__name__}: input={tuple(x.shape)} loss={loss.item():.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ascend HCCL/DDP smoke test for nanoSNN")
    parser.add_argument("--model_config", default="configs/model_configs/sdt_v1_small.yaml")
    parser.add_argument("--data_config", default="configs/data_configs/cifar10.yaml")
    parser.add_argument("--train_config", default="configs/train_configs/default.yaml")
    parser.add_argument("--optimizer_config", default="configs/optimizer_configs/sdtv1_cifar10.yaml")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--skip_model", action="store_true")
    parser.add_argument("--require_multi", action="store_true", help="Fail unless at least two NPUs are visible.")
    args = parser.parse_args()

    distributed = False
    try:
        device, rank, local_rank, world_size = _setup_npu(require_multi=args.require_multi)
        distributed = _setup_dist(world_size)
        _run_collective(device, rank, world_size, distributed)
        _run_tiny_ddp(device, local_rank, distributed)
        if not args.skip_model:
            _run_nanosnn_step(args, device, local_rank, distributed)
        if distributed:
            dist.barrier()
        if rank == 0:
            print("[PASS] Ascend HCCL/DDP smoke completed", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
