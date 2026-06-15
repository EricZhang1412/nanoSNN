#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.load_config import load_config  # noqa: E402
from models.build_model import build_model  # noqa: E402


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _run_triton_vector_add(device: torch.device) -> None:
    import triton
    import triton.language as tl

    globals()["tl"] = tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    size = 98432
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, size, BLOCK_SIZE=1024)
    torch.npu.synchronize()
    max_diff = (out - (x + y)).abs().max().item()
    if max_diff >= 1e-3:
        raise AssertionError(f"triton vector add mismatch: max_diff={max_diff:.3e}")
    print(f"[PASS] triton-ascend vector add max_diff={max_diff:.3e}")


def _dummy_input(model, model_config, data_config, batch_size: int, device: torch.device) -> torch.Tensor:
    image_size = int(getattr(data_config, "image_size", getattr(model_config, "image_size", 32)))
    in_channels = int(getattr(data_config, "in_channels", getattr(model_config, "in_channels", 3)))
    is_event = bool(getattr(data_config, "is_event", False))
    model_name = model.__class__.__name__
    model_key = str(getattr(model_config, "name", "")).lower()

    if model_key == "lra_transformer" or model_name == "LRATransformerEncoder":
        seq_len = int(getattr(model_config, "max_len", getattr(data_config, "seq_len", 1024)))
        vocab_size = int(getattr(model_config, "vocab_size", 256))
        return torch.randint(0, vocab_size, (batch_size, seq_len, 1), device=device)

    if model_key == "billeh_v1" or model_name == "BillehV1Classifier":
        steps = int(getattr(model_config, "T", getattr(data_config, "seq_len", 600)))
        if bool(getattr(model_config, "use_lgn", False)):
            height = int(getattr(model_config, "lgn_input_height", getattr(data_config, "lgn_height", image_size)))
            width = int(getattr(model_config, "lgn_input_width", getattr(data_config, "lgn_width", image_size)))
        else:
            height = width = image_size
        return torch.randn(steps, batch_size, in_channels, height, width, device=device)

    if model_name == "SpikeDrivenTransformerV3":
        return torch.randn(batch_size, in_channels, image_size, image_size, device=device)

    if is_event:
        steps = int(getattr(model_config, "T", getattr(data_config, "seq_len", 4)))
        return torch.randn(steps, batch_size, in_channels, image_size, image_size, device=device)

    return torch.randn(batch_size, in_channels, image_size, image_size, device=device)


def _run_model_step(args, device: torch.device) -> None:
    mcfg = load_config(args.model_config)
    dcfg = load_config(args.data_config)
    tcfg = load_config(args.train_config)
    ocfg = load_config(args.optimizer_config)

    lit = build_model(mcfg, ocfg, tcfg, dcfg).to(device)
    lit.train()
    x = _dummy_input(lit.model, mcfg, dcfg, args.batch_size, device)
    y = torch.randint(0, int(getattr(mcfg, "num_classes", getattr(dcfg, "num_classes", 10))), (args.batch_size,), device=device)

    optimizer_config = lit.configure_optimizers()
    optimizer = optimizer_config["optimizer"] if isinstance(optimizer_config, dict) else optimizer_config
    optimizer.zero_grad(set_to_none=True)
    logits = lit(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    torch.npu.synchronize()

    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in lit.parameters())
    if not grad_ok:
        raise AssertionError("no finite gradients were produced")
    print(f"[PASS] model step {lit.model.__class__.__name__}: input={tuple(x.shape)} loss={loss.item():.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ascend CANN 9.0.0 smoke test for nanoSNN")
    parser.add_argument("--model_config", default="configs/model_configs/sdt_v1_small.yaml")
    parser.add_argument("--data_config", default="configs/data_configs/cifar10.yaml")
    parser.add_argument("--train_config", default="configs/train_configs/default.yaml")
    parser.add_argument("--optimizer_config", default="configs/optimizer_configs/sdtv1_cifar10.yaml")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--skip_triton", action="store_true")
    parser.add_argument("--skip_model", action="store_true")
    args = parser.parse_args()

    _section("0. Imports and device")
    bisheng = shutil.which("bishengir-compile")
    print(f"bishengir-compile: {bisheng or 'not found'}")

    try:
        import torch_npu  # noqa: F401
    except Exception as exc:
        print(f"[FAIL] import torch_npu: {exc}")
        return 1

    print(f"torch: {torch.__version__}")
    print(f"torch_npu: {getattr(torch_npu, '__version__', 'unknown')}")
    if not torch.npu.is_available():
        print("[FAIL] torch.npu.is_available() is False")
        return 1
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    print(f"NPU count: {torch.npu.device_count()}")
    print(f"Using device: {device}")

    try:
        _section("1. torch-npu tensor ops")
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        out = a + b
        torch.npu.synchronize()
        if not torch.isfinite(out).all():
            raise AssertionError("non-finite result from tensor add")
        print("[PASS] torch-npu tensor add")

        if not args.skip_triton:
            _section("2. triton-ascend kernel")
            _run_triton_vector_add(device)

        if not args.skip_model:
            _section("3. nanoSNN model forward/backward")
            _run_model_step(args, device)
    except Exception:
        traceback.print_exc()
        return 1

    _section("Summary")
    print("[PASS] Ascend smoke test completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
