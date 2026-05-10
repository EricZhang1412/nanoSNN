"""Unit test for build_dvs_lgn_transform. Run as a script."""
from __future__ import annotations
import os, sys, types
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.transforms import build_dvs_lgn_transform  # noqa: E402


def _cfg(**kwargs):
    return types.SimpleNamespace(**kwargs)


def test_signed_diff_native_resolution():
    cfg = _cfg(name="cifar10dvs", image_size=128, polarity_mode="signed")
    transform = build_dvs_lgn_transform(cfg)
    on = np.zeros((16, 2, 128, 128), dtype=np.int16)
    on[:, 0] = 5
    out = transform(on)
    assert out.shape == (16, 1, 128, 128), out.shape
    assert (out > 0).all(), "ON-only must yield positive signed-diff"
    assert out.max().item() <= 1.0 + 1e-6, out.max().item()
    assert out.dtype == torch.float32, out.dtype


def test_signed_diff_resize_to_48():
    cfg = _cfg(name="cifar10dvs", image_size=48, polarity_mode="signed")
    transform = build_dvs_lgn_transform(cfg)
    on_off = np.zeros((16, 2, 128, 128), dtype=np.int16)
    on_off[:, 0, :64] = 3
    on_off[:, 1, 64:] = 3
    out = transform(on_off)
    assert out.shape == (16, 1, 48, 48), out.shape
    top_mean = out[:, 0, :24].mean().item()
    bot_mean = out[:, 0, 24:].mean().item()
    assert top_mean > 0.5 and bot_mean < -0.5, f"top={top_mean}, bot={bot_mean}"


def test_zero_input_returns_zero():
    cfg = _cfg(name="cifar10dvs", image_size=48, polarity_mode="signed")
    transform = build_dvs_lgn_transform(cfg)
    zero = np.zeros((16, 2, 128, 128), dtype=np.int16)
    out = transform(zero)
    assert out.shape == (16, 1, 48, 48)
    assert torch.all(out == 0), "zero input must yield zero output (denom clamp)"


def test_mean_polarity_mode():
    cfg = _cfg(name="cifar10dvs", image_size=48, polarity_mode="mean")
    transform = build_dvs_lgn_transform(cfg)
    arr = np.zeros((16, 2, 128, 128), dtype=np.int16)
    arr[:, 0] = 4; arr[:, 1] = 4
    out = transform(arr)
    assert out.shape == (16, 1, 48, 48)
    assert (out >= 0).all(), "mean mode is unsigned"


if __name__ == "__main__":
    test_signed_diff_native_resolution()
    test_signed_diff_resize_to_48()
    test_zero_input_returns_zero()
    test_mean_polarity_mode()
    print("[test_dvs_lgn_transform] OK")
