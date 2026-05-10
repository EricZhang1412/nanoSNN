"""Unit test for _to_b_t_n_via_lgn K-replay. Run as a script."""
from __future__ import annotations
import os, sys, types
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.load_config import load_config  # noqa: E402
from models.build_model import build_model  # noqa: E402


def _build():
    mcfg = load_config('configs/model_configs/billeh_v1_seqcifar10_lgn.yaml')
    dcfg = load_config('configs/data_configs/sequential_cifar10_fewshot.yaml')
    tcfg = load_config('configs/train_configs/default.yaml')
    ocfg = load_config('configs/optimizer_configs/billeh_seqcifar10.yaml')
    lit = build_model(mcfg, ocfg, tcfg, dcfg)
    return lit, lit.model


def test_replay_expands_t_dvs_to_t_lgn():
    """16-frame DVS-style input must expand to T_lgn=1024 by per-frame replication."""
    lit, m = _build()
    m.eval()
    device = next(m.parameters()).device
    T_dvs = 16
    movie = torch.arange(T_dvs, dtype=torch.float32, device=device).view(T_dvs, 1, 1, 1, 1)
    movie = movie.expand(T_dvs, 1, 1, 32, 32).contiguous()
    captured = {}
    real_lgn = m.lgn
    class _Spy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, x):
            captured['movie'] = x.detach().clone()
            return self.inner(x)
    m.lgn = _Spy(real_lgn).to(device)

    with torch.no_grad():
        m._to_b_t_n_via_lgn(movie)

    spy_movie = captured['movie']
    assert spy_movie.shape == (1, 1024, 32, 32), spy_movie.shape
    K = 1024 // T_dvs
    for i in range(T_dvs):
        block = spy_movie[0, i * K : (i + 1) * K]
        assert torch.allclose(block, torch.full_like(block, float(i))), (
            f"block {i} expected constant {i}, got mean {block.mean().item()}"
        )
    m.lgn = real_lgn


def test_replay_passes_through_when_t_matches():
    lit, m = _build()
    m.eval()
    device = next(m.parameters()).device
    movie = torch.zeros(1024, 1, 1, 32, 32, device=device)
    movie[100, 0, 0, 0, 0] = 1.0
    captured = {}
    real_lgn = m.lgn
    class _Spy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, x):
            captured['movie'] = x.detach().clone()
            return self.inner(x)
    m.lgn = _Spy(real_lgn).to(device)

    with torch.no_grad():
        m._to_b_t_n_via_lgn(movie)
    spy_movie = captured['movie']
    assert spy_movie.shape == (1, 1024, 32, 32)
    assert spy_movie[0, 100, 0, 0].item() == 1.0
    m.lgn = real_lgn


def test_replay_rejects_non_divisible():
    lit, m = _build()
    m.eval()
    device = next(m.parameters()).device
    movie = torch.zeros(15, 1, 1, 32, 32, device=device)
    try:
        m._to_b_t_n_via_lgn(movie)
    except ValueError as e:
        assert "positive multiple" in str(e), str(e)
        return
    raise AssertionError("expected ValueError for non-divisible T")


if __name__ == "__main__":
    test_replay_expands_t_dvs_to_t_lgn()
    test_replay_passes_through_when_t_matches()
    test_replay_rejects_non_divisible()
    print("[test_billeh_lgn_replay] OK")
