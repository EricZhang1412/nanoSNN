# Billeh-V1 + LGN on DVS Datasets — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train `billeh_v1` (V1 column + LGN front-end) on CIFAR10-DVS and DVS128Gesture, with biologically-calibrated firing rates and the existing Chen-Maass rate regularization.

**Architecture:** Data-side polarity collapse (signed ON−OFF) + per-sample magnitude normalization → spikingjelly's standard collate → model-side K-replay to align `T_dvs` (e.g. 16) with `T_lgn` (1024) → unchanged TorchLGN + V1 column. Per-dataset variation lives entirely in YAML configs.

**Tech Stack:** PyTorch 2.10, Lightning, spikingjelly 0.0.0.0.14, existing `BillehV1Classifier` + `TorchLGN`. Project venv at `.venv/bin/python`. Tests run as standalone scripts (no pytest setup in project).

**Spec:** `docs/superpowers/specs/2026-05-11-billeh-dvs-design.md`

---

## File Structure

**Modify:**
- `data/transforms.py` — add `build_dvs_lgn_transform`
- `data/event_datasets.py` — switch transform on `data_config.transform_type`
- `models/billeh_v1/model.py` — extend ndim=5 LGN branch with K-replay; fix 11-class localized-readout pool selection

**Create:**
- `configs/data_configs/cifar10dvs_lgn.yaml`
- `configs/data_configs/dvs128gesture_lgn.yaml`
- `configs/model_configs/billeh_v1_cifar10dvs_lgn.yaml`
- `configs/model_configs/billeh_v1_dvsgesture_lgn.yaml`
- `scripts/test_dvs_lgn_transform.py` — unit test for the transform
- `scripts/test_billeh_lgn_replay.py` — unit test for K-replay branch
- `scripts/test_billeh_readout_pools.py` — unit test for 10/11-class pool selection
- `scripts/smoke_billeh_v1_dvs.py` — end-to-end smoke test (parameterized)

**Cache directories the user must prepare beforehand:**
- `/data2/dataset/cifar10dvs/` — created automatically; spikingjelly downloads CIFAR10-DVS aedat files on first run.
- `/data2/dataset/dvs128gesture/download/` — user must place `DvsGesture.tar.gz` (IBM Box manual download) here before first run; spikingjelly cannot auto-download.

---

## Task 1: Regression baseline — confirm seq-CIFAR LGN smoke test still passes

Before any code changes, lock in a green baseline so we can detect regressions from the model.py edits.

**Files:**
- Run: `scripts/smoke_billeh_v1_lgn.py`-equivalent for seq-CIFAR (we'll build it inline)

- [ ] **Step 1: Run the existing seq-CIFAR LGN smoke check**

Run:
```bash
cd /data2/users/zhangjy/nanoSNN && .venv/bin/python -c "
import sys, torch
sys.path.insert(0, '.')
from utils.load_config import load_config
from models.build_model import build_model

mcfg = load_config('configs/model_configs/billeh_v1_seqcifar10_lgn.yaml')
dcfg = load_config('configs/data_configs/sequential_cifar10_fewshot.yaml')
tcfg = load_config('configs/train_configs/default.yaml')
ocfg = load_config('configs/optimizer_configs/billeh_seqcifar10.yaml')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lit = build_model(mcfg, ocfg, tcfg, dcfg).to(device)
lit.train()
m = lit.model
B = 2
x = torch.rand(B, 1024, 3, device=device)
logits = lit(x)
spikes = m.last_spikes
rate_hz = (spikes.detach().float().mean() * 1000.0).item()
assert logits.shape == (B, 10), logits.shape
assert m.n_input == 17400, m.n_input
assert 5.0 < rate_hz < 30.0, f'baseline rate out of band: {rate_hz}'
print(f'[regression] OK n_input=17400, rate={rate_hz:.2f} Hz')
"
```

Expected: prints `[regression] OK n_input=17400, rate=~21 Hz` (the value should be in [5, 30] Hz).

Do NOT proceed if this fails — the baseline is broken.

- [ ] **Step 2: No commit needed (no changes yet).**

---

## Task 2: Add `build_dvs_lgn_transform` (TDD)

**Files:**
- Modify: `data/transforms.py` — add function and import for `torch.nn.functional`
- Create: `scripts/test_dvs_lgn_transform.py`

- [ ] **Step 1: Write the failing unit test**

Create `scripts/test_dvs_lgn_transform.py`:
```python
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
    # ON-only event frame -> all positive; OFF-only -> all negative.
    on = np.zeros((16, 2, 128, 128), dtype=np.int16)
    on[:, 0] = 5  # 5 ON events per pixel
    out = transform(on)
    assert out.shape == (16, 1, 128, 128), out.shape
    assert (out > 0).all(), "ON-only must yield positive signed-diff"
    assert out.max().item() <= 1.0 + 1e-6, out.max().item()
    assert out.dtype == torch.float32, out.dtype


def test_signed_diff_resize_to_48():
    cfg = _cfg(name="cifar10dvs", image_size=48, polarity_mode="signed")
    transform = build_dvs_lgn_transform(cfg)
    on_off = np.zeros((16, 2, 128, 128), dtype=np.int16)
    on_off[:, 0, :64] = 3       # ON in top half
    on_off[:, 1, 64:] = 3       # OFF in bottom half
    out = transform(on_off)
    assert out.shape == (16, 1, 48, 48), out.shape
    # top half should be roughly +1, bottom half ~-1.
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
    arr[:, 0] = 4; arr[:, 1] = 4   # equal ON/OFF -> mean is positive (intensity)
    out = transform(arr)
    assert out.shape == (16, 1, 48, 48)
    assert (out >= 0).all(), "mean mode is unsigned"


if __name__ == "__main__":
    test_signed_diff_native_resolution()
    test_signed_diff_resize_to_48()
    test_zero_input_returns_zero()
    test_mean_polarity_mode()
    print("[test_dvs_lgn_transform] OK")
```

- [ ] **Step 2: Run the test, confirm it fails with ImportError**

Run:
```bash
cd /data2/users/zhangjy/nanoSNN && .venv/bin/python scripts/test_dvs_lgn_transform.py
```
Expected: `ImportError: cannot import name 'build_dvs_lgn_transform' from 'data.transforms'`.

- [ ] **Step 3: Implement `build_dvs_lgn_transform`**

Edit `data/transforms.py`. Add at the bottom (keeping all existing code):
```python
def build_dvs_lgn_transform(data_config):
    """Polarity collapse + per-sample magnitude normalization for spikingjelly DVS frames.

    Input: [T_dvs, 2, H_native, W_native] event counts (np.ndarray or tensor).
    Output: [T_dvs, 1, H_out, W_out] float32 in approximately [-1, 1].
    """
    import torch.nn.functional as F  # local import; transforms.py uses torchvision elsewhere

    image_size = int(getattr(data_config, "image_size", 0) or 0)
    polarity_mode = str(getattr(data_config, "polarity_mode", "signed")).lower()

    def _transform(frames):
        if isinstance(frames, np.ndarray):
            tensor = torch.from_numpy(frames)
        elif torch.is_tensor(frames):
            tensor = frames
        else:
            tensor = torch.tensor(frames)
        tensor = tensor.float()
        if tensor.ndim != 4 or tensor.shape[1] != 2:
            raise ValueError(
                f"build_dvs_lgn_transform expects [T, 2, H, W], got {tuple(tensor.shape)}"
            )

        if polarity_mode == "signed":
            collapsed = tensor[:, 0] - tensor[:, 1]    # [T, H, W]
        elif polarity_mode == "mean":
            collapsed = tensor.mean(dim=1)              # [T, H, W]
        else:
            raise ValueError(
                f"polarity_mode must be 'signed' or 'mean', got {polarity_mode!r}"
            )

        if image_size > 0 and (collapsed.shape[-1] != image_size or collapsed.shape[-2] != image_size):
            collapsed = F.interpolate(
                collapsed.unsqueeze(1),                 # [T, 1, H, W]
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        denom = collapsed.abs().amax().clamp_min(1e-3)
        collapsed = collapsed / denom
        return collapsed.unsqueeze(1).contiguous()      # [T, 1, H, W]

    return _transform
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `.venv/bin/python scripts/test_dvs_lgn_transform.py`
Expected: `[test_dvs_lgn_transform] OK`

- [ ] **Step 5: Commit**

```bash
git add data/transforms.py scripts/test_dvs_lgn_transform.py
git commit -m "$(cat <<'EOF'
[DVS]: add build_dvs_lgn_transform for signed-diff polarity collapse

Reduces spikingjelly's [T, 2, H, W] event frames to [T, 1, H, W] signed
luminance change, with bilinear resize to image_size and per-sample
magnitude normalization to ~[-1, 1]. Used by the billeh-v1 LGN path so
ON-cells and OFF-cells receive natural signed input.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Wire DVS-LGN transform into event dataset builder

**Files:**
- Modify: `data/event_datasets.py:476-484` (cifar10dvs / dvs128gesture branches)

- [ ] **Step 1: Locate the existing branches**

Read `data/event_datasets.py:470-485` and verify the current dispatch:
```python
def build_event_dataset(data_config, split: str):
    name = _dataset_name(data_config)
    root = os.path.expanduser(getattr(data_config, "root", "./datasets"))
    transform = build_event_transform(data_config)
    kwargs = _event_kwargs(data_config)

    if name == "cifar10dvs":
        ...
    if name == "dvs128gesture":
        ...
```

- [ ] **Step 2: Modify the transform selection**

Replace the line `transform = build_event_transform(data_config)` (currently at `data/event_datasets.py:473`) with:
```python
transform_type = str(getattr(data_config, "transform_type", "default")).lower()
if transform_type == "dvs_lgn":
    from .transforms import build_dvs_lgn_transform
    transform = build_dvs_lgn_transform(data_config)
else:
    transform = build_event_transform(data_config)
```

(`build_event_transform` is already imported at the top of the file.)

- [ ] **Step 3: Verify with a one-shot smoke (no dataset download yet)**

Run:
```bash
cd /data2/users/zhangjy/nanoSNN && .venv/bin/python -c "
import types
from data.event_datasets import build_event_dataset
# Use a fake config to verify the dispatch path; we just want to assert that
# transform_type='dvs_lgn' selects build_dvs_lgn_transform without crashing
# at the dispatch step. Actual dataset construction is exercised in the
# end-to-end smoke test (Task 8) once data is downloaded.
import data.event_datasets as ed
import data.transforms as dt
seen = {}
real_lgn = dt.build_dvs_lgn_transform
def spy(cfg):
    seen['called'] = True
    return real_lgn(cfg)
dt.build_dvs_lgn_transform = spy

cfg = types.SimpleNamespace(
    name='cifar10dvs', root='/tmp/__nonexistent_dvs_root__',
    transform_type='dvs_lgn', polarity_mode='signed', image_size=48,
    event_data_type='frame', frames_number=16, split_by='number',
    train_ratio=0.9, split_seed=42,
)
try:
    build_event_dataset(cfg, split='train')
except Exception as e:
    pass  # we only care that the transform branch fired before any dataset I/O.
assert seen.get('called'), 'dvs_lgn transform was not selected'
print('[wire-up] OK')
"
```
Expected: `[wire-up] OK`.

- [ ] **Step 4: Commit**

```bash
git add data/event_datasets.py
git commit -m "$(cat <<'EOF'
[DVS]: wire dvs_lgn transform into event dataset builder

Adds transform_type='dvs_lgn' switch in build_event_dataset so cifar10dvs
and dvs128gesture configs can opt into signed-diff polarity collapse for
the billeh-v1 LGN path without affecting non-LGN model usage.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Extend `_to_b_t_n_via_lgn` ndim=5 branch with K-replay (TDD)

**Files:**
- Modify: `models/billeh_v1/model.py:218-227` (ndim=5 branch)
- Create: `scripts/test_billeh_lgn_replay.py`

- [ ] **Step 1: Write the failing unit test**

Create `scripts/test_billeh_lgn_replay.py`:
```python
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
    # Build a [T_dvs=16, B=1, C=1, H=32, W=32] event movie where frame i = i.
    T_dvs = 16
    movie = torch.arange(T_dvs, dtype=torch.float32, device=device).view(T_dvs, 1, 1, 1, 1)
    movie = movie.expand(T_dvs, 1, 1, 32, 32).contiguous()
    # NOTE: _to_b_t_n_via_lgn returns LGN firing rates [B, T_lgn, n_lgn], so
    # we cannot directly compare frame-by-frame; instead we monkey-patch the
    # LGN to be the identity and check that the movie passed in has T_lgn=1024
    # with each input frame replicated 64 times.
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

    spy_movie = captured['movie']               # [B=1, T_lgn, H, W]
    assert spy_movie.shape == (1, 1024, 32, 32), spy_movie.shape
    K = 1024 // T_dvs
    # Each block of K frames must equal the corresponding input frame value.
    for i in range(T_dvs):
        block = spy_movie[0, i * K : (i + 1) * K]
        assert torch.allclose(block, torch.full_like(block, float(i))), (
            f"block {i} expected constant {i}, got mean {block.mean().item()}"
        )


def test_replay_passes_through_when_t_matches():
    """When T_input == self.T already, no replay required."""
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


def test_replay_rejects_non_divisible():
    """T must divide evenly into T_lgn."""
    lit, m = _build()
    m.eval()
    device = next(m.parameters()).device
    # T_lgn=1024, T_in=15 -> 1024 % 15 != 0 -> ValueError
    movie = torch.zeros(15, 1, 1, 32, 32, device=device)
    try:
        m._to_b_t_n_via_lgn(movie)
    except ValueError as e:
        assert "must be a multiple" in str(e) or "divides" in str(e).lower(), str(e)
        return
    raise AssertionError("expected ValueError for non-divisible T")


if __name__ == "__main__":
    test_replay_expands_t_dvs_to_t_lgn()
    test_replay_passes_through_when_t_matches()
    test_replay_rejects_non_divisible()
    print("[test_billeh_lgn_replay] OK")
```

- [ ] **Step 2: Run, confirm two of three tests FAIL**

Run: `.venv/bin/python scripts/test_billeh_lgn_replay.py`
Expected: First test FAILS with assertion on movie shape (current code passes T_dvs=16 to LGN, getting shape `(1, 16, 32, 32)` instead of `(1, 1024, 32, 32)`). The non-divisible test will likely fail too because the current code raises a different error inside LGN.

- [ ] **Step 3: Apply the K-replay diff**

Edit `models/billeh_v1/model.py`. Locate this block (currently around lines 218-227):
```python
        if x.ndim == 5:
            # [T, B, C, H, W] -> [B, T, H, W]
            x = x.permute(1, 0, 2, 3, 4)
            if x.shape[2] > 1:
                movie = x.mean(dim=2)
            else:
                movie = x[:, :, 0]
```

Replace with:
```python
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
                        f"model.T={t_lgn} must be a positive multiple of input T={t_in} for K-replay"
                    )
                k = t_lgn // t_in
                movie = movie.repeat_interleave(k, dim=1)
```

- [ ] **Step 4: Run the test again, confirm all three pass**

Run: `.venv/bin/python scripts/test_billeh_lgn_replay.py`
Expected: `[test_billeh_lgn_replay] OK`

- [ ] **Step 5: Re-run the regression baseline from Task 1**

Run the same `[regression]` block from Task 1 Step 1.
Expected: still prints `[regression] OK n_input=17400, rate=...`. The seq-CIFAR path does NOT enter the ndim=5 branch (its input is ndim=3), so this confirms no collateral breakage.

- [ ] **Step 6: Commit**

```bash
git add models/billeh_v1/model.py scripts/test_billeh_lgn_replay.py
git commit -m "$(cat <<'EOF'
[Billeh-v1]: K-replay short event sequences in LGN ndim=5 branch

DVS datasets feed [T_dvs=16, B, C, H, W] frames; LGN's temporal kernel and
V1's GLIF dynamics work at ms resolution. When self.T % T_input == 0,
each input frame is replicated self.T // T_input times via
repeat_interleave so LGN sees a slow movie at ms cadence. Backward-compat
when T_input == self.T (no-op) and rejects non-divisible mismatches with
a clear error.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Fix 11-class localized-readout pool selection (TDD)

**Files:**
- Modify: `models/billeh_v1/model.py:138-144`
- Create: `scripts/test_billeh_readout_pools.py`

- [ ] **Step 1: Write the failing unit test**

Create `scripts/test_billeh_readout_pools.py`:
```python
"""Unit test for 10/11-class localized-readout pool selection."""
from __future__ import annotations
import os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.load_config import load_config  # noqa: E402
from models.build_model import build_model  # noqa: E402


def _build_with_classes(num_classes: int):
    mcfg = load_config('configs/model_configs/billeh_v1_seqcifar10_lgn.yaml')
    mcfg.num_classes = num_classes
    dcfg = load_config('configs/data_configs/sequential_cifar10_fewshot.yaml')
    tcfg = load_config('configs/train_configs/default.yaml')
    ocfg = load_config('configs/optimizer_configs/billeh_seqcifar10.yaml')
    lit = build_model(mcfg, ocfg, tcfg, dcfg)
    return lit.model


def test_10_class_uses_pools_5_to_14():
    m = _build_with_classes(10)
    pool_sets = [set(m.readout.pool_indices(c).tolist()) for c in range(10)]
    # All disjoint.
    for i in range(10):
        for j in range(i + 1, 10):
            assert pool_sets[i].isdisjoint(pool_sets[j]), (
                f"pools {i} and {j} overlap (10-class)"
            )


def test_11_class_no_pool_aliasing():
    """The bug: with num_classes=11, current code aliases class 5 and class 10
    to the same pool (pool 10). Fix must give 11 disjoint pools."""
    m = _build_with_classes(11)
    pool_sets = [set(m.readout.pool_indices(c).tolist()) for c in range(11)]
    for i in range(11):
        assert len(pool_sets[i]) > 0, f"class {i} has empty pool"
        for j in range(i + 1, 11):
            assert pool_sets[i].isdisjoint(pool_sets[j]), (
                f"pools {i} and {j} overlap (11-class) — readout aliasing bug"
            )


if __name__ == "__main__":
    test_10_class_uses_pools_5_to_14()
    test_11_class_no_pool_aliasing()
    print("[test_billeh_readout_pools] OK")
```

- [ ] **Step 2: Run, confirm 11-class test FAILS**

Run: `.venv/bin/python scripts/test_billeh_readout_pools.py`
Expected: `AssertionError: pools 5 and 10 overlap (11-class) — readout aliasing bug`. (The 10-class test will pass.)

- [ ] **Step 3: Apply the readout fix**

Edit `models/billeh_v1/model.py`. Locate the pool-id loop (currently at lines 137-144):
```python
        # Pool ids: upstream uses pools 5..14 for the 10-class task (skip 0..4
        # reserved for the garrett 2-class task).
        network = loaded["network"]
        pool_ids = []
        for i in range(self.num_classes):
            key = f"localized_readout_neuron_ids_{i + 5}"
            if key not in network:
                # fall back to pools 0..(num_classes-1) if 5..14 are unavailable
                key = f"localized_readout_neuron_ids_{i}"
            ids = np.asarray(network[key]).reshape(-1)
            pool_ids.append(ids)
```

Replace with:
```python
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
```

- [ ] **Step 4: Run the test, confirm both pass**

Run: `.venv/bin/python scripts/test_billeh_readout_pools.py`
Expected: `[test_billeh_readout_pools] OK`

- [ ] **Step 5: Re-run regression baseline (Task 1 Step 1)**

Expected: still prints rate in [5, 30] Hz. (10-class path is preserved.)

- [ ] **Step 6: Commit**

```bash
git add models/billeh_v1/model.py scripts/test_billeh_readout_pools.py
git commit -m "$(cat <<'EOF'
[Billeh-v1]: fix localized-readout pool aliasing for non-10-class tasks

Previous code tried pool keys i+5 then fell back to key i, which for
num_classes=11 silently aliased class 5 and class 10 to the same pool. Now
selects offset=5 only when num_classes==10 and pool 5 exists; otherwise
uses offset=0 so each class gets a disjoint pool. Caps at 15 (the number
of preset readout origins).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Create CIFAR10-DVS data and model configs

**Files:**
- Create: `configs/data_configs/cifar10dvs_lgn.yaml`
- Create: `configs/model_configs/billeh_v1_cifar10dvs_lgn.yaml`

- [ ] **Step 1: Create the data config**

Write `configs/data_configs/cifar10dvs_lgn.yaml`:
```yaml
name: cifar10dvs
root: /data2/dataset/cifar10dvs
num_classes: 10
image_size: 48           # transform resizes from native 128x128 to 48x48
in_channels: 1           # post signed-diff polarity collapse
is_event: true

# spikingjelly frame integration: 16 frames per sample, each frame integrates
# events by count across the full duration.
event_data_type: frame
frames_number: 16
split_by: number

# DVS-LGN transform: signed-diff (ON - OFF), bilinear resize to image_size,
# per-sample magnitude normalization to ~[-1, 1].
transform_type: dvs_lgn
polarity_mode: signed

# Train/val split (CIFAR10-DVS has only a 10k single split — random 90/10).
train_ratio: 0.9
split_seed: 42

num_workers: 4
pin_memory: true
```

- [ ] **Step 2: Create the model config**

Write `configs/model_configs/billeh_v1_cifar10dvs_lgn.yaml`:
```yaml
name: billeh_v1
T: 1024                  # K-replay: 16 DVS frames × 64 LGN steps each
num_classes: 10
in_channels: 1

# Trial timing: skip the first DVS frame (~64 LGN ms) so LGN's temporal kernel
# has time to settle before the readout integrates.
pre_delay: 64
post_delay: 0
response_window_len: 0
down_sample: 64

# Billeh V1 backbone setup. Keep auto_n_input_from_in_channels=false so n_input
# stays at 17400 (LGN cell count), not collapsed to in_channels=1.
billeh_data_dir: /data2/dataset/LGN_GLIF_Models/new_0505/GLIF_network2
n_neurons: 3000
neurons_per_output: 30
localized_readout: false
full_core: false
train_v1: true
seed: 3407
use_input_t: true
auto_n_input_from_in_channels: false

# LGN front-end. Spatial size matches data_config.image_size.
use_lgn: true
auto_n_input_from_lgn: true
lgn_data_path: /data2/dataset/LGN_GLIF_Models/new_0505/GLIF_network2/lgn_full_col_cells_3.csv
lgn_temporal_kernels_path: /data2/dataset/LGN_GLIF_Models/new_0505/GLIF_network2/temporal_kernels.pkl
lgn_input_height: 48
lgn_input_width: 48

# Identity passes LGN firing-rate currents straight to V1.
encoding: identity
encoding_gain: 1.0

# Initial guess; calibrate via smoke test (Task 7) until V1 rate sits in
# [10, 30] Hz at init. DVS event magnitudes are typically smaller than the
# CIFAR luminance signal, so start ~7x smaller than the seq-CIFAR config.
lgn_input_scale: 1.0e-3

dampening_factor: 0.3
gauss_std: 0.5
chunk_size: 64
```

- [ ] **Step 3: Verify both YAMLs parse**

Run:
```bash
.venv/bin/python -c "
from utils.load_config import load_config
m = load_config('configs/model_configs/billeh_v1_cifar10dvs_lgn.yaml')
d = load_config('configs/data_configs/cifar10dvs_lgn.yaml')
assert m.num_classes == 10 and d.num_classes == 10
assert m.T == 1024 and d.frames_number == 16 and m.T % d.frames_number == 0
assert m.lgn_input_height == d.image_size
print('[configs] CIFAR10-DVS configs OK')
"
```
Expected: `[configs] CIFAR10-DVS configs OK`.

- [ ] **Step 4: Commit**

```bash
git add configs/data_configs/cifar10dvs_lgn.yaml configs/model_configs/billeh_v1_cifar10dvs_lgn.yaml
git commit -m "$(cat <<'EOF'
[DVS]: add CIFAR10-DVS configs for billeh-v1 LGN pipeline

Data config: spikingjelly frame integration T=16, signed-diff transform,
48x48 spatial. Model config: T=1024 (K=64 replay), n_input=17400, LGN
input scale 1e-3 (initial guess; tune via smoke test).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: End-to-end smoke test on CIFAR10-DVS (calibrate `lgn_input_scale`)

**Files:**
- Create: `scripts/smoke_billeh_v1_dvs.py`

- [ ] **Step 1: Write the smoke script**

Create `scripts/smoke_billeh_v1_dvs.py`:
```python
"""End-to-end smoke test for billeh-v1 LGN on DVS datasets.

Examples:
    .venv/bin/python scripts/smoke_billeh_v1_dvs.py \\
        --model_config configs/model_configs/billeh_v1_cifar10dvs_lgn.yaml \\
        --data_config configs/data_configs/cifar10dvs_lgn.yaml

    .venv/bin/python scripts/smoke_billeh_v1_dvs.py \\
        --model_config configs/model_configs/billeh_v1_dvsgesture_lgn.yaml \\
        --data_config configs/data_configs/dvs128gesture_lgn.yaml \\
        --batch 2

Checks: forward shape, n_input==17400, rate in [5, 30] Hz, finite gradients.
"""
from __future__ import annotations
import argparse
import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.load_config import load_config  # noqa: E402
from models.build_model import build_model  # noqa: E402
from data.event_datasets import build_event_dataset  # noqa: E402
from data.build import event_collate_fn  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_config", required=True)
    p.add_argument("--data_config", required=True)
    p.add_argument("--train_config", default="configs/train_configs/default.yaml")
    p.add_argument("--optimizer_config", default="configs/optimizer_configs/billeh_seqcifar10.yaml")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    print(f"[smoke] device={device}")

    mcfg = load_config(args.model_config)
    dcfg = load_config(args.data_config)
    tcfg = load_config(args.train_config)
    ocfg = load_config(args.optimizer_config)

    lit = build_model(mcfg, ocfg, tcfg, dcfg).to(device)
    lit.train()
    m = lit.model
    print(f"[smoke] model T={m.T}, n_input={m.n_input}, num_classes={m.num_classes}")
    assert m.n_input == 17400, m.n_input

    print("[smoke] building train split (may trigger spikingjelly preprocessing)…")
    ds = build_event_dataset(dcfg, split="train")
    print(f"[smoke] dataset size: {len(ds)}")

    samples = [ds[i] for i in range(args.batch)]
    x, y = event_collate_fn(samples)
    x = x.to(device)
    y = y.to(device)
    print(f"[smoke] input batch: {tuple(x.shape)}, dtype={x.dtype}, "
          f"range=[{x.min().item():.3f}, {x.max().item():.3f}]")

    logits = lit(x)
    print(f"[smoke] logits: {tuple(logits.shape)}")
    assert logits.shape == (args.batch, m.num_classes)

    spikes = m.last_spikes
    rate_hz = (spikes.detach().float().mean() * 1000.0).item()
    print(f"[smoke] population mean rate ≈ {rate_hz:.2f} Hz")
    if not (5.0 <= rate_hz <= 30.0):
        print(f"[smoke] WARNING: rate {rate_hz:.2f} Hz outside [5, 30] target — "
              f"adjust model_config.lgn_input_scale "
              f"(current={getattr(mcfg, 'lgn_input_scale', None)}).")

    loss = lit._shared_step((x, y), split="train")
    print(f"[smoke] loss={loss.item():.4f}")
    loss.backward()

    grads = [(n, p.grad.abs().mean().item())
             for n, p in lit.named_parameters()
             if p.grad is not None and p.grad.abs().sum() > 0]
    assert grads, "no non-zero gradients"
    print(f"[smoke] non-zero grad params: {len(grads)}")
    print("[smoke] OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Trigger spikingjelly's first-time CIFAR10-DVS download/preprocess**

This step takes 5-15 minutes the first time (downloads aedat files, integrates frames, writes cache). Run:
```bash
mkdir -p /data2/dataset/cifar10dvs
.venv/bin/python scripts/smoke_billeh_v1_dvs.py \
    --model_config configs/model_configs/billeh_v1_cifar10dvs_lgn.yaml \
    --data_config configs/data_configs/cifar10dvs_lgn.yaml \
    --batch 2 2>&1 | tail -40
```
Expected on first run: spikingjelly logs `[Downloading]`, `[Extracting]`, `[Integrating events…]`, then the smoke output.
On subsequent runs: skips straight to the smoke output.

- [ ] **Step 3: Read the rate output**

Look for the `[smoke] population mean rate ≈ X.XX Hz` line.

- If `5 ≤ X ≤ 30`: calibration is fine. Continue to Step 5.
- If `X > 30`: **decrease** `lgn_input_scale` in `configs/model_configs/billeh_v1_cifar10dvs_lgn.yaml` by 3-5×. Re-run Step 2.
- If `X < 5`: **increase** `lgn_input_scale` by 3-5×. Re-run Step 2.
- If `X ≈ 0`: increase by 10×.

The `[smoke] WARNING` line tells you which way to go.

- [ ] **Step 4: Iterate Step 2-3 until rate lands in [5, 30] Hz**

Typical convergence: 1-3 iterations. Record the final scale value.

- [ ] **Step 5: Verify the smoke prints `[smoke] OK`**

Expected: full smoke completes with `n_input=17400`, rate in band, non-zero grads.

- [ ] **Step 6: Commit**

```bash
git add scripts/smoke_billeh_v1_dvs.py configs/model_configs/billeh_v1_cifar10dvs_lgn.yaml
git commit -m "$(cat <<'EOF'
[DVS]: add billeh-v1 DVS smoke test and calibrate cifar10dvs LGN scale

End-to-end smoke verifies n_input=17400, forward/backward parity, and
initial population rate in [5, 30] Hz. The lgn_input_scale in
billeh_v1_cifar10dvs_lgn.yaml is set to the value found by calibration.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Launch CIFAR10-DVS training (sanity run, 1 epoch)

**Files:**
- None (uses existing `train.sh` and configs from Tasks 6-7)

- [ ] **Step 1: Inspect train.sh argument convention**

Read `train.sh` and verify it accepts `MODEL DATA OPTIMIZER` positional args. Default optimizer is `sdtv3_cifar10` — for billeh we override with `billeh_seqcifar10` (the rate-cost optimizer config from earlier work).

- [ ] **Step 2: Launch a single-epoch run**

Run (this will take 30-60 minutes for one epoch on a 4090):
```bash
bash train.sh billeh_v1_cifar10dvs_lgn cifar10dvs_lgn billeh_seqcifar10
```

Watch the first ~50 batches for:
- `train/spike_rate_hz` stays in [5, 40] Hz (rate-cost should pull rates back if they drift)
- `train/loss_step` decreasing (or at least not exploding)
- No NaN / Inf

If rate diverges or loss NaNs, abort with Ctrl-C, lower `lgn_input_scale` further or check Task 4-5 changes.

- [ ] **Step 3: Stop after 1 epoch validates, no commit needed**

This is a smoke run; we just want to confirm the pipeline trains end-to-end. Full training is a separate workstream.

---

## Task 9: Create DVS128Gesture data and model configs

**Files:**
- Create: `configs/data_configs/dvs128gesture_lgn.yaml`
- Create: `configs/model_configs/billeh_v1_dvsgesture_lgn.yaml`

- [ ] **Step 1: Pre-stage the DVS128Gesture archive**

DVS128Gesture is not auto-downloadable. Verify the archive is in place:
```bash
ls /data2/dataset/dvs128gesture/download/DvsGesture.tar.gz 2>&1
```
If absent: download `DvsGesture.tar.gz` from IBM Box (https://research.ibm.com/interactive/dvsgesture/) and place it at `/data2/dataset/dvs128gesture/download/DvsGesture.tar.gz`. spikingjelly extracts and integrates on first run.

If you can't get it right now, skip this task and Task 10; CIFAR10-DVS work is independent.

- [ ] **Step 2: Create the data config**

Write `configs/data_configs/dvs128gesture_lgn.yaml`:
```yaml
name: dvs128gesture
root: /data2/dataset/dvs128gesture
num_classes: 11
image_size: 128          # native sensor; LGN downsamples internally via grid_sample
in_channels: 1
is_event: true

# IMPORTANT: spikingjelly cannot auto-download DVS128Gesture. Place
# DvsGesture.tar.gz under <root>/download/ before first run.
event_data_type: frame
frames_number: 16
split_by: number

transform_type: dvs_lgn
polarity_mode: signed

num_workers: 4
pin_memory: true
```

- [ ] **Step 3: Create the model config**

Write `configs/model_configs/billeh_v1_dvsgesture_lgn.yaml`:
```yaml
name: billeh_v1
T: 1024
num_classes: 11
in_channels: 1

# Trial timing — same as cifar10dvs.
pre_delay: 64
post_delay: 0
response_window_len: 0
down_sample: 64

billeh_data_dir: /data2/dataset/LGN_GLIF_Models/new_0505/GLIF_network2
n_neurons: 3000
neurons_per_output: 30
localized_readout: false
full_core: false
train_v1: true
seed: 3407
use_input_t: true
auto_n_input_from_in_channels: false

use_lgn: true
auto_n_input_from_lgn: true
lgn_data_path: /data2/dataset/LGN_GLIF_Models/new_0505/GLIF_network2/lgn_full_col_cells_3.csv
lgn_temporal_kernels_path: /data2/dataset/LGN_GLIF_Models/new_0505/GLIF_network2/temporal_kernels.pkl
lgn_input_height: 128
lgn_input_width: 128

encoding: identity
encoding_gain: 1.0

# Carry over the calibrated value from CIFAR10-DVS as a starting point;
# DVS128Gesture's event density is similar but spatial extent is larger,
# so the LGN spatial-conv response will scale roughly with H·W. May need
# to lower further (smoke test in Task 10 will tell).
lgn_input_scale: 1.0e-3

dampening_factor: 0.3
gauss_std: 0.5
# H=128 is 7× more pixels than H=48; halve chunk_size to fit 4090 memory.
chunk_size: 32
```

- [ ] **Step 4: Verify the YAMLs parse and num_classes=11 is wired**

Run:
```bash
.venv/bin/python -c "
from utils.load_config import load_config
m = load_config('configs/model_configs/billeh_v1_dvsgesture_lgn.yaml')
d = load_config('configs/data_configs/dvs128gesture_lgn.yaml')
assert m.num_classes == 11 and d.num_classes == 11
assert m.lgn_input_height == d.image_size == 128
assert m.T % d.frames_number == 0
print('[configs] DVS128Gesture configs OK')
"
```
Expected: `[configs] DVS128Gesture configs OK`.

- [ ] **Step 5: Commit**

```bash
git add configs/data_configs/dvs128gesture_lgn.yaml configs/model_configs/billeh_v1_dvsgesture_lgn.yaml
git commit -m "$(cat <<'EOF'
[DVS]: add DVS128Gesture configs for billeh-v1 LGN pipeline

11-class config exercises the localized-readout pool fix (offset=0). Note
in data config that DvsGesture.tar.gz must be manually placed under
<root>/download/ before first run — spikingjelly cannot auto-download.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Smoke test DVS128Gesture and calibrate scale

**Files:**
- None (uses Task 7's smoke script and Task 9's configs)

- [ ] **Step 1: Run the smoke test**

```bash
.venv/bin/python scripts/smoke_billeh_v1_dvs.py \
    --model_config configs/model_configs/billeh_v1_dvsgesture_lgn.yaml \
    --data_config configs/data_configs/dvs128gesture_lgn.yaml \
    --batch 2 2>&1 | tail -40
```

First run: spikingjelly extracts `DvsGesture.tar.gz` and integrates frames (10-30 minutes). Subsequent runs: cached.

- [ ] **Step 2: Read rate, calibrate `lgn_input_scale` if needed**

Same logic as Task 7 Step 3. DVS128Gesture's gestures fill more of the visual field than CIFAR10-DVS objects, so the LGN spatial-conv response is typically larger; the calibrated scale will likely be smaller than CIFAR10-DVS's. Iterate to land rate in [5, 30] Hz.

Also verify `[smoke] non-zero grad params` line shows the localized-readout layer with non-zero gradient (the 11-class fix exercise).

- [ ] **Step 3: Commit any scale change**

```bash
git add configs/model_configs/billeh_v1_dvsgesture_lgn.yaml
git commit -m "$(cat <<'EOF'
[DVS]: calibrate dvsgesture lgn_input_scale from smoke test

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Optional — launch single-epoch DVS128Gesture training**

```bash
bash train.sh billeh_v1_dvsgesture_lgn dvs128gesture_lgn billeh_seqcifar10
```
Watch for the same signals as Task 8 Step 2.

---

## Self-Review Notes

After writing this plan, reviewed against the spec for coverage:

- ✅ `build_dvs_lgn_transform` (spec §1) → Task 2
- ✅ `build_event_dataset` dispatch (spec §2) → Task 3
- ✅ K-replay in `_to_b_t_n_via_lgn` (spec §3) → Task 4
- ✅ 11-class readout fix (spec §4) → Task 5
- ✅ CIFAR10-DVS configs (spec §5) → Task 6
- ✅ DVS128Gesture configs (spec §5) → Task 9
- ✅ Smoke test pipeline (spec §validation plan) → Tasks 7, 10
- ✅ Calibration order (spec §calibration) → Tasks 7, 8, 10

Open items deliberately deferred (matches spec §out-of-scope):
- No augmentation (random temporal crop, flips). Add later if baseline accuracy stalls.
- No DVS-specific lr/scheduler tuning. Reuses `billeh_seqcifar10.yaml` as spec mandates.

Tests required by TDD discipline are scripted (no pytest harness in the repo); each test is runnable as `python scripts/test_*.py` and uses `assert` statements.
