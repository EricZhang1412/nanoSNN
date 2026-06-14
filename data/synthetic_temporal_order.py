from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


class SyntheticTemporalOrderDataset(Dataset):
    """
    Two-cue temporal-order task.

    Each sample is a movie of shape [T, C=2, H, W].
    A 'cue-A' blob appears at one polarity channel in one timestep, and a
    'cue-B' blob at the other polarity in another timestep, at distinct
    spatial positions. Label = 0 if A precedes B, 1 if B precedes A.

    Important properties:
      - Shuffling frames destroys the label (used for Block B / gate 2).
      - Static-frame average has identical class statistics (no shortcut).
      - Cue presence/count is identical across classes.
    """

    def __init__(
        self,
        n_samples: int = 4096,
        T: int = 8,
        H: int = 32,
        W: int = 32,
        blob_size: int = 4,
        min_gap: int = 1,
        seed: int = 0,
        # Optional perturbations for Block B counterfactual evaluation.
        shuffle_time: bool = False,
        reverse_time: bool = False,
        first_last_only: bool = False,
    ):
        super().__init__()
        self.n_samples = int(n_samples)
        self.T = int(T)
        self.H = int(H)
        self.W = int(W)
        self.blob_size = int(blob_size)
        self.min_gap = int(min_gap)
        self.shuffle_time = bool(shuffle_time)
        self.reverse_time = bool(reverse_time)
        self.first_last_only = bool(first_last_only)
        if self.T < 2 * (self.min_gap + 1):
            raise ValueError(f"T={T} too small for min_gap={min_gap}")
        self.classes = ["A_before_B", "B_before_A"]
        self.gen = torch.Generator().manual_seed(seed)

        # Pre-sample every sample's parameters for determinism across epochs.
        self._params = []
        for _ in range(self.n_samples):
            label = int(torch.randint(0, 2, (1,), generator=self.gen).item())
            # Pick two distinct timesteps with order according to label.
            while True:
                t0 = int(torch.randint(0, self.T, (1,), generator=self.gen).item())
                t1 = int(torch.randint(0, self.T, (1,), generator=self.gen).item())
                if abs(t0 - t1) > self.min_gap:
                    break
            t_A, t_B = (min(t0, t1), max(t0, t1)) if label == 0 else (max(t0, t1), min(t0, t1))
            # Pick two distinct positions.
            posA = (int(torch.randint(0, self.H - self.blob_size, (1,), generator=self.gen).item()),
                    int(torch.randint(0, self.W - self.blob_size, (1,), generator=self.gen).item()))
            while True:
                posB = (int(torch.randint(0, self.H - self.blob_size, (1,), generator=self.gen).item()),
                        int(torch.randint(0, self.W - self.blob_size, (1,), generator=self.gen).item()))
                if abs(posB[0] - posA[0]) + abs(posB[1] - posA[1]) >= self.blob_size:
                    break
            self._params.append((label, t_A, t_B, posA, posB))

    def __len__(self):
        return self.n_samples

    def _draw_blob(self, frame: torch.Tensor, ch: int, pos):
        y, x = pos
        b = self.blob_size
        frame[ch, y:y + b, x:x + b] = 1.0

    def __getitem__(self, idx):
        label, t_A, t_B, posA, posB = self._params[idx]
        movie = torch.zeros(self.T, 2, self.H, self.W, dtype=torch.float32)
        # Cue A in channel 0, cue B in channel 1.
        self._draw_blob(movie[t_A], ch=0, pos=posA)
        self._draw_blob(movie[t_B], ch=1, pos=posB)

        if self.first_last_only:
            keep = torch.zeros(self.T, dtype=torch.bool)
            keep[0] = True
            keep[-1] = True
            movie = movie * keep.view(-1, 1, 1, 1)
        if self.reverse_time:
            movie = movie.flip(0)
        if self.shuffle_time:
            perm = torch.randperm(self.T)
            movie = movie[perm]

        return movie, int(label)


def build_synthetic_temporal_order(data_config, split: str) -> SyntheticTemporalOrderDataset:
    n_train = int(getattr(data_config, "n_train", 4096))
    n_val = int(getattr(data_config, "n_val", 512))
    n_test = int(getattr(data_config, "n_test", 1024))
    T = int(getattr(data_config, "T", 8))
    H = int(getattr(data_config, "image_size", 32))
    W = int(getattr(data_config, "image_size", 32))
    blob_size = int(getattr(data_config, "blob_size", 4))
    min_gap = int(getattr(data_config, "min_gap", 1))
    base_seed = int(getattr(data_config, "split_seed", 42))

    shuffle_time = bool(getattr(data_config, "shuffle_time", False))
    reverse_time = bool(getattr(data_config, "reverse_time", False))
    first_last_only = bool(getattr(data_config, "first_last_only", False))

    if split == "train":
        n, seed = n_train, base_seed
    elif split == "val":
        n, seed = n_val, base_seed + 1
    else:
        n, seed = n_test, base_seed + 2

    return SyntheticTemporalOrderDataset(
        n_samples=n, T=T, H=H, W=W,
        blob_size=blob_size, min_gap=min_gap, seed=seed,
        shuffle_time=shuffle_time, reverse_time=reverse_time,
        first_last_only=first_last_only,
    )


@dataclass(frozen=True)
class TemporalOrderV2Sample:
    """Pre-sampled metadata for one synthetic-v2 movie."""

    label: int
    target_events: tuple[tuple[str, int], ...]      # (symbol, time)
    distractors: tuple[tuple[int, int, tuple[int, int]], ...]  # (time, channel, pos)


class SyntheticTemporalOrderV2Dataset(Dataset):
    """
    Harder temporal-order benchmark for DyCo-SNN Block A.

    v1 only asks whether one A cue precedes one B cue. v2 keeps the same
    controllable binary target but adds pressure sources that should separate
    models before the task saturates:
      - longer movies (T=32/64 via config);
      - target cues at stable spatial anchors, with optional jitter;
      - same-channel and cross-channel distractor blobs, many inside the
        A/B interval;
      - optional multi-event grammar over A/B/C;
      - random background events controlled by noise_event_rate;
      - small train splits (e.g. 512/1024) for sample-efficiency tests.

    Labels are balanced by construction and depend only on temporal order.
    Static cue counts and target cue identities are label-invariant.
    """

    TARGET_PAIR_ORDER = "target_pair_order"
    THREE_EVENT_GRAMMAR = "three_event_grammar"

    def __init__(
        self,
        n_samples: int = 1024,
        T: int = 32,
        H: int = 32,
        W: int = 32,
        blob_size: int = 4,
        seed: int = 0,
        grammar: str = TARGET_PAIR_ORDER,
        time_margin: int = 1,
        target_gap_min: int = 8,
        target_gap_max: int = 0,
        target_jitter: int = 0,
        n_distractors: int = 12,
        n_between_distractors: int = 8,
        distractor_min_anchor_distance: int = 8,
        noise_event_rate: float = 0.0,
        noise_value: float = 1.0,
        shuffle_time: bool = False,
        reverse_time: bool = False,
        first_last_only: bool = False,
    ):
        super().__init__()
        self.n_samples = int(n_samples)
        self.T = int(T)
        self.H = int(H)
        self.W = int(W)
        self.blob_size = int(blob_size)
        self.seed = int(seed)
        self.grammar = str(grammar)
        self.time_margin = int(time_margin)
        self.target_gap_min = int(target_gap_min)
        self.target_gap_max = int(target_gap_max)
        self.target_jitter = int(target_jitter)
        self.n_distractors = int(n_distractors)
        self.n_between_distractors = int(n_between_distractors)
        self.distractor_min_anchor_distance = int(distractor_min_anchor_distance)
        self.noise_event_rate = float(noise_event_rate)
        self.noise_value = float(noise_value)
        self.shuffle_time = bool(shuffle_time)
        self.reverse_time = bool(reverse_time)
        self.first_last_only = bool(first_last_only)
        self.classes = ["class_0", "class_1"]

        if self.grammar not in {self.TARGET_PAIR_ORDER, self.THREE_EVENT_GRAMMAR}:
            raise ValueError(f"Unknown grammar={self.grammar!r}")
        if self.H < self.blob_size or self.W < self.blob_size:
            raise ValueError("image_size must be >= blob_size")
        if self.T <= 2 * self.time_margin + 1:
            raise ValueError(f"T={self.T} too small for time_margin={self.time_margin}")

        # Anchors make target identities observable even when distractors use
        # the same polarity channel. Values are top-left blob coordinates.
        y0, y1 = self.H // 4 - self.blob_size // 2, 3 * self.H // 4 - self.blob_size // 2
        x0, x1 = self.W // 4 - self.blob_size // 2, 3 * self.W // 4 - self.blob_size // 2
        self.symbol_channels = {"A": 0, "B": 1, "C": 0, "D": 1}
        self.symbol_anchors = {
            "A": self._clamp_pos((y0, x0)),
            "B": self._clamp_pos((y1, x1)),
            "C": self._clamp_pos((y0, x1)),
            "D": self._clamp_pos((y1, x0)),
        }

        gen = torch.Generator().manual_seed(self.seed)
        labels = [i % 2 for i in range(self.n_samples)]
        labels = [labels[i] for i in torch.randperm(self.n_samples, generator=gen).tolist()]
        self._params = tuple(self._make_sample_params(gen, label) for label in labels)

    def __len__(self):
        return self.n_samples

    # ---------- parameter sampling ----------

    def _valid_time_low_high(self) -> tuple[int, int]:
        low = self.time_margin
        high = self.T - 1 - self.time_margin
        return low, high

    def _randint(self, gen: torch.Generator, low: int, high_inclusive: int) -> int:
        if high_inclusive < low:
            raise ValueError(f"empty randint range [{low}, {high_inclusive}]")
        return int(torch.randint(low, high_inclusive + 1, (1,), generator=gen).item())

    def _sample_two_times(self, gen: torch.Generator) -> tuple[int, int]:
        low, high = self._valid_time_low_high()
        max_gap = self.target_gap_max if self.target_gap_max > 0 else self.T
        for _ in range(1000):
            t0 = self._randint(gen, low, high)
            t1 = self._randint(gen, low, high)
            gap = abs(t1 - t0)
            if self.target_gap_min <= gap <= max_gap:
                return min(t0, t1), max(t0, t1)
        raise ValueError(
            "Could not sample target times. Lower target_gap_min/target_gap_max "
            f"or increase T. T={self.T}, margin={self.time_margin}"
        )

    def _sample_ordered_times(self, gen: torch.Generator, n: int) -> tuple[int, ...]:
        low, high = self._valid_time_low_high()
        for _ in range(1000):
            vals = sorted({self._randint(gen, low, high) for _ in range(n * 4)})
            if len(vals) < n:
                continue
            # Prefer a spread-out subsequence so the grammar is not solved by
            # adjacent-frame artifacts.
            for start in range(0, len(vals) - n + 1):
                cand = vals[start:start + n]
                gaps = [cand[i + 1] - cand[i] for i in range(n - 1)]
                if all(g >= self.target_gap_min for g in gaps):
                    return tuple(cand)
        raise ValueError(
            "Could not sample grammar times. Lower target_gap_min or increase T."
        )

    def _make_sample_params(self, gen: torch.Generator, label: int) -> TemporalOrderV2Sample:
        if self.grammar == self.TARGET_PAIR_ORDER:
            early, late = self._sample_two_times(gen)
            target_events = (("A", early), ("B", late)) if label == 0 else (("A", late), ("B", early))
            span = (early, late)
        else:
            t0, t1, t2 = self._sample_ordered_times(gen, 3)
            # class_0 implements "A before B after C": C < A < B.
            # class_1 is a reversed foil: B < A < C.
            target_events = (("C", t0), ("A", t1), ("B", t2)) if label == 0 else (
                ("B", t0), ("A", t1), ("C", t2)
            )
            span = (t0, t2)

        distractors = self._sample_distractors(gen, span)
        return TemporalOrderV2Sample(label=label, target_events=target_events, distractors=distractors)

    def _sample_distractors(
        self, gen: torch.Generator, target_span: tuple[int, int]
    ) -> tuple[tuple[int, int, tuple[int, int]], ...]:
        if self.n_distractors <= 0:
            return tuple()

        low, high = self._valid_time_low_high()
        inner_low, inner_high = target_span[0] + 1, target_span[1] - 1
        n_between = max(0, min(self.n_between_distractors, self.n_distractors))

        channels = [i % 2 for i in range(self.n_distractors)]
        perm = torch.randperm(self.n_distractors, generator=gen).tolist()
        channels = [channels[i] for i in perm]

        out = []
        for i, ch in enumerate(channels):
            if i < n_between and inner_low <= inner_high:
                t = self._randint(gen, inner_low, inner_high)
            else:
                t = self._randint(gen, low, high)
            out.append((t, ch, self._sample_distractor_pos(gen)))
        return tuple(out)

    # ---------- drawing ----------

    def _clamp_pos(self, pos: tuple[int, int]) -> tuple[int, int]:
        y, x = pos
        y = max(0, min(int(y), self.H - self.blob_size))
        x = max(0, min(int(x), self.W - self.blob_size))
        return y, x

    def _target_pos(self, symbol: str, gen: torch.Generator) -> tuple[int, int]:
        y, x = self.symbol_anchors[symbol]
        if self.target_jitter <= 0:
            return y, x
        dy = self._randint(gen, -self.target_jitter, self.target_jitter)
        dx = self._randint(gen, -self.target_jitter, self.target_jitter)
        return self._clamp_pos((y + dy, x + dx))

    def _sample_distractor_pos(self, gen: torch.Generator) -> tuple[int, int]:
        anchors = tuple(self.symbol_anchors.values())
        max_y = self.H - self.blob_size
        max_x = self.W - self.blob_size
        for _ in range(100):
            pos = (self._randint(gen, 0, max_y), self._randint(gen, 0, max_x))
            if all(abs(pos[0] - a[0]) + abs(pos[1] - a[1]) >= self.distractor_min_anchor_distance
                   for a in anchors):
                return pos
        return (self._randint(gen, 0, max_y), self._randint(gen, 0, max_x))

    def _draw_blob(self, movie: torch.Tensor, t: int, ch: int, pos: tuple[int, int], value: float = 1.0):
        y, x = pos
        b = self.blob_size
        movie[t, ch, y:y + b, x:x + b] = value

    def __getitem__(self, idx):
        params = self._params[idx]
        # Per-index generator keeps noise/jitter/shuffle deterministic across
        # repeated evals and independent of DataLoader worker count.
        gen = torch.Generator().manual_seed(self.seed + int(idx) * 1_000_003)
        movie = torch.zeros(self.T, 2, self.H, self.W, dtype=torch.float32)

        for t, ch, pos in params.distractors:
            self._draw_blob(movie, t=t, ch=ch, pos=pos, value=1.0)

        for symbol, t in params.target_events:
            self._draw_blob(
                movie,
                t=t,
                ch=self.symbol_channels[symbol],
                pos=self._target_pos(symbol, gen),
                value=1.0,
            )

        if self.noise_event_rate > 0.0:
            noise = torch.rand(movie.shape, generator=gen, dtype=movie.dtype) < self.noise_event_rate
            movie = torch.maximum(movie, noise.to(movie.dtype) * self.noise_value)

        if self.first_last_only:
            keep = torch.zeros(self.T, dtype=torch.bool)
            keep[0] = True
            keep[-1] = True
            movie = movie * keep.view(-1, 1, 1, 1)
        if self.reverse_time:
            movie = movie.flip(0)
        if self.shuffle_time:
            movie = movie[torch.randperm(self.T, generator=gen)]

        return movie, int(params.label)


def build_synthetic_temporal_order_v2(data_config, split: str) -> SyntheticTemporalOrderV2Dataset:
    n_train = int(getattr(data_config, "n_train", 1024))
    n_val = int(getattr(data_config, "n_val", 512))
    n_test = int(getattr(data_config, "n_test", 1024))
    base_seed = int(getattr(data_config, "split_seed", 42))

    if split == "train":
        n, seed = n_train, base_seed
    elif split == "val":
        n, seed = n_val, base_seed + 1
    else:
        n, seed = n_test, base_seed + 2

    return SyntheticTemporalOrderV2Dataset(
        n_samples=n,
        T=int(getattr(data_config, "T", 32)),
        H=int(getattr(data_config, "image_size", 32)),
        W=int(getattr(data_config, "image_size", 32)),
        blob_size=int(getattr(data_config, "blob_size", 4)),
        seed=seed,
        grammar=str(getattr(data_config, "grammar", SyntheticTemporalOrderV2Dataset.TARGET_PAIR_ORDER)),
        time_margin=int(getattr(data_config, "time_margin", 1)),
        target_gap_min=int(getattr(data_config, "target_gap_min", 8)),
        target_gap_max=int(getattr(data_config, "target_gap_max", 0)),
        target_jitter=int(getattr(data_config, "target_jitter", 0)),
        n_distractors=int(getattr(data_config, "n_distractors", 12)),
        n_between_distractors=int(getattr(data_config, "n_between_distractors", 8)),
        distractor_min_anchor_distance=int(getattr(data_config, "distractor_min_anchor_distance", 8)),
        noise_event_rate=float(getattr(data_config, "noise_event_rate", 0.0)),
        noise_value=float(getattr(data_config, "noise_value", 1.0)),
        shuffle_time=bool(getattr(data_config, "shuffle_time", False)),
        reverse_time=bool(getattr(data_config, "reverse_time", False)),
        first_last_only=bool(getattr(data_config, "first_last_only", False)),
    )
