from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset


@dataclass
class SyntheticConfig:
    n_train: int = 1024
    n_val: int = 256
    n_test: int = 256
    num_channels: int = 3
    seq_len: int = 512
    num_classes: int = 4
    noise_std: float = 0.35
    seed: int = 0


def _make_split(n: int, cfg: SyntheticConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, cfg.seq_len, endpoint=False, dtype=np.float32)
    class_bands = [(3.0, 7.0), (9.0, 15.0), (18.0, 30.0), (36.0, 60.0), (70.0, 95.0)]
    x = np.zeros((n, cfg.num_channels, cfg.seq_len), dtype=np.float32)
    y = np.arange(n, dtype=np.int64) % cfg.num_classes
    rng.shuffle(y)

    for i, label in enumerate(y):
        lo, hi = class_bands[label % len(class_bands)]
        base_freq = rng.uniform(lo, hi)
        nuisance_freq = rng.uniform(1.0, 90.0)
        phase = rng.uniform(0, 2 * np.pi, size=(cfg.num_channels, 1)).astype(np.float32)
        gains = rng.normal(1.0, 0.15, size=(cfg.num_channels, 1)).astype(np.float32)
        signal = gains * np.sin(2 * np.pi * base_freq * t[None, :] + phase)
        signal += 0.25 * np.sin(2 * np.pi * (base_freq / 2.0) * t[None, :] + 0.5 * phase)
        signal += 0.12 * np.sin(2 * np.pi * nuisance_freq * t[None, :] + phase[::-1])

        center = rng.uniform(0.25, 0.75)
        width = rng.uniform(0.015, 0.04)
        transient = np.exp(-0.5 * ((t - center) / width) ** 2).astype(np.float32)
        polarity = -1.0 if label % 2 else 1.0
        signal += polarity * 0.35 * transient[None, :]

        baseline = rng.normal(0.0, 0.08, size=(cfg.num_channels, 1)).astype(np.float32)
        drift = baseline * np.sin(2 * np.pi * rng.uniform(0.2, 1.0) * t[None, :])
        noise = rng.normal(0.0, cfg.noise_std, size=signal.shape).astype(np.float32)
        x[i] = signal + drift + noise

    return x, y


def make_synthetic_splits(cfg: SyntheticConfig) -> dict[str, TensorDataset]:
    rng = np.random.default_rng(cfg.seed)
    splits = {}
    for name, n in {"train": cfg.n_train, "val": cfg.n_val, "test": cfg.n_test}.items():
        x, y = _make_split(n, cfg, rng)
        splits[name] = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return splits
