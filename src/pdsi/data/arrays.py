from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import TensorDataset


def _as_channel_first(x: np.ndarray, channels_last: bool = False) -> np.ndarray:
    if x.ndim == 2:
        x = x[:, None, :]
    elif channels_last:
        x = np.transpose(x, (0, 2, 1))
    if x.ndim != 3:
        raise ValueError(f"Expected x with 2 or 3 dimensions, got {x.shape}")
    return x.astype(np.float32, copy=False)


def _dataset(x: np.ndarray, y: np.ndarray, channels_last: bool) -> TensorDataset:
    x = _as_channel_first(x, channels_last=channels_last)
    if y.ndim == 1:
        y_t = torch.from_numpy(y.astype(np.int64, copy=False))
    else:
        y_t = torch.from_numpy(y.astype(np.float32, copy=False))
    return TensorDataset(torch.from_numpy(x), y_t)


def _stratified_take(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if fraction >= 1.0:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    if y.ndim != 1:
        scores = y.argmax(axis=1)
    else:
        scores = y
    keep = []
    for cls in np.unique(scores):
        idx = np.flatnonzero(scores == cls)
        n = max(1, int(round(len(idx) * fraction)))
        keep.append(rng.choice(idx, size=n, replace=False))
    out = np.concatenate(keep)
    rng.shuffle(out)
    return out


def load_npz_splits(
    path: str | Path,
    channels_last: bool = False,
    low_label_fraction: float = 1.0,
    seed: int = 0,
) -> dict[str, TensorDataset]:
    data: dict[str, Any] = dict(np.load(path, allow_pickle=True))
    if {"x_train", "y_train", "x_val", "y_val", "x_test", "y_test"}.issubset(data):
        x_train, y_train = data["x_train"], data["y_train"]
        keep = _stratified_take(y_train, low_label_fraction, seed)
        return {
            "train": _dataset(x_train[keep], y_train[keep], channels_last),
            "val": _dataset(data["x_val"], data["y_val"], channels_last),
            "test": _dataset(data["x_test"], data["y_test"], channels_last),
        }
    if {"x", "y", "split"}.issubset(data):
        split = data["split"].astype(str)
        x, y = data["x"], data["y"]
        train_idx = np.flatnonzero(split == "train")
        keep = train_idx[_stratified_take(y[train_idx], low_label_fraction, seed)]
        return {
            "train": _dataset(x[keep], y[keep], channels_last),
            "val": _dataset(x[split == "val"], y[split == "val"], channels_last),
            "test": _dataset(x[split == "test"], y[split == "test"], channels_last),
        }
    raise ValueError("NPZ must contain either train/val/test arrays or x, y, split arrays")
