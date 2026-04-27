from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _binary_auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(bool)
    pos = int(y.sum())
    neg = int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[y].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _binary_auprc(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(bool)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    recall_step = y_sorted.astype(np.float64) / positives
    return float(np.sum(precision * recall_step))


def _macro_f1_multiclass(y: np.ndarray, pred: np.ndarray, num_classes: int) -> float:
    f1s = []
    for cls in range(num_classes):
        tp = np.sum((pred == cls) & (y == cls))
        fp = np.sum((pred == cls) & (y != cls))
        fn = np.sum((pred != cls) & (y == cls))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(f1s))


def _balanced_accuracy(y: np.ndarray, pred: np.ndarray, num_classes: int) -> float:
    recalls = []
    for cls in range(num_classes):
        denom = np.sum(y == cls)
        recalls.append(float("nan") if denom == 0 else np.sum((pred == cls) & (y == cls)) / denom)
    return float(np.nanmean(recalls))


def _macro_f1_multilabel(y: np.ndarray, pred: np.ndarray) -> float:
    f1s = []
    for cls in range(y.shape[1]):
        yt = y[:, cls].astype(bool)
        yp = pred[:, cls].astype(bool)
        tp = np.sum(yt & yp)
        fp = np.sum(~yt & yp)
        fn = np.sum(yt & ~yp)
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(f1s))


def classification_metrics(logits: torch.Tensor, target: torch.Tensor, multilabel: bool = False) -> dict[str, Any]:
    logits_np = logits.detach().cpu().float().numpy()
    target_np = target.detach().cpu().numpy()
    if multilabel:
        probs = _sigmoid(logits_np)
        pred = probs >= 0.5
        y = target_np.astype(np.float32)
        aurocs = [_binary_auroc(y[:, i], probs[:, i]) for i in range(y.shape[1])]
        auprcs = [_binary_auprc(y[:, i], probs[:, i]) for i in range(y.shape[1])]
        return {
            "macro_f1": _macro_f1_multilabel(y, pred),
            "micro_accuracy": float((pred == y.astype(bool)).mean()),
            "macro_auroc": float(np.nanmean(aurocs)),
            "macro_auprc": float(np.nanmean(auprcs)),
        }

    num_classes = logits_np.shape[1]
    probs = _softmax(logits_np)
    pred = probs.argmax(axis=1)
    y = target_np.astype(np.int64)
    aurocs = [_binary_auroc((y == i).astype(int), probs[:, i]) for i in range(num_classes)]
    auprcs = [_binary_auprc((y == i).astype(int), probs[:, i]) for i in range(num_classes)]
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": _balanced_accuracy(y, pred, num_classes),
        "macro_f1": _macro_f1_multiclass(y, pred, num_classes),
        "macro_auroc": float(np.nanmean(aurocs)),
        "macro_auprc": float(np.nanmean(auprcs)),
    }
