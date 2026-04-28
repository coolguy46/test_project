from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit("PyTorch is required because this script reuses the repo metric utilities.") from exc

from pdsi.data.arrays import load_npz_splits
from pdsi.training.metrics import classification_metrics


def _build_classifier(name: str, seed: int, n_jobs: int):
    try:
        from aeon.classification.convolution_based import (
            HydraClassifier,
            MiniRocketClassifier,
            MultiRocketClassifier,
            MultiRocketHydraClassifier,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit("Install aeon first: python -m pip install 'pdsi[baselines]'") from exc

    name = name.lower()
    if name == "minirocket":
        return MiniRocketClassifier(n_kernels=10000, n_jobs=n_jobs, random_state=seed)
    if name == "multirocket":
        return MultiRocketClassifier(n_kernels=10000, n_jobs=n_jobs, random_state=seed)
    if name == "hydra":
        return HydraClassifier(n_kernels=8, n_groups=64, n_jobs=n_jobs, random_state=seed)
    if name in {"multirocket_hydra", "multirockethydra"}:
        return MultiRocketHydraClassifier(n_kernels=10000, n_jobs=n_jobs, random_state=seed)
    raise ValueError(f"Unknown aeon baseline: {name}")


def _dataset_to_numpy(dataset) -> tuple[np.ndarray, np.ndarray]:
    x_t, y_t = dataset.tensors
    return x_t.numpy().astype(np.float32, copy=False), y_t.numpy()


def _predict_scores(clf, x: np.ndarray, n_classes: int) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(x)
        if proba.ndim == 1:
            proba = np.stack([1.0 - proba, proba], axis=1)
        return np.clip(proba, 1e-6, 1.0)
    pred = clf.predict(x)
    scores = np.full((len(pred), n_classes), 1e-6, dtype=np.float32)
    scores[np.arange(len(pred)), pred.astype(int)] = 1.0
    return scores


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _fit_predict_single_label(name: str, x_train, y_train, x_eval, seed: int, n_jobs: int) -> np.ndarray:
    clf = _build_classifier(name, seed, n_jobs)
    clf.fit(x_train, y_train.astype(int))
    n_classes = int(np.max(y_train)) + 1
    proba = _predict_scores(clf, x_eval, n_classes)
    return np.log(proba)


def _fit_predict_multilabel(name: str, x_train, y_train, x_eval, seed: int, n_jobs: int) -> np.ndarray:
    columns = []
    for label_idx in range(y_train.shape[1]):
        clf = _build_classifier(name, seed + label_idx, n_jobs)
        y_col = y_train[:, label_idx].astype(int)
        if len(np.unique(y_col)) < 2:
            p = np.full(len(x_eval), float(y_col[0]), dtype=np.float32)
        else:
            clf.fit(x_train, y_col)
            proba = _predict_scores(clf, x_eval, 2)
            p = proba[:, 1]
        columns.append(_logit(p))
    return np.stack(columns, axis=1)


def run_one(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    datasets = load_npz_splits(
        args.dataset,
        channels_last=args.channels_last,
        low_label_fraction=args.low_label_fraction,
        seed=seed,
    )
    x_train, y_train = _dataset_to_numpy(datasets["train"])
    x_val, y_val = _dataset_to_numpy(datasets["val"])
    x_test, y_test = _dataset_to_numpy(datasets["test"])

    start = time.perf_counter()
    if y_train.ndim == 1:
        val_logits = _fit_predict_single_label(args.method, x_train, y_train, x_val, seed, args.n_jobs)
        test_logits = _fit_predict_single_label(args.method, x_train, y_train, x_test, seed, args.n_jobs)
        multilabel = False
    else:
        val_logits = _fit_predict_multilabel(args.method, x_train, y_train, x_val, seed, args.n_jobs)
        test_logits = _fit_predict_multilabel(args.method, x_train, y_train, x_test, seed, args.n_jobs)
        multilabel = True
    elapsed = time.perf_counter() - start

    val_metrics = classification_metrics(torch.tensor(val_logits), torch.tensor(y_val), multilabel=multilabel)
    test_metrics = classification_metrics(torch.tensor(test_logits), torch.tensor(y_test), multilabel=multilabel)
    return {
        "seed": seed,
        "method": args.method,
        "best_epoch": 0,
        "best_val_macro_f1": val_metrics["macro_f1"],
        "val": val_metrics,
        "test": test_metrics,
        "wall_clock_sec": elapsed,
        "peak_memory_mb": 0.0,
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = sorted(records[0]["test"].keys())
    out: dict[str, Any] = {"n_seeds": len(records), "seeds": [r["seed"] for r in records]}
    for metric in metrics:
        vals = [float(r["test"][metric]) for r in records]
        out[f"test_{metric}_mean"] = mean(vals)
        out[f"test_{metric}_std"] = stdev(vals) if len(vals) > 1 else 0.0
    out["wall_clock_sec_total"] = sum(float(r["wall_clock_sec"]) for r in records)
    out["wall_clock_sec_mean"] = mean(float(r["wall_clock_sec"]) for r in records)
    out["peak_memory_mb_max"] = 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run aeon MiniRocket/MultiRocket/Hydra baselines on SETM NPZ files.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--method", required=True, choices=["minirocket", "multirocket", "hydra", "multirocket_hydra"])
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--low-label-fraction", type=float, default=1.0)
    args = parser.parse_args()

    name = args.experiment_name or f"{args.dataset.stem}_{args.method}"
    out_dir = ROOT / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x]
    records = []
    for seed in seeds:
        record = run_one(args, seed)
        records.append(record)
        (out_dir / f"seed_{seed}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"seed={seed} method={args.method} test_macro_f1={record['test']['macro_f1']:.4f}")
    summary = _summary(records)
    summary["method"] = args.method
    summary["dataset"] = str(args.dataset)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
