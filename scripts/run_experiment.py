from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required to run experiments. Install a ROCm-enabled torch build on the MI300X machine."
    ) from exc

from pdsi.data.arrays import load_npz_splits
from pdsi.data.synthetic import SyntheticConfig, make_synthetic_splits
from pdsi.models.complexity import count_parameters, estimate_forward_flops
from pdsi.models.inception import ModelConfig, PDSIClassifier
from pdsi.training.trainer import TrainConfig, train_model


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_datasets(cfg: dict[str, Any], seed: int):
    dataset_cfg = cfg["dataset"]
    name = dataset_cfg["name"].lower()
    if name == "synthetic":
        syn = SyntheticConfig(**{**dataset_cfg.get("params", {}), "seed": seed})
        return make_synthetic_splits(syn)
    if name == "npz":
        return load_npz_splits(
            dataset_cfg["path"],
            channels_last=dataset_cfg.get("channels_last", False),
            low_label_fraction=dataset_cfg.get("low_label_fraction", 1.0),
            seed=seed,
        )
    raise ValueError(f"Unknown dataset: {name}")


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(records[0]["test"].keys())
    out: dict[str, Any] = {"n_seeds": len(records), "seeds": [r["seed"] for r in records]}
    for name in metric_names:
        vals = [float(r["test"][name]) for r in records]
        out[f"test_{name}_mean"] = mean(vals)
        out[f"test_{name}_std"] = stdev(vals) if len(vals) > 1 else 0.0
    out["wall_clock_sec_total"] = sum(float(r["wall_clock_sec"]) for r in records)
    out["wall_clock_sec_mean"] = mean(float(r["wall_clock_sec"]) for r in records)
    out["peak_memory_mb_max"] = max(float(r["peak_memory_mb"]) for r in records)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    exp_name = cfg.get("experiment_name", args.config.stem)
    out_dir = ROOT / "runs" / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    model_cfg = ModelConfig(**cfg["model"])
    model = PDSIClassifier(model_cfg).to(device)
    profile = estimate_forward_flops(
        model,
        seq_len=cfg["data"]["seq_len"],
        num_channels=cfg["data"]["num_channels"],
        batch_size=cfg.get("profile_batch_size", 1),
    )
    profile["model_config"] = asdict(model_cfg)
    (out_dir / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    if args.profile_only:
        print(json.dumps(profile, indent=2))
        return

    records = []
    for seed in cfg.get("seeds", [0]):
        datasets = _build_datasets(cfg, seed)
        train_cfg = TrainConfig(**{**cfg["train"], "seed": seed, "multilabel": model_cfg.multilabel})
        model = PDSIClassifier(model_cfg)
        record = train_model(model, datasets, train_cfg, device)
        record["params"] = count_parameters(model)
        records.append(record)
        (out_dir / f"seed_{seed}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"seed={seed} test_macro_f1={record['test']['macro_f1']:.4f} sec={record['wall_clock_sec']:.1f}")

    summary = _summary(records)
    summary["profile"] = profile
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
