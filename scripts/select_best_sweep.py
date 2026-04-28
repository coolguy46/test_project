from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank sweep runs by best validation Macro-F1.")
    parser.add_argument("--runs-dir", type=Path, required=True, help="Directory containing sweep run folders.")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    rows = []
    for summary in args.runs_dir.glob("*/summary.json"):
        cfg_path = summary.parent / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        best_vals = []
        for seed_path in summary.parent.glob("seed_*.json"):
            record = json.loads(seed_path.read_text(encoding="utf-8"))
            best_vals.append(float(record["best_val_macro_f1"]))
        if not best_vals:
            continue
        rows.append(
            {
                "run": summary.parent.name,
                "mean_best_val_macro_f1": sum(best_vals) / len(best_vals),
                "num_bands": cfg["model"].get("num_bands"),
                "num_prototypes": cfg["model"].get("num_prototypes"),
                "lambda_temporal_smooth": cfg["train"].get("lambda_temporal_smooth"),
                "config": str(cfg_path),
            }
        )

    rows.sort(key=lambda x: x["mean_best_val_macro_f1"], reverse=True)
    print(json.dumps(rows[: args.top_k], indent=2))


if __name__ == "__main__":
    main()
