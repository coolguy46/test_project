from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from statistics import mean, stdev


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_records(summary_path: Path) -> list[dict]:
    records = []
    for seed in _load(summary_path)["seeds"]:
        records.append(_load(summary_path.parent / f"seed_{seed}.json"))
    return records


def _exact_sign_flip_pvalue(diffs: list[float]) -> float:
    observed = abs(mean(diffs))
    count = 0
    extreme = 0
    for signs in itertools.product([-1, 1], repeat=len(diffs)):
        value = abs(mean([s * d for s, d in zip(signs, diffs)]))
        extreme += int(value >= observed - 1e-12)
        count += 1
    return extreme / count


def _report(primary: list[dict], baseline: list[dict] | None, metric: str) -> dict:
    vals = [float(r["test"][metric]) for r in primary]
    out = {
        "metric": metric,
        "mean": mean(vals),
        "std": stdev(vals) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }
    if baseline is not None:
        by_seed = {r["seed"]: r for r in baseline}
        diffs = [float(r["test"][metric]) - float(by_seed[r["seed"]]["test"][metric]) for r in primary]
        out.update(
            {
                "paired_delta_mean": mean(diffs),
                "paired_delta_std": stdev(diffs) if len(diffs) > 1 else 0.0,
                "paired_sign_flip_p": _exact_sign_flip_pvalue(diffs),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    primary = _seed_records(args.primary)
    baseline = _seed_records(args.baseline) if args.baseline else None
    metrics = sorted(primary[0]["test"].keys())
    reports = [_report(primary, baseline, metric) for metric in metrics]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
