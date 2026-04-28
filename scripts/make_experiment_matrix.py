from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def emit(cfg: dict[str, Any], out_dir: Path, name: str, commands: list[str]) -> None:
    cfg["experiment_name"] = name
    path = out_dir / f"{name}.json"
    write_json(path, cfg)
    commands.append(f"python scripts/run_experiment.py --config {path.as_posix()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SETM baseline, ablation, and small sweep configs.")
    parser.add_argument("--template", type=Path, default=Path("configs/npz_setm_template.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("configs/generated"))
    parser.add_argument("--prefix", default="ptbxl")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--seeds", default="", help="Comma-separated seeds, e.g. 0,1,2,3,4")
    parser.add_argument("--include-sweep", action="store_true")
    args = parser.parse_args()

    base = load_json(args.template)
    if args.dataset_path:
        base["dataset"]["path"] = args.dataset_path
    if args.seeds:
        base["seeds"] = [int(x) for x in args.seeds.split(",") if x]

    commands: list[str] = []

    # Main model and nearest causal ablations.
    for router_mode in ["uniform", "static", "direct", "prototype"]:
        cfg = copy.deepcopy(base)
        cfg["model"]["router_mode"] = router_mode
        emit(cfg, args.out_dir, f"{args.prefix}_setm_{router_mode}", commands)

    # Small, defensible SETM hyperparameter sweep. Run on validation only for selection;
    # report final test once with the selected setting and five fixed seeds.
    if args.include_sweep:
        for num_bands in [8, 16, 32]:
            for num_prototypes in [4, 8, 12]:
                for lambda_temporal_smooth in [0.0, 1e-4]:
                    cfg = copy.deepcopy(base)
                    cfg["experiment_name"] = ""
                    cfg["model"]["router_mode"] = "prototype"
                    cfg["model"]["num_bands"] = num_bands
                    cfg["model"]["num_prototypes"] = num_prototypes
                    cfg["train"]["lambda_temporal_smooth"] = lambda_temporal_smooth
                    cfg["seeds"] = [base["seeds"][0]]
                    name = f"{args.prefix}_sweep_b{num_bands}_p{num_prototypes}_ts{lambda_temporal_smooth:g}"
                    emit(cfg, args.out_dir / "sweep", name, commands)

    script_path = args.out_dir / f"run_{args.prefix}_matrix.sh"
    script_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n", encoding="utf-8")
    print(f"Wrote {len(commands)} configs")
    print(f"Run script: {script_path}")


if __name__ == "__main__":
    main()
