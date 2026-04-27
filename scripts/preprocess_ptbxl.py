from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np


def _imports():
    try:
        import pandas as pd
        import wfdb
    except ModuleNotFoundError as exc:
        raise SystemExit("Install optional dependencies first: pip install pandas wfdb") from exc
    return pd, wfdb


def _labels(metadata, scp_statements, classes: list[str]) -> np.ndarray:
    diagnostic = scp_statements[scp_statements["diagnostic"] == 1]
    class_to_idx = {name: i for i, name in enumerate(classes)}
    y = np.zeros((len(metadata), len(classes)), dtype=np.float32)
    for row_idx, codes_text in enumerate(metadata["scp_codes"]):
        codes = ast.literal_eval(codes_text)
        for code in codes:
            if code not in diagnostic.index:
                continue
            cls = diagnostic.loc[code, "diagnostic_class"]
            if cls in class_to_idx:
                y[row_idx, class_to_idx[cls]] = 1.0
    return y


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PTB-XL into the PDSI NPZ format.")
    parser.add_argument("--root", type=Path, required=True, help="PTB-XL root containing ptbxl_database.csv.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sampling-rate", type=int, choices=[100, 500], default=100)
    parser.add_argument("--classes", nargs="+", default=["NORM", "MI", "STTC", "CD", "HYP"])
    args = parser.parse_args()
    pd, wfdb = _imports()

    metadata = pd.read_csv(args.root / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(args.root / "scp_statements.csv", index_col=0)
    y = _labels(metadata, scp, args.classes)
    file_col = "filename_lr" if args.sampling_rate == 100 else "filename_hr"

    xs = []
    for filename in metadata[file_col]:
        signal, _ = wfdb.rdsamp(str(args.root / filename))
        xs.append(signal.T.astype(np.float32, copy=False))
    x = np.stack(xs, axis=0)

    folds = metadata["strat_fold"].to_numpy()
    train = folds <= 8
    val = folds == 9
    test = folds == 10
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        x_train=x[train],
        y_train=y[train],
        x_val=x[val],
        y_val=y[val],
        x_test=x[test],
        y_test=y[test],
        classes=np.array(args.classes),
        sampling_rate=args.sampling_rate,
    )
    print(f"saved {args.out} with x={x.shape}, classes={args.classes}")


if __name__ == "__main__":
    main()
