# Phase-Decoupled Spectral Inception

This repository contains a compact research scaffold for **PDSI**, a phase-preserving spectral gate placed before an InceptionTime-style classifier for biomedical time-series tasks.

The code is designed for fast ablations on a single AMD MI300X-class accelerator. The included synthetic benchmark is only a smoke test; the paper plan in [PROPOSAL.md](PROPOSAL.md) specifies the intended CHB-MIT, PTB-XL, and optional Sleep-EDF experiments.

## Setup

Use a ROCm-enabled PyTorch build on the MI300X machine. A typical environment is:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install numpy
# Install the current ROCm PyTorch wheel for your driver stack:
# https://pytorch.org/get-started/locally/
```

For analysis plots and raw PhysioNet preprocessing, install the optional packages:

```powershell
pip install matplotlib pandas mne wfdb
```

## Quick Smoke Run

```powershell
python scripts\run_experiment.py --config configs\synthetic_pdsi_smoke.json
python scripts\run_experiment.py --config configs\synthetic_baseline_smoke.json
```

Outputs are written to `runs/<experiment_name>/` as per-seed JSON plus `summary.json`.

## Real-Data Input Format

For CHB-MIT, PTB-XL, or Sleep-EDF, preprocess signals into an `.npz` file with either:

```text
x_train, y_train, x_val, y_val, x_test, y_test
```

or:

```text
x, y, split
```

where `x` has shape `(N, C, T)` by default, `y` is class indices `(N,)` for multiclass or binary matrix `(N, K)` for multilabel, and `split` contains `train`, `val`, or `test` strings.

Then copy `configs/npz_pdsi_template.json`, set `dataset.path`, `data.num_channels`, `data.seq_len`, and `model.num_classes`, and run:

```powershell
python scripts\run_experiment.py --config configs\npz_pdsi_template.json
```

Optional converters are included for the two primary benchmarks:

```powershell
python scripts\preprocess_ptbxl.py --root C:\path\to\ptb-xl --out data\ptbxl_superclass.npz --sampling-rate 100
python scripts\preprocess_chbmit.py --root C:\path\to\chbmit --out data\chbmit_windows.npz
```

## Main Ablations

Set `model.gate` in a config to:

- `none`: InceptionTime baseline.
- `pdsi`: bounded phase-preserving adaptive spectral gate.
- `complex`: complex spectral multiplier that can alter phase.
- `butterworth`: fixed DSP-style spectral bandpass control.

Aggregate significance across matched seeds:

```powershell
python scripts\summarize_results.py --primary runs\pdsi\summary.json --baseline runs\baseline\summary.json
```

## Code Map

- [src/pdsi/models/gates.py](src/pdsi/models/gates.py): spectral gate variants.
- [src/pdsi/models/inception.py](src/pdsi/models/inception.py): InceptionTime-style classifier.
- [src/pdsi/data/synthetic.py](src/pdsi/data/synthetic.py): synthetic oscillatory benchmark.
- [src/pdsi/data/arrays.py](src/pdsi/data/arrays.py): generic `.npz` loader.
- [src/pdsi/training/trainer.py](src/pdsi/training/trainer.py): training/evaluation loop.
- [scripts/run_experiment.py](scripts/run_experiment.py): experiment CLI.
- [scripts/profile_model.py](scripts/profile_model.py): parameter/FLOP/memory estimator.
- [scripts/preprocess_ptbxl.py](scripts/preprocess_ptbxl.py): optional PTB-XL converter.
- [scripts/preprocess_chbmit.py](scripts/preprocess_chbmit.py): optional CHB-MIT converter.
- [PROPOSAL.md](PROPOSAL.md): NeurIPS-style project plan and skeptical review.
