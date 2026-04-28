# Spectral Episode Temporal Mixer

This repository contains a compact research scaffold for **SETM**, a fast biomedical time-series classifier that combines:

- a tiny FFT-based **spectral summary**,
- a learned library of **spectral prototypes**,
- and a stack of **routed temporal mixer blocks** with sample-adaptive expert selection.

The core idea is intentionally narrow. SETM does **not** filter the input waveform and does **not** rely on an Inception-style backbone. Instead, it uses coarse spectral context to decide which temporal receptive fields to emphasize for each sample. The intended paper claim is modest: spectral prototype routing can improve the quality/efficiency tradeoff of a lightweight temporal mixer under matched training budgets.

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
python scripts\run_experiment.py --config configs\synthetic_uniform_smoke.json
python scripts\run_experiment.py --config configs\synthetic_setm_smoke.json
```

Outputs are written to `runs/<experiment_name>/` as per-seed JSON plus `summary.json`.

## Full Controlled Pipeline

After linking or downloading PTB-XL, CHB-MIT, and optional Sleep-EDF, run:

```powershell
bash scripts\run_full_neurips_pipeline.sh
```

This executes preprocessing, smoke tests, SETM router ablations, a small validation sweep, result comparisons, and profiling. Strong external baselines such as MiniRocket, Hydra, and upstream neural models should still be run from their maintained implementations and merged into the final paper table.

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

Then copy `configs/npz_setm_template.json`, set `dataset.path`, `data.num_channels`, `data.seq_len`, and `model.num_classes`, and run:

```powershell
python scripts\run_experiment.py --config configs\npz_setm_template.json
```

Optional converters are included for the main benchmarks:

```powershell
python scripts\preprocess_ptbxl.py --root C:\path\to\ptb-xl --out data\ptbxl_superclass.npz --sampling-rate 100
python scripts\preprocess_chbmit.py --root C:\path\to\chbmit --out data\chbmit_windows.npz
python scripts\preprocess_sleepedf.py --root C:\path\to\sleep-edf --out data\sleepedf_epochs.npz
```

## Main Ablations

Set `model.router_mode` in a config to:

- `uniform`: average temporal experts uniformly; no sample adaptation.
- `static`: one learned route schedule shared by the whole dataset.
- `direct`: direct MLP routing from spectral summary, no prototype library.
- `prototype`: full SETM with learned spectral prototypes and routed expert schedule.

Aggregate significance across matched seeds:

```powershell
python scripts\summarize_results.py --primary runs\ptbxl_setm_prototype\summary.json --baseline runs\ptbxl_setm_uniform\summary.json
```

## Code Map

- [src/pdsi/models/setm.py](C:/Users/ankey/True_Project/src/pdsi/models/setm.py): SETM model, spectral router, and routed temporal mixer blocks.
- [src/pdsi/data/synthetic.py](C:/Users/ankey/True_Project/src/pdsi/data/synthetic.py): synthetic oscillatory benchmark.
- [src/pdsi/data/arrays.py](C:/Users/ankey/True_Project/src/pdsi/data/arrays.py): generic `.npz` loader.
- [src/pdsi/training/trainer.py](C:/Users/ankey/True_Project/src/pdsi/training/trainer.py): training/evaluation loop.
- [scripts/run_experiment.py](C:/Users/ankey/True_Project/scripts/run_experiment.py): experiment CLI.
- [scripts/profile_model.py](C:/Users/ankey/True_Project/scripts/profile_model.py): parameter/FLOP estimator.
- [scripts/make_experiment_matrix.py](C:/Users/ankey/True_Project/scripts/make_experiment_matrix.py): creates router ablation and sweep configs.
- [scripts/run_aeon_baseline.py](C:/Users/ankey/True_Project/scripts/run_aeon_baseline.py): MiniRocket, MultiRocket, Hydra, and MultiRocket-Hydra baselines.
- [scripts/preprocess_ptbxl.py](C:/Users/ankey/True_Project/scripts/preprocess_ptbxl.py): optional PTB-XL converter.
- [scripts/preprocess_chbmit.py](C:/Users/ankey/True_Project/scripts/preprocess_chbmit.py): optional CHB-MIT converter.
- [scripts/preprocess_sleepedf.py](C:/Users/ankey/True_Project/scripts/preprocess_sleepedf.py): optional Sleep-EDF converter.
- [scripts/run_full_neurips_pipeline.sh](C:/Users/ankey/True_Project/scripts/run_full_neurips_pipeline.sh): full controlled experiment pipeline.
- [BASELINES.md](C:/Users/ankey/True_Project/BASELINES.md): baseline and ablation plan.
- [PROPOSAL.md](C:/Users/ankey/True_Project/PROPOSAL.md): paper framing and contribution boundaries.
