# Baseline Plan

For a NeurIPS main-conference submission, the paper should separate three baseline types.

## 1. Controlled In-Repo Baselines

These are implemented by `scripts/run_experiment.py` and should be run with identical splits, seeds, optimizer, and training budget.

- `inception + none`: primary InceptionTime-style baseline.
- `inception + butterworth`: fixed DSP filtering control.
- `inception + complex`: phase-altering spectral gate control.
- `fcn + none`: simple fully convolutional TSC baseline.
- `resnet + none`: residual CNN baseline.
- `tslanet_lite + none` and `timesnet_lite + none`: in-repo lightweight neural competitors inspired by TSLANet and TimesNet mechanisms.
- `fcn + pdsi`, `resnet + pdsi`, `tslanet_lite + pdsi`, and `timesnet_lite + pdsi`: portability checks showing the gate is not only tuned to InceptionTime.

Generate configs:

```bash
python scripts/make_experiment_matrix.py \
  --template configs/npz_pdsi_template.json \
  --prefix ptbxl \
  --dataset-path data/ptbxl_superclass.npz \
  --seeds 0,1,2,3,4
```

Then run:

```bash
bash configs/generated/run_ptbxl_matrix.sh
```

## 2. External Strong Time-Series Baselines

MiniRocket, MultiRocket, Hydra, and MultiRocket-Hydra can be run directly on the same NPZ files with the `aeon` runner:

```bash
python -m pip install ".[baselines]"

python scripts/run_aeon_baseline.py --dataset data/ptbxl_superclass.npz --method minirocket --experiment-name ptbxl_minirocket --seeds 0,1,2,3,4
python scripts/run_aeon_baseline.py --dataset data/ptbxl_superclass.npz --method multirocket --experiment-name ptbxl_multirocket --seeds 0,1,2,3,4
python scripts/run_aeon_baseline.py --dataset data/ptbxl_superclass.npz --method hydra --experiment-name ptbxl_hydra --seeds 0,1,2,3,4
python scripts/run_aeon_baseline.py --dataset data/ptbxl_superclass.npz --method multirocket_hydra --experiment-name ptbxl_multirocket_hydra --seeds 0,1,2,3,4
```

For official deep baselines, fetch the upstream repositories and run their classification pipelines with exact commit hashes:

```bash
bash scripts/setup_official_baselines.sh
```

- TSLANet official repository.
- Time-Series-Library for TimesNet.
- PatchTST or a compact Transformer classifier if time permits.

Do not spend the first experiment cycle on these. First establish whether PDSI beats the controlled InceptionTime baseline and ablations. If it does, run these baselines for reviewer credibility.

## 3. Hyperparameter Selection

A full hyperparameter sweep is not required and would weaken the compute story. A small validation-only sweep is recommended:

- `num_bands in {8, 16, 32}`
- `max_delta in {0.25, 0.5, 0.75}`
- `lambda_tv in {0, 1e-4}`

Use one seed for the sweep. Select by validation Macro-F1. Then freeze the chosen config and run five seeds once. Do not tune on test metrics.

Generate sweep configs:

```bash
python scripts/make_experiment_matrix.py \
  --template configs/npz_pdsi_template.json \
  --prefix ptbxl \
  --dataset-path data/ptbxl_superclass.npz \
  --seeds 0,1,2,3,4 \
  --include-sweep
```
