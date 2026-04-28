# Baseline Plan

For a NeurIPS main-track submission, the baseline story should separate three layers of evidence.

## 1. Controlled In-Repo Comparisons

These use the same backbone, training budget, and data splits.

- `router_mode = uniform`: same temporal mixer, no adaptive routing.
- `router_mode = static`: learned schedule, but not sample-adaptive.
- `router_mode = direct`: spectral summary routed directly, no prototype library.
- `router_mode = prototype`: full SETM.

This is the minimum set needed to support the paper's causal claim:
the benefit, if any, comes from prototype-mediated sample-adaptive routing rather than just from extra parameters or a different temporal backbone.

Generate configs:

```bash
python scripts/make_experiment_matrix.py \
  --template configs/npz_setm_template.json \
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

For reviewer credibility, also compare against at least one upstream compact neural model run from its maintained repository. Do not spend the first cycle on that. First check whether SETM beats the controlled `uniform` baseline and survives the `static` and `direct` ablations.

## 3. Hyperparameter Selection

A small validation-only sweep is enough for a low-tuning story:

- `num_bands in {8, 16, 32}`
- `num_prototypes in {4, 8, 12}`
- `lambda_temporal_smooth in {0, 1e-4}`

Use one seed for the sweep. Select by validation Macro-F1. Freeze the chosen config before running five seeds once.

Generate sweep configs:

```bash
python scripts/make_experiment_matrix.py \
  --template configs/npz_setm_template.json \
  --prefix ptbxl \
  --dataset-path data/ptbxl_superclass.npz \
  --seeds 0,1,2,3,4 \
  --include-sweep
```
