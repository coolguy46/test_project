# Baseline Plan

For a NeurIPS main-conference submission, the paper should separate three baseline types.

## 1. Controlled In-Repo Baselines

These are implemented by `scripts/run_experiment.py` and should be run with identical splits, seeds, optimizer, and training budget.

- `inception + none`: primary InceptionTime-style baseline.
- `inception + butterworth`: fixed DSP filtering control.
- `inception + complex`: phase-altering spectral gate control.
- `fcn + none`: simple fully convolutional TSC baseline.
- `resnet + none`: residual CNN baseline.
- `fcn + pdsi` and `resnet + pdsi`: portability checks showing the gate is not only tuned to InceptionTime.

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

These should be run from their official or well-maintained implementations and imported into the final table.

- MiniRocket or MultiRocket.
- Hydra.
- TSLANet.
- TimesNet.
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
