#!/usr/bin/env bash
set -euo pipefail

# End-to-end controlled PDSI experiment pipeline.
# External baselines such as Hydra, MiniRocket, TSLANet, and TimesNet must still
# be run from their own repositories and merged into the final table.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/scratch/cld-trans/datasets}"
SEEDS="${SEEDS:-0,1,2,3,4}"

PTBXL_ROOT="${PTBXL_ROOT:-${DATA_ROOT}/ptb-xl}"
CHBMIT_ROOT="${CHBMIT_ROOT:-${DATA_ROOT}/chb-mit}"
SLEEPEDF_ROOT="${SLEEPEDF_ROOT:-${DATA_ROOT}/sleep-edf}"

PTBXL_NPZ="${PTBXL_NPZ:-${DATA_ROOT}/ptbxl_superclass.npz}"
CHBMIT_NPZ="${CHBMIT_NPZ:-${DATA_ROOT}/chbmit_windows.npz}"
SLEEPEDF_NPZ="${SLEEPEDF_NPZ:-${DATA_ROOT}/sleepedf_epochs.npz}"

echo "Project root: ${PROJECT_ROOT}"
echo "Data root   : ${DATA_ROOT}"
echo "Seeds       : ${SEEDS}"

python -m pip install -e .

if [[ ! -d "${PTBXL_ROOT}" || ! -d "${CHBMIT_ROOT}" || ! -d "${SLEEPEDF_ROOT}" ]]; then
  echo "Linking existing datasets from ${SOURCE_DATA_ROOT}"
  bash scripts/link_existing_datasets.sh --source "${SOURCE_DATA_ROOT}" --target "${DATA_ROOT}"
fi

echo "Running synthetic smoke tests"
python scripts/run_experiment.py --config configs/synthetic_baseline_smoke.json
python scripts/run_experiment.py --config configs/synthetic_pdsi_smoke.json
mkdir -p runs
python scripts/summarize_results.py \
  --primary runs/synthetic_pdsi_smoke/summary.json \
  --baseline runs/synthetic_baseline_smoke/summary.json \
  > runs/synthetic_pdsi_vs_baseline.json

echo "Preprocessing PTB-XL"
if [[ ! -f "${PTBXL_NPZ}" ]]; then
  python scripts/preprocess_ptbxl.py --root "${PTBXL_ROOT}" --out "${PTBXL_NPZ}" --sampling-rate 100
else
  echo "exists: ${PTBXL_NPZ}"
fi

echo "Preprocessing CHB-MIT"
if [[ ! -f "${CHBMIT_NPZ}" ]]; then
  python scripts/preprocess_chbmit.py --root "${CHBMIT_ROOT}" --out "${CHBMIT_NPZ}"
else
  echo "exists: ${CHBMIT_NPZ}"
fi

# echo "Preprocessing Sleep-EDF"
# if [[ ! -f "${SLEEPEDF_NPZ}" ]]; then
#   python scripts/preprocess_sleepedf.py --root "${SLEEPEDF_ROOT}" --out "${SLEEPEDF_NPZ}"
# else
#   echo "exists: ${SLEEPEDF_NPZ}"
# fi

echo "Generating and running PTB-XL matrix"
python scripts/make_experiment_matrix.py \
  --template configs/npz_pdsi_template.json \
  --out-dir configs/generated/ptbxl \
  --prefix ptbxl \
  --dataset-path "${PTBXL_NPZ}" \
  --seeds "${SEEDS}" \
  --include-sweep
bash configs/generated/ptbxl/run_ptbxl_matrix.sh
python scripts/select_best_sweep.py --runs-dir runs --top-k 10 > runs/ptbxl_sweep_top10.json

echo "Creating CHB-MIT template"
python - <<'PY'
import json
p = "configs/chbmit_template.json"
cfg = json.load(open("configs/npz_pdsi_template.json"))
cfg["experiment_name"] = "chbmit_inception_pdsi"
cfg["data"]["num_channels"] = 18
cfg["data"]["seq_len"] = 2560
cfg["dataset"]["path"] = "data/chbmit_windows.npz"
cfg["dataset"]["channels_last"] = False
cfg["model"]["num_channels"] = 18
cfg["model"]["num_classes"] = 2
cfg["model"]["multilabel"] = False
cfg["train"]["batch_size"] = 128
json.dump(cfg, open(p, "w"), indent=2)
PY

echo "Generating and running CHB-MIT matrix"
python scripts/make_experiment_matrix.py \
  --template configs/chbmit_template.json \
  --out-dir configs/generated/chbmit \
  --prefix chbmit \
  --dataset-path "${CHBMIT_NPZ}" \
  --seeds "${SEEDS}"
bash configs/generated/chbmit/run_chbmit_matrix.sh

# echo "Creating Sleep-EDF template"
# python - <<'PY'
# import json
# p = "configs/sleepedf_template.json"
# cfg = json.load(open("configs/npz_pdsi_template.json"))
# cfg["experiment_name"] = "sleepedf_inception_pdsi"
# cfg["data"]["num_channels"] = 1
# cfg["data"]["seq_len"] = 3000
# cfg["dataset"]["path"] = "data/sleepedf_epochs.npz"
# cfg["dataset"]["channels_last"] = False
# cfg["model"]["num_channels"] = 1
# cfg["model"]["num_classes"] = 5
# cfg["model"]["multilabel"] = False
# cfg["train"]["batch_size"] = 128
# json.dump(cfg, open(p, "w"), indent=2)
# PY

# echo "Generating and running Sleep-EDF matrix"
# python scripts/make_experiment_matrix.py \
#   --template configs/sleepedf_template.json \
#   --out-dir configs/generated/sleepedf \
#   --prefix sleepedf \
#   --dataset-path "${SLEEPEDF_NPZ}" \
#   --seeds "${SEEDS}"
# bash configs/generated/sleepedf/run_sleepedf_matrix.sh

mkdir -p runs/comparisons

compare() {
  local name="$1"
  local primary="$2"
  local baseline="$3"
  if [[ -f "runs/${primary}/summary.json" && -f "runs/${baseline}/summary.json" ]]; then
    python scripts/summarize_results.py \
      --primary "runs/${primary}/summary.json" \
      --baseline "runs/${baseline}/summary.json" \
      > "runs/comparisons/${name}.json"
  else
    echo "missing comparison inputs: ${primary} vs ${baseline}" >&2
  fi
}

echo "Writing comparison summaries"
compare ptbxl_pdsi_vs_inception ptbxl_inception_pdsi ptbxl_inception_none
compare ptbxl_pdsi_vs_complex ptbxl_inception_pdsi ptbxl_inception_complex
compare ptbxl_pdsi_vs_butterworth ptbxl_inception_pdsi ptbxl_inception_butterworth
compare ptbxl_fcn_pdsi_vs_fcn ptbxl_fcn_pdsi ptbxl_fcn_none
compare ptbxl_resnet_pdsi_vs_resnet ptbxl_resnet_pdsi ptbxl_resnet_none
compare chbmit_pdsi_vs_inception chbmit_inception_pdsi chbmit_inception_none
# compare sleepedf_pdsi_vs_inception sleepedf_inception_pdsi sleepedf_inception_none

echo "Profiling model variants"
python scripts/profile_model.py --num-channels 12 --num-classes 5 --seq-len 1000 --gate none --batch-size 256 > runs/profile_ptbxl_none.json
python scripts/profile_model.py --num-channels 12 --num-classes 5 --seq-len 1000 --gate pdsi --batch-size 256 > runs/profile_ptbxl_pdsi.json
python scripts/profile_model.py --num-channels 12 --num-classes 5 --seq-len 1000 --gate complex --batch-size 256 > runs/profile_ptbxl_complex.json

if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("aeon") else 1)
PY
then
  echo "Running aeon external classical baselines"
  for dataset in ptbxl chbmit; do
    case "${dataset}" in
      ptbxl) npz="${PTBXL_NPZ}" ;;
      chbmit) npz="${CHBMIT_NPZ}" ;;
      sleepedf) npz="${SLEEPEDF_NPZ}" ;;
    esac
    for method in minirocket multirocket hydra multirocket_hydra; do
      python scripts/run_aeon_baseline.py \
        --dataset "${npz}" \
        --method "${method}" \
        --experiment-name "${dataset}_${method}" \
        --seeds "${SEEDS}"
    done
  done
else
  echo "Skipping aeon baselines. Install with: python -m pip install '.[baselines]'"
fi

echo "Controlled pipeline complete."
echo "Next: run official deep baselines with: bash scripts/setup_official_baselines.sh"
