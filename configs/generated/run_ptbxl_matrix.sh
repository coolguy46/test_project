#!/usr/bin/env bash
set -euo pipefail

python scripts/run_experiment.py --config configs/generated/ptbxl_inception_none.json
python scripts/run_experiment.py --config configs/generated/ptbxl_inception_pdsi.json
python scripts/run_experiment.py --config configs/generated/ptbxl_inception_complex.json
python scripts/run_experiment.py --config configs/generated/ptbxl_inception_butterworth.json
python scripts/run_experiment.py --config configs/generated/ptbxl_fcn_none.json
python scripts/run_experiment.py --config configs/generated/ptbxl_resnet_none.json
python scripts/run_experiment.py --config configs/generated/ptbxl_fcn_pdsi.json
python scripts/run_experiment.py --config configs/generated/ptbxl_resnet_pdsi.json
python scripts/run_experiment.py --config configs/generated/sweep/ptbxl_sweep_b8_d0p25_tv0.json
python scripts/run_experiment.py --config configs/generated/sweep/ptbxl_sweep_b8_d0p25_tv0.0001.json
python scripts/run_experiment.py --config configs/generated/sweep/ptbxl_sweep_b8_d0p5_tv0.json
python scripts/run_experiment.py --config configs/generated/sweep/ptbxl_sweep_b8_d0p5_tv0.0001.json
python scripts/run_experiment.py --config configs/generated/sweep/ptbxl_sweep_b8_d0p75_tv0.json
python scripts/run_experiment.py --config configs/generated/sweep/ptbxl_sweep_b8_d0p75_tv0.0001.json
python scripts/run_experiment.py --config configs/generated/sweep/ptbxl_sweep_b16_d0p25_tv0.json
python scripts/run_experiment.py --config configs/generated/sweep/ptbxl_sweep_b16_d0p25_tv0.0001.json
python scripts/run_experiment.py --config configs/generated/sweep/ptbxl_sweep_b16_d0p5_tv0.json
python scripts/run_experiment.py --config configs/generated/sweep/ptbxl_sweep_b16_d0p5_tv0.0001.json
python scripts/run_experiment.py --config configs/generated/sweep/ptbxl_sweep_b16_d0p75_tv0.json
python scripts/run_experiment.py --config configs/generated/sweep/ptbxl_sweep_b16_d0p75_tv0.0001.json
python scripts/run_experiment.py --config configs/generated/sweep/ptbxl_sweep_b32_d0p25_tv0.json
python scripts/run_experiment.py --config configs/generated/sweep/ptbxl_sweep_b32_d0p25_tv0.0001.json
python scripts/run_experiment.py --config configs/generated/sweep/ptbxl_sweep_b32_d0p5_tv0.json
python scripts/run_experiment.py --config configs/generated/sweep/ptbxl_sweep_b32_d0p5_tv0.0001.json
python scripts/run_experiment.py --config configs/generated/sweep/ptbxl_sweep_b32_d0p75_tv0.json
python scripts/run_experiment.py --config configs/generated/sweep/ptbxl_sweep_b32_d0p75_tv0.0001.json
