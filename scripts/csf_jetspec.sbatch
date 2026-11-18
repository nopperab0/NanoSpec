#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 12
#SBATCH -t 0-02:00
#SBATCH -J nanospec-jetspec
#SBATCH -o jetspec-opt-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load libs/cuda/12.8.1
module load tools/env/proxy
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HOME="$PWD/.hf-cache"
run="results/jetspec-${SLURM_JOB_ID}"
mkdir -p "$run"
nvidia-smi > "$run/nvidia-smi.txt"
.runtime/bin/python -m pip freeze --all > "$run/pip-freeze.txt"
.runtime/bin/python tools/run_jetspec_optimization.py \
  --source "$NANO_ORIGINAL/results/csf-24846937/performance" --out "$run"
tar -czf "${run}.tar.gz" "$run" "jetspec-opt-${SLURM_JOB_ID}.out"
