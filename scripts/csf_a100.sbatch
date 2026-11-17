#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 12
#SBATCH -t 1-00:00
#SBATCH -J nano-spec-a100
#SBATCH -o nano-spec-a100-%j.out
set -euo pipefail
cd "${NANO_PROJECT:-$SLURM_SUBMIT_DIR}"
export NANO_GPU=1
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
bash scripts/run_csf.sh
