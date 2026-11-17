#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 12
#SBATCH -t 1-00:00
#SBATCH -J nano-spec-five
#SBATCH -o nano-spec-five-%j.out
set -euo pipefail
cd "${NANO_PROJECT:-$SLURM_SUBMIT_DIR}"
module purge
module load libs/cuda/12.8.1
module load tools/env/proxy
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HOME="$PWD/.hf-cache"
export NANO_PYTHON="$PWD/.runtime/bin/python"
export NANO_RUNTIME_MODELS="$PWD/models"
bash scripts/run_csf.sh
