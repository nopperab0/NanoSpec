#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 8
#SBATCH -t 0-04:00
#SBATCH -J nano-spec-assets
#SBATCH -o nano-spec-assets-%j.out
set -euo pipefail
cd "${NANO_PROJECT:-$SLURM_SUBMIT_DIR}"
module purge
module load libs/cuda/12.8.1
module load tools/env/proxy
base_python="${NANO_BASE_PYTHON:?Set NANO_BASE_PYTHON to an existing compatible GPU Python executable}"
"$base_python" -m venv --system-site-packages .runtime
export PYTHONNOUSERSITE=1
export HF_HOME="$PWD/.hf-cache"
export TOKENIZERS_PARALLELISM=false
.runtime/bin/python -m pip install 'datasets==4.3.0'
.runtime/bin/python -m pip install --no-deps --no-build-isolation -e .
.runtime/bin/python -m pip freeze --all > .runtime/pip-freeze.txt
.runtime/bin/python tools/download_models.py --out models
.runtime/bin/python tools/prepare_panel.py --target models/Qwen3-8B --out benchmarks/inputs/speed-128.json
