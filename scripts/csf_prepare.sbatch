#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 8
#SBATCH -t 0-04:00
#SBATCH -J nano-spec-prepare
#SBATCH -o nano-spec-prepare-%j.out
set -euo pipefail
cd "${NANO_PROJECT:-$SLURM_SUBMIT_DIR}"
bash scripts/prepare_environment.sh
bash scripts/container.sh tools/download_models.py --out /models
bash scripts/container.sh tools/prepare_panel.py --target /models/Qwen3-8B --out benchmarks/inputs/speed-128.json
