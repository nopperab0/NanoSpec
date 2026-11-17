#!/bin/bash
set -euo pipefail
project=$(cd "$(dirname "$0")/.." && pwd)
env_root=${NANO_ENV_ROOT:-"$project/.csf-env"}
image_uri=${NANO_IMAGE_URI:-docker://pytorch/pytorch@sha256:adb0f2d3769e0796a5d86740b0f900a72835b6fa9577b77d6efed460f1322fcb}
engine=$(command -v apptainer || command -v singularity || true)
if [[ -z "$engine" ]]; then
    echo "Load the site's Apptainer/Singularity module, then rerun this script." >&2
    exit 2
fi
mkdir -p "$env_root" "$env_root/cache" "$env_root/models"
if [[ ! -s "$env_root/pytorch.sif" ]]; then
    "$engine" pull "$env_root/pytorch.sif" "$image_uri"
fi
printf '%s\n' "$image_uri" > "$env_root/image-uri.txt"
sha256sum "$env_root/pytorch.sif" > "$env_root/image.sha256"
"$engine" exec --cleanenv --bind "$project:/project,$env_root:/env" --pwd /project \
    --env PYTHONNOUSERSITE=1 --env MAX_JOBS=2 --env TORCH_CUDA_ARCH_LIST=8.0 \
    "$env_root/pytorch.sif" bash -euc '
        python -m venv --system-site-packages /env/venv
        /env/venv/bin/python -m pip install --no-build-isolation -r requirements-csf.txt
        /env/venv/bin/python -m pip install --no-deps --no-build-isolation -e .
        /env/venv/bin/python -m pip freeze --all > /env/pip-freeze.txt
    '
sha256sum "$project/requirements-csf.txt" > "$env_root/requirements.sha256"
printf 'Environment prepared.\n'
