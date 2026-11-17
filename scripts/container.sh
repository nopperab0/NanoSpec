#!/bin/bash
set -euo pipefail
project=$(cd "$(dirname "$0")/.." && pwd)
if [[ -n ${NANO_PYTHON:-} ]]; then
    cd "$project"
    if [[ ${1:-} == --shell ]]; then
        shift
        export PATH="$(dirname "$NANO_PYTHON"):$PATH"
        exec /bin/bash "$@"
    fi
    exec "$NANO_PYTHON" "$@"
fi
env_root=${NANO_ENV_ROOT:-"$project/.csf-env"}
models=${NANO_MODELS_ROOT:-"$env_root/models"}
engine=$(command -v apptainer || command -v singularity || true)
[[ -n "$engine" && -s "$env_root/pytorch.sif" && -x "$env_root/venv/bin/python" ]] || {
    echo "Prepare the environment first; see docs/CSF.md." >&2; exit 2;
}
mkdir -p "$models" "$env_root/cache"
flags=()
if [[ ${NANO_GPU:-0} == 1 ]]; then flags+=(--nv); fi
command=(/env/venv/bin/python)
if [[ ${1:-} == --shell ]]; then shift; command=(/bin/bash); fi
exec "$engine" exec --cleanenv "${flags[@]}" \
    --bind "$project:/project,$env_root:/env,$models:/models" --pwd /project \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
    --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 --env TOKENIZERS_PARALLELISM=false \
    --env "OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}" --env HF_HOME=/env/cache/huggingface \
    --env TRITON_CACHE_DIR=/env/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/env/cache/inductor \
    --env PATH=/env/venv/bin:/opt/conda/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
    "$env_root/pytorch.sif" "${command[@]}" "$@"
