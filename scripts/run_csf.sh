#!/bin/bash
set -euo pipefail
project=$(cd "$(dirname "$0")/.." && pwd)
cd "$project"
run_root="results/csf-${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$run_root"
cp requirements-csf.txt provenance.json "$run_root/"
env_root=${NANO_ENV_ROOT:-"$project/.csf-env"}
if [[ -n ${NANO_PYTHON:-} ]]; then
    "$NANO_PYTHON" -m pip freeze --all > "$run_root/pip-freeze.txt"
    printf '%s\n' "$NANO_PYTHON" > "$run_root/python-path.txt"
else
    cp "$env_root/image.sha256" "$env_root/pip-freeze.txt" "$run_root/"
fi
sha256sum requirements-csf.txt > "$run_root/requirements.sha256"
nvidia-smi > "$run_root/nvidia-smi.txt"
models=${NANO_RUNTIME_MODELS:-/models}
common=(--target "$models/Qwen3-8B" --naive-draft "$models/Qwen3-0.6B"
    --dflash-draft "$models/dflash_qwen3_8b_block7" --dspark-draft "$models/dspark_qwen3_8b_block7"
    --jetspec-draft "$models/jetspec-qwen3-8b" --target-kv-gib 16 --draft-kv-gib 8
    --naive-k 7 --tree-budget 32 --tree-width 4 --max-model-len 4096 --eager)
panel=${NANO_PANEL:-benchmarks/inputs/speed-128.json}
printf '%s\n' "$panel" > "$run_root/panel.txt"
read -r -a concurrency <<< "${NANO_CONCURRENCY:-1 8 32}"
read -r -a seeds <<< "${NANO_SEEDS:-29}"
args=(--concurrency "${concurrency[@]}" --seeds "${seeds[@]}" --repeats "${NANO_REPEATS:-1}"
    --max-tokens "${NANO_OUTPUT_TOKENS:-256}" --num-requests "${NANO_REQUESTS:-128}")
if [[ -n ${NANO_RESUME_FROM:-} ]]; then
    bash scripts/container.sh tools/run_single_seed.py --source "$NANO_RESUME_FROM" \
        --out "$run_root/performance" --seed "${seeds[0]}" --pass-index 1
else
    bash scripts/container.sh bench_spec.py run "${common[@]}" --prompts "$panel" \
        --load closed "${args[@]}" --out "$run_root/performance"
fi
bash scripts/container.sh tools/select_profile_jobs.py "$run_root/performance" --out "$run_root/profile-jobs"
for job in "$run_root"/profile-jobs/*.json; do
    method=$(basename "$job" .json)
    bash scripts/container.sh profile_spec.py --job "$job" --out "$run_root/profiles/$method"
done
printf 'Run completed: %s\n' "$run_root"
