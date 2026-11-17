#!/bin/bash
set -euo pipefail
mode=${1:?Usage: profile.sh torch|nsys|ncu JOB_JSON OUTPUT_DIRECTORY}
job=${2:?Missing job JSON}
out=${3:?Missing output directory}
python_bin=${PYTHON:-python}
mkdir -p "$(dirname "$out")"
case "$mode" in
    torch)
        "$python_bin" profile_spec.py --job "$job" --out "$out"
        ;;
    nsys)
        command -v nsys >/dev/null || { echo "Nsight Systems is unavailable in this environment." >&2; exit 2; }
        nsys profile --trace=cuda,nvtx,osrt --sample=none --capture-range=cudaProfilerApi \
            --capture-range-end=stop --output="$out-nsys" \
            "$python_bin" profile_spec.py --mode external --job "$job" --out "$out"
        ;;
    ncu)
        command -v ncu >/dev/null || { echo "Nsight Compute is unavailable in this environment." >&2; exit 2; }
        ncu --profile-from-start off --section SpeedOfLight --section MemoryWorkloadAnalysis \
            --launch-count "${NCU_KERNELS:-20}" --export "$out-ncu" \
            "$python_bin" profile_spec.py --mode external --job "$job" --out "$out"
        ;;
    *) echo "Unknown profiler: $mode" >&2; exit 2 ;;
esac
