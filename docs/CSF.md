# CSF: direct single-A100 run

The measured native run completed as job 24846937; see [results](../reports/a100-20261118/RESULTS.md). The container route below is an alternative that was not used in that run. The workflow runs the requested benchmark and profiling directly, with no preliminary experiment, test, GPU preflight or greedy gate.

## Allocation

scripts/csf_a100.sbatch requests `-p gpuA -G 1 -n 12 -t 1-00:00`: one A100, twelve host cores and a 24-hour limit. One Python engine process runs at a time, not twelve distributed ranks. The [CSF resource table](https://ri.itservices.manchester.ac.uk/csf3/batch-slurm/partitions/) lists A100 80 GB nodes and up to twelve CPU cores per GPU. Actual account eligibility and queue availability are controlled by CSF.

## Upload and prepare

Upload the delivered ZIP to your scratch directory using your usual authenticated connection. On CSF:

```bash
cd ~/scratch
unzip nano-vllm-speculative-tutorial.zip
cd nano-vllm-speculative
sbatch scripts/csf_prepare.sbatch
```

Load the site's Apptainer/Singularity module first if its command is not already available. No module version is guessed. Heavy installation runs in the preparation CPU allocation, not on the login node.

Preparation pulls the public PyTorch 2.8.0 / CUDA 12.6 devel image by the immutable registry digest recorded in CONTAINER.json. It creates a system-site-packages venv, installs requirements-csf.txt and the project, records the SIF SHA256 and package freeze, downloads all five pinned model snapshots, and freezes the English-only SPEED-Bench panel. FlashAttention may compile if no matching wheel is available.

This stage needs outbound access to Docker Hub, Hugging Face and package sources, along with sufficient scratch space. Large model/image files are not bundled. Default environment/model storage is `.csf-env/`. Export NANO_ENV_ROOT and/or NANO_MODELS_ROOT consistently to use other scratch directories. Scripts are LF-terminated and invoked with bash.

## Submit the direct run

After the preparation allocation has completed successfully:

```bash
sbatch scripts/csf_a100.sbatch
```

The run uses:

- Baseline plus naive, DFlash, DSpark and original JetSpec.
- Closed-loop C=1/8/32; seed 29; one pass.
- 128 requests per cell, 256 output tokens, temperature 0.7, top-k 20 and top-p 0.8.
- 16 GiB target KV for every method, 8 GiB separate naive draft KV, eager execution.
- Naive k=7, block7 heads, JetSpec tree budget=32 and width=4.
- Fifteen performance cells followed by separate C=1 PyTorch profiles for the baseline and four methods.

The larger KV budgets are intended to accommodate the 32-request, approximately 1k-prompt workload on an 80 GB A100. These are allocation settings, not measured memory guarantees; any actual failure is preserved in its cell log. For the six JetSpec eager and CUDA Graph cells and their four diagnostic profiles, set `NANO_ORIGINAL` to the source benchmark checkout containing `results/csf-24846937/performance`, then run `sbatch --export=ALL scripts/csf_jetspec.sbatch` from the prepared JetSpec checkout.

To set a different requested matrix directly:

```bash
export NANO_CONCURRENCY="1 8 32"
export NANO_SEEDS="29"
export NANO_REPEATS=1
export NANO_REQUESTS=128
export NANO_OUTPUT_TOKENS=256
sbatch --export=ALL scripts/csf_a100.sbatch
```

NANO_PANEL overrides the frozen panel path. No smaller workload or correctness run is inserted automatically.

## Results and monitoring

Results are written to `results/csf-JOB_ID/`: container/package provenance, performance/REPORT.md, performance/metrics.csv, raw cells/logs, copied profiling job specifications and five separate profiling directories. Copy the complete directory back to preserve request-level and phase-level evidence.

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
tail -n 60 nano-spec-a100-JOB_ID.out
scancel JOB_ID
```

The final command explicitly cancels a job; use it only when cancellation is intended. A failed cell stops the direct workflow rather than generating a partial speed ranking. No failed method is substituted with the baseline.

Nsight Systems/Compute are optional and require tools and counter permissions in the execution environment. The default profiling stage uses PyTorch/CUDA/NVTX. See PROFILING.md.

## Native environment route used for the measured run

The actual experiment borrowed an existing Linux GPU Python environment, then created a local system-site-packages
venv. The publication makes that input explicit instead of embedding a personal environment path:

```bash
export NANO_BASE_PYTHON=/absolute/path/to/existing/gpu/python
sbatch --export=ALL scripts/csf_existing_prepare.sbatch
# After successful preparation:
sbatch --export=ALL scripts/csf_existing_a100.sbatch
```

The existing environment must provide compatible CUDA PyTorch, Transformers, FlashAttention, Triton and build tooling.
The native launch uses `.runtime/bin/python` and `models/`. Recorded packages are in
`reports/a100-20261118/environment/pip-freeze.txt`; this route was measured with PyTorch 2.11.0+cu128.
The job requests one GPU; one Python worker uses the allocated host cores.

For exact input replay, recover the original `benchmarks/inputs/speed-128.json` and its `.metadata.json` from the
original CSF project. They are absent from the downloaded result bundle. Verify against the hash in `data/run.json`;
a freshly generated panel must be treated as a new workload unless its hash matches.

The measured run reused three exact compatible completed cells from a superseded job; future fresh runs need no
resume setting. Set `NANO_RESUME_FROM` only when deliberately reusing compatible results via `tools/run_single_seed.py`.
