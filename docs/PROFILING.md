# Profiling guide

Performance and profiling are separate processes and output directories. Profiling deliberately adds overhead; its cells carry `profiled=true` and cannot enter performance reports.

## PyTorch and NVTX

```bash
python tools/select_profile_jobs.py results/four-methods --out results/profile-jobs
python profile_spec.py --job results/profile-jobs/dspark.json --out profiles/dspark --wait 2 --steps 8
```

The selector takes one successful C=1 cell per method, keeps its model/sampling configuration, and caps its workload at four requests and 64 output tokens. This is a diagnostic workload, not another throughput repeat. The trace skips two steps, warms the collector for one step, then records eight steps. Check profile_status.json: short workloads may not fill that schedule.

Artifacts:

- trace.json: bounded CPU/CUDA timeline for Chrome/Perfetto-compatible viewers.
- operators.txt/json: operator call counts and self CPU/device time. Nested inclusive ranges must not be summed as independent costs.
- result.json: original request/phase/occupancy metrics, explicitly marked as profiled.
- gpu_samples.json: UUID-scoped device utilization, memory-controller activity, framebuffer use, power, clocks and temperature sampled at roughly 250 ms.
- profile_status.json: capture completeness and telemetry availability/errors.

Ranges cover scheduling, target forwards/logits/verification, draft proposal/catch-up/context store, acceptance, JetSpec head and whole speculative rounds. CUDA phase event timings separately cover the algorithm's phase boundaries. Python instrumentation is installed only inside the profiling context and restored afterwards.

Device utilization is not SM occupancy; nvidia-smi memory-controller utilization is not HBM GB/s. Short bursts can fall between samples. Use kernel counters to measure bandwidth. Telemetry is disabled if a CUDA UUID is unavailable, rather than accidentally sampling a different physical GPU on a shared CSF node.

## Nsight Systems

```bash
bash scripts/profile.sh nsys results/profile-jobs/dspark.json profiles/dspark-nsys
```

The launcher traces CUDA, NVTX and OS runtime activity. cudaProfilerStart/Stop bounds capture to the measured workload, excluding loading/warmup. Inspect CPU launch gaps, CUDA Graph launches, GPU idle gaps, H2D/D2H copies, kernel counts and critical paths. The external report is placed beside the output directory.

## Nsight Compute

```bash
NCU_KERNELS=20 bash scripts/profile.sh ncu results/profile-jobs/dspark.json profiles/dspark-ncu
```

This records SpeedOfLight and MemoryWorkloadAnalysis sections for up to twenty launches after cudaProfilerStart. Inspect achieved DRAM throughput, SM throughput, occupancy and memory traffic for selected kernels; derive effective bandwidth only from matching bytes and time counters. Kernel replay changes execution time, so never use this run for end-to-end speedup.

Nsight binaries and hardware performance-counter permissions must be available on the allocated node. The prepared PyTorch image is not claimed to include Nsight Systems. The scripts fail explicitly when a tool is missing. PyTorch profiling remains the default CSF diagnostic; Nsight is optional and site-dependent.

For the container environment, launch the scripts with `bash scripts/container.sh --shell scripts/profile.sh ...` after loading or binding the appropriate tools. Do not use the profiling launcher to bypass site counter restrictions.

References: [PyTorch profiler](https://docs.pytorch.org/docs/2.8/profiler.html), [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/), [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).
