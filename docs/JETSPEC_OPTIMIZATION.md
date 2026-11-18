# JetSpec execution optimization

The published A100 report describes the serial reference implementation in commit
`2b62616ef4515f99631834e7587d52990551d392`. The new implementation is a separate
execution path. No measured speedup for this change is claimed until its CSF run completes.

## Switches

```python
llm = LLM(
    target,
    method="jetspec",
    draft_model=draft,
    jetspec_backend="triton",       # reference retains the old per-request SDPA path
    jetspec_cuda_graph=True,
    jetspec_tree_budget=32,
    jetspec_tree_width=4,
    enforce_eager=True,              # ordinary nano runner; JetSpec graphs are separate
    enable_prefix_cache=False,
    kv_cache_memory_bytes=16 << 30,
    max_num_seqs=32,
    max_model_len=4096,
    seed=29,
)
```

The same switches are exposed as `--jetspec-backend` and `--jetspec-cuda-graph`
by `example.py` and `bench_spec.py run`.

## What changes

- The causal head processes all requests in one block forward. Target verification
  likewise processes `[batch, tree nodes]` together, rather than a Python request loop.
- A Triton tree builder retains best-first cumulative-log-probability expansion,
  per-depth top-W candidates and insertion-order tie breaking. Heap state and ancestor
  bitsets stay on the GPU. Budgets up to 64 nodes are supported by this backend;
  the reference backend supports larger budgets.
- Tree attention directly reads prefix and scratch KV through the paged block table.
  It uses online softmax and tensor-core dot products, maps GQA heads without repeating
  KV, and applies ancestor bitsets to scratch keys. There is no materialized prefix
  gather or dense `[query, prefix + nodes]` mask per target layer.
- Sampling retains the shared temperature/top-k/top-p truncation and exponential-race
  sampler, including tied threshold values. All node draws happen on the GPU, followed
  by device-side tree traversal. This is still target-distribution traversal, not the
  official JetSpec stochastic p/q tree rejection algorithm.
- Accepted KV entries are gathered into temporary storage across all requests and
  layers before any scatter. This preserves correctness when scratch source slots
  overlap committed destination slots. Accepted auxiliary features update head KV in
  one batch. Cache offsets use 64-bit arithmetic across the full multi-layer cache.
- One packed batch of counts/output tokens returns to nano's host scheduler per round.
  The host still owns sequence lengths, block allocation and output limits. There is
  no claim that the full engine runs without host synchronization.

## CUDA Graph scope

Optional graphs capture the draft-head and target-model forwards in power-of-two batch
buckets. Inputs, positions, block tables, prefix lengths, ancestor masks and output
buffers keep stable addresses. The paged tree attention loop reads prefix lengths on
device, so changing context lengths does not change the graph's Python control flow.
Graph buckets are created during the benchmark's existing full-workload warmup.

LM-head logits, stochastic sampling, tree construction/traversal, context commit and
the scheduler are outside these graphs. In particular, RNG draws are not frozen by
capture. `graph_captures`, `graph_replays`, `batched_target_forwards` and
`scheduler_batch_transfers` are recorded in cell stats. A measurement-time capture
count greater than zero indicates that warmup missed a bucket and must be disclosed.

Pre-sampling unvisited nodes and padding graph buckets change RNG consumption, so
equal seeds do not imply bitwise-identical output token sequences. The intended
target conditional distribution is unchanged; numerical parity has not been established
by the performance benchmark. Kernel numerical ties may also change the selected tree.

## CSF experiment

`scripts/csf_jetspec.sbatch` requests one A100 and runs six cells: Triton eager and
Triton forward graphs at concurrency 1/8/32, seed 29, one pass, the original frozen
128-input panel, 256 output tokens ignoring EOS and the original sampling settings.
Four separate profiles cover eager/graphs at C=1 and C=32. These diagnostic runs are
not throughput repetitions and do not enter performance ratios.

The old baseline and serial-tree cells are explicitly historical references. The
original node859 is drained, so a new node's hardware/runtime metadata must be checked
before interpreting the ratios as a controlled comparison. No extra correctness gate,
synthetic experiment or unit test is inserted before these actual workloads.

The profiling telemetry UUID is normalized to the `GPU-` form accepted by nvidia-smi;
capture status still records failures explicitly. The final result archive also
includes the original input panel and its metadata, missing from the earlier download.

Implementation files: `engine/jetspec_batched.py`, `layers/tree_kernels.py` and
`layers/paged_tree_attention.py`. The old `engine/tree.py` and SDPA reference remain
available for teaching and comparison.

Algorithmic background: [Triton fused-attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
The paged/tree-specific kernel here is implemented in this project.
