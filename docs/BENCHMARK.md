# Benchmark protocol

## Workload and execution

Each cell is `(method, concurrency, seed, repeat)` in a fresh process. Matrix order is shuffled with a fixed order seed. Every method is paired with the `none` baseline at the same C, seed and repeat. A failed cell stops execution and preserves its log; incomplete or mismatched matrices cannot be summarized.

- `closed`: keep up to C outstanding requests, refill immediately after completion.
- `bulk`: enqueue all requests at time zero; nano caps running requests at C.
- `rate`: enqueue requests at deterministic `index / arrival_rate` timestamps, including admission lateness in latency. This is open-loop load with an explicit queue, not a network server.

`num_requests=0` consumes the panel once. A larger count cycles it. Seed, panel hash, source hash, model config hashes, optional model SOURCE revisions, sampling settings, software versions and hardware identity are recorded. Random seeds reproduce a fixed batch composition; they do not make different speculative methods consume identical random streams.

The default comparison runs every method eagerly with the same target KV budget and prefix caching disabled. Naive has a separate draft KV budget. Block heads allocate additional KV outside the target budget. Weights, activations and graphs consume additional memory. Peak allocated and reserved memory are reported separately.

## Throughput windows

Wall throughput is the number of actual delivered output tokens divided by host-observed workload duration. It includes arrivals, prefill, draft, verification, sampling, scheduling and output token transfer. It excludes model loading, graph capture, warmup, report generation and text detokenization. CUDA is synchronized before and after the workload.

Closed-loop steady throughput uses steps after every initial-cohort request has delivered a token and before refilling can no longer maintain C outstanding requests. A step is included only if C requests are outstanding both before it and after refill. The denominator spans the first included step start through the last included step end, including intervening driver/admission overhead; the numerator is actual tokens delivered in exactly those steps. The explicit step flags are retained for auditing. There is no steady-state claim for bulk/rate mode or a run without a valid full window: the value is null/N/A.

Decode-step throughput excludes prefill steps and uses their host durations, so it measures decode service efficiency rather than end-to-end serving throughput. Compare it with wall/steady throughput, not as a substitute.

## Request latency and SLOs

All timestamps use the same host monotonic clock. They describe delivery at the engine API boundary, not remote-client/network latency.

- TTFT: first delivery minus scheduled arrival time.
- Completion latency: final delivery minus arrival.
- Initial queue delay: first scheduling step start minus arrival. Later preemption delays are included in completion latency, not relabeled as initial queue time.
- TPOT: `(last_delivery - first_delivery) / (output_tokens - 1)`, undefined for one-token outputs.
- Delivery gap: time between successive deliveries to one request. A speculative delivery can contain several tokens; no artificial per-token timestamps are invented.
- Goodput: completed requests, or their tokens, per wall second when TTFT, TPOT and maximum delivery gap all satisfy the configured thresholds. Defaults: 1000 ms TTFT, 50 ms TPOT, 100 ms maximum gap.

Raw request values are retained along with p50/p95/p99/max/mean distributions. Report tables show medians of per-cell quantiles; they are not pooled quantiles. SLO thresholds are part of the paired workload identity.

## Speculative and resource efficiency

- Actual committed tokens per sequence-round uses post-EOS/post-limit output deltas. Prefill tokens are excluded.
- Accepted-prefix histograms and survival probabilities use rejection/traversal decisions before output truncation.
- Verified positions per actual committed decode token exposes discarded verification work.
- CUDA phase distributions and microseconds per committed token locate draft, verification, sampling and commit costs.
- Target/draft KV block high-water marks are sampled immediately after reservation, before finished requests release their blocks.
- Preemption count and cached tokens invalidated by preemption expose recomputation pressure.

`accepted/proposed` has different denominators: chain candidates for naive/DFlash/DSpark, and all non-root tree nodes for JetSpec. Mutually exclusive tree branches make this unsuitable as a common acceptance-probability ranking. Use committed tokens per round, verified work and latency together.

## Controls and interpretation

The benchmark uses fixed output length with EOS ignored. Correctness tests separately exercise EOS. Top-p is applied after top-k and requires a positive top-k. Warmup is excluded, the target generator is reset afterwards, and counters are reset before measurement.

For roughly matched target verification budgets, use naive k=7, block7 DFlash/DSpark and JetSpec budget=8 including the root. For natural configurations, use JetSpec budget=32 and sweep width/budget explicitly. Its trained block16 head remains different from a block7 head. Checkpoint training and model capacity are confounders, even with matched verification work.

JetSpec uses batched paged Triton tree verification, measured both in eager mode and with CUDA Graphs for the draft-head and target forwards. DFlash and DSpark use batched FlashAttention verification.

The built-in panel is for smoke tests. The optional SPEED panel freezes unique first turns by salted SHA256 order, filters CJK text, applies the Qwen non-thinking template and stores token IDs, category labels, dataset revision and tokenizer metadata. It is not category-stratified and is not the reference repository's confirmation panel. Respect the dataset's evaluation license.

Use multiple seeds and repetitions. Report paired speedup medians and ranges, with timing noise. The report does not perform significance testing or claim statistical confidence. Run larger C and arrival-rate sweeps to locate throughput saturation and SLO capacity; failed/OOM cells are capacity failures, not speed measurements.
