# Validation record

Date: 2026-11-18. Local environment: Python 3.11.9, PyTorch 2.15.0+cpu, Transformers 4.57.6. See CPU_ENVIRONMENT.json and CPU_TEST_LOG.txt for the exact environment and the last complete test output recorded before the direct-run instruction.

## Executed locally

- Complete CPU correctness suite, including stochastic sampling, KV reservation/rollback, EOS/output-limit truncation, method switches, tree masks, tree GQA attention, overlapping KV gathers and benchmark pairing.
- Official-source numerical oracles: the JetSpec causal head with shared tiny random weights, and the DeepSpec Markov head with identical weights and inputs.
- Frozen JetSpec checkpoint tensor-header compatibility, without downloading large weights.
- Synthetic-clock workload tests for closed-loop refills, bulk queueing, rate arrivals, burst-aware latency, TPOT, SLO goodput and actual committed tokens.
- Matrix dry runs, Python compilation, CLI smoke checks, local Bash syntax validation, English-only source/document scan, package build and ZIP integrity checks.
- Public Docker registry verification of the exact container digest in CONTAINER.json.

The complete test log records passed, skipped and subtest counts. Skips include unavailable CUDA/real-weight fixtures and parameter combinations deliberately outside the draft length. Tests use TORCHDYNAMO_DISABLE=1 and CPU-only attention substitutes; these do not validate real GPU kernels.

## Executed on CSF

Job 24846937 completed on one A100-SXM4-80GB in 02:14:04, exit 0:0. All five real-checkpoint eager paths ran
at concurrency 1/8/32, one seed and one pass; five separate PyTorch profiles exported complete active traces.
See [the measured report](../reports/a100-20261118/RESULTS.md) and its provenance records.

These performance runs do not establish real-weight numerical parity or output quality. CUDA Graph replay was
not exercised. GPU telemetry failed with nvidia-smi exit 6; no Nsight hardware counters were collected.
The native runtime differs from the proposed container. The downloaded archive lacks the original frozen input
panel, preventing a claim of complete exact-input reproducibility from this repository alone.

DeepSpec's package initializer imports Gemma4 and can require Transformers 5. The Markov oracle loads the exact required numerical source files without changing their formulas. Full external DeepSpec fixture generation should run in its own supported environment, then exported logits can be checked with this project.

The last completed local suite reported 166 passed, 24 skipped and 89 subtests passed. Subsequent direct-entry and measurement-window edits were not retested, as requested. No new tests or defensive experiments were performed after that instruction.
