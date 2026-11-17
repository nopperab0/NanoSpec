# Baseline and change boundary

Here original nano-vLLM means GeeeekExplorer's teaching engine, not production vLLM V1. There is no upstream tag named v1. The exact base is pinned:

- [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm/commit/bb823b3e06983d71485a8e1f23715ebd87d98ef8): `bb823b3e06983d71485a8e1f23715ebd87d98ef8`.
- [Early speculative integration](https://github.com/tianyuantong/nano-vllm-speculative-decoding/commit/dec828dece18333aa8f8b1f4f6366cbf1e9ccdc4): `dec828dece18333aa8f8b1f4f6366cbf1e9ccdc4`, retaining standalone drafting before later candidate-window and variable-verification optimizations.
- [Original JetSpec method](https://github.com/hao-ai-lab/JetSpec/tree/1d3b21ef366448046c806be50827eb9c1f0a42d0/jetspec/methods/jetspec): independently implemented in nano; optional official-source numerical oracle only.

`layers/activation.py`, `layers/linear.py` and `llm.py` remain byte-identical to the base. Ordinary FlashAttention and the original Triton KV-store kernel are retained. The new tree backend is selected only by JetSpec.

Necessary changes include two-model KV bookkeeping, round reservations and commits, shared runtime ownership, all-position logits, exported target features, keys-only RoPE, exact sampling, and explicit method/limit validation. FP32 RMSNorm aliasing is corrected for numerical verification. New DeepSpec rope_parameters are read correctly under Transformers 4.x. Graph batch sizes cover irregular maximum batch limits.

Infrastructure changes expose optional delivery events, count actual post-truncation tokens, cap total running requests at the configured concurrency, and record preemption/recomputation. Benchmark and profiling logic stays in separate small modules rather than replacing nano's engine with a serving framework.

Excluded: candidate-only sampling, confidence-based verification lengths, round-robin prefill, speculative prefix caching, P/D disaggregation, custom optimized verification kernels, task routers, and JetSpec++. The original base's ordinary chunked-prefill/prefix-cache behavior is otherwise inherited. Prefix caching is disabled for speculation and controlled comparisons.

The reference repository's published performance numbers do not apply to this implementation. See NOTICE.md and provenance.json for attribution and exact sources.
