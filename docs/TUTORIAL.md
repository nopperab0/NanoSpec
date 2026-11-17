# Four methods, one nano engine

## Anchor and KV invariant

At round entry the committed sequence has L tokens. The final token is the anchor: it has been delivered but has not passed through the target yet. Target KV covers positions `0..L-2`, so `cached_tokens = L-1`. Every method restores this invariant.

Ordinary decoding processes the anchor and draws one token. Speculation processes candidate positions together, accepts a prefix/path, and makes the final target token the next uncached anchor.

Read `engine/sequence.py`, `engine/scheduler.py` and `engine/batch_metadata.py` first. They express token positions, physical block mappings and multi-token reservations separately.

## Naive standalone drafting

A small ordinary LM proposes k tokens sequentially, retaining the exact sampling distribution q at every position. The target processes `[anchor, d1, ..., dk]` once. Logit row zero validates d1; row k produces a bonus token if every draft is accepted.

Target and draft have independent KV states and block tables. On rejection, effective lengths roll back and the next round overwrites invalid entries. After full acceptance the draft can lag one cached position behind the target; `ModelDrafter.catch_up` computes it before proposing again.

## Chain rejection sampling

Accept candidate y with probability `min(1, p(y)/q(y))`. On the first rejection, draw the correction from the normalized positive residual `max(p-q, 0)`. After full acceptance, draw a bonus from the final target distribution. This preserves the target distribution.

The q supplied to acceptance must be the distribution that actually generated the candidate: temperature, top-k, top-p and any Markov bias included. `layers/spec_sampler.py` implements the shared transformation and rejection logic. Ordinary decoding uses the same sampling transform. Greedy verification instead accepts consecutive exact target-argmax matches and then emits the target token.

## DFlash block drafting

DFlash requires a matching trained head, not an arbitrary small LM. Target intermediate layer outputs are concatenated and projected into each draft layer's context K/V. One bidirectional pass over `[anchor, mask, ...]` produces the candidate block. DeepSpec block7 predicts seven draft tokens from seven input rows; target verification uses eight positions.

Each target prefill chunk updates draft context. Verified target features are written back after a round. Invalid speculative slots may remain physically present, but the next context excludes them and overwrites them before reuse.

## DSpark Markov correction

DSpark reuses the block backbone and adds `bias_j = W2(Embedding(previous_token))` to each position's logits. The previous token is the actual sampled predecessor, or the anchor for the first row. Sampling therefore runs left to right, while the expensive block forward still runs once.

Acceptance consumes each corrected q. The confidence-head parameters are loaded for checkpoint compatibility but this project always verifies the fixed block. Candidate-window sampling and confidence-based verification scheduling from later reference revisions are excluded.

## Independent JetSpec path

Original JetSpec uses a separately trained causal head sharing the target embedding and LM head. This port uses its block16 checkpoint. In-place prediction reads head output rows 1..15. A shifted checkpoint requires explicit `jetspec_draft_shift=True`; the slices are not interchangeable.

1. Store context K/V projected from accepted target features.
2. Run one causal head forward, extract per-depth top-W candidates and build a best-first cumulative-logprob tree up to node budget B. The root is the anchor and counts towards B.
3. Verify all nodes under an ancestor mask. Each node sees the cached prefix, ancestors and itself, but no sibling branches.
4. Traverse a target-generated path and emit its tokens plus an uncached tail.
5. Gather root and accepted-path KV from scratch slots into contiguous logical positions. Read the source before overlapping writes. Store only the corresponding accepted target features in the head context.

Tree depth determines RoPE position; node index determines the scratch slot. Treating these as the same breaks tree verification. An ordinary lower-triangular causal mask also leaks sibling information and is incorrect.

The base cumulative-logprob builder reuses per-depth candidates across expansion, matching the official basic `accum_logp` path. It does not rerun the head for each new branch or implement optional path refresh/router strategies.

Greedy traversal follows target argmax. Stochastic traversal draws from the full target conditional distribution at each visited node; it continues if that token is a child, otherwise emits it and ends the round. The tree is a lookup accelerator for already-computed target conditionals, not a renormalized target distribution. This preserves the target distribution, but differs from the paper's non-greedy p/q rejection protocol.

The readable tree attention backend is eager PyTorch SDPA, processed per request. Ordinary nano attention is unchanged. There is no JetSpec tree batching, graph capture, custom high-performance kernel or JetSpec++ pipeline here.

## Suggested experiments

Start with naive k=1, then k=3 and k=7. Compare drafting cost with actual tokens advanced per round. Contrast DFlash and DSpark to see whether extra accepted tokens repay sequential Markov sampling. Sweep JetSpec budget 8/16/32 and width 1/2/4 to see how branching changes verified work, latency and memory.

Always distinguish accepted candidates, actual committed tokens, request delivery gaps and per-token average latency. The benchmark retains all four so that a throughput improvement does not conceal worse tail delivery latency.
