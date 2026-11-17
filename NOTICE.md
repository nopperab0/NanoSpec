# Attribution

The original nano-vLLM LICENSE states Copyright (c) 2025 Xingkai Yu, MIT. The upstream repository is maintained by GeeeekExplorer.
See the unchanged LICENSE file.

The naive, DFlash and DSpark integration, sampling primitives, KV bookkeeping,
CPU test fixtures and the DFlash reference-check utilities are adapted from
[tianyuantong/nano-vllm-speculative-decoding](https://github.com/tianyuantong/nano-vllm-speculative-decoding),
commit `dec828dece18333aa8f8b1f4f6366cbf1e9ccdc4`, under its MIT license.
The baseline is nano-vLLM `bb823b3e06983d71485a8e1f23715ebd87d98ef8`.

The independent JetSpec implementation follows the checkpoint and forward contract
of [hao-ai-lab/JetSpec](https://github.com/hao-ai-lab/JetSpec), commit
`1d3b21ef366448046c806be50827eb9c1f0a42d0` (MIT, Copyright (c) 2026 The Parallel Tree
Decoding authors). Its runtime is not imported or vendored. The optional numerical
test imports an explicitly supplied source checkout as an oracle only.

DeepSpec model semantics are checked against [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec),
commit `005e03b81cec38b7da6399833d609ee89a2587f2` (MIT, Copyright (c) 2026 The DeepSpec Authors).

Model checkpoints retain their respective terms. Pinned source revisions are in
`benchmarks/models.json`; model weights are not included in this project.

Full upstream and reference MIT license texts are preserved under licenses/.
