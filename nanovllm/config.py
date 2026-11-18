import os
from dataclasses import dataclass
from transformers import AutoConfig

MAX_SPECULATIVE_BATCH_SIZE = 512   # the largest decode/verification CUDA-graph batch size
BLOCK_DRAFTS = ("dflash", "dspark")   # DeepSpec's block drafts: DFlash, and DSpark with its Markov head


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    enable_prefix_cache: bool = True
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    kv_cache_memory_bytes: int | None = None
    draft_model: str | None = None            # enables speculative decoding
    draft_type: str = "model"                 # "model": a draft LM with its own KV and decode steps; or one of BLOCK_DRAFTS
    method: str | None = None                 # public switch: none / naive / dflash / dspark / jetspec
    jetspec_tree_width: int = 4
    jetspec_tree_budget: int = 32              # includes the root/anchor
    jetspec_draft_shift: bool = False
    jetspec_backend: str = "triton"           # "reference": serial SDPA; "triton": batched GPU trees
    jetspec_cuda_graph: bool = False           # graph only JetSpec draft/target forwards, keep scheduler eager
    num_speculative_tokens: int = 0           # k: drafts proposed per round; >= 1 iff draft_model is set; DFlash: its block size
    draft_kv_cache_memory_bytes: int | None = None    # the model draft's KV budget; DFlash's KV mirrors the target's blocks
    seed: int | None = None                   # seed of the engine's sampling generator
    record_step_timings: bool = False         # record a pair of CUDA events per engine step
    record_request_events: bool = False       # expose per-step deliveries to the benchmark driver

    def __post_init__(self):
        if self.method is not None:
            if self.method not in ("none", "naive", *BLOCK_DRAFTS, "jetspec"):
                raise ValueError("method must be none, naive, dflash, dspark or jetspec")
            if self.method == "none":
                if self.draft_model is not None or self.num_speculative_tokens:
                    raise ValueError("method='none' cannot use a draft model or speculative tokens")
            else:
                if self.draft_model is None:
                    raise ValueError(f"method={self.method!r} requires draft_model")
                expected = "model" if self.method == "naive" else self.method
                if self.draft_type != "model" and self.draft_type != expected:
                    raise ValueError("method and draft_type disagree")
                self.draft_type = expected
                if self.method == "naive" and self.num_speculative_tokens == 0:
                    self.num_speculative_tokens = 3
        else:
            self.method = ("naive" if self.draft_type == "model" else self.draft_type) if self.draft_model else "none"
        assert os.path.isdir(self.model)
        if self.max_num_seqs < 1 or self.max_num_batched_tokens < 1 or self.max_model_len < 1:
            raise ValueError("batch and sequence limits must be positive")
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.kv_cache_memory_bytes is not None:
            if type(self.kv_cache_memory_bytes) is not int or self.kv_cache_memory_bytes <= 0:
                raise ValueError("kv_cache_memory_bytes must be a positive integer")
        if self.draft_model is not None:
            self._validate_speculative()
        elif self.num_speculative_tokens != 0:
            raise ValueError("num_speculative_tokens requires draft_model")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.hf_config.model_type != "qwen3":
            raise ValueError("this nano tutorial implements the dense Qwen3 architecture only")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.draft_type in (*BLOCK_DRAFTS, "jetspec") and self.draft_model:
            draft = AutoConfig.from_pretrained(self.draft_model)
            if draft.vocab_size != self.hf_config.vocab_size or draft.hidden_size != self.hf_config.hidden_size:
                raise ValueError("block draft vocabulary/hidden size must match the target")
            if getattr(draft, "num_target_layers", self.hf_config.num_hidden_layers) != self.hf_config.num_hidden_layers:
                raise ValueError("head was trained for a different target layer count")
            if self.max_model_len > draft.max_position_embeddings:
                raise ValueError("max_model_len exceeds the head checkpoint position limit")
            layers = draft.dflash_config["target_layer_ids"] if self.method == "jetspec" else draft.target_layer_ids
            if any(i < 0 or i >= self.hf_config.num_hidden_layers for i in layers):
                raise ValueError("draft target_layer_ids are outside the target's layers")

    def _validate_speculative(self) -> None:
        if not os.path.isdir(self.draft_model):
            raise ValueError(f"draft_model is not a directory: {self.draft_model}")
        if self.draft_type not in ("model", *BLOCK_DRAFTS, "jetspec"):
            raise ValueError(f"unknown draft_type {self.draft_type!r}")
        if self.draft_type == "jetspec":
            if self.jetspec_backend not in ("reference", "triton"):
                raise ValueError("jetspec_backend must be reference or triton")
            if self.jetspec_cuda_graph and self.jetspec_backend != "triton":
                raise ValueError("JetSpec CUDA Graphs require jetspec_backend='triton'")
            if self.jetspec_backend == "triton" and self.jetspec_tree_budget > 64:
                raise ValueError("GPU ancestor bitsets support at most 64 nodes; select reference for larger trees")
            draft = AutoConfig.from_pretrained(self.draft_model)
            if not getattr(draft, "dflash_config", {}).get("causal_head", False):
                raise ValueError("JetSpec requires a trained causal head checkpoint")
            if self.jetspec_tree_budget < 2 or self.jetspec_tree_width < 1 or draft.block_size < 2:
                raise ValueError("JetSpec requires budget >= 2, width >= 1 and block_size >= 2")
            if not self.enforce_eager:
                raise ValueError("JetSpec uses enforce_eager=True for the ordinary engine; enable jetspec_cuda_graph for its forward graphs")
            reserve = max(self.jetspec_tree_budget - 1, draft.block_size - 1)
            if self.num_speculative_tokens not in (0, reserve):
                raise ValueError("JetSpec reserves scratch slots from tree budget and checkpoint block size")
            self.num_speculative_tokens = reserve
        if self.draft_type in BLOCK_DRAFTS:
            draft_config = AutoConfig.from_pretrained(self.draft_model)
            markov_rank = getattr(draft_config, "markov_rank", 0)
            if (markov_rank > 0) != (self.draft_type == "dspark"):
                raise ValueError(f"draft_type {self.draft_type!r} does not match a checkpoint with markov_rank {markov_rank}: "
                                 "DFlash has no Markov head, DSpark has one")
            if markov_rank > 0 and draft_config.markov_head_type != "vanilla":
                raise ValueError(f"only DSpark's vanilla Markov head is implemented, not {draft_config.markov_head_type!r}")
            block_size = draft_config.block_size
            if self.num_speculative_tokens not in (0, block_size):
                raise ValueError(f"a block draft proposes its block size ({block_size}) tokens per round")
            self.num_speculative_tokens = block_size
        if self.num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1 with a draft model")
        if self.tensor_parallel_size != 1:
            raise ValueError("speculative decoding supports tensor_parallel_size=1 only")
        if self.max_num_seqs > MAX_SPECULATIVE_BATCH_SIZE:
            raise ValueError(f"speculative decoding captures verification graphs for at most {MAX_SPECULATIVE_BATCH_SIZE} sequences")
        if self.enable_prefix_cache:
            raise ValueError("speculative decoding requires enable_prefix_cache=False")
        if self.kv_cache_memory_bytes is None:
            raise ValueError("speculative decoding requires an explicit kv_cache_memory_bytes")
        if self.draft_type in (*BLOCK_DRAFTS, "jetspec"):
            return
        if self.draft_kv_cache_memory_bytes is None:
            raise ValueError("a model draft requires an explicit draft_kv_cache_memory_bytes")
        if type(self.draft_kv_cache_memory_bytes) is not int or self.draft_kv_cache_memory_bytes <= 0:
            raise ValueError("draft_kv_cache_memory_bytes must be a positive integer")
