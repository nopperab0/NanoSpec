"""JetSpec causal head in nano layers, using the target's embeddings and LM head.

Checkpoint contract: JetSpec/jetspec-qwen3-8b, in-place prediction by default.
No import of the official JetSpec runtime and no DFlash checkpoint substitution.
"""
import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.layers import attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.models.qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM


class JetSpecHead(nn.Module):
    packed_modules_mapping = Qwen3ForCausalLM.packed_modules_mapping

    def __init__(self, config):
        super().__init__()
        spec = config.dflash_config
        self.block_size = config.block_size
        self.mask_token_id = spec["mask_token_id"]
        self.aux_layer_ids = tuple(spec["target_layer_ids"])
        if spec.get("target_hidden_size", config.hidden_size) != config.hidden_size:
            raise ValueError("JetSpec hidden-dimension adapters are outside this Qwen3 tutorial")
        self.layers = nn.ModuleList(Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = ReplicatedLinear(len(self.aux_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def store_context(self, aux, positions, slots):
        context = self.hidden_norm(self.fc(aux))
        for layer in self.layers:
            attn = layer.self_attn
            k, v = F.linear(context, attn.qkv_proj.weight[attn.q_size:]).split([attn.kv_size, attn.kv_size], -1)
            k = attn.k_norm(k.view(-1, attn.num_kv_heads, attn.head_dim))
            k = attn.rotary_emb.apply_rotary_emb_k(positions, k)
            attention.store_kvcache(k, v.view_as(k), attn.attn.k_cache, attn.attn.v_cache, slots)

    def forward(self, embeddings, positions):
        hidden, residual = embeddings, None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        hidden, _ = self.norm(hidden, residual)
        return hidden
