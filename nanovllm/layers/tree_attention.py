"""Readable JetSpec tree attention. Only JetSpec selects this eager SDPA path."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TreeAttention:
    slots: torch.Tensor       # physical KV slots: cached prefix followed by tree nodes
    prefix_len: int
    ancestors: torch.Tensor   # [N, N], self and ancestors visible, siblings invisible


def tree_attention(q, k_cache, v_cache, metadata, scale):
    keys = k_cache.flatten(0, 1)[metadata.slots]
    values = v_cache.flatten(0, 1)[metadata.slots]
    prefix = torch.ones(q.shape[0], metadata.prefix_len, dtype=torch.bool, device=q.device)
    allowed = torch.cat([prefix, metadata.ancestors], dim=-1)
    # Repeat GQA heads explicitly for a transparent reference implementation.
    groups = q.shape[1] // keys.shape[1]
    keys = keys.repeat_interleave(groups, dim=1)
    values = values.repeat_interleave(groups, dim=1)
    output = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0),
        keys.transpose(0, 1).unsqueeze(0), values.transpose(0, 1).unsqueeze(0),
        attn_mask=allowed.unsqueeze(0).unsqueeze(0), dropout_p=0, scale=scale)
    return output.squeeze(0).transpose(0, 1).contiguous()
