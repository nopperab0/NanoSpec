"""Batched paged tree attention: online softmax, native GQA, no prefix KV gather.

Each CTA processes a query tile for one head and request. Prefix positions are
unconditionally visible; scratch nodes require their ancestor bit to be set.
The ordinary nano attention backend is not changed.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class PagedTreeAttention:
    block_tables: torch.Tensor
    prefix_lens: torch.Tensor
    ancestors: torch.Tensor
    nodes: int


@triton.jit
def _attention(Q, K, V, O, Tables, Prefix, Ancestors,
               N: tl.constexpr, H: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
               QROW: tl.constexpr, QHEAD: tl.constexpr, KPAGE: tl.constexpr,
               KSLOT: tl.constexpr, KHEAD: tl.constexpr, TABLE_WIDTH: tl.constexpr,
               PAGE: tl.constexpr, SCALE: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    tile, head, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qi = tile * BM + tl.arange(0, BM)
    kj = tl.arange(0, BN)
    dim = tl.arange(0, BD)
    prefix = tl.load(Prefix + b)
    bits = tl.load(Ancestors + b * N + qi, qi < N, other=0)
    q = tl.load(Q + (b * N + qi[:, None]) * QROW + head * QHEAD + dim[None, :],
        (qi[:, None] < N) & (dim[None, :] < D), other=0)
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    denominator = tl.full((BM,), 0., tl.float32)
    acc = tl.full((BM, BD), 0., tl.float32)
    kv_head = head // (H // HKV)
    for start in tl.range(0, tl.cdiv(prefix + N, BN)):
        positions = start * BN + kj
        valid = positions < prefix + N
        page = tl.load(Tables + b * TABLE_WIDTH + positions // PAGE, valid, other=0)
        offsets = page * KPAGE + (positions % PAGE) * KSLOT + kv_head * KHEAD
        k = tl.load(K + offsets[None, :] + dim[:, None], valid[None, :] & (dim[:, None] < D), other=0)
        v = tl.load(V + offsets[:, None] + dim[None, :], valid[:, None] & (dim[None, :] < D), other=0)
        node = positions - prefix
        allowed = (positions[None, :] < prefix) | (((bits[:, None] >> tl.maximum(node[None, :], 0)) & 1) != 0)
        scores = tl.dot(q, k).to(tl.float32) * SCALE
        scores = tl.where(valid[None, :] & allowed & (qi[:, None] < N), scores, -float('inf'))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        # Padded query lanes have no visible keys. Keep them finite internally.
        next_max = tl.where(qi < N, next_max, 0.)
        probability = tl.exp(scores - next_max[:, None])
        alpha = tl.exp(maximum - next_max)
        acc = acc * alpha[:, None]
        acc = tl.dot(probability.to(v.dtype), v, acc)
        denominator = denominator * alpha + tl.sum(probability, 1)
        maximum = next_max
    output = acc / tl.maximum(denominator[:, None], 1.e-20)
    tl.store(O + ((b * N + qi[:, None]) * H + head) * D + dim[None, :],
        output, (qi[:, None] < N) & (dim[None, :] < D))


def paged_tree_attention(q, k_cache, v_cache, metadata, scale):
    n = metadata.nodes
    batch = metadata.prefix_lens.numel()
    heads, dim = q.shape[1:]
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _attention[(triton.cdiv(n, 16), heads, batch)](q, k_cache, v_cache, output,
        metadata.block_tables, metadata.prefix_lens, metadata.ancestors,
        n, heads, k_cache.shape[2], dim, q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        metadata.block_tables.shape[1], k_cache.shape[1], scale,
        16, 64, triton.next_power_of_2(dim), num_warps=4, num_stages=2)
    return output
