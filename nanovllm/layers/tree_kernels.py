"""GPU tree construction, traversal and overlap-safe paged KV commit.

One tree program per request, with all heap state in registers. Selection follows
the reference's best-first parent expansion and insertion-order tie breaking.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _build(Ids, Scores, Anchors, Tokens, Parents, Depths, Ancestors,
           W: tl.constexpr, DEPTH: tl.constexpr, N: tl.constexpr, S: tl.constexpr):
    b = tl.program_id(0)
    i = tl.arange(0, S)
    tokens = tl.full((S,), 0, tl.int64)
    tokens = tl.where(i == 0, tl.load(Anchors + b), tokens)
    parents = tl.full((S,), -1, tl.int32)
    depths = tl.full((S,), 0, tl.int32)
    score = tl.where(i == 0, 0., -float('inf')).to(tl.float64)
    bits = tl.where(i == 0, 1, 0).to(tl.uint64)
    expanded = tl.full((S,), False, tl.int1)
    count = 1
    running = True
    while (count < N) & running:
        eligible = (i < count) & ~expanded & (depths < DEPTH)
        best = tl.max(tl.where(eligible, score, -float('inf')), 0)
        parent = tl.min(tl.where(eligible & (score == best), i, S), 0)
        running = parent != S
        depth = tl.sum(tl.where(i == parent, depths, 0), 0)
        parent_bits = tl.sum(tl.where(i == parent, bits, 0), 0)
        rank = i - count
        adding = (rank >= 0) & (rank < W) & (i < N) & running
        offset = b * DEPTH * W + depth * W + rank
        new_token = tl.load(Ids + offset, adding, other=0)
        new_score = tl.load(Scores + offset, adding, other=0)
        tokens = tl.where(adding, new_token, tokens)
        parents = tl.where(adding, parent, parents)
        depths = tl.where(adding, depth + 1, depths)
        score = tl.where(adding, best + new_score, score)
        node_bits = parent_bits | (tl.full((S,), 1, tl.uint64) << i)
        bits = tl.where(adding, node_bits, bits)
        expanded = expanded | (i == parent)
        count += tl.where(running, W, 0)
    tl.store(Tokens + b * N + i, tokens, i < N)
    tl.store(Parents + b * N + i, parents, i < N)
    tl.store(Depths + b * N + i, depths, i < N)
    # A narrow/depth-limited tree can leave unused fixed-shape rows. Give those
    # rows a self key to avoid all-masked attention; no parent points to them.
    bits = tl.where(i < count, bits, tl.full((S,), 1, tl.uint64) << i)
    tl.store(Ancestors + b * N + i, bits, i < N)


def build_trees(anchors, logits, width, tokens, parents, depths, ancestors):
    # Normalize only the top-W values: logsumexp still covers the whole vocabulary.
    fp32 = logits.float()
    values, ids = fp32.topk(width, dim=-1)
    scores = values - fp32.logsumexp(dim=-1, keepdim=True)
    batch, depth, _ = ids.shape
    n = tokens.shape[1]
    _build[(batch,)](ids, scores, anchors, tokens, parents, depths, ancestors,
        width, depth, n, triton.next_power_of_2(n), num_warps=4)


@triton.jit
def _traverse(Tokens, Parents, Draws, Path, Output, Counts,
              N: tl.constexpr, P: tl.constexpr, S: tl.constexpr):
    b = tl.program_id(0)
    i = tl.arange(0, S)
    tokens = tl.load(Tokens + b * N + i, i < N, other=-1)
    parents = tl.load(Parents + b * N + i, i < N, other=-2)
    node = 0
    count = 0
    running = True
    for step in range(P):
        if running:
            token = tl.load(Draws + b * N + node)
            tl.store(Path + b * P + step, node)
            tl.store(Output + b * (P + 1) + step + 1, token)
            count += 1
            child = tl.min(tl.where((i > 0) & (i < N) & (parents == node) & (tokens == token), i, N), 0)
            running = child < N
            node = tl.minimum(child, N - 1)
    tl.store(Counts + b, count)
    tl.store(Output + b * (P + 1), count)


def traverse_trees(tokens, parents, draws, path, output, counts):
    _traverse[(tokens.shape[0],)](tokens, parents, draws, path, output, counts,
        tokens.shape[1], path.shape[1], triton.next_power_of_2(tokens.shape[1]), num_warps=4)


@triton.jit
def _gather(Cache, Temporary, Tables, Prefix, Path, Counts,
            TABLE_WIDTH: tl.constexpr, PAGE: tl.constexpr, PHYSICAL: tl.constexpr,
            P: tl.constexpr, B: tl.constexpr, L: tl.constexpr, D: tl.constexpr, S: tl.constexpr):
    row, layer = tl.program_id(0), tl.program_id(1)
    b, pos = row // P, row % P
    active = pos < tl.load(Counts + b)
    node = tl.load(Path + row, active, other=0)
    virtual = tl.load(Prefix + b) + node
    block = tl.load(Tables + b * TABLE_WIDTH + virtual // PAGE)
    slot = block * PAGE + virtual % PAGE
    d = tl.arange(0, S)
    value = tl.load(Cache + (layer.to(tl.int64) * PHYSICAL + slot) * D + d, active & (d < D), other=0)
    tl.store(Temporary + (layer * B * P + row) * D + d, value, d < D)


@triton.jit
def _scatter(Cache, Temporary, Tables, Prefix, Counts,
             TABLE_WIDTH: tl.constexpr, PAGE: tl.constexpr, PHYSICAL: tl.constexpr,
             P: tl.constexpr, B: tl.constexpr, D: tl.constexpr, S: tl.constexpr):
    row, layer = tl.program_id(0), tl.program_id(1)
    b, pos = row // P, row % P
    active = pos < tl.load(Counts + b)
    virtual = tl.load(Prefix + b) + pos
    block = tl.load(Tables + b * TABLE_WIDTH + virtual // PAGE)
    slot = block * PAGE + virtual % PAGE
    d = tl.arange(0, S)
    value = tl.load(Temporary + (layer * B * P + row) * D + d, d < D, other=0)
    tl.store(Cache + (layer.to(tl.int64) * PHYSICAL + slot) * D + d, value, active & (d < D))


def commit_kv(cache, temporary, tables, prefix, path, counts):
    # Two launches are intentional: no overlapping branch source is overwritten
    # until every accepted node has been gathered across every target layer.
    _, layers, pages, page, heads, dim = cache.shape
    batch, p = path.shape
    d = heads * dim
    _gather[(batch * p, 2 * layers)](cache, temporary, tables, prefix, path, counts,
        tables.shape[1], page, pages * page, p, batch, layers, d, triton.next_power_of_2(d))
    _scatter[(batch * p, 2 * layers)](cache, temporary, tables, prefix, counts,
        tables.shape[1], page, pages * page, p, batch, d, triton.next_power_of_2(d))
