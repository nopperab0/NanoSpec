"""JetSpec cumulative-logprob tree and target-distribution traversal.

This implementation is independent of the official JetSpec Python package.
"""
from dataclasses import dataclass
import heapq

import torch


@dataclass
class DraftTree:
    tokens: list[int]
    parents: list[int]
    depths: list[int]

    def ancestor_mask(self, device):
        n = len(self.tokens)
        mask = torch.zeros(n, n, dtype=torch.bool, device=device)
        for i in range(n):
            j = i
            while j >= 0:
                mask[i, j] = True
                j = self.parents[j]
        return mask

    def children(self):
        maps = [{} for _ in self.tokens]
        for i in range(1, len(self.tokens)):
            maps[self.parents[i]][self.tokens[i]] = i
        return maps


def build_tree(anchor, logits, width, budget):
    """One causal-head pass -> per-depth top-W -> best-first expansion.

    Nodes are inserted after parents. Budget includes the root. Same cumulative
    log-probability baseline as JetSpec's accum_logp, with deterministic tie order.
    """
    if logits.ndim != 2 or width < 1 or budget < 1:
        raise ValueError("expected [depth, vocab] logits, positive width and budget")
    values, ids = logits.float().log_softmax(-1).topk(min(width, logits.shape[-1]), dim=-1)
    top_ids, top_values = ids.tolist(), values.tolist()
    tree = DraftTree([int(anchor)], [-1], [0])
    heap = [(0.0, 0)]
    while heap and len(tree.tokens) < budget:
        negative_score, parent = heapq.heappop(heap)
        depth = tree.depths[parent]
        if depth == logits.shape[0]:
            continue
        for token, score in zip(top_ids[depth], top_values[depth]):
            if len(tree.tokens) == budget:
                break
            node = len(tree.tokens)
            tree.tokens.append(token)
            tree.parents.append(parent)
            tree.depths.append(depth + 1)
            heapq.heappush(heap, (negative_score - score, node))
    return tree


def traverse(tree, draw):
    """Draw from target p at each visited node; continue if that token is a child.

    This exact target-sampling variant uses the tree as a lookup accelerator.
    It never samples from a tree-conditioned or renormalized target distribution.
    Returns cached node path (root included), and accepted drafts + uncached tail.
    It is deliberately distinct from the chain p/q rejection sampler.
    """
    children = tree.children()
    node, path, output = 0, [0], []
    while True:
        token = int(draw(node))
        output.append(token)
        child = children[node].get(token)
        if child is None:
            return path, output
        node = child
        path.append(node)
