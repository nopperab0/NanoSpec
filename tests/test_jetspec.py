import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import Qwen3Config

import gpu_import_stubs  # noqa: F401
import fake_paged_attention
from nanovllm.engine.tree import DraftTree, build_tree, traverse
from nanovllm.layers.tree_attention import TreeAttention, tree_attention
from nanovllm.layers import attention
from nanovllm.models.jetspec import JetSpecHead
from nanovllm.utils.context import set_context, reset_context
from nanovllm.utils.context import get_context
from nanovllm.utils.loader import load_model


def test_head_parameters_match_real_checkpoint_header(monkeypatch):
    from nanovllm.layers.rotary_embedding import get_rope
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    data = Path(__file__).parent / "data"
    config = Qwen3Config(**json.loads((data / "jetspec_qwen3_8b_block16.config.json").read_text()))
    with torch.device("meta"):
        head = JetSpecHead(config)
    get_rope.cache_clear()
    packed = {}
    for name, shape in json.loads((data / "jetspec_qwen3_8b_block16.tensors.json").read_text()).items():
        for key, (replacement, _) in head.packed_modules_mapping.items():
            if key in name:
                name = name.replace(key, replacement)
                break
        if name not in packed:
            packed[name] = shape.copy()
        else:
            assert packed[name][1:] == shape[1:]
            packed[name][0] += shape[0]
    assert {name: list(param.shape) for name, param in head.named_parameters()} == packed
    assert head.aux_layer_ids == (1, 9, 17, 25, 33) and head.block_size == 16


def test_tree_budget_ancestors_and_no_sibling_visibility():
    tree = build_tree(7, torch.tensor([[4., 3., 1.], [3., 4., 1.], [1., 3., 4.]]), width=2, budget=9)
    assert len(tree.tokens) == 9
    mask = tree.ancestor_mask("cpu")
    for i in range(1, len(tree.tokens)):
        assert tree.parents[i] < i and tree.depths[i] == tree.depths[tree.parents[i]] + 1
        assert mask[i, i] and mask[i, tree.parents[i]] and mask[i, 0]
        assert mask[i].sum() == tree.depths[i] + 1
    siblings = [i for i, p in enumerate(tree.parents) if p == 0]
    assert len(siblings) == 2
    assert not mask[siblings[0], siblings[1]] and not mask[siblings[1], siblings[0]]


def test_leaf_and_missing_child_always_emit_a_target_tail():
    tree = DraftTree([9, 1, 2, 3], [-1, 0, 0, 2], [0, 1, 1, 2])
    path, output = traverse(tree, lambda node: {0: 2, 2: 3, 3: 8}[node])
    assert path == [0, 2, 3] and output == [2, 3, 8]
    assert traverse(tree, lambda node: 7) == ([0], [7])


def test_random_target_traversal_preserves_joint_distribution():
    # Analytical two-token distribution; candidates omit some branches.
    tree = DraftTree([9, 0, 1, 0], [-1, 0, 0, 1], [0, 1, 1, 2])
    p0 = torch.tensor([0.3, 0.7])
    p1 = [torch.tensor([0.8, 0.2]), torch.tensor([0.1, 0.9])]
    generator = torch.Generator().manual_seed(719)
    counts = torch.zeros(2, 2)
    for _ in range(12000):
        def draw(node):
            p = p0 if node == 0 else p1[tree.tokens[node]]
            return torch.multinomial(p, 1, generator=generator).item()
        _, output = traverse(tree, draw)
        if len(output) < 2:
            output.append(torch.multinomial(p1[output[0]], 1, generator=generator).item())
        counts[output[0], output[1]] += 1
    expected = p0[:, None] * torch.stack(p1)
    assert torch.allclose(counts / counts.sum(), expected, atol=0.012, rtol=0)


def test_tree_sdpa_matches_each_separate_branch_with_paged_gqa():
    torch.manual_seed(9)
    tree = DraftTree([9, 1, 2, 3], [-1, 0, 0, 2], [0, 1, 1, 2])
    prefix, n = 3, 4
    q = torch.randn(n, 4, 8)
    k_cache, v_cache = torch.randn(4, 4, 2, 8), torch.randn(4, 4, 2, 8)
    slots = torch.tensor([9, 10, 11, 0, 1, 6, 7])
    meta = TreeAttention(slots, prefix, tree.ancestor_mask("cpu"))
    actual = tree_attention(q, k_cache, v_cache, meta, 8 ** -0.5)
    keys, values = k_cache.flatten(0, 1), v_cache.flatten(0, 1)
    for i in range(n):
        path = torch.where(meta.ancestors[i])[0]
        visible = torch.cat([slots[:prefix], slots[prefix + path]])
        k = keys[visible].repeat_interleave(2, dim=1).transpose(0, 1)
        v = values[visible].repeat_interleave(2, dim=1).transpose(0, 1)
        expected = F.scaled_dot_product_attention(q[i:i+1].transpose(0, 1), k, v, scale=8 ** -0.5)
        torch.testing.assert_close(actual[i:i+1], expected.transpose(0, 1), atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(not os.environ.get("JETSPEC_SOURCE"), reason="needs official JETSPEC_SOURCE checkout")
def test_head_matches_official_with_random_tiny_weights(tmp_path, monkeypatch):
    # The official module is a numerical oracle only; the installed project never imports it.
    source = Path(os.environ["JETSPEC_SOURCE"]) / "jetspec/methods/jetspec/draft_head.py"
    spec = importlib.util.spec_from_file_location("official_jetspec_head", source)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=8, max_position_embeddings=128, vocab_size=64,
                     attention_bias=False, rope_theta=1000000, block_size=4, num_target_layers=6,
                     dflash_config=dict(causal_head=True, mask_token_id=63, target_layer_ids=[1, 3]))
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(3)
    reference = official.DFlashDraftModel(cfg).eval()
    from safetensors.torch import save_file
    save_file(reference.state_dict(), tmp_path / "model.safetensors")
    head = JetSpecHead(cfg)
    load_model(head, str(tmp_path))
    prefix, block = 5, 4
    aux, noise = torch.randn(prefix, 64), torch.randn(block, 32)
    monkeypatch.setattr(attention, "store_kvcache", fake_paged_attention.store_kvcache)
    monkeypatch.setattr(attention.Attention, "forward", fake_paged_attention.attention_forward)
    for layer in head.layers:
        layer.self_attn.attn.k_cache = torch.zeros(8, 4, 2, 8)
        layer.self_attn.attn.v_cache = torch.zeros(8, 4, 2, 8)
    with torch.inference_mode():
        expected = reference(position_ids=torch.arange(prefix + block)[None], noise_embedding=noise[None],
                             target_hidden=aux[None], use_cache=False, is_causal=True)[0]
        head.store_context(aux, torch.arange(prefix), torch.arange(prefix, dtype=torch.int32))
        set_context(False, slot_mapping=torch.arange(prefix, prefix + block, dtype=torch.int32),
                    context_lens=torch.tensor([prefix + block], dtype=torch.int32),
                    block_tables=torch.arange(8, dtype=torch.int32)[None], query_len_per_seq=block, causal=True)
        try:
            actual = head(noise, torch.arange(prefix, prefix + block))
        finally:
            reset_context()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_real_round_gathers_noncontiguous_overlapping_path_and_aux():
    from collections import Counter
    from nanovllm.engine.jetspec import JetSpecDecoder
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams
    decoder = object.__new__(JetSpecDecoder)
    decoder.device, decoder.block_size = torch.device("cpu"), 4
    decoder.aux_width = 2
    decoder.config = SimpleNamespace(hf_config=SimpleNamespace(dtype=torch.float32))
    decoder.stats, decoder._phase_events, decoder._new_event = Counter(), [], None
    cache = torch.zeros(2, 1, 4, 4, 1, 1)
    class Model:
        def __call__(self, ids, positions, aux):
            slots = get_context().slot_mapping.long()
            cache.flatten(2, 3)[:, :, slots, 0, 0] = ids.float()
            aux.copy_(torch.stack([positions, ids], -1))
            logits = torch.full((4, 16), -100.0)
            logits[torch.arange(4), torch.tensor([5, 10, 6, 7])] = 100
            return logits
        def compute_logits(self, hidden, all_logits=False):
            return hidden
    decoder.target = SimpleNamespace(model=Model(), kv_cache=cache)
    stored = []
    decoder.head = SimpleNamespace(store_context=lambda aux, positions, slots: stored.append((aux.clone(), positions, slots)))
    tree = DraftTree([3, 4, 5, 6], [-1, 0, 0, 2], [0, 1, 1, 2])
    decoder._propose = lambda seq: tree
    seq = Sequence([1, 2, 3], SamplingParams(temperature=0))
    seq.block_table = [3, 0, 2, 1]
    seq.num_cached_tokens = 2
    assert decoder.run_round([seq]) == [[5, 6, 7]]
    assert cache.flatten(2, 3)[0, 0, [14, 15, 0], 0, 0].tolist() == [3, 5, 6]
    assert stored[0][0].tolist() == [[2, 3], [3, 5], [4, 6]]
    assert stored[0][1].tolist() == [2, 3, 4] and stored[0][2].tolist() == [14, 15, 0]
    assert decoder.stats["accepted_2"] == 1 and decoder.stats["proposed_nodes"] == 3
