import json

import pytest

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from test_config import TINY_QWEN3


@pytest.fixture
def paths(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(json.dumps(TINY_QWEN3))
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps({**TINY_QWEN3, "block_size": 4,
        "dflash_config": {"causal_head": True, "mask_token_id": 127, "target_layer_ids": [0]}}))
    return str(target), str(draft)


def test_naive_switch_defaults_to_small_model_k3(paths):
    target, _ = paths
    cfg = Config(target, method="naive", draft_model=target, enable_prefix_cache=False,
                 kv_cache_memory_bytes=1<<20, draft_kv_cache_memory_bytes=1<<20)
    assert cfg.method == "naive" and cfg.draft_type == "model" and cfg.num_speculative_tokens == 3


def test_jetspec_requires_causal_head_and_eager_and_reserves_tree_scratch(paths):
    target, draft = paths
    kwargs = dict(method="jetspec", draft_model=draft, enable_prefix_cache=False,
                  kv_cache_memory_bytes=1<<20, enforce_eager=True)
    cfg = Config(target, **kwargs, jetspec_tree_budget=9)
    assert cfg.draft_type == "jetspec" and cfg.num_speculative_tokens == 8
    with pytest.raises(ValueError, match="eager"):
        Config(target, **{**kwargs, "enforce_eager": False})
    with pytest.raises(ValueError, match="causal"):
        Config(target, **{**kwargs, "draft_model": target})


@pytest.mark.parametrize("method", ["naive", "dflash", "dspark", "jetspec"])
def test_no_silent_baseline_fallback(paths, method):
    with pytest.raises(ValueError, match="draft_model"):
        Config(paths[0], method=method)


def test_none_rejects_draft_and_unknown_method(paths):
    with pytest.raises(ValueError):
        Config(paths[0], method="none", draft_model=paths[1])
    with pytest.raises(ValueError):
        Config(paths[0], method="typo")


@pytest.mark.parametrize("kwargs", [{"max_tokens": 0}, {"max_tokens": -1}, {"temperature": float("inf")}, {"top_k": 1.5}])
def test_invalid_generation_settings_fail_early(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)
