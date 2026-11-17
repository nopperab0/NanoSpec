import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench_spec import plan, report
from benchmarks.common import METHODS, engine_kwargs, load_prompts


def options(tmp_path):
    panel = tmp_path / "prompts.json"
    panel.write_text(json.dumps(["one", "two"]))
    return SimpleNamespace(prompts=str(panel), target="target", methods=list(METHODS),
        naive_draft="small", dflash_draft="df", dspark_draft="ds", jetspec_draft="jet",
        seeds=[1, 2], repeats=2, concurrency=[1, 2], naive_k=7, max_tokens=8, temperature=0.7,
        top_k=20, top_p=0.8, max_model_len=128, max_num_batched_tokens=128, target_kv_gib=1,
        draft_kv_gib=1, eager=True, warmup=1, order_seed=13, tree_width=4, tree_budget=32)


def test_all_four_methods_have_correct_distinct_engine_options(tmp_path):
    args = options(tmp_path)
    jobs = plan(args)
    assert len(jobs) == 40 and jobs == plan(args)
    by_method = {j["method"]: engine_kwargs(j) for j in jobs}
    assert by_method["none"]["method"] == "none" and "draft_model" not in by_method["none"]
    assert by_method["naive"]["num_speculative_tokens"] == 7
    assert by_method["jetspec"]["jetspec_tree_budget"] == 32
    assert all(not kw["enable_prefix_cache"] for kw in by_method.values())


def test_report_uses_paired_ratios_and_rejects_missing_cells(tmp_path):
    jobs = plan(options(tmp_path))
    (tmp_path / "manifest.json").write_text(json.dumps({"jobs": jobs}))
    for i, job in enumerate(jobs):
        row = dict(job=job, status="ok", wall_tok_s=100 if job["method"] == "none" else 150,
                   acceptance_rate=None, source_sha256="same", environment={"gpu": "same"},
                   checkpoint_config_sha256={"target": "same"})
        (tmp_path / f"cell-{i:04d}.json").write_text(json.dumps(row))
    result = report(tmp_path)
    assert "1.500 [1.500, 1.500]" in result and "jetspec" in result
    (tmp_path / "cell-0000.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        report(tmp_path)


def test_panel_validation(tmp_path):
    panel = tmp_path / "bad.json"
    for value in ([], [""], [{"prompt_token_ids": [-1]}], [None]):
        panel.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            load_prompts(panel)


def test_greedy_gate_catches_divergence(tmp_path):
    from tools.check_greedy import check
    args = options(tmp_path)
    args.methods, args.seeds, args.concurrency, args.repeats = ["none", "jetspec"], [17], [1], 1
    args.temperature = 0
    jobs = plan(args)
    (tmp_path / "manifest.json").write_text(json.dumps({"jobs": jobs}))
    for i, job in enumerate(jobs):
        result = dict(job=job, status="ok", wall_tok_s=100, acceptance_rate=None, source_sha256="s",
                      environment={}, checkpoint_config_sha256={"target": "t"},
                      outputs=[{"token_ids": [1, 2] if job["method"] == "none" else [1, 3]}])
        (tmp_path / f"cell-{i:04d}.json").write_text(json.dumps(result))
    gate = check(tmp_path)
    assert not gate["passed"] and gate["differences"][0]["position"] == 1
