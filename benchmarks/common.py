import hashlib
import json
from pathlib import Path

METHODS = ("none", "naive", "dflash", "dspark", "jetspec")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_prompts(path):
    """JSON list of strings, or records with prompt / prompt_token_ids."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("prompts must be a nonempty JSON list")
    prompts = [r if isinstance(r, (str, list)) else r.get("prompt_token_ids", r.get("prompt"))
               if isinstance(r, dict) else None for r in rows]
    for p in prompts:
        if isinstance(p, str) and p.strip():
            continue
        if isinstance(p, list) and p and all(type(i) is int and i >= 0 for i in p):
            continue
        raise ValueError("each prompt must be nonempty text or nonnegative token IDs")
    return prompts


def engine_kwargs(job):
    method = job["method"]
    if method not in METHODS:
        raise ValueError(f"unsupported method {method}")
    options = dict(method=method, max_num_seqs=job["concurrency"],
                   max_model_len=job["max_model_len"], max_num_batched_tokens=job["max_num_batched_tokens"],
                   kv_cache_memory_bytes=job["target_kv_bytes"], enable_prefix_cache=False,
                   enforce_eager=job["eager"], seed=job["seed"], record_step_timings=True,
                   record_request_events=True)
    if method != "none":
        options["draft_model"] = job["drafts"][method]
    if method == "naive":
        options.update(num_speculative_tokens=job["naive_k"], draft_kv_cache_memory_bytes=job["draft_kv_bytes"])
    if method == "jetspec":
        options.update(jetspec_tree_width=job["tree_width"], jetspec_tree_budget=job["tree_budget"])
        options.update(jetspec_backend=job.get("jetspec_backend", "triton"),
                       jetspec_cuda_graph=job.get("jetspec_cuda_graph", False))
    return options


def source_hash(root):
    h = hashlib.sha256()
    for path in sorted(Path(root).rglob("*.py")):
        if "__pycache__" not in path.parts:
            h.update(path.relative_to(root).as_posix().encode())
            h.update(path.read_bytes())
    return h.hexdigest()
