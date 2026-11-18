"""Four speculative methods plus AR baseline, each cell in a fresh GPU process."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import statistics
import subprocess
import sys
from pathlib import Path

from benchmarks.common import METHODS, engine_kwargs, load_prompts, sha256, source_hash
from benchmarks.metrics import distribution
from benchmarks.runner import run_requests
from benchmarks.profiling import Profiler

ROOT = Path(__file__).resolve().parent


def run_cell(job):
    import torch
    import transformers
    from nanovllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("GPU benchmark requires CUDA, FlashAttention and Triton")
    prompts = load_prompts(job["prompts"])
    if sha256(job["prompts"]) != job["prompts_sha256"]:
        raise ValueError("prompt panel changed after planning")
    llm = LLM(job["target"], **engine_kwargs(job))
    try:
        prompts = [llm.tokenizer.encode(p) if isinstance(p, str) else p for p in prompts]
        if any(max(p) >= llm.config.hf_config.vocab_size for p in prompts):
            raise ValueError("prompt token IDs exceed target vocabulary")
        params = SamplingParams(temperature=job["temperature"], top_k=job["top_k"], top_p=job["top_p"],
                                max_tokens=job["max_tokens"], ignore_eos=True)
        # Warm actual shapes, including draft and verify; exclude from all measurements.
        for _ in range(job["warmup"]):
            llm.generate(prompts[:job["concurrency"]], params, use_tqdm=False)
        llm.model_runner.generator.manual_seed(job["seed"])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with Profiler(llm, job) as profiler:
            data = run_requests(llm, prompts, params, job, profiler, synchronize=torch.cuda.synchronize)
        elapsed = data["wall_seconds"]
        outputs = data["outputs"]
        steps = llm.step_timings()
        phases = llm.phase_timings()
        tokens = sum(len(o["token_ids"]) for o in outputs)
        stats = dict(llm.speculative.stats) if llm.speculative else {}
        counts = {int(k.split("_")[1]): v for k, v in stats.items() if k.startswith("accepted_")}
        rounds = sum(counts.values())
        accepted = sum(k * v for k, v in counts.items())
        k = llm.config.num_speculative_tokens
        env = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), platform=platform.platform())
        for package in ("flash-attn", "triton"):
            try:
                env[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                env[package] = None
        try:
            env["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,power.limit",
                                                       "--format=csv,noheader"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            env["nvidia_smi"] = None
        checkpoint_configs = {"target": sha256(Path(job["target"]) / "config.json")}
        if job["method"] != "none":
            checkpoint_configs["draft"] = sha256(Path(job["drafts"][job["method"]]) / "config.json")
        checkpoint_sources = {}
        directories = {"target": job["target"]}
        if job["method"] != "none":
            directories["draft"] = job["drafts"][job["method"]]
        for role, directory in directories.items():
            marker = Path(directory) / "SOURCE.json"
            checkpoint_sources[role] = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
        phases_by_name = {}
        for phase in phases:
            phases_by_name.setdefault(phase["phase"], []).append(phase["duration_ms"])
        data["metrics"]["phase_ms"] = {name: distribution(values) for name, values in phases_by_name.items()}
        committed = data["metrics"]["committed_decode_tokens"]
        data["metrics"]["verified_positions_per_committed_token"] = stats.get("verified_positions", 0) / committed if committed and k else None
        data["metrics"]["phase_us_per_committed_token"] = {
            name: sum(values) * 1000 / committed for name, values in phases_by_name.items()} if committed else {}
        data["metrics"]["accepted_prefix_survival"] = [
            sum(count for length, count in counts.items() if length >= position) / rounds
            for position in range(1, max(counts, default=0) + 1)] if rounds else []
        return dict(schema_version=2, job=job, status="ok", environment=env, **data,
                    profiled=job.get("profile_mode", "none") != "none",
                    checkpoint_sources=checkpoint_sources,
                    checkpoint_config_sha256=checkpoint_configs, source_sha256=source_hash(ROOT / "nanovllm"),
                    benchmark_source_sha256=hashlib.sha256((source_hash(ROOT / "benchmarks") +
                        sha256(ROOT / "bench_spec.py") + sha256(ROOT / "profile_spec.py")).encode()).hexdigest(),
                    output_tokens=tokens, wall_tok_s=tokens / elapsed,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    accepted_drafts=accepted, sequence_rounds=rounds,
                    acceptance_rate=accepted / stats.get("proposed_nodes", rounds * k) if rounds and k else None,
                    tokens_per_round_before_stop=(accepted + rounds) / rounds if rounds else None,
                    num_speculative_tokens=k, stats=stats, steps=steps, phases=phases)
    finally:
        llm.exit()


def plan(args):
    prompts = Path(args.prompts).resolve()
    rows = load_prompts(prompts)
    if args.warmup < 1 or args.repeats < 1 or args.max_tokens < 1 or args.naive_k < 1:
        raise ValueError("warmup, repeats, max_tokens and naive_k must be positive")
    if args.tree_budget < 2 or args.tree_width < 1:
        raise ValueError("tree budget must be >= 2 and tree width >= 1")
    if "jetspec" in args.methods and not args.eager:
        raise ValueError("JetSpec tutorial uses eager tree attention; use --eager for the four-method comparison")
    if any(len(v) != len(set(v)) for v in (args.methods, args.seeds, args.concurrency)):
        raise ValueError("methods, seeds and concurrency must not contain duplicates")
    if args.target_kv_gib <= 0 or args.draft_kv_gib <= 0:
        raise ValueError("KV budgets must be positive")
    if any(c < 1 or c > (getattr(args, "num_requests", 0) or len(rows)) or c > 512 for c in args.concurrency):
        raise ValueError("concurrency must be in 1..min(number of prompts, 512)")
    drafts = {m: str(Path(getattr(args, m + "_draft")).resolve()) for m in args.methods if m != "none"
              and getattr(args, m + "_draft")}
    if any(m != "none" and m not in drafts for m in args.methods):
        raise ValueError("each selected speculative method requires its --METHOD-draft")
    if "none" not in args.methods:
        raise ValueError("matrix requires 'none' for paired baseline comparisons")
    load = getattr(args, "load", "closed")
    num_requests = getattr(args, "num_requests", 0)
    arrival_rate = getattr(args, "arrival_rate", 1.0)
    if num_requests < 0 or not math.isfinite(arrival_rate) or arrival_rate <= 0:
        raise ValueError("num_requests must be nonnegative and arrival_rate positive")
    from nanovllm import SamplingParams
    SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, top_k=args.top_k, top_p=args.top_p)
    for name in ("slo_ttft_ms", "slo_tpot_ms", "slo_gap_ms"):
        value = getattr(args, name, 1000)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("SLO thresholds must be finite and positive")
    jobs = []
    for seed in args.seeds:
        for repeat in range(args.repeats):
            for c in args.concurrency:
                for method in args.methods:
                    jobs.append(dict(method=method, target=str(Path(args.target).resolve()), drafts=drafts,
                                     prompts=str(prompts), prompts_sha256=sha256(prompts), concurrency=c,
                                     seed=seed, repeat=repeat, naive_k=args.naive_k,
                                     tree_width=args.tree_width, tree_budget=args.tree_budget,
                                     max_tokens=args.max_tokens, temperature=args.temperature,
                                     top_k=args.top_k, top_p=args.top_p, max_model_len=args.max_model_len,
                                     max_num_batched_tokens=args.max_num_batched_tokens,
                                     target_kv_bytes=int(args.target_kv_gib * (1 << 30)),
                                     draft_kv_bytes=int(args.draft_kv_gib * (1 << 30)),
                                     eager=args.eager, warmup=args.warmup))
                    jobs[-1].update(load=load, num_requests=num_requests, arrival_rate=arrival_rate,
                        slo_ttft_ms=getattr(args, "slo_ttft_ms", 1000),
                        slo_tpot_ms=getattr(args, "slo_tpot_ms", 50), slo_gap_ms=getattr(args, "slo_gap_ms", 100))
                    if method == "jetspec":
                        jobs[-1].update(jetspec_backend=getattr(args, "jetspec_backend", "triton"),
                            jetspec_cuda_graph=getattr(args, "jetspec_cuda_graph", False))
    random.Random(args.order_seed).shuffle(jobs)
    return jobs


def report(directory):
    cells = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("cell-*.json"))]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if len(cells) != len(manifest["jobs"]) or any(c["status"] != "ok" for c in cells):
        raise ValueError("benchmark matrix is incomplete; inspect cell logs")
    if any(c.get("profiled", False) for c in cells):
        raise ValueError("profiled cells cannot be used in throughput reports")
    canonical = lambda j: json.dumps(j, sort_keys=True)
    if sorted(canonical(c["job"]) for c in cells) != sorted(canonical(j) for j in manifest["jobs"]):
        raise ValueError("result cells do not match the planned jobs")
    baselines = {(r["job"]["concurrency"], r["job"]["seed"], r["job"]["repeat"]): r
                 for r in cells if r["status"] == "ok" and r["job"]["method"] == "none"}
    groups = {}
    for r in cells:
        if r["status"] != "ok":
            continue
        j = r["job"]
        base = baselines.get((j["concurrency"], j["seed"], j["repeat"]))
        if base is None:
            raise ValueError("unpaired benchmark cell")
        comparable = ("prompts_sha256", "target", "max_tokens", "temperature", "top_k", "top_p", "eager",
                      "target_kv_bytes", "max_model_len", "max_num_batched_tokens", "warmup",
                      "load", "num_requests", "arrival_rate", "slo_ttft_ms", "slo_tpot_ms", "slo_gap_ms")
        if any(j[key] != base["job"][key] for key in comparable) or r["source_sha256"] != base["source_sha256"]:
            raise ValueError("paired cells have different workloads or engine sources")
        if r.get("benchmark_source_sha256") != base.get("benchmark_source_sha256"):
            raise ValueError("paired cells have different benchmark sources")
        if r["environment"] != base["environment"] or r["checkpoint_config_sha256"]["target"] != base["checkpoint_config_sha256"]["target"]:
            raise ValueError("paired cells have different hardware/software or target configs")
        if r.get("checkpoint_sources", {}).get("target") != base.get("checkpoint_sources", {}).get("target"):
            raise ValueError("paired cells use different target snapshot revisions")
        groups.setdefault((j["concurrency"], j["method"]), []).append((r, r["wall_tok_s"] / base["wall_tok_s"]))
    lines = ["# Local benchmark results", "", "Host-observed token delivery, including prefill and scheduling; excluding initialization, warmup and text detokenization.", "",
             "| C | method | n | median tok/s | paired speedup median [min, max] | accepted/proposed |", "|---:|---|---:|---:|---:|---|---:|"]
    for (c, m), values in sorted(groups.items()):
        ratios = [v for _, v in values]
        acceptance = [r["acceptance_rate"] for r, _ in values if r["acceptance_rate"] is not None]
        rate = f"{statistics.median(acceptance):.3f}" if acceptance else "—"
        lines.append(f"| {c} | {m} | {len(values)} | {statistics.median(r['wall_tok_s'] for r, _ in values):.2f} | "
                     f"{statistics.median(ratios):.3f} [{min(ratios):.3f}, {max(ratios):.3f}] | {rate} |")
    lines += ["", "## Infrastructure metrics", "",
        "| C | method | steady tok/s | TTFT p50/p99 ms | latency p50/p99 ms | TPOT p50 ms | gap p99 ms | goodput req/s | peak GiB |",
        "|---:|---|---:|---|---|---:|---:|---:|---:|"]
    def median(values):
        values = [v for v in values if v is not None]
        return f"{statistics.median(values):.2f}" if values else "N/A"
    for (c, method), values in sorted(groups.items()):
        rows = [r for r, _ in values if "metrics" in r]
        if not rows:
            continue
        def metric(name, field=None):
            return median([r["metrics"][name][field] if field else r["metrics"].get(name) for r in rows])
        lines.append(f"| {c} | {method} | {metric('steady_tok_s')} | "
            f"{metric('ttft_ms', 'p50')}/{metric('ttft_ms', 'p99')} | "
            f"{metric('latency_ms', 'p50')}/{metric('latency_ms', 'p99')} | "
            f"{metric('tpot_ms', 'p50')} | {metric('delivery_gap_ms', 'p99')} | "
            f"{metric('goodput_requests_s')} | {median([r['peak_allocated_bytes'] / (1 << 30) for r in rows])} |")
    lines += ["", "Latency columns are medians of cell quantiles, not pooled quantiles. Raw per-request records and cell distributions are retained.",
        "Steady throughput is available only for the explicitly bounded full-concurrency closed-loop window. N/A means no valid window.",
        "These ratios describe this workload and implementation; they do not establish statistical significance."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="plan and run a randomized, paired benchmark matrix")
    run.add_argument("--target", required=True)
    for m in METHODS[1:]:
        run.add_argument(f"--{m}-draft")
    run.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    run.add_argument("--prompts", required=True)
    run.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
    run.add_argument("--seeds", type=int, nargs="+", default=[29])
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--order-seed", type=int, default=2026)
    run.add_argument("--naive-k", type=int, default=7)
    run.add_argument("--tree-width", type=int, default=4)
    run.add_argument("--tree-budget", type=int, default=32)
    run.add_argument("--jetspec-backend", choices=["reference", "triton"], default="triton")
    run.add_argument("--jetspec-cuda-graph", action="store_true")
    run.add_argument("--max-tokens", type=int, default=128)
    run.add_argument("--temperature", type=float, default=0.7)
    run.add_argument("--top-k", type=int, default=20)
    run.add_argument("--top-p", type=float, default=0.8)
    run.add_argument("--max-model-len", type=int, default=4096)
    run.add_argument("--max-num-batched-tokens", type=int, default=16384)
    run.add_argument("--target-kv-gib", type=float, default=4)
    run.add_argument("--draft-kv-gib", type=float, default=1)
    run.add_argument("--warmup", type=int, default=1)
    run.add_argument("--load", choices=["bulk", "closed", "rate"], default="closed")
    run.add_argument("--num-requests", type=int, default=0, help="0 uses the panel once; larger counts cycle through it")
    run.add_argument("--arrival-rate", type=float, default=1.0, help="requests/s in rate mode")
    run.add_argument("--slo-ttft-ms", type=float, default=1000)
    run.add_argument("--slo-tpot-ms", type=float, default=50)
    run.add_argument("--slo-gap-ms", type=float, default=100)
    run.add_argument("--eager", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--out", type=Path, required=True)
    worker = sub.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--job", type=Path, required=True)
    worker.add_argument("--out", type=Path, required=True)
    rep = sub.add_parser("report")
    rep.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        result = run_cell(json.loads(args.job.read_text(encoding="utf-8")))
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.command == "report":
        print(report(args.directory))
    else:
        jobs = plan(args)
        args.out = args.out.resolve()
        args.out.mkdir(parents=True, exist_ok=True)
        if any(args.out.iterdir()):
            raise ValueError("output directory must be empty; use a new run directory")
        (args.out / "manifest.json").write_text(json.dumps(dict(schema_version=1, jobs=jobs), indent=2), encoding="utf-8")
        print(f"Planned {len(jobs)} cells in {args.out}", flush=True)
        if args.dry_run:
            return
        for i, job in enumerate(jobs):
            job_file = args.out / f"job-{i:04d}.json"
            result_file = args.out / f"cell-{i:04d}.json"
            job_file.write_text(json.dumps(job), encoding="utf-8")
            print(f"[{i+1}/{len(jobs)}] {job['method']} C={job['concurrency']} seed={job['seed']}", flush=True)
            with (args.out / f"cell-{i:04d}.log").open("w", encoding="utf-8") as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--job", str(job_file),
                                "--out", str(result_file)], check=True, stdout=log, stderr=subprocess.STDOUT)
        (args.out / "REPORT.md").write_text(report(args.out), encoding="utf-8")
        cells = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.out.glob("cell-*.json"))]
        import csv
        with (args.out / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["method", "concurrency", "seed", "repeat", "wall_tok_s",
                "steady_tok_s", "ttft_p99_ms", "latency_p99_ms", "tpot_p50_ms", "gap_p99_ms", "goodput_requests_s",
                "peak_allocated_bytes", "peak_reserved_bytes"])
            writer.writeheader()
            for cell in cells:
                m = cell["metrics"]
                writer.writerow({**{k: cell["job"][k] for k in ("method", "concurrency", "seed", "repeat")},
                    "wall_tok_s": cell["wall_tok_s"], "steady_tok_s": m["steady_tok_s"],
                    "ttft_p99_ms": m["ttft_ms"]["p99"], "latency_p99_ms": m["latency_ms"]["p99"],
                    "tpot_p50_ms": m["tpot_ms"]["p50"], "gap_p99_ms": m["delivery_gap_ms"]["p99"],
                    "goodput_requests_s": m["goodput_requests_s"], "peak_allocated_bytes": cell["peak_allocated_bytes"],
                    "peak_reserved_bytes": cell["peak_reserved_bytes"]})


if __name__ == "__main__":
    main()
