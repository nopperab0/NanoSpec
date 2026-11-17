"""Run one selected seed/pass, reusing exact completed cells from an existing matrix."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench_spec import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--pass-index", type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads((args.source / "manifest.json").read_text())
    jobs = [job for job in manifest["jobs"] if job["seed"] == args.seed and job["repeat"] == args.pass_index]
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "manifest.json").write_text(json.dumps(dict(schema_version=2, jobs=jobs,
        selection="one seed and one pass only; original pass identifier retained", source=str(args.source)), indent=2))
    canonical = lambda job: json.dumps(job, sort_keys=True)
    current_gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,power.limit",
        "--format=csv,noheader"], text=True).strip()
    completed = {}
    for path in args.source.glob("cell-*.json"):
        cell = json.loads(path.read_text())
        if cell["status"] == "ok" and cell["environment"].get("nvidia_smi") == current_gpu:
            completed[canonical(cell["job"])] = path
    for i, job in enumerate(jobs):
        destination = args.out / f"cell-{i:04d}.json"
        existing = completed.get(canonical(job))
        if existing is not None:
            shutil.copyfile(existing, destination)
            print(f"[{i+1}/{len(jobs)}] reuse {job['method']} C={job['concurrency']} seed={job['seed']}", flush=True)
            continue
        job_file = args.out / f"job-{i:04d}.json"
        job_file.write_text(json.dumps(job))
        print(f"[{i+1}/{len(jobs)}] run {job['method']} C={job['concurrency']} seed={job['seed']}", flush=True)
        with (args.out / f"cell-{i:04d}.log").open("w") as log:
            subprocess.run([sys.executable, str(ROOT / "bench_spec.py"), "worker", "--job", str(job_file),
                "--out", str(destination)], check=True, stdout=log, stderr=subprocess.STDOUT)
    (args.out / "REPORT.md").write_text(report(args.out), encoding="utf-8")
    with (args.out / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "concurrency", "seed", "wall_tok_s", "steady_tok_s", "ttft_p99_ms",
            "latency_p99_ms", "tpot_p50_ms", "gap_p99_ms", "goodput_requests_s", "peak_allocated_bytes"])
        for path in sorted(args.out.glob("cell-*.json")):
            cell = json.loads(path.read_text())
            job, metrics = cell["job"], cell["metrics"]
            writer.writerow([job["method"], job["concurrency"], job["seed"], cell["wall_tok_s"],
                metrics["steady_tok_s"], metrics["ttft_ms"]["p99"], metrics["latency_ms"]["p99"],
                metrics["tpot_ms"]["p50"], metrics["delivery_gap_ms"]["p99"],
                metrics["goodput_requests_s"], cell["peak_allocated_bytes"]])


if __name__ == "__main__":
    main()
