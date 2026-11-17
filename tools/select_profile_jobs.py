"""Select one existing successful C=1 cell per method for separate profiling."""
import argparse
import json
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    jobs = {}
    for path in sorted(args.directory.glob("cell-*.json")):
        cell = json.loads(path.read_text(encoding="utf-8"))
        job = cell["job"]
        if cell["status"] == "ok" and not cell.get("profiled", False) and job["concurrency"] == 1:
            jobs.setdefault(job["method"], job)
    if set(jobs) != {"none", "naive", "dflash", "dspark", "jetspec"}:
        raise ValueError("expected successful C=1 cells for the baseline and all four methods")
    args.out.mkdir(parents=True, exist_ok=False)
    for method, job in jobs.items():
        # Keep the measured settings, but cap the profiling workload and trace overhead.
        job = {**job, "num_requests": 4, "max_tokens": min(job["max_tokens"], 64)}
        (args.out / f"{method}.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
