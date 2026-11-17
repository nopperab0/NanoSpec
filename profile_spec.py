"""Profile one planned benchmark cell in a separate process."""
import argparse
import json
from pathlib import Path

from bench_spec import run_cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["torch", "external"], default="torch")
    parser.add_argument("--wait", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if args.wait < 0 or args.steps < 1:
        parser.error("wait must be nonnegative and steps positive")
    args.out.mkdir(parents=True, exist_ok=False)
    job = json.loads(args.job.read_text(encoding="utf-8"))
    job.update(profile_mode=args.mode, profile_dir=str(args.out.resolve()),
               profile_wait=args.wait, profile_steps=args.steps)
    result = run_cell(job)
    (args.out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
