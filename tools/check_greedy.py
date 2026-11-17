"""Require token equality to AR for every greedy benchmark cell; fail on any divergence."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench_spec import report


def check(directory):
    report(directory)  # also validates completeness and pairing
    cells = [json.loads(p.read_text(encoding="utf-8")) for p in directory.glob("cell-*.json")]
    if any(c["job"]["temperature"] != 0 for c in cells):
        raise ValueError("greedy equality requires temperature=0; equal random seeds do not imply equal token streams")
    key = lambda c: (c["job"]["concurrency"], c["job"]["seed"], c["job"]["repeat"])
    baselines = {key(c): c for c in cells if c["job"]["method"] == "none"}
    differences = []
    for cell in cells:
        base = baselines[key(cell)]
        if len(cell["outputs"]) != len(base["outputs"]):
            raise ValueError("different output request counts")
        for i, (output, expected) in enumerate(zip(cell["outputs"], base["outputs"])):
            a, b = output["token_ids"], expected["token_ids"]
            if a != b:
                first = next((t for t, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                differences.append(dict(method=cell["job"]["method"], concurrency=cell["job"]["concurrency"],
                    seed=cell["job"]["seed"], repeat=cell["job"]["repeat"], request=i, position=first,
                    actual=a[first:first+1], expected=b[first:first+1]))
    return dict(passed=not differences, differences=differences, cells=len(cells))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = check(args.directory)
    (args.directory / "GREEDY_GATE.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
