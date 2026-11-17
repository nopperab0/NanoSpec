"""Download pinned target and draft snapshots; does not load or execute model code."""
import argparse
import json
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--models", nargs="+", choices=["target", "naive", "dflash", "dspark", "jetspec"],
                        default=["target", "naive", "dflash", "dspark", "jetspec"])
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    registry = json.loads((Path(__file__).resolve().parents[1] / "benchmarks/models.json").read_text())
    for name in args.models:
        model = registry[name]
        directory = args.out / model["directory"]
        snapshot_download(model["repo"], revision=model["revision"], local_dir=str(directory),
                          allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken", "LICENSE*"])
        (directory / "SOURCE.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
