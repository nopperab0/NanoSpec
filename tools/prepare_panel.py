"""Freeze an English-only SPEED-Bench throughput_1k panel with Qwen3 non-thinking token IDs."""
import argparse
import hashlib
import json
from pathlib import Path


def select_rows(rows, count, salt):
    candidates = []
    seen = set()
    for row in rows:
        prompt = row["turns"][0]
        if not prompt or any("\u3400" <= c <= "\u9fff" for c in prompt) or prompt in seen:
            continue
        seen.add(prompt)
        score = hashlib.sha256((salt + row["question_id"]).encode()).hexdigest()
        candidates.append((score, row))
    candidates.sort(key=lambda item: item[0])
    if len(candidates) < count:
        raise ValueError("not enough unique English-only prompts")
    return [row for _, row in candidates[:count]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--salt", default="nano-spec-infra-v1")
    parser.add_argument("--revision", help="dataset revision; default resolves and records the current immutable SHA")
    args = parser.parse_args()
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    repo = "nvidia/SPEED-Bench"
    revision = args.revision or HfApi().dataset_info(repo).sha
    dataset = load_dataset(repo, "throughput_1k", split="test", revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    rows = select_rows(dataset, args.count, args.salt)
    panel = []
    for row in rows:
        tokens = tokenizer.apply_chat_template([dict(role="user", content=row["turns"][0])],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        panel.append(dict(id=row["question_id"], category=row["sub_category"], prompt_token_ids=tokens))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError("refusing to replace a frozen panel")
    args.out.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    metadata = dict(dataset=repo, revision=revision, subset="throughput_1k", split="test", salt=args.salt,
        count=len(panel), selection="SHA256 order, unique first turns, CJK excluded; not category stratified",
        tokenizer=args.target, chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        enable_thinking=False, sha256=hashlib.sha256(args.out.read_bytes()).hexdigest())
    args.out.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
