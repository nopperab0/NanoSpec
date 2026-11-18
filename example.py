"""Change --method to choose one of four speculative algorithms."""
import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True)
    p.add_argument("--method", choices=["none", "naive", "dflash", "dspark", "jetspec"], default="none")
    p.add_argument("--draft")
    p.add_argument("--k", type=int, default=3, help="naive draft length")
    p.add_argument("--tree-budget", type=int, default=32)
    p.add_argument("--tree-width", type=int, default=4)
    p.add_argument("--jetspec-backend", choices=["reference", "triton"], default="triton")
    p.add_argument("--jetspec-cuda-graph", action="store_true")
    p.add_argument("--target-kv-gib", type=float, default=4)
    p.add_argument("--draft-kv-gib", type=float, default=1)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--prompt", default="Explain how speculative decoding preserves the target distribution.")
    args = p.parse_args()
    from nanovllm import LLM, SamplingParams
    kwargs = dict(method=args.method, enforce_eager=True, enable_prefix_cache=False,
                  max_num_seqs=1, kv_cache_memory_bytes=int(args.target_kv_gib * (1 << 30)), seed=17)
    if args.method != "none":
        kwargs["draft_model"] = args.draft
    if args.method == "naive":
        kwargs.update(num_speculative_tokens=args.k, draft_kv_cache_memory_bytes=int(args.draft_kv_gib * (1 << 30)))
    if args.method == "jetspec":
        kwargs.update(jetspec_tree_budget=args.tree_budget, jetspec_tree_width=args.tree_width)
        kwargs.update(jetspec_backend=args.jetspec_backend, jetspec_cuda_graph=args.jetspec_cuda_graph)
    llm = LLM(args.target, **kwargs)
    try:
        result = llm.generate([args.prompt], SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens))
        print(result[0]["text"])
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
