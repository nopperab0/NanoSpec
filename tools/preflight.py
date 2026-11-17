"""Fail-fast validation for one allocated A100; run before loading large models."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import tempfile


def check():
    import torch
    import torch.distributed as dist
    import transformers
    import flash_attn
    import triton
    from nanovllm.layers.attention import Attention
    from nanovllm.utils.context import set_context, reset_context
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible CUDA GPU from the allocation")
    name = torch.cuda.get_device_name(0)
    if "A100" not in name:
        raise RuntimeError(f"expected an A100, received {name}")
    if torch.cuda.get_device_capability(0) != (8, 0) or not torch.cuda.is_bf16_supported():
        raise RuntimeError("expected Ampere SM80 with BF16 support")
    with tempfile.TemporaryDirectory() as directory:
        dist.init_process_group("nccl", init_method=(Path(directory) / "store").as_uri(), rank=0, world_size=1)
        try:
            value = torch.ones(1, device="cuda")
            dist.all_reduce(value)
            assert value.item() == 1
        finally:
            dist.destroy_process_group()
    device, dtype = torch.device("cuda"), torch.bfloat16
    layer = Attention(4, 128, 128 ** -.5, 2)
    layer.k_cache = torch.zeros(2, 256, 2, 128, device=device, dtype=dtype)
    layer.v_cache = torch.zeros_like(layer.k_cache)
    q = torch.randn(1, 4, 128, device=device, dtype=dtype)
    k = torch.randn(1, 2, 128, device=device, dtype=dtype)
    v = torch.randn_like(k)
    set_context(False, slot_mapping=torch.tensor([0], device=device, dtype=torch.int32),
        context_lens=torch.tensor([1], device=device, dtype=torch.int32),
        block_tables=torch.tensor([[0]], device=device, dtype=torch.int32))
    try:
        output = layer(q, k, v)
        torch.testing.assert_close(output, v.repeat_interleave(2, dim=1), atol=.02, rtol=.02)
    finally:
        reset_context()
    x = torch.randn(256, 256, device=device, dtype=dtype)
    assert torch.isfinite(x @ x).all().item()
    torch.cuda.synchronize()
    return dict(passed=True, python=platform.python_version(), torch=torch.__version__,
        transformers=transformers.__version__, flash_attn=flash_attn.__version__, triton=triton.__version__,
        cuda=torch.version.cuda, gpu=name, total_memory=torch.cuda.get_device_properties(0).total_memory,
        checks=["one A100", "SM80/BF16", "NCCL", "Triton KV store", "paged FlashAttention GQA", "BF16 GEMM"],
        nvidia_smi=subprocess.check_output(["nvidia-smi"], text=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = check()
    except Exception as error:
        args.out.write_text(json.dumps(dict(passed=False, error=str(error)), indent=2), encoding="utf-8")
        raise
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
