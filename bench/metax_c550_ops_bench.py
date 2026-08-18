"""MetaX C550 算子级基准: FlashAttention / RMSNorm / 融合链.

运行环境: muxi-01 容器 dsv4-d-50 (/opt/conda/bin/python, 设备名 "cuda").
产出 JSON 打印到 stdout, 供 TileSight 建模对比 (误差目标 <=15%).
"""
import json
import time

import torch

torch.manual_seed(0)

RESULTS = {"env": {"torch": torch.__version__}}


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    dev = "cuda:0"
    from flash_attn import flash_attn_func

    # ---- FlashAttention prefill ----
    RESULTS["fa_prefill_ms"] = {}
    RESULTS["fa_prefill_causal_ms"] = {}
    for S in (2048, 4096, 8192):
        B, H, D = 8, 32, 128
        q = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
        t = bench(lambda: flash_attn_func(q, k, v, causal=False))
        RESULTS["fa_prefill_ms"][S] = round(t * 1e3, 4)
        t = bench(lambda: flash_attn_func(q, k, v, causal=True))
        RESULTS["fa_prefill_causal_ms"][S] = round(t * 1e3, 4)
        del q, k, v

    # ---- FlashAttention decode (q_len=1, KV cache) ----
    RESULTS["fa_decode_ms"] = {}
    for S in (2048, 4096, 8192):
        B, H, D = 8, 32, 128
        q = torch.randn(B, 1, H, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
        t = bench(lambda: flash_attn_func(q, k, v, causal=False), iters=50)
        RESULTS["fa_decode_ms"][S] = round(t * 1e3, 4)
        del q, k, v

    # ---- RMSNorm ----
    RESULTS["rmsnorm_ms"] = {}
    for hidden in (4096, 8192):
        rows = 4096
        x = torch.randn(rows, hidden, device=dev, dtype=torch.bfloat16)
        ln = torch.nn.RMSNorm(hidden).to(dev).to(torch.bfloat16)
        t = bench(lambda: ln(x), iters=50)
        RESULTS["rmsnorm_ms"][hidden] = round(t * 1e3, 4)

    # ---- residual + RMSNorm: 串行 vs 融合 ----
    hidden = 4096
    rows = 4096
    x = torch.randn(rows, hidden, device=dev, dtype=torch.bfloat16)
    res = torch.randn(rows, hidden, device=dev, dtype=torch.bfloat16)
    ln = torch.nn.RMSNorm(hidden).to(dev).to(torch.bfloat16)

    def serial(x=x, res=res):
        return ln(x + res)

    fused = torch.compile(serial)
    t_serial = bench(serial, iters=50)
    try:
        fused(x, res)  # trigger compile
        torch.cuda.synchronize()
        t_fused = bench(lambda: fused(x, res), iters=50)
    except Exception as e:  # noqa: BLE001
        t_fused = None
        RESULTS["fusion_error"] = str(e)
    RESULTS["residual_rmsnorm_serial_ms"] = round(t_serial * 1e3, 4)
    if t_fused:
        RESULTS["residual_rmsnorm_fused_ms"] = round(t_fused * 1e3, 4)

    # ---- MatMul -> RMSNorm 链 (decode 典型路径: router projection + norm) ----
    M, K, N = 4096, 4096, 4096
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    ln2 = torch.nn.RMSNorm(N).to(dev).to(torch.bfloat16)

    def mm_norm(a=a, w=w):
        return ln2(torch.mm(a, w))

    t = bench(mm_norm, iters=20)
    RESULTS["matmul_rmsnorm_ms"] = round(t * 1e3, 4)
    t_mm = bench(lambda: torch.mm(a, w), iters=20)
    RESULTS["matmul_only_ms"] = round(t_mm * 1e3, 4)
    RESULTS["rmsnorm_after_mm_ms"] = round(RESULTS["matmul_rmsnorm_ms"] - RESULTS["matmul_only_ms"], 4)

    print(json.dumps(RESULTS, indent=2))


if __name__ == "__main__":
    main()
