"""sglang metax 镜像 RMSNorm 补丁 — 部署后有效性评估脚本.

在打补丁的容器内运行 (patch 默认启用):
    /opt/conda/bin/python /tmp/eval_patch.py
A/B 对照 (禁用补丁):
    METAX_RMSNORM_PATCH=0 /opt/conda/bin/python /tmp/eval_patch.py

评估三层:
  1. 补丁生效确认 (模块属性已被替换)
  2. RMSNorm 类级延迟 (4 shape, 含数值校验)
  3. 复合 transformer 层模拟 (3 matmul + 2 fused norm), 报告层级收益
"""
import json
import sys
import time

import torch

torch.manual_seed(0)
dev = "cuda:0"
R = {"argv_env_note": sys.argv}


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


# ---- 1. 生效确认 ----
import sglang.srt.layers.layernorm as L
R["patch_active"] = getattr(L, "_metax_rmsnorm_patched", False)
R["rmsnorm_is_patched"] = L.rmsnorm.__module__ == "metax_rmsnorm_patch"

from sglang.srt.layers.layernorm import RMSNorm

# ---- 2. 类级基准 ----
R["class_level"] = {}
for rows, hidden in [(4096, 4096), (4096, 8192), (16384, 8192), (4096, 5120)]:
    x = torch.randn(rows, hidden, device=dev, dtype=torch.bfloat16)
    res = torch.randn_like(x)
    ln = RMSNorm(hidden, eps=1e-6).to(dev).to(torch.bfloat16)
    ref = torch.nn.functional.rms_norm(
        x.float(), (hidden,), ln.weight.data.float(), 1e-6)
    out = ln(x.clone())
    ok = torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)
    R["class_level"][f"{rows}x{hidden}"] = {
        "no_residual_ms": round(bench(lambda: ln(x)) * 1e3, 4),
        "residual_ms": round(bench(lambda: ln(x, res)) * 1e3, 4),
        "numerics_ok": bool(ok),
    }

# ---- 3. 复合层模拟 (LLaMA-ish: 2 fused norm + 3 matmul) ----
H = 4096
w1 = torch.randn(H, H, device=dev, dtype=torch.bfloat16) * 0.02
w2 = torch.randn(H, H, device=dev, dtype=torch.bfloat16) * 0.02
w3 = torch.randn(H, H, device=dev, dtype=torch.bfloat16) * 0.02
ln1 = RMSNorm(H, eps=1e-6).to(dev).to(torch.bfloat16)
ln2 = RMSNorm(H, eps=1e-6).to(dev).to(torch.bfloat16)


def layer(x, res):
    h, res = ln1(x, res)
    h = h @ w1
    h, res = ln2(h, res)
    h = h @ w2
    return h @ w3, res


for rows in (4096, 8, 16384):  # prefill / decode / 大 batch
    x = torch.randn(rows, H, device=dev, dtype=torch.bfloat16)
    res = torch.randn_like(x)
    t = bench(lambda: layer(x, res), iters=30)
    R[f"composite_layer_{rows}rows_ms"] = round(t * 1e3, 4)

print(json.dumps(R, indent=2))
