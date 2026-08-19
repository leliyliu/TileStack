"""MetaX C550 RMSNorm Triton 调优 kernel + sglang 接入补丁.

背景 (bench/results_c550_sglang_align.json, 2026-08-19):
  sglang metax 镜像的 RMSNorm 走 flashinfer (0.0777ms@4096x4096) /
  sgl_kernel fused (0.1063ms); 本 triton kernel 达 0.0559 / 0.0989ms,
  经 sglang RMSNorm 类端到端验证提速 1.26-1.36x (非 residual) /
  1.09-1.25x (residual), 数值 allclose 通过。

调优要点 (C550 64 线程 wavefront):
  - 每行一个 program, BLOCK=next_pow2(hidden), 掩码最小化
  - num_warps = clamp(BLOCK//512, 1, 16): 4096 hidden → 8 warp = 512 线程
  - fp32 累加, bf16 存储; fused 版原地写回 (与 sgl_kernel 语义一致)

限制: 仅 hidden 为 2 的幂时优于 flashinfer (非 2 幂需掩码, 反而更慢),
      接入时必须保留 flashinfer 回退。

sglang 接入方式 (在沐曦适配源码上, 注意容器内 .py 与运行时 .pyc 不一致,
不要直接删 pyc 强制重编译):
  1. 将本文件两个 @triton.jit kernel 与 dispatch 函数并入
     sglang/srt/layers/layernorm.py
  2. 模块级 rmsnorm() wrapper 改为:
         if _metax_triton_rmsnorm_ok(input): return triton_rmsnorm(...)
         return flashinfer_rmsnorm(input, weight, eps, out, enable_pdl)
  3. forward_cuda residual 分支的 fused_add_rmsnorm 调用改为:
         if _metax_triton_rmsnorm_ok(x): triton_fused_add_rmsnorm(...)
         else: fused_add_rmsnorm(x, residual, weight, eps)
  或运行时 monkey-patch (见 verify_sglang_patch(), 已验证)。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(X, W, Y, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    y = (x * rstd * w).to(Y.dtype.element_ty)
    tl.store(Y + row * stride + cols, y, mask=mask)


@triton.jit
def _fused_add_rmsnorm_kernel(X, RES, W, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(RES + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + res
    tl.store(RES + row * stride + cols, x.to(RES.dtype.element_ty), mask=mask)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    y = (x * rstd * w).to(X.dtype.element_ty)
    tl.store(X + row * stride + cols, y, mask=mask)


def triton_ok(hidden, dtype):
    """仅 2 的幂 hidden + 半精度时启用 (其余回退 flashinfer/sgl_kernel)。"""
    return (hidden & (hidden - 1)) == 0 and dtype in (
        torch.bfloat16, torch.float16,
    ) and hidden >= 64


def _launch_cfg(hidden):
    block = max(triton.next_power_of_2(hidden), 64)
    num_warps = max(1, min(16, block // 512))
    return block, num_warps


def triton_rmsnorm(input, weight, eps=1e-6):
    n = input.shape[-1]
    x2 = input if input.dim() == 2 else input.contiguous().reshape(-1, n)
    y = torch.empty_like(x2)
    block, warps = _launch_cfg(n)
    _rmsnorm_kernel[(x2.numel() // n,)](
        x2, weight, y, x2.stride(0), n, eps, BLOCK=block, num_warps=warps)
    return y if input.dim() == 2 else y.view_as(input)


def triton_fused_add_rmsnorm(x, residual, weight, eps=1e-6):
    """原地语义 (与 sgl_kernel.fused_add_rmsnorm 一致): 返回 (norm_x, new_residual)。"""
    n = x.shape[-1]
    block, warps = _launch_cfg(n)
    _fused_add_rmsnorm_kernel[(x.numel() // n,)](
        x, residual, weight, x.stride(0), n, eps, BLOCK=block, num_warps=warps)
    return x, residual


def verify_sglang_patch():
    """在 sglang metax 容器内端到端验证 (monkey-patch 方式, 不改文件)。

    运行: docker exec <ctr> python /path/metax_rmsnorm_triton.py
    """
    import json
    import time

    import sglang.srt.layers.layernorm as L
    from sglang.srt.layers.layernorm import RMSNorm

    def bench(fn, iters=50, warmup=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters

    def patched_rmsnorm(input, weight, eps=1e-6, out=None, enable_pdl=False):
        if triton_ok(input.shape[-1], input.dtype):
            return triton_rmsnorm(input, weight, eps)
        return _orig_rmsnorm(input, weight, eps, out, enable_pdl)

    def patched_fused(x, residual, weight, eps):
        if triton_ok(x.shape[-1], x.dtype):
            return triton_fused_add_rmsnorm(x, residual, weight, eps)
        return _orig_fused(x, residual, weight, eps)

    results = {}
    for rows, hidden in [(4096, 4096), (4096, 8192), (16384, 8192), (4096, 5120)]:
        x = torch.randn(rows, hidden, device="cuda:0", dtype=torch.bfloat16)
        res = torch.randn_like(x)
        ln = RMSNorm(hidden, eps=1e-6).to("cuda:0").to(torch.bfloat16)
        base_nr, base_r = bench(lambda: ln(x)), bench(lambda: ln(x, res))
        ref = ln(x.clone()).float()

        global _orig_rmsnorm, _orig_fused
        _orig_rmsnorm, _orig_fused = L.rmsnorm, L.fused_add_rmsnorm
        L.rmsnorm, L.fused_add_rmsnorm = patched_rmsnorm, patched_fused
        try:
            ok = torch.allclose(ln(x.clone()).float(), ref, atol=2e-2, rtol=2e-2)
            p_nr, p_r = bench(lambda: ln(x)), bench(lambda: ln(x, res))
        finally:
            L.rmsnorm, L.fused_add_rmsnorm = _orig_rmsnorm, _orig_fused

        results[f"{rows}x{hidden}"] = {
            "no_residual": f"{base_nr*1e3:.4f} -> {p_nr*1e3:.4f} ms ({base_nr/p_nr:.2f}x)",
            "residual": f"{base_r*1e3:.4f} -> {p_r*1e3:.4f} ms ({base_r/p_r:.2f}x)",
            "numerics_ok": ok,
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    verify_sglang_patch()
