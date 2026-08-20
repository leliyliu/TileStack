"""MetaX C550 RMSNorm Triton kernels (独立文件: triton.jit 需要真实源码).

调优要点 (C550 64 线程 wavefront): 每行一个 program, BLOCK=next_pow2(hidden),
num_warps=clamp(BLOCK//512,1,16), fp32 累加, bf16 存储。
性能与来源见 metax_rmsnorm_patch.py 文件头。
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


def _ok(hidden, dtype):
    return (hidden >= 64 and (hidden & (hidden - 1)) == 0
            and dtype in (torch.bfloat16, torch.float16))


def _cfg(hidden):
    block = max(triton.next_power_of_2(hidden), 64)
    return block, max(1, min(16, block // 512))


def rmsnorm(input, weight, eps=1e-6):
    n = input.shape[-1]
    x2 = input if input.dim() == 2 else input.contiguous().reshape(-1, n)
    y = torch.empty_like(x2)
    block, warps = _cfg(n)
    _rmsnorm_kernel[(x2.numel() // n,)](
        x2, weight, y, x2.stride(0), n, eps, BLOCK=block, num_warps=warps)
    return y if input.dim() == 2 else y.view_as(input)


def fused_add_rmsnorm(x, residual, weight, eps=1e-6):
    """原地语义 (与 sgl_kernel.fused_add_rmsnorm 一致)。"""
    n = x.shape[-1]
    block, warps = _cfg(n)
    _fused_add_rmsnorm_kernel[(x.numel() // n,)](
        x, residual, weight, x.stride(0), n, eps, BLOCK=block, num_warps=warps)
    return x, residual
