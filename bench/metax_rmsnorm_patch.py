"""MetaX C550 RMSNorm Triton 调优 kernel — sglang 镜像集成补丁.

安装方式 (镜像内, 不修改 sglang 任何源文件, 规避 .py/.pyc 不一致问题):
  1. 本文件与 metax_rmsnorm_kernels.py 放入 site-packages/
  2. 创建 metax_rmsnorm_patch.pth, 内容一行: import metax_rmsnorm_patch

原理: .pth 在解释器启动时执行本模块 → 注册 import hook;
sglang.srt.layers.layernorm 加载完成后, 替换其模块级 rmsnorm /
fused_add_rmsnorm (forward_cuda 按模块全局名调用, 运行时 pyc 语义),
非 2 的幂 hidden / 非半精度自动回退原实现。

A/B 开关: 环境变量 METAX_RMSNORM_PATCH=0 禁用 (默认启用)。

性能 (muxi-01, 4096x4096 bf16):
  无 residual: flashinfer 0.0777ms → 0.0559ms (1.36x, 理论带宽 93%)
  带 residual: sgl_kernel 0.1063ms → 0.0989ms (1.09x)
来源: TileStack bench/results_c550_sglang_align.json (2026-08-19)

实现注记: kernel 必须放在真实源文件 metax_rmsnorm_kernels.py 中
(triton.jit 需要 inspect.getsourcelines, exec 字符串定义会
OSError: could not get source code)。
"""
import importlib.abc
import importlib.util
import os
import sys

_TARGET = "sglang.srt.layers.layernorm"
_FLAG = "METAX_RMSNORM_PATCH"
_PATCHED_FLAG = "_metax_rmsnorm_patched"


def _apply_patch(module):
    if os.environ.get(_FLAG, "1") == "0":
        print(f"[metax_rmsnorm_patch] disabled by {_FLAG}=0",
              file=sys.stderr)
        return
    if getattr(module, _PATCHED_FLAG, False):
        return
    # 懒导入: triton kernel 必须来自真实源文件 (见文件头注记)
    import metax_rmsnorm_kernels as K

    orig_rmsnorm = module.rmsnorm
    orig_fused = module.fused_add_rmsnorm

    def patched_rmsnorm(input, weight, eps=1e-6, out=None, enable_pdl=False):
        if K._ok(input.shape[-1], input.dtype):
            return K.rmsnorm(input, weight, eps)
        return orig_rmsnorm(input, weight, eps, out, enable_pdl)

    def patched_fused(x, residual, weight, eps):
        if K._ok(x.shape[-1], x.dtype):
            return K.fused_add_rmsnorm(x, residual, weight, eps)
        return orig_fused(x, residual, weight, eps)

    module.rmsnorm = patched_rmsnorm
    module.fused_add_rmsnorm = patched_fused
    setattr(module, _PATCHED_FLAG, True)
    print("[metax_rmsnorm_patch] applied: triton RMSNorm active "
          "(pow2 hidden, bf16/fp16; fallback flashinfer/sgl_kernel otherwise)",
          file=sys.stderr)


class _Hook(importlib.abc.MetaPathFinder):
    """sglang.srt.layers.layernorm 加载后立即打补丁 (一次性)。"""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        try:
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
        except (ValueError, ModuleNotFoundError, ImportError):
            return None
        finally:
            if self in sys.meta_path:
                sys.meta_path.remove(self)
        if spec is None or spec.loader is None:
            return None

        orig_exec = spec.loader.exec_module

        def wrapped_exec(module):
            orig_exec(module)
            try:
                _apply_patch(module)
            except Exception as e:  # noqa: BLE001
                print(f"[metax_rmsnorm_patch] patch failed, "
                      f"keep original kernels: {e}", file=sys.stderr)

        spec.loader.exec_module = wrapped_exec
        return spec


def _install():
    try:
        module = sys.modules.get(_TARGET)
        if module is not None:
            _apply_patch(module)
            return
        # 只探测顶层 sglang 是否安装 (对子模块 find_spec 会触发父包导入,
        # 启动时太重); hook 在 layernorm 真正被导入时才触发
        if importlib.util.find_spec("sglang") is not None:
            if not any(isinstance(f, _Hook) for f in sys.meta_path):
                sys.meta_path.insert(0, _Hook())
    except Exception as e:  # noqa: BLE001
        # 补丁绝不能破坏宿主进程启动
        print(f"[metax_rmsnorm_patch] install skipped: {e}",
              file=sys.stderr)


_install()
