"""RMSNorm 性能模型 (memory-bound, 单遍 smem 缓冲 kernel).

良实现 (FlashInfer/compiled) 的 RMSNorm 将整行缓冲在 WSM, 单遍完成
sum-of-squares 与 normalize, DDR IO = 读 N 份 (+residual 再读 1 份) + 写 1 份。

C550 实测校准 (bench/results_c550_ops.json, 4 个测点一致):
  良实现 elementwise/reduce kernel 有效带宽 ~0.9-1.0 TB/s
  = ddr_bandwidth(1430) x 0.65 → arch.elementwise_ddr_eff = 0.65
(NVIDIA 架构无此属性时默认 1.0。)
"""
import math
from collections import namedtuple

RMSNormResult = namedtuple("RMSNormResult", "total_latency ddr_io ddr_util compute_time")


def model_rmsnorm(rows, hidden, arch, dtype_bytes=2, residual=False):
    """建模单 kernel RMSNorm (+ 可选 residual add) 的延迟。

    Args:
        rows: 行数 (token 数)
        hidden: 每行元素数 (须 <= WSM 容量以单遍缓冲)
        arch: TileSight arch
        dtype_bytes: 元素字节数 (默认 bf16)
        residual: 是否融合 residual add (多读一份输入)
    """
    reads = 2 if residual else 1
    ddr_io = rows * hidden * dtype_bytes * (reads + 1)  # 读 N 份 + 写 1 份

    eff = getattr(arch, "elementwise_ddr_eff", 1.0)
    bw = arch.ddr_bandwidth * eff
    ddr_time = ddr_io / bw

    # 向量计算: 平方+累加 (2 flops/ele) + 归一化乘乘 (2 flops/ele)
    vec_flops = rows * hidden * 4
    compute_time = vec_flops / arch.fp16_cuda_core_flops
    # SFU rsqrt 每行 1 次, 量级可忽略 (rows << vec throughput)

    latency = max(ddr_time, compute_time)
    return RMSNormResult(
        total_latency=latency,
        ddr_io=ddr_io,
        ddr_util=ddr_time / latency,
        compute_time=compute_time,
    )
