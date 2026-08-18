"""FlashAttention prefill 三段拆解性能模型.

将 FA 拆为依赖链三段 (tile 级流水, 段间经 WSM 传递, 无 DDR 中间写回):
  段1 QK^T  : batched GEMM (BH, S, D) x (BH, D, S)   → tensor core
  段2 softmax: 行内 max/exp/sum/normalize             → smem + SFU (scores 驻留 WSM)
  段3 PV     : batched GEMM (BH, S, S) x (BH, S, D)   → tensor core

段1/段3 复用 calculate_matmul_pipeline_wave (含 cache 命中/occupancy/wave 建模),
段2 按 smem 带宽 + SFU 吞吐建模。Q/K/V/O 的 DDR 流量相对 S^2 项可忽略
(S*D << S*S, D=128)。

实测对照 (muxi-01, bench/results_c550_ops.json): 有效算力 ~150 TFLOPS。
"""
from collections import namedtuple

from .matmul_pipeline_wave import calculate_matmul_pipeline_wave

FAResult = namedtuple(
    "FAResult", "total_latency qk_time softmax_time pv_time effective_tflops flops"
)


def _scaled_arch(arch, factor):
    """返回 fp16_tensor_flops 缩放 factor 倍的 arch 浅拷贝 (其余不变)。"""
    import copy

    a = copy.copy(arch)
    a.fp16_tensor_flops = arch.fp16_tensor_flops * factor
    return a


def model_flash_attention_prefill(B, H, S, D, arch, causal=False,
                                  dtype_bytes=2, tb=(128, 128, 32),
                                  wp=(64, 64, 16), stage_num=2):
    """建模 FA prefill 总延迟。

    Args:
        B/H/S/D: batch / heads / seq_len / head_dim
        causal: 因果 mask (计算量与 softmax 量减半)
        tb/wp: 段1/段3 GEMM 的 thread-block 与 warp tile

    校准: arch.fa_tc_eff (可选) 为 FA 相对纯 GEMM 的 tensor core 有效系数
    (online softmax/rescale/非方 tile 的固有开销)。C550 实测 6 点一致
    ~150-161 TFLOPS / 275 峰值 = 0.58。
    """
    BH = B * H
    work_ratio = 0.5 if causal else 1.0
    fa_tc_eff = getattr(arch, "fa_tc_eff", 1.0)
    gemm_arch = _scaled_arch(arch, fa_tc_eff)

    mem_levels = {
        "in1": [1, 1, 1, dtype_bytes],
        "in2": [1, 1, 1, dtype_bytes],
        "out1": [0, 0, 1, 4],
    }

    # ---- 段1: QK^T (M=S, N=S, K=D, batch=BH) ----
    qk = calculate_matmul_pipeline_wave(
        op_shape=(S, S, D), tb_shape=tb, wp_shape=wp, stage_num=stage_num,
        arch=gemm_arch, mem_levels=mem_levels, batch=BH, mma_type="wmma",
    )
    qk_time = qk.total_latency * work_ratio

    # ---- 段2: softmax (scores 驻留 WSM, 不落 DDR) ----
    # 每 q-row-block: 读 scores tile + 写 归一化 tile (各 1 遍), 计 2 遍 smem IO
    scores_bytes = BH * S * S * dtype_bytes * work_ratio
    smem_io = scores_bytes * 2
    smem_time = smem_io / arch.smem_bandwidth
    # SFU: exp + 除法 (或乘 rsqrt), 每元素约 1 次 SFU op + 行归约
    sfu_flops = getattr(arch, "sfu_flops", 0)
    if sfu_flops > 0:
        sfu_ops = BH * S * S * work_ratio
        sfu_time = sfu_ops / sfu_flops
    else:
        # 无 SFU 吞吐数据 (C550): exp 走向量单元近似 (4 flops/ele)
        sfu_time = (BH * S * S * 4 * work_ratio) / arch.fp16_cuda_core_flops
    softmax_time = max(smem_time, sfu_time)

    # ---- 段3: PV (M=S, N=D, K=S, batch=BH) ----
    # PV 的 A 即 scores, 已由段2 驻留 WSM (FA 在线 softmax 直供 MMA),
    # 不产生 DDR 读; 仅 V 从 DDR 加载 → in1 走 smem 层 [0,1,1,b]
    pv_mem_levels = {
        "in1": [0, 1, 1, dtype_bytes],
        "in2": [1, 1, 1, dtype_bytes],
        "out1": [0, 0, 1, 4],
    }
    pv = calculate_matmul_pipeline_wave(
        op_shape=(S, D, S), tb_shape=(tb[0], D, tb[2]), wp_shape=(wp[0], D, wp[2]),
        stage_num=stage_num, arch=gemm_arch, mem_levels=pv_mem_levels, batch=BH,
        mma_type="wmma",
    )
    pv_time = pv.total_latency * work_ratio

    total = qk_time + softmax_time + pv_time
    flops = 4 * BH * S * S * D * work_ratio
    return FAResult(
        total_latency=total,
        qk_time=qk_time,
        softmax_time=softmax_time,
        pv_time=pv_time,
        effective_tflops=flops / total / 1e12 if total > 0 else 0.0,
        flops=flops,
    )
