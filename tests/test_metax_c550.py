"""MetaX C550 (MXC550) Arch profile 测试。

对应实测数据来源（2026-08-18, muxi-01, 8x C550, MACA 3.7.1.17 dsv4 容器,
torch 2.8.0+metax, 见 docs/metax_c550_integration.md）:
  - FP16 GEMM 峰值 275.03 TFLOPS (8192^3)
  - FP32 GEMM 124.5 TFLOPS (mcBLAS 矩阵路径)
  - HBM copy 1429 GB/s / add 1480 GB/s (STREAM 口径)
架构参数来源: macainfo (104 AP, 64KB smem/AP, 512KB regfile/AP, 8MB L2,
1600MHz max clock, 64-thread wavefront)。
"""
import math

import pytest

from tilesight.arch import MXC550


@pytest.fixture
def arch():
    return MXC550()


class TestFieldPresence:
    """建模层 (fused_op_pipeline_wave) 消费的全部 arch 字段必须存在且有效。"""

    def test_required_fields_positive(self, arch):
        positive_fields = [
            "sm_count", "ddr_bandwidth", "l2_bandwidth", "smem_bandwidth",
            "fp16_tensor_flops", "fp32_cuda_core_flops", "ddr_wave_bytes",
            "configurable_smem_capacity", "register_capacity_per_sm",
            "l2_capacity",
        ]
        for f in positive_fields:
            assert getattr(arch, f) > 0, f"{f} should be > 0"

    def test_get_tensor_core_minimum_ptx_exists(self, arch):
        # matmul_pipeline_wave.py:146 的硬 assert
        assert hasattr(arch, "get_tensor_core_minimum_ptx")


class TestMicrobenchDefaults:
    """默认口径 = 实测（microbench）。"""

    def test_fp16_tensor_flops_measured(self, arch):
        assert arch.fp16_tensor_flops == pytest.approx(275e12, rel=1e-6)

    def test_fp32_tensor_flops_measured(self, arch):
        # mcBLAS FP32 GEMM 走矩阵单元, 实测 124.5 TFLOPS
        assert arch.fp32_tensor_flops == pytest.approx(124.5e12, rel=1e-6)

    def test_ddr_bandwidth_measured(self, arch):
        # STREAM copy 实测 1429 GB/s (保守取值, add 为 1480)
        assert arch.ddr_bandwidth == pytest.approx(1430e9, rel=1e-6)

    def test_int8_tensor_flops_measured(self, arch):
        # C500 OAM 参考 560 TOPS, 与 FP16 280 同代同源
        assert arch.int8_tensor_flops == pytest.approx(560e12, rel=1e-6)


class TestSpecMode:
    """set_to_spec() 切换理论口径。"""

    def test_spec_values(self, arch):
        arch.set_to_spec()
        # C500 OAM 参考档: FP16 280 TFLOPS; 理论带宽 1843 GB/s
        assert arch.fp16_tensor_flops == pytest.approx(280e12, rel=1e-6)
        assert arch.ddr_bandwidth == pytest.approx(1843e9, rel=1e-2)

    def test_spec_returns_self(self, arch):
        assert arch.set_to_spec() is arch


class TestArchitectureParams:
    """架构层参数与 macainfo / 官方文档一致。"""

    def test_compute_hierarchy(self, arch):
        assert arch.sm_count == 104            # macainfo: 104 AP
        assert arch.max_freq == pytest.approx(1.6e9)
        assert arch.warp_schedulers_per_sm == 1  # 每 AP 单发射单元, 5 发射口

    def test_memory_hierarchy(self, arch):
        assert arch.configurable_smem_capacity == 64 * 1024   # WSM 64KB/AP
        assert arch.register_capacity_per_sm == 512 * 1024    # 向量寄存器堆
        assert arch.l2_capacity == 8 * 1024 * 1024            # macainfo L2 8MB
        # WSM 128B/cycle/AP (读写各 128B, 全双工)
        assert arch.smem_bandwidth == pytest.approx(104 * 1.6e9 * 128)

    def test_wavefront_64(self, arch):
        # C550 调度粒度为 64 线程 wavefront (非 NVIDIA 的 32)
        assert arch.wavefront_size == 64
        # 每 AP 32 wave / 2048 线程 → 64 线程 block 时最多 32 个驻留
        assert arch.max_blocks_per_sm == 32


class TestTensorCoreShapes:
    """WMMA 最小形状映射 (官方 WMMA 支持: half 16x16x16, tf32 16x16x8)。"""

    def test_fp16_shape(self, arch):
        assert arch.get_tensor_core_minimum_ptx(bytes=2) == (16, 16, 16)

    def test_fp32_tf32_shape(self, arch):
        assert arch.get_tensor_core_minimum_ptx(bytes=4) == (16, 16, 8)

    def test_int8_shape(self, arch):
        assert arch.get_tensor_core_minimum_ptx(bytes=1) == (16, 16, 32)

    def test_fp8_unsupported(self, arch):
        # FP8 是 C600 特性, C550 不支持
        with pytest.raises(ValueError):
            arch.get_tensor_core_minimum_ptx(bytes=0.5)


class TestWavefrontSemantics:
    """建模层必须按 arch.wavefront_size 计算, 而非硬编码 32。"""

    def test_arch_base_default_32(self):
        # NVIDIA 卡无自定义属性 → 基类默认 32, 行为不变
        from tilesight.arch.arch_base import Arch
        assert Arch().wavefront_size == 32

    def test_occupancy_uses_wavefront_size(self, arch):
        # C550: 64 线程/warp; 512KB regfile, reg_footprint=64 (per-warp 4B reg 数),
        # warps_per_block=4 (256 线程):
        #   per-block = 64 * 4B * 64线程 * 4warp = 64KB → 512KB/64KB = 8 blocks
        from tilesight.fused_op_pipeline_wave.occupancy import compute_occupancy
        tiles = compute_occupancy(smem_footprint=0, reg_footprint=64,
                                  warps_per_block=4, arch=arch)
        assert tiles == 8

    def test_thread_overhead_uses_wavefront_size(self, arch):
        # C550 wavefront=64: 64 线程无开销, 32 线程有 2x 开销
        from tilesight.fused_op_pipeline_wave.elementwise_pipeline_wave import (
            _compute_thread_overhead,
        )
        w = arch.wavefront_size
        assert _compute_thread_overhead(w, arch) == 1.0
        assert _compute_thread_overhead(w // 2, arch) == 2.0


MEASURED_RMSNORM_MS = {
    # bench/results_c550_ops.json: torch.compile 良实现 (单遍 smem 缓冲)
    (4096, 4096): 0.0731,
    (4096, 8192): 0.1374,
    (16384, 4096): 0.2681,
    (16384, 8192): 0.5181,
}


class TestRMSNormModel:
    """RMSNorm 建模 vs 实测 (compiled 良实现), 误差 <=15%。"""

    def test_rmsnorm_vs_measured(self, arch):
        from tilesight.fused_op_pipeline_wave.rmsnorm_pipeline_wave import (
            model_rmsnorm,
        )
        for (rows, hidden), measured_ms in MEASURED_RMSNORM_MS.items():
            r = model_rmsnorm(rows=rows, hidden=hidden, arch=arch)
            err = abs(r.total_latency - measured_ms * 1e-3) / (measured_ms * 1e-3)
            assert err < 0.15, (
                f"rows={rows} hidden={hidden}: modeled {r.total_latency*1e6:.1f}us "
                f"vs measured {measured_ms*1e3:.1f}us, err {err:.1%}"
            )

    def test_residual_rmsnorm_vs_measured(self, arch):
        # 融合 residual+RMSNorm 实测 0.1067 ms (IO: 读 2 份 + 写 1 份)
        from tilesight.fused_op_pipeline_wave.rmsnorm_pipeline_wave import (
            model_rmsnorm,
        )
        r = model_rmsnorm(rows=4096, hidden=4096, arch=arch, residual=True)
        err = abs(r.total_latency - 0.1067e-3) / 0.1067e-3
        assert err < 0.15, f"err {err:.1%}"

    def test_native_rmsnorm_inefficient_flagged(self, arch):
        # 原生 torch.nn.RMSNorm 0.731ms 远超带宽极限, 不应被模型匹配;
        # 模型输出应接近带宽上限, 从而暴露实现的低效
        from tilesight.fused_op_pipeline_wave.rmsnorm_pipeline_wave import (
            model_rmsnorm,
        )
        r = model_rmsnorm(rows=4096, hidden=4096, arch=arch)
        assert r.total_latency < 0.731e-3 / 3  # 模型上限应远低于差实现


MEASURED_FA_PREFILL_MS = {
    # bench/results_c550_ops.json: B=8, H=32, D=128, bf16
    False: {2048: 3.4087, 4096: 13.7377, 8192: 57.2449},
    True: {2048: 1.8919, 4096: 7.703, 8192: 29.3811},
}


class TestFlashAttentionModel:
    """FA prefill 三段拆解建模 vs 实测, 误差 <=15%。"""

    @pytest.mark.parametrize("causal", [False, True])
    @pytest.mark.parametrize("S", [2048, 4096, 8192])
    def test_prefill_vs_measured(self, arch, S, causal):
        from tilesight.fused_op_pipeline_wave.flash_attention_pipeline_wave import (
            model_flash_attention_prefill,
        )
        r = model_flash_attention_prefill(B=8, H=32, S=S, D=128, arch=arch,
                                          causal=causal)
        measured = MEASURED_FA_PREFILL_MS[causal][S] * 1e-3
        err = abs(r.total_latency - measured) / measured
        assert err < 0.15, (
            f"S={S} causal={causal}: modeled {r.total_latency*1e3:.2f}ms "
            f"vs measured {measured*1e3:.2f}ms, err {err:.1%}; "
            f"segments qk={r.qk_time*1e3:.1f} sm={r.softmax_time*1e3:.1f} "
            f"pv={r.pv_time*1e3:.1f}ms"
        )


class TestFusionGain:
    """融合收益建模 vs 实测 (compiled 良实现公平基线)。"""

    def test_residual_rmsnorm_serial_vs_measured(self, arch):
        # 实测: add 0.0714 + rmsnorm 0.0742 串行 = 0.142 ms
        from tilesight.fused_op_pipeline_wave.rmsnorm_pipeline_wave import (
            model_residual_rmsnorm,
        )
        t = model_residual_rmsnorm(rows=4096, hidden=4096, arch=arch, fused=False)
        assert abs(t - 0.142e-3) / 0.142e-3 < 0.15, f"serial {t*1e6:.1f}us vs 142us"

    def test_residual_rmsnorm_fused_vs_measured(self, arch):
        from tilesight.fused_op_pipeline_wave.rmsnorm_pipeline_wave import (
            model_residual_rmsnorm,
        )
        t = model_residual_rmsnorm(rows=4096, hidden=4096, arch=arch, fused=True)
        assert abs(t - 0.1069e-3) / 0.1069e-3 < 0.15, f"fused {t*1e6:.1f}us vs 106.9us"

    def test_fusion_gain_ratio(self, arch):
        # 实测收益比 0.142/0.1069 = 1.33x; 建模比误差 <=20%
        from tilesight.fused_op_pipeline_wave.rmsnorm_pipeline_wave import (
            model_residual_rmsnorm,
        )
        t_ser = model_residual_rmsnorm(4096, 4096, arch, fused=False)
        t_fus = model_residual_rmsnorm(4096, 4096, arch, fused=True)
        modeled_gain = t_ser / t_fus
        assert abs(modeled_gain - 1.33) / 1.33 < 0.20

    def test_matmul_rmsnorm_chain(self, arch):
        # 4096^3 bf16 GEMM + RMSNorm: 实测良实现 ≈ 0.511 + 0.074 = 0.585ms
        from tilesight.fused_op_pipeline_wave import calculate_matmul_pipeline_wave
        from tilesight.fused_op_pipeline_wave.rmsnorm_pipeline_wave import model_rmsnorm
        mm = calculate_matmul_pipeline_wave(
            (4096, 4096, 4096), (128, 128, 32), (64, 64, 16), 3, arch,
            {"in1": [1, 1, 1, 2], "in2": [1, 1, 1, 2], "out1": [0, 0, 1, 4]},
            mma_type="wmma")
        ln = model_rmsnorm(rows=4096, hidden=4096, arch=arch)
        total = mm.total_latency + ln.total_latency
        assert abs(total - 0.585e-3) / 0.585e-3 < 0.15, (
            f"chain {total*1e3:.3f}ms vs 0.585ms")


class TestMatmulModelSmoke:
    """端到端冒烟: 8192^3 FP16 GEMM 建模 vs 实测。

    实测 (muxi-01, 同批): 峰值 4.00 ms (275 TFLOPS),
    持续 4.77 ms (4000 次连续, 230 TFLOPS)。允许模型 3-15 ms 区间。
    """

    def test_8192_fp16_gemm(self, arch):
        from tilesight.fused_op_pipeline_wave import (
            calculate_matmul_pipeline_wave,
        )
        mem_levels = {
            "in1": [1, 1, 1, 2],   # ddr 加载, fp16
            "in2": [1, 1, 1, 2],
            "out1": [0, 0, 1, 4],  # 寄存器累加, fp32 累加器
        }
        result = calculate_matmul_pipeline_wave(
            op_shape=(8192, 8192, 8192),
            tb_shape=(128, 128, 32),
            wp_shape=(64, 64, 16),
            stage_num=3,
            arch=arch,
            mem_levels=mem_levels,
            mma_type="wmma",
        )
        assert result.total_latency > 0
        # 8192^3 x 2 = 1.0995e12 FLOPs; 实测峰值 4.0ms, 持续 4.77ms
        assert 3e-3 < result.total_latency < 15e-3, (
            f"modeled {result.total_latency*1e3:.2f} ms "
            f"vs measured 4.0-4.8 ms"
        )
