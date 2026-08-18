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
