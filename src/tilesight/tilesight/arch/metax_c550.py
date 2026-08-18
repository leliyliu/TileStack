"""MetaX (沐曦) 曦云 C550 (MXC550) Arch profile.

数据来源标注约定 (与 npus/07-C550硬件规格.md 一致):
  🔬 实测   — 2026-08-18 于 muxi-01 (8x C550, MACA 3.7.1.17 dsv4 容器,
              torch 2.8.0+metax) 微基准实测, 复现脚本与原始数据见
              docs/metax_c550_integration.md
  ✅ 官方   — 沐曦官网 / MXMACA 官方文档 (以 C500 同架构文档为参照)
  🔬 推算   — macainfo 实测参数 x 官方架构常数推导

默认口径为 **实测 (microbench)**; `set_to_spec()` 切换理论口径。

架构要点 (详见 npus/07-C550硬件规格.md):
  - 104 AP (对应 NVIDIA SM), 每 AP 4 PEU x 16 lane = 64 lane
  - 64 线程 wavefront (非 NVIDIA 32 线程 warp), 每 AP 32 wave / 2048 线程
  - 每 AP 单发射单元、5 发射口 (1 标量 + 1 MMA/向量ALU + 1 向量访存
    + 1 共享内存 + 1 杂项)
  - WSM (共享内存) 64 KB/AP, 32 bank x 4B, 独立 SRAM (非 L1 划扣)
  - 向量寄存器堆 512 KB/AP; VL1 32KB 默认关闭 (访存直达 L2)
  - L2 8 MB 全片共享, 128B cacheline, 实测 4096 B/cycle
"""
from .arch_base import Arch


class MXC550(Arch):
    def __init__(self):
        super().__init__()

        # ---- 基本标识与计算层次 ----
        self.core = "MXC550"
        self.sm_count = 104                 # 🔬 macainfo: Accelerator Processors
        self.base_freq = 0.45 * 1e9         # 🔬 mx-smi 空载 XCORE_CLK (不随负载更新)
        self.max_freq = 1.6 * 1e9           # 🔬 macainfo: Max Clock Freq 1600 MHz
        # 每 AP 1 个发射单元 (5 发射口), 不同于 NVIDIA 每 SM 4 warp scheduler
        self.warp_schedulers_per_sm = 1
        # 64 线程 wavefront (macainfo Wavefront Size); 每 AP 32 wave
        self.wavefront_size = 64
        self.max_blocks_per_sm = 32         # 🔬 推算: 32 wave/AP, 64 线程 block = 1 wave

        # ---- Tensor / 向量计算能力 ----
        # 🔬 实测 (microbench 口径, 默认):
        #   FP16 GEMM 8192^3 峰值 275.03 TFLOPS (BF16 269.34 同量级)
        #   印证 C500 OAM 档 "FP16 280 TFLOPS" 第三方数据
        self.fp16_tensor_flops = 275.0 * 1e12
        # 🔬 实测: mcBLAS FP32 GEMM 124.5 TFLOPS, 明显走矩阵单元
        # (TF32 类路径; 纯向量 FP32 理论值仅 21.3 TFLOPS, 见 fp32_cuda_core_flops)
        self.fp32_tensor_flops = 124.5 * 1e12
        # ⚠️ 第三方 (C500 OAM 参考 560 TOPS, 同架构同源; 未实测)
        self.int8_tensor_flops = 560.0 * 1e12
        # C550 无 FP8 (C600 特性), 不设置 fp8_tensor_flops

        # ---- 向量 core (CUDA core 对应物: 64 lane/AP) ----
        # 🔬 推算: 104 AP x 1.6GHz x 64 lane x 2 (FMA) = 21.35 TFLOPS
        # (C500 OAM 官方档 18 TFLOPS @ 更低频, 量级吻合)
        self.fp32_cores_per_sm = 64
        self.fp32_cuda_core_flops = (
            self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        )
        # macainfo "Fast Float16 Operation: TRUE" → FP16 向量双倍速率
        self.fp16_cuda_core_flops = self.fp32_cuda_core_flops * 2
        # ⚠️ 估计 (沿用 MI300X 惯例 fp64 = fp32/2; C550 FP64 未公布未实测)
        self.fp64_cuda_core_flops = self.fp32_cuda_core_flops * 0.5
        # SFU 规模未公开; 建模层对 sfu_flops 有 hasattr 保护, 此处不设置

        # ---- DDR (HBM2e 64GB) ----
        # 🔬 实测 (microbench 口径, 默认): torch copy 1429 GB/s (读+写),
        # add 1480 GB/s; 取 copy 为保守值。理论 1843 GB/s 见 set_to_spec。
        self.ddr_bandwidth = 1430 * 1e9
        self.ddr_capacity = 64 * (1024**3)  # ✅ 官方 + 🔬 实测
        # DRAM wave 量化粒度 (仅 apply_dram_wave_quantization 开启时使用);
        # 无实测依据, 沿用 B6000 的保守默认值
        self.ddr_wave_bytes = 4096

        # ---- L2 (全片共享) ----
        self.l2_capacity = 8 * (1024**2)    # 🔬 macainfo: 8192 KB
        # 🔬 推算: 实测 4096 B/cycle x 1.6 GHz
        self.l2_bandwidth = 4096 * self.max_freq

        # ---- WSM (共享内存) 与寄存器 ----
        # WSM 带宽: 128 B/cycle/AP (读、写各 128B 全双工; 官方调优指南)
        self.l1_smem_throughput_per_cycle = 128
        self.configurable_smem_capacity = 64 * 1024   # ✅ 官方 64KB/AP
        self.register_capacity_per_sm = 512 * 1024    # ✅ 官方 向量寄存器堆
        # sm_sub_partitions 语义为 "每 SM 的 32-lane 分区数"; C550 是
        # 4 PEU x 16 lane, register_bandwidth 直接按 64 lane x 4B 计算
        self.sm_sub_partitions = 4                  # PEU 数 (用于占用率语义)
        self.smem_bandwidth = (
            self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        )
        self.register_bandwidth = (
            self.sm_count * self.max_freq * 64 * 4
        )  # 64 lane x 4B/cycle/AP

        # ---- 利用率上限 ----
        self.ddr_max_util = 0.9
        self.l2_max_util = 0.9
        self.l1_max_util = 0.9
        self.compute_max_util = 0.9

        # 🔬 校准 (4 测点一致, bench/results_c550_ops.json): 良实现
        # elementwise/reduce kernel (compiled RMSNorm 等) 有效带宽
        # ~0.9-1.0 TB/s = 1430 x 0.65。GEMM/纯 copy 不受此因子影响。
        self.elementwise_ddr_eff = 0.65

        # 注: L1.5 / TMEM 不存在 (l1_5_group_size=0, tmem_bandwidth=0,
        # 基类默认即为关闭); VL1 默认关闭且容量 32KB 过小, 不建层次。

    def get_tensor_core_minimum_ptx(self, bytes=2):
        """WMMA 最小 MMA 形状 (官方 WMMA 支持: half 16x16x16, tf32 16x16x8)。

        对应硬件指令 MMA_16x16x16F16 (npus/07 §2.3.9)。
        """
        if bytes == 2:      # fp16 / bf16
            return (16, 16, 16)
        elif bytes == 4:    # fp32 (tf32 路径)
            return (16, 16, 8)
        elif bytes == 1:    # int8
            return (16, 16, 32)
        elif bytes == 0.5:  # fp8: C550 不支持 (C600 特性)
            raise ValueError("MXC550 has no FP8 tensor core support")
        else:
            raise ValueError("bytes must be 2, 4, or 1")

    def set_to_spec(self):
        """切换到理论 (spec) 口径。

        FP16 280 TFLOPS 为 C500 OAM 官方参考档 (⚠️ C550 分精度算力官方未公布,
        第三方 240 TFLOPS 与实测 275 矛盾, 实测更接近 280 档);
        DDR 1843 GB/s 为位宽 4096bit x 1800MHz 推算理论带宽。
        """
        self.ddr_max_util = 1.0
        self.l2_max_util = 1.0
        self.l1_max_util = 1.0
        self.compute_max_util = 1.0
        self.fp16_tensor_flops = 280.0 * 1e12
        self.ddr_bandwidth = 1843.2 * 1e9
        return self


if __name__ == "__main__":
    arch = MXC550()
    print(f"core={arch.core}, sm_count={arch.sm_count}")
    print(f"fp16_tensor={arch.fp16_tensor_flops/1e12:.1f} TFLOPS")
    print(f"ddr={arch.ddr_bandwidth/1e9:.0f} GB/s")
    print(f"l2={arch.l2_bandwidth/1e9:.0f} GB/s")
    print(f"smem={arch.smem_bandwidth/1e12:.2f} TB/s")
