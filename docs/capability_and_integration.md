# TileSight 能力评估与应用指南

> 本文档基于 `src/tilesight/` 源码与 `docs/TileSight.pdf` 论文，回答三个问题：(1) TileSight 的用途与相对 profiling/aiconfigurator 的优势；(2) 对实际部署中 PD 分离、异构硬件的支持情况；(3) 接入新 NPU 芯片的步骤。所有结论附源码位置或论文页码作为证据。

## 问题 1：TileSight 主要用途，相对 profiling 和 aiconfigurator 的优势

### 1.1 TileSight 主要能用来干嘛

TileSight 是一个 **tile-centric 的第一性原理分析性能建模工具**，把 tile 作为一等建模单元，用 prologue–steady–epilogue pipeline envelope 递归建模 GPU kernel 执行（论文 §3，`docs/TileSight.pdf` p.3-8）。

**核心能力**（源码证据）：

| 能力 | 实现位置 | 说明 |
|---|---|---|
| 单 kernel 延迟预测 | `fused_op_pipeline_wave/matmul_pipeline_wave.py:29` | 输入 tile shape/pipeline depth/arch，输出延迟与利用率，**无需运行 kernel** |
| 多级 cache 命中率预测 | `tile_cache/cache_model.py:15` + `util/sdcm.py:20` | 两级 L1.5+L2 SDCM cascade，论文 Eq.7-10 |
| 融合 kernel pipeline overlap | `fused_op_pipeline_wave/overlap_analysis.py:232` | DAG 拓扑序搜索选最优调度 |
| 分布式集合通信建模 | `distributed/collectives/all_reduce.py:30` 等 | 5 种 all-reduce 算法 + α-β stage cost（Eq.11） |
| 跨架构可移植 | `arch/` 31 个 profile | NVIDIA/AMD/CGRA/TPU，无需 per-arch 训练 |
| 白盒瓶颈诊断 | `overlap_analysis.py:HardwareUsage` 9 维分解 | 定位到具体 pipeline（TC/CUDA/SFU/DDR…） |

**主要用途**：论文 §5.6（PDF p.11-12）展示两类应用——(a) **性能预测**：在不运行 kernel 的情况下预测延迟/cache/分布式开销；(b) **优化指导**：作为 cost model 指导 tile 配置选择、schedule 搜索、瓶颈诊断。

**注意：TileSight 本身不做部署决策**——它不输出"用几个 prefill worker / 几个 decode worker"、不输出 K8s manifest、不做 SLA 搜索。它是性能建模引擎，不是部署配置工具。

### 1.2 相对 profiling（如 NCU）的优势

| 维度 | Profiling（NCU/Omniperf） | TileSight | 证据 |
|---|---|---|---|
| **时机** | post-hoc，必须先运行 kernel | 预测式，运行前即可预测 | 论文 §1 p.2「profilers are post-hoc」 |
| **配置探索** | 每个配置都要重跑 | 一次建模可枚举数千配置 | 论文 §5.6 Fig.12 剪枝 95% 候选 |
| **扰动** | 插桩/改时钟会扰动执行 | 纯分析，无扰动 | 论文 §1 p.2 |
| **解释性** | 报告 counter，不解释哪个 tile/调度致瓶颈 | 白盒分解到 tile 级 pipeline | `HardwareUsage` 9 维（`overlap_analysis.py:30`） |
| **新架构** | 需重新 profile | 微基准校准即可（~分钟级） | 论文 §3.8 p.8，`arch_base.py:set_to_microbench` |
| **局限** | 精确（ground truth） | 12.35% MAPE（论文 §5.2） | 论文 Table 4 p.9 |

**核心优势**：profiling 是"测量已发生的执行"，TileSight 是"预测未运行的配置"。在 schedule 搜索（数千候选）场景下，profiling 成本不可承受，TileSight 的分析模型是唯一可行解。

### 1.3 相对 aiconfigurator 的优势与定位差异

aiconfigurator（ai-dynamo，GitHub ~358 stars）是 **LLM 部署配置搜索工具**：给定模型+GPU 数+GPU 类型，搜索配置空间（聚合 vs 分离架构、prefill/decode workers 数、并行度），满足 SLA（TTFT/TPOT）并最大化吞吐，输出可部署配置（含 K8s manifest）。

**两者定位互补，非直接竞争**：

| 维度 | aiconfigurator | TileSight |
|---|---|---|
| **核心目标** | 部署配置搜索（输出 workers 数/并行度/manifest） | 性能建模（输出延迟/利用率/cache） |
| **方法论** | 基于收集的测量数据离线评估数千配置（`default`/`exp` 模式） | 第一性原理分析建模，无需测量数据 |
| **快速模式** | `generate` 用启发式公式 `TP*VRAM > 1.5*model_weight` | 无部署决策，但可秒级预测单配置延迟 |
| **硬件支持** | 仅 NVIDIA 6 种（h200/h100/gb200/b200/a100） | 31 profile 含 NVIDIA/AMD/CGRA/TPU，可扩展 |
| **PD 分离** | ✅ 核心能力（决定聚合 vs 分离 + workers 数） | ❌ 不支持（见问题 2） |
| **异构硬件** | ❌ 单 GPU 类型输入 | ❌ 同构集群假设（见问题 2） |
| **白盒解释** | ❌ 黑盒配置搜索 | ✅ 9 维 pipeline 分解 |
| **跨架构泛化** | ❌ 每架构需收集数据 | ✅ 微基准校准即可 |
| **框架绑定** | 主要 trtllm，输出 Dynamo/llm-d 配置 | 无框架绑定，纯建模库 |

**TileSight 的相对优势**：
1. **第一性原理，无需 per-arch 测量数据**——aiconfigurator 的 `default` 模式依赖"collected data for a target machine and framework"，换架构需重新收集；TileSight 仅需微基准校准（论文 §3.8）。
2. **跨架构可移植**——aiconfigurator 仅支持 6 种 NVIDIA GPU；TileSight 已覆盖 AMD/CGRA/TPU 且可扩展到新硬件。
3. **白盒可解释**——aiconfigurator 输出"用哪个配置"，不解释为什么慢；TileSight 分解到 tile/pipeline 级，能诊断瓶颈（论文 Table 5 的 1.07–8.97× 优化案例）。
4. **配置空间剪枝**——TileSight 可作为 aiconfigurator 的底层性能引擎，剪掉 95% 劣势候选（论文 Fig.12），加速其搜索。

**aiconfigurator 的相对优势**：开箱即用的部署决策（PD 分离、workers 数、K8s manifest）、SLA 优化、生产级集成。**TileSight 不替代 aiconfigurator，而是可以作为其更准确的性能预测后端**。

## 问题 2：PD 分离、异构硬件部署的支持情况

### 2.1 PD 分离（Prefill/Decode Disaggregation）—— ❌ 不支持部署架构层

**结论**：TileSight **不支持** PD 分离部署架构（即 prefill worker 与 decode worker 分离、KV cache 跨节点迁移）的建模。

**证据**：
- 源码中 `prefill`/`decode` 仅出现在 **attention 算子的两种计算 phase**（非部署架构）：
  - `tile_cache/reuse_distance_multilevel.py:51`「MLA decode grid (batch_tiles, head_num)」——这是 MLA decode 阶段的 tile 网格，不是部署分离
  - `fused_op_pipeline_wave/` 有 `fa_decode_wrapper`/`fa_prefill_wrapper`——是 FlashAttention 的 prefill/decode 两种 kernel 建模
- `distributed/` 层**无部署架构概念**：
  - 无 `role`/`worker_type`/`prefill_worker`/`decode_worker` 字段
  - 无 KV cache 跨节点 transfer 建模（grep `kv.*transfer`/`disaggregat` 无结果）
  - `DistributedTileMap`（`tile_distribution.py:42`）只描述单 op 的 tile 分区 + collective，无 worker 角色分离
  - `ClusterSpec`（`device.py:71`）只有 node/inter_node，无 prefill/decode pool 概念

**不适配原因**：TileSight 的建模粒度是"单 kernel / 单 op 的 tile 执行"，PD 分离是"部署调度层"概念（哪个请求路由到哪类 worker、KV 如何迁移），超出 TileSight 的建模范围。这属于 aiconfigurator/Vidur 等部署模拟器的领域。

**变通方案**：若要建模 PD 分离，需在 TileSight 之上自建调度层——分别对 prefill worker 和 decode worker 各跑一次 TileSight 建模，再手动叠加 KV transfer 开销（用 `distributed/collectives` 的 all-to-all 估算）。但这不是原生支持。

### 2.2 异构硬件 —— ❌ 不支持，同构集群假设

**结论**：TileSight **不支持**异构硬件集群（如 H100 + B200 混部、GPU + NPU 混部）的建模。

**证据**：
- `ClusterSpec`（`distributed/device.py:71`）只有一个 `node: NodeSpec`，`NodeSpec`（`device.py:46`）只有一个 `device_arch: Arch`——**整个集群单一架构假设**
- `cluster_presets.py` 的 `make_dgx_h100_cluster`/`make_dgx_b200_cluster` 都是同构
- `NetworkHierarchy`（`network.py:48`）从单一 ClusterSpec 构建，所有设备等价
- `DistributedTileMap.local_grids`（`tile_distribution.py:78`）按 `parallel.group_size(dim)` 均匀切分——假设每个设备算力相同
- 无 per-device 性能权重、无异构设备到 tile 的映射

**不适配原因**：TileSight 的分布式建模假设"所有设备同构"，tile 均匀分配、collective 对称。异构场景下（快慢设备混部）需要负载感知的 tile 分配与非对称 collective，当前实现均不支持。

**变通方案**：无原生方案。异构建模需要改造 `ClusterSpec` 支持多 arch、改造 `DistributedTileMap` 支持非均匀分配——这是较大改造。

## 问题 3：接入新 NPU 芯片的步骤

### 3.1 接入路径概览

接入新 NPU 分三步：(1) 编写 `Arch` profile 描述硬件；(2) 处理 op 建模层的 GPU 特定假设；(3) 微基准校准。**核心障碍在第 (2) 步**：op 建模层有硬编码的 NVIDIA GPU 假设。

### 3.2 步骤一：编写 NPU 的 Arch profile

继承 `arch/arch_base.py:Arch`（`arch_base.py:24`），填字段。**所有字段有默认值**，新 NPU 只需填非零项。

**必填字段**（参考 `cgra_ver20.py`/`tpuv4.py` 这两个非 GPU 先例）：

```python
# src/tilesight/tilesight/arch/my_npu.py
from .arch_base import Arch
import math

class MyNPU(Arch):
    def __init__(self):
        super().__init__()
        self.core = "MyNPU"                    # 标识符（注意：会进入 op 层分支判断）
        self.sm_count = 64                     # NPU 的计算核数（对应 SM 概念）
        self.base_freq = 1.5e9
        self.max_freq = 2.0e9
        # 矩阵单元（NPU 的 systolic array / MAC 阵列映射到 tensor_core 概念）
        self.tensor_cores_per_sm = 1           # 每核的矩阵单元数
        self.tensor_core_shape = (16, 16, 16)  # 矩阵单元形状 (M,N,K)
        self.tensor_core_flops = math.prod(self.tensor_core_shape) * 2
        self.fp32_cores_per_sm = 32            # 标量核
        # 内存层级
        self.ddr_bandwidth = 1024e9            # HBM/DDR 带宽
        self.ddr_capacity = 64 * 1024**3
        self.l2_bandwidth = 4096e9
        self.l2_capacity = 64 * 1024**2
        self.configurable_smem_capacity = 256 * 1024   # 片上 scratchpad
        self.register_capacity_per_sm = 256 * 1024
        # 占用度
        self.max_blocks_per_sm = 16
        self.sm_sub_partitions = 1
        self.warp_schedulers_per_sm = 1
        self.l1_smem_throughput_per_cycle = 128
        # 派生吞吐（按 sm_count*freq*cores 计算）
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4
        # 关键：控制是否走 GPU tensor-resource 方程
        self.use_tensor_core_resource_model = False  # NPU 走非 GPU 路径
```

**L1.5/TMEM**：NPU 若无对应结构，保持默认 0（禁用）；若有片上 cache 层，填 `l1_5_*`（参考 `b200.py:62`）。

### 3.3 步骤二：处理 op 建模层的 GPU 特定假设（核心障碍）

op 建模层（`fused_op_pipeline_wave/`）有四处硬编码假设，新 NPU 必须处理：

**障碍 1：`get_tensor_core_minimum_ptx` 断言**（`matmul_pipeline_wave.py:146`）
```python
assert hasattr(arch, 'get_tensor_core_minimum_ptx'), \
    "arch.get_tensor_core_minimum_ptx does not exist"
```
- 现状：仅 `b6000.py:96`/`h100_sxm.py:158`/`h200_sxm.py:155` 实现了该方法
- NPU 必须实现该方法（返回矩阵单元的最小执行粒度，如 `(8,8,8)`），否则 matmul 建模 assert 失败

**障碍 2：硬编码 `arch.core` 分支**（遍布 `fused_op_pipeline_wave/`）
- `matmul_pipeline_wave.py:193` `if arch.core in ("A100","H100","B200","A100_LUT")` —— L2 two-part cache 结构
- `matmul_pipeline_wave.py:213` `elif arch.core == "V100"` —— V100 专用 store 路径
- `matmul_pipeline_wave.py:222` `elif (arch.core in ("A100","H100")) and mma_type=="wmma"` —— wmma 专用
- `matmul_pipeline_wave.py:270` `if arch.core == "B200" and mma_type=="utcmma_cta2"` —— B200 专用
- `conv_pipeline_wave.py`/`elementwise_pipeline_wave.py` 各 3-6 处 `arch.core in ("A100","H100","B200")`
- NPU 会落入 `else` 分支（通用路径），需验证该路径对 NPU 是否合理

**障碍 3：`uses_gpu_resource_model` 判断**（`arch_base.py:8`）
- 若 `arch.use_tensor_core_resource_model=False` 且 `arch.core not in {A100,H100,B200}`，则 `uses_gpu_resource_model` 返回 False
- `matmul_pipeline_wave.py:289/465/497` 据此选择计算路径——NPU 走非 GPU 路径，需确认该路径完整

**障碍 4：`mma_type` 是 NVIDIA 术语**（`matmul_pipeline_wave.py:59`）
- 取值 `wmma`/`wgmma`/`utcmma_cta1`/`utcmma_cta2`，对应 NVIDIA 各代 MMA 指令
- NPU 需将自身的矩阵乘法单元映射到最接近的 mma_type（或新增一个 `npu_mma` 分支）

**处理策略**（按改造量从小到大）：
1. **最小接入**：实现 `get_tensor_core_minimum_ptx`，让 NPU 落入 `else` 通用分支，用最接近的 `mma_type`（如 `wmma`）。能跑，但精度可能下降（通用分支未针对 NPU 调优）。
2. **中等接入**：在关键 `arch.core` 分支处新增 NPU 专用分支（如把 NPU 加入 two-part cache 判断，或新增 NPU 专属 SMEM 路径）。
3. **完整接入**：为 NPU 新增 `mma_type="npu_mma"` 分支，针对 NPU 的矩阵单元行为精确建模。

### 3.4 步骤三：微基准校准

实现 `set_to_microbench`/`set_to_ncu`（参考 `b200.py:set_to_microbench`），用实测数据覆盖 spec 值：
- 带宽扫描（DDR/L2/SMEM 不同 working set 下的有效带宽，论文 Fig.2）
- 短矩阵乘探针（测 tensor core 有效吞吐）
- 这一步决定建模精度（论文 §3.8「values come from vendor specs and lightweight microbenchmarks」）

### 3.5 接入后如何评估

```python
from tilesight.arch.my_npu import MyNPU
from tilesight.fused_op_pipeline_wave.matmul_pipeline_wave import calculate_matmul_pipeline_wave

arch = MyNPU().set_to_microbench()
mem_levels = {'in1':[1,1,1,2], 'in2':[1,1,1,2], 'out1':[1,1,1,2]}
result = calculate_matmul_pipeline_wave(
    op_shape=(4096,4096,4096), tb_shape=(128,128,64),
    wp_shape=(64,64,64), stage_num=3, arch=arch,
    mem_levels=mem_levels, row_panel=8, mma_type="wmma")
print(result.total_latency, result.l2_hit_rate, result.tiles_per_sm)
```

**评估建议**：
1. 先用 NPU 上的 cutlass/CK 对应库跑一批 GEMM 形状作为 ground truth
2. 对比 TileSight 预测的 MAPE（论文目标 12.35%）
3. 若 MAPE 偏高，定位是哪个 pipeline 维度偏差（用 `HardwareUsage` 分解），回头补 microbench 校准或补 NPU 专用分支

### 3.6 接入工作量评估

| 步骤 | 工作量 | 风险 |
|---|---|---|
| Arch profile | 低（填字段，~1 天） | 低 |
| `get_tensor_core_minimum_ptx` | 低（~半天） | 需了解 NPU 矩阵单元最小粒度 |
| 硬编码分支适配 | **中-高**（视精度要求，1-3 天） | 通用分支可能精度不足，专用分支需懂 NPU 微架构 |
| 微基准校准 | 中（~1-2 天，需 NPU 可运行） | 校准质量直接决定精度 |
| 验证调优 | 中（~1-2 天） | 需 ground truth 数据 |

**总体**：最小可跑通 ~2 天，生产级精度 ~1-2 周。主要不确定性在 op 层 GPU 假设的处理——这要求接入者既懂 TileSight 的建模逻辑，又懂 NPU 的微架构特性。

## 总结表

| 问题 | 结论 |
|---|---|
| TileSight 用途 | tile-centric 第一性原理性能建模，预测延迟/cache/分布式，指导优化 |
| vs profiling | 预测式 vs post-hoc，可枚举配置，白盒可解释，跨架构 |
| vs aiconfigurator | 互补——TileSight 是建模引擎（下层），aiconfigurator 是部署搜索（上层）；TileSight 可作其后端 |
| PD 分离 | ❌ 不支持部署架构层（仅支持 attention 的 prefill/decode kernel 建模） |
| 异构硬件 | ❌ 不支持（ClusterSpec 单 arch，同构假设） |
| 新 NPU 接入 | 3 步：Arch profile + 处理 op 层 GPU 硬编码 + 微基准校准；最小 2 天，生产级 1-2 周 |
