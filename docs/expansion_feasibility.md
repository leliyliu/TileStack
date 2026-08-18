# TileSight 扩展适用性评估

> 本文档评估使用并扩展 `src/tilesight/` 完成三项任务的能力：(1) 指导国产芯片（摩尔线程、沐曦）kernel 优化与瓶颈定位；(2) 协助异构推理框架设计（参数调节、阶段-芯片匹配、并行策略选择）；(3) 离线评估快速反映线上效果（pd 分离、hicache 命中）。每项给出"能否完成 + 现状证据 + 缺口 + 扩展方案"。所有结论附源码位置。

## 评估结论速览

| 任务 | 现状能否完成 | 核心缺口 | 扩展难度 |
|---|---|---|---|
| 1. 国产芯片 kernel 优化与瓶颈定位 | ⚠️ 部分可完成（架构层 + 诊断框架已具备，op 层 GPU 硬编码需适配） | 无国产 profile；op 层 38 处 GPU 硬编码分支 | 中（每芯片 1-2 周） |
| 2. 异构推理框架设计指导 | ❌ 基本不能完成（需新建异构调度层） | 同构集群假设；无 trace 驱动；无并行策略搜索；无阶段-芯片匹配 | 高（需新建子系统） |
| 3. 离线评估反映线上（pd 分离/hicache） | ❌ 不能完成（需新建部署模拟层） | 无 pd/af 分离；无 hicache；无请求调度 | 高（需新建子系统） |

**总体判断**：TileSight 是一个 **kernel 级性能建模引擎**，其能力边界止于"单 kernel / 单 op 的 tile 执行 + 分布式 collective 通信"。任务 1 落在其能力域内（需适配国产芯片）；任务 2、3 属于**部署调度层**，超出 TileSight 现有能力，需要在其之上新建调度模拟层（或与 aiconfigurator/Vidur 类工具集成）。

---

## 任务 1：指导国产芯片 kernel 优化与瓶颈定位

### 1.1 需求拆解

- **标定上限**：为未充分调优的国产芯片 kernel 给出理论性能上限（roofline）
- **定位瓶颈**：指出现有 kernel 的瓶颈在哪个硬件单元（计算/DDR/L2/SMEM/...）
- **目标芯片**：摩尔线程（MUSA/Moore Threads）、沐曦（Metax）

### 1.2 现状能力（✅ 已具备的部分）

| 能力 | 源码证据 | 说明 |
|---|---|---|
| Roofline 上限计算 | `matmul_pipeline_wave.py:389-393` | `ddr_time_roofline = total_ddr/ddr_bandwidth`，`compute_time_roofline`，给出 DDR/计算两个理论上限 |
| 9 维瓶颈分解 | `overlap_analysis.py:HardwareUsage:30` | ddr/l2/l1_5/smem/tmem/tensor/cuda/sfu/network 9 个 pipeline 时间，`total_full_overlap` 取 max 定位瓶颈单元 |
| 利用率报告 | `resource_types.py:PipelineResult` | `ddr_util`/`l2_util`/`compute_util`/`smem_util`/`tiles_per_sm`/`l2_hit_rate` |
| 与实测对比验证 | `compare_with_ncu/process_statistics.py:6` | 对比 NCU 实测的 DDR util/L2 hit/SMEM footprint/TC util，验证瓶颈诊断准确性 |
| 多级 cache 建模 | `tile_cache/cache_model.py:15` + `util/sdcm.py` | L1.5+L2 两级 SDCM cascade，可定位 cache 瓶颈 |
| 非 GPU 芯片先例 | `arch/cgra_ver20.py`、`arch/tpuv4.py` | 已有 CGRA/TPU profile，证明架构层支持非 NVIDIA 芯片 |

### 1.3 缺口（⚠️ 需适配的部分）

**缺口 1：无国产芯片 profile**（确认：`ls arch/ | grep -i moore\|metax` 为空）
- 需为摩尔线程、沐曦各写一个 `Arch` 子类
- 参考 `cgra_ver20.py`（非 GPU 先例）：填 `core`/`sm_count`/`tensor_core_shape`/`ddr_bandwidth` 等字段

**缺口 2：op 层 38 处 GPU 硬编码分支**（核心障碍，证据确凿）

op 建模层（`fused_op_pipeline_wave/`）大量硬编码 NVIDIA GPU 假设：

| 文件 | 硬编码数 | 典型代码 |
|---|---|---|
| `matmul_pipeline_wave.py` | 23 处 | `arch.core in ("A100","H100","B200")`（L2 two-part cache）、`arch.core == "V100"`、`mma_type in ("wgmma","utcmma_cta1/cta2")` |
| `reduce_pipeline_wave.py` | 6 处 | 同上 |
| `conv_pipeline_wave.py` | 6 处 | 同上 |
| `elementwise_pipeline_wave.py` | 3 处 | 同上 |

**关键断言**（`matmul_pipeline_wave.py:146`）：
```python
assert hasattr(arch, 'get_tensor_core_minimum_ptx'), \
    "arch.get_tensor_core_minimum_ptx does not exist"
```
仅 `b6000.py`/`h100_sxm.py`/`h200_sxm.py` 实现了该方法（3 个 NVIDIA profile），国产芯片不实现会直接 assert 失败。

**影响**：国产芯片会落入 `else` 通用分支，能跑但精度可能下降；要高精度需新增国产芯片专用分支。

### 1.4 扩展方案

**步骤 1：编写国产芯片 Arch profile**（每芯片 ~1 天）
- 继承 `Arch`，填硬件字段（SM 数/矩阵单元形状/带宽/片上存储）
- 摩尔线程 MTT S80/S4000、沐曦 MX C500/N260 等需查规格书
- 关键：`use_tensor_core_resource_model=False`（走非 GPU 资源模型路径，`arch_base.py:8`）
- 实现 `get_tensor_core_minimum_ptx`（返回矩阵单元最小执行粒度，绕过 assert）

**步骤 2：处理 op 层硬编码**（每芯片 2-5 天，视精度要求）
- 最小接入：让芯片落入 `else` 通用分支，用最接近的 `mma_type`
- 中等接入：在关键 `arch.core` 分支新增国产芯片判断（如 two-part cache 结构是否类似）
- 完整接入：新增 `mma_type="mthreads_mma"`/`"metax_mma"` 专用分支

**步骤 3：微基准校准**（每芯片 1-2 天，需芯片可运行）
- 实现 `set_to_microbench`：带宽扫描（不同 working set）+ 短矩阵乘探针
- 这一步决定精度（论文 §3.8「lightweight microbenchmarks, ~minutes」）

**步骤 4：瓶颈诊断工作流**（复用现有能力）
```
跑现有 kernel → NCU 实测（ground truth）
                          ↓
TileSight 预测 → 对比 NCU → 定位偏差最大的 pipeline 维度
                          ↓
HardwareUsage 9 维分解 → 瓶颈单元（DDR? TC? SMEM?）
                          ↓
roofline 上限 → 当前 kernel 距上限多少 → 优化空间
```

### 1.5 任务 1 结论：⚠️ 部分可完成

- **标定上限**：✅ 可完成（roofline 计算已具备）
- **定位瓶颈**：✅ 可完成（9 维分解 + NCU 对比已具备）
- **国产芯片支持**：⚠️ 需适配（写 profile + 处理 38 处硬编码 + 微基准校准）
- **整体**：架构层与诊断框架已具备，核心工作是国产芯片适配。每芯片 1-2 周可达生产级精度。**这是三个任务中最现实的。**

---

## 任务 2：协助指导异构推理框架设计

### 2.1 需求拆解

- **参数调节**：给定负载，调优并行策略（TP/EP/SP/CP/DP/PP）、batch size 等
- **阶段-芯片匹配**：评估哪些卡适合 prefill / decode / attention 阶段
- **pd 分离 / af 分离**：建模分离架构的调度效果
- **trace 驱动**：输入请求 traces 评估特定负载下的最优配置

### 2.2 现状能力（❌ 基本不具备）

| 能力 | 现状 | 证据 |
|---|---|---|
| 并行策略搜索 | ❌ 无 | `distributed/parallelism.py` 只有 `ParallelScheme` 数据结构，无 `all_parallel_schemes`/`filter_*` 搜索函数（那是 mosaic 的，已不在本分支） |
| 异构硬件 | ❌ 同构假设 | `device.py:53` `NodeSpec.device_arch: Arch` 单 arch；`ClusterSpec` 单 node 类型；`tile_distribution.py:78` 均匀切分 |
| trace 驱动 | ❌ 无 | grep `trace` 仅命中 MoE routing trace（`all_to_all.py:52`，是 EP 算子内部用），无请求 trace 解析 |
| 阶段-芯片匹配 | ❌ 无 | 无 worker role / phase assignment 概念 |
| pd/af 分离 | ❌ 无 | grep `disaggregat`/`prefill.*pool`/`af_split` 全为空 |

### 2.3 TileSight 能提供的底层支持

虽然 TileSight 不具备调度层能力，但可为异构框架设计提供**单点性能数据**：

| 可提供 | 如何提供 |
|---|---|
| 某芯片跑某 op（GEMM/Attention）的延迟 | `calculate_matmul_pipeline_wave`/`fa_*_wrapper` 单卡建模 |
| 某 parallelism 下某 op 的通信开销 | `distributed_op.model_distributed_op`（但假设同构） |
| prefill vs decode kernel 各自延迟 | `fa_prefill_wrapper`/`fa_decode_wrapper` 分别建模（kernel 级，非部署级） |
| 不同芯片的算力/带宽参数 | 各芯片 `Arch` profile |

### 2.4 缺口与扩展方案

任务 2 需要**在 TileSight 之上新建异构调度模拟层**，TileSight 仅作性能后端：

```
新建层（需开发）:
  ├─ 异构集群描述: ClusterSpec 支持多 arch NodeSpec 列表
  ├─ 阶段-芯片分配: 把 prefill/decode/attention 分配到不同芯片池
  ├─ 并行策略搜索: 枚举 TP/EP/SP/... 组合，调用 TileSight 评估每个
  ├─ trace 驱动: 解析请求 trace，模拟调度
  └─ pd/af 分离调度: 建模 KV 迁移、请求路由
TileSight 现有层（可复用）:
  ├─ Arch profile（多芯片性能参数）
  ├─ 单 op 延迟建模
  └─ distributed collective 通信建模（需改造支持非对称）
```

**关键改造点**：
1. `ClusterSpec` 改为 `nodes: List[NodeSpec]`（异构）——`device.py:71` 当前单一 node
2. `DistributedTileMap.local_grids` 改为按芯片算力加权切分——`tile_distribution.py:78` 当前均匀
3. collective 支持非对称（快慢设备混部）——`collectives/*.py` 当前假设等速
4. 新建 trace 解析器 + 调度模拟器

### 2.5 任务 2 结论：❌ 基本不能完成

- TileSight 是 kernel 级建模引擎，**不含部署调度层**
- 异构、pd/af 分离、trace 驱动、并行策略搜索均需新建子系统
- 可复用 TileSight 的 Arch + 单 op 建模 + collective 通信作为性能后端
- **建议**：要么在 TileSight 之上自建调度模拟层（工作量大），要么与 aiconfigurator/Vidur 类工具集成（它们已有调度层，TileSight 替换其性能后端）

---

## 任务 3：离线评估快速反映线上效果（pd 分离 / hicache）

### 3.1 需求拆解

- **pd 分离**：离线评估 prefill/decode 分离部署的线上效果
- **hicache**：建模 KV cache / prefix cache 命中对延迟的影响
- **快速反映线上**：离线模型预测值与线上实测一致

### 3.2 现状能力（❌ 不具备）

| 能力 | 现状 | 证据 |
|---|---|---|
| pd 分离 | ❌ 无 | grep `disaggregat`/`prefill.*pool`/`decode.*pool` 全空；`distributed/` 无 worker role |
| hicache / KV cache 命中 | ❌ 无 | grep `hicache`/`kv.*cache`/`prefix.*cache` 全空（仅 L2/L1.5 硬件 cache 命中，非推理 cache） |
| 请求调度模拟 | ❌ 无 | 无请求队列/batch 调度/连续批处理建模 |
| 线上指标（TTFT/TPOT/吞吐） | ❌ 无 | 无端到端 serving 指标，只到 kernel/op 延迟 |

**重要区分**：TileSight 的 cache 命中建模（`tile_cache/`、`util/sdcm.py`）是**硬件 cache（L1.5/L2）的 tile reuse distance 呺中**，用于 kernel 级 DDR 流量估算，**不是推理系统的 KV cache / prefix cache 命中**。两者完全不同：
- 硬件 cache 命中：tile 在 SMEM/L2 是否命中，影响 kernel 内 DDR 访问
- 推理 cache 命中：KV cache / prefix 在 HBM 是否命中，影响 attention 的 KV 读取量

### 3.3 缺口与扩展方案

任务 3 需要**新建推理部署模拟层**，TileSight 仅能提供底层 kernel 延迟：

| 线上场景 | 需新建能力 | TileSight 可提供 |
|---|---|---|
| pd 分离 | prefill pool / decode pool 分离调度 + KV 迁移建模 | 各自 kernel 延迟（fa_prefill/decode） |
| hicache | KV cache 命中率 → KV 读取量 → attention 延迟 | attention 延迟（给定 KV 量） |
| 连续批处理 | 请求队列调度 + batch 组装 | 不同 batch 的 kernel 延迟 |
| TTFT/TPOT | 端到端指标聚合 | 各阶段延迟求和 |

**扩展路径**：
1. 新建 `serving/` 子包：请求 trace 解析、调度模拟、指标聚合
2. hicache 层：trace 统计 prefix 命中率 → 调整 attention 的 KV 读取量参数 → 调 TileSight 的 `fa_decode_wrapper(cached_kv=...)`
3. pd 分离层：分别建模 prefill worker / decode worker，叠加 KV 迁移（用 `collectives/all_to_all` 估算）
4. 与线上 A/B 对齐校准：用线上实测反校 TileSight 的 microbench 参数

### 3.4 任务 3 结论：❌ 不能完成

- TileSight 的 cache 建模是硬件 cache，**不是推理 cache**
- pd 分离、hicache、请求调度均需新建部署模拟层
- TileSight 仅能提供单 kernel 延迟作为底层输入
- **建议**：与 vLLM/SGLang 的调度器或 Vidur 类模拟器集成，TileSight 提供精确的 kernel 级延迟

---

## 总体结论与建议

### 能力边界定位

TileSight 是 **kernel 级 tile-centric 性能建模引擎**，能力域：

```
✅ 能力域内                          ❌ 能力域外（需新建层）
─────────────────                    ─────────────────
单 kernel 延迟预测                    请求调度 / 连续批处理
硬件 cache 命中（L1.5/L2）            推理 cache（KV/prefix）
pipeline overlap 建模                 pd/af 分离部署
分布式 collective 通信               异构硬件调度
瓶颈诊断（9 维分解）                  trace 驱动 serving 模拟
roofline 上限                        端到端 TTFT/TPOT
```

### 三任务可行性

| 任务 | 可行性 | 原因 |
|---|---|---|
| 1. 国产芯片 kernel 优化 | ⚠️ **可行，需适配** | 落在能力域内，诊断框架已具备，需写 profile + 处理 op 层硬编码 |
| 2. 异构推理框架设计 | ❌ **不可行，需新建层** | 异构/trace/并行搜索/pd-af 均在能力域外 |
| 3. 离线反映线上（pd/hicache） | ❌ **不可行，需新建层** | pd 分离/hicache/调度均在能力域外 |

### 扩展建议

**对任务 1（推荐推进）**：
- 这是 TileSight 的核心能力域，投入产出比最高
- 优先级：摩尔线程 profile → 沐曦 profile → op 层适配 → 微基准校准
- 每芯片 1-2 周可达生产级精度，可复用现有 roofline + 9 维诊断 + NCU 对比工作流

**对任务 2、3（建议集成而非自建）**：
- 自建调度模拟层工作量大（数月），且偏离 TileSight 的 kernel 建模定位
- 推荐路径：**TileSight 作为性能后端，集成到现有推理模拟器**
  - 任务 2：与 aiconfigurator（有部署配置搜索）或 Vidur（有 serving 模拟）集成
  - 任务 3：与 vLLM/SGLang 调度器或 Vidur 集成，TileSight 提供 kernel 延迟
- 若必须自建，建议分层：先做 `serving/` 调度层（任务 3 的 pd/hicache），再扩展异构（任务 2），复用 TileSight 的 Arch + 单 op 建模

### 关键风险

1. **op 层 GPU 硬编码**（38 处）是国产芯片适配的最大不确定性，需逐芯片验证 `else` 通用分支的精度
2. **专有 `.so`**（`distributed/noc/_model_support.so`）仅 Linux x86-64，分布式建模在其它平台受限
3. **tilelang/tvm 可选依赖**（`tir_interface/`）若需 TIR 分析需额外装
4. **TileSight 不是模拟器**：它预测单配置延迟，不模拟动态调度，任务 2/3 的"动态"场景需上层模拟器
