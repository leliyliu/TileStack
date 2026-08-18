# TileSight 扩展深化调研与开发计划

> 本文档基于三个任务的深化调研，给出完整开发计划。任务 1 聚焦"tile 粒度理想调度上限"建模（非 roofline）指导国产芯片 kernel 优化；任务 2/3 调研现有推理框架模拟器并规划接入。所有结论附源码位置或外部来源。

## 0. 总体策略

三个任务构成一个层次化体系，自底向上：

```
任务1 (底层)                    任务2/3 (上层)
┌──────────────────────┐    ┌──────────────────────────────┐
│ TileSight 内核扩展    │    │ 推理模拟器接入层              │
│ ├─ tile 级理想上限    │───→│ ├─ 国产芯片 kernel 延迟后端   │
│ ├─ 国产芯片适配       │    │ ├─ PD/AF 分离调度             │
│ └─ 瓶颈诊断工作流     │    │ ├─ hicache 命中建模           │
└──────────────────────┘    │ └─ trace 驱动 + 异构调度       │
                            └──────────────────────────────┘
```

**核心思路**：任务 1 扩展 TileSight 内核（让它支持国产芯片 + 输出理想上限）；任务 2/3 不在 TileSight 内自建调度层，而是**接入现有推理模拟器**，TileSight 作为 kernel 级性能后端。

---

## 1. 任务 1 深化：tile 粒度理想调度上限建模

### 1.1 用户需求澄清（与 roofline 的区别）

用户明确：**不要 roofline 上限**（`max(DDR_time, compute_time)` 过于粗糙，不可探索）。要的是 **tile 粒度的"理想调度上限"**——假设 tile 能被完美调度（最优拓扑序、满 occupancy、满 overlap）下的基本上限，再与实测 profiling 对比定位瓶颈。

### 1.2 TileSight 已有的"理想上限"机制（✅ 基础已具备）

关键发现：TileSight 的 `fused_op_pipeline_wave/overlap_analysis.py` 已实现理想调度上限的三个层次，比 roofline 细致得多：

| 层次 | 机制 | 源码证据 | 含义 |
|---|---|---|---|
| **单 tile 内** | `steady = max(mem, comp)` | `pipeline_overlap.py:185` | 稳态 load-compute 完全 overlap，无串行浪费 |
| **多 op 融合** | `model_overlap` 枚举所有合法拓扑序取 `min` | `overlap_analysis.py:307-313` `best_lat = min(σ)` | tile-action DAG 的最优调度（论文 Eq.5） |
| **多 pipeline 独立流水** | `total_full_overlap`（各单元取 max） | `overlap_analysis.py:54` + `simulate_schedule:265` `lat = total.total_full_overlap`（stage>=2） | TC/CUDA/SFU/DDR 各自独立流水，无跨单元串行 |

这三层合起来，就是"假设 tile 能被理想调度"的上限：**最优拓扑序 + 完全 overlap + 各 pipeline 独立流水**。对应论文 §3.4 Eq.5 `Tsteady = min_σ max_r Σ u_r(o)`。

### 1.3 与 roofline 的本质区别

| 维度 | roofline | TileSight tile 级上限 |
|---|---|---|
| 粒度 | 整 kernel 两个标量（FLOP/bytes） | per-tile 9 维资源向量 |
| 调度 | 无 | 枚举合法拓扑序取最优 |
| overlap | max(DDR, compute) | 9 个 pipeline 独立流水取 max + load-compute overlap |
| occupancy | 无 | `min(smem, reg, max_blocks)` |
| wave | 无 | head/tail 精确建模 |
| cache | 无 | L1.5+L2 两级 SDCM cascade |
| 可探索性 | 不可（固定公式） | 可（改变 tile shape/stage/swizzle 重新算） |

**结论**：TileSight 的上限是"给定 tile 配置 + 理想调度"的可探索上限，用户可枚举不同 tile shape/stage 看上限变化，这正是指导优化的关键。

### 1.4 任务 1 完整开发计划

#### 阶段 1.1：国产芯片 Arch profile（每芯片 1-2 天）

调研所得国产芯片信息：

| 芯片 | 关键规格 | 工具链 | 来源 |
|---|---|---|---|
| 摩尔线程 MTT S4000 | Tensor Core、48GB、768GB/s、450W | MUSA SDK + `musify`（CUDA 自动转换）+ `torch_musa` | en.mthreads.com/product/S4000 |
| 摩尔线程 MTT S80 | 4096 MUSA 核、1.8GHz、14.7 TFLOPS FP32、16GB GDDR6、448GB/s | 同上 | en.mthreads.com/product/S80 |
| 沐曦 C500 | A100 的 0.85-0.9x（训练）、CUDA 兼容 | MXMACA 平台 + MCCL | metax-tech.com |

**开发动作**：
- 新建 `src/tilesight/arch/mthreads_s4000.py`、`mthreads_s80.py`、`metax_c500.py`
- 继承 `Arch`，填字段（参考 `cgra_ver20.py` 非 GPU 先例）
- 关键：实现 `get_tensor_core_minimum_ptx`（绕过 `matmul_pipeline_wave.py:146` 的 assert）
- 设 `use_tensor_core_resource_model=False`（走非 GPU 路径，`arch_base.py:8`）

**风险**：沐曦 C500 详细规格（核心数/带宽）公开资料不全，需联系厂商或实测。

#### 阶段 1.2：op 层 GPU 硬编码适配（每芯片 3-5 天）

现状：`fused_op_pipeline_wave/` 共 38 处硬编码（matmul 23、reduce 6、conv 6、elementwise 3）。

**分层适配策略**：
1. **最小接入**（先跑通）：国产芯片落入 `else` 通用分支，用 `mma_type="wmma"`。验证能否跑通 + 粗略精度。
2. **精度适配**（按需）：对偏差大的 op，在 `arch.core` 分支新增国产芯片判断。重点：
   - L2 two-part cache 结构（`matmul:193` `arch.core in ("A100","H100","B200")`）—— 国产芯片是否类似？
   - SMEM 路径（`matmul:222` wmma 分支）—— 国产芯片 SMEM 行为
   - mma_type 映射：摩尔线程 MUSA MMA / 沐曦矩阵单元 → 选最接近的或新增分支

**建议**：先用最小接入跑通，用 NCU/profiler 实测对比定位哪些 op 偏差大，再针对性补分支，避免过度工程。

#### 阶段 1.3：理想上限输出与诊断工作流（3-5 天）

**开发动作**：
- 新增 `fused_op_pipeline_wave/upper_bound.py`：封装"理想上限"输出
  - `ideal_schedule_upper_bound(op_plan, arch)` → 返回最优拓扑序下的延迟 + 9 维利用率
  - 复用 `overlap_analysis.model_overlap(try_all_orders=True)` + `pipeline_overlap.steady=max(mem,comp)`
- 新增 `diagnosis/bottleneck_report.py`：对比实测与理想上限
  - 输入：TileSight 理想上限 + NCU/profiler 实测
  - 输出：每个 pipeline 维度的"理想 vs 实测"利用率差距，定位瓶颈单元
  - 复用 `compare_with_ncu/process_statistics.py:6`（已有 NCU 对比框架）

**诊断工作流**：
```
1. 给定 kernel + tile 配置
2. TileSight 算理想上限（最优调度 + 完全 overlap）
   → 输出: ideal_latency, ideal_util(9维)
3. NCU/profiler 实测现有 kernel
   → 输出: actual_latency, actual_util
4. 对比:
   ├─ latency 差距 = 优化空间
   ├─ 哪个 pipeline 维度 actual << ideal → 瓶颈所在
   └─ 改变 tile shape/stage → 理想上限变化 → 指导优化方向
```

#### 阶段 1.4：微基准校准（每芯片 1-2 天，需芯片可运行）

- 实现 `set_to_microbench`：带宽扫描（不同 working set 的 DDR/L2/SMEM 有效带宽）+ 短矩阵乘探针
- 摩尔线程用 MUSA SDK 跑微基准；沐曦用 MXMACA
- 校准质量直接决定上限精度

**任务 1 里程碑**：
- M1（2 周）：摩尔线程 S4000 可跑通 + 理想上限输出 + NCU 对比
- M2（4 周）：沐曦 C500 可跑通 + 诊断工作流闭环
- M3（6 周）：3-5 个国产芯片 kernel 优化案例验证（参考论文 Table 5 的 1.07-8.97× 提升）

---

## 2. 任务 2/3 深化：推理框架模拟器调研与接入

### 2.1 调研结论：不自建调度层，接入现有模拟器

用户需求（任务 2/3）：异构推理框架设计指导 + 离线反映线上（PD 分离/hicache）。这些是**部署调度层**能力，超出 TileSight 内核域。调研发现推理模拟器生态成熟，**接入优于自建**。

### 2.2 主流推理模拟器对比

| 模拟器 | 维护方 | PD/AF 分离 | 异构硬件 | 并行搜索 | trace 驱动 | 性能后端 | 接入难度 |
|---|---|---|---|---|---|---|---|
| **Vidur** | Microsoft | 开发中（issue 待合） | ❌ | ✅ | ✅ | **需 profiling**（每 GPU 跑一次） | 中 |
| **Frontier** | NetX-lab | ✅ 原生 PDD+AFD | 部分 | ✅ | ✅ | 分析模型 | 中 |
| **ROSS** | scitix | ✅ colocated+disagg | ❌ | ✅ Pareto | ✅ | **XGBoost 预测**（需训练数据） | 中 |
| **DistServe** | UCSD | ✅ PD 分离鼻祖 | ❌ | 部分 | ✅ | 分析模型 | 高（学术原型） |
| **DynoSim** | NVIDIA | ✅ Dynamo 栈 | ❌ | ✅ | ✅ | Dynamo profiling | 中（绑定 NVIDIA） |
| **LLMServingSim 2.0** | 学术 | ✅ | ✅ **原生异构** | 部分 | ✅ | 抽象模型 | 中 |
| **aiconfigurator** | ai-dynamo | ✅ | ❌ | ✅ SLA | 配置驱动 | 测量数据 | 低（黑盒） |

**关键发现**：
- **Vidur** 最成熟但性能后端需 profiling（每 GPU 跑一次微基准），换硬件成本高——**TileSight 可替换其 profiling 后端**，提供纯分析预测
- **Frontier** 原生支持 PDD（PD 分离）+ AFD（Attention-FFN 分离）+ 异构 worker，最贴合任务 2 需求
- **ROSS** 用 XGBoost 预测，需训练数据——TileSight 可作特征工程来源或替换 XGBoost
- **LLMServingSim 2.0** 原生异构硬件，但较新（2026），成熟度待验证

### 2.3 接入策略：TileSight 作为性能后端

**推荐组合**：**TileSight（kernel 延迟后端）+ Frontier（调度模拟）+ Vidur（备选/互补）**

```
┌─────────────────────────────────────────────────┐
│ 上层：推理调度模拟器（Frontier / Vidur）          │
│ ├─ 请求 trace 解析 + 调度（PD/AF 分离、连续批处理）│
│ ├─ 并行策略搜索（TP/EP/SP/...枚举）              │
│ ├─ hicache 命中建模（KV/prefix cache）           │
│ └─ 异构 worker 分配                              │
└──────────────────────┬──────────────────────────┘
                       │ 调用: "配置 X 下, kernel Y 的延迟?"
┌──────────────────────▼──────────────────────────┐
│ 中层：TileSight 性能后端适配层（新建）            │
│ ├─ 实现 Vidur ExecutionTimePredictor 接口        │
│ ├─ 实现 Frontier latency estimator 接口          │
│ └─ 把 serving 配置 → TileSight op 建模调用        │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│ 底层：TileSight 内核（任务 1 扩展后）             │
│ ├─ 国产芯片 Arch profile                         │
│ ├─ 单 op 延迟（GEMM/Attention prefill/decode）   │
│ ├─ distributed collective 通信                   │
│ └─ tile 级理想上限                               │
└─────────────────────────────────────────────────┘
```

### 2.4 hicache 命中建模方案

TileSight 现有 cache 建模是**硬件 cache**（L1.5/L2），非**推理 cache**（KV/prefix）。接入模拟器后：

- **hicache 命中率**由上层模拟器统计（trace 驱动，prefix 匹配）
- **命中后的延迟影响**通过 TileSight 参数传递：
  - KV cache 命中 → attention 的 `cached_kv` 参数减少 → `fa_decode_wrapper(cached_kv=命中部分)` 延迟降低
  - prefix 命中 → prefill 的 `seq` 参数减少 → `fa_prefill_wrapper(seq=剩余长度)`
- TileSight 不需新建推理 cache 模型，只需暴露 `cached_kv`/`seq` 参数接口（已有）

### 2.5 任务 2/3 完整开发计划

#### 阶段 2.1：模拟器选型与接入原型（2-3 周）

**开发动作**：
1. 克隆 Frontier + Vidur，跑通各自的 demo
2. 评估接入点：
   - Vidur：`ExecutionTimePredictorRegistry`（可注册自定义预测器）
   - Frontier：latency estimator 接口
3. 写 `adapter/vidur_adapter.py`：实现 Vidur 的 `ExecutionTimePredictor` 接口，内部调 TileSight
4. 验证：同一配置下，TileSight 后端 vs Vidur 原生 profiling 后端的延迟对比

**里程碑**：TileSight 作为 Vidur 性能后端跑通一个 Llama-7B 端到端模拟

#### 阶段 2.2：PD 分离 + hicache 接入（2-3 周）

**开发动作**：
1. 用 Frontier 的 PDD（PD 分离）抽象，配置 prefill pool / decode pool
2. hicache：在 adapter 层把 trace 统计的 prefix 命中率 → TileSight `fa_*_wrapper` 的 `cached_kv`/`seq` 参数
3. KV 迁移建模：用 TileSight `distributed/collectives/all_to_all` 估算 PD 间 KV transfer 开销
4. 验证：与线上 vLLM/SGLang PD 分离实测对比

**里程碑**：PD 分离 + hicache 场景离线预测与线上对齐

#### 阶段 2.3：异构硬件 + 并行策略搜索（2-3 周）

**开发动作**：
1. 评估 LLMServingSim 2.0 的异构支持是否可用；否则在 Frontier 上扩展异构 worker
2. 并行策略搜索：枚举 TP/EP/SP/CP/DP/PP 组合，TileSight 评估每个组合的 compute + comm 延迟
3. 异构阶段-芯片匹配：把 prefill/decode 分配到不同国产芯片，TileSight 各自建模
4. Pareto 前沿输出：吞吐 vs 延迟 vs 成本

**里程碑**：给定负载 + 异构芯片池，输出最优并行策略 + 阶段分配

#### 阶段 2.4：trace 驱动 + 线上对齐校准（2 周）

**开发动作**：
1. trace 解析器：支持 vLLM/SGLang 的请求 trace 格式
2. 线上指标聚合：TTFT/TPOT/吞吐
3. 用线上 A/B 实测反校 TileSight microbench 参数 + 模拟器调度参数

**里程碑**：离线模拟与线上实测误差 < 15%

---

## 3. 总体开发路线图

### 3.1 三阶段时间线（总计 ~5-6 个月）

```
月份 1-2          月份 3-4          月份 5-6
┌──────────┐    ┌──────────┐    ┌──────────┐
│ 任务1     │    │ 任务2/3   │    │ 整合验证  │
│ 国产芯片  │───→│ 模拟器接入│───→│ 端到端    │
│ +理想上限 │    │ PD/hicache│    │ 异构推理  │
└──────────┘    └──────────┘    └──────────┘
```

### 3.2 详细里程碑

| 阶段 | 时间 | 里程碑 | 验证标准 |
|---|---|---|---|
| 1.1 | 月1 W1-2 | 摩尔线程 S4000 Arch profile | `calculate_matmul_pipeline_wave` 可跑通 |
| 1.2 | 月1 W3-4 | op 层硬编码适配 + 微基准校准 | GEMM MAPE < 20% vs NCU |
| 1.3 | 月2 W1-2 | 理想上限输出 + 诊断工作流 | 3 个 kernel 定位瓶颈并验证优化方向 |
| 1.4 | 月2 W3-4 | 沐曦 C500 接入 + 案例验证 | 1.5×+ 优化案例 ≥ 2 个 |
| 2.1 | 月3 W1-3 | Vidur/Frontier 接入原型 | Llama-7B 端到端模拟跑通 |
| 2.2 | 月3 W4-月4 W2 | PD 分离 + hicache | 与 vLLM PD 实测误差 < 20% |
| 2.3 | 月4 W3-月5 W2 | 异构 + 并行搜索 | Pareto 前沿输出 |
| 2.4 | 月5 W3-4 | trace 驱动 + 线上对齐 | 离线 vs 线上误差 < 15% |
| 3 | 月6 | 端到端整合 | 国产芯片异构推理框架设计指导闭环 |

### 3.3 人员与技能需求

| 角色 | 技能 | 负责 |
|---|---|---|
| TileSight 内核扩展 | Python + GPU 微架构 + 论文理解 | 任务 1 |
| 国产芯片适配 | 摩尔线程 MUSA / 沐曦 MXMACA 工具链 | 任务 1 微基准 |
| 模拟器接入 | Python + Vidur/Frontier 架构理解 | 任务 2/3 |
| 推理框架 | vLLM/SGLang 调度 + PD 分离实践 | 任务 2/3 验证 |

### 3.4 关键风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 国产芯片规格不全（沐曦 C500） | Arch profile 精度 | 联系厂商 + 实测 microbench |
| op 层 38 处硬编码适配工作量 | 任务 1 延期 | 最小接入先跑通，按偏差优先级补分支 |
| Vidur 报告 ~700% P50 偏差（vs vLLM 0.9.1） | 任务 2/3 精度 | 用 TileSight 后端替换其 profiling 后端，可能改善；或选 Frontier |
| Frontier 成熟度（学术项目） | 接入稳定性 | 同步评估 Vidur 作备选 |
| 沐曦/摩尔线程 CUDA 兼容性 | 微基准可运行 | 两者均声称兼容（musify/MXMACA），先跑通带宽+GEMM 微基准 |
| 线上对齐误差 | 任务 3 可信度 | 用线上 A/B 反校，分层校准（kernel→op→serving） |

### 3.5 决策点

1. **月1 W4**：摩尔线程 op 层适配后，若 `else` 通用分支 MAPE > 30%，需投入更多专用分支开发——评估是否值得
2. **月3 W3**：Vidur 接入后，若与 vLLM 实测仍偏差大，切换到 Frontier 或评估 LLMServingSim 2.0
3. **月5 W2**：异构支持若 Frontier/LLMServingSim 都不理想，需在模拟器层自建异构 worker 抽象——评估工作量

---

## 4. 立即可启动的首批工作

无需等待，可立即启动（任务 1 阶段 1.1-1.3）：

1. **写摩尔线程 MTT S4000 Arch profile**（规格已从调研获得）
2. **实现 `get_tensor_core_minimum_ptx`** 绕过 assert
3. **新建 `upper_bound.py`** 封装理想上限输出（复用现有 `model_overlap` + `pipeline_overlap`）
4. **新建 `diagnosis/bottleneck_report.py`** 对比 NCU 实测（复用 `compare_with_ncu`）
5. **跑通一个 GEMM 案例**：TileSight 理想上限 vs 摩尔线程 NCU 实测，验证诊断工作流

这批工作 2-3 周内可完成，将验证整个任务 1 的可行性，为后续任务 2/3 接入提供国产芯片性能后端基础。

---

## 5. 证据索引

### 5.1 源码证据（TileSight 理想上限机制）

| 结论 | 证据 |
|---|---|
| 稳态完全 overlap = max(mem,comp) | `fused_op_pipeline_wave/pipeline_overlap.py:185` |
| 最优拓扑序 min_σ | `fused_op_pipeline_wave/overlap_analysis.py:307-313` |
| 各 pipeline 独立流水 total_full_overlap | `fused_op_pipeline_wave/overlap_analysis.py:54,265` |
| 9 维资源向量 | `fused_op_pipeline_wave/overlap_analysis.py:HardwareUsage:30` |
| NCU 对比框架 | `compare_with_ncu/process_statistics.py:6` |
| op 层 38 处硬编码 | matmul 23 + reduce 6 + conv 6 + elementwise 3 |
| get_tensor_core_minimum_ptx assert | `matmul_pipeline_wave.py:146` |

### 5.2 外部调研证据

| 结论 | 来源 |
|---|---|
| Vidur 架构 + ExecutionTimePredictorRegistry | github.com/microsoft/vidur |
| Frontier PDD+AFD 原生支持 | github.com/NetX-lab/Frontier |
| ROSS XGBoost 预测 | github.com/scitix/ross |
| DistServe PD 分离 4.48× goodput | github.com/LLMServe/DistServe, arxiv 2401.09670 |
| LLMServingSim 2.0 异构 | arxiv 2605.21312 |
| 摩尔线程 S4000 规格 + MUSA/musify | en.mthreads.com/product/S4000, docs.mthreads.com |
| 沐曦 C500 A100 0.85-0.9x + MXMACA | metax-tech.com |
| aiconfigurator 部署搜索 | github.com/ai-dynamo/aiconfigurator |
