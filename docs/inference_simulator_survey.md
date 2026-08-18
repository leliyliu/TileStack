# LLM 推理框架模拟器深度调研报告

> 本文档针对 `docs/dev_plan.md` 任务 2/3 中提到的推理框架模拟器做逐项深度调研与核实，目标是支撑后续选型决策。
> 调研时间：2026-08-18。所有关键论断均附证据（GitHub API 实测数据 / 论文原文 / 仓库 README / arXiv API）。
> **注意：本调研发现 dev_plan.md 存在若干引用错误，已在 §9 勘误，建议先看勘误。**

---

## 1. 调研对象与核实结果总览

| 模拟器 | 维护方 | 是否真实存在(公开) | Stars | 最近 push | License | 一句话定位 |
|---|---|---|---|---|---|---|
| **Vidur** | Microsoft | ✅ | 665 | 2025-07-25（主分支已停滞 >1 年） | MIT | 最经典的 LLM 推理模拟器（MLSys'24） |
| **Frontier** | CUHK NetX-lab | ✅ | 84 | 2026-08-17（非常活跃） | MIT | 基于 Vidur 重构的新一代离散事件模拟器，PDD+AFD 原生支持 |
| **ROSS** | scitix | ✅ | 2 | 2026-07-10 | ⚠️ **无 License 文件** | XGBoost 预测 + 复用真实框架调度器的离线预测器 |
| **DistServe** | 北大/普林斯顿等 | ✅ | 830 | 2025-04-06（已停滞） | Apache-2.0 | PD 分离鼻祖（OSDI'24），仓库内含 simdistserve 模拟器 |
| **DynoSim** | "NVIDIA" | ❌ **未找到任何公开证据** | - | - | - | dev_plan 中疑似误引，见 §9 勘误；最接近的真实项目是 Dynamo 的 "AI Simulate" |
| **LLMServingSim 2.0** | KAIST casys | ✅ | 365 | 2026-07-29（活跃） | 未验证（API 限流） | cycle 级异构 + 分离式服务模拟器（ISPASS'26） |
| **aiconfigurator** | NVIDIA ai-dynamo | ✅ | 411 | 2026-08-18（非常活跃） | Apache-2.0 | 部署配置搜索器（偏黑盒，输出可部署配置） |

数据来源：GitHub REST API 实测（2026-08-18）；ROSS 无 License 文件经本地 clone 验证（`ls /tmp/pi-github-repos/scitix/ross/`）。

---

## 2. 选型关键维度说明

结合 TileSight 的接入目标（TileSight 作为 kernel 级性能后端替换模拟器原有预测层），选型最关心的维度：

1. **性能后端类型**：分析公式 / 机器学习回归 / 实测 profiling 数据库 —— 决定 TileSight 的替换点在哪里
2. **PD 分离 / AF 分离**：任务 2 的核心需求
3. **异构硬件支持**：能否建模"prefill 用芯片 A、decode 用芯片 B"
4. **国产芯片可行性**：新增硬件需要付出什么代价（重新 profiling？训练数据？）
5. **trace 驱动**：能否回放线上请求 trace
6. **模拟精度**：与真实系统（vLLM/SGLang）对比的已发表误差数据
7. **代码健康度**：活跃度、license、社区规模
8. **接口扩展点**：自定义性能预测器的注册机制

---

## 3. 逐项详评

### 3.1 Vidur（Microsoft）

**基本信息**
- 仓库：github.com/microsoft/vidur，MIT，665 stars，创建于 2023-11，主分支最后 push 2025-07-25（GitHub API 实测）
- 论文：**MLSys 2024**，arXiv:2405.05465《Vidur: A Large-Scale Simulation Framework For LLM Inference》

**架构与性能后端**
- 组合式：实验 profiling + 预测建模（paper abstract: "a combination of experimental profiling and predictive modeling"）
- 核心是 `ExecutionTimePredictorRegistry`（源码证据：`vidur/scheduler/global_scheduler/base_global_scheduler.py` 直接 import 该注册类），默认用 Random Forest 预测器（README CLI 参数 `--random_forrest_execution_time_predictor_config_*`）——**这是 TileSight 接入的自然扩展点，dev_plan 判断正确**
- 每 GPU×模型组合需跑一次 profiling（README "without access to GPUs except for a quick initial profiling phase"）
- 支持 TP/PP 任意组合、chrome trace 导出、wandb 指标（README）
- **prefix caching / 新路由策略在 `canary` 分支，主分支没有**（README 原文："we have been working on several improvements... please use the `canary` branch"）

**能力矩阵**
- PD 分离：❌ 仍是 open PR #80《[Model][Core] Add DeepSeek-V3 and Mixtral-8x7B MoE configs + disaggregated P/D s》（GitHub issue API 实测，total 1 条相关，未合并）
- 异构硬件：❌ 仅 profiling 过的 SKU（A100/H100/A40 等，README 支持矩阵）
- trace 驱动：✅（AzureLLMInferenceTrace / Splitwise trace，README）
- 配置搜索：✅ Vidur-Search（论文：LLaMA2-70B 最优配置 CPU 上 1 小时 vs 42K GPU 小时）

**精度（关键！）**
- 论文自报：**推理延迟误差 < 9%**（arXiv:2405.05465 abstract 原文 "estimates inference latency with less than 9% error across the range"，2024 年对其 profiling 的系统验证）
- **第三方实测（Frontier 论文 §5，H800 BF16）**：
  - attention 算子 p50/p95 误差 **55.4% / 376.1%**；FlashAttention 平均误差 32.6%
  - 端到端误差 2.9%–45.5%（dense only）
  - 不支持 PDD、不支持 FP8
- dev_plan 中 "Vidur 报告 ~700% P50 偏差" **未找到任何依据**（见 §9 勘误）

**结论**：接口设计最好（注册器模式），但主分支停滞、无 PDD、精度在新硬件/新版本 vLLM 上退化明显。适合作为**接口范式参考 + 备选**。

---

### 3.2 Frontier（CUHK NetX-lab）★ 与任务 2/3 最贴合

**基本信息**
- 仓库：github.com/NetX-lab/Frontier，MIT，84 stars，创建 2026-05-21，最后 push 2026-08-17（GitHub API 实测）
- 论文：两个版本——arXiv:2508.03148（v1，2025-08）与 **arXiv:2605.21312**（v2, 2026，《Frontier: Towards Comprehensive and Accurate LLM Inference Simulation》）
- 作者：Yicheng Feng 等，CUHK（Hong Xu 组）；**README 明确声明 "Frontier is mainly built on top of Vidur"**，并参考了 ASTRA-Sim、htsim、aiconfigurator

**能力矩阵（全部来自 README + 论文原文）**
- **服务架构**：co-location / **PDD** / **AFD**（prefill、decode-attention、decode-FFN 三角色分离）——README "Latest News [2026/08]: PDD and AFD support is available in the new release"
- **运行时优化建模**（不是简单加速因子，README："models them as runtime behavior rather than simple speedup factors"）：CUDA Graph、speculative decoding/MTP、prefix caching、量化、chunked prefill、**hierarchical caching（分层 KV cache，即 hicache 场景！）**
- **异构 GPU**：✅ 论文 §6.2 在 1024-GPU 集群做异构角色分配（H800 + H20 混布，用 per-role stage metrics + counterfactual 判定哪些角色可用廉价卡）——**这是 dev_plan 阶段 2.3 需要的能力**
- **trace 驱动**：✅ ShareGPT trace replay、agentic 多轮 trace、RL rollout trace（4000 轨迹 burst）
- **并行策略搜索**：✅ 论文 §6.1 在 256×H800 上对 Llama-3.3-70B 全量扫描 **483,536 个候选配置**（PP/TP/DP/EP/DP-Attention），内存过滤 65,190 个 OOM 项，496 个满足 SLA
- **通信后端可插拔**：analytical（公式）/ ASTRA-Sim 拓扑 / htsim（collective_sim，需编译）（README Communication Backend 节）
- **MoE / 状态ful 请求**（reasoning、tool call、RL rollout 的 prefix 连续性）均支持
- CPU-only 模拟（1K+ GPU 规模），新硬件 profiling 只需 1 张 GPU（README Minimum Hardware Requirements）

**性能后端（TileSight 替换点）**
- 论文 §3.4：三组件 = Compute Operator Library（per-op 预测器）+ Memory-Capacity Model + Comm Backend
- 算子分三类：token-count 类（线性回归）／序列依赖 attention 类（随机森林，特征含 per-request 长度分布）／MoE 路由类（随机森林，特征含 expert 负载方差）
- profiling 在单 GPU 上按 shard 模式跑，kernel-only 与 launch-inclusive 两种模式
- **目前只有 h800 和 rtx_pro_6000 有全特征 profiling 数据集**（README 原文），其他硬件需自己采集——**这正是 TileSight 分析后端的切入机会**

**精度（论文 §5 原文数字，16×H800 验证）**
- 算子级 p50/p95：attention 3.5%/14.2%，linear 3.3%/6.4%，GroupedGEMM 1.4%/5.3%
- 端到端：co-location **6.4%**、disaggregation **2.6%**（此前 SOTA 分别为 44.9%/51.7%）；平均吞吐误差 <4%
- 对比基线数字：Vidur E2E 2.9–45.5%（不支持 PDD）、aiconfigurator E2E makespan 最高 **170–200%**
- H20 异构场景全部指标 <10%

**风险**
- 学术项目、社区小（84 stars / 5 open issues）、2026-05 才创建，API 可能不稳定
- 目前只模拟 vLLM 逻辑（README：SGLang/TRT-LLM 在计划中）
- dev_plan 说 Frontier "部分异构"，实际论文已演示完整异构用例——dev_plan 信息略滞后

**结论**：**任务 2/3 首选**。PDD+AFD+hierarchical cache+异构+trace 全部原生，精度最高，MIT 友好，且性能后端是"可校准的 per-op 预测器库"，TileSight 替换路径清晰。

---

### 3.3 ROSS（scitix）

**基本信息**
- 仓库：github.com/scitix/ross，创建 2026-03-30，最后 push 2026-07-10，**仅 2 stars / 0 forks**
- ⚠️ **仓库无 LICENSE 文件**（本地 clone 验证）——所有 rights reserved，商用/内用都有法律风险，除非联系作者授权
- 预训练 XGBoost 模型发布在 HuggingFace：CharlesCAOO/ross（README）

**架构（本地 clone 源码结构验证）**
- **数据面 = XGBoost stage-wise 回归器**：输入模型配置特征（非 per-model kernel 标定），输出各 stage 耗时；预训练于 H200/B200 profiling 数据
- **控制面 = 直接复用真实框架调度器**：`ross/vllm_sim/scheduler/` 与 `ross/sgl_sim/scheduler/` 是对 vLLM/SGLang 调度逻辑的移植（可 sidecar 对比真实 vLLM 调度，`--vllm-src-root`）
- CPU-only，无 GPU 也能跑（README "Installation"）

**能力矩阵**
- PD 分离：✅ 语法 `dp:pp:tp@dp:pp:tp`（如 `1:1:4@1:1:4` = 4 prefill + 4 decode GPU），**KV 迁移计入 virtual-clock 关键路径**（README "Prefill–decode disaggregation" 节）
- prefix cache：✅ `common/prefix_cache.py`、`common/sgl_prefix_tree.py`、`generated-shared-prefix` 合成负载（源码结构 + README）
- Pareto 搜索：✅ `--get-pareto-front`，228 候选 32B/8×B200 约 70 分钟 CPU，自称比实测穷举便宜 **1,258×**
- 模型族：Llama-3.1、Qwen2.5/3（含 MoE）、DeepSeek-V3、gpt-oss（README Supported Models）
- 新 GPU 适配：跑 `collector/` profiling 约 3–4 小时（8 卡节点）+ 重训回归器；官方称预训练模型对新 GPU "practical generalization"
- trace 评估：✅ `--eval` 计算 E2E/TTFT/TPOT/ITL 百分比误差

**亮点**：stage 级误差分解曾定位 SGLang TokenizerManager 的 batch-boundary 瓶颈（修复后高并发延迟降 36%，README）——方法论与 TileSight "分层定位瓶颈"思想一致。

**结论**：技术路线与 TileSight 互补性最强（XGBoost 特征可由 TileSight 提供，或直接被 TileSight 分析模型替换），但 **无 License + 2 stars 社区风险大**，只能作为方法论参考或联系作者后再用。

---

### 3.4 DistServe / simdistserve（学术原型）

**基本信息**
- 仓库：github.com/LLMServe/DistServe，Apache-2.0，830 stars，**最后 push 2025-04-06（停滞）**
- 论文：**OSDI 2024**，arXiv:2401.09670（PD 分离 goodput 优化，dev_plan 引用正确）

**关键澄清**：DistServe 本体是**真实 serving 系统**（Ray + SwiftTransformer C++ 后端），不是模拟器。模拟器是其仓库内的 `simdistserve/` 子包（本地 clone 结构验证）：`estimators/time_estimator.py` + `estimators/profile_data/profiler-a100-80g.*.json` + 二分搜索 `benchmarks/search_binary.py`、并行 `parallel_bisect.py`。

**能力限制**（README + 源码结构）
- 模型仅支持 GPT-2 / OPT / LLaMA2（无 Qwen/MoE/DeepSeek）
- 性能数据只有 A100-80G 两份 profile JSON
- 无活跃维护

**结论**：**仅作 PD 分离思想与模拟器评估方法（search_configs/二分搜索）的参考，不建议接入**。dev_plan "接入难度高（学术原型）"的判断成立，且应进一步降级为"不接入"。

---

### 3.5 DynoSim（"NVIDIA"）——❌ 未能证实存在

**核实过程（三层验证均无结果）**：
1. GitHub repository 搜索 "DynoSim"：只有 `RealDarthVader/DynoSim`（0 star，"A fun dyno simulator of an engine"，发动机玩具模拟器）等无关项目，**无任何 NVIDIA 相关仓库**（GitHub Search API 实测）
2. arXiv API 全文检索 "DynoSim"：**totalResults = 0**（export.arxiv.org 实测）
3. Web 搜索（多 provider）：未找到任何 NVIDIA 官方页面提到 "DynoSim"

**最接近的真实项目**：NVIDIA Dynamo 的 **"AI Simulate"**（`aisimulate` 包，docs.nvidia.com/dynamo → knowledge-base → modular-components → ai-simulate）：
- 官方文档实测：定位是 "Experimental backend-neutral configuration search"——黑盒优化器 Sweeper，把每个候选配置变成 `ReplaySpec` 发给注入的 `RunnerFactory` 重放执行，返回 Pareto 前沿
- Python 入口：`from aisimulate.sweeper import SmartSearchConfig, Sweeper`
- **它不是分析型性能模拟器**，而是"配置搜索框架 + 重放执行"，性能数据来自被测系统本身
- Spica 智能扫描器原属 aiconfigurator，现已并入 AI Simulate（aiconfigurator README 原文确认）

**结论**：dev_plan 中的 "DynoSim | NVIDIA | ✅ Dynamo 栈 | Dynamo profiling" 一行**无法证实**，大概率是混淆/幻觉条目，建议从计划中移除；若关心 NVIDIA 生态，跟踪对象应改为 **Dynamo AI Simulate（实验性）**。

---

### 3.6 LLMServingSim 2.0（KAIST casys）

**基本信息**
- 仓库：github.com/casys-kaist/LLMServingSim，365 stars，最后 push 2026-07-29（GitHub Search API 实测）；License 未能验证（API 限流，需后续确认）
- 论文（仓库 Citation 节原文）：
  - **IISWC 2024**（1.0 版）：《LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving at Scale》，DOI 10.1109/IISWC63097.2024.00012
  - **CAL 2025**（2.0 短文）：DOI 10.1109/LCA.2025.3628325
  - **ISPASS 2026**（2.0 正式版）：《LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure》，DOI 10.1109/ISPASS69572.2026.00012
- ⚠️ **dev_plan 把 arXiv:2605.21312 标为 LLMServingSim 2.0 是错误的——该编号是 Frontier v2 论文**（见 §9）

**架构（README "About" 原文）**
- **cycle 级**模拟器：Python 前端（镜像 vLLM continuous-batching 调度器）+ **ASTRA-Sim C++ 分析网络后端** + vLLM 层级 profiler 采集的 per-hardware 延迟数据
- 统一环境研究：**异构加速器**、**分离式内存层级（CPU/CXL/PIM）**、MoE 路由、TP/PP/EP/DP 多实例并行
- 依赖较重：需编译 ASTRA-Sim + Chakra（`./scripts/compile.sh`），有 docker 方案

**与任务 2/3 的关系**
- 异构硬件建模最深（memory tier 级），若未来要做"国产芯片 + CXL 内存扩展"这类架构探索，它是唯一选项
- 但接入成本最高（C++ 组件、cycle 级模拟速度慢于事件级）、对 PD 分离的支持细节论文对比中未给出具体误差数字（Frontier 论文将其列为基线但未报告其数字）
- dev_plan 判断"成熟度待验证"成立

**结论**：**异构架构研究的备选**，不适合作为第一个接入对象。

---

### 3.7 aiconfigurator（NVIDIA ai-dynamo）

**基本信息**
- 仓库：github.com/ai-dynamo/aiconfigurator，Apache-2.0，411 stars / 151 forks，最后 push 2026-08-18（GitHub API 实测，**极活跃**）
- 论文：arXiv:2601.06288《AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving》
- 定位：**不是通用模拟器，是部署配置搜索器**——给定模型/GPU 数量/型号 + SLA，输出可直接部署的 Dynamo / llm-d / FPM 配置文件

**能力矩阵（README 实测）**
- PD 分离：✅（disagg 模式，输出 xPyD replica 配置）
- 异构：✅ 通过 `exp` 模式 YAML 定制（"disagg vs agg, homogeneous vs heterogeneous"，README）
- 后端：trtllm（默认）/ vllm / sglang
- 模型：GPT/LLaMA/Qwen/DeepSeek-V3/MoE
- 性能模型：**实测 silicon 数据库（Parquet）+ 算子插值外推组合**；四种模式 SILICON（默认，可复现）/HYBRID/EMPIRICAL/SOL
- 量化建模最全：FP8/FP8-Block/INT8/INT4/NVFP4/SQ 等全支持
- 搜索速度极快：32 GPU 配置搜索 6.5 秒（README 示例输出）
- 新硬件：提供 collector 工作流自采数据（`--systems-paths`），内置库覆盖 h100/h200/b200/gb200/a100；**无国产芯片数据，需完全自建**
- procurement sizing：`recommend` 模式给出满足 SLA 的最小 GPU 数

**精度**
- Frontier 论文实测其 E2E makespan 误差：decode-heavy dense **170.0%**、balanced MoE **135.3%**、PDD 场景 **200.0%**——分析式外推在大规模/分离场景下误差显著，论文自评与第三方评测有差距

**结论**：作为**配置生成与快速基线工具**很好（低接入成本、可复现），作为**高精度离线模拟**不可靠；国产芯片场景需自采全套数据，TileSight 无法直接替换其内部估计器（Rust core，黑盒程度高）。

---

## 4. 横向对比总表

| 维度 | Vidur | Frontier | ROSS | simdistserve | LLMServingSim 2.0 | aiconfigurator |
|---|---|---|---|---|---|---|
| 性能后端 | RF 预测器（需 profiling） | per-op 线性/RF 预测器（需 profiling） | XGBoost（需训练数据） | 查表 + 分析 | cycle 级 + 层级 profiling | silicon 实测库 + 插值 |
| TileSight 替换点 | ✅ PredictorRegistry | ✅ per-op predictor/DB | ✅ 特征或整模型 | 理论可行但无价值 | profiling 数据 | ❌ 黑盒（Rust） |
| PD 分离 | ❌（PR #80 未合） | ✅ 原生 | ✅（`@` 语法） | ✅（论文核心） | 部分 | ✅ |
| AF 分离 | ❌ | ✅（2026/08 发布） | ❌ | ❌ | 部分 | ❌ |
| hicache/分层 KV | canary 分支 | ✅ hierarchical caching（tair-kvcache 计划中） | ✅ prefix tree | ❌ | 部分（CXL/PIM） | ❌ |
| 异构 GPU | ❌ | ✅（H800+H20 混布已验证） | ❌（H200/B200 训练） | ❌ | ✅（最强） | ✅（exp YAML） |
| trace 驱动 | ✅ | ✅（含 agentic/RL） | ✅（--eval） | ✅ | ✅ | ❌（配置驱动） |
| 并行策略搜索 | ✅ | ✅（48 万配置扫描） | ✅ Pareto | ✅ 二分 | ✅ | ✅（最快） |
| MoE | canary/PR | ✅ | ✅（DeepSeek-V3/Qwen3） | ❌ | ✅ | ✅ |
| 精度（已发表） | 自报 <9%；第三方 E2E 2.9–45.5% | **E2E 6.4%/2.6%，吞吐 <4%** | 自报可 --eval（无公开数字） | 未系统报告 | 未报告 | 第三方实测最高 200% |
| 活跃度（2026-08） | 停滞 1 年+ | 非常活跃 | 活跃但社区极小 | 停滞 | 活跃 | 非常活跃 |
| License | MIT | MIT | ⚠️ 无 | Apache-2.0 | 未验证 | Apache-2.0 |
| 国产芯片接入成本 | 需实机 profiling | 需实机 profiling（1 GPU） | 需实机 profiling 3-4h + 重训 | - | 需层级 profiling | 需全套数据自采 |

---

## 5. 分场景选型建议

### 场景 A：任务 2 主线（PD/AF 分离 + hicache + 线上对齐）→ **Frontier 首选**
理由（证据链）：
1. PDD+AFD+hierarchical caching+prefix cache 全部原生（README Latest News + Key Features）
2. 精度是所有候选中最高的（E2E 2.6–6.4% vs 之前 SOTA 44.9–51.7%）
3. 性能后端 = per-op 可校准预测器，TileSight 替换路径清晰（论文 §3.4）
4. MIT + 基于 Vidur 代码（Vidur 的接口经验可迁移）
5. trace 驱动含 agentic/RL 场景，覆盖任务 3 的"离线反映线上"

### 场景 B：快速部署配置基线 / 采购 sizing → **aiconfigurator 辅助**
活跃度最高、搜索最快（6.5s）、能直接产出可部署配置；但注意其精度短板（第三方实测最高 200% 误差）且无法接入 TileSight 后端——只做交叉验证用。

### 场景 C：异构芯片池架构研究（如 CXL/内存层级）→ **LLMServingSim 2.0 备选**
唯一做到 cycle 级 + 内存层级异构的模拟器；接入成本高，建议在 Frontier 异构能力（已验证 H800+H20）不满足需求时再引入。

### 场景 D：Vidur 的定位 → **接口范式参考 + 备选**
`ExecutionTimePredictorRegistry` 是最干净的替换接口设计，可作为 TileSight adapter 的接口蓝本；但主分支停滞 + 无 PDD，不建议作为主线。dev_plan 中"Vidur 主选/备选"的建议应更新为"Frontier 主选"。

### 场景 E：ROSS → **方法论参考（需先解决 License）**
stage 级误差分解定位瓶颈的方法值得借鉴；无 License 是硬伤，除非作者（scitix）补充授权，不进入交付链。

### 建议组合（更新 dev_plan §2.3）
```
TileSight（kernel 分析后端）
   └─ 替换 Frontier 的 per-op 预测器/profiling 库（主线）
        ├─ aiconfigurator：快速配置基线 + 交叉验证（辅助）
        ├─ Vidur：PredictorRegistry 接口蓝本（参考）
        └─ LLMServingSim 2.0：异构内存层级研究（远期备选）
```

---

## 6. 国产芯片接入要点（与任务 1 的衔接）

| 模拟器 | 新增国产芯片的工作量 | 证据 |
|---|---|---|
| Frontier | 在摩尔线程/沐曦上跑其 profiling 模块（单 GPU 即可），生成 per-op 数据库；或直接用 TileSight 分析模型填充该数据库（免实机） | README "Profiling: At least 1 GPU is required" + 论文 §3.4 |
| Vidur | 每模型×每 SKU 跑一次完整 profiling（成本最高） | README profiling.md |
| ROSS | 3-4h collector profiling + 重训 XGBoost | README "Profiling a new GPU platform" |
| aiconfigurator | collector 全套自采 + `--systems-paths`（无公开国产数据） | README Data Collection |
| LLMServingSim 2.0 | vLLM 层级 profiler 在目标硬件运行（依赖 vLLM 可用性） | README About |

**关键衔接点**：摩尔线程（torch_musa）/沐曦（MXMACA）都声称 CUDA 兼容，理论上 vLLM 类 profiling 脚本可移植——这是任务 1 阶段 1.4 微基准校准的延伸验证项，dev_plan 已列入。

---

## 7. 风险清单（选型后仍需跟踪）

1. **Frontier 成熟度**：84 stars、2026-05 才创建，API 不稳定风险；缓解：锁 commit + fork，同步关注其 SGLang/TRT-LLM 引擎支持进度
2. **Frontier profiling 数据集覆盖**：目前仅 h800/rtx_pro_6000 全特征（README），国产芯片需自采或由 TileSight 填充
3. **LLMServingSim License 未验证**：决定是否可用于交付前必须确认
4. **ROSS License 缺失**：硬性阻断项
5. **Vidur 社区迁移**：大量 Vidur 贡献者转向 Frontier（Frontier 明言基于 Vidur 构建），Vidur 长期维护存疑
6. **aiconfigurator 与 Dynamo AI Simulate 的边界变化**：Spica 已迁移至 AI Simulate，后续 aiconfigurator 功能可能继续拆分（README 已注明）

---

## 8. 建议的下一步验证动作（对应 dev_plan 阶段 2.1）

1. clone Frontier，跑通 `examples/architecture/pdd/offline/dense_model_basic.sh`（analytical 后端可 CPU 一键 smoke）
2. 定位其 per-op 预测器数据格式（h800 数据集），评估"TileSight 分析结果 → Frontier 预测器数据库"的转换器工作量
3. 确认 LLMServingSim License；联系 scitix 询问 ROSS 授权（如需）
4. 用 aiconfigurator `cli default` 跑同模型基线，与 Frontier 结果交叉对比
5. 验证 Vidur PR #80 是否有进展（作为 PDD 备胎的可行性复查）

---

## 9. dev_plan.md 勘误（重要）

| # | dev_plan 原文 | 核实结果 | 证据 |
|---|---|---|---|
| 1 | "LLMServingSim 2.0 异构 arxiv 2605.21312" | **arXiv:2605.21312 是 Frontier v2 论文**（作者 Feng et al., CUHK）；LLMServingSim 2.0 的正式出处是 **ISPASS 2026, DOI 10.1109/ISPASS69572.2026.00012** | Frontier 仓库 Citation 节 + arxiv 页面 |
| 2 | "DynoSim | NVIDIA | ✅ Dynamo 栈" | **未找到任何公开存在的证据**（GitHub/arXiv/Web 三层检索均无）；最接近的真实项目是 Dynamo 的 AI Simulate（实验性重放搜索框架，非分析模拟器） | GitHub Search API + arXiv API（totalResults=0）+ NVIDIA docs |
| 3 | "Vidur 报告 ~700% P50 偏差（vs vLLM 0.9.1）" | **未找到该数字**。可查证的第三方数据（Frontier 论文 §5，H800）：Vidur attention p50/p95 = 55.4%/376.1%，E2E 2.9–45.5% | arXiv:2605.21312 html 全文 |
| 4 | "Frontier 原生 PDD+AFD"（表格中"部分"） | 现已属实：PDD 2026/06 发布、AFD 2026/08 发布；且异构已用 H800+H20 完整验证（dev_plan 写"部分"已过时/保守） | Frontier README Latest News + 论文 §6.2 |
| 5 | "Vidur 最成熟" | 需修正：Vidur 主分支已停滞 >1 年、PD 分离 PR 未合、prefix cache 只在 canary 分支；"最成熟接口"成立，"最成熟项目"不成立 | GitHub API pushed_at=2025-07-25 + PR #80 open |
| 6 | "ROSS XGBoost 预测（github.com/scitix/ross）" | 属实，但遗漏关键风险：**无 License 文件**、仅 2 stars | 本地 clone 验证 |
| 7 | 风险表未提 aiconfigurator 精度 | 第三方实测其 E2E makespan 误差最高 170–200%（Frontier 论文），"接入难度低（黑盒）"的同时精度也是黑盒 | arXiv:2605.21312 §5 |

---

## 10. 证据索引

| 证据 | 来源 |
|---|---|
| 各仓库 stars/push/license | GitHub REST API（2026-08-18 实测） |
| Vidur <9% 误差自报 | arXiv:2405.05465 abstract（arXiv API 原文提取） |
| Vidur PD 分离未合 | github.com/microsoft/vidur/pull/80（open） |
| Frontier 全部能力与精度 | github.com/NetX-lab/Frontier README + arXiv:2605.21312v2 html 全文 |
| ROSS 架构/无 License | 本地 clone（/tmp/pi-github-repos/scitix/ross）README + 目录结构 |
| DistServe/simdistserve | 本地 clone（/tmp/pi-github-repos/LLMServe/DistServe）+ arXiv:2401.09670 |
| LLMServingSim 2.0 论文 | 仓库 Citation 节（ISPASS'26 DOI 10.1109/ISPASS69572.2026.00012） |
| aiconfigurator 能力 | 仓库 README（2026-08-18 抓取） |
| DynoSim 不存在 | GitHub Search API + arXiv API（totalResults=0）+ source_check（missing-evidence, confidence 0.20） |
| Dynamo AI Simulate | docs.nvidia.com/dynamo ai-simulate sweeper overview（fetch 实测） |
