# 国产 GPU 标准化接入 TileSight 流程

> 基于沐曦 C550 接入实践（`docs/metax_c550_integration.md`，2026-08-18）沉淀的可复制流程。
> 适用对象：沐曦（其余型号）、摩尔线程、寒武纪、海光 DCU、燧原等国产加速卡。
> 全程估时：规格调研 0.5 天 + 实测 0.5 天 + 实现 0.5 天 + 验证 0.5 天 ≈ **2 天/卡**。

## 流程总览

```
阶段 0        阶段 1         阶段 2        阶段 3        阶段 4        阶段 5
规格调研  →  环境勘察   →  微基准实测  →  参数推导   →  实现注册  →  验证归档
(npus/*.md)  (SSH/容器)    (profiling)   (逐字段)      (TDD)        (误差<15%)
```

每个阶段有明确的**准入/完成条件**，不满足不进入下一阶段。

---

## 阶段 0：规格调研（产出 `npus/XX-<型号>规格.md`）

| 步骤 | 内容 | 来源优先级 |
|---|---|---|
| 0.1 | 官网产品页：形态/显存/功耗/互连 | 官方 > 权威媒体 > 社区 |
| 0.2 | 官方编程/调优文档：线程模型、存储层次、矩阵指令形状 | 开发者文档站 |
| 0.3 | 同代同架构型号参照（如 C550↔C500） | 仅作交叉验证 |
| 0.4 | 第三方数据全部标注 ⚠️，矛盾点显式列出 | 不采信单一来源 |

**必须澄清的 8 个问题**（建模直接依赖）：

1. 计算单元数量与层次（AP/CU/SM ↔ TileSight `sm_count`）
2. 线程调度粒度（wavefront/warp 大小，每单元驻留 wave 数）
3. 发射结构（几发射单元、几发射口 → `warp_schedulers_per_sm`）
4. 共享内存容量/带宽/bank 结构（独立 SRAM 还是 L1 划扣）
5. 寄存器堆容量、分配粒度
6. L2 容量/带宽/cacheline；有无 L1.5/TMEM 等中间层
7. 矩阵单元最小 MMA 形状（各 dtype）→ `get_tensor_core_minimum_ptx`
8. 各精度算力口径（官方值优先，无则实测）

**完成条件**：每个问题有答案或显式标注 ❌ 未公布（未公布项全部进入实测计划）。

## 阶段 1：服务器环境勘察

```bash
# 1. 连通性与硬件在位
ssh <host> 'hostname; <厂商>-smi | head -30'
# 2. 软件栈版本（宿主 + 容器往往不同！）
ssh <host> 'ls /opt | grep -iE "maca|musa|..."'
# 3. 找到带 torch 的运行环境（通常在容器 /opt/conda）
docker exec <ctr> bash -lc 'python -c "import torch; print(torch.__version__, torch.cuda.is_available())"'
# 4. 设备属性查询工具（macainfo / rocminfo / mthreads-gmi ...）
```

**常见坑（C550 实测踩过）**：

- 厂商兼容层设备名可能是 `cuda`（cu-bridge）而非自研名（如 `musa`）——先探测再写脚本
- 容器内默认 PATH 的 python 可能无 torch；找 conda 环境
- 厂商 smi 工具的时钟读数可能不随负载更新 → 频率以设备属性 API 的 Max Clock 为准

**完成条件**：能跑通一个 `torch.mm` 并计时。

## 阶段 2：微基准实测（默认口径的数据来源）

最小基准集（torch 实现，约 100 行，模板见 `metax_c550_integration.md` 附录）：

| # | 基准 | 方法 | 产出字段 |
|---|---|---|---|
| 1 | HBM 带宽 | STREAM copy/add，≥1 GiB，读+写总流量口径 | `ddr_bandwidth` |
| 2 | GEMM 峰值 | 主精度（fp16/bf16）2048–8192³ 多档取最佳，≥20 次平均 | `fp16_tensor_flops` |
| 3 | GEMM FP32 | 同上（判断是否走矩阵单元：远超向量理论值即是） | `fp32_tensor_flops` |
| 4 | 持续算力 | 数千次连续 GEMM，验证散热/降频 | 校验峰值可信度 |
| 5 | （可选）L2 带宽 | K 小 M/N 大的 resident GEMM | `l2_bandwidth` |
| 6 | （可选）SMEM 带宽 | 块内规约类 kernel | `smem_bandwidth` |

**纪律**：每个数字记录日期/环境/迭代次数；实测值与第三方数据矛盾时**以实测为准**并在文档记录矛盾。

**完成条件**：带宽 + 主精度 GEMM 至少两项有可信实测值。

## 阶段 3：参数推导（逐字段过一遍消费方）

建模层（`fused_op_pipeline_wave`）消费的 arch 字段清单（`grep -roh "arch\.[a-z_0-9]*"` 可复核）：

```
必填（assert 或直接使用）:
  core, sm_count, ddr_bandwidth, ddr_wave_bytes, l2_bandwidth, l2_capacity,
  smem_bandwidth, fp16_tensor_flops, fp32_cuda_core_flops,
  configurable_smem_capacity, register_capacity_per_sm,
  get_tensor_core_minimum_ptx(), *_max_util
可选（hasattr 保护）:
  fp8_tensor_flops, int8_tensor_flops, sfu_flops, fp32_tensor_flops
基类默认关闭:
  l1_5_*, tmem_*（无该层次就不要开）
```

推导规则：

- 能实测的用实测（默认口径），理论值进 `set_to_spec()`
- 带宽类 = 实测 B/cycle × 实测最高频率（推算需标注）
- 发射单元数：单发射器架构（AMD/沐曦）填 1，多调度器（NVIDIA 4）填实际值
- MMA 最小形状：按官方 WMMA/MAPI 文档填；**不支持的精度要 raise 而非猜测**
- 无依据的估计值（如 fp64）必须在注释中标注 ⚠️ 估计 + 依据

## 阶段 4：实现与注册（TDD）

1. **先写测试** `tests/test_<arch>.py`，五组用例：
   - 字段存在性（必填清单全部 > 0；`get_tensor_core_minimum_ptx` 存在）
   - 实测口径默认值（与阶段 2 数据一致）
   - `set_to_spec()` 理论口径
   - 架构参数（sm_count / smem / regfile / wavefront）
   - MMA 形状映射（每个支持的 dtype + 不支持的 dtype raise）
   - 端到端冒烟（见阶段 5）
2. **实现** `src/tilesight/tilesight/arch/<vendor>_<model>.py`，仿照先例：
   - AMD 类（64 线程 wave / 单发射 / 独立 SRAM）→ 仿 `mi300x.py` + `metax_c550.py`
   - NVIDIA 类 → 仿 `h100_sxm.py`（注意 `ddr_wave_bytes` 等）
3. **注册** `arch/__init__.py` 追加 import
4. 运行：`PYTHONPATH=src/tilesight python -m pytest tests/test_<arch>.py -v`
   （注意 `src/tilesight` 必须在 PYTHONPATH，原因见接入记录第 5 节）

## 阶段 5：验证与归档

**端到端验证标准**：用实测同款 GEMM 配置建模，与实测持续值对比：

$$\text{误差} = \frac{|\text{建模延迟} - \text{实测持续延迟}|}{\text{实测持续延迟}} \le 15\%$$

（论文系统级 MAPE 为 12.35%，单算子 GEMM 应优于它；C550 实测口径达 4.6%）

同时检查 sanity：`compute_util` 与实测达成率（持续/峰值）同量级；spec 口径延迟应≈峰值延迟。

**归档产出**（三件套）：

1. `npus/XX-<型号>规格.md` —— 规格调研（阶段 0）
2. `docs/<vendor>_<model>_integration.md` —— 接入全记录：实测数据、逐字段推导表、
   验证结果、踩坑、遗留事项（模板 = `metax_c550_integration.md`）
3. `tests/test_<arch>.py` + `arch/<vendor>_<model>.py` —— 代码与测试

**遗留事项必须显式列出**：未实测字段、走通用分支的 op 层硬编码、未表达的架构特性。

---

## 已知限制与进阶路线

| 层级 | 现状 | 进阶动作 |
|---|---|---|
| Arch 层（本流程） | ✅ 标准化，2 天/卡 | — |
| op 层 | 38 处 NVIDIA 硬编码分支（`expansion_feasibility.md`），国产卡走通用 else 分支 | 每卡 1–2 周：为 L2 结构/MMA 形态特殊的卡加专用分支 |
| 微基准覆盖 | 带宽 + GEMM | 补 SMEM/寄存器 bank 冲突、小 tile、occupancy 曲线 |
| Profiler 对比 | 无（NVIDIA 有 compare_with_ncu） | 厂商 profiler（mcTracer/rocprof）接入对照 |

## 检查清单（Checklist，可复制）

```
阶段0 □ 官网/文档调研完成          □ 8 个关键问题已回答或标 ❌
阶段1 □ SSH 免密可用               □ 厂商 smi 确认卡在位
      □ 容器内 torch 可用          □ 设备名探测完成（cuda/musa/...）
阶段2 □ HBM copy/add 实测          □ 主精度 GEMM 多档实测
      □ FP32 GEMM 实测             □ 持续算力实测
      □ 数据与第三方矛盾已裁决
阶段3 □ 必填字段清单逐项有值        □ 每值有来源标注
      □ 不支持精度 → raise
阶段4 □ 测试先行且通过             □ __init__.py 已注册
      □ PYTHONPATH 运行方式注明
阶段5 □ 端到端误差 ≤15%            □ 三件套文档归档
      □ 遗留事项列出
```
