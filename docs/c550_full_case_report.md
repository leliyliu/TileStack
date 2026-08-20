# 沐曦 C550 接入 TileSight 与 RMSNorm 优化：完整案例记录

> **一份文档讲完整件事**：从硬件规格调研、TileSight 架构建模、算子级验证，到发现 sglang 镜像的性能空间、实施 kernel 优化并集成进新镜像的全过程。
>
> 周期：2026-08-18 ~ 2026-08-20，全部结论附实测数据与可复现路径。
> 细分文档索引见附录 A；本文为整合版汇报材料。

---

## 0. 一页总结（TL;DR）

| 阶段 | 做了什么 | 结果 |
|---|---|---|
| ① 硬件调研 + 实测 | 规格文档 + muxi-01 微基准 | FP16 GEMM 实测 275 TFLOPS、HBM 1430 GB/s，澄清第三方数据矛盾 |
| ② TileSight 接入 | 新增 `MXC550` Arch profile（实测口径默认） | 8192³ GEMM 建模误差 **4.6%** |
| ③ 算子级验证 | FA 三段拆解 / RMSNorm / 融合链 + 修复 3 处建模层 NVIDIA 硬编码 | FA 6 点误差 0.5–10.4%；RMSNorm 0.3–5.7% |
| ④ 镜像对齐调查 | 追查 sglang 沐曦镜像的真实分发路径 | 发现 flashinfer kernel 只发挥 **67% 硬件带宽** |
| ⑤ 优化 + 镜像集成 | Triton kernel（按 64 线程 wavefront 调优）+ .pth 钩子集成 | kernel 级最高 **1.43x**；decode 场景 **+6.3%**；新镜像已交付 |
| ⑥ 流程标准化 | 沉淀 6 阶段接入流程 + 算子级验证规范 | 国产卡接入约 **2 天/卡** 可复制 |

**方法论核心**：TileSight 给出"现状"与"硬件极限"两条线（67% vs 90%），把
"感觉 kernel 慢"变成"26% 的量化空间"，并排除硬件背锅——建模先行，优化有据。

---

## 1. 背景与环境

### 1.1 任务定义

TileSight 是 tile-centric 的第一性原理 GPU 性能建模工具（预测 kernel 延迟/瓶颈，
无需运行 kernel）。本案例的目标链：

```
读硬件规格 → 实测校准 → 接入 TileSight → 验证算子级精度（FA/RMSNorm/融合）
→ 与 sglang 沐曦镜像对齐 → 发现优化空间 → 实施优化 → 集成新镜像 → A/B 评估
```

### 1.2 环境信息

| 项 | 值 |
|---|---|
| 服务器 | muxi-01（免密 SSH，CX7Group1-host-050），8 × C550 全部可用 |
| 驱动/软件栈 | Kernel 3.3.12；宿主 MACA 3.2.1；容器 MACA 3.7.1.17 |
| 镜像 | `sglang:v0.5.12-deepseek-v4-rc1-maca.ai.3.7.1.110`（沐曦官方 sglang 镜像） |
| 容器 | dsv4-d-50：Python 3.10 + torch 2.8.0+metax + flashinfer/sgl_kernel/mcoplib 沐曦编译版 + triton/tilelang/flash_attn |
| 关键事实 | metax torch 走 cu-bridge，设备名是 **`cuda`**（不是 `musa`），`is_cuda()=True` |

---

## 2. C550 硬件规格（调研 + 实测校准）

> 完整规格调研见 `npus/07-C550硬件规格.md`（含可信度标注体系：✅官方/🔬实测/⚠️第三方/❌未公布）。此处只列建模直接依赖的结论。

### 2.1 架构关键参数

| 参数 | 数值 | 来源 | 建模含义 |
|---|---|---|---|
| 计算单元 | 104 AP × 4 PEU × 16 lane | 🔬 macainfo | AP ↔ NVIDIA SM，`sm_count=104` |
| 调度粒度 | **64 线程 wavefront**（NVIDIA 是 32） | 🔬 macainfo | wave 量化、寄存器分配粒度 |
| 每 AP 驻留 | 32 wave / 2048 线程 | 🔬 macainfo | occupancy 上限 |
| 发射结构 | 每 AP **单发射单元**、5 发射口（1标量+1 MMA/向量ALU 互斥+1向量访存+1共享内存+1杂项） | ✅ 官方 | `warp_schedulers_per_sm=1` |
| 共享内存 WSM | 64 KB/AP，128 B/cycle（读写全双工），32 bank×4B，独立 SRAM（非 L1 划扣） | ✅ 官方 | smem 带宽 = 104×1.6G×128 |
| 向量寄存器堆 | 512 KB/AP | ✅ 官方 | `register_capacity_per_sm` |
| L2 | 8 MB 全片共享，128B cacheline，实测 4096 B/cycle | 🔬 实测 | L2 带宽 6553 GB/s |
| 显存 | HBM2e 64GB，4096bit @1800MHz → 理论 1843 GB/s | 🔬 推算 | DDR 口径基准 |
| 矩阵单元 | WMMA 16×16×16（`MMA_16x16x16F16`），MMA 与向量 ALU 共发射槽 | ✅ 官方 | `get_tensor_core_minimum_ptx` |
| FP8 | 不支持（C600 特性） | ✅ 官方 | bytes=0.5 应 raise |
| 最高时钟 | 1600 MHz（mx-smi 时钟读数负载下不更新，不可用于校准） | 🔬 macainfo | `max_freq` |

### 2.2 微基准实测（2026-08-18，profiling 脚本见附录 B）

| 指标 | 实测值 | 说明 |
|---|---|---|
| GEMM FP16 峰值 | **275.03 TFLOPS**（8192³） | 印证 C500 OAM"280T"档；否证 CSDN"240T"（疑混淆 PCIe 档） |
| GEMM BF16 / FP32 | 269 / 124.5 TFLOPS | FP32 远超向量理论值 21T → mcBLAS 走矩阵单元（TF32 类路径），规格文档未记载 |
| GEMM 持续（4000 次连续） | 230 TFLOPS | 峰值的 84%，散热稳定 |
| HBM copy / add | 1429 / 1480 GB/s | 理论的 78–80%（torch 实现，优于手写 STREAM 的 762/1154） |

---

## 3. TileSight 接入（Arch profile）

### 3.1 实现方式

新增 `src/tilesight/tilesight/arch/metax_c550.py`（类 `MXC550`），仿 MI300X 先例
（同为 64 线程 wavefront/单发射/独立 SRAM）。**默认实测口径**，`set_to_spec()` 切换
理论口径（FP16 280T / DDR 1843 GB/s）。每个字段标注数据来源。

关键字段取值（推导依据）：

| 字段 | 值 | 依据 |
|---|---|---|
| `fp16_tensor_flops` | 275e12 | 🔬 实测峰值 |
| `fp32_tensor_flops` | 124.5e12 | 🔬 实测（mcBLAS 矩阵路径） |
| `int8_tensor_flops` | 560e12 | ⚠️ C500 OAM 参考档，未实测 |
| `ddr_bandwidth` | 1430e9 | 🔬 实测 copy（保守） |
| `l2_bandwidth` | 6553.6e9 | 🔬 推算 4096 B/cycle × 1.6GHz |
| `smem_bandwidth` | 21.3 TB/s | 104 × 1.6GHz × 128 B |
| `wavefront_size` | 64 | 🔬 macainfo（新增字段） |
| `warp_schedulers_per_sm` | 1 | ✅ 单发射单元 |
| `get_tensor_core_minimum_ptx(2/4/1/0.5)` | (16,16,16)/(16,16,8)/(16,16,32)/raise | ✅ WMMA 形状；FP8 不支持 |

L1.5/TMEM 层次不存在（VL1 默认关闭且 32KB 过小），保持基类关闭。

### 3.2 GEMM 端到端验证

8192³ FP16 GEMM（wmma，tb=(128,128,32)，wp=(64,64,16)，stage=3）：

| 口径 | 建模 | 实测对照 | 误差 |
|---|---|---|---|
| 实测口径（默认） | 4.55 ms | 持续 4.77 ms | **4.6%**（论文系统级 MAPE 12.35%） |
| spec 口径 | 4.01 ms | 峰值 4.00 ms | 恰好对齐满利用率理论值 |

### 3.3 建模层语义修复（对所有架构生效）

为让建模层真正理解 C550 的 GPU 执行语义，修复三处 NVIDIA 硬编码：

| 修复 | 位置 | 影响 |
|---|---|---|
| 寄存器占用硬编码 32 线程/warp | `occupancy.py` | C550 此前低估 2 倍 |
| 线程开销量化档 32/128/256/384 | `elementwise/reduce_pipeline_wave.py` | 改为随 `arch.wavefront_size`（64→64/256/512/768） |
| **batch GEMM 的 batch 维不参与 wave 全 SM 调度** | `matmul_pipeline_wave.py`（batch 折入 total_tiles） | 小 grid×大 batch 高估最高 5x |

---

## 4. 算子级验证（FA / RMSNorm / 融合）

### 4.1 实测基准（bench/results_c550_ops.json）

- FA prefill（B=8,H=32,D=128,bf16）：full 3.41/13.74/57.24 ms（S=2048/4096/8192），causal 恰为一半
- FA decode（q_len=1）：0.19–0.69 ms
- RMSNorm：原生 0.731 ms vs compiled 0.073 ms（**原生 kernel 低效 15x**）
- residual+RMSNorm：compiled 串行 0.142 ms vs 融合 0.107 ms（公平基线 1.33x）
- 4096³ bf16 matmul 0.511 ms = 270 TFLOPS

### 4.2 建模与对比

**FlashAttention 三段拆解**（QK^T → softmax → PV 依赖链；段1/3 复用 matmul pipeline
wave，PV 的 scores 驻留 WSM 不计 DDR 读；段2 按 smem+SFU）：

| 场景 | 误差 |
|---|---|
| full S=2048/4096/8192 | 2.2% / 0.5% / 3.7% |
| causal S=2048/4096/8192 | 7.9% / 10.4% / 6.2% |

**RMSNorm**（单遍 WSM 缓冲，IO=读N份+写1份）：4 shape 误差 0.3–5.7%。

**融合收益**：建模 1.39x vs 实测 1.33x（4.6%）；MatMul→RMSNorm 链 <15%。

### 4.3 新增 per-arch 校准常数（多测点交叉验证）

| 常数 | 值 | 含义 | 标定方法 |
|---|---|---|---|
| `elementwise_ddr_eff` | 0.65 | 含行规约 kernel 有效带宽（0.93 TB/s / 1430） | RMSNorm 4 测点 |
| `fa_tc_eff` | 0.75 | FA 有效 TC 算力系数 | **两端口径夹逼**：eff=1 低估 20-30%、eff=0.56 高估 20-32%，各测点线性交点 0.735–0.78 |

### 4.4 重要实测发现

1. 原生 `torch.nn.RMSNorm` 在沐曦上仅 95 GB/s（7% 带宽）——模型上限可暴露实现低效
2. 纯 elementwise（add，1.35 TB/s）与含规约 kernel（0.93 TB/s）带宽效率不同，建模需区分
3. FA 有效算力 6 测点一致 150–161 TFLOPS（GEMM 峰值的 56%）

---

## 5. sglang 镜像对齐调查（发现优化空间）

### 5.1 真实分发路径（两个工程发现）

**发现一：`forward_musa` 是死代码。** metax torch 的 cu-bridge 让 `is_cuda()=True`，
sglang RMSNorm 实际走 `forward_cuda`：

```
RMSNorm.forward_cuda
├─ residual is None → flashinfer.rmsnorm          (q/k norm 等)
└─ residual 存在   → sgl_kernel.fused_add_rmsnorm (decoder layer 主路径, 4参原地)
```

**发现二：.py 与 .pyc 不一致。** 磁盘 `layernorm.py` 是上游版（引用不存在的
`rms_norm`，删 pyc 强制重编译会 NameError），运行时 `.pyc` 是沐曦适配版。
→ 集成方案必须绕开源码修改（见 §6.3）。

### 5.2 三方对比（4096×4096 bf16）

| 实现 | 延迟 ms | 有效带宽 | 占理论带宽 |
|---|---|---|---|
| torch 原生 rms_norm | 0.7027 | 95 GB/s | 7% |
| **镜像实际：flashinfer** | **0.0777** | **864 GB/s** | **67%** |
| torch.compile | 0.0664 | 1010 GB/s | 78% |
| （优化后 Triton） | **0.0559** | **1200 GB/s** | **93%** |
| 理论（1430×0.9） | 0.0521 | 1287 GB/s | 100% |

fused 版（4 份流量）：sgl_kernel 0.1063 ms vs Triton 0.0989 ms。

### 5.3 模型如何定位"这是软件问题"

TileSight RMSNorm 模型给出两个口径：

| 口径 | 延迟（4096×4096） | 带宽 |
|---|---|---|
| 硬件能力口径（ddr×0.9） | 52.1 µs | 1287 GB/s |
| 现状口径（elementwise_ddr_eff=0.65） | 72.2 µs | 930 GB/s |
| 镜像 flashinfer 实测 | 77.5 µs | 864 GB/s |

三个读数：① 模型与现状对齐（72.2 vs 77.5 µs，误差 6.8%）；② flashinfer 距硬件能力
有 **26% 带宽空间**——RMSNorm 纯 memory-bound、无计算瓶颈背锅，差距只能来自访存组织；
③ 优化后 55.9 µs 落在两口径之间（93%），证明硬件能力口径可达、不乐观。

> **方法论核心**：TileSight 的价值不是预测一个数，而是给出"现状"与"极限"两条线，
> 让优化空间可量化，让"值不值得优化"有依据。

---

## 6. RMSNorm 优化与镜像集成

### 6.1 Kernel 设计（针对 C550 微架构）

```python
@triton.jit
def _rmsnorm_kernel(X, W, Y, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)              # ① 每行一个 program
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row*stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N     # ② fp32 累加
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    y = (x * rstd * w).to(Y.dtype.element_ty)
    tl.store(Y + row*stride + cols, y, mask=mask)
# launch: BLOCK = next_pow2(hidden); num_warps = clamp(BLOCK//512, 1, 16)
```

三个设计点：
1. **BLOCK=next_pow2(hidden)、掩码最小化**：2 幂 hidden 整行单次 load，合并访问充分
2. **num_warps=BLOCK/512**（4096→8 warp=512 线程）：按 C550 的 64 线程 wavefront、
   每 AP 32 wave/2048 线程的驻留约束推导——这正是"按具体架构建模"的直接产物
3. **fp32 累加 + bf16 存储**：与 flashinfer 数值语义一致（allclose 全过）

**适用边界**：非 2 幂 hidden（如 5120）Triton 带掩码反而更慢（0.1098 vs 0.0912 ms），
**必须保留 flashinfer 回退**（dispatch: 2 幂+半精度+hidden≥64 才走 Triton）。

### 6.2 集成方案：.pth 启动钩子

绕开 .py/.pyc 不一致：不改 sglang 任何源文件，site-packages 放 `.pth`
（解释器启动执行）→ import hook 在 `layernorm` 模块加载后替换其模块级
`rmsnorm`/`fused_add_rmsnorm`（forward_cuda 按全局名调用，与运行时 pyc 语义一致）。

工程细节（实测踩坑）：
- **triton.jit 必须真实源文件**：exec 字符串定义会 `OSError: could not get source
  code`（第一版静默失败的根因）→ kernel 独立为 `metax_rmsnorm_kernels.py`
- 懒加载 triton；补丁任何异常只告警、不破坏宿主进程
- **A/B 开关** `METAX_RMSNORM_PATCH=0`：评估与回滚零成本

### 6.3 A/B 实测结果（新镜像内）

**类级**（经 sglang RMSNorm 类端到端，含数值校验）：

| shape | 无 residual | 带 residual | 数值 |
|---|---|---|---|
| 4096×4096 | 0.0814→0.0568 ms（**1.43x**） | 0.1072→0.0985（1.09x） | ✓ |
| 4096×8192 | 0.1472→0.1135（1.30x） | 0.2385→0.1909（1.25x） | ✓ |
| 16384×8192 | 0.5411→0.4283（1.26x） | 0.9108→0.7251（1.26x） | ✓ |
| 4096×5120（回退） | 1.00x | 1.00x | ✓ |

**复合 transformer 层**（3 matmul + 2 fused norm，H=4096）：

| 场景 | 基线 ms | 补丁 ms | 收益 |
|---|---|---|---|
| prefill 4096 rows | 1.7451 | 1.7248 | +1.2% |
| **decode 8 rows** | 0.1248 | 0.1174 | **+6.3%** |
| 大 batch 16384 rows | 7.299 | 7.164 | +1.9% |

解读（与建模自洽）：H=4096 层内 norm 占比 ~12%，kernel 级 1.3x 收益在模型级稀释为
1–2%；**decode 小 batch 时 GEMM 退出饱和区、norm 占比升高，收益 6.3%——恰是
在线推理延迟最敏感的场景**。

### 6.4 镜像交付

| 项 | 状态 |
|---|---|
| 新镜像 `metax-sglang:rmsnorm-opt-v0.5.12-maca3.7.1` | ✅ docker commit（sha256:ed10775...，19.1GB） |
| 新容器验证 | ✅ 补丁自动生效 + GPU 可用 + 数值正确（需 `--privileged --network host --ipc host`，与原容器同规格） |
| 原 dsv4-d-50 | ✅ 已清理补丁文件、停回原状（对其他使用者零影响） |

---

## 7. 国产卡标准化接入流程（沉淀）

> 完整版含 checklist：`docs/domestic_gpu_integration_guide.md`。适用摩尔线程、寒武纪、海光 DCU、燧原等，约 **2 天/卡**。

```
阶段0 规格调研 → 阶段1 环境勘察 → 阶段2 微基准实测 → 阶段3 参数推导
→ 阶段4 TDD 实现注册 → 阶段5 验证归档(误差≤15%) → 阶段5.5 算子级验证
```

关键规范：
- **8 个必答问题**：计算单元层次/wavefront 大小/发射结构/共享内存结构/寄存器堆/
  L2 层次/MMA 形状/各精度算力
- **最小基准集**：STREAM 带宽、多档 GEMM（fp16/fp32/持续）、L2/SMEM 可选
- **两个校准常数的标定法**：`elementwise_ddr_eff`（多 shape 带宽反推）、
  `fa_tc_eff`（两端口径夹逼，禁止单点拟合）
- **5 项非 NVIDIA 语义检查**（本案例全部踩过）：寄存器 32 线程硬编码/线程开销
  量化档/batch wave 调度/WSM 驻留张量误计 DDR/原生算子实现低效
- **ground truth 纪律**：必须用 compiled/良实现做基线（原生 torch 在沐曦上可能
  低效 15x，否则把实现问题骗进模型误差）

---

## 8. 汇报要点与诚实局限

### 三句话总结

1. **建模先行，量化空间**：TileSight 用实测校准的 profile 给出 RMSNorm
   "现状 67% / 极限 90%"两条线，把优化判断从直觉变成 26% 的量化空间
2. **按微架构调优**：64 线程 wavefront、驻留约束是 C550 特有的，kernel launch
   参数由架构参数推导；明确适用边界（非 2 幂回退），不引入回归
3. **工程闭环可复制**：.pth 钩子集成、环境变量 A/B、三层评估（生效/类级/模型级）、
   docker commit 新镜像——runbook 已沉淀，下一个算子按同样流程推进

### 诚实的局限

- 模型级收益 1–2%（decode 6.3%）——单 memory-bound 小算子的宿命；更大收益需
  攻层内占比更高的算子（GEMM epilogue 融合、attention），TileSight 9 维瓶颈
  分解可直接定位下一个目标
- `num_warps` 是架构推导 + 经验验证，未做全空间 autotune
- fused 版收益（1.09–1.26x）小于无 residual 版——sgl_kernel 沐曦原生实现在该路径已较强
- 遗留：FA decode 建模、MMA/向量 ALU 共槽显式建模（现由 `fa_tc_eff` 吸收）、
  新版镜像（sglang 0.5.13/maca 3.8.1）上的适配回归

---

## 附录 A：细分文档与代码索引

| 内容 | 位置 |
|---|---|
| 硬件规格调研（完整版） | `npus/07-C550硬件规格.md` |
| 接入全过程记录（含逐字段推导表） | `docs/metax_c550_integration.md` |
| 国产卡标准化接入流程（含 checklist） | `docs/domestic_gpu_integration_guide.md` |
| RMSNorm 优化案例（单行本） | `docs/case_study_rmsnorm_optimization.md` |
| Arch profile | `src/tilesight/tilesight/arch/metax_c550.py` |
| FA 三段模型 / RMSNorm 模型 | `src/tilesight/tilesight/fused_op_pipeline_wave/{flash_attention,rmsnorm}_pipeline_wave.py` |
| 测试（36 用例） | `tests/test_metax_c550.py` |
| GEMM/算子实测数据 | `bench/results_c550_ops.json` |
| sglang 对齐对比数据 | `bench/results_c550_sglang_align.json` |
| Triton kernel + 集成补丁 + 评估脚本 | `bench/metax_rmsnorm_{kernels,patch}.py`、`bench/eval_metax_rmsnorm_patch.py` |

## 附录 B：关键实测脚本复现

```bash
# 微基准 profiling (GEMM/带宽) —— 容器内 /opt/conda/bin/python, 设备名 "cuda"
scp bench/metax_c550_ops_bench.py muxi-01:/tmp/
ssh muxi-01 'docker cp /tmp/metax_c550_ops_bench.py <ctr>:/tmp/ && docker exec <ctr> /opt/conda/bin/python /tmp/metax_c550_ops_bench.py'

# 补丁 A/B 评估 (三层: 生效/类级/复合层)
docker exec <ctr> bash -c "/opt/conda/bin/python /tmp/eval_metax_rmsnorm_patch.py > on.json 2>/dev/null; \
  METAX_RMSNORM_PATCH=0 /opt/conda/bin/python /tmp/eval_metax_rmsnorm_patch.py > off.json 2>/dev/null"
# 注: 输出含沐曦启动 banner, 解析时跳到首个 '{'

# TileSight 建模测试 (注意 PYTHONPATH)
uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e . numpy scipy networkx pandas matplotlib plotly pytest
PYTHONPATH=src/tilesight python -m pytest tests/test_metax_c550.py -v
```
