# 案例报告：TileSight 引导的沐曦 C550 RMSNorm 优化

> 从性能建模到 kernel 优化再到镜像集成的完整闭环（2026-08-18 ~ 08-20）。
> 本文档面向组内汇报，所有数据均为实测，可复现命令与原始数据见文中引用。

---

## 0. 一页总结（TL;DR）

| 问题 | sglang 沐曦镜像中的 RMSNorm kernel 只发挥了硬件带宽的 **67%** |
|---|---|
| 怎么发现的 | TileSight 建模给出"该算子理论可达 93% 带宽"，与镜像现状 67% 之间出现 **26% 的 gap**——模型证明了"是软件问题，不是硬件极限" |
| 怎么优化的 | 针对 C550 的 64 线程 wavefront 写 Triton kernel（每行一个 program、BLOCK=next_pow2、num_warps=BLOCK/512） |
| 最终效果 | kernel 级 **最高 1.43x**（0.0814→0.0568 ms）；复合层 **decode 场景 +6.3%**；已提交新镜像 `metax-sglang:rmsnorm-opt-v0.5.12-maca3.7.1` |
| 方法论价值 | 这套流程（建模找空间 → 实测定基线 → kernel 优化 → 镜像集成 → A/B 评估）是**可复制的**，标准化文档已沉淀 |

---

## 1. 背景

### 1.1 硬件与软件环境

| 项 | 值 |
|---|---|
| GPU | 沐曦曦云 C550（MXC550），8 卡服务器 muxi-01 |
| 架构要点 | 104 AP、**64 线程 wavefront**（非 NVIDIA 32）、单发射单元 5 发射口、WSM 64KB/AP 独立 SRAM、HBM2e 64GB |
| 镜像 | `sglang:v0.5.12-deepseek-v4-rc1-maca.ai.3.7.1.110`（沐曦官方 sglang 镜像，含 flashinfer/sgl_kernel 沐曦编译版） |
| 实测环境 | 容器 dsv4-d-50，torch 2.8.0+metax，设备名走 cu-bridge 报告为 `cuda` |

### 1.2 任务链（前序工作）

本轮优化建立在此前已完成的 TileSight C550 接入之上（`docs/metax_c550_integration.md` §1-8）：
用实测微基准校准了 `MXC550` Arch profile（FP16 GEMM 275T、HBM 1430 GB/s），并验证了
GEMM/FlashAttention/RMSNorm/融合链的建模精度（误差 0.5%–11.5%）。

---

## 2. 现有镜像中的实现是什么（调研发现）

### 2.1 sglang 在沐曦上的真实分发路径

第一个发现就推翻了直觉：**sglang 的 `forward_musa` 是死代码**。
沐曦 torch 的 cu-bridge 兼容层让 `is_cuda()` 返回 True，于是 RMSNorm 类实际分发到
`forward_cuda`（`sglang/srt/layers/layernorm.py`），落到两条沐曦编译的 kernel：

```
RMSNorm.forward_cuda
├─ residual is None → flashinfer.rmsnorm          (无残差: q/k norm 等)
└─ residual 存在   → sgl_kernel.fused_add_rmsnorm (带残差: decoder layer 主路径, 4 参原地)
```

> 验证方式：实例化 `RMSNorm` 检查 `_forward_method` 为 `forward_cuda`；
> 逐实现基准计时与路径逐级对质（`bench/results_c550_sglang_align.json`）。

### 2.2 一个工程陷阱：.py 与 .pyc 不一致

调研中发现磁盘上的 `layernorm.py` 引用了模块中**不存在的** `rms_norm` 名字——
磁盘源码是上游版本，而运行时加载的 `.pyc` 是沐曦适配版（pyc mtime 晚于 .py）。
直接改源码并删 pyc 强制重编译会立刻 NameError。这个发现直接决定了后面的
**集成方案**（见 §4.3：用 .pth 启动钩子，不改 sglang 任何文件）。

### 2.3 现状基线（4096×4096 bf16，实测）

| 实现 | 延迟 ms | 有效带宽 | 占理论带宽 |
|---|---|---|---|
| torch 原生 `rms_norm` | 0.7027 | 95 GB/s | 7%（严重低效） |
| **镜像实际：flashinfer** | **0.0777** | **864 GB/s** | **67%** |
| torch.compile | 0.0664 | 1010 GB/s | 78% |

RMSNorm 是 memory-bound 算子，IO = 读 1 份 + 写 1 份（fused 版 4 份）。
理论下限 = 流量 / 显存带宽。**镜像里的"官方优化 kernel"距离硬件能力还有约 1/3 的空间**——
但当时缺少一个可信的"硬件到底能到多少"的口径来量化这个判断。

---

## 3. TileSight 如何找到更优空间（建模 vs 现状的差）

### 3.1 模型给出两个口径

TileSight 的 RMSNorm 模型（`rmsnorm_pipeline_wave.model_rmsnorm`）按
"单遍 WSM 缓冲、IO=读 N 份+写 1 份"的第一性原理建模，带宽用实测校准值：

```
硬件能力口径:  ddr_bandwidth × ddr_max_util = 1430 × 0.9 → 1287 GB/s
镜像现状口径:  elementwise_ddr_eff = 0.65   → 930 GB/s
```

第二个口径 0.65 不是拍脑袋——它是用**镜像内 compiled RMSNorm 的 4 个 shape
实测**反推的（0.9–1.0 TB/s ≈ 1430×0.65），本质上是"当前软件生态的常规水平"。

### 3.2 关键对照：模型预言 vs 实测现实

| 口径 | 延迟（4096×4096） | 带宽 |
|---|---|---|
| TileSight 硬件能力口径（下限） | 52.1 µs | 1287 GB/s |
| TileSight 现状口径（eff=0.65） | 72.2 µs | 930 GB/s |
| 镜像 flashinfer 实测 | 77.5 µs | 864 GB/s |
| （优化后 Triton 实测） | **55.9 µs** | **1200 GB/s** |

三个信息从这里读出来：

1. **模型与现状对齐**：现状口径 72.2µs vs flashinfer 实测 77.5µs，误差 6.8%——
   模型不是空中楼阁，它准确刻画了"现在跑的是什么水平"
2. **模型证明 gap 是软件问题**：flashinfer（864 GB/s）距硬件能力（1287 GB/s）有
   **26% 的带宽空间**。RMSNorm 是纯 memory-bound，没有计算瓶颈可以背锅——
   差距只能来自 kernel 的访存组织（block 划分、wavefront 对齐、向量宽度）
3. **优化后实测 55.9µs 落在两个口径之间**（1200/1287 = 93%），验证了硬件能力口径
   是可达的、不是过于乐观的理论值

> 这一步是整个案例的方法论核心：**TileSight 的价值不是预测一个数字，而是给出
> "现状"与"极限"两条线，让优化空间可量化、让"值不值得优化"有依据。**
> 原生 torch kernel 只有 7% 带宽这个事实也由模型暴露——同一条建模线，95 GB/s
> 离地心一样远，一眼看出实现烂。

### 3.3 顺带的架构语义贡献

在跑通这条链路的过程中，我们还修复了 TileSight 建模层三处 NVIDIA 硬编码
（对任何非 32 线程 wavefront 的 GPU 都有意义）：

| 修复 | 位置 | C550 影响 |
|---|---|---|
| 寄存器占用按 32 线程/warp 计算 | `occupancy.py` | 此前低估 2 倍 |
| 线程开销量化档 32/128/256/384 | `elementwise/reduce_pipeline_wave.py` | 改为随 `arch.wavefront_size` |
| batch GEMM 的 batch 维不参与 wave 调度 | `matmul_pipeline_wave.py` | 小 grid×大 batch 高估最高 5x |

---

## 4. 如何基于这个空间进行优化

### 4.1 Kernel 设计（针对 C550 微架构的三点）

```python
@triton.jit
def _rmsnorm_kernel(X, W, Y, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)              # ① 每行一个 program
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row*stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N     # ② fp32 累加保证数值
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    y = (x * rstd * w).to(Y.dtype.element_ty)
    tl.store(Y + row*stride + cols, y, mask=mask)
```

1. **BLOCK = next_pow2(hidden)，掩码最小化**：hidden 是 2 的幂时整行单次 load，
   合并访问充分（C550 cacheline 128B、L2 全片共享 8MB，行级连续访问是带宽友好型）
2. **num_warps = clamp(BLOCK/512, 1, 16)**：4096 hidden → 8 warp = 512 线程。
   这是**为 64 线程 wavefront 专门调的**——C550 每 AP 驻留 32 wave/2048 线程，
   512 线程/block 保证多 block 驻留以隐藏访存延迟；这个参数在 NVIDIA（32 线程 warp）
   上的最优值会不同，这正是"按具体架构建模与调优"的意义
3. **fp32 累加 + bf16 存储**：与 flashinfer 数值语义一致（实测 allclose，
   atol=2e-2 全部通过）

fused 版（+residual）同理，原地写回语义与 `sgl_kernel.fused_add_rmsnorm` 对齐。

### 4.2 一个必须知道的边界：非 2 的幂 hidden 会更慢

hidden=5120（非 2 幂）时 Triton 必须带掩码，实测反而比 flashinfer 慢
（0.1098 vs 0.0912 ms）。因此 **dispatch 必须保留回退**：

```python
def patched_rmsnorm(input, weight, eps, ...):
    if _ok(hidden, dtype):        # 2的幂 + bf16/fp16 + hidden>=64
        return triton_rmsnorm(...)
    return orig_flashinfer(...)   # 其余形状回退镜像原 kernel
```

回退路径经专门验证（4096×5120 两个方向均为 1.00x，即零扰动）。

### 4.3 镜像集成方案：.pth 启动钩子（绕开 .py/.pyc 陷阱）

因为 §2.2 的不一致问题，改源码是高风险路径。改用 **site-packages `.pth` 方案**：
启动时执行补丁模块 → 注册 import hook → `sglang.srt.layers.layernorm` 加载后
自动替换其模块级 `rmsnorm` / `fused_add_rmsnorm` 两个名字（`forward_cuda` 按模块
全局名调用，与运行时 pyc 语义完全一致）。

工程细节（全部为实测踩坑）：
- **triton.jit 必须用真实源文件**：exec 字符串定义的 kernel 会
  `OSError: could not get source code`（第一版补丁静默失败的根因），kernel 独立为
  `metax_rmsnorm_kernels.py`
- **懒加载**：不在解释器启动时 import triton；补丁任何异常只打印警告、绝不破坏宿主
- **A/B 开关**：`METAX_RMSNORM_PATCH=0` 环境变量一键回到原 kernel，评估与回滚零成本
- 集成文件：`bench/metax_rmsnorm_patch.py` + `bench/metax_rmsnorm_kernels.py`

### 4.4 评估设计（三层）

1. **生效确认**：`patched: True`、`rmsnorm.__module__` 已被替换
2. **类级基准**：经 sglang RMSNorm 类端到端（不是裸 kernel 计时），4 个 shape，
   含数值校验
3. **复合层模拟**：3×matmul + 2×fused norm 的 transformer 层，prefill/decode/大 batch
   三档——因为**单 kernel 的收益在模型级会被稀释**，必须诚实度量

---

## 5. 最终实测效果

### 5.1 Kernel 级 / 类级（A/B，镜像内实测）

| shape | 无 residual | 带 residual | 数值 |
|---|---|---|---|
| 4096×4096 | 0.0814→0.0568 ms（**1.43x**） | 0.1072→0.0985（1.09x） | ✓ |
| 4096×8192 | 0.1472→0.1135（1.30x） | 0.2385→0.1909（1.25x） | ✓ |
| 16384×8192 | 0.5411→0.4283（1.26x） | 0.9108→0.7251（1.26x） | ✓ |
| 4096×5120（回退） | 1.00x | 1.00x | ✓ |

优化后带宽 1200 GB/s = 理论（1430×0.9）的 **93%**，逼近 §3.1 模型给出的硬件能力口径。

### 5.2 复合层（模型级）

| 场景 | 基线 ms | 补丁 ms | 收益 |
|---|---|---|---|
| prefill 4096 rows | 1.7451 | 1.7248 | +1.2% |
| **decode 8 rows** | 0.1248 | 0.1174 | **+6.3%** |
| 大 batch 16384 rows | 7.299 | 7.164 | +1.9% |

解读（与 §3.1 的模型口径完全自洽）：
- H=4096 层中 norm 时间占比 ~12%，1.3x 的 kernel 收益被稀释到 1–2%——**符合模型预期**，
  说明模型可以提前预判"这个优化在模型级值多少"
- **decode 小 batch 收益最大（6.3%）**：此时 GEMM 退出饱和区、norm 占比升高——
  恰好是在线推理延迟最敏感的场景

### 5.3 镜像交付

| 项 | 状态 |
|---|---|
| 新镜像 `metax-sglang:rmsnorm-opt-v0.5.12-maca3.7.1` | ✅ 已 commit（sha256:ed10775...） |
| 新起容器验证（补丁自动生效 + GPU 可用 + 数值正确） | ✅ 通过 |
| 原容器 dsv4-d-50 | ✅ 已清理补丁文件、停回原状（对其他使用者零影响） |
| 复现 | `docs/metax_c550_integration.md` §9 完整 runbook |

---

## 6. 汇报要点（给同事的三句话）

1. **建模先行，量化空间**：TileSight 用实测校准的 C550 profile 给出 RMSNorm 的
   "现状 67% / 硬件极限 90%"两条线，把"感觉 kernel 可以优化"变成"26% 的量化空间"，
   并且排除了硬件背锅的可能
2. **按微架构调优，不是按惯性调优**：64 线程 wavefront、block 驻留约束是 C550 特有的，
   Triton kernel 的 launch 参数（num_warps=BLOCK/512）直接由架构参数推导；
   同时明确了适用边界（非 2 幂 hidden 回退），不引入回归
3. **工程闭环可复制**：.pth 钩子集成（绕开 .py/.pyc 陷阱）、环境变量 A/B、三层评估
   （生效/类级/模型级）、docker commit 出新镜像——全过程 runbook 已沉淀在
   `docs/domestic_gpu_integration_guide.md`，下一个算子（如 elementwise 组合、
   activation fusion）可按同样流程推进

### 诚实的局限

- 模型级收益 1–2%（除 decode 6.3%）——单 memory-bound 小算子的宿命；更大的收益
  要找层内占比更高的算子（GEMM epilogue 融合、attention），TileSight 的 9 维
  瓶颈分解可以直接定位下一个目标
- `num_warps` 等参数目前是架构推导 + 经验验证，未做全空间扫描 autotune
- fused 版收益（1.09–1.26x）小于无 residual 版，sgl_kernel 的沐曦原生实现
  在该路径上已比较强

---

## 附：证据索引

| 内容 | 位置 |
|---|---|
| kernel 对比全量数据（5 shape × 7 实现） | `bench/results_c550_sglang_align.json` |
| A/B 评估脚本（三层） | `bench/eval_metax_rmsnorm_patch.py` |
| 补丁与 kernel 源码 | `bench/metax_rmsnorm_patch.py`、`bench/metax_rmsnorm_kernels.py` |
| 端到端验证脚本（monkey-patch 版） | `bench/metax_rmsnorm_triton.py` |
| 建模模型 | `src/tilesight/tilesight/fused_op_pipeline_wave/rmsnorm_pipeline_wave.py` |
| C550 Arch profile（含校准常数） | `src/tilesight/tilesight/arch/metax_c550.py` |
| 全过程记录 | `docs/metax_c550_integration.md`（§7-9） |
| 国产卡标准化接入流程 | `docs/domestic_gpu_integration_guide.md` |
| 硬件规格调研 | `npus/07-C550硬件规格.md` |
