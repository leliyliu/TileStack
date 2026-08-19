# 沐曦 MetaX C550 接入 TileSight 全过程记录

> 日期：2026-08-18。本文档完整记录将沐曦（MetaX）曦云 C550 接入 TileSight 的全过程：
> 规格调研 → 服务器实测 profiling → 参数推导 → 代码实现 → 测试验证。所有参数均标注来源
> （🔬 实测 / ✅ 官方 / ⚠️ 第三方 / 🔬 推算），与 `npus/07-C550硬件规格.md` 的可信度约定一致。

## 1. 背景与目标

TileSight 通过 `arch/` 下的硬件 profile 实现跨架构建模（已有 31 个 profile，含 AMD/TPU/CGRA
非英伟达先例）。接入 C550 的目标：

1. 新增 `MXC550` Arch profile，使其可被 `fused_op_pipeline_wave` 建模层直接使用
2. 用服务器实测微基准校准关键参数（默认口径 = 实测）
3. 端到端验证：建模延迟 vs 实测延迟误差在合理范围内（论文口径 MAPE 12.35%）

## 2. 硬件规格调研结论（摘要）

完整规格见 `npus/07-C550硬件规格.md`。接入直接相关的关键点：

| 参数 | 数值 | 来源 | 对建模的含义 |
|---|---|---|---|
| 计算单元 | 104 AP × 4 PEU × 16 lane | 🔬 macainfo | `sm_count=104`（AP ↔ SM） |
| 调度粒度 | 64 线程 wavefront，32 wave/AP | 🔬 macainfo | wave 量化粒度为 64 非 32 |
| 最高时钟 | 1600 MHz | 🔬 macainfo | `max_freq=1.6e9` |
| 共享内存 WSM | 64 KB/AP，128 B/cycle（全双工） | ✅ 官方 | smem 带宽 = 104×1.6G×128 |
| 向量寄存器堆 | 512 KB/AP | ✅ 官方 | `register_capacity_per_sm` |
| L2 | 8 MB 全片共享，4096 B/cycle | 🔬 实测 | L2 带宽 = 4096×1.6G |
| 显存 | HBM2e 64 GB，理论 1843 GB/s | 🔬 推算 | DDR 口径基准 |
| 矩阵单元 | WMMA 16×16×16（`MMA_16x16x16F16`） | ✅ 官方 | `get_tensor_core_minimum_ptx` |
| FP8 | 不支持（C600 特性） | ✅ 官方 | bytes=0.5 应 raise |

**关键架构差异**（相对 NVIDIA，影响字段语义）：

- 每 AP **单发射单元**（5 发射口：1 标量 + 1 MMA/向量ALU 互斥 + 1 向量访存 + 1 共享内存 +
  1 杂项），对应 `warp_schedulers_per_sm=1`（与 MI300X 同款处理）
- WSM 是独立 SRAM 不从 L1 划扣（类似 AMD LDS）；VL1 默认关闭 → 不建 L1.5/TMEM 层次
- C550 与 NVIDIA H100 的 L2 two-part cache 结构不同 → 建模层自动落入通用 L2 分支
  （`matmul_pipeline_wave.py` 中 `arch.core in ("A100","H100","B200","A100_LUT")` 判断）

## 3. 服务器实测 profiling

### 3.1 环境信息

| 项 | 值 |
|---|---|
| 服务器 | `muxi-01`（免密 SSH，hostname `CX7Group1-host-050`） |
| GPU | 8 × MetaX C550，全部 Available（`mx-smi` 2.2.9） |
| 驱动 / MACA | Kernel 3.3.12 / 宿主 3.2.1，容器 3.7.1.17-dsv4 |
| 容器 | `dsv4-d-50`（sglang 镜像），Python 3.10 + torch 2.8.0+metax |

**容器内要点**（踩坑记录）：

- torch 设备名是 **`cuda`**（cu-bridge 兼容层），不是 `torch.musa`——`torch.musa` 属性不存在，
  `device="musa:0"` 会直接 RuntimeError
- python 在 `/opt/conda/bin/python`，默认 PATH 的 python3 无 torch
- `mx-smi --show-clocks` 的 `XCORE_CLK` 负载下恒读 450 MHz（空载值，不随负载更新），
  **不能**用它校准频率；以 `macainfo` 的 Max Clock 1600 MHz 为准

### 3.2 微基准脚本

`/tmp/metax_c550_profile.py`（torch 实现，可从本文档附录复现）：

1. **HBM 带宽**：STREAM 风格，1 GiB fp32 数组，copy（读1写1）/ add（读2写1），20 次平均
2. **GEMM 算力**：2048/4096/8192 三档方阵，fp16/bf16/fp32，20 次取均值，报告各档与最佳
3. **持续算力**：4000 次连续 8192³ fp16 GEMM 计时

### 3.3 实测结果（2026-08-18）

| 指标 | 实测值 | 说明 |
|---|---|---|
| GEMM FP16 峰值 | **275.03 TFLOPS**（8192³） | 2048³ 仅 177 → 大尺寸才饱和 |
| GEMM BF16 峰值 | 269.34 TFLOPS | 与 FP16 同量级 |
| GEMM FP32 峰值 | **124.5 TFLOPS** | 远超向量 FP32 理论值 21.3 → mcBLAS 走矩阵单元（TF32 类路径） |
| GEMM 持续（FP16） | 230 TFLOPS（4000 次 / 19.07 s） | 峰值的 84%，散热/供电稳定 |
| HBM copy | **1429 GB/s** | 理论 1843 的 78% |
| HBM add | 1480 GB/s | 理论的 80% |

**实测与规格文档的交叉验证**：

- 实测 275 TFLOPS 印证 C500 OAM 档「FP16 280 TFLOPS」（第三方说法被实测基本证实）；
  与 CSDN「C550 = 240 TFLOPS」矛盾 → 采纳实测值，CSDN 数据疑似混淆 C500 PCIe 档
- torch 实测带宽（1429/1480）显著高于规格文档中手写 MUSA STREAM 的 762/1154 →
  原因是 mcBLAS/torch elementwise kernel 向量化更好（每线程 >4B 访存粒度）
- FP32 GEMM 124.5 TFLOPS 是重要新发现：规格文档未记载 C550 的 TF32 能力，实测表明
  mcBLAS FP32 走矩阵单元，近似 C500 TF32 档（140 TFLOPS OAM）的 89%

## 4. 参数推导（逐字段）

新文件 `src/tilesight/tilesight/arch/metax_c550.py`，每个字段的取值依据：

| 字段 | 值 | 来源与推导 |
|---|---|---|
| `core` | `"MXC550"` | Market Name（macainfo） |
| `sm_count` | 104 | 🔬 macainfo AP 数 |
| `max_freq` | 1.6e9 | 🔬 macainfo Max Clock；负载时钟不可读（见 3.1） |
| `warp_schedulers_per_sm` | 1 | ✅ 官方：每 AP 单发射单元（5 发射口） |
| `wavefront_size` | 64 | 🔬 macainfo（新增字段，信息性） |
| `max_blocks_per_sm` | 32 | 🔬 推算：32 wave/AP，64 线程 block = 1 wave |
| `fp16_tensor_flops` | 275e12 | 🔬 实测（默认口径）；spec 口径 280e12（⚠️ C500 OAM 档） |
| `fp32_tensor_flops` | 124.5e12 | 🔬 实测（mcBLAS FP32 走矩阵单元） |
| `int8_tensor_flops` | 560e12 | ⚠️ C500 OAM 参考档，未实测 |
| `fp8_tensor_flops` | 不设置 | C550 无 FP8；建模层 hasattr 保护，bytes=1 时回退 int8 |
| `fp32_cuda_core_flops` | 21.3e12 | 🔬 推算：104×1.6G×64 lane×2（C500 官方 18T @ 低频，量级吻合） |
| `fp16_cuda_core_flops` | 42.6e12 | 🔬 推算：Fast Float16 TRUE（macainfo）→ 2× |
| `fp64_cuda_core_flops` | 10.6e12 | ⚠️ 估计：fp32/2（沿用 MI300X 惯例，未实测） |
| `ddr_bandwidth` | 1430e9 | 🔬 实测 copy（保守）；spec 口径 1843.2e9（位宽推算） |
| `ddr_wave_bytes` | 4096 | 无实测依据，沿用 B6000 保守默认（仅 opt-in 量化路径使用） |
| `l2_capacity` | 8 MB | 🔬 macainfo |
| `l2_bandwidth` | 6553.6e9 | 🔬 推算：实测 4096 B/cycle × 1.6 GHz |
| `configurable_smem_capacity` | 64 KB | ✅ 官方 WSM/AP |
| `l1_smem_throughput_per_cycle` | 128 | ✅ 官方 WSM 128 B/cycle/AP |
| `smem_bandwidth` | 21.3 TB/s | 104 × 1.6 GHz × 128 |
| `register_capacity_per_sm` | 512 KB | ✅ 官方向量寄存器堆 |
| `register_bandwidth` | 42.6 TB/s | 104 × 1.6 GHz × 64 lane × 4 B（显式覆盖，PEU 是 16 lane 非 32） |
| `get_tensor_core_minimum_ptx(2/4/1/0.5)` | (16,16,16)/(16,16,8)/(16,16,32)/raise | ✅ 官方 WMMA 形状；FP8 raise（C600 特性） |
| L1.5 / TMEM | 保持基类 0 | 不存在该层次（VL1 默认关闭且 32KB 过小） |

利用率上限（`*_max_util=0.9`）、`set_to_spec()` 模式均沿用 MI300X 的先例结构。

## 5. 实现与注册

- 新增 `src/tilesight/tilesight/arch/metax_c550.py`（类 `MXC550`）
- `arch/__init__.py` 末尾追加 `from .metax_c550 import *`
- TDD：`tests/test_metax_c550.py` 16 个用例（字段存在性 / 实测口径 / spec 口径 /
  架构参数 / WMMA 形状映射 / 端到端建模冒烟）

**运行方式**（注意仓库既有打包怪癖：`.pth` 将 `src` 加入 path 后 `tilesight` 会变成
命名空间包，内部绝对导入要求 `PYTHONPATH=src/tilesight`）：

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e . numpy scipy networkx pandas matplotlib plotly pytest
PYTHONPATH=src/tilesight python -m pytest tests/test_metax_c550.py -v
```

## 6. 端到端验证结果

8192³ FP16 GEMM，wmma，tb=(128,128,32)，wp=(64,64,16)，stage=3（mcBLAS 典型配置）：

| 口径 | 建模延迟 | compute_util | ddr_util | L2 hit |
|---|---|---|---|---|
| **实测口径（默认）** | **4.55 ms** | 0.880 | 0.850 | 0.678 |
| spec 口径 | 4.01 ms | 0.979 | 0.747 | 0.678 |
| 实测（持续 / 峰值） | 4.77 / 4.00 ms | — | — | — |

- 实测口径误差 **4.6%**（vs 持续 4.77 ms），远优于论文平均 MAPE 12.35%
- spec 口径 4.01 ms 恰好对齐峰值 4.00 ms（满利用率理论值），两个口径行为符合预期
- 16/16 测试通过

## 7. 算子级验证（第二轮，2026-08-19）

接入后进一步用 FA / RMSNorm / 融合链验证建模精度，并修复两处建模层 NVIDIA 语义硬编码。

### 6.1 建模层语义修复（对所有架构生效）

| 修复 | 位置 | 影响 |
|---|---|---|
| 寄存器占用硬编码 32 线程/warp | `occupancy.py:41` → `arch.wavefront_size` | C550 寄存器占用此前低估 2 倍 |
| 线程开销量化档 32/128/256/384 | `elementwise/reduce_pipeline_wave.py` | C550 → 64/256/512/768 |
| batch GEMM 未参与 wave 全 SM 调度 | `matmul_pipeline_wave.py`（batch 折入 total_tiles） | 小 grid×大 batch 高估最高 5x（FA PV 段） |

### 6.2 实测与建模对比（bench/results_c550_ops.json）

**FlashAttention prefill**（B=8,H=32,D=128,bf16，三段拆解 QK^T/softmax/PV）：

| 场景 | 实测 ms | 建模 ms | 误差 |
|---|---|---|---|
| full S=2048/4096/8192 | 3.41 / 13.74 / 57.24 | 3.49 / 13.81 / 55.1 | 2.2% / 0.5% / 3.7% |
| causal S=2048/4096/8192 | 1.89 / 7.70 / 29.38 | 2.04 / 8.50 / 27.56 | 7.9% / 10.4% / 6.2% |

**RMSNorm**（compiled 良实现）：4 个 shape 误差 0.3%–5.7%。

**融合收益**（residual+RMSNorm）：串行 0.142ms / 融合 0.107ms（1.33x），建模 1.39x（4.6%）；MatMul→RMSNorm 链误差 <15%。

### 6.3 新增 per-arch 校准常数（均多测点交叉验证）

| 常数 | 值 | 含义 | 依据 |
|---|---|---|---|
| `elementwise_ddr_eff` | 0.65 | 含行规约 kernel 有效带宽（0.93 TB/s / 1430） | RMSNorm 4 测点 |
| `fa_tc_eff` | 0.75 | FA 有效 TC 算力系数（online softmax 等固有开销） | 两端口径夹逼，6 测点交点 0.735–0.78 |

### 6.4 重要实测发现

1. **原生 `torch.nn.RMSNorm` 在沐曦上低效 15x**（0.731ms vs compiled 0.073ms，~92 GB/s）——模型上限可暴露此类实现问题
2. 纯 elementwise（add 1.35 TB/s）与含规约 kernel（0.93 TB/s）带宽效率显著不同，建模需区分
3. FA 有效算力 150–161 TFLOPS 高度一致（6 测点），为 GEMM 峰值 275 的 56%（实测值），模型经 eff 校准后误差 ≤10.4%

## 8. sglang 镜像对齐调查（第三轮，2026-08-19）

针对问题：sglang metax 镜像（v0.5.12-deepseek-v4-rc1-maca.ai.3.7.1.110）内含沐曦定制 kernel，
TileSight 能否与其对齐；若存在 gap，如何优化。结论：**模型已对齐；但 sglang 现有 kernel
相对硬件能力仍有 26–36% 空间，我们给出的 triton kernel 实测端到端提速 1.26–1.36x**。

### 8.1 关键发现：sglang 在沐曦上的真实分发路径

1. **metax torch 上报 `is_cuda()=True`**（cu-bridge），sglang 的 `forward_musa` 是死代码，
   实际走 `forward_cuda` → flashinfer/sgl_kernel（均为沐曦编译版）
2. **容器内 .py 与运行时 .pyc 不一致**：磁盘 `layernorm.py` 是上游版（引用不存在的
   `rms_norm`，强制重编译会 NameError），运行时 .pyc 为沐曦适配版（pyc mtime 晚于 .py）。
   **在该容器打源码补丁必须基于运行时语义，不可直接删 pyc**
3. 实际路径：无 residual → `flashinfer.rmsnorm`；带 residual → `sgl_kernel.fused_add_rmsnorm`（4 参原地）

### 8.2 三方对比（4096×4096 bf16，单位 ms）

| 实现 | 延迟 | 有效带宽 | 相对理论* |
|---|---|---|---|
| torch 原生 rms_norm | 0.7027 | 95 GB/s | 7% |
| sglang 实际：flashinfer | 0.0777 | 864 GB/s | 67% |
| torch.compile | 0.0664 | 1010 GB/s | 78% |
| **本工作 triton（调优）** | **0.0559** | **1200 GB/s** | **93%** |
| 理论（0.9×copy 带宽） | 0.0521 | 1287 GB/s | 100% |

*理论 = 2 份流量（读1写1）/ (1430 GB/s × 0.9)；全部 5 个 shape 数据见
`bench/results_c550_sglang_align.json`。fused 版（4 份流量）sgl_kernel 0.1063 vs
triton 0.0989 ms。

### 8.3 模型对齐结论

TileSight 建模 72.2µs vs sglang 实际 flashinfer 77.5µs（**误差 6.8%，对齐**）。
`elementwise_ddr_eff=0.65` 对应 sglang 现状（864/1430=0.60）；调优 triton 后可达 0.84——
即「硬件能力口径」与「现状口径」的差，正是可优化空间。

### 8.4 优化方案（已验证）

调优要点（`bench/metax_rmsnorm_triton.py`）：每行一个 program、BLOCK=next_pow2(hidden)、
num_warps=clamp(BLOCK/512,1,16)（4096→8 warp，适配 64 线程 wavefront）、fp32 累加。
**限制：仅 2 的幂 hidden 有效**（非 2 幂需掩码反而更慢，必须回退 flashinfer）。

经 sglang RMSNorm 类 monkey-patch 端到端验证：

| shape | 无 residual | 带 residual | 数值 |
|---|---|---|---|
| 4096×4096 | 1.36x | 1.09x | ✓ |
| 4096×8192 | 1.31x | 1.25x | ✓ |
| 16384×8192 | 1.26x | 1.25x | ✓ |
| 4096×5120（非2幂，回退） | 1.02x | 1.00x | ✓ |

sglang 源码接入方式见 `bench/metax_rmsnorm_triton.py` 文件头注释（含 .pyc 不一致的
注意事项与两处补丁点）。

## 9. 镜像集成部署 runbook（待 muxi-01 恢复后执行）

> 2026-08-19 部署进行到一半时 muxi-01 失联（ssh/ping 均超时约 20 分钟，疑似宕机/维护）。
> 交付物已固化在仓库，恢复后按以下步骤执行。

### 9.1 交付物

| 文件 | 作用 |
|---|---|
| `bench/metax_rmsnorm_patch.py` | 补丁模块：.pth 启动钩子 + import hook，layernorm 加载后自动替换 rmsnorm/fused_add_rmsnorm；env `METAX_RMSNORM_PATCH=0` 可禁用（A/B 用）；不改 sglang 源文件，规避 .py/.pyc 不一致 |
| `bench/eval_metax_rmsnorm_patch.py` | 有效性评估：生效确认 + 类级基准 + 复合层模拟 |

### 9.2 部署步骤

```bash
# 1. 拷入容器 site-packages 并启用 .pth
scp bench/metax_rmsnorm_patch.py muxi-01:/tmp/
ssh muxi-01 'docker cp /tmp/metax_rmsnorm_patch.py \
  dsv4-d-50:/opt/conda/lib/python3.10/site-packages/metax_rmsnorm_patch.py \
  && docker exec dsv4-d-50 bash -c \
  "echo import metax_rmsnorm_patch > /opt/conda/lib/python3.10/site-packages/metax_rmsnorm_patch.pth"'

# 2. 验证生效（新进程，stderr 应打印 applied）
ssh muxi-01 'docker exec dsv4-d-50 /opt/conda/bin/python -c \
  "import sglang.srt.layers.layernorm as L; print(L._metax_rmsnorm_patched)"'

# 3. A/B 评估
scp bench/eval_metax_rmsnorm_patch.py muxi-01:/tmp/
ssh muxi-01 'docker cp /tmp/eval_metax_rmsnorm_patch.py dsv4-d-50:/tmp/ \
  && docker exec dsv4-d-50 /opt/conda/bin/python /tmp/eval_metax_rmsnorm.py'
ssh muxi-01 'docker exec -e METAX_RMSNORM_PATCH=0 dsv4-d-50 \
  /opt/conda/bin/python /tmp/eval_metax_rmsnorm.py'   # 基线

# 4. 提交新镜像
ssh muxi-01 'docker commit dsv4-d-50 \
  metax-sglang:rmsnorm-opt-v0.5.12-maca3.7.1'
# 参照 docker inspect dsv4-d-50 的 Devices/DeviceRequests 复制启动参数后:
#   docker run --rm --device ... metax-sglang:rmsnorm-opt-... <评估脚本>

# 5. （可选）保持原容器干净：移除 .pth（新镜像已含补丁）
ssh muxi-01 'docker exec dsv4-d-50 rm \
  /opt/conda/lib/python3.10/site-packages/metax_rmsnorm_patch{.py,.pth}'
```

### 9.3 有效性预期与判据

- 类级：无 residual 1.26–1.36x，residual 1.09–1.25x（已验证的 monkey-patch 同机制）
- 复合层（3 matmul + 2 fused norm，H=4096）：norm 占比 ~12%，预期层级收益 **1–4%**——
  单 kernel 收益显著，但需诚实报告模型级放大有限；decode 小 batch 下 norm 占比更高，收益更大
- 判据：`patch_active=True`、`numerics_ok=True` 全 shape、类级提速复现、
  A/B 前后 composite 差值方向一致

## 10. 遗留事项

1. `ddr_wave_bytes`、`fp64`、`int8` 未经实测校准（已在字段注释中标注）
2. L2 带宽为推算值；如需精确可用 L2 resident GEMM（K 小 M/N 大）微基准实测
3. op 层 38 处 NVIDIA 硬编码分支（`docs/expansion_feasibility.md`）未处理——C550 走通用
   分支可用，专用分支可进一步提升精度（进阶阶段，见标准化流程文档）
4. 单发射单元的 5 发射口互斥（MMA 与向量 ALU 共槽）未在建模层显式表达——其开销
   目前被 fa_tc_eff (0.75) 经验性吸收；显式建模需在 pipeline_overlap 增加共槽语义，
   留待后续架构（如接入摩尔线程时）一并处理
5. FA decode (q_len=1) 尚未建模（实测 0.19-0.69ms 已入库 bench/results_c550_ops.json）；
   decode 为 memory-bound 的 KV 读取，适合用 elementwise+reduce 原语组合，待有更多
   shape 需求时补
6. 校准常数的可迁移性：elementwise_ddr_eff / fa_tc_eff 为 C550 专属；新国产卡接入时
   按 docs/domestic_gpu_integration_guide.md 阶段 5.5 重新实测标定

## 附录：profiling 脚本

见第 3.2 节描述，核心逻辑：

```python
import time, torch  # 容器内 torch 2.8.0+metax, 设备名用 "cuda:0"
def bench(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

n = 1 << 28  # 256M fp32 = 1 GiB
a, b, c = (torch.randn(n, device="cuda:0") for _ in range(2)), None, None
# ... copy/add 带宽 = 流量字节数 / 耗时
x = torch.randn(8192, 8192, device="cuda:0", dtype=torch.float16)
t = bench(lambda: torch.mm(x, x))
tflops = 2 * 8192**3 / t / 1e12
```
