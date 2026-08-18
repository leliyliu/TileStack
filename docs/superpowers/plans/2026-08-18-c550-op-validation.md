# C550 算子级建模验证与整合（FA / RMSNorm / 融合）实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 验证 MXC550 profile 在 FlashAttention（三段拆解）、RMSNorm、算子融合场景下"准确反馈性能"（误差 ≤15%），并修复建模层两处 NVIDIA 32 线程/warp 硬编码语义。

**架构：** 实测基准（muxi-01 容器 flash_attn/torch）产生 ground truth → 本地 TileSight 用现有原语（matmul/reduce/elementwise pipeline_wave + overlap_analysis DAG）建模 → 对比校准。wavefront 语义修复通过 `arch.wavefront_size` 属性驱动，默认 32 保持 NVIDIA 行为不变。

**技术栈：** TileSight（本仓库）、muxi-01 容器（torch 2.8.0+metax，flash_attn 沐曦移植版，设备名 `cuda`）、pytest。

---

### 任务 1：wavefront_size 语义修复（建模层去 NVIDIA 硬编码）

**文件：**
- 修改：`src/tilesight/tilesight/arch/arch_base.py`（Arch.__init__ 增加 `self.wavefront_size = 32` 默认值）
- 修改：`src/tilesight/tilesight/fused_op_pipeline_wave/occupancy.py:41`（32 → arch.wavefront_size）
- 修改：`src/tilesight/tilesight/fused_op_pipeline_wave/elementwise_pipeline_wave.py:38` 与
  `src/tilesight/tilesight/fused_op_pipeline_wave/reduce_pipeline_wave.py:35`（`_compute_thread_overhead` 增加 wavefront 参数）
- 测试：`tests/test_metax_c550.py` 追加

- [ ] **步骤 1：编写失败的测试**

```python
class TestWavefrontSemantics:
    def test_arch_base_default_32(self):
        # NVIDIA 卡无 wavefront_size 属性 → 基类默认 32, 行为不变
        from tilesight.arch.arch_base import Arch
        assert Arch().wavefront_size == 32

    def test_mxc550_wavefront_64(self, arch):
        assert arch.wavefront_size == 64

    def test_occupancy_uses_wavefront_size(self, arch):
        # C550: 64 线程/warp, reg_footprint 单位为 per-warp 4B 寄存器数
        # 512KB regfile, reg_footprint=64, warps_per_block=4 (256 线程):
        #   per-block = 64*4B*64*4 = 64KB → 512KB/64KB = 8 blocks
        from tilesight.fused_op_pipeline_wave.occupancy import compute_occupancy
        tiles = compute_occupancy(smem_footprint=0, reg_footprint=64,
                                  warps_per_block=4, arch=arch)
        assert tiles == 8
```

- [ ] **步骤 2：运行验证失败** — `PYTHONPATH=src/tilesight python -m pytest tests/test_metax_c550.py::TestWavefrontSemantics -v`，预期 test_occupancy FAIL（occupancy 算出 16，因 32 硬编码）
- [ ] **步骤 3：实现** — arch_base 加默认；occupancy.py 改为：

```python
threads_per_warp = getattr(arch, 'wavefront_size', 32)
if reg_footprint > 0 and warps_per_block > 0:
    reg_per_block_bytes = reg_footprint * 4 * threads_per_warp * warps_per_block
```

`_compute_thread_overhead` 两处改为接收 `wavefront_size`，量化档位取 `[w for w in (wavefront_size, 4*w, 8*w, 12*w) if w <= thread_per_tb]` 逻辑等效迁移（NVIDIA 32 → 32/128/256/384 不变，C550 64 → 64/256/512/768）；调用点传 `getattr(arch, 'wavefront_size', 32)`。
- [ ] **步骤 4：跑全部既有测试确认无回归**（A100 等 NVIDIA profile 无 wavefront_size 属性 → 走默认 32）
- [ ] **步骤 5：Commit** — `fix: 建模层寄存器/线程开销按 arch.wavefront_size 计算 (C550=64)`

### 任务 2：实测基准脚本（FA / RMSNorm / 融合）

**文件：**
- 创建：`bench/metax_c550_ops_bench.py`（同步放 muxi-01 容器运行）
- 产出：`bench/results_c550_ops.json`（拷回本地）

- [ ] **步骤 1：编写脚本**，基准集（统一 bench_kernel 计时器，20 次平均，bf16）：
  - FA prefill：`flash_attn.flash_attn_func`，(B,S,H,D)=(8, {2048,4096,8192}, 32, 128)，causal=True 与 False
  - FA decode：同 shape q_len=1
  - RMSNorm：`torch.nn.RMSNorm(4096/8192)`，batch 4096 行
  - residual+RMSNorm：串行两 op（add + rmsnorm）实测，与 torch.compile 融合版对比
  - MatMul→RMSNorm 链：串行实测
- [ ] **步骤 2：scp + docker cp 到 muxi-01，运行采集**（注意设备名 `cuda`，`/opt/conda/bin/python`）
- [ ] **步骤 3：拷回 JSON，记录到文档**
- [ ] **步骤 4：Commit**（脚本与结果入库）

### 任务 3：RMSNorm 建模与对比

**文件：**
- 创建：`src/tilesight/tilesight/fused_op_pipeline_wave/rmsnorm_pipeline_wave.py`（组合 reduce + elementwise 原语，或直接用 `calculate_general_inter_thread_reduce_pipeline_wave` + scale 段）
- 测试：`tests/test_metax_c550.py` 追加 `TestRMSNormModel`

- [ ] **步骤 1：失败测试** — 对 hidden=4096/8192，建模延迟 vs 实测误差 ≤15%：

```python
def test_rmsnorm_vs_measured(self, arch):
    from tilesight.fused_op_pipeline_wave.rmsnorm_pipeline_wave import model_rmsnorm
    r = model_rmsnorm(rows=4096, hidden=4096, arch=arch)  # 返回 (latency, util)
    measured = MEASURED["rmsnorm_4096"]  # 从 bench/results_c550_ops.json 读
    assert abs(r.total_latency - measured) / measured < 0.15
```

- [ ] **步骤 2：验证失败** → **步骤 3：实现**（reduce：mean of x² + rsqrt(SFU)；scale：elementwise；DAG 内 reduce→scale 依赖链）→ **步骤 4：对比**，超差则调 ddr/L2 命中假设并记录 → **步骤 5：Commit**

### 任务 4：FlashAttention 三段 DAG 建模与对比

**文件：**
- 创建：`src/tilesight/tilesight/fused_op_pipeline_wave/flash_attention_pipeline_wave.py`
  - 段1 QK^T：`calculate_matmul_pipeline_wave`((B·H, S_q, S_k, D)，tb=(128,128,32) 等典型 tile)
  - 段2 softmax：按行 reduce + exp（SFU）+ 归一化，OpGroup 建模
  - 段3 PV：`calculate_matmul_pipeline_wave`((B·H, S_q, D, S_k))
  - K/V 的 L2 命中用 `flash_attention_L2_hit_rate` util
  - 三段用 `make_op_group`/`make_loop` 组 DAG（依赖链，串行 + 分块内在线 softmax 与下一段 load overlap）
- 测试：`tests/test_metax_c550.py` 追加 `TestFlashAttentionModel`（各 shape 误差 ≤15%）

- [ ] **步骤 1-5：同 TDD 循环**（实测数据来自任务 2 的 JSON）

### 任务 5：融合收益建模与对比

**文件：**
- 测试：`tests/test_metax_c550.py` 追加 `TestFusionGain`
- 用 overlap_analysis 对比：residual+RMSNorm 未融合（两 op 串行，中间写回 DDR）vs 融合（DAG 内 producer-consumer 经 SMEM/寄存器直传）；MatMul→RMSNorm 同理

- [ ] **步骤 1：失败测试** — 建模的融合收益比（串行延迟/融合延迟）与实测收益比方向一致且误差 ≤20% → **步骤 2-5：实现/对比/Commit**

### 任务 6：文档更新与收尾

- [ ] 更新 `docs/metax_c550_integration.md`：新增"算子级验证"章（FA/RMSNorm/融合对比表、校准记录、MMA/ALU 共槽限制说明）
- [ ] 更新 `docs/domestic_gpu_integration_guide.md`：标准化流程补"阶段 5.5 算子级验证"（FA 三段拆解模板 + wavefront 语义检查项）
- [ ] 全量测试 + commit + push
