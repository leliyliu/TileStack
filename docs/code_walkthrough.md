# TileSight 代码逻辑详解

> 本文档按子包逐个解析 `src/tilesight/` 的代码逻辑：数据结构、调用链、关键算法。与 `docs/tilesight_analysis.md`（论文机制覆盖度核对）互补——前者回答"论文要求是否实现"，本文档回答"代码具体怎么跑"。所有结论附源码位置。

## 1. 概览与包结构

TileSight 是一个 tile-centric 的 GPU 性能分析建模工具，把 tile 作为一等建模单元，用 prologue–steady–epilogue pipeline envelope 递归地作用到每一层。代码分四类子包：

| 类别 | 子包 | 职责 |
|---|---|---|
| **硬件抽象** | `arch/` | `Arch` 基类 + 31 个 profile |
| **论文级建模核心** | `fused_op_pipeline_wave/`、`tile_cache/`、`util/sdcm.py` | 资源向量 / occupancy / pipeline overlap / wave / DAG 调度 / SDCM 缓存 |
| **分布式** | `distributed/` | tile 分配 / NoC 拓扑 / 集合通信 / 分布式 op 组合 |
| **算子实现与接口** | `single_op/`、`fused_op*/`、`fusion_support/`、`tir_interface/`、`welder_*/`、`compare_with_ncu/`、`onnx_model/` | 各类算子建模与外部工具接口 |

顶层 `src/tilesight/tilesight/__init__.py` 仅显式导出 `arch`/`compare_with_ncu`/`fused_op_dtype`/`fusion_support`/`util`/`welder_*`，其余子包（含论文级核心 `fused_op_pipeline_wave`、`tile_cache`、`distributed`）按需导入，避免初始化时全量加载。

## 2. `arch/` 硬件抽象

### 2.1 `Arch` 基类（`arch/arch_base.py:Arch`，`arch_base.py:24`）

所有 profile 的基类，字段分四组：

| 组 | 字段 | 语义 |
|---|---|---|
| 计算 | `sm_count`/`base_freq`/`max_freq`/`tensor_cores_per_sm`/`tensor_core_shape`/`tensor_core_flops`/`fp32_cores_per_sm`/`sfu_cores_per_sm` | SM 数、频率、tensor core 形状与吞吐、CUDA/SFU 核数 |
| 派生吞吐 | `fp16_tensor_flops`/`fp8_tensor_flops`/`fp32_cuda_core_flops`/`sfu_flops`/`smem_bandwidth`/`register_bandwidth` | 由 `sm_count*freq*cores` 算出，profile 构造时填 |
| 内存层级 | `ddr_bandwidth`/`ddr_capacity`/`l2_bandwidth`/`l2_capacity`/`configurable_smem_capacity`/`register_capacity_per_sm` | DDR/L2/SMEM/Register 容量与带宽 |
| 利用率上限 | `ddr_max_util`/`l2_max_util`/`l1_max_util`/`compute_max_util` | 各级最大利用率（默认 0.9） |
| L1.5（per-group passive） | `l1_5_group_size`/`l1_5_capacity_per_group`/`l1_5_bandwidth`/`l1_5_associativity`/`l1_5_cacheline_bytes`/`l1_5_max_util` | 默认 `group_size=0` 禁用；B200/B6000 等 Blackwell 填非 0 启用 |
| TMEM（Blackwell tcgen05） | `tmem_capacity_per_sm`/`tmem_bandwidth`/`tmem_max_util` | 默认 0 禁用；SM100 填非 0 启用 |
| 占用度 | `max_blocks_per_sm`/`sm_sub_partitions`/`warp_schedulers_per_sm`/`l1_smem_throughput_per_cycle` | occupancy 计算用 |

**所有字段有默认值**（0/False/0.9，`arch_base.py:31-79`），新 device 继承后只需填非零字段，L1.5/TMEM 默认禁用保证向后兼容。

### 2.2 两个判断函数

- `uses_gpu_resource_model(arch, include_lut=False)`（`arch_base.py:8`）：若 `arch.use_tensor_core_resource_model=True` 或 `arch.core in {A100,H100,B200}` 则启用 GPU tensor-resource 方程；非 GPU（CGRA/TPU）走另一条路径。
- `uses_dram_wave_quantization(arch)`（`arch_base.py:23`）：是否对 DRAM 传输按架构 wave 大小取整。

### 2.3 31 个 profile

`arch/__init__.py` 全量导出。覆盖 NVIDIA（A100/H100/H200/B200/B6000/H20/V100/T4/P100/RTX3090/4090/5090/RTX_PRO_6000/A6000）、AMD（MI50/MI210/MI300X/MI325X）、非 GPU（`cgra_ver15`/`cgra_ver20`、`tpuv4`、`arc770`、`maia100`）。每个 profile 提供 `set_to_spec`/`set_to_microbench`/`set_to_ncu` 三套校准（如 `b200.py:set_to_microbench` 用实测带宽覆盖 spec 值）。

### 2.4 新 device 建模步骤

1. 继承 `Arch`，构造时填计算/内存/占用度字段；
2. 有 L1.5 填 `l1_5_*`（参考 `b200.py:62`），有 TMEM 填 `tmem_*`（参考 `b200.py:_init_tmem`）；
3. 派生吞吐按 `sm_count*freq*cores` 计算；
4. 提供微基准 `set_to_microbench` 覆盖 spec 带宽。

## 3. `fused_op_pipeline_wave/` 论文级建模核心

这是论文 §3.4-3.5 的完整实现，是 TileSight 的建模心脏。

### 3.1 数据结构（`resource_types.py`）

```
TileResources
├─ per_iter: PerIterationResources   # 一次内层迭代的资源(ddr/l2/l1_5/smem_io/compute_flops)
├─ prologue_epilogue: PrologueEpilogueResources  # 填充/排空/store 资源
├─ num_iterations: int               # 内层循环次数(GEMM=gridK)
├─ stage_num: int                    # 软件流水线深度
├─ smem_footprint / reg_footprint    # occupancy 计算用
├─ warps_per_block / grids           # 占用度与 wave 用
```

`PipelineDetail`（`resource_types.py:104`）记录 prologue/steady/epilogue 时间分解；`PipelineResult`（`resource_types.py:114`）含 per_tile_latency/total_latency/各级利用率/occupancy/wave 信息。

### 3.2 资源向量与时间转换（`pipeline_overlap.py:resources_to_times`，`pipeline_overlap.py:8`）

把资源量（bytes/FLOPs）转为时间（秒）：`ddr_io/ddr_bandwidth`、`l2_io/l2_bandwidth`、`smem_io/smem_bandwidth`、`l1_5_io/l1_5_bandwidth`、`compute_flops/fp16_tensor_flops`（按 `data_bytes` 选 fp16/fp8/fp32）。`_per_sm_iter_times`（`pipeline_overlap.py:47`）区分共享资源（DDR/L2 按 `active_sms` 平分）与 per-SM 资源（SMEM/Compute 按 `sm_count`）。

### 3.3 Occupancy（`occupancy.py:compute_occupancy`，`occupancy.py:7`）

`tiles_per_sm = min(smem_limit, reg_limit, max_blocks)`，至少为 1。其中 `smem_limit = configurable_smem_capacity/smem_footprint`，`reg_limit = register_capacity_per_sm/(reg_footprint*4*32*warps)`。utcmma_cta2 2-CTA cluster 折半。

### 3.4 Pipeline Overlap（`pipeline_overlap.py`）

**核心：prologue-steady-epilogue envelope（论文 Eq.2-3）。**

`compute_pipeline_tile_latency`（`pipeline_overlap.py:84`）：
- `stage_num<=1`（无流水线）：`per_iter = mem+comp`，`total = N*per_iter + store`
- `stage_num>=2`：
  - `prologue = (stage-1)*mem_t`（填充阶段只有 load）
  - `steady = max(N-(stage-1),0) * max(mem_t, comp_t)`（稳态：load 与 compute overlap）
  - `epilogue = (stage-1)*comp_t + store`（排空 + 写回）

`compute_pipeline_tile_latency_with_occupancy`（`pipeline_overlap.py:144`）：多 tile/SM 交织时 `effective_stage = stage_num * tiles_per_sm`（**精确对应论文 Eq.3 `d = stages × resident_tiles_per_SM − 1`**，`pipeline_overlap.py:167`），`total_iters = num_iterations * tiles_per_sm`。

### 3.5 Wave Model（`wave_model.py:compute_wave_adjusted_latency`，`wave_model.py:8`）

把 per-tile/per-SM 延迟聚合为 kernel 总延迟，建模 wave head/tail：
- `head_penalty = 1 + 0.1*(prologue_time/sm_latency)`（`wave_model.py:48`）
- `num_full_waves = total_tiles // (sm_count*tiles_per_sm)`，full wave 用 `sm_latency`
- **tail wave 精确重算**：`active_sms = tail_tiles`（当 tail<sm_count），调 `compute_pipeline_tile_latency_with_occupancy(..., active_sms=active_sms)` 重算 per-tile 延迟（`wave_model.py:82`）——对应论文"tail wave 用更少 SM，每 SM 分到更多共享带宽"

### 3.6 DAG Overlap Analysis（`overlap_analysis.py`，论文 Algorithm 2）

**这是融合算子（FA-3/MLA）建模的关键：把多个 op 组成 DAG，枚举合法调度选最优。**

数据结构：
- `HardwareUsage`（`overlap_analysis.py:30`）：9 维时间向量（ddr/l2/l1_5/smem/tmem/tensor/cuda/sfu/network），`total_full_overlap` 取 max、`total_no_overlap` 求和、`__add__` 累加
- `OpGroup`（`overlap_analysis.py:75`）：一个 op 或可 overlap 的 op 组，`depends_on` 声明数据依赖
- `LoopNode`（`overlap_analysis.py:98`）：循环层次，含 `sub_groups`/`num_iters`/`sw_pipeline_stage`/`is_inner_loop`

算法：
- `_build_dag`（`overlap_analysis.py:113`）：从 `depends_on` 构建邻接表+入度
- `all_topological_sorts_builtin`（`overlap_analysis.py:153`）：Kahn 变体递归枚举所有合法拓扑序（无依赖时退化为 n! 全排列）
- `all_topological_sorts_networkx`（`overlap_analysis.py:184`）+ `verify_topological_sorts`（`overlap_analysis.py:195`）：NetworkX 交叉验证
- `simulate_schedule`（`overlap_analysis.py:224`）：给定序累加资源向量（stage<=1 串行求和，stage>=2 取 max）
- `model_overlap(try_all_orders=True)`（`overlap_analysis.py:232`）：枚举所有合法序选最优（对应论文 Eq.5 `min_σ`）
- `analyze_loop`（`overlap_analysis.py:329`）：递归分析循环——`new_stage = sw_pipeline_stage * stage`，内层循环按 Eq.2 算 prologue/steady/epilogue，外层循环逐 subgroup 递归
- `overlap_analysis`（`overlap_analysis.py:392`）：顶层入口

便捷构造：`make_op_group`（`overlap_analysis.py:406`）、`make_loop`（`:424`）、`make_loop_group`（`:433`）、`overlap_analysis_full`（`:466`，含 wave 调整并输出 12-tuple 供 `hete_post_process` 用）。

### 3.7 四类 op 的 pipeline_wave 实现

| 文件 | 入口函数 | 特点 |
|---|---|---|
| `matmul_pipeline_wave.py` | `calculate_matmul_pipeline_wave`（`:29`）+ `calculate_matmul_triton_swizzle_pipeline_wave`（`:413`） | 调 `tile_cache.multi_level_hit_rate` 做 L1.5+L2 cascade；支持 `row_panel`/`column_panel`/`raster_axis`（along_m/along_n 映射 cutlass RasterOrder） |
| `elementwise_pipeline_wave.py` | `calculate_N_0/N_1/N_N_elementwise_pipeline_wave`（`:82/171/273`） | N_0/N_1/N_N 三类（单输入/双输入同形/双输入广播） |
| `reduce_pipeline_wave.py` | 7 个 `calculate_*_pipeline_wave`（`:131/369/458/564/579`） | general reduce / inter-thread reduce / with-stride 等 |
| `conv_pipeline_wave.py` | 3 个 `calculate_conv_*_pipeline_wave`（`:92/201/219`） | implicit_gemm / sdp / nchw |

**调用链（以 GEMM 为例）**：`calculate_matmul_pipeline_wave` → `multi_level_hit_rate`（算 cache 流量分配）→ 填 `TileResources` → `compute_occupancy` → `compute_pipeline_tile_latency_with_occupancy`（prologue/steady/epilogue）→ `compute_wave_adjusted_latency`（head/tail）→ `PipelineResult`。

## 4. `tile_cache/` + `util/sdcm.py` 缓存建模

### 4.1 SDCM（`util/sdcm.py:sdcm`，`sdcm.py:20`）

随机距离缓存模型，命中概率 `P(h|D_T)`：
- `D_T <= A-1` → 1（`sdcm.py:33`）
- `D_T <= 8` → binomial 精确 `Σ comb(D,a)(A/B)^a((B-A)/B)^(D-a)`（`sdcm.py:37`，论文 Eq.7）
- `D_T > 8` → Gaussian 近似 `μ=D·A/B, σ²=D·(A/B)(1-A/B)`，`norm_input=(A-1+0.5-μ)/σ`（`sdcm.py:42-44`，论文 Eq.8-9）
- CDF 用 Zelen-Severo 三系数近似 `norm_cdf_approx_3`（`sdcm.py:15`，论文 Eq.10），另有五系数 `norm_cdf_approx_5`

### 4.2 Tile Reuse Distance（`tile_cache/reuse_distance_multilevel.py`）

`TensorAccess`（`reuse_distance_multilevel.py:37`）：`reuse_dims`（沿哪些 grid 维复用）+ `access_count`（同 block 重复访问次数）+ footprint。对应论文 Eq.6 `key(x,R)=Linearize(x_d|d∉R)`。GEMM A reuse N、B reuse M；MLA KV reuse heads；Conv weight reuse 多维（`reuse_distance_multilevel.py:60-90`）。

两级 cascade：L1.5（per-group）+ L2（global）+ DDR residual（文件头文档）。`multilevel_hit_rate_general` 通用入口；`multilevel_hit_rate_reuse_distance` GEMM 便捷版。

### 4.3 Cascade 入口（`tile_cache/cache_model.py:multi_level_hit_rate`，`cache_model.py:15`）

自动检测 `arch.l1_5_group_size`：>0 走两级 cascade（`multilevel_hit_rate_reuse_distance`），=0 退化为单 L2（`L2_hit_rate_flow_sim_reuse_distance`）。返回 `{'l1_5_hit_rate', 'l2_hit_rate', ...}`。

### 4.4 其他 cache 工具（`util/`）

- `L2_hit_rate_flow_sim_reuse_distance.py`：单层 L2 flow-sim，含 `row_panel`/`column_panel`/`raster_axis`（along_m/along_n）
- `L2_hit_rate_flow_sim_reuse_distance_triton_swizzle.py`：triton swizzle 专用
- `flash_attention_L2_hit_rate.py`/`general_reduce_L2_hit_rate.py`/`conv_nchw_L2_hit_rate.py`/`Implicit_gemm_L2_hitrate_reuse_distance.py`：算子专用 cache 估算

## 5. `distributed/` Cross-device 建模

### 5.1 依赖方向（`distributed/__init__.py` 注释）

```
arch/ ← distributed/noc/ ← distributed/collectives/
                       ↑
arch/ ← distributed/device.py ← distributed/network.py
                       ↑
fused_op_pipeline_wave/ ← distributed/distributed_op.py
```

### 5.2 设备与互联描述（`distributed/device.py`）

`MemoryLevel`/`InterconnectSpec`（latency/bandwidth/total_bandwidth）/`NodeSpec`（num_devices + intra_node InterconnectSpec）/`ClusterSpec`（node + inter_node + 可选 inter_rack）。`cluster_presets.py` 提供常用集群预设。

### 5.3 并行策略（`distributed/parallelism.py:ParallelScheme`，`parallelism.py`）

`tp/ep/sp/cp/dp/pp/fsdp + ep1/ep2`，`group_size(dim)`/`world_size()`。

### 5.4 Tile 分配（`distributed/tile_distribution.py:DistributedTileMap`，`tile_distribution.py:42`）

`dim_mapping: Dict[str,str]` 把 op 空间维映射到并行维。`local_grids`/`local_shape` 全局→单卡。`requires_collective`（`tile_distribution.py:117`）从映射 + `_TP_COLLECTIVE_RULES` 推断所需 collective。工厂：
- `make_tp_column_parallel`（切 N → all_reduce）
- `make_tp_row_parallel`（切 K → all_reduce）
- `make_tp_column_parallel_sp`（→ reduce_scatter）
- `make_sp_to_tp_all_gather`（→ all_gather）
- `make_data_parallel`（→ all_reduce 梯度）

### 5.5 NoC 拓扑（`distributed/noc/noc_topo.py`，约 1959 行）

- `Topology`（`noc_topo.py` Topology）：单层拓扑，`TopoKind` 八种（SWITCH/ALL2ALL/RING/CHAIN/MESH2D/TORUS2D/MCHAIN_NRING/MRING_NCHAIN），含 `hop_latency`/`link_bandwidth`/`switch_center_in/out_bw`
- 工厂 `make_mesh_or_torus`/`make_ring`/`make_chain`/`make_switch`/`make_all2all`（`noc_topo.py:281-438`）
- `Hierarchy`（`noc_topo.py` Hierarchy）：≤3 层（外→内 L3/L2/L1），`port_spread`（EVEN/NEAREST）、`node_mapper`、`route(src,dst,bytes)` 返回 `Chain`（hop 序列）
- `get_extend_max_routes_with_traffic`（被 `network.py:172` 调用）：返回 `(hop_time, link_time, total_time, traffic)`——**对应论文 Eq.11 的 α（hop latency）+ β（bottleneck-link serialization）**
- `RouteStats`（`noc_topo.py:905`）：`max_hop_latency`/`max_link_load`/`max_link_time`/`stage_latency`
- `noc_config_set.py`：21 个预置 reference profile（`_PROFILE_IDS`），依赖专有 `_model_support.so`；自定义拓扑用源码 `make_*` 不依赖二进制
- `traffic_matrix.py`：流量矩阵

### 5.6 NetworkHierarchy（`distributed/network.py:NetworkHierarchy`，`network.py:48`）

`ClusterSpec → Hierarchy` 自动构建（`network.py:88`）：节点内 NVSwitch→Switch，节点间 ≤16 Switch/>16 Torus，可选 rack 间第三层。`estimate_collective`（`network.py:150`）dispatch all_reduce/reduce_scatter/all_gather/all_to_all。`route_traffic`（`network.py:172`）走 `get_extend_max_routes_with_traffic`。

### 5.7 Collectives（`distributed/collectives/`）

每算法显式构造 stage + `(s,d,b)` 三元组（`pairs.append([bytes, src, dst])`），每 stage 调 `get_extend_max_routes_with_traffic` 累加：

| 算法 | 文件 | stage 数 |
|---|---|---|
| all_reduce recursive_doubling | `all_reduce.py:30` | `log2(gsize)` |
| all_reduce ring | `all_reduce.py:63` | `2*(gsize-1)` |
| all_reduce rabenseifner | `all_reduce.py:88` | reduce-scatter + all-gather |
| all_reduce double_tree | `all_reduce.py:135` | reduction + broadcast levels |
| all_reduce all_to_all | `all_reduce.py:211` | 单 stage |
| reduce_scatter rabenseifner | `reduce_scatter.py:20` | — |
| all_gather ring/all_to_all | `all_gather.py:19/50` | — |
| all_to_all uniform / ep_all_to_all | `all_to_all.py:20/41` | — |

每算法返回 `CollectiveResult(hop, link, algorithm, traffic)`。`all_reduce_auto`（`all_reduce.py:243`）按 message size/device count 自动选算法。

### 5.8 分布式 op 组合（`distributed/distributed_op.py`）

`DistributedOpResult`（`distributed_op.py:30`）：含 `local_compute`（PipelineResult）+ `communication`（CommResult）+ `overlap_ratio`。`total_latency` 按 overlap_ratio 在 `compute+comm`（串行）与 `max(compute,comm)`（全 overlap）间线性插值。

**关键：Net 维度接入 intra-tile envelope**：
- `as_hardware_usage`（`distributed_op.py:62`）把 `comm_latency` 映射到 `HardwareUsage.network_time`
- `as_op_group`（`distributed_op.py:69`）包装成 `OpGroup(usage=HardwareUsage(network_time=...))` 插入 overlap_analysis DAG

`model_distributed_op`（`distributed_op.py:60`）：调单卡 op_fn → `requires_collective` 推断 → `network.estimate_collective` → 组合。`model_distributed_op_manual`（`:92`）：已有单卡结果追加通信。

### 5.9 分布式层组合（`distributed/distributed_model.py:model_distributed_layer`，`distributed_model.py:35`）

`comm_compute_overlap=False`：各 op `total_latency` 串行相加。`comm_compute_overlap=True`：op[i] 通信与 op[i+1] 计算 overlap，`exposed_comm = max(0, comm - next_comp)`（`distributed_model.py:66`）——对应论文"cross-device movement composes with local compute through the same envelope"。

## 6. 其他算子实现

### 6.1 `single_op/`（单算子，无融合）

`matmul_op.py:calculate_matmul_resource_utilization`（`matmul_op.py:7`）等，提供资源利用率计算，是 `fused_op_*` 的基础。

### 6.2 `fused_op/` 与 `fused_op_dtype/`（早期融合实现）

`fused_op/` 最早版本（无 dtype 泛化）；`fused_op_dtype/` 加 dtype/tiling/swizzle 变体（含 `ladder_matmul`/`bitnet`/`input_stationary`/`triton_swizzle`）。`operation.py` 定义算子基类。

### 6.3 `fused_op_dtype_wave/`（经验 wave 版）

`matmul_fused_op_new_api_wave.py:calculate_matmul_resource_utilization_new`（`:10`）：含 `stage_num`/prologue/epilogue，调 `util/L2_hit_rate_flow_sim_reuse_distance`（单层 L2）。是 mosaic op 层实际用的版本，但单层 cache、无 DAG 拓扑搜索——论文级的完整实现是 `fused_op_pipeline_wave/`。

### 6.4 `fusion_support/`（算子融合）

`reg_fusion`/`smem_fusion`（寄存器/SMEM 融合）、`hete_reg_fusion`/`hete_smem_fusion`（异构融合）、`hete_post_process_single_op`（异构后处理，输出 12-tuple）、`get_hete_metrics`。`hete_smem_fusion_relaxed` 放宽约束版。

## 7. `tir_interface/` 与 `welder_*/` 外部接口

### 7.1 `tir_interface/`（TVM TIR 分析，可选依赖 tilelang/tvm）

`TileSight.py` 顶层入口，从 TIR 分析 tile 程序。`analyze_tir/`（`after_vectorize.py`/`estimate_flops.py`）、`arch/`（TIR 专用 arch：`cuda/{a100_sxm,h100_*}`、`rocm/mi250x`）、`utils/`（`device_factory`/`get_device_name`）。**未装 tilelang/tvm 时此子包不可用，但不影响核心建模路径。**

### 7.2 `welder_info_extract/` 与 `welder_modeling_top/`（Welder 编译器接口）

`welder_info_extract/`：从 Welder 编译器输出提取信息（`extract_json`/`extract_relay`/`extract_config`/`parse_reg_fused_ops`/`process_fused_plan`/`dispatch_to_modeling`/`update_operation_levels`）。`welder_modeling_top/`：顶层建模（`welder_modeling_only`/`welder_modeling_compare_ncu`/`welder_modeling_only_specify_chip`）。

## 8. 端到端调用链示例

### 8.1 单 op（GEMM）建模

```python
from tilesight.arch import B200
from tilesight.fused_op_pipeline_wave.matmul_pipeline_wave import calculate_matmul_pipeline_wave

arch = B200().set_to_microbench()
mem_levels = {'in1':[1,1,1,2], 'in2':[1,1,1,2], 'out1':[1,1,1,2]}  # ddr
result = calculate_matmul_pipeline_wave(
    op_shape=(4096,4096,4096), tb_shape=(128,128,64),
    wp_shape=(64,64,64), stage_num=3, arch=arch,
    mem_levels=mem_levels, row_panel=8, raster_axis='along_m')
# result.total_latency / result.l2_hit_rate / result.tiles_per_sm ...
```

内部链路：`multi_level_hit_rate`(L1.5+L2 SDCM) → 填 `TileResources` → `compute_occupancy` → `compute_pipeline_tile_latency_with_occupancy`(Eq.2-3) → `compute_wave_adjusted_latency`(head/tail)。

### 8.2 融合算子（FA-3 式 DAG）建模

```python
from tilesight.fused_op_pipeline_wave.overlap_analysis import (
    make_op_group, make_loop, make_loop_group, overlap_analysis_full)
from tilesight.arch import H100

arch = H100().set_to_microbench()
# 各 op 作为 OpGroup，depends_on 声明数据依赖
ld_q = make_op_group('ld_q', arch, ddr_io=..., l2_io=..., depends_on=[])
gemm1 = make_op_group('gemm1', arch, tensor_flops=..., depends_on=[ld_q])
softmax = make_op_group('softmax', arch, cuda_flops=..., sfu_flops=..., depends_on=[gemm1])
# 组成内层循环
inner = make_loop('k_loop', [ld_q, gemm1, softmax, ...], num_iters=K_iters,
                  sw_pipeline_stage=3, is_inner=True)
root = make_loop('root', [make_loop_group('body', inner)], num_iters=1, is_inner=False)
result = overlap_analysis_full(root, grids=(gridM,gridN), arch=arch, ...)
```

内部链路：`overlap_analysis` → `analyze_loop`（递归）→ `model_overlap(try_all_orders=True)`（枚举合法拓扑序选最优）→ wave 调整。

### 8.3 分布式（TP GEMM）建模

```python
from tilesight.distributed.device import ClusterSpec, NodeSpec, InterconnectSpec
from tilesight.distributed.network import NetworkHierarchy
from tilesight.distributed.parallelism import ParallelScheme
from tilesight.distributed.tile_distribution import make_tp_column_parallel
from tilesight.distributed.distributed_op import model_distributed_op_manual

# 1. 描述集群 → NetworkHierarchy
intra = InterconnectSpec(latency=1e-6, bandwidth=300e9, total_bandwidth=900e9)
node = NodeSpec(num_devices=8, intra_node=intra)
cluster = ClusterSpec(node=node, num_nodes=1)
net = NetworkHierarchy(cluster)

# 2. 并行策略 + tile 分配
parallel = ParallelScheme(tp=8)
tile_map = make_tp_column_parallel(parallel, output_bytes=4096*4096*2)

# 3. 单卡建模结果 + 通信叠加
local_result = calculate_matmul_pipeline_wave(...)  # 单卡 local GEMM
dist_result = model_distributed_op_manual(
    local_result, net, parallel,
    collective_type='all_reduce', collective_dim='tp',
    data_bytes=tile_map.collective_data_bytes, overlap_ratio=0.5)
# dist_result.total_latency = compute + max(0, comm - overlap)
```

内部链路：`estimate_collective` → `all_reduce_auto` 选算法 → stage 分解 `(s,d,b)` → 每 stage `get_extend_max_routes_with_traffic`(Eq.11) 累加 → `DistributedOpResult`。

### 8.4 多 op 层级（含 compute-comm overlap）

```python
from tilesight.distributed.distributed_model import model_distributed_layer

layer = model_distributed_layer([dist_op1, dist_op2, dist_op3],
                                comm_compute_overlap=True)
# op[i] 通信与 op[i+1] 计算 overlap，exposed_comm = max(0, comm - next_comp)
```

## 9. 关键设计要点总结

1. **三层统一于 `HardwareUsage` 9 维资源向量**：intra-tile 分解、inter-tile DAG 累加、cross-device 的 Net 都落在同一向量，经同一 envelope overlap（论文 §3 核心思想）。
2. **递归 envelope**：`analyze_loop` 递归处理嵌套循环，`new_stage = sw_pipeline_stage * stage` 实现深度累乘（Eq.3）。
3. **DAG 拓扑序搜索**：`model_overlap(try_all_orders=True)` 枚举所有合法序选最优，受限于小 DAG（n<=8），并用 NetworkX 验证——是分析调度而非 autotuning。
4. **SDCM 两级 cascade**：L1.5(per-group) + L2(global)，`arch.l1_5_group_size` 控制启用，退化单 L2。
5. **α-β 通信模型**：`hop_latency`(α) + `link_bandwidth`(1/β)，`get_extend_max_routes_with_traffic` 返回两项（Eq.11）。
6. **placement 统一 fusion 与 cross-device**：op 层 `Tensor_Loc.loc`（ddr/smem/reg）编码 fusion；分布式层 `dim_mapping` 编码跨卡。
7. **平台/依赖隔离**：专有 `.so` 仅 Linux x86-64；`tir_interface` 依赖可选 tilelang/tvm；核心路径仅需 numpy/scipy/networkx/pandas/torch。
