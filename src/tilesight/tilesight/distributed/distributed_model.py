"""分布式模型层建模：多个分布式 op 的端到端组合。

组装层: 依赖 distributed_op.py + overlap_analysis
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from tilesight.fused_op_pipeline_wave.overlap_analysis import HardwareUsage, OpGroup, LoopNode

from .distributed_op import DistributedOpResult
from .network import CommResult

log = logging.getLogger(__name__)


@dataclass
class DistributedLayerResult:
    """一个分布式层（多个 op）的建模结果。"""
    ops: List[DistributedOpResult]
    total_compute: float = 0.0       # 所有 op 的本地计算总时间
    total_comm: float = 0.0          # 所有 op 的通信总时间
    total_latency: float = 0.0       # 含 overlap 的总延迟

    # 各并行维度的通信开销分解
    comm_breakdown: dict = field(default_factory=dict)


def model_distributed_layer(
    op_results: List[DistributedOpResult],
    comm_compute_overlap: bool = False,
) -> DistributedLayerResult:
    """组合多个 DistributedOpResult 为层级结果。

    Args:
        op_results: 各 op 的分布式建模结果
        comm_compute_overlap: 是否允许相邻 op 的通信与计算 overlap
            False: 每个 op 的 (compute + comm) 串行执行
            True: 使用 OpGroup DAG 建模 overlap

    当 comm_compute_overlap=True 时, 利用 HardwareUsage 的 network_time
    与下一个 op 的 compute time 取 max, 实现 pipeline 效果。
    """
    if not op_results:
        return DistributedLayerResult(ops=[], total_latency=0.0)

    total_compute = sum(r.compute_latency for r in op_results)
    total_comm = sum(r.comm_latency for r in op_results)

    if not comm_compute_overlap:
        # 简单模式: 每个 op 的 total_latency 串行相加
        total = sum(r.total_latency for r in op_results)
    else:
        # Overlap 模式: op[i] 的通信可以与 op[i+1] 的计算 overlap
        # 这是分布式训练中常见的 "communication hiding" 模式
        total = 0.0
        for i, r in enumerate(op_results):
            comp = r.compute_latency
            comm = r.comm_latency

            if i < len(op_results) - 1 and comm > 0:
                # 通信与下一个 op 的计算 overlap
                next_comp = op_results[i + 1].compute_latency
                # 本 op 贡献: compute + max(0, comm - next_comp)
                # (通信能被下一个计算隐藏的部分)
                exposed_comm = max(0.0, comm - next_comp)
                total += comp + exposed_comm
            else:
                total += r.total_latency

    # 通信分解
    comm_breakdown = {}
    for r in op_results:
        if r.communication is not None:
            key = r.communication.collective_type or "unknown"
            comm_breakdown[key] = comm_breakdown.get(key, 0.0) + r.comm_latency

    return DistributedLayerResult(
        ops=op_results,
        total_compute=total_compute,
        total_comm=total_comm,
        total_latency=total,
        comm_breakdown=comm_breakdown,
    )


def build_distributed_op_groups(
    op_results: List[DistributedOpResult],
    op_names: Optional[List[str]] = None,
) -> List[OpGroup]:
    """将分布式 op 结果转换为 OpGroup 列表，供 overlap_analysis 使用。

    每个 DistributedOpResult 生成两个 OpGroup:
    1. compute_group: 包含本地计算的 HardwareUsage (ddr/l2/smem/tensor/cuda/sfu)
    2. comm_group: 包含 network_time 的 HardwareUsage (如有通信)
       comm_group.depends_on = [compute_group] (通信在计算之后)

    这些 OpGroup 可以直接传入 overlap_analysis 的 model_overlap()
    与其他 op 的 OpGroup 一起做 DAG 调度。
    """
    groups = []
    if op_names is None:
        op_names = [f"op_{i}" for i in range(len(op_results))]

    for name, r in zip(op_names, op_results):
        # compute group
        comp_group = OpGroup(
            name=f"{name}_compute",
            usage=HardwareUsage(
                # 将单卡 total_latency 近似为某个 dominant 硬件单元
                # 精确建模需要从 PipelineResult 反推各单元时间
                # 这里用 ddr_time 近似 memory-bound, tensor_time 近似 compute-bound
                ddr_time=r.local_compute.total_latency * r.local_compute.ddr_util,
                tensor_time=r.local_compute.total_latency * r.local_compute.compute_util,
            ),
        )
        groups.append(comp_group)

        # comm group (如有)
        if r.communication is not None and r.comm_latency > 0:
            comm_group = OpGroup(
                name=f"{name}_comm",
                usage=HardwareUsage(network_time=r.comm_latency),
                depends_on=[comp_group],  # 通信在计算之后
            )
            groups.append(comm_group)

    return groups
