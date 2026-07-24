"""分布式 Op 建模：单卡计算 + 集合通信的统一入口。

这是组装层——唯一同时 import fused_op_pipeline_wave + distributed 子模块的地方。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

from tilesight.fused_op_pipeline_wave.resource_types import PipelineResult
from tilesight.fused_op_pipeline_wave.overlap_analysis import HardwareUsage, OpGroup

from .network import NetworkHierarchy, CommResult
from .parallelism import ParallelScheme
from .tile_distribution import DistributedTileMap

log = logging.getLogger(__name__)


@dataclass
class DistributedOpResult:
    """分布式单 op 建模结果: 本地计算 + 可选通信。"""
    local_compute: PipelineResult          # 单卡 pipeline wave 结果
    communication: Optional[CommResult] = None  # 通信开销
    overlap_ratio: float = 0.0              # 0=串行, 1=完全 overlap

    @property
    def compute_latency(self) -> float:
        return self.local_compute.total_latency

    @property
    def comm_latency(self) -> float:
        if self.communication is None:
            return 0.0
        return self.communication.total_time

    @property
    def total_latency(self) -> float:
        """根据 overlap_ratio 计算总延迟。

        overlap_ratio=0: total = compute + comm (完全串行)
        overlap_ratio=1: total = max(compute, comm) (完全 overlap)
        中间值: 线性插值
        """
        comp = self.compute_latency
        comm = self.comm_latency
        if comm == 0.0:
            return comp
        serial = comp + comm
        overlapped = max(comp, comm)
        return serial - (serial - overlapped) * self.overlap_ratio

    def as_hardware_usage(self) -> HardwareUsage:
        """转换为 HardwareUsage，可插入 overlap_analysis 的 OpGroup DAG。

        本地计算用原始 HardwareUsage（如有），
        通信映射到 network_time 字段。
        """
        return HardwareUsage(network_time=self.comm_latency)

    def as_op_group(self, name: str = "comm") -> OpGroup:
        """转换为 OpGroup，通信作为纯 network_time 的 op group。"""
        return OpGroup(
            name=name,
            usage=HardwareUsage(network_time=self.comm_latency),
        )


def model_distributed_op(
    op_fn: Callable[..., PipelineResult],
    op_kwargs: dict,
    network: NetworkHierarchy,
    tile_map: DistributedTileMap,
    overlap_ratio: float = 0.0,
) -> DistributedOpResult:
    """统一入口: 单 op 的分布式建模。

    步骤:
    1. 根据 tile_map 调整 op shape 为 local shape
    2. 调用现有 TileSight 单卡 op_fn 得到 PipelineResult
    3. 如有 collective, 通过 NetworkHierarchy 估算通信开销
    4. 组合为 DistributedOpResult

    Args:
        op_fn: 现有 calculate_xxx_pipeline_wave 函数
        op_kwargs: 传给 op_fn 的参数 (可包含 'op_shape' 字典用于自动调整)
        network: NetworkHierarchy 对象
        tile_map: DistributedTileMap 描述 tile 分布
        overlap_ratio: compute-comm overlap 比例 (0~1)
    """
    # 1. 调用单卡建模
    local_result = op_fn(**op_kwargs)

    # 2. 检查是否需要 collective
    coll_info = tile_map.requires_collective()
    comm_result = None
    if coll_info is not None:
        coll_type, coll_dim = coll_info
        data_bytes = tile_map.collective_data_bytes
        if data_bytes is None:
            log.warning("collective_data_bytes 未指定，跳过通信建模")
        else:
            comm_result = network.estimate_collective(
                coll_type, tile_map.parallel, coll_dim, data_bytes,
            )

    return DistributedOpResult(
        local_compute=local_result,
        communication=comm_result,
        overlap_ratio=overlap_ratio,
    )


def model_distributed_op_manual(
    local_result: PipelineResult,
    network: NetworkHierarchy,
    parallel: ParallelScheme,
    collective_type: str,
    collective_dim: str,
    data_bytes: int,
    overlap_ratio: float = 0.0,
) -> DistributedOpResult:
    """手动指定模式: 已有单卡结果，追加通信建模。

    适合已经跑完单卡建模、只需要叠加通信的场景。
    """
    comm_result = network.estimate_collective(
        collective_type, parallel, collective_dim, data_bytes,
    )
    return DistributedOpResult(
        local_compute=local_result,
        communication=comm_result,
        overlap_ratio=overlap_ratio,
    )
