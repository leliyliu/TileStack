"""All-Gather 集合通信建模。

从 DeepStack all-gather wrapper 适配。
"""
from __future__ import annotations

import logging
import numpy as np
from numpy import uint64

from ..noc.traffic_matrix import TrafficMatrix
from ..noc.noc_topo import Hierarchy, get_extend_max_routes_with_traffic
from ..parallelism import ParallelScheme
from .base import CollectiveResult

log = logging.getLogger(__name__)


def all_gather_ring(
    parallel: ParallelScheme,
    hierarchy: Hierarchy,
    dim: str,
    data_bytes: int,
) -> CollectiveResult:
    """Ring all-gather: (n-1) stages, 每 stage 1/n 数据沿环传递。

    每个设备持有 data_bytes/gsize 的数据，最终所有设备拥有完整数据。
    """
    gsize = getattr(parallel, dim.lower())
    if gsize <= 1:
        return CollectiveResult(0.0, 0.0, "ring_trivial", None)

    stages = gsize - 1
    bytes_per_stage = data_bytes / gsize

    tm = TrafficMatrix(parallel.world_size())
    pairs = []
    for i in range(gsize):
        pairs.append([bytes_per_stage, i, (i + 1) % gsize])

    tm.add_intra_group_traffic_pair_bulk(
        dim, pairs,
        tp=parallel.tp, ep=parallel.ep, sp=parallel.sp,
        cp=parallel.cp, dp=parallel.dp, pp=parallel.pp,
    )
    hop, link, _, traffic = get_extend_max_routes_with_traffic(tm, hierarchy)
    return CollectiveResult(hop * stages, link * stages, "ring", traffic * stages)


def all_gather_all_to_all(
    parallel: ParallelScheme,
    hierarchy: Hierarchy,
    dim: str,
    data_bytes: int,
) -> CollectiveResult:
    """All-to-all 方式的 all-gather: 单 stage, 每设备向其余所有设备发送自己的分片。"""
    gsize = getattr(parallel, dim.lower())
    if gsize <= 1:
        return CollectiveResult(0.0, 0.0, "all_to_all_trivial", None)

    bytes_per_pair = data_bytes / gsize

    tm = TrafficMatrix(parallel.world_size())
    tm.add_intra_group_traffic(
        dim, bytes_per_pair,
        tp=parallel.tp, ep=parallel.ep, sp=parallel.sp,
        cp=parallel.cp, dp=parallel.dp, pp=parallel.pp,
    )
    hop, link, _, traffic = get_extend_max_routes_with_traffic(tm, hierarchy)
    return CollectiveResult(hop, link, "all_to_all", traffic)


def all_gather_auto(
    parallel: ParallelScheme,
    hierarchy: Hierarchy,
    dim: str,
    data_bytes: int,
) -> CollectiveResult:
    """Auto-tuning wrapper: 选最优算法。"""
    data_bytes = int(uint64(data_bytes))
    candidates = [
        all_gather_ring(parallel, hierarchy, dim, data_bytes),
        all_gather_all_to_all(parallel, hierarchy, dim, data_bytes),
    ]
    best = min(candidates, key=lambda r: r.total_time)
    log.info(
        "all_gather_auto: selected=%s (total=%.3e)",
        best.algorithm, best.total_time,
    )
    return best
