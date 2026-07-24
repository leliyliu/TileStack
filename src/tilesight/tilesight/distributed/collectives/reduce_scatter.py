"""Reduce-Scatter 集合通信建模。

从 DeepStack reduce-scatter wrapper 适配。
"""
from __future__ import annotations

import math
import logging
import numpy as np
from numpy import uint64

from ..noc.traffic_matrix import TrafficMatrix
from ..noc.noc_topo import Hierarchy, get_extend_max_routes_with_traffic
from ..parallelism import ParallelScheme
from .base import CollectiveResult

log = logging.getLogger(__name__)


def reduce_scatter_rabenseifner(
    parallel: ParallelScheme,
    hierarchy: Hierarchy,
    dim: str,
    data_bytes: int,
) -> CollectiveResult:
    """Rabenseifner reduce-scatter: log(n) stages, 递减数据量。"""
    gsize = getattr(parallel, dim.lower())
    assert gsize & (gsize - 1) == 0, f"gsize must be power of 2, got {gsize}"

    stages = int(math.log2(gsize))
    total_hop = 0.0
    total_link = 0.0
    total_traffic = None

    tm = TrafficMatrix(parallel.world_size())
    for stage in range(stages):
        stride = 1 << stage
        bytes_per_iter = data_bytes / (1 << (stage + 1))
        pairs = []
        for i in range(gsize):
            partner = i ^ stride
            pairs.append([bytes_per_iter, i, partner])

        tm.add_intra_group_traffic_pair_bulk(
            dim, pairs,
            tp=parallel.tp, ep=parallel.ep, sp=parallel.sp,
            cp=parallel.cp, dp=parallel.dp, pp=parallel.pp,
        )
        hop, link, _, traffic = get_extend_max_routes_with_traffic(tm, hierarchy)
        total_hop += hop
        total_link += link
        total_traffic = traffic.copy() if total_traffic is None else total_traffic + traffic
        tm.reset()

    return CollectiveResult(total_hop, total_link, "rabenseifner", total_traffic)


def reduce_scatter_auto(
    parallel: ParallelScheme,
    hierarchy: Hierarchy,
    dim: str,
    data_bytes: int,
) -> CollectiveResult:
    """Auto-tuning wrapper (目前只有 rabenseifner)。"""
    data_bytes = int(uint64(data_bytes))
    return reduce_scatter_rabenseifner(parallel, hierarchy, dim, data_bytes)
