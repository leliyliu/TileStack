"""All-Reduce 集合通信建模：5 种算法 + auto-tuning wrapper。

从 DeepStack all-reduce wrapper 适配。
只依赖 distributed/noc/ 和 distributed/parallelism.py。
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


def _dim_size(parallel: ParallelScheme, dim: str) -> int:
    dim = dim.lower()
    return getattr(parallel, dim)


def all_reduce_recursive_doubling(
    parallel: ParallelScheme,
    hierarchy: Hierarchy,
    dim: str,
    data_bytes: int,
) -> CollectiveResult:
    """Recursive doubling: O(log n) stages, 每 stage 全量数据交换。"""
    gsize = _dim_size(parallel, dim)
    assert gsize & (gsize - 1) == 0, f"gsize must be power of 2, got {gsize}"

    stages = int(math.log2(gsize))
    total_hop = 0.0
    total_link = 0.0
    total_traffic = None

    tm = TrafficMatrix(parallel.world_size())
    for stage in range(stages):
        stride = 1 << stage
        pairs = []
        for i in range(gsize):
            partner = i ^ stride
            pairs.append([data_bytes, i, partner])

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

    return CollectiveResult(total_hop, total_link, "recursive_doubling", total_traffic)


def all_reduce_ring(
    parallel: ParallelScheme,
    hierarchy: Hierarchy,
    dim: str,
    data_bytes: int,
) -> CollectiveResult:
    """Ring all-reduce: 2*(n-1) stages, 每 stage 1/n 数据。"""
    gsize = _dim_size(parallel, dim)
    stages = (gsize - 1) * 2
    bytes_per_iter = data_bytes / gsize

    tm = TrafficMatrix(parallel.world_size())
    pairs = []
    for i in range(gsize):
        pairs.append([bytes_per_iter, i, (i + 1) % gsize])

    tm.add_intra_group_traffic_pair_bulk(
        dim, pairs,
        tp=parallel.tp, ep=parallel.ep, sp=parallel.sp,
        cp=parallel.cp, dp=parallel.dp, pp=parallel.pp,
    )
    hop, link, _, traffic = get_extend_max_routes_with_traffic(tm, hierarchy)
    return CollectiveResult(hop * stages, link * stages, "ring", traffic * stages)


def all_reduce_rabenseifner(
    parallel: ParallelScheme,
    hierarchy: Hierarchy,
    dim: str,
    data_bytes: int,
) -> CollectiveResult:
    """Rabenseifner: reduce-scatter + all-gather 组合。"""
    gsize = _dim_size(parallel, dim)
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

    # Rabenseifner = reduce-scatter + all-gather (对称), 乘 2
    return CollectiveResult(total_hop * 2, total_link * 2, "rabenseifner",
                            total_traffic * 2 if total_traffic is not None else None)


def all_reduce_double_tree(
    parallel: ParallelScheme,
    hierarchy: Hierarchy,
    dim: str,
    data_bytes: int,
) -> CollectiveResult:
    """双二叉树: A 树 (偶数叶→右根) + B 树 (奇数叶→左根), 先规约后广播。"""
    gsize = _dim_size(parallel, dim)
    assert gsize & (gsize - 1) == 0, f"gsize must be power of 2, got {gsize}"

    stages = int(math.log2(gsize))
    bytes_per_tree = data_bytes / 2

    # 构建 A 树 (偶数叶→右根)
    a_levels = _build_tree_levels(list(range(0, gsize, 2)), gsize - 1)
    # 构建 B 树 (奇数叶→左根)
    b_levels = _build_tree_levels(list(range(1, gsize, 2)), 0)
    assert len(a_levels) == stages and len(b_levels) == stages

    total_hop = 0.0
    total_link = 0.0
    total_traffic = None
    tm = TrafficMatrix(parallel.world_size())

    # 向上规约
    for lvl in range(stages):
        pairs = []
        for src, dst in a_levels[lvl]:
            pairs.append([bytes_per_tree, src, dst])
        for src, dst in b_levels[lvl]:
            pairs.append([bytes_per_tree, src, dst])
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

    # 向下广播 (反向)
    for lvl in range(stages - 1, -1, -1):
        pairs = []
        for src, dst in a_levels[lvl]:
            pairs.append([bytes_per_tree, dst, src])  # 反向
        for src, dst in b_levels[lvl]:
            pairs.append([bytes_per_tree, dst, src])
        tm.add_intra_group_traffic_pair_bulk(
            dim, pairs,
            tp=parallel.tp, ep=parallel.ep, sp=parallel.sp,
            cp=parallel.cp, dp=parallel.dp, pp=parallel.pp,
        )
        hop, link, _, traffic = get_extend_max_routes_with_traffic(tm, hierarchy)
        total_hop += hop
        total_link += link
        total_traffic = traffic if total_traffic is None else total_traffic + traffic
        tm.reset()

    return CollectiveResult(total_hop, total_link, "double_tree", total_traffic)


def _build_tree_levels(leaves: list, root: int) -> list:
    """构建二叉树层级: 叶子列表 → root, 返回 [[src,dst], ...] per level."""
    levels = []
    current = leaves
    while len(current) >= 2:
        next_nodes = []
        pairs = []
        for j in range(0, len(current), 2):
            left, right = current[j], current[j + 1]
            parent = (left + right) // 2
            pairs.append([left, parent])
            pairs.append([right, parent])
            next_nodes.append(parent)
        levels.append(pairs)
        current = next_nodes
    if len(current) == 1:
        levels.append([[current[0], root]])
    return levels


def all_reduce_all_to_all(
    parallel: ParallelScheme,
    hierarchy: Hierarchy,
    dim: str,
    data_bytes: int,
) -> CollectiveResult:
    """All-to-all: 单 stage, 全量全对全。"""
    tm = TrafficMatrix(parallel.world_size())
    tm.add_intra_group_traffic(
        dim, data_bytes,
        tp=parallel.tp, ep=parallel.ep, sp=parallel.sp,
        cp=parallel.cp, dp=parallel.dp, pp=parallel.pp,
    )
    hop, link, _, traffic = get_extend_max_routes_with_traffic(tm, hierarchy)
    return CollectiveResult(hop, link, "all_to_all", traffic)


def all_reduce_auto(
    parallel: ParallelScheme,
    hierarchy: Hierarchy,
    dim: str,
    data_bytes: int,
) -> CollectiveResult:
    """Auto-tuning wrapper: 评估所有算法，选最优。"""
    data_bytes = int(uint64(data_bytes))
    gsize = _dim_size(parallel, dim)

    candidates = []

    # 所有算法都试 (power-of-2 限制的跳过)
    is_pow2 = gsize & (gsize - 1) == 0

    if is_pow2:
        candidates.append(all_reduce_recursive_doubling(parallel, hierarchy, dim, data_bytes))
        candidates.append(all_reduce_rabenseifner(parallel, hierarchy, dim, data_bytes))
        candidates.append(all_reduce_double_tree(parallel, hierarchy, dim, data_bytes))

    candidates.append(all_reduce_ring(parallel, hierarchy, dim, data_bytes))
    candidates.append(all_reduce_all_to_all(parallel, hierarchy, dim, data_bytes))

    best = min(candidates, key=lambda r: r.total_time)

    log.info(
        "all_reduce_auto: selected=%s (total=%.3e), candidates: %s",
        best.algorithm,
        best.total_time,
        ", ".join(f"{c.algorithm}({c.total_time:.3e})" for c in candidates),
    )
    return best
