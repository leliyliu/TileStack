"""All-to-All 集合通信建模 (含 Expert Parallel 专用路径)。

从 DeepStack all-to-all wrapper 适配。
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


def all_to_all_uniform(
    parallel: ParallelScheme,
    hierarchy: Hierarchy,
    dim: str,
    data_bytes: int,
) -> CollectiveResult:
    """均匀 all-to-all: 每设备向该维度所有其他设备发送等量数据。"""
    gsize = getattr(parallel, dim.lower())
    if gsize <= 1:
        return CollectiveResult(0.0, 0.0, "uniform_trivial", None)

    tm = TrafficMatrix(parallel.world_size())
    tm.add_intra_group_traffic(
        dim, data_bytes,
        tp=parallel.tp, ep=parallel.ep, sp=parallel.sp,
        cp=parallel.cp, dp=parallel.dp, pp=parallel.pp,
    )
    hop, link, _, traffic = get_extend_max_routes_with_traffic(tm, hierarchy)
    return CollectiveResult(hop, link, "uniform", traffic)


def ep_all_to_all(
    parallel: ParallelScheme,
    hierarchy: Hierarchy,
    bytes_each_token: int,
    routing_array: np.ndarray,
    bs: int,
    seq: int,
    num_routed_experts: int,
    num_activated_experts: int,
    imbalance_overhead_ratio: float = 1.0,
) -> CollectiveResult:
    """Expert Parallel all-to-all: 基于 MOE routing trace 的精确流量建模。

    两条路径:
    - 大 batch (bs*seq >= 8192): 估算模式
    - 小 batch: trace-based 向量化精确建模
    """
    if parallel.ep <= 1:
        return CollectiveResult(0.0, 0.0, "ep_trivial", None)

    tm = TrafficMatrix(parallel.world_size())
    total_tokens = bs * seq

    # 路径 1: 估算模式 (大 batch)
    if total_tokens >= 8192:
        bytes_per_pair = (
            math.ceil(total_tokens / parallel.dp / parallel.sp / parallel.ep)
            * num_activated_experts / parallel.ep
            * bytes_each_token * imbalance_overhead_ratio
        )
        tm.add_intra_group_traffic(
            "ep", bytes_per_pair,
            tp=parallel.tp, ep=parallel.ep, sp=parallel.sp,
            cp=parallel.cp, dp=parallel.dp, pp=parallel.pp,
        )
        hop, link, _, traffic = get_extend_max_routes_with_traffic(tm, hierarchy)
        return CollectiveResult(hop, link, "ep_estimation", traffic)

    # 路径 2: trace-based 向量化精确建模
    if routing_array.ndim >= 2:
        routing_array = routing_array.reshape(-1, routing_array.shape[-1])
    else:
        raise ValueError("routing_array 维度应 >= 2")

    tokens_available, _ = routing_array.shape
    assert tokens_available >= total_tokens, "routing_array 行数不足"

    group_size = parallel.ep
    ep_groups = math.ceil(parallel.world_size() / group_size)
    groups_per_data = parallel.dp * parallel.sp
    tokens_each_ep_group = math.ceil(total_tokens / groups_per_data)

    # 预计算全局 Destination Rank 矩阵
    active_routing = routing_array[:total_tokens, :num_activated_experts]
    global_dst_ranks = (active_routing * group_size) // num_routed_experts
    np.clip(global_dst_ranks, 0, group_size - 1, out=global_dst_ranks)

    cols_count = num_activated_experts

    for ep_group_index in range(ep_groups):
        data_slot = ep_group_index % groups_per_data
        group_token_start = data_slot * tokens_each_ep_group
        group_token_end = min(group_token_start + tokens_each_ep_group, total_tokens)

        if group_token_start >= group_token_end:
            continue

        local_dst_ranks = global_dst_ranks[group_token_start:group_token_end, :]
        num_local_tokens = local_dst_ranks.shape[0]

        # 构建去重后的流量掩码
        row_indices = np.repeat(np.arange(num_local_tokens), cols_count)
        col_indices = local_dst_ranks.flatten()
        traffic_mask = np.zeros((num_local_tokens, group_size), dtype=bool)
        traffic_mask[row_indices, col_indices] = True

        # 按 Source Rank 切片并聚合
        base = num_local_tokens // group_size
        rem = num_local_tokens % group_size
        current_row_ptr = 0
        group_triples = []

        for src_index in range(group_size):
            count = base + (1 if src_index < rem else 0)
            if count == 0:
                continue

            src_mask = traffic_mask[current_row_ptr:current_row_ptr + count, :]
            token_counts = src_mask.sum(axis=0)
            token_counts[src_index] = 0  # 排除自环

            dst_indices = np.nonzero(token_counts)[0]
            for dst in dst_indices:
                vol_bytes = int(token_counts[dst]) * bytes_each_token
                group_triples.append([vol_bytes, src_index, int(dst)])

            current_row_ptr += count

        if group_triples:
            tm.add_intra_group_traffic_pair_at_group_bulk(
                "ep",
                triples=group_triples,
                group_index=ep_group_index,
                tp=parallel.tp, ep=parallel.ep, sp=parallel.sp,
                cp=parallel.cp, dp=parallel.dp, pp=parallel.pp,
            )

    hop, link, _, traffic = get_extend_max_routes_with_traffic(tm, hierarchy)
    return CollectiveResult(hop, link, "ep_trace", traffic)
