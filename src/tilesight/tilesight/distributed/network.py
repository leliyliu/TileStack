"""NetworkHierarchy: ClusterSpec → NoC Hierarchy 的薄封装。

依赖: distributed/noc/, distributed/device.py
不被 fused_op_pipeline_wave 依赖。
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from .device import ClusterSpec, InterconnectSpec
from .noc.noc_topo import (
    Hierarchy, Topology, TopoKind, PortSpread,
    make_switch, make_mesh_or_torus, make_ring,
    get_extend_max_routes_with_traffic,
)
from .noc.traffic_matrix import TrafficMatrix

log = logging.getLogger(__name__)


@dataclass
class CommResult:
    """集合通信建模结果。"""
    hop_time: float          # 跳延迟之和 (seconds)
    link_time: float         # 瓶颈链路时间 (seconds)
    total_time: float        # max(hop_time, link_time) 或两者之和，取决于路由实现
    collective_type: str = ""     # "all_reduce", "reduce_scatter", etc.
    algorithm: str = ""           # "ring", "recursive_doubling", etc.


class NetworkHierarchy:
    """将 ClusterSpec 自动转换为 NoC Hierarchy，提供通信估算 API。

    对于只有 1 个节点的场景，生成单层 Hierarchy（NVSwitch）。
    对于多节点场景，生成 2~3 层 Hierarchy（节点内 + 节点间 + 可选 rack 间）。

    用法:
        cluster = ClusterSpec(...)
        net = NetworkHierarchy(cluster)
        # 直接使用底层 NoC
        result = net.route_traffic(traffic_matrix)
        # 或者访问底层 Hierarchy
        h = net.hierarchy
    """

    def __init__(self, cluster: ClusterSpec,
                 port_spread: PortSpread = PortSpread.EVEN,
                 name: str = "auto"):
        self._cluster = cluster
        self._port_spread = port_spread
        self._name = name
        self._hierarchy = self._build_hierarchy()

    @property
    def cluster(self) -> ClusterSpec:
        return self._cluster

    @property
    def hierarchy(self) -> Hierarchy:
        """底层 NoC Hierarchy，高级用户可直接操作。"""
        return self._hierarchy

    @property
    def num_devices(self) -> int:
        return self._cluster.total_devices

    def _build_hierarchy(self) -> Hierarchy:
        """从 ClusterSpec 自动构建 NoC Hierarchy。

        策略:
        - 节点内: NVSwitch → Switch topology (num_devices per node)
        - 节点间: 根据 num_nodes 选择 Switch (≤16) 或 Mesh/Torus
        - Rack 间: 如果有 inter_rack，额外加一层
        """
        node = self._cluster.node
        layers = []

        # 节点间层 (如果多节点)
        if self._cluster.num_nodes > 1:
            inter = self._cluster.inter_node

            # 如果有多级互联 (rack 间)
            if self._cluster.inter_rack is not None and self._cluster.num_racks > 1:
                rack_inter = self._cluster.inter_rack
                # L3 (最外层): rack 间
                num_racks = self._cluster.num_racks
                l3 = _make_topology_for_count(
                    num_racks, rack_inter,
                    prefer_switch_threshold=16
                )
                layers.append(l3)

                # L2: rack 内节点间
                npr = self._cluster.nodes_per_rack
                l2 = _make_topology_for_count(npr, inter, prefer_switch_threshold=16)
                layers.append(l2)
            else:
                # 单层节点间
                l_inter = _make_topology_for_count(
                    self._cluster.num_nodes, inter,
                    prefer_switch_threshold=16
                )
                layers.append(l_inter)

        # 节点内层 (最内层): NVSwitch 抽象为 Switch
        intra = node.intra_node
        l_intra = make_switch(
            N=node.num_devices,
            hop_latency=intra.latency,
            link_bandwidth=intra.bandwidth,
            switch_center_in_bw=intra.total_bandwidth,
            switch_center_out_bw=intra.total_bandwidth,
        )
        layers.append(l_intra)

        name = self._name if self._name != "auto" else f"cluster_{self._cluster.total_devices}gpu"
        return Hierarchy(
            layers=layers,
            port_spread=self._port_spread,
            node_mapper=None,
            name=name,
        )

    def estimate_collective(
        self,
        collective_type: str,
        parallel: 'ParallelScheme',
        dim: str,
        data_bytes: int,
    ) -> CommResult:
        """高层 API: 估算指定集合通信的时间。

        collective_type: "all_reduce", "reduce_scatter", "all_gather", "all_to_all"
        dim: 并行维度 ("tp", "dp", "ep", ...)
        data_bytes: 每设备的数据量 (bytes)
        """
        from .collectives import all_reduce_auto, reduce_scatter_auto, all_gather_auto
        from .collectives.all_to_all import all_to_all_uniform

        dispatch = {
            "all_reduce": all_reduce_auto,
            "reduce_scatter": reduce_scatter_auto,
            "all_gather": all_gather_auto,
            "all_to_all": all_to_all_uniform,
        }
        fn = dispatch.get(collective_type)
        if fn is None:
            raise ValueError(f"Unknown collective type: {collective_type}. "
                           f"Supported: {list(dispatch.keys())}")

        result = fn(parallel, self._hierarchy, dim, data_bytes)
        return CommResult(
            hop_time=result.hop_time,
            link_time=result.link_time,
            total_time=result.total_time,
            collective_type=collective_type,
            algorithm=result.algorithm,
        )

    def route_traffic(self, tm: TrafficMatrix) -> CommResult:
        """将流量矩阵路由到网络，返回通信时间。"""
        hop_time, link_time, total_time, _traffic = get_extend_max_routes_with_traffic(
            tm, self._hierarchy
        )
        return CommResult(
            hop_time=hop_time,
            link_time=link_time,
            total_time=total_time,
        )

    def create_traffic_matrix(self) -> TrafficMatrix:
        """创建一个大小匹配的空 TrafficMatrix。"""
        return TrafficMatrix(N=self.num_devices)


def _make_topology_for_count(
    count: int,
    interconnect: InterconnectSpec,
    prefer_switch_threshold: int = 16,
) -> Topology:
    """根据节点数自动选择拓扑类型。

    ≤ threshold: 用 Switch (全交换)
    > threshold: 尝试 2D Torus (尽量方形)
    """
    if count <= prefer_switch_threshold:
        return make_switch(
            N=count,
            hop_latency=interconnect.latency,
            link_bandwidth=interconnect.bandwidth,
            switch_center_in_bw=interconnect.total_bandwidth,
            switch_center_out_bw=interconnect.total_bandwidth,
        )
    else:
        # 2D Torus: 找最接近 sqrt 的因数分解
        m, n = _factorize_near_square(count)
        return make_mesh_or_torus(
            M=m, N=n,
            kind=TopoKind.TORUS2D,
            hop_latency=interconnect.latency,
            link_bandwidth=interconnect.bandwidth,
        )


def _factorize_near_square(n: int) -> Tuple[int, int]:
    """将 n 分解为两个因数 (m, k)，使得 m*k=n 且 m≈k。"""
    best = (1, n)
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            best = (i, n // i)
    return best
