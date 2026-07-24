"""预定义集群配置：DGX-H100, DGX-B200 等。

依赖: distributed/device.py, tilesight/arch/
"""
from __future__ import annotations

from .device import ClusterSpec, NodeSpec, InterconnectSpec


def make_dgx_h100_node() -> NodeSpec:
    """DGX H100: 8×H100 SXM, NVSwitch 全连接。

    NVLink 4th gen: 每卡 18 条 link, 每条 ~25 GB/s 单向 → 900 GB/s 双向聚合。
    NVSwitch: 全双工交换，等效每端口 ~450 GB/s 单向。
    """
    from tilesight.arch.h100_sxm import H100_SXM
    arch = H100_SXM()

    nvlink = InterconnectSpec(
        bandwidth=25e9,       # 25 GB/s per link (单向)
        latency=5e-6,         # ~5 μs
        num_links=18,         # 每卡 18 条 NVLink
    )
    return NodeSpec(
        device_arch=arch,
        num_devices=8,
        intra_node=nvlink,
        cpu_mem_bandwidth=204.8e9,   # DDR5-4800 8ch ≈ 204.8 GB/s
        cpu_mem_capacity=2 * 1024**4, # 2 TB (典型)
    )


def make_dgx_h100_cluster(num_nodes: int = 1) -> ClusterSpec:
    """DGX H100 集群: 8×H100 per node, InfiniBand HDR/NDR 互联。

    NDR InfiniBand: 400 Gbps = 50 GB/s per port, 单 rail。
    8-rail 配置 (每卡一条 NIC): 聚合 400 GB/s。
    """
    node = make_dgx_h100_node()
    ib = InterconnectSpec(
        bandwidth=50e9,       # 50 GB/s per rail (NDR 400G)
        latency=1e-6,         # ~1 μs
        num_links=8,          # 8-rail (每卡一条)
    )
    return ClusterSpec(node=node, num_nodes=num_nodes, inter_node=ib)


def make_dgx_b200_node() -> NodeSpec:
    """DGX B200: 8×B200, NVLink 5th gen + NVSwitch。

    NVLink 5th gen: 每卡 18 条 link, 每条 ~50 GB/s 单向 → 1800 GB/s 双向。
    """
    from tilesight.arch.b200 import B200
    arch = B200()

    nvlink = InterconnectSpec(
        bandwidth=50e9,       # 50 GB/s per link (NVLink 5)
        latency=3e-6,         # ~3 μs (略优于 NVLink 4)
        num_links=18,
    )
    return NodeSpec(
        device_arch=arch,
        num_devices=8,
        intra_node=nvlink,
        cpu_mem_bandwidth=307.2e9,   # host CPU DDR5
        cpu_mem_capacity=2 * 1024**4,
    )


def make_dgx_b200_cluster(num_nodes: int = 1) -> ClusterSpec:
    """DGX B200 集群: NVLink 5 intra-node, IB NDR inter-node。"""
    node = make_dgx_b200_node()
    ib = InterconnectSpec(
        bandwidth=50e9,
        latency=1e-6,
        num_links=8,
    )
    return ClusterSpec(node=node, num_nodes=num_nodes, inter_node=ib)


def make_rtx_pro_6000_pcie_node_theoretical(num_devices: int = 2) -> NodeSpec:
    """Un-calibrated PCIe Gen 5 x16 theoretical-peak node spec (tests / debug).

    Use only when a measured peer-to-peer bandwidth is genuinely
    unavailable (e.g., dry-run smoke tests on hosts without the GPU).
    """
    from tilesight.arch.rtx_pro_6000_pcie import (
        RTXPro6000PCIe,
        PCIE_GEN5_X16_UNI_BW,
        PCIE_GEN5_X16_LATENCY_S,
    )
    arch = RTXPro6000PCIe()
    pcie = InterconnectSpec(
        bandwidth=float(PCIE_GEN5_X16_UNI_BW),
        latency=PCIE_GEN5_X16_LATENCY_S,
        num_links=1,
    )
    return NodeSpec(
        device_arch=arch,
        num_devices=num_devices,
        intra_node=pcie,
        cpu_mem_bandwidth=128e9,
        cpu_mem_capacity=256 * 1024**3,
    )


def make_rtx_pro_6000_pcie_cluster_theoretical(num_devices: int = 2) -> ClusterSpec:
    """Un-calibrated PCIe Gen 5 theoretical-peak cluster (tests / debug)."""
    node = make_rtx_pro_6000_pcie_node_theoretical(num_devices=num_devices)
    no_inter_node = InterconnectSpec(bandwidth=0.0, latency=0.0, num_links=0)
    return ClusterSpec(node=node, num_nodes=1, inter_node=no_inter_node)
