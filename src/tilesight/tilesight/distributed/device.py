"""设备层级抽象：从单卡 Arch → 节点 NodeSpec → 集群 ClusterSpec。

依赖: tilesight.arch (Arch 类)
不被 fused_op_pipeline_wave 依赖（单向）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from tilesight.arch.arch_base import Arch


class MemoryLevel(IntEnum):
    """统一的内存层级枚举，从最快到最慢。"""
    REGISTER = 0
    SMEM = 1         # per-SM shared memory
    L2 = 2           # on-chip L2 cache
    HBM = 3          # device main memory (Arch 中的 "ddr")
    NVLINK = 4       # GPU-GPU 互联
    PCIE = 5         # GPU-CPU 或 GPU-NIC
    CPU_DDR = 6      # 主机内存 (stub)
    SSD = 7          # 持久存储 (stub)


@dataclass
class InterconnectSpec:
    """描述一种互联类型的带宽和延迟。

    bandwidth: 单条链路带宽 (bytes/sec)
    latency:   单跳延迟 (seconds)
    num_links: 并行链路数（如 NVLink 有 18 条 link）
    """
    bandwidth: float
    latency: float
    num_links: int = 1

    @property
    def total_bandwidth(self) -> float:
        """总聚合带宽 (bytes/sec)。"""
        return self.bandwidth * self.num_links


@dataclass
class NodeSpec:
    """一个节点 = 多块同构 GPU + 共享互联 + 可选 CPU 内存。

    device_arch: 单卡规格（复用现有 Arch 对象）
    num_devices: 每节点 GPU 数（如 DGX-H100 = 8）
    intra_node:  节点内 GPU 间互联（如 NVLink/NVSwitch）
    """
    device_arch: Arch
    num_devices: int
    intra_node: InterconnectSpec

    # CPU 内存 stub — 架构上预留，暂不建模
    cpu_mem_bandwidth: float = 0.0   # bytes/sec
    cpu_mem_capacity: int = 0        # bytes
    # SSD stub
    ssd_bandwidth: float = 0.0      # bytes/sec
    ssd_capacity: int = 0           # bytes

    @property
    def total_gpu_memory(self) -> int:
        """节点总 GPU 显存。"""
        return self.device_arch.ddr_capacity * self.num_devices


@dataclass
class ClusterSpec:
    """一个集群 = 多个同构节点 + 节点间互联。

    支持可选的多级互联（rack 内 vs rack 间）。
    """
    node: NodeSpec
    num_nodes: int
    inter_node: InterconnectSpec       # 节点间互联（如 InfiniBand）

    # 可选: 多级互联
    inter_rack: Optional[InterconnectSpec] = None
    nodes_per_rack: int = 1

    @property
    def total_devices(self) -> int:
        """集群总 GPU 数。"""
        return self.node.num_devices * self.num_nodes

    @property
    def total_nodes(self) -> int:
        return self.num_nodes

    @property
    def num_racks(self) -> int:
        if self.nodes_per_rack <= 0:
            return 1
        return (self.num_nodes + self.nodes_per_rack - 1) // self.nodes_per_rack
