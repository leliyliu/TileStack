# tilesight.distributed.noc — 自包含 NoC 拓扑与流量建模
# 只依赖 numpy/标准库，不 import tilesight 其他模块
#
# 从 DeepStack NoC 模块移植

from .traffic_matrix import TrafficMatrix
from .noc_topo import (
    Topology, Hierarchy, Chain, Link,
    TopoKind, PortSpread,
    make_mesh_or_torus, make_ring, make_chain, make_switch, make_all2all,
    get_extend_max_routes_with_traffic,
)
from .noc_config_set import *
