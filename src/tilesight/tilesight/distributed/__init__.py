# tilesight.distributed — 分布式多卡/多节点建模扩展
#
# 依赖方向（严格单向）:
#   arch/ ← distributed/noc/ ← distributed/collectives/
#                                      ↑
#   arch/ ← distributed/device.py ← distributed/network.py
#                                      ↑
#   fused_op_pipeline_wave/ ← distributed/distributed_op.py

from .device import MemoryLevel, InterconnectSpec, NodeSpec, ClusterSpec
from .parallelism import ParallelScheme
from .network import NetworkHierarchy, CommResult
from .tile_distribution import (
    DistributedTileMap,
    make_tp_column_parallel,
    make_tp_row_parallel,
    make_data_parallel,
)
from .distributed_op import DistributedOpResult, model_distributed_op, model_distributed_op_manual
from .distributed_model import DistributedLayerResult, model_distributed_layer
