"""分布式 Tile 分配：将全局 grid 映射到单卡 local grid + 推断所需 collective。

依赖: parallelism.py (纯数据结构)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .parallelism import ParallelScheme


# ---- 常见并行模式下 op 输出后需要的 collective ----
# 格式: (并行维度, 切分方式) → (collective_type, 数据量系数说明)
_TP_COLLECTIVE_RULES = {
    # TP column-parallel GEMM (切 N): 输出是 partial sum → all-reduce
    "column_parallel": "all_reduce",
    # TP row-parallel GEMM (切 K): 输出是 partial sum → all-reduce
    "row_parallel": "all_reduce",
    # TP 切 N + Sequence Parallel: 输出做 reduce-scatter → SP 维度
    "column_parallel_sp": "reduce_scatter",
    # SP → TP 过渡: 需要 all-gather
    "sp_to_tp": "all_gather",
}


@dataclass
class DistributedTileMap:
    """描述一个 op 的 tile 如何分布到多设备。

    dim_mapping: op 的空间维度到并行维度的映射
        例: {"N": "tp"} 表示 output 的 N 维按 tp 切分
             {"M": "dp"} 表示 batch/M 维按 dp 切分
    collective_after: op 执行后需要的 collective (可选)
        例: "all_reduce" (TP GEMM 的 partial sum 归约)
             "reduce_scatter" (TP + SP 场景)
        如果为 None，由 infer_collective() 根据 dim_mapping 推断
    collective_dim: collective 在哪个并行维度上执行
        例: "tp" 表示 tp group 内做 all-reduce
    collective_data_bytes: collective 的数据量 (bytes)
        如果为 None，由 infer_collective_bytes() 根据 output shape 推断
    """
    parallel: ParallelScheme
    dim_mapping: Dict[str, str] = field(default_factory=dict)

    # 通信描述 (可手动指定或自动推断)
    collective_after: Optional[str] = None
    collective_dim: Optional[str] = None
    collective_data_bytes: Optional[int] = None

    def local_grids(
        self,
        global_grids: Tuple[int, ...],
        grid_names: Tuple[str, ...],
    ) -> Tuple[int, ...]:
        """全局 grid → 单卡 local grid。

        global_grids: (gridM, gridN) 等
        grid_names: ("M", "N") 等，与 global_grids 一一对应

        被映射到某并行维度的 grid 维度会被除以该维度的 size。
        例: gridN=128, dim_mapping={"N": "tp"}, tp=8 → local gridN=16
        """
        result = []
        for g, name in zip(global_grids, grid_names):
            dim = self.dim_mapping.get(name)
            if dim is not None:
                divisor = self.parallel.group_size(dim)
                result.append(math.ceil(g / divisor))
            else:
                result.append(g)
        return tuple(result)

    def local_shape(
        self,
        global_shape: Dict[str, int],
    ) -> Dict[str, int]:
        """全局 op shape → 单卡 local shape。

        global_shape: {"M": 4096, "N": 4096, "K": 4096}
        """
        result = {}
        for name, size in global_shape.items():
            dim = self.dim_mapping.get(name)
            if dim is not None:
                divisor = self.parallel.group_size(dim)
                result[name] = math.ceil(size / divisor)
            else:
                result[name] = size
        return result

    def requires_collective(self) -> Optional[Tuple[str, str]]:
        """该 op 之后需要什么 collective?

        Returns: (collective_type, parallel_dim) 或 None
        """
        if self.collective_after is not None and self.collective_dim is not None:
            return (self.collective_after, self.collective_dim)
        return None


def make_tp_column_parallel(parallel: ParallelScheme, output_bytes: int) -> DistributedTileMap:
    """TP Column Parallel GEMM: 切 output N 维, 之后 all-reduce。

    Y = X @ W, W 按列切 (N 维), Y 是 partial sum → all-reduce
    """
    return DistributedTileMap(
        parallel=parallel,
        dim_mapping={"N": "tp"},
        collective_after="all_reduce",
        collective_dim="tp",
        collective_data_bytes=output_bytes,
    )


def make_tp_row_parallel(parallel: ParallelScheme, output_bytes: int) -> DistributedTileMap:
    """TP Row Parallel GEMM: 切 input K 维, 之后 all-reduce。

    Y = X @ W, W 按行切 (K 维), Y 是 partial sum → all-reduce
    """
    return DistributedTileMap(
        parallel=parallel,
        dim_mapping={"K": "tp"},
        collective_after="all_reduce",
        collective_dim="tp",
        collective_data_bytes=output_bytes,
    )


def make_tp_column_parallel_sp(parallel: ParallelScheme, output_bytes: int) -> DistributedTileMap:
    """TP + SP: Column Parallel GEMM → reduce-scatter 到 SP 维度。"""
    return DistributedTileMap(
        parallel=parallel,
        dim_mapping={"N": "tp"},
        collective_after="reduce_scatter",
        collective_dim="tp",
        collective_data_bytes=output_bytes,
    )


def make_sp_to_tp_all_gather(parallel: ParallelScheme, input_bytes: int) -> DistributedTileMap:
    """SP → TP 过渡: all-gather 重组完整 activation。"""
    return DistributedTileMap(
        parallel=parallel,
        dim_mapping={"M": "tp"},  # SP 在 sequence 维度切分
        collective_after="all_gather",
        collective_dim="tp",
        collective_data_bytes=input_bytes,
    )


def make_data_parallel(parallel: ParallelScheme, gradient_bytes: int) -> DistributedTileMap:
    """Data Parallel: batch 维切分, 之后 all-reduce 梯度。"""
    return DistributedTileMap(
        parallel=parallel,
        dim_mapping={"M": "dp"},
        collective_after="all_reduce",
        collective_dim="dp",
        collective_data_bytes=gradient_bytes,
    )
