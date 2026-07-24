"""并行策略描述：TP/EP/SP/CP/DP/PP 维度。

与 DeepStack 的 ParallelScheme 兼容但独立——
不依赖 tilesight 其他模块（纯数据结构）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ParallelScheme:
    """描述一组并行维度的大小。

    维度排列顺序（从最内到最外，影响设备编号分组）：
      TP → EP → SP → CP → DP → PP
    world_size = tp * ep * sp * cp * dp * pp
    """
    tp: int = 1       # Tensor Parallel
    ep: int = 1       # Expert Parallel
    sp: int = 1       # Sequence Parallel
    cp: int = 1       # Context Parallel
    dp: int = 1       # Data Parallel
    pp: int = 1       # Pipeline Parallel
    fsdp: bool = False  # FSDP 是 DP 的可选特性

    # EP 二级分片 (可选, 用于 MOE)
    ep1: Optional[int] = None   # ep1 sharding batch
    ep2: Optional[int] = None   # ep2 sharding seq

    def world_size(self) -> int:
        return self.tp * self.ep * self.sp * self.cp * self.dp * self.pp

    def as_dict(self) -> Dict[str, int]:
        return {
            "tp": self.tp, "ep": self.ep, "sp": self.sp,
            "cp": self.cp, "dp": self.dp, "pp": self.pp,
            "fsdp": self.fsdp,
        }

    # ---- 设备分组工具 ----

    def group_size(self, dim: str) -> int:
        """返回指定并行维度的 group size。"""
        return getattr(self, dim)

    def stride(self, dim: str) -> int:
        """返回指定并行维度在 world rank 编号中的 stride。

        维度从内到外: tp, ep, sp, cp, dp, pp
        stride(tp) = 1
        stride(ep) = tp
        stride(sp) = tp * ep
        ...
        """
        order = ["tp", "ep", "sp", "cp", "dp", "pp"]
        s = 1
        for d in order:
            if d == dim:
                return s
            s *= getattr(self, d)
        raise ValueError(f"Unknown dimension: {dim}")

    def ranks_in_group(self, dim: str, rank: int) -> list:
        """返回 rank 所在的指定维度 group 中的所有 rank。"""
        sz = self.group_size(dim)
        st = self.stride(dim)
        # rank 在该维度上的 index
        dim_idx = (rank // st) % sz
        # group 的 "base" = rank 去掉该维度的贡献
        base = rank - dim_idx * st
        return [base + i * st for i in range(sz)]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParallelScheme):
            return False
        return (self.tp == other.tp and self.ep == other.ep
                and self.sp == other.sp and self.cp == other.cp
                and self.dp == other.dp and self.pp == other.pp
                and self.fsdp == other.fsdp)

    def __hash__(self) -> int:
        return hash((self.tp, self.ep, self.sp, self.cp, self.dp, self.pp, self.fsdp))

    def __str__(self) -> str:
        return (f"tp={self.tp}, ep={self.ep}, sp={self.sp}, "
                f"cp={self.cp}, dp={self.dp}, pp={self.pp}, fsdp={self.fsdp}")

    def __repr__(self) -> str:
        return f"ParallelScheme({self})"
