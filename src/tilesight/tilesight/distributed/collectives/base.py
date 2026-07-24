"""集合通信基础类型。

只依赖标准库。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CollectiveResult:
    """单次集合通信建模结果。"""
    hop_time: float         # 跳延迟之和 (seconds)
    link_time: float        # 瓶颈链路时间 (seconds)
    algorithm: str          # 算法名称
    traffic: Optional[np.ndarray] = None  # 路由后的流量矩阵

    @property
    def total_time(self) -> float:
        return self.hop_time + self.link_time
