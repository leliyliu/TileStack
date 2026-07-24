# input is mod, device_arch
from tilelang.tools.tilesight.arch import *
from tilelang.tools.tilesight.utils import *

from tilelang import tvm
import numpy as np
from dataclasses import dataclass
from tvm.tir.stmt_functor import ir_transform
# from tvm.tir.analysis.analysis import estimate_tir_flops
import logging
from typing import Optional
from .estimate_flops import estimate_flops_from_mod

def analyze_tir_after_vectorize(mod:tvm.ir.module.IRModule, device:Arch):
    # get the tir of the mod
    # tir = mod.script()
    # get the tir of the mod after vectorize
    # tir_after_vectorize = tir.vectorize()
    # get the tir of the mod after vectorize
    # print(f"mod after vectorize: {mod}")
    # flops = estimate_flops_from_mod(mod)
    # print(f"flops: {flops}")
    pass