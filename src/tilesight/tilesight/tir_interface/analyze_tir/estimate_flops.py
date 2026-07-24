

from tilelang import tvm
from tvm import tir
from tvm.tir.stmt_functor import ir_transform
from collections import defaultdict

# from tilelang import tvm
# from tvm import tir
from tvm.ir import IRModule
from tvm.tir import stmt_functor
from tvm.arith import Analyzer
# from collections import defaultdict
import math

class TResult:
    def __init__(self):
        self.data = defaultdict(float)

    def add(self, dtype):
        self.data[str(dtype)] += 1

    def __iadd__(self, other):
        for k, v in other.data.items():
            self.data[k] += v
        return self

    def total(self):
        return sum(self.data.values())


def is_float_dtype(dtype):
    return "float" in str(dtype)


def count_flops_expr(expr):
    """Recursively count floating-point operations in an expression."""
    res = TResult()

    # print(f"[DEBUG] Visiting: {type(expr)} - {expr}")

    if isinstance(expr, (tir.Add, tir.Sub, tir.Mul, tir.Div, tir.FloorDiv, tir.FloorMod, tir.Mod, tir.Min, tir.Max)):
        if is_float_dtype(expr.dtype):
            res.add(expr.dtype)
        res += count_flops_expr(expr.a)
        res += count_flops_expr(expr.b)
    elif isinstance(expr, tir.Select):
        res += count_flops_expr(expr.condition)
        res += count_flops_expr(expr.true_value)
        res += count_flops_expr(expr.false_value)
        if is_float_dtype(expr.dtype):
            res.add(expr.dtype)
    elif isinstance(expr, tir.Cast):
        res += count_flops_expr(expr.value)
    elif isinstance(expr, tir.Call):
        for arg in expr.args: 
            res += count_flops_expr(arg)
    elif isinstance(expr, tir.Load):
        res += count_flops_expr(expr.index)
    elif isinstance(expr, tir.BufferLoad):
        for idx in expr.indices:
            res += count_flops_expr(idx)
    elif isinstance(expr, tir.Ramp):
        res += count_flops_expr(expr.base)
        res += count_flops_expr(expr.stride)
    elif isinstance(expr, tir.Broadcast):
        res += count_flops_expr(expr.value)
    return res


def estimate_flops_from_mod(mod: IRModule) -> float:
    total_result = TResult()

    def post_expr(expr):
        print(f"[DEBUG] Visiting: {type(expr)} - {expr}")
        # Only interested in expressions (PrimExpr)
        if isinstance(expr, tir.PrimExpr):
            total_result += count_flops_expr(expr)
        return expr

    def noop_stmt(stmt):
        print(f"[DEBUG] Visiting: {type(stmt)} - {stmt}")
        return stmt

    for gvar, func in mod.functions.items():
        # print(f"[DEBUG] Visiting: {type(func)} - {func}")
        # print(f"[DEBUG] func.body: {func.body}")
        if isinstance(func, tir.PrimFunc):
            print(f"[DEBUG] Visiting PrimFunc: {type(func)} - {func}")
            ir_transform(func.body, preorder=noop_stmt, postorder=post_expr)

    return total_result.total()