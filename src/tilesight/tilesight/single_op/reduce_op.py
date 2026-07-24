# 文件名: reduce_op.py
import numpy as np

def calculate_reduce_resource_utilization(input_size, tb_tiling, warp_tiling, output_size, bytes_per_num):
    ddr_io = (input_size + output_size ) * bytes_per_num
    l2_hit_rate = 0
    l2_io = ddr_io
    smem_footprint = np.prod(tb_tiling) * bytes_per_num
    smem_l1_io = ddr_io
    reg_footprint = np.prod(warp_tiling) * bytes_per_num / 32
    compute_flops = 2 * input_size

    return ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops

