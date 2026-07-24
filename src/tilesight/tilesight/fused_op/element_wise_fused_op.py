# 文件名: element_wise_op.py
import numpy as np
import math

def calculate_elementwise_resource_utilization(input_size,output_size, tb_tiling, thread_nums, bytes_per_num,mem_levels):
    
    in1_level = mem_levels['in1']
    out1_level = mem_levels['out1']
    
    ddr_io =  bytes_per_num *(input_size*in1_level[0] + output_size*out1_level[0])

    l2_hit_rate = 0
    l2_io = bytes_per_num *(input_size*in1_level[0] + output_size*out1_level[0])
    smem_footprint = 0  # 在原始代码中，smem_footprint 的计算被注释掉了
    smem_l1_io = bytes_per_num *(input_size*in1_level[1] + output_size*out1_level[1])
    reg_footprint = (1+output_size/input_size)*np.prod(tb_tiling)/(np.prod(thread_nums)) * bytes_per_num/4
    compute_flops = 2 * input_size

    return ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops

