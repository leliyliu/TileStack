# 文件名: matmul_op.py
from ..util import *
# M, N, K, tb_m, tb_n, tb_k, L2_Cap, SM_Count, BYTE_per_num, stage_num, block_per_sm
import numpy as np
import math

def calculate_add_bias_resource_utilization(op_shape,tb_shape,dim_threads,bytes_per_num, stage_num,arch,mem_levels):

    m,n=op_shape
    tb_m,tb_n=tb_shape
    m_thread,n_thread=dim_threads

    DDR_non_ideal_para=1.0 #Unmerged fragmented accesses will be scaled to sector (32bytes) as the smallest unit
    REG_spill_para=1.1 # just like the last one, something we don't know clearly about ncu

    thread_per_tb=m_thread*n_thread
    thread_m=math.ceil(tb_m/m_thread)
    thread_n=math.ceil(tb_n/n_thread)

    l2_hit_rate=1-(m*n+n)/(m*n+n*m/tb_m)
    gridM=math.ceil(m/tb_m)
    gridN=math.ceil(n/tb_n)
    # example:
    #     mem_levels = {
    #     'in1': '[1,1,1]', for ddr
    #     'in2': '[0,1,1]', for smem
    #     'out1': [0,0,1],  for reg
    # }

    in1_level = mem_levels['in1']
    in2_level = mem_levels['in2']
    out1_level = mem_levels['out1']

    l2_read_io=(m*n*in1_level[0]+n*m/tb_m*in2_level[0])*bytes_per_num
    l2_store_io=m*n*bytes_per_num*out1_level[0]

    ddr_io=l2_read_io*(1-l2_hit_rate)+l2_store_io
    ddr_io=ddr_io*DDR_non_ideal_para

    if (arch.core=="A100" or arch.core=="H100" or arch.core=="B200"):
        #a100 with special two-part l2 cache structure
        l2_io=l2_read_io*(l2_hit_rate)+l2_read_io*(1-l2_hit_rate)*2+l2_store_io*2
    else:
        l2_io=l2_read_io+l2_store_io

    smem_footprint=tb_n*bytes_per_num

    smem_l1_io=(m*n*in1_level[1]+n*m/tb_m*(in2_level[0]+in2_level[1])+m*n*out1_level[1])*bytes_per_num

    reg_footprint=math.ceil((thread_m*thread_n+thread_n+thread_m*thread_n)*REG_spill_para)

    overheads=(32 / thread_per_tb if thread_per_tb <= 32 else
          128 / thread_per_tb if thread_per_tb <= 128 else
          256 / thread_per_tb if thread_per_tb <= 256 else
          384 / thread_per_tb if thread_per_tb <= 384 else
          512 / thread_per_tb)
    
    compute_flops=2*m*n*overheads

    return ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops
        
