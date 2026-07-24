# 文件名: matmul_op.py
from ..util import *
# M, N, K, tb_m, tb_n, tb_k, L2_Cap, SM_Count, BYTE_per_num, stage_num, block_per_sm
import numpy as np
import math

def calculate_relu_resource_utilization(n,tb_n,thread_per_tb,l2_cap,sm_count, bytes_per_num, stage_num,arch):

    DDR_non_ideal_para=1.0 #Unmerged fragmented accesses will be scaled to sector (32bytes) as the smallest unit
    REG_spill_para=1.1 # just like the last one, something we don't know clearly about ncu

    l2_read_io=n*bytes_per_num
    l2_store_io=n*bytes_per_num
    l2_hit_rate=0
    ddr_io= l2_read_io*(1-l2_hit_rate)+l2_store_io

    if (arch=="a100"):
        #a100 with special two-part l2 cache structure
        l2_io=l2_read_io*(l2_hit_rate)+l2_read_io*(1-l2_hit_rate)*2+l2_store_io*2
    else:
        l2_io=l2_read_io+l2_store_io

    smem_footprint=0

    smem_l1_io=l2_read_io+l2_store_io

    reg_footprint=32
    reg_footprint=math.ceil(reg_footprint*REG_spill_para)

    overheads=(32 / thread_per_tb if thread_per_tb <= 32 else
          128 / thread_per_tb if thread_per_tb <= 128 else
          256 / thread_per_tb if thread_per_tb <= 256 else
          384 / thread_per_tb if thread_per_tb <= 384 else
          512 / thread_per_tb)
    
    #  2 for FFMAD and FADD occupy the same 1 clock period
    compute_flops=2*n*overheads

    return ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops
        
