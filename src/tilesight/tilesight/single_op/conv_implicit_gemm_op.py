# 文件名: matmul_op.py
from ..util import *
# M, N, K, tb_m, tb_n, tb_k, L2_Cap, SM_Count, BYTE_per_num, stage_num, block_per_sm
import numpy as np
import math

# def calculate_conv_implicit_gemm_resource_utilization(m,n,k,tb_m,tb_n,tb_k,wp_m,wp_n,wp_k,l2_cap,sm_count, bytes_per_num, stage_num,arch):
def calculate_conv_implicit_gemm_resource_utilization(conv_n,conv_f,conv_h,conv_w,conv_c,conv_kh,conv_kw,conv_s,conv_d,conv_p,tb_m,tb_n,tb_k,wp_m,wp_n,wp_k,l2_cap,sm_count, bytes_per_num, stage_num,arch):

    # m,n,k,tb_m,tb_n,tb_k,wp_m,wp_n,wp_k,l2_cap,sm_count, bytes_per_num, stage_num,arch
    m=conv_n*conv_h*conv_w
    n=conv_f
    k=conv_kh*conv_kw*conv_c

    DDR_non_ideal_para=1.1 #Unmerged fragmented accesses will be scaled to sector (32bytes) as the smallest unit
    REG_spill_para=1.1 # just like the last one, something we don't know clearly about ncu

    compute_flops=2*m*n*k
    # implicit_gemm_l2_hitrate_reuse_distance(M, N, K, tb_m, tb_n, tb_k, L2_Cap, SM_Count, BYTE_per_num, CONV_C, CONV_KH, CONV_KW):
    l2_hit_rate=implicit_gemm_l2_hitrate_reuse_distance(m, n, k, tb_m, tb_n, tb_k, l2_cap, sm_count, bytes_per_num, conv_c, conv_kh, conv_kw)
    
    gridM=math.ceil(m/tb_m)
    gridN=math.ceil(n/tb_n)

    l2_read_io=gridM*gridN*k*(tb_m+tb_n)*bytes_per_num
    l2_store_io=gridM*gridN*(tb_m*tb_n)*bytes_per_num

    ddr_io=l2_read_io*(1-l2_hit_rate)+l2_store_io
    ddr_io=ddr_io*DDR_non_ideal_para

    active_warp_per_tb=(tb_m/wp_m)*(tb_n/wp_n)

    if (arch=="a100"):
        #a100 with special two-part l2 cache structure
        l2_io=l2_read_io*(l2_hit_rate)+l2_read_io*(1-l2_hit_rate)*2+l2_store_io*2
    else:
        l2_io=l2_read_io+l2_store_io

    smem_footprint=(tb_m+tb_n)*tb_k*bytes_per_num*stage_num

    if (arch=="a100"):
        reg_footprint = (math.ceil(wp_m * wp_n / 32 / (4 / bytes_per_num)) + math.ceil(wp_m * 32 / 32 / (4 / bytes_per_num)) + math.ceil(wp_n * 32 / 32 / (4 / bytes_per_num)))
        reg_footprint=reg_footprint*REG_spill_para
        reg_footprint=math.ceil(reg_footprint)
    else:#with dup
        reg_footprint = (math.ceil(wp_m * wp_n / 32 / (4 / bytes_per_num)) + 2*math.ceil(wp_m * 32 / 32 / (4 / bytes_per_num)) + 2*math.ceil(wp_n * 32 / 32 / (4 / bytes_per_num)))
        reg_footprint=reg_footprint*REG_spill_para
        reg_footprint=math.ceil(reg_footprint)


    if (arch=="a100"):
        #somehow 0.5* ldgsts
        smem_io=0.5*l2_read_io
        #shared load
        smem_io+=l2_read_io*active_warp_per_tb*(wp_m+wp_n)/(tb_m+tb_n)
        #epilogue, shared store
        smem_io+=l2_store_io
        #epilogue, shared load
        smem_io+=l2_store_io
        #store global
        l1_io=l2_store_io
        #add together
        smem_l1_io=l1_io+smem_io
    else:#noldgsts
        #load global
        l1_io=l2_read_io
        #store shared
        smem_io=l2_read_io
        #shared load
        smem_io+=l2_read_io*active_warp_per_tb*(wp_m+wp_n)/(tb_m+tb_n)
        #epilogue, shared store
        smem_io+=l2_store_io
        #epilogue, shared load
        smem_io+=l2_store_io
        #store global
        l1_io+=l2_store_io
        #add together
        smem_l1_io=l1_io+smem_io
    

    return ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops
        
