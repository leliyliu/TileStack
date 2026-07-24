# 文件名: matmul_op.py
from tilesight.util import *
from tilesight.arch import uses_gpu_resource_model
# from ..util import *
# M, N, K, tb_m, tb_n, tb_k, L2_Cap, SM_Count, BYTE_per_num, stage_num, block_per_sm
import numpy as np
import math

def calculate_matmul_resource_utilization(op_shape,tb_shape,wp_shape,bytes_per_num, stage_num,arch,mem_levels,row_panel):
    m,n,k=op_shape
    tb_m,tb_n,tb_k=tb_shape
    wp_m,wp_n,wp_k=wp_shape

    DDR_non_ideal_para=1.1 #Unmerged fragmented accesses will be scaled to sector (32bytes) as the smallest unit
    REG_spill_para=1.1 # just like the last one, something we don't know clearly about ncu

    compute_flops=2*m*n*k
    l2_hit_rate=L2_hit_rate_flow_sim_reuse_distance(m, n, k, tb_m, tb_n, tb_k, arch.l2_capacity, arch.sm_count, bytes_per_num, row_panel)
    # l2_hit_rate=0.5959
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

    l2_read_io=gridM*gridN*k*(tb_m*in1_level[0]+tb_n*in2_level[0])*bytes_per_num
    l2_store_io=gridM*gridN*(tb_m*tb_n)*bytes_per_num*out1_level[0]

    ddr_io=l2_read_io*(1-l2_hit_rate)+l2_store_io
    ddr_io=ddr_io*DDR_non_ideal_para

    active_warp_per_tb=(tb_m/wp_m)*(tb_n/wp_n)

    if (arch.core=="A100" or arch.core=="H100" or arch.core=="B200"):
        #a100 with special two-part l2 cache structure
        l2_io=l2_read_io*(l2_hit_rate)+l2_read_io*(1-l2_hit_rate)*2+l2_store_io*2
    else:
        l2_io=l2_read_io+l2_store_io

    smem_footprint=(tb_m*tb_k*(1-(in1_level[1]-in1_level[0]))+
                    tb_n*tb_k*(1-(in2_level[1]-in2_level[0]))+
                    tb_m*tb_n*(out1_level[1]-out1_level[0]))*bytes_per_num*stage_num

    if uses_gpu_resource_model(arch):
        reg_footprint = (math.ceil(wp_m * wp_n / 32 / (4 / bytes_per_num)) + math.ceil(wp_m * 32 / 32 / (4 / bytes_per_num))*in1_level[1] + math.ceil(wp_n * 32 / 32 / (4 / bytes_per_num)))*in2_level[1]
        reg_footprint=reg_footprint*REG_spill_para
        reg_footprint=math.ceil(reg_footprint)
    else:#with dup
        reg_footprint = (math.ceil(wp_m * wp_n / 32 / (4 / bytes_per_num)) + 1*math.ceil(wp_m * 32 / 32 / (4 / bytes_per_num))*in1_level[1] + 1*math.ceil(wp_n * 32 / 32 / (4 / bytes_per_num)))*in2_level[1]
        reg_footprint=reg_footprint*REG_spill_para
        reg_footprint=math.ceil(reg_footprint)


    if uses_gpu_resource_model(arch):
        #somehow 0.5 or 0.25 * ldgsts
        smem_io=0.5*(gridM*gridN*k*(tb_m*in1_level[0]+tb_n*in2_level[0])*bytes_per_num)  #derived in1&in2 level[0]
        #shared load
        smem_io+=(gridM*gridN*k*(tb_m+tb_n)*bytes_per_num)*active_warp_per_tb*(wp_m*in1_level[1]+wp_n*in2_level[1])/(tb_m+tb_n)
        #epilogue, shared store
        # maybe no epilogue? by cutlass with epilogue, but by welder no epilogue 
        # -----------------------
        # smem_io+=l2_store_io
        #epilogue, shared load
        # -----------------------
        # smem_io+=l2_store_io
        #store global
        l1_io=(gridM*gridN*(tb_m*tb_n)*bytes_per_num)*out1_level[1]
        #add together
        smem_l1_io=l1_io+smem_io
    else:#noldgsts
        #load global
        l1_io=(gridM*gridN*k*(tb_m*in1_level[0]+tb_n*in2_level[0])*bytes_per_num)
        #store shared
        smem_io=(gridM*gridN*k*(tb_m*in1_level[0]+tb_n*in2_level[0])*bytes_per_num)
        #shared load
        smem_io+=(gridM*gridN*k*(tb_m+tb_n)*bytes_per_num)*active_warp_per_tb*(wp_m*in1_level[1]+wp_n*in2_level[1])/(tb_m+tb_n)
        #epilogue, shared store
        # -----------------------
        # smem_io+=l2_store_io
        #epilogue, shared load
        # -----------------------
        # smem_io+=l2_store_io
        #store global
        l1_io=(gridM*gridN*(tb_m*tb_n)*bytes_per_num)*out1_level[1]
        #add together
        smem_l1_io=l1_io+smem_io
    
    ddr_read_io=l2_read_io*(1-l2_hit_rate)

    return ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io
        
# l2_hit_rate=L2_hit_rate_flow_sim_reuse_distance(m=8192, n=8192, k=8192, tb_m=128, tb_n=128, tb_k=32, l2_capacity=6*(1024)**2, sm_count=84, bytes_per_num=2, row_panel=8)
mem_levels = {
        'in1': [1,1,1,2],
        'in2': [1,1,1,2],
        'out1': [1,1,1,2],  
    }
l2_hit_rate=L2_hit_rate_flow_sim_reuse_distance(M=8192, N=8192, K=8192, tb_m=128, tb_n=128, tb_k=32, L2_Cap=6*(1024)**2, SM_Count=84, mem_levels=mem_levels, row_panel=8)
print(l2_hit_rate)
