# 文件名: matmul_op.py
from ..util import *
# M, N, K, tb_m, tb_n, tb_k, L2_Cap, SM_Count, bytes_per_num, stage_num, block_per_sm
import numpy as np
import math


def _row_reduce_compute_time(arch, k_outer, k_inner_outer, k_inner):
    """Return reduction time through the architecture's public interface."""

    reference_query = getattr(arch, "row_reduce_time", None)
    if callable(reference_query):
        return float(reference_query(k_outer, k_inner_outer, k_inner))
    return (
        k_outer * (k_inner_outer * 2 / arch.max_freq)
        + (k_inner * 2 + 1) / arch.max_freq
    )


def calculate_row_reduce_resource_utilization(m, k, tb_m, rstep, reduce_thread, m_thread, l2_cap, sm_count, bytes_per_num, accelerator, arch):
    DDR_non_ideal_para = 1.0
    REG_spill_para = 1.1
    SMEM_Conflict_para = 1.1

    thread_m = math.ceil(tb_m / m_thread)
    thread_k = math.ceil(rstep / reduce_thread)

    k_outer = math.ceil(k / rstep)
    k_inner_outer = math.ceil(rstep / reduce_thread)
    k_inner = math.log2(reduce_thread)

    l2_hit_rate = 0

    compute_time = _row_reduce_compute_time(
        arch, k_outer, k_inner_outer, k_inner
    )

    overall_time = compute_time * m / sm_count / tb_m 

    l2_read_avg = sm_count * tb_m * k * bytes_per_num
    l2_st_avg = sm_count * tb_m * 1 * bytes_per_num

    ddr_io_avg = l2_read_avg * (1 - l2_hit_rate) + l2_st_avg

    l2_bw_req = (l2_read_avg * ((1 - l2_hit_rate) * 2 + l2_hit_rate) + l2_st_avg * 2) / compute_time
    ddr_bw_req = ddr_io_avg / compute_time
    ddr_bw_req *= DDR_non_ideal_para

    if reduce_thread <= 32:
        smem_footprint = tb_m * rstep * bytes_per_num
    else:
        smem_footprint = (tb_m * rstep + reduce_thread) * bytes_per_num

    reg_footprint = 32
    reg_footprint = math.ceil(reg_footprint * REG_spill_para)
    active_warp_per_tb = 4
    # Assuming active_thread_num_per_tb is thread_m * thread_k
    active_thread_num_per_tb = thread_m * thread_k
    reg_footprint_per_tb = reg_footprint * active_thread_num_per_tb




    if reduce_thread <= 32:
        l1_io_avg = k_outer * tb_m * rstep * bytes_per_num
        smem_io_avg = k_outer * tb_m * rstep * bytes_per_num
        smem_io_avg += tb_m * k_outer * k_inner_outer * reduce_thread * bytes_per_num * max(tb_m // 4, 1)
        l1_io_avg += tb_m * bytes_per_num * reduce_thread
        smem_io_avg += l1_io_avg
        smem_io_avg *= SMEM_Conflict_para
    else:
        l1_io_avg = k_outer * tb_m * rstep * bytes_per_num
        smem_io_avg = k_outer * tb_m * rstep * bytes_per_num
        smem_io_avg += tb_m * k_outer * k_inner_outer * reduce_thread * bytes_per_num * max(tb_m // 4, 1)
        current_reduce = reduce_thread
        while current_reduce > 32:
            smem_io_avg += current_reduce * 3 / 2 * bytes_per_num * 2
            current_reduce //= 2
        smem_io_avg += math.ceil(math.log2(min(32, reduce_thread))) * 3 * 32 * bytes_per_num
        l1_io_avg += tb_m * bytes_per_num * reduce_thread
        smem_io_avg += l1_io_avg
        smem_io_avg *= SMEM_Conflict_para

    compute_time = _row_reduce_compute_time(
        arch, k_outer, k_inner_outer, k_inner
    )
    smem_bw_req = smem_io_avg * sm_count / compute_time

    compute_util=1
    ddr_bw_util=ddr_bw_req/arch.ddr_bandwidth
    l2_bw_util=l2_bw_req/arch.l2_bandwidth
    l1_smem_bw_util=smem_bw_req/arch.smem_bandwidth

    max_pctg=1/0.9*max(compute_util,ddr_bw_util,l2_bw_util,l1_smem_bw_util)
    compute_util=compute_util/max_pctg
    ddr_bw_util=ddr_bw_util/max_pctg
    l2_bw_util=l2_bw_util/max_pctg
    l1_smem_bw_util=l1_smem_bw_util/max_pctg

    overall_time=overall_time/compute_util
    # result = bound_time, 0, ddr_util, l2_hit_rate, l2_util, smem_footprint, smem_l1_util, reg_footprint, compute_util
    return overall_time, 0, ddr_bw_util, l2_hit_rate, l2_bw_util, smem_footprint, l1_smem_bw_util, reg_footprint, compute_util
        
