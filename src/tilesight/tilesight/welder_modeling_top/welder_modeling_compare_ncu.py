import json
import math
import time
import os
import cProfile
import pstats

from tilesight.onnx_model import *
from tilesight.arch import *
from tilesight.fused_op_dtype import *
from tilesight.util import *
from tilesight.welder_info_extract import *
from tilesight.compare_with_ncu import *
from tilesight.fusion_support import *

PROFILE_ENABLE = 0

def welder_modeling_compare_ncu(json_path, relay_path, ncu_path):
    # 初始化芯片
    chip = A100()

    # 定义在自动比较程序中支持的操作
    supported_ops = [
        "subtract", "multiply", "mean", "add", "sqrt",
        "divide", "welder_C2DImplicitGemm", "nn_bias_add",
        "reshape", "transpose", "nn_conv2d",
        "strided_slice", "nn_global_avg_pool2d", "nn_depth_to_space",
        "nn_matmul","welder_matmul","nn_relu","nn_avg_pool2d",
        "layout_transform",
    ]

    # 如果启用性能分析器
    if PROFILE_ENABLE == 1:
        profiler = cProfile.Profile()
        profiler.enable()

    # 从 JSON 文件中提取数据
    tuned_plan_list = extract_data_from_json(json_path)
    relay_list = extract_relay_list(relay_path, store_txt=False)

    ncu_metrics_list = []
    modeling_metrics_list = []
    json_time_list = []

    # 记录开始时间
    start_time = time.time()

    for idx in range(len(tuned_plan_list)):
        print(tuned_plan_list[idx])
        modeling_metrics = process_fused_plan(supported_ops, tuned_plan_list[idx], relay_list, chip)
        ncu_metrics = extract_metrics_from_row_by_id(ncu_path, idx)
        
        ncu_metrics_list.append(ncu_metrics)
        modeling_metrics_list.append(modeling_metrics)
        json_time_list.append(tuned_plan_list[idx]['latency'])
    
    print("Time added by tuning multi single kernel: ", sum(json_time_list), "ms")

    # 比较并绘制结果
    # plot_metrics_comparison(ncu_metrics_list, modeling_metrics_list)
    
    # 获取统计信息
    ncu_tuple_info,modeling_tuple_info=process_statistics(ncu_metrics_list, modeling_metrics_list, chip)

    # 记录整体执行结束时间
    end_time = time.time()
    total_execution_time = (end_time - start_time)
    print("")
    print(f"Total modeling execution time: {total_execution_time:.2f} s")

    # 如果启用性能分析器，输出性能分析结果
    if PROFILE_ENABLE == 1:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats()
        profiler.dump_stats('output.prof')

    # 返回相关的统计结果或数据，如果需要可以改为返回其他类型的结果
    # return ncu_metrics_list, modeling_metrics_list, total_execution_time

    return ncu_tuple_info,modeling_tuple_info


# 示例调用
if __name__ == "__main__":
    json_path = '../testing/json_log/NAFNet_bs64_fp16_after_reboot.json'  # 替换为实际的文件路径
    relay_path = '../testing/relay/relay_nafnet_bs64_fp16.txt'
    ncu_path = '../testing/ncu_rep/NAFNet_bs64_fp16.csv'
    
    result = welder_modeling_compare_ncu(json_path, relay_path, ncu_path)
    print(result)
