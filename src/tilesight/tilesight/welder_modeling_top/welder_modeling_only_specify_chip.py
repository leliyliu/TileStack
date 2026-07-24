import time
from tilesight.onnx_model import *
from tilesight.arch import *
from tilesight.fused_op_dtype import *
from tilesight.util import *
from tilesight.welder_info_extract import *
from tilesight.compare_with_ncu import *
from tilesight.fusion_support import *

def welder_modeling_only_specify_chip(json_path, relay_path,chip):
    # 初始化芯片
    # chip = A100()
    # chip = MI210()
    print("chip.smem_bandwidth",chip.smem_bandwidth)

    # 支持的操作
    supported_ops = [
        "maximum",
        "max","sum",
        "expand_dims","broadcast_to",
        "concatenate",
        "negative","take","where","less",
        "nn_batch_matmul","rsqrt","tanh",
        "ladder_layout_transform","ladder_perfect_matmul","cast",
        "subtract", "multiply", "mean", "add", "sqrt","exp","sigmoid",
        "divide", "welder_C2DImplicitGemm", "nn_bias_add",
        "reshape", "transpose", "nn_conv2d",
        "strided_slice", "nn_global_avg_pool2d", "nn_depth_to_space",
        "nn_matmul","welder_matmul","nn_relu","nn_avg_pool2d",
        "layout_transform",
    ]

    # 从 JSON 文件中提取数据
    tuned_plan_list = extract_data_from_json(json_path)
    relay_list = extract_relay_list(relay_path, store_txt=False)
    # for relay in relay_list:
    #     print("relay",relay)
    #     print("nononono")

    modeling_metrics_list = []
    json_time_list = []
    modeling_time_list = []
    op_name_list= []

    # 记录开始时间
    start_time = time.time()
    for idx in range(len(tuned_plan_list)):
        print(tuned_plan_list[idx])
        modeling_metrics = process_fused_plan(supported_ops, tuned_plan_list[idx], relay_list, chip)
        modeling_metrics_list.append(modeling_metrics)
        json_time_list.append(tuned_plan_list[idx]['latency'])
        modeling_time_list.append(modeling_metrics['Overall time /ms'])
        op_name_list.append(tuned_plan_list[idx]['node_names'])

    print("Time added by tuning multi single kernel: ", sum(json_time_list), "ms")
    print("Time added by modeling multi single kernel: ", sum(modeling_time_list), "ms")

    # 获取统计信息
    modeling_info_tuple = process_statistics_modeling_only(modeling_metrics_list, chip)

    plot_modeling_vs_real_by_op(modeling_time_list,json_time_list,op_name_list,json_path)
    


    # 记录整体执行结束时间
    end_time = time.time()
    total_execution_time = (end_time - start_time)
    print("")
    print(f"Total modeling execution time: {total_execution_time:.2f} s")

    # modeling_time_added,modeling_ddr_util,modeling_l2_hit_rate,modeling_l2_util,modeling_smem_footprint,modeling_smem_l1_util,modeling_reg_footprint,modeling_compute_util
    return modeling_info_tuple

# 示例调用
if __name__ == "__main__":
    json_path = 'json_log/NAFNet_bs64_fp16_after_reboot.json'  # 替换为实际的文件路径
    relay_path = 'relay/relay_nafnet_bs64_fp16.txt'
    result = welder_modeling_only(json_path, relay_path)
    print(result)
