from tilesight.arch import Arch
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go

def process_statistics(ncu_metrics_list,modeling_metrics_list,arch_spec:Arch):

    def safe_divide(numerator, denominator, default=100):
            """
            安全除法函数,避免被除数为0的情况。
            如果被除数为0,则返回一个默认值,否则返回除法结果。
            
            Parameters:
            - numerator: 分子
            - denominator: 分母
            - default: 如果分母为0,返回的默认值
            
            Returns:
            - 除法结果或默认值
            """
            if denominator == 0:
                return default
            else:
                return numerator / denominator

    selected_metrics = [
    "Overall time /ms",
    "DDR util",
    "L2 Read Hit Rate",
    "L2 util",
    "Smem Footprint per Thread Block/Bytes",
    "Smem/L1 util",
    "Reg footprint per Thread",
    # "Compute util - ALU",
    "Compute util - FMA",
    "Compute util - Tensor Core"
]
    ncu_time_added, ncu_ddr_util, ncu_l2_hit_rate, ncu_l2_util, ncu_smem_footprint, ncu_smem_l1_util, ncu_reg_footprint, ncu_fma_util, ncu_tc_util=0,0,0,0,0,0,0,0,0

    for ncu_metric in ncu_metrics_list:
        ncu_time_added+=ncu_metric["Overall time /ms"]
        ncu_ddr_util+=ncu_metric["Overall time /ms"]*ncu_metric["DDR util"]
        ncu_l2_hit_rate+=ncu_metric["Overall time /ms"]*ncu_metric["L2 Read Hit Rate"]
        ncu_l2_util+=ncu_metric["Overall time /ms"]*ncu_metric["L2 util"]
        ncu_smem_footprint+=ncu_metric["Overall time /ms"]*ncu_metric["Smem Footprint per Thread Block/Bytes"]
        ncu_smem_l1_util+=ncu_metric["Overall time /ms"]*ncu_metric["Smem/L1 util"]
        ncu_reg_footprint+=ncu_metric["Overall time /ms"]*ncu_metric["Reg footprint per Thread"]
        ncu_fma_util+=ncu_metric["Overall time /ms"]*ncu_metric["Compute util - FMA"]
        ncu_tc_util+=ncu_metric["Overall time /ms"]*ncu_metric["Compute util - Tensor Core"]

    ncu_ddr_util/=ncu_time_added
    ncu_l2_hit_rate/=ncu_time_added
    ncu_l2_util/=ncu_time_added
    ncu_smem_footprint/=ncu_time_added
    ncu_smem_l1_util/=ncu_time_added
    ncu_reg_footprint/=ncu_time_added
    ncu_fma_util/=ncu_time_added
    ncu_tc_util/=ncu_time_added

    ncu_tuple_info=(ncu_time_added, ncu_ddr_util, ncu_l2_hit_rate, ncu_l2_util, ncu_smem_footprint, ncu_smem_l1_util, ncu_reg_footprint, ncu_fma_util, ncu_tc_util)

    modeling_time_added, modeling_ddr_util, modeling_l2_hit_rate, modeling_l2_util, modeling_smem_footprint, modeling_smem_l1_util, modeling_reg_footprint, modeling_compute_util=0,0,0,0,0,0,0,0

    for modeling_metric in modeling_metrics_list:
        modeling_time_added+=modeling_metric["Overall time /ms"]
        modeling_ddr_util+=modeling_metric["Overall time /ms"]*modeling_metric["DDR util"]
        modeling_l2_hit_rate+=modeling_metric["Overall time /ms"]*modeling_metric["L2 Read Hit Rate"]
        modeling_l2_util+=modeling_metric["Overall time /ms"]*modeling_metric["L2 util"]
        modeling_smem_footprint+=modeling_metric["Overall time /ms"]*modeling_metric["Smem Footprint per Thread Block/Bytes"]
        modeling_smem_l1_util+=modeling_metric["Overall time /ms"]*modeling_metric["Smem/L1 util"]
        modeling_reg_footprint+=modeling_metric["Overall time /ms"]*modeling_metric["Reg footprint per Thread"]
        modeling_compute_util+=modeling_metric["Overall time /ms"]*modeling_metric["Compute util - FMA"]

    modeling_ddr_util/=modeling_time_added
    modeling_l2_hit_rate/=modeling_time_added
    modeling_l2_util/=modeling_time_added
    modeling_smem_footprint/=modeling_time_added
    modeling_smem_l1_util/=modeling_time_added
    modeling_reg_footprint/=modeling_time_added
    modeling_compute_util/=modeling_time_added

    # 打印标题
    print("{:<30} {:<10} {:<10}".format("Metric", "NCU", "Modeling"))

    # 打印各项指标
    print("{:<30} {:<10.4f} {:<10.4f}".format("Overall Time /ms", ncu_time_added, modeling_time_added))
    print("{:<30} {:<10.4f} {:<10.4f}".format("DDR Util", ncu_ddr_util, modeling_ddr_util))
    print("{:<30} {:<10.4f} {:<10.4f}".format("L2 Read Hit Rate", ncu_l2_hit_rate, modeling_l2_hit_rate))
    print("{:<30} {:<10.4f} {:<10.4f}".format("L2 Util", ncu_l2_util, modeling_l2_util))
    print("{:<30} {:<10.4f} {:<10.4f}".format("Smem Footprint per TB /Bytes", ncu_smem_footprint, modeling_smem_footprint))
    print("{:<30} {:<10.4f} {:<10.4f}".format("Smem/L1 Util", ncu_smem_l1_util, modeling_smem_l1_util))
    print("{:<30} {:<10.4f} {:<10.4f}".format("Reg Footprint per Thread", ncu_reg_footprint, modeling_reg_footprint))
    print("{:<30} {:<10.4f} {:<10.4f}".format("Compute Util - FMA", ncu_fma_util, modeling_compute_util))
    print("{:<30} {:<10.4f} {:<10.4f}".format("Compute Util - Tensor Core", ncu_tc_util,modeling_compute_util))

    modeling_tuple_info=(modeling_time_added,modeling_ddr_util,modeling_l2_hit_rate,modeling_l2_util,modeling_smem_footprint,modeling_smem_l1_util,modeling_reg_footprint,modeling_compute_util)
    
    print("-------------------------------------------------------------------------------")
    print("After letting ncu working @max utilization=90%")

    ncu_time_added, ncu_ddr_util, ncu_l2_hit_rate, ncu_l2_util, ncu_smem_footprint, ncu_smem_l1_util, ncu_reg_footprint, ncu_fma_util, ncu_tc_util=0,0,0,0,0,0,0,0,0

    for ncu_metric in ncu_metrics_list:
        # ratio=90/max(ncu_metric['DDR util'],ncu_metric['L2 util'],ncu_metric['Smem/L1 util'],ncu_metric['Compute util - FMA'],ncu_metric['Compute util - Tensor Core'],ncu_metric['Compute util - ALU'])
        # ratio=min(100*arch_spec.ddr_max_util/ncu_metric['DDR util'],100*arch_spec.l2_max_util/ncu_metric['L2 util'],100*arch_spec.l1_max_util/ncu_metric['Smem/L1 util'],100*arch_spec.compute_max_util/ncu_metric['Compute util - FMA'],100*arch_spec.compute_max_util/ncu_metric['Compute util - Tensor Core'],100*arch_spec.compute_max_util/ncu_metric['Compute util - ALU'])
        
        

        # 假设 arch_spec 和 ncu_metric 已经被正确定义
        ratio = min(
            100 * safe_divide(arch_spec.ddr_max_util, ncu_metric['DDR util']),
            100 * safe_divide(arch_spec.l2_max_util, ncu_metric['L2 util']),
            100 * safe_divide(arch_spec.l1_max_util, ncu_metric['Smem/L1 util']),
            100 * safe_divide(arch_spec.compute_max_util, ncu_metric['Compute util - FMA']),
            100 * safe_divide(arch_spec.compute_max_util, ncu_metric['Compute util - Tensor Core']),
            100 * safe_divide(arch_spec.compute_max_util, ncu_metric['Compute util - ALU'])
        )

        # print("ratio",ratio)
        # ratio=max(100*arch_spec.ddr_max_util/ncu_metric['DDR util'],100*arch_spec.l2_max_util/ncu_metric['L2 util'],100*arch_spec.l1_max_util/ncu_metric['Smem/L1 util'],100*arch_spec.compute_max_util/(ncu_metric['Compute util - FMA']+ncu_metric['Compute util - Tensor Core']+ncu_metric['Compute util - ALU']))
        ncu_metric['Overall time /ms']/=ratio
        ncu_metric['DDR util']*=ratio
        ncu_metric['L2 util']*=ratio
        ncu_metric['Smem/L1 util']*=ratio
        ncu_metric['Compute util - FMA']*=ratio
        ncu_metric['Compute util - Tensor Core']*=ratio
        ncu_metric['Compute util - ALU']*=ratio

        ncu_time_added+=ncu_metric["Overall time /ms"]
        ncu_ddr_util+=ncu_metric["Overall time /ms"]*ncu_metric["DDR util"]
        ncu_l2_hit_rate+=ncu_metric["Overall time /ms"]*ncu_metric["L2 Read Hit Rate"]
        ncu_l2_util+=ncu_metric["Overall time /ms"]*ncu_metric["L2 util"]
        ncu_smem_footprint+=ncu_metric["Overall time /ms"]*ncu_metric["Smem Footprint per Thread Block/Bytes"]
        ncu_smem_l1_util+=ncu_metric["Overall time /ms"]*ncu_metric["Smem/L1 util"]
        ncu_reg_footprint+=ncu_metric["Overall time /ms"]*ncu_metric["Reg footprint per Thread"]
        ncu_fma_util+=ncu_metric["Overall time /ms"]*ncu_metric["Compute util - FMA"]
        ncu_tc_util+=ncu_metric["Overall time /ms"]*ncu_metric["Compute util - Tensor Core"]

    ncu_ddr_util/=ncu_time_added
    ncu_l2_hit_rate/=ncu_time_added
    ncu_l2_util/=ncu_time_added
    ncu_smem_footprint/=ncu_time_added
    ncu_smem_l1_util/=ncu_time_added
    ncu_reg_footprint/=ncu_time_added
    ncu_fma_util/=ncu_time_added
    ncu_tc_util/=ncu_time_added

    # 打印标题
    print("{:<30} {:<10} {:<10}".format("Metric", "NCU", "Modeling"))

    # 打印各项指标
    print("{:<30} {:<10.4f} {:<10.4f}".format("Overall Time /ms", ncu_time_added, modeling_time_added))
    print("{:<30} {:<10.4f} {:<10.4f}".format("DDR Util", ncu_ddr_util, modeling_ddr_util))
    print("{:<30} {:<10.4f} {:<10.4f}".format("L2 Read Hit Rate", ncu_l2_hit_rate, modeling_l2_hit_rate))
    print("{:<30} {:<10.4f} {:<10.4f}".format("L2 Util", ncu_l2_util, modeling_l2_util))
    print("{:<30} {:<10.4f} {:<10.4f}".format("Smem Footprint per TB /Bytes", ncu_smem_footprint, modeling_smem_footprint))
    print("{:<30} {:<10.4f} {:<10.4f}".format("Smem/L1 Util", ncu_smem_l1_util, modeling_smem_l1_util))
    print("{:<30} {:<10.4f} {:<10.4f}".format("Reg Footprint per Thread", ncu_reg_footprint, modeling_reg_footprint))
    print("{:<30} {:<10.4f} {:<10.4f}".format("Compute Util - FMA", ncu_fma_util, modeling_compute_util))
    print("{:<30} {:<10.4f} {:<10.4f}".format("Compute Util - Tensor Core", ncu_tc_util,modeling_compute_util))

    return ncu_tuple_info,modeling_tuple_info

def get_modeling_metrics(post_data):

    modeling_metrics = {
    "Overall time /ms": post_data[0]*1000,
    "DDR util": post_data[2]*100,
    "L2 Read Hit Rate": post_data[3]*100,
    "L2 util": post_data[4]*100,
    "Smem Footprint per Thread Block/Bytes": post_data[5],  # 假定这是对应的
    "Smem/L1 util": post_data[6]*100,
    "Reg footprint per Thread": post_data[7],
    # "Compute util - ALU": post_data[8]*100,  # 假定 ALU Utilization 与 Compute Util 相同
    "Compute util - FMA": post_data[8]*100,  # 假定 ALU Utilization 与 Compute Util 相同
    "Compute util - Tensor Core": post_data[8]*100  # 假定 ALU Utilization 与 Compute Util 相同
}
    return modeling_metrics

def process_statistics_print_modeling_only(modeling_metrics_list):

    modeling_time_added, modeling_ddr_util, modeling_l2_hit_rate, modeling_l2_util, modeling_smem_footprint, modeling_smem_l1_util, modeling_reg_footprint, modeling_compute_util = 0, 0, 0, 0, 0, 0, 0, 0

    for modeling_metric in modeling_metrics_list:
        modeling_time_added += modeling_metric["Overall time /ms"]
        modeling_ddr_util += modeling_metric["Overall time /ms"] * modeling_metric["DDR util"]
        modeling_l2_hit_rate += modeling_metric["Overall time /ms"] * modeling_metric["L2 Read Hit Rate"]
        modeling_l2_util += modeling_metric["Overall time /ms"] * modeling_metric["L2 util"]
        modeling_smem_footprint += modeling_metric["Overall time /ms"] * modeling_metric["Smem Footprint per Thread Block/Bytes"]
        modeling_smem_l1_util += modeling_metric["Overall time /ms"] * modeling_metric["Smem/L1 util"]
        modeling_reg_footprint += modeling_metric["Overall time /ms"] * modeling_metric["Reg footprint per Thread"]
        modeling_compute_util += modeling_metric["Overall time /ms"] * modeling_metric["Compute util - FMA"]

    modeling_ddr_util /= modeling_time_added
    modeling_l2_hit_rate /= modeling_time_added
    modeling_l2_util /= modeling_time_added
    modeling_smem_footprint /= modeling_time_added
    modeling_smem_l1_util /= modeling_time_added
    modeling_reg_footprint /= modeling_time_added
    modeling_compute_util /= modeling_time_added

    # 打印标题
    print("{:<30} {:<10}".format("Metric", "Modeling"))

    # 打印各项指标
    print("{:<30} {:<10.4f}".format("Overall Time /ms", modeling_time_added))
    print("{:<30} {:<10.4f}".format("DDR Util", modeling_ddr_util))
    print("{:<30} {:<10.4f}".format("L2 Read Hit Rate", modeling_l2_hit_rate))
    print("{:<30} {:<10.4f}".format("L2 Util", modeling_l2_util))
    print("{:<30} {:<10.4f}".format("Smem Footprint per TB /Bytes", modeling_smem_footprint))
    print("{:<30} {:<10.4f}".format("Smem/L1 Util", modeling_smem_l1_util))
    print("{:<30} {:<10.4f}".format("Reg Footprint per Thread", modeling_reg_footprint))
    print("{:<30} {:<10.4f}".format("Compute Util - FMA, TC, SFU", modeling_compute_util))

    modeling_tuple_info = (
        modeling_time_added,
        modeling_ddr_util,
        modeling_l2_hit_rate,
        modeling_l2_util,
        modeling_smem_footprint,
        modeling_smem_l1_util,
        modeling_reg_footprint,
        modeling_compute_util
    )

    return modeling_tuple_info
  
def process_statistics_modeling_only(modeling_metrics_list,arch_spec:Arch):

    modeling_time_added, modeling_ddr_util, modeling_l2_hit_rate, modeling_l2_util, modeling_smem_footprint, modeling_smem_l1_util, modeling_reg_footprint, modeling_compute_util=0,0,0,0,0,0,0,0

    for modeling_metric in modeling_metrics_list:
        modeling_time_added+=modeling_metric["Overall time /ms"]
        modeling_ddr_util+=modeling_metric["Overall time /ms"]*modeling_metric["DDR util"]
        modeling_l2_hit_rate+=modeling_metric["Overall time /ms"]*modeling_metric["L2 Read Hit Rate"]
        modeling_l2_util+=modeling_metric["Overall time /ms"]*modeling_metric["L2 util"]
        modeling_smem_footprint+=modeling_metric["Overall time /ms"]*modeling_metric["Smem Footprint per Thread Block/Bytes"]
        modeling_smem_l1_util+=modeling_metric["Overall time /ms"]*modeling_metric["Smem/L1 util"]
        modeling_reg_footprint+=modeling_metric["Overall time /ms"]*modeling_metric["Reg footprint per Thread"]
        modeling_compute_util+=modeling_metric["Overall time /ms"]*modeling_metric["Compute util - FMA"]

    modeling_ddr_util/=modeling_time_added
    modeling_l2_hit_rate/=modeling_time_added
    modeling_l2_util/=modeling_time_added
    modeling_smem_footprint/=modeling_time_added
    modeling_smem_l1_util/=modeling_time_added
    modeling_reg_footprint/=modeling_time_added
    modeling_compute_util/=modeling_time_added

    return (modeling_time_added,modeling_ddr_util,modeling_l2_hit_rate,modeling_l2_util,modeling_smem_footprint,modeling_smem_l1_util,modeling_reg_footprint,modeling_compute_util)
  

def plot_metrics_comparison(ncu_metrics_list, modeling_metrics_list, folder_path):
    # 检查文件夹是否存在，不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # 提取时间数据
    ncu_times = [metric["Overall time /ms"] for metric in ncu_metrics_list]
    modeling_times = [metric["Overall time /ms"] for metric in modeling_metrics_list]
    ids = list(range(len(modeling_metrics_list)))  # 操作ID

    # 使用Matplotlib保存PNG文件
    plt.figure(figsize=(30, 10))
    plt.plot(ncu_times, label='NCU Metrics', marker='o')
    plt.plot(modeling_times, label='Modeling Metrics', marker='s')
    plt.title('Comparison of Overall Times from NCU and Modeling Metrics')
    plt.xlabel('Operation ID')
    plt.ylabel('Overall Time (ms)')
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'metrics_comparison.png'), dpi=300)
    plt.close()

    # 使用Plotly创建并保存HTML文件
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ids, y=ncu_times, mode='lines+markers', name='NCU Metrics'))
    fig.add_trace(go.Scatter(x=ids, y=modeling_times, mode='lines+markers', name='Modeling Metrics'))
    fig.update_layout(title='Interactive Comparison of Overall Times from NCU and Modeling Metrics',
                      xaxis_title='Operation ID',
                      yaxis_title='Overall Time (ms)')
    fig.write_html(os.path.join(folder_path, 'metrics_comparison.html'))

    # # 使用这个函数
    # ncu_example = [{"Overall time /ms": 10}, {"Overall time /ms": 15}, {"Overall time /ms": 20}, {"Overall time /ms": 25}]
    # modeling_example = [{"Overall time /ms": 8}, {"Overall time /ms": 18}, {"Overall time /ms": 22}, {"Overall time /ms": 30}]

    # plot_metrics_comparison(ncu_example, modeling_example)

def plot_modeling_time(modeling_metrics_list, folder_path='./cafe_time_plot'):
    # 检查文件夹是否存在，不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # 提取时间数据
    modeling_times = [metric["Overall time /ms"] for metric in modeling_metrics_list]
    ids = list(range(len(modeling_metrics_list)))  # 操作ID

    # 使用Matplotlib保存PNG文件
    x_len=len(ids)
    plt.figure(figsize=(x_len, 20))
    plt.plot(modeling_times, label='Modeling Metrics', marker='s')
    plt.title('Comparison of Overall Times from NCU and Modeling Metrics')
    plt.xlabel('Operation ID')
    plt.ylabel('Overall Time (ms)')
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'metrics_comparison.png'), dpi=300)
    plt.close()

    # 使用Plotly创建并保存HTML文件
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ids, y=modeling_times, mode='lines+markers', name='Modeling Metrics'))
    fig.update_layout(title='Interactive Comparison of Overall Times from NCU and Modeling Metrics',
                      xaxis_title='Operation ID',
                      yaxis_title='Overall Time (ms)')
    fig.write_html(os.path.join(folder_path, 'metrics_comparison.html'))

def plot_modeling_vs_real_by_op(modeling_time_list, json_time_list, op_name_list, json_path):
    # Create a directory to save the plot based on the json_path directory
    plot_dir = os.path.dirname(json_path)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(plot_dir, 'modeling_vs_real_times.png')

    # Format the operation names to display on x-axis
    # Assuming each element in op_name_list is a list of strings, we will join all parts and then simplify for display
    formatted_op_names = ['\n'.join(op) for op in op_name_list]

    fig, ax1 = plt.subplots()

    # Set the x-axis and tick positions
    x = range(len(op_name_list))
    ax1.set_xlabel('Operations')
    ax1.set_xticks(x)
    # ax1.set_xticklabels(formatted_op_names, rotation=45, ha="right", fontsize=8)

    # Create the first y-axis for modeling times
    color = 'tab:red'
    ax1.set_ylabel('Modeling Time (ms)', color=color)
    ax1.plot(x, modeling_time_list, color=color, label='Modeling Time',linewidth=1, marker='o', markersize=3)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for real times
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Real Time (ms)', color=color)
    ax2.plot(x, json_time_list, color=color, label='Real Time',linewidth=1, marker='o', markersize=3)
    ax2.tick_params(axis='y', labelcolor=color)

    # Set the same y-axis limits for both axes
    # y_min = min(min(modeling_time_list), min(json_time_list))
    y_min = 0
    y_max = max(max(modeling_time_list), max(json_time_list))
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    # Title and legend
    fig.tight_layout()
    fig.suptitle('Comparison of Modeling and Real Execution Times', va='bottom')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

    # Save the plot
    plt.savefig(plot_path,dpi=300)
    # plt.show()
    # print("Time added by tuning multi single kernel: ", sum(json_time_list), "ms")
    # print("Time added by modeling multi single kernel: ", sum(modeling_time_list), "ms")
    # print("len of json_time_list",len(json_time_list))
    # print("len of modeling_time_list",len(modeling_time_list))    
    # print("real_time_list",json_time_list)
    # print("modeling_time_list",modeling_time_list)

    print(f"Plot saved to {plot_path}")