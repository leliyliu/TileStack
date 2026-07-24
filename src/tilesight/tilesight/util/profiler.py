import os
import subprocess
import pandas as pd

# 定义输出CSV的函数
def export_ncu_to_csv(ncu_report_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_csv_file = os.path.join(output_folder, f"{os.path.splitext(ncu_report_file)[0]}.csv")

    # 构建ncu命令，将.ncu-rep文件转换为CSV
    ncu_command = f"ncu --import {ncu_report_file} --csv --page raw > {output_csv_file}"
    
    # 执行命令
    subprocess.run(ncu_command, shell=True, check=True)
    return output_csv_file

# def extract_metrics_from_row_by_id(csv_file, row_id):
#     df = pd.read_csv(csv_file)
    
#     # Check if the row ID exists in the DataFrame
#     if row_id not in df['ID'].values:
#         return f"No data found for ID {row_id}"
    
#     # Extract the row with the given ID
#     row_data = df[df['ID'] == row_id].iloc[0]

#     # Define metrics to extract
#     metrics_to_extract = {
#         "Compute util - ALU": "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
#         "Compute util - FMA": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
#         "Compute util - Tensor Core": "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
#         "Compute util - Tensor Core Int": "sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active",
#         "DDR Read util": "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
#         "DDR Write util": "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
#         "L2 Hit Count": "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
#         "L2 Miss Count": "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
#         "L2 util": "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
#         "Smem Static Footprint per Thread Block": "launch__shared_mem_per_block_static",
#         "Smem Dynamic Footprint per Thread Block": "launch__shared_mem_per_block_dynamic",
#         "Smem/L1 util": "l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed",
#         "Reg footprint per Thread": "launch__registers_per_thread",
#         "Overall time /ms": "gpu__time_duration.sum",
#         "Overall time/cycles": "gpc__cycles_elapsed.max",
#         "Frequency/GHz": "gpc__cycles_elapsed.avg.per_second",
#     }

#     # Extract metrics
#     extracted_metrics = {}
#     extracted_metrics_units = {}
#     unit_row = df.iloc[0]
#     for metric_name, column_name in metrics_to_extract.items():
#         # Extract the value and handle the percentage value
#         value = row_data[column_name]
#         unit = unit_row[column_name] if unit_row is not None else "Unit not found"
#         # print(metric_name,value,unit)
#         if isinstance(value, str) and '%' in value:
#             value = value.replace('%', '')
#         # Convert to numeric
#         try:
#             metric_value = pd.to_numeric(value, errors='coerce')
#             extracted_metrics[metric_name] = metric_value
#             extracted_metrics_units[metric_name] = unit
#         except TypeError as e:
#             extracted_metrics[metric_name] = f"Error: {str(e)}"

#     # 后处理特定度量
#     # GEMM/CONV: Tensor Core * 4
#     if "Compute util - Tensor Core" in extracted_metrics:
#         extracted_metrics["Compute util - Tensor Core"] *= 4
    
#     if "Compute util - Tensor Core Int" in extracted_metrics:
#         extracted_metrics["Compute util - Tensor Core Int"] *= 2

#     if ("DDR Read util" in extracted_metrics) & ("DDR Write util" in extracted_metrics):
#         extracted_metrics["DDR util"] =extracted_metrics["DDR Read util"]+extracted_metrics["DDR Write util"] 
    
#     if ("L2 Hit Count" in extracted_metrics) & ("L2 Miss Count" in extracted_metrics):
#         extracted_metrics["L2 Read Hit Rate"] =100*extracted_metrics["L2 Hit Count"]/(extracted_metrics["L2 Miss Count"]+extracted_metrics["L2 Hit Count"])
    
#     if ("Smem Static Footprint per Thread Block" in extracted_metrics) & ("Smem Dynamic Footprint per Thread Block" in extracted_metrics):
#         # if extracted_metrics_units["Smem Static Footprint per Thread Block"] == "byte/block":
#         #     pass
#         if extracted_metrics_units["Smem Static Footprint per Thread Block"] == "Kbyte/block":
#             extracted_metrics["Smem Static Footprint per Thread Block"]=extracted_metrics["Smem Static Footprint per Thread Block"]*1000

#         # if extracted_metrics_units["Smem Dynamic Footprint per Thread Block"] == "byte/block":
#         #     pass
#         if extracted_metrics_units["Smem Dynamic Footprint per Thread Block"] == "Kbyte/block":
#             extracted_metrics["Smem Dynamic Footprint per Thread Block"]=extracted_metrics["Smem Dynamic Footprint per Thread Block"]*1000
        
#         extracted_metrics["Smem Footprint per Thread Block/Bytes"] =(extracted_metrics["Smem Static Footprint per Thread Block"]+extracted_metrics["Smem Dynamic Footprint per Thread Block"]) #maybe 1000*

#     if ("Overall time /ms" in extracted_metrics)&("Overall time/cycles" in extracted_metrics)&("Frequency/GHz" in extracted_metrics):
#         if extracted_metrics_units["Frequency/GHz"] == "cycle/usecond":
#             extracted_metrics["Frequency/GHz"]=extracted_metrics["Frequency/GHz"]/1000
#         elif extracted_metrics_units["Frequency/GHz"] == "cycle/nsecond":
#             pass            


#         extracted_metrics["Overall time /ms"] = extracted_metrics["Overall time/cycles"]/extracted_metrics["Frequency/GHz"]/1e6
#     # print(extracted_metrics)

#     return extracted_metrics

# def extract_metrics_from_row_by_id(csv_file, row_id):
#     df = pd.read_csv(csv_file)
    
#     # Check if the row ID exists in the DataFrame
#     if row_id not in df['ID'].values:
#         return f"No data found for ID {row_id}"
    
#     # Extract the row with the given ID
#     row_data = df[df['ID'] == row_id].iloc[0]

#     # Define metrics to extract
#     metrics_to_extract = {
#         "Compute util - ALU": "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
#         "Compute util - FMA": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
#         "Compute util - Tensor Core": "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
#         "Compute util - Tensor Core Int": "sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active",
#         "DDR Read util": "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
#         "DDR Write util": "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
#         "L2 Hit Count": "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
#         "L2 Miss Count": "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
#         "L2 util": "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
#         "Smem Static Footprint per Thread Block": "launch__shared_mem_per_block_static",
#         "Smem Dynamic Footprint per Thread Block": "launch__shared_mem_per_block_dynamic",
#         "Smem/L1 util": "l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed",
#         "Reg footprint per Thread": "launch__registers_per_thread",
#         "Overall time /ms": "gpu__time_duration.sum",
#         "Overall time/cycles": "gpc__cycles_elapsed.max",
#         "Frequency/GHz": "gpc__cycles_elapsed.avg.per_second",
#     }

#     # Attempt to find units for each metric from a description row
#     # Assuming units are mentioned in the DataFrame in a specific row or pattern
#     # unit_row_index = df.apply(lambda row: row.astype(str).str.contains('unit|sec|ms|µs|ns|cycles', regex=True).any(), axis=1)
#     # # print("unit_row_index",unit_row_index)
#     # unit_row = df[unit_row_index].iloc[0] if unit_row_index.any() else None

#     unit_row = df.iloc[0]

#     # Extract metrics and their units
#     extracted_metrics = {}
#     for metric_name, column_name in metrics_to_extract.items():
#         value = row_data.get(column_name, "Metric not found")
#         unit = unit_row[column_name] if unit_row is not None else "Unit not found"
#         print(metric_name,value,unit)
        
#         # Handling percentage values
#         if isinstance(value, str) and '%' in value:
#             value = value.replace('%', '')

#         # Convert to numeric
#         try:
#             metric_value = pd.to_numeric(value, errors='coerce')
#             extracted_metrics[metric_name] = {'value': metric_value, 'unit': unit}
#         except TypeError as e:
#             extracted_metrics[metric_name] = {'error': str(e), 'unit': unit}

#     return extracted_metrics


# # 主函数
# def main():

#     # do_ncu=False
#     do_ncu=True

#     if(do_ncu==True):
#         # 定义 ncu 性能分析命令
#         NCU_PROFILE_CMD = "ncu -f -o ncu_test --set full --cache-control none python3 run_fused_ops.py"
#         # --cache-control none
        

#         # 运行 NCU 性能分析
#         # subprocess.run(NCU_PROFILE_CMD, shell=True, check=True)

#         # 假设 ncu_test.ncu-rep 是 ncu 性能分析的输出文件
#         # ncu_report_file = "ncu_test.ncu-rep"
#         ncu_report_file = "llama2_70b_bs512_seq1_int8.ncu-rep"
#         # ncu_report_file = "NAFNet-bs64-fp32-24-0126.ncu-rep"
#         output_folder = "ncu_rep"

#         # 将 .ncu-rep 文件转换为 CSV
#         csv_file = export_ncu_to_csv(ncu_report_file, output_folder)

#     csv_path = 'ncu_rep/llama2_13b_layer1_seq2048_bs1.csv'  # 指定您的 CSV 文件路径
#     # csv_path = 'ncu_rep/NAFNet-bs64-fp32.csv'  # 指定您的 CSV 文件路径

#     # 4,9,15,20
#     row_ids = [1]  # 指定要提取的行 ID
#     for row_id in row_ids:
#         metrics = extract_metrics_from_row_by_id(csv_path, row_id)
#         print("--------------------------")
#         for key, value in metrics.items():
#             print(f"{key}: {value}")

#     # # 定义输出文件夹和文件名
#     # log_folder = "ncu_rep/log"
#     # log_file = os.path.join(log_folder, "metrics_log.txt")

#     # # 确保输出文件夹存在
#     # if not os.path.exists(log_folder):
#     #     os.makedirs(log_folder)

#     # # 将度量数据写入文件
#     # with open(log_file, "w") as file:
#     #     for key, value in metrics.items():
#     #         file.write(f"{key}: {value}\n")

#     # print(f"Metrics written to {log_file}")

# if __name__ == "__main__":
#     extracted_metrics=extract_metrics_from_row_by_id("ncu.csv",0)
#     print("extracted_metrics",extracted_metrics)
