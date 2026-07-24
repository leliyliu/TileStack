import pandas as pd

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
#     for metric_name, column_name in metrics_to_extract.items():
#         # Extract the value and handle the percentage value
#         value = row_data[column_name]
#         if isinstance(value, str) and '%' in value:
#             value = value.replace('%', '')
#         # Convert to numeric
#         try:
#             metric_value = pd.to_numeric(value, errors='coerce')
#             extracted_metrics[metric_name] = metric_value
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
#         extracted_metrics["Smem Footprint per Thread Block/Bytes"] =(extracted_metrics["Smem Static Footprint per Thread Block"]+extracted_metrics["Smem Dynamic Footprint per Thread Block"])*1000 #maybe 1000*

#     if ("Overall time /ms" in extracted_metrics)&("Overall time/cycles" in extracted_metrics)&("Frequency/GHz" in extracted_metrics):
#         extracted_metrics["Overall time /ms"] = extracted_metrics["Overall time/cycles"]/extracted_metrics["Frequency/GHz"]/1e6
#     # print(extracted_metrics)

#     return extracted_metrics

def extract_metrics_from_row_by_id(csv_file, row_id):
    df = pd.read_csv(csv_file)
    
    # Check if the row ID exists in the DataFrame
    if row_id not in df['ID'].values:
        print("emepty kernel, return all zero")
        
        metrics_to_extract = [
            "Compute util - ALU",
            "Compute util - FMA",
            "Compute util - Tensor Core",
            "Compute util - Tensor Core Int",
            "Compute util - Tensor Core ALL",
            "Compute util - XU",
            "DDR Read util",
            "DDR Write util",
            "L2 Hit Count",
            "L2 Miss Count",
            "L2 util",
            "L2 util Hopper",
            "Smem Static Footprint per Thread Block",
            "Smem Dynamic Footprint per Thread Block",
            "Smem/L1 util",
            "Reg footprint per Thread",
            "Overall time /ms",
            "Overall time/cycles",
            "Frequency/GHz",
            "DDR util",
            "L2 Read Hit Rate",
            "Smem Footprint per Thread Block/Bytes",
            "L2 Read Hit Rate",
        ]
        extracted_metrics = {metric: 0 for metric in metrics_to_extract}
        # do this to avoid division by zero
        extracted_metrics['Overall time /ms'] = 1e-9
        extracted_metrics['Overall time/cycles'] = 1
        utilization_metrics = [
            "Compute util - ALU",
            "Compute util - FMA",
            "Compute util - Tensor Core",
            "Compute util - Tensor Core Int",
            "Compute util - Tensor Core ALL",
            "Compute util - XU",
            "DDR Read util",
            "DDR Write util",
            "DDR util",
            "L2 util",
            "L2 util Hopper",
            "Smem/L1 util",
        ]
        for metric in utilization_metrics:
            extracted_metrics[metric] = 0.01
        return extracted_metrics
        # return f"No data found for ID {row_id}"

    
    # Extract the row with the given ID
    row_data = df[df['ID'] == row_id].iloc[0]

    # Define metrics to extract
    metrics_to_extract = {
        "Compute util - ALU": "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
        "Compute util - FMA": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
        "Compute util - Tensor Core": "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
        "Compute util - Tensor Core Int": "sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active",
        "Compute util - Tensor Core ALL": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "Compute util - XU": "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
        "DDR Read util": "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
        "DDR Write util": "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
        "L2 Hit Count": "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
        "L2 Miss Count": "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
        "L2 util": "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
        "L2 util Hopper": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "Smem Static Footprint per Thread Block": "launch__shared_mem_per_block_static",
        "Smem Dynamic Footprint per Thread Block": "launch__shared_mem_per_block_dynamic",
        "Smem/L1 util": "l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed",
        "Reg footprint per Thread": "launch__registers_per_thread",
        "Overall time /ms": "gpu__time_duration.sum",
        "Overall time/cycles": "gpc__cycles_elapsed.max",
        "Frequency/GHz": "gpc__cycles_elapsed.avg.per_second",
    }

    # Extract metrics
    extracted_metrics = {}
    extracted_metrics_units = {}
    unit_row = df.iloc[0]
    for metric_name, column_name in metrics_to_extract.items():
        # Extract the value and handle the percentage value
        value = row_data[column_name]
        unit = unit_row[column_name] if unit_row is not None else "Unit not found"
        # print(metric_name,value,unit)
        if isinstance(value, str) and '%' in value:
            value = value.replace('%', '')
        # Convert to numeric
        try:
            metric_value = pd.to_numeric(value, errors='coerce')
            extracted_metrics[metric_name] = metric_value
            extracted_metrics_units[metric_name] = unit
        except TypeError as e:
            extracted_metrics[metric_name] = f"Error: {str(e)}"

    # 后处理特定度量
    # GEMM/CONV: Tensor Core * 4
    if "Compute util - Tensor Core" in extracted_metrics:
        extracted_metrics["Compute util - Tensor Core"] *= 4
    
    if "Compute util - Tensor Core Int" in extracted_metrics:
        extracted_metrics["Compute util - Tensor Core Int"] *= 2

    if ("DDR Read util" in extracted_metrics) & ("DDR Write util" in extracted_metrics):
        extracted_metrics["DDR util"] =extracted_metrics["DDR Read util"]+extracted_metrics["DDR Write util"] 
    
    if ("L2 Hit Count" in extracted_metrics) & ("L2 Miss Count" in extracted_metrics):
        extracted_metrics["L2 Read Hit Rate"] =100*extracted_metrics["L2 Hit Count"]/(extracted_metrics["L2 Miss Count"]+extracted_metrics["L2 Hit Count"])
    
    if ("Smem Static Footprint per Thread Block" in extracted_metrics) & ("Smem Dynamic Footprint per Thread Block" in extracted_metrics):
        # if extracted_metrics_units["Smem Static Footprint per Thread Block"] == "byte/block":
        #     pass
        if extracted_metrics_units["Smem Static Footprint per Thread Block"] == "Kbyte/block":
            extracted_metrics["Smem Static Footprint per Thread Block"]=extracted_metrics["Smem Static Footprint per Thread Block"]*1000

        # if extracted_metrics_units["Smem Dynamic Footprint per Thread Block"] == "byte/block":
        #     pass
        if extracted_metrics_units["Smem Dynamic Footprint per Thread Block"] == "Kbyte/block":
            extracted_metrics["Smem Dynamic Footprint per Thread Block"]=extracted_metrics["Smem Dynamic Footprint per Thread Block"]*1000
        
        extracted_metrics["Smem Footprint per Thread Block/Bytes"] =(extracted_metrics["Smem Static Footprint per Thread Block"]+extracted_metrics["Smem Dynamic Footprint per Thread Block"]) #maybe 1000*

    if ("Overall time /ms" in extracted_metrics)&("Overall time/cycles" in extracted_metrics)&("Frequency/GHz" in extracted_metrics):
        if extracted_metrics_units["Frequency/GHz"] == "cycle/usecond":
            extracted_metrics["Frequency/GHz"]=extracted_metrics["Frequency/GHz"]/1000
        elif extracted_metrics_units["Frequency/GHz"] == "cycle/nsecond":
            pass            


        extracted_metrics["Overall time /ms"] = extracted_metrics["Overall time/cycles"]/extracted_metrics["Frequency/GHz"]/1e6
    # print(extracted_metrics)

    return extracted_metrics


if __name__ == "__main__":
    extracted_metrics=extract_metrics_from_row_by_id("ncu.csv",0)
    print("extracted_metrics",extracted_metrics)
