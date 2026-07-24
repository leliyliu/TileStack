def extract_metrcis_from_modeling_data(post_data):

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
    "Compute util - Tensor Core": post_data[8]*100,  # 假定 ALU Utilization 与 Compute Util 相同
    "Compute util - SFU": post_data[8]*100  # 假定 ALU Utilization 与 Compute Util 相同
}
    return modeling_metrics