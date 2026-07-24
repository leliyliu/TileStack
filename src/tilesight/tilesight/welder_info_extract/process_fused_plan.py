from tilesight.compare_with_ncu.extract_metrcis_from_modeling_data import extract_metrcis_from_modeling_data
from tilesight.fusion_support.smem_fusion import smem_fusion
from tilesight.welder_info_extract.extract_config_with_weird__case import process_welder__config
from tilesight.welder_info_extract.process_reg_fused_nodes import process_reg_fused_nodes

def process_fused_plan(supported_ops,fused_plan,relay_list,chip):

    # # print json key&value --------------------------------------------------------------------------------------------
    # # print(type(fused_plan['nodes'])) # list, length means number of smem fused kernels
    # # print(type(fused_plan['node_names'])) # list, like the one above
    # # print(type(fused_plan['group_id'])) # int, 0~number-1
    # # print(type(fused_plan['block_size'])) # list, number of threads per block
    # # print(type(fused_plan['grid_size'])) #list, number of thread blocks per kernel
    # # print(type(fused_plan['latency'])) #float, 2Xlatency? maybe
    # # print(type(fused_plan['config'])) #str, need to analyze
    # # print(type(fused_plan['args'])) #str, need to analyze

    # # # print node_names in ['node_names'] key-value --------------------------------------------------------------------------------------------
    # for reg_fused_ops in fused_plan['node_names']:
    #     # print(parse_reg_fused_ops(reg_fused_ops,supported_ops))
    #     reg_fused_ops,reg_fused_op_id=parse_reg_fused_ops(reg_fused_ops,supported_ops)
    #     print(reg_fused_ops,reg_fused_op_id)
    #     # for single_op in reg_fused_ops:
    #     #     print(single_op)

    # print(process_welder__config(fused_plan['config']))
    smem_fused_config=process_welder__config(fused_plan['config'])
    global_arg_op_mapping_list=fused_plan['arg_op_mapping_list']
    global_input_desc_list=fused_plan['input_desc']
    global_output_desc_list=fused_plan['output_desc']
    # print("fused_plan",fused_plan)
    # print("fused_plan['config']",fused_plan['config'])
    # print("smem_fused_config",smem_fused_config)
    # global_desc_list=global_input_desc_list+global_output_desc_list
    # print("global_arg_op_mapping_list",global_arg_op_mapping_list)
    # print("global_input_desc_list",global_input_desc_list)
    # print("global_output_desc_list",global_output_desc_list)
    # print("global_desc_list",global_desc_list)
    # # return 
    # now analyze the rasterization information: have impact on L2 hit rate
    # print("Rasterization",smem_fused_config['Rasterization'])
    # print("smem_fused_config",smem_fused_config)
    if smem_fused_config['Rasterization']['type'] == "No":
        row_panel=chip.sm_count
    elif smem_fused_config['Rasterization']['type'] == "Row":
        row_panel=smem_fused_config['Rasterization']['number']
    else:
        row_panel=chip.sm_count//smem_fused_config['Rasterization']['number']

    smem_fused_nodes=smem_fused_config['Nodes'] 
    # 
    # print("smem_fused_nodes",smem_fused_nodes)

    time_list=[]
    smem_fusion_list=[]

    for reg_fused_nodes in smem_fused_nodes:# 同一个reg_fused_nodes内用相同的tiling，reg_fused_nodes里的n个single op的tensor形状需要自己去提取
        # print("reg_fused_nodes",reg_fused_nodes) 
        # # reg_fused_nodes {'node_name': 'divide_multiply_add_4', 'block': [1, 16, 1, 256], 'thread': [1, 4, 1, 32], 'step': [1, 1, 1, 2]}
        # single_op_list,reg_fused_op_id=parse_reg_fused_ops(reg_fused_nodes['node_name'],supported_ops) #op_id from 0 to the last; op_id =reg_fused_op_id
        
        
        # dispatch_reg_fused_global_arg_levels #TODO
        time_added,smem_to_process_data=process_reg_fused_nodes(reg_fused_nodes,relay_list,supported_ops,row_panel,global_arg_op_mapping_list,
                                                  global_input_desc_list,global_output_desc_list,chip)
        # print("len(reg_fused_op_list)",len(reg_fused_op_list))
        time_list.append(time_added)
        smem_fusion_list.append(smem_to_process_data)

    # print("Op idx ",idx,"'s single reg op added time =",1000*sum(time_list)," ms")

    grid_size=fused_plan['grid_size']
    reg_fusion_time_added,smem_fused_post_data=smem_fusion(smem_fusion_list,grid_size,chip)
    # print("Op idx ",idx,"'s reg fused op added time =",1000*reg_fusion_time_added," ms")
    modeling_metrics=extract_metrcis_from_modeling_data(smem_fused_post_data)
    

    # grids=[bs/512,num_attention_heads/1,seq/1,seq/1]
    # grids=[bs/2,num_attention_heads/64,seq/1,seq/1]
    # grids=[bs/2,num_attention_heads/64,seq/1,seq/1]
    # grids=[bs/2,num_attention_heads/64,seq/1]
    # row_id=7
    # return smem_fuse_analyze(time_list,smem_fusion_list,grids,row_id,csv_path)

        # print("single_op_list",single_op_list,"reg_fused_op_id",reg_fused_op_id)
        # # single_op_list ['reshape_transpose_reshape', 'nn_bias_add'] reg_fused_op_id 72
        # for single_op in single_op_list:
        #     print(single_op) # mean, add, sqrt
        
    # print("len(smem_fused_op_list)",len(smem_fused_op_list))
    # smem_fused_args=process_welder_args(fused_plan['args'])
    # print("smem_fused_args",smem_fused_args)
    # smem_fused_args [{'shape': [64, 16, 256, 256], 'op.name': 'p0'}, {'shape': [64, 1, 256, 256], 'op.name': 'p1'}, {'shape': [1, 16, 1, 1], 'op.name': 'p2'}, {'shape': [1, 16, 1, 1], 'op.name': 'p3'}, {'shape': [32, 16], 'op.name': 'p1'}, {'shape': [32], 'op.name': 'p1'}, {'shape': [64, 32, 256, 256], 'op.name': 'T_add'}]
    # # smem_fused_args [{'shape': [64, 64, 64, 64], 'op.name': 'input0'}, {'shape': [128, 256], 'op.name': 'input1'}, {'shape': [128], 'op.name': 'input2'}, {'shape': [64, 128, 32, 32], 'op.name': 'output0'}]
    # print("before update")
    # for reg_fused_op_list in smem_fused_op_list:
    #     for single_op in reg_fused_op_list:
    #         print (single_op)
    
    # ---------------------------------------old function---------
    # update_operation_levels(smem_fused_args, smem_fused_op_list)
    # ------------------------------------------------------------

    # update_each_reg_fused_global_arg_levels(global_input_desc_list,global_output_desc_list,global_arg_op_mapping_list,smem_fused_op_list)
    # print("after update")
    # for reg_fused_op_list in smem_fused_op_list:
        # for single_op in reg_fused_op_list:
            # print (single_op)

    return modeling_metrics
 