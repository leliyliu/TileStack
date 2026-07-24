import json

def extract_data_from_json(json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)

    tuned_plan_list = []
    for item in json_data:
        # 提取arg_op_mapping_list，如果不存在则默认为空列表
        arg_op_mapping_list = item.get('arg_op_mapping_list', [])
        input_desc_list= item.get('input_desc', [])
        output_desc_list= item.get('output_desc',[])
        
        item_data = {
            'nodes': item.get('nodes', None),
            'node_names': item.get('node_names', None),
            'group_id': item.get('group_id', None),
            'block_size': item.get('block_size', None),
            'grid_size': item.get('grid_size', None),
            'latency': item.get('latency', None),
            'config': item.get('config', None),
            'args': item.get('args', None),
            # 新增加的arg_op_mapping_list
            'arg_op_mapping_list': arg_op_mapping_list,
            'input_desc':input_desc_list,
            'output_desc':output_desc_list,
        }
        tuned_plan_list.append(item_data)

    return tuned_plan_list

# tuned_plan_list = extract_data_from_json("tuned.json")
# first_item_arg_op_mapping_list = tuned_plan_list[0]['arg_op_mapping_list']
# # 遍历arg_op_mapping_list中的每一项
# for arg_mapping in first_item_arg_op_mapping_list:
#     print(f"Reg Fused Op Name: {arg_mapping['reg_fused_op_name']}")
#     print(f"Tensor Shape: {arg_mapping['tensor_shape']}")
#     print(f"Dtype: {arg_mapping['dtype']}")
#     print(f"Pointer Name: {arg_mapping['pointer_name']}\n")

# print(tuned_plan_list[0])

# {'nodes': [0], 'node_names': ['subtract_multiply_0'], 'group_id': 0, 'block_size': [128, 1, 1], 'grid_size': [65536, 1, 1], 'latency': 4.782323837280273, 'config': "{'globals': {'Rasterization': <NoRasterization>}, <Node, subtract_multiply_0>: {'block': [2, 64, 1, 64], 'thread': [2, 1, 1, 64], 'rstep': []}}", 'args': '[Tensor(shape=[128, 64, 256, 256], op.name=p0), Tensor(shape=[128, 1, 256, 256], op.name=p1), Tensor(shape=[128, 64, 256, 256], op.name=T_subtract), Tensor(shape=[128, 64, 256, 256], op.name=T_multiply)]', 
# 'arg_op_mapping_list': [{'reg_fused_op_name': 'subtract_multiply_0', 'tensor_shape': [128, 64, 256, 256], 'dtype': 'float32', 'pointer_name': 'p0'}, {'reg_fused_op_name': 'subtract_multiply_0', 'tensor_shape': [128, 1, 256, 256], 'dtype': 'float32', 'pointer_name': 'p1'}, {'reg_fused_op_name': 'subtract_multiply_0', 'tensor_shape': [128, 64, 256, 256], 'dtype': 'float32', 'pointer_name': 'T_subtract'}, {'reg_fused_op_name': 'subtract_multiply_0', 'tensor_shape': [128, 64, 256, 256], 'dtype': 'float32', 'pointer_name': 'T_multiply'}], 
# 'input_desc': [['subtract_multiply_0', 0, 0], ['subtract_multiply_0', 0, 1]], 
# 'output_desc': [['subtract_multiply_0', 0, 0], ['subtract_multiply_0', 0, 1]]}
