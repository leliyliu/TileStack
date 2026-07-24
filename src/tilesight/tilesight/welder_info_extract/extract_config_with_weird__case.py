# speacil case for:
    # "config": "{'globals': {'Rasterization': <Rasterization2DColumn(16)>}, <Node, welder_C2DImplicitGemm_47__reshape_transpose_reshape_nn_bias_add_48>: 
    # {'block': [64, 128], 'warp': [32, 64], 'wmma': [16, 8, 16], 'use_cutlass': True, 'rstep': [32], 'use_tc': '80', 'strides': {3: <Stride, 0, 136>}}}",
    
    # "args": "[Tensor(shape=[64, 32, 128, 128], op.name=input0), Tensor(shape=[64, 128], op.name=input1), Tensor(shape=[64], op.name=input2), 
    # Tensor(shape=[64, 64, 64, 64], op.name=output0)]"
import ast
import re

def extract_rasterization_info(config_str):
    rasterization_match = re.search(r"'Rasterization': <(NoRasterization|Rasterization2D(Column|Row)\((\d+)\))>", config_str)
    if rasterization_match:
        rasterization_type = rasterization_match.group(1)
        if rasterization_type == "NoRasterization":
            return "No", None
        else:
            column_or_row = rasterization_match.group(2)
            number = int(rasterization_match.group(3))
            return column_or_row, number
    else:
        return None, None

def split_smem_fused_nodes(config_str):
    node_matches = re.findall(r"(<Node, [^>]+>: \{[^}]+\})", config_str)
    return node_matches

def extract_reg_fused_node_details(node_str):
    details = {}
    name_match = re.search(r"<Node, ([^>]+)>", node_str)
    if name_match:
        node_name = name_match.group(1)
        if '__' in node_name:
            split_names = node_name.split('__')
            details_first = extract_details(node_str, split_names[0])
            details_second = extract_details(node_str, split_names[1])
            return [details_first, details_second]
        else:
            return [extract_details(node_str, node_name)]
    return []

def extract_details(node_str, node_name):
    details = {'node_name': node_name}
    block_match = re.search(r"'block': (\[[^\]]+\])", node_str)
    if block_match:
        details['block'] = ast.literal_eval(block_match.group(1))
    thread_match = re.search(r"'thread': (\[[^\]]+\])", node_str)
    if thread_match:
        details['thread'] = ast.literal_eval(thread_match.group(1))
    rstep_match = re.search(r"'rstep': (\[[^\]]*\])", node_str)
    if rstep_match and ast.literal_eval(rstep_match.group(1)):
        details['rstep'] = ast.literal_eval(rstep_match.group(1))
    step_match = re.search(r"'step': (\[[^\]]+\])", node_str)
    if step_match:
        details['step'] = ast.literal_eval(step_match.group(1))
    warp_match = re.search(r"'warp': (\[[^\]]+\])", node_str)
    if warp_match:
        details['warp'] = ast.literal_eval(warp_match.group(1))
    wmma_match = re.search(r"'wmma': (\[[^\]]+\])", node_str)
    if wmma_match:
        details['wmma'] = ast.literal_eval(wmma_match.group(1))
    cutlass_match = re.search(r"'use_cutlass': (True|False)", node_str)
    if cutlass_match:
        details['use_cutlass'] = ast.literal_eval(cutlass_match.group(1))
    use_tc_match = re.search(r"'use_tc': '(\d+)'", node_str)
    if use_tc_match:
        details['use_tc'] = int(use_tc_match.group(1))
    reduce_thread_match = re.search(r"'reduce_thread': (\[[^\]]+\])", node_str)
    if reduce_thread_match:
        details['reduce_thread'] = ast.literal_eval(reduce_thread_match.group(1))
    return details

def process_welder__config(config_str):
    rasterization_type, rasterization_number = extract_rasterization_info(config_str)
    node_substrings = split_smem_fused_nodes(config_str)
    nodes_details = []
    for node_str in node_substrings:
        nodes_details.extend(extract_reg_fused_node_details(node_str))
    return {
        'Rasterization': {
            'type': rasterization_type,
            'number': rasterization_number
        },
        'Nodes': nodes_details
    }

# 测试新的功能
# test_str = "{'globals': {'Rasterization': <Rasterization2DColumn(16)>}, <Node, welder_C2DImplicitGemm_47__reshape_transpose_reshape_nn_bias_add_48>: {'block': [64, 128], 'warp': [32, 64], 'wmma': [16, 8, 16], 'use_cutlass': True, 'rstep': [32], 'use_tc': '80', 'strides': {3: <Stride, 0, 136>}}}"
# test_str="{'globals': {'Rasterization': <NoRasterization>}, <Node, welder_C2DImplicitGemm_84__reshape_transpose_reshape_nn_bias_add_multiply_add_85>: {'block': [128, 256], 'warp': [64, 128], 'wmma': [16, 8, 16], 'use_cutlass': True, 'rstep': [32], 'use_tc': '80', 'strides': {5: <Stride, 0, 264>}}}"
# test_str="{'globals': {'Rasterization': <NoRasterization>}, <Node, welder_C2DImplicitGemm_71__reshape_transpose_reshape_nn_bias_add_72>: {'block': [128, 64], 'warp': [64, 32], 'wmma': [16, 8, 16], 'use_cutlass': True, 'rstep': [32], 'use_tc': '80', 'strides': {3: <Stride, 0, 72>}}, <Node, reshape_transpose_reshape_nn_bias_add_multiply_add_70>: {'block': [1, 64, 1, 32], 'thread': [1, 16, 1, 8], 'rstep': [], 'step': [1, 1, 1, 2]}, <Node, welder_C2DImplicitGemm_84__reshape_transpose_reshape_nn_bias_add_multiply_add_85>: {'block': [128, 256], 'warp': [64, 128], 'wmma': [16, 8, 16], 'use_cutlass': True, 'rstep': [32], 'use_tc': '80', 'strides': {5: <Stride, 0, 264>}}}"

# print(process_welder__config(test_str))
