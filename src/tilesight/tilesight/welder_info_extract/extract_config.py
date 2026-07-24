import ast
import re

def extract_rasterization_info(config_str):
    """
    从配置字符串中提取 Rasterization 的信息。
    
    参数:
    - config_str: 配置字符串。
    
    返回:
    - Rasterization 的类型 (No, Row, Column)。
    - 如果是 Row 或 Column,返回相应的数字。
    """
    # 正则表达式匹配 Rasterization 信息
    rasterization_match = re.search(r"'Rasterization': <(NoRasterization|Rasterization2D(Column|Row)\((\d+)\))>", config_str)
    
    if rasterization_match:
        rasterization_type = rasterization_match.group(1)
        if rasterization_type == "NoRasterization":
            return "No", None
        else:
            # 提取 Column 或 Row 以及对应的数字
            column_or_row = rasterization_match.group(2)
            number = int(rasterization_match.group(3))
            return column_or_row, number
    else:
        return None, None

# # 测试示例字符串
# config_str1 = "{'globals': {'Rasterization': <NoRasterization>}, <Node, strided_slice_strided_slice_multiply_140>: {'block': [1, 32, 1, 64], 'thread': [1, 2, 1, 64], 'rstep': [], 'step': [1, 2, 1, 1]}, <Node, welder_C2DImplicitGemm_141>: {'block': [32, 64], 'warp': [16, 32], 'wmma': [8, 32, 16], 'use_cutlass': False, 'rstep': [32], 'use_tc': '80', 'strides': {2: <Stride, 0, 72>}}}"
# config_str2 = "{'globals': {'Rasterization': <Rasterization2DColumn(16)>}, <Node, welder_C2DImplicitGemm_47__reshape_transpose_reshape_nn_bias_add_48>: {'block': [64, 128], 'warp': [32, 64], 'wmma': [16, 8, 16], 'use_cutlass': True, 'rstep': [32], 'use_tc': '80', 'strides': {3: <Stride, 0, 136>}}}"

# # 运行测试
# print(extract_rasterization_info(config_str1), extract_rasterization_info(config_str2))

def split_smem_fused_nodes(config_str):
    """
    将配置字符串分割成多个 Node 的子字符串。
    
    参数:
    - config_str: 配置字符串。
    
    返回:
    - 分割出的 Node 子字符串列表。
    """
    # 使用正则表达式匹配所有 Node 定义
    node_matches = re.findall(r"(<Node, [^>]+>: \{[^}]+\})", config_str)
    
    return node_matches

# # 测试示例字符串
# print(split_smem_fused_nodes(config_str1))
# print(split_smem_fused_nodes(config_str2))

def extract_reg_fused_node_details(node_str):
    """
    从 Node 子字符串中提取详细信息。
    
    参数:
    - node_str: Node 子字符串。
    
    返回:
    - 包含提取信息的字典。
    """
    details = {}
    
    # 提取 node_name
    name_match = re.search(r"<Node, ([^>]+)>", node_str)
    if name_match:
        details['node_name'] = name_match.group(1)
    
    # 提取 block
    block_match = re.search(r"'block': (\[[^\]]+\])", node_str)
    if block_match:
        details['block'] = ast.literal_eval(block_match.group(1))
    
    # 提取 thread（如果存在）
    thread_match = re.search(r"'thread': (\[[^\]]+\])", node_str)
    if thread_match:
        details['thread'] = ast.literal_eval(thread_match.group(1))
    # else:
    #     details['thread'] = []
    
    # 提取 rstep（如果存在）
    rstep_match = re.search(r"'rstep': (\[[^\]]*\])", node_str)
    if rstep_match:
        if ast.literal_eval(rstep_match.group(1)):
            details['rstep'] = ast.literal_eval(rstep_match.group(1))
    
    # 提取 step
    step_match = re.search(r"'step': (\[[^\]]+\])", node_str)
    if step_match:
        details['step'] = ast.literal_eval(step_match.group(1))
    
    # 提取reduce_thread
    reduce_thread_match = re.search(r"'reduce_thread': (\[[^\]]+\])", node_str)
    if reduce_thread_match:
        details['reduce_thread'] = ast.literal_eval(step_match.group(1))

    # 提取 warp（如果存在）
    warp_match = re.search(r"'warp': (\[[^\]]+\])", node_str)
    if warp_match:
        details['warp'] = ast.literal_eval(warp_match.group(1))
    
    # 提取 wmma（如果存在）
    wmma_match = re.search(r"'wmma': (\[[^\]]+\])", node_str)
    if wmma_match:
        details['wmma'] = ast.literal_eval(wmma_match.group(1))
    
    # 提取 use_cutlass（如果存在）
    cutlass_match = re.search(r"'use_cutlass': (True|False)", node_str)
    if cutlass_match:
        details['use_cutlass'] = ast.literal_eval(cutlass_match.group(1))
    
    # 提取 use_tc（如果存在）
    use_tc_match = re.search(r"'use_tc': '(\d+)'", node_str)
    if use_tc_match:
        details['use_tc'] = int(use_tc_match.group(1))

    # if 'rstep' in details and details['rstep']:
    # 如果'rstep'存在于details中且对应的值不为None或空
    # 这里执行相关的逻辑
    
    return details

# # 测试示例 Node 子字符串
# node_str_example = "<Node, strided_slice_strided_slice_multiply_140>: {'block': [1, 32, 1, 64], 'thread': [1, 2, 1, 64], 'rstep': [], 'step': [1, 2, 1, 1]}"
# # print(extract_reg_fused_node_details(node_str_example))

# 定义一个函数来处理整个配置字符串，并提取所需的所有信息
def process_welder_config(config_str):
    # 首先提取 Rasterization 信息
    rasterization_type, rasterization_number = extract_rasterization_info(config_str)
    
    # 分割 Node 子字符串
    node_substrings = split_smem_fused_nodes(config_str)
    
    # 从每个 Node 子字符串中提取详细信息
    nodes_details = [extract_reg_fused_node_details(node_str) for node_str in node_substrings]
    
    return {
        'Rasterization': {
            'type': rasterization_type,
            'number': rasterization_number
        },
        'Nodes': nodes_details
    }


# for test -----------------------------------------------------------------------------------------------


# # 示例配置字符串
# config_str_examples = [
#     "{'globals': {'Rasterization': <NoRasterization>}, <Node, strided_slice_strided_slice_multiply_140>: {'block': [1, 32, 1, 64], 'thread': [1, 2, 1, 64], 'rstep': [], 'step': [1, 2, 1, 1]}, <Node, welder_C2DImplicitGemm_141>: {'block': [32, 64], 'warp': [16, 32], 'wmma': [8, 32, 16], 'use_cutlass': False, 'rstep': [32], 'use_tc': '80', 'strides': {2: <Stride, 0, 72>}}, <Node, reshape_transpose_reshape_nn_bias_add_multiply_add_142>: {'block': [1, 32, 1, 64], 'thread': [1, 32, 1, 4], 'rstep': [], 'step': [1, 1, 1, 2]}, <Node, welder_C2DImplicitGemm_143>: {'block': [64, 64], 'warp': [32, 32], 'wmma': [16, 16, 16], 'use_cutlass': False, 'rstep': [32], 'use_tc': '80', 'strides': {2: <Stride, 0, 72>}}}",
#     # "{'globals': {'Rasterization': <NoRasterization>}, <Node, welder_C2DImplicitGemm_95>: {'block': [32, 64], 'warp': [16, 32], 'wmma': [16, 8, 16], 'use_cutlass': True, 'rstep': [32], 'block_order': block_idx % 128 // 16 * 1024 + block_idx // 128 * 16 + block_idx % 16, 'use_tc': '80', 'strides': {2: <Stride, 0, 72>}}, <Node, reshape_transpose_reshape_nn_depth_to_space_add_96>: {'block': [1, 8, 4, 64], 'thread': [1, 8, 2, 8], 'rstep': [], 'step': [1, 1, 1, 2]}}",
#     # "{'globals': {'Rasterization': <Rasterization2DColumn(16)>}, <Node, welder_C2DImplicitGemm_47__reshape_transpose_reshape_nn_bias_add_48>: {'block': [64, 128], 'warp': [32, 64], 'wmma': [16, 8, 16], 'use_cutlass': True, 'rstep': [32], 'use_tc': '80', 'strides': {3: <Stride, 0, 136>}}}"
# ]

# # 处理每个配置字符串并打印结果
# # processed_configs = [process_welder_config(config_str) for config_str in config_str_examples]
# # print(processed_configs)

# for config_str in config_str_examples:
#     print(process_welder_config(config_str))
#     smem_fused_plan=process_welder_config(config_str)
#     print(smem_fused_plan['Rasterization'])
#     reg_fused_nodes=smem_fused_plan['Nodes']
#     print(reg_fused_nodes)
#     for node in reg_fused_nodes:
#         print(node)
#         # if(not node['rstep']):
#         # if(node['rstep']==False):
#         #     print("Null")

# # if 'thread' in node and node['thread']:
# #     print("'thread' 存在且不为空")
# # else:
# #     print("'thread' 不存在或为空列表")
