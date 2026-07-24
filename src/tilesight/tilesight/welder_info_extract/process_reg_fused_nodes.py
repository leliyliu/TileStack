import re
from tilesight.welder_info_extract import parse_reg_fused_ops
from tilesight.fused_op_dtype import Operation
from tilesight.welder_info_extract import dispatch_to_modeling
from tilesight.welder_info_extract.welder_config import Welder_Config
from tilesight.fusion_support import *

welder_name_to_relay_name = {
    # "nn_conv2d_nn_bias_add": handle_nn_conv2d_nn_bias_add,
    # "reshape_transpose_reshape": handle_reshape_transpose_reshape,
    "maximum": "maximum",
    "less": "less",
    "tanh": "tanh",
    "where": "where",
    "take":"take",
    "sigmoid": "sigmoid",
    "sum": "sum",
    "exp": "exp",
    "max": "max",
    "expand_dims": "expand_dims",
    "broadcast_to": "broadcast_to",
    "concatenate": "concatenate",
    "negative": "negative",
    "rsqrt":"rsqrt",
    "nn_batch_matmul":"nn.batch_matmul",
    "cast":"cast",
    "ladder_perfect_matmul":"ladder.perfect_matmul",
    "ladder_layout_transform":"ladder.layout_transform",
    "reshape": "reshape",
    "transpose": "transpose",
    "nn_conv2d": "nn.conv2d",
    "nn_bias_add": "nn.bias_add",
    "subtract": "subtract",
    "multiply": "multiply",
    "mean": "mean",
    "add": "add",
    "sqrt": "sqrt",
    "divide": "divide",
    "welder_C2DImplicitGemm": "welder.C2DImplicitGemm",
    "strided_slice": "strided_slice",
    "nn_global_avg_pool2d": "nn.global_avg_pool2d",
    "nn_depth_to_space": "nn.depth_to_space",
    "nn_matmul": "nn.matmul",
    "welder_matmul": "welder.matmul",
    "nn_relu": "nn.relu",
    "nn_avg_pool2d":"nn.avg_pool2d",
    "layout_transform":"layout_transform"
}

# supported_ops = [
#     "nn_conv2d_nn_bias_add", "subtract", "multiply", "mean", "add", "sqrt",
#     "divide", "welder_C2DImplicitGemm", "nn_bias_add",
#     "reshape_transpose_reshape",
#     "strided_slice", "nn_global_avg_pool2d","nn_depth_to_space",
# ]

dtype_to_bytes = {
    "float32": 4, "f32": 4,  # 映射float32和f32到4 bytes
    "float16": 2, "f16": 2,  # 映射float16和f16到2 bytes
    "int8": 1, "i8": 1,       # 映射int8和i8到1 byte
    "int32": 4, "i32": 4,     # 映射int32和i32到4 bytes
    "int64": 8, "i64": 8,     # 映射int64和i64到8 bytes
    "float64": 8,
    "bool"  : 1,
}

def find_firtst_tensor_string_after_item(def_line, input_item):
    # 查找input_item在字符串中的位置
    start_index = def_line.find(input_item)
    if start_index == -1:
        return "input_item not found"
    
    # 从input_item的位置开始，找到第一个"Tensor["的位置
    tensor_start_index = def_line.find("Tensor[", start_index)
    if tensor_start_index == -1:
        return "Tensor string not found"
    
    # 从"Tensor["的位置开始，找到对应的"]"的位置
    tensor_end_index = def_line.find("]", tensor_start_index)
    if tensor_end_index == -1:
        return "Closing bracket not found"
    
    # 提取并返回Tensor字符串
    tensor_string = def_line[tensor_start_index:tensor_end_index+1]
    return tensor_string

def find_last_tensor_string_in_one_line(def_line):
    """
    Finds the last tensor string in a given definition line.
    
    Args:
    - def_line (str): A string that may contain one or more tensor definitions.
    
    Returns:
    - str: The last tensor string in the definition line, if found; otherwise, returns "No match found".
    """
    # Regular expression to match the Tensor string with its dimensions and type
    # matches = re.findall(r"Tensor\[\(.*?\), float16\]", def_line)
    matches = re.findall(r"Tensor\[\(.*?\),.*?\]", def_line)
    

    # Return the last match if any matches are found
    if matches:
        return matches[-1]
    else:
        return "No match found"

def find_tensor_strings_after_arrow(s):
    # 查找 "->" 之后的所有内容
    arrow_pos = s.find("->")
    if arrow_pos == -1:
        return []

    after_arrow = s[arrow_pos + 2:]  # "+2" 是为了跳过 "->" 符号本身

    # 使用正则表达式查找所有匹配的Tensor字符串
    # pattern = r"Tensor\[\(.*?\), float16\]"
    pattern = r"Tensor\[\(.*?\),.*?\]"
    output_tensor_str_list = re.findall(pattern, after_arrow)

    return output_tensor_str_list

def split_metrics_in_line(line,op_name):
    # Initialize variables
    metrics = ["padding", "channels", "kernel_size","begin","end","strides","axes","keepdims","axis","groups","newshape","block_size","mode",
               "src_layout","dst_layout","data_layout","kernel_layout"]
    # extracted_metrics = []
    extracted_metrics = {}

    modified_line = line

    # Helper function to extract and transform values within brackets
    def extract_bracket_values(metric, value_str):
        if metric in ["padding", "kernel_size", "axis", "newshape","strides"]:
            # Convert string values within brackets to list of integers
            if (op_name!="strided_slice"):
                return [int(x.strip()) for x in value_str.split(',')]
            else:
                return value_str
            
        return value_str

    # Helper function to convert specific metric values
    def convert_value(metric, value_str):
        if metric in ["groups", "channels","block_size"]:
            return int(value_str)
        elif metric == "keepdims":
            return value_str.lower() == "true"
        elif metric in ["src_layout","dst_layout","data_layout","kernel_layout"]:
            return value_str.strip('"')
        return value_str

    # Iterate through each metric to find and extract its value from the line
    for metric in metrics:
        if metric in line:
            start_index = line.find(metric) + len(metric) + 1  # Find the start index of the value
            if line[start_index] == '[':
                # Find the end of the bracketed value
                end_index = line.find(']', start_index) + 1
                value_str = line[start_index+1:end_index-1]
                value = extract_bracket_values(metric, value_str)
            else:
                # Find the end of the value based on , or )
                comma_index = line.find(',', start_index)
                close_paren_index = line.find(')', start_index)
                if comma_index == -1:
                    end_index = close_paren_index
                elif close_paren_index == -1:
                    end_index = comma_index
                else:
                    end_index = min(comma_index, close_paren_index)
                
                value_str = line[start_index:end_index].strip()
                value = convert_value(metric, value_str)  # Convert value based on specific metric rules

            # Append the metric and its extracted value
            # extracted_metrics.append((metric, value))
            
            extracted_metrics.update({metric:value})

            # Remove the extracted metric from the line
            if line[end_index] == ',':
                end_index += 2  # Skip past the comma and space
            modified_line = modified_line.replace(line[start_index- len(metric) - 1:end_index], '')

    # Replace the first occurrence of ",)" with ")" if exists
    modified_line = modified_line.replace(", )", ")", 1)

    return extracted_metrics, modified_line

def extract_tensor_shape_dtype(tensor_info_str):
    # 使用正则表达式匹配形状信息中的数字
    shape_str = re.search(r'\((.*?)\)', tensor_info_str).group(1)
    shape_list = [int(num) for num in re.findall(r'\d+', shape_str)]
    
    # 提取数据类型
    dtype = tensor_info_str.split(", ")[-1].rstrip("]")
    
    
    return shape_list, dtype

def extract_single_op_shape_in_relay(current_relay, current_line, op_name, op_class):
    lines = current_relay.split("\n")
    operation_found = False
    input_details = []
    line_counter=current_line

    # Search for the operation from the current_line
    for line in lines[current_line:]:
        if op_name in line:
            extracted_metrics, line =split_metrics_in_line(line,op_name)
            # op_class.op_special_metric_dict.update('')
            operation_found = True
            founded_line=line_counter
            # Extracting the substring within parentheses
            start_idx = line.find("(") + 1
            end_idx = line.find(")")
            inputs = line[start_idx:end_idx].split(", ")
            # print("inputs",inputs)

            for input_item in inputs:
                if input_item.startswith("%"):
                    # Handling tensor pointers
                    if input_item[1] == "p":
                        # print("input_item",input_item)
                        # Predefined tensor, search at the beginning
                        for def_line in lines:
                            if input_item in def_line:
                                # print("def_line",def_line)
                                # tensor_info = def_line.split(": ")[1]
                                tensor_info = find_firtst_tensor_string_after_item(def_line,input_item)
                                shape, dtype = extract_tensor_shape_dtype(tensor_info)
                                input_details.append({"shape": shape, "dtype": dtype, "level":"smem"})
                                break
                    else:
                        # Tensor defined within the relay, find its definition
                        for def_line in lines[1:-1]:
                            if input_item in def_line:
                                # print("def_line",def_line)
                                tensor_info = find_last_tensor_string_in_one_line(def_line)
                                shape, dtype = extract_tensor_shape_dtype(tensor_info)
                                input_details.append({"shape": shape, "dtype": dtype, "level":"reg"})
                                break

                elif input_item[0].isdigit():
                    # print("input_item",input_item)
                    # print(input_item)
                    # Handling numeric constants
                    dtype = "float16" if "f16" in input_item else ("float32" if "f32" in input_item else "int8")
                    input_details.append({"shape": [1], "dtype": dtype, "level":"ddr"})

            break
        line_counter+=1

    if not operation_found:
        print("current_relay",current_relay)
        raise ValueError(f"Operation '{op_name}' not found after line {current_line}")

    output_tensor_info = find_last_tensor_string_in_one_line(lines[founded_line])
    output_shape, output_dtype = extract_tensor_shape_dtype(output_tensor_info)
    output_level="reg"

    op_class.output_shape=output_shape
    op_class.output_bytes=dtype_to_bytes[output_dtype]
    op_class.output_level=output_level

    # print("input_details",input_details)
    # print("founded_line",founded_line)
    # print(len(input_details))

    input_counter=0

    for input_tensor in input_details:
        if(input_counter==0):
            op_class.input1_shape=input_tensor["shape"]
            op_class.input1_bytes=dtype_to_bytes[input_tensor["dtype"]]
            op_class.input1_level=input_tensor["level"]
        if(input_counter==1):
            op_class.input2_shape=input_tensor["shape"]
            op_class.input2_bytes=dtype_to_bytes[input_tensor["dtype"]]
            op_class.input2_level=input_tensor["level"]

        input_counter+=1
    founded_line+=1

    op_class.op_special_metric_dict=extracted_metrics

    return founded_line







# 定义通用的执行函数
def extract_operation_arg(single_op_name, current_relay, current_line):
    
    relay_op_name=welder_name_to_relay_name[single_op_name]
    single_op=Operation(relay_op_name)
    new_line =extract_single_op_shape_in_relay(current_relay, current_line, relay_op_name, single_op)

    return single_op,new_line
    pass
    # handler = op_handlers.get(single_op_name)
    # if handler:
    #     # 接收处理函数的返回值
    #     result = handler(single_op_name, reg_fused_nodes, current_relay, current_line, row_panel)
    #     return result  # 将处理函数的返回值返回给调用者
    # else:
    #     raise ValueError(f"No handler found for operation: {single_op_name}")


    
def update_reg_fused_global_arg(global_input_desc_list, global_output_desc_list, global_arg_list, reg_operation_list, reg_fused_op_name):
    # print("global_arg_list",global_arg_list)
    # print("global_input_desc_list",global_input_desc_list)
    # print("global_output_desc_list",global_output_desc_list)
    # 处理输入描述
    for index in range(len(global_input_desc_list)):
        # temp_op_name = global_input_desc_list[index][0]
        temp_op_name = global_arg_list[index]['reg_fused_op_name']
        # # # print("temp_op_name",temp_op_name,"reg_fused_op_name",reg_fused_op_name)
        # # re_temp_op_name = re.sub(r'_[0-9]+$', '', temp_op_name)
        # # re_reg_fused_op_name = re.sub(r'_[0-9]+$', '', reg_fused_op_name)
        # # print("re_temp_op_name",re_temp_op_name,"re_reg_fused_op_name",re_reg_fused_op_name)
        
        # 用 "__" 分割 temp_op_name
        split_temp_op_names = temp_op_name.split("__")
        # 对分割出来的每个子字符串应用 re.sub 规则
        re_temp_op_names = [re.sub(r'_[0-9]+$', '', part) for part in split_temp_op_names]

        # 对 reg_fused_op_name 应用 re.sub 规则
        re_reg_fused_op_name = re.sub(r'_[0-9]+$', '', reg_fused_op_name)

        print("re_temp_op_names", re_temp_op_names, "re_reg_fused_op_name", re_reg_fused_op_name)

        flag = 0

        # 如果任意一个 re_temp_op_names 中的子字符串与 re_reg_fused_op_name 匹配就进入循环
        if any(re_temp_op_name == re_reg_fused_op_name for re_temp_op_name in re_temp_op_names):

        # flag=0
        # if re_temp_op_name == re_reg_fused_op_name:



            # print("re_temp_op_name",re_temp_op_name,"re_reg_fused_op_name",re_reg_fused_op_name)
            # print("index",index,"global_arg_list[index]",global_arg_list[index])
            tensor_shape = global_arg_list[index]['tensor_shape']
            tensor_dtype = global_arg_list[index]['dtype']
            # print("tensor_shape",tensor_shape,"dtype_to_bytes[tensor_dtype]",dtype_to_bytes[tensor_dtype])
            
            
            for operation in reg_operation_list:
                inputs = [('input1', operation.input1_shape, operation.input1_bytes, 'input1_level'),
                          ('input2', operation.input2_shape, operation.input2_bytes, 'input2_level')]
                
                for input_name, input_shape, input_bytes, input_level in inputs:
                    if tensor_shape == input_shape and dtype_to_bytes[tensor_dtype] == input_bytes and flag==0:
                        if getattr(operation, input_level) == "smem":
                            setattr(operation, input_level, "ddr")
                            flag=1
                            break  # 找到匹配项后退出内层循环

    # 处理输出描述
    offset = len(global_input_desc_list)
    for index in range(len(global_output_desc_list)):
        # # temp_op_name = global_output_desc_list[index][0]
        # temp_op_name = global_arg_list[index + offset]['reg_fused_op_name']
        # # print("temp_op_name",temp_op_name,"reg_fused_op_name",reg_fused_op_name)
        # re_temp_op_name = re.sub(r'_[0-9]+$', '', temp_op_name)
        # re_reg_fused_op_name = re.sub(r'_[0-9]+$', '', reg_fused_op_name)
        # # print("re_temp_op_name",re_temp_op_name,"re_reg_fused_op_name",re_reg_fused_op_name)
        # flag=0
        # if re_temp_op_name == re_reg_fused_op_name:
        

        # 用 "__" 分割 temp_op_name
        split_temp_op_names = temp_op_name.split("__")
        # 对分割出来的每个子字符串应用 re.sub 规则
        re_temp_op_names = [re.sub(r'_[0-9]+$', '', part) for part in split_temp_op_names]

        # 对 reg_fused_op_name 应用 re.sub 规则
        re_reg_fused_op_name = re.sub(r'_[0-9]+$', '', reg_fused_op_name)

        print("re_temp_op_names", re_temp_op_names, "re_reg_fused_op_name", re_reg_fused_op_name)

        flag = 0

        # 如果任意一个 re_temp_op_names 中的子字符串与 re_reg_fused_op_name 匹配就进入循环
        if any(re_temp_op_name == re_reg_fused_op_name for re_temp_op_name in re_temp_op_names):

            # print("output re_temp_op_name",re_temp_op_name,"re_reg_fused_op_name",re_reg_fused_op_name)
            tensor_shape = global_arg_list[index + offset]['tensor_shape']
            tensor_dtype = global_arg_list[index + offset]['dtype']

            for operation in reg_operation_list:
                if tensor_shape == operation.output_shape and dtype_to_bytes[tensor_dtype] == operation.output_bytes and flag==0:
                    if operation.output_level == "smem":
                        operation.output_level = "ddr"
                        flag=1
                        break  # 找到匹配项后退出循环

# 注意：这个示例假设reg_operation_list中的每个操作都有input1_shape, input1_bytes, input1_level, input2_shape, input2_bytes, input2_level,
# output_shape, output_bytes, output_level等属性。你需要根据你的实际情况来调整属性名和逻辑。



def process_reg_fused_nodes(reg_fused_nodes,relay_list,supported_ops,row_panel,
                            global_arg_op_mapping_list,global_input_desc_list,global_output_desc_list,chip):
    # print("reg_fused_nodes",reg_fused_nodes)
    reg_fused_op_name=reg_fused_nodes['node_name']
    # node_name, block, thread
    # print("reg_fused_nodes['node_name']",reg_fused_nodes['node_name'])
    single_op_name_list,reg_fused_op_id=parse_reg_fused_ops(reg_fused_nodes['node_name'],supported_ops) #op_id from 0 to the last; op_id =reg_fused_op_id
    # # reg_fused_nodes {'node_name': 'subtract_multiply_2', 'block': [1, 16, 1, 256], 'thread': [1, 1, 1, 128], 'step': [1, 1, 1, 2]}
    current_relay=relay_list[reg_fused_op_id]
    # print("reg_fused_op_id",reg_fused_op_id)
    # print("relay_list[reg_fused_op_id]",relay_list[reg_fused_op_id])
    # print("reg_fused_op_name",reg_fused_op_name)
    # print("reg_fused_nodes",reg_fused_nodes)
    # print("current_relay",current_relay)
    '''
    reg_fused_nodes 
    {'node_name': 'reshape_transpose_reshape_nn_bias_add_6', 'block': [1, 32, 1, 256], 'thread': [1, 16, 1, 8], 'step': [1, 1, 1, 2]}
    
    current_relay 
%272 = fn (%p0136: Tensor[(32, 4194304), float16] /* ty=Tensor[(32, 4194304), float16] */, %p1114: Tensor[(32), float16] /* ty=Tensor[(32), float16] */) -> Tensor[(64, 32, 256, 256), float16] {
    %248 = reshape(%p0136, newshape=[32, 64, 256, 256]) /* ty=Tensor[(32, 64, 256, 256), float16] */;
    %249 = transpose(%248, axes=[1, 0, 2, 3]) /* ty=Tensor[(64, 32, 256, 256), float16] */;
    %250 = reshape(%249, newshape=[64, 32, 256, 256]) /* ty=Tensor[(64, 32, 256, 256), float16] */;
    nn.bias_add(%250, %p1114) /* ty=Tensor[(64, 32, 256, 256), float16] span=/encoders.0/encoders.0.0/conv1/Conv:0:0 */
  } /* ty=fn (Tensor[(32, 4194304), float16], Tensor[(32), float16]) -> Tensor[(64, 32, 256, 256), float16] */;
    '''
    current_line=0
    reg_operation_list=[]
    for single_op_name in single_op_name_list:
        # print("single_op_name",single_op_name)
        returned_op,current_line=extract_operation_arg(single_op_name,current_relay,current_line)
        # print("returned_op_list",returned_op_list)
        # reg_operation_list.extend(returned_op_list)
        reg_operation_list.append(returned_op)
    
    # print(len(operation_list))


    relay_lines = current_relay.split("\n")
    output_tensor_str_list=find_tensor_strings_after_arrow(relay_lines[-1])

    # 正序有些问题,还是倒序好-------------------------------------------------------------------
    # output_index=0

    # for output_tensor_str in output_tensor_str_list:
    #     outshape, dtype=extract_tensor_shape_dtype(output_tensor_str)
    #     output_bytes=dtype_to_bytes[dtype]

    #     idx=output_index

    #     for op in operation_list[output_index:] :
    #         idx+=1
    #         print("before modify")
    #         print(op)

    #         if(op.output_shape==outshape and op.output_bytes==output_bytes):
    #             op.output_level="smem"
    #             output_index=idx
    #             print("after modify")
    #             print(op)
    #             break
    # -----------------------------------------------------------------------------------------------

    output_index = len(reg_operation_list) - 1  # 初始化为最后一个元素的索引

    for output_tensor_str in reversed(output_tensor_str_list):
        outshape, dtype = extract_tensor_shape_dtype(output_tensor_str)  # 提取形状和数据类型
        output_bytes = dtype_to_bytes[dtype]  # 计算字节大小

        for idx, op in enumerate(reversed(reg_operation_list[:output_index + 1]), 1):
            # print("before modify")
            # print(op)

            if op.output_shape == outshape and op.output_bytes == output_bytes :
                op.output_level = "smem"  # 标记为使用共享内存
                output_index -= idx  # 更新索引，向前移动
                # print("after modify")
                # print(op)
                break
    
    # print("Before update_reg_fused_global_arg")
    # for op in reg_operation_list:
    #     print(op)

    # print("global_input_desc_list",global_input_desc_list)
    # print("global_output_desc_list",global_output_desc_list)
    # print("global_arg_op_mapping_list",global_arg_op_mapping_list)
    # print("reg_operation_list",reg_operation_list)
    # print("reg_fused_op_name",reg_fused_op_name)

    update_reg_fused_global_arg(global_input_desc_list, global_output_desc_list, global_arg_op_mapping_list, reg_operation_list, reg_fused_op_name)

    print("After update_reg_fused_global_arg")
    for op in reg_operation_list:
        print(op)

    # TODO: we shall handle each single op's schedule/tiling config/ in each single op's handler function.

    if len(reg_operation_list)!=len(single_op_name_list):
        raise ValueError(f"len(reg_operation_list) != len(single_op_name_list), please check!!!")

    # print("reg_fused_nodes",reg_fused_nodes)

    reg_fused_welder_config=Welder_Config(reg_fused_nodes,row_panel)
    # print("Welder Config of reg_fused_welder_config:",reg_fused_welder_config)

    # return ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io
    # print("ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io")
    # return modeling_raw_tuple
    reg_post_list=[]

    for index in range(len(reg_operation_list)):
        single_op_class=reg_operation_list[index]
        reg_post_data=dispatch_to_modeling(single_op_class,reg_fused_welder_config,chip)
        if reg_post_data is None:
            print(f"Error in dispatch_to_modeling, please check!!!")
        else:
            # print("reg_post_data in process_reg_fused_nodes:",reg_post_data)
            reg_post_list.append(reg_post_data)

    time_added,smem_to_process_data=reg_fusion(reg_post_list,chip)

    

    return time_added,smem_to_process_data



    pass