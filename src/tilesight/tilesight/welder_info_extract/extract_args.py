import re

# 定义一个函数来处理和解析字符串
def process_welder_args(args_str):
    # 定义正则表达式匹配Tensor对象的shape和op.name
    tensor_pattern = re.compile(r"Tensor\(shape=\[(.*?)\], op.name=(.*?)\)")
    
    # 使用正则表达式找到所有匹配项
    matches = tensor_pattern.findall(args_str)
    
    # 初始化结果列表
    result = []
    
    # 遍历所有匹配项，将每个项的shape和op.name提取出来，然后添加到结果列表中
    for match in matches:
        shape_str, op_name = match
        # 将shape字符串转换为整数列表
        shape = [int(dim) for dim in shape_str.split(", ")]
        result.append({"shape": shape, "op.name": op_name})
    
    return result

# # 示例字符串，用于测试
# args_str_example = "[Tensor(shape=[64, 32, 128, 128], op.name=input0), Tensor(shape=[64, 128], op.name=input1), Tensor(shape=[64], op.name=input2), Tensor(shape=[64, 64, 64, 64], op.name=output0)]"

# # 调用函数进行测试（该调用在最终提交时会被注释）
# result = process_welder_args(args_str_example)
# print(result)

# # 由于环境限制，上面的函数调用代码在提交给用户之前应该保持被注释状态。
