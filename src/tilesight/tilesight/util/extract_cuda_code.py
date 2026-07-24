import json
import os

def extract_cuda_code(json_filepath):
    # 读取JSON文件
    with open(json_filepath, 'r') as file:
        data = json.load(file)

    # 获取JSON文件所在的目录
    directory = os.path.dirname(json_filepath)

    # 遍历每一项数据
    for item in data:
        # 提取需要的数据
        node_names = item.get("node_names")
        group_id = item.get("group_id")
        code = item.get("code")
        
        # 打印节点名称和组ID
        print("Node Names:", node_names)
        print("Group ID:", group_id)

        # 将代码中的\n替换为实际的换行符，并保存到文件中
        code_with_newlines = code.replace("\\n", "\n")
        filename = f"code_{group_id}_{node_names}.cu"  # 创建文件名
        full_path = os.path.join(directory, filename)  # 创建文件的完整路径

        with open(full_path, 'w') as code_file:
            code_file.write(code_with_newlines)
        print(f"Code has been saved to {full_path}")

# 检查是否作为主程序运行
if __name__ == '__main__':
    # 调用函数，传入JSON文件的路径
    extract_cuda_code("ladder_tuning_model_json/tuned.json")
