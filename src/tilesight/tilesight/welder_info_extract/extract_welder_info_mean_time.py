import os

def extract_welder_info_mean_time(file_path):
    """
    提取给定文件中的 mean 值。
    
    参数：
    file_path (str): 文件路径
    
    返回值：
    mean_value (float or None): 提取到的 mean 值，如果未找到则为 None
    found (bool): 是否找到 "Execution time summary"
    """
    # mean_value = None
    median_value = None
    found = False

    # 检查文件是否存在
    if not os.path.exists(file_path):
        return median_value, found

    # 读取文件
    with open(file_path, 'r') as file:
        # 读取所有行
        lines = file.readlines()

        # 从文件末尾开始查找包含 "Execution time summary" 的行
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith("Execution time summary"):
                # 查找到 "Execution time summary" 行
                # 读取下一行的时间数据
                time_data_line = lines[i + 2].strip()  # 跳过标题行
                # 按空格分割并提取 mean 值
                time_data = [float(x) for x in time_data_line.split() if x.strip()]
                # mean_value = time_data[0]  # mean 值是第一个
                median_value = time_data[1]
                found = True
                break

    # return mean_value, found
    return median_value, found
