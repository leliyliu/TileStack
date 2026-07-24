# 定义解析函数
def parse_reg_fused_ops(s, substrings):
    """
    解析给定的字符串s,根据预定义的子字符串列表进行匹配。
    
    参数:
    - s: 要解析的字符串。
    - substrings: 已知的子字符串列表。
    
    返回:
    - 匹配到的子字符串列表和最后剩余的数字。
    """
    # 初始化匹配到的子字符串列表
    matched_substrings = []
    # 当前处理的字符串
    current_string = s
    
    # 循环直到字符串被完全解析
    while current_string:
        # 假设没有匹配的子字符串
        match_found = False
        
        for substring in substrings:
            # 如果当前字符串以某个子字符串开头
            if current_string.startswith(substring):
                # 添加到匹配列表
                matched_substrings.append(substring)
                # 移除匹配的子字符串和紧随其后的下划线（如果存在）
                current_string = current_string[len(substring):].lstrip("_")
                # 标记找到匹配
                match_found = True
                break  # 退出循环，从新的位置开始匹配
        
        # 如果没有找到匹配，假设剩余部分是数字
        if not match_found:
            if current_string.isdigit():
                return matched_substrings, int(current_string)  # 返回匹配列表和数字
                # return matched_substrings  # 返回匹配列表
            else:
                raise ValueError("剩余字符串不是有效数字: " + current_string)
    
    # 如果完全没有匹配，返回空列表和None
    return matched_substrings, None