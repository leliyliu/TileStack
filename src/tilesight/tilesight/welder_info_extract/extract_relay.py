import re

def extract_relay_list(relay_path,store_txt=False):
    # Load the relay file content
    with open(relay_path, "r") as file:
        relay_content = file.read()

    # Define the pattern to capture the reg_fuse_list sections
    pattern = re.compile(r"(%\d+ = fn.*?\{.*?\n\s*\}.+?;)", re.DOTALL)

    # Find all relay_list of the reg_fuse_list sections
    relay_list = pattern.findall(relay_content)
    # print(relay_list[168])

    if(store_txt==True):
        new_content = "\n\n".join(relay_list)
        if relay_path.endswith('.txt'):
            new_relay_path = relay_path.replace('.txt', '_1.txt')
        elif relay_path.endswith('.cu'):
            new_relay_path = relay_path.replace('.cu', '_1.cu')
        
        with open(new_relay_path, "w") as new_file:
            new_file.write(new_content)        

    return relay_list



# ------------------------------------------------------------------------------------------------
# extract_relay_1_list()
# 读取上传的 Relay 程序文件到内存中
# file_path_1 = 'relay/relay_nafnet_bs64_fp16_1.txt'

# # 读取文件内容
# with open("relay/relay_nafnet_bs64_fp16_1.txt", "r") as file:
#     content = file.read()

# # 使用空行分割内容为多个部分
# blocks = content.split('\n\n')

# # 定义一个函数来根据给定的索引ID访问对应的代码块
# def access_block_by_id(block_id):
#     if 0 <= block_id < len(blocks):
#         return blocks[block_id]
#     else:
#         return "Block ID out of range."

# # 示例：访问索引为2的代码块
# example_block_id = 0
# print(access_block_by_id(example_block_id))




# # 计算每个block的行数
# block_line_counts = [block.count('\n') + 1 for block in blocks]

# # 准备要写入文件的内容
# file_content = "\n".join([f"Block ID {i} has {count} lines." for i, count in enumerate(block_line_counts)])

# # 写入到临时文件tmp.txt
# with open("relay/tmp.txt", "w") as tmp_file:
#     tmp_file.write(file_content)

# print("信息已写入tmp.txt文件。")

# ------------------------------------------------------------------------------------------------