def update_operation_levels(smem_fused_args, smem_fused_op_list):
    blocks = []
    for op_list_idx, reg_fused_op_list in enumerate(smem_fused_op_list):
        for reg_op_idx, operation in enumerate(reg_fused_op_list):
            for shape_idx in range(3):  # 0: input1, 1: input2, 2: output
                blocks.append([op_list_idx, reg_op_idx, shape_idx])

    # 初始化下一轮搜索的起始索引为blocks的末尾
    next_start_index = len(blocks) - 1

    for shape_value_dict in reversed(smem_fused_args):
        shape_value = shape_value_dict['shape']
        found = False

        # 从记录的下一个起始位置开始逆序搜索
        for idx in reversed(range(0, next_start_index + 1)):
            op_list_idx, reg_op_idx, shape_idx = blocks[idx]
            operation = smem_fused_op_list[op_list_idx][reg_op_idx]
            shape, level_attr = [(operation.input1_shape, 'input1_level'), (operation.input2_shape, 'input2_level'), (operation.output_shape, 'output_level')][shape_idx]

            if shape == shape_value and getattr(operation, level_attr) == "smem":
                setattr(operation, level_attr, "ddr")
                found = True
                # print(f"Found match for {shape_value} at {blocks[idx]}, setting {level_attr} to 'ddr'")
                # 更新下一轮搜索的起始索引为当前找到匹配项的前一个位置
                next_start_index = idx - 1
                break  # 找到匹配项，跳出循环

        if not found:
            raise ValueError(f"当前shape_value未找到匹配项: {shape_value}")
