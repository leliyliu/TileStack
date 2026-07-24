# 文件名: onnx_model_base.py
class OnnxModel:
    def __init__(self):
        self.ops_info = {
            'MatMulOp': [],
            'ElementWiseOp': [],
            'ReduceOp': []
        }

    def add_op_info(self, op_type, info):
        self.ops_info[op_type].append(info)

    def add_matmul_op(self, m, n, k, count):
        info = {'dims': (m, n, k), 'count': count}
        self.add_op_info('MatMulOp', info)

    def add_elementwise_op(self, input_size, output_size, count):
        info = {'input_size': input_size, 'output_size': output_size, 'count': count}
        self.add_op_info('ElementWiseOp', info)

    def add_reduce_op(self, input_size, output_size, count):
        info = {'input_size': input_size, 'output_size': output_size, 'count': count}
        self.add_op_info('ReduceOp', info)
