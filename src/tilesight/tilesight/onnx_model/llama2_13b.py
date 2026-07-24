# 文件名: llama2_13b.py
from .onnx_model_base import OnnxModel

class Llama2_13b(OnnxModel):
    def __init__(self, seq=1, num_layer=40):
        super().__init__()
        self.seq = seq
        self.num_layer = num_layer

        #Embedding for only the beginning
        self.add_reduce_op(seq+32000*5120, seq*5120, 1)

        #the body of 40 layers 

        self.add_elementwise_op(seq*5120, seq*5120, 56*num_layer)
        self.add_elementwise_op(seq*seq*40, seq*seq*40, 7*num_layer)
        self.add_matmul_op(seq, 5120, 5120, 3*num_layer)
        self.add_matmul_op(seq, seq, 128, 40*num_layer)
        self.add_matmul_op(seq, 128, seq, 40*num_layer)
        self.add_matmul_op(seq, 5120, 5120, 1*num_layer)
        self.add_matmul_op(seq, 13824, 5120, 2*num_layer)
        self.add_matmul_op(seq, 5120, 13824, 1*num_layer)

        #lm_head
        self.add_elementwise_op(seq*5120, seq*5120, 9)
        self.add_reduce_op(seq*32000, seq*32000, 1)
        self.add_matmul_op(seq, 32000, 5120, 1)
        

    # ... 其他可能的特定于 Llama2_7b 的方法和属性
