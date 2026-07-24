# 文件名: llama2_13b.py
from .onnx_model_base import OnnxModel

class Opt_2dot7b(OnnxModel):
    def __init__(self, seq=1, num_layer=32):
        super().__init__()
        self.seq = seq
        self.num_layer = num_layer

        #Embedding for only the beginning
        self.add_reduce_op(seq+50272*2560, seq*2560, 1)
        self.add_reduce_op(seq+2560*2560, seq*2560, 1)
        self.add_elementwise_op(seq, seq, 8)
        self.add_elementwise_op(seq*seq, seq*seq, 2)
        self.add_elementwise_op(seq, seq, 7)
        self.add_elementwise_op(1, 1, 8)

        #the body of 40 layers 

        self.add_elementwise_op(seq*2560, seq*2560, 37.5*num_layer)
        self.add_elementwise_op(seq*seq*32, seq*seq*32, 11*num_layer)
        self.add_reduce_op(seq*2560+2560, seq*2560, 8*num_layer)

        self.add_matmul_op(seq, 2560, 2560, 3*num_layer)
        self.add_matmul_op(seq, seq, 128, 32*num_layer)
        self.add_matmul_op(seq, 128, seq, 32*num_layer)
        self.add_matmul_op(seq, 2560, 2560, 1*num_layer)
        self.add_matmul_op(seq, 10240, 2560, 1*num_layer)
        self.add_matmul_op(seq, 2560, 10240, 1*num_layer)

        #lm_head
        self.add_elementwise_op(seq*2560, seq*2560, 11)
        self.add_reduce_op(seq*2560+2560, seq*2560, 2)
        self.add_matmul_op(seq, 50272, 2560, 1)
        

