# 文件名: llama2_13b.py
from .onnx_model_base import OnnxModel

class Code_Llama_34b(OnnxModel):
    def __init__(self, seq=1, num_layer=48):
        super().__init__()
        self.seq = seq
        self.num_layer = num_layer

        #Embedding for only the beginning
        self.add_reduce_op(seq+32000*8192, seq*8192, 1)

        #the body of 48 layers 

        self.add_elementwise_op(seq*8192, seq*8192, 38*num_layer)
        self.add_elementwise_op(seq*1024, seq*1024, 20*num_layer)
        self.add_elementwise_op(seq*seq*64, seq*seq*64, 7*num_layer)

        self.add_reduce_op(8192, 0, 2*num_layer)

        self.add_matmul_op(seq, 8192, 8192, 2*num_layer)
        self.add_matmul_op(seq, 1024, 8192, 2*num_layer)
        self.add_matmul_op(seq, seq, 128, 64*num_layer)
        self.add_matmul_op(seq, 128, seq, 64*num_layer)

        self.add_matmul_op(seq, 22016, 8192, 2*num_layer)
        self.add_matmul_op(seq, 8192, 22016, 1*num_layer)

        #lm_head
        self.add_elementwise_op(seq*8192, seq*8192, 9)
        self.add_reduce_op(seq*32000, seq*32000, 1)
        self.add_matmul_op(seq, 32000, 8192, 1)
        

