from .arch_base import Arch
import math

class A100_LUT(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments
        
        # After calling the base class constructor, set the properties
        self.core = "A100_LUT"
        self.sm_count = 108
        self.base_freq = 1.41 * 1e9
        self.max_freq = 1.41 * 1e9
        self.tensor_cores_per_sm = 4
        # self.tensor_core_shape = (8, 64, 4)
        # self.tensor_core_shape = (2, 64, 4)
        self.tensor_core_shape = (8, 4, 8)
        self.fp32_cores_per_sm = 64
        self.ddr_bandwidth = 1935 * 1e9
        self.ddr_capacity = 80 * (1024**3)
        self.l2_bandwidth = 5288 * 1e9 * 1 * 0.75
        self.l2_capacity = 30 * (1024**2) # effective cap here; A100 with dup
        self.sm_sub_partitions = 4
        self.l1_smem_throughput_per_cycle = 128
        self.configurable_smem_capacity = 164 * (1024**1)
        self.register_capacity_per_sm = 256 * (1024**1)
        self.warp_schedulers_per_sm = 4
        # self.fp16_tensor_flops = 311.87 * 1e12
        # self.fp32_cuda_core_flops = 19.49 * 1e12
        
        # Now calculate the derived properties
        self.tensor_core_flops = math.prod(self.tensor_core_shape)*2
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        # print("fp16_tensor_flops: ", self.fp16_tensor_flops)
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 2
