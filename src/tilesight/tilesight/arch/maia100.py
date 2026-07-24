# 文件名: a100.py
from .arch_base import Arch

class Maia100(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments
        
        # After calling the base class constructor, set the properties
        self.core = "Maia100"
        self.sm_count = 140
        self.base_freq = 1.4 * 1e9
        self.max_freq = 1.4 * 1e9
        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (8, 4, 16)
        self.tensor_core_flops = 1024
        self.fp32_cores_per_sm = 128
        self.ddr_bandwidth = 1600 * 1e9
        self.ddr_capacity = 64 * (1024**3)
        self.l2_bandwidth = 10000 * 1e9
        self.l2_capacity = 448 * (1024**2) # effective cap here; A100 with dup
        self.sm_sub_partitions = 4
        self.l1_smem_throughput_per_cycle = 128
        self.configurable_smem_capacity = 800 * (1024**1)
        self.register_capacity_per_sm = 256 * (1024**1)
        self.warp_schedulers_per_sm = 4
        # self.fp16_tensor_flops = 311.87 * 1e12
        # self.fp32_cuda_core_flops = 19.49 * 1e12
        
        # Now calculate the derived properties
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

