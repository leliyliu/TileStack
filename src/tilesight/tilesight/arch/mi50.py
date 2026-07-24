# 文件名: a100.py
from .arch_base import Arch
# https://www.techpowerup.com/gpu-specs/radeon-instinct-mi50.c3335
# Vega 20, GFX ID=gfx906 
class MI50(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments
        
        # After calling the base class constructor, set the properties
        self.core = "MI50"
        self.sm_count = 60
        self.base_freq = 1.20 * 1e9
        self.max_freq = 1.20 * 1e9
        # self.base_freq = 1.75 * 1e9
        # self.max_freq = 1.75 * 1e9

        # self.tensor_cores_per_sm = 4
        # self.tensor_core_shape = (8, 4, 8)
        # self.tensor_core_flops = 512
        self.fp32_cores_per_sm = 64
        self.ddr_bandwidth = 1024 * 1e9
        self.ddr_capacity = 16 * (1024**3)
        # self.l2_bandwidth = 5288 * 1e9 ?????????????
        # The L2 cache is shared across the whole chip and physically partitioned into multiple slices. For the MI100, the cache is 16-way set
            # associative and comprises 32 slices (twice as many as in MI50) in total for an aggregate capacity of 8MB. Each slice can sustain 64B/cycle
            # for an aggregate bandwidth over 3TB/s across the GPU.
        self.l2_bandwidth = 1500 * 1e9
        self.l2_capacity = 4 * (1024**2) # effective cap here; A100 with dup
        # amd's wavefronts are 64 threads, 10 wavefronts per CU, so 640 threads per CU
        self.sm_sub_partitions = 1
        # self.l1_smem_throughput_per_cycle = 128 / 1.33
        
        # 64KB Local Data Share (LDS, or shared memory)
        # 16 KB Read/Write L1 vector data cache
        self.l1_smem_throughput_per_cycle = 64 # just guess
        self.configurable_smem_capacity = 64 * (1024**1)
        self.register_capacity_per_sm = 128 * (1024**1) ## or 256?
        self.warp_schedulers_per_sm = 1
        # self.sfu_cores_per_sm  = 16
        # self.fp16_tensor_flops = 311.87 * 1e12
        # self.fp32_cuda_core_flops = 19.49 * 1e12
        
        # Now calculate the derived properties
        # self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        # self.fp16_tensor_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        # self.int8_tensor_flops = self.fp16_tensor_flops * 2 
        # self.int8_int2_flops = self.fp16_tensor_flops * 4
        # self.int8_int1_flops = self.fp16_tensor_flops * 2
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 2
        self.fp64_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 0.5
        # self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm * 2 
        # self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        # self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

        self.ddr_max_util=0.9
        self.l2_max_util=0.9
        self.l1_max_util=0.9
        self.compute_max_util=0.9
