from .arch_base import Arch
# CDNA3 Whitepaper
# And check this
# https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md
class MI300X(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments
        
        # After calling the base class constructor, set the properties
        self.core = "MI300X"
        self.sm_count = 304
        self.base_freq = 1.64 * 1e9
        self.max_freq = 1.64 * 1e9
        # omniperf: 10241348.5 cycles, 7779381.5ns, 1.316GHz
        # self.base_freq = 2.1 * 1e9
        # self.max_freq = 2.1 * 1e9

        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (8, 4, 8)
        self.tensor_core_flops = 512
        # self.fp32_cores_per_sm = 64
        self.fp32_cores_per_sm = 128 # considering packed fp32, a easy to use optimization, which makes fp32 cores double
        self.ddr_bandwidth = 5300 * 1e9
        self.ddr_capacity = 192 * (1024**3)
        # self.l2_bandwidth = 5288 * 1e9 ?????????????
        # The L2 cache is shared across the whole chip and physically partitioned into multiple slices. For the MI100, the cache is 16-way set
            # associative and comprises 32 slices (twice as many as in MI50) in total for an aggregate capacity of 8MB. Each slice can sustain 64B/cycle
            # for an aggregate bandwidth over 3TB/s across the GPU.
        self.l2_bandwidth = 10600 * 1e9
        self.l2_capacity = 4 * (1024**2) # effective cap here; A100 with dup
        # amd's wavefronts are 64 threads, 10 wavefronts per CU, so 640 threads per CU
        self.sm_sub_partitions = 4
        # self.l1_smem_throughput_per_cycle = 128 / 1.33
        
        # 64KB Local Data Share (LDS, or shared memory)
        # 16 KB Read/Write L1 vector data cache
        self.l1_smem_throughput_per_cycle = 128 # 110CUs*1700MHz*4byte*32bank=23.936TB/s, with info by omniperf
        # vL1D 11.968, sL1D 6.092, iL1D 6.092, TB/s
        self.configurable_smem_capacity = 64 * (1024**1)
        # MI2xx with: 12.5 KiB SGPRs, 256 KiB VGPRs, 256 KiB AGPRs per CU
        # GFX9 features large register files. Registers are DWORD-sized (4 B), and are split into 3 general groups:
        # SGPRs: Scalar registers (uniform value within subgroup threads). Up to 104 SGPRs per workgroup on MI300.
        # VGPRs: General-purpose vector registers (each thread holds a different value). Up to 256 VGPRs per thread on MI300.
        # AGPRs: Matrix accumulation vector registers (each thread holds a different value). Up to 256 AGPRs per thread on MI300.
        self.register_capacity_per_sm = 256 * (1024**1) ##
         
        self.warp_schedulers_per_sm = 1
        self.sfu_cores_per_sm  = 16
        # self.fp16_tensor_flops = 311.87 * 1e12
        # self.fp32_cuda_core_flops = 19.49 * 1e12
        
        # Now calculate the derived properties
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        # self.int8_tensor_flops = self.fp16_tensor_flops * 2 
        # self.int8_int2_flops = self.fp16_tensor_flops * 4
        # self.int8_int1_flops = self.fp16_tensor_flops * 2
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 2
        self.fp64_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 0.5
        self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm * 2 
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

        self.ddr_max_util=0.9
        self.l2_max_util=0.9
        self.l1_max_util=0.9
        self.compute_max_util=0.9

        # self.fp16_tensor_flops=168e12
        # self.ddr_bandwidth = 1406 * 1e9
        # self.smem_bandwidth  = 14691.22 * 1e9
        # # self.smem_bandwidth = 17556 * 1e9 by omniperf
        # self.fp32_cuda_core_flops = 35.245e12
        # self.fp16_cuda_core_flops = 35.245e12
        


        # self.ddr_max_util=0.9
        # self.l2_max_util=0.9
        # self.l1_max_util=0.9
        # self.compute_max_util=0.9

    def set_to_microbench(self):
        self.base_freq = 1.64 * 1e9
        self.max_freq = 1.64 * 1e9
        # self.ddr_max_util=0.9
        # self.l2_max_util=0.9
        # self.l1_max_util=0.9
        # self.compute_max_util=0.9
        self.ddr_max_util=1.0
        self.l2_max_util=1.0
        self.l1_max_util=1.0
        self.compute_max_util=1.0

        self.ddr_bandwidth = 3816.48 * 1e9
        # self.l2_bandwidth= 16629.11 *1e9 
        self.l2_bandwidth= 10000 *1e9 
        # self.smem_bandwidth = 50160.04 * 1.64 / 2.1 * 1e9
        self.smem_bandwidth = 50160.04 * 1e9


        self.fp32_cuda_core_flops = 108.40 * 1e12
        # self.fp16_cuda_core_flops = 44.43 * 1e12
        # self.fp64_cuda_core_flops = 23.74 * 1e12
        self.fp16_cuda_core_flops = 108.40 * 1e12 * 2
        self.fp64_cuda_core_flops = 108.40 * 1e12 * 0.5
        self.fp16_tensor_flops = 962.498 * 1e12

        return self

    def set_to_spec(self):
        self.ddr_max_util=1.0
        self.l2_max_util=1.0
        self.l1_max_util=1.0
        self.compute_max_util=1.0
        self.base_freq = 2.1 * 1e9
        self.max_freq = 2.1 * 1e9
        
        return self
