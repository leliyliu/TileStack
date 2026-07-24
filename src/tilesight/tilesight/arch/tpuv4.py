# 文件名: a100.py
from .arch_base import Arch
# Figure 2 below shows a TPU v4 package and four of them mounted on the printed circuit board. 
# Like TPU v3, each TPU v4 contains two TensorCores (TC). 
# Each TC contains four 128x128 Matrix Multiply Units (MXUs) and a Vector Processing Unit (VPU) with 128 lanes (16 ALUs per lane) 
# and a 16 MiB Vector Memory (VMEM). 
# The two TCs share a 128 MiB Common Memory (CMEM). 
# The PCB embeds 4 Inter-Core Interconnect (ICI) links, connected as a 2×2 mesh; 16 external ICI links go to other trays for constructing the 3D torus. 
# Figure 3 below shows one row of eight racks, where each rack contains 16 tray-host server pairs. 
# Passive electrical cables create a 4×4×4 3D mesh in a rack. 
# Electrical-to -optical conversions happen at the fiber connector to the TPU trays.


# On Chip Memory
# 128 (CMEM) +
# 32 MiB (VMEM) +
# 10 MiB (spMEM)

# Register File Size 0.25 MiB

class TPUv4(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments
        
        # After calling the base class constructor, set the properties
        self.core = "TPUv4"
        self.sm_count = 2
        self.base_freq = 1.05 * 1e9
        self.max_freq = 1.05 * 1e9
        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (128,128,1) #systolic array
        self.tensor_core_flops = 128*128*2
        # self.fp16_tensor_flops = 1.05*1e9*4*128*128*2*2 = 275TFLOPs
        self.fp32_cores_per_sm = 128*16 #just guess, "a Vector Processing Unit (VPU) with 128 lanes (16 ALUs per lane)"
        self.ddr_bandwidth = 1200 * 1e9
        self.ddr_capacity = 32 * (1024**3)
        self.l2_bandwidth = 2400 * 1e9 # just guess
        self.l2_capacity = 128 * (1024**2) # 128 MiB shared across two TCs
        self.sm_sub_partitions = 4
        # self.l1_smem_throughput_per_cycle = 128 / 1.33
        self.l1_smem_throughput_per_cycle = 128 * 16 * 2 # just guess, to match 128lanes with 16ALUs per lane, bf16
        self.configurable_smem_capacity = 16*1024 * (1024**1) # 32 MiB (VMEM) + 10 MiB (spMEM)
        self.register_capacity_per_sm = 128 * (1024**1) # Register File Size 0.25 MiB
        self.warp_schedulers_per_sm = 4
        self.sfu_cores_per_sm  = 128 # just guess = =
        
        # self.fp32_cuda_core_flops = 19.49 * 1e12
        
        # Now calculate the derived properties
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        # self.fp16_tensor_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.int8_tensor_flops = self.fp16_tensor_flops * 1
        # really just guess
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

