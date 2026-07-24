from ..base_arch import Arch, ComputeUnit


class MI250X(Arch):
    # actually it's 2x MI210? controlled separately
    def __init__(self):

        super().__init__()
        self.set_to_spec()

    def set_to_spec(self):

        self.core = "MI250X"
        self.sm_count = 110
        self.freq = 2.1 * 1e9  # Core frequency (in Hz); subclasses will set this.

        self.ddr_bandwidth = 1600 * 1e9
        self.ddr_capacity = 64 * (1024**3)
        self.l2_bandwidth = 3200 * 1e9  # 96* self.max_freq * 2 * 32 # no compression, 2 sectors/cycle?, 95???
        self.l2_capacity = 8 * (1024**2)  # H100 with dup
        self.l2_partitions = 1

        self.sm_sub_partitions = 4
        self.l1_smem_throughput_per_cycle = 128  # Bytes per cycle per SM
        self.configurable_smem_capacity = 164 * (1024**1)  # Bytes
        self.register_capacity_per_sm = 256 * (1024**1)  # Bytes

        # ComputeUnit instances will be assigned by subclasses.
        self.fp16_tensor_core_unit = ComputeUnit(
            unit_type="tensor",
            dtype="fp16",
            instances=4,  # 4 tensor cores per SM
            shape=[8, 4, 8],  # M, N, K shape for tensor ops
            ops_per_element=2  # FMA for tensor cores
        )

        self.fp32_cuda_core_unit = ComputeUnit(
            unit_type="vector",
            dtype="fp32",
            instances=64,  # 128 FP32 cores per SM
            shape=[1],  # No specific operational shape for scalar/vector cores
            ops_per_element=2  # FMA for FP32
        )
        self.fp16_cuda_core_unit = ComputeUnit(
            unit_type="vector",
            dtype="fp16",
            instances=64,  # Assuming same count as FP32, adjust if different
            shape=[1],
            ops_per_element=2)
        # Assuming H100 has FP64 CUDA Cores
        self.fp64_cuda_core_unit = ComputeUnit(
            unit_type="vector",
            dtype="fp64",
            instances=32,  # Assuming half the count of FP32 for FP64, adjust if different
            shape=[1],
            ops_per_element=2)

        self.sfu_unit = ComputeUnit(
            unit_type="vector",
            dtype="fp32",  # SFU might handle FP32, adjust dtype if needed
            instances=16,  # 16 SFU per SM
            shape=[1],
            ops_per_element=2)

        # Utilization limits
        self.ddr_max_util = 0.9
        self.l2_max_util = 0.9
        self.l1_max_util = 0.9
        self.compute_max_util = 0.9

        self.compute_metrics()

        return self

    def set_to_microbench(self):

        self.freq = 1.36 * 1e9  # Set operating frequency

        # Update utilization if microbenchmarks hit different limits
        self.ddr_max_util = 1.0
        self.l2_max_util = 1.0
        self.l1_max_util = 1.0
        self.compute_max_util = 1.0

        # Update bandwidths if microbenchmarks show different achieved rates

        self.ddr_bandwidth = 1654.81 * 1e9
        self.l2_bandwidth = 3234.77 * 1e9
        self.smem_bandwidth = 19491 * 1e9

        self.fp32_cuda_core_flops = 19.017181 * 1e12
        self.fp16_cuda_core_flops = 19.017181 * 1e12
        # self.fp64_cuda_core_flops =
        self.fp16_tensor_flops = 298.951 * 1e12

        return self

    def set_to_ncu(self):
        """
        Configures H100_PCIE based on NCU observed parameters.
        This method adjusts parameters to match real-world profiling data from NCU.
        """
        self.freq = 1.06 * 1e9  # Set operating frequency based on NCU data

        self.ddr_max_util = 0.9
        self.l2_max_util = 0.9
        self.l1_max_util = 0.9
        self.compute_max_util = 0.9

        # --- Re-calculate all raw performance metrics ---
        self.compute_metrics()
        return self
