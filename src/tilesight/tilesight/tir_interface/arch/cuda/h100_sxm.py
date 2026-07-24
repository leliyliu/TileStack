from ..base_arch import Arch, ComputeUnit


class H100_SXM(Arch):

    def __init__(self):

        super().__init__()
        self.set_to_spec()

    def set_to_spec(self):
        """
        Configures H100_PCIE to its official specification parameters.
        This method sets the hardware's fixed parameters and instantiates ComputeUnit objects.
        """
        self.core = "H100_SXM"
        self.sm_count = 132
        self.freq = 1.83 * 1e9  # Core frequency (in Hz); subclasses will set this.

        self.ddr_bandwidth = 3352.32 * 1e9
        self.ddr_capacity = 90 * (1024**3)
        self.l2_bandwidth = 8748e9  # 96* self.max_freq * 2 * 32 # no compression, 2 sectors/cycle?, 95???
        self.l2_capacity = 50 * (1024**2)  # H100 with dup
        self.l2_partitions = 2

        self.sm_sub_partitions = 4
        self.l1_smem_throughput_per_cycle = 128  # Bytes per cycle per SM
        self.configurable_smem_capacity = 228 * (1024**1)  # Bytes
        self.register_capacity_per_sm = 256 * (1024**1)  # Bytes

        # ComputeUnit instances will be assigned by subclasses.
        self.fp16_tensor_core_unit = ComputeUnit(
            unit_type="tensor",
            dtype="fp16",
            instances=4,  # 4 tensor cores per SM
            shape=[8, 4, 16],  # M, N, K shape for tensor ops
            ops_per_element=2  # FMA for tensor cores
        )

        self.fp8_tensor_core_unit = ComputeUnit(
            unit_type="tensor", dtype="fp8", instances=4, shape=[8, 4, 32], ops_per_element=2)
        self.fp32_cuda_core_unit = ComputeUnit(
            unit_type="vector",
            dtype="fp32",
            instances=128,  # 128 FP32 cores per SM
            shape=[1],  # No specific operational shape for scalar/vector cores
            ops_per_element=2  # FMA for FP32
        )
        self.fp16_cuda_core_unit = ComputeUnit(
            unit_type="vector",
            dtype="fp16",
            instances=128,  # Assuming same count as FP32, adjust if different
            shape=[1],
            ops_per_element=2)
        # Assuming H100 has FP64 CUDA Cores
        self.fp64_cuda_core_unit = ComputeUnit(
            unit_type="vector",
            dtype="fp64",
            instances=64,  # Assuming half the count of FP32 for FP64, adjust if different
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

        # TODO: Not tested yet

        return self

    def set_to_ncu(self):
        """
        Configures H100_PCIE based on NCU observed parameters.
        This method adjusts parameters to match real-world profiling data from NCU.
        """
        self.freq = 1.44 * 1e9  # Set operating frequency based on NCU data

        self.ddr_max_util = 0.9
        self.l2_max_util = 0.9
        self.l1_max_util = 0.9
        self.compute_max_util = 0.9

        # --- Re-calculate all raw performance metrics ---
        self.compute_metrics()
        return self
