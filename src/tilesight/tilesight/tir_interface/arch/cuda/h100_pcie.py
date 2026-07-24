from ..base_arch import Arch, ComputeUnit


class H100_PCIE(Arch):

    def __init__(self):
        """
        Initializes the H100_PCIE architecture.
        By default, it configures itself to the official specifications.
        """
        super().__init__()
        # Call set_to_spec to configure initial parameters and trigger metric calculation
        self.set_to_spec()

    def set_to_spec(self):
        """
        Configures H100_PCIE to its official specification parameters.
        This method sets the hardware's fixed parameters and instantiates ComputeUnit objects.
        """
        self.core = "H100_PCIE"
        self.sm_count = 114
        self.freq = 1.755 * 1e9  # Core frequency (in Hz); subclasses will set this.

        self.ddr_bandwidth = 2039 * 1e9
        self.ddr_capacity = 80 * (1024**3)
        self.l2_bandwidth = 7598.47e9  # 96* self.max_freq * 2 * 32 # no compression, 2 sectors/cycle?, 95???
        self.l2_capacity = 50 * (1024**2)  # H100 with dup
        self.l2_partitions = 2

        self.sm_sub_partitions = 4
        self.l1_smem_throughput_per_cycle = 128  # Bytes per cycle per SM
        self.configurable_smem_capacity = 228 * (1024**1)  # Bytes
        self.register_capacity_per_sm = 256 * (1024**1)  # Bytes
        self.warp_schedulers_per_sm = 4

        # --- Instantiate ComputeUnit objects for H100 ---
        # These represent the physical compute cores with their specific capabilities.
        self.fp16_tensor_core_unit = ComputeUnit(
            unit_type="tensor",
            dtype="fp16",
            instances=4,  # 4 tensor cores per SM
            shape=[8, 4, 16],  # M, N, K shape for tensor ops
            ops_per_element=2  # FMA for tensor cores
        )
        self.fp8_tensor_core_unit = ComputeUnit(
            unit_type="tensor", dtype="fp8", instances=4, shape=[8, 4, 16], ops_per_element=2)
        # If H100 does NOT have FP4 Tensor Cores, we simply don't instantiate self.fp4_tensor_core_unit.
        # It remains None as initialized in Arch.__init__.

        self.fp32_cuda_core_unit = ComputeUnit(
            unit_type="vector",  # Or "scalar" depending on the precise definition
            dtype="fp32",
            instances=128,  # 128 FP32 cores per SM
            shape=[],  # No specific operational shape for scalar/vector cores
            ops_per_element=2  # FMA for FP32
        )
        self.int32_cuda_core_unit = ComputeUnit(
            unit_type="vector",
            dtype="int32",
            instances=64,  # 64 INT32 cores per SM
            shape=[],
            ops_per_element=2)
        # Assuming H100 has dedicated FP16 CUDA Cores (separate from Tensor Cores)
        self.fp16_cuda_core_unit = ComputeUnit(
            unit_type="vector",
            dtype="fp16",
            instances=128,  # Assuming same count as FP32, adjust if different
            shape=[],
            ops_per_element=2)
        # Assuming H100 has FP64 CUDA Cores
        self.fp64_cuda_core_unit = ComputeUnit(
            unit_type="vector",
            dtype="fp64",
            instances=64,  # Assuming half the count of FP32 for FP64, adjust if different
            shape=[],
            ops_per_element=2)
        self.sfu_unit = ComputeUnit(
            unit_type="sfu",
            dtype="fp32",  # SFU might handle FP32, adjust dtype if needed
            instances=16,  # 16 SFU per SM
            shape=[],
            ops_per_element=2)

        # --- Other Arch-level physical attributes ---
        # These are direct input parameters for the hardware's specifications.

        # Utilization limits
        self.ddr_max_util = 0.9
        self.l2_max_util = 0.9
        self.l1_max_util = 0.9
        self.compute_max_util = 0.9

        # --- Calculate all raw performance metrics based on the set parameters ---
        self.compute_metrics()

        # H100-specific derived properties (now in raw FLOPs/s)
        # These are calculations that might not fit neatly into a ComputeUnit or Arch's general bandwidth.
        # Ensure 'fp16_tensor_flops' is available from compute_metrics.
        self.int8_tensor_flops = self.fp16_tensor_flops * 2  # Example: INT8 FLOPs derived from FP16 TC FLOPs
        return self

    def set_to_microbench(self):
        """
        Configures H100_PCIE for microbenchmarking.
        This method updates relevant parameters for microbenchmarking scenarios.
        """
        self.base_freq = 1.42 * 1e9
        self.max_freq = 1.42 * 1e9
        self.freq = 1.42 * 1e9  # Set operating frequency

        # Update utilization if microbenchmarks hit different limits
        self.ddr_max_util = 1.0
        self.l2_max_util = 1.0
        self.l1_max_util = 1.0
        self.compute_max_util = 1.0

        # Update bandwidths if microbenchmarks show different achieved rates
        self.ddr_bandwidth = 1864.22 * 1e9  # Raw B/s
        self.l2_bandwidth = 7598.48 * 1e9  # Raw B/s
        self.smem_bandwidth = 20720.6 * 1e9  # Raw B/s (if this is a distinct microbench value)

        # --- Re-calculate all raw performance metrics ---
        # Note: We assume core configurations (ComputeUnit instances) generally remain the same
        # across spec/microbench/ncu, only frequencies/bandwidths change.
        # If core configurations DO change for microbench, you'd re-instantiate ComputeUnits here.
        self.compute_metrics()
        self.int8_tensor_flops = self.fp16_tensor_flops * 2
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
        self.int8_tensor_flops = self.fp16_tensor_flops * 2
        return self
