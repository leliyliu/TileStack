import math


class ComputeUnit:

    def __init__(self,
                 unit_type: str,
                 dtype: str,
                 instances: int = 1,
                 shape: list = None,
                 ops_per_element: int = 2):
        """
        Defines a generic compute unit (e.g., Tensor Core, CUDA Core, SFU).

        Args:
            unit_type (str): The general type of the compute unit (e.g., "tensor", "vector", "scalar").
            dtype (str): The data type it operates on (e.g., "fp32", "fp16", "int8").
            instances (int): Number of parallel instances of this unit per SM. Defaults to 1.
            shape (list, optional): The operational shape of the unit, if applicable (e.g., [M, N, K] for tensor cores).
                                     Defaults to None.
            ops_per_element (int): Number of operations performed per element/component in a cycle.
                                   Defaults to 2 (e.g., FMA for FP32).
        """
        self.unit_type = unit_type
        self.dtype = dtype
        self.instances = instances
        self.shape = shape if shape is not None else []  # Ensure it's a list if not provided
        self.ops_per_element = ops_per_element  # Stored as an instance attribute

    def calculate_flops(self, sm_count: int, freq_hz: float) -> float:
        """
        Calculates the total floating-point operations (FLOPs) provided by this compute unit type
        across all SMs per second.

        Args:
            sm_count (int): Total number of Streaming Multiprocessors (SMs).
            freq_hz (float): The operating frequency in Hz.

        Returns:
            float: Total FLOPs per second.
        """
        if sm_count <= 0 or freq_hz <= 0:
            return 0.0

        # Calculate base operations per unit per cycle based on its shape (e.g., prod(shape) for matrix, 1 for scalar)
        # If shape is empty (e.g., for scalar/vector units), math.prod([]) returns 1.
        shape_product = math.prod(self.shape) if self.shape else 1

        # Total operations per cycle for THIS unit type per SM
        ops_per_sm_per_cycle = self.instances * shape_product * self.ops_per_element

        # Total operations per second across ALL SMs (FLOPs/s)
        total_flops_per_second = ops_per_sm_per_cycle * sm_count * freq_hz

        return float(total_flops_per_second)


class Arch:

    def __init__(self):
        """
        Initializes the common base attributes for the architecture.
        Optional hardware features and derived performance metrics are NOT initialized here,
        but rather set by subclasses or calculated in compute_raw_metrics.
        """
        self.core = None
        self.sm_count = 0
        self.freq = 0.0  # Core frequency (in Hz); subclasses will set this.

        self.ddr_bandwidth = 0.0  # Bytes per second
        self.ddr_capacity = 0  # Bytes
        self.l2_bandwidth = 0.0  # Bytes per second
        self.l2_capacity = 0  # Bytes
        self.l2_partitions = 1

        self.sm_sub_partitions = 0
        self.l1_smem_throughput_per_cycle = 0  # Bytes per cycle per SM
        self.configurable_smem_capacity = 0  # Bytes
        self.register_capacity_per_sm = 0  # Bytes

        # ComputeUnit instances will be assigned by subclasses.
        self.fp16_tensor_core_unit = None
        self.fp32_cuda_core_unit = None
        self.sfu_unit = None

        # Derived performance metrics (will be calculated in compute_raw_metrics)
        # These attributes now store raw FLOPs/second (not TFLOPS) and raw bandwidth (not GB/s).
        self.fp16_tensor_flops = 0.0
        self.fp32_cuda_core_flops = 0.0
        self.sfu_flops = 0.0
        self.smem_bandwidth = 0.0  # Bytes per second
        self.register_bandwidth = 0.0  # Bytes per second

        # Utilization limits
        self.ddr_max_util = 0.9
        self.l2_max_util = 0.9
        self.l1_max_util = 0.9
        self.compute_max_util = 0.9

    def compute_flops(self):
        """
        Calculates various raw throughputs (FLOPs/second) based on the instantiated ComputeUnit objects and other architecture parameters.
        No unit conversions (e.g., to TFLOPS) are performed here.
        """
        # --- Tensor Core related calculations ---
        if self.fp16_tensor_core_unit and self.sm_count > 0 and self.freq > 0:
            self.fp16_tensor_flops = self.fp16_tensor_core_unit.calculate_flops(
                self.sm_count, self.freq)
        else:
            self.fp16_tensor_flops = 0.0

        if hasattr(self, 'fp8_tensor_core_unit'
                  ) and self.fp8_tensor_core_unit and self.sm_count > 0 and self.freq > 0:
            self.fp8_tensor_flops = self.fp8_tensor_core_unit.calculate_flops(
                self.sm_count, self.freq)
        else:
            self.fp8_tensor_flops = 0.0

        if hasattr(self, 'fp4_tensor_core_unit'
                  ) and self.fp4_tensor_core_unit and self.sm_count > 0 and self.freq > 0:
            self.fp4_tensor_flops = self.fp4_tensor_core_unit.calculate_flops(
                self.sm_count, self.freq)
        else:
            self.fp4_tensor_flops = 0.0

        # --- CUDA Core related calculations ---
        if self.fp32_cuda_core_unit and self.sm_count > 0 and self.freq > 0:
            self.fp32_cuda_core_flops = self.fp32_cuda_core_unit.calculate_flops(
                self.sm_count, self.freq)
        else:
            self.fp32_cuda_core_flops = 0.0

        if hasattr(self, 'fp16_cuda_core_unit'
                  ) and self.fp16_cuda_core_unit and self.sm_count > 0 and self.freq > 0:
            self.fp16_cuda_core_flops = self.fp16_cuda_core_unit.calculate_flops(
                self.sm_count, self.freq)
        else:
            self.fp16_cuda_core_flops = 0.0

        if hasattr(self, 'fp64_cuda_core_unit'
                  ) and self.fp64_cuda_core_unit and self.sm_count > 0 and self.freq > 0:
            self.fp64_cuda_core_flops = self.fp64_cuda_core_unit.calculate_flops(
                self.sm_count, self.freq)
        else:
            self.fp64_cuda_core_flops = 0.0

        if hasattr(self, 'int32_cuda_core_unit'
                  ) and self.int32_cuda_core_unit and self.sm_count > 0 and self.freq > 0:
            self.int32_cuda_core_flops = self.int32_cuda_core_unit.calculate_flops(
                self.sm_count, self.freq)
        else:
            self.int32_cuda_core_flops = 0.0

        # --- SFU calculations ---
        if self.sfu_unit and self.sm_count > 0 and self.freq > 0:
            self.sfu_flops = self.sfu_unit.calculate_flops(self.sm_count, self.freq)
        else:
            self.sfu_flops = 0.0

        return self

    def compute_bandwidth(self):
        """
        Calculates various raw bandwidths (Bytes/second) based on the architecture parameters.
        No unit conversions (e.g., to GB/s) are performed here.
        """
        # --- Bandwidth calculations (these still depend on direct Arch attributes) ---
        # Assuming l1_smem_throughput_per_cycle is in Bytes per cycle per SM
        if self.sm_count > 0 and self.freq > 0 and self.l1_smem_throughput_per_cycle > 0:
            self.smem_bandwidth = self.sm_count * self.freq * self.l1_smem_throughput_per_cycle
        else:
            self.smem_bandwidth = 0.0

        # Assuming 32 * 4 Bytes per sub-partition per cycle
        if self.sm_count > 0 and self.freq > 0 and self.sm_sub_partitions > 0:
            self.register_bandwidth = self.sm_count * self.freq * self.sm_sub_partitions * 32 * 4
        else:
            self.register_bandwidth = 0.0

        return self

    def compute_metrics(self):
        """
        Orchestrates the calculation of all raw performance metrics (FLOPs/second and Bytes/second).
        This method should be called in subclasses after all parameters are set.
        """
        self.compute_flops()
        self.compute_bandwidth()
        return self

    # --- Abstract Configuration Methods ---
    def set_to_spec(self):
        """
        Configures the architecture to its official specification parameters.
        This method is abstract and must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement set_to_spec() to configure the specific hardware's official specifications."
        )

    def set_to_microbench(self):
        """
        Configures the architecture to parameters optimized for microbenchmarking.
        This method is abstract and must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement set_to_microbench() to configure the specific hardware's microbenchmark parameters."
        )

    def set_to_ncu(self):
        """
        Configures the architecture to parameters observed via NCU profiling.
        This method is abstract and must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement set_to_ncu() to configure the specific hardware's NCU observed parameters."
        )
