"""
Current supported devices:

NVIDIA H100 PCIe
NVIDIA H100 80GB HBM3
NVIDIA H100 NVL
AMD INSTINCT MI250X
NVIDIA A100-SXM4-80GB

Tested but not added yet:

NVIDIA RTX A6000
AMD Instinct MI300X VF
NVIDIA GeForce RTX 4090
NVIDIA GeForce RTX 3090

"""

from tilelang.tools.tilesight.arch import *

# --- Device to Arch Class Mapping ---
# This dictionary maps TVM device names to their corresponding Arch subclass constructors.
DEVICE_ARCH_MAP = {
    # NVIDIA H100 Series
    "NVIDIA H100 PCIe": H100_PCIE,
    "NVIDIA H100 80GB HBM3": H100_SXM,
    "NVIDIA H100 NVL": H100_NVL,
    
    # NVIDIA A100 Series
    "NVIDIA A100-SXM4-80GB": A100_SXM,

    # AMD Instinct Series
    "AMD INSTINCT MI250X": MI250X,

}


def create_device_arch(device_name: str) -> Arch:
    """
    Factory function to create an instance of the appropriate Arch subclass

    
    based on the given device name.

    Args:
        device_name (str): The exact name of the device as returned by TVM
                           (e.g., "NVIDIA H100 PCIe").

    Returns:
        Arch: An instance of the specific Arch subclass for the given device.

    Raises:
        ValueError: If no known Arch class is found for the device name,
                    or if the mapped class is a placeholder (due to import error).
    """
    arch_class = DEVICE_ARCH_MAP.get(device_name)
    if arch_class is None:
        raise ValueError(
            f"No hardware abstraction class found for device: '{device_name}'. "
            "Please ensure the device name is correct and its corresponding "
            "Arch subclass is implemented and added to DEVICE_ARCH_MAP."
        )
    # Check if the class was loaded successfully (not a placeholder from ImportError)
    if arch_class is None: # This check is actually redundant due to the previous `if arch_class is None:`
        raise ValueError(f"Hardware abstraction class for '{device_name}' could not be loaded. "
                         "Check for missing or incorrectly implemented device Arch files.")
    
    return arch_class() # Instantiate the Arch subclass