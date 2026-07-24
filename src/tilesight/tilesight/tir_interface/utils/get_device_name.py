from tilelang import tvm

# from tvm import runtime


# default device id is 0
def get_device_name(device_id: int = 0) -> str:
    if tvm.cuda(device_id).exist:
        return tvm.cuda(device_id).device_name
    elif tvm.rocm(device_id).exist:
        return tvm.rocm(device_id).device_name
    else:
        raise ValueError(
            f"Device {device_id} not found. Current we only support cuda and rocm devices.")


if __name__ == "__main__":
    print(get_device_name())
