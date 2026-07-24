from tilelang.tools.tilesight.utils import *
from tilelang.tools.tilesight.arch import *




if __name__ == "__main__":
    print("TileSight")
    device_name = get_device_name()
    print(device_name)
    device_arch = create_device_arch(device_name)
    print(device_arch)