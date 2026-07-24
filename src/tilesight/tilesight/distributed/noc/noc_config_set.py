"""Readable facades for binary-backed reference interconnect profiles."""

from __future__ import annotations

from . import _model_support as _provider
from .noc_topo import Hierarchy


_PROFILE_IDS = {
    "torus_mesh_switch_1": 0,
    "torus_mesh_switch_2": 1,
    "torus_mesh_mesh_3": 2,
    "strong_torus_mesh_switch_4": 3,
    "weak_torus_mesh_switch_5": 4,
    "torus_mesh_switch_7": 5,
    "torus_mesh_switch_8": 6,
    "torus_mesh_switch_9": 7,
    "h200x8": 8,
    "h100x8": 9,
    "h100x32_strong": 10,
    "h100x32_medium": 11,
    "h100_8x1_8": 12,
    "h100_8x2_16": 13,
    "h100_8x4_32": 14,
    "h100_8x8_64": 15,
    "stacked_gpu_4x4": 16,
    "stacked_gpu_4x8": 17,
    "stacked_gpu_4x16": 18,
    "stacked_gpu_8x8": 19,
    "stacked_gpu_8x16": 20,
    "b200_8x1_8": 21,
}

_PROFILE_OUTPUT_NAMES = {
    "h100x8": "h200x8",
    "stacked_gpu_4x4": "stacked_gpu_4x4_64",
    "stacked_gpu_4x8": "stacked_gpu_4x8_128",
    "stacked_gpu_4x16": "stacked_gpu_4x16_256",
    "stacked_gpu_8x8": "stacked_gpu_8x8_256",
    "stacked_gpu_8x16": "stacked_gpu_8x16_512",
}


def _fixed_profile(public_name: str) -> Hierarchy:
    hierarchy = _provider.p00(_PROFILE_IDS[public_name])
    hierarchy.name = _PROFILE_OUTPUT_NAMES.get(public_name, public_name)
    return hierarchy


def _install_fixed_profile(public_name: str) -> None:
    def factory() -> Hierarchy:
        return _fixed_profile(public_name)

    factory.__name__ = public_name
    factory.__qualname__ = public_name
    factory.__doc__ = f"Construct the {public_name} interconnect profile."
    globals()[public_name] = factory


for _profile_name in _PROFILE_IDS:
    _install_fixed_profile(_profile_name)


def h100x(num_nodes: int) -> Hierarchy:
    hierarchy = _provider.p01(0, int(num_nodes))
    hierarchy.name = "h200x8"
    return hierarchy


def model_support_api_version() -> int:
    return int(_provider.p98())


__all__ = [
    *_PROFILE_IDS,
    "h100x",
    "model_support_api_version",
]
