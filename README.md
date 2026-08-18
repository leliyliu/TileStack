# TileSight

A tile-centric analytical GPU performance model from cores to clusters. This
repository is a source-visible snapshot of the TileSight modeling tool
(`tilesight-release-2026-07`), supporting single-kernel latency, multi-level
cache behavior, fused-kernel pipeline overlap, and distributed multi-GPU
communication modeling without per-architecture training or profiling.

TileSight treats the **tile** as the first-class modeling unit and adopts a
prologue–steady–epilogue pipeline envelope applied recursively at every level.
It composes three hierarchical levels with unified tile-based abstractions:

- **Intra-tile** — each tile is decomposed into a per-pipeline resource vector
  spanning compute (tensor cores / CUDA cores / SFU / TMEM), memory (SMEM /
  L1.5 / L2 / DDR), and network.
- **Inter-tile** — tiles are related through producer–consumer dependencies,
  concurrent issue, and execution order; a topological-order search over the
  tile-action DAG picks the best legal pipeline overlap, and a tile reuse
  distance analysis with a stochastic distance cache model (SDCM) derives
  multi-level cache hit rates.
- **Cross-device** — cross-device execution is a placement case of the same
  intra-tile abstraction: remote tensor accesses are inferred from
  producer–consumer placement and decomposed into ordered stages of logical
  exchanges, whose routed α–β cost populates the network entry of the per-tile
  resource vector.

See `docs/TileSight.pdf` for the full paper and `docs/` for source-level
documentation.

## Documentation

| Document | Contents |
|---|---|
| `docs/TileSight.pdf` | The TileSight paper (arXiv:2607.22432). |
| `docs/tilesight_analysis.md` | Mechanism-by-mechanism mapping from the paper's §3–4 to the source code, proving each equation and algorithm step is implemented. |
| `docs/code_walkthrough.md` | A walkthrough of the code logic: package structure, data structures, call chains, and key algorithms per subpackage. |

## Package structure

| Path | Contents |
|---|---|
| `src/tilesight/arch/` | Hardware abstraction (`Arch` base class) and 31 GPU/non-GPU profiles (NVIDIA, AMD, CGRA, TPU, …). |
| `src/tilesight/fused_op_pipeline_wave/` | Paper-level modeling: resource vectors, occupancy, pipeline overlap, wave model, DAG overlap analysis. |
| `src/tilesight/tile_cache/` | Multi-level cache hit-rate modeling (two-level L1.5+L2 SDCM cascade). |
| `src/tilesight/distributed/` | Cross-device modeling: `DistributedTileMap`, `NetworkHierarchy`, collectives, distributed op composition. |
| `src/tilesight/fused_op_dtype_wave/` | Empirical wave-version fused-op modeling (matmul/element/reduce). |
| `src/tilesight/fused_op/`, `src/tilesight/fused_op_dtype/` | Earlier fused-op implementations. |
| `src/tilesight/single_op/` | Single-operator models (matmul/conv/element/reduce). |
| `src/tilesight/util/` | SDCM, reuse-distance flow sim, L2 hit-rate estimators, profilers. |
| `src/tilesight/fusion_support/` | Operator fusion (register/SMEM, heterogeneous). |
| `src/tilesight/tir_interface/` | TIR (TVM) interface for analyzing tile programs. |
| `src/tilesight/welder_info_extract/`, `src/tilesight/welder_modeling_top/` | Welder compiler info extraction and top-level modeling. |
| `src/tilesight/compare_with_ncu/` | Comparison with Nsight Compute profilers. |
| `src/tilesight/onnx_model/` | ONNX model descriptions. |

## Installation

```bash
pip install -e .
```

Core runtime dependencies: `numpy`, `scipy`, `networkx`, `pandas`, `torch`
(CPU build). The `tir_interface/` subpackage additionally requires `tilelang`
/`tvm` (optional; not imported by the core modeling path).

> **Platform note.** The bundled `distributed/noc/_model_support.*.so` is a
> Linux x86-64 CPython 3.11 extension. On other platforms, the prebuilt
> reference NoC topology profiles are unavailable; custom topologies can still
> be constructed via the source-visible `noc_topo.make_*` factories.

## Licensing

Mixed-license. Source code, scripts, and documentation are licensed under
Apache-2.0. The bundled precompiled NoC support library
(`src/tilesight/tilesight/distributed/noc/_model_support.*.so`) is proprietary
and licensed separately under
[`LicenseRef-DeepStack-AE-Binary-1.0`](LICENSES/LicenseRef-DeepStack-AE-Binary-1.0.txt).
The repository as a whole is not Apache-2.0; see [`LICENSING.md`](LICENSING.md)
and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
