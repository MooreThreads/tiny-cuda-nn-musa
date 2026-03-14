# MUSIFY — tiny-cuda-nn

| Field | Value |
|-------|-------|
| Project | tiny-cuda-nn |
| Source repo | https://github.com/NVlabs/tiny-cuda-nn |
| Migration branch | musa |
| Artifact type | Hybrid |
| Build system | CMake (LANGUAGES CXX CUDA) + setup.py (CUDAExtension) |
| Test infrastructure | CTest (tests/: test_encodings, test_grid, test_jit_losses, test_networks) |
| MUSA SDK version | 4.3.5 |
| torch_musa version | v2.7.1 |
| Docker image | 4.3.5_kuae2.1_20260119_torch2.7.1_ubuntu |
| MT GPU type | S5000 |
| Pipeline start | 2026-03-14 15:03:20 UTC |
| Pipeline end | |
| Elapsed | |
| Last updated | 2026-03-14 15:18 |
| Director | xiaofeng |
| Status | in-progress |

## Migration Plan

Migrate tiny-cuda-nn (NVlabs' tiny CUDA neural network framework) from CUDA to MUSA. The project is a Hybrid type (CMake C++ library + PyTorch Python bindings via setup.py). It includes 18 primary .cu files + ~90 headers, with a vendored CUTLASS submodule (~450 files). The pipeline will: analyze CUDA code, resolve API mappings, port code via SimplePorting, adapt CMake + setup.py build systems, and verify on MT GPU hardware.

**Key challenges:**
1. PTX inline assembly: mma.sync and wmma tensor core instructions (mma.h)
2. NVRTC runtime compilation (rtc_kernel.cu) — needs MUSA equivalent
3. Hardcoded WARP_SIZE = 32 (common.h) — MUSA S5000 warp size is 128
4. CUDA Driver API for JIT kernel loading
5. CUDA Graph API usage (cuda_graph.h)
6. Vendored CUTLASS dependency for matrix operations

## CUDA Dependencies

None — all dependencies are either CUDA ecosystem (handled by API mapping) or CUDA-free.

![CUDA Dependency Tree](CUDA_DEPENDENCIES.svg)

| # | Dependency | Classification | Notes |
|---|-----------|---------------|-------|
| 1 | CUTLASS | CUDA-ecosystem (vendored submodule) | Template library for matrix ops, ~450 files |
| 2 | cuBLAS | CUDA-ecosystem (optional) | Only for Shampoo optimizer (TCNN_ALLOW_CUBLAS_CUSOLVER) |
| 3 | nvrtc | CUDA-ecosystem | Runtime compilation for JIT kernels |
| 4 | CUDA Driver API | CUDA-ecosystem | Module loading for JIT |
| 5 | CUDA Graphs | CUDA-ecosystem | Graph capture/launch |
| 6 | fmt | CUDA-free | C++ formatting library |
| 7 | cmrc | CUDA-free | CMake resource compiler |
| 8 | pcg32 | CUDA-aware | RNG, header-only |
| 9 | nlohmann/json | CUDA-free | JSON parsing |

## Pipeline Progress

### tiny-cuda-nn (Main)

| Stage | Status | Started | Completed | Notes |
|-------|--------|---------|-----------|-------|
| 0. sdk-version | done | 2026-03-14 15:03 | 2026-03-14 15:05 | Core manifest resolved |
| 1a. reader (code analysis) | done | 2026-03-14 15:05 | 2026-03-14 15:12 | 18 .cu + ~90 .h primary files |
| 1b. lookuper (document-site) | done | 2026-03-14 15:05 | 2026-03-14 15:12 | Pre-existing, extended versions resolved |
| 1c. lookuper (docker setup) | done | 2026-03-14 15:05 | 2026-03-14 15:12 | Container running, GPU verified |
| 2. api-mapping | done | 2026-03-14 15:12 | 2026-03-14 15:18 | Full mapping complete, nvrtc NOT AVAILABLE |
| 3. musa-adapt | in-progress | 2026-03-14 15:18 | | |

#### Stage Details

**Stage 0 — SDK Version Resolution (Core):**
- torch_musa version: v2.7.1
- MUSA SDK version: 4.3.5
- Docker image: 4.3.5_kuae2.1_20260119_torch2.7.1_ubuntu
- Pull URL: sh-harbor.mthreads.com/mcctest/musa-train:4.3.5_kuae2.1_20260119_torch2.7.1_ubuntu

**Stage 1 — Parallel Preparation:**

*1a. Reader (code analysis):*
- Full report: [CODE_ANALYSIS.md](CODE_ANALYSIS.md)
- Project type: Hybrid (CMake C++ lib + PyTorch extension)
- CUDA files: 18 .cu (primary), ~90 .h headers, ~450 .cu/.cuh (vendored cutlass)
- Minimum CUDA version: 10.2 (Explicit), 11.8+ for full JIT functionality
- Dependencies requiring musification: none
- High-risk: PTX inline asm (mma.sync, wmma), NVRTC, WARP_SIZE=32, CUDA Driver API, CUDA Graphs
- Medium-risk: Vendored CUTLASS, __CUDA_ARCH__ guards
- Low-risk: cuBLAS (optional), half precision, Python bindings

*1b. Lookuper (document-site + extended versions):*
- document-site path: /cephfs/jing.feng/codebases/auto_musify/document-site/
- Branch `pre`: ready
- Branch `master`: ready
- Component versions resolved:
  - Driver: 3.3.5
  - MUSA Toolkits: 4.3.5
  - muDNN: 3.1.5
  - MCCL: 2.1.5
  - Triton: 1.4.5

*1c. Lookuper (docker setup):*
- Container name: musify_container
- GPU visible: yes (3x MTT S5000)
- MUSA version: 4.3.5
- torch_musa version: 2.7.1+5ee0a64
- PyTorch version: 2.7.1

**Dep Selection:** Zero in-scope deps. Skipped. Proceeding autonomously.

**Stage 2 — API Mapping:**
- Full report: [API_MAPPING.md](API_MAPPING.md)
- Mapping coverage: ~40 CUDA APIs with verified MUSA equivalents (Layer 1 via general.json)
- Custom rules needed: 10 (Layer 2 — CUDAExtension, Python APIs, cutlass->mutlass, warp size)
- **CRITICAL: nvrtc NOT AVAILABLE** — no MUSA equivalent for runtime compilation. Must build with `TCNN_BUILD_WITH_RTC=OFF`.
- **CRITICAL: WARP_SIZE** — must change from 32 to 128 for S5000.
- **HIGH RISK: PTX inline assembly** — mma.sync and wmma instructions in mma.h have no MUSA equivalent. Must be guarded or replaced.
- Ecosystem counterpart found: CUTLASS -> mutlass (MooreThreads/mutlass)

## Issues

| # | Team | Stage | Severity | Summary | Status | Resolution |
|---|------|-------|----------|---------|--------|------------|
| 1 | Main | 2. api-mapping | degraded | nvrtc (runtime compilation) has no MUSA equivalent | open | Build with TCNN_BUILD_WITH_RTC=OFF, disabling JIT-fused MLPs |
| 2 | Main | 2. api-mapping | degraded | PTX mma.sync/wmma instructions not available in MUSA | open | Guard with __MUSA_ARCH__ or disable fully_fused_mlp |
| 3 | Main | 2. api-mapping | blocking | WARP_SIZE=32 hardcoded, S5000 needs 128 | open | Change constant and verify all warp-level operations |

## Knowledge Base

(No entries yet.)

## Final Report

(Pending pipeline completion.)
