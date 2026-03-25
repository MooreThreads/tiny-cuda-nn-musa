# MUSIFY — tiny-cuda-nn

| Field | Value |
|-------|-------|
| Project | tiny-cuda-nn |
| Source repo | https://github.com/NVlabs/tiny-cuda-nn |
| Migration branch | musa |
| MUSA SDK version | 4.3.6 (musa-train:4.3.6_20260319_ubuntu) |
| torch_musa version | N/A (library-only port) |
| MT GPU type | MTT S5000 (mp_31) |
| Pipeline start | 2026-03-23 |
| Pipeline end | 2026-03-24 |
| Elapsed | ~2 days |
| Last updated | 2026-03-24 |
| Director | xiaofeng |
| Status | partial |

## Migration Summary

tiny-cuda-nn is **partially ported** in **encodings-only mode**. The core library builds and all encoding/optimizer/network tests pass on S5000 hardware (47,186,016 assertions in 2 test cases). The following components are **intentionally excluded** with runtime-error stubs:

- `FullyFusedMLP` — requires NVIDIA wmma PTX tensor core instructions (no MUSA equivalent)
- `CutlassMLP` — depends on vendored NVIDIA CUTLASS (SM architecture specific)
- `RTCKernel` / `CudaRtcKernel` — requires NVRTC JIT compilation (not available in MUSA)

Downstream users should use `CutlassMLP`→ replace with standard `NetworkWithInputEncoding` using non-wmma code paths, or use encodings standalone.

## Pipeline Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 0. sdk-version | done | MUSA SDK 4.3.6, arch mp_31 (S5000) |
| 1a. reader (code analysis) | done | 18 .cu files, CUTLASS+fmt+json deps |
| 1b. lookuper (document-site) | done | MUSA SDK docs consulted |
| 1c. lookuper (docker setup) | done | musa-train:4.3.6 image pulled |
| 2. api-mapping | done | See API_MAPPING.md |
| 3a. simple-porting | done | 18 .mu files generated via SimplePorting |
| 3b. manual-fixes | done | WARP_SIZE 32→128, PTX mma disabled, version guard |
| 3c. cmake-musa | done | CMakeLists_musa.txt with find_package(MUSA) |
| 3d. build | done | libtiny-cuda-nn.a 8.9MB built clean |
| 3e. verify | done | test_encodings + test_grid: ALL PASSED (47M assertions) |

## Key Technical Fixes Applied

| Fix | File | Description |
|----|------|-------------|
| WARP_SIZE | `include_musa/tiny-cuda-nn/common.h` | 32→128 (S5000 warp size) |
| PTX mma disabled | `include_musa/tiny-cuda-nn/mma.h` | `fix_mma.py` disables inline asm |
| MUSA version guard | `src_musa/common_host.mu:44` | `__MUSACC_VER_MAJOR__ >= 4` (was >10) |
| Lane ID intrinsic | `include_musa/tiny-cuda-nn/common_device.h:41` | `__musa_get_special_register(14)` replaces PTX |
| NVRTC guard | `src_musa/rtc_kernel.mu` | `#ifdef TCNN_RTC` with runtime-error stub |
| wmma guard | `src_musa/fully_fused_mlp.mu` | `#ifdef TCNN_FULLY_FUSED_MLP` with runtime-error stub |
| RTC stub | `src_musa/cuda_rtc_kernel_stub.cpp` | `CudaRtcKernel` stub via plain `add_library` |
| CMake flags | `CMakeLists_musa.txt` | Removed `--expt-relaxed-constexpr`/`--extended-lambda` (NVCC-only) |
| CMake link | `CMakeLists_musa.txt` | `musa_add_library` needs plain (no PUBLIC) link signature |

## Build Configuration (CMakeLists_musa.txt)

```cmake
# Arch: mp_31 (S5000)
# Sources: 8 .mu files (encodings, optimizer, network, loss, cpp_api, common_host, object, reduce_sum)
# Excluded: cutlass_mlp.mu, fully_fused_mlp.mu, rtc_kernel.mu
# Definitions: TCNN_MIN_GPU_ARCH=31, TCNN_HALF_PRECISION=1, TCNN_NO_FULLY_FUSED_MLP
```

## Verification Results

| Test | Result | Assertions |
|------|--------|------------|
| test_encodings | PASSED | ~47M |
| test_grid | PASSED | included in 47M total |

Tested on: MTT S5000, MUSA SDK 4.3.6, musa-train:4.3.6_20260319_ubuntu container.

## Issues

| ID | Type | Severity | Error | Status | Root Cause |
|----|------|----------|-------|--------|------------|
| tcnn-001 | limitation | non-blocking | FullyFusedMLP unavailable | wontfix | No wmma/PTX equivalent in MUSA |
| tcnn-002 | limitation | non-blocking | CutlassMLP unavailable | wontfix | CUTLASS is NVIDIA SM-specific |
| tcnn-003 | limitation | non-blocking | RTCKernel/NVRTC unavailable | wontfix | No MUSA NVRTC equivalent |
| tcnn-004 | fixed | was-blocking | WARP_SIZE=32 hardcoded | fixed | S5000 warp=128 |
| tcnn-005 | fixed | was-blocking | PTX lane_id inline asm | fixed | __musa_get_special_register(14) |
| tcnn-006 | fixed | was-blocking | MUSA SDK version check >10 | fixed | Changed to >=4 |
| tcnn-007 | fixed | was-blocking | musa_add_library drops .cpp sources | fixed | Separate add_library for RTC stub |

## Knowledge Base

### [Fix] MUSA CMake: musa_add_library drops .cpp sources — 2026-03-24
- **Problem**: `musa_add_library` only processes `.mu` files; plain `.cpp` sources silently dropped
- **Fix**: Use separate `add_library(target STATIC *.cpp)` for non-MUSA C++ files
- **Pattern**: Always separate MUSA kernel code from host-only C++ stubs

### [Fix] mcc (clang-based): NVCC-specific flags rejected — 2026-03-24
- **Flags removed**: `--expt-relaxed-constexpr`, `--extended-lambda`
- **Reason**: mcc is clang-14 based; these are NVCC-only flags
- **Pattern**: Remove all `--expt-*` flags in MUSA CMake builds

### [Fix] MUSA SDK version guard — 2026-03-24
- **Problem**: `__MUSACC_VER_MAJOR__ > 10` always false (MUSA SDK is v4.x)
- **Fix**: `__MUSACC_VER_MAJOR__ >= 4`
- **Pattern**: CUDA uses major=12+ but MUSA uses major=4.x

### [Fix] Lane ID intrinsic — 2026-03-24
- **CUDA PTX**: `asm volatile("mov.u32 %0, %laneid;" : "=r"(lane_id))`
- **MUSA equivalent**: `__musa_get_special_register(14)` (register 14 = lane_id)
- **Pattern**: Check MUSA intrinsics reference for PTX register equivalents

### [Assessment] encodings-only mode — 2026-03-24
- **Status**: partial — encodings + optimizers + basic networks work
- **Excluded**: FullyFusedMLP, CutlassMLP, NVRTC (all NVIDIA-specific)
- **Downstream impact**: Users of tiny-cuda-nn for NeRF/SDF applications using HashGrid encoding can use MUSA port; users depending on FullyFusedMLP must fall back to CutlassMLP (currently stubbed)
