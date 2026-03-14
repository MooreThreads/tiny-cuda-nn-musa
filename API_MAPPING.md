# API Mapping Report: tiny-cuda-nn

**SDK Version:** 4.3.5
**torch_musa Version:** v2.7.1
**Generated:** 2026-03-14
**Last Updated:** 2026-03-14

## Ecosystem Library Counterparts

| CUDA Dependency | MUSA Counterpart | Repo / Path | Notes |
|---|---|---|---|
| CUTLASS | mutlass | `MooreThreads/mutlass` (on GitHub), found at `/workspace/flash-attn2/3rdparty/mutlass` in container | Template library for matrix operations. Need to replace vendored cutlass submodule with mutlass. |
| cuBLAS | muBLAS | `/usr/local/musa/include/mublas.h` (SDK-bundled) | Direct counterpart. Optional (only for Shampoo optimizer). |
| CUB | CUB (MUSA SDK bundled) | `/usr/local/musa/include/cub/` | CUB headers available in MUSA SDK. |
| Thrust | Thrust (MUSA SDK bundled) | `/usr/local/musa/include/thrust/` | Thrust available in MUSA SDK. |
| nvrtc | **NOT AVAILABLE** | N/A | No MUSA equivalent for runtime compilation. Must disable RTC (`TCNN_BUILD_WITH_RTC=OFF`). |

## Layer 1: SimplePorting Default Coverage

| Category | Covered | Total | Coverage |
|----------|---------|-------|----------|
| CUDA Runtime API | 15 | 15 | 100% |
| CUDA Driver API | 6 | 6 | 100% |
| CUDA Graph API | 6 | 6 | 100% |
| Compiler macros | 4 | 4 | 100% |
| Header includes | 1 | 1 | 100% |
| Stream/Event types | 5 | 5 | 100% |
| PyTorch extension | 1 | 2 | 50% (CUDAExtension needs manual) |
| nvrtc | 0 | 6 | 0% (NOT AVAILABLE) |

## Layer 2: Custom Mapping Rules

| CUDA | MUSA | Category | Source (abs path) | Verified |
|------|------|----------|-------------------|----------|
| `CUDAExtension` | `MUSAExtension` | Build system | `torch_musa.utils.musa_extension` | Yes (container) |
| `torch.cuda.is_available()` | `torch.musa.is_available()` | Python API | torch_musa docs | Yes (container) |
| `torch.cuda.get_device_capability()` | `torch.musa.get_device_capability()` | Python API | torch_musa docs | Yes (container) |
| `torch.cuda.device_count()` | `torch.musa.device_count()` | Python API | torch_musa docs | Yes (container) |
| `.cuda()` | `.musa()` | Tensor device | torch_musa docs | Yes (container) |
| `device="cuda"` | `device="musa"` | Device string | torch_musa docs | Yes (container) |
| `is_cuda` | `is_musa` | Tensor check | torch_musa docs | Yes (container) |
| `cutlass/` include path | `mutlass/` include path | Library include | MooreThreads/mutlass | Yes (container) |
| `nvcc` | `mcc` | Compiler | `/usr/local/musa/bin/mcc` | Yes (container) |
| `WARP_SIZE = 32` | `WARP_SIZE = 128` | Hardware constant | MUSA S5000 hardware spec | **CRITICAL** |

## Detailed API Mapping Table

### CUDA Runtime API

| CUDA API | MUSA Equivalent | Layer | Source | Verified |
|----------|----------------|-------|--------|----------|
| `cudaMalloc` | `musaMalloc` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaFree` | `musaFree` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaMemcpy` | `musaMemcpy` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaMemcpyAsync` | `musaMemcpyAsync` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaStreamCreate` | `musaStreamCreate` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaStreamSynchronize` | `musaStreamSynchronize` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaStreamIsCapturing` | `musaStreamIsCapturing` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaGetDeviceProperties` | `musaGetDeviceProperties` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaSetDevice` | `musaSetDevice` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaGetDevice` | `musaGetDevice` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaDeviceSynchronize` | `musaDeviceSynchronize` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaGetLastError` | `musaGetLastError` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaPeekAtLastError` | `musaPeekAtLastError` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaError_t` | `musaError_t` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaSuccess` | `musaSuccess` | L1 | `/usr/local/musa/tools/general.json` | Yes |

### CUDA Driver API

| CUDA API | MUSA Equivalent | Layer | Source | Verified |
|----------|----------------|-------|--------|----------|
| `cuModuleLoadData` | `muModuleLoadData` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cuModuleGetFunction` | `muModuleGetFunction` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cuLaunchKernel` | `muLaunchKernel` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `CUmodule` | `MUmodule` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `CUfunction` | `MUfunction` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `CUresult` | `MUresult` | L1 | `/usr/local/musa/tools/general.json` | Yes |

### CUDA Graph API

| CUDA API | MUSA Equivalent | Layer | Source | Verified |
|----------|----------------|-------|--------|----------|
| `cudaStreamBeginCapture` | `musaStreamBeginCapture` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaStreamEndCapture` | `musaStreamEndCapture` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaGraphInstantiate` | `musaGraphInstantiate` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaGraphLaunch` | `musaGraphLaunch` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaStreamCaptureStatusNone` | `musaStreamCaptureStatusNone` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaErrorStreamCaptureImplicit` | `musaErrorStreamCaptureImplicit` | L1 | `/usr/local/musa/tools/general.json` | Yes |

### Compiler Macros

| CUDA Macro | MUSA Equivalent | Layer | Source | Verified |
|------------|----------------|-------|--------|----------|
| `__CUDACC__` | `__MUSACC__` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `__CUDA_ARCH__` | `__MUSA_ARCH__` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `__NVCC__` | `__MTCC__` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `CUDART_VERSION` | `MUSART_VERSION` | L1 | `/usr/local/musa/tools/general.json` | Yes |

### Stream/Event Types

| CUDA Type | MUSA Equivalent | Layer | Source | Verified |
|-----------|----------------|-------|--------|----------|
| `cudaStream_t` | `musaStream_t` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaEvent_t` | `musaEvent_t` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaStreamLegacy` | `musaStreamLegacy` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `cudaStreamPerThread` | `musaStreamPerThread` | L1 | `/usr/local/musa/tools/general.json` | Yes |
| `CUDAStream` | `MUSAStream` | L1 | torch_musa extra.json | Yes |

### Header Includes

| CUDA Header | MUSA Equivalent | Layer | Source | Verified |
|-------------|----------------|-------|--------|----------|
| `cuda.h` | `musa.h` | L1 | torch_musa include.json | Yes |
| `cuda_runtime.h` | `musa_runtime.h` | L1 | torch_musa include.json | Yes |
| `cuda_fp16.h` | `musa_fp16.h` | L1 | Container verified | Yes |
| `cublas_v2.h` | `mublas.h` | L1 | Container verified | Yes |

## Not Found / Unverified

| CUDA API | Status | Risk | Notes |
|----------|--------|------|-------|
| `nvrtcCreateProgram` | **NOT AVAILABLE** | **CRITICAL** | No MUSA equivalent for runtime compilation. Must disable RTC. |
| `nvrtcCompileProgram` | **NOT AVAILABLE** | **CRITICAL** | No MUSA equivalent. |
| `nvrtcGetPTX` | **NOT AVAILABLE** | **CRITICAL** | No MUSA equivalent. |
| `nvrtcGetErrorString` | **NOT AVAILABLE** | **CRITICAL** | No MUSA equivalent. |
| `NVRTC_SUCCESS` | **NOT AVAILABLE** | **CRITICAL** | No MUSA equivalent. |
| `nvrtc.h` | **NOT AVAILABLE** | **CRITICAL** | No MUSA equivalent header. |
| PTX `mma.sync.aligned.m16n8k16` | **NOT AVAILABLE** | **HIGH** | NVIDIA-specific PTX instruction. MUSA has different tensor core ISA. Requires `__MUSA_ARCH__` guard or disable. |
| PTX `wmma.mma.sync.aligned` | **NOT AVAILABLE** | **HIGH** | NVIDIA WMMA instruction. Not available in MUSA. Requires guard or disable. |
| PTX `mov.u32 %laneid` | **NEEDS INVESTIGATION** | **MEDIUM** | NVIDIA PTX instruction. May need MUSA equivalent inline ASM or alternative. |
| `WARP_SIZE = 32` | Needs change to `128` | **CRITICAL** | S5000 has 128-thread warps. Hardcoded in `common.h`. Affects warp shuffle, reduce, and all MMA operations. |

## Corrected Mappings

(No corrections yet.)
