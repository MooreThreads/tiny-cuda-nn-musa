# CUDA Code Analysis Report — tiny-cuda-nn

**Last Updated:** 2026-03-14

## 1. CUDA Source Inventory

### Primary Repository (excluding dependencies/)

| Directory | Files | Type |
|-----------|-------|------|
| `src/` | 11 `.cu` files | Core library: common_host, cpp_api, cutlass_mlp, encoding, fully_fused_mlp, loss, network, object, optimizer, reduce_sum, rtc_kernel |
| `include/tiny-cuda-nn/` | 58 `.h` headers | Headers: common, common_device, common_host, config, cpp_api, cuda_graph, cutlass_matmul, encoding, gpu_matrix, gpu_memory, loss, mma, multi_stream, network, object, optimizer, random, reduce_sum, rtc_kernel, trainer, vec |
| `include/tiny-cuda-nn/encodings/` | 10 `.h` headers | Encoding implementations: composite, empty, frequency, grid, grid_interface, identity, multi_level_interface, oneblob, spherical_harmonics, triangle_wave |
| `include/tiny-cuda-nn/networks/` | 2 `.h` headers | Network implementations: cutlass_mlp, fully_fused_mlp |
| `include/tiny-cuda-nn/optimizers/` | 10 `.h` headers | Optimizer implementations: adam, average, batched, composite, ema, exponential_decay, lookahead, novograd, sgd, shampoo |
| `include/tiny-cuda-nn/losses/` | 9 `.h` headers | Loss implementations: constant, cross_entropy, l1, l2, mape, relative_l1, relative_l2, relative_l2_luminance, smape, variance_is |
| `tests/` | 4 `.cu` + 1 `.h` | Tests: test_encodings, test_grid, test_jit_losses, test_networks + test_common.h |
| `benchmarks/` | 2 `.cu` | Benchmarks: bench_ours (image), bench_mlp_ours (mlp) |
| `samples/` | 1 `.cu` | Sample: mlp_learning_an_image |
| `bindings/torch/` | 1 `.cpp` + 2 `.py` | PyTorch binding: bindings.cpp, modules.py, __init__.py |

**Primary CUDA file count:** 18 `.cu` + ~90 `.h` headers

### Dependencies (vendored submodules)

| Dependency | Path | CUDA Classification | `.cu`/`.cuh` Files |
|------------|------|--------------------|--------------------|
| **cutlass** | `dependencies/cutlass/` | CUDA-ecosystem (CUTLASS template library) | ~450 files |
| **fmt** | `dependencies/fmt/` | CUDA-free (C++ formatting library) | 0 |
| **cmrc** | `dependencies/cmrc/` | CUDA-free (C++ resource compiler) | 0 |
| **pcg32** | `dependencies/pcg32/` | CUDA-aware (header-only, used in device code) | 0 |
| **json** | `dependencies/json/` | CUDA-free (nlohmann/json) | 0 |
| **stbi** | `dependencies/stbi/` | CUDA-free (image I/O) | 0 |
| **pybind11_json** | `dependencies/pybind11_json/` | CUDA-free | 0 |
| **catch2** | `dependencies/catch2/` | CUDA-free (test framework) | 0 |

## 2. CUDA API Surface

### Runtime API

| API | Files | Occurrences |
|-----|-------|-------------|
| `cudaMalloc` / `cudaFree` | src/common_host.cu, include/gpu_memory.h | Multiple (GPU memory management) |
| `cudaMemcpy` / `cudaMemcpyAsync` | include/gpu_memory.h, include/gpu_matrix.h | Multiple |
| `cudaStreamCreate` / `cudaStreamSynchronize` | include/multi_stream.h, include/common_host.h | Multiple |
| `cudaGetDeviceProperties` | src/common_host.cu, CMakeLists.txt (autodetect) | Multiple |
| `cudaDeviceSynchronize` | tests/*.cu, samples/*.cu | Multiple |
| `cudaGetLastError` / `cudaPeekAtLastError` | include/common.h (CUDA_CHECK_THROW macro) | Throughout |
| `cudaSetDevice` / `cudaGetDevice` | include/common_host.h | Multiple |
| `cudaEventCreate` / `cudaEventRecord` | include/cuda_graph.h | Multiple |

### CUDA Graph API

| API | File | Notes |
|-----|------|-------|
| `cudaStreamBeginCapture` | include/cuda_graph.h | Graph capture |
| `cudaStreamEndCapture` | include/cuda_graph.h | Graph capture |
| `cudaGraphInstantiate` | include/cuda_graph.h | Graph instantiation |
| `cudaGraphLaunch` | include/cuda_graph.h | Graph execution |
| `cudaStreamIsCapturing` | include/cuda_graph.h | Capture status check |

### NVRTC (Runtime Compilation)

| API | File | Notes |
|-----|------|-------|
| `nvrtcCreateProgram` | src/rtc_kernel.cu | JIT kernel compilation |
| `nvrtcCompileProgram` | src/rtc_kernel.cu | Compile with nvcc flags |
| `nvrtcGetPTX` | src/rtc_kernel.cu | Get compiled PTX |
| `cuModuleLoadData` | src/rtc_kernel.cu | Load compiled module |
| `cuModuleGetFunction` | src/rtc_kernel.cu | Get kernel function |
| `cuLaunchKernel` | src/rtc_kernel.cu | Launch JIT kernel |

### cuBLAS (Optional — Shampoo optimizer only)

| API | File | Notes |
|-----|------|-------|
| `cublasSgemm` | include/optimizers/shampoo.h | Only when TCNN_ALLOW_CUBLAS_CUSOLVER=ON |

### Compiler Macros & Types

| Macro/Type | Files | Notes |
|------------|-------|-------|
| `__CUDACC__` | include/common.h, include/vec.h | Compiler detection |
| `__CUDA_ARCH__` | include/common.h, include/common_device.h, include/mma.h, include/vec.h | Architecture guards |
| `CUDART_VERSION` | src/common_host.cu, src/rtc_kernel.cu | Runtime version check |
| `__half` / `cuda_fp16.h` | include/common.h, include/vec.h, include/mma.h, others | Half precision throughout |
| `__nv_bfloat16` | Not used in primary repo | |

### PTX Inline Assembly (CRITICAL)

| Instruction | File | Notes |
|------------|------|-------|
| `asm volatile("mov.u32 %0, %laneid;")` | include/common_device.h:39 | Lane ID query |
| `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` | include/mma.h:95,106 | Tensor core MMA (SM >= 80) |
| `wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16` | include/mma.h:120 | WMMA fallback (SM 70-79) |

### Kernel Launch

| Pattern | Files | Notes |
|---------|-------|-------|
| `<<<...>>>` kernel launches | Throughout src/ and include/ | Standard kernel launches |
| `cuLaunchKernel` (driver API) | src/rtc_kernel.cu | JIT kernel launches |

## 3. Architecture-Specific Code

### __CUDA_ARCH__ Guards

| Guard | File:Line | Behavior |
|-------|-----------|----------|
| `>= 600` | include/vec.h:212,327,358 | atomicAdd for half types |
| `>= 700` | include/vec.h:303,329 | atomicAdd for half2 types |
| `>= 800` | include/mma.h:93, include/common_device.h:101 | Use `mma.sync` PTX instead of `wmma`; use `__hmax` |

### Warp Size Assumptions

- **`WARP_SIZE = 32`** hardcoded constant in `include/tiny-cuda-nn/common.h:248`
- Used extensively throughout: `reduce_sum.h`, `trainer.h`, `cutlass_mlp.h`, `fully_fused_mlp.h`, `network_with_input_encoding.h`, `encodings/composite.h`
- Also uses runtime `warpSize` in `reduce_sum.h` for warp shuffle reductions

### Shared Memory

- Static `__shared__` arrays in reduce_sum.h
- Dynamic extern shared memory in fully_fused_mlp kernels

### Thread Block Dimensions

- Block sizes depend on network width (16, 32, 64, 128)
- Warp-level operations assume 32-thread warps

## 4. Build System Analysis

### CMake (Primary Build)

- **CMake minimum**: 3.18
- **Languages**: CXX + CUDA (`project(... LANGUAGES CXX CUDA)`)
- **Output**: Static library `libtiny-cuda-nn.a`
- **CUDA standard**: 14
- **Architecture autodetection**: Custom `TCNN_AUTODETECT_CUDA_ARCHITECTURES()` function
- **Minimum CUDA version**: 10.2 (explicit fatal_error check)
- **Conditional features**:
  - `TCNN_BUILD_WITH_RTC` (default ON) — requires `nvrtc` library
  - `TCNN_ALLOW_CUBLAS_CUSOLVER` (default OFF) — Shampoo optimizer
  - `TCNN_HALF_PRECISION` — auto-enabled for SM > 52 and != 61
  - `TCNN_BUILD_NO_FWD_BWD` — builds without offline compiled fwd/bwd kernels
- **Key NVCC flags**: `--extended-lambda`, `--expt-relaxed-constexpr`, `--use_fast_math`
- **Include paths**: `include/`, `dependencies/`, `dependencies/cutlass/include/`, `dependencies/cutlass/tools/util/include/`
- **Linked libraries**: `cuda` (driver API), `nvrtc` (when RTC enabled), `cublas` (when Shampoo enabled), `fmt`, `cmrc`
- **Test system**: CTest (`enable_testing()` + `add_subdirectory(tests)`) when `TCNN_BUILD_TESTS=ON`

### setup.py (Python Extension Build)

- **Location**: `bindings/torch/setup.py`
- **Extension type**: `torch.utils.cpp_extension.CUDAExtension`
- **Libraries**: `["cuda", "nvrtc"]`
- **Source files**: bindings.cpp + src/*.cu + fmt
- **NVCC flags**: `--extended-lambda`, `--use_fast_math`, `--expt-relaxed-constexpr`, per-capability gencode
- **Capabilities**: Auto-detected from PyTorch or `TCNN_CUDA_ARCHITECTURES` env var
- **Definitions**: `-DTCNN_PARAMS_UNALIGNED`, `-DTCNN_RTC`, `-DTCNN_RTC_USE_FAST_MATH`, `-DTCNN_MIN_GPU_ARCH=<cc>`

### Artifact Type: **Hybrid**

CMake builds a static C++ library (`libtiny-cuda-nn.a`) consumed by other C++ projects (samples, benchmarks, tests), AND `setup.py` builds a Python extension (`tinycudann_bindings`) loaded via `import tinycudann`.

### Test Infrastructure

- **CTest**: 4 test files in `tests/` (test_encodings, test_grid, test_jit_losses, test_networks)
- **Test CMakeLists.txt**: uses `add_test()` with CTest
- **No pytest**: Python tests not found
- **Verification tier**: V2 (test suite via CTest)

## 5. Python/PyTorch Layer

### Python Files

| File | Role | CUDA Usage |
|------|------|------------|
| `bindings/torch/tinycudann/__init__.py` | Package entry | Imports modules, no direct CUDA |
| `bindings/torch/tinycudann/modules.py` | Library code | `torch.cuda.is_available()`, `torch.cuda.get_device_capability()`, `torch.cuda.device_count()`, `.cuda()` device transfers |
| `bindings/torch/setup.py` | Build config | `CUDAExtension`, nvcc version detection |

### CUDA Usage in Python

- `torch.cuda.is_available()` — device availability check
- `torch.cuda.get_device_capability()` — compute capability detection
- `torch.cuda.device_count()` — multi-GPU support
- `.cuda()` — tensor device transfer
- `torch.autograd.Function` — custom forward/backward with native CUDA ops
- No distributed training (no NCCL/DDP)
- No AMP/autocast

## 6. Minimum CUDA Version Estimation

| Component | Minimum Version | Confidence | Source |
|-----------|----------------|------------|--------|
| CUDA Toolkit | 10.2 | **Explicit** | `CMakeLists.txt:236`: `CUDA version too low. tiny-cuda-nn require CUDA 10.2 or higher.` |
| JIT/RTC feature | 11.8 | **Explicit** | `src/rtc_kernel.cu:95`: `cuda_runtime_version() >= 11080` |
| SM Architecture | sm_75 (minimum for JIT) | **Explicit** | `src/rtc_kernel.cu:95`: `cuda_supported_compute_capability(device) >= 75` |
| MMA Tensor Cores | sm_70 (WMMA) / sm_80 (MMA.sync) | **Explicit** | `include/mma.h:93,120`: architecture guards |
| Half precision | sm_53+ (general), sm_60+ (atomicAdd) | **Explicit** | `include/vec.h:212,327`: `__CUDA_ARCH__ >= 600` |
| cuBLAS | 10.1+ (optional, Shampoo only) | Estimated | Uses basic `cublasSgemm` |
| CUTLASS | 2.x | **Detected** | Vendored at `dependencies/cutlass/` (commit 1eb635) |
| PyTorch | 1.x+ | Estimated | Uses `torch.utils.cpp_extension.CUDAExtension`, basic autograd |
| Python | 3.6+ | Estimated | Uses f-strings, no explicit constraint |

**Practical minimum for full functionality: CUDA 11.8+** (required for JIT/RTC support, which is the project's key differentiator for performance).

## Migration Risk Assessment

### High Risk

1. **PTX Inline Assembly**: `mma.sync` and `wmma` instructions in `mma.h` are NVIDIA-specific tensor core instructions. MUSA GPU has different tensor core instruction sets.
2. **NVRTC Runtime Compilation**: The JIT compilation system (`rtc_kernel.cu`) uses `nvrtcCreateProgram`, `nvrtcCompileProgram` to compile kernels at runtime. MUSA equivalent needs investigation.
3. **Warp Size = 32**: Hardcoded throughout. MUSA S5000 warp size is 128.
4. **CUDA Driver API**: `cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel` used for JIT kernels.

### Medium Risk

5. **CUTLASS dependency**: Vendored CUTLASS is a CUDA-specific template library for matrix operations. Needs MUSA-compatible CUTLASS or equivalent.
6. **Architecture guards**: Multiple `__CUDA_ARCH__` checks with NVIDIA-specific SM levels.
7. **CUDA Graph API**: Full graph capture/instantiate/launch cycle.

### Low Risk

8. **cuBLAS**: Optional, only for Shampoo optimizer (can be disabled).
9. **Half precision**: Standard `__half` type, broadly supported.
10. **Python bindings**: Thin wrapper around C++ extension, minimal CUDA-specific code.
