#!/usr/bin/env python3
"""
SimplePorting script for tiny-cuda-nn CUDA-to-MUSA migration.
Run inside the container:
    python3 /workspace/tiny-cuda-nn/simple_porting_tcnn.py
"""
import os
import shutil
from torch_musa.utils.simple_porting import SimplePorting

PROJECT_DIR = "/workspace/tiny-cuda-nn"

# Custom mapping rules (project-specific, extend defaults)
MAPPING_RULES = {
    # Compiler macros
    "__NVCC__": "__MTCC__",

    # CUDA version macros
    "CUDART_VERSION": "MUSART_VERSION",

    # Driver API types
    "CUmodule": "MUmodule",
    "CUfunction": "MUfunction",
    "CUresult": "MUresult",
    "cuModuleLoadData": "muModuleLoadData",
    "cuModuleGetFunction": "muModuleGetFunction",
    "cuLaunchKernel": "muLaunchKernel",

    # Stream types
    "cudaStreamLegacy": "musaStreamLegacy",
    "cudaStreamPerThread": "musaStreamPerThread",

    # CUDA Graph API
    "cudaGraphInstantiate": "musaGraphInstantiate",
    "cudaGraphLaunch": "musaGraphLaunch",
    "cudaGraphCreate": "musaGraphCreate",
    "cudaStreamBeginCapture": "musaStreamBeginCapture",
    "cudaStreamEndCapture": "musaStreamEndCapture",
    "cudaStreamCaptureStatusNone": "musaStreamCaptureStatusNone",
    "cudaErrorStreamCaptureImplicit": "musaErrorStreamCaptureImplicit",
    "cudaStreamIsCapturing": "musaStreamIsCapturing",

    # Error handling
    "cudaGetLastError": "musaGetLastError",
    "cudaPeekAtLastError": "musaPeekAtLastError",
    "cudaError_t": "musaError_t",
    "cudaSuccess": "musaSuccess",

    # Memory management
    "cudaMalloc": "musaMalloc",
    "cudaFree": "musaFree",
    "cudaMemcpy": "musaMemcpy",
    "cudaMemcpyAsync": "musaMemcpyAsync",
    "cudaMemset": "musaMemset",
    "cudaMemsetAsync": "musaMemsetAsync",

    # Device management
    "cudaGetDeviceProperties": "musaGetDeviceProperties",
    "cudaSetDevice": "musaSetDevice",
    "cudaGetDevice": "musaGetDevice",
    "cudaDeviceSynchronize": "musaDeviceSynchronize",
    "cudaGetDeviceCount": "musaGetDeviceCount",

    # Stream/Event
    "cudaStreamCreate": "musaStreamCreate",
    "cudaStreamDestroy": "musaStreamDestroy",
    "cudaStreamSynchronize": "musaStreamSynchronize",
    "cudaEventCreate": "musaEventCreate",
    "cudaEventRecord": "musaEventRecord",
    "cudaEventSynchronize": "musaEventSynchronize",
    "cudaEventDestroy": "musaEventDestroy",
    "cudaEventElapsedTime": "musaEventElapsedTime",

    # Types
    "cudaStream_t": "musaStream_t",
    "cudaEvent_t": "musaEvent_t",
    "cudaDeviceProp": "musaDeviceProp",
    "cudaMemcpyHostToDevice": "musaMemcpyHostToDevice",
    "cudaMemcpyDeviceToHost": "musaMemcpyDeviceToHost",
    "cudaMemcpyDeviceToDevice": "musaMemcpyDeviceToDevice",

    # Graph types
    "cudaGraph_t": "musaGraph_t",
    "cudaGraphExec_t": "musaGraphExec_t",
    "cudaGraphNode_t": "musaGraphNode_t",
    "cudaStreamCaptureStatus": "musaStreamCaptureStatus",
    "cudaStreamCaptureMode": "musaStreamCaptureMode",
    "cudaStreamCaptureModeGlobal": "musaStreamCaptureModeGlobal",

    # Misc CUDA runtime
    "cudaFuncSetAttribute": "musaFuncSetAttribute",
    "cudaFuncAttribute": "musaFuncAttribute",
    "cudaOccupancyMaxActiveBlocksPerMultiprocessor": "musaOccupancyMaxActiveBlocksPerMultiprocessor",

    # Compiler detection
    "__CUDA_ARCH__": "__MUSA_ARCH__",
    "__CUDACC__": "__MUSACC__",

    # Header includes
    "cuda_runtime.h": "musa_runtime.h",
    "cuda_runtime_api.h": "musa_runtime_api.h",
    "cuda_fp16.h": "musa_fp16.h",
    "cuda.h": "musa.h",
    "cublas_v2.h": "mublas.h",

    # CUDA check macro renaming
    "CUDA_CHECK_THROW": "MUSA_CHECK_THROW",

    # CUDA Driver API (Virtual Memory Management)
    "CUdeviceptr": "MUdeviceptr",
    "CUmemGenericAllocationHandle": "MUmemGenericAllocationHandle",
    "CUmemAllocationProp": "MUmemAllocationProp",
    "CUmemAccessDesc": "MUmemAccessDesc",
    "CU_MEM_ALLOCATION_TYPE_PINNED": "MU_MEM_ALLOCATION_TYPE_PINNED",
    "CU_MEM_LOCATION_TYPE_DEVICE": "MU_MEM_LOCATION_TYPE_DEVICE",
    "CU_MEM_ACCESS_FLAGS_PROT_READWRITE": "MU_MEM_ACCESS_FLAGS_PROT_READWRITE",
    "CU_MEM_ALLOCATION_GRANULARITY_MINIMUM": "MU_MEM_ALLOCATION_GRANULARITY_MINIMUM",
    "cuMemAddressReserve": "muMemAddressReserve",
    "cuMemAddressFree": "muMemAddressFree",
    "cuMemCreate": "muMemCreate",
    "cuMemRelease": "muMemRelease",
    "cuMemMap": "muMemMap",
    "cuMemUnmap": "muMemUnmap",
    "cuMemSetAccess": "muMemSetAccess",
    "cuMemGetAllocationGranularity": "muMemGetAllocationGranularity",
    "cuGetErrorName": "muGetErrorName",
    "CUDA_SUCCESS": "MUSA_SUCCESS",

    # CUDA Driver API error handling
    "CU_CHECK_THROW": "MU_CHECK_THROW",
    "CUDA_ERROR_NOT_SUPPORTED": "MUSA_ERROR_NOT_SUPPORTED",

    # CUDA Driver API function attributes
    "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES": "MU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES",
    "CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED": "MU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED",
    "CU_MEM_ALLOC_GRANULARITY_MINIMUM": "MU_MEM_ALLOC_GRANULARITY_MINIMUM",
    "cuFuncSetAttribute": "muFuncSetAttribute",
    "cuDeviceGetAttribute": "muDeviceGetAttribute",
    "cuMemGetAllocationGranularity": "muMemGetAllocationGranularity",

    # CUDA compiler version macros
    "__CUDACC_VER_MAJOR__": "__MUSACC_VER_MAJOR__",
    "__CUDACC_VER_MINOR__": "__MUSACC_VER_MINOR__",
}


def clean_existing_musa_dirs():
    """Remove existing _musa directories before regeneration."""
    for dirname in ["src_musa", "include_musa", "tests_musa", "samples_musa", "benchmarks_musa"]:
        path = os.path.join(PROJECT_DIR, dirname)
        if os.path.exists(path):
            print(f"Removing: {path}")
            shutil.rmtree(path)


def port_directories():
    """Port each CUDA directory using SimplePorting."""
    dirs_to_port = [
        # (cuda_dir, ignore_dirs)
        (os.path.join(PROJECT_DIR, "src"), []),
        (os.path.join(PROJECT_DIR, "include"), []),
        (os.path.join(PROJECT_DIR, "tests"), []),
        (os.path.join(PROJECT_DIR, "samples"), []),
        (os.path.join(PROJECT_DIR, "benchmarks"), []),
    ]
    for cuda_dir, ignore_dirs in dirs_to_port:
        if os.path.exists(cuda_dir):
            print(f"Porting {cuda_dir}")
            SimplePorting(
                cuda_dir_path=cuda_dir,
                ignore_dir_paths=ignore_dirs,
                mapping_rule=MAPPING_RULES,
                drop_default_mapping=False,
            ).run()
            print(f"  Done: {cuda_dir}")


if __name__ == "__main__":
    clean_existing_musa_dirs()
    port_directories()
    print("SimplePorting complete.")
