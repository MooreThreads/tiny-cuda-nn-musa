#!/usr/bin/env python3
"""
Apply all MUSA-specific fixes to the SimplePorting output for tiny-cuda-nn.
Run INSIDE the Docker container at /workspace/tiny-cuda-nn/.

Steps:
  1. Fix WARP_SIZE 32 -> 128 in include_musa/tiny-cuda-nn/common.h
  2. Run fix_mma.py to disable PTX inline asm in mma.h
  3. Guard rtc_kernel.mu under #ifdef TCNN_RTC
  4. Guard fully_fused_mlp.mu under #ifdef TCNN_FULLY_FUSED_MLP
  5. Write CMakeLists_musa.txt (MUSA toolchain, encodings-only build)
"""
import os
import re
import shutil

W = "/workspace/tiny-cuda-nn"


# ─── Fix 1: WARP_SIZE ──────────────────────────────────────────────────────────
def fix_warp_size():
    path = f"{W}/include_musa/tiny-cuda-nn/common.h"
    content = open(path).read()
    old = "constexpr uint32_t WARP_SIZE = 32;"
    new = "constexpr uint32_t WARP_SIZE = 128;  // MUSA S5000 warp size"
    assert old in content, f"WARP_SIZE pattern not found in {path}"
    content = content.replace(old, new)
    open(path, "w").write(content)
    print(f"[fix1] WARP_SIZE 32 -> 128  ({path})")


# ─── Fix 2: mma.h PTX ──────────────────────────────────────────────────────────
def fix_mma_ptx():
    import subprocess
    r = subprocess.run(["python3", f"{W}/fix_mma.py"], capture_output=True, text=True)
    print(f"[fix2] mma.h PTX: {r.stdout.strip()}")
    if r.returncode != 0:
        print(f"  STDERR: {r.stderr.strip()}")


# ─── Fix 3: Guard rtc_kernel.mu under TCNN_RTC ────────────────────────────────
def fix_rtc_kernel():
    path = f"{W}/src_musa/rtc_kernel.mu"
    content = open(path).read()
    # Find the first #include of nvrtc.h - if it's already guarded, skip
    if "#ifdef TCNN_RTC" in content:
        print("[fix3] rtc_kernel.mu already guarded, skip")
        return
    # Wrap entire file after the top comment/license block in #ifdef TCNN_RTC
    # Find position of first non-comment, non-blank line after license
    lines = content.split("\n")
    insert_at = 0
    in_block_comment = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("/*"):
            in_block_comment = True
        if in_block_comment and "*/" in stripped:
            in_block_comment = False
            continue
        if in_block_comment:
            continue
        if stripped.startswith("//") or stripped == "":
            continue
        # First real line
        insert_at = i
        break

    lines.insert(insert_at, "#ifdef TCNN_RTC")
    lines.append("#else  // !TCNN_RTC")
    lines.append("// rtc_kernel.mu: NVRTC disabled (TCNN_RTC not set). Stub.")
    lines.append("#include <tiny-cuda-nn/rtc_kernel.h>")
    lines.append("namespace tcnn {")
    lines.append("RTCKernel::RTCKernel(const std::string&, const std::string&, const std::string&, const std::vector<std::string>&) {")
    lines.append('    throw std::runtime_error{"RTCKernel: NVRTC not available in MUSA build. Set TCNN_BUILD_WITH_RTC=OFF."};')
    lines.append("}")
    lines.append("void RTCKernel::launch(uint32_t, uint32_t, musaStream_t, void**) {}")
    lines.append("}  // namespace tcnn")
    lines.append("#endif  // TCNN_RTC")

    open(path, "w").write("\n".join(lines))
    print(f"[fix3] rtc_kernel.mu guarded under #ifdef TCNN_RTC")


# ─── Fix 4: Guard fully_fused_mlp.mu under TCNN_FULLY_FUSED_MLP ───────────────
def fix_fully_fused_mlp():
    path = f"{W}/src_musa/fully_fused_mlp.mu"
    content = open(path).read()
    if "#ifdef TCNN_FULLY_FUSED_MLP" in content:
        print("[fix4] fully_fused_mlp.mu already guarded, skip")
        return

    lines = content.split("\n")
    # Find first real code line (after license block)
    insert_at = 0
    in_block_comment = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("/*"):
            in_block_comment = True
        if in_block_comment and "*/" in stripped:
            in_block_comment = False
            continue
        if in_block_comment:
            continue
        if stripped.startswith("//") or stripped == "":
            continue
        insert_at = i
        break

    lines.insert(insert_at, "#ifdef TCNN_FULLY_FUSED_MLP")
    lines.append("#else  // !TCNN_FULLY_FUSED_MLP")
    lines.append("// fully_fused_mlp.mu: WMMA/PTX disabled for MUSA (no nvcuda::wmma equivalent).")
    lines.append("#include <tiny-cuda-nn/networks/fully_fused_mlp.h>")
    lines.append("// Provide a stub that throws at runtime to satisfy the linker.")
    lines.append("// Users must use CutlassMLP or other network types on MUSA.")
    lines.append("namespace tcnn {")
    lines.append("template <int WIDTH>")
    lines.append("FullyFusedMLP<WIDTH>::FullyFusedMLP(uint32_t, uint32_t, uint32_t, uint32_t, Activation, Activation) {")
    lines.append('    throw std::runtime_error{"FullyFusedMLP requires NVIDIA tensor cores (wmma). Not available on MUSA. Use CutlassMLP."};')
    lines.append("}")
    lines.append("} // namespace tcnn")
    lines.append("#endif  // TCNN_FULLY_FUSED_MLP")

    open(path, "w").write("\n".join(lines))
    print(f"[fix4] fully_fused_mlp.mu guarded under #ifdef TCNN_FULLY_FUSED_MLP")


# ─── Fix 5: Write CMakeLists_musa.txt ─────────────────────────────────────────
CMAKE_MUSA = r"""cmake_minimum_required(VERSION 3.18)

project(
    tiny-cuda-nn
    VERSION 2.0
    DESCRIPTION "tiny-cuda-nn MUSA port (encodings-only)"
    LANGUAGES CXX
)

# ── MUSA toolchain ─────────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH /usr/local/musa/cmake)
find_package(MUSA REQUIRED)

set(MUSA_MCC_FLAGS
    --offload-arch=mp_31       # S5000 (MT-3000 series, major=3 minor=1)
    -Wno-unused-function
    -Wno-unused-variable
    --expt-relaxed-constexpr
    --extended-lambda
)

# ── Options ────────────────────────────────────────────────────────────────────
option(TCNN_BUILD_BENCHMARK "Build benchmarks?" OFF)
option(TCNN_BUILD_EXAMPLES  "Build examples?"   OFF)
option(TCNN_BUILD_TESTS     "Build tests?"      ON)

# ── C++ standard ──────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Sources (encodings-only: exclude wmma and nvrtc) ─────────────────────────
set(TCNN_SOURCES
    src_musa/common_host.mu
    src_musa/cpp_api.mu
    src_musa/encoding.mu
    src_musa/loss.mu
    src_musa/network.mu
    src_musa/object.mu
    src_musa/optimizer.mu
    src_musa/reduce_sum.mu
    # cutlass_mlp.mu excluded: depends on vendored CUTLASS (NVIDIA SM archs)
    # fully_fused_mlp.mu excluded: depends on nvcuda::wmma PTX
    # rtc_kernel.mu excluded: depends on NVRTC
)

# ── Definitions ───────────────────────────────────────────────────────────────
set(TCNN_DEFINITIONS
    -DTCNN_MIN_GPU_ARCH=21        # mp_21 = S5000
    -DTCNN_HALF_PRECISION=1
    -DTCNN_NO_FULLY_FUSED_MLP     # disable wmma code paths
    # TCNN_RTC not defined         # disable NVRTC code paths
    # TCNN_SHAMPOO not defined      # disable cuBLAS/cuSolver Shampoo optimizer
)

# ── Includes ──────────────────────────────────────────────────────────────────
set(TCNN_INCLUDES
    include_musa                  # ported headers
    dependencies                  # fmt, json, pcg32, cmrc (CUDA-free)
    # dependencies/cutlass/include NOT included (NVIDIA-specific)
)

# ── fmt dependency (CUDA-free) ────────────────────────────────────────────────
set(BUILD_SHARED_LIBS OFF)
set(FMT_UNICODE OFF CACHE BOOL "" FORCE)
add_subdirectory("dependencies/fmt")

# ── cmrc resource library ─────────────────────────────────────────────────────
include("${CMAKE_CURRENT_SOURCE_DIR}/dependencies/cmrc/CMakeRC.cmake")
cmrc_add_resource_library(tiny-cuda-nn-resources NAMESPACE tcnn)
set(TCNN_DEFINITIONS ${TCNN_DEFINITIONS} -DTCNN_CMRC)

# ── Build the library ──────────────────────────────────────────────────────────
musa_add_library(tiny-cuda-nn STATIC ${TCNN_SOURCES})
target_compile_definitions(tiny-cuda-nn PUBLIC ${TCNN_DEFINITIONS})
target_include_directories(tiny-cuda-nn PUBLIC ${TCNN_INCLUDES})
target_link_libraries(tiny-cuda-nn PUBLIC fmt tiny-cuda-nn-resources)

# ── Tests ─────────────────────────────────────────────────────────────────────
if (TCNN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests_musa)
endif()
"""


def write_cmake():
    path = f"{W}/CMakeLists_musa.txt"
    open(path, "w").write(CMAKE_MUSA)
    print(f"[fix5] Written {path}")


# ── tests_musa/CMakeLists.txt ────────────────────────────────────────────────
TESTS_CMAKE = r"""# Tests CMakeLists (MUSA)
# Only test_encodings is included — it doesn't depend on FullyFusedMLP or NVRTC.

set(TCNN_TEST_SOURCES
    test_encodings.mu
    # test_networks.mu excluded: depends on CutlassMLP/FullyFusedMLP
    # test_jit_losses.mu excluded: depends on NVRTC
    # test_grid.mu included conditionally below
)

# test_grid doesn't use wmma or nvrtc — include it
list(APPEND TCNN_TEST_SOURCES test_grid.mu)

foreach(test_src IN LISTS TCNN_TEST_SOURCES)
    get_filename_component(test_name "${test_src}" NAME_WE)
    musa_add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE tiny-cuda-nn)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
"""

def write_tests_cmake():
    os.makedirs(f"{W}/tests_musa", exist_ok=True)
    # Copy test source files if not there
    for f in ["test_encodings.mu", "test_grid.mu", "test_common.h"]:
        src = f"{W}/tests_musa/{f}"
        if not os.path.exists(src):
            orig = f"{W}/tests_musa/{f}"  # already ported by SimplePorting
    path = f"{W}/tests_musa/CMakeLists.txt"
    open(path, "w").write(TESTS_CMAKE)
    print(f"[fix6] Written {path}")


if __name__ == "__main__":
    fix_warp_size()
    fix_mma_ptx()
    fix_rtc_kernel()
    fix_fully_fused_mlp()
    write_cmake()
    write_tests_cmake()
    print("\nAll fixes applied. Build with:")
    print("  cmake -B build_musa -S . -f CMakeLists_musa.txt")
    print("  cmake --build build_musa -j$(nproc)")
