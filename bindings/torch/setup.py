import os

import re
from setuptools import setup
from pkg_resources import parse_version
import subprocess
import shutil
import sys
import torch
from glob import glob

FORCE_MUSA = os.getenv("FORCE_MUSA", "0") == "1"

if FORCE_MUSA:
	from torch_musa.utils.musa_extension import MUSAExtension as Extension, BuildExtension
	from torch_musa.testing import get_musa_arch
else:
	from torch.utils.cpp_extension import BuildExtension, CUDAExtension as Extension

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Find version of tinycudann by scraping CMakeLists.txt
with open(os.path.join(ROOT_DIR, "CMakeLists.txt"), "r") as cmakelists:
	for line in cmakelists.readlines():
		if line.strip().startswith("VERSION"):
			VERSION = line.split("VERSION")[-1].strip()
			break

print(f"Building PyTorch extension for tiny-cuda-nn version {VERSION}")

ext_modules = []

if FORCE_MUSA:
	# MUSA build path
	musa_arch = get_musa_arch()
	os.environ['MUSA_ARCH'] = str(musa_arch)
	print(f"Building for MUSA architecture: mp_{musa_arch}")
	compute_capabilities = [musa_arch]
else:
	if "TCNN_CUDA_ARCHITECTURES" in os.environ and os.environ["TCNN_CUDA_ARCHITECTURES"]:
		compute_capabilities = [int(x) for x in os.environ["TCNN_CUDA_ARCHITECTURES"].replace(";", ",").split(",")]
		print(f"Obtained compute capabilities {compute_capabilities} from environment variable TCNN_CUDA_ARCHITECTURES")
	elif torch.cuda.is_available():
		major, minor = torch.cuda.get_device_capability()
		compute_capabilities = [major * 10 + minor]
		print(f"Obtained compute capability {compute_capabilities[0]} from PyTorch")
	else:
		raise EnvironmentError("Unknown compute capability. Specify the target compute capabilities in the TCNN_CUDA_ARCHITECTURES environment variable or install PyTorch with the CUDA backend to detect it automatically.")

include_networks = True
if "--no-networks" in sys.argv:
	include_networks = False
	sys.argv.remove("--no-networks")
	print("Building >> without << neural networks (just the input encodings)")

cpp_standard = 17

print(f"Targeting C++ standard {cpp_standard}")

# List of sources.
bindings_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(bindings_dir, "../.."))

if FORCE_MUSA:
	# MUSA build: use ported _musa source files, disable RTC (no nvrtc equivalent)
	base_definitions = [
		"-DTCNN_PARAMS_UNALIGNED",
		"-DFMT_USE_FLOAT128=0",
		# RTC disabled - no MUSA equivalent for nvrtc
	]

	min_compute_capability = min(compute_capabilities) if compute_capabilities else 22
	enable_half = True
	base_definitions.append(f"-DTCNN_HALF_PRECISION={int(enable_half)}")

	# For MUSA build, networks are disabled because:
	# 1. cutlass_mlp requires NVIDIA CUTLASS headers (cuda_fp16.h) - need mutlass integration
	# 2. fully_fused_mlp requires PTX mma.sync/wmma instructions
	# Encodings (the most commonly used feature) still work.
	include_networks = False

	base_source_files = [
		"tinycudann/bindings.cpp",
		"../../dependencies/fmt/src/format.cc",
		"../../dependencies/fmt/src/os.cc",
		"../../src_musa/cpp_api.mu",
		"../../src_musa/common_host.mu",
		"../../src_musa/encoding.mu",
		"../../src_musa/object.mu",
		"../../src_musa/rtc_kernel.mu",
	]

	base_definitions.append("-DTCNN_NO_NETWORKS")

	base_mcc_flags = [
		f"-std=c++{cpp_standard}",
		"-Xcompiler=-Wno-float-conversion",
		"-Xcompiler=-fno-strict-aliasing",
		"-Xcompiler=-Wno-error",
	]

	base_cflags = [f"-std=c++{cpp_standard}", "-DFORCE_MUSA=1"]

	def make_extension(compute_capability):
		mcc_flags = base_mcc_flags[:]
		definitions = base_definitions + [f"-DTCNN_MIN_GPU_ARCH={compute_capability}", "-DFORCE_MUSA=1"]

		# fully_fused_mlp requires PTX mma.sync/wmma instructions not available on MUSA
		# Always exclude from MUSA builds; cutlass_mlp is the fallback
		source_files = base_source_files

		mcc_flags = mcc_flags + definitions
		cflags = base_cflags + definitions

		ext = Extension(
			name=f"tinycudann_bindings._{compute_capability}_C",
			sources=source_files,
			include_dirs=[
				f"{root_dir}/include_musa",
				f"{root_dir}/include",
				f"{root_dir}/dependencies",
				f"{root_dir}/dependencies/fmt/include",
			],
			extra_compile_args={"cxx": cflags, "mcc": mcc_flags},
			libraries=["musa"],
		)
		return ext

	ext_modules = [make_extension(comp) for comp in compute_capabilities]

else:
	# Original CUDA build path
	def min_supported_compute_capability(cuda_version):
		if cuda_version >= parse_version("13.0"):
			return 75
		elif cuda_version >= parse_version("12.0"):
			return 50
		else:
			return 20

	def max_supported_compute_capability(cuda_version):
		if cuda_version < parse_version("11.0"):
			return 75
		elif cuda_version < parse_version("11.1"):
			return 80
		elif cuda_version < parse_version("11.8"):
			return 86
		elif cuda_version < parse_version("12.8"):
			return 90
		else:
			return 120

	if os.name == "nt":
		def find_cl_path():
			import glob as gl
			for executable in ["Program Files (x86)", "Program Files"]:
				for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
					paths = sorted(gl.glob(f"C:\\{executable}\\Microsoft Visual Studio\\*\\{edition}\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64"), reverse=True)
					if paths:
						return paths[0]

		if shutil.which("cl.exe") is None:
			cl_path = find_cl_path()
			if cl_path is None:
				raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
			os.environ["PATH"] += ";" + cl_path
		else:
			os.environ["DISTUTILS_USE_SDK"] = "1"

	min_compute_capability = min(compute_capabilities)

	base_nvcc_flags = [
		f"-std=c++{cpp_standard}",
		"--extended-lambda",
		"--use_fast_math",
		"--expt-relaxed-constexpr",
		"-U__CUDA_NO_HALF_OPERATORS__",
		"-U__CUDA_NO_HALF_CONVERSIONS__",
		"-U__CUDA_NO_HALF2_OPERATORS__",
	]

	if os.name == "posix":
		base_cflags = [f"-std=c++{cpp_standard}"]
		base_nvcc_flags += [
			"-Xcompiler=-Wno-float-conversion",
			"-Xcompiler=-fno-strict-aliasing",
		]
	elif os.name == "nt":
		base_cflags = [f"/std:c++{cpp_standard}"]

	os.environ["TORCH_CUDA_ARCH_LIST"] = ""

	base_definitions = [
		"-DTCNN_PARAMS_UNALIGNED",
		"-DTCNN_RTC",
		"-DTCNN_RTC_USE_FAST_MATH",
	]

	if "TCNN_HALF_PRECISION" in os.environ:
		enable_half = os.environ["TCNN_HALF_PRECISION"].lower() in ["1", "true", "on", "yes"]
		base_definitions.append(f"-DTCNN_HALF_PRECISION={int(enable_half)}")
		print(f"Forcing TCNN_HALF_PRECISION to {'ON' if enable_half else 'OFF'}")
	else:
		if min_compute_capability == 61 or min_compute_capability <= 52:
			enable_half = False
		else:
			enable_half = True
		print(f"Auto-detecting TCNN_HALF_PRECISION: {'ON' if enable_half else 'OFF'} (Arch: {min_compute_capability})")
	base_definitions.append(f"-DTCNN_HALF_PRECISION={int(enable_half)}")

	base_source_files = [
		"tinycudann/bindings.cpp",
		"../../dependencies/fmt/src/format.cc",
		"../../dependencies/fmt/src/os.cc",
		"../../src/cpp_api.cu",
		"../../src/common_host.cu",
		"../../src/encoding.cu",
		"../../src/object.cu",
		"../../src/rtc_kernel.cu",
	]

	if include_networks:
		base_source_files += [
			"../../src/network.cu",
			"../../src/cutlass_mlp.cu",
		]
	else:
		base_definitions.append("-DTCNN_NO_NETWORKS")

	# Copy headers required by RTC at runtime
	rtc_dir = os.path.join(bindings_dir, "tinycudann", "rtc")
	rtc_include_dir = os.path.join(rtc_dir, "include")
	rtc_cache_dir = os.path.join(rtc_dir, "cache")
	shutil.rmtree(rtc_dir, ignore_errors=True)
	os.makedirs(rtc_include_dir, exist_ok=True)
	os.makedirs(rtc_cache_dir, exist_ok=True)

	nvcc_path = shutil.which("nvcc")
	if nvcc_path is None:
		print(f"WARNING: could not find CUDA include directory. JIT compilation will not be supported.")
	else:
		cuda_include_dir = os.path.join(os.path.dirname(os.path.dirname(nvcc_path)), "include")
		cuda_headers = glob(f"{cuda_include_dir}/cuda_fp16*") + glob(f"{cuda_include_dir}/vector*")
		tcnn_headers = glob(f"{root_dir}/include/tiny-cuda-nn/*", recursive=True)
		pcg32_headers = glob(f"{root_dir}/dependencies/pcg32/*")

		def copy_files(whence, files):
			for h in files:
				if not os.path.isfile(h):
					continue
				tgt = os.path.join(rtc_include_dir, os.path.relpath(h, whence))
				os.makedirs(os.path.dirname(tgt), exist_ok=True)
				shutil.copyfile(h, tgt)

		copy_files(cuda_include_dir, cuda_headers)
		copy_files(f"{root_dir}/include", tcnn_headers)
		copy_files(f"{root_dir}/dependencies", pcg32_headers)

	def make_extension(compute_capability):
		nvcc_flags = base_nvcc_flags + [f"-gencode=arch=compute_{compute_capability},code={code}_{compute_capability}" for code in ["compute", "sm"]]
		definitions = base_definitions + [f"-DTCNN_MIN_GPU_ARCH={compute_capability}"]

		if include_networks and compute_capability > 70:
			source_files = base_source_files + ["../../src/fully_fused_mlp.cu"]
		else:
			source_files = base_source_files

		nvcc_flags = nvcc_flags + definitions
		cflags = base_cflags + definitions

		ext = Extension(
			name=f"tinycudann_bindings._{compute_capability}_C",
			sources=source_files,
			include_dirs=[
				f"{root_dir}/include",
				f"{root_dir}/dependencies",
				f"{root_dir}/dependencies/cutlass/include",
				f"{root_dir}/dependencies/cutlass/tools/util/include",
				f"{root_dir}/dependencies/fmt/include",
			],
			extra_compile_args={"cxx": cflags, "nvcc": nvcc_flags},
			libraries=["cuda", "nvrtc"],
		)
		return ext

	ext_modules = [make_extension(comp) for comp in compute_capabilities]

def package_files(directory):
	paths = []
	for (path, _, filenames) in os.walk(directory):
		for filename in filenames:
			paths.append(os.path.join('..', path, filename))
	return paths

if not FORCE_MUSA:
	rtc_dir = os.path.join(bindings_dir, "tinycudann", "rtc")
	package_data_files = package_files(rtc_dir)
else:
	package_data_files = []

setup(
	name="tinycudann",
	version=VERSION,
	description="tiny-cuda-nn extension for PyTorch",
	long_description="tiny-cuda-nn extension for PyTorch",
	classifiers=[
		"Development Status :: 4 - Beta",
		"License :: BSD 3-Clause",
		"Programming Language :: C++",
		"Programming Language :: Python :: 3 :: Only",
		"Topic :: Multimedia :: Graphics",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
		"Topic :: Scientific/Engineering :: Image Processing",
	],
	keywords="PyTorch,machine learning",
	url="https://github.com/nvlabs/tiny-cuda-nn",
	author="Thomas Mueller, Jacob Munkberg, Jon Hasselgren, Or Perel",
	author_email="tmueller@nvidia.com, jmunkberg@nvidia.com, jhasselgren@nvidia.com, operel@nvidia.com",
	maintainer="Thomas Mueller",
	maintainer_email="tmueller@nvidia.com",
	download_url=f"https://github.com/nvlabs/tiny-cuda-nn",
	license="BSD 3-Clause",
	packages=["tinycudann"],
	package_data={"": package_data_files},
	install_requires=[],
	include_package_data=True,
	zip_safe=False,
	ext_modules=ext_modules,
	cmdclass={"build_ext": BuildExtension}
)
