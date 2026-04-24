/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 * BSD 3-clause — see original rtc_kernel.cu header.
 *
 * MUSA port (auto-musify round 2):
 *   Runtime-compilation helper. Upstream uses NVRTC + CUDA driver module-load
 *   with PTX as the intermediate form. On MUSA:
 *
 *     NVRTC → MTRTC (mtrtc*) — same API shape (prog/compile/log/lowered name)
 *     PTX → fatbin — mtrtcGetPTX* is absent; mtrtcGetFatBin{,Size} is the
 *                    authoritative output. The MUSA driver accepts the fatbin
 *                    via muModuleLoadFatBinary.
 *     --gpu-architecture=compute_<cc>  →  --gpu-architecture=mp_<ARCH>
 *          (S5000 = mp_21; queried at runtime from musa device property rather
 *           than hardcoded so the same object file works on S4000 / future HW).
 *
 *   Rule-3 parallel-folder porting: original rtc_kernel.cu (CUDA/NVRTC)
 *   remains untouched under src/; this file under src_musa/ is the MUSA
 *   replacement.
 *
 *   Rule-4 API mappings: every mtrtc* name used below was looked up in
 *   /usr/local/musa/include/mtrtc.h (SDK 5.1.0 ships nvrtc-shaped API with the
 *   mtrtc prefix). Fatbin-not-PTX behavior is documented there as well
 *   (mtrtcGetPTX* commented out; mtrtcGetFatBin is the live entry point).
 */

#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/rtc_kernel.h>

#ifdef TCNN_CMRC
#include <cmrc/cmrc.hpp>
CMRC_DECLARE(tcnn);
#endif

#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>

namespace tcnn {
std::string& rtc_cache_dir() {
	static std::string cache_dir = "";
	return cache_dir;
}

void rtc_set_cache_dir(const std::string& dir) {
	rtc_cache_dir() = dir;
}

std::string& rtc_include_dir() {
	static std::string header_dir = "";
	return header_dir;
}

void rtc_set_include_dir(const std::string& dir) {
	rtc_include_dir() = dir;
}
}

#ifndef TCNN_RTC
namespace tcnn {

bool supports_jit_fusion(int device) {
	return false;
}

CudaRtcKernel::CudaRtcKernel(const std::string& name, const std::string& kernel_code, const std::vector<std::pair<std::string, const char*>>& extra_includes) {
	throw std::runtime_error{"tiny-cuda-nn was not compiled with runtime compilation (RTC) support."};
}

CudaRtcKernel::~CudaRtcKernel() {}
void CudaRtcKernel::clear() {}

}
#else
#include <mtrtc.h>
#include <musa.h>
#include <musa_runtime.h>

/// Checks the result of an mtrtc* call and throws on failure.
#define MTRTC_CHECK_THROW(x)                                                                                              \
	do {                                                                                                                    \
		mtrtcResult _result = (x);                                                                                            \
		if (_result != MTRTC_SUCCESS)                                                                                         \
			throw std::runtime_error{std::string(FILE_LINE " " #x " failed with error ") + mtrtcGetErrorString(_result)};    \
	} while (0)

namespace tcnn {

// Pick an mtrtc --gpu-architecture tag. mtrtc accepts {mp_10, mp_21, mp_22,
// mp_31}. Map the MUSA device's (major, minor) compute capability to the
// nearest supported mp tag. Observed at runtime on SDK 5.1.0:
//   S5000 → (3,1) → mp_31   (PH1 — was (2,1) under older SDKs; bumped)
//   S4000 → (2,2) → mp_22   (QY2)
//   S90   → (2,2) → mp_22   (QY2)
// The mapping is device-driven — do NOT hardcode per build-time
// `--offload-arch` choice; mtrtc JIT and mcc static-compile can legitimately
// differ (see Round 7 issue-007 handoff).
static std::string musa_mtrtc_arch_for_device(int device) {
	musaDeviceProp prop{};
	if (musaGetDeviceProperties(&prop, device) != musaSuccess) {
		// Fall back to mp_31 — SDK 5.1.0's canonical PH1 JIT arch. Older
		// SDKs where S5000 reported (2,1) are out of support scope.
		return "mp_31";
	}
	int code = prop.major * 10 + prop.minor;
	switch (code) {
		case 10: return "mp_10";
		case 21: return "mp_21";
		case 22: return "mp_22";
		case 31: return "mp_31";
		default:
			// Unknown arch: pick the newest mtrtc-supported tag. mtrtc's runtime
			// is forward-tolerant — if the HW is newer than mp_31, producing
			// mp_31 fatbin still loads (PH2 extends PH1's ISA).
			return "mp_31";
	}
}

bool supports_jit_fusion(int device) {
	// On MUSA the CUDA-era "Turing+" gate (compute >= 75) doesn't translate
	// directly. We require TCNN_HALF_PRECISION (kernel fusion relies on
	// __half), and confirm the device exists + has a supported arch tag.
	if (!TCNN_HALF_PRECISION) {
		return false;
	}
	musaDeviceProp prop{};
	if (musaGetDeviceProperties(&prop, device) != musaSuccess) {
		return false;
	}
	// S4000 (mp_22) and S5000 (mp_21) both support fused MLP JIT. Keep this
	// conservative and allow anything mtrtc recognizes.
	auto arch = musa_mtrtc_arch_for_device(device);
	return arch != "mp_10";  // mp_10 is legacy / not a real shipping SKU
}

#ifdef TCNN_CMRC
std::vector<std::pair<std::string, const char*>> all_files(const cmrc::embedded_filesystem& fs, const std::string& dir) {
	std::vector<std::pair<std::string, const char*>> result;
	for (auto&& entry : fs.iterate_directory(dir)) {
		auto fn = dir.empty() ? entry.filename() : fmt::format("{}/{}", dir, entry.filename());
		if (entry.is_file()) {
			result.emplace_back(fn, fs.open(fn).begin());
		} else if (entry.is_directory()) {
			auto files_in_dir = all_files(fs, fn);
			result.insert(std::end(result), std::begin(files_in_dir), std::end(files_in_dir));
		}
	}

	return result;
}
#endif

// If a cache dir is provided, compilation artifacts are cached in there and
// re-loaded upon program restart. Improves user experience.
CudaRtcKernel::CudaRtcKernel(const std::string& name, const std::string& kernel_code, const std::vector<std::pair<std::string, const char*>>& extra_includes) {
	if (!supports_jit_fusion()) {
		throw std::runtime_error{fmt::format(
			"JIT: unsupported on this MUSA device. Requires TCNN_HALF_PRECISION and a device mtrtc recognizes (mp_21/mp_22/mp_31)."
		)};
	}

	ScopeGuard cleanup_guard{[&]() { clear(); }};

	const auto start_time = std::chrono::steady_clock::now();

	std::string arch_tag = musa_mtrtc_arch_for_device(cuda_device());

	// Round 3: mtrtc 14.0 rejects the short-form `-DNAME=VALUE` spelling (per
	// verifier bug instant-ngp-003; verified empirically at runtime even
	// though /usr/local/musa/include/mtrtc.h docs `-D` as a short form). The
	// robust fix is to bake the defines directly into the generated source
	// preamble as literal `#define` directives — that avoids depending on
	// mtrtc's argv parser behavior entirely. For include-path, use the
	// long-form `--include-path=<dir>` spelling documented in the same
	// header to stay on the conservative side of the same parser.
	//
	// Macros baked below are injected via the {RTC_DEFINES} slot of
	// complete_code; the options vector below carries only arch/misc.
	//
	// Round 4: mtrtc additionally rejects every documented `--std=` /
	// `-std=` / `--std <val>` form for both c++14 and c++17 (verifier bug
	// instant-ngp-005). Empirically confirmed via a standalone mtrtc probe
	// (5 variants tried, all INVALID_OPTION; only the no-flag variant
	// succeeded). Dropping the flag entirely is the only working path on
	// SDK 5.1.0; mtrtc's built-in default is ≥ C++17 and correctly compiles
	// tcnn's generated code without an explicit --std. If a future SDK
	// accepts any spelling, the flag can be reintroduced in that spelling.
	//
	// Round 5: comprehensive options probe run in-container (fork-isolated
	// to survive mtrtc's allocator-corrupting-on-INVALID_OPTION behavior)
	// enumerated every tcnn/NVRTC-era option. Authoritative accept/reject
	// set for mtrtc 14.0 / SDK 5.1.0 recorded in MUSIFY.md § Knowledge Base
	// [Round 5]. Summary: `--gpu-architecture=`, `--use_fast_math`,
	// `--include-path=`, `--disable-warnings`/`-w`, `--ftz=` are accepted;
	// `--extra-device-vectorization`, `-I<dir>` (short form), every
	// `--prec-*`/`--fmad=`, `--maxrregcount=`, `--ptxas-options`/`-Xptxas`,
	// `--std=*`, `-D*`, and surprisingly the header-documented
	// `--define-macro=` are all rejected.
	//
	// Applied drops: `--extra-device-vectorization` (was line 188). Kept:
	// `--use_fast_math` (gated, PASS), `--include-path=<dir>` (PASS, long
	// form used — short form `-I<dir>` rejected).
	std::vector<std::string> opts = {
		fmt::format("--gpu-architecture={}", arch_tag),
		// --std=c++14 intentionally omitted — see Round 4 comment above.
#ifdef TCNN_RTC_USE_FAST_MATH
		"--use_fast_math",
#endif
		// --extra-device-vectorization intentionally omitted — see Round 5
		// comment above. Dropped per verifier bug instant-ngp-006; probe
		// confirmed no mtrtc-accepted spelling exists for this NVRTC flag.
	};

	if (!rtc_include_dir().empty()) {
		opts.emplace_back(fmt::format("--include-path={}", rtc_include_dir()));
	}

	// Round 3: defines moved from the options vector into the source preamble
	// as literal #define lines (see long comment on options construction
	// above). Join so they sit before generate_device_code_preamble()'s
	// contents — tcnn's kernel templates expect TCNN_HALF_PRECISION and
	// TCNN_MIN_GPU_ARCH to be visible before any tcnn header is included.
	std::string rtc_defines = fmt::format(
		"#define TCNN_HALF_PRECISION {}\n"
		"#define TCNN_MIN_GPU_ARCH {}\n",
		TCNN_HALF_PRECISION, TCNN_MIN_GPU_ARCH
	);

	std::string complete_code = dfmt(0, R"(
			/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
			 */

			/** @file   {KERNEL_NAME}.mu
			 *  @author Thomas Müller, NVIDIA
			 *  @brief  Automatically generated kernel {KERNEL_NAME}
			 */

			/* Compiler options
			{OPTS}
			*/

			// Round 3: defines injected here instead of via options vector —
			// mtrtc 14.0 rejected `-DNAME=VALUE` at runtime.
			{RTC_DEFINES}

			// MTRTC does not come with the C++ standard library out of the box and
			// it would be troublesome to bundle it or require users to have it
			// installed in readily available paths. So we instead include a minimal
			// custom implementation of just those functions of std:: that we require.
			#include <tiny-cuda-nn/ministd.h>
			{PREAMBLE}

			{KERNEL_CODE}
		)",
		"KERNEL_NAME"_a = name,
		"RTC_DEFINES"_a = rtc_defines,
		"PREAMBLE"_a = generate_device_code_preamble(),
		"OPTS"_a = join(opts, "\n"),
		"KERNEL_CODE"_a = kernel_code
	);

	size_t code_hash = hash_combine(0, complete_code);

	std::vector<const char*> headers = {
		// Define header names that we wish mtrtc to treat as empty (they won't
		// be available at runtime); ministd.h above provides the subset of the
		// STL that we actually use.
		"algorithm", "cassert", "cmath", "cstddef", "cstdint", "cstdio", "musa.h", "limits", "type_traits", "initializer_list",
	};
	std::vector<const char*> headers_content(headers.size(), "");

	// Add all headers bundled with tcnn and anything the caller supplied.
	std::vector<std::pair<std::string, const char*>> includes;
#if TCNN_CMRC
	includes = all_files(cmrc::tcnn::get_filesystem());
#endif
	includes.insert(std::end(includes), std::begin(extra_includes), std::end(extra_includes));
	for (const auto& entry : includes) {
		headers.emplace_back(entry.first.c_str());
		headers_content.emplace_back(entry.second);
		code_hash = hash_combine(code_hash, std::string{entry.second});
	}

	std::string code_hash_string = fmt::format("{:016x}", code_hash);
	std::string filename = fmt::format("{}.{}.mu", name, code_hash_string);

	std::string cache_dir = rtc_cache_dir();
	bool use_cache = !cache_dir.empty();

	std::string cached_code_filename = fmt::format("{}/{}", cache_dir, filename);
	// Cache the fatbin (not PTX) — mtrtc emits fatbin only. Extension kept
	// deliberately-different from the CUDA/.ptx form to avoid cross-pollution
	// when a repo is re-built on alternating toolchains.
	std::string cached_fatbin_filename = fmt::format("{}/{}.{}.fatbin", cache_dir, name, code_hash_string);

	std::vector<char> fatbin;
	std::string lowered_kernel_name;

	if (use_cache) {
		{
			std::ifstream f{cached_code_filename};
			if (!f) {
				// Dump source code to disk for easier debugging + caching.
				std::ofstream of{cached_code_filename};
				of << complete_code;
			}
		}

		// Check if we cached the resulting fatbin last time around and load it.
		std::ifstream f{cached_fatbin_filename, std::ios::binary};
		if (f) {
			// The first line of the cached fatbin contains a comment of the form
			//   `//lowered_kernel_name=<value>\n` which we need to parse. This
			// is a tcnn sidecar convention preserved from the CUDA path — the
			// name prefix is stripped off before feeding to muModuleLoadFatBinary.
			std::string first_line;
			std::getline(f, first_line);
			auto s = split(first_line, "=");
			if (s.size() == 2 && s[0].find("lowered_kernel_name") != std::string::npos) {
				lowered_kernel_name = s[1];
				f.seekg(0, std::ios::end);
				size_t size = f.tellg();
				fatbin.resize(size, '\0');
				f.seekg(0);
				f.read(fatbin.data(), size);
			}
		}
	}

	// If we haven't loaded fatbin from cache, compile the program.
	bool cached_fatbin = !fatbin.empty();
	if (!cached_fatbin) {
		mtrtcProgram prog;
		MTRTC_CHECK_THROW(mtrtcCreateProgram(&prog, complete_code.c_str(), filename.c_str(), (int)headers.size(), headers_content.data(), headers.data()));

		MTRTC_CHECK_THROW(mtrtcAddNameExpression(prog, name.c_str()));

		std::vector<const char*> opts_c_str;
		for (const auto& opt : opts) {
			opts_c_str.emplace_back(opt.c_str());
		}

		mtrtcResult compile_result = mtrtcCompileProgram(prog, (int)opts_c_str.size(), opts_c_str.data());
		if (compile_result != MTRTC_SUCCESS) {
			size_t log_size;
			MTRTC_CHECK_THROW(mtrtcGetProgramLogSize(prog, &log_size));
			std::vector<char> log(log_size + 1, '\0');
			MTRTC_CHECK_THROW(mtrtcGetProgramLog(prog, log.data()));
			throw std::runtime_error{fmt::format("JIT: compiling {} failed:\n{}", filename, log.data())};
		}

		const char* lowered_kernel_name_cstr;
		MTRTC_CHECK_THROW(mtrtcGetLoweredName(prog, name.c_str(), &lowered_kernel_name_cstr));
		lowered_kernel_name = lowered_kernel_name_cstr;

		size_t fatbin_size;
		MTRTC_CHECK_THROW(mtrtcGetFatBinSize(prog, &fatbin_size));
		fatbin.resize(fatbin_size, '\0');
		MTRTC_CHECK_THROW(mtrtcGetFatBin(prog, fatbin.data()));

		std::string lowered_kernel_name_comment = fmt::format("//lowered_kernel_name={}\n", lowered_kernel_name);
		fatbin.insert(std::begin(fatbin), std::begin(lowered_kernel_name_comment), std::end(lowered_kernel_name_comment));

		MTRTC_CHECK_THROW(mtrtcDestroyProgram(&prog));

		if (use_cache) {
			std::ofstream f{cached_fatbin_filename, std::ios::binary};
			if (f) {
				f.write(fatbin.data(), fatbin.size());
			}
		}
	}

	// Strip the `//lowered_kernel_name=<value>\n` prefix before feeding to the
	// MUSA driver — muModuleLoadFatBinary wants a pure fatbin image.
	const char* fatbin_begin = fatbin.data();
	size_t fatbin_len = fatbin.size();
	{
		// Find the first newline; everything after it is the actual fatbin.
		const char* nl = static_cast<const char*>(std::memchr(fatbin_begin, '\n', fatbin_len));
		if (nl && std::string(fatbin_begin, nl - fatbin_begin).find("lowered_kernel_name=") != std::string::npos) {
			size_t consumed = (size_t)(nl - fatbin_begin) + 1;
			fatbin_begin += consumed;
			fatbin_len   -= consumed;
		}
	}

	CU_CHECK_THROW(muModuleLoadFatBinary(&m_module, fatbin_begin));
	CU_CHECK_THROW(muModuleGetFunction(&m_kernel, m_module, lowered_kernel_name.c_str()));

	float compilation_duration_seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - start_time).count();
	if (!cached_fatbin || compilation_duration_seconds > 0.1f) { // Don't spam if load-from-cache was fast.
		log_success("JIT: compiled {} in {:.02f}s (cached_fatbin={})", filename, compilation_duration_seconds, cached_fatbin);
	}

	cleanup_guard.disarm();
}

CudaRtcKernel::~CudaRtcKernel() {
	clear();
}

void CudaRtcKernel::clear() {
	if (m_module) {
		muModuleUnload(m_module);
		m_module = {};
	}
}

void CudaRtcKernel::set(MUfunction_attribute attrib, int value) {
	CU_CHECK_THROW(muFuncSetAttribute(m_kernel, attrib, value));
}
}
#endif
