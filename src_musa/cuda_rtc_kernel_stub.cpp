// MUSA stub: CudaRtcKernel -- provides link stubs when TCNN_RTC is not defined.
// Compiled as plain C++ (not .mu) to avoid musa_add_library cmake configure-time
// compilation issues (which run before _cmrc headers are generated).
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <musa.h>
#include <tiny-cuda-nn/rtc_kernel.h>

// ngp / tcnn both call `all_files(cmrc::embedded_filesystem&, const std::string&)`
// unconditionally — only under `if (m_jit_fusion)` at runtime, but linker
// needs the symbol. Since supports_jit_fusion() returns false on MUSA, the
// function is never actually invoked; return empty.
#include <cmrc/cmrc.hpp>

namespace tcnn {

std::vector<std::pair<std::string, const char*>> all_files(
    const cmrc::embedded_filesystem& /*fs*/,
    const std::string& /*dir*/)
{
    return {};
}

} // namespace tcnn

namespace tcnn {

bool supports_jit_fusion(int /*device*/) {
    return false;
}

// RTC cache/include dir stubs -- no-ops since NVRTC is not available.
static std::string s_rtc_cache_dir;
static std::string s_rtc_include_dir;

std::string& rtc_cache_dir() { return s_rtc_cache_dir; }
void rtc_set_cache_dir(const std::string& dir) { s_rtc_cache_dir = dir; }

std::string& rtc_include_dir() { return s_rtc_include_dir; }
void rtc_set_include_dir(const std::string& dir) { s_rtc_include_dir = dir; }

CudaRtcKernel::CudaRtcKernel(
    const std::string& /*name*/,
    const std::string& /*kernel_code*/,
    const std::vector<std::pair<std::string, const char*>>& /*extra_includes*/)
{
    throw std::runtime_error("CudaRtcKernel: NVRTC not available in MUSA build");
}

CudaRtcKernel::~CudaRtcKernel() {}

void CudaRtcKernel::clear() {}

void CudaRtcKernel::set(MUfunction_attribute /*attr*/, int /*value*/) {
    throw std::runtime_error("CudaRtcKernel::set: NVRTC not available in MUSA build");
}

} // namespace tcnn
