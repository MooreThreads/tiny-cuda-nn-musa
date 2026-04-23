// MUSA stub: CutlassMLP -- provides link-time stubs so ngp's
// testbed_nerf/testbed_sdf resolve even though vendored CUTLASS is not
// portable to mcc (needs cuda_fp16.h + NVIDIA SM archs).
//
// On MUSA, select_network() forces CutlassMLP for MIN_GPU_ARCH<=70
// (S5000=21 qualifies). Our stub ctor throws immediately — any
// NeRF/SDF/Image training attempt will surface the gap at runtime with a
// clear message. This matches tcnn sibling port's "wontfix" status for
// FullyFusedMLP/CutlassMLP on mcc. Build/link succeed; smoke test (--version)
// still runs; full training path is a Known Limitation.
//
// .mu extension because common_host.h (indirectly via networks/cutlass_mlp.h)
// pulls in device headers (__half etc.). mcc accepts it fine.

#include <tiny-cuda-nn/networks/cutlass_mlp.h>

#include <stdexcept>

namespace tcnn {

template <typename T>
CutlassMLP<T>::CutlassMLP(
    uint32_t /*input_width*/, uint32_t /*network_width*/,
    uint32_t /*output_width*/, uint32_t /*n_hidden_layers*/,
    Activation /*activation*/, Activation /*output_activation*/)
{
    throw std::runtime_error(
        "CutlassMLP: NVIDIA CUTLASS not portable to MUSA SDK 5.1.0 "
        "(needs cuda_fp16.h + hardcoded SM archs). "
        "Training paths that reach CutlassMLP fallback are unsupported in "
        "this build. See MUSIFY.md Known Limitations."
    );
}

// Virtual methods must exist for link-time vtable resolution.
// They're never invoked because the ctor above throws first.

template <typename T>
void CutlassMLP<T>::set_params_impl(T* /*params*/, T* /*inference*/, T* /*gradients*/) {}

template <typename T>
void CutlassMLP<T>::initialize_params(pcg32& /*rnd*/, float* /*params*/, float /*scale*/) {}

#if !defined(TCNN_NO_FWD_BWD)
template <typename T>
void CutlassMLP<T>::inference_mixed_precision_impl(
    musaStream_t /*stream*/, const GPUMatrixDynamic<T>& /*input*/,
    GPUMatrixDynamic<T>& /*output*/, bool /*use_inference_params*/) {}

template <typename T>
std::unique_ptr<Context> CutlassMLP<T>::forward_impl(
    musaStream_t /*stream*/, const GPUMatrixDynamic<T>& /*input*/,
    GPUMatrixDynamic<T>* /*output*/, bool /*use_inference_params*/,
    bool /*prepare_input_gradients*/)
{
    return nullptr;
}

template <typename T>
void CutlassMLP<T>::backward_impl(
    musaStream_t /*stream*/, const Context& /*ctx*/,
    const GPUMatrixDynamic<T>& /*input*/, const GPUMatrixDynamic<T>& /*output*/,
    const GPUMatrixDynamic<T>& /*dL_doutput*/, GPUMatrixDynamic<T>* /*dL_dinput*/,
    bool /*use_inference_params*/, GradientMode /*param_gradients_mode*/) {}
#endif

// Explicit template instantiations (match network_precision_t usages).
template class CutlassMLP<float>;
template class CutlassMLP<__half>;

} // namespace tcnn
