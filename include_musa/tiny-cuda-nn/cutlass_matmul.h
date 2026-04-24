/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 * BSD 3-clause — see original cutlass_matmul.h header.
 *
 * MUSA port (auto-musify round 2):
 *   muBLAS-backed reimplementation of tcnn's cutlass_matmul.h API surface —
 *   specifically, the fc_multiply and fc_multiply_split_k function templates
 *   consumed by cutlass_mlp.cu / fully_fused_mlp.cu.
 *
 *   Why muBLAS (not mutlass):
 *     - MooreThreads mutlass (https://github.com/MooreThreads/mutlass) is
 *       built on CuTe (CUTLASS 3.x). Its public entry is
 *       cutlass::gemm::device::GemmUniversalAdapter.
 *     - tcnn's cutlass_matmul.h targets the CUTLASS 2.x API:
 *       cutlass::gemm::device::Gemm + GemmSplitKParallel. These classes
 *       do not exist in mutlass (verified via wiki + upstream README).
 *     - The verified pattern from the MUSA bugfix-library-deps wiki (Fix #1,
 *       validated in the MNN port) is to expose tcnn's narrow CUTLASS 2.x
 *       surface via a muBLAS-backed shim. This file is that shim.
 *
 *   Scope: only fc_multiply / fc_multiply_split_k are implemented. The rest
 *   of cutlass_matmul.h's machinery (epilogue templates, SmArch tags,
 *   GemmShape types) is collapsed into runtime arguments here.
 *
 *   Rule-3 parallel-folder porting: original cutlass_matmul.h (CUDA/CUTLASS)
 *   remains untouched under include/; this file under include_musa/ is the
 *   MUSA replacement.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network.h>  // activation_gpu / activation_backward_gpu

#include <mublas.h>
#include <musa_runtime.h>
#include <musa_fp16.h>

#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace tcnn {

// ─────────────────────────────────────────────────────────────────────────────
// mublas error helpers
// ─────────────────────────────────────────────────────────────────────────────

inline const char* mublas_get_status_string(mublasStatus_t s) {
	switch (s) {
		case MUBLAS_STATUS_SUCCESS:          return "MUBLAS_STATUS_SUCCESS";
		case MUBLAS_STATUS_NOT_INITIALIZED:  return "MUBLAS_STATUS_NOT_INITIALIZED";
		case MUBLAS_STATUS_ALLOC_FAILED:     return "MUBLAS_STATUS_ALLOC_FAILED";
		case MUBLAS_STATUS_INVALID_VALUE:    return "MUBLAS_STATUS_INVALID_VALUE";
		case MUBLAS_STATUS_ARCH_MISMATCH:    return "MUBLAS_STATUS_ARCH_MISMATCH";
		case MUBLAS_STATUS_MAPPING_ERROR:    return "MUBLAS_STATUS_MAPPING_ERROR";
		case MUBLAS_STATUS_EXECUTION_FAILED: return "MUBLAS_STATUS_EXECUTION_FAILED";
		case MUBLAS_STATUS_INTERNAL_ERROR:   return "MUBLAS_STATUS_INTERNAL_ERROR";
		case MUBLAS_STATUS_NOT_SUPPORTED:    return "MUBLAS_STATUS_NOT_SUPPORTED";
		default:                             return "MUBLAS_STATUS_unknown";
	}
}

#define MUBLAS_CHECK_THROW(x)                                                                                             \
	do {                                                                                                                    \
		mublasStatus_t _result = (x);                                                                                         \
		if (_result != MUBLAS_STATUS_SUCCESS)                                                                                 \
			throw std::runtime_error{std::string(FILE_LINE " " #x " failed with ") + mublas_get_status_string(_result)};     \
	} while (0)

// Process-wide mublas handle. muBLAS is documented thread-safe: one handle
// across multiple streams is fine; mublasCreate/Destroy cost is
// amortized once.
inline mublasHandle_t& tcnn_mublas_handle() {
	static mublasHandle_t handle = nullptr;
	static std::once_flag once;
	std::call_once(once, [] {
		MUBLAS_CHECK_THROW(mublasCreate(&handle));
	});
	return handle;
}

// ─────────────────────────────────────────────────────────────────────────────
// Row-major trick: muBLAS is column-major. To compute row-major C = A @ B we
// issue mublas( op_B, op_A, N, M, K, B, ldb, A, lda, C, ldc ) which
// column-major-computes (B^T @ A^T)^T = A @ B in row-major storage.
// A matrix stored row-major with logical (rows m, cols n, stride s) looks
// like a column-major (n x m, stride s) "implicit-transposed" matrix to
// mublas, so its mublas op is MUBLAS_OP_T; column-major stays MUBLAS_OP_N.
// ─────────────────────────────────────────────────────────────────────────────

template <typename T, MatrixLayout Layout>
static inline mublasOperation_t _mublas_op(const GPUMatrix<T, Layout>& /*M*/) {
	return Layout == RM ? MUBLAS_OP_T : MUBLAS_OP_N;
}

template <typename T, MatrixLayout Layout>
static inline int _mublas_ld(const GPUMatrix<T, Layout>& M) {
	return (int)M.stride();
}

// half-precision GEMM path (used when network_precision_t == __half)
static inline void _fc_mublas_gemm(
	musaStream_t stream,
	mublasOperation_t opA, mublasOperation_t opB,
	int M, int N, int K,
	const __half* A, int lda,
	const __half* B, int ldb,
	__half* C, int ldc,
	float alpha_f, float beta_f
) {
	__half alpha = __float2half(alpha_f);
	__half beta  = __float2half(beta_f);
	auto& handle = tcnn_mublas_handle();
	MUBLAS_CHECK_THROW(mublasSetStream(handle, stream));
	MUBLAS_CHECK_THROW(mublasHgemm(
		handle,
		opB, opA,
		N, M, K,
		&alpha,
		B, ldb,
		A, lda,
		&beta,
		C, ldc
	));
}

// float GEMM path
static inline void _fc_mublas_gemm(
	musaStream_t stream,
	mublasOperation_t opA, mublasOperation_t opB,
	int M, int N, int K,
	const float* A, int lda,
	const float* B, int ldb,
	float* C, int ldc,
	float alpha_f, float beta_f
) {
	auto& handle = tcnn_mublas_handle();
	MUBLAS_CHECK_THROW(mublasSetStream(handle, stream));
	MUBLAS_CHECK_THROW(mublasSgemm(
		handle,
		opB, opA,
		N, M, K,
		&alpha_f,
		B, ldb,
		A, lda,
		&beta_f,
		C, ldc
	));
}

// ─────────────────────────────────────────────────────────────────────────────
// fc_multiply — preserves exact signatures from upstream cutlass_matmul.h so
// cutlass_mlp.cu / fully_fused_mlp.cu compile unchanged.
//
//   D = activation_epilogue( A @ B + (sum_source ? C : 0) )
//
// - On CUDA/CUTLASS the epilogue is fused into the GEMM kernel.
// - On MUSA we decompose: (1) mublas GEMM; (2) activation_gpu post-pass.
//   Correctness-equivalent; some performance cost from the extra kernel.
// ─────────────────────────────────────────────────────────────────────────────

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, MatrixLayout LayoutC, typename TypeD, MatrixLayout LayoutD>
void fc_multiply(
	musaStream_t stream,
	const GPUMatrix<TypeA, LayoutA>& A,
	const GPUMatrix<TypeB, LayoutB>& B,
	const GPUMatrix<TypeC, LayoutC>& C,
	const GPUMatrix<TypeD, LayoutD>& D,
	Activation act = Activation::None,
	bool transfer = false,
	bool sum_source = false
) {
	static_assert(std::is_same<TypeA, TypeB>::value, "Type of matrix A and B must be equal");
	static_assert(std::is_same<TypeC, TypeD>::value, "Type of matrix C and D must be equal");
	static_assert(LayoutC == LayoutD, "Layout of matrix C and D must be equal");

	if (A.n() != B.m()) {
		throw std::runtime_error("Matrices A and B can not be multiplied together");
	}

	const int M = (int)A.m();
	const int K = (int)A.n();
	const int N = (int)B.n();

	if ((int)C.m() != M || (int)C.n() != N) {
		throw std::runtime_error{fmt::format("Matrix C has incorrect size {}x{} != {}x{}", C.m(), C.n(), M, N)};
	}
	if ((int)D.m() != M || (int)D.n() != N) {
		throw std::runtime_error{fmt::format("Matrix D has incorrect size {}x{} != {}x{}", D.m(), D.n(), M, N)};
	}

	mublasOperation_t opA = _mublas_op(A);
	mublasOperation_t opB = _mublas_op(B);
	int lda = _mublas_ld(A);
	int ldb = _mublas_ld(B);
	int ldd = _mublas_ld(D);

	// If sum_source, pre-seed D with C (when distinct buffers); then issue
	// GEMM with beta=1.0 to accumulate.
	const float alpha = 1.0f;
	const float beta  = sum_source ? 1.0f : 0.0f;

	if (sum_source && (const void*)C.data() != (const void*)D.data()) {
		CUDA_CHECK_THROW(musaMemcpyAsync(
			(void*)D.data(),
			(const void*)C.data(),
			C.n_elements() * sizeof(TypeC),
			musaMemcpyDeviceToDevice,
			stream
		));
	}

	_fc_mublas_gemm(
		stream,
		opA, opB,
		M, N, K,
		(const TypeA*)A.data(), lda,
		(const TypeB*)B.data(), ldb,
		(TypeD*)D.data(), ldd,
		alpha, beta
	);

	// Post-GEMM activation. In tcnn's backward path with transfer=true, the
	// activation-backward is fused into the epilogue on CUDA; on MUSA we
	// invoke activation_backward_gpu here.
	if (transfer) {
		GPUMatrixDynamic<TypeC> C_view((TypeC*)C.data(), C.m(), C.n(), LayoutC, C.stride(), nullptr, nullptr);
		GPUMatrixDynamic<TypeD> D_view((TypeD*)D.data(), D.m(), D.n(), LayoutD, D.stride(), nullptr, nullptr);
		activation_backward_gpu(stream, act, C_view, D_view);
	} else if (act != Activation::None) {
		GPUMatrixDynamic<TypeD> D_view((TypeD*)D.data(), D.m(), D.n(), LayoutD, D.stride(), nullptr, nullptr);
		activation_gpu(stream, act, D_view, D_view);
	}
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, typename TypeD>
void fc_multiply(
	musaStream_t stream,
	const GPUMatrix<TypeA, LayoutA>& A,
	const GPUMatrix<TypeB, LayoutB>& B,
	const GPUMatrixDynamic<TypeC>& C,
	const GPUMatrixDynamic<TypeD>& D,
	Activation act = Activation::None,
	bool transfer = false,
	bool sum_source = false
) {
	if (C.layout() != D.layout()) {
		throw std::runtime_error{"fc_multiply: Layout of GPUMatrixDynamic C and D must be equal"};
	}
	if (D.layout() == CM) {
		fc_multiply<config>(stream, A, B, C.cm(), D.cm(), act, transfer, sum_source);
	} else {
		fc_multiply<config>(stream, A, B, C.rm(), D.rm(), act, transfer, sum_source);
	}
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, typename TypeC, typename TypeD>
void fc_multiply(
	musaStream_t stream,
	const GPUMatrix<TypeA, LayoutA>& A,
	const GPUMatrixDynamic<TypeB>& B,
	const GPUMatrixDynamic<TypeC>& C,
	const GPUMatrixDynamic<TypeD>& D,
	Activation act = Activation::None,
	bool transfer = false,
	bool sum_source = false
) {
	if (B.layout() == CM) {
		fc_multiply<config>(stream, A, B.cm(), C, D, act, transfer, sum_source);
	} else {
		fc_multiply<config>(stream, A, B.rm(), C, D, act, transfer, sum_source);
	}
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, typename TypeD>
void fc_multiply(
	musaStream_t stream,
	const GPUMatrix<TypeA, LayoutA>& A,
	const GPUMatrixDynamic<TypeB>& B,
	const GPUMatrixDynamic<TypeD>& D,
	Activation act = Activation::None
) {
	fc_multiply<config>(stream, A, B, D, D, act);
}

// ─────────────────────────────────────────────────────────────────────────────
// fc_multiply_split_k — CUTLASS splits K across CTAs and reduces; muBLAS
// internally parallelizes K for us, so we collapse split_k_slices into a
// single mublas call, honoring the requested beta coefficient.
// ─────────────────────────────────────────────────────────────────────────────

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, MatrixLayout LayoutC, typename TypeD, MatrixLayout LayoutD>
void fc_multiply_split_k(
	musaStream_t stream,
	const GPUMatrix<TypeA, LayoutA>& A,
	const GPUMatrix<TypeB, LayoutB>& B,
	const GPUMatrix<TypeC, LayoutC>& C,
	const GPUMatrix<TypeD, LayoutD>& D,
	uint32_t /*split_k_slices*/ = 1,
	float beta = 0.0f
) {
	static_assert(std::is_same<TypeA, TypeB>::value, "Type of matrix A and B must be equal");
	static_assert(std::is_same<TypeC, TypeD>::value, "Type of matrix C and D must be equal");
	static_assert(LayoutC == LayoutD, "Layout of matrix C and D must be equal");

	if (A.n() != B.m()) {
		throw std::runtime_error("Matrices A and B can not be multiplied together");
	}

	const int M = (int)A.m();
	const int K = (int)A.n();
	const int N = (int)B.n();

	if ((int)C.m() != M || (int)C.n() != N) {
		throw std::runtime_error{fmt::format("Matrix C has incorrect size {}x{} != {}x{}", C.m(), C.n(), M, N)};
	}
	if ((int)D.m() != M || (int)D.n() != N) {
		throw std::runtime_error{fmt::format("Matrix D has incorrect size {}x{} != {}x{}", D.m(), D.n(), M, N)};
	}

	mublasOperation_t opA = _mublas_op(A);
	mublasOperation_t opB = _mublas_op(B);
	int lda = _mublas_ld(A);
	int ldb = _mublas_ld(B);
	int ldd = _mublas_ld(D);

	if (beta != 0.0f && (const void*)C.data() != (const void*)D.data()) {
		CUDA_CHECK_THROW(musaMemcpyAsync(
			(void*)D.data(),
			(const void*)C.data(),
			C.n_elements() * sizeof(TypeC),
			musaMemcpyDeviceToDevice,
			stream
		));
	}

	_fc_mublas_gemm(
		stream,
		opA, opB,
		M, N, K,
		(const TypeA*)A.data(), lda,
		(const TypeB*)B.data(), ldb,
		(TypeD*)D.data(), ldd,
		1.0f, beta
	);
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, typename TypeD>
void fc_multiply_split_k(
	musaStream_t stream,
	const GPUMatrix<TypeA, LayoutA>& A,
	const GPUMatrix<TypeB, LayoutB>& B,
	const GPUMatrixDynamic<TypeC>& C,
	const GPUMatrixDynamic<TypeD>& D,
	uint32_t split_k_slices = 1,
	float beta = 0.0f
) {
	if (C.layout() != D.layout()) {
		throw std::runtime_error{"fc_multiply_split_k: Layout of GPUMatrixDynamic C and D must be equal"};
	}
	if (D.layout() == CM) {
		fc_multiply_split_k<config>(stream, A, B, C.cm(), D.cm(), split_k_slices, beta);
	} else {
		fc_multiply_split_k<config>(stream, A, B, C.rm(), D.rm(), split_k_slices, beta);
	}
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, typename TypeC, typename TypeD>
void fc_multiply_split_k(
	musaStream_t stream,
	const GPUMatrix<TypeA, LayoutA>& A,
	const GPUMatrixDynamic<TypeB>& B,
	const GPUMatrixDynamic<TypeC>& C,
	const GPUMatrixDynamic<TypeD>& D,
	uint32_t split_k_slices = 1,
	float beta = 0.0f
) {
	if (B.layout() == CM) {
		fc_multiply_split_k<config>(stream, A, B.cm(), C, D, split_k_slices, beta);
	} else {
		fc_multiply_split_k<config>(stream, A, B.rm(), C, D, split_k_slices, beta);
	}
}

template <typename config, typename TypeA, typename TypeB, typename TypeC, typename TypeD>
void fc_multiply_split_k(
	musaStream_t stream,
	const GPUMatrixDynamic<TypeA>& A,
	const GPUMatrixDynamic<TypeB>& B,
	const GPUMatrixDynamic<TypeC>& C,
	const GPUMatrixDynamic<TypeD>& D,
	uint32_t split_k_slices = 1,
	float beta = 0.0f
) {
	if (A.layout() == CM) {
		fc_multiply_split_k<config>(stream, A.cm(), B, C, D, split_k_slices, beta);
	} else {
		fc_multiply_split_k<config>(stream, A.rm(), B, C, D, split_k_slices, beta);
	}
}

template <typename config, typename TypeA, typename TypeB, typename TypeD>
void fc_multiply_split_k(
	musaStream_t stream,
	const GPUMatrixDynamic<TypeA>& A,
	const GPUMatrixDynamic<TypeB>& B,
	const GPUMatrixDynamic<TypeD>& D,
	uint32_t split_k_slices,
	float beta
) {
	fc_multiply_split_k<config>(stream, A, B, D, D, split_k_slices, beta);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tag types for tcnn's FullLayer / LastLayer / FullLayerK / LastLayerK
// aliases. tcnn's cutlass_mlp.h templates fc_multiply<FullLayer>(...) etc.,
// but those template params are opaque — tcnn never peeks inside them after
// this point, so empty tag types are sufficient.
// ─────────────────────────────────────────────────────────────────────────────

template <typename Shape1, typename Shape2>
struct LayerConfig { using k_thread_block = Shape1; using k_warp = Shape2; };

template <int M, int N, int K>
struct GemmShape_ {
	static constexpr int kM = M;
	static constexpr int kN = N;
	static constexpr int kK = K;
};

using _tcnn_dummy_shape = GemmShape_<128, 128, 32>;
using FullLayer   = LayerConfig<_tcnn_dummy_shape, _tcnn_dummy_shape>;
using LastLayer   = LayerConfig<_tcnn_dummy_shape, _tcnn_dummy_shape>;
using FullLayerK  = LayerConfig<_tcnn_dummy_shape, _tcnn_dummy_shape>;
using LastLayerK  = LayerConfig<_tcnn_dummy_shape, _tcnn_dummy_shape>;

} // namespace tcnn
