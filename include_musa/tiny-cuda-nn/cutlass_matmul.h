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
	// Row-major (m x n, stride=n) viewed column-major (ld=n) == A^T, dims
	// (n x m). To compute row-major C = A @ B via column-major we issue
	//   Z_cm = B_cm @ A_cm = B^T @ A^T = (AB)^T = C_cm  (which sits in the
	// same memory as row-major C), with no mublas-op on either side —
	// the column-major "view" already contains the transpose. So for
	// row-major inputs the mublas op is *N* (not T). For column-major
	// inputs the column-major view IS the matrix itself (no implicit
	// transpose), so we need op=T to flip it into the C_cm = B^T @ A^T
	// target form. Round-2 had this flipped, causing mublasHgemm to see
	// inconsistent (rows_pre_op vs ld) pairs and reject with
	// MUBLAS_STATUS_INVALID_VALUE on the first non-stub call (bug
	// instant-ngp-002).
	return Layout == RM ? MUBLAS_OP_N : MUBLAS_OP_T;
}

template <typename T, MatrixLayout Layout>
static inline int _mublas_ld(const GPUMatrix<T, Layout>& M) {
	return (int)M.stride();
}

// ─────────────────────────────────────────────────────────────────────────────
// Round-3 F16 → F32 upcast helpers (wiki bugfix-library-deps Fix #2, verified
// in MNN). mublasHgemm rejects tcnn's shapes at runtime on S5000/SDK 5.1.0
// with MUBLAS_STATUS_INVALID_VALUE; the known-good path is to convert __half
// inputs to float, run mublasSgemm, then convert the FP32 result back to
// __half. Correctness first — numerically identical to mublasHgemm on the
// same inputs modulo FP16-rounding of intermediate products (and potentially
// tighter since accumulation is FP32).
//
// Strided cast kernels honor leading-dim (mublas's `ldX`) instead of densely
// packing because GPUMatrix allocations commonly use row-padded strides; a
// dense memcpy-style cast would corrupt layout.
// ─────────────────────────────────────────────────────────────────────────────

static __global__ void _tcnn_cast_f16_to_f32_strided(
	const __half* __restrict__ src, int src_ld,
	float*        __restrict__ dst, int dst_ld,
	int rows, int cols
) {
	// Each thread writes one (r, c) element.
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	if (c >= cols || r >= rows) return;
	// Column-major layout: index = r + c * ld.
	dst[r + c * dst_ld] = __half2float(src[r + c * src_ld]);
}

static __global__ void _tcnn_cast_f32_to_f16_strided(
	const float*  __restrict__ src, int src_ld,
	__half*       __restrict__ dst, int dst_ld,
	int rows, int cols
) {
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	if (c >= cols || r >= rows) return;
	dst[r + c * dst_ld] = __float2half(src[r + c * src_ld]);
}

// Allocate `bytes` of device scratch on `stream` via tcnn's arena. Returns a
// GPUMemoryArena::Allocation (RAII-owning); caller holds it for the
// GEMM + cast lifetime.
static inline GPUMemoryArena::Allocation _tcnn_mublas_scratch(musaStream_t stream, size_t bytes) {
	return allocate_workspace(stream, bytes);
}

// Launch a strided F16→F32 cast and return an owning Allocation holding the
// F32 buffer (ld = rows_after_op, column-major packed). `rows_after_op` and
// `cols_after_op` are in mublas's POST-op sense (if op==T, rows/cols of the
// transposed matrix); input stride `src_ld` is the caller's original ld. We
// interpret the source as column-major with (rows_after_op, cols_after_op)
// laid out given op — mublasHgemm takes the same opA/opB; converting doesn't
// affect those semantics as long as we preserve the memory layout. Easiest:
// cast every logical element (rows_before_op*cols_before_op) into a densely
// packed column-major buffer with ld = rows_before_op, and feed that to
// mublasSgemm with the same opA/opB the caller supplied.
static inline GPUMemoryArena::Allocation _tcnn_upcast_f16_to_f32(
	musaStream_t stream,
	const __half* src, int src_ld,
	int rows_pre_op, int cols_pre_op,
	float** out_ptr, int* out_ld
) {
	const size_t numel = (size_t)rows_pre_op * (size_t)cols_pre_op;
	auto alloc = _tcnn_mublas_scratch(stream, numel * sizeof(float));
	float* dst = reinterpret_cast<float*>(alloc.data());
	dim3 block(32, 8);
	dim3 grid(
		(cols_pre_op + block.x - 1) / block.x,
		(rows_pre_op + block.y - 1) / block.y
	);
	_tcnn_cast_f16_to_f32_strided<<<grid, block, 0, stream>>>(
		src, src_ld, dst, rows_pre_op, rows_pre_op, cols_pre_op
	);
	*out_ptr = dst;
	*out_ld  = rows_pre_op;  // densely-packed column-major
	return alloc;
}

// half-precision GEMM path (used when network_precision_t == __half). Round 3
// reroute: upcast to float, run mublasSgemm, downcast back into the caller's
// __half output buffer. Accepts opA/opB/M/N/K in the caller's (already-swapped
// for row-major-via-column-major) convention; we deduce pre-op shapes of A/B
// from opX + M/N/K and allocate dense F32 scratch buffers at those shapes.
static inline void _fc_mublas_gemm(
	musaStream_t stream,
	mublasOperation_t opA, mublasOperation_t opB,
	int M, int N, int K,
	const __half* A, int lda,
	const __half* B, int ldb,
	__half* C, int ldc,
	float alpha_f, float beta_f
) {
	auto& handle = tcnn_mublas_handle();
	MUBLAS_CHECK_THROW(mublasSetStream(handle, stream));

	// In mublas's column-major view with our row-major-swap convention, the
	// call _fc_mublas_gemm(opA, opB, M, N, K, A, B, C) is about to issue
	//   mublasXgemm(opB, opA, N, M, K, B, A, C).
	// The logical shapes fed to mublas are:
	//   B_post_op: N x K,  A_post_op: K x M,  C: N x M  (column-major)
	// The PRE-op shapes (how B and A physically sit in memory before mublas
	// applies opB/opA) are:
	//   B_pre_op: (opB==N) ? (N x K) : (K x N)
	//   A_pre_op: (opA==N) ? (K x M) : (M x K)
	// We upcast the pre-op buffers densely; mublas then applies opB/opA to
	// the dense float buffers exactly as it would to the original __half
	// buffers.
	int B_pre_rows = (opB == MUBLAS_OP_N) ? N : K;
	int B_pre_cols = (opB == MUBLAS_OP_N) ? K : N;
	int A_pre_rows = (opA == MUBLAS_OP_N) ? K : M;
	int A_pre_cols = (opA == MUBLAS_OP_N) ? M : K;

	float* A_f32 = nullptr; int A_f32_ld = 0;
	float* B_f32 = nullptr; int B_f32_ld = 0;
	auto A_alloc = _tcnn_upcast_f16_to_f32(stream, A, lda, A_pre_rows, A_pre_cols, &A_f32, &A_f32_ld);
	auto B_alloc = _tcnn_upcast_f16_to_f32(stream, B, ldb, B_pre_rows, B_pre_cols, &B_f32, &B_f32_ld);

	// Output C is N x M column-major. If beta != 0 we need C_f32 pre-seeded
	// from C_f16 (caller-passed). If beta == 0 we still upcast (it's cheap and
	// avoids a second code path) — wiki Fix #2 also notes muBLAS beta=0 may
	// not fully overwrite, so pre-seeding keeps the output well-defined.
	const int C_rows = N;
	const int C_cols = M;
	const size_t C_numel = (size_t)C_rows * (size_t)C_cols;
	auto C_alloc = _tcnn_mublas_scratch(stream, C_numel * sizeof(float));
	float* C_f32 = reinterpret_cast<float*>(C_alloc.data());
	int C_f32_ld = C_rows;

	// Pre-seed C_f32 from caller's C_f16 (honors the original ldc layout on
	// input → dense column-major on the float side). If beta == 0 the GEMM
	// will overwrite it fully but we still zero for safety — cheap vs GEMM.
	if (beta_f != 0.0f) {
		dim3 block(32, 8);
		dim3 grid(
			(C_cols + block.x - 1) / block.x,
			(C_rows + block.y - 1) / block.y
		);
		_tcnn_cast_f16_to_f32_strided<<<grid, block, 0, stream>>>(
			C, ldc, C_f32, C_f32_ld, C_rows, C_cols
		);
	} else {
		CUDA_CHECK_THROW(musaMemsetAsync(C_f32, 0, C_numel * sizeof(float), stream));
	}

	// Run the Sgemm on the dense F32 buffers with the caller's opA/opB.
	MUBLAS_CHECK_THROW(mublasSgemm(
		handle,
		opB, opA,
		N, M, K,
		&alpha_f,
		B_f32, B_f32_ld,
		A_f32, A_f32_ld,
		&beta_f,
		C_f32, C_f32_ld
	));

	// Downcast C_f32 (dense N x M column-major) back into caller's C (__half
	// with stride ldc).
	{
		dim3 block(32, 8);
		dim3 grid(
			(C_cols + block.x - 1) / block.x,
			(C_rows + block.y - 1) / block.y
		);
		_tcnn_cast_f32_to_f16_strided<<<grid, block, 0, stream>>>(
			C_f32, C_f32_ld, C, ldc, C_rows, C_cols
		);
	}

	// A_alloc / B_alloc / C_alloc destructors return their memory to the arena
	// at end-of-scope. Because tcnn's arena allocator is stream-ordered
	// (allocate_workspace(stream, n)), the freed memory won't be reclaimed
	// until all kernels launched on `stream` above have completed. No
	// explicit synchronize needed.
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
