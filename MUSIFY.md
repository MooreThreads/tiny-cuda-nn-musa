# MUSIFY — tiny-cuda-nn

| Field | Value |
|-------|-------|
| Project | tiny-cuda-nn |
| Source repo | https://github.com/NVlabs/tiny-cuda-nn |
| Migration branch | musa |
| MUSA SDK version | N/A |
| torch_musa version | N/A |
| MT GPU type | N/A |
| Pipeline start | N/A |
| Pipeline end | N/A |
| Elapsed | N/A |
| Last updated | 2026-03-23 06:25 |
| Director | xiaofeng |
| Status | blocked |

## Blocker Summary

| Field | Value |
|-------|-------|
| Blocker Category | No NVRTC + PTX Assembly |
| Blocker Title | PTX inline assembly (mma.sync/wmma) + NVRTC JIT compilation |
| Feasibility | Blocked on MUSA SDK |
| GitHub Stars | 4,407 |
| Domain | Other |
| Rank | #75 |

## Blocker Details

tiny-cuda-nn's core differentiator — fully fused MLPs via tensor core instructions — uses PTX inline assembly that has no MUSA equivalent. Additionally requires NVRTC for JIT-fused kernels. WARP_SIZE=32 hardcoded but MUSA S5000 actual warp size is 128. C++ API uses `at::kCUDA` device type (should be PrivateUse1 for torch_musa). Compilation succeeds in encodings-only mode but runtime fails on device allocation.

## Migration Plan

Migration is **blocked** due to: **PTX inline assembly (mma.sync/wmma) + NVRTC JIT compilation**.

**Blocker Category: No NVRTC + PTX Assembly**

### Resolution Path

This project is blocked on **MUSA SDK capabilities**. Potential resolution:

1. **MooreThreads SDK team** provides the missing SDK component
2. **Community workaround** develops an alternative approach
3. **Upstream project** adds pluggable backend support

## CUDA Dependencies

N/A — Migration not attempted due to blocker.

## Pipeline Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 0. sdk-version | skipped | Blocked before pipeline start |
| 1a. reader (code analysis) | skipped | Blocked before pipeline start |
| 1b. lookuper (document-site) | skipped | Blocked before pipeline start |
| 1c. lookuper (docker setup) | skipped | Blocked before pipeline start |
| 2. api-mapping | skipped | Blocked before pipeline start |
| 3. musa-adapt | skipped | Blocked before pipeline start |

## Issues

| ID | Type | Severity | Error | Status | Root Cause |
|----|------|----------|-------|--------|------------|
| tiny-cud-001 | blocker | blocking | PTX inline assembly (mma.sync/wmma) + NVRTC JIT compilation | wontfix | No NVRTC + PTX Assembly |

## Knowledge Base

### [Assessment] blocker-analysis — 2026-03-23 06:25
- **Status**: blocked
- **What**: Evaluated project for CUDA-to-MUSA migration feasibility
- **Result**: Project is blocked — cannot proceed with auto-musify pipeline
- **Root cause**: PTX inline assembly (mma.sync/wmma) + NVRTC JIT compilation
- **Key insight**: No NVRTC + PTX Assembly — blocked on musa sdk

## Final Report

### Summary
- Migration status: **BLOCKED**
- Blocker: PTX inline assembly (mma.sync/wmma) + NVRTC JIT compilation
- Category: No NVRTC + PTX Assembly
- Feasibility: Blocked on MUSA SDK

### Known Limitations
tiny-cuda-nn's core differentiator — fully fused MLPs via tensor core instructions — uses PTX inline assembly that has no MUSA equivalent. Additionally requires NVRTC for JIT-fused kernels. WARP_SIZE=32 hardcoded but MUSA S5000 actual warp size is 128. C++ API uses `at::kCUDA` device type (should be PrivateUse1 for torch_musa). Compilation succeeds in encodings-only mode but runtime fails on device allocation.
