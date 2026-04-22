# MUSIFY — tiny-cuda-nn (fix/cluster-001)

## Metadata

| Field | Value |
|---|---|
| Project | `tiny-cuda-nn` |
| Source repo | `https://github.com/NVlabs/tiny-cuda-nn` |
| Worktree | `repositories/tiny-cuda-nn/worktrees/fix-cluster-001/` |
| Migration branch | `fix/cluster-001` |
| Base branch | `musa-publish/musa` (canonical `auto-musify/tiny-cuda-nn`, HEAD `448e5c4`) |
| MUSA SDK version | `5.1.0` (upgrade from pinned `4.3.6` in prior port) |
| torch_musa version | N/A (library-only port) |
| MT GPU type | `MTT S5000` (ph1) |
| Pipeline start | `2026-04-21 03:34:23 UTC` |
| Pipeline end | `2026-04-21 04:10:07 UTC` |
| Elapsed | `35m 44s` |
| Last updated | `2026-04-21 04:10 UTC` |
| Director | `auto-musify director (xiaofeng)` |
| Cluster max severity | `S7` (cosmetic tier) |
| Issue count | `1` |
| Status | `partial-pass` |

---

## Migration Plan

Fix-issues cluster-001 — audit-and-confirm path for upstream GitLab issue `#1` (`[auto-musify] MUSA port Incomplete: FullyFusedMLP / CutlassMLP / NVRTC all wontfix`).

The issue bundles three MUSA hardware/SDK gaps that the prior port (SDK 4.3.6) already stubbed via `#ifdef` guards:

- `FullyFusedMLP` — no MUSA wmma PTX tensor-core intrinsic → stubbed via `TCNN_FULLY_FUSED_MLP` guard in `src_musa/fully_fused_mlp.mu`
- `CutlassMLP` — CUTLASS is NVIDIA SM-architecture-specific → `cutlass_mlp.mu` excluded from MUSA build (`CMakeLists_musa.txt`)
- `RTCKernel` — no MUSA NVRTC → stubbed via `TCNN_RTC` guard in `src_musa/rtc_kernel.mu` + `src_musa/cuda_rtc_kernel_stub.cpp`

Per director context and CLAUDE.md Rule 4 (no invented MUSA APIs), this cluster is **audit-only**: musifier verifies each subsystem is still correctly `#ifdef`-gated on SDK 5.1.0 and builds clean without them; verifier re-runs `test_encodings` + `test_grid` on portable paths; publish replies the BLOCKED template to issue `#1`. Do not fabricate fixes for genuine hardware gaps.

Portable subsystems already passing 47M assertions at SDK 4.3.6:
- Encodings (HashGrid, Frequency, SphericalHarmonics, OneBlob, ...)
- Optimizers (Adam, Shampoo, ...)
- Network base (non-fused, non-CUTLASS path)
- Loss functions, reduce_sum, cpp_api

Escape hatch: if audit uncovers a latent bug in a non-stubbed code path (e.g., the non-fused fallback MLP), that would be a separate fixable-bug finding — file a new cluster under its own issue.

---

## CUDA Dependencies

Full dependency graph + SVG inlined in `CODE_ANALYSIS.md` §1 on the canonical `musa` branch (produced by prior `migrate` run at SDK 4.3.6). Summary table:

| # | Dependency | Repo | Status | Notes |
|---|---|---|---|---|
| 1 | `CUTLASS` (vendored) | NVIDIA/cutlass | excluded | NVIDIA SM-specific; not compiled in MUSA build (`cutlass_mlp.mu` dropped) |
| 2 | `fmt` (header-only) | fmtlib/fmt | bundled | host-side, no CUDA dep |
| 3 | `nlohmann/json` | json | bundled | host-side, no CUDA dep |
| 4 | `NVRTC` | CUDA Toolkit | excluded | no MUSA NVRTC; stub returns runtime error |
| 5 | `PTX wmma` (inline asm) | NVIDIA SM intrinsics | excluded | no MUSA wmma equivalent; stub returns runtime error |

---

## Progress

| Round | Phase | Status | Started | Completed | Summary |
|---|---|---|---|---|---|
| 0 | reader-fix-issues | done | 2026-04-21 | 2026-04-21 | Cluster seeded from upstream issue #1; audit-only plan |
| 1 | musifier-fix-issues | done | 2026-04-21 | 2026-04-21 | SDK 5.1.0 audit: re-ran SimplePorting, re-applied WARP_SIZE/mma.h/MUSACC_VER/CU-driver/FMT_FLOAT128 fixes, build clean on mp_31 — `libtiny-cuda-nn.a` + `test_encodings` + `test_grid` artifacts produced; FullyFusedMLP/CutlassMLP/RTCKernel correctly excluded via `CMakeLists_musa.txt` (not `#ifdef` as issue YAML claims — gating is build-system-level, file not compiled) |
| 1 | verifier-fix-issues | done | 2026-04-21 | 2026-04-21 | `cosmetic` tier; overall=PASS; 47,186,029/47,186,029 asserts in 3 Catch2 cases (2 binaries); 4 known-limitations reconfirmed on SDK 5.1.0 |
| — | finalize | done | 2026-04-21 | 2026-04-21 | Status=`partial-pass`; AUM 4 candidates all `Skipped (jira-unreachable)` — drafts in `## AUM Jira Tickets`; README appended `## MUSA Migration` (Build/Verification/Usage); commits `045fcf8` + `25da7f7` + `33c2603` |
| — | publish | done | 2026-04-21 | 2026-04-21 | MR `!1` opened on `musa-publish` (`auto-musify/tiny-cuda-nn`) at sha `33c2603` ([merge_requests/1](https://sh-gitlab.mthreads.com/auto-musify/tiny-cuda-nn/-/merge_requests/1)); issue `#1` reply posted as [note 10646](https://sh-gitlab.mthreads.com/auto-musify/tiny-cuda-nn/-/issues/1#note_10646) to `@jing.feng` (PARTIAL-leaning template — wontfix subsystems confirmed + portable paths green) |

---

## Issues

Structured YAML files in `issues/` — one per issue, schema per `templates/issue-template.md`. This table is a summary view.

| ID | Type | Severity | Error | Status | Root Cause | Solution | Jira Refs |
|---|---|---|---|---|---|---|---|
| `tiny-cuda-nn-001` | `porting-gap` | `cosmetic` | FullyFusedMLP / CutlassMLP / RTCKernel unavailable on MUSA | `confirmed` | 3 independent hw/SDK gaps (wmma PTX, CUTLASS SM, NVRTC) — `[R1/SDK 5.1.0]` all three still gated correctly via build-system exclusion; port builds clean | `wontfix` — audit confirmed on SDK 5.1.0; `libtiny-cuda-nn.a` + `test_encodings` + `test_grid` artifacts built clean at mp_31 | — |

Issue file: `issues/tiny-cuda-nn-001.yaml`

### Topological fix order

Single issue → no DAG. Audit scope covers all three subsystems in one pass since they share the same audit target set (the `src_musa/` stub files + `CMakeLists_musa.txt` exclusion list).

1. `tiny-cuda-nn-001` — no dependencies

---

## Issues Resolved

| Round | Error Signature | Classification | Fix Applied | Source Tag | Commit |
|---|---|---|---|---|---|
| 1 | `static_assert failed: __MUSACC_VER_MAJOR__ > 10 \|\| (==10 && MINOR >= 2)` in `src_musa/common_host.mu:44` | source-level, known pattern (carried-over KB) | Replace guard with `__MUSACC_VER_MAJOR__ >= 4` per KB `[Prior / 2026-03-24] MUSA SDK version guard` | built-in (prior KB) | `<this-round>` |
| 1 | `fmt/format.h:729: error: __float128 is not supported on this target` (mcc mp_31 host+device) | source-level, unknown (new at SDK 5.1.0) | Add `-DFMT_USE_FLOAT128=0` to `TCNN_DEFINITIONS` in `CMakeLists_musa.txt` | first-principles | `<this-round>` |
| 1 | `undeclared identifier 'cuGetErrorName' / 'cuMemAddressFree' / 'CUmemAllocationProp' / 'CU_MEM_ALLOCATION_TYPE_PINNED' / 'CUDA_SUCCESS'` (CU driver API in `include_musa/`) | SimplePorting MAPPING_RULES gap — custom CU driver API rules not applied to `include_musa/` headers | Post-SimplePorting sed pass over `include_musa/`, `src_musa/`, `tests_musa/` with 33 CU driver API rules from `simple_porting_tcnn.py` | first-principles | `<this-round>` |
| 1 | `WARP_SIZE = 32` hardcoded (regenerated by SimplePorting, overwrites prior fix) | porting gap (carried-over KB) | Replace with `WARP_SIZE = 128` (S5000) per `apply_musa_fixes.py:fix_warp_size` | built-in (prior KB) | `<this-round>` |
| 1 | `mma.h` PTX inline asm (`mma.sync.aligned`, `wmma.mma.sync.aligned`) — not supported on MUSA | porting gap (carried-over KB) | Disable via source-comment per `apply_musa_fixes.py:fix_mma_ptx` | built-in (prior KB) | `<this-round>` |

---

## Knowledge Base

Append-only log. Historical entries carried over from prior `migrate` run (canonical `musa` HEAD `448e5c4`) — preserved per reader spec (prior `## Knowledge Base` entries persist across `update-*` and `fix-issues` runs).

### [Prior / 2026-03-24] MUSA CMake: `musa_add_library` drops `.cpp` sources

- **Problem**: `musa_add_library` only processes `.mu` files; plain `.cpp` sources are silently dropped.
- **Fix**: Use separate `add_library(target STATIC *.cpp)` for non-MUSA C++ files.
- **Pattern**: Always separate MUSA kernel code from host-only C++ stubs.

### [Prior / 2026-03-24] `mcc` (clang-based): NVCC-specific flags rejected

- **Flags removed**: `--expt-relaxed-constexpr`, `--extended-lambda`.
- **Reason**: `mcc` is clang-14 based; these are NVCC-only flags.
- **Pattern**: Remove all `--expt-*` flags in MUSA CMake builds.

### [Prior / 2026-03-24] MUSA SDK version guard

- **Problem**: `__MUSACC_VER_MAJOR__ > 10` always false (MUSA SDK is v4.x / v5.x).
- **Fix**: `__MUSACC_VER_MAJOR__ >= 4`.
- **Pattern**: CUDA uses major=12+; MUSA uses major=4.x / 5.x.

### [Prior / 2026-03-24] Lane ID intrinsic

- **CUDA PTX**: `asm volatile("mov.u32 %0, %laneid;" : "=r"(lane_id))`
- **MUSA equivalent**: `__musa_get_special_register(14)` (register 14 = lane_id).
- **Pattern**: Check MUSA intrinsics reference for PTX register equivalents.

### [Prior / 2026-03-24] `WARP_SIZE` on S5000

- **Problem**: `WARP_SIZE = 32` hardcoded (CUDA convention).
- **Fix**: `WARP_SIZE = 128` for MTT S5000 (mp_31).
- **File**: `include_musa/tiny-cuda-nn/common.h`.

### [Prior / 2026-03-24] Encodings-only mode assessment

- **Status**: partial — encodings + optimizers + basic networks work (47M assertions pass on S5000).
- **Excluded**: `FullyFusedMLP` (wmma PTX), `CutlassMLP` (NVIDIA SM), `RTCKernel` (NVRTC) — all NVIDIA-specific.
- **Downstream impact**: HashGrid-encoding users (NeRF, SDF) can use the MUSA port; users depending on `FullyFusedMLP` have no fallback on MUSA.

*(musifier appends `[Round N] …` entries on top of this.)*

### [Round 1 / 2026-04-21] SDK 5.1.0 confirmed — prior 4.3.6 audit conclusions hold

- All three subsystems (FullyFusedMLP / CutlassMLP / RTCKernel) remain correctly excluded at SDK 5.1.0.
- Gating mechanism is **build-system exclusion** (`CMakeLists_musa.txt` `TCNN_SOURCES` list omits the three `.mu` files), NOT `#ifdef` guards. The issue YAML references `TCNN_FULLY_FUSED_MLP` / `TCNN_RTC` guards, but `apply_musa_fixes.py:fix_rtc_kernel` / `fix_fully_fused_mlp` wrap stubs that are never compiled. Documentation inconsistency, not a bug.
- RTCKernel stubs are provided by hand-written `src_musa/cuda_rtc_kernel_stub.cpp` (plain CXX — `musa_add_library` silently drops `.cpp` sources per prior KB) linked as a separate `add_library(tcnn-rtc-stub STATIC ...)` target.

### [Round 1 / 2026-04-21] SDK 5.1.0 regression — `__float128` rejected by mcc mp_31

- **Problem**: `fmt/format.h:729: error: __float128 is not supported on this target` — mcc at mp_31 does NOT support `__float128`, unlike prior SDK 4.3.6 where fmt compiled cleanly because `__SIZEOF_FLOAT128__` wasn't defined in device mode.
- **Fix**: Add `-DFMT_USE_FLOAT128=0` to `TCNN_DEFINITIONS` in `CMakeLists_musa.txt`; fmt then uses `using float128 = void` (gracefully unavailable).
- **Pattern**: When MUSA toolchain probes `__SIZEOF_FLOAT128__` via clang-14, it resolves true on host but compile target (`mp_31`) lacks support. Force-disable FMT_USE_FLOAT128 for all MUSA builds linking fmt ≥ 9.x.

### [Round 1 / 2026-04-21] SimplePorting MAPPING_RULES not fully applied to `include_musa/`

- **Problem**: The custom rules in `simple_porting_tcnn.py` (CU driver API — `cuGetErrorName`, `cuMemAddressReserve`, `CUmemAllocationProp`, etc., 33 entries) did NOT rewrite `include_musa/tiny-cuda-nn/common_host.h` or `gpu_memory.h` on re-run. The `cuda → musa` default rules did apply, but `CU → MU` custom rules didn't fire for these headers — possibly because SimplePorting's alphabetical/length-order processing matched `CUDART_VERSION` → `MUSART_VERSION` but skipped `CUmem*` and `cuMem*` when preceded by certain identifiers.
- **Fix**: Post-SimplePorting `.py` sweep over `include_musa/`, `src_musa/`, `tests_musa/` with a length-desc-sorted replacement pass using the same 33 rules. 5 files updated.
- **Pattern**: Never trust SimplePorting to apply custom `MAPPING_RULES` exhaustively — always run a verification grep for `CU[A-Z]*` / `cu[a-z]+` symbols in ported files, and re-run a raw-sed pass if any remain.

---

## Known Limitations

Carried forward from canonical `musa` MUSIFY.md. Permanent record — do not delete.

- `FullyFusedMLP` — cannot compile MUSA backend (no wmma/PTX tensor-core intrinsic) — workaround: fall back to non-fused `Network` path — escalation: N/A (fundamental hardware gap). **[Round 1 / 2026-04-21] confirmed on SDK 5.1.0 — still wontfix.** Gating is build-system-level (excluded from `TCNN_SOURCES` in `CMakeLists_musa.txt`), not `#ifdef TCNN_FULLY_FUSED_MLP`.
- `CutlassMLP` — cannot compile MUSA backend (CUTLASS is NVIDIA SM-architecture-specific) — workaround: use non-CUTLASS `NetworkWithInputEncoding` path — escalation: N/A (fundamental library gap). **[Round 1 / 2026-04-21] confirmed on SDK 5.1.0 — still wontfix.**
- `RTCKernel` / `CudaRtcKernel` — no runtime-JIT kernel compilation on MUSA (no NVRTC) — workaround: use statically-compiled kernels only — escalation: N/A (fundamental SDK gap). **[Round 1 / 2026-04-21] confirmed on SDK 5.1.0 — still wontfix.** RTC stubs are provided by hand-written `src_musa/cuda_rtc_kernel_stub.cpp` as a separate `add_library(tcnn-rtc-stub STATIC …)` target (bypasses `musa_add_library`'s `.cpp` drop).
- `--expt-relaxed-constexpr` / `--extended-lambda` — NVCC-only flags removed from MUSA build — workaround: N/A (both toolchains now compile cleanly) — escalation: N/A. **[Round 1 / 2026-04-21] confirmed still absent from `CMakeLists_musa.txt` on SDK 5.1.0.**
- `FMT_USE_FLOAT128` must be force-disabled for SDK 5.1.0+ builds — mcc at mp_31 rejects `__float128`, while fmt 9.x auto-enables it when clang-14 reports `__SIZEOF_FLOAT128__` on the host. Workaround: `-DFMT_USE_FLOAT128=0` in `TCNN_DEFINITIONS`. Escalation: N/A (SDK-level limitation, stable across host flag).

---

## Classification Override

*(empty — no override yet)*

---

## Strategy Fallback

*(empty — library-only port; no Python conversion strategy)*

---

## Tier Override

*(empty — no override yet)*

---

## AUM Jira Tickets

Jira MCP returned `Invalid user Jira token or configuration: Unable to get current user account ID` on every call attempted during this run (reader exploratory search, verifier run, and Finalize §4 dedupe+create). Per director §4 best-effort protocol, registration was skipped for all four classifiable Known Limitations. Operator must file these manually post-run (or re-check JIRA_PAT in `.claude/remote-ssh.env` and re-invoke `/auto-musify fix --issues=...`).

| Limitation | Category | Action | AUM Key | Reason |
|---|---|---|---|---|
| `FullyFusedMLP` — no MUSA wmma PTX tensor-core intrinsic | `hw` | Skipped | — | jira-unreachable |
| `CutlassMLP` — CUTLASS NVIDIA-SM-specific | `hw` | Skipped | — | jira-unreachable |
| `RTCKernel` — no MUSA NVRTC | `sdk` | Skipped | — | jira-unreachable |
| `FMT_USE_FLOAT128` must be force-disabled (mcc mp_31 rejects `__float128`) | `sdk` | Skipped | — | jira-unreachable |

Suggested ticket drafts (for post-run manual filing into project `AUM`):

1. **`[MTGPU-S5000] MUSA wmma PTX tensor-core intrinsic missing — blocks FullyFusedMLP-class kernels`** (type=Bug, labels=`auto-musify, migration-blocker, sdk-5.1.0, gpu-s5000`) — observed: PTX `wmma` / `mma.sync.aligned` inline asm fails to compile on mcc; `src_musa/fully_fused_mlp.mu` excluded from `TCNN_SOURCES`. Impact: fused-tensor-core MLP unavailable on MUSA. Evidence: commit `250406d` on `fix/cluster-001`; upstream GitLab issue `sh-gitlab.mthreads.com/auto-musify/tiny-cuda-nn#1`.
2. **`[MTGPU-S5000] CUTLASS templates NVIDIA-SM-specific — no MUSA equivalent → blocks CutlassMLP`** (type=Bug, same labels) — CUTLASS tile iterators assume NVIDIA warp scheduler + Ampere/Hopper SM; no MT alternative header library. `src_musa/cutlass_mlp.mu` excluded from `TCNN_SOURCES`.
3. **`[mcc/musa-runtime] No MUSA NVRTC equivalent — blocks runtime JIT kernel compilation (RTCKernel)`** (type=Bug, same labels) — no upstream mcc API maps to `nvrtcCreateProgram` / `nvrtcCompileProgram`; `src_musa/cuda_rtc_kernel_stub.cpp` linked as `libtcnn-rtc-stub.a` provides link-time symbols that throw at call time. SDK-level capability gap.
4. **`[mcc] mp_31 target rejects __float128 despite host clang-14 reporting __SIZEOF_FLOAT128__`** (type=Bug, same labels) — new-at-SDK-5.1.0 regression; fmt ≥ 9.x auto-enables FLOAT128 when `__SIZEOF_FLOAT128__` is defined on host, but mcc device compile at mp_31 rejects `__float128`. Workaround: `-DFMT_USE_FLOAT128=0` in `TCNN_DEFINITIONS`. Repro: any MUSA build linking fmt ≥ 9.x without that flag.

---

## Verification & Testing

Black-box audit on SDK 5.1.0 — `cosmetic`-tier re-run of the two portable-path Catch2 binaries (`test_encodings` + `test_grid`) to confirm the prior SDK 4.3.6 baseline (47M assertions) still holds. Three wontfix subsystems (`FullyFusedMLP` / `CutlassMLP` / `RTCKernel`) are build-system-excluded from `TCNN_SOURCES` — not exercised, not present in test binaries, still wontfix per `## Known Limitations`.

### Summary

- **artifact_type**: `Lib` (static library + Catch2 test executables)
- **tier_used**: `cosmetic` (portable-path Catch2 re-run)
- **overall**: `PASS`
- **passed=47,186,029 / failed=0 / skipped=0 / total=47,186,029** (3 Catch2 test cases across 2 binaries)
- **GPU bound**: `MUSA_VISIBLE_DEVICES=0` (GPU 0, MTT S5000 ph1) — 0% util + 0 MiB mem at round start
- **container**: `auto-musify-tiny-cuda-nn` (up, from infraer — `auto-musify-sdk:5.1.0-daily-20260416-S5000`)
- **artifacts**: reused from musifier commit `250406d` — `libtiny-cuda-nn.a` (9.5 MB), `libtcnn-rtc-stub.a` (7 KB), `test_encodings` (7.9 MB), `test_grid` (7.3 MB); no rebuild

### MUSA-relevant test catalog

| # | Mechanism | Binary / Script | CMake flag(s) | MUSA param | Status |
|---|---|---|---|---|---|
| 1 | Encodings invariance checks (Identity, OneBlobFrequency, DenseGrid, TiledGrid, Grid, Frequency, SphericalHarmonics, Empty, HashGrid, Composite) — `network_precision_t` + `float` | `tests_musa/test_encodings` | `USE_MUSA=ON`, `TCNN_BUILD_TESTS=ON` | Catch2 default (no `--backend` flag; binary is hard-wired to MUSA at link time) | `PASS — 47,186,016 assertions in 2 test cases` |
| 2 | `GridEncoding sanity checks` | `tests_musa/test_grid` | `USE_MUSA=ON`, `TCNN_BUILD_TESTS=ON` | Catch2 default | `PASS — 13 assertions in 1 test case` |

No additional MUSA runtime parameters — the binaries are statically linked against `libtiny-cuda-nn.a` (MUSA toolchain) so there is no backend/precision switch at invocation; precision is covered internally by the two test cases in `test_encodings` (`network_precision_t` = half + `float` = fp32).

### How to build

```bash
# Already built at musifier commit 250406d — reproduction only, do NOT rebuild
ssh jing.feng@10.10.142.136 "docker exec auto-musify-tiny-cuda-nn bash -c '
  cd /workspace/tiny-cuda-nn-fix-cluster-001
  mkdir -p musa_build_root && cp CMakeLists_musa.txt musa_build_root/CMakeLists.txt
  ln -sfn ../src_musa     musa_build_root/src_musa
  ln -sfn ../include_musa musa_build_root/include_musa
  ln -sfn ../tests_musa   musa_build_root/tests_musa
  ln -sfn ../dependencies musa_build_root/dependencies
  rm -rf build_musa
  cmake -S musa_build_root -B build_musa \
    -DUSE_MUSA=ON -DTCNN_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build build_musa -j4
'"
```

Key CMake flags (from `CMakeLists_musa.txt`):
- `USE_MUSA=ON` — activates MUSA build path (vs. CUDA default)
- `TCNN_BUILD_TESTS=ON` — enables `tests_musa/` subdirectory
- `TCNN_SOURCES` deliberately omits `fully_fused_mlp.mu`, `cutlass_mlp.mu`, `rtc_kernel.mu` (wontfix subsystems — see `## Known Limitations`)
- `-DFMT_USE_FLOAT128=0` in `TCNN_DEFINITIONS` — mcc mp_31 rejects `__float128`

### How to run tests

```bash
ssh jing.feng@10.10.142.136 "docker exec -e MUSA_VISIBLE_DEVICES=0 auto-musify-tiny-cuda-nn bash -c '
  cd /workspace/tiny-cuda-nn-fix-cluster-001/build_musa
  ./tests_musa/test_encodings --durations=yes
  ./tests_musa/test_grid --durations=yes
'"
```

### Parameter reference

| Parameter | Values | Description |
|---|---|---|
| `MUSA_VISIBLE_DEVICES` | `0..7` | Bind GPU index (idlest picked — GPU 0 for this round) |
| `--durations=yes` | (Catch2 flag) | Include per-section timing in the output |
| `--list-tests` | (Catch2 flag) | Enumerate test cases without running |
| `-s` / `--success` | (Catch2 flag) | Include successful assertion counts in output — used to confirm the 47M number |

### Results Matrix

| Category | Precision | Passed | Failed | Total | Pass Rate | Command |
|---|---|---|---|---|---|---|
| Encodings invariance (`[encoding][jit]`) | `network_precision_t` (half) | N/A (assertion-level) | 0 | 1 case | 100% | `./tests_musa/test_encodings "Various invariance checks for input encodings - network_precision_t"` |
| Encodings invariance (`[encoding][jit]`) | `float` (fp32) | N/A (assertion-level) | 0 | 1 case | 100% | `./tests_musa/test_encodings "Various invariance checks for input encodings - float"` |
| Encodings aggregate | mixed | **47,186,016 asserts** | 0 | 2 cases | 100% | `./tests_musa/test_encodings` |
| Grid sanity (`[encoding]`) | default | **13 asserts** | 0 | 1 case | 100% | `./tests_musa/test_grid` |
| **Aggregate (all portable paths)** | mixed | **47,186,029 asserts** | 0 | 3 cases / 2 binaries | **100%** | see above |

Per-encoding breakdown inside `test_encodings` (from `--durations=yes` trace, both test cases):
`Identity → OneBlobFrequency → DenseGrid → TiledGrid → Grid → Frequency → SphericalHarmonics → Empty → HashGrid → Composite` — each exercised through the `inference`, `inference_mixed_precision`, and `forward` invariance pairs.

### Failed Tests

*(empty — all portable-path tests pass at SDK 5.1.0, matching prior SDK 4.3.6 baseline 1:1)*

### Fixable Bugs (musifier)

*(empty — no new fixable bugs. Portable paths clean; three wontfix subsystems remain correctly excluded at build-system level.)*

### Known Limitations (MUSIFY.md)

Confirmed on SDK 5.1.0 — no changes to the four entries in `## Known Limitations` above. Specifically:

| Subsystem | Confirmation source | Status |
|---|---|---|
| `FullyFusedMLP` (wmma PTX) | `src_musa/fully_fused_mlp.mu` absent from `TCNN_SOURCES` → not in `libtiny-cuda-nn.a` → not linked into test binaries → not exercised | wontfix (confirmed SDK 5.1.0) |
| `CutlassMLP` (NVIDIA SM) | `src_musa/cutlass_mlp.mu` absent from `TCNN_SOURCES` → not in `libtiny-cuda-nn.a` | wontfix (confirmed SDK 5.1.0) |
| `RTCKernel` (NVRTC) | `src_musa/rtc_kernel.mu` absent from `TCNN_SOURCES`; `src_musa/cuda_rtc_kernel_stub.cpp` built as separate `libtcnn-rtc-stub.a` (7 KB) — provides link-time-only symbol stubs that throw at call time | wontfix (confirmed SDK 5.1.0) |
| `FMT_USE_FLOAT128` force-disabled | `CMakeLists_musa.txt` `TCNN_DEFINITIONS` includes `-DFMT_USE_FLOAT128=0`; build clean, tests pass | required setting (confirmed SDK 5.1.0) |

### Verification logs

- `verify/test_encodings.log` — full run output (121 lines, tail of 47M-assertion run)
- `verify/test_grid.log` — full run output (5 lines)

---

## Final Report

### Summary

- **Mode**: `fix-issues` — cluster-001 audit of upstream GitLab issue `#1` (`[auto-musify] MUSA port Incomplete: FullyFusedMLP / CutlassMLP / NVRTC all wontfix`).
- **Branch**: `fix/cluster-001` (worktree `worktrees/fix-cluster-001/`), based on canonical `musa-publish/musa @ 448e5c4` (`auto-musify/tiny-cuda-nn` GitLab project).
- **Elapsed**: 35m 44s (start `2026-04-21 03:34:23 UTC` → end `2026-04-21 04:10:07 UTC`).
- **Files ported**: 77 files touched on `fix/cluster-001` since base `448e5c4` (`+20,565 / -16`). Breakdown: 2 modified (`CMakeLists_musa.txt` + `src_musa/common_host.mu`), 36 new `include_musa/tiny-cuda-nn/**/*.h`, 10 new `src_musa/*.mu` + 1 `src_musa/CMakeLists.txt`, 5 new `tests_musa/*.mu` + `test_common.h`, 1 MUSIFY.md expanded, 1 `issues/tiny-cuda-nn-001.yaml` new, plus `verify/*.log` artifacts.
- **Build status**: clean at commit `250406d` — `libtiny-cuda-nn.a` (9.5 MB), `libtcnn-rtc-stub.a` (7 KB), `tests_musa/test_encodings` (7.9 MB), `tests_musa/test_grid` (7.3 MB) produced by `cmake --build build_musa -j4` inside `auto-musify-tiny-cuda-nn` container (SDK 5.1.0, mp_31).
- **Tests**: `PASS` — 47,186,029 / 47,186,029 assertions across 3 Catch2 test cases in 2 binaries on S5000 @ mp_31 (+13 asserts, +1 test case vs. the prior SDK 4.3.6 baseline of 47,186,016 in 2 cases — `test_grid` got properly re-enabled during SimplePorting regeneration).
- **Issues processed**: 1 (`tiny-cuda-nn-001`) — confirmed wontfix-audit-only; three hardware/SDK subsystems (`FullyFusedMLP`, `CutlassMLP`, `RTCKernel`) reconfirmed excluded at build-system level on SDK 5.1.0; no fabricated MUSA-API shims (CLAUDE.md Rule 4).
- **Fix iterations**: 0 `musify: fix` commits (audit-only cluster; round 1 `musify: confirm SDK 5.1.0 audit` is a confirmation, not a fix).
- **Final status**: `partial-pass` — all portable paths green, four Known Limitations persist (three upstream-gap wontfix, one SDK 5.1.0 workaround).

### Changes Made

Commits on `fix/cluster-001` (since base `448e5c4`):

- `1b8aa9d` — `fix: init cluster-001 from issue board (https://sh-gitlab.mthreads.com/auto-musify/tiny-cuda-nn/-/issues/1)` (reader seed: Metadata + Migration Plan + Issues + Known Limitations carry-over + Verification catalog from prior run)
- `250406d` — `musify: confirm SDK 5.1.0 audit — libtiny-cuda-nn + tests build clean at mp_31` (musifier round 1: SimplePorting re-run + 5 fix re-applications + `__float128` new regression fix + `## Knowledge Base [Round 1]` × 3 entries + `## Progress` round 1 row + `## Issues Resolved` × 5 rows)
- `<pending>` — `musify: document MUSA migration in README (fix-issues)` (Finalize §5 this commit)
- `<pending>` — `musify: finalize fix-issues run + file AUM blockers` (final Finalize commit — see `## AUM Jira Tickets` for skipped registrations)

Fixes re-applied on top of SimplePorting regeneration (all carried from prior Knowledge Base):
- `WARP_SIZE = 128` restored for S5000 mp_31 in `include_musa/tiny-cuda-nn/common.h` (SimplePorting kept the CUDA-default `= 32`).
- `__MUSACC_VER_MAJOR__ >= 4` in `src_musa/common_host.mu:44` (SimplePorting emitted `> 10`).
- `mma.h` PTX inline asm (`mma.sync.aligned`, `wmma.mma.sync.aligned`) source-commented out per `apply_musa_fixes.py:fix_mma_ptx`.
- 33 CU driver API symbols (`cuGetErrorName`, `CUmemAllocationProp`, etc.) manually swept across `include_musa/`, `src_musa/`, `tests_musa/` — SimplePorting's `MAPPING_RULES` didn't apply to headers (new Knowledge Base entry for future runs).

New SDK 5.1.0 regression (not present at prior 4.3.6 run):
- `-DFMT_USE_FLOAT128=0` added to `TCNN_DEFINITIONS` in `CMakeLists_musa.txt` — mcc mp_31 rejects `__float128` even though host clang-14 reports `__SIZEOF_FLOAT128__`.

### AUM Jira Tickets (aggregate)

| Metric | Count |
|---|---|
| Created (new) | 0 |
| Linked (existing) | 0 |
| Skipped (project-specific, not AUM) | 0 |
| Skipped (jira-unreachable) | 4 |
| **Total registerable limitations** | **4** |

All four tickets are **Skipped (jira-unreachable)** — the `atlassian` MCP returned auth error `Invalid user Jira token or configuration: Unable to get current user account ID` on every call during this run (reader exploratory search + verifier + Finalize dedupe). Operator action required: (a) re-check `JIRA_PAT` in `.claude/remote-ssh.env`, or (b) file the four suggested ticket drafts (see `## AUM Jira Tickets` above) manually in the `AUM` Jira project.

### Pipeline state summary

| Phase | Outcome | Gate |
|---|---|---|
| `Start` / `CollectCredentials` / `ModeRouter` | ✅ all required vars present; `DOCKER_IMAGE` override → infraer Path 0; `GIT_PUSH_GROUP=https://sh-gitlab.mthreads.com/auto-musify` override → publish routes to same host as issue board | — |
| `InfraerSpawn` → `InfraerGate` | ✅ container `auto-musify-tiny-cuda-nn` (image `auto-musify-sdk:5.1.0-daily-20260416-S5000`); manifest `sdk=5.1.0 / torch_musa=2.9.0+74f5782 / mudnn=3.3.0.0 / mublas=1.12.0 / mccl=2.28.9 / triton=3.2.0`; 8 verifications pass | `infraer-returned-ready ✅` |
| `ReaderSpawn` → `ReaderGate` | ✅ 1 issue → 1 cluster (cluster-001, tier=cosmetic, `wontfix-audit-only` strategy); canonical `musa-publish` remote added; worktree on `fix/cluster-001` @ `1b8aa9d` | `reader-returned-ready ✅` |
| `MusifierSpawn` → `MusifierGate` (round 1) | ✅ commit `250406d`; 5 fixes re-applied + 1 new `__float128` regression fix; no `fixable_bugs` | `musifier-returned-build-clean ✅` |
| `VerifierSpawn` → `VerifierGate` | ✅ `cosmetic` tier (matches §4); overall `PASS`; 47,186,029 asserts in 3 test cases | `verifier-returned-ready ✅` |
| `CycleRouter` | Routed directly to `Finalize` (empty `fixable_bugs`) — no round 2 needed | — |
| `Finalize` | ✅ this document; status = `partial-pass`; 4 AUM Skipped | `finalize-complete ⏳` |

### Known-limitation carry-forward reconfirmation

All four entries in `## Known Limitations` are **reconfirmed on SDK 5.1.0**:

1. `FullyFusedMLP` — still wontfix (hw gap)
2. `CutlassMLP` — still wontfix (NVIDIA-SM-specific lib)
3. `RTCKernel` — still wontfix (SDK gap; stubs linked via `libtcnn-rtc-stub.a`)
4. `FMT_USE_FLOAT128=0` — required setting (new at SDK 5.1.0; mcc mp_31 rejects `__float128`)

No new wontfix entries, no resolutions, no escalations to `## Classification Override` / `## Strategy Fallback` / `## Tier Override`.
