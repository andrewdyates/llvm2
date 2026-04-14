# Wave 8 Integration Quality Audit

**Date:** 2026-04-14
**Auditor:** W8-A1
**Scope:** Wave 6-7 code quality review, cross-module integration, P1 issue status
**Part of:** #141

---

## 1. Build & Test Status

### Workspace Build: BLOCKED

`cargo check --workspace` fails due to z4 git dependency in llvm2-verify.

**Root cause:** `crates/llvm2-verify/Cargo.toml` line 27 declares:
```toml
z4 = { git = "https://github.com/dropbox-ai-prototypes/z4", optional = true }
```
Even though z4 is `optional = true`, Cargo resolves all workspace member git URLs during dependency resolution. The z4 repo requires authentication, causing the entire workspace build to fail.

**Impact:** No `cargo check` or `cargo test` can run for ANY crate in the workspace. This blocks all verification of code correctness.

**Existing issue:** #175 (P1: z4 git dependency blocks entire workspace build)
**Status:** Still open, still blocking. TL1 in Wave 8 is assigned to fix this.

### Workaround Attempted

`cargo check --workspace --exclude llvm2-verify` was tried but also fails -- Cargo resolves all workspace member dependencies before excluding any crate.

---

## 2. Code Quality Review: Wave 6-7 Files

### 2.1 gpu_semantics.rs (llvm2-verify)

**Path:** `crates/llvm2-verify/src/gpu_semantics.rs` (~1176 lines)
**Purpose:** Metal GPU parallel operation SMT semantic encoding

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Documentation | GOOD | Module-level docs, per-function docs, Metal spec references |
| Error handling | FAIR | `apply_fp()` panics on bitwise ops instead of returning Result |
| Test coverage | GOOD | 30+ tests: all operations, commutativity, associativity, identity elements, end-to-end |
| Dead code | CLEAN | No unused items detected |
| API design | GOOD | Clean enum-based dispatch, consistent parameter conventions |

**Finding:** `apply_fp()` uses `panic!("Bitwise ops not supported for FP")` instead of returning `Result<SmtExpr, SmtError>`. This is inconsistent with the crate's error handling convention and will crash at runtime on unexpected input rather than propagating an error.

**Severity:** P3 (not on critical path, but violates error handling convention)

### 2.2 ane_semantics.rs (llvm2-verify)

**Path:** `crates/llvm2-verify/src/ane_semantics.rs` (~1373 lines)
**Purpose:** Apple Neural Engine semantic encoding (GEMM, Conv2D, activations, FP16 quantization)

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Documentation | GOOD | References to Higham's numerical algorithms textbook for error bounds |
| Error handling | GOOD | Uses Result types appropriately |
| Test coverage | GOOD | 40+ tests: shapes, conv params, GEMM, activations, elementwise, bounded error |
| Dead code | CLEAN | All public items are used in tests or by other modules |
| API design | GOOD | Clear separation of exact ops (ReLU) vs approximated ops (sigmoid/tanh via UF) |

**Finding:** `encode_fp16_quantize_elem()` uses an uninterpreted function (UF) rather than concrete FP16 rounding semantics. This means verification cannot detect FP16-specific rounding errors -- it only proves the abstract property that "some rounding function was applied." The concrete `simulate_fp16_round()` in `ane_precision_proofs.rs` shows the team knows the concrete semantics; this should be wired into the SMT encoding.

**Severity:** P2 (verification gap -- proofs are weaker than they could be)

### 2.3 ane_precision_proofs.rs (llvm2-verify)

**Path:** `crates/llvm2-verify/src/ane_precision_proofs.rs` (~961 lines)
**Purpose:** FP16 bounded error proof obligations for ANE operations

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Documentation | GOOD | Constants well-cited (FP16_UNIT_ROUNDOFF = 2^-11, etc.) |
| Error handling | GOOD | Uses ProofObligation/VerificationResult pattern consistently |
| Test coverage | GOOD | Tests verify all 11+ proofs pass via verify_by_evaluation |
| Dead code | CLEAN | `all_ane_precision_proofs()` collects all proofs for batch verification |
| Correctness | GOOD | Error bounds follow Higham's framework correctly |

**No significant findings.** This is well-written verification code.

### 2.4 unified_synthesis.rs (llvm2-verify)

**Path:** `crates/llvm2-verify/src/unified_synthesis.rs` (~2271 lines)
**Purpose:** Cross-target CEGIS synthesis engine (scalar, NEON, GPU, ANE)

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Documentation | GOOD | Module docs, per-struct docs, clear architecture description |
| Error handling | GOOD | Uses Result types, CegisResult enum |
| Test coverage | GOOD | 50+ tests: candidate generation, CEGIS verification, cross-target, GPU/ANE estimation |
| Dead code | CLEAN | All types and functions exercised in tests |
| API design | FAIR | Two-layer architecture is sensible but GPU/ANE integration is incomplete |

**Critical Finding: GPU/ANE candidates are NOT SMT-verified.**

The `UnifiedSynthesisEngine` generates GPU and ANE candidates via `generate_gpu_candidates()` and `generate_ane_candidates()`, but these candidates are only cost-estimated -- they are never verified via SMT. The existing `gpu_semantics` and `ane_semantics` modules provide the semantic encoding functions (`encode_parallel_map`, `encode_ane_gemm`, etc.) but `unified_synthesis.rs` does not call them. Comments reference these modules but no actual verification integration exists.

This means:
- Scalar and NEON candidates: verified via CEGIS (correct)
- GPU candidates: cost-estimated only, no semantic verification
- ANE candidates: cost-estimated only, no semantic verification

**Severity:** P1 (violates the project's core principle: "verify before merge")

### 2.5 cost_model.rs (llvm2-ir)

**Path:** `crates/llvm2-ir/src/cost_model.rs` (~2660 lines)
**Purpose:** Multi-target cost model with ProfitabilityAnalyzer

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Documentation | GOOD | Per-opcode costs cited from Apple Silicon docs and benchmarks |
| Error handling | GOOD | Clean threshold-based decisions, no panics |
| Test coverage | EXCELLENT | 100+ tests: per-opcode costs, cross-target comparisons, profitability thresholds |
| Dead code | CLEAN | All types used in tests or by other modules |
| API design | GOOD | Clean separation: AppleSiliconCostModel (scalar) + MultiTargetCostModel + ProfitabilityAnalyzer |

**No significant findings.** This is the most thoroughly tested module in the review set.

---

## 3. Cross-Module Integration

### 3.1 UnifiedSynthesisEngine <-> gpu_semantics / ane_semantics

**Status: NOT INTEGRATED**

As noted in Section 2.4, `unified_synthesis.rs` does not actually import or call semantic encoding functions from `gpu_semantics` or `ane_semantics`. GPU/ANE candidates are generated with heuristic cost estimation only.

**What exists:**
- `gpu_semantics.rs`: Full SMT encoding for Metal GPU ops (parallel_map, parallel_reduce, map_reduce, barriers, SIMD group ops, UMA transfer)
- `ane_semantics.rs`: Full SMT encoding for ANE ops (GEMM, Conv2D, activations, elementwise, FP16 quantization)
- `unified_synthesis.rs`: Cost estimation stubs for GPU/ANE candidates

**What's missing:**
- CEGIS verification loop for GPU candidates using `gpu_semantics` encodings
- CEGIS verification loop for ANE candidates using `ane_semantics` encodings
- Integration of `ane_precision_proofs` bounded-error checks into the ANE synthesis path

**Severity:** P1 -- these modules were built in Waves 6-7 specifically to enable verified GPU/ANE synthesis, but the integration has not been completed.

### 3.2 ProfitabilityAnalyzer <-> compute_graph

**Status: NOT INTEGRATED**

`crates/llvm2-lower/src/compute_graph.rs` has a `target_recommendations()` method and `TargetRecommendation` struct that make target dispatch decisions. However, it does NOT import or use `ProfitabilityAnalyzer` from `crates/llvm2-ir/src/cost_model.rs`.

**What exists:**
- `cost_model.rs`: `ProfitabilityAnalyzer` with `GpuThresholds` (elementwise_min=4096, gemm_min_flops=32768) and `AneThresholds` (gemm_min_flops=131072, elementwise_min=65536)
- `compute_graph.rs`: Own target recommendation logic using `ComputeTarget` enum

**What's missing:**
- `compute_graph.rs` should use `ProfitabilityAnalyzer` for its target recommendations instead of duplicating threshold logic
- The thresholds in the two modules may diverge over time without a single source of truth

**Severity:** P2 -- code duplication with divergence risk, but not blocking correctness

---

## 4. P1 Issue Status Review

| Issue | Title | Status | Notes |
|-------|-------|--------|-------|
| #175 | z4 git dependency blocks workspace build | OPEN | Still blocking. W8-TL1 assigned. Critical path. |
| #146 | z4-bindings API audit | OPEN (needs-review) | Design work done, awaiting review |
| #142 | Memory model: array-based Load/Store | OPEN | Depends on #122 (QF_ABV array theory) |
| #125 | tMIR proof annotations | OPEN (blocked) | Blocked on tMIR repo integration |
| #122 | QF_ABV array theory | OPEN (in-progress) | Active work in Wave 8 |
| #121 | Unified solver architecture | OPEN (epic) | Tracking issue, multiple sub-tasks |
| #73 | Type duplication across crates | OPEN | W8-TL2 assigned to fix this |
| #24 | AArch64 Backend Implementation | OPEN (epic) | Long-running epic, significant progress |

**Assessment:** None of the 8 open P1 issues are resolved. #175 is the most critical as it blocks all build verification. #73 (type duplication) is actively being worked in Wave 8.

---

## 5. New Findings

### Finding 1: GPU/ANE synthesis lacks verification (P1)

GPU and ANE candidate instructions generated by UnifiedSynthesisEngine are not verified via SMT. The semantic encoding modules exist but are not wired into the synthesis loop. This violates the project's core principle of verified lowering.

**Recommendation:** File issue to integrate gpu_semantics and ane_semantics into UnifiedSynthesisEngine's CEGIS loop for GPU/ANE candidates.

### Finding 2: compute_graph does not use ProfitabilityAnalyzer (P2)

Target dispatch recommendations in compute_graph duplicate logic from ProfitabilityAnalyzer. This creates divergence risk.

**Recommendation:** Refactor compute_graph to delegate profitability decisions to ProfitabilityAnalyzer.

### Finding 3: FP16 quantization uses UF instead of concrete semantics (P2)

`ane_semantics.rs` uses an uninterpreted function for FP16 quantization instead of the concrete rounding semantics already implemented in `ane_precision_proofs.rs`. This weakens the verification guarantee.

**Recommendation:** Wire concrete FP16 rounding into the SMT encoding.

### Finding 4: gpu_semantics apply_fp panics on bitwise ops (P3)

`apply_fp()` in `gpu_semantics.rs` panics instead of returning Result, violating the crate's error handling convention.

**Recommendation:** Convert to Result return type.

---

## 6. Summary

| Area | Status |
|------|--------|
| Workspace build | BLOCKED (#175) |
| Code quality (5 files) | Generally GOOD, 1 P1 + 2 P2 + 1 P3 findings |
| Cross-module integration | 2 significant gaps (GPU/ANE verify, cost model) |
| P1 issues | 0 of 8 resolved |
| New issues filed | See below |

**Issues filed from this audit:**
- #176 (P1): GPU/ANE synthesis candidates lack SMT verification
- #177 (P2): compute_graph should use ProfitabilityAnalyzer for target dispatch

**Overall assessment:** Wave 6-7 produced high-quality individual modules with good documentation and test coverage. The critical gap is integration -- the semantic encoding modules for GPU and ANE exist but are not connected to the synthesis engine that should use them for verification. This is the highest-priority integration task after the z4 build blocker is resolved.
