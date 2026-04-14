# Wave 7 Workspace Quality Audit

**Date:** 2026-04-14
**Auditor:** A1 (Wave 7)
**Scope:** Full workspace quality audit after 6 waves of rapid development

---

## 1. Full Workspace Test/Build Status

### Build Status: BLOCKED

The entire workspace is **unbuildable** due to a z4 git dependency authentication
failure. Even `cargo test -p llvm2-ir --lib` (which does not use z4) fails because
Cargo resolves all workspace member dependencies before building any single crate.

**Error:**
```
error: failed to get `z4` as a dependency of package `llvm2-verify v0.1.0`
  Authentication failed for 'https://github.com/dropbox-ai-prototypes/z4/'
```

**Root cause:** `crates/llvm2-verify/Cargo.toml` line 27 declares:
```toml
z4 = { git = "https://github.com/dropbox-ai-prototypes/z4", optional = true }
```
Even though `z4` is `optional = true` and behind a feature gate, Cargo still
resolves the git URL for workspace dependency resolution. The local cargo git
cache (`~/.cargo/git/db/z4-18f012635b05e4f2`) has no refs, meaning z4 was
never successfully fetched.

**No Cargo.lock in worktree:** The main repo has a `Cargo.lock` but this
worktree did not, causing Cargo to attempt a fresh resolution.

**Impact:** Cannot run `cargo test`, `cargo check`, or `cargo build` for ANY
crate in the workspace. All test counts, warning detection, and compilation
verification are blocked.

**Severity: P1** -- This is a workspace-wide build blocker.

**Recommended fix:** Comment out line 27 of `crates/llvm2-verify/Cargo.toml`
(the uncommented z4 dependency). There is already a commented-out version on
line 26. The z4 feature gate still works -- only the declaration needs to be
conditional on the feature, which Cargo does not support for git deps. The
workaround is to keep it commented and only enable via path override.

### Estimated Test Counts (from source analysis)

Since builds are blocked, test counts are derived from `#[test]` annotations:

| Crate | LOC (total) | Test Count | Source Files |
|-------|-------------|------------|--------------|
| llvm2-ir | 9,842 | 310 | 12 |
| llvm2-lower | 14,676 | 250 | 8 |
| llvm2-opt | 8,970 | 171 | 15 |
| llvm2-regalloc | 11,368 | 265 | 13 |
| llvm2-verify | 24,370 | 763 | 23 |
| llvm2-codegen | 26,956 | 758 | 27 |
| **Total** | **96,182** | **2,517** | **98** |

---

## 2. Cross-Crate API Consistency

### llvm2-verify/src/lib.rs -- Module exports

All new modules are properly declared as `pub mod`:

- `gpu_semantics` -- line 69
- `ane_semantics` -- line 68
- `unified_synthesis` -- line 70
- `neon_semantics` -- line 67
- `neon_lowering_proofs` -- line 71

**Missing module:** `ane_precision_proofs` does NOT exist as a file or module.
The audit task asked about it, but it was never implemented. This is expected
per the design docs -- ANE precision proofs are in the design phase
(`designs/2026-04-14-ane-precision-verification.md` referenced by Wave 6 R1
commit `db60864`), not yet implemented as code.

**Finding (P3):** No `ane_precision_proofs` module exists. This is design debt,
not a bug.

### llvm2-ir/src/lib.rs -- Cost model exports

The `cost_model` module is declared as `pub mod` (line 43) but **none of its 9
public types are re-exported** at the crate root:

- `CostModel` (trait)
- `CostModelGen` (enum)
- `AppleSiliconCostModel` (struct)
- `ComputeTarget` (enum)
- `NeonArrangement` (enum)
- `NeonOp` (enum)
- `CostEstimate` (struct)
- `DataTransferCost` (struct)
- `MultiTargetCostModel` (struct)

Users must write `llvm2_ir::cost_model::ComputeTarget` instead of
`llvm2_ir::ComputeTarget`. All other modules in llvm2-ir have root-level
re-exports. This is an **inconsistency** but not a compilation error.

**Finding (P2):** `cost_model` public types not re-exported at crate root,
unlike all other modules in llvm2-ir.

### llvm2-lower/src/lib.rs -- compute_graph exports

The `compute_graph` module is declared as `pub mod` (line 19) and has a
root-level re-export: `pub use compute_graph::TargetRecommendation` (line 30).
This is correct, though only one of many public types is re-exported. The
pattern matches other modules in this crate.

---

## 3. Dead Code Detection

Since `cargo check` cannot run, dead code detection is based on source analysis:

### Explicit `#[allow(dead_code)]` annotations

| File | Line | Item | Assessment |
|------|------|------|------------|
| `llvm2-lower/src/isel.rs` | 815 | `fn select_binop()` | Internal helper, likely used via codegen paths. Acceptable suppression. |
| `llvm2-codegen/src/relax.rs` | 70 | `struct BlockInfo` | Used in relaxation algorithm. Fields may be read in non-obvious ways. Acceptable. |
| `llvm2-codegen/benches/compile_bench.rs` | 68, 334 | Benchmark helper types | Benchmark code, dead code normal. |
| `llvm2-codegen/benches/compare_clang.rs` | 471 | Benchmark helper | Benchmark code, dead code normal. |

**Finding (P3):** 2 production `#[allow(dead_code)]` annotations. Both are
borderline acceptable -- the isel `select_binop` should be investigated to
determine if it's truly dead or just not yet wired up.

### No `#[ignore]` test annotations found

Zero instances of `#[ignore]`, `@skip`, or `xfail` across all 98 source files.
This is correct per project rules.

---

## 4. Documentation Gaps

### Public type documentation

Spot-checked all public types in the new Wave 5-6 modules:

- **gpu_semantics.rs**: All 4 public types (`GpuKernelShape`, `GpuReduceOp`,
  `SimdGroupOp`) and 10 public functions have doc comments with `///`.
  References to Metal Shading Language Specification included.

- **ane_semantics.rs**: All 5 public types (`AneElementType`, `AneTensorShape`,
  `Conv2dParams`, `ActivationFn`, `ElementWiseOp`) and 10 public functions
  have doc comments. Higham and Apple CoreML references included.

- **unified_synthesis.rs**: All 5 public types (`SynthTarget`,
  `TargetCandidate`, `NeonSynthOpcode`, `UnifiedSearchConfig`,
  `UnifiedProvenRule`, `UnifiedCegisLoop`) have doc comments.

- **cost_model.rs**: All 9 public types have doc comments with source
  citations (Dougall Johnson M1 Firestorm data).

- **neon_semantics.rs**: All public functions have doc comments with ARM DDI
  0487 section references.

**Finding:** Documentation quality is excellent across all new modules. No gaps
found in public API documentation.

---

## 5. Integration Coherence

### GPU and ANE semantics consistency

Both modules use consistent patterns:

| Aspect | GPU semantics | ANE semantics |
|--------|--------------|---------------|
| Array representation | `SmtExpr` arrays with BV64 indices | `SmtExpr` arrays with BV64 indices |
| Element types | Bitvector (`SmtSort::BitVec`) | FloatingPoint (`SmtSort::FloatingPoint(5,11)` for FP16) |
| Rounding mode | Not applicable (integer) | `RoundingMode::RNE` throughout |
| Function prefix | `encode_parallel_*` | `encode_ane_*` |
| Test count | 40 tests | 42 tests |

The divergence in element types is correct -- GPU operations are integer
bitvector parallel operations while ANE is FP16/FP32 matrix operations. Both
import from `crate::smt::{SmtExpr, SmtSort, RoundingMode}`.

**Finding:** Patterns are consistent and appropriate for each target's
characteristics.

### Cost model target coverage

The `MultiTargetCostModel` in `llvm2-ir/src/cost_model.rs` covers all 4 targets:
- `CpuScalar`: Per-opcode costs from `AppleSiliconCostModel`
- `Neon`: Per-arrangement vector costs with lane normalization
- `Gpu`: Dispatch overhead + kernel throughput model
- `Ane`: CoreML compilation + inference model

Tests verify GPU/ANE cost characteristics (62 tests in cost_model.rs).

### Unified synthesis target awareness

**Finding (P2):** `UnifiedCegisLoop` in `llvm2-verify/src/unified_synthesis.rs`
only searches **Scalar and NEON** targets. It does NOT import or reference
`gpu_semantics` or `ane_semantics` modules. The `SynthTarget` enum has only
two variants: `Scalar` and `Neon(VectorArrangement)`.

This means GPU and ANE have semantic encodings and cost models but NO
synthesis search integration. The synthesis loop cannot discover or verify
GPU/ANE lowering rules.

This is likely intentional for the current development phase (GPU/ANE synthesis
requires fundamentally different search strategies than instruction-level
scalar/NEON synthesis), but it creates a gap: the verification pipeline can
express GPU/ANE semantics but cannot automatically discover optimal GPU/ANE
lowerings.

---

## Summary of Findings

| ID | Severity | Description |
|----|----------|-------------|
| W7-1 | **P1** | Workspace build blocked by z4 git auth failure. Cannot compile or test any crate. |
| W7-2 | **P2** | `cost_model` public types not re-exported at `llvm2-ir` crate root (inconsistency) |
| W7-3 | **P2** | `UnifiedCegisLoop` only covers Scalar/NEON targets; GPU/ANE semantics not wired into synthesis |
| W7-4 | **P3** | `ane_precision_proofs` module not yet implemented (design-only) |
| W7-5 | **P3** | 2 `#[allow(dead_code)]` in production code (isel.rs:815, relax.rs:70) |

### Workspace Statistics

- **Total Rust LOC:** 96,182 across 6 crates and 98 source files
- **Total tests:** 2,517 (all via `#[test]`, zero `#[ignore]`)
- **New modules since Wave 4:** gpu_semantics, ane_semantics, unified_synthesis,
  neon_lowering_proofs, compute_graph, target_analysis, cost_model (expanded)
- **Documentation:** All public types documented with citations
- **Code quality:** No ignored tests, consistent patterns, proper module structure
