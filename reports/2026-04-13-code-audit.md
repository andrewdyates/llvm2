# LLVM2 Code Quality Audit Report

**Date:** 2026-04-13
**Auditor:** Claude (Researcher role)
**Baseline:** 69,717 LOC Rust across 6 crates + 4 tMIR stubs
**Part of #121**

---

## Executive Summary

LLVM2 is a substantial verified compiler backend (~70K LOC) with a clean architecture, comprehensive test suite (1,682+ tests, all passing), and a working end-to-end pipeline that compiles tMIR programs to AArch64 Mach-O object files. The project is in an impressive state for its age. However, several issues were found that range from a critical silent miscompilation (immediate shifts emit NOP) to architectural duplication and testing gaps. Eight issues were filed.

**Overall assessment:** Strong foundation with good test coverage and clean crate boundaries. The primary risks are: (1) encoding completeness gaps that cause silent miscompilation, (2) the verification system uses probabilistic testing rather than formal SMT proofs for 32/64-bit, and (3) register allocator complexity outpacing its test coverage.

---

## Build Status

| Check | Result |
|-------|--------|
| `cargo build` | PASS (5.36s) |
| `cargo test` | PASS (1,682+ tests, 0 failed, 0 ignored) |
| Compiler warnings | 4 warnings (unreachable patterns in isel.rs) |

All 6 crates compile cleanly. Tests span unit tests, integration tests, end-to-end pipeline tests, and Mach-O validation tests (including otool verification).

---

## Per-Crate Findings

### llvm2-ir (1,261 LOC inst.rs, 1,397 LOC aarch64_regs.rs)

**Architecture:** Clean type authority pattern -- single source of truth for all machine IR types. Well-documented with module-level docs explaining type ownership. 181 tests.

**Issue Found:**
- **[P1] Duplicate opcode: `Cset` (line 68) vs `CSet` (line 150)** in AArch64Opcode enum. Both represent the same CSET instruction but are used by different crates. Filed as #133.

**Strengths:**
- Typed index wrappers (InstId, BlockId, VRegId) prevent index confusion
- InstFlags bitfield is well-designed with proper set operations
- x86-64 scaffolding is clean and non-intrusive

### llvm2-lower (6,307 LOC isel.rs, 2,583 LOC adapter.rs)

**Architecture:** Clean separation between tMIR adapter (translation) and ISel (lowering). The adapter correctly translates all tMIR constructs. ISel covers arithmetic, comparison, branch, call, load/store, bitfield, float, and aggregate operations. 163 tests.

**Issues Found:**
- **4 unreachable pattern warnings** in isel.rs:5859 and 5993-5994 from duplicate `StrRI | StrRI` match arms. Likely copy-paste error. Filed as #136.
- ABI implementation covers basic Apple AArch64 calling convention but lacks HFA support and large struct by-reference. Filed as #140.

**Strengths:**
- Comprehensive variadic call support (Apple ABI: varargs floats on stack)
- Good aggregate type handling (struct return via sret, GEP lowering)
- Proof annotation extraction plumbing is in place

### llvm2-opt (14 modules, ~7K LOC)

**Architecture:** Well-factored optimization passes with a pass manager, pipeline levels (O0-O3), and proper dominator/loop analysis infrastructure. 171 tests.

**Strengths:**
- Proof-guided optimizations are unique and well-implemented (proof_opts.rs: 1,546 LOC)
  - NoOverflow: ADDS -> ADD (removes flag-setting overhead)
  - NonZeroDivisor: eliminates division-by-zero checks
  - NotNull: eliminates null checks
  - ValidBorrow: refines memory effects
  - ValidShift: eliminates shift-range checks
  - PositiveRefcount: eliminates retain/release pairs
- CSE handles commutativity correctly with proof merging
- Address mode folding (fold ADD imm into LDR/STR offset)
- Compare-select formation (CMP+branch+MOV -> CSEL/CSET)

**Gaps:**
- LICM has only 6 tests for 558 LOC
- No strength reduction pass (multiply by power of 2 -> shift)
- No tail call optimization
- No inlining (expected at this stage but worth tracking)

### llvm2-regalloc (12 modules, ~5K LOC)

**Architecture:** Two allocators (linear scan and greedy) with supporting infrastructure. Clean pipeline: liveness -> phi elimination -> allocation -> spill code -> post-RA coalescing. 133 tests.

**Issue Found:**
- **[P2] Test coverage insufficient for complexity.** Filed as #139. The greedy allocator (1,302 LOC) has complex eviction/splitting/cascade logic that needs more adversarial testing.

**Strengths:**
- Post-RA coalescing pass is well-implemented
- Spill slot reuse optimization
- Rematerialization support
- Critical edge splitting in phi elimination

**Gaps:**
- No register pressure tracking for spill cost estimation
- No live-through-call FPR allocation tests

### llvm2-codegen (10+ modules, ~15K LOC)

**Architecture:** Clean separation between encoding (per-format), unified encoder (dispatch), Mach-O writer, frame lowering, branch relaxation, and DWARF debug info. 503 tests + 7 integration test files.

**Issues Found:**
- **[P1] Immediate shifts (LslRI/LsrRI/AsrRI) emit NOP** instead of UBFM/SBFM encoding. Silent miscompilation. Filed as #134.
- **[P2] TBZ/TBNZ emit NOP** instead of test-and-branch encoding. Filed as #135.
- **[P2] 7 opcodes unimplemented** in unified encoder (Ubfm, Sbfm, Bfm, LdrRO, StrRO, LdrGot, LdrTlvp). Filed as #137.
- **14 LLVM-style typed aliases** return encoding errors instead of mapping to generic opcodes.

**Strengths:**
- AArch64 integer/memory/FP/branch encoding is thorough with ARM ARM references
- Mach-O writer produces valid object files (verified by otool in tests)
- Branch relaxation with fixpoint convergence
- DWARF CFI and compact unwind both implemented
- x86-64 encoder stub with proper REX/ModRM infrastructure

**Encoding Spot-Check (vs ARM ARM DDI 0487):**
- `encode_add_sub_shifted_reg`: Bit layout matches ARM ARM C4.1.3. sf/op/S/01011/shift/0/Rm/imm6/Rn/Rd. CORRECT.
- `encode_add_sub_imm`: Bit layout matches ARM ARM C4.1.2. sf/op/S/100010/sh/imm12/Rn/Rd. CORRECT.
- `encode_move_wide`: sf/opc/100101/hw/imm16/Rd. CORRECT.
- `encode_cond_branch`: 01010100/imm19/0/cond. CORRECT.
- Condition codes: All 16 values match ARM ARM C1.2.4.

### llvm2-verify (12 modules, ~9K LOC)

**Architecture:** SMT bitvector expression AST with concrete evaluator, proof obligation framework, and z4 CLI bridge. 322 tests.

**Issue Found:**
- **[P2] "Proofs" are actually probabilistic tests for 32/64-bit.** Filed as #138. The verify_by_evaluation function uses random sampling (100K trials) for widths > 8 bits. Only 8-bit proofs are truly exhaustive. The z4 bridge exists but is not used in tests.

**Strengths:**
- SMT expression AST is well-designed with proper BV/Bool sort separation
- Proof obligations correctly model negated equivalence checking
- Coverage: 5 arithmetic proofs, 4 NZCV flag proofs, 10 comparison proofs, 3 64-bit comparison proofs, 4 branch proofs, 11 peephole identity proofs, constant folding proofs, copy propagation proofs, CSE/LICM proofs, DCE proofs, CFG proofs, memory model proofs
- Negative tests verify that wrong rules ARE detected
- SMT2 serialization produces valid QF_BV queries

**Soundness of Proof Architecture:**
- The `negated_equivalence` construction is mathematically correct: `precond AND NOT(equiv)` being UNSAT implies `precond => equiv` is valid
- NZCV flag proofs correctly model the ARM carry flag as `a >=_u b` (not `a >_u b`)
- Overflow detection formula `sign(a) != sign(b) AND sign(a) != sign(a-b)` matches ARM ARM

---

## Design Review

### 2026-04-13-unified-solver-architecture.md (Master Design)

**Quality:** Excellent conceptual framework unifying superoptimization, transparency, AI-native compilation, and heterogeneous compute through a single solver abstraction.

**Gap:** Requires z4 features that do not exist (QF_ABV array theory for GPU, QF_FP for floating-point). All dependencies are blocked issues (#122-125).

### 2026-04-13-superoptimization.md

**Quality:** Thorough prior art review (STOKE, Souper, Denali/egg, Optgen, Alive2). Architecture is sound.

**Gap:** No implementation exists. CEGIS loop and offline synthesis are P2 features with no code.

### 2026-04-13-debugging-transparency.md

**Quality:** Good comparison of LLVM opt-remarks, GCC fopt-info, CompCert certificates.

**Gap:** No provenance tracking infrastructure exists. Event log is a future feature.

### 2026-04-13-ai-native-compilation.md

**Quality:** Well-motivated (ML compiler interaction via MLGO, CompilerGym, TVM).

**Gap:** No Compiler struct or library API exists. No trait-based heuristic hooks.

### 2026-04-13-heterogeneous-compute.md

**Quality:** Creative register-allocation analogy for compute allocation. Apple Silicon target analysis.

**Gap:** Furthest from implementation. Requires Metal, CoreML, and ANE backends.

**Overall Design Assessment:** Filed as #141. Designs are internally consistent and well-researched but describe a future 10x larger than the current codebase. The gap-analysis.md correctly identifies immediate priorities.

---

## Issues Filed

| # | Title | Severity | Category |
|---|-------|----------|----------|
| #133 | Duplicate opcode: Cset vs CSet in AArch64Opcode enum | P1 | Bug |
| #134 | Immediate shift opcodes emit NOP instead of real encoding | P1 | Bug |
| #135 | TBZ/TBNZ opcodes emit NOP instead of real encoding | P2 | Bug |
| #136 | 4 unreachable pattern warnings in isel.rs | P3 | Bug |
| #137 | Unimplemented opcodes in encode.rs (7 opcodes + 14 aliases) | P2 | Feature |
| #138 | Verification proofs use random sampling, not formal SMT | P2 | Documentation |
| #139 | Register allocator subsystem under-tested | P2 | Bug |
| #140 | Missing ABI features (HFA, large structs, exception handling) | P2 | Feature |
| #141 | Design docs disconnected from current implementation state | P3 | Documentation |

---

## Recommendations

### Immediate (P1)

1. **Fix immediate shift encoding** (#134) -- this is a silent miscompilation that will affect any program using constant shifts. The fix is straightforward: implement UBFM/SBFM encoding.
2. **Consolidate Cset/CSet** (#133) -- merge into single opcode to prevent confusion.

### Short-term (P2)

3. **Complete encoding gaps** (#137) -- Ubfm/Sbfm/Bfm are essential for bitfield operations and sign/zero extension. TBZ/TBNZ (#135) for efficient boolean branching.
4. **Integrate z4 CLI verification** (#138) -- run at least the lowering proofs through an actual SMT solver to upgrade from probabilistic to formal.
5. **Expand regalloc tests** (#139) -- focus on greedy allocator with complex CFGs and high register pressure.

### Long-term (P3)

6. **Type unification** (existing #73) -- the 3 independent MachFunction definitions are the single largest architectural debt.
7. **ABI completeness** (#140) -- HFA support, exception handling.
8. **Bridge designs to implementation** (#141) -- phase the vision docs into actionable milestones.

---

## Test Count Summary

| Crate | Unit Tests | Integration Tests | Doc Tests | Total |
|-------|-----------|-------------------|-----------|-------|
| llvm2-codegen | 503 | 160 | 3 | 666 |
| llvm2-ir | 181 | 22 | 0 | 203 |
| llvm2-lower | 163 | 17 | 0 | 180 |
| llvm2-opt | 171 | 0 | 1 | 172 |
| llvm2-regalloc | 133 | 0 | 1 | 134 |
| llvm2-verify | 322 | 0 | 1 | 323 |
| **Total** | **1,473** | **199** | **6** | **1,678** |
