# Wave 22 Status and Wave 23 Planning

**Author:** R1 (Researcher Agent)
**Date:** 2026-04-15
**Commit base:** c963a1c (main, post-Wave 21 merges)
**Scope:** Post-Wave 21 codebase assessment, verification coverage matrix, Wave 23 planning

---

## Executive Summary

The LLVM2 codebase has grown to **175,278 LOC Rust** across 6 production crates + stubs, with **1,514 passing tests** (3 failing due to stale proof count assertions, issue #219). The project has **25 proof categories** covering lowering, optimization, and backend correctness. Wave 21 delivered GVN, tail call optimization, loop opt proofs, Mach-O proofs, greedy RA proofs, and FP ISel (FABS/FSQRT).

Wave 22 is currently running with 7 items: stale proof count fix (#219), strength reduction proofs (#212), cmp combine proofs (#213), GVN proofs (#216), if-conversion pass (#207), NEON encoding (#215), and x86-64 RA adapter (#214).

**Key finding:** The verification gap has narrowed significantly since Wave 20, but four optimization passes still lack proofs: **GVN**, **TailCallOptimization**, **IfConversion** (being added), and **InstructionScheduler**. The scheduler has proofs for dependency preservation but lacks correctness proofs for the actual scheduling algorithm output. Additionally, the pipeline now runs 15 passes at O2/O3 (up from 13 in Wave 20) without corresponding proof coverage increases for the newest passes.

**Wave 23 focus recommendation:** Close the remaining verification gaps (GVN proofs, tail call proofs, scheduler algorithm proofs), complete the AArch64 encoding coverage (NEON SIMD), and start atomics support for real-program compilation.

**Snapshot:** 175,278 LOC | 4,729 `#[test]` annotations | 25 proof categories | 121 AArch64 opcodes defined / 105 encoded | 3 targets

---

## 1. Codebase Metrics

### 1.1 Scale by Crate

| Crate | LOC | Tests (#[test]) | Role |
|-------|-----|-----------------|------|
| llvm2-codegen | 52,298 | 1,450 | Encoding, Mach-O, ELF, DWARF, RISC-V, x86-64, frame, pipeline |
| llvm2-verify | 49,863 | 1,524 | SMT proofs, CEGIS, proof database, z4 bridge |
| llvm2-lower | 27,012 | 640 | ISel, ABI, tMIR adapter, compute graph |
| llvm2-opt | 19,184 | 401 | 15 optimization passes (O0-O3) |
| llvm2-regalloc | 13,241 | 330 | Linear scan + greedy allocator |
| llvm2-ir | 12,909 | 384 | MachFunction, opcodes, registers, cost model |
| stubs | 771 | - | tMIR development stubs |
| **Total** | **175,278** | **4,729** | |

### 1.2 Test Results

- **1,514 passed**, 3 failed, 0 ignored
- Failing tests (all in llvm2-verify, all stale count assertions):
  1. `proof_database::tests::test_all_categories_is_exhaustive` -- expects 24 categories, got 25
  2. `proof_database::tests::test_memory_proofs_count` -- expects 41 memory proofs, got 62
  3. `verify::tests::test_verifier_memory_model` -- expects 41, got 62
- Root cause: Wave 20/21 added proofs and a new category without updating hardcoded assertions
- Fix: Issue #219 (assigned to Wave 22)

### 1.3 Optimization Pipeline (O2/O3)

15 passes at O2/O3 (source: `crates/llvm2-opt/src/pipeline.rs`):

1. ProofOptimization (proof-guided)
2. ConstantFolding
3. CopyPropagation
4. CommonSubexprElim (CSE)
5. **GlobalValueNumbering (GVN)** -- NEW in Wave 21
6. LoopInvariantCodeMotion (LICM)
7. StrengthReduction
8. LoopUnroll
9. Peephole
10. AddrModeFormation
11. CmpSelectCombine
12. CmpBranchFusion
13. **TailCallOptimization** -- NEW in Wave 21
14. DeadCodeElimination
15. CfgSimplify

O3 iterates to fixpoint (max 10). O1 runs DCE + Peephole only.

---

## 2. Verification Coverage Matrix

### 2.1 Proof Categories (25 total)

| # | Category | Proofs | Status | Notes |
|---|----------|--------|--------|-------|
| 1 | Arithmetic | 16+ | COMPLETE | add/sub/mul/neg, I8-I64 |
| 2 | Division | 4 | COMPLETE | sdiv/udiv, I32/I64, preconditioned |
| 3 | Floating-Point | 8 | COMPLETE | fadd/fsub/fmul/fneg, F32/F64 |
| 4 | NZCV Flags | 4 | COMPLETE | N, Z, C, V flag lemmas |
| 5 | Comparison | 20 | COMPLETE | 10 conds x I32/I64 |
| 6 | Branch | 20 | COMPLETE | 10 conds x I32/I64 |
| 7 | Peephole | 18 | COMPLETE | 9 rules x 32-bit + 64-bit |
| 8 | Optimization (general) | varies | COMPLETE | const fold, AND/OR absorb, DCE, copy prop |
| 9 | Constant Folding | varies | COMPLETE | binary/unary ops, identities |
| 10 | Copy Propagation | varies | COMPLETE | multi-width variants |
| 11 | CSE/LICM | varies | COMPLETE | commutativity, determinism, loop invariance |
| 12 | Dead Code Elimination | varies | COMPLETE | multi-width variants |
| 13 | CFG Simplification | varies | COMPLETE | branch folding, empty blocks, duplicates |
| 14 | Memory | 62 | COMPLETE | load/store/roundtrip/interference/endianness/alignment/forwarding/subword/write-combining/array axioms/array range |
| 15 | NEON Lowering | 22 | COMPLETE | 11 ops x 2 arrangements |
| 16 | Vectorization | 31 | COMPLETE | scalar-to-NEON mapping |
| 17 | ANE Precision | varies | COMPLETE | FP16 bounded error |
| 18 | Register Allocation | 43+ | COMPLETE | non-interference, spill, phi, coalescing, greedy |
| 19 | Bitwise/Shift | 14 | COMPLETE | 7 ops x 2 widths |
| 20 | Constant Materialization | varies | COMPLETE | MOVZ, MOVK, ORR, MOVN strategies |
| 21 | Address Mode | varies | COMPLETE | base+imm, base+reg, scaled, writeback |
| 22 | Frame Layout | varies | COMPLETE | offsets, alignment, callee-save |
| 23 | Instruction Scheduling | varies | PARTIAL | dependency/memory ordering proofs, but not algorithm output |
| 24 | Mach-O Emission | varies | COMPLETE | relocation, symbol binding, structural invariants |
| 25 | Loop Optimization | varies | COMPLETE | unrolling, strength reduction, IV elimination |

### 2.2 Optimization Pass vs Proof Coverage Matrix

| Pass | In Pipeline | Has Proofs | Proof Category | Gap |
|------|-------------|------------|----------------|-----|
| ProofOptimization | O2, O3 | N/A | N/A | Consumes proofs, does not transform |
| ConstantFolding | O2, O3 | YES | ConstantFolding (#9) | None |
| CopyPropagation | O2, O3 | YES | CopyPropagation (#10) | None |
| CommonSubexprElim | O2, O3 | YES | CSE/LICM (#11) | None |
| **GlobalValueNumbering** | O2, O3 | **NO** | -- | **GVN is strictly stronger than CSE; needs its own proofs** |
| LoopInvariantCodeMotion | O2, O3 | YES | CSE/LICM (#11) | None |
| StrengthReduction | O2, O3 | PARTIAL | LoopOptimization (#25) | Loop opt proofs cover mul-to-add; Wave 22 #212 adding more |
| LoopUnroll | O2, O3 | YES | LoopOptimization (#25) | None |
| Peephole | O2, O3 | YES | Peephole (#7) | None |
| AddrModeFormation | O2, O3 | YES | AddressMode (#21) | None |
| CmpSelectCombine | O2, O3 | **NO** | -- | **Wave 22 #213 in progress** |
| CmpBranchFusion | O2, O3 | **NO** | -- | **Wave 22 #213 in progress** |
| **TailCallOptimization** | O2, O3 | **NO** | -- | **New pass from Wave 21, no proofs filed** |
| DeadCodeElimination | O2, O3 | YES | DeadCodeElimination (#12) | None |
| CfgSimplify | O2, O3 | YES | CfgSimplification (#13) | None |
| InstructionScheduler | Post-RA | PARTIAL | InstructionScheduling (#23) | Algorithm output correctness incomplete |

**Summary:** 4 of 15 O2/O3 passes have **NO** correctness proofs (GVN, CmpSelect, CmpBranch, TailCall). Wave 22 addresses 2 of these (CmpSelect/CmpBranch in #213). GVN proofs are filed as #216. TailCall proofs are filed as #218 but at P3.

### 2.3 Lowering Coverage Analysis

| tMIR Opcode Category | ISel Status | Proof Status |
|----------------------|-------------|--------------|
| Integer arithmetic (add/sub/mul/neg) | COMPLETE | COMPLETE |
| Integer division (sdiv/udiv/srem/urem) | COMPLETE | Division: COMPLETE |
| Bitwise (and/or/xor/bic/orn) | COMPLETE | BitwiseShift: COMPLETE |
| Shifts (shl/lsr/asr) | COMPLETE | BitwiseShift: COMPLETE |
| Integer comparisons (10 conditions) | COMPLETE | Comparison: COMPLETE |
| Branches (jump/brif/switch) | COMPLETE | Branch: COMPLETE |
| FP arithmetic (fadd/fsub/fmul/fdiv) | COMPLETE | FloatingPoint: COMPLETE |
| FP unary (fneg/fabs/fsqrt) | COMPLETE | FP: partial (fneg proved, fabs/fsqrt: Wave 21) |
| FP conversions (fcvt/scvtf/ucvtf) | COMPLETE | NOT PROVED |
| FP comparisons | COMPLETE | NOT PROVED |
| Memory (load/store) | COMPLETE | Memory: COMPLETE |
| Extensions (sext/zext/trunc/bitcast) | COMPLETE | NOT PROVED |
| Constants (iconst/fconst) | COMPLETE | ConstMaterialization: COMPLETE |
| Calls (direct/indirect/variadic) | COMPLETE | NOT PROVED |
| Struct GEP | COMPLETE | NOT PROVED |
| **Atomics** | **NOT IMPLEMENTED** | NOT PROVED |
| **NEON/SIMD ISel** | **NOT IMPLEMENTED** | NEON lowering proofs exist |

---

## 3. Gap Analysis

### 3.1 Highest-Impact Gaps

**Gap 1: Unproven O2/O3 optimization passes (P1)**
- GVN (#216), TailCall (#218), CmpSelect/CmpBranch (#213 in W22)
- Impact: Violates "verify before merge" rule

**Gap 2: AArch64 encoding coverage (P2)**
- 121 enum variants, 105 have encoders, 16 missing
- Missing are mostly pseudos (Phi, StackAlloc, Copy, Nop) and traps (TrapOverflow, etc.) which are lowered before encoding
- Real gaps: Retain/Release pseudo-ops need lowering to call sequences
- NEON SIMD encoding is the critical missing piece (#215 in W22)

**Gap 3: Atomic operations (P2)**
- No ISel, no encoding, no proofs for atomics (LDADD, CAS, etc.)
- Blocks compilation of concurrent programs
- Issue #217 filed

**Gap 4: FP conversion and comparison proofs (P2)**
- FcvtzsRR, FcvtzuRR, ScvtfRR, UcvtfRR, FcvtSD, FcvtDS all in ISel and encoder
- No proofs for float-to-int or int-to-float conversion correctness
- Fcmp has encoding but no semantic equivalence proof

**Gap 5: Extension/truncation lowering proofs (P3)**
- Sextend, Uextend, Trunc have ISel and encoding (SBFM, UBFM)
- No proofs that the bitfield moves produce correct results

### 3.2 ABI Completeness

| Feature | Status |
|---------|--------|
| Apple AArch64 calling convention | COMPLETE |
| Integer argument passing (X0-X7) | COMPLETE |
| FP argument passing (D0-D7, S0-S7) | COMPLETE |
| Vector argument passing | NOT IMPLEMENTED |
| Stack argument passing | COMPLETE |
| Variadic functions | COMPLETE |
| Aggregate passing (small struct) | COMPLETE |
| HFA classification | COMPLETE |
| Callee-saved registers | COMPLETE |
| Frame lowering (prologue/epilogue) | COMPLETE |
| Compact unwind info | COMPLETE |
| DWARF CFI | COMPLETE |
| Exception handling data structures | STUB (1,016 LOC) |
| C++ exception interop | NOT IMPLEMENTED |
| Stack unwinding (libunwind) | NOT INTEGRATED |

### 3.3 End-to-End Pipeline Status

| Phase | AArch64 | x86-64 | RISC-V |
|-------|---------|--------|--------|
| ISel | Full | Partial | Partial |
| Optimization | Full (15 passes) | Shared | Shared |
| Register Alloc | Full (linear + greedy) | Simplified | Simplified |
| Frame Lowering | Full | Basic | Basic |
| Encoding | 105/121 opcodes | ~30 opcodes | ~40 opcodes |
| Object Format | Mach-O (full) | ELF (full) | ELF (full) |
| E2E Tests | Mach-O link+run | ELF link+run | ELF link+run |

---

## 4. Wave 22 In-Flight Assessment

| TL | Issue | Title | Status |
|----|-------|-------|--------|
| TL1 | #219 | Fix stale proof counts | P1, bug -- straightforward |
| TL2 | #212 | Strength reduction proofs | P1, verification |
| TL3 | #213 | CmpBranch/CmpSelect proofs | P2, verification |
| TL4 | #216 | GVN proofs | P2, verification |
| TL5 | #207 | If-conversion pass | P2, new optimization |
| TL6 | #215 | NEON SIMD encoding | P2, codegen |
| TL7 | #214 | x86-64 RA adapter | P2, codegen |

**Assessment:** Wave 22 is heavily verification-focused (4/7 items are proofs), which is appropriate given the gap analysis. The if-conversion pass (#207) will add a 16th optimization pass to the pipeline, which will need proofs in Wave 23.

---

## 5. Wave 23 Issues (to be filed)

NOTE: GitHub authentication is expired. These issues should be filed when auth is restored.

### Issue 1: [verify] GVN optimization correctness proofs (P1)

**Labels:** P1, feature

The GVN pass (`crates/llvm2-opt/src/gvn.rs`, 1,187 LOC) runs at O2/O3 without correctness proofs. GVN is strictly more powerful than CSE (which has proofs) -- it uses value numbers for transitive reasoning and eliminates redundant loads.

**Scope:** (1) Value-number-based elimination correctness, (2) commutative normalization, (3) load value numbering with no intervening store, (4) store/call barrier invalidation.

**Acceptance:** 8+ proof obligations, new GVN proof category in ProofCategory enum, all pass.

**Blocked by:** #216 (if completed in W22, close this as duplicate).

---

### Issue 2: [verify] Tail call optimization correctness proofs (P1)

**Labels:** P1, feature

TailCallOptimization (`crates/llvm2-opt/src/tail_call.rs`, 810 LOC) runs at O2/O3 without correctness proofs. It modifies stack frames (reuses caller's frame for callee), which is safety-critical.

**Scope:** (1) Tail position detection correctness, (2) stack frame reuse without corruption, (3) callee-saved register preservation, (4) return value forwarding.

**Acceptance:** 6+ proof obligations, new TailCall proof category, all pass.

**References:** Frame proofs (`crates/llvm2-verify/src/frame_proofs.rs`) as analogous model.

---

### Issue 3: [verify] If-conversion correctness proofs (P2)

**Labels:** P2, feature

The if-conversion pass (being added in Wave 22 #207) converts diamond CFG patterns to predicated instructions (CSEL/CSINC). Once landed, it will run at O2/O3 without proofs.

**Scope:** (1) Diamond pattern detection correctness, (2) CSEL semantics matches branch-based original, (3) CSINC alias correctness, (4) no-op on non-diamond patterns.

**Acceptance:** 6+ proofs, all pass.

**Blocked by:** #207 (pass must exist before proofs can be written).

---

### Issue 4: [verify] FP conversion lowering proofs (P2)

**Labels:** P2, feature

Six FP conversion opcodes have ISel and encoding but no semantic equivalence proofs:
- FcvtzsRR (float to signed int, round toward zero)
- FcvtzuRR (float to unsigned int, round toward zero)
- ScvtfRR (signed int to float)
- UcvtfRR (unsigned int to float)
- FcvtSD (f32 to f64, precision widen)
- FcvtDS (f64 to f32, precision narrow)

**Scope:** Prove tMIR conversion semantics equals AArch64 instruction semantics for each opcode across applicable types.

**Acceptance:** 6+ proofs (one per opcode minimum), new FPConversion proof category or extension of FloatingPoint category.

**References:** FP lowering proofs (`crates/llvm2-verify/src/lowering_proof.rs:all_fp_lowering_proofs`).

---

### Issue 5: [lower][codegen] Atomic memory operations (P2, if not completed in W22)

**Labels:** P2, feature

No ISel, encoding, or proofs for atomic operations. Blocks compilation of any concurrent program.

**Scope:** (1) ISel for atomic load/store/RMW (tMIR atomics -> LDADD/LDCLR/SWPAL/CAS), (2) AArch64 encoding for LDADD, CAS, SWP, LDCLR + memory barrier DMB/DSB/ISB, (3) at least 4 proof obligations for store-load ordering.

**Acceptance:** Compile and correctly encode a function with atomic increment.

**Blocked by:** #217 (carry-forward from W22 if not completed).

---

### Issue 6: [codegen] Retain/Release lowering to runtime calls (P2)

**Labels:** P2, feature

The `Retain` and `Release` pseudo-instructions are in the AArch64Opcode enum with correct flags (READS_MEMORY | WRITES_MEMORY | HAS_SIDE_EFFECTS) but have no lowering to actual runtime call sequences. They need to be expanded to `BL _swift_retain` / `BL _swift_release` (or equivalent tRust runtime symbols) during a pseudo-expansion pass before encoding.

**Scope:** (1) Pseudo-expansion pass that converts Retain/Release to BL + appropriate ABI setup, (2) symbol emission for retain/release runtime functions, (3) test that the expanded sequence encodes correctly.

**Acceptance:** E2E test compiling a function with retain/release that produces valid Mach-O.

---

### Issue 7: [verify] Extension and truncation lowering proofs (P3)

**Labels:** P3, feature

Sextend, Uextend, and Trunc have ISel lowering (to SBFM/UBFM/AND) and AArch64 encoding but no proofs that the bitfield moves produce semantically correct results.

**Scope:** Prove: (1) SBFM with correct immr/imms equals sign-extension, (2) UBFM with correct immr/imms equals zero-extension, (3) AND mask truncation preserves low bits.

**Acceptance:** 6+ proofs covering I8/I16/I32/I64 width combinations.

**References:** Bitwise/shift proofs as model (`crates/llvm2-verify/src/lowering_proof.rs:all_bitwise_shift_proofs`).

---

## 6. Verification Coverage Summary

```
PROOF COVERAGE BY PIPELINE STAGE:
==================================

tMIR -> MachIR Lowering (ISel):
  [PROVED] Integer arithmetic (add/sub/mul/neg) .......... 16+ proofs
  [PROVED] Integer division (sdiv/udiv) .................. 4 proofs
  [PROVED] Bitwise/shift (and/or/xor/shl/lsr/asr) ....... 14 proofs
  [PROVED] Comparisons (10 conds x I32/I64) .............. 20 proofs
  [PROVED] Branches (10 conds x I32/I64) ................. 20 proofs
  [PROVED] FP arithmetic (fadd/fsub/fmul/fneg) ........... 8 proofs
  [PROVED] NZCV flags ................................... 4 proofs
  [PROVED] Memory (load/store/roundtrip) ................. 62 proofs
  [PROVED] NEON lowering (11 ops x 2 arrangements) ....... 22 proofs
  [PROVED] Constant materialization ...................... varies
  [UNPROVED] FP conversions (fcvt/scvtf/ucvtf)
  [UNPROVED] FP comparisons (fcmp)
  [UNPROVED] Extensions (sext/zext/trunc)
  [UNPROVED] Calls (direct/indirect/variadic)
  [MISSING]  Atomics (no ISel)

Optimization Passes:
  [PROVED] ConstantFolding ............. ConstantFolding category
  [PROVED] CopyPropagation ............ CopyPropagation category
  [PROVED] CSE ........................ CSE/LICM category
  [PROVED] LICM ....................... CSE/LICM category
  [PROVED] StrengthReduction .......... LoopOptimization (partial, W22 expanding)
  [PROVED] LoopUnroll ................. LoopOptimization category
  [PROVED] Peephole ................... Peephole category
  [PROVED] AddrModeFormation .......... AddressMode category
  [PROVED] DCE ........................ DeadCodeElimination category
  [PROVED] CfgSimplify ................ CfgSimplification category
  [IN PROGRESS] CmpSelectCombine ...... W22 #213
  [IN PROGRESS] CmpBranchFusion ....... W22 #213
  [UNPROVED] GlobalValueNumbering ..... W22 #216 (may not complete)
  [UNPROVED] TailCallOptimization ..... #218 P3
  [NOT YET] IfConversion .............. W22 #207 (pass being added)

Register Allocation:
  [PROVED] Linear scan + greedy ........ 43+ proofs (non-interference, spill, coalescing)

Backend:
  [PROVED] Frame layout ................ Frame category
  [PROVED] Instruction scheduling ....... Scheduling category (partial)
  [PROVED] Mach-O emission ............. MachOEmission category
  [PROVED] Vectorization ............... Vectorization category (31 proofs)
  [PROVED] ANE precision ............... AnePrecision category
```

---

## 7. Recommendations for Next Phase

1. **Achieve "all O2 passes proved" milestone.** This is the most important near-term goal. After Wave 22, only GVN, TailCall, and IfConversion should remain unproven. Wave 23 should close all three.

2. **FP conversion proofs are a significant gap.** Six opcodes (FcvtzsRR, FcvtzuRR, ScvtfRR, UcvtfRR, FcvtSD, FcvtDS) are in the ISel and encoder but have no semantic equivalence proofs. These are non-trivial to prove (IEEE 754 rounding semantics) but are critical for any program that mixes integers and floats.

3. **Atomics are the biggest functional gap.** No concurrent program can compile without LDADD, CAS, SWPAL, DMB, etc. This is the #1 feature gap for real-program compilation.

4. **Consider marking the scheduler as "needs deeper proofs."** The current scheduler proofs cover dependency preservation and memory ordering constraints, but do not prove that the actual scheduling algorithm produces correct output (i.e., that the reordered instruction sequence computes the same values). This is a subtle gap.

5. **The stale proof count pattern (#219) should be eliminated architecturally.** Replace hardcoded numeric assertions with computed values (e.g., `assert_eq!(count, expected_from_registry())`) to prevent this recurring issue.
