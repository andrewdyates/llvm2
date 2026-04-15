# Wave 21 Planning and Gap Analysis

**Author:** R1 (Researcher Agent)
**Date:** 2026-04-15
**Commit base:** 2d34b2e (main, post-Wave 20 merges)
**Scope:** Assess current state, identify gaps, plan Wave 22

---

## Executive Summary

The LLVM2 codebase has grown to 168,574 LOC Rust across 6 production crates with 4,591 tests and 23 proof categories. Wave 20 delivered QF_ABV array theory, QF_FP floating-point theory, bounded quantifiers, RISC-V RV64GC pipeline, tMIR proof annotations, and x86-64 E2E ELF tests. Wave 21 is currently in-flight with 7 techleads addressing stack overflow bugs, GVN optimization, loop proof gaps, Mach-O linking proofs, tail call optimization, greedy regalloc proofs, and FP instruction selection.

**Key finding:** Three optimization passes (strength reduction, cmp-branch fusion, cmp-select combine) have NO correctness proofs. These are executed at O2/O3 and represent the largest verification gap. Additionally, x86-64 and RISC-V pipelines lack regalloc integration (using simplified direct assignment) and the ABI layer is missing exception handling support.

**Wave 22 focus recommendation:** Verification of unproven optimization passes, x86-64 regalloc adapter, and ABI completeness. The project has transitioned from "build pipeline stages" to "prove stages correct" -- every new pass must ship with proofs.

**Snapshot:** 168,574 LOC Rust | 4,591 tests | 23 proof categories | ~110 AArch64 opcodes encoded | 3 targets (AArch64 full, x86-64/RISC-V pipeline stubs)

---

## 1. Codebase Status

### Scale by Crate

| Crate | Files | LOC | Tests | Role |
|-------|-------|-----|-------|------|
| llvm2-codegen | 42 | 39,575 | 1,450 | Encoding, Mach-O, ELF, DWARF, frame |
| llvm2-verify | 35 | 46,365 | 1,452 | SMT proofs, CEGIS, proof database |
| llvm2-lower | 11 | 25,391 | 594 | ISel, ABI, tMIR adapter |
| llvm2-opt | 23 | 17,174 | 357 | 13 optimization passes (O0-O3) |
| llvm2-regalloc | 13 | 13,241 | 330 | Linear scan + greedy allocator |
| llvm2-ir | 16 | 12,352 | 362 | MachFunction, opcodes, registers |
| **Total** | **140** | **168,574** | **4,591** | |

### Optimization Pipeline (O2/O3)

The pipeline runs 13 passes at O2/O3 (source: `crates/llvm2-opt/src/pipeline.rs`):

1. ProofOptimization (proof-guided)
2. ConstantFolding
3. CopyPropagation
4. CommonSubexprElim (CSE)
5. LoopInvariantCodeMotion (LICM)
6. StrengthReduction (Wave 19)
7. LoopUnroll (Wave 19)
8. Peephole
9. AddrModeFormation
10. CmpSelectCombine
11. CmpBranchFusion
12. DeadCodeElimination
13. CfgSimplify

O3 iterates to fixpoint (max 10 iterations). O1 runs only DCE + Peephole.

### Proof Database (23 categories)

| Category | Proofs | Status |
|----------|--------|--------|
| Arithmetic | 16+ | Complete (add/sub/mul/neg, I8-I64) |
| Division | 4 | Complete (sdiv/udiv, I32/I64) |
| Floating-Point | 8 | Complete (fadd/fsub/fmul/fneg, F32/F64) |
| NZCV Flags | 4 | Complete |
| Comparison | 20 | Complete (10 conds x I32/I64) |
| Branch | 20 | Complete (10 conds x I32/I64) |
| Peephole | 18 | Complete (9 rules x 64-bit + 32-bit) |
| Optimization (general) | varies | Complete (const fold, AND/OR absorb, DCE, copy prop) |
| Constant Folding | varies | Complete (binary/unary/identities) |
| Copy Propagation | varies | Complete |
| CSE/LICM | varies | Complete |
| Dead Code Elimination | varies | Complete |
| CFG Simplification | varies | Complete (branch folding, empty blocks) |
| Memory | 41 | Complete (load/store/roundtrip/non-interference) |
| NEON Lowering | 22 | Complete (11 ops x 2 arrangements) |
| Vectorization | 31 | Complete |
| ANE Precision | varies | Complete (FP16 bounded error) |
| Register Allocation | varies | Complete (non-interference, spill, phi, coalescing) |
| Bitwise/Shift | 14 | Complete (7 ops x 2 widths) |
| Constant Materialization | varies | Complete (MOVZ, MOVK, ORR, MOVN) |
| Address Mode | varies | Complete (base+imm, base+reg, scaled, writeback) |
| Frame Layout | varies | Complete (offsets, alignment, callee-save) |
| Instruction Scheduling | varies | Complete (dependency, memory ordering) |

### End-to-End Pipeline Status

| Phase | AArch64 | x86-64 | RISC-V |
|-------|---------|--------|--------|
| ISel | Full | Partial (basic ops) | Partial (basic ops) |
| Optimization | Full (13 passes) | Shared | Shared |
| Register Allocation | Full (linear scan + greedy) | Simplified (no adapter) | Simplified (no adapter) |
| Frame Lowering | Full (prologue/epilogue/unwind) | Basic (prologue/epilogue) | Basic (prologue/epilogue) |
| Encoding | ~110 opcodes | ~30 opcodes | ~40 opcodes |
| Object Format | Mach-O (full) | ELF (full) | ELF (full) |
| E2E Test | Mach-O link+run | ELF link+run | ELF link+run |

---

## 2. Wave 21 In-Flight Assessment

| TL | Issue | Title | Expected Impact |
|----|-------|-------|----------------|
| TL1 | #205 | Stack overflow in test_compile_ir_function_with_proofs | P1 bug fix -- unblocks CI |
| TL2 | #204 | GVN optimization pass | New optimization -- subsumes CSE for cross-block redundancy |
| TL3 | #202 | Loop optimization correctness proofs | Verification -- proofs for unrolling + strength reduction |
| TL4 | #203 | E2E Mach-O linking correctness proofs | Verification -- relocation arithmetic, symbol binding |
| TL5 | #206 | Tail call optimization pass | New optimization -- eliminates tail-recursive stack frames |
| TL6 | #208 | Greedy register allocator correctness proofs | Verification -- eviction, splitting, cascade proofs |
| TL7 | #211 | FP instruction selection | ISel -- FSQRT, FMA, FABS, FMAX, FMIN, rounding |

**Assessment:** Well-balanced wave mixing bug fixes (1), new optimizations (2), verification deepening (3), and ISel expansion (1). The GVN pass (#204) and loop proofs (#202) are highest-impact items.

---

## 3. Gap Analysis

### 3.1 Unverified Optimization Passes (CRITICAL)

Three optimization passes run at O2/O3 but have NO correctness proofs:

| Pass | Module | In Pipeline | Proof Module | Gap |
|------|--------|-------------|-------------|-----|
| StrengthReduction | `strength_reduce.rs` | O2, O3 | NONE | No proof of mul->shift/add equivalence |
| CmpBranchFusion | `cmp_branch_fusion.rs` | O2, O3 | NONE | No proof of CMP+BCond -> CBZ/CBNZ/TBZ/TBNZ |
| CmpSelectCombine | `cmp_select.rs` | O2, O3 | NONE | No proof of diamond CFG -> CSEL/CSET |

**Why this matters:** LLVM2's core value proposition is verified compilation. Running unproven optimization passes at O2 contradicts this promise. These three passes were added in Waves 18-19 without accompanying proofs.

Wave 21 TL3 (#202) covers loop optimization proofs (unrolling + strength reduction), which may partially address the strength reduction gap. However, cmp_branch_fusion and cmp_select proofs are not assigned.

### 3.2 x86-64 / RISC-V Regalloc Gap

Both x86-64 and RISC-V pipelines use "simplified register assignment" -- a direct linear mapping of VRegs to physical registers. This works for simple test functions but will fail for:
- Functions with more live values than allocatable registers
- Spill/reload scenarios
- Call-clobber handling

The existing regalloc (`llvm2-regalloc`) operates on AArch64-centric types (`PReg`, `AArch64Opcode`). An x86-64 adapter (analogous to `ir_to_regalloc()` in `pipeline.rs`) is needed.

### 3.3 Missing ABI Features

| Feature | Status | Impact |
|---------|--------|--------|
| Exception handling (unwind tables) | Stub only (`exception_handling.rs`, 1,016 LOC) | Cannot interop with C++ or Swift code |
| SIMD vector argument passing | Not implemented | Cannot pass vector types through ABI |
| Variadic function support | Partial (ISel handles, ABI stub) | Limited to basic variadic calls |
| Stack unwinding (libunwind) | Not integrated | Debuggers cannot walk stack |

Issue #140 tracks SIMD args and libunwind but is in `needs-review` state.

### 3.4 Encoding Coverage Gaps

The AArch64 encoder handles ~110 opcode match arms covering most integer, memory, branch, FP, and extension instructions. Notable gaps:

| Missing Category | Examples | Impact |
|-----------------|----------|--------|
| NEON SIMD encoding | FADD.4S, ADD.8H, MUL.4S, etc. | Cannot emit vectorized code |
| Atomic operations | LDADD, LDCLR, SWPAL, CAS, etc. | No concurrent data structure support |
| System instructions | MRS, MSR, DMB, DSB, ISB | No cache/barrier/system register access |
| Crypto extensions | AESD, AESE, SHA256H, etc. | No hardware crypto |

The NEON gap is most impactful since the optimizer already has vectorization support (`vectorize.rs`) but the codegen cannot emit the NEON instructions.

### 3.5 ISel Coverage Gaps

The ISel (`crates/llvm2-lower/src/isel.rs`) handles core arithmetic, logic, shifts, comparisons, branches, FP arithmetic, memory, and extensions. Wave 21 #211 is adding FP ISel (sqrt, fma, abs, max, min). Remaining gaps:

- No NEON/SIMD ISel (tMIR vector ops -> AArch64 NEON)
- No atomic memory operations
- No overflow-checked multiply (only add/sub have ADDS/SUBS)

---

## 4. Wave 22 Recommendations

### Filed Issues (see below)

| Priority | Issue | Title | Category |
|----------|-------|-------|----------|
| P1 | #212 | Strength reduction correctness proofs | Verification |
| P2 | #213 | CmpBranchFusion + CmpSelectCombine correctness proofs | Verification |
| P2 | #214 | x86-64 register allocator adapter | Codegen |
| P2 | #215 | NEON SIMD binary encoding | Codegen |
| P2 | #216 | GVN optimization correctness proofs | Verification |
| P2 | #217 | Atomic memory operations (ISel + encoding) | Lower + Codegen |
| P3 | #218 | Tail call optimization correctness proofs | Verification |

### Suggested Wave 22 Assignment (7 Techleads)

| TL | Issue | Type | Rationale |
|----|-------|------|-----------|
| TL1 | #212 Strength reduction proofs | Verification | CRITICAL: unverified pass in O2 pipeline |
| TL2 | #213 CmpBranch/CmpSelect proofs | Verification | CRITICAL: two unverified passes in O2 |
| TL3 | #216 GVN correctness proofs | Verification | Verify GVN pass from Wave 21 |
| TL4 | #214 x86-64 regalloc adapter | Codegen | Unblocks x86-64 for real functions |
| TL5 | #215 NEON SIMD encoding | Codegen | Connects vectorizer to binary output |
| TL6 | #217 Atomic operations | Lower + Codegen | Foundation for concurrent programs |
| TL7 | #218 TCO proofs (if TCO lands W21) | Verification | Verify tail call pass from Wave 21 |

**Theme:** Wave 22 is a "verification catch-up" wave. The project added several optimization passes in Waves 18-20 without proofs. These must be caught up before adding more features.

---

## 5. Metrics Comparison

| Metric | Wave 19 | Wave 20 | Current (pre-W21 merge) | Trend |
|--------|---------|---------|--------------------------|-------|
| LOC (total Rust) | 163,605 | 168,574 | 168,574 | +3.0% |
| Test count | 4,458 | 4,591 | 4,591 | +3.0% |
| Proof categories | 23 | 23 | 23 | +0 |
| Source files (crates) | 142 | 140 | 140 | -2 (consolidation) |
| AArch64 encoded opcodes | ~105 | ~110 | ~110 | +5 |
| Open issues (actionable) | 16 | 12+ | 12+ | stable |

---

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Unverified passes ship to production | HIGH | HIGH | Wave 22 proof catch-up (this report) |
| x86-64 simplified regalloc breaks on complex functions | HIGH | MEDIUM | x86-64 regalloc adapter issue filed |
| NEON vectorizer produces code that cannot be encoded | MEDIUM | HIGH | NEON encoding issue filed |
| Stack overflow (#205) blocks CI | HIGH | HIGH | Wave 21 TL1 assigned |
| z4 integration remains blocked | HIGH | LOW (short-term) | Mock evaluation sufficient for now |

---

## 7. Architectural Notes

### Pipeline Maturity Assessment

The AArch64 pipeline is functionally complete: ISel -> opt (13 passes) -> regalloc (linear scan + greedy) -> frame lowering (prologue/epilogue/compact unwind) -> encoding (~110 opcodes) -> Mach-O emission -> E2E link+run verified. The primary gap is verification of 3 optimization passes.

x86-64 and RISC-V are at "proof of concept" maturity: basic ISel, simplified regalloc, basic frame lowering, partial encoding, ELF emission, E2E link+run for simple functions. Both need regalloc adapters to handle realistic functions.

### Verification Coverage Model

The proof database has 23 categories covering: instruction lowering, flag computation, comparisons, branches, all major optimization passes (const fold, copy prop, CSE, LICM, DCE, CFG simplify, peephole, addr mode), memory model, NEON/vectorization, ANE precision, register allocation, frame layout, and instruction scheduling. The three unverified passes (strength reduction, cmp-branch fusion, cmp-select combine) are the only gap in the "every pass has proofs" goal.
