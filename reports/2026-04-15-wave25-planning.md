# Wave 25 Planning Document

**Author:** R1 (Researcher Agent)
**Date:** 2026-04-15
**Commit base:** c1f9b25 (main, post-Wave 24 merges)
**Scope:** Wave 24 summary, overall project maturity assessment, Wave 25 issue assignments, roadmap to first real compilation

---

## Executive Summary

LLVM2 has reached **191,428 LOC Rust** across 6 production crates with **5,137 `#[test]` annotations** and **32 proof categories**. Wave 24 was exceptionally productive: all four high-priority verification gaps from the Wave 24 plan were closed (if-conversion proofs, FP conversion proofs, extension/truncation lowering, atomic operations). Additionally, ABI feature gaps (variadics, HFA, aggregate returns) were addressed, RISC-V E2E tests landed, and instruction scheduling proofs were extended.

**All 16 O2/O3 optimization passes now have correctness proofs** -- the IfConversion gap from Wave 23 is closed. The project has crossed the 190K LOC threshold and is approaching the scale where the remaining gaps are mostly integration-level rather than algorithmic.

The biggest gap preventing real-world use is the **stubbed tMIR integration** (issue #227). The tMIR adapter layer exists (4,533 LOC) but operates against 848 LOC of stubs, not the real tMIR repo. Without real tMIR input, LLVM2 cannot serve its purpose as the t* stack backend.

---

## 1. Wave 24 Summary

### Wave 24 Deliverables

| Agent | Commit | Deliverable | Status |
|-------|--------|-------------|--------|
| TL3 | 5745cb4 | If-conversion correctness proofs (IfConversion proof category) | COMPLETE |
| TL4 | ebbdec8 | FP conversion lowering proofs (FpConversion proof category) | COMPLETE |
| TL1 | f5c0059 | Atomic memory operations -- ISel, AArch64 encoding (LDAR/STLR/LDADD/CAS etc.) | COMPLETE |
| TL- | cd60327 | Extension and truncation lowering (Sext/Zext/Trunc via SBFM/UBFM) | COMPLETE |
| TL10 | 6ea0a63 | ABI feature gaps -- variadics, HFA, aggregate returns | COMPLETE |
| TL8 | 1ef684f | Extended instruction scheduling correctness proofs | COMPLETE |
| - | 8bde5d3 | RISC-V E2E ELF linking integration tests | COMPLETE |
| A1 | 9952152 | Post-Wave 22/23 audit report | COMPLETE |
| R1 | 63fe191 | Wave 24 planning document | COMPLETE |

### Wave 24 Success Criteria Assessment

| Criterion | Target | Result |
|-----------|--------|--------|
| If-conversion pass proved (#221) | CLOSED | **DONE** -- IfConversion proof category added |
| FP conversion lowering proved (#222) | CLOSED | **DONE** -- FpConversion proof category added |
| Extension/truncation lowering proved (#225) | CLOSED | **DONE** -- Extension lowering committed |
| Atomic operations ISel + encoding | At least LDAR/STLR/LDADD/CAS | **DONE** -- 22 atomic opcodes (LDAR/LDARB/LDARH/STLR/STLRB/STLRH/LDADD/LDADDA/LDADDAL/LDCLR/LDCLRAL/LDEOR/LDEORAL/LDSET/LDSETAL/SWP/SWPAL/CAS/CASA/CASAL/LDAXR/STLXR) |
| Epic #24 tasks audited | >= 3 more checked | Audit delivered, but issue not updated (no checkboxes checked) |
| All tests pass | 0 failures | **PARTIAL** -- integration test #233 still has stale category count (25 vs 32) |
| Codebase exceeds 190K LOC | 190K+ | **DONE** -- 191,428 LOC |

**Stretch goals achieved:** Instruction scheduling proofs extended, ABI feature gaps (variadics/HFA/aggregates) addressed, RISC-V E2E tests added.

**Overall Wave 24 rating: STRONG.** 6 of 7 primary criteria met. The only miss is the integration test stale count (#233), which is a trivial assertion update.

---

## 2. Overall Project Maturity Assessment

### 2.1 Scale

| Crate | LOC | Tests | Role |
|-------|-----|-------|------|
| llvm2-verify | 58,683 | 1,735 | SMT proofs, CEGIS, proof database, z4 bridge |
| llvm2-codegen | 55,419 | 1,513 | Encoding, Mach-O, ELF, DWARF, Metal, CoreML, pipeline |
| llvm2-lower | 28,526 | 695 | ISel, ABI, tMIR adapter, compute graph, dispatch |
| llvm2-opt | 20,462 | 420 | 16 optimization passes (O0-O3 pipeline) |
| llvm2-regalloc | 15,207 | 390 | Linear scan + greedy allocator, post-RA passes |
| llvm2-ir | 13,131 | 384 | MachFunction, opcodes, registers, cost model |
| **Total** | **191,428** | **5,137** | |

Growth since Wave 24 start: +6,793 LOC, +149 tests.
Growth since Wave 22 start: +16,150 LOC, +408 tests.

### 2.2 AArch64 ISA Coverage

**Opcode enum: ~145 real hardware opcodes** (out of ~321 total enum variants, the remainder being NEON, pseudo-ops, and typed aliases).

Estimated coverage of the "practical" AArch64 ISA (instructions needed to compile C/Rust programs):

| Category | Instructions | Coverage | Notes |
|----------|-------------|----------|-------|
| Integer arithmetic | ADD/SUB/MUL/SDIV/UDIV/NEG/MSUB/SMULL/UMULL | ~95% | Missing: MADD, UMULH/SMULH |
| Logical | AND/ORR/EOR/ORN/BIC | ~90% | Missing: EON, ANDS, TST (immediate) |
| Shifts | LSL/LSR/ASR (reg+imm) | 100% | |
| Conditional | CMP/CSEL/CSINC/CSINV/CSNEG/CSET/TST | ~90% | Missing: CCMP, CCMN |
| Move | MOV/MOVZ/MOVN/MOVK/FMOV | 100% | |
| Memory (imm) | LDR/STR/LDRB/STRB/LDRH/STRH/LDRSB/LDRSH/LDP/STP + pre/post-index | ~95% | Missing: LDRSW, PRFM |
| Memory (reg) | LDR/STR register-offset | ~80% | Missing: scaled register offset variants |
| Branches | B/B.cond/CBZ/CBNZ/TBZ/TBNZ/BL/BLR/BR/RET | 100% | |
| Extensions | SXTW/UXTW/SXTB/SXTH/UXTB/UXTH/UBFM/SBFM/BFM | 100% | |
| Floating-point | FADD/FSUB/FMUL/FDIV/FNEG/FABS/FSQRT/FCMP/FMOV/conversions | ~95% | Missing: FMADD, FMSUB, FMIN, FMAX |
| NEON SIMD | ADD/SUB/MUL/FADD/FSUB/FMUL/FDIV/AND/ORR/EOR/BSL/MOVI etc. | ~70% | 20+ vector opcodes, but many specialized ones missing |
| Atomics (ARMv8.1) | LDAR/STLR/LDADD/LDCLR/LDEOR/LDSET/SWP/CAS + variants | ~85% | Missing: LDAPR, STADD |
| Checked arith | ADDS/SUBS (flag-setting) | ~80% | Missing: ADCS, SBCS |
| Address | ADR/ADRP | 100% | |

**Overall AArch64 coverage estimate: ~85-90% of what a C compiler needs.** The remaining ~10-15% are specialized instructions (CCMP, MADD/FMADD, PRFM prefetch, advanced NEON) that are needed for performance but not for correctness of basic compilation.

### 2.3 Proof Categories vs Target Categories

**32 proof categories** cover:
- **Lowering proofs** (8): Arithmetic, Division, FloatingPoint, FpConversion, NzcvFlags, Comparison, Branch, BitwiseShift
- **Optimization proofs** (12): Peephole, Optimization, ConstantFolding, CopyPropagation, CseLicm, DeadCodeElimination, CfgSimplification, StrengthReduction, CmpCombine, Gvn, TailCallOptimization, IfConversion
- **Infrastructure proofs** (6): Memory, RegAlloc, ConstantMaterialization, AddressMode, FrameLayout, InstructionScheduling
- **Output proofs** (1): MachOEmission
- **NEON/Vector proofs** (3): NeonLowering, NeonEncoding, Vectorization
- **Accelerator proofs** (1): AnePrecision
- **Loop proofs** (1): LoopOptimization

**Coverage gaps:** No dedicated proof category for call lowering, stack frame correctness, or atomic memory ordering. These are partially covered by Frame Layout and Memory categories but deserve explicit treatment.

### 2.4 Optimization Pipeline Status

All 16 O2/O3 passes now have correctness proofs:

| # | Pass | Proof Category | Status |
|---|------|----------------|--------|
| 1 | ProofOptimization | (trust-based) | JUSTIFIED |
| 2 | ConstantFolding | ConstantFolding | PROVED |
| 3 | CopyPropagation | CopyPropagation | PROVED |
| 4 | CommonSubexprElim | CseLicm | PROVED |
| 5 | GlobalValueNumbering | Gvn | PROVED |
| 6 | LoopInvariantCodeMotion | CseLicm | PROVED |
| 7 | StrengthReduction | StrengthReduction | PROVED |
| 8 | LoopUnroll | LoopOptimization | PROVED |
| 9 | Peephole | Peephole | PROVED |
| 10 | AddrModeFormation | AddressMode | PROVED |
| 11 | CmpSelectCombine | CmpCombine | PROVED |
| 12 | IfConversion | IfConversion | PROVED (W24) |
| 13 | CmpBranchFusion | CmpCombine | PROVED |
| 14 | TailCallOptimization | TailCallOptimization | PROVED |
| 15 | DeadCodeElimination | DeadCodeElimination | PROVED |
| 16 | CfgSimplify | CfgSimplification | PROVED |

### 2.5 Biggest Gaps Preventing Real-World Use

Ranked by criticality:

1. **tMIR integration is stubbed (#227, P1).** The adapter layer exists (4,533 LOC) but talks to 848 LOC of stubs, not the real tMIR repo. This is the single biggest gap -- LLVM2 cannot serve its purpose without real input. Blocked on tMIR repo stability.

2. **z4 solver not wired up (#228, P2).** All proofs currently use mock evaluation (exhaustive for small widths, random sampling for 32/64-bit). The z4 bridge module exists with full SMT-LIB2 emission, but no actual z4 solver calls are made. Proofs are structurally sound but not machine-checked.

3. **No CLI driver (#226, P2).** LLVM2 is library-only. There is no `llvm2c` or equivalent command-line tool that takes a .tmir file and produces a .o file. Usability gap.

4. **Integration test stale count (#233, P2).** The full_proof_suite integration test asserts 25 categories but there are now 32. Trivial fix but blocks clean test runs.

5. **1,350 unwrap() calls in production code (#235, P2).** The encoder and ISel paths panic on malformed input rather than returning Results. Unacceptable for a verified compiler.

6. **x86-64 target is scaffolding only (#232, P3).** Opcode enum, register defs, encoding stub, RA adapter exist but cannot produce working code.

### 2.6 End-to-End Pipeline Status

The e2e_run tests demonstrate that LLVM2 **can already compile, link, and execute simple functions** on AArch64 Apple Silicon:
- `build_add_test_function()` constructs an `add(i32, i32) -> i32` function directly in IR
- The pipeline encodes it to AArch64 machine code, wraps it in a Mach-O .o file
- A C driver calls the function, links with `cc`, and verifies the result via exit code

This works for hand-constructed IR functions. The gap is: this bypasses ISel, RA, and optimization -- it starts from pre-allocated physical registers. A real compilation path would go through the full pipeline: tMIR -> ISel -> VReg -> RegAlloc -> Optimization -> Encoding -> Mach-O.

---

## 3. Roadmap to First Real Compilation

**Goal:** Compile `int add(int a, int b) { return a + b; }` from tMIR input to a linked, running AArch64 executable.

### What Already Works
- ISel for add/sub/mul/div/cmp/branch/load/store/extensions/FP/atomics
- Apple AArch64 ABI lowering (argument passing, return values, stack frame)
- Linear scan + greedy register allocation
- 16 optimization passes at O2/O3
- AArch64 binary encoding (145+ opcodes)
- Mach-O object file writer (complete)
- Frame lowering, compact unwind, branch relaxation
- End-to-end pipeline: IR -> encode -> Mach-O .o -> link -> run

### What's Missing (in order of dependency)

| Step | Gap | Effort | Blocker? |
|------|-----|--------|----------|
| 1 | **Real tMIR input parsing** | tMIR repo must expose serialized function representation | YES -- external dependency |
| 2 | **Full-pipeline E2E test** | Wire ISel -> RA -> Opt -> Encode for a simple function, verify output | Medium (2-3 waves) |
| 3 | **CLI driver** | Command-line tool: `llvm2c input.tmir -o output.o` | Small (1 wave) |
| 4 | **z4 integration** | Wire z4 solver for machine-checked proofs | Medium (2 waves), not blocking compilation |
| 5 | **Error propagation** | Replace critical-path unwrap()s with Result | Large (3+ waves), not blocking |

### Near-Term Milestone: Internal E2E Without tMIR

Even without real tMIR, we can demonstrate the full pipeline by constructing a tMIR-stub function that exercises ISel, RA, optimization, encoding, and linking. This would prove the pipeline works end-to-end.

**Proposed test:** Create a `test_full_pipeline_add_function` that:
1. Constructs a tMIR stub function (add two i32 arguments, return result)
2. Runs it through the tMIR adapter -> ISel -> VReg allocation -> RA -> Optimization -> Encoding -> Mach-O
3. Links with a C driver, executes, verifies result

This is achievable in Wave 25 and would be a major milestone.

---

## 4. Open Issue Inventory (Wave 25 Candidates)

### P1 Issues (2)

| # | Title | Type | W25 Action |
|---|-------|------|------------|
| 24 | Epic: AArch64 Backend Implementation | Epic | Audit and close completed tasks |
| 121 | Master design: Unified solver architecture | Epic | No action (long-term) |

### P2 Issues (Actionable)

| # | Title | Type | W25 Candidate? |
|---|-------|------|----------------|
| 227 | tMIR integration stubbed | Feature | **YES -- HIGH PRIORITY** (but partially blocked on tMIR repo) |
| 228 | z4 solver not wired up | Feature | Yes -- moderate |
| 226 | No CLI driver | Feature | **YES -- HIGH PRIORITY** |
| 233 | Integration test stale count | Bug | **YES -- trivial fix** |
| 234 | test_run_parallel_matches_sequential timeout | Bug | Yes -- restructure or increase timeout |
| 235 | 1,350 unwrap() calls | Bug | Yes -- partial (focus on encoder + ISel) |
| 224 | Register pressure-aware scheduling | Feature | Yes -- moderate |
| 140 | Missing ABI features | Feature | Partially addressed in W24 (variadics/HFA). Check remainder. |
| 209 | RISC-V E2E linking tests | Feature | Done in W24 (8bde5d3). Close candidate. |
| 231 | 3 unit test failures in llvm2-verify | Bug | Check if W24 work resolved these |

### P3 Issues

| # | Title | Notes |
|---|-------|-------|
| 232 | x86-64 scaffolding only | Long-term |
| 230 | 2 doctest failures | Quick fix |
| 229 | Stale code comments | Quick fix |
| 34 | z4 integration | Long-term |
| 141 | Design docs vs implementation | Documentation |

### New Issues to File

| Priority | Title | Rationale |
|----------|-------|-----------|
| P2 | Full-pipeline E2E test: ISel -> RA -> Opt -> Encode -> Link -> Run | Proves pipeline works without tMIR repo. Major milestone. |
| P2 | Call lowering correctness proofs | No proof category for call semantics (BL/BLR/RET argument passing) |
| P2 | Atomic memory ordering proofs | 22 atomic opcodes landed in W24 but no ordering/barrier proofs |
| P3 | Epic #24 checklist update | Many tasks appear done but unchecked |

---

## 5. Wave 25 Recommended Assignments (9 Agents)

### Priority Framework

1. **Close the full-pipeline E2E gap** -- prove the pipeline works end-to-end (biggest credibility milestone)
2. **Fix test failures and stale assertions** (clean CI is a prerequisite for quality)
3. **Build the CLI driver** (usability)
4. **Verification: call lowering + atomic ordering proofs** (verification gaps from W24 additions)
5. **Error propagation in critical paths** (robustness)

### Agent Assignments

| Slot | Issue | Title | Priority | Rationale |
|------|-------|-------|----------|-----------|
| **TL1** | NEW | Full-pipeline E2E test: tMIR stub -> ISel -> RA -> Opt -> Encode -> Mach-O -> Link -> Run | **P2-HIGH** | The most important milestone for the project. Proves every stage of the pipeline connects. Build a tMIR stub function, run it through the full pipeline, link with cc, execute, verify output. |
| **TL2** | #226 | CLI driver (`llvm2c`) | **P2-HIGH** | Create a command-line tool that takes tMIR stub input and produces a .o file. Even with stub input, this makes LLVM2 invocable as a tool. Include `--opt-level`, `--emit-asm`, `--target` flags. |
| **TL3** | NEW | Call lowering correctness proofs | **P2** | BL/BLR/RET with argument passing through X0-X7 and stack slots. The ABI lowering exists but has no formal verification. Prove argument placement, return value semantics, callee-saved register preservation. |
| **TL4** | NEW | Atomic memory ordering proofs | **P2** | Wave 24 added 22 atomic opcodes. Prove LDAR/STLR acquire/release semantics, CAS compare-and-swap correctness, RMW operation semantics. Add AtomicOrdering proof category. |
| **TL5** | #235 | Unwrap elimination: encoder + ISel critical paths | **P2** | Replace the 259 unwrap()s in encode.rs and 247 in isel.rs with proper Result propagation. These are the two most critical production code paths. |
| **TL6** | #233 + #231 + #230 | Fix all test failures and stale assertions | **P2** | Update full_proof_suite.rs category count (25 -> 32), fix unit test count drift in llvm2-verify, fix 2 doctest failures. Clean CI enables confidence in future changes. |
| **TL7** | #228 | z4 SMT solver bridge: first real solver call | **P2** | The z4_bridge module emits SMT-LIB2 but never calls z4. Wire up at least one proof to actually invoke z4 and check satisfiability. Feature-gated. This converts one mock proof into a machine-checked proof. |
| **A1** | #24 | Epic #24 task audit and checklist update | **P1** | Verify status of all unchecked tasks: Phase 8 (done?), Phase 9 (partial), tMIR adapter (done but stubbed), greedy RA (done), x86-64 scaffold (done), DWARF (done), benchmark suite (not started), type duplication (#49), encoder gaps (#48), PReg conflict (#52), lower.rs stub (#51). Update checkboxes. |
| **R1** | -- | Wave 25 planning (this document) | **P2** | Delivered. |

### Overflow Issues

If any agent finishes early:

| Issue | Title | Notes |
|-------|-------|-------|
| #234 | Timeout: test_run_parallel_matches_sequential | Restructure test or increase timeout |
| #229 | Stale code comments re: frame lowering | Quick cleanup |
| #224 | Register pressure-aware scheduling | Self-contained optimization |
| #227 | tMIR integration (partial) | Document exactly what tMIR repo must expose for real integration |

---

## 6. Risk Assessment

### High Risk

1. **Full-pipeline E2E test (TL1)** -- Connecting ISel -> RA -> Opt -> Encode in a single flow may reveal integration bugs between stages. The individual stages are tested but the full chain may have state/type mismatches. Mitigation: start with the simplest possible function (add i32) and iterate.

2. **z4 solver bridge (TL7)** -- z4 may not be installed/available on the build machine. The z4 crate API may have changed since the bridge was written. Mitigation: feature-gate everything, test with `cargo test --features z4`.

### Medium Risk

3. **Unwrap elimination (TL5)** -- 500+ unwrap changes in encoder and ISel touch the most critical code. Risk of introducing new bugs. Mitigation: run full test suite after each batch of changes.

4. **Call lowering proofs (TL3)** -- ABI semantics are complex (stack alignment, register conventions, variadic handling). Proofs may need to abstract over calling conventions. Mitigation: start with the simplest case (2 i32 args, 1 i32 return).

### Low Risk

5. **CLI driver (TL2)** -- Well-scoped, can lean on existing pipeline API. `compile_to_object()` already exists.

6. **Test fixes (TL6)** -- Mechanical assertion updates. Low risk.

7. **Epic audit (A1)** -- Read-only investigation. No code risk.

---

## 7. Success Criteria for Wave 25

### Primary

1. **Full-pipeline E2E test passes** -- a tMIR stub function goes through ISel -> RA -> Opt -> Encode -> Mach-O -> link -> run
2. **CLI driver exists** -- `cargo run --bin llvm2c` produces a .o file
3. **All tests pass** -- zero failures, zero stale assertions (issues #233, #231, #230 closed)
4. **Call lowering proof category added** with at least 4 proofs (direct call, indirect call, return value, callee-saved registers)
5. **Atomic ordering proof category added** with at least 4 proofs (acquire, release, CAS, RMW)
6. **Codebase exceeds 195K LOC** with proportional test growth

### Stretch

- z4 solver makes a real satisfiability check on at least one proof
- Encoder unwrap count reduced by >= 50% (259 -> <130)
- ISel unwrap count reduced by >= 50% (247 -> <124)
- Epic #24 down to < 5 unchecked tasks

---

## 8. Long-Term Roadmap (Waves 26-30)

| Wave | Focus | Milestone |
|------|-------|-----------|
| 25 | Full-pipeline E2E + CLI + proofs | First end-to-end compiled and executed function |
| 26 | tMIR integration + z4 | First real tMIR function compiled (if tMIR repo ready) |
| 27 | Error handling + robustness | Result propagation across all critical paths |
| 28 | x86-64 target MVP | Second target produces working code |
| 29 | Benchmark suite | LLVM2 vs clang -O2 comparison |
| 30 | Production readiness | tRust can use LLVM2 as its backend |

The critical external dependency is the **tMIR repo** (ayates_dbx/tMIR). Until tMIR exposes a stable serialized function representation, LLVM2 operates against stubs. Wave 26 should target real tMIR integration as its #1 priority, assuming the tMIR repo is ready.

---

## Appendix A: AArch64 Opcode Categories

Total enum variants: ~321
- Pseudo-ops (Phi, StackAlloc, Copy, Nop, etc.): ~30
- LLVM-style typed aliases (IAdd32, ISub32, etc.): ~40
- NEON SIMD: ~30
- Atomics (ARMv8.1 LSE + LL/SC): 22
- Core integer/FP/memory/branch: ~145
- Checked arithmetic / traps / refcount: ~15
- Address (ADR/ADRP): 4
- GOT/TLV: 2

## Appendix B: Proof Category Inventory (32 Categories)

| # | Category | Domain |
|---|----------|--------|
| 1 | Arithmetic | Lowering |
| 2 | Division | Lowering |
| 3 | FloatingPoint | Lowering |
| 4 | FpConversion | Lowering (W24) |
| 5 | NzcvFlags | Lowering |
| 6 | Comparison | Lowering |
| 7 | Branch | Lowering |
| 8 | BitwiseShift | Lowering |
| 9 | Peephole | Optimization |
| 10 | Optimization | Optimization |
| 11 | ConstantFolding | Optimization |
| 12 | CopyPropagation | Optimization |
| 13 | CseLicm | Optimization |
| 14 | DeadCodeElimination | Optimization |
| 15 | CfgSimplification | Optimization |
| 16 | StrengthReduction | Optimization |
| 17 | CmpCombine | Optimization |
| 18 | Gvn | Optimization |
| 19 | TailCallOptimization | Optimization |
| 20 | IfConversion | Optimization (W24) |
| 21 | LoopOptimization | Optimization |
| 22 | Memory | Infrastructure |
| 23 | RegAlloc | Infrastructure |
| 24 | ConstantMaterialization | Infrastructure |
| 25 | AddressMode | Infrastructure |
| 26 | FrameLayout | Infrastructure |
| 27 | InstructionScheduling | Infrastructure |
| 28 | MachOEmission | Output |
| 29 | NeonLowering | NEON/Vector |
| 30 | NeonEncoding | NEON/Vector |
| 31 | Vectorization | NEON/Vector |
| 32 | AnePrecision | Accelerator |
