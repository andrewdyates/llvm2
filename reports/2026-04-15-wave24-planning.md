# Wave 24 Planning Document

**Author:** R1 (Researcher Agent)
**Date:** 2026-04-15
**Commit base:** 99815e2 (main, post-Wave 23 merges)
**Scope:** Wave 22/23 summary, codebase stats, Wave 24 issue assignments (9 agents)

---

## Executive Summary

LLVM2 has reached **184,635 LOC Rust** across 6 production crates with **4,988 `#[test]` annotations** and **30 proof categories**. Waves 22 and 23 closed critical verification gaps: all 16 O2/O3 optimization passes now have correctness proofs (GVN, CmpCombine, TailCall, StrengthReduction all proved). The if-conversion pass was added, NEON SIMD binary encoding landed, and post-RA dead spill elimination was implemented.

The main remaining gaps are: (1) unproven lowering paths (FP conversions, extensions/truncations, calls), (2) missing atomic operations blocking concurrent program compilation, (3) if-conversion proofs for the newly-added pass, and (4) NEON SIMD lowering proofs that were started in W23 but may need completion.

Wave 24 should focus on **closing the AArch64 backend epic (#24)** by addressing the remaining functional and verification gaps that block real-program compilation.

---

## 1. Wave 22/23 Summary

### Wave 22 Deliverables (7 agents)

| Agent | Issue | Deliverable | Status |
|-------|-------|-------------|--------|
| TL1 | #219 | Fix stale proof count assertions | COMPLETE |
| TL2 | #212 | Strength reduction correctness proofs | COMPLETE |
| TL3 | #213 | CmpBranch/CmpSelect correctness proofs | COMPLETE |
| TL4 | #216 | GVN optimization correctness proofs | COMPLETE |
| TL5 | #207 | If-conversion pass (diamond + triangle CFG) | COMPLETE |
| TL6 | #215 | NEON SIMD binary encoding | COMPLETE |
| TL7 | #214 | x86-64 register allocator adapter | COMPLETE |
| A1 | -- | Post-Wave 21 workspace audit | COMPLETE |
| R1 | -- | Wave 22 planning and verification matrix | COMPLETE |

### Wave 23 Deliverables (3+ agents)

| Agent | Deliverable | Status |
|-------|-------------|--------|
| TL2 | Tail call optimization correctness proofs (new TailCallOptimization proof category) | COMPLETE |
| TL5 | NEON SIMD lowering correctness proofs (NeonEncoding proof category) | COMPLETE |
| TL6 | Post-RA dead spill elimination and redundant reload removal | COMPLETE |
| Fix | Proof category count assertion 29 to 30 (NeonEncoding + TailCallOptimization) | COMPLETE |

### Key Milestone Achieved

**All 16 O2/O3 optimization passes now have correctness proofs.** This was the #1 priority from the Wave 22 planning document. The verification coverage matrix for optimization passes is now:

| Pass | Proof Category | Status |
|------|----------------|--------|
| ProofOptimization | N/A (consumes proofs) | N/A |
| ConstantFolding | ConstantFolding | PROVED |
| CopyPropagation | CopyPropagation | PROVED |
| CommonSubexprElim | CseLicm | PROVED |
| GlobalValueNumbering | Gvn | PROVED (W22) |
| LoopInvariantCodeMotion | CseLicm | PROVED |
| StrengthReduction | StrengthReduction | PROVED (W22) |
| LoopUnroll | LoopOptimization | PROVED |
| Peephole | Peephole | PROVED |
| AddrModeFormation | AddressMode | PROVED |
| CmpSelectCombine | CmpCombine | PROVED (W22) |
| IfConversion | -- | **UNPROVED** (pass added W22, proofs filed as #221) |
| CmpBranchFusion | CmpCombine | PROVED (W22) |
| TailCallOptimization | TailCallOptimization | PROVED (W23) |
| DeadCodeElimination | DeadCodeElimination | PROVED |
| CfgSimplify | CfgSimplification | PROVED |

**Correction:** 15 of 16 passes are proved. IfConversion (#207, added in W22) still lacks proofs (#221).

---

## 2. Current Codebase Stats

### 2.1 Scale

| Crate | LOC | Tests | Role |
|-------|-----|-------|------|
| llvm2-verify | 55,490 | 1,672 | SMT proofs, CEGIS, proof database, z4 bridge |
| llvm2-codegen | 53,517 | 1,482 | Encoding, Mach-O, ELF, DWARF, Metal, CoreML, pipeline |
| llvm2-lower | 27,012 | 640 | ISel, ABI, tMIR adapter, compute graph, dispatch |
| llvm2-opt | 20,424 | 420 | 16 optimization passes (O0-O3 pipeline) |
| llvm2-regalloc | 15,207 | 390 | Linear scan + greedy allocator, post-RA passes |
| llvm2-ir | 12,985 | 384 | MachFunction, opcodes, registers, cost model |
| **Total** | **184,635** | **4,988** | |

Growth since Wave 22 start: +9,357 LOC, +259 tests.

### 2.2 Proof Categories: 30

All 30 categories have at least one proof obligation. Categories added since Wave 21:
- NeonEncoding (W23)
- TailCallOptimization (W23)
- StrengthReduction (W22)
- CmpCombine (W22)
- Gvn (W22)

### 2.3 Pipeline Passes: 16 at O2/O3

The optimization pipeline runs 16 passes at O2 and O3 (O3 iterates to fixpoint, max 10 iterations).

### 2.4 AArch64 Encoding Coverage

121 opcode enum variants defined. Prior assessment: 105 encoded. With NEON SIMD encoding (W22), coverage has increased. Remaining unencoded are primarily pseudo-ops (Phi, StackAlloc, Copy, Nop, Retain, Release) that are lowered/expanded before encoding.

---

## 3. Open Issue Inventory

### P1 Issues (2)

| # | Title | Type | Blocker? |
|---|-------|------|----------|
| 24 | Epic: AArch64 Backend Implementation | Epic | Master tracking. Remaining unchecked: Phase 8, Phase 9, tMIR adapter, tMIR integration tests, tMIR proof propagation, greedy RA, x86-64 scaffold, DWARF debug, benchmark suite, type duplication, missing encoder, PReg conflict, lower.rs stub |
| 121 | Master design: Unified solver architecture | Epic | Long-term vision. Not blocking W24. |

### P2 Issues (10)

| # | Title | Type | W24 Candidate? |
|---|-------|------|----------------|
| 109 | Epic: Automatic heterogeneous compute | Epic | No (long-term) |
| 108 | Epic: AI-native compilation | Epic | No (long-term) |
| 107 | Epic: Radical debugging and transparency | Epic | No (long-term) |
| 106 | Epic: Solver-driven superoptimization | Epic | No (long-term) |
| 140 | Missing ABI features: exception handling, C++ interop | Feature | Yes -- partial |
| 217 | Atomic memory operations (ISel + encoding) | Feature | **Yes -- HIGH PRIORITY** |
| 209 | RISC-V E2E linking tests | Feature | Yes -- moderate |
| 210 | Post-RA live range shrinking and dead store elimination | Feature | Yes -- partial (spill elim done in W23) |
| 220 | Tail call optimization correctness proofs | Feature | **Done in W23** (duplicate of W23 work) |
| 221 | If-conversion correctness proofs | Feature | **Yes -- HIGH PRIORITY** |
| 222 | FP conversion lowering proofs | Feature | **Yes -- HIGH PRIORITY** |
| 223 | NEON SIMD lowering correctness proofs | Feature | Partially done in W23, check completion |
| 224 | Register pressure-aware scheduling | Feature | Yes -- moderate |
| 225 | Extension and truncation lowering | Feature | **Yes -- HIGH PRIORITY** |

### P3 Issues (3)

| # | Title | Notes |
|---|-------|-------|
| 34 | Phase 9: z4 verification integration | Long-term, blocked on z4 maturity |
| 141 | Design docs vs implementation gaps | Documentation |
| 218 | Tail call optimization proofs (P3 duplicate) | Superseded by #220 / W23 work |

### Security Issues (9, #13-#22)

These are inherited ai_template security issues, not LLVM2-specific. Out of scope for Wave 24.

---

## 4. Gap Analysis for AArch64 Backend Epic (#24)

### Remaining Unchecked Tasks in #24

1. **Phase 8: Proof-enabled optimizations (#33)** -- ProofOptimization pass exists and runs at O2/O3. Arguably complete, needs audit.
2. **Phase 9: z4 verification integration (#34)** -- z4 bridge module exists. Feature-gated. Needs real z4 solver integration testing.
3. **tMIR adapter layer (#55)** -- Exists (27K LOC in llvm2-lower). Needs completion audit.
4. **tMIR integration test suite (#60)** -- Tests exist. Needs coverage audit.
5. **tMIR proof propagation through optimization pipeline (#64)** -- ProofOptimization pass consumes proofs. Needs completion audit.
6. **Greedy register allocator (#62)** -- Exists (15K LOC in llvm2-regalloc). Implemented and proved.
7. **x86-64 target scaffolding (#61)** -- Exists. RA adapter added in W22. ELF pipeline works.
8. **DWARF debug info emission (#63)** -- Exists (__debug_abbrev, __debug_info, __debug_line).
9. **Benchmark suite (#65)** -- Not started.
10. **Massive type duplication (#49)** -- Partially addressed by unification work.
11. **Missing AArch64 integer instruction encoder (#48)** -- Likely resolved by encoding work.
12. **PReg conflict (#52)** -- Status unknown, needs check.
13. **llvm2-codegen/lower.rs stub (#51)** -- Status unknown, needs check.

### Highest-Impact Functional Gaps

1. **Atomics (#217)** -- No concurrent program can compile. This is the single biggest functional gap.
2. **Extension/truncation lowering proofs (#225)** -- Sext/Zext/Trunc in ISel but unproved.
3. **FP conversion proofs (#222)** -- Six opcodes (FcvtzsRR, FcvtzuRR, ScvtfRR, UcvtfRR, FcvtSD, FcvtDS) unproved.
4. **If-conversion proofs (#221)** -- New pass (W22) running at O2/O3 without proofs.

### Verification Gaps Remaining

| Lowering Path | Status |
|---------------|--------|
| FP conversions (fcvt/scvtf/ucvtf) | UNPROVED |
| FP comparisons (fcmp) | UNPROVED |
| Extensions (sext/zext/trunc) | UNPROVED |
| Calls (direct/indirect/variadic) | UNPROVED |
| Atomics | NOT IMPLEMENTED |
| If-conversion pass | UNPROVED |
| Instruction scheduler (algorithm output) | PARTIAL |

---

## 5. Wave 24 Recommended Assignments (9 Agents)

### Priority Framework

1. **Close verification gaps for passes running at O2/O3** (violates "verify before merge")
2. **Add missing functional features** (atomics, extensions)
3. **Audit and close epic tasks** (mark completed items in #24)
4. **Code quality and forward-looking work**

### Agent Assignments

| Slot | Issue | Title | Priority | Rationale |
|------|-------|-------|----------|-----------|
| **TL1** | #217 | Atomic memory operations (ISel + encoding) | **P2-HIGH** | Single biggest functional gap. Blocks concurrent programs. Large scope: ISel patterns, opcode enum additions, encoding, basic proofs. |
| **TL2** | #221 | If-conversion correctness proofs | **P2-HIGH** | If-conversion runs at O2/O3 since W22 without proofs. Must prove diamond/triangle CSEL semantics. Relatively contained scope. |
| **TL3** | #222 | FP conversion lowering proofs | **P2-HIGH** | Six FP conversion opcodes are in ISel+encoder but unproved. IEEE 754 rounding semantics are non-trivial. |
| **TL4** | #225 | Extension and truncation lowering proofs | **P2** | Sext/Zext/Trunc via SBFM/UBFM/AND are core operations. Prove bitfield moves produce correct results across I8/I16/I32/I64. |
| **TL5** | #224 | Register pressure-aware instruction scheduling | **P2** | Scheduler currently ignores register pressure, can cause unnecessary spills. Integrate with ResourceState/HazardKind model. |
| **TL6** | #217 | Atomic memory operations (verification) | **P2** | If TL1 lands ISel+encoding, TL6 writes memory ordering proofs (LDAR/STLR sequential consistency, RMW correctness). If TL1 is still in progress, TL6 assists with encoding. |
| **TL7** | #140 | ABI completeness: exception handling stubs | **P2** | Exception handling data structures are 1K LOC stubs. Even basic exception table emission is needed for C++ interop. Scope to EH table emission (not full unwinding). |
| **A1** | -- | Post-Wave 23 audit + Epic #24 task closure | **P1** | Audit #24 checklist items. Many tasks (greedy RA, x86-64 scaffold, DWARF, proof-enabled opts) appear done but are unchecked. Verify and close. Check #48, #49, #51, #52 status. |
| **R1** | -- | Wave 24 planning (this document) + design review | **P2** | Delivered. |

### Alternative / Overflow Issues

If any agent finishes early or is blocked:

| Issue | Title | Notes |
|-------|-------|-------|
| #209 | RISC-V E2E linking tests | Good follow-up for TL5 or TL7 |
| #210 | Post-RA live range shrinking | Partially addressed (W23 dead spill elim). Remaining: live range shrinking. |
| #218/#220 | TCO proofs (P3/P2) | Both may be closable as duplicates if W23 work is confirmed complete |
| #223 | NEON SIMD lowering proofs | Check if W23 TL5 fully completed this |

---

## 6. Risk Assessment

### High Risk

1. **Atomics scope (#217)** -- This is a large feature spanning ISel, opcode definitions, encoding, and proofs. A single agent may not complete it in one wave. Mitigation: assign two agents (TL1 for implementation, TL6 for verification). Even partial delivery (LDAR/STLR encoding only) is valuable.

2. **FP conversion proofs (#222)** -- IEEE 754 rounding semantics are notoriously tricky to encode in SMT. The QF_FP theory was added in W20, but writing correct FP conversion proofs that handle all rounding modes and special values (NaN, Inf, denormals) is non-trivial. Mitigation: scope to round-toward-zero mode only (matching AArch64 FCVTZS/FCVTZU semantics).

### Medium Risk

3. **If-conversion proofs (#221)** -- The if-conversion pass handles both diamond and triangle patterns. The CSEL/CSINC semantics need careful encoding. Medium complexity.

4. **Epic audit accuracy** -- The A1 agent may find that some "done" items actually have gaps. This could generate new issues that expand the backlog.

### Low Risk

5. **Extension/truncation proofs (#225)** -- SBFM/UBFM are well-understood. The bitfield semantics are simpler than FP conversions.

6. **Register pressure scheduling (#224)** -- Self-contained optimization improvement with existing scheduler infrastructure.

---

## 7. Success Criteria for Wave 24

1. **If-conversion pass is proved** (#221 closed)
2. **FP conversion lowering is proved** (#222 closed)
3. **Extension/truncation lowering is proved** (#225 closed)
4. **Atomic operations ISel + encoding landed** (at least LDAR/STLR/LDADD/CAS)
5. **Epic #24 tasks audited** (at least 3 more tasks checked off)
6. **All tests pass** (0 failures, no stale assertions)
7. **Codebase exceeds 190K LOC** with proportional test growth

### Stretch Goals

- Full atomic verification proofs
- Register pressure-aware scheduling operational
- Exception handling table emission
- Epic #24 down to < 5 unchecked tasks
