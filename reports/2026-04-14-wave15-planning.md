# Wave 15 Issue Closure and Wave 16 Planning

**Author:** R1 (Researcher Agent)
**Date:** 2026-04-14
**Commit base:** 27c4f94 (main)
**Scope:** Close 8 Wave-14-reviewed issues, update #140, file new issues, plan Wave 16

---

## Executive Summary

Closed 8 issues that were marked close-ready after Wave 14 review. Updated #140 (ABI features) with remaining work scope. Filed 7 new issues covering the verification and encoding gaps identified during Wave 14/15 analysis. The project's next critical path runs through register allocation proofs (#181) and bitwise/shift lowering proofs (#180).

**Snapshot:** 134,532 LOC Rust, 3,546 tests, 369 proof functions, 6 production crates.

---

## 1. Issues Closed This Wave

| # | Title | Key Evidence |
|---|-------|-------------|
| #178 | [codegen] pipeline.rs has 843 LOC with zero unit tests | 2,649 LOC, 69 unit tests (was 843 LOC, 0 tests). 18 classify_def_use tests, 3 ir_to_regalloc tests. |
| #177 | [lower] compute_graph should use ProfitabilityAnalyzer | ProfitabilityAnalyzer from cost_model.rs integrated into ComputeGraph. 56 tests, 7+ integration tests. |
| #176 | [verify] GPU/ANE synthesis candidates lack SMT verification | unified_synthesis.rs (5,130 LOC, 141 tests) calls gpu_semantics + ane_semantics encode functions. 50+ verification tests. |
| #143 | [verify] Cross-target dispatch plan verification | verify_dispatch_plan_properties() verifies 5 properties. 41 dispatch tests + 10 e2e integration tests. |
| #118 | NEON/SIMD auto-vectorization | 4,634 LOC, 132 tests, 78 proof functions across vectorize.rs, neon_semantics.rs, vectorization_proofs.rs, neon_lowering_proofs.rs. |
| #33  | Phase 8: Proof-enabled optimizations | All 6 deliverables in proof_opts.rs (2,020 LOC, 45 tests): NoOverflow, InBounds, NotNull, ValidBorrow, PositiveRefCount. |
| #119 | GPU targeting via Metal IR emission | metal_emitter.rs (2,222 LOC, 39 tests). MSL kernel emission for parallel_map, parallel_reduce, matmul. |
| #120 | Neural Engine targeting via CoreML lowering | coreml_emitter.rs (1,757 LOC, 28 tests). MIL emission for GEMM, Conv2D, activations. Fusion support. |

**Total closed: 8 issues**

---

## 2. Issues Updated (Not Closed)

### #140 -- Missing ABI features (reduced scope)

4 of 6 original items complete (HFA, large structs, variadic, LSDA). Remaining:
- SIMD/vector type arguments in ABI parameter classification
- DWARF CFI validation against real libunwind

---

## 3. New Issues Filed

| # | Title | Priority | Category |
|---|-------|----------|----------|
| #180 | [verify] Bitwise/shift lowering correctness proofs | P2 | Verification gap |
| #181 | [verify] Register allocation correctness proofs | P1 | Critical verification gap |
| #182 | [codegen] x86-64 instruction encoding implementation | P2 | New target |
| #183 | [verify] Address mode formation correctness proofs | P2 | Verification gap |
| #184 | [verify] Constant materialization correctness proofs | P2 | Verification gap |
| #185 | [verify] Frame index elimination correctness proofs | P2 | Verification gap |
| #186 | [verify] End-to-end verification pipeline | P2 | Infrastructure |

**Total new issues: 7**

---

## 4. Open Issue Inventory (Post-Wave 15)

### Implementation Issues (actionable)

| # | Title | Priority | Status |
|---|-------|----------|--------|
| #181 | Register allocation correctness proofs | P1 | NEW |
| #180 | Bitwise/shift lowering proofs | P2 | NEW |
| #182 | x86-64 instruction encoding | P2 | NEW |
| #183 | Address mode formation proofs | P2 | NEW |
| #184 | Constant materialization proofs | P2 | NEW |
| #185 | Frame index elimination proofs | P2 | NEW |
| #186 | End-to-end verification pipeline | P2 | NEW |
| #140 | ABI features (SIMD args, libunwind) | P2 | Reduced scope |
| #103 | RISC-V target definitions | P3 | Not started |

### z4 Integration (in-progress, external dependency)

| # | Title | Priority | Status |
|---|-------|----------|--------|
| #122 | QF_ABV: Array theory for GPU/memory | P1 | In progress |
| #123 | QF_FP: Floating-point theory | P2 | In progress |
| #124 | Bounded quantifiers for array-range proofs | P2 | In progress |
| #34  | Phase 9: z4 verification integration | P3 | Depends on #122/#123/#124 |

### Blocked/External

| # | Title | Priority | Status |
|---|-------|----------|--------|
| #125 | tMIR proof annotations | P1 | Blocked on external tMIR repo |
| #121 | Unified solver architecture (epic) | P1 | Depends on z4 integration |

### Epics (tracking only)

| # | Title | Status |
|---|-------|--------|
| #24  | AArch64 Backend Implementation | Most tasks complete |
| #109 | Automatic heterogeneous compute | #118/#119/#120 closed this wave |
| #106 | Solver-driven superoptimization | Depends on z4 |
| #107 | Radical debugging and transparency | Partially implemented |
| #108 | AI-native compilation | Partially implemented |

### Audit/Documentation

| # | Title | Priority |
|---|-------|----------|
| #179 | Wave 14 A1 audit | P2 |
| #141 | Design docs vs implementation gaps | P3 |

---

## 5. Recommended Wave 16 Assignments

### Techlead Slots (5-7 agents)

| Slot | Issue | Deliverable | Effort |
|------|-------|-------------|--------|
| TL1 | #181 | Register allocation correctness proofs (Phase 1: liveness) | Medium |
| TL2 | #180 | Bitwise/shift lowering proofs (AND/OR/XOR/LSL/LSR/ASR) | Small |
| TL3 | #184 | Constant materialization proofs (MOVZ/MOVK sequences) | Small |
| TL4 | #183 | Address mode formation proofs (base+offset, base+reg) | Small |
| TL5 | #185 | Frame index elimination proofs (stack layout, prologue/epilogue) | Medium |
| TL6 | #186 | End-to-end verification pipeline (verify_function API) | Medium |
| TL7 | #140 | ABI: SIMD/vector argument passing in abi.rs | Small |

### Alternative: If fewer slots available (5 agents)

| Slot | Issue | Rationale |
|------|-------|-----------|
| TL1 | #181 | Highest priority (P1), largest gap |
| TL2 | #180 | Quick win, completes lowering proof coverage |
| TL3 | #184 + #183 | Combined: both are small constant/address proofs |
| TL4 | #185 | Frame correctness is safety-critical |
| TL5 | #186 | Infrastructure enables automated verification |

### Researcher Slot (1 agent)

| Slot | Task |
|------|------|
| R1 | Design doc for x86-64 backend roadmap (#182). Research Intel SDM encoding tables, plan phased implementation. Review Wave 16 agent output for architectural coherence. |

### Deferred to Wave 17+

| Issue | Reason |
|-------|--------|
| #182 (x86-64 encoding) | Large effort, AArch64 is primary target. File after proof gaps are closed. |
| #103 (RISC-V) | P3, blocked on x86-64 completion first. |
| #122/#123/#124 (z4) | External dependency, in-progress independently. |

---

## 6. Progress Toward Completion Milestones

### Milestone: Complete AArch64 Lowering Proofs
**Status: ~85% complete**

| Category | Proofs | Status |
|----------|--------|--------|
| Integer arithmetic (add/sub/mul/neg/div) | 20 | DONE |
| Floating-point (add/sub/mul/neg) | 8 | DONE |
| NZCV flags | 4 | DONE |
| Comparisons | 20 | DONE |
| Conditional branches | 20 | DONE |
| Load/store lowering | 10 | DONE |
| **Bitwise/shift lowering** | **0** | **GAP (#180)** |
| **Constant materialization** | **0** | **GAP (#184)** |
| **Address mode formation** | **0** | **GAP (#183)** |

After #180 and #184: ~95% coverage of instruction categories.

### Milestone: Complete Optimization Proofs
**Status: ~90% complete**

| Category | Proofs | Status |
|----------|--------|--------|
| Constant folding | 34 | DONE |
| Peephole identities | 27 | DONE |
| Copy propagation | 15 | DONE |
| CSE/LICM | 28 | DONE |
| DCE | 11 | DONE |
| CFG simplification | 16 | DONE |
| Composed optimizations | 14 | DONE |
| NEON/vectorization | 78 | DONE |
| **Address mode formation** | **0** | **GAP (#183)** |

### Milestone: Register Allocation Verification
**Status: 0% -- Critical gap**

No proofs exist. #181 starts this work. Full regalloc verification is a multi-wave effort.

### Milestone: x86-64 Target
**Status: Scaffolding only**

- Register definitions: DONE (1,166 LOC)
- Opcode enum: DONE (513 LOC)
- Instruction selection: DONE (1,893 LOC, 30+ tests)
- Binary encoding: STUB (312 LOC, returns NotImplemented)
- Proofs: None

### Milestone: z4 Integration
**Status: In progress (external)**

When z4 ships QF_ABV (#122), all 369 existing proofs can switch from mock evaluation to real SMT solving. The z4_bridge module (2,182 LOC, 55 tests) is ready.

---

## 7. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Regalloc proofs are hard (CompCert took years) | High | Start with validated-translation approach: prove the _result_ is valid, not the algorithm. Phase 1 focuses on liveness soundness. |
| Bitwise proofs are trivially correct but easy to get wrong in encoding | Low | Follow established pattern from proof_iadd_i32. Straightforward extension. |
| x86-64 encoding is a large effort | Medium | Defer to Wave 17+. AArch64 is production target. |
| z4 dependency blocks real SMT verification | Medium | Mock evaluator is thorough for small widths. When z4 ships, existing proofs activate. |

---

## 8. Wave 15 vs Wave 14 Summary

| Metric | Wave 14 End | Wave 15 Actions | Net Change |
|--------|-------------|-----------------|------------|
| Open implementation issues | 9 needs-review | 8 closed, 7 filed | -1 (net) |
| Proof functions | 369 | (no code changes) | 369 |
| Total LOC | 134,532 | (no code changes) | 134,532 |
| Total tests | 3,546 | (no code changes) | 3,546 |

Wave 15 R1 was a planning-only wave (issue management + filing). Wave 16 techleads will produce the next code+proof batch.

---

*End of report.*
