# Wave 17 Issue Triage and Wave 18 Planning

**Author:** R1 (Researcher Agent)
**Date:** 2026-04-14
**Commit base:** c1b9ef6 (main)
**Scope:** Close completed issues, update in-progress work, file new issues, plan Wave 18

---

## Executive Summary

Closed 4 issues completed during Waves 15-16 but not yet closed. Updated 5 in-progress issues with current status. Filed 5 new issues for Wave 18 priorities. The project is at 141,325 LOC, 3,762 tests, and 395 proof functions across 6 production crates.

**Key finding:** Waves 15-16 accomplished significant work that was never reflected in issue state. Three substantial deliverables (bitwise proofs, regalloc proofs, x86-64 encoding) were complete but issues remained open.

**Snapshot:** 141,325 LOC Rust, 3,762 tests, 395 proof functions, 6 production crates + 4 stubs.

---

## 1. Issues Closed This Wave

| # | Title | Evidence | Closed By |
|---|-------|----------|-----------|
| #179 | [Audit] Wave 14 A1: Post-Wave 13 workspace validation | Superseded by W15/W16 audits. 3,762 tests, 0 failures. | Audit superseded |
| #180 | [verify] Bitwise/shift lowering correctness proofs | Commit 85ed210: 14 proof obligations, 36 new tests. AND/OR/XOR/NOT/SHL/LSR/ASR for I32/I64. | Wave 15 |
| #181 | [verify] Register allocation correctness proofs (Phase 1) | Commit e497e7b: 16 proof obligations, 7 correctness properties. Phase 1 scope complete. | Wave 15 |
| #182 | [codegen] x86-64 instruction encoding (Phase 1+2) | Commit 4442dac: 312 LOC stub -> 1,774 LOC. 20+ encoding functions, REX/ModR/M/SIB. | Wave 15 |

**Total closed: 4 issues**

---

## 2. Issues Updated (Not Closed)

| # | Title | Status |
|---|-------|--------|
| #103 | RISC-V target definitions | Registers + opcodes done (1,604 LOC). Remaining: target.rs wiring, encoding, ISel. |
| #187 | 22 AArch64 opcodes unencoded | Wave 17 TL actively working. P1 priority. |
| #188 | ELF writer doctest failure | Wave 17 TL actively working. |
| #186 | E2E verification pipeline | Wave 17 TL actively working. |
| #184 | Constant materialization proofs | Wave 17 TL actively working. |

---

## 3. New Issues Filed (Wave 18 Backlog)

| # | Title | Priority | Category |
|---|-------|----------|----------|
| #191 | [codegen] x86-64 E2E compilation pipeline | P2 | Pipeline integration |
| #192 | [verify] Register allocation Phase 2: semantic preservation | P2 | Verification (depends on z4 #122) |
| #193 | [codegen] RISC-V target.rs wiring and encoding stub | P3 | New target |
| #194 | [codegen] Instruction scheduling Phase 2: dependency tracking | P2 | Correctness gap |
| #195 | [verify] Instruction scheduling correctness proof | P2 | Verification gap |

**Total new issues: 5**

---

## 4. Open Issue Inventory (Post-Wave 17 Triage)

### Active (Wave 17 in progress)

| # | Title | Priority | Agent |
|---|-------|----------|-------|
| #187 | 22 AArch64 opcodes unencoded | P1 | Wave 17 TL |
| #188 | ELF writer doctest failure | P2 | Wave 17 TL |
| #186 | E2E verification pipeline | P2 | Wave 17 TL |
| #184 | Constant materialization proofs | P2 | Wave 17 TL |
| #183 | Address mode formation proofs | P2 | Wave 17 TL |

### Ready for Wave 18

| # | Title | Priority | Effort |
|---|-------|----------|--------|
| #191 | x86-64 E2E compilation pipeline | P2 | Large |
| #194 | Instruction scheduling Phase 2 (dependency tracking) | P2 | Medium |
| #195 | Instruction scheduling correctness proof | P2 | Medium |
| #185 | Frame index elimination proofs | P2 | Small |
| #140 | ABI features (SIMD args, libunwind) | P2 | Medium |
| #192 | Register allocation Phase 2 proofs | P2 | Large (z4 dependent) |
| #193 | RISC-V wiring + encoding stub | P3 | Medium |

### Blocked/External

| # | Title | Priority | Blocker |
|---|-------|----------|---------|
| #125 | tMIR proof annotations | P1 | External tMIR repo |
| #122 | z4 QF_ABV array theory | P1 | External z4 repo |
| #123 | z4 QF_FP floating-point theory | P2 | External z4 repo |
| #124 | z4 bounded quantifiers | P2 | External z4 repo |
| #121 | Unified solver architecture (epic) | P1 | Depends on z4 #122/#123/#124 |

### Documentation/Audit (Low Priority)

| # | Title | Priority |
|---|-------|----------|
| #141 | Design docs vs implementation gaps | P3 |
| #189 | tmir-semantics stub dead code | P3 |
| #190 | lower.rs overlaps with isel.rs | P3 |
| #103 | RISC-V target definitions (reduced scope) | P3 |

### Epics (Tracking Only)

| # | Title | Status |
|---|-------|--------|
| #24  | AArch64 Backend Implementation | ~90% complete |
| #109 | Automatic heterogeneous compute | Dependencies closed |
| #106 | Solver-driven superoptimization | Depends on z4 |
| #107 | Radical debugging/transparency | Partially implemented |
| #108 | AI-native compilation | Partially implemented |

### Security (Inherited from template)

Issues #5-#22: Security findings from ai_template. These are template-inherited and not actionable by LLVM2. No changes recommended.

---

## 5. Recommended Wave 18 Assignments

### Priority 1: Complete Wave 17 leftovers

Any Wave 17 issues not fully completed should be first priority for Wave 18 TL agents. Expected carryover:
- Remaining unencoded opcodes from #187 (if not all 22 are done)
- Verification pipeline #186 (if verify_function() not complete)

### Techlead Slots (5-7 agents)

| Slot | Issue | Deliverable | Effort | Rationale |
|------|-------|-------------|--------|-----------|
| TL1 | #194 | Instruction scheduling dependency tracking | Medium | Correctness gap: scheduler can misorder dependent instructions |
| TL2 | #191 | x86-64 E2E pipeline (Phase 1: pipeline wiring) | Medium | Enables second target, validates architecture generality |
| TL3 | #185 | Frame index elimination proofs | Small | Stack correctness is safety-critical |
| TL4 | #195 | Instruction scheduling correctness proofs | Medium | Verify the scheduling pass from TL1 |
| TL5 | #140 | ABI: SIMD vector argument passing + libunwind test | Medium | Needed for any code using SIMD structs |
| TL6 | W17 carryover | Complete any unfinished Wave 17 issues | Variable | Clean up before new work |
| TL7 | #193 | RISC-V target.rs wiring + R-type encoding | Small | Low-hanging fruit, unblocks future RISC-V work |

### Researcher Slot (1 agent)

| Slot | Task |
|------|------|
| R1 | Post-Wave 17 triage. Architecture review of scheduling pass (correctness of current implementation). Review Wave 18 agent output. |

### Auditor Slot (1 agent)

| Slot | Task |
|------|------|
| A1 | Post-Wave 17 workspace validation. Full cargo check + cargo test audit. |

---

## 6. Progress Toward Completion Milestones

### Milestone: Complete AArch64 Lowering Proofs
**Status: ~95% complete** (up from ~85% pre-Wave 15)

| Category | Proofs | Status |
|----------|--------|--------|
| Integer arithmetic (add/sub/mul/neg/div) | 20 | DONE |
| Floating-point (add/sub/mul/neg) | 8 | DONE |
| NZCV flags | 4 | DONE |
| Comparisons | 20 | DONE |
| Conditional branches | 20 | DONE |
| Load/store lowering | 10 | DONE |
| Bitwise/shift lowering | 14 | DONE (Wave 15, #180 closed) |
| **Constant materialization** | **0** | **GAP (#184, Wave 17 in progress)** |
| **Address mode formation** | **0** | **GAP (#183, Wave 17 in progress)** |

### Milestone: Register Allocation Verification
**Status: Phase 1 complete** (up from 0%)

- Phase 1: 16 proof obligations covering 7 properties -- DONE
- Phase 2: Semantic preservation with array theory -- #192 filed
- Phase 3: Full regalloc correctness (multi-wave, z4 dependent) -- future

### Milestone: Multi-Target Support
**Status: AArch64 primary, x86-64 functional, RISC-V scaffolding**

| Target | ISel | Encoding | Pipeline | Status |
|--------|------|----------|----------|--------|
| AArch64 | 7,133 LOC | 3,580 LOC | Full pipeline | Production |
| x86-64 | 1,893 LOC | 1,774 LOC | Not wired (#191) | Encoding done, needs E2E |
| RISC-V | None | None | Not wired (#193) | Registers + opcodes only |

### Milestone: Object File Output
**Status: Mach-O complete, ELF partially complete**

- Mach-O: Full writer, 20 integration tests, validated -- DONE
- ELF: Writer exists (Wave 16), doctest broken (#188), needs integration test

---

## 7. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Instruction scheduling without dependency tracking causes miscompilation | High | #194 is top Wave 18 priority. Current scheduler may be unsound. |
| x86-64 E2E remains disconnected | Medium | #191 wires it. AArch64 is production target; x86-64 can wait. |
| z4 external dependency blocks real SMT verification | Medium | Mock evaluator covers small widths. 395 proofs ready for z4 activation. |
| Wave 17 carryover causes slot contention in Wave 18 | Low | Dedicate TL6 to carryover. Remaining slots focus on new work. |

---

## 8. Wave 16 vs Wave 17 Summary

| Metric | Wave 16 End | Wave 17 Actions | Net Change |
|--------|-------------|-----------------|------------|
| Open implementation issues | 13 | 4 closed, 5 filed | +1 (net) |
| Open total (excl. security) | ~25 | -4 closed, +5 filed | +1 (net) |
| Closed total | ~30 | +4 | 34 |

**Key shift:** The issue backlog is transitioning from foundational gaps (encoding, basic proofs) to integration work (E2E pipelines, dependency tracking, semantic preservation). This reflects codebase maturity -- the individual components are solid and the challenge is now connecting them.

---

*End of report.*
