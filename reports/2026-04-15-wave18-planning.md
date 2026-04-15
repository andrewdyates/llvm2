# Wave 18 Issue Triage and Wave 19 Planning

**Author:** R1 (Researcher Agent)
**Date:** 2026-04-15
**Commit base:** fadfc3f (main)
**Scope:** Close completed issues from Wave 17, file new issues, plan Wave 19

---

## Executive Summary

Closed 4 issues completed during Wave 17 (issues #180 and #182 were already closed by Wave 17 R1). Filed 5 new issues for Wave 19 priorities. The project is at 272,845 LOC, 1,140 tests, and 477 proof functions across 6 production crates + 4 stubs.

**Key finding:** Wave 17 was highly productive -- all 6 assigned issues were fully completed. The project's main gap has shifted from missing encodings/proofs to missing end-to-end integration testing. Individual components (ISel, encoding, Mach-O, verification) are solid, but nothing proves they compose correctly into a working binary.

**Snapshot:** 272,845 LOC Rust, 1,140 tests (1139 pass, 1 RISC-V FSD bug), 477 proof functions, 47 open issues.

---

## 1. Issues Closed This Wave

| # | Title | Evidence | Closed By |
|---|-------|----------|-----------|
| #187 | [codegen] 22 AArch64 opcodes unencoded | Commit 0134981: 9 real + 13 alias encodings, 44 tests | Wave 17 TL1 |
| #188 | [codegen] ELF writer doctest failure | Commit b03e886: doctest fixed, debug.rs (430 LOC) added | Wave 17 TL3 |
| #186 | [verify] E2E verification pipeline | Commit de10cfb: FunctionVerifier, 32 tests | Wave 17 TL4 |
| #184 | [verify] Constant materialization proofs | Commit b8f15a2: 22 proofs (1094 LOC), 36 tests | Wave 17 TL7 |

Previously closed by Wave 17 R1:
| #180 | [verify] Bitwise/shift proofs | Already closed | Wave 17 R1 |
| #182 | [codegen] x86-64 encoding | Already closed | Wave 17 R1 |

**Total closed this wave: 4 issues (6 total from Wave 17 work)**

---

## 2. Issues Updated (Not Closed)

No partially-addressed issues requiring label updates. All Wave 17 assignments were fully completed.

---

## 3. New Issues Filed (Wave 19 Backlog)

| # | Title | Priority | Category | Rationale |
|---|-------|----------|----------|-----------|
| #196 | [codegen] RISC-V FSD encoding bug | P2 | Bug fix | Only failing test in workspace (1/1140) |
| #197 | [codegen] Wire verify_function() into Pipeline | P2 | Integration | #186 built the verifier but it is not in the pipeline |
| #198 | [codegen] E2E Mach-O linking test | P1 | Integration | No test proves pipeline stages compose to a valid binary |
| #199 | [opt] Loop optimization passes | P2 | Optimization | Unrolling, strength reduction, IV simplification missing |
| #200 | [quality] Fix 17+ compiler warnings | P3 | Code quality | Warnings mask real issues |

**Total new issues: 5**

---

## 4. Wave 19 Priority Recommendations

### Tier 1 (Must-do)

| # | Title | Why |
|---|-------|-----|
| #198 | E2E Mach-O linking test | Proves the backend works end-to-end. P1 priority. |
| #196 | RISC-V FSD encoding bug | Only failing test. Quick fix, high signal. |
| #197 | Wire verify_function() into Pipeline | Completes the verification story started by #186. |

### Tier 2 (Should-do)

| # | Title | Why |
|---|-------|-----|
| #191 | x86-64 E2E compilation pipeline | Second target needs integration testing. |
| #194 | Instruction scheduling Phase 2 | Dependency tracking needed for correctness. |
| #185 | Frame index elimination proofs | Verification gap in stack frame handling. |
| #183 | Address mode formation proofs | Verification gap in addressing. |

### Tier 3 (Nice-to-have)

| # | Title | Why |
|---|-------|-----|
| #199 | Loop optimization passes | Important for code quality but not blocking. |
| #200 | Compiler warnings cleanup | Hygiene. Quick wins. |
| #193 | RISC-V target.rs wiring | Third target, lower priority than completing first two. |

### Suggested Wave 19 Assignment (6-7 techleads)

| TL | Issue | Type |
|----|-------|------|
| TL1 | #198 E2E Mach-O linking | Integration test (P1) |
| TL2 | #196 RISC-V FSD bug | Bug fix (quick) + #200 warnings cleanup |
| TL3 | #197 Pipeline verification wiring | Integration |
| TL4 | #185 Frame index elimination proofs | Verification |
| TL5 | #183 Address mode formation proofs | Verification |
| TL6 | #191 x86-64 E2E pipeline | Pipeline integration |
| TL7 | #194 Instruction scheduling Phase 2 | Codegen |

---

## 5. Open Issue Inventory (Post-Triage)

### By Priority

| Priority | Count | Key Issues |
|----------|-------|------------|
| P1 | 4 | #198 (E2E test), #122 (z4 QF_ABV), #125 (tMIR proofs), #24 (epic) |
| P2 | 15 | #196, #197, #199, #185, #183, #191, #192, #194, #195, #140, #123, #124, + epics |
| P3 | 6 | #200, #190, #189, #141, #193, #103 |
| Untagged | 22 | Security issues (#5-#22), tRust mail (#23) |

### By Category

| Category | Count |
|----------|-------|
| Verification (proofs/pipeline) | 8 |
| Codegen (encoding/pipeline) | 6 |
| Optimization | 2 |
| Code quality / docs | 5 |
| External dependencies (z4/tMIR) | 4 |
| Epics (tracking) | 5 |
| Security (template) | 17 |

---

## 6. Velocity and Trajectory

Wave 17 completed 6 issues across 7 techleads with the following deliverables:
- 22 opcode encodings + 44 tests (TL1)
- Copy coalescing enhancement (TL2)
- ELF debug sections, 430 LOC (TL3)
- FunctionVerifier, 32 tests (TL4)
- 14 bitwise/shift proofs, 18 tests (TL5)
- x86-64 SSE/CMOV/SETCC/LEA/BSF encoding (TL6)
- 22 constant materialization proofs, 36 tests (TL7)

The project has crossed the threshold where individual component quality is solid. The next inflection point is proving end-to-end composition works (issue #198). After that, the focus should shift to:
1. Completing verification coverage for all optimization passes
2. Linker-consumable output for all three targets
3. Performance benchmarking against LLVM/Cranelift baselines
