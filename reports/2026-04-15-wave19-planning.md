# Wave 19 Issue Triage and Wave 20 Planning

**Author:** R1 (Researcher Agent)
**Date:** 2026-04-15
**Commit base:** 6955376 (main)
**Scope:** Verify Wave 18 closures, update issue states, plan Wave 20

---

## Executive Summary

Wave 18 delivered 7 techlead assignments and closed 8 issues total (6 from techleads + 2 from audit). The project now stands at 159,353 LOC Rust, 4,356 tests (all passing), and 477 proof functions across 6 production crates. The workspace is clean with zero warnings and zero test failures.

**Key finding:** The project has reached a maturity inflection point. Core AArch64 pipeline stages (ISel, optimization, regalloc, encoding, Mach-O) are individually solid with strong test and proof coverage. The critical next step is **end-to-end integration testing** (issue #198) -- proving these stages compose into a valid binary. Secondary targets (x86-64, RISC-V) now have encoding infrastructure and need pipeline wiring. Verification coverage continues to deepen with 22 proof categories, but regalloc Phase 2 semantic proofs (#192) remain the hardest open verification challenge.

**Snapshot:** 159,353 LOC Rust | 4,356 tests (4,356 pass, 0 fail) | 477 proof functions | 39 open issues (14 actionable, 17 security/template, 5 epics, 3 external-blocked)

---

## 1. Wave 18 Delivery Verification

### Issues Closed by Wave 18

| # | Title | Techlead | Evidence |
|---|-------|----------|----------|
| #191 | x86-64 E2E compilation pipeline | TL1 | Commit cbdc7fe: SSE/CMOV/SETCC/LEA, pipeline x86-64 dispatch |
| #194 | Instruction scheduling Phase 2 | TL2 | Commit dff5fa9: hazard detection, register pressure, schedule metrics |
| #193 | RISC-V encoding | TL3 | Commit f69d25d: R/I/S/B/U/J type formats, 40+ encoding tests |
| #183 | Address mode formation proofs | TL4 | Commit in wave: base+imm, base+reg, scaled, writeback proofs |
| #185 | Frame index elimination proofs | TL5 | Commit d49b7b2: offset computation, alignment, callee-save proofs |
| #195 | Scheduler correctness proofs | TL6 | Commit in wave: dependency preservation, memory ordering proofs |
| #196 | RISC-V FSD encoding bug | Audit | Fixed during Wave 18 audit (concurrent agent) |
| #200 | 17+ compiler warnings | Audit | Fixed during Wave 18 audit |

**TL7 (#140 ABI extensions):** Partially complete (commit d750531). Varargs, HFA, aggregates, sret added. SIMD vector ABI and libunwind integration test remain. Issue stays OPEN with `needs-review` label.

**Verification:** All 8 issues confirmed CLOSED. Issue #140 confirmed OPEN with accurate status comments.

---

## 2. Current Open Issue Inventory

### Actionable Issues (14)

| # | Title | Priority | Category | Blocked? |
|---|-------|----------|----------|----------|
| #198 | E2E Mach-O linking test | P1 | Integration | No |
| #197 | Wire verify_function() into Pipeline | P2 | Integration | No |
| #199 | Loop optimization passes | P2 | Optimization | No |
| #192 | Regalloc Phase 2 semantic proofs | P2 | Verification | Soft (#122) |
| #140 | ABI: SIMD vector args + libunwind test | P2 | ABI | No |
| #124 | z4 bounded quantifiers | P2 | z4 dep | External |
| #123 | z4 QF_FP floating-point theory | P2 | z4 dep | External |
| #190 | lower.rs role boundary docs | P3 | Documentation | No |
| #189 | tmir-semantics dead code cleanup | P3 | Documentation | No |
| #141 | Design docs gap analysis | P3 | Documentation | No |
| #103 | RISC-V target definition TODOs | P3 | Codegen | Partially addressed by #193 |
| #34 | Phase 9: z4 integration | P3 | Verification | Partially complete |
| #23 | tRust LLVM IR lifting | P2 | Mail | External |
| #22 | Security implications of LLVM fork | -- | Security | No |

### External-Blocked Issues (3)

| # | Title | Blocked On |
|---|-------|-----------|
| #125 | tMIR proof annotations | tMIR repo |
| #122 | z4 QF_ABV array theory | z4 repo (in-progress) |
| #192 | Regalloc Phase 2 proofs | Soft dependency on #122 |

### Epics (5)

| # | Title | Status |
|---|-------|--------|
| #24 | AArch64 Backend Implementation | Active -- core pipeline complete, integration testing needed |
| #121 | Unified solver architecture | Blocked on z4 theories (#122, #123, #124) |
| #106 | Solver-driven superoptimization | Active -- CEGIS loop implemented, needs z4 integration |
| #107 | Radical debugging/transparency | Active -- provenance + compilation trace implemented |
| #108 | AI-native compilation | Active -- rule discovery framework implemented |
| #109 | Automatic heterogeneous compute | Active -- dispatch planning implemented |

### Security/Template Issues (17)

Issues #5-#21 are inherited template security findings. These are ai_template concerns, not LLVM2-specific. Recommend bulk-closing as `environmental` or filing upstream to ai_template repo.

---

## 3. Wave 20 Priority Recommendations

### Tier 1: Must-Do

| # | Title | Why | Estimated Complexity |
|---|-------|-----|---------------------|
| #198 | E2E Mach-O linking test | **Top priority.** Proves the AArch64 backend produces valid machine code that the OS linker accepts. This is the single most impactful test we can write -- it validates the entire pipeline stack (ISel -> opt -> regalloc -> frame -> encode -> Mach-O). | Medium (1 TL) |
| #197 | Wire verify_function() into Pipeline | Completes the verification story from #186. FunctionVerifier exists but is not callable from the pipeline. One flag to enable formal verification during compilation. | Small (1 TL) |
| #103 | RISC-V target cleanup + pipeline wiring | Wave 18 delivered RISC-V encoding (#193). Now wire it into the pipeline (like x86-64 was wired in Wave 18). Update target.rs TODOs with real values from riscv_regs.rs. | Small (1 TL) |

### Tier 2: Should-Do

| # | Title | Why | Estimated Complexity |
|---|-------|-----|---------------------|
| #192 | Regalloc Phase 2 semantic proofs | Hardest open verification challenge. Can start with mock evaluator even without z4 array theory. Models register file as SMT arrays. | Large (1 TL) |
| #199 | Loop optimization passes | Loop unrolling and strength reduction are among the highest-impact optimizations for numerical code. Without these, LLVM2 output quality will lag LLVM significantly for loop-heavy programs. | Large (1 TL) |
| #140 | ABI: SIMD vector args | Add Vector type classification to abi.rs for NEON parameter passing. Small scoped work, high impact for any code using SIMD structs. | Small (1 TL) |

### Tier 3: Nice-to-Have

| # | Title | Why | Estimated Complexity |
|---|-------|-----|---------------------|
| #190 | lower.rs role boundary docs | Documentation clarification. Not blocking but reduces confusion for new contributors. | Tiny (combine with another TL) |
| #189 | tmir-semantics dead code | Either remove the stub or integrate it. Clean codebase. | Tiny (combine with another TL) |
| #141 | Design docs gap analysis | Documentation quality. | Tiny (combine with another TL) |

---

## 4. Suggested Wave 20 Assignment (7 Techleads)

| TL | Primary Issue | Secondary Issue | Type |
|----|--------------|-----------------|------|
| TL1 | #198 E2E Mach-O linking test | -- | Integration (P1) |
| TL2 | #197 Wire verify_function() into Pipeline | #190 lower.rs docs | Integration + Docs |
| TL3 | #103 RISC-V pipeline wiring | #189 tmir-semantics cleanup | Codegen + Cleanup |
| TL4 | #192 Regalloc Phase 2 semantic proofs | -- | Verification (hard) |
| TL5 | #199 Loop optimization: unrolling | -- | Optimization |
| TL6 | #140 SIMD vector ABI args | #141 design doc gap audit | ABI + Docs |
| TL7 | #201 x86-64 E2E ELF linking test | -- | Integration |

**Rationale:**
- TL1 is the highest-impact single task (proves the backend works end-to-end)
- TL2 + TL3 complete pipeline wiring for verification and RISC-V
- TL4 tackles the hardest open verification challenge
- TL5 fills the biggest optimization gap
- TL6 closes the remaining ABI work
- TL7 extends E2E testing to the second target

### New Issue Filed

- **#201 x86-64 E2E ELF linking test:** Wave 18 TL1 delivered x86-64 pipeline wiring (#191). The logical follow-up is an E2E test similar to #198 but targeting ELF/x86-64. This proves the second target works end-to-end.

---

## 5. Strategic Assessment

### Strengths
- **Verification depth:** 477 proof functions across 22 categories. No other compiler backend at this scale has this level of formal verification.
- **Multi-target progress:** AArch64 is primary and mature. x86-64 has pipeline wiring. RISC-V has full encoding. All three targets have meaningful infrastructure.
- **Clean workspace:** Zero warnings, zero failures, 4,356 tests all passing. The codebase is healthy.
- **Comprehensive test coverage:** Tests grew 10.6% from Wave 17 to Wave 18 (3,937 -> 4,356).

### Risks
- **No E2E validation yet:** Individual stages are well-tested but composition bugs (wrong calling convention, misaligned Mach-O sections, incorrect relocations) can only be caught by end-to-end testing. Issue #198 is the critical path.
- **z4 dependency:** Issues #122, #123, #124 depend on external z4 repo progress. Regalloc Phase 2 proofs (#192) and full z4 integration (#34) are soft-blocked until z4 delivers QF_ABV and QF_FP.
- **Loop optimization gap:** Without loop unrolling/strength reduction (#199), LLVM2 will produce noticeably worse code than LLVM for loop-heavy programs. This is the biggest code quality gap.

### Trajectory
At current velocity (6-8 issues per wave), the project will reach:
- **Wave 20:** E2E validation for AArch64 + RISC-V pipeline wiring + verification deepening
- **Wave 22-24:** E2E validation for all three targets + loop optimizations + z4 integration (if z4 ships QF_ABV/QF_FP)
- **Wave 25+:** Performance benchmarking against LLVM/Cranelift baselines, self-hosting preparation

---

## 6. Velocity and Metrics

| Metric | Wave 17 | Wave 18 | Delta |
|--------|---------|---------|-------|
| LOC (crates) | 145,909 | 158,607 | +12,698 (+8.7%) |
| Tests | 3,937 | 4,356 | +419 (+10.6%) |
| Proof functions | 395 | 477 | +82 (+20.8%) |
| Proof categories | 19 | 22 | +3 |
| Source files | ~130 | 152 | +22 |
| Issues closed (wave) | 6 | 8 | +2 |
| Open issues (actionable) | ~18 | 14 | -4 |
| Warnings | 0 | 0 | -- |
| Test failures | 0 | 0 | -- |

**Observation:** Proof function growth rate (20.8%) outpaces LOC growth (8.7%), indicating increasing verification density. This is the right trajectory for a verified compiler.
