# Issue Triage Report — 2026-04-14

**Author:** R1 (Researcher Agent)
**Scope:** Post-Wave 12 issue triage and gap analysis
**Codebase:** 123,825 LOC Rust across 123 files, 3,273 tests

---

## Summary

Waves 9-12 addressed the majority of open implementation issues. Of 30 open issues, 9 non-epic implementation issues were triaged and marked `needs-review`. 4 epics, 5 security issues, and several dependency/infrastructure issues remain correctly open.

---

## Issues Marked `needs-review` (Substantially Complete)

| Issue | Title | Evidence | Remaining |
|-------|-------|----------|-----------|
| #178 | pipeline.rs tests | 69 tests in pipeline.rs (was 0); 932 tests across llvm2-codegen | None |
| #176 | GPU/ANE SMT verification | gpu_semantics (40 tests), ane_semantics (42 tests), unified_synthesis (5,130 LOC) | z4 backend (separate issue) |
| #177 | Profitability dispatch | ProfitabilityAnalyzer wired in dispatch.rs (41 tests), compute_graph.rs (56 tests) | None |
| #143 | Dispatch plan verification | DispatchVerifier in pipeline with 3 modes (Off/Fallback/Error), 69 pipeline tests | z4 backend (separate issue) |
| #140 | ABI features | compact unwind (24 tests), DWARF CFI (27 tests), DWARF info (39 tests) | C++ exception interop |
| #125 | tMIR proof annotations | ProofAnnotation enum, extract_proofs (10+ tests), proof_opts (45 tests) | External tMIR dependency (blocked) |
| #118 | NEON auto-vectorization | vectorize.rs (2,155 LOC), neon_semantics (1,464 LOC), vectorization_proofs (838 LOC) | SVE, SLP vectorization |
| #33 | Proof-enabled optimizations | 5/6 deliverables done: overflow/bounds/null/borrow/CSE elimination (2,020 LOC, 45 tests) | PositiveRefCount (tSwift) |
| #119 | Metal IR emission | metal_emitter.rs (2,222 LOC, 39 tests), wired into pipeline | MPS, Metal 3 mesh |
| #120 | CoreML/ANE lowering | coreml_emitter.rs (1,757 LOC, 28 tests), wired into pipeline | Hardware profiling |

---

## Issues Correctly Remaining Open

### Blocked / External Dependencies
| Issue | Title | Status | Blocker |
|-------|-------|--------|---------|
| #125 | tMIR proof annotations | LLVM2-side done, blocked on external tMIR repo | tMIR repo integration |
| #122 | z4 QF_ABV array theory | In-progress | z4 repo |
| #123 | z4 QF_FP floating-point | In-progress | z4 repo |
| #124 | z4 bounded quantifiers | In-progress | z4 repo |
| #23 | tRust LLVM IR lifting | Mail from tRust | tRust repo |

### Epics (Tracking Only)
| Issue | Title | Notes |
|-------|-------|-------|
| #24 | AArch64 Backend | Parent epic; many sub-tasks complete |
| #109 | Heterogeneous compute | GPU/ANE/CPU dispatch substantially implemented |
| #108 | AI-native compilation | Long-term vision |
| #107 | Radical debugging | Long-term vision |
| #106 | Solver-driven superoptimization | Long-term vision |
| #121 | Unified solver architecture | Blocked on z4 integration |

### Infrastructure / Low Priority
| Issue | Title | Notes |
|-------|-------|-------|
| #141 | Design docs audit | P3, documentation concern |
| #103 | RISC-V target definitions | P3, placeholder values in target.rs |
| #34 | z4 verification integration | P3, blocked on z4 repos |

### Security (Do Not Close)
| Issue | Title |
|-------|-------|
| #22 | LLVM Fork security implications |
| #21 | GitHub App token cache predictable path |
| #20 | Arbitrary code execution via looper.py |
| #19 | Arbitrary code execution in sync validation |
| #18 | Arbitrary command execution via shell=True |
| #17 | Command injection in test logger |
| #16 | Shell injection in dependency checker |

---

## Identified Gaps for Future Work

### Priority 1: z4 Solver Integration
The mock-based verification (exhaustive for small widths, random sampling for 32/64-bit) is the biggest remaining gap. Real SMT verification requires z4 QF_ABV (#122), QF_FP (#123), and bounded quantifiers (#124). This is the critical path to the project's verification mission.

### Priority 2: End-to-End Verification Pipeline
While individual proofs exist, there is no automated "compile and verify" pipeline that runs all lowering proofs against a given tMIR input. This would be the integration test for the verification story.

### Priority 3: RISC-V and x86-64 Targets
- x86-64 has encoding stubs but no instruction selection
- RISC-V has placeholder values in target.rs (#103)
- Both are far behind AArch64 maturity

### Priority 4: C++ Exception Handling
DWARF CFI and compact unwind are done, but C++ exception handling (LSDA tables, personality routines) is not implemented. Needed for C++ interop via tC.

### Priority 5: Benchmark Infrastructure
Several issues (#33 acceptance criteria) require benchmark comparisons. No benchmark infrastructure exists to measure compilation speed or output code quality against LLVM/Cranelift baselines.

---

## Codebase Statistics (Post-Wave 12)

| Crate | Key Modules | Test Count |
|-------|-------------|------------|
| llvm2-codegen | pipeline, metal_emitter, coreml_emitter, aarch64/encode, unwind, dwarf_cfi, macho | ~932 |
| llvm2-verify | gpu/ane/neon_semantics, unified_synthesis, lowering_proof, peephole_proofs | ~650 |
| llvm2-opt | proof_opts, vectorize, peephole, cse, dce, copy_prop, passes | ~580 |
| llvm2-lower | isel, abi, adapter, dispatch, compute_graph, target_analysis | ~700 |
| llvm2-ir | inst, lib, cost_model, trace | ~300 |
| llvm2-regalloc | linear_scan, greedy, liveness, spill, coalesce | ~111 |
| **Total** | **123 files, 123,825 LOC** | **3,273 tests** |
