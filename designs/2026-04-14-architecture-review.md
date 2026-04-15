# LLVM2 Architecture Review -- Wave 16

**Date:** 2026-04-14
**Author:** Wave 16 R1 (Researcher)
**Scope:** Comprehensive codebase audit after Waves 9-16

---

## Executive Summary

LLVM2 has grown to **~146K LOC** across **6 crates** (143 Rust source files) plus 4 tMIR stubs (~750 LOC). The workspace compiles cleanly (`cargo check` passes). Test suite: **3,837 tests across all crates** (3,830 pass, 7 ignored in integration tests requiring AArch64 + `cc`, 0 permanent failures in lib tests). One doctest failure exists in the new ELF writer module. The AArch64 pipeline is functional end-to-end (tMIR -> ISel -> Optimize -> RegAlloc -> Frame -> Encode -> Mach-O -> Link -> Run). x86-64 has ISel + encoding scaffolding. RISC-V exists only as placeholder constants.

**Overall maturity: Late Alpha.** The happy path works for simple functions. Gaps remain in opcode encoding coverage, ABI edge cases, and the formal verification path (z4 not integrated; all proofs use mock evaluation).

---

## 1. Per-Crate Maturity Assessment

### 1.1 llvm2-ir (Shared Machine Model)

| Metric | Value |
|--------|-------|
| LOC | 12,828 |
| Files | 17 |
| Modules | 13 (aarch64_regs, cc, cost_model, function, inst, operand, provenance, regs, trace, type_hierarchy, types, x86_64_ops, x86_64_regs) |
| Tests | 346 |
| **Maturity** | **Functional** |

**Assessment:** Solid foundation. Defines canonical types used by all other crates: `MachInst`, `MachFunction`, `MachBlock`, `MachOperand`, `VReg`/`PReg`, `AArch64Opcode`, `X86Opcode`. AArch64 register definitions are comprehensive (GPR, FPR, SIMD, system regs). x86-64 register/opcode definitions exist and are substantial (1,166 + 513 LOC). Cost model is ambitious (2,660 LOC covering CPU/NEON/GPU/ANE latencies). Provenance tracking (1,064 LOC) and compilation trace (704 LOC) provide debugging infrastructure.

**Gaps:**
- No RISC-V register or opcode definitions (only placeholder values in `target.rs`)
- Cost model values appear sourced from Apple documentation but lack empirical calibration
- `type_hierarchy.rs` is documentation-only (111 LOC), not enforcement code

### 1.2 llvm2-lower (tMIR -> MachIR Lowering)

| Metric | Value |
|--------|-------|
| LOC | 24,046 |
| Files | 12 |
| Modules | 10 (abi, adapter, compute_graph, dispatch, function, instructions, isel, target_analysis, types, x86_64_isel) |
| Tests | 503 |
| **Maturity** | **Functional** (AArch64), **Scaffolding** (x86-64) |

**Assessment:** The AArch64 ISel (`isel.rs`, 7,133 LOC) is the largest single module and handles: arithmetic, comparisons, branches, calls, returns, loads, stores, constants, type casts, FP operations. The ABI module (`abi.rs`, 2,484 LOC) covers Apple AArch64 calling convention with unwind info generation. The adapter layer (`adapter.rs`, 3,752 LOC) translates tMIR modules/functions/types and extracts proof annotations.

Heterogeneous compute subsystem is substantial: compute graph analysis (3,253 LOC), dispatch planning (2,763 LOC), and proof-guided target analysis (1,245 LOC). This enables CPU/NEON/GPU/ANE scheduling.

x86-64 ISel (`x86_64_isel.rs`, 1,893 LOC) covers basic integer arithmetic, comparisons, branches, calls, returns, loads, stores. It is functional for simple functions but lacks FP operations, SIMD, and many edge cases.

**Gaps:**
- ISel has a `TODO: Full 64-bit materialization for very large constants` at line 1673
- No RISC-V ISel
- Switch statement lowering uses a simple comparison chain (no jump table)
- Struct/aggregate lowering is minimal
- Heterogeneous dispatch is theoretical -- no hardware-level integration yet

### 1.3 llvm2-opt (Verified Optimizations)

| Metric | Value |
|--------|-------|
| LOC | 14,517 |
| Files | 21 |
| Modules | 19 (addr_mode, cfg_simplify, cmp_branch_fusion, cmp_select, const_fold, const_materialize, copy_prop, cse, dce, dom, effects, licm, loops, pass_manager, passes, peephole, pipeline, proof_opts, scheduler, vectorize) |
| Tests | 266 (lib) + integration |
| **Maturity** | **Functional** |

**Assessment:** Comprehensive optimization pass suite with pipeline support (O0-O3 levels). Key passes: DCE, constant folding, copy propagation, peephole, CSE (dominator-based), LICM, CFG simplification, address mode formation, CMP+branch fusion, compare-select combines, constant materialization (MOVZ/MOVK/MOVN), NEON auto-vectorization. All passes operate on `llvm2_ir::MachFunction`.

The pass manager supports `run_once` and `run_to_fixpoint` modes. Memory effects model classifies opcodes for safety analysis.

Proof-guided optimization (`proof_opts.rs`, 2,020 LOC) consumes tMIR proof annotations to eliminate runtime checks -- this is the key differentiator vs. LLVM.

New additions: `cmp_branch_fusion` (676 LOC, fuses CMP/TST + BCond into CBZ/CBNZ/TBZ/TBNZ) and `scheduler` (909 LOC, instruction scheduling).

**Gaps:**
- Vectorizer is analysis-only; rewriting replaces opcodes but does not yet produce optimal NEON code sequences
- No global value numbering (GVN)
- No strength reduction
- No tail call optimization
- Scheduler is newly added; integration with pipeline unclear

### 1.4 llvm2-regalloc (Register Allocation)

| Metric | Value |
|--------|-------|
| LOC | 13,034 |
| Files | 13 |
| Modules | 12 (call_clobber, coalesce, greedy, liveness, linear_scan, machine_types, phi_elim, post_ra_coalesce, remat, spill, spill_slot_reuse, split) |
| Tests | 296 |
| **Maturity** | **Functional** |

**Assessment:** Two allocation strategies: linear scan (1,262 LOC) and greedy with eviction/splitting (1,341 LOC). Full pipeline: critical edge splitting -> phi elimination -> liveness analysis -> copy coalescing -> allocation -> rematerialization -> spill code insertion -> spill slot reuse -> call save/restore -> post-RA coalescing.

The `allocate()` entry point cleanly orchestrates all phases with configurable strategy. AArch64 register set is fully defined. Tests cover straight-line, diamond, loop, high-pressure, call-crossing, FPR64, and empty function cases.

**Gaps:**
- No x86-64 register set configuration (would need different allocatable_regs)
- No RISC-V support
- Greedy allocator splitting heuristics may need tuning for real-world code
- No second-chance allocation or live range splitting at block boundaries
- `machine_types.rs` defines its own `MachFunction`/`MachInst`/`MachBlock` adapter types rather than using `llvm2_ir` types directly (issue #73 remnant)

### 1.5 llvm2-verify (Formal Verification)

| Metric | Value |
|--------|-------|
| LOC | 38,330 |
| Files | 31 |
| Modules | 27 (smt, tmir_semantics, aarch64_semantics, nzcv, lowering_proof, peephole_proofs, opt_proofs, const_fold_proofs, cse_licm_proofs, dce_proofs, copy_prop_proofs, cfg_proofs, memory_model, memory_proofs, verify, z4_bridge, synthesis, cegis, rule_discovery, neon_semantics, ane_semantics, gpu_semantics, unified_synthesis, neon_lowering_proofs, vectorization_proofs, ane_precision_proofs, regalloc_proofs, proof_database, verification_runner) |
| Tests | 1,219 |
| **Maturity** | **Functional** (mock evaluation), **Stub** (z4 integration) |

**Assessment:** The largest crate by LOC. Contains a self-contained SMT bitvector expression AST + evaluator (`smt.rs`, 2,599 LOC) that enables verification without an external solver. The z4 bridge (`z4_bridge.rs`, 2,182 LOC) can serialize to SMT-LIB2 format but z4 integration is feature-gated and not connected.

**Proof categories (18) and approximate counts:**
- Arithmetic (add/sub/mul/neg, I8-I64): ~16 proofs
- Division (sdiv/udiv): ~4 proofs
- Floating-point (fadd/fsub/fmul/fneg): ~8 proofs
- NZCV flags: ~4 proofs
- Comparisons (10 conditions x I32/I64): ~20 proofs
- Branches (conditional): ~20 proofs
- Peephole identities: ~54 proofs
- Optimization (const fold, absorb, DCE, copy prop): ~28 proofs
- Constant folding (comprehensive): ~68 proofs
- Copy propagation: ~30 proofs
- CSE/LICM: ~56 proofs
- DCE: ~22 proofs
- CFG simplification: ~32 proofs
- Memory (load/store): ~91 proofs
- NEON lowering: ~55 proofs
- Vectorization: ~45 proofs
- ANE precision: ~5 proofs
- Register allocation: ~16 proofs

**Total: ~570+ registered proof obligations** (verified via mock evaluation: exhaustive for 8-bit, statistical sampling for 32/64-bit).

Advanced features: CEGIS loop (1,013 LOC), peephole synthesis (1,654 LOC), unified multi-target synthesis (5,130 LOC), AI-native rule discovery (1,388 LOC). These are functional but not integrated into the build pipeline.

**Gaps:**
- All proofs use mock evaluation, not formal z4/z3 SMT solving
- Statistical verification for 32/64-bit proofs provides high confidence but is not a formal proof
- No proofs for: bitwise/shift lowering (#180), address mode formation (#183), constant materialization (#184), frame index elimination (#185), end-to-end verify_function() (#186)
- z4 bridge exists but is not wired up (z4 dependency is commented out in workspace Cargo.toml)
- NEON/GPU/ANE semantics exist but are not integrated into the verification pipeline
- `verify_function()` entry point is a TODO (issue #186)

### 1.6 llvm2-codegen (Code Generation)

| Metric | Value |
|--------|-------|
| LOC | 43,153 |
| Files | 49 |
| Modules | 17 top-level + sub-modules (aarch64: encode/encoding/encoding_fp/encoding_mem, macho: constants/fixup/header/reloc/section/symbol/writer, elf: constants/header/reloc/section/symbol/writer, x86_64: encode) |
| Tests | 1,127 |
| **Maturity** | **Functional** (AArch64+Mach-O), **Scaffolding** (x86-64, ELF) |

**Assessment:** The largest crate. The AArch64 encoder (`aarch64/encode.rs`, 3,580 LOC + 3 encoding sub-modules) handles integer, memory, FP, and branch instructions. Mach-O writer (7 modules, ~3,600 LOC total) produces valid .o files that link with the system `cc`. ELF writer (7 modules, ~2,400 LOC) is newly added for Linux x86-64/AArch64 targets. Frame lowering (2,286 LOC) generates prologue/epilogue with Darwin compact unwind encoding. Branch relaxation (1,829 LOC) handles distance-dependent encoding. DWARF CFI (864 LOC) and DWARF debug info (1,533 LOC) provide debugging support.

The pipeline module (2,649 LOC) wires everything together: tMIR -> ISel -> Adapt -> Optimize -> RegAlloc -> Apply Allocation -> Frame Lower -> Encode -> Mach-O.

Heterogeneous emitters: Metal MSL kernel emitter (2,222 LOC) and CoreML MIL emitter (1,757 LOC) for GPU and ANE targets.

**End-to-end tests confirm:** compiling simple functions (addition, conditional return, loops), linking with C drivers via `cc`, and executing on Apple Silicon with correct results.

**Gaps:**
- 9 AArch64 opcodes lack encoding: `AndRI`, `OrrRI`, `EorRI`, `BicRR`, `Csinc`, `Csinv`, `Csneg`, `Movn`, `FmovImm`
- 13 LLVM-style typed aliases lack encoding: `MOVWrr`, `MOVXrr`, `STRWui`, `STRXui`, `STRSui`, `STRDui`, `BL`, `BLR`, `CMPWrr`, `CMPXrr`, `CMPWri`, `CMPXri`, `Bcc`
- x86-64 encoder is scaffolded but not connected to the pipeline
- ELF writer has a doctest failure (new code, likely minor)
- No RISC-V encoding
- Exception handling (1,016 LOC) is defined but not integrated into the pipeline
- `lower.rs` (2,264 LOC) appears to duplicate some ISel functionality

---

## 2. Cross-Crate Dependency Analysis

```
                    tmir-types  tmir-instrs  tmir-func  tmir-semantics (stubs)
                        |           |            |
                        v           v            v
                    llvm2-ir  <-- llvm2-lower ------+
                        |           |               |
                        +-----+-----+               |
                        |     |     |               |
                        v     v     v               |
                   llvm2-opt  llvm2-regalloc        |
                        |     |                     |
                        +-----+-----+               |
                              |     |               |
                              v     v               v
                         llvm2-verify          llvm2-codegen
                              |                     |
                              +---------------------+
                                        |
                                  (codegen depends on
                                   ir, lower, opt,
                                   regalloc, verify)
```

**Key observations:**
1. `llvm2-ir` is the leaf dependency -- all crates depend on it, it depends on nothing internal
2. `llvm2-codegen` is the root -- depends on ALL other 5 crates + tMIR stubs
3. `llvm2-verify` depends on both `llvm2-ir` and `llvm2-lower` (for proof obligations that reference lowering rules)
4. `llvm2-opt` and `llvm2-regalloc` depend only on `llvm2-ir` -- clean separation
5. tMIR stubs are used directly by `llvm2-lower` (adapter) and `llvm2-codegen` (pipeline, tests)
6. `tmir-semantics` is defined but not depended on by any crate -- dead stub for now

**Risk:** `llvm2-codegen` importing all crates creates a compile-time bottleneck. Any change to `llvm2-ir` forces recompilation of everything. This is acceptable at current scale (~7s incremental build) but may become painful at 500K+ LOC.

---

## 3. Compilation Pipeline Completeness

| Phase | Module | Status | Notes |
|-------|--------|--------|-------|
| 1. tMIR Parse | `llvm2-lower::adapter` | **Functional** | Translates tMIR Module/Function/Type; extracts proofs |
| 2. Instruction Selection | `llvm2-lower::isel` | **Functional** (AArch64) | Tree-pattern matching, ~50+ patterns |
| 2b. x86-64 ISel | `llvm2-lower::x86_64_isel` | **Scaffolding** | Basic arithmetic, branches, calls |
| 3. ISel -> IR Adapt | `llvm2-codegen::pipeline` | **Functional** | ISelFunction -> MachFunction conversion |
| 4. Optimization | `llvm2-opt::pipeline` | **Functional** | 12+ passes, O0-O3 levels |
| 5. IR -> RegAlloc Adapt | `llvm2-codegen::pipeline` | **Functional** | MachFunction -> RegAllocFunction |
| 6. Register Allocation | `llvm2-regalloc` | **Functional** | Linear scan + greedy, full pipeline |
| 7. Apply Allocation | `llvm2-codegen::pipeline` | **Functional** | VReg -> PReg rewriting |
| 8. Frame Lowering | `llvm2-codegen::frame` | **Functional** | Prologue/epilogue, frame index elim |
| 9. Branch Relaxation | `llvm2-codegen::relax` | **Functional** | Distance-dependent encoding |
| 10. Binary Encoding | `llvm2-codegen::aarch64` | **Functional** | ~80+ AArch64 opcodes encoded |
| 10b. x86-64 Encoding | `llvm2-codegen::x86_64` | **Scaffolding** | REX/ModRM framework, basic ops |
| 11. Mach-O Emission | `llvm2-codegen::macho` | **Functional** | Valid .o files, symbols, relocations |
| 11b. ELF Emission | `llvm2-codegen::elf` | **Scaffolding** | New module, doctest failure |
| 12. Unwind/DWARF | `llvm2-codegen::unwind`, `dwarf_cfi`, `dwarf_info` | **Functional** | Compact unwind + DWARF CFI/debug |
| 13. Link + Run | E2E tests | **Functional** | Verified on AArch64 Apple Silicon |

**Pipeline verdict:** AArch64 pipeline is complete from tMIR to running binary for simple functions. The happy path works. Complex cases (large structs, varargs, exception handling, dynamic dispatch) are not yet supported.

---

## 4. Target Maturity Comparison

| Feature | AArch64 | x86-64 | RISC-V |
|---------|---------|--------|--------|
| Register definitions | **Complete** (GPR/FPR/SIMD/System) | **Complete** (GPR/XMM/YMM) | **None** (hardcoded counts) |
| Opcode enum | **Complete** (~180+ opcodes) | **Functional** (~80+ opcodes) | **None** |
| Instruction selection | **Functional** (7,133 LOC) | **Scaffolding** (1,893 LOC) | **None** |
| Binary encoding | **Functional** (3,580 LOC + sub-modules) | **Scaffolding** (1,774 LOC) | **None** |
| ABI / Calling convention | **Functional** (Apple AAPCS64) | **Defined** (System V AMD64 params) | **Defined** (riscv_lp64d params) |
| Frame lowering | **Functional** (2,286 LOC) | **None** | **None** |
| Object format | **Functional** (Mach-O) | **Scaffolding** (ELF) | **None** |
| Branch relaxation | **Functional** | **None** | **None** |
| Unwind info | **Functional** (compact + DWARF) | **None** | **None** |
| E2E test (link+run) | **Yes** (passes) | **No** | **No** |
| Verification proofs | **Yes** (570+ proofs) | **None** | **None** |

**Summary:** AArch64 is 1-2 orders of magnitude ahead. x86-64 has meaningful scaffolding (~5K LOC across ISel+encoding+registers+ELF) but no working pipeline. RISC-V is issue #103 with only placeholder integer constants in `target.rs`.

---

## 5. Verification Coverage

### 5.1 Covered Subsystems

| Subsystem | Proof Count | Strength | Notes |
|-----------|-------------|----------|-------|
| Arithmetic lowering (add/sub/mul/neg) | ~16 | Exhaustive (8-bit), Statistical (32/64-bit) | I8-I64 widths |
| Division (sdiv/udiv) | ~4 | Statistical (with preconditions) | Div-by-zero guard |
| Floating-point lowering | ~8 | Statistical | F32/F64 |
| NZCV flags | ~4 | Exhaustive (8-bit) | N, Z, C, V proofs |
| Comparison lowering | ~20 | Exhaustive (8-bit), Statistical (32/64-bit) | 10 conditions x 2 widths |
| Branch lowering | ~20 | Exhaustive/Statistical | Conditional branches |
| Peephole identities | ~54 | Exhaustive (8-bit), Statistical (32-bit) | |
| Optimization passes | ~28 | Exhaustive/Statistical | AND/OR absorb, identities |
| Constant folding | ~68 | Exhaustive/Statistical | Binary ops, unary, identities |
| Copy propagation | ~30 | Exhaustive/Statistical | |
| CSE/LICM | ~56 | Exhaustive/Statistical | |
| DCE | ~22 | Exhaustive/Statistical | |
| CFG simplification | ~32 | Exhaustive/Statistical | Branch folding, const folding |
| Memory (load/store) | ~91 | Exhaustive/Statistical | SMT array theory |
| NEON lowering | ~55 | Exhaustive/Statistical | Vector ops to NEON |
| Vectorization | ~45 | Exhaustive/Statistical | Scalar-to-NEON mapping |
| ANE precision | ~5 | Statistical | FP16 quantization error |
| Register allocation | ~16 | Structural | Non-interference, completeness |
| **Total** | **~570+** | | |

### 5.2 Uncovered Subsystems

- **Bitwise/shift lowering** (issue #180): AND, OR, XOR, SHL, AShr, LShr not proven
- **Address mode formation** (issue #183): ADD+LDR/STR folding not proven
- **Constant materialization** (issue #184): MOVZ/MOVK/MOVN not proven
- **Frame index elimination** (issue #185): stack offset computation not proven
- **Register allocation invariants** (issue #181): full regalloc correctness not proven
- **Binary encoding correctness**: no proofs that encoder produces correct bytes
- **Mach-O format correctness**: no proofs that object file is well-formed
- **End-to-end pipeline**: no `verify_function()` entry point (issue #186)
- **x86-64**: zero verification coverage
- **RISC-V**: zero verification coverage

### 5.3 Verification Strength Assessment

All proofs currently use mock evaluation:
- **8-bit proofs**: truly exhaustive (all 65,536 input pairs tested) -- sound
- **32/64-bit proofs**: statistical (36 edge cases + 100K random samples) -- high confidence but NOT formal
- **z4 integration**: SMT-LIB2 serialization works but z4 solver is not connected

**The verification gap is the single largest risk.** Statistical testing catches implementation bugs effectively but does not provide the formal guarantees that LLVM2's mission requires. Until z4 is integrated, the "verified compiler" claim is aspirational.

---

## 6. Technical Debt Inventory

### 6.1 Critical Debt

1. **22 unencoded AArch64 opcodes** (encode.rs:1324-1356): ISel can emit instructions that the encoder rejects. This means certain tMIR patterns will compile through ISel but fail at encoding.

2. **Type adapter duplication** (issue #73 remnant): `llvm2-regalloc` defines its own `MachFunction`/`MachInst`/`MachBlock` types in `machine_types.rs` rather than using `llvm2_ir` types. The pipeline converts between them via `From`/`TryFrom` impls, adding complexity and allocation overhead.

3. **z4 dependency commented out** (workspace Cargo.toml line 35): The verification solver is not available, making all proofs informal.

4. **ELF writer doctest failure**: New code with a broken doctest, indicating incomplete integration.

### 6.2 Moderate Debt

5. **Hardcoded RISC-V values** (target.rs lines 59, 68, 77, 86): Magic numbers `8`, `8`, `12`, `28` with TODO comments instead of proper register definitions.

6. **Instruction duplication**: `llvm2-codegen::lower.rs` (2,264 LOC) appears to contain lowering logic that overlaps with `llvm2-lower::isel.rs`. The role boundary is unclear.

7. **`tmir-semantics` stub is dead code**: Defined but not imported by any crate. Either integrate or remove.

8. **No integration between synthesis modules and build**: CEGIS, peephole synthesis, unified synthesis, and rule discovery (9,185 LOC total) are functional but isolated from the compilation pipeline.

9. **Cost model uncalibrated**: 2,660 LOC of latency/throughput data with no benchmarking harness to validate against actual hardware.

### 6.3 Low-Priority Debt

10. **Large files**: `isel.rs` (7,133 LOC) and `unified_synthesis.rs` (5,130 LOC) could benefit from splitting.

11. **Security issues #5-#21**: 17 security issues related to ai_template infrastructure, not LLVM2 code itself, but technically open.

12. **Test infrastructure duplication**: E2E test helpers (make_test_dir, write_c_driver, link_with_cc) are copy-pasted across multiple test files.

---

## 7. Top 10 Priorities for MVP (End-to-End Compile+Link+Run)

MVP definition: compile a non-trivial tMIR program (multiple functions, control flow, memory operations, function calls) to a working AArch64 binary.

| Priority | Item | Blocking? | Est. Effort |
|----------|------|-----------|-------------|
| **P1** | Encode remaining 22 AArch64 opcodes | **Yes** -- ISel emits them | Medium |
| **P2** | Wire `verify_function()` entry point (#186) | No -- MVP works without | Medium |
| **P3** | Fix ELF writer doctest failure | No -- Mach-O path works | Small |
| **P4** | Bitwise/shift lowering proofs (#180) | No -- lowering works, proofs missing | Medium |
| **P5** | Register allocation correctness proofs (#181) | No -- RA works, proofs missing | Large |
| **P6** | Multi-function compilation + linking | **Partial** -- single function works | Medium |
| **P7** | Struct/aggregate type support through pipeline | **Yes** for non-trivial programs | Large |
| **P8** | z4 solver integration | No for MVP, **Yes** for "verified" claim | Large |
| **P9** | x86-64 pipeline wiring (ISel -> encode -> ELF) | No for AArch64 MVP | Large |
| **P10** | Exception handling integration | No for MVP, Yes for production | Medium |

**Immediate MVP blockers (P1, P6, P7):** The 22 unencoded opcodes are the most critical because they represent instructions that ISel can generate but the encoder cannot emit. Multi-function compilation with cross-references (function calls between compiled functions) needs proper symbol/relocation support. Struct types require correct ABI lowering for aggregate arguments/returns.

---

## 8. Comparison with Plan Milestones

Based on CLAUDE.md goals and the original design documents:

### Achieved

- [x] AArch64 instruction selection (tree-pattern matching)
- [x] Apple AArch64 ABI lowering (AAPCS64)
- [x] 12+ verified optimization passes
- [x] Linear scan + greedy register allocation
- [x] AArch64 binary encoding (~80+ opcodes)
- [x] Mach-O object file emission
- [x] Frame lowering with compact unwind
- [x] DWARF CFI and debug info
- [x] Branch relaxation
- [x] End-to-end compile, link, run on Apple Silicon
- [x] 570+ proof obligations (mock evaluation)
- [x] Proof-guided optimization (tMIR proof annotations)
- [x] NEON auto-vectorization analysis
- [x] Cost model (CPU/NEON/GPU/ANE)
- [x] Heterogeneous compute dispatch planning
- [x] Metal MSL kernel emission
- [x] CoreML MIL emission for ANE
- [x] CEGIS-based synthesis
- [x] x86-64 ISel + encoding scaffolding
- [x] ELF object file writer (new)
- [x] Instruction scheduling (new)
- [x] Compare-and-branch fusion (new)
- [x] Provenance tracking + compilation trace

### In Progress

- [ ] z4 SMT solver integration (bridge exists, solver not connected)
- [ ] x86-64 full pipeline (ISel exists, not wired to encoding + ELF)
- [ ] Unified multi-target verification (synthesis framework exists, proofs isolated)
- [ ] Exception handling (module exists, not integrated)

### Not Started

- [ ] RISC-V target (issue #103, placeholder values only)
- [ ] Self-hosting (compiling tRust with LLVM2)
- [ ] Benchmark suite vs. LLVM/Cranelift
- [ ] Real tMIR integration (stubs only; depends on ayates_dbx/tMIR repo)
- [ ] Formal z4 proofs (all verification is statistical)
- [ ] Production ABI features: C++ interop, stack unwinding edge cases (#140)
- [ ] Varargs support
- [ ] TLS (Thread-Local Storage) beyond stub
- [ ] Dynamic linking / PIC code generation

### Goal Assessment

| Goal | Status |
|------|--------|
| Proven-correct instruction lowering from tMIR to machine code | **Partial** -- lowering works, proofs are statistical not formal |
| Verified optimizations | **Partial** -- passes work, proofs are statistical |
| Support AArch64 (primary), x86-64, RISC-V | **AArch64: Functional**, x86-64: Scaffolding, RISC-V: Not started |
| Universal backend for tRust, tSwift, tC | **Blocked** on real tMIR integration |
| At least as fast as LLVM in compilation speed | **Unknown** -- no benchmarks yet |
| At least as fast as LLVM in output code quality | **Unknown** -- no benchmarks yet |
| 100% Rust, zero external dependencies for core | **Yes** -- only serde, thiserror, anyhow |

---

## 9. Codebase Statistics Summary

| Metric | Value |
|--------|-------|
| Total Rust LOC (crates) | 145,908 |
| Total Rust LOC (stubs) | 746 |
| Source files (crates) | 143 |
| Source files (stubs) | 4 |
| Crates | 6 + 4 stubs |
| Tests (lib + integration) | ~3,837 |
| Proof obligations | ~570+ |
| Design documents | 28 |
| Open issues | 41 (17 security/template, 24 LLVM2-specific) |
| Build status | Clean (`cargo check` passes) |
| Test status | 1 doctest failure (ELF), 7 ignored (require AArch64 + cc) |

---

## 10. Recommendations

1. **Focus on encoding completeness** before adding new features. The 22 unencoded opcodes are the most dangerous gap because they create silent failures in the pipeline.

2. **Wire z4 integration** as the next major milestone. Without it, "verified compiler" is marketing, not engineering. The bridge code is ready; the gap is the z4 crate dependency and integration testing.

3. **Add a benchmark suite** to validate performance claims against LLVM and Cranelift. Without data, the "faster compilation" and "faster output" goals are unsubstantiated.

4. **Consolidate type adapters** (issue #73). The regalloc adapter layer adds complexity that obscures bugs and complicates the pipeline.

5. **Remove or integrate dead code**: `tmir-semantics` stub, synthesis modules not connected to pipeline, `lower.rs` overlap with `isel.rs`.

6. **Prioritize x86-64 pipeline wiring** over RISC-V. x86-64 has ~5K LOC of real code; RISC-V has nothing. Getting x86-64 to the "link and run" stage would demonstrate multi-target capability.
