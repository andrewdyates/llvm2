# LLVM2 Next Phase Design: Wave 7-8 Architecture Review and Roadmap

**Date:** 2026-04-13
**Author:** Andrew Yates <ayates@dropbox.com>
**Status:** Design
**Epic:** #24 (AArch64 Backend Implementation)
**Related:** designs/2026-04-12-aarch64-backend.md, reports/2026-04-13-wave4-report.md

---

## 1. Current State

LLVM2 is a verified AArch64 compiler backend with 43,595 lines of Rust across 6 crates. After 5 waves of factory development, the system can compile, link, and run AArch64 binaries on Apple Silicon.

### What works end-to-end

- 9-phase compilation pipeline: ISel, optimization (11 passes), linear scan register allocation, frame lowering, AArch64 encoding, Mach-O emission
- 4 passing E2E tests: return_const, add, sub, max (conditional branch)
- 904 unit/integration tests
- 26 SMT-verified lowering rules
- 5 proof-consuming optimizations unique to LLVM2 (NoOverflow, InBounds, NotNull, ValidBorrow, PositiveRefCount)
- Compilation throughput: 20.9M instructions/sec at O0 (encoding + emission only)

### Crate sizes

| Crate | LOC | Role |
|-------|-----|------|
| llvm2-codegen | ~15,000 | Encoder (40 opcodes), Mach-O writer, frame lowering, pipeline |
| llvm2-lower | ~5,600 | ISel (79 patterns), ABI, tMIR adapter |
| llvm2-opt | ~5,400 | 11 optimization passes + proof-consuming opts |
| llvm2-ir | ~4,500 | Shared machine model |
| llvm2-regalloc | ~4,000 | Linear scan RA + liveness + spill + phi elim + coalesce |
| llvm2-verify | ~2,600 | SMT encoding + 26 lowering proofs |
| tmir stubs | ~665 | Development stubs (tmir-types/instrs/func/semantics) |

---

## 2. Architectural Issues (Prioritized)

### 2.1 CRITICAL: Type Duplication (3 Independent Type Systems)

**Issue:** #73

The most severe architectural problem. Three crates define independent MachFunction, MachInst, MachOperand, and opcode types:

- `llvm2-lower` has its own AArch64Opcode (80+ variants, LLVM-style: ADDWrr, ADDXrr)
- `llvm2-ir` has its own AArch64Opcode (70+ variants, shorter: AddRR)
- `llvm2-regalloc` has its own MachInst (separate defs/uses fields)

This creates three adapter layers in `pipeline.rs` that translate between representations. The adapters are the primary source of correctness bugs:

- `map_isel_opcode()` has a catch-all `_ => IrOpcode::Nop` that silently drops conditional selects, bitfield operations, BIC/ORN logical ops
- `convert_isel_operand()` converts CondCode to `Imm(0)` and Symbol to `Imm(0)`, destroying condition code and call target information (#72)
- Adding a new instruction requires changes in 3 opcode enums + 3 adapter functions

**Recommendation:** Unify around `llvm2_ir` types. ISel should emit `llvm2_ir::MachInst` directly. RegAlloc should consume `llvm2_ir::MachFunction` with a def/use classification method on MachInst. Target: eliminate all adapter layers and the catch-all Nop fallback.

### 2.2 CRITICAL: Frame Lowering Encoding Mismatch

**Issues:** #66, #67

Frame lowering generates correct instruction sequences (STP/LDP pre-index for callee-save, ADD X29,SP for frame pointer) but the encoder does not support the required addressing modes:

- STP/LDP pre-index `[SP, #-16]!` encodes as signed-offset form, SP never adjusted
- MOV X29, SP encodes as `ORR X29, XZR, XZR` (= 0) because register 31 in ORR context is XZR, not SP

This means all non-leaf functions crash. The E2E tests bypass frame lowering to avoid this.

**Recommendation:** Fix the encoder to handle pre/post-index addressing modes and the MOV-from-SP special case (encode as `ADD X29, SP, #0` instead of `ORR`). This is the single highest-priority bug.

### 2.3 HIGH: CondCode/Symbol Operand Loss

**Issue:** #72

The `MachOperand` enum in `llvm2-ir` lacks CondCode and Symbol variants. During ISel-to-IR conversion, these operands are silently converted to `Imm(0)`. This means:

- All conditional branches (B.cond) lose their condition through the full pipeline
- All function calls (BL) lose their target symbol through the full pipeline

This is masked by E2E tests that build IR directly with pre-encoded immediate values for conditions.

**Recommendation:** Add `CondCode(u8)` and `Symbol(String)` to `MachOperand`. Update encoder to extract condition codes from the operand rather than relying on immediate encoding.

### 2.4 HIGH: Missing Instruction Encodings

The encoder covers ~40 opcodes but has gaps blocking common operations:

| Missing | Impact | Encoding format |
|---------|--------|-----------------|
| MUL/MADD | No multiplication (factorial test blocked) | Data-processing 3-source |
| SDIV/UDIV | No division | Data-processing 2-source |
| CSEL/CSET | No conditional select (idiomatic codegen) | Conditional select |
| STP/LDP pre/post-index | No non-leaf functions | Load/store pair pre/post |

Note: The MulRR encoding was added in the encoder (`encode.rs`) but the pipeline adapter maps MUL through `isel_to_ir()` which may not correctly propagate operands due to the type duplication issue (#73).

### 2.5 MEDIUM: Error Handling -- unreachable! Panics

**Issue:** #71

9 uses of `unreachable!()` in production code paths. These cause unrecoverable panics on any unexpected input rather than returning structured errors. Key locations: ISel (3), codegen/lower (1), frame (1), const_fold (4).

### 2.6 MEDIUM: No Aggregate Type Support

**Issue:** #74

The tMIR adapter rejects all struct, array, and field operations. No real-world program can be compiled without at least basic aggregate support (struct field access, array indexing, pass-by-value ABI).

### 2.7 MEDIUM: Optimization Passes Unverified

**Issue:** #75

11 optimization passes with zero formal proofs. The verification chain has a gap: lowering rules are proven (26 proofs) but optimizations could introduce miscompilation. Only 3 of 14 peephole patterns have SMT proofs.

### 2.8 LOW: Compiler Warnings

6 warnings in `llvm2-codegen` from unreachable patterns in `lower.rs`. These indicate dead code from opcode additions (#296) that didn't fully update all match arms.

---

## 3. Test Coverage Assessment

### Strengths

- **904 total tests** across 6 crates
- **4,529 lines of dedicated test code** in `tests/` directories
- **4 E2E tests** that compile, link, and execute on real hardware
- **Comprehensive encoding tests** (1,354 lines in `aarch64_encoding.rs`)
- **tMIR integration tests** (885 lines covering all tMIR instruction types)

### Gaps

1. **No full-pipeline E2E tests through ISel**: All passing E2E tests build IR directly, bypassing ISel and the adapter layers. The adapter bugs (#72) are therefore invisible to testing.

2. **No negative tests for encoding**: The encoder is tested with valid inputs only. No tests verify correct error behavior for out-of-range immediates, invalid register classes, or unsupported addressing modes.

3. **No fuzz testing**: The encoder, adapter, and ISel have no fuzz harness. Given the extensive bit manipulation in AArch64 encoding, fuzz testing would catch edge cases efficiently.

4. **No comparison testing against LLVM/Clang**: The project has no test that compiles the same function with both LLVM2 and Clang and compares output semantics. This is the gold standard for compiler testing.

5. **Register allocator stress tests are minimal**: Only 5 regalloc integration tests (straight-line, diamond, loop, high-pressure, call). No tests for register pressure with mixed GPR/FPR, no tests for callee-saved register spilling, no tests with complex control flow.

6. **Optimization pass testing is unit-level only**: Each pass has unit tests but there are no tests verifying that the optimization pipeline as a whole preserves semantics across multiple interacting passes.

---

## 4. Recommended Wave 7-8 Targets

### Wave 7: Fix Critical Infrastructure (estimated: 10-15 issues)

The goal of Wave 7 is to fix the foundational issues that block correct compilation through the full pipeline. No new features.

#### 7A. Fix Frame Lowering Encoding (P1, #66 + #67)
- Implement STP/LDP pre/post-index encoding in `encode.rs`
- Implement MOV-from-SP as `ADD Xd, SP, #0` special case
- Enable frame lowering in E2E tests
- Verify: compile and run a non-leaf function (e.g., function that calls another function)

#### 7B. Add CondCode + Symbol Operands to IR (P1, #72)
- Add `CondCode(u8)` and `Symbol(String)` to `MachOperand`
- Update ISel adapter to preserve these operands
- Update encoder to consume CondCode from operand
- Update Mach-O writer to emit relocations for Symbol operands

#### 7C. Fix ISel-to-IR Opcode Fallthrough (P1, part of #73)
- Replace `_ => IrOpcode::Nop` with exhaustive match
- Add missing IR opcodes: CSEL, CSET, CSINC, BIC, ORN, UBFM, SBFM
- Every ISel opcode must map to a real IR opcode or return error

#### 7D. Complete Encoding Gaps (P1, part of #27)
- MUL (MADD with Ra=XZR) -- already partially done, verify through full pipeline
- SDIV/UDIV (Data-processing 2-source format)
- CSEL/CSET (Conditional select format)
- MOVK (Move keep, for 64-bit constant materialization)

#### 7E. Error Handling Hardening (P2, #71)
- Replace all unreachable!() with Result types
- Add structured error types to ISel, encoder, and optimizer
- Enable cargo clippy and fix all warnings

#### 7F. Full-Pipeline E2E Test Suite (P1)
- Write E2E tests that go through ISel (not direct IR construction)
- Test: add(i32,i32)->i32, max(i32,i32)->i32, factorial(i32)->i32, fibonacci(i32)->i32
- Compare output against Clang -O0 for same function

### Wave 8: Expand Capability (estimated: 10-15 issues)

The goal of Wave 8 is to extend LLVM2 to handle a useful subset of real programs.

#### 8A. Type Unification (P1, #73)
- Unify ISel to emit llvm2-ir types directly
- Unify RegAlloc to consume llvm2-ir types directly
- Delete adapter layers from pipeline.rs
- This is the largest refactor and should be done as a dedicated effort

#### 8B. Basic Aggregate Support (P2, #74)
- Struct field access (load/store with constant offset)
- Array indexing (load/store with scaled index)
- Small struct pass-by-value (AAPCS64: <=16 bytes in registers)
- Large struct pass-by-pointer (sret)

#### 8C. Verification of Peephole Rules (P2, #75)
- Prove remaining 11 peephole rules via SMT
- Prove constant folding rules for each bitwidth
- Add copy propagation soundness proof
- Target: 40+ total proven rules (up from 26)

#### 8D. Proof Propagation Through Pipeline (P2, #64)
- Proofs currently live on MachInst but are not tracked through optimization
- Implement proof metadata passing: ISel attaches proofs, optimizer reads/transforms them
- Proof-consuming optimizations should verify preconditions before eliminating checks

#### 8E. Greedy Register Allocator (P2, #62)
- Linear scan produces acceptable but suboptimal code
- Greedy RA with live range splitting for better allocation quality
- Only needed when code quality matters (O2/O3)

---

## 5. Risk Areas (Where Correctness Is Most Fragile)

### 5.1 Adapter Layers (Highest Risk)

Every data conversion between crate-specific types is an opportunity for silent data loss. The CondCode/Symbol loss (#72) and the Nop fallback in opcode mapping are evidence. Until type unification (#73) is complete, every new instruction must be added in 3 places or it silently compiles to NOP.

**Mitigation:** Wave 7C eliminates the catch-all Nop. Wave 8A eliminates the adapter layers entirely.

### 5.2 AArch64 Encoding Correctness

The encoder does extensive bit manipulation to produce 32-bit instruction words. A single bit error produces a valid but wrong instruction (AArch64's fixed-width encoding means most bit patterns decode to something). The current test strategy is "encode and compare against known-good bytes" which is effective but requires correct reference values.

**Mitigation:** Differential testing against `llvm-objdump --disassemble` would catch encoding errors by round-tripping: encode with LLVM2, disassemble with LLVM, verify the disassembly matches the intended instruction.

### 5.3 Register Allocation Liveness

Linear scan depends on correct liveness intervals. If liveness analysis misses a live-out register (e.g., across a call or branch), the allocator may assign the same physical register to two live ranges, causing incorrect code. The current liveness analysis has limited testing for complex control flow.

**Mitigation:** Add stress tests with complex CFG patterns (nested loops, exception-like paths, critical edges). Compare register allocation output against a reference allocator.

### 5.4 Frame Lowering Stack Offsets

Stack offset calculation affects every memory access to spilled values, local variables, and function arguments. An off-by-one error in frame index elimination produces code that reads/writes the wrong stack slot, causing data corruption that may not manifest until much later.

**Mitigation:** The frame lowering has stack alignment assertions but needs tests that verify actual memory layout by compiling functions that pass data through the stack and checking values survive.

### 5.5 Optimization Interaction

Individual optimization passes may be correct in isolation but produce incorrect code when composed. Example: LICM hoists a load above a store; if the effects model incorrectly classifies the store as non-aliasing, LICM introduces a use-before-def. The effects model (`llvm2-opt/src/effects.rs`) is the safety net for all transformations.

**Mitigation:** End-to-end optimization pipeline tests with known-correct inputs/outputs. Verify that `O0` and `O2` produce the same observable behavior for a suite of test functions.

---

## 6. Dependency Map

```
                          tMIR (external)
                               |
                      llvm2-lower (ISel, ABI)
                               |
              [adapter -- ELIMINATE in Wave 8A]
                               |
                       llvm2-ir (shared model)
                      /    |    \        \
             llvm2-opt  llvm2-regalloc  llvm2-verify
                      \    |
              [adapter -- ELIMINATE in Wave 8A]
                               |
                     llvm2-codegen (encode, Mach-O)
                               |
                        .o file (AArch64)
```

The critical path for Wave 7 is: fix encoder (#66, #67) -> fix operands (#72) -> fix opcode mapping (#73 partial) -> full-pipeline E2E tests.

The critical path for Wave 8 is: type unification (#73) -> aggregate support (#74) -> verification (#75).

---

## 7. Summary

LLVM2 has made remarkable progress in 5 waves: a working AArch64 backend with 43K lines of Rust, a 9-phase compilation pipeline, and 26 SMT-verified lowering rules. The architecture is sound and the crate separation is clean.

The three most important things to fix, in order:

1. **Frame lowering encoding** (#66, #67) -- unblocks non-leaf functions
2. **CondCode/Symbol operand loss** (#72) -- unblocks correct conditional branches and function calls through the full pipeline
3. **Type duplication** (#73) -- eliminates the primary source of correctness bugs and the main obstacle to adding new instructions

Once these three are resolved, LLVM2 can compile a meaningful subset of real programs through the full tMIR-to-binary pipeline with formal verification of every lowering rule.
