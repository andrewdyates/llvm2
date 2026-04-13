# LLVM2 Gap Analysis: Test Coverage, Encoder, ISel, and Next-Phase Planning

**Date:** 2026-04-13
**Author:** Andrew Yates <ayates@dropbox.com>
**Baseline:** Wave 8, 1,119 tests passing, ~46K Rust LOC (source + tests)
**Part of #24** (AArch64 Backend Epic)

---

## 1. Test Coverage Gap Analysis

### 1.1 Test Distribution by Module

| Module Area | Test Count | Assessment |
|-------------|-----------|------------|
| aarch64 (encoding) | 168 | Excellent -- all opcodes covered |
| macho (object writer) | 55 | Good -- includes roundtrips |
| isel (instruction selection) | 47 | Good -- covers all ISel patterns |
| inst (MachInst model) | 41 | Good -- flags, construction, proofs |
| lowering_proof | 34 | Solid -- 56+ SMT-verified rules |
| function (MachFunction) | 32 | Adequate |
| memory_model | 30 | Good -- store/load roundtrips |
| adapter (tMIR) | 29 | Adequate -- basic translations |
| operand | 27 | Good |
| frame (prologue/epilogue) | 27 | Adequate |
| opt_proofs | 23 | Good -- identity proofs |
| nzcv (flag semantics) | 21 | Good |
| lower (codegen lowering) | 21 | Adequate |
| peephole_proofs | 20 | Good |
| proof_opts | 19 | Good |
| cse | 15 | Adequate |
| addr_mode | 15 | Good |
| greedy (regalloc) | 13 | **Weak -- complex allocator, few tests** |
| relax (branch relaxation) | 12 | Adequate |
| cfg_simplify | 11 | Adequate |
| smt | 11 | Adequate |
| dom (dominator tree) | 10 | Adequate |
| const_fold | 10 | Adequate |
| loops | 7 | **Weak -- only basic loop detection** |
| split (interval splitting) | 7 | **Weak** |
| remat (rematerialization) | 7 | **Weak** |
| effects | 7 | Minimal |
| dce | 7 | Minimal |
| licm | 6 | **Weak -- only 6 tests for 558 LOC** |
| layout | 6 | Minimal |
| pass_manager | 5 | Minimal |
| coalesce | 5 | **Weak -- 467 LOC, only 5 tests** |
| copy_prop | 4 | **Weak** |
| call_clobber | 4 | Minimal |
| spill_slot_reuse | 4 | Minimal |
| pipeline (codegen) | 4 | Minimal |
| liveness | 3 | **Critical gap -- 409 LOC, only 3 tests** |
| phi_elim | 2 | **Critical gap -- 318 LOC, only 2 tests** |
| linear_scan | 2 | **Critical gap -- 318 LOC, only 2 tests** |
| spill | 1 | **Critical gap -- 179 LOC, only 1 test** |

### 1.2 Critical Test Coverage Gaps

**Priority 1: Register Allocation (llvm2-regalloc)**

The register allocation subsystem has 5,079 LOC across 12 modules but only 57 total tests. Several critical modules are severely under-tested:

- `liveness.rs` (409 LOC, 3 tests): Liveness analysis is foundational for RA. Bugs here cause silent miscompilation. Needs: tests for live-range computation across basic blocks, phi node handling, call-crossing intervals, variable-length live ranges.
- `linear_scan.rs` (318 LOC, 2 tests): The primary allocator has only 2 tests. Needs: tests for register pressure, spill decisions, interference graph, callee-saved register handling.
- `phi_elim.rs` (318 LOC, 2 tests): SSA deconstruction is error-prone. Needs: tests for parallel copy insertion, lost-copy problem, swap problem, critical edge splitting.
- `spill.rs` (179 LOC, 1 test): Spill generation with a single test. Needs: tests for spill slot allocation, reload placement, rematerialization preference.
- `coalesce.rs` (467 LOC, 5 tests): Copy coalescing. Needs: tests for aggressive vs conservative coalescing, interference detection.
- `split.rs` (430 LOC, 7 tests): Interval splitting. Needs: tests for split point selection, split-everywhere vs split-at-use.

**Priority 2: Optimization Passes (llvm2-opt)**

The optimization subsystem has 7,149 LOC across 15 modules with 122 tests. Key gaps:

- `licm.rs` (558 LOC, 6 tests): Loop-invariant code motion with very few tests. Needs: tests for hoisting stores (should NOT hoist), hoisting loads (depends on aliasing), multi-level loop nests, critical edge handling.
- `copy_prop.rs` (263 LOC, 4 tests): Needs more tests for chain propagation and phi-node copy propagation.
- `dce.rs` (321 LOC, 7 tests): Dead code elimination. Needs: tests for side-effect preservation, chain elimination.
- `loops.rs` (636 LOC, 7 tests): Loop analysis. Needs: tests for irreducible control flow, loop depth computation, exit block detection.

**Priority 3: Missing E2E Test Scenarios**

Current E2E tests cover:
- [x] Single-block function (return_const, add, sub)
- [x] Multi-block with conditional branch (max)
- [x] Loop with MUL (factorial)
- [x] Multiple functions (multi_func)
- [x] Full pipeline with frame lowering

Missing E2E scenarios:
- [ ] **Function calls** -- no E2E test exercises BL/BLR with actual callee
- [ ] **Stack spills** -- no E2E test with register pressure forcing spills
- [ ] **Floating-point** -- no E2E test compiles/runs FP arithmetic
- [ ] **Nested loops** -- no E2E test with multi-level loop nests
- [ ] **Switch/multi-way branch** -- no E2E test for computed branches
- [ ] **Global variable access** -- no E2E test with ADRP+ADD pattern
- [ ] **Struct/aggregate** -- blocked on #74 (aggregate type support)

### 1.3 Error Path Testing

Error path tests exist for:
- Encoding boundary errors (imm12, imm7, imm9, imm21 overflow)
- FP register overflow
- Unsupported type errors in adapter
- Fixup overflow
- Relocation encode/decode errors

Missing error path tests:
- ISel encountering unsupported tMIR instructions (returns ISelError)
- Encoder receiving malformed operands (e.g., wrong operand count)
- Register allocator running out of registers (should spill, not crash)
- Frame lowering with invalid stack alignment
- Pipeline encountering unsupported opcode sequences

---

## 2. Encoder Coverage Audit

### 2.1 AArch64Opcode Enum (inst.rs) vs Encoder (encode.rs)

All 64 opcodes in the `AArch64Opcode` enum have encoder coverage. The test `test_all_opcodes_handled` in encode.rs exhaustively verifies every variant is dispatched.

**Complete encoder coverage:**

| Category | Opcodes | Count | Encoded |
|----------|---------|-------|---------|
| Arithmetic | AddRR, AddRI, SubRR, SubRI, MulRR, Msub, Smull, Umull, SDiv, UDiv, Neg | 11 | All |
| Logical | AndRR, OrrRR, EorRR | 3 | All |
| Shift (reg) | LslRR, LsrRR, AsrRR | 3 | All |
| Shift (imm) | LslRI, LsrRI, AsrRI | 3 | All |
| Compare | CmpRR, CmpRI, Tst | 3 | All |
| Move | MovR, MovI, Movz, Movk | 4 | All |
| Memory | LdrRI, StrRI, LdrLiteral, LdpRI, StpRI, StpPreIndex, LdpPostIndex | 7 | All |
| Branch | B, BCond, Cbz, Cbnz, Tbz, Tbnz, Br, Bl, Blr, Ret | 10 | All |
| Extension | Sxtw, Uxtw, Sxtb, Sxth | 4 | All |
| FP Arith | FaddRR, FsubRR, FmulRR, FdivRR | 4 | All |
| FP Other | Fcmp, FcvtzsRR, ScvtfRR | 3 | All |
| Address | Adrp, AddPCRel | 2 | All |
| Checked | AddsRR, AddsRI, SubsRR, SubsRI | 4 | All |
| Trap/Pseudo | TrapOverflow, TrapBoundsCheck, TrapNull, Retain, Release, Phi, StackAlloc, Nop | 8 | All (NOP/BRK) |
| **Total** | | **64** | **64/64 (100%)** |

### 2.2 Encoding Gaps (Not Opcodes, But Missing Functionality)

While all opcodes are handled, the encoder has functional gaps:

1. **32-bit mode (sf=0) is largely untested.** The `sf_from_operand()` helper always returns 1 (64-bit). 32-bit register operations (W-register) will produce incorrect encodings. This affects: AddRR/AddRI/SubRR/SubRI/MulRR with W-register operands, logical ops, shifts.

2. **Shifted register forms are unused.** ADD/SUB shifted register encoding accepts shift type and amount, but the encoder always passes `shift_type=0, shift_amount=0`. No instruction pattern generates `ADD X0, X1, X2, LSL #3`.

3. **Logical immediate encoding is missing.** There is no `encode_logical_imm()` for AND/ORR/EOR with immediate operands. Only shifted-register forms exist in the encoder. The ISel generates `ANDWri`/`ANDXri`/etc. but these have no path to the encoder.

4. **FP size is hardcoded to Double.** The `fp_size_from_inst()` helper always returns `FpSize::Double`. Single-precision (F32) FP operations will produce wrong encodings. This affects: FaddRR, FsubRR, FmulRR, FdivRR, Fcmp, FcvtzsRR, ScvtfRR with single-precision operands.

5. **No MOVN (move-wide with NOT) encoding path.** ISel generates `MOVNWi`/`MOVNXi` for negative immediates, but the canonical IR `AArch64Opcode` enum has no Movn variant. These get lost in the ISel-to-IR adapter.

6. **No CSEL/CSINC/CSINV/CSNEG encoding.** ISel generates these (CSELWr, CSINCWr, etc.) but the canonical IR has no corresponding opcodes and the encoder has no match arms for them.

---

## 3. ISel Pattern Coverage

### 3.1 ISel Patterns Implemented

The ISel (`isel.rs`) handles the following LIR opcodes via `select_instruction()`:

| LIR Opcode | ISel Pattern | AArch64 Output |
|-----------|-------------|----------------|
| `Iconst` | `select_iconst` | MOVZ/MOVN/MOVZ+MOVK |
| `Fconst` | `select_fconst` | FMOV |
| `Iadd` | `select_binop(Add)` | ADDWrr/ADDXrr |
| `Isub` | `select_binop(Sub)` | SUBWrr/SUBXrr |
| `Imul` | `select_binop(Mul)` | MULWrrr/MULXrrr |
| `Sdiv` | `select_binop(Sdiv)` | SDIVWrr/SDIVXrr |
| `Udiv` | `select_binop(Udiv)` | UDIVWrr/UDIVXrr |
| `Ishl` | `select_shift(Lsl)` | LSLVWr/LSLVXr or LSLWi/LSLXi |
| `Ushr` | `select_shift(Lsr)` | LSRVWr/LSRVXr or LSRWi/LSRXi |
| `Sshr` | `select_shift(Asr)` | ASRVWr/ASRVXr or ASRWi/ASRXi |
| `Band` | `select_logic(And)` | ANDWrr/ANDXrr |
| `Bor` | `select_logic(Orr)` | ORRWrr/ORRXrr |
| `Bxor` | `select_logic(Eor)` | EORWrr/EORXrr |
| `BandNot` | `select_logic(Bic)` | BICWrr/BICXrr |
| `BorNot` | `select_logic(Orn)` | ORNWrr/ORNXrr |
| `Sextend` | `select_extend` | SXTB/SXTH/SXTW |
| `Uextend` | `select_extend` | UXTB/UXTH/MOV |
| `ExtractBits` | `select_bitfield_extract` | UBFM |
| `SextractBits` | `select_bitfield_extract` | SBFM |
| `InsertBits` | `select_bitfield_insert` | BFM |
| `Select` | `select_csel` | CMP + CSEL |
| `Icmp` | `select_cmp` | CMP + CSET |
| `Fadd` | `select_fp_binop(Fadd)` | FADD |
| `Fsub` | `select_fp_binop(Fsub)` | FSUB |
| `Fmul` | `select_fp_binop(Fmul)` | FMUL |
| `Fdiv` | `select_fp_binop(Fdiv)` | FDIV |
| `Fcmp` | `select_fcmp` | FCMP + CSET |
| `FcvtToInt` | `select_fcvt_to_int` | FCVTZS |
| `FcvtFromInt` | `select_fcvt_from_int` | SCVTF |
| `GlobalRef` | `select_global_ref` | ADRP + ADD |
| `StackAddr` | `select_stack_addr` | ADD SP, #offset |
| `Jump` | `select_jump` | B |
| `Brif` | `select_brif` | CMP + B.cond + B |
| `Return` | `select_return` | MOV to ret regs + RET |
| `Call` | `select_call_from_lir` | MOV args + BL + MOV rets |
| `Load` | `select_load` | LDR |
| `Store` | `select_store` | STR |

**Total: 36 LIR opcodes handled** across arithmetic, logic, shifts, comparisons, FP, control flow, memory, and addressing.

### 3.2 tMIR Instructions with No ISel Lowering

Cross-referencing `tmir-instrs/src/lib.rs` (the tMIR instruction set) against the LIR opcodes and ISel patterns:

| tMIR Instruction | Status | Notes |
|-----------------|--------|-------|
| `BinOp::SRem` | **No ISel** | Signed remainder: needs SDIV + MSUB pattern |
| `BinOp::URem` | **No ISel** | Unsigned remainder: needs UDIV + MSUB pattern |
| `UnOp::Neg` | **No ISel** | Integer negation: needs SUB from XZR |
| `UnOp::Not` | **No ISel** | Bitwise NOT: needs ORN with XZR or MVN |
| `UnOp::FNeg` | **No ISel** | Float negation: needs FNEG |
| `CastOp::ZExt` | Partial | Via `Uextend` but not all width combos |
| `CastOp::SExt` | Partial | Via `Sextend` but not all width combos |
| `CastOp::Trunc` | **No ISel** | Integer truncation (typically a no-op on AArch64 but needs masking for correctness) |
| `CastOp::FPToUI` | **No ISel** | Float to unsigned integer: needs FCVTZU |
| `CastOp::UIToFP` | **No ISel** | Unsigned int to float: needs UCVTF |
| `CastOp::FPExt` | **No ISel** | Float precision extension: needs FCVT S->D |
| `CastOp::FPTrunc` | **No ISel** | Float precision truncation: needs FCVT D->S |
| `CastOp::PtrToInt` | **No ISel** | Pointer to integer (no-op on AArch64) |
| `CastOp::IntToPtr` | **No ISel** | Integer to pointer (no-op on AArch64) |
| `CastOp::Bitcast` | **No ISel** | Bitcast (may need FMOV for int<->fp) |
| `Instr::Alloc` | **No ISel** | Stack allocation: needs frame lowering |
| `Instr::Dealloc` | **No ISel** | Deallocation hint |
| `Instr::Borrow` | **No ISel** | Immutable borrow (tMIR-specific) |
| `Instr::BorrowMut` | **No ISel** | Mutable borrow (tMIR-specific) |
| `Instr::EndBorrow` | **No ISel** | End borrow lifetime |
| `Instr::Retain` | Partial | ISel opcode exists but no tMIR adapter |
| `Instr::Release` | Partial | ISel opcode exists but no tMIR adapter |
| `Instr::IsUnique` | **No ISel** | Uniqueness check |
| `Instr::Switch` | **No ISel** | Multi-way branch: needs jump table or chain |
| `Instr::CallIndirect` | **No ISel** | Indirect call: needs BLR |
| `Instr::Struct` | **No ISel** | Blocked on #74 |
| `Instr::Field` | **No ISel** | Blocked on #74 |
| `Instr::Index` | **No ISel** | Array indexing |
| `Instr::Phi` | Partial | ISel PHI pseudo exists but tMIR phi lowering unclear |

**Summary:** 20+ tMIR instructions have no ISel lowering path. The most important gaps for compiling real programs are: SRem/URem (integer remainder), Trunc/FPToUI/UIToFP/FPExt/FPTrunc (type conversions), Switch (multi-way branches), CallIndirect (function pointers), and Neg/Not/FNeg (unary operations).

---

## 4. Recommended Wave 9-10 Work Items

### Priority 1 (Blocks real-world compilation)

1. **Integer remainder (SRem/URem):** Implement as SDIV+MSUB/UDIV+MSUB pattern. Required for any program using `%` operator. ~50 LOC ISel + encoder + tests.

2. **Unary operations (Neg, Not, FNeg):** Add LIR opcodes, ISel patterns, and encoder support. Neg = SUB from XZR (already in encoder), Not = ORN with XZR (new encoder arm), FNeg = FNEG encoding (new opcode in ir/inst.rs + encoder).

3. **Type conversions (Trunc, FPToUI, UIToFP, FPExt, FPTrunc, Bitcast):** Add 6 LIR opcodes and ISel patterns. Critical for any program with mixed-width integers or float/int conversions. ~200 LOC.

4. **Switch/multi-way branch:** Add LIR opcode and ISel lowering to either jump table (for dense cases) or comparison chain (for sparse cases). Required for match/switch statements. ~150 LOC.

5. **Indirect call (CallIndirect):** Add BLR-based ISel pattern. Required for function pointers and vtable dispatch. ~50 LOC.

### Priority 2 (Correctness and quality)

6. **Fix 32-bit encoding (sf=0):** The encoder's `sf_from_operand()` always returns 1. This means 32-bit arithmetic silently produces 64-bit instructions. Add register-class-aware size detection. ~50 LOC + tests.

7. **Fix FP size detection:** FP size helpers always return Double. Single-precision FP operations produce wrong encodings. Add operand-type-aware size detection. ~30 LOC + tests.

8. **CSEL/CSINC/CSNEG encoder support:** ISel generates these but they cannot reach the encoder. Add 6 new `AArch64Opcode` variants and encoder arms. ~100 LOC.

9. **Register allocator tests:** Add 30+ tests for liveness, phi_elim, linear_scan, spill, and coalesce. Target the critical gaps identified in Section 1.2. ~500 LOC of tests.

10. **E2E test expansion:** Add function-call, FP arithmetic, and stack-spill E2E tests. ~300 LOC.

### Priority 3 (Architecture and debt)

11. **Logical immediate encoding:** ANDri/ORRri/EORri ISel patterns have no encoding path. Add bitmask immediate encoder per ARM ARM. ~200 LOC.

12. **MOVN encoding path:** ISel generates MOVN for negative immediates but the canonical IR has no Movn opcode. Either add it or ensure the adapter translates MOVN to equivalent sequences.

13. **Type unification (#73):** Eliminate the 3 adapter layers. This is the largest architectural debt and primary bug source.

---

## 5. Issues to File

Based on this analysis, the following new issues are recommended:

1. **P1: Integer remainder (SRem/URem) lowering missing** -- Blocks `%` operator. Implement SDIV+MSUB pattern.

2. **P2: Unary operations (Neg/Not/FNeg) missing from tMIR-to-ISel path** -- Three unary operations have no lowering.

3. **P2: Type conversion instructions missing (Trunc, FPToUI, UIToFP, FPExt, FPTrunc, Bitcast)** -- Six cast operations have no ISel patterns.

4. **P2: 32-bit encoding broken -- sf_from_operand always returns 1** -- All W-register operations silently produce X-register encodings.

5. **P2: Register allocator critically under-tested (57 tests for 5,079 LOC)** -- liveness (3 tests), phi_elim (2), linear_scan (2), spill (1).

---

## Appendix: Test Count Methodology

Test counts derived from `cargo test --workspace 2>&1 | grep "^test "` output (1,119 test lines as of this analysis). Module attribution based on test name prefixes. Some tests appear in integration test files and are counted under the nearest module prefix.
