# LLVM2 Memory Model Verification: Status and Path to z4

**Date:** 2026-04-14
**Author:** Andrew Yates <ayates@dropbox.com>
**Status:** Draft
**Part of:** #142

---

## Overview

This document assesses the current state of memory model verification in
`llvm2-verify`, identifies what is missing, and charts a path from the current
mock-evaluation proofs to full z4-backed formal verification using array theory.

---

## Current Implementation: Two Complementary Modules

LLVM2 has **two** memory verification modules that prove overlapping but
complementary properties. Together they provide 27 + 17 = 44 proof obligations
covering Load/Store lowering from tMIR to AArch64.

### Module 1: `memory_proofs.rs` -- Symbolic SMT Array Encoding (27 proofs)

**Source:** `crates/llvm2-verify/src/memory_proofs.rs`

This module builds **symbolic SMT expression trees** using the array theory
(QF_ABV). Memory is modeled as `Array(BitVec64, BitVec8)` -- a byte-addressable
array indexed by 64-bit addresses with 8-bit byte elements. Multi-byte accesses
use little-endian byte ordering.

The proofs are verified by concrete evaluation of the symbolic expression trees
via `verify_by_evaluation()` (exhaustive for small widths, random sampling for
32/64-bit). The SMT-LIB2 serialization is already implemented and tested --
the `to_smt2()` method emits valid QF_ABV queries that have been validated
against z3 in integration tests.

**Proof categories:**

| Category | Count | What it proves |
|----------|-------|----------------|
| Load equivalence | 6 | tMIR Load at various sizes/offsets == AArch64 LDR |
| Store equivalence | 6 | tMIR Store at various sizes/offsets == AArch64 STR |
| Store-load roundtrip | 4 | Store then load at same address returns stored value |
| Non-interference | 8 | Store at addr A does not affect load at addr B (disjoint) |
| Endianness | 3 | Little-endian byte ordering: LSB at lowest address |

**Detailed proof obligations:**

**Load equivalence (6):**
1. `Load_I8 -> LDRBui [Xn, #0]` -- 8-bit load, zero offset
2. `Load_I16 -> LDRHui [Xn, #0]` -- 16-bit load, zero offset
3. `Load_I32 -> LDRWui [Xn, #0]` -- 32-bit load, zero offset
4. `Load_I32 -> LDRWui [Xn, #4]` -- 32-bit load, scaled offset 4 (byte offset 16)
5. `Load_I64 -> LDRXui [Xn, #0]` -- 64-bit load, zero offset
6. `Load_I64 -> LDRXui [Xn, #3]` -- 64-bit load, scaled offset 3 (byte offset 24)

**Store equivalence (6):**
7. `Store_I8 -> STRBui [Xn, #0]`
8. `Store_I16 -> STRHui [Xn, #0]`
9. `Store_I32 -> STRWui [Xn, #0]`
10. `Store_I32 -> STRWui [Xn, #2]` (byte offset 8)
11. `Store_I64 -> STRXui [Xn, #0]`
12. `Store_I64 -> STRXui [Xn, #2]` (byte offset 16)

**Store-load roundtrip (4):**
13-16. Roundtrip for I8, I16, I32, I64

**Non-interference (8):**
17. Store I32 at A, load I32 at A+8 unchanged
18. Store I64 at A, load I64 at A+16 unchanged
19. Store I32 at A, load I32 at A+4 unchanged (adjacent)
20. Store I64 at A, load I64 at A+8 unchanged (adjacent)
21. Store I64 at A, load I32 at A+8 unchanged (cross-size)
22. Store I32 at A, load I64 at A+4 unchanged (cross-size)
23. Non-interference I32, symbolic gap (strongest form, arbitrary disjoint A and B)
24. Non-interference I64, symbolic gap (strongest form)

**Endianness (3):**
25. I32: byte[0] == value[7:0] (LSB first)
26. I32: byte[3] == value[31:24] (MSB last)
27. I64: byte[0] == value[7:0] (LSB first)

**Negative tests (4, not counted in proof obligations):**
- Overlapping I32 partial (gap=2, size=4) detected as Invalid
- Overlapping I64 partial (gap=4, size=8) detected as Invalid
- Cross-size overlap (store I64, load I32 at +6) detected as Invalid
- Single-byte boundary overlap (store I32, load I32 at +3) detected as Invalid

### Module 2: `memory_model.rs` -- Concrete HashMap Evaluation (17 proofs)

**Source:** `crates/llvm2-verify/src/memory_model.rs`

This module uses a concrete `SmtMemory` struct (HashMap<u64, u8>) for
byte-level memory simulation. Proofs are verified by random sampling over
50,000 trials plus edge cases.

**Proof categories:**

| Category | Count | What it proves |
|----------|-------|----------------|
| Load equivalence | 4 | I32 and I64, with and without offsets |
| Store equivalence | 4 | I32 and I64, with and without offsets |
| Store-load roundtrip | 2 | I32 and I64 |
| Non-interference | 2 | Store at A, load at B unchanged |
| Load pair (LDP) | 3 | Two adjacent loads == AArch64 LDP |
| Alignment | 2 | Natural alignment check |

The `memory_model.rs` module covers **LDP** (load pair) and **alignment** which
`memory_proofs.rs` does not, while `memory_proofs.rs` covers **I8/I16 sizes**,
**cross-size interference**, **symbolic gap non-interference**, and
**endianness** which `memory_model.rs` does not.

---

## Pipeline Integration

Memory verification is fully wired into the verification pipeline:

```
Verifier::verify_comprehensive()
  -> verify_arithmetic()          -- 5 proofs
  -> verify_nzcv()                -- 21 proofs
  -> verify_peephole()            -- 11+ proofs
  -> verify_memory_model()        -- 27 proofs (from memory_proofs.rs)
  -> verify_optimizations()       -- const fold, copy prop, CSE/LICM, DCE, CFG
```

The `verify_memory_model()` method in `verify.rs` calls
`memory_proofs::all_memory_proofs()` and runs each through
`verify_by_evaluation()`. The comprehensive verifier reports per-category
pass/fail counts.

---

## What is Missing

### Missing Memory Operations

| Operation | Status | Priority | Notes |
|-----------|--------|----------|-------|
| LDRB/STRB (byte) | DONE | -- | 8-bit load/store in both modules |
| LDRH/STRH (halfword) | DONE | -- | 16-bit in memory_proofs.rs |
| LDR/STR (word) | DONE | -- | 32-bit in both modules |
| LDR/STR (doubleword) | DONE | -- | 64-bit in both modules |
| LDP/STP (load/store pair) | PARTIAL | P2 | LDP in memory_model.rs, STP not verified |
| LDRSB/LDRSH/LDRSW (sign-extend) | MISSING | P2 | Sign-extending loads not modeled |
| LDR (register offset) | MISSING | P2 | `[Xn, Xm]` addressing mode |
| LDR (pre/post-index) | MISSING | P3 | `[Xn, #imm]!` and `[Xn], #imm` |
| LDR (literal/PC-relative) | MISSING | P3 | Used for constant pools |
| LDXR/STXR (exclusive) | MISSING | P3 | Atomic load-exclusive/store-exclusive |
| DMB/DSB (barriers) | MISSING | P3 | Memory ordering barriers |

### Missing Memory Properties

| Property | Status | Priority | What it would verify |
|----------|--------|----------|---------------------|
| Volatile loads/stores | MISSING | P2 | Volatile memory is not reordered or optimized away |
| Alignment fault model | PARTIAL | P2 | Alignment checked but not modeled as fault |
| Multi-threaded ordering | MISSING | P3 | Sequential consistency, acquire/release semantics |
| Stack memory model | MISSING | P2 | SP-relative loads/stores, frame layout correctness |
| Heap memory model | MISSING | P3 | Allocation/deallocation, use-after-free prevention |
| Address aliasing | MISSING | P2 | Two pointers to same location produce consistent results |
| Wrapping address arithmetic | PARTIAL | P2 | No-wrap precondition in symbolic proofs, but needs formalization |

### Missing tMIR Proof Integration

The tMIR IR carries proof annotations that the memory verification does not yet
consume. These are defined in `llvm2-ir/src/inst.rs` as `ProofAnnotation`:

| Annotation | How it connects to memory verification |
|-----------|---------------------------------------|
| `ValidBorrow` | Proves non-aliasing -- could strengthen non-interference proofs from "disjoint addresses" to "non-aliasing borrows" |
| `InBounds` | Proves array access is within bounds -- could eliminate alignment fault checks and enable unchecked memory access verification |
| `NotNull` | Proves pointer is non-null -- could be used as precondition for load/store proofs |
| `NoOverflow` | Proves arithmetic does not overflow -- relevant for address computation (base + offset) |
| `NonZeroDivisor` | Not directly memory-related |
| `ValidShift` | Not directly memory-related |
| `PositiveRefCount` | Proves object is still alive -- prevents use-after-free |

The path to integrating these is:
1. Model proof annotations as SMT preconditions
2. When `ValidBorrow` is present, strengthen non-interference to full non-aliasing
3. When `InBounds` is present, add `addr < object_size` as a known constraint
4. When `NotNull` is present, add `addr != 0` as a known constraint

---

## Path to Formal Verification via z4

### Phase 1: SMT-LIB2 Readiness (DONE)

The `memory_proofs.rs` module already generates valid SMT-LIB2 output:
- `ProofObligation::to_smt2()` emits complete QF_ABV queries
- `z4_bridge.rs` has `generate_smt2_query()` with automatic logic inference
- Integration test `test_cli_verify_memory_roundtrip_i8` passes with z3
- Array operations (select, store, const_array) serialize correctly

### Phase 2: z3 CLI Verification (Ready to execute)

The infrastructure exists in `z4_bridge.rs`:
- `verify_with_cli()` -- writes SMT-LIB2 to temp file, invokes z3, parses output
- `generate_smt2_query_with_arrays()` -- handles array-sorted variable declarations
- `parse_solver_output()` -- extracts SAT/UNSAT/counterexample
- `Z4Config` -- configurable timeout (default 5000ms), solver path, model production

**What is needed:**
- Run `verify_with_cli()` on all 27 memory proof obligations
- This would promote proofs from "valid by evaluation" to "formally verified by z3"
- Estimated time: the proofs are QF_ABV (decidable), typically solve in <1s each

### Phase 3: z4 Native API (Requires z4 array theory support)

The `z4_bridge.rs` module has a feature-gated `verify_with_z4_api()` that uses
z4's native Rust API. However, the `translate_expr_to_z4()` function currently
returns an error for array operations:

```rust
SmtExpr::Select { .. }
| SmtExpr::Store { .. }
| SmtExpr::ConstArray { .. } => {
    Err("Array theory (QF_ABV) not yet supported in z4 native API; \
         use CLI fallback with z3".to_string())
}
```

**What is needed in z4:**
- `z4-theories` must implement the array theory solver
- `z4-bindings` already has the Expr API for Select/Store/ConstArray (confirmed
  in `~/z4/crates/z4-bindings/src/expr/arrays.rs`)
- The `z4-dpll` crate has `bv_array_smoke_tests.rs`, suggesting array theory
  is under development

**Migration path:**
1. z4 implements array theory decision procedure
2. `translate_expr_to_z4()` gains Select/Store/ConstArray translation
3. `verify_with_z4()` routes memory proofs through native API instead of CLI
4. All 27 proofs run in-process without subprocess overhead

### Phase 4: Memory Model Extensions

Once formal verification is working:

1. **Sign-extending loads** (LDRSB, LDRSH, LDRSW): Add `encode_aarch64_ldr_signed()`
   that sign-extends the loaded value. This is straightforward BV sign_ext.

2. **Store pair (STP)**: Mirror the LDP proof pattern in `memory_model.rs`.

3. **tMIR proof annotations**: Model ValidBorrow/InBounds/NotNull as SMT
   preconditions that strengthen the proofs.

4. **Stack memory**: Model the stack as a separate memory region with SP-relative
   addressing. This connects to `llvm2-codegen`'s frame lowering.

5. **Volatile operations**: Add a side-condition that volatile loads/stores
   cannot be reordered or eliminated by the optimizer.

---

## Verification Methodology Assessment

### Current Approach: Mock Evaluation

- **Technique:** Exhaustive testing for small bitwidths (8-bit), random sampling
  for larger widths (32-bit: 1M samples, 64-bit: 100K samples)
- **Strength:** Fast, no external solver dependency, catches most bugs
- **Weakness:** Not a formal proof -- could miss corner cases in 64-bit space
- **Appropriate for:** Development iteration, regression testing

### Target Approach: SMT Solving

- **Technique:** Universal verification via z3/z4 SAT/UNSAT decision
- **Strength:** Complete -- if UNSAT, the property holds for ALL inputs
- **Weakness:** Slower (seconds per proof), requires solver dependency
- **Appropriate for:** Release verification, CI gate

### Recommended Hybrid

1. **Development:** Mock evaluation (fast feedback, no dependencies)
2. **Pre-merge:** z3 CLI verification (complete proofs, ~30s for all 27)
3. **Release:** z4 native API (in-process, no subprocess overhead)

---

## Summary

| Metric | Value |
|--------|-------|
| Total memory proof obligations | 44 (27 symbolic + 17 concrete) |
| Proof categories covered | Load/Store equivalence, roundtrip, non-interference, endianness, LDP, alignment |
| Sizes covered | I8, I16, I32, I64 |
| Addressing modes | Unsigned immediate (scaled), zero offset |
| SMT-LIB2 serialization | Complete and tested |
| z3 CLI integration | Working (tested for roundtrip I8) |
| z4 native API | Blocked on z4 array theory support |
| Missing: sign-extending loads | P2 |
| Missing: register-offset addressing | P2 |
| Missing: tMIR proof annotations | P2 |
| Missing: multi-threaded ordering | P3 |
| Missing: volatile semantics | P2 |
