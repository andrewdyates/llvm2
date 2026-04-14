# Verification Architecture Status Report

**Date:** 2026-04-14
**Author:** R1 (Researcher)
**Crate:** `llvm2-verify`
**Source:** `crates/llvm2-verify/src/`

---

## Executive Summary

The `llvm2-verify` crate contains **26 modules** implementing Alive2-style (PLDI 2021) SMT bitvector verification for LLVM2's compiler backend. The verification system currently provides **~200+ proof obligations** across 12 proof-bearing modules, validated by **952 tests**.

All proofs currently use a **mock evaluation engine** (`verify_by_evaluation`) rather than an external SMT solver. Two verification strength levels are active:

| Level | Method | Bit Width | Completeness |
|-------|--------|-----------|--------------|
| **Exhaustive** | All 2^(w*n) input combinations | <=8-bit | Sound and complete for the given width |
| **Statistical** | Edge cases + 100K random samples (LCG PRNG) | 32/64-bit | High confidence, not formally complete |

A third level -- **Formal (z4 SMT solver)** -- has infrastructure (`z4_bridge.rs`, `Z4Config`) but is not yet the default verification path.

---

## Module Inventory

### Proof-Bearing Modules (12)

| Module | File | Proof Count | Description |
|--------|------|-------------|-------------|
| `lowering_proof` | `lowering_proof.rs` | 26 | Core tMIR-to-AArch64 lowering (arithmetic, NZCV, comparisons, branches) |
| `peephole_proofs` | `peephole_proofs.rs` | 11 | Peephole identity rules (ADD+0, SUB-0, MUL*1, shifts, bitwise) |
| `opt_proofs` | `opt_proofs.rs` | 14 | General optimization proofs (const fold, absorb, DCE, copy prop) |
| `const_fold_proofs` | `const_fold_proofs.rs` | 34 | Constant folding (10 binary ops + 7 algebraic identities, with 8-bit variants) |
| `cse_licm_proofs` | `cse_licm_proofs.rs` | ~28 | CSE commutativity, LICM pure determinism, memory ordering |
| `dce_proofs` | `dce_proofs.rs` | 11 | Dead code elimination safety + live computation preservation |
| `copy_prop_proofs` | `copy_prop_proofs.rs` | 15 | Copy propagation identity, expression substitution, chain resolution |
| `cfg_proofs` | `cfg_proofs.rs` | 16 | CFG simplification (branch fold, constant branch, dup branch, empty block) |
| `neon_lowering_proofs` | `neon_lowering_proofs.rs` | 22 | NEON SIMD lowering (11 ops x 2 arrangements: 64-bit, 128-bit) |
| `ane_precision_proofs` | `ane_precision_proofs.rs` | ~11 | ANE FP16 precision bounds (element-wise, GEMM, ReLU, range) |
| `memory_proofs` | `memory_proofs.rs` | 27 | Memory model (load/store equiv, roundtrip, non-interference, endianness) |
| `unified_synthesis` | `unified_synthesis.rs` | ~10 | Multi-target CEGIS synthesis (Scalar + NEON) |

**Total proof obligations: ~225**

### Infrastructure Modules (14)

| Module | Purpose |
|--------|---------|
| `smt` | SMT bitvector expression AST (SmtExpr) + evaluator |
| `tmir_semantics` | tMIR instruction semantic encoder |
| `aarch64_semantics` | AArch64 instruction semantic encoder |
| `nzcv` | NZCV flag computation model |
| `verify` | VerificationResult enum, driver |
| `z4_bridge` | Future z4 CLI integration (Z4Config, pipe-based) |
| `synthesis` | Single-target synthesis (SearchConfig, SynthOpcode) |
| `cegis` | CEGIS loop (Counter-Example Guided Inductive Synthesis) |
| `rule_discovery` | Automatic rule discovery engine |
| `neon_semantics` | NEON SIMD instruction semantic model |
| `ane_semantics` | Apple Neural Engine semantic model |
| `gpu_semantics` | GPU compute semantic model |
| `memory_model` | SMT array-based memory model (QF_ABV theory) |
| `lib` | Module declarations, re-exports |

---

## Detailed Proof Inventory

### 1. Lowering Proofs (`lowering_proof.rs`) -- 26 proofs

#### Arithmetic Lowering (5 proofs)

| Proof | Operation | Width | Strength |
|-------|-----------|-------|----------|
| `proof_iadd_i32` | ADD (32-bit) | 32 | Statistical |
| `proof_iadd_i64` | ADD (64-bit) | 64 | Statistical |
| `proof_isub_i32` | SUB (32-bit) | 32 | Statistical |
| `proof_imul_i32` | MUL (32-bit) | 32 | Statistical |
| `proof_neg_i32` | NEG (32-bit) | 32 | Statistical |

#### NZCV Flag Proofs (4 proofs)

| Proof | Flag | Width | Strength |
|-------|------|-------|----------|
| `proof_n_flag_i32` | N (negative) | 32 | Statistical |
| `proof_z_flag_i32` | Z (zero) | 32 | Statistical |
| `proof_c_flag_i32` | C (carry) | 32 | Statistical |
| `proof_v_flag_i32` | V (overflow) | 32 | Statistical |

#### Comparison Lowering (13 proofs)

| Proof | Condition | Width | Strength |
|-------|-----------|-------|----------|
| `proof_icmp_eq_i32` | EQ | 32 | Statistical |
| `proof_icmp_ne_i32` | NE | 32 | Statistical |
| `proof_icmp_slt_i32` | SLT | 32 | Statistical |
| `proof_icmp_sge_i32` | SGE | 32 | Statistical |
| `proof_icmp_sgt_i32` | SGT | 32 | Statistical |
| `proof_icmp_sle_i32` | SLE | 32 | Statistical |
| `proof_icmp_ult_i32` | ULT | 32 | Statistical |
| `proof_icmp_uge_i32` | UGE | 32 | Statistical |
| `proof_icmp_ugt_i32` | UGT | 32 | Statistical |
| `proof_icmp_ule_i32` | ULE | 32 | Statistical |
| `proof_icmp_eq_i64` | EQ | 64 | Statistical |
| `proof_icmp_slt_i64` | SLT | 64 | Statistical |
| `proof_icmp_ult_i64` | ULT | 64 | Statistical |

#### Branch Lowering (4 proofs)

| Proof | Condition | Width | Strength |
|-------|-----------|-------|----------|
| `proof_condbr_eq_i32` | EQ | 32 | Statistical |
| `proof_condbr_ne_i32` | NE | 32 | Statistical |
| `proof_condbr_slt_i32` | SLT | 32 | Statistical |
| `proof_condbr_ult_i32` | ULT | 32 | Statistical |

### 2. Peephole Proofs (`peephole_proofs.rs`) -- 11 proofs

| Proof | Rule | Width | Strength |
|-------|------|-------|----------|
| `proof_add_zero_identity` | ADD Xd, Xn, #0 = MOV | 64 | Statistical |
| `proof_sub_zero_identity` | SUB Xd, Xn, #0 = MOV | 64 | Statistical |
| `proof_mul_one_identity` | MUL Xd, Xn, #1 = MOV | 64 | Statistical |
| `proof_lsl_zero_identity` | LSL Xd, Xn, #0 = MOV | 64 | Statistical |
| `proof_lsr_zero_identity` | LSR Xd, Xn, #0 = MOV | 64 | Statistical |
| `proof_asr_zero_identity` | ASR Xd, Xn, #0 = MOV | 64 | Statistical |
| `proof_orr_self_identity` | ORR Xd, Xn, Xn = MOV | 64 | Statistical |
| `proof_and_self_identity` | AND Xd, Xn, Xn = MOV | 64 | Statistical |
| `proof_eor_zero_identity` | EOR Xd, Xn, #0 = MOV | 64 | Statistical |
| `proof_add_zero_identity_w32` | ADD Wd, Wn, #0 = MOV | 32 | Statistical |
| `proof_sub_zero_identity_w32` | SUB Wd, Wn, #0 = MOV | 32 | Statistical |

### 3. General Optimization Proofs (`opt_proofs.rs`) -- 14 proofs

| Proof | Operation | Width | Strength |
|-------|-----------|-------|----------|
| `proof_const_fold_add` | Constant fold ADD | 64 | Statistical |
| `proof_const_fold_sub` | Constant fold SUB | 64 | Statistical |
| `proof_and_absorb` | AND absorb (x & 0 = 0) | 64 | Statistical |
| `proof_or_absorb` | OR absorb (x \| 0xff..ff = 0xff..ff) | 64 | Statistical |
| `proof_dce_safety` | DCE: dead ADD removal | 64 | Statistical |
| `proof_copy_prop_identity` | Copy prop: COPY(x) = x | 64 | Statistical |
| `proof_const_fold_add_8bit` | Constant fold ADD | 8 | Exhaustive |
| `proof_const_fold_sub_8bit` | Constant fold SUB | 8 | Exhaustive |
| `proof_and_absorb_8bit` | AND absorb | 8 | Exhaustive |
| `proof_or_absorb_8bit` | OR absorb | 8 | Exhaustive |
| `proof_dce_safety_8bit` | DCE safety | 8 | Exhaustive |
| `proof_copy_prop_identity_8bit` | Copy prop identity | 8 | Exhaustive |
| `proof_const_fold_add_w32` | Constant fold ADD | 32 | Statistical |
| `proof_const_fold_sub_w32` | Constant fold SUB | 32 | Statistical |

### 4. Constant Folding Proofs (`const_fold_proofs.rs`) -- 34 proofs

#### Binary Operations (10 x 64-bit Statistical + 10 x 8-bit Exhaustive)

| Operation | 64-bit | 8-bit |
|-----------|--------|-------|
| ADD | Statistical | Exhaustive |
| SUB | Statistical | Exhaustive |
| MUL | Statistical | Exhaustive |
| AND | Statistical | Exhaustive |
| OR | Statistical | Exhaustive |
| XOR | Statistical | Exhaustive |
| SHL (shift guard) | Statistical | Exhaustive |
| SDIV (non-zero guard) | Statistical | Exhaustive |
| NEG | Statistical | Exhaustive |
| NOT | Statistical | Exhaustive |

#### Algebraic Identities (7 x 64-bit Statistical + 7 x 8-bit Exhaustive)

| Identity | 64-bit | 8-bit |
|----------|--------|-------|
| x + 0 = x | Statistical | Exhaustive |
| x * 1 = x | Statistical | Exhaustive |
| x * 0 = 0 | Statistical | Exhaustive |
| x & 0 = 0 | Statistical | Exhaustive |
| x \| 0 = x | Statistical | Exhaustive |
| x ^ 0 = x | Statistical | Exhaustive |
| x - x = 0 | Statistical | Exhaustive |

### 5. CSE/LICM Proofs (`cse_licm_proofs.rs`) -- ~28 proofs

| Category | Count | Width | Strength |
|----------|-------|-------|----------|
| CSE basic (add, sub, mul, and, or, xor) | 6 | 64 | Statistical |
| CSE commutativity (add, mul, and, or, xor) | 5 | 64 | Statistical |
| CSE 8-bit variants | 6 | 8 | Exhaustive |
| LICM pure determinism (add, mul, and, or, xor, sub) | 6 | 64 | Statistical |
| LICM 8-bit variants | 2 | 8 | Exhaustive |
| Pure determinism (add, mul) | 2 | 64 | Statistical |
| Memory effects (store/load ordering, LICM negative) | 2 | N/A | Concrete (10K samples) |

### 6. DCE Proofs (`dce_proofs.rs`) -- 11 proofs

| Category | Count | Width | Strength |
|----------|-------|-------|----------|
| Dead instruction removal (ADD, MUL, AND, SHL) | 4 | 64 | Statistical |
| Live computation preservation (ADD, MUL, SUB, XOR) | 4 | 64 | Statistical |
| 8-bit exhaustive variants | 3 | 8 | Exhaustive |

### 7. Copy Propagation Proofs (`copy_prop_proofs.rs`) -- 15 proofs

| Category | Count | Width | Strength |
|----------|-------|-------|----------|
| Direct copy identity | 1 | 64 | Statistical |
| Copy through expressions (ADD, SUB, MUL, AND, OR, XOR) | 6 | 64 | Statistical |
| Chain resolution (2-level, 3-level, in-expr) | 3 | 64 | Statistical |
| 8-bit exhaustive variants (identity, add, sub, chain-two, chain-in-expr) | 5 | 8 | Exhaustive |

### 8. CFG Simplification Proofs (`cfg_proofs.rs`) -- 16 proofs

| Category | Count | Width | Strength |
|----------|-------|-------|----------|
| Unconditional branch fold (value, computation) | 2 | 64 | Statistical |
| Constant branch fold (CBZ zero/nonzero, CBNZ nonzero/zero) | 4 | N/A | Concrete (constant inputs) |
| CBZ deterministic | 1 | 64 | Statistical |
| Duplicate branch elimination | 1 | 64 | Statistical |
| Empty block redirect | 1 | 64 | Statistical |
| Branch threading | 1 | 64 | Statistical |
| Unreachable block removal | 1 | 64 | Statistical |
| 8-bit exhaustive variants (value, computation, CBZ det, dup branch, redirect) | 5 | 8 | Exhaustive |

### 9. NEON Lowering Proofs (`neon_lowering_proofs.rs`) -- 22 proofs

| Operation | 64-bit (2x32) | 128-bit (4x32) | Strength |
|-----------|---------------|-----------------|----------|
| ADD | 1 | 1 | Statistical |
| SUB | 1 | 1 | Statistical |
| MUL | 1 | 1 | Statistical |
| NEG | 1 | 1 | Statistical |
| AND | 1 | 1 | Statistical |
| ORR | 1 | 1 | Statistical |
| EOR | 1 | 1 | Statistical |
| BIC | 1 | 1 | Statistical |
| SHL | 1 | 1 | Statistical |
| USHR | 1 | 1 | Statistical |
| SSHR | 1 | 1 | Statistical |

All NEON proofs use lane-wise scalar decomposition: 128-bit vectors are split into lo/hi 64-bit halves, with 3-4 independent 64-bit symbolic variables.

### 10. ANE Precision Proofs (`ane_precision_proofs.rs`) -- ~11 proofs

| Category | Count | Method | Strength |
|----------|-------|--------|----------|
| Element-wise FP16 error bounds | 4 | Concrete f64 sampling | Statistical/Concrete |
| GEMM accumulation error bounds | 3 | Concrete f64 sampling | Statistical/Concrete |
| ReLU FP16 safety | 1 | Concrete f64 sampling | Statistical/Concrete |
| Range check proofs | 3 | Concrete f64 sampling | Statistical/Concrete |

Note: ANE proofs use concrete floating-point arithmetic (f64), not symbolic SMT bitvectors. FP16 constants: unit_roundoff = 2^{-11}, epsilon = 2^{-10}, max = 65504.

### 11. Memory Proofs (`memory_proofs.rs`) -- 27 proofs

| Category | Count | Width | Strength |
|----------|-------|-------|----------|
| Load equivalence (I8, I16, I32, I64, offset variants) | 6 | 64-bit addr | Statistical |
| Store equivalence (I8, I16, I32, I64, offset variants) | 6 | 64-bit addr | Statistical |
| Load-store roundtrip (I8, I16, I32, I64) | 4 | 64-bit addr | Statistical |
| Non-interference (fixed gap, cross-size, symbolic gap) | 8 | 64-bit addr | Statistical |
| Endianness (I16, I32, I64 byte order) | 3 | 64-bit addr | Statistical |

Uses SMT array theory (QF_ABV): `Array(BitVec64, BitVec8)` for byte-addressable, little-endian memory model.

### 12. Unified Synthesis (`unified_synthesis.rs`) -- ~10 proofs

Multi-target CEGIS loop generating candidates across `SynthTarget::Scalar` and `SynthTarget::Neon(arrangement)`. Proofs are dynamically generated during synthesis, not statically declared. The synthesis infrastructure includes `TargetCandidate` ranking by cost model.

---

## Verification Strength Summary

| Strength | Count | Confidence | Notes |
|----------|-------|------------|-------|
| **Exhaustive (8-bit)** | ~48 | 100% for 8-bit | All 2^(w*n) combinations checked |
| **Statistical (32/64-bit)** | ~160 | Very high (100K samples + edges) | Not formally complete; edge cases + random |
| **Concrete (FP/memory)** | ~15 | High | Specific constant inputs or 10K random trials |
| **Formal (z4 SMT)** | 0 | N/A | Infrastructure exists, not yet default |

**Total: ~225 proof obligations, 952 tests**

---

## Critical Gaps

### GAP-1: Missing Arithmetic Lowering Widths

**Severity: P2**

Current arithmetic proofs cover only I32 and I64 for ADD, and I32 for SUB/MUL/NEG. Missing:

| Operation | Missing Widths |
|-----------|---------------|
| ADD | I8, I16 |
| SUB | I8, I16, I64 |
| MUL | I8, I16, I64 |
| NEG | I8, I16, I64 |

These are straightforward to add using the existing `ProofObligation` pattern. I8 variants would be exhaustive.

### GAP-2: Incomplete I64 Comparison and Branch Proofs

**Severity: P2**

Comparison proofs at I64 width exist for only 3 of 10 conditions (EQ, SLT, ULT). Missing 7 conditions at I64: NE, SGE, SGT, SLE, UGE, UGT, ULE.

Branch lowering proofs exist only at I32 for 4 of 10 conditions (EQ, NE, SLT, ULT). No I64 branch proofs exist. Missing 6 conditions at I32 (SGE, SGT, SLE, UGE, UGT, ULE) and all 10 at I64.

### GAP-3: No Division Lowering Proofs

**Severity: P2**

No tMIR-to-AArch64 lowering proofs exist for SDIV or UDIV. `const_fold_proofs.rs` has SDIV constant folding (with non-zero guard), but there are no proofs that `tMIR::SDIV -> AArch64::SDIV` preserves semantics.

### GAP-4: No Floating-Point Lowering Proofs

**Severity: P2**

No proofs for FP32/FP64 arithmetic lowering (FADD, FSUB, FMUL, FDIV, FCMP). The ANE precision proofs cover FP16 error bounds but not core FP lowering. The SMT infrastructure (`SmtExpr`) has FP support (`SmtSort::Fp`, `RoundingMode`), but no lowering proofs use it.

### GAP-5: No x86-64 Verification

**Severity: P2**

All 225+ proofs target AArch64 semantics. The x86-64 backend has scaffolding (opcode enum, register definitions, encoding stub) but zero verification coverage. When x86-64 lowering is implemented, an entirely parallel set of proofs will be needed.

### GAP-6: z4 SMT Solver Not Integrated as Default

**Severity: P2**

The `z4_bridge.rs` module and `Z4Config` struct exist, but all proofs use `verify_by_evaluation()` (mock evaluator). The 32/64-bit proofs are statistical (100K random samples), not formally complete. Integrating z4 would upgrade all statistical proofs to formal proofs with mathematical guarantees.

### GAP-7: No RISC-V Verification

**Severity: P3**

RISC-V is listed as a target in CLAUDE.md goals but has no backend code or verification. Lower priority than x86-64.

### GAP-8: No Peephole 8-bit Exhaustive Variants for All Rules

**Severity: P3**

Only 2 of 9 peephole rules have 8-bit exhaustive variants (ADD+0, SUB-0). The remaining 7 rules (MUL*1, LSL<<0, LSR>>0, ASR>>>0, ORR|self, AND&self, EOR^0) lack 8-bit proofs. These are trivial to add.

### GAP-9: NEON Proofs Limited to 32-bit Lanes

**Severity: P3**

NEON proofs cover `2S` (2x32-bit) and `4S` (4x32-bit) arrangements only. Missing: `8B`, `16B`, `4H`, `8H`, `2D` arrangements. Lane-wise correctness for 8/16/64-bit element widths is not verified.

---

## Architecture Assessment

### Strengths

1. **Consistent proof pattern.** All modules use the same `ProofObligation` structure with `tmir_expr`/`aarch64_expr` pairs, making proofs composable and uniform.

2. **Negative testing.** Every proof module includes negative tests that verify incorrect transformations are detected (counterexamples found). This validates the verifier itself.

3. **SMT-LIB2 serialization.** All `ProofObligation`s can be serialized to SMT-LIB2 format (`to_smt2()`), ready for external solver integration.

4. **Dual-width coverage.** Most optimization proofs exist in both 64-bit (statistical) and 8-bit (exhaustive) variants, providing both practical confidence and theoretical soundness (at small widths).

5. **Comprehensive optimization coverage.** All major optimization passes (DCE, CSE, LICM, constant folding, copy propagation, peephole, CFG simplification) have dedicated proof modules.

6. **Memory model.** The QF_ABV array-based memory model enables reasoning about load/store semantics, non-interference, and endianness.

### Weaknesses

1. **No formal proofs.** The z4 integration gap means all 32/64-bit proofs rely on random sampling. A single missed counterexample could hide a real bug.

2. **Width coverage gaps.** Lowering proofs are concentrated at I32 with limited I64 and no I8/I16 coverage.

3. **No FP lowering proofs.** FP is a critical part of any real-world compiler backend.

4. **Single-target verification.** Only AArch64 is verified; x86-64 and RISC-V are unverified.

### Recommendations

1. **Priority 1:** Integrate z4 as the default verifier for all 32/64-bit proofs. This is the highest-impact change -- it would upgrade ~160 statistical proofs to formal proofs.

2. **Priority 2:** Complete I64 comparison and branch proofs (7 missing conditions each). These are mechanical additions to the existing patterns.

3. **Priority 3:** Add I8/I16 arithmetic lowering proofs. I8 variants would be exhaustive; I16 would be statistical.

4. **Priority 4:** Add FP lowering proofs using the existing `SmtSort::Fp` infrastructure.

5. **Priority 5:** Add 8-bit exhaustive variants for remaining peephole rules (7 rules, trivial additions).

---

## Test Infrastructure

| Metric | Value |
|--------|-------|
| Total tests | 952 |
| Verification config | `DEFAULT_SAMPLE_COUNT = 100,000` |
| Exhaustive threshold | `EXHAUSTIVE_WIDTH_THRESHOLD = 8` bits |
| PRNG | Deterministic LCG (seed-based) |
| Evaluation engine | `verify_by_evaluation()` (mock, in-process) |
| SMT output | SMT-LIB2 via `ProofObligation::to_smt2()` |
| z4 bridge | `z4_bridge.rs` (Z4Config, pipe-based, not yet default) |
| CEGIS | `cegis.rs` (Counter-Example Guided Inductive Synthesis) |

---

## Appendix: Module Source Locations

| Module | Path | Lines |
|--------|------|-------|
| `lowering_proof` | `crates/llvm2-verify/src/lowering_proof.rs` | ~1342 |
| `peephole_proofs` | `crates/llvm2-verify/src/peephole_proofs.rs` | ~577 |
| `opt_proofs` | `crates/llvm2-verify/src/opt_proofs.rs` | ~662 |
| `const_fold_proofs` | `crates/llvm2-verify/src/const_fold_proofs.rs` | ~1253 |
| `cse_licm_proofs` | `crates/llvm2-verify/src/cse_licm_proofs.rs` | ~1284 |
| `dce_proofs` | `crates/llvm2-verify/src/dce_proofs.rs` | ~528 |
| `copy_prop_proofs` | `crates/llvm2-verify/src/copy_prop_proofs.rs` | ~621 |
| `cfg_proofs` | `crates/llvm2-verify/src/cfg_proofs.rs` | ~753 |
| `neon_lowering_proofs` | `crates/llvm2-verify/src/neon_lowering_proofs.rs` | ~754 |
| `ane_precision_proofs` | `crates/llvm2-verify/src/ane_precision_proofs.rs` | ~961 |
| `memory_proofs` | `crates/llvm2-verify/src/memory_proofs.rs` | ~1631 |
| `unified_synthesis` | `crates/llvm2-verify/src/unified_synthesis.rs` | ~1800+ |
