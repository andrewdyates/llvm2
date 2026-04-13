# LLVM2 Verification Architecture: z4 SMT Integration

**Date:** 2026-04-13
**Author:** Andrew Yates <ayates@dropbox.com>
**Status:** Draft
**Part of:** #34 (Phase 9: z4 verification integration)

---

## Overview

This document specifies the verification architecture for LLVM2 -- a system that
proves every lowering rule `tMIR_inst -> Vec<AArch64Inst>` correct using z4 SMT
solving. The approach is adapted from Alive2 (Lopes et al., PLDI 2021) but
simplified for our context: we verify instruction selection and peephole
optimizations, not general-purpose IR transformations.

**Key insight:** Unlike Alive2, which must handle LLVM's full generality (undef,
poison, freeze, volatile, atomics, 258+ intrinsics), LLVM2 operates on tMIR --
a well-defined IR with no undef/poison distinction and explicit proof annotations.
This dramatically simplifies the verification problem.

**Scope:**
- Prove correctness of each instruction lowering rule
- Prove correctness of each peephole optimization
- Optional: offline verification, not on the hot compilation path
- In-process z4 solving via `z4-bindings` Rust API (no subprocess)

**Non-goals for MVP:**
- Memory operation verification (load/store -- requires full memory model)
- Control flow verification (branch, call -- requires CFG semantics)
- Floating-point verification (requires FP theory, which z4 supports but is complex)
- Full-program verification (we verify rule-by-rule, not whole functions)

---

## Reference: Alive2 Approach

**Source:** Nuno P. Lopes, Juneyoung Lee, Chung-Kil Hur, Zhengyang Liu, John Regehr.
"Alive2: Bounded Translation Validation for LLVM." PLDI 2021.
https://doi.org/10.1145/3453483.3454030

### How Alive2 Works

Alive2 verifies LLVM IR transformations by checking **refinement**: the target
(optimized) function must be a refinement of the source (original) function.
Specifically, for every possible input state, every behavior of the target must
be a possible behavior of the source.

**Program state** = `(R, M, b)` where:
- `R`: register file (maps variable names to Values)
- `M`: memory (maps BlockID to MemBlock with data, size, alignment, alive flag)
- `b`: boolean flag (whether the execution triggered UB)

**Value** = either `poison`, a set of integers/floats/pointers, or an `Aggregate`.
The set representation handles `undef` (non-determinism).

**Refinement of values:** `v >=_r v'` means v' is "at least as defined" as v:
- `poison >=_r v` for all v (poison refines anything)
- `undef >=_r v` for all concrete v (undef can be any value)
- Concrete values refine only themselves

**Refinement of final states:** `(r, M, true) >=_st (r', M', ub')` always holds
when source has UB. When source does not have UB: target must not have UB, and
return value and memory must be refined.

**Verification condition:** Source function `f_src` is refined by target `f_tgt` iff:
```
forall I_src, I_tgt .
  (I_src >=_r I_tgt) AND valid(I_src, I_tgt) =>
  exists N_tgt .
    (exists N_src . pre_src(I_src, N_src) AND [f_src](I_src, N_src) = O_src)
    => (exists N_src . pre_src(I_src, N_src) AND [f_tgt](I_tgt, N_tgt) >=_st O_tgt)
```

Where `N_src`, `N_tgt` are non-determinism variables (from undef/freeze).

**SMT theories used:** QF_BV (bitvectors) for integer operations, arrays for
memory, FP theory for floating-point. All integer values are bitvectors at
their semantic width. Memory is `Array(BitVec64, BitVec8)` (byte-addressed).

**Key techniques:**
1. Each instruction's semantics encoded as an SMT expression over bitvectors
2. Poison propagation tracked per-value via boolean side-variables
3. UB tracked as a boolean flag that, once set, makes all behaviors valid
4. `nsw`/`nuw` flags generate additional overflow-check constraints
5. Undef values represented as fresh SMT variables (universally quantified in source, existentially in target)

### What We Adapt, What We Simplify

| Alive2 Feature | LLVM2 Status | Rationale |
|----------------|-------------|-----------|
| Poison tracking | Not needed | tMIR has no poison semantics |
| Undef handling | Not needed | tMIR values are fully defined |
| Memory model | Deferred (Phase 2) | Focus on register-to-register ops first |
| Refinement relation | Simplified to equivalence | No non-determinism in tMIR |
| Loop unrolling | Not needed | We verify individual rules, not loops |
| Function calls | Not needed | Rules are local |
| FP semantics | Deferred (Phase 3) | z4 supports FP but it adds complexity |

**Our proof obligation simplifies to:**
```
forall inputs: tMIR_semantics(inputs) = AArch64_semantics(inputs)
```

This is a bitvector equivalence check -- the simplest class of SMT problem.

---

## Architecture

### Crate: `llvm2-verify`

```
crates/llvm2-verify/
  src/
    lib.rs              -- public API: Verifier, VerificationResult
    smt.rs              -- SMT encoding utilities (Sort mapping, Expr helpers)
    tmir_semantics.rs   -- tMIR instruction semantics as z4 Exprs
    aarch64_semantics.rs -- AArch64 instruction semantics as z4 Exprs
    lowering_proof.rs   -- Per-rule proof obligations
    peephole_proof.rs   -- Per-optimization proof obligations
    nzcv.rs             -- AArch64 NZCV flag modeling
    runner.rs           -- Batch verification runner
  Cargo.toml
```

### Dependency Graph

```
llvm2-verify
  ├── llvm2-ir        (MachInst, AArch64Opcode, operand types)
  ├── llvm2-lower     (tMIR types, instruction definitions, isel rules)
  └── z4-bindings     (Z4Program, Expr, Sort -- feature-gated)
```

### Feature Gate

z4 is an optional dependency. The core backend compiles without it:

```toml
[features]
default = []
verify = ["dep:z4-bindings"]

[dependencies]
z4-bindings = { git = "https://github.com/dropbox-ai-prototypes/z4", optional = true }
```

When `verify` is not enabled, `llvm2-verify` compiles but all verification
functions return `VerificationResult::Unknown { reason: "z4 not available" }`.

---

## SMT Encoding: Type Mapping

### tMIR Types to SMT Sorts

Every tMIR type maps to a fixed-width bitvector sort:

| tMIR Type | SMT Sort | Width | Notes |
|-----------|----------|-------|-------|
| `B1` | `BitVec(1)` | 1 bit | Boolean as 1-bit BV, not SMT Bool |
| `I8` | `BitVec(8)` | 8 bits | |
| `I16` | `BitVec(16)` | 16 bits | |
| `I32` | `BitVec(32)` | 32 bits | Maps to W registers |
| `I64` | `BitVec(64)` | 64 bits | Maps to X registers |
| `I128` | `BitVec(128)` | 128 bits | Lowered to register pairs |
| `F32` | `FP(8, 24)` | 32 bits | IEEE 754 single (deferred) |
| `F64` | `FP(11, 53)` | 64 bits | IEEE 754 double (deferred) |

**Why BitVec(1) for B1, not SMT Bool:** AArch64 condition codes and CSET
produce integer 0/1 values, not abstract booleans. Using BitVec(1) preserves
the correspondence between the SMT model and the machine semantics.

### Rust mapping function

```rust
use z4_bindings::Sort;
use llvm2_lower::types::Type;

fn type_to_sort(ty: Type) -> Sort {
    match ty {
        Type::B1  => Sort::bitvec(1),
        Type::I8  => Sort::bitvec(8),
        Type::I16 => Sort::bitvec(16),
        Type::I32 => Sort::bitvec(32),
        Type::I64 => Sort::bitvec(64),
        Type::I128 => Sort::bitvec(128),
        // FP types deferred
        Type::F32 | Type::F64 => unimplemented!("FP verification not yet supported"),
    }
}
```

---

## SMT Encoding: tMIR Instruction Semantics

Each tMIR instruction is encoded as a pure function from input bitvectors to
output bitvectors. The encoding is defined in `tmir_semantics.rs`.

### Arithmetic Operations

| tMIR Opcode | SMT Encoding | z4 API |
|-------------|-------------|--------|
| `Iadd` | `bvadd(a, b)` | `a.bvadd(b)` |
| `Isub` | `bvsub(a, b)` | `a.bvsub(b)` |
| `Imul` | `bvmul(a, b)` | `a.bvmul(b)` |
| `Sdiv` | `bvsdiv(a, b)` | `a.bvsdiv(b)` |
| `Udiv` | `bvudiv(a, b)` | `a.bvudiv(b)` |

**Division by zero:** Both `bvsdiv` and `bvudiv` in SMT-LIB are total functions
(they return a defined value when the divisor is zero). In tMIR, division by
zero is UB. We handle this by adding a precondition: `b != 0`. The lowering
rule need only be correct when the precondition holds.

### Comparison Operations

| tMIR Opcode | Condition | SMT Encoding | z4 API |
|-------------|-----------|-------------|--------|
| `Icmp` | `Equal` | `bveq(a, b)` | `a.eq(b)` |
| `Icmp` | `NotEqual` | `not(bveq(a, b))` | `a.eq(b).not()` |
| `Icmp` | `SignedLessThan` | `bvslt(a, b)` | `a.bvslt(b)` |
| `Icmp` | `SignedGreaterThanOrEqual` | `bvsge(a, b)` | `a.bvsge(b)` |
| `Icmp` | `SignedGreaterThan` | `bvsgt(a, b)` | `a.bvsgt(b)` |
| `Icmp` | `SignedLessThanOrEqual` | `bvsle(a, b)` | `a.bvsle(b)` |
| `Icmp` | `UnsignedLessThan` | `bvult(a, b)` | `a.bvult(b)` |
| `Icmp` | `UnsignedGreaterThanOrEqual` | `bvuge(a, b)` | `a.bvuge(b)` |
| `Icmp` | `UnsignedGreaterThan` | `bvugt(a, b)` | `a.bvugt(b)` |
| `Icmp` | `UnsignedLessThanOrEqual` | `bvule(a, b)` | `a.bvule(b)` |

Comparison results are 1-bit bitvectors: `BitVec(1)` with value 0 (false)
or 1 (true).

### Constant Materialization

`Iconst(ty, imm)` produces the bitvector constant `imm` at width `ty.bits()`:

```rust
fn encode_iconst(ty: Type, imm: i64) -> Expr {
    let width = ty.bits();
    Expr::bitvec_const(imm, width)
}
```

---

## SMT Encoding: AArch64 Instruction Semantics

Each AArch64 instruction is encoded as a pure function from input bitvectors to
output bitvectors, modeling the instruction's effect on registers. The encoding
is defined in `aarch64_semantics.rs`.

### Key Principle: 32-bit Operations Zero-Extend to 64 Bits

On AArch64, every 32-bit (`W` register) operation implicitly zeros the upper 32
bits of the corresponding 64-bit (`X`) register. For verification purposes, we
model 32-bit operations as operating on `BitVec(32)` inputs and producing
`BitVec(32)` outputs. The zero-extension is a separate concern handled at the
register-width level.

When verifying a lowering rule that maps a 32-bit tMIR operation to a 32-bit
AArch64 instruction, we only need to verify the 32-bit semantics match. The
zero-extension property is verified once as a separate lemma.

### Arithmetic Instructions

| AArch64 Opcode | SMT Encoding | Notes |
|----------------|-------------|-------|
| `ADDWrr` / `ADDXrr` | `bvadd(Rn, Rm)` | Register-register add |
| `ADDWri` / `ADDXri` | `bvadd(Rn, imm12)` | Register-immediate add |
| `SUBWrr` / `SUBXrr` | `bvsub(Rn, Rm)` | Register-register sub |
| `SUBWri` / `SUBXri` | `bvsub(Rn, imm12)` | Register-immediate sub |
| `MULWrrr` / `MULXrrr` | `bvmul(Rn, Rm)` | Actually MADD Rd, Rn, Rm, XZR |
| `SDIVWrr` / `SDIVXrr` | `bvsdiv(Rn, Rm)` | Signed divide |
| `UDIVWrr` / `UDIVXrr` | `bvudiv(Rn, Rm)` | Unsigned divide |

### Comparison Instructions and NZCV Flags

AArch64 comparisons set the NZCV (Negative, Zero, Carry, oVerflow) flags.
These are modeled as four boolean expressions derived from the subtraction
result:

```rust
/// Encode CMP semantics: CMP Rn, Rm computes Rn - Rm and sets NZCV.
fn encode_cmp(rn: Expr, rm: Expr, width: u32) -> NzcvFlags {
    let result = rn.clone().bvsub(rm.clone());
    let zero = Expr::bitvec_const(0i64, width);

    NzcvFlags {
        // N: sign bit of result
        n: result.clone().extract(width - 1, width - 1).eq(Expr::bitvec_const(1u64, 1)),
        // Z: result == 0
        z: result.clone().eq(zero),
        // C: unsigned borrow (Rn >= Rm for unsigned)
        c: rn.clone().bvuge(rm.clone()),
        // V: signed overflow
        v: {
            // Overflow when: signs of operands differ AND sign of result
            // differs from sign of first operand
            let rn_neg = rn.clone().extract(width - 1, width - 1);
            let rm_neg = rm.clone().extract(width - 1, width - 1);
            let res_neg = result.extract(width - 1, width - 1);
            rn_neg.clone().eq(rm_neg.clone()).not()
                .and(rn_neg.eq(res_neg).not())
        },
    }
}

struct NzcvFlags {
    n: Expr,  // Bool
    z: Expr,  // Bool
    c: Expr,  // Bool
    v: Expr,  // Bool
}
```

### Condition Code Evaluation

`CSET Wd, cc` produces 1 if condition `cc` is true, 0 otherwise. Each condition
code is a boolean function of NZCV:

```rust
fn eval_condition(cc: AArch64CC, flags: &NzcvFlags) -> Expr {
    let result_bool = match cc {
        AArch64CC::EQ => flags.z.clone(),                           // Z==1
        AArch64CC::NE => flags.z.clone().not(),                     // Z==0
        AArch64CC::HS => flags.c.clone(),                           // C==1
        AArch64CC::LO => flags.c.clone().not(),                     // C==0
        AArch64CC::MI => flags.n.clone(),                           // N==1
        AArch64CC::PL => flags.n.clone().not(),                     // N==0
        AArch64CC::VS => flags.v.clone(),                           // V==1
        AArch64CC::VC => flags.v.clone().not(),                     // V==0
        AArch64CC::HI => flags.c.clone().and(flags.z.clone().not()), // C==1 & Z==0
        AArch64CC::LS => flags.c.clone().not().or(flags.z.clone()), // C==0 | Z==1
        AArch64CC::GE => flags.n.clone().eq(flags.v.clone()),       // N==V
        AArch64CC::LT => flags.n.clone().eq(flags.v.clone()).not(), // N!=V
        AArch64CC::GT => flags.z.clone().not()                      // Z==0 & N==V
                           .and(flags.n.clone().eq(flags.v.clone())),
        AArch64CC::LE => flags.z.clone()                            // Z==1 | N!=V
                           .or(flags.n.clone().eq(flags.v.clone()).not()),
    };
    // Convert Bool to BitVec(1): ite(cond, 1, 0)
    Expr::ite(result_bool, Expr::bitvec_const(1u64, 1), Expr::bitvec_const(0u64, 1))
}
```

### Constant Materialization

| AArch64 Opcode | SMT Encoding |
|----------------|-------------|
| `MOVZWi(imm16)` | `zero_extend(imm16, 16)` for 32-bit |
| `MOVZXi(imm16)` | `zero_extend(imm16, 48)` for 64-bit |
| `MOVNWi(imm16)` | `bvnot(zero_extend(imm16, 16))` for 32-bit |
| `MOVNXi(imm16)` | `bvnot(zero_extend(imm16, 48))` for 64-bit |
| `MOVKXi(Rd, imm16, shift)` | `bvor(bvand(Rd, mask), bvshl(zext(imm16), shift))` |

MOVK is special: it inserts 16 bits at a given position, preserving other bits.
The mask clears the target 16-bit lane: `mask = bvnot(bvshl(0xFFFF, shift))`.

---

## Proof Obligations

### Per-Lowering-Rule Verification

For each instruction selection rule in `llvm2-lower/src/isel.rs`, we generate
a verification condition:

```
Rule: tMIR::Iadd(a: I32, b: I32) -> result: I32
  emits: ADDWrr Wd, Wn, Wm

Proof obligation:
  forall a: BitVec(32), b: BitVec(32) .
    bvadd(a, b) == bvadd(a, b)
```

This is trivially true for direct mappings. The interesting cases are:

#### 1. Multi-instruction lowering

```
Rule: tMIR::Iconst(I64, 0x1234_5678) -> result: I64
  emits: MOVZXi Xd, #0x5678
         MOVKXi Xd, #0x1234, LSL #16

Proof obligation:
  forall (none -- constant) .
    bitvec_const(0x12345678, 64)
    ==
    bvor(
      zero_extend(bitvec_const(0x5678, 16), 48),
      bvshl(zero_extend(bitvec_const(0x1234, 16), 48), bitvec_const(16, 64))
    )
```

This checks that the MOVZ+MOVK sequence produces the correct 64-bit constant.

#### 2. Comparison lowering (CMP + CSET)

```
Rule: tMIR::Icmp(SignedLessThan, a: I32, b: I32) -> result: B1
  emits: CMPWrr Wn, Wm
         CSETWcc Wd, LT

Proof obligation:
  forall a: BitVec(32), b: BitVec(32) .
    ite(bvslt(a, b), bv1(1), bv1(0))
    ==
    eval_condition(LT, encode_cmp(a, b, 32))
```

This checks that the tMIR signed-less-than comparison produces the same
1-bit result as the CMP + CSET LT instruction sequence.

#### 3. Width-dependent lowering

```
Rule: tMIR::Iadd(a: I32, b: I32) -> result: I32
  emits: ADDWrr Wd, Wn, Wm    (32-bit variant)

Rule: tMIR::Iadd(a: I64, b: I64) -> result: I64
  emits: ADDXrr Xd, Xn, Xm    (64-bit variant)

Both are: forall a, b . bvadd(a, b) == bvadd(a, b)
But at different widths (32 vs 64).
```

### Per-Peephole-Optimization Verification

Each peephole optimization in `llvm2-opt` is also a verifiable unit:

```
Optimization: ADDWrr Wd, Wn, #0 -> MOVWrr Wd, Wn

Proof obligation:
  forall a: BitVec(32) .
    bvadd(a, bitvec_const(0, 32)) == a
```

More complex examples:

```
Optimization: SUBWrr Wd, Wn, Wn -> MOVZWi Wd, #0

Proof obligation:
  forall a: BitVec(32) .
    bvsub(a, a) == bitvec_const(0, 32)
```

```
Optimization: ADDWrr + ADDWrr (associative combine)
  ADDWrr Wd, Wn, Wm  ; tmp = a + b
  ADDWrr Wd2, Wd, Wk  ; result = tmp + c
  ->
  ADDWrr Wd, Wm, Wk   ; tmp = b + c
  ADDWrr Wd2, Wn, Wd   ; result = a + tmp

Proof obligation:
  forall a, b, c: BitVec(32) .
    bvadd(bvadd(a, b), c) == bvadd(a, bvadd(b, c))
```

This holds because bitvector addition is associative (unlike signed integer
addition in C, which can overflow differently).

---

## Concrete Example: Proving `tMIR::Iadd(I32, a, b) -> ADDWrr` Correct

### Step 1: Encode as Z4Program

```rust
use z4_bindings::{Z4Program, Sort, Expr};

fn verify_iadd_i32() -> VerificationResult {
    let mut program = Z4Program::qf_bv();

    // Declare symbolic inputs
    let a = program.declare_const("a", Sort::bitvec(32));
    let b = program.declare_const("b", Sort::bitvec(32));

    // tMIR semantics: Iadd(a, b) = bvadd(a, b)
    let tmir_result = a.clone().bvadd(b.clone());

    // AArch64 semantics: ADDWrr Wd, Wn, Wm = bvadd(Wn, Wm)
    let aarch64_result = a.clone().bvadd(b.clone());

    // Proof obligation: NOT(tmir == aarch64) should be UNSAT
    // If UNSAT: the two are always equal (proof succeeds)
    // If SAT: the solver found a counterexample (proof fails)
    program.assert(tmir_result.eq(aarch64_result).not());
    program.check_sat();

    // Serialize and solve
    let smt2 = program.to_string();
    // ... invoke z4 solver ...
    // If result == UNSAT -> VerificationResult::Valid
    // If result == SAT  -> VerificationResult::Invalid { counterexample }
}
```

### Step 2: Generated SMT-LIB2

```smt2
(set-logic QF_BV)
(declare-const a (_ BitVec 32))
(declare-const b (_ BitVec 32))
(assert (not (= (bvadd a b) (bvadd a b))))
(check-sat)
```

### Step 3: z4 Result

```
unsat
```

The formula `not(bvadd(a, b) == bvadd(a, b))` is unsatisfiable for all 32-bit
bitvectors a and b. Therefore the lowering rule is correct.

### A Non-Trivial Example: Constant Materialization

```rust
fn verify_iconst_0x12345678() -> VerificationResult {
    let mut program = Z4Program::qf_bv();

    // tMIR semantics: Iconst(I64, 0x12345678)
    let tmir_result = Expr::bitvec_const(0x12345678i64, 64);

    // AArch64 semantics: MOVZXi #0x5678 + MOVKXi #0x1234, LSL #16
    let movz = Expr::bitvec_const(0x5678i64, 64);
    let movk_imm = Expr::bitvec_const(0x1234i64, 64);
    let shift = Expr::bitvec_const(16i64, 64);
    let mask = Expr::bitvec_const(!0xFFFF_0000i64, 64);
    let aarch64_result = movz.clone()
        .bvand(mask)                    // clear bits 31:16 (already 0 from MOVZ)
        .bvor(movk_imm.bvshl(shift));  // insert 0x1234 at bits 31:16

    // Proof obligation
    program.assert(tmir_result.eq(aarch64_result).not());
    program.check_sat();
    // Expected: UNSAT
}
```

### A Non-Trivial Example: Signed Comparison

```rust
fn verify_icmp_slt_i32() -> VerificationResult {
    let mut program = Z4Program::qf_bv();

    let a = program.declare_const("a", Sort::bitvec(32));
    let b = program.declare_const("b", Sort::bitvec(32));

    // tMIR semantics: Icmp(SignedLessThan, a, b) -> 1 if a <_s b, else 0
    let tmir_result = Expr::ite(
        a.clone().bvslt(b.clone()),
        Expr::bitvec_const(1u64, 1),
        Expr::bitvec_const(0u64, 1),
    );

    // AArch64 semantics: CMP Wn, Wm ; CSET Wd, LT
    // CMP sets NZCV from (Rn - Rm)
    let diff = a.clone().bvsub(b.clone());
    let zero32 = Expr::bitvec_const(0i64, 32);

    // N flag: sign bit of result
    let n = diff.clone().extract(31, 31).eq(Expr::bitvec_const(1u64, 1));
    // V flag: signed overflow of subtraction
    let a_neg = a.clone().extract(31, 31);
    let b_neg = b.clone().extract(31, 31);
    let r_neg = diff.extract(31, 31);
    let v = a_neg.clone().eq(b_neg).not().and(a_neg.eq(r_neg).not());
    // LT condition: N != V
    let lt_cond = n.eq(v).not();

    let aarch64_result = Expr::ite(
        lt_cond,
        Expr::bitvec_const(1u64, 1),
        Expr::bitvec_const(0u64, 1),
    );

    // Proof obligation: NOT(tmir == aarch64) should be UNSAT
    program.assert(tmir_result.eq(aarch64_result).not());
    program.check_sat();
    // Expected: UNSAT -- confirms CMP+CSET LT correctly implements signed <
}
```

---

## Handling Special Cases

### Register Widths and Zero-Extension

AArch64 32-bit operations (`W` registers) implicitly zero-extend to 64 bits.
This must be verified when a 32-bit tMIR result is later used in a 64-bit context:

```
Lemma: zero_extend_32_to_64
  forall a: BitVec(32), b: BitVec(32) .
    zero_extend(bvadd(a, b), 32) == bvadd(zero_extend(a, 32), zero_extend(b, 32))
```

This holds for add, sub, and, or, xor, and shift-left. It does NOT hold for
signed operations like `bvashr` or `bvsdiv`, which is why those need 64-bit
forms when operating on 64-bit values.

### Immediate Encoding Constraints

AArch64 immediates have encoding constraints (e.g., 12-bit unsigned for ADD/SUB).
The verifier checks that the lowering only emits immediate forms when the value
fits:

```
Precondition for ADDWri: 0 <= imm < 4096
Precondition for ADDXri: 0 <= imm < 4096
```

These are not proof obligations per se, but assertions checked at lowering time
and verified as preconditions in the SMT encoding.

### Division by Zero

tMIR division is UB when divisor is zero. The lowering need only be correct
when the divisor is non-zero:

```
Rule: tMIR::Sdiv(a: I32, b: I32) -> SDIVWrr Wd, Wn, Wm
  Precondition: b != 0
  Proof: forall a, b: BV32 . b != 0 => bvsdiv(a, b) == bvsdiv(a, b)
```

Additionally, `INT_MIN / -1` overflows in two's complement. On AArch64, SDIV
returns 0 for this case (hardware-defined behavior). If tMIR defines this as
UB, we add a precondition excluding it.

### NZCV Flag Correctness

We verify the NZCV encoding independently from the instruction lowering.
This is a one-time proof that our `encode_cmp` function correctly models the
ARM Architecture Reference Manual's specification:

```
Lemma: nzcv_n_correct
  forall a, b: BV32 .
    N_flag(a - b) == (a - b)[31]

Lemma: nzcv_z_correct
  forall a, b: BV32 .
    Z_flag(a - b) == (a - b == 0)

Lemma: nzcv_c_correct
  forall a, b: BV32 .
    C_flag(a - b) == (a >=_u b)
    // C flag for subtraction: borrow-out is inverted carry

Lemma: nzcv_v_correct
  forall a, b: BV32 .
    V_flag(a - b) == (sign(a) != sign(b) AND sign(a) != sign(a - b))
```

Each lemma is a separate SMT check.

---

## Integration with Compilation Pipeline

### Architecture

```
                           ┌──────────────────────┐
                           │   Compilation Path    │
                           │   (hot, always runs)  │
                           │                       │
   tMIR ──> llvm2-lower ──┼──> llvm2-opt ──> ... ──> binary
                           │                       │
                           └──────────────────────┘
                                     │
                              (optional, offline)
                                     │
                           ┌─────────▼────────────┐
                           │   Verification Path   │
                           │   (cold, on-demand)   │
                           │                       │
                           │   llvm2-verify         │
                           │   ├── tmir_semantics   │
                           │   ├── aarch64_semantics│
                           │   ├── lowering_proof   │
                           │   └── z4 solver        │
                           │                       │
                           │   Result: Valid /      │
                           │   Invalid(cex) /       │
                           │   Unknown(timeout)     │
                           └───────────────────────┘
```

### Verification Modes

1. **Build-time verification** (`cargo test --features verify`):
   Run all lowering rule proofs as unit tests. Each test constructs a Z4Program,
   asserts the negation of the equivalence, and checks for UNSAT.

2. **CI gate**: Run verification as part of CI. Any new lowering rule or
   peephole optimization must include a corresponding proof test.

3. **Offline batch verification** (`llvm2-verify --check-all`):
   CLI tool that iterates all registered lowering rules and reports
   verification status.

### Registration Pattern

Each lowering rule registers its proof obligation:

```rust
/// Registry of lowering rules and their proof obligations.
pub struct LoweringRuleRegistry {
    rules: Vec<LoweringRule>,
}

pub struct LoweringRule {
    /// Human-readable name: "Iadd_I32 -> ADDWrr"
    pub name: String,
    /// tMIR opcode being lowered
    pub tmir_opcode: &'static str,
    /// AArch64 opcodes emitted
    pub aarch64_opcodes: Vec<&'static str>,
    /// Function that builds the proof obligation as a Z4Program
    pub build_proof: fn() -> Z4Program,
}
```

---

## Implementation Plan

### Phase 1: Arithmetic Rules (MVP)

**Target:** Prove all arithmetic lowering rules correct (Iadd, Isub, Imul, Sdiv, Udiv)
at both 32-bit and 64-bit widths.

**Deliverables:**
- `tmir_semantics.rs`: Encode Iadd, Isub, Imul, Sdiv, Udiv
- `aarch64_semantics.rs`: Encode ADDWrr/Xrr, SUBWrr/Xrr, MULWrrr/Xrrr, SDIVWrr/Xrr, UDIVWrr/Xrr
- `lowering_proof.rs`: 10 proof obligations (5 opcodes x 2 widths)
- Integration test: all 10 proofs pass

**Estimated effort:** 1 techlead session

### Phase 2: Comparison and Constant Rules

**Target:** NZCV flag model, all 10 comparison conditions, constant materialization
(MOVZ, MOVN, MOVK sequences).

**Deliverables:**
- `nzcv.rs`: NZCV flag encoding + 4 correctness lemmas
- Comparison proofs: 10 IntCC conditions x 2 widths = 20 proofs
- Constant proofs: small (MOVZ), negative (MOVN), large (MOVZ+MOVK) = 6+ proofs

**Estimated effort:** 1-2 techlead sessions

### Phase 3: Peephole Optimizations

**Target:** Prove each peephole optimization in `llvm2-opt` correct.

**Deliverables:**
- `peephole_proof.rs`: One proof per optimization rule
- Proofs for: add-zero elision, sub-self elision, shift combines, etc.

**Estimated effort:** 1 techlead session per 10-15 optimization rules

### Phase 4: Memory Operations (Future)

**Target:** Prove load/store lowering correct using z4's memory model.

**Approach:** Use z4-bindings `MemoryModel` with `Array(BitVec64, BitVec8)` for
byte-addressed memory. Verify that:
- `tMIR::Load(I32, addr)` == `LDRWui [Xn, #offset]` (reads same 4 bytes)
- `tMIR::Store(val, addr)` == `STRWui [Xn, #offset]` (writes same 4 bytes)

This requires modeling byte ordering (little-endian on AArch64) and scaled
vs unscaled offset encoding.

**Estimated effort:** 2-3 techlead sessions

### Phase 5: Floating-Point Operations (Future)

**Target:** Prove FP lowering correct using z4's FP theory.

z4 supports IEEE 754 floating-point via `Sort::fp(ebits, sbits)` and operations
like `fp.add`, `fp.mul`, etc. The main challenge is rounding mode handling.

---

## z4 Solver Integration Details

### In-Process Solving

When the `verify` feature is enabled, verification uses z4 in-process via
the `execute_direct` module (requires `z4-bindings` with `direct` feature):

```rust
#[cfg(feature = "verify")]
use z4_bindings::execute_direct::SmtContext;

pub fn solve(program: &Z4Program) -> VerificationResult {
    let ctx = SmtContext::new();
    match ctx.check_sat(program) {
        SolveResult::Unsat => VerificationResult::Valid,
        SolveResult::Sat(model) => VerificationResult::Invalid {
            counterexample: format!("{}", model),
        },
        SolveResult::Unknown(reason) => VerificationResult::Unknown { reason },
    }
}
```

### Fallback: SMT-LIB2 File + CLI

If the `direct` feature is not available, fall back to serializing the Z4Program
to SMT-LIB2 text and invoking the `z4` binary:

```rust
pub fn solve_cli(program: &Z4Program) -> VerificationResult {
    let smt2 = program.to_string();
    let tmpfile = write_temp_file(&smt2);
    let output = Command::new("z4")
        .arg(&tmpfile)
        .arg("-t:30000")  // 30s timeout
        .output()?;
    parse_result(&output.stdout)
}
```

### Timeout Policy

- Arithmetic rule proofs: 5 seconds (these are trivial for QF_BV)
- NZCV flag proofs: 10 seconds
- Constant materialization proofs: 10 seconds
- Peephole optimization proofs: 30 seconds
- Memory operation proofs: 60 seconds

If a proof times out, it is reported as `Unknown` and flagged for investigation.
A timeout usually indicates a bug in the encoding, not a hard problem.

---

## Comparison with Related Work

| System | Scope | SMT Solver | Language | Memory Model | UB Handling |
|--------|-------|-----------|----------|-------------|-------------|
| **Alive2** | LLVM IR optimizations | Z3 | C++ | Array-based | Poison + undef |
| **CompCert** | Full C compiler | None (Coq proofs) | Coq/OCaml | Concrete | Defined by semantics |
| **CSmith** | Random testing | None | C | N/A | Avoids UB |
| **LLVM2 (this)** | ISel + peepholes | z4 | Rust | z4 MemoryModel | tMIR preconditions |

**Advantages of our approach:**
1. **In-process Rust solver**: No C++ FFI, no subprocess, no serialization fragility
2. **Simpler IR**: tMIR has no undef/poison, reducing encoding complexity
3. **Rule-by-rule**: Each proof is small and fast (seconds, not minutes)
4. **z4 proof certificates**: Can emit UNSAT proofs for external validation

---

## Open Questions

1. **How to handle tMIR extensions?** As tMIR adds new instructions (bitwise ops,
   shifts, extensions, type conversions), each needs semantics defined here.
   Should tMIR carry its own SMT semantics, or should LLVM2 define them?

2. **Proof certificate format?** z4 can emit DRAT/LRAT proofs for UNSAT results.
   Should we store these as build artifacts for auditability?

3. **Regression testing strategy?** When a lowering rule changes, do we re-verify
   only the changed rule, or all rules? (All rules is cheap -- seconds total.)

4. **Interaction with proof-enabled optimizations?** tMIR carries proof annotations
   (NoOverflow, InBounds, NotNull). These could become additional preconditions
   in the verification conditions, enabling stronger optimizations to be proven
   correct.

---

## References

1. Lopes, Lee, Hur, Liu, Regehr. "Alive2: Bounded Translation Validation for LLVM."
   PLDI 2021. https://doi.org/10.1145/3453483.3454030

2. Leroy. "Formal Verification of a Realistic Compiler." CACM 2009.
   (CompCert verified C compiler)

3. ARM Architecture Reference Manual for A-profile architecture (DDI 0487).
   Section C6: AArch64 Instruction Set Encoding.

4. z4 SMT Solver. https://github.com/dropbox-ai-prototypes/z4
   - `z4-bindings`: Typed builder API (Sort, Expr, Z4Program)
   - `z4-bindings/src/expr/bv.rs`: Bitvector operations
   - `z4-bindings/src/memory.rs`: Split-pointer memory model

5. LLVM2 AArch64 Backend Design. `designs/2026-04-12-aarch64-backend.md`
