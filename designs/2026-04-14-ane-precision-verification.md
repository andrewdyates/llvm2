# ANE Precision Verification: FP32-to-FP16 Bounded Error Proofs

**Author:** Andrew Yates
**Date:** 2026-04-14
**Status:** Design
**Part of:** #170 (ANE precision verification: FP32-to-FP16 bounded error proofs)
**Related:** #128 (Multi-target semantic encoding), #120 (Neural Engine targeting), `designs/2026-04-14-ane-semantics.md`

---

## Implementation Status (as of 2026-04-15)

**Overall: Precision verification framework is implemented using mock (f64) evaluation. Not connected to a real FP SMT solver.**

| Component | Status | Details |
|-----------|--------|---------|
| **ANE precision proofs** (`ane_precision_proofs.rs`) | IMPLEMENTED (mock) | FP32-to-FP16 bounded error proofs using f64 concrete arithmetic to simulate FP16 rounding. |
| **FP16 quantization model** | IMPLEMENTED | In `ane_semantics.rs`. Models rounding to nearest FP16 representable value. |
| **SMT FP theory proofs** | NOT CONNECTED | z4's FP theory (`Sort::fp(5, 11)`) for FP16 not used. Proofs use f64 mock only. See #236. |

---

## Problem

When the compiler lowers tMIR operations from FP32 to ANE's FP16 execution, it introduces quantization error. Without formal bounds on this error, users cannot trust that the ANE-compiled code preserves the numerical properties of the source program. The verification system must prove that for every input in a specified range, the FP16 result is within a bounded distance of the FP32 reference result.

The existing ANE semantics design (`designs/2026-04-14-ane-semantics.md`, Section "Precision Handling: FP16 Quantization") outlines three strategies (exact equivalence, bounded error, structural equivalence) and provides a `verify_ane_bounded_error` sketch. This document specifies the full precision verification framework: the mathematical foundations, SMT encoding approach, safety conditions for ANE dispatch, mixed-precision accumulation, and integration with the existing `ProofObligation` / `VerificationReport` pipeline.

---

## IEEE 754 Half-Precision (FP16) Properties

### Bit Layout

```
FP16: 1 sign + 5 exponent + 10 significand = 16 bits
      (_ FloatingPoint 5 11) in SMT-LIB2

FP32: 1 sign + 8 exponent + 23 significand = 32 bits
      (_ FloatingPoint 8 24) in SMT-LIB2
```

### Key Numerical Properties

| Property | FP16 | FP32 |
|----------|------|------|
| Unit roundoff (u) | 2^{-11} ~ 4.88e-4 | 2^{-24} ~ 5.96e-8 |
| Machine epsilon (eps = 2u) | 2^{-10} ~ 9.77e-4 | 2^{-23} ~ 1.19e-7 |
| Max representable | 65504 | ~3.4e38 |
| Min normal | 2^{-14} ~ 6.10e-5 | 2^{-126} ~ 1.18e-38 |
| Min subnormal | 2^{-24} ~ 5.96e-8 | 2^{-149} ~ 1.40e-45 |
| Decimal precision | ~3.3 digits | ~7.2 digits |

**Ref:** IEEE 754-2019 Section 3.6 (binary16), Table 3.5.

### Rounding Error Model

For RNE (round-to-nearest-even), the standard rounding model states:

```
fl(x) = x * (1 + delta)   where |delta| <= u
```

where `u = 2^{-11}` for FP16. This means every FP16 rounding introduces at most a relative error of `u` for normal numbers.

For subnormal numbers (|x| < 2^{-14}), the error is absolute rather than relative:

```
|fl(x) - x| <= 2^{-24}  (the smallest subnormal)
```

**Ref:** Higham, N. J. "Accuracy and Stability of Numerical Algorithms." SIAM, 2002. Chapter 2, Theorem 2.2.

---

## FP32-to-FP16 Conversion Error Bounds

### Single Value Conversion

The conversion `fp16(x)` where `x` is an FP32 value introduces error:

```
|fp16(x) - x| <= u * |x|        for normal FP16 range
|fp16(x) - x| <= 2^{-24}       for subnormal range
fp16(x) = Inf                    for |x| > 65504 (overflow)
fp16(x) = 0                      for |x| < 2^{-25} (underflow to zero, RNE)
```

### Proof Obligation: Safe Conversion

For a value `x` known to satisfy `lo <= x <= hi`, the conversion is safe if:

1. **No overflow:** `hi <= 65504` and `lo >= -65504`
2. **Bounded relative error:** The error `|fp16(x) - x|` is within application tolerance `epsilon`
3. **No catastrophic underflow:** If the application requires nonzero results, `|x| >= min_normal_fp16 = 2^{-14}`

The SMT encoding of condition (1):

```smt2
; Declare x as FP32
(declare-const x (_ FloatingPoint 8 24))
; Precondition: x in [lo, hi]
(assert (fp.leq (fp #b0 #b01111111 #b00000000000000000000000) x))  ; lo
(assert (fp.leq x (fp #b0 #b10001111 #b11111111110000000000000)))  ; hi = 65504

; Convert to FP16
(declare-const x_fp16 (_ FloatingPoint 5 11))
(assert (= x_fp16 ((_ to_fp 5 11) RNE x)))

; Convert back to FP32 for comparison
(declare-const x_back (_ FloatingPoint 8 24))
(assert (= x_back ((_ to_fp 8 24) RNE x_fp16)))

; Check: |x_back - x| < epsilon
(assert (fp.gt (fp.abs (fp.sub RNE x_back x)) epsilon))

; If UNSAT: error is always bounded by epsilon for x in [lo, hi]
(check-sat)
```

---

## Per-Operation Error Accumulation

### Element-wise Operations

For a single FP16 arithmetic operation (add, sub, mul, div), the standard error model gives:

| Operation | FP16 Result | Error Bound |
|-----------|------------|-------------|
| `fp16(a) + fp16(b)` | `(a+b)(1+d)` | `u * |a+b|` |
| `fp16(a) - fp16(b)` | `(a-b)(1+d)` | `u * |a-b|` |
| `fp16(a) * fp16(b)` | `(a*b)(1+d)` | `u * |a*b|` |
| `fp16(a) / fp16(b)` | `(a/b)(1+d)` | `u * |a/b|` |

where `|d| <= u = 2^{-11}`.

**Catastrophic cancellation:** For subtraction when `a ~ b`, the relative error of `a - b` can be arbitrarily large even though the absolute error is bounded. The proof system must track whether subsequent operations depend on the subtraction result's magnitude.

### GEMM Error Accumulation

For a dot product of K terms: `sum_{k=0}^{K-1} a[k] * b[k]`

Using the standard FP16 error analysis (Higham Ch. 3):

```
|dot_fp16 - dot_exact| <= K * u * max(|a|) * max(|b|) + O(u^2)
```

This is the worst-case. In practice, errors partially cancel (expected error grows as `sqrt(K) * u` rather than `K * u`), but the proof must bound the worst case.

For a full GEMM `C = A * B` where A is M x K and B is K x N:

```
|C_fp16[i,j] - C_exact[i,j]| <= K * u * max(|A|) * max(|B|) + O(u^2)
```

The error grows linearly with the inner dimension K. This is the fundamental limitation of FP16 accumulation.

### Conv2D Error Accumulation

For a 2D convolution with kernel size KH x KW and C_in input channels:

```
|out_fp16[n,c,h,w] - out_exact[n,c,h,w]|
    <= C_in * KH * KW * u * max(|input|) * max(|kernel|) + O(u^2)
```

The accumulation depth is `C_in * KH * KW`, analogous to the K dimension in GEMM.

---

## Mixed-Precision Strategy: FP32 Accumulators with FP16 Inputs

### The Problem with Pure FP16

For a GEMM with K=256, the worst-case FP16 error bound is:

```
256 * 2^{-11} * max(|A|) * max(|B|) = 0.125 * max(|A|) * max(|B|)
```

This is ~12.5% relative error -- unacceptable for most applications.

### Mixed-Precision Solution

Modern ANE hardware (M1+) uses FP32 accumulators internally for GEMM and convolution, even when inputs are FP16. The computation is:

```
// Pseudocode for mixed-precision GEMM
for i in 0..M:
  for j in 0..N:
    acc: f32 = 0.0                       // FP32 accumulator
    for k in 0..K:
      acc += fp32(a_fp16[i,k]) * fp32(b_fp16[k,j])  // products in FP32
    c_fp16[i,j] = fp16(acc)              // final truncation to FP16
```

The error is now:

```
|c_mixed[i,j] - c_exact[i,j]|
    <= u_fp16 * max(|A|) * max(|B|)     // input quantization: 2 * u_fp16
    +  K * u_fp32 * max(|A|) * max(|B|) // accumulation error: K * u_fp32
    +  u_fp16 * |c_exact[i,j]|          // output truncation: u_fp16
```

Since `u_fp32 = 2^{-24}` is much smaller than `u_fp16 = 2^{-11}`, the accumulation error is negligible for reasonable K values. The dominant error is the input quantization (FP32->FP16 conversion at input) and output truncation (FP32->FP16 at output).

### SMT Encoding of Mixed-Precision GEMM

```rust
/// Encode mixed-precision ANE GEMM: FP16 inputs, FP32 accumulator, FP16 output.
///
/// Proof obligation: |mixed_result - exact_result| <= epsilon
/// where epsilon accounts for input quantization + output truncation.
pub fn encode_mixed_precision_gemm(
    a_fp32: &SmtExpr,     // Original FP32 input A, flattened M*K
    b_fp32: &SmtExpr,     // Original FP32 input B, flattened K*N
    m: u64, k: u64, n: u64,
) -> (SmtExpr, SmtExpr) {  // (mixed_result_fp32, error_bound)
    let fp16 = SmtSort::fp16();
    let fp32 = SmtSort::fp32();
    let rne = RoundingMode::RNE;

    // Step 1: Quantize inputs to FP16
    // a_fp16[i,k] = fp16(a_fp32[i,k])
    // Error: |a_fp16 - a_fp32| <= u_fp16 * |a_fp32| per element

    // Step 2: Multiply in FP32 (by promoting FP16 inputs back)
    // This is exact for the FP16 representation.

    // Step 3: Accumulate in FP32
    // acc += fp32(a_fp16[i,k]) * fp32(b_fp16[k,j])
    // Each product is exact (FP16*FP16 fits in FP32 significand).
    // Accumulation introduces FP32 rounding: K * u_fp32.

    // Step 4: Truncate to FP16
    // c_fp16[i,j] = fp16(acc)
    // Final truncation: u_fp16 * |acc|

    // For small M,K,N: unroll fully and prove bitwise.
    // For large: use the error bound formula above.

    // Return placeholder for API shape
    (SmtExpr::fp_const(0, 8, 24), SmtExpr::fp_const(0, 8, 24))
}
```

### When Is Mixed-Precision Sufficient?

| Scenario | Input Range | K (accumulation depth) | Error Bound | Verdict |
|----------|-------------|------------------------|-------------|---------|
| Inference, normalized inputs | [-1, 1] | 256 | ~2 * 2^{-11} + 256 * 2^{-24} ~ 0.001 | Safe |
| Inference, normalized inputs | [-1, 1] | 4096 | ~2 * 2^{-11} + 4096 * 2^{-24} ~ 0.001 | Safe |
| Training, large gradients | [-100, 100] | 256 | ~200 * 2^{-11} + 256 * 200^2 * 2^{-24} ~ 0.1 | Marginal |
| Attention, long sequences | [-10, 10] | 8192 | ~20 * 2^{-11} + 8192 * 100 * 2^{-24} ~ 0.06 | Check carefully |
| Financial computation | [0.01, 1e6] | any | unbounded overflow risk | Unsafe |

---

## Safety Conditions for ANE FP16 Dispatch

### Condition 1: Bounded Inputs (Required)

The tMIR proof annotations must establish that inputs are bounded:

```
forall x in inputs: lo <= x <= hi
  where |lo|, |hi| <= 65504  (FP16 max representable)
```

**How to obtain bounds:** tMIR proof annotations can carry range information from:
- Type constraints (e.g., `normalized: f32` implies `[-1, 1]`)
- Explicit bounds proofs from the source language
- Post-activation bounds (ReLU output is always >= 0)
- Batch normalization output bounds (mean ~0, std ~1)

### Condition 2: Acceptable Error Tolerance (Required)

The user or source program must specify an acceptable error tolerance `epsilon`. The verifier then checks:

```
forall inputs in [lo, hi]:
  |ane_fp16_result - cpu_fp32_result| <= epsilon
```

If no tolerance is specified, use the default: `epsilon = K * 2^{-10} * max(|input|)^2` where K is the accumulation depth.

### Condition 3: No Harmful Underflow (Recommended)

For operations where zero vs nonzero matters (e.g., gating, masking), verify that no value that should be nonzero underflows to zero:

```
forall x in inputs:
  |x| >= min_normal_fp16 OR application tolerates underflow
```

### Condition 4: Monotonicity Preservation (Optional)

For comparison operations or sorting, verify that the FP16 quantization preserves ordering:

```
forall a, b in inputs:
  a < b => fp16(a) <= fp16(b)
```

This holds for FP16 when the gap between a and b exceeds 2 * u * max(|a|, |b|).

---

## SMT Encoding for the Proof System

### ProofObligation Structure

A new `PrecisionProofObligation` extends the existing `ProofObligation` framework:

```rust
/// Precision proof obligation for FP32->FP16 lowering.
///
/// Integrates with VerificationReport via the existing ProofResult/
/// VerificationResult types in verify.rs.
pub struct PrecisionProofObligation {
    /// Name for the proof (e.g., "gemm_fp16_error_bound_4x4x4")
    pub name: String,

    /// The ANE operation being verified.
    pub operation: AneOperation,

    /// Input value range [lo, hi] (FP32).
    pub input_range: (f64, f64),

    /// Maximum acceptable absolute error.
    pub epsilon: f64,

    /// Whether the ANE uses mixed-precision (FP32 accumulator).
    pub mixed_precision: bool,

    /// tMIR proof annotations that justify the input range.
    pub range_proofs: Vec<RangeProof>,
}

/// Range proof extracted from tMIR annotations.
pub enum RangeProof {
    /// Explicit bound: x in [lo, hi]
    Bounded { lo: f64, hi: f64 },
    /// Post-ReLU: x >= 0
    NonNegative,
    /// Post-BatchNorm: x ~ N(0, 1), practically in [-4, 4]
    Normalized { sigma_bound: f64 },
    /// Post-Sigmoid: x in [0, 1]
    Sigmoid,
    /// Post-Tanh: x in [-1, 1]
    Tanh,
}
```

### SMT Formula Generation

For each `PrecisionProofObligation`, generate an SMT formula:

```rust
/// Generate SMT formula for precision verification.
///
/// The formula asserts the NEGATION of the correctness condition:
///   exists inputs in [lo, hi] such that |fp16_result - fp32_result| >= epsilon
///
/// If z4 returns UNSAT, the property holds (error is always < epsilon).
/// If z4 returns SAT, it provides a counterexample (input where error >= epsilon).
pub fn encode_precision_proof(
    obligation: &PrecisionProofObligation,
) -> Vec<SmtExpr> {
    let mut assertions = Vec::new();

    match &obligation.operation {
        AneOperation::Gemm { m, k, n } => {
            // For small GEMM (m*k*n <= 512): fully unrolled FP encoding
            if m * k * n <= 512 {
                // Declare FP32 input variables for A and B
                // Constrain to [lo, hi]
                // Compute FP32 GEMM result
                // Compute FP16 GEMM result (with or without FP32 accumulator)
                // Assert |difference| >= epsilon
                // UNSAT => always within epsilon
                encode_gemm_precision_unrolled(
                    *m, *k, *n,
                    obligation.input_range,
                    obligation.epsilon,
                    obligation.mixed_precision,
                    &mut assertions,
                );
            } else {
                // For large GEMM: use the analytical error bound
                // Verify: K * u * max_input^2 < epsilon
                encode_gemm_precision_analytical(
                    *m, *k, *n,
                    obligation.input_range,
                    obligation.epsilon,
                    obligation.mixed_precision,
                    &mut assertions,
                );
            }
        }
        AneOperation::ElementWise { op, n } => {
            // Element-wise: single-operation error bound
            encode_elementwise_precision(
                *op, *n,
                obligation.input_range,
                obligation.epsilon,
                &mut assertions,
            );
        }
        AneOperation::Activation { act, n } => {
            // Activation functions: ReLU is exact, others need care
            encode_activation_precision(
                *act, *n,
                obligation.input_range,
                obligation.epsilon,
                &mut assertions,
            );
        }
        _ => {
            // Conv2D, Pooling, Reduction: use the generic approach
            encode_generic_precision(
                &obligation.operation,
                obligation.input_range,
                obligation.epsilon,
                obligation.mixed_precision,
                &mut assertions,
            );
        }
    }

    assertions
}
```

### Concrete Element-wise Encoding

For a single element-wise FP16 addition, the full SMT encoding is:

```rust
fn encode_elementwise_add_precision(
    input_range: (f64, f64),
    epsilon: f64,
) -> Vec<SmtExpr> {
    let (lo, hi) = input_range;

    // Declare FP32 input variables
    let a = SmtExpr::Var { name: "a".into(), width: 32 };  // placeholder
    let b = SmtExpr::Var { name: "b".into(), width: 32 };

    // In full FP theory:
    //   declare-const a (_ FloatingPoint 8 24)
    //   declare-const b (_ FloatingPoint 8 24)
    //   assert (fp.leq lo_fp32 a)
    //   assert (fp.leq a hi_fp32)
    //   assert (fp.leq lo_fp32 b)
    //   assert (fp.leq b hi_fp32)
    //
    //   ; FP32 reference
    //   define-fun ref () (_ FloatingPoint 8 24) (fp.add RNE a b)
    //
    //   ; FP16 computation
    //   define-fun a16 () (_ FloatingPoint 5 11) ((_ to_fp 5 11) RNE a)
    //   define-fun b16 () (_ FloatingPoint 5 11) ((_ to_fp 5 11) RNE b)
    //   define-fun sum16 () (_ FloatingPoint 5 11) (fp.add RNE a16 b16)
    //   define-fun result () (_ FloatingPoint 8 24) ((_ to_fp 8 24) RNE sum16)
    //
    //   ; Negated correctness: exists a,b such that error >= epsilon
    //   assert (fp.geq (fp.abs (fp.sub RNE result ref)) epsilon_fp32)
    //
    //   check-sat  ; UNSAT => always safe

    // Using the existing SmtExpr FP API:
    let lo_expr = SmtExpr::fp_const(f32::to_bits(lo as f32) as u64, 8, 24);
    let hi_expr = SmtExpr::fp_const(f32::to_bits(hi as f32) as u64, 8, 24);
    let eps_expr = SmtExpr::fp_const(f32::to_bits(epsilon as f32) as u64, 8, 24);

    // The assertions would be constructed using the FP operations
    // from smt.rs: FPAdd, FPMul, FPEq, FPLt, etc.
    // Full encoding requires fp.to_fp (FP16<->FP32 conversion) which
    // needs to be added to SmtExpr as a new variant.

    vec![]  // placeholder
}
```

### Required SmtExpr Extensions

The current `SmtExpr` in `smt.rs` (filepath: `crates/llvm2-verify/src/smt.rs`) supports `FPAdd`, `FPMul`, `FPDiv`, `FPNeg`, `FPEq`, `FPLt`, and `FPConst`. Precision verification requires these additional variants:

```rust
/// Additional SmtExpr variants needed for precision verification:

/// `(fp.sub rm a b)` -- floating-point subtraction.
/// Currently missing from SmtExpr; needed for error computation.
FPSub { rm: RoundingMode, lhs: Box<SmtExpr>, rhs: Box<SmtExpr> },

/// `(fp.abs a)` -- floating-point absolute value.
/// Needed for |error| computation.
FPAbs { operand: Box<SmtExpr> },

/// `((_ to_fp eb sb) rm expr)` -- FP precision conversion.
/// Critical: this is how FP32->FP16 and FP16->FP32 are encoded.
FPToFP { rm: RoundingMode, operand: Box<SmtExpr>, target_eb: u32, target_sb: u32 },

/// `(fp.geq a b)` -- floating-point greater-or-equal (returns Bool).
/// Needed for range constraint assertions.
FPGeq { lhs: Box<SmtExpr>, rhs: Box<SmtExpr> },

/// `(fp.leq a b)` -- floating-point less-or-equal (returns Bool).
/// Needed for range constraint assertions.
FPLeq { lhs: Box<SmtExpr>, rhs: Box<SmtExpr> },

/// `(fp.isNormal a)` -- check if value is normal (not subnormal/zero/inf/nan).
/// Needed for underflow safety checks.
FPIsNormal { operand: Box<SmtExpr> },
```

These additions are backward-compatible with the existing `SmtExpr` and evaluator.

---

## Proof Structure Summary

For each ANE operation lowered from FP32 tMIR:

```
Step 1: Extract input range from tMIR proof annotations
        => RangeProof::Bounded { lo, hi }

Step 2: Compute analytical error bound
        => epsilon = f(operation, K, input_range)

Step 3: Generate PrecisionProofObligation
        => { operation, input_range, epsilon, mixed_precision }

Step 4: Encode as SMT formula
        => For small operations: fully unrolled FP16/FP32 comparison
        => For large operations: verify analytical bound assumptions

Step 5: Solve with z4
        => UNSAT: safe to lower to ANE FP16
        => SAT:   counterexample found, FP16 is unsafe for this input range

Step 6: Record in VerificationReport
        => ProofResult { name, category: "precision", result }
```

---

## Activation Function Precision

### ReLU: Exact

`ReLU(x) = max(0, x)` is exact in FP16 because it is a comparison followed by selection -- no arithmetic rounding occurs.

```
Proof: forall x: relu_fp16(fp16(x)) == fp16(relu_fp32(x))
This holds because:
  - If x >= 0: relu(x) = x, and fp16(x) >= 0 (rounding preserves sign for finite x)
  - If x < 0: relu(x) = 0, which is exact in any precision
```

### Sigmoid: Bounded by Construction

`sigmoid(x) = 1 / (1 + exp(-x))` has output in (0, 1). The FP16 error is bounded by:

```
|sigmoid_fp16(x) - sigmoid_fp32(x)| <= u_fp16 ~ 4.88e-4
```

This is because sigmoid output is always in [0, 1], so the relative error of FP16 representation is at most `u`.

The existing UF (uninterpreted function) encoding for sigmoid in `ane-semantics.md` should carry these axiomatic constraints:

```
0 <= sigmoid(x) <= 1                    (range)
sigmoid(0) = 0.5                         (midpoint)
x1 < x2 => sigmoid(x1) < sigmoid(x2)   (monotonicity)
```

### GELU / Tanh: Bounded Output

Both GELU and Tanh produce bounded outputs:
- `tanh(x) in [-1, 1]`
- `GELU(x) ~ x * sigmoid(1.702 * x)`, bounded for bounded input

FP16 error for bounded functions: `|f_fp16(x) - f_fp32(x)| <= u_fp16 * max(|f(x)|)`.

---

## Solver Complexity and Tractability

### Per-Operation Complexity

| Proof Type | Operation Size | SMT Theory | Estimated Time |
|------------|---------------|------------|----------------|
| Element-wise precision (1 op) | 1 element | QF_FP | < 1s |
| Element-wise precision (64 ops) | 64 elements | QF_FP | 5-30s |
| GEMM precision (4x4x4, unrolled) | 64 FP ops | QF_FP | 10-60s |
| GEMM precision (8x8x8, unrolled) | 512 FP ops | QF_FP | timeout likely |
| GEMM precision (large, analytical) | bounds check | QF_LRA | < 1s |
| Conv2D 3x3 precision | ~576 FP ops | QF_FP | 30-120s |
| ReLU precision | N ite ops | QF_FP | < 1s per element |
| Mixed-precision GEMM (4x4x4) | 64 FP ops | QF_FP | 10-60s |

### Scalability Strategy

1. **Small operations (< 500 FP ops):** Fully unroll and prove bitwise with QF_FP. This gives complete precision guarantees.

2. **Medium operations (500-5000 FP ops):** Use the analytical error bound formula. The SMT query reduces to verifying the bound assumptions (input range, accumulation depth) which is QF_LRA or even propositional.

3. **Large operations (> 5000 FP ops):** Use the UF abstraction from `ane-semantics.md` with additional precision axioms. The error bound becomes a theorem about the abstract operation rather than a concrete FP computation.

### z4 Theory Requirements

The precision proofs require the QF_FP (floating-point) theory in z4. Current status from `ane-semantics.md`:

| Theory | Need | z4 Status |
|--------|------|-----------|
| QF_FP (FP16) | Critical | #123, in progress |
| QF_FP (FP32) | Critical | #123, in progress |
| FP conversion (to_fp) | Critical | Needed |
| QF_LRA | Analytical bounds | Available |

Until z4 FP support is complete, the framework uses the mock evaluator in `smt.rs` (`try_eval` with `EvalResult::Float`) for exhaustive testing at small widths and random sampling at 32-bit widths, matching the existing verification approach.

---

## Integration with ProofObligation Framework

### File: `crates/llvm2-verify/src/precision_proofs.rs` (new)

```rust
// precision_proofs.rs - FP16 precision verification for ANE lowering
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

use crate::smt::{SmtExpr, SmtSort, RoundingMode};
use crate::verify::{VerificationResult, ProofResult};

/// Category name for precision proofs in VerificationReport.
pub const PRECISION_CATEGORY: &str = "ane_precision";

/// Run all precision proof obligations for a given ANE lowering.
///
/// Returns a Vec<ProofResult> to be merged into VerificationReport.
pub fn verify_ane_precision(
    obligations: &[PrecisionProofObligation],
) -> Vec<ProofResult> {
    obligations.iter().map(|ob| {
        let result = verify_single_precision(ob);
        ProofResult {
            name: ob.name.clone(),
            category: PRECISION_CATEGORY.to_string(),
            result,
        }
    }).collect()
}
```

### Integration Point: `verify.rs`

The existing `Verifier::verify_comprehensive` method should be extended to include precision proofs:

```rust
// In Verifier::verify_comprehensive():
//   let precision_results = precision_proofs::verify_ane_precision(&obligations);
//   report.results.extend(precision_results);
```

---

## Implementation Plan

### Step 1: SmtExpr Extensions (1 session)
- Add `FPSub`, `FPAbs`, `FPToFP`, `FPGeq`, `FPLeq`, `FPIsNormal` to `SmtExpr`
- Add evaluation support for these in `try_eval`
- Add Display (SMT-LIB2) formatting
- File: `crates/llvm2-verify/src/smt.rs`

### Step 2: Precision Proof Types (1 session)
- Define `PrecisionProofObligation`, `RangeProof`
- Implement `encode_precision_proof` for element-wise operations
- File: `crates/llvm2-verify/src/precision_proofs.rs`

### Step 3: Element-wise Precision Proofs (1 session)
- Implement fully unrolled FP16 vs FP32 comparison for add/sub/mul/div
- Test with exhaustive FP16 evaluation (65536 values)
- Verify error bounds match analytical predictions

### Step 4: GEMM Precision Proofs (1-2 sessions)
- Implement unrolled precision proof for small GEMM (4x4x4)
- Implement analytical bound verification for large GEMM
- Test mixed-precision vs pure-FP16 accumulation

### Step 5: Activation Precision Proofs (1 session)
- ReLU: prove exactness
- Sigmoid/GELU/Tanh: prove bounded error using UF axioms
- Test against mock evaluator

### Step 6: Integration with Verification Pipeline (1 session)
- Wire into `verify.rs` and `VerificationReport`
- Add precision proofs to the verification driver
- End-to-end test: tMIR function -> ANE lowering -> precision verification

**Total estimated effort: 6-8 techlead sessions.**

---

## References

1. IEEE 754-2019. "IEEE Standard for Floating-Point Arithmetic." IEEE, 2019.
2. Higham, N. J. "Accuracy and Stability of Numerical Algorithms." SIAM, 2002. Chapters 2-3.
3. Micikevicius, P. et al. "Mixed Precision Training." ICLR 2018. (FP16 training with FP32 accumulators)
4. Apple. "Core ML Model Intermediate Language (MIL) Specification." (ANE precision behavior)
5. LLVM2 ANE semantics: `designs/2026-04-14-ane-semantics.md`
6. LLVM2 verification architecture: `designs/2026-04-13-verification-architecture.md`
7. LLVM2 SMT expression AST: `crates/llvm2-verify/src/smt.rs`
8. LLVM2 verification interface: `crates/llvm2-verify/src/verify.rs`
9. Lopes et al. "Alive2: Bounded Translation Validation for LLVM." PLDI 2021.
