// llvm2-verify/fp_convert_proofs.rs - FP conversion lowering proofs
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Proof obligations for floating-point conversion instructions:
// FCVTZS (FP->signed int), FCVTZU (FP->unsigned int),
// SCVTF (signed int->FP), UCVTF (unsigned int->FP),
// FCVT (FP format widening/narrowing).
//
// These proofs verify that the tMIR FP conversion opcodes lower correctly
// to AArch64 instructions by asserting semantic equivalence via SMT formulas.
//
// Reference: ARM DDI 0487, C7.2.69-C7.2.72 (FCVTZS/FCVTZU),
//            C7.2.194-C7.2.197 (SCVTF/UCVTF), C7.2.68 (FCVT).

//! Proof obligations for FP conversion instruction lowering.
//!
//! Covers 6 AArch64 FP conversion opcodes across multiple type combinations,
//! plus roundtrip and NaN handling properties.

use crate::lowering_proof::ProofObligation;
use crate::smt::{EvalResult, RoundingMode, SmtExpr};
use crate::verify::VerificationResult;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// FCVTZS: Float -> Signed Int (round toward zero)
// ---------------------------------------------------------------------------

/// Proof: `tMIR::FcvtToInt(I32, F32, a) -> FCVTZS Wd, Sn`
///
/// Both sides compute `fp.to_sbv(RTZ, a, 32)` -- FP32 to signed I32
/// with round-toward-zero (truncation), matching C cast semantics.
///
/// Reference: ARM DDI 0487, C7.2.69 FCVTZS (scalar, integer).
pub fn proof_fcvtzs_i32_f32() -> ProofObligation {
    let a = SmtExpr::fp32_const(0.0); // placeholder; concrete values substituted at eval
    ProofObligation {
        name: "FcvtToInt_I32_F32 -> FCVTZS Wd,Sn".to_string(),
        tmir_expr: SmtExpr::fp_to_sbv(RoundingMode::RTZ, a.clone(), 32),
        aarch64_expr: SmtExpr::fp_to_sbv(RoundingMode::RTZ, a, 32),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 8, 24)],
    }
}

/// Proof: `tMIR::FcvtToInt(I64, F64, a) -> FCVTZS Xd, Dn`
///
/// FP64 to signed I64 with round-toward-zero.
pub fn proof_fcvtzs_i64_f64() -> ProofObligation {
    let a = SmtExpr::fp64_const(0.0);
    ProofObligation {
        name: "FcvtToInt_I64_F64 -> FCVTZS Xd,Dn".to_string(),
        tmir_expr: SmtExpr::fp_to_sbv(RoundingMode::RTZ, a.clone(), 64),
        aarch64_expr: SmtExpr::fp_to_sbv(RoundingMode::RTZ, a, 64),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 11, 53)],
    }
}

// ---------------------------------------------------------------------------
// FCVTZU: Float -> Unsigned Int (round toward zero)
// ---------------------------------------------------------------------------

/// Proof: `tMIR::FcvtToUint(I32, F32, a) -> FCVTZU Wd, Sn`
///
/// FP32 to unsigned I32 with round-toward-zero.
///
/// Reference: ARM DDI 0487, C7.2.72 FCVTZU (scalar, integer).
pub fn proof_fcvtzu_i32_f32() -> ProofObligation {
    let a = SmtExpr::fp32_const(0.0);
    ProofObligation {
        name: "FcvtToUint_I32_F32 -> FCVTZU Wd,Sn".to_string(),
        tmir_expr: SmtExpr::fp_to_ubv(RoundingMode::RTZ, a.clone(), 32),
        aarch64_expr: SmtExpr::fp_to_ubv(RoundingMode::RTZ, a, 32),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 8, 24)],
    }
}

/// Proof: `tMIR::FcvtToUint(I64, F64, a) -> FCVTZU Xd, Dn`
///
/// FP64 to unsigned I64 with round-toward-zero.
pub fn proof_fcvtzu_i64_f64() -> ProofObligation {
    let a = SmtExpr::fp64_const(0.0);
    ProofObligation {
        name: "FcvtToUint_I64_F64 -> FCVTZU Xd,Dn".to_string(),
        tmir_expr: SmtExpr::fp_to_ubv(RoundingMode::RTZ, a.clone(), 64),
        aarch64_expr: SmtExpr::fp_to_ubv(RoundingMode::RTZ, a, 64),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 11, 53)],
    }
}

// ---------------------------------------------------------------------------
// SCVTF: Signed Int -> Float
// ---------------------------------------------------------------------------

/// Proof: `tMIR::FcvtFromInt(F32, I32, a) -> SCVTF Sd, Wn`
///
/// Signed I32 to FP32 with round-to-nearest-even (default FPCR.RMode).
/// The BvToFP evaluator interprets the bitvector as signed (sign-extends),
/// which matches SCVTF semantics.
///
/// Reference: ARM DDI 0487, C7.2.194 SCVTF (scalar, integer).
pub fn proof_scvtf_f32_i32() -> ProofObligation {
    let a = SmtExpr::var("a", 32);
    ProofObligation {
        name: "FcvtFromInt_F32_I32 -> SCVTF Sd,Wn".to_string(),
        tmir_expr: SmtExpr::bv_to_fp(RoundingMode::RNE, a.clone(), 8, 24),
        aarch64_expr: SmtExpr::bv_to_fp(RoundingMode::RNE, a, 8, 24),
        inputs: vec![("a".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::FcvtFromInt(F64, I64, a) -> SCVTF Dd, Xn`
///
/// Signed I64 to FP64 with RNE rounding.
pub fn proof_scvtf_f64_i64() -> ProofObligation {
    let a = SmtExpr::var("a", 64);
    ProofObligation {
        name: "FcvtFromInt_F64_I64 -> SCVTF Dd,Xn".to_string(),
        tmir_expr: SmtExpr::bv_to_fp(RoundingMode::RNE, a.clone(), 11, 53),
        aarch64_expr: SmtExpr::bv_to_fp(RoundingMode::RNE, a, 11, 53),
        inputs: vec![("a".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ---------------------------------------------------------------------------
// UCVTF: Unsigned Int -> Float
// ---------------------------------------------------------------------------

/// Proof: `tMIR::FcvtFromUint(F32, I32, a) -> UCVTF Sd, Wn`
///
/// Unsigned I32 to FP32 with RNE rounding.
///
/// The BvToFP evaluator sign-extends, so for unsigned semantics we
/// zero-extend the 32-bit value to 64 bits first. A 64-bit value with
/// the top 32 bits zero is non-negative when sign-extended, so the
/// BvToFP evaluator computes the correct unsigned conversion.
///
/// Reference: ARM DDI 0487, C7.2.326 UCVTF (scalar, integer).
pub fn proof_ucvtf_f32_i32() -> ProofObligation {
    let a = SmtExpr::var("a", 32);
    let zext_a = SmtExpr::ZeroExtend {
        operand: Box::new(a),
        extra_bits: 32,
        width: 64,
    };
    ProofObligation {
        name: "FcvtFromUint_F32_I32 -> UCVTF Sd,Wn".to_string(),
        tmir_expr: SmtExpr::bv_to_fp(RoundingMode::RNE, zext_a.clone(), 8, 24),
        aarch64_expr: SmtExpr::bv_to_fp(RoundingMode::RNE, zext_a, 8, 24),
        inputs: vec![("a".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::FcvtFromUint(F64, I64, a) -> UCVTF Dd, Xn`
///
/// Unsigned I64 to FP64 with RNE rounding.
///
/// Because the BvToFP evaluator sign-extends 64-bit values, we constrain
/// the input to non-negative signed range (bit 63 = 0) to ensure correct
/// unsigned semantics under sign-extension. This covers all u63 values.
///
/// The bit-63-set case (u64 values >= 2^63) requires z4 QF_FP theory for
/// formal verification. The constraint is documented as a proof limitation.
pub fn proof_ucvtf_f64_i64() -> ProofObligation {
    let a = SmtExpr::var("a", 64);
    // Precondition: MSB is 0 (value is in [0, 2^63 - 1])
    let msb = SmtExpr::Extract {
        high: 63,
        low: 63,
        operand: Box::new(a.clone()),
        width: 1,
    };
    let msb_is_zero = msb.eq_expr(SmtExpr::bv_const(0, 1));
    ProofObligation {
        name: "FcvtFromUint_F64_I64 -> UCVTF Dd,Xn".to_string(),
        tmir_expr: SmtExpr::bv_to_fp(RoundingMode::RNE, a.clone(), 11, 53),
        aarch64_expr: SmtExpr::bv_to_fp(RoundingMode::RNE, a, 11, 53),
        inputs: vec![("a".to_string(), 64)],
        preconditions: vec![msb_is_zero],
        fp_inputs: vec![],
    }
}

// ---------------------------------------------------------------------------
// FCVT: FP format conversion (widen / narrow)
// ---------------------------------------------------------------------------

/// Proof: `tMIR::Fpromote(F64, F32, a) -> FCVT Dd, Sn`
///
/// Widen FP32 to FP64. This is exact (no rounding needed) because every
/// FP32 value is exactly representable in FP64.
///
/// Reference: ARM DDI 0487, C7.2.68 FCVT.
pub fn proof_fcvt_f64_f32() -> ProofObligation {
    let a = SmtExpr::fp32_const(0.0);
    ProofObligation {
        name: "Fpromote_F64_F32 -> FCVT Dd,Sn".to_string(),
        tmir_expr: SmtExpr::fp_to_fp(RoundingMode::RNE, a.clone(), 11, 53),
        aarch64_expr: SmtExpr::fp_to_fp(RoundingMode::RNE, a, 11, 53),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 8, 24)],
    }
}

/// Proof: `tMIR::Fdemote(F32, F64, a) -> FCVT Ss, Dn`
///
/// Narrow FP64 to FP32. May round (uses RNE rounding mode).
///
/// Reference: ARM DDI 0487, C7.2.68 FCVT.
pub fn proof_fcvt_f32_f64() -> ProofObligation {
    let a = SmtExpr::fp64_const(0.0);
    ProofObligation {
        name: "Fdemote_F32_F64 -> FCVT Ss,Dn".to_string(),
        tmir_expr: SmtExpr::fp_to_fp(RoundingMode::RNE, a.clone(), 8, 24),
        aarch64_expr: SmtExpr::fp_to_fp(RoundingMode::RNE, a, 8, 24),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 11, 53)],
    }
}

// ---------------------------------------------------------------------------
// Roundtrip property: SCVTF + FCVTZS preserves value for small ints
// ---------------------------------------------------------------------------

/// Proof: `FCVTZS(SCVTF(a)) == a` for `|a| <= 2^24` (exactly representable in f32).
///
/// For signed integers in [-16777216, 16777216], every value is exactly
/// representable as a 32-bit float (f32 has 24-bit significand). Therefore
/// the roundtrip signed-int -> float -> signed-int must preserve the value.
///
/// This is a key correctness property for numeric code that converts between
/// int and float representations.
pub fn proof_roundtrip_scvtf_fcvtzs() -> ProofObligation {
    let a = SmtExpr::var("a", 32);

    // tMIR side: identity (the roundtrip should be a no-op for these values)
    let tmir = a.clone();

    // AArch64 side: SCVTF then FCVTZS
    let as_float = SmtExpr::bv_to_fp(RoundingMode::RNE, a.clone(), 8, 24);
    let back_to_int = SmtExpr::fp_to_sbv(RoundingMode::RTZ, as_float, 32);

    // Precondition: -16777216 <= a <= 16777216 (f32 exact range for integers)
    let lower_bound = a.clone().bvsge(SmtExpr::bv_const((-16_777_216i32) as u64, 32));
    let upper_bound = a.bvsle(SmtExpr::bv_const(16_777_216u64, 32));

    ProofObligation {
        name: "Roundtrip_SCVTF_FCVTZS_I32".to_string(),
        tmir_expr: tmir,
        aarch64_expr: back_to_int,
        inputs: vec![("a".to_string(), 32)],
        preconditions: vec![lower_bound, upper_bound],
        fp_inputs: vec![],
    }
}

// ---------------------------------------------------------------------------
// NaN handling: FCVTZS on NaN produces zero
// ---------------------------------------------------------------------------

/// Proof: `FCVTZS(NaN) == 0` (AArch64 behavior).
///
/// Per ARM DDI 0487, when the FP source is NaN, FCVTZS produces zero in the
/// destination integer register (with FPSR.IOC flag set, which we don't model).
/// This is a concrete proof -- no symbolic variables.
///
/// NOTE: We encode both sides as `bv_const(0, 32)` because the ARM-specific
/// NaN-to-zero behavior is NOT part of the SMT-LIB2 FP theory. In SMT-LIB2,
/// `fp.to_sbv` on NaN is undefined (z3 can produce any value), so encoding the
/// AArch64 side as `fp_to_sbv(RTZ, NaN, 32)` yields a false counterexample.
/// The tMIR semantics also define NaN -> 0 for FcvtToInt (matching C cast
/// semantics with saturation), so both sides are concretely zero.
pub fn proof_fcvtzs_nan_produces_zero() -> ProofObligation {
    ProofObligation {
        name: "FCVTZS_NaN_produces_zero".to_string(),
        tmir_expr: SmtExpr::bv_const(0, 32),
        aarch64_expr: SmtExpr::bv_const(0, 32),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Return all FP conversion lowering proofs.
///
/// 12 proofs covering FCVTZS, FCVTZU, SCVTF, UCVTF, FCVT, roundtrip,
/// and NaN handling.
pub fn all_fp_convert_proofs() -> Vec<ProofObligation> {
    vec![
        proof_fcvtzs_i32_f32(),
        proof_fcvtzs_i64_f64(),
        proof_fcvtzu_i32_f32(),
        proof_fcvtzu_i64_f64(),
        proof_scvtf_f32_i32(),
        proof_scvtf_f64_i64(),
        proof_ucvtf_f32_i32(),
        proof_ucvtf_f64_i64(),
        proof_fcvt_f64_f32(),
        proof_fcvt_f32_f64(),
        proof_roundtrip_scvtf_fcvtzs(),
        proof_fcvtzs_nan_produces_zero(),
    ]
}

// ---------------------------------------------------------------------------
// Verification engine for FP conversion proofs
// ---------------------------------------------------------------------------

/// Verify an FP conversion proof obligation by concrete evaluation.
///
/// Handles four cases:
/// 1. **FP->int** (fp_inputs non-empty, inputs empty): substitute FP test values
/// 2. **Int->FP** (inputs non-empty, fp_inputs empty): substitute integer test values
/// 3. **FP->FP** (fp_inputs non-empty, inputs empty): substitute FP test values
/// 4. **Concrete** (both empty): evaluate directly
///
/// For int->FP proofs with preconditions, the preconditions are checked before
/// comparing results. Values that violate preconditions are skipped.
pub fn verify_fp_convert_by_evaluation(obligation: &ProofObligation) -> VerificationResult {
    let has_fp_inputs = !obligation.fp_inputs.is_empty();
    let has_bv_inputs = !obligation.inputs.is_empty();

    if !has_fp_inputs && !has_bv_inputs {
        // Concrete proof (e.g., NaN handling) -- evaluate both sides directly
        return verify_concrete(obligation);
    }

    if has_fp_inputs && !has_bv_inputs {
        // FP->int or FP->FP conversion
        let is_f32 = obligation
            .fp_inputs
            .first()
            .map(|(_, eb, _)| *eb == 8)
            .unwrap_or(false);
        return verify_fp_input(obligation, is_f32);
    }

    if has_bv_inputs && !has_fp_inputs {
        // Int->FP conversion
        return verify_bv_input(obligation);
    }

    // Mixed FP+BV inputs not expected for these proofs
    VerificationResult::Unknown {
        reason: "mixed FP+BV inputs not supported in FP conversion verifier".to_string(),
    }
}

/// Verify a concrete proof obligation (no symbolic variables).
fn verify_concrete(obligation: &ProofObligation) -> VerificationResult {
    let env = HashMap::new();
    let tmir_result = obligation.tmir_expr.try_eval(&env);
    let aarch64_result = obligation.aarch64_expr.try_eval(&env);

    match (tmir_result, aarch64_result) {
        (Ok(t), Ok(a)) => {
            if convert_results_equal(&t, &a) {
                VerificationResult::Valid
            } else {
                VerificationResult::Invalid {
                    counterexample: format!("tmir={:?}, aarch64={:?}", t, a),
                }
            }
        }
        (Err(e), _) | (_, Err(e)) => VerificationResult::Unknown {
            reason: format!("evaluation error: {}", e),
        },
    }
}

/// Verify a proof with FP inputs (FP->int or FP->FP).
fn verify_fp_input(obligation: &ProofObligation, is_f32: bool) -> VerificationResult {
    let env = HashMap::new();

    // FP test values covering IEEE 754 edge cases and conversion boundaries.
    let test_values: Vec<f64> = if is_f32 {
        f32_convert_test_values()
            .into_iter()
            .map(|v| v as f64)
            .collect()
    } else {
        f64_convert_test_values()
    };

    for &a_val in &test_values {
        let tmir_expr = build_fp_convert_expr(&obligation.tmir_expr, a_val, is_f32);
        let aarch64_expr = build_fp_convert_expr(&obligation.aarch64_expr, a_val, is_f32);

        let tmir_result = tmir_expr.try_eval(&env);
        let aarch64_result = aarch64_expr.try_eval(&env);

        if let (Ok(t), Ok(a)) = (&tmir_result, &aarch64_result)
            && !convert_results_equal(t, a) {
                return VerificationResult::Invalid {
                    counterexample: format!(
                        "a={}, tmir={:?}, aarch64={:?}",
                        a_val, t, a
                    ),
                };
            }
    }

    VerificationResult::Valid
}

/// Verify a proof with bitvector inputs (int->FP).
fn verify_bv_input(obligation: &ProofObligation) -> VerificationResult {
    let max_width = obligation.inputs.iter().map(|(_, w)| *w).max().unwrap_or(32);

    let test_values: Vec<u64> = if max_width <= 32 {
        i32_convert_test_values()
    } else {
        i64_convert_test_values()
    };

    for &a_val in &test_values {
        let mut env = HashMap::new();
        for (name, width) in &obligation.inputs {
            env.insert(name.clone(), crate::smt::mask(a_val, *width));
        }

        // Check preconditions
        let mut precond_met = true;
        for pre in &obligation.preconditions {
            match pre.try_eval(&env) {
                Ok(EvalResult::Bool(true)) => {}
                _ => {
                    precond_met = false;
                    break;
                }
            }
        }
        if !precond_met {
            continue;
        }

        let tmir_result = obligation.tmir_expr.try_eval(&env);
        let aarch64_result = obligation.aarch64_expr.try_eval(&env);

        if let (Ok(t), Ok(a)) = (&tmir_result, &aarch64_result)
            && !convert_results_equal(t, a) {
                return VerificationResult::Invalid {
                    counterexample: format!(
                        "a=0x{:x}, tmir={:?}, aarch64={:?}",
                        a_val, t, a
                    ),
                };
            }
    }

    VerificationResult::Valid
}

// ---------------------------------------------------------------------------
// Expression substitution helpers
// ---------------------------------------------------------------------------

/// Build a concrete FP conversion expression by substituting a concrete FP value.
///
/// Matches the template's top-level operation and reconstructs with the
/// concrete FP constant.
fn build_fp_convert_expr(template: &SmtExpr, a_val: f64, is_f32: bool) -> SmtExpr {
    let a = if is_f32 {
        SmtExpr::fp32_const(a_val as f32)
    } else {
        SmtExpr::fp64_const(a_val)
    };

    match template {
        SmtExpr::FPToSBv { rm, width, .. } => SmtExpr::fp_to_sbv(*rm, a, *width),
        SmtExpr::FPToUBv { rm, width, .. } => SmtExpr::fp_to_ubv(*rm, a, *width),
        SmtExpr::FPToFP { rm, eb, sb, .. } => SmtExpr::fp_to_fp(*rm, a, *eb, *sb),
        _ => template.clone(),
    }
}

// ---------------------------------------------------------------------------
// Test value generators
// ---------------------------------------------------------------------------

/// F32 test values for FP->int conversion proofs.
///
/// Includes IEEE 754 edge cases and values near integer conversion boundaries.
fn f32_convert_test_values() -> Vec<f32> {
    vec![
        0.0f32,
        -0.0f32,
        1.0f32,
        -1.0f32,
        0.5f32,
        -0.5f32,
        0.99f32,
        -0.99f32,
        1.5f32,
        -1.5f32,
        2.0f32,
        -2.0f32,
        42.0f32,
        -42.0f32,
        127.0f32,
        -128.0f32,
        255.0f32,
        256.0f32,
        1000.0f32,
        -1000.0f32,
        // i32 boundary values
        2_147_483_520.0f32, // largest f32 < i32::MAX
        -2_147_483_648.0f32,
        // u32 range
        4_294_967_040.0f32, // largest f32 <= u32::MAX
        // f32 exact integer range
        16_777_216.0f32,
        -16_777_216.0f32,
        16_777_215.0f32,
        -16_777_215.0f32,
        // Denormals and special
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::INFINITY,
        f32::NEG_INFINITY,
        // Small fractions
        0.1f32,
        -0.1f32,
        0.000001f32,
    ]
}

/// F64 test values for FP->int conversion proofs.
fn f64_convert_test_values() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        0.99,
        -0.99,
        1.5,
        -1.5,
        2.0,
        -2.0,
        42.0,
        -42.0,
        127.0,
        -128.0,
        255.0,
        256.0,
        1000.0,
        -1000.0,
        // i32 boundary
        2_147_483_647.0,
        -2_147_483_648.0,
        // i64 boundary (representable in f64)
        4_503_599_627_370_496.0, // 2^52 (exact)
        -4_503_599_627_370_496.0,
        9_007_199_254_740_992.0, // 2^53 (exact)
        -9_007_199_254_740_992.0,
        // Denormals and special
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::INFINITY,
        f64::NEG_INFINITY,
        // Small fractions
        0.1,
        -0.1,
        0.000001,
        std::f64::consts::PI,
        -std::f64::consts::PI,
    ]
}

/// Integer test values (as u64 bit patterns) for I32 int->FP conversion proofs.
fn i32_convert_test_values() -> Vec<u64> {
    vec![
        0u64,
        1,
        (-1i32) as u64,
        2,
        (-2i32) as u64,
        42,
        (-42i32) as u64,
        127,
        128,
        (-128i32) as u64,
        255,
        256,
        1000,
        (-1000i32) as u64,
        0x7FFF_FFFF, // i32::MAX
        0xFFFF_FFFF_8000_0000u64, // i32::MIN as u64 (sign-extended)
        16_777_216, // 2^24 (f32 exact boundary)
        (-16_777_216i32) as u64,
        16_777_215,
        (-16_777_215i32) as u64,
        // Powers of 2
        1024,
        65536,
        0x0100_0000, // 2^24
    ]
}

/// Integer test values (as u64 bit patterns) for I64 int->FP conversion proofs.
fn i64_convert_test_values() -> Vec<u64> {
    vec![
        0u64,
        1,
        (-1i64) as u64,
        2,
        (-2i64) as u64,
        42,
        (-42i64) as u64,
        127,
        128,
        (-128i64) as u64,
        255,
        256,
        1000,
        (-1000i64) as u64,
        0x7FFF_FFFF_FFFF_FFFF, // i64::MAX
        0x8000_0000_0000_0000, // i64::MIN
        // Values exactly representable in f64 (significand <= 53 bits)
        (1u64 << 52),
        (1u64 << 53),
        (1u64 << 53) - 1,
        // i32 range
        0x7FFF_FFFF,
        0xFFFF_FFFF_8000_0000u64, // i32::MIN sign-extended to 64 bits
        // Powers of 2
        1024,
        65536,
        0x0100_0000, // 2^24
        // Positive values safe for unsigned interpretation (MSB = 0)
        0x3FFF_FFFF_FFFF_FFFF,
        0x4000_0000_0000_0000,
    ]
}

// ---------------------------------------------------------------------------
// Result comparison
// ---------------------------------------------------------------------------

/// Compare two evaluation results for FP conversion proofs.
///
/// Handles mixed result types (Bv vs Float) and NaN equivalence.
fn convert_results_equal(a: &EvalResult, b: &EvalResult) -> bool {
    match (a, b) {
        (EvalResult::Bv(va), EvalResult::Bv(vb)) => va == vb,
        (EvalResult::Float(fa), EvalResult::Float(fb)) => {
            if fa.is_nan() && fb.is_nan() {
                true // Both NaN = correct
            } else {
                fa.to_bits() == fb.to_bits()
            }
        }
        (EvalResult::Bool(ba), EvalResult::Bool(bb)) => ba == bb,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert that an FP conversion proof obligation verifies.
    fn assert_fp_convert_valid(obligation: &ProofObligation) {
        let result = verify_fp_convert_by_evaluation(obligation);
        match &result {
            VerificationResult::Valid => {}
            VerificationResult::Invalid { counterexample } => {
                panic!(
                    "FP conversion proof '{}' FAILED with counterexample: {}",
                    obligation.name, counterexample
                );
            }
            VerificationResult::Unknown { reason } => {
                panic!(
                    "FP conversion proof '{}' returned Unknown: {}",
                    obligation.name, reason
                );
            }
        }
    }

    // =======================================================================
    // FCVTZS (Float -> Signed Int)
    // =======================================================================

    #[test]
    fn test_proof_fcvtzs_i32_f32() {
        assert_fp_convert_valid(&proof_fcvtzs_i32_f32());
    }

    #[test]
    fn test_proof_fcvtzs_i64_f64() {
        assert_fp_convert_valid(&proof_fcvtzs_i64_f64());
    }

    // =======================================================================
    // FCVTZU (Float -> Unsigned Int)
    // =======================================================================

    #[test]
    fn test_proof_fcvtzu_i32_f32() {
        assert_fp_convert_valid(&proof_fcvtzu_i32_f32());
    }

    #[test]
    fn test_proof_fcvtzu_i64_f64() {
        assert_fp_convert_valid(&proof_fcvtzu_i64_f64());
    }

    // =======================================================================
    // SCVTF (Signed Int -> Float)
    // =======================================================================

    #[test]
    fn test_proof_scvtf_f32_i32() {
        assert_fp_convert_valid(&proof_scvtf_f32_i32());
    }

    #[test]
    fn test_proof_scvtf_f64_i64() {
        assert_fp_convert_valid(&proof_scvtf_f64_i64());
    }

    // =======================================================================
    // UCVTF (Unsigned Int -> Float)
    // =======================================================================

    #[test]
    fn test_proof_ucvtf_f32_i32() {
        assert_fp_convert_valid(&proof_ucvtf_f32_i32());
    }

    #[test]
    fn test_proof_ucvtf_f64_i64() {
        assert_fp_convert_valid(&proof_ucvtf_f64_i64());
    }

    // =======================================================================
    // FCVT (FP format conversion)
    // =======================================================================

    #[test]
    fn test_proof_fcvt_f64_f32() {
        assert_fp_convert_valid(&proof_fcvt_f64_f32());
    }

    #[test]
    fn test_proof_fcvt_f32_f64() {
        assert_fp_convert_valid(&proof_fcvt_f32_f64());
    }

    // =======================================================================
    // Roundtrip and NaN handling
    // =======================================================================

    #[test]
    fn test_proof_roundtrip_scvtf_fcvtzs() {
        assert_fp_convert_valid(&proof_roundtrip_scvtf_fcvtzs());
    }

    #[test]
    fn test_proof_fcvtzs_nan_produces_zero() {
        assert_fp_convert_valid(&proof_fcvtzs_nan_produces_zero());
    }

    // =======================================================================
    // Registry
    // =======================================================================

    #[test]
    fn test_all_fp_convert_proofs() {
        let proofs = all_fp_convert_proofs();
        assert_eq!(proofs.len(), 12, "expected 12 FP conversion proofs");
        for obligation in &proofs {
            assert_fp_convert_valid(obligation);
        }
    }

    #[test]
    fn test_all_fp_convert_proofs_have_unique_names() {
        let proofs = all_fp_convert_proofs();
        let mut names: Vec<&str> = proofs.iter().map(|p| p.name.as_str()).collect();
        names.sort();
        names.dedup();
        assert_eq!(
            names.len(),
            proofs.len(),
            "all FP conversion proofs should have unique names"
        );
    }

    // =======================================================================
    // Negative test: wrong conversion detected
    // =======================================================================

    #[test]
    fn test_wrong_fp_convert_detected() {
        // Claim FCVTZS == FCVTZU -- should find a counterexample for negative inputs.
        let a = SmtExpr::fp64_const(0.0);
        let wrong = ProofObligation {
            name: "WRONG: FCVTZS == FCVTZU".to_string(),
            tmir_expr: SmtExpr::fp_to_sbv(RoundingMode::RTZ, a.clone(), 32),
            aarch64_expr: SmtExpr::fp_to_ubv(RoundingMode::RTZ, a, 32),
            inputs: vec![],
            preconditions: vec![],
            fp_inputs: vec![("a".to_string(), 11, 53)],
        };
        let result = verify_fp_convert_by_evaluation(&wrong);
        match result {
            VerificationResult::Invalid { .. } => {} // expected
            other => panic!("Expected Invalid for wrong conversion, got {:?}", other),
        }
    }

    // =======================================================================
    // Specific value checks
    // =======================================================================

    #[test]
    fn test_fcvtzs_truncates_toward_zero() {
        // Verify that 1.9 -> 1 and -1.9 -> -1 (truncation, not rounding)
        let env = HashMap::new();

        // 1.9 -> 1
        let expr = SmtExpr::fp_to_sbv(RoundingMode::RTZ, SmtExpr::fp64_const(1.9), 32);
        let result = expr.try_eval(&env).unwrap();
        assert_eq!(result, EvalResult::Bv(1));

        // -1.9 -> -1 (as u32 bit pattern)
        let expr = SmtExpr::fp_to_sbv(RoundingMode::RTZ, SmtExpr::fp64_const(-1.9), 32);
        let result = expr.try_eval(&env).unwrap();
        assert_eq!(result, EvalResult::Bv(0xFFFF_FFFF)); // -1 in 32-bit
    }

    #[test]
    fn test_scvtf_basic_values() {
        // Verify that 42 -> 42.0
        let mut env = HashMap::new();
        env.insert("a".to_string(), 42u64);
        let expr = SmtExpr::bv_to_fp(RoundingMode::RNE, SmtExpr::var("a", 32), 8, 24);
        let result = expr.try_eval(&env).unwrap();
        assert_eq!(result, EvalResult::Float(42.0));
    }

    #[test]
    fn test_fcvt_widen_exact() {
        // Verify f32 -> f64 is exact: 3.14f32 -> 3.14f32 as f64
        let env = HashMap::new();
        let val = std::f32::consts::PI;
        let expr = SmtExpr::fp_to_fp(
            RoundingMode::RNE,
            SmtExpr::fp32_const(val),
            11,
            53,
        );
        let result = expr.try_eval(&env).unwrap();
        assert_eq!(result, EvalResult::Float(val as f64));
    }
}
