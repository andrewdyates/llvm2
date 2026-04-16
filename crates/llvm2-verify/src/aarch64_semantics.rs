// llvm2-verify/aarch64_semantics.rs - AArch64 instruction semantics as SMT formulas
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Encodes AArch64 instruction semantics as bitvector SMT expressions.
// Each instruction maps to a pure function from input bitvectors to output
// bitvectors, modeling the instruction's effect on destination registers.
//
// Reference: ARM Architecture Reference Manual (DDI 0487), Section C6.
// Reference: designs/2026-04-13-verification-architecture.md

//! AArch64 instruction semantics encoded as [`SmtExpr`] bitvector formulas.
//!
//! Key principle: 32-bit operations (W registers) produce 32-bit results.
//! The zero-extension to 64-bit X registers is verified separately as a lemma.
//! When verifying a 32-bit lowering rule, we compare 32-bit tMIR result with
//! 32-bit AArch64 result.

use crate::smt::SmtExpr;
use llvm2_ir::cc::OperandSize;

/// Encode `ADD Wd, Wn, Wm` or `ADD Xd, Xn, Xm` — register-register add.
///
/// Semantics: `Rd = Rn + Rm` (wrapping).
pub fn encode_add_rr(size: OperandSize, rn: SmtExpr, rm: SmtExpr) -> SmtExpr {
    let _ = size; // Width is carried by the expressions themselves.
    rn.bvadd(rm)
}

/// Encode `SUB Wd, Wn, Wm` or `SUB Xd, Xn, Xm` — register-register subtract.
///
/// Semantics: `Rd = Rn - Rm` (wrapping).
pub fn encode_sub_rr(size: OperandSize, rn: SmtExpr, rm: SmtExpr) -> SmtExpr {
    let _ = size;
    rn.bvsub(rm)
}

/// Encode `MUL Wd, Wn, Wm` or `MUL Xd, Xn, Xm` — register-register multiply.
///
/// On AArch64 this is actually `MADD Rd, Rn, Rm, XZR` (multiply-add with zero).
/// Semantics: `Rd = Rn * Rm` (wrapping, lower bits).
pub fn encode_mul_rr(size: OperandSize, rn: SmtExpr, rm: SmtExpr) -> SmtExpr {
    let _ = size;
    rn.bvmul(rm)
}

/// Encode `SDIV Wd, Wn, Wm` or `SDIV Xd, Xn, Xm` — signed divide.
///
/// Semantics: `Rd = Rn /s Rm` (truncation toward zero).
/// Precondition: `Rm != 0` (division by zero is UB in tMIR; AArch64 SDIV
/// returns 0, but we only need to verify when precondition holds).
pub fn encode_sdiv_rr(size: OperandSize, rn: SmtExpr, rm: SmtExpr) -> SmtExpr {
    let _ = size;
    rn.bvsdiv(rm)
}

/// Encode `UDIV Wd, Wn, Wm` or `UDIV Xd, Xn, Xm` — unsigned divide.
///
/// Semantics: `Rd = Rn /u Rm`.
/// Precondition: `Rm != 0`.
pub fn encode_udiv_rr(size: OperandSize, rn: SmtExpr, rm: SmtExpr) -> SmtExpr {
    let _ = size;
    rn.bvudiv(rm)
}

/// Encode `NEG Wd, Wn` or `NEG Xd, Xn` — negate.
///
/// On AArch64, `NEG Rd, Rn` is an alias for `SUB Rd, XZR/WZR, Rn`.
/// Semantics: `Rd = 0 - Rn` (two's complement negation).
pub fn encode_neg(size: OperandSize, rn: SmtExpr) -> SmtExpr {
    let _ = size;
    rn.bvneg()
}

// ---------------------------------------------------------------------------
// Floating-point instruction semantics
// ---------------------------------------------------------------------------

/// Floating-point precision selector.
///
/// Maps to AArch64 S (single) and D (double) register sizes for FP operations.
/// Reference: ARM DDI 0487, Section C7 (SIMD and Floating-Point Instructions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FPSize {
    /// Single-precision (32-bit, S registers). IEEE 754 binary32.
    Single,
    /// Double-precision (64-bit, D registers). IEEE 754 binary64.
    Double,
}

impl FPSize {
    /// Exponent bits for this FP size.
    pub fn eb(self) -> u32 {
        match self {
            FPSize::Single => 8,
            FPSize::Double => 11,
        }
    }

    /// Significand bits (including implicit bit) for this FP size.
    pub fn sb(self) -> u32 {
        match self {
            FPSize::Single => 24,
            FPSize::Double => 53,
        }
    }
}

/// Encode `FADD Sd, Sn, Sm` or `FADD Dd, Dn, Dm` -- floating-point add.
///
/// Semantics: `Fd = Fn + Fm` using RNE rounding mode (default FPCR.RMode).
/// Reference: ARM DDI 0487, C7.2.74 FADD (scalar).
pub fn encode_fadd_rr(_size: FPSize, rn: SmtExpr, rm: SmtExpr) -> SmtExpr {
    use crate::smt::RoundingMode;
    SmtExpr::fp_add(RoundingMode::RNE, rn, rm)
}

/// Encode `FSUB Sd, Sn, Sm` or `FSUB Dd, Dn, Dm` -- floating-point subtract.
///
/// Semantics: `Fd = Fn - Fm` using RNE rounding mode.
/// Reference: ARM DDI 0487, C7.2.161 FSUB (scalar).
pub fn encode_fsub_rr(_size: FPSize, rn: SmtExpr, rm: SmtExpr) -> SmtExpr {
    use crate::smt::RoundingMode;
    SmtExpr::fp_sub(RoundingMode::RNE, rn, rm)
}

/// Encode `FMUL Sd, Sn, Sm` or `FMUL Dd, Dn, Dm` -- floating-point multiply.
///
/// Semantics: `Fd = Fn * Fm` using RNE rounding mode.
/// Reference: ARM DDI 0487, C7.2.128 FMUL (scalar).
pub fn encode_fmul_rr(_size: FPSize, rn: SmtExpr, rm: SmtExpr) -> SmtExpr {
    use crate::smt::RoundingMode;
    SmtExpr::fp_mul(RoundingMode::RNE, rn, rm)
}

/// Encode `FNEG Sd, Sn` or `FNEG Dd, Dn` -- floating-point negate.
///
/// Semantics: `Fd = -Fn` (bitwise sign flip, no rounding needed).
/// Reference: ARM DDI 0487, C7.2.132 FNEG (scalar).
pub fn encode_fneg(_size: FPSize, rn: SmtExpr) -> SmtExpr {
    rn.fp_neg()
}

/// Encode `FDIV Sd, Sn, Sm` or `FDIV Dd, Dn, Dm` -- floating-point divide.
///
/// Semantics: `Fd = Fn / Fm` using RNE rounding mode.
/// Reference: ARM DDI 0487, C7.2.77 FDIV (scalar).
pub fn encode_fdiv_rr(_size: FPSize, rn: SmtExpr, rm: SmtExpr) -> SmtExpr {
    use crate::smt::RoundingMode;
    SmtExpr::fp_div(RoundingMode::RNE, rn, rm)
}

/// Encode `FCMP Sn, Sm` / `FCMP Dn, Dm` + condition code extraction.
///
/// AArch64 FCMP sets NZCV flags; the condition code determines the boolean
/// result. This function returns a 1-bit bitvector: `bv1(1)` if the
/// condition holds, `bv1(0)` otherwise.
///
/// All 14 `FloatCC` variants are supported, covering both ordered (false
/// when NaN) and unordered (true when NaN) comparisons.
///
/// Reference: ARM DDI 0487, C7.2.76 FCMP (scalar).
pub fn encode_fcmp(_size: FPSize, rn: SmtExpr, rm: SmtExpr, cond: &llvm2_lower::instructions::FloatCC) -> SmtExpr {
    use llvm2_lower::instructions::FloatCC;

    let a_nan = rn.clone().fp_is_nan();
    let b_nan = rm.clone().fp_is_nan();
    let either_nan = a_nan.clone().or_expr(b_nan.clone());

    let bool_result = match cond {
        // Ordered comparisons (false when NaN)
        FloatCC::Equal => rn.fp_eq(rm),
        FloatCC::NotEqual => rn.fp_eq(rm).not_expr(),
        FloatCC::LessThan => rn.fp_lt(rm),
        FloatCC::LessThanOrEqual => rn.fp_le(rm),
        FloatCC::GreaterThan => rn.fp_gt(rm),
        FloatCC::GreaterThanOrEqual => rn.fp_ge(rm),
        FloatCC::Ordered => a_nan.not_expr().and_expr(b_nan.not_expr()),
        FloatCC::Unordered => either_nan,
        // Unordered comparisons (true when NaN)
        FloatCC::UnorderedEqual => rn.fp_eq(rm).or_expr(either_nan),
        FloatCC::UnorderedNotEqual => rn.fp_eq(rm).not_expr().or_expr(either_nan),
        FloatCC::UnorderedLessThan => rn.fp_lt(rm).or_expr(either_nan),
        FloatCC::UnorderedLessThanOrEqual => rn.fp_le(rm).or_expr(either_nan),
        FloatCC::UnorderedGreaterThan => rn.fp_gt(rm).or_expr(either_nan),
        FloatCC::UnorderedGreaterThanOrEqual => rn.fp_ge(rm).or_expr(either_nan),
    };

    SmtExpr::ite(bool_result, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1))
}

// ---------------------------------------------------------------------------
// Bitwise instruction semantics
// ---------------------------------------------------------------------------

/// Encode `AND Wd, Wn, Wm` or `AND Xd, Xn, Xm` — bitwise AND.
///
/// Semantics: `Rd = Rn & Rm`.
/// Reference: ARM DDI 0487, C6.2.12 AND (shifted register).
pub fn encode_and_rr(size: OperandSize, rn: SmtExpr, rm: SmtExpr) -> SmtExpr {
    let _ = size;
    rn.bvand(rm)
}

/// Encode `ORR Wd, Wn, Wm` or `ORR Xd, Xn, Xm` — bitwise inclusive OR.
///
/// Semantics: `Rd = Rn | Rm`.
/// Reference: ARM DDI 0487, C6.2.230 ORR (shifted register).
pub fn encode_orr_rr(size: OperandSize, rn: SmtExpr, rm: SmtExpr) -> SmtExpr {
    let _ = size;
    rn.bvor(rm)
}

/// Encode `EOR Wd, Wn, Wm` or `EOR Xd, Xn, Xm` — bitwise exclusive OR.
///
/// Semantics: `Rd = Rn ^ Rm`.
/// Reference: ARM DDI 0487, C6.2.87 EOR (shifted register).
pub fn encode_eor_rr(size: OperandSize, rn: SmtExpr, rm: SmtExpr) -> SmtExpr {
    let _ = size;
    rn.bvxor(rm)
}

/// Encode `MVN Wd, Wm` or `MVN Xd, Xm` — bitwise NOT (move NOT).
///
/// On AArch64, `MVN Rd, Rm` is an alias for `ORN Rd, XZR/WZR, Rm`.
/// Semantics: `Rd = ~Rm` (bitwise complement).
/// Reference: ARM DDI 0487, C6.2.192 MVN.
pub fn encode_mvn(size: OperandSize, rn: SmtExpr) -> SmtExpr {
    let width = operand_size_bits(size);
    let all_ones = SmtExpr::bv_const(crate::smt::mask(u64::MAX, width), width);
    rn.bvxor(all_ones)
}

// ---------------------------------------------------------------------------
// Shift instruction semantics
// ---------------------------------------------------------------------------

/// Encode `LSL Wd, Wn, Wm` or `LSL Xd, Xn, Xm` — logical shift left.
///
/// On AArch64, `LSL Rd, Rn, Rm` is an alias for `LSLV Rd, Rn, Rm`.
/// Semantics: `Rd = Rn << (Rm mod width)`.
/// Reference: ARM DDI 0487, C6.2.171 LSL (register).
pub fn encode_lsl_rr(size: OperandSize, rn: SmtExpr, rm: SmtExpr) -> SmtExpr {
    let _ = size;
    rn.bvshl(rm)
}

/// Encode `LSR Wd, Wn, Wm` or `LSR Xd, Xn, Xm` — logical shift right.
///
/// On AArch64, `LSR Rd, Rn, Rm` is an alias for `LSRV Rd, Rn, Rm`.
/// Semantics: `Rd = Rn >> (Rm mod width)` (unsigned / zero-fill).
/// Reference: ARM DDI 0487, C6.2.173 LSR (register).
pub fn encode_lsr_rr(size: OperandSize, rn: SmtExpr, rm: SmtExpr) -> SmtExpr {
    let _ = size;
    rn.bvlshr(rm)
}

/// Encode `ASR Wd, Wn, Wm` or `ASR Xd, Xn, Xm` — arithmetic shift right.
///
/// On AArch64, `ASR Rd, Rn, Rm` is an alias for `ASRV Rd, Rn, Rm`.
/// Semantics: `Rd = Rn >>_s (Rm mod width)` (sign-extending).
/// Reference: ARM DDI 0487, C6.2.16 ASR (register).
pub fn encode_asr_rr(size: OperandSize, rn: SmtExpr, rm: SmtExpr) -> SmtExpr {
    let _ = size;
    rn.bvashr(rm)
}

/// Width in bits for an OperandSize.
pub fn operand_size_bits(size: OperandSize) -> u32 {
    match size {
        OperandSize::S32 => 32,
        OperandSize::S64 => 64,
    }
}

/// Map an OperandSize to the corresponding llvm2-lower Type.
pub fn operand_size_to_type(size: OperandSize) -> llvm2_lower::types::Type {
    match size {
        OperandSize::S32 => llvm2_lower::types::Type::I32,
        OperandSize::S64 => llvm2_lower::types::Type::I64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smt::EvalResult;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn sym32() -> (SmtExpr, SmtExpr) {
        (SmtExpr::var("a", 32), SmtExpr::var("b", 32))
    }

    #[test]
    fn test_add_rr_32() {
        let (a, b) = sym32();
        let expr = encode_add_rr(OperandSize::S32, a, b);
        let result = expr.eval(&env(&[("a", 100), ("b", 200)]));
        assert_eq!(result, EvalResult::Bv(300));
    }

    #[test]
    fn test_sub_rr_32() {
        let (a, b) = sym32();
        let expr = encode_sub_rr(OperandSize::S32, a, b);
        let result = expr.eval(&env(&[("a", 10), ("b", 3)]));
        assert_eq!(result, EvalResult::Bv(7));
    }

    #[test]
    fn test_mul_rr_32() {
        let (a, b) = sym32();
        let expr = encode_mul_rr(OperandSize::S32, a, b);
        let result = expr.eval(&env(&[("a", 6), ("b", 7)]));
        assert_eq!(result, EvalResult::Bv(42));
    }

    #[test]
    fn test_neg_32() {
        let a = SmtExpr::var("a", 32);
        let expr = encode_neg(OperandSize::S32, a);
        let result = expr.eval(&env(&[("a", 1)]));
        assert_eq!(result, EvalResult::Bv(0xFFFF_FFFF));
    }

    #[test]
    fn test_neg_zero() {
        let a = SmtExpr::var("a", 32);
        let expr = encode_neg(OperandSize::S32, a);
        let result = expr.eval(&env(&[("a", 0)]));
        assert_eq!(result, EvalResult::Bv(0));
    }

    // -----------------------------------------------------------------------
    // Floating-point instruction semantics tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fadd_single() {
        let a = SmtExpr::fp32_const(1.5f32);
        let b = SmtExpr::fp32_const(2.5f32);
        let expr = encode_fadd_rr(FPSize::Single, a, b);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(4.0));
    }

    #[test]
    fn test_fadd_double() {
        let a = SmtExpr::fp64_const(100.0);
        let b = SmtExpr::fp64_const(200.0);
        let expr = encode_fadd_rr(FPSize::Double, a, b);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(300.0));
    }

    #[test]
    fn test_fsub_single() {
        let a = SmtExpr::fp32_const(10.0f32);
        let b = SmtExpr::fp32_const(3.5f32);
        let expr = encode_fsub_rr(FPSize::Single, a, b);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(6.5));
    }

    #[test]
    fn test_fsub_double() {
        let a = SmtExpr::fp64_const(100.0);
        let b = SmtExpr::fp64_const(42.0);
        let expr = encode_fsub_rr(FPSize::Double, a, b);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(58.0));
    }

    #[test]
    fn test_fmul_single() {
        let a = SmtExpr::fp32_const(3.0f32);
        let b = SmtExpr::fp32_const(7.0f32);
        let expr = encode_fmul_rr(FPSize::Single, a, b);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(21.0));
    }

    #[test]
    fn test_fmul_double() {
        let a = SmtExpr::fp64_const(6.0);
        let b = SmtExpr::fp64_const(7.0);
        let expr = encode_fmul_rr(FPSize::Double, a, b);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(42.0));
    }

    #[test]
    fn test_fneg_single() {
        let a = SmtExpr::fp32_const(42.0f32);
        let expr = encode_fneg(FPSize::Single, a);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(-42.0));
    }

    #[test]
    fn test_fneg_double() {
        let a = SmtExpr::fp64_const(std::f64::consts::PI);
        let expr = encode_fneg(FPSize::Double, a);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(-std::f64::consts::PI));
    }

    #[test]
    fn test_fneg_double_negative() {
        let a = SmtExpr::fp64_const(-100.0);
        let expr = encode_fneg(FPSize::Double, a);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(100.0));
    }

    #[test]
    fn test_fdiv_single() {
        let a = SmtExpr::fp32_const(10.0f32);
        let b = SmtExpr::fp32_const(4.0f32);
        let expr = encode_fdiv_rr(FPSize::Single, a, b);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(2.5));
    }

    #[test]
    fn test_fdiv_double() {
        let a = SmtExpr::fp64_const(10.0);
        let b = SmtExpr::fp64_const(4.0);
        let expr = encode_fdiv_rr(FPSize::Double, a, b);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(2.5));
    }

    #[test]
    fn test_fcmp_eq_true() {
        use llvm2_lower::instructions::FloatCC;
        let a = SmtExpr::fp64_const(1.0);
        let b = SmtExpr::fp64_const(1.0);
        let expr = encode_fcmp(FPSize::Double, a, b, &FloatCC::Equal);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Bv(1));
    }

    #[test]
    fn test_fcmp_eq_false() {
        use llvm2_lower::instructions::FloatCC;
        let a = SmtExpr::fp64_const(1.0);
        let b = SmtExpr::fp64_const(2.0);
        let expr = encode_fcmp(FPSize::Double, a, b, &FloatCC::Equal);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Bv(0));
    }

    #[test]
    fn test_fcmp_lt_true() {
        use llvm2_lower::instructions::FloatCC;
        let a = SmtExpr::fp64_const(1.0);
        let b = SmtExpr::fp64_const(2.0);
        let expr = encode_fcmp(FPSize::Double, a, b, &FloatCC::LessThan);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Bv(1));
    }

    #[test]
    fn test_fcmp_gt_true() {
        use llvm2_lower::instructions::FloatCC;
        let a = SmtExpr::fp64_const(3.0);
        let b = SmtExpr::fp64_const(2.0);
        let expr = encode_fcmp(FPSize::Double, a, b, &FloatCC::GreaterThan);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Bv(1));
    }

    #[test]
    fn test_fcmp_ordered_no_nan() {
        use llvm2_lower::instructions::FloatCC;
        let a = SmtExpr::fp64_const(1.0);
        let b = SmtExpr::fp64_const(2.0);
        let expr = encode_fcmp(FPSize::Double, a, b, &FloatCC::Ordered);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Bv(1));
    }

    #[test]
    fn test_fcmp_unordered_no_nan() {
        use llvm2_lower::instructions::FloatCC;
        let a = SmtExpr::fp64_const(1.0);
        let b = SmtExpr::fp64_const(2.0);
        let expr = encode_fcmp(FPSize::Double, a, b, &FloatCC::Unordered);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Bv(0));
    }

    #[test]
    fn test_fp_size_parameters() {
        assert_eq!(FPSize::Single.eb(), 8);
        assert_eq!(FPSize::Single.sb(), 24);
        assert_eq!(FPSize::Double.eb(), 11);
        assert_eq!(FPSize::Double.sb(), 53);
    }

    // -----------------------------------------------------------------------
    // Bitwise instruction semantics tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_and_rr_32() {
        let (a, b) = sym32();
        let expr = encode_and_rr(OperandSize::S32, a, b);
        let result = expr.eval(&env(&[("a", 0xFF00_FF00), ("b", 0x0F0F_0F0F)]));
        assert_eq!(result, EvalResult::Bv(0x0F00_0F00));
    }

    #[test]
    fn test_orr_rr_32() {
        let (a, b) = sym32();
        let expr = encode_orr_rr(OperandSize::S32, a, b);
        let result = expr.eval(&env(&[("a", 0xFF00_0000), ("b", 0x00FF_0000)]));
        assert_eq!(result, EvalResult::Bv(0xFFFF_0000));
    }

    #[test]
    fn test_eor_rr_32() {
        let (a, b) = sym32();
        let expr = encode_eor_rr(OperandSize::S32, a, b);
        let result = expr.eval(&env(&[("a", 0xAAAA_AAAA), ("b", 0x5555_5555)]));
        assert_eq!(result, EvalResult::Bv(0xFFFF_FFFF));
    }

    #[test]
    fn test_mvn_32() {
        let a = SmtExpr::var("a", 32);
        let expr = encode_mvn(OperandSize::S32, a);
        let result = expr.eval(&env(&[("a", 0)]));
        assert_eq!(result, EvalResult::Bv(0xFFFF_FFFF));
    }

    #[test]
    fn test_mvn_32_all_ones() {
        let a = SmtExpr::var("a", 32);
        let expr = encode_mvn(OperandSize::S32, a);
        let result = expr.eval(&env(&[("a", 0xFFFF_FFFF)]));
        assert_eq!(result, EvalResult::Bv(0));
    }

    // -----------------------------------------------------------------------
    // Shift instruction semantics tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lsl_rr_32() {
        let (a, b) = sym32();
        let expr = encode_lsl_rr(OperandSize::S32, a, b);
        let result = expr.eval(&env(&[("a", 1), ("b", 4)]));
        assert_eq!(result, EvalResult::Bv(16));
    }

    #[test]
    fn test_lsr_rr_32() {
        let (a, b) = sym32();
        let expr = encode_lsr_rr(OperandSize::S32, a, b);
        let result = expr.eval(&env(&[("a", 0x8000_0000), ("b", 4)]));
        assert_eq!(result, EvalResult::Bv(0x0800_0000));
    }

    #[test]
    fn test_asr_rr_32() {
        let (a, b) = sym32();
        let expr = encode_asr_rr(OperandSize::S32, a, b);
        // Arithmetic shift right of 0x80000000 by 4 = 0xF8000000 (sign-extends)
        let result = expr.eval(&env(&[("a", 0x8000_0000), ("b", 4)]));
        assert_eq!(result, EvalResult::Bv(0xF800_0000));
    }

    #[test]
    fn test_asr_rr_32_positive() {
        let (a, b) = sym32();
        let expr = encode_asr_rr(OperandSize::S32, a, b);
        // Positive value: 0x7FFFFFFF >> 4 = 0x07FFFFFF (zero-fills)
        let result = expr.eval(&env(&[("a", 0x7FFF_FFFF), ("b", 4)]));
        assert_eq!(result, EvalResult::Bv(0x07FF_FFFF));
    }
}
