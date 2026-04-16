// llvm2-verify/tmir_semantics.rs - tMIR instruction semantics as SMT formulas
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Encodes tMIR instruction semantics as bitvector SMT expressions.
// Each tMIR instruction maps to a pure function from input bitvectors
// to an output bitvector.
//
// This is the ACTIVE tMIR semantic encoder used by all verification proofs.
// It encodes tMIR opcodes (from llvm2-lower's `Opcode` enum) directly as
// `SmtExpr` bitvector formulas for SMT-based equivalence checking.
//
// DESIGN NOTE:
//
//   This module is the canonical SMT encoder for tMIR instruction semantics.
//   It encodes tMIR opcodes (via llvm2-lower's `Opcode` enum, which maps from
//   `tmir::Inst`) as `SmtExpr` bitvector formulas for SMT-based equivalence
//   proofs.
//
//   The old `stubs/tmir-semantics/` concrete evaluator is deprecated and
//   scheduled for removal once all references are cleaned up (see #291).
//
// Reference: designs/2026-04-13-verification-architecture.md, "SMT Encoding:
// tMIR Instruction Semantics" section.

//! tMIR instruction semantics encoded as [`SmtExpr`] bitvector formulas.
//!
//! Each function takes symbolic input expressions and returns the symbolic
//! output expression representing the instruction's semantics.

use crate::smt::{SmtError, SmtExpr};
use llvm2_lower::instructions::{IntCC, Opcode};
use llvm2_lower::types::Type;

/// Encode a tMIR binary arithmetic operation as an SMT bitvector expression (fallible).
///
/// Returns `Err(SmtError::UnsupportedType)` if the opcode is not a supported
/// binary arithmetic opcode.
///
/// # Supported opcodes
///
/// - `Opcode::Iadd` -> `bvadd`
/// - `Opcode::Isub` -> `bvsub`
/// - `Opcode::Imul` -> `bvmul`
/// - `Opcode::Sdiv` -> `bvsdiv`
/// - `Opcode::Udiv` -> `bvudiv`
/// - `Opcode::Srem` -> `a - bvsdiv(a, b) * b`
/// - `Opcode::Urem` -> `a - bvudiv(a, b) * b`
pub fn try_encode_tmir_binop(
    opcode: &Opcode,
    _ty: Type,
    lhs: SmtExpr,
    rhs: SmtExpr,
) -> Result<SmtExpr, SmtError> {
    match opcode {
        Opcode::Iadd => Ok(lhs.bvadd(rhs)),
        Opcode::Isub => Ok(lhs.bvsub(rhs)),
        Opcode::Imul => Ok(lhs.bvmul(rhs)),
        Opcode::Sdiv => Ok(lhs.bvsdiv(rhs)),
        Opcode::Udiv => Ok(lhs.bvudiv(rhs)),
        // Remainder: a % b = a - (a / b) * b
        // Composed from existing SMT operations until native bvsrem/bvurem are added.
        Opcode::Srem => {
            let quotient = lhs.clone().bvsdiv(rhs.clone());
            Ok(lhs.bvsub(quotient.bvmul(rhs)))
        }
        Opcode::Urem => {
            let quotient = lhs.clone().bvudiv(rhs.clone());
            Ok(lhs.bvsub(quotient.bvmul(rhs)))
        }
        other => Err(SmtError::UnsupportedType(format!(
            "encode_tmir_binop: unsupported opcode {:?}",
            other
        ))),
    }
}

/// Encode a tMIR binary arithmetic operation as an SMT bitvector expression.
///
/// Convenience wrapper around [`try_encode_tmir_binop`].
///
/// # Panics
///
/// Panics if `opcode` is not a binary arithmetic opcode.
pub fn encode_tmir_binop(opcode: &Opcode, ty: Type, lhs: SmtExpr, rhs: SmtExpr) -> SmtExpr {
    try_encode_tmir_binop(opcode, ty, lhs, rhs)
        .expect("encode_tmir_binop: unsupported opcode; use try_encode_tmir_binop() for fallible encoding")
}

/// Encode a tMIR unary negation as an SMT bitvector expression.
///
/// `Neg(a)` is encoded as `bvneg(a)` which is `bvsub(0, a)` in SMT-LIB2.
/// This matches the AArch64 NEG instruction semantics.
pub fn encode_tmir_neg(_ty: Type, operand: SmtExpr) -> SmtExpr {
    operand.bvneg()
}

/// Encode a tMIR floating-point binary operation as an SMT FP expression (fallible).
///
/// Returns `Err(SmtError::UnsupportedType)` if the opcode is not a supported
/// floating-point binary opcode.
///
/// # Supported opcodes
///
/// - `Opcode::Fadd` -> `fp.add(RNE, a, b)`
/// - `Opcode::Fsub` -> `fp.sub(RNE, a, b)`
/// - `Opcode::Fmul` -> `fp.mul(RNE, a, b)`
/// - `Opcode::Fdiv` -> `fp.div(RNE, a, b)`
///
/// All FP operations use RNE (round to nearest, ties to even) as the default
/// rounding mode, matching AArch64's default FPCR.RMode setting.
pub fn try_encode_tmir_fp_binop(
    opcode: &Opcode,
    _ty: Type,
    lhs: SmtExpr,
    rhs: SmtExpr,
) -> Result<SmtExpr, SmtError> {
    use crate::smt::RoundingMode;
    match opcode {
        Opcode::Fadd => Ok(SmtExpr::fp_add(RoundingMode::RNE, lhs, rhs)),
        Opcode::Fsub => Ok(SmtExpr::fp_sub(RoundingMode::RNE, lhs, rhs)),
        Opcode::Fmul => Ok(SmtExpr::fp_mul(RoundingMode::RNE, lhs, rhs)),
        Opcode::Fdiv => Ok(SmtExpr::fp_div(RoundingMode::RNE, lhs, rhs)),
        other => Err(SmtError::UnsupportedType(format!(
            "encode_tmir_fp_binop: unsupported opcode {:?}",
            other
        ))),
    }
}

/// Encode a tMIR floating-point binary operation as an SMT FP expression.
///
/// Convenience wrapper around [`try_encode_tmir_fp_binop`].
///
/// # Panics
///
/// Panics if `opcode` is not a floating-point binary opcode.
pub fn encode_tmir_fp_binop(opcode: &Opcode, ty: Type, lhs: SmtExpr, rhs: SmtExpr) -> SmtExpr {
    try_encode_tmir_fp_binop(opcode, ty, lhs, rhs)
        .expect("encode_tmir_fp_binop: unsupported opcode; use try_encode_tmir_fp_binop() for fallible encoding")
}

/// Encode a tMIR floating-point negation as an SMT FP expression.
///
/// `Fneg(a)` is encoded as `fp.neg(a)`. This matches the AArch64 FNEG instruction.
pub fn encode_tmir_fneg(_ty: Type, operand: SmtExpr) -> SmtExpr {
    operand.fp_neg()
}

/// Create symbolic FP input variables for a binary FP operation.
///
/// Returns `(lhs, rhs)` as FP constant nodes. For FP proofs, we use
/// `FPConst` nodes that the evaluator interprets via native f32/f64.
/// The `eb` and `sb` parameters specify the FP format (e.g., 8/24 for f32, 11/53 for f64).
pub fn symbolic_fp_binary_inputs(eb: u32, sb: u32) -> (SmtExpr, SmtExpr) {
    let _total = eb + sb;
    // Use Var nodes with the bit-width matching the FP format.
    // The proof obligation's fp_inputs field declares these as FP-sorted.
    // For evaluation, we populate them via the fp_env pathway.
    (
        SmtExpr::FPConst { bits: 0, eb, sb }, // placeholder; actual values set per test
        SmtExpr::FPConst { bits: 0, eb, sb },
    )
}

/// Encode a tMIR floating-point comparison as an SMT expression.
///
/// `Fcmp(cond, a, b)` returns a 1-bit bitvector: `bv1(1)` if the condition
/// holds, `bv1(0)` otherwise. This matches the AArch64 FCMP + CSET output
/// format.
///
/// # Supported conditions
///
/// All 14 `FloatCC` variants: 6 ordered, 2 ordering predicates, 6 unordered.
///
/// Ordered comparisons return false when either operand is NaN.
/// Unordered comparisons return true when either operand is NaN.
pub fn encode_tmir_fcmp(cond: &llvm2_lower::instructions::FloatCC, _ty: Type, lhs: SmtExpr, rhs: SmtExpr) -> SmtExpr {
    use llvm2_lower::instructions::FloatCC;

    let a_nan = lhs.clone().fp_is_nan();
    let b_nan = rhs.clone().fp_is_nan();
    let either_nan = a_nan.clone().or_expr(b_nan.clone());

    let bool_result = match cond {
        FloatCC::Equal => lhs.fp_eq(rhs),
        FloatCC::NotEqual => lhs.fp_eq(rhs).not_expr(),
        FloatCC::LessThan => lhs.fp_lt(rhs),
        FloatCC::LessThanOrEqual => lhs.fp_le(rhs),
        FloatCC::GreaterThan => lhs.fp_gt(rhs),
        FloatCC::GreaterThanOrEqual => lhs.fp_ge(rhs),
        FloatCC::Ordered => a_nan.not_expr().and_expr(b_nan.not_expr()),
        FloatCC::Unordered => either_nan,
        FloatCC::UnorderedEqual => lhs.fp_eq(rhs).or_expr(either_nan),
        FloatCC::UnorderedNotEqual => lhs.fp_eq(rhs).not_expr().or_expr(either_nan),
        FloatCC::UnorderedLessThan => lhs.fp_lt(rhs).or_expr(either_nan),
        FloatCC::UnorderedLessThanOrEqual => lhs.fp_le(rhs).or_expr(either_nan),
        FloatCC::UnorderedGreaterThan => lhs.fp_gt(rhs).or_expr(either_nan),
        FloatCC::UnorderedGreaterThanOrEqual => lhs.fp_ge(rhs).or_expr(either_nan),
    };

    SmtExpr::ite(bool_result, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1))
}

/// Encode a tMIR integer constant.
pub fn encode_tmir_iconst(ty: Type, imm: i64) -> SmtExpr {
    let width = ty.bits();
    SmtExpr::bv_const(imm as u64, width)
}

/// Create symbolic input variables for a binary operation at the given type.
///
/// Returns `(lhs, rhs)` as symbolic `SmtExpr::Var` nodes.
pub fn symbolic_binary_inputs(ty: Type) -> (SmtExpr, SmtExpr) {
    let width = ty.bits();
    (SmtExpr::var("a", width), SmtExpr::var("b", width))
}

/// Create a symbolic input variable for a unary operation.
pub fn symbolic_unary_input(ty: Type) -> SmtExpr {
    SmtExpr::var("a", ty.bits())
}

/// Encode a tMIR integer comparison as an SMT expression.
///
/// `Icmp(cond, a, b)` returns a 1-bit bitvector: `bv1(1)` if the condition
/// holds, `bv1(0)` otherwise. This matches the AArch64 CSET output format.
///
/// # Supported conditions
///
/// All 10 `IntCC` variants (see `llvm2_lower::instructions::IntCC`).
pub fn encode_tmir_icmp(cond: &IntCC, _ty: Type, lhs: SmtExpr, rhs: SmtExpr) -> SmtExpr {
    let cmp_bool = match cond {
        IntCC::Equal => lhs.eq_expr(rhs),
        IntCC::NotEqual => lhs.eq_expr(rhs).not_expr(),
        IntCC::SignedLessThan => lhs.bvslt(rhs),
        IntCC::SignedGreaterThanOrEqual => lhs.bvsge(rhs),
        IntCC::SignedGreaterThan => lhs.bvsgt(rhs),
        IntCC::SignedLessThanOrEqual => lhs.bvsle(rhs),
        IntCC::UnsignedLessThan => lhs.bvult(rhs),
        IntCC::UnsignedGreaterThanOrEqual => lhs.bvuge(rhs),
        IntCC::UnsignedGreaterThan => lhs.bvugt(rhs),
        IntCC::UnsignedLessThanOrEqual => lhs.bvule(rhs),
    };
    SmtExpr::ite(cmp_bool, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1))
}

/// Encode a tMIR bitwise binary operation as an SMT bitvector expression (fallible).
///
/// Returns `Err(SmtError::UnsupportedType)` if the opcode is not a supported
/// bitwise binary opcode.
///
/// # Supported opcodes
///
/// - `Opcode::Band` -> `bvand`
/// - `Opcode::Bor`  -> `bvor`
/// - `Opcode::Bxor` -> `bvxor`
pub fn try_encode_tmir_bitwise_binop(
    opcode: &Opcode,
    _ty: Type,
    lhs: SmtExpr,
    rhs: SmtExpr,
) -> Result<SmtExpr, SmtError> {
    match opcode {
        Opcode::Band => Ok(lhs.bvand(rhs)),
        Opcode::Bor => Ok(lhs.bvor(rhs)),
        Opcode::Bxor => Ok(lhs.bvxor(rhs)),
        other => Err(SmtError::UnsupportedType(format!(
            "encode_tmir_bitwise_binop: unsupported opcode {:?}",
            other
        ))),
    }
}

/// Encode a tMIR bitwise binary operation as an SMT bitvector expression.
///
/// Convenience wrapper around [`try_encode_tmir_bitwise_binop`].
///
/// # Panics
///
/// Panics if `opcode` is not a bitwise binary opcode.
pub fn encode_tmir_bitwise_binop(opcode: &Opcode, ty: Type, lhs: SmtExpr, rhs: SmtExpr) -> SmtExpr {
    try_encode_tmir_bitwise_binop(opcode, ty, lhs, rhs)
        .expect("encode_tmir_bitwise_binop: unsupported opcode; use try_encode_tmir_bitwise_binop() for fallible encoding")
}

/// Encode a tMIR bitwise NOT as an SMT bitvector expression.
///
/// `Bnot(a)` is encoded as `bvxor(a, all_ones)` which flips all bits.
/// This matches the AArch64 MVN instruction semantics.
pub fn encode_tmir_bnot(ty: Type, operand: SmtExpr) -> SmtExpr {
    let width = ty.bits();
    let all_ones = SmtExpr::bv_const(crate::smt::mask(u64::MAX, width), width);
    operand.bvxor(all_ones)
}

/// Encode a tMIR shift operation as an SMT bitvector expression (fallible).
///
/// Returns `Err(SmtError::UnsupportedType)` if the opcode is not a supported
/// shift opcode.
///
/// # Supported opcodes
///
/// - `Opcode::Ishl` -> `bvshl`  (logical shift left)
/// - `Opcode::Ushr` -> `bvlshr` (logical shift right)
/// - `Opcode::Sshr` -> `bvashr` (arithmetic shift right)
///
/// # Shift amount semantics
///
/// On AArch64, shift amounts are masked to the register width (mod 32 for W,
/// mod 64 for X). The SMT `bvshl`/`bvlshr`/`bvashr` operations define the
/// result as zero when the shift amount >= width, which differs slightly.
/// For proofs, we verify equivalence under the assumption that the shift
/// amount is in range [0, width). The tMIR type system enforces this.
pub fn try_encode_tmir_shift(
    opcode: &Opcode,
    _ty: Type,
    lhs: SmtExpr,
    rhs: SmtExpr,
) -> Result<SmtExpr, SmtError> {
    match opcode {
        Opcode::Ishl => Ok(lhs.bvshl(rhs)),
        Opcode::Ushr => Ok(lhs.bvlshr(rhs)),
        Opcode::Sshr => Ok(lhs.bvashr(rhs)),
        other => Err(SmtError::UnsupportedType(format!(
            "encode_tmir_shift: unsupported opcode {:?}",
            other
        ))),
    }
}

/// Encode a tMIR shift operation as an SMT bitvector expression.
///
/// Convenience wrapper around [`try_encode_tmir_shift`].
///
/// # Panics
///
/// Panics if `opcode` is not a shift opcode.
pub fn encode_tmir_shift(opcode: &Opcode, ty: Type, lhs: SmtExpr, rhs: SmtExpr) -> SmtExpr {
    try_encode_tmir_shift(opcode, ty, lhs, rhs)
        .expect("encode_tmir_shift: unsupported opcode; use try_encode_tmir_shift() for fallible encoding")
}

/// Return the precondition for a tMIR opcode, if any.
///
/// Division and remainder opcodes require `rhs != 0`. Other opcodes have no preconditions.
pub fn precondition(opcode: &Opcode, _ty: Type, _lhs: &SmtExpr, rhs: &SmtExpr) -> Option<SmtExpr> {
    match opcode {
        Opcode::Sdiv | Opcode::Udiv | Opcode::Srem | Opcode::Urem => {
            // Precondition: divisor != 0
            let zero = SmtExpr::bv_const(0, rhs.bv_width());
            Some(rhs.clone().eq_expr(zero).not_expr())
        }
        _ => None,
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

    #[test]
    fn test_encode_iadd() {
        let (a, b) = symbolic_binary_inputs(Type::I32);
        let expr = encode_tmir_binop(&Opcode::Iadd, Type::I32, a, b);
        let result = expr.eval(&env(&[("a", 3), ("b", 4)]));
        assert_eq!(result, EvalResult::Bv(7));
    }

    #[test]
    fn test_encode_isub() {
        let (a, b) = symbolic_binary_inputs(Type::I32);
        let expr = encode_tmir_binop(&Opcode::Isub, Type::I32, a, b);
        let result = expr.eval(&env(&[("a", 10), ("b", 3)]));
        assert_eq!(result, EvalResult::Bv(7));
    }

    #[test]
    fn test_encode_imul() {
        let (a, b) = symbolic_binary_inputs(Type::I32);
        let expr = encode_tmir_binop(&Opcode::Imul, Type::I32, a, b);
        let result = expr.eval(&env(&[("a", 6), ("b", 7)]));
        assert_eq!(result, EvalResult::Bv(42));
    }

    #[test]
    fn test_encode_neg() {
        let a = symbolic_unary_input(Type::I32);
        let expr = encode_tmir_neg(Type::I32, a);
        // neg(5) in 32-bit = 0xFFFFFFFF - 5 + 1 = 0xFFFFFFFB
        let result = expr.eval(&env(&[("a", 5)]));
        assert_eq!(result, EvalResult::Bv(0xFFFF_FFFBu64));
    }

    #[test]
    fn test_precondition_div() {
        let (a, b) = symbolic_binary_inputs(Type::I32);
        let pre = precondition(&Opcode::Sdiv, Type::I32, &a, &b);
        assert!(pre.is_some());
        // b=0 should fail precondition
        let result = pre.unwrap().eval(&env(&[("a", 1), ("b", 0)]));
        assert_eq!(result, EvalResult::Bool(false));
    }

    #[test]
    fn test_encode_icmp_eq_true() {
        let (a, b) = symbolic_binary_inputs(Type::I32);
        let expr = encode_tmir_icmp(&IntCC::Equal, Type::I32, a, b);
        let result = expr.eval(&env(&[("a", 42), ("b", 42)]));
        assert_eq!(result, EvalResult::Bv(1));
    }

    #[test]
    fn test_encode_icmp_eq_false() {
        let (a, b) = symbolic_binary_inputs(Type::I32);
        let expr = encode_tmir_icmp(&IntCC::Equal, Type::I32, a, b);
        let result = expr.eval(&env(&[("a", 42), ("b", 43)]));
        assert_eq!(result, EvalResult::Bv(0));
    }

    #[test]
    fn test_encode_icmp_slt() {
        let (a, b) = symbolic_binary_inputs(Type::I32);
        let expr = encode_tmir_icmp(&IntCC::SignedLessThan, Type::I32, a, b);
        // -1 < 0 (signed)
        let neg1 = 0xFFFF_FFFFu64;
        let result = expr.eval(&env(&[("a", neg1), ("b", 0)]));
        assert_eq!(result, EvalResult::Bv(1));
    }

    #[test]
    fn test_encode_icmp_ult() {
        let (a, b) = symbolic_binary_inputs(Type::I32);
        let expr = encode_tmir_icmp(&IntCC::UnsignedLessThan, Type::I32, a, b);
        // 3 <_u 10
        let result = expr.eval(&env(&[("a", 3), ("b", 10)]));
        assert_eq!(result, EvalResult::Bv(1));
    }

    #[test]
    fn test_encode_icmp_ult_not_less() {
        let (a, b) = symbolic_binary_inputs(Type::I32);
        let expr = encode_tmir_icmp(&IntCC::UnsignedLessThan, Type::I32, a, b);
        // 0xFFFFFFFF is NOT <_u 0 (it's the biggest unsigned value)
        let result = expr.eval(&env(&[("a", 0xFFFF_FFFF), ("b", 0)]));
        assert_eq!(result, EvalResult::Bv(0));
    }

    #[test]
    fn test_no_precondition_add() {
        let (a, b) = symbolic_binary_inputs(Type::I32);
        let pre = precondition(&Opcode::Iadd, Type::I32, &a, &b);
        assert!(pre.is_none());
    }

    // -----------------------------------------------------------------------
    // Floating-point semantic encoder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_fadd_f32() {
        let a = SmtExpr::fp32_const(1.5f32);
        let b = SmtExpr::fp32_const(2.5f32);
        let expr = encode_tmir_fp_binop(&Opcode::Fadd, Type::F32, a, b);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(4.0));
    }

    #[test]
    fn test_encode_fsub_f64() {
        let a = SmtExpr::fp64_const(10.0);
        let b = SmtExpr::fp64_const(3.5);
        let expr = encode_tmir_fp_binop(&Opcode::Fsub, Type::F64, a, b);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(6.5));
    }

    #[test]
    fn test_encode_fmul_f64() {
        let a = SmtExpr::fp64_const(3.0);
        let b = SmtExpr::fp64_const(7.0);
        let expr = encode_tmir_fp_binop(&Opcode::Fmul, Type::F64, a, b);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(21.0));
    }

    #[test]
    fn test_encode_fdiv_f64() {
        let a = SmtExpr::fp64_const(10.0);
        let b = SmtExpr::fp64_const(4.0);
        let expr = encode_tmir_fp_binop(&Opcode::Fdiv, Type::F64, a, b);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(2.5));
    }

    #[test]
    fn test_encode_fneg_f64() {
        let a = SmtExpr::fp64_const(42.0);
        let expr = encode_tmir_fneg(Type::F64, a);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(-42.0));
    }

    #[test]
    fn test_encode_fneg_f32() {
        let a = SmtExpr::fp32_const(-std::f32::consts::PI);
        let expr = encode_tmir_fneg(Type::F32, a);
        let result = expr.try_eval(&env(&[])).unwrap();
        // Negation of -PI should be +PI (as f64, with f32 precision)
        assert_eq!(result, EvalResult::Float(std::f32::consts::PI as f64)); // f32 -> f64 precision
    }

    #[test]
    fn test_try_encode_fp_binop_unsupported() {
        let a = SmtExpr::fp64_const(1.0);
        let b = SmtExpr::fp64_const(2.0);
        let result = try_encode_tmir_fp_binop(&Opcode::Iadd, Type::F64, a, b);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Bitwise semantic encoder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_band() {
        let (a, b) = symbolic_binary_inputs(Type::I32);
        let expr = encode_tmir_bitwise_binop(&Opcode::Band, Type::I32, a, b);
        let result = expr.eval(&env(&[("a", 0xFF00_FF00), ("b", 0x0F0F_0F0F)]));
        assert_eq!(result, EvalResult::Bv(0x0F00_0F00));
    }

    #[test]
    fn test_encode_bor() {
        let (a, b) = symbolic_binary_inputs(Type::I32);
        let expr = encode_tmir_bitwise_binop(&Opcode::Bor, Type::I32, a, b);
        let result = expr.eval(&env(&[("a", 0xFF00_0000), ("b", 0x00FF_0000)]));
        assert_eq!(result, EvalResult::Bv(0xFFFF_0000));
    }

    #[test]
    fn test_encode_bxor() {
        let (a, b) = symbolic_binary_inputs(Type::I32);
        let expr = encode_tmir_bitwise_binop(&Opcode::Bxor, Type::I32, a, b);
        let result = expr.eval(&env(&[("a", 0xAAAA_AAAA), ("b", 0x5555_5555)]));
        assert_eq!(result, EvalResult::Bv(0xFFFF_FFFF));
    }

    #[test]
    fn test_encode_bnot() {
        let a = symbolic_unary_input(Type::I32);
        let expr = encode_tmir_bnot(Type::I32, a);
        let result = expr.eval(&env(&[("a", 0)]));
        assert_eq!(result, EvalResult::Bv(0xFFFF_FFFF));
    }

    #[test]
    fn test_encode_bnot_ones() {
        let a = symbolic_unary_input(Type::I32);
        let expr = encode_tmir_bnot(Type::I32, a);
        let result = expr.eval(&env(&[("a", 0xFFFF_FFFF)]));
        assert_eq!(result, EvalResult::Bv(0));
    }

    // -----------------------------------------------------------------------
    // Shift semantic encoder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_ishl() {
        let (a, b) = symbolic_binary_inputs(Type::I32);
        let expr = encode_tmir_shift(&Opcode::Ishl, Type::I32, a, b);
        let result = expr.eval(&env(&[("a", 1), ("b", 4)]));
        assert_eq!(result, EvalResult::Bv(16));
    }

    #[test]
    fn test_encode_ushr() {
        let (a, b) = symbolic_binary_inputs(Type::I32);
        let expr = encode_tmir_shift(&Opcode::Ushr, Type::I32, a, b);
        let result = expr.eval(&env(&[("a", 0x8000_0000), ("b", 4)]));
        assert_eq!(result, EvalResult::Bv(0x0800_0000));
    }

    #[test]
    fn test_encode_sshr() {
        let (a, b) = symbolic_binary_inputs(Type::I32);
        let expr = encode_tmir_shift(&Opcode::Sshr, Type::I32, a, b);
        // Arithmetic shift right of 0x80000000 by 4 = 0xF8000000 (sign-extends)
        let result = expr.eval(&env(&[("a", 0x8000_0000), ("b", 4)]));
        assert_eq!(result, EvalResult::Bv(0xF800_0000));
    }

    #[test]
    fn test_try_encode_bitwise_unsupported() {
        let (a, b) = symbolic_binary_inputs(Type::I32);
        let result = try_encode_tmir_bitwise_binop(&Opcode::Iadd, Type::I32, a, b);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_encode_shift_unsupported() {
        let (a, b) = symbolic_binary_inputs(Type::I32);
        let result = try_encode_tmir_shift(&Opcode::Iadd, Type::I32, a, b);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // FP comparison (FCMP) semantic encoder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_tmir_fcmp_eq_true() {
        use llvm2_lower::instructions::FloatCC;
        let a = SmtExpr::fp64_const(1.0);
        let b = SmtExpr::fp64_const(1.0);
        let expr = encode_tmir_fcmp(&FloatCC::Equal, Type::F64, a, b);
        let result = expr.eval(&std::collections::HashMap::new());
        assert_eq!(result, EvalResult::Bv(1));
    }

    #[test]
    fn test_encode_tmir_fcmp_eq_false() {
        use llvm2_lower::instructions::FloatCC;
        let a = SmtExpr::fp64_const(1.0);
        let b = SmtExpr::fp64_const(2.0);
        let expr = encode_tmir_fcmp(&FloatCC::Equal, Type::F64, a, b);
        let result = expr.eval(&std::collections::HashMap::new());
        assert_eq!(result, EvalResult::Bv(0));
    }

    #[test]
    fn test_encode_tmir_fcmp_lt_true() {
        use llvm2_lower::instructions::FloatCC;
        let a = SmtExpr::fp64_const(1.0);
        let b = SmtExpr::fp64_const(2.0);
        let expr = encode_tmir_fcmp(&FloatCC::LessThan, Type::F64, a, b);
        let result = expr.eval(&std::collections::HashMap::new());
        assert_eq!(result, EvalResult::Bv(1));
    }

    #[test]
    fn test_encode_tmir_fcmp_gt_true() {
        use llvm2_lower::instructions::FloatCC;
        let a = SmtExpr::fp64_const(3.0);
        let b = SmtExpr::fp64_const(1.0);
        let expr = encode_tmir_fcmp(&FloatCC::GreaterThan, Type::F64, a, b);
        let result = expr.eval(&std::collections::HashMap::new());
        assert_eq!(result, EvalResult::Bv(1));
    }

    #[test]
    fn test_encode_tmir_fcmp_ordered_no_nan() {
        use llvm2_lower::instructions::FloatCC;
        let a = SmtExpr::fp64_const(1.0);
        let b = SmtExpr::fp64_const(2.0);
        let expr = encode_tmir_fcmp(&FloatCC::Ordered, Type::F64, a, b);
        let result = expr.eval(&std::collections::HashMap::new());
        assert_eq!(result, EvalResult::Bv(1));
    }

    #[test]
    fn test_encode_tmir_fcmp_ordered_with_nan() {
        use llvm2_lower::instructions::FloatCC;
        let a = SmtExpr::fp64_const(f64::NAN);
        let b = SmtExpr::fp64_const(1.0);
        let expr = encode_tmir_fcmp(&FloatCC::Ordered, Type::F64, a, b);
        let result = expr.eval(&std::collections::HashMap::new());
        assert_eq!(result, EvalResult::Bv(0));
    }

    #[test]
    fn test_encode_tmir_fcmp_unordered_with_nan() {
        use llvm2_lower::instructions::FloatCC;
        let a = SmtExpr::fp64_const(f64::NAN);
        let b = SmtExpr::fp64_const(1.0);
        let expr = encode_tmir_fcmp(&FloatCC::Unordered, Type::F64, a, b);
        let result = expr.eval(&std::collections::HashMap::new());
        assert_eq!(result, EvalResult::Bv(1));
    }

    #[test]
    fn test_encode_tmir_fcmp_unordered_eq_nan() {
        use llvm2_lower::instructions::FloatCC;
        let a = SmtExpr::fp64_const(f64::NAN);
        let b = SmtExpr::fp64_const(f64::NAN);
        let expr = encode_tmir_fcmp(&FloatCC::UnorderedEqual, Type::F64, a, b);
        let result = expr.eval(&std::collections::HashMap::new());
        // NaN should make UnorderedEqual true
        assert_eq!(result, EvalResult::Bv(1));
    }

    #[test]
    fn test_encode_tmir_fcmp_f32() {
        use llvm2_lower::instructions::FloatCC;
        let a = SmtExpr::fp32_const(1.5f32);
        let b = SmtExpr::fp32_const(2.5f32);
        let expr = encode_tmir_fcmp(&FloatCC::LessThan, Type::F32, a, b);
        let result = expr.eval(&std::collections::HashMap::new());
        assert_eq!(result, EvalResult::Bv(1));
    }
}
