// llvm2-verify/tmir_semantics.rs - tMIR instruction semantics as SMT formulas
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Encodes tMIR instruction semantics as bitvector SMT expressions.
// Each tMIR instruction maps to a pure function from input bitvectors
// to an output bitvector.
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
}
