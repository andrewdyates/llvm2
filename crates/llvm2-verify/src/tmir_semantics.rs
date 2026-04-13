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

use crate::smt::SmtExpr;
use llvm2_lower::instructions::Opcode;
use llvm2_lower::types::Type;

/// Encode a tMIR binary arithmetic operation as an SMT bitvector expression.
///
/// The returned expression computes `lhs <op> rhs` at the given type's
/// bit-width using wrapping (two's complement) semantics.
///
/// # Supported opcodes
///
/// - `Opcode::Iadd` -> `bvadd`
/// - `Opcode::Isub` -> `bvsub`
/// - `Opcode::Imul` -> `bvmul`
/// - `Opcode::Sdiv` -> `bvsdiv`
/// - `Opcode::Udiv` -> `bvudiv`
///
/// # Panics
///
/// Panics if `opcode` is not a binary arithmetic opcode.
pub fn encode_tmir_binop(opcode: &Opcode, _ty: Type, lhs: SmtExpr, rhs: SmtExpr) -> SmtExpr {
    match opcode {
        Opcode::Iadd => lhs.bvadd(rhs),
        Opcode::Isub => lhs.bvsub(rhs),
        Opcode::Imul => lhs.bvmul(rhs),
        Opcode::Sdiv => lhs.bvsdiv(rhs),
        Opcode::Udiv => lhs.bvudiv(rhs),
        _ => panic!("encode_tmir_binop: unsupported opcode {:?}", opcode),
    }
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

/// Return the precondition for a tMIR opcode, if any.
///
/// Division opcodes require `rhs != 0`. Other opcodes have no preconditions.
pub fn precondition(opcode: &Opcode, _ty: Type, _lhs: &SmtExpr, rhs: &SmtExpr) -> Option<SmtExpr> {
    match opcode {
        Opcode::Sdiv | Opcode::Udiv => {
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
    fn test_no_precondition_add() {
        let (a, b) = symbolic_binary_inputs(Type::I32);
        let pre = precondition(&Opcode::Iadd, Type::I32, &a, &b);
        assert!(pre.is_none());
    }
}
