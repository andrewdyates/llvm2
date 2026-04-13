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
}
