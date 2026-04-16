// llvm2-verify/x86_64_semantics.rs - x86-64 instruction semantics as SMT formulas
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Encodes x86-64 instruction semantics as bitvector SMT expressions.
// Each instruction maps to a pure function from input bitvectors to output
// bitvectors, modeling the instruction's effect on destination registers.
//
// Key difference from AArch64: x86-64 is a two-operand ISA where the
// destination is typically the same as the first source (dst = dst op src).
// However, for verification purposes, we model the semantics as pure
// functions from inputs to outputs, abstracting away the destructive update.
//
// Reference: Intel 64 and IA-32 Architectures Software Developer's Manual
// Reference: designs/2026-04-13-verification-architecture.md

//! x86-64 instruction semantics encoded as [`SmtExpr`] bitvector formulas.
//!
//! Key difference from AArch64: x86-64 division uses implicit RDX:RAX
//! register pair. For verification, we model the quotient result (RAX)
//! as a simple division operation since the tMIR division semantic is
//! also a simple division. The RDX:RAX widening is an ABI detail that
//! doesn't affect the semantic equivalence of the quotient.

use crate::smt::SmtExpr;

/// Operand size for x86-64 instructions.
///
/// Maps to the REX.W prefix and operand-size override behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum X86OperandSize {
    /// 32-bit operand (no REX.W prefix, default in 64-bit mode for most insns)
    S32,
    /// 64-bit operand (REX.W prefix)
    S64,
}

/// Width in bits for an X86OperandSize.
pub fn x86_operand_size_bits(size: X86OperandSize) -> u32 {
    match size {
        X86OperandSize::S32 => 32,
        X86OperandSize::S64 => 64,
    }
}

// ---------------------------------------------------------------------------
// Integer arithmetic instruction semantics
// ---------------------------------------------------------------------------

/// Encode `ADD r/m64, r64` or `ADD r/m32, r32` -- register-register add.
///
/// Semantics: `dst = src1 + src2` (wrapping).
/// Reference: Intel SDM Vol 2A, ADD instruction.
pub fn encode_add_rr(_size: X86OperandSize, src1: SmtExpr, src2: SmtExpr) -> SmtExpr {
    src1.bvadd(src2)
}

/// Encode `SUB r/m64, r64` or `SUB r/m32, r32` -- register-register subtract.
///
/// Semantics: `dst = src1 - src2` (wrapping).
/// Reference: Intel SDM Vol 2B, SUB instruction.
pub fn encode_sub_rr(_size: X86OperandSize, src1: SmtExpr, src2: SmtExpr) -> SmtExpr {
    src1.bvsub(src2)
}

/// Encode `IMUL r64, r/m64` -- two-operand signed multiply.
///
/// Semantics: `dst = src1 * src2` (wrapping, lower bits).
/// The two-operand IMUL form stores the lower half of the product in dst,
/// which is equivalent to wrapping multiplication.
/// Reference: Intel SDM Vol 2A, IMUL instruction.
pub fn encode_imul_rr(_size: X86OperandSize, src1: SmtExpr, src2: SmtExpr) -> SmtExpr {
    src1.bvmul(src2)
}

/// Encode `IDIV r/m64` -- signed divide (quotient).
///
/// On x86-64, IDIV divides RDX:RAX by the source operand. The quotient
/// goes to RAX and the remainder to RDX. For verification of tMIR Sdiv,
/// we model only the quotient (RAX result).
///
/// Semantics: `RAX = (RDX:RAX) /s src` (truncation toward zero).
/// Since tMIR Sdiv operates on single-width values (not double-width),
/// and the ISel zeros RDX (via CDQ/CQO sign-extension), the effective
/// semantic is: `dst = src1 /s src2`.
///
/// Precondition: `src2 != 0` (division by zero raises #DE).
/// Reference: Intel SDM Vol 2A, IDIV instruction.
pub fn encode_idiv_quotient(_size: X86OperandSize, src1: SmtExpr, src2: SmtExpr) -> SmtExpr {
    src1.bvsdiv(src2)
}

/// Encode `DIV r/m64` -- unsigned divide (quotient).
///
/// Similar to IDIV but unsigned. The quotient goes to RAX.
///
/// Semantics: `dst = src1 /u src2`.
/// Precondition: `src2 != 0` (division by zero raises #DE).
/// Reference: Intel SDM Vol 2A, DIV instruction.
pub fn encode_div_quotient(_size: X86OperandSize, src1: SmtExpr, src2: SmtExpr) -> SmtExpr {
    src1.bvudiv(src2)
}

/// Encode `NEG r/m64` -- two's complement negate.
///
/// Semantics: `dst = 0 - src` (two's complement negation).
/// Reference: Intel SDM Vol 2B, NEG instruction.
pub fn encode_neg(_size: X86OperandSize, src: SmtExpr) -> SmtExpr {
    src.bvneg()
}

// ---------------------------------------------------------------------------
// Bitwise instruction semantics
// ---------------------------------------------------------------------------

/// Encode `AND r/m64, r64` -- bitwise AND.
///
/// Semantics: `dst = src1 & src2`.
/// Reference: Intel SDM Vol 2A, AND instruction.
pub fn encode_and_rr(_size: X86OperandSize, src1: SmtExpr, src2: SmtExpr) -> SmtExpr {
    src1.bvand(src2)
}

/// Encode `OR r/m64, r64` -- bitwise OR.
///
/// Semantics: `dst = src1 | src2`.
/// Reference: Intel SDM Vol 2B, OR instruction.
pub fn encode_or_rr(_size: X86OperandSize, src1: SmtExpr, src2: SmtExpr) -> SmtExpr {
    src1.bvor(src2)
}

/// Encode `XOR r/m64, r64` -- bitwise XOR.
///
/// Semantics: `dst = src1 ^ src2`.
/// Reference: Intel SDM Vol 2B, XOR instruction.
pub fn encode_xor_rr(_size: X86OperandSize, src1: SmtExpr, src2: SmtExpr) -> SmtExpr {
    src1.bvxor(src2)
}

/// Encode `NOT r/m64` -- bitwise complement.
///
/// Semantics: `dst = ~src` (one's complement).
/// Reference: Intel SDM Vol 2B, NOT instruction.
pub fn encode_not(_size: X86OperandSize, src: SmtExpr) -> SmtExpr {
    let width = src.bv_width();
    let all_ones = SmtExpr::bv_const(crate::smt::mask(u64::MAX, width), width);
    src.bvxor(all_ones)
}

// ---------------------------------------------------------------------------
// Shift instruction semantics
// ---------------------------------------------------------------------------

/// Encode `SHL r/m64, CL` -- logical shift left by register.
///
/// Semantics: `dst = src1 << (src2 mod width)`.
/// On x86-64, the shift amount is masked to 5 bits (mod 32) for 32-bit
/// operations and 6 bits (mod 64) for 64-bit operations.
/// Reference: Intel SDM Vol 2B, SHL instruction.
pub fn encode_shl_rr(_size: X86OperandSize, src1: SmtExpr, src2: SmtExpr) -> SmtExpr {
    src1.bvshl(src2)
}

/// Encode `SHR r/m64, CL` -- logical shift right by register.
///
/// Semantics: `dst = src1 >> (src2 mod width)` (zero-fill).
/// Reference: Intel SDM Vol 2B, SHR instruction.
pub fn encode_shr_rr(_size: X86OperandSize, src1: SmtExpr, src2: SmtExpr) -> SmtExpr {
    src1.bvlshr(src2)
}

/// Encode `SAR r/m64, CL` -- arithmetic shift right by register.
///
/// Semantics: `dst = src1 >>_s (src2 mod width)` (sign-extending).
/// Reference: Intel SDM Vol 2B, SAR instruction.
pub fn encode_sar_rr(_size: X86OperandSize, src1: SmtExpr, src2: SmtExpr) -> SmtExpr {
    src1.bvashr(src2)
}

// ---------------------------------------------------------------------------
// Floating-point instruction semantics (SSE)
// ---------------------------------------------------------------------------

/// Floating-point precision selector for x86-64 SSE instructions.
///
/// Maps to SSE scalar single (SS) and scalar double (SD) instruction variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum X86FPSize {
    /// Single-precision (32-bit, XMM lower 32 bits). IEEE 754 binary32.
    /// Uses ADDSS, SUBSS, MULSS, DIVSS instructions.
    Single,
    /// Double-precision (64-bit, XMM lower 64 bits). IEEE 754 binary64.
    /// Uses ADDSD, SUBSD, MULSD, DIVSD instructions.
    Double,
}

impl X86FPSize {
    /// Exponent bits for this FP size.
    pub fn eb(self) -> u32 {
        match self {
            X86FPSize::Single => 8,
            X86FPSize::Double => 11,
        }
    }

    /// Significand bits (including implicit bit) for this FP size.
    pub fn sb(self) -> u32 {
        match self {
            X86FPSize::Single => 24,
            X86FPSize::Double => 53,
        }
    }
}

/// Encode `ADDSD xmm, xmm` or `ADDSS xmm, xmm` -- scalar FP add.
///
/// Semantics: `dst = src1 + src2` using RNE rounding mode (default MXCSR).
/// Reference: Intel SDM Vol 2A, ADDSD/ADDSS instructions.
pub fn encode_fp_add_rr(_size: X86FPSize, src1: SmtExpr, src2: SmtExpr) -> SmtExpr {
    use crate::smt::RoundingMode;
    SmtExpr::fp_add(RoundingMode::RNE, src1, src2)
}

/// Encode `SUBSD xmm, xmm` or `SUBSS xmm, xmm` -- scalar FP subtract.
///
/// Semantics: `dst = src1 - src2` using RNE rounding mode.
/// Reference: Intel SDM Vol 2B, SUBSD/SUBSS instructions.
pub fn encode_fp_sub_rr(_size: X86FPSize, src1: SmtExpr, src2: SmtExpr) -> SmtExpr {
    use crate::smt::RoundingMode;
    SmtExpr::fp_sub(RoundingMode::RNE, src1, src2)
}

/// Encode `MULSD xmm, xmm` or `MULSS xmm, xmm` -- scalar FP multiply.
///
/// Semantics: `dst = src1 * src2` using RNE rounding mode.
/// Reference: Intel SDM Vol 2B, MULSD/MULSS instructions.
pub fn encode_fp_mul_rr(_size: X86FPSize, src1: SmtExpr, src2: SmtExpr) -> SmtExpr {
    use crate::smt::RoundingMode;
    SmtExpr::fp_mul(RoundingMode::RNE, src1, src2)
}

/// Encode `DIVSD xmm, xmm` or `DIVSS xmm, xmm` -- scalar FP divide.
///
/// Semantics: `dst = src1 / src2` using RNE rounding mode.
/// Reference: Intel SDM Vol 2B, DIVSD/DIVSS instructions.
pub fn encode_fp_div_rr(_size: X86FPSize, src1: SmtExpr, src2: SmtExpr) -> SmtExpr {
    use crate::smt::RoundingMode;
    SmtExpr::fp_div(RoundingMode::RNE, src1, src2)
}

// ===========================================================================
// Tests
// ===========================================================================

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

    fn sym64() -> (SmtExpr, SmtExpr) {
        (SmtExpr::var("a", 64), SmtExpr::var("b", 64))
    }

    // -----------------------------------------------------------------------
    // Integer arithmetic tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_rr_32() {
        let (a, b) = sym32();
        let expr = encode_add_rr(X86OperandSize::S32, a, b);
        let result = expr.eval(&env(&[("a", 100), ("b", 200)]));
        assert_eq!(result, EvalResult::Bv(300));
    }

    #[test]
    fn test_add_rr_64() {
        let (a, b) = sym64();
        let expr = encode_add_rr(X86OperandSize::S64, a, b);
        let result = expr.eval(&env(&[("a", 0x1_0000_0000), ("b", 0x2_0000_0000)]));
        assert_eq!(result, EvalResult::Bv(0x3_0000_0000));
    }

    #[test]
    fn test_sub_rr_32() {
        let (a, b) = sym32();
        let expr = encode_sub_rr(X86OperandSize::S32, a, b);
        let result = expr.eval(&env(&[("a", 10), ("b", 3)]));
        assert_eq!(result, EvalResult::Bv(7));
    }

    #[test]
    fn test_imul_rr_32() {
        let (a, b) = sym32();
        let expr = encode_imul_rr(X86OperandSize::S32, a, b);
        let result = expr.eval(&env(&[("a", 6), ("b", 7)]));
        assert_eq!(result, EvalResult::Bv(42));
    }

    #[test]
    fn test_neg_32() {
        let a = SmtExpr::var("a", 32);
        let expr = encode_neg(X86OperandSize::S32, a);
        let result = expr.eval(&env(&[("a", 1)]));
        assert_eq!(result, EvalResult::Bv(0xFFFF_FFFF));
    }

    #[test]
    fn test_neg_zero() {
        let a = SmtExpr::var("a", 32);
        let expr = encode_neg(X86OperandSize::S32, a);
        let result = expr.eval(&env(&[("a", 0)]));
        assert_eq!(result, EvalResult::Bv(0));
    }

    // -----------------------------------------------------------------------
    // Bitwise instruction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_and_rr_32() {
        let (a, b) = sym32();
        let expr = encode_and_rr(X86OperandSize::S32, a, b);
        let result = expr.eval(&env(&[("a", 0xFF00_FF00), ("b", 0x0F0F_0F0F)]));
        assert_eq!(result, EvalResult::Bv(0x0F00_0F00));
    }

    #[test]
    fn test_or_rr_32() {
        let (a, b) = sym32();
        let expr = encode_or_rr(X86OperandSize::S32, a, b);
        let result = expr.eval(&env(&[("a", 0xFF00_0000), ("b", 0x00FF_0000)]));
        assert_eq!(result, EvalResult::Bv(0xFFFF_0000));
    }

    #[test]
    fn test_xor_rr_32() {
        let (a, b) = sym32();
        let expr = encode_xor_rr(X86OperandSize::S32, a, b);
        let result = expr.eval(&env(&[("a", 0xAAAA_AAAA), ("b", 0x5555_5555)]));
        assert_eq!(result, EvalResult::Bv(0xFFFF_FFFF));
    }

    #[test]
    fn test_not_32() {
        let a = SmtExpr::var("a", 32);
        let expr = encode_not(X86OperandSize::S32, a);
        let result = expr.eval(&env(&[("a", 0)]));
        assert_eq!(result, EvalResult::Bv(0xFFFF_FFFF));
    }

    #[test]
    fn test_not_32_all_ones() {
        let a = SmtExpr::var("a", 32);
        let expr = encode_not(X86OperandSize::S32, a);
        let result = expr.eval(&env(&[("a", 0xFFFF_FFFF)]));
        assert_eq!(result, EvalResult::Bv(0));
    }

    // -----------------------------------------------------------------------
    // Shift instruction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_shl_rr_32() {
        let (a, b) = sym32();
        let expr = encode_shl_rr(X86OperandSize::S32, a, b);
        let result = expr.eval(&env(&[("a", 1), ("b", 4)]));
        assert_eq!(result, EvalResult::Bv(16));
    }

    #[test]
    fn test_shr_rr_32() {
        let (a, b) = sym32();
        let expr = encode_shr_rr(X86OperandSize::S32, a, b);
        let result = expr.eval(&env(&[("a", 0x8000_0000), ("b", 4)]));
        assert_eq!(result, EvalResult::Bv(0x0800_0000));
    }

    #[test]
    fn test_sar_rr_32() {
        let (a, b) = sym32();
        let expr = encode_sar_rr(X86OperandSize::S32, a, b);
        // Arithmetic shift right of 0x80000000 by 4 = 0xF8000000 (sign-extends)
        let result = expr.eval(&env(&[("a", 0x8000_0000), ("b", 4)]));
        assert_eq!(result, EvalResult::Bv(0xF800_0000));
    }

    #[test]
    fn test_sar_rr_32_positive() {
        let (a, b) = sym32();
        let expr = encode_sar_rr(X86OperandSize::S32, a, b);
        // Positive value: 0x7FFFFFFF >> 4 = 0x07FFFFFF (zero-fills)
        let result = expr.eval(&env(&[("a", 0x7FFF_FFFF), ("b", 4)]));
        assert_eq!(result, EvalResult::Bv(0x07FF_FFFF));
    }

    // -----------------------------------------------------------------------
    // Floating-point instruction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fp_add_single() {
        let a = SmtExpr::fp32_const(1.5f32);
        let b = SmtExpr::fp32_const(2.5f32);
        let expr = encode_fp_add_rr(X86FPSize::Single, a, b);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(4.0));
    }

    #[test]
    fn test_fp_add_double() {
        let a = SmtExpr::fp64_const(100.0);
        let b = SmtExpr::fp64_const(200.0);
        let expr = encode_fp_add_rr(X86FPSize::Double, a, b);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(300.0));
    }

    #[test]
    fn test_fp_sub_double() {
        let a = SmtExpr::fp64_const(100.0);
        let b = SmtExpr::fp64_const(42.0);
        let expr = encode_fp_sub_rr(X86FPSize::Double, a, b);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(58.0));
    }

    #[test]
    fn test_fp_mul_double() {
        let a = SmtExpr::fp64_const(6.0);
        let b = SmtExpr::fp64_const(7.0);
        let expr = encode_fp_mul_rr(X86FPSize::Double, a, b);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(42.0));
    }

    #[test]
    fn test_fp_div_double() {
        let a = SmtExpr::fp64_const(84.0);
        let b = SmtExpr::fp64_const(2.0);
        let expr = encode_fp_div_rr(X86FPSize::Double, a, b);
        let result = expr.try_eval(&env(&[])).unwrap();
        assert_eq!(result, EvalResult::Float(42.0));
    }

    #[test]
    fn test_fp_size_parameters() {
        assert_eq!(X86FPSize::Single.eb(), 8);
        assert_eq!(X86FPSize::Single.sb(), 24);
        assert_eq!(X86FPSize::Double.eb(), 11);
        assert_eq!(X86FPSize::Double.sb(), 53);
    }
}
