// llvm2-verify/x86_64_eflags.rs - x86-64 EFLAGS model and condition code evaluation
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Models the subset of x86-64 EFLAGS needed for integer CMP/Jcc/SETcc
// verification. This is the x86-64 analogue of `nzcv.rs`.
//
// Reference: Intel 64 and IA-32 Architectures Software Developer's Manual
// Reference: designs/2026-04-13-verification-architecture.md, "Flag Correctness"

//! x86-64 EFLAGS model and condition code evaluation.
//!
//! The x86-64 `CMP` instruction computes `src1 - src2`, discards the numeric
//! result, and updates condition flags consumed by `Jcc`, `SETcc`, and `CMOVcc`.
//! For integer comparison verification we model four flags:
//!
//! - `CF` (Carry): set when subtraction borrows, i.e. `src1 <_u src2`
//! - `ZF` (Zero): set when the subtraction result is zero
//! - `SF` (Sign): set to the sign bit of the subtraction result
//! - `OF` (Overflow): set on signed overflow
//!
//! This differs from AArch64 NZCV in one important way: for subtraction,
//! x86 `CF` means "borrow occurred", while AArch64 `C` means "no borrow
//! occurred". The overflow formula is otherwise the same as AArch64 `V`.
//!
//! Parity is not modeled yet. `P` and `NP` conditions therefore use explicit
//! placeholder expressions in [`eval_x86_condition`].

use crate::smt::SmtExpr;
use llvm2_ir::X86CondCode;

/// x86-64 EFLAGS represented as boolean SMT expressions.
///
/// Each field is a Bool-sorted [`SmtExpr`] describing whether that flag is set
/// after a comparison.
#[derive(Debug, Clone)]
pub struct EflagsFlags {
    /// CF: carry/borrow flag.
    pub cf: SmtExpr,
    /// ZF: zero flag.
    pub zf: SmtExpr,
    /// SF: sign flag.
    pub sf: SmtExpr,
    /// OF: signed overflow flag, spelled `of_` to avoid Rust keyword.
    pub of_: SmtExpr,
}

/// Encode x86-64 `CMP` semantics as EFLAGS expressions.
///
/// `CMP src1, src2` computes `src1 - src2` and updates flags without retaining
/// the arithmetic result. The result width should match the operands and is
/// expected to be 32 or 64 bits for the current lowering rules.
///
/// Flag definitions for `result = src1 - src2`:
///
/// - `CF = (src1 <_u src2)`
/// - `ZF = (result == 0)`
/// - `SF = result[MSB]`
/// - `OF = (sign(src1) != sign(src2)) AND (sign(src1) != sign(result))`
///
/// Note that x86 `CF` is the inverse of AArch64 subtraction carry:
/// x86 sets `CF` on borrow, while AArch64 sets `C` when no borrow occurs.
pub fn encode_cmp_eflags(src1: SmtExpr, src2: SmtExpr, width: u32) -> EflagsFlags {
    let result = src1.clone().bvsub(src2.clone());
    let zero = SmtExpr::bv_const(0, width);

    // CF: subtraction borrowed in the unsigned domain.
    let cf = src1.clone().bvult(src2.clone());

    // ZF: result is exactly zero.
    let zf = result.clone().eq_expr(zero);

    // SF: sign bit of the result.
    let sf = result
        .clone()
        .extract(width - 1, width - 1)
        .eq_expr(SmtExpr::bv_const(1, 1));

    // OF: signed overflow on subtraction.
    let src1_msb = src1.extract(width - 1, width - 1);
    let src2_msb = src2.extract(width - 1, width - 1);
    let result_msb = result.extract(width - 1, width - 1);
    let of_ = src1_msb
        .clone()
        .eq_expr(src2_msb)
        .not_expr()
        .and_expr(src1_msb.eq_expr(result_msb).not_expr());

    EflagsFlags { cf, zf, sf, of_ }
}

/// Evaluate an x86 condition code against EFLAGS.
///
/// This maps the hardware `Jcc`/`SETcc` condition encoding to a Bool-sorted
/// SMT expression. Conditions follow the Intel SDM definitions:
///
/// | CC  | Meaning              | EFLAGS           |
/// |-----|----------------------|------------------|
/// | O   | Overflow             | OF=1             |
/// | NO  | No overflow          | OF=0             |
/// | B   | Below (unsigned <)   | CF=1             |
/// | AE  | Above or equal       | CF=0             |
/// | E   | Equal                | ZF=1             |
/// | NE  | Not equal            | ZF=0             |
/// | BE  | Below or equal       | CF=1 OR ZF=1     |
/// | A   | Above (unsigned >)   | CF=0 AND ZF=0    |
/// | S   | Sign (negative)      | SF=1             |
/// | NS  | No sign              | SF=0             |
/// | P   | Parity even          | (placeholder)    |
/// | NP  | Parity odd           | (placeholder)    |
/// | L   | Less (signed <)      | SF!=OF           |
/// | GE  | Greater or equal     | SF=OF            |
/// | LE  | Less or equal        | ZF=1 OR SF!=OF   |
/// | G   | Greater (signed >)   | ZF=0 AND SF=OF   |
///
/// `P` and `NP` are currently placeholders because `PF` is not modeled yet.
/// They evaluate to symbolic expressions that are always false / always true.
pub fn eval_x86_condition(cc: X86CondCode, flags: &EflagsFlags) -> SmtExpr {
    match cc {
        X86CondCode::O => flags.of_.clone(),
        X86CondCode::NO => flags.of_.clone().not_expr(),
        X86CondCode::B => flags.cf.clone(),
        X86CondCode::AE => flags.cf.clone().not_expr(),
        X86CondCode::E => flags.zf.clone(),
        X86CondCode::NE => flags.zf.clone().not_expr(),
        X86CondCode::BE => flags.cf.clone().or_expr(flags.zf.clone()),
        X86CondCode::A => flags
            .cf
            .clone()
            .not_expr()
            .and_expr(flags.zf.clone().not_expr()),
        X86CondCode::S => flags.sf.clone(),
        X86CondCode::NS => flags.sf.clone().not_expr(),
        // PF not modeled: P is always false, NP is always true.
        X86CondCode::P => flags.zf.clone().and_expr(flags.zf.clone().not_expr()),
        X86CondCode::NP => flags.zf.clone().or_expr(flags.zf.clone().not_expr()),
        X86CondCode::L => flags.sf.clone().eq_expr(flags.of_.clone()).not_expr(),
        X86CondCode::GE => flags.sf.clone().eq_expr(flags.of_.clone()),
        X86CondCode::LE => flags.zf.clone().or_expr(
            flags.sf.clone().eq_expr(flags.of_.clone()).not_expr(),
        ),
        X86CondCode::G => flags
            .zf
            .clone()
            .not_expr()
            .and_expr(flags.sf.clone().eq_expr(flags.of_.clone())),
    }
}

/// Encode `SETcc` as a 1-bit bitvector.
///
/// Returns `1` when the condition holds and `0` otherwise. This matches the
/// verifier's comparison-result convention of a `B1` bitvector rather than a
/// Bool-sorted SMT expression.
pub fn encode_setcc(cc: X86CondCode, flags: &EflagsFlags) -> SmtExpr {
    let cond = eval_x86_condition(cc, flags);
    SmtExpr::ite(cond, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1))
}

/// Encode the full x86-64 compare-and-set sequence:
///
/// `CMP src1, src2` ; `SETcc`
///
/// Returns a 1-bit bitvector result (same type as tMIR Icmp output).
pub fn encode_cmp_setcc(
    src1: SmtExpr,
    src2: SmtExpr,
    width: u32,
    cc: X86CondCode,
) -> SmtExpr {
    let flags = encode_cmp_eflags(src1, src2, width);
    encode_setcc(cc, &flags)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smt::EvalResult;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn cmp_flags(width: u32) -> EflagsFlags {
        let src1 = SmtExpr::var("src1", width);
        let src2 = SmtExpr::var("src2", width);
        encode_cmp_eflags(src1, src2, width)
    }

    fn assert_bool(expr: SmtExpr, pairs: &[(&str, u64)], expected: bool) {
        assert_eq!(expr.eval(&env(pairs)), EvalResult::Bool(expected));
    }

    fn assert_bv(expr: SmtExpr, pairs: &[(&str, u64)], expected: u64) {
        assert_eq!(expr.eval(&env(pairs)), EvalResult::Bv(expected));
    }

    fn assert_condition(width: u32, cc: X86CondCode, src1: u64, src2: u64, expected: bool) {
        let flags = cmp_flags(width);
        let cond = eval_x86_condition(cc, &flags);
        assert_bool(cond, &[("src1", src1), ("src2", src2)], expected);
    }

    fn assert_setcc(width: u32, cc: X86CondCode, src1: u64, src2: u64, expected: u64) {
        let src1_expr = SmtExpr::var("src1", width);
        let src2_expr = SmtExpr::var("src2", width);
        let setcc = encode_cmp_setcc(src1_expr, src2_expr, width, cc);
        assert_bv(setcc, &[("src1", src1), ("src2", src2)], expected);
    }

    // -----------------------------------------------------------------------
    // EFLAGS flag tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cf_tracks_unsigned_borrow_32() {
        let flags = cmp_flags(32);

        // 3 - 10 borrows in unsigned arithmetic.
        assert_bool(flags.cf.clone(), &[("src1", 3), ("src2", 10)], true);

        // 10 - 3 does not borrow.
        assert_bool(flags.cf, &[("src1", 10), ("src2", 3)], false);
    }

    #[test]
    fn test_zf_tracks_zero_result_32() {
        let flags = cmp_flags(32);

        // Equal operands produce zero.
        assert_bool(flags.zf.clone(), &[("src1", 5), ("src2", 5)], true);

        // Unequal operands produce a non-zero result.
        assert_bool(flags.zf, &[("src1", 5), ("src2", 3)], false);
    }

    #[test]
    fn test_sf_tracks_result_sign_32() {
        let flags = cmp_flags(32);

        // 3 - 10 = 0xFFFF_FFF9, sign bit set.
        assert_bool(flags.sf.clone(), &[("src1", 3), ("src2", 10)], true);

        // 10 - 3 = 7, sign bit clear.
        assert_bool(flags.sf, &[("src1", 10), ("src2", 3)], false);
    }

    #[test]
    fn test_of_tracks_signed_overflow_32() {
        let flags = cmp_flags(32);

        // INT_MAX - (-1) => 0x7FFF_FFFF - 0xFFFF_FFFF = 0x8000_0000, overflow.
        assert_bool(
            flags.of_.clone(),
            &[("src1", 0x7FFF_FFFF), ("src2", 0xFFFF_FFFF)],
            true,
        );

        // A small positive subtraction does not overflow.
        assert_bool(flags.of_, &[("src1", 10), ("src2", 3)], false);
    }

    #[test]
    fn test_eflags_flags_64bit() {
        let flags = cmp_flags(64);

        // 0 - 1 borrows and produces a negative signed result.
        assert_bool(flags.cf.clone(), &[("src1", 0), ("src2", 1)], true);
        assert_bool(flags.sf.clone(), &[("src1", 0), ("src2", 1)], true);

        // Equal 64-bit values set ZF.
        assert_bool(
            flags.zf.clone(),
            &[
                ("src1", 0x1234_5678_9ABC_DEF0),
                ("src2", 0x1234_5678_9ABC_DEF0),
            ],
            true,
        );

        // INT64_MIN - 1 overflows in the signed domain.
        assert_bool(
            flags.of_,
            &[("src1", 0x8000_0000_0000_0000), ("src2", 1)],
            true,
        );
    }

    // -----------------------------------------------------------------------
    // Condition code evaluation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_conditions_equal_and_not_equal() {
        assert_condition(32, X86CondCode::E, 5, 5, true);
        assert_condition(32, X86CondCode::E, 5, 3, false);

        assert_condition(32, X86CondCode::NE, 5, 5, false);
        assert_condition(32, X86CondCode::NE, 5, 3, true);
    }

    #[test]
    fn test_conditions_unsigned_relations() {
        assert_condition(32, X86CondCode::B, 3, 10, true);
        assert_condition(32, X86CondCode::B, 10, 3, false);

        assert_condition(32, X86CondCode::AE, 3, 10, false);
        assert_condition(32, X86CondCode::AE, 10, 3, true);

        // BE is true both for borrow and for equality.
        assert_condition(32, X86CondCode::BE, 3, 10, true);
        assert_condition(32, X86CondCode::BE, 5, 5, true);
        assert_condition(32, X86CondCode::BE, 10, 3, false);

        assert_condition(32, X86CondCode::A, 10, 3, true);
        assert_condition(32, X86CondCode::A, 5, 5, false);
    }

    #[test]
    fn test_conditions_sign_and_not_sign() {
        assert_condition(32, X86CondCode::S, 3, 10, true);
        assert_condition(32, X86CondCode::S, 10, 3, false);

        assert_condition(32, X86CondCode::NS, 3, 10, false);
        assert_condition(32, X86CondCode::NS, 10, 3, true);
    }

    #[test]
    fn test_conditions_signed_relations() {
        let neg1 = 0xFFFF_FFFF;

        // -1 < 0
        assert_condition(32, X86CondCode::L, neg1, 0, true);
        assert_condition(32, X86CondCode::GE, neg1, 0, false);
        assert_condition(32, X86CondCode::LE, neg1, 0, true);
        assert_condition(32, X86CondCode::G, neg1, 0, false);

        // 0 > -1
        assert_condition(32, X86CondCode::L, 0, neg1, false);
        assert_condition(32, X86CondCode::GE, 0, neg1, true);
        assert_condition(32, X86CondCode::LE, 0, neg1, false);
        assert_condition(32, X86CondCode::G, 0, neg1, true);
    }

    #[test]
    fn test_conditions_overflow_and_no_overflow() {
        // INT_MIN - 1 overflows.
        assert_condition(32, X86CondCode::O, 0x8000_0000, 1, true);
        assert_condition(32, X86CondCode::NO, 0x8000_0000, 1, false);

        // 10 - 3 does not overflow.
        assert_condition(32, X86CondCode::O, 10, 3, false);
        assert_condition(32, X86CondCode::NO, 10, 3, true);
    }

    #[test]
    fn test_parity_conditions_are_placeholders() {
        // PF is not modeled yet, so P is always false and NP is always true.
        assert_condition(32, X86CondCode::P, 5, 5, false);
        assert_condition(32, X86CondCode::NP, 5, 5, true);

        assert_condition(32, X86CondCode::P, 5, 3, false);
        assert_condition(32, X86CondCode::NP, 5, 3, true);
    }

    // -----------------------------------------------------------------------
    // SETcc tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_setcc_equal_and_not_equal_32() {
        assert_setcc(32, X86CondCode::E, 42, 42, 1);
        assert_setcc(32, X86CondCode::E, 42, 43, 0);

        assert_setcc(32, X86CondCode::NE, 42, 42, 0);
        assert_setcc(32, X86CondCode::NE, 42, 43, 1);
    }

    #[test]
    fn test_setcc_unsigned_relations_32() {
        assert_setcc(32, X86CondCode::B, 3, 10, 1);
        assert_setcc(32, X86CondCode::AE, 3, 10, 0);

        assert_setcc(32, X86CondCode::BE, 5, 5, 1);
        assert_setcc(32, X86CondCode::A, 10, 3, 1);
        assert_setcc(32, X86CondCode::A, 5, 5, 0);
    }

    #[test]
    fn test_setcc_signed_relations_32() {
        let neg5 = 0xFFFF_FFFB;
        let neg1 = 0xFFFF_FFFF;

        assert_setcc(32, X86CondCode::L, neg5, 5, 1);
        assert_setcc(32, X86CondCode::GE, neg5, 5, 0);

        assert_setcc(32, X86CondCode::LE, neg1, 0, 1);
        assert_setcc(32, X86CondCode::G, 0, neg1, 1);
    }

    #[test]
    fn test_setcc_overflow_and_sign_32() {
        assert_setcc(32, X86CondCode::O, 0x8000_0000, 1, 1);
        assert_setcc(32, X86CondCode::NO, 0x8000_0000, 1, 0);

        assert_setcc(32, X86CondCode::S, 3, 10, 1);
        assert_setcc(32, X86CondCode::NS, 10, 3, 1);
    }

    #[test]
    fn test_setcc_64bit_relations() {
        let int64_min = 0x8000_0000_0000_0000;

        // Signed less-than and unsigned above can both be true for the same bit
        // pattern, which is exactly why x86 uses different flag combinations.
        assert_setcc(64, X86CondCode::L, int64_min, 0, 1);
        assert_setcc(64, X86CondCode::GE, int64_min, 0, 0);
        assert_setcc(64, X86CondCode::A, int64_min, 0, 1);
        assert_setcc(64, X86CondCode::B, int64_min, 0, 0);

        assert_setcc(
            64,
            X86CondCode::E,
            0x1234_5678_9ABC_DEF0,
            0x1234_5678_9ABC_DEF0,
            1,
        );
    }
}
