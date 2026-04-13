// llvm2-verify/nzcv.rs - AArch64 NZCV flag model and condition code evaluation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Models AArch64 NZCV flags as boolean SMT expressions derived from
// subtraction results. Used to verify comparison and branch lowering rules.
//
// Reference: ARM Architecture Reference Manual (DDI 0487), Section C5.2.9
// Reference: designs/2026-04-13-verification-architecture.md, "NZCV Flag Correctness"

//! AArch64 NZCV flag model and condition code evaluation.
//!
//! The AArch64 processor status register contains four condition flags:
//! - **N** (Negative): set to the MSB of the result
//! - **Z** (Zero): set if the result is zero
//! - **C** (Carry): set if unsigned carry out occurred
//! - **V** (oVerflow): set if signed overflow occurred
//!
//! These flags are set by CMP (which is SUBS with result discarded),
//! ADDS, SUBS, and a few other instructions. They are consumed by
//! conditional branches (B.cond), conditional select (CSEL), and
//! conditional set (CSET).

use crate::smt::SmtExpr;
use llvm2_lower::isel::AArch64CC;

/// NZCV flags as boolean SMT expressions.
///
/// Each field is a Bool-sorted `SmtExpr` representing whether that flag is set.
#[derive(Debug, Clone)]
pub struct NzcvFlags {
    /// N: Negative flag = result MSB.
    pub n: SmtExpr,
    /// Z: Zero flag = (result == 0).
    pub z: SmtExpr,
    /// C: Carry flag = unsigned carry out (for subtraction: Rn >= Rm unsigned).
    pub c: SmtExpr,
    /// V: oVerflow flag = signed overflow.
    pub v: SmtExpr,
}

/// Encode CMP semantics: `CMP Rn, Rm` computes `Rn - Rm` and sets NZCV.
///
/// CMP is an alias for SUBS with the result register being XZR/WZR (discarded).
/// The flags reflect the subtraction `Rn - Rm`.
///
/// # Flag definitions (ARM DDI 0487, C5.2.9)
///
/// For a subtraction `result = Rn - Rm`:
/// - N = result[MSB]
/// - Z = (result == 0)
/// - C = (Rn >=_u Rm)  (unsigned borrow is inverted carry)
/// - V = (sign(Rn) != sign(Rm)) AND (sign(Rn) != sign(result))
///
/// # Arguments
///
/// - `rn`, `rm`: operand expressions (must be same bit-width)
/// - `width`: bitvector width (32 or 64)
pub fn encode_cmp(rn: SmtExpr, rm: SmtExpr, width: u32) -> NzcvFlags {
    let result = rn.clone().bvsub(rm.clone());
    let zero = SmtExpr::bv_const(0, width);

    // N: sign bit of result
    let n = result.clone().extract(width - 1, width - 1)
        .eq_expr(SmtExpr::bv_const(1, 1));

    // Z: result == 0
    let z = result.clone().eq_expr(zero);

    // C: unsigned borrow-out for subtraction is inverted carry.
    // C = 1 iff Rn >=_u Rm (no borrow needed)
    let c = rn.clone().bvuge(rm.clone());

    // V: signed overflow of subtraction.
    // Overflow when: operand signs differ AND result sign differs from Rn sign.
    let rn_msb = rn.extract(width - 1, width - 1);
    let rm_msb = rm.extract(width - 1, width - 1);
    let res_msb = result.extract(width - 1, width - 1);
    let v = rn_msb.clone().eq_expr(rm_msb).not_expr()
        .and_expr(rn_msb.eq_expr(res_msb).not_expr());

    NzcvFlags { n, z, c, v }
}

/// Evaluate a condition code against NZCV flags.
///
/// Returns a Bool-sorted `SmtExpr` that is true iff the condition holds.
///
/// # Condition code definitions (ARM DDI 0487, Table C5-1)
///
/// | CC | Meaning            | Flags         |
/// |----|--------------------|---------------|
/// | EQ | Equal              | Z==1          |
/// | NE | Not Equal          | Z==0          |
/// | HS | Unsigned >=        | C==1          |
/// | LO | Unsigned <         | C==0          |
/// | MI | Negative           | N==1          |
/// | PL | Positive or zero   | N==0          |
/// | VS | Overflow           | V==1          |
/// | VC | No overflow        | V==0          |
/// | HI | Unsigned >         | C==1 AND Z==0 |
/// | LS | Unsigned <=        | C==0 OR Z==1  |
/// | GE | Signed >=          | N==V          |
/// | LT | Signed <           | N!=V          |
/// | GT | Signed >           | Z==0 AND N==V |
/// | LE | Signed <=          | Z==1 OR N!=V  |
pub fn eval_condition(cc: AArch64CC, flags: &NzcvFlags) -> SmtExpr {
    match cc {
        AArch64CC::EQ => flags.z.clone(),
        AArch64CC::NE => flags.z.clone().not_expr(),
        AArch64CC::HS => flags.c.clone(),
        AArch64CC::LO => flags.c.clone().not_expr(),
        AArch64CC::MI => flags.n.clone(),
        AArch64CC::PL => flags.n.clone().not_expr(),
        AArch64CC::VS => flags.v.clone(),
        AArch64CC::VC => flags.v.clone().not_expr(),
        AArch64CC::HI => flags.c.clone().and_expr(flags.z.clone().not_expr()),
        AArch64CC::LS => flags.c.clone().not_expr().or_expr(flags.z.clone()),
        AArch64CC::GE => flags.n.clone().eq_expr(flags.v.clone()),
        AArch64CC::LT => flags.n.clone().eq_expr(flags.v.clone()).not_expr(),
        AArch64CC::GT => flags.z.clone().not_expr()
            .and_expr(flags.n.clone().eq_expr(flags.v.clone())),
        AArch64CC::LE => flags.z.clone()
            .or_expr(flags.n.clone().eq_expr(flags.v.clone()).not_expr()),
    }
}

/// Encode `CSET Wd, cc` — conditional set.
///
/// Produces a 1-bit bitvector: `bv1(1)` if condition holds, `bv1(0)` otherwise.
/// This matches the tMIR comparison result type `B1`.
pub fn encode_cset(cc: AArch64CC, flags: &NzcvFlags) -> SmtExpr {
    let cond = eval_condition(cc, flags);
    SmtExpr::ite(
        cond,
        SmtExpr::bv_const(1, 1),
        SmtExpr::bv_const(0, 1),
    )
}

/// Encode the full AArch64 comparison lowering sequence:
///   `CMP Rn, Rm` ; `CSET Wd, cc`
///
/// Returns a 1-bit bitvector result (same type as tMIR Icmp output).
pub fn encode_cmp_cset(rn: SmtExpr, rm: SmtExpr, width: u32, cc: AArch64CC) -> SmtExpr {
    let flags = encode_cmp(rn, rm, width);
    encode_cset(cc, &flags)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smt::EvalResult;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    // -----------------------------------------------------------------------
    // NZCV flag unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_n_flag_positive_result() {
        let rn = SmtExpr::var("rn", 32);
        let rm = SmtExpr::var("rm", 32);
        let flags = encode_cmp(rn, rm, 32);
        // 10 - 3 = 7, MSB=0 -> N=false
        let result = flags.n.eval(&env(&[("rn", 10), ("rm", 3)]));
        assert_eq!(result, EvalResult::Bool(false));
    }

    #[test]
    fn test_n_flag_negative_result() {
        let rn = SmtExpr::var("rn", 32);
        let rm = SmtExpr::var("rm", 32);
        let flags = encode_cmp(rn, rm, 32);
        // 3 - 10 = -7 (0xFFFFFFF9), MSB=1 -> N=true
        let result = flags.n.eval(&env(&[("rn", 3), ("rm", 10)]));
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_z_flag_zero_result() {
        let rn = SmtExpr::var("rn", 32);
        let rm = SmtExpr::var("rm", 32);
        let flags = encode_cmp(rn, rm, 32);
        // 5 - 5 = 0 -> Z=true
        let result = flags.z.eval(&env(&[("rn", 5), ("rm", 5)]));
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_z_flag_nonzero_result() {
        let rn = SmtExpr::var("rn", 32);
        let rm = SmtExpr::var("rm", 32);
        let flags = encode_cmp(rn, rm, 32);
        // 5 - 3 = 2 -> Z=false
        let result = flags.z.eval(&env(&[("rn", 5), ("rm", 3)]));
        assert_eq!(result, EvalResult::Bool(false));
    }

    #[test]
    fn test_c_flag_no_borrow() {
        let rn = SmtExpr::var("rn", 32);
        let rm = SmtExpr::var("rm", 32);
        let flags = encode_cmp(rn, rm, 32);
        // 10 >= 3 (unsigned) -> C=true (no borrow)
        let result = flags.c.eval(&env(&[("rn", 10), ("rm", 3)]));
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_c_flag_borrow() {
        let rn = SmtExpr::var("rn", 32);
        let rm = SmtExpr::var("rm", 32);
        let flags = encode_cmp(rn, rm, 32);
        // 3 < 10 (unsigned) -> C=false (borrow needed)
        let result = flags.c.eval(&env(&[("rn", 3), ("rm", 10)]));
        assert_eq!(result, EvalResult::Bool(false));
    }

    #[test]
    fn test_v_flag_no_overflow() {
        let rn = SmtExpr::var("rn", 32);
        let rm = SmtExpr::var("rm", 32);
        let flags = encode_cmp(rn, rm, 32);
        // 10 - 3: same-sign operands (both positive), no overflow
        let result = flags.v.eval(&env(&[("rn", 10), ("rm", 3)]));
        assert_eq!(result, EvalResult::Bool(false));
    }

    #[test]
    fn test_v_flag_overflow() {
        let rn = SmtExpr::var("rn", 32);
        let rm = SmtExpr::var("rm", 32);
        let flags = encode_cmp(rn, rm, 32);
        // INT_MAX - (-1) = INT_MAX + 1 = overflow
        // 0x7FFFFFFF - 0xFFFFFFFF: rn is positive, rm is negative,
        // result is 0x80000000 (negative) -> sign(rn) != sign(result) -> V=true
        let result = flags.v.eval(&env(&[("rn", 0x7FFF_FFFF), ("rm", 0xFFFF_FFFF)]));
        assert_eq!(result, EvalResult::Bool(true));
    }

    // -----------------------------------------------------------------------
    // Condition code evaluation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_eq_condition_equal() {
        let rn = SmtExpr::var("rn", 32);
        let rm = SmtExpr::var("rm", 32);
        let flags = encode_cmp(rn, rm, 32);
        let cond = eval_condition(AArch64CC::EQ, &flags);
        // 5 == 5
        let result = cond.eval(&env(&[("rn", 5), ("rm", 5)]));
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_eq_condition_not_equal() {
        let rn = SmtExpr::var("rn", 32);
        let rm = SmtExpr::var("rm", 32);
        let flags = encode_cmp(rn, rm, 32);
        let cond = eval_condition(AArch64CC::EQ, &flags);
        // 5 != 3
        let result = cond.eval(&env(&[("rn", 5), ("rm", 3)]));
        assert_eq!(result, EvalResult::Bool(false));
    }

    #[test]
    fn test_lt_condition_signed_less() {
        let rn = SmtExpr::var("rn", 32);
        let rm = SmtExpr::var("rm", 32);
        let flags = encode_cmp(rn, rm, 32);
        let cond = eval_condition(AArch64CC::LT, &flags);
        // -1 < 0 (signed)
        let result = cond.eval(&env(&[("rn", 0xFFFF_FFFF), ("rm", 0)]));
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_lo_condition_unsigned_less() {
        let rn = SmtExpr::var("rn", 32);
        let rm = SmtExpr::var("rm", 32);
        let flags = encode_cmp(rn, rm, 32);
        let cond = eval_condition(AArch64CC::LO, &flags);
        // 3 <_u 10
        let result = cond.eval(&env(&[("rn", 3), ("rm", 10)]));
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_lo_condition_not_unsigned_less() {
        let rn = SmtExpr::var("rn", 32);
        let rm = SmtExpr::var("rm", 32);
        let flags = encode_cmp(rn, rm, 32);
        let cond = eval_condition(AArch64CC::LO, &flags);
        // 10 is NOT <_u 3
        let result = cond.eval(&env(&[("rn", 10), ("rm", 3)]));
        assert_eq!(result, EvalResult::Bool(false));
    }

    #[test]
    fn test_hi_condition_unsigned_greater() {
        let rn = SmtExpr::var("rn", 32);
        let rm = SmtExpr::var("rm", 32);
        let flags = encode_cmp(rn, rm, 32);
        let cond = eval_condition(AArch64CC::HI, &flags);
        // 10 >_u 3
        let result = cond.eval(&env(&[("rn", 10), ("rm", 3)]));
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_gt_condition_signed_greater() {
        let rn = SmtExpr::var("rn", 32);
        let rm = SmtExpr::var("rm", 32);
        let flags = encode_cmp(rn, rm, 32);
        let cond = eval_condition(AArch64CC::GT, &flags);
        // 0 > -1 (signed)
        let result = cond.eval(&env(&[("rn", 0), ("rm", 0xFFFF_FFFF)]));
        assert_eq!(result, EvalResult::Bool(true));
    }

    // -----------------------------------------------------------------------
    // CSET tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cset_eq_equal() {
        let rn = SmtExpr::var("rn", 32);
        let rm = SmtExpr::var("rm", 32);
        let cset = encode_cmp_cset(rn, rm, 32, AArch64CC::EQ);
        let result = cset.eval(&env(&[("rn", 42), ("rm", 42)]));
        assert_eq!(result, EvalResult::Bv(1));
    }

    #[test]
    fn test_cset_eq_not_equal() {
        let rn = SmtExpr::var("rn", 32);
        let rm = SmtExpr::var("rm", 32);
        let cset = encode_cmp_cset(rn, rm, 32, AArch64CC::EQ);
        let result = cset.eval(&env(&[("rn", 42), ("rm", 43)]));
        assert_eq!(result, EvalResult::Bv(0));
    }

    #[test]
    fn test_cset_lt_negative() {
        let rn = SmtExpr::var("rn", 32);
        let rm = SmtExpr::var("rm", 32);
        let cset = encode_cmp_cset(rn, rm, 32, AArch64CC::LT);
        // -5 < 5 -> CSET produces 1
        let neg5 = 0xFFFF_FFFBu64;
        let result = cset.eval(&env(&[("rn", neg5), ("rm", 5)]));
        assert_eq!(result, EvalResult::Bv(1));
    }

    #[test]
    fn test_cset_lo_unsigned() {
        let rn = SmtExpr::var("rn", 32);
        let rm = SmtExpr::var("rm", 32);
        let cset = encode_cmp_cset(rn, rm, 32, AArch64CC::LO);
        // 3 <_u 10 -> CSET produces 1
        let result = cset.eval(&env(&[("rn", 3), ("rm", 10)]));
        assert_eq!(result, EvalResult::Bv(1));
    }

    // -----------------------------------------------------------------------
    // 64-bit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cmp_64bit_zero() {
        let rn = SmtExpr::var("rn", 64);
        let rm = SmtExpr::var("rm", 64);
        let flags = encode_cmp(rn, rm, 64);
        // Equal values -> Z=true
        let result = flags.z.eval(&env(&[("rn", 0x1234_5678_9ABC_DEF0), ("rm", 0x1234_5678_9ABC_DEF0)]));
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_cmp_64bit_negative() {
        let rn = SmtExpr::var("rn", 64);
        let rm = SmtExpr::var("rm", 64);
        let flags = encode_cmp(rn, rm, 64);
        // 0 - 1 = max (negative in signed) -> N=true
        let result = flags.n.eval(&env(&[("rn", 0), ("rm", 1)]));
        assert_eq!(result, EvalResult::Bool(true));
    }
}
