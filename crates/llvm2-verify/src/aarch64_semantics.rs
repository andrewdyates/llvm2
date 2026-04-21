// llvm2-verify/aarch64_semantics.rs - AArch64 instruction semantics as SMT formulas
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
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

/// Encode `MSUB Wd, Wn, Wm, Wa` or `MSUB Xd, Xn, Xm, Xa` -- multiply-subtract.
///
/// Semantics: `Rd = Ra - (Rn * Rm)` (wrapping, lower bits).
/// Reference: ARM DDI 0487, C6.2.183 MSUB.
///
/// Used by `Urem`/`Srem` lowering: `rem = a - (a / b) * b`.
pub fn encode_msub_rr(size: OperandSize, rn: SmtExpr, rm: SmtExpr, ra: SmtExpr) -> SmtExpr {
    let _ = size;
    ra.bvsub(rn.bvmul(rm))
}

/// Encode `MOV Wd, Wn` or `MOV Xd, Xn` -- register-register move.
///
/// On AArch64, `MOV` is an alias for `ORR Rd, XZR/WZR, Rn` for GPRs.
/// Semantics: `Rd = Rn` (identity).
///
/// Used by `Bitcast` lowering between same-width GPR types.
pub fn encode_mov_rr(size: OperandSize, rn: SmtExpr) -> SmtExpr {
    let _ = size;
    rn
}

/// Encode `FMOV Sd, Sn` / `FMOV Dd, Dn` / `FMOV Dd, Xn` / `FMOV Xd, Dn` --
/// floating-point (or GPR<->FPR) register move.
///
/// Semantics: `Fd = Fn` / `Xd = Dn` -- pure bit-level copy with no rounding,
/// no NaN sanitization, and no width change. Used by `Bitcast` lowering
/// between integer and FP registers of the same width (e.g. `i32<->f32`,
/// `i64<->f64`).
/// Reference: ARM DDI 0487, C7.2.140 FMOV (register), C7.2.141 FMOV (general).
pub fn encode_fmov(rn: SmtExpr) -> SmtExpr {
    rn
}

/// Encode `UBFM Wd, Wn, #immr, #imms` -- unsigned bitfield move.
///
/// This helper covers the extract sub-case (`imms >= immr`), which is how
/// tMIR `ExtractBits { lsb, width }` lowers. With `immr = lsb` and
/// `imms = lsb + width - 1`:
///
///   Wd = zero_extend((Wn >> lsb) & mask(width))
///
/// The helper takes the bitvector width `bv_width` (8 for i8 proofs) and
/// assumes the input is masked to that width. At 8 bits, the result is
/// simply `(rn lsr lsb) & mask(width)` -- zero-extension is a no-op within
/// the 8-bit domain. Requires `lsb + width <= bv_width` and `width >= 1`.
///
/// Reference: ARM DDI 0487, C6.2.335 UBFM (and alias C6.2.334 UBFX).
pub fn encode_ubfm_extract(rn: SmtExpr, lsb: u32, width: u32, bv_width: u32) -> SmtExpr {
    debug_assert!(width >= 1, "encode_ubfm_extract: width must be >= 1");
    debug_assert!(
        lsb + width <= bv_width,
        "encode_ubfm_extract: lsb + width must fit in bv_width"
    );
    debug_assert_eq!(
        rn.bv_width(),
        bv_width,
        "encode_ubfm_extract: operand width must match bv_width"
    );

    let shifted = rn.bvlshr(SmtExpr::bv_const(lsb as u64, bv_width));
    let mask = SmtExpr::bv_const(crate::smt::mask(u64::MAX, width), bv_width);
    shifted.bvand(mask)
}

/// Encode `SBFM Wd, Wn, #immr, #imms` -- signed bitfield move.
///
/// This helper covers the extract sub-case (`imms >= immr`), which is how
/// tMIR `SextractBits { lsb, width }` lowers. With `immr = lsb` and
/// `imms = lsb + width - 1`:
///
///   Wd = sign_extend((Wn >> lsb) & mask(width), from bit (width-1))
///
/// In an 8-bit bitvector domain, we realize this as:
///
///   1. Extract the `width`-bit slice `Wn[lsb+width-1 : lsb]`.
///   2. Sign-extend that slice back up to `bv_width` bits (replicating
///      bit `width-1` of the slice to fill the upper bits).
///
/// SMT `(extract)` + `(sign_extend)` expresses this directly.
/// Requires `lsb + width <= bv_width` and `width >= 1`.
///
/// Reference: ARM DDI 0487, C6.2.266 SBFM (and alias C6.2.264 SBFX).
pub fn encode_sbfm_extract(rn: SmtExpr, lsb: u32, width: u32, bv_width: u32) -> SmtExpr {
    debug_assert!(width >= 1, "encode_sbfm_extract: width must be >= 1");
    debug_assert!(
        lsb + width <= bv_width,
        "encode_sbfm_extract: lsb + width must fit in bv_width"
    );
    debug_assert_eq!(
        rn.bv_width(),
        bv_width,
        "encode_sbfm_extract: operand width must match bv_width"
    );

    let high = lsb + width - 1;
    let slice = rn.extract(high, lsb);
    if width == bv_width {
        // No extension needed -- slice already has the full width.
        slice
    } else {
        slice.sign_ext(bv_width - width)
    }
}

/// Encode `BFM Wd, Wn, #immr, #imms` in its bitfield-insert form (BFI alias).
///
/// `BFI Wd, Wn, #lsb, #width` (`BFM Wd, Wn, #(reg_size - lsb) mod reg_size,
/// #(width - 1)`) copies the low `width` bits of `Wn` into `Wd[lsb+width-1:lsb]`,
/// leaving the other bits of `Wd` unchanged. This is how tMIR
/// `InsertBits { lsb, width }` lowers -- `rd` holds the old value of the
/// destination (propagated from `args[0]` via a `COPY` emitted by ISel;
/// see `isel.rs::select_bitfield_insert`), and `rn` is the source of the
/// bits to insert (`args[1]`).
///
/// Semantics:
///
///   Wd = (Wd_old & ~(mask(width) << lsb)) | ((Wn & mask(width)) << lsb)
///
/// Requires `lsb + width <= bv_width` and `width >= 1`.
///
/// Reference: ARM DDI 0487, C6.2.46 BFM (and alias C6.2.45 BFI).
pub fn encode_bfm_insert(
    rd: SmtExpr,
    rn: SmtExpr,
    lsb: u32,
    width: u32,
    bv_width: u32,
) -> SmtExpr {
    debug_assert!(width >= 1, "encode_bfm_insert: width must be >= 1");
    debug_assert!(
        lsb + width <= bv_width,
        "encode_bfm_insert: lsb + width must fit in bv_width"
    );
    debug_assert_eq!(
        rd.bv_width(),
        bv_width,
        "encode_bfm_insert: Wd width must match bv_width"
    );
    debug_assert_eq!(
        rn.bv_width(),
        bv_width,
        "encode_bfm_insert: Wn width must match bv_width"
    );

    let width_mask = crate::smt::mask(u64::MAX, width);
    let shifted_mask = crate::smt::mask(width_mask << lsb, bv_width);
    let inv_mask = crate::smt::mask(!shifted_mask, bv_width);

    let preserved = rd.bvand(SmtExpr::bv_const(inv_mask, bv_width));
    let insert_slice = rn
        .bvand(SmtExpr::bv_const(width_mask, bv_width))
        .bvshl(SmtExpr::bv_const(lsb as u64, bv_width));

    preserved.bvor(insert_slice)
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

/// Encode `FABS Sd, Sn` or `FABS Dd, Dn` -- floating-point absolute value.
///
/// Semantics: `Fd = |Fn|` (clear sign bit, no rounding needed).
/// Reference: ARM DDI 0487, C7.2.73 FABS (scalar).
pub fn encode_fabs(_size: FPSize, rn: SmtExpr) -> SmtExpr {
    rn.fp_abs()
}

/// Encode `FSQRT Sd, Sn` or `FSQRT Dd, Dn` -- floating-point square root.
///
/// Semantics: `Fd = sqrt(Fn)` with default RNE rounding mode.
/// Reference: ARM DDI 0487, C7.2.160 FSQRT (scalar).
pub fn encode_fsqrt(_size: FPSize, rn: SmtExpr) -> SmtExpr {
    use crate::smt::RoundingMode;
    SmtExpr::fp_sqrt(RoundingMode::RNE, rn)
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

/// Encode `BIC Wd, Wn, Wm` or `BIC Xd, Xn, Xm` — bitwise bit clear (AND-NOT).
///
/// Semantics: `Rd = Rn & ~Rm`.
/// Reference: ARM DDI 0487, C6.2.21 BIC (shifted register).
///
/// Used by `tMIR::BandNot` lowering (`select_logic(..., AArch64LogicOp::Bic, ...)`
/// in `llvm2_lower::isel`). The complement is taken at `rm`'s actual
/// bitvector width so the encoder composes correctly at sub-register widths
/// — I8/I16 proofs encode operands as 8/16-bit bitvectors even though the
/// machine instruction uses a 32-bit W register. The `size` parameter is
/// accepted for consistency with the rest of this file but ignored for SMT
/// encoding (width comes from the operand sort, matching `encode_and_rr` /
/// `encode_orr_rr`).
pub fn encode_bic_rr(size: OperandSize, rn: SmtExpr, rm: SmtExpr) -> SmtExpr {
    let _ = size;
    let width = rm.bv_width();
    let all_ones = SmtExpr::bv_const(crate::smt::mask(u64::MAX, width), width);
    rn.bvand(rm.bvxor(all_ones))
}

/// Encode `ORN Wd, Wn, Wm` or `ORN Xd, Xn, Xm` — bitwise inclusive OR NOT.
///
/// Semantics: `Rd = Rn | ~Rm`.
/// Reference: ARM DDI 0487, C6.2.229 ORN (shifted register).
///
/// Used by `tMIR::BorNot` lowering (`select_logic(..., AArch64LogicOp::Orn, ...)`
/// in `llvm2_lower::isel`). Complements at `rm`'s actual bitvector width for
/// the same sub-register reason noted on [`encode_bic_rr`].
pub fn encode_orn_rr(size: OperandSize, rn: SmtExpr, rm: SmtExpr) -> SmtExpr {
    let _ = size;
    let width = rm.bv_width();
    let all_ones = SmtExpr::bv_const(crate::smt::mask(u64::MAX, width), width);
    rn.bvor(rm.bvxor(all_ones))
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

    #[test]
    fn test_bic_rr_32() {
        let (a, b) = sym32();
        let expr = encode_bic_rr(OperandSize::S32, a, b);
        // a & ~b — clear bits of a where b is set
        let result = expr.eval(&env(&[("a", 0xFFFF_FFFF), ("b", 0x0F0F_0F0F)]));
        assert_eq!(result, EvalResult::Bv(0xF0F0_F0F0));
    }

    #[test]
    fn test_bic_rr_8() {
        // Sub-register width: I8 BandNot lowering proof uses 8-bit operands
        // even though the machine instruction runs in W registers.
        let a = SmtExpr::var("a", 8);
        let b = SmtExpr::var("b", 8);
        let expr = encode_bic_rr(OperandSize::S32, a, b);
        let result = expr.eval(&env(&[("a", 0xFF), ("b", 0x0F)]));
        assert_eq!(result, EvalResult::Bv(0xF0));
    }

    #[test]
    fn test_orn_rr_32() {
        let (a, b) = sym32();
        let expr = encode_orn_rr(OperandSize::S32, a, b);
        // a | ~b
        let result = expr.eval(&env(&[("a", 0x0000_FFFF), ("b", 0xFFFF_0000)]));
        assert_eq!(result, EvalResult::Bv(0x0000_FFFF));
    }

    #[test]
    fn test_orn_rr_8() {
        let a = SmtExpr::var("a", 8);
        let b = SmtExpr::var("b", 8);
        let expr = encode_orn_rr(OperandSize::S32, a, b);
        let result = expr.eval(&env(&[("a", 0x00), ("b", 0x0F)]));
        // ~0x0F (8-bit) = 0xF0
        assert_eq!(result, EvalResult::Bv(0xF0));
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

    // -----------------------------------------------------------------------
    // MSUB / MOV / FMOV semantics tests (issue #435)
    // -----------------------------------------------------------------------

    #[test]
    fn test_msub_rr_32() {
        let (a, b) = sym32();
        let c = SmtExpr::var("c", 32);
        // MSUB: c - (a * b) = 100 - (3 * 4) = 88
        let expr = encode_msub_rr(OperandSize::S32, a, b, c);
        let result = expr.eval(&env(&[("a", 3), ("b", 4), ("c", 100)]));
        assert_eq!(result, EvalResult::Bv(88));
    }

    #[test]
    fn test_msub_rr_models_urem() {
        // Urem lowering: rem = a - (a /u b) * b
        // Concretely: 17 urem 5 = 2 = 17 - (17/5)*5 = 17 - 15
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        let q = a.clone().bvudiv(b.clone());
        let expr = encode_msub_rr(OperandSize::S32, q, b, a);
        let result = expr.eval(&env(&[("a", 17), ("b", 5)]));
        assert_eq!(result, EvalResult::Bv(2));
    }

    #[test]
    fn test_mov_rr_identity() {
        let a = SmtExpr::var("a", 32);
        let expr = encode_mov_rr(OperandSize::S32, a);
        let result = expr.eval(&env(&[("a", 0xDEAD_BEEF)]));
        assert_eq!(result, EvalResult::Bv(0xDEAD_BEEF));
    }

    #[test]
    fn test_fmov_identity() {
        // FMOV between GPR/FPR is a pure bit-level copy.
        let a = SmtExpr::var("a", 64);
        let expr = encode_fmov(a);
        let result = expr.eval(&env(&[("a", 0x3FF0_0000_0000_0000)]));
        assert_eq!(result, EvalResult::Bv(0x3FF0_0000_0000_0000));
    }

    // -----------------------------------------------------------------------
    // UBFM / SBFM / BFM semantics tests (issue #452)
    // -----------------------------------------------------------------------

    #[test]
    fn test_ubfm_extract_mid_nibble_i8() {
        // x = 0b10110100; lsb=2, width=4 -> slice = 0b1101 = 13.
        let a = SmtExpr::var("a", 8);
        let expr = encode_ubfm_extract(a, 2, 4, 8);
        let result = expr.eval(&env(&[("a", 0b1011_0100)]));
        assert_eq!(result, EvalResult::Bv(0b0000_1101));
    }

    #[test]
    fn test_ubfm_extract_low_nibble_i8() {
        // lsb=0, width=4 -> low nibble.
        let a = SmtExpr::var("a", 8);
        let expr = encode_ubfm_extract(a, 0, 4, 8);
        let result = expr.eval(&env(&[("a", 0xAB)]));
        assert_eq!(result, EvalResult::Bv(0x0B));
    }

    #[test]
    fn test_ubfm_extract_full_width_i8() {
        // lsb=0, width=8 -> whole byte, identity.
        let a = SmtExpr::var("a", 8);
        let expr = encode_ubfm_extract(a, 0, 8, 8);
        let result = expr.eval(&env(&[("a", 0xDE)]));
        assert_eq!(result, EvalResult::Bv(0xDE));
    }

    #[test]
    fn test_sbfm_extract_negative_slice_i8() {
        // x = 0b0010_1100; lsb=2, width=4 -> slice = 0b1011 (top bit set).
        // Sign-extends to 0xFB (-5 in i8).
        let a = SmtExpr::var("a", 8);
        let expr = encode_sbfm_extract(a, 2, 4, 8);
        let result = expr.eval(&env(&[("a", 0b0010_1100)]));
        assert_eq!(result, EvalResult::Bv(0xFB));
    }

    #[test]
    fn test_sbfm_extract_nonnegative_slice_i8() {
        // x = 0b0001_0100; lsb=2, width=4 -> slice = 0b0101 (top bit clear).
        // Sign-extend yields 0b0000_0101 = 5.
        let a = SmtExpr::var("a", 8);
        let expr = encode_sbfm_extract(a, 2, 4, 8);
        let result = expr.eval(&env(&[("a", 0b0001_0100)]));
        assert_eq!(result, EvalResult::Bv(0x05));
    }

    #[test]
    fn test_sbfm_extract_full_width_i8() {
        // lsb=0, width=8 -> whole byte, identity (sign-extend by 0).
        let a = SmtExpr::var("a", 8);
        let expr = encode_sbfm_extract(a, 0, 8, 8);
        let result = expr.eval(&env(&[("a", 0x80)]));
        assert_eq!(result, EvalResult::Bv(0x80));
    }

    #[test]
    fn test_bfm_insert_mid_nibble_i8() {
        // Wd = 0b1010_1010; Wn = 0b0000_1101; lsb=2, width=4.
        // Mask of width 4 shifted by 2: 0b0011_1100.
        // Preserved: Wd & ~mask = 0b1010_1010 & 0b1100_0011 = 0b1000_0010.
        // Insert: (Wn & 0b1111) << 2 = 0b0011_0100.
        // Result: 0b1000_0010 | 0b0011_0100 = 0b1011_0110.
        let d = SmtExpr::var("d", 8);
        let n = SmtExpr::var("n", 8);
        let expr = encode_bfm_insert(d, n, 2, 4, 8);
        let result = expr.eval(&env(&[("d", 0b1010_1010), ("n", 0b0000_1101)]));
        assert_eq!(result, EvalResult::Bv(0b1011_0110));
    }

    #[test]
    fn test_bfm_insert_low_nibble_clear_i8() {
        // Wd = 0xFF; Wn = 0x00; lsb=0, width=4 -> clear low nibble.
        let d = SmtExpr::var("d", 8);
        let n = SmtExpr::var("n", 8);
        let expr = encode_bfm_insert(d, n, 0, 4, 8);
        let result = expr.eval(&env(&[("d", 0xFF), ("n", 0x00)]));
        assert_eq!(result, EvalResult::Bv(0xF0));
    }

    #[test]
    fn test_bfm_insert_full_width_i8() {
        // lsb=0, width=8 -> replace entire byte with Wn (Wd ignored).
        let d = SmtExpr::var("d", 8);
        let n = SmtExpr::var("n", 8);
        let expr = encode_bfm_insert(d, n, 0, 8, 8);
        let result = expr.eval(&env(&[("d", 0x12), ("n", 0x34)]));
        assert_eq!(result, EvalResult::Bv(0x34));
    }
}
