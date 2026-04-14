// llvm2-verify/neon_semantics.rs - AArch64 NEON SIMD instruction semantics
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Encodes AArch64 NEON (Advanced SIMD) instruction semantics as bitvector
// SMT expressions using lane decomposition. Each NEON instruction operates
// on a 64-bit or 128-bit vector register, treating it as multiple lanes of
// smaller elements. The semantic encoding extracts lanes, applies the scalar
// operation per-lane, and reassembles the result.
//
// Reference: ARM Architecture Reference Manual (DDI 0487), Sections C7.2
//   (SIMD and Floating-point Instructions, alphabetical listing)
// Reference: designs/2026-04-13-verification-architecture.md

//! AArch64 NEON SIMD instruction semantics encoded as [`SmtExpr`] formulas.
//!
//! Key principle: NEON integer operations decompose into per-lane scalar
//! operations. A 128-bit `ADD.4S` is semantically equivalent to four
//! independent 32-bit additions. This decomposition is the foundation for
//! verifying NEON lowering rules: we prove each lane independently.
//!
//! The lane decomposition pattern:
//! 1. Extract each lane from operand vectors using `lane_extract`
//! 2. Apply the scalar operation to corresponding lanes
//! 3. Reassemble using `concat_lanes`
//!
//! Bitwise operations (AND, ORR, EOR) operate on the full 128-bit vector
//! without lane decomposition since they are bit-parallel.

use crate::smt::{
    SmtExpr, VectorArrangement,
    map_lanes_binary, map_lanes_binary_imm, map_lanes_unary,
};

// ---------------------------------------------------------------------------
// NEON integer arithmetic
// ---------------------------------------------------------------------------

/// Encode `ADD.<T> Vd, Vn, Vm` -- NEON vector integer add.
///
/// Semantics: for each lane `i`: `Vd[i] = Vn[i] + Vm[i]` (wrapping).
///
/// Valid arrangements: 8B, 16B, 4H, 8H, 2S, 4S, 2D.
/// Reference: ARM DDI 0487, C7.2.1 ADD (vector).
pub fn encode_neon_add(
    arrangement: VectorArrangement,
    vn: &SmtExpr,
    vm: &SmtExpr,
) -> SmtExpr {
    map_lanes_binary(vn, vm, arrangement, |a, b| a.bvadd(b))
}

/// Encode `SUB.<T> Vd, Vn, Vm` -- NEON vector integer subtract.
///
/// Semantics: for each lane `i`: `Vd[i] = Vn[i] - Vm[i]` (wrapping).
///
/// Valid arrangements: 8B, 16B, 4H, 8H, 2S, 4S, 2D.
/// Reference: ARM DDI 0487, C7.2.323 SUB (vector).
pub fn encode_neon_sub(
    arrangement: VectorArrangement,
    vn: &SmtExpr,
    vm: &SmtExpr,
) -> SmtExpr {
    map_lanes_binary(vn, vm, arrangement, |a, b| a.bvsub(b))
}

/// Encode `MUL.<T> Vd, Vn, Vm` -- NEON vector integer multiply.
///
/// Semantics: for each lane `i`: `Vd[i] = Vn[i] * Vm[i]` (wrapping, lower bits).
///
/// Valid arrangements: 8B, 16B, 4H, 8H, 2S, 4S.
/// Note: no 2D (64-bit lane) integer multiply in AArch64 NEON.
/// Reference: ARM DDI 0487, C7.2.208 MUL (vector).
pub fn encode_neon_mul(
    arrangement: VectorArrangement,
    vn: &SmtExpr,
    vm: &SmtExpr,
) -> SmtExpr {
    debug_assert!(
        arrangement != VectorArrangement::D2,
        "NEON MUL does not support 2D arrangement"
    );
    map_lanes_binary(vn, vm, arrangement, |a, b| a.bvmul(b))
}

/// Encode `NEG.<T> Vd, Vn` -- NEON vector integer negate.
///
/// Semantics: for each lane `i`: `Vd[i] = -Vn[i]` (two's complement).
///
/// Valid arrangements: 8B, 16B, 4H, 8H, 2S, 4S, 2D.
/// Reference: ARM DDI 0487, C7.2.209 NEG (vector).
pub fn encode_neon_neg(
    arrangement: VectorArrangement,
    vn: &SmtExpr,
) -> SmtExpr {
    map_lanes_unary(vn, arrangement, |a| a.bvneg())
}

// ---------------------------------------------------------------------------
// NEON bitwise operations
// ---------------------------------------------------------------------------

/// Encode `AND.16B Vd, Vn, Vm` -- NEON bitwise AND.
///
/// Semantics: `Vd = Vn AND Vm` (full 128-bit bitwise).
/// No lane decomposition needed -- bitwise AND is bit-parallel.
///
/// Only valid for 16B arrangement (operates on full 128-bit register).
/// 8B variant operates on 64-bit (lower half).
/// Reference: ARM DDI 0487, C7.2.9 AND (vector).
pub fn encode_neon_and(vn: &SmtExpr, vm: &SmtExpr) -> SmtExpr {
    vn.clone().bvand(vm.clone())
}

/// Encode `ORR.16B Vd, Vn, Vm` -- NEON bitwise OR.
///
/// Semantics: `Vd = Vn OR Vm` (full 128-bit bitwise).
/// Reference: ARM DDI 0487, C7.2.215 ORR (vector, register).
pub fn encode_neon_orr(vn: &SmtExpr, vm: &SmtExpr) -> SmtExpr {
    vn.clone().bvor(vm.clone())
}

/// Encode `EOR.16B Vd, Vn, Vm` -- NEON bitwise exclusive OR.
///
/// Semantics: `Vd = Vn XOR Vm` (full 128-bit bitwise).
/// Reference: ARM DDI 0487, C7.2.71 EOR (vector).
pub fn encode_neon_eor(vn: &SmtExpr, vm: &SmtExpr) -> SmtExpr {
    vn.clone().bvxor(vm.clone())
}

/// Encode `BIC.16B Vd, Vn, Vm` -- NEON bitwise bit clear (AND NOT).
///
/// Semantics: `Vd = Vn AND NOT(Vm)`.
/// Reference: ARM DDI 0487, C7.2.15 BIC (vector, register).
pub fn encode_neon_bic(vn: &SmtExpr, vm: &SmtExpr) -> SmtExpr {
    // BIC = AND with complement of second operand.
    // Since we don't have a BvNot, we XOR with all-ones then AND.
    let width = vm.bv_width();
    let all_ones = SmtExpr::bv_const(if width >= 64 { u64::MAX } else { (1u64 << width) - 1 }, width);
    let not_vm = vm.clone().bvxor(all_ones);
    vn.clone().bvand(not_vm)
}

// ---------------------------------------------------------------------------
// NEON shift operations
// ---------------------------------------------------------------------------

/// Encode `SHL.<T> Vd, Vn, #imm` -- NEON vector shift left (immediate).
///
/// Semantics: for each lane `i`: `Vd[i] = Vn[i] << imm`.
/// The shift amount is a compile-time constant, not a register.
///
/// Valid arrangements: 8B, 16B, 4H, 8H, 2S, 4S, 2D.
/// Constraint: `0 <= imm < lane_bits`.
/// Reference: ARM DDI 0487, C7.2.268 SHL (vector).
pub fn encode_neon_shl(
    arrangement: VectorArrangement,
    vn: &SmtExpr,
    imm: u32,
) -> SmtExpr {
    debug_assert!(
        imm < arrangement.lane_bits(),
        "SHL immediate must be < lane_bits"
    );
    map_lanes_binary_imm(vn, imm as u64, arrangement, |a, b| a.bvshl(b))
}

/// Encode `USHR.<T> Vd, Vn, #imm` -- NEON vector unsigned shift right (immediate).
///
/// Semantics: for each lane `i`: `Vd[i] = Vn[i] >> imm` (logical/unsigned).
/// The shift amount is a compile-time constant.
///
/// Valid arrangements: 8B, 16B, 4H, 8H, 2S, 4S, 2D.
/// Constraint: `1 <= imm <= lane_bits`.
/// Reference: ARM DDI 0487, C7.2.387 USHR (vector).
pub fn encode_neon_ushr(
    arrangement: VectorArrangement,
    vn: &SmtExpr,
    imm: u32,
) -> SmtExpr {
    debug_assert!(
        imm >= 1 && imm <= arrangement.lane_bits(),
        "USHR immediate must be in [1, lane_bits]"
    );
    map_lanes_binary_imm(vn, imm as u64, arrangement, |a, b| a.bvlshr(b))
}

/// Encode `SSHR.<T> Vd, Vn, #imm` -- NEON vector signed shift right (immediate).
///
/// Semantics: for each lane `i`: `Vd[i] = Vn[i] >>s imm` (arithmetic/signed).
/// The shift amount is a compile-time constant.
///
/// Valid arrangements: 8B, 16B, 4H, 8H, 2S, 4S, 2D.
/// Constraint: `1 <= imm <= lane_bits`.
/// Reference: ARM DDI 0487, C7.2.304 SSHR (vector).
pub fn encode_neon_sshr(
    arrangement: VectorArrangement,
    vn: &SmtExpr,
    imm: u32,
) -> SmtExpr {
    debug_assert!(
        imm >= 1 && imm <= arrangement.lane_bits(),
        "SSHR immediate must be in [1, lane_bits]"
    );
    map_lanes_binary_imm(vn, imm as u64, arrangement, |a, b| a.bvashr(b))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smt::EvalResult;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    // -----------------------------------------------------------------------
    // Helper: pack lanes into a vector value for testing.
    // We can only test 64-bit total vectors (S2, H4, B8) using the u64 evaluator.
    // 128-bit vectors (S4, H8, B16, D2) would need u128 support in the evaluator.
    // -----------------------------------------------------------------------

    /// Pack two 32-bit values into a 64-bit vector (S2 arrangement).
    /// Lane 0 is least-significant.
    fn pack_2s(lane0: u32, lane1: u32) -> u64 {
        (lane1 as u64) << 32 | (lane0 as u64)
    }

    /// Pack four 16-bit values into a 64-bit vector (H4 arrangement).
    fn pack_4h(l0: u16, l1: u16, l2: u16, l3: u16) -> u64 {
        (l3 as u64) << 48 | (l2 as u64) << 32 | (l1 as u64) << 16 | (l0 as u64)
    }

    /// Pack eight 8-bit values into a 64-bit vector (B8 arrangement).
    fn pack_8b(l0: u8, l1: u8, l2: u8, l3: u8, l4: u8, l5: u8, l6: u8, l7: u8) -> u64 {
        (l7 as u64) << 56
            | (l6 as u64) << 48
            | (l5 as u64) << 40
            | (l4 as u64) << 32
            | (l3 as u64) << 24
            | (l2 as u64) << 16
            | (l1 as u64) << 8
            | (l0 as u64)
    }

    // =======================================================================
    // ADD tests
    // =======================================================================

    #[test]
    fn test_neon_add_2s() {
        let vn = SmtExpr::var("vn", 64);
        let vm = SmtExpr::var("vm", 64);
        let result = encode_neon_add(VectorArrangement::S2, &vn, &vm);

        // vn = [10, 20], vm = [3, 7]
        let e = env(&[
            ("vn", pack_2s(10, 20)),
            ("vm", pack_2s(3, 7)),
        ]);
        assert_eq!(result.eval(&e), EvalResult::Bv(pack_2s(13, 27)));
    }

    #[test]
    fn test_neon_add_2s_wrapping() {
        let vn = SmtExpr::var("vn", 64);
        let vm = SmtExpr::var("vm", 64);
        let result = encode_neon_add(VectorArrangement::S2, &vn, &vm);

        // Wrapping: 0xFFFFFFFF + 1 = 0 per lane
        let e = env(&[
            ("vn", pack_2s(0xFFFFFFFF, 0x80000000)),
            ("vm", pack_2s(1, 0x80000000)),
        ]);
        assert_eq!(result.eval(&e), EvalResult::Bv(pack_2s(0, 0)));
    }

    #[test]
    fn test_neon_add_4h() {
        let vn = SmtExpr::var("vn", 64);
        let vm = SmtExpr::var("vm", 64);
        let result = encode_neon_add(VectorArrangement::H4, &vn, &vm);

        let e = env(&[
            ("vn", pack_4h(1, 2, 3, 4)),
            ("vm", pack_4h(10, 20, 30, 40)),
        ]);
        assert_eq!(result.eval(&e), EvalResult::Bv(pack_4h(11, 22, 33, 44)));
    }

    #[test]
    fn test_neon_add_8b() {
        let vn = SmtExpr::var("vn", 64);
        let vm = SmtExpr::var("vm", 64);
        let result = encode_neon_add(VectorArrangement::B8, &vn, &vm);

        let e = env(&[
            ("vn", pack_8b(1, 2, 3, 4, 5, 6, 7, 8)),
            ("vm", pack_8b(10, 20, 30, 40, 50, 60, 70, 80)),
        ]);
        assert_eq!(
            result.eval(&e),
            EvalResult::Bv(pack_8b(11, 22, 33, 44, 55, 66, 77, 88))
        );
    }

    #[test]
    fn test_neon_add_8b_wrapping() {
        let vn = SmtExpr::var("vn", 64);
        let vm = SmtExpr::var("vm", 64);
        let result = encode_neon_add(VectorArrangement::B8, &vn, &vm);

        // 0xFF + 1 = 0 per byte lane (wrapping)
        let e = env(&[
            ("vn", pack_8b(0xFF, 200, 0, 0, 0, 0, 0, 0)),
            ("vm", pack_8b(1, 100, 0, 0, 0, 0, 0, 0)),
        ]);
        let r = result.eval(&e);
        // Lane 0: 0xFF + 1 = 0x00 (wrapping), Lane 1: 200 + 100 = 300 & 0xFF = 44
        assert_eq!(r, EvalResult::Bv(pack_8b(0, 44, 0, 0, 0, 0, 0, 0)));
    }

    // =======================================================================
    // SUB tests
    // =======================================================================

    #[test]
    fn test_neon_sub_2s() {
        let vn = SmtExpr::var("vn", 64);
        let vm = SmtExpr::var("vm", 64);
        let result = encode_neon_sub(VectorArrangement::S2, &vn, &vm);

        let e = env(&[
            ("vn", pack_2s(100, 200)),
            ("vm", pack_2s(30, 50)),
        ]);
        assert_eq!(result.eval(&e), EvalResult::Bv(pack_2s(70, 150)));
    }

    #[test]
    fn test_neon_sub_4h() {
        let vn = SmtExpr::var("vn", 64);
        let vm = SmtExpr::var("vm", 64);
        let result = encode_neon_sub(VectorArrangement::H4, &vn, &vm);

        let e = env(&[
            ("vn", pack_4h(100, 200, 300, 400)),
            ("vm", pack_4h(10, 20, 30, 40)),
        ]);
        assert_eq!(result.eval(&e), EvalResult::Bv(pack_4h(90, 180, 270, 360)));
    }

    #[test]
    fn test_neon_sub_8b() {
        let vn = SmtExpr::var("vn", 64);
        let vm = SmtExpr::var("vm", 64);
        let result = encode_neon_sub(VectorArrangement::B8, &vn, &vm);

        let e = env(&[
            ("vn", pack_8b(50, 100, 150, 200, 10, 20, 30, 40)),
            ("vm", pack_8b(10, 20, 30, 40, 5, 10, 15, 20)),
        ]);
        assert_eq!(
            result.eval(&e),
            EvalResult::Bv(pack_8b(40, 80, 120, 160, 5, 10, 15, 20))
        );
    }

    // =======================================================================
    // MUL tests
    // =======================================================================

    #[test]
    fn test_neon_mul_2s() {
        let vn = SmtExpr::var("vn", 64);
        let vm = SmtExpr::var("vm", 64);
        let result = encode_neon_mul(VectorArrangement::S2, &vn, &vm);

        let e = env(&[
            ("vn", pack_2s(6, 7)),
            ("vm", pack_2s(7, 6)),
        ]);
        assert_eq!(result.eval(&e), EvalResult::Bv(pack_2s(42, 42)));
    }

    #[test]
    fn test_neon_mul_4h() {
        let vn = SmtExpr::var("vn", 64);
        let vm = SmtExpr::var("vm", 64);
        let result = encode_neon_mul(VectorArrangement::H4, &vn, &vm);

        let e = env(&[
            ("vn", pack_4h(2, 3, 4, 5)),
            ("vm", pack_4h(10, 10, 10, 10)),
        ]);
        assert_eq!(result.eval(&e), EvalResult::Bv(pack_4h(20, 30, 40, 50)));
    }

    #[test]
    fn test_neon_mul_8b() {
        let vn = SmtExpr::var("vn", 64);
        let vm = SmtExpr::var("vm", 64);
        let result = encode_neon_mul(VectorArrangement::B8, &vn, &vm);

        // 7 * 6 = 42, 15 * 17 = 255 (fits in u8)
        let e = env(&[
            ("vn", pack_8b(7, 15, 3, 0, 0, 0, 0, 0)),
            ("vm", pack_8b(6, 17, 5, 0, 0, 0, 0, 0)),
        ]);
        assert_eq!(
            result.eval(&e),
            EvalResult::Bv(pack_8b(42, 255, 15, 0, 0, 0, 0, 0))
        );
    }

    #[test]
    fn test_neon_mul_8b_wrapping() {
        let vn = SmtExpr::var("vn", 64);
        let vm = SmtExpr::var("vm", 64);
        let result = encode_neon_mul(VectorArrangement::B8, &vn, &vm);

        // 200 * 2 = 400, wrapping to 400 & 0xFF = 144
        let e = env(&[
            ("vn", pack_8b(200, 0, 0, 0, 0, 0, 0, 0)),
            ("vm", pack_8b(2, 0, 0, 0, 0, 0, 0, 0)),
        ]);
        assert_eq!(
            result.eval(&e),
            EvalResult::Bv(pack_8b(144, 0, 0, 0, 0, 0, 0, 0))
        );
    }

    // =======================================================================
    // NEG tests
    // =======================================================================

    #[test]
    fn test_neon_neg_2s() {
        let vn = SmtExpr::var("vn", 64);
        let result = encode_neon_neg(VectorArrangement::S2, &vn);

        // neg(1) = 0xFFFFFFFF in 32-bit, neg(0) = 0
        let e = env(&[("vn", pack_2s(1, 0))]);
        assert_eq!(result.eval(&e), EvalResult::Bv(pack_2s(0xFFFFFFFF, 0)));
    }

    #[test]
    fn test_neon_neg_4h() {
        let vn = SmtExpr::var("vn", 64);
        let result = encode_neon_neg(VectorArrangement::H4, &vn);

        // neg(1) = 0xFFFF, neg(2) = 0xFFFE, neg(0) = 0, neg(100) = 0xFF9C
        let e = env(&[("vn", pack_4h(1, 2, 0, 100))]);
        assert_eq!(
            result.eval(&e),
            EvalResult::Bv(pack_4h(0xFFFF, 0xFFFE, 0, 0xFF9C))
        );
    }

    // =======================================================================
    // Bitwise operation tests
    // =======================================================================

    #[test]
    fn test_neon_and_64bit() {
        let vn = SmtExpr::var("vn", 64);
        let vm = SmtExpr::var("vm", 64);
        let result = encode_neon_and(&vn, &vm);

        let e = env(&[
            ("vn", 0xFF00_FF00_FF00_FF00),
            ("vm", 0x0F0F_0F0F_0F0F_0F0F),
        ]);
        assert_eq!(result.eval(&e), EvalResult::Bv(0x0F00_0F00_0F00_0F00));
    }

    #[test]
    fn test_neon_orr_64bit() {
        let vn = SmtExpr::var("vn", 64);
        let vm = SmtExpr::var("vm", 64);
        let result = encode_neon_orr(&vn, &vm);

        let e = env(&[
            ("vn", 0xFF00_0000_0000_0000),
            ("vm", 0x00FF_0000_0000_0000),
        ]);
        assert_eq!(result.eval(&e), EvalResult::Bv(0xFFFF_0000_0000_0000));
    }

    #[test]
    fn test_neon_eor_64bit() {
        let vn = SmtExpr::var("vn", 64);
        let vm = SmtExpr::var("vm", 64);
        let result = encode_neon_eor(&vn, &vm);

        let e = env(&[
            ("vn", 0xAAAA_AAAA_AAAA_AAAA),
            ("vm", 0xFFFF_FFFF_FFFF_FFFF),
        ]);
        assert_eq!(result.eval(&e), EvalResult::Bv(0x5555_5555_5555_5555));
    }

    #[test]
    fn test_neon_bic_64bit() {
        let vn = SmtExpr::var("vn", 64);
        let vm = SmtExpr::var("vm", 64);
        let result = encode_neon_bic(&vn, &vm);

        // BIC = vn AND NOT(vm)
        // vn = 0xFF, vm = 0x0F => result = 0xFF AND NOT(0x0F) = 0xFF AND 0xF0 = 0xF0
        let e = env(&[
            ("vn", 0xFFFF_FFFF_FFFF_FFFF),
            ("vm", 0x0F0F_0F0F_0F0F_0F0F),
        ]);
        assert_eq!(result.eval(&e), EvalResult::Bv(0xF0F0_F0F0_F0F0_F0F0));
    }

    // =======================================================================
    // Shift tests
    // =======================================================================

    #[test]
    fn test_neon_shl_2s() {
        let vn = SmtExpr::var("vn", 64);
        let result = encode_neon_shl(VectorArrangement::S2, &vn, 4);

        // Each 32-bit lane shifted left by 4: 1 << 4 = 16, 0xFF << 4 = 0xFF0
        let e = env(&[("vn", pack_2s(1, 0xFF))]);
        assert_eq!(result.eval(&e), EvalResult::Bv(pack_2s(16, 0xFF0)));
    }

    #[test]
    fn test_neon_shl_4h() {
        let vn = SmtExpr::var("vn", 64);
        let result = encode_neon_shl(VectorArrangement::H4, &vn, 1);

        // Each 16-bit lane shifted left by 1
        let e = env(&[("vn", pack_4h(1, 2, 3, 0x7FFF))]);
        assert_eq!(result.eval(&e), EvalResult::Bv(pack_4h(2, 4, 6, 0xFFFE)));
    }

    #[test]
    fn test_neon_shl_8b() {
        let vn = SmtExpr::var("vn", 64);
        let result = encode_neon_shl(VectorArrangement::B8, &vn, 2);

        // Each 8-bit lane shifted left by 2: 1 << 2 = 4, 63 << 2 = 252
        let e = env(&[("vn", pack_8b(1, 63, 0, 0, 0, 0, 0, 0))]);
        assert_eq!(
            result.eval(&e),
            EvalResult::Bv(pack_8b(4, 252, 0, 0, 0, 0, 0, 0))
        );
    }

    #[test]
    fn test_neon_ushr_2s() {
        let vn = SmtExpr::var("vn", 64);
        let result = encode_neon_ushr(VectorArrangement::S2, &vn, 4);

        // Each 32-bit lane logical shift right by 4
        let e = env(&[("vn", pack_2s(0x100, 0xF0000000))]);
        assert_eq!(
            result.eval(&e),
            EvalResult::Bv(pack_2s(0x10, 0x0F000000))
        );
    }

    #[test]
    fn test_neon_ushr_4h() {
        let vn = SmtExpr::var("vn", 64);
        let result = encode_neon_ushr(VectorArrangement::H4, &vn, 8);

        // Each 16-bit lane logical shift right by 8
        let e = env(&[("vn", pack_4h(0xFF00, 0x1234, 0, 0))]);
        assert_eq!(result.eval(&e), EvalResult::Bv(pack_4h(0xFF, 0x12, 0, 0)));
    }

    #[test]
    fn test_neon_sshr_2s() {
        let vn = SmtExpr::var("vn", 64);
        let result = encode_neon_sshr(VectorArrangement::S2, &vn, 4);

        // Arithmetic shift right: sign bit fills.
        // 0x80000000 >> 4 = 0xF8000000 (signed), 0x10 >> 4 = 0x01
        let e = env(&[("vn", pack_2s(0x80000000, 0x10))]);
        assert_eq!(
            result.eval(&e),
            EvalResult::Bv(pack_2s(0xF8000000, 0x01))
        );
    }

    #[test]
    fn test_neon_sshr_4h() {
        let vn = SmtExpr::var("vn", 64);
        let result = encode_neon_sshr(VectorArrangement::H4, &vn, 1);

        // 0x8000 >>s 1 = 0xC000, 0x0002 >>s 1 = 0x0001
        let e = env(&[("vn", pack_4h(0x8000, 0x0002, 0, 0))]);
        assert_eq!(result.eval(&e), EvalResult::Bv(pack_4h(0xC000, 0x0001, 0, 0)));
    }

    #[test]
    fn test_neon_sshr_8b() {
        let vn = SmtExpr::var("vn", 64);
        let result = encode_neon_sshr(VectorArrangement::B8, &vn, 1);

        // 0x80 >>s 1 = 0xC0 (sign extend), 0x02 >>s 1 = 0x01
        let e = env(&[("vn", pack_8b(0x80, 0x02, 0, 0, 0, 0, 0, 0))]);
        assert_eq!(
            result.eval(&e),
            EvalResult::Bv(pack_8b(0xC0, 0x01, 0, 0, 0, 0, 0, 0))
        );
    }

    // =======================================================================
    // Cross-check: ADD then SUB = identity
    // =======================================================================

    #[test]
    fn test_add_sub_identity_2s() {
        let vn = SmtExpr::var("vn", 64);
        let vm = SmtExpr::var("vm", 64);

        // (vn + vm) - vm = vn for all lane values
        let added = encode_neon_add(VectorArrangement::S2, &vn, &vm);
        let result = encode_neon_sub(VectorArrangement::S2, &added, &vm);

        // Test with arbitrary values
        let e = env(&[
            ("vn", pack_2s(0xDEADBEEF, 0x12345678)),
            ("vm", pack_2s(0xCAFEBABE, 0x87654321)),
        ]);
        assert_eq!(result.eval(&e), EvalResult::Bv(pack_2s(0xDEADBEEF, 0x12345678)));
    }

    // =======================================================================
    // Cross-check: SHL then USHR roundtrip (for shift < lane_bits)
    // =======================================================================

    #[test]
    fn test_shl_ushr_clears_high_and_low_bits() {
        let vn = SmtExpr::var("vn", 64);

        // SHL by 4 then USHR by 4: clears both high 4 bits and low 4 bits of each lane.
        // SHL shifts out the top 4 bits, USHR zeros the new top 4 bits.
        let shifted_left = encode_neon_shl(VectorArrangement::S2, &vn, 4);
        let shifted_back = encode_neon_ushr(VectorArrangement::S2, &shifted_left, 4);

        // 0x1234ABCD: SHL 4 -> 0x234ABCD0, USHR 4 -> 0x0234ABCD
        // 0xFFFFFFFF: SHL 4 -> 0xFFFFFFF0, USHR 4 -> 0x0FFFFFFF
        let e = env(&[("vn", pack_2s(0x1234ABCD, 0xFFFFFFFF))]);
        assert_eq!(
            shifted_back.eval(&e),
            EvalResult::Bv(pack_2s(0x0234ABCD, 0x0FFFFFFF))
        );
    }
}
