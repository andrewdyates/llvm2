// llvm2-verify/neon_semantics.rs - AArch64 NEON SIMD instruction semantics
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
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
    concat_lanes, lane_extract, lane_insert,
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
    let all_ones = if width <= 64 {
        SmtExpr::bv_const(if width >= 64 { u64::MAX } else { (1u64 << width) - 1 }, width)
    } else {
        // For > 64-bit widths (e.g., 128-bit NEON), build all-ones via concat.
        let lo = SmtExpr::bv_const(u64::MAX, 64);
        let hi_width = width - 64;
        let hi = SmtExpr::bv_const(if hi_width >= 64 { u64::MAX } else { (1u64 << hi_width) - 1 }, hi_width);
        hi.concat(lo)
    };
    let not_vm = vm.clone().bvxor(all_ones);
    vn.clone().bvand(not_vm)
}

/// Encode `NOT.16B Vd, Vn` (alias for `MVN`) -- NEON bitwise NOT.
///
/// Semantics: `Vd = NOT(Vn)` (full-width bitwise inversion).
/// Implemented as `bvxor(vn, all_ones)`.
///
/// Only valid for 16B/8B arrangements (bitwise on full register).
/// Reference: ARM DDI 0487, C7.2.210 NOT (vector).
pub fn encode_neon_not(vn: &SmtExpr) -> SmtExpr {
    let width = vn.bv_width();
    let all_ones = if width <= 64 {
        SmtExpr::bv_const(if width >= 64 { u64::MAX } else { (1u64 << width) - 1 }, width)
    } else {
        let lo = SmtExpr::bv_const(u64::MAX, 64);
        let hi_width = width - 64;
        let hi = SmtExpr::bv_const(
            if hi_width >= 64 { u64::MAX } else { (1u64 << hi_width) - 1 },
            hi_width,
        );
        hi.concat(lo)
    };
    vn.clone().bvxor(all_ones)
}

/// Encode `DUP Vd.4S, Wn` -- NEON broadcast scalar to all lanes.
///
/// Semantics: for each lane `i`: `Vd[i] = Wn` (zero-extended to lane width).
/// The scalar value is replicated across all lanes.
///
/// Reference: ARM DDI 0487, C7.2.59 DUP (general).
pub fn encode_neon_dup(
    arrangement: VectorArrangement,
    scalar: &SmtExpr,
) -> SmtExpr {
    let n = arrangement.lane_count();
    let lane_bits = arrangement.lane_bits();
    let lane_val = if scalar.bv_width() > lane_bits {
        scalar.clone().extract(lane_bits - 1, 0)
    } else if scalar.bv_width() < lane_bits {
        // Zero-extend
        let pad = SmtExpr::bv_const(0, lane_bits - scalar.bv_width());
        pad.concat(scalar.clone())
    } else {
        scalar.clone()
    };
    let lanes: Vec<SmtExpr> = (0..n).map(|_| lane_val.clone()).collect();
    concat_lanes(&lanes, arrangement)
}

/// Encode `INS Vd.T[idx], Vn.T[0]` -- NEON lane insert.
///
/// Semantics: only lane `idx` of `Vd` is modified; other lanes unchanged.
/// The inserted value comes from `new_lane_val`.
///
/// This is a thin wrapper around `lane_insert()` for documentation.
///
/// Reference: ARM DDI 0487, C7.2.106 INS (element).
pub fn encode_neon_ins(
    vec: &SmtExpr,
    arrangement: VectorArrangement,
    idx: u32,
    new_lane_val: SmtExpr,
) -> SmtExpr {
    lane_insert(vec, arrangement, idx, new_lane_val)
}

/// Encode `CMEQ.<T> Vd, Vn, Vm` -- NEON per-lane equality comparison.
///
/// Semantics: for each lane `i`:
///   `Vd[i] = if Vn[i] == Vm[i] then all_ones else 0`
///
/// The result mask is all-ones (0xFF...F) for matching lanes and all-zeros
/// for non-matching lanes. This is the standard NEON compare behavior.
///
/// Reference: ARM DDI 0487, C7.2.29 CMEQ (register).
pub fn encode_neon_cmeq(
    arrangement: VectorArrangement,
    vn: &SmtExpr,
    vm: &SmtExpr,
) -> SmtExpr {
    let lane_bits = arrangement.lane_bits();
    let all_ones_lane = SmtExpr::bv_const(
        if lane_bits >= 64 { u64::MAX } else { (1u64 << lane_bits) - 1 },
        lane_bits,
    );
    let zero_lane = SmtExpr::bv_const(0, lane_bits);
    map_lanes_binary(vn, vm, arrangement, |a, b| {
        SmtExpr::ite(a.eq_expr(b), all_ones_lane.clone(), zero_lane.clone())
    })
}

/// Encode `CMGT.<T> Vd, Vn, Vm` -- NEON per-lane signed greater-than comparison.
///
/// Semantics: for each lane `i`:
///   `Vd[i] = if Vn[i] >s Vm[i] then all_ones else 0`
///
/// The result mask is all-ones (0xFF...F) for matching lanes and all-zeros
/// for non-matching lanes. This is the standard NEON compare behavior.
///
/// Reference: ARM DDI 0487, C7.2.31 CMGT (register).
pub fn encode_neon_cmgt(
    arrangement: VectorArrangement,
    vn: &SmtExpr,
    vm: &SmtExpr,
) -> SmtExpr {
    let lane_bits = arrangement.lane_bits();
    let all_ones_lane = SmtExpr::bv_const(
        if lane_bits >= 64 { u64::MAX } else { (1u64 << lane_bits) - 1 },
        lane_bits,
    );
    let zero_lane = SmtExpr::bv_const(0, lane_bits);
    map_lanes_binary(vn, vm, arrangement, |a, b| {
        SmtExpr::ite(a.bvsgt(b), all_ones_lane.clone(), zero_lane.clone())
    })
}

/// Encode `CMGE.<T> Vd, Vn, Vm` -- NEON per-lane signed greater-or-equal comparison.
///
/// Semantics: for each lane `i`:
///   `Vd[i] = if Vn[i] >=s Vm[i] then all_ones else 0`
///
/// The result mask is all-ones (0xFF...F) for matching lanes and all-zeros
/// for non-matching lanes. This is the standard NEON compare behavior.
///
/// Reference: ARM DDI 0487, C7.2.30 CMGE (register).
pub fn encode_neon_cmge(
    arrangement: VectorArrangement,
    vn: &SmtExpr,
    vm: &SmtExpr,
) -> SmtExpr {
    let lane_bits = arrangement.lane_bits();
    let all_ones_lane = SmtExpr::bv_const(
        if lane_bits >= 64 { u64::MAX } else { (1u64 << lane_bits) - 1 },
        lane_bits,
    );
    let zero_lane = SmtExpr::bv_const(0, lane_bits);
    map_lanes_binary(vn, vm, arrangement, |a, b| {
        SmtExpr::ite(a.bvsge(b), all_ones_lane.clone(), zero_lane.clone())
    })
}

// ---------------------------------------------------------------------------
// NEON integer min/max operations
// ---------------------------------------------------------------------------

/// Encode `SMIN.<T> Vd, Vn, Vm` -- NEON vector signed minimum.
///
/// Semantics: for each lane `i`: `Vd[i] = if Vn[i] <s Vm[i] then Vn[i] else Vm[i]`.
///
/// Valid arrangements: 8B, 16B, 4H, 8H, 2S, 4S.
/// Note: no 2D (64-bit lane) signed integer minimum in AArch64 NEON.
/// Reference: ARM DDI 0487, C7.2.277 SMIN (vector).
pub fn encode_neon_smin(
    arrangement: VectorArrangement,
    vn: &SmtExpr,
    vm: &SmtExpr,
) -> SmtExpr {
    debug_assert!(
        arrangement != VectorArrangement::D2,
        "NEON SMIN does not support 2D arrangement"
    );
    map_lanes_binary(vn, vm, arrangement, |a, b| {
        SmtExpr::ite(a.clone().bvslt(b.clone()), a, b)
    })
}

/// Encode `UMIN.<T> Vd, Vn, Vm` -- NEON vector unsigned minimum.
///
/// Semantics: for each lane `i`: `Vd[i] = if Vn[i] <u Vm[i] then Vn[i] else Vm[i]`.
///
/// Valid arrangements: 8B, 16B, 4H, 8H, 2S, 4S.
/// Note: no 2D (64-bit lane) unsigned integer minimum in AArch64 NEON.
/// Reference: ARM DDI 0487, C7.2.378 UMIN (vector).
pub fn encode_neon_umin(
    arrangement: VectorArrangement,
    vn: &SmtExpr,
    vm: &SmtExpr,
) -> SmtExpr {
    debug_assert!(
        arrangement != VectorArrangement::D2,
        "NEON UMIN does not support 2D arrangement"
    );
    map_lanes_binary(vn, vm, arrangement, |a, b| {
        SmtExpr::ite(a.clone().bvult(b.clone()), a, b)
    })
}

/// Encode `SMAX.<T> Vd, Vn, Vm` -- NEON vector signed maximum.
///
/// Semantics: for each lane `i`: `Vd[i] = if Vn[i] >s Vm[i] then Vn[i] else Vm[i]`.
///
/// Valid arrangements: 8B, 16B, 4H, 8H, 2S, 4S.
/// Note: no 2D (64-bit lane) signed integer maximum in AArch64 NEON.
/// Reference: ARM DDI 0487, C7.2.274 SMAX (vector).
pub fn encode_neon_smax(
    arrangement: VectorArrangement,
    vn: &SmtExpr,
    vm: &SmtExpr,
) -> SmtExpr {
    debug_assert!(
        arrangement != VectorArrangement::D2,
        "NEON SMAX does not support 2D arrangement"
    );
    map_lanes_binary(vn, vm, arrangement, |a, b| {
        SmtExpr::ite(a.clone().bvsgt(b.clone()), a, b)
    })
}

/// Encode `UMAX.<T> Vd, Vn, Vm` -- NEON vector unsigned maximum.
///
/// Semantics: for each lane `i`: `Vd[i] = if Vn[i] >u Vm[i] then Vn[i] else Vm[i]`.
///
/// Valid arrangements: 8B, 16B, 4H, 8H, 2S, 4S.
/// Note: no 2D (64-bit lane) unsigned integer maximum in AArch64 NEON.
/// Reference: ARM DDI 0487, C7.2.375 UMAX (vector).
pub fn encode_neon_umax(
    arrangement: VectorArrangement,
    vn: &SmtExpr,
    vm: &SmtExpr,
) -> SmtExpr {
    debug_assert!(
        arrangement != VectorArrangement::D2,
        "NEON UMAX does not support 2D arrangement"
    );
    map_lanes_binary(vn, vm, arrangement, |a, b| {
        SmtExpr::ite(a.clone().bvugt(b.clone()), a, b)
    })
}

// ---------------------------------------------------------------------------
// NEON multiply-accumulate
// ---------------------------------------------------------------------------

/// Encode `MLA.<T> Vd, Vn, Vm` -- NEON vector multiply-accumulate.
///
/// Semantics: for each lane `i`: `Vd[i] = Va[i] + Vn[i] * Vm[i]` (wrapping).
///
/// Valid arrangements: 8B, 16B, 4H, 8H, 2S, 4S.
/// Note: no 2D (64-bit lane) integer multiply-accumulate in AArch64 NEON.
/// Reference: ARM DDI 0487, C7.2.200 MLA (vector).
pub fn encode_neon_mla(
    arrangement: VectorArrangement,
    va: &SmtExpr,
    vn: &SmtExpr,
    vm: &SmtExpr,
) -> SmtExpr {
    debug_assert!(
        arrangement != VectorArrangement::D2,
        "NEON MLA does not support 2D arrangement"
    );
    let lane_count = arrangement.lane_count();
    let lanes: Vec<SmtExpr> = (0..lane_count)
        .map(|i| {
            let a = lane_extract(va, arrangement, i);
            let n = lane_extract(vn, arrangement, i);
            let m = lane_extract(vm, arrangement, i);
            a.bvadd(n.bvmul(m))
        })
        .collect();
    concat_lanes(&lanes, arrangement)
}

/// Encode `MOVI Vd.16B, #imm` -- NEON move immediate (byte broadcast).
///
/// Semantics: every byte of `Vd` is set to `imm` (8-bit immediate).
/// For a 128-bit register, this means all 16 bytes are identical.
/// For a 64-bit register, all 8 bytes are identical.
///
/// Reference: ARM DDI 0487, C7.2.206 MOVI.
pub fn encode_neon_movi(width: u32, imm: u8) -> SmtExpr {
    let byte_count = width / 8;
    let byte_val = SmtExpr::bv_const(imm as u64, 8);
    let mut result = byte_val.clone();
    for _ in 1..byte_count {
        result = byte_val.clone().concat(result);
    }
    result
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
    //
    // 64-bit vectors (S2, H4, B8) pack into a single u64 and can be tested
    // directly via `EvalResult::Bv(u64)`.
    //
    // 128-bit vectors (S4, H8, B16, D2) exceed the u64 evaluator range.
    // We test them by extracting individual lanes from the result expression
    // before evaluation -- each lane fits in u64. The helper `assert_lane`
    // encapsulates this pattern.
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

    // -----------------------------------------------------------------------
    // 128-bit test helpers
    //
    // Since EvalResult::Bv stores u64, we cannot evaluate full 128-bit vectors
    // directly. Instead, we extract each lane from the result *symbolically*
    // (producing a sub-64-bit expression), then evaluate that lane.
    //
    // This is architecturally faithful: NEON ops are defined per-lane, so
    // verifying each lane independently is a valid correctness strategy.
    // -----------------------------------------------------------------------

    use crate::smt::lane_extract;

    /// Assert that a specific lane of a 128-bit result expression evaluates
    /// to the expected value.
    fn assert_lane(
        result_expr: &SmtExpr,
        arrangement: VectorArrangement,
        lane_idx: u32,
        expected: u64,
        env: &HashMap<String, u64>,
    ) {
        let lane_expr = lane_extract(result_expr, arrangement, lane_idx);
        let actual = lane_expr.eval(env);
        assert_eq!(
            actual,
            EvalResult::Bv(expected),
            "lane {} mismatch: expected 0x{:X}, got {:?}",
            lane_idx, expected, actual
        );
    }

    /// Assert all lanes of a 128-bit result match expected values.
    fn assert_all_lanes(
        result_expr: &SmtExpr,
        arrangement: VectorArrangement,
        expected_lanes: &[u64],
        env: &HashMap<String, u64>,
    ) {
        assert_eq!(
            expected_lanes.len() as u32,
            arrangement.lane_count(),
            "wrong number of expected lanes"
        );
        for (i, &expected) in expected_lanes.iter().enumerate() {
            assert_lane(result_expr, arrangement, i as u32, expected, env);
        }
    }

    /// Build a 128-bit vector from two 64-bit halves (lo, hi).
    ///
    /// The input variables are named `{prefix}_lo` (bits [63:0]) and
    /// `{prefix}_hi` (bits [127:64]). Returns the concatenated expression
    /// and the environment entries to set.
    fn var_128(prefix: &str) -> SmtExpr {
        let lo = SmtExpr::var(format!("{}_lo", prefix), 64);
        let hi = SmtExpr::var(format!("{}_hi", prefix), 64);
        hi.concat(lo)
    }

    /// Insert both halves of a 128-bit variable into the environment.
    fn set_128(env: &mut HashMap<String, u64>, prefix: &str, lo: u64, hi: u64) {
        env.insert(format!("{}_lo", prefix), lo);
        env.insert(format!("{}_hi", prefix), hi);
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

    // =======================================================================
    // 128-bit ADD tests (16B, 8H, 4S, 2D)
    // =======================================================================

    #[test]
    fn test_neon_add_4s() {
        let vn = var_128("vn");
        let vm = var_128("vm");
        let result = encode_neon_add(VectorArrangement::S4, &vn, &vm);

        let mut e = HashMap::new();
        // vn = [10, 20, 30, 40] as 4x32-bit
        // lo 64 bits: lane0=10, lane1=20 => pack_2s(10, 20)
        // hi 64 bits: lane2=30, lane3=40 => pack_2s(30, 40)
        set_128(&mut e, "vn", pack_2s(10, 20), pack_2s(30, 40));
        // vm = [3, 7, 11, 13]
        set_128(&mut e, "vm", pack_2s(3, 7), pack_2s(11, 13));

        assert_all_lanes(&result, VectorArrangement::S4, &[13, 27, 41, 53], &e);
    }

    #[test]
    fn test_neon_add_4s_wrapping() {
        let vn = var_128("vn");
        let vm = var_128("vm");
        let result = encode_neon_add(VectorArrangement::S4, &vn, &vm);

        let mut e = HashMap::new();
        set_128(&mut e, "vn", pack_2s(0xFFFFFFFF, 0x80000000), pack_2s(1, 100));
        set_128(&mut e, "vm", pack_2s(1, 0x80000000), pack_2s(0xFFFFFFFF, 200));

        // Lane 0: 0xFFFFFFFF + 1 = 0 (wrapping)
        // Lane 1: 0x80000000 + 0x80000000 = 0 (wrapping)
        // Lane 2: 1 + 0xFFFFFFFF = 0 (wrapping)
        // Lane 3: 100 + 200 = 300
        assert_all_lanes(&result, VectorArrangement::S4, &[0, 0, 0, 300], &e);
    }

    #[test]
    fn test_neon_add_8h() {
        let vn = var_128("vn");
        let vm = var_128("vm");
        let result = encode_neon_add(VectorArrangement::H8, &vn, &vm);

        let mut e = HashMap::new();
        // vn: lanes [1,2,3,4, 5,6,7,8] as 8x16-bit
        set_128(&mut e, "vn", pack_4h(1, 2, 3, 4), pack_4h(5, 6, 7, 8));
        // vm: lanes [10,20,30,40, 50,60,70,80]
        set_128(&mut e, "vm", pack_4h(10, 20, 30, 40), pack_4h(50, 60, 70, 80));

        assert_all_lanes(
            &result,
            VectorArrangement::H8,
            &[11, 22, 33, 44, 55, 66, 77, 88],
            &e,
        );
    }

    #[test]
    fn test_neon_add_16b() {
        let vn = var_128("vn");
        let vm = var_128("vm");
        let result = encode_neon_add(VectorArrangement::B16, &vn, &vm);

        let mut e = HashMap::new();
        set_128(
            &mut e,
            "vn",
            pack_8b(1, 2, 3, 4, 5, 6, 7, 8),
            pack_8b(9, 10, 11, 12, 13, 14, 15, 16),
        );
        set_128(
            &mut e,
            "vm",
            pack_8b(10, 20, 30, 40, 50, 60, 70, 80),
            pack_8b(90, 100, 110, 120, 130, 140, 150, 160),
        );

        assert_all_lanes(
            &result,
            VectorArrangement::B16,
            &[11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176],
            &e,
        );
    }

    #[test]
    fn test_neon_add_16b_wrapping() {
        let vn = var_128("vn");
        let vm = var_128("vm");
        let result = encode_neon_add(VectorArrangement::B16, &vn, &vm);

        let mut e = HashMap::new();
        set_128(
            &mut e,
            "vn",
            pack_8b(0xFF, 200, 0, 0, 0, 0, 0, 0),
            pack_8b(0xFF, 0, 0, 0, 0, 0, 0, 0),
        );
        set_128(
            &mut e,
            "vm",
            pack_8b(1, 100, 0, 0, 0, 0, 0, 0),
            pack_8b(2, 0, 0, 0, 0, 0, 0, 0),
        );

        // Lane 0: 0xFF + 1 = 0x00 (wrapping)
        // Lane 1: 200 + 100 = 300 & 0xFF = 44
        // Lane 8: 0xFF + 2 = 0x01 (wrapping)
        assert_lane(&result, VectorArrangement::B16, 0, 0x00, &e);
        assert_lane(&result, VectorArrangement::B16, 1, 44, &e);
        assert_lane(&result, VectorArrangement::B16, 8, 0x01, &e);
    }

    #[test]
    fn test_neon_add_2d() {
        let vn = var_128("vn");
        let vm = var_128("vm");
        let result = encode_neon_add(VectorArrangement::D2, &vn, &vm);

        let mut e = HashMap::new();
        // 2D: lane0 = bits[63:0] = lo, lane1 = bits[127:64] = hi
        set_128(&mut e, "vn", 100, 200);
        set_128(&mut e, "vm", 30, 50);

        assert_all_lanes(&result, VectorArrangement::D2, &[130, 250], &e);
    }

    // =======================================================================
    // 128-bit SUB tests
    // =======================================================================

    #[test]
    fn test_neon_sub_4s() {
        let vn = var_128("vn");
        let vm = var_128("vm");
        let result = encode_neon_sub(VectorArrangement::S4, &vn, &vm);

        let mut e = HashMap::new();
        set_128(&mut e, "vn", pack_2s(100, 200), pack_2s(300, 400));
        set_128(&mut e, "vm", pack_2s(10, 20), pack_2s(30, 40));

        assert_all_lanes(&result, VectorArrangement::S4, &[90, 180, 270, 360], &e);
    }

    #[test]
    fn test_neon_sub_8h() {
        let vn = var_128("vn");
        let vm = var_128("vm");
        let result = encode_neon_sub(VectorArrangement::H8, &vn, &vm);

        let mut e = HashMap::new();
        set_128(
            &mut e,
            "vn",
            pack_4h(100, 200, 300, 400),
            pack_4h(500, 600, 700, 800),
        );
        set_128(
            &mut e,
            "vm",
            pack_4h(10, 20, 30, 40),
            pack_4h(50, 60, 70, 80),
        );

        assert_all_lanes(
            &result,
            VectorArrangement::H8,
            &[90, 180, 270, 360, 450, 540, 630, 720],
            &e,
        );
    }

    #[test]
    fn test_neon_sub_16b() {
        let vn = var_128("vn");
        let vm = var_128("vm");
        let result = encode_neon_sub(VectorArrangement::B16, &vn, &vm);

        let mut e = HashMap::new();
        set_128(
            &mut e,
            "vn",
            pack_8b(50, 100, 150, 200, 10, 20, 30, 40),
            pack_8b(50, 100, 150, 200, 10, 20, 30, 40),
        );
        set_128(
            &mut e,
            "vm",
            pack_8b(10, 20, 30, 40, 5, 10, 15, 20),
            pack_8b(10, 20, 30, 40, 5, 10, 15, 20),
        );

        assert_all_lanes(
            &result,
            VectorArrangement::B16,
            &[40, 80, 120, 160, 5, 10, 15, 20, 40, 80, 120, 160, 5, 10, 15, 20],
            &e,
        );
    }

    #[test]
    fn test_neon_sub_2d() {
        let vn = var_128("vn");
        let vm = var_128("vm");
        let result = encode_neon_sub(VectorArrangement::D2, &vn, &vm);

        let mut e = HashMap::new();
        set_128(&mut e, "vn", 1000, 2000);
        set_128(&mut e, "vm", 300, 500);

        assert_all_lanes(&result, VectorArrangement::D2, &[700, 1500], &e);
    }

    // =======================================================================
    // 128-bit MUL tests (no D2 -- MUL does not support 2D)
    // =======================================================================

    #[test]
    fn test_neon_mul_4s() {
        let vn = var_128("vn");
        let vm = var_128("vm");
        let result = encode_neon_mul(VectorArrangement::S4, &vn, &vm);

        let mut e = HashMap::new();
        set_128(&mut e, "vn", pack_2s(6, 7), pack_2s(8, 9));
        set_128(&mut e, "vm", pack_2s(7, 6), pack_2s(5, 4));

        assert_all_lanes(&result, VectorArrangement::S4, &[42, 42, 40, 36], &e);
    }

    #[test]
    fn test_neon_mul_8h() {
        let vn = var_128("vn");
        let vm = var_128("vm");
        let result = encode_neon_mul(VectorArrangement::H8, &vn, &vm);

        let mut e = HashMap::new();
        set_128(&mut e, "vn", pack_4h(2, 3, 4, 5), pack_4h(6, 7, 8, 9));
        set_128(&mut e, "vm", pack_4h(10, 10, 10, 10), pack_4h(10, 10, 10, 10));

        assert_all_lanes(
            &result,
            VectorArrangement::H8,
            &[20, 30, 40, 50, 60, 70, 80, 90],
            &e,
        );
    }

    #[test]
    fn test_neon_mul_16b() {
        let vn = var_128("vn");
        let vm = var_128("vm");
        let result = encode_neon_mul(VectorArrangement::B16, &vn, &vm);

        let mut e = HashMap::new();
        set_128(
            &mut e,
            "vn",
            pack_8b(7, 15, 3, 2, 0, 0, 0, 0),
            pack_8b(5, 10, 4, 0, 0, 0, 0, 0),
        );
        set_128(
            &mut e,
            "vm",
            pack_8b(6, 17, 5, 3, 0, 0, 0, 0),
            pack_8b(8, 10, 3, 0, 0, 0, 0, 0),
        );

        // lane 0: 7*6=42, lane 1: 15*17=255, lane 2: 3*5=15, lane 3: 2*3=6
        // lane 8: 5*8=40, lane 9: 10*10=100, lane 10: 4*3=12, lane 11: 0*0=0
        assert_lane(&result, VectorArrangement::B16, 0, 42, &e);
        assert_lane(&result, VectorArrangement::B16, 1, 255, &e);
        assert_lane(&result, VectorArrangement::B16, 2, 15, &e);
        assert_lane(&result, VectorArrangement::B16, 3, 6, &e);
        assert_lane(&result, VectorArrangement::B16, 8, 40, &e);
        assert_lane(&result, VectorArrangement::B16, 9, 100, &e);
        assert_lane(&result, VectorArrangement::B16, 10, 12, &e);
    }

    #[test]
    fn test_neon_mul_4s_wrapping() {
        let vn = var_128("vn");
        let vm = var_128("vm");
        let result = encode_neon_mul(VectorArrangement::S4, &vn, &vm);

        let mut e = HashMap::new();
        // 0x80000000 * 2 = 0x100000000 => wraps to 0x00000000
        set_128(&mut e, "vn", pack_2s(0x80000000, 1), pack_2s(100, 0));
        set_128(&mut e, "vm", pack_2s(2, 1), pack_2s(200, 0));

        assert_lane(&result, VectorArrangement::S4, 0, 0, &e);
        assert_lane(&result, VectorArrangement::S4, 1, 1, &e);
        assert_lane(&result, VectorArrangement::S4, 2, 20000, &e);
        assert_lane(&result, VectorArrangement::S4, 3, 0, &e);
    }

    // =======================================================================
    // 128-bit NEG tests
    // =======================================================================

    #[test]
    fn test_neon_neg_4s() {
        let vn = var_128("vn");
        let result = encode_neon_neg(VectorArrangement::S4, &vn);

        let mut e = HashMap::new();
        set_128(&mut e, "vn", pack_2s(1, 0), pack_2s(100, 0xFFFFFFFF));

        // neg(1) = 0xFFFFFFFF, neg(0) = 0, neg(100) = 0xFFFFFF9C, neg(0xFFFFFFFF) = 1
        assert_all_lanes(
            &result,
            VectorArrangement::S4,
            &[0xFFFFFFFF, 0, 0xFFFFFF9C, 1],
            &e,
        );
    }

    #[test]
    fn test_neon_neg_8h() {
        let vn = var_128("vn");
        let result = encode_neon_neg(VectorArrangement::H8, &vn);

        let mut e = HashMap::new();
        set_128(&mut e, "vn", pack_4h(1, 2, 0, 100), pack_4h(0xFFFF, 0x8000, 3, 50));

        assert_all_lanes(
            &result,
            VectorArrangement::H8,
            &[0xFFFF, 0xFFFE, 0, 0xFF9C, 1, 0x8000, 0xFFFD, 0xFFCE],
            &e,
        );
    }

    #[test]
    fn test_neon_neg_16b() {
        let vn = var_128("vn");
        let result = encode_neon_neg(VectorArrangement::B16, &vn);

        let mut e = HashMap::new();
        set_128(
            &mut e,
            "vn",
            pack_8b(1, 0, 0xFF, 0x80, 0, 0, 0, 0),
            pack_8b(2, 0, 0xFE, 0x7F, 0, 0, 0, 0),
        );

        // lane 0: neg(1) = 0xFF
        // lane 1: neg(0) = 0
        // lane 2: neg(0xFF) = 1
        // lane 3: neg(0x80) = 0x80
        // lane 8: neg(2) = 0xFE
        // lane 10: neg(0xFE) = 2
        // lane 11: neg(0x7F) = 0x81
        assert_lane(&result, VectorArrangement::B16, 0, 0xFF, &e);
        assert_lane(&result, VectorArrangement::B16, 1, 0, &e);
        assert_lane(&result, VectorArrangement::B16, 2, 1, &e);
        assert_lane(&result, VectorArrangement::B16, 3, 0x80, &e);
        assert_lane(&result, VectorArrangement::B16, 8, 0xFE, &e);
        assert_lane(&result, VectorArrangement::B16, 10, 2, &e);
        assert_lane(&result, VectorArrangement::B16, 11, 0x81, &e);
    }

    #[test]
    fn test_neon_neg_2d() {
        let vn = var_128("vn");
        let result = encode_neon_neg(VectorArrangement::D2, &vn);

        let mut e = HashMap::new();
        set_128(&mut e, "vn", 1, 0);

        // neg(1) in 64-bit = 0xFFFFFFFF_FFFFFFFF
        // neg(0) = 0
        assert_lane(&result, VectorArrangement::D2, 0, 0xFFFFFFFF_FFFFFFFF, &e);
        assert_lane(&result, VectorArrangement::D2, 1, 0, &e);
    }

    // =======================================================================
    // 128-bit bitwise operation tests (AND, ORR, EOR, BIC)
    //
    // Bitwise ops are width-agnostic: they operate on whatever width the
    // input SmtExpr has. For 128-bit, we construct 128-bit inputs and
    // verify the result lane-by-lane.
    // =======================================================================

    #[test]
    fn test_neon_and_128bit() {
        let vn = var_128("vn");
        let vm = var_128("vm");
        let result = encode_neon_and(&vn, &vm);

        let mut e = HashMap::new();
        set_128(&mut e, "vn", 0xFF00_FF00_FF00_FF00, 0xAAAA_AAAA_AAAA_AAAA);
        set_128(&mut e, "vm", 0x0F0F_0F0F_0F0F_0F0F, 0xFFFF_0000_FFFF_0000);

        // Verify via D2 lane extraction (64-bit halves)
        // lo half: 0xFF00_FF00_FF00_FF00 AND 0x0F0F_0F0F_0F0F_0F0F = 0x0F00_0F00_0F00_0F00
        // hi half: 0xAAAA_AAAA_AAAA_AAAA AND 0xFFFF_0000_FFFF_0000 = 0xAAAA_0000_AAAA_0000
        assert_lane(&result, VectorArrangement::D2, 0, 0x0F00_0F00_0F00_0F00, &e);
        assert_lane(&result, VectorArrangement::D2, 1, 0xAAAA_0000_AAAA_0000, &e);
    }

    #[test]
    fn test_neon_orr_128bit() {
        let vn = var_128("vn");
        let vm = var_128("vm");
        let result = encode_neon_orr(&vn, &vm);

        let mut e = HashMap::new();
        set_128(&mut e, "vn", 0xFF00_0000_0000_0000, 0x0000_0000_0000_00FF);
        set_128(&mut e, "vm", 0x00FF_0000_0000_0000, 0x0000_0000_0000_FF00);

        assert_lane(&result, VectorArrangement::D2, 0, 0xFFFF_0000_0000_0000, &e);
        assert_lane(&result, VectorArrangement::D2, 1, 0x0000_0000_0000_FFFF, &e);
    }

    #[test]
    fn test_neon_eor_128bit() {
        let vn = var_128("vn");
        let vm = var_128("vm");
        let result = encode_neon_eor(&vn, &vm);

        let mut e = HashMap::new();
        set_128(&mut e, "vn", 0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555);
        set_128(&mut e, "vm", 0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF);

        assert_lane(&result, VectorArrangement::D2, 0, 0x5555_5555_5555_5555, &e);
        assert_lane(&result, VectorArrangement::D2, 1, 0xAAAA_AAAA_AAAA_AAAA, &e);
    }

    #[test]
    fn test_neon_bic_128bit() {
        let vn = var_128("vn");
        let vm = var_128("vm");
        let result = encode_neon_bic(&vn, &vm);

        let mut e = HashMap::new();
        set_128(&mut e, "vn", 0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF);
        set_128(&mut e, "vm", 0x0F0F_0F0F_0F0F_0F0F, 0xF0F0_F0F0_F0F0_F0F0);

        // BIC = vn AND NOT(vm)
        assert_lane(&result, VectorArrangement::D2, 0, 0xF0F0_F0F0_F0F0_F0F0, &e);
        assert_lane(&result, VectorArrangement::D2, 1, 0x0F0F_0F0F_0F0F_0F0F, &e);
    }

    // =======================================================================
    // 128-bit shift tests (SHL, USHR, SSHR)
    // =======================================================================

    #[test]
    fn test_neon_shl_4s() {
        let vn = var_128("vn");
        let result = encode_neon_shl(VectorArrangement::S4, &vn, 4);

        let mut e = HashMap::new();
        set_128(&mut e, "vn", pack_2s(1, 0xFF), pack_2s(0x1000, 0x80000000));

        // 1 << 4 = 16, 0xFF << 4 = 0xFF0
        // 0x1000 << 4 = 0x10000, 0x80000000 << 4 = 0 (wrapping 32-bit)
        assert_all_lanes(
            &result,
            VectorArrangement::S4,
            &[16, 0xFF0, 0x10000, 0],
            &e,
        );
    }

    #[test]
    fn test_neon_shl_8h() {
        let vn = var_128("vn");
        let result = encode_neon_shl(VectorArrangement::H8, &vn, 1);

        let mut e = HashMap::new();
        set_128(
            &mut e,
            "vn",
            pack_4h(1, 2, 3, 0x7FFF),
            pack_4h(4, 5, 0x8000, 0xFFFF),
        );

        // Each 16-bit lane << 1
        assert_all_lanes(
            &result,
            VectorArrangement::H8,
            &[2, 4, 6, 0xFFFE, 8, 10, 0, 0xFFFE],
            &e,
        );
    }

    #[test]
    fn test_neon_shl_16b() {
        let vn = var_128("vn");
        let result = encode_neon_shl(VectorArrangement::B16, &vn, 2);

        let mut e = HashMap::new();
        set_128(
            &mut e,
            "vn",
            pack_8b(1, 63, 0, 0, 0, 0, 0, 0),
            pack_8b(10, 0x40, 0, 0, 0, 0, 0, 0),
        );

        // 1 << 2 = 4, 63 << 2 = 252, 10 << 2 = 40, 0x40 << 2 = 0x00 (wrapping 8-bit)
        assert_lane(&result, VectorArrangement::B16, 0, 4, &e);
        assert_lane(&result, VectorArrangement::B16, 1, 252, &e);
        assert_lane(&result, VectorArrangement::B16, 8, 40, &e);
        assert_lane(&result, VectorArrangement::B16, 9, 0, &e);
    }

    #[test]
    fn test_neon_shl_2d() {
        let vn = var_128("vn");
        let result = encode_neon_shl(VectorArrangement::D2, &vn, 8);

        let mut e = HashMap::new();
        set_128(&mut e, "vn", 0xFF, 0x0100_0000_0000_0000);

        // 0xFF << 8 = 0xFF00
        // 0x0100_0000_0000_0000 << 8 = 0 (bit 56 shifted to bit 64, lost in 64-bit)
        assert_lane(&result, VectorArrangement::D2, 0, 0xFF00, &e);
        assert_lane(&result, VectorArrangement::D2, 1, 0, &e);
    }

    #[test]
    fn test_neon_ushr_4s() {
        let vn = var_128("vn");
        let result = encode_neon_ushr(VectorArrangement::S4, &vn, 4);

        let mut e = HashMap::new();
        set_128(&mut e, "vn", pack_2s(0x100, 0xF0000000), pack_2s(0xFFFFFFFF, 16));

        assert_all_lanes(
            &result,
            VectorArrangement::S4,
            &[0x10, 0x0F000000, 0x0FFFFFFF, 1],
            &e,
        );
    }

    #[test]
    fn test_neon_ushr_8h() {
        let vn = var_128("vn");
        let result = encode_neon_ushr(VectorArrangement::H8, &vn, 8);

        let mut e = HashMap::new();
        set_128(
            &mut e,
            "vn",
            pack_4h(0xFF00, 0x1234, 0, 0),
            pack_4h(0xABCD, 0x00FF, 0, 0),
        );

        assert_lane(&result, VectorArrangement::H8, 0, 0xFF, &e);
        assert_lane(&result, VectorArrangement::H8, 1, 0x12, &e);
        assert_lane(&result, VectorArrangement::H8, 4, 0xAB, &e);
        assert_lane(&result, VectorArrangement::H8, 5, 0x00, &e);
    }

    #[test]
    fn test_neon_ushr_2d() {
        let vn = var_128("vn");
        let result = encode_neon_ushr(VectorArrangement::D2, &vn, 32);

        let mut e = HashMap::new();
        set_128(&mut e, "vn", 0xDEADBEEF_12345678, 0x00000001_00000000);

        // 0xDEADBEEF_12345678 >> 32 = 0xDEADBEEF
        // 0x00000001_00000000 >> 32 = 1
        assert_lane(&result, VectorArrangement::D2, 0, 0xDEADBEEF, &e);
        assert_lane(&result, VectorArrangement::D2, 1, 1, &e);
    }

    #[test]
    fn test_neon_sshr_4s() {
        let vn = var_128("vn");
        let result = encode_neon_sshr(VectorArrangement::S4, &vn, 4);

        let mut e = HashMap::new();
        set_128(&mut e, "vn", pack_2s(0x80000000, 0x10), pack_2s(0xF0000000, 0x7FFFFFFF));

        // 0x80000000 >>s 4 = 0xF8000000 (sign fills)
        // 0x10 >>s 4 = 0x01
        // 0xF0000000 >>s 4 = 0xFF000000 (sign fills)
        // 0x7FFFFFFF >>s 4 = 0x07FFFFFF (positive, zero fills)
        assert_all_lanes(
            &result,
            VectorArrangement::S4,
            &[0xF8000000, 0x01, 0xFF000000, 0x07FFFFFF],
            &e,
        );
    }

    #[test]
    fn test_neon_sshr_8h() {
        let vn = var_128("vn");
        let result = encode_neon_sshr(VectorArrangement::H8, &vn, 1);

        let mut e = HashMap::new();
        set_128(
            &mut e,
            "vn",
            pack_4h(0x8000, 0x0002, 0, 0),
            pack_4h(0xFFFF, 0x7FFF, 0, 0),
        );

        // 0x8000 >>s 1 = 0xC000
        // 0x0002 >>s 1 = 0x0001
        // 0xFFFF >>s 1 = 0xFFFF (-1 >>s 1 = -1)
        // 0x7FFF >>s 1 = 0x3FFF
        assert_lane(&result, VectorArrangement::H8, 0, 0xC000, &e);
        assert_lane(&result, VectorArrangement::H8, 1, 0x0001, &e);
        assert_lane(&result, VectorArrangement::H8, 4, 0xFFFF, &e);
        assert_lane(&result, VectorArrangement::H8, 5, 0x3FFF, &e);
    }

    #[test]
    fn test_neon_sshr_16b() {
        let vn = var_128("vn");
        let result = encode_neon_sshr(VectorArrangement::B16, &vn, 1);

        let mut e = HashMap::new();
        set_128(
            &mut e,
            "vn",
            pack_8b(0x80, 0x02, 0, 0, 0, 0, 0, 0),
            pack_8b(0xFF, 0x7F, 0, 0, 0, 0, 0, 0),
        );

        // 0x80 >>s 1 = 0xC0, 0x02 >>s 1 = 0x01
        // 0xFF >>s 1 = 0xFF (-1 >>s 1 = -1 in 8-bit)
        // 0x7F >>s 1 = 0x3F
        assert_lane(&result, VectorArrangement::B16, 0, 0xC0, &e);
        assert_lane(&result, VectorArrangement::B16, 1, 0x01, &e);
        assert_lane(&result, VectorArrangement::B16, 8, 0xFF, &e);
        assert_lane(&result, VectorArrangement::B16, 9, 0x3F, &e);
    }

    #[test]
    fn test_neon_sshr_2d() {
        let vn = var_128("vn");
        let result = encode_neon_sshr(VectorArrangement::D2, &vn, 4);

        let mut e = HashMap::new();
        // 0x8000_0000_0000_0000 is negative in signed 64-bit
        set_128(&mut e, "vn", 0x8000_0000_0000_0000, 0x10);

        // 0x8000_0000_0000_0000 >>s 4 = 0xF800_0000_0000_0000
        // 0x10 >>s 4 = 0x01
        assert_lane(&result, VectorArrangement::D2, 0, 0xF800_0000_0000_0000, &e);
        assert_lane(&result, VectorArrangement::D2, 1, 0x01, &e);
    }

    // =======================================================================
    // 128-bit cross-checks
    // =======================================================================

    #[test]
    fn test_add_sub_identity_4s() {
        // (vn + vm) - vm = vn for all lane values
        let vn = var_128("vn");
        let vm = var_128("vm");

        let added = encode_neon_add(VectorArrangement::S4, &vn, &vm);
        let result = encode_neon_sub(VectorArrangement::S4, &added, &vm);

        let mut e = HashMap::new();
        set_128(&mut e, "vn", pack_2s(0xDEADBEEF, 0x12345678), pack_2s(0xCAFEBABE, 0x87654321));
        set_128(&mut e, "vm", pack_2s(0xCAFEBABE, 0x87654321), pack_2s(0xDEADBEEF, 0x12345678));

        assert_all_lanes(
            &result,
            VectorArrangement::S4,
            &[0xDEADBEEF, 0x12345678, 0xCAFEBABE, 0x87654321],
            &e,
        );
    }

    #[test]
    fn test_shl_ushr_roundtrip_4s() {
        let vn = var_128("vn");

        let shifted_left = encode_neon_shl(VectorArrangement::S4, &vn, 4);
        let shifted_back = encode_neon_ushr(VectorArrangement::S4, &shifted_left, 4);

        let mut e = HashMap::new();
        set_128(
            &mut e,
            "vn",
            pack_2s(0x1234ABCD, 0xFFFFFFFF),
            pack_2s(0x0000000F, 0x12345678),
        );

        // SHL 4 then USHR 4 clears top 4 bits of each 32-bit lane
        assert_all_lanes(
            &shifted_back,
            VectorArrangement::S4,
            &[0x0234ABCD, 0x0FFFFFFF, 0x0000000F, 0x02345678],
            &e,
        );
    }

    #[test]
    fn test_add_sub_identity_2d() {
        let vn = var_128("vn");
        let vm = var_128("vm");

        let added = encode_neon_add(VectorArrangement::D2, &vn, &vm);
        let result = encode_neon_sub(VectorArrangement::D2, &added, &vm);

        let mut e = HashMap::new();
        set_128(&mut e, "vn", 0xDEADBEEFCAFEBABE, 0x123456789ABCDEF0);
        set_128(&mut e, "vm", 0xCAFEBABEDEADBEEF, 0x9ABCDEF012345678);

        assert_all_lanes(
            &result,
            VectorArrangement::D2,
            &[0xDEADBEEFCAFEBABE, 0x123456789ABCDEF0],
            &e,
        );
    }
}
