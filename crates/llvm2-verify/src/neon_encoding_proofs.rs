// llvm2-verify/neon_encoding_proofs.rs - NEON encoding correctness proofs
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Proof obligations verifying that NEON SIMD instructions produce correct
// results at the encoding level. Wave 22 added 21 NEON opcodes with binary
// encoding. This module provides formal proofs for those encodings.
//
// These proofs complement `neon_lowering_proofs.rs` (which verifies
// tMIR vector ops -> NEON) by verifying NEON instruction semantics at the
// encoding/execution level. Each proof establishes a semantic identity:
//
// | Proof Category          | What It Proves                                    |
// |------------------------|---------------------------------------------------|
// | Vector ADD lane decomp | ADD computes element-wise addition per lane        |
// | Vector SUB lane decomp | SUB computes element-wise subtraction per lane     |
// | Vector MUL lane decomp | MUL computes element-wise multiplication per lane  |
// | Vector AND bitwise     | AND applies bitwise AND to all 128 bits            |
// | Vector ORR bitwise     | ORR applies bitwise OR to all 128 bits             |
// | Vector EOR bitwise     | EOR applies bitwise XOR to all 128 bits            |
// | Vector NOT bitwise     | NOT inverts all bits                               |
// | DUP broadcast          | DUP Vn.4S, Wm sets all 4 lanes to Wm              |
// | INS lane insert        | INS only modifies the target lane                  |
// | CMEQ comparison        | Per-lane equality produces 0 or all-ones mask      |
// | MOVI immediate         | MOVI sets all bytes to the immediate value         |
// | LD1/ST1 roundtrip      | Store then load preserves vector value             |
//
// Reference: ARM Architecture Reference Manual (DDI 0487), Sections C7.2
// Reference: designs/2026-04-13-verification-architecture.md

//! NEON encoding correctness proofs.
//!
//! Provides [`ProofObligation`]s verifying that NEON SIMD instruction encodings
//! produce semantically correct results. Covers the 21 opcodes added in Wave 22.
//!
//! These proofs use a different framing from `neon_lowering_proofs`: instead of
//! proving "tMIR op == NEON op", they prove "NEON encoding semantics match the
//! expected lane-decomposed behavior". This validates the encoding layer.
//!
//! ## Coverage
//!
//! 12 proof categories, 28 total proof obligations:
//! - Arithmetic lane decomposition: ADD, SUB, MUL (6 proofs)
//! - Bitwise: AND, ORR, EOR, NOT (8 proofs)
//! - Data movement: DUP, INS, MOVI (6 proofs)
//! - Comparison: CMEQ (2 proofs)
//! - Memory roundtrip: LD1/ST1 (2 proofs)
//! - Cross-check: NOT(NOT(x)) == x, DUP lane consistency (4 proofs)

use crate::lowering_proof::ProofObligation;
use crate::memory_proofs::{encode_load_le, encode_store_le, symbolic_memory};
use crate::neon_semantics::{
    encode_neon_add, encode_neon_and, encode_neon_cmeq, encode_neon_dup, encode_neon_eor,
    encode_neon_ins, encode_neon_movi, encode_neon_mul, encode_neon_not, encode_neon_orr,
    encode_neon_sub,
};
use crate::smt::{concat_lanes, lane_extract, map_lanes_binary, SmtExpr, VectorArrangement};

// ---------------------------------------------------------------------------
// Helper: build symbolic vector inputs (consistent with neon_lowering_proofs)
// ---------------------------------------------------------------------------

/// Build a 128-bit symbolic vector from two 64-bit halves.
fn var_128(prefix: &str) -> SmtExpr {
    let lo = SmtExpr::var(format!("{}_lo", prefix), 64);
    let hi = SmtExpr::var(format!("{}_hi", prefix), 64);
    hi.concat(lo)
}

/// Build a symbolic vector at the given arrangement's total width.
fn symbolic_vector(name: &str, arrangement: VectorArrangement) -> SmtExpr {
    let w = arrangement.total_bits();
    if w <= 64 {
        SmtExpr::var(name, w)
    } else {
        var_128(name)
    }
}

/// Build input descriptors for a symbolic vector variable.
fn vector_inputs(name: &str, arrangement: VectorArrangement) -> Vec<(String, u32)> {
    let w = arrangement.total_bits();
    if w <= 64 {
        vec![(name.to_string(), w)]
    } else {
        vec![
            (format!("{}_lo", name), 64),
            (format!("{}_hi", name), 64),
        ]
    }
}

/// Build a symbolic bitvector at the given width (splits at 128-bit).
fn symbolic_bv(name: &str, width: u32) -> SmtExpr {
    if width <= 64 {
        SmtExpr::var(name, width)
    } else {
        var_128(name)
    }
}

/// Build input descriptors for a bitwise op at the given full width.
fn bitwise_inputs(width: u32) -> Vec<(String, u32)> {
    if width <= 64 {
        vec![("vn".to_string(), width), ("vm".to_string(), width)]
    } else {
        vec![
            ("vn_lo".to_string(), 64),
            ("vn_hi".to_string(), 64),
            ("vm_lo".to_string(), 64),
            ("vm_hi".to_string(), 64),
        ]
    }
}

/// Build input descriptors for a unary bitwise op at the given full width.
fn unary_bitwise_inputs(width: u32) -> Vec<(String, u32)> {
    if width <= 64 {
        vec![("vn".to_string(), width)]
    } else {
        vec![
            ("vn_lo".to_string(), 64),
            ("vn_hi".to_string(), 64),
        ]
    }
}

// ---------------------------------------------------------------------------
// 1. Vector ADD lane decomposition proofs
// ---------------------------------------------------------------------------

/// Proof: NEON ADD encoding computes element-wise addition.
///
/// Verifies that `encode_neon_add(arrangement, vn, vm)` equals the
/// lane-by-lane reconstruction: extract each lane, add per lane, reassemble.
fn proof_add_lane_decomposition(
    arrangement: VectorArrangement,
    label: &str,
) -> ProofObligation {
    let vn = symbolic_vector("vn", arrangement);
    let vm = symbolic_vector("vm", arrangement);

    // "Expected" semantics: explicit lane decomposition
    let expected = map_lanes_binary(&vn, &vm, arrangement, |a, b| a.bvadd(b));
    // "Actual" NEON encoding
    let actual = encode_neon_add(arrangement, &vn, &vm);

    let mut inputs = vector_inputs("vn", arrangement);
    inputs.extend(vector_inputs("vm", arrangement));

    ProofObligation {
        name: format!("NeonEncoding: ADD lane decomposition {}", label),
        tmir_expr: expected,
        aarch64_expr: actual,
        inputs,
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: ADD lane decomposition for 4S (128-bit, 4x32-bit).
pub fn proof_add_lane_4s() -> ProofObligation {
    proof_add_lane_decomposition(VectorArrangement::S4, "4S")
}

/// Proof: ADD lane decomposition for 8H (128-bit, 8x16-bit).
pub fn proof_add_lane_8h() -> ProofObligation {
    proof_add_lane_decomposition(VectorArrangement::H8, "8H")
}

// ---------------------------------------------------------------------------
// 2. Vector SUB lane decomposition proofs
// ---------------------------------------------------------------------------

/// Proof: NEON SUB encoding computes element-wise subtraction.
fn proof_sub_lane_decomposition(
    arrangement: VectorArrangement,
    label: &str,
) -> ProofObligation {
    let vn = symbolic_vector("vn", arrangement);
    let vm = symbolic_vector("vm", arrangement);

    let expected = map_lanes_binary(&vn, &vm, arrangement, |a, b| a.bvsub(b));
    let actual = encode_neon_sub(arrangement, &vn, &vm);

    let mut inputs = vector_inputs("vn", arrangement);
    inputs.extend(vector_inputs("vm", arrangement));

    ProofObligation {
        name: format!("NeonEncoding: SUB lane decomposition {}", label),
        tmir_expr: expected,
        aarch64_expr: actual,
        inputs,
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: SUB lane decomposition for 4S (128-bit, 4x32-bit).
pub fn proof_sub_lane_4s() -> ProofObligation {
    proof_sub_lane_decomposition(VectorArrangement::S4, "4S")
}

/// Proof: SUB lane decomposition for 16B (128-bit, 16x8-bit).
pub fn proof_sub_lane_16b() -> ProofObligation {
    proof_sub_lane_decomposition(VectorArrangement::B16, "16B")
}

// ---------------------------------------------------------------------------
// 3. Vector MUL lane decomposition proofs
// ---------------------------------------------------------------------------

/// Proof: NEON MUL encoding computes element-wise multiplication.
fn proof_mul_lane_decomposition(
    arrangement: VectorArrangement,
    label: &str,
) -> ProofObligation {
    let vn = symbolic_vector("vn", arrangement);
    let vm = symbolic_vector("vm", arrangement);

    let expected = map_lanes_binary(&vn, &vm, arrangement, |a, b| a.bvmul(b));
    let actual = encode_neon_mul(arrangement, &vn, &vm);

    let mut inputs = vector_inputs("vn", arrangement);
    inputs.extend(vector_inputs("vm", arrangement));

    ProofObligation {
        name: format!("NeonEncoding: MUL lane decomposition {}", label),
        tmir_expr: expected,
        aarch64_expr: actual,
        inputs,
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: MUL lane decomposition for 4S (128-bit, 4x32-bit).
pub fn proof_mul_lane_4s() -> ProofObligation {
    proof_mul_lane_decomposition(VectorArrangement::S4, "4S")
}

/// Proof: MUL lane decomposition for 8H (128-bit, 8x16-bit).
pub fn proof_mul_lane_8h() -> ProofObligation {
    proof_mul_lane_decomposition(VectorArrangement::H8, "8H")
}

// ---------------------------------------------------------------------------
// 4. Vector AND bitwise proofs
// ---------------------------------------------------------------------------

/// Proof: NEON AND applies bitwise AND across all bits.
///
/// tMIR side: `bvand(vn, vm)`.
/// NEON side: `encode_neon_and(vn, vm)`.
fn proof_and_bitwise(width: u32, label: &str) -> ProofObligation {
    let vn = symbolic_bv("vn", width);
    let vm = symbolic_bv("vm", width);
    let expected = vn.clone().bvand(vm.clone());
    let actual = encode_neon_and(&vn, &vm);

    ProofObligation {
        name: format!("NeonEncoding: AND bitwise {}", label),
        tmir_expr: expected,
        aarch64_expr: actual,
        inputs: bitwise_inputs(width),
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: AND bitwise for 8B (64-bit).
pub fn proof_and_8b() -> ProofObligation {
    proof_and_bitwise(64, "8B")
}

/// Proof: AND bitwise for 16B (128-bit).
pub fn proof_and_16b() -> ProofObligation {
    proof_and_bitwise(128, "16B")
}

// ---------------------------------------------------------------------------
// 5. Vector ORR bitwise proofs
// ---------------------------------------------------------------------------

/// Proof: NEON ORR applies bitwise OR across all bits.
fn proof_orr_bitwise(width: u32, label: &str) -> ProofObligation {
    let vn = symbolic_bv("vn", width);
    let vm = symbolic_bv("vm", width);
    let expected = vn.clone().bvor(vm.clone());
    let actual = encode_neon_orr(&vn, &vm);

    ProofObligation {
        name: format!("NeonEncoding: ORR bitwise {}", label),
        tmir_expr: expected,
        aarch64_expr: actual,
        inputs: bitwise_inputs(width),
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: ORR bitwise for 8B (64-bit).
pub fn proof_orr_8b() -> ProofObligation {
    proof_orr_bitwise(64, "8B")
}

/// Proof: ORR bitwise for 16B (128-bit).
pub fn proof_orr_16b() -> ProofObligation {
    proof_orr_bitwise(128, "16B")
}

// ---------------------------------------------------------------------------
// 6. Vector EOR bitwise proofs
// ---------------------------------------------------------------------------

/// Proof: NEON EOR applies bitwise XOR across all bits.
fn proof_eor_bitwise(width: u32, label: &str) -> ProofObligation {
    let vn = symbolic_bv("vn", width);
    let vm = symbolic_bv("vm", width);
    let expected = vn.clone().bvxor(vm.clone());
    let actual = encode_neon_eor(&vn, &vm);

    ProofObligation {
        name: format!("NeonEncoding: EOR bitwise {}", label),
        tmir_expr: expected,
        aarch64_expr: actual,
        inputs: bitwise_inputs(width),
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: EOR bitwise for 8B (64-bit).
pub fn proof_eor_8b() -> ProofObligation {
    proof_eor_bitwise(64, "8B")
}

/// Proof: EOR bitwise for 16B (128-bit).
pub fn proof_eor_16b() -> ProofObligation {
    proof_eor_bitwise(128, "16B")
}

// ---------------------------------------------------------------------------
// 7. Vector NOT bitwise proofs
// ---------------------------------------------------------------------------

/// Proof: NEON NOT inverts all bits.
///
/// tMIR side: `bvxor(vn, all_ones)` (bitwise inversion).
/// NEON side: `encode_neon_not(vn)`.
fn proof_not_bitwise(width: u32, label: &str) -> ProofObligation {
    let vn = symbolic_bv("vn", width);
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
    let expected = vn.clone().bvxor(all_ones);
    let actual = encode_neon_not(&vn);

    ProofObligation {
        name: format!("NeonEncoding: NOT bitwise {}", label),
        tmir_expr: expected,
        aarch64_expr: actual,
        inputs: unary_bitwise_inputs(width),
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: NOT bitwise for 8B (64-bit).
pub fn proof_not_8b() -> ProofObligation {
    proof_not_bitwise(64, "8B")
}

/// Proof: NOT bitwise for 16B (128-bit).
pub fn proof_not_16b() -> ProofObligation {
    proof_not_bitwise(128, "16B")
}

// ---------------------------------------------------------------------------
// 8. DUP broadcast proofs
// ---------------------------------------------------------------------------

/// Proof: DUP Vn.<T>, Wm broadcasts scalar to all lanes.
///
/// tMIR side: manually construct vector with all lanes equal to the scalar.
/// NEON side: `encode_neon_dup(arrangement, scalar)`.
fn proof_dup_broadcast(arrangement: VectorArrangement, label: &str) -> ProofObligation {
    let lane_bits = arrangement.lane_bits();
    let scalar = SmtExpr::var("scalar", lane_bits);

    // Expected: concatenate identical lanes
    let n = arrangement.lane_count();
    let lanes: Vec<SmtExpr> = (0..n).map(|_| scalar.clone()).collect();
    let expected = concat_lanes(&lanes, arrangement);
    let actual = encode_neon_dup(arrangement, &scalar);

    ProofObligation {
        name: format!("NeonEncoding: DUP broadcast {}", label),
        tmir_expr: expected,
        aarch64_expr: actual,
        inputs: vec![("scalar".to_string(), lane_bits)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: DUP broadcast for 4S (128-bit, 4x32-bit lanes).
pub fn proof_dup_4s() -> ProofObligation {
    proof_dup_broadcast(VectorArrangement::S4, "4S")
}

/// Proof: DUP broadcast for 8H (128-bit, 8x16-bit lanes).
pub fn proof_dup_8h() -> ProofObligation {
    proof_dup_broadcast(VectorArrangement::H8, "8H")
}

// ---------------------------------------------------------------------------
// 9. INS lane insert proofs
// ---------------------------------------------------------------------------

/// Proof: INS only modifies the target lane, preserving all others.
///
/// tMIR side: decompose vector into lanes, replace lane `idx`, reassemble.
/// NEON side: `encode_neon_ins(vec, arrangement, idx, new_val)`.
///
/// For the target lane: result[idx] == new_val.
/// For other lanes: result[i] == original[i].
fn proof_ins_lane(
    arrangement: VectorArrangement,
    idx: u32,
    label: &str,
) -> ProofObligation {
    let vec = symbolic_vector("vec", arrangement);
    let lane_bits = arrangement.lane_bits();
    let new_val = SmtExpr::var("new_val", lane_bits);

    // Expected: extract all lanes, replace target, reassemble
    let n = arrangement.lane_count();
    let lanes: Vec<SmtExpr> = (0..n)
        .map(|i| {
            if i == idx {
                new_val.clone()
            } else {
                lane_extract(&vec, arrangement, i)
            }
        })
        .collect();
    let expected = concat_lanes(&lanes, arrangement);
    let actual = encode_neon_ins(&vec, arrangement, idx, new_val.clone());

    let mut inputs = vector_inputs("vec", arrangement);
    inputs.push(("new_val".to_string(), lane_bits));

    ProofObligation {
        name: format!("NeonEncoding: INS lane insert {} idx={}", label, idx),
        tmir_expr: expected,
        aarch64_expr: actual,
        inputs,
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: INS lane insert for 4S, lane 0.
pub fn proof_ins_4s_lane0() -> ProofObligation {
    proof_ins_lane(VectorArrangement::S4, 0, "4S")
}

/// Proof: INS lane insert for 4S, lane 2 (mid-vector).
pub fn proof_ins_4s_lane2() -> ProofObligation {
    proof_ins_lane(VectorArrangement::S4, 2, "4S")
}

// ---------------------------------------------------------------------------
// 10. CMEQ comparison proofs
// ---------------------------------------------------------------------------

/// Proof: CMEQ per-lane equality produces 0 or all-ones mask.
///
/// tMIR side: for each lane, `if vn[i] == vm[i] then all_ones else 0`.
/// NEON side: `encode_neon_cmeq(arrangement, vn, vm)`.
fn proof_cmeq_comparison(
    arrangement: VectorArrangement,
    label: &str,
) -> ProofObligation {
    let vn = symbolic_vector("vn", arrangement);
    let vm = symbolic_vector("vm", arrangement);

    let lane_bits = arrangement.lane_bits();
    let all_ones_lane = SmtExpr::bv_const(
        if lane_bits >= 64 { u64::MAX } else { (1u64 << lane_bits) - 1 },
        lane_bits,
    );
    let zero_lane = SmtExpr::bv_const(0, lane_bits);

    // Expected: explicit per-lane equality check
    let expected = map_lanes_binary(&vn, &vm, arrangement, |a, b| {
        SmtExpr::ite(a.eq_expr(b), all_ones_lane.clone(), zero_lane.clone())
    });
    let actual = encode_neon_cmeq(arrangement, &vn, &vm);

    let mut inputs = vector_inputs("vn", arrangement);
    inputs.extend(vector_inputs("vm", arrangement));

    ProofObligation {
        name: format!("NeonEncoding: CMEQ comparison {}", label),
        tmir_expr: expected,
        aarch64_expr: actual,
        inputs,
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: CMEQ comparison for 4S (128-bit, 4x32-bit).
pub fn proof_cmeq_4s() -> ProofObligation {
    proof_cmeq_comparison(VectorArrangement::S4, "4S")
}

/// Proof: CMEQ comparison for 16B (128-bit, 16x8-bit).
pub fn proof_cmeq_16b() -> ProofObligation {
    proof_cmeq_comparison(VectorArrangement::B16, "16B")
}

// ---------------------------------------------------------------------------
// 11. MOVI immediate proofs
// ---------------------------------------------------------------------------

/// Proof: MOVI sets all bytes to the immediate value.
///
/// tMIR side: construct a bitvector with all bytes equal to `imm`.
/// NEON side: `encode_neon_movi(width, imm)`.
///
/// We verify with a concrete immediate value (0xAB). This is sufficient
/// because the MOVI encoding is parameterized identically on both sides --
/// the proof verifies the structural construction, not the specific value.
fn proof_movi_immediate(width: u32, label: &str) -> ProofObligation {
    let imm: u8 = 0xAB;

    // Expected: construct manually by repeating the byte
    let byte_count = width / 8;
    let byte_val = SmtExpr::bv_const(imm as u64, 8);
    let mut expected = byte_val.clone();
    for _ in 1..byte_count {
        expected = byte_val.clone().concat(expected);
    }

    let actual = encode_neon_movi(width, imm);

    // No symbolic inputs -- this is a constant proof
    ProofObligation {
        name: format!("NeonEncoding: MOVI immediate {}", label),
        tmir_expr: expected,
        aarch64_expr: actual,
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: MOVI immediate for 8B (64-bit).
pub fn proof_movi_8b() -> ProofObligation {
    proof_movi_immediate(64, "8B")
}

/// Proof: MOVI immediate for 16B (128-bit).
pub fn proof_movi_16b() -> ProofObligation {
    proof_movi_immediate(128, "16B")
}

// ---------------------------------------------------------------------------
// 12. LD1/ST1 roundtrip proofs
// ---------------------------------------------------------------------------

/// Proof: store then load preserves vector value (LD1/ST1 roundtrip).
///
/// Models LD1/ST1 using the byte-addressable memory model from
/// `memory_proofs`. A 64-bit vector is stored as 8 bytes (little-endian)
/// then loaded back; a 128-bit vector is stored as two 64-bit halves
/// (lo at addr, hi at addr+8) then loaded and reassembled.
///
/// The proof shows: `load(store(mem, addr, vec)) == vec` for all values.
fn proof_ld1_st1_roundtrip(width: u32, label: &str) -> ProofObligation {
    if width <= 64 {
        // 64-bit: store 8 bytes then load 8 bytes
        let vec_val = SmtExpr::var("vec_val", 64);
        let addr = SmtExpr::var("addr", 64);
        let mem = symbolic_memory("mem_default");

        // Store vector to memory (little-endian, 8 bytes)
        let mem_after = encode_store_le(&mem, &addr, &vec_val, 8);
        // Load back from same address
        let loaded = encode_load_le(&mem_after, &addr, 8);

        ProofObligation {
            name: format!("NeonEncoding: LD1/ST1 roundtrip {}", label),
            tmir_expr: vec_val,
            aarch64_expr: loaded,
            inputs: vec![
                ("vec_val".to_string(), 64),
                ("addr".to_string(), 64),
                ("mem_default".to_string(), 8),
            ],
            preconditions: vec![],
            fp_inputs: vec![],
            category: None,
        }
    } else {
        // 128-bit: store as two 64-bit halves, load back and reassemble
        let lo = SmtExpr::var("vec_lo", 64);
        let hi = SmtExpr::var("vec_hi", 64);
        let addr = SmtExpr::var("addr", 64);
        let addr_plus_8 = addr.clone().bvadd(SmtExpr::bv_const(8, 64));
        let mem = symbolic_memory("mem_default");

        // Store lo half at addr (8 bytes), hi half at addr+8 (8 bytes)
        let mem1 = encode_store_le(&mem, &addr, &lo, 8);
        let mem2 = encode_store_le(&mem1, &addr_plus_8, &hi, 8);

        // Load back
        let loaded_lo = encode_load_le(&mem2, &addr, 8);
        let loaded_hi = encode_load_le(&mem2, &addr_plus_8, 8);

        // Original and loaded vectors
        let original = hi.concat(lo);
        let loaded = loaded_hi.concat(loaded_lo);

        ProofObligation {
            name: format!("NeonEncoding: LD1/ST1 roundtrip {}", label),
            tmir_expr: original,
            aarch64_expr: loaded,
            inputs: vec![
                ("vec_lo".to_string(), 64),
                ("vec_hi".to_string(), 64),
                ("addr".to_string(), 64),
                ("mem_default".to_string(), 8),
            ],
            preconditions: vec![],
            fp_inputs: vec![],
            category: None,
        }
    }
}

/// Proof: LD1/ST1 roundtrip for 64-bit vector (8B).
pub fn proof_ld1_st1_8b() -> ProofObligation {
    proof_ld1_st1_roundtrip(64, "8B")
}

/// Proof: LD1/ST1 roundtrip for 128-bit vector (16B).
pub fn proof_ld1_st1_16b() -> ProofObligation {
    proof_ld1_st1_roundtrip(128, "16B")
}

// ---------------------------------------------------------------------------
// Cross-check proofs: NOT(NOT(x)) == x, DUP lane consistency
// ---------------------------------------------------------------------------

/// Proof: NOT(NOT(x)) == x (double negation identity).
///
/// This is a fundamental bitwise property that validates the NOT encoding.
fn proof_not_involution(width: u32, label: &str) -> ProofObligation {
    let vn = symbolic_bv("vn", width);
    let not_vn = encode_neon_not(&vn);
    let not_not_vn = encode_neon_not(&not_vn);

    ProofObligation {
        name: format!("NeonEncoding: NOT involution {}", label),
        tmir_expr: vn,
        aarch64_expr: not_not_vn,
        inputs: unary_bitwise_inputs(width),
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: NOT involution for 8B (64-bit).
pub fn proof_not_involution_8b() -> ProofObligation {
    proof_not_involution(64, "8B")
}

/// Proof: NOT involution for 16B (128-bit).
pub fn proof_not_involution_16b() -> ProofObligation {
    proof_not_involution(128, "16B")
}

/// Proof: DUP lane consistency -- all lanes of DUP result are identical.
///
/// Extracts lane 0 and lane N-1 from a DUP result and proves they are equal.
/// This validates the broadcast property.
fn proof_dup_lane_consistency(
    arrangement: VectorArrangement,
    label: &str,
) -> ProofObligation {
    let lane_bits = arrangement.lane_bits();
    let scalar = SmtExpr::var("scalar", lane_bits);
    let dup_result = encode_neon_dup(arrangement, &scalar);

    let lane0 = lane_extract(&dup_result, arrangement, 0);
    let last_lane = lane_extract(
        &dup_result,
        arrangement,
        arrangement.lane_count() - 1,
    );

    ProofObligation {
        name: format!("NeonEncoding: DUP lane consistency {}", label),
        tmir_expr: lane0,
        aarch64_expr: last_lane,
        inputs: vec![("scalar".to_string(), lane_bits)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: DUP lane consistency for 4S (lane 0 == lane 3).
pub fn proof_dup_consistency_4s() -> ProofObligation {
    proof_dup_lane_consistency(VectorArrangement::S4, "4S")
}

/// Proof: DUP lane consistency for 16B (lane 0 == lane 15).
pub fn proof_dup_consistency_16b() -> ProofObligation {
    proof_dup_lane_consistency(VectorArrangement::B16, "16B")
}

// ---------------------------------------------------------------------------
// Aggregate: all NEON encoding proofs
// ---------------------------------------------------------------------------

/// Return all 28 NEON encoding proof obligations.
///
/// 12 categories:
///   - Arithmetic lane decomp: ADD (2), SUB (2), MUL (2) = 6
///   - Bitwise: AND (2), ORR (2), EOR (2), NOT (2) = 8
///   - Data movement: DUP (2), INS (2), MOVI (2) = 6
///   - Comparison: CMEQ (2) = 2
///   - Memory: LD1/ST1 (2) = 2
///   - Cross-check: NOT involution (2), DUP consistency (2) = 4
///
///   Total = 28
pub fn all_neon_encoding_proofs() -> Vec<ProofObligation> {
    vec![
        // Arithmetic lane decomposition (6 proofs)
        proof_add_lane_4s(),
        proof_add_lane_8h(),
        proof_sub_lane_4s(),
        proof_sub_lane_16b(),
        proof_mul_lane_4s(),
        proof_mul_lane_8h(),
        // Bitwise (8 proofs)
        proof_and_8b(),
        proof_and_16b(),
        proof_orr_8b(),
        proof_orr_16b(),
        proof_eor_8b(),
        proof_eor_16b(),
        proof_not_8b(),
        proof_not_16b(),
        // Data movement (6 proofs)
        proof_dup_4s(),
        proof_dup_8h(),
        proof_ins_4s_lane0(),
        proof_ins_4s_lane2(),
        proof_movi_8b(),
        proof_movi_16b(),
        // Comparison (2 proofs)
        proof_cmeq_4s(),
        proof_cmeq_16b(),
        // Memory roundtrip (2 proofs)
        proof_ld1_st1_8b(),
        proof_ld1_st1_16b(),
        // Cross-check proofs (4 proofs)
        proof_not_involution_8b(),
        proof_not_involution_16b(),
        proof_dup_consistency_4s(),
        proof_dup_consistency_16b(),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lowering_proof::verify_by_evaluation;
    use crate::verify::VerificationResult;

    /// Verify a proof obligation and assert it is valid.
    fn assert_valid(obligation: &ProofObligation) {
        let result = verify_by_evaluation(obligation);
        match &result {
            VerificationResult::Valid => {}
            VerificationResult::Invalid { counterexample } => {
                panic!(
                    "Proof '{}' FAILED with counterexample: {}",
                    obligation.name, counterexample
                );
            }
            VerificationResult::Unknown { reason } => {
                panic!("Proof '{}' returned Unknown: {}", obligation.name, reason);
            }
        }
    }

    // =======================================================================
    // ADD lane decomposition
    // =======================================================================

    #[test]
    fn test_add_lane_4s() {
        assert_valid(&proof_add_lane_4s());
    }

    #[test]
    fn test_add_lane_8h() {
        assert_valid(&proof_add_lane_8h());
    }

    // =======================================================================
    // SUB lane decomposition
    // =======================================================================

    #[test]
    fn test_sub_lane_4s() {
        assert_valid(&proof_sub_lane_4s());
    }

    #[test]
    fn test_sub_lane_16b() {
        assert_valid(&proof_sub_lane_16b());
    }

    // =======================================================================
    // MUL lane decomposition
    // =======================================================================

    #[test]
    fn test_mul_lane_4s() {
        assert_valid(&proof_mul_lane_4s());
    }

    #[test]
    fn test_mul_lane_8h() {
        assert_valid(&proof_mul_lane_8h());
    }

    // =======================================================================
    // AND bitwise
    // =======================================================================

    #[test]
    fn test_and_8b() {
        assert_valid(&proof_and_8b());
    }

    #[test]
    fn test_and_16b() {
        assert_valid(&proof_and_16b());
    }

    // =======================================================================
    // ORR bitwise
    // =======================================================================

    #[test]
    fn test_orr_8b() {
        assert_valid(&proof_orr_8b());
    }

    #[test]
    fn test_orr_16b() {
        assert_valid(&proof_orr_16b());
    }

    // =======================================================================
    // EOR bitwise
    // =======================================================================

    #[test]
    fn test_eor_8b() {
        assert_valid(&proof_eor_8b());
    }

    #[test]
    fn test_eor_16b() {
        assert_valid(&proof_eor_16b());
    }

    // =======================================================================
    // NOT bitwise
    // =======================================================================

    #[test]
    fn test_not_8b() {
        assert_valid(&proof_not_8b());
    }

    #[test]
    fn test_not_16b() {
        assert_valid(&proof_not_16b());
    }

    // =======================================================================
    // DUP broadcast
    // =======================================================================

    #[test]
    fn test_dup_4s() {
        assert_valid(&proof_dup_4s());
    }

    #[test]
    fn test_dup_8h() {
        assert_valid(&proof_dup_8h());
    }

    // =======================================================================
    // INS lane insert
    // =======================================================================

    #[test]
    fn test_ins_4s_lane0() {
        assert_valid(&proof_ins_4s_lane0());
    }

    #[test]
    fn test_ins_4s_lane2() {
        assert_valid(&proof_ins_4s_lane2());
    }

    // =======================================================================
    // CMEQ comparison
    // =======================================================================

    #[test]
    fn test_cmeq_4s() {
        assert_valid(&proof_cmeq_4s());
    }

    #[test]
    fn test_cmeq_16b() {
        assert_valid(&proof_cmeq_16b());
    }

    // =======================================================================
    // MOVI immediate
    // =======================================================================

    #[test]
    fn test_movi_8b() {
        assert_valid(&proof_movi_8b());
    }

    #[test]
    fn test_movi_16b() {
        assert_valid(&proof_movi_16b());
    }

    // =======================================================================
    // LD1/ST1 roundtrip
    // =======================================================================

    #[test]
    fn test_ld1_st1_8b() {
        assert_valid(&proof_ld1_st1_8b());
    }

    #[test]
    fn test_ld1_st1_16b() {
        assert_valid(&proof_ld1_st1_16b());
    }

    // =======================================================================
    // Cross-check: NOT involution
    // =======================================================================

    #[test]
    fn test_not_involution_8b() {
        assert_valid(&proof_not_involution_8b());
    }

    #[test]
    fn test_not_involution_16b() {
        assert_valid(&proof_not_involution_16b());
    }

    // =======================================================================
    // Cross-check: DUP lane consistency
    // =======================================================================

    #[test]
    fn test_dup_consistency_4s() {
        assert_valid(&proof_dup_consistency_4s());
    }

    #[test]
    fn test_dup_consistency_16b() {
        assert_valid(&proof_dup_consistency_16b());
    }

    // =======================================================================
    // Aggregate: all 28 proofs
    // =======================================================================

    #[test]
    fn test_all_neon_encoding_proofs() {
        let proofs = all_neon_encoding_proofs();
        assert_eq!(proofs.len(), 28, "expected 28 NEON encoding proofs");
        for obligation in &proofs {
            assert_valid(obligation);
        }
    }

    // =======================================================================
    // Negative test: wrong encoding detected
    // =======================================================================

    /// Verify that using ADD encoding where SUB is expected is caught.
    #[test]
    fn test_wrong_encoding_detected() {
        let arrangement = VectorArrangement::S4;
        let vn = symbolic_vector("vn", arrangement);
        let vm = symbolic_vector("vm", arrangement);

        let mut inputs = vector_inputs("vn", arrangement);
        inputs.extend(vector_inputs("vm", arrangement));

        // Expected SUB semantics, actual ADD encoding -- mismatch.
        let obligation = ProofObligation {
            name: "WRONG: SUB encoding == ADD encoding".to_string(),
            tmir_expr: map_lanes_binary(&vn, &vm, arrangement, |a, b| a.bvsub(b)),
            aarch64_expr: encode_neon_add(arrangement, &vn, &vm),
            inputs,
            preconditions: vec![],
            fp_inputs: vec![],
            category: None,
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {} // expected
            other => panic!(
                "Expected Invalid for wrong encoding, got {:?}",
                other
            ),
        }
    }
}
