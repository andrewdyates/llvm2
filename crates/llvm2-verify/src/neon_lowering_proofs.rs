// llvm2-verify/neon_lowering_proofs.rs - NEON SIMD lowering verification proofs
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Proof obligations verifying that tMIR vector operations lower correctly
// to AArch64 NEON instructions. Each proof pairs a tMIR-side semantic
// encoding (lane-wise scalar operation on bitvector vectors) with the
// corresponding NEON semantic encoding from `neon_semantics.rs`.
//
// The tMIR side encodes vector operations as per-lane scalar ops applied
// to a flat bitvector, which is the canonical tMIR vector semantics.
// The NEON side uses the instruction-specific encoders. The proof shows
// these are semantically equivalent for all inputs.
//
// 128-bit vectors are represented as pairs of 64-bit symbolic variables
// (`{name}_lo`, `{name}_hi`) concatenated via `hi.concat(lo)`. This is
// required because the mock evaluator uses u64 concrete values.
//
// Reference: designs/2026-04-13-verification-architecture.md
// Reference: ARM Architecture Reference Manual (DDI 0487), Sections C7.2

//! NEON SIMD lowering verification proofs.
//!
//! Provides [`ProofObligation`]s that verify tMIR vector operations lower
//! correctly to AArch64 NEON instructions. Each proof covers both a 64-bit
//! and a 128-bit vector arrangement.
//!
//! Verified operations:
//! - Arithmetic: vector add, sub, mul, neg
//! - Bitwise: and, or (orr), xor (eor), bit clear (bic)
//! - Shifts: shl, ushr (logical shift right), sshr (arithmetic shift right)

use crate::lowering_proof::ProofObligation;
use crate::neon_semantics::{
    encode_neon_add, encode_neon_and, encode_neon_bic, encode_neon_eor, encode_neon_mul,
    encode_neon_neg, encode_neon_orr, encode_neon_shl, encode_neon_sshr, encode_neon_sub,
    encode_neon_ushr,
};
use crate::smt::{map_lanes_binary, map_lanes_binary_imm, map_lanes_unary, SmtExpr,
    VectorArrangement};

// ---------------------------------------------------------------------------
// Helper: build symbolic vector inputs
// ---------------------------------------------------------------------------

/// Build a 128-bit symbolic vector from two 64-bit halves.
///
/// Returns `hi.concat(lo)` where `lo = {prefix}_lo` (bits [63:0]) and
/// `hi = {prefix}_hi` (bits [127:64]).
fn var_128(prefix: &str) -> SmtExpr {
    let lo = SmtExpr::var(format!("{}_lo", prefix), 64);
    let hi = SmtExpr::var(format!("{}_hi", prefix), 64);
    hi.concat(lo)
}

/// Build a symbolic vector at the given arrangement's total width.
///
/// For 64-bit arrangements, returns a single 64-bit variable.
/// For 128-bit arrangements, returns `hi.concat(lo)` (two 64-bit halves).
fn symbolic_vector(name: &str, arrangement: VectorArrangement) -> SmtExpr {
    let w = arrangement.total_bits();
    if w <= 64 {
        SmtExpr::var(name, w)
    } else {
        var_128(name)
    }
}

/// Build input descriptors for a symbolic vector variable.
///
/// For 64-bit: `[(name, 64)]`.
/// For 128-bit: `[(name_lo, 64), (name_hi, 64)]`.
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

/// Build a symbolic bitvector at the given width (splits at 128).
fn symbolic_bv(name: &str, width: u32) -> SmtExpr {
    if width <= 64 {
        SmtExpr::var(name, width)
    } else {
        var_128(name)
    }
}

// ---------------------------------------------------------------------------
// Vector ADD proofs
// ---------------------------------------------------------------------------

/// Proof: tMIR vector_add -> NEON ADD at the specified arrangement.
///
/// tMIR semantics: per-lane `bvadd`.
/// NEON semantics: `encode_neon_add`.
fn proof_vector_add(arrangement: VectorArrangement, label: &str) -> ProofObligation {
    let vn = symbolic_vector("vn", arrangement);
    let vm = symbolic_vector("vm", arrangement);
    let tmir_expr = map_lanes_binary(&vn, &vm, arrangement, |a, b| a.bvadd(b));
    let neon_expr = encode_neon_add(arrangement, &vn, &vm);

    let mut inputs = vector_inputs("vn", arrangement);
    inputs.extend(vector_inputs("vm", arrangement));

    ProofObligation {
        name: format!("VectorAdd -> NEON ADD.{}", label),
        tmir_expr,
        aarch64_expr: neon_expr,
        inputs,
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: tMIR vector_add -> NEON ADD.2S (64-bit, 2x32-bit lanes).
pub fn proof_vector_add_2s() -> ProofObligation {
    proof_vector_add(VectorArrangement::S2, "2S")
}

/// Proof: tMIR vector_add -> NEON ADD.4S (128-bit, 4x32-bit lanes).
pub fn proof_vector_add_4s() -> ProofObligation {
    proof_vector_add(VectorArrangement::S4, "4S")
}

// ---------------------------------------------------------------------------
// Vector SUB proofs
// ---------------------------------------------------------------------------

/// Proof: tMIR vector_sub -> NEON SUB at the specified arrangement.
fn proof_vector_sub(arrangement: VectorArrangement, label: &str) -> ProofObligation {
    let vn = symbolic_vector("vn", arrangement);
    let vm = symbolic_vector("vm", arrangement);
    let tmir_expr = map_lanes_binary(&vn, &vm, arrangement, |a, b| a.bvsub(b));
    let neon_expr = encode_neon_sub(arrangement, &vn, &vm);

    let mut inputs = vector_inputs("vn", arrangement);
    inputs.extend(vector_inputs("vm", arrangement));

    ProofObligation {
        name: format!("VectorSub -> NEON SUB.{}", label),
        tmir_expr,
        aarch64_expr: neon_expr,
        inputs,
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: tMIR vector_sub -> NEON SUB.4H (64-bit, 4x16-bit lanes).
pub fn proof_vector_sub_4h() -> ProofObligation {
    proof_vector_sub(VectorArrangement::H4, "4H")
}

/// Proof: tMIR vector_sub -> NEON SUB.8H (128-bit, 8x16-bit lanes).
pub fn proof_vector_sub_8h() -> ProofObligation {
    proof_vector_sub(VectorArrangement::H8, "8H")
}

// ---------------------------------------------------------------------------
// Vector MUL proofs
// ---------------------------------------------------------------------------

/// Proof: tMIR vector_mul -> NEON MUL at the specified arrangement.
///
/// Note: NEON MUL does not support D2 (64-bit lane) arrangement.
fn proof_vector_mul(arrangement: VectorArrangement, label: &str) -> ProofObligation {
    let vn = symbolic_vector("vn", arrangement);
    let vm = symbolic_vector("vm", arrangement);
    let tmir_expr = map_lanes_binary(&vn, &vm, arrangement, |a, b| a.bvmul(b));
    let neon_expr = encode_neon_mul(arrangement, &vn, &vm);

    let mut inputs = vector_inputs("vn", arrangement);
    inputs.extend(vector_inputs("vm", arrangement));

    ProofObligation {
        name: format!("VectorMul -> NEON MUL.{}", label),
        tmir_expr,
        aarch64_expr: neon_expr,
        inputs,
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: tMIR vector_mul -> NEON MUL.8B (64-bit, 8x8-bit lanes).
pub fn proof_vector_mul_8b() -> ProofObligation {
    proof_vector_mul(VectorArrangement::B8, "8B")
}

/// Proof: tMIR vector_mul -> NEON MUL.16B (128-bit, 16x8-bit lanes).
pub fn proof_vector_mul_16b() -> ProofObligation {
    proof_vector_mul(VectorArrangement::B16, "16B")
}

// ---------------------------------------------------------------------------
// Vector NEG proofs
// ---------------------------------------------------------------------------

/// Proof: tMIR vector_neg -> NEON NEG at the specified arrangement.
fn proof_vector_neg(arrangement: VectorArrangement, label: &str) -> ProofObligation {
    let vn = symbolic_vector("vn", arrangement);
    let tmir_expr = map_lanes_unary(&vn, arrangement, |a| a.bvneg());
    let neon_expr = encode_neon_neg(arrangement, &vn);

    ProofObligation {
        name: format!("VectorNeg -> NEON NEG.{}", label),
        tmir_expr,
        aarch64_expr: neon_expr,
        inputs: vector_inputs("vn", arrangement),
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: tMIR vector_neg -> NEON NEG.2S (64-bit, 2x32-bit lanes).
pub fn proof_vector_neg_2s() -> ProofObligation {
    proof_vector_neg(VectorArrangement::S2, "2S")
}

/// Proof: tMIR vector_neg -> NEON NEG.2D (128-bit, 2x64-bit lanes).
pub fn proof_vector_neg_2d() -> ProofObligation {
    proof_vector_neg(VectorArrangement::D2, "2D")
}

// ---------------------------------------------------------------------------
// Vector AND proofs
// ---------------------------------------------------------------------------

/// Proof: tMIR vector_and -> NEON AND at the specified bit-width.
///
/// Bitwise AND is width-agnostic (no lane decomposition).
/// tMIR: `bvand(vn, vm)`. NEON: `encode_neon_and(vn, vm)`.
fn proof_vector_and(width: u32, label: &str) -> ProofObligation {
    let vn = symbolic_bv("vn", width);
    let vm = symbolic_bv("vm", width);
    let tmir_expr = vn.clone().bvand(vm.clone());
    let neon_expr = encode_neon_and(&vn, &vm);

    ProofObligation {
        name: format!("VectorAnd -> NEON AND.{}", label),
        tmir_expr,
        aarch64_expr: neon_expr,
        inputs: bitwise_inputs(width),
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: tMIR vector_and -> NEON AND.8B (64-bit).
pub fn proof_vector_and_8b() -> ProofObligation {
    proof_vector_and(64, "8B")
}

/// Proof: tMIR vector_and -> NEON AND.16B (128-bit).
pub fn proof_vector_and_16b() -> ProofObligation {
    proof_vector_and(128, "16B")
}

// ---------------------------------------------------------------------------
// Vector ORR proofs
// ---------------------------------------------------------------------------

/// Proof: tMIR vector_or -> NEON ORR at the specified bit-width.
fn proof_vector_orr(width: u32, label: &str) -> ProofObligation {
    let vn = symbolic_bv("vn", width);
    let vm = symbolic_bv("vm", width);
    let tmir_expr = vn.clone().bvor(vm.clone());
    let neon_expr = encode_neon_orr(&vn, &vm);

    ProofObligation {
        name: format!("VectorOr -> NEON ORR.{}", label),
        tmir_expr,
        aarch64_expr: neon_expr,
        inputs: bitwise_inputs(width),
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: tMIR vector_or -> NEON ORR.8B (64-bit).
pub fn proof_vector_orr_8b() -> ProofObligation {
    proof_vector_orr(64, "8B")
}

/// Proof: tMIR vector_or -> NEON ORR.16B (128-bit).
pub fn proof_vector_orr_16b() -> ProofObligation {
    proof_vector_orr(128, "16B")
}

// ---------------------------------------------------------------------------
// Vector EOR proofs
// ---------------------------------------------------------------------------

/// Proof: tMIR vector_xor -> NEON EOR at the specified bit-width.
fn proof_vector_eor(width: u32, label: &str) -> ProofObligation {
    let vn = symbolic_bv("vn", width);
    let vm = symbolic_bv("vm", width);
    let tmir_expr = vn.clone().bvxor(vm.clone());
    let neon_expr = encode_neon_eor(&vn, &vm);

    ProofObligation {
        name: format!("VectorXor -> NEON EOR.{}", label),
        tmir_expr,
        aarch64_expr: neon_expr,
        inputs: bitwise_inputs(width),
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: tMIR vector_xor -> NEON EOR.8B (64-bit).
pub fn proof_vector_eor_8b() -> ProofObligation {
    proof_vector_eor(64, "8B")
}

/// Proof: tMIR vector_xor -> NEON EOR.16B (128-bit).
pub fn proof_vector_eor_16b() -> ProofObligation {
    proof_vector_eor(128, "16B")
}

// ---------------------------------------------------------------------------
// Vector BIC proofs
// ---------------------------------------------------------------------------

/// Proof: tMIR vector_bic (and-not) -> NEON BIC at the specified bit-width.
///
/// tMIR: `bvand(vn, bvnot(vm))` where bvnot is `bvxor(vm, all_ones)`.
/// NEON: `encode_neon_bic(vn, vm)`.
fn proof_vector_bic(width: u32, label: &str) -> ProofObligation {
    let vn = symbolic_bv("vn", width);
    let vm = symbolic_bv("vm", width);

    // tMIR BIC semantics: AND with complement of second operand.
    // Build NOT(vm) as XOR with all-ones, then AND with vn.
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
    let tmir_expr = vn.clone().bvand(vm.clone().bvxor(all_ones));
    let neon_expr = encode_neon_bic(&vn, &vm);

    ProofObligation {
        name: format!("VectorBic -> NEON BIC.{}", label),
        tmir_expr,
        aarch64_expr: neon_expr,
        inputs: bitwise_inputs(width),
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: tMIR vector_bic -> NEON BIC.8B (64-bit).
pub fn proof_vector_bic_8b() -> ProofObligation {
    proof_vector_bic(64, "8B")
}

/// Proof: tMIR vector_bic -> NEON BIC.16B (128-bit).
pub fn proof_vector_bic_16b() -> ProofObligation {
    proof_vector_bic(128, "16B")
}

// ---------------------------------------------------------------------------
// Vector SHL proofs
// ---------------------------------------------------------------------------

/// Proof: tMIR vector_shl -> NEON SHL at the specified arrangement.
///
/// tMIR semantics: per-lane `bvshl` by immediate.
/// NEON semantics: `encode_neon_shl`.
fn proof_vector_shl(arrangement: VectorArrangement, imm: u32, label: &str) -> ProofObligation {
    let vn = symbolic_vector("vn", arrangement);
    let tmir_expr = map_lanes_binary_imm(&vn, imm as u64, arrangement, |a, b| a.bvshl(b));
    let neon_expr = encode_neon_shl(arrangement, &vn, imm);

    ProofObligation {
        name: format!("VectorShl -> NEON SHL.{} #imm={}", label, imm),
        tmir_expr,
        aarch64_expr: neon_expr,
        inputs: vector_inputs("vn", arrangement),
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: tMIR vector_shl -> NEON SHL.4H #4 (64-bit, 4x16-bit lanes, shift by 4).
pub fn proof_vector_shl_4h() -> ProofObligation {
    proof_vector_shl(VectorArrangement::H4, 4, "4H")
}

/// Proof: tMIR vector_shl -> NEON SHL.4S #8 (128-bit, 4x32-bit lanes, shift by 8).
pub fn proof_vector_shl_4s() -> ProofObligation {
    proof_vector_shl(VectorArrangement::S4, 8, "4S")
}

// ---------------------------------------------------------------------------
// Vector USHR proofs
// ---------------------------------------------------------------------------

/// Proof: tMIR vector_ushr -> NEON USHR at the specified arrangement.
///
/// tMIR semantics: per-lane `bvlshr` by immediate.
/// NEON semantics: `encode_neon_ushr`.
fn proof_vector_ushr(arrangement: VectorArrangement, imm: u32, label: &str) -> ProofObligation {
    let vn = symbolic_vector("vn", arrangement);
    let tmir_expr = map_lanes_binary_imm(&vn, imm as u64, arrangement, |a, b| a.bvlshr(b));
    let neon_expr = encode_neon_ushr(arrangement, &vn, imm);

    ProofObligation {
        name: format!("VectorUshr -> NEON USHR.{} #imm={}", label, imm),
        tmir_expr,
        aarch64_expr: neon_expr,
        inputs: vector_inputs("vn", arrangement),
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: tMIR vector_ushr -> NEON USHR.8B #2 (64-bit, 8x8-bit lanes, shift by 2).
pub fn proof_vector_ushr_8b() -> ProofObligation {
    proof_vector_ushr(VectorArrangement::B8, 2, "8B")
}

/// Proof: tMIR vector_ushr -> NEON USHR.2D #16 (128-bit, 2x64-bit lanes, shift by 16).
pub fn proof_vector_ushr_2d() -> ProofObligation {
    proof_vector_ushr(VectorArrangement::D2, 16, "2D")
}

// ---------------------------------------------------------------------------
// Vector SSHR proofs
// ---------------------------------------------------------------------------

/// Proof: tMIR vector_sshr -> NEON SSHR at the specified arrangement.
///
/// tMIR semantics: per-lane `bvashr` by immediate.
/// NEON semantics: `encode_neon_sshr`.
fn proof_vector_sshr(arrangement: VectorArrangement, imm: u32, label: &str) -> ProofObligation {
    let vn = symbolic_vector("vn", arrangement);
    let tmir_expr = map_lanes_binary_imm(&vn, imm as u64, arrangement, |a, b| a.bvashr(b));
    let neon_expr = encode_neon_sshr(arrangement, &vn, imm);

    ProofObligation {
        name: format!("VectorSshr -> NEON SSHR.{} #imm={}", label, imm),
        tmir_expr,
        aarch64_expr: neon_expr,
        inputs: vector_inputs("vn", arrangement),
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: tMIR vector_sshr -> NEON SSHR.2S #1 (64-bit, 2x32-bit lanes, shift by 1).
pub fn proof_vector_sshr_2s() -> ProofObligation {
    proof_vector_sshr(VectorArrangement::S2, 1, "2S")
}

/// Proof: tMIR vector_sshr -> NEON SSHR.8H #4 (128-bit, 8x16-bit lanes, shift by 4).
pub fn proof_vector_sshr_8h() -> ProofObligation {
    proof_vector_sshr(VectorArrangement::H8, 4, "8H")
}

// ---------------------------------------------------------------------------
// Aggregate: all NEON lowering proofs
// ---------------------------------------------------------------------------

/// Return all 22 NEON SIMD lowering proof obligations.
///
/// 11 operations x 2 arrangements (one 64-bit, one 128-bit) = 22 proofs.
pub fn all_neon_lowering_proofs() -> Vec<ProofObligation> {
    vec![
        // Arithmetic (4 ops x 2 arrangements = 8 proofs)
        proof_vector_add_2s(),
        proof_vector_add_4s(),
        proof_vector_sub_4h(),
        proof_vector_sub_8h(),
        proof_vector_mul_8b(),
        proof_vector_mul_16b(),
        proof_vector_neg_2s(),
        proof_vector_neg_2d(),
        // Bitwise (4 ops x 2 arrangements = 8 proofs)
        proof_vector_and_8b(),
        proof_vector_and_16b(),
        proof_vector_orr_8b(),
        proof_vector_orr_16b(),
        proof_vector_eor_8b(),
        proof_vector_eor_16b(),
        proof_vector_bic_8b(),
        proof_vector_bic_16b(),
        // Shifts (3 ops x 2 arrangements = 6 proofs)
        proof_vector_shl_4h(),
        proof_vector_shl_4s(),
        proof_vector_ushr_8b(),
        proof_vector_ushr_2d(),
        proof_vector_sshr_2s(),
        proof_vector_sshr_8h(),
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
    // Vector ADD
    // =======================================================================

    #[test]
    fn test_proof_vector_add_2s() {
        assert_valid(&proof_vector_add_2s());
    }

    #[test]
    fn test_proof_vector_add_4s() {
        assert_valid(&proof_vector_add_4s());
    }

    // =======================================================================
    // Vector SUB
    // =======================================================================

    #[test]
    fn test_proof_vector_sub_4h() {
        assert_valid(&proof_vector_sub_4h());
    }

    #[test]
    fn test_proof_vector_sub_8h() {
        assert_valid(&proof_vector_sub_8h());
    }

    // =======================================================================
    // Vector MUL
    // =======================================================================

    #[test]
    fn test_proof_vector_mul_8b() {
        assert_valid(&proof_vector_mul_8b());
    }

    #[test]
    fn test_proof_vector_mul_16b() {
        assert_valid(&proof_vector_mul_16b());
    }

    // =======================================================================
    // Vector NEG
    // =======================================================================

    #[test]
    fn test_proof_vector_neg_2s() {
        assert_valid(&proof_vector_neg_2s());
    }

    #[test]
    fn test_proof_vector_neg_2d() {
        assert_valid(&proof_vector_neg_2d());
    }

    // =======================================================================
    // Vector AND
    // =======================================================================

    #[test]
    fn test_proof_vector_and_8b() {
        assert_valid(&proof_vector_and_8b());
    }

    #[test]
    fn test_proof_vector_and_16b() {
        assert_valid(&proof_vector_and_16b());
    }

    // =======================================================================
    // Vector ORR
    // =======================================================================

    #[test]
    fn test_proof_vector_orr_8b() {
        assert_valid(&proof_vector_orr_8b());
    }

    #[test]
    fn test_proof_vector_orr_16b() {
        assert_valid(&proof_vector_orr_16b());
    }

    // =======================================================================
    // Vector EOR
    // =======================================================================

    #[test]
    fn test_proof_vector_eor_8b() {
        assert_valid(&proof_vector_eor_8b());
    }

    #[test]
    fn test_proof_vector_eor_16b() {
        assert_valid(&proof_vector_eor_16b());
    }

    // =======================================================================
    // Vector BIC
    // =======================================================================

    #[test]
    fn test_proof_vector_bic_8b() {
        assert_valid(&proof_vector_bic_8b());
    }

    #[test]
    fn test_proof_vector_bic_16b() {
        assert_valid(&proof_vector_bic_16b());
    }

    // =======================================================================
    // Vector SHL
    // =======================================================================

    #[test]
    fn test_proof_vector_shl_4h() {
        assert_valid(&proof_vector_shl_4h());
    }

    #[test]
    fn test_proof_vector_shl_4s() {
        assert_valid(&proof_vector_shl_4s());
    }

    // =======================================================================
    // Vector USHR
    // =======================================================================

    #[test]
    fn test_proof_vector_ushr_8b() {
        assert_valid(&proof_vector_ushr_8b());
    }

    #[test]
    fn test_proof_vector_ushr_2d() {
        assert_valid(&proof_vector_ushr_2d());
    }

    // =======================================================================
    // Vector SSHR
    // =======================================================================

    #[test]
    fn test_proof_vector_sshr_2s() {
        assert_valid(&proof_vector_sshr_2s());
    }

    #[test]
    fn test_proof_vector_sshr_8h() {
        assert_valid(&proof_vector_sshr_8h());
    }

    // =======================================================================
    // Aggregate test: all 22 proofs
    // =======================================================================

    #[test]
    fn test_all_neon_lowering_proofs() {
        let proofs = all_neon_lowering_proofs();
        assert_eq!(proofs.len(), 22, "expected 11 ops x 2 arrangements = 22 proofs");
        for obligation in &proofs {
            assert_valid(obligation);
        }
    }

    // =======================================================================
    // Negative test: wrong NEON instruction detected
    // =======================================================================

    /// Verify that mapping vector_add to NEON SUB is caught as invalid.
    #[test]
    fn test_wrong_neon_lowering_detected() {
        let arrangement = VectorArrangement::S2;
        let vn = symbolic_vector("vn", arrangement);
        let vm = symbolic_vector("vm", arrangement);

        let mut inputs = vector_inputs("vn", arrangement);
        inputs.extend(vector_inputs("vm", arrangement));

        // tMIR says ADD, NEON says SUB -- should find counterexample.
        let obligation = ProofObligation {
            name: "WRONG: VectorAdd -> NEON SUB.2S".to_string(),
            tmir_expr: map_lanes_binary(&vn, &vm, arrangement, |a, b| a.bvadd(b)),
            aarch64_expr: encode_neon_sub(arrangement, &vn, &vm),
            inputs,
            preconditions: vec![],
            fp_inputs: vec![],
            category: None,
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {} // expected
            other => panic!(
                "Expected Invalid for wrong NEON lowering, got {:?}",
                other
            ),
        }
    }
}
