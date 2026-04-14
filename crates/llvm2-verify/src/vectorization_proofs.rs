// llvm2-verify/vectorization_proofs.rs - NEON auto-vectorization lowering proofs
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Proof obligations verifying that the NEON auto-vectorization pass
// (`llvm2_opt::vectorize::scalar_to_neon_op()`) correctly maps scalar
// AArch64 operations to semantically equivalent NEON SIMD operations.
//
// Each proof shows: for every lane value, applying the scalar operation
// per-lane produces the same result as the corresponding NEON vector
// instruction. This validates the 13 mappings in `scalar_to_neon_op()`:
//
// | Scalar Opcode(s)     | NEON Op | Proof Name            |
// |---------------------|---------|-----------------------|
// | AddRR / AddRI       | Add     | ScalarAdd -> NEON ADD |
// | SubRR / SubRI       | Sub     | ScalarSub -> NEON SUB |
// | MulRR               | Mul     | ScalarMul -> NEON MUL |
// | Neg                 | Neg     | ScalarNeg -> NEON NEG |
// | AndRR / AndRI       | And     | ScalarAnd -> NEON AND |
// | OrrRR / OrrRI       | Orr     | ScalarOrr -> NEON ORR |
// | EorRR / EorRI       | Eor     | ScalarEor -> NEON EOR |
// | BicRR               | Bic     | ScalarBic -> NEON BIC |
// | LslRI               | Shl     | ScalarShl -> NEON SHL |
// | LsrRI               | Ushr    | ScalarUshr -> NEON USHR |
// | AsrRI               | Sshr    | ScalarSshr -> NEON SSHR |
// | FaddRR              | Fadd    | (future: FP semantics) |
// | FmulRR              | Fmul    | (future: FP semantics) |
//
// The key difference from `neon_lowering_proofs.rs`: those proofs verify
// tMIR vector ops -> NEON instructions. These proofs verify the
// *auto-vectorization decision* that scalar_op applied per-lane equals
// the NEON vector op -- justifying the vectorization rewrite.
//
// Reference: crates/llvm2-opt/src/vectorize.rs (scalar_to_neon_op)
// Reference: designs/2026-04-13-verification-architecture.md

//! NEON auto-vectorization lowering verification proofs.
//!
//! Verifies that the scalar-to-NEON operation mappings used by the
//! auto-vectorization pass are semantically correct. Each proof
//! obligation shows that applying a scalar operation independently
//! to each lane of a vector produces the same result as the
//! corresponding NEON SIMD instruction.
//!
//! ## Proof structure
//!
//! For a binary operation `op`:
//! ```text
//! tMIR side:  map_lanes_binary(vn, vm, arrangement, |a, b| scalar_op(a, b))
//! NEON side:  encode_neon_op(arrangement, vn, vm)
//! Proof:      forall vn, vm: tMIR_side == NEON_side
//! ```
//!
//! This validates the vectorization pass's assumption that NEON
//! instructions are lane-wise equivalent to their scalar counterparts.
//!
//! ## Coverage
//!
//! 11 integer/bitwise operations x 2 arrangements = 22 proofs.
//! FADD and FMUL proofs are deferred pending FP semantic encoders.

use crate::lowering_proof::ProofObligation;
use crate::neon_semantics::{
    encode_neon_add, encode_neon_and, encode_neon_bic, encode_neon_eor, encode_neon_mul,
    encode_neon_neg, encode_neon_orr, encode_neon_shl, encode_neon_sshr, encode_neon_sub,
    encode_neon_ushr,
};
use crate::smt::{map_lanes_binary, map_lanes_binary_imm, map_lanes_unary, SmtExpr,
    VectorArrangement};

// ---------------------------------------------------------------------------
// Helper: build symbolic vector inputs (same as neon_lowering_proofs)
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

// ---------------------------------------------------------------------------
// Vectorization proof: scalar ADD -> NEON ADD
// ---------------------------------------------------------------------------

/// Proof: scalar per-lane add -> NEON ADD at the specified arrangement.
///
/// The vectorization pass maps `AddRR`/`AddRI` to `NeonOp::Add`.
/// This proof verifies: for all lane values, independently adding each
/// lane produces the same result as the NEON vector ADD instruction.
///
/// tMIR semantics: per-lane `bvadd` (wrapping).
/// NEON semantics: `encode_neon_add`.
fn proof_vectorize_add(arrangement: VectorArrangement, label: &str) -> ProofObligation {
    let vn = symbolic_vector("vn", arrangement);
    let vm = symbolic_vector("vm", arrangement);
    let scalar_per_lane = map_lanes_binary(&vn, &vm, arrangement, |a, b| a.bvadd(b));
    let neon_result = encode_neon_add(arrangement, &vn, &vm);

    let mut inputs = vector_inputs("vn", arrangement);
    inputs.extend(vector_inputs("vm", arrangement));

    ProofObligation {
        name: format!("Vectorize: ScalarAdd -> NEON ADD.{}", label),
        tmir_expr: scalar_per_lane,
        aarch64_expr: neon_result,
        inputs,
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: scalar add -> NEON ADD.4S (128-bit, 4x32-bit -- primary i32 vectorization).
pub fn proof_vectorize_add_4s() -> ProofObligation {
    proof_vectorize_add(VectorArrangement::S4, "4S")
}

/// Proof: scalar add -> NEON ADD.2D (128-bit, 2x64-bit -- i64 vectorization).
pub fn proof_vectorize_add_2d() -> ProofObligation {
    proof_vectorize_add(VectorArrangement::D2, "2D")
}

// ---------------------------------------------------------------------------
// Vectorization proof: scalar SUB -> NEON SUB
// ---------------------------------------------------------------------------

/// Proof: scalar per-lane sub -> NEON SUB at the specified arrangement.
fn proof_vectorize_sub(arrangement: VectorArrangement, label: &str) -> ProofObligation {
    let vn = symbolic_vector("vn", arrangement);
    let vm = symbolic_vector("vm", arrangement);
    let scalar_per_lane = map_lanes_binary(&vn, &vm, arrangement, |a, b| a.bvsub(b));
    let neon_result = encode_neon_sub(arrangement, &vn, &vm);

    let mut inputs = vector_inputs("vn", arrangement);
    inputs.extend(vector_inputs("vm", arrangement));

    ProofObligation {
        name: format!("Vectorize: ScalarSub -> NEON SUB.{}", label),
        tmir_expr: scalar_per_lane,
        aarch64_expr: neon_result,
        inputs,
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: scalar sub -> NEON SUB.4S (128-bit, 4x32-bit).
pub fn proof_vectorize_sub_4s() -> ProofObligation {
    proof_vectorize_sub(VectorArrangement::S4, "4S")
}

/// Proof: scalar sub -> NEON SUB.8H (128-bit, 8x16-bit).
pub fn proof_vectorize_sub_8h() -> ProofObligation {
    proof_vectorize_sub(VectorArrangement::H8, "8H")
}

// ---------------------------------------------------------------------------
// Vectorization proof: scalar MUL -> NEON MUL
// ---------------------------------------------------------------------------

/// Proof: scalar per-lane mul -> NEON MUL at the specified arrangement.
///
/// Note: NEON MUL does not support D2 (64-bit lane) arrangement.
fn proof_vectorize_mul(arrangement: VectorArrangement, label: &str) -> ProofObligation {
    let vn = symbolic_vector("vn", arrangement);
    let vm = symbolic_vector("vm", arrangement);
    let scalar_per_lane = map_lanes_binary(&vn, &vm, arrangement, |a, b| a.bvmul(b));
    let neon_result = encode_neon_mul(arrangement, &vn, &vm);

    let mut inputs = vector_inputs("vn", arrangement);
    inputs.extend(vector_inputs("vm", arrangement));

    ProofObligation {
        name: format!("Vectorize: ScalarMul -> NEON MUL.{}", label),
        tmir_expr: scalar_per_lane,
        aarch64_expr: neon_result,
        inputs,
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: scalar mul -> NEON MUL.4S (128-bit, 4x32-bit).
pub fn proof_vectorize_mul_4s() -> ProofObligation {
    proof_vectorize_mul(VectorArrangement::S4, "4S")
}

/// Proof: scalar mul -> NEON MUL.16B (128-bit, 16x8-bit).
pub fn proof_vectorize_mul_16b() -> ProofObligation {
    proof_vectorize_mul(VectorArrangement::B16, "16B")
}

// ---------------------------------------------------------------------------
// Vectorization proof: scalar NEG -> NEON NEG
// ---------------------------------------------------------------------------

/// Proof: scalar per-lane neg -> NEON NEG at the specified arrangement.
fn proof_vectorize_neg(arrangement: VectorArrangement, label: &str) -> ProofObligation {
    let vn = symbolic_vector("vn", arrangement);
    let scalar_per_lane = map_lanes_unary(&vn, arrangement, |a| a.bvneg());
    let neon_result = encode_neon_neg(arrangement, &vn);

    ProofObligation {
        name: format!("Vectorize: ScalarNeg -> NEON NEG.{}", label),
        tmir_expr: scalar_per_lane,
        aarch64_expr: neon_result,
        inputs: vector_inputs("vn", arrangement),
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: scalar neg -> NEON NEG.4S (128-bit, 4x32-bit).
pub fn proof_vectorize_neg_4s() -> ProofObligation {
    proof_vectorize_neg(VectorArrangement::S4, "4S")
}

/// Proof: scalar neg -> NEON NEG.2D (128-bit, 2x64-bit).
pub fn proof_vectorize_neg_2d() -> ProofObligation {
    proof_vectorize_neg(VectorArrangement::D2, "2D")
}

// ---------------------------------------------------------------------------
// Vectorization proof: scalar AND -> NEON AND
// ---------------------------------------------------------------------------

/// Proof: scalar bitwise AND -> NEON AND at the specified bit-width.
///
/// Bitwise AND is width-agnostic (no lane decomposition needed).
/// tMIR: `bvand(vn, vm)`. NEON: `encode_neon_and(vn, vm)`.
fn proof_vectorize_and(width: u32, label: &str) -> ProofObligation {
    let vn = symbolic_bv("vn", width);
    let vm = symbolic_bv("vm", width);
    let scalar_result = vn.clone().bvand(vm.clone());
    let neon_result = encode_neon_and(&vn, &vm);

    ProofObligation {
        name: format!("Vectorize: ScalarAnd -> NEON AND.{}", label),
        tmir_expr: scalar_result,
        aarch64_expr: neon_result,
        inputs: bitwise_inputs(width),
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: scalar AND -> NEON AND.8B (64-bit).
pub fn proof_vectorize_and_8b() -> ProofObligation {
    proof_vectorize_and(64, "8B")
}

/// Proof: scalar AND -> NEON AND.16B (128-bit).
pub fn proof_vectorize_and_16b() -> ProofObligation {
    proof_vectorize_and(128, "16B")
}

// ---------------------------------------------------------------------------
// Vectorization proof: scalar ORR -> NEON ORR
// ---------------------------------------------------------------------------

/// Proof: scalar bitwise OR -> NEON ORR at the specified bit-width.
fn proof_vectorize_orr(width: u32, label: &str) -> ProofObligation {
    let vn = symbolic_bv("vn", width);
    let vm = symbolic_bv("vm", width);
    let scalar_result = vn.clone().bvor(vm.clone());
    let neon_result = encode_neon_orr(&vn, &vm);

    ProofObligation {
        name: format!("Vectorize: ScalarOrr -> NEON ORR.{}", label),
        tmir_expr: scalar_result,
        aarch64_expr: neon_result,
        inputs: bitwise_inputs(width),
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: scalar OR -> NEON ORR.8B (64-bit).
pub fn proof_vectorize_orr_8b() -> ProofObligation {
    proof_vectorize_orr(64, "8B")
}

/// Proof: scalar OR -> NEON ORR.16B (128-bit).
pub fn proof_vectorize_orr_16b() -> ProofObligation {
    proof_vectorize_orr(128, "16B")
}

// ---------------------------------------------------------------------------
// Vectorization proof: scalar EOR -> NEON EOR
// ---------------------------------------------------------------------------

/// Proof: scalar bitwise XOR -> NEON EOR at the specified bit-width.
fn proof_vectorize_eor(width: u32, label: &str) -> ProofObligation {
    let vn = symbolic_bv("vn", width);
    let vm = symbolic_bv("vm", width);
    let scalar_result = vn.clone().bvxor(vm.clone());
    let neon_result = encode_neon_eor(&vn, &vm);

    ProofObligation {
        name: format!("Vectorize: ScalarEor -> NEON EOR.{}", label),
        tmir_expr: scalar_result,
        aarch64_expr: neon_result,
        inputs: bitwise_inputs(width),
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: scalar XOR -> NEON EOR.8B (64-bit).
pub fn proof_vectorize_eor_8b() -> ProofObligation {
    proof_vectorize_eor(64, "8B")
}

/// Proof: scalar XOR -> NEON EOR.16B (128-bit).
pub fn proof_vectorize_eor_16b() -> ProofObligation {
    proof_vectorize_eor(128, "16B")
}

// ---------------------------------------------------------------------------
// Vectorization proof: scalar BIC -> NEON BIC
// ---------------------------------------------------------------------------

/// Proof: scalar bit-clear (AND NOT) -> NEON BIC at the specified bit-width.
///
/// tMIR BIC: `bvand(vn, bvxor(vm, all_ones))` where bvxor with all-ones = NOT.
/// NEON BIC: `encode_neon_bic(vn, vm)`.
fn proof_vectorize_bic(width: u32, label: &str) -> ProofObligation {
    let vn = symbolic_bv("vn", width);
    let vm = symbolic_bv("vm", width);

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
    let scalar_result = vn.clone().bvand(vm.clone().bvxor(all_ones));
    let neon_result = encode_neon_bic(&vn, &vm);

    ProofObligation {
        name: format!("Vectorize: ScalarBic -> NEON BIC.{}", label),
        tmir_expr: scalar_result,
        aarch64_expr: neon_result,
        inputs: bitwise_inputs(width),
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: scalar BIC -> NEON BIC.8B (64-bit).
pub fn proof_vectorize_bic_8b() -> ProofObligation {
    proof_vectorize_bic(64, "8B")
}

/// Proof: scalar BIC -> NEON BIC.16B (128-bit).
pub fn proof_vectorize_bic_16b() -> ProofObligation {
    proof_vectorize_bic(128, "16B")
}

// ---------------------------------------------------------------------------
// Vectorization proof: scalar SHL -> NEON SHL
// ---------------------------------------------------------------------------

/// Proof: scalar per-lane shift left -> NEON SHL at the specified arrangement.
///
/// The vectorization pass maps `LslRI` to `NeonOp::Shl`.
/// tMIR semantics: per-lane `bvshl` by immediate.
/// NEON semantics: `encode_neon_shl`.
fn proof_vectorize_shl(
    arrangement: VectorArrangement,
    imm: u32,
    label: &str,
) -> ProofObligation {
    let vn = symbolic_vector("vn", arrangement);
    let scalar_per_lane = map_lanes_binary_imm(&vn, imm as u64, arrangement, |a, b| a.bvshl(b));
    let neon_result = encode_neon_shl(arrangement, &vn, imm);

    ProofObligation {
        name: format!("Vectorize: ScalarShl -> NEON SHL.{} #imm={}", label, imm),
        tmir_expr: scalar_per_lane,
        aarch64_expr: neon_result,
        inputs: vector_inputs("vn", arrangement),
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: scalar shl -> NEON SHL.4S #8 (128-bit, 4x32-bit lanes, shift by 8).
pub fn proof_vectorize_shl_4s() -> ProofObligation {
    proof_vectorize_shl(VectorArrangement::S4, 8, "4S")
}

/// Proof: scalar shl -> NEON SHL.8H #4 (128-bit, 8x16-bit lanes, shift by 4).
pub fn proof_vectorize_shl_8h() -> ProofObligation {
    proof_vectorize_shl(VectorArrangement::H8, 4, "8H")
}

// ---------------------------------------------------------------------------
// Vectorization proof: scalar USHR -> NEON USHR
// ---------------------------------------------------------------------------

/// Proof: scalar per-lane unsigned shift right -> NEON USHR.
///
/// The vectorization pass maps `LsrRI` to `NeonOp::Ushr`.
fn proof_vectorize_ushr(
    arrangement: VectorArrangement,
    imm: u32,
    label: &str,
) -> ProofObligation {
    let vn = symbolic_vector("vn", arrangement);
    let scalar_per_lane = map_lanes_binary_imm(&vn, imm as u64, arrangement, |a, b| a.bvlshr(b));
    let neon_result = encode_neon_ushr(arrangement, &vn, imm);

    ProofObligation {
        name: format!("Vectorize: ScalarUshr -> NEON USHR.{} #imm={}", label, imm),
        tmir_expr: scalar_per_lane,
        aarch64_expr: neon_result,
        inputs: vector_inputs("vn", arrangement),
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: scalar ushr -> NEON USHR.4S #4 (128-bit, 4x32-bit lanes, shift by 4).
pub fn proof_vectorize_ushr_4s() -> ProofObligation {
    proof_vectorize_ushr(VectorArrangement::S4, 4, "4S")
}

/// Proof: scalar ushr -> NEON USHR.2D #16 (128-bit, 2x64-bit lanes, shift by 16).
pub fn proof_vectorize_ushr_2d() -> ProofObligation {
    proof_vectorize_ushr(VectorArrangement::D2, 16, "2D")
}

// ---------------------------------------------------------------------------
// Vectorization proof: scalar SSHR -> NEON SSHR
// ---------------------------------------------------------------------------

/// Proof: scalar per-lane arithmetic shift right -> NEON SSHR.
///
/// The vectorization pass maps `AsrRI` to `NeonOp::Sshr`.
fn proof_vectorize_sshr(
    arrangement: VectorArrangement,
    imm: u32,
    label: &str,
) -> ProofObligation {
    let vn = symbolic_vector("vn", arrangement);
    let scalar_per_lane = map_lanes_binary_imm(&vn, imm as u64, arrangement, |a, b| a.bvashr(b));
    let neon_result = encode_neon_sshr(arrangement, &vn, imm);

    ProofObligation {
        name: format!("Vectorize: ScalarSshr -> NEON SSHR.{} #imm={}", label, imm),
        tmir_expr: scalar_per_lane,
        aarch64_expr: neon_result,
        inputs: vector_inputs("vn", arrangement),
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: scalar sshr -> NEON SSHR.4S #1 (128-bit, 4x32-bit lanes, shift by 1).
pub fn proof_vectorize_sshr_4s() -> ProofObligation {
    proof_vectorize_sshr(VectorArrangement::S4, 1, "4S")
}

/// Proof: scalar sshr -> NEON SSHR.8H #4 (128-bit, 8x16-bit lanes, shift by 4).
pub fn proof_vectorize_sshr_8h() -> ProofObligation {
    proof_vectorize_sshr(VectorArrangement::H8, 4, "8H")
}

// ---------------------------------------------------------------------------
// Aggregate: all vectorization lowering proofs
// ---------------------------------------------------------------------------

/// Return all 22 vectorization lowering proof obligations.
///
/// 11 operations x 2 arrangements (one 64-bit or primary, one 128-bit) = 22 proofs.
///
/// These validate every integer/bitwise mapping in `scalar_to_neon_op()`.
/// FADD and FMUL are deferred pending FP semantic encoders in `neon_semantics.rs`.
pub fn all_vectorization_proofs() -> Vec<ProofObligation> {
    vec![
        // Arithmetic (4 ops x 2 arrangements = 8 proofs)
        proof_vectorize_add_4s(),
        proof_vectorize_add_2d(),
        proof_vectorize_sub_4s(),
        proof_vectorize_sub_8h(),
        proof_vectorize_mul_4s(),
        proof_vectorize_mul_16b(),
        proof_vectorize_neg_4s(),
        proof_vectorize_neg_2d(),
        // Bitwise (4 ops x 2 arrangements = 8 proofs)
        proof_vectorize_and_8b(),
        proof_vectorize_and_16b(),
        proof_vectorize_orr_8b(),
        proof_vectorize_orr_16b(),
        proof_vectorize_eor_8b(),
        proof_vectorize_eor_16b(),
        proof_vectorize_bic_8b(),
        proof_vectorize_bic_16b(),
        // Shifts (3 ops x 2 arrangements = 6 proofs)
        proof_vectorize_shl_4s(),
        proof_vectorize_shl_8h(),
        proof_vectorize_ushr_4s(),
        proof_vectorize_ushr_2d(),
        proof_vectorize_sshr_4s(),
        proof_vectorize_sshr_8h(),
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
    // Vectorization ADD proofs
    // =======================================================================

    #[test]
    fn test_vectorize_add_4s() {
        assert_valid(&proof_vectorize_add_4s());
    }

    #[test]
    fn test_vectorize_add_2d() {
        assert_valid(&proof_vectorize_add_2d());
    }

    // =======================================================================
    // Vectorization SUB proofs
    // =======================================================================

    #[test]
    fn test_vectorize_sub_4s() {
        assert_valid(&proof_vectorize_sub_4s());
    }

    #[test]
    fn test_vectorize_sub_8h() {
        assert_valid(&proof_vectorize_sub_8h());
    }

    // =======================================================================
    // Vectorization MUL proofs
    // =======================================================================

    #[test]
    fn test_vectorize_mul_4s() {
        assert_valid(&proof_vectorize_mul_4s());
    }

    #[test]
    fn test_vectorize_mul_16b() {
        assert_valid(&proof_vectorize_mul_16b());
    }

    // =======================================================================
    // Vectorization NEG proofs
    // =======================================================================

    #[test]
    fn test_vectorize_neg_4s() {
        assert_valid(&proof_vectorize_neg_4s());
    }

    #[test]
    fn test_vectorize_neg_2d() {
        assert_valid(&proof_vectorize_neg_2d());
    }

    // =======================================================================
    // Vectorization AND proofs
    // =======================================================================

    #[test]
    fn test_vectorize_and_8b() {
        assert_valid(&proof_vectorize_and_8b());
    }

    #[test]
    fn test_vectorize_and_16b() {
        assert_valid(&proof_vectorize_and_16b());
    }

    // =======================================================================
    // Vectorization ORR proofs
    // =======================================================================

    #[test]
    fn test_vectorize_orr_8b() {
        assert_valid(&proof_vectorize_orr_8b());
    }

    #[test]
    fn test_vectorize_orr_16b() {
        assert_valid(&proof_vectorize_orr_16b());
    }

    // =======================================================================
    // Vectorization EOR proofs
    // =======================================================================

    #[test]
    fn test_vectorize_eor_8b() {
        assert_valid(&proof_vectorize_eor_8b());
    }

    #[test]
    fn test_vectorize_eor_16b() {
        assert_valid(&proof_vectorize_eor_16b());
    }

    // =======================================================================
    // Vectorization BIC proofs
    // =======================================================================

    #[test]
    fn test_vectorize_bic_8b() {
        assert_valid(&proof_vectorize_bic_8b());
    }

    #[test]
    fn test_vectorize_bic_16b() {
        assert_valid(&proof_vectorize_bic_16b());
    }

    // =======================================================================
    // Vectorization SHL proofs
    // =======================================================================

    #[test]
    fn test_vectorize_shl_4s() {
        assert_valid(&proof_vectorize_shl_4s());
    }

    #[test]
    fn test_vectorize_shl_8h() {
        assert_valid(&proof_vectorize_shl_8h());
    }

    // =======================================================================
    // Vectorization USHR proofs
    // =======================================================================

    #[test]
    fn test_vectorize_ushr_4s() {
        assert_valid(&proof_vectorize_ushr_4s());
    }

    #[test]
    fn test_vectorize_ushr_2d() {
        assert_valid(&proof_vectorize_ushr_2d());
    }

    // =======================================================================
    // Vectorization SSHR proofs
    // =======================================================================

    #[test]
    fn test_vectorize_sshr_4s() {
        assert_valid(&proof_vectorize_sshr_4s());
    }

    #[test]
    fn test_vectorize_sshr_8h() {
        assert_valid(&proof_vectorize_sshr_8h());
    }

    // =======================================================================
    // Aggregate test: all 22 vectorization proofs
    // =======================================================================

    #[test]
    fn test_all_vectorization_proofs() {
        let proofs = all_vectorization_proofs();
        assert_eq!(proofs.len(), 22, "expected 11 ops x 2 arrangements = 22 proofs");
        for obligation in &proofs {
            assert_valid(obligation);
        }
    }

    // =======================================================================
    // Negative test: wrong vectorization mapping detected
    // =======================================================================

    /// Verify that mapping scalar ADD to NEON SUB is caught as invalid.
    /// This demonstrates the proof system detects incorrect vectorization
    /// decisions.
    #[test]
    fn test_wrong_vectorization_mapping_detected() {
        let arrangement = VectorArrangement::S4;
        let vn = symbolic_vector("vn", arrangement);
        let vm = symbolic_vector("vm", arrangement);

        let mut inputs = vector_inputs("vn", arrangement);
        inputs.extend(vector_inputs("vm", arrangement));

        // Scalar says ADD per lane, but NEON does SUB -- should find counterexample.
        let obligation = ProofObligation {
            name: "WRONG: ScalarAdd -> NEON SUB.4S".to_string(),
            tmir_expr: map_lanes_binary(&vn, &vm, arrangement, |a, b| a.bvadd(b)),
            aarch64_expr: encode_neon_sub(arrangement, &vn, &vm),
            inputs,
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {} // expected
            other => panic!(
                "Expected Invalid for wrong vectorization mapping, got {:?}",
                other
            ),
        }
    }

    // =======================================================================
    // Cross-arrangement consistency: ADD.4S lanes match ADD.2S lanes
    // =======================================================================

    /// Verify that ADD with 4S arrangement (128-bit) produces the same
    /// per-lane results as ADD with 2S arrangement (64-bit) for the
    /// overlapping lower two lanes. This validates arrangement consistency.
    #[test]
    fn test_vectorize_add_arrangement_consistency() {
        // 2S (64-bit): two 32-bit lanes
        let vn_64 = SmtExpr::var("vn", 64);
        let vm_64 = SmtExpr::var("vm", 64);
        let add_2s = encode_neon_add(VectorArrangement::S2, &vn_64, &vm_64);

        // 4S (128-bit): four 32-bit lanes, lower 64 bits = same as 2S
        let vn_128 = var_128("vn128");
        let vm_128 = var_128("vm128");
        let add_4s = encode_neon_add(VectorArrangement::S4, &vn_128, &vm_128);

        // Extract lower 64 bits of the 4S result
        let lower_4s = add_4s.extract(63, 0);

        // Build proof: lower 64 bits of 4S result == full 2S result,
        // when the lower 64 bits of 128-bit inputs match the 64-bit inputs.
        let obligation = ProofObligation {
            name: "Vectorize: ADD.4S[63:0] == ADD.2S (consistency)".to_string(),
            tmir_expr: add_2s,
            aarch64_expr: lower_4s,
            inputs: vec![
                ("vn".to_string(), 64),
                ("vm".to_string(), 64),
                ("vn128_lo".to_string(), 64),
                ("vn128_hi".to_string(), 64),
                ("vm128_lo".to_string(), 64),
                ("vm128_hi".to_string(), 64),
            ],
            // Precondition: lower halves must match
            preconditions: vec![
                vn_64.clone().eq_expr(SmtExpr::var("vn128_lo", 64)),
                vm_64.clone().eq_expr(SmtExpr::var("vm128_lo", 64)),
            ],
            fp_inputs: vec![],
        };

        assert_valid(&obligation);
    }
}
