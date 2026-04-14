// llvm2-verify/peephole_proofs.rs - SMT proofs for peephole optimization rules
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Proves that each peephole optimization in llvm2-opt/peephole.rs preserves
// semantics. Each rule transforms one instruction into MOV (identity copy),
// so we prove: forall inputs, LHS_semantics(inputs) == RHS_semantics(inputs).
//
// Technique: Alive2-style (PLDI 2021). For each rule, encode LHS and RHS
// as SMT bitvector expressions and check `NOT(LHS == RHS)` for UNSAT.
// If UNSAT, the optimization is proven correct for all inputs.
//
// Reference: crates/llvm2-opt/src/peephole.rs

//! SMT proofs for AArch64 peephole optimization identity rules.
//!
//! Each proof corresponds to a peephole rule in `llvm2-opt::peephole`:
//!
//! | Rule | LHS | RHS | Proof |
//! |------|-----|-----|-------|
//! | Identity add | `ADD Xd, Xn, #0` | `MOV Xd, Xn` | [`proof_add_zero_identity`] |
//! | Identity sub | `SUB Xd, Xn, #0` | `MOV Xd, Xn` | [`proof_sub_zero_identity`] |
//! | Identity mul | `MUL Xd, Xn, #1` | `MOV Xd, Xn` | [`proof_mul_one_identity`] |
//! | Identity LSL | `LSL Xd, Xn, #0` | `MOV Xd, Xn` | [`proof_lsl_zero_identity`] |
//! | Identity LSR | `LSR Xd, Xn, #0` | `MOV Xd, Xn` | [`proof_lsr_zero_identity`] |
//! | Identity ASR | `ASR Xd, Xn, #0` | `MOV Xd, Xn` | [`proof_asr_zero_identity`] |
//! | OR self | `ORR Xd, Xn, Xn` | `MOV Xd, Xn` | [`proof_orr_self_identity`] |
//! | AND self | `AND Xd, Xn, Xn` | `MOV Xd, Xn` | [`proof_and_self_identity`] |
//! | XOR zero | `EOR Xd, Xn, #0` | `MOV Xd, Xn` | [`proof_eor_zero_identity`] |

use crate::lowering_proof::ProofObligation;
use crate::smt::SmtExpr;

// ---------------------------------------------------------------------------
// Semantic encoding helpers
// ---------------------------------------------------------------------------

/// Encode `ADD Xd, Xn, #imm` semantics: `Xd = Xn + imm`.
fn encode_add_ri(xn: SmtExpr, imm: u64, width: u32) -> SmtExpr {
    xn.bvadd(SmtExpr::bv_const(imm, width))
}

/// Encode `SUB Xd, Xn, #imm` semantics: `Xd = Xn - imm`.
fn encode_sub_ri(xn: SmtExpr, imm: u64, width: u32) -> SmtExpr {
    xn.bvsub(SmtExpr::bv_const(imm, width))
}

/// Encode `MUL Xd, Xn, Xm` semantics: `Xd = Xn * Xm`.
/// For the identity proof, Xm is the constant 1.
fn encode_mul_ri(xn: SmtExpr, imm: u64, width: u32) -> SmtExpr {
    xn.bvmul(SmtExpr::bv_const(imm, width))
}

/// Encode `LSL Xd, Xn, #imm` semantics: `Xd = Xn << imm`.
fn encode_lsl_ri(xn: SmtExpr, imm: u64, width: u32) -> SmtExpr {
    xn.bvshl(SmtExpr::bv_const(imm, width))
}

/// Encode `LSR Xd, Xn, #imm` semantics: `Xd = Xn >> imm` (logical).
fn encode_lsr_ri(xn: SmtExpr, imm: u64, width: u32) -> SmtExpr {
    xn.bvlshr(SmtExpr::bv_const(imm, width))
}

/// Encode `ASR Xd, Xn, #imm` semantics: `Xd = Xn >>a imm` (arithmetic).
fn encode_asr_ri(xn: SmtExpr, imm: u64, width: u32) -> SmtExpr {
    xn.bvashr(SmtExpr::bv_const(imm, width))
}

/// Encode `ORR Xd, Xn, Xm` semantics: `Xd = Xn | Xm`.
fn encode_orr_rr(xn: SmtExpr, xm: SmtExpr) -> SmtExpr {
    xn.bvor(xm)
}

/// Encode `AND Xd, Xn, Xm` semantics: `Xd = Xn & Xm`.
fn encode_and_rr(xn: SmtExpr, xm: SmtExpr) -> SmtExpr {
    xn.bvand(xm)
}

/// Encode `EOR Xd, Xn, #imm` semantics: `Xd = Xn ^ imm`.
fn encode_eor_ri(xn: SmtExpr, imm: u64, width: u32) -> SmtExpr {
    xn.bvxor(SmtExpr::bv_const(imm, width))
}

/// Encode `MOV Xd, Xn` semantics: `Xd = Xn` (identity).
///
/// MOV is simply the input value — the destination equals the source.
fn encode_mov(xn: SmtExpr) -> SmtExpr {
    xn
}

// ---------------------------------------------------------------------------
// Proof obligations for each peephole identity rule
// ---------------------------------------------------------------------------

/// Proof: `ADD Xd, Xn, #0` is equivalent to `MOV Xd, Xn`.
///
/// Theorem: forall xn : BV64 . xn + 0 == xn
///
/// This is the additive identity property. Proven at 64-bit width;
/// the 32-bit case follows by the same argument (width is parametric).
pub fn proof_add_zero_identity() -> ProofObligation {
    let width = 64;
    let xn = SmtExpr::var("xn", width);

    ProofObligation {
        name: "Peephole: ADD Xd, Xn, #0 ≡ MOV Xd, Xn".to_string(),
        tmir_expr: encode_add_ri(xn.clone(), 0, width),
        aarch64_expr: encode_mov(xn),
        inputs: vec![("xn".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `SUB Xd, Xn, #0` is equivalent to `MOV Xd, Xn`.
///
/// Theorem: forall xn : BV64 . xn - 0 == xn
///
/// Subtractive identity.
pub fn proof_sub_zero_identity() -> ProofObligation {
    let width = 64;
    let xn = SmtExpr::var("xn", width);

    ProofObligation {
        name: "Peephole: SUB Xd, Xn, #0 ≡ MOV Xd, Xn".to_string(),
        tmir_expr: encode_sub_ri(xn.clone(), 0, width),
        aarch64_expr: encode_mov(xn),
        inputs: vec![("xn".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `MUL Xd, Xn, #1` is equivalent to `MOV Xd, Xn`.
///
/// Theorem: forall xn : BV64 . xn * 1 == xn
///
/// Multiplicative identity. Note: AArch64 MUL has no immediate form;
/// the peephole catches cases where constant propagation has resolved
/// the second operand to 1.
pub fn proof_mul_one_identity() -> ProofObligation {
    let width = 64;
    let xn = SmtExpr::var("xn", width);

    ProofObligation {
        name: "Peephole: MUL Xd, Xn, #1 ≡ MOV Xd, Xn".to_string(),
        tmir_expr: encode_mul_ri(xn.clone(), 1, width),
        aarch64_expr: encode_mov(xn),
        inputs: vec![("xn".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `LSL Xd, Xn, #0` is equivalent to `MOV Xd, Xn`.
///
/// Theorem: forall xn : BV64 . xn << 0 == xn
///
/// Left shift by zero is identity.
pub fn proof_lsl_zero_identity() -> ProofObligation {
    let width = 64;
    let xn = SmtExpr::var("xn", width);

    ProofObligation {
        name: "Peephole: LSL Xd, Xn, #0 ≡ MOV Xd, Xn".to_string(),
        tmir_expr: encode_lsl_ri(xn.clone(), 0, width),
        aarch64_expr: encode_mov(xn),
        inputs: vec![("xn".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `LSR Xd, Xn, #0` is equivalent to `MOV Xd, Xn`.
///
/// Theorem: forall xn : BV64 . xn >>_l 0 == xn
///
/// Logical right shift by zero is identity.
pub fn proof_lsr_zero_identity() -> ProofObligation {
    let width = 64;
    let xn = SmtExpr::var("xn", width);

    ProofObligation {
        name: "Peephole: LSR Xd, Xn, #0 ≡ MOV Xd, Xn".to_string(),
        tmir_expr: encode_lsr_ri(xn.clone(), 0, width),
        aarch64_expr: encode_mov(xn),
        inputs: vec![("xn".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `ASR Xd, Xn, #0` is equivalent to `MOV Xd, Xn`.
///
/// Theorem: forall xn : BV64 . xn >>_a 0 == xn
///
/// Arithmetic right shift by zero is identity.
pub fn proof_asr_zero_identity() -> ProofObligation {
    let width = 64;
    let xn = SmtExpr::var("xn", width);

    ProofObligation {
        name: "Peephole: ASR Xd, Xn, #0 ≡ MOV Xd, Xn".to_string(),
        tmir_expr: encode_asr_ri(xn.clone(), 0, width),
        aarch64_expr: encode_mov(xn),
        inputs: vec![("xn".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `ORR Xd, Xn, Xn` is equivalent to `MOV Xd, Xn`.
///
/// Theorem: forall xn : BV64 . xn | xn == xn
///
/// Bitwise OR is idempotent.
pub fn proof_orr_self_identity() -> ProofObligation {
    let width = 64;
    let xn = SmtExpr::var("xn", width);

    ProofObligation {
        name: "Peephole: ORR Xd, Xn, Xn ≡ MOV Xd, Xn".to_string(),
        tmir_expr: encode_orr_rr(xn.clone(), xn.clone()),
        aarch64_expr: encode_mov(xn),
        inputs: vec![("xn".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `AND Xd, Xn, Xn` is equivalent to `MOV Xd, Xn`.
///
/// Theorem: forall xn : BV64 . xn & xn == xn
///
/// Bitwise AND is idempotent.
pub fn proof_and_self_identity() -> ProofObligation {
    let width = 64;
    let xn = SmtExpr::var("xn", width);

    ProofObligation {
        name: "Peephole: AND Xd, Xn, Xn ≡ MOV Xd, Xn".to_string(),
        tmir_expr: encode_and_rr(xn.clone(), xn.clone()),
        aarch64_expr: encode_mov(xn),
        inputs: vec![("xn".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `EOR Xd, Xn, #0` is equivalent to `MOV Xd, Xn`.
///
/// Theorem: forall xn : BV64 . xn ^ 0 == xn
///
/// XOR with zero is identity.
pub fn proof_eor_zero_identity() -> ProofObligation {
    let width = 64;
    let xn = SmtExpr::var("xn", width);

    ProofObligation {
        name: "Peephole: EOR Xd, Xn, #0 ≡ MOV Xd, Xn".to_string(),
        tmir_expr: encode_eor_ri(xn.clone(), 0, width),
        aarch64_expr: encode_mov(xn),
        inputs: vec![("xn".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ---------------------------------------------------------------------------
// 32-bit variants (W-register)
// ---------------------------------------------------------------------------

/// Proof: `ADD Wd, Wn, #0` is equivalent to `MOV Wd, Wn` (32-bit).
pub fn proof_add_zero_identity_w32() -> ProofObligation {
    let width = 32;
    let xn = SmtExpr::var("xn", width);

    ProofObligation {
        name: "Peephole: ADD Wd, Wn, #0 ≡ MOV Wd, Wn (32-bit)".to_string(),
        tmir_expr: encode_add_ri(xn.clone(), 0, width),
        aarch64_expr: encode_mov(xn),
        inputs: vec![("xn".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `SUB Wd, Wn, #0` is equivalent to `MOV Wd, Wn` (32-bit).
pub fn proof_sub_zero_identity_w32() -> ProofObligation {
    let width = 32;
    let xn = SmtExpr::var("xn", width);

    ProofObligation {
        name: "Peephole: SUB Wd, Wn, #0 ≡ MOV Wd, Wn (32-bit)".to_string(),
        tmir_expr: encode_sub_ri(xn.clone(), 0, width),
        aarch64_expr: encode_mov(xn),
        inputs: vec![("xn".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ---------------------------------------------------------------------------
// Aggregate accessors
// ---------------------------------------------------------------------------

/// Return all 9 core peephole identity rule proofs (64-bit).
pub fn all_peephole_proofs() -> Vec<ProofObligation> {
    vec![
        proof_add_zero_identity(),
        proof_sub_zero_identity(),
        proof_mul_one_identity(),
        proof_lsl_zero_identity(),
        proof_lsr_zero_identity(),
        proof_asr_zero_identity(),
        proof_orr_self_identity(),
        proof_and_self_identity(),
        proof_eor_zero_identity(),
    ]
}

/// Return all peephole proofs including 32-bit variants (11 total).
pub fn all_peephole_proofs_with_32bit() -> Vec<ProofObligation> {
    let mut proofs = all_peephole_proofs();
    proofs.push(proof_add_zero_identity_w32());
    proofs.push(proof_sub_zero_identity_w32());
    proofs
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lowering_proof::verify_by_evaluation;
    use crate::verify::VerificationResult;

    /// Helper: verify a proof obligation and assert it is Valid.
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

    // -----------------------------------------------------------------------
    // Individual proof tests (64-bit)
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_add_zero_identity() {
        assert_valid(&proof_add_zero_identity());
    }

    #[test]
    fn test_proof_sub_zero_identity() {
        assert_valid(&proof_sub_zero_identity());
    }

    #[test]
    fn test_proof_mul_one_identity() {
        assert_valid(&proof_mul_one_identity());
    }

    #[test]
    fn test_proof_lsl_zero_identity() {
        assert_valid(&proof_lsl_zero_identity());
    }

    #[test]
    fn test_proof_lsr_zero_identity() {
        assert_valid(&proof_lsr_zero_identity());
    }

    #[test]
    fn test_proof_asr_zero_identity() {
        assert_valid(&proof_asr_zero_identity());
    }

    #[test]
    fn test_proof_orr_self_identity() {
        assert_valid(&proof_orr_self_identity());
    }

    #[test]
    fn test_proof_and_self_identity() {
        assert_valid(&proof_and_self_identity());
    }

    #[test]
    fn test_proof_eor_zero_identity() {
        assert_valid(&proof_eor_zero_identity());
    }

    // -----------------------------------------------------------------------
    // 32-bit variant tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_add_zero_identity_w32() {
        assert_valid(&proof_add_zero_identity_w32());
    }

    #[test]
    fn test_proof_sub_zero_identity_w32() {
        assert_valid(&proof_sub_zero_identity_w32());
    }

    // -----------------------------------------------------------------------
    // Aggregate test
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_peephole_proofs() {
        for obligation in all_peephole_proofs() {
            assert_valid(&obligation);
        }
    }

    #[test]
    fn test_all_peephole_proofs_with_32bit() {
        for obligation in all_peephole_proofs_with_32bit() {
            assert_valid(&obligation);
        }
    }

    // -----------------------------------------------------------------------
    // Negative tests: verify that incorrect rules are detected
    // -----------------------------------------------------------------------

    /// Negative test: ADD Xd, Xn, #1 is NOT equivalent to MOV Xd, Xn.
    #[test]
    fn test_wrong_add_nonzero_detected() {
        let width = 8; // Use 8-bit for exhaustive verification
        let xn = SmtExpr::var("xn", width);

        let obligation = ProofObligation {
            name: "WRONG: ADD Xd, Xn, #1 ≡ MOV Xd, Xn".to_string(),
            tmir_expr: encode_add_ri(xn.clone(), 1, width),
            aarch64_expr: encode_mov(xn),
            inputs: vec![("xn".to_string(), width)],
            preconditions: vec![],
        fp_inputs: vec![],
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {} // expected
            other => panic!("Expected Invalid for wrong rule, got {:?}", other),
        }
    }

    /// Negative test: MUL Xd, Xn, #2 is NOT equivalent to MOV Xd, Xn.
    #[test]
    fn test_wrong_mul_non_one_detected() {
        let width = 8;
        let xn = SmtExpr::var("xn", width);

        let obligation = ProofObligation {
            name: "WRONG: MUL Xd, Xn, #2 ≡ MOV Xd, Xn".to_string(),
            tmir_expr: encode_mul_ri(xn.clone(), 2, width),
            aarch64_expr: encode_mov(xn),
            inputs: vec![("xn".to_string(), width)],
            preconditions: vec![],
        fp_inputs: vec![],
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {} // expected
            other => panic!("Expected Invalid for wrong rule, got {:?}", other),
        }
    }

    /// Negative test: ORR Xd, Xn, Xm (different regs) is NOT equivalent to MOV.
    #[test]
    fn test_wrong_orr_different_regs_detected() {
        let width = 8;
        let xn = SmtExpr::var("xn", width);
        let xm = SmtExpr::var("xm", width);

        let obligation = ProofObligation {
            name: "WRONG: ORR Xd, Xn, Xm ≡ MOV Xd, Xn (different regs)".to_string(),
            tmir_expr: encode_orr_rr(xn.clone(), xm),
            aarch64_expr: encode_mov(xn),
            inputs: vec![("xn".to_string(), width), ("xm".to_string(), width)],
            preconditions: vec![],
        fp_inputs: vec![],
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {} // expected
            other => panic!("Expected Invalid for wrong rule, got {:?}", other),
        }
    }

    /// Negative test: LSL Xd, Xn, #1 is NOT equivalent to MOV Xd, Xn.
    #[test]
    fn test_wrong_lsl_nonzero_detected() {
        let width = 8;
        let xn = SmtExpr::var("xn", width);

        let obligation = ProofObligation {
            name: "WRONG: LSL Xd, Xn, #1 ≡ MOV Xd, Xn".to_string(),
            tmir_expr: encode_lsl_ri(xn.clone(), 1, width),
            aarch64_expr: encode_mov(xn),
            inputs: vec![("xn".to_string(), width)],
            preconditions: vec![],
        fp_inputs: vec![],
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {} // expected
            other => panic!("Expected Invalid for wrong rule, got {:?}", other),
        }
    }

    /// Negative test: EOR Xd, Xn, #0xFF is NOT equivalent to MOV Xd, Xn.
    #[test]
    fn test_wrong_eor_nonzero_detected() {
        let width = 8;
        let xn = SmtExpr::var("xn", width);

        let obligation = ProofObligation {
            name: "WRONG: EOR Xd, Xn, #0xFF ≡ MOV Xd, Xn".to_string(),
            tmir_expr: encode_eor_ri(xn.clone(), 0xFF, width),
            aarch64_expr: encode_mov(xn),
            inputs: vec![("xn".to_string(), width)],
            preconditions: vec![],
        fp_inputs: vec![],
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {} // expected
            other => panic!("Expected Invalid for wrong rule, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // SMT-LIB2 output tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_smt2_output_add_zero() {
        let obligation = proof_add_zero_identity();
        let smt2 = obligation.to_smt2();
        assert!(smt2.contains("(set-logic QF_BV)"));
        assert!(smt2.contains("(declare-const xn (_ BitVec 64))"));
        assert!(smt2.contains("(check-sat)"));
        assert!(smt2.contains("(assert"));
    }

    #[test]
    fn test_smt2_output_orr_self() {
        let obligation = proof_orr_self_identity();
        let smt2 = obligation.to_smt2();
        assert!(smt2.contains("(set-logic QF_BV)"));
        assert!(smt2.contains("bvor"));
        assert!(smt2.contains("(check-sat)"));
    }
}
