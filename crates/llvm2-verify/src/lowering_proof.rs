// llvm2-verify/lowering_proof.rs - Lowering rule proof obligations
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Defines proof obligations for tMIR -> AArch64 lowering rules and a
// verification harness that checks semantic equivalence.
//
// The core technique: for a lowering rule `tMIR_inst -> AArch64_inst(s)`,
// assert `NOT(tmir_result == aarch64_result)` and check for UNSAT.
// If UNSAT, the rule is proven correct for all inputs.
//
// Reference: Alive2 (PLDI 2021), designs/2026-04-13-verification-architecture.md

//! Proof obligations for lowering rule verification.
//!
//! A [`ProofObligation`] pairs the tMIR-side and AArch64-side semantic
//! expressions and can be checked for equivalence using either a mock
//! solver (exhaustive/random testing) or a real SMT solver (z4).

use crate::smt::{mask, SmtExpr};
use crate::verify::VerificationResult;
use std::collections::HashMap;

/// A proof obligation asserting semantic equivalence of a lowering rule.
///
/// Given:
/// - `tmir_expr`: the tMIR instruction's semantics as an SmtExpr
/// - `aarch64_expr`: the AArch64 instruction(s) semantics as an SmtExpr
/// - `inputs`: symbolic variable names and their bitvector widths
/// - `preconditions`: optional constraints (e.g., divisor != 0)
///
/// The proof obligation is:
/// ```text
/// forall inputs satisfying preconditions:
///     tmir_expr == aarch64_expr
/// ```
///
/// To verify via SMT: assert `NOT(tmir_expr == aarch64_expr)` under
/// preconditions and check for UNSAT.
#[derive(Debug, Clone)]
pub struct ProofObligation {
    /// Human-readable rule name, e.g., "Iadd_I32 -> ADDWrr".
    pub name: String,
    /// tMIR semantics expression.
    pub tmir_expr: SmtExpr,
    /// AArch64 semantics expression.
    pub aarch64_expr: SmtExpr,
    /// Symbolic input variables: (name, bit-width).
    pub inputs: Vec<(String, u32)>,
    /// Optional preconditions that must hold (e.g., divisor != 0).
    pub preconditions: Vec<SmtExpr>,
}

impl ProofObligation {
    /// Build the negated equivalence formula for SMT solving.
    ///
    /// Returns the expression: `preconditions => NOT(tmir == aarch64)`.
    /// If this is UNSAT, the lowering is correct.
    pub fn negated_equivalence(&self) -> SmtExpr {
        let equiv = self.tmir_expr.clone().eq_expr(self.aarch64_expr.clone());
        let not_equiv = equiv.not_expr();

        if self.preconditions.is_empty() {
            not_equiv
        } else {
            // precond_1 AND precond_2 AND ... AND NOT(equiv)
            let mut combined = not_equiv;
            for pre in &self.preconditions {
                combined = pre.clone().and_expr(combined);
            }
            combined
        }
    }

    /// Serialize the proof obligation to SMT-LIB2 format (for z4 CLI).
    pub fn to_smt2(&self) -> String {
        let mut lines = Vec::new();
        lines.push("(set-logic QF_BV)".to_string());

        // Declare symbolic inputs
        for (name, width) in &self.inputs {
            lines.push(format!(
                "(declare-const {} (_ BitVec {}))",
                name, width
            ));
        }

        // Assert the negated equivalence
        let formula = self.negated_equivalence();
        lines.push(format!("(assert {})", formula));
        lines.push("(check-sat)".to_string());

        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Mock verification (concrete evaluation)
// ---------------------------------------------------------------------------

/// Verify a proof obligation by exhaustive testing for small widths
/// or random sampling for larger widths.
///
/// For widths <= 8: tests all 2^(n*inputs) combinations.
/// For widths > 8: tests random samples (default: 100_000 trials).
pub fn verify_by_evaluation(obligation: &ProofObligation) -> VerificationResult {
    let width = obligation.inputs.first().map(|(_, w)| *w).unwrap_or(32);

    if width <= 8 {
        verify_exhaustive(obligation, width)
    } else {
        verify_random(obligation, width, 100_000)
    }
}

/// Exhaustive verification for small bit-widths.
fn verify_exhaustive(obligation: &ProofObligation, width: u32) -> VerificationResult {
    let max_val = 1u64 << width;
    let num_inputs = obligation.inputs.len();

    if num_inputs == 1 {
        let name = &obligation.inputs[0].0;
        for a in 0..max_val {
            let mut env = HashMap::new();
            env.insert(name.clone(), a);
            if let Some(result) = check_single_point(obligation, &env) {
                return result;
            }
        }
    } else if num_inputs == 2 {
        let name_a = &obligation.inputs[0].0;
        let name_b = &obligation.inputs[1].0;
        for a in 0..max_val {
            for b in 0..max_val {
                let mut env = HashMap::new();
                env.insert(name_a.clone(), a);
                env.insert(name_b.clone(), b);
                if let Some(result) = check_single_point(obligation, &env) {
                    return result;
                }
            }
        }
    } else {
        return VerificationResult::Unknown {
            reason: format!("exhaustive check not implemented for {} inputs", num_inputs),
        };
    }

    VerificationResult::Valid
}

/// Random-sampling verification for larger bit-widths.
fn verify_random(obligation: &ProofObligation, width: u32, trials: u64) -> VerificationResult {
    // Simple pseudo-random: use a deterministic but well-distributed sequence.
    // We use a linear congruential generator seeded from the rule name hash.
    let mut rng_state: u64 = {
        let mut h: u64 = 0xcafe_babe_dead_beef;
        for byte in obligation.name.bytes() {
            h = h.wrapping_mul(6364136223846793005).wrapping_add(byte as u64);
        }
        h
    };

    let mask_val = mask(u64::MAX, width);

    // Always test edge cases first: 0, 1, max, midpoints.
    let edge_cases: Vec<u64> = vec![
        0,
        1,
        mask_val,
        mask_val.wrapping_sub(1),
        1u64 << (width.saturating_sub(1)),
        (1u64 << (width.saturating_sub(1))).wrapping_sub(1),
    ];

    for a_val in &edge_cases {
        for b_val in &edge_cases {
            let env = build_env(&obligation.inputs, *a_val, Some(*b_val), width);
            if let Some(result) = check_single_point(obligation, &env) {
                return result;
            }
        }
    }

    for _ in 0..trials {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let a_val = mask(rng_state, width);
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let b_val = mask(rng_state, width);

        let env = build_env(&obligation.inputs, a_val, Some(b_val), width);
        if let Some(result) = check_single_point(obligation, &env) {
            return result;
        }
    }

    VerificationResult::Valid
}

/// Build an environment from input descriptors and concrete values.
fn build_env(
    inputs: &[(String, u32)],
    a_val: u64,
    b_val: Option<u64>,
    width: u32,
) -> HashMap<String, u64> {
    let mut env = HashMap::new();
    if let Some((name, _)) = inputs.first() {
        env.insert(name.clone(), mask(a_val, width));
    }
    if let Some((name, _)) = inputs.get(1) {
        if let Some(bv) = b_val {
            env.insert(name.clone(), mask(bv, width));
        }
    }
    env
}

/// Check a single test point. Returns `Some(Invalid{..})` if counterexample found.
fn check_single_point(
    obligation: &ProofObligation,
    env: &HashMap<String, u64>,
) -> Option<VerificationResult> {
    // Check preconditions
    for pre in &obligation.preconditions {
        if !pre.eval(env).as_bool() {
            return None; // Precondition not satisfied, skip this point.
        }
    }

    let tmir_result = obligation.tmir_expr.eval(env);
    let aarch64_result = obligation.aarch64_expr.eval(env);

    if tmir_result != aarch64_result {
        let cex = format!(
            "inputs: {:?}, tmir={:?}, aarch64={:?}",
            env, tmir_result, aarch64_result
        );
        Some(VerificationResult::Invalid { counterexample: cex })
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Registry of standard lowering rule proofs
// ---------------------------------------------------------------------------

/// Build the proof obligation for: `tMIR::Iadd(I32, a, b) -> ADDWrr Wd, Wn, Wm`
pub fn proof_iadd_i32() -> ProofObligation {
    use crate::aarch64_semantics::encode_add_rr;
    use crate::tmir_semantics::encode_tmir_binop;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "Iadd_I32 -> ADDWrr".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Iadd, Type::I32, a.clone(), b.clone()),
        aarch64_expr: encode_add_rr(OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
    }
}

/// Build the proof obligation for: `tMIR::Iadd(I64, a, b) -> ADDXrr Xd, Xn, Xm`
pub fn proof_iadd_i64() -> ProofObligation {
    use crate::aarch64_semantics::encode_add_rr;
    use crate::tmir_semantics::encode_tmir_binop;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "Iadd_I64 -> ADDXrr".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Iadd, Type::I64, a.clone(), b.clone()),
        aarch64_expr: encode_add_rr(OperandSize::S64, a, b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
    }
}

/// Build the proof obligation for: `tMIR::Isub(I32, a, b) -> SUBWrr Wd, Wn, Wm`
pub fn proof_isub_i32() -> ProofObligation {
    use crate::aarch64_semantics::encode_sub_rr;
    use crate::tmir_semantics::encode_tmir_binop;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "Isub_I32 -> SUBWrr".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Isub, Type::I32, a.clone(), b.clone()),
        aarch64_expr: encode_sub_rr(OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
    }
}

/// Build the proof obligation for: `tMIR::Imul(I32, a, b) -> MULWrrr Wd, Wn, Wm`
pub fn proof_imul_i32() -> ProofObligation {
    use crate::aarch64_semantics::encode_mul_rr;
    use crate::tmir_semantics::encode_tmir_binop;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "Imul_I32 -> MULWrrr".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Imul, Type::I32, a.clone(), b.clone()),
        aarch64_expr: encode_mul_rr(OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
    }
}

/// Build the proof obligation for: `tMIR::Neg(I32, a) -> NEG Wd, Wn`
///
/// NEG is `SUB Wd, WZR, Wn`, which is two's complement negation.
/// tMIR Neg is encoded as `bvneg(a)`, AArch64 NEG is also `bvneg(a)`.
pub fn proof_neg_i32() -> ProofObligation {
    use crate::aarch64_semantics::encode_neg;
    use crate::tmir_semantics::encode_tmir_neg;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);

    ProofObligation {
        name: "Neg_I32 -> NEG Wd".to_string(),
        tmir_expr: encode_tmir_neg(Type::I32, a.clone()),
        aarch64_expr: encode_neg(OperandSize::S32, a),
        inputs: vec![("a".to_string(), 32)],
        preconditions: vec![],
    }
}

/// Return all standard arithmetic lowering rule proofs.
pub fn all_arithmetic_proofs() -> Vec<ProofObligation> {
    vec![
        proof_iadd_i32(),
        proof_iadd_i64(),
        proof_isub_i32(),
        proof_imul_i32(),
        proof_neg_i32(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: verify a proof obligation using the mock evaluator and assert Valid.
    fn assert_valid(obligation: &ProofObligation) {
        let result = verify_by_evaluation(obligation);
        match &result {
            VerificationResult::Valid => {} // expected
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

    #[test]
    fn test_proof_iadd_i32() {
        assert_valid(&proof_iadd_i32());
    }

    #[test]
    fn test_proof_iadd_i64() {
        assert_valid(&proof_iadd_i64());
    }

    #[test]
    fn test_proof_isub_i32() {
        assert_valid(&proof_isub_i32());
    }

    #[test]
    fn test_proof_imul_i32() {
        assert_valid(&proof_imul_i32());
    }

    #[test]
    fn test_proof_neg_i32() {
        assert_valid(&proof_neg_i32());
    }

    #[test]
    fn test_all_arithmetic_proofs() {
        for obligation in all_arithmetic_proofs() {
            assert_valid(&obligation);
        }
    }

    #[test]
    fn test_smt2_output() {
        let obligation = proof_iadd_i32();
        let smt2 = obligation.to_smt2();
        assert!(smt2.contains("(set-logic QF_BV)"));
        assert!(smt2.contains("(declare-const a (_ BitVec 32))"));
        assert!(smt2.contains("(declare-const b (_ BitVec 32))"));
        assert!(smt2.contains("(check-sat)"));
        assert!(smt2.contains("(assert"));
    }

    /// Negative test: verify that a deliberately wrong rule is detected.
    #[test]
    fn test_wrong_rule_detected() {
        // Claim add = sub — should find a counterexample.
        let a = SmtExpr::var("a", 8);
        let b = SmtExpr::var("b", 8);

        let obligation = ProofObligation {
            name: "WRONG: Iadd -> SUBWrr".to_string(),
            tmir_expr: a.clone().bvadd(b.clone()),   // add
            aarch64_expr: a.bvsub(b),                  // sub (wrong!)
            inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
            preconditions: vec![],
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {} // expected
            other => panic!("Expected Invalid for wrong rule, got {:?}", other),
        }
    }

    /// Test that exhaustive verification catches all 8-bit values.
    #[test]
    fn test_exhaustive_8bit_add() {
        let a = SmtExpr::var("a", 8);
        let b = SmtExpr::var("b", 8);

        let obligation = ProofObligation {
            name: "Iadd_I8 -> ADD (8-bit exhaustive)".to_string(),
            tmir_expr: a.clone().bvadd(b.clone()),
            aarch64_expr: a.bvadd(b),
            inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
            preconditions: vec![],
        };

        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid));
    }
}
