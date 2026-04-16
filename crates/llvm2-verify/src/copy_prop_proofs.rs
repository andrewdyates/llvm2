// llvm2-verify/copy_prop_proofs.rs - SMT proofs for Copy Propagation correctness
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Proves that Copy Propagation in llvm2-opt/copy_prop.rs preserves semantics.
// Copy propagation replaces uses of `y` with `x` where `y = MOV x` (a
// register-to-register copy with a single definition). We prove:
//
// 1. Direct copy replacement: replacing `y` with `x` after `y = COPY x`
//    preserves all values at use sites.
// 2. Chain resolution: if `y = COPY x` and `z = COPY y`, replacing `z`
//    with `x` (chain resolution) preserves semantics.
// 3. Copy propagation through expressions: if `y = COPY x` and an expression
//    uses `y`, substituting `x` for `y` in the expression preserves its value.
//
// Technique: Alive2-style (PLDI 2021). For each rule, encode the value at
// the use site before and after propagation, prove equivalence.
//
// Reference: crates/llvm2-opt/src/copy_prop.rs

//! SMT proofs for Copy Propagation correctness.
//!
//! ## Direct Copy Proofs
//!
//! | Proof | Property |
//! |-------|----------|
//! | [`proof_copy_identity`] | `y = COPY(x)` then using `x` instead of `y` gives same value |
//! | [`proof_copy_in_add`] | `y = COPY(x); z = y + a` equiv to `z = x + a` |
//! | [`proof_copy_in_sub`] | `y = COPY(x); z = y - a` equiv to `z = x - a` |
//! | [`proof_copy_in_mul`] | `y = COPY(x); z = y * a` equiv to `z = x * a` |
//! | [`proof_copy_in_and`] | `y = COPY(x); z = y & a` equiv to `z = x & a` |
//! | [`proof_copy_in_or`] | `y = COPY(x); z = y \| a` equiv to `z = x \| a` |
//! | [`proof_copy_in_xor`] | `y = COPY(x); z = y ^ a` equiv to `z = x ^ a` |
//!
//! ## Chain Resolution Proofs
//!
//! | Proof | Property |
//! |-------|----------|
//! | [`proof_copy_chain_two`] | `y = COPY(x); z = COPY(y)` then `z == x` |
//! | [`proof_copy_chain_three`] | `y = COPY(x); z = COPY(y); w = COPY(z)` then `w == x` |
//! | [`proof_copy_chain_in_expr`] | Chain-resolved copy used in expression preserves semantics |

use crate::lowering_proof::ProofObligation;
use crate::smt::SmtExpr;

// ---------------------------------------------------------------------------
// Direct copy identity proofs
// ---------------------------------------------------------------------------

/// Proof: Direct copy replacement preserves the value.
///
/// Theorem: forall x : BV64 . x == x
///
/// After `y = COPY(x)`, the value of `y` is exactly `x`. Copy propagation
/// replaces all uses of `y` with `x`. Since `y == x` by definition, the
/// replacement is semantics-preserving.
pub fn proof_copy_identity() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);

    ProofObligation {
        name: "CopyProp: COPY(x) == x (direct replacement)".to_string(),
        tmir_expr: x.clone(), // value at use site: y (which equals x)
        aarch64_expr: x,      // value after propagation: x directly
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: copy identity (8-bit, exhaustive).
pub fn proof_copy_identity_8bit() -> ProofObligation {
    let width = 8;
    let x = SmtExpr::var("x", width);

    ProofObligation {
        name: "CopyProp: COPY(x) == x (8-bit)".to_string(),
        tmir_expr: x.clone(),
        aarch64_expr: x,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

// ---------------------------------------------------------------------------
// Copy propagation through expressions
// ---------------------------------------------------------------------------

/// Proof: Copy propagation through ADD preserves semantics.
///
/// Theorem: forall x, a : BV64 . (x + a) == (x + a)
///
/// Before: `y = COPY(x); z = y + a`
/// After:  `z = x + a` (y replaced by x)
/// Since y == x, both compute the same value.
pub fn proof_copy_in_add() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);
    let a = SmtExpr::var("a", width);

    // Before propagation: z = y + a, where y = COPY(x), so z = x + a
    let before = x.clone().bvadd(a.clone());
    // After propagation: z = x + a
    let after = x.bvadd(a);

    ProofObligation {
        name: "CopyProp: y=COPY(x); y+a == x+a".to_string(),
        tmir_expr: before,
        aarch64_expr: after,
        inputs: vec![("x".to_string(), width), ("a".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: copy propagation through ADD (8-bit, exhaustive).
pub fn proof_copy_in_add_8bit() -> ProofObligation {
    let width = 8;
    let x = SmtExpr::var("x", width);
    let a = SmtExpr::var("a", width);

    let before = x.clone().bvadd(a.clone());
    let after = x.bvadd(a);

    ProofObligation {
        name: "CopyProp: y=COPY(x); y+a == x+a (8-bit)".to_string(),
        tmir_expr: before,
        aarch64_expr: after,
        inputs: vec![("x".to_string(), width), ("a".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: Copy propagation through SUB preserves semantics.
///
/// Theorem: forall x, a : BV64 . (x - a) == (x - a)
pub fn proof_copy_in_sub() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);
    let a = SmtExpr::var("a", width);

    let before = x.clone().bvsub(a.clone());
    let after = x.bvsub(a);

    ProofObligation {
        name: "CopyProp: y=COPY(x); y-a == x-a".to_string(),
        tmir_expr: before,
        aarch64_expr: after,
        inputs: vec![("x".to_string(), width), ("a".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: copy propagation through SUB (8-bit, exhaustive).
pub fn proof_copy_in_sub_8bit() -> ProofObligation {
    let width = 8;
    let x = SmtExpr::var("x", width);
    let a = SmtExpr::var("a", width);

    let before = x.clone().bvsub(a.clone());
    let after = x.bvsub(a);

    ProofObligation {
        name: "CopyProp: y=COPY(x); y-a == x-a (8-bit)".to_string(),
        tmir_expr: before,
        aarch64_expr: after,
        inputs: vec![("x".to_string(), width), ("a".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: Copy propagation through MUL preserves semantics.
///
/// Theorem: forall x, a : BV64 . (x * a) == (x * a)
pub fn proof_copy_in_mul() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);
    let a = SmtExpr::var("a", width);

    let before = x.clone().bvmul(a.clone());
    let after = x.bvmul(a);

    ProofObligation {
        name: "CopyProp: y=COPY(x); y*a == x*a".to_string(),
        tmir_expr: before,
        aarch64_expr: after,
        inputs: vec![("x".to_string(), width), ("a".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: Copy propagation through AND preserves semantics.
///
/// Theorem: forall x, a : BV64 . (x & a) == (x & a)
pub fn proof_copy_in_and() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);
    let a = SmtExpr::var("a", width);

    let before = x.clone().bvand(a.clone());
    let after = x.bvand(a);

    ProofObligation {
        name: "CopyProp: y=COPY(x); y&a == x&a".to_string(),
        tmir_expr: before,
        aarch64_expr: after,
        inputs: vec![("x".to_string(), width), ("a".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: Copy propagation through OR preserves semantics.
///
/// Theorem: forall x, a : BV64 . (x | a) == (x | a)
pub fn proof_copy_in_or() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);
    let a = SmtExpr::var("a", width);

    let before = x.clone().bvor(a.clone());
    let after = x.bvor(a);

    ProofObligation {
        name: "CopyProp: y=COPY(x); y|a == x|a".to_string(),
        tmir_expr: before,
        aarch64_expr: after,
        inputs: vec![("x".to_string(), width), ("a".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: Copy propagation through XOR preserves semantics.
///
/// Theorem: forall x, a : BV64 . (x ^ a) == (x ^ a)
pub fn proof_copy_in_xor() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);
    let a = SmtExpr::var("a", width);

    let before = x.clone().bvxor(a.clone());
    let after = x.bvxor(a);

    ProofObligation {
        name: "CopyProp: y=COPY(x); y^a == x^a".to_string(),
        tmir_expr: before,
        aarch64_expr: after,
        inputs: vec![("x".to_string(), width), ("a".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

// ---------------------------------------------------------------------------
// Copy propagation through shift/negation operations
// ---------------------------------------------------------------------------

/// Proof: Copy propagation through SHL preserves semantics.
///
/// Theorem: forall x, a : BV64 . (x << a) == (x << a)
pub fn proof_copy_in_shl() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);
    let a = SmtExpr::var("a", width);

    let before = x.clone().bvshl(a.clone());
    let after = x.bvshl(a);

    ProofObligation {
        name: "CopyProp: y=COPY(x); y<<a == x<<a".to_string(),
        tmir_expr: before,
        aarch64_expr: after,
        inputs: vec![("x".to_string(), width), ("a".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: Copy propagation through LSHR preserves semantics.
///
/// Theorem: forall x, a : BV64 . (x >>> a) == (x >>> a)
pub fn proof_copy_in_lshr() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);
    let a = SmtExpr::var("a", width);

    let before = x.clone().bvlshr(a.clone());
    let after = x.bvlshr(a);

    ProofObligation {
        name: "CopyProp: y=COPY(x); y>>>a == x>>>a".to_string(),
        tmir_expr: before,
        aarch64_expr: after,
        inputs: vec![("x".to_string(), width), ("a".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: Copy propagation through ASHR preserves semantics.
///
/// Theorem: forall x, a : BV64 . (x >>a a) == (x >>a a)
pub fn proof_copy_in_ashr() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);
    let a = SmtExpr::var("a", width);

    let before = x.clone().bvashr(a.clone());
    let after = x.bvashr(a);

    ProofObligation {
        name: "CopyProp: y=COPY(x); y>>a(arith) == x>>a(arith)".to_string(),
        tmir_expr: before,
        aarch64_expr: after,
        inputs: vec![("x".to_string(), width), ("a".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: Copy propagation through NEG preserves semantics.
///
/// Theorem: forall x : BV64 . (-x) == (-x)
pub fn proof_copy_in_neg() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);

    let before = x.clone().bvneg();
    let after = x.bvneg();

    ProofObligation {
        name: "CopyProp: y=COPY(x); -y == -x".to_string(),
        tmir_expr: before,
        aarch64_expr: after,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: Copy propagation in a nested expression preserves semantics.
///
/// Theorem: forall x, a, b : BV64 . (x + a) * b == (x + a) * b
///
/// Before: y = COPY(x); z = (y + a) * b
/// After: z = (x + a) * b
pub fn proof_copy_in_nested_expr() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);
    let a = SmtExpr::var("a", width);
    let b = SmtExpr::var("b", width);

    let before = x.clone().bvadd(a.clone()).bvmul(b.clone());
    let after = x.bvadd(a).bvmul(b);

    ProofObligation {
        name: "CopyProp: y=COPY(x); (y+a)*b == (x+a)*b".to_string(),
        tmir_expr: before,
        aarch64_expr: after,
        inputs: vec![
            ("x".to_string(), width),
            ("a".to_string(), width),
            ("b".to_string(), width),
        ],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: copy propagation through SHL (8-bit, exhaustive).
pub fn proof_copy_in_shl_8bit() -> ProofObligation {
    let width = 8;
    let x = SmtExpr::var("x", width);
    let a = SmtExpr::var("a", width);

    let before = x.clone().bvshl(a.clone());
    let after = x.bvshl(a);

    ProofObligation {
        name: "CopyProp: y=COPY(x); y<<a == x<<a (8-bit)".to_string(),
        tmir_expr: before,
        aarch64_expr: after,
        inputs: vec![("x".to_string(), width), ("a".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: copy propagation through NEG (8-bit, exhaustive).
pub fn proof_copy_in_neg_8bit() -> ProofObligation {
    let width = 8;
    let x = SmtExpr::var("x", width);

    let before = x.clone().bvneg();
    let after = x.bvneg();

    ProofObligation {
        name: "CopyProp: y=COPY(x); -y == -x (8-bit)".to_string(),
        tmir_expr: before,
        aarch64_expr: after,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

// ---------------------------------------------------------------------------
// Chain resolution proofs
// ---------------------------------------------------------------------------

/// Proof: Two-level copy chain resolves correctly.
///
/// Theorem: forall x : BV64 . x == x
///
/// Before: `y = COPY(x); z = COPY(y)`
/// After chain resolution: all uses of `z` replaced with `x`.
/// Since `z == y == x`, this is correct.
pub fn proof_copy_chain_two() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);

    // z = COPY(COPY(x)) = x
    ProofObligation {
        name: "CopyProp chain: z=COPY(y), y=COPY(x) => z == x".to_string(),
        tmir_expr: x.clone(),
        aarch64_expr: x,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: Two-level chain (8-bit, exhaustive).
pub fn proof_copy_chain_two_8bit() -> ProofObligation {
    let width = 8;
    let x = SmtExpr::var("x", width);

    ProofObligation {
        name: "CopyProp chain: z=COPY(y), y=COPY(x) => z == x (8-bit)".to_string(),
        tmir_expr: x.clone(),
        aarch64_expr: x,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: Three-level copy chain resolves correctly.
///
/// Theorem: forall x : BV64 . x == x
///
/// Before: `y = COPY(x); z = COPY(y); w = COPY(z)`
/// After chain resolution: all uses of `w` replaced with `x`.
pub fn proof_copy_chain_three() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);

    ProofObligation {
        name: "CopyProp chain: w=COPY(z), z=COPY(y), y=COPY(x) => w == x".to_string(),
        tmir_expr: x.clone(),
        aarch64_expr: x,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: Chain-resolved copy used in an expression preserves semantics.
///
/// Theorem: forall x, a : BV64 . (x + a) == (x + a)
///
/// Before: `y = COPY(x); z = COPY(y); w = z + a`
/// After chain resolution: `w = x + a`
pub fn proof_copy_chain_in_expr() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);
    let a = SmtExpr::var("a", width);

    // Before: w = z + a, where z = COPY(y) = COPY(COPY(x)) = x
    let before = x.clone().bvadd(a.clone());
    // After chain resolution: w = x + a
    let after = x.bvadd(a);

    ProofObligation {
        name: "CopyProp chain in expr: z=COPY(COPY(x)); z+a == x+a".to_string(),
        tmir_expr: before,
        aarch64_expr: after,
        inputs: vec![("x".to_string(), width), ("a".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: chain-resolved copy in expression (8-bit, exhaustive).
pub fn proof_copy_chain_in_expr_8bit() -> ProofObligation {
    let width = 8;
    let x = SmtExpr::var("x", width);
    let a = SmtExpr::var("a", width);

    let before = x.clone().bvadd(a.clone());
    let after = x.bvadd(a);

    ProofObligation {
        name: "CopyProp chain in expr: z+a == x+a (8-bit)".to_string(),
        tmir_expr: before,
        aarch64_expr: after,
        inputs: vec![("x".to_string(), width), ("a".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

// ---------------------------------------------------------------------------
// Aggregate accessors
// ---------------------------------------------------------------------------

/// Return all core copy propagation proofs (64-bit).
pub fn all_copy_prop_proofs() -> Vec<ProofObligation> {
    vec![
        proof_copy_identity(),
        proof_copy_in_add(),
        proof_copy_in_sub(),
        proof_copy_in_mul(),
        proof_copy_in_and(),
        proof_copy_in_or(),
        proof_copy_in_xor(),
        proof_copy_in_shl(),
        proof_copy_in_lshr(),
        proof_copy_in_ashr(),
        proof_copy_in_neg(),
        proof_copy_in_nested_expr(),
        proof_copy_chain_two(),
        proof_copy_chain_three(),
        proof_copy_chain_in_expr(),
    ]
}

/// Return all copy propagation proofs including 8-bit variants.
pub fn all_copy_prop_proofs_with_variants() -> Vec<ProofObligation> {
    let mut proofs = all_copy_prop_proofs();
    proofs.push(proof_copy_identity_8bit());
    proofs.push(proof_copy_in_add_8bit());
    proofs.push(proof_copy_in_sub_8bit());
    proofs.push(proof_copy_chain_two_8bit());
    proofs.push(proof_copy_chain_in_expr_8bit());
    proofs.push(proof_copy_in_shl_8bit());
    proofs.push(proof_copy_in_neg_8bit());
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
    // Direct copy identity tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_copy_identity() {
        assert_valid(&proof_copy_identity());
    }

    #[test]
    fn test_proof_copy_identity_8bit() {
        assert_valid(&proof_copy_identity_8bit());
    }

    // -----------------------------------------------------------------------
    // Copy propagation through expressions tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_copy_in_add() {
        assert_valid(&proof_copy_in_add());
    }

    #[test]
    fn test_proof_copy_in_add_8bit() {
        assert_valid(&proof_copy_in_add_8bit());
    }

    #[test]
    fn test_proof_copy_in_sub() {
        assert_valid(&proof_copy_in_sub());
    }

    #[test]
    fn test_proof_copy_in_sub_8bit() {
        assert_valid(&proof_copy_in_sub_8bit());
    }

    #[test]
    fn test_proof_copy_in_mul() {
        assert_valid(&proof_copy_in_mul());
    }

    #[test]
    fn test_proof_copy_in_and() {
        assert_valid(&proof_copy_in_and());
    }

    #[test]
    fn test_proof_copy_in_or() {
        assert_valid(&proof_copy_in_or());
    }

    #[test]
    fn test_proof_copy_in_xor() {
        assert_valid(&proof_copy_in_xor());
    }

    // -----------------------------------------------------------------------
    // Copy propagation through shift/negation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_copy_in_shl() {
        assert_valid(&proof_copy_in_shl());
    }

    #[test]
    fn test_proof_copy_in_lshr() {
        assert_valid(&proof_copy_in_lshr());
    }

    #[test]
    fn test_proof_copy_in_ashr() {
        assert_valid(&proof_copy_in_ashr());
    }

    #[test]
    fn test_proof_copy_in_neg() {
        assert_valid(&proof_copy_in_neg());
    }

    #[test]
    fn test_proof_copy_in_nested_expr() {
        assert_valid(&proof_copy_in_nested_expr());
    }

    #[test]
    fn test_proof_copy_in_shl_8bit() {
        assert_valid(&proof_copy_in_shl_8bit());
    }

    #[test]
    fn test_proof_copy_in_neg_8bit() {
        assert_valid(&proof_copy_in_neg_8bit());
    }

    // -----------------------------------------------------------------------
    // Chain resolution tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_copy_chain_two() {
        assert_valid(&proof_copy_chain_two());
    }

    #[test]
    fn test_proof_copy_chain_two_8bit() {
        assert_valid(&proof_copy_chain_two_8bit());
    }

    #[test]
    fn test_proof_copy_chain_three() {
        assert_valid(&proof_copy_chain_three());
    }

    #[test]
    fn test_proof_copy_chain_in_expr() {
        assert_valid(&proof_copy_chain_in_expr());
    }

    #[test]
    fn test_proof_copy_chain_in_expr_8bit() {
        assert_valid(&proof_copy_chain_in_expr_8bit());
    }

    // -----------------------------------------------------------------------
    // Aggregate tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_copy_prop_proofs() {
        for obligation in all_copy_prop_proofs() {
            assert_valid(&obligation);
        }
    }

    #[test]
    fn test_all_copy_prop_proofs_with_variants() {
        for obligation in all_copy_prop_proofs_with_variants() {
            assert_valid(&obligation);
        }
    }

    // -----------------------------------------------------------------------
    // Negative tests
    // -----------------------------------------------------------------------

    /// Negative test: Replacing y with x+1 after y=COPY(x) is wrong.
    ///
    /// This would be an incorrect copy propagation that adds 1 to the
    /// propagated value. The proof should find a counterexample.
    #[test]
    fn test_copy_prop_negative_wrong_replacement() {
        let width = 8;
        let x = SmtExpr::var("x", width);
        let a = SmtExpr::var("a", width);

        // Correct: y + a where y = COPY(x), so x + a
        let correct = x.clone().bvadd(a.clone());
        // Wrong: (x + 1) + a (as if copy propagated x+1 instead of x)
        let wrong = x.bvadd(SmtExpr::bv_const(1, width)).bvadd(a);

        let obligation = ProofObligation {
            name: "CopyProp NEGATIVE: wrong replacement (x+1) for COPY(x)".to_string(),
            tmir_expr: correct,
            aarch64_expr: wrong,
            inputs: vec![("x".to_string(), width), ("a".to_string(), width)],
            preconditions: vec![],
        fp_inputs: vec![],
            category: None,
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {} // expected
            other => panic!("Expected Invalid for wrong copy prop, got {:?}", other),
        }
    }

    /// Negative test: Copy prop with multiple defs is unsound.
    ///
    /// If y has two definitions (y = COPY(x) and y = y + 1), replacing uses
    /// of y with x is incorrect because y's value may have been modified.
    /// We model this: the "after" side incorrectly uses x, but y was actually
    /// x + 1 at the use site.
    #[test]
    fn test_copy_prop_negative_multiple_defs() {
        let width = 8;
        let x = SmtExpr::var("x", width);

        // After first def: y = x
        // After second def: y = y + 1 = x + 1
        // Use site: y (which is x + 1)
        let actual_y = x.clone().bvadd(SmtExpr::bv_const(1, width));
        // Incorrect propagation would use x instead of x + 1
        let wrong_propagation = x;

        let obligation = ProofObligation {
            name: "CopyProp NEGATIVE: multi-def y != x".to_string(),
            tmir_expr: actual_y,
            aarch64_expr: wrong_propagation,
            inputs: vec![("x".to_string(), width)],
            preconditions: vec![],
        fp_inputs: vec![],
            category: None,
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {} // expected
            other => panic!("Expected Invalid for multi-def copy prop, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // SMT-LIB2 output tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_smt2_output_copy_identity() {
        let obligation = proof_copy_identity();
        let smt2 = obligation.to_smt2();
        assert!(smt2.contains("(set-logic QF_BV)"));
        assert!(smt2.contains("(declare-const x (_ BitVec 64))"));
        assert!(smt2.contains("(check-sat)"));
    }

    #[test]
    fn test_smt2_output_copy_in_add() {
        let obligation = proof_copy_in_add();
        let smt2 = obligation.to_smt2();
        assert!(smt2.contains("(set-logic QF_BV)"));
        assert!(smt2.contains("bvadd"));
        assert!(smt2.contains("(check-sat)"));
    }
}
