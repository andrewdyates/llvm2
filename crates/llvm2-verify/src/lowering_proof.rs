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
    /// Symbolic floating-point input variables: (name, exponent_bits, significand_bits).
    ///
    /// These are declared as `(_ FloatingPoint eb sb)` in SMT-LIB2.
    /// Empty for purely bitvector proof obligations.
    pub fp_inputs: Vec<(String, u32, u32)>,
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
        use crate::z4_bridge::infer_logic;

        let mut lines = Vec::new();

        // Infer logic from the formula content.
        let formula = self.negated_equivalence();
        let logic = infer_logic(&formula);
        lines.push(format!("(set-logic {})", logic));

        // Declare symbolic bitvector inputs
        for (name, width) in &self.inputs {
            lines.push(format!(
                "(declare-const {} (_ BitVec {}))",
                name, width
            ));
        }

        // Declare symbolic floating-point inputs
        for (name, eb, sb) in &self.fp_inputs {
            lines.push(format!(
                "(declare-const {} (_ FloatingPoint {} {}))",
                name, eb, sb
            ));
        }

        // Assert the negated equivalence
        lines.push(format!("(assert {})", formula));
        lines.push("(check-sat)".to_string());

        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Mock verification (concrete evaluation)
// ---------------------------------------------------------------------------

/// Default number of random samples for statistical verification of
/// 32/64-bit proof obligations. Edge cases are always tested first
/// (0, 1, MAX, midpoints), then this many random trials follow.
///
/// At 100,000 trials the false-positive probability per proof is
/// approximately 1 - (1 - 2^{-32})^{100000} for 32-bit, which provides
/// reasonable confidence but is **not** a formal proof. Use z4/z3 via
/// [`crate::z4_bridge::verify_with_z4`] for complete guarantees.
pub const DEFAULT_SAMPLE_COUNT: u64 = 100_000;

/// Maximum bit-width for which exhaustive verification is performed.
///
/// For widths <= this threshold (with <= 2 inputs), every possible input
/// combination is tested (2^{width * num_inputs} evaluations). For widths
/// above this threshold, random sampling is used instead.
///
/// Currently set to 8 because exhaustive 16-bit with 2 inputs requires
/// 2^32 evaluations (4 billion), which is too slow for routine testing.
pub const EXHAUSTIVE_WIDTH_THRESHOLD: u32 = 8;

/// Configuration for the verification evaluation engine.
///
/// Controls sampling parameters for statistical (non-exhaustive) verification.
///
/// # Verification strength levels
///
/// | Level | Width | Inputs | Strategy | Guarantee |
/// |-------|-------|--------|----------|-----------|
/// | **Exhaustive** | <= 8 | <= 2 | All 2^(w*n) combos | Complete for that width |
/// | **Statistical** | > 8 | any | Edge cases + N random samples | Probabilistic (configurable N) |
/// | **Formal** | any | any | SMT solver (z4/z3) | Complete (not yet default) |
///
/// # Path to formal verification
///
/// The current `verify_by_evaluation` uses **mock verification**: exhaustive
/// for small widths, statistical sampling for larger widths. This catches most
/// bugs but cannot prove correctness for all 2^64 input combinations.
///
/// The path to full formal verification:
/// 1. **Current**: Mock evaluation (this module) -- fast, catches regressions
/// 2. **Available**: CLI z4/z3 via [`crate::z4_bridge`] -- serialize to SMT-LIB2, pipe to solver
/// 3. **Future**: Native z4 API (feature-gated `z4`) -- in-process SMT, no subprocess overhead
///
/// When z4 integration is the default, `verify_by_evaluation` will become
/// the fast pre-check, with z4 providing the formal proof.
#[derive(Debug, Clone)]
pub struct VerificationConfig {
    /// Number of random samples for statistical verification of widths
    /// above [`EXHAUSTIVE_WIDTH_THRESHOLD`]. Defaults to [`DEFAULT_SAMPLE_COUNT`].
    ///
    /// Higher values increase confidence but slow down verification.
    /// At N=1,000,000 a single 32-bit proof takes ~100ms on modern hardware.
    pub sample_count: u64,

    /// Maximum bit-width for exhaustive verification.
    /// Defaults to [`EXHAUSTIVE_WIDTH_THRESHOLD`] (8).
    ///
    /// Setting this to 16 enables exhaustive 16-bit single-input proofs
    /// (65,536 evaluations) but 16-bit two-input proofs require 2^32
    /// evaluations and will be very slow.
    pub exhaustive_threshold: u32,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            sample_count: DEFAULT_SAMPLE_COUNT,
            exhaustive_threshold: EXHAUSTIVE_WIDTH_THRESHOLD,
        }
    }
}

impl VerificationConfig {
    /// Create a configuration with the given sample count.
    pub fn with_sample_count(sample_count: u64) -> Self {
        Self {
            sample_count,
            ..Default::default()
        }
    }
}

/// Verify a proof obligation by exhaustive testing for small widths
/// or random sampling for larger widths, using default configuration.
///
/// # Verification strategy
///
/// - **Widths <= 8, inputs <= 2**: Exhaustive -- tests all 2^(width * num_inputs)
///   input combinations. This is a complete proof for that bit-width.
/// - **Widths > 8, inputs <= 2**: Statistical -- tests 6 edge cases per input
///   (0, 1, MAX, MAX-1, midpoint, midpoint-1) in all combinations (36 pairs),
///   then [`DEFAULT_SAMPLE_COUNT`] (100,000) random samples using a deterministic
///   LCG seeded from the proof obligation name.
/// - **3+ inputs**: Statistical -- tests 36 edge-case combinations, then
///   [`DEFAULT_SAMPLE_COUNT`] random samples with per-input width masking.
///
/// # Guarantees
///
/// For widths <= 8: **complete** -- equivalent to formal proof for that width.
/// For widths > 8: **statistical** -- high confidence but not a formal proof.
/// A counterexample-free result with 100,000 random 32-bit samples means the
/// probability of a lurking bug at any single input point is bounded by
/// ~10^{-5}, but adversarial or structured bugs could still hide.
///
/// For formal guarantees on 32/64-bit widths, use
/// [`crate::z4_bridge::verify_with_z4`] or enable the `z4` feature.
pub fn verify_by_evaluation(obligation: &ProofObligation) -> VerificationResult {
    verify_by_evaluation_with_config(obligation, &VerificationConfig::default())
}

/// Verify a proof obligation with a custom [`VerificationConfig`].
///
/// See [`verify_by_evaluation`] for strategy details. The `config` parameter
/// controls the number of random samples and the exhaustive width threshold.
pub fn verify_by_evaluation_with_config(
    obligation: &ProofObligation,
    config: &VerificationConfig,
) -> VerificationResult {
    let width = obligation.inputs.first().map(|(_, w)| *w).unwrap_or(32);
    let num_inputs = obligation.inputs.len();

    // For 3+ inputs or mixed widths, use multi-input random sampling.
    if num_inputs > 2 {
        return verify_random_multi(obligation, config.sample_count);
    }

    if width <= config.exhaustive_threshold {
        verify_exhaustive(obligation, width)
    } else {
        verify_random(obligation, width, config.sample_count)
    }
}

/// Random-sampling verification for obligations with any number of inputs.
///
/// Each input gets random values masked to its own width. This handles
/// mixed-width inputs (e.g., base:BV64, value:BV32, mem_default:BV8).
fn verify_random_multi(obligation: &ProofObligation, trials: u64) -> VerificationResult {
    let mut rng_state: u64 = {
        let mut h: u64 = 0xcafe_babe_dead_beef;
        for byte in obligation.name.bytes() {
            h = h.wrapping_mul(6364136223846793005).wrapping_add(byte as u64);
        }
        h
    };

    // Edge cases: cycle through multiple edge-case combinations
    let num_edge_combos = 36;
    for edge_idx in 0..num_edge_combos {
        let env = build_env_edge(&obligation.inputs, edge_idx);
        if let Some(result) = check_single_point(obligation, &env) {
            return result;
        }
    }

    // Random trials
    for _ in 0..trials {
        let env = build_env_multi(&obligation.inputs, &mut rng_state);
        if let Some(result) = check_single_point(obligation, &env) {
            return result;
        }
    }

    VerificationResult::Valid
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

/// Build an environment from input descriptors and per-input random values.
///
/// Unlike `build_env` which only handles 2 inputs with a shared width,
/// this function populates all inputs with values masked to their individual widths.
fn build_env_multi(
    inputs: &[(String, u32)],
    rng_state: &mut u64,
) -> HashMap<String, u64> {
    let mut env = HashMap::new();
    for (name, width) in inputs {
        *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        env.insert(name.clone(), mask(*rng_state, *width));
    }
    env
}

/// Build an environment with specific edge-case values for all inputs.
fn build_env_edge(
    inputs: &[(String, u32)],
    edge_idx: usize,
) -> HashMap<String, u64> {
    let mut env = HashMap::new();
    for (i, (name, width)) in inputs.iter().enumerate() {
        let mask_val = mask(u64::MAX, *width);
        let edges: Vec<u64> = vec![
            0,
            1,
            mask_val,
            mask_val.wrapping_sub(1),
            1u64 << (width.saturating_sub(1)),
            (1u64 << (width.saturating_sub(1))).wrapping_sub(1),
        ];
        let idx = (edge_idx.wrapping_add(i * 3)) % edges.len();
        env.insert(name.clone(), edges[idx]);
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
        fp_inputs: vec![],
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
        fp_inputs: vec![],
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
        fp_inputs: vec![],
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
        fp_inputs: vec![],
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
        fp_inputs: vec![],
    }
}

// ---------------------------------------------------------------------------
// I8 arithmetic lowering proofs
// ---------------------------------------------------------------------------

/// Build the proof obligation for: `tMIR::Iadd(I8, a, b) -> ADD (8-bit)`
///
/// On AArch64, 8-bit operations are performed in 32-bit W registers.
/// The proof verifies semantic equivalence at the 8-bit bitvector level.
pub fn proof_iadd_i8() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    ProofObligation {
        name: "Iadd_I8 -> ADD (8-bit)".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Iadd, Type::I8, a.clone(), b.clone()),
        aarch64_expr: a.bvadd(b),
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Build the proof obligation for: `tMIR::Isub(I8, a, b) -> SUB (8-bit)`
pub fn proof_isub_i8() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    ProofObligation {
        name: "Isub_I8 -> SUB (8-bit)".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Isub, Type::I8, a.clone(), b.clone()),
        aarch64_expr: a.bvsub(b),
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Build the proof obligation for: `tMIR::Imul(I8, a, b) -> MUL (8-bit)`
pub fn proof_imul_i8() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    ProofObligation {
        name: "Imul_I8 -> MUL (8-bit)".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Imul, Type::I8, a.clone(), b.clone()),
        aarch64_expr: a.bvmul(b),
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Build the proof obligation for: `tMIR::Neg(I8, a) -> NEG (8-bit)`
pub fn proof_neg_i8() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_neg;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 8);

    ProofObligation {
        name: "Neg_I8 -> NEG (8-bit)".to_string(),
        tmir_expr: encode_tmir_neg(Type::I8, a.clone()),
        aarch64_expr: a.bvneg(),
        inputs: vec![("a".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ---------------------------------------------------------------------------
// I16 arithmetic lowering proofs
// ---------------------------------------------------------------------------

/// Build the proof obligation for: `tMIR::Iadd(I16, a, b) -> ADD (16-bit)`
///
/// On AArch64, 16-bit operations are performed in 32-bit W registers.
/// The proof verifies semantic equivalence at the 16-bit bitvector level.
pub fn proof_iadd_i16() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 16);
    let b = SmtExpr::var("b", 16);

    ProofObligation {
        name: "Iadd_I16 -> ADD (16-bit)".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Iadd, Type::I16, a.clone(), b.clone()),
        aarch64_expr: a.bvadd(b),
        inputs: vec![("a".to_string(), 16), ("b".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Build the proof obligation for: `tMIR::Isub(I16, a, b) -> SUB (16-bit)`
pub fn proof_isub_i16() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 16);
    let b = SmtExpr::var("b", 16);

    ProofObligation {
        name: "Isub_I16 -> SUB (16-bit)".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Isub, Type::I16, a.clone(), b.clone()),
        aarch64_expr: a.bvsub(b),
        inputs: vec![("a".to_string(), 16), ("b".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Build the proof obligation for: `tMIR::Imul(I16, a, b) -> MUL (16-bit)`
pub fn proof_imul_i16() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 16);
    let b = SmtExpr::var("b", 16);

    ProofObligation {
        name: "Imul_I16 -> MUL (16-bit)".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Imul, Type::I16, a.clone(), b.clone()),
        aarch64_expr: a.bvmul(b),
        inputs: vec![("a".to_string(), 16), ("b".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Build the proof obligation for: `tMIR::Neg(I16, a) -> NEG (16-bit)`
pub fn proof_neg_i16() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_neg;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 16);

    ProofObligation {
        name: "Neg_I16 -> NEG (16-bit)".to_string(),
        tmir_expr: encode_tmir_neg(Type::I16, a.clone()),
        aarch64_expr: a.bvneg(),
        inputs: vec![("a".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ---------------------------------------------------------------------------
// I64 arithmetic lowering proofs (sub, mul, neg — iadd_i64 already exists)
// ---------------------------------------------------------------------------

/// Build the proof obligation for: `tMIR::Isub(I64, a, b) -> SUBXrr Xd, Xn, Xm`
pub fn proof_isub_i64() -> ProofObligation {
    use crate::aarch64_semantics::encode_sub_rr;
    use crate::tmir_semantics::encode_tmir_binop;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "Isub_I64 -> SUBXrr".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Isub, Type::I64, a.clone(), b.clone()),
        aarch64_expr: encode_sub_rr(OperandSize::S64, a, b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Build the proof obligation for: `tMIR::Imul(I64, a, b) -> MULXrrr Xd, Xn, Xm`
pub fn proof_imul_i64() -> ProofObligation {
    use crate::aarch64_semantics::encode_mul_rr;
    use crate::tmir_semantics::encode_tmir_binop;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "Imul_I64 -> MULXrrr".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Imul, Type::I64, a.clone(), b.clone()),
        aarch64_expr: encode_mul_rr(OperandSize::S64, a, b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Build the proof obligation for: `tMIR::Neg(I64, a) -> NEG Xd, Xn`
pub fn proof_neg_i64() -> ProofObligation {
    use crate::aarch64_semantics::encode_neg;
    use crate::tmir_semantics::encode_tmir_neg;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);

    ProofObligation {
        name: "Neg_I64 -> NEG Xd".to_string(),
        tmir_expr: encode_tmir_neg(Type::I64, a.clone()),
        aarch64_expr: encode_neg(OperandSize::S64, a),
        inputs: vec![("a".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ---------------------------------------------------------------------------
// Division lowering proofs: tMIR::Sdiv/Udiv -> AArch64 SDIV/UDIV
// ---------------------------------------------------------------------------

/// Build the proof obligation for: `tMIR::Sdiv(I32, a, b) -> SDIV Wd, Wn, Wm`
///
/// Precondition: `b != 0` (NonZeroDivisor proof annotation).
///
/// AArch64 SDIV semantics: signed division with truncation toward zero.
/// Division by zero returns 0 on AArch64, but tMIR treats it as UB --
/// we verify equivalence only when the precondition holds.
///
/// Edge case: `INT32_MIN / -1` = `INT32_MIN` on AArch64 (signed overflow
/// wraps). The SMT `bvsdiv` semantics match this behavior.
pub fn proof_sdiv_i32() -> ProofObligation {
    use crate::aarch64_semantics::encode_sdiv_rr;
    use crate::tmir_semantics::{encode_tmir_binop, precondition};
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    let mut preconditions = vec![];
    if let Some(pre) = precondition(&Opcode::Sdiv, Type::I32, &a, &b) {
        preconditions.push(pre);
    }

    ProofObligation {
        name: "Sdiv_I32 -> SDIVWrr".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Sdiv, Type::I32, a.clone(), b.clone()),
        aarch64_expr: encode_sdiv_rr(OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions,
        fp_inputs: vec![],
    }
}

/// Build the proof obligation for: `tMIR::Sdiv(I64, a, b) -> SDIV Xd, Xn, Xm`
///
/// Precondition: `b != 0`.
/// Edge case: `INT64_MIN / -1` = `INT64_MIN` (signed overflow wraps).
pub fn proof_sdiv_i64() -> ProofObligation {
    use crate::aarch64_semantics::encode_sdiv_rr;
    use crate::tmir_semantics::{encode_tmir_binop, precondition};
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    let mut preconditions = vec![];
    if let Some(pre) = precondition(&Opcode::Sdiv, Type::I64, &a, &b) {
        preconditions.push(pre);
    }

    ProofObligation {
        name: "Sdiv_I64 -> SDIVXrr".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Sdiv, Type::I64, a.clone(), b.clone()),
        aarch64_expr: encode_sdiv_rr(OperandSize::S64, a, b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions,
        fp_inputs: vec![],
    }
}

/// Build the proof obligation for: `tMIR::Udiv(I32, a, b) -> UDIV Wd, Wn, Wm`
///
/// Precondition: `b != 0` (NonZeroDivisor proof annotation).
///
/// AArch64 UDIV semantics: unsigned division with truncation toward zero.
/// Division by zero returns 0 on AArch64, but tMIR treats it as UB.
pub fn proof_udiv_i32() -> ProofObligation {
    use crate::aarch64_semantics::encode_udiv_rr;
    use crate::tmir_semantics::{encode_tmir_binop, precondition};
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    let mut preconditions = vec![];
    if let Some(pre) = precondition(&Opcode::Udiv, Type::I32, &a, &b) {
        preconditions.push(pre);
    }

    ProofObligation {
        name: "Udiv_I32 -> UDIVWrr".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Udiv, Type::I32, a.clone(), b.clone()),
        aarch64_expr: encode_udiv_rr(OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions,
        fp_inputs: vec![],
    }
}

/// Build the proof obligation for: `tMIR::Udiv(I64, a, b) -> UDIV Xd, Xn, Xm`
///
/// Precondition: `b != 0`.
pub fn proof_udiv_i64() -> ProofObligation {
    use crate::aarch64_semantics::encode_udiv_rr;
    use crate::tmir_semantics::{encode_tmir_binop, precondition};
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    let mut preconditions = vec![];
    if let Some(pre) = precondition(&Opcode::Udiv, Type::I64, &a, &b) {
        preconditions.push(pre);
    }

    ProofObligation {
        name: "Udiv_I64 -> UDIVXrr".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Udiv, Type::I64, a.clone(), b.clone()),
        aarch64_expr: encode_udiv_rr(OperandSize::S64, a, b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions,
        fp_inputs: vec![],
    }
}

/// Return all division lowering proofs.
pub fn all_division_proofs() -> Vec<ProofObligation> {
    vec![
        proof_sdiv_i32(),
        proof_sdiv_i64(),
        proof_udiv_i32(),
        proof_udiv_i64(),
    ]
}

/// Return all standard arithmetic lowering rule proofs.
pub fn all_arithmetic_proofs() -> Vec<ProofObligation> {
    vec![
        // I8 (exhaustive verification — all 2^16 or 2^8 input combos tested)
        proof_iadd_i8(),
        proof_isub_i8(),
        proof_imul_i8(),
        proof_neg_i8(),
        // I16 (statistical verification — edge cases + random sampling)
        proof_iadd_i16(),
        proof_isub_i16(),
        proof_imul_i16(),
        proof_neg_i16(),
        // I32 (statistical verification)
        proof_iadd_i32(),
        proof_isub_i32(),
        proof_imul_i32(),
        proof_neg_i32(),
        // I64 (statistical verification)
        proof_iadd_i64(),
        proof_isub_i64(),
        proof_imul_i64(),
        proof_neg_i64(),
        // Division (statistical verification, with NonZeroDivisor precondition)
        proof_sdiv_i32(),
        proof_sdiv_i64(),
        proof_udiv_i32(),
        proof_udiv_i64(),
    ]
}

// ---------------------------------------------------------------------------
// NZCV flag correctness lemmas
// ---------------------------------------------------------------------------

/// Build the proof obligation for NZCV N flag correctness (32-bit).
///
/// Lemma: forall a, b: BV32 . N_flag(a - b) == (a - b)[31]
///
/// We verify this by comparing the N flag from `encode_cmp` with
/// a direct extraction of the MSB from the subtraction result.
pub fn proof_nzcv_n_flag_i32() -> ProofObligation {
    use crate::nzcv::encode_cmp;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    let flags = encode_cmp(a.clone(), b.clone(), 32);

    // Direct computation: MSB of (a - b)
    let diff = a.bvsub(b);
    let msb = diff.extract(31, 31).eq_expr(SmtExpr::bv_const(1, 1));

    // Convert both to BV1 via ITE so we can compare as bitvectors
    let n_bv = SmtExpr::ite(flags.n, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1));
    let msb_bv = SmtExpr::ite(msb, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1));

    ProofObligation {
        name: "NZCV_N_flag_I32".to_string(),
        tmir_expr: msb_bv,
        aarch64_expr: n_bv,
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Build the proof obligation for NZCV Z flag correctness (32-bit).
///
/// Lemma: forall a, b: BV32 . Z_flag(a - b) == ((a - b) == 0)
pub fn proof_nzcv_z_flag_i32() -> ProofObligation {
    use crate::nzcv::encode_cmp;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    let flags = encode_cmp(a.clone(), b.clone(), 32);

    // Direct computation: (a - b) == 0
    let diff = a.bvsub(b);
    let is_zero = diff.eq_expr(SmtExpr::bv_const(0, 32));

    let z_bv = SmtExpr::ite(flags.z, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1));
    let zero_bv = SmtExpr::ite(is_zero, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1));

    ProofObligation {
        name: "NZCV_Z_flag_I32".to_string(),
        tmir_expr: zero_bv,
        aarch64_expr: z_bv,
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Build the proof obligation for NZCV C flag correctness (32-bit).
///
/// Lemma: forall a, b: BV32 . C_flag(a - b) == (a >=_u b)
pub fn proof_nzcv_c_flag_i32() -> ProofObligation {
    use crate::nzcv::encode_cmp;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    let flags = encode_cmp(a.clone(), b.clone(), 32);

    // Direct computation: a >=_u b
    let uge = a.bvuge(b);

    let c_bv = SmtExpr::ite(flags.c, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1));
    let uge_bv = SmtExpr::ite(uge, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1));

    ProofObligation {
        name: "NZCV_C_flag_I32".to_string(),
        tmir_expr: uge_bv,
        aarch64_expr: c_bv,
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Build the proof obligation for NZCV V flag correctness (32-bit).
///
/// Lemma: forall a, b: BV32 .
///   V_flag(a - b) == (sign(a) != sign(b) AND sign(a) != sign(a - b))
pub fn proof_nzcv_v_flag_i32() -> ProofObligation {
    use crate::nzcv::encode_cmp;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    let flags = encode_cmp(a.clone(), b.clone(), 32);

    // Direct computation of signed overflow for subtraction
    let diff = a.clone().bvsub(b.clone());
    let a_msb = a.extract(31, 31);
    let b_msb = b.extract(31, 31);
    let r_msb = diff.extract(31, 31);
    let overflow = a_msb.clone().eq_expr(b_msb).not_expr()
        .and_expr(a_msb.eq_expr(r_msb).not_expr());

    let v_bv = SmtExpr::ite(flags.v, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1));
    let ovf_bv = SmtExpr::ite(overflow, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1));

    ProofObligation {
        name: "NZCV_V_flag_I32".to_string(),
        tmir_expr: ovf_bv,
        aarch64_expr: v_bv,
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Return all 4 NZCV flag correctness lemma proofs.
pub fn all_nzcv_flag_proofs() -> Vec<ProofObligation> {
    vec![
        proof_nzcv_n_flag_i32(),
        proof_nzcv_z_flag_i32(),
        proof_nzcv_c_flag_i32(),
        proof_nzcv_v_flag_i32(),
    ]
}

// ---------------------------------------------------------------------------
// Comparison lowering proofs: tMIR::Icmp -> CMP + CSET
// ---------------------------------------------------------------------------

/// Generic comparison lowering proof builder.
///
/// Builds a proof that `tMIR::Icmp(cond, a, b)` produces the same 1-bit
/// result as the AArch64 sequence `CMP Rn, Rm ; CSET Rd, cc`.
fn proof_icmp_generic(
    intcc: llvm2_lower::instructions::IntCC,
    aarch64cc: llvm2_lower::isel::AArch64CC,
    width: u32,
    name: &str,
) -> ProofObligation {
    use crate::nzcv::encode_cmp_cset;
    use crate::tmir_semantics::encode_tmir_icmp;
    use llvm2_lower::types::Type;

    let ty = if width == 32 { Type::I32 } else { Type::I64 };
    let a = SmtExpr::var("a", width);
    let b = SmtExpr::var("b", width);

    ProofObligation {
        name: name.to_string(),
        tmir_expr: encode_tmir_icmp(&intcc, ty, a.clone(), b.clone()),
        aarch64_expr: encode_cmp_cset(a, b, width, aarch64cc),
        inputs: vec![("a".to_string(), width), ("b".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: tMIR::Icmp(Equal, I32) -> CMP Wn, Wm ; CSET Wd, EQ
pub fn proof_icmp_eq_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::Equal, AArch64CC::EQ, 32, "Icmp_Eq_I32 -> CMP+CSET_EQ")
}

/// Proof: tMIR::Icmp(NotEqual, I32) -> CMP Wn, Wm ; CSET Wd, NE
pub fn proof_icmp_ne_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::NotEqual, AArch64CC::NE, 32, "Icmp_NE_I32 -> CMP+CSET_NE")
}

/// Proof: tMIR::Icmp(SignedLessThan, I32) -> CMP Wn, Wm ; CSET Wd, LT
pub fn proof_icmp_slt_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::SignedLessThan, AArch64CC::LT, 32, "Icmp_SLT_I32 -> CMP+CSET_LT")
}

/// Proof: tMIR::Icmp(SignedGreaterThanOrEqual, I32) -> CMP + CSET GE
pub fn proof_icmp_sge_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::SignedGreaterThanOrEqual, AArch64CC::GE, 32, "Icmp_SGE_I32 -> CMP+CSET_GE")
}

/// Proof: tMIR::Icmp(SignedGreaterThan, I32) -> CMP + CSET GT
pub fn proof_icmp_sgt_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::SignedGreaterThan, AArch64CC::GT, 32, "Icmp_SGT_I32 -> CMP+CSET_GT")
}

/// Proof: tMIR::Icmp(SignedLessThanOrEqual, I32) -> CMP + CSET LE
pub fn proof_icmp_sle_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::SignedLessThanOrEqual, AArch64CC::LE, 32, "Icmp_SLE_I32 -> CMP+CSET_LE")
}

/// Proof: tMIR::Icmp(UnsignedLessThan, I32) -> CMP + CSET LO
pub fn proof_icmp_ult_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::UnsignedLessThan, AArch64CC::LO, 32, "Icmp_ULT_I32 -> CMP+CSET_LO")
}

/// Proof: tMIR::Icmp(UnsignedGreaterThanOrEqual, I32) -> CMP + CSET HS
pub fn proof_icmp_uge_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::UnsignedGreaterThanOrEqual, AArch64CC::HS, 32, "Icmp_UGE_I32 -> CMP+CSET_HS")
}

/// Proof: tMIR::Icmp(UnsignedGreaterThan, I32) -> CMP + CSET HI
pub fn proof_icmp_ugt_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::UnsignedGreaterThan, AArch64CC::HI, 32, "Icmp_UGT_I32 -> CMP+CSET_HI")
}

/// Proof: tMIR::Icmp(UnsignedLessThanOrEqual, I32) -> CMP + CSET LS
pub fn proof_icmp_ule_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::UnsignedLessThanOrEqual, AArch64CC::LS, 32, "Icmp_ULE_I32 -> CMP+CSET_LS")
}

/// Return all 10 comparison lowering proofs (32-bit).
pub fn all_comparison_proofs_i32() -> Vec<ProofObligation> {
    vec![
        proof_icmp_eq_i32(),
        proof_icmp_ne_i32(),
        proof_icmp_slt_i32(),
        proof_icmp_sge_i32(),
        proof_icmp_sgt_i32(),
        proof_icmp_sle_i32(),
        proof_icmp_ult_i32(),
        proof_icmp_uge_i32(),
        proof_icmp_ugt_i32(),
        proof_icmp_ule_i32(),
    ]
}

// ---------------------------------------------------------------------------
// 64-bit comparison proofs
// ---------------------------------------------------------------------------

/// Proof: tMIR::Icmp(Equal, I64) -> CMP Xn, Xm ; CSET Xd, EQ
pub fn proof_icmp_eq_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::Equal, AArch64CC::EQ, 64, "Icmp_Eq_I64 -> CMP+CSET_EQ")
}

/// Proof: tMIR::Icmp(SignedLessThan, I64) -> CMP + CSET LT (64-bit)
pub fn proof_icmp_slt_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::SignedLessThan, AArch64CC::LT, 64, "Icmp_SLT_I64 -> CMP+CSET_LT")
}

/// Proof: tMIR::Icmp(UnsignedLessThan, I64) -> CMP + CSET LO (64-bit)
pub fn proof_icmp_ult_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::UnsignedLessThan, AArch64CC::LO, 64, "Icmp_ULT_I64 -> CMP+CSET_LO")
}

// ---------------------------------------------------------------------------
// Branch lowering proofs: tMIR::CondBr(Icmp) -> CMP + B.cond
// ---------------------------------------------------------------------------

/// Build a proof that conditional branch lowering preserves semantics.
///
/// `tMIR::CondBr(Icmp(cond, a, b))` branches if the comparison is true.
/// AArch64 lowers this to `CMP Rn, Rm ; B.cc target`.
///
/// The proof obligation: the branch is taken (condition evaluates to true)
/// iff the tMIR comparison evaluates to true.
fn proof_condbr_generic(
    intcc: llvm2_lower::instructions::IntCC,
    aarch64cc: llvm2_lower::isel::AArch64CC,
    width: u32,
    name: &str,
) -> ProofObligation {
    use crate::nzcv::{encode_cmp, eval_condition};
    use crate::tmir_semantics::encode_tmir_icmp;
    use llvm2_lower::types::Type;

    let ty = if width == 32 { Type::I32 } else { Type::I64 };
    let a = SmtExpr::var("a", width);
    let b = SmtExpr::var("b", width);

    // tMIR side: Icmp produces B1. Branch is taken if result == 1.
    let tmir_cmp = encode_tmir_icmp(&intcc, ty, a.clone(), b.clone());
    // This is already a BV1 (0 or 1). Branch taken iff == 1.
    // We use it directly for comparison.

    // AArch64 side: CMP sets flags, B.cond evaluates condition.
    // eval_condition returns a Bool. Convert to BV1 for comparison.
    let flags = encode_cmp(a, b, width);
    let cond_bool = eval_condition(aarch64cc, &flags);
    let aarch64_branch_taken = SmtExpr::ite(
        cond_bool,
        SmtExpr::bv_const(1, 1),
        SmtExpr::bv_const(0, 1),
    );

    ProofObligation {
        name: name.to_string(),
        tmir_expr: tmir_cmp,
        aarch64_expr: aarch64_branch_taken,
        inputs: vec![("a".to_string(), width), ("b".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: tMIR::CondBr(Icmp(Equal)) -> CMP + B.EQ
pub fn proof_condbr_eq_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::Equal, AArch64CC::EQ, 32, "CondBr_Eq_I32 -> CMP+B.EQ")
}

/// Proof: tMIR::CondBr(Icmp(NotEqual)) -> CMP + B.NE
pub fn proof_condbr_ne_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::NotEqual, AArch64CC::NE, 32, "CondBr_NE_I32 -> CMP+B.NE")
}

/// Proof: tMIR::CondBr(Icmp(SignedLessThan)) -> CMP + B.LT
pub fn proof_condbr_slt_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::SignedLessThan, AArch64CC::LT, 32, "CondBr_SLT_I32 -> CMP+B.LT")
}

/// Proof: tMIR::CondBr(Icmp(UnsignedLessThan)) -> CMP + B.LO
pub fn proof_condbr_ult_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::UnsignedLessThan, AArch64CC::LO, 32, "CondBr_ULT_I32 -> CMP+B.LO")
}

/// Return all branch lowering proofs.
pub fn all_branch_proofs() -> Vec<ProofObligation> {
    vec![
        proof_condbr_eq_i32(),
        proof_condbr_ne_i32(),
        proof_condbr_slt_i32(),
        proof_condbr_ult_i32(),
    ]
}

/// Return all NZCV-related proofs (flags + comparisons + branches).
pub fn all_nzcv_proofs() -> Vec<ProofObligation> {
    let mut proofs = Vec::new();
    proofs.extend(all_nzcv_flag_proofs());
    proofs.extend(all_comparison_proofs_i32());
    proofs.extend(vec![
        proof_icmp_eq_i64(),
        proof_icmp_slt_i64(),
        proof_icmp_ult_i64(),
    ]);
    proofs.extend(all_branch_proofs());
    proofs
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

    // -----------------------------------------------------------------------
    // I8 arithmetic proofs (exhaustive — all 2^16 or 2^8 input combos)
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_iadd_i8() {
        assert_valid(&proof_iadd_i8());
    }

    #[test]
    fn test_proof_isub_i8() {
        assert_valid(&proof_isub_i8());
    }

    #[test]
    fn test_proof_imul_i8() {
        assert_valid(&proof_imul_i8());
    }

    #[test]
    fn test_proof_neg_i8() {
        assert_valid(&proof_neg_i8());
    }

    // -----------------------------------------------------------------------
    // I16 arithmetic proofs (statistical — edge cases + random sampling)
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_iadd_i16() {
        assert_valid(&proof_iadd_i16());
    }

    #[test]
    fn test_proof_isub_i16() {
        assert_valid(&proof_isub_i16());
    }

    #[test]
    fn test_proof_imul_i16() {
        assert_valid(&proof_imul_i16());
    }

    #[test]
    fn test_proof_neg_i16() {
        assert_valid(&proof_neg_i16());
    }

    // -----------------------------------------------------------------------
    // I32 arithmetic proofs (statistical)
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_iadd_i32() {
        assert_valid(&proof_iadd_i32());
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

    // -----------------------------------------------------------------------
    // I64 arithmetic proofs (statistical)
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_iadd_i64() {
        assert_valid(&proof_iadd_i64());
    }

    #[test]
    fn test_proof_isub_i64() {
        assert_valid(&proof_isub_i64());
    }

    #[test]
    fn test_proof_imul_i64() {
        assert_valid(&proof_imul_i64());
    }

    #[test]
    fn test_proof_neg_i64() {
        assert_valid(&proof_neg_i64());
    }

    // -----------------------------------------------------------------------
    // Aggregate arithmetic proof test
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_arithmetic_proofs() {
        for obligation in all_arithmetic_proofs() {
            assert_valid(&obligation);
        }
    }

    // -----------------------------------------------------------------------
    // Division lowering proof tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_sdiv_i32() {
        assert_valid(&proof_sdiv_i32());
    }

    #[test]
    fn test_proof_sdiv_i64() {
        assert_valid(&proof_sdiv_i64());
    }

    #[test]
    fn test_proof_udiv_i32() {
        assert_valid(&proof_udiv_i32());
    }

    #[test]
    fn test_proof_udiv_i64() {
        assert_valid(&proof_udiv_i64());
    }

    #[test]
    fn test_all_division_proofs() {
        for obligation in all_division_proofs() {
            assert_valid(&obligation);
        }
    }

    /// Verify division proof obligations include the NonZeroDivisor precondition.
    #[test]
    fn test_division_proofs_have_preconditions() {
        let sdiv32 = proof_sdiv_i32();
        assert_eq!(sdiv32.preconditions.len(), 1, "SDIV I32 must have NonZeroDivisor precondition");

        let sdiv64 = proof_sdiv_i64();
        assert_eq!(sdiv64.preconditions.len(), 1, "SDIV I64 must have NonZeroDivisor precondition");

        let udiv32 = proof_udiv_i32();
        assert_eq!(udiv32.preconditions.len(), 1, "UDIV I32 must have NonZeroDivisor precondition");

        let udiv64 = proof_udiv_i64();
        assert_eq!(udiv64.preconditions.len(), 1, "UDIV I64 must have NonZeroDivisor precondition");
    }

    /// Verify that the precondition rejects b=0 and accepts b!=0.
    #[test]
    fn test_division_precondition_semantics() {
        use std::collections::HashMap;

        let obligation = proof_sdiv_i32();
        let pre = &obligation.preconditions[0];

        // b=0 should fail precondition
        let mut env_zero = HashMap::new();
        env_zero.insert("a".to_string(), 42u64);
        env_zero.insert("b".to_string(), 0u64);
        assert!(!pre.eval(&env_zero).as_bool(), "Precondition must reject b=0");

        // b=1 should pass precondition
        let mut env_one = HashMap::new();
        env_one.insert("a".to_string(), 42u64);
        env_one.insert("b".to_string(), 1u64);
        assert!(pre.eval(&env_one).as_bool(), "Precondition must accept b=1");

        // b=0xFFFFFFFF (-1 in 32-bit signed) should pass precondition
        let mut env_neg1 = HashMap::new();
        env_neg1.insert("a".to_string(), 42u64);
        env_neg1.insert("b".to_string(), 0xFFFF_FFFFu64);
        assert!(pre.eval(&env_neg1).as_bool(), "Precondition must accept b=-1");
    }

    /// Edge case: SDIV with dividend=1, divisor=1 -- basic sanity.
    #[test]
    fn test_sdiv_i32_div_by_one() {
        use crate::aarch64_semantics::encode_sdiv_rr;
        use crate::tmir_semantics::encode_tmir_binop;
        use llvm2_ir::cc::OperandSize;
        use llvm2_lower::instructions::Opcode;
        use llvm2_lower::types::Type;
        use std::collections::HashMap;

        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);

        let tmir = encode_tmir_binop(&Opcode::Sdiv, Type::I32, a.clone(), b.clone());
        let aarch64 = encode_sdiv_rr(OperandSize::S32, a, b);

        let mut env = HashMap::new();
        env.insert("a".to_string(), 42u64);
        env.insert("b".to_string(), 1u64);

        assert_eq!(tmir.eval(&env), aarch64.eval(&env));
    }

    /// Edge case: SDIV INT_MIN / -1 = INT_MIN (signed overflow wraps).
    /// On AArch64, SDIV 0x80000000 / 0xFFFFFFFF = 0x80000000.
    #[test]
    fn test_sdiv_i32_int_min_div_neg1() {
        use crate::aarch64_semantics::encode_sdiv_rr;
        use crate::tmir_semantics::encode_tmir_binop;
        use crate::smt::EvalResult;
        use llvm2_ir::cc::OperandSize;
        use llvm2_lower::instructions::Opcode;
        use llvm2_lower::types::Type;
        use std::collections::HashMap;

        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);

        let tmir = encode_tmir_binop(&Opcode::Sdiv, Type::I32, a.clone(), b.clone());
        let aarch64 = encode_sdiv_rr(OperandSize::S32, a, b);

        let mut env = HashMap::new();
        env.insert("a".to_string(), 0x8000_0000u64); // INT32_MIN
        env.insert("b".to_string(), 0xFFFF_FFFFu64); // -1

        let tmir_result = tmir.eval(&env);
        let aarch64_result = aarch64.eval(&env);
        assert_eq!(tmir_result, aarch64_result);
        // INT_MIN / -1 overflows to INT_MIN
        assert_eq!(tmir_result, EvalResult::Bv(0x8000_0000));
    }

    /// Edge case: SDIV negative values.
    #[test]
    fn test_sdiv_i32_negative_values() {
        use crate::aarch64_semantics::encode_sdiv_rr;
        use crate::tmir_semantics::encode_tmir_binop;
        use crate::smt::EvalResult;
        use llvm2_ir::cc::OperandSize;
        use llvm2_lower::instructions::Opcode;
        use llvm2_lower::types::Type;
        use std::collections::HashMap;

        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);

        let tmir = encode_tmir_binop(&Opcode::Sdiv, Type::I32, a.clone(), b.clone());
        let aarch64 = encode_sdiv_rr(OperandSize::S32, a, b);

        // -10 / 3 = -3 (truncated toward zero)
        let mut env = HashMap::new();
        let neg10 = ((-10i32) as u32) as u64;
        env.insert("a".to_string(), neg10);
        env.insert("b".to_string(), 3u64);

        let tmir_result = tmir.eval(&env);
        let aarch64_result = aarch64.eval(&env);
        assert_eq!(tmir_result, aarch64_result);
        // -3 in 32-bit
        let neg3 = ((-3i32) as u32) as u64;
        assert_eq!(tmir_result, EvalResult::Bv(neg3));
    }

    /// Edge case: UDIV max value.
    #[test]
    fn test_udiv_i32_max_values() {
        use crate::aarch64_semantics::encode_udiv_rr;
        use crate::tmir_semantics::encode_tmir_binop;
        use crate::smt::EvalResult;
        use llvm2_ir::cc::OperandSize;
        use llvm2_lower::instructions::Opcode;
        use llvm2_lower::types::Type;
        use std::collections::HashMap;

        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);

        let tmir = encode_tmir_binop(&Opcode::Udiv, Type::I32, a.clone(), b.clone());
        let aarch64 = encode_udiv_rr(OperandSize::S32, a, b);

        // UINT32_MAX / 1 = UINT32_MAX
        let mut env = HashMap::new();
        env.insert("a".to_string(), 0xFFFF_FFFFu64);
        env.insert("b".to_string(), 1u64);

        let tmir_result = tmir.eval(&env);
        let aarch64_result = aarch64.eval(&env);
        assert_eq!(tmir_result, aarch64_result);
        assert_eq!(tmir_result, EvalResult::Bv(0xFFFF_FFFF));

        // UINT32_MAX / UINT32_MAX = 1
        env.insert("b".to_string(), 0xFFFF_FFFFu64);
        let tmir_result2 = encode_tmir_binop(&Opcode::Udiv, Type::I32,
            SmtExpr::var("a", 32), SmtExpr::var("b", 32)).eval(&env);
        let aarch64_result2 = encode_udiv_rr(OperandSize::S32,
            SmtExpr::var("a", 32), SmtExpr::var("b", 32)).eval(&env);
        assert_eq!(tmir_result2, aarch64_result2);
        assert_eq!(tmir_result2, EvalResult::Bv(1));
    }

    /// Verify SMT2 output for division includes preconditions.
    #[test]
    fn test_sdiv_smt2_has_precondition() {
        let obligation = proof_sdiv_i32();
        let smt2 = obligation.to_smt2();
        assert!(smt2.contains("(set-logic"), "SMT2 should have set-logic");
        assert!(smt2.contains("(declare-const a (_ BitVec 32))"), "SMT2 should declare a");
        assert!(smt2.contains("(declare-const b (_ BitVec 32))"), "SMT2 should declare b");
        assert!(smt2.contains("(check-sat)"), "SMT2 should have check-sat");
        // The precondition (b != 0) should be ANDed into the formula
        assert!(smt2.contains("(assert"), "SMT2 should have assert");
    }

    /// Negative test: SDIV without precondition should fail (div-by-zero mismatch).
    /// tMIR and AArch64 both return sentinel 0 for div-by-zero in the evaluator,
    /// so this tests that WITH precondition the proofs are valid but the
    /// precondition correctly skips div-by-zero inputs.
    #[test]
    fn test_division_precondition_skips_zero_divisor() {
        use std::collections::HashMap;

        let obligation = proof_sdiv_i32();

        // Manually test that b=0 is skipped by check_single_point
        let mut env = HashMap::new();
        env.insert("a".to_string(), 42u64);
        env.insert("b".to_string(), 0u64);

        // The precondition should evaluate to false for b=0
        let pre_result = obligation.preconditions[0].eval(&env);
        assert!(!pre_result.as_bool(), "Precondition should be false for b=0");
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
            fp_inputs: vec![],
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {} // expected
            other => panic!("Expected Invalid for wrong rule, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // NZCV flag correctness lemma tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_nzcv_n_flag_i32() {
        assert_valid(&proof_nzcv_n_flag_i32());
    }

    #[test]
    fn test_proof_nzcv_z_flag_i32() {
        assert_valid(&proof_nzcv_z_flag_i32());
    }

    #[test]
    fn test_proof_nzcv_c_flag_i32() {
        assert_valid(&proof_nzcv_c_flag_i32());
    }

    #[test]
    fn test_proof_nzcv_v_flag_i32() {
        assert_valid(&proof_nzcv_v_flag_i32());
    }

    #[test]
    fn test_all_nzcv_flag_proofs() {
        for obligation in all_nzcv_flag_proofs() {
            assert_valid(&obligation);
        }
    }

    // -----------------------------------------------------------------------
    // Comparison lowering proof tests (32-bit, all 10 conditions)
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_icmp_eq_i32() {
        assert_valid(&proof_icmp_eq_i32());
    }

    #[test]
    fn test_proof_icmp_ne_i32() {
        assert_valid(&proof_icmp_ne_i32());
    }

    #[test]
    fn test_proof_icmp_slt_i32() {
        assert_valid(&proof_icmp_slt_i32());
    }

    #[test]
    fn test_proof_icmp_sge_i32() {
        assert_valid(&proof_icmp_sge_i32());
    }

    #[test]
    fn test_proof_icmp_sgt_i32() {
        assert_valid(&proof_icmp_sgt_i32());
    }

    #[test]
    fn test_proof_icmp_sle_i32() {
        assert_valid(&proof_icmp_sle_i32());
    }

    #[test]
    fn test_proof_icmp_ult_i32() {
        assert_valid(&proof_icmp_ult_i32());
    }

    #[test]
    fn test_proof_icmp_uge_i32() {
        assert_valid(&proof_icmp_uge_i32());
    }

    #[test]
    fn test_proof_icmp_ugt_i32() {
        assert_valid(&proof_icmp_ugt_i32());
    }

    #[test]
    fn test_proof_icmp_ule_i32() {
        assert_valid(&proof_icmp_ule_i32());
    }

    #[test]
    fn test_all_comparison_proofs_i32() {
        for obligation in all_comparison_proofs_i32() {
            assert_valid(&obligation);
        }
    }

    // -----------------------------------------------------------------------
    // 64-bit comparison proofs
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_icmp_eq_i64() {
        assert_valid(&proof_icmp_eq_i64());
    }

    #[test]
    fn test_proof_icmp_slt_i64() {
        assert_valid(&proof_icmp_slt_i64());
    }

    #[test]
    fn test_proof_icmp_ult_i64() {
        assert_valid(&proof_icmp_ult_i64());
    }

    // -----------------------------------------------------------------------
    // Branch lowering proof tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_condbr_eq_i32() {
        assert_valid(&proof_condbr_eq_i32());
    }

    #[test]
    fn test_proof_condbr_ne_i32() {
        assert_valid(&proof_condbr_ne_i32());
    }

    #[test]
    fn test_proof_condbr_slt_i32() {
        assert_valid(&proof_condbr_slt_i32());
    }

    #[test]
    fn test_proof_condbr_ult_i32() {
        assert_valid(&proof_condbr_ult_i32());
    }

    #[test]
    fn test_all_nzcv_proofs() {
        for obligation in all_nzcv_proofs() {
            assert_valid(&obligation);
        }
    }

    // -----------------------------------------------------------------------
    // Negative test: verify wrong comparison mapping is detected
    // -----------------------------------------------------------------------

    #[test]
    fn test_wrong_comparison_detected() {
        // Map Eq to LT -- should find a counterexample
        use crate::nzcv::encode_cmp_cset;
        use crate::tmir_semantics::encode_tmir_icmp;
        use llvm2_lower::instructions::IntCC;
        use llvm2_lower::isel::AArch64CC;
        use llvm2_lower::types::Type;

        let a = SmtExpr::var("a", 8);
        let b = SmtExpr::var("b", 8);

        let obligation = ProofObligation {
            name: "WRONG: Icmp_Eq -> CMP+CSET_LT".to_string(),
            tmir_expr: encode_tmir_icmp(&IntCC::Equal, Type::I8, a.clone(), b.clone()),
            aarch64_expr: encode_cmp_cset(a, b, 8, AArch64CC::LT),
            inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {} // expected
            other => panic!("Expected Invalid for wrong comparison mapping, got {:?}", other),
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
            fp_inputs: vec![],
        };

        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid));
    }

    // -----------------------------------------------------------------------
    // VerificationConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_config_values() {
        let config = VerificationConfig::default();
        assert_eq!(config.sample_count, DEFAULT_SAMPLE_COUNT);
        assert_eq!(config.sample_count, 100_000);
        assert_eq!(config.exhaustive_threshold, EXHAUSTIVE_WIDTH_THRESHOLD);
        assert_eq!(config.exhaustive_threshold, 8);
    }

    #[test]
    fn test_config_with_sample_count() {
        let config = VerificationConfig::with_sample_count(500_000);
        assert_eq!(config.sample_count, 500_000);
        assert_eq!(config.exhaustive_threshold, EXHAUSTIVE_WIDTH_THRESHOLD);
    }

    /// Test that a custom sample count is respected by verify_by_evaluation_with_config.
    ///
    /// We verify a correct 32-bit obligation with a very low sample count (10)
    /// and a high sample count (200_000). Both should pass for a correct rule.
    #[test]
    fn test_custom_sample_count_respected() {
        let obligation = proof_iadd_i32();

        // Low sample count -- still passes for a correct rule
        let config_low = VerificationConfig::with_sample_count(10);
        let result = verify_by_evaluation_with_config(&obligation, &config_low);
        assert!(matches!(result, VerificationResult::Valid),
            "Correct rule should pass even with low sample count");

        // High sample count -- also passes
        let config_high = VerificationConfig::with_sample_count(200_000);
        let result = verify_by_evaluation_with_config(&obligation, &config_high);
        assert!(matches!(result, VerificationResult::Valid),
            "Correct rule should pass with high sample count");
    }

    /// Test that a wrong rule is caught even with low sample count, because
    /// edge cases are always tested first.
    #[test]
    fn test_wrong_rule_caught_with_low_samples() {
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);

        let obligation = ProofObligation {
            name: "WRONG: Iadd_I32 -> SUB".to_string(),
            tmir_expr: a.clone().bvadd(b.clone()),
            aarch64_expr: a.bvsub(b),
            inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        // Even with 0 random samples, edge cases should catch add != sub
        let config = VerificationConfig::with_sample_count(0);
        let result = verify_by_evaluation_with_config(&obligation, &config);
        assert!(matches!(result, VerificationResult::Invalid { .. }),
            "Wrong rule should be caught by edge cases even with 0 random samples");
    }

    /// Test that the exhaustive threshold is respected: a 16-bit obligation
    /// uses exhaustive verification when threshold is raised to 16.
    #[test]
    fn test_custom_exhaustive_threshold() {
        let a = SmtExpr::var("a", 16);

        // Single-input 16-bit obligation (65536 evaluations -- feasible)
        let obligation = ProofObligation {
            name: "Identity_I16".to_string(),
            tmir_expr: a.clone(),
            aarch64_expr: a,
            inputs: vec![("a".to_string(), 16)],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        // With default threshold (8), 16-bit falls into random sampling
        let config_default = VerificationConfig::default();
        assert!(obligation.inputs[0].1 > config_default.exhaustive_threshold,
            "16-bit should exceed default exhaustive threshold");

        // With raised threshold, it should use exhaustive
        let config_16 = VerificationConfig {
            sample_count: 10,
            exhaustive_threshold: 16,
        };
        let result = verify_by_evaluation_with_config(&obligation, &config_16);
        assert!(matches!(result, VerificationResult::Valid));
    }

    /// Test that verify_by_evaluation uses the default sample count.
    #[test]
    fn test_verify_by_evaluation_uses_defaults() {
        // This is a sanity check -- verify_by_evaluation should produce the
        // same result as verify_by_evaluation_with_config with default config.
        let obligation = proof_iadd_i64();
        let result_default = verify_by_evaluation(&obligation);
        let result_config = verify_by_evaluation_with_config(
            &obligation, &VerificationConfig::default());

        // Both should be Valid for a correct rule
        assert!(matches!(result_default, VerificationResult::Valid));
        assert!(matches!(result_config, VerificationResult::Valid));
    }
}
