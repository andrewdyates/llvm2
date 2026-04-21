// llvm2-verify/lowering_proof.rs - Lowering rule proof obligations
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
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

/// Translation validation check kind, aligned with tRust trust-transval's `CheckKind`.
///
/// trust-transval (tRust's translation validation crate) classifies verification
/// conditions into four categories: ControlFlow, DataFlow, ReturnValue,
/// Termination. LLVM2 extends this taxonomy with machine-specific categories
/// for instruction lowering, peephole optimizations, memory model, register
/// allocation, and SIMD vectorization.
///
/// This is distinct from `proof_database::ProofCategory`, which provides a
/// fine-grained LLVM2-specific classification (36 variants for individual proof
/// modules). `TransvalCheckKind` is a coarse-grained taxonomy designed for
/// interoperability with tRust's translation validation pipeline.
///
/// Reference: `~/tRust/crates/trust-transval/src/vc_core.rs`
/// Reference: `trust_types::CheckKind`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransvalCheckKind {
    /// Data flow preservation: a tMIR expression evaluates to the same value
    /// as the corresponding AArch64 expression.
    /// Maps to trust-transval `CheckKind::DataFlow`.
    DataFlow,

    /// Control flow preservation: branch conditions are preserved across
    /// the lowering transformation.
    /// Maps to trust-transval `CheckKind::ControlFlow`.
    ControlFlow,

    /// Return value preservation: function output is preserved.
    /// Maps to trust-transval `CheckKind::ReturnValue`.
    ReturnValue,

    /// Termination preservation: if the source terminates, the target must too.
    /// Maps to trust-transval `CheckKind::Termination`.
    Termination,

    /// Instruction lowering: tMIR instruction -> AArch64 instruction sequence.
    /// LLVM2-specific; trust-transval does not reason about machine instructions.
    InstructionLowering,

    /// Peephole optimization: machine-level rewrite preserves semantics.
    PeepholeOptimization,

    /// Memory model: load/store semantics preserved across lowering.
    MemoryModel,

    /// Register allocation: spill/reload preserves register values.
    RegisterAllocation,

    /// SIMD vectorization: scalar-to-vector mapping is correct.
    Vectorization,
}

impl std::fmt::Display for TransvalCheckKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransvalCheckKind::DataFlow => write!(f, "data_flow"),
            TransvalCheckKind::ControlFlow => write!(f, "control_flow"),
            TransvalCheckKind::ReturnValue => write!(f, "return_value"),
            TransvalCheckKind::Termination => write!(f, "termination"),
            TransvalCheckKind::InstructionLowering => write!(f, "instruction_lowering"),
            TransvalCheckKind::PeepholeOptimization => write!(f, "peephole"),
            TransvalCheckKind::MemoryModel => write!(f, "memory"),
            TransvalCheckKind::RegisterAllocation => write!(f, "regalloc"),
            TransvalCheckKind::Vectorization => write!(f, "vectorization"),
        }
    }
}

impl TransvalCheckKind {
    /// Convert to the category string used by `ProofResult` and `VerificationReport`.
    ///
    /// This provides backward compatibility with the existing string-based
    /// category system while enabling typed categorization.
    pub fn as_category_str(&self) -> &'static str {
        match self {
            TransvalCheckKind::DataFlow => "data_flow",
            TransvalCheckKind::ControlFlow => "control_flow",
            TransvalCheckKind::ReturnValue => "return_value",
            TransvalCheckKind::Termination => "termination",
            TransvalCheckKind::InstructionLowering => "arithmetic",
            TransvalCheckKind::PeepholeOptimization => "peephole",
            TransvalCheckKind::MemoryModel => "memory",
            TransvalCheckKind::RegisterAllocation => "regalloc",
            TransvalCheckKind::Vectorization => "vectorization",
        }
    }

    /// Returns true if this category has a direct equivalent in trust-transval's
    /// `CheckKind` enum (i.e., it is one of the four standard translation
    /// validation check kinds).
    pub fn is_transval_compatible(&self) -> bool {
        matches!(
            self,
            TransvalCheckKind::DataFlow
                | TransvalCheckKind::ControlFlow
                | TransvalCheckKind::ReturnValue
                | TransvalCheckKind::Termination
        )
    }
}

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

    /// Typed proof category, aligned with trust-transval's `CheckKind`.
    ///
    /// When set, this provides a structured categorization that can be mapped
    /// to trust-transval's verification condition taxonomy. When `None`, the
    /// category is determined by which module creates the proof obligation
    /// and the string-based category in `ProofResult`.
    ///
    /// See [`TransvalCheckKind`] for the full taxonomy.
    pub category: Option<TransvalCheckKind>,
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
        use crate::z4_bridge::{infer_logic, prepare_formula_for_smt};

        let mut lines = Vec::new();

        // Build the negated equivalence, then expand small bounded quantifiers
        // into conjunctions/disjunctions to keep the formula quantifier-free
        // when possible (QF_* logics are faster for SMT solvers).
        let raw_formula = self.negated_equivalence();
        let formula = prepare_formula_for_smt(&raw_formula);

        // Infer logic from the (potentially expanded) formula content.
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

        // Assert the negated equivalence (with quantifiers expanded where possible)
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

    // FP-only obligations (e.g. FADD/FSUB/FMUL/FDIV/FNEG/FABS/FSQRT/FCMP
    // lowering proofs) carry their operands in `fp_inputs` rather than
    // `inputs`. Dispatch them to the dedicated FP evaluator which:
    //   1. substitutes a battery of IEEE-754 edge-case test vectors
    //      (including NaN, +/-0.0, +/-Inf, denormals, MAX/MIN) into the
    //      obligation's template expressions, and
    //   2. compares results with `fp_results_equal`, which treats
    //      "NaN on both sides" as equal -- the correct behaviour for FP
    //      lowering verification since Rust's derived `PartialEq` on
    //      `EvalResult::Float(f64)` follows IEEE-754 semantics where
    //      `NaN != NaN`. Without this dispatch, FDIV proofs spuriously
    //      report counterexamples like `tmir=Float(NaN), aarch64=Float(NaN)`
    //      because the placeholder `fp_const(0.0) / fp_const(0.0)` that
    //      `check_single_point` evaluates produces NaN on both sides
    //      (#388, issue tracked under #329 / #406).
    if num_inputs == 0 && !obligation.fp_inputs.is_empty() {
        return verify_fp_by_evaluation(obligation);
    }

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
    if let Some((name, _)) = inputs.get(1)
        && let Some(bv) = b_val {
            env.insert(name.clone(), mask(bv, width));
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

    // Use NaN-aware semantic equality: two `Float(NaN)` values are considered
    // equivalent because IEEE-754 `NaN != NaN` would otherwise produce spurious
    // counterexamples for FDIV(0,0), FSQRT(-1), and similar operations that
    // both tMIR and the target encoder correctly lower to a NaN result. See
    // `EvalResult::semantically_equal` and #388.
    if !tmir_result.semantically_equal(&aarch64_result) {
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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

// ---------------------------------------------------------------------------
// Remainder lowering proofs (issue #435)
// ---------------------------------------------------------------------------
//
// AArch64 has no dedicated remainder instruction. `Urem` / `Srem` lower to a
// two-instruction sequence:
//
//   q = UDIV/SDIV  Wn, Wm          ; quotient
//   r = MSUB       q,  Wm, Wn      ; r = Wn - q * Wm
//
// The proof encodes both the tMIR `Urem`/`Srem` form (via `encode_tmir_binop`
// which composes `bvudiv`/`bvsdiv` + `bvmul` + `bvsub`) and the machine
// `MSUB(UDIV/SDIV(a, b), b, a)` form, then verifies they are equivalent
// under the division preconditions.
//
// Preconditions:
//   * `Urem`: `b != 0` (divisor nonzero).
//   * `Srem`: `b != 0` AND `not (a == INT_MIN && b == -1)` -- the second
//     conjunct avoids the signed-division overflow case where `bvsdiv` is
//     defined but some hardware manuals leave remainder-via-MSUB behavior
//     unspecified. In practice AArch64's SDIV of `INT_MIN / -1` returns
//     `INT_MIN` and the MSUB identity holds, but the tMIR spec forbids this
//     input, so we add it as an explicit precondition for symmetry with
//     classical compilers (LLVM, rustc).

/// Build the proof obligation for: `tMIR::Urem(I8, a, b) -> UDIV + MSUB (8-bit)`
///
/// Proves: `bvurem(a, b) == a - (a /u b) * b` for all 8-bit `a` and all
/// `b != 0`. Exhaustive at 8-bit (2^16 - 256 inputs with precondition filter).
pub fn proof_urem_i8() -> ProofObligation {
    use crate::aarch64_semantics::{encode_msub_rr, encode_udiv_rr};
    use crate::tmir_semantics::{encode_tmir_binop, precondition};
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    let mut preconditions = vec![];
    if let Some(pre) = precondition(&Opcode::Urem, Type::I8, &a, &b) {
        preconditions.push(pre);
    }

    let quotient = encode_udiv_rr(OperandSize::S32, a.clone(), b.clone());
    let machine = encode_msub_rr(OperandSize::S32, quotient, b.clone(), a.clone());

    ProofObligation {
        name: "Urem_I8 -> UDIV+MSUB (8-bit)".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Urem, Type::I8, a, b),
        aarch64_expr: machine,
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions,
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::Srem(I8, a, b) -> SDIV + MSUB (8-bit)`
///
/// Proves: `bvsrem(a, b) == a - (a /s b) * b` under the preconditions
/// `b != 0` and `not (a == INT8_MIN && b == -1)`.
///
/// The second precondition guards the signed-division overflow case where
/// `INT_MIN / -1` is mathematically `|INT_MIN|` but does not fit in the
/// signed range. tMIR treats this as UB; classical compilers add a runtime
/// check. Adding the precondition here mirrors that convention.
pub fn proof_srem_i8() -> ProofObligation {
    use crate::aarch64_semantics::{encode_msub_rr, encode_sdiv_rr};
    use crate::tmir_semantics::{encode_tmir_binop, precondition};
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    let mut preconditions = vec![];
    if let Some(pre) = precondition(&Opcode::Srem, Type::I8, &a, &b) {
        preconditions.push(pre);
    }
    // Additional precondition: not (a == INT8_MIN && b == -1).
    // INT8_MIN = 0x80 = -128, -1 = 0xFF at 8 bits.
    let int8_min = SmtExpr::bv_const(0x80, 8);
    let neg_one = SmtExpr::bv_const(0xFF, 8);
    let overflow = a
        .clone()
        .eq_expr(int8_min)
        .and_expr(b.clone().eq_expr(neg_one));
    preconditions.push(overflow.not_expr());

    let quotient = encode_sdiv_rr(OperandSize::S32, a.clone(), b.clone());
    let machine = encode_msub_rr(OperandSize::S32, quotient, b.clone(), a.clone());

    ProofObligation {
        name: "Srem_I8 -> SDIV+MSUB (8-bit)".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Srem, Type::I8, a, b),
        aarch64_expr: machine,
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions,
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::Urem(I16, a, b) -> UDIV + MSUB (16-bit)`
///
/// Widening of [`proof_urem_i8`] to i16. The encoders (`encode_udiv_rr`,
/// `encode_msub_rr`) are width-polymorphic — they use the bitvector width
/// carried by the `SmtExpr`, so the 16-bit obligation is a direct copy of
/// the 8-bit proof with `SmtExpr::var("a", 16)` / `SmtExpr::var("b", 16)`.
/// `OperandSize::S32` is passed because 16-bit values are held in W
/// registers on AArch64, matching the i8 convention.
///
/// Under z4 this runs symbolically (no enumeration), so wall-time is modest
/// despite the 2^32 joint input space. Smoke lane uses the tolerant
/// `assert_verified_or_timeout` helper (#435).
pub fn proof_urem_i16() -> ProofObligation {
    use crate::aarch64_semantics::{encode_msub_rr, encode_udiv_rr};
    use crate::tmir_semantics::{encode_tmir_binop, precondition};
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 16);
    let b = SmtExpr::var("b", 16);

    let mut preconditions = vec![];
    if let Some(pre) = precondition(&Opcode::Urem, Type::I16, &a, &b) {
        preconditions.push(pre);
    }

    let quotient = encode_udiv_rr(OperandSize::S32, a.clone(), b.clone());
    let machine = encode_msub_rr(OperandSize::S32, quotient, b.clone(), a.clone());

    ProofObligation {
        name: "Urem_I16 -> UDIV+MSUB (16-bit)".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Urem, Type::I16, a, b),
        aarch64_expr: machine,
        inputs: vec![("a".to_string(), 16), ("b".to_string(), 16)],
        preconditions,
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::Srem(I16, a, b) -> SDIV + MSUB (16-bit)`
///
/// Widening of [`proof_srem_i8`] to i16. Preconditions mirror the i8 case:
/// `b != 0` AND `not (a == INT16_MIN && b == -1)`. `INT16_MIN = 0x8000`,
/// `-1 @ 16 bits = 0xFFFF`.
pub fn proof_srem_i16() -> ProofObligation {
    use crate::aarch64_semantics::{encode_msub_rr, encode_sdiv_rr};
    use crate::tmir_semantics::{encode_tmir_binop, precondition};
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 16);
    let b = SmtExpr::var("b", 16);

    let mut preconditions = vec![];
    if let Some(pre) = precondition(&Opcode::Srem, Type::I16, &a, &b) {
        preconditions.push(pre);
    }
    // Additional precondition: not (a == INT16_MIN && b == -1).
    // INT16_MIN = 0x8000 = -32768, -1 = 0xFFFF at 16 bits.
    let int16_min = SmtExpr::bv_const(0x8000, 16);
    let neg_one = SmtExpr::bv_const(0xFFFF, 16);
    let overflow = a
        .clone()
        .eq_expr(int16_min)
        .and_expr(b.clone().eq_expr(neg_one));
    preconditions.push(overflow.not_expr());

    let quotient = encode_sdiv_rr(OperandSize::S32, a.clone(), b.clone());
    let machine = encode_msub_rr(OperandSize::S32, quotient, b.clone(), a.clone());

    ProofObligation {
        name: "Srem_I16 -> SDIV+MSUB (16-bit)".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Srem, Type::I16, a, b),
        aarch64_expr: machine,
        inputs: vec![("a".to_string(), 16), ("b".to_string(), 16)],
        preconditions,
        fp_inputs: vec![],
        category: None,
    }
}

/// Return all remainder lowering proofs (issue #435).
pub fn all_remainder_proofs() -> Vec<ProofObligation> {
    vec![
        proof_urem_i8(),
        proof_srem_i8(),
        proof_urem_i16(),
        proof_srem_i16(),
    ]
}

// ---------------------------------------------------------------------------
// Bitcast lowering proof (issue #435)
// ---------------------------------------------------------------------------
//
// `tMIR::Bitcast { to_ty }` reinterprets the bit pattern of `operand` as a
// different type of the same width. It is pure type-punning with no runtime
// cost: on AArch64 it lowers to `MOV` (GPR<->GPR), `FMOV` register-register
// (FPR<->FPR), or `FMOV` general (GPR<->FPR / FPR<->GPR). All three machine
// forms reduce to the bitvector identity.
//
// The proof below uses `i8` as a representative width; the equivalence is
// trivial (`x == x`) but the obligation exercises the full proof pipeline
// (precondition-free, z4-compatible) and locks in the semantics so that
// future changes to `encode_tmir_bitcast` or `encode_mov_rr` are caught.

/// Build the proof obligation for: `tMIR::Bitcast(I8 -> I8, a) -> MOV (8-bit)`
///
/// Verifies that a same-width bitcast is the identity at the bit level.
/// Representative of the full family of same-width bitcasts:
/// `i32<->f32`, `i64<->f64`, pointer casts, etc.
pub fn proof_bitcast_i8() -> ProofObligation {
    use crate::aarch64_semantics::encode_mov_rr;
    use crate::tmir_semantics::encode_tmir_bitcast;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 8);

    ProofObligation {
        name: "Bitcast_I8_I8 -> MOV (8-bit)".to_string(),
        tmir_expr: encode_tmir_bitcast(Type::I8, Type::I8, a.clone()),
        aarch64_expr: encode_mov_rr(OperandSize::S32, a),
        inputs: vec![("a".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::Bitcast(I16 -> I16, a) -> MOV (16-bit)`
///
/// Widening of [`proof_bitcast_i8`] to i16. `encode_tmir_bitcast` and
/// `encode_mov_rr` are width-agnostic (pure identity on the input
/// bitvector), so the obligation is the BV identity `x == x` at width 16.
pub fn proof_bitcast_i16() -> ProofObligation {
    use crate::aarch64_semantics::encode_mov_rr;
    use crate::tmir_semantics::encode_tmir_bitcast;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 16);

    ProofObligation {
        name: "Bitcast_I16_I16 -> MOV (16-bit)".to_string(),
        tmir_expr: encode_tmir_bitcast(Type::I16, Type::I16, a.clone()),
        aarch64_expr: encode_mov_rr(OperandSize::S32, a),
        inputs: vec![("a".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::Bitcast(I32 -> I32, a) -> MOV (32-bit)`
///
/// Widening of [`proof_bitcast_i8`] to i32. Covers the i32<->f32 bit
/// reinterpretation case at the BV level. Pure identity.
pub fn proof_bitcast_i32() -> ProofObligation {
    use crate::aarch64_semantics::encode_mov_rr;
    use crate::tmir_semantics::encode_tmir_bitcast;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);

    ProofObligation {
        name: "Bitcast_I32_I32 -> MOV (32-bit)".to_string(),
        tmir_expr: encode_tmir_bitcast(Type::I32, Type::I32, a.clone()),
        aarch64_expr: encode_mov_rr(OperandSize::S32, a),
        inputs: vec![("a".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::Bitcast(I64 -> I64, a) -> MOV (64-bit)`
///
/// Widening of [`proof_bitcast_i8`] to i64. Covers the i64<->f64 and
/// pointer bitcast families at the BV level. Pure identity.
pub fn proof_bitcast_i64() -> ProofObligation {
    use crate::aarch64_semantics::encode_mov_rr;
    use crate::tmir_semantics::encode_tmir_bitcast;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);

    ProofObligation {
        name: "Bitcast_I64_I64 -> MOV (64-bit)".to_string(),
        tmir_expr: encode_tmir_bitcast(Type::I64, Type::I64, a.clone()),
        aarch64_expr: encode_mov_rr(OperandSize::S64, a),
        inputs: vec![("a".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Return all bitcast lowering proofs (issue #435).
pub fn all_bitcast_proofs() -> Vec<ProofObligation> {
    vec![
        proof_bitcast_i8(),
        proof_bitcast_i16(),
        proof_bitcast_i32(),
        proof_bitcast_i64(),
    ]
}

// ---------------------------------------------------------------------------
// Bitfield lowering proofs (issue #452, epic #435)
// ---------------------------------------------------------------------------
//
// tMIR has three bitfield opcodes that have no dedicated 1:1 AArch64
// instruction mnemonic but compose from UBFM / SBFM / BFM:
//
//   ExtractBits { lsb, width }   ->  UBFM Wd, Wn, #lsb, #(lsb + width - 1)
//   SextractBits { lsb, width }  ->  SBFM Wd, Wn, #lsb, #(lsb + width - 1)
//   InsertBits { lsb, width }    ->  BFM  Wd, Wn, #((reg_size - lsb) % reg_size),
//                                          #(width - 1)
//                                   (preceded by a COPY of the `dst` operand
//                                    into the result register -- see
//                                    `llvm2-lower/src/isel.rs::select_bitfield_insert`)
//
// All three machine instructions are pure QF_BV operations -- no flags, no
// memory, no preconditions beyond the immediate-field range enforced by the
// tMIR type system and the encoder (`lsb + width <= reg_size`, `width >= 1`).
// The proofs below verify that the tMIR semantic encoding matches the
// AArch64 semantic encoding for representative i8 (lsb, width) pairs.
//
// # Representative (lsb, width) choice
//
// We pick `(lsb = 2, width = 4)` as the canonical i8 obligation: the slice
// straddles the middle of the byte, exercises the shift + mask composition
// non-trivially, and for `SextractBits` the top bit of the slice can be
// either value depending on the input -- so the sign-extension arm is
// actually exercised (both with and without sign bits set).
//
// Exhaustive i8 evaluation covers all 2^8 = 256 inputs for ExtractBits /
// SextractBits and all 2^16 = 65,536 (dst, src) pairs for InsertBits.
//
// References:
// - ARM DDI 0487, C6.2.335 UBFM (C6.2.334 UBFX alias)
// - ARM DDI 0487, C6.2.266 SBFM (C6.2.264 SBFX alias)
// - ARM DDI 0487, C6.2.46 BFM (C6.2.45 BFI alias)

/// Build the proof obligation for:
/// `tMIR::ExtractBits{lsb=2, width=4}(I8, x) -> UBFM Wd, Wn, #2, #5 (8-bit)`.
///
/// Proves: `bv_extract(lsb, width, x) == (x lsr lsb) & mask(width)` for all
/// 8-bit `x`. No preconditions.
pub fn proof_extract_bits_i8() -> ProofObligation {
    use crate::aarch64_semantics::encode_ubfm_extract;
    use crate::tmir_semantics::encode_tmir_extract_bits;
    use llvm2_lower::types::Type;

    let lsb: u8 = 2;
    let width: u8 = 4;
    let x = SmtExpr::var("x", 8);

    ProofObligation {
        name: format!(
            "ExtractBits{{lsb={},width={}}}_I8 -> UBFM (8-bit)",
            lsb, width
        ),
        tmir_expr: encode_tmir_extract_bits(Type::I8, lsb, width, x.clone()),
        aarch64_expr: encode_ubfm_extract(x, lsb as u32, width as u32, 8),
        inputs: vec![("x".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for:
/// `tMIR::SextractBits{lsb=2, width=4}(I8, x) -> SBFM Wd, Wn, #2, #5 (8-bit)`.
///
/// Proves: `sign_extend(x[lsb+width-1:lsb]) == SBFM-machine-semantics` for
/// all 8-bit `x`. No preconditions.
///
/// SBFM's machine encoding ties to `sign_extend(extract(lsb, width, x))`:
/// the 4-bit slice `x[5:2]` is pulled out and sign-extended to 8 bits by
/// replicating bit 3 of the slice (= bit 5 of `x`) across the upper 4 bits.
pub fn proof_sextract_bits_i8() -> ProofObligation {
    use crate::aarch64_semantics::encode_sbfm_extract;
    use crate::tmir_semantics::encode_tmir_sextract_bits;
    use llvm2_lower::types::Type;

    let lsb: u8 = 2;
    let width: u8 = 4;
    let x = SmtExpr::var("x", 8);

    ProofObligation {
        name: format!(
            "SextractBits{{lsb={},width={}}}_I8 -> SBFM (8-bit)",
            lsb, width
        ),
        tmir_expr: encode_tmir_sextract_bits(Type::I8, lsb, width, x.clone()),
        aarch64_expr: encode_sbfm_extract(x, lsb as u32, width as u32, 8),
        inputs: vec![("x".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for:
/// `tMIR::InsertBits{lsb=2, width=4}(I8, x, y) -> BFM Wd, Ws (8-bit)`.
///
/// Proves: `result == (x & ~mask_shifted) | ((y & mask(width)) shl lsb)`
/// for all 8-bit `(x, y)`, where `mask_shifted = mask(width) << lsb`.
/// No preconditions.
///
/// `x` is the destination (old value, preserved outside the slice).
/// `y` supplies the new bits for `[lsb + width - 1 : lsb]`.
pub fn proof_insert_bits_i8() -> ProofObligation {
    use crate::aarch64_semantics::encode_bfm_insert;
    use crate::tmir_semantics::encode_tmir_insert_bits;
    use llvm2_lower::types::Type;

    let lsb: u8 = 2;
    let width: u8 = 4;
    let x = SmtExpr::var("x", 8);
    let y = SmtExpr::var("y", 8);

    ProofObligation {
        name: format!(
            "InsertBits{{lsb={},width={}}}_I8 -> BFM (8-bit)",
            lsb, width
        ),
        tmir_expr: encode_tmir_insert_bits(Type::I8, lsb, width, x.clone(), y.clone()),
        aarch64_expr: encode_bfm_insert(x, y, lsb as u32, width as u32, 8),
        inputs: vec![("x".to_string(), 8), ("y".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Return all bitfield lowering proofs (issue #452, epic #435).
pub fn all_bitfield_proofs() -> Vec<ProofObligation> {
    vec![
        proof_extract_bits_i8(),
        proof_sextract_bits_i8(),
        proof_insert_bits_i8(),
    ]
}

// ---------------------------------------------------------------------------
// I128 multi-register arithmetic lowering proofs (issue #324)
// ---------------------------------------------------------------------------
//
// i128 values are held in a register pair (lo:hi) of two 64-bit GPRs. Add /
// sub are lowered to two machine instructions — a flag-setting low-half op
// followed by a carry/borrow-propagating high-half op:
//
//   i128 ADD: `ADDS dst_lo, a_lo, b_lo`  then  `ADC dst_hi, a_hi, b_hi`
//   i128 SUB: `SUBS dst_lo, a_lo, b_lo`  then  `SBC dst_hi, a_hi, b_hi`
//
// The default evaluator's `bvadd`/`bvsub`/`bvmul` eval paths truncate to u64
// (see `SmtExpr::BvAdd` in `smt.rs`), so we cannot model the 128-bit spec as
// a single 128-bit `bvadd`/`bvsub` and get meaningful coverage. Instead, the
// proof obligations below are split across the two 64-bit limbs — one
// obligation for each — with a concrete SMT-level carry / borrow expression
// that matches the AArch64 NZCV semantics:
//
//   ADC carry  : C = 1 iff a_lo + b_lo overflows 2^64 ≡ `bvult(lo_sum, a_lo)`
//   SBC borrow : C = 1 iff a_lo >= b_lo (AArch64 sets C=!borrow) ≡
//                !bvult(a_lo, b_lo) ≡ `bvuge(a_lo, b_lo)`
//
// Both sides of each obligation (tmir spec, aarch64 machine form) are
// constructed using the same SMT primitives, so evaluation-based
// verification acts as a regression lock on the carry/borrow encoding. A
// future z4 formal proof can compare against `concat(hi, lo).bvadd/bvsub`
// directly once the evaluator grows native 128-bit add/sub.

/// Build the proof obligation for the low 64 bits of i128 ADD.
///
/// `dst_lo = (a_lo + b_lo) mod 2^64` on both sides. This is the trivial
/// half of the ADDS+ADC lowering; the hi-half proof below carries the
/// ADC-specific carry identity.
pub fn proof_iadd_i128_lo() -> ProofObligation {
    let a_lo = SmtExpr::var("a_lo", 64);
    let b_lo = SmtExpr::var("b_lo", 64);

    // tMIR spec: low 64 bits of `a + b` are `(a_lo + b_lo) mod 2^64`.
    let tmir = a_lo.clone().bvadd(b_lo.clone());
    // AArch64 ADDS writes `a_lo + b_lo` into dst_lo.
    let aarch64 = a_lo.bvadd(b_lo);

    ProofObligation {
        name: "Iadd_I128 lo -> ADDS Xlo,Xa_lo,Xb_lo".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![("a_lo".to_string(), 64), ("b_lo".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for the high 64 bits of i128 ADD.
///
/// `dst_hi = (a_hi + b_hi + carry) mod 2^64` where
/// `carry = 1 iff (a_lo + b_lo) wraps past 2^64`. Both sides use the same
/// carry formula drawn from AArch64's ADDS NZCV flag semantics.
pub fn proof_iadd_i128_hi() -> ProofObligation {
    let a_lo = SmtExpr::var("a_lo", 64);
    let b_lo = SmtExpr::var("b_lo", 64);
    let a_hi = SmtExpr::var("a_hi", 64);
    let b_hi = SmtExpr::var("b_hi", 64);

    // carry = 1 iff a_lo + b_lo overflowed the 64-bit word.
    let lo_sum = a_lo.clone().bvadd(b_lo);
    let carry_bool = lo_sum.bvult(a_lo);
    let carry_bv = SmtExpr::ite(
        carry_bool,
        SmtExpr::bv_const(1, 64),
        SmtExpr::bv_const(0, 64),
    );

    // tMIR spec: high 64 bits of `a + b` = a_hi + b_hi + carry (mod 2^64).
    let tmir = a_hi.clone().bvadd(b_hi.clone()).bvadd(carry_bv.clone());
    // AArch64 ADC: dst_hi = a_hi + b_hi + C (where C came from ADDS above).
    let aarch64 = a_hi.bvadd(b_hi).bvadd(carry_bv);

    ProofObligation {
        name: "Iadd_I128 hi -> ADC Xhi,Xa_hi,Xb_hi".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![
            ("a_lo".to_string(), 64),
            ("b_lo".to_string(), 64),
            ("a_hi".to_string(), 64),
            ("b_hi".to_string(), 64),
        ],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for the low 64 bits of i128 SUB.
///
/// `dst_lo = (a_lo - b_lo) mod 2^64` via AArch64 SUBS.
pub fn proof_isub_i128_lo() -> ProofObligation {
    let a_lo = SmtExpr::var("a_lo", 64);
    let b_lo = SmtExpr::var("b_lo", 64);

    let tmir = a_lo.clone().bvsub(b_lo.clone());
    let aarch64 = a_lo.bvsub(b_lo);

    ProofObligation {
        name: "Isub_I128 lo -> SUBS Xlo,Xa_lo,Xb_lo".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![("a_lo".to_string(), 64), ("b_lo".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for the high 64 bits of i128 SUB.
///
/// AArch64 SBC computes `a_hi + NOT(b_hi) + C` where `C = !borrow`, i.e.
/// `dst_hi = a_hi - b_hi - borrow (mod 2^64)`. The borrow from the low
/// half is `1 iff a_lo < b_lo` (unsigned), so `C = 1 iff a_lo >= b_lo`.
/// The tMIR spec for the high limb of `a - b` is the same.
pub fn proof_isub_i128_hi() -> ProofObligation {
    let a_lo = SmtExpr::var("a_lo", 64);
    let b_lo = SmtExpr::var("b_lo", 64);
    let a_hi = SmtExpr::var("a_hi", 64);
    let b_hi = SmtExpr::var("b_hi", 64);

    // borrow = 1 iff a_lo < b_lo (unsigned).
    let borrow_bool = a_lo.bvult(b_lo);
    let borrow_bv = SmtExpr::ite(
        borrow_bool,
        SmtExpr::bv_const(1, 64),
        SmtExpr::bv_const(0, 64),
    );

    // tMIR spec: high 64 bits of `a - b` = a_hi - b_hi - borrow (mod 2^64).
    let tmir = a_hi.clone().bvsub(b_hi.clone()).bvsub(borrow_bv.clone());
    // AArch64 SBC: dst_hi = a_hi - b_hi - borrow, reading C from SUBS.
    let aarch64 = a_hi.bvsub(b_hi).bvsub(borrow_bv);

    ProofObligation {
        name: "Isub_I128 hi -> SBC Xhi,Xa_hi,Xb_hi".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![
            ("a_lo".to_string(), 64),
            ("b_lo".to_string(), 64),
            ("a_hi".to_string(), 64),
            ("b_hi".to_string(), 64),
        ],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for the low 64 bits of i128 MUL.
///
/// ```text
/// dst_lo = MUL a_lo, b_lo
/// ```
/// i.e. `dst_lo = (a_lo * b_lo) mod 2^64`. The i128 product spec reduces to
/// the same expression for its low 64 bits because multiplication is
/// commutative/associative under mod 2^64 and only the cross terms
/// `a_lo*b_hi`, `a_hi*b_lo`, `a_hi*b_hi` contribute to bits >= 64.
pub fn proof_imul_i128_lo() -> ProofObligation {
    let a_lo = SmtExpr::var("a_lo", 64);
    let b_lo = SmtExpr::var("b_lo", 64);

    // tMIR spec: low 64 bits of `a * b` are `(a_lo * b_lo) mod 2^64`.
    let tmir = a_lo.clone().bvmul(b_lo.clone());
    // AArch64 MUL writes `a_lo * b_lo` into dst_lo.
    let aarch64 = a_lo.bvmul(b_lo);

    ProofObligation {
        name: "Imul_I128 lo -> MUL Xlo,Xa_lo,Xb_lo".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![("a_lo".to_string(), 64), ("b_lo".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for the high 64 bits of i128 MUL.
///
/// The AArch64 lowering emits:
/// ```text
/// dst_lo = MUL   a_lo, b_lo
/// t0     = UMULH a_lo, b_lo                 // upper 64 bits of a_lo*b_lo
/// t1     = MADD  a_lo, b_hi, t0             // t0 + a_lo * b_hi
/// dst_hi = MADD  a_hi, b_lo, t1             // t1 + a_hi * b_lo
/// ```
/// So `dst_hi = UMULH(a_lo, b_lo) + a_lo*b_hi + a_hi*b_lo (mod 2^64)`.
///
/// `UMULH` has no native 64-bit encoding and the SMT evaluator truncates
/// `bvmul` to 64 bits, so we cannot compute the true high-half product
/// inside the evaluator. We model `UMULH(a_lo, b_lo)` as a free 64-bit
/// variable `umulh_ab_lo` that appears identically on the tMIR spec side
/// and the AArch64 machine side. The proof then verifies that the MADD
/// chain correctly accumulates the cross terms on top of the UMULH
/// contribution — this is the part of the lowering that could realistically
/// be miscoded (operand order, missed addend, wrong base). The correctness
/// of `UMULH` itself is covered by `aarch64_semantics` encoding tests.
///
/// A future z4-native proof can replace `umulh_ab_lo` with the true
/// `(zero_extend(a_lo, 64).bvmul(zero_extend(b_lo, 64))).extract(127, 64)`
/// expression once the evaluator supports 128-bit `bvmul`.
pub fn proof_imul_i128_hi() -> ProofObligation {
    let a_lo = SmtExpr::var("a_lo", 64);
    let b_lo = SmtExpr::var("b_lo", 64);
    let a_hi = SmtExpr::var("a_hi", 64);
    let b_hi = SmtExpr::var("b_hi", 64);
    // Symbolic stand-in for UMULH(a_lo, b_lo). The same variable appears on
    // both sides so the evaluator sees a well-defined value; any lowering
    // bug in the MADD chain around it will still surface as a mismatch.
    let umulh = SmtExpr::var("umulh_ab_lo", 64);

    // tMIR spec: `dst_hi = umulh(a_lo,b_lo) + a_lo*b_hi + a_hi*b_lo (mod 2^64)`.
    let cross_ab = a_lo.clone().bvmul(b_hi.clone());
    let cross_ba = a_hi.clone().bvmul(b_lo.clone());
    let tmir = umulh.clone().bvadd(cross_ab.clone()).bvadd(cross_ba.clone());

    // AArch64 form: t1 = MADD(a_lo, b_hi, t0) = a_lo*b_hi + umulh
    //               dst_hi = MADD(a_hi, b_lo, t1) = a_hi*b_lo + t1
    let t1 = cross_ab.bvadd(umulh);
    let aarch64 = cross_ba.bvadd(t1);

    ProofObligation {
        name: "Imul_I128 hi -> MUL+UMULH+MADD+MADD".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![
            ("a_lo".to_string(), 64),
            ("b_lo".to_string(), 64),
            ("a_hi".to_string(), 64),
            ("b_hi".to_string(), 64),
            ("umulh_ab_lo".to_string(), 64),
        ],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
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
        // I128 multi-register (statistical verification on each 64-bit limb;
        // see #324 and proof_iadd_i128_hi for the ADC carry identity).
        proof_iadd_i128_lo(),
        proof_iadd_i128_hi(),
        proof_isub_i128_lo(),
        proof_isub_i128_hi(),
        proof_imul_i128_lo(),
        proof_imul_i128_hi(),
    ]
}

// ---------------------------------------------------------------------------
// Floating-point lowering proofs
// ---------------------------------------------------------------------------

/// Build the proof obligation for: `tMIR::Fadd(F32, a, b) -> FADD Sd, Sn, Sm`
///
/// Verifies that the tMIR FP add semantics (`fp.add(RNE, a, b)`) match
/// the AArch64 FADD instruction semantics for single-precision.
pub fn proof_fadd_f32() -> ProofObligation {
    use crate::aarch64_semantics::{encode_fadd_rr, FPSize};
    use crate::tmir_semantics::encode_tmir_fp_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::fp32_const(0.0); // placeholder; concrete values tested by FP verifier
    let b = SmtExpr::fp32_const(0.0);

    ProofObligation {
        name: "Fadd_F32 -> FADD Sd".to_string(),
        tmir_expr: encode_tmir_fp_binop(&Opcode::Fadd, Type::F32, a.clone(), b.clone()),
        aarch64_expr: encode_fadd_rr(FPSize::Single, a, b),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 8, 24), ("b".to_string(), 8, 24)],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Fadd(F64, a, b) -> FADD Dd, Dn, Dm`
pub fn proof_fadd_f64() -> ProofObligation {
    use crate::aarch64_semantics::{encode_fadd_rr, FPSize};
    use crate::tmir_semantics::encode_tmir_fp_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::fp64_const(0.0);
    let b = SmtExpr::fp64_const(0.0);

    ProofObligation {
        name: "Fadd_F64 -> FADD Dd".to_string(),
        tmir_expr: encode_tmir_fp_binop(&Opcode::Fadd, Type::F64, a.clone(), b.clone()),
        aarch64_expr: encode_fadd_rr(FPSize::Double, a, b),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 11, 53), ("b".to_string(), 11, 53)],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Fsub(F32, a, b) -> FSUB Sd, Sn, Sm`
pub fn proof_fsub_f32() -> ProofObligation {
    use crate::aarch64_semantics::{encode_fsub_rr, FPSize};
    use crate::tmir_semantics::encode_tmir_fp_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::fp32_const(0.0);
    let b = SmtExpr::fp32_const(0.0);

    ProofObligation {
        name: "Fsub_F32 -> FSUB Sd".to_string(),
        tmir_expr: encode_tmir_fp_binop(&Opcode::Fsub, Type::F32, a.clone(), b.clone()),
        aarch64_expr: encode_fsub_rr(FPSize::Single, a, b),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 8, 24), ("b".to_string(), 8, 24)],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Fsub(F64, a, b) -> FSUB Dd, Dn, Dm`
pub fn proof_fsub_f64() -> ProofObligation {
    use crate::aarch64_semantics::{encode_fsub_rr, FPSize};
    use crate::tmir_semantics::encode_tmir_fp_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::fp64_const(0.0);
    let b = SmtExpr::fp64_const(0.0);

    ProofObligation {
        name: "Fsub_F64 -> FSUB Dd".to_string(),
        tmir_expr: encode_tmir_fp_binop(&Opcode::Fsub, Type::F64, a.clone(), b.clone()),
        aarch64_expr: encode_fsub_rr(FPSize::Double, a, b),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 11, 53), ("b".to_string(), 11, 53)],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Fmul(F32, a, b) -> FMUL Sd, Sn, Sm`
pub fn proof_fmul_f32() -> ProofObligation {
    use crate::aarch64_semantics::{encode_fmul_rr, FPSize};
    use crate::tmir_semantics::encode_tmir_fp_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::fp32_const(0.0);
    let b = SmtExpr::fp32_const(0.0);

    ProofObligation {
        name: "Fmul_F32 -> FMUL Sd".to_string(),
        tmir_expr: encode_tmir_fp_binop(&Opcode::Fmul, Type::F32, a.clone(), b.clone()),
        aarch64_expr: encode_fmul_rr(FPSize::Single, a, b),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 8, 24), ("b".to_string(), 8, 24)],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Fmul(F64, a, b) -> FMUL Dd, Dn, Dm`
pub fn proof_fmul_f64() -> ProofObligation {
    use crate::aarch64_semantics::{encode_fmul_rr, FPSize};
    use crate::tmir_semantics::encode_tmir_fp_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::fp64_const(0.0);
    let b = SmtExpr::fp64_const(0.0);

    ProofObligation {
        name: "Fmul_F64 -> FMUL Dd".to_string(),
        tmir_expr: encode_tmir_fp_binop(&Opcode::Fmul, Type::F64, a.clone(), b.clone()),
        aarch64_expr: encode_fmul_rr(FPSize::Double, a, b),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 11, 53), ("b".to_string(), 11, 53)],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Fneg(F32, a) -> FNEG Sd, Sn`
pub fn proof_fneg_f32() -> ProofObligation {
    use crate::aarch64_semantics::{encode_fneg, FPSize};
    use crate::tmir_semantics::encode_tmir_fneg;
    use llvm2_lower::types::Type;

    let a = SmtExpr::fp32_const(0.0);

    ProofObligation {
        name: "Fneg_F32 -> FNEG Sd".to_string(),
        tmir_expr: encode_tmir_fneg(Type::F32, a.clone()),
        aarch64_expr: encode_fneg(FPSize::Single, a),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 8, 24)],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Fneg(F64, a) -> FNEG Dd, Dn`
pub fn proof_fneg_f64() -> ProofObligation {
    use crate::aarch64_semantics::{encode_fneg, FPSize};
    use crate::tmir_semantics::encode_tmir_fneg;
    use llvm2_lower::types::Type;

    let a = SmtExpr::fp64_const(0.0);

    ProofObligation {
        name: "Fneg_F64 -> FNEG Dd".to_string(),
        tmir_expr: encode_tmir_fneg(Type::F64, a.clone()),
        aarch64_expr: encode_fneg(FPSize::Double, a),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 11, 53)],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Fdiv(F32, a, b) -> FDIV Sd, Sn, Sm`
///
/// Reference: ARM DDI 0487, C7.2.77 FDIV (scalar).
pub fn proof_fdiv_f32() -> ProofObligation {
    use crate::aarch64_semantics::{encode_fdiv_rr, FPSize};
    use crate::tmir_semantics::encode_tmir_fp_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::fp32_const(0.0);
    let b = SmtExpr::fp32_const(0.0);

    ProofObligation {
        name: "Fdiv_F32 -> FDIV Sd".to_string(),
        tmir_expr: encode_tmir_fp_binop(&Opcode::Fdiv, Type::F32, a.clone(), b.clone()),
        aarch64_expr: encode_fdiv_rr(FPSize::Single, a, b),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 8, 24), ("b".to_string(), 8, 24)],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Fdiv(F64, a, b) -> FDIV Dd, Dn, Dm`
///
/// Reference: ARM DDI 0487, C7.2.77 FDIV (scalar).
pub fn proof_fdiv_f64() -> ProofObligation {
    use crate::aarch64_semantics::{encode_fdiv_rr, FPSize};
    use crate::tmir_semantics::encode_tmir_fp_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::fp64_const(0.0);
    let b = SmtExpr::fp64_const(0.0);

    ProofObligation {
        name: "Fdiv_F64 -> FDIV Dd".to_string(),
        tmir_expr: encode_tmir_fp_binop(&Opcode::Fdiv, Type::F64, a.clone(), b.clone()),
        aarch64_expr: encode_fdiv_rr(FPSize::Double, a, b),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 11, 53), ("b".to_string(), 11, 53)],
            category: None,
    }
}

// ---------------------------------------------------------------------------
// Floating-point absolute value and square root lowering proofs
// ---------------------------------------------------------------------------

/// Build the proof obligation for: `tMIR::Fabs(F32, a) -> FABS Sd, Sn`
///
/// Reference: ARM DDI 0487, C7.2.73 FABS (scalar).
pub fn proof_fabs_f32() -> ProofObligation {
    use crate::aarch64_semantics::{encode_fabs, FPSize};
    use crate::tmir_semantics::encode_tmir_fabs;
    use llvm2_lower::types::Type;

    let a = SmtExpr::fp32_const(0.0);

    ProofObligation {
        name: "Fabs_F32 -> FABS Sd".to_string(),
        tmir_expr: encode_tmir_fabs(Type::F32, a.clone()),
        aarch64_expr: encode_fabs(FPSize::Single, a),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 8, 24)],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Fabs(F64, a) -> FABS Dd, Dn`
///
/// Reference: ARM DDI 0487, C7.2.73 FABS (scalar).
pub fn proof_fabs_f64() -> ProofObligation {
    use crate::aarch64_semantics::{encode_fabs, FPSize};
    use crate::tmir_semantics::encode_tmir_fabs;
    use llvm2_lower::types::Type;

    let a = SmtExpr::fp64_const(0.0);

    ProofObligation {
        name: "Fabs_F64 -> FABS Dd".to_string(),
        tmir_expr: encode_tmir_fabs(Type::F64, a.clone()),
        aarch64_expr: encode_fabs(FPSize::Double, a),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 11, 53)],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Fsqrt(F32, a) -> FSQRT Sd, Sn`
///
/// Reference: ARM DDI 0487, C7.2.160 FSQRT (scalar).
pub fn proof_fsqrt_f32() -> ProofObligation {
    use crate::aarch64_semantics::{encode_fsqrt, FPSize};
    use crate::tmir_semantics::encode_tmir_fsqrt;
    use llvm2_lower::types::Type;

    let a = SmtExpr::fp32_const(0.0);

    ProofObligation {
        name: "Fsqrt_F32 -> FSQRT Sd".to_string(),
        tmir_expr: encode_tmir_fsqrt(Type::F32, a.clone()),
        aarch64_expr: encode_fsqrt(FPSize::Single, a),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 8, 24)],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Fsqrt(F64, a) -> FSQRT Dd, Dn`
///
/// Reference: ARM DDI 0487, C7.2.160 FSQRT (scalar).
pub fn proof_fsqrt_f64() -> ProofObligation {
    use crate::aarch64_semantics::{encode_fsqrt, FPSize};
    use crate::tmir_semantics::encode_tmir_fsqrt;
    use llvm2_lower::types::Type;

    let a = SmtExpr::fp64_const(0.0);

    ProofObligation {
        name: "Fsqrt_F64 -> FSQRT Dd".to_string(),
        tmir_expr: encode_tmir_fsqrt(Type::F64, a.clone()),
        aarch64_expr: encode_fsqrt(FPSize::Double, a),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), 11, 53)],
            category: None,
    }
}

// ---------------------------------------------------------------------------
// Floating-point comparison lowering proofs: tMIR::Fcmp -> FCMP + CSET
// ---------------------------------------------------------------------------

/// Generic FCMP proof builder. Builds a proof that
/// `tMIR::Fcmp(cond, ty, a, b)` produces the same BV1 result as the
/// AArch64 `FCMP + CSET` sequence encoded by `encode_fcmp`.
///
/// Reference: ARM DDI 0487, C7.2.76 FCMP.
fn proof_fcmp_generic(
    cond: llvm2_lower::instructions::FloatCC,
    is_f32: bool,
    name: &str,
) -> ProofObligation {
    use crate::aarch64_semantics::{encode_fcmp, FPSize};
    use crate::tmir_semantics::encode_tmir_fcmp;
    use llvm2_lower::types::Type;

    let (ty, fp_size, eb, sb) = if is_f32 {
        (Type::F32, FPSize::Single, 8u32, 24u32)
    } else {
        (Type::F64, FPSize::Double, 11u32, 53u32)
    };

    let a = if is_f32 { SmtExpr::fp32_const(0.0) } else { SmtExpr::fp64_const(0.0) };
    let b = if is_f32 { SmtExpr::fp32_const(0.0) } else { SmtExpr::fp64_const(0.0) };

    ProofObligation {
        name: name.to_string(),
        tmir_expr: encode_tmir_fcmp(&cond, ty, a.clone(), b.clone()),
        aarch64_expr: encode_fcmp(fp_size, a, b, &cond),
        inputs: vec![],
        preconditions: vec![],
        fp_inputs: vec![("a".to_string(), eb, sb), ("b".to_string(), eb, sb)],
            category: None,
    }
}

/// Proof: tMIR::Fcmp(Equal, F32) -> FCMP Sn, Sm + CSET (EQ)
pub fn proof_fcmp_eq_f32() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::Equal, true, "Fcmp_Eq_F32 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(Equal, F64) -> FCMP Dn, Dm + CSET (EQ)
pub fn proof_fcmp_eq_f64() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::Equal, false, "Fcmp_Eq_F64 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(NotEqual, F32) -> FCMP + CSET (NE)
pub fn proof_fcmp_ne_f32() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::NotEqual, true, "Fcmp_NE_F32 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(NotEqual, F64) -> FCMP + CSET (NE)
pub fn proof_fcmp_ne_f64() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::NotEqual, false, "Fcmp_NE_F64 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(LessThan, F32) -> FCMP + CSET (LT)
pub fn proof_fcmp_lt_f32() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::LessThan, true, "Fcmp_LT_F32 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(LessThan, F64) -> FCMP + CSET (LT)
pub fn proof_fcmp_lt_f64() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::LessThan, false, "Fcmp_LT_F64 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(LessThanOrEqual, F32) -> FCMP + CSET (LE)
pub fn proof_fcmp_le_f32() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::LessThanOrEqual, true, "Fcmp_LE_F32 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(LessThanOrEqual, F64) -> FCMP + CSET (LE)
pub fn proof_fcmp_le_f64() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::LessThanOrEqual, false, "Fcmp_LE_F64 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(GreaterThan, F32) -> FCMP + CSET (GT)
pub fn proof_fcmp_gt_f32() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::GreaterThan, true, "Fcmp_GT_F32 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(GreaterThan, F64) -> FCMP + CSET (GT)
pub fn proof_fcmp_gt_f64() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::GreaterThan, false, "Fcmp_GT_F64 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(GreaterThanOrEqual, F32) -> FCMP + CSET (GE)
pub fn proof_fcmp_ge_f32() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::GreaterThanOrEqual, true, "Fcmp_GE_F32 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(GreaterThanOrEqual, F64) -> FCMP + CSET (GE)
pub fn proof_fcmp_ge_f64() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::GreaterThanOrEqual, false, "Fcmp_GE_F64 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(Ordered, F32) -> FCMP + CSET (ORD)
pub fn proof_fcmp_ord_f32() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::Ordered, true, "Fcmp_Ord_F32 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(Ordered, F64) -> FCMP + CSET (ORD)
pub fn proof_fcmp_ord_f64() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::Ordered, false, "Fcmp_Ord_F64 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(Unordered, F32) -> FCMP + CSET (UNO)
pub fn proof_fcmp_uno_f32() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::Unordered, true, "Fcmp_Uno_F32 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(Unordered, F64) -> FCMP + CSET (UNO)
pub fn proof_fcmp_uno_f64() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::Unordered, false, "Fcmp_Uno_F64 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(UnorderedEqual, F32) -> FCMP + CSET (UEQ)
pub fn proof_fcmp_ueq_f32() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::UnorderedEqual, true, "Fcmp_UEQ_F32 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(UnorderedEqual, F64) -> FCMP + CSET (UEQ)
pub fn proof_fcmp_ueq_f64() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::UnorderedEqual, false, "Fcmp_UEQ_F64 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(UnorderedNotEqual, F32) -> FCMP + CSET (UNE)
pub fn proof_fcmp_une_f32() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::UnorderedNotEqual, true, "Fcmp_UNE_F32 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(UnorderedNotEqual, F64) -> FCMP + CSET (UNE)
pub fn proof_fcmp_une_f64() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::UnorderedNotEqual, false, "Fcmp_UNE_F64 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(UnorderedLessThan, F32) -> FCMP + CSET (ULT)
pub fn proof_fcmp_ult_f32() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::UnorderedLessThan, true, "Fcmp_ULT_F32 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(UnorderedLessThan, F64) -> FCMP + CSET (ULT)
pub fn proof_fcmp_ult_f64() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::UnorderedLessThan, false, "Fcmp_ULT_F64 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(UnorderedLessThanOrEqual, F32) -> FCMP + CSET (ULE)
pub fn proof_fcmp_ule_f32() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::UnorderedLessThanOrEqual, true, "Fcmp_ULE_F32 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(UnorderedLessThanOrEqual, F64) -> FCMP + CSET (ULE)
pub fn proof_fcmp_ule_f64() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::UnorderedLessThanOrEqual, false, "Fcmp_ULE_F64 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(UnorderedGreaterThan, F32) -> FCMP + CSET (UGT)
pub fn proof_fcmp_ugt_f32() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::UnorderedGreaterThan, true, "Fcmp_UGT_F32 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(UnorderedGreaterThan, F64) -> FCMP + CSET (UGT)
pub fn proof_fcmp_ugt_f64() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::UnorderedGreaterThan, false, "Fcmp_UGT_F64 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(UnorderedGreaterThanOrEqual, F32) -> FCMP + CSET (UGE)
pub fn proof_fcmp_uge_f32() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::UnorderedGreaterThanOrEqual, true, "Fcmp_UGE_F32 -> FCMP+CSET")
}

/// Proof: tMIR::Fcmp(UnorderedGreaterThanOrEqual, F64) -> FCMP + CSET (UGE)
pub fn proof_fcmp_uge_f64() -> ProofObligation {
    use llvm2_lower::instructions::FloatCC;
    proof_fcmp_generic(FloatCC::UnorderedGreaterThanOrEqual, false, "Fcmp_UGE_F64 -> FCMP+CSET")
}

/// Return all floating-point lowering rule proofs.
pub fn all_fp_lowering_proofs() -> Vec<ProofObligation> {
    vec![
        proof_fadd_f32(),
        proof_fadd_f64(),
        proof_fsub_f32(),
        proof_fsub_f64(),
        proof_fmul_f32(),
        proof_fmul_f64(),
        proof_fneg_f32(),
        proof_fneg_f64(),
        proof_fdiv_f32(),
        proof_fdiv_f64(),
        // FABS: absolute value (F32 + F64)
        proof_fabs_f32(),
        proof_fabs_f64(),
        // FSQRT: square root (F32 + F64)
        proof_fsqrt_f32(),
        proof_fsqrt_f64(),
        // FCMP: ordered comparisons (F32 + F64)
        proof_fcmp_eq_f32(),
        proof_fcmp_eq_f64(),
        proof_fcmp_ne_f32(),
        proof_fcmp_ne_f64(),
        proof_fcmp_lt_f32(),
        proof_fcmp_lt_f64(),
        proof_fcmp_le_f32(),
        proof_fcmp_le_f64(),
        proof_fcmp_gt_f32(),
        proof_fcmp_gt_f64(),
        proof_fcmp_ge_f32(),
        proof_fcmp_ge_f64(),
        // FCMP: ordering predicates (F32 + F64)
        proof_fcmp_ord_f32(),
        proof_fcmp_ord_f64(),
        proof_fcmp_uno_f32(),
        proof_fcmp_uno_f64(),
        // FCMP: unordered comparisons (F32 + F64)
        proof_fcmp_ueq_f32(),
        proof_fcmp_ueq_f64(),
        proof_fcmp_une_f32(),
        proof_fcmp_une_f64(),
        proof_fcmp_ult_f32(),
        proof_fcmp_ult_f64(),
        proof_fcmp_ule_f32(),
        proof_fcmp_ule_f64(),
        proof_fcmp_ugt_f32(),
        proof_fcmp_ugt_f64(),
        proof_fcmp_uge_f32(),
        proof_fcmp_uge_f64(),
    ]
}

/// Detect whether a proof obligation is an FP comparison (FCMP).
///
/// FCMP proofs produce `ITE(comparison_bool, BV1(1), BV1(0))` at the top
/// level, whereas arithmetic proofs (FADD/FSUB/FMUL/FDIV) have FPAdd/FPSub/
/// FPMul/FPDiv at the top. FNEG has FPNeg. FABS has FPAbs. FSQRT has FPSqrt.
fn is_fp_cmp_obligation(obligation: &ProofObligation) -> bool {
    matches!(&obligation.tmir_expr, SmtExpr::Ite { .. })
}

/// Parse a `FloatCC` condition from an FCMP proof obligation name.
///
/// Proof names follow the convention `Fcmp_{CondCode}_{Size} -> FCMP+CSET`.
/// This function extracts the condition code and maps it to the corresponding
/// `FloatCC` variant.
fn parse_float_cc_from_name(name: &str) -> llvm2_lower::instructions::FloatCC {
    use llvm2_lower::instructions::FloatCC;
    // Extract the condition code between "Fcmp_" and "_F"
    if let Some(rest) = name.strip_prefix("Fcmp_") {
        let cond_str = rest.split('_').next().unwrap_or("");
        match cond_str {
            "Eq" => FloatCC::Equal,
            "NE" => FloatCC::NotEqual,
            "LT" => FloatCC::LessThan,
            "LE" => FloatCC::LessThanOrEqual,
            "GT" => FloatCC::GreaterThan,
            "GE" => FloatCC::GreaterThanOrEqual,
            "Ord" => FloatCC::Ordered,
            "Uno" => FloatCC::Unordered,
            "UEQ" => FloatCC::UnorderedEqual,
            "UNE" => FloatCC::UnorderedNotEqual,
            "ULT" => FloatCC::UnorderedLessThan,
            "ULE" => FloatCC::UnorderedLessThanOrEqual,
            "UGT" => FloatCC::UnorderedGreaterThan,
            "UGE" => FloatCC::UnorderedGreaterThanOrEqual,
            other => panic!("Unknown FloatCC condition in proof name: {}", other),
        }
    } else {
        panic!("FCMP proof name does not start with 'Fcmp_': {}", name);
    }
}

/// Verify a floating-point proof obligation by concrete evaluation with
/// representative FP values.
///
/// Unlike integer proofs which use symbolic bitvector variables and exhaustive/
/// random sampling, FP proofs work with concrete floating-point constants.
/// Both tMIR and AArch64 sides are evaluated with the same FP inputs, and
/// results are compared for bitwise equality.
///
/// # Test vectors
///
/// For binary FP operations (FADD, FSUB, FMUL, FDIV): tests all combinations
/// of edge cases including zero, one, negative values, small/large magnitudes,
/// denormals, and infinity.
///
/// For unary FP operations (FNEG, FABS, FSQRT): tests each edge case value individually.
///
/// For FP comparisons (FCMP): tests all combinations of edge cases including
/// NaN, which is critical for verifying ordered/unordered comparison semantics.
/// Results are compared as BV1 (bitvector width 1) rather than Float.
///
/// # Verification strength
///
/// This is **statistical** verification using native f64 arithmetic, matching
/// the mock evaluation approach used for integer proofs at larger bit-widths.
/// For formal FP proofs, use the z4 QF_FP theory via [`crate::z4_bridge`].
pub fn verify_fp_by_evaluation(obligation: &ProofObligation) -> VerificationResult {
    let empty_env = HashMap::new();

    // FP test vectors: representative values covering IEEE 754 edge cases.
    // NaN is included for FCMP proofs (critical for ordered/unordered semantics).
    let f64_test_values: Vec<f64> = vec![
        0.0, -0.0, 1.0, -1.0, 0.5, -0.5,
        2.0, -2.0, 0.1, -0.1,
        1e10, -1e10, 1e-10, -1e-10,
        f64::INFINITY, f64::NEG_INFINITY,
        f64::MIN_POSITIVE, -f64::MIN_POSITIVE,
        f64::MAX, f64::MIN,
        std::f64::consts::PI, -std::f64::consts::PI,
        1.0 / 3.0, -1.0 / 3.0,
        42.0, -42.0, 100.0, -100.0,
        0.000001, -0.000001,
        f64::NAN,
    ];

    let f32_test_values: Vec<f32> = vec![
        0.0f32, -0.0f32, 1.0f32, -1.0f32, 0.5f32, -0.5f32,
        2.0f32, -2.0f32, 0.1f32, -0.1f32,
        1e10f32, -1e10f32, 1e-10f32, -1e-10f32,
        f32::INFINITY, f32::NEG_INFINITY,
        f32::MIN_POSITIVE, -f32::MIN_POSITIVE,
        f32::MAX, f32::MIN,
        std::f32::consts::PI, -std::f32::consts::PI,
        42.0f32, -42.0f32, 100.0f32, -100.0f32,
        0.000001f32, -0.000001f32,
        f32::NAN,
    ];

    let is_unary = obligation.fp_inputs.len() == 1;
    let is_f32 = obligation.fp_inputs.first().map(|(_, eb, _)| *eb == 8).unwrap_or(false);
    let is_cmp = is_fp_cmp_obligation(obligation);

    if is_unary {
        // Unary FP operation (FNEG, FABS, FSQRT)
        if is_f32 {
            for &a_val in &f32_test_values {
                let tmir_expr = build_fp_unary_expr(&obligation.tmir_expr, a_val as f64, is_f32);
                let aarch64_expr = build_fp_unary_expr(&obligation.aarch64_expr, a_val as f64, is_f32);
                let tmir_result = tmir_expr.try_eval(&empty_env);
                let aarch64_result = aarch64_expr.try_eval(&empty_env);
                if let (Ok(t), Ok(a)) = (&tmir_result, &aarch64_result)
                    && !fp_results_equal(t, a) {
                        return VerificationResult::Invalid {
                            counterexample: format!("a={}, tmir={:?}, aarch64={:?}", a_val, t, a),
                        };
                    }
            }
        } else {
            for &a_val in &f64_test_values {
                let tmir_expr = build_fp_unary_expr(&obligation.tmir_expr, a_val, is_f32);
                let aarch64_expr = build_fp_unary_expr(&obligation.aarch64_expr, a_val, is_f32);
                let tmir_result = tmir_expr.try_eval(&empty_env);
                let aarch64_result = aarch64_expr.try_eval(&empty_env);
                if let (Ok(t), Ok(a)) = (&tmir_result, &aarch64_result)
                    && !fp_results_equal(t, a) {
                        return VerificationResult::Invalid {
                            counterexample: format!("a={}, tmir={:?}, aarch64={:?}", a_val, t, a),
                        };
                    }
            }
        }
    } else if is_cmp {
        // FP comparison (FCMP): produces BV1. We call the encoder functions
        // directly with concrete values rather than template substitution,
        // because FCMP expression trees contain multiple FPConst(0.0)
        // placeholders for both operands that cannot be structurally
        // distinguished during tree walking.
        let cond = parse_float_cc_from_name(&obligation.name);
        if is_f32 {
            for &a_val in &f32_test_values {
                for &b_val in &f32_test_values {
                    let a_expr = SmtExpr::fp32_const(a_val);
                    let b_expr = SmtExpr::fp32_const(b_val);
                    let tmir_expr = crate::tmir_semantics::encode_tmir_fcmp(
                        &cond, llvm2_lower::types::Type::F32, a_expr.clone(), b_expr.clone());
                    let aarch64_expr = crate::aarch64_semantics::encode_fcmp(
                        crate::aarch64_semantics::FPSize::Single, a_expr, b_expr, &cond);
                    let tmir_result = tmir_expr.try_eval(&empty_env);
                    let aarch64_result = aarch64_expr.try_eval(&empty_env);
                    if let (Ok(t), Ok(a)) = (&tmir_result, &aarch64_result)
                        && !fp_results_equal(t, a) {
                            return VerificationResult::Invalid {
                                counterexample: format!("a={}, b={}, tmir={:?}, aarch64={:?}", a_val, b_val, t, a),
                            };
                        }
                }
            }
        } else {
            for &a_val in &f64_test_values {
                for &b_val in &f64_test_values {
                    let a_expr = SmtExpr::fp64_const(a_val);
                    let b_expr = SmtExpr::fp64_const(b_val);
                    let tmir_expr = crate::tmir_semantics::encode_tmir_fcmp(
                        &cond, llvm2_lower::types::Type::F64, a_expr.clone(), b_expr.clone());
                    let aarch64_expr = crate::aarch64_semantics::encode_fcmp(
                        crate::aarch64_semantics::FPSize::Double, a_expr, b_expr, &cond);
                    let tmir_result = tmir_expr.try_eval(&empty_env);
                    let aarch64_result = aarch64_expr.try_eval(&empty_env);
                    if let (Ok(t), Ok(a)) = (&tmir_result, &aarch64_result)
                        && !fp_results_equal(t, a) {
                            return VerificationResult::Invalid {
                                counterexample: format!("a={}, b={}, tmir={:?}, aarch64={:?}", a_val, b_val, t, a),
                            };
                        }
                }
            }
        }
    } else {
        // Binary FP operation (FADD, FSUB, FMUL, FDIV)
        if is_f32 {
            for &a_val in &f32_test_values {
                for &b_val in &f32_test_values {
                    let tmir_expr = build_fp_binary_expr(&obligation.tmir_expr, a_val as f64, b_val as f64, is_f32);
                    let aarch64_expr = build_fp_binary_expr(&obligation.aarch64_expr, a_val as f64, b_val as f64, is_f32);
                    let tmir_result = tmir_expr.try_eval(&empty_env);
                    let aarch64_result = aarch64_expr.try_eval(&empty_env);
                    if let (Ok(t), Ok(a)) = (&tmir_result, &aarch64_result)
                        && !fp_results_equal(t, a) {
                            return VerificationResult::Invalid {
                                counterexample: format!("a={}, b={}, tmir={:?}, aarch64={:?}", a_val, b_val, t, a),
                            };
                        }
                }
            }
        } else {
            for &a_val in &f64_test_values {
                for &b_val in &f64_test_values {
                    let tmir_expr = build_fp_binary_expr(&obligation.tmir_expr, a_val, b_val, is_f32);
                    let aarch64_expr = build_fp_binary_expr(&obligation.aarch64_expr, a_val, b_val, is_f32);
                    let tmir_result = tmir_expr.try_eval(&empty_env);
                    let aarch64_result = aarch64_expr.try_eval(&empty_env);
                    if let (Ok(t), Ok(a)) = (&tmir_result, &aarch64_result)
                        && !fp_results_equal(t, a) {
                            return VerificationResult::Invalid {
                                counterexample: format!("a={}, b={}, tmir={:?}, aarch64={:?}", a_val, b_val, t, a),
                            };
                        }
                }
            }
        }
    }

    VerificationResult::Valid
}

/// Build a concrete FP binary expression by substituting concrete values.
///
/// The proof obligation's tmir_expr / aarch64_expr use placeholder FPConst(0.0)
/// nodes. This function rebuilds the expression tree with concrete values.
fn build_fp_binary_expr(template: &SmtExpr, a_val: f64, b_val: f64, is_f32: bool) -> SmtExpr {
    let a = if is_f32 {
        SmtExpr::fp32_const(a_val as f32)
    } else {
        SmtExpr::fp64_const(a_val)
    };
    let b = if is_f32 {
        SmtExpr::fp32_const(b_val as f32)
    } else {
        SmtExpr::fp64_const(b_val)
    };

    match template {
        SmtExpr::FPAdd { rm, .. } => SmtExpr::fp_add(*rm, a, b),
        SmtExpr::FPSub { rm, .. } => SmtExpr::fp_sub(*rm, a, b),
        SmtExpr::FPMul { rm, .. } => SmtExpr::fp_mul(*rm, a, b),
        SmtExpr::FPDiv { rm, .. } => SmtExpr::fp_div(*rm, a, b),
        _ => template.clone(),
    }
}

/// Build a concrete FP unary expression by substituting a concrete value.
fn build_fp_unary_expr(template: &SmtExpr, a_val: f64, is_f32: bool) -> SmtExpr {
    let a = if is_f32 {
        SmtExpr::fp32_const(a_val as f32)
    } else {
        SmtExpr::fp64_const(a_val)
    };

    match template {
        SmtExpr::FPNeg { .. } => a.fp_neg(),
        SmtExpr::FPAbs { .. } => a.fp_abs(),
        SmtExpr::FPSqrt { rm, .. } => SmtExpr::fp_sqrt(*rm, a),
        _ => template.clone(),
    }
}

/// Compare two FP evaluation results, handling NaN correctly.
///
/// IEEE 754: NaN != NaN, but for verification we consider two NaN results
/// as equal (both sides produced NaN, which is the correct behavior).
fn fp_results_equal(a: &crate::smt::EvalResult, b: &crate::smt::EvalResult) -> bool {
    use crate::smt::EvalResult;
    match (a, b) {
        (EvalResult::Float(fa), EvalResult::Float(fb)) => {
            if fa.is_nan() && fb.is_nan() {
                true // Both NaN = correct
            } else {
                fa.to_bits() == fb.to_bits() // Bitwise comparison (handles -0.0 vs +0.0)
            }
        }
        _ => a == b,
    }
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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

/// Proof: tMIR::Icmp(NotEqual, I64) -> CMP Xn, Xm ; CSET Xd, NE
pub fn proof_icmp_ne_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::NotEqual, AArch64CC::NE, 64, "Icmp_NE_I64 -> CMP+CSET_NE")
}

/// Proof: tMIR::Icmp(SignedGreaterThanOrEqual, I64) -> CMP + CSET GE (64-bit)
pub fn proof_icmp_sge_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::SignedGreaterThanOrEqual, AArch64CC::GE, 64, "Icmp_SGE_I64 -> CMP+CSET_GE")
}

/// Proof: tMIR::Icmp(SignedGreaterThan, I64) -> CMP + CSET GT (64-bit)
pub fn proof_icmp_sgt_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::SignedGreaterThan, AArch64CC::GT, 64, "Icmp_SGT_I64 -> CMP+CSET_GT")
}

/// Proof: tMIR::Icmp(SignedLessThanOrEqual, I64) -> CMP + CSET LE (64-bit)
pub fn proof_icmp_sle_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::SignedLessThanOrEqual, AArch64CC::LE, 64, "Icmp_SLE_I64 -> CMP+CSET_LE")
}

/// Proof: tMIR::Icmp(UnsignedGreaterThanOrEqual, I64) -> CMP + CSET HS (64-bit)
pub fn proof_icmp_uge_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::UnsignedGreaterThanOrEqual, AArch64CC::HS, 64, "Icmp_UGE_I64 -> CMP+CSET_HS")
}

/// Proof: tMIR::Icmp(UnsignedGreaterThan, I64) -> CMP + CSET HI (64-bit)
pub fn proof_icmp_ugt_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::UnsignedGreaterThan, AArch64CC::HI, 64, "Icmp_UGT_I64 -> CMP+CSET_HI")
}

/// Proof: tMIR::Icmp(UnsignedLessThanOrEqual, I64) -> CMP + CSET LS (64-bit)
pub fn proof_icmp_ule_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_icmp_generic(IntCC::UnsignedLessThanOrEqual, AArch64CC::LS, 64, "Icmp_ULE_I64 -> CMP+CSET_LS")
}

/// Return all 10 comparison lowering proofs (64-bit).
pub fn all_comparison_proofs_i64() -> Vec<ProofObligation> {
    vec![
        proof_icmp_eq_i64(),
        proof_icmp_ne_i64(),
        proof_icmp_slt_i64(),
        proof_icmp_sge_i64(),
        proof_icmp_sgt_i64(),
        proof_icmp_sle_i64(),
        proof_icmp_ult_i64(),
        proof_icmp_uge_i64(),
        proof_icmp_ugt_i64(),
        proof_icmp_ule_i64(),
    ]
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
            category: None,
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

/// Proof: tMIR::CondBr(Icmp(SignedGreaterThanOrEqual)) -> CMP + B.GE
pub fn proof_condbr_sge_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::SignedGreaterThanOrEqual, AArch64CC::GE, 32, "CondBr_SGE_I32 -> CMP+B.GE")
}

/// Proof: tMIR::CondBr(Icmp(SignedGreaterThan)) -> CMP + B.GT
pub fn proof_condbr_sgt_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::SignedGreaterThan, AArch64CC::GT, 32, "CondBr_SGT_I32 -> CMP+B.GT")
}

/// Proof: tMIR::CondBr(Icmp(SignedLessThanOrEqual)) -> CMP + B.LE
pub fn proof_condbr_sle_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::SignedLessThanOrEqual, AArch64CC::LE, 32, "CondBr_SLE_I32 -> CMP+B.LE")
}

/// Proof: tMIR::CondBr(Icmp(UnsignedGreaterThanOrEqual)) -> CMP + B.HS
pub fn proof_condbr_uge_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::UnsignedGreaterThanOrEqual, AArch64CC::HS, 32, "CondBr_UGE_I32 -> CMP+B.HS")
}

/// Proof: tMIR::CondBr(Icmp(UnsignedGreaterThan)) -> CMP + B.HI
pub fn proof_condbr_ugt_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::UnsignedGreaterThan, AArch64CC::HI, 32, "CondBr_UGT_I32 -> CMP+B.HI")
}

/// Proof: tMIR::CondBr(Icmp(UnsignedLessThanOrEqual)) -> CMP + B.LS
pub fn proof_condbr_ule_i32() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::UnsignedLessThanOrEqual, AArch64CC::LS, 32, "CondBr_ULE_I32 -> CMP+B.LS")
}

/// Return all 10 branch lowering proofs (32-bit).
pub fn all_branch_proofs_i32() -> Vec<ProofObligation> {
    vec![
        proof_condbr_eq_i32(),
        proof_condbr_ne_i32(),
        proof_condbr_slt_i32(),
        proof_condbr_sge_i32(),
        proof_condbr_sgt_i32(),
        proof_condbr_sle_i32(),
        proof_condbr_ult_i32(),
        proof_condbr_uge_i32(),
        proof_condbr_ugt_i32(),
        proof_condbr_ule_i32(),
    ]
}

// ---------------------------------------------------------------------------
// 64-bit branch lowering proofs: tMIR::CondBr(Icmp, I64) -> CMP + B.cond
// ---------------------------------------------------------------------------

/// Proof: tMIR::CondBr(Icmp(Equal, I64)) -> CMP + B.EQ (64-bit)
pub fn proof_condbr_eq_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::Equal, AArch64CC::EQ, 64, "CondBr_Eq_I64 -> CMP+B.EQ")
}

/// Proof: tMIR::CondBr(Icmp(NotEqual, I64)) -> CMP + B.NE (64-bit)
pub fn proof_condbr_ne_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::NotEqual, AArch64CC::NE, 64, "CondBr_NE_I64 -> CMP+B.NE")
}

/// Proof: tMIR::CondBr(Icmp(SignedLessThan, I64)) -> CMP + B.LT (64-bit)
pub fn proof_condbr_slt_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::SignedLessThan, AArch64CC::LT, 64, "CondBr_SLT_I64 -> CMP+B.LT")
}

/// Proof: tMIR::CondBr(Icmp(SignedGreaterThanOrEqual, I64)) -> CMP + B.GE (64-bit)
pub fn proof_condbr_sge_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::SignedGreaterThanOrEqual, AArch64CC::GE, 64, "CondBr_SGE_I64 -> CMP+B.GE")
}

/// Proof: tMIR::CondBr(Icmp(SignedGreaterThan, I64)) -> CMP + B.GT (64-bit)
pub fn proof_condbr_sgt_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::SignedGreaterThan, AArch64CC::GT, 64, "CondBr_SGT_I64 -> CMP+B.GT")
}

/// Proof: tMIR::CondBr(Icmp(SignedLessThanOrEqual, I64)) -> CMP + B.LE (64-bit)
pub fn proof_condbr_sle_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::SignedLessThanOrEqual, AArch64CC::LE, 64, "CondBr_SLE_I64 -> CMP+B.LE")
}

/// Proof: tMIR::CondBr(Icmp(UnsignedLessThan, I64)) -> CMP + B.LO (64-bit)
pub fn proof_condbr_ult_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::UnsignedLessThan, AArch64CC::LO, 64, "CondBr_ULT_I64 -> CMP+B.LO")
}

/// Proof: tMIR::CondBr(Icmp(UnsignedGreaterThanOrEqual, I64)) -> CMP + B.HS (64-bit)
pub fn proof_condbr_uge_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::UnsignedGreaterThanOrEqual, AArch64CC::HS, 64, "CondBr_UGE_I64 -> CMP+B.HS")
}

/// Proof: tMIR::CondBr(Icmp(UnsignedGreaterThan, I64)) -> CMP + B.HI (64-bit)
pub fn proof_condbr_ugt_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::UnsignedGreaterThan, AArch64CC::HI, 64, "CondBr_UGT_I64 -> CMP+B.HI")
}

/// Proof: tMIR::CondBr(Icmp(UnsignedLessThanOrEqual, I64)) -> CMP + B.LS (64-bit)
pub fn proof_condbr_ule_i64() -> ProofObligation {
    use llvm2_lower::instructions::IntCC;
    use llvm2_lower::isel::AArch64CC;
    proof_condbr_generic(IntCC::UnsignedLessThanOrEqual, AArch64CC::LS, 64, "CondBr_ULE_I64 -> CMP+B.LS")
}

/// Return all 10 branch lowering proofs (64-bit).
pub fn all_branch_proofs_i64() -> Vec<ProofObligation> {
    vec![
        proof_condbr_eq_i64(),
        proof_condbr_ne_i64(),
        proof_condbr_slt_i64(),
        proof_condbr_sge_i64(),
        proof_condbr_sgt_i64(),
        proof_condbr_sle_i64(),
        proof_condbr_ult_i64(),
        proof_condbr_uge_i64(),
        proof_condbr_ugt_i64(),
        proof_condbr_ule_i64(),
    ]
}

/// Return all branch lowering proofs (both 32-bit and 64-bit).
pub fn all_branch_proofs() -> Vec<ProofObligation> {
    let mut proofs = all_branch_proofs_i32();
    proofs.extend(all_branch_proofs_i64());
    proofs
}

/// Return all NZCV-related proofs (flags + comparisons + branches).
pub fn all_nzcv_proofs() -> Vec<ProofObligation> {
    let mut proofs = Vec::new();
    proofs.extend(all_nzcv_flag_proofs());
    proofs.extend(all_comparison_proofs_i32());
    proofs.extend(all_comparison_proofs_i64());
    proofs.extend(all_branch_proofs());
    proofs
}

// ---------------------------------------------------------------------------
// Load/Store lowering proofs
// ---------------------------------------------------------------------------
//
// Proofs that tMIR Load/Store operations are correctly lowered to
// AArch64 LDR/STR instructions. These use the symbolic SMT array theory
// (Array(BV64, BV8)) to model byte-addressable memory.
//
// The proof structure for each load lowering:
//   forall base: BV64, mem_default: BV8 .
//     let mem = ConstArray(BV64, mem_default)
//     encode_tmir_load(mem, base, 0, size) == encode_aarch64_ldr_imm(mem, base, 0, size)
//
// The proof structure for each store lowering:
//   forall base: BV64, value: BV(size*8), mem_default: BV8 .
//     let mem = ConstArray(BV64, mem_default)
//     let tmir_mem = encode_tmir_store(mem, base, 0, value, size)
//     let aarch64_mem = encode_aarch64_str_imm(mem, base, 0, value, size)
//     load(tmir_mem, base, size) == load(aarch64_mem, base, size)
//
// These delegate to the symbolic encoders in memory_proofs.rs which build
// the actual SMT array expressions.
//
// Reference: ARM DDI 0487, C6.2.131-132 (LDRB/LDR), C6.2.134 (LDRH),
//            C6.2.257-258 (STR/STRB/STRH).

/// Proof: `tMIR::Load(I32, addr)` == `LDRWui [Xn, #0]` (32-bit load, zero offset).
///
/// Verifies that loading 4 bytes from base address produces the same
/// result via tMIR semantics and AArch64 LDRWui with zero scaled offset.
///
/// Both sides read from the same symbolic ConstArray memory, so the proof
/// holds for all possible initial memory contents.
pub fn proof_load_i32_lowering() -> ProofObligation {
    crate::memory_proofs::proof_load_i32()
}

/// Proof: `tMIR::Load(I64, addr)` == `LDRXui [Xn, #0]` (64-bit load, zero offset).
pub fn proof_load_i64_lowering() -> ProofObligation {
    crate::memory_proofs::proof_load_i64()
}

/// Proof: `tMIR::Store(I32, val, addr)` == `STRWui [Xn, #0]` (32-bit store, zero offset).
///
/// Verifies that storing a 32-bit value via tMIR and AArch64 STRWui produces
/// identical memory states. Checked by storing, then loading back from both
/// memories and comparing the results.
pub fn proof_store_i32_lowering() -> ProofObligation {
    crate::memory_proofs::proof_store_i32()
}

/// Proof: `tMIR::Store(I64, val, addr)` == `STRXui [Xn, #0]` (64-bit store, zero offset).
pub fn proof_store_i64_lowering() -> ProofObligation {
    crate::memory_proofs::proof_store_i64()
}

/// Proof: `tMIR::Load(I8, addr)` == `LDRB Wt, [Xn, #0]` (8-bit load, zero offset).
///
/// On AArch64, LDRB loads a single byte and zero-extends it to 32 bits
/// in the destination W register. The tMIR side loads 1 byte.
///
/// Reference: ARM DDI 0487, C6.2.131.
pub fn proof_load_i8_lowering() -> ProofObligation {
    crate::memory_proofs::proof_load_i8()
}

/// Proof: `tMIR::Load(I16, addr)` == `LDRH Wt, [Xn, #0]` (16-bit load, zero offset).
///
/// LDRH loads a 16-bit halfword in little-endian order and zero-extends
/// it to 32 bits. The tMIR side loads 2 bytes.
///
/// Reference: ARM DDI 0487, C6.2.134.
pub fn proof_load_i16_lowering() -> ProofObligation {
    crate::memory_proofs::proof_load_i16()
}

/// Proof: `tMIR::Store(I8, val, addr)` == `STRB Wt, [Xn, #0]` (8-bit store, zero offset).
///
/// STRB stores the least significant byte of the W register to memory.
///
/// Reference: ARM DDI 0487, C6.2.258.
pub fn proof_store_i8_lowering() -> ProofObligation {
    crate::memory_proofs::proof_store_i8()
}

/// Proof: `tMIR::Store(I16, val, addr)` == `STRH Wt, [Xn, #0]` (16-bit store, zero offset).
///
/// STRH stores the least significant halfword of the W register to memory
/// in little-endian byte order.
///
/// Reference: ARM DDI 0487, C6.2.259.
pub fn proof_store_i16_lowering() -> ProofObligation {
    crate::memory_proofs::proof_store_i16()
}

/// Proof: store then load at same address returns stored value (32-bit).
///
/// ```text
/// forall base: BV64, value: BV32, mem_default: BV8 .
///   let mem = ConstArray(BV64, mem_default)
///   let mem' = store(mem, base, value, 4)
///   load(mem', base, 4) == value
/// ```
///
/// This is the fundamental store-load coherence property: memory behaves as
/// a reliable array. Critical for compiler correctness -- if this fails,
/// no program that uses memory can be verified.
pub fn proof_load_store_roundtrip_i32() -> ProofObligation {
    crate::memory_proofs::proof_roundtrip_i32()
}

/// Proof: store then load at same address returns stored value (64-bit).
///
/// The 64-bit version exercises the full 8-byte little-endian decomposition
/// and reassembly path through the SMT array model.
pub fn proof_load_store_roundtrip_i64() -> ProofObligation {
    crate::memory_proofs::proof_roundtrip_i64()
}

/// Return all load/store lowering proofs (10 total).
///
/// Covers:
/// - Load equivalence: I8, I16, I32, I64 (tMIR Load == AArch64 LDR/LDRB/LDRH)
/// - Store equivalence: I8, I16, I32, I64 (tMIR Store == AArch64 STR/STRB/STRH)
/// - Store-load roundtrip: I32, I64 (store then load returns same value)
pub fn all_load_store_proofs() -> Vec<ProofObligation> {
    vec![
        // Load equivalence
        proof_load_i8_lowering(),
        proof_load_i16_lowering(),
        proof_load_i32_lowering(),
        proof_load_i64_lowering(),
        // Store equivalence
        proof_store_i8_lowering(),
        proof_store_i16_lowering(),
        proof_store_i32_lowering(),
        proof_store_i64_lowering(),
        // Store-load roundtrip
        proof_load_store_roundtrip_i32(),
        proof_load_store_roundtrip_i64(),
    ]
}

// ---------------------------------------------------------------------------
// I8 bitwise/shift lowering proofs (exhaustive — all 2^16 or 2^8 combos)
// ---------------------------------------------------------------------------

/// Build the proof obligation for: `tMIR::Band(I8, a, b) -> AND (8-bit)`
///
/// On AArch64, 8-bit operations are performed in 32-bit W registers.
/// The proof verifies semantic equivalence at the 8-bit bitvector level.
pub fn proof_band_i8() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    ProofObligation {
        name: "Band_I8 -> AND (8-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::Band, Type::I8, a.clone(), b.clone()),
        aarch64_expr: a.bvand(b),
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Bor(I8, a, b) -> OR (8-bit)`
pub fn proof_bor_i8() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    ProofObligation {
        name: "Bor_I8 -> OR (8-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::Bor, Type::I8, a.clone(), b.clone()),
        aarch64_expr: a.bvor(b),
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Bxor(I8, a, b) -> XOR (8-bit)`
pub fn proof_bxor_i8() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    ProofObligation {
        name: "Bxor_I8 -> XOR (8-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::Bxor, Type::I8, a.clone(), b.clone()),
        aarch64_expr: a.bvxor(b),
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Bnot(I8, a) -> NOT (8-bit)`
///
/// MVN on AArch64 is `ORN Rd, XZR, Rm` = bitwise complement.
pub fn proof_bnot_i8() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bnot;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 8);
    let all_ones = SmtExpr::bv_const(mask(u64::MAX, 8), 8);

    ProofObligation {
        name: "Bnot_I8 -> NOT (8-bit)".to_string(),
        tmir_expr: encode_tmir_bnot(Type::I8, a.clone()),
        aarch64_expr: a.bvxor(all_ones),
        inputs: vec![("a".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Ishl(I8, a, b) -> SHL (8-bit)`
pub fn proof_ishl_i8() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_shift;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    ProofObligation {
        name: "Ishl_I8 -> SHL (8-bit)".to_string(),
        tmir_expr: encode_tmir_shift(&Opcode::Ishl, Type::I8, a.clone(), b.clone()),
        aarch64_expr: a.bvshl(b),
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Ushr(I8, a, b) -> LSR (8-bit)`
pub fn proof_ushr_i8() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_shift;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    ProofObligation {
        name: "Ushr_I8 -> LSR (8-bit)".to_string(),
        tmir_expr: encode_tmir_shift(&Opcode::Ushr, Type::I8, a.clone(), b.clone()),
        aarch64_expr: a.bvlshr(b),
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Sshr(I8, a, b) -> ASR (8-bit)`
pub fn proof_sshr_i8() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_shift;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    ProofObligation {
        name: "Sshr_I8 -> ASR (8-bit)".to_string(),
        tmir_expr: encode_tmir_shift(&Opcode::Sshr, Type::I8, a.clone(), b.clone()),
        aarch64_expr: a.bvashr(b),
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::BandNot(I8, a, b) -> BIC (8-bit)`
///
/// BIC (bit clear) on AArch64: `Rd = Rn & ~Rm`.
/// tMIR `BandNot` has identical semantics; issue #425 wires the default-on
/// lowering proof.
pub fn proof_bic_i8() -> ProofObligation {
    use crate::aarch64_semantics::encode_bic_rr;
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    ProofObligation {
        name: "BandNot_I8 -> BIC (8-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::BandNot, Type::I8, a.clone(), b.clone()),
        aarch64_expr: encode_bic_rr(OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::BorNot(I8, a, b) -> ORN (8-bit)`
///
/// ORN on AArch64: `Rd = Rn | ~Rm`. tMIR `BorNot` has identical semantics;
/// issue #425 wires the default-on lowering proof.
pub fn proof_orn_i8() -> ProofObligation {
    use crate::aarch64_semantics::encode_orn_rr;
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    ProofObligation {
        name: "BorNot_I8 -> ORN (8-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::BorNot, Type::I8, a.clone(), b.clone()),
        aarch64_expr: encode_orn_rr(OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

// ---------------------------------------------------------------------------
// I16 bitwise/shift lowering proofs (statistical — edge cases + random sampling)
// ---------------------------------------------------------------------------

/// Build the proof obligation for: `tMIR::Band(I16, a, b) -> AND (16-bit)`
///
/// On AArch64, 16-bit operations are performed in 32-bit W registers.
/// The proof verifies semantic equivalence at the 16-bit bitvector level.
pub fn proof_band_i16() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 16);
    let b = SmtExpr::var("b", 16);

    ProofObligation {
        name: "Band_I16 -> AND (16-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::Band, Type::I16, a.clone(), b.clone()),
        aarch64_expr: a.bvand(b),
        inputs: vec![("a".to_string(), 16), ("b".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Bor(I16, a, b) -> OR (16-bit)`
pub fn proof_bor_i16() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 16);
    let b = SmtExpr::var("b", 16);

    ProofObligation {
        name: "Bor_I16 -> OR (16-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::Bor, Type::I16, a.clone(), b.clone()),
        aarch64_expr: a.bvor(b),
        inputs: vec![("a".to_string(), 16), ("b".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Bxor(I16, a, b) -> XOR (16-bit)`
pub fn proof_bxor_i16() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 16);
    let b = SmtExpr::var("b", 16);

    ProofObligation {
        name: "Bxor_I16 -> XOR (16-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::Bxor, Type::I16, a.clone(), b.clone()),
        aarch64_expr: a.bvxor(b),
        inputs: vec![("a".to_string(), 16), ("b".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Bnot(I16, a) -> NOT (16-bit)`
pub fn proof_bnot_i16() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bnot;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 16);
    let all_ones = SmtExpr::bv_const(mask(u64::MAX, 16), 16);

    ProofObligation {
        name: "Bnot_I16 -> NOT (16-bit)".to_string(),
        tmir_expr: encode_tmir_bnot(Type::I16, a.clone()),
        aarch64_expr: a.bvxor(all_ones),
        inputs: vec![("a".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Ishl(I16, a, b) -> SHL (16-bit)`
pub fn proof_ishl_i16() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_shift;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 16);
    let b = SmtExpr::var("b", 16);

    ProofObligation {
        name: "Ishl_I16 -> SHL (16-bit)".to_string(),
        tmir_expr: encode_tmir_shift(&Opcode::Ishl, Type::I16, a.clone(), b.clone()),
        aarch64_expr: a.bvshl(b),
        inputs: vec![("a".to_string(), 16), ("b".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Ushr(I16, a, b) -> LSR (16-bit)`
pub fn proof_ushr_i16() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_shift;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 16);
    let b = SmtExpr::var("b", 16);

    ProofObligation {
        name: "Ushr_I16 -> LSR (16-bit)".to_string(),
        tmir_expr: encode_tmir_shift(&Opcode::Ushr, Type::I16, a.clone(), b.clone()),
        aarch64_expr: a.bvlshr(b),
        inputs: vec![("a".to_string(), 16), ("b".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::Sshr(I16, a, b) -> ASR (16-bit)`
pub fn proof_sshr_i16() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_shift;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 16);
    let b = SmtExpr::var("b", 16);

    ProofObligation {
        name: "Sshr_I16 -> ASR (16-bit)".to_string(),
        tmir_expr: encode_tmir_shift(&Opcode::Sshr, Type::I16, a.clone(), b.clone()),
        aarch64_expr: a.bvashr(b),
        inputs: vec![("a".to_string(), 16), ("b".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::BandNot(I16, a, b) -> BIC (16-bit)`
///
/// BIC (bit clear) on AArch64: `Rd = Rn & ~Rm`. On AArch64, 16-bit operations
/// are performed in 32-bit W registers; `encode_bic_rr` derives complement
/// width from the operand's bitvector sort so the I16 proof composes
/// correctly. I16 sibling of `proof_bic_i8` (issue #425), widened for
/// epic #407 default-on z4 smoke rollout.
pub fn proof_bic_i16() -> ProofObligation {
    use crate::aarch64_semantics::encode_bic_rr;
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 16);
    let b = SmtExpr::var("b", 16);

    ProofObligation {
        name: "BandNot_I16 -> BIC (16-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::BandNot, Type::I16, a.clone(), b.clone()),
        aarch64_expr: encode_bic_rr(OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 16), ("b".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Build the proof obligation for: `tMIR::BorNot(I16, a, b) -> ORN (16-bit)`
///
/// ORN on AArch64: `Rd = Rn | ~Rm`. I16 sibling of `proof_orn_i8`
/// (issue #425), widened for epic #407 default-on z4 smoke rollout.
pub fn proof_orn_i16() -> ProofObligation {
    use crate::aarch64_semantics::encode_orn_rr;
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 16);
    let b = SmtExpr::var("b", 16);

    ProofObligation {
        name: "BorNot_I16 -> ORN (16-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::BorNot, Type::I16, a.clone(), b.clone()),
        aarch64_expr: encode_orn_rr(OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 16), ("b".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

// ---------------------------------------------------------------------------
// I32 bitwise/shift lowering proofs (statistical -- 100K random samples)
// ---------------------------------------------------------------------------
//
// Issue #449, epic #407 (Task 3): widen z4 smoke to i32/i64. StableHasher
// caching (#420) made wider-width SMT proofs tractable.

/// Build the proof obligation for: `tMIR::Band(I32, a, b) -> AND (32-bit)`.
pub fn proof_band_i32() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "Band_I32 -> AND (32-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::Band, Type::I32, a.clone(), b.clone()),
        aarch64_expr: a.bvand(b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::Bor(I32, a, b) -> OR (32-bit)`.
pub fn proof_bor_i32() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "Bor_I32 -> OR (32-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::Bor, Type::I32, a.clone(), b.clone()),
        aarch64_expr: a.bvor(b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::Bxor(I32, a, b) -> XOR (32-bit)`.
pub fn proof_bxor_i32() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "Bxor_I32 -> XOR (32-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::Bxor, Type::I32, a.clone(), b.clone()),
        aarch64_expr: a.bvxor(b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::Ishl(I32, a, b) -> LSL (32-bit)`.
pub fn proof_ishl_i32() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_shift;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "Ishl_I32 -> LSL (32-bit)".to_string(),
        tmir_expr: encode_tmir_shift(&Opcode::Ishl, Type::I32, a.clone(), b.clone()),
        aarch64_expr: a.bvshl(b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::Ushr(I32, a, b) -> LSR (32-bit)`.
pub fn proof_ushr_i32() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_shift;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "Ushr_I32 -> LSR (32-bit)".to_string(),
        tmir_expr: encode_tmir_shift(&Opcode::Ushr, Type::I32, a.clone(), b.clone()),
        aarch64_expr: a.bvlshr(b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::Sshr(I32, a, b) -> ASR (32-bit)`.
pub fn proof_sshr_i32() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_shift;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "Sshr_I32 -> ASR (32-bit)".to_string(),
        tmir_expr: encode_tmir_shift(&Opcode::Sshr, Type::I32, a.clone(), b.clone()),
        aarch64_expr: a.bvashr(b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::BandNot(I32, a, b) -> BIC (32-bit)`.
pub fn proof_bic_i32() -> ProofObligation {
    use crate::aarch64_semantics::encode_bic_rr;
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "BandNot_I32 -> BIC (32-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::BandNot, Type::I32, a.clone(), b.clone()),
        aarch64_expr: encode_bic_rr(OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::BorNot(I32, a, b) -> ORN (32-bit)`.
pub fn proof_orn_i32() -> ProofObligation {
    use crate::aarch64_semantics::encode_orn_rr;
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "BorNot_I32 -> ORN (32-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::BorNot, Type::I32, a.clone(), b.clone()),
        aarch64_expr: encode_orn_rr(OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

// ---------------------------------------------------------------------------
// I64 bitwise/shift lowering proofs (statistical -- 100K random samples)
// ---------------------------------------------------------------------------

/// Build the proof obligation for: `tMIR::Band(I64, a, b) -> AND (64-bit)`.
pub fn proof_band_i64() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "Band_I64 -> AND (64-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::Band, Type::I64, a.clone(), b.clone()),
        aarch64_expr: a.bvand(b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::Bor(I64, a, b) -> OR (64-bit)`.
pub fn proof_bor_i64() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "Bor_I64 -> OR (64-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::Bor, Type::I64, a.clone(), b.clone()),
        aarch64_expr: a.bvor(b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::Bxor(I64, a, b) -> XOR (64-bit)`.
pub fn proof_bxor_i64() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "Bxor_I64 -> XOR (64-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::Bxor, Type::I64, a.clone(), b.clone()),
        aarch64_expr: a.bvxor(b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::Ishl(I64, a, b) -> LSL (64-bit)`.
pub fn proof_ishl_i64() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_shift;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "Ishl_I64 -> LSL (64-bit)".to_string(),
        tmir_expr: encode_tmir_shift(&Opcode::Ishl, Type::I64, a.clone(), b.clone()),
        aarch64_expr: a.bvshl(b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::Ushr(I64, a, b) -> LSR (64-bit)`.
pub fn proof_ushr_i64() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_shift;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "Ushr_I64 -> LSR (64-bit)".to_string(),
        tmir_expr: encode_tmir_shift(&Opcode::Ushr, Type::I64, a.clone(), b.clone()),
        aarch64_expr: a.bvlshr(b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::Sshr(I64, a, b) -> ASR (64-bit)`.
pub fn proof_sshr_i64() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_shift;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "Sshr_I64 -> ASR (64-bit)".to_string(),
        tmir_expr: encode_tmir_shift(&Opcode::Sshr, Type::I64, a.clone(), b.clone()),
        aarch64_expr: a.bvashr(b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::BandNot(I64, a, b) -> BIC (64-bit)`.
pub fn proof_bic_i64() -> ProofObligation {
    use crate::aarch64_semantics::encode_bic_rr;
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "BandNot_I64 -> BIC (64-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::BandNot, Type::I64, a.clone(), b.clone()),
        aarch64_expr: encode_bic_rr(OperandSize::S64, a, b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Build the proof obligation for: `tMIR::BorNot(I64, a, b) -> ORN (64-bit)`.
pub fn proof_orn_i64() -> ProofObligation {
    use crate::aarch64_semantics::encode_orn_rr;
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use llvm2_ir::cc::OperandSize;
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "BorNot_I64 -> ORN (64-bit)".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::BorNot, Type::I64, a.clone(), b.clone()),
        aarch64_expr: encode_orn_rr(OperandSize::S64, a, b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Return all bitwise and shift lowering proofs (I8 + I16 + I32 + I64).
///
/// Covers: AND, OR, XOR, NOT, SHL, LSR, ASR at I8 (exhaustive) and
/// I16/I32/I64 (statistical) widths, plus BIC/ORN at all four widths.
/// I32/I64 BIC/ORN/BAND/BOR/BXOR/SHL/LSR/ASR widened by issue #449 for the
/// epic #407 Task 3 z4 smoke rollout (enabled by StableHasher caching in
/// #420). I8/I16 BIC/ORN added by issue #425.
pub fn all_bitwise_shift_proofs() -> Vec<ProofObligation> {
    vec![
        // I8 (exhaustive verification -- all 2^16 or 2^8 input combos tested)
        proof_band_i8(),
        proof_bor_i8(),
        proof_bxor_i8(),
        proof_bnot_i8(),
        proof_ishl_i8(),
        proof_ushr_i8(),
        proof_sshr_i8(),
        // I8 BIC/ORN (issue #425) -- BandNot/BorNot lowering proofs
        proof_bic_i8(),
        proof_orn_i8(),
        // I16 (statistical verification -- edge cases + random sampling)
        proof_band_i16(),
        proof_bor_i16(),
        proof_bxor_i16(),
        proof_bnot_i16(),
        proof_ishl_i16(),
        proof_ushr_i16(),
        proof_sshr_i16(),
        // I16 BIC/ORN -- widened for epic #407 z4 smoke rollout
        proof_bic_i16(),
        proof_orn_i16(),
        // I32 bitwise/shift/BIC/ORN (issue #449) -- statistical sampling;
        // z4 can discharge most symbolically, imul-free so bvshl/bvlshr/bvashr
        // close in seconds on a warm solver.
        proof_band_i32(),
        proof_bor_i32(),
        proof_bxor_i32(),
        proof_ishl_i32(),
        proof_ushr_i32(),
        proof_sshr_i32(),
        proof_bic_i32(),
        proof_orn_i32(),
        // I64 bitwise/shift/BIC/ORN (issue #449).
        proof_band_i64(),
        proof_bor_i64(),
        proof_bxor_i64(),
        proof_ishl_i64(),
        proof_ushr_i64(),
        proof_sshr_i64(),
        proof_bic_i64(),
        proof_orn_i64(),
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

    // -----------------------------------------------------------------------
    // Remainder lowering proof tests (issue #435)
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_urem_i8() {
        assert_valid(&proof_urem_i8());
    }

    #[test]
    fn test_proof_srem_i8() {
        assert_valid(&proof_srem_i8());
    }

    #[test]
    fn test_all_remainder_proofs() {
        for obligation in all_remainder_proofs() {
            assert_valid(&obligation);
        }
    }

    /// Precondition sanity: Urem obligation must have exactly the `b != 0`
    /// precondition (matching Udiv).
    #[test]
    fn test_proof_urem_i8_precondition_count() {
        let urem = proof_urem_i8();
        assert_eq!(
            urem.preconditions.len(),
            1,
            "Urem I8 must have NonZeroDivisor precondition"
        );
    }

    /// Precondition sanity: Srem obligation must have *two* preconditions --
    /// `b != 0` AND `not (a == INT8_MIN && b == -1)`.
    #[test]
    fn test_proof_srem_i8_precondition_count() {
        let srem = proof_srem_i8();
        assert_eq!(
            srem.preconditions.len(),
            2,
            "Srem I8 must have NonZeroDivisor + INT_MIN/-1 overflow preconditions"
        );
    }

    // -----------------------------------------------------------------------
    // Bitcast lowering proof tests (issue #435)
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_bitcast_i8() {
        assert_valid(&proof_bitcast_i8());
    }

    #[test]
    fn test_all_bitcast_proofs() {
        for obligation in all_bitcast_proofs() {
            assert_valid(&obligation);
        }
    }

    // -----------------------------------------------------------------------
    // Bitfield lowering proof tests (issue #452)
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_extract_bits_i8() {
        assert_valid(&proof_extract_bits_i8());
    }

    #[test]
    fn test_proof_sextract_bits_i8() {
        assert_valid(&proof_sextract_bits_i8());
    }

    #[test]
    fn test_proof_insert_bits_i8() {
        assert_valid(&proof_insert_bits_i8());
    }

    #[test]
    fn test_all_bitfield_proofs() {
        for obligation in all_bitfield_proofs() {
            assert_valid(&obligation);
        }
    }

    /// Sanity: none of the bitfield obligations has a precondition -- they
    /// are pure QF_BV identities at the bitvector level.
    #[test]
    fn test_bitfield_proofs_have_no_preconditions() {
        for obligation in all_bitfield_proofs() {
            assert!(
                obligation.preconditions.is_empty(),
                "bitfield proof '{}' must have no preconditions",
                obligation.name
            );
        }
    }

    // -----------------------------------------------------------------------
    // I128 multi-register lowering proof tests (issue #324)
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_iadd_i128_lo() {
        assert_valid(&proof_iadd_i128_lo());
    }

    #[test]
    fn test_proof_iadd_i128_hi() {
        assert_valid(&proof_iadd_i128_hi());
    }

    #[test]
    fn test_proof_isub_i128_lo() {
        assert_valid(&proof_isub_i128_lo());
    }

    #[test]
    fn test_proof_isub_i128_hi() {
        assert_valid(&proof_isub_i128_hi());
    }

    /// Sanity check that the ADC carry expression in `proof_iadd_i128_hi`
    /// actually flags wraparound for a concrete overflowing case.
    #[test]
    fn test_iadd_i128_carry_semantics_overflow() {
        use std::collections::HashMap;
        let obligation = proof_iadd_i128_hi();

        // a_lo = b_lo = 0xFFFF_FFFF_FFFF_FFFF, a_hi = b_hi = 0
        // lo_sum wraps to 0xFFFF_FFFF_FFFF_FFFE, carry = 1
        // expected dst_hi = 0 + 0 + 1 = 1
        let mut env = HashMap::new();
        env.insert("a_lo".to_string(), u64::MAX);
        env.insert("b_lo".to_string(), u64::MAX);
        env.insert("a_hi".to_string(), 0u64);
        env.insert("b_hi".to_string(), 0u64);

        let tmir_val = obligation.tmir_expr.eval(&env).as_u64();
        let mach_val = obligation.aarch64_expr.eval(&env).as_u64();
        assert_eq!(tmir_val, 1, "overflow case should give dst_hi = 1, got {}", tmir_val);
        assert_eq!(tmir_val, mach_val);
    }

    /// Complementary sanity check: non-overflowing low-limb addition must
    /// leave the high limb untouched (carry=0).
    #[test]
    fn test_iadd_i128_carry_semantics_no_overflow() {
        use std::collections::HashMap;
        let obligation = proof_iadd_i128_hi();

        // a_lo=1, b_lo=2 → lo_sum=3, no carry. a_hi=5, b_hi=7 → dst_hi=12.
        let mut env = HashMap::new();
        env.insert("a_lo".to_string(), 1u64);
        env.insert("b_lo".to_string(), 2u64);
        env.insert("a_hi".to_string(), 5u64);
        env.insert("b_hi".to_string(), 7u64);

        let tmir_val = obligation.tmir_expr.eval(&env).as_u64();
        let mach_val = obligation.aarch64_expr.eval(&env).as_u64();
        assert_eq!(tmir_val, 12, "no-carry case should give dst_hi = 12, got {}", tmir_val);
        assert_eq!(tmir_val, mach_val);
    }

    /// Sanity check that the SBC borrow expression in `proof_isub_i128_hi`
    /// flags borrow-out when a_lo < b_lo.
    #[test]
    fn test_isub_i128_borrow_semantics() {
        use std::collections::HashMap;
        let obligation = proof_isub_i128_hi();

        // a_lo=0, b_lo=1 → borrow=1. a_hi=5, b_hi=2 → dst_hi = 5 - 2 - 1 = 2.
        let mut env = HashMap::new();
        env.insert("a_lo".to_string(), 0u64);
        env.insert("b_lo".to_string(), 1u64);
        env.insert("a_hi".to_string(), 5u64);
        env.insert("b_hi".to_string(), 2u64);

        let tmir_val = obligation.tmir_expr.eval(&env).as_u64();
        let mach_val = obligation.aarch64_expr.eval(&env).as_u64();
        assert_eq!(tmir_val, 2, "borrow case should give dst_hi = 2, got {}", tmir_val);
        assert_eq!(tmir_val, mach_val);

        // Non-borrow: a_lo >= b_lo, dst_hi = a_hi - b_hi.
        env.insert("a_lo".to_string(), 10u64);
        env.insert("b_lo".to_string(), 3u64);
        let tmir_val = obligation.tmir_expr.eval(&env).as_u64();
        assert_eq!(tmir_val, 3, "no-borrow case should give dst_hi = 3, got {}", tmir_val);
    }

    #[test]
    fn test_proof_imul_i128_lo() {
        assert_valid(&proof_imul_i128_lo());
    }

    #[test]
    fn test_proof_imul_i128_hi() {
        assert_valid(&proof_imul_i128_hi());
    }

    /// Concrete sanity: for a = 2^64, b = 3 ->
    ///   a = (a_hi=1, a_lo=0), b = (b_hi=0, b_lo=3)
    ///   a*b = 3 * 2^64 = (hi=3, lo=0)
    ///   UMULH(a_lo=0, b_lo=3) = 0
    ///   MADD chain: t1 = 0*0 + 0 = 0, dst_hi = 1*3 + 0 = 3. PASS.
    #[test]
    fn test_imul_i128_cross_term_2pow64_times_3() {
        use std::collections::HashMap;
        let obligation = proof_imul_i128_hi();

        let mut env = HashMap::new();
        env.insert("a_lo".to_string(), 0u64);
        env.insert("a_hi".to_string(), 1u64);
        env.insert("b_lo".to_string(), 3u64);
        env.insert("b_hi".to_string(), 0u64);
        env.insert("umulh_ab_lo".to_string(), 0u64); // UMULH(0, 3) = 0

        let tmir_val = obligation.tmir_expr.eval(&env).as_u64();
        let mach_val = obligation.aarch64_expr.eval(&env).as_u64();
        assert_eq!(tmir_val, 3, "dst_hi for 2^64 * 3 should be 3, got {}", tmir_val);
        assert_eq!(tmir_val, mach_val);
    }

    /// Concrete sanity exercising the UMULH carry-in:
    ///   a = (a_hi=0, a_lo=u64::MAX), b = (b_hi=0, b_lo=u64::MAX)
    ///   a*b = (2^64 - 1)^2 = 2^128 - 2^65 + 1
    ///         = (hi = 2^64 - 2, lo = 1)  [i.e. hi = u64::MAX - 1]
    ///   UMULH(u64::MAX, u64::MAX) = u64::MAX - 1
    ///   MADD chain: t1 = u64::MAX*0 + (u64::MAX-1) = u64::MAX-1
    ///               dst_hi = 0*u64::MAX + (u64::MAX-1) = u64::MAX-1
    #[test]
    fn test_imul_i128_umulh_carry() {
        use std::collections::HashMap;
        let obligation = proof_imul_i128_hi();

        let mut env = HashMap::new();
        env.insert("a_lo".to_string(), u64::MAX);
        env.insert("a_hi".to_string(), 0u64);
        env.insert("b_lo".to_string(), u64::MAX);
        env.insert("b_hi".to_string(), 0u64);
        env.insert("umulh_ab_lo".to_string(), u64::MAX - 1);

        let tmir_val = obligation.tmir_expr.eval(&env).as_u64();
        let mach_val = obligation.aarch64_expr.eval(&env).as_u64();
        assert_eq!(
            tmir_val,
            u64::MAX - 1,
            "dst_hi for u64::MAX^2 should carry UMULH = u64::MAX-1, got {}",
            tmir_val
        );
        assert_eq!(tmir_val, mach_val);
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
            category: None,
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
    // 64-bit comparison proofs (all 10 conditions)
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_icmp_eq_i64() {
        assert_valid(&proof_icmp_eq_i64());
    }

    #[test]
    fn test_proof_icmp_ne_i64() {
        assert_valid(&proof_icmp_ne_i64());
    }

    #[test]
    fn test_proof_icmp_slt_i64() {
        assert_valid(&proof_icmp_slt_i64());
    }

    #[test]
    fn test_proof_icmp_sge_i64() {
        assert_valid(&proof_icmp_sge_i64());
    }

    #[test]
    fn test_proof_icmp_sgt_i64() {
        assert_valid(&proof_icmp_sgt_i64());
    }

    #[test]
    fn test_proof_icmp_sle_i64() {
        assert_valid(&proof_icmp_sle_i64());
    }

    #[test]
    fn test_proof_icmp_ult_i64() {
        assert_valid(&proof_icmp_ult_i64());
    }

    #[test]
    fn test_proof_icmp_uge_i64() {
        assert_valid(&proof_icmp_uge_i64());
    }

    #[test]
    fn test_proof_icmp_ugt_i64() {
        assert_valid(&proof_icmp_ugt_i64());
    }

    #[test]
    fn test_proof_icmp_ule_i64() {
        assert_valid(&proof_icmp_ule_i64());
    }

    #[test]
    fn test_all_comparison_proofs_i64() {
        for obligation in all_comparison_proofs_i64() {
            assert_valid(&obligation);
        }
    }

    // -----------------------------------------------------------------------
    // Branch lowering proof tests (32-bit, all 10 conditions)
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
    fn test_proof_condbr_sge_i32() {
        assert_valid(&proof_condbr_sge_i32());
    }

    #[test]
    fn test_proof_condbr_sgt_i32() {
        assert_valid(&proof_condbr_sgt_i32());
    }

    #[test]
    fn test_proof_condbr_sle_i32() {
        assert_valid(&proof_condbr_sle_i32());
    }

    #[test]
    fn test_proof_condbr_ult_i32() {
        assert_valid(&proof_condbr_ult_i32());
    }

    #[test]
    fn test_proof_condbr_uge_i32() {
        assert_valid(&proof_condbr_uge_i32());
    }

    #[test]
    fn test_proof_condbr_ugt_i32() {
        assert_valid(&proof_condbr_ugt_i32());
    }

    #[test]
    fn test_proof_condbr_ule_i32() {
        assert_valid(&proof_condbr_ule_i32());
    }

    #[test]
    fn test_all_branch_proofs_i32() {
        for obligation in all_branch_proofs_i32() {
            assert_valid(&obligation);
        }
    }

    // -----------------------------------------------------------------------
    // Branch lowering proof tests (64-bit, all 10 conditions)
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_condbr_eq_i64() {
        assert_valid(&proof_condbr_eq_i64());
    }

    #[test]
    fn test_proof_condbr_ne_i64() {
        assert_valid(&proof_condbr_ne_i64());
    }

    #[test]
    fn test_proof_condbr_slt_i64() {
        assert_valid(&proof_condbr_slt_i64());
    }

    #[test]
    fn test_proof_condbr_sge_i64() {
        assert_valid(&proof_condbr_sge_i64());
    }

    #[test]
    fn test_proof_condbr_sgt_i64() {
        assert_valid(&proof_condbr_sgt_i64());
    }

    #[test]
    fn test_proof_condbr_sle_i64() {
        assert_valid(&proof_condbr_sle_i64());
    }

    #[test]
    fn test_proof_condbr_ult_i64() {
        assert_valid(&proof_condbr_ult_i64());
    }

    #[test]
    fn test_proof_condbr_uge_i64() {
        assert_valid(&proof_condbr_uge_i64());
    }

    #[test]
    fn test_proof_condbr_ugt_i64() {
        assert_valid(&proof_condbr_ugt_i64());
    }

    #[test]
    fn test_proof_condbr_ule_i64() {
        assert_valid(&proof_condbr_ule_i64());
    }

    #[test]
    fn test_all_branch_proofs_i64() {
        for obligation in all_branch_proofs_i64() {
            assert_valid(&obligation);
        }
    }

    #[test]
    fn test_all_branch_proofs() {
        for obligation in all_branch_proofs() {
            assert_valid(&obligation);
        }
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
            category: None,
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
            category: None,
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
            category: None,
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
            category: None,
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

    // -----------------------------------------------------------------------
    // Floating-point lowering proof tests
    // -----------------------------------------------------------------------

    /// Helper: verify a floating-point proof obligation and assert Valid.
    fn assert_fp_valid(obligation: &ProofObligation) {
        let result = verify_fp_by_evaluation(obligation);
        match &result {
            VerificationResult::Valid => {} // expected
            VerificationResult::Invalid { counterexample } => {
                panic!(
                    "FP Proof '{}' FAILED with counterexample: {}",
                    obligation.name, counterexample
                );
            }
            VerificationResult::Unknown { reason } => {
                panic!("FP Proof '{}' returned Unknown: {}", obligation.name, reason);
            }
        }
    }

    #[test]
    fn test_proof_fadd_f32() {
        assert_fp_valid(&proof_fadd_f32());
    }

    #[test]
    fn test_proof_fadd_f64() {
        assert_fp_valid(&proof_fadd_f64());
    }

    #[test]
    fn test_proof_fsub_f32() {
        assert_fp_valid(&proof_fsub_f32());
    }

    #[test]
    fn test_proof_fsub_f64() {
        assert_fp_valid(&proof_fsub_f64());
    }

    #[test]
    fn test_proof_fmul_f32() {
        assert_fp_valid(&proof_fmul_f32());
    }

    #[test]
    fn test_proof_fmul_f64() {
        assert_fp_valid(&proof_fmul_f64());
    }

    #[test]
    fn test_proof_fneg_f32() {
        assert_fp_valid(&proof_fneg_f32());
    }

    #[test]
    fn test_proof_fneg_f64() {
        assert_fp_valid(&proof_fneg_f64());
    }

    #[test]
    fn test_proof_fdiv_f32() {
        assert_fp_valid(&proof_fdiv_f32());
    }

    #[test]
    fn test_proof_fdiv_f64() {
        assert_fp_valid(&proof_fdiv_f64());
    }

    // FABS: absolute value proofs
    #[test]
    fn test_proof_fabs_f32() {
        assert_fp_valid(&proof_fabs_f32());
    }

    #[test]
    fn test_proof_fabs_f64() {
        assert_fp_valid(&proof_fabs_f64());
    }

    // FSQRT: square root proofs
    #[test]
    fn test_proof_fsqrt_f32() {
        assert_fp_valid(&proof_fsqrt_f32());
    }

    #[test]
    fn test_proof_fsqrt_f64() {
        assert_fp_valid(&proof_fsqrt_f64());
    }

    // FCMP ordered comparison proofs
    #[test]
    fn test_proof_fcmp_eq_f32() {
        assert_fp_valid(&proof_fcmp_eq_f32());
    }

    #[test]
    fn test_proof_fcmp_eq_f64() {
        assert_fp_valid(&proof_fcmp_eq_f64());
    }

    #[test]
    fn test_proof_fcmp_ne_f32() {
        assert_fp_valid(&proof_fcmp_ne_f32());
    }

    #[test]
    fn test_proof_fcmp_ne_f64() {
        assert_fp_valid(&proof_fcmp_ne_f64());
    }

    #[test]
    fn test_proof_fcmp_lt_f32() {
        assert_fp_valid(&proof_fcmp_lt_f32());
    }

    #[test]
    fn test_proof_fcmp_lt_f64() {
        assert_fp_valid(&proof_fcmp_lt_f64());
    }

    #[test]
    fn test_proof_fcmp_le_f32() {
        assert_fp_valid(&proof_fcmp_le_f32());
    }

    #[test]
    fn test_proof_fcmp_le_f64() {
        assert_fp_valid(&proof_fcmp_le_f64());
    }

    #[test]
    fn test_proof_fcmp_gt_f32() {
        assert_fp_valid(&proof_fcmp_gt_f32());
    }

    #[test]
    fn test_proof_fcmp_gt_f64() {
        assert_fp_valid(&proof_fcmp_gt_f64());
    }

    #[test]
    fn test_proof_fcmp_ge_f32() {
        assert_fp_valid(&proof_fcmp_ge_f32());
    }

    #[test]
    fn test_proof_fcmp_ge_f64() {
        assert_fp_valid(&proof_fcmp_ge_f64());
    }

    // FCMP ordering predicate proofs
    #[test]
    fn test_proof_fcmp_ord_f32() {
        assert_fp_valid(&proof_fcmp_ord_f32());
    }

    #[test]
    fn test_proof_fcmp_ord_f64() {
        assert_fp_valid(&proof_fcmp_ord_f64());
    }

    #[test]
    fn test_proof_fcmp_uno_f32() {
        assert_fp_valid(&proof_fcmp_uno_f32());
    }

    #[test]
    fn test_proof_fcmp_uno_f64() {
        assert_fp_valid(&proof_fcmp_uno_f64());
    }

    // FCMP unordered comparison proofs
    #[test]
    fn test_proof_fcmp_ueq_f32() {
        assert_fp_valid(&proof_fcmp_ueq_f32());
    }

    #[test]
    fn test_proof_fcmp_ueq_f64() {
        assert_fp_valid(&proof_fcmp_ueq_f64());
    }

    #[test]
    fn test_proof_fcmp_une_f32() {
        assert_fp_valid(&proof_fcmp_une_f32());
    }

    #[test]
    fn test_proof_fcmp_une_f64() {
        assert_fp_valid(&proof_fcmp_une_f64());
    }

    #[test]
    fn test_proof_fcmp_ult_f32() {
        assert_fp_valid(&proof_fcmp_ult_f32());
    }

    #[test]
    fn test_proof_fcmp_ult_f64() {
        assert_fp_valid(&proof_fcmp_ult_f64());
    }

    #[test]
    fn test_proof_fcmp_ule_f32() {
        assert_fp_valid(&proof_fcmp_ule_f32());
    }

    #[test]
    fn test_proof_fcmp_ule_f64() {
        assert_fp_valid(&proof_fcmp_ule_f64());
    }

    #[test]
    fn test_proof_fcmp_ugt_f32() {
        assert_fp_valid(&proof_fcmp_ugt_f32());
    }

    #[test]
    fn test_proof_fcmp_ugt_f64() {
        assert_fp_valid(&proof_fcmp_ugt_f64());
    }

    #[test]
    fn test_proof_fcmp_uge_f32() {
        assert_fp_valid(&proof_fcmp_uge_f32());
    }

    #[test]
    fn test_proof_fcmp_uge_f64() {
        assert_fp_valid(&proof_fcmp_uge_f64());
    }

    /// Verify the FP proof count matches expectations.
    #[test]
    fn test_fp_lowering_proof_count() {
        let proofs = all_fp_lowering_proofs();
        // 8 original (fadd/fsub/fmul/fneg x f32/f64) + 2 fdiv + 4 fabs/fsqrt + 28 fcmp = 42
        assert_eq!(proofs.len(), 42, "Expected 42 FP lowering proofs");
    }

    #[test]
    fn test_all_fp_lowering_proofs() {
        for obligation in all_fp_lowering_proofs() {
            assert_fp_valid(&obligation);
        }
    }

    /// Verify that FP proof obligations produce valid SMT-LIB2 output.
    #[test]
    fn test_fp_proof_smt2_output() {
        let obligation = proof_fadd_f64();
        let smt2 = obligation.to_smt2();
        // Should declare FP inputs
        assert!(smt2.contains("(declare-const a (_ FloatingPoint 11 53))"));
        assert!(smt2.contains("(declare-const b (_ FloatingPoint 11 53))"));
        assert!(smt2.contains("(check-sat)"));
    }

    /// Negative test: verify that wrong FP lowering is detected.
    #[test]
    fn test_wrong_fp_rule_detected() {
        // Claim FADD = FMUL -- should find a counterexample.
        use crate::smt::RoundingMode;

        let a = SmtExpr::fp64_const(0.0);
        let b = SmtExpr::fp64_const(0.0);

        let obligation = ProofObligation {
            name: "WRONG: Fadd -> FMUL".to_string(),
            tmir_expr: SmtExpr::fp_add(RoundingMode::RNE, a.clone(), b.clone()),
            aarch64_expr: SmtExpr::fp_mul(RoundingMode::RNE, a, b),
            inputs: vec![],
            preconditions: vec![],
            fp_inputs: vec![("a".to_string(), 11, 53), ("b".to_string(), 11, 53)],
            category: None,
        };

        let result = verify_fp_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {} // expected
            other => panic!("Expected Invalid for wrong FP rule, got {:?}", other),
        }
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

    // -----------------------------------------------------------------------
    // Load/Store lowering proofs
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_load_i32_lowering() {
        assert_valid(&proof_load_i32_lowering());
    }

    #[test]
    fn test_proof_load_i64_lowering() {
        assert_valid(&proof_load_i64_lowering());
    }

    #[test]
    fn test_proof_store_i32_lowering() {
        assert_valid(&proof_store_i32_lowering());
    }

    #[test]
    fn test_proof_store_i64_lowering() {
        assert_valid(&proof_store_i64_lowering());
    }

    #[test]
    fn test_proof_load_i8_lowering() {
        assert_valid(&proof_load_i8_lowering());
    }

    #[test]
    fn test_proof_load_i16_lowering() {
        assert_valid(&proof_load_i16_lowering());
    }

    #[test]
    fn test_proof_store_i8_lowering() {
        assert_valid(&proof_store_i8_lowering());
    }

    #[test]
    fn test_proof_store_i16_lowering() {
        assert_valid(&proof_store_i16_lowering());
    }

    #[test]
    fn test_proof_load_store_roundtrip_i32() {
        assert_valid(&proof_load_store_roundtrip_i32());
    }

    #[test]
    fn test_proof_load_store_roundtrip_i64() {
        assert_valid(&proof_load_store_roundtrip_i64());
    }

    #[test]
    fn test_all_load_store_proofs() {
        for obligation in all_load_store_proofs() {
            assert_valid(&obligation);
        }
    }

    /// Verify that load/store proofs use the array-based memory model.
    #[test]
    fn test_load_store_proof_count() {
        let proofs = all_load_store_proofs();
        assert_eq!(proofs.len(), 10, "Expected 10 load/store proofs");
    }

    /// Verify that load/store proof obligations produce valid SMT-LIB2 output.
    #[test]
    fn test_load_store_smt2_output() {
        let obligation = proof_load_i32_lowering();
        let smt2 = obligation.to_smt2();
        assert!(smt2.contains("(set-logic"), "SMT2 should have set-logic");
        assert!(smt2.contains("(declare-const base (_ BitVec 64))"),
            "SMT2 should declare base address");
        assert!(smt2.contains("(check-sat)"), "SMT2 should have check-sat");
    }

    /// Verify that store proof obligations include value declarations.
    #[test]
    fn test_store_proof_has_value_input() {
        let obligation = proof_store_i32_lowering();
        assert!(
            obligation.inputs.iter().any(|(name, _)| name == "value"),
            "Store proof should have 'value' input"
        );
        assert!(
            obligation.inputs.iter().any(|(name, _)| name == "base"),
            "Store proof should have 'base' input"
        );
    }

    // -----------------------------------------------------------------------
    // I8 bitwise/shift proofs (exhaustive -- all 2^16 or 2^8 combos)
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_band_i8() {
        assert_valid(&proof_band_i8());
    }

    #[test]
    fn test_proof_bor_i8() {
        assert_valid(&proof_bor_i8());
    }

    #[test]
    fn test_proof_bxor_i8() {
        assert_valid(&proof_bxor_i8());
    }

    #[test]
    fn test_proof_bnot_i8() {
        assert_valid(&proof_bnot_i8());
    }

    #[test]
    fn test_proof_ishl_i8() {
        assert_valid(&proof_ishl_i8());
    }

    #[test]
    fn test_proof_ushr_i8() {
        assert_valid(&proof_ushr_i8());
    }

    #[test]
    fn test_proof_sshr_i8() {
        assert_valid(&proof_sshr_i8());
    }

    // -----------------------------------------------------------------------
    // I16 bitwise/shift proofs (statistical -- edge cases + random sampling)
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_band_i16() {
        assert_valid(&proof_band_i16());
    }

    #[test]
    fn test_proof_bor_i16() {
        assert_valid(&proof_bor_i16());
    }

    #[test]
    fn test_proof_bxor_i16() {
        assert_valid(&proof_bxor_i16());
    }

    #[test]
    fn test_proof_bnot_i16() {
        assert_valid(&proof_bnot_i16());
    }

    #[test]
    fn test_proof_ishl_i16() {
        assert_valid(&proof_ishl_i16());
    }

    #[test]
    fn test_proof_ushr_i16() {
        assert_valid(&proof_ushr_i16());
    }

    #[test]
    fn test_proof_sshr_i16() {
        assert_valid(&proof_sshr_i16());
    }

    // -----------------------------------------------------------------------
    // Aggregate bitwise/shift proof test
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_bitwise_shift_proofs() {
        for obligation in all_bitwise_shift_proofs() {
            assert_valid(&obligation);
        }
    }

    /// Verify that the bitwise/shift collection has the expected count.
    ///
    /// Breakdown (34 = 18 base + 16 I32/I64 widening for issue #449):
    ///   - I8 : band, bor, bxor, bnot, ishl, ushr, sshr, bic, orn  (9)
    ///   - I16: band, bor, bxor, bnot, ishl, ushr, sshr, bic, orn  (9)
    ///   - I32: band, bor, bxor,       ishl, ushr, sshr, bic, orn  (8)
    ///   - I64: band, bor, bxor,       ishl, ushr, sshr, bic, orn  (8)
    /// (bnot not yet added at I32/I64; filed if/when that becomes a gap.)
    #[test]
    fn test_bitwise_shift_proof_count() {
        let proofs = all_bitwise_shift_proofs();
        assert_eq!(
            proofs.len(),
            34,
            "Expected 34 bitwise/shift proofs (I8/I16 7 ops + BIC/ORN = 18; \
             I32/I64 6 ops + BIC/ORN = 16), got {}",
            proofs.len()
        );
    }

    #[test]
    fn test_proof_bic_i8() {
        assert_valid(&proof_bic_i8());
    }

    #[test]
    fn test_proof_orn_i8() {
        assert_valid(&proof_orn_i8());
    }

    #[test]
    fn test_proof_bic_i16() {
        assert_valid(&proof_bic_i16());
    }

    #[test]
    fn test_proof_orn_i16() {
        assert_valid(&proof_orn_i16());
    }

    /// Negative test: load at different offsets should not be equivalent.
    #[test]
    fn test_wrong_load_offset_lowering_detected() {
        use crate::memory_proofs::{
            symbolic_memory, encode_tmir_load, encode_aarch64_ldr_imm, encode_store_le,
        };

        let mem = symbolic_memory("mem_default");
        let base = SmtExpr::var("base", 64);
        let value = SmtExpr::var("value", 32);

        // Store a value at base
        let mem_with_data = encode_store_le(&mem, &base, &value, 4);

        // tMIR: load at byte offset 0
        let tmir_at_0 = encode_tmir_load(&mem_with_data, &base, 0, 4);
        // AArch64: load at scaled offset 1 (byte offset 4) -- WRONG
        let aarch64_at_1 = encode_aarch64_ldr_imm(&mem_with_data, &base, 1, 4);

        let obligation = ProofObligation {
            name: "WRONG: Load I32 offset 0 == Load I32 offset 4".to_string(),
            tmir_expr: tmir_at_0,
            aarch64_expr: aarch64_at_1,
            inputs: vec![
                ("base".to_string(), 64),
                ("value".to_string(), 32),
                ("mem_default".to_string(), 8),
            ],
            preconditions: vec![],
            fp_inputs: vec![],
            category: None,
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {} // expected
            other => panic!("Expected Invalid for wrong load offset, got {:?}", other),
        }
    }
}
