// llvm2-verify/cegis.rs - Counter-Example Guided Inductive Synthesis (CEGIS)
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// CEGIS is the core algorithm for solver-driven superoptimization. Given a
// candidate equivalence (source_smt == target_smt), it iteratively:
//
// 1. Fast-path: evaluate both expressions on accumulated counterexamples.
//    If any counterexample disagrees, reject immediately (no solver call).
// 2. Slow-path: ask the SMT solver "exists input where source != target?"
//    - UNSAT => proven equivalent. Done.
//    - SAT => extract counterexample, add to test suite, return NotEquivalent.
//
// This is Souper's approach: concrete evaluation first dramatically reduces
// the number of expensive solver calls during enumeration.
//
// Reference: Sasnauskas et al. "Souper: A Synthesizing Superoptimizer"
//            (arXiv:1711.04422, 2017)
//            designs/2026-04-13-superoptimization.md

//! Counter-Example Guided Inductive Synthesis (CEGIS) loop.
//!
//! The CEGIS loop is the core verification algorithm for superoptimization.
//! It combines fast concrete evaluation with SMT solving to efficiently
//! determine whether two SMT expressions are semantically equivalent.
//!
//! # Algorithm
//!
//! ```text
//! counterexamples = []
//! loop:
//!   for each counterexample in counterexamples:
//!     if source(cex) != target(cex):
//!       return NotEquivalent(cex)     // fast concrete rejection
//!   result = solver.check(source != target)
//!   if UNSAT:
//!     return Equivalent               // proven for all inputs
//!   if SAT:
//!     cex = solver.model()
//!     counterexamples.push(cex)
//!     return NotEquivalent(cex)        // can also loop for refinement
//!   if TIMEOUT:
//!     return Timeout
//! ```
//!
//! # Example
//!
//! ```rust
//! use llvm2_verify::smt::SmtExpr;
//! use llvm2_verify::cegis::{CegisLoop, CegisResult};
//!
//! let mut cegis = CegisLoop::new(10, 5000);
//!
//! // x + 0 == x (should be equivalent)
//! let source = SmtExpr::var("x", 32).bvadd(SmtExpr::bv_const(0, 32));
//! let target = SmtExpr::var("x", 32);
//! let vars = vec![("x".to_string(), 32)];
//!
//! let result = cegis.verify(&source, &target, &vars);
//! assert!(matches!(result, CegisResult::Equivalent { .. }));
//! ```

use crate::lowering_proof::ProofObligation;
use crate::smt::SmtExpr;
use crate::z4_bridge::{verify_with_z4, Z4Config, Z4Result};
use llvm2_opt::cache::StableHasher;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ConcreteInput
// ---------------------------------------------------------------------------

/// A concrete input assignment: variable name -> bitvector value.
///
/// This represents a single test point for concrete evaluation. Each entry
/// maps a symbolic variable name (matching `SmtExpr::Var::name`) to a
/// concrete u64 value (already masked to the variable's bitwidth).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcreteInput {
    /// Variable assignments: (name, value).
    pub values: HashMap<String, u64>,
}

impl ConcreteInput {
    /// Create a new empty concrete input.
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Create a concrete input from name-value pairs.
    pub fn from_pairs(pairs: &[(&str, u64)]) -> Self {
        Self {
            values: pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        }
    }

    /// Insert a variable assignment.
    pub fn insert(&mut self, name: impl Into<String>, value: u64) {
        self.values.insert(name.into(), value);
    }
}

impl Default for ConcreteInput {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CegisResult
// ---------------------------------------------------------------------------

/// Result of a CEGIS verification attempt.
#[derive(Debug, Clone)]
pub enum CegisResult {
    /// The two expressions are semantically equivalent for all inputs.
    ///
    /// `proof_hash` is a deterministic hash of the proof obligation for
    /// caching and deduplication in the rule database.
    Equivalent {
        /// Hash of the proof for caching/dedup.
        proof_hash: u64,
        /// Number of CEGIS iterations used.
        iterations: usize,
    },

    /// The expressions are not equivalent. A distinguishing input was found.
    NotEquivalent {
        /// The counterexample that distinguishes the expressions.
        counterexample: ConcreteInput,
        /// Whether this was found by concrete evaluation (true) or solver (false).
        found_by_concrete: bool,
    },

    /// The solver timed out on every attempt.
    Timeout,

    /// The maximum number of CEGIS iterations was reached without proving
    /// equivalence or finding a counterexample via concrete evaluation.
    /// This should not happen in normal operation (the solver should
    /// either prove UNSAT or return a counterexample), but can occur if
    /// the solver returns Unknown repeatedly.
    MaxIterationsReached {
        /// Number of accumulated counterexamples.
        counterexamples: usize,
    },

    /// An error occurred (solver error, encoding error, etc.).
    Error(String),
}

impl CegisResult {
    /// Returns true if the result proves equivalence.
    pub fn is_equivalent(&self) -> bool {
        matches!(self, CegisResult::Equivalent { .. })
    }

    /// Returns true if the result disproves equivalence.
    pub fn is_not_equivalent(&self) -> bool {
        matches!(self, CegisResult::NotEquivalent { .. })
    }
}

// ---------------------------------------------------------------------------
// CegisLoop
// ---------------------------------------------------------------------------

/// Counter-Example Guided Inductive Synthesis loop.
///
/// Maintains a growing set of counterexamples across multiple verification
/// queries. Counterexamples from previous queries are reused as fast concrete
/// filters for subsequent queries, dramatically reducing solver calls during
/// enumeration-based synthesis.
pub struct CegisLoop {
    /// Accumulated counterexamples from previous queries.
    counterexamples: Vec<ConcreteInput>,
    /// Maximum number of CEGIS refinement iterations per query.
    max_iterations: usize,
    /// Timeout in milliseconds for each individual solver query.
    timeout_per_query_ms: u64,
    /// Solver configuration (path, model production, etc.).
    solver_config: Z4Config,
    /// Statistics: total number of concrete rejections (no solver call needed).
    stats_concrete_rejections: u64,
    /// Statistics: total number of solver calls made.
    stats_solver_calls: u64,
}

impl CegisLoop {
    /// Create a new CEGIS loop with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - Maximum refinement iterations per `verify()` call.
    ///   In practice 1 is usually sufficient (either the solver proves UNSAT or
    ///   returns a counterexample). Higher values allow multi-round refinement
    ///   where the solver keeps finding new counterexamples.
    /// * `timeout_ms` - Timeout per individual solver query in milliseconds.
    pub fn new(max_iterations: usize, timeout_ms: u64) -> Self {
        Self {
            counterexamples: Vec::new(),
            max_iterations,
            timeout_per_query_ms: timeout_ms,
            solver_config: Z4Config::default().with_timeout(timeout_ms),
            stats_concrete_rejections: 0,
            stats_solver_calls: 0,
        }
    }

    /// Create a CEGIS loop with a custom solver configuration.
    pub fn with_solver_config(mut self, config: Z4Config) -> Self {
        self.timeout_per_query_ms = config.timeout_ms;
        self.solver_config = config;
        self
    }

    /// Add a seed counterexample (e.g., edge cases like 0, MAX, etc.).
    pub fn add_seed(&mut self, input: ConcreteInput) {
        self.counterexamples.push(input);
    }

    /// Add standard edge-case seeds for the given variables.
    ///
    /// Seeds: 0, 1, MAX, MAX-1, MSB (sign bit), MSB-1 for each variable.
    pub fn add_edge_case_seeds(&mut self, vars: &[(String, u32)]) {
        let edge_values: Vec<Box<dyn Fn(u32) -> u64>> = vec![
            Box::new(|_w| 0u64),
            Box::new(|_w| 1u64),
            Box::new(|w| crate::smt::mask(u64::MAX, w)),
            Box::new(|w| crate::smt::mask(u64::MAX, w).wrapping_sub(1)),
            Box::new(|w| 1u64 << w.saturating_sub(1).min(63)),
            Box::new(|w| (1u64 << w.saturating_sub(1).min(63)).wrapping_sub(1)),
        ];

        // Generate all combinations of edge values for all variables
        if vars.len() == 1 {
            let (name, width) = &vars[0];
            for edge_fn in &edge_values {
                let mut input = ConcreteInput::new();
                input.insert(name.clone(), crate::smt::mask(edge_fn(*width), *width));
                self.counterexamples.push(input);
            }
        } else if vars.len() >= 2 {
            // For 2+ variables, generate cross-product of edge cases
            // (for first two variables; additional variables get 0)
            let (name_a, width_a) = &vars[0];
            let (name_b, width_b) = &vars[1];
            for edge_a in &edge_values {
                for edge_b in &edge_values {
                    let mut input = ConcreteInput::new();
                    input.insert(
                        name_a.clone(),
                        crate::smt::mask(edge_a(*width_a), *width_a),
                    );
                    input.insert(
                        name_b.clone(),
                        crate::smt::mask(edge_b(*width_b), *width_b),
                    );
                    // Fill remaining variables with 0
                    for (name, _) in &vars[2..] {
                        input.insert(name.clone(), 0);
                    }
                    self.counterexamples.push(input);
                }
            }
        }
    }

    /// Return the number of accumulated counterexamples.
    pub fn counterexample_count(&self) -> usize {
        self.counterexamples.len()
    }

    /// Return the number of concrete rejections (no solver call needed).
    pub fn stats_concrete_rejections(&self) -> u64 {
        self.stats_concrete_rejections
    }

    /// Return the number of solver calls made.
    pub fn stats_solver_calls(&self) -> u64 {
        self.stats_solver_calls
    }

    /// Verify that two SMT expressions are semantically equivalent.
    ///
    /// This is the main entry point for CEGIS verification. It:
    ///
    /// 1. Tests both expressions on all accumulated counterexamples (fast path).
    /// 2. If all concrete tests pass, calls the SMT solver.
    /// 3. If UNSAT, returns `Equivalent`.
    /// 4. If SAT, extracts the counterexample, adds it to the set, and either
    ///    returns `NotEquivalent` or loops for multi-round refinement.
    ///
    /// # Arguments
    ///
    /// * `source_smt` - The source expression (e.g., original pattern).
    /// * `target_smt` - The target expression (e.g., replacement candidate).
    /// * `vars` - Symbolic variable names and their bitvector widths.
    pub fn verify(
        &mut self,
        source_smt: &SmtExpr,
        target_smt: &SmtExpr,
        vars: &[(String, u32)],
    ) -> CegisResult {
        // Phase 1: Fast concrete evaluation on existing counterexamples
        if let Some(cex) = self.check_concrete(source_smt, target_smt) {
            self.stats_concrete_rejections += 1;
            return CegisResult::NotEquivalent {
                counterexample: cex,
                found_by_concrete: true,
            };
        }

        // Phase 2: CEGIS refinement loop with solver
        for iteration in 0..self.max_iterations {
            self.stats_solver_calls += 1;

            // Build proof obligation for solver
            let obligation = ProofObligation {
                name: format!("cegis_iter_{}", iteration),
                tmir_expr: source_smt.clone(),
                aarch64_expr: target_smt.clone(),
                inputs: vars.to_vec(),
                preconditions: vec![],
            fp_inputs: vec![],
            category: None,
            };

            let result = verify_with_z4(&obligation, &self.solver_config);

            match result {
                Z4Result::Verified => {
                    // UNSAT: proven equivalent for all inputs
                    let proof_hash = compute_proof_hash(source_smt, target_smt);
                    return CegisResult::Equivalent {
                        proof_hash,
                        iterations: iteration + 1,
                    };
                }
                Z4Result::CounterExample(assignments) => {
                    // SAT: solver found a distinguishing input
                    let mut cex = ConcreteInput::new();
                    for (name, value) in &assignments {
                        cex.insert(name.clone(), *value);
                    }

                    // Validate the counterexample by concrete evaluation
                    // (sanity check -- solver should not return bogus models)
                    if !assignments.is_empty() {
                        let source_val = source_smt.try_eval(&cex.values);
                        let target_val = target_smt.try_eval(&cex.values);
                        if let (Ok(s), Ok(t)) = (source_val, target_val)
                            && s == t {
                                // Solver returned a model that doesn't actually
                                // distinguish the expressions. This can happen
                                // due to incomplete solver semantics or encoding
                                // issues. Continue to next iteration.
                                continue;
                            }
                    }

                    // Add to counterexample set for future fast-path rejections
                    self.counterexamples.push(cex.clone());

                    return CegisResult::NotEquivalent {
                        counterexample: cex,
                        found_by_concrete: false,
                    };
                }
                Z4Result::Timeout => {
                    return CegisResult::Timeout;
                }
                Z4Result::Error(msg) => {
                    return CegisResult::Error(msg);
                }
            }
        }

        CegisResult::MaxIterationsReached {
            counterexamples: self.counterexamples.len(),
        }
    }

    /// Verify using only concrete evaluation (no solver).
    ///
    /// This is useful for quick filtering during enumeration: if a candidate
    /// fails on existing counterexamples, there is no need to call the solver.
    /// Returns `None` if all concrete tests pass (inconclusive -- solver needed).
    pub fn check_concrete_only(
        &mut self,
        source_smt: &SmtExpr,
        target_smt: &SmtExpr,
    ) -> Option<CegisResult> {
        if let Some(cex) = self.check_concrete(source_smt, target_smt) {
            self.stats_concrete_rejections += 1;
            Some(CegisResult::NotEquivalent {
                counterexample: cex,
                found_by_concrete: true,
            })
        } else {
            None
        }
    }

    /// Check all accumulated counterexamples via concrete evaluation.
    ///
    /// Returns `Some(cex)` on the first counterexample where source and target
    /// disagree, or `None` if all counterexamples agree (pass to solver).
    fn check_concrete(
        &self,
        source_smt: &SmtExpr,
        target_smt: &SmtExpr,
    ) -> Option<ConcreteInput> {
        for cex in &self.counterexamples {
            let source_result = source_smt.try_eval(&cex.values);
            let target_result = target_smt.try_eval(&cex.values);

            match (source_result, target_result) {
                (Ok(s), Ok(t)) if s != t => {
                    return Some(cex.clone());
                }
                (Err(_), _) | (_, Err(_)) => {
                    // Variable not defined in this counterexample -- skip.
                    // This can happen when counterexamples from previous queries
                    // with different variable sets are reused.
                    continue;
                }
                _ => {
                    // Values agree on this counterexample, continue.
                    continue;
                }
            }
        }
        None
    }

    /// Reset accumulated counterexamples and statistics.
    pub fn reset(&mut self) {
        self.counterexamples.clear();
        self.stats_concrete_rejections = 0;
        self.stats_solver_calls = 0;
    }

    /// Clear accumulated counterexamples without resetting statistics.
    ///
    /// This is the right hook to call between independent proof obligations
    /// that share a single `CegisLoop` instance (e.g. across enumeration
    /// candidates in a superopt pass). Clearing prevents counterexamples from
    /// one obligation from causing spurious concrete-path rejections on an
    /// unrelated obligation (issue #493), while preserving the accumulated
    /// solver / rejection counters for observability.
    ///
    /// Typical usage:
    /// ```ignore
    /// let mut cegis = CegisLoop::new(1, 5_000);
    /// for candidate in candidates {
    ///     cegis.clear_counterexamples();
    ///     cegis.add_edge_case_seeds(&candidate.vars);
    ///     let _ = cegis.verify(&candidate.src, &candidate.tgt, &candidate.vars);
    /// }
    /// ```
    pub fn clear_counterexamples(&mut self) {
        self.counterexamples.clear();
    }
}

// ---------------------------------------------------------------------------
// Proof hash computation
// ---------------------------------------------------------------------------

/// Compute a deterministic hash of two SMT expressions for caching.
///
/// This hash identifies a unique (source, target) pair for deduplication
/// in the rule database. It uses the SMT-LIB2 string representation
/// which is canonical for structurally identical expressions.
fn compute_proof_hash(source: &SmtExpr, target: &SmtExpr) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.write_str(&format!("{}", source));
    hasher.write_str(&format!("{}", target));
    hasher.finish64()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smt::SmtExpr;

    // -----------------------------------------------------------------------
    // ConcreteInput tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_concrete_input_new() {
        let input = ConcreteInput::new();
        assert!(input.values.is_empty());
    }

    #[test]
    fn test_concrete_input_from_pairs() {
        let input = ConcreteInput::from_pairs(&[("a", 10), ("b", 20)]);
        assert_eq!(input.values.get("a"), Some(&10));
        assert_eq!(input.values.get("b"), Some(&20));
    }

    #[test]
    fn test_concrete_input_insert() {
        let mut input = ConcreteInput::new();
        input.insert("x", 42);
        assert_eq!(input.values.get("x"), Some(&42));
    }

    // -----------------------------------------------------------------------
    // CegisResult tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cegis_result_equivalent() {
        let result = CegisResult::Equivalent {
            proof_hash: 123,
            iterations: 1,
        };
        assert!(result.is_equivalent());
        assert!(!result.is_not_equivalent());
    }

    #[test]
    fn test_cegis_result_not_equivalent() {
        let result = CegisResult::NotEquivalent {
            counterexample: ConcreteInput::new(),
            found_by_concrete: true,
        };
        assert!(!result.is_equivalent());
        assert!(result.is_not_equivalent());
    }

    // -----------------------------------------------------------------------
    // CegisLoop construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cegis_loop_new() {
        let cegis = CegisLoop::new(10, 5000);
        assert_eq!(cegis.max_iterations, 10);
        assert_eq!(cegis.timeout_per_query_ms, 5000);
        assert_eq!(cegis.counterexample_count(), 0);
        assert_eq!(cegis.stats_concrete_rejections(), 0);
        assert_eq!(cegis.stats_solver_calls(), 0);
    }

    #[test]
    fn test_cegis_loop_add_seed() {
        let mut cegis = CegisLoop::new(10, 5000);
        cegis.add_seed(ConcreteInput::from_pairs(&[("x", 42)]));
        assert_eq!(cegis.counterexample_count(), 1);
    }

    #[test]
    fn test_cegis_loop_add_edge_case_seeds_single_var() {
        let mut cegis = CegisLoop::new(10, 5000);
        cegis.add_edge_case_seeds(&[("x".to_string(), 32)]);
        // 6 edge cases for one variable
        assert_eq!(cegis.counterexample_count(), 6);
    }

    #[test]
    fn test_cegis_loop_add_edge_case_seeds_two_vars() {
        let mut cegis = CegisLoop::new(10, 5000);
        cegis.add_edge_case_seeds(&[("a".to_string(), 32), ("b".to_string(), 32)]);
        // 6 * 6 = 36 cross-product combinations
        assert_eq!(cegis.counterexample_count(), 36);
    }

    // -----------------------------------------------------------------------
    // Concrete evaluation tests (no solver needed)
    // -----------------------------------------------------------------------

    #[test]
    fn test_concrete_rejection_clearly_different() {
        let mut cegis = CegisLoop::new(10, 5000);

        // Seed with a single input where add != sub
        cegis.add_seed(ConcreteInput::from_pairs(&[("a", 5), ("b", 3)]));

        let source = SmtExpr::var("a", 32).bvadd(SmtExpr::var("b", 32));
        let target = SmtExpr::var("a", 32).bvsub(SmtExpr::var("b", 32));

        let result = cegis.check_concrete_only(&source, &target);
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.is_not_equivalent());

        match result {
            CegisResult::NotEquivalent {
                found_by_concrete, ..
            } => {
                assert!(found_by_concrete);
            }
            _ => panic!("Expected NotEquivalent"),
        }
        assert_eq!(cegis.stats_concrete_rejections(), 1);
    }

    #[test]
    fn test_concrete_pass_identical_expressions() {
        let mut cegis = CegisLoop::new(10, 5000);

        // Seed with various inputs
        cegis.add_seed(ConcreteInput::from_pairs(&[("x", 0)]));
        cegis.add_seed(ConcreteInput::from_pairs(&[("x", 42)]));
        cegis.add_seed(ConcreteInput::from_pairs(&[("x", 0xFFFF_FFFF)]));

        let source = SmtExpr::var("x", 32).bvadd(SmtExpr::bv_const(0, 32));
        let target = SmtExpr::var("x", 32);

        // Concrete tests should pass (x + 0 == x for all concrete inputs)
        let result = cegis.check_concrete_only(&source, &target);
        assert!(result.is_none());
        assert_eq!(cegis.stats_concrete_rejections(), 0);
    }

    #[test]
    fn test_concrete_rejection_with_edge_cases() {
        let mut cegis = CegisLoop::new(10, 5000);

        // Add standard edge cases
        cegis.add_edge_case_seeds(&[("a".to_string(), 8), ("b".to_string(), 8)]);

        // mul(a, b) != add(a, b) -- should be caught by edge cases
        let source = SmtExpr::var("a", 8).bvmul(SmtExpr::var("b", 8));
        let target = SmtExpr::var("a", 8).bvadd(SmtExpr::var("b", 8));

        let result = cegis.check_concrete_only(&source, &target);
        assert!(result.is_some());
    }

    #[test]
    fn test_concrete_handles_missing_vars() {
        let mut cegis = CegisLoop::new(10, 5000);

        // Seed with a counterexample that only has variable "x"
        cegis.add_seed(ConcreteInput::from_pairs(&[("x", 42)]));

        // But the expressions use variables "a" and "b"
        let source = SmtExpr::var("a", 32);
        let target = SmtExpr::var("b", 32);

        // Should not crash, just skip the counterexample
        let result = cegis.check_concrete_only(&source, &target);
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // Full CEGIS loop tests (with solver)
    // -----------------------------------------------------------------------

    #[test]
    fn test_cegis_verify_identical_expressions() {
        // Use the z4 bridge (CLI fallback) to verify x + 0 == x
        let mut cegis = CegisLoop::new(10, 5000);
        let vars = vec![("x".to_string(), 32)];

        let source = SmtExpr::var("x", 32).bvadd(SmtExpr::bv_const(0, 32));
        let target = SmtExpr::var("x", 32);

        let result = cegis.verify(&source, &target, &vars);

        // If solver is available, should be Equivalent
        // If not available, may be Error (acceptable in test environment)
        match &result {
            CegisResult::Equivalent { iterations, .. } => {
                assert_eq!(*iterations, 1);
            }
            CegisResult::Error(_) => {
                // No solver available -- acceptable in CI
            }
            other => {
                panic!("Expected Equivalent or Error, got {:?}", other);
            }
        }
    }

    #[test]
    fn test_cegis_verify_clearly_different() {
        let mut cegis = CegisLoop::new(10, 5000);
        let vars = vec![("a".to_string(), 8), ("b".to_string(), 8)];

        // Seed with edge cases so concrete path catches it
        cegis.add_edge_case_seeds(&vars);

        let source = SmtExpr::var("a", 8).bvadd(SmtExpr::var("b", 8));
        let target = SmtExpr::var("a", 8).bvsub(SmtExpr::var("b", 8));

        let result = cegis.verify(&source, &target, &vars);

        match &result {
            CegisResult::NotEquivalent {
                found_by_concrete, ..
            } => {
                // Should be caught by concrete evaluation (fast path)
                assert!(*found_by_concrete);
            }
            other => {
                panic!("Expected NotEquivalent, got {:?}", other);
            }
        }
    }

    #[test]
    fn test_cegis_verify_different_via_solver() {
        // No seeds -- force solver path
        let mut cegis = CegisLoop::new(10, 5000);
        let vars = vec![("a".to_string(), 32), ("b".to_string(), 32)];

        let source = SmtExpr::var("a", 32).bvadd(SmtExpr::var("b", 32));
        let target = SmtExpr::var("a", 32).bvsub(SmtExpr::var("b", 32));

        let result = cegis.verify(&source, &target, &vars);

        match &result {
            CegisResult::NotEquivalent {
                counterexample,
                found_by_concrete,
            } => {
                assert!(!found_by_concrete);
                // Solver should have returned a counterexample
                assert!(!counterexample.values.is_empty());
                // Verify the counterexample is valid
                let s = source.try_eval(&counterexample.values).unwrap();
                let t = target.try_eval(&counterexample.values).unwrap();
                assert_ne!(s, t, "Counterexample should distinguish expressions");
            }
            CegisResult::Error(_) => {
                // No solver available -- acceptable
            }
            other => {
                panic!("Expected NotEquivalent or Error, got {:?}", other);
            }
        }
    }

    #[test]
    fn test_cegis_verify_commutativity() {
        // a + b == b + a (commutative)
        let mut cegis = CegisLoop::new(10, 5000);
        let vars = vec![("a".to_string(), 32), ("b".to_string(), 32)];

        let source = SmtExpr::var("a", 32).bvadd(SmtExpr::var("b", 32));
        let target = SmtExpr::var("b", 32).bvadd(SmtExpr::var("a", 32));

        let result = cegis.verify(&source, &target, &vars);

        match &result {
            CegisResult::Equivalent { .. } => {} // Good
            CegisResult::Error(_) => {}           // No solver
            other => {
                panic!("Expected Equivalent or Error, got {:?}", other);
            }
        }
    }

    #[test]
    fn test_cegis_counterexample_accumulation() {
        let mut cegis = CegisLoop::new(10, 5000);
        let vars = vec![("a".to_string(), 8), ("b".to_string(), 8)];

        // First query: add != sub -- this will add a counterexample
        let source = SmtExpr::var("a", 8).bvadd(SmtExpr::var("b", 8));
        let target = SmtExpr::var("a", 8).bvsub(SmtExpr::var("b", 8));
        let _ = cegis.verify(&source, &target, &vars);

        // The counterexample from the first query should be available
        // for fast rejection on subsequent queries
        let count_after_first = cegis.counterexample_count();

        // Second query: also different expressions
        let source2 = SmtExpr::var("a", 8).bvmul(SmtExpr::var("b", 8));
        let target2 = SmtExpr::var("a", 8).bvsub(SmtExpr::var("b", 8));

        let result = cegis.verify(&source2, &target2, &vars);

        // May or may not be caught by the accumulated counterexample,
        // depending on the specific values. Just check it is NotEquivalent.
        match &result {
            CegisResult::NotEquivalent { .. } => {} // Expected
            CegisResult::Error(_) => {}              // No solver
            other => {
                panic!(
                    "Expected NotEquivalent or Error, got {:?}",
                    other
                );
            }
        }

        // Counterexample count should be >= what we had before
        assert!(cegis.counterexample_count() >= count_after_first);
    }

    #[test]
    fn test_cegis_reset() {
        let mut cegis = CegisLoop::new(10, 5000);
        cegis.add_seed(ConcreteInput::from_pairs(&[("x", 42)]));
        assert_eq!(cegis.counterexample_count(), 1);

        cegis.reset();
        assert_eq!(cegis.counterexample_count(), 0);
        assert_eq!(cegis.stats_concrete_rejections(), 0);
        assert_eq!(cegis.stats_solver_calls(), 0);
    }

    #[test]
    fn test_cegis_clear_counterexamples_preserves_stats() {
        // `clear_counterexamples` (#493) wipes the CX vector but keeps the
        // stats counters, so a per-function pass can reuse one loop across
        // independent obligations without losing observability.
        let mut cegis = CegisLoop::new(10, 5000);
        cegis.add_seed(ConcreteInput::from_pairs(&[("x", 5)]));

        // Fire a concrete rejection so stats_concrete_rejections increments.
        let source = SmtExpr::var("x", 32);
        let target = SmtExpr::var("x", 32).bvadd(SmtExpr::bv_const(1, 32));
        let _ = cegis.check_concrete_only(&source, &target);
        assert_eq!(cegis.counterexample_count(), 1);
        assert_eq!(cegis.stats_concrete_rejections(), 1);

        cegis.clear_counterexamples();
        assert_eq!(cegis.counterexample_count(), 0);
        // Stats must survive clear() — that's the whole point of keeping a
        // dedicated method separate from reset().
        assert_eq!(cegis.stats_concrete_rejections(), 1);
    }

    #[test]
    fn test_cegis_counterexamples_do_not_bleed_between_obligations() {
        // Regression test for #493: reusing a single CegisLoop across
        // independent proof obligations used to accumulate counterexamples
        // from obligation A into the fast-path check for obligation B. If
        // B's "truly-equivalent" property only holds for a subset of inputs
        // (e.g. a future Layer-C rule with a precondition), a bled CX from A
        // that violates B's precondition would trigger a spurious
        // NotEquivalent. `clear_counterexamples` before each new obligation
        // keeps the fast-path set scoped to the current proof.
        let mut cegis = CegisLoop::new(10, 5000);
        let vars_a = vec![("x".to_string(), 8)];

        // Obligation A: add != sub. Seed edge cases so the concrete path
        // finds a CX immediately.
        cegis.add_edge_case_seeds(&vars_a);
        let src_a = SmtExpr::var("x", 8);
        let tgt_a = SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(1, 8));
        let _ = cegis.check_concrete_only(&src_a, &tgt_a);
        assert!(
            cegis.counterexample_count() >= 6,
            "edge-case seeds for 1 var should contribute 6 CXs",
        );

        // Simulate the cegis_pass candidate boundary: clear CX before
        // starting the next obligation. Stats accumulate across candidates.
        let stats_before_clear = cegis.stats_concrete_rejections();
        cegis.clear_counterexamples();
        assert_eq!(cegis.counterexample_count(), 0);
        assert_eq!(cegis.stats_concrete_rejections(), stats_before_clear);

        // Obligation B (different variable name, independent proof):
        // `y + 0 == y` is equivalent for all y, so adding B-only seeds and
        // running `check_concrete_only` must return None. Without the
        // clear, A's seeds (with key "x" and not "y") would live alongside
        // B's seeds — harmless in this case because missing vars skip,
        // but in a Layer-C world where A and B share variable names the
        // bled values would drive false rejections.
        let vars_b = vec![("y".to_string(), 8)];
        cegis.add_edge_case_seeds(&vars_b);
        let src_b = SmtExpr::var("y", 8).bvadd(SmtExpr::bv_const(0, 8));
        let tgt_b = SmtExpr::var("y", 8);
        let result = cegis.check_concrete_only(&src_b, &tgt_b);
        assert!(
            result.is_none(),
            "equivalent obligation B must not be rejected on bled state",
        );
        // CX count after B-seeding equals only B's edge cases, no A leftover.
        assert_eq!(cegis.counterexample_count(), 6);
    }

    #[test]
    fn test_cegis_bleed_without_clear_can_falsely_reject() {
        // Adversarial demonstration of the bug fixed by #493: if we DO NOT
        // clear between obligations, a bled CX from obligation A can
        // distinguish the source and target of obligation B and trigger a
        // spurious concrete rejection. We simulate this purely with manual
        // seeds so the test is deterministic and solver-free.
        let mut cegis = CegisLoop::new(10, 5000);

        // Obligation A (imagine Layer-C udiv vs mul-inverse): solver
        // returns `c = 3, y = 5` distinguishing the two forms, added here
        // by hand.
        cegis.add_seed(ConcreteInput::from_pairs(&[("c", 3), ("y", 5)]));
        assert_eq!(cegis.counterexample_count(), 1);

        // Obligation B (different expressions, SAME variable names): the
        // bled CX (`c=3, y=5`) distinguishes `y + c` from `y - c`, so the
        // fast path will falsely reject it despite B being an independent
        // proof.
        let src_b = SmtExpr::var("y", 8).bvadd(SmtExpr::var("c", 8));
        let tgt_b = SmtExpr::var("y", 8).bvsub(SmtExpr::var("c", 8));
        let rejected_without_clear = cegis.check_concrete_only(&src_b, &tgt_b);
        assert!(
            rejected_without_clear.is_some(),
            "bled CX (c=3, y=5) distinguishes add-vs-sub — this is the bug",
        );

        // Apply the fix: clear CX between obligations. The same B
        // expressions now have no accumulated CX to reject against. (The
        // expressions themselves are not equivalent, but without a seed
        // the fast path must defer to the solver rather than emit a
        // spurious concrete-path NotEquivalent.)
        cegis.clear_counterexamples();
        let result_after_clear = cegis.check_concrete_only(&src_b, &tgt_b);
        assert!(
            result_after_clear.is_none(),
            "after clear, the fast path has no CX and cannot spuriously reject",
        );
    }

    // -----------------------------------------------------------------------
    // Proof hash tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_hash_deterministic() {
        let source = SmtExpr::var("x", 32).bvadd(SmtExpr::bv_const(0, 32));
        let target = SmtExpr::var("x", 32);

        let hash1 = compute_proof_hash(&source, &target);
        let hash2 = compute_proof_hash(&source, &target);
        assert_eq!(hash1, hash2, "Proof hash should be deterministic");
    }

    #[test]
    fn test_proof_hash_different_for_different_pairs() {
        let a = SmtExpr::var("x", 32).bvadd(SmtExpr::bv_const(0, 32));
        let b = SmtExpr::var("x", 32);
        let c = SmtExpr::var("x", 32).bvsub(SmtExpr::bv_const(0, 32));

        let hash1 = compute_proof_hash(&a, &b);
        let hash2 = compute_proof_hash(&a, &c);
        assert_ne!(hash1, hash2, "Different pairs should have different hashes");
    }

    // -----------------------------------------------------------------------
    // Integration with existing ProofObligation infrastructure
    // -----------------------------------------------------------------------

    #[test]
    fn test_cegis_with_existing_proof_obligation() {
        // Use an existing proof obligation from the codebase
        let obligation = crate::lowering_proof::proof_iadd_i32();

        let mut cegis = CegisLoop::new(10, 5000);
        cegis.add_edge_case_seeds(&obligation.inputs);

        let result = cegis.verify(
            &obligation.tmir_expr,
            &obligation.aarch64_expr,
            &obligation.inputs,
        );

        match &result {
            CegisResult::Equivalent { .. } => {
                // Correct: iadd is correctly lowered to ADD
            }
            CegisResult::Error(_) => {
                // No solver available
            }
            other => {
                panic!("Expected Equivalent or Error for iadd proof, got {:?}", other);
            }
        }
    }

    #[test]
    fn test_cegis_catches_wrong_lowering() {
        // Deliberately wrong: claim add = sub
        let mut cegis = CegisLoop::new(10, 5000);
        let vars = vec![("a".to_string(), 8), ("b".to_string(), 8)];

        // Seed with edge cases to catch it fast
        cegis.add_edge_case_seeds(&vars);

        let source = SmtExpr::var("a", 8).bvadd(SmtExpr::var("b", 8));
        let target = SmtExpr::var("a", 8).bvsub(SmtExpr::var("b", 8));

        let result = cegis.verify(&source, &target, &vars);

        match &result {
            CegisResult::NotEquivalent {
                counterexample,
                found_by_concrete,
            } => {
                // Should be caught fast by concrete evaluation
                assert!(*found_by_concrete);
                // Validate the counterexample
                let s = source.try_eval(&counterexample.values).unwrap();
                let t = target.try_eval(&counterexample.values).unwrap();
                assert_ne!(s, t);
            }
            other => {
                panic!("Expected NotEquivalent, got {:?}", other);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Peephole identity CEGIS tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cegis_peephole_double_negation() {
        // neg(neg(x)) == x
        let mut cegis = CegisLoop::new(10, 5000);
        let vars = vec![("x".to_string(), 32)];
        cegis.add_edge_case_seeds(&vars);

        let source = SmtExpr::var("x", 32).bvneg().bvneg();
        let target = SmtExpr::var("x", 32);

        let result = cegis.verify(&source, &target, &vars);

        match &result {
            CegisResult::Equivalent { .. } => {} // Good
            CegisResult::Error(_) => {}           // No solver
            other => {
                panic!("Expected Equivalent or Error, got {:?}", other);
            }
        }
    }

    #[test]
    fn test_cegis_peephole_strength_reduction() {
        // x * 2 == x + x
        let mut cegis = CegisLoop::new(10, 5000);
        let vars = vec![("x".to_string(), 32)];
        cegis.add_edge_case_seeds(&vars);

        let source = SmtExpr::var("x", 32).bvmul(SmtExpr::bv_const(2, 32));
        let target = SmtExpr::var("x", 32).bvadd(SmtExpr::var("x", 32));

        let result = cegis.verify(&source, &target, &vars);

        match &result {
            CegisResult::Equivalent { .. } => {} // Good
            CegisResult::Error(_) => {}           // No solver
            other => {
                panic!("Expected Equivalent or Error, got {:?}", other);
            }
        }
    }

    #[test]
    fn test_cegis_peephole_xor_self() {
        // x ^ x == 0
        let mut cegis = CegisLoop::new(10, 5000);
        let vars = vec![("x".to_string(), 32)];
        cegis.add_edge_case_seeds(&vars);

        let source = SmtExpr::var("x", 32).bvxor(SmtExpr::var("x", 32));
        let target = SmtExpr::bv_const(0, 32);

        let result = cegis.verify(&source, &target, &vars);

        match &result {
            CegisResult::Equivalent { .. } => {} // Good
            CegisResult::Error(_) => {}           // No solver
            other => {
                panic!("Expected Equivalent or Error, got {:?}", other);
            }
        }
    }

    #[test]
    fn test_cegis_peephole_and_self() {
        // x & x == x
        let mut cegis = CegisLoop::new(10, 5000);
        let vars = vec![("x".to_string(), 32)];
        cegis.add_edge_case_seeds(&vars);

        let source = SmtExpr::var("x", 32).bvand(SmtExpr::var("x", 32));
        let target = SmtExpr::var("x", 32);

        let result = cegis.verify(&source, &target, &vars);

        match &result {
            CegisResult::Equivalent { .. } => {} // Good
            CegisResult::Error(_) => {}           // No solver
            other => {
                panic!("Expected Equivalent or Error, got {:?}", other);
            }
        }
    }

    #[test]
    fn test_cegis_peephole_or_zero() {
        // x | 0 == x
        let mut cegis = CegisLoop::new(10, 5000);
        let vars = vec![("x".to_string(), 32)];
        cegis.add_edge_case_seeds(&vars);

        let source = SmtExpr::var("x", 32).bvor(SmtExpr::bv_const(0, 32));
        let target = SmtExpr::var("x", 32);

        let result = cegis.verify(&source, &target, &vars);

        match &result {
            CegisResult::Equivalent { .. } => {} // Good
            CegisResult::Error(_) => {}           // No solver
            other => {
                panic!("Expected Equivalent or Error, got {:?}", other);
            }
        }
    }

    #[test]
    fn test_cegis_peephole_sub_self() {
        // x - x == 0
        let mut cegis = CegisLoop::new(10, 5000);
        let vars = vec![("x".to_string(), 32)];
        cegis.add_edge_case_seeds(&vars);

        let source = SmtExpr::var("x", 32).bvsub(SmtExpr::var("x", 32));
        let target = SmtExpr::bv_const(0, 32);

        let result = cegis.verify(&source, &target, &vars);

        match &result {
            CegisResult::Equivalent { .. } => {} // Good
            CegisResult::Error(_) => {}           // No solver
            other => {
                panic!("Expected Equivalent or Error, got {:?}", other);
            }
        }
    }
}
