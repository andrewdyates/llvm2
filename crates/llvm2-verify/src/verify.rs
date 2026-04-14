// llvm2-verify/verify.rs - Verification interface
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! High-level verification interface.
//!
//! The [`Verifier`] provides a unified entry point for running all proof
//! obligations in the LLVM2 verification pipeline. Individual proof categories
//! (arithmetic lowering, NZCV flags, comparisons, branches, peephole
//! optimizations, memory model) can be run independently or together via
//! [`Verifier::verify_comprehensive`].
//!
//! Results are collected into a [`VerificationReport`] that tracks per-proof
//! pass/fail status and provides summary statistics.
//!
//! # Verification strength levels
//!
//! The verification system supports three strength levels, described by
//! [`VerificationStrength`]:
//!
//! | Level | Implementation | Guarantee | When used |
//! |-------|---------------|-----------|-----------|
//! | [`VerificationStrength::Exhaustive`] | Concrete evaluation of all inputs | **Complete** for that bit-width | Widths <= 8, inputs <= 2 |
//! | [`VerificationStrength::Statistical`] | Edge cases + N random samples | **Probabilistic** (configurable N, default 100K) | Widths > 8 (32-bit, 64-bit) |
//! | [`VerificationStrength::Formal`] | SMT solver (z4/z3) | **Complete** for all widths | Via [`crate::z4_bridge`], not yet default |
//!
//! ## Current status
//!
//! All proofs currently run at **Exhaustive** or **Statistical** strength
//! depending on bit-width. The 32/64-bit proofs use statistical verification
//! with 100,000 random samples (configurable via [`crate::lowering_proof::VerificationConfig`]).
//! This provides high confidence but is not a formal proof.
//!
//! ## Path to formal verification
//!
//! 1. **Current** (mock evaluation): Fast, catches regressions and most bugs.
//!    Exhaustive for 8-bit, statistical for 32/64-bit.
//! 2. **Available** (z4 CLI): Proof obligations can be serialized to SMT-LIB2
//!    via [`crate::lowering_proof::ProofObligation::to_smt2`] and verified with
//!    an external z3/z4 solver. See [`crate::z4_bridge::verify_with_z4`].
//! 3. **Future** (z4 native): The `z4` feature gate enables in-process SMT
//!    solving with no subprocess overhead. When this becomes the default,
//!    mock evaluation will serve as a fast pre-check.

use llvm2_lower::Function;
use thiserror::Error;

use crate::lowering_proof::{ProofObligation, verify_by_evaluation, VerificationConfig};

/// Describes the strength of verification applied to a proof obligation.
///
/// This enum classifies the guarantee level of a verification result.
/// It is informational -- it tells the caller what kind of verification
/// was actually performed, so they can assess confidence appropriately.
///
/// # Strength levels
///
/// - **Exhaustive**: Every possible input was tested. This is equivalent to
///   a formal proof for the tested bit-width, but does not extend to wider
///   types. For example, exhaustive 8-bit verification proves correctness
///   for all 256 (or 65,536 for two inputs) values, but says nothing about
///   32-bit behavior.
///
/// - **Statistical**: A combination of edge cases and random samples was tested.
///   The default configuration tests 36 edge-case combinations plus 100,000
///   random samples. This provides high confidence but is not a formal proof.
///   Structured or adversarial bugs could theoretically hide in the untested
///   input space.
///
/// - **Formal**: An SMT solver (z4/z3) proved the property for ALL inputs
///   of the given bit-width. This is a complete mathematical proof. Currently
///   available via [`crate::z4_bridge::verify_with_z4`] but not yet the
///   default verification mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationStrength {
    /// Complete enumeration of all inputs. Proof is exhaustive for the
    /// tested bit-width (<= 8-bit with <= 2 inputs by default).
    Exhaustive,

    /// Edge cases plus N random samples. High confidence but not a formal
    /// proof. Used for 32/64-bit proofs. The sample count is configurable
    /// via [`VerificationConfig`].
    ///
    /// The `sample_count` field records how many random trials were used.
    Statistical {
        /// Number of random samples that were tested (excludes edge cases).
        sample_count: u64,
    },

    /// SMT solver provided a complete formal proof for all inputs.
    /// Available via z4/z3 CLI or native z4 API (feature-gated).
    Formal,
}

impl VerificationStrength {
    /// Determine the verification strength that would be used for a proof
    /// obligation with the given parameters and default configuration.
    pub fn for_obligation(obligation: &ProofObligation) -> Self {
        Self::for_obligation_with_config(obligation, &VerificationConfig::default())
    }

    /// Determine the verification strength for a proof obligation with
    /// a custom configuration.
    pub fn for_obligation_with_config(
        obligation: &ProofObligation,
        config: &VerificationConfig,
    ) -> Self {
        let width = obligation.inputs.first().map(|(_, w)| *w).unwrap_or(32);
        let num_inputs = obligation.inputs.len();

        if num_inputs <= 2 && width <= config.exhaustive_threshold {
            VerificationStrength::Exhaustive
        } else {
            VerificationStrength::Statistical {
                sample_count: config.sample_count,
            }
        }
    }

    /// Returns true if this is a complete (exhaustive or formal) verification.
    pub fn is_complete(&self) -> bool {
        matches!(self, VerificationStrength::Exhaustive | VerificationStrength::Formal)
    }

    /// Returns a human-readable description of the verification strength.
    pub fn description(&self) -> String {
        match self {
            VerificationStrength::Exhaustive => {
                "Exhaustive: all input combinations tested".to_string()
            }
            VerificationStrength::Statistical { sample_count } => {
                format!(
                    "Statistical: edge cases + {} random samples (not a formal proof)",
                    sample_count
                )
            }
            VerificationStrength::Formal => {
                "Formal: SMT solver proved correctness for all inputs".to_string()
            }
        }
    }
}

impl std::fmt::Display for VerificationStrength {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerificationStrength::Exhaustive => write!(f, "Exhaustive"),
            VerificationStrength::Statistical { sample_count } => {
                write!(f, "Statistical({})", sample_count)
            }
            VerificationStrength::Formal => write!(f, "Formal"),
        }
    }
}

/// Verification result for a single proof obligation.
#[derive(Debug, Clone)]
pub enum VerificationResult {
    /// Verification succeeded - property holds for all inputs.
    Valid,
    /// Verification failed - counterexample found.
    Invalid { counterexample: String },
    /// Verification inconclusive (timeout, unknown, z4 not available).
    Unknown { reason: String },
}

/// Result for a single named proof obligation.
#[derive(Debug, Clone)]
pub struct ProofResult {
    /// Name of the proof obligation.
    pub name: String,
    /// Category this proof belongs to (e.g., "memory", "arithmetic", "peephole").
    pub category: String,
    /// The verification outcome.
    pub result: VerificationResult,
    /// The strength of verification that was applied.
    /// See [`VerificationStrength`] for what each level guarantees.
    pub strength: VerificationStrength,
}

impl ProofResult {
    /// Returns true if the proof was verified as valid.
    pub fn is_valid(&self) -> bool {
        matches!(self.result, VerificationResult::Valid)
    }

    /// Returns true if the proof found a counterexample.
    pub fn is_invalid(&self) -> bool {
        matches!(self.result, VerificationResult::Invalid { .. })
    }
}

/// Aggregated verification report across multiple proof categories.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// All individual proof results.
    pub results: Vec<ProofResult>,
}

impl VerificationReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self { results: Vec::new() }
    }

    /// Total number of proofs checked.
    pub fn total(&self) -> usize {
        self.results.len()
    }

    /// Number of proofs that passed (Valid).
    pub fn passed(&self) -> usize {
        self.results.iter().filter(|r| r.is_valid()).count()
    }

    /// Number of proofs that failed (Invalid).
    pub fn failed(&self) -> usize {
        self.results.iter().filter(|r| r.is_invalid()).count()
    }

    /// Number of proofs that were inconclusive (Unknown).
    pub fn unknown(&self) -> usize {
        self.results.iter().filter(|r| matches!(r.result, VerificationResult::Unknown { .. })).count()
    }

    /// Returns true if all proofs passed.
    pub fn all_valid(&self) -> bool {
        self.results.iter().all(|r| r.is_valid())
    }

    /// Returns only the failed proof results.
    pub fn failures(&self) -> Vec<&ProofResult> {
        self.results.iter().filter(|r| r.is_invalid()).collect()
    }

    /// Returns results for a specific category.
    pub fn by_category(&self, category: &str) -> Vec<&ProofResult> {
        self.results.iter().filter(|r| r.category == category).collect()
    }

    /// Merge another report into this one.
    pub fn merge(&mut self, other: VerificationReport) {
        self.results.extend(other.results);
    }

    /// Number of proofs verified with exhaustive (complete) strength.
    pub fn exhaustive_count(&self) -> usize {
        self.results.iter().filter(|r| matches!(r.strength, VerificationStrength::Exhaustive)).count()
    }

    /// Number of proofs verified with statistical (sampling) strength.
    pub fn statistical_count(&self) -> usize {
        self.results.iter().filter(|r| matches!(r.strength, VerificationStrength::Statistical { .. })).count()
    }

    /// Number of proofs verified with formal (SMT solver) strength.
    pub fn formal_count(&self) -> usize {
        self.results.iter().filter(|r| matches!(r.strength, VerificationStrength::Formal)).count()
    }

    /// Format a human-readable summary.
    ///
    /// The summary includes per-category pass/fail counts and a breakdown
    /// of verification strength levels so the reader understands the
    /// confidence level of each proof.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "Verification Report: {}/{} passed, {} failed, {} unknown",
            self.passed(), self.total(), self.failed(), self.unknown()
        ));
        lines.push(format!(
            "  Strength: {} exhaustive, {} statistical, {} formal",
            self.exhaustive_count(), self.statistical_count(), self.formal_count()
        ));

        // Group by category
        let mut categories: Vec<String> = self.results.iter()
            .map(|r| r.category.clone())
            .collect();
        categories.sort();
        categories.dedup();

        for cat in &categories {
            let cat_results = self.by_category(cat);
            let cat_passed = cat_results.iter().filter(|r| r.is_valid()).count();
            let cat_total = cat_results.len();
            let cat_exhaustive = cat_results.iter()
                .filter(|r| matches!(r.strength, VerificationStrength::Exhaustive))
                .count();
            let cat_statistical = cat_results.iter()
                .filter(|r| matches!(r.strength, VerificationStrength::Statistical { .. }))
                .count();
            lines.push(format!(
                "  {}: {}/{} passed ({} exhaustive, {} statistical)",
                cat, cat_passed, cat_total, cat_exhaustive, cat_statistical
            ));

            // List failures
            for r in &cat_results {
                if r.is_invalid() {
                    if let VerificationResult::Invalid { ref counterexample } = r.result {
                        lines.push(format!("    FAIL: {} — {}", r.name, counterexample));
                    }
                }
            }
        }

        lines.join("\n")
    }
}

impl Default for VerificationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Verification error.
#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("encoding error: {0}")]
    Encoding(String),
    #[error("solver error: {0}")]
    Solver(String),
}

/// Run a set of proof obligations and collect results into a report.
///
/// Each proof is verified using mock evaluation ([`verify_by_evaluation`])
/// with default configuration. The verification strength (exhaustive vs
/// statistical) is determined by the proof obligation's bit-width and
/// input count. See [`VerificationStrength`] for details.
fn run_proofs(obligations: &[ProofObligation], category: &str) -> VerificationReport {
    let config = VerificationConfig::default();
    let results = obligations.iter().map(|obligation| {
        let strength = VerificationStrength::for_obligation_with_config(obligation, &config);
        let result = verify_by_evaluation(obligation);
        ProofResult {
            name: obligation.name.clone(),
            category: category.to_string(),
            result,
            strength,
        }
    }).collect();

    VerificationReport { results }
}

/// Function verifier -- unified entry point for the verification pipeline.
///
/// Runs proof obligations from all verification categories:
/// - Arithmetic lowering (add, sub, mul, neg)
/// - NZCV flags (N, Z, C, V correctness)
/// - Comparison lowering (10 conditions, 32-bit + 64-bit)
/// - Branch lowering (conditional branch semantics)
/// - Peephole optimizations (identity rules)
/// - Memory model (load/store equivalence, roundtrip, non-interference, endianness)
/// - Optimization proofs (constant folding, CSE/LICM, DCE, CFG, copy propagation)
///
/// # Verification strength
///
/// All proofs are currently run using **mock evaluation** via
/// [`crate::lowering_proof::verify_by_evaluation`]. This means:
///
/// - **8-bit proofs** (with <= 2 inputs): Exhaustive -- every input combination
///   is tested. Equivalent to a formal proof for 8-bit semantics.
/// - **32/64-bit proofs**: Statistical -- edge cases (0, 1, MAX, etc.) plus
///   100,000 random samples. High confidence but **not a formal proof**.
///
/// To get formal guarantees for 32/64-bit proofs, use
/// [`crate::z4_bridge::verify_with_z4`] on individual proof obligations.
///
/// The sample count for statistical verification is configurable via
/// [`VerificationConfig`] and [`crate::lowering_proof::verify_by_evaluation_with_config`].
pub struct Verifier {
    timeout_ms: u64,
}

impl Verifier {
    /// Create a new verifier with default settings.
    pub fn new() -> Self {
        Self { timeout_ms: 30000 }
    }

    /// Set solver timeout in milliseconds.
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Get the configured timeout.
    pub fn timeout_ms(&self) -> u64 {
        self.timeout_ms
    }

    /// Verify that a transformation is semantics-preserving.
    ///
    /// This is a placeholder for whole-function verification.
    /// For per-rule verification, use [`crate::lowering_proof`] directly.
    pub fn verify_transformation(
        &self,
        _original: &Function,
        _transformed: &Function,
    ) -> Result<VerificationResult, VerifyError> {
        // TODO: Implement whole-function verification using z4.
        // For now, per-rule verification is in lowering_proof.rs.
        Ok(VerificationResult::Unknown {
            reason: "whole-function verification not yet implemented; use lowering_proof::verify_by_evaluation for per-rule proofs".to_string(),
        })
    }

    // -------------------------------------------------------------------
    // Per-category verification methods
    // -------------------------------------------------------------------

    /// Verify all arithmetic lowering proofs (add, sub, mul, neg).
    ///
    /// Returns a report with 5 proof results.
    pub fn verify_arithmetic(&self) -> VerificationReport {
        use crate::lowering_proof::all_arithmetic_proofs;
        run_proofs(&all_arithmetic_proofs(), "arithmetic")
    }

    /// Verify all NZCV flag correctness proofs and comparison/branch lowering.
    ///
    /// Includes: 4 flag proofs, 10 comparison proofs (32-bit), 3 comparison
    /// proofs (64-bit), 4 branch proofs = 21 total.
    pub fn verify_nzcv(&self) -> VerificationReport {
        use crate::lowering_proof::all_nzcv_proofs;
        run_proofs(&all_nzcv_proofs(), "nzcv")
    }

    /// Verify all peephole optimization identity proofs.
    ///
    /// Returns a report with 11+ proof results (identity rules for
    /// add-zero, sub-zero, mul-one, shifts, OR/AND/XOR, etc.).
    pub fn verify_peephole(&self) -> VerificationReport {
        use crate::peephole_proofs::all_peephole_proofs;
        run_proofs(&all_peephole_proofs(), "peephole")
    }

    /// Verify all array-based memory model proofs (27 obligations).
    ///
    /// Includes:
    /// - 6 load equivalence proofs (tMIR Load == AArch64 LDR)
    /// - 6 store equivalence proofs (tMIR Store == AArch64 STR)
    /// - 4 store-load roundtrip proofs (store then load returns original)
    /// - 8 non-interference proofs (store at A doesn't affect B, with overlap guards)
    /// - 3 endianness proofs (little-endian byte ordering)
    pub fn verify_memory_model(&self) -> VerificationReport {
        use crate::memory_proofs::all_memory_proofs;
        run_proofs(&all_memory_proofs(), "memory")
    }

    /// Verify all optimization proofs (constant folding, CSE/LICM, DCE, etc.).
    pub fn verify_optimizations(&self) -> VerificationReport {
        let mut report = VerificationReport::new();

        // Constant folding proofs
        {
            use crate::const_fold_proofs::all_const_fold_proofs;
            report.merge(run_proofs(&all_const_fold_proofs(), "const_fold"));
        }

        // Copy propagation proofs
        {
            use crate::copy_prop_proofs::all_copy_prop_proofs;
            report.merge(run_proofs(&all_copy_prop_proofs(), "copy_prop"));
        }

        // CSE/LICM proofs
        {
            use crate::cse_licm_proofs::all_cse_licm_proofs;
            report.merge(run_proofs(&all_cse_licm_proofs(), "cse_licm"));
        }

        // DCE proofs
        {
            use crate::dce_proofs::all_dce_proofs;
            report.merge(run_proofs(&all_dce_proofs(), "dce"));
        }

        // CFG proofs
        {
            use crate::cfg_proofs::all_cfg_proofs;
            report.merge(run_proofs(&all_cfg_proofs(), "cfg"));
        }

        // General opt proofs
        {
            use crate::opt_proofs::all_opt_proofs;
            report.merge(run_proofs(&all_opt_proofs(), "opt"));
        }

        report
    }

    /// Run all proof obligations across the entire verification pipeline.
    ///
    /// This is the comprehensive verification entry point. It runs:
    /// 1. Arithmetic lowering proofs
    /// 2. NZCV flag + comparison + branch proofs
    /// 3. Peephole optimization proofs
    /// 4. Memory model proofs (load/store equivalence, roundtrip, endianness)
    /// 5. Optimization proofs (const fold, copy prop, CSE/LICM, DCE, CFG)
    ///
    /// Returns a [`VerificationReport`] with per-proof pass/fail results
    /// and summary statistics.
    pub fn verify_comprehensive(&self) -> VerificationReport {
        let mut report = VerificationReport::new();
        report.merge(self.verify_arithmetic());
        report.merge(self.verify_nzcv());
        report.merge(self.verify_peephole());
        report.merge(self.verify_memory_model());
        report.merge(self.verify_optimizations());
        report
    }
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_report_empty() {
        let report = VerificationReport::new();
        assert_eq!(report.total(), 0);
        assert_eq!(report.passed(), 0);
        assert_eq!(report.failed(), 0);
        assert!(report.all_valid());
    }

    #[test]
    fn test_verification_report_mixed() {
        let report = VerificationReport {
            results: vec![
                ProofResult {
                    name: "proof_a".to_string(),
                    category: "test".to_string(),
                    result: VerificationResult::Valid,
                    strength: VerificationStrength::Exhaustive,
                },
                ProofResult {
                    name: "proof_b".to_string(),
                    category: "test".to_string(),
                    result: VerificationResult::Invalid {
                        counterexample: "a=1, b=2".to_string(),
                    },
                    strength: VerificationStrength::Statistical { sample_count: 100_000 },
                },
                ProofResult {
                    name: "proof_c".to_string(),
                    category: "other".to_string(),
                    result: VerificationResult::Unknown {
                        reason: "timeout".to_string(),
                    },
                    strength: VerificationStrength::Formal,
                },
            ],
        };
        assert_eq!(report.total(), 3);
        assert_eq!(report.passed(), 1);
        assert_eq!(report.failed(), 1);
        assert_eq!(report.unknown(), 1);
        assert!(!report.all_valid());
        assert_eq!(report.failures().len(), 1);
        assert_eq!(report.by_category("test").len(), 2);
        assert_eq!(report.by_category("other").len(), 1);
        assert_eq!(report.exhaustive_count(), 1);
        assert_eq!(report.statistical_count(), 1);
        assert_eq!(report.formal_count(), 1);
    }

    #[test]
    fn test_verification_report_merge() {
        let mut r1 = VerificationReport {
            results: vec![ProofResult {
                name: "a".to_string(),
                category: "cat1".to_string(),
                result: VerificationResult::Valid,
                strength: VerificationStrength::Exhaustive,
            }],
        };
        let r2 = VerificationReport {
            results: vec![ProofResult {
                name: "b".to_string(),
                category: "cat2".to_string(),
                result: VerificationResult::Valid,
                strength: VerificationStrength::Statistical { sample_count: 100_000 },
            }],
        };
        r1.merge(r2);
        assert_eq!(r1.total(), 2);
        assert!(r1.all_valid());
    }

    #[test]
    fn test_verification_report_summary_format() {
        let report = VerificationReport {
            results: vec![
                ProofResult {
                    name: "proof_a".to_string(),
                    category: "arithmetic".to_string(),
                    result: VerificationResult::Valid,
                    strength: VerificationStrength::Exhaustive,
                },
                ProofResult {
                    name: "proof_b".to_string(),
                    category: "memory".to_string(),
                    result: VerificationResult::Valid,
                    strength: VerificationStrength::Statistical { sample_count: 100_000 },
                },
            ],
        };
        let summary = report.summary();
        assert!(summary.contains("2/2 passed"));
        assert!(summary.contains("1 exhaustive, 1 statistical"));
        assert!(summary.contains("arithmetic: 1/1 passed"));
        assert!(summary.contains("memory: 1/1 passed"));
    }

    #[test]
    fn test_verifier_arithmetic() {
        let verifier = Verifier::new();
        let report = verifier.verify_arithmetic();
        assert_eq!(report.total(), 9);
        assert!(report.all_valid(), "Arithmetic proofs failed:\n{}", report.summary());
    }

    #[test]
    fn test_verifier_memory_model() {
        let verifier = Verifier::new();
        let report = verifier.verify_memory_model();
        assert_eq!(report.total(), 27);
        assert!(report.all_valid(), "Memory proofs failed:\n{}", report.summary());
    }

    #[test]
    fn test_verifier_nzcv() {
        let verifier = Verifier::new();
        let report = verifier.verify_nzcv();
        // 4 flag proofs + 10 comparison (32-bit) + 3 comparison (64-bit) + 4 branch = 21
        assert_eq!(report.total(), 21);
        assert!(report.all_valid(), "NZCV proofs failed:\n{}", report.summary());
    }

    #[test]
    fn test_verifier_peephole() {
        let verifier = Verifier::new();
        let report = verifier.verify_peephole();
        assert!(report.total() > 0);
        assert!(report.all_valid(), "Peephole proofs failed:\n{}", report.summary());
    }

    // -----------------------------------------------------------------------
    // VerificationStrength tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_verification_strength_for_8bit_obligation() {
        use crate::lowering_proof::ProofObligation;
        use crate::smt::SmtExpr;

        let a = SmtExpr::var("a", 8);
        let b = SmtExpr::var("b", 8);
        let obligation = ProofObligation {
            name: "test_8bit".to_string(),
            tmir_expr: a.clone().bvadd(b.clone()),
            aarch64_expr: a.bvadd(b),
            inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let strength = VerificationStrength::for_obligation(&obligation);
        assert_eq!(strength, VerificationStrength::Exhaustive);
        assert!(strength.is_complete());
        assert!(strength.description().contains("Exhaustive"));
        assert_eq!(format!("{}", strength), "Exhaustive");
    }

    #[test]
    fn test_verification_strength_for_32bit_obligation() {
        use crate::lowering_proof::ProofObligation;
        use crate::smt::SmtExpr;

        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        let obligation = ProofObligation {
            name: "test_32bit".to_string(),
            tmir_expr: a.clone().bvadd(b.clone()),
            aarch64_expr: a.bvadd(b),
            inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let strength = VerificationStrength::for_obligation(&obligation);
        assert_eq!(strength, VerificationStrength::Statistical { sample_count: 100_000 });
        assert!(!strength.is_complete());
        assert!(strength.description().contains("Statistical"));
        assert!(strength.description().contains("100000"));
        assert_eq!(format!("{}", strength), "Statistical(100000)");
    }

    #[test]
    fn test_verification_strength_for_64bit_obligation() {
        use crate::lowering_proof::ProofObligation;
        use crate::smt::SmtExpr;

        let a = SmtExpr::var("a", 64);
        let obligation = ProofObligation {
            name: "test_64bit".to_string(),
            tmir_expr: a.clone(),
            aarch64_expr: a,
            inputs: vec![("a".to_string(), 64)],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let strength = VerificationStrength::for_obligation(&obligation);
        assert_eq!(strength, VerificationStrength::Statistical { sample_count: 100_000 });
        assert!(!strength.is_complete());
    }

    #[test]
    fn test_verification_strength_with_custom_config() {
        use crate::lowering_proof::ProofObligation;
        use crate::smt::SmtExpr;

        let a = SmtExpr::var("a", 32);
        let obligation = ProofObligation {
            name: "test_custom".to_string(),
            tmir_expr: a.clone(),
            aarch64_expr: a,
            inputs: vec![("a".to_string(), 32)],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let config = VerificationConfig::with_sample_count(500_000);
        let strength = VerificationStrength::for_obligation_with_config(&obligation, &config);
        assert_eq!(strength, VerificationStrength::Statistical { sample_count: 500_000 });
    }

    #[test]
    fn test_verification_strength_formal() {
        let strength = VerificationStrength::Formal;
        assert!(strength.is_complete());
        assert!(strength.description().contains("Formal"));
        assert!(strength.description().contains("SMT"));
        assert_eq!(format!("{}", strength), "Formal");
    }

    #[test]
    fn test_verification_strength_in_report() {
        // Verify that the Verifier populates strength fields correctly
        let verifier = Verifier::new();
        let report = verifier.verify_arithmetic();

        // Arithmetic proofs include both 32-bit and 64-bit obligations,
        // so we should see statistical strength for those.
        assert!(report.statistical_count() > 0,
            "Arithmetic proofs should include statistical (32/64-bit) verifications");

        // The report summary should mention strength breakdown
        let summary = report.summary();
        assert!(summary.contains("exhaustive") || summary.contains("statistical"),
            "Summary should mention verification strength");
    }
}
