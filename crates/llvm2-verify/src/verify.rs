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

use llvm2_lower::Function;
use thiserror::Error;

use crate::lowering_proof::{ProofObligation, verify_by_evaluation};

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

    /// Format a human-readable summary.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "Verification Report: {}/{} passed, {} failed, {} unknown",
            self.passed(), self.total(), self.failed(), self.unknown()
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
            lines.push(format!("  {}: {}/{} passed", cat, cat_passed, cat_total));

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
fn run_proofs(obligations: &[ProofObligation], category: &str) -> VerificationReport {
    let results = obligations.iter().map(|obligation| {
        let result = verify_by_evaluation(obligation);
        ProofResult {
            name: obligation.name.clone(),
            category: category.to_string(),
            result,
        }
    }).collect();

    VerificationReport { results }
}

/// Function verifier — unified entry point for the verification pipeline.
///
/// Runs proof obligations from all verification categories:
/// - Arithmetic lowering (add, sub, mul, neg)
/// - NZCV flags (N, Z, C, V correctness)
/// - Comparison lowering (10 conditions, 32-bit + 64-bit)
/// - Branch lowering (conditional branch semantics)
/// - Peephole optimizations (identity rules)
/// - Memory model (load/store equivalence, roundtrip, non-interference, endianness)
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

    /// Verify all array-based memory model proofs (21 obligations).
    ///
    /// Includes:
    /// - 6 load equivalence proofs (tMIR Load == AArch64 LDR)
    /// - 6 store equivalence proofs (tMIR Store == AArch64 STR)
    /// - 4 store-load roundtrip proofs (store then load returns original)
    /// - 2 non-interference proofs (store at A doesn't affect B)
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
                },
                ProofResult {
                    name: "proof_b".to_string(),
                    category: "test".to_string(),
                    result: VerificationResult::Invalid {
                        counterexample: "a=1, b=2".to_string(),
                    },
                },
                ProofResult {
                    name: "proof_c".to_string(),
                    category: "other".to_string(),
                    result: VerificationResult::Unknown {
                        reason: "timeout".to_string(),
                    },
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
    }

    #[test]
    fn test_verification_report_merge() {
        let mut r1 = VerificationReport {
            results: vec![ProofResult {
                name: "a".to_string(),
                category: "cat1".to_string(),
                result: VerificationResult::Valid,
            }],
        };
        let r2 = VerificationReport {
            results: vec![ProofResult {
                name: "b".to_string(),
                category: "cat2".to_string(),
                result: VerificationResult::Valid,
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
                },
                ProofResult {
                    name: "proof_b".to_string(),
                    category: "memory".to_string(),
                    result: VerificationResult::Valid,
                },
            ],
        };
        let summary = report.summary();
        assert!(summary.contains("2/2 passed"));
        assert!(summary.contains("arithmetic: 1/1 passed"));
        assert!(summary.contains("memory: 1/1 passed"));
    }

    #[test]
    fn test_verifier_arithmetic() {
        let verifier = Verifier::new();
        let report = verifier.verify_arithmetic();
        assert_eq!(report.total(), 5);
        assert!(report.all_valid(), "Arithmetic proofs failed:\n{}", report.summary());
    }

    #[test]
    fn test_verifier_memory_model() {
        let verifier = Verifier::new();
        let report = verifier.verify_memory_model();
        assert_eq!(report.total(), 21);
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
}
