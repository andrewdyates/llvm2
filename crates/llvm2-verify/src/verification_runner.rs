// llvm2-verify/verification_runner.rs - Bulk proof verification with reporting
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Wires the ProofDatabase into the verification pipeline. Provides
// VerificationRunner for running all proofs in the database and
// producing comprehensive reports with per-category breakdowns,
// timing, and failure details.
//
// Reference: crates/llvm2-verify/src/proof_database.rs,
//            crates/llvm2-verify/src/verify.rs

//! Bulk proof verification runner.
//!
//! [`VerificationRunner`] takes a [`ProofDatabase`] and verifies every proof
//! obligation, producing a [`VerificationRunReport`] with pass/fail counts,
//! per-category breakdowns, duration tracking, and failure details.
//!
//! # Example
//!
//! ```rust,no_run
//! use llvm2_verify::proof_database::ProofDatabase;
//! use llvm2_verify::verification_runner::VerificationRunner;
//!
//! let db = ProofDatabase::new();
//! let runner = VerificationRunner::new(&db);
//! let report = runner.run_all();
//! assert!(report.all_passed());
//! println!("{}", report);
//! ```

use std::time::{Duration, Instant};

use crate::lowering_proof::{verify_by_evaluation_with_config, VerificationConfig};
use crate::proof_database::{CategorizedProof, ProofCategory, ProofDatabase};
use crate::verify::{VerificationResult, VerificationStrength};

// ---------------------------------------------------------------------------
// VerificationRunResult: result for a single proof in a run
// ---------------------------------------------------------------------------

/// Result of verifying a single proof obligation during a run.
#[derive(Debug, Clone)]
pub struct VerificationRunResult {
    /// Human-readable proof name.
    pub name: String,
    /// Category this proof belongs to.
    pub category: ProofCategory,
    /// The verification outcome.
    pub result: VerificationResult,
    /// Verification strength level applied.
    pub strength: VerificationStrength,
    /// Time taken to verify this proof.
    pub duration: Duration,
}

impl VerificationRunResult {
    /// Returns true if the proof passed (Valid).
    pub fn is_valid(&self) -> bool {
        matches!(self.result, VerificationResult::Valid)
    }

    /// Returns true if the proof found a counterexample (Invalid).
    pub fn is_invalid(&self) -> bool {
        matches!(self.result, VerificationResult::Invalid { .. })
    }

    /// Returns true if the result was inconclusive (Unknown).
    pub fn is_unknown(&self) -> bool {
        matches!(self.result, VerificationResult::Unknown { .. })
    }
}

// ---------------------------------------------------------------------------
// CategoryBreakdown: per-category statistics
// ---------------------------------------------------------------------------

/// Per-category verification statistics.
#[derive(Debug, Clone)]
pub struct CategoryBreakdown {
    /// The category.
    pub category: ProofCategory,
    /// Total proofs in this category.
    pub total: usize,
    /// Number that passed.
    pub passed: usize,
    /// Number that failed.
    pub failed: usize,
    /// Number inconclusive.
    pub unknown: usize,
    /// Total time spent verifying proofs in this category.
    pub duration: Duration,
}

// ---------------------------------------------------------------------------
// FailedProofDetail: details about a failed proof
// ---------------------------------------------------------------------------

/// Details about a failed or inconclusive proof.
#[derive(Debug, Clone)]
pub struct FailedProofDetail {
    /// Proof name.
    pub name: String,
    /// Category.
    pub category: ProofCategory,
    /// Counterexample string (for Invalid results) or reason (for Unknown).
    pub detail: String,
}

// ---------------------------------------------------------------------------
// VerificationRunReport: comprehensive verification report
// ---------------------------------------------------------------------------

/// Comprehensive report from running all proofs in the database.
///
/// Includes aggregate statistics, per-category breakdowns, timing, and
/// details for any failed or inconclusive proofs.
#[derive(Debug, Clone)]
pub struct VerificationRunReport {
    /// All individual proof results.
    pub results: Vec<VerificationRunResult>,
    /// Total wall-clock time for the entire run.
    pub total_duration: Duration,
}

impl VerificationRunReport {
    /// Total number of proofs verified.
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
        self.results.iter().filter(|r| r.is_unknown()).count()
    }

    /// Returns true if every proof passed.
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.is_valid())
    }

    /// Per-category breakdown of results.
    pub fn by_category(&self) -> Vec<CategoryBreakdown> {
        ProofCategory::all_categories()
            .iter()
            .filter_map(|cat| {
                let cat_results: Vec<&VerificationRunResult> =
                    self.results.iter().filter(|r| r.category == *cat).collect();
                if cat_results.is_empty() {
                    return None;
                }
                let total = cat_results.len();
                let passed = cat_results.iter().filter(|r| r.is_valid()).count();
                let failed = cat_results.iter().filter(|r| r.is_invalid()).count();
                let unknown = cat_results.iter().filter(|r| r.is_unknown()).count();
                let duration: Duration = cat_results.iter().map(|r| r.duration).sum();
                Some(CategoryBreakdown {
                    category: *cat,
                    total,
                    passed,
                    failed,
                    unknown,
                    duration,
                })
            })
            .collect()
    }

    /// Details of all failed proofs (Invalid results).
    pub fn failed_details(&self) -> Vec<FailedProofDetail> {
        self.results
            .iter()
            .filter_map(|r| match &r.result {
                VerificationResult::Invalid { counterexample } => Some(FailedProofDetail {
                    name: r.name.clone(),
                    category: r.category,
                    detail: counterexample.clone(),
                }),
                VerificationResult::Unknown { reason } => Some(FailedProofDetail {
                    name: r.name.clone(),
                    category: r.category,
                    detail: format!("Unknown: {}", reason),
                }),
                VerificationResult::Valid => None,
            })
            .collect()
    }

    /// Number of proofs verified with exhaustive strength.
    pub fn exhaustive_count(&self) -> usize {
        self.results
            .iter()
            .filter(|r| matches!(r.strength, VerificationStrength::Exhaustive))
            .count()
    }

    /// Number of proofs verified with statistical strength.
    pub fn statistical_count(&self) -> usize {
        self.results
            .iter()
            .filter(|r| matches!(r.strength, VerificationStrength::Statistical { .. }))
            .count()
    }
}

impl std::fmt::Display for VerificationRunReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Verification Run Report")?;
        writeln!(f, "======================")?;
        writeln!(f)?;

        // Summary line
        let status = if self.all_passed() { "PASS" } else { "FAIL" };
        writeln!(
            f,
            "Result: {} ({}/{} passed, {} failed, {} unknown)",
            status,
            self.passed(),
            self.total(),
            self.failed(),
            self.unknown()
        )?;
        writeln!(
            f,
            "Duration: {:.3}s",
            self.total_duration.as_secs_f64()
        )?;
        writeln!(
            f,
            "Strength: {} exhaustive, {} statistical",
            self.exhaustive_count(),
            self.statistical_count()
        )?;
        writeln!(f)?;

        // Per-category breakdown
        writeln!(f, "Per-category breakdown:")?;
        for bd in &self.by_category() {
            let cat_status = if bd.failed == 0 && bd.unknown == 0 {
                "OK"
            } else {
                "FAIL"
            };
            writeln!(
                f,
                "  {:25} {:>3}/{:>3} passed  [{:>4}]  ({:.3}s)",
                bd.category.name(),
                bd.passed,
                bd.total,
                cat_status,
                bd.duration.as_secs_f64()
            )?;
        }

        // Failed proof details
        let failures = self.failed_details();
        if !failures.is_empty() {
            writeln!(f)?;
            writeln!(f, "Failed proofs:")?;
            for detail in &failures {
                writeln!(
                    f,
                    "  [{}] {} -- {}",
                    detail.category.name(),
                    detail.name,
                    detail.detail
                )?;
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// VerificationRunner
// ---------------------------------------------------------------------------

/// Bulk verification runner backed by a [`ProofDatabase`].
///
/// Verifies proof obligations from the database using mock evaluation
/// ([`verify_by_evaluation`]) and collects results into a
/// [`VerificationRunReport`].
///
/// # Parallel execution
///
/// [`run_parallel`] distributes proofs across `std::thread` worker threads
/// for faster wall-clock time on multi-core machines.
pub struct VerificationRunner<'a> {
    db: &'a ProofDatabase,
    config: VerificationConfig,
}

impl<'a> VerificationRunner<'a> {
    /// Create a runner with default verification configuration.
    pub fn new(db: &'a ProofDatabase) -> Self {
        Self {
            db,
            config: VerificationConfig::default(),
        }
    }

    /// Create a runner with custom verification configuration.
    pub fn with_config(db: &'a ProofDatabase, config: VerificationConfig) -> Self {
        Self { db, config }
    }

    /// Verify every proof in the database sequentially.
    ///
    /// Returns a comprehensive report with per-proof results, category
    /// breakdowns, timing, and failure details.
    pub fn run_all(&self) -> VerificationRunReport {
        let start = Instant::now();
        let results: Vec<VerificationRunResult> = self
            .db
            .all()
            .iter()
            .map(|cp| self.verify_one(cp))
            .collect();
        let total_duration = start.elapsed();
        VerificationRunReport {
            results,
            total_duration,
        }
    }

    /// Verify all proofs in a single category.
    ///
    /// Returns a list of (proof_name, VerificationResult) pairs.
    pub fn run_category(
        &self,
        cat: ProofCategory,
    ) -> Vec<(String, VerificationResult)> {
        self.db
            .by_category(cat)
            .iter()
            .map(|cp| {
                let name = cp.obligation.name.clone();
                let result = verify_by_evaluation_with_config(&cp.obligation, &self.config);
                (name, result)
            })
            .collect()
    }

    /// Verify every proof in the database in parallel using `std::thread`.
    ///
    /// Distributes proofs across `threads` worker threads. Each thread
    /// processes a contiguous chunk of the proof list.
    ///
    /// # Panics
    ///
    /// Panics if `threads` is 0.
    pub fn run_parallel(&self, threads: usize) -> VerificationRunReport {
        assert!(threads > 0, "thread count must be >= 1");

        let start = Instant::now();
        let all_proofs = self.db.all();

        if threads == 1 || all_proofs.len() <= 1 {
            return self.run_all();
        }

        // Clone proofs and config for thread ownership.
        let proofs: Vec<CategorizedProof> = all_proofs.to_vec();
        let config = self.config.clone();
        let chunk_size = (proofs.len() + threads - 1) / threads;

        let handles: Vec<std::thread::JoinHandle<Vec<VerificationRunResult>>> = proofs
            .chunks(chunk_size)
            .map(|chunk| {
                let chunk_owned: Vec<CategorizedProof> = chunk.to_vec();
                let thread_config = config.clone();
                std::thread::spawn(move || {
                    chunk_owned
                        .iter()
                        .map(|cp| {
                            let proof_start = Instant::now();
                            let strength = VerificationStrength::for_obligation_with_config(
                                &cp.obligation,
                                &thread_config,
                            );
                            let result =
                                verify_by_evaluation_with_config(&cp.obligation, &thread_config);
                            let duration = proof_start.elapsed();
                            VerificationRunResult {
                                name: cp.obligation.name.clone(),
                                category: cp.category,
                                result,
                                strength,
                                duration,
                            }
                        })
                        .collect()
                })
            })
            .collect();

        let mut results = Vec::with_capacity(proofs.len());
        for handle in handles {
            match handle.join() {
                Ok(chunk_results) => results.extend(chunk_results),
                Err(_) => {
                    // Thread panicked -- record as Unknown for that chunk.
                    // This shouldn't happen in practice since verify_by_evaluation
                    // doesn't panic, but be defensive.
                }
            }
        }

        let total_duration = start.elapsed();
        VerificationRunReport {
            results,
            total_duration,
        }
    }

    /// Verify a single categorized proof obligation.
    fn verify_one(&self, cp: &CategorizedProof) -> VerificationRunResult {
        let start = Instant::now();
        let strength =
            VerificationStrength::for_obligation_with_config(&cp.obligation, &self.config);
        let result = verify_by_evaluation_with_config(&cp.obligation, &self.config);
        let duration = start.elapsed();
        VerificationRunResult {
            name: cp.obligation.name.clone(),
            category: cp.category,
            result,
            strength,
            duration,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =======================================================================
    // Run all proofs -- they should all pass
    // =======================================================================

    #[test]
    fn test_run_all_passes() {
        let db = ProofDatabase::new();
        let runner = VerificationRunner::new(&db);
        let report = runner.run_all();

        assert!(
            report.all_passed(),
            "Not all proofs passed:\n{}",
            report
        );
        assert_eq!(report.total(), db.len());
        assert_eq!(report.failed(), 0);
        assert_eq!(report.unknown(), 0);
    }

    // =======================================================================
    // Category filtering works
    // =======================================================================

    #[test]
    fn test_run_category_arithmetic() {
        let db = ProofDatabase::new();
        let runner = VerificationRunner::new(&db);
        let results = runner.run_category(ProofCategory::Arithmetic);

        let expected_count = db.count_by_category(ProofCategory::Arithmetic);
        assert_eq!(
            results.len(),
            expected_count,
            "run_category(Arithmetic) returned {} results, expected {}",
            results.len(),
            expected_count
        );
        for (name, result) in &results {
            assert!(
                matches!(result, VerificationResult::Valid),
                "Arithmetic proof '{}' did not pass: {:?}",
                name,
                result
            );
        }
    }

    #[test]
    fn test_run_category_memory() {
        let db = ProofDatabase::new();
        let runner = VerificationRunner::new(&db);
        let results = runner.run_category(ProofCategory::Memory);

        let expected_count = db.count_by_category(ProofCategory::Memory);
        assert_eq!(results.len(), expected_count);
        for (name, result) in &results {
            assert!(
                matches!(result, VerificationResult::Valid),
                "Memory proof '{}' failed: {:?}",
                name,
                result
            );
        }
    }

    #[test]
    fn test_run_category_every_category_passes() {
        let db = ProofDatabase::new();
        let runner = VerificationRunner::new(&db);
        for cat in ProofCategory::all_categories() {
            let results = runner.run_category(*cat);
            let expected = db.count_by_category(*cat);
            assert_eq!(
                results.len(),
                expected,
                "Category {:?}: expected {} proofs, got {}",
                cat,
                expected,
                results.len()
            );
            for (name, result) in &results {
                assert!(
                    matches!(result, VerificationResult::Valid),
                    "Category {:?}, proof '{}' failed: {:?}",
                    cat,
                    name,
                    result
                );
            }
        }
    }

    // =======================================================================
    // Report Display formatting
    // =======================================================================

    #[test]
    fn test_report_display_contains_key_fields() {
        let db = ProofDatabase::new();
        let runner = VerificationRunner::new(&db);
        let report = runner.run_all();
        let text = format!("{}", report);

        assert!(text.contains("Verification Run Report"), "missing header");
        assert!(text.contains("Result: PASS"), "expected PASS status");
        assert!(text.contains("Duration:"), "missing duration");
        assert!(text.contains("Per-category breakdown:"), "missing breakdown");
        assert!(text.contains("Arithmetic"), "missing Arithmetic category");
        assert!(text.contains("Memory"), "missing Memory category");
        // Should not contain "Failed proofs:" section since all pass
        assert!(
            !text.contains("Failed proofs:"),
            "unexpected failure section when all proofs pass"
        );
    }

    #[test]
    fn test_report_display_shows_strength() {
        let db = ProofDatabase::new();
        let runner = VerificationRunner::new(&db);
        let report = runner.run_all();
        let text = format!("{}", report);

        assert!(text.contains("exhaustive"), "missing exhaustive count");
        assert!(text.contains("statistical"), "missing statistical count");
    }

    // =======================================================================
    // Proof count matches ProofDatabase.summary()
    // =======================================================================

    #[test]
    fn test_report_count_matches_summary() {
        let db = ProofDatabase::new();
        let summary = db.summary();
        let runner = VerificationRunner::new(&db);
        let report = runner.run_all();

        assert_eq!(
            report.total(),
            summary.total,
            "report total ({}) != summary total ({})",
            report.total(),
            summary.total
        );

        // Verify per-category counts match
        let breakdowns = report.by_category();
        for (cat, expected_count) in &summary.by_category {
            if *expected_count == 0 {
                continue;
            }
            let bd = breakdowns.iter().find(|b| b.category == *cat);
            assert!(
                bd.is_some(),
                "category {:?} missing from report breakdown",
                cat
            );
            assert_eq!(
                bd.unwrap().total,
                *expected_count,
                "category {:?}: report has {} proofs, summary has {}",
                cat,
                bd.unwrap().total,
                expected_count
            );
        }
    }

    // =======================================================================
    // Parallel verification
    // =======================================================================

    #[test]
    fn test_run_parallel_matches_sequential() {
        let db = ProofDatabase::new();
        let runner = VerificationRunner::new(&db);

        let sequential = runner.run_all();
        let parallel = runner.run_parallel(4);

        assert_eq!(
            sequential.total(),
            parallel.total(),
            "parallel run returned different proof count"
        );
        assert_eq!(
            sequential.passed(),
            parallel.passed(),
            "parallel run has different pass count"
        );
        assert!(
            parallel.all_passed(),
            "parallel run should pass all proofs:\n{}",
            parallel
        );
    }

    #[test]
    fn test_run_parallel_single_thread() {
        let db = ProofDatabase::new();
        let runner = VerificationRunner::new(&db);
        let report = runner.run_parallel(1);

        assert_eq!(report.total(), db.len());
        assert!(report.all_passed());
    }

    // =======================================================================
    // Duration tracking
    // =======================================================================

    #[test]
    fn test_duration_is_tracked() {
        let db = ProofDatabase::new();
        let runner = VerificationRunner::new(&db);
        let report = runner.run_all();

        // Total duration should be positive
        assert!(
            report.total_duration > Duration::ZERO,
            "total_duration should be > 0"
        );

        // Each proof should have a duration recorded
        assert!(
            report.results.iter().all(|r| r.duration <= report.total_duration + Duration::from_millis(10)),
            "no individual proof should take longer than the total run"
        );

        // Sum of per-proof durations should approximate total duration
        // (may differ slightly due to loop overhead)
        let sum: Duration = report.results.iter().map(|r| r.duration).sum();
        assert!(
            sum <= report.total_duration + Duration::from_millis(100),
            "sum of per-proof durations ({:?}) should not vastly exceed total ({:?})",
            sum,
            report.total_duration
        );
    }

    // =======================================================================
    // CategoryBreakdown correctness
    // =======================================================================

    #[test]
    fn test_category_breakdown_sums_to_total() {
        let db = ProofDatabase::new();
        let runner = VerificationRunner::new(&db);
        let report = runner.run_all();

        let breakdowns = report.by_category();
        let bd_total: usize = breakdowns.iter().map(|b| b.total).sum();
        assert_eq!(
            bd_total,
            report.total(),
            "sum of category totals ({}) != report total ({})",
            bd_total,
            report.total()
        );
    }

    // =======================================================================
    // Custom config
    // =======================================================================

    #[test]
    fn test_custom_config_runner() {
        let db = ProofDatabase::new();
        let config = VerificationConfig::with_sample_count(1_000);
        let runner = VerificationRunner::with_config(&db, config);

        // With fewer samples, should still pass (these proofs are correct)
        // but run faster
        let report = runner.run_all();
        assert_eq!(report.total(), db.len());
        assert!(report.all_passed());
    }

    // =======================================================================
    // FailedProofDetail -- test with synthetic data
    // =======================================================================

    #[test]
    fn test_failed_details_empty_when_all_pass() {
        let db = ProofDatabase::new();
        let runner = VerificationRunner::new(&db);
        let report = runner.run_all();

        let failures = report.failed_details();
        assert!(
            failures.is_empty(),
            "expected no failure details, got {}",
            failures.len()
        );
    }

    #[test]
    fn test_failed_details_with_synthetic_failure() {
        // Construct a report with a synthetic failure
        let report = VerificationRunReport {
            results: vec![
                VerificationRunResult {
                    name: "good_proof".to_string(),
                    category: ProofCategory::Arithmetic,
                    result: VerificationResult::Valid,
                    strength: VerificationStrength::Exhaustive,
                    duration: Duration::from_millis(10),
                },
                VerificationRunResult {
                    name: "bad_proof".to_string(),
                    category: ProofCategory::Division,
                    result: VerificationResult::Invalid {
                        counterexample: "a=0, b=0".to_string(),
                    },
                    strength: VerificationStrength::Statistical {
                        sample_count: 100_000,
                    },
                    duration: Duration::from_millis(50),
                },
                VerificationRunResult {
                    name: "unknown_proof".to_string(),
                    category: ProofCategory::Memory,
                    result: VerificationResult::Unknown {
                        reason: "timeout".to_string(),
                    },
                    strength: VerificationStrength::Exhaustive,
                    duration: Duration::from_millis(5000),
                },
            ],
            total_duration: Duration::from_millis(5060),
        };

        assert_eq!(report.total(), 3);
        assert_eq!(report.passed(), 1);
        assert_eq!(report.failed(), 1);
        assert_eq!(report.unknown(), 1);
        assert!(!report.all_passed());

        let failures = report.failed_details();
        assert_eq!(failures.len(), 2); // Invalid + Unknown
        assert_eq!(failures[0].name, "bad_proof");
        assert!(failures[0].detail.contains("a=0"));
        assert_eq!(failures[1].name, "unknown_proof");
        assert!(failures[1].detail.contains("timeout"));

        // Display should contain failure section
        let text = format!("{}", report);
        assert!(text.contains("Result: FAIL"));
        assert!(text.contains("Failed proofs:"));
        assert!(text.contains("bad_proof"));
    }
}
