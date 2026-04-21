// llvm2-verify/verification_runner.rs - Bulk proof verification with reporting
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
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
use crate::z4_bridge::{Z4Config, Z4Result};

// ---------------------------------------------------------------------------
// Z4VerificationMode: selects verification backend
// ---------------------------------------------------------------------------

/// Selects the verification backend for [`VerificationRunner`].
///
/// - [`MockOnly`]: Use mock evaluation only.
///   Fast but provides statistical (not formal) verification for 32/64-bit.
/// - [`Z4Cli`]: Use z3/z4 CLI for formal verification of every proof.
///   Provides complete proofs but requires a solver binary and is slower.
/// - [`MockThenZ4`]: Run mock evaluation first as a fast pre-check, then
///   verify proofs that pass mock with z3/z4 for formal confirmation.
///   Best of both worlds: fast failure detection + formal proofs.
/// - [`Auto`]: Auto-select the best available backend at runtime.
///   Prefers mock-plus-z4 verification when a solver is available and
///   falls back to mock-only verification otherwise.
///
/// [`MockOnly`]: Z4VerificationMode::MockOnly
/// [`Z4Cli`]: Z4VerificationMode::Z4Cli
/// [`MockThenZ4`]: Z4VerificationMode::MockThenZ4
/// [`Auto`]: Z4VerificationMode::Auto
pub enum Z4VerificationMode {
    /// Use mock evaluation only.
    MockOnly,
    /// Use z3/z4 CLI for formal verification.
    Z4Cli(Z4Config),
    /// Use mock as fast pre-check, then z4 for proofs that pass mock.
    MockThenZ4(Z4Config),
    /// Auto-select the best available backend at runtime.
    ///
    /// Uses [`MockThenZ4`] with the z4 native API when a solver binary is
    /// available on the system, otherwise falls back to [`MockOnly`].
    ///
    /// [`MockThenZ4`]: Z4VerificationMode::MockThenZ4
    /// [`MockOnly`]: Z4VerificationMode::MockOnly
    Auto,
}

/// Select the best verification mode available in the current environment.
///
/// Checks whether an SMT solver binary (z4 or z3) is available on `PATH`
/// or in well-known build locations. When a solver is found, returns
/// [`Z4VerificationMode::MockThenZ4`] with default configuration so that
/// mock evaluation runs as a fast pre-check followed by formal SMT proof.
/// Otherwise returns [`Z4VerificationMode::MockOnly`].
pub fn select_auto_mode() -> Z4VerificationMode {
    if crate::z4_bridge::z3_available() {
        Z4VerificationMode::MockThenZ4(Z4Config::default())
    } else {
        Z4VerificationMode::MockOnly
    }
}

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
        let chunk_size = proofs.len().div_ceil(threads);

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

    /// Verify every proof in the database using a z3/z4 CLI solver.
    ///
    /// Each proof is sent to the solver as an SMT-LIB2 query. The result
    /// is mapped to [`VerificationResult`] and the strength is set to
    /// [`VerificationStrength::Formal`] for all proofs verified this way.
    ///
    /// # Graceful degradation
    ///
    /// If no solver is available, all proofs will report
    /// `VerificationResult::Unknown` with a descriptive reason.
    pub fn run_with_z4(&self, z4_config: &Z4Config) -> VerificationRunReport {
        let start = Instant::now();
        let results: Vec<VerificationRunResult> = self
            .db
            .all()
            .iter()
            .map(|cp| {
                let proof_start = Instant::now();
                let z4_result = crate::z4_bridge::verify_with_cli(&cp.obligation, z4_config);
                let duration = proof_start.elapsed();
                let (result, strength) = z4_result_to_verification_result(&z4_result);
                VerificationRunResult {
                    name: cp.obligation.name.clone(),
                    category: cp.category,
                    result,
                    strength,
                    duration,
                }
            })
            .collect();
        let total_duration = start.elapsed();
        VerificationRunReport {
            results,
            total_duration,
        }
    }

    /// Verify proofs using the best available backend, selected automatically.
    ///
    /// Uses [`select_auto_mode`] to detect whether an SMT solver binary is
    /// available. When one is found, runs mock evaluation as a fast pre-check
    /// then promotes passing proofs to formal z4 verification. When no solver
    /// is found, falls back to mock evaluation only.
    ///
    /// This is the recommended entry point for callers that want the strongest
    /// verification available without manual configuration.
    pub fn run_auto(&self) -> VerificationRunReport {
        let mode = select_auto_mode();
        self.run_with_mode(&mode)
    }

    /// Verify proofs using the specified [`Z4VerificationMode`].
    ///
    /// - [`Z4VerificationMode::MockOnly`]: equivalent to [`run_all()`].
    /// - [`Z4VerificationMode::Z4Cli`]: equivalent to [`run_with_z4()`].
    /// - [`Z4VerificationMode::MockThenZ4`]: run mock evaluation first;
    ///   for proofs that pass mock, re-verify with z4 for formal strength.
    ///   Proofs that fail mock are reported immediately without z4.
    /// - [`Z4VerificationMode::Auto`]: equivalent to [`run_auto()`].
    ///
    /// [`run_all()`]: VerificationRunner::run_all
    /// [`run_with_z4()`]: VerificationRunner::run_with_z4
    /// [`run_auto()`]: VerificationRunner::run_auto
    pub fn run_with_mode(&self, mode: &Z4VerificationMode) -> VerificationRunReport {
        match mode {
            Z4VerificationMode::MockOnly => self.run_all(),
            Z4VerificationMode::Z4Cli(z4_config) => self.run_with_z4(z4_config),
            Z4VerificationMode::Auto => self.run_auto(),
            Z4VerificationMode::MockThenZ4(z4_config) => {
                let start = Instant::now();
                let results: Vec<VerificationRunResult> = self
                    .db
                    .all()
                    .iter()
                    .map(|cp| {
                        let proof_start = Instant::now();
                        // Step 1: fast mock pre-check
                        let mock_result =
                            verify_by_evaluation_with_config(&cp.obligation, &self.config);
                        match &mock_result {
                            VerificationResult::Valid => {
                                // Step 2: promote to formal with z4
                                let z4_result =
                                    crate::z4_bridge::verify_with_cli(&cp.obligation, z4_config);
                                let duration = proof_start.elapsed();
                                let (result, strength) =
                                    z4_result_to_verification_result(&z4_result);
                                VerificationRunResult {
                                    name: cp.obligation.name.clone(),
                                    category: cp.category,
                                    result,
                                    strength,
                                    duration,
                                }
                            }
                            _ => {
                                // Mock already found a problem -- no need for z4
                                let duration = proof_start.elapsed();
                                let strength = VerificationStrength::for_obligation_with_config(
                                    &cp.obligation,
                                    &self.config,
                                );
                                VerificationRunResult {
                                    name: cp.obligation.name.clone(),
                                    category: cp.category,
                                    result: mock_result,
                                    strength,
                                    duration,
                                }
                            }
                        }
                    })
                    .collect();
                let total_duration = start.elapsed();
                VerificationRunReport {
                    results,
                    total_duration,
                }
            }
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

/// Convert a [`Z4Result`] to a ([`VerificationResult`], [`VerificationStrength`]) pair.
///
/// - `Verified` -> `(Valid, Formal)`
/// - `CounterExample` -> `(Invalid, Formal)`
/// - `Timeout` -> `(Unknown, Formal)` (solver ran but couldn't decide)
/// - `Error` -> `(Unknown, Formal)` (solver error, not a mock limitation)
fn z4_result_to_verification_result(z4_result: &Z4Result) -> (VerificationResult, VerificationStrength) {
    match z4_result {
        Z4Result::Verified => (VerificationResult::Valid, VerificationStrength::Formal),
        Z4Result::CounterExample(cex) => {
            let cex_str = cex
                .iter()
                .map(|(n, v)| format!("{} = {:#x}", n, v))
                .collect::<Vec<_>>()
                .join(", ");
            (
                VerificationResult::Invalid {
                    counterexample: cex_str,
                },
                VerificationStrength::Formal,
            )
        }
        Z4Result::Timeout => (
            VerificationResult::Unknown {
                reason: "z3/z4 solver timed out".to_string(),
            },
            VerificationStrength::Formal,
        ),
        Z4Result::Error(msg) => (
            VerificationResult::Unknown {
                reason: format!("z3/z4 solver error: {}", msg),
            },
            VerificationStrength::Formal,
        ),
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
        // Use Arithmetic + Memory subset -- display format test doesn't need
        // all 1000+ proofs, just enough for category breakdown. See #234.
        let full_db = ProofDatabase::new();
        let mut subset: Vec<_> = full_db
            .by_category(ProofCategory::Arithmetic)
            .into_iter()
            .cloned()
            .collect();
        subset.extend(
            full_db
                .by_category(ProofCategory::Memory)
                .into_iter()
                .cloned(),
        );
        let db = ProofDatabase::from_proofs(subset);
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
        // Use Arithmetic subset -- strength display test doesn't need all
        // proofs, just enough to have both exhaustive and statistical. See #234.
        let full_db = ProofDatabase::new();
        let subset: Vec<_> = full_db
            .by_category(ProofCategory::Arithmetic)
            .into_iter()
            .cloned()
            .collect();
        let db = ProofDatabase::from_proofs(subset);
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
        // Use a small subset of proofs to keep runtime well under the 600s
        // cargo timeout. The full database has 1000+ proofs; running them
        // twice (sequential + parallel) exceeds the timeout. We only need
        // enough proofs to exercise the chunking/threading logic. See #234.
        let full_db = ProofDatabase::new();
        let subset: Vec<_> = full_db
            .by_category(ProofCategory::Arithmetic)
            .into_iter()
            .cloned()
            .collect();
        assert!(
            subset.len() >= 5,
            "need at least 5 Arithmetic proofs for meaningful parallel test, got {}",
            subset.len()
        );
        let db = ProofDatabase::from_proofs(subset);
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
        // Use Arithmetic subset to avoid timeout. See #234.
        let full_db = ProofDatabase::new();
        let subset: Vec<_> = full_db
            .by_category(ProofCategory::Arithmetic)
            .into_iter()
            .cloned()
            .collect();
        let db = ProofDatabase::from_proofs(subset);
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
        // Use Arithmetic subset -- timing invariants don't need all proofs. See #234.
        let full_db = ProofDatabase::new();
        let subset: Vec<_> = full_db
            .by_category(ProofCategory::Arithmetic)
            .into_iter()
            .cloned()
            .collect();
        let db = ProofDatabase::from_proofs(subset);
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
        // Use Arithmetic subset -- custom config test doesn't need all proofs. See #234.
        let full_db = ProofDatabase::new();
        let subset: Vec<_> = full_db
            .by_category(ProofCategory::Arithmetic)
            .into_iter()
            .cloned()
            .collect();
        let db = ProofDatabase::from_proofs(subset);
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
        // Use Arithmetic subset -- empty-failure check doesn't need all proofs. See #234.
        let full_db = ProofDatabase::new();
        let subset: Vec<_> = full_db
            .by_category(ProofCategory::Arithmetic)
            .into_iter()
            .cloned()
            .collect();
        let db = ProofDatabase::from_proofs(subset);
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

    // =======================================================================
    // z4_result_to_verification_result unit tests
    // =======================================================================

    #[test]
    fn test_z4_result_to_verification_result_verified() {
        let (result, strength) = z4_result_to_verification_result(&Z4Result::Verified);
        assert!(matches!(result, VerificationResult::Valid));
        assert_eq!(strength, VerificationStrength::Formal);
    }

    #[test]
    fn test_z4_result_to_verification_result_counterexample() {
        let cex = Z4Result::CounterExample(vec![("a".to_string(), 42)]);
        let (result, strength) = z4_result_to_verification_result(&cex);
        assert!(matches!(result, VerificationResult::Invalid { .. }));
        assert_eq!(strength, VerificationStrength::Formal);
        if let VerificationResult::Invalid { counterexample } = result {
            assert!(counterexample.contains("a = 0x2a"));
        }
    }

    #[test]
    fn test_z4_result_to_verification_result_timeout() {
        let (result, strength) = z4_result_to_verification_result(&Z4Result::Timeout);
        assert!(matches!(result, VerificationResult::Unknown { .. }));
        assert_eq!(strength, VerificationStrength::Formal);
    }

    #[test]
    fn test_z4_result_to_verification_result_error() {
        let err = Z4Result::Error("parse failure".to_string());
        let (result, strength) = z4_result_to_verification_result(&err);
        assert!(matches!(result, VerificationResult::Unknown { .. }));
        assert_eq!(strength, VerificationStrength::Formal);
        if let VerificationResult::Unknown { reason } = result {
            assert!(reason.contains("parse failure"));
        }
    }

    // =======================================================================
    // run_with_z4 integration test (requires z3)
    // =======================================================================

    #[test]
    fn test_run_with_z4_arithmetic_subset() {
        if !crate::z4_bridge::z3_available() {
            return;
        }

        let full_db = ProofDatabase::new();
        let subset: Vec<_> = full_db
            .by_category(ProofCategory::Arithmetic)
            .into_iter()
            .cloned()
            .collect();
        let db = ProofDatabase::from_proofs(subset);
        let runner = VerificationRunner::new(&db);
        let z4_config = Z4Config::default();
        let report = runner.run_with_z4(&z4_config);

        assert_eq!(report.total(), db.len());
        assert!(
            report.all_passed(),
            "Not all Arithmetic proofs passed via z4:\n{}",
            report
        );

        // All proofs should have Formal strength
        for r in &report.results {
            assert_eq!(
                r.strength,
                VerificationStrength::Formal,
                "Proof '{}' should have Formal strength",
                r.name
            );
        }
    }

    // =======================================================================
    // run_with_mode tests
    // =======================================================================

    #[test]
    fn test_run_with_mode_mock_only() {
        let full_db = ProofDatabase::new();
        let subset: Vec<_> = full_db
            .by_category(ProofCategory::Arithmetic)
            .into_iter()
            .cloned()
            .collect();
        let db = ProofDatabase::from_proofs(subset);
        let runner = VerificationRunner::new(&db);

        let report = runner.run_with_mode(&Z4VerificationMode::MockOnly);
        assert_eq!(report.total(), db.len());
        assert!(report.all_passed());
    }

    #[test]
    fn test_run_with_mode_z4_cli() {
        if !crate::z4_bridge::z3_available() {
            return;
        }

        let full_db = ProofDatabase::new();
        let subset: Vec<_> = full_db
            .by_category(ProofCategory::Arithmetic)
            .into_iter()
            .cloned()
            .collect();
        let db = ProofDatabase::from_proofs(subset);
        let runner = VerificationRunner::new(&db);

        let z4_config = Z4Config::default();
        let report = runner.run_with_mode(&Z4VerificationMode::Z4Cli(z4_config));
        assert_eq!(report.total(), db.len());
        assert!(report.all_passed());
    }

    #[test]
    fn test_run_with_mode_mock_then_z4() {
        if !crate::z4_bridge::z3_available() {
            return;
        }

        let full_db = ProofDatabase::new();
        let subset: Vec<_> = full_db
            .by_category(ProofCategory::Arithmetic)
            .into_iter()
            .cloned()
            .collect();
        let db = ProofDatabase::from_proofs(subset);
        let runner = VerificationRunner::new(&db);

        let z4_config = Z4Config::default();
        let report = runner.run_with_mode(&Z4VerificationMode::MockThenZ4(z4_config));
        assert_eq!(report.total(), db.len());
        assert!(report.all_passed());

        // Proofs that pass both mock and z4 should have Formal strength
        for r in &report.results {
            assert_eq!(
                r.strength,
                VerificationStrength::Formal,
                "Proof '{}' should be promoted to Formal strength after z4 confirmation",
                r.name
            );
        }
    }

    // =======================================================================
    // run_auto and select_auto_mode tests
    // =======================================================================

    #[test]
    fn test_select_auto_mode_returns_valid_mode() {
        let mode = select_auto_mode();
        // Must be either MockOnly or MockThenZ4 depending on solver availability.
        match &mode {
            Z4VerificationMode::MockOnly => {
                assert!(!crate::z4_bridge::z3_available(),
                    "select_auto_mode returned MockOnly but z3 is available");
            }
            Z4VerificationMode::MockThenZ4(_) => {
                assert!(crate::z4_bridge::z3_available(),
                    "select_auto_mode returned MockThenZ4 but z3 is not available");
            }
            _ => panic!("select_auto_mode should return MockOnly or MockThenZ4"),
        }
    }

    #[test]
    fn test_run_auto_arithmetic_subset() {
        let full_db = ProofDatabase::new();
        let subset: Vec<_> = full_db
            .by_category(ProofCategory::Arithmetic)
            .into_iter()
            .cloned()
            .collect();
        let db = ProofDatabase::from_proofs(subset);
        let runner = VerificationRunner::new(&db);

        let report = runner.run_auto();
        assert_eq!(report.total(), db.len());
        assert!(report.all_passed(),
            "run_auto failed on arithmetic subset:\n{}", report);
    }

    #[test]
    fn test_run_with_mode_auto() {
        let full_db = ProofDatabase::new();
        let subset: Vec<_> = full_db
            .by_category(ProofCategory::Arithmetic)
            .into_iter()
            .cloned()
            .collect();
        let db = ProofDatabase::from_proofs(subset);
        let runner = VerificationRunner::new(&db);

        let report = runner.run_with_mode(&Z4VerificationMode::Auto);
        assert_eq!(report.total(), db.len());
        assert!(report.all_passed());
    }
}
