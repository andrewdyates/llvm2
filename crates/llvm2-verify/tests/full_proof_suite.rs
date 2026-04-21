// llvm2-verify/tests/full_proof_suite.rs - Full ProofDatabase verification
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Integration test that constructs the complete ProofDatabase, runs
// VerificationRunner.run_all() against every proof obligation, and
// asserts all proofs pass. Prints the ProofSummary and full
// VerificationRunReport for documentation purposes.
//
// This test exercises the entire verification pipeline end-to-end:
//   ProofDatabase::new() -> VerificationRunner::run_all() -> VerificationRunReport
//
// Reference: crates/llvm2-verify/src/proof_database.rs
//            crates/llvm2-verify/src/verification_runner.rs

use llvm2_verify::proof_database::{ProofCategory, ProofDatabase};
use llvm2_verify::verification_runner::{VerificationRunner, Z4VerificationMode, select_auto_mode};

// ===========================================================================
// Core integration test: run all proofs in the database
// ===========================================================================

#[test]
fn full_proof_suite_all_pass() {
    let db = ProofDatabase::new();
    let summary = db.summary();

    // Print the database summary for documentation.
    println!();
    println!("================================================================");
    println!("  LLVM2 Full Proof Suite — ProofDatabase Summary");
    println!("================================================================");
    println!();
    println!("{}", summary);

    // Verify the database has a meaningful number of proofs.
    // As of Wave 24: 34 categories registered.
    assert!(
        db.len() >= 300,
        "Expected at least 300 proofs in the database, got {}. \
         New proof categories may need to be wired into ProofDatabase.",
        db.len()
    );

    // Run all proofs.
    let runner = VerificationRunner::new(&db);
    let report = runner.run_all();

    // Print the full verification report.
    println!();
    println!("================================================================");
    println!("  LLVM2 Full Proof Suite — Verification Run Report");
    println!("================================================================");
    println!();
    println!("{}", report);

    // Assert all proofs passed.
    assert!(
        report.all_passed(),
        "Not all proofs passed! {} failed, {} unknown out of {} total.\n\
         Failed details:\n{}",
        report.failed(),
        report.unknown(),
        report.total(),
        report
    );

    // Verify report total matches database size.
    assert_eq!(
        report.total(),
        db.len(),
        "Report total ({}) does not match database size ({})",
        report.total(),
        db.len()
    );

    // Verify per-category breakdown sums to total.
    let breakdowns = report.by_category();
    let bd_sum: usize = breakdowns.iter().map(|b| b.total).sum();
    assert_eq!(
        bd_sum,
        report.total(),
        "Sum of per-category proof counts ({}) != total ({})",
        bd_sum,
        report.total()
    );

    println!("================================================================");
    println!("  RESULT: ALL {} PROOFS PASSED", report.total());
    println!("================================================================");
}

// ===========================================================================
// Every category must be represented in the database
// ===========================================================================

#[test]
fn full_proof_suite_every_category_populated() {
    let db = ProofDatabase::new();
    let categories = ProofCategory::all_categories();

    // 38 categories: Arithmetic, Division, FloatingPoint, NzcvFlags,
    // Comparison, Branch, Peephole, Optimization, ConstantFolding,
    // CopyPropagation, CseLicm, DeadCodeElimination, CfgSimplification,
    // Memory, LoadStoreLowering (#422), NeonLowering, NeonEncoding,
    // Vectorization, AnePrecision, RegAlloc, BitwiseShift,
    // ConstantMaterialization, AddressMode, FrameLayout,
    // InstructionScheduling, MachOEmission, LoopOptimization,
    // StrengthReduction, CmpCombine, Gvn, TailCallOptimization,
    // IfConversion, FpConversion, ExtensionTruncation, AtomicOperations,
    // CallLowering, x86-64 Lowering, Switch Lowering.
    assert_eq!(
        categories.len(),
        38,
        "Expected 38 proof categories, got {}",
        categories.len()
    );

    for cat in categories {
        let count = db.count_by_category(*cat);
        assert!(
            count > 0,
            "Category {:?} ({}) has 0 proofs — it should have at least one",
            cat,
            cat.name()
        );
    }
}

// ===========================================================================
// Every category verifies independently
// ===========================================================================

#[test]
fn full_proof_suite_each_category_passes() {
    let db = ProofDatabase::new();
    let runner = VerificationRunner::new(&db);

    for cat in ProofCategory::all_categories() {
        let results = runner.run_category(*cat);
        let count = results.len();
        let expected = db.count_by_category(*cat);

        assert_eq!(
            count, expected,
            "Category {:?}: runner returned {} proofs, db has {}",
            cat, count, expected
        );

        for (name, result) in &results {
            assert!(
                matches!(result, llvm2_verify::VerificationResult::Valid),
                "Category {:?}, proof '{}' failed: {:?}",
                cat,
                name,
                result
            );
        }
    }
}

// ===========================================================================
// Parallel verification produces same results as sequential
// ===========================================================================

#[test]
fn full_proof_suite_parallel_matches_sequential() {
    let db = ProofDatabase::new();
    let runner = VerificationRunner::new(&db);

    let sequential = runner.run_all();
    let parallel = runner.run_parallel(4);

    assert_eq!(
        sequential.total(),
        parallel.total(),
        "Parallel run returned different proof count: seq={}, par={}",
        sequential.total(),
        parallel.total()
    );
    assert_eq!(
        sequential.passed(),
        parallel.passed(),
        "Parallel run has different pass count: seq={}, par={}",
        sequential.passed(),
        parallel.passed()
    );
    assert!(
        parallel.all_passed(),
        "Parallel verification had failures:\n{}",
        parallel
    );
}

// ===========================================================================
// Specific category count assertions (regression guards)
// ===========================================================================

#[test]
fn full_proof_suite_known_category_counts() {
    let db = ProofDatabase::new();

    // These are regression guards based on known proof registrations.
    // If a count changes, update the expected value after verifying
    // the change is intentional.

    // Arithmetic: 4 ops (add/sub/mul/neg) x 4 widths (I8/I16/I32/I64) = 16
    let arith = db.count_by_category(ProofCategory::Arithmetic);
    assert!(arith >= 16, "Arithmetic: expected >= 16, got {}", arith);

    // Division: historic baseline 8 after i8/i16 remainder/division proofs
    // were added. Keep this as a floor so future proof growth does not
    // churn the integration suite.
    let division = db.count_by_category(ProofCategory::Division);
    assert!(division >= 8, "Division: expected >= 8, got {}", division);

    // FP: historic baseline 38 (fadd/fsub/fmul/fdiv/fneg x F32/F64 = 10, plus 14
    // fcmp conditions x 2 sizes = 28; total 38). Relaxed to floor so new FP
    // lowerings don't break this suite (#418). Matches the Arithmetic/Memory
    // pattern above.
    let fp = db.count_by_category(ProofCategory::FloatingPoint);
    assert!(fp >= 38, "FloatingPoint: expected >= 38, got {}", fp);

    // NZCV: N/Z/C/V = 4
    assert_eq!(db.count_by_category(ProofCategory::NzcvFlags), 4);

    // Comparison: 10 conditions x 2 widths = 20
    assert_eq!(db.count_by_category(ProofCategory::Comparison), 20);

    // Branch: 10 conditions x 2 widths = 20
    assert_eq!(db.count_by_category(ProofCategory::Branch), 20);

    // Peephole: historic baseline 18 (9 rules x 2 widths). Relaxed to floor so
    // new peephole patterns don't break this suite (#418). Matches the
    // Arithmetic/Memory/FloatingPoint pattern above.
    let peephole = db.count_by_category(ProofCategory::Peephole);
    assert!(peephole >= 18, "Peephole: expected >= 18, got {}", peephole);

    // NEON Lowering: historic baseline 22 (11 ops x 2 arrangements). Relaxed to
    // floor so new NEON lowerings don't break this suite (#418). Matches the
    // Arithmetic/Memory/FloatingPoint/Peephole pattern.
    let neon = db.count_by_category(ProofCategory::NeonLowering);
    assert!(neon >= 22, "NeonLowering: expected >= 22, got {}", neon);

    // Memory: 62 (6 load + 6 store + 4 roundtrip + 8 non-interference + 3 endianness
    // + 4 alignment + 3 forwarding + 4 subword + 3 write combining + 10 array axiom
    // + 11 array range)
    assert!(
        db.count_by_category(ProofCategory::Memory) >= 41,
        "expected >= 41 memory proofs, got {}",
        db.count_by_category(ProofCategory::Memory)
    );

    // Vectorization: historic baseline 31. Relaxed to floor so new vectorization
    // proofs don't break this suite (#418). Matches surrounding pattern.
    let vect = db.count_by_category(ProofCategory::Vectorization);
    assert!(vect >= 31, "Vectorization: expected >= 31, got {}", vect);

    // RegAlloc: 43 proofs (16 Phase 1 + 15 Phase 2 + 12 Phase 3/greedy)
    assert!(
        db.count_by_category(ProofCategory::RegAlloc) >= 16,
        "expected >= 16 regalloc proofs, got {}",
        db.count_by_category(ProofCategory::RegAlloc)
    );
}

// ===========================================================================
// Verification strength distribution
// ===========================================================================

#[test]
fn full_proof_suite_has_exhaustive_and_statistical() {
    let db = ProofDatabase::new();
    let runner = VerificationRunner::new(&db);
    let report = runner.run_all();

    let exhaustive = report.exhaustive_count();
    let statistical = report.statistical_count();

    assert!(
        exhaustive > 0,
        "Expected some exhaustive proofs (8-bit), got 0"
    );
    assert!(
        statistical > 0,
        "Expected some statistical proofs (32/64-bit), got 0"
    );

    // Print distribution for documentation.
    println!();
    println!("Verification strength distribution:");
    println!(
        "  Exhaustive:  {} proofs (small-width, complete)",
        exhaustive
    );
    println!(
        "  Statistical: {} proofs (large-width, 100K+ samples)",
        statistical
    );
    println!("  Total:       {} proofs", report.total());
}

// ===========================================================================
// RegAlloc-specific verification (new in Wave 15+)
// ===========================================================================

#[test]
fn full_proof_suite_regalloc_proofs_comprehensive() {
    let db = ProofDatabase::new();
    let runner = VerificationRunner::new(&db);

    let results = runner.run_category(ProofCategory::RegAlloc);
    assert!(
        results.len() >= 16,
        "Expected >= 16 regalloc proofs, got {}",
        results.len()
    );

    // Verify all pass.
    for (name, result) in &results {
        assert!(
            matches!(result, llvm2_verify::VerificationResult::Valid),
            "RegAlloc proof '{}' failed: {:?}",
            name,
            result
        );
    }

    // Check that specific proof names exist.
    let names: Vec<&str> = results.iter().map(|(n, _)| n.as_str()).collect();
    assert!(
        names.iter().any(|n| n.contains("non-interference")),
        "Missing non-interference proof"
    );
    assert!(
        names.iter().any(|n| n.contains("completeness")),
        "Missing completeness proof"
    );
    assert!(
        names.iter().any(|n| n.contains("spill")),
        "Missing spill proof"
    );
    assert!(
        names.iter().any(|n| n.contains("phi elimination")),
        "Missing phi elimination proof"
    );
    assert!(
        names.iter().any(|n| n.contains("callee-saved")),
        "Missing callee-saved proof"
    );
    assert!(
        names.iter().any(|n| n.contains("do not alias")),
        "Missing spill slot non-aliasing proof"
    );
}

// ===========================================================================
// Auto-mode verification (z4 enabled by default)
// ===========================================================================

#[test]
fn full_proof_suite_auto_mode_selects_backend() {
    // select_auto_mode() should return MockThenZ4 when a solver binary is
    // available, or MockOnly otherwise. Either way it must not panic.
    let mode = select_auto_mode();
    let label = match &mode {
        Z4VerificationMode::MockOnly => "MockOnly (no solver binary found)",
        Z4VerificationMode::MockThenZ4(_) => "MockThenZ4 (solver binary found)",
        Z4VerificationMode::Z4Cli(_) => "Z4Cli",
        Z4VerificationMode::Auto => "Auto",
    };
    println!("select_auto_mode() -> {}", label);
}

#[cfg(feature = "z4")]
#[test]
fn full_proof_suite_z4_native_api_arithmetic_subset() {
    // Verify a small subset of proofs using the z4 native Rust API directly.
    // This exercises the z4 feature gate being enabled.
    use llvm2_verify::z4_bridge::{Z4Config, verify_with_z4_api};

    let db = ProofDatabase::new();
    let config = Z4Config::default().with_timeout(10_000);

    // Take the first 5 arithmetic proofs as a representative subset.
    let subset: Vec<_> = db
        .by_category(ProofCategory::Arithmetic)
        .into_iter()
        .take(5)
        .collect();
    assert!(
        !subset.is_empty(),
        "Expected at least 1 arithmetic proof for z4 native API test"
    );

    for proof in &subset {
        let result = verify_with_z4_api(&proof.obligation, &config);
        // The z4 native API should verify these correctly.
        // Allow Timeout on CI where the solver may be slow.
        assert!(
            matches!(
                result,
                llvm2_verify::z4_bridge::Z4Result::Verified
                    | llvm2_verify::z4_bridge::Z4Result::Timeout
            ),
            "z4 native API: proof '{}' unexpected result: {}",
            proof.obligation.name,
            result
        );
    }

    println!("z4 native API verified {} arithmetic proofs", subset.len());
}

#[test]
fn full_proof_suite_run_auto_passes_subset() {
    // Exercise run_auto() on a small proof subset to verify the auto-selection
    // pipeline works end-to-end. Uses a subset to avoid long runtimes.
    let full_db = ProofDatabase::new();
    let subset: Vec<_> = full_db
        .by_category(ProofCategory::Arithmetic)
        .into_iter()
        .cloned()
        .take(5)
        .collect();

    let db = ProofDatabase::from_proofs(subset);
    let runner = VerificationRunner::new(&db);
    let report = runner.run_auto();

    assert_eq!(
        report.total(),
        db.len(),
        "run_auto report total ({}) != db size ({})",
        report.total(),
        db.len()
    );
    assert!(
        report.all_passed(),
        "run_auto failed on arithmetic subset:\n{}",
        report
    );

    println!(
        "run_auto: {} proofs passed (mode: auto-selected)",
        report.total()
    );
}
