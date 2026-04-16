// llvm2-verify/tests/full_proof_suite.rs - Full ProofDatabase verification
//
// Author: Andrew Yates <ayates@dropbox.com>
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
use llvm2_verify::verification_runner::VerificationRunner;

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

    // 35 categories: Arithmetic, Division, FloatingPoint, NzcvFlags,
    // Comparison, Branch, Peephole, Optimization, ConstantFolding,
    // CopyPropagation, CseLicm, DeadCodeElimination, CfgSimplification,
    // Memory, NeonLowering, NeonEncoding, Vectorization, AnePrecision,
    // RegAlloc, BitwiseShift, ConstantMaterialization, AddressMode,
    // FrameLayout, InstructionScheduling, MachOEmission, LoopOptimization,
    // StrengthReduction, CmpCombine, Gvn, TailCallOptimization,
    // IfConversion, FpConversion, ExtensionTruncation, AtomicOperations,
    // CallLowering.
    assert_eq!(
        categories.len(),
        35,
        "Expected 35 proof categories, got {}",
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

    // Division: sdiv/udiv x I32/I64 = 4
    assert_eq!(db.count_by_category(ProofCategory::Division), 4);

    // FP: fadd/fsub/fmul/fdiv/fneg x F32/F64 = 10, plus 14 fcmp conditions x 2 sizes = 28; total = 38
    assert_eq!(db.count_by_category(ProofCategory::FloatingPoint), 38);

    // NZCV: N/Z/C/V = 4
    assert_eq!(db.count_by_category(ProofCategory::NzcvFlags), 4);

    // Comparison: 10 conditions x 2 widths = 20
    assert_eq!(db.count_by_category(ProofCategory::Comparison), 20);

    // Branch: 10 conditions x 2 widths = 20
    assert_eq!(db.count_by_category(ProofCategory::Branch), 20);

    // Peephole: 9 rules x 2 widths = 18
    assert_eq!(db.count_by_category(ProofCategory::Peephole), 18);

    // NEON Lowering: 11 ops x 2 arrangements = 22
    assert_eq!(db.count_by_category(ProofCategory::NeonLowering), 22);

    // Memory: 62 (6 load + 6 store + 4 roundtrip + 8 non-interference + 3 endianness
    // + 4 alignment + 3 forwarding + 4 subword + 3 write combining + 10 array axiom
    // + 11 array range)
    assert!(db.count_by_category(ProofCategory::Memory) >= 41,
        "expected >= 41 memory proofs, got {}", db.count_by_category(ProofCategory::Memory));

    // Vectorization: 31
    assert_eq!(db.count_by_category(ProofCategory::Vectorization), 31);

    // RegAlloc: 43 proofs (16 Phase 1 + 15 Phase 2 + 12 Phase 3/greedy)
    assert!(db.count_by_category(ProofCategory::RegAlloc) >= 16,
        "expected >= 16 regalloc proofs, got {}", db.count_by_category(ProofCategory::RegAlloc));
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
    println!("  Exhaustive:  {} proofs (small-width, complete)", exhaustive);
    println!("  Statistical: {} proofs (large-width, 100K+ samples)", statistical);
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
