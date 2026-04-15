// llvm2-verify/proof_database.rs - Unified proof obligation database
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Consolidates all proof obligation registries across llvm2-verify into a
// single queryable database. Previously, proofs were scattered across 8+
// modules with separate registry functions (all_arithmetic_proofs(),
// all_division_proofs(), all_fp_lowering_proofs(), etc.). This module
// provides a unified ProofDatabase for inventory, reporting, and
// verification orchestration.
//
// Reference: designs/2026-04-13-verification-architecture.md

//! Unified proof obligation database.
//!
//! [`ProofDatabase`] collects all [`ProofObligation`]s from every proof
//! module in `llvm2-verify` and exposes them through a queryable API:
//!
//! - [`ProofDatabase::all()`] -- every proof obligation in the system.
//! - [`ProofDatabase::by_category()`] -- filter by [`ProofCategory`].
//! - [`ProofDatabase::summary()`] -- counts per category, width, and strength.
//!
//! # Example
//!
//! ```rust
//! use llvm2_verify::proof_database::{ProofDatabase, ProofCategory};
//!
//! let db = ProofDatabase::new();
//! let total = db.all().len();
//! let arith = db.by_category(ProofCategory::Arithmetic).len();
//! let summary = db.summary();
//! assert!(total > 0);
//! assert!(arith > 0);
//! assert_eq!(summary.total, total);
//! ```

use crate::lowering_proof::ProofObligation;

// ---------------------------------------------------------------------------
// ProofCategory enum
// ---------------------------------------------------------------------------

/// Categories of proof obligations in the verification system.
///
/// Each proof obligation belongs to exactly one category, determined by
/// the module and registry function that produces it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProofCategory {
    /// Integer arithmetic lowering: add, sub, mul, neg (I8-I64).
    /// Source: `lowering_proof::all_arithmetic_proofs()` (excludes division subset).
    Arithmetic,

    /// Integer division lowering: sdiv, udiv (I32, I64).
    /// Source: `lowering_proof::all_division_proofs()`.
    Division,

    /// Floating-point lowering: fadd, fsub, fmul, fneg (F32, F64).
    /// Source: `lowering_proof::all_fp_lowering_proofs()`.
    FloatingPoint,

    /// NZCV flag correctness lemmas: N, Z, C, V flags.
    /// Source: `lowering_proof::all_nzcv_flag_proofs()`.
    NzcvFlags,

    /// Comparison lowering: icmp -> CMP + CSET (10 conditions x I32/I64).
    /// Source: `lowering_proof::all_comparison_proofs_i32()` + `_i64()`.
    Comparison,

    /// Branch lowering: condbr -> CMP + B.cond (10 conditions x I32/I64).
    /// Source: `lowering_proof::all_branch_proofs()`.
    Branch,

    /// Peephole optimization identity rules.
    /// Source: `peephole_proofs::all_peephole_proofs_with_32bit()`.
    Peephole,

    /// General optimization pass proofs (const fold, AND/OR absorb, DCE, copy prop).
    /// Source: `opt_proofs::all_opt_proofs()`.
    Optimization,

    /// Constant folding proofs (comprehensive: binary ops, unary ops, identities).
    /// Source: `const_fold_proofs::all_const_fold_proofs_with_variants()`.
    ConstantFolding,

    /// Copy propagation proofs.
    /// Source: `copy_prop_proofs::all_copy_prop_proofs_with_variants()`.
    CopyPropagation,

    /// CSE and LICM proofs.
    /// Source: `cse_licm_proofs::all_cse_licm_proofs()`.
    CseLicm,

    /// Dead code elimination proofs.
    /// Source: `dce_proofs::all_dce_proofs_with_variants()`.
    DeadCodeElimination,

    /// CFG simplification proofs (branch folding, empty block elimination).
    /// Source: `cfg_proofs::all_cfg_proofs_with_variants()`.
    CfgSimplification,

    /// Memory load/store proofs (SMT array theory).
    /// Source: `memory_proofs::all_memory_proofs()`.
    Memory,

    /// NEON SIMD lowering proofs (tMIR vector ops -> NEON instructions).
    /// Source: `neon_lowering_proofs::all_neon_lowering_proofs()`.
    NeonLowering,

    /// Vectorization proofs (scalar-to-NEON mapping correctness).
    /// Source: `vectorization_proofs::all_vectorization_proofs()`.
    Vectorization,

    /// ANE precision proofs (FP16 quantization bounded error).
    /// Source: `ane_precision_proofs::all_ane_precision_proofs()`.
    AnePrecision,

    /// Register allocation correctness proofs (non-interference, completeness,
    /// spill correctness, copy insertion, phi elimination, calling convention,
    /// live-through, spill slot non-aliasing).
    /// Source: `regalloc_proofs::all_regalloc_proofs()`.
    RegAlloc,

    /// Bitwise and shift lowering proofs: AND, OR, XOR, NOT, SHL, LSR, ASR (I8, I16).
    /// Source: `lowering_proof::all_bitwise_shift_proofs()`.
    BitwiseShift,

    /// Constant materialization proofs (MOVZ, MOVZ+MOVK, ORR logical imm, MOVN).
    /// Source: `const_materialize_proofs::all_const_materialize_proofs_with_variants()`.
    ConstantMaterialization,
}

impl ProofCategory {
    /// Return all category variants in declaration order.
    pub fn all_categories() -> &'static [ProofCategory] {
        &[
            ProofCategory::Arithmetic,
            ProofCategory::Division,
            ProofCategory::FloatingPoint,
            ProofCategory::NzcvFlags,
            ProofCategory::Comparison,
            ProofCategory::Branch,
            ProofCategory::Peephole,
            ProofCategory::Optimization,
            ProofCategory::ConstantFolding,
            ProofCategory::CopyPropagation,
            ProofCategory::CseLicm,
            ProofCategory::DeadCodeElimination,
            ProofCategory::CfgSimplification,
            ProofCategory::Memory,
            ProofCategory::NeonLowering,
            ProofCategory::Vectorization,
            ProofCategory::AnePrecision,
            ProofCategory::RegAlloc,
            ProofCategory::BitwiseShift,
            ProofCategory::ConstantMaterialization,
        ]
    }

    /// Human-readable name for this category.
    pub fn name(&self) -> &'static str {
        match self {
            ProofCategory::Arithmetic => "Arithmetic",
            ProofCategory::Division => "Division",
            ProofCategory::FloatingPoint => "Floating-Point",
            ProofCategory::NzcvFlags => "NZCV Flags",
            ProofCategory::Comparison => "Comparison",
            ProofCategory::Branch => "Branch",
            ProofCategory::Peephole => "Peephole",
            ProofCategory::Optimization => "Optimization",
            ProofCategory::ConstantFolding => "Constant Folding",
            ProofCategory::CopyPropagation => "Copy Propagation",
            ProofCategory::CseLicm => "CSE/LICM",
            ProofCategory::DeadCodeElimination => "Dead Code Elimination",
            ProofCategory::CfgSimplification => "CFG Simplification",
            ProofCategory::Memory => "Memory",
            ProofCategory::NeonLowering => "NEON Lowering",
            ProofCategory::Vectorization => "Vectorization",
            ProofCategory::AnePrecision => "ANE Precision",
            ProofCategory::RegAlloc => "Register Allocation",
            ProofCategory::BitwiseShift => "Bitwise/Shift",
            ProofCategory::ConstantMaterialization => "Constant Materialization",
        }
    }
}

impl std::fmt::Display for ProofCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// CategorizedProof: a proof obligation paired with its category
// ---------------------------------------------------------------------------

/// A proof obligation paired with its category.
#[derive(Debug, Clone)]
pub struct CategorizedProof {
    /// The proof obligation.
    pub obligation: ProofObligation,
    /// The category this proof belongs to.
    pub category: ProofCategory,
}

// ---------------------------------------------------------------------------
// ProofSummary: aggregated statistics
// ---------------------------------------------------------------------------

/// Aggregated statistics about the proof database.
#[derive(Debug, Clone)]
pub struct ProofSummary {
    /// Total number of proof obligations.
    pub total: usize,
    /// Count of proofs per category.
    pub by_category: Vec<(ProofCategory, usize)>,
    /// Count of proofs per maximum input bit-width.
    /// Key is the width (8, 16, 32, 64, 128, or 0 for FP-only).
    pub by_width: Vec<(u32, usize)>,
    /// Count of proofs that use floating-point inputs.
    pub fp_proof_count: usize,
    /// Count of proofs with preconditions.
    pub preconditioned_count: usize,
}

impl std::fmt::Display for ProofSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ProofDatabase Summary")?;
        writeln!(f, "=====================")?;
        writeln!(f, "Total proofs: {}", self.total)?;
        writeln!(f)?;
        writeln!(f, "By category:")?;
        for (cat, count) in &self.by_category {
            if *count > 0 {
                writeln!(f, "  {:25} {:>4}", cat.name(), count)?;
            }
        }
        writeln!(f)?;
        writeln!(f, "By max input width:")?;
        for (width, count) in &self.by_width {
            if *count > 0 {
                let label = if *width == 0 { "FP-only".to_string() } else { format!("{}-bit", width) };
                writeln!(f, "  {:25} {:>4}", label, count)?;
            }
        }
        writeln!(f)?;
        writeln!(f, "FP proofs:            {}", self.fp_proof_count)?;
        writeln!(f, "With preconditions:   {}", self.preconditioned_count)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ProofDatabase
// ---------------------------------------------------------------------------

/// Unified database of all proof obligations in llvm2-verify.
///
/// Collects proofs from every registry function across all proof modules
/// and provides query and reporting capabilities.
///
/// # Construction
///
/// ```rust
/// use llvm2_verify::proof_database::ProofDatabase;
/// let db = ProofDatabase::new();
/// ```
///
/// The database is constructed eagerly -- all proofs are materialized
/// on `new()`. For lazy iteration, use the individual module registries.
pub struct ProofDatabase {
    proofs: Vec<CategorizedProof>,
}

impl ProofDatabase {
    /// Construct the database by collecting all proofs from all registries.
    pub fn new() -> Self {
        let mut proofs = Vec::new();

        // Arithmetic lowering (I8-I64: add, sub, mul, neg + division)
        // Note: all_arithmetic_proofs() includes division proofs, so we
        // separate them by registering division first, then arithmetic
        // without the division subset.
        for p in crate::lowering_proof::all_division_proofs() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::Division });
        }
        // Arithmetic minus division: the all_arithmetic_proofs() includes
        // division, so we take the non-division subset (first 16 of 20).
        let all_arith = crate::lowering_proof::all_arithmetic_proofs();
        let div_count = crate::lowering_proof::all_division_proofs().len();
        let arith_take = all_arith.len().saturating_sub(div_count);
        for p in all_arith.into_iter().take(arith_take) {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::Arithmetic });
        }

        // Floating-point lowering
        for p in crate::lowering_proof::all_fp_lowering_proofs() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::FloatingPoint });
        }

        // NZCV flag proofs
        for p in crate::lowering_proof::all_nzcv_flag_proofs() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::NzcvFlags });
        }

        // Comparison proofs (I32 + I64)
        for p in crate::lowering_proof::all_comparison_proofs_i32() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::Comparison });
        }
        for p in crate::lowering_proof::all_comparison_proofs_i64() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::Comparison });
        }

        // Branch proofs (I32 + I64)
        for p in crate::lowering_proof::all_branch_proofs() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::Branch });
        }

        // Peephole proofs (64-bit + 32-bit variants)
        for p in crate::peephole_proofs::all_peephole_proofs_with_32bit() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::Peephole });
        }

        // Optimization pass proofs
        for p in crate::opt_proofs::all_opt_proofs() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::Optimization });
        }

        // Constant folding proofs (with variants)
        for p in crate::const_fold_proofs::all_const_fold_proofs_with_variants() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::ConstantFolding });
        }

        // Copy propagation proofs (with variants)
        for p in crate::copy_prop_proofs::all_copy_prop_proofs_with_variants() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::CopyPropagation });
        }

        // CSE/LICM proofs
        for p in crate::cse_licm_proofs::all_cse_licm_proofs() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::CseLicm });
        }

        // DCE proofs (with variants)
        for p in crate::dce_proofs::all_dce_proofs_with_variants() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::DeadCodeElimination });
        }

        // CFG simplification proofs (with variants)
        for p in crate::cfg_proofs::all_cfg_proofs_with_variants() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::CfgSimplification });
        }

        // Memory proofs (array-based)
        for p in crate::memory_proofs::all_memory_proofs() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::Memory });
        }

        // NEON lowering proofs
        for p in crate::neon_lowering_proofs::all_neon_lowering_proofs() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::NeonLowering });
        }

        // Vectorization proofs
        for p in crate::vectorization_proofs::all_vectorization_proofs() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::Vectorization });
        }

        // ANE precision proofs
        for p in crate::ane_precision_proofs::all_ane_precision_proofs() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::AnePrecision });
        }

        // Register allocation correctness proofs
        for p in crate::regalloc_proofs::all_regalloc_proofs() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::RegAlloc });
        }

        // Bitwise and shift lowering proofs (I8 exhaustive + I16 statistical)
        for p in crate::lowering_proof::all_bitwise_shift_proofs() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::BitwiseShift });
        }

        // Constant materialization proofs (MOVZ, MOVZ+MOVK, ORR, MOVN)
        for p in crate::const_materialize_proofs::all_const_materialize_proofs_with_variants() {
            proofs.push(CategorizedProof { obligation: p, category: ProofCategory::ConstantMaterialization });
        }

        ProofDatabase { proofs }
    }

    /// Return all proof obligations in the database.
    pub fn all(&self) -> &[CategorizedProof] {
        &self.proofs
    }

    /// Return all proof obligations matching the given category.
    pub fn by_category(&self, cat: ProofCategory) -> Vec<&CategorizedProof> {
        self.proofs.iter().filter(|p| p.category == cat).collect()
    }

    /// Return the count of proofs in the given category.
    pub fn count_by_category(&self, cat: ProofCategory) -> usize {
        self.proofs.iter().filter(|p| p.category == cat).count()
    }

    /// Search proofs by name substring (case-insensitive).
    pub fn search(&self, query: &str) -> Vec<&CategorizedProof> {
        let query_lower = query.to_lowercase();
        self.proofs
            .iter()
            .filter(|p| p.obligation.name.to_lowercase().contains(&query_lower))
            .collect()
    }

    /// Return a summary of the proof database.
    pub fn summary(&self) -> ProofSummary {
        let total = self.proofs.len();

        // Count by category
        let by_category: Vec<(ProofCategory, usize)> = ProofCategory::all_categories()
            .iter()
            .map(|cat| (*cat, self.count_by_category(*cat)))
            .collect();

        // Count by max input width
        let mut width_counts = std::collections::HashMap::new();
        for p in &self.proofs {
            let max_width = max_input_width(&p.obligation);
            *width_counts.entry(max_width).or_insert(0usize) += 1;
        }
        let mut by_width: Vec<(u32, usize)> = width_counts.into_iter().collect();
        by_width.sort_by_key(|(w, _)| *w);

        // Count FP proofs
        let fp_proof_count = self
            .proofs
            .iter()
            .filter(|p| !p.obligation.fp_inputs.is_empty())
            .count();

        // Count preconditioned proofs
        let preconditioned_count = self
            .proofs
            .iter()
            .filter(|p| !p.obligation.preconditions.is_empty())
            .count();

        ProofSummary {
            total,
            by_category,
            by_width,
            fp_proof_count,
            preconditioned_count,
        }
    }

    /// Return all distinct proof names (sorted).
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.proofs.iter().map(|p| p.obligation.name.as_str()).collect();
        names.sort();
        names
    }

    /// Total number of proofs.
    pub fn len(&self) -> usize {
        self.proofs.len()
    }

    /// Whether the database is empty.
    pub fn is_empty(&self) -> bool {
        self.proofs.is_empty()
    }
}

impl Default for ProofDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Determine the maximum input bit-width for a proof obligation.
///
/// Returns 0 for FP-only proofs (no bitvector inputs).
fn max_input_width(obligation: &ProofObligation) -> u32 {
    obligation
        .inputs
        .iter()
        .map(|(_, w)| *w)
        .max()
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =======================================================================
    // Construction and basic queries
    // =======================================================================

    #[test]
    fn test_database_is_non_empty() {
        let db = ProofDatabase::new();
        assert!(!db.is_empty(), "database should contain proofs");
        assert!(db.len() > 100, "expected 100+ proofs, got {}", db.len());
    }

    #[test]
    fn test_all_returns_same_count_as_len() {
        let db = ProofDatabase::new();
        assert_eq!(db.all().len(), db.len());
    }

    #[test]
    fn test_default_same_as_new() {
        let db1 = ProofDatabase::new();
        let db2 = ProofDatabase::default();
        assert_eq!(db1.len(), db2.len());
    }

    // =======================================================================
    // Category-specific counts
    // =======================================================================

    #[test]
    fn test_arithmetic_proofs_count() {
        let db = ProofDatabase::new();
        let count = db.count_by_category(ProofCategory::Arithmetic);
        // 4 ops x 4 widths = 16 (excluding division)
        assert!(count >= 16, "expected >= 16 arithmetic proofs, got {}", count);
    }

    #[test]
    fn test_division_proofs_count() {
        let db = ProofDatabase::new();
        let count = db.count_by_category(ProofCategory::Division);
        // sdiv/udiv x I32/I64 = 4
        assert_eq!(count, 4, "expected 4 division proofs, got {}", count);
    }

    #[test]
    fn test_fp_lowering_proofs_count() {
        let db = ProofDatabase::new();
        let count = db.count_by_category(ProofCategory::FloatingPoint);
        // fadd/fsub/fmul/fneg x F32/F64 = 8
        assert_eq!(count, 8, "expected 8 FP proofs, got {}", count);
    }

    #[test]
    fn test_nzcv_flag_proofs_count() {
        let db = ProofDatabase::new();
        let count = db.count_by_category(ProofCategory::NzcvFlags);
        assert_eq!(count, 4, "expected 4 NZCV flag proofs, got {}", count);
    }

    #[test]
    fn test_comparison_proofs_count() {
        let db = ProofDatabase::new();
        let count = db.count_by_category(ProofCategory::Comparison);
        // 10 conditions x 2 widths = 20
        assert_eq!(count, 20, "expected 20 comparison proofs, got {}", count);
    }

    #[test]
    fn test_branch_proofs_count() {
        let db = ProofDatabase::new();
        let count = db.count_by_category(ProofCategory::Branch);
        // 10 conditions x 2 widths = 20
        assert_eq!(count, 20, "expected 20 branch proofs, got {}", count);
    }

    #[test]
    fn test_peephole_proofs_count() {
        let db = ProofDatabase::new();
        let count = db.count_by_category(ProofCategory::Peephole);
        // 9 core (64-bit) + 9 (32-bit) variants = 18
        assert_eq!(count, 18, "expected 18 peephole proofs, got {}", count);
    }

    #[test]
    fn test_neon_lowering_proofs_count() {
        let db = ProofDatabase::new();
        let count = db.count_by_category(ProofCategory::NeonLowering);
        // 11 ops x 2 arrangements = 22
        assert_eq!(count, 22, "expected 22 NEON lowering proofs, got {}", count);
    }

    #[test]
    fn test_vectorization_proofs_count() {
        let db = ProofDatabase::new();
        let count = db.count_by_category(ProofCategory::Vectorization);
        // 11 ops x 2 arrangements = 22 base + 9 additional vector proofs from Wave 13
        assert_eq!(count, 31, "expected 31 vectorization proofs, got {}", count);
    }

    #[test]
    fn test_memory_proofs_count() {
        let db = ProofDatabase::new();
        let count = db.count_by_category(ProofCategory::Memory);
        // 6 load + 6 store + 4 roundtrip + 8 non-interference + 3 endianness = 27 base + 14 additional memory proofs from Wave 13
        assert_eq!(count, 41, "expected 41 memory proofs, got {}", count);
    }

    // =======================================================================
    // Summary
    // =======================================================================

    #[test]
    fn test_summary_total_matches_len() {
        let db = ProofDatabase::new();
        let summary = db.summary();
        assert_eq!(summary.total, db.len());
    }

    #[test]
    fn test_summary_category_counts_sum_to_total() {
        let db = ProofDatabase::new();
        let summary = db.summary();
        let category_sum: usize = summary.by_category.iter().map(|(_, c)| c).sum();
        assert_eq!(
            category_sum, summary.total,
            "sum of category counts ({}) != total ({})",
            category_sum, summary.total
        );
    }

    #[test]
    fn test_summary_width_counts_sum_to_total() {
        let db = ProofDatabase::new();
        let summary = db.summary();
        let width_sum: usize = summary.by_width.iter().map(|(_, c)| c).sum();
        assert_eq!(
            width_sum, summary.total,
            "sum of width counts ({}) != total ({})",
            width_sum, summary.total
        );
    }

    #[test]
    fn test_summary_fp_proofs_exist() {
        let db = ProofDatabase::new();
        let summary = db.summary();
        assert!(
            summary.fp_proof_count > 0,
            "expected at least one FP proof"
        );
    }

    #[test]
    fn test_summary_preconditioned_proofs_exist() {
        let db = ProofDatabase::new();
        let summary = db.summary();
        // Division proofs have non-zero divisor preconditions
        assert!(
            summary.preconditioned_count > 0,
            "expected at least one preconditioned proof"
        );
    }

    // =======================================================================
    // Search
    // =======================================================================

    #[test]
    fn test_search_add() {
        let db = ProofDatabase::new();
        let results = db.search("add");
        assert!(!results.is_empty(), "search for 'add' should find proofs");
    }

    #[test]
    fn test_search_case_insensitive() {
        let db = ProofDatabase::new();
        let upper = db.search("NEON");
        let lower = db.search("neon");
        assert_eq!(upper.len(), lower.len(), "search should be case-insensitive");
    }

    #[test]
    fn test_search_no_match() {
        let db = ProofDatabase::new();
        let results = db.search("zzz_nonexistent_xyz");
        assert!(results.is_empty(), "bogus query should return no results");
    }

    // =======================================================================
    // Names
    // =======================================================================

    #[test]
    fn test_names_sorted() {
        let db = ProofDatabase::new();
        let names = db.names();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "names() should return sorted list");
    }

    #[test]
    fn test_names_count_matches_len() {
        let db = ProofDatabase::new();
        assert_eq!(db.names().len(), db.len());
    }

    // =======================================================================
    // Every category has at least one proof
    // =======================================================================

    #[test]
    fn test_every_category_has_proofs() {
        let db = ProofDatabase::new();
        for cat in ProofCategory::all_categories() {
            let count = db.count_by_category(*cat);
            assert!(
                count > 0,
                "category {:?} has 0 proofs -- every category should have at least one",
                cat
            );
        }
    }

    // =======================================================================
    // ProofCategory::all_categories covers all variants
    // =======================================================================

    #[test]
    fn test_all_categories_is_exhaustive() {
        let categories = ProofCategory::all_categories();
        assert_eq!(
            categories.len(),
            19,
            "expected 19 categories, got {}",
            categories.len()
        );
    }

    #[test]
    fn test_bitwise_shift_proofs_count() {
        let db = ProofDatabase::new();
        let count = db.count_by_category(ProofCategory::BitwiseShift);
        // 7 ops x 2 widths = 14
        assert_eq!(count, 14, "expected 14 bitwise/shift proofs, got {}", count);
    }

    // =======================================================================
    // Summary Display
    // =======================================================================

    #[test]
    fn test_summary_display() {
        let db = ProofDatabase::new();
        let summary = db.summary();
        let text = format!("{}", summary);
        assert!(text.contains("Total proofs:"), "display should show total");
        assert!(text.contains("By category:"), "display should show categories");
    }

    // =======================================================================
    // ProofCategory Display
    // =======================================================================

    #[test]
    fn test_category_display() {
        assert_eq!(format!("{}", ProofCategory::Arithmetic), "Arithmetic");
        assert_eq!(format!("{}", ProofCategory::NeonLowering), "NEON Lowering");
        assert_eq!(format!("{}", ProofCategory::AnePrecision), "ANE Precision");
    }
}
