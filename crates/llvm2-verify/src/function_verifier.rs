// llvm2-verify/function_verifier.rs - Function-level verification pipeline
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Provides verify_function(): given a MachFunction, walk every instruction,
// map each opcode to a proof obligation from the ProofDatabase, run the
// proof, and produce a FunctionVerificationReport with per-instruction
// results and coverage metrics.
//
// Reference: designs/2026-04-13-verification-architecture.md,
//            crates/llvm2-verify/src/proof_database.rs

//! Function-level verification pipeline.
//!
//! [`verify_function`] inspects every instruction in a [`MachFunction`],
//! maps each AArch64 opcode to the corresponding proof obligation from
//! the [`ProofDatabase`], runs the proof via [`verify_by_evaluation`],
//! and produces a [`FunctionVerificationReport`] with per-instruction
//! results and a coverage percentage.
//!
//! # Example
//!
//! ```rust,no_run
//! use llvm2_ir::{MachFunction, Signature};
//! use llvm2_verify::function_verifier::verify_function;
//!
//! let func = MachFunction::new("example".to_string(), Signature::new(vec![], vec![]));
//! let report = verify_function(&func);
//! println!("Coverage: {:.1}%", report.coverage_percent());
//! ```

use llvm2_ir::{AArch64Opcode, MachFunction};

use crate::lowering_proof::{verify_by_evaluation_with_config, VerificationConfig};
use crate::proof_database::{ProofCategory, ProofDatabase};
use crate::verify::{VerificationResult, VerificationStrength};

// ---------------------------------------------------------------------------
// InstructionVerificationResult
// ---------------------------------------------------------------------------

/// Result of verifying a single instruction within a function.
#[derive(Debug, Clone)]
pub enum InstructionVerificationResult {
    /// Instruction was verified against a proof obligation.
    Verified {
        /// Name of the proof obligation that was matched.
        proof_name: String,
        /// Category of the proof.
        category: ProofCategory,
        /// Verification strength achieved.
        strength: VerificationStrength,
    },

    /// Instruction has no corresponding proof obligation in the database.
    Unverified {
        /// Reason verification was not possible.
        reason: String,
    },

    /// Instruction was skipped (pseudo-op with no hardware semantics).
    Skipped {
        /// Why the instruction was skipped.
        reason: String,
    },

    /// Instruction had a matching proof but verification failed.
    Failed {
        /// Name of the proof that failed.
        proof_name: String,
        /// Detail of the failure.
        detail: String,
    },
}

impl InstructionVerificationResult {
    /// Returns true if the instruction was successfully verified.
    pub fn is_verified(&self) -> bool {
        matches!(self, Self::Verified { .. })
    }

    /// Returns true if the instruction was skipped (pseudo-op).
    pub fn is_skipped(&self) -> bool {
        matches!(self, Self::Skipped { .. })
    }

    /// Returns true if no proof was available.
    pub fn is_unverified(&self) -> bool {
        matches!(self, Self::Unverified { .. })
    }

    /// Returns true if verification was attempted but failed.
    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed { .. })
    }
}

// ---------------------------------------------------------------------------
// InstructionReport: per-instruction entry
// ---------------------------------------------------------------------------

/// Per-instruction verification entry in the report.
#[derive(Debug, Clone)]
pub struct InstructionReport {
    /// Index of the instruction in `MachFunction::insts`.
    pub inst_index: usize,
    /// The AArch64 opcode.
    pub opcode: AArch64Opcode,
    /// Verification result for this instruction.
    pub result: InstructionVerificationResult,
}

// ---------------------------------------------------------------------------
// FunctionVerificationReport
// ---------------------------------------------------------------------------

/// Report from verifying all instructions in a MachFunction.
#[derive(Debug, Clone)]
pub struct FunctionVerificationReport {
    /// Function name.
    pub function_name: String,
    /// Per-instruction results.
    pub instructions: Vec<InstructionReport>,
}

impl FunctionVerificationReport {
    /// Total number of instructions examined.
    pub fn total(&self) -> usize {
        self.instructions.len()
    }

    /// Number of instructions that were successfully verified.
    pub fn verified_count(&self) -> usize {
        self.instructions.iter().filter(|r| r.result.is_verified()).count()
    }

    /// Number of instructions that had no proof (unverified).
    pub fn unverified_count(&self) -> usize {
        self.instructions.iter().filter(|r| r.result.is_unverified()).count()
    }

    /// Number of instructions that were skipped (pseudo-ops).
    pub fn skipped_count(&self) -> usize {
        self.instructions.iter().filter(|r| r.result.is_skipped()).count()
    }

    /// Number of instructions where verification failed.
    pub fn failed_count(&self) -> usize {
        self.instructions.iter().filter(|r| r.result.is_failed()).count()
    }

    /// Coverage percentage: verified / (total - skipped) * 100.
    ///
    /// Returns 100.0 for empty functions or functions with only pseudo-ops
    /// (vacuous truth: all non-pseudo instructions are verified).
    pub fn coverage_percent(&self) -> f64 {
        let denominator = self.total() - self.skipped_count();
        if denominator == 0 {
            100.0
        } else {
            (self.verified_count() as f64 / denominator as f64) * 100.0
        }
    }

    /// Returns true if every non-pseudo instruction was verified or skipped.
    pub fn all_verified(&self) -> bool {
        self.unverified_count() == 0 && self.failed_count() == 0
    }

    /// Returns only the unverified instruction reports.
    pub fn unverified_instructions(&self) -> Vec<&InstructionReport> {
        self.instructions.iter().filter(|r| r.result.is_unverified()).collect()
    }

    /// Returns only the failed instruction reports.
    pub fn failed_instructions(&self) -> Vec<&InstructionReport> {
        self.instructions.iter().filter(|r| r.result.is_failed()).collect()
    }
}

impl std::fmt::Display for FunctionVerificationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Function Verification Report: {}", self.function_name)?;
        writeln!(f, "============================================")?;
        writeln!(
            f,
            "Total: {} instructions ({} verified, {} unverified, {} skipped, {} failed)",
            self.total(),
            self.verified_count(),
            self.unverified_count(),
            self.skipped_count(),
            self.failed_count(),
        )?;
        writeln!(f, "Coverage: {:.1}%", self.coverage_percent())?;

        if self.unverified_count() > 0 {
            writeln!(f)?;
            writeln!(f, "Unverified instructions:")?;
            for ir in self.unverified_instructions() {
                if let InstructionVerificationResult::Unverified { ref reason } = ir.result {
                    writeln!(f, "  [{}] {:?} -- {}", ir.inst_index, ir.opcode, reason)?;
                }
            }
        }

        if self.failed_count() > 0 {
            writeln!(f)?;
            writeln!(f, "Failed instructions:")?;
            for ir in self.failed_instructions() {
                if let InstructionVerificationResult::Failed {
                    ref proof_name,
                    ref detail,
                } = ir.result
                {
                    writeln!(
                        f,
                        "  [{}] {:?} -- proof '{}': {}",
                        ir.inst_index, ir.opcode, proof_name, detail
                    )?;
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FunctionVerifier
// ---------------------------------------------------------------------------

/// Verifier that maps MachFunction instructions to proof obligations.
pub struct FunctionVerifier {
    db: ProofDatabase,
    config: VerificationConfig,
}

impl FunctionVerifier {
    /// Create a new function verifier with default configuration.
    pub fn new() -> Self {
        Self {
            db: ProofDatabase::new(),
            config: VerificationConfig::default(),
        }
    }

    /// Create a new function verifier with custom verification configuration.
    pub fn with_config(config: VerificationConfig) -> Self {
        Self {
            db: ProofDatabase::new(),
            config,
        }
    }

    /// Map an AArch64 opcode to a proof search query and category.
    ///
    /// Returns `Some((search_substring, category))` for opcodes that have
    /// corresponding proofs in the database, or `None` for opcodes without
    /// proof coverage.
    pub fn opcode_to_proof_query(opcode: AArch64Opcode) -> Option<(&'static str, ProofCategory)> {
        use AArch64Opcode::*;
        match opcode {
            // Arithmetic
            AddRR | AddRI => Some(("add", ProofCategory::Arithmetic)),
            SubRR | SubRI => Some(("sub", ProofCategory::Arithmetic)),
            MulRR => Some(("mul", ProofCategory::Arithmetic)),
            Neg => Some(("neg", ProofCategory::Arithmetic)),

            // Division
            SDiv => Some(("sdiv", ProofCategory::Division)),
            UDiv => Some(("udiv", ProofCategory::Division)),

            // Compare / NZCV
            CmpRR | CmpRI | CMPWrr | CMPXrr | CMPWri | CMPXri => {
                Some(("cmp", ProofCategory::Comparison))
            }
            Tst => Some(("tst", ProofCategory::NzcvFlags)),

            // Branch
            BCond | Bcc => Some(("condbr", ProofCategory::Branch)),

            // Memory
            LdrRI | LdrbRI | LdrhRI | LdrsbRI | LdrshRI | LdrRO => {
                Some(("load", ProofCategory::Memory))
            }
            StrRI | StrbRI | StrhRI | StrRO | STRWui | STRXui | STRSui | STRDui => {
                Some(("store", ProofCategory::Memory))
            }

            // Floating-point
            FaddRR => Some(("fadd", ProofCategory::FloatingPoint)),
            FsubRR => Some(("fsub", ProofCategory::FloatingPoint)),
            FmulRR => Some(("fmul", ProofCategory::FloatingPoint)),
            FnegRR => Some(("fneg", ProofCategory::FloatingPoint)),

            // Peephole-covered logical ops
            AndRR | AndRI => Some(("and", ProofCategory::Peephole)),
            OrrRR | OrrRI => Some(("or", ProofCategory::Peephole)),
            EorRR | EorRI => Some(("eor", ProofCategory::Peephole)),

            // Peephole-covered shifts
            LslRR | LslRI => Some(("lsl", ProofCategory::Peephole)),
            LsrRR | LsrRI => Some(("lsr", ProofCategory::Peephole)),
            AsrRR | AsrRI => Some(("asr", ProofCategory::Peephole)),

            // Conditional select (covered by comparison proofs)
            CSet => Some(("cmp", ProofCategory::Comparison)),

            // No proof for these opcodes (yet)
            _ => None,
        }
    }

    /// Verify all instructions in a MachFunction.
    pub fn verify(&self, func: &MachFunction) -> FunctionVerificationReport {
        let mut instructions = Vec::new();

        // Walk all blocks and their instructions.
        for block in &func.blocks {
            for &inst_id in &block.insts {
                let idx = inst_id.0 as usize;
                if idx >= func.insts.len() {
                    continue;
                }
                let inst = &func.insts[idx];

                let result = if inst.opcode.is_pseudo() {
                    InstructionVerificationResult::Skipped {
                        reason: format!("{:?} is a pseudo-instruction", inst.opcode),
                    }
                } else if let Some((query, category)) =
                    Self::opcode_to_proof_query(inst.opcode)
                {
                    // Search the proof database for a matching proof.
                    let matching_proofs = self.db.by_category(category);
                    let proof = matching_proofs
                        .iter()
                        .find(|p| p.obligation.name.to_lowercase().contains(query));

                    match proof {
                        Some(cp) => {
                            let strength =
                                VerificationStrength::for_obligation_with_config(
                                    &cp.obligation,
                                    &self.config,
                                );
                            let vresult = verify_by_evaluation_with_config(
                                &cp.obligation,
                                &self.config,
                            );
                            match vresult {
                                VerificationResult::Valid => {
                                    InstructionVerificationResult::Verified {
                                        proof_name: cp.obligation.name.clone(),
                                        category,
                                        strength,
                                    }
                                }
                                VerificationResult::Invalid { counterexample } => {
                                    InstructionVerificationResult::Failed {
                                        proof_name: cp.obligation.name.clone(),
                                        detail: counterexample,
                                    }
                                }
                                VerificationResult::Unknown { reason } => {
                                    InstructionVerificationResult::Failed {
                                        proof_name: cp.obligation.name.clone(),
                                        detail: format!("Unknown: {}", reason),
                                    }
                                }
                            }
                        }
                        None => InstructionVerificationResult::Unverified {
                            reason: format!(
                                "no proof matching '{}' in category {:?}",
                                query, category
                            ),
                        },
                    }
                } else {
                    InstructionVerificationResult::Unverified {
                        reason: format!("no proof mapping for opcode {:?}", inst.opcode),
                    }
                };

                instructions.push(InstructionReport {
                    inst_index: idx,
                    opcode: inst.opcode,
                    result,
                });
            }
        }

        FunctionVerificationReport {
            function_name: func.name.clone(),
            instructions,
        }
    }
}

impl Default for FunctionVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: verify all instructions in a MachFunction using
/// default configuration.
///
/// This is the primary entry point for function-level verification.
/// Creates a [`FunctionVerifier`] with default settings, walks every
/// instruction in `func`, and returns a [`FunctionVerificationReport`].
pub fn verify_function(func: &MachFunction) -> FunctionVerificationReport {
    let verifier = FunctionVerifier::new();
    verifier.verify(func)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::{AArch64Opcode, MachInst, Signature, Type};
    use llvm2_ir::types::InstId;

    // =======================================================================
    // Test helpers
    // =======================================================================

    fn make_empty_func() -> MachFunction {
        MachFunction::new("test_func".to_string(), Signature::new(vec![], vec![]))
    }

    fn make_func_with_insts(insts: Vec<MachInst>) -> MachFunction {
        let mut func = make_empty_func();
        for (i, inst) in insts.into_iter().enumerate() {
            func.insts.push(inst);
            func.blocks[0].insts.push(InstId(i as u32));
        }
        func
    }

    fn inst(opcode: AArch64Opcode) -> MachInst {
        MachInst::new(opcode, vec![])
    }

    // =======================================================================
    // 1. Empty function
    // =======================================================================

    #[test]
    fn test_empty_function() {
        let func = make_empty_func();
        let report = verify_function(&func);
        assert_eq!(report.total(), 0);
        assert_eq!(report.verified_count(), 0);
        assert_eq!(report.skipped_count(), 0);
        assert_eq!(report.unverified_count(), 0);
        assert_eq!(report.coverage_percent(), 100.0);
        assert!(report.all_verified());
    }

    // =======================================================================
    // 2-7. Single verified instructions
    // =======================================================================

    #[test]
    fn test_single_add_verified() {
        let func = make_func_with_insts(vec![inst(AArch64Opcode::AddRR)]);
        let report = verify_function(&func);
        assert_eq!(report.total(), 1);
        assert_eq!(report.verified_count(), 1);
        assert!(report.instructions[0].result.is_verified());
    }

    #[test]
    fn test_single_sub_verified() {
        let func = make_func_with_insts(vec![inst(AArch64Opcode::SubRR)]);
        let report = verify_function(&func);
        assert_eq!(report.verified_count(), 1);
    }

    #[test]
    fn test_single_mul_verified() {
        let func = make_func_with_insts(vec![inst(AArch64Opcode::MulRR)]);
        let report = verify_function(&func);
        assert_eq!(report.verified_count(), 1);
    }

    #[test]
    fn test_single_neg_verified() {
        let func = make_func_with_insts(vec![inst(AArch64Opcode::Neg)]);
        let report = verify_function(&func);
        assert_eq!(report.verified_count(), 1);
    }

    #[test]
    fn test_sdiv_verified() {
        let func = make_func_with_insts(vec![inst(AArch64Opcode::SDiv)]);
        let report = verify_function(&func);
        assert_eq!(report.verified_count(), 1);
    }

    #[test]
    fn test_udiv_verified() {
        let func = make_func_with_insts(vec![inst(AArch64Opcode::UDiv)]);
        let report = verify_function(&func);
        assert_eq!(report.verified_count(), 1);
    }

    // =======================================================================
    // 8-9. Comparison and branch
    // =======================================================================

    #[test]
    fn test_cmp_verified() {
        let func = make_func_with_insts(vec![inst(AArch64Opcode::CmpRR)]);
        let report = verify_function(&func);
        assert_eq!(report.verified_count(), 1);
        if let InstructionVerificationResult::Verified { category, .. } =
            &report.instructions[0].result
        {
            assert_eq!(*category, ProofCategory::Comparison);
        } else {
            panic!("expected Verified result for CmpRR");
        }
    }

    #[test]
    fn test_bcond_verified() {
        let func = make_func_with_insts(vec![inst(AArch64Opcode::BCond)]);
        let report = verify_function(&func);
        assert_eq!(report.verified_count(), 1);
        if let InstructionVerificationResult::Verified { category, .. } =
            &report.instructions[0].result
        {
            assert_eq!(*category, ProofCategory::Branch);
        } else {
            panic!("expected Verified result for BCond");
        }
    }

    // =======================================================================
    // 10-11. Memory
    // =======================================================================

    #[test]
    fn test_load_verified() {
        let func = make_func_with_insts(vec![inst(AArch64Opcode::LdrRI)]);
        let report = verify_function(&func);
        assert_eq!(report.verified_count(), 1);
        if let InstructionVerificationResult::Verified { category, .. } =
            &report.instructions[0].result
        {
            assert_eq!(*category, ProofCategory::Memory);
        } else {
            panic!("expected Verified result for LdrRI");
        }
    }

    #[test]
    fn test_store_verified() {
        let func = make_func_with_insts(vec![inst(AArch64Opcode::StrRI)]);
        let report = verify_function(&func);
        assert_eq!(report.verified_count(), 1);
        if let InstructionVerificationResult::Verified { category, .. } =
            &report.instructions[0].result
        {
            assert_eq!(*category, ProofCategory::Memory);
        } else {
            panic!("expected Verified result for StrRI");
        }
    }

    // =======================================================================
    // 12. Floating-point
    // =======================================================================

    #[test]
    fn test_fadd_verified() {
        let func = make_func_with_insts(vec![inst(AArch64Opcode::FaddRR)]);
        let report = verify_function(&func);
        assert_eq!(report.verified_count(), 1);
        if let InstructionVerificationResult::Verified { category, .. } =
            &report.instructions[0].result
        {
            assert_eq!(*category, ProofCategory::FloatingPoint);
        } else {
            panic!("expected Verified result for FaddRR");
        }
    }

    // =======================================================================
    // 13. Pseudo-op skipping
    // =======================================================================

    #[test]
    fn test_pseudo_op_skipped() {
        let func = make_func_with_insts(vec![
            inst(AArch64Opcode::Phi),
            inst(AArch64Opcode::Nop),
            inst(AArch64Opcode::Copy),
        ]);
        let report = verify_function(&func);
        assert_eq!(report.total(), 3);
        assert_eq!(report.skipped_count(), 3);
        assert_eq!(report.verified_count(), 0);
        assert_eq!(report.unverified_count(), 0);
        assert!(report.all_verified());
        assert_eq!(report.coverage_percent(), 100.0);
    }

    // =======================================================================
    // 14. Mixed function
    // =======================================================================

    #[test]
    fn test_mixed_function() {
        let func = make_func_with_insts(vec![
            inst(AArch64Opcode::AddRR),   // verified
            inst(AArch64Opcode::Phi),      // skipped
            inst(AArch64Opcode::Ret),      // unverified (no proof for Ret)
            inst(AArch64Opcode::SubRR),    // verified
            inst(AArch64Opcode::Nop),      // skipped
        ]);
        let report = verify_function(&func);
        assert_eq!(report.total(), 5);
        assert_eq!(report.verified_count(), 2);
        assert_eq!(report.skipped_count(), 2);
        assert_eq!(report.unverified_count(), 1);
        assert!(!report.all_verified());
    }

    // =======================================================================
    // 15. 100% coverage
    // =======================================================================

    #[test]
    fn test_coverage_100_percent() {
        let func = make_func_with_insts(vec![
            inst(AArch64Opcode::AddRR),
            inst(AArch64Opcode::SubRR),
            inst(AArch64Opcode::MulRR),
        ]);
        let report = verify_function(&func);
        assert_eq!(report.verified_count(), 3);
        assert_eq!(report.coverage_percent(), 100.0);
        assert!(report.all_verified());
    }

    // =======================================================================
    // 16. 50% coverage
    // =======================================================================

    #[test]
    fn test_coverage_50_percent() {
        let func = make_func_with_insts(vec![
            inst(AArch64Opcode::AddRR),  // verified
            inst(AArch64Opcode::Ret),    // unverified
        ]);
        let report = verify_function(&func);
        assert_eq!(report.verified_count(), 1);
        assert_eq!(report.unverified_count(), 1);
        assert_eq!(report.coverage_percent(), 50.0);
    }

    // =======================================================================
    // 17. 0% coverage
    // =======================================================================

    #[test]
    fn test_coverage_0_percent() {
        let func = make_func_with_insts(vec![
            inst(AArch64Opcode::Ret),    // unverified
            inst(AArch64Opcode::Br),     // unverified
        ]);
        let report = verify_function(&func);
        assert_eq!(report.verified_count(), 0);
        assert_eq!(report.unverified_count(), 2);
        assert_eq!(report.coverage_percent(), 0.0);
    }

    // =======================================================================
    // 18. Display formatting
    // =======================================================================

    #[test]
    fn test_report_display() {
        let func = make_func_with_insts(vec![
            inst(AArch64Opcode::AddRR),
            inst(AArch64Opcode::Ret),
            inst(AArch64Opcode::Phi),
        ]);
        let report = verify_function(&func);
        let text = format!("{}", report);
        assert!(text.contains("Function Verification Report"));
        assert!(text.contains("test_func"));
        assert!(text.contains("Coverage:"));
        assert!(text.contains("verified"));
        assert!(text.contains("Unverified instructions:"));
    }

    // =======================================================================
    // 19. Default trait
    // =======================================================================

    #[test]
    fn test_verifier_default() {
        let verifier = FunctionVerifier::default();
        let func = make_empty_func();
        let report = verifier.verify(&func);
        assert_eq!(report.total(), 0);
        assert!(report.all_verified());
    }

    // =======================================================================
    // 20. Multiple blocks
    // =======================================================================

    #[test]
    fn test_multiple_blocks() {
        let mut func = make_empty_func();

        // Add instructions to block 0
        func.insts.push(inst(AArch64Opcode::AddRR));
        func.blocks[0].insts.push(InstId(0));

        // Create block 1
        let mut block1 = llvm2_ir::MachBlock::new();
        func.insts.push(inst(AArch64Opcode::SubRR));
        block1.insts.push(InstId(1));
        func.blocks.push(block1);

        let report = verify_function(&func);
        assert_eq!(report.total(), 2);
        assert_eq!(report.verified_count(), 2);
        assert_eq!(report.coverage_percent(), 100.0);
    }

    // =======================================================================
    // 21. All pseudo-ops = 100% coverage (vacuous)
    // =======================================================================

    #[test]
    fn test_all_pseudo_ops_skipped() {
        let func = make_func_with_insts(vec![
            inst(AArch64Opcode::Phi),
            inst(AArch64Opcode::Nop),
            inst(AArch64Opcode::Copy),
            inst(AArch64Opcode::StackAlloc),
        ]);
        let report = verify_function(&func);
        assert_eq!(report.total(), 4);
        assert_eq!(report.skipped_count(), 4);
        assert_eq!(report.verified_count(), 0);
        assert_eq!(report.coverage_percent(), 100.0);
        assert!(report.all_verified());
    }

    // =======================================================================
    // 22. Logical ops verified
    // =======================================================================

    #[test]
    fn test_and_or_eor_verified() {
        let func = make_func_with_insts(vec![
            inst(AArch64Opcode::AndRR),
            inst(AArch64Opcode::OrrRR),
            inst(AArch64Opcode::EorRR),
        ]);
        let report = verify_function(&func);
        assert_eq!(report.verified_count(), 3);
        for ir in &report.instructions {
            if let InstructionVerificationResult::Verified { category, .. } = &ir.result {
                assert_eq!(*category, ProofCategory::Peephole);
            } else {
                panic!("expected Verified for logical ops");
            }
        }
    }

    // =======================================================================
    // 23. Shift ops verified
    // =======================================================================

    #[test]
    fn test_shift_ops_verified() {
        let func = make_func_with_insts(vec![
            inst(AArch64Opcode::LslRR),
            inst(AArch64Opcode::LsrRR),
            inst(AArch64Opcode::AsrRR),
        ]);
        let report = verify_function(&func);
        assert_eq!(report.verified_count(), 3);
        for ir in &report.instructions {
            if let InstructionVerificationResult::Verified { category, .. } = &ir.result {
                assert_eq!(*category, ProofCategory::Peephole);
            } else {
                panic!("expected Verified for shift ops");
            }
        }
    }

    // =======================================================================
    // 24. Ret is unverified
    // =======================================================================

    #[test]
    fn test_ret_unverified() {
        let func = make_func_with_insts(vec![inst(AArch64Opcode::Ret)]);
        let report = verify_function(&func);
        assert_eq!(report.unverified_count(), 1);
        assert!(report.instructions[0].result.is_unverified());
    }

    // =======================================================================
    // 25. Report counts are accurate
    // =======================================================================

    #[test]
    fn test_report_verified_count() {
        let func = make_func_with_insts(vec![
            inst(AArch64Opcode::AddRR),   // verified
            inst(AArch64Opcode::SubRR),   // verified
            inst(AArch64Opcode::MulRR),   // verified
            inst(AArch64Opcode::Neg),     // verified
            inst(AArch64Opcode::Phi),     // skipped
            inst(AArch64Opcode::Ret),     // unverified
        ]);
        let report = verify_function(&func);
        assert_eq!(report.total(), 6);
        assert_eq!(report.verified_count(), 4);
        assert_eq!(report.skipped_count(), 1);
        assert_eq!(report.unverified_count(), 1);
        assert_eq!(report.failed_count(), 0);
        // coverage = 4 / (6 - 1) = 80%
        let expected = 4.0 / 5.0 * 100.0;
        assert!((report.coverage_percent() - expected).abs() < 0.01);
    }

    // =======================================================================
    // 26. with_config constructor
    // =======================================================================

    #[test]
    fn test_with_config() {
        let config = VerificationConfig::with_sample_count(1_000);
        let verifier = FunctionVerifier::with_config(config);
        let func = make_func_with_insts(vec![inst(AArch64Opcode::AddRR)]);
        let report = verifier.verify(&func);
        assert_eq!(report.verified_count(), 1);
    }

    // =======================================================================
    // 27. FP ops coverage
    // =======================================================================

    #[test]
    fn test_fp_ops_verified() {
        let func = make_func_with_insts(vec![
            inst(AArch64Opcode::FaddRR),
            inst(AArch64Opcode::FsubRR),
            inst(AArch64Opcode::FmulRR),
            inst(AArch64Opcode::FnegRR),
        ]);
        let report = verify_function(&func);
        assert_eq!(report.verified_count(), 4);
        for ir in &report.instructions {
            if let InstructionVerificationResult::Verified { category, .. } = &ir.result {
                assert_eq!(*category, ProofCategory::FloatingPoint);
            } else {
                panic!("expected Verified for FP ops");
            }
        }
    }

    // =======================================================================
    // 28. opcode_to_proof_query returns None for unmapped
    // =======================================================================

    #[test]
    fn test_opcode_to_proof_query_none() {
        assert!(FunctionVerifier::opcode_to_proof_query(AArch64Opcode::Ret).is_none());
        assert!(FunctionVerifier::opcode_to_proof_query(AArch64Opcode::B).is_none());
        assert!(FunctionVerifier::opcode_to_proof_query(AArch64Opcode::Bl).is_none());
        assert!(FunctionVerifier::opcode_to_proof_query(AArch64Opcode::Blr).is_none());
    }

    // =======================================================================
    // 29. opcode_to_proof_query returns Some for mapped
    // =======================================================================

    #[test]
    fn test_opcode_to_proof_query_some() {
        let (query, cat) = FunctionVerifier::opcode_to_proof_query(AArch64Opcode::AddRR).unwrap();
        assert_eq!(query, "add");
        assert_eq!(cat, ProofCategory::Arithmetic);

        let (query, cat) = FunctionVerifier::opcode_to_proof_query(AArch64Opcode::SDiv).unwrap();
        assert_eq!(query, "sdiv");
        assert_eq!(cat, ProofCategory::Division);

        let (query, cat) = FunctionVerifier::opcode_to_proof_query(AArch64Opcode::LdrRI).unwrap();
        assert_eq!(query, "load");
        assert_eq!(cat, ProofCategory::Memory);
    }

    // =======================================================================
    // 30. Report with function signature
    // =======================================================================

    #[test]
    fn test_function_with_signature() {
        let sig = Signature::new(vec![Type::I64, Type::I64], vec![Type::I64]);
        let mut func = MachFunction::new("add_two".to_string(), sig);
        func.insts.push(inst(AArch64Opcode::AddRR));
        func.blocks[0].insts.push(InstId(0));
        func.insts.push(inst(AArch64Opcode::Ret));
        func.blocks[0].insts.push(InstId(1));

        let report = verify_function(&func);
        assert_eq!(report.function_name, "add_two");
        assert_eq!(report.total(), 2);
        assert_eq!(report.verified_count(), 1);
        assert_eq!(report.unverified_count(), 1);
    }

    // =======================================================================
    // 31. Trap pseudo-ops are skipped
    // =======================================================================

    #[test]
    fn test_trap_pseudo_ops_skipped() {
        let func = make_func_with_insts(vec![
            inst(AArch64Opcode::TrapOverflow),
            inst(AArch64Opcode::TrapBoundsCheck),
            inst(AArch64Opcode::TrapNull),
            inst(AArch64Opcode::TrapDivZero),
            inst(AArch64Opcode::TrapShiftRange),
        ]);
        let report = verify_function(&func);
        assert_eq!(report.skipped_count(), 5);
        assert_eq!(report.coverage_percent(), 100.0);
    }

    // =======================================================================
    // 32. Immediate variants map to same category
    // =======================================================================

    #[test]
    fn test_immediate_variants() {
        let func = make_func_with_insts(vec![
            inst(AArch64Opcode::AddRI),
            inst(AArch64Opcode::SubRI),
            inst(AArch64Opcode::CmpRI),
        ]);
        let report = verify_function(&func);
        assert_eq!(report.verified_count(), 3);
    }
}
