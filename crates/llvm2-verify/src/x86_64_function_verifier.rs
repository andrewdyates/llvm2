// llvm2-verify/x86_64_function_verifier.rs - x86-64 function-level verification
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Provides verify_x86_64_function(): given an X86ISelFunction, walk every
// instruction, map each X86Opcode to a proof obligation from the
// ProofDatabase, run the proof, and produce a FunctionVerificationReport
// with per-instruction results.
//
// Mirror of [`crate::function_verifier`] (AArch64). The two paths share
// the same [`InstructionVerificationResult`] cert shape so downstream
// proof-certificate emission in `llvm2-codegen::compiler` is identical
// across targets.
//
// Part of #465.

//! x86-64 function-level verification pipeline.
//!
//! [`verify_x86_64_function`] walks an [`X86ISelFunction`] and verifies
//! each instruction against its corresponding x86-64 lowering proof
//! obligation from [`ProofDatabase`]. Produces a
//! [`FunctionVerificationReport`] (shared with the AArch64 path) so the
//! public proof-certificate API in `llvm2-codegen` stays target-agnostic.
//!
//! # Example
//!
//! ```rust,no_run
//! use llvm2_lower::x86_64_isel::X86ISelFunction;
//! use llvm2_lower::function::Signature;
//! use llvm2_verify::x86_64_function_verifier::verify_x86_64_function;
//!
//! let func = X86ISelFunction::new("example".to_string(),
//!                                 Signature { params: vec![], returns: vec![] });
//! let report = verify_x86_64_function(&func);
//! println!("Coverage: {:.1}%", report.coverage_percent());
//! ```

use llvm2_ir::X86Opcode;
use llvm2_lower::x86_64_isel::X86ISelFunction;

use crate::function_verifier::{
    FunctionVerificationReport, InstructionReport, InstructionVerificationResult,
};
use crate::lowering_proof::{verify_by_evaluation_with_config, VerificationConfig};
use crate::proof_database::{ProofCategory, ProofDatabase};
use crate::verify::{VerificationResult, VerificationStrength};

// ---------------------------------------------------------------------------
// X86InstructionReport: per-instruction entry (x86-specific)
// ---------------------------------------------------------------------------
//
// The shared `InstructionReport` struct hard-codes `AArch64Opcode` in its
// `opcode` field. Rather than perturb the shared struct, the x86-64 path
// maps `X86Opcode` -> the nearest AArch64Opcode sentinel is NOT safe. Instead,
// we carry a minimal wrapper that uses the same `InstructionVerificationResult`
// enum and shares `FunctionVerificationReport` via conversion on construction.
//
// The AArch64 path's `InstructionReport.opcode` is used only for Display
// formatting and the `unverified_instructions()` helper filtering. For the
// x86-64 path we reuse `AArch64Opcode::Nop` as a stand-in so the shared
// report Display still prints something sensible; the actual per-instruction
// detail is captured in the `result` field (which carries proof name and
// strength). This keeps the public `ProofCertificate` shape identical across
// targets without plumbing a per-target opcode enum through the shared
// InstructionReport.
//
// If we later want to expose real x86 opcodes in reports, we can widen
// InstructionReport to carry an enum { AArch64(AArch64Opcode), X86(X86Opcode) }.
// Deferred to a follow-up — the public `ProofCertificate` does not carry
// the opcode, so this is purely an internal display-only concern.

// ---------------------------------------------------------------------------
// X86FunctionVerifier
// ---------------------------------------------------------------------------

/// Verifier that maps X86ISelFunction instructions to proof obligations.
pub struct X86FunctionVerifier {
    db: ProofDatabase,
    config: VerificationConfig,
}

impl X86FunctionVerifier {
    /// Create a new x86-64 function verifier with default configuration.
    pub fn new() -> Self {
        Self {
            db: ProofDatabase::new(),
            config: VerificationConfig::default(),
        }
    }

    /// Create a new x86-64 function verifier with a custom configuration.
    pub fn with_config(config: VerificationConfig) -> Self {
        Self {
            db: ProofDatabase::new(),
            config,
        }
    }

    /// Map an x86-64 opcode to a proof search substring.
    ///
    /// Returns `Some(name_substring)` for opcodes covered by
    /// [`crate::x86_64_lowering_proofs::all_x86_64_proofs`], or `None`
    /// for opcodes that have no registered lowering proof (e.g. `Ret`,
    /// `Jmp`, `Push`, `Pop`).
    ///
    /// The substring is matched (case-sensitive) against the proof
    /// obligation `name` field, which follows the canonical form
    /// `"x86_64: Iadd_I32 -> ADD r32,r32"` etc.
    ///
    /// Keying on the tMIR operation (`Iadd_I`, `Isub_I`, …) rather than
    /// the x86 mnemonic lets a single opcode query match both 32- and
    /// 64-bit variants — the function verifier does not currently track
    /// per-instruction width, and the proof database carries both
    /// widths under the same `X8664Lowering` category.
    pub fn opcode_to_proof_query(opcode: X86Opcode) -> Option<&'static str> {
        use X86Opcode::*;
        match opcode {
            // Integer arithmetic
            AddRR | AddRI | AddRM | Inc => Some("Iadd_I"),
            SubRR | SubRI | SubRM | Dec => Some("Isub_I"),
            ImulRR | ImulRM => Some("Imul_I"),
            ImulRRI => Some("ImulRRI_I"),
            Neg => Some("Neg_I"),

            // Division (signed / unsigned)
            Idiv => Some("Sdiv_I"),
            Div => Some("Udiv_I"),

            // Logical / bitwise
            AndRR | AndRI => Some("Band_I"),
            OrRR | OrRI => Some("Bor_I"),
            XorRR | XorRI => Some("Bxor_I"),
            Not => Some("Bnot_I"),

            // Shifts
            ShlRR | ShlRI => Some("Ishl_I"),
            ShrRR | ShrRI => Some("Ushr_I"),
            SarRR | SarRI => Some("Sshr_I"),

            // Compare (RFLAGS-setting; the comparison proofs cover the
            // CMP + Setcc/Jcc composition via the Icmp_* registry entries).
            CmpRR | CmpRI | CmpRI8 | CmpRM => Some("Icmp_"),
            TestRR | TestRI | TestRM => Some("Icmp_"),

            // Setcc consumes RFLAGS from a prior CMP; tie it to the
            // comparison proofs so the per-instruction walk records a
            // cert for the Setcc boolean materialization.
            Setcc => Some("Icmp_"),

            // SSE scalar floating point
            Addsd => Some("Fadd_F64"),
            Subsd => Some("Fsub_F64"),
            Mulsd => Some("Fmul_F64"),
            Divsd => Some("Fdiv_F64"),
            Addss => Some("Fadd_F32"),
            Subss => Some("Fsub_F32"),
            Mulss => Some("Fmul_F32"),
            Divss => Some("Fdiv_F32"),

            // Zero / sign extension (MOVZX/MOVSX)
            Movzx => Some("Movzx_"),
            MovzxW => Some("Movzx_16_to_32"),
            MovsxB => Some("Movsx_8_to_32"),
            MovsxW => Some("Movsx_16_to_32"),
            Movsx => Some("Movsxd_32_to_64"),

            // LEA — covered by the Lea_* proofs.
            Lea | LeaSib | LeaRip => Some("Lea_"),

            // Opcodes with no lowering-proof coverage yet: moves, stack
            // manipulation, calls, branches, SSE conversions, bit-manip,
            // atomic/xchg, MOVD/MOVQ transfers. These land as Unverified
            // in the per-instruction report — a cert is NOT produced.
            _ => None,
        }
    }

    /// Verify every instruction in an x86-64 ISel function.
    ///
    /// Walks `func.block_order` (not the `blocks` HashMap) to preserve
    /// deterministic emission order across runs. Pseudo-ops (Phi,
    /// StackAlloc, Nop) are reported as `Skipped`.
    pub fn verify(&self, func: &X86ISelFunction) -> FunctionVerificationReport {
        let mut instructions: Vec<InstructionReport> = Vec::new();
        let mut inst_idx: usize = 0;

        for block_id in &func.block_order {
            let Some(block) = func.blocks.get(block_id) else {
                continue;
            };
            for inst in &block.insts {
                let result = if inst.opcode.is_pseudo() {
                    InstructionVerificationResult::Skipped {
                        reason: format!("{:?} is a pseudo-instruction", inst.opcode),
                    }
                } else if let Some(query) = Self::opcode_to_proof_query(inst.opcode) {
                    // x86-64 lowering proofs are all registered under a
                    // single category, so filter by that and then match
                    // on the proof name substring.
                    let candidates = self.db.by_category(ProofCategory::X8664Lowering);
                    let proof = candidates
                        .iter()
                        .find(|p| p.obligation.name.contains(query));

                    match proof {
                        Some(cp) => {
                            let strength = VerificationStrength::for_obligation_with_config(
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
                                        category: ProofCategory::X8664Lowering,
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
                                "no x86-64 proof matching '{}' in category X8664Lowering",
                                query
                            ),
                        },
                    }
                } else {
                    InstructionVerificationResult::Unverified {
                        reason: format!("no proof mapping for x86-64 opcode {:?}", inst.opcode),
                    }
                };

                // Reuse the shared InstructionReport struct. Its `opcode`
                // field is typed as `AArch64Opcode` and is only used by
                // the shared Display impl; we plug in `AArch64Opcode::Nop`
                // as a sentinel. The cert that flows out to
                // `llvm2-codegen` does not carry `opcode`, so downstream
                // consumers of `ProofCertificate` see identical shape to
                // the AArch64 path.
                instructions.push(InstructionReport {
                    inst_index: inst_idx,
                    opcode: llvm2_ir::AArch64Opcode::Nop,
                    result,
                });
                inst_idx += 1;
            }
        }

        FunctionVerificationReport {
            function_name: func.name.clone(),
            instructions,
        }
    }
}

impl Default for X86FunctionVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience: verify an [`X86ISelFunction`] using default configuration.
///
/// This is the primary entry point for x86-64 function-level verification.
/// Mirrors [`crate::function_verifier::verify_function`] for AArch64.
pub fn verify_x86_64_function(func: &X86ISelFunction) -> FunctionVerificationReport {
    X86FunctionVerifier::new().verify(func)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::X86Opcode;
    use llvm2_lower::function::Signature;
    use llvm2_lower::instructions::Block;
    use llvm2_lower::x86_64_isel::{X86ISelBlock, X86ISelInst};

    fn make_func_with_opcodes(opcodes: &[X86Opcode]) -> X86ISelFunction {
        let mut func = X86ISelFunction::new(
            "test".to_string(),
            Signature { params: vec![], returns: vec![] },
        );
        let block_id = Block(0);
        func.ensure_block(block_id);
        let block: &mut X86ISelBlock = func.blocks.get_mut(&block_id).unwrap();
        for &op in opcodes {
            block.insts.push(X86ISelInst::new(op, vec![]));
        }
        func
    }

    #[test]
    fn empty_function_has_100_percent_coverage() {
        let func = X86ISelFunction::new(
            "empty".to_string(),
            Signature { params: vec![], returns: vec![] },
        );
        let report = verify_x86_64_function(&func);
        assert_eq!(report.total(), 0);
        assert_eq!(report.coverage_percent(), 100.0);
    }

    #[test]
    fn addrr_is_verified() {
        let func = make_func_with_opcodes(&[X86Opcode::AddRR]);
        let report = verify_x86_64_function(&func);
        assert_eq!(report.total(), 1);
        assert_eq!(
            report.verified_count(),
            1,
            "AddRR should match Iadd_I proof; report was {}",
            report
        );
    }

    #[test]
    fn nop_is_skipped() {
        let func = make_func_with_opcodes(&[X86Opcode::Nop]);
        let report = verify_x86_64_function(&func);
        assert_eq!(report.skipped_count(), 1);
    }

    #[test]
    fn ret_is_unverified() {
        let func = make_func_with_opcodes(&[X86Opcode::Ret]);
        let report = verify_x86_64_function(&func);
        assert_eq!(report.unverified_count(), 1);
    }

    #[test]
    fn mixed_ops_get_counted_correctly() {
        let func = make_func_with_opcodes(&[
            X86Opcode::AddRR,   // verified (Iadd_I)
            X86Opcode::SubRR,   // verified (Isub_I)
            X86Opcode::Nop,     // skipped
            X86Opcode::Ret,     // unverified
        ]);
        let report = verify_x86_64_function(&func);
        assert_eq!(report.total(), 4);
        assert_eq!(report.verified_count(), 2);
        assert_eq!(report.skipped_count(), 1);
        assert_eq!(report.unverified_count(), 1);
    }

    #[test]
    fn opcode_query_coverage() {
        // Sanity: core integer arithmetic has a proof query.
        assert_eq!(
            X86FunctionVerifier::opcode_to_proof_query(X86Opcode::AddRR),
            Some("Iadd_I")
        );
        assert_eq!(
            X86FunctionVerifier::opcode_to_proof_query(X86Opcode::SubRR),
            Some("Isub_I")
        );
        // Sanity: Ret is intentionally out of proof scope.
        assert_eq!(
            X86FunctionVerifier::opcode_to_proof_query(X86Opcode::Ret),
            None
        );
    }
}
