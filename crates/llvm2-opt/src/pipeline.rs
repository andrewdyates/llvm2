// llvm2-opt - Optimization pipeline
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Optimization pipeline configuration and execution.
//!
//! Builds a `PassManager` with passes appropriate for the requested
//! optimization level. The pipeline follows the ordering from the
//! design doc (designs/2026-04-12-aarch64-backend.md):
//!
//! Pre-register-allocation:
//! 1. Constant folding
//! 2. Copy propagation
//! 3. CSE (common subexpression elimination)
//! 4. LICM (loop-invariant code motion)
//! 5. Peephole
//! 6. DCE
//!
//! Higher optimization levels run additional iterations and enable
//! more aggressive transforms.

use llvm2_ir::MachFunction;

use crate::const_fold::ConstantFolding;
use crate::copy_prop::CopyPropagation;
use crate::cse::CommonSubexprElim;
use crate::dce::DeadCodeElimination;
use crate::licm::LoopInvariantCodeMotion;
use crate::pass_manager::{PassManager, PassStats};
use crate::peephole::Peephole;
use crate::proof_opts::ProofOptimization;

/// Optimization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimizations (fastest compile).
    O0,
    /// Basic optimizations (DCE + peephole).
    O1,
    /// Standard optimizations (full pipeline, 1 iteration).
    O2,
    /// Aggressive optimizations (full pipeline, iterated to fixpoint).
    O3,
    /// Size optimization (same as O2 for now).
    Os,
}

/// Optimization pipeline configuration.
pub struct OptimizationPipeline {
    pub level: OptLevel,
}

impl OptimizationPipeline {
    /// Create a pipeline for the given optimization level.
    pub fn new(level: OptLevel) -> Self {
        Self { level }
    }

    /// Build a PassManager configured for the current optimization level.
    pub fn build_pass_manager(&self) -> PassManager {
        match self.level {
            OptLevel::O0 => {
                // No optimizations at O0.
                PassManager::new()
            }
            OptLevel::O1 => {
                // Basic: just DCE and peephole.
                PassManager::new()
                    .with_pass(Box::new(DeadCodeElimination))
                    .with_pass(Box::new(Peephole))
            }
            OptLevel::O2 | OptLevel::Os => {
                // Standard: full pipeline with proof-consuming opts + CSE and LICM.
                // Proof opts run first: they eliminate checks that DCE/peephole
                // can then clean up further.
                PassManager::new()
                    .with_pass(Box::new(ProofOptimization::new()))
                    .with_pass(Box::new(ConstantFolding))
                    .with_pass(Box::new(CopyPropagation))
                    .with_pass(Box::new(CommonSubexprElim))
                    .with_pass(Box::new(LoopInvariantCodeMotion))
                    .with_pass(Box::new(Peephole))
                    .with_pass(Box::new(DeadCodeElimination))
            }
            OptLevel::O3 => {
                // Aggressive: full pipeline with proof-consuming opts (will be iterated).
                PassManager::new()
                    .with_pass(Box::new(ProofOptimization::new()))
                    .with_pass(Box::new(ConstantFolding))
                    .with_pass(Box::new(CopyPropagation))
                    .with_pass(Box::new(CommonSubexprElim))
                    .with_pass(Box::new(LoopInvariantCodeMotion))
                    .with_pass(Box::new(Peephole))
                    .with_pass(Box::new(DeadCodeElimination))
            }
        }
    }

    /// Run the optimization pipeline on a machine function.
    ///
    /// Returns statistics about the optimization run.
    pub fn run(&self, func: &mut MachFunction) -> PassStats {
        let mut pm = self.build_pass_manager();

        match self.level {
            OptLevel::O0 => {
                // No-op, return empty stats.
                PassStats::default()
            }
            OptLevel::O1 | OptLevel::O2 | OptLevel::Os => {
                pm.run_once_with_stats(func)
            }
            OptLevel::O3 => {
                // Iterate to fixed point (max 10 iterations to bound compile time).
                pm.run_to_fixpoint(func, 10)
            }
        }
    }
}

impl Default for OptimizationPipeline {
    fn default() -> Self {
        Self::new(OptLevel::O2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::{
        AArch64Opcode, MachFunction, MachInst, MachOperand, RegClass, Signature, VReg,
    };

    fn vreg(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
    }

    fn imm(val: i64) -> MachOperand {
        MachOperand::Imm(val)
    }

    fn make_func_with_insts(insts: Vec<MachInst>) -> MachFunction {
        let mut func = MachFunction::new(
            "test_pipeline".to_string(),
            Signature::new(vec![], vec![]),
        );
        let block = func.entry;
        for inst in insts {
            let id = func.push_inst(inst);
            func.append_inst(block, id);
        }
        func
    }

    #[test]
    fn test_o0_no_change() {
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(0), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let pipeline = OptimizationPipeline::new(OptLevel::O0);
        let stats = pipeline.run(&mut func);
        assert_eq!(stats.changes, 0);

        // add #0 should NOT have been peepholed at O0
        let inst = func.inst(llvm2_ir::InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::AddRI);
    }

    #[test]
    fn test_o1_peephole() {
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(0), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let pipeline = OptimizationPipeline::new(OptLevel::O1);
        pipeline.run(&mut func);

        let block = func.block(func.entry);
        // DCE removes the add (v0 unused), only ret remains.
        assert_eq!(block.insts.len(), 1);
    }

    #[test]
    fn test_o2_full_pipeline() {
        let m0 = MachInst::new(AArch64Opcode::MovI, vec![vreg(0), imm(10)]);
        let a1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(0)]);
        let a2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(1), imm(5)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![m0, a1, a2, ret]);

        let pipeline = OptimizationPipeline::new(OptLevel::O2);
        let stats = pipeline.run(&mut func);
        assert!(stats.changes > 0);
    }

    #[test]
    fn test_o3_iterates() {
        let pipeline = OptimizationPipeline::new(OptLevel::O3);
        let pm = pipeline.build_pass_manager();
        // 7 passes: proof-opts + const-fold + copy-prop + cse + licm + peephole + dce
        assert_eq!(pm.num_passes(), 7);
    }
}
