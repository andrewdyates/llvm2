// llvm2-opt - Declarative rewrite as a MachinePass
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! [`DeclarativeRewritePass`] wraps a [`RewriteEngine`] behind the
//! [`MachinePass`] trait so it can be plugged into the existing
//! [`PassManager`](crate::pass_manager::PassManager) pipeline.
//!
//! The wrapper runs the engine's internal fixed-point loop once per
//! invocation and returns whether anything changed. The outer
//! PassManager's fixed-point loop still iterates with other passes.

use llvm2_ir::MachFunction;

use crate::pass_manager::MachinePass;
use crate::rewrite::engine::RewriteEngine;

/// A [`MachinePass`] that runs a [`RewriteEngine`] to local fixed point.
pub struct DeclarativeRewritePass {
    engine: RewriteEngine,
    name: &'static str,
    max_inner_iterations: u32,
}

impl DeclarativeRewritePass {
    /// Create a new pass with the given engine, pass name, and inner
    /// fixed-point iteration cap.
    pub fn new(name: &'static str, engine: RewriteEngine, max_inner_iterations: u32) -> Self {
        Self {
            engine,
            name,
            max_inner_iterations,
        }
    }

    /// Access the underlying engine (e.g., to register more rules).
    pub fn engine_mut(&mut self) -> &mut RewriteEngine {
        &mut self.engine
    }
}

impl MachinePass for DeclarativeRewritePass {
    fn name(&self) -> &str {
        self.name
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        let stats = self.engine.run_to_fixpoint(func, self.max_inner_iterations);
        stats.rewrites > 0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rewrite::patterns::register_migrated;
    use llvm2_ir::{
        AArch64Opcode, AArch64Target, MachInst, MachOperand, RegClass, Signature, TargetInfo, VReg,
    };

    #[test]
    fn pass_fires_on_migrated_patterns() {
        let mut func = MachFunction::new("t".into(), Signature::new(vec![], vec![]));
        let entry = func.entry;
        let inst = MachInst::new(
            AArch64Opcode::AddRI,
            vec![
                MachOperand::VReg(VReg::new(0, RegClass::Gpr64)),
                MachOperand::VReg(VReg::new(1, RegClass::Gpr64)),
                MachOperand::Imm(0),
            ],
        );
        let id = func.push_inst(inst);
        func.append_inst(entry, id);

        let mut engine = RewriteEngine::new();
        register_migrated(&mut engine);
        let mut pass = DeclarativeRewritePass::new("declarative-rewrite", engine, 16);
        assert!(pass.run(&mut func));
        assert_eq!(
            func.inst(func.block(entry).insts[0]).opcode,
            AArch64Target::mov_rr()
        );
    }

    #[test]
    fn pass_idempotent_on_no_match() {
        let mut func = MachFunction::new("t".into(), Signature::new(vec![], vec![]));
        let entry = func.entry;
        let inst = MachInst::new(
            AArch64Opcode::AddRI,
            vec![
                MachOperand::VReg(VReg::new(0, RegClass::Gpr64)),
                MachOperand::VReg(VReg::new(1, RegClass::Gpr64)),
                MachOperand::Imm(42),
            ],
        );
        let id = func.push_inst(inst);
        func.append_inst(entry, id);

        let mut engine = RewriteEngine::new();
        register_migrated(&mut engine);
        let mut pass = DeclarativeRewritePass::new("declarative-rewrite", engine, 16);
        assert!(!pass.run(&mut func));
        assert!(!pass.run(&mut func));
    }
}
