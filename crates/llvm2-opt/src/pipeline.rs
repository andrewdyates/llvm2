// llvm2-opt - Optimization pipeline
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
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
//! 7. Instruction scheduling
//!
//! O1 uses the basic `InstructionScheduler` as the final pass, while
//! higher optimization levels use `PressureAwareScheduler` to balance
//! ILP against register pressure.
//!
//! Higher optimization levels run additional iterations and enable
//! more aggressive transforms.

use llvm2_ir::MachFunction;

use crate::addr_mode::AddrModeFormation;
use crate::cfg_simplify::CfgSimplify;
use crate::cmp_branch_fusion::CmpBranchFusion;
use crate::cmp_select::CmpSelectCombine;
use crate::const_fold::ConstantFolding;
use crate::copy_prop::CopyPropagation;
use crate::cse::CommonSubexprElim;
use crate::dce::DeadCodeElimination;
use crate::gvn::GlobalValueNumbering;
use crate::if_convert::IfConversion;
use crate::inline::FunctionInlining;
use crate::licm::LoopInvariantCodeMotion;
use crate::loop_unroll::LoopUnroll;
use crate::pass_manager::{PassManager, PassStats};
use crate::peephole::Peephole;
use crate::proof_opts::ProofOptimization;
use crate::rewrite::patterns::register_migrated;
use crate::rewrite::{DeclarativeRewritePass, RewriteEngine};
use crate::scheduler::{InstructionScheduler, PressureAwareScheduler};
use crate::sroa::ScalarReplacementOfAggregates;
use crate::strength_reduce::StrengthReduction;
use crate::tail_call::TailCallOptimization;
use crate::vectorize::VectorizationPass;

/// Returns true if declarative rewrite is enabled.
///
/// **Default: ON.** Wave 3 (#393) flipped the default after the single-
/// instruction peephole patterns landed in the declarative framework.
/// The kill switch `LLVM2_DISABLE_DECLARATIVE_REWRITE=1` forces it off
/// so anyone who hits a regression has a clean rollback path without
/// needing to branch from a stable tag.
///
/// `LLVM2_ENABLE_DECLARATIVE_REWRITE` is still honored for one release
/// cycle (explicit opt-in overrides the disable) so transitional CI
/// pipelines that set it do not silently turn OFF the pass when the
/// default flipped.
fn declarative_rewrite_enabled() -> bool {
    // Explicit opt-in always wins (transitional — harmless once all
    // call sites drop the variable).
    if std::env::var("LLVM2_ENABLE_DECLARATIVE_REWRITE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return true;
    }
    // Explicit disable — kill switch.
    if std::env::var("LLVM2_DISABLE_DECLARATIVE_REWRITE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return false;
    }
    // Default: on.
    true
}

/// Construct a fresh declarative rewrite pass pre-loaded with the
/// currently migrated peephole rules. Runs by default; can be turned
/// off with `LLVM2_DISABLE_DECLARATIVE_REWRITE=1`.
fn make_declarative_rewrite_pass() -> DeclarativeRewritePass {
    let mut engine = RewriteEngine::new();
    register_migrated(&mut engine);
    DeclarativeRewritePass::new("declarative-rewrite", engine, 16)
}

/// Optimization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimizations (fastest compile).
    O0,
    /// Basic optimizations (DCE + peephole + scheduling).
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
                // Basic: DCE, declarative rewrite, peephole, late scheduling.
                //
                // Wave 3 (#393) flipped declarative rewrite to default-on.
                // It runs BEFORE the hand-written `Peephole` pass: the
                // declarative pass fires on migrated patterns (single-inst
                // + same-opcode def-use chains), then the hand-written
                // peephole cleans up multi-instruction patterns not yet
                // migrated. Kill switch: `LLVM2_DISABLE_DECLARATIVE_REWRITE=1`.
                let mut pm = PassManager::new().with_pass(Box::new(DeadCodeElimination));
                if declarative_rewrite_enabled() {
                    pm = pm.with_pass(Box::new(make_declarative_rewrite_pass()));
                }
                // SROA before peephole: scalarise aggregate locals so the
                // subsequent peephole/scheduler see plain vreg moves instead
                // of LDR/STR pairs. Part of #391 (aggregate lowering).
                pm.with_pass(Box::new(ScalarReplacementOfAggregates))
                    .with_pass(Box::new(Peephole))
                    .with_pass(Box::new(InstructionScheduler))
            }
            OptLevel::O2 | OptLevel::Os => {
                // Standard: inlining + proof-consuming opts + scalar opts +
                // vectorization (before LICM!) + loop opts + peephole combines,
                // DCE, CFG cleanup, and pressure-aware scheduling last.
                //
                // Vectorization runs BEFORE LICM because LICM hoists
                // loop-invariant pure instructions out of loops, removing
                // the instructions the vectorizer needs to see. This matches
                // LLVM's ordering where LoopVectorize runs early in the loop
                // optimization pipeline. After vectorization rewrites scalar
                // ops to NEON, LICM can still hoist any remaining invariants.
                //
                // Dev note (#366 bisect): set LLVM2_DISABLE_PASSES to a
                // comma-separated list of pass short names to skip at O2.
                // Names: inline, proof, cfold, copyprop, cse, gvn, sroa, vec,
                // licm, strred, unroll, declrewrite, peep, addrmode, cmpsel,
                // ifconv, cmpbr, tailcall, dce, cfgsimp, sched. Empty/missing
                // = run all.
                //
                // Declarative rewrite (#393) is DEFAULT-ON as of Wave 3.
                // It runs BEFORE the hand-written peephole pass — both run
                // in parallel until parity is reached across all patterns.
                // Kill switch: `LLVM2_DISABLE_DECLARATIVE_REWRITE=1` (or
                // `LLVM2_DISABLE_PASSES=declrewrite` for bisects).
                let mut pm = PassManager::new();
                let disabled = std::env::var("LLVM2_DISABLE_PASSES").unwrap_or_default();
                let is_disabled = |name: &str| -> bool {
                    disabled.split(',').any(|d| d.trim() == name)
                };
                if !is_disabled("inline") { pm = pm.with_pass(Box::new(FunctionInlining::new())); }
                if !is_disabled("proof") { pm = pm.with_pass(Box::new(ProofOptimization::new())); }
                if !is_disabled("cfold") { pm = pm.with_pass(Box::new(ConstantFolding)); }
                if !is_disabled("copyprop") { pm = pm.with_pass(Box::new(CopyPropagation)); }
                if !is_disabled("cse") { pm = pm.with_pass(Box::new(CommonSubexprElim)); }
                if !is_disabled("gvn") { pm = pm.with_pass(Box::new(GlobalValueNumbering)); }
                // SROA: scalarise non-escaping aggregate locals (#391 phase 2b).
                // Runs after load-eliminating passes so it sees the canonical
                // LDR/STR pattern, and before loop/peep transforms that might
                // fold scalar moves back into memory ops.
                if !is_disabled("sroa") { pm = pm.with_pass(Box::new(ScalarReplacementOfAggregates)); }
                if !is_disabled("vec") { pm = pm.with_pass(Box::new(VectorizationPass::new())); }
                if !is_disabled("licm") { pm = pm.with_pass(Box::new(LoopInvariantCodeMotion)); }
                if !is_disabled("strred") { pm = pm.with_pass(Box::new(StrengthReduction)); }
                if !is_disabled("unroll") { pm = pm.with_pass(Box::new(LoopUnroll)); }
                if declarative_rewrite_enabled() && !is_disabled("declrewrite") {
                    pm = pm.with_pass(Box::new(make_declarative_rewrite_pass()));
                }
                if !is_disabled("peep") { pm = pm.with_pass(Box::new(Peephole)); }
                if !is_disabled("addrmode") { pm = pm.with_pass(Box::new(AddrModeFormation)); }
                if !is_disabled("cmpsel") { pm = pm.with_pass(Box::new(CmpSelectCombine)); }
                if !is_disabled("ifconv") { pm = pm.with_pass(Box::new(IfConversion)); }
                if !is_disabled("cmpbr") { pm = pm.with_pass(Box::new(CmpBranchFusion)); }
                if !is_disabled("tailcall") { pm = pm.with_pass(Box::new(TailCallOptimization)); }
                if !is_disabled("dce") { pm = pm.with_pass(Box::new(DeadCodeElimination)); }
                if !is_disabled("cfgsimp") { pm = pm.with_pass(Box::new(CfgSimplify)); }
                if !is_disabled("sched") { pm = pm.with_pass(Box::new(PressureAwareScheduler)); }
                pm
            }
            OptLevel::O3 => {
                // Aggressive: same as O2 but iterated to fixpoint.
                // Vectorization before LICM (see O2 comment for rationale).
                //
                // Declarative rewrite (#393) runs BEFORE peephole when the
                // default is on (Wave 3). The kill switch
                // `LLVM2_DISABLE_DECLARATIVE_REWRITE=1` drops the pass for
                // forensic rollback.
                //
                // SROA (#391 Phase 2b) runs once before vectorization,
                // between GVN and VectorizationPass.
                let mut pm = PassManager::new()
                    .with_pass(Box::new(FunctionInlining::new()))
                    .with_pass(Box::new(ProofOptimization::new()))
                    .with_pass(Box::new(ConstantFolding))
                    .with_pass(Box::new(CopyPropagation))
                    .with_pass(Box::new(CommonSubexprElim))
                    .with_pass(Box::new(GlobalValueNumbering))
                    .with_pass(Box::new(ScalarReplacementOfAggregates))
                    .with_pass(Box::new(VectorizationPass::new()))
                    .with_pass(Box::new(LoopInvariantCodeMotion))
                    .with_pass(Box::new(StrengthReduction))
                    .with_pass(Box::new(LoopUnroll));
                if declarative_rewrite_enabled() {
                    pm = pm.with_pass(Box::new(make_declarative_rewrite_pass()));
                }
                pm.with_pass(Box::new(Peephole))
                    .with_pass(Box::new(AddrModeFormation))
                    .with_pass(Box::new(CmpSelectCombine))
                    .with_pass(Box::new(IfConversion))
                    .with_pass(Box::new(CmpBranchFusion))
                    .with_pass(Box::new(TailCallOptimization))
                    .with_pass(Box::new(DeadCodeElimination))
                    .with_pass(Box::new(CfgSimplify))
                    .with_pass(Box::new(PressureAwareScheduler))
            }
        }
    }

    /// Return the number of passes registered for the current optimization
    /// level (single iteration). For `O3` this is the per-iteration count;
    /// actual executions may be higher due to fixpoint iteration.
    pub fn pass_count(&self) -> usize {
        self.build_pass_manager().num_passes()
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
            OptLevel::O1 | OptLevel::O2 | OptLevel::Os => pm.run_once_with_stats(func),
            OptLevel::O3 => {
                // Iterate to fixed point (max 4 iterations to bound compile time).
                // The analysis cache in PassManager avoids redundant domtree
                // recomputation within each iteration.
                pm.run_to_fixpoint(func, 4)
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
        AArch64Opcode, InstId, MachFunction, MachInst, MachOperand, RegClass, Signature,
        VReg,
    };

    fn vreg(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
    }

    fn vreg32(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr32))
    }

    fn imm(val: i64) -> MachOperand {
        MachOperand::Imm(val)
    }

    /// Build a simple vectorizable add loop (i32, trip count 100).
    ///
    /// ```text
    ///   bb0 (entry) -> bb1
    ///   bb1 (header): add v2 = v0 + v1 (i32), cmp v3 #100, bcond bb2/bb3
    ///   bb3 (latch): br bb1
    ///   bb2 (exit): ret
    /// ```
    fn make_vectorizable_loop() -> (MachFunction, InstId) {
        let mut func = MachFunction::new(
            "pipeline_vec_loop".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let bb1 = func.create_block();
        let bb2 = func.create_block();
        let bb3 = func.create_block();

        // bb0: branch to header
        let br0 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb0, br0);

        // bb1 (header): add v2 = v0 + v1 (i32)
        let add = func.push_inst(MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg32(2), vreg32(0), vreg32(1)],
        ));
        func.append_inst(bb1, add);

        // cmp v3, #100
        let cmp = func.push_inst(MachInst::new(
            AArch64Opcode::CmpRI,
            vec![vreg32(3), imm(100)],
        ));
        func.append_inst(bb1, cmp);

        // bcond exit or latch
        let bcond = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb2), MachOperand::Block(bb3)],
        ));
        func.append_inst(bb1, bcond);

        // bb3 (latch): branch back to header
        let br3 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb3, br3);

        // bb2 (exit): ret
        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret);

        func.add_edge(bb0, bb1);
        func.add_edge(bb1, bb2);
        func.add_edge(bb1, bb3);
        func.add_edge(bb3, bb1);

        (func, add)
    }

    /// Build a vectorizable loop with add + sub + mul (i32, trip count 100).
    fn make_multi_op_vectorizable_loop() -> (MachFunction, Vec<InstId>) {
        let mut func = MachFunction::new(
            "pipeline_multi_vec".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let bb1 = func.create_block();
        let bb2 = func.create_block();
        let bb3 = func.create_block();

        let br0 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb0, br0);

        // add v2 = v0 + v1
        let add = func.push_inst(MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg32(2), vreg32(0), vreg32(1)],
        ));
        func.append_inst(bb1, add);

        // sub v3 = v2 - v1
        let sub = func.push_inst(MachInst::new(
            AArch64Opcode::SubRR,
            vec![vreg32(3), vreg32(2), vreg32(1)],
        ));
        func.append_inst(bb1, sub);

        // mul v4 = v3 * v0
        let mul = func.push_inst(MachInst::new(
            AArch64Opcode::MulRR,
            vec![vreg32(4), vreg32(3), vreg32(0)],
        ));
        func.append_inst(bb1, mul);

        let cmp = func.push_inst(MachInst::new(
            AArch64Opcode::CmpRI,
            vec![vreg32(5), imm(100)],
        ));
        func.append_inst(bb1, cmp);

        let bcond = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb2), MachOperand::Block(bb3)],
        ));
        func.append_inst(bb1, bcond);

        let br3 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb3, br3);

        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret);

        func.add_edge(bb0, bb1);
        func.add_edge(bb1, bb2);
        func.add_edge(bb1, bb3);
        func.add_edge(bb3, bb1);

        (func, vec![add, sub, mul])
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

    /// Regression test for issue #432 — CSE Movz/Movn miscompile.
    ///
    /// Before the fix, CSE keyed on `OpcodeCategory::MovRI` + operands,
    /// which collapsed `Movz Xd, #2` (materializes +2) into the same key as
    /// `Movn Xd, #2` (materializes ~2 = -3). The second instruction was
    /// eliminated and its uses rewritten to the first — silently producing
    /// `+2 + +2 = +4` instead of `+2 + -3 = -1`.
    ///
    /// This is a pipeline-context regression: it runs only the CSE pass
    /// (the one that contained the bug) on a MachFunction that exercises
    /// the O2 code path. Running the full `OptimizationPipeline` would
    /// have constant folding absorb the Movz/Movn into concrete integers
    /// before CSE ran, masking the underlying bug. The CSE pass itself is
    /// what ends up in the O1+/O2 pipeline; proving it correct in isolation
    /// is equivalent.
    ///
    /// See `cse::tests::test_cse_movz_movn_same_imm_not_merged` for the
    /// authoritative CSE-pass regression test (empirically verified to
    /// FAIL on pre-fix code and PASS after the fix).
    #[test]
    fn test_cse_pass_preserves_movz_movn_distinct() {
        use crate::cse::CommonSubexprElim;
        use crate::pass_manager::MachinePass;

        // v2 = Movz #2          (materializes +2)
        // v3 = Movn #2          (materializes ~2 = -3)
        // v4 = AddRR v2, v3     (should compute -1 at runtime)
        // ret
        let mz = MachInst::new(AArch64Opcode::Movz, vec![vreg(2), imm(2)]);
        let mn = MachInst::new(AArch64Opcode::Movn, vec![vreg(3), imm(2)]);
        let add = MachInst::new(AArch64Opcode::AddRR, vec![vreg(4), vreg(2), vreg(3)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![mz, mn, add, ret]);

        // Run only the CSE pass (the one that contained the bug).
        let mut cse = CommonSubexprElim;
        let _changed = cse.run(&mut func);

        // Both Movz and Movn must survive.
        let block = func.block(func.entry);
        let mut saw_movz = false;
        let mut saw_movn = false;
        let mut add_srcs: Option<(MachOperand, MachOperand)> = None;
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            match inst.opcode {
                AArch64Opcode::Movz => saw_movz = true,
                AArch64Opcode::Movn => saw_movn = true,
                AArch64Opcode::AddRR => {
                    assert_eq!(inst.operands.len(), 3, "AddRR should have 3 operands");
                    add_srcs = Some((inst.operands[1].clone(), inst.operands[2].clone()));
                }
                _ => {}
            }
        }
        assert!(saw_movz, "CSE eliminated Movz (regression #432)");
        assert!(saw_movn, "CSE eliminated Movn (regression #432)");

        // The AddRR's two sources must be distinct vregs (one per Movz/Movn).
        // If CSE collapses them, both sources point at the same vreg — the
        // exact shape of the miscompile.
        let (src1, src2) = add_srcs.expect("AddRR must remain in the function");
        if let (MachOperand::VReg(a), MachOperand::VReg(b)) = (&src1, &src2) {
            assert_ne!(
                a.id, b.id,
                "AddRR sources collapsed to same vreg — CSE merged Movz+Movn (regression #432)"
            );
        } else {
            panic!("AddRR sources should still be VRegs after CSE: got {:?}, {:?}", src1, src2);
        }
    }

    #[test]
    fn test_o3_iterates() {
        // Force declarative rewrite off so the count is independent of
        // Wave 3's default flip (#393). With default-on, O3 is 21 passes.
        with_declarative_rewrite_env(None, Some("1"), || {
            let pipeline = OptimizationPipeline::new(OptLevel::O3);
            let pm = pipeline.build_pass_manager();
            // 20 passes: inline + proof-opts + const-fold + copy-prop + cse + gvn +
            // sroa + vectorize + licm + strength-reduce + loop-unroll + peephole +
            // addr-mode + cmp-select + if-convert + cmp-branch-fusion + tail-call +
            // dce + cfg-simplify + pressure-aware-scheduler.
            assert_eq!(pm.num_passes(), 20);
        });
    }

    // =========================================================================
    // Pipeline integration tests for NEON auto-vectorization
    // =========================================================================

    /// Verify that O2 pipeline fires vectorization on a simple i32 add loop.
    ///
    /// The add instruction's operands should be upgraded from Gpr32 to Fpr128
    /// (NEON SIMD) and an arrangement immediate should be appended.
    #[test]
    fn test_o2_vectorization_fires_on_loop() {
        let (mut func, add_id) = make_vectorizable_loop();

        // Precondition: add operands are Gpr32 before pipeline.
        let add_inst = func.inst(add_id);
        assert_eq!(add_inst.opcode, AArch64Opcode::AddRR);
        assert_eq!(add_inst.operands.len(), 3, "pre-pipeline: 3 operands");
        if let MachOperand::VReg(vreg) = &add_inst.operands[0] {
            assert_eq!(vreg.class, RegClass::Gpr32, "pre-pipeline: Gpr32");
        }

        let pipeline = OptimizationPipeline::new(OptLevel::O2);
        let stats = pipeline.run(&mut func);
        assert!(stats.changes > 0, "O2 pipeline should report changes");

        // Postcondition: the add instruction should have SIMD register class.
        // After vectorization, operands are Fpr128 and an arrangement
        // immediate is appended (4 operands: dst, src1, src2, arrangement).
        let add_inst = func.inst(add_id);
        let has_fpr128 = add_inst.operands.iter().any(|op| {
            if let MachOperand::VReg(vreg) = op {
                vreg.class == RegClass::Fpr128
            } else {
                false
            }
        });
        assert!(
            has_fpr128,
            "O2 pipeline should vectorize the add: operands should include Fpr128 registers"
        );

        // Arrangement immediate should be present (4 for i32 4S arrangement).
        let has_arrangement = add_inst.operands.iter().any(|op| {
            matches!(op, MachOperand::Imm(4))
        });
        assert!(
            has_arrangement,
            "O2 pipeline should append arrangement encoding (Imm(4) for 4S)"
        );
    }

    /// Verify that O3 pipeline fires vectorization on a simple i32 add loop.
    #[test]
    fn test_o3_vectorization_fires_on_loop() {
        let (mut func, add_id) = make_vectorizable_loop();

        let pipeline = OptimizationPipeline::new(OptLevel::O3);
        let stats = pipeline.run(&mut func);
        assert!(stats.changes > 0, "O3 pipeline should report changes");

        // Same postcondition: Fpr128 after vectorization.
        let add_inst = func.inst(add_id);
        let has_fpr128 = add_inst.operands.iter().any(|op| {
            if let MachOperand::VReg(vreg) = op {
                vreg.class == RegClass::Fpr128
            } else {
                false
            }
        });
        assert!(
            has_fpr128,
            "O3 pipeline should vectorize the add: operands should include Fpr128 registers"
        );
    }

    /// Verify that O2 pipeline vectorizes multiple arithmetic ops in a loop.
    #[test]
    fn test_o2_vectorization_multi_op() {
        let (mut func, op_ids) = make_multi_op_vectorizable_loop();

        // Precondition: all ops have Gpr32 operands.
        for &id in &op_ids {
            let inst = func.inst(id);
            if let MachOperand::VReg(vreg) = &inst.operands[0] {
                assert_eq!(vreg.class, RegClass::Gpr32, "pre-pipeline: Gpr32");
            }
        }

        let pipeline = OptimizationPipeline::new(OptLevel::O2);
        let stats = pipeline.run(&mut func);
        assert!(stats.changes > 0, "O2 pipeline should report changes");

        // Postcondition: all three ops should be vectorized to Fpr128.
        for &id in &op_ids {
            let inst = func.inst(id);
            let has_fpr128 = inst.operands.iter().any(|op| {
                if let MachOperand::VReg(vreg) = op {
                    vreg.class == RegClass::Fpr128
                } else {
                    false
                }
            });
            assert!(
                has_fpr128,
                "O2 pipeline should vectorize {:?} (inst {:?}): expected Fpr128",
                inst.opcode, id,
            );
        }
    }

    /// Verify that O1 does NOT fire vectorization (not in O1 pipeline).
    #[test]
    fn test_o1_no_vectorization() {
        let (mut func, add_id) = make_vectorizable_loop();

        let pipeline = OptimizationPipeline::new(OptLevel::O1);
        pipeline.run(&mut func);

        // O1 has only DCE + SROA + Peephole + Scheduler -- no vectorization.
        // The add instruction should NOT have Fpr128 operands.
        let add_inst = func.inst(add_id);
        let has_fpr128 = add_inst.operands.iter().any(|op| {
            if let MachOperand::VReg(vreg) = op {
                vreg.class == RegClass::Fpr128
            } else {
                false
            }
        });
        assert!(
            !has_fpr128,
            "O1 pipeline should NOT vectorize (vectorization is O2+)"
        );
    }

    /// Verify that O2 pass count includes vectorization pass.
    ///
    /// Forces declarative rewrite off so this test measures the scalar
    /// pipeline shape independent of Wave 3's default flip (#393).
    #[test]
    fn test_o2_includes_vectorize_pass() {
        with_declarative_rewrite_env(None, Some("1"), || {
            let pipeline = OptimizationPipeline::new(OptLevel::O2);
            // O2 (decl-rewrite off) = 20 passes: inline + proof-opts +
            // const-fold + copy-prop + cse + gvn + sroa + licm +
            // strength-reduce + loop-unroll + vectorize + peephole +
            // addr-mode + cmp-select + if-convert + cmp-branch-fusion +
            // tail-call + dce + cfg-simplify + pressure-aware-scheduler.
            assert_eq!(pipeline.pass_count(), 20);
        });
    }

    // =========================================================================
    // Declarative rewrite flag tests (#393)
    // =========================================================================
    //
    // Wave 3 flipped the default to ON. The kill switch
    // `LLVM2_DISABLE_DECLARATIVE_REWRITE=1` turns it off; the legacy
    // opt-in variable `LLVM2_ENABLE_DECLARATIVE_REWRITE=1` still forces
    // it on (transitional).
    //
    // Tests that toggle these env vars must run serially because env is
    // process-wide. A dedicated mutex serializes them; both variables
    // are saved + restored per test so they do not leak across cases.
    //
    // SAFETY: `std::env::set_var` and `std::env::remove_var` are marked
    // `unsafe` in edition 2024 because they race with other threads
    // reading env; Rust 2024 forces us to acknowledge that. Within a
    // single `#[test]` guarded by the mutex nothing else in this binary
    // should observe env, so the contract is upheld.

    use std::sync::Mutex;

    static DECL_REWRITE_ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Helper: run `f` with the two declarative-rewrite env vars forced
    /// to the given values, restoring the previous values on exit. The
    /// first element is `LLVM2_ENABLE_DECLARATIVE_REWRITE`, the second
    /// is `LLVM2_DISABLE_DECLARATIVE_REWRITE`.
    fn with_declarative_rewrite_env<R>(
        enable: Option<&str>,
        disable: Option<&str>,
        f: impl FnOnce() -> R,
    ) -> R {
        let _guard = DECL_REWRITE_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let enable_key = "LLVM2_ENABLE_DECLARATIVE_REWRITE";
        let disable_key = "LLVM2_DISABLE_DECLARATIVE_REWRITE";
        let prev_enable = std::env::var(enable_key).ok();
        let prev_disable = std::env::var(disable_key).ok();
        // SAFETY: see module-level note; the lock serializes env mutations.
        unsafe {
            match enable {
                Some(v) => std::env::set_var(enable_key, v),
                None => std::env::remove_var(enable_key),
            }
            match disable {
                Some(v) => std::env::set_var(disable_key, v),
                None => std::env::remove_var(disable_key),
            }
        }
        let result = f();
        // SAFETY: see above.
        unsafe {
            match prev_enable {
                Some(v) => std::env::set_var(enable_key, v),
                None => std::env::remove_var(enable_key),
            }
            match prev_disable {
                Some(v) => std::env::set_var(disable_key, v),
                None => std::env::remove_var(disable_key),
            }
        }
        result
    }

    #[test]
    fn declarative_rewrite_defaults_on_o1() {
        // Wave 3 flipped default-on. Neither env var set → pass is in
        // the pipeline: DCE, declarative-rewrite, SROA, Peephole, Scheduler = 5.
        with_declarative_rewrite_env(None, None, || {
            let pipeline = OptimizationPipeline::new(OptLevel::O1);
            assert_eq!(pipeline.pass_count(), 5);
        });
    }

    #[test]
    fn declarative_rewrite_kill_switch_o1() {
        // `LLVM2_DISABLE_DECLARATIVE_REWRITE=1` removes the declarative
        // pass: DCE, SROA, Peephole, Scheduler = 4.
        with_declarative_rewrite_env(None, Some("1"), || {
            let pipeline = OptimizationPipeline::new(OptLevel::O1);
            assert_eq!(pipeline.pass_count(), 4);
        });
    }

    #[test]
    fn declarative_rewrite_defaults_on_o2() {
        // Wave 3 default: 20 base passes (incl. SROA) + 1 declarative = 21.
        with_declarative_rewrite_env(None, None, || {
            let pipeline = OptimizationPipeline::new(OptLevel::O2);
            assert_eq!(pipeline.pass_count(), 21);
        });
    }

    #[test]
    fn declarative_rewrite_kill_switch_o2() {
        // Kill switch drops back to 20 passes (base O2 incl. SROA,
        // without the declarative-rewrite pass).
        with_declarative_rewrite_env(None, Some("1"), || {
            let pipeline = OptimizationPipeline::new(OptLevel::O2);
            assert_eq!(pipeline.pass_count(), 20);
        });
    }

    #[test]
    fn declarative_rewrite_enable_var_still_honored_o1() {
        // Transitional: explicit opt-in wins over the disable variable.
        with_declarative_rewrite_env(Some("1"), Some("1"), || {
            let pipeline = OptimizationPipeline::new(OptLevel::O1);
            // Enable beats disable → pass included (5 total: DCE +
            // declarative-rewrite + SROA + Peephole + Scheduler).
            assert_eq!(pipeline.pass_count(), 5);
        });
    }

    #[test]
    fn declarative_rewrite_respects_disable_list_o2() {
        // Per-pass bisect still works: LLVM2_DISABLE_PASSES=declrewrite
        // removes the pass even when default-on.
        with_declarative_rewrite_env(None, None, || {
            // SAFETY: serialized via DECL_REWRITE_ENV_LOCK inside the
            // helper; LLVM2_DISABLE_PASSES is also process-wide. Save +
            // restore around the test.
            let prev_disabled = std::env::var("LLVM2_DISABLE_PASSES").ok();
            unsafe { std::env::set_var("LLVM2_DISABLE_PASSES", "declrewrite"); }
            let pipeline = OptimizationPipeline::new(OptLevel::O2);
            // Declarative disabled via bisect list → 20 base passes
            // (incl. SROA) + 0 declarative = 20.
            assert_eq!(pipeline.pass_count(), 20);
            unsafe {
                match prev_disabled {
                    Some(v) => std::env::set_var("LLVM2_DISABLE_PASSES", v),
                    None => std::env::remove_var("LLVM2_DISABLE_PASSES"),
                }
            }
        });
    }

    #[test]
    fn declarative_rewrite_fires_at_o1_by_default() {
        // Default-on: the declarative pass should rewrite
        // `add x, y, #0` → `mov x, y` (migrated pattern #1). The
        // hand-written peephole then normalizes any residue; DCE drops
        // the unused result. No env vars required.
        with_declarative_rewrite_env(None, None, || {
            let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(0), vreg(1), imm(0)]);
            let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
            let mut func = make_func_with_insts(vec![add, ret]);

            let pipeline = OptimizationPipeline::new(OptLevel::O1);
            pipeline.run(&mut func);

            // DCE drops the unused result.
            let block = func.block(func.entry);
            assert_eq!(block.insts.len(), 1);
        });
    }
}
