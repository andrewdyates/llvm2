// llvm2-opt - Dead Code Elimination
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Dead Code Elimination (DCE) for machine-level IR.
//!
//! Removes instructions whose definitions are never used, provided they
//! have no side effects. This is a simple backward-scan DCE that operates
//! on each basic block independently.
//!
//! # Algorithm
//!
//! For each block (in reverse instruction order):
//! 1. Build a set of "used" virtual registers by scanning all operands.
//! 2. Walk instructions backward. If an instruction:
//!    - Defines a VReg that is NOT in the used set, AND
//!    - Has no side effects (not a call, branch, store, or flag-setting op)
//!    Then mark it for removal.
//! 3. Remove marked instructions from the block.
//!
//! # Side-Effect Barriers
//!
//! Instructions with any of these properties are NEVER eliminated:
//! - `IS_CALL` — may have arbitrary side effects
//! - `IS_BRANCH` / `IS_TERMINATOR` — control flow
//! - `IS_RETURN` — control flow
//! - `HAS_SIDE_EFFECTS` — flag-setting, stores, etc.
//! - `WRITES_MEMORY` — stores
//! - `READS_MEMORY` — loads (conservative; a more aggressive DCE could
//!   remove loads whose result is unused, but we keep it simple)
//!
//! Reference: LLVM `DeadMachineInstrElim.cpp`

use std::collections::HashSet;

use llvm2_ir::{InstFlags, InstId, MachFunction, MachOperand, AArch64Opcode};

use crate::pass_manager::MachinePass;

/// Dead Code Elimination pass.
pub struct DeadCodeElimination;

impl MachinePass for DeadCodeElimination {
    fn name(&self) -> &str {
        "dce"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        let mut changed = false;

        // Step 1: Collect all VReg IDs that are *used* (appear as source operands)
        // across the entire function.
        let used_vregs = collect_used_vregs(func);

        // Step 2: Find instructions to remove. An instruction is dead if:
        //   (a) it defines a VReg not in the used set, AND
        //   (b) it has no side effects.
        let mut dead_insts: HashSet<InstId> = HashSet::new();

        for block_id in func.block_order.clone() {
            let block = func.block(block_id);
            for &inst_id in &block.insts {
                let inst = func.inst(inst_id);

                // Never remove instructions with side effects.
                if has_side_effects(inst) {
                    continue;
                }

                // Check if any defined VReg is used.
                let def_vreg = get_def_vreg(inst);
                match def_vreg {
                    Some(vreg_id) => {
                        if !used_vregs.contains(&vreg_id) {
                            dead_insts.insert(inst_id);
                            changed = true;
                        }
                    }
                    // Instructions with no def (and no side effects) that are
                    // not branches/terminators are dead. But be conservative:
                    // only Nop qualifies here.
                    None => {
                        if inst.opcode == AArch64Opcode::Nop {
                            dead_insts.insert(inst_id);
                            changed = true;
                        }
                    }
                }
            }
        }

        // Step 3: Remove dead instructions from blocks.
        if changed {
            for block_id in func.block_order.clone() {
                let block = func.block_mut(block_id);
                block.insts.retain(|id| !dead_insts.contains(id));
            }
        }

        changed
    }
}

/// Collect all VReg IDs that appear as source (non-def) operands.
///
/// Convention: operand[0] is the destination (def) for instructions that
/// produce a value. Operands[1..] are sources (uses). For instructions
/// that don't produce a value (CMP, STR, branches), all operands are uses.
fn collect_used_vregs(func: &MachFunction) -> HashSet<u32> {
    let mut used = HashSet::new();

    for block_id in &func.block_order {
        let block = func.block(*block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);

            // Determine which operands are "uses" vs "defs".
            let use_start = if produces_value(inst) { 1 } else { 0 };

            for operand in &inst.operands[use_start..] {
                if let MachOperand::VReg(vreg) = operand {
                    used.insert(vreg.id);
                }
            }

            // Phi nodes: all operands except the first (def) are uses.
            // Already handled by use_start = 1 for Phi.
        }
    }

    used
}

/// Returns the VReg ID defined by this instruction, if any.
///
/// Convention: for instructions that produce a value, operand[0] is the
/// destination register (def).
fn get_def_vreg(inst: &llvm2_ir::MachInst) -> Option<u32> {
    if !produces_value(inst) {
        return None;
    }
    if let Some(MachOperand::VReg(vreg)) = inst.operands.first() {
        Some(vreg.id)
    } else {
        None
    }
}

/// Returns true if this instruction produces a value (has a def operand).
///
/// Instructions that don't produce values: CMP, TST, STR, STP, branches,
/// returns, NOP, and calls that return void (simplified: we treat all
/// calls as having side effects anyway).
fn produces_value(inst: &llvm2_ir::MachInst) -> bool {
    use AArch64Opcode::*;
    match inst.opcode {
        // Compare/test: set flags, no register def
        CmpRR | CmpRI | Tst | Fcmp => false,
        // Stores: write to memory, no register def
        StrRI | StpRI | StpPreIndex => false,
        // Branches and returns: control flow, no register def
        B | BCond | Cbz | Cbnz | Tbz | Tbnz | Br | Ret => false,
        // Trap pseudo-instructions: control flow, no register def
        TrapOverflow | TrapBoundsCheck | TrapNull | TrapDivZero | TrapShiftRange => false,
        // Reference counting: side effects, no register def
        Retain | Release => false,
        // Nop: no def
        Nop => false,
        // Calls: they DO produce a result (in X0 typically), but that's
        // handled via implicit defs. For our simple model, calls have
        // side effects and won't be DCE'd anyway.
        Bl | Blr => false,
        // Everything else produces a value in operand[0]
        // (including AddsRR/SubsRR which produce a result plus set flags)
        _ => true,
    }
}

/// Returns true if this instruction has side effects that prevent removal.
fn has_side_effects(inst: &llvm2_ir::MachInst) -> bool {
    let flags = inst.flags;

    // Any of these flags means the instruction cannot be removed.
    flags.contains(InstFlags::IS_CALL)
        || flags.contains(InstFlags::IS_BRANCH)
        || flags.contains(InstFlags::IS_TERMINATOR)
        || flags.contains(InstFlags::IS_RETURN)
        || flags.contains(InstFlags::HAS_SIDE_EFFECTS)
        || flags.contains(InstFlags::WRITES_MEMORY)
        || flags.contains(InstFlags::READS_MEMORY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::{
        AArch64Opcode, MachFunction, MachInst, MachOperand, Signature, VReg,
        RegClass,
    };
    use crate::pass_manager::MachinePass;

    fn vreg(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
    }

    fn imm(val: i64) -> MachOperand {
        MachOperand::Imm(val)
    }

    fn make_func_with_insts(insts: Vec<MachInst>) -> MachFunction {
        let mut func = MachFunction::new(
            "test_dce".to_string(),
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
    fn test_dce_removes_unused_add() {
        // v0 = add v1, v2  -- v0 is never used → dead
        // ret
        let add = MachInst::new(AArch64Opcode::AddRR, vec![vreg(0), vreg(1), vreg(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let mut dce = DeadCodeElimination;
        assert!(dce.run(&mut func));

        // Only ret should remain
        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 1);
    }

    #[test]
    fn test_dce_preserves_used_add() {
        // v0 = add v1, #5
        // v2 = sub v0, #1  -- uses v0, so add is live
        // ret
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(0), vreg(1), imm(5)]);
        let sub = MachInst::new(AArch64Opcode::SubRI, vec![vreg(2), vreg(0), imm(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, sub, ret]);

        let mut dce = DeadCodeElimination;
        // v2 is also unused, but v0 is used by sub.
        // First pass: removes sub (v2 unused). v0 becomes unused.
        assert!(dce.run(&mut func));

        let block = func.block(func.entry);
        // sub removed (v2 unused), but add kept (v0 used by sub was evaluated
        // globally — hmm, actually v0 IS used by sub which is still present
        // when we scan). Wait: we collect used vregs first, then find dead.
        // sub uses v0, so v0 is in used set. sub defines v2 which is unused.
        // So sub is dead, add is live.
        assert_eq!(block.insts.len(), 2); // add + ret
    }

    #[test]
    fn test_dce_preserves_stores() {
        // str v0, [sp, #8] — has WRITES_MEMORY, never removed
        let store = MachInst::new(AArch64Opcode::StrRI, vec![vreg(0), imm(8)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![store, ret]);

        let mut dce = DeadCodeElimination;
        assert!(!dce.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2);
    }

    #[test]
    fn test_dce_preserves_branches() {
        let branch = MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(llvm2_ir::BlockId(1))]);
        let mut func = make_func_with_insts(vec![branch]);

        let mut dce = DeadCodeElimination;
        assert!(!dce.run(&mut func));
    }

    #[test]
    fn test_dce_removes_nop() {
        let nop = MachInst::new(AArch64Opcode::Nop, vec![]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![nop, ret]);

        let mut dce = DeadCodeElimination;
        assert!(dce.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 1); // only ret
    }

    #[test]
    fn test_dce_preserves_cmp() {
        // cmp v0, v1 — sets flags, has HAS_SIDE_EFFECTS
        let cmp = MachInst::new(AArch64Opcode::CmpRR, vec![vreg(0), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![cmp, ret]);

        let mut dce = DeadCodeElimination;
        assert!(!dce.run(&mut func));
    }

    #[test]
    fn test_dce_idempotent() {
        let add = MachInst::new(AArch64Opcode::AddRR, vec![vreg(0), vreg(1), vreg(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let mut dce = DeadCodeElimination;
        assert!(dce.run(&mut func)); // First pass removes dead add
        assert!(!dce.run(&mut func)); // Second pass: nothing to do
    }
}
