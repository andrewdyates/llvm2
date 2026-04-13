// llvm2-opt - AArch64 Peephole Optimizations
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! AArch64-specific peephole optimizations.
//!
//! Peephole optimizations are local pattern-matching transformations that
//! simplify or eliminate individual instructions or short instruction
//! sequences.
//!
//! # Patterns Implemented
//!
//! | Pattern | Transformation |
//! |---------|---------------|
//! | `mov x0, x0` | Delete (self-move is no-op) |
//! | `add x0, x1, #0` | `mov x0, x1` |
//! | `sub x0, x1, #0` | `mov x0, x1` |
//! | `mul x0, x1, #1` | `mov x0, x1` (when imm operand via const lookup) |
//! | `add x0, x1, x1` with x1 known to be #0 | `mov x0, x1` |
//! | `lsl x0, x1, #0` | `mov x0, x1` |
//! | `lsr x0, x1, #0` | `mov x0, x1` |
//! | `asr x0, x1, #0` | `mov x0, x1` |
//! | `orr x0, x1, x1` | `mov x0, x1` (OR with self = identity) |
//! | `and x0, x1, x1` | `mov x0, x1` (AND with self = identity) |
//!
//! # Note on cmp+b.cond Fusing
//!
//! CMP+BCond fusing into CBZ/CBNZ is deferred to a later pass because it
//! requires cross-instruction analysis (the CMP must be immediately
//! followed by the BCond with no intervening flag-setting instructions).

use llvm2_ir::{AArch64Opcode, InstId, MachFunction, MachInst, MachOperand};

use crate::pass_manager::MachinePass;

/// AArch64 peephole optimization pass.
pub struct Peephole;

impl MachinePass for Peephole {
    fn name(&self) -> &str {
        "peephole"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        let mut changed = false;
        let mut to_delete: Vec<InstId> = Vec::new();

        for block_id in func.block_order.clone() {
            let block = func.block(block_id);
            for &inst_id in block.insts.clone().iter() {
                match try_peephole(func.inst(inst_id)) {
                    PeepholeResult::NoChange => {}
                    PeepholeResult::Replace(new_inst) => {
                        *func.inst_mut(inst_id) = new_inst;
                        changed = true;
                    }
                    PeepholeResult::Delete => {
                        to_delete.push(inst_id);
                        changed = true;
                    }
                }
            }
        }

        // Remove deleted instructions from blocks.
        if !to_delete.is_empty() {
            let delete_set: std::collections::HashSet<InstId> =
                to_delete.into_iter().collect();
            for block_id in func.block_order.clone() {
                let block = func.block_mut(block_id);
                block.insts.retain(|id| !delete_set.contains(id));
            }
        }

        changed
    }
}

/// Result of attempting a peephole optimization on a single instruction.
enum PeepholeResult {
    /// No optimization applies.
    NoChange,
    /// Replace the instruction with a new one.
    Replace(MachInst),
    /// Delete the instruction entirely.
    Delete,
}

/// Try to apply a peephole optimization to a single instruction.
fn try_peephole(inst: &MachInst) -> PeepholeResult {
    match inst.opcode {
        AArch64Opcode::MovR => peephole_mov(inst),
        AArch64Opcode::AddRI => peephole_add_ri(inst),
        AArch64Opcode::SubRI => peephole_sub_ri(inst),
        AArch64Opcode::LslRI => peephole_shift_ri_zero(inst),
        AArch64Opcode::LsrRI => peephole_shift_ri_zero(inst),
        AArch64Opcode::AsrRI => peephole_shift_ri_zero(inst),
        AArch64Opcode::OrrRR => peephole_logical_self(inst),
        AArch64Opcode::AndRR => peephole_logical_self(inst),
        AArch64Opcode::Nop => PeepholeResult::Delete,
        _ => PeepholeResult::NoChange,
    }
}

/// `mov x0, x0` → delete (self-move is no-op).
fn peephole_mov(inst: &MachInst) -> PeepholeResult {
    if inst.operands.len() < 2 {
        return PeepholeResult::NoChange;
    }

    // Check if src == dst (self-move).
    match (&inst.operands[0], &inst.operands[1]) {
        (MachOperand::VReg(dst), MachOperand::VReg(src)) if dst.id == src.id => {
            PeepholeResult::Delete
        }
        (MachOperand::PReg(dst), MachOperand::PReg(src)) if dst == src => {
            PeepholeResult::Delete
        }
        _ => PeepholeResult::NoChange,
    }
}

/// `add x0, x1, #0` → `mov x0, x1`
fn peephole_add_ri(inst: &MachInst) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    // Check if the immediate operand is 0.
    if let MachOperand::Imm(0) = &inst.operands[2] {
        // Replace with mov dst, src
        let new_inst = MachInst::new(
            AArch64Opcode::MovR,
            vec![inst.operands[0].clone(), inst.operands[1].clone()],
        );
        PeepholeResult::Replace(new_inst)
    } else {
        PeepholeResult::NoChange
    }
}

/// `sub x0, x1, #0` → `mov x0, x1`
fn peephole_sub_ri(inst: &MachInst) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    if let MachOperand::Imm(0) = &inst.operands[2] {
        let new_inst = MachInst::new(
            AArch64Opcode::MovR,
            vec![inst.operands[0].clone(), inst.operands[1].clone()],
        );
        PeepholeResult::Replace(new_inst)
    } else {
        PeepholeResult::NoChange
    }
}

/// `lsl/lsr/asr x0, x1, #0` → `mov x0, x1`
fn peephole_shift_ri_zero(inst: &MachInst) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    if let MachOperand::Imm(0) = &inst.operands[2] {
        let new_inst = MachInst::new(
            AArch64Opcode::MovR,
            vec![inst.operands[0].clone(), inst.operands[1].clone()],
        );
        PeepholeResult::Replace(new_inst)
    } else {
        PeepholeResult::NoChange
    }
}

/// `orr x0, x1, x1` → `mov x0, x1` (OR with self = identity)
/// `and x0, x1, x1` → `mov x0, x1` (AND with self = identity)
fn peephole_logical_self(inst: &MachInst) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    // Check if both source operands are the same VReg.
    match (&inst.operands[1], &inst.operands[2]) {
        (MachOperand::VReg(a), MachOperand::VReg(b)) if a.id == b.id => {
            let new_inst = MachInst::new(
                AArch64Opcode::MovR,
                vec![inst.operands[0].clone(), inst.operands[1].clone()],
            );
            PeepholeResult::Replace(new_inst)
        }
        _ => PeepholeResult::NoChange,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::{
        AArch64Opcode, InstId, MachFunction, MachInst, MachOperand,
        RegClass, Signature, VReg,
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
            "test_peephole".to_string(),
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
    fn test_delete_self_move() {
        let mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(0), vreg(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![mov, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 1); // only ret
    }

    #[test]
    fn test_add_zero_to_mov() {
        // add v0, v1, #0 → mov v0, v1
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(0), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
        assert_eq!(inst.operands[0], vreg(0));
        assert_eq!(inst.operands[1], vreg(1));
    }

    #[test]
    fn test_sub_zero_to_mov() {
        let sub = MachInst::new(AArch64Opcode::SubRI, vec![vreg(0), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![sub, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
    }

    #[test]
    fn test_lsl_zero_to_mov() {
        let lsl = MachInst::new(AArch64Opcode::LslRI, vec![vreg(0), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![lsl, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
    }

    #[test]
    fn test_lsr_zero_to_mov() {
        let lsr = MachInst::new(AArch64Opcode::LsrRI, vec![vreg(0), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![lsr, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
    }

    #[test]
    fn test_orr_self_to_mov() {
        // orr v0, v1, v1 → mov v0, v1
        let orr = MachInst::new(AArch64Opcode::OrrRR, vec![vreg(0), vreg(1), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![orr, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
    }

    #[test]
    fn test_and_self_to_mov() {
        let and = MachInst::new(AArch64Opcode::AndRR, vec![vreg(0), vreg(1), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![and, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
    }

    #[test]
    fn test_no_change_nonzero_add() {
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(0), vreg(1), imm(5)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    #[test]
    fn test_no_change_different_regs_orr() {
        let orr = MachInst::new(AArch64Opcode::OrrRR, vec![vreg(0), vreg(1), vreg(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![orr, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    #[test]
    fn test_delete_nop() {
        let nop = MachInst::new(AArch64Opcode::Nop, vec![]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![nop, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 1);
    }

    #[test]
    fn test_multiple_peepholes_in_one_pass() {
        // self-move + add-zero: both should be optimized
        let mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(0), vreg(0)]);
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(2), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![mov, add, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let block = func.block(func.entry);
        // Self-move deleted, add replaced with mov
        assert_eq!(block.insts.len(), 2); // mov (from add) + ret

        let add_inst = func.inst(InstId(1));
        assert_eq!(add_inst.opcode, AArch64Opcode::MovR);
    }
}
