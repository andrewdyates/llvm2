// llvm2-opt - Copy Propagation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Copy propagation pass for machine-level IR.
//!
//! When a `MovR dst, src` instruction is the only definition of `dst`,
//! all uses of `dst` are replaced with `src`, and the MOV becomes dead
//! (subsequently removed by DCE).
//!
//! # Algorithm
//!
//! 1. Scan all instructions to find `MovR` instructions where:
//!    - The destination is a VReg.
//!    - The source is a VReg.
//!    - The destination VReg has exactly one definition (this MOV).
//! 2. Build a replacement map: `dst → src`.
//! 3. Chase chains: if `src` itself maps to another register, follow
//!    the chain to the root source (avoiding cycles).
//! 4. Replace all occurrences of `dst` with the resolved source in all
//!    instruction operands.
//!
//! The MOV instructions themselves are NOT removed by this pass — that's
//! left to DCE, which will see that `dst` is no longer used.

use std::collections::HashMap;

use llvm2_ir::{AArch64Opcode, MachFunction, MachOperand, VReg};

use crate::pass_manager::MachinePass;

/// Copy propagation pass.
pub struct CopyPropagation;

impl MachinePass for CopyPropagation {
    fn name(&self) -> &str {
        "copy-prop"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        // Step 1: Count definitions per VReg.
        let def_counts = count_defs(func);

        // Step 2: Find copy instructions (MovR dst, src) where dst has
        // exactly one definition.
        let mut copy_map: HashMap<u32, VReg> = HashMap::new();

        for block_id in &func.block_order {
            let block = func.block(*block_id);
            for &inst_id in &block.insts {
                let inst = func.inst(inst_id);
                if inst.opcode != AArch64Opcode::MovR {
                    continue;
                }
                if inst.operands.len() < 2 {
                    continue;
                }

                let dst = match &inst.operands[0] {
                    MachOperand::VReg(v) => *v,
                    _ => continue,
                };
                let src = match &inst.operands[1] {
                    MachOperand::VReg(v) => *v,
                    _ => continue,
                };

                // Only propagate if dst has exactly one definition.
                if def_counts.get(&dst.id).copied().unwrap_or(0) == 1 {
                    copy_map.insert(dst.id, src);
                }
            }
        }

        if copy_map.is_empty() {
            return false;
        }

        // Step 3: Resolve chains. If v0 → v1 and v1 → v2, then v0 → v2.
        let resolved = resolve_chains(&copy_map);

        // Step 4: Replace operands throughout the function.
        let mut changed = false;
        for block_id in func.block_order.clone() {
            let block = func.block(block_id);
            for &inst_id in block.insts.clone().iter() {
                let inst = func.inst_mut(inst_id);

                // For MovR that defines a copied register, we skip replacing
                // the def operand (operand[0]) but DO replace use operands.
                // For other instructions, replace all VReg operands that are
                // in the resolved map.
                let use_start = if is_def_inst(inst) { 1 } else { 0 };

                for i in use_start..inst.operands.len() {
                    if let MachOperand::VReg(vreg) = &inst.operands[i]
                        && let Some(replacement) = resolved.get(&vreg.id) {
                            inst.operands[i] = MachOperand::VReg(*replacement);
                            changed = true;
                        }
                }
            }
        }

        changed
    }
}

/// Count how many times each VReg is defined across the function.
fn count_defs(func: &MachFunction) -> HashMap<u32, u32> {
    let mut counts: HashMap<u32, u32> = HashMap::new();

    for block_id in &func.block_order {
        let block = func.block(*block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            if is_def_inst(inst)
                && let Some(MachOperand::VReg(vreg)) = inst.operands.first() {
                    *counts.entry(vreg.id).or_insert(0) += 1;
                }
        }
    }

    counts
}

/// Returns true if this instruction defines a value (operand[0] is a def).
fn is_def_inst(inst: &llvm2_ir::MachInst) -> bool {
    use AArch64Opcode::*;
    match inst.opcode {
        CmpRR | CmpRI | Tst | Fcmp => false,
        StrRI | StpRI | StpPreIndex => false,
        B | BCond | Cbz | Cbnz | Br | Ret => false,
        Nop => false,
        Bl | Blr => false,
        _ => true,
    }
}

/// Resolve copy chains. If v0 → v1 → v2, then v0 → v2.
///
/// Detects cycles and stops chasing if a cycle is found (a register
/// maps to itself or forms a loop).
fn resolve_chains(copy_map: &HashMap<u32, VReg>) -> HashMap<u32, VReg> {
    let mut resolved: HashMap<u32, VReg> = HashMap::new();

    for (&dst_id, &src) in copy_map {
        let mut current = src;
        let mut depth = 0;
        const MAX_CHAIN: u32 = 64;

        while let Some(next) = copy_map.get(&current.id) {
            if next.id == dst_id || depth >= MAX_CHAIN {
                // Cycle detected or chain too long.
                break;
            }
            current = *next;
            depth += 1;
        }

        // Only add to resolved if the final source differs from the
        // original destination.
        if current.id != dst_id {
            resolved.insert(dst_id, current);
        }
    }

    resolved
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
            "test_cp".to_string(),
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
    fn test_simple_copy_prop() {
        // v1 = mov v0
        // v2 = add v1, #5  → v2 = add v0, #5
        let mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(1), vreg(0)]);
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(1), imm(5)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![mov, add, ret]);

        let mut cp = CopyPropagation;
        assert!(cp.run(&mut func));

        // add should now use v0 instead of v1
        let add_inst = func.inst(InstId(1));
        assert_eq!(add_inst.operands[1], vreg(0));
    }

    #[test]
    fn test_chain_copy_prop() {
        // v1 = mov v0
        // v2 = mov v1
        // v3 = add v2, #5  → v3 = add v0, #5
        let m1 = MachInst::new(AArch64Opcode::MovR, vec![vreg(1), vreg(0)]);
        let m2 = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(1)]);
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(3), vreg(2), imm(5)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![m1, m2, add, ret]);

        let mut cp = CopyPropagation;
        assert!(cp.run(&mut func));

        let add_inst = func.inst(InstId(2));
        assert_eq!(add_inst.operands[1], vreg(0));
    }

    #[test]
    fn test_no_prop_multiple_defs() {
        // v1 = mov v0
        // v1 = add v1, #1  (second def of v1 → don't propagate)
        // v2 = add v1, #5
        let m1 = MachInst::new(AArch64Opcode::MovR, vec![vreg(1), vreg(0)]);
        let a1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(1), imm(1)]);
        let a2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(1), imm(5)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![m1, a1, a2, ret]);

        let mut cp = CopyPropagation;
        assert!(!cp.run(&mut func));
    }

    #[test]
    fn test_no_change_no_copies() {
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(0), vreg(1), imm(5)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let mut cp = CopyPropagation;
        assert!(!cp.run(&mut func));
    }
}
