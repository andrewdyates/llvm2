// llvm2-regalloc/remat.rs - Rematerialization for the register allocator
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Rematerialization: recompute cheap values instead of loading from spill slots.
//!
//! For values defined by instructions that are cheap to recompute (constants,
//! address computations, simple ALU ops with immediates), we can clone the
//! defining instruction at each use site instead of inserting a spill load.
//! This eliminates the memory traffic of the spill and often produces better
//! code.
//!
//! Reference: LLVM `InlineSpiller.cpp` — rematerialization during spilling.

use crate::linear_scan::SpillInfo;
use crate::machine_types::{
    InstId, MachFunction, MachInst, MachOperand, VReg,
};
use crate::spill::PSEUDO_SPILL_LOAD;

/// Cost classification for rematerialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RematCost {
    /// Free to rematerialize: instruction uses only immediates.
    /// Examples: `MOV Xd, #imm`, `FMOV Dd, #fimm`.
    Free,
    /// Cheap to rematerialize: one VReg input plus an immediate.
    /// Examples: `ADD Xd, Xn, #imm`, `SUB Xd, Xn, #imm`.
    Cheap,
    /// Too expensive to rematerialize: multiple register inputs,
    /// memory operations, calls, or side effects.
    Expensive,
}

/// A candidate for rematerialization.
#[derive(Debug, Clone)]
pub struct RematCandidate {
    /// The spilled virtual register.
    pub vreg: VReg,
    /// The instruction that defines this VReg.
    pub defining_inst_id: InstId,
    /// The cost classification.
    pub cost: RematCost,
}

/// Classify the rematerialization cost of an instruction.
///
/// Rules:
/// - Instructions with IS_CALL, READS_MEMORY, WRITES_MEMORY, or
///   HAS_SIDE_EFFECTS flags are always Expensive.
/// - Instructions where all uses are Imm/FImm are Free.
/// - Instructions with exactly one VReg use and at least one Imm/FImm
///   use are Cheap.
/// - Everything else is Expensive.
pub fn classify_remat_cost(inst: &MachInst) -> RematCost {
    let flags = inst.flags;

    // Side-effectful or memory instructions cannot be rematerialized.
    if flags.is_call()
        || flags.reads_memory()
        || flags.writes_memory()
        || flags.has_side_effects()
    {
        return RematCost::Expensive;
    }

    // Phi and branch instructions are not rematerializable.
    if flags.is_phi() || flags.is_branch() || flags.is_terminator() || flags.is_return() {
        return RematCost::Expensive;
    }

    let mut vreg_count = 0u32;
    let mut imm_count = 0u32;

    for op in &inst.uses {
        match op {
            MachOperand::VReg(_) => vreg_count += 1,
            MachOperand::PReg(_) => vreg_count += 1, // physical regs count as register uses
            MachOperand::Imm(_) | MachOperand::FImm(_) => imm_count += 1,
            MachOperand::Block(_) | MachOperand::StackSlot(_) => {
                return RematCost::Expensive;
            }
        }
    }

    if vreg_count == 0 && imm_count > 0 {
        RematCost::Free
    } else if vreg_count == 1 && imm_count >= 1 {
        RematCost::Cheap
    } else {
        RematCost::Expensive
    }
}

/// Find rematerialization candidates among spilled VRegs.
///
/// For each spilled VReg, looks up its single defining instruction and
/// checks if it's cheap enough to rematerialize. Returns candidates
/// that are Free or Cheap.
pub fn find_remat_candidates(
    func: &MachFunction,
    spilled_vregs: &[VReg],
) -> Vec<RematCandidate> {
    let mut candidates = Vec::new();

    for &vreg in spilled_vregs {
        // Find the defining instruction for this VReg.
        if let Some(def_inst_id) = find_defining_inst(func, vreg) {
            let inst = &func.insts[def_inst_id.0 as usize];
            let cost = classify_remat_cost(inst);

            match cost {
                RematCost::Free | RematCost::Cheap => {
                    candidates.push(RematCandidate {
                        vreg,
                        defining_inst_id: def_inst_id,
                        cost,
                    });
                }
                RematCost::Expensive => {}
            }
        }
    }

    candidates
}

/// Apply rematerialization: replace spill loads with cloned defining
/// instructions.
///
/// For each remat candidate:
/// 1. Find all PSEUDO_SPILL_LOAD instructions that load this VReg.
/// 2. Replace each with a clone of the defining instruction.
/// 3. Remove the candidate from `spill_infos` (no spill slot needed).
///
/// Returns the number of rematerializations performed.
pub fn apply_rematerialization(
    func: &mut MachFunction,
    candidates: &[RematCandidate],
    spill_infos: &mut Vec<SpillInfo>,
) -> u32 {
    let mut remat_count = 0u32;
    let remat_vregs: std::collections::HashSet<u32> =
        candidates.iter().map(|c| c.vreg.id).collect();

    // Clone the defining instructions upfront to avoid borrow conflicts.
    let def_inst_clones: std::collections::HashMap<u32, MachInst> = candidates
        .iter()
        .map(|c| (c.vreg.id, func.insts[c.defining_inst_id.0 as usize].clone()))
        .collect();

    // Phase 1: Scan for spill loads to replace. Collect the plan.
    // Each entry: (block_idx, inst_position_in_block, vreg_id).
    let mut replacements: Vec<(usize, usize, u32)> = Vec::new();

    for (block_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst_id) in block.insts.iter().enumerate() {
            let inst = &func.insts[inst_id.0 as usize];
            if inst.opcode != PSEUDO_SPILL_LOAD {
                continue;
            }

            // Check if the loaded VReg is a remat candidate.
            if let Some(loaded_vreg) = inst.defs.first().and_then(|op| op.as_vreg()) {
                if remat_vregs.contains(&loaded_vreg.id) {
                    replacements.push((block_idx, pos, loaded_vreg.id));
                }
            }
        }
    }

    // Phase 2: Apply replacements in reverse order to preserve positions.
    for &(block_idx, pos, vreg_id) in replacements.iter().rev() {
        if let Some(def_inst) = def_inst_clones.get(&vreg_id) {
            let remat_inst = def_inst.clone();
            let new_inst_id = InstId(func.insts.len() as u32);
            func.insts.push(remat_inst);

            // Replace the spill load with the rematerialized instruction.
            func.blocks[block_idx].insts[pos] = new_inst_id;
            remat_count += 1;
        }
    }

    // Phase 3: Remove spill infos for rematerialized VRegs.
    spill_infos.retain(|si| !remat_vregs.contains(&si.vreg.id));

    remat_count
}

/// Find the defining instruction for a VReg.
///
/// Scans all blocks for an instruction that defines the given VReg.
/// Returns the first definition found (SSA property: exactly one def).
fn find_defining_inst(func: &MachFunction, vreg: VReg) -> Option<InstId> {
    for block in &func.blocks {
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.0 as usize];
            for def_op in &inst.defs {
                if let Some(def_vreg) = def_op.as_vreg() {
                    if def_vreg.id == vreg.id {
                        return Some(inst_id);
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine_types::{
        BlockId, InstFlags, MachBlock, MachFunction, MachInst, MachOperand,
        RegClass, StackSlotId, VReg,
    };
    use std::collections::HashMap;

    fn vreg(id: u32) -> VReg {
        VReg {
            id,
            class: RegClass::Gpr64,
        }
    }

    fn make_func(insts: Vec<MachInst>) -> MachFunction {
        let inst_ids: Vec<InstId> = (0..insts.len()).map(|i| InstId(i as u32)).collect();
        MachFunction {
            name: "test".into(),
            insts,
            blocks: vec![MachBlock {
                insts: inst_ids,
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: 32,
            next_stack_slot: 0,
            stack_slots: HashMap::new(),
        }
    }

    #[test]
    fn test_classify_free_remat() {
        // MOV Xd, #42 — only immediate uses.
        let inst = MachInst {
            opcode: 1,
            defs: vec![MachOperand::VReg(vreg(0))],
            uses: vec![MachOperand::Imm(42)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        };
        assert_eq!(classify_remat_cost(&inst), RematCost::Free);
    }

    #[test]
    fn test_classify_cheap_remat() {
        // ADD Xd, Xn, #10 — one VReg + one immediate.
        let inst = MachInst {
            opcode: 2,
            defs: vec![MachOperand::VReg(vreg(0))],
            uses: vec![MachOperand::VReg(vreg(1)), MachOperand::Imm(10)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        };
        assert_eq!(classify_remat_cost(&inst), RematCost::Cheap);
    }

    #[test]
    fn test_classify_expensive_memory() {
        // LDR Xd, [Xn] — reads memory.
        let inst = MachInst {
            opcode: 3,
            defs: vec![MachOperand::VReg(vreg(0))],
            uses: vec![MachOperand::VReg(vreg(1))],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::READS_MEMORY,
        };
        assert_eq!(classify_remat_cost(&inst), RematCost::Expensive);
    }

    #[test]
    fn test_classify_expensive_call() {
        let inst = MachInst {
            opcode: 4,
            defs: vec![MachOperand::VReg(vreg(0))],
            uses: vec![],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::IS_CALL,
        };
        assert_eq!(classify_remat_cost(&inst), RematCost::Expensive);
    }

    #[test]
    fn test_classify_expensive_multi_reg() {
        // ADD Xd, Xn, Xm — two VReg uses, no immediate.
        let inst = MachInst {
            opcode: 5,
            defs: vec![MachOperand::VReg(vreg(0))],
            uses: vec![MachOperand::VReg(vreg(1)), MachOperand::VReg(vreg(2))],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        };
        assert_eq!(classify_remat_cost(&inst), RematCost::Expensive);
    }

    #[test]
    fn test_find_remat_candidates() {
        let func = make_func(vec![
            // inst 0: MOV v0, #42 (Free remat)
            MachInst {
                opcode: 1,
                defs: vec![MachOperand::VReg(vreg(0))],
                uses: vec![MachOperand::Imm(42)],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::default(),
            },
            // inst 1: LDR v1, [v2] (Expensive)
            MachInst {
                opcode: 2,
                defs: vec![MachOperand::VReg(vreg(1))],
                uses: vec![MachOperand::VReg(vreg(2))],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::READS_MEMORY,
            },
        ]);

        let spilled = vec![vreg(0), vreg(1)];
        let candidates = find_remat_candidates(&func, &spilled);

        // Only v0 should be a candidate (Free remat).
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].vreg.id, 0);
        assert_eq!(candidates[0].cost, RematCost::Free);
    }

    #[test]
    fn test_apply_rematerialization() {
        let mut func = make_func(vec![
            // inst 0: MOV v0, #42 (defining instruction)
            MachInst {
                opcode: 1,
                defs: vec![MachOperand::VReg(vreg(0))],
                uses: vec![MachOperand::Imm(42)],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::default(),
            },
            // inst 1: SPILL_LOAD v0 from stack
            MachInst {
                opcode: PSEUDO_SPILL_LOAD,
                defs: vec![MachOperand::VReg(vreg(0))],
                uses: vec![MachOperand::StackSlot(StackSlotId(0))],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::READS_MEMORY,
            },
            // inst 2: use v0
            MachInst {
                opcode: 3,
                defs: vec![],
                uses: vec![MachOperand::VReg(vreg(0))],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::default(),
            },
        ]);

        let candidates = vec![RematCandidate {
            vreg: vreg(0),
            defining_inst_id: InstId(0),
            cost: RematCost::Free,
        }];

        let mut spill_infos = vec![SpillInfo {
            vreg: vreg(0),
            slot: StackSlotId(0),
        }];

        let count = apply_rematerialization(&mut func, &candidates, &mut spill_infos);

        assert_eq!(count, 1);
        assert!(spill_infos.is_empty()); // Spill info removed.

        // The spill load (inst position 1 in block) should be replaced.
        let replaced_inst_id = func.blocks[0].insts[1];
        let replaced_inst = &func.insts[replaced_inst_id.0 as usize];
        // Should now be a clone of the defining instruction (opcode 1, MOV).
        assert_eq!(replaced_inst.opcode, 1);
        assert_eq!(replaced_inst.uses.len(), 1);
        assert_eq!(replaced_inst.uses[0], MachOperand::Imm(42));
    }
}
