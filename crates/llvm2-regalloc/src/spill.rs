// llvm2-regalloc/spill.rs - Spill code generation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Spill code generation: inserts loads/stores for spilled virtual registers.
//!
//! After register allocation determines which VRegs must be spilled, this
//! module rewrites the machine function to insert:
//! - Stores after each definition of a spilled VReg.
//! - Loads before each use of a spilled VReg.
//!
//! Each spilled VReg gets a dedicated stack slot. Future work will add
//! spill-slot reuse (share slots for non-overlapping spilled intervals)
//! and rematerialization (recompute cheap values instead of loading).
//!
//! Reference: LLVM's `InlineSpiller` and `SpillPlacement`.

use crate::linear_scan::SpillInfo;
use crate::machine_types::{
    InstFlags, InstId, MachFunction, MachInst, MachOperand, PReg, RegClass, StackSlotId, VReg,
};
use std::collections::HashMap;

/// Pseudo-opcodes for spill instructions.
///
/// These will be lowered to real load/store instructions by the target-specific
/// backend (e.g., `STR Xn, [SP, #offset]` on AArch64).
pub const PSEUDO_SPILL_STORE: u16 = 0xFFF0;
pub const PSEUDO_SPILL_LOAD: u16 = 0xFFF1;

/// Insert spill code for all spilled VRegs in the allocation result.
///
/// For each spill:
/// 1. Allocate a stack slot.
/// 2. After each def of the spilled VReg, insert a store to the stack slot.
/// 3. Before each use of the spilled VReg, insert a load from the stack slot.
///
/// The spilled VReg operands are rewritten to use a fresh temporary VReg
/// that is immediately loaded/stored, so it has a very short live range
/// (just one instruction). This temporary can be trivially allocated in
/// a second allocation pass or assigned a dedicated spill register.
pub fn insert_spill_code(
    func: &mut MachFunction,
    spilled_vregs: &[VReg],
    _allocation: &HashMap<VReg, PReg>,
) -> Vec<SpillInfo> {
    let mut spill_infos = Vec::new();
    let mut vreg_to_slot: HashMap<u32, StackSlotId> = HashMap::new();

    // Allocate stack slots for each spilled VReg.
    for &vreg in spilled_vregs {
        let size = reg_class_size(vreg.class);
        let slot = func.alloc_stack_slot(size, size); // align = size
        vreg_to_slot.insert(vreg.id, slot);
        spill_infos.push(SpillInfo { vreg, slot });
    }

    // Phase 1: Scan all blocks and collect the spill insertions needed.
    // We must read func.insts immutably first, then mutate afterward.
    // Each entry: (original_inst_id, loads_before, stores_after).
    let num_blocks = func.blocks.len();
    let mut block_plans: Vec<Vec<(InstId, Vec<MachInst>, Vec<MachInst>)>> =
        Vec::with_capacity(num_blocks);

    for block in &func.blocks {
        let mut plan = Vec::new();

        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.0 as usize];

            // Collect loads needed before this instruction (for spilled uses).
            let mut loads = Vec::new();
            for op in &inst.uses {
                if let Some(vreg) = op.as_vreg() {
                    if let Some(&slot) = vreg_to_slot.get(&vreg.id) {
                        loads.push(make_spill_load(vreg, slot));
                    }
                }
            }

            // Collect stores needed after this instruction (for spilled defs).
            let mut stores = Vec::new();
            for op in &inst.defs {
                if let Some(vreg) = op.as_vreg() {
                    if let Some(&slot) = vreg_to_slot.get(&vreg.id) {
                        stores.push(make_spill_store(vreg, slot));
                    }
                }
            }

            plan.push((inst_id, loads, stores));
        }

        block_plans.push(plan);
    }

    // Phase 2: Apply the plans — add new instructions and rebuild block inst lists.
    for (block_idx, plan) in block_plans.into_iter().enumerate() {
        let mut new_insts = Vec::new();

        for (inst_id, loads, stores) in plan {
            // Insert loads before the instruction.
            for load in loads {
                let load_id = InstId(func.insts.len() as u32);
                func.insts.push(load);
                new_insts.push(load_id);
            }

            // Keep the original instruction.
            new_insts.push(inst_id);

            // Insert stores after the instruction.
            for store in stores {
                let store_id = InstId(func.insts.len() as u32);
                func.insts.push(store);
                new_insts.push(store_id);
            }
        }

        func.blocks[block_idx].insts = new_insts;
    }

    spill_infos
}

/// Returns the size in bytes for a register class.
fn reg_class_size(class: RegClass) -> u32 {
    match class {
        RegClass::Gpr32 | RegClass::Fpr32 => 4,
        RegClass::Gpr64 | RegClass::Fpr64 => 8,
        RegClass::Fpr128 => 16,
        // Smaller FPR classes: use their natural size
        RegClass::Fpr16 => 2,
        RegClass::Fpr8 => 1,
        RegClass::System => 4,
    }
}

/// Create a pseudo spill-load instruction: load VReg from stack slot.
fn make_spill_load(vreg: VReg, slot: StackSlotId) -> MachInst {
    MachInst {
        opcode: PSEUDO_SPILL_LOAD,
        defs: vec![MachOperand::VReg(vreg)],
        uses: vec![MachOperand::StackSlot(slot)],
        implicit_defs: Vec::new(),
        implicit_uses: Vec::new(),
        flags: InstFlags(InstFlags::READS_MEMORY),
    }
}

/// Create a pseudo spill-store instruction: store VReg to stack slot.
fn make_spill_store(vreg: VReg, slot: StackSlotId) -> MachInst {
    MachInst {
        opcode: PSEUDO_SPILL_STORE,
        defs: vec![],
        uses: vec![MachOperand::VReg(vreg), MachOperand::StackSlot(slot)],
        implicit_defs: Vec::new(),
        implicit_uses: Vec::new(),
        flags: InstFlags(InstFlags::WRITES_MEMORY),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reg_class_sizes() {
        assert_eq!(reg_class_size(RegClass::Gpr32), 4);
        assert_eq!(reg_class_size(RegClass::Gpr64), 8);
        assert_eq!(reg_class_size(RegClass::Fpr32), 4);
        assert_eq!(reg_class_size(RegClass::Fpr64), 8);
        assert_eq!(reg_class_size(RegClass::Fpr128), 16);
        assert_eq!(reg_class_size(RegClass::Fpr16), 2);
        assert_eq!(reg_class_size(RegClass::Fpr8), 1);
        assert_eq!(reg_class_size(RegClass::System), 4);
    }
}
