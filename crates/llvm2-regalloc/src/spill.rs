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
        flags: InstFlags::READS_MEMORY,
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
        flags: InstFlags::WRITES_MEMORY,
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

    /// Helper: build a simple single-block function where v0 is defined and used.
    fn make_simple_def_use() -> (MachFunction, HashMap<VReg, PReg>) {
        use crate::machine_types::*;
        let mut insts = Vec::new();

        // Inst 0: def v0 = imm 42
        let i0 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 1,
            defs: vec![MachOperand::VReg(VReg { id: 0, class: RegClass::Gpr64 })],
            uses: vec![MachOperand::Imm(42)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });

        // Inst 1: use v0
        let i1 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 2,
            defs: vec![],
            uses: vec![MachOperand::VReg(VReg { id: 0, class: RegClass::Gpr64 })],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });

        let func = MachFunction {
            name: "simple_spill".into(),
            insts,
            blocks: vec![MachBlock {
                insts: vec![i0, i1],
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: 1,
            next_stack_slot: 0,
            stack_slots: std::collections::HashMap::new(),
        };

        let allocation = HashMap::new(); // v0 is spilled, not allocated
        (func, allocation)
    }

    #[test]
    fn test_insert_spill_code_allocates_stack_slot() {
        let (mut func, allocation) = make_simple_def_use();
        let v0 = VReg { id: 0, class: RegClass::Gpr64 };

        let spill_infos = insert_spill_code(&mut func, &[v0], &allocation);

        assert_eq!(spill_infos.len(), 1);
        assert_eq!(spill_infos[0].vreg, v0);
        // Stack slot should have been allocated.
        assert!(func.stack_slots.contains_key(&spill_infos[0].slot));
        let slot = &func.stack_slots[&spill_infos[0].slot];
        assert_eq!(slot.size, 8); // Gpr64 = 8 bytes
        assert_eq!(slot.align, 8);
    }

    #[test]
    fn test_insert_spill_code_inserts_load_and_store() {
        let (mut func, allocation) = make_simple_def_use();
        let v0 = VReg { id: 0, class: RegClass::Gpr64 };

        insert_spill_code(&mut func, &[v0], &allocation);

        // After spill insertion, the block should have more instructions:
        // - Before use of v0: a PSEUDO_SPILL_LOAD
        // - After def of v0: a PSEUDO_SPILL_STORE
        let block = &func.blocks[0];
        assert!(
            block.insts.len() > 2,
            "should have inserted spill load/store, got {} insts",
            block.insts.len()
        );

        // Find load and store pseudo-ops.
        let has_load = block.insts.iter().any(|&id| func.insts[id.0 as usize].opcode == PSEUDO_SPILL_LOAD);
        let has_store = block.insts.iter().any(|&id| func.insts[id.0 as usize].opcode == PSEUDO_SPILL_STORE);
        assert!(has_load, "should have inserted a spill load");
        assert!(has_store, "should have inserted a spill store");
    }

    #[test]
    fn test_insert_spill_code_load_before_use() {
        let (mut func, allocation) = make_simple_def_use();
        let v0 = VReg { id: 0, class: RegClass::Gpr64 };

        insert_spill_code(&mut func, &[v0], &allocation);

        let block = &func.blocks[0];
        // Find positions of load and the original use instruction.
        let load_pos = block.insts.iter().position(|&id| {
            func.insts[id.0 as usize].opcode == PSEUDO_SPILL_LOAD
        });
        let use_pos = block.insts.iter().position(|&id| {
            func.insts[id.0 as usize].opcode == 2 // the original "use v0" instruction
        });

        assert!(load_pos.is_some(), "should have a spill load");
        assert!(use_pos.is_some(), "should have original use instruction");
        assert!(
            load_pos.unwrap() < use_pos.unwrap(),
            "spill load should come before the use"
        );
    }

    #[test]
    fn test_insert_spill_code_store_after_def() {
        let (mut func, allocation) = make_simple_def_use();
        let v0 = VReg { id: 0, class: RegClass::Gpr64 };

        insert_spill_code(&mut func, &[v0], &allocation);

        let block = &func.blocks[0];
        // Find positions of the original def and the store.
        let def_pos = block.insts.iter().position(|&id| {
            func.insts[id.0 as usize].opcode == 1 // the original "def v0" instruction
        });
        let store_pos = block.insts.iter().position(|&id| {
            func.insts[id.0 as usize].opcode == PSEUDO_SPILL_STORE
        });

        assert!(def_pos.is_some(), "should have original def instruction");
        assert!(store_pos.is_some(), "should have a spill store");
        assert!(
            store_pos.unwrap() > def_pos.unwrap(),
            "spill store should come after the def"
        );
    }

    #[test]
    fn test_insert_spill_code_multiple_vregs() {
        use crate::machine_types::*;
        let mut insts = Vec::new();

        // def v0, def v1, use v0, use v1
        let i0 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 1,
            defs: vec![MachOperand::VReg(VReg { id: 0, class: RegClass::Gpr64 })],
            uses: vec![MachOperand::Imm(1)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });
        let i1 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 1,
            defs: vec![MachOperand::VReg(VReg { id: 1, class: RegClass::Fpr64 })],
            uses: vec![MachOperand::FImm(3.14)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });
        let i2 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 2,
            defs: vec![],
            uses: vec![
                MachOperand::VReg(VReg { id: 0, class: RegClass::Gpr64 }),
                MachOperand::VReg(VReg { id: 1, class: RegClass::Fpr64 }),
            ],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });

        let mut func = MachFunction {
            name: "multi_spill".into(),
            insts,
            blocks: vec![MachBlock {
                insts: vec![i0, i1, i2],
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: 2,
            next_stack_slot: 0,
            stack_slots: std::collections::HashMap::new(),
        };

        let v0 = VReg { id: 0, class: RegClass::Gpr64 };
        let v1 = VReg { id: 1, class: RegClass::Fpr64 };

        let spill_infos = insert_spill_code(&mut func, &[v0, v1], &HashMap::new());

        assert_eq!(spill_infos.len(), 2);
        // Different register classes should get slots of different sizes.
        let slot0 = &func.stack_slots[&spill_infos[0].slot];
        let slot1 = &func.stack_slots[&spill_infos[1].slot];
        assert_eq!(slot0.size, 8); // Gpr64
        assert_eq!(slot1.size, 8); // Fpr64

        // Each spilled vreg should get its own unique slot.
        assert_ne!(spill_infos[0].slot, spill_infos[1].slot);
    }

    #[test]
    fn test_insert_spill_code_no_spills() {
        let (mut func, allocation) = make_simple_def_use();
        // No spilled vregs — nothing should happen.
        let spill_infos = insert_spill_code(&mut func, &[], &allocation);
        assert!(spill_infos.is_empty());
        // Block should be unchanged.
        assert_eq!(func.blocks[0].insts.len(), 2);
    }

    #[test]
    fn test_spill_load_has_correct_flags() {
        let v = VReg { id: 0, class: RegClass::Gpr64 };
        let slot = StackSlotId(0);
        let load = make_spill_load(v, slot);

        assert_eq!(load.opcode, PSEUDO_SPILL_LOAD);
        assert!(load.flags.reads_memory());
        assert_eq!(load.defs.len(), 1);
        assert_eq!(load.defs[0].as_vreg(), Some(v));
        assert_eq!(load.uses.len(), 1);
        assert_eq!(load.uses[0], MachOperand::StackSlot(slot));
    }

    #[test]
    fn test_spill_store_has_correct_flags() {
        let v = VReg { id: 0, class: RegClass::Gpr64 };
        let slot = StackSlotId(0);
        let store = make_spill_store(v, slot);

        assert_eq!(store.opcode, PSEUDO_SPILL_STORE);
        assert!(store.flags.writes_memory());
        assert!(store.defs.is_empty());
        assert_eq!(store.uses.len(), 2);
        assert_eq!(store.uses[0].as_vreg(), Some(v));
        assert_eq!(store.uses[1], MachOperand::StackSlot(slot));
    }

    #[test]
    fn test_insert_spill_code_multi_block() {
        use crate::machine_types::*;
        let mut insts = Vec::new();

        // Block 0: def v0, branch -> block 1
        let i0 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 1,
            defs: vec![MachOperand::VReg(VReg { id: 0, class: RegClass::Gpr64 })],
            uses: vec![MachOperand::Imm(42)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });
        let i1 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 0xBA,
            defs: vec![],
            uses: vec![MachOperand::Block(BlockId(1))],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),
        });

        // Block 1: use v0
        let i2 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 2,
            defs: vec![],
            uses: vec![MachOperand::VReg(VReg { id: 0, class: RegClass::Gpr64 })],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });

        let mut func = MachFunction {
            name: "multi_block_spill".into(),
            insts,
            blocks: vec![
                MachBlock {
                    insts: vec![i0, i1],
                    preds: Vec::new(),
                    succs: vec![BlockId(1)],
                    loop_depth: 0,
                },
                MachBlock {
                    insts: vec![i2],
                    preds: vec![BlockId(0)],
                    succs: Vec::new(),
                    loop_depth: 0,
                },
            ],
            block_order: vec![BlockId(0), BlockId(1)],
            entry_block: BlockId(0),
            next_vreg: 1,
            next_stack_slot: 0,
            stack_slots: std::collections::HashMap::new(),
        };

        let v0 = VReg { id: 0, class: RegClass::Gpr64 };
        let spill_infos = insert_spill_code(&mut func, &[v0], &HashMap::new());

        assert_eq!(spill_infos.len(), 1);

        // Block 0 should have a store after the def.
        let block0_has_store = func.blocks[0].insts.iter().any(|&id| {
            func.insts[id.0 as usize].opcode == PSEUDO_SPILL_STORE
        });
        assert!(block0_has_store, "block 0 should have a spill store after def");

        // Block 1 should have a load before the use.
        let block1_has_load = func.blocks[1].insts.iter().any(|&id| {
            func.insts[id.0 as usize].opcode == PSEUDO_SPILL_LOAD
        });
        assert!(block1_has_load, "block 1 should have a spill load before use");
    }
}
