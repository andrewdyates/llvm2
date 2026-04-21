// llvm2-regalloc/post_ra_opt.rs - Post-register-allocation spill optimization
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Post-register-allocation spill optimization.
//!
//! After register allocation inserts spill code (store/load pseudo-instructions),
//! conservative spill decisions may leave dead stores and redundant reloads.
//! This pass cleans up the spill code in three sub-passes:
//!
//! ## Pass 1: Dead Spill Store Elimination
//!
//! A spill store (PSEUDO_SPILL_STORE) writes a register to a stack slot. If that
//! stack slot is never reloaded (no PSEUDO_SPILL_LOAD with the same slot exists
//! anywhere in the function), the store is dead and can be removed.
//!
//! ## Pass 2: Redundant Reload Elimination
//!
//! Within a basic block, if the same stack slot is loaded multiple times and the
//! register holding the first load's result has not been redefined, subsequent
//! loads from the same slot are redundant. The redundant loads are removed and
//! all uses of the redundant load's def VReg are rewritten to the original
//! load's def VReg.
//!
//! ## Pass 3: Dead Definition Elimination
//!
//! After removing stores and loads, some instructions may have become dead:
//! their only consumer was a now-removed spill store, and they have no side
//! effects. These dead definitions are removed.
//!
//! Reference: LLVM `MachineLateOptimization.cpp` — post-RA dead store/load
//! elimination.

use crate::machine_types::{
    InstFlags, InstId, MachFunction, MachInst, MachOperand, StackSlotId, VReg,
};
use crate::spill::{PSEUDO_SPILL_STORE, PSEUDO_SPILL_LOAD};
use std::collections::{HashMap, HashSet};

/// NOP opcode for deleted instructions (same as post_ra_coalesce.rs).
const NOP_OPCODE: u16 = 0xFFFF;

/// Statistics from the post-RA spill optimization pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PostRAOptResult {
    /// Number of dead spill stores removed.
    pub dead_stores_removed: u32,
    /// Number of redundant reloads removed.
    pub redundant_reloads_removed: u32,
    /// Number of dead definitions removed.
    pub dead_defs_removed: u32,
}

/// Run post-register-allocation spill optimization on the given function.
///
/// This pass modifies the function in-place, removing dead spill stores,
/// redundant reloads, and dead definitions that result from spill cleanup.
/// Deleted instructions are replaced with NOP pseudo-instructions and then
/// removed from block instruction lists.
///
/// Returns statistics about what was eliminated.
pub fn post_ra_optimize(func: &mut MachFunction) -> PostRAOptResult {
    let mut result = PostRAOptResult::default();

    // Pass 1: Dead spill store elimination.
    eliminate_dead_stores(func, &mut result);

    // Pass 2: Redundant reload elimination.
    eliminate_redundant_reloads(func, &mut result);

    // Pass 3: Dead definition elimination.
    eliminate_dead_defs(func, &mut result);

    // Remove NOP'd instructions from all block instruction lists.
    for block in &mut func.blocks {
        block.insts.retain(|inst_id| {
            func.insts[inst_id.0 as usize].opcode != NOP_OPCODE
        });
    }

    result
}

/// Pass 1: Remove spill stores to stack slots that are never reloaded.
fn eliminate_dead_stores(func: &mut MachFunction, result: &mut PostRAOptResult) {
    // Collect all stack slot IDs that appear in any PSEUDO_SPILL_LOAD.
    let mut loaded_slots: HashSet<StackSlotId> = HashSet::new();

    for block in &func.blocks {
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.0 as usize];
            if inst.opcode == PSEUDO_SPILL_LOAD
                && let Some(slot) = extract_stack_slot_from_uses(inst) {
                    loaded_slots.insert(slot);
                }
        }
    }

    // NOP any PSEUDO_SPILL_STORE whose slot is not in the loaded set.
    for block in &func.blocks {
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.0 as usize];
            if inst.opcode == PSEUDO_SPILL_STORE
                && let Some(slot) = extract_stack_slot_from_uses(inst)
                    && !loaded_slots.contains(&slot) {
                        nop_inst(&mut func.insts[inst_id.0 as usize]);
                        result.dead_stores_removed += 1;
                    }
        }
    }
}

/// Pass 2: Remove redundant reloads within each basic block.
///
/// For each block, tracks which stack slots have been loaded and into which
/// VReg. When the same slot is loaded again and the previous load's def VReg
/// has not been redefined or clobbered, the second load is redundant.
fn eliminate_redundant_reloads(func: &mut MachFunction, result: &mut PostRAOptResult) {
    let block_indices: Vec<usize> = if func.block_order.is_empty() {
        (0..func.blocks.len()).collect()
    } else {
        func.block_order.iter().map(|b| b.0 as usize).collect()
    };

    for block_idx in block_indices {
        let block_insts: Vec<InstId> = func.blocks[block_idx].insts.clone();
        eliminate_redundant_reloads_in_block(func, &block_insts, result);
    }
}

fn eliminate_redundant_reloads_in_block(
    func: &mut MachFunction,
    block_insts: &[InstId],
    result: &mut PostRAOptResult,
) {
    // Maps stack slot -> (def VReg of the load, InstId of the load).
    // Active means the def VReg has not been redefined.
    let mut active_loads: HashMap<StackSlotId, (VReg, InstId)> = HashMap::new();

    // Collect rewrites to apply: (old_vreg, new_vreg).
    let mut vreg_rewrites: HashMap<u32, VReg> = HashMap::new();

    for &inst_id in block_insts {
        let inst = &func.insts[inst_id.0 as usize];

        // Skip already-NOP'd instructions.
        if inst.opcode == NOP_OPCODE {
            continue;
        }

        if inst.opcode == PSEUDO_SPILL_STORE {
            // A store to a slot invalidates the active load for that slot.
            if let Some(slot) = extract_stack_slot_from_uses(inst) {
                active_loads.remove(&slot);
            }
            continue;
        }

        if inst.opcode == PSEUDO_SPILL_LOAD {
            if let (Some(load_vreg), Some(slot)) = (
                inst.defs.first().and_then(|op| op.as_vreg()),
                extract_stack_slot_from_uses(inst),
            ) {
                if let Some(&(prev_vreg, _prev_inst_id)) = active_loads.get(&slot) {
                    // Redundant reload: same slot already loaded into prev_vreg.
                    // Record the rewrite and NOP this load.
                    vreg_rewrites.insert(load_vreg.id, prev_vreg);
                    nop_inst(&mut func.insts[inst_id.0 as usize]);
                    result.redundant_reloads_removed += 1;
                } else {
                    // First load from this slot in this block.
                    active_loads.insert(slot, (load_vreg, inst_id));
                }
            }
            continue;
        }

        // For non-spill instructions: check if any def clobbers an active
        // load's VReg, invalidating it.
        let defs_to_check: Vec<VReg> = inst.defs.iter()
            .filter_map(|op| op.as_vreg())
            .collect();

        for def_vreg in &defs_to_check {
            // Remove any active load whose def VReg matches.
            active_loads.retain(|_slot, (v, _)| v.id != def_vreg.id);
        }
    }

    // Apply VReg rewrites to all remaining instructions in the block.
    if !vreg_rewrites.is_empty() {
        for &inst_id in block_insts {
            let inst = &mut func.insts[inst_id.0 as usize];
            if inst.opcode == NOP_OPCODE {
                continue;
            }
            rewrite_vreg_uses(inst, &vreg_rewrites);
        }
    }
}

/// Pass 3: Remove instructions whose defs are unused and that have no side
/// effects.
///
/// An instruction is a dead definition if:
/// - It defines one or more VRegs
/// - None of those VRegs are used by any remaining (non-NOP) instruction
/// - The instruction has no side effects (not a call, branch, memory op, etc.)
fn eliminate_dead_defs(func: &mut MachFunction, result: &mut PostRAOptResult) {
    // Collect all VRegs that are used by non-NOP instructions.
    let mut used_vregs: HashSet<u32> = HashSet::new();

    for block in &func.blocks {
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.0 as usize];
            if inst.opcode == NOP_OPCODE {
                continue;
            }
            for use_op in &inst.uses {
                if let Some(vreg) = use_op.as_vreg() {
                    used_vregs.insert(vreg.id);
                }
            }
        }
    }

    // NOP any pure definition instruction whose defs are all unused.
    for block in &func.blocks {
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.0 as usize];
            if inst.opcode == NOP_OPCODE {
                continue;
            }

            // Skip instructions with side effects.
            if has_side_effects(inst) {
                continue;
            }

            // Must define at least one VReg.
            let def_vregs: Vec<u32> = inst.defs.iter()
                .filter_map(|op| op.as_vreg().map(|v| v.id))
                .collect();

            if def_vregs.is_empty() {
                continue;
            }

            // All defs must be unused.
            if def_vregs.iter().all(|id| !used_vregs.contains(id)) {
                nop_inst(&mut func.insts[inst_id.0 as usize]);
                result.dead_defs_removed += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the StackSlotId from an instruction's uses list.
fn extract_stack_slot_from_uses(inst: &MachInst) -> Option<StackSlotId> {
    for op in &inst.uses {
        if let MachOperand::StackSlot(slot) = op {
            return Some(*slot);
        }
    }
    None
}

/// Check if an instruction has side effects that prevent dead-def elimination.
fn has_side_effects(inst: &MachInst) -> bool {
    inst.flags.is_call()
        || inst.flags.is_branch()
        || inst.flags.is_terminator()
        || inst.flags.writes_memory()
        || inst.flags.reads_memory()
        || inst.flags.has_side_effects()
}

/// Replace an instruction with a NOP (will be removed from block later).
fn nop_inst(inst: &mut MachInst) {
    inst.opcode = NOP_OPCODE;
    inst.defs.clear();
    inst.uses.clear();
    inst.implicit_defs.clear();
    inst.implicit_uses.clear();
    inst.flags = InstFlags::default();
}

/// Rewrite VReg uses in an instruction according to the rewrite map.
fn rewrite_vreg_uses(inst: &mut MachInst, rewrites: &HashMap<u32, VReg>) {
    for op in &mut inst.uses {
        if let MachOperand::VReg(vreg) = op
            && let Some(new_vreg) = rewrites.get(&vreg.id) {
                *vreg = *new_vreg;
            }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine_types::{
        BlockId, InstFlags, MachBlock, MachFunction, MachInst, MachOperand,
        RegClass, StackSlotId, VReg,
    };
    use std::collections::HashMap;

    /// Helper: create a spill store instruction (VReg -> StackSlot).
    fn spill_store(vreg: VReg, slot: StackSlotId) -> MachInst {
        MachInst {
            opcode: PSEUDO_SPILL_STORE,
            defs: vec![],
            uses: vec![MachOperand::VReg(vreg), MachOperand::StackSlot(slot)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::WRITES_MEMORY,
        }
    }

    /// Helper: create a spill load instruction (StackSlot -> VReg).
    fn spill_load(vreg: VReg, slot: StackSlotId) -> MachInst {
        MachInst {
            opcode: PSEUDO_SPILL_LOAD,
            defs: vec![MachOperand::VReg(vreg)],
            uses: vec![MachOperand::StackSlot(slot)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::READS_MEMORY,
        }
    }

    /// Helper: create a generic instruction that defines a VReg.
    fn def_inst(opcode: u16, vreg: VReg) -> MachInst {
        MachInst {
            opcode,
            defs: vec![MachOperand::VReg(vreg)],
            uses: vec![MachOperand::Imm(42)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        }
    }

    /// Helper: create a generic instruction that uses a VReg.
    fn use_inst(opcode: u16, vreg: VReg) -> MachInst {
        MachInst {
            opcode,
            defs: vec![],
            uses: vec![MachOperand::VReg(vreg)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        }
    }

    /// Helper: create a side-effecting instruction that defines a VReg.
    fn side_effect_def_inst(opcode: u16, vreg: VReg) -> MachInst {
        MachInst {
            opcode,
            defs: vec![MachOperand::VReg(vreg)],
            uses: vec![MachOperand::Imm(0)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::HAS_SIDE_EFFECTS,
        }
    }

    /// Build a MachFunction from a list of blocks, each being a list of MachInsts.
    fn make_function(blocks_insts: Vec<Vec<MachInst>>) -> MachFunction {
        let mut insts = Vec::new();
        let mut blocks = Vec::new();
        let mut block_order = Vec::new();

        for block_insts in blocks_insts {
            let block_id = BlockId(blocks.len() as u32);
            let mut inst_ids = Vec::new();

            for inst in block_insts {
                let inst_id = InstId(insts.len() as u32);
                insts.push(inst);
                inst_ids.push(inst_id);
            }

            blocks.push(MachBlock {
                insts: inst_ids,
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            });
            block_order.push(block_id);
        }

        MachFunction {
            name: "test_post_ra_opt".into(),
            insts,
            blocks,
            block_order,
            entry_block: BlockId(0),
            next_vreg: 100,
            next_stack_slot: 10,
            stack_slots: HashMap::new(),
        }
    }

    fn v(id: u32) -> VReg {
        VReg { id, class: RegClass::Gpr64 }
    }

    fn slot(id: u32) -> StackSlotId {
        StackSlotId(id)
    }

    // -- Dead store elimination tests --

    #[test]
    fn test_dead_store_no_reload() {
        // Store to slot 0, but no load from slot 0 anywhere.
        let mut func = make_function(vec![vec![
            def_inst(1, v(0)),
            spill_store(v(0), slot(0)),
        ]]);

        let result = post_ra_optimize(&mut func);

        assert_eq!(result.dead_stores_removed, 1);
        // Only the def instruction should remain (store removed, def may be
        // removed by dead-def if unused).
        assert_eq!(func.blocks[0].insts.len(), 0);
    }

    #[test]
    fn test_store_kept_when_reload_exists() {
        // Store to slot 0, and a reload from slot 0 exists.
        let mut func = make_function(vec![vec![
            def_inst(1, v(0)),
            spill_store(v(0), slot(0)),
            spill_load(v(1), slot(0)),
            use_inst(2, v(1)),
        ]]);

        let result = post_ra_optimize(&mut func);

        assert_eq!(result.dead_stores_removed, 0);
        // Store and load should both remain.
        let opcodes: Vec<u16> = func.blocks[0].insts.iter()
            .map(|&id| func.insts[id.0 as usize].opcode)
            .collect();
        assert!(opcodes.contains(&PSEUDO_SPILL_STORE));
        assert!(opcodes.contains(&PSEUDO_SPILL_LOAD));
    }

    #[test]
    fn test_dead_store_multi_slot() {
        // Store to slot 0 (dead) and slot 1 (has load).
        let mut func = make_function(vec![vec![
            def_inst(1, v(0)),
            spill_store(v(0), slot(0)),
            def_inst(1, v(1)),
            spill_store(v(1), slot(1)),
            spill_load(v(2), slot(1)),
            use_inst(2, v(2)),
        ]]);

        let result = post_ra_optimize(&mut func);

        assert_eq!(result.dead_stores_removed, 1);
        // Slot 1's store should remain, slot 0's store should be gone.
        let has_store_slot0 = func.blocks[0].insts.iter().any(|&id| {
            let inst = &func.insts[id.0 as usize];
            inst.opcode == PSEUDO_SPILL_STORE
                && extract_stack_slot_from_uses(inst) == Some(slot(0))
        });
        assert!(!has_store_slot0, "dead store to slot 0 should be removed");
    }

    #[test]
    fn test_dead_store_multi_block_reload() {
        // Store in block 0, reload in block 1 — store should be kept.
        let mut func = make_function(vec![
            vec![
                def_inst(1, v(0)),
                spill_store(v(0), slot(0)),
            ],
            vec![
                spill_load(v(1), slot(0)),
                use_inst(2, v(1)),
            ],
        ]);

        let result = post_ra_optimize(&mut func);

        assert_eq!(result.dead_stores_removed, 0);
    }

    // -- Redundant reload elimination tests --

    #[test]
    fn test_redundant_reload_same_slot() {
        // Two loads from slot 0 without intervening store — second is redundant.
        let mut func = make_function(vec![vec![
            def_inst(1, v(0)),
            spill_store(v(0), slot(0)),
            spill_load(v(1), slot(0)),
            use_inst(2, v(1)),
            spill_load(v(2), slot(0)),
            use_inst(3, v(2)),
        ]]);

        let result = post_ra_optimize(&mut func);

        assert_eq!(result.redundant_reloads_removed, 1);
        // The second use should now reference v(1) instead of v(2).
        let use3_inst = func.blocks[0].insts.iter().find(|&&id| {
            func.insts[id.0 as usize].opcode == 3
        });
        assert!(use3_inst.is_some());
        let uses = &func.insts[use3_inst.unwrap().0 as usize].uses;
        assert_eq!(uses[0].as_vreg(), Some(v(1)));
    }

    #[test]
    fn test_reload_kept_after_store_invalidation() {
        // Load from slot 0, then store to slot 0, then load again.
        // The second load is NOT redundant because the slot was overwritten.
        let mut func = make_function(vec![vec![
            spill_load(v(1), slot(0)),
            use_inst(2, v(1)),
            def_inst(1, v(3)),
            spill_store(v(3), slot(0)),
            spill_load(v(4), slot(0)),
            use_inst(3, v(4)),
        ]]);

        let result = post_ra_optimize(&mut func);

        assert_eq!(result.redundant_reloads_removed, 0);
    }

    #[test]
    fn test_reload_kept_after_vreg_redef() {
        // Load v1 from slot 0, then redefine v1, then load v2 from slot 0.
        // Since v1 was redefined, the active tracking for slot 0 is cleared,
        // so the second load is NOT redundant.
        let mut func = make_function(vec![vec![
            spill_load(v(1), slot(0)),
            use_inst(2, v(1)),
            def_inst(5, v(1)),  // redefine v1
            spill_load(v(2), slot(0)),
            use_inst(3, v(2)),
        ]]);

        let result = post_ra_optimize(&mut func);

        assert_eq!(result.redundant_reloads_removed, 0);
    }

    #[test]
    fn test_redundant_reload_different_slots() {
        // Loads from two different slots — neither is redundant.
        let mut func = make_function(vec![vec![
            spill_load(v(1), slot(0)),
            use_inst(2, v(1)),
            spill_load(v(2), slot(1)),
            use_inst(3, v(2)),
        ]]);

        let result = post_ra_optimize(&mut func);

        assert_eq!(result.redundant_reloads_removed, 0);
    }

    #[test]
    fn test_redundant_reload_triple() {
        // Three loads from slot 0 without intervening store — last two are redundant.
        let mut func = make_function(vec![vec![
            def_inst(1, v(0)),
            spill_store(v(0), slot(0)),
            spill_load(v(1), slot(0)),
            use_inst(2, v(1)),
            spill_load(v(2), slot(0)),
            use_inst(3, v(2)),
            spill_load(v(3), slot(0)),
            use_inst(4, v(3)),
        ]]);

        let result = post_ra_optimize(&mut func);

        assert_eq!(result.redundant_reloads_removed, 2);
    }

    // -- Dead definition elimination tests --

    #[test]
    fn test_dead_def_after_store_removal() {
        // def v0 -> store v0 to slot 0 (dead, no load).
        // After store is removed, v0 has no uses -> def is dead too.
        let mut func = make_function(vec![vec![
            def_inst(1, v(0)),
            spill_store(v(0), slot(0)),
        ]]);

        let result = post_ra_optimize(&mut func);

        assert_eq!(result.dead_stores_removed, 1);
        assert_eq!(result.dead_defs_removed, 1);
        assert_eq!(func.blocks[0].insts.len(), 0);
    }

    #[test]
    fn test_dead_def_not_removed_with_side_effects() {
        // Side-effect instruction defines v0, store is dead.
        // The def should NOT be removed because it has side effects.
        let mut func = make_function(vec![vec![
            side_effect_def_inst(1, v(0)),
            spill_store(v(0), slot(0)),
        ]]);

        let result = post_ra_optimize(&mut func);

        assert_eq!(result.dead_stores_removed, 1);
        assert_eq!(result.dead_defs_removed, 0);
        assert_eq!(func.blocks[0].insts.len(), 1); // side-effect inst remains
    }

    #[test]
    fn test_dead_def_not_removed_when_used() {
        // def v0 is used by both the spill store AND a regular use.
        // Store is dead (no load), but def is still used by the regular use.
        let mut func = make_function(vec![vec![
            def_inst(1, v(0)),
            spill_store(v(0), slot(0)),
            use_inst(2, v(0)),
        ]]);

        let result = post_ra_optimize(&mut func);

        assert_eq!(result.dead_stores_removed, 1);
        assert_eq!(result.dead_defs_removed, 0);
        // def and use should remain.
        assert_eq!(func.blocks[0].insts.len(), 2);
    }

    // -- Empty function / edge cases --

    #[test]
    fn test_empty_function() {
        let mut func = make_function(vec![vec![]]);
        let result = post_ra_optimize(&mut func);
        assert_eq!(result, PostRAOptResult::default());
    }

    #[test]
    fn test_no_spill_instructions() {
        // Function with only regular instructions — nothing to optimize.
        let mut func = make_function(vec![vec![
            def_inst(1, v(0)),
            use_inst(2, v(0)),
        ]]);

        let result = post_ra_optimize(&mut func);

        assert_eq!(result.dead_stores_removed, 0);
        assert_eq!(result.redundant_reloads_removed, 0);
        assert_eq!(result.dead_defs_removed, 0);
        assert_eq!(func.blocks[0].insts.len(), 2);
    }

    #[test]
    fn test_store_and_load_same_block_kept() {
        // Store and load in same block — both should be kept (load uses the store).
        let mut func = make_function(vec![vec![
            def_inst(1, v(0)),
            spill_store(v(0), slot(0)),
            spill_load(v(1), slot(0)),
            use_inst(2, v(1)),
        ]]);

        let result = post_ra_optimize(&mut func);

        assert_eq!(result.dead_stores_removed, 0);
        assert_eq!(result.redundant_reloads_removed, 0);
    }

    #[test]
    fn test_fpr_register_spill() {
        // FPR register spills follow the same pattern.
        let fv0 = VReg { id: 50, class: RegClass::Fpr64 };
        let fv1 = VReg { id: 51, class: RegClass::Fpr64 };
        let mut func = make_function(vec![vec![
            MachInst {
                opcode: 1,
                defs: vec![MachOperand::VReg(fv0)],
                uses: vec![MachOperand::FImm(2.78)],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::default(),
            },
            spill_store(fv0, slot(5)),
            spill_load(fv1, slot(5)),
            spill_load(VReg { id: 52, class: RegClass::Fpr64 }, slot(5)),
            use_inst(2, VReg { id: 52, class: RegClass::Fpr64 }),
        ]]);

        let result = post_ra_optimize(&mut func);

        // Second load from slot 5 is redundant.
        assert_eq!(result.redundant_reloads_removed, 1);
    }

    #[test]
    fn test_multiple_stores_same_slot_one_dead() {
        // Two stores to slot 0, but there IS a load. Both stores are kept
        // because the slot has at least one load somewhere.
        let mut func = make_function(vec![vec![
            def_inst(1, v(0)),
            spill_store(v(0), slot(0)),
            def_inst(1, v(1)),
            spill_store(v(1), slot(0)),
            spill_load(v(2), slot(0)),
            use_inst(2, v(2)),
        ]]);

        let result = post_ra_optimize(&mut func);

        // No dead stores: slot 0 has a load.
        assert_eq!(result.dead_stores_removed, 0);
    }

    #[test]
    fn test_cross_block_no_redundant_reload() {
        // Load in block 0, load in block 1 — NOT redundant (cross-block).
        // Redundant reload elimination is block-local.
        let mut func = make_function(vec![
            vec![
                def_inst(1, v(0)),
                spill_store(v(0), slot(0)),
                spill_load(v(1), slot(0)),
                use_inst(2, v(1)),
            ],
            vec![
                spill_load(v(2), slot(0)),
                use_inst(3, v(2)),
            ],
        ]);

        let result = post_ra_optimize(&mut func);

        assert_eq!(result.redundant_reloads_removed, 0);
    }

    #[test]
    fn test_chain_def_store_no_load_all_removed() {
        // Chain: def v0 -> store v0 to slot 0, def v1 -> store v1 to slot 1.
        // Neither slot is loaded. Both stores and both defs should be removed.
        let mut func = make_function(vec![vec![
            def_inst(1, v(0)),
            spill_store(v(0), slot(0)),
            def_inst(1, v(1)),
            spill_store(v(1), slot(1)),
        ]]);

        let result = post_ra_optimize(&mut func);

        assert_eq!(result.dead_stores_removed, 2);
        assert_eq!(result.dead_defs_removed, 2);
        assert_eq!(func.blocks[0].insts.len(), 0);
    }

    #[test]
    fn test_mixed_spill_and_regular_instructions() {
        // Mix of regular instructions and spill code.
        let mut func = make_function(vec![vec![
            def_inst(1, v(0)),             // def v0
            use_inst(2, v(0)),             // use v0 (keeps def alive)
            def_inst(1, v(5)),             // def v5
            spill_store(v(5), slot(3)),    // dead store (no load of slot 3)
            def_inst(1, v(6)),             // def v6
            use_inst(3, v(6)),             // use v6
        ]]);

        let result = post_ra_optimize(&mut func);

        assert_eq!(result.dead_stores_removed, 1);
        // v5's def is now dead (only consumer was the removed store).
        assert_eq!(result.dead_defs_removed, 1);
        // Remaining: def v0, use v0, def v6, use v6.
        assert_eq!(func.blocks[0].insts.len(), 4);
    }

    #[test]
    fn test_redundant_reload_rewrite_propagates() {
        // After eliminating a redundant reload, the rewritten VReg should
        // be used correctly by downstream instructions.
        let mut func = make_function(vec![vec![
            def_inst(1, v(0)),
            spill_store(v(0), slot(0)),
            spill_load(v(1), slot(0)),       // first load
            spill_load(v(2), slot(0)),       // redundant, v2 -> v1
            // Instruction using v2 should be rewritten to use v1.
            MachInst {
                opcode: 10,
                defs: vec![MachOperand::VReg(v(3))],
                uses: vec![MachOperand::VReg(v(1)), MachOperand::VReg(v(2))],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::default(),
            },
            use_inst(11, v(3)),
        ]]);

        let result = post_ra_optimize(&mut func);

        assert_eq!(result.redundant_reloads_removed, 1);
        // Find the instruction with opcode 10 and verify its uses.
        let inst10 = func.blocks[0].insts.iter().find(|&&id| {
            func.insts[id.0 as usize].opcode == 10
        });
        assert!(inst10.is_some());
        let uses = &func.insts[inst10.unwrap().0 as usize].uses;
        // Both operands should now be v(1).
        assert_eq!(uses[0].as_vreg(), Some(v(1)));
        assert_eq!(uses[1].as_vreg(), Some(v(1)));
    }

    #[test]
    fn test_all_three_passes_combined() {
        // Scenario: def v0, store to slot 0 (dead), def v1, store to slot 1,
        // load v2 from slot 1, load v3 from slot 1 (redundant), use v3.
        let mut func = make_function(vec![vec![
            def_inst(1, v(0)),              // dead def (after store removal)
            spill_store(v(0), slot(0)),     // dead store (no load from slot 0)
            def_inst(1, v(1)),
            spill_store(v(1), slot(1)),
            spill_load(v(2), slot(1)),
            spill_load(v(3), slot(1)),      // redundant
            use_inst(2, v(3)),              // will be rewritten to use v(2)
        ]]);

        let result = post_ra_optimize(&mut func);

        assert_eq!(result.dead_stores_removed, 1);   // slot 0 store
        assert_eq!(result.redundant_reloads_removed, 1); // second load from slot 1
        assert_eq!(result.dead_defs_removed, 1);   // def v0
        // Remaining: def v1, store slot 1, load v2 from slot 1, use v2.
        assert_eq!(func.blocks[0].insts.len(), 4);
    }
}
