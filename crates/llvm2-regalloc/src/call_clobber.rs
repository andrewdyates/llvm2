// llvm2-regalloc/call_clobber.rs - Call-clobber handling for register allocation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Call-clobber handling for the register allocator.
//!
//! When a live interval spans a call instruction, the value must be preserved
//! across the call. There are two strategies:
//!
//! 1. **Assign to a callee-saved register.** The callee will save/restore it.
//!    This is free in terms of our codegen (the callee handles it), but uses
//!    a callee-saved register that may be scarce.
//!
//! 2. **Save/restore around the call.** Insert a store before the call and
//!    a load after it. This is like a targeted spill just for the call site.
//!
//! This module analyzes which intervals cross calls and adjusts their
//! allocation preferences accordingly.
//!
//! Reference: LLVM `RegAllocGreedy.cpp` — call-crossing interval handling.

use crate::liveness::LiveInterval;
use crate::machine_types::{
    InstFlags, InstId, MachFunction, MachInst, MachOperand, PReg, RegClass, StackSlotId, VReg,
};
use crate::spill::{PSEUDO_SPILL_LOAD, PSEUDO_SPILL_STORE};
use std::collections::{HashMap, HashSet};

/// Information about a call instruction and the live values crossing it.
#[derive(Debug, Clone)]
pub struct CallCrossing {
    /// The instruction index (global numbering) of the call.
    pub call_inst_idx: u32,
    /// The InstId of the call instruction.
    pub call_inst_id: InstId,
    /// VRegs that are live across this call.
    pub live_across: Vec<VReg>,
}

/// AArch64 caller-saved registers (clobbered by calls).
/// X0-X15 (excluding X18) and V0-V7.
pub fn aarch64_caller_saved_regs() -> HashSet<PReg> {
    let mut regs = HashSet::new();
    // GPR caller-saved: X0-X15 (skip X16-X17 scratch, X18 reserved).
    for i in 0..=15 {
        regs.insert(PReg(i));
    }
    // FPR caller-saved: V0-V7 (encoded as 32-39).
    for i in 32..=39 {
        regs.insert(PReg(i));
    }
    regs
}

/// AArch64 callee-saved registers (preserved across calls).
/// X19-X28, V8-V15.
pub fn aarch64_callee_saved_regs() -> HashSet<PReg> {
    let mut regs = HashSet::new();
    // GPR callee-saved: X19-X28.
    for i in 19..=28 {
        regs.insert(PReg(i));
    }
    // FPR callee-saved: V8-V15 (encoded as 40-47).
    for i in 40..=47 {
        regs.insert(PReg(i));
    }
    regs
}

/// Find all call instructions and the VRegs live across them.
///
/// Returns a list of CallCrossing records, one per call instruction.
pub fn find_call_crossings(
    func: &MachFunction,
    intervals: &HashMap<u32, LiveInterval>,
    inst_numbering: &HashMap<InstId, u32>,
) -> Vec<CallCrossing> {
    let mut crossings = Vec::new();

    for block in &func.blocks {
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.0 as usize];
            if !inst.flags.is_call() {
                continue;
            }

            let call_idx = match inst_numbering.get(&inst_id) {
                Some(&idx) => idx,
                None => continue,
            };

            // Find all intervals that are live at this call.
            let mut live_across = Vec::new();
            for interval in intervals.values() {
                if interval.is_live_at(call_idx) {
                    live_across.push(interval.vreg);
                }
            }

            if !live_across.is_empty() {
                crossings.push(CallCrossing {
                    call_inst_idx: call_idx,
                    call_inst_id: inst_id,
                    live_across,
                });
            }
        }
    }

    crossings
}

/// Insert save/restore code around calls for VRegs assigned to caller-saved
/// registers.
///
/// For each call crossing:
/// 1. Check which live-across VRegs are assigned to caller-saved registers.
/// 2. For those VRegs, insert a store before the call and a load after.
///
/// Returns the number of save/restore pairs inserted.
pub fn insert_call_save_restore(
    func: &mut MachFunction,
    crossings: &[CallCrossing],
    allocation: &HashMap<VReg, PReg>,
    caller_saved: &HashSet<PReg>,
) -> u32 {
    let mut pairs_inserted = 0u32;

    // Build a plan first, then apply (to avoid mutating while iterating).
    // Plan entry: (block_idx, inst_pos_in_block, saves_before, restores_after).
    let mut plans: Vec<(usize, usize, Vec<MachInst>, Vec<MachInst>)> = Vec::new();

    for crossing in crossings {
        // Find which block and position this call is in.
        for (block_idx, block) in func.blocks.iter().enumerate() {
            if let Some(pos) = block.insts.iter().position(|&id| id == crossing.call_inst_id) {
                let mut saves = Vec::new();
                let mut restores = Vec::new();

                for &vreg in &crossing.live_across {
                    if let Some(&preg) = allocation.get(&vreg) {
                        if caller_saved.contains(&preg) {
                            // Need to save before and restore after.
                            let slot = StackSlotId(func.next_stack_slot);
                            func.next_stack_slot += 1;
                            let size = reg_class_size(vreg.class);
                            func.stack_slots.insert(
                                slot,
                                crate::machine_types::StackSlot { size, align: size },
                            );

                            // Store before call.
                            saves.push(MachInst {
                                opcode: PSEUDO_SPILL_STORE,
                                defs: vec![],
                                uses: vec![
                                    MachOperand::VReg(vreg),
                                    MachOperand::StackSlot(slot),
                                ],
                                implicit_defs: Vec::new(),
                                implicit_uses: Vec::new(),
                                flags: InstFlags(InstFlags::WRITES_MEMORY),
                            });

                            // Load after call.
                            restores.push(MachInst {
                                opcode: PSEUDO_SPILL_LOAD,
                                defs: vec![MachOperand::VReg(vreg)],
                                uses: vec![MachOperand::StackSlot(slot)],
                                implicit_defs: Vec::new(),
                                implicit_uses: Vec::new(),
                                flags: InstFlags(InstFlags::READS_MEMORY),
                            });

                            pairs_inserted += 1;
                        }
                    }
                }

                if !saves.is_empty() || !restores.is_empty() {
                    plans.push((block_idx, pos, saves, restores));
                }
                break;
            }
        }
    }

    // Apply plans in reverse order to maintain position validity.
    plans.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));

    for (block_idx, pos, saves, restores) in plans {
        let block = &mut func.blocks[block_idx];

        // Insert restores after the call.
        for (i, restore) in restores.into_iter().enumerate() {
            let restore_id = InstId(func.insts.len() as u32);
            func.insts.push(restore);
            block.insts.insert(pos + 1 + i, restore_id);
        }

        // Insert saves before the call.
        for (_i, save) in saves.into_iter().rev().enumerate() {
            let save_id = InstId(func.insts.len() as u32);
            func.insts.push(save);
            block.insts.insert(pos, save_id);
        }
    }

    pairs_inserted
}

/// Compute register allocation hints for call-crossing intervals.
///
/// Returns a map from VReg to a preferred PReg. Call-crossing intervals
/// should prefer callee-saved registers to avoid save/restore overhead.
pub fn compute_call_crossing_hints(
    crossings: &[CallCrossing],
    callee_saved: &HashSet<PReg>,
    allocatable: &HashMap<RegClass, Vec<PReg>>,
) -> HashMap<VReg, Vec<PReg>> {
    let mut hints: HashMap<VReg, Vec<PReg>> = HashMap::new();

    // Collect all VRegs that cross at least one call.
    let mut call_crossing_vregs: HashSet<VReg> = HashSet::new();
    for crossing in crossings {
        for &vreg in &crossing.live_across {
            call_crossing_vregs.insert(vreg);
        }
    }

    // For each call-crossing VReg, prefer callee-saved registers.
    for vreg in call_crossing_vregs {
        if let Some(regs) = allocatable.get(&vreg.class) {
            let preferred: Vec<PReg> = regs
                .iter()
                .filter(|r| callee_saved.contains(r))
                .copied()
                .collect();
            if !preferred.is_empty() {
                hints.insert(vreg, preferred);
            }
        }
    }

    hints
}

fn reg_class_size(class: RegClass) -> u32 {
    match class {
        RegClass::Gpr32 | RegClass::Fpr32 => 4,
        RegClass::Gpr64 | RegClass::Fpr64 => 8,
        RegClass::Vec128 => 16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liveness::LiveInterval;
    use crate::machine_types::{
        BlockId, InstFlags, InstId, MachBlock, MachFunction, MachInst, MachOperand, PReg,
        RegClass, VReg,
    };
    use std::collections::HashMap;

    fn vreg(id: u32) -> VReg {
        VReg {
            id,
            class: RegClass::Gpr64,
        }
    }

    fn interval_at(id: u32, start: u32, end: u32) -> LiveInterval {
        let mut iv = LiveInterval::new(vreg(id));
        iv.add_range(start, end);
        iv
    }

    #[test]
    fn test_aarch64_caller_saved() {
        let cs = aarch64_caller_saved_regs();
        assert!(cs.contains(&PReg(0)));  // X0
        assert!(cs.contains(&PReg(15))); // X15
        assert!(!cs.contains(&PReg(19))); // X19 is callee-saved
        assert!(cs.contains(&PReg(32))); // V0
        assert!(cs.contains(&PReg(39))); // V7
        assert!(!cs.contains(&PReg(40))); // V8 is callee-saved
    }

    #[test]
    fn test_aarch64_callee_saved() {
        let cs = aarch64_callee_saved_regs();
        assert!(!cs.contains(&PReg(0)));  // X0 is caller-saved
        assert!(cs.contains(&PReg(19))); // X19
        assert!(cs.contains(&PReg(28))); // X28
        assert!(cs.contains(&PReg(40))); // V8
        assert!(cs.contains(&PReg(47))); // V15
    }

    #[test]
    fn test_find_call_crossings() {
        // Build a simple function: define v0, call, use v0.
        let insts = vec![
            // inst 0: def v0
            MachInst {
                opcode: 1,
                defs: vec![MachOperand::VReg(vreg(0))],
                uses: vec![],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::default(),
            },
            // inst 1: CALL
            MachInst {
                opcode: 2,
                defs: vec![],
                uses: vec![],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags(InstFlags::IS_CALL),
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
        ];

        let inst_ids: Vec<InstId> = (0..3).map(|i| InstId(i)).collect();
        let func = MachFunction {
            name: "test".into(),
            insts,
            blocks: vec![MachBlock {
                insts: inst_ids.clone(),
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: 1,
            next_stack_slot: 0,
            stack_slots: HashMap::new(),
        };

        let numbering: HashMap<InstId, u32> = inst_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i as u32))
            .collect();

        // v0 is live [0, 3) — spans the call at index 1.
        let intervals = HashMap::from([(0u32, interval_at(0, 0, 3))]);

        let crossings = find_call_crossings(&func, &intervals, &numbering);
        assert_eq!(crossings.len(), 1);
        assert_eq!(crossings[0].live_across.len(), 1);
        assert_eq!(crossings[0].live_across[0], vreg(0));
    }

    #[test]
    fn test_compute_call_crossing_hints() {
        let crossings = vec![CallCrossing {
            call_inst_idx: 5,
            call_inst_id: InstId(5),
            live_across: vec![vreg(0)],
        }];

        let callee_saved = aarch64_callee_saved_regs();
        let mut allocatable = HashMap::new();
        let gpr: Vec<PReg> = (0..=15).chain(19..=28).map(PReg).collect();
        allocatable.insert(RegClass::Gpr64, gpr);

        let hints = compute_call_crossing_hints(&crossings, &callee_saved, &allocatable);
        assert!(hints.contains_key(&vreg(0)));
        let prefs = &hints[&vreg(0)];
        // All preferred registers should be callee-saved (X19-X28).
        for &preg in prefs {
            assert!(callee_saved.contains(&preg), "{:?} is not callee-saved", preg);
        }
    }
}
