// llvm2-opt - Loop-Invariant Code Motion
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Loop-Invariant Code Motion (LICM) for machine-level IR.
//!
//! Hoists loop-invariant instructions to the loop preheader, reducing
//! redundant computation inside loops.
//!
//! # Safety Requirements
//!
//! **Only Pure instructions are hoisted.** The memory-effects model
//! classifies each opcode; only `MemoryEffect::Pure` instructions can
//! be safely moved out of loops. Loads, stores, and calls must NOT be
//! hoisted because:
//! - Loads may return different values on each iteration.
//! - Stores have side effects that must execute each iteration.
//! - Calls may have arbitrary side effects.
//!
//! # Loop Invariance
//!
//! An instruction is loop-invariant if ALL its source operands are:
//! - Defined outside the loop, OR
//! - Themselves loop-invariant (transitive).
//!
//! # Algorithm
//!
//! 1. Compute dominator tree and loop analysis.
//! 2. For each loop (innermost first for best results):
//!    a. Ensure a preheader exists (create one if needed).
//!    b. Build def-map: which instructions define which vregs.
//!    c. Iteratively identify loop-invariant instructions:
//!       - Pure instructions whose source operands are all loop-invariant.
//!    d. Hoist identified instructions to the preheader.
//! 3. Iterate until no more instructions can be hoisted.
//!
//! Reference: LLVM `LICM.cpp`

use std::collections::{HashMap, HashSet};

use llvm2_ir::{AArch64Opcode, BlockId, InstId, MachFunction, MachOperand};

use crate::dom::DomTree;
use crate::effects::{opcode_effect, MemoryEffect};
use crate::loops::{create_preheader, LoopAnalysis, NaturalLoop};
use crate::pass_manager::MachinePass;

/// Loop-Invariant Code Motion pass.
pub struct LoopInvariantCodeMotion;

impl MachinePass for LoopInvariantCodeMotion {
    fn name(&self) -> &str {
        "licm"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        let dom = DomTree::compute(func);
        let loop_analysis = LoopAnalysis::compute(func, &dom);

        if loop_analysis.is_empty() {
            return false;
        }

        let mut changed = false;

        // Process loops innermost-first (higher depth first).
        let mut loops: Vec<_> = loop_analysis.all_loops().cloned().collect();
        loops.sort_by(|a, b| b.depth.cmp(&a.depth));

        for lp in &loops {
            if hoist_loop_invariants(func, lp) {
                changed = true;
            }
        }

        changed
    }
}

/// Hoist loop-invariant instructions from a single loop to its preheader.
///
/// Returns true if any instructions were hoisted.
fn hoist_loop_invariants(func: &mut MachFunction, lp: &NaturalLoop) -> bool {
    // Ensure a preheader exists.
    let preheader = match lp.preheader {
        Some(ph) => ph,
        None => {
            // Need to create a preheader.
            create_preheader(func, lp.header, &lp.body)
        }
    };

    // Build a map: vreg_id -> (inst_id, block_id) for definitions inside the loop.
    let loop_defs = build_loop_defs(func, &lp.body);

    // Set of vreg IDs known to be loop-invariant.
    let mut invariant_vregs: HashSet<u32> = HashSet::new();

    // Instructions to hoist: (inst_id, source_block).
    let mut to_hoist: Vec<(InstId, BlockId)> = Vec::new();

    // Iteratively find loop-invariant instructions.
    // Keep iterating until no new invariants are found (transitive closure).
    let mut found_new = true;
    while found_new {
        found_new = false;

        for &block_id in &func.block_order {
            if !lp.body.contains(&block_id) {
                continue;
            }

            let block = func.block(block_id);
            for &inst_id in &block.insts {
                // Skip instructions already marked for hoisting.
                if to_hoist.iter().any(|(id, _)| *id == inst_id) {
                    continue;
                }

                let inst = func.inst(inst_id);

                // Only hoist pure instructions that produce a value.
                if opcode_effect(inst.opcode) != MemoryEffect::Pure {
                    continue;
                }
                if !produces_value(inst.opcode) {
                    continue;
                }

                // Don't hoist branches, terminators, or phis.
                if inst.is_branch() || inst.is_terminator() || inst.opcode.is_phi() {
                    continue;
                }

                // Check if all source operands are loop-invariant.
                let use_start = 1; // operand[0] is the def
                let all_invariant = inst.operands[use_start..]
                    .iter()
                    .all(|op| is_operand_loop_invariant(op, &loop_defs, &invariant_vregs));

                if all_invariant {
                    // Mark this instruction's def as invariant.
                    if let Some(MachOperand::VReg(def)) = inst.operands.first() {
                        invariant_vregs.insert(def.id);
                    }
                    to_hoist.push((inst_id, block_id));
                    found_new = true;
                }
            }
        }
    }

    if to_hoist.is_empty() {
        return false;
    }

    // Hoist instructions to the preheader.
    // Insert before the terminator (branch) of the preheader.
    for (inst_id, source_block) in &to_hoist {
        // Remove from source block.
        let block = func.block_mut(*source_block);
        block.insts.retain(|id| id != inst_id);

        // Insert into preheader, before the terminator.
        let ph_block = func.block_mut(preheader);
        let insert_pos = if ph_block.insts.is_empty() {
            0
        } else {
            // Insert before the last instruction (which should be the branch).
            ph_block.insts.len() - 1
        };
        ph_block.insts.insert(insert_pos, *inst_id);
    }

    true
}

/// Build a map from vreg_id to (InstId, BlockId) for all definitions
/// inside the loop body.
fn build_loop_defs(
    func: &MachFunction,
    body: &HashSet<BlockId>,
) -> HashMap<u32, (InstId, BlockId)> {
    let mut defs: HashMap<u32, (InstId, BlockId)> = HashMap::new();

    for &block_id in &func.block_order {
        if !body.contains(&block_id) {
            continue;
        }
        let block = func.block(block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            if produces_value(inst.opcode) {
                if let Some(MachOperand::VReg(def)) = inst.operands.first() {
                    defs.insert(def.id, (inst_id, block_id));
                }
            }
        }
    }

    defs
}

/// Check if an operand is loop-invariant.
///
/// An operand is loop-invariant if:
/// - It is an immediate or other non-vreg operand.
/// - It is a vreg defined outside the loop.
/// - It is a vreg defined inside the loop but already marked invariant.
fn is_operand_loop_invariant(
    operand: &MachOperand,
    loop_defs: &HashMap<u32, (InstId, BlockId)>,
    invariant_vregs: &HashSet<u32>,
) -> bool {
    match operand {
        MachOperand::VReg(vreg) => {
            // If this vreg is defined inside the loop...
            if loop_defs.contains_key(&vreg.id) {
                // ...it's only invariant if we've already marked it so.
                invariant_vregs.contains(&vreg.id)
            } else {
                // Defined outside the loop — invariant by definition.
                true
            }
        }
        // Immediates, physical registers, blocks, etc. are invariant.
        _ => true,
    }
}

/// Returns true if this opcode produces a value (operand[0] is a def).
fn produces_value(opcode: AArch64Opcode) -> bool {
    use AArch64Opcode::*;
    match opcode {
        CmpRR | CmpRI | Tst | Fcmp => false,
        StrRI | StpRI | StpPreIndex => false,
        B | BCond | Cbz | Cbnz | Br | Ret => false,
        Nop => false,
        Bl | Blr => false,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass_manager::MachinePass;
    use llvm2_ir::{
        AArch64Opcode, MachFunction, MachInst, MachOperand, RegClass, Signature, VReg,
    };

    fn vreg(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
    }

    fn imm(val: i64) -> MachOperand {
        MachOperand::Imm(val)
    }

    /// Build a loop with a loop-invariant instruction:
    ///
    /// ```text
    ///   bb0 (entry):
    ///     v0 = movi #10
    ///     v1 = movi #20
    ///     b bb1
    ///
    ///   bb1 (header) <---+
    ///     v2 = add v0, v1      ← loop-invariant (v0, v1 defined outside)
    ///     v3 = add v2, v4      ← NOT invariant (v4 is defined in loop)
    ///     v4 = sub v3, #1
    ///     b.cond bb2, bb1
    ///                    |
    ///   bb2 (exit):      +
    ///     ret
    /// ```
    fn make_loop_with_invariant() -> MachFunction {
        let mut func = MachFunction::new(
            "licm_test".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let bb1 = func.create_block();
        let bb2 = func.create_block();

        // bb0: define loop-invariant inputs
        let m0 = func.push_inst(MachInst::new(
            AArch64Opcode::MovI,
            vec![vreg(0), imm(10)],
        ));
        func.append_inst(bb0, m0);
        let m1 = func.push_inst(MachInst::new(
            AArch64Opcode::MovI,
            vec![vreg(1), imm(20)],
        ));
        func.append_inst(bb0, m1);
        let br0 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb0, br0);

        // bb1 (loop header):
        let add_inv = func.push_inst(MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(2), vreg(0), vreg(1)],
        ));
        func.append_inst(bb1, add_inv);
        let add_var = func.push_inst(MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(3), vreg(2), vreg(4)],
        ));
        func.append_inst(bb1, add_var);
        let sub = func.push_inst(MachInst::new(
            AArch64Opcode::SubRI,
            vec![vreg(4), vreg(3), imm(1)],
        ));
        func.append_inst(bb1, sub);
        let br1 = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb2), MachOperand::Block(bb1)],
        ));
        func.append_inst(bb1, br1);

        // bb2 (exit):
        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret);

        func.add_edge(bb0, bb1);
        func.add_edge(bb1, bb2);
        func.add_edge(bb1, bb1); // back-edge

        func
    }

    #[test]
    fn test_licm_hoists_invariant() {
        let mut func = make_loop_with_invariant();

        let mut licm = LoopInvariantCodeMotion;
        assert!(licm.run(&mut func));

        // The loop-invariant `add v0, v1` should be hoisted to bb0 (preheader).
        // bb1 should now have: add_var, sub, bcond (3 instructions instead of 4)
        let bb1 = func.block(BlockId(1));
        assert_eq!(bb1.insts.len(), 3);

        // bb0 (preheader) should now have: movi, movi, add, b (4 instructions)
        let bb0 = func.block(BlockId(0));
        assert_eq!(bb0.insts.len(), 4);

        // Verify the hoisted instruction is the add
        let hoisted = func.inst(bb0.insts[2]);
        assert_eq!(hoisted.opcode, AArch64Opcode::AddRR);
        assert_eq!(hoisted.operands[1], vreg(0));
        assert_eq!(hoisted.operands[2], vreg(1));
    }

    #[test]
    fn test_licm_no_hoist_store() {
        // Loop with a store — should NOT be hoisted.
        let mut func = MachFunction::new(
            "licm_store".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let bb1 = func.create_block();
        let bb2 = func.create_block();

        let br0 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb0, br0);

        let store = func.push_inst(MachInst::new(
            AArch64Opcode::StrRI,
            vec![vreg(0), imm(8)],
        ));
        func.append_inst(bb1, store);
        let br1 = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb2), MachOperand::Block(bb1)],
        ));
        func.append_inst(bb1, br1);

        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret);

        func.add_edge(bb0, bb1);
        func.add_edge(bb1, bb2);
        func.add_edge(bb1, bb1);

        let mut licm = LoopInvariantCodeMotion;
        assert!(!licm.run(&mut func));
    }

    #[test]
    fn test_licm_no_hoist_variant() {
        // Loop with instruction whose operand is defined in the loop.
        let mut func = MachFunction::new(
            "licm_variant".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let bb1 = func.create_block();
        let bb2 = func.create_block();

        let br0 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb0, br0);

        // v1 defined in loop, v0 = add v1, #5 depends on v1
        let sub = func.push_inst(MachInst::new(
            AArch64Opcode::SubRI,
            vec![vreg(1), vreg(1), imm(1)],
        ));
        func.append_inst(bb1, sub);
        let add = func.push_inst(MachInst::new(
            AArch64Opcode::AddRI,
            vec![vreg(0), vreg(1), imm(5)],
        ));
        func.append_inst(bb1, add);
        let br1 = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb2), MachOperand::Block(bb1)],
        ));
        func.append_inst(bb1, br1);

        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret);

        func.add_edge(bb0, bb1);
        func.add_edge(bb1, bb2);
        func.add_edge(bb1, bb1);

        let mut licm = LoopInvariantCodeMotion;
        assert!(!licm.run(&mut func));
    }

    #[test]
    fn test_licm_no_loops() {
        // No loops → LICM should be a no-op.
        let mut func = MachFunction::new(
            "no_loops".to_string(),
            Signature::new(vec![], vec![]),
        );
        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(func.entry, ret);

        let mut licm = LoopInvariantCodeMotion;
        assert!(!licm.run(&mut func));
    }

    #[test]
    fn test_licm_transitive_invariance() {
        // v0, v1 defined outside loop.
        // v2 = add v0, v1  ← invariant
        // v3 = mul v2, v0  ← also invariant (v2 and v0 are both invariant)
        // Both should be hoisted.
        let mut func = MachFunction::new(
            "licm_transitive".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let bb1 = func.create_block();
        let bb2 = func.create_block();

        let m0 = func.push_inst(MachInst::new(
            AArch64Opcode::MovI,
            vec![vreg(0), imm(10)],
        ));
        func.append_inst(bb0, m0);
        let m1 = func.push_inst(MachInst::new(
            AArch64Opcode::MovI,
            vec![vreg(1), imm(20)],
        ));
        func.append_inst(bb0, m1);
        let br0 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb0, br0);

        let add = func.push_inst(MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(2), vreg(0), vreg(1)],
        ));
        func.append_inst(bb1, add);
        let mul = func.push_inst(MachInst::new(
            AArch64Opcode::MulRR,
            vec![vreg(3), vreg(2), vreg(0)],
        ));
        func.append_inst(bb1, mul);
        let br1 = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb2), MachOperand::Block(bb1)],
        ));
        func.append_inst(bb1, br1);

        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret);

        func.add_edge(bb0, bb1);
        func.add_edge(bb1, bb2);
        func.add_edge(bb1, bb1);

        let mut licm = LoopInvariantCodeMotion;
        assert!(licm.run(&mut func));

        // Both add and mul should be hoisted.
        let bb1_block = func.block(BlockId(1));
        assert_eq!(bb1_block.insts.len(), 1); // just bcond

        // Preheader (bb0) should have: movi, movi, add, mul, b
        let bb0_block = func.block(BlockId(0));
        assert_eq!(bb0_block.insts.len(), 5);
    }

    #[test]
    fn test_licm_no_hoist_call() {
        // A BL (call) instruction should not be hoisted even if operands are invariant.
        let mut func = MachFunction::new(
            "licm_call".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let bb1 = func.create_block();
        let bb2 = func.create_block();

        let br0 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb0, br0);

        let call = func.push_inst(MachInst::new(AArch64Opcode::Bl, vec![imm(0x1000)]));
        func.append_inst(bb1, call);
        let br1 = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb2), MachOperand::Block(bb1)],
        ));
        func.append_inst(bb1, br1);

        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret);

        func.add_edge(bb0, bb1);
        func.add_edge(bb1, bb2);
        func.add_edge(bb1, bb1);

        let mut licm = LoopInvariantCodeMotion;
        assert!(!licm.run(&mut func));
    }
}
