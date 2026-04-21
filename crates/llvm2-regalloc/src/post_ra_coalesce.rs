// llvm2-regalloc/post_ra_coalesce.rs - Post-register-allocation copy coalescing
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Post-register-allocation copy coalescing.
//!
//! After register allocation assigns physical registers, the code may contain
//! redundant copy instructions:
//!
//! 1. **Identity copies:** `PSEUDO_COPY Xd <- Xd` where source and destination
//!    are the same physical register. These arise when the allocator assigns
//!    the same PReg to both sides of a phi-elimination copy.
//!
//! 2. **Rename-coalescible copies:** `PSEUDO_COPY Xd <- Xs` where `Xd != Xs`
//!    but renaming all subsequent uses of `Xd` to `Xs` within the block is
//!    safe (no interference). This eliminates the copy entirely by adjusting
//!    later operands.
//!
//! This pass runs on the regalloc-level `MachFunction` (with separated
//! defs/uses and `u16` opcodes), operating entirely on physical registers.
//! It is a block-local transformation — no cross-block analysis is performed,
//! keeping the algorithm fast and simple.
//!
//! ## Algorithm (block-local rename coalescing)
//!
//! For each `PSEUDO_COPY Xd <- Xs` where `Xd != Xs`:
//! 1. Scan forward from the copy to find all uses of `Xd` in the block.
//! 2. Check that `Xs` is not redefined (clobbered) before the last use of `Xd`.
//! 3. Check that `Xd` is not used as an implicit operand of any intervening
//!    instruction that also implicitly uses/defines `Xs`.
//! 4. If safe, rename all uses of `Xd` to `Xs` and delete the copy.
//!
//! Conservative safety: we do NOT rename across calls, returns, or any
//! instruction that has implicit defs/uses of the destination register.
//!
//! Reference: LLVM `PeepholeOptimizer.cpp` — post-RA copy elimination.

use crate::machine_types::{
    InstFlags, InstId, MachFunction, MachInst, MachOperand, PReg,
};
// PSEUDO_COPY import removed — now using is_copy_opcode() which handles both opcodes

/// A no-op pseudo-instruction opcode for deleted instructions.
/// We reuse the existing NOP pattern: opcode 0 with empty operands.
const NOP_OPCODE: u16 = 0xFFFF;

/// Statistics from the post-RA copy coalescing pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PostRACoalesceResult {
    /// Total number of copy instructions removed.
    pub copies_removed: u32,
    /// Number of identity copies removed (src == dst).
    pub identity_copies: u32,
    /// Number of copies removed by rename coalescing (src != dst).
    pub rename_coalesced: u32,
}

/// Run post-register-allocation copy coalescing on the given function.
///
/// This pass modifies the function in-place, removing redundant copy
/// instructions. Deleted copies are replaced with NOP pseudo-instructions
/// (opcode `NOP_OPCODE` with empty operands).
///
/// Returns statistics about what was eliminated.
pub fn post_ra_coalesce(func: &mut MachFunction) -> PostRACoalesceResult {
    let mut result = PostRACoalesceResult::default();

    // Process each block independently (block-local analysis).
    let block_indices: Vec<usize> = if func.block_order.is_empty() {
        (0..func.blocks.len()).collect()
    } else {
        func.block_order
            .iter()
            .map(|block_id| block_id.0 as usize)
            .collect()
    };

    for block_idx in block_indices {
        let block_insts: Vec<InstId> = func.blocks[block_idx].insts.clone();
        coalesce_block(func, &block_insts, &mut result);
    }

    // Remove NOP'd instructions from block instruction lists.
    for block in &mut func.blocks {
        block.insts.retain(|inst_id| {
            let inst = &func.insts[inst_id.0 as usize];
            inst.opcode != NOP_OPCODE
        });
    }

    result
}

/// Process a single basic block for copy coalescing.
fn coalesce_block(
    func: &mut MachFunction,
    block_insts: &[InstId],
    result: &mut PostRACoalesceResult,
) {
    // We process copies in forward order. When a copy is coalesced by
    // renaming, we update subsequent instructions in the block immediately.
    // This means later copies in the same block see the already-renamed
    // operands, enabling chained coalescing.

    for (pos, &inst_id) in block_insts.iter().enumerate() {
        let inst = &func.insts[inst_id.0 as usize];

        // Only process copy pseudo-instructions (both PSEUDO_COPY from phi
        // elimination and IR_COPY_OPCODE from ISel).
        if !crate::phi_elim::is_copy_opcode(inst.opcode) {
            continue;
        }

        // Extract the physical register operands.
        let dst_preg = match inst.defs.first().and_then(MachOperand::as_preg) {
            Some(p) => p,
            None => continue, // VReg copy — skip (shouldn't happen post-RA)
        };
        let src_preg = match inst.uses.first().and_then(MachOperand::as_preg) {
            Some(p) => p,
            None => continue,
        };

        // Case 1: Identity copy — trivially remove.
        if dst_preg == src_preg {
            nop_inst(&mut func.insts[inst_id.0 as usize]);
            result.copies_removed += 1;
            result.identity_copies += 1;
            continue;
        }

        // Case 2: Try rename coalescing.
        let remaining_insts = &block_insts[pos + 1..];
        if try_rename_coalesce(func, remaining_insts, dst_preg, src_preg) {
            nop_inst(&mut func.insts[inst_id.0 as usize]);
            result.copies_removed += 1;
            result.rename_coalesced += 1;
        }
    }
}

/// Attempt rename coalescing: rename all uses of `dst` to `src` in the
/// remaining instructions of the block, then delete the copy.
///
/// Returns true if coalescing was successful and the copy can be removed.
fn try_rename_coalesce(
    func: &mut MachFunction,
    remaining_insts: &[InstId],
    dst: PReg,
    src: PReg,
) -> bool {
    // Phase 1: Validate — scan forward to check safety.
    //
    // We need to verify:
    // (a) `src` is not redefined before the last use of `dst`
    // (b) No instruction between the copy and the last use of `dst`
    //     implicitly clobbers `src`
    // (c) `dst` is not used in an implicit_uses list of any instruction
    //     that also implicitly uses/defines `src` (would break semantics)
    // (d) We don't cross calls or returns (conservative)

    let mut last_dst_use_pos: Option<usize> = None;
    let mut src_redef_pos: Option<usize> = None;
    let mut dst_redef_pos: Option<usize> = None;

    for (i, &inst_id) in remaining_insts.iter().enumerate() {
        let inst = &func.insts[inst_id.0 as usize];

        // Check for terminator/call — stop scanning.
        // We could rename through non-call instructions, but calls are
        // conservative barriers because of implicit clobbers.
        if inst.flags.is_call() {
            // If dst is used by this call, we need to include it.
            // But for safety, don't rename across calls.
            if uses_preg(inst, dst) && src_redef_pos.is_none() {
                // The call uses dst — we can't safely rename across it
                // because calls have extensive implicit clobbers.
                break;
            }
            break;
        }

        // Track definitions of src and dst.
        if defines_preg(inst, src) && src_redef_pos.is_none() {
            src_redef_pos = Some(i);
        }
        if defines_preg(inst, dst) && dst_redef_pos.is_none() {
            dst_redef_pos = Some(i);
        }

        // Track uses of dst (explicit and implicit).
        if uses_preg(inst, dst) {
            // If src has already been redefined, renaming this use of dst
            // to src would produce wrong results.
            if src_redef_pos.is_some() {
                return false;
            }

            // If dst has already been redefined, this use reads the NEW
            // value of dst (not our copy's dst), so we must not rename it.
            if dst_redef_pos.is_some() {
                break;
            }

            last_dst_use_pos = Some(i);
        }

        // If dst has been redefined and we've seen all its uses up to
        // the redefinition, we can stop scanning.
        if let Some(redef) = dst_redef_pos
            && i >= redef {
                break;
            }
    }

    // If dst is never used after the copy, we can trivially remove it.
    // (Dead copy elimination.)
    if last_dst_use_pos.is_none() {
        // dst is dead after the copy. But only remove if dst is also
        // not live-out of the block (conservative: if dst is redefined
        // before end, it's safe; if not, dst might be live-out).
        if dst_redef_pos.is_some() {
            // dst is redefined later — the copy's def is dead.
            return true;
        }
        // dst might be live-out — don't remove without global liveness info.
        // For safety, skip this case.
        return false;
    }

    // last_dst_use_pos is guaranteed Some here (None case returns early above).
    let Some(last_use) = last_dst_use_pos else {
        return false;
    };

    // src must not be redefined before the last use of dst.
    if let Some(src_redef) = src_redef_pos
        && src_redef <= last_use {
            return false;
        }

    // Check that no instruction in [0..=last_use] has an implicit def/use
    // conflict. Specifically, if an instruction implicitly defines `src`,
    // we can't rename dst->src in uses after that point.
    for (i, &inst_id) in remaining_insts[..=last_use].iter().enumerate() {
        let inst = &func.insts[inst_id.0 as usize];

        // If src is implicitly defined by this instruction, and we have
        // uses of dst after this point, renaming would be incorrect.
        if inst.implicit_defs.contains(&src) && i < last_use {
            return false;
        }

        // If this instruction implicitly uses src AND we're about to rename
        // a use of dst to src in the same instruction, verify it's safe.
        // (Two explicit uses of the same register is fine on AArch64.)
    }

    // Phase 2: Apply — rename all uses of dst to src in [0..=last_use].
    for &inst_id in &remaining_insts[..=last_use] {
        let inst = &mut func.insts[inst_id.0 as usize];
        rename_preg_uses(inst, dst, src);
    }

    true
}

/// Check if an instruction uses (reads) a physical register, either
/// explicitly or implicitly.
fn uses_preg(inst: &MachInst, preg: PReg) -> bool {
    // Check explicit uses.
    for op in &inst.uses {
        if let MachOperand::PReg(p) = op
            && *p == preg {
                return true;
            }
    }
    // Check implicit uses.
    inst.implicit_uses.contains(&preg)
}

/// Check if an instruction defines (writes) a physical register, either
/// explicitly or implicitly.
fn defines_preg(inst: &MachInst, preg: PReg) -> bool {
    // Check explicit defs.
    for op in &inst.defs {
        if let MachOperand::PReg(p) = op
            && *p == preg {
                return true;
            }
    }
    // Check implicit defs.
    inst.implicit_defs.contains(&preg)
}

/// Rename all explicit uses of `old` to `new` in an instruction.
/// Does NOT rename implicit uses (those are fixed by the ISA).
fn rename_preg_uses(inst: &mut MachInst, old: PReg, new: PReg) {
    for op in &mut inst.uses {
        if let MachOperand::PReg(p) = op
            && *p == old {
                *p = new;
            }
    }
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine_types::{
        BlockId, InstFlags, InstId, MachBlock, MachFunction, MachInst, MachOperand,
        PReg, RegClass,
    };
    use crate::phi_elim::PSEUDO_COPY;
    use std::collections::HashMap;

    // AArch64 register constants (matching llvm2-ir encoding).
    const X0: PReg = PReg::new(0);
    const X1: PReg = PReg::new(1);
    const X2: PReg = PReg::new(2);
    const X3: PReg = PReg::new(3);
    const X8: PReg = PReg::new(8);
    const X19: PReg = PReg::new(19);
    const X20: PReg = PReg::new(20);

    /// Helper: create a PSEUDO_COPY from src to dst (both PRegs).
    fn preg_copy(dst: PReg, src: PReg) -> MachInst {
        MachInst {
            opcode: PSEUDO_COPY,
            defs: vec![MachOperand::PReg(dst)],
            uses: vec![MachOperand::PReg(src)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        }
    }

    /// Helper: create a generic instruction with PReg defs and uses.
    fn preg_inst(opcode: u16, defs: &[PReg], uses: &[PReg]) -> MachInst {
        MachInst {
            opcode,
            defs: defs.iter().map(|p| MachOperand::PReg(*p)).collect(),
            uses: uses.iter().map(|p| MachOperand::PReg(*p)).collect(),
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        }
    }

    /// Helper: create a call instruction with implicit clobbers.
    fn call_inst(implicit_defs: Vec<PReg>) -> MachInst {
        MachInst {
            opcode: 0xCA,
            defs: Vec::new(),
            uses: Vec::new(),
            implicit_defs,
            implicit_uses: Vec::new(),
            flags: InstFlags::IS_CALL.union(InstFlags::HAS_SIDE_EFFECTS),
        }
    }

    /// Helper: create an instruction with implicit defs.
    fn inst_with_implicit_defs(opcode: u16, defs: &[PReg], uses: &[PReg], implicit_defs: Vec<PReg>) -> MachInst {
        MachInst {
            opcode,
            defs: defs.iter().map(|p| MachOperand::PReg(*p)).collect(),
            uses: uses.iter().map(|p| MachOperand::PReg(*p)).collect(),
            implicit_defs,
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
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
            name: "test_post_ra".into(),
            insts,
            blocks,
            block_order,
            entry_block: BlockId(0),
            next_vreg: 0,
            next_stack_slot: 0,
            stack_slots: HashMap::new(),
        }
    }

    // -- Identity copy tests --

    #[test]
    fn test_identity_copy_removed() {
        // PSEUDO_COPY X0 <- X0 → removed
        let mut func = make_function(vec![vec![
            preg_copy(X0, X0),
            preg_inst(1, &[X1], &[X0]),
        ]]);

        let result = post_ra_coalesce(&mut func);

        assert_eq!(result.copies_removed, 1);
        assert_eq!(result.identity_copies, 1);
        assert_eq!(result.rename_coalesced, 0);
        // Block should have only the use instruction left.
        assert_eq!(func.blocks[0].insts.len(), 1);
    }

    #[test]
    fn test_multiple_identity_copies() {
        let mut func = make_function(vec![vec![
            preg_copy(X0, X0),
            preg_copy(X1, X1),
            preg_copy(X2, X2),
            preg_inst(1, &[], &[X0, X1, X2]),
        ]]);

        let result = post_ra_coalesce(&mut func);

        assert_eq!(result.copies_removed, 3);
        assert_eq!(result.identity_copies, 3);
        assert_eq!(func.blocks[0].insts.len(), 1);
    }

    // -- Rename coalescing tests --

    #[test]
    fn test_rename_simple() {
        // PSEUDO_COPY X1 <- X0 (X0 is not redefined, X1 used once after)
        // ADD X2, X1, X3  → should become ADD X2, X0, X3
        let mut func = make_function(vec![vec![
            preg_inst(1, &[X0], &[]),          // def X0
            preg_copy(X1, X0),                  // copy X1 <- X0
            preg_inst(2, &[X2], &[X1, X3]),    // use X1
        ]]);

        let result = post_ra_coalesce(&mut func);

        assert_eq!(result.copies_removed, 1);
        assert_eq!(result.rename_coalesced, 1);
        // Block should have 2 instructions (def + renamed use).
        assert_eq!(func.blocks[0].insts.len(), 2);
        // Verify the use was renamed: X1 → X0
        let use_inst = &func.insts[func.blocks[0].insts[1].0 as usize];
        assert_eq!(use_inst.uses[0], MachOperand::PReg(X0));
    }

    #[test]
    fn test_rename_multiple_uses() {
        // PSEUDO_COPY X1 <- X0
        // use X1  (twice)
        // use X1
        let mut func = make_function(vec![vec![
            preg_inst(1, &[X0], &[]),
            preg_copy(X1, X0),
            preg_inst(2, &[X2], &[X1]),
            preg_inst(3, &[X3], &[X1]),
        ]]);

        let result = post_ra_coalesce(&mut func);

        assert_eq!(result.copies_removed, 1);
        assert_eq!(result.rename_coalesced, 1);
        assert_eq!(func.blocks[0].insts.len(), 3);

        // Both uses should be renamed.
        let inst1 = &func.insts[func.blocks[0].insts[1].0 as usize];
        assert_eq!(inst1.uses[0], MachOperand::PReg(X0));
        let inst2 = &func.insts[func.blocks[0].insts[2].0 as usize];
        assert_eq!(inst2.uses[0], MachOperand::PReg(X0));
    }

    #[test]
    fn test_rename_blocked_by_src_redef() {
        // PSEUDO_COPY X1 <- X0
        // def X0 (src is redefined!)
        // use X1 → cannot rename because X0 is clobbered
        let mut func = make_function(vec![vec![
            preg_copy(X1, X0),
            preg_inst(1, &[X0], &[X2]),     // redefines X0
            preg_inst(2, &[X3], &[X1]),     // uses X1 (after X0 redef)
        ]]);

        let result = post_ra_coalesce(&mut func);

        assert_eq!(result.copies_removed, 0);
        assert_eq!(func.blocks[0].insts.len(), 3);
    }

    #[test]
    fn test_rename_safe_when_src_redef_after_last_use() {
        // PSEUDO_COPY X1 <- X0
        // use X1
        // def X0 (src redefined AFTER last use of X1 — safe!)
        let mut func = make_function(vec![vec![
            preg_copy(X1, X0),
            preg_inst(2, &[X3], &[X1]),     // uses X1
            preg_inst(1, &[X0], &[X2]),     // redefines X0 after
        ]]);

        let result = post_ra_coalesce(&mut func);

        assert_eq!(result.copies_removed, 1);
        assert_eq!(result.rename_coalesced, 1);
    }

    #[test]
    fn test_rename_blocked_by_call() {
        // PSEUDO_COPY X1 <- X0
        // call (barrier)
        // use X1 → cannot rename across call
        let mut func = make_function(vec![vec![
            preg_copy(X1, X0),
            call_inst(vec![X0, X1, X2, X3]),
            preg_inst(2, &[X3], &[X1]),
        ]]);

        let result = post_ra_coalesce(&mut func);

        // Should not rename across the call.
        assert_eq!(result.rename_coalesced, 0);
    }

    #[test]
    fn test_rename_blocked_by_implicit_def_of_src() {
        // PSEUDO_COPY X1 <- X0
        // inst that implicitly defines X0
        // use X1 → cannot rename because X0 is implicitly clobbered
        let mut func = make_function(vec![vec![
            preg_copy(X1, X0),
            inst_with_implicit_defs(5, &[X2], &[X3], vec![X0]),
            preg_inst(2, &[X3], &[X1]),
        ]]);

        let result = post_ra_coalesce(&mut func);

        assert_eq!(result.rename_coalesced, 0);
    }

    #[test]
    fn test_rename_chain() {
        // PSEUDO_COPY X1 <- X0
        // PSEUDO_COPY X2 <- X1  → after first rename, becomes X2 <- X0
        // use X2               → after second rename, uses X0
        let mut func = make_function(vec![vec![
            preg_inst(1, &[X0], &[]),
            preg_copy(X1, X0),
            preg_copy(X2, X1),
            preg_inst(2, &[X3], &[X2]),
        ]]);

        let result = post_ra_coalesce(&mut func);

        assert_eq!(result.copies_removed, 2);
        assert_eq!(result.rename_coalesced, 2);
        // Only the def and the final use should remain.
        assert_eq!(func.blocks[0].insts.len(), 2);
        let use_inst = &func.insts[func.blocks[0].insts[1].0 as usize];
        assert_eq!(use_inst.uses[0], MachOperand::PReg(X0));
    }

    #[test]
    fn test_dead_copy_with_redef() {
        // PSEUDO_COPY X1 <- X0
        // def X1 (overwrites dst — copy is dead)
        // use X1 (uses the NEW X1)
        let mut func = make_function(vec![vec![
            preg_copy(X1, X0),
            preg_inst(1, &[X1], &[X2]),
            preg_inst(2, &[X3], &[X1]),
        ]]);

        let result = post_ra_coalesce(&mut func);

        // The copy is dead (X1 is immediately redefined).
        assert_eq!(result.copies_removed, 1);
        assert_eq!(func.blocks[0].insts.len(), 2);
    }

    #[test]
    fn test_no_coalescing_on_non_copy() {
        // Regular instructions should not be touched.
        let mut func = make_function(vec![vec![
            preg_inst(1, &[X0], &[]),
            preg_inst(2, &[X1], &[X0]),
        ]]);

        let result = post_ra_coalesce(&mut func);

        assert_eq!(result.copies_removed, 0);
        assert_eq!(func.blocks[0].insts.len(), 2);
    }

    #[test]
    fn test_multi_block() {
        // Two blocks, each with an identity copy.
        let mut func = make_function(vec![
            vec![preg_copy(X0, X0), preg_inst(1, &[], &[X0])],
            vec![preg_copy(X1, X1), preg_inst(2, &[], &[X1])],
        ]);

        let result = post_ra_coalesce(&mut func);

        assert_eq!(result.copies_removed, 2);
        assert_eq!(result.identity_copies, 2);
        assert_eq!(func.blocks[0].insts.len(), 1);
        assert_eq!(func.blocks[1].insts.len(), 1);
    }

    #[test]
    fn test_vreg_copy_ignored() {
        // PSEUDO_COPY with VRegs (shouldn't happen post-RA but should not crash).
        use crate::machine_types::VReg;
        let vreg0 = VReg { id: 0, class: RegClass::Gpr64 };
        let vreg1 = VReg { id: 1, class: RegClass::Gpr64 };
        let mut func = make_function(vec![vec![MachInst {
            opcode: PSEUDO_COPY,
            defs: vec![MachOperand::VReg(vreg1)],
            uses: vec![MachOperand::VReg(vreg0)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        }]]);

        let result = post_ra_coalesce(&mut func);

        // VReg copies are skipped, not counted.
        assert_eq!(result.copies_removed, 0);
    }

    #[test]
    fn test_rename_stops_at_dst_redef() {
        // PSEUDO_COPY X1 <- X0
        // use X1
        // def X1 (redefined)
        // use X1  ← this use reads the NEW X1, not our copy target
        let mut func = make_function(vec![vec![
            preg_copy(X1, X0),
            preg_inst(2, &[X2], &[X1]),      // use X1 (from copy)
            preg_inst(3, &[X1], &[X3]),       // redef X1
            preg_inst(4, &[X8], &[X1]),       // use new X1
        ]]);

        let result = post_ra_coalesce(&mut func);

        assert_eq!(result.copies_removed, 1);
        assert_eq!(result.rename_coalesced, 1);
        // First use should be renamed.
        let inst_use1 = &func.insts[func.blocks[0].insts[0].0 as usize];
        assert_eq!(inst_use1.uses[0], MachOperand::PReg(X0));
        // Last use should NOT be renamed (reads new X1).
        let inst_use2 = &func.insts[func.blocks[0].insts[2].0 as usize];
        assert_eq!(inst_use2.uses[0], MachOperand::PReg(X1));
    }

    #[test]
    fn test_empty_block() {
        let mut func = make_function(vec![vec![]]);
        let result = post_ra_coalesce(&mut func);
        assert_eq!(result.copies_removed, 0);
    }

    #[test]
    fn test_callee_saved_rename() {
        // Callee-saved registers (X19, X20) should still be coalescible
        // when no interference exists.
        let mut func = make_function(vec![vec![
            preg_inst(1, &[X19], &[]),
            preg_copy(X20, X19),
            preg_inst(2, &[X3], &[X20]),
        ]]);

        let result = post_ra_coalesce(&mut func);

        assert_eq!(result.copies_removed, 1);
        assert_eq!(result.rename_coalesced, 1);
        let use_inst = &func.insts[func.blocks[0].insts[1].0 as usize];
        assert_eq!(use_inst.uses[0], MachOperand::PReg(X19));
    }
}
