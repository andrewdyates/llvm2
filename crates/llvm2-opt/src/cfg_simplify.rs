// llvm2-opt - CFG Simplification
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! CFG simplification / branch folding pass for machine-level IR.
//!
//! Simplifies the control flow graph by eliminating unnecessary branches
//! and blocks. This is a pre-register-allocation pass that runs after DCE.
//!
//! # Transforms
//!
//! | Transform | Description |
//! |-----------|-------------|
//! | Unreachable block removal | Remove blocks not reachable from entry |
//! | Empty block elimination | Redirect predecessors past single-branch blocks |
//! | Branch target simplification | Thread branches through single-jump blocks |
//! | Unconditional branch folding | Merge block into sole predecessor |
//! | Constant branch folding | Convert known-constant conditionals to unconditional |
//! | Duplicate branch elimination | Convert same-target conditionals to unconditional |
//!
//! # Algorithm
//!
//! All six sub-passes are iterated in a single `run()` invocation until no
//! sub-pass reports a change (local fixed point). After any structural CFG
//! modification the predecessor/successor lists are rebuilt from scratch by
//! scanning every terminator for `Block` operands.
//!
//! Reference: LLVM `SimplifyCFG`, `BranchFolding`

use std::collections::{HashMap, HashSet, VecDeque};

use llvm2_ir::{AArch64Opcode, BlockId, InstId, MachFunction, MachInst, MachOperand};

use crate::pass_manager::MachinePass;

/// CFG simplification pass.
pub struct CfgSimplify;

impl MachinePass for CfgSimplify {
    fn name(&self) -> &str {
        "cfg-simplify"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        let mut ever_changed = false;
        // Safety bound to prevent infinite loops if sub-passes oscillate.
        let max_iterations: usize = 32;

        // Iterate sub-passes until fixed point (bounded).
        for _ in 0..max_iterations {
            let mut changed = false;

            rebuild_cfg_edges(func);

            changed |= remove_unreachable_blocks(func);
            rebuild_cfg_edges(func);

            changed |= eliminate_empty_blocks(func);
            rebuild_cfg_edges(func);

            changed |= simplify_branch_targets(func);
            rebuild_cfg_edges(func);

            changed |= fold_unconditional_branches(func);
            rebuild_cfg_edges(func);

            changed |= fold_constant_branches(func);
            rebuild_cfg_edges(func);

            changed |= eliminate_duplicate_branches(func);
            rebuild_cfg_edges(func);

            if changed {
                ever_changed = true;
            } else {
                break;
            }
        }

        ever_changed
    }
}

// ---------------------------------------------------------------------------
// CFG edge reconstruction
// ---------------------------------------------------------------------------

/// Rebuild all predecessor/successor lists from scratch by scanning
/// every block's terminator instructions for `Block` operands and
/// inferring fallthrough edges for conditional branches.
fn rebuild_cfg_edges(func: &mut MachFunction) {
    // Clear all edges.
    for block in &mut func.blocks {
        block.preds.clear();
        block.succs.clear();
    }

    // First pass: collect all edges (block_id → target) to avoid borrow conflicts.
    let mut edges: Vec<(BlockId, BlockId)> = Vec::new();
    let order = func.block_order.clone();
    for (layout_idx, &block_id) in order.iter().enumerate() {
        let block = &func.blocks[block_id.0 as usize];
        let Some(&last_inst_id) = block.insts.last() else {
            continue;
        };
        let last_inst = &func.insts[last_inst_id.0 as usize];

        // Collect explicit Block targets from terminators.
        let mut has_explicit_target = false;
        let mut is_unconditional_jump = false;

        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.0 as usize];
            if !inst.is_branch() && !inst.is_terminator() {
                continue;
            }
            for operand in &inst.operands {
                if let MachOperand::Block(target) = operand {
                    edges.push((block_id, *target));
                    has_explicit_target = true;
                }
            }
        }

        // Determine if the block falls through to the next in layout.
        // Unconditional branches (B, Br) and returns (Ret) do NOT fall through.
        // Conditional branches (BCond, Cbz, Cbnz, Tbz, Tbnz) DO fall through.
        if last_inst.is_unconditional_branch() || last_inst.is_return() {
            is_unconditional_jump = true;
        }

        if !is_unconditional_jump && has_explicit_target {
            // Conditional branch: add fallthrough edge to next block in layout.
            if let Some(&next_block) = order.get(layout_idx + 1) {
                edges.push((block_id, next_block));
            }
        }
    }

    // Second pass: apply edges, deduplicating.
    for (from, to) in edges {
        if !func.blocks[from.0 as usize].succs.contains(&to) {
            func.blocks[from.0 as usize].succs.push(to);
            func.blocks[to.0 as usize].preds.push(from);
        }
    }
}

// ---------------------------------------------------------------------------
// 1. Unreachable block removal
// ---------------------------------------------------------------------------

/// Remove blocks that are not reachable from the entry block.
fn remove_unreachable_blocks(func: &mut MachFunction) -> bool {
    let reachable = compute_reachable(func);

    let before = func.block_order.len();
    func.block_order.retain(|bid| reachable.contains(bid));
    let after = func.block_order.len();

    before != after
}

/// BFS from entry to find all reachable block IDs.
fn compute_reachable(func: &MachFunction) -> HashSet<BlockId> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    visited.insert(func.entry);
    queue.push_back(func.entry);

    while let Some(bid) = queue.pop_front() {
        let block = func.block(bid);
        for &succ in &block.succs {
            if visited.insert(succ) {
                queue.push_back(succ);
            }
        }
    }

    visited
}

// ---------------------------------------------------------------------------
// 2. Empty block elimination
// ---------------------------------------------------------------------------

/// If a block contains exactly one instruction — an unconditional `B` — then
/// redirect all predecessors to the target and remove the block from layout.
fn eliminate_empty_blocks(func: &mut MachFunction) -> bool {
    let mut changed = false;

    // Collect candidates: (empty_block, target).
    let candidates: Vec<(BlockId, BlockId)> = func
        .block_order
        .iter()
        .filter_map(|&bid| {
            // Never eliminate the entry block.
            if bid == func.entry {
                return None;
            }
            let block = func.block(bid);
            if block.insts.len() != 1 {
                return None;
            }
            let inst = func.inst(block.insts[0]);
            if !inst.is_unconditional_branch() {
                return None;
            }
            // Extract target.
            for op in &inst.operands {
                if let MachOperand::Block(target) = op {
                    // Don't create self-loops.
                    if *target != bid {
                        return Some((bid, *target));
                    }
                }
            }
            None
        })
        .collect();

    for (empty_bid, target) in &candidates {
        // Redirect all branch operands that point to empty_bid → target.
        redirect_branches(func, *empty_bid, *target);
        changed = true;
    }

    // Remove eliminated blocks from layout.
    if changed {
        let removed: HashSet<BlockId> = candidates.iter().map(|(b, _)| *b).collect();
        func.block_order.retain(|b| !removed.contains(b));
    }

    changed
}

// ---------------------------------------------------------------------------
// 3. Branch target simplification (thread-through)
// ---------------------------------------------------------------------------

/// If a branch targets a block that contains only a single unconditional `B`,
/// rewrite the branch target to the final destination (thread through).
fn simplify_branch_targets(func: &mut MachFunction) -> bool {
    // Build a map of single-jump blocks: block → final target.
    let mut jump_map: HashMap<BlockId, BlockId> = HashMap::new();
    for &bid in &func.block_order {
        let block = func.block(bid);
        if block.insts.len() == 1 {
            let inst = func.inst(block.insts[0]);
            if inst.is_unconditional_branch() {
                for op in &inst.operands {
                    if let MachOperand::Block(target) = op
                        && *target != bid {
                            jump_map.insert(bid, *target);
                        }
                }
            }
        }
    }

    if jump_map.is_empty() {
        return false;
    }

    // Resolve chains: if A→B→C, resolve A→C.
    let resolved = resolve_chains(&jump_map);

    let mut changed = false;

    // Collect (inst_id, new_operands) pairs to apply after scanning.
    let mut rewrites: Vec<(InstId, Vec<MachOperand>)> = Vec::new();

    for &bid in &func.block_order {
        let block = &func.blocks[bid.0 as usize];
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.0 as usize];
            if !inst.is_branch() && !inst.is_terminator() {
                continue;
            }
            let mut inst_changed = false;
            let new_operands: Vec<MachOperand> = inst
                .operands
                .iter()
                .map(|op| {
                    if let MachOperand::Block(target) = op
                        && let Some(&final_target) = resolved.get(target) {
                            inst_changed = true;
                            return MachOperand::Block(final_target);
                        }
                    op.clone()
                })
                .collect();
            if inst_changed {
                rewrites.push((inst_id, new_operands));
            }
        }
    }

    // Apply rewrites.
    for (inst_id, new_ops) in rewrites {
        func.insts[inst_id.0 as usize].operands = new_ops;
        changed = true;
    }

    changed
}

/// Resolve chains in the jump map: if A→B and B→C, produce A→C.
/// Limit chain length to prevent infinite loops on cycles.
fn resolve_chains(jump_map: &HashMap<BlockId, BlockId>) -> HashMap<BlockId, BlockId> {
    let mut resolved = HashMap::new();
    for &src in jump_map.keys() {
        let mut target = src;
        let mut depth = 0;
        while let Some(&next) = jump_map.get(&target) {
            if depth > 32 || next == src {
                break;
            }
            target = next;
            depth += 1;
        }
        if target != src {
            resolved.insert(src, target);
        }
    }
    resolved
}

// ---------------------------------------------------------------------------
// 4. Unconditional branch folding (merge single-predecessor blocks)
// ---------------------------------------------------------------------------

/// If block A ends with an unconditional `B` to block B, and B has exactly
/// one predecessor (A), then merge B's instructions into A (removing the
/// trailing `B` from A).
fn fold_unconditional_branches(func: &mut MachFunction) -> bool {
    let mut changed = false;

    // We may need multiple passes since merging can create new opportunities.
    // But the outer loop in run() handles this; do one pass here.
    let order = func.block_order.clone();
    let mut merged_away: HashSet<BlockId> = HashSet::new();

    for &block_a in &order {
        if merged_away.contains(&block_a) {
            continue;
        }

        let block = func.block(block_a);
        let Some(&last_inst_id) = block.insts.last() else {
            continue;
        };

        // Check if last instruction is unconditional B.
        let last_inst = func.inst(last_inst_id);
        if !last_inst.is_unconditional_branch() {
            continue;
        }

        // Extract target block.
        let target = match last_inst.operands.iter().find_map(|op| {
            if let MachOperand::Block(bid) = op {
                Some(*bid)
            } else {
                None
            }
        }) {
            Some(t) => t,
            None => continue,
        };

        // Don't merge a block into itself.
        if target == block_a {
            continue;
        }

        // Don't merge the entry block away.
        if target == func.entry {
            continue;
        }

        // Target must have exactly one predecessor (block_a).
        let target_preds = func.block(target).preds.clone();
        if target_preds.len() != 1 || target_preds[0] != block_a {
            continue;
        }

        // Merge: remove trailing B from A, append B's instructions to A.
        let target_insts = func.block(target).insts.clone();
        let block_a_mut = func.block_mut(block_a);
        block_a_mut.insts.pop(); // Remove trailing B
        block_a_mut.insts.extend(target_insts);

        // Clear target block's instructions.
        func.block_mut(target).insts.clear();

        merged_away.insert(target);
        changed = true;
    }

    // Remove merged blocks from layout.
    if changed {
        func.block_order.retain(|b| !merged_away.contains(b));
    }

    changed
}

// ---------------------------------------------------------------------------
// 5. Constant branch folding
// ---------------------------------------------------------------------------

/// If a Cbz/Cbnz instruction's condition register is defined by a MovI with
/// a known constant, convert the conditional branch to an unconditional B.
fn fold_constant_branches(func: &mut MachFunction) -> bool {
    let mut changed = false;

    // Build map: vreg_id → constant value (from MovI).
    let mut constants: HashMap<u32, i64> = HashMap::new();
    for &bid in &func.block_order {
        let block = func.block(bid);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            if inst.is_move_imm()
                && let (Some(MachOperand::VReg(dst)), Some(MachOperand::Imm(val))) =
                    (inst.operands.first(), inst.operands.get(1))
                {
                    constants.insert(dst.id, *val);
                }
        }
    }

    // Scan for Cbz/Cbnz with constant conditions.
    for &bid in &func.block_order.clone() {
        let block = func.block(bid);
        let Some(&last_inst_id) = block.insts.last() else {
            continue;
        };
        let inst = func.inst(last_inst_id);

        // Use generic opcode queries for dispatch; AArch64Opcode::B is
        // still used to construct the replacement unconditional branch.
        let is_cbz = inst.opcode.is_cbz();
        let is_cbnz = inst.opcode.is_cbnz();
        if !is_cbz && !is_cbnz {
            continue;
        }

        if let Some(MachOperand::VReg(cond)) = inst.operands.first()
            && let Some(&val) = constants.get(&cond.id)
                && let Some(target) = find_block_operand(&inst.operands) {
                    // Determine if the branch is taken based on the constant value.
                    let branch_taken = (is_cbz && val == 0) || (is_cbnz && val != 0);

                    if branch_taken {
                        // Branch IS taken: convert to unconditional B to target.
                        *func.inst_mut(last_inst_id) = MachInst::new(
                            AArch64Opcode::B,
                            vec![MachOperand::Block(target)],
                        );
                    } else {
                        // Branch NOT taken: convert to unconditional B to fallthrough.
                        if let Some(fallthrough) = get_fallthrough(func, bid) {
                            *func.inst_mut(last_inst_id) = MachInst::new(
                                AArch64Opcode::B,
                                vec![MachOperand::Block(fallthrough)],
                            );
                        } else {
                            continue;
                        }
                    }
                    changed = true;
                }
    }

    changed
}

/// Find the first `Block` operand in an operand list.
fn find_block_operand(operands: &[MachOperand]) -> Option<BlockId> {
    operands.iter().find_map(|op| {
        if let MachOperand::Block(bid) = op {
            Some(*bid)
        } else {
            None
        }
    })
}

/// Get the fallthrough block (next in layout order) for a given block.
fn get_fallthrough(func: &MachFunction, bid: BlockId) -> Option<BlockId> {
    let pos = func.block_order.iter().position(|b| *b == bid)?;
    func.block_order.get(pos + 1).copied()
}

// ---------------------------------------------------------------------------
// 6. Duplicate branch elimination
// ---------------------------------------------------------------------------

/// If a conditional branch (BCond, Cbz, Cbnz, Tbz, Tbnz) targets the same
/// block for both taken and not-taken (fallthrough), convert to unconditional B.
fn eliminate_duplicate_branches(func: &mut MachFunction) -> bool {
    let mut changed = false;

    for &bid in &func.block_order.clone() {
        let block = func.block(bid);
        let Some(&last_inst_id) = block.insts.last() else {
            continue;
        };
        let inst = func.inst(last_inst_id);

        // Only applies to conditional branches.
        if !inst.is_branch() {
            continue;
        }
        if !inst.is_conditional_branch() {
            continue;
        }

        // Get the taken target (Block operand).
        let taken = match find_block_operand(&inst.operands) {
            Some(t) => t,
            None => continue,
        };

        // Get the fallthrough (not-taken) target.
        let fallthrough = match get_fallthrough(func, bid) {
            Some(ft) => ft,
            None => continue,
        };

        if taken == fallthrough {
            *func.inst_mut(last_inst_id) = MachInst::new(
                AArch64Opcode::B,
                vec![MachOperand::Block(taken)],
            );
            changed = true;
        }
    }

    changed
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Redirect all branch operands across the function that target `from` to
/// instead target `to`.
fn redirect_branches(func: &mut MachFunction, from: BlockId, to: BlockId) {
    for &bid in &func.block_order.clone() {
        let block = func.block(bid);
        let inst_ids: Vec<InstId> = block.insts.clone();
        for &inst_id in &inst_ids {
            let inst = func.inst(inst_id);
            if !inst.is_branch() && !inst.is_terminator() {
                continue;
            }
            let mut rewritten = false;
            let new_ops: Vec<MachOperand> = inst
                .operands
                .iter()
                .map(|op| {
                    if let MachOperand::Block(target) = op
                        && *target == from {
                            rewritten = true;
                            return MachOperand::Block(to);
                        }
                    op.clone()
                })
                .collect();
            if rewritten {
                func.inst_mut(inst_id).operands = new_ops;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass_manager::MachinePass;
    use llvm2_ir::{
        AArch64Opcode, BlockId, MachFunction, MachInst, MachOperand, RegClass, Signature, VReg,
    };

    fn vreg(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
    }

    fn imm(val: i64) -> MachOperand {
        MachOperand::Imm(val)
    }

    fn block(id: u32) -> MachOperand {
        MachOperand::Block(BlockId(id))
    }

    fn empty_func() -> MachFunction {
        MachFunction::new(
            "test_cfg".to_string(),
            Signature::new(vec![], vec![]),
        )
    }

    // ---- Unconditional branch folding ----

    #[test]
    fn test_uncond_branch_folding_single_pred() {
        // bb0: B bb1
        // bb1: ret   (single pred = bb0)
        // After: bb0: ret  (bb1 merged into bb0)
        let mut func = empty_func();
        let bb1 = func.create_block();

        let b_inst = func.push_inst(MachInst::new(AArch64Opcode::B, vec![block(1)]));
        func.append_inst(BlockId(0), b_inst);

        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb1, ret);

        func.add_edge(BlockId(0), bb1);

        let mut pass = CfgSimplify;
        assert!(pass.run(&mut func));

        // bb0 should now contain ret directly (bb1 merged in).
        let bb0 = func.block(BlockId(0));
        assert_eq!(bb0.insts.len(), 1);
        assert_eq!(func.inst(bb0.insts[0]).opcode, AArch64Opcode::Ret);

        // bb1 should be gone from layout.
        assert!(!func.block_order.contains(&bb1));
    }

    // ---- Empty block elimination ----

    #[test]
    fn test_empty_block_elimination() {
        // bb0: B bb1
        // bb1: B bb2   (empty: only an unconditional branch)
        // bb2: ret
        // After: bb0: B bb2, bb2: ret  (bb1 eliminated)
        let mut func = empty_func();
        let bb1 = func.create_block();
        let bb2 = func.create_block();

        let b0 = func.push_inst(MachInst::new(AArch64Opcode::B, vec![block(1)]));
        func.append_inst(BlockId(0), b0);

        let b1 = func.push_inst(MachInst::new(AArch64Opcode::B, vec![block(2)]));
        func.append_inst(bb1, b1);

        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret);

        func.add_edge(BlockId(0), bb1);
        func.add_edge(bb1, bb2);

        let mut pass = CfgSimplify;
        assert!(pass.run(&mut func));

        // After simplification, bb0 should reach bb2 (bb1 eliminated or threaded).
        // bb1 should be gone from layout.
        assert!(!func.block_order.contains(&bb1));

        // bb0's branch should target bb2 (via thread or merge).
        let bb0 = func.block(BlockId(0));
        // bb0 may have been merged with bb2 (only ret left) since bb1 was eliminated
        // and bb0→bb2 with single pred.
        let last = func.inst(*bb0.insts.last().unwrap());
        assert!(
            last.opcode == AArch64Opcode::Ret
                || (last.opcode == AArch64Opcode::B
                    && last.operands.contains(&MachOperand::Block(bb2)))
        );
    }

    // ---- Unreachable block removal ----

    #[test]
    fn test_unreachable_block_removal() {
        // bb0: ret
        // bb1: ret   (unreachable — no edges to bb1)
        let mut func = empty_func();
        let bb1 = func.create_block();

        let ret0 = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(BlockId(0), ret0);

        let ret1 = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb1, ret1);

        // No edge from bb0 to bb1.

        let mut pass = CfgSimplify;
        assert!(pass.run(&mut func));

        // bb1 should be removed from layout.
        assert!(!func.block_order.contains(&bb1));
        assert_eq!(func.block_order.len(), 1);
        assert_eq!(func.block_order[0], BlockId(0));
    }

    // ---- Branch target simplification (thread-through) ----

    #[test]
    fn test_branch_target_simplification() {
        // Layout: [bb0, bb3, bb1, bb2]
        // bb0: cbz v0, bb1   (fallthrough = bb3)
        // bb3: ret            (fallthrough target)
        // bb1: B bb2          (single jump block, not fallthrough)
        // bb2: ret
        // After thread-through: bb0 cbz target rewritten from bb1 → bb2.
        // bb1 becomes unreachable and is removed.
        let mut func = empty_func();
        let bb1 = func.create_block();  // BlockId(1)
        let bb2 = func.create_block();  // BlockId(2)
        let bb3 = func.create_block();  // BlockId(3)

        // Reorder layout: [bb0, bb3, bb1, bb2]
        func.block_order = vec![BlockId(0), bb3, bb1, bb2];

        let cbz = func.push_inst(MachInst::new(
            AArch64Opcode::Cbz,
            vec![vreg(0), block(bb1.0)],
        ));
        func.append_inst(BlockId(0), cbz);

        let ret3 = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb3, ret3);

        let b1 = func.push_inst(MachInst::new(AArch64Opcode::B, vec![block(bb2.0)]));
        func.append_inst(bb1, b1);

        let ret2 = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret2);

        func.add_edge(BlockId(0), bb1);
        func.add_edge(BlockId(0), bb3);
        func.add_edge(bb1, bb2);

        let mut pass = CfgSimplify;
        assert!(pass.run(&mut func));

        // bb1 (the single-jump block) should be eliminated.
        assert!(!func.block_order.contains(&bb1));

        // bb0's cbz should now reference bb2 directly.
        let bb0 = func.block(BlockId(0));
        let last = func.inst(*bb0.insts.last().unwrap());
        let has_bb2 = last.operands.contains(&MachOperand::Block(bb2));
        assert!(has_bb2, "branch target should be threaded to bb2");
    }

    // ---- Duplicate branch elimination ----

    #[test]
    fn test_duplicate_branch_elim() {
        // bb0: cbz v0, bb1   (fallthrough = bb1 too)
        // bb1: ret
        // Both arms go to bb1 → convert to B bb1 → merge bb1 into bb0.
        // Final: bb0 ends with Ret.
        let mut func = empty_func();
        let bb1 = func.create_block();

        let cbz = func.push_inst(MachInst::new(
            AArch64Opcode::Cbz,
            vec![vreg(0), block(1)],
        ));
        func.append_inst(BlockId(0), cbz);

        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb1, ret);

        func.add_edge(BlockId(0), bb1);

        let mut pass = CfgSimplify;
        assert!(pass.run(&mut func));

        // After dup branch elim (cbz→B) and branch folding (merge bb1 into bb0),
        // bb0 ends with Ret and bb1 is removed.
        let bb0 = func.block(BlockId(0));
        let last = func.inst(*bb0.insts.last().unwrap());
        assert_eq!(last.opcode, AArch64Opcode::Ret);
        assert!(!func.block_order.contains(&bb1));
    }

    // ---- Constant branch folding ----

    #[test]
    fn test_constant_branch_fold_cbz_taken() {
        // bb0: v0 = movi #0; cbz v0, bb2
        // bb1: ret (fallthrough)
        // bb2: ret (target)
        // v0 == 0 → branch taken → B bb2 → merge bb2 into bb0.
        // Final: bb0 has [movi, ret], bb1 unreachable and removed.
        let mut func = empty_func();
        let bb1 = func.create_block();
        let bb2 = func.create_block();

        let movi = func.push_inst(MachInst::new(
            AArch64Opcode::MovI,
            vec![vreg(0), imm(0)],
        ));
        func.append_inst(BlockId(0), movi);

        let cbz = func.push_inst(MachInst::new(
            AArch64Opcode::Cbz,
            vec![vreg(0), block(2)],
        ));
        func.append_inst(BlockId(0), cbz);

        let ret1 = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb1, ret1);

        let ret2 = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret2);

        func.add_edge(BlockId(0), bb1);
        func.add_edge(BlockId(0), bb2);

        let mut pass = CfgSimplify;
        assert!(pass.run(&mut func));

        // After constant fold + merge, bb0 ends with Ret.
        let bb0 = func.block(BlockId(0));
        let last = func.inst(*bb0.insts.last().unwrap());
        assert_eq!(last.opcode, AArch64Opcode::Ret);
        // bb1 becomes unreachable and is removed.
        assert!(!func.block_order.contains(&bb1));
    }

    #[test]
    fn test_constant_branch_fold_cbz_not_taken() {
        // bb0: v0 = movi #42; cbz v0, bb2
        // bb1: ret (fallthrough)
        // bb2: ret (target)
        // v0 != 0 → branch NOT taken → B bb1 → merge bb1 into bb0.
        // Final: bb0 has [movi, ret], bb2 unreachable and removed.
        let mut func = empty_func();
        let bb1 = func.create_block();
        let bb2 = func.create_block();

        let movi = func.push_inst(MachInst::new(
            AArch64Opcode::MovI,
            vec![vreg(0), imm(42)],
        ));
        func.append_inst(BlockId(0), movi);

        let cbz = func.push_inst(MachInst::new(
            AArch64Opcode::Cbz,
            vec![vreg(0), block(2)],
        ));
        func.append_inst(BlockId(0), cbz);

        let ret1 = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb1, ret1);

        let ret2 = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret2);

        func.add_edge(BlockId(0), bb1);
        func.add_edge(BlockId(0), bb2);

        let mut pass = CfgSimplify;
        assert!(pass.run(&mut func));

        // After constant fold + merge, bb0 ends with Ret.
        let bb0 = func.block(BlockId(0));
        let last = func.inst(*bb0.insts.last().unwrap());
        assert_eq!(last.opcode, AArch64Opcode::Ret);
        // bb2 becomes unreachable and is removed.
        assert!(!func.block_order.contains(&bb2));
    }

    // ---- Idempotent ----

    #[test]
    fn test_idempotent() {
        // After running once, running again should produce no changes.
        let mut func = empty_func();
        let bb1 = func.create_block();

        let b = func.push_inst(MachInst::new(AArch64Opcode::B, vec![block(1)]));
        func.append_inst(BlockId(0), b);

        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb1, ret);

        func.add_edge(BlockId(0), bb1);

        let mut pass = CfgSimplify;
        assert!(pass.run(&mut func));  // First pass: merges bb1 into bb0
        assert!(!pass.run(&mut func)); // Second pass: nothing to do
    }

    // ---- Entry block preservation ----

    #[test]
    fn test_entry_block_preserved() {
        // Even with no branches, the entry block must not be removed.
        let mut func = empty_func();
        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(BlockId(0), ret);

        let mut pass = CfgSimplify;
        assert!(!pass.run(&mut func));

        assert!(func.block_order.contains(&BlockId(0)));
    }

    // ---- No changes on already-simplified function ----

    #[test]
    fn test_no_changes_simple_function() {
        let mut func = empty_func();
        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(BlockId(0), ret);

        let mut pass = CfgSimplify;
        assert!(!pass.run(&mut func));
    }

    // ---- Constant branch folding: cbnz ----

    #[test]
    fn test_constant_branch_fold_cbnz_taken() {
        // v0 = movi #5; cbnz v0, bb2  → v0 != 0, taken → B bb2 → merge.
        // Final: bb0 has [movi, ret], bb1 unreachable.
        let mut func = empty_func();
        let bb1 = func.create_block();
        let bb2 = func.create_block();

        let movi = func.push_inst(MachInst::new(
            AArch64Opcode::MovI,
            vec![vreg(0), imm(5)],
        ));
        func.append_inst(BlockId(0), movi);

        let cbnz = func.push_inst(MachInst::new(
            AArch64Opcode::Cbnz,
            vec![vreg(0), block(2)],
        ));
        func.append_inst(BlockId(0), cbnz);

        let ret1 = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb1, ret1);

        let ret2 = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret2);

        func.add_edge(BlockId(0), bb1);
        func.add_edge(BlockId(0), bb2);

        let mut pass = CfgSimplify;
        assert!(pass.run(&mut func));

        // After fold + merge, bb0 ends with Ret.
        let bb0 = func.block(BlockId(0));
        let last = func.inst(*bb0.insts.last().unwrap());
        assert_eq!(last.opcode, AArch64Opcode::Ret);
        assert!(!func.block_order.contains(&bb1));
    }
}
