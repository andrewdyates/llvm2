// llvm2-codegen/layout.rs - Basic block layout pass
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Basic block layout pass for AArch64.
//!
//! Orders basic blocks to minimize taken branches. Uses a greedy chain-based
//! algorithm: starting from the entry block, greedily place the best
//! fall-through successor next. Unreachable blocks are placed at the end.
//!
//! Reference: LLVM's MachineBlockPlacement (simplified).

use llvm2_ir::{AArch64Opcode, BlockId, MachFunction, MachInst, MachOperand};

// ---------------------------------------------------------------------------
// Block layout
// ---------------------------------------------------------------------------

/// Compute block layout order for a MachFunction.
///
/// Updates `func.block_order` in place. The entry block is always first.
/// The algorithm greedily chains blocks by fall-through affinity:
/// - For a block ending in a conditional branch (BCond/Cbz/Cbnz/Tbz/Tbnz)
///   without a trailing unconditional B, the implicit fall-through successor
///   is preferred as the next block.
/// - For a block ending in conditional + unconditional B, the non-conditional
///   successor (the one NOT targeted by the conditional branch) is preferred
///   as fall-through.
/// - Blocks with no predecessors (other than entry) are placed last.
pub fn compute_block_layout(func: &mut MachFunction) {
    let num_blocks = func.blocks.len();
    if num_blocks <= 1 {
        return;
    }

    let mut placed = vec![false; num_blocks];
    let mut order = Vec::with_capacity(num_blocks);

    // Always start with the entry block.
    let entry = func.entry;
    placed[entry.0 as usize] = true;
    order.push(entry);

    // Greedy chain construction: for the last placed block, find the best
    // unplaced successor to place next.
    let mut current = entry;
    loop {
        let next = pick_best_successor(func, current, &placed);
        match next {
            Some(succ) => {
                placed[succ.0 as usize] = true;
                order.push(succ);
                current = succ;
            }
            None => {
                // No unplaced successor found for current chain.
                // Start a new chain from any unplaced block that has placed
                // predecessors (prefer blocks with in-edges from placed blocks).
                if let Some(new_head) = find_next_chain_head(func, &placed) {
                    placed[new_head.0 as usize] = true;
                    order.push(new_head);
                    current = new_head;
                } else {
                    break;
                }
            }
        }
    }

    // Append any remaining unplaced blocks (unreachable blocks).
    for i in 0..num_blocks {
        if !placed[i] {
            order.push(BlockId(i as u32));
        }
    }

    func.block_order = order;
}

/// Pick the best unplaced successor for `block` to be placed as fall-through.
///
/// Prefers the fall-through path: for a conditional branch block, the
/// successor that would be the implicit fall-through if placed next.
fn pick_best_successor(
    func: &MachFunction,
    block: BlockId,
    placed: &[bool],
) -> Option<BlockId> {
    let blk = func.block(block);
    if blk.succs.is_empty() {
        return None;
    }

    // Determine fall-through preference from the terminator pattern.
    let ft = get_fallthrough_successor(func, block);

    // Prefer the fall-through successor if it's unplaced.
    if let Some(ft_block) = ft
        && !placed[ft_block.0 as usize] {
            return Some(ft_block);
        }

    // Otherwise pick the first unplaced successor.
    blk.succs.iter().find(|&&succ| !placed[succ.0 as usize]).copied()
}

/// Find the next chain head: an unplaced block that has at least one placed
/// predecessor. If no such block exists, return any unplaced block.
fn find_next_chain_head(func: &MachFunction, placed: &[bool]) -> Option<BlockId> {
    let mut any_unplaced = None;

    for i in 0..func.blocks.len() {
        if placed[i] {
            continue;
        }
        if any_unplaced.is_none() {
            any_unplaced = Some(BlockId(i as u32));
        }
        // Prefer blocks reachable from placed blocks.
        let blk = &func.blocks[i];
        for &pred in &blk.preds {
            if placed[pred.0 as usize] {
                return Some(BlockId(i as u32));
            }
        }
    }

    any_unplaced
}

/// Returns the last instruction in a block, if any.
pub fn get_block_terminator(func: &MachFunction, block: BlockId) -> Option<&MachInst> {
    let blk = func.block(block);
    blk.insts.last().map(|&id| func.inst(id))
}

/// Returns the second-to-last instruction in a block, if it exists.
fn get_second_to_last(func: &MachFunction, block: BlockId) -> Option<&MachInst> {
    let blk = func.block(block);
    if blk.insts.len() >= 2 {
        Some(func.inst(blk.insts[blk.insts.len() - 2]))
    } else {
        None
    }
}

/// Determine the fall-through successor for a block based on its terminator.
///
/// - `BCond target` + `B fallthrough_target`: the fall-through is the B
///   target, because the conditional branch goes to `target` and the code
///   falls through to the unconditional B which jumps to `fallthrough_target`.
///   When we lay out `fallthrough_target` right after this block, we can
///   eliminate the trailing `B`.
/// - `BCond target` (no trailing B): fall-through is the non-conditional
///   successor. We look at the block's succs and return the one that is NOT
///   the conditional target.
/// - `B target` (unconditional only): no fall-through.
/// - `Ret` / `Br`: no fall-through.
/// - Same logic for Cbz, Cbnz, Tbz, Tbnz.
pub fn get_fallthrough_successor(
    func: &MachFunction,
    block: BlockId,
) -> Option<BlockId> {
    let term = get_block_terminator(func, block)?;

    match term.opcode {
        // Unconditional branch or return: no fall-through.
        AArch64Opcode::B | AArch64Opcode::Ret | AArch64Opcode::Br => {
            // Check if the second-to-last instruction is a conditional branch.
            // Pattern: BCond target ; B fallthrough
            // In this case, the fall-through is the B's target.
            if term.opcode == AArch64Opcode::B
                && let Some(prev) = get_second_to_last(func, block)
                    && is_conditional_branch(prev.opcode) {
                        // The B's target is where we'd fall through to if placed next.
                        return get_branch_target(term);
                    }
            None
        }

        // Conditional branch without trailing B: fall-through is the
        // non-conditional successor.
        AArch64Opcode::BCond
        | AArch64Opcode::Cbz
        | AArch64Opcode::Cbnz
        | AArch64Opcode::Tbz
        | AArch64Opcode::Tbnz => {
            let cond_target = get_branch_target(term);
            let blk = func.block(block);
            // Return the successor that is NOT the conditional target.
            for &succ in &blk.succs {
                if Some(succ) != cond_target {
                    return Some(succ);
                }
            }
            // If both succs are the same (weird but possible), return it.
            blk.succs.first().copied()
        }

        _ => None,
    }
}

/// Returns true if the opcode is a conditional branch.
fn is_conditional_branch(opcode: AArch64Opcode) -> bool {
    matches!(
        opcode,
        AArch64Opcode::BCond
            | AArch64Opcode::Cbz
            | AArch64Opcode::Cbnz
            | AArch64Opcode::Tbz
            | AArch64Opcode::Tbnz
    )
}

/// Extract the block target from a branch instruction's operands.
fn get_branch_target(inst: &MachInst) -> Option<BlockId> {
    for op in &inst.operands {
        if let MachOperand::Block(bid) = op {
            return Some(*bid);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::{AArch64CC, InstId, MachOperand, Signature};

    /// Helper: create a minimal function with given number of blocks.
    /// Returns the function with blocks but no instructions or edges.
    fn make_func(name: &str, num_blocks: usize) -> MachFunction {
        let sig = Signature::new(vec![], vec![]);
        let mut func = MachFunction::new(name.to_string(), sig);
        // MachFunction::new already creates block 0 (entry).
        for _ in 1..num_blocks {
            func.create_block();
        }
        func
    }

    /// Helper: add an instruction to a block.
    fn add_inst(func: &mut MachFunction, block: BlockId, inst: MachInst) -> InstId {
        let id = func.push_inst(inst);
        func.append_inst(block, id);
        id
    }

    // -----------------------------------------------------------------------
    // test_linear_layout: bb0 -> bb1 -> bb2 (all fall-through)
    // -----------------------------------------------------------------------
    #[test]
    fn test_linear_layout() {
        let mut func = make_func("linear", 3);
        let bb0 = BlockId(0);
        let bb1 = BlockId(1);
        let bb2 = BlockId(2);

        // bb0: B bb1
        add_inst(
            &mut func,
            bb0,
            MachInst::new(AArch64Opcode::AddRI, vec![]),
        );
        add_inst(
            &mut func,
            bb0,
            MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(bb1)]),
        );
        func.add_edge(bb0, bb1);

        // bb1: B bb2
        add_inst(
            &mut func,
            bb1,
            MachInst::new(AArch64Opcode::AddRI, vec![]),
        );
        add_inst(
            &mut func,
            bb1,
            MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(bb2)]),
        );
        func.add_edge(bb1, bb2);

        // bb2: Ret
        add_inst(
            &mut func,
            bb2,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        );

        compute_block_layout(&mut func);

        // Linear chain should stay in order: bb0, bb1, bb2.
        assert_eq!(func.block_order, vec![bb0, bb1, bb2]);
    }

    // -----------------------------------------------------------------------
    // test_diamond_layout: bb0 branches conditionally to bb1/bb2, both go to bb3
    // -----------------------------------------------------------------------
    #[test]
    fn test_diamond_layout() {
        let mut func = make_func("diamond", 4);
        let bb0 = BlockId(0);
        let bb1 = BlockId(1);
        let bb2 = BlockId(2);
        let bb3 = BlockId(3);

        // bb0: BCond bb1 ; B bb2
        add_inst(
            &mut func,
            bb0,
            MachInst::new(
                AArch64Opcode::BCond,
                vec![MachOperand::Imm(AArch64CC::EQ as i64), MachOperand::Block(bb1)],
            ),
        );
        add_inst(
            &mut func,
            bb0,
            MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(bb2)]),
        );
        func.add_edge(bb0, bb1);
        func.add_edge(bb0, bb2);

        // bb1: B bb3
        add_inst(
            &mut func,
            bb1,
            MachInst::new(AArch64Opcode::AddRI, vec![]),
        );
        add_inst(
            &mut func,
            bb1,
            MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(bb3)]),
        );
        func.add_edge(bb1, bb3);

        // bb2: B bb3
        add_inst(
            &mut func,
            bb2,
            MachInst::new(AArch64Opcode::AddRI, vec![]),
        );
        add_inst(
            &mut func,
            bb2,
            MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(bb3)]),
        );
        func.add_edge(bb2, bb3);

        // bb3: Ret
        add_inst(
            &mut func,
            bb3,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        );

        compute_block_layout(&mut func);

        // bb0 has BCond bb1 + B bb2. Fall-through = bb2 (the B target).
        // So layout should be: bb0, bb2, then bb1, then bb3.
        assert_eq!(func.block_order[0], bb0);
        assert_eq!(func.block_order[1], bb2);
        // bb1 and bb3 follow (bb2 chains to bb3, then bb1 is leftover).
        // Actually bb2 -> bb3, so: bb0, bb2, bb3, bb1
        // But bb1 also -> bb3 (already placed), so bb1 has no unplaced successor.
        assert!(func.block_order.contains(&bb1));
        assert!(func.block_order.contains(&bb3));
        assert_eq!(func.block_order.len(), 4);
    }

    // -----------------------------------------------------------------------
    // test_loop_layout: bb0 -> bb1 -> bb2 -> bb1 (loop), bb2 also -> bb3
    // -----------------------------------------------------------------------
    #[test]
    fn test_loop_layout() {
        let mut func = make_func("loop", 4);
        let bb0 = BlockId(0);
        let bb1 = BlockId(1);
        let bb2 = BlockId(2);
        let bb3 = BlockId(3);

        // bb0: B bb1
        add_inst(
            &mut func,
            bb0,
            MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(bb1)]),
        );
        func.add_edge(bb0, bb1);

        // bb1: (loop header) some work
        add_inst(
            &mut func,
            bb1,
            MachInst::new(AArch64Opcode::AddRI, vec![]),
        );
        add_inst(
            &mut func,
            bb1,
            MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(bb2)]),
        );
        func.add_edge(bb1, bb2);

        // bb2: BCond bb1 (loop back) ; B bb3 (exit)
        add_inst(
            &mut func,
            bb2,
            MachInst::new(
                AArch64Opcode::BCond,
                vec![MachOperand::Imm(AArch64CC::NE as i64), MachOperand::Block(bb1)],
            ),
        );
        add_inst(
            &mut func,
            bb2,
            MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(bb3)]),
        );
        func.add_edge(bb2, bb1);
        func.add_edge(bb2, bb3);

        // bb3: Ret
        add_inst(
            &mut func,
            bb3,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        );

        compute_block_layout(&mut func);

        // Entry first.
        assert_eq!(func.block_order[0], bb0);
        // Loop body should be contiguous: bb1, bb2 together.
        let pos1 = func.block_order.iter().position(|&b| b == bb1).unwrap();
        let pos2 = func.block_order.iter().position(|&b| b == bb2).unwrap();
        assert_eq!(pos2, pos1 + 1, "loop body blocks bb1, bb2 should be contiguous");
        // bb3 (exit) after loop.
        let pos3 = func.block_order.iter().position(|&b| b == bb3).unwrap();
        assert!(pos3 > pos2, "exit block bb3 should be after loop body");
    }

    // -----------------------------------------------------------------------
    // test_unreachable_blocks: bb0 -> bb1, bb2 is unreachable
    // -----------------------------------------------------------------------
    #[test]
    fn test_unreachable_blocks() {
        let mut func = make_func("unreachable", 3);
        let bb0 = BlockId(0);
        let bb1 = BlockId(1);
        let bb2 = BlockId(2);

        // bb0: B bb1
        add_inst(
            &mut func,
            bb0,
            MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(bb1)]),
        );
        func.add_edge(bb0, bb1);

        // bb1: Ret
        add_inst(
            &mut func,
            bb1,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        );

        // bb2: unreachable (no predecessors)
        add_inst(
            &mut func,
            bb2,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        );

        compute_block_layout(&mut func);

        // Unreachable bb2 should be last.
        assert_eq!(func.block_order[0], bb0);
        assert_eq!(func.block_order[1], bb1);
        assert_eq!(func.block_order[2], bb2);
    }

    // -----------------------------------------------------------------------
    // test_cbz_fallthrough: block ending in Cbz (no trailing B)
    // -----------------------------------------------------------------------
    #[test]
    fn test_cbz_fallthrough() {
        let mut func = make_func("cbz_ft", 3);
        let bb0 = BlockId(0);
        let bb1 = BlockId(1);
        let bb2 = BlockId(2);

        // bb0: Cbz target=bb2 (fall-through to bb1)
        add_inst(
            &mut func,
            bb0,
            MachInst::new(AArch64Opcode::Cbz, vec![MachOperand::Imm(0), MachOperand::Block(bb2)]),
        );
        func.add_edge(bb0, bb2);
        func.add_edge(bb0, bb1);

        // bb1: Ret
        add_inst(
            &mut func,
            bb1,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        );

        // bb2: Ret
        add_inst(
            &mut func,
            bb2,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        );

        compute_block_layout(&mut func);

        // Cbz targets bb2, so fall-through should prefer bb1.
        assert_eq!(func.block_order[0], bb0);
        assert_eq!(func.block_order[1], bb1);
        assert_eq!(func.block_order[2], bb2);
    }

    // -----------------------------------------------------------------------
    // test_single_block: only entry block
    // -----------------------------------------------------------------------
    #[test]
    fn test_single_block() {
        let mut func = make_func("single", 1);
        add_inst(
            &mut func,
            BlockId(0),
            MachInst::new(AArch64Opcode::Ret, vec![]),
        );

        compute_block_layout(&mut func);
        assert_eq!(func.block_order, vec![BlockId(0)]);
    }
}
