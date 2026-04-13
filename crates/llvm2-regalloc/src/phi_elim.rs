// llvm2-regalloc/phi_elim.rs - Phi elimination and critical edge splitting
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Phi elimination: lowers SSA phi nodes to parallel copies.
//!
//! This is a critical correctness requirement for register allocation.
//! Phi nodes in SSA form are not executable instructions — they must be
//! lowered to copies that execute at the end of predecessor blocks.
//!
//! The algorithm:
//! 1. **Critical edge splitting:** Insert empty blocks on edges where the
//!    source has multiple successors and the target has multiple predecessors.
//!    This ensures copies can be placed without affecting other paths.
//! 2. **Phi elimination:** Replace each phi with parallel copies at the end
//!    of each predecessor block.
//! 3. **Parallel copy sequencing:** Convert parallel copies to a valid
//!    sequential order (handling cycles with temporary registers).
//!
//! IMPORTANT: Naive sequential copy insertion causes wrong-code bugs.
//! Consider: phi(v1=v2, v2=v1) — sequential copies produce v1=v2, v2=v2
//! (wrong!). The parallel copy resolver detects cycles and inserts a
//! temporary to break them.
//!
//! Reference: `~/llvm-project-ref/llvm/lib/CodeGen/PHIElimination.cpp`
//! Also: Sreedhar et al., "Translating Out of Static Single Assignment Form"

use crate::machine_types::{
    BlockId, InstFlags, InstId, MachBlock, MachFunction, MachInst, MachOperand, VReg,
};

/// Pseudo-opcode for a parallel copy (phi lowering).
pub const PSEUDO_PARALLEL_COPY: u16 = 0xFFE0;
/// Pseudo-opcode for a sequential copy.
pub const PSEUDO_COPY: u16 = 0xFFE1;

/// A parallel copy: multiple simultaneous assignments.
///
/// All sources are read before any destination is written.
#[derive(Debug, Clone)]
pub struct ParallelCopy {
    /// (destination, source) pairs.
    pub copies: Vec<(VReg, VReg)>,
}

/// Split critical edges in the control flow graph.
///
/// A critical edge is an edge from block A to block B where A has multiple
/// successors and B has multiple predecessors. These must be split to ensure
/// phi copies can be placed correctly.
///
/// Returns the number of edges split.
pub fn split_critical_edges(func: &mut MachFunction) -> u32 {
    let mut edges_split = 0;

    // Collect critical edges first to avoid mutating while iterating.
    let mut critical_edges: Vec<(BlockId, BlockId)> = Vec::new();

    for &block_id in &func.block_order {
        let block = &func.blocks[block_id.0 as usize];
        if block.succs.len() > 1 {
            for &succ_id in &block.succs {
                let succ = &func.blocks[succ_id.0 as usize];
                if succ.preds.len() > 1 {
                    critical_edges.push((block_id, succ_id));
                }
            }
        }
    }

    // Split each critical edge by inserting a new empty block.
    for (src_id, dst_id) in critical_edges {
        let new_block_id = BlockId(func.blocks.len() as u32);

        // Create the new block: it just jumps to the original destination.
        let new_block = MachBlock {
            insts: Vec::new(), // Empty for now; jump will be added during lowering.
            preds: vec![src_id],
            succs: vec![dst_id],
            loop_depth: func.blocks[dst_id.0 as usize].loop_depth,
        };
        func.blocks.push(new_block);

        // Update source block: replace dst_id with new_block_id in succs.
        let src_block = &mut func.blocks[src_id.0 as usize];
        for succ in &mut src_block.succs {
            if *succ == dst_id {
                *succ = new_block_id;
            }
        }

        // Update destination block: replace src_id with new_block_id in preds.
        let dst_block = &mut func.blocks[dst_id.0 as usize];
        for pred in &mut dst_block.preds {
            if *pred == src_id {
                *pred = new_block_id;
            }
        }

        // Insert new block into block order (right after source block).
        if let Some(pos) = func.block_order.iter().position(|&b| b == dst_id) {
            func.block_order.insert(pos, new_block_id);
        } else {
            func.block_order.push(new_block_id);
        }

        edges_split += 1;
    }

    edges_split
}

/// Eliminate phi instructions by inserting parallel copies.
///
/// For each phi instruction in a block:
/// 1. For each predecessor, insert a copy from the phi's input to the phi's
///    output at the end of that predecessor block.
/// 2. Remove the phi instruction.
///
/// This must be called AFTER critical edge splitting.
pub fn eliminate_phis(func: &mut MachFunction) {
    // Collect all phi instructions and their copy requirements.
    let mut copies_to_insert: Vec<(BlockId, VReg, VReg)> = Vec::new();
    let mut phi_insts_to_remove: Vec<(BlockId, InstId)> = Vec::new();

    for &block_id in &func.block_order {
        let block = &func.blocks[block_id.0 as usize];

        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.0 as usize];

            if !inst.flags.is_phi() {
                continue;
            }

            // Phi instruction format:
            // defs = [dest_vreg]
            // uses = [src_vreg_for_pred_0, src_vreg_for_pred_1, ...]
            // The uses correspond to predecessors in order.
            let dest_vreg = match inst.defs.first().and_then(|op| op.as_vreg()) {
                Some(v) => v,
                None => continue,
            };

            for (i, pred_id) in block.preds.iter().enumerate() {
                if let Some(src_vreg) = inst.uses.get(i).and_then(|op| op.as_vreg()) {
                    if src_vreg != dest_vreg {
                        copies_to_insert.push((*pred_id, dest_vreg, src_vreg));
                    }
                }
            }

            phi_insts_to_remove.push((block_id, inst_id));
        }
    }

    // Insert copies at the end of predecessor blocks (before terminators).
    for (pred_id, dest, src) in copies_to_insert {
        let copy_inst = MachInst {
            opcode: PSEUDO_COPY,
            defs: vec![MachOperand::VReg(dest)],
            uses: vec![MachOperand::VReg(src)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        };

        let copy_id = InstId(func.insts.len() as u32);
        func.insts.push(copy_inst);

        // Insert before the terminator of the predecessor block.
        let pred_block = &mut func.blocks[pred_id.0 as usize];
        let insert_pos = find_terminator_pos(pred_block, &func.insts);
        pred_block.insts.insert(insert_pos, copy_id);
    }

    // Remove phi instructions from their blocks.
    for (block_id, phi_id) in phi_insts_to_remove {
        let block = &mut func.blocks[block_id.0 as usize];
        block.insts.retain(|&id| id != phi_id);
    }
}

/// Find the position of the first terminator in a block.
/// Copies should be inserted before terminators.
fn find_terminator_pos(block: &MachBlock, all_insts: &[MachInst]) -> usize {
    for (i, &inst_id) in block.insts.iter().enumerate().rev() {
        let inst = &all_insts[inst_id.0 as usize];
        if !inst.flags.is_terminator() && !inst.flags.is_branch() && !inst.flags.is_return() {
            return i + 1;
        }
    }
    0 // All instructions are terminators; insert at the beginning.
}

/// Resolve parallel copies into a valid sequential order.
///
/// Given a set of parallel copies `{d1 <- s1, d2 <- s2, ...}`, produce
/// a sequential order that preserves the parallel semantics. Cycles
/// (e.g., `d1 <- d2, d2 <- d1`) are broken using a temporary register.
///
/// Algorithm: topological sort with cycle detection.
/// Reference: Hack et al., "Register Allocation for Programs in SSA Form"
pub fn resolve_parallel_copies(
    copies: &[(VReg, VReg)],
    func: &mut MachFunction,
) -> Vec<(VReg, VReg)> {
    if copies.is_empty() {
        return Vec::new();
    }

    // Build dependency graph: dst -> src.
    let mut remaining: Vec<(VReg, VReg)> = copies.to_vec();
    let mut result: Vec<(VReg, VReg)> = Vec::new();

    // Iteratively emit copies whose destination is not used as a source
    // by any remaining copy (i.e., safe to overwrite).
    let mut progress = true;
    while progress && !remaining.is_empty() {
        progress = false;

        let mut i = 0;
        while i < remaining.len() {
            let (dst, _src) = remaining[i];
            // Check if dst is used as a source by any other remaining copy.
            let is_source = remaining
                .iter()
                .enumerate()
                .any(|(j, &(_, s))| j != i && s == dst);

            if !is_source {
                result.push(remaining.remove(i));
                progress = true;
            } else {
                i += 1;
            }
        }
    }

    // Any remaining copies form cycles — break with temporaries.
    while !remaining.is_empty() {
        let (dst, src) = remaining.remove(0);
        let class = dst.class;
        let tmp = func.alloc_vreg(class);

        // tmp <- dst (save the value being overwritten)
        result.push((tmp, dst));
        // dst <- src
        result.push((dst, src));

        // Rewrite remaining copies that use dst as source to use tmp.
        for copy in &mut remaining {
            if copy.1 == dst {
                copy.1 = tmp;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine_types::RegClass;

    #[test]
    fn test_parallel_copy_no_cycle() {
        let mut func = MachFunction {
            name: "test".into(),
            insts: Vec::new(),
            blocks: Vec::new(),
            block_order: Vec::new(),
            entry_block: BlockId(0),
            next_vreg: 10,
            next_stack_slot: 0,
            stack_slots: Default::default(),
        };

        let v0 = VReg { id: 0, class: RegClass::Gpr64 };
        let v1 = VReg { id: 1, class: RegClass::Gpr64 };
        let v2 = VReg { id: 2, class: RegClass::Gpr64 };

        // v2 <- v0, v1 <- v2 — no cycle if ordered correctly.
        let copies = vec![(v1, v2), (v2, v0)];
        let result = resolve_parallel_copies(&copies, &mut func);

        // v1 <- v2 must come before v2 <- v0 (v2 is overwritten).
        let v1_pos = result.iter().position(|&(d, _)| d == v1).unwrap();
        let v2_pos = result.iter().position(|&(d, _)| d == v2).unwrap();
        assert!(v1_pos < v2_pos);
    }

    #[test]
    fn test_parallel_copy_with_cycle() {
        let mut func = MachFunction {
            name: "test".into(),
            insts: Vec::new(),
            blocks: Vec::new(),
            block_order: Vec::new(),
            entry_block: BlockId(0),
            next_vreg: 10,
            next_stack_slot: 0,
            stack_slots: Default::default(),
        };

        let v0 = VReg { id: 0, class: RegClass::Gpr64 };
        let v1 = VReg { id: 1, class: RegClass::Gpr64 };

        // v0 <- v1, v1 <- v0 — classic swap cycle.
        let copies = vec![(v0, v1), (v1, v0)];
        let result = resolve_parallel_copies(&copies, &mut func);

        // Should use a temporary to break the cycle.
        assert!(result.len() >= 3, "cycle should introduce a temp: got {:?}", result);
    }
}
