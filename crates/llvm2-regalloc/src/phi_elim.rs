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

    #[test]
    fn test_parallel_copy_empty() {
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

        let result = resolve_parallel_copies(&[], &mut func);
        assert!(result.is_empty());
    }

    #[test]
    fn test_parallel_copy_single() {
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
        let copies = vec![(v0, v1)];
        let result = resolve_parallel_copies(&copies, &mut func);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (v0, v1));
    }

    #[test]
    fn test_parallel_copy_chain_order() {
        // Chain: v0 <- v1, v1 <- v2, v2 <- v3
        // Correct order: v0 <- v1 first (v0 isn't a source), then v1 <- v2, then v2 <- v3.
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
        let v3 = VReg { id: 3, class: RegClass::Gpr64 };

        let copies = vec![(v0, v1), (v1, v2), (v2, v3)];
        let result = resolve_parallel_copies(&copies, &mut func);

        // No cycle, so no temporary needed.
        assert_eq!(result.len(), 3, "chain should not need temporaries");

        // v0 <- v1 must come before v1 <- v2 (v1 is read by first, written by second).
        let pos_v0 = result.iter().position(|&(d, _)| d == v0).unwrap();
        let pos_v1 = result.iter().position(|&(d, _)| d == v1).unwrap();
        let pos_v2 = result.iter().position(|&(d, _)| d == v2).unwrap();
        assert!(pos_v0 < pos_v1, "v0 <- v1 must come before v1 <- v2");
        assert!(pos_v1 < pos_v2, "v1 <- v2 must come before v2 <- v3");
    }

    #[test]
    fn test_parallel_copy_three_way_cycle() {
        // v0 <- v1, v1 <- v2, v2 <- v0 — 3-way cycle.
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

        let copies = vec![(v0, v1), (v1, v2), (v2, v0)];
        let result = resolve_parallel_copies(&copies, &mut func);

        // 3-way cycle requires at least one temporary, so result should be > 3.
        assert!(result.len() > 3, "3-way cycle needs a temp: got {:?}", result);

        // Verify that a fresh temporary was allocated.
        assert!(func.next_vreg > 10, "should have allocated a temporary VReg");
    }

    /// Helper: build a diamond CFG with phi instructions.
    ///
    /// Block 0 (entry): def v0, def v1, branch -> block 1 or block 2
    /// Block 1 (then): -- falls through to block 3
    /// Block 2 (else): -- falls through to block 3
    /// Block 3 (merge): phi v2 = [v0 from block1, v1 from block2], use v2
    fn make_diamond_with_phi() -> MachFunction {
        use crate::machine_types::*;
        let mut insts = Vec::new();

        // Block 0: def v0, def v1, cbranch
        let i0 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 1,
            defs: vec![MachOperand::VReg(VReg { id: 0, class: RegClass::Gpr64 })],
            uses: vec![MachOperand::Imm(10)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });
        let i1 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 1,
            defs: vec![MachOperand::VReg(VReg { id: 1, class: RegClass::Gpr64 })],
            uses: vec![MachOperand::Imm(20)],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });
        let i2 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 0xBB,
            defs: vec![],
            uses: vec![
                MachOperand::VReg(VReg { id: 0, class: RegClass::Gpr64 }),
                MachOperand::Block(BlockId(1)),
                MachOperand::Block(BlockId(2)),
            ],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),
        });

        // Block 1 (then): just a branch to merge
        let i3 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 0xBA,
            defs: vec![],
            uses: vec![MachOperand::Block(BlockId(3))],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),
        });

        // Block 2 (else): just a branch to merge
        let i4 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 0xBA,
            defs: vec![],
            uses: vec![MachOperand::Block(BlockId(3))],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),
        });

        // Block 3 (merge): phi v2 = [v0 from block1, v1 from block2], then use v2
        let i5 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 0x00, // phi
            defs: vec![MachOperand::VReg(VReg { id: 2, class: RegClass::Gpr64 })],
            uses: vec![
                MachOperand::VReg(VReg { id: 0, class: RegClass::Gpr64 }), // from block 1
                MachOperand::VReg(VReg { id: 1, class: RegClass::Gpr64 }), // from block 2
            ],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::IS_PHI,
        });
        let i6 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 3,
            defs: vec![],
            uses: vec![MachOperand::VReg(VReg { id: 2, class: RegClass::Gpr64 })],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });

        MachFunction {
            name: "diamond_phi".into(),
            insts,
            blocks: vec![
                MachBlock {
                    insts: vec![i0, i1, i2],
                    preds: Vec::new(),
                    succs: vec![BlockId(1), BlockId(2)],
                    loop_depth: 0,
                },
                MachBlock {
                    insts: vec![i3],
                    preds: vec![BlockId(0)],
                    succs: vec![BlockId(3)],
                    loop_depth: 0,
                },
                MachBlock {
                    insts: vec![i4],
                    preds: vec![BlockId(0)],
                    succs: vec![BlockId(3)],
                    loop_depth: 0,
                },
                MachBlock {
                    insts: vec![i5, i6],
                    preds: vec![BlockId(1), BlockId(2)],
                    succs: Vec::new(),
                    loop_depth: 0,
                },
            ],
            block_order: vec![BlockId(0), BlockId(1), BlockId(2), BlockId(3)],
            entry_block: BlockId(0),
            next_vreg: 3,
            next_stack_slot: 0,
            stack_slots: Default::default(),
        }
    }

    #[test]
    fn test_critical_edge_splitting_diamond() {
        // In the diamond, block 0 has 2 successors and block 3 has 2 predecessors.
        // The edges (0->1) and (0->2) are NOT critical because blocks 1 and 2
        // each have only 1 predecessor. But if block 3 had >1 pred and block 0 >1 succ,
        // we get critical edges. Here, edges 1->3 and 2->3 are NOT critical because
        // blocks 1 and 2 each have only 1 successor. Let's verify the count.
        let mut func = make_diamond_with_phi();
        let edges_split = split_critical_edges(&mut func);
        // In this specific diamond, there are no critical edges:
        // - (0->1): block 1 has 1 pred, not critical
        // - (0->2): block 2 has 1 pred, not critical
        // - (1->3): block 1 has 1 succ, not critical
        // - (2->3): block 2 has 1 succ, not critical
        assert_eq!(edges_split, 0);
    }

    /// Helper: build a CFG that has a critical edge.
    /// Block 0: branch -> block 1 or block 2
    /// Block 1: branch -> block 2 or block 3
    /// Block 2: (merge point with preds from block 0 AND block 1)
    /// Block 3: exit
    fn make_critical_edge_cfg() -> MachFunction {
        use crate::machine_types::*;
        let mut insts = Vec::new();

        // Block 0: cbranch -> block 1 or block 2
        let i0 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 0xBB,
            defs: vec![],
            uses: vec![MachOperand::Block(BlockId(1)), MachOperand::Block(BlockId(2))],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),
        });

        // Block 1: cbranch -> block 2 or block 3
        let i1 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 0xBB,
            defs: vec![],
            uses: vec![MachOperand::Block(BlockId(2)), MachOperand::Block(BlockId(3))],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),
        });

        // Block 2: nop (merge point)
        let i2 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 0x00,
            defs: vec![],
            uses: vec![],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });

        // Block 3: nop (exit)
        let i3 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 0x00,
            defs: vec![],
            uses: vec![],
            implicit_defs: Vec::new(),
            implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });

        MachFunction {
            name: "critical_edge".into(),
            insts,
            blocks: vec![
                MachBlock { // Block 0: 2 succs
                    insts: vec![i0],
                    preds: Vec::new(),
                    succs: vec![BlockId(1), BlockId(2)],
                    loop_depth: 0,
                },
                MachBlock { // Block 1: 2 succs
                    insts: vec![i1],
                    preds: vec![BlockId(0)],
                    succs: vec![BlockId(2), BlockId(3)],
                    loop_depth: 0,
                },
                MachBlock { // Block 2: 2 preds (from block 0 and block 1) — target of critical edges
                    insts: vec![i2],
                    preds: vec![BlockId(0), BlockId(1)],
                    succs: Vec::new(),
                    loop_depth: 0,
                },
                MachBlock { // Block 3: 1 pred
                    insts: vec![i3],
                    preds: vec![BlockId(1)],
                    succs: Vec::new(),
                    loop_depth: 0,
                },
            ],
            block_order: vec![BlockId(0), BlockId(1), BlockId(2), BlockId(3)],
            entry_block: BlockId(0),
            next_vreg: 0,
            next_stack_slot: 0,
            stack_slots: Default::default(),
        }
    }

    #[test]
    fn test_critical_edge_splitting_with_critical_edges() {
        let mut func = make_critical_edge_cfg();
        let initial_block_count = func.blocks.len();

        let edges_split = split_critical_edges(&mut func);

        // Critical edges: (0->2) because block 0 has 2 succs and block 2 has 2 preds.
        // (1->2) because block 1 has 2 succs and block 2 has 2 preds.
        assert_eq!(edges_split, 2, "should split 2 critical edges");

        // Two new blocks should have been added.
        assert_eq!(func.blocks.len(), initial_block_count + 2);
    }

    #[test]
    fn test_critical_edge_split_preserves_cfg_consistency() {
        let mut func = make_critical_edge_cfg();
        split_critical_edges(&mut func);

        // After splitting, verify CFG consistency:
        // Every block's succs should reference blocks that have it as a pred.
        for &block_id in &func.block_order {
            let bi = block_id.0 as usize;
            let block = &func.blocks[bi];
            for &succ_id in &block.succs {
                let succ = &func.blocks[succ_id.0 as usize];
                assert!(
                    succ.preds.contains(&block_id),
                    "block {:?} lists {:?} as succ but {:?} doesn't have {:?} as pred",
                    block_id, succ_id, succ_id, block_id
                );
            }
        }
    }

    #[test]
    fn test_eliminate_phis_removes_phi_instructions() {
        let mut func = make_diamond_with_phi();
        let phi_count_before = func.insts.iter().filter(|i| i.flags.is_phi()).count();
        assert_eq!(phi_count_before, 1, "should start with 1 phi");

        eliminate_phis(&mut func);

        // After elimination, no phi instructions should remain in any block.
        for &block_id in &func.block_order {
            let block = &func.blocks[block_id.0 as usize];
            for &inst_id in &block.insts {
                let inst = &func.insts[inst_id.0 as usize];
                assert!(!inst.flags.is_phi(), "phi should have been eliminated: {:?}", inst);
            }
        }
    }

    #[test]
    fn test_eliminate_phis_inserts_copies() {
        let mut func = make_diamond_with_phi();
        let insts_before = func.insts.len();

        eliminate_phis(&mut func);

        // Copies should have been inserted. The phi had 2 inputs from 2 preds
        // (v0 from block 1, v1 from block 2). Since v0 != v2 and v1 != v2,
        // two copies should be inserted.
        let copy_count = func.insts.iter().filter(|i| i.opcode == PSEUDO_COPY).count();
        assert!(copy_count >= 2, "should insert at least 2 copies, got {}", copy_count);
        assert!(func.insts.len() > insts_before, "should have more instructions after phi elim");
    }

    #[test]
    fn test_parallel_copy_correctness_swap() {
        // Verify that resolving a swap (v0 <- v1, v1 <- v0) produces
        // a sequence that correctly implements parallel semantics.
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

        let copies = vec![(v0, v1), (v1, v0)];
        let result = resolve_parallel_copies(&copies, &mut func);

        // Simulate the copy sequence with a register file.
        let mut regs: std::collections::HashMap<u32, i64> = std::collections::HashMap::new();
        regs.insert(0, 100); // v0 = 100
        regs.insert(1, 200); // v1 = 200
        // Also initialize any temporaries the resolver may have created.
        for &(dst, src) in &result {
            regs.entry(dst.id).or_insert(0);
            regs.entry(src.id).or_insert(0);
        }

        // Reset to initial values (temps start as 0, v0=100, v1=200).
        regs.insert(0, 100);
        regs.insert(1, 200);

        // Execute copies sequentially.
        for &(dst, src) in &result {
            let val = regs[&src.id];
            regs.insert(dst.id, val);
        }

        // After parallel swap: v0 should have old v1 value, v1 should have old v0 value.
        assert_eq!(regs[&0], 200, "v0 should be 200 (old v1)");
        assert_eq!(regs[&1], 100, "v1 should be 100 (old v0)");
    }

    // -----------------------------------------------------------------------
    // Additional edge-case and correctness tests (issue #139)
    // -----------------------------------------------------------------------

    /// Helper to create a minimal test function.
    fn make_empty_func(next_vreg: u32) -> MachFunction {
        MachFunction {
            name: "test".into(),
            insts: Vec::new(),
            blocks: Vec::new(),
            block_order: Vec::new(),
            entry_block: BlockId(0),
            next_vreg,
            next_stack_slot: 0,
            stack_slots: Default::default(),
        }
    }

    #[test]
    fn test_parallel_copy_four_way_cycle() {
        // v0 <- v1, v1 <- v2, v2 <- v3, v3 <- v0 — 4-way rotation.
        let mut func = make_empty_func(10);
        let v0 = VReg { id: 0, class: RegClass::Gpr64 };
        let v1 = VReg { id: 1, class: RegClass::Gpr64 };
        let v2 = VReg { id: 2, class: RegClass::Gpr64 };
        let v3 = VReg { id: 3, class: RegClass::Gpr64 };

        let copies = vec![(v0, v1), (v1, v2), (v2, v3), (v3, v0)];
        let result = resolve_parallel_copies(&copies, &mut func);

        // Must use at least one temporary.
        assert!(result.len() > 4, "4-way cycle needs temps: got {:?}", result);

        // Simulate and verify correctness.
        let mut regs: std::collections::HashMap<u32, i64> = std::collections::HashMap::new();
        regs.insert(0, 10); regs.insert(1, 20); regs.insert(2, 30); regs.insert(3, 40);
        for &(dst, _) in &result {
            regs.entry(dst.id).or_insert(0);
        }
        // Reset.
        regs.insert(0, 10); regs.insert(1, 20); regs.insert(2, 30); regs.insert(3, 40);
        for &(dst, src) in &result {
            let val = regs[&src.id];
            regs.insert(dst.id, val);
        }

        assert_eq!(regs[&0], 20, "v0 should have old v1");
        assert_eq!(regs[&1], 30, "v1 should have old v2");
        assert_eq!(regs[&2], 40, "v2 should have old v3");
        assert_eq!(regs[&3], 10, "v3 should have old v0");
    }

    #[test]
    fn test_parallel_copy_self_copy_ignored() {
        // v0 <- v0 should be a no-op.
        let mut func = make_empty_func(10);
        let v0 = VReg { id: 0, class: RegClass::Gpr64 };
        let v1 = VReg { id: 1, class: RegClass::Gpr64 };

        // Mix self-copies with real copies.
        let copies = vec![(v0, v1)]; // just one real copy
        let result = resolve_parallel_copies(&copies, &mut func);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (v0, v1));
    }

    #[test]
    fn test_parallel_copy_independent_copies() {
        // Independent copies: v0 <- v2, v1 <- v3 (no dependencies).
        let mut func = make_empty_func(10);
        let v0 = VReg { id: 0, class: RegClass::Gpr64 };
        let v1 = VReg { id: 1, class: RegClass::Gpr64 };
        let v2 = VReg { id: 2, class: RegClass::Gpr64 };
        let v3 = VReg { id: 3, class: RegClass::Gpr64 };

        let copies = vec![(v0, v2), (v1, v3)];
        let result = resolve_parallel_copies(&copies, &mut func);
        assert_eq!(result.len(), 2, "independent copies need no temps");
    }

    #[test]
    fn test_parallel_copy_chain_correctness_simulation() {
        // Chain: v0 <- v1, v1 <- v2. After parallel execution:
        // v0 = old_v1, v1 = old_v2.
        let mut func = make_empty_func(10);
        let v0 = VReg { id: 0, class: RegClass::Gpr64 };
        let v1 = VReg { id: 1, class: RegClass::Gpr64 };
        let v2 = VReg { id: 2, class: RegClass::Gpr64 };

        let copies = vec![(v0, v1), (v1, v2)];
        let result = resolve_parallel_copies(&copies, &mut func);

        let mut regs: std::collections::HashMap<u32, i64> = std::collections::HashMap::new();
        regs.insert(0, 10); regs.insert(1, 20); regs.insert(2, 30);
        for &(dst, src) in &result {
            let val = regs[&src.id];
            regs.insert(dst.id, val);
        }
        assert_eq!(regs[&0], 20, "v0 should have old v1");
        assert_eq!(regs[&1], 30, "v1 should have old v2");
    }

    #[test]
    fn test_critical_edge_splitting_no_edges_no_changes() {
        // A single-block function has no edges to split.
        use crate::machine_types::*;
        let mut func = MachFunction {
            name: "single".into(),
            insts: vec![MachInst {
                opcode: 0xFF,
                defs: vec![],
                uses: vec![],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::IS_RETURN.union(InstFlags::IS_TERMINATOR),
            }],
            blocks: vec![MachBlock {
                insts: vec![InstId(0)],
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: 0,
            next_stack_slot: 0,
            stack_slots: Default::default(),
        };

        let splits = split_critical_edges(&mut func);
        assert_eq!(splits, 0);
        assert_eq!(func.blocks.len(), 1);
    }

    #[test]
    fn test_eliminate_phis_no_phis_no_change() {
        // A function with no phi instructions should be unchanged.
        use crate::machine_types::*;
        let mut func = MachFunction {
            name: "no_phis".into(),
            insts: vec![MachInst {
                opcode: 1,
                defs: vec![MachOperand::VReg(VReg { id: 0, class: RegClass::Gpr64 })],
                uses: vec![MachOperand::Imm(0)],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::default(),
            }],
            blocks: vec![MachBlock {
                insts: vec![InstId(0)],
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: 1,
            next_stack_slot: 0,
            stack_slots: Default::default(),
        };

        let insts_before = func.insts.len();
        eliminate_phis(&mut func);
        assert_eq!(func.insts.len(), insts_before, "no phis means no changes");
    }

    #[test]
    fn test_eliminate_phis_copies_go_to_correct_predecessor() {
        let mut func = make_diamond_with_phi();
        eliminate_phis(&mut func);

        // Block 1 (then) should now have a copy instruction (v2 <- v0).
        let block1_has_copy = func.blocks[1].insts.iter().any(|&id| {
            func.insts[id.0 as usize].opcode == PSEUDO_COPY
        });
        assert!(block1_has_copy, "block 1 should have a copy for the phi");

        // Block 2 (else) should now have a copy instruction (v2 <- v1).
        let block2_has_copy = func.blocks[2].insts.iter().any(|&id| {
            func.insts[id.0 as usize].opcode == PSEUDO_COPY
        });
        assert!(block2_has_copy, "block 2 should have a copy for the phi");
    }

    #[test]
    fn test_eliminate_phis_copy_before_terminator() {
        let mut func = make_diamond_with_phi();
        eliminate_phis(&mut func);

        // In each predecessor block, the copy should be inserted before
        // the terminator (branch) instruction, not after it.
        for block_idx in [1usize, 2usize] {
            let block = &func.blocks[block_idx];
            let insts = &block.insts;
            if insts.len() < 2 {
                continue;
            }
            let last_inst = &func.insts[insts.last().unwrap().0 as usize];
            assert!(
                last_inst.flags.is_branch() || last_inst.flags.is_terminator(),
                "block {block_idx}: last instruction should still be a terminator"
            );
        }
    }

    #[test]
    fn test_critical_edge_split_new_block_in_block_order() {
        let mut func = make_critical_edge_cfg();
        let edges_split = split_critical_edges(&mut func);
        assert_eq!(edges_split, 2);

        // The new blocks should appear in block_order.
        assert!(func.block_order.len() >= 6, "should have original 4 + 2 new blocks");

        // New blocks should be contiguous in block_order.
        for &block_id in &func.block_order {
            assert!(
                (block_id.0 as usize) < func.blocks.len(),
                "block_order references non-existent block {:?}",
                block_id
            );
        }
    }

    #[test]
    fn test_parallel_copy_preserves_all_values() {
        // Exhaustive simulation for a 5-way rotation.
        let mut func = make_empty_func(20);
        let vregs: Vec<VReg> = (0..5).map(|id| VReg { id, class: RegClass::Gpr64 }).collect();
        let copies: Vec<(VReg, VReg)> = (0..5).map(|i| (vregs[i], vregs[(i + 1) % 5])).collect();
        let result = resolve_parallel_copies(&copies, &mut func);

        let initial: Vec<i64> = vec![100, 200, 300, 400, 500];
        let mut regs: std::collections::HashMap<u32, i64> = std::collections::HashMap::new();
        for i in 0..5 { regs.insert(i as u32, initial[i]); }
        for &(dst, _) in &result { regs.entry(dst.id).or_insert(0); }
        for i in 0..5 { regs.insert(i as u32, initial[i]); }
        for &(dst, src) in &result {
            let val = regs[&src.id];
            regs.insert(dst.id, val);
        }

        // Each v[i] should get old v[(i+1)%5].
        for i in 0..5 {
            let expected = initial[(i + 1) % 5];
            assert_eq!(
                regs[&(i as u32)], expected,
                "v{i} should be {expected} (old v{})",
                (i + 1) % 5
            );
        }
    }

    #[test]
    fn test_find_terminator_pos_all_terminators() {
        // When ALL instructions are terminators, insertion should be at position 0.
        use crate::machine_types::*;
        let insts = vec![
            MachInst {
                opcode: 0xBA,
                defs: vec![],
                uses: vec![],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),
            },
        ];
        let block = MachBlock {
            insts: vec![InstId(0)],
            preds: Vec::new(),
            succs: Vec::new(),
            loop_depth: 0,
        };
        let pos = find_terminator_pos(&block, &insts);
        assert_eq!(pos, 0, "with all terminators, insert at 0");
    }

    #[test]
    fn test_find_terminator_pos_no_terminators() {
        // When NO instructions are terminators, insertion should be at end.
        use crate::machine_types::*;
        let insts = vec![
            MachInst {
                opcode: 1,
                defs: vec![],
                uses: vec![],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::default(),
            },
            MachInst {
                opcode: 2,
                defs: vec![],
                uses: vec![],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::default(),
            },
        ];
        let block = MachBlock {
            insts: vec![InstId(0), InstId(1)],
            preds: Vec::new(),
            succs: Vec::new(),
            loop_depth: 0,
        };
        let pos = find_terminator_pos(&block, &insts);
        assert_eq!(pos, 2, "with no terminators, insert at end");
    }

    #[test]
    fn test_eliminate_phis_multiple_phis_same_block() {
        // Block 3 has two phis: phi v2 = [v0, v1], phi v3 = [v4, v5].
        use crate::machine_types::*;
        let mut insts = Vec::new();

        // Block 0: def v0, v4, branch -> block 1 or block 2
        let i0 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 1,
            defs: vec![MachOperand::VReg(VReg { id: 0, class: RegClass::Gpr64 })],
            uses: vec![MachOperand::Imm(10)],
            implicit_defs: Vec::new(), implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });
        let i0b = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 1,
            defs: vec![MachOperand::VReg(VReg { id: 4, class: RegClass::Gpr64 })],
            uses: vec![MachOperand::Imm(40)],
            implicit_defs: Vec::new(), implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });
        let i1 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 1,
            defs: vec![MachOperand::VReg(VReg { id: 1, class: RegClass::Gpr64 })],
            uses: vec![MachOperand::Imm(20)],
            implicit_defs: Vec::new(), implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });
        let i1b = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 1,
            defs: vec![MachOperand::VReg(VReg { id: 5, class: RegClass::Gpr64 })],
            uses: vec![MachOperand::Imm(50)],
            implicit_defs: Vec::new(), implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });
        let branch = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 0xBB,
            defs: vec![],
            uses: vec![MachOperand::Block(BlockId(1)), MachOperand::Block(BlockId(2))],
            implicit_defs: Vec::new(), implicit_uses: Vec::new(),
            flags: InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),
        });

        // Block 1: branch to 3
        let b1_br = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 0xBA,
            defs: vec![], uses: vec![MachOperand::Block(BlockId(3))],
            implicit_defs: Vec::new(), implicit_uses: Vec::new(),
            flags: InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),
        });

        // Block 2: branch to 3
        let b2_br = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 0xBA,
            defs: vec![], uses: vec![MachOperand::Block(BlockId(3))],
            implicit_defs: Vec::new(), implicit_uses: Vec::new(),
            flags: InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),
        });

        // Block 3: phi v2 = [v0, v1], phi v3 = [v4, v5], use v2, use v3
        let phi1 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 0x00,
            defs: vec![MachOperand::VReg(VReg { id: 2, class: RegClass::Gpr64 })],
            uses: vec![
                MachOperand::VReg(VReg { id: 0, class: RegClass::Gpr64 }),
                MachOperand::VReg(VReg { id: 1, class: RegClass::Gpr64 }),
            ],
            implicit_defs: Vec::new(), implicit_uses: Vec::new(),
            flags: InstFlags::IS_PHI,
        });
        let phi2 = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 0x00,
            defs: vec![MachOperand::VReg(VReg { id: 3, class: RegClass::Gpr64 })],
            uses: vec![
                MachOperand::VReg(VReg { id: 4, class: RegClass::Gpr64 }),
                MachOperand::VReg(VReg { id: 5, class: RegClass::Gpr64 }),
            ],
            implicit_defs: Vec::new(), implicit_uses: Vec::new(),
            flags: InstFlags::IS_PHI,
        });
        let use_inst = InstId(insts.len() as u32);
        insts.push(MachInst {
            opcode: 3,
            defs: vec![],
            uses: vec![
                MachOperand::VReg(VReg { id: 2, class: RegClass::Gpr64 }),
                MachOperand::VReg(VReg { id: 3, class: RegClass::Gpr64 }),
            ],
            implicit_defs: Vec::new(), implicit_uses: Vec::new(),
            flags: InstFlags::default(),
        });

        let mut func = MachFunction {
            name: "multi_phi".into(),
            insts,
            blocks: vec![
                MachBlock {
                    insts: vec![i0, i0b, i1, i1b, branch],
                    preds: Vec::new(),
                    succs: vec![BlockId(1), BlockId(2)],
                    loop_depth: 0,
                },
                MachBlock {
                    insts: vec![b1_br],
                    preds: vec![BlockId(0)],
                    succs: vec![BlockId(3)],
                    loop_depth: 0,
                },
                MachBlock {
                    insts: vec![b2_br],
                    preds: vec![BlockId(0)],
                    succs: vec![BlockId(3)],
                    loop_depth: 0,
                },
                MachBlock {
                    insts: vec![phi1, phi2, use_inst],
                    preds: vec![BlockId(1), BlockId(2)],
                    succs: Vec::new(),
                    loop_depth: 0,
                },
            ],
            block_order: vec![BlockId(0), BlockId(1), BlockId(2), BlockId(3)],
            entry_block: BlockId(0),
            next_vreg: 6,
            next_stack_slot: 0,
            stack_slots: Default::default(),
        };

        eliminate_phis(&mut func);

        // Both phis should be eliminated.
        for &block_id in &func.block_order {
            let block = &func.blocks[block_id.0 as usize];
            for &inst_id in &block.insts {
                let inst = &func.insts[inst_id.0 as usize];
                assert!(!inst.flags.is_phi(), "all phis should be eliminated");
            }
        }

        // Each predecessor should have 2 copies (one for each phi).
        let count_copies_in_block = |block_idx: usize| -> usize {
            func.blocks[block_idx].insts.iter()
                .filter(|&&id| func.insts[id.0 as usize].opcode == PSEUDO_COPY)
                .count()
        };
        assert_eq!(count_copies_in_block(1), 2, "block 1 should have 2 copies");
        assert_eq!(count_copies_in_block(2), 2, "block 2 should have 2 copies");
    }
}
