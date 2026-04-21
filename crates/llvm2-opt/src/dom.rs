// llvm2-opt - Dominator tree analysis
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Dominator tree computation using the Cooper/Harvey/Kennedy algorithm.
//!
//! The dominator tree is a fundamental data structure used by CSE (to ensure
//! a dominating definition exists before eliminating a duplicate) and LICM
//! (to identify loop preheaders and safe hoisting points).
//!
//! # Algorithm
//!
//! Uses the iterative algorithm from:
//! Cooper, Harvey, Kennedy. "A Simple, Fast Dominance Algorithm." (2001)
//!
//! 1. Compute reverse postorder (RPO) numbering via DFS from entry.
//! 2. Iteratively compute immediate dominators using the "intersect" operation.
//! 3. Derive the dominator tree from idom[].
//! 4. Compute dominance frontiers from the tree (for future SSA passes).
//!
//! # Complexity
//!
//! Almost linear in practice (O(N * alpha(N)) for structured programs).
//!
//! Reference: LLVM `llvm/include/llvm/Support/GenericDomTree.h`

use std::collections::{HashMap, HashSet};

use llvm2_ir::{BlockId, MachFunction};

/// Dominator tree for a machine function.
///
/// Provides O(1) immediate-dominator queries and O(depth) dominance
/// checks. Also computes dominance frontiers for SSA construction.
#[derive(Debug, Clone)]
pub struct DomTree {
    /// Immediate dominator for each block. Entry block maps to itself.
    idom: HashMap<BlockId, BlockId>,
    /// Children in the dominator tree (block -> dominated children).
    children: HashMap<BlockId, Vec<BlockId>>,
    /// Dominance frontiers: DF(b) = set of blocks where b's dominance ends.
    dom_frontier: HashMap<BlockId, HashSet<BlockId>>,
    /// Reverse postorder numbering (block -> RPO index). Lower = earlier.
    rpo_number: HashMap<BlockId, u32>,
    /// Reverse postorder sequence of blocks.
    rpo_order: Vec<BlockId>,
}

impl DomTree {
    /// Compute the dominator tree for a machine function.
    pub fn compute(func: &MachFunction) -> Self {
        let entry = func.entry;

        // Step 1: Compute reverse postorder via DFS.
        let rpo_order = compute_rpo(func, entry);
        let mut rpo_number: HashMap<BlockId, u32> = HashMap::new();
        for (i, &block) in rpo_order.iter().enumerate() {
            rpo_number.insert(block, i as u32);
        }

        // Step 2: Initialize idom. Entry dominates itself.
        let mut idom: HashMap<BlockId, BlockId> = HashMap::new();
        idom.insert(entry, entry);

        // Step 3: Iterate until fixpoint (Cooper/Harvey/Kennedy).
        let mut changed = true;
        while changed {
            changed = false;
            for &block in &rpo_order {
                if block == entry {
                    continue;
                }

                let preds = &func.block(block).preds;
                // Find first processed predecessor (one with idom already set).
                let mut new_idom = None;
                for &pred in preds {
                    if idom.contains_key(&pred) {
                        new_idom = Some(pred);
                        break;
                    }
                }

                let Some(mut new_idom_val) = new_idom else {
                    // No processed predecessor yet — skip this iteration.
                    continue;
                };

                // Intersect with remaining processed predecessors.
                for &pred in preds {
                    if pred == new_idom_val {
                        continue;
                    }
                    if idom.contains_key(&pred) {
                        new_idom_val = intersect(pred, new_idom_val, &idom, &rpo_number);
                    }
                }

                if idom.get(&block) != Some(&new_idom_val) {
                    idom.insert(block, new_idom_val);
                    changed = true;
                }
            }
        }

        // Step 4: Build children map from idom.
        let mut children: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
        for (&block, &dom) in &idom {
            if block != dom {
                children.entry(dom).or_default().push(block);
            }
        }
        // Sort children by RPO order for deterministic traversal.
        for kids in children.values_mut() {
            kids.sort_by_key(|b| rpo_number.get(b).copied().unwrap_or(u32::MAX));
        }

        // Step 5: Compute dominance frontiers.
        let dom_frontier = compute_dominance_frontiers(&idom, func, &rpo_order);

        Self {
            idom,
            children,
            dom_frontier,
            rpo_number,
            rpo_order,
        }
    }

    /// Returns the immediate dominator of `block`.
    /// The entry block's idom is itself.
    pub fn idom(&self, block: BlockId) -> Option<BlockId> {
        self.idom.get(&block).copied()
    }

    /// Returns true if `a` dominates `b`.
    ///
    /// Every block dominates itself. Uses the idom chain walk.
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        if a == b {
            return true;
        }
        let mut current = b;
        loop {
            let dom = match self.idom.get(&current) {
                Some(&d) => d,
                None => return false,
            };
            if dom == a {
                return true;
            }
            if dom == current {
                // Reached the root (entry) without finding `a`.
                return false;
            }
            current = dom;
        }
    }

    /// Returns true if `a` strictly dominates `b` (a dominates b and a != b).
    pub fn strictly_dominates(&self, a: BlockId, b: BlockId) -> bool {
        a != b && self.dominates(a, b)
    }

    /// Returns the children of `block` in the dominator tree.
    pub fn children(&self, block: BlockId) -> &[BlockId] {
        self.children.get(&block).map_or(&[], |v| v.as_slice())
    }

    /// Returns the dominance frontier of `block`.
    pub fn dominance_frontier(&self, block: BlockId) -> Option<&HashSet<BlockId>> {
        self.dom_frontier.get(&block)
    }

    /// Returns the reverse postorder of blocks.
    pub fn rpo_order(&self) -> &[BlockId] {
        &self.rpo_order
    }

    /// Returns the RPO number for a block (lower = earlier in RPO).
    pub fn rpo_number(&self, block: BlockId) -> Option<u32> {
        self.rpo_number.get(&block).copied()
    }
}

/// Intersect operation from Cooper/Harvey/Kennedy.
///
/// Walks two fingers up the idom tree until they meet. Uses RPO numbers
/// for comparison — higher RPO number means later in the ordering.
fn intersect(
    mut b1: BlockId,
    mut b2: BlockId,
    idom: &HashMap<BlockId, BlockId>,
    rpo_number: &HashMap<BlockId, u32>,
) -> BlockId {
    while b1 != b2 {
        let n1 = rpo_number.get(&b1).copied().unwrap_or(u32::MAX);
        let n2 = rpo_number.get(&b2).copied().unwrap_or(u32::MAX);
        if n1 > n2 {
            b1 = idom[&b1];
        } else {
            b2 = idom[&b2];
        }
    }
    b1
}

/// Compute reverse postorder via iterative DFS from entry.
fn compute_rpo(func: &MachFunction, entry: BlockId) -> Vec<BlockId> {
    let mut visited: HashSet<BlockId> = HashSet::new();
    let mut postorder: Vec<BlockId> = Vec::new();
    let mut stack: Vec<(BlockId, usize)> = vec![(entry, 0)];
    visited.insert(entry);

    while let Some((block, next_succ_idx)) = stack.last_mut() {
        let block_id = *block;
        let succs = &func.block(block_id).succs;
        if *next_succ_idx < succs.len() {
            let succ = succs[*next_succ_idx];
            *next_succ_idx += 1;
            if visited.insert(succ) {
                stack.push((succ, 0));
            }
        } else {
            postorder.push(block_id);
            stack.pop();
        }
    }

    postorder.reverse();
    postorder
}

/// Compute dominance frontiers using the standard algorithm.
///
/// For each join point (block with >= 2 predecessors), walk up the idom
/// tree from each predecessor until we reach the block's immediate dominator.
/// All blocks on those walks (excluding the idom) have the join point in
/// their dominance frontier.
fn compute_dominance_frontiers(
    idom: &HashMap<BlockId, BlockId>,
    func: &MachFunction,
    rpo_order: &[BlockId],
) -> HashMap<BlockId, HashSet<BlockId>> {
    let mut df: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();

    for &block in rpo_order {
        let preds = &func.block(block).preds;
        if preds.len() < 2 {
            continue;
        }
        for &pred in preds {
            let mut runner = pred;
            while runner != *idom.get(&block).unwrap_or(&block) {
                df.entry(runner).or_default().insert(block);
                let next = idom.get(&runner).copied().unwrap_or(runner);
                if next == runner {
                    break; // reached root
                }
                runner = next;
            }
        }
    }

    df
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::{AArch64Opcode, MachFunction, MachInst, MachOperand, Signature};

    /// Build a diamond CFG:
    ///
    /// ```text
    ///     bb0 (entry)
    ///    /   \
    ///  bb1   bb2
    ///    \   /
    ///     bb3
    /// ```
    fn make_diamond() -> MachFunction {
        let mut func = MachFunction::new(
            "diamond".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry; // bb0
        let bb1 = func.create_block(); // bb1
        let bb2 = func.create_block(); // bb2
        let bb3 = func.create_block(); // bb3

        // Add terminators
        let br0 = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb1), MachOperand::Block(bb2)],
        ));
        func.append_inst(bb0, br0);

        let br1 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb3)],
        ));
        func.append_inst(bb1, br1);

        let br2 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb3)],
        ));
        func.append_inst(bb2, br2);

        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb3, ret);

        // CFG edges
        func.add_edge(bb0, bb1);
        func.add_edge(bb0, bb2);
        func.add_edge(bb1, bb3);
        func.add_edge(bb2, bb3);

        func
    }

    /// Build a simple loop:
    ///
    /// ```text
    ///   bb0 (entry)
    ///    |
    ///   bb1 (header) <---+
    ///   / \               |
    /// bb2  bb3 (latch) --+
    /// ```
    fn make_loop() -> MachFunction {
        let mut func = MachFunction::new(
            "loop".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let bb1 = func.create_block();
        let bb2 = func.create_block();
        let bb3 = func.create_block();

        let br0 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb0, br0);

        let br1 = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb2), MachOperand::Block(bb3)],
        ));
        func.append_inst(bb1, br1);

        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret);

        let br3 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb3, br3);

        func.add_edge(bb0, bb1);
        func.add_edge(bb1, bb2);
        func.add_edge(bb1, bb3);
        func.add_edge(bb3, bb1);

        func
    }

    #[test]
    fn test_diamond_idom() {
        let func = make_diamond();
        let dom = DomTree::compute(&func);

        // bb0 dominates everything
        assert_eq!(dom.idom(BlockId(0)), Some(BlockId(0))); // entry self-dom
        assert_eq!(dom.idom(BlockId(1)), Some(BlockId(0)));
        assert_eq!(dom.idom(BlockId(2)), Some(BlockId(0)));
        assert_eq!(dom.idom(BlockId(3)), Some(BlockId(0)));
    }

    #[test]
    fn test_diamond_dominates() {
        let func = make_diamond();
        let dom = DomTree::compute(&func);

        assert!(dom.dominates(BlockId(0), BlockId(0)));
        assert!(dom.dominates(BlockId(0), BlockId(1)));
        assert!(dom.dominates(BlockId(0), BlockId(3)));
        // bb1 does NOT dominate bb3 (bb2 is an alternate path)
        assert!(!dom.dominates(BlockId(1), BlockId(3)));
        assert!(!dom.dominates(BlockId(2), BlockId(3)));
        // bb1 does NOT dominate bb2
        assert!(!dom.dominates(BlockId(1), BlockId(2)));
    }

    #[test]
    fn test_diamond_strictly_dominates() {
        let func = make_diamond();
        let dom = DomTree::compute(&func);

        assert!(!dom.strictly_dominates(BlockId(0), BlockId(0)));
        assert!(dom.strictly_dominates(BlockId(0), BlockId(1)));
    }

    #[test]
    fn test_diamond_dominance_frontier() {
        let func = make_diamond();
        let dom = DomTree::compute(&func);

        // DF(bb1) = {bb3} (bb1 -> bb3 is a join point)
        let df1 = dom.dominance_frontier(BlockId(1)).unwrap();
        assert!(df1.contains(&BlockId(3)));
        // DF(bb2) = {bb3}
        let df2 = dom.dominance_frontier(BlockId(2)).unwrap();
        assert!(df2.contains(&BlockId(3)));
        // DF(bb0) = {} (bb0 dominates everything)
        let df0 = dom.dominance_frontier(BlockId(0));
        assert!(df0.is_none() || df0.unwrap().is_empty());
    }

    #[test]
    fn test_loop_idom() {
        let func = make_loop();
        let dom = DomTree::compute(&func);

        assert_eq!(dom.idom(BlockId(0)), Some(BlockId(0)));
        assert_eq!(dom.idom(BlockId(1)), Some(BlockId(0)));
        assert_eq!(dom.idom(BlockId(2)), Some(BlockId(1)));
        assert_eq!(dom.idom(BlockId(3)), Some(BlockId(1)));
    }

    #[test]
    fn test_loop_dominates() {
        let func = make_loop();
        let dom = DomTree::compute(&func);

        // bb1 (header) dominates bb2 and bb3
        assert!(dom.dominates(BlockId(1), BlockId(2)));
        assert!(dom.dominates(BlockId(1), BlockId(3)));
        // bb3 (latch) does NOT dominate bb1 (back-edge)
        assert!(!dom.dominates(BlockId(3), BlockId(1)));
    }

    #[test]
    fn test_loop_dominance_frontier() {
        let func = make_loop();
        let dom = DomTree::compute(&func);

        // DF(bb3) = {bb1} (back-edge target)
        let df3 = dom.dominance_frontier(BlockId(3)).unwrap();
        assert!(df3.contains(&BlockId(1)));
    }

    #[test]
    fn test_children() {
        let func = make_diamond();
        let dom = DomTree::compute(&func);

        // bb0 dominates bb1, bb2, bb3
        let kids = dom.children(BlockId(0));
        assert!(kids.contains(&BlockId(1)));
        assert!(kids.contains(&BlockId(2)));
        assert!(kids.contains(&BlockId(3)));

        // bb1 has no children in the dominator tree
        assert!(dom.children(BlockId(1)).is_empty());
    }

    #[test]
    fn test_rpo_order() {
        let func = make_diamond();
        let dom = DomTree::compute(&func);

        let rpo = dom.rpo_order();
        // Entry should be first
        assert_eq!(rpo[0], BlockId(0));
        // bb3 should be after bb1 and bb2
        let pos1 = rpo.iter().position(|&b| b == BlockId(1)).unwrap();
        let pos2 = rpo.iter().position(|&b| b == BlockId(2)).unwrap();
        let pos3 = rpo.iter().position(|&b| b == BlockId(3)).unwrap();
        assert!(pos3 > pos1);
        assert!(pos3 > pos2);
    }

    #[test]
    fn test_single_block() {
        let func = MachFunction::new(
            "single".to_string(),
            Signature::new(vec![], vec![]),
        );
        let dom = DomTree::compute(&func);

        assert_eq!(dom.idom(BlockId(0)), Some(BlockId(0)));
        assert!(dom.dominates(BlockId(0), BlockId(0)));
        assert!(dom.children(BlockId(0)).is_empty());
    }
}
