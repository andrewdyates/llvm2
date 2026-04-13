// llvm2-opt - Loop analysis
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Natural loop detection and analysis.
//!
//! Identifies natural loops by finding back-edges in the dominator tree,
//! then computing the loop body for each back-edge. Supports loop nesting
//! and preheader identification.
//!
//! # Algorithm
//!
//! 1. Compute dominator tree.
//! 2. Find back-edges: edge (latch -> header) where header dominates latch.
//! 3. For each back-edge, compute loop body via reverse reachability
//!    from latch to header (standard natural loop body algorithm).
//! 4. Detect nesting by subset relationships.
//! 5. Identify preheaders (unique non-backedge predecessor of header).
//!
//! Reference: LLVM `llvm/include/llvm/Analysis/LoopInfo.h`

use std::collections::{HashMap, HashSet, VecDeque};

use llvm2_ir::{AArch64Opcode, BlockId, MachFunction, MachInst, MachOperand};

use crate::dom::DomTree;

/// A natural loop identified by its header and body blocks.
#[derive(Debug, Clone)]
pub struct NaturalLoop {
    /// The loop header block (target of the back-edge).
    pub header: BlockId,
    /// The latch block (source of the back-edge).
    pub latch: BlockId,
    /// All blocks in the loop body (includes header and latch).
    pub body: HashSet<BlockId>,
    /// Preheader block, if one exists.
    /// A preheader is the unique predecessor of the header that is NOT
    /// part of the loop body. If the header has multiple non-loop
    /// predecessors, there is no preheader (one must be created).
    pub preheader: Option<BlockId>,
    /// Depth in the loop nest (outermost = 1).
    pub depth: u32,
    /// Parent loop header, if this is a nested loop.
    pub parent: Option<BlockId>,
}

/// Loop analysis results for a machine function.
#[derive(Debug, Clone)]
pub struct LoopAnalysis {
    /// All natural loops, keyed by header block ID.
    /// A header may have multiple back-edges; we merge them into one loop.
    loops: HashMap<BlockId, NaturalLoop>,
    /// Map from block -> innermost loop header containing it.
    block_to_loop: HashMap<BlockId, BlockId>,
}

impl LoopAnalysis {
    /// Compute loop analysis for a machine function.
    pub fn compute(func: &MachFunction, dom: &DomTree) -> Self {
        // Step 1: Find all back-edges.
        let back_edges = find_back_edges(func, dom);

        // Step 2: For each back-edge, compute the natural loop body.
        let mut loops: HashMap<BlockId, NaturalLoop> = HashMap::new();

        for (latch, header) in &back_edges {
            let body = compute_loop_body(func, *header, *latch);

            // Merge with existing loop for this header (multiple latches).
            if let Some(existing) = loops.get_mut(header) {
                existing.body = existing.body.union(&body).copied().collect();
                // Keep the latch that has the highest RPO number (latest).
                // For simplicity, just update if this latch is different.
            } else {
                let preheader = find_preheader(func, *header, &body);
                loops.insert(
                    *header,
                    NaturalLoop {
                        header: *header,
                        latch: *latch,
                        body,
                        preheader,
                        depth: 1,
                        parent: None,
                    },
                );
            }
        }

        // Step 3: Compute nesting depth and parent relationships.
        compute_nesting(&mut loops);

        // Step 4: Build block-to-loop mapping (innermost loop).
        let mut block_to_loop: HashMap<BlockId, BlockId> = HashMap::new();
        // Sort loops by body size (smallest = innermost) to ensure innermost wins.
        let mut sorted_loops: Vec<_> = loops.values().collect();
        sorted_loops.sort_by_key(|l| std::cmp::Reverse(l.body.len()));
        for lp in sorted_loops {
            for &block in &lp.body {
                block_to_loop.insert(block, lp.header);
            }
        }

        Self {
            loops,
            block_to_loop,
        }
    }

    /// Returns the loop with the given header, if it exists.
    pub fn get_loop(&self, header: BlockId) -> Option<&NaturalLoop> {
        self.loops.get(&header)
    }

    /// Returns the innermost loop containing `block`, if any.
    pub fn containing_loop(&self, block: BlockId) -> Option<&NaturalLoop> {
        self.block_to_loop
            .get(&block)
            .and_then(|h| self.loops.get(h))
    }

    /// Returns true if `block` is inside any loop.
    pub fn is_in_loop(&self, block: BlockId) -> bool {
        self.block_to_loop.contains_key(&block)
    }

    /// Returns all loops.
    pub fn all_loops(&self) -> impl Iterator<Item = &NaturalLoop> {
        self.loops.values()
    }

    /// Returns the number of detected loops.
    pub fn num_loops(&self) -> usize {
        self.loops.len()
    }

    /// Returns true if the analysis found no loops.
    pub fn is_empty(&self) -> bool {
        self.loops.is_empty()
    }
}

/// Find back-edges: (latch, header) where header dominates latch.
fn find_back_edges(func: &MachFunction, dom: &DomTree) -> Vec<(BlockId, BlockId)> {
    let mut back_edges = Vec::new();

    for &block in &func.block_order {
        let succs = &func.block(block).succs;
        for &succ in succs {
            // A back-edge is an edge where the target dominates the source.
            if dom.dominates(succ, block) {
                back_edges.push((block, succ));
            }
        }
    }

    back_edges
}

/// Compute the natural loop body for a back-edge (latch -> header).
///
/// The body is the set of blocks from which the latch is reachable
/// without going through the header. Computed via reverse BFS from
/// the latch, stopping at the header.
fn compute_loop_body(func: &MachFunction, header: BlockId, latch: BlockId) -> HashSet<BlockId> {
    let mut body: HashSet<BlockId> = HashSet::new();
    body.insert(header);
    body.insert(latch);

    if header == latch {
        // Self-loop: body is just the header.
        return body;
    }

    let mut worklist: VecDeque<BlockId> = VecDeque::new();
    worklist.push_back(latch);

    while let Some(block) = worklist.pop_front() {
        for &pred in &func.block(block).preds {
            if body.insert(pred) {
                worklist.push_back(pred);
            }
        }
    }

    body
}

/// Find the preheader of a loop, if one exists.
///
/// A preheader is the unique predecessor of the header that is NOT in
/// the loop body. If the header has zero or multiple non-loop
/// predecessors, there is no natural preheader.
fn find_preheader(
    func: &MachFunction,
    header: BlockId,
    body: &HashSet<BlockId>,
) -> Option<BlockId> {
    let preds = &func.block(header).preds;
    let non_loop_preds: Vec<BlockId> = preds
        .iter()
        .filter(|p| !body.contains(p))
        .copied()
        .collect();

    if non_loop_preds.len() == 1 {
        Some(non_loop_preds[0])
    } else {
        None
    }
}

/// Compute loop nesting depths and parent relationships.
///
/// A loop L1 is nested inside L2 if L1.body is a strict subset of L2.body
/// and L1.header != L2.header. The parent is the smallest enclosing loop.
fn compute_nesting(loops: &mut HashMap<BlockId, NaturalLoop>) {
    let headers: Vec<BlockId> = loops.keys().copied().collect();

    for &h1 in &headers {
        let mut best_parent: Option<BlockId> = None;
        let mut best_parent_size = usize::MAX;

        let body1 = loops[&h1].body.clone();

        for &h2 in &headers {
            if h1 == h2 {
                continue;
            }
            let body2 = &loops[&h2].body;

            // Check if loop h1 is contained in loop h2.
            if body1.is_subset(body2) && body1.len() < body2.len() {
                if body2.len() < best_parent_size {
                    best_parent = Some(h2);
                    best_parent_size = body2.len();
                }
            }
        }

        if let Some(parent_header) = best_parent {
            loops.get_mut(&h1).unwrap().parent = Some(parent_header);
        }
    }

    // Compute depths from parent chain.
    for &h in &headers {
        let mut depth = 1u32;
        let mut current = loops[&h].parent;
        while let Some(parent) = current {
            depth += 1;
            current = loops[&parent].parent;
        }
        loops.get_mut(&h).unwrap().depth = depth;
    }
}

/// Create a preheader block for a loop that doesn't have one.
///
/// The preheader is inserted between the non-loop predecessors and the
/// header. All non-loop predecessors are redirected to the preheader,
/// which unconditionally branches to the header.
///
/// Returns the new preheader BlockId.
pub fn create_preheader(
    func: &mut MachFunction,
    header: BlockId,
    body: &HashSet<BlockId>,
) -> BlockId {
    // Create new preheader block.
    let preheader = func.create_block();

    // Add unconditional branch from preheader to header.
    let br = func.push_inst(MachInst::new(
        AArch64Opcode::B,
        vec![MachOperand::Block(header)],
    ));
    func.append_inst(preheader, br);

    // Identify non-loop predecessors of header.
    let non_loop_preds: Vec<BlockId> = func
        .block(header)
        .preds
        .iter()
        .filter(|p| !body.contains(p))
        .copied()
        .collect();

    // Update CFG: non-loop preds -> preheader -> header
    func.add_edge(preheader, header);

    for pred in &non_loop_preds {
        // Remove old edge pred -> header from pred's succs.
        let succs = &mut func.block_mut(*pred).succs;
        for s in succs.iter_mut() {
            if *s == header {
                *s = preheader;
            }
        }
        // Remove pred from header's preds.
        let header_preds = &mut func.block_mut(header).preds;
        header_preds.retain(|p| *p != *pred);

        // Add pred -> preheader edge.
        func.block_mut(preheader).preds.push(*pred);

        // Update branch targets in predecessor's terminator.
        let pred_block = func.block(*pred);
        if let Some(&last_inst_id) = pred_block.insts.last() {
            let inst = func.inst_mut(last_inst_id);
            for operand in &mut inst.operands {
                if let MachOperand::Block(target) = operand {
                    if *target == header {
                        *target = preheader;
                    }
                }
            }
        }
    }

    // Place preheader just before header in block_order.
    let header_pos = func
        .block_order
        .iter()
        .position(|&b| b == header)
        .unwrap_or(func.block_order.len());
    // Remove the preheader from its current position (create_block appends it).
    func.block_order.retain(|&b| b != preheader);
    func.block_order.insert(header_pos, preheader);

    preheader
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dom::DomTree;
    use llvm2_ir::{AArch64Opcode, MachFunction, MachInst, MachOperand, Signature};

    /// Build a simple loop:
    ///
    /// ```text
    ///   bb0 (entry / preheader)
    ///    |
    ///   bb1 (header) <---+
    ///   / \               |
    /// bb2  bb3 (latch) --+
    /// ```
    fn make_simple_loop() -> MachFunction {
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

    /// Build nested loops:
    ///
    /// ```text
    ///   bb0 (entry)
    ///    |
    ///   bb1 (outer header) <---------+
    ///    |                             |
    ///   bb2 (inner header) <---+      |
    ///    |                     |      |
    ///   bb3 (inner latch) ----+      |
    ///    |                            |
    ///   bb4 (outer latch) ----------+
    ///    |
    ///   bb5 (exit)
    /// ```
    fn make_nested_loops() -> MachFunction {
        let mut func = MachFunction::new(
            "nested".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let bb1 = func.create_block();
        let bb2 = func.create_block();
        let bb3 = func.create_block();
        let bb4 = func.create_block();
        let bb5 = func.create_block();

        let br0 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb0, br0);

        let br1 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb2)],
        ));
        func.append_inst(bb1, br1);

        let br2 = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb3), MachOperand::Block(bb4)],
        ));
        func.append_inst(bb2, br2);

        let br3 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb2)],
        ));
        func.append_inst(bb3, br3);

        let br4 = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb1), MachOperand::Block(bb5)],
        ));
        func.append_inst(bb4, br4);

        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb5, ret);

        func.add_edge(bb0, bb1);
        func.add_edge(bb1, bb2);
        func.add_edge(bb2, bb3);
        func.add_edge(bb2, bb4);
        func.add_edge(bb3, bb2);
        func.add_edge(bb4, bb1);
        func.add_edge(bb4, bb5);

        func
    }

    #[test]
    fn test_simple_loop_detection() {
        let func = make_simple_loop();
        let dom = DomTree::compute(&func);
        let la = LoopAnalysis::compute(&func, &dom);

        assert_eq!(la.num_loops(), 1);

        let lp = la.get_loop(BlockId(1)).expect("loop at bb1");
        assert_eq!(lp.header, BlockId(1));
        assert_eq!(lp.latch, BlockId(3));
        assert!(lp.body.contains(&BlockId(1)));
        assert!(lp.body.contains(&BlockId(3)));
        // bb0 is NOT in the loop body
        assert!(!lp.body.contains(&BlockId(0)));
        // bb2 is NOT in the loop body (it's the exit)
        assert!(!lp.body.contains(&BlockId(2)));
    }

    #[test]
    fn test_preheader_found() {
        let func = make_simple_loop();
        let dom = DomTree::compute(&func);
        let la = LoopAnalysis::compute(&func, &dom);

        let lp = la.get_loop(BlockId(1)).unwrap();
        assert_eq!(lp.preheader, Some(BlockId(0)));
    }

    #[test]
    fn test_nested_loops() {
        let func = make_nested_loops();
        let dom = DomTree::compute(&func);
        let la = LoopAnalysis::compute(&func, &dom);

        assert_eq!(la.num_loops(), 2);

        // Outer loop: header=bb1
        let outer = la.get_loop(BlockId(1)).expect("outer loop at bb1");
        assert_eq!(outer.header, BlockId(1));

        // Inner loop: header=bb2
        let inner = la.get_loop(BlockId(2)).expect("inner loop at bb2");
        assert_eq!(inner.header, BlockId(2));

        // Inner loop is nested inside outer loop.
        assert!(inner.body.is_subset(&outer.body));
        assert_eq!(inner.parent, Some(BlockId(1)));
        assert_eq!(inner.depth, 2);
        assert_eq!(outer.depth, 1);
    }

    #[test]
    fn test_no_loops() {
        // Diamond CFG has no loops.
        let mut func = MachFunction::new(
            "diamond".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let bb1 = func.create_block();
        let bb2 = func.create_block();
        let bb3 = func.create_block();

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

        func.add_edge(bb0, bb1);
        func.add_edge(bb0, bb2);
        func.add_edge(bb1, bb3);
        func.add_edge(bb2, bb3);

        let dom = DomTree::compute(&func);
        let la = LoopAnalysis::compute(&func, &dom);

        assert!(la.is_empty());
    }

    #[test]
    fn test_block_to_loop_mapping() {
        let func = make_simple_loop();
        let dom = DomTree::compute(&func);
        let la = LoopAnalysis::compute(&func, &dom);

        assert!(la.is_in_loop(BlockId(1)));
        assert!(la.is_in_loop(BlockId(3)));
        assert!(!la.is_in_loop(BlockId(0)));
        assert!(!la.is_in_loop(BlockId(2)));

        let inner = la.containing_loop(BlockId(3)).unwrap();
        assert_eq!(inner.header, BlockId(1));
    }

    #[test]
    fn test_create_preheader() {
        let mut func = make_simple_loop();
        let dom = DomTree::compute(&func);
        let la = LoopAnalysis::compute(&func, &dom);

        let lp = la.get_loop(BlockId(1)).unwrap();
        // Already has a preheader (bb0), but test create_preheader anyway
        // by pretending there's a second non-loop predecessor.
        let body = lp.body.clone();

        // For a proper test, add a second entry to the loop header.
        let bb_extra = func.create_block();
        func.add_edge(bb_extra, BlockId(1));

        // Now create a preheader since there are two non-loop preds.
        let ph = create_preheader(&mut func, BlockId(1), &body);

        // Preheader should have an unconditional branch to header.
        let ph_block = func.block(ph);
        assert_eq!(ph_block.insts.len(), 1);
        let br = func.inst(ph_block.insts[0]);
        assert_eq!(br.opcode, AArch64Opcode::B);

        // Preheader should be a predecessor of header.
        assert!(func.block(BlockId(1)).preds.contains(&ph));
    }

    #[test]
    fn test_self_loop() {
        // bb0 -> bb1, bb1 -> bb1 (self-loop), bb1 -> bb2
        let mut func = MachFunction::new(
            "self_loop".to_string(),
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

        let br1 = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb1), MachOperand::Block(bb2)],
        ));
        func.append_inst(bb1, br1);

        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret);

        func.add_edge(bb0, bb1);
        func.add_edge(bb1, bb1);
        func.add_edge(bb1, bb2);

        let dom = DomTree::compute(&func);
        let la = LoopAnalysis::compute(&func, &dom);

        assert_eq!(la.num_loops(), 1);
        let lp = la.get_loop(BlockId(1)).unwrap();
        assert_eq!(lp.header, BlockId(1));
        assert_eq!(lp.latch, BlockId(1));
        assert_eq!(lp.body.len(), 1); // just bb1
    }
}
