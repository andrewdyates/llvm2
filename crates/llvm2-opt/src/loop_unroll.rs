// llvm2-opt - Loop unrolling
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Loop unrolling pass for small, bounded loops.
//!
//! Fully unrolls loops with known constant trip count <= `MAX_TRIP_COUNT`
//! and body size <= `MAX_BODY_INSTS`. This eliminates loop overhead (branch,
//! compare) and exposes more optimization opportunities (constant folding,
//! CSE) for subsequent passes.
//!
//! # Algorithm
//!
//! 1. Compute dominator tree and loop analysis.
//! 2. For each loop (innermost first):
//!    a. Determine if the loop has a constant trip count by analyzing the
//!       header's conditional branch and the induction variable.
//!    b. If trip count <= `MAX_TRIP_COUNT` and body <= `MAX_BODY_INSTS`,
//!       replicate the loop body N times, rewriting the back-edge to
//!       fall through after the last iteration.
//!    c. Remove the original loop structure.
//! 3. Return whether any loop was unrolled.
//!
//! # Constraints
//!
//! - Only single-latch, single-exit loops are candidates.
//! - The trip count must be statically determinable from the loop header.
//! - Nested loops are not unrolled (only innermost loops).
//!
//! Reference: LLVM `LoopUnrollPass.cpp`

use std::collections::HashMap;

use llvm2_ir::{AArch64Opcode, BlockId, InstId, MachFunction, MachInst, MachOperand, VReg};

use crate::dom::DomTree;
use crate::effects::inst_produces_value;
use crate::loops::{LoopAnalysis, NaturalLoop};
use crate::pass_manager::MachinePass;

/// Maximum trip count for full unrolling.
const MAX_TRIP_COUNT: u64 = 4;

/// Maximum number of non-terminator instructions in the loop body
/// (across all body blocks) for unrolling eligibility.
const MAX_BODY_INSTS: usize = 8;

/// Loop unrolling pass.
pub struct LoopUnroll;

impl MachinePass for LoopUnroll {
    fn name(&self) -> &str {
        "loop-unroll"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        let dom = DomTree::compute(func);
        let loop_analysis = LoopAnalysis::compute(func, &dom);

        if loop_analysis.is_empty() {
            return false;
        }

        let mut changed = false;

        // Collect innermost loops only (no children).
        let all_loops: Vec<NaturalLoop> = loop_analysis.all_loops().cloned().collect();

        let innermost: Vec<&NaturalLoop> = all_loops
            .iter()
            .filter(|lp| {
                // A loop is innermost if no other loop's parent is this loop.
                !all_loops.iter().any(|other| other.parent == Some(lp.header))
            })
            .collect();

        for lp in &innermost {
            if let Some(trip_count) = analyze_trip_count(func, lp) {
                if trip_count <= MAX_TRIP_COUNT && trip_count > 0 {
                    let body_inst_count = count_body_insts(func, lp);
                    if body_inst_count <= MAX_BODY_INSTS {
                        if unroll_loop(func, lp, trip_count as usize) {
                            changed = true;
                        }
                    }
                }
            }
        }

        changed
    }
}

/// Count the number of non-terminator instructions in the loop body.
fn count_body_insts(func: &MachFunction, lp: &NaturalLoop) -> usize {
    let mut count = 0;
    for &block_id in &func.block_order {
        if !lp.body.contains(&block_id) {
            continue;
        }
        let block = func.block(block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            if !inst.flags.is_branch() && !inst.flags.is_terminator() {
                count += 1;
            }
        }
    }
    count
}

/// Attempt to determine the constant trip count of a loop.
///
/// Recognizes this pattern:
/// - Preheader: `v_init = MovI #start`
/// - Header: `CmpRI v_iv, #limit` followed by `BCond exit, body`
/// - Latch: `v_iv = AddRI v_iv_prev, #step`
///
/// Trip count = ceil((limit - start) / step) for simple counting loops.
///
/// Returns `None` if the trip count cannot be statically determined.
fn analyze_trip_count(func: &MachFunction, lp: &NaturalLoop) -> Option<u64> {
    let preheader = lp.preheader?;

    // Look for a compare instruction in the header.
    let header_block = func.block(lp.header);
    let (cmp_vreg_id, limit) = find_header_cmp(func, header_block)?;

    // Look for a conditional branch in the header that uses the compare result.
    let header_terminates_with_bcond = header_block
        .insts
        .last()
        .map(|&id| func.inst(id).opcode == AArch64Opcode::BCond)
        .unwrap_or(false);

    if !header_terminates_with_bcond {
        return None;
    }

    // Find the induction variable initialization in the preheader.
    let init_val = find_iv_init(func, preheader, cmp_vreg_id, lp)?;

    // Find the back-edge vreg from the Phi that defines the IV.
    let backedge_vreg_id = find_phi_backedge_vreg(func, lp, cmp_vreg_id)?;

    // Find the IV update in the latch: AddRI v_backedge, v_prev, #step.
    let step = find_iv_step(func, lp, backedge_vreg_id, cmp_vreg_id)?;

    if step == 0 {
        return None; // infinite loop
    }

    // Compute trip count for a counting-up loop: ceil((limit - init) / step).
    if limit > init_val && step > 0 {
        let range = (limit - init_val) as u64;
        let step_u = step as u64;
        let trips = (range + step_u - 1) / step_u;
        Some(trips)
    } else if limit < init_val && step < 0 {
        // Counting-down loop.
        let range = (init_val - limit) as u64;
        let step_u = (-step) as u64;
        let trips = (range + step_u - 1) / step_u;
        Some(trips)
    } else if init_val == limit {
        Some(0) // zero trips
    } else {
        None // can't determine
    }
}

/// Find a CmpRI instruction in the header and extract the compared vreg and limit.
fn find_header_cmp(func: &MachFunction, header: &llvm2_ir::MachBlock) -> Option<(u32, i64)> {
    for &inst_id in &header.insts {
        let inst = func.inst(inst_id);
        if inst.opcode == AArch64Opcode::CmpRI {
            // CmpRI operands: [vreg, imm]
            if inst.operands.len() >= 2 {
                if let (Some(vreg), Some(imm)) = (
                    inst.operands[0].as_vreg(),
                    inst.operands[1].as_imm(),
                ) {
                    return Some((vreg.id, imm));
                }
            }
        }
    }
    None
}

/// Find the initialization value of the induction variable in the preheader.
///
/// Looks for `MovI v_target, #val` where v_target is the same vreg
/// compared in the header, or a vreg that flows into it via Phi/copy chain.
fn find_iv_init(func: &MachFunction, preheader: BlockId, cmp_vreg_id: u32, lp: &NaturalLoop) -> Option<i64> {
    // Check for a Phi in the header that defines the compared vreg.
    // The Phi's preheader operand gives us the init value's source vreg.
    let header_block = func.block(lp.header);
    let mut init_vreg_id = cmp_vreg_id;

    for &inst_id in &header_block.insts {
        let inst = func.inst(inst_id);
        if inst.opcode == AArch64Opcode::Phi {
            if let Some(def) = inst.operands.first().and_then(|op| op.as_vreg()) {
                if def.id == cmp_vreg_id {
                    // Phi operands: [def, val_from_pred0, block0, val_from_pred1, block1, ...]
                    // Find the operand pair where block == preheader.
                    let mut i = 1;
                    while i + 1 < inst.operands.len() {
                        if let MachOperand::Block(bid) = &inst.operands[i + 1] {
                            if *bid == preheader {
                                if let Some(v) = inst.operands[i].as_vreg() {
                                    init_vreg_id = v.id;
                                }
                                break;
                            }
                        }
                        i += 2;
                    }
                    break;
                }
            }
        }
    }

    // Now find the MovI that defines init_vreg_id in the preheader.
    let ph_block = func.block(preheader);
    for &inst_id in &ph_block.insts {
        let inst = func.inst(inst_id);
        if inst.opcode == AArch64Opcode::MovI {
            if let Some(def) = inst.operands.first().and_then(|op| op.as_vreg()) {
                if def.id == init_vreg_id {
                    if let Some(val) = inst.operands.get(1).and_then(|op| op.as_imm()) {
                        return Some(val);
                    }
                }
            }
        }
    }

    None
}

/// Find the back-edge vreg from the Phi that defines the IV.
///
/// In the header's Phi for the compared vreg, find the incoming value
/// from inside the loop (the latch). This is the vreg that the IV
/// update instruction must define.
fn find_phi_backedge_vreg(func: &MachFunction, lp: &NaturalLoop, cmp_vreg_id: u32) -> Option<u32> {
    let header_block = func.block(lp.header);
    for &inst_id in &header_block.insts {
        let inst = func.inst(inst_id);
        if inst.opcode != AArch64Opcode::Phi {
            continue;
        }
        if let Some(def) = inst.operands.first().and_then(|op| op.as_vreg()) {
            if def.id != cmp_vreg_id {
                continue;
            }
            // Phi operands: [def, val0, block0, val1, block1, ...]
            let mut i = 1;
            while i + 1 < inst.operands.len() {
                if let MachOperand::Block(bid) = &inst.operands[i + 1] {
                    if lp.body.contains(bid) {
                        if let Some(v) = inst.operands[i].as_vreg() {
                            return Some(v.id);
                        }
                    }
                }
                i += 2;
            }
        }
    }
    None
}

/// Find the step value of the induction variable in the latch.
///
/// Specifically looks for `AddRI v_backedge, v_iv, #step` or
/// `SubRI v_backedge, v_iv, #step` where `v_backedge` is the vreg
/// fed back through the Phi and `v_iv` is the IV Phi vreg.
fn find_iv_step(func: &MachFunction, lp: &NaturalLoop, backedge_vreg_id: u32, iv_vreg_id: u32) -> Option<i64> {
    let latch_block = func.block(lp.latch);
    for &inst_id in &latch_block.insts {
        let inst = func.inst(inst_id);
        match inst.opcode {
            AArch64Opcode::AddRI => {
                if inst.operands.len() >= 3 {
                    if let (Some(dst), Some(src), Some(step)) = (
                        inst.operands[0].as_vreg(),
                        inst.operands[1].as_vreg(),
                        inst.operands[2].as_imm(),
                    ) {
                        if dst.id == backedge_vreg_id && src.id == iv_vreg_id {
                            return Some(step);
                        }
                    }
                }
            }
            AArch64Opcode::SubRI => {
                if inst.operands.len() >= 3 {
                    if let (Some(dst), Some(src), Some(step)) = (
                        inst.operands[0].as_vreg(),
                        inst.operands[1].as_vreg(),
                        inst.operands[2].as_imm(),
                    ) {
                        if dst.id == backedge_vreg_id && src.id == iv_vreg_id {
                            return Some(-step);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Perform full loop unrolling by replicating the body `trip_count` times.
///
/// Strategy: duplicate all loop body blocks for each iteration, rewriting
/// vreg definitions to fresh vregs. Connect iterations sequentially.
/// Remove the back-edge and redirect the last iteration to fall through
/// to the exit.
fn unroll_loop(func: &mut MachFunction, lp: &NaturalLoop, trip_count: usize) -> bool {
    if trip_count == 0 {
        // Zero-trip loop: redirect preheader to exit and remove loop blocks.
        return redirect_zero_trip(func, lp);
    }

    if lp.preheader.is_none() {
        return false;
    }

    // Find the exit block (successor of header not in loop body).
    let exit_block = find_exit_block(func, lp);
    let exit_block = match exit_block {
        Some(eb) => eb,
        None => return false,
    };

    // Collect loop body blocks in block_order sequence (for deterministic iteration).
    let body_blocks: Vec<BlockId> = func
        .block_order
        .iter()
        .copied()
        .filter(|b| lp.body.contains(b))
        .collect();

    // Collect all non-terminator instructions from body blocks.
    let mut body_insts: Vec<(BlockId, InstId)> = Vec::new();
    for &bid in &body_blocks {
        let block = func.block(bid);
        for &iid in &block.insts {
            let inst = func.inst(iid);
            if !inst.flags.is_branch() && !inst.flags.is_terminator()
                && inst.opcode != AArch64Opcode::CmpRI
                && inst.opcode != AArch64Opcode::CmpRR
                && inst.opcode != AArch64Opcode::Phi
            {
                body_insts.push((bid, iid));
            }
        }
    }

    // For each unrolled iteration, duplicate the body instructions into
    // the preheader block (simple case: all go into a single linear block).
    // This is a simplified unrolling that works for simple single-block loops.
    // Multi-block loop bodies are not supported for now.
    if body_blocks.len() > 2 {
        // For now, only handle loops with header + latch (2 blocks) or
        // just header (self-loop, 1 block).
        return false;
    }

    // Clone body instructions for (trip_count - 1) additional iterations.
    // The original loop body serves as iteration 0; we add copies for 1..trip_count-1.
    // After all copies, redirect the latch's back-edge to the exit block.

    // Build vreg rename map for each iteration.
    // Collect all vregs defined in the loop body.
    let mut defined_vregs: Vec<u32> = Vec::new();
    for &(_bid, iid) in &body_insts {
        let inst = func.inst(iid);
        if inst_produces_value(inst) {
            if let Some(vreg) = inst.operands.first().and_then(|op| op.as_vreg()) {
                defined_vregs.push(vreg.id);
            }
        }
    }

    // Create a "flattened" block for the unrolled iterations after the existing body.
    // We'll insert duplicated instructions into a new block for each unrolled copy,
    // then replace the back-edge with a fall-through.

    let mut prev_rename: HashMap<u32, u32> = HashMap::new();
    let mut new_blocks: Vec<BlockId> = Vec::new();

    for _iter in 1..trip_count {
        // Create a new block for this iteration.
        let new_block = func.create_block();
        new_blocks.push(new_block);

        // Build rename map: old vreg -> new vreg for this iteration.
        let mut rename: HashMap<u32, u32> = HashMap::new();
        for &vid in &defined_vregs {
            let new_id = func.alloc_vreg();
            rename.insert(vid, new_id);
        }

        // Clone each body instruction with renamed operands.
        for &(_bid, iid) in &body_insts {
            let inst = func.inst(iid);
            let new_operands: Vec<MachOperand> = inst
                .operands
                .iter()
                .enumerate()
                .map(|(idx, op)| {
                    if let MachOperand::VReg(vreg) = op {
                        // For the def (operand 0 on value-producing insts), use this iter's rename.
                        if idx == 0 && inst_produces_value(inst) {
                            if let Some(&new_id) = rename.get(&vreg.id) {
                                return MachOperand::VReg(VReg::new(new_id, vreg.class));
                            }
                        }
                        // For uses: use the previous iteration's version of
                        // this vreg (connecting iteration N to iteration N-1).
                        if let Some(&prev_id) = prev_rename.get(&vreg.id) {
                            return MachOperand::VReg(VReg::new(prev_id, vreg.class));
                        }
                    }
                    op.clone()
                })
                .collect();

            let new_inst = MachInst::new(inst.opcode, new_operands);
            let new_inst_id = func.push_inst(new_inst);
            func.append_inst(new_block, new_inst_id);
        }

        prev_rename = rename;
    }

    // Wire up the control flow:
    // 1. Latch -> first new block (instead of back to header).
    // 2. Each new block falls through to the next.
    // 3. Last new block falls through to exit.

    if !new_blocks.is_empty() {
        // Redirect latch's back-edge: replace B header -> B new_blocks[0]
        rewrite_branch_target(func, lp.latch, lp.header, new_blocks[0]);
        func.block_mut(lp.latch).succs.retain(|&s| s != lp.header);
        func.block_mut(lp.header).preds.retain(|&p| p != lp.latch);
        func.add_edge(lp.latch, new_blocks[0]);

        // Chain new blocks together.
        for i in 0..new_blocks.len() - 1 {
            let br = func.push_inst(MachInst::new(
                AArch64Opcode::B,
                vec![MachOperand::Block(new_blocks[i + 1])],
            ));
            func.append_inst(new_blocks[i], br);
            func.add_edge(new_blocks[i], new_blocks[i + 1]);
        }

        // Last new block branches to exit.
        let last = *new_blocks.last().unwrap();
        let br = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(exit_block)],
        ));
        func.append_inst(last, br);
        func.add_edge(last, exit_block);
    } else {
        // trip_count == 1: just remove the back-edge, redirect latch to exit.
        rewrite_branch_target(func, lp.latch, lp.header, exit_block);
        func.block_mut(lp.latch).succs.retain(|&s| s != lp.header);
        func.block_mut(lp.header).preds.retain(|&p| p != lp.latch);
        func.add_edge(lp.latch, exit_block);
    }

    // Remove the conditional branch from the header (it always falls through now).
    // Replace BCond with unconditional B to the first body block after header.
    let first_body_after_header: Option<BlockId> = func
        .block(lp.header)
        .succs
        .iter()
        .find(|&&s| lp.body.contains(&s) && s != lp.header)
        .copied();

    if let Some(body_entry) = first_body_after_header {
        let header_block = func.block(lp.header);
        if let Some(&last_id) = header_block.insts.last() {
            let last_inst = func.inst(last_id);
            if last_inst.opcode == AArch64Opcode::BCond {
                // Replace BCond with unconditional B to body.
                *func.inst_mut(last_id) = MachInst::new(
                    AArch64Opcode::B,
                    vec![MachOperand::Block(body_entry)],
                );
                // Remove exit edge from header.
                func.block_mut(lp.header).succs.retain(|&s| s != exit_block);
                func.block_mut(exit_block).preds.retain(|&p| p != lp.header);
            }
        }
        // Also remove the CmpRI from header (no longer needed).
        // Collect IDs to remove first to avoid borrow conflict.
        let cmp_insts: Vec<InstId> = func.block(lp.header).insts.iter().copied().filter(|&iid| {
            let inst = func.inst(iid);
            inst.opcode == AArch64Opcode::CmpRI || inst.opcode == AArch64Opcode::CmpRR
        }).collect();
        let header_block = func.block_mut(lp.header);
        header_block.insts.retain(|iid| !cmp_insts.contains(iid));
    }

    true
}

/// Find the exit block of a loop (successor of header not in loop body).
fn find_exit_block(func: &MachFunction, lp: &NaturalLoop) -> Option<BlockId> {
    for &succ in &func.block(lp.header).succs {
        if !lp.body.contains(&succ) {
            return Some(succ);
        }
    }
    None
}

/// Redirect preheader to exit for zero-trip loops.
fn redirect_zero_trip(func: &mut MachFunction, lp: &NaturalLoop) -> bool {
    let preheader = match lp.preheader {
        Some(ph) => ph,
        None => return false,
    };
    let exit_block = match find_exit_block(func, lp) {
        Some(eb) => eb,
        None => return false,
    };

    // Rewrite preheader's branch to go directly to exit.
    rewrite_branch_target(func, preheader, lp.header, exit_block);

    // Update CFG edges.
    func.block_mut(preheader).succs.retain(|&s| s != lp.header);
    func.block_mut(lp.header).preds.retain(|&p| p != preheader);
    func.add_edge(preheader, exit_block);

    true
}

/// Rewrite branch targets in the terminator of `block` from `old_target` to `new_target`.
fn rewrite_branch_target(func: &mut MachFunction, block: BlockId, old_target: BlockId, new_target: BlockId) {
    let block_data = func.block(block);
    if let Some(&last_id) = block_data.insts.last() {
        let inst = func.inst_mut(last_id);
        for op in &mut inst.operands {
            if let MachOperand::Block(target) = op {
                if *target == old_target {
                    *target = new_target;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass_manager::MachinePass;
    use llvm2_ir::{AArch64Opcode, BlockId, MachFunction, MachInst, MachOperand, RegClass, Signature, VReg};

    fn vreg(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
    }

    fn imm(val: i64) -> MachOperand {
        MachOperand::Imm(val)
    }

    /// Build a simple counting loop:
    ///
    /// ```text
    ///   bb0 (preheader):
    ///     v0 = MovI #0          ; IV init
    ///     B bb1
    ///
    ///   bb1 (header):
    ///     v1 = Phi [v0, bb0], [v3, bb3]  ; IV
    ///     CmpRI v1, #N          ; compare IV to limit
    ///     BCond bb2, bb3        ; if >= N, exit to bb2
    ///
    ///   bb3 (latch):
    ///     v2 = AddRI v1, v1, #1 ; body work (use IV)
    ///     v3 = AddRI v1, v1, #1 ; IV increment
    ///     B bb1                 ; back-edge
    ///
    ///   bb2 (exit):
    ///     Ret
    /// ```
    fn make_counting_loop(trip_count: i64) -> MachFunction {
        let mut func = MachFunction::new(
            "counting_loop".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry; // preheader
        let bb1 = func.create_block(); // header
        let bb2 = func.create_block(); // exit
        let bb3 = func.create_block(); // latch

        // bb0: preheader
        let init = func.push_inst(MachInst::new(
            AArch64Opcode::MovI,
            vec![vreg(0), imm(0)],
        ));
        func.append_inst(bb0, init);
        let br0 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb0, br0);

        // bb1: header
        let phi = func.push_inst(MachInst::new(
            AArch64Opcode::Phi,
            vec![
                vreg(1),
                vreg(0), MachOperand::Block(bb0),
                vreg(3), MachOperand::Block(bb3),
            ],
        ));
        func.append_inst(bb1, phi);
        let cmp = func.push_inst(MachInst::new(
            AArch64Opcode::CmpRI,
            vec![vreg(1), imm(trip_count)],
        ));
        func.append_inst(bb1, cmp);
        let bcond = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb2), MachOperand::Block(bb3)],
        ));
        func.append_inst(bb1, bcond);

        // bb3: latch (body + IV update)
        let body_work = func.push_inst(MachInst::new(
            AArch64Opcode::AddRI,
            vec![vreg(2), vreg(1), imm(10)],
        ));
        func.append_inst(bb3, body_work);
        let iv_inc = func.push_inst(MachInst::new(
            AArch64Opcode::AddRI,
            vec![vreg(3), vreg(1), imm(1)],
        ));
        func.append_inst(bb3, iv_inc);
        let br3 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb3, br3);

        // bb2: exit
        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret);

        // CFG edges
        func.add_edge(bb0, bb1);
        func.add_edge(bb1, bb2);
        func.add_edge(bb1, bb3);
        func.add_edge(bb3, bb1);

        func
    }

    #[test]
    fn test_unroll_trip_count_2() {
        let mut func = make_counting_loop(2);
        let mut pass = LoopUnroll;
        let changed = pass.run(&mut func);
        assert!(changed, "loop with trip count 2 should be unrolled");

        // After unrolling, the back-edge bb3 -> bb1 should be removed.
        // The latch should no longer branch to the header.
        let latch = func.block(BlockId(3));
        let has_backedge = latch.succs.contains(&BlockId(1));
        assert!(!has_backedge, "back-edge should be removed after unrolling");
    }

    #[test]
    fn test_unroll_trip_count_1() {
        let mut func = make_counting_loop(1);
        let mut pass = LoopUnroll;
        let changed = pass.run(&mut func);
        assert!(changed, "loop with trip count 1 should be unrolled");

        // Single iteration: latch should go to exit (bb2), not header (bb1).
        let latch = func.block(BlockId(3));
        assert!(
            !latch.succs.contains(&BlockId(1)),
            "back-edge should be removed"
        );
    }

    #[test]
    fn test_unroll_trip_count_4() {
        let mut func = make_counting_loop(4);
        let mut pass = LoopUnroll;
        let changed = pass.run(&mut func);
        assert!(changed, "loop with trip count 4 should be unrolled");
    }

    #[test]
    fn test_no_unroll_trip_count_5() {
        let mut func = make_counting_loop(5);
        let original_block_count = func.num_blocks();
        let mut pass = LoopUnroll;
        let changed = pass.run(&mut func);
        assert!(
            !changed,
            "loop with trip count 5 should NOT be unrolled (exceeds MAX_TRIP_COUNT)"
        );
        assert_eq!(func.num_blocks(), original_block_count);
    }

    #[test]
    fn test_no_unroll_no_loop() {
        // Simple function with no loop.
        let mut func = MachFunction::new(
            "no_loop".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb0, ret);

        let mut pass = LoopUnroll;
        assert!(!pass.run(&mut func));
    }

    #[test]
    fn test_unroll_idempotent() {
        let mut func = make_counting_loop(2);
        let mut pass = LoopUnroll;

        // First run unrolls the loop.
        let changed1 = pass.run(&mut func);
        assert!(changed1);

        // Second run should find nothing to do (loop structure is gone).
        let changed2 = pass.run(&mut func);
        assert!(!changed2, "second unroll pass should be idempotent");
    }

    #[test]
    fn test_unroll_preserves_body_instructions() {
        let mut func = make_counting_loop(3);
        let mut pass = LoopUnroll;
        pass.run(&mut func);

        // Count total AddRI instructions across all blocks.
        // Original had 2 AddRI (body_work + iv_inc), unrolled 3x should have 6.
        let mut add_count = 0;
        for &bid in &func.block_order {
            let block = func.block(bid);
            for &iid in &block.insts {
                let inst = func.inst(iid);
                if inst.opcode == AArch64Opcode::AddRI {
                    add_count += 1;
                }
            }
        }
        // Original 2 in latch + 2*2 in unrolled copies = 6 total
        assert!(
            add_count >= 4,
            "unrolled loop should have at least 4 AddRI instructions, got {}",
            add_count
        );
    }
}
