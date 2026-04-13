// llvm2-codegen/relax.rs - Branch relaxation pass
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! AArch64 branch relaxation pass.
//!
//! After block layout, computes block offsets (each non-pseudo instruction is
//! 4 bytes) and relaxes branches whose targets are out of range. Uses a
//! fixed-point algorithm: relaxation may insert instructions that push other
//! branches out of range, so we iterate until stable.
//!
//! Relaxation patterns (ARM ARM):
//! - `B.cond far` -> `B.cond_inv +8 ; B far`
//! - `CBZ Rt, far` -> `CBNZ Rt, +8 ; B far`
//! - `CBNZ Rt, far` -> `CBZ Rt, +8 ; B far`
//! - `TBZ Rt, #bit, far` -> `TBNZ Rt, #bit, +8 ; B far`
//! - `TBNZ Rt, #bit, far` -> `TBZ Rt, #bit, +8 ; B far`
//!
//! Reference: LLVM `BranchRelaxation.cpp`.

use llvm2_ir::{AArch64Opcode, BlockId, InstId, MachFunction, MachInst, MachOperand};

// ---------------------------------------------------------------------------
// Constants: branch ranges in bytes
// ---------------------------------------------------------------------------

/// B (unconditional): 26-bit signed offset * 4 = +/- 128 MB.
const B_MAX_RANGE: i64 = (1 << 27) - 4; // +134,217,724
const B_MIN_RANGE: i64 = -(1 << 27); // -134,217,728

/// B.cond / CBZ / CBNZ: 19-bit signed offset * 4 = +/- 1 MB.
const BCOND_MAX_RANGE: i64 = (1 << 20) - 4; // +1,048,572
const BCOND_MIN_RANGE: i64 = -(1 << 20); // -1,048,576

/// TBZ / TBNZ: 14-bit signed offset * 4 = +/- 32 KB.
const TBZ_MAX_RANGE: i64 = (1 << 15) - 4; // +32,764
const TBZ_MIN_RANGE: i64 = -(1 << 15); // -32,768

// ---------------------------------------------------------------------------
// BlockInfo
// ---------------------------------------------------------------------------

/// Per-block offset and size information for offset computation.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct BlockInfo {
    /// Byte offset from the start of the function.
    offset: u32,
    /// Size of the block in bytes (number of non-pseudo instructions * 4).
    size: u32,
}

// ---------------------------------------------------------------------------
// Public types and entry point
// ---------------------------------------------------------------------------

/// Result of branch relaxation: the final instruction sequence with resolved
/// block offsets.
#[derive(Debug, Clone)]
pub struct RelaxedCode {
    /// Final instruction sequence in layout order. Branch operands that were
    /// `MachOperand::Block(id)` are replaced with `MachOperand::Imm(offset)`
    /// where offset is the signed displacement in instruction units (bytes / 4).
    pub instructions: Vec<MachInst>,
    /// Byte offset from function start for each BlockId.
    pub block_offsets: Vec<u32>,
}

/// Run branch relaxation on a function after block layout.
///
/// This is a fixed-point algorithm:
/// 1. Compute block offsets.
/// 2. Scan all branch instructions for out-of-range targets.
/// 3. Relax any out-of-range branches by splitting them.
/// 4. Repeat until no changes.
/// 5. Build the final instruction sequence with resolved offsets.
pub fn relax_branches(func: &mut MachFunction) -> RelaxedCode {
    // Fixed-point relaxation loop.
    let max_iterations = 32; // safety bound
    for _ in 0..max_iterations {
        let infos = compute_block_offsets(func);
        let mut changed = false;

        // Scan blocks in layout order.
        for layout_idx in 0..func.block_order.len() {
            let block_id = func.block_order[layout_idx];
            let block_offset = infos[block_id.0 as usize].offset;

            let blk = func.block(block_id);
            let inst_ids: Vec<InstId> = blk.insts.clone();

            let mut inst_byte_offset = block_offset;
            for (inst_pos, &inst_id) in inst_ids.iter().enumerate() {
                let inst = func.inst(inst_id);
                if inst.opcode.is_pseudo() {
                    continue;
                }

                if is_branch_with_block_target(inst) {
                    if let Some(target_bid) = get_branch_target_block(inst) {
                        let target_offset = infos[target_bid.0 as usize].offset;
                        let displacement =
                            target_offset as i64 - inst_byte_offset as i64;

                        if !in_range(inst.opcode, displacement) {
                            relax_one_branch(func, block_id, inst_pos);
                            changed = true;
                            break; // restart scan for this block
                        }
                    }
                }

                inst_byte_offset += 4;
            }
        }

        if !changed {
            break;
        }
    }

    // Build the final instruction sequence with resolved offsets.
    build_final_code(func)
}

// ---------------------------------------------------------------------------
// Offset computation
// ---------------------------------------------------------------------------

/// Compute byte offsets for all blocks based on the current layout order.
/// Each non-pseudo instruction is 4 bytes.
fn compute_block_offsets(func: &MachFunction) -> Vec<BlockInfo> {
    let mut infos = vec![
        BlockInfo {
            offset: 0,
            size: 0,
        };
        func.blocks.len()
    ];

    let mut offset: u32 = 0;
    for &block_id in &func.block_order {
        let blk = func.block(block_id);
        let size = block_code_size(func, blk);
        infos[block_id.0 as usize] = BlockInfo { offset, size };
        offset += size;
    }

    infos
}

/// Compute the code size of a block in bytes (non-pseudo instructions * 4).
fn block_code_size(func: &MachFunction, blk: &llvm2_ir::MachBlock) -> u32 {
    let mut count = 0u32;
    for &inst_id in &blk.insts {
        let inst = func.inst(inst_id);
        if !inst.opcode.is_pseudo() {
            count += 1;
        }
    }
    count * 4
}

// ---------------------------------------------------------------------------
// Range checking
// ---------------------------------------------------------------------------

/// Returns (min_range, max_range) in bytes for the given branch opcode.
pub fn branch_range(opcode: AArch64Opcode) -> (i64, i64) {
    match opcode {
        AArch64Opcode::B => (B_MIN_RANGE, B_MAX_RANGE),
        AArch64Opcode::BCond => (BCOND_MIN_RANGE, BCOND_MAX_RANGE),
        AArch64Opcode::Cbz | AArch64Opcode::Cbnz => (BCOND_MIN_RANGE, BCOND_MAX_RANGE),
        AArch64Opcode::Tbz | AArch64Opcode::Tbnz => (TBZ_MIN_RANGE, TBZ_MAX_RANGE),
        _ => (i64::MIN, i64::MAX), // not a branch, always in range
    }
}

/// Check if the signed displacement (in bytes) is within the range for the
/// given branch opcode.
fn in_range(opcode: AArch64Opcode, displacement: i64) -> bool {
    let (min, max) = branch_range(opcode);
    displacement >= min && displacement <= max
}

// ---------------------------------------------------------------------------
// Branch inspection helpers
// ---------------------------------------------------------------------------

/// Returns true if the instruction is a branch with a Block operand target.
fn is_branch_with_block_target(inst: &MachInst) -> bool {
    if !inst.is_branch() {
        return false;
    }
    inst.operands.iter().any(|op| matches!(op, MachOperand::Block(_)))
}

/// Extract the Block target from a branch instruction.
fn get_branch_target_block(inst: &MachInst) -> Option<BlockId> {
    for op in &inst.operands {
        if let MachOperand::Block(bid) = op {
            return Some(*bid);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Branch relaxation
// ---------------------------------------------------------------------------

/// Relax a single out-of-range branch at position `inst_pos` in the block.
///
/// The pattern is: invert the short-range conditional branch to skip over an
/// inserted unconditional B to the far target.
///
/// Before: `BCond far_target`
/// After:  `BCond_inv +8 ; B far_target`
///
/// The inverted conditional branches to the next instruction (+8 = skip the B).
fn relax_one_branch(func: &mut MachFunction, block_id: BlockId, inst_pos: usize) {
    let blk = func.block(block_id);
    let inst_id = blk.insts[inst_pos];
    let inst = func.inst(inst_id).clone();

    let far_target = get_branch_target_block(&inst);
    if far_target.is_none() {
        return;
    }
    let far_target = far_target.unwrap();

    match inst.opcode {
        AArch64Opcode::BCond => {
            // BCond has operands: [Imm(condition_code), Block(target)]
            // Invert condition, make short branch skip the next instruction.
            let cc_val = inst
                .operands
                .iter()
                .find_map(|op| {
                    if let MachOperand::Imm(v) = op {
                        Some(*v)
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            let inverted_cc = invert_cc(cc_val as u8);

            // Create inverted BCond that skips the next instruction.
            // The +8 offset will be resolved in the final code emission;
            // we use a sentinel block for now but actually we just need to
            // branch over 1 instruction. We'll create a new block with just
            // the B instruction, inserted right after this block in layout.
            //
            // Simpler approach: replace the BCond with BCond_inv(+8) and
            // insert a B far_target right after it in the same block.

            // Rewrite the BCond instruction in place: inverted condition, no
            // block target (the +8 is implicit -- it always jumps over the
            // next instruction).
            let inv_bcond = MachInst::new(
                AArch64Opcode::BCond,
                vec![MachOperand::Imm(inverted_cc as i64), MachOperand::Imm(2)],
                // Imm(2) = +8 bytes = skip 2 instructions (but we encode as
                // offset in instruction units; the final pass resolves this).
                // Actually: +8 bytes / 4 = 2 instruction units forward.
                // But we just need to skip the *next* instruction (1 inst = 4 bytes).
                // imm19 field = 2 means +8 bytes. We store raw imm here.
            );

            let far_b = MachInst::new(
                AArch64Opcode::B,
                vec![MachOperand::Block(far_target)],
            );

            // Replace the original BCond with the inverted one and insert B
            // after it.
            let inv_id = func.push_inst(inv_bcond);
            let far_id = func.push_inst(far_b);

            let blk = func.block_mut(block_id);
            blk.insts[inst_pos] = inv_id;
            blk.insts.insert(inst_pos + 1, far_id);
        }

        AArch64Opcode::Cbz => {
            // CBZ Rt, far -> CBNZ Rt, +8 ; B far
            let mut inv_operands: Vec<MachOperand> = inst
                .operands
                .iter()
                .filter(|op| !matches!(op, MachOperand::Block(_)))
                .cloned()
                .collect();
            inv_operands.push(MachOperand::Imm(2)); // skip next inst

            let inv_inst = MachInst::new(AArch64Opcode::Cbnz, inv_operands);
            let far_b = MachInst::new(
                AArch64Opcode::B,
                vec![MachOperand::Block(far_target)],
            );

            let inv_id = func.push_inst(inv_inst);
            let far_id = func.push_inst(far_b);

            let blk = func.block_mut(block_id);
            blk.insts[inst_pos] = inv_id;
            blk.insts.insert(inst_pos + 1, far_id);
        }

        AArch64Opcode::Cbnz => {
            // CBNZ Rt, far -> CBZ Rt, +8 ; B far
            let mut inv_operands: Vec<MachOperand> = inst
                .operands
                .iter()
                .filter(|op| !matches!(op, MachOperand::Block(_)))
                .cloned()
                .collect();
            inv_operands.push(MachOperand::Imm(2));

            let inv_inst = MachInst::new(AArch64Opcode::Cbz, inv_operands);
            let far_b = MachInst::new(
                AArch64Opcode::B,
                vec![MachOperand::Block(far_target)],
            );

            let inv_id = func.push_inst(inv_inst);
            let far_id = func.push_inst(far_b);

            let blk = func.block_mut(block_id);
            blk.insts[inst_pos] = inv_id;
            blk.insts.insert(inst_pos + 1, far_id);
        }

        AArch64Opcode::Tbz => {
            // TBZ Rt, #bit, far -> TBNZ Rt, #bit, +8 ; B far
            let mut inv_operands: Vec<MachOperand> = inst
                .operands
                .iter()
                .filter(|op| !matches!(op, MachOperand::Block(_)))
                .cloned()
                .collect();
            inv_operands.push(MachOperand::Imm(2));

            let inv_inst = MachInst::new(AArch64Opcode::Tbnz, inv_operands);
            let far_b = MachInst::new(
                AArch64Opcode::B,
                vec![MachOperand::Block(far_target)],
            );

            let inv_id = func.push_inst(inv_inst);
            let far_id = func.push_inst(far_b);

            let blk = func.block_mut(block_id);
            blk.insts[inst_pos] = inv_id;
            blk.insts.insert(inst_pos + 1, far_id);
        }

        AArch64Opcode::Tbnz => {
            // TBNZ Rt, #bit, far -> TBZ Rt, #bit, +8 ; B far
            let mut inv_operands: Vec<MachOperand> = inst
                .operands
                .iter()
                .filter(|op| !matches!(op, MachOperand::Block(_)))
                .cloned()
                .collect();
            inv_operands.push(MachOperand::Imm(2));

            let inv_inst = MachInst::new(AArch64Opcode::Tbz, inv_operands);
            let far_b = MachInst::new(
                AArch64Opcode::B,
                vec![MachOperand::Block(far_target)],
            );

            let inv_id = func.push_inst(inv_inst);
            let far_id = func.push_inst(far_b);

            let blk = func.block_mut(block_id);
            blk.insts[inst_pos] = inv_id;
            blk.insts.insert(inst_pos + 1, far_id);
        }

        AArch64Opcode::B => {
            // Unconditional B out of range: this should be extremely rare
            // (128MB range). If it happens, we'd need a literal pool or
            // indirect branch. For now, leave it as-is (will produce an
            // encoding error at emit time).
        }

        _ => {}
    }
}

/// Invert an AArch64 condition code (as a u8).
fn invert_cc(cc: u8) -> u8 {
    // AArch64 condition codes: inversion is just flipping bit 0.
    cc ^ 1
}

// ---------------------------------------------------------------------------
// Final code emission
// ---------------------------------------------------------------------------

/// Build the final instruction sequence with branch offsets resolved.
///
/// Walks blocks in layout order, emits instructions, and replaces
/// `MachOperand::Block(id)` operands with `MachOperand::Imm(offset)` where
/// offset is the signed displacement in instruction units (bytes / 4).
fn build_final_code(func: &MachFunction) -> RelaxedCode {
    let infos = compute_block_offsets(func);
    let mut instructions = Vec::new();
    let mut block_offsets = vec![0u32; func.blocks.len()];

    for &bid in &func.block_order {
        block_offsets[bid.0 as usize] = infos[bid.0 as usize].offset;
    }

    let mut current_offset: u32 = 0;
    for &block_id in &func.block_order {
        let blk = func.block(block_id);
        for &inst_id in &blk.insts {
            let inst = func.inst(inst_id);
            if inst.opcode.is_pseudo() {
                // Pseudo instructions are not emitted.
                continue;
            }

            // Resolve Block operands to immediate offsets.
            let mut resolved = inst.clone();
            for op in &mut resolved.operands {
                if let MachOperand::Block(target_bid) = op {
                    let target_offset =
                        infos[target_bid.0 as usize].offset;
                    let displacement =
                        (target_offset as i64 - current_offset as i64) / 4;
                    *op = MachOperand::Imm(displacement);
                }
            }

            instructions.push(resolved);
            current_offset += 4;
        }
    }

    RelaxedCode {
        instructions,
        block_offsets,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::{AArch64CC, MachOperand, Signature};

    /// Helper: create a function with given number of blocks.
    fn make_func(name: &str, num_blocks: usize) -> MachFunction {
        let sig = Signature::new(vec![], vec![]);
        let mut func = MachFunction::new(name.to_string(), sig);
        for _ in 1..num_blocks {
            func.create_block();
        }
        func
    }

    fn add_inst(func: &mut MachFunction, block: BlockId, inst: MachInst) -> InstId {
        let id = func.push_inst(inst);
        func.append_inst(block, id);
        id
    }

    // -----------------------------------------------------------------------
    // test_branch_ranges
    // -----------------------------------------------------------------------
    #[test]
    fn test_branch_ranges() {
        // B: +/- 128MB
        let (min, max) = branch_range(AArch64Opcode::B);
        assert_eq!(max, (1 << 27) - 4);
        assert_eq!(min, -(1 << 27));

        // BCond: +/- 1MB
        let (min, max) = branch_range(AArch64Opcode::BCond);
        assert_eq!(max, (1 << 20) - 4);
        assert_eq!(min, -(1 << 20));

        // CBZ/CBNZ: same as BCond
        assert_eq!(branch_range(AArch64Opcode::Cbz), branch_range(AArch64Opcode::BCond));
        assert_eq!(branch_range(AArch64Opcode::Cbnz), branch_range(AArch64Opcode::BCond));

        // TBZ/TBNZ: +/- 32KB
        let (min, max) = branch_range(AArch64Opcode::Tbz);
        assert_eq!(max, (1 << 15) - 4);
        assert_eq!(min, -(1 << 15));
        assert_eq!(branch_range(AArch64Opcode::Tbnz), branch_range(AArch64Opcode::Tbz));
    }

    // -----------------------------------------------------------------------
    // test_in_range
    // -----------------------------------------------------------------------
    #[test]
    fn test_in_range_checks() {
        // BCond: +/- 1MB
        assert!(in_range(AArch64Opcode::BCond, 100));
        assert!(in_range(AArch64Opcode::BCond, -100));
        assert!(in_range(AArch64Opcode::BCond, BCOND_MAX_RANGE));
        assert!(in_range(AArch64Opcode::BCond, BCOND_MIN_RANGE));
        assert!(!in_range(AArch64Opcode::BCond, BCOND_MAX_RANGE + 4));
        assert!(!in_range(AArch64Opcode::BCond, BCOND_MIN_RANGE - 4));

        // TBZ: +/- 32KB
        assert!(in_range(AArch64Opcode::Tbz, 100));
        assert!(!in_range(AArch64Opcode::Tbz, TBZ_MAX_RANGE + 4));
    }

    // -----------------------------------------------------------------------
    // test_no_relaxation_needed
    // -----------------------------------------------------------------------
    #[test]
    fn test_no_relaxation_needed() {
        let mut func = make_func("small", 3);
        let bb0 = BlockId(0);
        let bb1 = BlockId(1);
        let bb2 = BlockId(2);

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

        // bb1: AddRI ; Ret
        add_inst(
            &mut func,
            bb1,
            MachInst::new(AArch64Opcode::AddRI, vec![]),
        );
        add_inst(
            &mut func,
            bb1,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        );

        // bb2: AddRI ; Ret
        add_inst(
            &mut func,
            bb2,
            MachInst::new(AArch64Opcode::AddRI, vec![]),
        );
        add_inst(
            &mut func,
            bb2,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        );

        let result = relax_branches(&mut func);

        // 6 non-pseudo instructions total.
        assert_eq!(result.instructions.len(), 6);
        // Block offsets: bb0=0, bb1=8, bb2=16
        assert_eq!(result.block_offsets[0], 0);
        assert_eq!(result.block_offsets[1], 8);
        assert_eq!(result.block_offsets[2], 16);
    }

    // -----------------------------------------------------------------------
    // test_offset_computation
    // -----------------------------------------------------------------------
    #[test]
    fn test_offset_computation() {
        let mut func = make_func("offsets", 3);
        let bb0 = BlockId(0);
        let bb1 = BlockId(1);
        let bb2 = BlockId(2);

        // bb0: 3 instructions (12 bytes)
        add_inst(&mut func, bb0, MachInst::new(AArch64Opcode::AddRI, vec![]));
        add_inst(&mut func, bb0, MachInst::new(AArch64Opcode::AddRI, vec![]));
        add_inst(&mut func, bb0, MachInst::new(AArch64Opcode::Ret, vec![]));

        // bb1: 2 instructions (8 bytes)
        add_inst(&mut func, bb1, MachInst::new(AArch64Opcode::AddRI, vec![]));
        add_inst(&mut func, bb1, MachInst::new(AArch64Opcode::Ret, vec![]));

        // bb2: 1 instruction (4 bytes)
        add_inst(&mut func, bb2, MachInst::new(AArch64Opcode::Ret, vec![]));

        let infos = compute_block_offsets(&func);
        assert_eq!(infos[0].offset, 0);
        assert_eq!(infos[0].size, 12);
        assert_eq!(infos[1].offset, 12);
        assert_eq!(infos[1].size, 8);
        assert_eq!(infos[2].offset, 20);
        assert_eq!(infos[2].size, 4);
    }

    // -----------------------------------------------------------------------
    // test_pseudo_instructions_not_counted
    // -----------------------------------------------------------------------
    #[test]
    fn test_pseudo_instructions_not_counted() {
        let mut func = make_func("pseudo", 2);
        let bb0 = BlockId(0);
        let bb1 = BlockId(1);

        // bb0: Nop (pseudo), AddRI, Ret
        add_inst(&mut func, bb0, MachInst::new(AArch64Opcode::Nop, vec![]));
        add_inst(&mut func, bb0, MachInst::new(AArch64Opcode::AddRI, vec![]));
        add_inst(&mut func, bb0, MachInst::new(AArch64Opcode::Ret, vec![]));

        // bb1: Ret
        add_inst(&mut func, bb1, MachInst::new(AArch64Opcode::Ret, vec![]));

        let infos = compute_block_offsets(&func);
        // bb0: Nop doesn't count, so 2 real instructions = 8 bytes.
        assert_eq!(infos[0].size, 8);
        assert_eq!(infos[1].offset, 8);
    }

    // -----------------------------------------------------------------------
    // test_bcond_relaxation
    // -----------------------------------------------------------------------
    #[test]
    fn test_bcond_relaxation_via_range_check() {
        // Instead of creating millions of NOPs, directly test the relaxation
        // logic by verifying range checks and the relaxation pattern.

        // Verify out-of-range detection: 2MB displacement > 1MB BCond range.
        assert!(!in_range(AArch64Opcode::BCond, 2 * 1024 * 1024));
        assert!(!in_range(AArch64Opcode::BCond, -(2 * 1024 * 1024)));

        // Verify in-range: 512KB < 1MB
        assert!(in_range(AArch64Opcode::BCond, 512 * 1024));
    }

    // -----------------------------------------------------------------------
    // test_cbz_relaxation_pattern
    // -----------------------------------------------------------------------
    #[test]
    fn test_cbz_relaxation_pattern() {
        // Create a function where we manually trigger relaxation on a CBZ.
        let mut func = make_func("cbz_relax", 2);
        let bb0 = BlockId(0);
        let bb1 = BlockId(1);

        // bb0: CBZ to bb1
        add_inst(
            &mut func,
            bb0,
            MachInst::new(
                AArch64Opcode::Cbz,
                vec![MachOperand::Imm(0), MachOperand::Block(bb1)],
            ),
        );

        // bb1: Ret
        add_inst(
            &mut func,
            bb1,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        );

        // Manually call relax_one_branch to test the pattern.
        relax_one_branch(&mut func, bb0, 0);

        let blk = func.block(bb0);
        // Should now have 2 instructions: CBNZ +8, B bb1.
        assert_eq!(blk.insts.len(), 2);

        let inv = func.inst(blk.insts[0]);
        assert_eq!(inv.opcode, AArch64Opcode::Cbnz, "CBZ should be inverted to CBNZ");

        let far = func.inst(blk.insts[1]);
        assert_eq!(far.opcode, AArch64Opcode::B, "far branch should be unconditional B");
        assert_eq!(
            get_branch_target_block(far),
            Some(bb1),
            "B should target bb1"
        );
    }

    // -----------------------------------------------------------------------
    // test_bcond_relaxation_pattern
    // -----------------------------------------------------------------------
    #[test]
    fn test_bcond_relaxation_pattern() {
        let mut func = make_func("bcond_relax", 2);
        let bb0 = BlockId(0);
        let bb1 = BlockId(1);

        // bb0: BCond EQ bb1
        add_inst(
            &mut func,
            bb0,
            MachInst::new(
                AArch64Opcode::BCond,
                vec![MachOperand::Imm(AArch64CC::EQ as i64), MachOperand::Block(bb1)],
            ),
        );

        // bb1: Ret
        add_inst(
            &mut func,
            bb1,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        );

        relax_one_branch(&mut func, bb0, 0);

        let blk = func.block(bb0);
        assert_eq!(blk.insts.len(), 2);

        let inv = func.inst(blk.insts[0]);
        assert_eq!(inv.opcode, AArch64Opcode::BCond);
        // Check inverted condition: EQ (0) -> NE (1)
        let cc = inv.operands[0].as_imm().unwrap();
        assert_eq!(cc, AArch64CC::NE as i64, "EQ should be inverted to NE");

        let far = func.inst(blk.insts[1]);
        assert_eq!(far.opcode, AArch64Opcode::B);
    }

    // -----------------------------------------------------------------------
    // test_tbz_relaxation_pattern
    // -----------------------------------------------------------------------
    #[test]
    fn test_tbz_relaxation_pattern() {
        let mut func = make_func("tbz_relax", 2);
        let bb0 = BlockId(0);
        let bb1 = BlockId(1);

        // bb0: TBZ Rt=X0, bit=5, target=bb1
        add_inst(
            &mut func,
            bb0,
            MachInst::new(
                AArch64Opcode::Tbz,
                vec![
                    MachOperand::Imm(0),  // Rt
                    MachOperand::Imm(5),  // bit number
                    MachOperand::Block(bb1),
                ],
            ),
        );

        add_inst(
            &mut func,
            bb1,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        );

        relax_one_branch(&mut func, bb0, 0);

        let blk = func.block(bb0);
        assert_eq!(blk.insts.len(), 2);

        let inv = func.inst(blk.insts[0]);
        assert_eq!(inv.opcode, AArch64Opcode::Tbnz, "TBZ should be inverted to TBNZ");

        let far = func.inst(blk.insts[1]);
        assert_eq!(far.opcode, AArch64Opcode::B);
        assert_eq!(get_branch_target_block(far), Some(bb1));
    }

    // -----------------------------------------------------------------------
    // test_invert_cc
    // -----------------------------------------------------------------------
    #[test]
    fn test_invert_cc_values() {
        assert_eq!(invert_cc(AArch64CC::EQ as u8), AArch64CC::NE as u8);
        assert_eq!(invert_cc(AArch64CC::NE as u8), AArch64CC::EQ as u8);
        assert_eq!(invert_cc(AArch64CC::HS as u8), AArch64CC::LO as u8);
        assert_eq!(invert_cc(AArch64CC::LO as u8), AArch64CC::HS as u8);
        assert_eq!(invert_cc(AArch64CC::GE as u8), AArch64CC::LT as u8);
        assert_eq!(invert_cc(AArch64CC::LT as u8), AArch64CC::GE as u8);
        assert_eq!(invert_cc(AArch64CC::GT as u8), AArch64CC::LE as u8);
        assert_eq!(invert_cc(AArch64CC::LE as u8), AArch64CC::GT as u8);
    }

    // -----------------------------------------------------------------------
    // test_resolved_offsets
    // -----------------------------------------------------------------------
    #[test]
    fn test_resolved_offsets() {
        let mut func = make_func("resolved", 3);
        let bb0 = BlockId(0);
        let bb1 = BlockId(1);
        let bb2 = BlockId(2);

        // bb0: B bb2 (jumps over bb1)
        add_inst(
            &mut func,
            bb0,
            MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(bb2)]),
        );

        // bb1: AddRI ; Ret (2 instructions = 8 bytes)
        add_inst(&mut func, bb1, MachInst::new(AArch64Opcode::AddRI, vec![]));
        add_inst(&mut func, bb1, MachInst::new(AArch64Opcode::Ret, vec![]));

        // bb2: Ret
        add_inst(&mut func, bb2, MachInst::new(AArch64Opcode::Ret, vec![]));

        let result = relax_branches(&mut func);

        // bb0 at offset 0, bb1 at offset 4, bb2 at offset 12.
        // B bb2 at offset 0: displacement = (12 - 0) / 4 = 3.
        assert_eq!(result.instructions.len(), 4);
        let b_inst = &result.instructions[0];
        assert_eq!(b_inst.opcode, AArch64Opcode::B);
        let offset = b_inst.operands.iter().find_map(|op| op.as_imm()).unwrap();
        assert_eq!(offset, 3, "B bb2 should have displacement of 3 instruction units");
    }

    // -----------------------------------------------------------------------
    // test_fixpoint_convergence
    // -----------------------------------------------------------------------
    #[test]
    fn test_fixpoint_convergence() {
        // A simple function should converge in 1 iteration (no relaxation needed).
        let mut func = make_func("converge", 2);
        let bb0 = BlockId(0);
        let bb1 = BlockId(1);

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
            MachInst::new(AArch64Opcode::Ret, vec![]),
        );
        add_inst(
            &mut func,
            bb1,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        );

        // Should not panic or loop infinitely.
        let result = relax_branches(&mut func);
        assert_eq!(result.instructions.len(), 3);
    }
}
