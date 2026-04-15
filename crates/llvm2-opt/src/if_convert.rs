// llvm2-opt - AArch64 If-Conversion Pass
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! General if-conversion pass for AArch64.
//!
//! Converts diamond and triangle CFG patterns into predicated conditional
//! select instructions (CSEL, CSINC, CSNEG), eliminating branches and
//! improving instruction-level parallelism on modern AArch64 cores.
//!
//! # Difference from CmpSelectCombine
//!
//! [`crate::cmp_select::CmpSelectCombine`] handles the narrow case where
//! both diamond arms contain exactly one MOV instruction. This pass is
//! more general:
//!
//! - **Diamond patterns**: Arms may contain up to 2 simple instructions
//!   (arithmetic, logical, moves) plus a branch — not just a single MOV.
//! - **Triangle patterns**: If-then (no else) where one arm has a single
//!   assignment that falls through to the merge block.
//! - **CSINC/CSNEG formation**: Recognizes `ADD dst, src, #1` and
//!   `NEG dst, src` patterns in diamond arms for more compact codegen.
//!
//! # Patterns
//!
//! | Pattern | Transformation |
//! |---------|---------------|
//! | Diamond: MOV + ADD #1 | `CSINC Xd, Xn, Xm, cond` |
//! | Diamond: MOV + NEG    | `CSNEG Xd, Xn, Xm, cond` |
//! | Diamond: MOV + MOV (general) | `CSEL Xd, Xn, Xm, cond` |
//! | Triangle: single assign + fallthrough | `CSEL Xd, Xn, Xd, cond` |
//!
//! # Diamond CFG Shape
//!
//! ```text
//!   header:
//!     ...
//!     CMP Xn, Xm
//!     B.cond true_block
//!   false_block:
//!     <1-2 simple insts>
//!     B join
//!   true_block:
//!     <1-2 simple insts>
//!     B join
//!   join:
//!     ...
//! ```
//!
//! # Triangle CFG Shape
//!
//! ```text
//!   header:
//!     ...
//!     CMP Xn, Xm
//!     B.cond then_block
//!     (fallthrough to join)
//!   then_block:
//!     MOV Xd, Xn (single value-producing inst)
//!     B join
//!   join:
//!     ...
//! ```
//!
//! # Profitability
//!
//! Only converts when:
//! - Each arm has at most 2 non-branch instructions
//! - No memory operations (loads/stores) in either arm
//! - No calls in either arm
//! - The branch condition is still available
//!
//! # Safety Constraints
//!
//! - Arm blocks must have exactly 1 predecessor (the header)
//! - Diamond arms must branch to the same merge block
//! - Triangle then-block must branch to the header's fallthrough successor
//! - No flag-setting instructions in arms (would clobber NZCV used by CSEL)

use llvm2_ir::{AArch64Opcode, BlockId, CondCode, InstId, MachFunction, MachInst, MachOperand};

use crate::effects::opcode_effect;
use crate::pass_manager::MachinePass;

/// AArch64 if-conversion pass.
///
/// Converts diamond and triangle CFG patterns into conditional select
/// instructions. Runs at O2+ after CmpSelectCombine (which handles the
/// simplest cases) and before DCE (which cleans up dead instructions).
pub struct IfConversion;

impl MachinePass for IfConversion {
    fn name(&self) -> &str {
        "if-convert"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        let mut changed = false;

        // Collect diamond transforms first to avoid borrow issues.
        let diamond_xforms = collect_diamond_transforms(func);
        for xform in &diamond_xforms {
            apply_diamond_transform(func, xform);
            changed = true;
        }

        // Collect triangle transforms (must re-scan after diamonds).
        let triangle_xforms = collect_triangle_transforms(func);
        for xform in &triangle_xforms {
            apply_triangle_transform(func, xform);
            changed = true;
        }

        changed
    }
}

// ---------------------------------------------------------------------------
// Diamond if-conversion
// ---------------------------------------------------------------------------

/// A diamond transform: replaces a BCond + two arm blocks with a
/// conditional select instruction in the header.
struct DiamondTransform {
    header: BlockId,
    true_block: BlockId,
    false_block: BlockId,
    join_block: BlockId,
    /// The conditional select instruction(s) to insert before the BCond.
    new_insts: Vec<MachInst>,
    bcond_inst_id: InstId,
}

/// Scan for diamond CFG patterns suitable for if-conversion.
fn collect_diamond_transforms(func: &MachFunction) -> Vec<DiamondTransform> {
    let mut transforms = Vec::new();

    for &header_id in &func.block_order {
        let header = func.block(header_id);
        if header.insts.is_empty() {
            continue;
        }

        // Last instruction must be BCond.
        let last_inst_id = *header.insts.last().unwrap();
        let last_inst = func.inst(last_inst_id);
        if last_inst.opcode != AArch64Opcode::BCond {
            continue;
        }

        // BCond operands: [Imm(cond_code), Block(target)]
        if last_inst.operands.len() < 2 {
            continue;
        }
        let cond_encoding = match last_inst.operands[0].as_imm() {
            Some(v) => v as u8,
            None => continue,
        };
        let true_block = match &last_inst.operands[1] {
            MachOperand::Block(bid) => *bid,
            _ => continue,
        };

        // Header must have exactly 2 successors.
        if header.succs.len() != 2 {
            continue;
        }

        // Determine false block: the successor that is NOT the BCond target.
        let false_block = if header.succs[0] == true_block {
            header.succs[1]
        } else if header.succs[1] == true_block {
            header.succs[0]
        } else {
            continue;
        };

        // Both arm blocks must have exactly 1 predecessor (the header).
        if func.block(true_block).preds.len() != 1
            || func.block(false_block).preds.len() != 1
        {
            continue;
        }

        // Both arm blocks must have exactly 1 successor (the join block).
        let true_blk = func.block(true_block);
        let false_blk = func.block(false_block);
        if true_blk.succs.len() != 1 || false_blk.succs.len() != 1 {
            continue;
        }

        // Both arms must branch to the same join block.
        let join_block = true_blk.succs[0];
        if false_blk.succs[0] != join_block {
            continue;
        }

        // Last instruction of each arm must be unconditional B.
        if true_blk.insts.is_empty() || false_blk.insts.is_empty() {
            continue;
        }
        let true_last = func.inst(*true_blk.insts.last().unwrap());
        let false_last = func.inst(*false_blk.insts.last().unwrap());
        if true_last.opcode != AArch64Opcode::B || false_last.opcode != AArch64Opcode::B {
            continue;
        }

        // Non-branch instructions in each arm (excluding the trailing B).
        let true_body: Vec<InstId> = true_blk.insts[..true_blk.insts.len() - 1].to_vec();
        let false_body: Vec<InstId> = false_blk.insts[..false_blk.insts.len() - 1].to_vec();

        // Profitability check: at most 2 non-branch instructions per arm.
        if true_body.len() > 2 || false_body.len() > 2 {
            continue;
        }

        // Safety check: all non-branch instructions must be safe to speculate.
        if !all_safe_to_speculate(func, &true_body) || !all_safe_to_speculate(func, &false_body) {
            continue;
        }

        let cond = match decode_cond(cond_encoding) {
            Some(c) => c,
            None => continue,
        };

        // Try to form a single conditional select from the last value-producing
        // instruction in each arm. Both arms must write to the same destination.
        if true_body.len() == 1 && false_body.len() == 1 {
            let true_inst = func.inst(true_body[0]);
            let false_inst = func.inst(false_body[0]);

            // Both must have a destination operand and it must be the same.
            if true_inst.operands.is_empty() || false_inst.operands.is_empty() {
                continue;
            }
            if true_inst.operands[0] != false_inst.operands[0] {
                continue;
            }

            // Try CSINC: true arm is MOV, false arm is ADD src, #1.
            if let Some(inst) = try_csinc(true_inst, false_inst, cond) {
                transforms.push(DiamondTransform {
                    header: header_id,
                    true_block,
                    false_block,
                    join_block,
                    new_insts: vec![inst],
                    bcond_inst_id: last_inst_id,
                });
                continue;
            }

            // Try CSNEG: true arm is MOV, false arm is NEG.
            if let Some(inst) = try_csneg(true_inst, false_inst, cond) {
                transforms.push(DiamondTransform {
                    header: header_id,
                    true_block,
                    false_block,
                    join_block,
                    new_insts: vec![inst],
                    bcond_inst_id: last_inst_id,
                });
                continue;
            }

            // Try general CSEL: both arms produce a value into the same dest.
            if let Some(inst) = try_general_csel(true_inst, false_inst, cond) {
                transforms.push(DiamondTransform {
                    header: header_id,
                    true_block,
                    false_block,
                    join_block,
                    new_insts: vec![inst],
                    bcond_inst_id: last_inst_id,
                });
                continue;
            }
        }

        // Multi-instruction diamond: hoist all body instructions and add a CSEL
        // for the final value-producing instruction in each arm.
        if (true_body.len() == 2 && false_body.len() <= 2 || true_body.len() <= 2 && false_body.len() == 2)
            && let Some(insts) = try_multi_inst_diamond(func, &true_body, &false_body, cond) {
                transforms.push(DiamondTransform {
                    header: header_id,
                    true_block,
                    false_block,
                    join_block,
                    new_insts: insts,
                    bcond_inst_id: last_inst_id,
                });
            }
    }

    transforms
}

/// Check if an instruction is safe to speculate (hoist past a branch).
/// Must be pure, not set flags, and not be a branch/call.
fn is_safe_to_speculate(opcode: AArch64Opcode) -> bool {
    // Must have no memory effects.
    if !opcode_effect(opcode).is_pure() {
        return false;
    }
    // Must not set condition flags.
    use AArch64Opcode::*;
    !matches!(
        opcode,
        CmpRR | CmpRI | CMPWrr | CMPXrr | CMPWri | CMPXri | Tst | Fcmp
            | AddsRR | AddsRI | SubsRR | SubsRI
            | B | BCond | Cbz | Cbnz | Tbz | Tbnz | Br | Ret | Bl | Blr | BL | BLR
    )
}

/// Check that all instructions in a list are safe to speculate.
fn all_safe_to_speculate(func: &MachFunction, insts: &[InstId]) -> bool {
    insts.iter().all(|&id| is_safe_to_speculate(func.inst(id).opcode))
}

/// Try to form CSINC: true arm is MOV dst, src_n; false arm is ADD dst, src_m, #1
/// -> CSINC dst, src_n, src_m, cond
///
/// Also handles the swapped case: true is ADD #1, false is MOV.
fn try_csinc(true_inst: &MachInst, false_inst: &MachInst, cond: CondCode) -> Option<MachInst> {
    let dst = true_inst.operands[0].clone();

    // Case 1: true = MOV, false = ADD #1
    if is_mov(true_inst) && is_add_imm1(false_inst) {
        let true_src = mov_source(true_inst)?;
        let false_src = add_base(false_inst)?;
        return Some(MachInst::new(
            AArch64Opcode::Csinc,
            vec![dst, true_src, false_src, MachOperand::Imm(cond.encoding() as i64)],
        ));
    }

    // Case 2: true = ADD #1, false = MOV -> CSINC dst, false_src, true_base, inverted
    if is_add_imm1(true_inst) && is_mov(false_inst) {
        let false_src = mov_source(false_inst)?;
        let true_base = add_base(true_inst)?;
        let inv = cond.invert();
        return Some(MachInst::new(
            AArch64Opcode::Csinc,
            vec![dst, false_src, true_base, MachOperand::Imm(inv.encoding() as i64)],
        ));
    }

    None
}

/// Try to form CSNEG: true arm is MOV dst, src_n; false arm is NEG dst, src_m
/// -> CSNEG dst, src_n, src_m, cond
fn try_csneg(true_inst: &MachInst, false_inst: &MachInst, cond: CondCode) -> Option<MachInst> {
    let dst = true_inst.operands[0].clone();

    // Case 1: true = MOV, false = NEG
    if is_mov(true_inst) && false_inst.opcode == AArch64Opcode::Neg {
        let true_src = mov_source(true_inst)?;
        let neg_src = false_inst.operands.get(1)?.clone();
        return Some(MachInst::new(
            AArch64Opcode::Csneg,
            vec![dst, true_src, neg_src, MachOperand::Imm(cond.encoding() as i64)],
        ));
    }

    // Case 2: true = NEG, false = MOV -> CSNEG dst, false_src, true_neg_src, inverted
    if true_inst.opcode == AArch64Opcode::Neg && is_mov(false_inst) {
        let false_src = mov_source(false_inst)?;
        let neg_src = true_inst.operands.get(1)?.clone();
        let inv = cond.invert();
        return Some(MachInst::new(
            AArch64Opcode::Csneg,
            vec![dst, false_src, neg_src, MachOperand::Imm(inv.encoding() as i64)],
        ));
    }

    None
}

/// Try to form a general CSEL from two single-instruction arms.
/// Both arms must be MOV (MovR or MovI) writing to the same destination.
fn try_general_csel(true_inst: &MachInst, false_inst: &MachInst, cond: CondCode) -> Option<MachInst> {
    if !is_mov(true_inst) || !is_mov(false_inst) {
        return None;
    }
    let dst = true_inst.operands[0].clone();
    let true_src = mov_source(true_inst)?;
    let false_src = mov_source(false_inst)?;

    Some(MachInst::new(
        AArch64Opcode::Csel,
        vec![dst, true_src, false_src, MachOperand::Imm(cond.encoding() as i64)],
    ))
}

/// Try to convert a multi-instruction diamond (2 insts in at least one arm).
/// Hoists the non-final instructions into the header and produces a CSEL
/// for the final value.
///
/// This only works when the final instructions in both arms write to the
/// same destination and are MOVs (the earlier instructions are hoisted
/// unconditionally since they are safe to speculate).
fn try_multi_inst_diamond(
    func: &MachFunction,
    true_body: &[InstId],
    false_body: &[InstId],
    cond: CondCode,
) -> Option<Vec<MachInst>> {
    if true_body.is_empty() || false_body.is_empty() {
        return None;
    }

    let true_last = func.inst(*true_body.last().unwrap());
    let false_last = func.inst(*false_body.last().unwrap());

    // Final instructions must both be MOVs to the same destination.
    if !is_mov(true_last) || !is_mov(false_last) {
        return None;
    }
    if true_last.operands.is_empty() || false_last.operands.is_empty() {
        return None;
    }
    if true_last.operands[0] != false_last.operands[0] {
        return None;
    }

    let mut hoisted = Vec::new();

    // Hoist all non-final instructions from both arms.
    for &inst_id in &true_body[..true_body.len() - 1] {
        hoisted.push(func.inst(inst_id).clone());
    }
    for &inst_id in &false_body[..false_body.len() - 1] {
        hoisted.push(func.inst(inst_id).clone());
    }

    // Add the CSEL for the final value.
    let dst = true_last.operands[0].clone();
    let true_src = mov_source(true_last)?;
    let false_src = mov_source(false_last)?;

    hoisted.push(MachInst::new(
        AArch64Opcode::Csel,
        vec![dst, true_src, false_src, MachOperand::Imm(cond.encoding() as i64)],
    ));

    Some(hoisted)
}

/// Apply a diamond transform to the function.
fn apply_diamond_transform(func: &mut MachFunction, xform: &DiamondTransform) {
    // 1. Insert new instructions in the header just before the BCond.
    let header = func.block(xform.header);
    let bcond_pos = header
        .insts
        .iter()
        .position(|&id| id == xform.bcond_inst_id);

    let mut new_inst_ids = Vec::new();
    for inst in &xform.new_insts {
        let id = func.push_inst(inst.clone());
        new_inst_ids.push(id);
    }

    if let Some(pos) = bcond_pos {
        let header_mut = func.block_mut(xform.header);
        for (i, id) in new_inst_ids.into_iter().enumerate() {
            header_mut.insts.insert(pos + i, id);
        }
    }

    // 2. Replace BCond with B .join.
    let b_join = MachInst::new(
        AArch64Opcode::B,
        vec![MachOperand::Block(xform.join_block)],
    );
    *func.inst_mut(xform.bcond_inst_id) = b_join;

    // 3. Update CFG: header's successors become just [join_block].
    let header = func.block_mut(xform.header);
    header.succs.clear();
    header.succs.push(xform.join_block);

    // 4. Update join block predecessors.
    let join = func.block_mut(xform.join_block);
    join.preds.retain(|&bid| bid != xform.true_block && bid != xform.false_block);
    if !join.preds.contains(&xform.header) {
        join.preds.push(xform.header);
    }

    // 5. Remove arm blocks from block_order.
    func.block_order
        .retain(|&bid| bid != xform.true_block && bid != xform.false_block);
}

// ---------------------------------------------------------------------------
// Triangle if-conversion
// ---------------------------------------------------------------------------

/// A triangle transform: replaces a conditional branch over a single-
/// assignment block with a CSEL in the header.
struct TriangleTransform {
    header: BlockId,
    then_block: BlockId,
    join_block: BlockId,
    new_inst: MachInst,
    bcond_inst_id: InstId,
}

/// Scan for triangle CFG patterns.
///
/// ```text
///   header:
///     ...
///     B.cond then_block
///     (fallthrough to join)
///   then_block:
///     MOV Xd, Xn
///     B join
///   join:
///     ...
/// ```
fn collect_triangle_transforms(func: &MachFunction) -> Vec<TriangleTransform> {
    let mut transforms = Vec::new();

    for &header_id in &func.block_order {
        let header = func.block(header_id);
        if header.insts.is_empty() {
            continue;
        }

        // Last instruction must be BCond.
        let last_inst_id = *header.insts.last().unwrap();
        let last_inst = func.inst(last_inst_id);
        if last_inst.opcode != AArch64Opcode::BCond {
            continue;
        }

        if last_inst.operands.len() < 2 {
            continue;
        }
        let cond_encoding = match last_inst.operands[0].as_imm() {
            Some(v) => v as u8,
            None => continue,
        };
        let then_block = match &last_inst.operands[1] {
            MachOperand::Block(bid) => *bid,
            _ => continue,
        };

        // Header must have exactly 2 successors.
        if header.succs.len() != 2 {
            continue;
        }

        // Determine the fallthrough (join) block.
        let join_block = if header.succs[0] == then_block {
            header.succs[1]
        } else if header.succs[1] == then_block {
            header.succs[0]
        } else {
            continue;
        };

        // Triangle: then_block must jump to join_block.
        let then_blk = func.block(then_block);
        if then_blk.succs.len() != 1 || then_blk.succs[0] != join_block {
            continue;
        }

        // Then block must have exactly 1 predecessor (the header).
        if then_blk.preds.len() != 1 {
            continue;
        }

        // Then block must have exactly 2 instructions: one value-producing + B.
        if then_blk.insts.len() != 2 {
            continue;
        }

        let then_inst_id = then_blk.insts[0];
        let then_br_id = then_blk.insts[1];
        let then_inst = func.inst(then_inst_id);
        let then_br = func.inst(then_br_id);

        if then_br.opcode != AArch64Opcode::B {
            continue;
        }

        // The then instruction must be a MOV (MovR or MovI) for safe conversion.
        if !is_mov(then_inst) {
            continue;
        }

        // Safety check.
        if !is_safe_to_speculate(then_inst.opcode) {
            continue;
        }

        let cond = match decode_cond(cond_encoding) {
            Some(c) => c,
            None => continue,
        };

        // Build CSEL: dst = cond ? then_value : dst
        // When the condition is true, we take the then path (then_value).
        // When false, we fall through (identity: keep dst unchanged).
        if then_inst.operands.len() < 2 {
            continue;
        }
        let dst = then_inst.operands[0].clone();
        let then_src = mov_source(then_inst);
        let then_src = match then_src {
            Some(s) => s,
            None => continue,
        };

        let csel = MachInst::new(
            AArch64Opcode::Csel,
            vec![
                dst.clone(),
                then_src,
                dst,
                MachOperand::Imm(cond.encoding() as i64),
            ],
        );

        transforms.push(TriangleTransform {
            header: header_id,
            then_block,
            join_block,
            new_inst: csel,
            bcond_inst_id: last_inst_id,
        });
    }

    transforms
}

/// Apply a triangle transform.
fn apply_triangle_transform(func: &mut MachFunction, xform: &TriangleTransform) {
    // 1. Insert CSEL before BCond.
    let new_inst_id = func.push_inst(xform.new_inst.clone());
    let header = func.block_mut(xform.header);
    if let Some(pos) = header.insts.iter().position(|&id| id == xform.bcond_inst_id) {
        header.insts.insert(pos, new_inst_id);
    }

    // 2. Replace BCond with B .join.
    let b_join = MachInst::new(
        AArch64Opcode::B,
        vec![MachOperand::Block(xform.join_block)],
    );
    *func.inst_mut(xform.bcond_inst_id) = b_join;

    // 3. Update CFG.
    let header = func.block_mut(xform.header);
    header.succs.clear();
    header.succs.push(xform.join_block);

    let join = func.block_mut(xform.join_block);
    join.preds.retain(|&bid| bid != xform.then_block);
    if !join.preds.contains(&xform.header) {
        join.preds.push(xform.header);
    }

    // 4. Remove then block from block_order.
    func.block_order.retain(|&bid| bid != xform.then_block);
}

// ---------------------------------------------------------------------------
// Instruction helpers
// ---------------------------------------------------------------------------

/// Returns true if the instruction is a MOV (MovR or MovI).
fn is_mov(inst: &MachInst) -> bool {
    matches!(inst.opcode, AArch64Opcode::MovR | AArch64Opcode::MovI)
}

/// Returns the source operand of a MOV instruction.
fn mov_source(inst: &MachInst) -> Option<MachOperand> {
    if !is_mov(inst) || inst.operands.len() < 2 {
        return None;
    }
    Some(inst.operands[1].clone())
}

/// Returns true if the instruction is ADD dst, src, #1.
fn is_add_imm1(inst: &MachInst) -> bool {
    inst.opcode == AArch64Opcode::AddRI
        && inst.operands.len() >= 3
        && inst.operands[2].as_imm() == Some(1)
}

/// Returns the base register of an ADD #1 instruction (operand[1]).
fn add_base(inst: &MachInst) -> Option<MachOperand> {
    if !is_add_imm1(inst) || inst.operands.len() < 2 {
        return None;
    }
    Some(inst.operands[1].clone())
}

/// Decode a condition code encoding (0-15) to a CondCode variant.
fn decode_cond(encoding: u8) -> Option<CondCode> {
    match encoding {
        0b0000 => Some(CondCode::EQ),
        0b0001 => Some(CondCode::NE),
        0b0010 => Some(CondCode::HS),
        0b0011 => Some(CondCode::LO),
        0b0100 => Some(CondCode::MI),
        0b0101 => Some(CondCode::PL),
        0b0110 => Some(CondCode::VS),
        0b0111 => Some(CondCode::VC),
        0b1000 => Some(CondCode::HI),
        0b1001 => Some(CondCode::LS),
        0b1010 => Some(CondCode::GE),
        0b1011 => Some(CondCode::LT),
        0b1100 => Some(CondCode::GT),
        0b1101 => Some(CondCode::LE),
        0b1110 => Some(CondCode::AL),
        0b1111 => Some(CondCode::NV),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass_manager::MachinePass;
    use llvm2_ir::{
        AArch64Opcode, BlockId, CondCode, MachFunction, MachInst, MachOperand, RegClass,
        Signature, VReg,
    };

    fn vreg(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
    }

    fn imm(val: i64) -> MachOperand {
        MachOperand::Imm(val)
    }

    /// Build a diamond CFG:
    /// bb0: CMP + BCond -> bb1
    /// bb2 (false): false_insts + B bb3
    /// bb1 (true): true_insts + B bb3
    /// bb3 (join): RET
    fn make_diamond(
        cmp: MachInst,
        cond: CondCode,
        true_insts: Vec<MachInst>,
        false_insts: Vec<MachInst>,
    ) -> MachFunction {
        let mut func = MachFunction::new(
            "test_if_convert".to_string(),
            Signature::new(vec![], vec![]),
        );

        let bb0 = func.entry;
        let bb1 = func.create_block(); // true
        let bb2 = func.create_block(); // false
        let bb3 = func.create_block(); // join

        // bb0: CMP + BCond
        let cmp_id = func.push_inst(cmp);
        func.append_inst(bb0, cmp_id);
        let bcond = MachInst::new(
            AArch64Opcode::BCond,
            vec![imm(cond.encoding() as i64), MachOperand::Block(bb1)],
        );
        let bcond_id = func.push_inst(bcond);
        func.append_inst(bb0, bcond_id);

        // bb1 (true): true_insts + B bb3
        for inst in true_insts {
            let id = func.push_inst(inst);
            func.append_inst(bb1, id);
        }
        let b1 = func.push_inst(MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(bb3)]));
        func.append_inst(bb1, b1);

        // bb2 (false): false_insts + B bb3
        for inst in false_insts {
            let id = func.push_inst(inst);
            func.append_inst(bb2, id);
        }
        let b2 = func.push_inst(MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(bb3)]));
        func.append_inst(bb2, b2);

        // bb3 (join): RET
        let ret_id = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb3, ret_id);

        func.add_edge(bb0, bb1);
        func.add_edge(bb0, bb2);
        func.add_edge(bb1, bb3);
        func.add_edge(bb2, bb3);

        func
    }

    /// Build a triangle CFG:
    /// bb0: CMP + BCond -> bb1, fallthrough -> bb2
    /// bb1 (then): then_inst + B bb2
    /// bb2 (join): RET
    fn make_triangle(
        cmp: MachInst,
        cond: CondCode,
        then_inst: MachInst,
    ) -> MachFunction {
        let mut func = MachFunction::new(
            "test_triangle".to_string(),
            Signature::new(vec![], vec![]),
        );

        let bb0 = func.entry;
        let bb1 = func.create_block(); // then
        let bb2 = func.create_block(); // join

        // bb0: CMP + BCond
        let cmp_id = func.push_inst(cmp);
        func.append_inst(bb0, cmp_id);
        let bcond = MachInst::new(
            AArch64Opcode::BCond,
            vec![imm(cond.encoding() as i64), MachOperand::Block(bb1)],
        );
        let bcond_id = func.push_inst(bcond);
        func.append_inst(bb0, bcond_id);

        // bb1 (then): then_inst + B bb2
        let ti = func.push_inst(then_inst);
        func.append_inst(bb1, ti);
        let b1 = func.push_inst(MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(bb2)]));
        func.append_inst(bb1, b1);

        // bb2 (join): RET
        let ret_id = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret_id);

        func.add_edge(bb0, bb1);
        func.add_edge(bb0, bb2);
        func.add_edge(bb1, bb2);

        func
    }

    // ---- Diamond: CSEL formation ----

    #[test]
    fn test_diamond_csel_movr() {
        let cmp = MachInst::new(AArch64Opcode::CmpRR, vec![vreg(0), vreg(1)]);
        let true_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(3)]);
        let false_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(4)]);

        let mut func = make_diamond(cmp, CondCode::EQ, vec![true_mov], vec![false_mov]);
        let mut pass = IfConversion;
        assert!(pass.run(&mut func));

        let header = func.block(BlockId(0));
        assert_eq!(header.insts.len(), 3); // CMP, CSEL, B

        let csel = func.inst(header.insts[1]);
        assert_eq!(csel.opcode, AArch64Opcode::Csel);
        assert_eq!(csel.operands[0], vreg(2));
        assert_eq!(csel.operands[1], vreg(3));
        assert_eq!(csel.operands[2], vreg(4));
        assert_eq!(csel.operands[3], imm(CondCode::EQ.encoding() as i64));

        // Arm blocks removed.
        assert!(!func.block_order.contains(&BlockId(1)));
        assert!(!func.block_order.contains(&BlockId(2)));
    }

    #[test]
    fn test_diamond_csel_movi() {
        let cmp = MachInst::new(AArch64Opcode::CmpRR, vec![vreg(0), vreg(1)]);
        let true_mov = MachInst::new(AArch64Opcode::MovI, vec![vreg(2), imm(42)]);
        let false_mov = MachInst::new(AArch64Opcode::MovI, vec![vreg(2), imm(99)]);

        let mut func = make_diamond(cmp, CondCode::GE, vec![true_mov], vec![false_mov]);
        let mut pass = IfConversion;
        assert!(pass.run(&mut func));

        let header = func.block(BlockId(0));
        let csel = func.inst(header.insts[1]);
        assert_eq!(csel.opcode, AArch64Opcode::Csel);
        assert_eq!(csel.operands[1], imm(42));
        assert_eq!(csel.operands[2], imm(99));
    }

    // ---- Diamond: CSINC formation ----

    #[test]
    fn test_diamond_csinc() {
        // true: MOV v2, v3; false: ADD v2, v4, #1
        // -> CSINC v2, v3, v4, EQ
        let cmp = MachInst::new(AArch64Opcode::CmpRR, vec![vreg(0), vreg(1)]);
        let true_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(3)]);
        let false_add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(4), imm(1)]);

        let mut func = make_diamond(cmp, CondCode::EQ, vec![true_mov], vec![false_add]);
        let mut pass = IfConversion;
        assert!(pass.run(&mut func));

        let header = func.block(BlockId(0));
        let csinc = func.inst(header.insts[1]);
        assert_eq!(csinc.opcode, AArch64Opcode::Csinc);
        assert_eq!(csinc.operands[0], vreg(2)); // dst
        assert_eq!(csinc.operands[1], vreg(3)); // true_src
        assert_eq!(csinc.operands[2], vreg(4)); // false_base (will be incremented)
        assert_eq!(csinc.operands[3], imm(CondCode::EQ.encoding() as i64));
    }

    #[test]
    fn test_diamond_csinc_swapped() {
        // true: ADD v2, v3, #1; false: MOV v2, v4
        // -> CSINC v2, v4, v3, NE (inverted from EQ)
        let cmp = MachInst::new(AArch64Opcode::CmpRR, vec![vreg(0), vreg(1)]);
        let true_add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(3), imm(1)]);
        let false_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(4)]);

        let mut func = make_diamond(cmp, CondCode::EQ, vec![true_add], vec![false_mov]);
        let mut pass = IfConversion;
        assert!(pass.run(&mut func));

        let header = func.block(BlockId(0));
        let csinc = func.inst(header.insts[1]);
        assert_eq!(csinc.opcode, AArch64Opcode::Csinc);
        assert_eq!(csinc.operands[0], vreg(2));
        assert_eq!(csinc.operands[1], vreg(4)); // MOV source (now true for inverted cond)
        assert_eq!(csinc.operands[2], vreg(3)); // ADD base
        assert_eq!(csinc.operands[3], imm(CondCode::NE.encoding() as i64));
    }

    // ---- Diamond: CSNEG formation ----

    #[test]
    fn test_diamond_csneg() {
        // true: MOV v2, v3; false: NEG v2, v4
        // -> CSNEG v2, v3, v4, LT
        let cmp = MachInst::new(AArch64Opcode::CmpRR, vec![vreg(0), vreg(1)]);
        let true_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(3)]);
        let false_neg = MachInst::new(AArch64Opcode::Neg, vec![vreg(2), vreg(4)]);

        let mut func = make_diamond(cmp, CondCode::LT, vec![true_mov], vec![false_neg]);
        let mut pass = IfConversion;
        assert!(pass.run(&mut func));

        let header = func.block(BlockId(0));
        let csneg = func.inst(header.insts[1]);
        assert_eq!(csneg.opcode, AArch64Opcode::Csneg);
        assert_eq!(csneg.operands[0], vreg(2));
        assert_eq!(csneg.operands[1], vreg(3));
        assert_eq!(csneg.operands[2], vreg(4));
        assert_eq!(csneg.operands[3], imm(CondCode::LT.encoding() as i64));
    }

    #[test]
    fn test_diamond_csneg_swapped() {
        // true: NEG v2, v3; false: MOV v2, v4
        // -> CSNEG v2, v4, v3, GE (inverted from LT)
        let cmp = MachInst::new(AArch64Opcode::CmpRR, vec![vreg(0), vreg(1)]);
        let true_neg = MachInst::new(AArch64Opcode::Neg, vec![vreg(2), vreg(3)]);
        let false_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(4)]);

        let mut func = make_diamond(cmp, CondCode::LT, vec![true_neg], vec![false_mov]);
        let mut pass = IfConversion;
        assert!(pass.run(&mut func));

        let header = func.block(BlockId(0));
        let csneg = func.inst(header.insts[1]);
        assert_eq!(csneg.opcode, AArch64Opcode::Csneg);
        assert_eq!(csneg.operands[0], vreg(2));
        assert_eq!(csneg.operands[1], vreg(4)); // MOV source
        assert_eq!(csneg.operands[2], vreg(3)); // NEG source
        assert_eq!(csneg.operands[3], imm(CondCode::GE.encoding() as i64));
    }

    // ---- Diamond: multi-instruction ----

    #[test]
    fn test_diamond_multi_inst() {
        // true: ADD v5, v3, #10; MOV v2, v5; false: MOV v2, v4
        // -> hoist ADD v5, v3, #10; CSEL v2, v5, v4, EQ
        let cmp = MachInst::new(AArch64Opcode::CmpRR, vec![vreg(0), vreg(1)]);
        let true_add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(5), vreg(3), imm(10)]);
        let true_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(5)]);
        let false_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(4)]);

        let mut func = make_diamond(
            cmp,
            CondCode::EQ,
            vec![true_add, true_mov],
            vec![false_mov],
        );
        let mut pass = IfConversion;
        assert!(pass.run(&mut func));

        let header = func.block(BlockId(0));
        // CMP, hoisted ADD, CSEL, B = 4 instructions
        assert_eq!(header.insts.len(), 4);

        let add = func.inst(header.insts[1]);
        assert_eq!(add.opcode, AArch64Opcode::AddRI);

        let csel = func.inst(header.insts[2]);
        assert_eq!(csel.opcode, AArch64Opcode::Csel);
        assert_eq!(csel.operands[0], vreg(2));
        assert_eq!(csel.operands[1], vreg(5));
        assert_eq!(csel.operands[2], vreg(4));
    }

    // ---- Triangle: CSEL formation ----

    #[test]
    fn test_triangle_csel_movr() {
        // bb0: CMP; B.EQ bb1; (fallthrough bb2)
        // bb1: MOV v2, v3; B bb2
        // bb2: RET
        // -> CSEL v2, v3, v2, EQ
        let cmp = MachInst::new(AArch64Opcode::CmpRR, vec![vreg(0), vreg(1)]);
        let then_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(3)]);

        let mut func = make_triangle(cmp, CondCode::EQ, then_mov);
        let mut pass = IfConversion;
        assert!(pass.run(&mut func));

        let header = func.block(BlockId(0));
        assert_eq!(header.insts.len(), 3); // CMP, CSEL, B

        let csel = func.inst(header.insts[1]);
        assert_eq!(csel.opcode, AArch64Opcode::Csel);
        assert_eq!(csel.operands[0], vreg(2)); // dst
        assert_eq!(csel.operands[1], vreg(3)); // then_src
        assert_eq!(csel.operands[2], vreg(2)); // identity (keep dst)
        assert_eq!(csel.operands[3], imm(CondCode::EQ.encoding() as i64));

        // Then block removed.
        assert!(!func.block_order.contains(&BlockId(1)));
    }

    #[test]
    fn test_triangle_csel_movi() {
        let cmp = MachInst::new(AArch64Opcode::CmpRI, vec![vreg(0), imm(5)]);
        let then_mov = MachInst::new(AArch64Opcode::MovI, vec![vreg(2), imm(100)]);

        let mut func = make_triangle(cmp, CondCode::GT, then_mov);
        let mut pass = IfConversion;
        assert!(pass.run(&mut func));

        let header = func.block(BlockId(0));
        let csel = func.inst(header.insts[1]);
        assert_eq!(csel.opcode, AArch64Opcode::Csel);
        assert_eq!(csel.operands[1], imm(100));
        assert_eq!(csel.operands[3], imm(CondCode::GT.encoding() as i64));
    }

    // ---- Negative tests ----

    #[test]
    fn test_no_convert_different_destinations() {
        let cmp = MachInst::new(AArch64Opcode::CmpRR, vec![vreg(0), vreg(1)]);
        let true_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(3)]);
        let false_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(5), vreg(4)]);

        let mut func = make_diamond(cmp, CondCode::EQ, vec![true_mov], vec![false_mov]);
        let mut pass = IfConversion;
        assert!(!pass.run(&mut func));
        assert_eq!(func.block_order.len(), 4);
    }

    #[test]
    fn test_no_convert_memory_in_arm() {
        // False arm has a load -> not safe to speculate.
        let cmp = MachInst::new(AArch64Opcode::CmpRR, vec![vreg(0), vreg(1)]);
        let true_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(3)]);
        let false_ldr = MachInst::new(
            AArch64Opcode::LdrRI,
            vec![vreg(2), vreg(4), imm(0)],
        );

        let mut func = make_diamond(cmp, CondCode::EQ, vec![true_mov], vec![false_ldr]);
        let mut pass = IfConversion;
        assert!(!pass.run(&mut func));
    }

    #[test]
    fn test_no_convert_call_in_arm() {
        let cmp = MachInst::new(AArch64Opcode::CmpRR, vec![vreg(0), vreg(1)]);
        let true_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(3)]);
        let false_call = MachInst::new(AArch64Opcode::Bl, vec![]);

        let mut func = make_diamond(cmp, CondCode::EQ, vec![true_mov], vec![false_call]);
        let mut pass = IfConversion;
        assert!(!pass.run(&mut func));
    }

    #[test]
    fn test_no_convert_too_many_insts() {
        // True arm has 3 instructions -> exceeds limit.
        let cmp = MachInst::new(AArch64Opcode::CmpRR, vec![vreg(0), vreg(1)]);
        let t1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(5), vreg(3), imm(1)]);
        let t2 = MachInst::new(AArch64Opcode::SubRI, vec![vreg(6), vreg(5), imm(2)]);
        let t3 = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(6)]);
        let false_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(4)]);

        let mut func = make_diamond(cmp, CondCode::EQ, vec![t1, t2, t3], vec![false_mov]);
        let mut pass = IfConversion;
        assert!(!pass.run(&mut func));
        assert_eq!(func.block_order.len(), 4);
    }

    #[test]
    fn test_no_convert_flag_setting_in_arm() {
        // True arm has ADDS (sets flags) -> not safe.
        let cmp = MachInst::new(AArch64Opcode::CmpRR, vec![vreg(0), vreg(1)]);
        let true_adds = MachInst::new(AArch64Opcode::AddsRR, vec![vreg(2), vreg(3), vreg(4)]);
        let false_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(5)]);

        let mut func = make_diamond(cmp, CondCode::EQ, vec![true_adds], vec![false_mov]);
        let mut pass = IfConversion;
        assert!(!pass.run(&mut func));
    }

    // ---- Idempotency ----

    #[test]
    fn test_idempotent() {
        let cmp = MachInst::new(AArch64Opcode::CmpRR, vec![vreg(0), vreg(1)]);
        let true_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(3)]);
        let false_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(4)]);

        let mut func = make_diamond(cmp, CondCode::EQ, vec![true_mov], vec![false_mov]);
        let mut pass = IfConversion;
        assert!(pass.run(&mut func));
        assert!(!pass.run(&mut func));
    }

    // ---- Edge cases ----

    #[test]
    fn test_no_change_empty_func() {
        let mut func = MachFunction::new(
            "empty".to_string(),
            Signature::new(vec![], vec![]),
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(func.entry, ret_id);

        let mut pass = IfConversion;
        assert!(!pass.run(&mut func));
    }

    #[test]
    fn test_cfgcleanup_join_preds() {
        // After diamond conversion, join block should have correct preds.
        let cmp = MachInst::new(AArch64Opcode::CmpRR, vec![vreg(0), vreg(1)]);
        let true_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(3)]);
        let false_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(4)]);

        let mut func = make_diamond(cmp, CondCode::NE, vec![true_mov], vec![false_mov]);
        let mut pass = IfConversion;
        pass.run(&mut func);

        let join = func.block(BlockId(3));
        assert_eq!(join.preds.len(), 1);
        assert!(join.preds.contains(&BlockId(0)));
    }

    #[test]
    fn test_triangle_cfgcleanup() {
        let cmp = MachInst::new(AArch64Opcode::CmpRR, vec![vreg(0), vreg(1)]);
        let then_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(3)]);

        let mut func = make_triangle(cmp, CondCode::EQ, then_mov);
        let mut pass = IfConversion;
        pass.run(&mut func);

        // Join block (bb2) should only have header (bb0) as pred.
        let join = func.block(BlockId(2));
        assert_eq!(join.preds.len(), 1);
        assert!(join.preds.contains(&BlockId(0)));

        // Header should only have join as successor.
        let header = func.block(BlockId(0));
        assert_eq!(header.succs.len(), 1);
        assert_eq!(header.succs[0], BlockId(2));
    }

    // ---- Multiple conditions ----

    #[test]
    fn test_diamond_all_cond_codes() {
        // Verify the pass works with various condition codes.
        for &cc in &[CondCode::EQ, CondCode::NE, CondCode::LT, CondCode::GT, CondCode::LE, CondCode::GE] {
            let cmp = MachInst::new(AArch64Opcode::CmpRR, vec![vreg(0), vreg(1)]);
            let true_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(3)]);
            let false_mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(2), vreg(4)]);

            let mut func = make_diamond(cmp, cc, vec![true_mov], vec![false_mov]);
            let mut pass = IfConversion;
            assert!(pass.run(&mut func), "should convert for {:?}", cc);

            let header = func.block(BlockId(0));
            let csel = func.inst(header.insts[1]);
            assert_eq!(csel.opcode, AArch64Opcode::Csel);
            assert_eq!(csel.operands[3], imm(cc.encoding() as i64));
        }
    }
}
