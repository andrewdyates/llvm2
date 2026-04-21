// llvm2-opt - Constant Folding
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Constant folding pass for machine-level IR.
//!
//! Evaluates instructions whose operands are all known constants at
//! compile time and replaces them with immediate moves.
//!
//! # Supported Operations
//!
//! | Opcode  | Folding |
//! |---------|---------|
//! | AddRI   | dst = src + imm (when src is known constant) |
//! | SubRI   | dst = src - imm (when src is known constant) |
//! | AddRR   | dst = lhs + rhs (when both are known constants) |
//! | SubRR   | dst = lhs - rhs (when both are known constants) |
//! | MulRR   | dst = lhs * rhs (when both are known constants) |
//! | AndRR   | dst = lhs & rhs (when both are known constants) |
//! | OrrRR   | dst = lhs | rhs (when both are known constants) |
//! | EorRR   | dst = lhs ^ rhs (when both are known constants) |
//! | LslRI   | dst = src << imm (when src is known constant) |
//! | LsrRI   | dst = src >> imm (when src is known constant, logical) |
//! | AsrRI   | dst = src >> imm (when src is known constant, arithmetic) |
//! | Neg     | dst = -src (when src is known constant) |
//!
//! # Implementation
//!
//! Single forward walk in program order over `func.block_order` and each
//! block's `insts`. For each instruction we:
//!
//! 1. **Track constant materialization** — simulate MOVZ/MOVN/MOVK bit-chunk
//!    updates so the tracker reflects the full 64-bit value of constants
//!    built by the standard AArch64 MOVZ+MOVK chain.
//! 2. **Try to fold** — if the instruction's source operands are all known
//!    constants, compute the result and replace the instruction with
//!    `MovI dst, #result`, updating the tracker with the new value.
//! 3. **Invalidate on unfolded redefinition** — any other instruction that
//!    produces a value into a vreg that was previously tracked must drop
//!    the stale entry, since later MOVK-style writes would layer on top.
//!
//! Division is NOT folded to avoid division-by-zero at compile time.
//!
//! # Why a single pass with explicit MOVK handling
//!
//! ISel materializes 64-bit constants like `0x165667919E3779F9` as:
//!
//! ```text
//! v100 = MOVZ     #0x79F9
//! v100 = MOVK v100, #0x9E37, lsl #16
//! v100 = MOVK v100, #0x6791, lsl #32
//! v100 = MOVK v100, #0x1656, lsl #48
//! ```
//!
//! MOVK reads+writes the same vreg (the standard AArch64 idiom; not
//! strict SSA). If we ignore MOVK, the tracker only sees the first MOVZ
//! and records the low 16 bits. Any subsequent fold that uses v100 will
//! silently use the wrong value (see issue #366).

use std::collections::HashMap;

use llvm2_ir::{AArch64Opcode, InstId, MachFunction, MachInst, MachOperand, OpcodeCategory, VReg};

use crate::pass_manager::MachinePass;

/// Constant folding pass.
pub struct ConstantFolding;

impl MachinePass for ConstantFolding {
    fn name(&self) -> &str {
        "const-fold"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        let mut changed = false;

        // Map from VReg ID to known 64-bit constant value.
        //
        // We store the full 64-bit constant so MOVZ+MOVK chains (which
        // build the value in-place across multiple instructions) are
        // tracked correctly.
        let mut constants: HashMap<u32, i64> = HashMap::new();

        // Single forward walk in program order: track MOVZ/MOVN/MOVK
        // materialization, fold foldable instructions, and invalidate
        // stale tracker entries when a vreg is redefined by something
        // we can't simulate.
        //
        // We rebuild each block's instruction list as we go so that we
        // can splice MOVK chains after a folded wide constant without
        // disturbing the walk.
        for block_id in func.block_order.clone() {
            let original_insts = func.block(block_id).insts.clone();
            let mut new_insts: Vec<InstId> = Vec::with_capacity(original_insts.len());

            for inst_id in original_insts {
                let inst = func.inst(inst_id);

                // 1. Fold if possible. A folded instruction becomes a
                //    materialization of the computed constant in the
                //    destination vreg, and no later MOVK will target
                //    the dst (ISel only emits MOVK chains immediately
                //    after the seeding MOVZ on a freshly-allocated
                //    vreg).
                if let Some((dst_vreg, result)) = try_fold(inst, &constants) {
                    let extra = rewrite_as_constant_materialization(
                        func, inst_id, dst_vreg, result,
                    );
                    new_insts.push(inst_id);
                    for extra_id in extra {
                        new_insts.push(extra_id);
                    }
                    constants.insert(dst_vreg.id, result);
                    changed = true;
                    continue;
                }

                // 2. Simulate MOVZ/MOVN/MOVK writes so subsequent
                //    instructions see the correct composed value.
                if update_const_tracker(inst, &mut constants) {
                    new_insts.push(inst_id);
                    continue;
                }

                // 3. Invalidate: if this instruction defines a vreg and
                //    we didn't fold or track it, the stale constant
                //    entry (if any) is no longer valid.
                if inst.produces_value()
                    && let Some(MachOperand::VReg(dst)) = inst.operands.first()
                {
                    constants.remove(&dst.id);
                }

                new_insts.push(inst_id);
            }

            func.block_mut(block_id).insts = new_insts;
        }

        changed
    }
}

/// Rewrite the instruction at `inst_id` as a constant materialization of
/// `result` into `dst_vreg`.
///
/// For values that fit in a single 16-bit MOVZ field (`0..=0xFFFF`), we
/// keep the slot as a single `MovI` (the encoder emits this as a plain
/// MOVZ).
///
/// For wider values we rewrite the slot as a MOVZ of the low 16 bits and
/// *append* MOVK instructions to a freshly allocated fan of InstIds. The
/// caller must splice those InstIds into the block's instruction list
/// immediately after `inst_id` to preserve program order.
fn rewrite_as_constant_materialization(
    func: &mut MachFunction,
    inst_id: InstId,
    dst_vreg: VReg,
    result: i64,
) -> Vec<InstId> {
    let result_u = result as u64;

    // Preserve source_loc from the instruction being rewritten (issue #376):
    // constant folding replaces a computation with a materialization, but
    // the source-level statement is unchanged — lldb must still point at
    // the original tMIR source line.
    let src_loc = func.inst(inst_id).source_loc;

    // Fast path: fits in a single 16-bit MOVZ field.
    if result >= 0 && result_u <= 0xFFFF {
        let mut new_inst = MachInst::new(
            AArch64Opcode::MovI,
            vec![MachOperand::VReg(dst_vreg), MachOperand::Imm(result)],
        );
        new_inst.source_loc = src_loc;
        *func.inst_mut(inst_id) = new_inst;
        return Vec::new();
    }

    // Wide: rewrite slot as MOVZ of low 16 bits, then append MOVK for
    // each non-zero remaining 16-bit chunk.
    let low16 = result_u & 0xFFFF;
    let mut new_movz = MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::VReg(dst_vreg), MachOperand::Imm(low16 as i64)],
    );
    new_movz.source_loc = src_loc;
    *func.inst_mut(inst_id) = new_movz;

    let mut extra = Vec::new();
    for shift in [16u64, 32u64, 48u64] {
        let chunk = (result_u >> shift) & 0xFFFF;
        if chunk != 0 {
            let mut movk = MachInst::new(
                AArch64Opcode::Movk,
                vec![
                    MachOperand::VReg(dst_vreg),
                    MachOperand::Imm(chunk as i64),
                    MachOperand::Imm(shift as i64),
                ],
            );
            movk.source_loc = src_loc;
            extra.push(func.push_inst(movk));
        }
    }
    extra
}

/// Simulate MOVZ/MOVN/MOVK writes against the constant tracker.
///
/// Returns `true` if the instruction was a recognized move-wide variant
/// (whether or not the value could be fully determined). Returning
/// `true` signals the caller that this instruction's effect on the
/// tracker has been handled — no further invalidation is needed.
///
/// # Operand conventions (from ISel; see `llvm2-lower/src/isel.rs`)
///
/// * `MovI` / `Movz` — `[VReg(dst), Imm(value)]` or
///   `[VReg(dst), Imm(value), Imm(shift)]`. MovI carries the full
///   already-computed constant; Movz zero-extends a shifted 16-bit
///   immediate.
/// * `Movn`  — `[VReg(dst), Imm(value)]` or
///   `[VReg(dst), Imm(value), Imm(shift)]`. Bitwise NOT of the shifted
///   16-bit field, zero-extended.
/// * `Movk`  — `[VReg(dst), Imm(value), Imm(shift)]`. Writes the 16-bit
///   field at `shift`, preserving the other bits of `dst`. If `dst` is
///   not tracked we cannot reconstruct the full value and drop any
///   stale entry.
fn update_const_tracker(inst: &MachInst, constants: &mut HashMap<u32, i64>) -> bool {
    let Some(MachOperand::VReg(dst)) = inst.operands.first() else {
        return false;
    };
    match inst.opcode {
        AArch64Opcode::MovI => {
            // MovI carries the fully-resolved constant; this is what the
            // folder itself emits. Honor it verbatim.
            if let Some(MachOperand::Imm(v)) = inst.operands.get(1) {
                constants.insert(dst.id, *v);
                return true;
            }
            // Malformed MovI — play safe.
            constants.remove(&dst.id);
            true
        }
        AArch64Opcode::Movz | AArch64Opcode::MOVZWi | AArch64Opcode::MOVZXi => {
            if let Some(MachOperand::Imm(v)) = inst.operands.get(1) {
                let shift = match inst.operands.get(2) {
                    Some(MachOperand::Imm(s)) => (*s as u32) & 0x3F,
                    _ => 0,
                };
                let stored = ((*v as u64) & 0xFFFF) << shift;
                constants.insert(dst.id, stored as i64);
                return true;
            }
            constants.remove(&dst.id);
            true
        }
        AArch64Opcode::Movn => {
            if let Some(MachOperand::Imm(v)) = inst.operands.get(1) {
                let shift = match inst.operands.get(2) {
                    Some(MachOperand::Imm(s)) => (*s as u32) & 0x3F,
                    _ => 0,
                };
                let shifted = ((*v as u64) & 0xFFFF) << shift;
                let stored = !shifted;
                constants.insert(dst.id, stored as i64);
                return true;
            }
            constants.remove(&dst.id);
            true
        }
        AArch64Opcode::Movk => {
            // MOVK: overwrite the 16-bit chunk at `shift`, preserving
            // the rest of the register's current contents.
            let shift = match inst.operands.get(2) {
                Some(MachOperand::Imm(s)) => (*s as u32) & 0x3F,
                _ => 0,
            };
            if let Some(MachOperand::Imm(v)) = inst.operands.get(1) {
                if let Some(old) = constants.get(&dst.id).copied() {
                    let mask: u64 = (0xFFFFu64) << shift;
                    let ins = ((*v as u64) & 0xFFFF) << shift;
                    let new_val = ((old as u64) & !mask) | ins;
                    constants.insert(dst.id, new_val as i64);
                } else {
                    // Cannot reconstruct full value; leave unknown.
                    constants.remove(&dst.id);
                }
            } else {
                constants.remove(&dst.id);
            }
            true
        }
        _ => false,
    }
}

/// Try to constant-fold an instruction.
///
/// Returns `Some((dst_vreg, folded_value))` if the instruction can be folded,
/// `None` otherwise.
///
/// Dispatch uses [`OpcodeCategory`] for target-independent pattern matching:
/// the category determines *which* folding rule applies (add, sub, shift, etc.),
/// while the concrete [`AArch64Opcode`] is still available for target-specific
/// details (e.g., shift amount masking). This means constant folding will work
/// automatically for any new target once its opcodes implement `categorize()`.
fn try_fold(
    inst: &MachInst,
    constants: &HashMap<u32, i64>,
) -> Option<(VReg, i64)> {
    let opcode = inst.opcode;
    let category = opcode.categorize();
    let ops = &inst.operands;

    match category {
        // Binary register-register: dst = op(lhs, rhs)
        // Operands: [dst, lhs, rhs]
        OpcodeCategory::AddRR
        | OpcodeCategory::SubRR
        | OpcodeCategory::MulRR
        | OpcodeCategory::AndRR
        | OpcodeCategory::OrRR
        | OpcodeCategory::XorRR => {
            if ops.len() < 3 {
                return None;
            }
            let dst = ops[0].as_vreg()?;
            let lhs = lookup_const(&ops[1], constants)?;
            let rhs = lookup_const(&ops[2], constants)?;

            let result = match category {
                OpcodeCategory::AddRR => lhs.wrapping_add(rhs),
                OpcodeCategory::SubRR => lhs.wrapping_sub(rhs),
                OpcodeCategory::MulRR => lhs.wrapping_mul(rhs),
                OpcodeCategory::AndRR => lhs & rhs,
                OpcodeCategory::OrRR => lhs | rhs,
                OpcodeCategory::XorRR => lhs ^ rhs,
                _ => unreachable!("inner match constrained by outer arm"),
            };
            Some((dst, result))
        }

        // Binary register-immediate: dst = op(src, imm)
        // Operands: [dst, src, imm]
        OpcodeCategory::AddRI | OpcodeCategory::SubRI => {
            if ops.len() < 3 {
                return None;
            }
            let dst = ops[0].as_vreg()?;
            let src = lookup_const(&ops[1], constants)?;
            let imm = ops[2].as_imm()?;

            let result = match category {
                OpcodeCategory::AddRI => src.wrapping_add(imm),
                OpcodeCategory::SubRI => src.wrapping_sub(imm),
                _ => unreachable!("inner match constrained by outer arm"),
            };
            Some((dst, result))
        }

        // Shift register-immediate: dst = src shift imm
        // Operands: [dst, src, imm]
        OpcodeCategory::ShlRI | OpcodeCategory::ShrRI | OpcodeCategory::SarRI => {
            if ops.len() < 3 {
                return None;
            }
            let dst = ops[0].as_vreg()?;
            let src = lookup_const(&ops[1], constants)?;
            let shift = ops[2].as_imm()?;

            // Validate shift amount (0..63 for 64-bit).
            if !(0..=63).contains(&shift) {
                return None;
            }
            let shift = shift as u32;

            let result = match category {
                OpcodeCategory::ShlRI => src.wrapping_shl(shift),
                OpcodeCategory::ShrRI => ((src as u64).wrapping_shr(shift)) as i64,
                OpcodeCategory::SarRI => src.wrapping_shr(shift),
                _ => unreachable!("inner match constrained by outer arm"),
            };
            Some((dst, result))
        }

        // Shift register-register: dst = src shift amount
        // Operands: [dst, src, amount]
        OpcodeCategory::ShlRR | OpcodeCategory::ShrRR | OpcodeCategory::SarRR => {
            if ops.len() < 3 {
                return None;
            }
            let dst = ops[0].as_vreg()?;
            let src = lookup_const(&ops[1], constants)?;
            let amount = lookup_const(&ops[2], constants)?;

            // Mask shift amount to 0..63 (same behavior on AArch64 and x86-64).
            let shift = (amount & 63) as u32;

            let result = match category {
                OpcodeCategory::ShlRR => src.wrapping_shl(shift),
                OpcodeCategory::ShrRR => ((src as u64).wrapping_shr(shift)) as i64,
                OpcodeCategory::SarRR => src.wrapping_shr(shift),
                _ => unreachable!("inner match constrained by outer arm"),
            };
            Some((dst, result))
        }

        // Unary negate: dst = -src
        // Operands: [dst, src]
        OpcodeCategory::Neg => {
            if ops.len() < 2 {
                return None;
            }
            let dst = ops[0].as_vreg()?;
            let src = lookup_const(&ops[1], constants)?;
            Some((dst, src.wrapping_neg()))
        }

        // Other categories (Load, Store, Call, Mov, Cmp, etc.) are not foldable
        // or their folding requires target-specific logic.
        _ => None,
    }
}

/// Look up the constant value for an operand.
///
/// Returns `Some(value)` if the operand is an immediate or a VReg with
/// a known constant value, `None` otherwise.
fn lookup_const(operand: &MachOperand, constants: &HashMap<u32, i64>) -> Option<i64> {
    match operand {
        MachOperand::Imm(v) => Some(*v),
        MachOperand::VReg(vreg) => constants.get(&vreg.id).copied(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::{MachFunction, MachInst, MachOperand, RegClass, Signature, VReg};
    use crate::pass_manager::MachinePass;

    fn vreg(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
    }

    fn imm(val: i64) -> MachOperand {
        MachOperand::Imm(val)
    }

    fn make_func_with_insts(insts: Vec<MachInst>) -> MachFunction {
        let mut func = MachFunction::new(
            "test_cf".to_string(),
            Signature::new(vec![], vec![]),
        );
        let block = func.entry;
        for inst in insts {
            let id = func.push_inst(inst);
            func.append_inst(block, id);
        }
        func
    }

    /// Reconstruct the runtime value that `vreg_id` would hold after the
    /// entry block executes, by replaying MovI/MOVZ/MOVN/MOVK semantics.
    ///
    /// After `ConstantFolding` rewrites a folded instruction, the slot
    /// may be a plain `MovI` (for values that fit in 16 bits) OR a
    /// `Movz` followed by one or more `Movk` instructions (for wider
    /// values). Tests should compare against the reconstructed value
    /// rather than inspecting a specific slot's opcode.
    fn materialized_value(func: &MachFunction, vreg_id: u32) -> Option<i64> {
        let mut constants: HashMap<u32, i64> = HashMap::new();
        for block_id in &func.block_order {
            let block = func.block(*block_id);
            for inst_id in &block.insts {
                let inst = func.inst(*inst_id);
                update_const_tracker(inst, &mut constants);
            }
        }
        constants.get(&vreg_id).copied()
    }

    #[test]
    fn test_fold_add_ri() {
        // v0 = movi #10
        // v1 = add v0, #20  → v1 = movi #30
        let movi = MachInst::new(AArch64Opcode::MovI, vec![vreg(0), imm(10)]);
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(20)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![movi, add, ret]);

        let mut cf = ConstantFolding;
        assert!(cf.run(&mut func));

        // add should be replaced with MovI
        let folded = func.inst(llvm2_ir::InstId(1));
        assert_eq!(folded.opcode, AArch64Opcode::MovI);
        assert_eq!(folded.operands[1], imm(30));
    }

    #[test]
    fn test_fold_sub_ri() {
        let movi = MachInst::new(AArch64Opcode::MovI, vec![vreg(0), imm(50)]);
        let sub = MachInst::new(AArch64Opcode::SubRI, vec![vreg(1), vreg(0), imm(20)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![movi, sub, ret]);

        let mut cf = ConstantFolding;
        assert!(cf.run(&mut func));

        let folded = func.inst(llvm2_ir::InstId(1));
        assert_eq!(folded.opcode, AArch64Opcode::MovI);
        assert_eq!(folded.operands[1], imm(30));
    }

    #[test]
    fn test_fold_add_rr() {
        // v0 = movi #10
        // v1 = movi #20
        // v2 = add v0, v1  → v2 = movi #30
        let m0 = MachInst::new(AArch64Opcode::MovI, vec![vreg(0), imm(10)]);
        let m1 = MachInst::new(AArch64Opcode::MovI, vec![vreg(1), imm(20)]);
        let add = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![m0, m1, add, ret]);

        let mut cf = ConstantFolding;
        assert!(cf.run(&mut func));

        let folded = func.inst(llvm2_ir::InstId(2));
        assert_eq!(folded.opcode, AArch64Opcode::MovI);
        assert_eq!(folded.operands[1], imm(30));
    }

    #[test]
    fn test_fold_mul_rr() {
        let m0 = MachInst::new(AArch64Opcode::MovI, vec![vreg(0), imm(6)]);
        let m1 = MachInst::new(AArch64Opcode::MovI, vec![vreg(1), imm(7)]);
        let mul = MachInst::new(AArch64Opcode::MulRR, vec![vreg(2), vreg(0), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![m0, m1, mul, ret]);

        let mut cf = ConstantFolding;
        assert!(cf.run(&mut func));

        let folded = func.inst(llvm2_ir::InstId(2));
        assert_eq!(folded.opcode, AArch64Opcode::MovI);
        assert_eq!(folded.operands[1], imm(42));
    }

    #[test]
    fn test_fold_logical() {
        let m0 = MachInst::new(AArch64Opcode::MovI, vec![vreg(0), imm(0xFF)]);
        let m1 = MachInst::new(AArch64Opcode::MovI, vec![vreg(1), imm(0x0F)]);
        let and = MachInst::new(AArch64Opcode::AndRR, vec![vreg(2), vreg(0), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![m0, m1, and, ret]);

        let mut cf = ConstantFolding;
        assert!(cf.run(&mut func));

        let folded = func.inst(llvm2_ir::InstId(2));
        assert_eq!(folded.operands[1], imm(0x0F));
    }

    #[test]
    fn test_fold_shift() {
        let m0 = MachInst::new(AArch64Opcode::MovI, vec![vreg(0), imm(1)]);
        let lsl = MachInst::new(AArch64Opcode::LslRI, vec![vreg(1), vreg(0), imm(4)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![m0, lsl, ret]);

        let mut cf = ConstantFolding;
        assert!(cf.run(&mut func));

        let folded = func.inst(llvm2_ir::InstId(1));
        assert_eq!(folded.operands[1], imm(16));
    }

    #[test]
    fn test_fold_neg() {
        let m0 = MachInst::new(AArch64Opcode::MovI, vec![vreg(0), imm(42)]);
        let neg = MachInst::new(AArch64Opcode::Neg, vec![vreg(1), vreg(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![m0, neg, ret]);

        let mut cf = ConstantFolding;
        assert!(cf.run(&mut func));

        // -42 is a wide negative value (0xFFFF_FFFF_FFFF_FFD6 as u64), so
        // the folder expands it as MOVZ + MOVK chain rather than a single
        // MovI. Check the reconstructed runtime value rather than a
        // specific slot.
        assert_eq!(materialized_value(&func, 1), Some(-42));
    }

    #[test]
    fn test_no_fold_unknown_operand() {
        // v1 = add v0, #5  where v0 is NOT a known constant
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(5)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let mut cf = ConstantFolding;
        assert!(!cf.run(&mut func));
    }

    #[test]
    fn test_chain_folding() {
        // v0 = movi #10
        // v1 = add v0, #5   → v1 = movi #15
        // v2 = add v1, #3   → v2 = movi #18 (chain: v1 now known)
        let m0 = MachInst::new(AArch64Opcode::MovI, vec![vreg(0), imm(10)]);
        let a1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(5)]);
        let a2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(1), imm(3)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![m0, a1, a2, ret]);

        let mut cf = ConstantFolding;
        assert!(cf.run(&mut func));

        // Both adds should be folded
        let folded1 = func.inst(llvm2_ir::InstId(1));
        assert_eq!(folded1.opcode, AArch64Opcode::MovI);
        assert_eq!(folded1.operands[1], imm(15));

        let folded2 = func.inst(llvm2_ir::InstId(2));
        assert_eq!(folded2.opcode, AArch64Opcode::MovI);
        assert_eq!(folded2.operands[1], imm(18));
    }

    #[test]
    fn test_fold_wrapping() {
        // Test wrapping arithmetic (no overflow trap).
        //
        // Note: i64::MAX seeds v0, but the seed is already a wide value —
        // it will not be foldable at the AddRI because our MovI seed in
        // the test doesn't fit in 16 bits. Use a small seed and a wide
        // computed result instead: 0xFFFF + 1 = 0x10000 (wide).
        let m0 = MachInst::new(AArch64Opcode::MovI, vec![vreg(0), imm(0xFFFF)]);
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![m0, add, ret]);

        let mut cf = ConstantFolding;
        assert!(cf.run(&mut func));

        // 0x10000 is > 16 bits so the folder expands the AddRI slot as a
        // MOVZ+MOVK chain. Validate the composed runtime value.
        assert_eq!(materialized_value(&func, 1), Some(0x10000));
    }

    // --- regression tests for issue #366 --------------------------------
    //
    // ISel materializes 64-bit constants with a MOVZ + MOVK chain writing
    // to the same vreg. Before the fix, the folder seeded its tracker from
    // `is_move_imm()` (which excludes MOVK) and used the raw 16-bit MOVZ
    // immediate as the "full" constant, producing wrong folded values for
    // any vreg built this way.

    #[test]
    fn test_fold_movz_movk_chain_materializes_full_constant() {
        // v0 = MOVZ #0x79F9
        // v0 = MOVK v0, #0x9E37, lsl #16
        // v0 = MOVK v0, #0x6791, lsl #32
        // v0 = MOVK v0, #0x1656, lsl #48
        // v1 = MOVZ #0x0002
        // v2 = MulRR v0, v1
        // expect v2 folded to 0x165667919E3779F9 * 2 (wrapping)
        let k = 0x165667919E3779F9u64 as i64;

        let movz = MachInst::new(AArch64Opcode::Movz, vec![vreg(0), imm(0x79F9)]);
        let movk1 = MachInst::new(
            AArch64Opcode::Movk,
            vec![vreg(0), imm(0x9E37), imm(16)],
        );
        let movk2 = MachInst::new(
            AArch64Opcode::Movk,
            vec![vreg(0), imm(0x6791), imm(32)],
        );
        let movk3 = MachInst::new(
            AArch64Opcode::Movk,
            vec![vreg(0), imm(0x1656), imm(48)],
        );
        let m1 = MachInst::new(AArch64Opcode::MovI, vec![vreg(1), imm(2)]);
        let mul = MachInst::new(
            AArch64Opcode::MulRR,
            vec![vreg(2), vreg(0), vreg(1)],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![movz, movk1, movk2, movk3, m1, mul, ret]);

        let mut cf = ConstantFolding;
        assert!(cf.run(&mut func));

        // The MulRR is replaced by a materialization of the full 64-bit
        // product. Since the product is wide, the folder expands it as a
        // MOVZ + MOVK chain writing into v2.
        let expected = k.wrapping_mul(2);
        assert_eq!(
            materialized_value(&func, 2),
            Some(expected),
            "fold should see full 64-bit constant 0x{:016x}, not low16 only",
            k as u64
        );
    }

    #[test]
    fn test_fold_movz_movk_xor_pattern() {
        // The xxh3 avalanche uses a 64-bit EOR with a freshly built
        // constant. Regression for #366.
        let k = 0x165667919E3779F9u64 as i64;

        let movz = MachInst::new(AArch64Opcode::Movz, vec![vreg(0), imm(0x79F9)]);
        let movk1 = MachInst::new(
            AArch64Opcode::Movk,
            vec![vreg(0), imm(0x9E37), imm(16)],
        );
        let movk2 = MachInst::new(
            AArch64Opcode::Movk,
            vec![vreg(0), imm(0x6791), imm(32)],
        );
        let movk3 = MachInst::new(
            AArch64Opcode::Movk,
            vec![vreg(0), imm(0x1656), imm(48)],
        );
        let m1 = MachInst::new(AArch64Opcode::MovI, vec![vreg(1), imm(0x1234)]);
        let eor = MachInst::new(
            AArch64Opcode::EorRR,
            vec![vreg(2), vreg(0), vreg(1)],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![movz, movk1, movk2, movk3, m1, eor, ret]);

        let mut cf = ConstantFolding;
        assert!(cf.run(&mut func));

        assert_eq!(materialized_value(&func, 2), Some(k ^ 0x1234));
    }

    #[test]
    fn test_fold_movn_zero_is_all_ones() {
        // MOVN #0  →  !0  =  0xFFFF_FFFF_FFFF_FFFF (as i64: -1)
        // v0 = MOVN #0
        // v1 = MOVI #1
        // v2 = AddRR v0, v1  → 0 (wrapping)
        let movn = MachInst::new(AArch64Opcode::Movn, vec![vreg(0), imm(0)]);
        let m1 = MachInst::new(AArch64Opcode::MovI, vec![vreg(1), imm(1)]);
        let add = MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(2), vreg(0), vreg(1)],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![movn, m1, add, ret]);

        let mut cf = ConstantFolding;
        assert!(cf.run(&mut func));

        let folded = func.inst(llvm2_ir::InstId(2));
        assert_eq!(folded.opcode, AArch64Opcode::MovI);
        assert_eq!(folded.operands[1], imm(0));
    }

    #[test]
    fn test_movk_without_seed_does_not_produce_wrong_value() {
        // If MOVK runs on a vreg whose value is unknown (e.g. an
        // argument), we MUST NOT synthesize a bogus value. Downstream
        // folds that use that vreg must fall through.
        //
        // v0 = <unknown, e.g. arg>
        // v0 = MOVK v0, #0xABCD, lsl #0   (patching low16 of an unknown)
        // v1 = MovI #2
        // v2 = MulRR v0, v1      -- must NOT fold
        let movk = MachInst::new(
            AArch64Opcode::Movk,
            vec![vreg(0), imm(0xABCD), imm(0)],
        );
        let m1 = MachInst::new(AArch64Opcode::MovI, vec![vreg(1), imm(2)]);
        let mul = MachInst::new(
            AArch64Opcode::MulRR,
            vec![vreg(2), vreg(0), vreg(1)],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![movk, m1, mul, ret]);

        let mut cf = ConstantFolding;
        cf.run(&mut func);

        let mul_after = func.inst(llvm2_ir::InstId(2));
        assert_eq!(
            mul_after.opcode,
            AArch64Opcode::MulRR,
            "MulRR must not fold when one operand has an unknown base value"
        );
    }

    #[test]
    fn test_redef_invalidates_stale_constant() {
        // v0 = MovI #10
        // v0 = AddRR v0, v_unknown   -- redef v0 with a non-foldable op
        // v2 = MulRR v0, v1
        //
        // Without invalidation, the tracker would still believe v0 = 10
        // and fold the mul using the stale value.
        let m0 = MachInst::new(AArch64Opcode::MovI, vec![vreg(0), imm(10)]);
        // v99 is an unknown vreg; AddRR cannot be folded.
        let redef = MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(0), vreg(0), vreg(99)],
        );
        let m1 = MachInst::new(AArch64Opcode::MovI, vec![vreg(1), imm(3)]);
        let mul = MachInst::new(
            AArch64Opcode::MulRR,
            vec![vreg(2), vreg(0), vreg(1)],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![m0, redef, m1, mul, ret]);

        let mut cf = ConstantFolding;
        cf.run(&mut func);

        let mul_after = func.inst(llvm2_ir::InstId(3));
        assert_eq!(
            mul_after.opcode,
            AArch64Opcode::MulRR,
            "MulRR must not fold after a non-foldable redefinition of v0"
        );
    }

    /// Regression for issue #366 residual: the full xxh3 empty-input
    /// avalanche mixing sequence should fold to the correct constant.
    ///
    /// avalanche(h) = { h ^= h >> 37; h *= 0x165667919E3779F9; h ^= h >> 32 }
    ///
    /// Input h0 = 0x7c1b74eed9f584e5 (xor of secret[56..64] and secret[64..72]).
    /// Expected final value = 0x067e2f2a6d83f618.
    #[test]
    fn test_fold_xxh3_empty_avalanche() {
        // Materialize h0 = 0x7c1b74eed9f584e5 via MOVZ + MOVK chain.
        let h0 = 0x7c1b74eed9f584e5u64 as i64;
        let mut insts = Vec::new();
        insts.push(MachInst::new(AArch64Opcode::Movz, vec![vreg(0), imm(0x84e5)]));
        insts.push(MachInst::new(AArch64Opcode::Movk, vec![vreg(0), imm(0xd9f5), imm(16)]));
        insts.push(MachInst::new(AArch64Opcode::Movk, vec![vreg(0), imm(0x74ee), imm(32)]));
        insts.push(MachInst::new(AArch64Opcode::Movk, vec![vreg(0), imm(0x7c1b), imm(48)]));

        // h_shr37 = h0 >> 37
        insts.push(MachInst::new(AArch64Opcode::LsrRI, vec![vreg(1), vreg(0), imm(37)]));
        // h1 = h0 ^ h_shr37
        insts.push(MachInst::new(AArch64Opcode::EorRR, vec![vreg(2), vreg(0), vreg(1)]));

        // mul_const = 0x165667919E3779F9 via MOVZ+MOVK
        insts.push(MachInst::new(AArch64Opcode::Movz, vec![vreg(3), imm(0x79F9)]));
        insts.push(MachInst::new(AArch64Opcode::Movk, vec![vreg(3), imm(0x9E37), imm(16)]));
        insts.push(MachInst::new(AArch64Opcode::Movk, vec![vreg(3), imm(0x6791), imm(32)]));
        insts.push(MachInst::new(AArch64Opcode::Movk, vec![vreg(3), imm(0x1656), imm(48)]));

        // h2 = h1 * mul_const
        insts.push(MachInst::new(AArch64Opcode::MulRR, vec![vreg(4), vreg(2), vreg(3)]));

        // h_shr32 = h2 >> 32
        insts.push(MachInst::new(AArch64Opcode::LsrRI, vec![vreg(5), vreg(4), imm(32)]));
        // h3 = h2 ^ h_shr32
        insts.push(MachInst::new(AArch64Opcode::EorRR, vec![vreg(6), vreg(4), vreg(5)]));

        insts.push(MachInst::new(AArch64Opcode::Ret, vec![]));

        let mut func = make_func_with_insts(insts);

        let mut cf = ConstantFolding;
        cf.run(&mut func);

        // Expected intermediate values.
        let h_shr37 = (h0 as u64 >> 37) as i64;
        let h1 = h0 ^ h_shr37;
        let mul_const = 0x165667919E3779F9u64 as i64;
        let h2 = h1.wrapping_mul(mul_const);
        let h_shr32 = (h2 as u64 >> 32) as i64;
        let h3 = h2 ^ h_shr32;
        assert_eq!(h3 as u64, 0x067e2f2a6d83f618u64, "sanity: expected value");

        assert_eq!(materialized_value(&func, 0), Some(h0), "h0 materialized");
        assert_eq!(materialized_value(&func, 2), Some(h1), "h1 = h0 ^ (h0>>37)");
        assert_eq!(materialized_value(&func, 3), Some(mul_const), "mul_const materialized");
        assert_eq!(materialized_value(&func, 4), Some(h2), "h2 = h1 * mul_const");
        assert_eq!(materialized_value(&func, 5), Some(h_shr32), "h_shr32 = h2 >> 32");
        assert_eq!(
            materialized_value(&func, 6),
            Some(h3),
            "h3 = h2 ^ (h2>>32) = {:#018x}",
            h3 as u64
        );
    }

    // ------------------------------------------------------------------
    // source_loc preservation across constant-folding rewrites (#376).
    //
    // Constant folding rewrites the destination instruction slot as a
    // MOVZ/MOVK sequence. The original source line (e.g., the `x + y`
    // expression the user wrote) must carry through to every emitted
    // materialization instruction, otherwise lldb __debug_line has a
    // gap at the folded computation.
    // ------------------------------------------------------------------

    #[test]
    fn test_source_loc_preserved_across_const_fold_narrow() {
        use llvm2_ir::{AArch64Opcode, InstId, SourceLoc};
        let loc = SourceLoc { file: 3, line: 77, col: 12 };

        // v0 = movi #10
        // v1 = add v0, #20  (with source_loc)
        // ret
        let m0 = MachInst::new(AArch64Opcode::MovI, vec![vreg(0), imm(10)]);
        let mut add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(20)]);
        add.source_loc = Some(loc);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![m0, add, ret]);

        let mut cf = ConstantFolding;
        assert!(cf.run(&mut func));

        // After folding, InstId(1) is a MovI/MOVZ-style materialization.
        let folded = func.inst(InstId(1));
        assert_eq!(
            folded.source_loc,
            Some(loc),
            "const-fold must preserve source_loc on narrow materialization (issue #376)"
        );
    }

    #[test]
    fn test_source_loc_preserved_across_const_fold_wide() {
        use llvm2_ir::{AArch64Opcode, InstId, SourceLoc};
        let loc = SourceLoc { file: 4, line: 200, col: 0 };

        // Build a large constant via MOVZ + MOVKs, then multiply to force a
        // wide (multi-instruction) materialization when the product folds.
        let movz = MachInst::new(AArch64Opcode::Movz, vec![vreg(0), imm(0x79F9)]);
        let m1 = MachInst::new(AArch64Opcode::MovI, vec![vreg(1), imm(2)]);
        let mut mul = MachInst::new(
            AArch64Opcode::MulRR,
            vec![vreg(2), vreg(0), vreg(1)],
        );
        mul.source_loc = Some(loc);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![movz, m1, mul, ret]);

        let mut cf = ConstantFolding;
        assert!(cf.run(&mut func));

        // The original mul slot becomes MOVZ (low 16 bits) with source_loc.
        let folded = func.inst(InstId(2));
        assert_eq!(
            folded.source_loc,
            Some(loc),
            "const-fold must preserve source_loc on wide MOVZ slot (issue #376)"
        );

        // Any MOVK instructions appended to materialize upper 16-bit chunks
        // must also carry source_loc. Scan remaining new insts in the block.
        let block = func.block(func.entry);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            if inst.opcode == AArch64Opcode::Movk {
                assert_eq!(
                    inst.source_loc,
                    Some(loc),
                    "const-fold MOVK materialization must preserve source_loc (issue #376)"
                );
            }
        }
    }
}
