// llvm2-opt - AArch64 Compare-and-Branch Fusion
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! AArch64 compare-and-branch fusion pass.
//!
//! Fuses separate compare/test + conditional branch instruction pairs into
//! single combined compare-and-branch instructions, reducing code size and
//! improving branch prediction on AArch64.
//!
//! # Patterns
//!
//! | Pattern | Transformation |
//! |---------|---------------|
//! | `CMP Rn, #0` + `B.EQ target` | `CBZ Rn, target` |
//! | `CMP Rn, #0` + `B.NE target` | `CBNZ Rn, target` |
//! | `TST Rn, #(1<<bit)` + `B.EQ target` | `TBZ Rn, #bit, target` |
//! | `TST Rn, #(1<<bit)` + `B.NE target` | `TBNZ Rn, #bit, target` |
//!
//! # Safety Constraints
//!
//! - The CMP/TST and BCond must be consecutive in the basic block (no
//!   intervening flag-setting instructions).
//! - After fusion, the CMP/TST instruction is removed (it is dead because
//!   the fused instruction encodes the comparison implicitly).
//! - CBZ/CBNZ only works with compare-to-zero; non-zero immediate comparisons
//!   are not fusible.
//! - TBZ/TBNZ only works with TST against a single-bit mask (power of 2).
//!
//! # Relationship to Other Passes
//!
//! - `peephole.rs` handles single-instruction simplifications (e.g., `add x0, x1, #0` -> `mov`).
//! - `cmp_select.rs` handles diamond CFG patterns -> CSEL/CSET.
//! - This pass handles linear CMP/TST + BCond fusion -> CBZ/CBNZ/TBZ/TBNZ.

use std::collections::HashSet;

use llvm2_ir::{AArch64Opcode, CondCode, InstId, MachFunction, MachInst, MachOperand};

use crate::pass_manager::MachinePass;

/// AArch64 compare-and-branch fusion pass.
pub struct CmpBranchFusion;

impl MachinePass for CmpBranchFusion {
    fn name(&self) -> &str {
        "cmp-branch-fusion"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        let mut changed = false;
        let mut to_delete: HashSet<InstId> = HashSet::new();

        for block_id in func.block_order.clone() {
            let block = func.block(block_id);
            let insts = block.insts.clone();

            // Sliding window of consecutive pairs.
            if insts.len() < 2 {
                continue;
            }

            for i in 0..insts.len() - 1 {
                let cmp_id = insts[i];
                let bcond_id = insts[i + 1];

                // Skip if the CMP is already marked for deletion.
                if to_delete.contains(&cmp_id) {
                    continue;
                }

                let cmp_inst = func.inst(cmp_id);
                let bcond_inst = func.inst(bcond_id);

                // Second instruction must be BCond.
                if bcond_inst.opcode != AArch64Opcode::BCond {
                    continue;
                }

                // Decode BCond operands: [Imm(cond_encoding), Block(target)]
                if bcond_inst.operands.len() < 2 {
                    continue;
                }
                let cond_encoding = match bcond_inst.operands[0].as_imm() {
                    Some(v) => v as u8,
                    None => continue,
                };
                let target = match &bcond_inst.operands[1] {
                    MachOperand::Block(bid) => *bid,
                    _ => continue,
                };
                let cond = match decode_cond(cond_encoding) {
                    Some(c) => c,
                    None => continue,
                };

                // First instruction must be a flag-setting instruction.
                if !sets_flags(cmp_inst.opcode) {
                    continue;
                }

                // Try CBZ/CBNZ fusion (CmpRI Rn, #0).
                if let Some(fused) = try_fuse_cbz(cmp_inst, cond, target) {
                    *func.inst_mut(bcond_id) = fused;
                    to_delete.insert(cmp_id);
                    changed = true;
                    continue;
                }

                // Try TBZ/TBNZ fusion (Tst Rn, #(1<<bit)).
                if let Some(fused) = try_fuse_tbz(cmp_inst, cond, target) {
                    *func.inst_mut(bcond_id) = fused;
                    to_delete.insert(cmp_id);
                    changed = true;
                    continue;
                }
            }
        }

        // Remove dead CMP/TST instructions from blocks.
        if !to_delete.is_empty() {
            for block_id in func.block_order.clone() {
                let block = func.block_mut(block_id);
                block.insts.retain(|id| !to_delete.contains(id));
            }
        }

        changed
    }
}

/// Try to fuse CmpRI Rn, #0 + B.EQ/B.NE into CBZ/CBNZ.
///
/// - CmpRI Rn, #0 + B.EQ target -> CBZ Rn, target
/// - CmpRI Rn, #0 + B.NE target -> CBNZ Rn, target
fn try_fuse_cbz(
    cmp_inst: &MachInst,
    cond: CondCode,
    target: llvm2_ir::BlockId,
) -> Option<MachInst> {
    // Must be CmpRI with immediate 0.
    if cmp_inst.opcode != AArch64Opcode::CmpRI {
        return None;
    }
    if cmp_inst.operands.len() < 2 {
        return None;
    }

    // Second operand must be immediate 0.
    let imm_val = cmp_inst.operands[1].as_imm()?;
    if imm_val != 0 {
        return None;
    }

    // Only EQ and NE conditions are fusible to CBZ/CBNZ.
    let opcode = match cond {
        CondCode::EQ => AArch64Opcode::Cbz,
        CondCode::NE => AArch64Opcode::Cbnz,
        _ => return None,
    };

    // CBZ/CBNZ operands: [Rn, Block(target)]
    let rn = cmp_inst.operands[0].clone();
    Some(MachInst::new(opcode, vec![rn, MachOperand::Block(target)]))
}

/// Try to fuse TST Rn, #(1<<bit) + B.EQ/B.NE into TBZ/TBNZ.
///
/// - TST Rn, #(1<<bit) + B.EQ target -> TBZ Rn, #bit, target
/// - TST Rn, #(1<<bit) + B.NE target -> TBNZ Rn, #bit, target
fn try_fuse_tbz(
    cmp_inst: &MachInst,
    cond: CondCode,
    target: llvm2_ir::BlockId,
) -> Option<MachInst> {
    // Must be TST.
    if cmp_inst.opcode != AArch64Opcode::Tst {
        return None;
    }
    if cmp_inst.operands.len() < 2 {
        return None;
    }

    // Second operand must be an immediate that is a power of 2.
    // Note: 1i64 << 63 is negative in i64 but valid as a 64-bit mask.
    // We cast to u64 for the power-of-two check.
    let mask = cmp_inst.operands[1].as_imm()?;
    let mask_u64 = mask as u64;
    if !is_power_of_two(mask_u64) {
        return None;
    }

    let bit = mask_u64.trailing_zeros() as i64;

    // Only EQ and NE conditions are fusible.
    // TST sets Z flag: B.EQ (Z=1) means bit was 0 -> TBZ
    //                  B.NE (Z=0) means bit was 1 -> TBNZ
    let opcode = match cond {
        CondCode::EQ => AArch64Opcode::Tbz,
        CondCode::NE => AArch64Opcode::Tbnz,
        _ => return None,
    };

    // TBZ/TBNZ operands: [Rn, Imm(bit), Block(target)]
    let rn = cmp_inst.operands[0].clone();
    Some(MachInst::new(
        opcode,
        vec![rn, MachOperand::Imm(bit), MachOperand::Block(target)],
    ))
}

/// Returns true if `v` is a power of two (exactly one bit set).
fn is_power_of_two(v: u64) -> bool {
    v != 0 && (v & (v - 1)) == 0
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

/// Returns true if the given opcode sets the NZCV condition flags.
fn sets_flags(opcode: AArch64Opcode) -> bool {
    matches!(
        opcode,
        AArch64Opcode::CmpRR
            | AArch64Opcode::CmpRI
            | AArch64Opcode::CMPWrr
            | AArch64Opcode::CMPXrr
            | AArch64Opcode::CMPWri
            | AArch64Opcode::CMPXri
            | AArch64Opcode::Tst
            | AArch64Opcode::Fcmp
            | AArch64Opcode::AddsRR
            | AArch64Opcode::AddsRI
            | AArch64Opcode::SubsRR
            | AArch64Opcode::SubsRI
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass_manager::MachinePass;
    use llvm2_ir::{
        AArch64Opcode, BlockId, CondCode, MachFunction, MachInst, MachOperand,
        RegClass, Signature, VReg,
    };

    fn vreg(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
    }

    fn imm(val: i64) -> MachOperand {
        MachOperand::Imm(val)
    }

    /// Build a function with CMP/TST + BCond as the last two instructions.
    /// The BCond targets a second block (bb1) containing only RET.
    fn make_func_with_branch(
        cmp: MachInst,
        cond: CondCode,
    ) -> MachFunction {
        let mut func = MachFunction::new(
            "test_fusion".to_string(),
            Signature::new(vec![], vec![]),
        );

        let bb0 = func.entry;
        let bb1 = func.create_block();

        // bb0: CMP + BCond
        let cmp_id = func.push_inst(cmp);
        func.append_inst(bb0, cmp_id);

        let bcond = MachInst::new(
            AArch64Opcode::BCond,
            vec![imm(cond.encoding() as i64), MachOperand::Block(bb1)],
        );
        let bcond_id = func.push_inst(bcond);
        func.append_inst(bb0, bcond_id);

        // bb1: RET
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(bb1, ret_id);

        func.add_edge(bb0, bb1);

        func
    }

    // ---- CBZ fusion tests ----

    #[test]
    fn test_cbz_from_cmpri_zero_beq() {
        // CMP v0, #0; B.EQ bb1 -> CBZ v0, bb1
        let cmp = MachInst::new(AArch64Opcode::CmpRI, vec![vreg(0), imm(0)]);
        let mut func = make_func_with_branch(cmp, CondCode::EQ);

        let mut pass = CmpBranchFusion;
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        // CMP should be deleted, only fused CBZ remains.
        assert_eq!(block.insts.len(), 1);

        let fused = func.inst(block.insts[0]);
        assert_eq!(fused.opcode, AArch64Opcode::Cbz);
        assert_eq!(fused.operands[0], vreg(0));
        assert_eq!(fused.operands[1], MachOperand::Block(BlockId(1)));
    }

    #[test]
    fn test_cbnz_from_cmpri_zero_bne() {
        // CMP v0, #0; B.NE bb1 -> CBNZ v0, bb1
        let cmp = MachInst::new(AArch64Opcode::CmpRI, vec![vreg(0), imm(0)]);
        let mut func = make_func_with_branch(cmp, CondCode::NE);

        let mut pass = CmpBranchFusion;
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 1);

        let fused = func.inst(block.insts[0]);
        assert_eq!(fused.opcode, AArch64Opcode::Cbnz);
        assert_eq!(fused.operands[0], vreg(0));
        assert_eq!(fused.operands[1], MachOperand::Block(BlockId(1)));
    }

    // ---- TBZ fusion tests ----

    #[test]
    fn test_tbz_from_tst_beq() {
        // TST v0, #(1<<3); B.EQ bb1 -> TBZ v0, #3, bb1
        let tst = MachInst::new(AArch64Opcode::Tst, vec![vreg(0), imm(1 << 3)]);
        let mut func = make_func_with_branch(tst, CondCode::EQ);

        let mut pass = CmpBranchFusion;
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 1);

        let fused = func.inst(block.insts[0]);
        assert_eq!(fused.opcode, AArch64Opcode::Tbz);
        assert_eq!(fused.operands[0], vreg(0));
        assert_eq!(fused.operands[1], imm(3)); // bit number
        assert_eq!(fused.operands[2], MachOperand::Block(BlockId(1)));
    }

    #[test]
    fn test_tbnz_from_tst_bne() {
        // TST v0, #(1<<7); B.NE bb1 -> TBNZ v0, #7, bb1
        let tst = MachInst::new(AArch64Opcode::Tst, vec![vreg(0), imm(1 << 7)]);
        let mut func = make_func_with_branch(tst, CondCode::NE);

        let mut pass = CmpBranchFusion;
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 1);

        let fused = func.inst(block.insts[0]);
        assert_eq!(fused.opcode, AArch64Opcode::Tbnz);
        assert_eq!(fused.operands[0], vreg(0));
        assert_eq!(fused.operands[1], imm(7)); // bit number
        assert_eq!(fused.operands[2], MachOperand::Block(BlockId(1)));
    }

    #[test]
    fn test_tbz_bit_zero() {
        // TST v0, #1; B.EQ bb1 -> TBZ v0, #0, bb1
        let tst = MachInst::new(AArch64Opcode::Tst, vec![vreg(0), imm(1)]);
        let mut func = make_func_with_branch(tst, CondCode::EQ);

        let mut pass = CmpBranchFusion;
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        let fused = func.inst(block.insts[0]);
        assert_eq!(fused.opcode, AArch64Opcode::Tbz);
        assert_eq!(fused.operands[1], imm(0));
    }

    #[test]
    fn test_tbz_bit_63() {
        // TST v0, #(1<<63); B.EQ bb1 -> TBZ v0, #63, bb1
        let tst = MachInst::new(AArch64Opcode::Tst, vec![vreg(0), imm(1i64 << 63)]);
        let mut func = make_func_with_branch(tst, CondCode::EQ);

        let mut pass = CmpBranchFusion;
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        let fused = func.inst(block.insts[0]);
        assert_eq!(fused.opcode, AArch64Opcode::Tbz);
        // 1<<63 is negative in i64 but bit 63 as u64 trailing_zeros = 63
        assert_eq!(fused.operands[1], imm(63));
    }

    // ---- Negative tests ----

    #[test]
    fn test_no_fusion_cmp_nonzero() {
        // CMP v0, #42; B.EQ bb1 -> NO fusion (CBZ only for zero)
        let cmp = MachInst::new(AArch64Opcode::CmpRI, vec![vreg(0), imm(42)]);
        let mut func = make_func_with_branch(cmp, CondCode::EQ);

        let mut pass = CmpBranchFusion;
        assert!(!pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // CMP + BCond both remain
    }

    #[test]
    fn test_no_fusion_cmp_rr() {
        // CMP v0, v1; B.EQ bb1 -> NO fusion (CmpRR not fusible to CBZ)
        let cmp = MachInst::new(AArch64Opcode::CmpRR, vec![vreg(0), vreg(1)]);
        let mut func = make_func_with_branch(cmp, CondCode::EQ);

        let mut pass = CmpBranchFusion;
        assert!(!pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2);
    }

    #[test]
    fn test_no_fusion_cmp_zero_bge() {
        // CMP v0, #0; B.GE bb1 -> NO fusion (GE not fusible to CBZ/CBNZ)
        let cmp = MachInst::new(AArch64Opcode::CmpRI, vec![vreg(0), imm(0)]);
        let mut func = make_func_with_branch(cmp, CondCode::GE);

        let mut pass = CmpBranchFusion;
        assert!(!pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2);
    }

    #[test]
    fn test_no_fusion_tst_non_power_of_two() {
        // TST v0, #3; B.EQ bb1 -> NO fusion (3 is not a power of 2)
        let tst = MachInst::new(AArch64Opcode::Tst, vec![vreg(0), imm(3)]);
        let mut func = make_func_with_branch(tst, CondCode::EQ);

        let mut pass = CmpBranchFusion;
        assert!(!pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2);
    }

    #[test]
    fn test_no_fusion_tst_zero_mask() {
        // TST v0, #0; B.EQ bb1 -> NO fusion (0 is not a valid single-bit mask)
        let tst = MachInst::new(AArch64Opcode::Tst, vec![vreg(0), imm(0)]);
        let mut func = make_func_with_branch(tst, CondCode::EQ);

        let mut pass = CmpBranchFusion;
        assert!(!pass.run(&mut func));
    }

    #[test]
    fn test_no_fusion_tst_bge() {
        // TST v0, #4; B.GE bb1 -> NO fusion (GE not fusible)
        let tst = MachInst::new(AArch64Opcode::Tst, vec![vreg(0), imm(4)]);
        let mut func = make_func_with_branch(tst, CondCode::GE);

        let mut pass = CmpBranchFusion;
        assert!(!pass.run(&mut func));
    }

    #[test]
    fn test_no_fusion_non_consecutive() {
        // CMP v0, #0; ADD v1, v2, v3; B.EQ bb1
        // ADD between CMP and BCond breaks the pair.
        let mut func = MachFunction::new(
            "test_non_consecutive".to_string(),
            Signature::new(vec![], vec![]),
        );

        let bb0 = func.entry;
        let bb1 = func.create_block();

        let cmp_id = func.push_inst(MachInst::new(
            AArch64Opcode::CmpRI,
            vec![vreg(0), imm(0)],
        ));
        func.append_inst(bb0, cmp_id);

        let add_id = func.push_inst(MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(1), vreg(2), vreg(3)],
        ));
        func.append_inst(bb0, add_id);

        let bcond_id = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![imm(CondCode::EQ.encoding() as i64), MachOperand::Block(bb1)],
        ));
        func.append_inst(bb0, bcond_id);

        let ret_id = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb1, ret_id);

        func.add_edge(bb0, bb1);

        let mut pass = CmpBranchFusion;
        // The CMP and BCond are not consecutive (ADD is between them),
        // so no fusion should occur.
        assert!(!pass.run(&mut func));

        let block = func.block(bb0);
        assert_eq!(block.insts.len(), 3);
    }

    // ---- Idempotency ----

    #[test]
    fn test_idempotent() {
        let cmp = MachInst::new(AArch64Opcode::CmpRI, vec![vreg(0), imm(0)]);
        let mut func = make_func_with_branch(cmp, CondCode::EQ);

        let mut pass = CmpBranchFusion;
        assert!(pass.run(&mut func));   // First: transforms
        assert!(!pass.run(&mut func));  // Second: no change
    }

    // ---- Edge cases ----

    #[test]
    fn test_empty_block_no_crash() {
        let mut func = MachFunction::new(
            "empty".to_string(),
            Signature::new(vec![], vec![]),
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(func.entry, ret_id);

        let mut pass = CmpBranchFusion;
        assert!(!pass.run(&mut func));
    }

    #[test]
    fn test_single_instruction_block() {
        let mut func = MachFunction::new(
            "single".to_string(),
            Signature::new(vec![], vec![]),
        );
        // Block with only one instruction: no pair to fuse.
        let cmp = MachInst::new(AArch64Opcode::CmpRI, vec![vreg(0), imm(0)]);
        let cmp_id = func.push_inst(cmp);
        func.append_inst(func.entry, cmp_id);

        let mut pass = CmpBranchFusion;
        assert!(!pass.run(&mut func));
    }

    #[test]
    fn test_multiple_fusions_in_different_blocks() {
        // Two blocks, each with a fusible CMP+BCond.
        let mut func = MachFunction::new(
            "multi_block".to_string(),
            Signature::new(vec![], vec![]),
        );

        let bb0 = func.entry;
        let bb1 = func.create_block();
        let bb2 = func.create_block();

        // bb0: CMP v0, #0; B.EQ bb1
        let cmp0_id = func.push_inst(MachInst::new(
            AArch64Opcode::CmpRI,
            vec![vreg(0), imm(0)],
        ));
        func.append_inst(bb0, cmp0_id);
        let bcond0_id = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![imm(CondCode::EQ.encoding() as i64), MachOperand::Block(bb1)],
        ));
        func.append_inst(bb0, bcond0_id);

        // bb1: TST v1, #4; B.NE bb2
        let tst1_id = func.push_inst(MachInst::new(
            AArch64Opcode::Tst,
            vec![vreg(1), imm(4)],
        ));
        func.append_inst(bb1, tst1_id);
        let bcond1_id = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![imm(CondCode::NE.encoding() as i64), MachOperand::Block(bb2)],
        ));
        func.append_inst(bb1, bcond1_id);

        // bb2: RET
        let ret_id = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret_id);

        func.add_edge(bb0, bb1);
        func.add_edge(bb1, bb2);

        let mut pass = CmpBranchFusion;
        assert!(pass.run(&mut func));

        // bb0 should have 1 inst: CBZ
        let block0 = func.block(bb0);
        assert_eq!(block0.insts.len(), 1);
        assert_eq!(func.inst(block0.insts[0]).opcode, AArch64Opcode::Cbz);

        // bb1 should have 1 inst: TBNZ
        let block1 = func.block(bb1);
        assert_eq!(block1.insts.len(), 1);
        assert_eq!(func.inst(block1.insts[0]).opcode, AArch64Opcode::Tbnz);
        assert_eq!(func.inst(block1.insts[0]).operands[1], imm(2)); // bit 2 for mask 4
    }

    // ---- Helper function tests ----

    #[test]
    fn test_is_power_of_two() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(4));
        assert!(is_power_of_two(1 << 63));
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(6));
        assert!(!is_power_of_two(u64::MAX));
    }

    #[test]
    fn test_decode_cond_all_values() {
        assert_eq!(decode_cond(0b0000), Some(CondCode::EQ));
        assert_eq!(decode_cond(0b0001), Some(CondCode::NE));
        assert_eq!(decode_cond(0b0010), Some(CondCode::HS));
        assert_eq!(decode_cond(0b0011), Some(CondCode::LO));
        assert_eq!(decode_cond(0b1110), Some(CondCode::AL));
        assert_eq!(decode_cond(0b1111), Some(CondCode::NV));
        assert_eq!(decode_cond(16), None);
    }

    #[test]
    fn test_sets_flags() {
        assert!(sets_flags(AArch64Opcode::CmpRR));
        assert!(sets_flags(AArch64Opcode::CmpRI));
        assert!(sets_flags(AArch64Opcode::Tst));
        assert!(sets_flags(AArch64Opcode::AddsRR));
        assert!(!sets_flags(AArch64Opcode::AddRR));
        assert!(!sets_flags(AArch64Opcode::MovR));
        assert!(!sets_flags(AArch64Opcode::B));
    }
}
