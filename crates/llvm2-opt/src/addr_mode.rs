// llvm2-opt - AArch64 Address Mode Formation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! AArch64 address mode formation pass.
//!
//! A late machine-level optimization that folds ADD instructions into
//! the addressing mode of subsequent LDR/STR instructions, exploiting
//! AArch64's rich addressing modes.
//!
//! # Patterns
//!
//! | Pattern | Transformation |
//! |---------|---------------|
//! | `ADD Xd, Xn, #imm` + `LDR Xt, [Xd, #0]` | `LDR Xt, [Xn, #imm]` |
//! | `ADD Xd, Xn, #imm` + `STR Xt, [Xd, #0]` | `STR Xt, [Xn, #imm]` |
//! | `ADD Xd, Xn, #imm1` + `LDR Xt, [Xd, #imm2]` | `LDR Xt, [Xn, #(imm1+imm2)]` |
//! | `ADD Xd, Xn, #imm1` + `STR Xt, [Xd, #imm2]` | `STR Xt, [Xn, #(imm1+imm2)]` |
//!
//! # Safety Constraints
//!
//! - The ADD result must have exactly one use (the LDR/STR base operand).
//!   If the ADD result is used elsewhere, folding would break those uses.
//! - The combined offset must fit in the AArch64 unsigned immediate range
//!   for LDR/STR (0..=32760 for 64-bit loads/stores, scaled by 8).
//!   We use a conservative bound of 0..=32760.
//! - Proof annotations on the LDR/STR are preserved (unchanged).
//! - The ADD instruction is deleted after folding.
//!
//! # Offset Encoding
//!
//! AArch64 LDR/STR unsigned immediate offset is 12 bits, scaled by the
//! access size. For 64-bit (8-byte) accesses: offset = imm12 * 8,
//! giving a range of 0..=32760. We enforce this conservatively.

use std::collections::{HashMap, HashSet};

use llvm2_ir::{AArch64Opcode, InstId, MachFunction, MachOperand};

use crate::effects::inst_produces_value;
use crate::pass_manager::MachinePass;

/// Maximum unsigned immediate byte offset for 64-bit LDR/STR.
/// AArch64: 12-bit immediate scaled by 8 -> 4095 * 8 = 32760.
const MAX_UNSIGNED_OFFSET: i64 = 32760;

/// AArch64 address mode formation pass.
pub struct AddrModeFormation;

impl MachinePass for AddrModeFormation {
    fn name(&self) -> &str {
        "addr-mode"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        let mut changed = false;

        // Step 1: Count uses of each VReg as a source operand.
        let use_counts = count_vreg_uses(func);

        // Step 2: Build a map from VReg ID -> InstId for AddRI definitions.
        let add_defs = collect_add_ri_defs(func);

        // Step 3: Scan for LDR/STR instructions whose base operand is
        // defined by an AddRI with a single use, and fold.
        let mut to_delete: HashSet<InstId> = HashSet::new();

        for block_id in func.block_order.clone() {
            let block = func.block(block_id);
            for &inst_id in block.insts.clone().iter() {
                let inst = func.inst(inst_id);

                // Only handle LdrRI and StrRI.
                let (base_idx, offset_idx) = match inst.opcode {
                    // LdrRI: [dst, base, offset] — base is operand[1]
                    AArch64Opcode::LdrRI => (1, 2),
                    // StrRI: [src, base, offset] — base is operand[1]
                    AArch64Opcode::StrRI => (1, 2),
                    _ => continue,
                };

                if inst.operands.len() <= offset_idx {
                    continue;
                }

                // Get the base VReg.
                let base_vreg = match inst.operands[base_idx].as_vreg() {
                    Some(v) => v,
                    None => continue,
                };

                // Get the current offset immediate.
                let mem_offset = match inst.operands[offset_idx].as_imm() {
                    Some(v) => v,
                    None => continue,
                };

                // Look up if the base is defined by an AddRI.
                let add_inst_id = match add_defs.get(&base_vreg.id) {
                    Some(&id) => id,
                    None => continue,
                };

                // Already scheduled for deletion (avoid double-fold).
                if to_delete.contains(&add_inst_id) {
                    continue;
                }

                // Check the ADD result has exactly 1 use.
                let count = use_counts.get(&base_vreg.id).copied().unwrap_or(0);
                if count != 1 {
                    continue;
                }

                // Read the ADD instruction: AddRI [dst, src, imm]
                let add_inst = func.inst(add_inst_id);
                if add_inst.operands.len() < 3 {
                    continue;
                }

                let add_base = add_inst.operands[1].clone();
                let add_offset = match add_inst.operands[2].as_imm() {
                    Some(v) => v,
                    None => continue,
                };

                // Compute combined offset.
                let combined_offset = add_offset + mem_offset;

                // Validate range: must be non-negative and fit in unsigned
                // 12-bit scaled immediate.
                if combined_offset < 0 || combined_offset > MAX_UNSIGNED_OFFSET {
                    continue;
                }

                // Rewrite the LDR/STR: replace base with the ADD's source,
                // replace offset with combined offset.
                let load_store = func.inst_mut(inst_id);
                load_store.operands[base_idx] = add_base;
                load_store.operands[offset_idx] = MachOperand::Imm(combined_offset);

                // Mark the ADD for deletion.
                to_delete.insert(add_inst_id);
                changed = true;
            }
        }

        // Step 4: Delete folded ADD instructions from blocks.
        if !to_delete.is_empty() {
            for block_id in func.block_order.clone() {
                let block = func.block_mut(block_id);
                block.insts.retain(|id| !to_delete.contains(id));
            }
        }

        changed
    }
}

/// Count how many times each VReg ID appears as a source (use) operand
/// across the entire function.
///
/// Convention: for value-producing instructions, operand[0] is the def;
/// operands[1..] are uses. For non-value-producing instructions (stores,
/// compares, branches), all operands are uses.
fn count_vreg_uses(func: &MachFunction) -> HashMap<u32, u32> {
    let mut counts: HashMap<u32, u32> = HashMap::new();

    for block_id in &func.block_order {
        let block = func.block(*block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            let use_start = if inst_produces_value(inst) { 1 } else { 0 };

            for operand in &inst.operands[use_start..] {
                if let MachOperand::VReg(vreg) = operand {
                    *counts.entry(vreg.id).or_insert(0) += 1;
                }
            }
        }
    }

    counts
}

/// Collect a map from VReg ID (def) -> InstId for all AddRI instructions.
///
/// Only records AddRI instructions since those are the ones we can fold
/// into addressing modes.
fn collect_add_ri_defs(func: &MachFunction) -> HashMap<u32, InstId> {
    let mut defs: HashMap<u32, InstId> = HashMap::new();

    for block_id in &func.block_order {
        let block = func.block(*block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            if inst.opcode == AArch64Opcode::AddRI {
                if let Some(vreg) = inst.operands.first().and_then(|op| op.as_vreg()) {
                    defs.insert(vreg.id, inst_id);
                }
            }
        }
    }

    defs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass_manager::MachinePass;
    use llvm2_ir::{
        AArch64Opcode, InstId, MachFunction, MachInst, MachOperand, ProofAnnotation,
        RegClass, Signature, VReg,
    };

    fn vreg(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
    }

    fn imm(val: i64) -> MachOperand {
        MachOperand::Imm(val)
    }

    fn make_func_with_insts(insts: Vec<MachInst>) -> MachFunction {
        let mut func = MachFunction::new(
            "test_addr_mode".to_string(),
            Signature::new(vec![], vec![]),
        );
        let block = func.entry;
        for inst in insts {
            let id = func.push_inst(inst);
            func.append_inst(block, id);
        }
        func
    }

    // ---- Basic folding tests ----

    #[test]
    fn test_fold_add_imm_into_ldr() {
        // ADD v1, v0, #16
        // LDR v2, [v1, #0]
        // RET
        // -> LDR v2, [v0, #16], ADD deleted
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(16)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        // ADD should be deleted, leaving LDR + RET
        assert_eq!(block.insts.len(), 2);

        // LDR should now use v0 as base with offset 16
        let ldr_inst = func.inst(InstId(1));
        assert_eq!(ldr_inst.opcode, AArch64Opcode::LdrRI);
        assert_eq!(ldr_inst.operands[0], vreg(2)); // dst unchanged
        assert_eq!(ldr_inst.operands[1], vreg(0)); // base = ADD's source
        assert_eq!(ldr_inst.operands[2], imm(16)); // offset = ADD's imm
    }

    #[test]
    fn test_fold_add_imm_into_str() {
        // ADD v1, v0, #8
        // STR v2, [v1, #0]
        // RET
        // -> STR v2, [v0, #8], ADD deleted
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(8)]);
        let str_inst = MachInst::new(AArch64Opcode::StrRI, vec![vreg(2), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, str_inst, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // STR + RET

        let str_result = func.inst(InstId(1));
        assert_eq!(str_result.opcode, AArch64Opcode::StrRI);
        assert_eq!(str_result.operands[0], vreg(2)); // src unchanged
        assert_eq!(str_result.operands[1], vreg(0)); // base = ADD's source
        assert_eq!(str_result.operands[2], imm(8));  // offset = ADD's imm
    }

    #[test]
    fn test_fold_combined_offsets() {
        // ADD v1, v0, #8
        // LDR v2, [v1, #16]
        // RET
        // -> LDR v2, [v0, #24]
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(8)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(16)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2);

        let ldr_inst = func.inst(InstId(1));
        assert_eq!(ldr_inst.operands[1], vreg(0));
        assert_eq!(ldr_inst.operands[2], imm(24)); // 8 + 16
    }

    // ---- Safety: no fold when multiple uses ----

    #[test]
    fn test_no_fold_multiple_uses() {
        // ADD v1, v0, #16
        // LDR v2, [v1, #0]
        // ADD v3, v1, #4    <- second use of v1
        // RET
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(16)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)]);
        let add2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(3), vreg(1), imm(4)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, add2, ret]);

        let mut pass = AddrModeFormation;
        assert!(!pass.run(&mut func));

        // All instructions preserved
        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 4);
    }

    // ---- Safety: offset range validation ----

    #[test]
    fn test_no_fold_offset_out_of_range() {
        // ADD v1, v0, #32000
        // LDR v2, [v1, #1000]
        // RET
        // Combined = 33000, exceeds 32760
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(32000)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(1000)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(!pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 3); // unchanged
    }

    #[test]
    fn test_no_fold_negative_offset() {
        // ADD v1, v0, #-8
        // LDR v2, [v1, #0]
        // RET
        // Combined offset = -8, negative -> reject
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(-8)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(!pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 3);
    }

    // ---- Non-ADD definitions ----

    #[test]
    fn test_no_fold_non_add_def() {
        // MOV v1, v0
        // LDR v2, [v1, #0]
        // RET
        // v1 defined by MOV, not ADD -> no fold
        let mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(1), vreg(0)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![mov, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(!pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 3);
    }

    // ---- Idempotency ----

    #[test]
    fn test_idempotent() {
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(16)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));  // First run: folds
        assert!(!pass.run(&mut func)); // Second run: nothing to do
    }

    // ---- Proof annotation preservation ----

    #[test]
    fn test_preserves_ldr_proof() {
        // ADD v1, v0, #16
        // LDR v2, [v1, #0] [InBounds]
        // RET
        // -> LDR v2, [v0, #16] [InBounds] — proof preserved
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(16)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)])
            .with_proof(ProofAnnotation::InBounds);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        let ldr_inst = func.inst(InstId(1));
        assert_eq!(ldr_inst.proof, Some(ProofAnnotation::InBounds));
    }

    // ---- Edge cases ----

    #[test]
    fn test_max_valid_offset() {
        // ADD v1, v0, #32760 (max valid)
        // LDR v2, [v1, #0]
        // RET
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(32760)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        let ldr_inst = func.inst(InstId(1));
        assert_eq!(ldr_inst.operands[2], imm(32760));
    }

    #[test]
    fn test_just_over_max_offset() {
        // ADD v1, v0, #32761 (one over max)
        // LDR v2, [v1, #0]
        // RET
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(32761)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(!pass.run(&mut func));
    }

    #[test]
    fn test_zero_offset_add() {
        // ADD v1, v0, #0 + LDR v2, [v1, #8] -> LDR v2, [v0, #8]
        // (Peephole would normally remove this ADD, but if it reaches
        // addr-mode first, we can still fold it.)
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(0)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(8)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ldr, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        let ldr_inst = func.inst(InstId(1));
        assert_eq!(ldr_inst.operands[1], vreg(0));
        assert_eq!(ldr_inst.operands[2], imm(8));
    }

    #[test]
    fn test_multiple_folds_in_one_block() {
        // ADD v1, v0, #8
        // LDR v2, [v1, #0]   <- fold this
        // ADD v4, v3, #16
        // STR v5, [v4, #0]   <- fold this
        // RET
        let add1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(8)]);
        let ldr = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(1), imm(0)]);
        let add2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(4), vreg(3), imm(16)]);
        let str_inst = MachInst::new(AArch64Opcode::StrRI, vec![vreg(5), vreg(4), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add1, ldr, add2, str_inst, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        // Both ADDs deleted: LDR + STR + RET = 3
        assert_eq!(block.insts.len(), 3);

        let ldr_inst = func.inst(InstId(1));
        assert_eq!(ldr_inst.operands[1], vreg(0));
        assert_eq!(ldr_inst.operands[2], imm(8));

        let str_inst = func.inst(InstId(3));
        assert_eq!(str_inst.operands[1], vreg(3));
        assert_eq!(str_inst.operands[2], imm(16));
    }

    #[test]
    fn test_no_change_empty_function() {
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![ret]);

        let mut pass = AddrModeFormation;
        assert!(!pass.run(&mut func));
    }

    #[test]
    fn test_str_base_is_use_not_def() {
        // Ensure that for STR [src, base, offset], the base (operand[1])
        // is correctly identified as a use, not a def.
        // ADD v1, v0, #8    (v1 is used once by STR as base)
        // STR v2, [v1, #0]  (v1 is base at index 1, v2 is src at index 0)
        // RET
        //
        // v1 use count: used by STR operand[1] = 1 use
        // v2 use count: used by STR operand[0] = 1 use (STR has no def, all are uses)
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(8)]);
        let str_inst = MachInst::new(AArch64Opcode::StrRI, vec![vreg(2), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, str_inst, ret]);

        let mut pass = AddrModeFormation;
        assert!(pass.run(&mut func));

        // ADD deleted, STR rewritten
        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2);

        let str_result = func.inst(InstId(1));
        assert_eq!(str_result.operands[1], vreg(0));
        assert_eq!(str_result.operands[2], imm(8));
    }
}
