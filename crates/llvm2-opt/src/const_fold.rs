// llvm2-opt - Constant Folding
//
// Author: Andrew Yates <ayates@dropbox.com>
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
//! 1. Build a map from VReg ID → known constant value (seeded by MovI instructions).
//! 2. Scan instructions forward. For each foldable instruction where all
//!    source operands are known constants, compute the result and replace
//!    the instruction with `MovI dst, #result`.
//! 3. Update the constant map with the newly computed value.
//!
//! Division is NOT folded to avoid division-by-zero at compile time.

use std::collections::HashMap;

use llvm2_ir::{AArch64Opcode, MachFunction, MachInst, MachOperand, VReg};

use crate::pass_manager::MachinePass;

/// Constant folding pass.
pub struct ConstantFolding;

impl MachinePass for ConstantFolding {
    fn name(&self) -> &str {
        "const-fold"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        let mut changed = false;

        // Map from VReg ID to known constant value.
        let mut constants: HashMap<u32, i64> = HashMap::new();

        // First pass: seed constants from MovI instructions.
        for block_id in &func.block_order {
            let block = func.block(*block_id);
            for &inst_id in &block.insts {
                let inst = func.inst(inst_id);
                if inst.opcode == AArch64Opcode::MovI {
                    if let (Some(MachOperand::VReg(dst)), Some(MachOperand::Imm(val))) =
                        (inst.operands.first(), inst.operands.get(1))
                    {
                        constants.insert(dst.id, *val);
                    }
                }
            }
        }

        // Second pass: fold instructions with constant operands.
        for block_id in func.block_order.clone() {
            let block = func.block(block_id);
            for &inst_id in block.insts.clone().iter() {
                let inst = func.inst(inst_id);

                if let Some((dst_vreg, result)) = try_fold(inst, &constants) {
                    // Replace instruction with MovI dst, #result
                    let new_inst = MachInst::new(
                        AArch64Opcode::MovI,
                        vec![
                            MachOperand::VReg(dst_vreg),
                            MachOperand::Imm(result),
                        ],
                    );
                    *func.inst_mut(inst_id) = new_inst;
                    constants.insert(dst_vreg.id, result);
                    changed = true;
                }
            }
        }

        changed
    }
}

/// Try to constant-fold an instruction.
///
/// Returns `Some((dst_vreg, folded_value))` if the instruction can be folded,
/// `None` otherwise.
fn try_fold(
    inst: &MachInst,
    constants: &HashMap<u32, i64>,
) -> Option<(VReg, i64)> {
    let opcode = inst.opcode;
    let ops = &inst.operands;

    match opcode {
        // Binary register-register: dst = op(lhs, rhs)
        // Operands: [dst, lhs, rhs]
        AArch64Opcode::AddRR
        | AArch64Opcode::SubRR
        | AArch64Opcode::MulRR
        | AArch64Opcode::AndRR
        | AArch64Opcode::OrrRR
        | AArch64Opcode::EorRR => {
            if ops.len() < 3 {
                return None;
            }
            let dst = ops[0].as_vreg()?;
            let lhs = lookup_const(&ops[1], constants)?;
            let rhs = lookup_const(&ops[2], constants)?;

            let result = match opcode {
                AArch64Opcode::AddRR => lhs.wrapping_add(rhs),
                AArch64Opcode::SubRR => lhs.wrapping_sub(rhs),
                AArch64Opcode::MulRR => lhs.wrapping_mul(rhs),
                AArch64Opcode::AndRR => lhs & rhs,
                AArch64Opcode::OrrRR => lhs | rhs,
                AArch64Opcode::EorRR => lhs ^ rhs,
                // SAFETY: inner match is constrained by outer arm to these 6 opcodes
                _ => unreachable!("inner match constrained by outer arm"),
            };
            Some((dst, result))
        }

        // Binary register-immediate: dst = op(src, imm)
        // Operands: [dst, src, imm]
        AArch64Opcode::AddRI | AArch64Opcode::SubRI => {
            if ops.len() < 3 {
                return None;
            }
            let dst = ops[0].as_vreg()?;
            let src = lookup_const(&ops[1], constants)?;
            let imm = ops[2].as_imm()?;

            let result = match opcode {
                AArch64Opcode::AddRI => src.wrapping_add(imm),
                AArch64Opcode::SubRI => src.wrapping_sub(imm),
                // SAFETY: inner match is constrained by outer arm to AddRI|SubRI
                _ => unreachable!("inner match constrained by outer arm"),
            };
            Some((dst, result))
        }

        // Shift register-immediate: dst = src shift imm
        // Operands: [dst, src, imm]
        AArch64Opcode::LslRI | AArch64Opcode::LsrRI | AArch64Opcode::AsrRI => {
            if ops.len() < 3 {
                return None;
            }
            let dst = ops[0].as_vreg()?;
            let src = lookup_const(&ops[1], constants)?;
            let shift = ops[2].as_imm()?;

            // Validate shift amount (0..63 for 64-bit).
            if shift < 0 || shift > 63 {
                return None;
            }
            let shift = shift as u32;

            let result = match opcode {
                AArch64Opcode::LslRI => src.wrapping_shl(shift),
                AArch64Opcode::LsrRI => ((src as u64).wrapping_shr(shift)) as i64,
                AArch64Opcode::AsrRI => src.wrapping_shr(shift),
                // SAFETY: inner match is constrained by outer arm to LslRI|LsrRI|AsrRI
                _ => unreachable!("inner match constrained by outer arm"),
            };
            Some((dst, result))
        }

        // Shift register-register: dst = src shift amount
        // Operands: [dst, src, amount]
        AArch64Opcode::LslRR | AArch64Opcode::LsrRR | AArch64Opcode::AsrRR => {
            if ops.len() < 3 {
                return None;
            }
            let dst = ops[0].as_vreg()?;
            let src = lookup_const(&ops[1], constants)?;
            let amount = lookup_const(&ops[2], constants)?;

            // AArch64 masks shift amount to 0..63.
            let shift = (amount & 63) as u32;

            let result = match opcode {
                AArch64Opcode::LslRR => src.wrapping_shl(shift),
                AArch64Opcode::LsrRR => ((src as u64).wrapping_shr(shift)) as i64,
                AArch64Opcode::AsrRR => src.wrapping_shr(shift),
                // SAFETY: inner match is constrained by outer arm to LslRR|LsrRR|AsrRR
                _ => unreachable!("inner match constrained by outer arm"),
            };
            Some((dst, result))
        }

        // Unary: dst = -src
        // Operands: [dst, src]
        AArch64Opcode::Neg => {
            if ops.len() < 2 {
                return None;
            }
            let dst = ops[0].as_vreg()?;
            let src = lookup_const(&ops[1], constants)?;
            Some((dst, src.wrapping_neg()))
        }

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

        let folded = func.inst(llvm2_ir::InstId(1));
        assert_eq!(folded.operands[1], imm(-42));
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
        let m0 = MachInst::new(AArch64Opcode::MovI, vec![vreg(0), imm(i64::MAX)]);
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![m0, add, ret]);

        let mut cf = ConstantFolding;
        assert!(cf.run(&mut func));

        let folded = func.inst(llvm2_ir::InstId(1));
        assert_eq!(folded.operands[1], imm(i64::MIN));
    }
}
