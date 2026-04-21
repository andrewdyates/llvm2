// llvm2-dialect - machir.* -> llvm2_ir::MachFunction adapter
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Translate a fully-lowered `machir.*` [`DialectFunction`] into a
//! `llvm2_ir::MachFunction`. This is the v1 bridge that lets the dialect
//! framework reach the existing LLVM2 codegen pipeline without modifying the
//! pipeline itself.
//!
//! Only ops from the `machir` dialect are accepted. Encountering any other
//! dialect is an [`EmitError::UnexpectedDialect`]. Variants not yet
//! implemented by [`dialects::machir::to_aarch64`] raise [`EmitError::Unsupported`].

use std::collections::HashMap;

use llvm2_ir::regs::RegClass;
use llvm2_ir::{
    AArch64Opcode, BlockId as MachBlockId, InstId, MachFunction, MachInst, MachOperand,
    Signature, VReg,
};

use crate::dialects::machir::{self, MACHIR_ADD_RR, MACHIR_EOR_RR, MACHIR_MOVZ_I64, MACHIR_RET};
use crate::id::{DialectId, ValueId};
use crate::module::DialectModule;
use crate::op::Attribute;

/// Error returned by [`emit_mach_function`].
#[derive(Debug, Clone)]
pub enum EmitError {
    /// Encountered an op belonging to a dialect other than `machir`.
    UnexpectedDialect {
        expected: DialectId,
        got: DialectId,
        op_name: String,
    },
    /// `machir` op has no corresponding [`AArch64Opcode`] yet.
    Unsupported(String),
    /// Op has an unexpected shape (wrong operand count / missing attribute).
    Malformed(String),
    /// Referenced a SSA value that no preceding op defined.
    UndefinedValue(ValueId),
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitError::UnexpectedDialect { expected, got, op_name } => write!(
                f,
                "expected machir dialect {:?}, got {:?} for op {}",
                expected, got, op_name
            ),
            EmitError::Unsupported(m) => write!(f, "unsupported machir op: {}", m),
            EmitError::Malformed(m) => write!(f, "malformed machir op: {}", m),
            EmitError::UndefinedValue(v) => write!(f, "undefined SSA value {:?}", v),
        }
    }
}

impl std::error::Error for EmitError {}

/// Translate a fully-lowered `machir.*` [`DialectFunction`] into a
/// `MachFunction`. The function's parameters become the first N VReg
/// allocations (so parameter ValueIds map to vreg ids 0..N-1), and each op is
/// translated in program order and appended to the entry block.
pub fn emit_mach_function(
    module: &DialectModule,
    func_index: usize,
) -> Result<MachFunction, EmitError> {
    let machir_id = module
        .registry
        .by_name("machir")
        .ok_or_else(|| EmitError::Malformed("machir dialect is not registered".to_string()))?;
    let func = &module.functions[func_index];

    let sig = Signature::new(
        func.params.iter().map(|(_, t)| t.clone()).collect(),
        func.results.clone(),
    );
    let mut mf = MachFunction::new(func.name.clone(), sig);

    // Parameter value -> VReg mapping (vreg ids 0..N-1, classed by Type).
    let mut value_to_vreg: HashMap<ValueId, VReg> = HashMap::new();
    for (src, ty) in &func.params {
        let id = mf.alloc_vreg();
        let class = RegClass::for_type(ty);
        value_to_vreg.insert(*src, VReg::new(id, class));
    }

    let entry: MachBlockId = mf.entry;

    for op in func.iter_ops() {
        if op.op.dialect != machir_id {
            let name = module
                .resolve(op.op)
                .map(|d| d.name.to_string())
                .unwrap_or_else(|| format!("<unregistered op {:?}>", op.op));
            return Err(EmitError::UnexpectedDialect {
                expected: machir_id,
                got: op.op.dialect,
                op_name: name,
            });
        }

        match op.op.op {
            MACHIR_MOVZ_I64 => emit_movz(op, &mut mf, &mut value_to_vreg, entry)?,
            MACHIR_ADD_RR => {
                emit_binary(op, &mut mf, &mut value_to_vreg, entry, AArch64Opcode::AddRR)?
            }
            MACHIR_EOR_RR => {
                emit_binary(op, &mut mf, &mut value_to_vreg, entry, AArch64Opcode::EorRR)?
            }
            MACHIR_RET => emit_ret(op, &mut mf, &value_to_vreg, entry)?,
            other => {
                // Fall back to the generic mapping so newly-added machir ops
                // without dedicated emitters at least produce *some* MachInst.
                if let Some(opcode) = machir::to_aarch64(other) {
                    return Err(EmitError::Unsupported(format!(
                        "no dedicated emitter for machir op {:?} ({:?})",
                        other, opcode
                    )));
                }
                return Err(EmitError::Unsupported(format!(
                    "unknown machir opcode {:?}",
                    other
                )));
            }
        }
    }

    Ok(mf)
}

fn emit_movz(
    op: &crate::op::DialectOp,
    mf: &mut MachFunction,
    value_to_vreg: &mut HashMap<ValueId, VReg>,
    block: MachBlockId,
) -> Result<(), EmitError> {
    if op.results.len() != 1 || !op.operands.is_empty() {
        return Err(EmitError::Malformed(
            "machir.movz.i64 expects 0 operands and 1 result".to_string(),
        ));
    }
    let (src_result, ty) = op.results[0].clone();
    let imm = op
        .attr("value")
        .and_then(|a| match a {
            Attribute::U64(v) => Some(*v as i64),
            Attribute::I64(v) => Some(*v),
            _ => None,
        })
        .ok_or_else(|| {
            EmitError::Malformed(
                "machir.movz.i64 missing i64/u64 `value` attribute".to_string(),
            )
        })?;

    let dst = mf.alloc_vreg();
    let dst_reg = VReg::new(dst, RegClass::for_type(&ty));
    value_to_vreg.insert(src_result, dst_reg);

    let inst = MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::VReg(dst_reg), MachOperand::Imm(imm)],
    );
    let id = mf.push_inst(inst);
    append(mf, block, id);
    Ok(())
}

fn emit_binary(
    op: &crate::op::DialectOp,
    mf: &mut MachFunction,
    value_to_vreg: &mut HashMap<ValueId, VReg>,
    block: MachBlockId,
    opcode: AArch64Opcode,
) -> Result<(), EmitError> {
    if op.operands.len() != 2 || op.results.len() != 1 {
        return Err(EmitError::Malformed(format!(
            "{:?} expects 2 operands and 1 result",
            opcode
        )));
    }
    let a = *value_to_vreg
        .get(&op.operands[0])
        .ok_or(EmitError::UndefinedValue(op.operands[0]))?;
    let b = *value_to_vreg
        .get(&op.operands[1])
        .ok_or(EmitError::UndefinedValue(op.operands[1]))?;
    let (src_result, ty) = op.results[0].clone();
    let dst_id = mf.alloc_vreg();
    let dst_reg = VReg::new(dst_id, RegClass::for_type(&ty));
    value_to_vreg.insert(src_result, dst_reg);

    let inst = MachInst::new(
        opcode,
        vec![
            MachOperand::VReg(dst_reg),
            MachOperand::VReg(a),
            MachOperand::VReg(b),
        ],
    );
    let id = mf.push_inst(inst);
    append(mf, block, id);
    Ok(())
}

fn emit_ret(
    op: &crate::op::DialectOp,
    mf: &mut MachFunction,
    value_to_vreg: &HashMap<ValueId, VReg>,
    block: MachBlockId,
) -> Result<(), EmitError> {
    if op.operands.len() > 1 {
        return Err(EmitError::Malformed(
            "machir.ret expects 0 or 1 operands".to_string(),
        ));
    }
    // Return value, if present, is modeled as an implicit use on Ret. We still
    // encode it as an explicit operand for verifiability; real lowering would
    // move the value into X0/W0 via a preceding Mov.
    let mut operands = Vec::new();
    if let Some(val) = op.operands.first() {
        let vreg = *value_to_vreg
            .get(val)
            .ok_or(EmitError::UndefinedValue(*val))?;
        operands.push(MachOperand::VReg(vreg));
    }
    let inst = MachInst::new(AArch64Opcode::Ret, operands);
    let id = mf.push_inst(inst);
    append(mf, block, id);
    Ok(())
}

fn append(mf: &mut MachFunction, block: MachBlockId, inst_id: InstId) {
    mf.append_inst(block, inst_id);
}
