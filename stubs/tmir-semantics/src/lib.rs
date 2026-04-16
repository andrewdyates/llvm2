// tmir-semantics stub — minimal tMIR instruction semantics for LLVM2 verification
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// STATUS: Development stub. NOT currently imported by any LLVM2 crate.
//
// This stub defines the trait interface (`InstrSemantics`) that the real tMIR
// semantics crate (ayates_dbx/tMIR) will implement. It exists to document the
// expected API contract for future integration.
//
// Currently, LLVM2's verification pipeline uses its own tMIR semantic encoding
// in `crates/llvm2-verify/src/tmir_semantics.rs`, which encodes tMIR opcodes
// directly as `SmtExpr` bitvector formulas using llvm2-lower's `Opcode` enum.
// That approach works because LLVM2 has its own tMIR adapter layer (llvm2-lower)
// that defines the opcodes and types internally.
//
// When the real tMIR repo is integrated:
//   1. Replace this stub with the real tmir-semantics crate
//   2. Have llvm2-verify depend on it for the `InstrSemantics` trait
//   3. Bridge between `InstrSemantics::eval` and the existing `SmtExpr`-based
//      proof infrastructure in llvm2-verify
//
// See also:
//   - crates/llvm2-verify/src/tmir_semantics.rs (current verification encoder)
//   - designs/2026-04-13-verification-architecture.md (architecture overview)
//   - designs/2026-04-13-tmir-integration.md (integration plan)

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use tmir_instrs::Instr;
use tmir_types::Ty;

/// A bitvector value used in semantic formulas.
///
/// Represents a concrete or symbolic bitvector for SMT encoding.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BitVec {
    /// Concrete value with width.
    Concrete { width: u16, value: i128 },
    /// Symbolic variable (to be created in SMT solver).
    Symbolic { width: u16, name: String },
}

impl BitVec {
    pub fn width(&self) -> u16 {
        match self {
            BitVec::Concrete { width, .. } | BitVec::Symbolic { width, .. } => *width,
        }
    }
}

/// A semantic value: bitvector, float, bool, or undefined.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SemValue {
    Int(BitVec),
    Float { width: u16, value: f64 },
    Bool(bool),
    /// Poison / undefined behavior marker.
    Undef,
}

/// Trait for encoding tMIR instruction semantics as SMT-checkable formulas.
///
/// LLVM2's verification pipeline uses this trait to:
/// 1. Encode the tMIR instruction's semantics
/// 2. Encode the machine instruction's semantics
/// 3. Prove equivalence via SMT solver (z4)
pub trait InstrSemantics {
    /// Encode the semantics of a tMIR instruction.
    ///
    /// Given input values, compute the output value(s) according to
    /// the instruction's formal semantics.
    fn eval(&self, instr: &Instr, inputs: &[SemValue]) -> Vec<SemValue>;

    /// Returns the result type(s) of an instruction given input types.
    fn result_types(&self, instr: &Instr, input_types: &[Ty]) -> Vec<Ty>;

    /// Check if an instruction can produce undefined behavior for given inputs.
    /// Returns a human-readable description of the UB condition, if any.
    fn ub_condition(&self, instr: &Instr) -> Option<String>;
}

/// Default semantics implementation (concrete evaluation).
///
/// This evaluates tMIR instructions on concrete values. For SMT-based
/// verification, LLVM2 will use a symbolic implementation that produces
/// z4 formulas instead.
pub struct ConcreteSemantics;

impl InstrSemantics for ConcreteSemantics {
    fn eval(&self, instr: &Instr, inputs: &[SemValue]) -> Vec<SemValue> {
        match instr {
            Instr::BinOp { op, .. } => {
                if inputs.len() < 2 {
                    return vec![SemValue::Undef];
                }
                // Stub: return first input as placeholder
                // Real implementation performs the arithmetic
                let _ = op;
                vec![inputs[0].clone()]
            }
            Instr::UnOp { op, .. } => {
                if inputs.is_empty() {
                    return vec![SemValue::Undef];
                }
                let _ = op;
                vec![inputs[0].clone()]
            }
            Instr::Cmp { .. } => {
                vec![SemValue::Bool(false)]
            }
            Instr::Const { value, .. } => {
                vec![SemValue::Int(BitVec::Concrete { width: 64, value: *value as i128 })]
            }
            Instr::FConst { value, ty } => {
                let width = ty.bits().unwrap_or(64);
                vec![SemValue::Float { width, value: *value }]
            }
            Instr::Select { .. } => {
                // Select: result = cond ? true_val : false_val
                if inputs.len() < 3 {
                    return vec![SemValue::Undef];
                }
                match &inputs[0] {
                    SemValue::Bool(true) => vec![inputs[1].clone()],
                    SemValue::Bool(false) => vec![inputs[2].clone()],
                    _ => vec![SemValue::Undef],
                }
            }
            Instr::GetElementPtr { .. } => {
                // GEP: result is a pointer (address). Stub returns Undef.
                vec![SemValue::Undef]
            }
            Instr::Nop => vec![],
            // Most instructions need full implementation for real verification.
            // Return Undef as placeholder for the stub.
            _ => vec![SemValue::Undef],
        }
    }

    fn result_types(&self, instr: &Instr, _input_types: &[Ty]) -> Vec<Ty> {
        match instr {
            Instr::BinOp { ty, .. }
            | Instr::UnOp { ty, .. }
            | Instr::Load { ty, .. }
            | Instr::Alloc { ty, .. }
            | Instr::Borrow { ty, .. }
            | Instr::BorrowMut { ty, .. }
            | Instr::Struct { ty, .. }
            | Instr::Field { ty, .. }
            | Instr::Index { ty, .. }
            | Instr::Phi { ty, .. }
            | Instr::Const { ty, .. }
            | Instr::FConst { ty, .. } => vec![ty.clone()],
            Instr::Select { ty, .. } => vec![ty.clone()],
            Instr::GetElementPtr { .. } => vec![Ty::ptr(Ty::void())],
            Instr::Cmp { .. } | Instr::IsUnique { .. } => vec![Ty::bool_ty()],
            Instr::Cast { dst_ty, .. } => vec![dst_ty.clone()],
            Instr::Call { ret_ty, .. } | Instr::CallIndirect { ret_ty, .. } => ret_ty.clone(),
            // Void instructions
            Instr::Store { .. }
            | Instr::Dealloc { .. }
            | Instr::EndBorrow { .. }
            | Instr::Retain { .. }
            | Instr::Release { .. }
            | Instr::Br { .. }
            | Instr::CondBr { .. }
            | Instr::Switch { .. }
            | Instr::Return { .. }
            | Instr::Nop => vec![],
        }
    }

    fn ub_condition(&self, instr: &Instr) -> Option<String> {
        match instr {
            Instr::BinOp { op: tmir_instrs::BinOp::SDiv | tmir_instrs::BinOp::UDiv, .. } => {
                Some("Division by zero".to_string())
            }
            Instr::BinOp { op: tmir_instrs::BinOp::SRem | tmir_instrs::BinOp::URem, .. } => {
                Some("Remainder by zero".to_string())
            }
            _ => None,
        }
    }
}
