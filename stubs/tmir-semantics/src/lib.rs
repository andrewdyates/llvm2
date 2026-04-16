// tmir-semantics — Concrete tMIR instruction semantics evaluator
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// STATUS: Development stub with concrete evaluation. Will be replaced by the
// real tMIR semantics crate (ayates_dbx/tMIR) when integrated (#283).
//
// This crate provides TWO things:
//   1. `InstrSemantics` trait — the API contract for tMIR instruction semantics
//   2. `ConcreteSemantics` — a concrete evaluator that computes actual results
//
// There are two complementary semantic implementations in LLVM2:
//
//   - **This crate** (`tmir-semantics`): Concrete evaluator. Given concrete input
//     values, computes the concrete output. Used for testing, interpretation, and
//     as a reference implementation.
//
//   - **`crates/llvm2-verify/src/tmir_semantics.rs`**: SMT encoder. Encodes tMIR
//     opcodes as `SmtExpr` bitvector formulas for SMT-based equivalence proofs.
//     Uses llvm2-lower's `Opcode` enum (not tmir-instrs directly).
//
// Both will be replaced when the real tMIR semantics crate arrives. Until then,
// they serve different purposes: this one evaluates, that one encodes for proofs.
//
// See also:
//   - crates/llvm2-verify/src/tmir_semantics.rs (SMT encoder)
//   - designs/2026-04-13-verification-architecture.md (architecture overview)
//   - designs/2026-04-13-tmir-integration.md (integration plan)

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use tmir_instrs::{BinOp, CastOp, CmpOp, Instr, UnOp};
use tmir_types::Ty;

// ---------------------------------------------------------------------------
// Bitvector arithmetic helpers
// ---------------------------------------------------------------------------

/// Mask an i128 value to the given bit width. For width=128, returns the value
/// unchanged (all 128 bits are valid). For width=0, returns 0.
fn mask_to_width(val: i128, width: u16) -> i128 {
    if width == 0 {
        return 0;
    }
    if width >= 128 {
        return val;
    }
    val & ((1i128 << width) - 1)
}

/// Interpret an unsigned i128 (masked to `width` bits) as a signed value.
/// E.g., for width=32: 0xFFFF_FFFF -> -1.
fn to_signed(val: i128, width: u16) -> i128 {
    if width == 0 {
        return 0;
    }
    let masked = mask_to_width(val, width);
    if width >= 128 {
        return masked;
    }
    let sign_bit = 1i128 << (width - 1);
    if masked & sign_bit != 0 {
        // Sign-extend: fill upper bits with 1s
        masked | !((1i128 << width) - 1)
    } else {
        masked
    }
}

/// Sign-extend a value from `from_width` bits to full i128, then mask to
/// `to_width` bits.
fn sign_extend(val: i128, from_width: u16, to_width: u16) -> i128 {
    let signed = to_signed(val, from_width);
    mask_to_width(signed, to_width)
}

/// Extract two concrete integer values and their shared width from inputs.
/// Returns None if inputs are not two concrete integers of the same width.
fn extract_int_pair(inputs: &[SemValue]) -> Option<(i128, i128, u16)> {
    if inputs.len() < 2 {
        return None;
    }
    match (&inputs[0], &inputs[1]) {
        (
            SemValue::Int(BitVec::Concrete { width: w1, value: v1 }),
            SemValue::Int(BitVec::Concrete { width: w2, value: v2 }),
        ) => {
            // Use the larger width if they differ (should be equal in practice)
            let w = (*w1).max(*w2);
            Some((mask_to_width(*v1, w), mask_to_width(*v2, w), w))
        }
        _ => None,
    }
}

/// Extract a single concrete integer value and its width from an input.
fn extract_int(input: &SemValue) -> Option<(i128, u16)> {
    match input {
        SemValue::Int(BitVec::Concrete { width, value }) => {
            Some((mask_to_width(*value, *width), *width))
        }
        _ => None,
    }
}

/// Extract two float values from inputs.
fn extract_float_pair(inputs: &[SemValue]) -> Option<(f64, f64, u16)> {
    if inputs.len() < 2 {
        return None;
    }
    match (&inputs[0], &inputs[1]) {
        (
            SemValue::Float { width: w1, value: v1 },
            SemValue::Float { width: w2, value: v2 },
        ) => {
            let w = (*w1).max(*w2);
            Some((*v1, *v2, w))
        }
        _ => None,
    }
}

/// Create a concrete integer SemValue.
fn int_result(val: i128, width: u16) -> SemValue {
    SemValue::Int(BitVec::Concrete {
        width,
        value: mask_to_width(val, width),
    })
}

/// Create a float SemValue.
fn float_result(val: f64, width: u16) -> SemValue {
    SemValue::Float { width, value: val }
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A bitvector value used in semantic evaluation.
///
/// Represents a concrete or symbolic bitvector. Concrete values are used for
/// interpretation; symbolic values are placeholders for SMT encoding.
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

/// Trait for evaluating tMIR instruction semantics.
///
/// Implementations:
///   - `ConcreteSemantics` (this crate): concrete evaluation on actual values
///   - llvm2-verify's SMT encoder: symbolic encoding as `SmtExpr` formulas
///
/// When the real tMIR semantics crate arrives, both will be replaced.
pub trait InstrSemantics {
    /// Evaluate the semantics of a tMIR instruction.
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

/// Concrete semantics evaluator.
///
/// Evaluates tMIR instructions on concrete `SemValue` inputs, producing concrete
/// outputs. All integer arithmetic is performed in wrapping mode with results
/// masked to the operand bit width. Division/remainder by zero returns `Undef`.
///
/// This is the reference implementation for tMIR instruction behavior. The SMT
/// encoder in `llvm2-verify/src/tmir_semantics.rs` encodes the same semantics
/// as symbolic formulas — any divergence between the two is a bug.
pub struct ConcreteSemantics;

impl ConcreteSemantics {
    /// Evaluate a binary integer operation.
    fn eval_int_binop(&self, op: &BinOp, a: i128, b: i128, w: u16) -> SemValue {
        match op {
            BinOp::Add => int_result(a.wrapping_add(b), w),
            BinOp::Sub => int_result(a.wrapping_sub(b), w),
            BinOp::Mul => int_result(a.wrapping_mul(b), w),
            BinOp::SDiv => {
                if b == 0 {
                    return SemValue::Undef;
                }
                let sa = to_signed(a, w);
                let sb = to_signed(b, w);
                int_result(sa.wrapping_div(sb), w)
            }
            BinOp::UDiv => {
                if b == 0 {
                    return SemValue::Undef;
                }
                // Both operands are already masked (unsigned)
                int_result(a.wrapping_div(b), w)
            }
            BinOp::SRem => {
                if b == 0 {
                    return SemValue::Undef;
                }
                let sa = to_signed(a, w);
                let sb = to_signed(b, w);
                int_result(sa.wrapping_rem(sb), w)
            }
            BinOp::URem => {
                if b == 0 {
                    return SemValue::Undef;
                }
                int_result(a.wrapping_rem(b), w)
            }
            BinOp::And => int_result(a & b, w),
            BinOp::Or => int_result(a | b, w),
            BinOp::Xor => int_result(a ^ b, w),
            BinOp::Shl => {
                let shift = mask_to_width(b, w) as u32;
                if shift >= w as u32 {
                    int_result(0, w)
                } else {
                    int_result(a << shift, w)
                }
            }
            BinOp::LShr => {
                // Logical shift right: shift the masked (unsigned) value
                let shift = mask_to_width(b, w) as u32;
                if shift >= w as u32 {
                    int_result(0, w)
                } else {
                    int_result(a >> shift, w)
                }
            }
            BinOp::AShr => {
                // Arithmetic shift right: sign-extend, shift, re-mask
                let shift = mask_to_width(b, w) as u32;
                let signed = to_signed(a, w);
                if shift >= w as u32 {
                    // All bits become the sign bit
                    if signed < 0 {
                        int_result(mask_to_width(-1i128, w), w)
                    } else {
                        int_result(0, w)
                    }
                } else {
                    int_result(signed >> shift, w)
                }
            }
            // Float binops should not reach here
            BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv => SemValue::Undef,
        }
    }

    /// Evaluate a binary float operation.
    fn eval_float_binop(&self, op: &BinOp, a: f64, b: f64, w: u16) -> SemValue {
        match op {
            BinOp::FAdd => float_result(a + b, w),
            BinOp::FSub => float_result(a - b, w),
            BinOp::FMul => float_result(a * b, w),
            BinOp::FDiv => float_result(a / b, w),
            _ => SemValue::Undef,
        }
    }

    /// Evaluate a comparison operation on integers.
    fn eval_int_cmp(&self, op: &CmpOp, a: i128, b: i128, w: u16) -> SemValue {
        let result = match op {
            CmpOp::Eq => a == b,
            CmpOp::Ne => a != b,
            CmpOp::Slt => to_signed(a, w) < to_signed(b, w),
            CmpOp::Sle => to_signed(a, w) <= to_signed(b, w),
            CmpOp::Sgt => to_signed(a, w) > to_signed(b, w),
            CmpOp::Sge => to_signed(a, w) >= to_signed(b, w),
            CmpOp::Ult => a < b,
            CmpOp::Ule => a <= b,
            CmpOp::Ugt => a > b,
            CmpOp::Uge => a >= b,
            // Float comparisons should not reach here
            _ => return SemValue::Undef,
        };
        SemValue::Bool(result)
    }

    /// Evaluate a comparison operation on floats.
    fn eval_float_cmp(&self, op: &CmpOp, a: f64, b: f64) -> SemValue {
        let either_nan = a.is_nan() || b.is_nan();
        let result = match op {
            // Ordered comparisons: false if either is NaN
            CmpOp::FOeq => !either_nan && a == b,
            CmpOp::FOne => !either_nan && a != b,
            CmpOp::FOlt => !either_nan && a < b,
            CmpOp::FOle => !either_nan && a <= b,
            CmpOp::FOgt => !either_nan && a > b,
            CmpOp::FOge => !either_nan && a >= b,
            // Unordered comparisons: true if either is NaN
            CmpOp::FUeq => either_nan || a == b,
            CmpOp::FUne => either_nan || a != b,
            CmpOp::FUlt => either_nan || a < b,
            CmpOp::FUle => either_nan || a <= b,
            CmpOp::FUgt => either_nan || a > b,
            CmpOp::FUge => either_nan || a >= b,
            // Integer comparisons should not reach here
            _ => return SemValue::Undef,
        };
        SemValue::Bool(result)
    }
}

impl InstrSemantics for ConcreteSemantics {
    fn eval(&self, instr: &Instr, inputs: &[SemValue]) -> Vec<SemValue> {
        match instr {
            Instr::BinOp { op, ty, .. } => {
                if inputs.len() < 2 {
                    return vec![SemValue::Undef];
                }
                // Try integer first, then float
                if let Some((a, b, w)) = extract_int_pair(inputs) {
                    let width = ty.bits().unwrap_or(w);
                    let a = mask_to_width(a, width);
                    let b = mask_to_width(b, width);
                    vec![self.eval_int_binop(op, a, b, width)]
                } else if let Some((a, b, w)) = extract_float_pair(inputs) {
                    let width = ty.bits().unwrap_or(w);
                    vec![self.eval_float_binop(op, a, b, width)]
                } else {
                    vec![SemValue::Undef]
                }
            }

            Instr::UnOp { op, ty, .. } => {
                if inputs.is_empty() {
                    return vec![SemValue::Undef];
                }
                let result = match op {
                    UnOp::Neg => {
                        if let Some((val, w)) = extract_int(&inputs[0]) {
                            let width = ty.bits().unwrap_or(w);
                            // Two's complement negation: -x = (~x) + 1 = 0 - x
                            int_result(0i128.wrapping_sub(val), width)
                        } else {
                            SemValue::Undef
                        }
                    }
                    UnOp::Not => {
                        if let Some((val, w)) = extract_int(&inputs[0]) {
                            let width = ty.bits().unwrap_or(w);
                            // Bitwise NOT: flip all bits within width
                            int_result(!val, width)
                        } else {
                            SemValue::Undef
                        }
                    }
                    UnOp::FNeg => match &inputs[0] {
                        SemValue::Float { width, value } => {
                            float_result(-value, *width)
                        }
                        _ => SemValue::Undef,
                    },
                    UnOp::FAbs => match &inputs[0] {
                        SemValue::Float { width, value } => {
                            float_result(value.abs(), *width)
                        }
                        _ => SemValue::Undef,
                    },
                    UnOp::FSqrt => match &inputs[0] {
                        SemValue::Float { width, value } => {
                            float_result(value.sqrt(), *width)
                        }
                        _ => SemValue::Undef,
                    },
                };
                vec![result]
            }

            Instr::Cmp { op, ty, .. } => {
                if inputs.len() < 2 {
                    return vec![SemValue::Undef];
                }
                // Try integer comparison
                if let Some((a, b, w)) = extract_int_pair(inputs) {
                    let width = ty.bits().unwrap_or(w);
                    let a = mask_to_width(a, width);
                    let b = mask_to_width(b, width);
                    vec![self.eval_int_cmp(op, a, b, width)]
                } else if let Some((a, b, _)) = extract_float_pair(inputs) {
                    vec![self.eval_float_cmp(op, a, b)]
                } else {
                    vec![SemValue::Undef]
                }
            }

            Instr::Cast {
                op,
                src_ty,
                dst_ty,
                ..
            } => {
                if inputs.is_empty() {
                    return vec![SemValue::Undef];
                }
                let src_width = src_ty.bits().unwrap_or(64);
                let dst_width = dst_ty.bits().unwrap_or(64);

                match op {
                    CastOp::ZExt => {
                        // Zero-extend: the masked value is already zero-extended
                        if let Some((val, _)) = extract_int(&inputs[0]) {
                            vec![int_result(mask_to_width(val, src_width), dst_width)]
                        } else {
                            vec![SemValue::Undef]
                        }
                    }
                    CastOp::SExt => {
                        // Sign-extend: check sign bit, extend with 1s if set
                        if let Some((val, _)) = extract_int(&inputs[0]) {
                            vec![int_result(
                                sign_extend(val, src_width, dst_width),
                                dst_width,
                            )]
                        } else {
                            vec![SemValue::Undef]
                        }
                    }
                    CastOp::Trunc => {
                        // Truncate: mask to destination width
                        if let Some((val, _)) = extract_int(&inputs[0]) {
                            vec![int_result(val, dst_width)]
                        } else {
                            vec![SemValue::Undef]
                        }
                    }
                    CastOp::FPToSI => {
                        // Float to signed integer
                        if let SemValue::Float { value, .. } = &inputs[0] {
                            let ival = *value as i128;
                            vec![int_result(ival, dst_width)]
                        } else {
                            vec![SemValue::Undef]
                        }
                    }
                    CastOp::FPToUI => {
                        // Float to unsigned integer
                        if let SemValue::Float { value, .. } = &inputs[0] {
                            // Cast through u128 to get unsigned semantics
                            let ival = if *value < 0.0 {
                                0i128
                            } else {
                                *value as u128 as i128
                            };
                            vec![int_result(ival, dst_width)]
                        } else {
                            vec![SemValue::Undef]
                        }
                    }
                    CastOp::SIToFP => {
                        // Signed integer to float
                        if let Some((val, w)) = extract_int(&inputs[0]) {
                            let signed = to_signed(val, w);
                            vec![float_result(signed as f64, dst_width)]
                        } else {
                            vec![SemValue::Undef]
                        }
                    }
                    CastOp::UIToFP => {
                        // Unsigned integer to float
                        if let Some((val, _)) = extract_int(&inputs[0]) {
                            // val is already masked (unsigned)
                            vec![float_result(val as f64, dst_width)]
                        } else {
                            vec![SemValue::Undef]
                        }
                    }
                    CastOp::FPExt | CastOp::FPTrunc => {
                        // Float precision change: we use f64 internally for both
                        // F32 and F64, so just adjust the width tag. For FPTrunc,
                        // round through f32 to match hardware behavior.
                        if let SemValue::Float { value, .. } = &inputs[0] {
                            let out_val = if *op == CastOp::FPTrunc && dst_width == 32 {
                                (*value as f32) as f64
                            } else {
                                *value
                            };
                            vec![float_result(out_val, dst_width)]
                        } else {
                            vec![SemValue::Undef]
                        }
                    }
                    CastOp::Bitcast | CastOp::PtrToInt | CastOp::IntToPtr => {
                        // Reinterpret bits: keep the value, change the width
                        if let Some((val, _)) = extract_int(&inputs[0]) {
                            vec![int_result(val, dst_width)]
                        } else {
                            vec![inputs[0].clone()]
                        }
                    }
                }
            }

            Instr::Const { value, ty } => {
                let width = ty.bits().unwrap_or(64);
                vec![int_result(*value as i128, width)]
            }

            Instr::FConst { value, ty } => {
                let width = ty.bits().unwrap_or(64);
                vec![float_result(*value, width)]
            }

            Instr::Select { .. } => {
                if inputs.len() < 3 {
                    return vec![SemValue::Undef];
                }
                match &inputs[0] {
                    SemValue::Bool(true) => vec![inputs[1].clone()],
                    SemValue::Bool(false) => vec![inputs[2].clone()],
                    // Also support integer condition: non-zero = true
                    SemValue::Int(BitVec::Concrete { value, .. }) => {
                        if *value != 0 {
                            vec![inputs[1].clone()]
                        } else {
                            vec![inputs[2].clone()]
                        }
                    }
                    _ => vec![SemValue::Undef],
                }
            }

            Instr::GetElementPtr { .. } => {
                // GEP: would need pointer arithmetic model. Return Undef in stub.
                vec![SemValue::Undef]
            }

            Instr::Nop => vec![],

            // Memory, control flow, ownership ops: no concrete value to compute
            // without a memory/execution model. Return Undef.
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
            | Instr::FConst { ty, .. }
            | Instr::AtomicLoad { ty, .. }
            | Instr::AtomicRmw { ty, .. } => vec![ty.clone()],
            // CmpXchg returns (old_value: ty, success: bool).
            Instr::CmpXchg { ty, .. } => vec![ty.clone(), Ty::bool_ty()],
            Instr::Select { ty, .. } => vec![ty.clone()],
            Instr::GetElementPtr { .. } => vec![Ty::ptr(Ty::void())],
            Instr::Cmp { .. } | Instr::IsUnique { .. } => vec![Ty::bool_ty()],
            Instr::Cast { dst_ty, .. } => vec![dst_ty.clone()],
            Instr::Call { ret_ty, .. } | Instr::CallIndirect { ret_ty, .. } => ret_ty.clone(),
            // Void instructions
            Instr::Store { .. }
            | Instr::AtomicStore { .. }
            | Instr::Fence { .. }
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
            Instr::BinOp {
                op: BinOp::SDiv | BinOp::UDiv,
                ..
            } => Some("Division by zero".to_string()),
            Instr::BinOp {
                op: BinOp::SRem | BinOp::URem,
                ..
            } => Some("Remainder by zero".to_string()),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tmir_instrs::Operand;

    fn sem() -> ConcreteSemantics {
        ConcreteSemantics
    }

    /// Helper: create a concrete integer SemValue.
    fn iv(val: i128, width: u16) -> SemValue {
        SemValue::Int(BitVec::Concrete {
            width,
            value: mask_to_width(val, width),
        })
    }

    /// Helper: create a float SemValue.
    fn fv(val: f64, width: u16) -> SemValue {
        SemValue::Float { width, value: val }
    }

    /// Helper: create a BinOp instruction with dummy operands.
    fn binop(op: BinOp, ty: Ty) -> Instr {
        Instr::BinOp {
            op,
            ty,
            lhs: Operand::value(tmir_types::ValueId(0)),
            rhs: Operand::value(tmir_types::ValueId(1)),
        }
    }

    /// Helper: create a UnOp instruction.
    fn unop(op: UnOp, ty: Ty) -> Instr {
        Instr::UnOp {
            op,
            ty,
            operand: Operand::value(tmir_types::ValueId(0)),
        }
    }

    /// Helper: create a Cmp instruction.
    fn cmp(op: CmpOp, ty: Ty) -> Instr {
        Instr::Cmp {
            op,
            ty,
            lhs: Operand::value(tmir_types::ValueId(0)),
            rhs: Operand::value(tmir_types::ValueId(1)),
        }
    }

    /// Helper: create a Cast instruction.
    fn cast(op: CastOp, src_ty: Ty, dst_ty: Ty) -> Instr {
        Instr::Cast {
            op,
            src_ty,
            dst_ty,
            operand: Operand::value(tmir_types::ValueId(0)),
        }
    }

    // -----------------------------------------------------------------------
    // mask_to_width tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mask_to_width_basic() {
        assert_eq!(mask_to_width(0xFF, 8), 0xFF);
        assert_eq!(mask_to_width(0x1FF, 8), 0xFF);
        assert_eq!(mask_to_width(0xFFFF_FFFF, 32), 0xFFFF_FFFF);
        assert_eq!(mask_to_width(0x1_FFFF_FFFF, 32), 0xFFFF_FFFF);
    }

    #[test]
    fn test_mask_to_width_zero() {
        assert_eq!(mask_to_width(42, 0), 0);
    }

    #[test]
    fn test_mask_to_width_128() {
        let big = i128::MAX;
        assert_eq!(mask_to_width(big, 128), big);
    }

    // -----------------------------------------------------------------------
    // to_signed tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_signed_positive() {
        assert_eq!(to_signed(42, 32), 42);
    }

    #[test]
    fn test_to_signed_negative() {
        // 0xFFFF_FFFF in 32-bit = -1
        assert_eq!(to_signed(0xFFFF_FFFF, 32), -1);
        // 0xFF in 8-bit = -1
        assert_eq!(to_signed(0xFF, 8), -1);
        // 0x80 in 8-bit = -128
        assert_eq!(to_signed(0x80, 8), -128);
    }

    // -----------------------------------------------------------------------
    // BinOp: integer arithmetic
    // -----------------------------------------------------------------------

    #[test]
    fn test_binop_add() {
        let r = sem().eval(&binop(BinOp::Add, Ty::i32()), &[iv(3, 32), iv(4, 32)]);
        assert_eq!(r, vec![iv(7, 32)]);
    }

    #[test]
    fn test_binop_add_overflow() {
        // 0xFFFF_FFFF + 1 wraps to 0 in 32-bit
        let r = sem().eval(
            &binop(BinOp::Add, Ty::i32()),
            &[iv(0xFFFF_FFFF, 32), iv(1, 32)],
        );
        assert_eq!(r, vec![iv(0, 32)]);
    }

    #[test]
    fn test_binop_sub() {
        let r = sem().eval(&binop(BinOp::Sub, Ty::i32()), &[iv(10, 32), iv(3, 32)]);
        assert_eq!(r, vec![iv(7, 32)]);
    }

    #[test]
    fn test_binop_sub_underflow() {
        // 0 - 1 wraps to 0xFFFF_FFFF in 32-bit
        let r = sem().eval(&binop(BinOp::Sub, Ty::i32()), &[iv(0, 32), iv(1, 32)]);
        assert_eq!(r, vec![iv(0xFFFF_FFFF, 32)]);
    }

    #[test]
    fn test_binop_mul() {
        let r = sem().eval(&binop(BinOp::Mul, Ty::i32()), &[iv(6, 32), iv(7, 32)]);
        assert_eq!(r, vec![iv(42, 32)]);
    }

    #[test]
    fn test_binop_sdiv() {
        // -10 / 3 = -3 (truncated toward zero)
        let neg10_32 = mask_to_width(-10i128, 32);
        let r = sem().eval(
            &binop(BinOp::SDiv, Ty::i32()),
            &[iv(neg10_32, 32), iv(3, 32)],
        );
        // -3 in 32-bit unsigned representation
        let neg3_32 = mask_to_width(-3i128, 32);
        assert_eq!(r, vec![iv(neg3_32, 32)]);
    }

    #[test]
    fn test_binop_udiv() {
        let r = sem().eval(&binop(BinOp::UDiv, Ty::i32()), &[iv(42, 32), iv(6, 32)]);
        assert_eq!(r, vec![iv(7, 32)]);
    }

    #[test]
    fn test_binop_div_by_zero() {
        let r = sem().eval(&binop(BinOp::SDiv, Ty::i32()), &[iv(42, 32), iv(0, 32)]);
        assert_eq!(r, vec![SemValue::Undef]);
        let r = sem().eval(&binop(BinOp::UDiv, Ty::i32()), &[iv(42, 32), iv(0, 32)]);
        assert_eq!(r, vec![SemValue::Undef]);
    }

    #[test]
    fn test_binop_srem() {
        // -10 % 3 = -1
        let neg10_32 = mask_to_width(-10i128, 32);
        let r = sem().eval(
            &binop(BinOp::SRem, Ty::i32()),
            &[iv(neg10_32, 32), iv(3, 32)],
        );
        let neg1_32 = mask_to_width(-1i128, 32);
        assert_eq!(r, vec![iv(neg1_32, 32)]);
    }

    #[test]
    fn test_binop_urem() {
        let r = sem().eval(&binop(BinOp::URem, Ty::i32()), &[iv(10, 32), iv(3, 32)]);
        assert_eq!(r, vec![iv(1, 32)]);
    }

    #[test]
    fn test_binop_rem_by_zero() {
        let r = sem().eval(&binop(BinOp::SRem, Ty::i32()), &[iv(10, 32), iv(0, 32)]);
        assert_eq!(r, vec![SemValue::Undef]);
        let r = sem().eval(&binop(BinOp::URem, Ty::i32()), &[iv(10, 32), iv(0, 32)]);
        assert_eq!(r, vec![SemValue::Undef]);
    }

    // -----------------------------------------------------------------------
    // BinOp: bitwise operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_binop_and() {
        let r = sem().eval(
            &binop(BinOp::And, Ty::i32()),
            &[iv(0xFF00_FF00, 32), iv(0x0F0F_0F0F, 32)],
        );
        assert_eq!(r, vec![iv(0x0F00_0F00, 32)]);
    }

    #[test]
    fn test_binop_or() {
        let r = sem().eval(
            &binop(BinOp::Or, Ty::i32()),
            &[iv(0xFF00_0000, 32), iv(0x00FF_0000, 32)],
        );
        assert_eq!(r, vec![iv(0xFFFF_0000, 32)]);
    }

    #[test]
    fn test_binop_xor() {
        let r = sem().eval(
            &binop(BinOp::Xor, Ty::i32()),
            &[iv(0xAAAA_AAAA, 32), iv(0x5555_5555, 32)],
        );
        assert_eq!(r, vec![iv(0xFFFF_FFFF, 32)]);
    }

    // -----------------------------------------------------------------------
    // BinOp: shift operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_binop_shl() {
        let r = sem().eval(&binop(BinOp::Shl, Ty::i32()), &[iv(1, 32), iv(4, 32)]);
        assert_eq!(r, vec![iv(16, 32)]);
    }

    #[test]
    fn test_binop_lshr() {
        let r = sem().eval(
            &binop(BinOp::LShr, Ty::i32()),
            &[iv(0x8000_0000, 32), iv(4, 32)],
        );
        assert_eq!(r, vec![iv(0x0800_0000, 32)]);
    }

    #[test]
    fn test_binop_ashr() {
        // Arithmetic shift right of 0x80000000 by 4 = 0xF8000000 (sign-extends)
        let r = sem().eval(
            &binop(BinOp::AShr, Ty::i32()),
            &[iv(0x8000_0000, 32), iv(4, 32)],
        );
        assert_eq!(r, vec![iv(0xF800_0000, 32)]);
    }

    #[test]
    fn test_binop_ashr_positive() {
        // Positive value: same as logical shift right
        let r = sem().eval(
            &binop(BinOp::AShr, Ty::i32()),
            &[iv(0x4000_0000, 32), iv(4, 32)],
        );
        assert_eq!(r, vec![iv(0x0400_0000, 32)]);
    }

    // -----------------------------------------------------------------------
    // BinOp: floating point
    // -----------------------------------------------------------------------

    #[test]
    fn test_binop_fadd() {
        let r = sem().eval(
            &binop(BinOp::FAdd, Ty::f64()),
            &[fv(1.5, 64), fv(2.5, 64)],
        );
        assert_eq!(r, vec![fv(4.0, 64)]);
    }

    #[test]
    fn test_binop_fsub() {
        let r = sem().eval(
            &binop(BinOp::FSub, Ty::f64()),
            &[fv(10.0, 64), fv(3.5, 64)],
        );
        assert_eq!(r, vec![fv(6.5, 64)]);
    }

    #[test]
    fn test_binop_fmul() {
        let r = sem().eval(
            &binop(BinOp::FMul, Ty::f64()),
            &[fv(3.0, 64), fv(7.0, 64)],
        );
        assert_eq!(r, vec![fv(21.0, 64)]);
    }

    #[test]
    fn test_binop_fdiv() {
        let r = sem().eval(
            &binop(BinOp::FDiv, Ty::f64()),
            &[fv(10.0, 64), fv(4.0, 64)],
        );
        assert_eq!(r, vec![fv(2.5, 64)]);
    }

    // -----------------------------------------------------------------------
    // UnOp tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_unop_neg() {
        let r = sem().eval(&unop(UnOp::Neg, Ty::i32()), &[iv(5, 32)]);
        // neg(5) = 0xFFFFFFFB
        assert_eq!(r, vec![iv(0xFFFF_FFFB, 32)]);
    }

    #[test]
    fn test_unop_neg_zero() {
        let r = sem().eval(&unop(UnOp::Neg, Ty::i32()), &[iv(0, 32)]);
        assert_eq!(r, vec![iv(0, 32)]);
    }

    #[test]
    fn test_unop_not() {
        let r = sem().eval(&unop(UnOp::Not, Ty::i32()), &[iv(0, 32)]);
        assert_eq!(r, vec![iv(0xFFFF_FFFF, 32)]);
    }

    #[test]
    fn test_unop_not_ones() {
        let r = sem().eval(&unop(UnOp::Not, Ty::i32()), &[iv(0xFFFF_FFFF, 32)]);
        assert_eq!(r, vec![iv(0, 32)]);
    }

    #[test]
    fn test_unop_fneg() {
        let r = sem().eval(&unop(UnOp::FNeg, Ty::f64()), &[fv(42.0, 64)]);
        assert_eq!(r, vec![fv(-42.0, 64)]);
    }

    #[test]
    fn test_unop_fabs() {
        let r = sem().eval(&unop(UnOp::FAbs, Ty::f64()), &[fv(-3.14, 64)]);
        assert_eq!(r, vec![fv(3.14, 64)]);
    }

    #[test]
    fn test_unop_fsqrt() {
        let r = sem().eval(&unop(UnOp::FSqrt, Ty::f64()), &[fv(9.0, 64)]);
        assert_eq!(r, vec![fv(3.0, 64)]);
    }

    // -----------------------------------------------------------------------
    // Comparison: integer
    // -----------------------------------------------------------------------

    #[test]
    fn test_cmp_eq_true() {
        let r = sem().eval(&cmp(CmpOp::Eq, Ty::i32()), &[iv(42, 32), iv(42, 32)]);
        assert_eq!(r, vec![SemValue::Bool(true)]);
    }

    #[test]
    fn test_cmp_eq_false() {
        let r = sem().eval(&cmp(CmpOp::Eq, Ty::i32()), &[iv(42, 32), iv(43, 32)]);
        assert_eq!(r, vec![SemValue::Bool(false)]);
    }

    #[test]
    fn test_cmp_ne() {
        let r = sem().eval(&cmp(CmpOp::Ne, Ty::i32()), &[iv(42, 32), iv(43, 32)]);
        assert_eq!(r, vec![SemValue::Bool(true)]);
    }

    #[test]
    fn test_cmp_slt_signed() {
        // -1 (0xFFFFFFFF) <_s 0 = true
        let r = sem().eval(
            &cmp(CmpOp::Slt, Ty::i32()),
            &[iv(0xFFFF_FFFF, 32), iv(0, 32)],
        );
        assert_eq!(r, vec![SemValue::Bool(true)]);
    }

    #[test]
    fn test_cmp_ult_unsigned() {
        // 0xFFFFFFFF <_u 0 = false (it's the biggest unsigned value)
        let r = sem().eval(
            &cmp(CmpOp::Ult, Ty::i32()),
            &[iv(0xFFFF_FFFF, 32), iv(0, 32)],
        );
        assert_eq!(r, vec![SemValue::Bool(false)]);
    }

    #[test]
    fn test_cmp_ult_true() {
        let r = sem().eval(&cmp(CmpOp::Ult, Ty::i32()), &[iv(3, 32), iv(10, 32)]);
        assert_eq!(r, vec![SemValue::Bool(true)]);
    }

    #[test]
    fn test_cmp_sge() {
        // 5 >=_s 5 = true
        let r = sem().eval(&cmp(CmpOp::Sge, Ty::i32()), &[iv(5, 32), iv(5, 32)]);
        assert_eq!(r, vec![SemValue::Bool(true)]);
    }

    #[test]
    fn test_cmp_sgt() {
        let r = sem().eval(&cmp(CmpOp::Sgt, Ty::i32()), &[iv(10, 32), iv(5, 32)]);
        assert_eq!(r, vec![SemValue::Bool(true)]);
    }

    #[test]
    fn test_cmp_sle() {
        let r = sem().eval(&cmp(CmpOp::Sle, Ty::i32()), &[iv(5, 32), iv(5, 32)]);
        assert_eq!(r, vec![SemValue::Bool(true)]);
    }

    #[test]
    fn test_cmp_uge() {
        let r = sem().eval(
            &cmp(CmpOp::Uge, Ty::i32()),
            &[iv(0xFFFF_FFFF, 32), iv(0, 32)],
        );
        assert_eq!(r, vec![SemValue::Bool(true)]);
    }

    #[test]
    fn test_cmp_ugt() {
        let r = sem().eval(&cmp(CmpOp::Ugt, Ty::i32()), &[iv(10, 32), iv(3, 32)]);
        assert_eq!(r, vec![SemValue::Bool(true)]);
    }

    #[test]
    fn test_cmp_ule() {
        let r = sem().eval(&cmp(CmpOp::Ule, Ty::i32()), &[iv(3, 32), iv(3, 32)]);
        assert_eq!(r, vec![SemValue::Bool(true)]);
    }

    // -----------------------------------------------------------------------
    // Comparison: floating point
    // -----------------------------------------------------------------------

    #[test]
    fn test_cmp_foeq() {
        let r = sem().eval(&cmp(CmpOp::FOeq, Ty::f64()), &[fv(1.0, 64), fv(1.0, 64)]);
        assert_eq!(r, vec![SemValue::Bool(true)]);
    }

    #[test]
    fn test_cmp_foeq_nan() {
        let r = sem().eval(
            &cmp(CmpOp::FOeq, Ty::f64()),
            &[fv(f64::NAN, 64), fv(1.0, 64)],
        );
        assert_eq!(r, vec![SemValue::Bool(false)]);
    }

    #[test]
    fn test_cmp_fueq_nan() {
        // Unordered: true if either is NaN
        let r = sem().eval(
            &cmp(CmpOp::FUeq, Ty::f64()),
            &[fv(f64::NAN, 64), fv(1.0, 64)],
        );
        assert_eq!(r, vec![SemValue::Bool(true)]);
    }

    #[test]
    fn test_cmp_folt() {
        let r = sem().eval(&cmp(CmpOp::FOlt, Ty::f64()), &[fv(1.0, 64), fv(2.0, 64)]);
        assert_eq!(r, vec![SemValue::Bool(true)]);
    }

    #[test]
    fn test_cmp_fone() {
        let r = sem().eval(&cmp(CmpOp::FOne, Ty::f64()), &[fv(1.0, 64), fv(2.0, 64)]);
        assert_eq!(r, vec![SemValue::Bool(true)]);
    }

    #[test]
    fn test_cmp_fune_nan() {
        let r = sem().eval(
            &cmp(CmpOp::FUne, Ty::f64()),
            &[fv(f64::NAN, 64), fv(f64::NAN, 64)],
        );
        assert_eq!(r, vec![SemValue::Bool(true)]);
    }

    // -----------------------------------------------------------------------
    // Cast operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_cast_zext() {
        // Zero-extend 8-bit 0xFF to 32-bit = 0x000000FF
        let r = sem().eval(&cast(CastOp::ZExt, Ty::i8(), Ty::i32()), &[iv(0xFF, 8)]);
        assert_eq!(r, vec![iv(0xFF, 32)]);
    }

    #[test]
    fn test_cast_sext() {
        // Sign-extend 8-bit 0xFF (-1) to 32-bit = 0xFFFFFFFF
        let r = sem().eval(&cast(CastOp::SExt, Ty::i8(), Ty::i32()), &[iv(0xFF, 8)]);
        assert_eq!(r, vec![iv(0xFFFF_FFFF, 32)]);
    }

    #[test]
    fn test_cast_sext_positive() {
        // Sign-extend 8-bit 0x7F (127) to 32-bit = 0x0000007F
        let r = sem().eval(&cast(CastOp::SExt, Ty::i8(), Ty::i32()), &[iv(0x7F, 8)]);
        assert_eq!(r, vec![iv(0x7F, 32)]);
    }

    #[test]
    fn test_cast_trunc() {
        // Truncate 32-bit 0xDEAD_BEEF to 8-bit = 0xEF
        let r = sem().eval(
            &cast(CastOp::Trunc, Ty::i32(), Ty::i8()),
            &[iv(0xDEAD_BEEF, 32)],
        );
        assert_eq!(r, vec![iv(0xEF, 8)]);
    }

    #[test]
    fn test_cast_fptosi() {
        let r = sem().eval(
            &cast(CastOp::FPToSI, Ty::f64(), Ty::i32()),
            &[fv(-3.7, 64)],
        );
        assert_eq!(r, vec![iv(mask_to_width(-3i128, 32), 32)]);
    }

    #[test]
    fn test_cast_fptoui() {
        let r = sem().eval(
            &cast(CastOp::FPToUI, Ty::f64(), Ty::i32()),
            &[fv(42.9, 64)],
        );
        assert_eq!(r, vec![iv(42, 32)]);
    }

    #[test]
    fn test_cast_sitofp() {
        // -1 as signed 8-bit to f64
        let r = sem().eval(
            &cast(CastOp::SIToFP, Ty::i8(), Ty::f64()),
            &[iv(0xFF, 8)],
        );
        assert_eq!(r, vec![fv(-1.0, 64)]);
    }

    #[test]
    fn test_cast_uitofp() {
        // 0xFF as unsigned 8-bit to f64 = 255.0
        let r = sem().eval(
            &cast(CastOp::UIToFP, Ty::u8(), Ty::f64()),
            &[iv(0xFF, 8)],
        );
        assert_eq!(r, vec![fv(255.0, 64)]);
    }

    // -----------------------------------------------------------------------
    // Const / FConst
    // -----------------------------------------------------------------------

    #[test]
    fn test_const() {
        let instr = Instr::Const {
            ty: Ty::i32(),
            value: 42,
        };
        let r = sem().eval(&instr, &[]);
        assert_eq!(r, vec![iv(42, 32)]);
    }

    #[test]
    fn test_fconst() {
        let instr = Instr::FConst {
            ty: Ty::f64(),
            value: 3.14,
        };
        let r = sem().eval(&instr, &[]);
        assert_eq!(r, vec![fv(3.14, 64)]);
    }

    // -----------------------------------------------------------------------
    // Select
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_true() {
        let instr = Instr::Select {
            ty: Ty::i32(),
            cond: Operand::value(tmir_types::ValueId(0)),
            true_val: Operand::value(tmir_types::ValueId(1)),
            false_val: Operand::value(tmir_types::ValueId(2)),
        };
        let r = sem().eval(&instr, &[SemValue::Bool(true), iv(10, 32), iv(20, 32)]);
        assert_eq!(r, vec![iv(10, 32)]);
    }

    #[test]
    fn test_select_false() {
        let instr = Instr::Select {
            ty: Ty::i32(),
            cond: Operand::value(tmir_types::ValueId(0)),
            true_val: Operand::value(tmir_types::ValueId(1)),
            false_val: Operand::value(tmir_types::ValueId(2)),
        };
        let r = sem().eval(&instr, &[SemValue::Bool(false), iv(10, 32), iv(20, 32)]);
        assert_eq!(r, vec![iv(20, 32)]);
    }

    #[test]
    fn test_select_int_cond() {
        // Non-zero integer condition = true
        let instr = Instr::Select {
            ty: Ty::i32(),
            cond: Operand::value(tmir_types::ValueId(0)),
            true_val: Operand::value(tmir_types::ValueId(1)),
            false_val: Operand::value(tmir_types::ValueId(2)),
        };
        let r = sem().eval(&instr, &[iv(1, 32), iv(10, 32), iv(20, 32)]);
        assert_eq!(r, vec![iv(10, 32)]);
    }

    // -----------------------------------------------------------------------
    // Nop
    // -----------------------------------------------------------------------

    #[test]
    fn test_nop() {
        let r = sem().eval(&Instr::Nop, &[]);
        assert!(r.is_empty());
    }

    // -----------------------------------------------------------------------
    // 8-bit operations (verify width handling)
    // -----------------------------------------------------------------------

    #[test]
    fn test_8bit_add_overflow() {
        // 200 + 100 = 300, masked to 8 bits = 44
        let r = sem().eval(&binop(BinOp::Add, Ty::i8()), &[iv(200, 8), iv(100, 8)]);
        assert_eq!(r, vec![iv(44, 8)]);
    }

    #[test]
    fn test_8bit_neg() {
        // neg(1) in 8-bit = 0xFF = 255
        let r = sem().eval(&unop(UnOp::Neg, Ty::i8()), &[iv(1, 8)]);
        assert_eq!(r, vec![iv(0xFF, 8)]);
    }

    // -----------------------------------------------------------------------
    // result_types
    // -----------------------------------------------------------------------

    #[test]
    fn test_result_types_binop() {
        let instr = binop(BinOp::Add, Ty::i32());
        let tys = sem().result_types(&instr, &[]);
        assert_eq!(tys, vec![Ty::i32()]);
    }

    #[test]
    fn test_result_types_cmp() {
        let instr = cmp(CmpOp::Eq, Ty::i32());
        let tys = sem().result_types(&instr, &[]);
        assert_eq!(tys, vec![Ty::bool_ty()]);
    }

    #[test]
    fn test_result_types_nop() {
        let tys = sem().result_types(&Instr::Nop, &[]);
        assert!(tys.is_empty());
    }

    // -----------------------------------------------------------------------
    // ub_condition
    // -----------------------------------------------------------------------

    #[test]
    fn test_ub_condition_div() {
        let instr = binop(BinOp::SDiv, Ty::i32());
        assert!(sem().ub_condition(&instr).is_some());
    }

    #[test]
    fn test_ub_condition_add() {
        let instr = binop(BinOp::Add, Ty::i32());
        assert!(sem().ub_condition(&instr).is_none());
    }

    // -----------------------------------------------------------------------
    // Missing inputs -> Undef
    // -----------------------------------------------------------------------

    #[test]
    fn test_binop_missing_inputs() {
        let r = sem().eval(&binop(BinOp::Add, Ty::i32()), &[iv(42, 32)]);
        assert_eq!(r, vec![SemValue::Undef]);
    }

    #[test]
    fn test_unop_missing_inputs() {
        let r = sem().eval(&unop(UnOp::Neg, Ty::i32()), &[]);
        assert_eq!(r, vec![SemValue::Undef]);
    }

    #[test]
    fn test_cmp_missing_inputs() {
        let r = sem().eval(&cmp(CmpOp::Eq, Ty::i32()), &[iv(42, 32)]);
        assert_eq!(r, vec![SemValue::Undef]);
    }

    // -----------------------------------------------------------------------
    // 64-bit operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_64bit_add() {
        let r = sem().eval(
            &binop(BinOp::Add, Ty::i64()),
            &[iv(0x1_0000_0000, 64), iv(0x2_0000_0000, 64)],
        );
        assert_eq!(r, vec![iv(0x3_0000_0000, 64)]);
    }

    #[test]
    fn test_64bit_slt() {
        // Compare large unsigned value interpreted as signed
        let big = mask_to_width(-1i128, 64); // 0xFFFF_FFFF_FFFF_FFFF
        let r = sem().eval(&cmp(CmpOp::Slt, Ty::i64()), &[iv(big, 64), iv(0, 64)]);
        assert_eq!(r, vec![SemValue::Bool(true)]); // -1 < 0
    }
}
