// llvm2-verify/smt.rs - SMT expression AST and bitvector evaluator
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Self-contained SMT expression AST for verification of lowering rules.
// When z4-bindings becomes a direct dependency, these types will serialize
// to the z4 Expr/Sort API. Until then, we evaluate locally using Rust
// wrapping arithmetic (two's complement bitvector semantics).

//! SMT bitvector expression AST and concrete evaluator.
//!
//! This module defines a lightweight expression tree for bitvector operations
//! matching SMT-LIB2 QF_BV semantics. Expressions can be:
//!
//! 1. **Symbolically constructed** to describe proof obligations
//! 2. **Concretely evaluated** via [`SmtExpr::eval`] for testing/verification

use llvm2_lower::types::Type;
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

// ---------------------------------------------------------------------------
// RoundingMode (IEEE 754)
// ---------------------------------------------------------------------------

/// IEEE 754 rounding mode for floating-point operations.
///
/// Maps to SMT-LIB2 rounding modes in the QF_FP theory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoundingMode {
    /// Round to nearest, ties to even (default).
    RNE,
    /// Round to nearest, ties away from zero.
    RNA,
    /// Round toward positive infinity.
    RTP,
    /// Round toward negative infinity.
    RTN,
    /// Round toward zero (truncation).
    RTZ,
}

// ---------------------------------------------------------------------------
// SmtError
// ---------------------------------------------------------------------------

/// Errors arising from SMT expression construction or evaluation.
#[derive(Debug, Error)]
pub enum SmtError {
    /// A type cannot be represented in the SMT bitvector domain.
    #[error("unsupported type for SMT encoding: {0}")]
    UnsupportedType(String),

    /// `bv_width()` called on a Bool-sorted expression.
    #[error("bv_width called on Bool-sorted expression")]
    BoolHasNoWidth,

    /// Variable not found during concrete evaluation.
    #[error("variable '{0}' not found in evaluation environment")]
    UndefinedVariable(String),

    /// Recursive evaluation failed.
    #[error("evaluation error: {0}")]
    EvalError(String),
}

// ---------------------------------------------------------------------------
// SmtSort
// ---------------------------------------------------------------------------

/// SMT sort (type) for bitvector verification.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SmtSort {
    /// Fixed-width bitvector.  Width must be > 0.
    BitVec(u32),
    /// Boolean (used for comparison results, preconditions).
    Bool,
    /// Array sort: `(Array index_sort element_sort)`.
    ///
    /// Maps to SMT-LIB2 QF_ABV (arrays of bitvectors) or QF_AUFBV.
    Array(Box<SmtSort>, Box<SmtSort>),
    /// IEEE 754 floating-point sort: `(_ FloatingPoint eb sb)`.
    ///
    /// `eb` = exponent bits, `sb` = significand bits (including implicit bit).
    /// Maps to SMT-LIB2 QF_FP theory.
    FloatingPoint(u32, u32),
}

impl SmtSort {
    /// Bitvector width, or `None` for non-bitvector sorts.
    pub fn bv_width(&self) -> Option<u32> {
        match self {
            SmtSort::BitVec(w) => Some(*w),
            _ => None,
        }
    }

    /// IEEE 754 half-precision: `(_ FloatingPoint 5 11)`.
    pub fn fp16() -> Self {
        SmtSort::FloatingPoint(5, 11)
    }

    /// IEEE 754 single-precision: `(_ FloatingPoint 8 24)`.
    pub fn fp32() -> Self {
        SmtSort::FloatingPoint(8, 24)
    }

    /// IEEE 754 double-precision: `(_ FloatingPoint 11 53)`.
    pub fn fp64() -> Self {
        SmtSort::FloatingPoint(11, 53)
    }

    /// Convenience: array from bitvectors to bitvectors.
    pub fn bv_array(index_width: u32, element_width: u32) -> Self {
        SmtSort::Array(
            Box::new(SmtSort::BitVec(index_width)),
            Box::new(SmtSort::BitVec(element_width)),
        )
    }
}

impl TryFrom<Type> for SmtSort {
    type Error = SmtError;

    fn try_from(ty: Type) -> Result<Self, SmtError> {
        match ty {
            Type::B1 => Ok(SmtSort::BitVec(1)),
            Type::I8 => Ok(SmtSort::BitVec(8)),
            Type::I16 => Ok(SmtSort::BitVec(16)),
            Type::I32 => Ok(SmtSort::BitVec(32)),
            Type::I64 => Ok(SmtSort::BitVec(64)),
            Type::I128 => Ok(SmtSort::BitVec(128)),
            Type::F32 => Ok(SmtSort::fp32()),
            Type::F64 => Ok(SmtSort::fp64()),
            Type::Struct(_) => Err(SmtError::UnsupportedType(
                "struct type verification not yet supported".to_string(),
            )),
            Type::Array(elem_ty, count) => {
                let elem_sort = SmtSort::try_from(*elem_ty)?;
                // Index sort: bitvector wide enough to address `count` elements.
                let index_bits = if count == 0 { 1 } else { 32u32.max((count as f64).log2().ceil() as u32).max(1) };
                Ok(SmtSort::Array(
                    Box::new(SmtSort::BitVec(index_bits)),
                    Box::new(elem_sort),
                ))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SmtExpr
// ---------------------------------------------------------------------------

/// A bitvector SMT expression.
///
/// All bitvector operations use wrapping (two's complement) semantics.
/// The `width` field on BV nodes tracks the bitvector width for masking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmtExpr {
    /// Symbolic variable: `(declare-const name (_ BitVec width))`
    Var { name: String, width: u32 },

    /// Bitvector constant.
    BvConst { value: u64, width: u32 },

    /// Boolean constant.
    BoolConst(bool),

    // -- Bitvector arithmetic --

    /// `bvadd(lhs, rhs)` -- wrapping addition.
    BvAdd { lhs: Box<SmtExpr>, rhs: Box<SmtExpr>, width: u32 },

    /// `bvsub(lhs, rhs)` -- wrapping subtraction.
    BvSub { lhs: Box<SmtExpr>, rhs: Box<SmtExpr>, width: u32 },

    /// `bvmul(lhs, rhs)` -- wrapping multiplication.
    BvMul { lhs: Box<SmtExpr>, rhs: Box<SmtExpr>, width: u32 },

    /// `bvsdiv(lhs, rhs)` -- signed division (truncates toward zero).
    BvSDiv { lhs: Box<SmtExpr>, rhs: Box<SmtExpr>, width: u32 },

    /// `bvudiv(lhs, rhs)` -- unsigned division.
    BvUDiv { lhs: Box<SmtExpr>, rhs: Box<SmtExpr>, width: u32 },

    /// `bvneg(operand)` -- two's complement negation.
    BvNeg { operand: Box<SmtExpr>, width: u32 },

    // -- Bitvector comparison (result is Bool) --

    /// `bveq(lhs, rhs)` -- equality.
    Eq { lhs: Box<SmtExpr>, rhs: Box<SmtExpr> },

    /// `not(operand)` -- boolean negation.
    Not { operand: Box<SmtExpr> },

    /// `bvslt(lhs, rhs)` -- signed less-than.
    BvSlt { lhs: Box<SmtExpr>, rhs: Box<SmtExpr>, width: u32 },

    /// `bvsge(lhs, rhs)` -- signed greater-or-equal.
    BvSge { lhs: Box<SmtExpr>, rhs: Box<SmtExpr>, width: u32 },

    /// `bvuge(lhs, rhs)` -- unsigned greater-or-equal.
    BvUge { lhs: Box<SmtExpr>, rhs: Box<SmtExpr>, width: u32 },

    /// `and(lhs, rhs)` -- boolean AND.
    And { lhs: Box<SmtExpr>, rhs: Box<SmtExpr> },

    /// `or(lhs, rhs)` -- boolean OR.
    Or { lhs: Box<SmtExpr>, rhs: Box<SmtExpr> },

    /// `bvsgt(lhs, rhs)` -- signed greater-than.
    BvSgt { lhs: Box<SmtExpr>, rhs: Box<SmtExpr>, width: u32 },

    /// `bvsle(lhs, rhs)` -- signed less-or-equal.
    BvSle { lhs: Box<SmtExpr>, rhs: Box<SmtExpr>, width: u32 },

    /// `bvult(lhs, rhs)` -- unsigned less-than.
    BvUlt { lhs: Box<SmtExpr>, rhs: Box<SmtExpr>, width: u32 },

    /// `bvugt(lhs, rhs)` -- unsigned greater-than.
    BvUgt { lhs: Box<SmtExpr>, rhs: Box<SmtExpr>, width: u32 },

    /// `bvule(lhs, rhs)` -- unsigned less-or-equal.
    BvUle { lhs: Box<SmtExpr>, rhs: Box<SmtExpr>, width: u32 },

    /// `ite(cond, then_expr, else_expr)` -- if-then-else.
    Ite { cond: Box<SmtExpr>, then_expr: Box<SmtExpr>, else_expr: Box<SmtExpr> },

    /// `extract(high, low, operand)` -- bit extraction `operand[high:low]`.
    Extract { high: u32, low: u32, operand: Box<SmtExpr>, width: u32 },

    // -- Bitwise operations --

    /// `bvand(lhs, rhs)` -- bitwise AND.
    BvAnd { lhs: Box<SmtExpr>, rhs: Box<SmtExpr>, width: u32 },

    /// `bvor(lhs, rhs)` -- bitwise OR.
    BvOr { lhs: Box<SmtExpr>, rhs: Box<SmtExpr>, width: u32 },

    /// `bvxor(lhs, rhs)` -- bitwise XOR.
    BvXor { lhs: Box<SmtExpr>, rhs: Box<SmtExpr>, width: u32 },

    // -- Shift operations --

    /// `bvshl(lhs, rhs)` -- logical shift left.
    BvShl { lhs: Box<SmtExpr>, rhs: Box<SmtExpr>, width: u32 },

    /// `bvlshr(lhs, rhs)` -- logical shift right.
    BvLshr { lhs: Box<SmtExpr>, rhs: Box<SmtExpr>, width: u32 },

    /// `bvashr(lhs, rhs)` -- arithmetic shift right.
    BvAshr { lhs: Box<SmtExpr>, rhs: Box<SmtExpr>, width: u32 },

    /// `concat(hi, lo)` -- bitvector concatenation.
    ///
    /// Produces a bitvector of width `hi.width + lo.width` where the high bits
    /// come from `hi` and the low bits come from `lo`.
    /// SMT-LIB2: `(concat hi lo)`.
    Concat { hi: Box<SmtExpr>, lo: Box<SmtExpr>, width: u32 },

    /// `zero_extend(operand, extra_bits)` -- zero-extend by `extra_bits` bits.
    ///
    /// Produces a bitvector of width `operand.width + extra_bits`.
    /// SMT-LIB2: `((_ zero_extend extra_bits) operand)`.
    ZeroExtend { operand: Box<SmtExpr>, extra_bits: u32, width: u32 },

    /// `sign_extend(operand, extra_bits)` -- sign-extend by `extra_bits` bits.
    ///
    /// Produces a bitvector of width `operand.width + extra_bits`.
    /// SMT-LIB2: `((_ sign_extend extra_bits) operand)`.
    SignExtend { operand: Box<SmtExpr>, extra_bits: u32, width: u32 },

    // -- Array operations (QF_ABV theory) --

    /// `(select array index)` -- read element at index.
    Select {
        array: Box<SmtExpr>,
        index: Box<SmtExpr>,
    },

    /// `(store array index value)` -- write element at index, producing new array.
    Store {
        array: Box<SmtExpr>,
        index: Box<SmtExpr>,
        value: Box<SmtExpr>,
    },

    /// `((as const (Array idx_sort elem_sort)) value)` -- constant array.
    ConstArray {
        index_sort: SmtSort,
        value: Box<SmtExpr>,
    },

    // -- Floating-point operations (QF_FP theory) --

    /// `(fp.add rm a b)` -- floating-point addition.
    FPAdd { rm: RoundingMode, lhs: Box<SmtExpr>, rhs: Box<SmtExpr> },

    /// `(fp.mul rm a b)` -- floating-point multiplication.
    FPMul { rm: RoundingMode, lhs: Box<SmtExpr>, rhs: Box<SmtExpr> },

    /// `(fp.div rm a b)` -- floating-point division.
    FPDiv { rm: RoundingMode, lhs: Box<SmtExpr>, rhs: Box<SmtExpr> },

    /// `(fp.neg a)` -- floating-point negation.
    FPNeg { operand: Box<SmtExpr> },

    /// `(fp.eq a b)` -- floating-point equality (returns Bool).
    FPEq { lhs: Box<SmtExpr>, rhs: Box<SmtExpr> },

    /// `(fp.lt a b)` -- floating-point less-than (returns Bool).
    FPLt { lhs: Box<SmtExpr>, rhs: Box<SmtExpr> },

    /// Floating-point constant from f64 bits.
    ///
    /// `eb` = exponent bits, `sb` = significand bits.
    /// The `bits` field holds the IEEE 754 bit pattern.
    FPConst { bits: u64, eb: u32, sb: u32 },

    // -- Uninterpreted functions (QF_UF theory) --

    /// `(name arg1 arg2 ...)` -- uninterpreted function application.
    UF { name: String, args: Vec<SmtExpr>, ret_sort: SmtSort },

    /// `(declare-fun name (arg_sorts...) ret_sort)` -- function declaration.
    ///
    /// This is not an expression per se but a declaration node used in
    /// query generation to emit the function signature.
    UFDecl { name: String, arg_sorts: Vec<SmtSort>, ret_sort: SmtSort },
}

// ---------------------------------------------------------------------------
// SmtExpr constructors (ergonomic builder API)
// ---------------------------------------------------------------------------

impl SmtExpr {
    /// Symbolic variable of given width.
    pub fn var(name: impl Into<String>, width: u32) -> Self {
        SmtExpr::Var { name: name.into(), width }
    }

    /// Bitvector constant.
    pub fn bv_const(value: u64, width: u32) -> Self {
        SmtExpr::BvConst { value: mask(value, width), width }
    }

    /// Boolean constant.
    pub fn bool_const(value: bool) -> Self {
        SmtExpr::BoolConst(value)
    }

    /// `bvadd`
    pub fn bvadd(self, other: Self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvAdd { lhs: Box::new(self), rhs: Box::new(other), width: w }
    }

    /// `bvsub`
    pub fn bvsub(self, other: Self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvSub { lhs: Box::new(self), rhs: Box::new(other), width: w }
    }

    /// `bvmul`
    pub fn bvmul(self, other: Self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvMul { lhs: Box::new(self), rhs: Box::new(other), width: w }
    }

    /// `bvsdiv`
    pub fn bvsdiv(self, other: Self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvSDiv { lhs: Box::new(self), rhs: Box::new(other), width: w }
    }

    /// `bvudiv`
    pub fn bvudiv(self, other: Self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvUDiv { lhs: Box::new(self), rhs: Box::new(other), width: w }
    }

    /// `bvneg`
    pub fn bvneg(self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvNeg { operand: Box::new(self), width: w }
    }

    /// `=` (equality)
    pub fn eq_expr(self, other: Self) -> Self {
        SmtExpr::Eq { lhs: Box::new(self), rhs: Box::new(other) }
    }

    /// `not`
    pub fn not_expr(self) -> Self {
        SmtExpr::Not { operand: Box::new(self) }
    }

    /// `bvslt` (signed less-than)
    pub fn bvslt(self, other: Self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvSlt { lhs: Box::new(self), rhs: Box::new(other), width: w }
    }

    /// `bvsge` (signed greater-or-equal)
    pub fn bvsge(self, other: Self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvSge { lhs: Box::new(self), rhs: Box::new(other), width: w }
    }

    /// `bvuge` (unsigned greater-or-equal)
    pub fn bvuge(self, other: Self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvUge { lhs: Box::new(self), rhs: Box::new(other), width: w }
    }

    /// `and`
    pub fn and_expr(self, other: Self) -> Self {
        SmtExpr::And { lhs: Box::new(self), rhs: Box::new(other) }
    }

    /// `or`
    pub fn or_expr(self, other: Self) -> Self {
        SmtExpr::Or { lhs: Box::new(self), rhs: Box::new(other) }
    }

    /// `bvsgt` (signed greater-than)
    pub fn bvsgt(self, other: Self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvSgt { lhs: Box::new(self), rhs: Box::new(other), width: w }
    }

    /// `bvsle` (signed less-or-equal)
    pub fn bvsle(self, other: Self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvSle { lhs: Box::new(self), rhs: Box::new(other), width: w }
    }

    /// `bvult` (unsigned less-than)
    pub fn bvult(self, other: Self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvUlt { lhs: Box::new(self), rhs: Box::new(other), width: w }
    }

    /// `bvugt` (unsigned greater-than)
    pub fn bvugt(self, other: Self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvUgt { lhs: Box::new(self), rhs: Box::new(other), width: w }
    }

    /// `bvule` (unsigned less-or-equal)
    pub fn bvule(self, other: Self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvUle { lhs: Box::new(self), rhs: Box::new(other), width: w }
    }

    /// `ite` (if-then-else)
    pub fn ite(cond: Self, then_expr: Self, else_expr: Self) -> Self {
        SmtExpr::Ite {
            cond: Box::new(cond),
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
        }
    }

    /// `bvand` -- bitwise AND.
    pub fn bvand(self, other: Self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvAnd { lhs: Box::new(self), rhs: Box::new(other), width: w }
    }

    /// `bvor` -- bitwise OR.
    pub fn bvor(self, other: Self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvOr { lhs: Box::new(self), rhs: Box::new(other), width: w }
    }

    /// `bvxor` -- bitwise XOR.
    pub fn bvxor(self, other: Self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvXor { lhs: Box::new(self), rhs: Box::new(other), width: w }
    }

    /// `bvshl` -- logical shift left.
    pub fn bvshl(self, other: Self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvShl { lhs: Box::new(self), rhs: Box::new(other), width: w }
    }

    /// `bvlshr` -- logical shift right.
    pub fn bvlshr(self, other: Self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvLshr { lhs: Box::new(self), rhs: Box::new(other), width: w }
    }

    /// `bvashr` -- arithmetic shift right.
    pub fn bvashr(self, other: Self) -> Self {
        let w = self.bv_width();
        SmtExpr::BvAshr { lhs: Box::new(self), rhs: Box::new(other), width: w }
    }

    /// `extract(high, low)` -- bit extraction.
    pub fn extract(self, high: u32, low: u32) -> Self {
        let result_width = high - low + 1;
        SmtExpr::Extract {
            high,
            low,
            operand: Box::new(self),
            width: result_width,
        }
    }

    /// `concat(hi, lo)` -- bitvector concatenation.
    ///
    /// The result has width `hi.width + lo.width`, with `hi` in the upper bits.
    pub fn concat(self, lo: Self) -> Self {
        let w = self.bv_width() + lo.bv_width();
        SmtExpr::Concat { hi: Box::new(self), lo: Box::new(lo), width: w }
    }

    /// `zero_extend(extra_bits)` -- zero-extend this bitvector.
    pub fn zero_ext(self, extra_bits: u32) -> Self {
        let w = self.bv_width() + extra_bits;
        SmtExpr::ZeroExtend { operand: Box::new(self), extra_bits, width: w }
    }

    /// `sign_extend(extra_bits)` -- sign-extend this bitvector.
    pub fn sign_ext(self, extra_bits: u32) -> Self {
        let w = self.bv_width() + extra_bits;
        SmtExpr::SignExtend { operand: Box::new(self), extra_bits, width: w }
    }

    // -- Array constructors --

    /// `(select array index)` -- read from array.
    pub fn select(array: Self, index: Self) -> Self {
        SmtExpr::Select {
            array: Box::new(array),
            index: Box::new(index),
        }
    }

    /// `(store array index value)` -- write to array.
    pub fn store(array: Self, index: Self, value: Self) -> Self {
        SmtExpr::Store {
            array: Box::new(array),
            index: Box::new(index),
            value: Box::new(value),
        }
    }

    /// `((as const ...) value)` -- constant array filled with `value`.
    pub fn const_array(index_sort: SmtSort, value: Self) -> Self {
        SmtExpr::ConstArray {
            index_sort,
            value: Box::new(value),
        }
    }

    // -- Floating-point constructors --

    /// Floating-point constant from raw IEEE 754 bits.
    pub fn fp_const(bits: u64, eb: u32, sb: u32) -> Self {
        SmtExpr::FPConst { bits, eb, sb }
    }

    /// FP32 constant from an f32 value.
    pub fn fp32_const(v: f32) -> Self {
        SmtExpr::FPConst { bits: v.to_bits() as u64, eb: 8, sb: 24 }
    }

    /// FP64 constant from an f64 value.
    pub fn fp64_const(v: f64) -> Self {
        SmtExpr::FPConst { bits: v.to_bits(), eb: 11, sb: 53 }
    }

    /// `(fp.add rm a b)` -- floating-point addition.
    pub fn fp_add(rm: RoundingMode, a: Self, b: Self) -> Self {
        SmtExpr::FPAdd { rm, lhs: Box::new(a), rhs: Box::new(b) }
    }

    /// `(fp.mul rm a b)` -- floating-point multiplication.
    pub fn fp_mul(rm: RoundingMode, a: Self, b: Self) -> Self {
        SmtExpr::FPMul { rm, lhs: Box::new(a), rhs: Box::new(b) }
    }

    /// `(fp.div rm a b)` -- floating-point division.
    pub fn fp_div(rm: RoundingMode, a: Self, b: Self) -> Self {
        SmtExpr::FPDiv { rm, lhs: Box::new(a), rhs: Box::new(b) }
    }

    /// `(fp.neg a)` -- floating-point negation.
    pub fn fp_neg(self) -> Self {
        SmtExpr::FPNeg { operand: Box::new(self) }
    }

    /// `(fp.eq a b)` -- floating-point equality (returns Bool).
    pub fn fp_eq(self, other: Self) -> Self {
        SmtExpr::FPEq { lhs: Box::new(self), rhs: Box::new(other) }
    }

    /// `(fp.lt a b)` -- floating-point less-than (returns Bool).
    pub fn fp_lt(self, other: Self) -> Self {
        SmtExpr::FPLt { lhs: Box::new(self), rhs: Box::new(other) }
    }

    // -- Uninterpreted function constructors --

    /// Uninterpreted function application.
    pub fn uf(name: impl Into<String>, args: Vec<Self>, ret_sort: SmtSort) -> Self {
        SmtExpr::UF { name: name.into(), args, ret_sort }
    }

    /// Uninterpreted function declaration.
    pub fn uf_decl(name: impl Into<String>, arg_sorts: Vec<SmtSort>, ret_sort: SmtSort) -> Self {
        SmtExpr::UFDecl { name: name.into(), arg_sorts, ret_sort }
    }

    /// Return the bitvector width of this expression, or an error for Bool-sorted expressions.
    pub fn try_bv_width(&self) -> Result<u32, SmtError> {
        match self {
            SmtExpr::Var { width, .. } => Ok(*width),
            SmtExpr::BvConst { width, .. } => Ok(*width),
            SmtExpr::BvAdd { width, .. } => Ok(*width),
            SmtExpr::BvSub { width, .. } => Ok(*width),
            SmtExpr::BvMul { width, .. } => Ok(*width),
            SmtExpr::BvSDiv { width, .. } => Ok(*width),
            SmtExpr::BvUDiv { width, .. } => Ok(*width),
            SmtExpr::BvNeg { width, .. } => Ok(*width),
            SmtExpr::Extract { width, .. } => Ok(*width),
            SmtExpr::BvAnd { width, .. } => Ok(*width),
            SmtExpr::BvOr { width, .. } => Ok(*width),
            SmtExpr::BvXor { width, .. } => Ok(*width),
            SmtExpr::BvShl { width, .. } => Ok(*width),
            SmtExpr::BvLshr { width, .. } => Ok(*width),
            SmtExpr::BvAshr { width, .. } => Ok(*width),
            SmtExpr::Concat { width, .. } => Ok(*width),
            SmtExpr::ZeroExtend { width, .. } => Ok(*width),
            SmtExpr::SignExtend { width, .. } => Ok(*width),
            SmtExpr::Ite { then_expr, .. } => then_expr.try_bv_width(),
            // Array select returns the element sort; if it's BV, extract width.
            SmtExpr::Select { array, .. } => {
                if let SmtSort::Array(_, elem_sort) = array.sort() {
                    elem_sort.bv_width().ok_or(SmtError::BoolHasNoWidth)
                } else {
                    Err(SmtError::BoolHasNoWidth)
                }
            }
            // UF returns its declared sort.
            SmtExpr::UF { ret_sort, .. } => ret_sort.bv_width().ok_or(SmtError::BoolHasNoWidth),
            SmtExpr::BoolConst(_)
            | SmtExpr::Eq { .. }
            | SmtExpr::Not { .. }
            | SmtExpr::BvSlt { .. }
            | SmtExpr::BvSge { .. }
            | SmtExpr::BvSgt { .. }
            | SmtExpr::BvSle { .. }
            | SmtExpr::BvUlt { .. }
            | SmtExpr::BvUge { .. }
            | SmtExpr::BvUgt { .. }
            | SmtExpr::BvUle { .. }
            | SmtExpr::And { .. }
            | SmtExpr::Or { .. }
            | SmtExpr::FPEq { .. }
            | SmtExpr::FPLt { .. } => Err(SmtError::BoolHasNoWidth),
            // FP / array / UF decl nodes have no BV width.
            SmtExpr::FPAdd { .. }
            | SmtExpr::FPMul { .. }
            | SmtExpr::FPDiv { .. }
            | SmtExpr::FPNeg { .. }
            | SmtExpr::FPConst { .. }
            | SmtExpr::Store { .. }
            | SmtExpr::ConstArray { .. }
            | SmtExpr::UFDecl { .. } => Err(SmtError::BoolHasNoWidth),
        }
    }

    /// Return the bitvector width of this expression.
    ///
    /// # Panics
    ///
    /// Panics if called on a Bool-sorted expression (comparisons, logical ops).
    /// Callers that may encounter Bool expressions should use [`try_bv_width`]
    /// instead.
    pub fn bv_width(&self) -> u32 {
        self.try_bv_width().expect("bv_width called on Bool-sorted expression; use try_bv_width() for fallible access")
    }

    /// Return the sort of this expression.
    pub fn sort(&self) -> SmtSort {
        match self {
            SmtExpr::BoolConst(_)
            | SmtExpr::Eq { .. }
            | SmtExpr::Not { .. }
            | SmtExpr::BvSlt { .. }
            | SmtExpr::BvSge { .. }
            | SmtExpr::BvSgt { .. }
            | SmtExpr::BvSle { .. }
            | SmtExpr::BvUlt { .. }
            | SmtExpr::BvUge { .. }
            | SmtExpr::BvUgt { .. }
            | SmtExpr::BvUle { .. }
            | SmtExpr::And { .. }
            | SmtExpr::Or { .. }
            | SmtExpr::FPEq { .. }
            | SmtExpr::FPLt { .. } => SmtSort::Bool,
            // Floating-point expressions
            SmtExpr::FPAdd { lhs, .. }
            | SmtExpr::FPMul { lhs, .. }
            | SmtExpr::FPDiv { lhs, .. } => lhs.sort(),
            SmtExpr::FPNeg { operand } => operand.sort(),
            SmtExpr::FPConst { eb, sb, .. } => SmtSort::FloatingPoint(*eb, *sb),
            // Array expressions
            SmtExpr::Store { array, .. } => array.sort(),
            SmtExpr::ConstArray { index_sort, value } => {
                SmtSort::Array(Box::new(index_sort.clone()), Box::new(value.sort()))
            }
            SmtExpr::Select { array, .. } => {
                // Element sort of the array
                if let SmtSort::Array(_, elem_sort) = array.sort() {
                    *elem_sort
                } else {
                    // Fallback: shouldn't happen for well-typed expressions.
                    SmtSort::Bool
                }
            }
            // Uninterpreted functions
            SmtExpr::UF { ret_sort, .. } => ret_sort.clone(),
            SmtExpr::UFDecl { ret_sort, .. } => ret_sort.clone(),
            // All BV expressions
            _ => SmtSort::BitVec(self.bv_width()),
        }
    }

    /// Collect all free variable names referenced in this expression.
    pub fn free_vars(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_vars(&mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_vars(&self, vars: &mut Vec<String>) {
        match self {
            SmtExpr::Var { name, .. } => vars.push(name.clone()),
            SmtExpr::BvConst { .. } | SmtExpr::BoolConst(_) | SmtExpr::FPConst { .. } => {}
            SmtExpr::BvAdd { lhs, rhs, .. }
            | SmtExpr::BvSub { lhs, rhs, .. }
            | SmtExpr::BvMul { lhs, rhs, .. }
            | SmtExpr::BvSDiv { lhs, rhs, .. }
            | SmtExpr::BvUDiv { lhs, rhs, .. }
            | SmtExpr::BvAnd { lhs, rhs, .. }
            | SmtExpr::BvOr { lhs, rhs, .. }
            | SmtExpr::BvXor { lhs, rhs, .. }
            | SmtExpr::BvShl { lhs, rhs, .. }
            | SmtExpr::BvLshr { lhs, rhs, .. }
            | SmtExpr::BvAshr { lhs, rhs, .. }
            | SmtExpr::Eq { lhs, rhs }
            | SmtExpr::BvSlt { lhs, rhs, .. }
            | SmtExpr::BvSge { lhs, rhs, .. }
            | SmtExpr::BvSgt { lhs, rhs, .. }
            | SmtExpr::BvSle { lhs, rhs, .. }
            | SmtExpr::BvUlt { lhs, rhs, .. }
            | SmtExpr::BvUge { lhs, rhs, .. }
            | SmtExpr::BvUgt { lhs, rhs, .. }
            | SmtExpr::BvUle { lhs, rhs, .. }
            | SmtExpr::And { lhs, rhs }
            | SmtExpr::Or { lhs, rhs }
            | SmtExpr::FPEq { lhs, rhs }
            | SmtExpr::FPLt { lhs, rhs } => {
                lhs.collect_vars(vars);
                rhs.collect_vars(vars);
            }
            SmtExpr::FPAdd { lhs, rhs, .. }
            | SmtExpr::FPMul { lhs, rhs, .. }
            | SmtExpr::FPDiv { lhs, rhs, .. } => {
                lhs.collect_vars(vars);
                rhs.collect_vars(vars);
            }
            SmtExpr::BvNeg { operand, .. }
            | SmtExpr::Not { operand }
            | SmtExpr::Extract { operand, .. }
            | SmtExpr::ZeroExtend { operand, .. }
            | SmtExpr::SignExtend { operand, .. }
            | SmtExpr::FPNeg { operand } => {
                operand.collect_vars(vars);
            }
            SmtExpr::Concat { hi, lo, .. } => {
                hi.collect_vars(vars);
                lo.collect_vars(vars);
            }
            SmtExpr::Ite { cond, then_expr, else_expr } => {
                cond.collect_vars(vars);
                then_expr.collect_vars(vars);
                else_expr.collect_vars(vars);
            }
            SmtExpr::Select { array, index } => {
                array.collect_vars(vars);
                index.collect_vars(vars);
            }
            SmtExpr::Store { array, index, value } => {
                array.collect_vars(vars);
                index.collect_vars(vars);
                value.collect_vars(vars);
            }
            SmtExpr::ConstArray { value, .. } => {
                value.collect_vars(vars);
            }
            SmtExpr::UF { args, .. } => {
                for arg in args {
                    arg.collect_vars(vars);
                }
            }
            SmtExpr::UFDecl { .. } => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Concrete evaluation
// ---------------------------------------------------------------------------

/// Evaluation result: bitvector, boolean, floating-point, or array.
#[derive(Debug, Clone, PartialEq)]
pub enum EvalResult {
    Bv(u64),
    /// Wide bitvector (65-128 bits). Used for 128-bit NEON vector intermediates.
    ///
    /// NEON operations produce 128-bit results via `Concat`. The lane-extraction
    /// pattern (`Extract` after `Concat`) reduces back to <= 64 bits for final
    /// comparison. `Bv128` exists to carry the intermediate without overflow.
    Bv128(u128),
    Bool(bool),
    /// Floating-point value stored as f64 (sufficient for FP16/FP32/FP64).
    Float(f64),
    /// Array value: maps bitvector index (u64) to EvalResult.
    /// The default value is used for indices not in the map.
    Array {
        entries: HashMap<u64, Box<EvalResult>>,
        default: Box<EvalResult>,
    },
}

impl Eq for EvalResult {}

impl EvalResult {
    pub fn as_u64(self) -> u64 {
        match self {
            EvalResult::Bv(v) => v,
            EvalResult::Bv128(v) => v as u64,
            EvalResult::Bool(b) => b as u64,
            EvalResult::Float(f) => f.to_bits(),
            EvalResult::Array { .. } => 0, // arrays don't have a scalar representation
        }
    }

    /// Convert to u128. `Bv` values are zero-extended.
    pub fn as_u128(self) -> u128 {
        match self {
            EvalResult::Bv(v) => v as u128,
            EvalResult::Bv128(v) => v,
            EvalResult::Bool(b) => b as u128,
            EvalResult::Float(f) => f.to_bits() as u128,
            EvalResult::Array { .. } => 0,
        }
    }

    pub fn as_bool(self) -> bool {
        match self {
            EvalResult::Bool(b) => b,
            EvalResult::Bv(v) => v != 0,
            EvalResult::Bv128(v) => v != 0,
            EvalResult::Float(f) => f != 0.0,
            EvalResult::Array { .. } => false,
        }
    }

    pub fn as_f64(&self) -> f64 {
        match self {
            EvalResult::Float(f) => *f,
            EvalResult::Bv(v) => *v as f64,
            EvalResult::Bv128(v) => *v as f64,
            EvalResult::Bool(b) => if *b { 1.0 } else { 0.0 },
            EvalResult::Array { .. } => 0.0,
        }
    }
}

/// Mask a value to the given bitvector width.
pub fn mask(value: u64, width: u32) -> u64 {
    if width >= 64 {
        value
    } else {
        value & ((1u64 << width) - 1)
    }
}

/// Mask a 128-bit value to the given bitvector width.
fn mask128(value: u128, width: u32) -> u128 {
    if width >= 128 {
        value
    } else {
        value & ((1u128 << width) - 1)
    }
}

/// Sign-extend a `width`-bit value stored in a u64 to i64.
fn sign_extend(value: u64, width: u32) -> i64 {
    if width == 0 {
        return 0;
    }
    if width >= 64 {
        return value as i64;
    }
    let shift = 64 - width;
    ((value << shift) as i64) >> shift
}

impl SmtExpr {
    /// Evaluate this expression under the given variable assignment (fallible).
    ///
    /// Variables map name -> u64 value (already masked to width).
    /// Returns `Err(SmtError::UndefinedVariable)` if a variable is not found.
    pub fn try_eval(&self, env: &HashMap<String, u64>) -> Result<EvalResult, SmtError> {
        match self {
            SmtExpr::Var { name, width } => {
                let v = env.get(name).ok_or_else(|| {
                    SmtError::UndefinedVariable(name.clone())
                })?;
                Ok(EvalResult::Bv(mask(*v, *width)))
            }
            SmtExpr::BvConst { value, width } => {
                Ok(EvalResult::Bv(mask(*value, *width)))
            }
            SmtExpr::BoolConst(b) => Ok(EvalResult::Bool(*b)),

            SmtExpr::BvAdd { lhs, rhs, width } => {
                let a = lhs.try_eval(env)?.as_u64();
                let b = rhs.try_eval(env)?.as_u64();
                Ok(EvalResult::Bv(mask(a.wrapping_add(b), *width)))
            }
            SmtExpr::BvSub { lhs, rhs, width } => {
                let a = lhs.try_eval(env)?.as_u64();
                let b = rhs.try_eval(env)?.as_u64();
                Ok(EvalResult::Bv(mask(a.wrapping_sub(b), *width)))
            }
            SmtExpr::BvMul { lhs, rhs, width } => {
                let a = lhs.try_eval(env)?.as_u64();
                let b = rhs.try_eval(env)?.as_u64();
                Ok(EvalResult::Bv(mask(a.wrapping_mul(b), *width)))
            }
            SmtExpr::BvSDiv { lhs, rhs, width } => {
                let a = sign_extend(lhs.try_eval(env)?.as_u64(), *width);
                let b = sign_extend(rhs.try_eval(env)?.as_u64(), *width);
                if b == 0 {
                    // SMT-LIB: bvsdiv by zero is defined (returns all-ones
                    // for positive dividend, etc.). For verification we gate
                    // on b != 0 as a precondition, but we still need a defined
                    // value here. Return 0 as sentinel.
                    Ok(EvalResult::Bv(0))
                } else if a == i64::MIN && b == -1 && *width == 64 {
                    // Overflow: INT_MIN / -1.
                    Ok(EvalResult::Bv(mask(a as u64, *width)))
                } else {
                    let result = a.wrapping_div(b);
                    Ok(EvalResult::Bv(mask(result as u64, *width)))
                }
            }
            SmtExpr::BvUDiv { lhs, rhs, width } => {
                let a = lhs.try_eval(env)?.as_u64();
                let b = rhs.try_eval(env)?.as_u64();
                if b == 0 {
                    Ok(EvalResult::Bv(0)) // sentinel, gated by precondition
                } else {
                    Ok(EvalResult::Bv(mask(a / b, *width)))
                }
            }
            SmtExpr::BvAnd { lhs, rhs, width } => {
                if *width > 64 {
                    let a = lhs.try_eval(env)?.as_u128();
                    let b = rhs.try_eval(env)?.as_u128();
                    Ok(EvalResult::Bv128(mask128(a & b, *width)))
                } else {
                    let a = lhs.try_eval(env)?.as_u64();
                    let b = rhs.try_eval(env)?.as_u64();
                    Ok(EvalResult::Bv(mask(a & b, *width)))
                }
            }
            SmtExpr::BvOr { lhs, rhs, width } => {
                if *width > 64 {
                    let a = lhs.try_eval(env)?.as_u128();
                    let b = rhs.try_eval(env)?.as_u128();
                    Ok(EvalResult::Bv128(mask128(a | b, *width)))
                } else {
                    let a = lhs.try_eval(env)?.as_u64();
                    let b = rhs.try_eval(env)?.as_u64();
                    Ok(EvalResult::Bv(mask(a | b, *width)))
                }
            }
            SmtExpr::BvXor { lhs, rhs, width } => {
                if *width > 64 {
                    let a = lhs.try_eval(env)?.as_u128();
                    let b = rhs.try_eval(env)?.as_u128();
                    Ok(EvalResult::Bv128(mask128(a ^ b, *width)))
                } else {
                    let a = lhs.try_eval(env)?.as_u64();
                    let b = rhs.try_eval(env)?.as_u64();
                    Ok(EvalResult::Bv(mask(a ^ b, *width)))
                }
            }
            SmtExpr::BvShl { lhs, rhs, width } => {
                let a = lhs.try_eval(env)?.as_u64();
                let b = rhs.try_eval(env)?.as_u64();
                // SMT-LIB: if shift amount >= width, result is 0.
                if b >= *width as u64 {
                    Ok(EvalResult::Bv(0))
                } else {
                    Ok(EvalResult::Bv(mask(a << b, *width)))
                }
            }
            SmtExpr::BvLshr { lhs, rhs, width } => {
                let a = lhs.try_eval(env)?.as_u64();
                let b = rhs.try_eval(env)?.as_u64();
                if b >= *width as u64 {
                    Ok(EvalResult::Bv(0))
                } else {
                    Ok(EvalResult::Bv(mask(a >> b, *width)))
                }
            }
            SmtExpr::BvAshr { lhs, rhs, width } => {
                let a = sign_extend(lhs.try_eval(env)?.as_u64(), *width);
                let b = rhs.try_eval(env)?.as_u64();
                if b >= *width as u64 {
                    // Sign-fill: all 1s if negative, all 0s if positive.
                    if a < 0 {
                        Ok(EvalResult::Bv(mask(u64::MAX, *width)))
                    } else {
                        Ok(EvalResult::Bv(0))
                    }
                } else {
                    Ok(EvalResult::Bv(mask((a >> b) as u64, *width)))
                }
            }
            SmtExpr::BvNeg { operand, width } => {
                let a = operand.try_eval(env)?.as_u64();
                // Two's complement negation = wrapping negate.
                Ok(EvalResult::Bv(mask((!a).wrapping_add(1), *width)))
            }

            SmtExpr::Eq { lhs, rhs } => {
                let a = lhs.try_eval(env)?;
                let b = rhs.try_eval(env)?;
                Ok(EvalResult::Bool(a == b))
            }
            SmtExpr::Not { operand } => {
                Ok(EvalResult::Bool(!operand.try_eval(env)?.as_bool()))
            }
            SmtExpr::BvSlt { lhs, rhs, width } => {
                let a = sign_extend(lhs.try_eval(env)?.as_u64(), *width);
                let b = sign_extend(rhs.try_eval(env)?.as_u64(), *width);
                Ok(EvalResult::Bool(a < b))
            }
            SmtExpr::BvSge { lhs, rhs, width } => {
                let a = sign_extend(lhs.try_eval(env)?.as_u64(), *width);
                let b = sign_extend(rhs.try_eval(env)?.as_u64(), *width);
                Ok(EvalResult::Bool(a >= b))
            }
            SmtExpr::BvUge { lhs, rhs, .. } => {
                let a = lhs.try_eval(env)?.as_u64();
                let b = rhs.try_eval(env)?.as_u64();
                Ok(EvalResult::Bool(a >= b))
            }
            SmtExpr::BvSgt { lhs, rhs, width } => {
                let a = sign_extend(lhs.try_eval(env)?.as_u64(), *width);
                let b = sign_extend(rhs.try_eval(env)?.as_u64(), *width);
                Ok(EvalResult::Bool(a > b))
            }
            SmtExpr::BvSle { lhs, rhs, width } => {
                let a = sign_extend(lhs.try_eval(env)?.as_u64(), *width);
                let b = sign_extend(rhs.try_eval(env)?.as_u64(), *width);
                Ok(EvalResult::Bool(a <= b))
            }
            SmtExpr::BvUlt { lhs, rhs, .. } => {
                let a = lhs.try_eval(env)?.as_u64();
                let b = rhs.try_eval(env)?.as_u64();
                Ok(EvalResult::Bool(a < b))
            }
            SmtExpr::BvUgt { lhs, rhs, .. } => {
                let a = lhs.try_eval(env)?.as_u64();
                let b = rhs.try_eval(env)?.as_u64();
                Ok(EvalResult::Bool(a > b))
            }
            SmtExpr::BvUle { lhs, rhs, .. } => {
                let a = lhs.try_eval(env)?.as_u64();
                let b = rhs.try_eval(env)?.as_u64();
                Ok(EvalResult::Bool(a <= b))
            }
            SmtExpr::And { lhs, rhs } => {
                Ok(EvalResult::Bool(lhs.try_eval(env)?.as_bool() && rhs.try_eval(env)?.as_bool()))
            }
            SmtExpr::Or { lhs, rhs } => {
                Ok(EvalResult::Bool(lhs.try_eval(env)?.as_bool() || rhs.try_eval(env)?.as_bool()))
            }
            SmtExpr::Ite { cond, then_expr, else_expr } => {
                if cond.try_eval(env)?.as_bool() {
                    then_expr.try_eval(env)
                } else {
                    else_expr.try_eval(env)
                }
            }
            SmtExpr::Extract { high, low, operand, width } => {
                // Use u128 for extraction to handle wide intermediates (e.g., 128-bit NEON vectors).
                let v = operand.try_eval(env)?.as_u128();
                let extracted = (v >> low) & mask128(u128::MAX, *width);
                let _ = high; // used in width calculation
                // Result always fits in u64 since extract width <= 64 for valid NEON lanes.
                Ok(EvalResult::Bv(extracted as u64))
            }
            SmtExpr::Concat { hi, lo, width } => {
                let hi_val = hi.try_eval(env)?.as_u128();
                let lo_val = lo.try_eval(env)?.as_u128();
                let lo_width = lo.bv_width();
                // Place hi bits above lo bits using u128 to avoid overflow.
                let result = mask128((hi_val << lo_width) | lo_val, *width);
                if *width <= 64 {
                    Ok(EvalResult::Bv(result as u64))
                } else {
                    Ok(EvalResult::Bv128(result))
                }
            }
            SmtExpr::ZeroExtend { operand, .. } => {
                // Value is already stored in u64 with upper bits zero.
                let v = operand.try_eval(env)?.as_u64();
                Ok(EvalResult::Bv(v))
            }
            SmtExpr::SignExtend { operand, extra_bits, width } => {
                let v = operand.try_eval(env)?.as_u64();
                let src_width = *width - *extra_bits;
                let extended = sign_extend(v, src_width) as u64;
                Ok(EvalResult::Bv(mask(extended, *width)))
            }

            // -- Array evaluation --

            SmtExpr::ConstArray { value, .. } => {
                let v = value.try_eval(env)?;
                Ok(EvalResult::Array {
                    entries: HashMap::new(),
                    default: Box::new(v),
                })
            }
            SmtExpr::Select { array, index } => {
                let arr = array.try_eval(env)?;
                let idx = index.try_eval(env)?.as_u64();
                match arr {
                    EvalResult::Array { entries, default } => {
                        Ok(entries.get(&idx).map(|v| *v.clone()).unwrap_or(*default))
                    }
                    _ => Err(SmtError::EvalError("select on non-array value".to_string())),
                }
            }
            SmtExpr::Store { array, index, value } => {
                let arr = array.try_eval(env)?;
                let idx = index.try_eval(env)?.as_u64();
                let val = value.try_eval(env)?;
                match arr {
                    EvalResult::Array { mut entries, default } => {
                        entries.insert(idx, Box::new(val));
                        Ok(EvalResult::Array { entries, default })
                    }
                    _ => Err(SmtError::EvalError("store on non-array value".to_string())),
                }
            }

            // -- Floating-point evaluation (using Rust native f32/f64) --

            SmtExpr::FPConst { bits, eb, sb } => {
                let f = if *eb == 8 && *sb == 24 {
                    // FP32: interpret lower 32 bits as f32
                    f32::from_bits(*bits as u32) as f64
                } else {
                    // FP64 or other: interpret as f64
                    f64::from_bits(*bits)
                };
                Ok(EvalResult::Float(f))
            }
            SmtExpr::FPAdd { lhs, rhs, .. } => {
                let a = lhs.try_eval(env)?.as_f64();
                let b = rhs.try_eval(env)?.as_f64();
                Ok(EvalResult::Float(a + b))
            }
            SmtExpr::FPMul { lhs, rhs, .. } => {
                let a = lhs.try_eval(env)?.as_f64();
                let b = rhs.try_eval(env)?.as_f64();
                Ok(EvalResult::Float(a * b))
            }
            SmtExpr::FPDiv { lhs, rhs, .. } => {
                let a = lhs.try_eval(env)?.as_f64();
                let b = rhs.try_eval(env)?.as_f64();
                Ok(EvalResult::Float(a / b))
            }
            SmtExpr::FPNeg { operand } => {
                let a = operand.try_eval(env)?.as_f64();
                Ok(EvalResult::Float(-a))
            }
            SmtExpr::FPEq { lhs, rhs } => {
                let a = lhs.try_eval(env)?.as_f64();
                let b = rhs.try_eval(env)?.as_f64();
                Ok(EvalResult::Bool(a == b))
            }
            SmtExpr::FPLt { lhs, rhs } => {
                let a = lhs.try_eval(env)?.as_f64();
                let b = rhs.try_eval(env)?.as_f64();
                Ok(EvalResult::Bool(a < b))
            }

            // -- Uninterpreted functions --
            // UF evaluation is not meaningful in concrete evaluation (they are
            // uninterpreted). Return an error for now; real verification uses
            // the SMT solver for UF reasoning.
            SmtExpr::UF { name, .. } => {
                Err(SmtError::EvalError(format!(
                    "cannot concretely evaluate uninterpreted function '{}'", name
                )))
            }
            SmtExpr::UFDecl { name, .. } => {
                Err(SmtError::EvalError(format!(
                    "cannot evaluate UF declaration '{}'", name
                )))
            }
        }
    }

    /// Evaluate this expression under the given variable assignment.
    ///
    /// Variables map name -> u64 value (already masked to width).
    ///
    /// # Panics
    ///
    /// Panics if a variable is not found in the environment. Use [`try_eval`]
    /// for fallible evaluation.
    pub fn eval(&self, env: &HashMap<String, u64>) -> EvalResult {
        self.try_eval(env).expect("SmtExpr::eval failed; use try_eval() for fallible evaluation")
    }
}

// ---------------------------------------------------------------------------
// Display (SMT-LIB2 format for debugging)
// ---------------------------------------------------------------------------

impl fmt::Display for SmtExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtExpr::Var { name, .. } => write!(f, "{}", name),
            SmtExpr::BvConst { value, width } => {
                write!(f, "(_ bv{} {})", value, width)
            }
            SmtExpr::BoolConst(b) => write!(f, "{}", b),
            SmtExpr::BvAdd { lhs, rhs, .. } => write!(f, "(bvadd {} {})", lhs, rhs),
            SmtExpr::BvSub { lhs, rhs, .. } => write!(f, "(bvsub {} {})", lhs, rhs),
            SmtExpr::BvMul { lhs, rhs, .. } => write!(f, "(bvmul {} {})", lhs, rhs),
            SmtExpr::BvSDiv { lhs, rhs, .. } => write!(f, "(bvsdiv {} {})", lhs, rhs),
            SmtExpr::BvUDiv { lhs, rhs, .. } => write!(f, "(bvudiv {} {})", lhs, rhs),
            SmtExpr::BvAnd { lhs, rhs, .. } => write!(f, "(bvand {} {})", lhs, rhs),
            SmtExpr::BvOr { lhs, rhs, .. } => write!(f, "(bvor {} {})", lhs, rhs),
            SmtExpr::BvXor { lhs, rhs, .. } => write!(f, "(bvxor {} {})", lhs, rhs),
            SmtExpr::BvShl { lhs, rhs, .. } => write!(f, "(bvshl {} {})", lhs, rhs),
            SmtExpr::BvLshr { lhs, rhs, .. } => write!(f, "(bvlshr {} {})", lhs, rhs),
            SmtExpr::BvAshr { lhs, rhs, .. } => write!(f, "(bvashr {} {})", lhs, rhs),
            SmtExpr::BvNeg { operand, .. } => write!(f, "(bvneg {})", operand),
            SmtExpr::Eq { lhs, rhs } => write!(f, "(= {} {})", lhs, rhs),
            SmtExpr::Not { operand } => write!(f, "(not {})", operand),
            SmtExpr::BvSlt { lhs, rhs, .. } => write!(f, "(bvslt {} {})", lhs, rhs),
            SmtExpr::BvSge { lhs, rhs, .. } => write!(f, "(bvsge {} {})", lhs, rhs),
            SmtExpr::BvSgt { lhs, rhs, .. } => write!(f, "(bvsgt {} {})", lhs, rhs),
            SmtExpr::BvSle { lhs, rhs, .. } => write!(f, "(bvsle {} {})", lhs, rhs),
            SmtExpr::BvUlt { lhs, rhs, .. } => write!(f, "(bvult {} {})", lhs, rhs),
            SmtExpr::BvUge { lhs, rhs, .. } => write!(f, "(bvuge {} {})", lhs, rhs),
            SmtExpr::BvUgt { lhs, rhs, .. } => write!(f, "(bvugt {} {})", lhs, rhs),
            SmtExpr::BvUle { lhs, rhs, .. } => write!(f, "(bvule {} {})", lhs, rhs),
            SmtExpr::And { lhs, rhs } => write!(f, "(and {} {})", lhs, rhs),
            SmtExpr::Or { lhs, rhs } => write!(f, "(or {} {})", lhs, rhs),
            SmtExpr::Ite { cond, then_expr, else_expr } => {
                write!(f, "(ite {} {} {})", cond, then_expr, else_expr)
            }
            SmtExpr::Extract { high, low, operand, .. } => {
                write!(f, "((_ extract {} {}) {})", high, low, operand)
            }
            SmtExpr::Concat { hi, lo, .. } => {
                write!(f, "(concat {} {})", hi, lo)
            }
            SmtExpr::ZeroExtend { operand, extra_bits, .. } => {
                write!(f, "((_ zero_extend {}) {})", extra_bits, operand)
            }
            SmtExpr::SignExtend { operand, extra_bits, .. } => {
                write!(f, "((_ sign_extend {}) {})", extra_bits, operand)
            }
            // Array operations
            SmtExpr::Select { array, index } => {
                write!(f, "(select {} {})", array, index)
            }
            SmtExpr::Store { array, index, value } => {
                write!(f, "(store {} {} {})", array, index, value)
            }
            SmtExpr::ConstArray { index_sort, value } => {
                // SMT-LIB2: ((as const (Array idx_sort elem_sort)) value)
                let elem_sort = value.sort();
                let array_sort = SmtSort::Array(
                    Box::new(index_sort.clone()),
                    Box::new(elem_sort),
                );
                write!(f, "((as const {}) {})", array_sort, value)
            }
            // Floating-point operations
            SmtExpr::FPConst { bits, eb, sb } => {
                // Emit as fp literal with bitvector decomposition
                let total = eb + sb;
                write!(f, "(fp #b{} #b{} #b{})",
                    // sign bit
                    if bits >> (total - 1) & 1 == 1 { "1" } else { "0" },
                    // exponent bits
                    format!("{:0>width$b}", (bits >> (sb - 1)) & ((1u64 << eb) - 1), width = *eb as usize),
                    // significand bits (without implicit bit)
                    format!("{:0>width$b}", bits & ((1u64 << (sb - 1)) - 1), width = (*sb - 1) as usize),
                )
            }
            SmtExpr::FPAdd { rm, lhs, rhs } => {
                write!(f, "(fp.add {} {} {})", rm, lhs, rhs)
            }
            SmtExpr::FPMul { rm, lhs, rhs } => {
                write!(f, "(fp.mul {} {} {})", rm, lhs, rhs)
            }
            SmtExpr::FPDiv { rm, lhs, rhs } => {
                write!(f, "(fp.div {} {} {})", rm, lhs, rhs)
            }
            SmtExpr::FPNeg { operand } => {
                write!(f, "(fp.neg {})", operand)
            }
            SmtExpr::FPEq { lhs, rhs } => {
                write!(f, "(fp.eq {} {})", lhs, rhs)
            }
            SmtExpr::FPLt { lhs, rhs } => {
                write!(f, "(fp.lt {} {})", lhs, rhs)
            }
            // Uninterpreted functions
            SmtExpr::UF { name, args, .. } => {
                if args.is_empty() {
                    write!(f, "{}", name)
                } else {
                    write!(f, "({}", name)?;
                    for arg in args {
                        write!(f, " {}", arg)?;
                    }
                    write!(f, ")")
                }
            }
            SmtExpr::UFDecl { name, arg_sorts, ret_sort } => {
                write!(f, "(declare-fun {} (", name)?;
                for (i, sort) in arg_sorts.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", sort)?;
                }
                write!(f, ") {})", ret_sort)
            }
        }
    }
}

impl fmt::Display for RoundingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RoundingMode::RNE => write!(f, "RNE"),
            RoundingMode::RNA => write!(f, "RNA"),
            RoundingMode::RTP => write!(f, "RTP"),
            RoundingMode::RTN => write!(f, "RTN"),
            RoundingMode::RTZ => write!(f, "RTZ"),
        }
    }
}

impl fmt::Display for SmtSort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtSort::BitVec(w) => write!(f, "(_ BitVec {})", w),
            SmtSort::Bool => write!(f, "Bool"),
            SmtSort::Array(idx, elem) => write!(f, "(Array {} {})", idx, elem),
            SmtSort::FloatingPoint(eb, sb) => write!(f, "(_ FloatingPoint {} {})", eb, sb),
        }
    }
}

// ---------------------------------------------------------------------------
// NEON / SIMD lane helpers
// ---------------------------------------------------------------------------

/// NEON vector arrangement: describes element size and lane count.
///
/// ARM DDI 0487: "The arrangement specifier determines the size of elements
/// and the number of lanes in the vector register."
///
/// Naming convention follows ARM assembly syntax:
/// - `8B` = 8 lanes of 8-bit bytes (64-bit, lower half of V register)
/// - `16B` = 16 lanes of 8-bit bytes (128-bit, full V register)
/// - `4H` = 4 lanes of 16-bit halfwords (64-bit)
/// - `8H` = 8 lanes of 16-bit halfwords (128-bit)
/// - `2S` = 2 lanes of 32-bit words (64-bit)
/// - `4S` = 4 lanes of 32-bit words (128-bit)
/// - `2D` = 2 lanes of 64-bit doublewords (128-bit)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorArrangement {
    B8,  // 8 x 8-bit (64-bit total)
    B16, // 16 x 8-bit (128-bit total)
    H4,  // 4 x 16-bit (64-bit total)
    H8,  // 8 x 16-bit (128-bit total)
    S2,  // 2 x 32-bit (64-bit total)
    S4,  // 4 x 32-bit (128-bit total)
    D2,  // 2 x 64-bit (128-bit total)
}

impl VectorArrangement {
    /// Number of lanes in this arrangement.
    pub fn lane_count(self) -> u32 {
        match self {
            VectorArrangement::B8 => 8,
            VectorArrangement::B16 => 16,
            VectorArrangement::H4 => 4,
            VectorArrangement::H8 => 8,
            VectorArrangement::S2 => 2,
            VectorArrangement::S4 => 4,
            VectorArrangement::D2 => 2,
        }
    }

    /// Bit-width of each lane element.
    pub fn lane_bits(self) -> u32 {
        match self {
            VectorArrangement::B8 | VectorArrangement::B16 => 8,
            VectorArrangement::H4 | VectorArrangement::H8 => 16,
            VectorArrangement::S2 | VectorArrangement::S4 => 32,
            VectorArrangement::D2 => 64,
        }
    }

    /// Total bit-width of the vector (64 or 128).
    pub fn total_bits(self) -> u32 {
        self.lane_count() * self.lane_bits()
    }
}

/// Extract lane `idx` from a vector expression.
///
/// Returns `expr[hi:lo]` where `lo = idx * lane_bits` and `hi = lo + lane_bits - 1`.
///
/// # Panics
///
/// Panics if `idx >= arrangement.lane_count()`.
pub fn lane_extract(expr: &SmtExpr, arrangement: VectorArrangement, idx: u32) -> SmtExpr {
    assert!(idx < arrangement.lane_count(), "lane index out of bounds");
    let lane_bits = arrangement.lane_bits();
    let lo = idx * lane_bits;
    let hi = lo + lane_bits - 1;
    expr.clone().extract(hi, lo)
}

/// Build a vector from individual lane expressions by concatenating them.
///
/// `lanes[0]` is the least-significant lane (lane 0), `lanes[last]` is the
/// most-significant lane. Each lane expression must have width `arrangement.lane_bits()`.
///
/// # Panics
///
/// Panics if `lanes.len() != arrangement.lane_count()`.
pub fn concat_lanes(lanes: &[SmtExpr], arrangement: VectorArrangement) -> SmtExpr {
    assert_eq!(
        lanes.len() as u32,
        arrangement.lane_count(),
        "wrong number of lanes for arrangement"
    );
    // Build from lane 0 (LSB) upward: concat(lane[n-1], concat(lane[n-2], ... concat(lane[1], lane[0])))
    let mut result = lanes[0].clone();
    for lane in &lanes[1..] {
        // Each `concat` places the new lane in the higher bits.
        result = lane.clone().concat(result);
    }
    result
}

/// Insert a lane value into a vector, returning the modified vector.
///
/// Decomposes the vector into lanes, replaces lane `idx` with `new_lane`,
/// and reassembles. This is the symbolic equivalent of `INS Vd.T[idx], Vn.T[0]`.
///
/// # Panics
///
/// Panics if `idx >= arrangement.lane_count()`.
pub fn lane_insert(
    vec: &SmtExpr,
    arrangement: VectorArrangement,
    idx: u32,
    new_lane: SmtExpr,
) -> SmtExpr {
    assert!(idx < arrangement.lane_count(), "lane index out of bounds");
    let n = arrangement.lane_count();
    let lanes: Vec<SmtExpr> = (0..n)
        .map(|i| {
            if i == idx {
                new_lane.clone()
            } else {
                lane_extract(vec, arrangement, i)
            }
        })
        .collect();
    concat_lanes(&lanes, arrangement)
}

/// Apply a binary operation lane-wise to two vector expressions.
///
/// For each lane `i`, extracts the lane from both operands, applies `op`, and
/// reassembles the result. This is the core pattern for NEON integer SIMD ops.
pub fn map_lanes_binary<F>(
    lhs: &SmtExpr,
    rhs: &SmtExpr,
    arrangement: VectorArrangement,
    op: F,
) -> SmtExpr
where
    F: Fn(SmtExpr, SmtExpr) -> SmtExpr,
{
    let n = arrangement.lane_count();
    let lanes: Vec<SmtExpr> = (0..n)
        .map(|i| {
            let a = lane_extract(lhs, arrangement, i);
            let b = lane_extract(rhs, arrangement, i);
            op(a, b)
        })
        .collect();
    concat_lanes(&lanes, arrangement)
}

/// Apply a unary operation lane-wise to a vector expression.
pub fn map_lanes_unary<F>(
    operand: &SmtExpr,
    arrangement: VectorArrangement,
    op: F,
) -> SmtExpr
where
    F: Fn(SmtExpr) -> SmtExpr,
{
    let n = arrangement.lane_count();
    let lanes: Vec<SmtExpr> = (0..n)
        .map(|i| {
            let a = lane_extract(operand, arrangement, i);
            op(a)
        })
        .collect();
    concat_lanes(&lanes, arrangement)
}

/// Apply a binary operation where the second operand is a constant (e.g., shift immediate).
pub fn map_lanes_binary_imm<F>(
    lhs: &SmtExpr,
    imm: u64,
    arrangement: VectorArrangement,
    op: F,
) -> SmtExpr
where
    F: Fn(SmtExpr, SmtExpr) -> SmtExpr,
{
    let lane_bits = arrangement.lane_bits();
    let n = arrangement.lane_count();
    let lanes: Vec<SmtExpr> = (0..n)
        .map(|i| {
            let a = lane_extract(lhs, arrangement, i);
            let b = SmtExpr::bv_const(imm, lane_bits);
            op(a, b)
        })
        .collect();
    concat_lanes(&lanes, arrangement)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn test_bvadd_wrapping() {
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        let expr = a.bvadd(b);
        // 0xFFFFFFFF + 1 = 0 (wrapping)
        let result = expr.eval(&env(&[("a", 0xFFFF_FFFF), ("b", 1)]));
        assert_eq!(result, EvalResult::Bv(0));
    }

    #[test]
    fn test_bvsub() {
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        let expr = a.bvsub(b);
        let result = expr.eval(&env(&[("a", 10), ("b", 3)]));
        assert_eq!(result, EvalResult::Bv(7));
    }

    #[test]
    fn test_bvmul() {
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        let expr = a.bvmul(b);
        let result = expr.eval(&env(&[("a", 7), ("b", 6)]));
        assert_eq!(result, EvalResult::Bv(42));
    }

    #[test]
    fn test_bvneg() {
        let a = SmtExpr::var("a", 32);
        let expr = a.bvneg();
        // neg(1) = 0xFFFFFFFF in 32-bit
        let result = expr.eval(&env(&[("a", 1)]));
        assert_eq!(result, EvalResult::Bv(0xFFFF_FFFF));
    }

    #[test]
    fn test_bvneg_zero() {
        let a = SmtExpr::var("a", 32);
        let expr = a.bvneg();
        let result = expr.eval(&env(&[("a", 0)]));
        assert_eq!(result, EvalResult::Bv(0));
    }

    #[test]
    fn test_eq_true() {
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        let expr = a.eq_expr(b);
        let result = expr.eval(&env(&[("a", 42), ("b", 42)]));
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_eq_false() {
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        let expr = a.eq_expr(b);
        let result = expr.eval(&env(&[("a", 42), ("b", 43)]));
        assert_eq!(result, EvalResult::Bool(false));
    }

    #[test]
    fn test_sign_extend_negative() {
        // -1 in 8 bits = 0xFF
        assert_eq!(sign_extend(0xFF, 8), -1i64);
        // -128 in 8 bits = 0x80
        assert_eq!(sign_extend(0x80, 8), -128i64);
    }

    #[test]
    fn test_bvslt() {
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        let expr = a.bvslt(b);
        // -1 < 0 in signed
        let neg1_32 = 0xFFFF_FFFFu64;
        let result = expr.eval(&env(&[("a", neg1_32), ("b", 0)]));
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_display() {
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        let expr = a.bvadd(b);
        assert_eq!(format!("{}", expr), "(bvadd a b)");
    }

    #[test]
    fn test_mask_widths() {
        assert_eq!(mask(0xFF, 8), 0xFF);
        assert_eq!(mask(0x1FF, 8), 0xFF);
        assert_eq!(mask(0xFFFF_FFFF_FFFF_FFFF, 32), 0xFFFF_FFFF);
    }

    #[test]
    fn test_concat_basic() {
        // concat(0xAB : 8bit, 0xCD : 8bit) = 0xABCD : 16bit
        let hi = SmtExpr::bv_const(0xAB, 8);
        let lo = SmtExpr::bv_const(0xCD, 8);
        let expr = hi.concat(lo);
        assert_eq!(expr.bv_width(), 16);
        let result = expr.eval(&env(&[]));
        assert_eq!(result, EvalResult::Bv(0xABCD));
    }

    #[test]
    fn test_concat_32bit() {
        // concat(0xDEAD : 16bit, 0xBEEF : 16bit) = 0xDEADBEEF : 32bit
        let hi = SmtExpr::bv_const(0xDEAD, 16);
        let lo = SmtExpr::bv_const(0xBEEF, 16);
        let expr = hi.concat(lo);
        assert_eq!(expr.bv_width(), 32);
        let result = expr.eval(&env(&[]));
        assert_eq!(result, EvalResult::Bv(0xDEAD_BEEF));
    }

    #[test]
    fn test_zero_extend() {
        // zero_extend(0xFF : 8bit, 8) = 0x00FF : 16bit
        let a = SmtExpr::bv_const(0xFF, 8);
        let expr = a.zero_ext(8);
        assert_eq!(expr.bv_width(), 16);
        let result = expr.eval(&env(&[]));
        assert_eq!(result, EvalResult::Bv(0xFF));
    }

    #[test]
    fn test_sign_extend_expr() {
        // sign_extend(0xFF : 8bit, 8) = 0xFFFF : 16bit (since 0xFF is -1 in 8-bit)
        let a = SmtExpr::bv_const(0xFF, 8);
        let expr = a.sign_ext(8);
        assert_eq!(expr.bv_width(), 16);
        let result = expr.eval(&env(&[]));
        assert_eq!(result, EvalResult::Bv(0xFFFF));
    }

    #[test]
    fn test_sign_extend_positive() {
        // sign_extend(0x7F : 8bit, 8) = 0x007F : 16bit (positive stays positive)
        let a = SmtExpr::bv_const(0x7F, 8);
        let expr = a.sign_ext(8);
        assert_eq!(expr.bv_width(), 16);
        let result = expr.eval(&env(&[]));
        assert_eq!(result, EvalResult::Bv(0x7F));
    }

    #[test]
    fn test_extract_then_concat_roundtrip() {
        // Extract two 8-bit lanes from a 16-bit value, then concat back.
        let v = SmtExpr::var("v", 16);
        let lo = v.clone().extract(7, 0);   // bits [7:0]
        let hi = v.clone().extract(15, 8);  // bits [15:8]
        let reassembled = hi.concat(lo);
        // For v = 0xABCD: lo=0xCD, hi=0xAB, concat=0xABCD
        let result = reassembled.eval(&env(&[("v", 0xABCD)]));
        assert_eq!(result, EvalResult::Bv(0xABCD));
    }

    #[test]
    fn test_lane_extract_2s() {
        // 64-bit vector with 2 x 32-bit lanes: [0x12345678, 0xAABBCCDD]
        // Lane 0 (bits [31:0]) = 0xAABBCCDD, Lane 1 (bits [63:32]) = 0x12345678
        let v = SmtExpr::var("v", 64);
        let lane0 = lane_extract(&v, VectorArrangement::S2, 0);
        let lane1 = lane_extract(&v, VectorArrangement::S2, 1);
        let val = 0x12345678_AABBCCDD_u64;
        let e = env(&[("v", val)]);
        assert_eq!(lane0.eval(&e), EvalResult::Bv(0xAABBCCDD));
        assert_eq!(lane1.eval(&e), EvalResult::Bv(0x12345678));
    }

    #[test]
    fn test_concat_lanes_roundtrip() {
        // Decompose a 64-bit value as 2 x 32-bit lanes (S2), then reassemble.
        let v64 = SmtExpr::var("v64", 64);
        let l0 = lane_extract(&v64, VectorArrangement::S2, 0);
        let l1 = lane_extract(&v64, VectorArrangement::S2, 1);
        let reassembled = concat_lanes(&[l0, l1], VectorArrangement::S2);
        let val = 0xDEAD_BEEF_CAFE_BABEu64;
        let result = reassembled.eval(&env(&[("v64", val)]));
        assert_eq!(result, EvalResult::Bv(val));
    }

    #[test]
    fn test_map_lanes_binary_add_s2() {
        // Two 64-bit vectors, each with 2 x 32-bit lanes.
        // a = [0x00000001, 0x00000002], b = [0x00000003, 0x00000004]
        // Result: [(1+3), (2+4)] = [0x00000004, 0x00000006]
        let a = SmtExpr::var("a", 64);
        let b = SmtExpr::var("b", 64);
        let result = map_lanes_binary(&a, &b, VectorArrangement::S2, |x, y| x.bvadd(y));
        // a: lane0=0x00000002, lane1=0x00000001 (little-endian bit layout)
        // Encoding: 0x00000001_00000002
        let a_val = (1u64 << 32) | 2;
        let b_val = (3u64 << 32) | 4;
        let e = env(&[("a", a_val), ("b", b_val)]);
        let r = result.eval(&e);
        // Expected: lane0=2+4=6, lane1=1+3=4 => (4 << 32) | 6
        assert_eq!(r, EvalResult::Bv((4u64 << 32) | 6));
    }

    #[test]
    fn test_concat_display() {
        let hi = SmtExpr::var("hi", 8);
        let lo = SmtExpr::var("lo", 8);
        let expr = hi.concat(lo);
        assert_eq!(format!("{}", expr), "(concat hi lo)");
    }

    // -----------------------------------------------------------------------
    // Array theory tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_const_array_select() {
        // Create a constant array filled with 42, then select index 0.
        let arr = SmtExpr::const_array(
            SmtSort::BitVec(32),
            SmtExpr::bv_const(42, 32),
        );
        let expr = SmtExpr::select(arr, SmtExpr::bv_const(0, 32));
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bv(42));
    }

    #[test]
    fn test_const_array_select_any_index() {
        // Constant array: any index should return the default.
        let arr = SmtExpr::const_array(
            SmtSort::BitVec(8),
            SmtExpr::bv_const(0xFF, 8),
        );
        let expr = SmtExpr::select(arr, SmtExpr::bv_const(99, 8));
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bv(0xFF));
    }

    #[test]
    fn test_store_then_select() {
        // store(const_array(0), idx=5, val=100) then select idx=5
        let arr = SmtExpr::const_array(
            SmtSort::BitVec(8),
            SmtExpr::bv_const(0, 32),
        );
        let arr = SmtExpr::store(arr, SmtExpr::bv_const(5, 8), SmtExpr::bv_const(100, 32));
        let expr = SmtExpr::select(arr, SmtExpr::bv_const(5, 8));
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bv(100));
    }

    #[test]
    fn test_store_preserves_other_indices() {
        // store at index 5, select at index 3 should return default.
        let arr = SmtExpr::const_array(
            SmtSort::BitVec(8),
            SmtExpr::bv_const(0, 32),
        );
        let arr = SmtExpr::store(arr, SmtExpr::bv_const(5, 8), SmtExpr::bv_const(100, 32));
        let expr = SmtExpr::select(arr, SmtExpr::bv_const(3, 8));
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bv(0));
    }

    #[test]
    fn test_array_sort() {
        let arr = SmtExpr::const_array(
            SmtSort::BitVec(32),
            SmtExpr::bv_const(0, 64),
        );
        assert_eq!(arr.sort(), SmtSort::Array(
            Box::new(SmtSort::BitVec(32)),
            Box::new(SmtSort::BitVec(64)),
        ));
    }

    #[test]
    fn test_array_display() {
        let arr = SmtExpr::const_array(
            SmtSort::BitVec(32),
            SmtExpr::bv_const(0, 32),
        );
        let sel = SmtExpr::select(arr.clone(), SmtExpr::bv_const(1, 32));
        assert_eq!(
            format!("{}", sel),
            "(select ((as const (Array (_ BitVec 32) (_ BitVec 32))) (_ bv0 32)) (_ bv1 32))"
        );

        let st = SmtExpr::store(arr, SmtExpr::bv_const(1, 32), SmtExpr::bv_const(42, 32));
        assert_eq!(
            format!("{}", st),
            "(store ((as const (Array (_ BitVec 32) (_ BitVec 32))) (_ bv0 32)) (_ bv1 32) (_ bv42 32))"
        );
    }

    // -----------------------------------------------------------------------
    // Floating-point theory tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fp32_add() {
        let a = SmtExpr::fp32_const(1.5f32);
        let b = SmtExpr::fp32_const(2.5f32);
        let expr = SmtExpr::fp_add(RoundingMode::RNE, a, b);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Float(4.0));
    }

    #[test]
    fn test_fp64_mul() {
        let a = SmtExpr::fp64_const(3.0f64);
        let b = SmtExpr::fp64_const(7.0f64);
        let expr = SmtExpr::fp_mul(RoundingMode::RNE, a, b);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Float(21.0));
    }

    #[test]
    fn test_fp_div() {
        let a = SmtExpr::fp64_const(10.0);
        let b = SmtExpr::fp64_const(4.0);
        let expr = SmtExpr::fp_div(RoundingMode::RNE, a, b);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Float(2.5));
    }

    #[test]
    fn test_fp_neg() {
        let a = SmtExpr::fp64_const(42.0);
        let expr = a.fp_neg();
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Float(-42.0));
    }

    #[test]
    fn test_fp_eq() {
        let a = SmtExpr::fp64_const(1.0);
        let b = SmtExpr::fp64_const(1.0);
        let expr = a.fp_eq(b);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_fp_lt() {
        let a = SmtExpr::fp64_const(1.0);
        let b = SmtExpr::fp64_const(2.0);
        let expr = a.fp_lt(b);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_fp_sort() {
        let a = SmtExpr::fp32_const(1.0f32);
        assert_eq!(a.sort(), SmtSort::FloatingPoint(8, 24));

        let b = SmtExpr::fp64_const(1.0);
        assert_eq!(b.sort(), SmtSort::FloatingPoint(11, 53));
    }

    #[test]
    fn test_fp_display() {
        let expr = SmtExpr::fp_add(
            RoundingMode::RNE,
            SmtExpr::fp64_const(1.0),
            SmtExpr::fp64_const(2.0),
        );
        let s = format!("{}", expr);
        assert!(s.starts_with("(fp.add RNE"));
    }

    #[test]
    fn test_smt_sort_constructors() {
        assert_eq!(SmtSort::fp16(), SmtSort::FloatingPoint(5, 11));
        assert_eq!(SmtSort::fp32(), SmtSort::FloatingPoint(8, 24));
        assert_eq!(SmtSort::fp64(), SmtSort::FloatingPoint(11, 53));
        assert_eq!(
            SmtSort::bv_array(32, 64),
            SmtSort::Array(Box::new(SmtSort::BitVec(32)), Box::new(SmtSort::BitVec(64)))
        );
    }

    #[test]
    fn test_smt_sort_display() {
        assert_eq!(format!("{}", SmtSort::BitVec(32)), "(_ BitVec 32)");
        assert_eq!(format!("{}", SmtSort::Bool), "Bool");
        assert_eq!(
            format!("{}", SmtSort::fp32()),
            "(_ FloatingPoint 8 24)"
        );
        assert_eq!(
            format!("{}", SmtSort::bv_array(32, 8)),
            "(Array (_ BitVec 32) (_ BitVec 8))"
        );
    }

    // -----------------------------------------------------------------------
    // Uninterpreted function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_uf_display() {
        let uf = SmtExpr::uf(
            "hash",
            vec![SmtExpr::bv_const(42, 32)],
            SmtSort::BitVec(64),
        );
        assert_eq!(format!("{}", uf), "(hash (_ bv42 32))");
    }

    #[test]
    fn test_uf_decl_display() {
        let decl = SmtExpr::uf_decl(
            "hash",
            vec![SmtSort::BitVec(32)],
            SmtSort::BitVec(64),
        );
        assert_eq!(
            format!("{}", decl),
            "(declare-fun hash ((_ BitVec 32)) (_ BitVec 64))"
        );
    }

    #[test]
    fn test_uf_sort() {
        let uf = SmtExpr::uf("f", vec![], SmtSort::BitVec(32));
        assert_eq!(uf.sort(), SmtSort::BitVec(32));
    }

    #[test]
    fn test_uf_eval_errors() {
        let uf = SmtExpr::uf("f", vec![], SmtSort::BitVec(32));
        assert!(uf.try_eval(&HashMap::new()).is_err());
    }

    #[test]
    fn test_rounding_mode_display() {
        assert_eq!(format!("{}", RoundingMode::RNE), "RNE");
        assert_eq!(format!("{}", RoundingMode::RNA), "RNA");
        assert_eq!(format!("{}", RoundingMode::RTP), "RTP");
        assert_eq!(format!("{}", RoundingMode::RTN), "RTN");
        assert_eq!(format!("{}", RoundingMode::RTZ), "RTZ");
    }

    #[test]
    fn test_free_vars_with_new_exprs() {
        // Array expression references no free vars if built from constants.
        let arr = SmtExpr::const_array(SmtSort::BitVec(8), SmtExpr::bv_const(0, 32));
        let sel = SmtExpr::select(arr, SmtExpr::var("idx", 8));
        assert_eq!(sel.free_vars(), vec!["idx".to_string()]);

        // FP expression
        let fp_expr = SmtExpr::fp_add(
            RoundingMode::RNE,
            SmtExpr::fp64_const(1.0),
            SmtExpr::fp64_const(2.0),
        );
        assert!(fp_expr.free_vars().is_empty());

        // UF with args
        let uf = SmtExpr::uf("f", vec![SmtExpr::var("x", 32)], SmtSort::BitVec(32));
        assert_eq!(uf.free_vars(), vec!["x".to_string()]);
    }
}
