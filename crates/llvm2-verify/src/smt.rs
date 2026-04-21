// llvm2-verify/smt.rs - SMT expression AST and bitvector evaluator
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
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

    /// ConstArray index sort must be a bitvector sort.
    #[error("ConstArray index sort must be BitVec, got: {0}")]
    InvalidArrayIndexSort(String),

    /// Store/Select operation on a non-array expression.
    #[error("expected array sort, got: {0}")]
    NotAnArraySort(String),

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
            Type::V128 => Ok(SmtSort::BitVec(128)),
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

    /// `(fp.sub rm a b)` -- floating-point subtraction.
    FPSub { rm: RoundingMode, lhs: Box<SmtExpr>, rhs: Box<SmtExpr> },

    /// `(fp.div rm a b)` -- floating-point division.
    FPDiv { rm: RoundingMode, lhs: Box<SmtExpr>, rhs: Box<SmtExpr> },

    /// `(fp.neg a)` -- floating-point negation.
    FPNeg { operand: Box<SmtExpr> },

    /// `(fp.eq a b)` -- floating-point equality (returns Bool).
    FPEq { lhs: Box<SmtExpr>, rhs: Box<SmtExpr> },

    /// `(fp.lt a b)` -- floating-point less-than (returns Bool).
    FPLt { lhs: Box<SmtExpr>, rhs: Box<SmtExpr> },

    /// `(fp.gt a b)` -- floating-point greater-than (returns Bool).
    FPGt { lhs: Box<SmtExpr>, rhs: Box<SmtExpr> },

    /// `(fp.geq a b)` -- floating-point greater-or-equal (returns Bool).
    FPGe { lhs: Box<SmtExpr>, rhs: Box<SmtExpr> },

    /// `(fp.leq a b)` -- floating-point less-or-equal (returns Bool).
    FPLe { lhs: Box<SmtExpr>, rhs: Box<SmtExpr> },

    /// Floating-point constant from f64 bits.
    ///
    /// `eb` = exponent bits, `sb` = significand bits.
    /// The `bits` field holds the IEEE 754 bit pattern.
    FPConst { bits: u64, eb: u32, sb: u32 },

    /// `(fp.sqrt rm a)` -- floating-point square root.
    FPSqrt { rm: RoundingMode, operand: Box<SmtExpr> },

    /// `(fp.abs a)` -- floating-point absolute value.
    FPAbs { operand: Box<SmtExpr> },

    /// `(fp.fma rm a b c)` -- floating-point fused multiply-add: `a * b + c`.
    FPFma { rm: RoundingMode, a: Box<SmtExpr>, b: Box<SmtExpr>, c: Box<SmtExpr> },

    /// `(fp.isNaN a)` -- true if the argument is NaN (returns Bool).
    FPIsNaN { operand: Box<SmtExpr> },

    /// `(fp.isInfinite a)` -- true if the argument is +/- infinity (returns Bool).
    FPIsInf { operand: Box<SmtExpr> },

    /// `(fp.isZero a)` -- true if the argument is +/- zero (returns Bool).
    FPIsZero { operand: Box<SmtExpr> },

    /// `(fp.isNormal a)` -- true if the argument is a normal FP number (returns Bool).
    FPIsNormal { operand: Box<SmtExpr> },

    /// `((_ fp.to_sbv width) rm a)` -- convert FP to signed bitvector.
    FPToSBv { rm: RoundingMode, operand: Box<SmtExpr>, width: u32 },

    /// `((_ fp.to_ubv width) rm a)` -- convert FP to unsigned bitvector.
    FPToUBv { rm: RoundingMode, operand: Box<SmtExpr>, width: u32 },

    /// `((_ to_fp eb sb) rm bv)` -- convert bitvector to FP with rounding.
    BvToFP { rm: RoundingMode, operand: Box<SmtExpr>, eb: u32, sb: u32 },

    /// `((_ to_fp eb sb) rm fp)` -- convert between FP formats with rounding.
    FPToFP { rm: RoundingMode, operand: Box<SmtExpr>, eb: u32, sb: u32 },

    // -- Uninterpreted functions (QF_UF theory) --

    /// `(name arg1 arg2 ...)` -- uninterpreted function application.
    UF { name: String, args: Vec<SmtExpr>, ret_sort: SmtSort },

    /// `(declare-fun name (arg_sorts...) ret_sort)` -- function declaration.
    ///
    /// This is not an expression per se but a declaration node used in
    /// query generation to emit the function signature.
    UFDecl { name: String, arg_sorts: Vec<SmtSort>, ret_sort: SmtSort },

    // -- Bounded quantifiers --

    /// `(forall ((var (_ BitVec w))) (=> (and (bvuge var lo) (bvult var hi)) body))`
    ///
    /// Bounded universal quantifier: for all values of `var` in `[lower, upper)`,
    /// `body` holds. The bound variable has bitvector width `var_width`.
    ///
    /// Concrete evaluation: unrolls the quantifier for small ranges (upper - lower <= 256)
    /// and checks that `body` evaluates to true for every value in the range.
    ForAll {
        var: String,
        var_width: u32,
        lower: Box<SmtExpr>,
        upper: Box<SmtExpr>,
        body: Box<SmtExpr>,
    },

    /// `(exists ((var (_ BitVec w))) (and (bvuge var lo) (bvult var hi) body))`
    ///
    /// Bounded existential quantifier: there exists a value of `var` in `[lower, upper)`
    /// such that `body` holds. The bound variable has bitvector width `var_width`.
    ///
    /// Concrete evaluation: unrolls the quantifier for small ranges (upper - lower <= 256)
    /// and checks that `body` evaluates to true for at least one value in the range.
    Exists {
        var: String,
        var_width: u32,
        lower: Box<SmtExpr>,
        upper: Box<SmtExpr>,
        body: Box<SmtExpr>,
    },
}

// ---------------------------------------------------------------------------
// Array sort validation helper
// ---------------------------------------------------------------------------

/// Validate that an expression has an Array sort, when statically determinable.
///
/// Expressions whose sort is statically known to be non-Array (e.g., `BvConst`,
/// `BoolConst`, comparison results) cause an error. Expressions whose sort
/// cannot be cheaply determined at construction time (e.g., `Var`, `Ite`) are
/// allowed through — runtime validation occurs in `try_eval`.
fn validate_array_sort(expr: &SmtExpr) -> Result<(), SmtError> {
    match expr {
        // Known array-sorted expressions: always OK.
        SmtExpr::ConstArray { .. } | SmtExpr::Store { .. } => Ok(()),
        // Expressions whose sort is ambiguous at construction time: allow.
        // Var, Select (returns element sort), Ite, UF — we can't cheaply
        // determine array sort without recursion, so defer to eval.
        SmtExpr::Var { .. }
        | SmtExpr::Ite { .. }
        | SmtExpr::Select { .. }
        | SmtExpr::UF { .. } => Ok(()),
        // Everything else is statically known to NOT be array-sorted.
        other => {
            let sort = other.sort();
            if matches!(sort, SmtSort::Array(_, _)) {
                Ok(())
            } else {
                Err(SmtError::NotAnArraySort(format!("{}", sort)))
            }
        }
    }
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
    ///
    /// Validates that `array` has an `Array` sort when statically determinable
    /// (i.e., for `ConstArray` and `Store` expressions). For `Var` expressions
    /// whose sort cannot be determined at construction time, validation is
    /// deferred to evaluation via [`try_eval`].
    pub fn select(array: Self, index: Self) -> Self {
        Self::try_select(array, index)
            .expect("select: first argument must have Array sort; use try_select() for fallible construction")
    }

    /// Fallible `(select array index)`.
    ///
    /// Returns `Err(SmtError::NotAnArraySort)` if the array expression's sort
    /// is statically known to not be an Array sort.
    pub fn try_select(array: Self, index: Self) -> Result<Self, SmtError> {
        validate_array_sort(&array)?;
        Ok(SmtExpr::Select {
            array: Box::new(array),
            index: Box::new(index),
        })
    }

    /// `(store array index value)` -- write to array.
    ///
    /// Validates that `array` has an `Array` sort when statically determinable.
    pub fn store(array: Self, index: Self, value: Self) -> Self {
        Self::try_store(array, index, value)
            .expect("store: first argument must have Array sort; use try_store() for fallible construction")
    }

    /// Fallible `(store array index value)`.
    ///
    /// Returns `Err(SmtError::NotAnArraySort)` if the array expression's sort
    /// is statically known to not be an Array sort.
    pub fn try_store(array: Self, index: Self, value: Self) -> Result<Self, SmtError> {
        validate_array_sort(&array)?;
        Ok(SmtExpr::Store {
            array: Box::new(array),
            index: Box::new(index),
            value: Box::new(value),
        })
    }

    /// `((as const ...) value)` -- constant array filled with `value`.
    ///
    /// # Panics
    ///
    /// Panics if `index_sort` is not `SmtSort::BitVec`. Use [`try_const_array`]
    /// for fallible construction.
    pub fn const_array(index_sort: SmtSort, value: Self) -> Self {
        Self::try_const_array(index_sort, value)
            .expect("const_array: index_sort must be BitVec; use try_const_array() for fallible construction")
    }

    /// Fallible `((as const ...) value)` -- constant array filled with `value`.
    ///
    /// Returns `Err(SmtError::InvalidArrayIndexSort)` if `index_sort` is not
    /// `SmtSort::BitVec`.
    pub fn try_const_array(index_sort: SmtSort, value: Self) -> Result<Self, SmtError> {
        if !matches!(index_sort, SmtSort::BitVec(_)) {
            return Err(SmtError::InvalidArrayIndexSort(format!("{}", index_sort)));
        }
        Ok(SmtExpr::ConstArray {
            index_sort,
            value: Box::new(value),
        })
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

    /// `(fp.sub rm a b)` -- floating-point subtraction.
    pub fn fp_sub(rm: RoundingMode, a: Self, b: Self) -> Self {
        SmtExpr::FPSub { rm, lhs: Box::new(a), rhs: Box::new(b) }
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

    /// `(fp.gt a b)` -- floating-point greater-than (returns Bool).
    pub fn fp_gt(self, other: Self) -> Self {
        SmtExpr::FPGt { lhs: Box::new(self), rhs: Box::new(other) }
    }

    /// `(fp.geq a b)` -- floating-point greater-or-equal (returns Bool).
    pub fn fp_ge(self, other: Self) -> Self {
        SmtExpr::FPGe { lhs: Box::new(self), rhs: Box::new(other) }
    }

    /// `(fp.leq a b)` -- floating-point less-or-equal (returns Bool).
    pub fn fp_le(self, other: Self) -> Self {
        SmtExpr::FPLe { lhs: Box::new(self), rhs: Box::new(other) }
    }

    /// `(fp.sqrt rm a)` -- floating-point square root.
    pub fn fp_sqrt(rm: RoundingMode, a: Self) -> Self {
        SmtExpr::FPSqrt { rm, operand: Box::new(a) }
    }

    /// `(fp.abs a)` -- floating-point absolute value.
    pub fn fp_abs(self) -> Self {
        SmtExpr::FPAbs { operand: Box::new(self) }
    }

    /// `(fp.fma rm a b c)` -- floating-point fused multiply-add: `a * b + c`.
    pub fn fp_fma(rm: RoundingMode, a: Self, b: Self, c: Self) -> Self {
        SmtExpr::FPFma { rm, a: Box::new(a), b: Box::new(b), c: Box::new(c) }
    }

    /// `(fp.isNaN a)` -- true if the argument is NaN (returns Bool).
    pub fn fp_is_nan(self) -> Self {
        SmtExpr::FPIsNaN { operand: Box::new(self) }
    }

    /// `(fp.isInfinite a)` -- true if the argument is infinity (returns Bool).
    pub fn fp_is_inf(self) -> Self {
        SmtExpr::FPIsInf { operand: Box::new(self) }
    }

    /// `(fp.isZero a)` -- true if the argument is +/- zero (returns Bool).
    pub fn fp_is_zero(self) -> Self {
        SmtExpr::FPIsZero { operand: Box::new(self) }
    }

    /// `(fp.isNormal a)` -- true if the argument is a normal FP number (returns Bool).
    pub fn fp_is_normal(self) -> Self {
        SmtExpr::FPIsNormal { operand: Box::new(self) }
    }

    /// `((_ fp.to_sbv width) rm a)` -- convert FP to signed bitvector.
    pub fn fp_to_sbv(rm: RoundingMode, a: Self, width: u32) -> Self {
        SmtExpr::FPToSBv { rm, operand: Box::new(a), width }
    }

    /// `((_ fp.to_ubv width) rm a)` -- convert FP to unsigned bitvector.
    pub fn fp_to_ubv(rm: RoundingMode, a: Self, width: u32) -> Self {
        SmtExpr::FPToUBv { rm, operand: Box::new(a), width }
    }

    /// `((_ to_fp eb sb) rm bv)` -- convert signed bitvector to FP.
    pub fn bv_to_fp(rm: RoundingMode, bv: Self, eb: u32, sb: u32) -> Self {
        SmtExpr::BvToFP { rm, operand: Box::new(bv), eb, sb }
    }

    /// `((_ to_fp eb sb) rm fp)` -- convert between FP formats.
    pub fn fp_to_fp(rm: RoundingMode, fp: Self, eb: u32, sb: u32) -> Self {
        SmtExpr::FPToFP { rm, operand: Box::new(fp), eb, sb }
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

    // -- Bounded quantifier constructors --

    /// Bounded universal quantifier: `forall var in [lower, upper). body`.
    ///
    /// The bound variable has bitvector width `var_width`. During concrete evaluation,
    /// the quantifier is unrolled: for each value `v` in `[lower, upper)`, `body` is
    /// evaluated with `var` bound to `v`. All must be true for the result to be true.
    ///
    /// Maximum unrolling bound: 256 iterations. Exceeding this returns an eval error.
    pub fn forall(
        var: impl Into<String>,
        var_width: u32,
        lower: Self,
        upper: Self,
        body: Self,
    ) -> Self {
        SmtExpr::ForAll {
            var: var.into(),
            var_width,
            lower: Box::new(lower),
            upper: Box::new(upper),
            body: Box::new(body),
        }
    }

    /// Bounded existential quantifier: `exists var in [lower, upper). body`.
    ///
    /// The bound variable has bitvector width `var_width`. During concrete evaluation,
    /// the quantifier is unrolled: for each value `v` in `[lower, upper)`, `body` is
    /// evaluated with `var` bound to `v`. At least one must be true for the result to be true.
    ///
    /// Maximum unrolling bound: 256 iterations. Exceeding this returns an eval error.
    pub fn exists(
        var: impl Into<String>,
        var_width: u32,
        lower: Self,
        upper: Self,
        body: Self,
    ) -> Self {
        SmtExpr::Exists {
            var: var.into(),
            var_width,
            lower: Box::new(lower),
            upper: Box::new(upper),
            body: Box::new(body),
        }
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
            // FP-to-BV conversions produce bitvectors.
            SmtExpr::FPToSBv { width, .. } | SmtExpr::FPToUBv { width, .. } => Ok(*width),
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
            | SmtExpr::FPLt { .. }
            | SmtExpr::FPGt { .. }
            | SmtExpr::FPGe { .. }
            | SmtExpr::FPLe { .. }
            | SmtExpr::FPIsNaN { .. }
            | SmtExpr::FPIsInf { .. }
            | SmtExpr::FPIsZero { .. }
            | SmtExpr::FPIsNormal { .. }
            | SmtExpr::ForAll { .. }
            | SmtExpr::Exists { .. } => Err(SmtError::BoolHasNoWidth),
            // FP / array / UF decl nodes have no BV width.
            SmtExpr::FPAdd { .. }
            | SmtExpr::FPSub { .. }
            | SmtExpr::FPMul { .. }
            | SmtExpr::FPDiv { .. }
            | SmtExpr::FPNeg { .. }
            | SmtExpr::FPAbs { .. }
            | SmtExpr::FPSqrt { .. }
            | SmtExpr::FPFma { .. }
            | SmtExpr::FPConst { .. }
            | SmtExpr::BvToFP { .. }
            | SmtExpr::FPToFP { .. }
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
            | SmtExpr::FPLt { .. }
            | SmtExpr::FPGt { .. }
            | SmtExpr::FPGe { .. }
            | SmtExpr::FPLe { .. }
            | SmtExpr::FPIsNaN { .. }
            | SmtExpr::FPIsInf { .. }
            | SmtExpr::FPIsZero { .. }
            | SmtExpr::FPIsNormal { .. }
            | SmtExpr::ForAll { .. }
            | SmtExpr::Exists { .. } => SmtSort::Bool,
            // Floating-point expressions
            SmtExpr::FPAdd { lhs, .. }
            | SmtExpr::FPSub { lhs, .. }
            | SmtExpr::FPMul { lhs, .. }
            | SmtExpr::FPDiv { lhs, .. } => lhs.sort(),
            SmtExpr::FPSqrt { operand, .. }
            | SmtExpr::FPAbs { operand }
            | SmtExpr::FPNeg { operand } => operand.sort(),
            SmtExpr::FPFma { a, .. } => a.sort(),
            SmtExpr::FPConst { eb, sb, .. } => SmtSort::FloatingPoint(*eb, *sb),
            SmtExpr::BvToFP { eb, sb, .. } | SmtExpr::FPToFP { eb, sb, .. } => {
                SmtSort::FloatingPoint(*eb, *sb)
            }
            // FP-to-BV conversions produce bitvectors.
            SmtExpr::FPToSBv { width, .. } | SmtExpr::FPToUBv { width, .. } => {
                SmtSort::BitVec(*width)
            }
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
            | SmtExpr::FPLt { lhs, rhs }
            | SmtExpr::FPGt { lhs, rhs }
            | SmtExpr::FPGe { lhs, rhs }
            | SmtExpr::FPLe { lhs, rhs } => {
                lhs.collect_vars(vars);
                rhs.collect_vars(vars);
            }
            SmtExpr::FPAdd { lhs, rhs, .. }
            | SmtExpr::FPSub { lhs, rhs, .. }
            | SmtExpr::FPMul { lhs, rhs, .. }
            | SmtExpr::FPDiv { lhs, rhs, .. } => {
                lhs.collect_vars(vars);
                rhs.collect_vars(vars);
            }
            SmtExpr::FPFma { a, b, c, .. } => {
                a.collect_vars(vars);
                b.collect_vars(vars);
                c.collect_vars(vars);
            }
            SmtExpr::BvNeg { operand, .. }
            | SmtExpr::Not { operand }
            | SmtExpr::Extract { operand, .. }
            | SmtExpr::ZeroExtend { operand, .. }
            | SmtExpr::SignExtend { operand, .. }
            | SmtExpr::FPNeg { operand }
            | SmtExpr::FPAbs { operand }
            | SmtExpr::FPSqrt { operand, .. }
            | SmtExpr::FPIsNaN { operand }
            | SmtExpr::FPIsInf { operand }
            | SmtExpr::FPIsZero { operand }
            | SmtExpr::FPIsNormal { operand }
            | SmtExpr::FPToSBv { operand, .. }
            | SmtExpr::FPToUBv { operand, .. }
            | SmtExpr::BvToFP { operand, .. }
            | SmtExpr::FPToFP { operand, .. } => {
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
            SmtExpr::ForAll { var, lower, upper, body, .. }
            | SmtExpr::Exists { var, lower, upper, body, .. } => {
                lower.collect_vars(vars);
                upper.collect_vars(vars);
                // Collect vars from body, but the bound variable is not free.
                let mut body_vars = Vec::new();
                body.collect_vars(&mut body_vars);
                for v in body_vars {
                    if v != *var {
                        vars.push(v);
                    }
                }
            }
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

    /// Semantic equality that treats NaN values as equal to other NaN values.
    ///
    /// The default `PartialEq`/`Eq` derive compares `f64` with `==`, which
    /// is IEEE-754 equality: `NaN != NaN`. For verification of FP lowerings,
    /// "both sides produce NaN" is a passing result — the tMIR and the target
    /// instruction agree that the operation is not a number, which is the
    /// strongest semantic guarantee available without tracking exact payload
    /// bits. This method returns `true` in that case.
    ///
    /// For non-NaN floats, bit-level equality is used (so +0.0 == +0.0 but
    /// +0.0 != -0.0), matching IEEE-754 comparison except for the NaN rule.
    /// For non-Float variants, ordinary `==` is used.
    ///
    /// # Rationale
    ///
    /// IEEE-754 FDIV(0.0, 0.0) yields NaN; so does AArch64 `FDIV`. Rust's
    /// default `PartialEq` on `f64` returns false for NaN != NaN, causing
    /// the evaluator to flag a spurious counterexample even though both
    /// sides produce the canonical NaN result. See #388.
    pub fn semantically_equal(&self, other: &Self) -> bool {
        match (self, other) {
            (EvalResult::Float(a), EvalResult::Float(b)) => {
                // Both NaN = semantically equal.
                if a.is_nan() && b.is_nan() {
                    return true;
                }
                // Otherwise compare by bit pattern so +0.0 == +0.0 but
                // +0.0 != -0.0 and signalling vs quiet NaN (already handled
                // above) stay distinct for finite values.
                a.to_bits() == b.to_bits()
            }
            _ => self == other,
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

/// Sign-extend a `width`-bit value stored in a u128 to i128.
fn sign_extend128(value: u128, width: u32) -> i128 {
    if width == 0 {
        return 0;
    }
    if width >= 128 {
        return value as i128;
    }
    let shift = 128 - width;
    ((value << shift) as i128) >> shift
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
                if *width > 64 {
                    let a = lhs.try_eval(env)?.as_u128();
                    let b = rhs.try_eval(env)?.as_u128();
                    if b >= *width as u128 {
                        Ok(EvalResult::Bv128(0))
                    } else {
                        Ok(EvalResult::Bv128(mask128(a << b, *width)))
                    }
                } else {
                    let a = lhs.try_eval(env)?.as_u64();
                    let b = rhs.try_eval(env)?.as_u64();
                    // SMT-LIB: if shift amount >= width, result is 0.
                    if b >= *width as u64 {
                        Ok(EvalResult::Bv(0))
                    } else {
                        Ok(EvalResult::Bv(mask(a << b, *width)))
                    }
                }
            }
            SmtExpr::BvLshr { lhs, rhs, width } => {
                if *width > 64 {
                    let a = lhs.try_eval(env)?.as_u128();
                    let b = rhs.try_eval(env)?.as_u128();
                    if b >= *width as u128 {
                        Ok(EvalResult::Bv128(0))
                    } else {
                        Ok(EvalResult::Bv128(mask128(a >> b, *width)))
                    }
                } else {
                    let a = lhs.try_eval(env)?.as_u64();
                    let b = rhs.try_eval(env)?.as_u64();
                    if b >= *width as u64 {
                        Ok(EvalResult::Bv(0))
                    } else {
                        Ok(EvalResult::Bv(mask(a >> b, *width)))
                    }
                }
            }
            SmtExpr::BvAshr { lhs, rhs, width } => {
                if *width > 64 {
                    let a = sign_extend128(lhs.try_eval(env)?.as_u128(), *width);
                    let b = rhs.try_eval(env)?.as_u128();
                    if b >= *width as u128 {
                        if a < 0 {
                            Ok(EvalResult::Bv128(mask128(u128::MAX, *width)))
                        } else {
                            Ok(EvalResult::Bv128(0))
                        }
                    } else {
                        Ok(EvalResult::Bv128(mask128((a >> b) as u128, *width)))
                    }
                } else {
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
            SmtExpr::FPSub { lhs, rhs, .. } => {
                let a = lhs.try_eval(env)?.as_f64();
                let b = rhs.try_eval(env)?.as_f64();
                Ok(EvalResult::Float(a - b))
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
            SmtExpr::FPGt { lhs, rhs } => {
                let a = lhs.try_eval(env)?.as_f64();
                let b = rhs.try_eval(env)?.as_f64();
                Ok(EvalResult::Bool(a > b))
            }
            SmtExpr::FPGe { lhs, rhs } => {
                let a = lhs.try_eval(env)?.as_f64();
                let b = rhs.try_eval(env)?.as_f64();
                Ok(EvalResult::Bool(a >= b))
            }
            SmtExpr::FPLe { lhs, rhs } => {
                let a = lhs.try_eval(env)?.as_f64();
                let b = rhs.try_eval(env)?.as_f64();
                Ok(EvalResult::Bool(a <= b))
            }
            SmtExpr::FPSqrt { operand, .. } => {
                let a = operand.try_eval(env)?.as_f64();
                Ok(EvalResult::Float(a.sqrt()))
            }
            SmtExpr::FPAbs { operand } => {
                let a = operand.try_eval(env)?.as_f64();
                Ok(EvalResult::Float(a.abs()))
            }
            SmtExpr::FPFma { a, b, c, .. } => {
                let av = a.try_eval(env)?.as_f64();
                let bv = b.try_eval(env)?.as_f64();
                let cv = c.try_eval(env)?.as_f64();
                Ok(EvalResult::Float(av.mul_add(bv, cv)))
            }
            SmtExpr::FPIsNaN { operand } => {
                let a = operand.try_eval(env)?.as_f64();
                Ok(EvalResult::Bool(a.is_nan()))
            }
            SmtExpr::FPIsInf { operand } => {
                let a = operand.try_eval(env)?.as_f64();
                Ok(EvalResult::Bool(a.is_infinite()))
            }
            SmtExpr::FPIsZero { operand } => {
                let a = operand.try_eval(env)?.as_f64();
                Ok(EvalResult::Bool(a == 0.0))
            }
            SmtExpr::FPIsNormal { operand } => {
                let a = operand.try_eval(env)?.as_f64();
                Ok(EvalResult::Bool(a.is_normal()))
            }
            SmtExpr::FPToSBv { operand, width, .. } => {
                let a = operand.try_eval(env)?.as_f64();
                // Truncate toward zero (RTZ), clamp to signed range.
                let trunc = a as i64;
                Ok(EvalResult::Bv(mask(trunc as u64, *width)))
            }
            SmtExpr::FPToUBv { operand, width, .. } => {
                let a = operand.try_eval(env)?.as_f64();
                // Truncate toward zero (RTZ), clamp to unsigned range.
                let trunc = a as u64;
                Ok(EvalResult::Bv(mask(trunc, *width)))
            }
            SmtExpr::BvToFP { operand, eb, sb, .. } => {
                let v = operand.try_eval(env)?.as_u64();
                let src_width = operand.bv_width();
                // Interpret as signed integer, convert to f64.
                let signed = sign_extend(v, src_width);
                let f = if *eb == 8 && *sb == 24 {
                    (signed as f32) as f64
                } else {
                    signed as f64
                };
                Ok(EvalResult::Float(f))
            }
            SmtExpr::FPToFP { operand, eb, sb, .. } => {
                let f = operand.try_eval(env)?.as_f64();
                // Convert between FP formats via Rust f32/f64.
                let result = if *eb == 8 && *sb == 24 {
                    (f as f32) as f64
                } else {
                    f
                };
                Ok(EvalResult::Float(result))
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

            // -- Bounded quantifier evaluation (loop unrolling) --

            SmtExpr::ForAll { var, var_width, lower, upper, body } => {
                let lo = lower.try_eval(env)?.as_u64();
                let hi = upper.try_eval(env)?.as_u64();
                if hi <= lo {
                    // Empty range: vacuously true.
                    return Ok(EvalResult::Bool(true));
                }
                let count = hi - lo;
                if count > 256 {
                    return Err(SmtError::EvalError(format!(
                        "forall range too large for unrolling: {} (max 256)", count
                    )));
                }
                let mut local_env = env.clone();
                for i in lo..hi {
                    local_env.insert(var.clone(), mask(i, *var_width));
                    let result = body.try_eval(&local_env)?;
                    if !result.as_bool() {
                        return Ok(EvalResult::Bool(false));
                    }
                }
                Ok(EvalResult::Bool(true))
            }

            SmtExpr::Exists { var, var_width, lower, upper, body } => {
                let lo = lower.try_eval(env)?.as_u64();
                let hi = upper.try_eval(env)?.as_u64();
                if hi <= lo {
                    // Empty range: vacuously false.
                    return Ok(EvalResult::Bool(false));
                }
                let count = hi - lo;
                if count > 256 {
                    return Err(SmtError::EvalError(format!(
                        "exists range too large for unrolling: {} (max 256)", count
                    )));
                }
                let mut local_env = env.clone();
                for i in lo..hi {
                    local_env.insert(var.clone(), mask(i, *var_width));
                    let result = body.try_eval(&local_env)?;
                    if result.as_bool() {
                        return Ok(EvalResult::Bool(true));
                    }
                }
                Ok(EvalResult::Bool(false))
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

    /// Serialize this expression to an SMT-LIB2 expression string.
    ///
    /// This is a convenience method that delegates to the [`Display`] implementation,
    /// which produces valid SMT-LIB2 syntax for each expression variant.
    ///
    /// # Example
    ///
    /// ```
    /// use llvm2_verify::SmtExpr;
    /// let a = SmtExpr::var("a", 32);
    /// let b = SmtExpr::var("b", 32);
    /// let expr = a.bvadd(b);
    /// assert_eq!(expr.to_smt2_expr(), "(bvadd a b)");
    /// ```
    pub fn to_smt2_expr(&self) -> String {
        format!("{}", self)
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
                let sign = if bits >> (total - 1) & 1 == 1 { "1" } else { "0" };
                let exp = format!("{:0>width$b}", (bits >> (sb - 1)) & ((1u64 << eb) - 1), width = *eb as usize);
                let sig = format!("{:0>width$b}", bits & ((1u64 << (sb - 1)) - 1), width = (*sb - 1) as usize);
                write!(f, "(fp #b{} #b{} #b{})", sign, exp, sig)
            }
            SmtExpr::FPAdd { rm, lhs, rhs } => {
                write!(f, "(fp.add {} {} {})", rm, lhs, rhs)
            }
            SmtExpr::FPSub { rm, lhs, rhs } => {
                write!(f, "(fp.sub {} {} {})", rm, lhs, rhs)
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
            SmtExpr::FPGt { lhs, rhs } => {
                write!(f, "(fp.gt {} {})", lhs, rhs)
            }
            SmtExpr::FPGe { lhs, rhs } => {
                write!(f, "(fp.geq {} {})", lhs, rhs)
            }
            SmtExpr::FPLe { lhs, rhs } => {
                write!(f, "(fp.leq {} {})", lhs, rhs)
            }
            SmtExpr::FPSqrt { rm, operand } => {
                write!(f, "(fp.sqrt {} {})", rm, operand)
            }
            SmtExpr::FPAbs { operand } => {
                write!(f, "(fp.abs {})", operand)
            }
            SmtExpr::FPFma { rm, a, b, c } => {
                write!(f, "(fp.fma {} {} {} {})", rm, a, b, c)
            }
            SmtExpr::FPIsNaN { operand } => {
                write!(f, "(fp.isNaN {})", operand)
            }
            SmtExpr::FPIsInf { operand } => {
                write!(f, "(fp.isInfinite {})", operand)
            }
            SmtExpr::FPIsZero { operand } => {
                write!(f, "(fp.isZero {})", operand)
            }
            SmtExpr::FPIsNormal { operand } => {
                write!(f, "(fp.isNormal {})", operand)
            }
            SmtExpr::FPToSBv { rm, operand, width } => {
                write!(f, "((_ fp.to_sbv {}) {} {})", width, rm, operand)
            }
            SmtExpr::FPToUBv { rm, operand, width } => {
                write!(f, "((_ fp.to_ubv {}) {} {})", width, rm, operand)
            }
            SmtExpr::BvToFP { rm, operand, eb, sb } => {
                write!(f, "((_ to_fp {} {}) {} {})", eb, sb, rm, operand)
            }
            SmtExpr::FPToFP { rm, operand, eb, sb } => {
                write!(f, "((_ to_fp {} {}) {} {})", eb, sb, rm, operand)
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
            // Bounded quantifiers: emit SMT-LIB2 with range predicate as guard.
            // ForAll: (forall ((var (_ BitVec w))) (=> (and (bvuge var lo) (bvult var hi)) body))
            SmtExpr::ForAll { var, var_width, lower, upper, body } => {
                write!(
                    f,
                    "(forall (({} (_ BitVec {}))) (=> (and (bvuge {} {}) (bvult {} {})) {}))",
                    var, var_width, var, lower, var, upper, body
                )
            }
            // Exists: (exists ((var (_ BitVec w))) (and (bvuge var lo) (bvult var hi) body))
            SmtExpr::Exists { var, var_width, lower, upper, body } => {
                write!(
                    f,
                    "(exists (({} (_ BitVec {}))) (and (bvuge {} {}) (bvult {} {}) {}))",
                    var, var_width, var, lower, var, upper, body
                )
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

    // -----------------------------------------------------------------------
    // ConstArray index sort validation tests (#167)
    // -----------------------------------------------------------------------

    #[test]
    fn test_const_array_bv_index_sort_ok() {
        // Valid: BitVec index sort should succeed.
        let result = SmtExpr::try_const_array(SmtSort::BitVec(32), SmtExpr::bv_const(0, 64));
        assert!(result.is_ok());
        let arr = result.unwrap();
        assert_eq!(
            arr.sort(),
            SmtSort::Array(Box::new(SmtSort::BitVec(32)), Box::new(SmtSort::BitVec(64)))
        );
    }

    #[test]
    fn test_const_array_bool_index_sort_rejected() {
        // Invalid: Bool index sort should fail.
        let result = SmtExpr::try_const_array(SmtSort::Bool, SmtExpr::bv_const(0, 32));
        assert!(result.is_err());
        match result.unwrap_err() {
            SmtError::InvalidArrayIndexSort(msg) => {
                assert!(msg.contains("Bool"), "error should mention Bool sort: {}", msg);
            }
            other => panic!("expected InvalidArrayIndexSort, got: {:?}", other),
        }
    }

    #[test]
    fn test_const_array_array_index_sort_rejected() {
        // Invalid: Array index sort should fail.
        let nested_sort = SmtSort::Array(
            Box::new(SmtSort::BitVec(8)),
            Box::new(SmtSort::BitVec(8)),
        );
        let result = SmtExpr::try_const_array(nested_sort, SmtExpr::bv_const(0, 32));
        assert!(result.is_err());
        match result.unwrap_err() {
            SmtError::InvalidArrayIndexSort(msg) => {
                assert!(msg.contains("Array"), "error should mention Array sort: {}", msg);
            }
            other => panic!("expected InvalidArrayIndexSort, got: {:?}", other),
        }
    }

    #[test]
    fn test_const_array_fp_index_sort_rejected() {
        // Invalid: FloatingPoint index sort should fail.
        let result = SmtExpr::try_const_array(SmtSort::fp32(), SmtExpr::bv_const(0, 32));
        assert!(result.is_err());
        match result.unwrap_err() {
            SmtError::InvalidArrayIndexSort(msg) => {
                assert!(msg.contains("FloatingPoint"), "error should mention FP sort: {}", msg);
            }
            other => panic!("expected InvalidArrayIndexSort, got: {:?}", other),
        }
    }

    #[test]
    #[should_panic(expected = "index_sort must be BitVec")]
    fn test_const_array_panics_on_bool_index() {
        // The non-try version should panic.
        let _ = SmtExpr::const_array(SmtSort::Bool, SmtExpr::bv_const(0, 32));
    }

    // -----------------------------------------------------------------------
    // Select/Store sort validation tests (#167)
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_on_array_ok() {
        // Valid: selecting from a ConstArray.
        let arr = SmtExpr::const_array(SmtSort::BitVec(8), SmtExpr::bv_const(42, 32));
        let result = SmtExpr::try_select(arr, SmtExpr::bv_const(0, 8));
        assert!(result.is_ok());
    }

    #[test]
    fn test_store_on_array_ok() {
        // Valid: storing to a ConstArray.
        let arr = SmtExpr::const_array(SmtSort::BitVec(8), SmtExpr::bv_const(0, 32));
        let result = SmtExpr::try_store(arr, SmtExpr::bv_const(1, 8), SmtExpr::bv_const(99, 32));
        assert!(result.is_ok());
    }

    #[test]
    fn test_select_on_non_array_rejected() {
        // Invalid: selecting from a BvConst should fail.
        let result = SmtExpr::try_select(SmtExpr::bv_const(0, 32), SmtExpr::bv_const(0, 8));
        assert!(result.is_err());
        match result.unwrap_err() {
            SmtError::NotAnArraySort(msg) => {
                assert!(msg.contains("BitVec"), "error should mention sort: {}", msg);
            }
            other => panic!("expected NotAnArraySort, got: {:?}", other),
        }
    }

    #[test]
    fn test_store_on_non_array_rejected() {
        // Invalid: storing to a BoolConst should fail.
        let result = SmtExpr::try_store(
            SmtExpr::bool_const(true),
            SmtExpr::bv_const(0, 8),
            SmtExpr::bv_const(42, 32),
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            SmtError::NotAnArraySort(msg) => {
                assert!(msg.contains("Bool"), "error should mention sort: {}", msg);
            }
            other => panic!("expected NotAnArraySort, got: {:?}", other),
        }
    }

    #[test]
    #[should_panic(expected = "must have Array sort")]
    fn test_select_panics_on_non_array() {
        let _ = SmtExpr::select(SmtExpr::bv_const(0, 32), SmtExpr::bv_const(0, 8));
    }

    #[test]
    #[should_panic(expected = "must have Array sort")]
    fn test_store_panics_on_non_array() {
        let _ = SmtExpr::store(
            SmtExpr::bv_const(0, 32),
            SmtExpr::bv_const(0, 8),
            SmtExpr::bv_const(42, 32),
        );
    }

    #[test]
    fn test_select_on_var_allowed() {
        // Var sort can't be statically determined as Array, but we defer
        // validation to eval time for flexibility.
        let result = SmtExpr::try_select(SmtExpr::var("mem", 64), SmtExpr::bv_const(0, 8));
        assert!(result.is_ok());
    }

    #[test]
    fn test_store_then_select_well_typed() {
        // Full roundtrip: const_array -> store -> select with matching types.
        let arr = SmtExpr::const_array(SmtSort::BitVec(8), SmtExpr::bv_const(0, 32));
        let arr = SmtExpr::store(arr, SmtExpr::bv_const(7, 8), SmtExpr::bv_const(255, 32));
        let sel = SmtExpr::select(arr, SmtExpr::bv_const(7, 8));
        let result = sel.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bv(255));
    }

    // -----------------------------------------------------------------------
    // 128-bit shift operation tests (BvShl, BvLshr, BvAshr with width > 64)
    // -----------------------------------------------------------------------

    /// Helper: build a 128-bit BvShl expression from two concatenated 64-bit halves.
    fn make_128bit_shl(hi_val: u64, lo_val: u64, shift: u64) -> EvalResult {
        let hi = SmtExpr::var("hi", 64);
        let lo = SmtExpr::var("lo", 64);
        let vec128 = hi.concat(lo); // 128-bit value
        let shift_amt = SmtExpr::bv_const(shift, 128);
        let expr = SmtExpr::BvShl {
            lhs: Box::new(vec128),
            rhs: Box::new(shift_amt),
            width: 128,
        };
        expr.eval(&env(&[("hi", hi_val), ("lo", lo_val)]))
    }

    #[test]
    fn test_bvshl_128bit_basic() {
        // 1 << 64 should set bit 64
        let result = make_128bit_shl(0, 1, 64);
        assert_eq!(result, EvalResult::Bv128(1u128 << 64));
    }

    #[test]
    fn test_bvshl_128bit_shift_by_zero() {
        // Shift by 0 = identity
        let result = make_128bit_shl(0xDEAD, 0xBEEF, 0);
        let expected = (0xDEADu128 << 64) | 0xBEEFu128;
        assert_eq!(result, EvalResult::Bv128(expected));
    }

    #[test]
    fn test_bvshl_128bit_shift_by_width_minus_1() {
        // 1 << 127 should produce the sign bit
        let result = make_128bit_shl(0, 1, 127);
        assert_eq!(result, EvalResult::Bv128(1u128 << 127));
    }

    #[test]
    fn test_bvshl_128bit_shift_by_width() {
        // Shift by >= width produces 0
        let result = make_128bit_shl(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF, 128);
        assert_eq!(result, EvalResult::Bv128(0));
    }

    #[test]
    fn test_bvshl_128bit_shift_exceeds_width() {
        // Shift by > width produces 0
        let result = make_128bit_shl(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF, 200);
        assert_eq!(result, EvalResult::Bv128(0));
    }

    /// Helper: build a 128-bit BvLshr expression.
    fn make_128bit_lshr(hi_val: u64, lo_val: u64, shift: u64) -> EvalResult {
        let hi = SmtExpr::var("hi", 64);
        let lo = SmtExpr::var("lo", 64);
        let vec128 = hi.concat(lo);
        let shift_amt = SmtExpr::bv_const(shift, 128);
        let expr = SmtExpr::BvLshr {
            lhs: Box::new(vec128),
            rhs: Box::new(shift_amt),
            width: 128,
        };
        expr.eval(&env(&[("hi", hi_val), ("lo", lo_val)]))
    }

    #[test]
    fn test_bvlshr_128bit_basic() {
        // (1 << 64) >> 64 = 1
        let result = make_128bit_lshr(1, 0, 64);
        assert_eq!(result, EvalResult::Bv128(1));
    }

    #[test]
    fn test_bvlshr_128bit_shift_by_zero() {
        let result = make_128bit_lshr(0xDEAD, 0xBEEF, 0);
        let expected = (0xDEADu128 << 64) | 0xBEEFu128;
        assert_eq!(result, EvalResult::Bv128(expected));
    }

    #[test]
    fn test_bvlshr_128bit_shift_by_width_minus_1() {
        // All 1s >> 127 = 1 (only the top bit survives)
        let result = make_128bit_lshr(u64::MAX, u64::MAX, 127);
        assert_eq!(result, EvalResult::Bv128(1));
    }

    #[test]
    fn test_bvlshr_128bit_shift_by_width() {
        let result = make_128bit_lshr(u64::MAX, u64::MAX, 128);
        assert_eq!(result, EvalResult::Bv128(0));
    }

    /// Helper: build a 128-bit BvAshr expression.
    fn make_128bit_ashr(hi_val: u64, lo_val: u64, shift: u64) -> EvalResult {
        let hi = SmtExpr::var("hi", 64);
        let lo = SmtExpr::var("lo", 64);
        let vec128 = hi.concat(lo);
        let shift_amt = SmtExpr::bv_const(shift, 128);
        let expr = SmtExpr::BvAshr {
            lhs: Box::new(vec128),
            rhs: Box::new(shift_amt),
            width: 128,
        };
        expr.eval(&env(&[("hi", hi_val), ("lo", lo_val)]))
    }

    #[test]
    fn test_bvashr_128bit_positive() {
        // Positive value (MSB = 0): same as logical shift right
        let result = make_128bit_ashr(0x7FFF_FFFF_FFFF_FFFF, 0, 64);
        // 0x7FFFFFFFFFFFFFFF_0000000000000000 >> 64 = 0x7FFFFFFFFFFFFFFF
        assert_eq!(result, EvalResult::Bv128(0x7FFF_FFFF_FFFF_FFFF));
    }

    #[test]
    fn test_bvashr_128bit_negative() {
        // Negative value (MSB = 1): sign-extends with 1s
        // All 1s >> 1 should stay all 1s (arithmetic)
        let result = make_128bit_ashr(u64::MAX, u64::MAX, 1);
        assert_eq!(result, EvalResult::Bv128(u128::MAX)); // all 1s
    }

    #[test]
    fn test_bvashr_128bit_negative_shift_by_width() {
        // Negative >> width = all 1s (sign fill)
        let result = make_128bit_ashr(0x8000_0000_0000_0000, 0, 128);
        assert_eq!(result, EvalResult::Bv128(u128::MAX));
    }

    #[test]
    fn test_bvashr_128bit_positive_shift_by_width() {
        // Positive >> width = 0
        let result = make_128bit_ashr(0x7FFF_FFFF_FFFF_FFFF, u64::MAX, 128);
        assert_eq!(result, EvalResult::Bv128(0));
    }

    #[test]
    fn test_bvashr_128bit_shift_by_zero() {
        let result = make_128bit_ashr(0xDEAD, 0xBEEF, 0);
        let expected = (0xDEADu128 << 64) | 0xBEEFu128;
        assert_eq!(result, EvalResult::Bv128(expected));
    }

    #[test]
    fn test_bvshl_128bit_mixed_operands() {
        // One operand from Concat (Bv128), shift amount from BvConst (Bv).
        // This tests the as_u128() promotion on Bv values.
        let hi = SmtExpr::bv_const(0, 64);
        let lo = SmtExpr::bv_const(0xFF, 64);
        let vec128 = hi.concat(lo); // Bv128
        // Shift amount is a plain 128-bit const (internally Bv since value fits u64)
        let shift = SmtExpr::bv_const(8, 128);
        let expr = SmtExpr::BvShl {
            lhs: Box::new(vec128),
            rhs: Box::new(shift),
            width: 128,
        };
        let result = expr.eval(&env(&[]));
        assert_eq!(result, EvalResult::Bv128(0xFF00));
    }

    #[test]
    fn test_bvlshr_128bit_mixed_operands() {
        let hi = SmtExpr::bv_const(0, 64);
        let lo = SmtExpr::bv_const(0xFF00, 64);
        let vec128 = hi.concat(lo);
        let shift = SmtExpr::bv_const(8, 128);
        let expr = SmtExpr::BvLshr {
            lhs: Box::new(vec128),
            rhs: Box::new(shift),
            width: 128,
        };
        let result = expr.eval(&env(&[]));
        assert_eq!(result, EvalResult::Bv128(0xFF));
    }

    #[test]
    fn test_bvashr_128bit_mixed_operands() {
        // Negative 128-bit value with sign bit set, shift from BvConst
        let hi = SmtExpr::bv_const(0x8000_0000_0000_0000, 64);
        let lo = SmtExpr::bv_const(0, 64);
        let vec128 = hi.concat(lo); // MSB set = negative
        let shift = SmtExpr::bv_const(64, 128);
        let expr = SmtExpr::BvAshr {
            lhs: Box::new(vec128),
            rhs: Box::new(shift),
            width: 128,
        };
        let result = expr.eval(&env(&[]));
        // Arithmetic shift right of a negative value:
        // 0x80000000_00000000_00000000_00000000 (i128::MIN) >> 64 (arithmetic)
        // = 0xFFFFFFFF_FFFFFFFF_80000000_00000000
        // The upper 64 bits fill with 1s (sign extension), the original MSB
        // (0x80000000_00000000) moves to the lower 64 bits.
        let expected: u128 = 0xFFFF_FFFF_FFFF_FFFF_8000_0000_0000_0000;
        assert_eq!(result, EvalResult::Bv128(expected));
    }

    #[test]
    fn test_sign_extend128_helper() {
        // -1 in 8 bits = 0xFF
        assert_eq!(sign_extend128(0xFF, 8), -1i128);
        // -128 in 8 bits = 0x80
        assert_eq!(sign_extend128(0x80, 8), -128i128);
        // Positive: 0x7F in 8 bits = 127
        assert_eq!(sign_extend128(0x7F, 8), 127i128);
        // Full width: passthrough
        assert_eq!(sign_extend128(u128::MAX, 128), -1i128);
        // Zero width
        assert_eq!(sign_extend128(0xFF, 0), 0i128);
    }

    // -----------------------------------------------------------------------
    // QF_FP extended operations tests (#123)
    // -----------------------------------------------------------------------

    #[test]
    fn test_fp_sqrt() {
        let a = SmtExpr::fp64_const(9.0);
        let expr = SmtExpr::fp_sqrt(RoundingMode::RNE, a);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Float(3.0));
    }

    #[test]
    fn test_fp_sqrt_display() {
        let expr = SmtExpr::fp_sqrt(RoundingMode::RTZ, SmtExpr::fp64_const(4.0));
        assert_eq!(format!("{}", expr).split_once(' ').unwrap().0, "(fp.sqrt");
    }

    #[test]
    fn test_fp_abs_positive() {
        let a = SmtExpr::fp64_const(3.5);
        let expr = a.fp_abs();
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Float(3.5));
    }

    #[test]
    fn test_fp_abs_negative() {
        let a = SmtExpr::fp64_const(-7.25);
        let expr = a.fp_abs();
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Float(7.25));
    }

    #[test]
    fn test_fp_abs_display() {
        let expr = SmtExpr::fp64_const(-1.0).fp_abs();
        assert!(format!("{}", expr).starts_with("(fp.abs"));
    }

    #[test]
    fn test_fp_fma() {
        // fma(2.0, 3.0, 4.0) = 2.0 * 3.0 + 4.0 = 10.0
        let expr = SmtExpr::fp_fma(
            RoundingMode::RNE,
            SmtExpr::fp64_const(2.0),
            SmtExpr::fp64_const(3.0),
            SmtExpr::fp64_const(4.0),
        );
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Float(10.0));
    }

    #[test]
    fn test_fp_fma_display() {
        let expr = SmtExpr::fp_fma(
            RoundingMode::RNE,
            SmtExpr::fp64_const(1.0),
            SmtExpr::fp64_const(2.0),
            SmtExpr::fp64_const(3.0),
        );
        assert!(format!("{}", expr).starts_with("(fp.fma RNE"));
    }

    #[test]
    fn test_fp_gt() {
        let a = SmtExpr::fp64_const(3.0);
        let b = SmtExpr::fp64_const(2.0);
        let expr = a.fp_gt(b);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_fp_ge() {
        let a = SmtExpr::fp64_const(2.0);
        let b = SmtExpr::fp64_const(2.0);
        let expr = a.fp_ge(b);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_fp_le() {
        let a = SmtExpr::fp64_const(1.0);
        let b = SmtExpr::fp64_const(2.0);
        let expr = a.fp_le(b);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_fp_comparison_display() {
        let a = SmtExpr::fp64_const(1.0);
        let b = SmtExpr::fp64_const(2.0);
        assert!(format!("{}", a.clone().fp_gt(b.clone())).starts_with("(fp.gt"));
        assert!(format!("{}", a.clone().fp_ge(b.clone())).starts_with("(fp.geq"));
        assert!(format!("{}", a.fp_le(b)).starts_with("(fp.leq"));
    }

    #[test]
    fn test_fp_is_nan() {
        let nan = SmtExpr::fp64_const(f64::NAN);
        let expr = nan.fp_is_nan();
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bool(true));

        let normal = SmtExpr::fp64_const(1.0);
        let expr2 = normal.fp_is_nan();
        let result2 = expr2.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result2, EvalResult::Bool(false));
    }

    #[test]
    fn test_fp_is_inf() {
        let inf = SmtExpr::fp64_const(f64::INFINITY);
        let expr = inf.fp_is_inf();
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bool(true));

        let neg_inf = SmtExpr::fp64_const(f64::NEG_INFINITY);
        let expr2 = neg_inf.fp_is_inf();
        let result2 = expr2.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result2, EvalResult::Bool(true));

        let normal = SmtExpr::fp64_const(42.0);
        let expr3 = normal.fp_is_inf();
        let result3 = expr3.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result3, EvalResult::Bool(false));
    }

    #[test]
    fn test_fp_is_zero() {
        let zero = SmtExpr::fp64_const(0.0);
        let expr = zero.fp_is_zero();
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bool(true));

        let neg_zero = SmtExpr::fp64_const(-0.0);
        let expr2 = neg_zero.fp_is_zero();
        let result2 = expr2.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result2, EvalResult::Bool(true));

        let nonzero = SmtExpr::fp64_const(1.0);
        let expr3 = nonzero.fp_is_zero();
        let result3 = expr3.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result3, EvalResult::Bool(false));
    }

    #[test]
    fn test_fp_is_normal() {
        let normal = SmtExpr::fp64_const(1.0);
        let expr = normal.fp_is_normal();
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bool(true));

        // Subnormals are not normal
        let subnormal = SmtExpr::fp64_const(5e-324);
        let expr2 = subnormal.fp_is_normal();
        let result2 = expr2.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result2, EvalResult::Bool(false));

        // Zero is not normal
        let zero = SmtExpr::fp64_const(0.0);
        let expr3 = zero.fp_is_normal();
        let result3 = expr3.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result3, EvalResult::Bool(false));
    }

    #[test]
    fn test_fp_predicate_display() {
        let a = SmtExpr::fp64_const(1.0);
        assert!(format!("{}", a.clone().fp_is_nan()).starts_with("(fp.isNaN"));
        assert!(format!("{}", a.clone().fp_is_inf()).starts_with("(fp.isInfinite"));
        assert!(format!("{}", a.clone().fp_is_zero()).starts_with("(fp.isZero"));
        assert!(format!("{}", a.fp_is_normal()).starts_with("(fp.isNormal"));
    }

    #[test]
    fn test_fp_to_sbv() {
        let a = SmtExpr::fp64_const(42.7);
        let expr = SmtExpr::fp_to_sbv(RoundingMode::RTZ, a, 32);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bv(42));
    }

    #[test]
    fn test_fp_to_sbv_negative() {
        let a = SmtExpr::fp64_const(-10.9);
        let expr = SmtExpr::fp_to_sbv(RoundingMode::RTZ, a, 32);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        // -10 as u32 = 0xFFFFFFF6
        assert_eq!(result, EvalResult::Bv(mask((-10i64) as u64, 32)));
    }

    #[test]
    fn test_fp_to_ubv() {
        let a = SmtExpr::fp64_const(255.9);
        let expr = SmtExpr::fp_to_ubv(RoundingMode::RTZ, a, 8);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bv(255));
    }

    #[test]
    fn test_fp_to_bv_display() {
        let a = SmtExpr::fp64_const(1.0);
        let sbv = SmtExpr::fp_to_sbv(RoundingMode::RTZ, a.clone(), 32);
        assert!(format!("{}", sbv).starts_with("((_ fp.to_sbv 32)"));

        let ubv = SmtExpr::fp_to_ubv(RoundingMode::RNE, a, 64);
        assert!(format!("{}", ubv).starts_with("((_ fp.to_ubv 64)"));
    }

    #[test]
    fn test_bv_to_fp() {
        let bv = SmtExpr::bv_const(42, 32);
        let expr = SmtExpr::bv_to_fp(RoundingMode::RNE, bv, 8, 24);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Float(42.0f32 as f64));
    }

    #[test]
    fn test_bv_to_fp_negative() {
        // -1 in 8 bits = 0xFF
        let bv = SmtExpr::bv_const(0xFF, 8);
        let expr = SmtExpr::bv_to_fp(RoundingMode::RNE, bv, 11, 53);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Float(-1.0));
    }

    #[test]
    fn test_bv_to_fp_display() {
        let bv = SmtExpr::bv_const(42, 32);
        let expr = SmtExpr::bv_to_fp(RoundingMode::RNE, bv, 8, 24);
        assert!(format!("{}", expr).starts_with("((_ to_fp 8 24)"));
    }

    #[test]
    fn test_fp_to_fp_downcast() {
        // FP64 -> FP32 (lossy conversion)
        let a = SmtExpr::fp64_const(1.5);
        let expr = SmtExpr::fp_to_fp(RoundingMode::RNE, a, 8, 24);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Float(1.5f32 as f64));
    }

    #[test]
    fn test_fp_to_fp_display() {
        let a = SmtExpr::fp64_const(1.0);
        let expr = SmtExpr::fp_to_fp(RoundingMode::RTZ, a, 5, 11);
        assert!(format!("{}", expr).starts_with("((_ to_fp 5 11)"));
    }

    #[test]
    fn test_fp_sort_new_variants() {
        // FPSqrt returns same FP sort as operand
        let sqrt = SmtExpr::fp_sqrt(RoundingMode::RNE, SmtExpr::fp32_const(4.0));
        assert_eq!(sqrt.sort(), SmtSort::FloatingPoint(8, 24));

        // FPAbs returns same FP sort as operand
        let abs = SmtExpr::fp64_const(-1.0).fp_abs();
        assert_eq!(abs.sort(), SmtSort::FloatingPoint(11, 53));

        // FPFma returns same FP sort as first operand
        let fma = SmtExpr::fp_fma(
            RoundingMode::RNE,
            SmtExpr::fp32_const(1.0),
            SmtExpr::fp32_const(2.0),
            SmtExpr::fp32_const(3.0),
        );
        assert_eq!(fma.sort(), SmtSort::FloatingPoint(8, 24));

        // FP predicates return Bool
        assert_eq!(SmtExpr::fp64_const(1.0).fp_is_nan().sort(), SmtSort::Bool);
        assert_eq!(SmtExpr::fp64_const(1.0).fp_is_inf().sort(), SmtSort::Bool);
        assert_eq!(SmtExpr::fp64_const(1.0).fp_is_zero().sort(), SmtSort::Bool);
        assert_eq!(SmtExpr::fp64_const(1.0).fp_is_normal().sort(), SmtSort::Bool);
        assert_eq!(SmtExpr::fp64_const(1.0).fp_gt(SmtExpr::fp64_const(2.0)).sort(), SmtSort::Bool);
        assert_eq!(SmtExpr::fp64_const(1.0).fp_ge(SmtExpr::fp64_const(2.0)).sort(), SmtSort::Bool);
        assert_eq!(SmtExpr::fp64_const(1.0).fp_le(SmtExpr::fp64_const(2.0)).sort(), SmtSort::Bool);

        // FP-to-BV conversions return BitVec
        let to_sbv = SmtExpr::fp_to_sbv(RoundingMode::RTZ, SmtExpr::fp64_const(1.0), 32);
        assert_eq!(to_sbv.sort(), SmtSort::BitVec(32));
        let to_ubv = SmtExpr::fp_to_ubv(RoundingMode::RTZ, SmtExpr::fp64_const(1.0), 64);
        assert_eq!(to_ubv.sort(), SmtSort::BitVec(64));

        // BV-to-FP and FP-to-FP conversions return FloatingPoint
        let bv_to_fp = SmtExpr::bv_to_fp(RoundingMode::RNE, SmtExpr::bv_const(42, 32), 8, 24);
        assert_eq!(bv_to_fp.sort(), SmtSort::FloatingPoint(8, 24));
        let fp_to_fp = SmtExpr::fp_to_fp(RoundingMode::RNE, SmtExpr::fp64_const(1.0), 5, 11);
        assert_eq!(fp_to_fp.sort(), SmtSort::FloatingPoint(5, 11));
    }

    #[test]
    fn test_fp_free_vars_new_variants() {
        // FPFma with vars
        let fma = SmtExpr::fp_fma(
            RoundingMode::RNE,
            SmtExpr::fp64_const(1.0),
            SmtExpr::fp64_const(2.0),
            SmtExpr::fp64_const(3.0),
        );
        assert!(fma.free_vars().is_empty());

        // FPSqrt, FPAbs with constants have no free vars
        let sqrt = SmtExpr::fp_sqrt(RoundingMode::RNE, SmtExpr::fp64_const(4.0));
        assert!(sqrt.free_vars().is_empty());

        let abs = SmtExpr::fp64_const(-1.0).fp_abs();
        assert!(abs.free_vars().is_empty());
    }

    #[test]
    fn test_fp_to_sbv_bv_width() {
        // FPToSBv should have a BV width.
        let expr = SmtExpr::fp_to_sbv(RoundingMode::RTZ, SmtExpr::fp64_const(1.0), 32);
        assert_eq!(expr.try_bv_width().unwrap(), 32);
    }

    #[test]
    fn test_fp_to_ubv_bv_width() {
        let expr = SmtExpr::fp_to_ubv(RoundingMode::RTZ, SmtExpr::fp64_const(1.0), 16);
        assert_eq!(expr.try_bv_width().unwrap(), 16);
    }

    #[test]
    fn test_fp_new_ops_no_bv_width() {
        // FP operations should return BoolHasNoWidth.
        assert!(SmtExpr::fp_sqrt(RoundingMode::RNE, SmtExpr::fp64_const(4.0))
            .try_bv_width().is_err());
        assert!(SmtExpr::fp64_const(-1.0).fp_abs().try_bv_width().is_err());
        assert!(SmtExpr::fp64_const(1.0).fp_is_nan().try_bv_width().is_err());
        assert!(SmtExpr::bv_to_fp(RoundingMode::RNE, SmtExpr::bv_const(0, 32), 8, 24)
            .try_bv_width().is_err());
    }

    // -----------------------------------------------------------------------
    // Bounded quantifier tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_forall_all_true() {
        // ForAll i in [0, 4): i < 4 (always true in unsigned 8-bit)
        let body = SmtExpr::var("i", 8).bvult(SmtExpr::bv_const(4, 8));
        let expr = SmtExpr::forall("i", 8, SmtExpr::bv_const(0, 8), SmtExpr::bv_const(4, 8), body);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_forall_one_false() {
        // ForAll i in [0, 4): i < 3 (false when i=3)
        let body = SmtExpr::var("i", 8).bvult(SmtExpr::bv_const(3, 8));
        let expr = SmtExpr::forall("i", 8, SmtExpr::bv_const(0, 8), SmtExpr::bv_const(4, 8), body);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bool(false));
    }

    #[test]
    fn test_forall_empty_range() {
        // ForAll i in [5, 3): body — empty range is vacuously true.
        let body = SmtExpr::bool_const(false);
        let expr = SmtExpr::forall("i", 8, SmtExpr::bv_const(5, 8), SmtExpr::bv_const(3, 8), body);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_exists_one_true() {
        // Exists i in [0, 4): i == 2 (true for i=2)
        let body = SmtExpr::var("i", 8).eq_expr(SmtExpr::bv_const(2, 8));
        let expr = SmtExpr::exists("i", 8, SmtExpr::bv_const(0, 8), SmtExpr::bv_const(4, 8), body);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_exists_none_true() {
        // Exists i in [0, 4): i == 10 (false for all i in [0,4))
        let body = SmtExpr::var("i", 8).eq_expr(SmtExpr::bv_const(10, 8));
        let expr = SmtExpr::exists("i", 8, SmtExpr::bv_const(0, 8), SmtExpr::bv_const(4, 8), body);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bool(false));
    }

    #[test]
    fn test_exists_empty_range() {
        // Exists i in [5, 3): body — empty range is vacuously false.
        let body = SmtExpr::bool_const(true);
        let expr = SmtExpr::exists("i", 8, SmtExpr::bv_const(5, 8), SmtExpr::bv_const(3, 8), body);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bool(false));
    }

    #[test]
    fn test_forall_with_array() {
        // ForAll i in [0, 3): select(store(store(store(const(0), 0, 42), 1, 42), 2, 42), i) == 42
        let arr = SmtExpr::const_array(SmtSort::BitVec(8), SmtExpr::bv_const(0, 8));
        let arr = SmtExpr::store(arr, SmtExpr::bv_const(0, 8), SmtExpr::bv_const(42, 8));
        let arr = SmtExpr::store(arr, SmtExpr::bv_const(1, 8), SmtExpr::bv_const(42, 8));
        let arr = SmtExpr::store(arr, SmtExpr::bv_const(2, 8), SmtExpr::bv_const(42, 8));
        let body = SmtExpr::select(arr, SmtExpr::var("i", 8))
            .eq_expr(SmtExpr::bv_const(42, 8));
        let expr = SmtExpr::forall("i", 8, SmtExpr::bv_const(0, 8), SmtExpr::bv_const(3, 8), body);
        let result = expr.try_eval(&HashMap::new()).unwrap();
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_forall_with_env() {
        // ForAll i in [0, n): select(arr, i) == val
        // Test with external variable n=2
        let arr = SmtExpr::const_array(SmtSort::BitVec(8), SmtExpr::bv_const(0xFF, 8));
        let body = SmtExpr::select(arr, SmtExpr::var("i", 8))
            .eq_expr(SmtExpr::bv_const(0xFF, 8));
        let expr = SmtExpr::forall(
            "i", 8,
            SmtExpr::bv_const(0, 8),
            SmtExpr::var("n", 8),
            body,
        );
        let result = expr.try_eval(&env(&[("n", 5)])).unwrap();
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_forall_range_too_large() {
        let body = SmtExpr::bool_const(true);
        let expr = SmtExpr::forall("i", 32, SmtExpr::bv_const(0, 32), SmtExpr::bv_const(1000, 32), body);
        let result = expr.try_eval(&HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_forall_sort_is_bool() {
        let body = SmtExpr::bool_const(true);
        let expr = SmtExpr::forall("i", 8, SmtExpr::bv_const(0, 8), SmtExpr::bv_const(1, 8), body);
        assert_eq!(expr.sort(), SmtSort::Bool);
    }

    #[test]
    fn test_exists_sort_is_bool() {
        let body = SmtExpr::bool_const(true);
        let expr = SmtExpr::exists("i", 8, SmtExpr::bv_const(0, 8), SmtExpr::bv_const(1, 8), body);
        assert_eq!(expr.sort(), SmtSort::Bool);
    }

    #[test]
    fn test_forall_free_vars() {
        // ForAll i in [0, n): body(i, x)
        // Free vars: n, x (not i)
        let body = SmtExpr::var("i", 8).bvadd(SmtExpr::var("x", 8))
            .eq_expr(SmtExpr::bv_const(0, 8));
        let expr = SmtExpr::forall(
            "i", 8,
            SmtExpr::bv_const(0, 8),
            SmtExpr::var("n", 8),
            body,
        );
        let vars = expr.free_vars();
        assert!(vars.contains(&"n".to_string()));
        assert!(vars.contains(&"x".to_string()));
        assert!(!vars.contains(&"i".to_string()));
    }

    #[test]
    fn test_forall_display() {
        let body = SmtExpr::var("i", 32).bvult(SmtExpr::var("n", 32));
        let expr = SmtExpr::forall(
            "i", 32,
            SmtExpr::bv_const(0, 32),
            SmtExpr::bv_const(10, 32),
            body,
        );
        let s = format!("{}", expr);
        assert!(s.contains("forall"));
        assert!(s.contains("(_ BitVec 32)"));
        assert!(s.contains("bvuge"));
        assert!(s.contains("bvult"));
    }

    #[test]
    fn test_exists_display() {
        let body = SmtExpr::var("i", 8).eq_expr(SmtExpr::bv_const(5, 8));
        let expr = SmtExpr::exists(
            "i", 8,
            SmtExpr::bv_const(0, 8),
            SmtExpr::bv_const(10, 8),
            body,
        );
        let s = format!("{}", expr);
        assert!(s.contains("exists"));
        assert!(s.contains("(_ BitVec 8)"));
        assert!(s.contains("bvuge"));
        assert!(s.contains("bvult"));
    }

    // -----------------------------------------------------------------------
    // NaN-aware equality (#388)
    // -----------------------------------------------------------------------

    #[test]
    fn test_semantically_equal_nan_vs_nan() {
        // Canonical NaN from 0.0/0.0.
        let a = EvalResult::Float(0.0_f64 / 0.0_f64);
        let b = EvalResult::Float(0.0_f64 / 0.0_f64);
        // IEEE-754: NaN != NaN. PartialEq reflects that and reports not-equal.
        assert!(!(a == b));
        // Semantic equality: both NaN = equal for verification purposes.
        assert!(a.semantically_equal(&b));
    }

    #[test]
    fn test_semantically_equal_distinct_nan_payloads() {
        // f64::NAN and a custom NaN bit pattern are both NaN and must compare equal.
        let a = EvalResult::Float(f64::NAN);
        let b = EvalResult::Float(f64::from_bits(0x7ff0_0000_0000_0001));
        assert!(b.as_f64().is_nan());
        assert!(a.semantically_equal(&b));
        assert!(b.semantically_equal(&a));
    }

    #[test]
    fn test_semantically_equal_nan_vs_finite() {
        let nan = EvalResult::Float(f64::NAN);
        let zero = EvalResult::Float(0.0);
        let one = EvalResult::Float(1.0);
        assert!(!nan.semantically_equal(&zero));
        assert!(!zero.semantically_equal(&nan));
        assert!(!nan.semantically_equal(&one));
    }

    #[test]
    fn test_semantically_equal_finite_bit_exact() {
        // Zero sign matters for bit-exact comparison (so +0 != -0, matching
        // AArch64 FMOV bit-pattern semantics).
        let plus_zero = EvalResult::Float(0.0);
        let neg_zero = EvalResult::Float(-0.0);
        assert!(!plus_zero.semantically_equal(&neg_zero));

        // Equal finite values with identical bit pattern.
        let a = EvalResult::Float(3.14);
        let b = EvalResult::Float(3.14);
        assert!(a.semantically_equal(&b));
    }

    #[test]
    fn test_semantically_equal_non_float_unchanged() {
        let a = EvalResult::Bv(42);
        let b = EvalResult::Bv(42);
        let c = EvalResult::Bv(7);
        assert!(a.semantically_equal(&b));
        assert!(!a.semantically_equal(&c));

        let t = EvalResult::Bool(true);
        let f = EvalResult::Bool(false);
        assert!(t.semantically_equal(&EvalResult::Bool(true)));
        assert!(!t.semantically_equal(&f));
    }
}
