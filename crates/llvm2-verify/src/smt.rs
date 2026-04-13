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
}

impl SmtSort {
    /// Bitvector width, or `None` for Bool.
    pub fn bv_width(&self) -> Option<u32> {
        match self {
            SmtSort::BitVec(w) => Some(*w),
            SmtSort::Bool => None,
        }
    }
}

impl From<Type> for SmtSort {
    fn from(ty: Type) -> Self {
        match ty {
            Type::B1 => SmtSort::BitVec(1),
            Type::I8 => SmtSort::BitVec(8),
            Type::I16 => SmtSort::BitVec(16),
            Type::I32 => SmtSort::BitVec(32),
            Type::I64 => SmtSort::BitVec(64),
            Type::I128 => SmtSort::BitVec(128),
            Type::F32 | Type::F64 => panic!("FP verification not yet supported"),
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

    /// Return the bitvector width of this expression (panics for Bool).
    pub fn bv_width(&self) -> u32 {
        match self {
            SmtExpr::Var { width, .. } => *width,
            SmtExpr::BvConst { width, .. } => *width,
            SmtExpr::BvAdd { width, .. } => *width,
            SmtExpr::BvSub { width, .. } => *width,
            SmtExpr::BvMul { width, .. } => *width,
            SmtExpr::BvSDiv { width, .. } => *width,
            SmtExpr::BvUDiv { width, .. } => *width,
            SmtExpr::BvNeg { width, .. } => *width,
            SmtExpr::Extract { width, .. } => *width,
            SmtExpr::BvAnd { width, .. } => *width,
            SmtExpr::BvOr { width, .. } => *width,
            SmtExpr::BvXor { width, .. } => *width,
            SmtExpr::BvShl { width, .. } => *width,
            SmtExpr::BvLshr { width, .. } => *width,
            SmtExpr::BvAshr { width, .. } => *width,
            SmtExpr::Ite { then_expr, .. } => then_expr.bv_width(),
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
            | SmtExpr::Or { .. } => {
                panic!("bv_width called on Bool-sorted expression")
            }
        }
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
            | SmtExpr::Or { .. } => SmtSort::Bool,
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
            SmtExpr::BvConst { .. } | SmtExpr::BoolConst(_) => {}
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
            | SmtExpr::Or { lhs, rhs } => {
                lhs.collect_vars(vars);
                rhs.collect_vars(vars);
            }
            SmtExpr::BvNeg { operand, .. }
            | SmtExpr::Not { operand }
            | SmtExpr::Extract { operand, .. } => {
                operand.collect_vars(vars);
            }
            SmtExpr::Ite { cond, then_expr, else_expr } => {
                cond.collect_vars(vars);
                then_expr.collect_vars(vars);
                else_expr.collect_vars(vars);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Concrete evaluation
// ---------------------------------------------------------------------------

/// Evaluation result: either a bitvector value or a boolean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalResult {
    Bv(u64),
    Bool(bool),
}

impl EvalResult {
    pub fn as_u64(self) -> u64 {
        match self {
            EvalResult::Bv(v) => v,
            EvalResult::Bool(b) => b as u64,
        }
    }

    pub fn as_bool(self) -> bool {
        match self {
            EvalResult::Bool(b) => b,
            EvalResult::Bv(v) => v != 0,
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
    /// Evaluate this expression under the given variable assignment.
    ///
    /// Variables map name -> u64 value (already masked to width).
    /// Panics if a variable is not found in the environment.
    pub fn eval(&self, env: &HashMap<String, u64>) -> EvalResult {
        match self {
            SmtExpr::Var { name, width } => {
                let v = env.get(name).unwrap_or_else(|| {
                    panic!("variable '{}' not found in environment", name)
                });
                EvalResult::Bv(mask(*v, *width))
            }
            SmtExpr::BvConst { value, width } => {
                EvalResult::Bv(mask(*value, *width))
            }
            SmtExpr::BoolConst(b) => EvalResult::Bool(*b),

            SmtExpr::BvAdd { lhs, rhs, width } => {
                let a = lhs.eval(env).as_u64();
                let b = rhs.eval(env).as_u64();
                EvalResult::Bv(mask(a.wrapping_add(b), *width))
            }
            SmtExpr::BvSub { lhs, rhs, width } => {
                let a = lhs.eval(env).as_u64();
                let b = rhs.eval(env).as_u64();
                EvalResult::Bv(mask(a.wrapping_sub(b), *width))
            }
            SmtExpr::BvMul { lhs, rhs, width } => {
                let a = lhs.eval(env).as_u64();
                let b = rhs.eval(env).as_u64();
                EvalResult::Bv(mask(a.wrapping_mul(b), *width))
            }
            SmtExpr::BvSDiv { lhs, rhs, width } => {
                let a = sign_extend(lhs.eval(env).as_u64(), *width);
                let b = sign_extend(rhs.eval(env).as_u64(), *width);
                if b == 0 {
                    // SMT-LIB: bvsdiv by zero is defined (returns all-ones
                    // for positive dividend, etc.). For verification we gate
                    // on b != 0 as a precondition, but we still need a defined
                    // value here. Return 0 as sentinel.
                    EvalResult::Bv(0)
                } else if a == i64::MIN && b == -1 && *width == 64 {
                    // Overflow: INT_MIN / -1.
                    EvalResult::Bv(mask(a as u64, *width))
                } else {
                    let result = a.wrapping_div(b);
                    EvalResult::Bv(mask(result as u64, *width))
                }
            }
            SmtExpr::BvUDiv { lhs, rhs, width } => {
                let a = lhs.eval(env).as_u64();
                let b = rhs.eval(env).as_u64();
                if b == 0 {
                    EvalResult::Bv(0) // sentinel, gated by precondition
                } else {
                    EvalResult::Bv(mask(a / b, *width))
                }
            }
            SmtExpr::BvAnd { lhs, rhs, width } => {
                let a = lhs.eval(env).as_u64();
                let b = rhs.eval(env).as_u64();
                EvalResult::Bv(mask(a & b, *width))
            }
            SmtExpr::BvOr { lhs, rhs, width } => {
                let a = lhs.eval(env).as_u64();
                let b = rhs.eval(env).as_u64();
                EvalResult::Bv(mask(a | b, *width))
            }
            SmtExpr::BvXor { lhs, rhs, width } => {
                let a = lhs.eval(env).as_u64();
                let b = rhs.eval(env).as_u64();
                EvalResult::Bv(mask(a ^ b, *width))
            }
            SmtExpr::BvShl { lhs, rhs, width } => {
                let a = lhs.eval(env).as_u64();
                let b = rhs.eval(env).as_u64();
                // SMT-LIB: if shift amount >= width, result is 0.
                if b >= *width as u64 {
                    EvalResult::Bv(0)
                } else {
                    EvalResult::Bv(mask(a << b, *width))
                }
            }
            SmtExpr::BvLshr { lhs, rhs, width } => {
                let a = lhs.eval(env).as_u64();
                let b = rhs.eval(env).as_u64();
                if b >= *width as u64 {
                    EvalResult::Bv(0)
                } else {
                    EvalResult::Bv(mask(a >> b, *width))
                }
            }
            SmtExpr::BvAshr { lhs, rhs, width } => {
                let a = sign_extend(lhs.eval(env).as_u64(), *width);
                let b = rhs.eval(env).as_u64();
                if b >= *width as u64 {
                    // Sign-fill: all 1s if negative, all 0s if positive.
                    if a < 0 {
                        EvalResult::Bv(mask(u64::MAX, *width))
                    } else {
                        EvalResult::Bv(0)
                    }
                } else {
                    EvalResult::Bv(mask((a >> b) as u64, *width))
                }
            }
            SmtExpr::BvNeg { operand, width } => {
                let a = operand.eval(env).as_u64();
                // Two's complement negation = wrapping negate.
                EvalResult::Bv(mask((!a).wrapping_add(1), *width))
            }

            SmtExpr::Eq { lhs, rhs } => {
                let a = lhs.eval(env);
                let b = rhs.eval(env);
                EvalResult::Bool(a == b)
            }
            SmtExpr::Not { operand } => {
                EvalResult::Bool(!operand.eval(env).as_bool())
            }
            SmtExpr::BvSlt { lhs, rhs, width } => {
                let a = sign_extend(lhs.eval(env).as_u64(), *width);
                let b = sign_extend(rhs.eval(env).as_u64(), *width);
                EvalResult::Bool(a < b)
            }
            SmtExpr::BvSge { lhs, rhs, width } => {
                let a = sign_extend(lhs.eval(env).as_u64(), *width);
                let b = sign_extend(rhs.eval(env).as_u64(), *width);
                EvalResult::Bool(a >= b)
            }
            SmtExpr::BvUge { lhs, rhs, .. } => {
                let a = lhs.eval(env).as_u64();
                let b = rhs.eval(env).as_u64();
                EvalResult::Bool(a >= b)
            }
            SmtExpr::BvSgt { lhs, rhs, width } => {
                let a = sign_extend(lhs.eval(env).as_u64(), *width);
                let b = sign_extend(rhs.eval(env).as_u64(), *width);
                EvalResult::Bool(a > b)
            }
            SmtExpr::BvSle { lhs, rhs, width } => {
                let a = sign_extend(lhs.eval(env).as_u64(), *width);
                let b = sign_extend(rhs.eval(env).as_u64(), *width);
                EvalResult::Bool(a <= b)
            }
            SmtExpr::BvUlt { lhs, rhs, .. } => {
                let a = lhs.eval(env).as_u64();
                let b = rhs.eval(env).as_u64();
                EvalResult::Bool(a < b)
            }
            SmtExpr::BvUgt { lhs, rhs, .. } => {
                let a = lhs.eval(env).as_u64();
                let b = rhs.eval(env).as_u64();
                EvalResult::Bool(a > b)
            }
            SmtExpr::BvUle { lhs, rhs, .. } => {
                let a = lhs.eval(env).as_u64();
                let b = rhs.eval(env).as_u64();
                EvalResult::Bool(a <= b)
            }
            SmtExpr::And { lhs, rhs } => {
                EvalResult::Bool(lhs.eval(env).as_bool() && rhs.eval(env).as_bool())
            }
            SmtExpr::Or { lhs, rhs } => {
                EvalResult::Bool(lhs.eval(env).as_bool() || rhs.eval(env).as_bool())
            }
            SmtExpr::Ite { cond, then_expr, else_expr } => {
                if cond.eval(env).as_bool() {
                    then_expr.eval(env)
                } else {
                    else_expr.eval(env)
                }
            }
            SmtExpr::Extract { high, low, operand, width } => {
                let v = operand.eval(env).as_u64();
                let extracted = (v >> low) & ((1u64 << width) - 1);
                let _ = high; // used in width calculation
                EvalResult::Bv(extracted)
            }
        }
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
        }
    }
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
}
