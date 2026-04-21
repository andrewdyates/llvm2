// llvm2-dialect - Sample `tmir` dialect (facade over real tMIR)
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Minimal `tmir` dialect used by the PoC lowering pipeline.
//!
//! This is a *facade* — the real tMIR ops live in the `tmir` crate and come in
//! through `llvm2-lower`'s adapter. The dialect wrapper here defines the op
//! identifiers that conversion patterns use as targets for `VerifToTmir` and as
//! sources for `TmirToMachir`. Future work will replace these with direct
//! translation against the real tMIR crate (see design doc §10).

use crate::dialect::{Arity, Capabilities, Dialect, OpDef, TypeConstraint};
use crate::id::OpCode;

pub const TMIR_CONST: OpCode = OpCode(0);
pub const TMIR_ADD: OpCode = OpCode(1);
pub const TMIR_XOR: OpCode = OpCode(2);
pub const TMIR_RET: OpCode = OpCode(3);

pub struct TmirDialect;

impl TmirDialect {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TmirDialect {
    fn default() -> Self {
        Self::new()
    }
}

// --- Type-constraint slices ------------------------------------------------
//
// Shared static slices keep the `OpDef` table readable and avoid duplicated
// constraint data. Polymorphic binaries (`tmir.add`, `tmir.xor`) accept any
// integer operand pair and produce the same type as `operand[0]` — matching
// the declarative rewrite engine's expectations in #393.

/// `tmir.const` results are `AnyInt` — the concrete width comes from the
/// `value` attribute. Future work (once f32/f64 literals land) expands this.
static TMIR_CONST_RESULTS: &[TypeConstraint] = &[TypeConstraint::AnyInt];

/// Binary integer ops: both operands are `AnyInt`; the result width matches
/// `operand[0]`.
static TMIR_BINARY_OPERANDS: &[TypeConstraint] =
    &[TypeConstraint::AnyInt, TypeConstraint::SameAs(0)];
static TMIR_BINARY_RESULTS: &[TypeConstraint] = &[TypeConstraint::SameAs(0)];

/// `tmir.ret` is variadic (0 or 1 operands). A single-element slice applies
/// the constraint to every operand present.
static TMIR_RET_OPERANDS: &[TypeConstraint] = &[TypeConstraint::AnyScalar];

static TMIR_OPS: &[OpDef] = &[
    OpDef {
        op: TMIR_CONST,
        name: "tmir.const",
        capabilities: Capabilities::PURE,
        num_operands: Arity::Fixed(0),
        num_results: Arity::Fixed(1),
        operand_types: &[],
        result_types: TMIR_CONST_RESULTS,
    },
    OpDef {
        op: TMIR_ADD,
        name: "tmir.add",
        capabilities: Capabilities::PURE,
        num_operands: Arity::Fixed(2),
        num_results: Arity::Fixed(1),
        operand_types: TMIR_BINARY_OPERANDS,
        result_types: TMIR_BINARY_RESULTS,
    },
    OpDef {
        op: TMIR_XOR,
        name: "tmir.xor",
        capabilities: Capabilities::PURE,
        num_operands: Arity::Fixed(2),
        num_results: Arity::Fixed(1),
        operand_types: TMIR_BINARY_OPERANDS,
        result_types: TMIR_BINARY_RESULTS,
    },
    OpDef {
        op: TMIR_RET,
        name: "tmir.ret",
        capabilities: Capabilities(
            Capabilities::IS_TERMINATOR.bits() | Capabilities::HAS_SIDE_EFFECT.bits(),
        ),
        num_operands: Arity::Variadic(Some(1)),
        num_results: Arity::Fixed(0),
        operand_types: TMIR_RET_OPERANDS,
        result_types: &[],
    },
];

impl Dialect for TmirDialect {
    fn namespace(&self) -> &'static str {
        "tmir"
    }
    fn ops(&self) -> &[OpDef] {
        TMIR_OPS
    }
}
