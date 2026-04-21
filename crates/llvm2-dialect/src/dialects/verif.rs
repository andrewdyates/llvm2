// llvm2-dialect - Sample `verif` dialect
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Minimal `verif` dialect. Provides a single stub op,
//! `verif.fingerprint_batch_stub(ptr, len) -> i64`, used by the end-to-end
//! progressive-lowering test. The op's "semantics" are intentionally trivial
//! (an XOR chain) — the proof-of-concept is the plumbing, not the math.

use llvm2_ir::Type;

use crate::dialect::{Arity, Capabilities, Dialect, OpDef, TypeConstraint};
use crate::id::OpCode;

/// Stable opcode for `verif.fingerprint_batch_stub`. Exposed as a constant so
/// downstream conversion patterns can refer to it without string lookups.
pub const FINGERPRINT_BATCH_STUB: OpCode = OpCode(0);

pub struct VerifDialect;

impl VerifDialect {
    pub fn new() -> Self {
        Self
    }
}

impl Default for VerifDialect {
    fn default() -> Self {
        Self::new()
    }
}

// `verif.fingerprint_batch_stub(ptr: i64, len: i64) -> i64`.
// Today `Ptr` values arrive as `I64` from the adapter; once #391 plumbs through
// a distinct pointer type the first constraint becomes `Specific(Type::Ptr)`.
static FINGERPRINT_OPERAND_TYPES: &[TypeConstraint] =
    &[TypeConstraint::Specific(Type::I64), TypeConstraint::Specific(Type::I64)];
static FINGERPRINT_RESULT_TYPES: &[TypeConstraint] = &[TypeConstraint::Specific(Type::I64)];

static VERIF_OPS: &[OpDef] = &[OpDef {
    op: FINGERPRINT_BATCH_STUB,
    name: "verif.fingerprint_batch_stub",
    capabilities: Capabilities(Capabilities::PURE.bits() | Capabilities::BOUNDED_LOOPS.bits()),
    num_operands: Arity::Fixed(2),
    num_results: Arity::Fixed(1),
    operand_types: FINGERPRINT_OPERAND_TYPES,
    result_types: FINGERPRINT_RESULT_TYPES,
}];

impl Dialect for VerifDialect {
    fn namespace(&self) -> &'static str {
        "verif"
    }
    fn ops(&self) -> &[OpDef] {
        VERIF_OPS
    }
}
