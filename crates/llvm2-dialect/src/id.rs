// llvm2-dialect - Typed identifiers for dialects, ops, values, blocks.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Typed `u16`/`u32` wrappers used throughout the dialect framework.

/// Index of a dialect in a [`DialectRegistry`](crate::registry::DialectRegistry).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DialectId(pub u16);

/// Opaque opcode. Interpreted per-dialect — the pair `(DialectId, OpCode)`
/// uniquely identifies an op globally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpCode(pub u16);

/// Globally-unique op handle: dialect-qualified opcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DialectOpId {
    pub dialect: DialectId,
    pub op: OpCode,
}

impl DialectOpId {
    pub const fn new(dialect: DialectId, op: OpCode) -> Self {
        Self { dialect, op }
    }
}

/// SSA value identifier, scoped to a single [`DialectFunction`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

/// Arena index of a [`DialectOp`] within a [`DialectFunction`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpId(pub u32);

/// Arena index of a [`DialectBlock`] within a [`DialectFunction`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);
