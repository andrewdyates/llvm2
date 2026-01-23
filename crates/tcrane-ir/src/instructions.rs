// tcrane-ir/instructions.rs - IR instruction set
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Instruction definitions for Cranelift IR.

use crate::types::Type;
use serde::{Deserialize, Serialize};

/// A value reference in the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Value(pub u32);

/// A basic block reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Block(pub u32);

/// Cranelift IR opcodes (subset).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Opcode {
    // Constants
    Iconst { ty: Type, imm: i64 },
    Fconst { ty: Type, imm: f64 },

    // Arithmetic
    Iadd,
    Isub,
    Imul,
    Udiv,
    Sdiv,

    // Comparisons
    Icmp { cond: IntCC },

    // Control flow
    Jump { dest: Block },
    Brif { cond: Value, then_dest: Block, else_dest: Block },
    Return,

    // Memory
    Load { ty: Type },
    Store,
}

/// Integer comparison conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntCC {
    Equal,
    NotEqual,
    SignedLessThan,
    SignedGreaterThanOrEqual,
    SignedGreaterThan,
    SignedLessThanOrEqual,
    UnsignedLessThan,
    UnsignedGreaterThanOrEqual,
    UnsignedGreaterThan,
    UnsignedLessThanOrEqual,
}

/// An instruction in the IR.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Instruction {
    pub opcode: Opcode,
    pub args: Vec<Value>,
    pub results: Vec<Value>,
}
