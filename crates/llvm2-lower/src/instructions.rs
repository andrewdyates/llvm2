// llvm2-lower/instructions.rs - LIR instruction set
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Instruction definitions for LLVM2 Low-level IR.

use crate::types::Type;
use serde::{Deserialize, Serialize};

/// A value reference in the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Value(pub u32);

/// A basic block reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Block(pub u32);

/// LIR opcodes.
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

    // Shift operations
    Ishl,   // Logical shift left
    Ushr,   // Logical shift right (unsigned)
    Sshr,   // Arithmetic shift right (signed)

    // Logical operations
    Band,   // Bitwise AND
    Bor,    // Bitwise OR
    Bxor,   // Bitwise XOR
    BandNot, // Bitwise AND-NOT (BIC)
    BorNot,  // Bitwise OR-NOT (ORN)

    // Extensions
    Sextend { from_ty: Type, to_ty: Type },   // Sign-extend
    Uextend { from_ty: Type, to_ty: Type },   // Zero-extend

    // Bitfield operations
    ExtractBits { lsb: u8, width: u8 },        // Unsigned bitfield extract (UBFM)
    SextractBits { lsb: u8, width: u8 },       // Signed bitfield extract (SBFM)
    InsertBits { lsb: u8, width: u8 },         // Bitfield insert (BFM)

    // Conditional select
    Select { cond: IntCC },   // csel(cond, lhs, rhs, cc_val) -> result

    // Comparisons
    Icmp { cond: IntCC },

    // Floating-point arithmetic
    Fadd,
    Fsub,
    Fmul,
    Fdiv,
    Fcmp { cond: FloatCC },
    FcvtToInt { dst_ty: Type },    // Float -> Int conversion
    FcvtFromInt { src_ty: Type },  // Int -> Float conversion

    // Addressing
    GlobalRef { name: String },     // Reference to a global symbol (ADRP + ADD)
    StackAddr { slot: u32 },        // Address of a stack slot (SP + offset)

    // Control flow
    Jump { dest: Block },
    Brif { cond: Value, then_dest: Block, else_dest: Block },
    Return,
    Call { name: String },          // Direct function call by symbol name

    // Memory
    Load { ty: Type },
    Store,
}

/// Floating-point comparison conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FloatCC {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Ordered,    // Neither operand is NaN
    Unordered,  // At least one operand is NaN
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
