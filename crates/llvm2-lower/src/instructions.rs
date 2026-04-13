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
    Urem,
    Srem,
    Ineg,   // Integer negate: result = -operand

    // Bitwise unary
    Bnot,   // Bitwise NOT: result = ~operand

    // Floating-point unary
    Fneg,   // Floating-point negate: result = -operand

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
    FcvtToInt { dst_ty: Type },    // Float -> signed Int conversion (FCVTZS)
    FcvtToUint { dst_ty: Type },   // Float -> unsigned Int conversion (FCVTZU)
    FcvtFromInt { src_ty: Type },  // Signed Int -> Float conversion (SCVTF)
    FcvtFromUint { src_ty: Type }, // Unsigned Int -> Float conversion (UCVTF)
    FPExt,                         // Float precision widen (f32 -> f64)
    FPTrunc,                       // Float precision narrow (f64 -> f32)

    // Type conversions
    Trunc { to_ty: Type },         // Integer truncation (narrow: i64->i32, etc.)
    Bitcast { to_ty: Type },       // Reinterpret bits between same-size types

    // Addressing
    GlobalRef { name: String },     // Reference to a local global symbol (ADRP + ADD)
    ExternRef { name: String },     // Reference to an external symbol via GOT (ADRP + LDR from GOT)
    TlsRef { name: String },        // Reference to a thread-local variable via TLV descriptor
    StackAddr { slot: u32 },        // Address of a stack slot (SP + offset)

    // Control flow
    Jump { dest: Block },
    Brif { cond: Value, then_dest: Block, else_dest: Block },
    Return,
    Call { name: String },          // Direct function call by symbol name
    /// Indirect function call via a register-held function pointer.
    ///
    /// `args[0]` is the function pointer (I64 address).
    /// `args[1..]` are the call arguments, classified per ABI.
    /// Lowered to BLR on AArch64.
    CallIndirect,
    /// Variadic function call (e.g., printf, NSLog).
    ///
    /// Apple AArch64 ABI: fixed args use normal register/stack classification,
    /// ALL variadic args are placed on the stack (8-byte aligned).
    /// `fixed_args` is the count of non-variadic parameters.
    CallVariadic { name: String, fixed_args: u32 },
    /// Multi-way branch (switch statement).
    ///
    /// `args[0]` is the selector value (integer).
    /// `cases` maps integer values to target blocks.
    /// `default` is the fallthrough block when no case matches.
    /// Lowered as a cascading CMP+B.EQ chain with default fallthrough.
    Switch { cases: Vec<(i64, Block)>, default: Block },

    // Memory
    Load { ty: Type },
    Store,

    // Aggregate operations
    /// Compute address of a struct field: base_ptr + offset_of(struct_ty, field_index).
    /// args[0] = base pointer (pointer to struct), result = pointer to field.
    StructGep { struct_ty: Type, field_index: u32 },
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
