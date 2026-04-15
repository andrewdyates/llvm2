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
    Fabs,   // Floating-point absolute value: result = |operand|
    Fsqrt,  // Floating-point square root: result = sqrt(operand)

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
    /// Invoke — call that may throw an exception.
    ///
    /// Like `Call`, but has two successors: a normal continuation block and
    /// an unwind landing pad block. If the callee returns normally, control
    /// transfers to `normal_dest`. If the callee throws an exception that
    /// is caught by a landing pad in this function, control transfers to
    /// `unwind_dest`.
    ///
    /// `args` are the call arguments, classified per ABI (same as Call).
    /// Results are the call return values (same as Call).
    ///
    /// Lowered to BL on AArch64, but with EH metadata: the call site
    /// gets an entry in the LSDA call site table pointing to the
    /// landing pad at `unwind_dest`.
    ///
    /// Reference: LLVM IR `invoke` instruction
    Invoke {
        name: String,
        normal_dest: Block,
        unwind_dest: Block,
    },

    /// Landing pad — exception handler entry point.
    ///
    /// Marks the beginning of an exception handler block. When the unwinder
    /// dispatches to this landing pad, it provides:
    /// - The exception object pointer (args[0] result, I64)
    /// - The type selector value (args[1] result, I32)
    ///
    /// The type selector is used by downstream code to determine which
    /// catch clause matched (if any), or whether this is a cleanup-only
    /// handler.
    ///
    /// `is_cleanup`: If true, this landing pad runs cleanup code (destructors)
    /// and then resumes unwinding. If false, it catches specific exception types.
    ///
    /// `catch_type_indices`: 1-based indices into the type table for catch
    /// clauses. Index 0 means catch-all. Empty for cleanup-only pads.
    ///
    /// Reference: LLVM IR `landingpad` instruction
    LandingPad {
        is_cleanup: bool,
        catch_type_indices: Vec<u32>,
    },

    /// Resume unwinding — re-throw the current exception.
    ///
    /// Used at the end of a cleanup landing pad to continue unwinding
    /// after executing cleanup code. `args[0]` is the exception object
    /// pointer. Lowered to a call to `_Unwind_Resume`.
    Resume,

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

    // Atomic memory operations
    /// Atomic load with acquire semantics: result = atomic_load(ptr).
    /// args[0] = ptr. Lowered to LDAR on AArch64.
    AtomicLoad { ty: Type, ordering: AtomicOrdering },
    /// Atomic store with release semantics: atomic_store(ptr, value).
    /// args[0] = value, args[1] = ptr. Lowered to STLR on AArch64.
    AtomicStore { ordering: AtomicOrdering },
    /// Atomic read-modify-write: result (old value) = atomic_rmw(op, ptr, val).
    /// args[0] = val, args[1] = ptr. Lowered to LDADD/LDCLR/LDSET/LDEOR/SWP.
    AtomicRmw { op: AtomicRmwOp, ty: Type, ordering: AtomicOrdering },
    /// Compare-and-swap: result (old value) = cmpxchg(ptr, expected, desired).
    /// args[0] = expected, args[1] = desired, args[2] = ptr.
    /// Lowered to CAS (LSE) or LDAXR/STLXR loop (non-LSE).
    CmpXchg { ty: Type, ordering: AtomicOrdering },
    /// Memory fence. No args, no results. Lowered to DMB.
    Fence { ordering: AtomicOrdering },

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

/// Memory ordering for atomic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AtomicOrdering {
    /// No ordering constraint.
    Relaxed,
    /// Acquire: subsequent reads/writes cannot be reordered before this.
    Acquire,
    /// Release: preceding reads/writes cannot be reordered after this.
    Release,
    /// Acquire + Release combined.
    AcqRel,
    /// Sequential consistency (strongest).
    SeqCst,
}

/// Atomic read-modify-write operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AtomicRmwOp {
    /// Atomic add.
    Add,
    /// Atomic subtract.
    Sub,
    /// Atomic AND.
    And,
    /// Atomic OR.
    Or,
    /// Atomic XOR.
    Xor,
    /// Atomic exchange (swap).
    Xchg,
}

/// An instruction in the IR.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Instruction {
    pub opcode: Opcode,
    pub args: Vec<Value>,
    pub results: Vec<Value>,
}
