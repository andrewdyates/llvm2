// tmir-instrs stub — minimal tMIR instruction set for LLVM2 development
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// This is a development stub. The real tMIR crate (ayates_dbx/tMIR) defines
// the full instruction set. This stub provides the ~25 core instructions
// needed by LLVM2's instruction selector.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use tmir_types::{BlockId, FuncId, TmirProof, Ty, ValueId};

/// Binary arithmetic/logic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    SDiv,
    UDiv,
    SRem,
    URem,
    And,
    Or,
    Xor,
    Shl,
    AShr,
    LShr,
    FAdd,
    FSub,
    FMul,
    FDiv,
}

/// Unary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnOp {
    Neg,
    Not,
    FNeg,
    FAbs,
    FSqrt,
}

/// Integer/float comparison predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CmpOp {
    Eq,
    Ne,
    Slt,
    Sle,
    Sgt,
    Sge,
    Ult,
    Ule,
    Ugt,
    Uge,
    // Float comparisons (ordered)
    FOeq,
    FOne,
    FOlt,
    FOle,
    FOgt,
    FOge,
    // Float comparisons (unordered)
    FUeq,
    FUne,
    FUlt,
    FUle,
    FUgt,
    FUge,
}

/// Type cast operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CastOp {
    /// Zero-extend integer.
    ZExt,
    /// Sign-extend integer.
    SExt,
    /// Truncate integer.
    Trunc,
    /// Float to signed integer.
    FPToSI,
    /// Float to unsigned integer.
    FPToUI,
    /// Signed integer to float.
    SIToFP,
    /// Unsigned integer to float.
    UIToFP,
    /// Float precision conversion.
    FPExt,
    /// Float precision truncation.
    FPTrunc,
    /// Pointer to integer.
    PtrToInt,
    /// Integer to pointer.
    IntToPtr,
    /// Bitcast (same size, different type).
    Bitcast,
}

/// Memory ordering for atomic operations (matches C++/LLVM orderings).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryOrdering {
    /// No ordering constraint (not valid for atomics, only for fences).
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
    /// Atomic add: *ptr += val.
    Add,
    /// Atomic subtract: *ptr -= val.
    Sub,
    /// Atomic AND: *ptr &= val.
    And,
    /// Atomic OR: *ptr |= val.
    Or,
    /// Atomic XOR: *ptr ^= val.
    Xor,
    /// Atomic swap: old = *ptr; *ptr = val.
    Xchg,
}

/// A single switch case: value -> target block.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SwitchCase {
    pub value: i64,
    pub target: BlockId,
}

/// tMIR instruction set.
///
/// These are the ~25 core instructions that LLVM2 must lower to machine code.
/// Each instruction operates on SSA values and produces zero or more results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Instr {
    /// Binary operation: result = lhs op rhs.
    BinOp {
        op: BinOp,
        ty: Ty,
        lhs: ValueId,
        rhs: ValueId,
    },

    /// Unary operation: result = op operand.
    UnOp {
        op: UnOp,
        ty: Ty,
        operand: ValueId,
    },

    /// Comparison: result (Bool) = lhs cmp rhs.
    Cmp {
        op: CmpOp,
        ty: Ty,
        lhs: ValueId,
        rhs: ValueId,
    },

    /// Type cast: result = cast operand from src_ty to dst_ty.
    Cast {
        op: CastOp,
        src_ty: Ty,
        dst_ty: Ty,
        operand: ValueId,
    },

    /// Load from memory: result = *ptr.
    Load {
        ty: Ty,
        ptr: ValueId,
    },

    /// Store to memory: *ptr = value.
    Store {
        ty: Ty,
        ptr: ValueId,
        value: ValueId,
    },

    /// Stack allocation: result = alloca(ty, count).
    Alloc {
        ty: Ty,
        count: Option<ValueId>,
    },

    /// Deallocation hint (for verified memory management).
    Dealloc {
        ptr: ValueId,
    },

    // -- Atomic memory operations --

    /// Atomic load: result = atomic_load(ptr, ordering).
    /// Provides acquire semantics (or stronger).
    AtomicLoad {
        ty: Ty,
        ptr: ValueId,
        ordering: MemoryOrdering,
    },

    /// Atomic store: atomic_store(ptr, value, ordering).
    /// Provides release semantics (or stronger).
    AtomicStore {
        ty: Ty,
        ptr: ValueId,
        value: ValueId,
        ordering: MemoryOrdering,
    },

    /// Atomic read-modify-write: result (old value) = atomic_rmw(op, ptr, val, ordering).
    AtomicRmw {
        op: AtomicRmwOp,
        ty: Ty,
        ptr: ValueId,
        value: ValueId,
        ordering: MemoryOrdering,
    },

    /// Compare and swap: (old_value, success) = cmpxchg(ptr, expected, desired, ordering).
    /// If *ptr == expected, stores desired and returns (expected, true).
    /// Otherwise returns (old_value, false).
    CmpXchg {
        ty: Ty,
        ptr: ValueId,
        expected: ValueId,
        desired: ValueId,
        success_ordering: MemoryOrdering,
        failure_ordering: MemoryOrdering,
    },

    /// Memory fence: enforces ordering constraint without accessing memory.
    Fence {
        ordering: MemoryOrdering,
    },

    // -- Ownership instructions (tMIR-specific) --

    /// Immutable borrow: result = &value.
    Borrow {
        ty: Ty,
        value: ValueId,
    },

    /// Mutable borrow: result = &mut value.
    BorrowMut {
        ty: Ty,
        value: ValueId,
    },

    /// End a borrow lifetime.
    EndBorrow {
        borrow: ValueId,
    },

    /// Increment reference count (ARC).
    Retain {
        value: ValueId,
    },

    /// Decrement reference count (ARC).
    Release {
        value: ValueId,
    },

    /// Check uniqueness of a reference (for COW optimization).
    IsUnique {
        value: ValueId,
    },

    // -- Control flow --

    /// Unconditional branch.
    Br {
        target: BlockId,
        args: Vec<ValueId>,
    },

    /// Conditional branch.
    CondBr {
        cond: ValueId,
        then_target: BlockId,
        then_args: Vec<ValueId>,
        else_target: BlockId,
        else_args: Vec<ValueId>,
    },

    /// Multi-way branch (switch).
    Switch {
        value: ValueId,
        cases: Vec<SwitchCase>,
        default: BlockId,
    },

    /// Return from function.
    Return {
        values: Vec<ValueId>,
    },

    /// Direct function call.
    Call {
        func: FuncId,
        args: Vec<ValueId>,
        ret_ty: Vec<Ty>,
    },

    /// Indirect function call (call through pointer).
    CallIndirect {
        callee: ValueId,
        args: Vec<ValueId>,
        ret_ty: Vec<Ty>,
    },

    // -- Aggregate operations --

    /// Construct a struct value.
    Struct {
        ty: Ty,
        fields: Vec<ValueId>,
    },

    /// Extract a struct field: result = value.field[index].
    Field {
        ty: Ty,
        value: ValueId,
        index: u32,
    },

    /// Array/pointer index: result = base[index].
    Index {
        ty: Ty,
        base: ValueId,
        index: ValueId,
    },

    /// SSA phi node (block parameter).
    Phi {
        ty: Ty,
        incoming: Vec<(BlockId, ValueId)>,
    },

    /// Conditional value selection: result = cond ? true_val : false_val.
    ///
    /// Unlike CondBr (which is control flow), Select is a value-level operation
    /// that produces a result without branching. Lowered to CSEL on AArch64.
    Select {
        ty: Ty,
        cond: ValueId,
        true_val: ValueId,
        false_val: ValueId,
    },

    /// Get element pointer: typed pointer arithmetic with stride.
    ///
    /// Computes: base + index * sizeof(elem_ty) + byte_offset.
    /// This is the tMIR equivalent of LLVM's GEP instruction. It differs from
    /// Index (which is array access returning the element address) by supporting
    /// an explicit byte offset for struct field access within indexed elements.
    ///
    /// Example: accessing arr[i].field_at_offset_8
    ///   GetElementPtr { elem_ty: StructTy, base: arr_ptr, index: i, offset: 8 }
    GetElementPtr {
        /// The element type (determines stride = sizeof(elem_ty)).
        elem_ty: Ty,
        /// Base pointer.
        base: ValueId,
        /// Index (multiplied by element size).
        index: ValueId,
        /// Additional byte offset added after indexing.
        offset: i32,
    },

    /// No operation (used as placeholder).
    Nop,

    /// Integer constant.
    Const {
        ty: Ty,
        value: i64,
    },

    /// Float constant.
    FConst {
        ty: Ty,
        value: f64,
    },
}

/// A tMIR instruction with its result value(s) and proof annotations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstrNode {
    /// The instruction opcode and operands.
    pub instr: Instr,
    /// Result value(s) produced by this instruction. Empty for void ops
    /// (Store, Br, CondBr, Switch, Return, Nop, EndBorrow, Retain, Release, Dealloc).
    pub results: Vec<ValueId>,
    /// Proof annotations attached to this instruction by the source-language
    /// compiler (tRust, tSwift, tC). These have been formally verified by z4.
    /// The LLVM2 adapter extracts these for downstream optimization passes.
    #[serde(default)]
    pub proofs: Vec<TmirProof>,
}

impl InstrNode {
    /// Create a new instruction node without proof annotations.
    pub fn new(instr: Instr, results: Vec<ValueId>) -> Self {
        Self {
            instr,
            results,
            proofs: Vec::new(),
        }
    }

    /// Create a new instruction node with proof annotations.
    pub fn with_proofs(instr: Instr, results: Vec<ValueId>, proofs: Vec<TmirProof>) -> Self {
        Self {
            instr,
            results,
            proofs,
        }
    }
}
