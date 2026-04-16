// tmir-instrs stub — development mirror of the real tMIR instruction API.
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// This file is a development stub used by LLVM2. It is intentionally kept
// outside the main workspace and mirrors the public instruction-set API of the
// real tMIR crate closely enough for adapter code and stub-only tests.
//
// As of 2026-04-16, the real tMIR crate is used by the main workspace via a git
// dependency. These stubs remain as documentation and for stub-only tests.
//
// Key API changes from the previous (pre-real-tMIR) stub:
//   - Inst (not Instr), InstrNode.inst (not .instr)
//   - ICmp/FCmp (not unified Cmp)
//   - Alloca (not Alloc), GEP (not GetElementPtr)
//   - AtomicRMW/AtomicRMWOp (capitalized), Ordering (not MemoryOrdering)
//   - All operands are ValueId (no Operand wrapper)
//   - Const value is Constant enum (not i64)
//   - New: Overflow, NullPtr, Undef, Assume, Assert, Unreachable, Copy,
//     ExtractField, InsertField, ExtractElement, InsertElement
//   - Removed: Phi, Nop, FConst, Struct, Field, Index, Borrow, BorrowMut,
//     EndBorrow, Retain, Release, IsUnique, Invoke, LandingPad, Resume, Dealloc

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use tmir_types::{BlockId, FuncId, FuncTyId, TmirProof, Ty, ValueId};

/// Binary arithmetic, bitwise, shift, and floating-point operations.
///
/// This development stub mirrors the operation names used by the real tMIR
/// instruction API so downstream code can pattern-match on the same surface.
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
    FRem,
}

/// Unary arithmetic and bitwise operations.
///
/// These are the unary operators exposed by the real tMIR instruction API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnOp {
    Neg,
    Not,
    FNeg,
}

/// Integer comparison predicates.
///
/// Signed and unsigned predicates are split explicitly because the real tMIR
/// API exposes integer comparisons this way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ICmpOp {
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
}

/// Floating-point comparison predicates.
///
/// Ordered predicates require non-NaN operands; unordered predicates treat NaN
/// as a valid match according to the usual LLVM-style semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FCmpOp {
    OEq,
    ONe,
    OLt,
    OLe,
    OGt,
    OGe,
    UEq,
    UNe,
    ULt,
    ULe,
    UGt,
    UGe,
}

/// Arithmetic operations that also report overflow information.
///
/// These variants model the overflow-producing operations exposed by the real
/// tMIR API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OverflowOp {
    AddOverflow,
    SubOverflow,
    MulOverflow,
}

/// Type conversion operations.
///
/// These names match the real tMIR cast instruction surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CastOp {
    ZExt,
    SExt,
    Trunc,
    FPToSI,
    FPToUI,
    SIToFP,
    UIToFP,
    FPExt,
    FPTrunc,
    PtrToInt,
    IntToPtr,
    Bitcast,
}

/// Atomic memory orderings.
///
/// This mirrors the ordering enum used by the real tMIR atomic instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ordering {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

/// Atomic read-modify-write operations.
///
/// These are the RMW operators currently surfaced by the real tMIR API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AtomicRMWOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Xchg,
}

/// Constant values embedded directly in instructions.
///
/// This type intentionally derives only `PartialEq` rather than `Eq` or `Hash`
/// because it can carry `f64` values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Constant {
    Int(i128),
    Float(f64),
    Bool(bool),
    Aggregate(Vec<Constant>),
}

/// A single explicit switch destination.
///
/// Each case matches a constant value and transfers control to a target block
/// with a block-argument payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwitchCase {
    pub value: Constant,
    pub target: BlockId,
    pub args: Vec<ValueId>,
}

/// Landing-pad clause metadata preserved as an LLVM2 extension.
///
/// The real tMIR instruction API mirrored by [`Inst`] does not currently expose
/// landing-pad instructions in this stub, but LLVM2 still keeps this type
/// around for exception-handling compatibility.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LandingPadClause {
    Catch(Ty),
    Filter(Vec<Ty>),
}

/// A single tMIR instruction.
///
/// This development stub mirrors the real tMIR instruction enum names and field
/// layout so adapter code can compile against a local stand-in crate.
///
/// All operands are bare `ValueId` references (no inline constant wrapper).
/// Constants are represented as separate `Inst::Const` instructions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Inst {
    /// Binary operation: `result = lhs op rhs`.
    BinOp {
        op: BinOp,
        ty: Ty,
        lhs: ValueId,
        rhs: ValueId,
    },

    /// Unary operation: `result = op operand`.
    UnOp {
        op: UnOp,
        ty: Ty,
        operand: ValueId,
    },

    /// Integer comparison producing a boolean result.
    ICmp {
        op: ICmpOp,
        ty: Ty,
        lhs: ValueId,
        rhs: ValueId,
    },

    /// Floating-point comparison producing a boolean result.
    FCmp {
        op: FCmpOp,
        ty: Ty,
        lhs: ValueId,
        rhs: ValueId,
    },

    /// Type conversion from `src_ty` to `dst_ty`.
    Cast {
        op: CastOp,
        src_ty: Ty,
        dst_ty: Ty,
        operand: ValueId,
    },

    /// Load from memory: `result = *ptr`.
    Load {
        ty: Ty,
        ptr: ValueId,
    },

    /// Store to memory: `*ptr = value`.
    Store {
        ty: Ty,
        ptr: ValueId,
        value: ValueId,
    },

    /// Stack allocation: `result = alloca(ty, count)`.
    Alloca {
        ty: Ty,
        count: Option<ValueId>,
    },

    /// Typed pointer arithmetic (get-element-pointer).
    GEP {
        pointee_ty: Ty,
        base: ValueId,
        indices: Vec<ValueId>,
    },

    /// Atomic load: `result = atomic_load(ptr, ordering)`.
    AtomicLoad {
        ty: Ty,
        ptr: ValueId,
        ordering: Ordering,
    },

    /// Atomic store: `atomic_store(ptr, value, ordering)`.
    AtomicStore {
        ty: Ty,
        ptr: ValueId,
        value: ValueId,
        ordering: Ordering,
    },

    /// Atomic read-modify-write: `result = atomic_rmw(op, ptr, val, ordering)`.
    AtomicRMW {
        op: AtomicRMWOp,
        ty: Ty,
        ptr: ValueId,
        value: ValueId,
        ordering: Ordering,
    },

    /// Atomic compare-and-exchange.
    ///
    /// If `*ptr == expected`, stores `desired` and returns `(expected, true)`.
    /// Otherwise returns `(old_value, false)`.
    CmpXchg {
        ty: Ty,
        ptr: ValueId,
        expected: ValueId,
        desired: ValueId,
        success: Ordering,
        failure: Ordering,
    },

    /// Atomic fence: enforces ordering constraint without accessing memory.
    Fence {
        ordering: Ordering,
    },

    /// Unconditional branch with block arguments.
    Br {
        target: BlockId,
        args: Vec<ValueId>,
    },

    /// Conditional branch with explicit block arguments for both edges.
    CondBr {
        cond: ValueId,
        then_target: BlockId,
        then_args: Vec<ValueId>,
        else_target: BlockId,
        else_args: Vec<ValueId>,
    },

    /// Multi-way branch with explicit default and per-case block arguments.
    Switch {
        value: ValueId,
        default: BlockId,
        default_args: Vec<ValueId>,
        cases: Vec<SwitchCase>,
    },

    /// Direct call by function identifier.
    Call {
        callee: FuncId,
        args: Vec<ValueId>,
    },

    /// Indirect call by SSA value plus explicit function signature ID.
    CallIndirect {
        callee: ValueId,
        sig: FuncTyId,
        args: Vec<ValueId>,
    },

    /// Function return.
    Return {
        values: Vec<ValueId>,
    },

    /// Extract a field from an aggregate value.
    ExtractField {
        ty: Ty,
        aggregate: ValueId,
        field: u32,
    },

    /// Insert a field into an aggregate value (returns new aggregate).
    InsertField {
        ty: Ty,
        aggregate: ValueId,
        field: u32,
        value: ValueId,
    },

    /// Extract an element from an array-like aggregate.
    ExtractElement {
        ty: Ty,
        array: ValueId,
        index: ValueId,
    },

    /// Insert an element into an array-like aggregate.
    InsertElement {
        ty: Ty,
        array: ValueId,
        index: ValueId,
        value: ValueId,
    },

    /// Arithmetic operation that also reports overflow.
    ///
    /// Produces two results: `(result, overflow_flag)`.
    Overflow {
        op: OverflowOp,
        ty: Ty,
        lhs: ValueId,
        rhs: ValueId,
    },

    /// Branchless value selection: `result = cond ? then_val : else_val`.
    Select {
        ty: Ty,
        cond: ValueId,
        then_val: ValueId,
        else_val: ValueId,
    },

    /// Inline constant materialization.
    Const {
        ty: Ty,
        value: Constant,
    },

    /// Null pointer constant.
    NullPtr,

    /// Undefined value of a specific type.
    Undef {
        ty: Ty,
    },

    /// Optimizer assumption (hint to the verifier).
    Assume {
        cond: ValueId,
    },

    /// Checked assertion with a diagnostic message.
    Assert {
        cond: ValueId,
        msg: String,
    },

    /// Instruction that cannot be reached in well-formed control flow.
    Unreachable,

    /// Explicit SSA copy (identity operation).
    Copy {
        ty: Ty,
        operand: ValueId,
    },
}

/// A tMIR instruction node carrying results and proof annotations.
///
/// The real tMIR API uses an instruction node wrapper; this development stub
/// keeps that shape and stores proof metadata needed by LLVM2-side code.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstrNode {
    /// The instruction opcode and operands.
    pub inst: Inst,
    /// SSA values defined by this instruction. Empty for void ops
    /// (Store, Br, CondBr, Switch, Return, Fence, Assume, Assert, Unreachable).
    pub results: Vec<ValueId>,
    /// Optional proof annotations attached by upstream tooling (tRust, tSwift, tC).
    /// These have been formally verified by z4.
    #[serde(default)]
    pub proofs: Vec<TmirProof>,
}

impl InstrNode {
    /// Creates a new instruction node with no proof annotations.
    pub fn new(inst: Inst, results: Vec<ValueId>) -> Self {
        Self {
            inst,
            results,
            proofs: Vec::new(),
        }
    }

    /// Creates a new instruction node with explicit proof annotations.
    pub fn with_proofs(inst: Inst, results: Vec<ValueId>, proofs: Vec<TmirProof>) -> Self {
        Self {
            inst,
            results,
            proofs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip_instr(node: &InstrNode) -> InstrNode {
        let json = serde_json::to_string(node).expect("serialize instruction node");
        serde_json::from_str(&json).expect("deserialize instruction node")
    }

    #[test]
    fn test_binop_serde_round_trip() {
        let instr = InstrNode::new(
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::i32(),
                lhs: ValueId(1),
                rhs: ValueId(2),
            },
            vec![ValueId(3)],
        );
        let round_tripped = round_trip_instr(&instr);
        assert_eq!(instr, round_tripped);
    }

    #[test]
    fn test_const_i128_serde_round_trip() {
        let instr = InstrNode::new(
            Inst::Const {
                ty: Ty::i128(),
                value: Constant::Int(1_i128 << 100),
            },
            vec![ValueId(9)],
        );
        let round_tripped = round_trip_instr(&instr);
        assert_eq!(instr, round_tripped);
    }

    #[test]
    fn test_switch_serde_round_trip() {
        let instr = InstrNode::new(
            Inst::Switch {
                value: ValueId(0),
                default: BlockId(99),
                default_args: vec![ValueId(10)],
                cases: vec![
                    SwitchCase {
                        value: Constant::Int(0),
                        target: BlockId(1),
                        args: vec![ValueId(11)],
                    },
                    SwitchCase {
                        value: Constant::Int(1),
                        target: BlockId(2),
                        args: vec![ValueId(12), ValueId(13)],
                    },
                ],
            },
            vec![],
        );
        let round_tripped = round_trip_instr(&instr);
        assert_eq!(instr, round_tripped);
    }

    #[test]
    fn test_icmp_serde_round_trip() {
        let instr = InstrNode::new(
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::i64(),
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
            vec![ValueId(2)],
        );
        let round_tripped = round_trip_instr(&instr);
        assert_eq!(instr, round_tripped);
    }

    #[test]
    fn test_fcmp_serde_round_trip() {
        let instr = InstrNode::new(
            Inst::FCmp {
                op: FCmpOp::OLt,
                ty: Ty::f64(),
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
            vec![ValueId(2)],
        );
        let round_tripped = round_trip_instr(&instr);
        assert_eq!(instr, round_tripped);
    }

    #[test]
    fn test_overflow_serde_round_trip() {
        let instr = InstrNode::new(
            Inst::Overflow {
                op: OverflowOp::AddOverflow,
                ty: Ty::i32(),
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
            vec![ValueId(2), ValueId(3)],
        );
        let round_tripped = round_trip_instr(&instr);
        assert_eq!(instr, round_tripped);
    }

    #[test]
    fn test_select_serde_round_trip() {
        let instr = InstrNode::new(
            Inst::Select {
                ty: Ty::i32(),
                cond: ValueId(0),
                then_val: ValueId(1),
                else_val: ValueId(2),
            },
            vec![ValueId(3)],
        );
        let round_tripped = round_trip_instr(&instr);
        assert_eq!(instr, round_tripped);
    }

    #[test]
    fn test_gep_serde_round_trip() {
        let instr = InstrNode::new(
            Inst::GEP {
                pointee_ty: Ty::i32(),
                base: ValueId(0),
                indices: vec![ValueId(1), ValueId(2)],
            },
            vec![ValueId(3)],
        );
        let round_tripped = round_trip_instr(&instr);
        assert_eq!(instr, round_tripped);
    }

    #[test]
    fn test_null_ptr_serde_round_trip() {
        let instr = InstrNode::new(Inst::NullPtr, vec![ValueId(0)]);
        let round_tripped = round_trip_instr(&instr);
        assert_eq!(instr, round_tripped);
    }

    #[test]
    fn test_unreachable_serde_round_trip() {
        let instr = InstrNode::new(Inst::Unreachable, vec![]);
        let round_tripped = round_trip_instr(&instr);
        assert_eq!(instr, round_tripped);
    }

    #[test]
    fn test_copy_serde_round_trip() {
        let instr = InstrNode::new(
            Inst::Copy {
                ty: Ty::i64(),
                operand: ValueId(5),
            },
            vec![ValueId(6)],
        );
        let round_tripped = round_trip_instr(&instr);
        assert_eq!(instr, round_tripped);
    }

    #[test]
    fn test_landing_pad_clause_serde() {
        let catch = LandingPadClause::Catch(Ty::i32());
        let json = serde_json::to_string(&catch).unwrap();
        let rt: LandingPadClause = serde_json::from_str(&json).unwrap();
        assert_eq!(catch, rt);

        let filter = LandingPadClause::Filter(vec![Ty::i32(), Ty::i64()]);
        let json = serde_json::to_string(&filter).unwrap();
        let rt: LandingPadClause = serde_json::from_str(&json).unwrap();
        assert_eq!(filter, rt);
    }

    #[test]
    fn test_with_proofs() {
        let instr = InstrNode::with_proofs(
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::i32(),
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
            vec![ValueId(2)],
            vec![TmirProof::NoOverflow { signed: true }],
        );
        assert_eq!(instr.proofs.len(), 1);
        let round_tripped = round_trip_instr(&instr);
        assert_eq!(instr, round_tripped);
    }
}
