// tmir-instrs stub — minimal tMIR instruction set for LLVM2 development
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// This is a development stub. The real tMIR crate (ayates_dbx/tMIR) defines
// the full instruction set. This stub provides the ~25 core instructions
// needed by LLVM2's instruction selector.
//
// Operand model: The real tMIR uses Operand (Value | Constant) for instruction
// inputs, matching how SSA IRs carry inline constants. Our stub mirrors this
// model so the adapter layer can handle both value references and inline
// constants uniformly. See ~/tMIR/crates/tmir-instrs/src/lib.rs for reference.
//
// Key differences from real tMIR that are intentional LLVM2 extensions:
//   - InstrNode wrapper: carries proof annotations (real tMIR has no proofs)
//   - Atomic instructions: AtomicLoad/Store/Rmw/CmpXchg/Fence
//   - Select instruction: branchless conditional (real tMIR uses control flow)
//   - GetElementPtr: typed pointer arithmetic (real tMIR uses Index)
//   - Const/FConst instructions: kept for backward compat, prefer Operand::Constant
//   - Signed/unsigned op variants: SDiv/UDiv, Slt/Ult, etc. (real tMIR is simpler)

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use tmir_types::{BlockId, FuncId, TmirProof, Ty, ValueId};

/// Binary arithmetic/logic operations.
///
/// Note: Real tMIR uses simpler variants (Div instead of SDiv/UDiv, Shr instead
/// of AShr/LShr). Our stub preserves the signed/unsigned distinction because the
/// adapter and ISel need it for correct AArch64 lowering. The adapter maps
/// real tMIR's Div/Rem/Shr based on the operand type's signedness.
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
///
/// Note: Real tMIR has only Neg and Not. FNeg/FAbs/FSqrt are LLVM2 extensions
/// for float-specific operations that the ISel needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnOp {
    Neg,
    Not,
    FNeg,
    FAbs,
    FSqrt,
}

/// Integer/float comparison predicates.
///
/// Note: Real tMIR has simpler predicates (Lt/Le/Gt/Ge without signed/unsigned
/// distinction, no float comparisons). The adapter maps based on operand type.
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
///
/// Note: Real tMIR has simpler cast ops (IntToFloat/FloatToInt without
/// signed/unsigned distinction). The adapter maps based on source type.
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

// ---------------------------------------------------------------------------
// Operand model (aligned with real tMIR)
// ---------------------------------------------------------------------------

/// A constant value (inline in an operand, not a separate instruction).
///
/// In the real tMIR, constants are carried inline within Operand::Constant
/// rather than as separate Const/FConst instructions. This stub mirrors that
/// model. The adapter layer resolves Operand::Constant to LIR Iconst/Fconst
/// instructions during lowering.
///
/// Reference: ~/tMIR/crates/tmir-instrs/src/lib.rs (Constant enum)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Constant {
    /// Integer constant with type. Uses i128 to match real tMIR (not i64).
    Int { value: i128, ty: Ty },
    /// Float constant with type.
    Float { value: f64, ty: Ty },
    /// Boolean constant.
    Bool(bool),
    /// Unit constant (void/zero-sized).
    Unit,
}

/// An operand: either an SSA value reference or an inline constant.
///
/// This is the fundamental operand model of real tMIR. Instructions reference
/// their inputs as Operand rather than bare ValueId, allowing constants to be
/// carried inline without separate Const instructions. This enables simpler
/// pattern matching in the instruction selector.
///
/// Reference: ~/tMIR/crates/tmir-instrs/src/lib.rs (Operand enum)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Operand {
    /// Reference to an SSA value defined by another instruction.
    Value(ValueId),
    /// Inline constant value.
    Constant(Constant),
}

impl Operand {
    /// Create an operand referencing an SSA value.
    pub fn value(vid: ValueId) -> Self {
        Operand::Value(vid)
    }

    /// Create an integer constant operand.
    pub fn int(value: i128, ty: Ty) -> Self {
        Operand::Constant(Constant::Int { value, ty })
    }

    /// Create a float constant operand.
    pub fn float(value: f64, ty: Ty) -> Self {
        Operand::Constant(Constant::Float { value, ty })
    }

    /// Create a boolean constant operand.
    pub fn bool_const(value: bool) -> Self {
        Operand::Constant(Constant::Bool(value))
    }

    /// Returns true if this operand is a constant.
    pub fn is_constant(&self) -> bool {
        matches!(self, Operand::Constant(_))
    }

    /// Returns true if this operand is a value reference.
    pub fn is_value(&self) -> bool {
        matches!(self, Operand::Value(_))
    }

    /// Get the ValueId if this is a value reference.
    pub fn as_value(&self) -> Option<ValueId> {
        match self {
            Operand::Value(v) => Some(*v),
            _ => None,
        }
    }

    /// Get the constant if this is a constant operand.
    pub fn as_constant(&self) -> Option<&Constant> {
        match self {
            Operand::Constant(c) => Some(c),
            _ => None,
        }
    }
}

/// Convenience: convert a ValueId directly into an Operand::Value.
impl From<ValueId> for Operand {
    fn from(vid: ValueId) -> Self {
        Operand::Value(vid)
    }
}

/// A single switch case: value -> target block.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SwitchCase {
    pub value: i64,
    pub target: BlockId,
}

/// A clause in a landing pad instruction.
///
/// Catch clauses match a specific exception type. Filter clauses specify a set
/// of types that may be thrown; if the thrown type is not in the filter list,
/// the filter matches (triggering unexpected-exception handling).
///
/// This mirrors the LLVM landingpad clause model used by C++ EH and Rust panics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LandingPadClause {
    /// Catch an exception of the given type.
    Catch(Ty),
    /// Filter: matches if the thrown type is NOT in the list (for `throw()` specs).
    Filter(Vec<Ty>),
}

/// tMIR instruction set.
///
/// These are the ~25 core instructions that LLVM2 must lower to machine code.
/// Each instruction operates on SSA values and produces zero or more results.
///
/// ## Operand model
///
/// Instructions that accept general inputs use `Operand` (value or constant),
/// matching the real tMIR operand model. Instructions that require a specific
/// SSA value (e.g., pointer for Load, borrow target) use bare `ValueId`.
///
/// The adapter layer's `resolve_operand` method materializes constants into
/// LIR Iconst/Fconst instructions when an `Operand::Constant` is encountered.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Instr {
    /// Binary operation: result = lhs op rhs.
    /// Both operands may be values or inline constants.
    BinOp {
        op: BinOp,
        ty: Ty,
        lhs: Operand,
        rhs: Operand,
    },

    /// Unary operation: result = op operand.
    UnOp {
        op: UnOp,
        ty: Ty,
        operand: Operand,
    },

    /// Comparison: result (Bool) = lhs cmp rhs.
    /// Both operands may be values or inline constants.
    Cmp {
        op: CmpOp,
        ty: Ty,
        lhs: Operand,
        rhs: Operand,
    },

    /// Type cast: result = cast operand from src_ty to dst_ty.
    Cast {
        op: CastOp,
        src_ty: Ty,
        dst_ty: Ty,
        operand: Operand,
    },

    /// Load from memory: result = *ptr.
    /// ptr must be a value (cannot load from a constant address).
    Load {
        ty: Ty,
        ptr: ValueId,
    },

    /// Store to memory: *ptr = value.
    /// ptr must be a value; stored value may be a constant.
    Store {
        ty: Ty,
        ptr: ValueId,
        value: Operand,
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

    // -- Atomic memory operations (LLVM2 extension, not in real tMIR yet) --

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
        value: Operand,
        ordering: MemoryOrdering,
    },

    /// Atomic read-modify-write: result (old value) = atomic_rmw(op, ptr, val, ordering).
    AtomicRmw {
        op: AtomicRmwOp,
        ty: Ty,
        ptr: ValueId,
        value: Operand,
        ordering: MemoryOrdering,
    },

    /// Compare and swap: (old_value, success) = cmpxchg(ptr, expected, desired, ordering).
    /// If *ptr == expected, stores desired and returns (expected, true).
    /// Otherwise returns (old_value, false).
    CmpXchg {
        ty: Ty,
        ptr: ValueId,
        expected: Operand,
        desired: Operand,
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
    /// Branch args may be values or constants (matching real tMIR).
    Br {
        target: BlockId,
        args: Vec<Operand>,
    },

    /// Conditional branch.
    /// Condition and branch args may be values or constants.
    CondBr {
        cond: Operand,
        then_target: BlockId,
        then_args: Vec<Operand>,
        else_target: BlockId,
        else_args: Vec<Operand>,
    },

    /// Multi-way branch (switch).
    /// The switched value may be a value or constant.
    Switch {
        value: Operand,
        cases: Vec<SwitchCase>,
        default: BlockId,
    },

    /// Return from function.
    /// Return values may be values or constants (matching real tMIR).
    Return {
        values: Vec<Operand>,
    },

    // -- Exception handling --

    /// Invoke a function that may throw an exception.
    ///
    /// Like Call, but with normal and unwind successor blocks.
    /// If the callee returns normally, control transfers to normal_dest with normal_args.
    /// If the callee throws, control transfers to unwind_dest with unwind_args.
    /// Arguments may be values or constants (matching real tMIR).
    Invoke {
        func: FuncId,
        args: Vec<Operand>,
        ret_ty: Vec<Ty>,
        normal_dest: BlockId,
        normal_args: Vec<Operand>,
        unwind_dest: BlockId,
        unwind_args: Vec<Operand>,
    },

    /// Exception landing pad.
    ///
    /// This is the first instruction in an unwind destination block.
    /// It specifies the personality function's catch/filter clauses.
    /// The result is the caught exception value.
    LandingPad {
        ty: Ty,
        clauses: Vec<LandingPadClause>,
    },

    /// Resume unwinding after a landing pad.
    ///
    /// Continues propagating the exception to the next handler up the call stack.
    /// The value operand is the exception value from the landing pad.
    Resume {
        value: ValueId,
    },

    /// Direct function call.
    /// Arguments may be values or constants (matching real tMIR).
    Call {
        func: FuncId,
        args: Vec<Operand>,
        ret_ty: Vec<Ty>,
    },

    /// Indirect function call (call through pointer).
    /// callee must be a value; arguments may be values or constants.
    CallIndirect {
        callee: ValueId,
        args: Vec<Operand>,
        ret_ty: Vec<Ty>,
    },

    // -- Aggregate operations --

    /// Construct a struct value.
    /// Fields may be values or constants.
    Struct {
        ty: Ty,
        fields: Vec<Operand>,
    },

    /// Extract a struct field: result = value.field[index].
    /// The value must be an SSA value (cannot extract from a constant).
    Field {
        ty: Ty,
        value: ValueId,
        index: u32,
    },

    /// Array/pointer index: result = base[index].
    /// Index may be a value or constant.
    Index {
        ty: Ty,
        base: ValueId,
        index: Operand,
    },

    /// SSA phi node (block parameter).
    /// Incoming values may be values or constants (matching real tMIR).
    Phi {
        ty: Ty,
        incoming: Vec<(BlockId, Operand)>,
    },

    /// Conditional value selection: result = cond ? true_val : false_val.
    ///
    /// Unlike CondBr (which is control flow), Select is a value-level operation
    /// that produces a result without branching. Lowered to CSEL on AArch64.
    /// LLVM2 extension (real tMIR uses control flow for this).
    Select {
        ty: Ty,
        cond: Operand,
        true_val: Operand,
        false_val: Operand,
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
        /// Index (multiplied by element size). May be value or constant.
        index: Operand,
        /// Additional byte offset added after indexing.
        offset: i32,
    },

    /// No operation (used as placeholder).
    Nop,

    /// Integer constant (legacy — prefer Operand::Constant for new code).
    ///
    /// Kept for backward compatibility. The adapter handles both this form and
    /// Operand::Constant(Constant::Int{..}).
    Const {
        ty: Ty,
        value: i64,
    },

    /// Float constant (legacy — prefer Operand::Constant for new code).
    ///
    /// Kept for backward compatibility. The adapter handles both this form and
    /// Operand::Constant(Constant::Float{..}).
    FConst {
        ty: Ty,
        value: f64,
    },
}

/// A tMIR instruction with its result value(s) and proof annotations.
///
/// This is an LLVM2 extension over real tMIR. In real tMIR, result values are
/// fields on each instruction variant (e.g., `BinOp { result: Value, ... }`).
/// Our InstrNode wrapper separates results from the instruction for uniform
/// handling in the adapter and ISel. Proof annotations are another LLVM2
/// extension (real tMIR does not yet carry per-instruction proofs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstrNode {
    /// The instruction opcode and operands.
    pub instr: Instr,
    /// Result value(s) produced by this instruction. Empty for void ops
    /// (Store, Br, CondBr, Switch, Return, Resume, Nop, EndBorrow, Retain, Release, Dealloc).
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: JSON round-trip an InstrNode and verify equality.
    fn round_trip_instr(node: &InstrNode) -> InstrNode {
        let json = serde_json::to_string(node).expect("serialize");
        serde_json::from_str(&json).expect("deserialize")
    }

    #[test]
    fn test_invoke_serde_round_trip() {
        let instr = InstrNode::new(
            Instr::Invoke {
                func: FuncId(1),
                args: vec![Operand::Value(ValueId(0)), Operand::int(42, Ty::i32())],
                ret_ty: vec![Ty::i32()],
                normal_dest: BlockId(1),
                normal_args: vec![],
                unwind_dest: BlockId(2),
                unwind_args: vec![],
            },
            vec![ValueId(10)],
        );
        let rt = round_trip_instr(&instr);
        assert_eq!(instr, rt);
    }

    #[test]
    fn test_landing_pad_serde_round_trip() {
        let instr = InstrNode::new(
            Instr::LandingPad {
                ty: Ty::i64(),
                clauses: vec![
                    LandingPadClause::Catch(Ty::i32()),
                    LandingPadClause::Filter(vec![Ty::i32(), Ty::i64()]),
                ],
            },
            vec![ValueId(5)],
        );
        let rt = round_trip_instr(&instr);
        assert_eq!(instr, rt);
    }

    #[test]
    fn test_landing_pad_empty_clauses() {
        let instr = InstrNode::new(
            Instr::LandingPad {
                ty: Ty::i64(),
                clauses: vec![],
            },
            vec![ValueId(0)],
        );
        let rt = round_trip_instr(&instr);
        assert_eq!(instr, rt);
    }

    #[test]
    fn test_resume_serde_round_trip() {
        let instr = InstrNode::new(
            Instr::Resume {
                value: ValueId(5),
            },
            vec![],
        );
        let rt = round_trip_instr(&instr);
        assert_eq!(instr, rt);
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
    fn test_invoke_no_results() {
        // Invoke for a void function
        let instr = InstrNode::new(
            Instr::Invoke {
                func: FuncId(0),
                args: vec![],
                ret_ty: vec![],
                normal_dest: BlockId(1),
                normal_args: vec![Operand::Value(ValueId(0))],
                unwind_dest: BlockId(2),
                unwind_args: vec![Operand::Value(ValueId(1))],
            },
            vec![],
        );
        let rt = round_trip_instr(&instr);
        assert_eq!(instr, rt);
    }

    #[test]
    fn test_invoke_multiple_return_types() {
        let instr = InstrNode::new(
            Instr::Invoke {
                func: FuncId(3),
                args: vec![Operand::Value(ValueId(0))],
                ret_ty: vec![Ty::i32(), Ty::i64()],
                normal_dest: BlockId(10),
                normal_args: vec![],
                unwind_dest: BlockId(20),
                unwind_args: vec![],
            },
            vec![ValueId(1), ValueId(2)],
        );
        let rt = round_trip_instr(&instr);
        assert_eq!(instr, rt);
    }
}
