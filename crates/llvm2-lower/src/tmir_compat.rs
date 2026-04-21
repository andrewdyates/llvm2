// llvm2-lower/tmir_compat.rs - Compatibility layer for tMIR API
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// This module provides re-exports and compatibility types that bridge the
// unified tmir crate API with LLVM2-internal types. The old 4-crate tMIR
// stubs (tmir-types, tmir-instrs, tmir-func, tmir-semantics) have been
// replaced by the real tmir crate. This module preserves backward-compatible
// types (CmpOp, Operand, CallingConv, etc.) that are LLVM2-internal and
// don't exist in the real tmir crate.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Direct re-exports from tmir (types that are identical or very similar)
// ---------------------------------------------------------------------------

// Core ID types (same API: .new(u32), .index(), .0 field)
pub use tmir::{BlockId, FuncId, StructId, ValueId};
pub use tmir::FuncTyId;
pub use tmir::TyId;

// Type system
pub use tmir::{Ty, FuncTy, StructDef, FieldDef};

// Instructions
pub use tmir::{BinOp, CastOp, UnOp, Inst, InstrNode, Constant};
pub use tmir::ICmpOp;
pub use tmir::FCmpOp;
pub use tmir::OverflowOp;
pub use tmir::SwitchCase;

// Atomics (renamed for compat)
pub use tmir::Ordering as MemoryOrdering;
pub use tmir::AtomicRMWOp as AtomicRmwOp;

// Proof system (renamed for compat with old TmirProof name)
pub use tmir::ProofAnnotation;
pub use tmir::{ProofSummary, ProofObligation, ProofStatus, ObligationKind};
pub use tmir::{ProofEvidence, ProofCertificate};
pub use tmir::{ProofId, ProofTag};

// Module/Function/Block structures
pub use tmir::{Module, Function, Block, Global};

// ---------------------------------------------------------------------------
// CmpOp compatibility enum
// ---------------------------------------------------------------------------

/// Unified comparison operation (LLVM2 extension).
///
/// The real tmir crate splits comparisons into `ICmpOp` (integer) and `FCmpOp`
/// (float). This unified enum is used by the LLVM2 ISel and adapter layers
/// where a single comparison type is more convenient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpOp {
    // Integer comparisons
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

impl CmpOp {
    /// Convert to an ICmpOp if this is an integer comparison.
    pub fn to_icmp(&self) -> Option<ICmpOp> {
        match self {
            CmpOp::Eq => Some(ICmpOp::Eq),
            CmpOp::Ne => Some(ICmpOp::Ne),
            CmpOp::Slt => Some(ICmpOp::Slt),
            CmpOp::Sle => Some(ICmpOp::Sle),
            CmpOp::Sgt => Some(ICmpOp::Sgt),
            CmpOp::Sge => Some(ICmpOp::Sge),
            CmpOp::Ult => Some(ICmpOp::Ult),
            CmpOp::Ule => Some(ICmpOp::Ule),
            CmpOp::Ugt => Some(ICmpOp::Ugt),
            CmpOp::Uge => Some(ICmpOp::Uge),
            _ => None,
        }
    }

    /// Convert to an FCmpOp if this is a float comparison.
    pub fn to_fcmp(&self) -> Option<FCmpOp> {
        match self {
            CmpOp::FOeq => Some(FCmpOp::OEq),
            CmpOp::FOne => Some(FCmpOp::ONe),
            CmpOp::FOlt => Some(FCmpOp::OLt),
            CmpOp::FOle => Some(FCmpOp::OLe),
            CmpOp::FOgt => Some(FCmpOp::OGt),
            CmpOp::FOge => Some(FCmpOp::OGe),
            CmpOp::FUeq => Some(FCmpOp::UEq),
            CmpOp::FUne => Some(FCmpOp::UNe),
            CmpOp::FUlt => Some(FCmpOp::ULt),
            CmpOp::FUle => Some(FCmpOp::ULe),
            CmpOp::FUgt => Some(FCmpOp::UGt),
            CmpOp::FUge => Some(FCmpOp::UGe),
            _ => None,
        }
    }

    /// Returns true if this is a float comparison.
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            CmpOp::FOeq
                | CmpOp::FOne
                | CmpOp::FOlt
                | CmpOp::FOle
                | CmpOp::FOgt
                | CmpOp::FOge
                | CmpOp::FUeq
                | CmpOp::FUne
                | CmpOp::FUlt
                | CmpOp::FUle
                | CmpOp::FUgt
                | CmpOp::FUge
        )
    }

    /// Create from an ICmpOp.
    pub fn from_icmp(op: ICmpOp) -> Self {
        match op {
            ICmpOp::Eq => CmpOp::Eq,
            ICmpOp::Ne => CmpOp::Ne,
            ICmpOp::Slt => CmpOp::Slt,
            ICmpOp::Sle => CmpOp::Sle,
            ICmpOp::Sgt => CmpOp::Sgt,
            ICmpOp::Sge => CmpOp::Sge,
            ICmpOp::Ult => CmpOp::Ult,
            ICmpOp::Ule => CmpOp::Ule,
            ICmpOp::Ugt => CmpOp::Ugt,
            ICmpOp::Uge => CmpOp::Uge,
        }
    }

    /// Create from an FCmpOp.
    pub fn from_fcmp(op: FCmpOp) -> Self {
        match op {
            FCmpOp::OEq => CmpOp::FOeq,
            FCmpOp::ONe => CmpOp::FOne,
            FCmpOp::OLt => CmpOp::FOlt,
            FCmpOp::OLe => CmpOp::FOle,
            FCmpOp::OGt => CmpOp::FOgt,
            FCmpOp::OGe => CmpOp::FOge,
            FCmpOp::UEq => CmpOp::FUeq,
            FCmpOp::UNe => CmpOp::FUne,
            FCmpOp::ULt => CmpOp::FUlt,
            FCmpOp::ULe => CmpOp::FUle,
            FCmpOp::UGt => CmpOp::FUgt,
            FCmpOp::UGe => CmpOp::FUge,
        }
    }
}

// ---------------------------------------------------------------------------
// Operand compatibility type
// ---------------------------------------------------------------------------

/// LLVM2-internal operand type.
///
/// The real tmir crate uses bare `ValueId` for instruction operands.
/// This `Operand` type (Value | Constant) is an LLVM2 extension used
/// by the adapter layer and test code.
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    /// Reference to an SSA value.
    Value(ValueId),
    /// Inline constant value.
    Constant(OperandConstant),
}

/// Constant value carried in an Operand (backward compat).
#[derive(Debug, Clone, PartialEq)]
pub enum OperandConstant {
    Int { value: i128, ty: Ty },
    Float { value: f64, ty: Ty },
    Bool(bool),
    Unit,
}

impl Operand {
    pub fn value(vid: ValueId) -> Self {
        Operand::Value(vid)
    }

    pub fn int(value: i128, ty: Ty) -> Self {
        Operand::Constant(OperandConstant::Int { value, ty })
    }

    pub fn float(value: f64, ty: Ty) -> Self {
        Operand::Constant(OperandConstant::Float { value, ty })
    }

    pub fn bool_const(value: bool) -> Self {
        Operand::Constant(OperandConstant::Bool(value))
    }

    pub fn is_constant(&self) -> bool {
        matches!(self, Operand::Constant(_))
    }

    pub fn is_value(&self) -> bool {
        matches!(self, Operand::Value(_))
    }

    pub fn as_value(&self) -> Option<ValueId> {
        match self {
            Operand::Value(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_constant(&self) -> Option<&OperandConstant> {
        match self {
            Operand::Constant(c) => Some(c),
            _ => None,
        }
    }
}

impl From<ValueId> for Operand {
    fn from(vid: ValueId) -> Self {
        Operand::Value(vid)
    }
}

// ---------------------------------------------------------------------------
// LLVM2-internal types not present in the tmir crate
// ---------------------------------------------------------------------------

/// Calling convention (LLVM2 extension, not in real tmir).
///
/// The real tmir crate does not include calling conventions. This is kept
/// as an LLVM2-internal type for the adapter/ABI layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CallingConv {
    #[default]
    C,
    Fast,
    Swift,
    Cold,
    PreserveMost,
}

/// Symbol visibility (LLVM2 extension, not in real tmir).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Visibility {
    #[default]
    Default,
    Hidden,
    Protected,
}

impl Visibility {
    pub fn is_default(&self) -> bool {
        matches!(self, Visibility::Default)
    }

    pub fn is_hidden(&self) -> bool {
        matches!(self, Visibility::Hidden)
    }

    pub fn macho_flags(&self) -> u8 {
        match self {
            Visibility::Default | Visibility::Protected => 0x01,
            Visibility::Hidden => 0x01 | 0x10,
        }
    }
}

/// Data layout (LLVM2 extension, not in real tmir).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataLayout {
    pub pointer_size: u32,
    pub pointer_align: u32,
    pub stack_align: u32,
    pub big_endian: bool,
    pub int_align: Vec<(u16, u32)>,
}

impl DataLayout {
    pub fn aarch64() -> Self {
        Self {
            pointer_size: 8,
            pointer_align: 8,
            stack_align: 16,
            big_endian: false,
            int_align: vec![(8, 1), (16, 2), (32, 4), (64, 8), (128, 16)],
        }
    }

    pub fn x86_64() -> Self {
        Self {
            pointer_size: 8,
            pointer_align: 8,
            stack_align: 16,
            big_endian: false,
            int_align: vec![(8, 1), (16, 2), (32, 4), (64, 8), (128, 16)],
        }
    }

    pub fn align_of_int(&self, bits: u16) -> u32 {
        self.int_align
            .iter()
            .find(|(b, _)| *b == bits)
            .map(|(_, a)| *a)
            .unwrap_or((bits as u32).div_ceil(8))
    }
}

/// Linkage type (LLVM2 extension, not in real tmir).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Linkage {
    #[default]
    External,
    Internal,
    Weak,
    AvailableExternally,
}

/// Global variable definition (LLVM2 extension wrapping tmir::Global).
///
/// The real tmir crate has a simpler `Global` struct. This extends it
/// with LLVM2-specific fields like linkage and visibility.
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalDef {
    pub name: String,
    pub ty: Ty,
    pub is_const: bool,
    pub linkage: Linkage,
    pub visibility: Visibility,
    pub initializer: Option<Vec<u8>>,
    pub align: Option<u32>,
}

/// Legacy SwitchCase from old stubs (value was i64, no args).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LegacySwitchCase {
    pub value: i64,
    pub target: BlockId,
}

/// Landing pad clause (LLVM2 extension, not in real tmir).
#[derive(Debug, Clone, PartialEq)]
pub enum LandingPadClause {
    Catch(Ty),
    Filter(Vec<Ty>),
}
