// tmir-types stub — minimal tMIR type definitions for LLVM2 development
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// This is a development stub. The real tMIR crate (ayates_dbx/tMIR) defines
// the full type system. This stub provides only the subset needed by LLVM2.
//
// Aligned with real tMIR's nested enum representation as of 2026-04-16.
// See ~/tMIR/crates/tmir-types/src/lib.rs for the canonical definitions.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Sub-enums matching real tMIR nested type representation
// ---------------------------------------------------------------------------

/// Integer bit width (matches real tMIR IntWidth).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntWidth {
    I8,
    I16,
    I32,
    I64,
    I128,
}

impl IntWidth {
    pub fn bits(&self) -> u16 {
        match self {
            IntWidth::I8 => 8,
            IntWidth::I16 => 16,
            IntWidth::I32 => 32,
            IntWidth::I64 => 64,
            IntWidth::I128 => 128,
        }
    }

    /// Convert a raw bit width to IntWidth. Returns None for unsupported widths.
    pub fn from_bits(bits: u16) -> Option<Self> {
        match bits {
            8 => Some(IntWidth::I8),
            16 => Some(IntWidth::I16),
            32 => Some(IntWidth::I32),
            64 => Some(IntWidth::I64),
            128 => Some(IntWidth::I128),
            _ => None,
        }
    }
}

/// Float bit width (matches real tMIR FloatWidth).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FloatWidth {
    F32,
    F64,
}

impl FloatWidth {
    pub fn bits(&self) -> u16 {
        match self {
            FloatWidth::F32 => 32,
            FloatWidth::F64 => 64,
        }
    }

    /// Convert a raw bit width to FloatWidth. Returns None for unsupported widths.
    pub fn from_bits(bits: u16) -> Option<Self> {
        match bits {
            32 => Some(FloatWidth::F32),
            64 => Some(FloatWidth::F64),
            _ => None,
        }
    }
}

/// Primitive types (matches real tMIR PrimitiveType).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimitiveType {
    /// Integer type with width and signedness.
    Int { width: IntWidth, signed: bool },
    /// Floating point type with width.
    Float(FloatWidth),
    /// Boolean.
    Bool,
    /// Unit type (zero-sized, replaces old Void).
    Unit,
    /// Never type (diverging, no values).
    Never,
}

/// Reference mutability (matches real tMIR Mutability).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Mutability {
    Immutable,
    Mutable,
}

/// Reference kind (matches real tMIR RefKind).
///
/// Distinguishes Rust borrows, raw pointers, and ARC references.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RefKind {
    /// Rust borrow (&T or &mut T).
    Borrow(Mutability),
    /// Raw pointer (*const T or *mut T).
    Raw(Mutability),
    /// Reference counted (Swift ARC).
    Rc,
}

/// A field in a struct (matches real tMIR Field).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub ty: Ty,
}

/// A variant in an enum (matches real tMIR Variant).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Variant {
    pub name: String,
    pub fields: Vec<Ty>,
}

// ---------------------------------------------------------------------------
// Main type enum
// ---------------------------------------------------------------------------

/// A tMIR type.
///
/// tMIR uses a nested enum type system suitable for verified compilation.
/// Every value in tMIR has exactly one type, and types carry enough
/// information for the backend to select correct machine instructions.
///
/// This matches the real tMIR type representation (nested enums) rather than
/// the original flat enum. The adapter layer in llvm2-lower bridges this
/// representation to the internal LIR type system.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ty {
    /// Primitive type (integers, floats, bool, unit, never).
    Primitive(PrimitiveType),

    /// Fixed-size array: `[T; N]`.
    Array {
        element: Box<Ty>,
        len: u64,
    },

    /// Tuple type (T1, T2, ...).
    Tuple(Vec<Ty>),

    /// Struct type (inline definition with named fields).
    ///
    /// Matches real tMIR's `Type::Struct { name, fields }`.
    StructDef {
        name: String,
        fields: Vec<Field>,
    },

    /// Enum type with named variants.
    ///
    /// Matches real tMIR's `Type::Enum { name, variants }`.
    Enum {
        name: String,
        variants: Vec<Variant>,
    },

    /// Reference type (borrows, raw pointers, ARC).
    ///
    /// Replaces the old `Ptr(Box<Ty>)` with the richer `RefKind` from real tMIR.
    Ref {
        kind: RefKind,
        pointee: Box<Ty>,
    },

    /// Function pointer type.
    ///
    /// Matches real tMIR's `Type::FnPtr { params, ret }`.
    FnPtr {
        params: Vec<Ty>,
        ret: Box<Ty>,
    },

    // -- LLVM2-specific extensions below (not in real tMIR) --

    /// Named struct type reference (index into a struct definition table).
    ///
    /// LLVM2 extension: the real tMIR uses inline `Struct { name, fields }`
    /// instead of ID-based indirection. Kept for backward compatibility with
    /// the adapter's StructDef resolution path.
    Struct(StructId),

    /// Function type with multiple returns (LLVM2 extension).
    ///
    /// Kept for backward compatibility with FuncTy-based signatures.
    /// The real tMIR uses `FnPtr { params, ret }` with single return.
    Func(FuncTy),
}

impl Ty {
    // -- Convenience constructors matching real tMIR --

    pub fn i8() -> Self { Ty::Primitive(PrimitiveType::Int { width: IntWidth::I8, signed: true }) }
    pub fn i16() -> Self { Ty::Primitive(PrimitiveType::Int { width: IntWidth::I16, signed: true }) }
    pub fn i32() -> Self { Ty::Primitive(PrimitiveType::Int { width: IntWidth::I32, signed: true }) }
    pub fn i64() -> Self { Ty::Primitive(PrimitiveType::Int { width: IntWidth::I64, signed: true }) }
    pub fn i128() -> Self { Ty::Primitive(PrimitiveType::Int { width: IntWidth::I128, signed: true }) }
    pub fn u8() -> Self { Ty::Primitive(PrimitiveType::Int { width: IntWidth::I8, signed: false }) }
    pub fn u16() -> Self { Ty::Primitive(PrimitiveType::Int { width: IntWidth::I16, signed: false }) }
    pub fn u32() -> Self { Ty::Primitive(PrimitiveType::Int { width: IntWidth::I32, signed: false }) }
    pub fn u64() -> Self { Ty::Primitive(PrimitiveType::Int { width: IntWidth::I64, signed: false }) }
    pub fn f32() -> Self { Ty::Primitive(PrimitiveType::Float(FloatWidth::F32)) }
    pub fn f64() -> Self { Ty::Primitive(PrimitiveType::Float(FloatWidth::F64)) }

    pub fn bool_ty() -> Self { Ty::Primitive(PrimitiveType::Bool) }
    pub fn unit() -> Self { Ty::Primitive(PrimitiveType::Unit) }
    pub fn never() -> Self { Ty::Primitive(PrimitiveType::Never) }

    /// Construct a raw pointer type (backward-compat for old `Ty::Ptr(inner)`).
    pub fn ptr(inner: Ty) -> Self {
        Ty::Ref { kind: RefKind::Raw(Mutability::Immutable), pointee: Box::new(inner) }
    }

    /// Construct a mutable raw pointer type.
    pub fn ptr_mut(inner: Ty) -> Self {
        Ty::Ref { kind: RefKind::Raw(Mutability::Mutable), pointee: Box::new(inner) }
    }

    /// Construct an immutable borrow type.
    pub fn borrow(pointee: Ty) -> Self {
        Ty::Ref { kind: RefKind::Borrow(Mutability::Immutable), pointee: Box::new(pointee) }
    }

    /// Construct a mutable borrow type.
    pub fn borrow_mut(pointee: Ty) -> Self {
        Ty::Ref { kind: RefKind::Borrow(Mutability::Mutable), pointee: Box::new(pointee) }
    }

    /// Construct an ARC reference type.
    pub fn rc(pointee: Ty) -> Self {
        Ty::Ref { kind: RefKind::Rc, pointee: Box::new(pointee) }
    }

    /// Construct an array type.
    pub fn array(element: Ty, len: u64) -> Self {
        Ty::Array { element: Box::new(element), len }
    }

    // -- Backward-compat constructors for migration from flat representation --

    /// Construct a signed integer type from raw bit width.
    /// Panics if bit width is not one of 8, 16, 32, 64, 128.
    pub fn int(bits: u16) -> Self {
        let width = IntWidth::from_bits(bits)
            .unwrap_or_else(|| panic!("unsupported integer bit width: {bits}"));
        Ty::Primitive(PrimitiveType::Int { width, signed: true })
    }

    /// Construct an unsigned integer type from raw bit width.
    /// Panics if bit width is not one of 8, 16, 32, 64, 128.
    pub fn uint(bits: u16) -> Self {
        let width = IntWidth::from_bits(bits)
            .unwrap_or_else(|| panic!("unsupported integer bit width: {bits}"));
        Ty::Primitive(PrimitiveType::Int { width, signed: false })
    }

    /// Construct a float type from raw bit width.
    /// Panics if bit width is not 32 or 64.
    pub fn float(bits: u16) -> Self {
        let width = FloatWidth::from_bits(bits)
            .unwrap_or_else(|| panic!("unsupported float bit width: {bits}"));
        Ty::Primitive(PrimitiveType::Float(width))
    }

    /// Backward-compat alias for `unit()` (replaces old `Ty::Void`).
    pub fn void() -> Self { Ty::unit() }

    // -- Query methods --

    /// Size in bytes for scalar types. Returns None for aggregates.
    pub fn scalar_bytes(&self) -> Option<u32> {
        match self {
            Ty::Primitive(PrimitiveType::Bool) => Some(1),
            Ty::Primitive(PrimitiveType::Int { width, .. }) => {
                Some((width.bits() as u32).div_ceil(8))
            }
            Ty::Primitive(PrimitiveType::Float(fw)) => {
                Some((fw.bits() as u32).div_ceil(8))
            }
            Ty::Primitive(PrimitiveType::Unit) => Some(0),
            Ty::Primitive(PrimitiveType::Never) => Some(0),
            Ty::Ref { .. } => Some(8), // 64-bit pointers
            _ => None,
        }
    }

    /// Bit-width for scalar types.
    pub fn bits(&self) -> Option<u16> {
        match self {
            Ty::Primitive(PrimitiveType::Bool) => Some(1),
            Ty::Primitive(PrimitiveType::Int { width, .. }) => Some(width.bits()),
            Ty::Primitive(PrimitiveType::Float(fw)) => Some(fw.bits()),
            Ty::Primitive(PrimitiveType::Unit) => Some(0),
            Ty::Primitive(PrimitiveType::Never) => Some(0),
            Ty::Ref { .. } => Some(64),
            _ => None,
        }
    }

    /// Returns true if this is a floating-point type.
    pub fn is_float(&self) -> bool {
        matches!(self, Ty::Primitive(PrimitiveType::Float(_)))
    }

    /// Returns true if this is an integer or boolean type.
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            Ty::Primitive(PrimitiveType::Bool)
                | Ty::Primitive(PrimitiveType::Int { .. })
        )
    }

    /// Returns true if this is a pointer/reference type.
    pub fn is_pointer(&self) -> bool {
        matches!(self, Ty::Ref { .. })
    }

    /// Returns true if this is the unit type (void).
    pub fn is_unit(&self) -> bool {
        matches!(self, Ty::Primitive(PrimitiveType::Unit))
    }

    /// Returns true if this is the never type.
    pub fn is_never(&self) -> bool {
        matches!(self, Ty::Primitive(PrimitiveType::Never))
    }

    /// Returns true if this is a signed integer type.
    pub fn is_signed(&self) -> bool {
        matches!(self, Ty::Primitive(PrimitiveType::Int { signed: true, .. }))
    }

    /// Returns true if this is an unsigned integer type.
    pub fn is_unsigned(&self) -> bool {
        matches!(self, Ty::Primitive(PrimitiveType::Int { signed: false, .. }))
    }

    /// Returns the IntWidth if this is an integer type.
    pub fn int_width(&self) -> Option<IntWidth> {
        match self {
            Ty::Primitive(PrimitiveType::Int { width, .. }) => Some(*width),
            _ => None,
        }
    }

    /// Returns the FloatWidth if this is a float type.
    pub fn float_width(&self) -> Option<FloatWidth> {
        match self {
            Ty::Primitive(PrimitiveType::Float(fw)) => Some(*fw),
            _ => None,
        }
    }

    /// Returns the pointee type if this is a reference/pointer type.
    pub fn pointee(&self) -> Option<&Ty> {
        match self {
            Ty::Ref { pointee, .. } => Some(pointee),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// LLVM2-specific types (not in real tMIR)
// ---------------------------------------------------------------------------

/// Struct type identifier (index into a struct definition table).
///
/// LLVM2 extension: real tMIR uses inline struct definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StructId(pub u32);

/// Function type: parameter types and return types.
///
/// LLVM2 extension: real tMIR uses `FnPtr { params, ret }` with single return
/// boxed as `Box<Type>`. This struct supports multiple returns for LLVM2.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FuncTy {
    pub params: Vec<Ty>,
    pub returns: Vec<Ty>,
}

/// A named field in an LLVM2 struct definition (for StructId resolution).
///
/// LLVM2 extension for backward compatibility with the adapter's struct table.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FieldDef {
    pub name: String,
    pub ty: Ty,
    /// Byte offset within the struct (populated by layout).
    pub offset: Option<u32>,
}

/// Struct definition: name + ordered fields (for StructId resolution).
///
/// LLVM2 extension for backward compatibility with the adapter's struct table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructDef {
    pub id: StructId,
    pub name: String,
    pub fields: Vec<FieldDef>,
    /// Total size in bytes (populated by layout).
    pub size: Option<u32>,
    /// Alignment in bytes (populated by layout).
    pub align: Option<u32>,
}

/// A tMIR value reference (SSA value ID within a function).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ValueId(pub u32);

/// A basic block identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(pub u32);

/// A function identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FuncId(pub u32);

// ---------------------------------------------------------------------------
// Proof annotations
// ---------------------------------------------------------------------------

/// Proof annotations attached to tMIR instructions or functions.
///
/// These are generated by the source-language compiler (tRust, tSwift, tC)
/// and formally verified by z4. Each annotation represents a proven property
/// that enables downstream optimizations in LLVM2.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TmirProof {
    /// Function or computation has no side effects.
    /// Enables: GPU/ANE dispatch, aggressive reordering, CSE.
    Pure,

    /// Borrow is guaranteed valid (lifetime within scope).
    /// Enables: skip liveness check, treat as raw pointer, memory aliasing refinement.
    ValidBorrow { borrow: ValueId },

    /// Array/pointer access is within bounds.
    /// Enables: skip bounds check, use direct load/store, GPU dispatch.
    InBounds { base: ValueId, index: ValueId },

    /// Operation is associative: (a op b) op c = a op (b op c).
    /// Enables: parallel reduction, operation reordering.
    Associative,

    /// Operation is commutative: a op b = b op a.
    /// Enables: parallel reduction, operand reordering.
    Commutative,

    /// Addition/subtraction guaranteed not to overflow.
    /// Enables: skip overflow check, use wrapping arithmetic directly.
    NoOverflow { signed: bool },

    /// Pointer is guaranteed non-null.
    /// Enables: skip null check.
    NotNull { ptr: ValueId },

    /// Division divisor is guaranteed non-zero.
    /// Enables: skip divide-by-zero check.
    NonZeroDivisor { divisor: ValueId },

    /// Shift amount is within [0, bitwidth).
    /// Enables: skip shift-amount masking.
    ValidShift { amount: ValueId, bitwidth: u16 },

    /// Value is within a specific range [lo, hi].
    /// Enables: range-based optimizations (width narrowing, skip sign extension).
    InRange { lo: i128, hi: i128 },

    /// Operation is idempotent: f(f(x)) = f(x).
    /// Enables: redundant application elimination.
    Idempotent,
}
