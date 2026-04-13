// llvm2-lower/types.rs - LIR type system
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Type system for LLVM2 Low-level IR (input-level types).
//!
//! ## Why this is NOT a re-export of `llvm2_ir::function::Type`
//!
//! This `Type` enum represents tMIR/LIR input-level scalar types, while
//! `llvm2_ir::function::Type` represents machine-level types used by the
//! backend after instruction selection. The key differences:
//!
//! | Aspect | `llvm2_lower::Type` | `llvm2_ir::Type` |
//! |--------|---------------------|------------------|
//! | Purpose | tMIR input types | MachIR types |
//! | `Ptr` variant | No (pointers are I64 at LIR level) | Yes |
//! | `serde` derives | Yes (for tMIR serialization) | No (zero-dep core) |
//! | `bits()` method | Yes | No |
//!
//! Once tMIR integration matures, this enum will align with tmir-types.
//! See issue #37 for tracking type unification.

use serde::{Deserialize, Serialize};

/// Types in LIR (input-level, pre-instruction-selection).
///
/// This is intentionally separate from `llvm2_ir::function::Type` — see
/// module-level docs for rationale. Includes aggregate types (Struct, Array)
/// for representing tMIR aggregate operations before they are decomposed
/// into scalar loads/stores during instruction selection.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Type {
    /// 8-bit integer
    I8,
    /// 16-bit integer
    I16,
    /// 32-bit integer
    I32,
    /// 64-bit integer
    I64,
    /// 128-bit integer
    I128,
    /// 32-bit float
    F32,
    /// 64-bit float
    F64,
    /// Boolean (1-bit)
    B1,
    /// Aggregate structure type with C-like field layout.
    Struct(Vec<Type>),
    /// Fixed-size array type: element type and count.
    Array(Box<Type>, u32),
}

impl Type {
    /// Round `offset` up to the next multiple of `align`.
    fn align_to(offset: u32, align: u32) -> u32 {
        if align <= 1 {
            offset
        } else {
            let rem = offset % align;
            if rem == 0 { offset } else { offset + (align - rem) }
        }
    }

    /// Returns the size in bytes.
    ///
    /// For structs, uses C-like layout with alignment padding.
    /// For arrays, returns element_size * count.
    pub fn bytes(&self) -> u32 {
        match self {
            Type::B1 | Type::I8 => 1,
            Type::I16 => 2,
            Type::I32 | Type::F32 => 4,
            Type::I64 | Type::F64 => 8,
            Type::I128 => 16,
            Type::Struct(fields) => {
                let mut offset: u32 = 0;
                let mut max_align: u32 = 1;
                for field in fields {
                    let a = field.align();
                    max_align = max_align.max(a);
                    offset = Self::align_to(offset, a);
                    offset += field.bytes();
                }
                Self::align_to(offset, max_align)
            }
            Type::Array(elem, count) => elem.bytes() * count,
        }
    }

    /// Alias for `bytes()`.
    pub fn size_of(&self) -> u32 {
        self.bytes()
    }

    /// Natural alignment in bytes.
    pub fn align(&self) -> u32 {
        match self {
            Type::Struct(fields) => fields.iter().map(|f| f.align()).max().unwrap_or(1),
            Type::Array(elem, _) => elem.align(),
            _ => self.bytes().min(8),
        }
    }

    /// Alias for `align()`.
    pub fn align_of(&self) -> u32 {
        self.align()
    }

    /// Byte offset of a struct field using C-like layout rules.
    ///
    /// Returns `None` if not a struct type or index is out of range.
    pub fn offset_of(&self, field_index: usize) -> Option<u32> {
        let Self::Struct(fields) = self else {
            return None;
        };
        if field_index >= fields.len() {
            return None;
        }
        let mut offset: u32 = 0;
        for (idx, field) in fields.iter().enumerate() {
            offset = Self::align_to(offset, field.align());
            if idx == field_index {
                return Some(offset);
            }
            offset += field.bytes();
        }
        None
    }

    /// Returns the semantic width in bits.
    ///
    /// Note: B1 returns 1 (semantic bit-width), not 8 (storage size).
    /// Use `bytes()` for the storage size.
    /// Returns None for aggregate types (use `bytes()` instead).
    pub fn bits(&self) -> u32 {
        match self {
            Type::B1 => 1,
            Type::Struct(_) | Type::Array(_, _) => self.bytes() * 8,
            _ => self.bytes() * 8,
        }
    }

    /// Returns true if this is an aggregate type.
    pub fn is_aggregate(&self) -> bool {
        matches!(self, Self::Struct(_) | Self::Array(_, _))
    }

    /// Returns true if this is a scalar (non-aggregate) type.
    pub fn is_scalar(&self) -> bool {
        !self.is_aggregate()
    }
}
