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

/// Scalar types in LIR (input-level, pre-instruction-selection).
///
/// This is intentionally separate from `llvm2_ir::function::Type` — see
/// module-level docs for rationale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
}

impl Type {
    /// Returns the size in bytes.
    pub fn bytes(self) -> u32 {
        match self {
            Type::B1 => 1,
            Type::I8 => 1,
            Type::I16 => 2,
            Type::I32 | Type::F32 => 4,
            Type::I64 | Type::F64 => 8,
            Type::I128 => 16,
        }
    }

    /// Returns the semantic width in bits.
    ///
    /// Note: B1 returns 1 (semantic bit-width), not 8 (storage size).
    /// Use `bytes()` for the storage size.
    pub fn bits(self) -> u32 {
        match self {
            Type::B1 => 1,
            _ => self.bytes() * 8,
        }
    }
}
