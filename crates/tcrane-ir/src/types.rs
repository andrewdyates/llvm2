// tcrane-ir/types.rs - IR type system
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Type system for Cranelift IR.

use serde::{Deserialize, Serialize};

/// Scalar types in Cranelift IR.
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

    /// Returns the size in bits.
    pub fn bits(self) -> u32 {
        self.bytes() * 8
    }
}
