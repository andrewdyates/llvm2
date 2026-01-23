// tcrane-verify/smt.rs - SMT encoding
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! SMT encoding of Cranelift IR for verification.

use tcrane_ir::types::Type;

/// SMT sort corresponding to IR types.
#[derive(Debug, Clone)]
pub enum Sort {
    BitVec(u32),
    Bool,
    Float32,
    Float64,
}

impl From<Type> for Sort {
    fn from(ty: Type) -> Self {
        match ty {
            Type::B1 => Sort::Bool,
            Type::I8 => Sort::BitVec(8),
            Type::I16 => Sort::BitVec(16),
            Type::I32 => Sort::BitVec(32),
            Type::I64 => Sort::BitVec(64),
            Type::I128 => Sort::BitVec(128),
            Type::F32 => Sort::Float32,
            Type::F64 => Sort::Float64,
        }
    }
}

/// SMT expression builder (placeholder for z4 integration).
pub struct SmtBuilder {
    // Will integrate with z4 solver
}

impl SmtBuilder {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for SmtBuilder {
    fn default() -> Self {
        Self::new()
    }
}
