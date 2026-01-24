// llvm2-codegen/lower.rs - Instruction lowering
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Instruction lowering from LIR to machine code.

use crate::target::Target;
use llvm2_lower::Function;
use thiserror::Error;

/// Lowering error.
#[derive(Debug, Error)]
pub enum LowerError {
    #[error("unsupported instruction: {0}")]
    UnsupportedInstruction(String),
    #[error("verification failed: {0}")]
    VerificationFailed(String),
}

/// Lower a function to machine code for the given target.
pub fn lower_function(_func: &Function, _target: Target) -> Result<Vec<u8>, LowerError> {
    // TODO: Implement lowering with verification
    Ok(Vec::new())
}
