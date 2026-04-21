// llvm2-codegen/error.rs - Unified error types for the codegen crate
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Unified error types for the llvm2-codegen crate.
//!
//! Provides a top-level [`CodegenError`] that aggregates errors from all
//! codegen subsystems: encoding, frame lowering, branch relaxation,
//! relocations, and the compilation pipeline.

use thiserror::Error;

use crate::aarch64::encode::EncodeError;
use crate::lower::LowerError;
use crate::pipeline::PipelineError;
use crate::relax::RelaxError;

/// Top-level error type for the llvm2-codegen crate.
///
/// Wraps errors from all codegen subsystems into a single error type
/// for callers that use multiple codegen facilities.
#[derive(Debug, Error)]
pub enum CodegenError {
    /// Instruction encoding error (AArch64 encoder).
    #[error("encoding error: {0}")]
    Encoding(#[from] EncodeError),

    /// Machine code lowering error (instruction selection to binary).
    #[error("lowering error: {0}")]
    Lowering(#[from] LowerError),

    /// Branch relaxation error (out-of-range branches).
    #[error("relaxation error: {0}")]
    Relaxation(#[from] RelaxError),

    /// Compilation pipeline error.
    #[error("pipeline error: {0}")]
    Pipeline(#[from] PipelineError),
}
