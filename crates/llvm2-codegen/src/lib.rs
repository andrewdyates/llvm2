// llvm2-codegen - Verified machine code generation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Verified machine code generation for LLVM2.
//!
//! This crate generates machine code from LIR with formal verification
//! of correctness. Each lowering step is verified using llvm2-verify.
//!
//! Target: AArch64 macOS (Apple Silicon) for MVP.
//! x86-64 and RISC-V are future targets.

pub mod aarch64;
pub mod compiler;
pub mod coreml_emitter;
pub mod dwarf_cfi;
pub mod dwarf_info;
pub mod error;
pub mod frame;
pub mod layout;
pub mod lower;
pub mod macho;
pub mod metal_emitter;
pub mod pipeline;
pub mod relax;
pub mod target;
pub mod unwind;
pub mod x86_64;

pub use compiler::{
    CompilationMetrics, CompilationResult, CompileError,
    Compiler, CompilerConfig, CompilerTrace, CompilerTraceLevel, ProofCertificate,
};
pub use error::CodegenError;
pub use pipeline::{Pipeline, PipelineConfig, PipelineError, compile_to_object};
pub use relax::RelaxError;
pub use target::{CallingConvention, Target};
