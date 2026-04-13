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
pub mod frame;
pub mod lower;
pub mod macho;
pub mod pipeline;
pub mod target;
pub mod unwind;

pub use pipeline::{Pipeline, PipelineConfig, PipelineError, compile_to_object};
pub use target::Target;
