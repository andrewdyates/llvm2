// llvm2-codegen - Verified machine code generation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Verified machine code generation for LLVM2.
//!
//! This crate generates machine code from LIR with formal verification
//! of correctness. Each lowering step is verified using llvm2-verify.
//!
//! Supported targets:
//! - x86-64
//! - AArch64
//! - RISC-V

pub mod lower;
pub mod macho;
pub mod target;

pub use target::Target;
