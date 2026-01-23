// tcrane-codegen - Verified code generation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Verified code generation for tCrane.
//!
//! This crate generates machine code from Cranelift IR with formal
//! verification of correctness. Each lowering step is verified using
//! the tcrane-verify backend.

pub mod target;
pub mod lower;

pub use target::Target;
