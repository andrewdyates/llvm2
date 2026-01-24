// llvm2-verify - Verification backend
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Verification backend for LLVM2.
//!
//! This crate provides formal verification of tMIR → machine code transformations
//! using SMT solving (z4). It can verify:
//!
//! - Instruction lowering correctness
//! - Optimization validity
//! - Type safety
//! - Memory safety

pub mod smt;
pub mod verify;

pub use verify::{VerificationResult, Verifier};
