// tcrane-verify - Verification backend
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Verification backend for tCrane.
//!
//! This crate provides formal verification of Cranelift IR transformations
//! using SMT solving (z4). It can verify:
//!
//! - Instruction lowering correctness
//! - Optimization validity
//! - Type safety
//! - Memory safety

pub mod smt;
pub mod verify;

pub use verify::{VerificationResult, Verifier};
