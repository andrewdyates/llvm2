// tcrane-ir - Cranelift IR representation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Cranelift IR types and data structures for tCrane.
//!
//! This crate provides the core IR representation used by tCrane,
//! compatible with Cranelift's CLIF format but designed for verification.

pub mod types;
pub mod instructions;
pub mod function;

pub use types::Type;
pub use function::Function;
