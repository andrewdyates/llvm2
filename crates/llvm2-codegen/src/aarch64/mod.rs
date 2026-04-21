// llvm2-codegen/aarch64/mod.rs - AArch64 target encoding modules
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! AArch64 (ARM64) instruction encoding.
//!
//! Submodules:
//! - [`encode`] — Unified instruction encoder (dispatches to format-specific modules)
//! - [`encoding`] — Data-processing instruction encoding (ALU, branches, moves)
//! - [`encoding_mem`] — Load/store and address (ADRP) encoding
//! - [`encoding_fp`] — Floating-point instruction encoding

pub mod encode;
pub mod encoding;
pub mod encoding_fp;
pub mod encoding_mem;
pub mod encoding_neon;

pub use encode::{encode_instruction, EncodeError};
