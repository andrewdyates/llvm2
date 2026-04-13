// llvm2-ir - Shared machine model
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Shared machine model for LLVM2.
//!
//! This crate defines the core types used across the compiler backend:
//! physical registers (PReg), register classes, operands, machine
//! instructions, and stack slots. Target-specific register definitions
//! live in submodules (e.g., `aarch64_regs`).

pub mod aarch64_regs;
