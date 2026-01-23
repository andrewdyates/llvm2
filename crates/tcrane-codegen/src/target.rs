// tcrane-codegen/target.rs - Target architectures
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Target architecture definitions.

/// Supported target architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// x86-64 (AMD64)
    X86_64,
    /// AArch64 (ARM64)
    Aarch64,
    /// RISC-V 64-bit
    Riscv64,
}

impl Target {
    /// Returns the pointer size in bytes for this target.
    pub fn pointer_bytes(self) -> u32 {
        match self {
            Target::X86_64 | Target::Aarch64 | Target::Riscv64 => 8,
        }
    }

    /// Returns the name of this target.
    pub fn name(self) -> &'static str {
        match self {
            Target::X86_64 => "x86_64",
            Target::Aarch64 => "aarch64",
            Target::Riscv64 => "riscv64",
        }
    }
}
