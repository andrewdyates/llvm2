// llvm2-ir - Shared machine IR model
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! AArch64 condition codes and operand sizes.
//!
//! The canonical condition code type is [`CondCode`] from `aarch64_regs`.
//! `AArch64CC` is a type alias for backward compatibility.

/// AArch64 condition codes — type alias for [`crate::aarch64_regs::CondCode`].
///
/// Use `CondCode` directly for new code. This alias exists for backward
/// compatibility with code that imported `llvm2_ir::AArch64CC`.
pub type AArch64CC = crate::aarch64_regs::CondCode;

/// Integer operand size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperandSize {
    /// 32-bit operation (uses W registers).
    S32,
    /// 64-bit operation (uses X registers).
    S64,
}

impl OperandSize {
    /// Returns the sf bit value for AArch64 encoding (0 = 32-bit, 1 = 64-bit).
    pub fn sf_bit(self) -> u32 {
        match self {
            Self::S32 => 0,
            Self::S64 => 1,
        }
    }

    /// Returns the size in bytes.
    pub fn bytes(self) -> u32 {
        match self {
            Self::S32 => 4,
            Self::S64 => 8,
        }
    }
}

/// Floating-point operand size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatSize {
    /// 32-bit float (S registers).
    F32,
    /// 64-bit float (D registers).
    F64,
}

impl FloatSize {
    /// Returns the ftype encoding for AArch64 (0 = F32, 1 = F64).
    pub fn ftype(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F64 => 1,
        }
    }

    /// Returns the size in bytes.
    pub fn bytes(self) -> u32 {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}
