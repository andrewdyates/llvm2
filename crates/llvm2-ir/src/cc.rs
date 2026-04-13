// llvm2-ir - Shared machine IR model
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! AArch64 condition codes and operand sizes.

/// AArch64 condition codes (4-bit encoding, ARM ARM C1.2.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AArch64CC {
    /// Equal (Z == 1)
    EQ = 0b0000,
    /// Not equal (Z == 0)
    NE = 0b0001,
    /// Carry set / unsigned higher or same (C == 1)
    HS = 0b0010,
    /// Carry clear / unsigned lower (C == 0)
    LO = 0b0011,
    /// Minus / negative (N == 1)
    MI = 0b0100,
    /// Plus / positive or zero (N == 0)
    PL = 0b0101,
    /// Overflow (V == 1)
    VS = 0b0110,
    /// No overflow (V == 0)
    VC = 0b0111,
    /// Unsigned higher (C == 1 && Z == 0)
    HI = 0b1000,
    /// Unsigned lower or same (C == 0 || Z == 1)
    LS = 0b1001,
    /// Signed greater or equal (N == V)
    GE = 0b1010,
    /// Signed less than (N != V)
    LT = 0b1011,
    /// Signed greater than (Z == 0 && N == V)
    GT = 0b1100,
    /// Signed less or equal (Z == 1 || N != V)
    LE = 0b1101,
    /// Always (unconditional)
    AL = 0b1110,
    /// Never (reserved, behaves as always)
    NV = 0b1111,
}

impl AArch64CC {
    /// Returns the inverted condition code.
    pub fn invert(self) -> Self {
        match self {
            Self::EQ => Self::NE,
            Self::NE => Self::EQ,
            Self::HS => Self::LO,
            Self::LO => Self::HS,
            Self::MI => Self::PL,
            Self::PL => Self::MI,
            Self::VS => Self::VC,
            Self::VC => Self::VS,
            Self::HI => Self::LS,
            Self::LS => Self::HI,
            Self::GE => Self::LT,
            Self::LT => Self::GE,
            Self::GT => Self::LE,
            Self::LE => Self::GT,
            Self::AL => Self::NV,
            Self::NV => Self::AL,
        }
    }

    /// Returns the 4-bit encoding for this condition code.
    pub fn encoding(self) -> u8 {
        self as u8
    }
}

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
