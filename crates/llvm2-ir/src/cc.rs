// llvm2-ir - Shared machine IR model
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

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

#[cfg(test)]
mod tests {
    use super::*;

    // ---- AArch64CC encoding tests ----

    #[test]
    fn cc_encoding_values() {
        assert_eq!(AArch64CC::EQ.encoding(), 0b0000);
        assert_eq!(AArch64CC::NE.encoding(), 0b0001);
        assert_eq!(AArch64CC::HS.encoding(), 0b0010);
        assert_eq!(AArch64CC::LO.encoding(), 0b0011);
        assert_eq!(AArch64CC::MI.encoding(), 0b0100);
        assert_eq!(AArch64CC::PL.encoding(), 0b0101);
        assert_eq!(AArch64CC::VS.encoding(), 0b0110);
        assert_eq!(AArch64CC::VC.encoding(), 0b0111);
        assert_eq!(AArch64CC::HI.encoding(), 0b1000);
        assert_eq!(AArch64CC::LS.encoding(), 0b1001);
        assert_eq!(AArch64CC::GE.encoding(), 0b1010);
        assert_eq!(AArch64CC::LT.encoding(), 0b1011);
        assert_eq!(AArch64CC::GT.encoding(), 0b1100);
        assert_eq!(AArch64CC::LE.encoding(), 0b1101);
        assert_eq!(AArch64CC::AL.encoding(), 0b1110);
        assert_eq!(AArch64CC::NV.encoding(), 0b1111);
    }

    #[test]
    fn cc_encoding_is_4_bit() {
        let all = [
            AArch64CC::EQ, AArch64CC::NE, AArch64CC::HS, AArch64CC::LO,
            AArch64CC::MI, AArch64CC::PL, AArch64CC::VS, AArch64CC::VC,
            AArch64CC::HI, AArch64CC::LS, AArch64CC::GE, AArch64CC::LT,
            AArch64CC::GT, AArch64CC::LE, AArch64CC::AL, AArch64CC::NV,
        ];
        for cc in &all {
            assert!(cc.encoding() <= 0b1111, "encoding must fit in 4 bits");
        }
    }

    // ---- AArch64CC inversion tests ----

    #[test]
    fn cc_inversion_pairs() {
        assert_eq!(AArch64CC::EQ.invert(), AArch64CC::NE);
        assert_eq!(AArch64CC::NE.invert(), AArch64CC::EQ);
        assert_eq!(AArch64CC::HS.invert(), AArch64CC::LO);
        assert_eq!(AArch64CC::LO.invert(), AArch64CC::HS);
        assert_eq!(AArch64CC::MI.invert(), AArch64CC::PL);
        assert_eq!(AArch64CC::PL.invert(), AArch64CC::MI);
        assert_eq!(AArch64CC::VS.invert(), AArch64CC::VC);
        assert_eq!(AArch64CC::VC.invert(), AArch64CC::VS);
        assert_eq!(AArch64CC::HI.invert(), AArch64CC::LS);
        assert_eq!(AArch64CC::LS.invert(), AArch64CC::HI);
        assert_eq!(AArch64CC::GE.invert(), AArch64CC::LT);
        assert_eq!(AArch64CC::LT.invert(), AArch64CC::GE);
        assert_eq!(AArch64CC::GT.invert(), AArch64CC::LE);
        assert_eq!(AArch64CC::LE.invert(), AArch64CC::GT);
        assert_eq!(AArch64CC::AL.invert(), AArch64CC::NV);
        assert_eq!(AArch64CC::NV.invert(), AArch64CC::AL);
    }

    #[test]
    fn cc_double_inversion_is_identity() {
        let all = [
            AArch64CC::EQ, AArch64CC::NE, AArch64CC::HS, AArch64CC::LO,
            AArch64CC::MI, AArch64CC::PL, AArch64CC::VS, AArch64CC::VC,
            AArch64CC::HI, AArch64CC::LS, AArch64CC::GE, AArch64CC::LT,
            AArch64CC::GT, AArch64CC::LE, AArch64CC::AL, AArch64CC::NV,
        ];
        for cc in &all {
            assert_eq!(cc.invert().invert(), *cc, "double invert must be identity for {:?}", cc);
        }
    }

    #[test]
    fn cc_invert_changes_value() {
        let all = [
            AArch64CC::EQ, AArch64CC::NE, AArch64CC::HS, AArch64CC::LO,
            AArch64CC::MI, AArch64CC::PL, AArch64CC::VS, AArch64CC::VC,
            AArch64CC::HI, AArch64CC::LS, AArch64CC::GE, AArch64CC::LT,
            AArch64CC::GT, AArch64CC::LE, AArch64CC::AL, AArch64CC::NV,
        ];
        for cc in &all {
            assert_ne!(*cc, cc.invert(), "invert must produce different value for {:?}", cc);
        }
    }

    // ---- OperandSize tests ----

    #[test]
    fn operand_size_sf_bit() {
        assert_eq!(OperandSize::S32.sf_bit(), 0);
        assert_eq!(OperandSize::S64.sf_bit(), 1);
    }

    #[test]
    fn operand_size_bytes() {
        assert_eq!(OperandSize::S32.bytes(), 4);
        assert_eq!(OperandSize::S64.bytes(), 8);
    }

    #[test]
    fn operand_size_equality() {
        assert_eq!(OperandSize::S32, OperandSize::S32);
        assert_eq!(OperandSize::S64, OperandSize::S64);
        assert_ne!(OperandSize::S32, OperandSize::S64);
    }

    #[test]
    fn operand_size_copy_clone() {
        let a = OperandSize::S32;
        let b = a; // Copy
        let c = a; // Clone
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    // ---- FloatSize tests ----

    #[test]
    fn float_size_ftype() {
        assert_eq!(FloatSize::F32.ftype(), 0);
        assert_eq!(FloatSize::F64.ftype(), 1);
    }

    #[test]
    fn float_size_bytes() {
        assert_eq!(FloatSize::F32.bytes(), 4);
        assert_eq!(FloatSize::F64.bytes(), 8);
    }

    #[test]
    fn float_size_equality() {
        assert_eq!(FloatSize::F32, FloatSize::F32);
        assert_eq!(FloatSize::F64, FloatSize::F64);
        assert_ne!(FloatSize::F32, FloatSize::F64);
    }

    #[test]
    fn float_size_copy_clone() {
        let a = FloatSize::F64;
        let b = a;
        let c = a;
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    // ---- Cross-type consistency tests ----

    #[test]
    fn operand_size_and_float_size_byte_values_consistent() {
        // 32-bit variants should both be 4 bytes
        assert_eq!(OperandSize::S32.bytes(), FloatSize::F32.bytes());
        // 64-bit variants should both be 8 bytes
        assert_eq!(OperandSize::S64.bytes(), FloatSize::F64.bytes());
    }
}
