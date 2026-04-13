// llvm2-codegen/aarch64/encoding_fp.rs - AArch64 floating-point instruction encoding
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! AArch64 floating-point instruction encoding.
//!
//! Implements encoding for AArch64 FP instruction formats:
//! - FP data-processing 2-source (FADD, FSUB, FMUL, FDIV)
//! - FP data-processing 1-source (FMOV reg, FABS, FNEG, FSQRT)
//! - FP compare (FCMP, FCMPE, with register or zero)
//! - FP ↔ integer conversion (FCVTZS, SCVTF, FMOV between GP and FP)
//!
//! Encoding formats follow the ARM Architecture Reference Manual (DDI 0487).

use thiserror::Error;

/// Errors produced during floating-point instruction encoding.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum FpEncodeError {
    #[error("register index {reg} out of range (max {max})")]
    RegisterOutOfRange { reg: u8, max: u8 },
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Floating-point data size (maps to `ftype` field).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpSize {
    /// Single precision (32-bit) — ftype = 0b00
    Single = 0b00,
    /// Double precision (64-bit) — ftype = 0b01
    Double = 0b01,
    /// Half precision (16-bit) — ftype = 0b11
    Half = 0b11,
}

/// FP arithmetic operation (2-source data-processing).
///
/// Maps to the 4-bit `opcode` field in bits [15:12].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpArithOp {
    /// FMUL — opcode = 0b0000
    Mul = 0b0000,
    /// FDIV — opcode = 0b0001
    Div = 0b0001,
    /// FADD — opcode = 0b0010
    Add = 0b0010,
    /// FSUB — opcode = 0b0011
    Sub = 0b0011,
}

/// FP compare operation.
///
/// Maps to the 5-bit `opc` field in bits [4:0] of FCMP encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpCmpOp {
    /// FCMP (register) — opc = 0b00000
    Cmp = 0b00000,
    /// FCMP with zero — opc = 0b01000
    CmpZero = 0b01000,
    /// FCMPE (signalling, register) — opc = 0b10000
    Cmpe = 0b10000,
    /// FCMPE with zero — opc = 0b11000
    CmpeZero = 0b11000,
}

/// FP ↔ integer conversion operation.
///
/// Each variant encodes a (rmode, opcode) pair for bits [20:19] and [18:16].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpConvOp {
    /// FCVTZS — FP to signed integer, round toward zero.
    /// rmode = 0b11, opcode = 0b000
    FcvtzsToInt,
    /// SCVTF — signed integer to FP.
    /// rmode = 0b00, opcode = 0b010
    ScvtfToFp,
    /// FMOV — move GP register to FP register.
    /// rmode = 0b00, opcode = 0b111
    FmovToFp,
    /// FMOV — move FP register to GP register.
    /// rmode = 0b00, opcode = 0b110
    FmovToGp,
}

impl FpConvOp {
    /// Returns (rmode, opcode) for the conversion instruction.
    fn rmode_opcode(self) -> (u32, u32) {
        match self {
            FpConvOp::FcvtzsToInt => (0b11, 0b000),
            FpConvOp::ScvtfToFp => (0b00, 0b010),
            FpConvOp::FmovToFp => (0b00, 0b111),
            FpConvOp::FmovToGp => (0b00, 0b110),
        }
    }
}

/// FP unary (1-source) data-processing operation.
///
/// Maps to the 2-bit `opcode` field in bits [16:15].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpUnaryOp {
    /// FMOV (register to register) — opcode = 0b00
    FmovReg = 0b00,
    /// FABS — opcode = 0b01
    Fabs = 0b01,
    /// FNEG — opcode = 0b10
    Fneg = 0b10,
    /// FSQRT — opcode = 0b11
    Fsqrt = 0b11,
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

#[inline]
fn check_reg(reg: u8, max: u8) -> Result<(), FpEncodeError> {
    if reg > max {
        return Err(FpEncodeError::RegisterOutOfRange { reg, max });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// FP data-processing 2-source (FADD, FSUB, FMUL, FDIV)
// ---------------------------------------------------------------------------

/// Encode a 2-source FP data-processing instruction.
///
/// ```text
/// 0 | 00 | 11110 | ftype(2) | 1 | Rm(5) | opcode(4) | 10 | Rn(5) | Rd(5)
/// ```
pub fn encode_fp_arith(
    fp_size: FpSize,
    op: FpArithOp,
    rm: u8,
    rn: u8,
    rd: u8,
) -> Result<u32, FpEncodeError> {
    check_reg(rm, 31)?;
    check_reg(rn, 31)?;
    check_reg(rd, 31)?;

    let mut inst: u32 = 0;
    // bits[31:29] = 000
    inst |= 0b11110 << 24;
    inst |= (fp_size as u32) << 22;
    inst |= 1 << 21;
    inst |= (rm as u32) << 16;
    inst |= (op as u32) << 12;
    inst |= 0b10 << 10;
    inst |= (rn as u32) << 5;
    inst |= rd as u32;
    Ok(inst)
}

// ---------------------------------------------------------------------------
// FCMP / FCMPE
// ---------------------------------------------------------------------------

/// Encode an FP compare instruction.
///
/// ```text
/// 0 | 00 | 11110 | ftype(2) | 1 | Rm(5) | 00 | 1000 | Rn(5) | opc(5)
/// ```
///
/// For zero-compare variants (`CmpZero`, `CmpeZero`), `rm` is ignored and
/// encoded as 0 (matching LLVM's `fixOneOperandFPComparison`).
pub fn encode_fcmp(
    fp_size: FpSize,
    cmp_op: FpCmpOp,
    rm: u8,
    rn: u8,
) -> Result<u32, FpEncodeError> {
    check_reg(rn, 31)?;

    // For register-compare variants, validate Rm.
    let rm_val = match cmp_op {
        FpCmpOp::CmpZero | FpCmpOp::CmpeZero => 0u32,
        _ => {
            check_reg(rm, 31)?;
            rm as u32
        }
    };

    let mut inst: u32 = 0;
    inst |= 0b11110 << 24;
    inst |= (fp_size as u32) << 22;
    inst |= 1 << 21;
    inst |= rm_val << 16;
    inst |= 0b00 << 14;
    inst |= 0b1000 << 10;
    inst |= (rn as u32) << 5;
    inst |= cmp_op as u32;
    Ok(inst)
}

// ---------------------------------------------------------------------------
// FP ↔ integer conversion
// ---------------------------------------------------------------------------

/// Encode an FP-to-integer or integer-to-FP conversion instruction.
///
/// ```text
/// sf(1) | 00 | 11110 | ftype(2) | 1 | rmode(2) | opcode(3) | 000000 | Rn(5) | Rd(5)
/// ```
///
/// * `sf_64` — `true` for 64-bit integer (X register), `false` for 32-bit (W)
/// * `fp_size` — floating-point precision
/// * `conv_op` — conversion type
/// * `rn` — source register (0..31)
/// * `rd` — destination register (0..31)
pub fn encode_fp_int_conv(
    sf_64: bool,
    fp_size: FpSize,
    conv_op: FpConvOp,
    rn: u8,
    rd: u8,
) -> Result<u32, FpEncodeError> {
    check_reg(rn, 31)?;
    check_reg(rd, 31)?;

    let (rmode, opcode) = conv_op.rmode_opcode();

    let mut inst: u32 = 0;
    inst |= (sf_64 as u32) << 31;
    // bits[30:29] = 00
    inst |= 0b11110 << 24;
    inst |= (fp_size as u32) << 22;
    inst |= 1 << 21;
    inst |= rmode << 19;
    inst |= opcode << 16;
    // bits[15:10] = 000000
    inst |= (rn as u32) << 5;
    inst |= rd as u32;
    Ok(inst)
}

// ---------------------------------------------------------------------------
// FP data-processing 1-source (FMOV reg, FABS, FNEG, FSQRT)
// ---------------------------------------------------------------------------

/// Encode a 1-source FP data-processing instruction.
///
/// ```text
/// 0 | 00 | 11110 | ftype(2) | 1 | 0000 | opcode(2) | 10000 | Rn(5) | Rd(5)
/// ```
pub fn encode_fp_unary(
    fp_size: FpSize,
    op: FpUnaryOp,
    rn: u8,
    rd: u8,
) -> Result<u32, FpEncodeError> {
    check_reg(rn, 31)?;
    check_reg(rd, 31)?;

    let mut inst: u32 = 0;
    inst |= 0b11110 << 24;
    inst |= (fp_size as u32) << 22;
    inst |= 1 << 21;
    // bits[20:17] = 0000
    inst |= (op as u32) << 15;
    inst |= 0b10000 << 10;
    inst |= (rn as u32) << 5;
    inst |= rd as u32;
    Ok(inst)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // === FP arithmetic (2-source) ===

    #[test]
    fn test_fadd_single() {
        // FADD S0, S1, S2
        let enc = encode_fp_arith(FpSize::Single, FpArithOp::Add, 2, 1, 0).unwrap();
        assert_eq!(enc, 0x1E22_2820);
    }

    #[test]
    fn test_fadd_double() {
        // FADD D0, D1, D2
        let enc = encode_fp_arith(FpSize::Double, FpArithOp::Add, 2, 1, 0).unwrap();
        assert_eq!(enc, 0x1E62_2820);
    }

    #[test]
    fn test_fsub_double() {
        // FSUB D3, D4, D5
        let enc = encode_fp_arith(FpSize::Double, FpArithOp::Sub, 5, 4, 3).unwrap();
        assert_eq!(enc, 0x1E65_3883);
    }

    #[test]
    fn test_fmul_single() {
        // FMUL S10, S11, S12
        let enc = encode_fp_arith(FpSize::Single, FpArithOp::Mul, 12, 11, 10).unwrap();
        assert_eq!(enc, 0x1E2C_096A);
    }

    #[test]
    fn test_fdiv_double() {
        // FDIV D0, D1, D2
        let enc = encode_fp_arith(FpSize::Double, FpArithOp::Div, 2, 1, 0).unwrap();
        assert_eq!(enc, 0x1E62_1820);
    }

    #[test]
    fn test_fp_arith_half() {
        // FADD H5, H6, H7 — ftype=11
        let enc = encode_fp_arith(FpSize::Half, FpArithOp::Add, 7, 6, 5).unwrap();
        // Expected: 0|00|11110|11|1|00111|0010|10|00110|00101
        let expected = (0b11110u32 << 24) | (0b11 << 22) | (1 << 21)
            | (7 << 16) | (0b0010 << 12) | (0b10 << 10) | (6 << 5) | 5;
        assert_eq!(enc, expected);
    }

    #[test]
    fn test_fp_arith_reg_overflow() {
        let err = encode_fp_arith(FpSize::Single, FpArithOp::Add, 32, 0, 0);
        assert!(matches!(err, Err(FpEncodeError::RegisterOutOfRange { reg: 32, max: 31 })));
    }

    // === FCMP ===

    #[test]
    fn test_fcmp_single_regs() {
        // FCMP S1, S2
        let enc = encode_fcmp(FpSize::Single, FpCmpOp::Cmp, 2, 1).unwrap();
        assert_eq!(enc, 0x1E22_2020);
    }

    #[test]
    fn test_fcmp_double_zero() {
        // FCMP D1, #0.0
        let enc = encode_fcmp(FpSize::Double, FpCmpOp::CmpZero, 0, 1).unwrap();
        assert_eq!(enc, 0x1E60_2028);
    }

    #[test]
    fn test_fcmpe_single_regs() {
        // FCMPE S0, S1
        let enc = encode_fcmp(FpSize::Single, FpCmpOp::Cmpe, 1, 0).unwrap();
        // opc = 0b10000
        let expected = (0b11110u32 << 24) | (0b00 << 22) | (1 << 21)
            | (1 << 16) | (0b1000 << 10) | 0b10000;
        assert_eq!(enc, expected);
    }

    #[test]
    fn test_fcmpe_double_zero() {
        // FCMPE D0, #0.0  — Rm zeroed
        let enc = encode_fcmp(FpSize::Double, FpCmpOp::CmpeZero, 255, 0).unwrap();
        // Rm is forced to 0 for zero variants (rm parameter ignored)
        let expected = (0b11110u32 << 24) | (0b01 << 22) | (1 << 21)
            | (0b1000 << 10) | 0b11000;
        assert_eq!(enc, expected);
    }

    // === FP ↔ integer conversion ===

    #[test]
    fn test_fcvtzs_w_s() {
        // FCVTZS W0, S1: sf=0, ftype=00, rmode=11, opcode=000
        let enc = encode_fp_int_conv(false, FpSize::Single, FpConvOp::FcvtzsToInt, 1, 0).unwrap();
        assert_eq!(enc, 0x1E38_0020);
    }

    #[test]
    fn test_scvtf_s_w() {
        // SCVTF S0, W1: sf=0, ftype=00, rmode=00, opcode=010
        let enc = encode_fp_int_conv(false, FpSize::Single, FpConvOp::ScvtfToFp, 1, 0).unwrap();
        assert_eq!(enc, 0x1E22_0020);
    }

    #[test]
    fn test_scvtf_d_x() {
        // SCVTF D0, X1: sf=1, ftype=01, rmode=00, opcode=010
        let enc = encode_fp_int_conv(true, FpSize::Double, FpConvOp::ScvtfToFp, 1, 0).unwrap();
        assert_eq!(enc, 0x9E62_0020);
    }

    #[test]
    fn test_fmov_gp_to_fp() {
        // FMOV S0, W1: sf=0, ftype=00, rmode=00, opcode=111
        let enc = encode_fp_int_conv(false, FpSize::Single, FpConvOp::FmovToFp, 1, 0).unwrap();
        let expected = (0b11110u32 << 24) | (1 << 21) | (0b111 << 16) | (1 << 5);
        assert_eq!(enc, expected);
    }

    #[test]
    fn test_fmov_fp_to_gp() {
        // FMOV W0, S1: sf=0, ftype=00, rmode=00, opcode=110
        let enc = encode_fp_int_conv(false, FpSize::Single, FpConvOp::FmovToGp, 1, 0).unwrap();
        let expected = (0b11110u32 << 24) | (1 << 21) | (0b110 << 16) | (1 << 5);
        assert_eq!(enc, expected);
    }

    #[test]
    fn test_fp_conv_reg_overflow() {
        let err = encode_fp_int_conv(false, FpSize::Single, FpConvOp::FcvtzsToInt, 32, 0);
        assert!(matches!(err, Err(FpEncodeError::RegisterOutOfRange { reg: 32, .. })));
    }

    // === FP unary (1-source) ===

    #[test]
    fn test_fneg_single() {
        // FNEG S0, S1: ftype=00, opcode=10
        let enc = encode_fp_unary(FpSize::Single, FpUnaryOp::Fneg, 1, 0).unwrap();
        assert_eq!(enc, 0x1E21_4020);
    }

    #[test]
    fn test_fabs_double() {
        // FABS D3, D4: ftype=01, opcode=01
        let enc = encode_fp_unary(FpSize::Double, FpUnaryOp::Fabs, 4, 3).unwrap();
        assert_eq!(enc, 0x1E60_C083);
    }

    #[test]
    fn test_fsqrt_double() {
        // FSQRT D0, D1: ftype=01, opcode=11
        let enc = encode_fp_unary(FpSize::Double, FpUnaryOp::Fsqrt, 1, 0).unwrap();
        assert_eq!(enc, 0x1E61_C020);
    }

    #[test]
    fn test_fmov_reg_single() {
        // FMOV S5, S6: ftype=00, opcode=00
        let enc = encode_fp_unary(FpSize::Single, FpUnaryOp::FmovReg, 6, 5).unwrap();
        let expected = (0b11110u32 << 24) | (1 << 21) | (0b10000 << 10) | (6 << 5) | 5;
        assert_eq!(enc, expected);
    }

    #[test]
    fn test_fp_unary_reg_overflow() {
        let err = encode_fp_unary(FpSize::Single, FpUnaryOp::Fneg, 0, 32);
        assert!(matches!(err, Err(FpEncodeError::RegisterOutOfRange { reg: 32, .. })));
    }

    // === All FpArithOp opcodes ===

    #[test]
    fn test_fp_arith_opcodes() {
        for (op, expected_bits) in [
            (FpArithOp::Mul, 0b0000u32),
            (FpArithOp::Div, 0b0001),
            (FpArithOp::Add, 0b0010),
            (FpArithOp::Sub, 0b0011),
        ] {
            let enc = encode_fp_arith(FpSize::Single, op, 0, 0, 0).unwrap();
            let opcode_field = (enc >> 12) & 0b1111;
            assert_eq!(opcode_field, expected_bits, "opcode mismatch for {:?}", op);
        }
    }

    // === All FpSize ftype values ===

    #[test]
    fn test_fp_size_ftype() {
        for (size, expected_bits) in [
            (FpSize::Single, 0b00u32),
            (FpSize::Double, 0b01),
            (FpSize::Half, 0b11),
        ] {
            let enc = encode_fp_arith(size, FpArithOp::Add, 0, 0, 0).unwrap();
            let ftype_field = (enc >> 22) & 0b11;
            assert_eq!(ftype_field, expected_bits, "ftype mismatch for {:?}", size);
        }
    }
}
