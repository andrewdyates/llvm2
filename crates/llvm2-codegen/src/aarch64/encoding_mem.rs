// llvm2-codegen/aarch64/encoding_mem.rs - AArch64 load/store and address encoding
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! AArch64 load/store instruction encoding.
//!
//! Implements encoding for all AArch64 memory access instruction formats:
//! - LDR/STR with unsigned offset, pre-index, post-index, and register offset
//! - LDP/STP with signed offset and pre-index
//! - ADRP (PC-relative page address)
//!
//! Encoding formats follow the ARM Architecture Reference Manual (DDI 0487).
//! Each function validates all field widths and returns a `Result` with the
//! encoded 32-bit instruction word.

use thiserror::Error;

/// Errors produced during load/store instruction encoding.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum EncodeError {
    #[error("register index {reg} out of range (max {max})")]
    RegisterOutOfRange { reg: u8, max: u8 },
    #[error("unsigned immediate {value} out of 12-bit range (0..4095)")]
    Imm12OutOfRange { value: u16 },
    #[error("signed immediate {value} out of 9-bit range (-256..255)")]
    Imm9OutOfRange { value: i16 },
    #[error("signed immediate {value} out of 7-bit range (-64..63)")]
    Imm7OutOfRange { value: i8 },
    #[error("signed immediate {value} out of 21-bit range (-1048576..1048575)")]
    Imm21OutOfRange { value: i32 },
    #[error("invalid extend option {0:#05b}")]
    InvalidExtend(u8),
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Data size for scalar load/store instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadStoreSize {
    /// Byte (8-bit) — size field = 0b00
    Byte = 0b00,
    /// Halfword (16-bit) — size field = 0b01
    Half = 0b01,
    /// Word (32-bit) — size field = 0b10
    Word = 0b10,
    /// Doubleword (64-bit) — size field = 0b11
    Double = 0b11,
}

/// Load vs store selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadStoreOp {
    /// Store — opc = 0b00
    Store = 0b00,
    /// Load — opc = 0b01
    Load = 0b01,
}

/// Register extend / index option for register-offset addressing.
///
/// Maps directly to the 3-bit `option` field in the encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegExtend {
    /// UXTW — option = 0b010
    Uxtw = 0b010,
    /// LSL (default, 64-bit) — option = 0b011
    Lsl = 0b011,
    /// SXTW — option = 0b110
    Sxtw = 0b110,
    /// SXTX — option = 0b111
    Sxtx = 0b111,
}

/// Load-pair vs store-pair selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairOp {
    /// Store pair — L = 0
    StorePair = 0,
    /// Load pair — L = 1
    LoadPair = 1,
}

/// Data size for load/store pair instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairSize {
    /// 32-bit (W registers) — opc = 0b00
    W32 = 0b00,
    /// 64-bit (X registers) — opc = 0b10
    X64 = 0b10,
}

/// Addressing mode for load/store pair instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairMode {
    /// Post-index — mode = 0b01
    PostIndex = 0b01,
    /// Signed offset — mode = 0b10
    SignedOffset = 0b10,
    /// Pre-index — mode = 0b11
    PreIndex = 0b11,
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

#[inline]
fn check_reg(reg: u8, max: u8) -> Result<(), EncodeError> {
    if reg > max {
        return Err(EncodeError::RegisterOutOfRange { reg, max });
    }
    Ok(())
}

#[inline]
fn check_imm12(value: u16) -> Result<(), EncodeError> {
    if value > 4095 {
        return Err(EncodeError::Imm12OutOfRange { value });
    }
    Ok(())
}

#[inline]
fn check_imm9(value: i16) -> Result<(), EncodeError> {
    if !(-256..=255).contains(&value) {
        return Err(EncodeError::Imm9OutOfRange { value });
    }
    Ok(())
}

#[inline]
fn check_imm7(value: i8) -> Result<(), EncodeError> {
    if !(-64..=63).contains(&value) {
        return Err(EncodeError::Imm7OutOfRange { value });
    }
    Ok(())
}

#[inline]
fn check_imm21(value: i32) -> Result<(), EncodeError> {
    if !(-1_048_576..=1_048_575).contains(&value) {
        return Err(EncodeError::Imm21OutOfRange { value });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// LDR / STR — unsigned offset
// ---------------------------------------------------------------------------

/// Encode `LDR`/`STR` with unsigned immediate offset.
///
/// ```text
/// size(2) | 111 | V(1) | 01 | opc(2) | imm12(12) | Rn(5) | Rt(5)
/// ```
///
/// * `size` — data width
/// * `v` — `true` for SIMD/FP register, `false` for integer
/// * `op` — load or store
/// * `imm12` — unsigned 12-bit immediate (0..4095), already scaled
/// * `rn` — base register (0..31, 31 = SP)
/// * `rt` — transfer register (0..31)
pub fn encode_ldr_str_unsigned_offset(
    size: LoadStoreSize,
    v: bool,
    op: LoadStoreOp,
    imm12: u16,
    rn: u8,
    rt: u8,
) -> Result<u32, EncodeError> {
    check_reg(rn, 31)?;
    check_reg(rt, 31)?;
    check_imm12(imm12)?;

    let mut inst: u32 = 0;
    inst |= (size as u32) << 30;
    inst |= 0b111 << 27;
    inst |= (v as u32) << 26;
    inst |= 0b01 << 24;
    inst |= (op as u32) << 22;
    inst |= (imm12 as u32) << 10;
    inst |= (rn as u32) << 5;
    inst |= rt as u32;
    Ok(inst)
}

// ---------------------------------------------------------------------------
// LDR / STR — pre-index
// ---------------------------------------------------------------------------

/// Encode `LDR`/`STR` with pre-index addressing.
///
/// ```text
/// size(2) | 111 | V(1) | 00 | opc(2) | 0 | imm9(9) | 11 | Rn(5) | Rt(5)
/// ```
pub fn encode_ldr_str_pre_index(
    size: LoadStoreSize,
    v: bool,
    op: LoadStoreOp,
    imm9: i16,
    rn: u8,
    rt: u8,
) -> Result<u32, EncodeError> {
    check_reg(rn, 31)?;
    check_reg(rt, 31)?;
    check_imm9(imm9)?;

    let imm9_bits = (imm9 as u16 & 0x1FF) as u32;

    let mut inst: u32 = 0;
    inst |= (size as u32) << 30;
    inst |= 0b111 << 27;
    inst |= (v as u32) << 26;
    // bits [25:24] = 00 (unscaled/pre/post family)
    inst |= (op as u32) << 22;
    // bit [21] = 0 (not register offset)
    inst |= imm9_bits << 12;
    inst |= 0b11 << 10; // pre-index marker
    inst |= (rn as u32) << 5;
    inst |= rt as u32;
    Ok(inst)
}

// ---------------------------------------------------------------------------
// LDR / STR — post-index
// ---------------------------------------------------------------------------

/// Encode `LDR`/`STR` with post-index addressing.
///
/// ```text
/// size(2) | 111 | V(1) | 00 | opc(2) | 0 | imm9(9) | 01 | Rn(5) | Rt(5)
/// ```
pub fn encode_ldr_str_post_index(
    size: LoadStoreSize,
    v: bool,
    op: LoadStoreOp,
    imm9: i16,
    rn: u8,
    rt: u8,
) -> Result<u32, EncodeError> {
    check_reg(rn, 31)?;
    check_reg(rt, 31)?;
    check_imm9(imm9)?;

    let imm9_bits = (imm9 as u16 & 0x1FF) as u32;

    let mut inst: u32 = 0;
    inst |= (size as u32) << 30;
    inst |= 0b111 << 27;
    inst |= (v as u32) << 26;
    inst |= (op as u32) << 22;
    inst |= imm9_bits << 12;
    inst |= 0b01 << 10; // post-index marker
    inst |= (rn as u32) << 5;
    inst |= rt as u32;
    Ok(inst)
}

// ---------------------------------------------------------------------------
// LDR / STR — register offset
// ---------------------------------------------------------------------------

/// Encode `LDR`/`STR` with register offset.
///
/// ```text
/// size(2) | 111 | V(1) | 00 | opc(2) | 1 | Rm(5) | option(3) | S(1) | 10 | Rn(5) | Rt(5)
/// ```
///
/// * `extend` — register extend/shift type
/// * `shift` — `true` to shift by access size (S=1), `false` for no shift (S=0)
pub fn encode_ldr_str_register(
    size: LoadStoreSize,
    v: bool,
    op: LoadStoreOp,
    rm: u8,
    extend: RegExtend,
    shift: bool,
    rn: u8,
    rt: u8,
) -> Result<u32, EncodeError> {
    check_reg(rm, 31)?;
    check_reg(rn, 31)?;
    check_reg(rt, 31)?;

    let mut inst: u32 = 0;
    inst |= (size as u32) << 30;
    inst |= 0b111 << 27;
    inst |= (v as u32) << 26;
    inst |= (op as u32) << 22;
    inst |= 1 << 21; // register-offset marker
    inst |= (rm as u32) << 16;
    inst |= (extend as u32) << 13;
    inst |= (shift as u32) << 12;
    inst |= 0b10 << 10;
    inst |= (rn as u32) << 5;
    inst |= rt as u32;
    Ok(inst)
}

// ---------------------------------------------------------------------------
// LDP / STP — generic (offset, pre-index, post-index)
// ---------------------------------------------------------------------------

/// Encode `LDP`/`STP` (load/store pair) with the given addressing mode.
///
/// ```text
/// opc(2) | 101 | V(1) | mode(2) | L(1) | imm7(7) | Rt2(5) | Rn(5) | Rt(5)
/// ```
///
/// * `pair_size` — W32 or X64
/// * `v` — `true` for SIMD/FP registers
/// * `pair_op` — load pair or store pair
/// * `mode` — addressing mode (signed offset, pre-index, or post-index)
/// * `imm7` — signed 7-bit immediate (-64..63), already scaled
/// * `rt2` — second transfer register
/// * `rn` — base register (0..31, 31 = SP)
/// * `rt` — first transfer register
pub fn encode_ldp_stp(
    pair_size: PairSize,
    v: bool,
    pair_op: PairOp,
    mode: PairMode,
    imm7: i8,
    rt2: u8,
    rn: u8,
    rt: u8,
) -> Result<u32, EncodeError> {
    check_reg(rt, 31)?;
    check_reg(rt2, 31)?;
    check_reg(rn, 31)?;
    check_imm7(imm7)?;

    let imm7_bits = (imm7 as u8 & 0x7F) as u32;

    let mut inst: u32 = 0;
    inst |= (pair_size as u32) << 30;
    inst |= 0b101 << 27;
    inst |= (v as u32) << 26;
    inst |= (mode as u32) << 23;
    inst |= (pair_op as u32) << 22;
    inst |= imm7_bits << 15;
    inst |= (rt2 as u32) << 10;
    inst |= (rn as u32) << 5;
    inst |= rt as u32;
    Ok(inst)
}

/// Convenience: encode `LDP`/`STP` with signed offset addressing.
pub fn encode_ldp_stp_offset(
    pair_size: PairSize,
    v: bool,
    pair_op: PairOp,
    imm7: i8,
    rt2: u8,
    rn: u8,
    rt: u8,
) -> Result<u32, EncodeError> {
    encode_ldp_stp(pair_size, v, pair_op, PairMode::SignedOffset, imm7, rt2, rn, rt)
}

/// Convenience: encode `LDP`/`STP` with pre-index addressing.
pub fn encode_ldp_stp_pre_index(
    pair_size: PairSize,
    v: bool,
    pair_op: PairOp,
    imm7: i8,
    rt2: u8,
    rn: u8,
    rt: u8,
) -> Result<u32, EncodeError> {
    encode_ldp_stp(pair_size, v, pair_op, PairMode::PreIndex, imm7, rt2, rn, rt)
}

/// Convenience: encode `LDP`/`STP` with post-index addressing.
pub fn encode_ldp_stp_post_index(
    pair_size: PairSize,
    v: bool,
    pair_op: PairOp,
    imm7: i8,
    rt2: u8,
    rn: u8,
    rt: u8,
) -> Result<u32, EncodeError> {
    encode_ldp_stp(pair_size, v, pair_op, PairMode::PostIndex, imm7, rt2, rn, rt)
}

// ---------------------------------------------------------------------------
// ADRP — PC-relative page address
// ---------------------------------------------------------------------------

/// Encode `ADRP` (form PC-relative page address).
///
/// ```text
/// 1 | immlo(2) | 10000 | immhi(19) | Rd(5)
/// ```
///
/// * `imm21` — signed 21-bit page offset (-1048576..1048575)
/// * `rd` — destination register (0..30)
pub fn encode_adrp(imm21: i32, rd: u8) -> Result<u32, EncodeError> {
    check_reg(rd, 31)?;
    check_imm21(imm21)?;

    let bits = (imm21 as u32) & 0x1F_FFFF; // mask to 21 bits
    let immlo = bits & 0x3;
    let immhi = bits >> 2;

    let mut inst: u32 = 0;
    inst |= 1 << 31; // op = 1 (ADRP)
    inst |= immlo << 29;
    inst |= 0b10000 << 24;
    inst |= immhi << 5;
    inst |= rd as u32;
    Ok(inst)
}

/// Encode `ADR Xd, label` — form PC-relative address.
///
/// ADR uses the same encoding as ADRP but with `op=0` (bit 31 clear).
///
/// ```text
/// 0 | immlo(2) | 10000 | immhi(19) | Rd(5)
/// ```
///
/// * `imm21` — signed 21-bit byte offset (-1048576..1048575)
/// * `rd` — destination register (0..30)
pub fn encode_adr(imm21: i32, rd: u8) -> Result<u32, EncodeError> {
    check_reg(rd, 31)?;
    check_imm21(imm21)?;

    let bits = (imm21 as u32) & 0x1F_FFFF; // mask to 21 bits
    let immlo = bits & 0x3;
    let immhi = bits >> 2;

    let mut inst: u32 = 0;
    // op = 0 (ADR, not ADRP) — bit 31 stays clear
    inst |= immlo << 29;
    inst |= 0b10000 << 24;
    inst |= immhi << 5;
    inst |= rd as u32;
    Ok(inst)
}

/// Encode `LDRSW Xt, [Xn, Xm, LSL #2]` — load signed word with register offset.
///
/// ARM ARM encoding: Load/store register (register offset)
///
/// ```text
/// size(2) | 111 | V(1) | 00 | opc(2) | 1 | Rm(5) | option(3) | S(1) | 10 | Rn(5) | Rt(5)
/// ```
///
/// For LDRSW: size=10, V=0, opc=10, option=011 (LSL), S=1 (shift by 2)
///
/// * `rm` — index register (0..30)
/// * `rn` — base register (0..31, 31 = SP)
/// * `rt` — destination register (0..30)
pub fn encode_ldrsw_register(rm: u8, rn: u8, rt: u8) -> Result<u32, EncodeError> {
    check_reg(rm, 31)?;
    check_reg(rn, 31)?;
    check_reg(rt, 31)?;

    let mut inst: u32 = 0;
    inst |= 0b10 << 30;           // size = 10 (Word)
    inst |= 0b111 << 27;          // load/store register class
    // V = 0 (bit 26 clear — GPR, not SIMD)
    inst |= 0b10 << 22;           // opc = 10 (LDRSW)
    inst |= 1 << 21;              // register-offset marker
    inst |= (rm as u32) << 16;
    inst |= 0b011 << 13;          // option = LSL
    inst |= 1 << 12;              // S = 1 (shift by access size = 2)
    inst |= 0b10 << 10;
    inst |= (rn as u32) << 5;
    inst |= rt as u32;
    Ok(inst)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // === LDR/STR unsigned offset ===

    #[test]
    fn test_ldr_word_unsigned_offset() {
        // LDR W3, [X5, #16] (imm12=16, already scaled)
        let enc = encode_ldr_str_unsigned_offset(
            LoadStoreSize::Word, false, LoadStoreOp::Load, 16, 5, 3,
        ).unwrap();
        assert_eq!(enc, 0xB940_40A3);
    }

    #[test]
    fn test_str_double_unsigned_offset_zero() {
        // STR X0, [SP, #0]
        let enc = encode_ldr_str_unsigned_offset(
            LoadStoreSize::Double, false, LoadStoreOp::Store, 0, 31, 0,
        ).unwrap();
        assert_eq!(enc, 0xF900_03E0);
    }

    #[test]
    fn test_ldr_double_unsigned_offset() {
        // LDR X0, [X1, #1] (imm12=1)
        let enc = encode_ldr_str_unsigned_offset(
            LoadStoreSize::Double, false, LoadStoreOp::Load, 1, 1, 0,
        ).unwrap();
        assert_eq!(enc, 0xF940_0420);
    }

    #[test]
    fn test_ldr_simd_unsigned_offset() {
        // LDR Q31, [X31, #0] — V=1, size=Double, opc=01
        let enc = encode_ldr_str_unsigned_offset(
            LoadStoreSize::Double, true, LoadStoreOp::Store, 0, 31, 31,
        ).unwrap();
        assert_eq!(enc, 0xFD00_03FF);
    }

    #[test]
    fn test_ldr_unsigned_offset_imm12_max() {
        // LDR X30, [X31, #4095]
        let enc = encode_ldr_str_unsigned_offset(
            LoadStoreSize::Double, false, LoadStoreOp::Load, 4095, 31, 30,
        ).unwrap();
        assert_eq!(enc, 0xF97F_FFFE);
    }

    #[test]
    fn test_unsigned_offset_imm12_overflow() {
        let err = encode_ldr_str_unsigned_offset(
            LoadStoreSize::Byte, false, LoadStoreOp::Load, 4096, 0, 0,
        );
        assert!(matches!(err, Err(EncodeError::Imm12OutOfRange { value: 4096 })));
    }

    #[test]
    fn test_unsigned_offset_reg_overflow() {
        let err = encode_ldr_str_unsigned_offset(
            LoadStoreSize::Word, false, LoadStoreOp::Load, 0, 32, 0,
        );
        assert!(matches!(err, Err(EncodeError::RegisterOutOfRange { reg: 32, max: 31 })));
    }

    // === LDR/STR pre-index ===

    #[test]
    fn test_str_double_pre_index_neg16() {
        // STR X2, [SP, #-16]!
        let enc = encode_ldr_str_pre_index(
            LoadStoreSize::Double, false, LoadStoreOp::Store, -16, 31, 2,
        ).unwrap();
        assert_eq!(enc, 0xF81F_0FE2);
    }

    #[test]
    fn test_pre_index_imm9_min() {
        // LDR H0, [X31, #-256]!  size=Half, V=0, opc=LDR
        let enc = encode_ldr_str_pre_index(
            LoadStoreSize::Half, false, LoadStoreOp::Load, -256, 31, 0,
        ).unwrap();
        assert_eq!(enc, 0x7850_0FE0);
    }

    #[test]
    fn test_pre_index_imm9_max() {
        // LDR H0, [X31, #255]!
        let enc = encode_ldr_str_pre_index(
            LoadStoreSize::Half, false, LoadStoreOp::Load, 255, 31, 0,
        ).unwrap();
        assert_eq!(enc, 0x784F_FFE0);
    }

    #[test]
    fn test_pre_index_imm9_overflow() {
        let err = encode_ldr_str_pre_index(
            LoadStoreSize::Byte, false, LoadStoreOp::Load, 256, 0, 0,
        );
        assert!(matches!(err, Err(EncodeError::Imm9OutOfRange { value: 256 })));
    }

    #[test]
    fn test_pre_index_imm9_underflow() {
        let err = encode_ldr_str_pre_index(
            LoadStoreSize::Byte, false, LoadStoreOp::Load, -257, 0, 0,
        );
        assert!(matches!(err, Err(EncodeError::Imm9OutOfRange { value: -257 })));
    }

    // === LDR/STR post-index ===

    #[test]
    fn test_ldr_double_post_index() {
        // LDR X30, [SP], #16
        let enc = encode_ldr_str_post_index(
            LoadStoreSize::Double, false, LoadStoreOp::Load, 16, 31, 30,
        ).unwrap();
        assert_eq!(enc, 0xF841_07FE);
    }

    #[test]
    fn test_ldr_byte_post_index_max() {
        // LDR B30, [X0], #255
        let enc = encode_ldr_str_post_index(
            LoadStoreSize::Byte, false, LoadStoreOp::Load, 255, 0, 30,
        ).unwrap();
        assert_eq!(enc, 0x384F_F41E);
    }

    // === LDR/STR register offset ===

    #[test]
    fn test_ldr_double_register_lsl() {
        // LDR X7, [X6, X4, LSL #3]
        let enc = encode_ldr_str_register(
            LoadStoreSize::Double, false, LoadStoreOp::Load,
            4, RegExtend::Lsl, true, 6, 7,
        ).unwrap();
        assert_eq!(enc, 0xF864_78C7);
    }

    #[test]
    fn test_str_word_register_sxtw() {
        // STR W0, [X1, W2, SXTW]
        let enc = encode_ldr_str_register(
            LoadStoreSize::Word, false, LoadStoreOp::Store,
            2, RegExtend::Sxtw, false, 1, 0,
        ).unwrap();
        // size=10, V=0, opc=00, Rm=2, option=110, S=0, Rn=1, Rt=0
        let expected = ((0b10 << 30) | (0b111 << 27)) | (1 << 21)
            | (2 << 16) | (0b110 << 13) | (0b10 << 10) | (1 << 5);
        assert_eq!(enc, expected);
    }

    #[test]
    fn test_register_offset_reg_overflow() {
        let err = encode_ldr_str_register(
            LoadStoreSize::Byte, false, LoadStoreOp::Load,
            32, RegExtend::Lsl, false, 0, 0,
        );
        assert!(matches!(err, Err(EncodeError::RegisterOutOfRange { reg: 32, .. })));
    }

    // === LDP/STP ===

    #[test]
    fn test_stp_x64_pre_index() {
        // STP X29, X30, [SP, #-16]! — classic prologue
        // opc=10, V=0, mode=11(pre), L=0, imm7=-2 (=-16/8), Rt2=30, Rn=31, Rt=29
        let enc = encode_ldp_stp_pre_index(
            PairSize::X64, false, PairOp::StorePair, -2, 30, 31, 29,
        ).unwrap();
        assert_eq!(enc, 0xA9BF_7BFD);
    }

    #[test]
    fn test_ldp_x64_post_index() {
        // LDP X29, X30, [SP], #16
        // opc=10, V=0, mode=01(post), L=1, imm7=2, Rt2=30, Rn=31, Rt=29
        let enc = encode_ldp_stp_post_index(
            PairSize::X64, false, PairOp::LoadPair, 2, 30, 31, 29,
        ).unwrap();
        assert_eq!(enc, 0xA8C1_7BFD);
    }

    #[test]
    fn test_ldp_x64_signed_offset() {
        // LDP X9, X10, [X31, #-64] (imm7=-8 for 64-bit)
        let enc = encode_ldp_stp_offset(
            PairSize::X64, false, PairOp::LoadPair, -8, 10, 31, 9,
        ).unwrap();
        assert_eq!(enc, 0xA97C_2BE9);
    }

    #[test]
    fn test_stp_w32_pre_index() {
        // STP W1, W2, [X3, #252] (imm7=63 for 32-bit)
        let enc = encode_ldp_stp_pre_index(
            PairSize::W32, false, PairOp::StorePair, 63, 2, 3, 1,
        ).unwrap();
        assert_eq!(enc, 0x299F_8861);
    }

    #[test]
    fn test_ldp_imm7_boundary_min() {
        // imm7 = -64 (minimum)
        let enc = encode_ldp_stp_offset(
            PairSize::W32, false, PairOp::StorePair, -64, 1, 31, 0,
        ).unwrap();
        assert_eq!(enc, 0x2920_07E0);
    }

    #[test]
    fn test_ldp_imm7_boundary_max_pre() {
        // imm7 = 63 (maximum), pre-index, load
        let enc = encode_ldp_stp_pre_index(
            PairSize::X64, false, PairOp::LoadPair, 63, 30, 31, 29,
        ).unwrap();
        assert_eq!(enc, 0xA9DF_FBFD);
    }

    #[test]
    fn test_pair_imm7_overflow() {
        let err = encode_ldp_stp_offset(
            PairSize::X64, false, PairOp::LoadPair, 64, 0, 0, 0,
        );
        assert!(matches!(err, Err(EncodeError::Imm7OutOfRange { .. })));
    }

    // === ADRP ===

    #[test]
    fn test_adrp_positive_1() {
        // ADRP X5, #1
        let enc = encode_adrp(1, 5).unwrap();
        assert_eq!(enc, 0xB000_0005);
    }

    #[test]
    fn test_adrp_zero() {
        // ADRP X0, #0
        let enc = encode_adrp(0, 0).unwrap();
        assert_eq!(enc, 0x9000_0000);
    }

    #[test]
    fn test_adrp_negative_1() {
        // ADRP X5, #-1
        let enc = encode_adrp(-1, 5).unwrap();
        assert_eq!(enc, 0xF0FF_FFE5);
    }

    #[test]
    fn test_adrp_max() {
        // ADRP X30, #1048575 (max positive)
        let enc = encode_adrp(1_048_575, 30).unwrap();
        assert_eq!(enc, 0xF07F_FFFE);
    }

    #[test]
    fn test_adrp_min() {
        // ADRP X30, #-1048576 (min negative)
        let enc = encode_adrp(-1_048_576, 30).unwrap();
        assert_eq!(enc, 0x9080_001E);
    }

    #[test]
    fn test_adrp_imm21_overflow() {
        let err = encode_adrp(1_048_576, 0);
        assert!(matches!(err, Err(EncodeError::Imm21OutOfRange { .. })));
    }

    #[test]
    fn test_adrp_imm21_underflow() {
        let err = encode_adrp(-1_048_577, 0);
        assert!(matches!(err, Err(EncodeError::Imm21OutOfRange { .. })));
    }

    // === Mixed / edge-case coverage ===

    #[test]
    fn test_str_byte_unsigned_offset() {
        // STR B0, [X0, #0]
        let enc = encode_ldr_str_unsigned_offset(
            LoadStoreSize::Byte, false, LoadStoreOp::Store, 0, 0, 0,
        ).unwrap();
        // size=00, V=0, opc=00, imm12=0, Rn=0, Rt=0
        // 00 111 0 01 00 000000000000 00000 00000
        let expected = 0b00_111_0_01_00_000000000000_00000_00000;
        assert_eq!(enc, expected);
    }

    #[test]
    fn test_all_sizes_load() {
        // Verify size field encoding for all widths
        for (size, expected_bits) in [
            (LoadStoreSize::Byte, 0b00u32),
            (LoadStoreSize::Half, 0b01),
            (LoadStoreSize::Word, 0b10),
            (LoadStoreSize::Double, 0b11),
        ] {
            let enc = encode_ldr_str_unsigned_offset(size, false, LoadStoreOp::Load, 0, 0, 0).unwrap();
            assert_eq!(enc >> 30, expected_bits, "size field mismatch for {:?}", size);
        }
    }

    #[test]
    fn test_register_offset_all_extends() {
        for (extend, expected_option) in [
            (RegExtend::Uxtw, 0b010u32),
            (RegExtend::Lsl, 0b011),
            (RegExtend::Sxtw, 0b110),
            (RegExtend::Sxtx, 0b111),
        ] {
            let enc = encode_ldr_str_register(
                LoadStoreSize::Double, false, LoadStoreOp::Load,
                0, extend, false, 0, 0,
            ).unwrap();
            let option_field = (enc >> 13) & 0b111;
            assert_eq!(option_field, expected_option, "option mismatch for {:?}", extend);
        }
    }
}
