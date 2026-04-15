// llvm2-codegen integration test: AArch64 instruction encoding
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Comprehensive integration tests for the AArch64 instruction encoders.
// Verifies encodings against known-good values from the ARM Architecture
// Reference Manual (DDI 0487), tests boundary conditions, bit-field extraction,
// SP/XZR encoding traps, and Mach-O round-trip via otool -tv.

use llvm2_codegen::aarch64::encoding_fp::*;
use llvm2_codegen::aarch64::encoding_mem::*;
use llvm2_codegen::macho::MachOWriter;

use std::io::Write;
use std::process::Command;

// ===========================================================================
// Helpers
// ===========================================================================

/// Write bytes to a temp file and return the path.
fn write_temp_o(bytes: &[u8], name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("llvm2_aarch64_test_{}.o", name));
    let mut f = std::fs::File::create(&path).expect("failed to create temp file");
    f.write_all(bytes).expect("failed to write temp file");
    path
}

/// Run `otool -tv` on a .o file and return stdout (text disassembly).
fn run_otool_tv(path: &std::path::Path) -> String {
    let output = Command::new("otool")
        .args(["-tv", path.to_str().unwrap()])
        .output()
        .expect("failed to run otool");
    assert!(
        output.status.success(),
        "otool -tv failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).to_string()
}

// ===========================================================================
// 1. Encoding Verification Tests — LDR/STR Unsigned Offset
// ===========================================================================

// Encoding format: size(2) | 111 | V(1) | 01 | opc(2) | imm12(12) | Rn(5) | Rt(5)

#[test]
fn test_ldr_x0_x1_zero_offset() {
    // LDR X0, [X1]  =>  size=11, V=0, opc=01, imm12=0, Rn=1, Rt=0
    // 11_111_0_01_01_000000000000_00001_00000
    let enc = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Double, false, LoadStoreOp::Load, 0, 1, 0,
    ).unwrap();
    assert_eq!(enc, 0xF940_0020);
}

#[test]
fn test_ldr_w3_sp_4() {
    // LDR W3, [SP, #4] (scaled imm12=4 means raw 4 in our API, since "already scaled")
    // size=10, V=0, opc=01, imm12=4, Rn=31(SP), Rt=3
    // 10_111_0_01_01_000000000100_11111_00011
    let enc = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Word, false, LoadStoreOp::Load, 4, 31, 3,
    ).unwrap();
    // bits: 10 111 0 01 01 000000000100 11111 00011
    //     = 1011_1001_0100_0000_0001_0011_1110_0011
    assert_eq!(enc, 0xB940_13E3);
}

#[test]
fn test_str_x29_x28_zero() {
    // STR X29, [X28, #0]  =>  size=11, V=0, opc=00, imm12=0, Rn=28, Rt=29
    let enc = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Double, false, LoadStoreOp::Store, 0, 28, 29,
    ).unwrap();
    assert_eq!(enc, 0xF900_039D);
}

#[test]
fn test_ldrb_w0_x1() {
    // LDRB W0, [X1, #0]  =>  size=00, V=0, opc=01, imm12=0, Rn=1, Rt=0
    let enc = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Byte, false, LoadStoreOp::Load, 0, 1, 0,
    ).unwrap();
    assert_eq!(enc, 0x3940_0020);
}

#[test]
fn test_ldrh_w0_x1() {
    // LDRH W0, [X1, #0]  =>  size=01, V=0, opc=01, imm12=0, Rn=1, Rt=0
    let enc = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Half, false, LoadStoreOp::Load, 0, 1, 0,
    ).unwrap();
    assert_eq!(enc, 0x7940_0020);
}

#[test]
fn test_strb_w0_x0() {
    // STRB W0, [X0, #0]  =>  size=00, V=0, opc=00, imm12=0, Rn=0, Rt=0
    let enc = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Byte, false, LoadStoreOp::Store, 0, 0, 0,
    ).unwrap();
    assert_eq!(enc, 0x3900_0000);
}

#[test]
fn test_str_x0_sp_zero() {
    // STR X0, [SP, #0]  =>  size=11, V=0, opc=00, imm12=0, Rn=31, Rt=0
    let enc = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Double, false, LoadStoreOp::Store, 0, 31, 0,
    ).unwrap();
    assert_eq!(enc, 0xF900_03E0);
}

#[test]
fn test_ldr_x30_sp_max_imm12() {
    // LDR X30, [SP, #4095]
    let enc = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Double, false, LoadStoreOp::Load, 4095, 31, 30,
    ).unwrap();
    assert_eq!(enc, 0xF97F_FFFE);
}

#[test]
fn test_ldr_unsigned_v_bit() {
    // STR with V=1, size=Double, opc=Store => SIMD store
    let enc = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Double, true, LoadStoreOp::Store, 0, 31, 31,
    ).unwrap();
    assert_eq!(enc, 0xFD00_03FF);
}

// ===========================================================================
// 1b. Encoding Verification — LDR/STR Pre-Index
// ===========================================================================

// Format: size(2) | 111 | V(1) | 00 | opc(2) | 0 | imm9(9) | 11 | Rn(5) | Rt(5)

#[test]
fn test_str_x29_sp_pre_neg16() {
    // STR X29, [SP, #-16]!
    // size=11, V=0, opc=00, imm9=-16 (0x1F0), Rn=31, Rt=29
    let enc = encode_ldr_str_pre_index(
        LoadStoreSize::Double, false, LoadStoreOp::Store, -16, 31, 29,
    ).unwrap();
    assert_eq!(enc, 0xF81F_0FFD);
}

#[test]
fn test_ldr_x30_sp_pre_pos16() {
    // LDR X30, [SP, #16]!
    // size=11, V=0, opc=01, imm9=16(0x010), Rn=31, Rt=30
    let enc = encode_ldr_str_pre_index(
        LoadStoreSize::Double, false, LoadStoreOp::Load, 16, 31, 30,
    ).unwrap();
    assert_eq!(enc, 0xF841_0FFE);
}

#[test]
fn test_str_x0_x1_pre_neg1() {
    // STR X0, [X1, #-1]!
    // imm9=-1 => 0x1FF
    let enc = encode_ldr_str_pre_index(
        LoadStoreSize::Double, false, LoadStoreOp::Store, -1, 1, 0,
    ).unwrap();
    assert_eq!(enc, 0xF81F_FC20);
}

#[test]
fn test_pre_index_max_imm9() {
    // STR X0, [X1, #255]!
    let enc = encode_ldr_str_pre_index(
        LoadStoreSize::Double, false, LoadStoreOp::Store, 255, 1, 0,
    ).unwrap();
    // imm9=255 (0x0FF)
    assert_eq!(enc, 0xF80F_FC20);
}

#[test]
fn test_pre_index_min_imm9() {
    // STR X0, [X1, #-256]!
    let enc = encode_ldr_str_pre_index(
        LoadStoreSize::Double, false, LoadStoreOp::Store, -256, 1, 0,
    ).unwrap();
    // imm9=-256 => 0x100
    assert_eq!(enc, 0xF810_0C20);
}

// ===========================================================================
// 1c. Encoding Verification — LDR/STR Post-Index
// ===========================================================================

// Format: size(2) | 111 | V(1) | 00 | opc(2) | 0 | imm9(9) | 01 | Rn(5) | Rt(5)

#[test]
fn test_ldr_x0_x1_post_8() {
    // LDR X0, [X1], #8
    let enc = encode_ldr_str_post_index(
        LoadStoreSize::Double, false, LoadStoreOp::Load, 8, 1, 0,
    ).unwrap();
    // imm9=8 (0x008), bits[11:10]=01
    assert_eq!(enc, 0xF840_8420);
}

#[test]
fn test_str_x0_sp_post_neg16() {
    // STR X0, [SP], #-16
    let enc = encode_ldr_str_post_index(
        LoadStoreSize::Double, false, LoadStoreOp::Store, -16, 31, 0,
    ).unwrap();
    // imm9=-16 => 0x1F0
    assert_eq!(enc, 0xF81F_07E0);
}

#[test]
fn test_ldr_w0_x1_post_max() {
    // LDR W0, [X1], #255
    let enc = encode_ldr_str_post_index(
        LoadStoreSize::Word, false, LoadStoreOp::Load, 255, 1, 0,
    ).unwrap();
    assert_eq!(enc, 0xB84F_F420);
}

#[test]
fn test_ldr_w0_x1_post_min() {
    // LDR W0, [X1], #-256
    let enc = encode_ldr_str_post_index(
        LoadStoreSize::Word, false, LoadStoreOp::Load, -256, 1, 0,
    ).unwrap();
    assert_eq!(enc, 0xB850_0420);
}

// ===========================================================================
// 1d. Encoding Verification — LDR/STR Register Offset
// ===========================================================================

// Format: size(2) | 111 | V(1) | 00 | opc(2) | 1 | Rm(5) | option(3) | S(1) | 10 | Rn(5) | Rt(5)

#[test]
fn test_ldr_x0_x1_x2_lsl_no_shift() {
    // LDR X0, [X1, X2] (LSL, S=0 — no shift)
    let enc = encode_ldr_str_register(
        LoadStoreSize::Double, false, LoadStoreOp::Load,
        2, RegExtend::Lsl, false, 1, 0,
    ).unwrap();
    // size=11, V=0, opc=01, Rm=2, option=011, S=0, Rn=1, Rt=0
    let expected = (0b11u32 << 30) | (0b111 << 27) | (0b01 << 22) | (1 << 21)
        | (2 << 16) | (0b011 << 13) | (0b10 << 10) | (1 << 5);
    assert_eq!(enc, expected);
}

#[test]
fn test_ldr_x0_x1_x2_lsl_shift() {
    // LDR X0, [X1, X2, LSL #3]
    let enc = encode_ldr_str_register(
        LoadStoreSize::Double, false, LoadStoreOp::Load,
        2, RegExtend::Lsl, true, 1, 0,
    ).unwrap();
    let expected = (0b11u32 << 30) | (0b111 << 27) | (0b01 << 22) | (1 << 21)
        | (2 << 16) | (0b011 << 13) | (1 << 12) | (0b10 << 10) | (1 << 5);
    assert_eq!(enc, expected);
}

#[test]
fn test_ldr_w0_x1_w2_sxtw() {
    // LDR W0, [X1, W2, SXTW]
    let enc = encode_ldr_str_register(
        LoadStoreSize::Word, false, LoadStoreOp::Load,
        2, RegExtend::Sxtw, false, 1, 0,
    ).unwrap();
    let expected = (0b10u32 << 30) | (0b111 << 27) | (0b01 << 22) | (1 << 21)
        | (2 << 16) | (0b110 << 13) | (0b10 << 10) | (1 << 5);
    assert_eq!(enc, expected);
}

#[test]
fn test_ldr_x0_x1_w2_uxtw() {
    // LDR X0, [X1, W2, UXTW]
    let enc = encode_ldr_str_register(
        LoadStoreSize::Double, false, LoadStoreOp::Load,
        2, RegExtend::Uxtw, false, 1, 0,
    ).unwrap();
    let expected = (0b11u32 << 30) | (0b111 << 27) | (0b01 << 22) | (1 << 21)
        | (2 << 16) | (0b010 << 13) | (0b10 << 10) | (1 << 5);
    assert_eq!(enc, expected);
}

#[test]
fn test_ldr_x7_x6_x4_lsl3() {
    // LDR X7, [X6, X4, LSL #3] (from existing unit test)
    let enc = encode_ldr_str_register(
        LoadStoreSize::Double, false, LoadStoreOp::Load,
        4, RegExtend::Lsl, true, 6, 7,
    ).unwrap();
    assert_eq!(enc, 0xF864_78C7);
}

// ===========================================================================
// 1e. Encoding Verification — LDP/STP
// ===========================================================================

// Format: opc(2) | 101 | V(1) | mode(2) | L(1) | imm7(7) | Rt2(5) | Rn(5) | Rt(5)

#[test]
fn test_stp_x29_x30_sp_pre_neg16() {
    // STP X29, X30, [SP, #-16]! (classic prologue)
    // opc=10, V=0, mode=11(pre), L=0, imm7=-2 (=-16/8), Rt2=30, Rn=31, Rt=29
    let enc = encode_ldp_stp_pre_index(
        PairSize::X64, false, PairOp::StorePair, -2, 30, 31, 29,
    ).unwrap();
    assert_eq!(enc, 0xA9BF_7BFD);
}

#[test]
fn test_ldp_x29_x30_sp_post_16() {
    // LDP X29, X30, [SP], #16 (classic epilogue)
    // opc=10, V=0, mode=01(post), L=1, imm7=2
    let enc = encode_ldp_stp_post_index(
        PairSize::X64, false, PairOp::LoadPair, 2, 30, 31, 29,
    ).unwrap();
    assert_eq!(enc, 0xA8C1_7BFD);
}

#[test]
fn test_stp_w0_w1_sp_pre_neg8() {
    // STP W0, W1, [SP, #-8]! (32-bit pair)
    // opc=00, V=0, mode=11(pre), L=0, imm7=-2 (=-8/4), Rt2=1, Rn=31, Rt=0
    let enc = encode_ldp_stp_pre_index(
        PairSize::W32, false, PairOp::StorePair, -2, 1, 31, 0,
    ).unwrap();
    // 00 101 0 11 0 1111110 00001 11111 00000
    let expected = (0b101 << 27) | (0b11 << 23)
        | (((-2i8 as u8 & 0x7F) as u32) << 15) | (1 << 10) | (31 << 5);
    assert_eq!(enc, expected);
}

#[test]
fn test_ldp_x0_x1_x2_zero_offset() {
    // LDP X0, X1, [X2, #0]
    let enc = encode_ldp_stp_offset(
        PairSize::X64, false, PairOp::LoadPair, 0, 1, 2, 0,
    ).unwrap();
    // opc=10, V=0, mode=10(signed), L=1, imm7=0, Rt2=1, Rn=2, Rt=0
    let expected = (0b10u32 << 30) | (0b101 << 27) | (0b10 << 23) | (1 << 22)
        | (1 << 10) | (2 << 5);
    assert_eq!(enc, expected);
}

#[test]
fn test_stp_imm7_min() {
    // STP W0, W1, [SP, <imm7=-64>]
    let enc = encode_ldp_stp_offset(
        PairSize::W32, false, PairOp::StorePair, -64, 1, 31, 0,
    ).unwrap();
    assert_eq!(enc, 0x2920_07E0);
}

#[test]
fn test_ldp_imm7_max() {
    // LDP X29, X30, [SP, <imm7=63>]!  (pre-index)
    let enc = encode_ldp_stp_pre_index(
        PairSize::X64, false, PairOp::LoadPair, 63, 30, 31, 29,
    ).unwrap();
    assert_eq!(enc, 0xA9DF_FBFD);
}

// ===========================================================================
// 1f. Encoding Verification — ADRP
// ===========================================================================

// Format: 1 | immlo(2) | 10000 | immhi(19) | Rd(5)

#[test]
fn test_adrp_x0_zero() {
    let enc = encode_adrp(0, 0).unwrap();
    assert_eq!(enc, 0x9000_0000);
}

#[test]
fn test_adrp_x1_one() {
    let enc = encode_adrp(1, 1).unwrap();
    // imm21=1 => immlo=01, immhi=0
    // 1_01_10000_0000000000000000000_00001
    assert_eq!(enc, 0xB000_0001);
}

#[test]
fn test_adrp_x0_neg1() {
    let enc = encode_adrp(-1, 0).unwrap();
    // imm21=-1 => bits=0x1FFFFF, immlo=11, immhi=0x7FFFF
    assert_eq!(enc, 0xF0FF_FFE0);
}

#[test]
fn test_adrp_x30_max() {
    let enc = encode_adrp(1_048_575, 30).unwrap();
    assert_eq!(enc, 0xF07F_FFFE);
}

#[test]
fn test_adrp_x30_min() {
    let enc = encode_adrp(-1_048_576, 30).unwrap();
    assert_eq!(enc, 0x9080_001E);
}

// ===========================================================================
// 1g. Encoding Verification — FP Arithmetic (2-source)
// ===========================================================================

// Format: 0 | 00 | 11110 | ftype(2) | 1 | Rm(5) | opcode(4) | 10 | Rn(5) | Rd(5)

#[test]
fn test_fadd_s0_s1_s2() {
    let enc = encode_fp_arith(FpSize::Single, FpArithOp::Add, 2, 1, 0).unwrap();
    assert_eq!(enc, 0x1E22_2820);
}

#[test]
fn test_fadd_d0_d1_d2() {
    let enc = encode_fp_arith(FpSize::Double, FpArithOp::Add, 2, 1, 0).unwrap();
    assert_eq!(enc, 0x1E62_2820);
}

#[test]
fn test_fsub_d3_d4_d5() {
    let enc = encode_fp_arith(FpSize::Double, FpArithOp::Sub, 5, 4, 3).unwrap();
    assert_eq!(enc, 0x1E65_3883);
}

#[test]
fn test_fmul_s10_s11_s12() {
    let enc = encode_fp_arith(FpSize::Single, FpArithOp::Mul, 12, 11, 10).unwrap();
    assert_eq!(enc, 0x1E2C_096A);
}

#[test]
fn test_fdiv_d0_d1_d2() {
    let enc = encode_fp_arith(FpSize::Double, FpArithOp::Div, 2, 1, 0).unwrap();
    assert_eq!(enc, 0x1E62_1820);
}

#[test]
fn test_fadd_d31_d31_d31() {
    // Test max register numbers
    let enc = encode_fp_arith(FpSize::Double, FpArithOp::Add, 31, 31, 31).unwrap();
    let expected = (0b11110u32 << 24) | (0b01 << 22) | (1 << 21)
        | (31 << 16) | (0b0010 << 12) | (0b10 << 10) | (31 << 5) | 31;
    assert_eq!(enc, expected);
}

#[test]
fn test_fsub_s0_s0_s0() {
    // Test min register numbers
    let enc = encode_fp_arith(FpSize::Single, FpArithOp::Sub, 0, 0, 0).unwrap();
    let expected = (0b11110u32 << 24) | (1 << 21) | (0b0011 << 12) | (0b10 << 10);
    assert_eq!(enc, expected);
}

// ===========================================================================
// 1h. Encoding Verification — FCMP
// ===========================================================================

// Format: 0 | 00 | 11110 | ftype(2) | 1 | Rm(5) | 00 | 1000 | Rn(5) | opc(5)

#[test]
fn test_fcmp_s0_s1() {
    let enc = encode_fcmp(FpSize::Single, FpCmpOp::Cmp, 1, 0).unwrap();
    // Rn=0, Rm=1
    let expected = (0b11110u32 << 24) | (1 << 21)
        | (1 << 16) | (0b1000 << 10);
    assert_eq!(enc, expected);
}

#[test]
fn test_fcmp_d0_d1() {
    let enc = encode_fcmp(FpSize::Double, FpCmpOp::Cmp, 1, 0).unwrap();
    let expected = (0b11110u32 << 24) | (0b01 << 22) | (1 << 21)
        | (1 << 16) | (0b1000 << 10);
    assert_eq!(enc, expected);
}

#[test]
fn test_fcmp_s0_zero() {
    let enc = encode_fcmp(FpSize::Single, FpCmpOp::CmpZero, 0, 0).unwrap();
    let expected = (0b11110u32 << 24) | (1 << 21) | (0b1000 << 10) | 0b01000;
    assert_eq!(enc, expected);
}

#[test]
fn test_fcmp_d0_zero() {
    let enc = encode_fcmp(FpSize::Double, FpCmpOp::CmpZero, 0, 0).unwrap();
    assert_eq!(enc, 0x1E60_2008);
}

#[test]
fn test_fcmp_d1_zero() {
    let enc = encode_fcmp(FpSize::Double, FpCmpOp::CmpZero, 0, 1).unwrap();
    assert_eq!(enc, 0x1E60_2028);
}

#[test]
fn test_fcmpe_d0_zero() {
    let enc = encode_fcmp(FpSize::Double, FpCmpOp::CmpeZero, 0, 0).unwrap();
    let expected = (0b11110u32 << 24) | (0b01 << 22) | (1 << 21)
        | (0b1000 << 10) | 0b11000;
    assert_eq!(enc, expected);
}

// ===========================================================================
// 1i. Encoding Verification — FP Conversions
// ===========================================================================

// Format: sf(1) | 00 | 11110 | ftype(2) | 1 | rmode(2) | opcode(3) | 000000 | Rn(5) | Rd(5)

#[test]
fn test_fcvtzs_w0_s1() {
    let enc = encode_fp_int_conv(false, FpSize::Single, FpConvOp::FcvtzsToInt, 1, 0).unwrap();
    assert_eq!(enc, 0x1E38_0020);
}

#[test]
fn test_fcvtzs_x0_d1() {
    let enc = encode_fp_int_conv(true, FpSize::Double, FpConvOp::FcvtzsToInt, 1, 0).unwrap();
    // sf=1, ftype=01, rmode=11, opcode=000
    assert_eq!(enc, 0x9E78_0020);
}

#[test]
fn test_scvtf_s0_w1() {
    let enc = encode_fp_int_conv(false, FpSize::Single, FpConvOp::ScvtfToFp, 1, 0).unwrap();
    assert_eq!(enc, 0x1E22_0020);
}

#[test]
fn test_scvtf_d0_x1() {
    let enc = encode_fp_int_conv(true, FpSize::Double, FpConvOp::ScvtfToFp, 1, 0).unwrap();
    assert_eq!(enc, 0x9E62_0020);
}

#[test]
fn test_fmov_s0_w1() {
    // FMOV S0, W1: sf=0, ftype=00, rmode=00, opcode=111
    let enc = encode_fp_int_conv(false, FpSize::Single, FpConvOp::FmovToFp, 1, 0).unwrap();
    let expected = (0b11110u32 << 24) | (1 << 21) | (0b111 << 16) | (1 << 5);
    assert_eq!(enc, expected);
}

#[test]
fn test_fmov_w0_s1() {
    // FMOV W0, S1: sf=0, ftype=00, rmode=00, opcode=110
    let enc = encode_fp_int_conv(false, FpSize::Single, FpConvOp::FmovToGp, 1, 0).unwrap();
    let expected = (0b11110u32 << 24) | (1 << 21) | (0b110 << 16) | (1 << 5);
    assert_eq!(enc, expected);
}

// ===========================================================================
// 1j. Encoding Verification — FP Unary (1-source)
// ===========================================================================

// Format: 0 | 00 | 11110 | ftype(2) | 1 | 0000 | opcode(2) | 10000 | Rn(5) | Rd(5)

#[test]
fn test_fneg_s0_s1() {
    let enc = encode_fp_unary(FpSize::Single, FpUnaryOp::Fneg, 1, 0).unwrap();
    assert_eq!(enc, 0x1E21_4020);
}

#[test]
fn test_fneg_d0_d1() {
    let enc = encode_fp_unary(FpSize::Double, FpUnaryOp::Fneg, 1, 0).unwrap();
    assert_eq!(enc, 0x1E61_4020);
}

#[test]
fn test_fabs_d3_d4() {
    let enc = encode_fp_unary(FpSize::Double, FpUnaryOp::Fabs, 4, 3).unwrap();
    assert_eq!(enc, 0x1E60_C083);
}

#[test]
fn test_fabs_s0_s1() {
    let enc = encode_fp_unary(FpSize::Single, FpUnaryOp::Fabs, 1, 0).unwrap();
    assert_eq!(enc, 0x1E20_C020);
}

#[test]
fn test_fsqrt_d0_d1() {
    let enc = encode_fp_unary(FpSize::Double, FpUnaryOp::Fsqrt, 1, 0).unwrap();
    assert_eq!(enc, 0x1E61_C020);
}

#[test]
fn test_fsqrt_s0_s1() {
    let enc = encode_fp_unary(FpSize::Single, FpUnaryOp::Fsqrt, 1, 0).unwrap();
    assert_eq!(enc, 0x1E21_C020);
}

#[test]
fn test_fmov_s5_s6() {
    let enc = encode_fp_unary(FpSize::Single, FpUnaryOp::FmovReg, 6, 5).unwrap();
    let expected = (0b11110u32 << 24) | (1 << 21) | (0b10000 << 10) | (6 << 5) | 5;
    assert_eq!(enc, expected);
}

#[test]
fn test_fmov_d0_d1() {
    let enc = encode_fp_unary(FpSize::Double, FpUnaryOp::FmovReg, 1, 0).unwrap();
    let expected = (0b11110u32 << 24) | (0b01 << 22) | (1 << 21)
        | (0b10000 << 10) | (1 << 5);
    assert_eq!(enc, expected);
}

// ===========================================================================
// 2. Boundary Tests
// ===========================================================================

#[test]
fn test_imm12_boundary_ok() {
    // 0 and 4095 should both work
    assert!(encode_ldr_str_unsigned_offset(LoadStoreSize::Byte, false, LoadStoreOp::Load, 0, 0, 0).is_ok());
    assert!(encode_ldr_str_unsigned_offset(LoadStoreSize::Byte, false, LoadStoreOp::Load, 4095, 0, 0).is_ok());
}

#[test]
fn test_imm12_boundary_err() {
    let err = encode_ldr_str_unsigned_offset(LoadStoreSize::Byte, false, LoadStoreOp::Load, 4096, 0, 0);
    assert!(matches!(err, Err(EncodeError::Imm12OutOfRange { value: 4096 })));
}

#[test]
fn test_imm9_boundary_ok() {
    assert!(encode_ldr_str_pre_index(LoadStoreSize::Double, false, LoadStoreOp::Store, -256, 0, 0).is_ok());
    assert!(encode_ldr_str_pre_index(LoadStoreSize::Double, false, LoadStoreOp::Store, 255, 0, 0).is_ok());
}

#[test]
fn test_imm9_boundary_err_positive() {
    let err = encode_ldr_str_pre_index(LoadStoreSize::Double, false, LoadStoreOp::Store, 256, 0, 0);
    assert!(matches!(err, Err(EncodeError::Imm9OutOfRange { value: 256 })));
}

#[test]
fn test_imm9_boundary_err_negative() {
    let err = encode_ldr_str_pre_index(LoadStoreSize::Double, false, LoadStoreOp::Store, -257, 0, 0);
    assert!(matches!(err, Err(EncodeError::Imm9OutOfRange { value: -257 })));
}

#[test]
fn test_imm7_boundary_ok() {
    assert!(encode_ldp_stp_offset(PairSize::X64, false, PairOp::LoadPair, -64, 0, 0, 0).is_ok());
    assert!(encode_ldp_stp_offset(PairSize::X64, false, PairOp::LoadPair, 63, 0, 0, 0).is_ok());
}

#[test]
fn test_imm7_boundary_err() {
    // i8 can't represent 64 (overflows to -64), so this tests the Rust type boundary.
    // The overflow at 64 -> -64 still passes the check since -64 is in range.
    // Test with the function that takes i8: 64i8 would overflow at caller side.
    // Instead test -65 which IS representable as i8:
    let err = encode_ldp_stp_offset(PairSize::X64, false, PairOp::LoadPair, -65, 0, 0, 0);
    assert!(matches!(err, Err(EncodeError::Imm7OutOfRange { value: -65 })));
}

#[test]
fn test_imm21_boundary_ok() {
    assert!(encode_adrp(-1_048_576, 0).is_ok());
    assert!(encode_adrp(1_048_575, 0).is_ok());
}

#[test]
fn test_imm21_boundary_err_positive() {
    let err = encode_adrp(1_048_576, 0);
    assert!(matches!(err, Err(EncodeError::Imm21OutOfRange { .. })));
}

#[test]
fn test_imm21_boundary_err_negative() {
    let err = encode_adrp(-1_048_577, 0);
    assert!(matches!(err, Err(EncodeError::Imm21OutOfRange { .. })));
}

#[test]
fn test_register_boundary_ok() {
    assert!(encode_ldr_str_unsigned_offset(LoadStoreSize::Double, false, LoadStoreOp::Load, 0, 0, 0).is_ok());
    assert!(encode_ldr_str_unsigned_offset(LoadStoreSize::Double, false, LoadStoreOp::Load, 0, 31, 31).is_ok());
}

#[test]
fn test_register_boundary_err_rn() {
    let err = encode_ldr_str_unsigned_offset(LoadStoreSize::Double, false, LoadStoreOp::Load, 0, 32, 0);
    assert!(matches!(err, Err(EncodeError::RegisterOutOfRange { reg: 32, max: 31 })));
}

#[test]
fn test_register_boundary_err_rt() {
    let err = encode_ldr_str_unsigned_offset(LoadStoreSize::Double, false, LoadStoreOp::Load, 0, 0, 32);
    assert!(matches!(err, Err(EncodeError::RegisterOutOfRange { reg: 32, max: 31 })));
}

#[test]
fn test_fp_register_boundary_ok() {
    assert!(encode_fp_arith(FpSize::Double, FpArithOp::Add, 0, 0, 0).is_ok());
    assert!(encode_fp_arith(FpSize::Double, FpArithOp::Add, 31, 31, 31).is_ok());
}

#[test]
fn test_fp_register_boundary_err_rm() {
    let err = encode_fp_arith(FpSize::Double, FpArithOp::Add, 32, 0, 0);
    assert!(matches!(err, Err(FpEncodeError::RegisterOutOfRange { reg: 32, max: 31 })));
}

#[test]
fn test_fp_register_boundary_err_rn() {
    let err = encode_fp_arith(FpSize::Double, FpArithOp::Add, 0, 32, 0);
    assert!(matches!(err, Err(FpEncodeError::RegisterOutOfRange { reg: 32, max: 31 })));
}

#[test]
fn test_fp_register_boundary_err_rd() {
    let err = encode_fp_arith(FpSize::Double, FpArithOp::Add, 0, 0, 32);
    assert!(matches!(err, Err(FpEncodeError::RegisterOutOfRange { reg: 32, max: 31 })));
}

// ===========================================================================
// 3. SP vs XZR Encoding Trap Tests
// ===========================================================================

// Register 31 means SP for base address registers (Rn in LDR/STR/LDP/STP)
// and XZR for data registers. The encoder takes raw numbers (0..31) and
// the instruction semantics determine interpretation.

#[test]
fn test_sp_as_base_ldr_unsigned() {
    // LDR X0, [SP, #0] — Rn=31 is SP
    let enc = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Double, false, LoadStoreOp::Load, 0, 31, 0,
    ).unwrap();
    // Verify Rn field is 31
    assert_eq!((enc >> 5) & 0x1F, 31);
    assert_eq!(enc, 0xF940_03E0);
}

#[test]
fn test_sp_as_base_str_pre() {
    // STR X0, [SP, #-16]! — Rn=31 is SP
    let enc = encode_ldr_str_pre_index(
        LoadStoreSize::Double, false, LoadStoreOp::Store, -16, 31, 0,
    ).unwrap();
    assert_eq!((enc >> 5) & 0x1F, 31);
}

#[test]
fn test_sp_as_base_stp_pre() {
    // STP X29, X30, [SP, #-16]! — Rn=31 is SP
    let enc = encode_ldp_stp_pre_index(
        PairSize::X64, false, PairOp::StorePair, -2, 30, 31, 29,
    ).unwrap();
    assert_eq!((enc >> 5) & 0x1F, 31);
    assert_eq!(enc, 0xA9BF_7BFD);
}

#[test]
fn test_xzr_as_rt() {
    // STR XZR, [X1, #0] — Rt=31 is XZR (zero register)
    let enc = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Double, false, LoadStoreOp::Store, 0, 1, 31,
    ).unwrap();
    assert_eq!(enc & 0x1F, 31);
}

#[test]
fn test_rn31_and_rt31_simultaneously() {
    // LDR XZR, [SP, #0] — both Rn=31(SP) and Rt=31(XZR)
    let enc = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Double, false, LoadStoreOp::Load, 0, 31, 31,
    ).unwrap();
    assert_eq!((enc >> 5) & 0x1F, 31); // Rn = SP
    assert_eq!(enc & 0x1F, 31);        // Rt = XZR
}

#[test]
fn test_sp_in_pair_rn() {
    // LDP X0, X1, [SP, #0]
    let enc = encode_ldp_stp_offset(
        PairSize::X64, false, PairOp::LoadPair, 0, 1, 31, 0,
    ).unwrap();
    assert_eq!((enc >> 5) & 0x1F, 31); // Rn = SP
}

// ===========================================================================
// 4. Bit-Field Extraction Tests
// ===========================================================================

#[test]
fn test_extract_size_field_from_ldr_str() {
    for (size, expected) in [
        (LoadStoreSize::Byte, 0b00u32),
        (LoadStoreSize::Half, 0b01),
        (LoadStoreSize::Word, 0b10),
        (LoadStoreSize::Double, 0b11),
    ] {
        let enc = encode_ldr_str_unsigned_offset(size, false, LoadStoreOp::Load, 0, 0, 0).unwrap();
        let extracted = (enc >> 30) & 0b11;
        assert_eq!(extracted, expected, "size field for {:?}", size);
    }
}

#[test]
fn test_extract_v_bit() {
    // V=0 (integer)
    let enc_int = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Double, false, LoadStoreOp::Load, 0, 0, 0,
    ).unwrap();
    assert_eq!((enc_int >> 26) & 1, 0);

    // V=1 (SIMD/FP)
    let enc_fp = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Double, true, LoadStoreOp::Load, 0, 0, 0,
    ).unwrap();
    assert_eq!((enc_fp >> 26) & 1, 1);
}

#[test]
fn test_extract_opc_field() {
    // Load: opc=01
    let enc_load = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Double, false, LoadStoreOp::Load, 0, 0, 0,
    ).unwrap();
    assert_eq!((enc_load >> 22) & 0b11, 0b01);

    // Store: opc=00
    let enc_store = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Double, false, LoadStoreOp::Store, 0, 0, 0,
    ).unwrap();
    assert_eq!((enc_store >> 22) & 0b11, 0b00);
}

#[test]
fn test_extract_imm12_field() {
    let enc = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Double, false, LoadStoreOp::Load, 2048, 0, 0,
    ).unwrap();
    let imm12 = (enc >> 10) & 0xFFF;
    assert_eq!(imm12, 2048);
}

#[test]
fn test_extract_rn_rt_fields() {
    let enc = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Double, false, LoadStoreOp::Load, 0, 17, 23,
    ).unwrap();
    let rn = (enc >> 5) & 0x1F;
    let rt = enc & 0x1F;
    assert_eq!(rn, 17);
    assert_eq!(rt, 23);
}

#[test]
fn test_extract_ftype_from_fp() {
    for (size, expected) in [
        (FpSize::Single, 0b00u32),
        (FpSize::Double, 0b01),
        (FpSize::Half, 0b11),
    ] {
        let enc = encode_fp_arith(size, FpArithOp::Add, 0, 0, 0).unwrap();
        let ftype = (enc >> 22) & 0b11;
        assert_eq!(ftype, expected, "ftype for {:?}", size);
    }
}

#[test]
fn test_extract_fp_opcode_field() {
    for (op, expected) in [
        (FpArithOp::Mul, 0b0000u32),
        (FpArithOp::Div, 0b0001),
        (FpArithOp::Add, 0b0010),
        (FpArithOp::Sub, 0b0011),
    ] {
        let enc = encode_fp_arith(FpSize::Single, op, 0, 0, 0).unwrap();
        let opcode = (enc >> 12) & 0b1111;
        assert_eq!(opcode, expected, "opcode for {:?}", op);
    }
}

#[test]
fn test_extract_pair_mode_field() {
    for (mode, expected) in [
        (PairMode::PostIndex, 0b01u32),
        (PairMode::SignedOffset, 0b10),
        (PairMode::PreIndex, 0b11),
    ] {
        let enc = encode_ldp_stp(
            PairSize::X64, false, PairOp::StorePair, mode, 0, 0, 0, 0,
        ).unwrap();
        let mode_bits = (enc >> 23) & 0b11;
        assert_eq!(mode_bits, expected, "mode for {:?}", mode);
    }
}

#[test]
fn test_extract_imm9_field_negative() {
    // imm9 = -16 => 0x1F0 in 9-bit 2's complement
    let enc = encode_ldr_str_pre_index(
        LoadStoreSize::Double, false, LoadStoreOp::Store, -16, 0, 0,
    ).unwrap();
    let imm9_raw = (enc >> 12) & 0x1FF;
    assert_eq!(imm9_raw, 0x1F0);
}

#[test]
fn test_extract_imm7_field_negative() {
    // imm7 = -2 => 0x7E in 7-bit 2's complement
    let enc = encode_ldp_stp_pre_index(
        PairSize::X64, false, PairOp::StorePair, -2, 0, 0, 0,
    ).unwrap();
    let imm7_raw = (enc >> 15) & 0x7F;
    assert_eq!(imm7_raw, 0x7E);
}

// ===========================================================================
// 5. Mach-O Integration Tests (otool -tv round-trip)
// ===========================================================================

#[test]
fn test_macho_roundtrip_prologue_epilogue() {
    // Encode a realistic function prologue/epilogue sequence
    let mut code = Vec::new();

    // STP X29, X30, [SP, #-16]!
    let stp = encode_ldp_stp_pre_index(
        PairSize::X64, false, PairOp::StorePair, -2, 30, 31, 29,
    ).unwrap();
    code.extend_from_slice(&stp.to_le_bytes());

    // MOV X29, SP => ADD X29, SP, #0 (the pipeline encodes this via AddRI;
    // this test uses a NOP placeholder since it builds raw bytes directly)
    let nop = 0xD503201Fu32;
    code.extend_from_slice(&nop.to_le_bytes());

    // LDP X29, X30, [SP], #16
    let ldp = encode_ldp_stp_post_index(
        PairSize::X64, false, PairOp::LoadPair, 2, 30, 31, 29,
    ).unwrap();
    code.extend_from_slice(&ldp.to_le_bytes());

    // RET
    let ret = 0xD65F03C0u32;
    code.extend_from_slice(&ret.to_le_bytes());

    let mut writer = MachOWriter::new();
    writer.add_text_section(&code);
    writer.add_symbol("_test_func", 1, 0, true);

    let bytes = writer.write();
    let path = write_temp_o(&bytes, "prologue_epilogue");
    let disasm = run_otool_tv(&path);

    // Verify key mnemonics appear in disassembly
    assert!(disasm.contains("stp"), "Missing stp in disassembly: {}", disasm);
    assert!(disasm.contains("ldp"), "Missing ldp in disassembly: {}", disasm);
    assert!(disasm.contains("ret"), "Missing ret in disassembly: {}", disasm);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_macho_roundtrip_load_store() {
    let mut code = Vec::new();

    // STR X0, [SP, #0]
    let str_inst = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Double, false, LoadStoreOp::Store, 0, 31, 0,
    ).unwrap();
    code.extend_from_slice(&str_inst.to_le_bytes());

    // LDR X1, [SP, #0]
    let ldr_inst = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Double, false, LoadStoreOp::Load, 0, 31, 1,
    ).unwrap();
    code.extend_from_slice(&ldr_inst.to_le_bytes());

    // ADRP X2, #0
    let adrp = encode_adrp(0, 2).unwrap();
    code.extend_from_slice(&adrp.to_le_bytes());

    // RET
    code.extend_from_slice(&0xD65F03C0u32.to_le_bytes());

    let mut writer = MachOWriter::new();
    writer.add_text_section(&code);
    writer.add_symbol("_test_loads", 1, 0, true);

    let bytes = writer.write();
    let path = write_temp_o(&bytes, "load_store");
    let disasm = run_otool_tv(&path);

    assert!(disasm.contains("str"), "Missing str in disassembly: {}", disasm);
    assert!(disasm.contains("ldr"), "Missing ldr in disassembly: {}", disasm);
    assert!(disasm.contains("adrp"), "Missing adrp in disassembly: {}", disasm);
    assert!(disasm.contains("ret"), "Missing ret in disassembly: {}", disasm);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_macho_roundtrip_fp_instructions() {
    let mut code = Vec::new();

    // FADD D0, D1, D2
    let fadd = encode_fp_arith(FpSize::Double, FpArithOp::Add, 2, 1, 0).unwrap();
    code.extend_from_slice(&fadd.to_le_bytes());

    // FSUB D3, D4, D5
    let fsub = encode_fp_arith(FpSize::Double, FpArithOp::Sub, 5, 4, 3).unwrap();
    code.extend_from_slice(&fsub.to_le_bytes());

    // FMUL D6, D7, D8
    let fmul = encode_fp_arith(FpSize::Double, FpArithOp::Mul, 8, 7, 6).unwrap();
    code.extend_from_slice(&fmul.to_le_bytes());

    // FDIV D9, D10, D11
    let fdiv = encode_fp_arith(FpSize::Double, FpArithOp::Div, 11, 10, 9).unwrap();
    code.extend_from_slice(&fdiv.to_le_bytes());

    // FCMP D0, #0.0
    let fcmp = encode_fcmp(FpSize::Double, FpCmpOp::CmpZero, 0, 0).unwrap();
    code.extend_from_slice(&fcmp.to_le_bytes());

    // FNEG D0, D1
    let fneg = encode_fp_unary(FpSize::Double, FpUnaryOp::Fneg, 1, 0).unwrap();
    code.extend_from_slice(&fneg.to_le_bytes());

    // FABS D2, D3
    let fabs = encode_fp_unary(FpSize::Double, FpUnaryOp::Fabs, 3, 2).unwrap();
    code.extend_from_slice(&fabs.to_le_bytes());

    // FSQRT D4, D5
    let fsqrt = encode_fp_unary(FpSize::Double, FpUnaryOp::Fsqrt, 5, 4).unwrap();
    code.extend_from_slice(&fsqrt.to_le_bytes());

    // RET
    code.extend_from_slice(&0xD65F03C0u32.to_le_bytes());

    let mut writer = MachOWriter::new();
    writer.add_text_section(&code);
    writer.add_symbol("_test_fp", 1, 0, true);

    let bytes = writer.write();
    let path = write_temp_o(&bytes, "fp_instructions");
    let disasm = run_otool_tv(&path);

    assert!(disasm.contains("fadd"), "Missing fadd: {}", disasm);
    assert!(disasm.contains("fsub"), "Missing fsub: {}", disasm);
    assert!(disasm.contains("fmul"), "Missing fmul: {}", disasm);
    assert!(disasm.contains("fdiv"), "Missing fdiv: {}", disasm);
    assert!(disasm.contains("fcmp"), "Missing fcmp: {}", disasm);
    assert!(disasm.contains("fneg"), "Missing fneg: {}", disasm);
    assert!(disasm.contains("fabs"), "Missing fabs: {}", disasm);
    assert!(disasm.contains("fsqrt"), "Missing fsqrt: {}", disasm);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_macho_roundtrip_mixed_sequence() {
    // A realistic mixed function: prologue, some loads, FP work, epilogue
    let mut code = Vec::new();

    // STP X29, X30, [SP, #-16]!
    let stp = encode_ldp_stp_pre_index(
        PairSize::X64, false, PairOp::StorePair, -2, 30, 31, 29,
    ).unwrap();
    code.extend_from_slice(&stp.to_le_bytes());

    // ADRP X0, #0
    let adrp = encode_adrp(0, 0).unwrap();
    code.extend_from_slice(&adrp.to_le_bytes());

    // LDR X1, [X0, #0]
    let ldr = encode_ldr_str_unsigned_offset(
        LoadStoreSize::Double, false, LoadStoreOp::Load, 0, 0, 1,
    ).unwrap();
    code.extend_from_slice(&ldr.to_le_bytes());

    // FADD D0, D1, D2
    let fadd = encode_fp_arith(FpSize::Double, FpArithOp::Add, 2, 1, 0).unwrap();
    code.extend_from_slice(&fadd.to_le_bytes());

    // FCMP D0, #0.0
    let fcmp = encode_fcmp(FpSize::Double, FpCmpOp::CmpZero, 0, 0).unwrap();
    code.extend_from_slice(&fcmp.to_le_bytes());

    // LDP X29, X30, [SP], #16
    let ldp = encode_ldp_stp_post_index(
        PairSize::X64, false, PairOp::LoadPair, 2, 30, 31, 29,
    ).unwrap();
    code.extend_from_slice(&ldp.to_le_bytes());

    // RET
    code.extend_from_slice(&0xD65F03C0u32.to_le_bytes());

    let mut writer = MachOWriter::new();
    writer.add_text_section(&code);
    writer.add_symbol("_mixed_func", 1, 0, true);

    let bytes = writer.write();
    let path = write_temp_o(&bytes, "mixed_sequence");
    let disasm = run_otool_tv(&path);

    // Verify the disassembly contains the key instructions
    assert!(disasm.contains("stp"), "Missing stp: {}", disasm);
    assert!(disasm.contains("adrp"), "Missing adrp: {}", disasm);
    assert!(disasm.contains("ldr"), "Missing ldr: {}", disasm);
    assert!(disasm.contains("fadd"), "Missing fadd: {}", disasm);
    assert!(disasm.contains("fcmp"), "Missing fcmp: {}", disasm);
    assert!(disasm.contains("ldp"), "Missing ldp: {}", disasm);
    assert!(disasm.contains("ret"), "Missing ret: {}", disasm);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_macho_roundtrip_fp_conversions() {
    let mut code = Vec::new();

    // FCVTZS W0, S1
    let fcvtzs = encode_fp_int_conv(false, FpSize::Single, FpConvOp::FcvtzsToInt, 1, 0).unwrap();
    code.extend_from_slice(&fcvtzs.to_le_bytes());

    // SCVTF S0, W1
    let scvtf = encode_fp_int_conv(false, FpSize::Single, FpConvOp::ScvtfToFp, 1, 0).unwrap();
    code.extend_from_slice(&scvtf.to_le_bytes());

    // RET
    code.extend_from_slice(&0xD65F03C0u32.to_le_bytes());

    let mut writer = MachOWriter::new();
    writer.add_text_section(&code);
    writer.add_symbol("_test_conv", 1, 0, true);

    let bytes = writer.write();
    let path = write_temp_o(&bytes, "fp_conv");
    let disasm = run_otool_tv(&path);

    assert!(disasm.contains("fcvtzs"), "Missing fcvtzs: {}", disasm);
    assert!(disasm.contains("scvtf"), "Missing scvtf: {}", disasm);

    std::fs::remove_file(&path).ok();
}

// ===========================================================================
// 6. Parametric Exhaustive Tests
// ===========================================================================

#[test]
fn test_all_load_store_size_x_op_combinations() {
    // All LoadStoreSize x LoadStoreOp should encode successfully
    let sizes = [
        (LoadStoreSize::Byte, 0b00u32),
        (LoadStoreSize::Half, 0b01),
        (LoadStoreSize::Word, 0b10),
        (LoadStoreSize::Double, 0b11),
    ];
    let ops = [
        (LoadStoreOp::Store, 0b00u32),
        (LoadStoreOp::Load, 0b01),
    ];

    for (size, size_bits) in &sizes {
        for (op, op_bits) in &ops {
            let enc = encode_ldr_str_unsigned_offset(*size, false, *op, 0, 0, 0).unwrap();
            assert_eq!((enc >> 30) & 0b11, *size_bits, "size mismatch for {:?}", size);
            assert_eq!((enc >> 22) & 0b11, *op_bits, "opc mismatch for {:?}", op);
            // Verify the fixed bits pattern: bits[29:27] = 111, bits[25:24] = 01
            assert_eq!((enc >> 27) & 0b111, 0b111);
            assert_eq!((enc >> 24) & 0b11, 0b01);
        }
    }
}

#[test]
fn test_all_fp_size_x_arith_op_combinations() {
    let sizes = [
        (FpSize::Single, 0b00u32),
        (FpSize::Double, 0b01),
        (FpSize::Half, 0b11),
    ];
    let ops = [
        (FpArithOp::Mul, 0b0000u32),
        (FpArithOp::Div, 0b0001),
        (FpArithOp::Add, 0b0010),
        (FpArithOp::Sub, 0b0011),
    ];

    for (size, ftype) in &sizes {
        for (op, opcode) in &ops {
            let enc = encode_fp_arith(*size, *op, 0, 0, 0).unwrap();
            assert_eq!((enc >> 22) & 0b11, *ftype, "ftype for {:?}", size);
            assert_eq!((enc >> 12) & 0b1111, *opcode, "opcode for {:?}", op);
            // Verify fixed bits: [28:24]=11110, [21]=1, [11:10]=10
            assert_eq!((enc >> 24) & 0b11111, 0b11110);
            assert_eq!((enc >> 21) & 1, 1);
            assert_eq!((enc >> 10) & 0b11, 0b10);
        }
    }
}

#[test]
fn test_all_fp_unary_op_x_size_combinations() {
    let sizes = [
        (FpSize::Single, 0b00u32),
        (FpSize::Double, 0b01),
        (FpSize::Half, 0b11),
    ];
    let ops = [
        (FpUnaryOp::FmovReg, 0b00u32),
        (FpUnaryOp::Fabs, 0b01),
        (FpUnaryOp::Fneg, 0b10),
        (FpUnaryOp::Fsqrt, 0b11),
    ];

    for (size, ftype) in &sizes {
        for (op, opcode) in &ops {
            let enc = encode_fp_unary(*size, *op, 0, 0).unwrap();
            assert_eq!((enc >> 22) & 0b11, *ftype, "ftype for {:?}", size);
            assert_eq!((enc >> 15) & 0b11, *opcode, "opcode for {:?}", op);
            // Fixed bits: [28:24]=11110, [21]=1, [14:10]=10000
            assert_eq!((enc >> 24) & 0b11111, 0b11110);
            assert_eq!((enc >> 21) & 1, 1);
            assert_eq!((enc >> 10) & 0b11111, 0b10000);
        }
    }
}

#[test]
fn test_all_pair_size_x_op_x_mode_combinations() {
    let pair_sizes = [
        (PairSize::W32, 0b00u32),
        (PairSize::X64, 0b10),
    ];
    let pair_ops = [
        (PairOp::StorePair, 0u32),
        (PairOp::LoadPair, 1),
    ];
    let modes = [
        (PairMode::PostIndex, 0b01u32),
        (PairMode::SignedOffset, 0b10),
        (PairMode::PreIndex, 0b11),
    ];

    for (ps, ps_bits) in &pair_sizes {
        for (po, po_bits) in &pair_ops {
            for (m, m_bits) in &modes {
                let enc = encode_ldp_stp(*ps, false, *po, *m, 0, 0, 0, 0).unwrap();
                assert_eq!((enc >> 30) & 0b11, *ps_bits, "opc for {:?}", ps);
                assert_eq!((enc >> 22) & 1, *po_bits, "L for {:?}", po);
                assert_eq!((enc >> 23) & 0b11, *m_bits, "mode for {:?}", m);
                // Fixed bits: [29:27] = 101
                assert_eq!((enc >> 27) & 0b111, 0b101);
            }
        }
    }
}

#[test]
fn test_all_reg_extend_x_shift_combinations() {
    let extends = [
        (RegExtend::Uxtw, 0b010u32),
        (RegExtend::Lsl, 0b011),
        (RegExtend::Sxtw, 0b110),
        (RegExtend::Sxtx, 0b111),
    ];
    let shifts = [false, true];

    for (ext, ext_bits) in &extends {
        for &s in &shifts {
            let enc = encode_ldr_str_register(
                LoadStoreSize::Double, false, LoadStoreOp::Load,
                0, *ext, s, 0, 0,
            ).unwrap();
            assert_eq!((enc >> 13) & 0b111, *ext_bits, "option for {:?}", ext);
            assert_eq!((enc >> 12) & 1, s as u32, "S for shift={}", s);
            // Verify register-offset marker bit[21]=1
            assert_eq!((enc >> 21) & 1, 1);
        }
    }
}

#[test]
fn test_all_registers_rt_parametric() {
    // Verify Rt field encodes correctly for all 32 registers
    for rt in 0u8..=31 {
        let enc = encode_ldr_str_unsigned_offset(
            LoadStoreSize::Double, false, LoadStoreOp::Load, 0, 0, rt,
        ).unwrap();
        assert_eq!(enc & 0x1F, rt as u32, "Rt mismatch for {}", rt);
    }
}

#[test]
fn test_all_registers_rn_parametric() {
    // Verify Rn field encodes correctly for all 32 registers
    for rn in 0u8..=31 {
        let enc = encode_ldr_str_unsigned_offset(
            LoadStoreSize::Double, false, LoadStoreOp::Load, 0, rn, 0,
        ).unwrap();
        assert_eq!((enc >> 5) & 0x1F, rn as u32, "Rn mismatch for {}", rn);
    }
}

#[test]
fn test_all_fp_registers_parametric() {
    // Verify Rd, Rn, Rm fields for all 32 FP registers
    for r in 0u8..=31 {
        let enc = encode_fp_arith(FpSize::Double, FpArithOp::Add, r, r, r).unwrap();
        assert_eq!(enc & 0x1F, r as u32, "Rd mismatch for {}", r);
        assert_eq!((enc >> 5) & 0x1F, r as u32, "Rn mismatch for {}", r);
        assert_eq!((enc >> 16) & 0x1F, r as u32, "Rm mismatch for {}", r);
    }
}

#[test]
fn test_all_fcmp_ops_parametric() {
    let cmp_ops = [
        (FpCmpOp::Cmp, 0b00000u32),
        (FpCmpOp::CmpZero, 0b01000),
        (FpCmpOp::Cmpe, 0b10000),
        (FpCmpOp::CmpeZero, 0b11000),
    ];

    for (op, expected_opc) in &cmp_ops {
        let enc = encode_fcmp(FpSize::Double, *op, 0, 0).unwrap();
        assert_eq!(enc & 0x1F, *expected_opc, "opc mismatch for {:?}", op);
    }
}

#[test]
fn test_all_fp_conv_ops_parametric() {
    let conv_ops = [
        (FpConvOp::FcvtzsToInt, 0b11u32, 0b000u32),
        (FpConvOp::ScvtfToFp, 0b00, 0b010),
        (FpConvOp::FmovToFp, 0b00, 0b111),
        (FpConvOp::FmovToGp, 0b00, 0b110),
    ];

    for (op, rmode, opcode) in &conv_ops {
        let enc = encode_fp_int_conv(false, FpSize::Single, *op, 0, 0).unwrap();
        assert_eq!((enc >> 19) & 0b11, *rmode, "rmode for {:?}", op);
        assert_eq!((enc >> 16) & 0b111, *opcode, "opcode for {:?}", op);
    }
}
