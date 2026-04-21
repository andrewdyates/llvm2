// llvm2-codegen/aarch64/encoding.rs - AArch64 instruction encoding
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! AArch64 binary instruction encoding.
//!
//! Each function encodes a specific AArch64 instruction format into a 32-bit
//! word. Bit layouts are from the ARM Architecture Reference Manual (DDI 0487).
//!
//! Reference: `~/llvm-project-ref/llvm/lib/Target/AArch64/AArch64InstrFormats.td`

// ---------------------------------------------------------------------------
// Condition codes (ARM ARM C1.2.4)
// ---------------------------------------------------------------------------

/// AArch64 condition codes for conditional branches and selects.
pub mod cond {
    pub const EQ: u32 = 0b0000; // Equal (Z==1)
    pub const NE: u32 = 0b0001; // Not equal (Z==0)
    pub const CS: u32 = 0b0010; // Carry set / unsigned higher or same
    pub const HS: u32 = CS;
    pub const CC: u32 = 0b0011; // Carry clear / unsigned lower
    pub const LO: u32 = CC;
    pub const MI: u32 = 0b0100; // Minus / negative
    pub const PL: u32 = 0b0101; // Plus / positive or zero
    pub const VS: u32 = 0b0110; // Overflow
    pub const VC: u32 = 0b0111; // No overflow
    pub const HI: u32 = 0b1000; // Unsigned higher
    pub const LS: u32 = 0b1001; // Unsigned lower or same
    pub const GE: u32 = 0b1010; // Signed greater or equal
    pub const LT: u32 = 0b1011; // Signed less than
    pub const GT: u32 = 0b1100; // Signed greater than
    pub const LE: u32 = 0b1101; // Signed less or equal
    pub const AL: u32 = 0b1110; // Always
    pub const NV: u32 = 0b1111; // Always (alternate)
}

// ---------------------------------------------------------------------------
// Shift types (ARM ARM C6.2)
// ---------------------------------------------------------------------------

/// AArch64 shift types for data-processing (shifted register) instructions.
pub mod shift {
    pub const LSL: u32 = 0b00;
    pub const LSR: u32 = 0b01;
    pub const ASR: u32 = 0b10;
    pub const ROR: u32 = 0b11; // Only valid for logical instructions
}

// ---------------------------------------------------------------------------
// Encoding functions
// ---------------------------------------------------------------------------

/// Encode **Add/Subtract (shifted register)** — ADD, SUB, ADDS, SUBS.
///
/// ```text
/// 31  30 29  28:24   23:22  21  20:16  15:10  9:5  4:0
/// sf  op  S  01011  shift   0    Rm    imm6   Rn   Rd
/// ```
///
/// Source: `BaseAddSubSReg` in `AArch64InstrFormats.td`
pub fn encode_add_sub_shifted_reg(
    sf: u32,
    op: u32,
    s: u32,
    shift: u32,
    rm: u32,
    imm6: u32,
    rn: u32,
    rd: u32,
) -> u32 {
    debug_assert!(sf <= 1);
    debug_assert!(op <= 1);
    debug_assert!(s <= 1);
    debug_assert!(shift <= 0b10); // ROR (0b11) is reserved for add/sub
    debug_assert!(rm <= 31);
    debug_assert!(imm6 <= 63);
    debug_assert!(rn <= 31);
    debug_assert!(rd <= 31);

    (sf << 31)
        | (op << 30)
        | (s << 29)
        | (0b01011 << 24)
        | (shift << 22)
        // bit 21 = 0 (implicit)
        | (rm << 16)
        | (imm6 << 10)
        | (rn << 5)
        | rd
}

/// Encode **Logical (shifted register)** — AND, ORR, EOR, ANDS, BIC, ORN, EON, BICS.
///
/// ```text
/// 31  30:29  28:24  23:22  21  20:16  15:10  9:5  4:0
/// sf   opc   01010  shift   N    Rm    imm6   Rn   Rd
/// ```
///
/// Source: `BaseLogicalSReg` in `AArch64InstrFormats.td`
pub fn encode_logical_shifted_reg(
    sf: u32,
    opc: u32,
    shift: u32,
    n: u32,
    rm: u32,
    imm6: u32,
    rn: u32,
    rd: u32,
) -> u32 {
    debug_assert!(sf <= 1);
    debug_assert!(opc <= 0b11);
    debug_assert!(shift <= 0b11);
    debug_assert!(n <= 1);
    debug_assert!(rm <= 31);
    debug_assert!(imm6 <= 63);
    debug_assert!(rn <= 31);
    debug_assert!(rd <= 31);

    (sf << 31)
        | (opc << 29)
        | (0b01010 << 24)
        | (shift << 22)
        | (n << 21)
        | (rm << 16)
        | (imm6 << 10)
        | (rn << 5)
        | rd
}

/// Encode **Add/Subtract (immediate)** — ADD imm, SUB imm, ADDS imm, SUBS imm.
///
/// ```text
/// 31  30 29  28:23   22   21:10   9:5  4:0
/// sf  op  S  100010   sh  imm12    Rn   Rd
/// ```
///
/// Source: `BaseAddSubImm` in `AArch64InstrFormats.td`
pub fn encode_add_sub_imm(
    sf: u32,
    op: u32,
    s: u32,
    sh: u32,
    imm12: u32,
    rn: u32,
    rd: u32,
) -> u32 {
    debug_assert!(sf <= 1);
    debug_assert!(op <= 1);
    debug_assert!(s <= 1);
    debug_assert!(sh <= 1);
    debug_assert!(imm12 <= 0xFFF);
    debug_assert!(rn <= 31);
    debug_assert!(rd <= 31);

    (sf << 31)
        | (op << 30)
        | (s << 29)
        | (0b100010 << 23)
        | (sh << 22)
        | (imm12 << 10)
        | (rn << 5)
        | rd
}

/// Encode **Move Wide (immediate)** — MOVN, MOVZ, MOVK.
///
/// ```text
/// 31  30:29  28:23   22:21  20:5    4:0
/// sf   opc   100101   hw    imm16    Rd
/// ```
///
/// `opc`: 00=MOVN, 10=MOVZ, 11=MOVK (01 is unallocated).
///
/// Callers are responsible for validating operand ranges and returning
/// `EncodeError::InvalidOperand` when an untrusted input (e.g. a Movk
/// shift operand) is out of range. For defense in depth this encoder
/// masks each sub-field to its legal bit-width instead of tripping a
/// `debug_assert!` — that used to crash the process on malformed input
/// from the panic-fuzz harness (#387 / #447). The assertions about
/// `sf`, `opc != 0b01`, and `rd <= 31` remain because they indicate
/// *programmer error* internal to the encoder, not attacker-controlled
/// inputs.
///
/// Source: `BaseMoveImmediate` in `AArch64InstrFormats.td`
pub fn encode_move_wide(sf: u32, opc: u32, hw: u32, imm16: u32, rd: u32) -> u32 {
    debug_assert!(sf <= 1);
    debug_assert!(opc <= 0b11 && opc != 0b01); // 01 is unallocated
    debug_assert!(rd <= 31);

    // Mask `hw` and `imm16` defensively: on untrusted inputs (e.g. a
    // Movk dispatch arm that took a garbage shift operand) the caller
    // is expected to have already returned `Err(..)` — see #447. The
    // masking here guarantees that if it hasn't, we still produce a
    // well-formed (but semantically unspecified) encoding instead of
    // panicking in debug mode.
    let hw = hw & 0b11;
    let imm16 = imm16 & 0xFFFF;
    let rd = rd & 0b1_1111;

    (sf << 31)
        | (opc << 29)
        | (0b100101 << 23)
        | (hw << 21)
        | (imm16 << 5)
        | rd
}

/// Encode **Conditional Branch** — B.cond.
///
/// ```text
/// 31:24       23:5    4    3:0
/// 01010100    imm19   0    cond
/// ```
///
/// Source: `BranchCond` in `AArch64InstrFormats.td`
pub fn encode_cond_branch(imm19: u32, cond: u32) -> u32 {
    debug_assert!(imm19 <= 0x7FFFF); // 19 bits
    debug_assert!(cond <= 0xF);

    (0b01010100 << 24) | (imm19 << 5) | cond
    // bit 4 is 0 (o1 field)
}

/// Encode **Unconditional Branch (immediate)** — B, BL.
///
/// ```text
/// 31   30:26   25:0
/// op   00101   imm26
/// ```
///
/// `op`: 0=B, 1=BL.
///
/// Source: `BImm` in `AArch64InstrFormats.td`
pub fn encode_uncond_branch(op: u32, imm26: u32) -> u32 {
    debug_assert!(op <= 1);
    debug_assert!(imm26 <= 0x3FFFFFF); // 26 bits

    (op << 31) | (0b00101 << 26) | imm26
}

/// Encode **Branch to Register** — BR, BLR, RET.
///
/// ```text
/// 31:25     24:21  20:16    15:10    9:5  4:0
/// 1101011   opc    11111    000000    Rn   00000
/// ```
///
/// `opc`: 0000=BR, 0001=BLR, 0010=RET.
///
/// Source: `BaseBranchReg` in `AArch64InstrFormats.td`
pub fn encode_branch_reg(opc: u32, rn: u32) -> u32 {
    debug_assert!(opc <= 0b1111);
    debug_assert!(rn <= 31);

    (0b1101011 << 25)
        | (opc << 21)
        | (0b11111 << 16)
        // bits 15:10 = 000000 (implicit)
        | (rn << 5)
    // bits 4:0 = 00000 (implicit)
}

/// Encode **Load/Store Register (unsigned immediate offset)**.
///
/// ```text
/// 31:30  29:27  26   25:24  23:22  21:10   9:5  4:0
/// size    111    V    01     opc    imm12    Rn   Rt
/// ```
///
/// The `imm12` field is the *already-scaled* offset (e.g. byte offset / access-size).
///
/// Source: `BaseLoadStoreUI` in `AArch64InstrFormats.td`
pub fn encode_load_store_ui(
    size: u32,
    v: u32,
    opc: u32,
    imm12: u32,
    rn: u32,
    rt: u32,
) -> u32 {
    debug_assert!(size <= 0b11);
    debug_assert!(v <= 1);
    debug_assert!(opc <= 0b11);
    debug_assert!(imm12 <= 0xFFF);
    debug_assert!(rn <= 31);
    debug_assert!(rt <= 31);

    (size << 30)
        | (0b111 << 27)
        | (v << 26)
        | (0b01 << 24)
        | (opc << 22)
        | (imm12 << 10)
        | (rn << 5)
        | rt
}

/// Encode **Load/Store (unscaled immediate)** — LDUR, STUR.
///
/// ```text
/// 31:30  29:27  26   25:24  23:22  21   20:12   11:10  9:5  4:0
/// size    111    V    00     opc    0    imm9     00     Rn   Rt
/// ```
///
/// * `size` — data width (00=byte, 01=half, 10=word, 11=double)
/// * `v` — 1 for SIMD/FP, 0 for integer
/// * `opc` — 00=store, 01=load
/// * `imm9` — signed 9-bit offset (-256..255), NOT scaled
/// * `rn` — base register (0..31, 31 = SP)
/// * `rt` — transfer register (0..31)
///
/// Source: ARM Architecture Reference Manual (DDI 0487), LDUR/STUR encoding
pub fn encode_load_store_unscaled(
    size: u32,
    v: u32,
    opc: u32,
    imm9: i32,
    rn: u32,
    rt: u32,
) -> u32 {
    debug_assert!(size <= 0b11);
    debug_assert!(v <= 1);
    debug_assert!(opc <= 0b11);
    debug_assert!((-256..=255).contains(&imm9));
    debug_assert!(rn <= 31);
    debug_assert!(rt <= 31);

    let imm9_bits = (imm9 as u32) & 0x1FF;

    (size << 30)
        | (0b111 << 27)
        | (v << 26)
        // bits [25:24] = 00 (unscaled/pre/post family)
        | (opc << 22)
        // bit [21] = 0
        | (imm9_bits << 12)
        // bits [11:10] = 00 (unscaled)
        | (rn << 5)
        | rt
}

/// Encode **Load/Store Pair (signed offset)** — LDP, STP.
///
/// ```text
/// 31:30  29:27  26   25:23  22   21:15  14:10  9:5  4:0
///  opc    101    V    010    L    imm7    Rt2    Rn   Rt
/// ```
///
/// The `imm7` field is the *already-scaled* signed offset.
///
/// Source: `BaseLoadStorePairOffset` in `AArch64InstrFormats.td`
pub fn encode_load_store_pair(
    opc: u32,
    v: u32,
    l: u32,
    imm7: u32,
    rt2: u32,
    rn: u32,
    rt: u32,
) -> u32 {
    debug_assert!(opc <= 0b11);
    debug_assert!(v <= 1);
    debug_assert!(l <= 1);
    debug_assert!(imm7 <= 0x7F);
    debug_assert!(rt2 <= 31);
    debug_assert!(rn <= 31);
    debug_assert!(rt <= 31);

    (opc << 30)
        | (0b101 << 27)
        | (v << 26)
        | (0b010 << 23)
        | (l << 22)
        | (imm7 << 15)
        | (rt2 << 10)
        | (rn << 5)
        | rt
}

/// Encode **Compare and Branch** — CBZ, CBNZ.
///
/// ```text
/// 31   30:25    24   23:5    4:0
/// sf   011010   op   imm19    Rt
/// ```
///
/// `op`: 0=CBZ, 1=CBNZ.
///
/// Source: `BaseCmpBranch` in `AArch64InstrFormats.td`
pub fn encode_cmp_branch(sf: u32, op: u32, imm19: u32, rt: u32) -> u32 {
    debug_assert!(sf <= 1);
    debug_assert!(op <= 1);
    debug_assert!(imm19 <= 0x7FFFF);
    debug_assert!(rt <= 31);

    (sf << 31) | (0b011010 << 25) | (op << 24) | (imm19 << 5) | rt
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Add/Subtract (shifted register)
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_x0_x1_x2() {
        // ADD X0, X1, X2
        // sf=1, op=0(ADD), S=0, shift=00(LSL), Rm=2, imm6=0, Rn=1, Rd=0
        // Manual: 1_0_0_01011_00_0_00010_000000_00001_00000
        //       = 0b1000_1011_0000_0000_0010_0000_0010_0000
        //       = 0x8B020020
        let enc = encode_add_sub_shifted_reg(1, 0, 0, 0, 2, 0, 1, 0);
        assert_eq!(enc, 0x8B020020, "ADD X0, X1, X2 = 0x{enc:08X}");
    }

    #[test]
    fn test_sub_w3_w4_w5() {
        // SUB W3, W4, W5
        // sf=0, op=1(SUB), S=0, shift=00, Rm=5, imm6=0, Rn=4, Rd=3
        // Manual: 0_1_0_01011_00_0_00101_000000_00100_00011
        //       = 0b0100_1011_0000_0000_0101_0000_1000_0011
        //       = 0x4B050083
        let enc = encode_add_sub_shifted_reg(0, 1, 0, 0, 5, 0, 4, 3);
        assert_eq!(enc, 0x4B050083, "SUB W3, W4, W5 = 0x{enc:08X}");
    }

    #[test]
    fn test_adds_x0_x1_x2() {
        // ADDS X0, X1, X2
        // sf=1, op=0, S=1, shift=00, Rm=2, imm6=0, Rn=1, Rd=0
        // Manual: 1_0_1_01011_00_0_00010_000000_00001_00000
        //       = 0xAB020020
        let enc = encode_add_sub_shifted_reg(1, 0, 1, 0, 2, 0, 1, 0);
        assert_eq!(enc, 0xAB020020, "ADDS X0, X1, X2 = 0x{enc:08X}");
    }

    #[test]
    fn test_add_shifted_reg_with_shift() {
        // ADD X0, X1, X2, LSL #3
        // sf=1, op=0, S=0, shift=00(LSL), Rm=2, imm6=3, Rn=1, Rd=0
        // Manual: 1_0_0_01011_00_0_00010_000011_00001_00000
        let expected = (1u32 << 31)
            | (0b01011 << 24)
            | (2 << 16)
            | (3 << 10)
            | (1 << 5);
        let enc = encode_add_sub_shifted_reg(1, 0, 0, 0, 2, 3, 1, 0);
        assert_eq!(enc, expected, "ADD X0, X1, X2, LSL #3 = 0x{enc:08X}");
    }

    #[test]
    fn test_sub_shifted_reg_xzr() {
        // SUB XZR, X0, X1 (NEG alias): rd=31
        // sf=1, op=1, S=0, shift=00, Rm=1, imm6=0, Rn=0, Rd=31
        let expected = ((1u32 << 31)
            | (1 << 30)
            | (0b01011 << 24)
            | (1 << 16))
            | 31;
        let enc = encode_add_sub_shifted_reg(1, 1, 0, 0, 1, 0, 0, 31);
        assert_eq!(enc, expected, "SUB XZR, X0, X1 = 0x{enc:08X}");
    }

    #[test]
    fn test_add_shifted_reg_asr() {
        // ADD X5, X6, X7, ASR #4
        // sf=1, op=0, S=0, shift=10(ASR), Rm=7, imm6=4, Rn=6, Rd=5
        let expected = (1u32 << 31)
            | (0b01011 << 24)
            | (0b10 << 22)
            | (7 << 16)
            | (4 << 10)
            | (6 << 5)
            | 5;
        let enc = encode_add_sub_shifted_reg(1, 0, 0, 0b10, 7, 4, 6, 5);
        assert_eq!(enc, expected, "ADD X5, X6, X7, ASR #4 = 0x{enc:08X}");
    }

    // -----------------------------------------------------------------------
    // Logical (shifted register)
    // -----------------------------------------------------------------------

    #[test]
    fn test_and_x0_x1_x2() {
        // AND X0, X1, X2
        // sf=1, opc=00(AND), shift=00, N=0, Rm=2, imm6=0, Rn=1, Rd=0
        // Manual: 1_00_01010_00_0_00010_000000_00001_00000
        //       = 0x8A020020
        let enc = encode_logical_shifted_reg(1, 0b00, 0, 0, 2, 0, 1, 0);
        assert_eq!(enc, 0x8A020020, "AND X0, X1, X2 = 0x{enc:08X}");
    }

    #[test]
    fn test_orr_x0_x1_x2() {
        // ORR X0, X1, X2
        // sf=1, opc=01(ORR), shift=00, N=0, Rm=2, imm6=0, Rn=1, Rd=0
        // Manual: 1_01_01010_00_0_00010_000000_00001_00000
        //       = 0xAA020020
        let enc = encode_logical_shifted_reg(1, 0b01, 0, 0, 2, 0, 1, 0);
        assert_eq!(enc, 0xAA020020, "ORR X0, X1, X2 = 0x{enc:08X}");
    }

    #[test]
    fn test_eor_x0_x1_x2() {
        // EOR X0, X1, X2
        // sf=1, opc=10(EOR), shift=00, N=0, Rm=2, imm6=0, Rn=1, Rd=0
        // Manual: 1_10_01010_00_0_00010_000000_00001_00000
        //       = 0xCA020020
        let enc = encode_logical_shifted_reg(1, 0b10, 0, 0, 2, 0, 1, 0);
        assert_eq!(enc, 0xCA020020, "EOR X0, X1, X2 = 0x{enc:08X}");
    }

    #[test]
    fn test_ands_x0_x1_x2() {
        // ANDS X0, X1, X2 (TST alias uses XZR as Rd)
        // sf=1, opc=11(ANDS), shift=00, N=0, Rm=2, imm6=0, Rn=1, Rd=0
        // Manual: 1_11_01010_00_0_00010_000000_00001_00000
        //       = 0xEA020020
        let enc = encode_logical_shifted_reg(1, 0b11, 0, 0, 2, 0, 1, 0);
        assert_eq!(enc, 0xEA020020, "ANDS X0, X1, X2 = 0x{enc:08X}");
    }

    #[test]
    fn test_bic_x0_x1_x2() {
        // BIC X0, X1, X2  (AND with N=1 — inverted Rm)
        // sf=1, opc=00, shift=00, N=1, Rm=2, imm6=0, Rn=1, Rd=0
        let expected = (1u32 << 31)
            | (0b01010 << 24)
            | (1 << 21) // N=1
            | (2 << 16)
            | (1 << 5);
        let enc = encode_logical_shifted_reg(1, 0b00, 0, 1, 2, 0, 1, 0);
        assert_eq!(enc, expected, "BIC X0, X1, X2 = 0x{enc:08X}");
    }

    #[test]
    fn test_orr_shifted_ror() {
        // ORR X3, X4, X5, ROR #7
        // sf=1, opc=01, shift=11(ROR), N=0, Rm=5, imm6=7, Rn=4, Rd=3
        let expected = (1u32 << 31)
            | (0b01 << 29)
            | (0b01010 << 24)
            | (0b11 << 22) // ROR
            | (5 << 16)
            | (7 << 10)
            | (4 << 5)
            | 3;
        let enc = encode_logical_shifted_reg(1, 0b01, 0b11, 0, 5, 7, 4, 3);
        assert_eq!(enc, expected, "ORR X3, X4, X5, ROR #7 = 0x{enc:08X}");
    }

    #[test]
    fn test_and_w0_w1_w2() {
        // AND W0, W1, W2 (32-bit)
        // sf=0, opc=00, shift=00, N=0, Rm=2, imm6=0, Rn=1, Rd=0
        let expected = (0b01010 << 24) | (2 << 16) | (1 << 5);
        let enc = encode_logical_shifted_reg(0, 0b00, 0, 0, 2, 0, 1, 0);
        assert_eq!(enc, expected, "AND W0, W1, W2 = 0x{enc:08X}");
    }

    // -----------------------------------------------------------------------
    // Add/Subtract (immediate)
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_x0_x1_imm42() {
        // ADD X0, X1, #42
        // sf=1, op=0, S=0, sh=0, imm12=42, Rn=1, Rd=0
        // Manual: 1_0_0_100010_0_000000101010_00001_00000
        // Bits: sf=1(31), op=0(30), S=0(29), 100010(28:23), sh=0(22),
        //       imm12=42=0b000000101010(21:10), Rn=1(9:5), Rd=0(4:0)
        let expected = (1u32 << 31)
            | (0b100010 << 23)
            | (42 << 10)
            | (1 << 5);
        let enc = encode_add_sub_imm(1, 0, 0, 0, 42, 1, 0);
        assert_eq!(enc, expected, "ADD X0, X1, #42 = 0x{enc:08X}");
        assert_eq!(enc, 0x9100A820);
    }

    #[test]
    fn test_sub_x0_x1_imm42() {
        // SUB X0, X1, #42
        // sf=1, op=1, S=0, sh=0, imm12=42, Rn=1, Rd=0
        let expected = (1u32 << 31)
            | (1 << 30) // SUB
            | (0b100010 << 23)
            | (42 << 10)
            | (1 << 5);
        let enc = encode_add_sub_imm(1, 1, 0, 0, 42, 1, 0);
        assert_eq!(enc, expected, "SUB X0, X1, #42 = 0x{enc:08X}");
        assert_eq!(enc, 0xD100A820);
    }

    #[test]
    fn test_add_w0_w1_imm1() {
        // ADD W0, W1, #1
        // sf=0, op=0, S=0, sh=0, imm12=1, Rn=1, Rd=0
        let expected = (0b100010u32 << 23) | (1 << 10) | (1 << 5);
        let enc = encode_add_sub_imm(0, 0, 0, 0, 1, 1, 0);
        assert_eq!(enc, expected, "ADD W0, W1, #1 = 0x{enc:08X}");
        assert_eq!(enc, 0x11000420);
    }

    #[test]
    fn test_add_imm_shifted() {
        // ADD X0, X1, #4096 (i.e., #1, LSL #12)
        // sf=1, op=0, S=0, sh=1, imm12=1, Rn=1, Rd=0
        let expected = (1u32 << 31)
            | (0b100010 << 23)
            | (1 << 22) // sh=1
            | (1 << 10) // imm12=1
            | (1 << 5);
        let enc = encode_add_sub_imm(1, 0, 0, 1, 1, 1, 0);
        assert_eq!(enc, expected, "ADD X0, X1, #1, LSL #12 = 0x{enc:08X}");
    }

    #[test]
    fn test_subs_imm_max() {
        // SUBS X0, X1, #4095  (max imm12)
        // sf=1, op=1, S=1, sh=0, imm12=0xFFF, Rn=1, Rd=0
        let expected = (1u32 << 31)
            | (1 << 30)
            | (1 << 29)
            | (0b100010 << 23)
            | (0xFFF << 10)
            | (1 << 5);
        let enc = encode_add_sub_imm(1, 1, 1, 0, 0xFFF, 1, 0);
        assert_eq!(enc, expected, "SUBS X0, X1, #4095 = 0x{enc:08X}");
    }

    #[test]
    fn test_add_imm_sp() {
        // ADD SP, SP, #16 (Rd=31=SP, Rn=31=SP)
        // sf=1, op=0, S=0, sh=0, imm12=16, Rn=31, Rd=31
        let expected = (1u32 << 31)
            | (0b100010 << 23)
            | (16 << 10)
            | (31 << 5)
            | 31;
        let enc = encode_add_sub_imm(1, 0, 0, 0, 16, 31, 31);
        assert_eq!(enc, expected, "ADD SP, SP, #16 = 0x{enc:08X}");
    }

    // -----------------------------------------------------------------------
    // Move Wide (immediate)
    // -----------------------------------------------------------------------

    #[test]
    fn test_movz_x0_0x1234() {
        // MOVZ X0, #0x1234
        // sf=1, opc=10(MOVZ), hw=0, imm16=0x1234, Rd=0
        // Manual: 1_10_100101_00_0001001000110100_00000
        let expected = (1u32 << 31)
            | (0b10 << 29)
            | (0b100101 << 23)
            | (0x1234u32 << 5);
        let enc = encode_move_wide(1, 0b10, 0, 0x1234, 0);
        assert_eq!(enc, expected, "MOVZ X0, #0x1234 = 0x{enc:08X}");
        assert_eq!(enc, 0xD2824680);
    }

    #[test]
    fn test_movk_x0_0x5678_lsl16() {
        // MOVK X0, #0x5678, LSL #16
        // sf=1, opc=11(MOVK), hw=1, imm16=0x5678, Rd=0
        let expected = (1u32 << 31)
            | (0b11 << 29)
            | (0b100101 << 23)
            | (0b01 << 21) // hw=1
            | (0x5678u32 << 5);
        let enc = encode_move_wide(1, 0b11, 1, 0x5678, 0);
        assert_eq!(enc, expected, "MOVK X0, #0x5678, LSL#16 = 0x{enc:08X}");
        assert_eq!(enc, 0xF2AACF00);
    }

    #[test]
    fn test_movn_x0_0() {
        // MOVN X0, #0  (encodes -1 i.e. all 1s)
        // sf=1, opc=00(MOVN), hw=0, imm16=0, Rd=0
        let expected = (1u32 << 31) | (0b100101 << 23);
        let enc = encode_move_wide(1, 0b00, 0, 0, 0);
        assert_eq!(enc, expected, "MOVN X0, #0 = 0x{enc:08X}");
        assert_eq!(enc, 0x92800000);
    }

    #[test]
    fn test_movz_w0_0xffff() {
        // MOVZ W0, #0xFFFF
        // sf=0, opc=10, hw=0, imm16=0xFFFF, Rd=0
        let expected = (0b10u32 << 29)
            | (0b100101 << 23)
            | (0xFFFF << 5);
        let enc = encode_move_wide(0, 0b10, 0, 0xFFFF, 0);
        assert_eq!(enc, expected, "MOVZ W0, #0xFFFF = 0x{enc:08X}");
    }

    #[test]
    fn test_movz_x31() {
        // MOVZ XZR, #0 (writes to zero register — no-op but valid encoding)
        // sf=1, opc=10, hw=0, imm16=0, Rd=31
        let expected = (1u32 << 31) | (0b10 << 29) | (0b100101 << 23) | 31;
        let enc = encode_move_wide(1, 0b10, 0, 0, 31);
        assert_eq!(enc, expected, "MOVZ XZR, #0 = 0x{enc:08X}");
    }

    // -----------------------------------------------------------------------
    // Conditional Branch
    // -----------------------------------------------------------------------

    #[test]
    fn test_b_eq_plus8() {
        // B.EQ <+8>  (2 instructions forward)
        // imm19=2, cond=0000(EQ)
        // Manual: 01010100_0000000000000000010_0_0000
        let expected = (0b01010100u32 << 24) | (2 << 5);
        let enc = encode_cond_branch(2, cond::EQ);
        assert_eq!(enc, expected, "B.EQ <+8> = 0x{enc:08X}");
        assert_eq!(enc, 0x54000040);
    }

    #[test]
    fn test_b_ne() {
        // B.NE <+16> (4 instructions forward)
        // imm19=4, cond=0001(NE)
        let expected = (0b01010100u32 << 24) | (4 << 5) | 1;
        let enc = encode_cond_branch(4, cond::NE);
        assert_eq!(enc, expected, "B.NE <+16> = 0x{enc:08X}");
    }

    #[test]
    fn test_b_al() {
        // B.AL <+4> (1 instruction forward)
        // imm19=1, cond=1110(AL)
        let expected = (0b01010100u32 << 24) | (1 << 5) | 0b1110;
        let enc = encode_cond_branch(1, cond::AL);
        assert_eq!(enc, expected, "B.AL <+4> = 0x{enc:08X}");
    }

    #[test]
    fn test_cond_branch_max_offset() {
        // B.GE with maximum positive imm19
        // imm19 = 0x3FFFF (max positive 19-bit), cond=GE
        let imm = 0x3FFFF;
        let expected = (0b01010100u32 << 24) | (imm << 5) | cond::GE;
        let enc = encode_cond_branch(imm, cond::GE);
        assert_eq!(enc, expected);
    }

    // -----------------------------------------------------------------------
    // Unconditional Branch
    // -----------------------------------------------------------------------

    #[test]
    fn test_b_plus8() {
        // B <+8>
        // op=0, imm26=2
        let expected = (0b00101u32 << 26) | 2;
        let enc = encode_uncond_branch(0, 2);
        assert_eq!(enc, expected, "B <+8> = 0x{enc:08X}");
        assert_eq!(enc, 0x14000002);
    }

    #[test]
    fn test_bl_plus8() {
        // BL <+8>
        // op=1, imm26=2
        let expected = (1u32 << 31) | (0b00101 << 26) | 2;
        let enc = encode_uncond_branch(1, 2);
        assert_eq!(enc, expected, "BL <+8> = 0x{enc:08X}");
        assert_eq!(enc, 0x94000002);
    }

    #[test]
    fn test_b_max_offset() {
        // B with max forward offset
        let imm = 0x3FFFFFF; // 26-bit max
        let expected = (0b00101u32 << 26) | imm;
        let enc = encode_uncond_branch(0, imm);
        assert_eq!(enc, expected);
    }

    // -----------------------------------------------------------------------
    // Branch Register
    // -----------------------------------------------------------------------

    #[test]
    fn test_br_x30() {
        // BR X30
        // opc=0000, Rn=30
        // Manual: 1101011_0000_11111_000000_11110_00000
        let expected = (0b1101011u32 << 25) | (0b11111 << 16) | (30 << 5);
        let enc = encode_branch_reg(0b0000, 30);
        assert_eq!(enc, expected, "BR X30 = 0x{enc:08X}");
        assert_eq!(enc, 0xD61F03C0);
    }

    #[test]
    fn test_blr_x0() {
        // BLR X0
        // opc=0001, Rn=0
        // Manual: 1101011_0001_11111_000000_00000_00000
        let expected = (0b1101011u32 << 25) | (0b0001 << 21) | (0b11111 << 16);
        let enc = encode_branch_reg(0b0001, 0);
        assert_eq!(enc, expected, "BLR X0 = 0x{enc:08X}");
        assert_eq!(enc, 0xD63F0000);
    }

    #[test]
    fn test_ret_x30() {
        // RET (X30)
        // opc=0010, Rn=30
        // Manual: 1101011_0010_11111_000000_11110_00000
        let expected = (0b1101011u32 << 25)
            | (0b0010 << 21)
            | (0b11111 << 16)
            | (30 << 5);
        let enc = encode_branch_reg(0b0010, 30);
        assert_eq!(enc, expected, "RET X30 = 0x{enc:08X}");
        assert_eq!(enc, 0xD65F03C0);
    }

    #[test]
    fn test_blr_x15() {
        // BLR X15
        // opc=0001, Rn=15
        let expected = (0b1101011u32 << 25)
            | (0b0001 << 21)
            | (0b11111 << 16)
            | (15 << 5);
        let enc = encode_branch_reg(0b0001, 15);
        assert_eq!(enc, expected, "BLR X15 = 0x{enc:08X}");
    }

    // -----------------------------------------------------------------------
    // Load/Store Register (unsigned offset)
    // -----------------------------------------------------------------------

    #[test]
    fn test_ldr_x0_x1_8() {
        // LDR X0, [X1, #8]
        // size=11(64-bit), V=0, opc=01(LDR), imm12=1 (8/8=1), Rn=1, Rt=0
        // Manual: 11_111_0_01_01_000000000001_00001_00000
        let expected = (0b11u32 << 30)
            | (0b111 << 27)
            | (0b01 << 24)
            | (0b01 << 22) // LDR
            | (1 << 10) // imm12=1
            | (1 << 5);
        let enc = encode_load_store_ui(0b11, 0, 0b01, 1, 1, 0);
        assert_eq!(enc, expected, "LDR X0, [X1, #8] = 0x{enc:08X}");
        assert_eq!(enc, 0xF9400420);
    }

    #[test]
    fn test_str_x0_x1() {
        // STR X0, [X1]
        // size=11, V=0, opc=00(STR), imm12=0, Rn=1, Rt=0
        let expected = (0b11u32 << 30)
            | (0b111 << 27)
            | (0b01 << 24)
            | (1 << 5);
        let enc = encode_load_store_ui(0b11, 0, 0b00, 0, 1, 0);
        assert_eq!(enc, expected, "STR X0, [X1] = 0x{enc:08X}");
        assert_eq!(enc, 0xF9000020);
    }

    #[test]
    fn test_ldr_w0_x1() {
        // LDR W0, [X1]
        // size=10(32-bit), V=0, opc=01(LDR), imm12=0, Rn=1, Rt=0
        let expected = (0b10u32 << 30)
            | (0b111 << 27)
            | (0b01 << 24)
            | (0b01 << 22)
            | (1 << 5);
        let enc = encode_load_store_ui(0b10, 0, 0b01, 0, 1, 0);
        assert_eq!(enc, expected, "LDR W0, [X1] = 0x{enc:08X}");
        assert_eq!(enc, 0xB9400020);
    }

    #[test]
    fn test_ldrb_w0_x1() {
        // LDRB W0, [X1]
        // size=00(byte), V=0, opc=01, imm12=0, Rn=1, Rt=0
        let expected = (0b111u32 << 27) | (0b01 << 24) | (0b01 << 22) | (1 << 5);
        let enc = encode_load_store_ui(0b00, 0, 0b01, 0, 1, 0);
        assert_eq!(enc, expected, "LDRB W0, [X1] = 0x{enc:08X}");
    }

    #[test]
    fn test_str_x_sp_base() {
        // STR X0, [SP, #0]  (Rn=31=SP)
        // size=11, V=0, opc=00, imm12=0, Rn=31, Rt=0
        let expected = (0b11u32 << 30)
            | (0b111 << 27)
            | (0b01 << 24)
            | (31 << 5);
        let enc = encode_load_store_ui(0b11, 0, 0b00, 0, 31, 0);
        assert_eq!(enc, expected, "STR X0, [SP] = 0x{enc:08X}");
    }

    #[test]
    fn test_ldr_max_offset() {
        // LDR X0, [X1, #32760]  (max for 8-byte: 32760/8 = 4095 = 0xFFF)
        // size=11, V=0, opc=01, imm12=0xFFF, Rn=1, Rt=0
        let expected = (0b11u32 << 30)
            | (0b111 << 27)
            | (0b01 << 24)
            | (0b01 << 22)
            | (0xFFF << 10)
            | (1 << 5);
        let enc = encode_load_store_ui(0b11, 0, 0b01, 0xFFF, 1, 0);
        assert_eq!(enc, expected, "LDR X0, [X1, #32760] = 0x{enc:08X}");
    }

    // -----------------------------------------------------------------------
    // Load/Store Pair (signed offset)
    // -----------------------------------------------------------------------

    #[test]
    fn test_stp_x0_x1_sp_16() {
        // STP X0, X1, [SP, #16]
        // opc=10(64-bit), V=0, L=0(store), imm7=2 (16/8=2), Rt2=1, Rn=31(SP), Rt=0
        // Manual: 10_101_0_010_0_0000010_00001_11111_00000
        let expected = (0b10u32 << 30)
            | (0b101 << 27)
            | (0b010 << 23)
            | (2 << 15) // imm7=2
            | (1 << 10) // Rt2=1
            | (31 << 5); // Rt=0
        let enc = encode_load_store_pair(0b10, 0, 0, 2, 1, 31, 0);
        assert_eq!(enc, expected, "STP X0, X1, [SP, #16] = 0x{enc:08X}");
        assert_eq!(enc, 0xA90107E0);
    }

    #[test]
    fn test_ldp_x0_x1_sp() {
        // LDP X0, X1, [SP]
        // opc=10, V=0, L=1(load), imm7=0, Rt2=1, Rn=31, Rt=0
        let expected = (0b10u32 << 30)
            | (0b101 << 27)
            | (0b010 << 23)
            | (1 << 22) // L=1
            | (1 << 10) // Rt2=1
            | (31 << 5); // Rn=SP
        let enc = encode_load_store_pair(0b10, 0, 1, 0, 1, 31, 0);
        assert_eq!(enc, expected, "LDP X0, X1, [SP] = 0x{enc:08X}");
        assert_eq!(enc, 0xA94007E0);
    }

    #[test]
    fn test_stp_w_pair() {
        // STP W2, W3, [X4, #8]
        // opc=00(32-bit), V=0, L=0, imm7=2 (8/4=2), Rt2=3, Rn=4, Rt=2
        let expected = (0b101u32 << 27)
            | (0b010 << 23)
            | (2 << 15)
            | (3 << 10)
            | (4 << 5)
            | 2;
        let enc = encode_load_store_pair(0b00, 0, 0, 2, 3, 4, 2);
        assert_eq!(enc, expected, "STP W2, W3, [X4, #8] = 0x{enc:08X}");
    }

    #[test]
    fn test_ldp_negative_offset() {
        // LDP X0, X1, [SP, #-16]
        // opc=10, V=0, L=1, imm7 = (-16/8) & 0x7F = (-2) & 0x7F = 0x7E
        // Signed 7-bit: -2 = 0b1111110
        let imm7 = (-2i32 as u32) & 0x7F;
        let expected = (0b10u32 << 30)
            | (0b101 << 27)
            | (0b010 << 23)
            | (1 << 22)
            | (imm7 << 15)
            | (1 << 10)
            | (31 << 5);
        let enc = encode_load_store_pair(0b10, 0, 1, imm7, 1, 31, 0);
        assert_eq!(enc, expected, "LDP X0, X1, [SP, #-16] = 0x{enc:08X}");
    }

    // -----------------------------------------------------------------------
    // Compare and Branch
    // -----------------------------------------------------------------------

    #[test]
    fn test_cbz_x0_plus8() {
        // CBZ X0, <+8>
        // sf=1, op=0(CBZ), imm19=2, Rt=0
        // Manual: 1_011010_0_0000000000000000010_00000
        let expected = (1u32 << 31) | (0b011010 << 25) | (2 << 5);
        let enc = encode_cmp_branch(1, 0, 2, 0);
        assert_eq!(enc, expected, "CBZ X0, <+8> = 0x{enc:08X}");
        assert_eq!(enc, 0xB4000040);
    }

    #[test]
    fn test_cbnz_w0_plus8() {
        // CBNZ W0, <+8>
        // sf=0, op=1(CBNZ), imm19=2, Rt=0
        let expected = (0b011010u32 << 25) | (1 << 24) | (2 << 5);
        let enc = encode_cmp_branch(0, 1, 2, 0);
        assert_eq!(enc, expected, "CBNZ W0, <+8> = 0x{enc:08X}");
        assert_eq!(enc, 0x35000040);
    }

    #[test]
    fn test_cbz_w31() {
        // CBZ WZR, <+4> (Rt=31 = WZR in 32-bit)
        // sf=0, op=0, imm19=1, Rt=31
        let expected = (0b011010u32 << 25) | (1 << 5) | 31;
        let enc = encode_cmp_branch(0, 0, 1, 31);
        assert_eq!(enc, expected, "CBZ WZR, <+4> = 0x{enc:08X}");
    }

    #[test]
    fn test_cbnz_max_offset() {
        // CBNZ X0, max forward offset
        let imm = 0x3FFFF; // max positive 19-bit (bit 18 is 0 in unsigned view)
        let expected = (1u32 << 31) | (0b011010 << 25) | (1 << 24) | (imm << 5);
        let enc = encode_cmp_branch(1, 1, imm, 0);
        assert_eq!(enc, expected);
    }

    // -----------------------------------------------------------------------
    // Condition code module
    // -----------------------------------------------------------------------

    #[test]
    fn test_condition_codes() {
        assert_eq!(cond::EQ, 0);
        assert_eq!(cond::NE, 1);
        assert_eq!(cond::CS, cond::HS);
        assert_eq!(cond::CC, cond::LO);
        assert_eq!(cond::MI, 4);
        assert_eq!(cond::PL, 5);
        assert_eq!(cond::VS, 6);
        assert_eq!(cond::VC, 7);
        assert_eq!(cond::HI, 8);
        assert_eq!(cond::LS, 9);
        assert_eq!(cond::GE, 10);
        assert_eq!(cond::LT, 11);
        assert_eq!(cond::GT, 12);
        assert_eq!(cond::LE, 13);
        assert_eq!(cond::AL, 14);
        assert_eq!(cond::NV, 15);
    }

    // -----------------------------------------------------------------------
    // Shift type module
    // -----------------------------------------------------------------------

    #[test]
    fn test_shift_types() {
        assert_eq!(shift::LSL, 0);
        assert_eq!(shift::LSR, 1);
        assert_eq!(shift::ASR, 2);
        assert_eq!(shift::ROR, 3);
    }

    // -----------------------------------------------------------------------
    // Cross-format sanity: bit 28:24 fixed fields
    // -----------------------------------------------------------------------

    #[test]
    fn test_fixed_bits_add_sub_shifted() {
        // Bits 28:24 must be 01011 for add/sub shifted register
        let enc = encode_add_sub_shifted_reg(0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!((enc >> 24) & 0x1F, 0b01011);
    }

    #[test]
    fn test_fixed_bits_logical_shifted() {
        // Bits 28:24 must be 01010 for logical shifted register
        let enc = encode_logical_shifted_reg(0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!((enc >> 24) & 0x1F, 0b01010);
    }

    #[test]
    fn test_fixed_bits_add_sub_imm() {
        // Bits 28:23 must be 100010 for add/sub immediate
        let enc = encode_add_sub_imm(0, 0, 0, 0, 0, 0, 0);
        assert_eq!((enc >> 23) & 0x3F, 0b100010);
    }

    #[test]
    fn test_fixed_bits_move_wide() {
        // Bits 28:23 must be 100101 for move wide
        let enc = encode_move_wide(0, 0b10, 0, 0, 0);
        assert_eq!((enc >> 23) & 0x3F, 0b100101);
    }

    #[test]
    fn test_fixed_bits_cond_branch() {
        // Bits 31:24 must be 01010100 for conditional branch
        let enc = encode_cond_branch(0, 0);
        assert_eq!((enc >> 24) & 0xFF, 0b01010100);
    }

    #[test]
    fn test_fixed_bits_uncond_branch() {
        // Bits 30:26 must be 00101 for unconditional branch
        let enc = encode_uncond_branch(0, 0);
        assert_eq!((enc >> 26) & 0x1F, 0b00101);
    }

    #[test]
    fn test_fixed_bits_branch_reg() {
        // Bits 31:25 must be 1101011, bits 20:16 must be 11111,
        // bits 15:10 must be 000000, bits 4:0 must be 00000
        let enc = encode_branch_reg(0, 0);
        assert_eq!((enc >> 25) & 0x7F, 0b1101011);
        assert_eq!((enc >> 16) & 0x1F, 0b11111);
        assert_eq!((enc >> 10) & 0x3F, 0);
        assert_eq!(enc & 0x1F, 0);
    }

    #[test]
    fn test_fixed_bits_load_store_ui() {
        // Bits 29:27 must be 111, bits 25:24 must be 01
        let enc = encode_load_store_ui(0, 0, 0, 0, 0, 0);
        assert_eq!((enc >> 27) & 0x7, 0b111);
        assert_eq!((enc >> 24) & 0x3, 0b01);
    }

    #[test]
    fn test_fixed_bits_load_store_pair() {
        // Bits 29:27 must be 101, bits 25:23 must be 010
        let enc = encode_load_store_pair(0, 0, 0, 0, 0, 0, 0);
        assert_eq!((enc >> 27) & 0x7, 0b101);
        assert_eq!((enc >> 23) & 0x7, 0b010);
    }

    #[test]
    fn test_fixed_bits_cmp_branch() {
        // Bits 30:25 must be 011010
        let enc = encode_cmp_branch(0, 0, 0, 0);
        assert_eq!((enc >> 25) & 0x3F, 0b011010);
    }
}
