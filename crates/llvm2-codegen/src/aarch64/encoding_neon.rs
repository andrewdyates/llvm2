// llvm2-codegen/aarch64/encoding_neon.rs - AArch64 NEON SIMD instruction encoding
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! AArch64 NEON (Advanced SIMD) instruction encoding.
//!
//! Implements encoding for AArch64 NEON instruction formats from the
//! ARM Architecture Reference Manual (DDI 0487):
//!
//! - **Three-register same**: vector arithmetic (ADD, SUB, MUL, FADD, FSUB, FMUL, FDIV),
//!   logic (AND, ORR, EOR, BIC), compare (CMEQ, CMGT, CMGE)
//! - **Two-register misc**: NOT (bitwise NOT / MVN)
//! - **Modified immediate**: MOVI (move immediate to vector)
//! - **Vector duplicate**: DUP (scalar to vector), INS (insert element)
//! - **SIMD load/store**: LD1, ST1 (single-structure, post-index)
//!
//! Reference: ARM ARM C7.2.x, `~/llvm-project-ref/llvm/lib/Target/AArch64/AArch64InstrFormats.td`

use thiserror::Error;

/// Errors produced during NEON instruction encoding.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum NeonEncodeError {
    #[error("register index {reg} out of range (max 31)")]
    RegisterOutOfRange { reg: u8 },
    #[error("invalid arrangement size {0}")]
    InvalidSize(u8),
    #[error("invalid lane index {lane} for arrangement")]
    InvalidLane { lane: u8 },
    #[error("immediate {0:#X} out of range for MOVI")]
    ImmediateOutOfRange(u64),
}

// ---------------------------------------------------------------------------
// Arrangement / size enums
// ---------------------------------------------------------------------------

/// NEON vector arrangement (Q and size fields).
///
/// Determines element size and whether the instruction operates on
/// 64-bit (D-register) or 128-bit (Q-register / V-register) vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorArrangement {
    /// 8B:  8 x 8-bit  elements in 64-bit register (Q=0, size=00)
    B8,
    /// 16B: 16 x 8-bit elements in 128-bit register (Q=1, size=00)
    B16,
    /// 4H:  4 x 16-bit elements in 64-bit register (Q=0, size=01)
    H4,
    /// 8H:  8 x 16-bit elements in 128-bit register (Q=1, size=01)
    H8,
    /// 2S:  2 x 32-bit elements in 64-bit register (Q=0, size=10)
    S2,
    /// 4S:  4 x 32-bit elements in 128-bit register (Q=1, size=10)
    S4,
    /// 2D:  2 x 64-bit elements in 128-bit register (Q=1, size=11)
    D2,
}

impl VectorArrangement {
    /// Returns (Q, size) fields for this arrangement.
    pub fn q_size(self) -> (u32, u32) {
        match self {
            Self::B8  => (0, 0b00),
            Self::B16 => (1, 0b00),
            Self::H4  => (0, 0b01),
            Self::H8  => (1, 0b01),
            Self::S2  => (0, 0b10),
            Self::S4  => (1, 0b10),
            Self::D2  => (1, 0b11),
        }
    }

    /// Returns the Q bit for this arrangement.
    pub fn q(self) -> u32 {
        self.q_size().0
    }

    /// Returns the size field for this arrangement.
    pub fn size(self) -> u32 {
        self.q_size().1
    }
}

/// FP vector arrangement (for FADD/FSUB/FMUL/FDIV).
///
/// AArch64 NEON FP instructions use `sz` (1 bit) instead of `size` (2 bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpVectorArrangement {
    /// 2S: 2 x f32 in 64-bit register (Q=0, sz=0)
    S2,
    /// 4S: 4 x f32 in 128-bit register (Q=1, sz=0)
    S4,
    /// 2D: 2 x f64 in 128-bit register (Q=1, sz=1)
    D2,
}

impl FpVectorArrangement {
    /// Returns (Q, sz) for this FP arrangement.
    pub fn q_sz(self) -> (u32, u32) {
        match self {
            Self::S2 => (0, 0),
            Self::S4 => (1, 0),
            Self::D2 => (1, 1),
        }
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

#[inline]
fn check_reg(reg: u8) -> Result<(), NeonEncodeError> {
    if reg > 31 {
        return Err(NeonEncodeError::RegisterOutOfRange { reg });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Three-register same: integer vector arithmetic
// ARM ARM C7.2.x: Advanced SIMD three same
//
//   0 | Q | U | 01110 | size(2) | 1 | Rm(5) | opcode(5) | 1 | Rn(5) | Rd(5)
//
// ---------------------------------------------------------------------------

/// Integer vector three-register-same opcode (5-bit `opcode` field, bits [15:11]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntVec3Op {
    /// ADD: opcode = 10000, U = 0
    Add,
    /// SUB: opcode = 10000, U = 1
    Sub,
    /// MUL: opcode = 10011, U = 0
    Mul,
    /// CMEQ (register): opcode = 10001, U = 1
    Cmeq,
    /// CMGT (signed): opcode = 00110, U = 0
    Cmgt,
    /// CMGE (signed): opcode = 00111, U = 0
    Cmge,
}

impl IntVec3Op {
    /// Returns (U, opcode) for the instruction.
    fn u_opcode(self) -> (u32, u32) {
        match self {
            Self::Add  => (0, 0b10000),
            Self::Sub  => (1, 0b10000),
            Self::Mul  => (0, 0b10011),
            Self::Cmeq => (1, 0b10001),
            Self::Cmgt => (0, 0b00110),
            Self::Cmge => (0, 0b00111),
        }
    }
}

/// Encode a NEON integer three-register-same instruction.
///
/// Format: `0 | Q | U | 01110 | size(2) | 1 | Rm(5) | opcode(5) | 1 | Rn(5) | Rd(5)`
///
/// ARM ARM: C7.2.1 (ADD vector), C7.2.299 (SUB vector), C7.2.211 (MUL vector)
pub fn encode_int_vec3_same(
    arr: VectorArrangement,
    op: IntVec3Op,
    rm: u8,
    rn: u8,
    rd: u8,
) -> Result<u32, NeonEncodeError> {
    check_reg(rm)?;
    check_reg(rn)?;
    check_reg(rd)?;

    let (q, size) = arr.q_size();
    let (u, opcode) = op.u_opcode();

    Ok((q << 30)
        | (u << 29)
        | (0b01110 << 24)
        | (size << 22)
        | (1 << 21)
        | ((rm as u32) << 16)
        | (opcode << 11)
        | (1 << 10)
        | ((rn as u32) << 5)
        | (rd as u32))
}

// ---------------------------------------------------------------------------
// Three-register same: logic vector (AND, ORR, EOR, BIC)
// ARM ARM C7.2.x: Advanced SIMD three same
//
//   0 | Q | op2(2) | 01110 | size(2) | 1 | Rm(5) | 00011 | 1 | Rn(5) | Rd(5)
//
// Logic instructions use a special encoding:
//   AND: 0|Q|0|01110|00|1|Rm|00011|1|Rn|Rd
//   ORR: 0|Q|0|01110|10|1|Rm|00011|1|Rn|Rd
//   EOR: 0|Q|1|01110|00|1|Rm|00011|1|Rn|Rd
//   BIC: 0|Q|0|01110|01|1|Rm|00011|1|Rn|Rd
// ---------------------------------------------------------------------------

/// Vector logic operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecLogicOp {
    /// AND vector: U=0, size=00
    And,
    /// ORR vector: U=0, size=10
    Orr,
    /// EOR vector: U=1, size=00
    Eor,
    /// BIC vector (AND-NOT): U=0, size=01
    Bic,
}

impl VecLogicOp {
    /// Returns (U, size) for the logic instruction.
    fn u_size(self) -> (u32, u32) {
        match self {
            Self::And => (0, 0b00),
            Self::Orr => (0, 0b10),
            Self::Eor => (1, 0b00),
            Self::Bic => (0, 0b01),
        }
    }
}

/// Encode a NEON vector logic instruction (AND, ORR, EOR, BIC).
///
/// Format: `0 | Q | U | 01110 | size(2) | 1 | Rm(5) | 00011 | 1 | Rn(5) | Rd(5)`
///
/// ARM ARM: C7.2.8 (AND vector), C7.2.219 (ORR vector), C7.2.87 (EOR vector)
pub fn encode_vec_logic(
    q: u32,
    op: VecLogicOp,
    rm: u8,
    rn: u8,
    rd: u8,
) -> Result<u32, NeonEncodeError> {
    check_reg(rm)?;
    check_reg(rn)?;
    check_reg(rd)?;

    let (u, size) = op.u_size();

    Ok((q << 30)
        | (u << 29)
        | (0b01110 << 24)
        | (size << 22)
        | (1 << 21)
        | ((rm as u32) << 16)
        | (0b00011 << 11)
        | (1 << 10)
        | ((rn as u32) << 5)
        | (rd as u32))
}

// ---------------------------------------------------------------------------
// Two-register misc: NOT (bitwise NOT / MVN)
// ARM ARM C7.2.216: NOT (vector)
//
//   0 | Q | 1 | 01110 | 00 | 10000 | 00101 | 10 | Rn(5) | Rd(5)
//
// This is actually encoded as: 2Q|1|01110|size=00|10000|opcode=00101|10|Rn|Rd
// NOT is an alias for MVN: Q|U=1|01110|00|10000|00101|10|Rn|Rd
// ---------------------------------------------------------------------------

/// Encode a NEON NOT (bitwise NOT / MVN) instruction.
///
/// Format: `0 | Q | 1 | 01110 | 00 | 10000 | 00101 | 10 | Rn(5) | Rd(5)`
///
/// ARM ARM: C7.2.216 NOT (vector)
pub fn encode_vec_not(
    q: u32,
    rn: u8,
    rd: u8,
) -> Result<u32, NeonEncodeError> {
    check_reg(rn)?;
    check_reg(rd)?;

    // NOT is MVN (alias): 0|Q|1|01110|00|10000|00101|10|Rn|Rd
    Ok(((q << 30)
        | (1 << 29)       // U = 1
        | (0b01110 << 24))    // size = 00
        | (0b10000 << 17)
        | (0b00101 << 12)
        | (0b10 << 10)
        | ((rn as u32) << 5)
        | (rd as u32))
}

// ---------------------------------------------------------------------------
// Three-register same: FP vector arithmetic (FADD, FSUB, FMUL, FDIV)
// ARM ARM: Advanced SIMD three same (FP)
//
//   0 | Q | U | 01110 | 0 | sz | 1 | Rm(5) | opcode(3) | 0 | 1 | Rn(5) | Rd(5)
//
// FADD: U=0, opcode=11010 → bits[15:11] = 11010
// FSUB: U=0, opcode=11010 → same but U=1
// Actually per ARM ARM:
//   FADD: 0|Q|0|01110|0|sz|1|Rm|110101|Rn|Rd
//   FSUB: 0|Q|0|01110|1|sz|1|Rm|110101|Rn|Rd
//   FMUL: 0|Q|1|01110|0|sz|1|Rm|110111|Rn|Rd
//   FDIV: 0|Q|1|01110|0|sz|1|Rm|111111|Rn|Rd
// ---------------------------------------------------------------------------

/// FP vector three-register-same operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpVec3Op {
    /// FADD vector
    Fadd,
    /// FSUB vector
    Fsub,
    /// FMUL vector
    Fmul,
    /// FDIV vector
    Fdiv,
}

impl FpVec3Op {
    /// Returns (U, bit23, opcode_bits[15:10]) for encoding.
    ///
    /// The FP three-same encoding uses:
    ///   bit 29 = U, bit 23 = extra distinguisher, bits [15:10] = opcode|1
    fn u_bit23_opcode(self) -> (u32, u32, u32) {
        match self {
            //               U    bit23  bits[15:10]
            Self::Fadd => (0,   0,     0b110101),
            Self::Fsub => (0,   1,     0b110101),
            Self::Fmul => (1,   0,     0b110111),
            Self::Fdiv => (1,   0,     0b111111),
        }
    }
}

/// Encode a NEON FP three-register-same instruction.
///
/// Format: `0 | Q | U | 01110 | bit23 | sz | 1 | Rm(5) | opcode(6) | Rn(5) | Rd(5)`
///
/// ARM ARM: C7.2.93 (FADD vector), C7.2.118 (FSUB vector),
///          C7.2.114 (FMUL vector), C7.2.97 (FDIV vector)
pub fn encode_fp_vec3_same(
    arr: FpVectorArrangement,
    op: FpVec3Op,
    rm: u8,
    rn: u8,
    rd: u8,
) -> Result<u32, NeonEncodeError> {
    check_reg(rm)?;
    check_reg(rn)?;
    check_reg(rd)?;

    let (q, sz) = arr.q_sz();
    let (u, bit23, opcode) = op.u_bit23_opcode();

    Ok((q << 30)
        | (u << 29)
        | (0b01110 << 24)
        | (bit23 << 23)
        | (sz << 22)
        | (1 << 21)
        | ((rm as u32) << 16)
        | (opcode << 10)
        | ((rn as u32) << 5)
        | (rd as u32))
}

// ---------------------------------------------------------------------------
// DUP (element, scalar to vector)
// ARM ARM C7.2.82: DUP (element)
//
//   0 | Q | 0 | 01110000 | imm5(5) | 0 | 0000 | 1 | Rn(5) | Rd(5)
//
// imm5 encodes which lane and element size:
//   B: imm5 = xxxx1, H: imm5 = xxx10, S: imm5 = xx100, D: imm5 = x1000
// DUP (general): duplicates a GPR into all vector lanes
//   0 | Q | 0 | 01110000 | imm5(5) | 0 | 0011 | 1 | Rn(5) | Rd(5)
// ---------------------------------------------------------------------------

/// Element size for DUP / INS operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementSize {
    B = 1,   // imm5[0] = 1
    H = 2,   // imm5[1:0] = 10
    S = 4,   // imm5[2:0] = 100
    D = 8,   // imm5[3:0] = 1000
}

/// Encode DUP (element) - duplicate a vector element to all lanes.
///
/// `lane` specifies which element of `Rn` to duplicate.
///
/// ARM ARM C7.2.82
pub fn encode_dup_element(
    q: u32,
    elem: ElementSize,
    lane: u8,
    rn: u8,
    rd: u8,
) -> Result<u32, NeonEncodeError> {
    check_reg(rn)?;
    check_reg(rd)?;

    let imm5 = match elem {
        ElementSize::B => ((lane as u32) << 1) | 0b1,
        ElementSize::H => ((lane as u32) << 2) | 0b10,
        ElementSize::S => ((lane as u32) << 3) | 0b100,
        ElementSize::D => ((lane as u32) << 4) | 0b1000,
    };

    if imm5 > 0b11111 {
        return Err(NeonEncodeError::InvalidLane { lane });
    }

    // DUP (element): 0|Q|0|01110000|imm5|0|0000|1|Rn|Rd
    Ok(((q << 30)
        | (0b001110000 << 21)
        | (imm5 << 16))
        | (1 << 10)
        | ((rn as u32) << 5)
        | (rd as u32))
}

/// Encode DUP (general) - duplicate a GPR value to all vector lanes.
///
/// ARM ARM C7.2.83
pub fn encode_dup_general(
    q: u32,
    elem: ElementSize,
    rn: u8,
    rd: u8,
) -> Result<u32, NeonEncodeError> {
    check_reg(rn)?;
    check_reg(rd)?;

    let imm5: u32 = match elem {
        ElementSize::B => 0b00001,
        ElementSize::H => 0b00010,
        ElementSize::S => 0b00100,
        ElementSize::D => 0b01000,
    };

    // DUP (general): 0|Q|0|01110000|imm5|0|0011|1|Rn|Rd
    Ok(((q << 30)
        | (0b001110000 << 21)
        | (imm5 << 16))
        | (0b0011 << 11)
        | (1 << 10)
        | ((rn as u32) << 5)
        | (rd as u32))
}

// ---------------------------------------------------------------------------
// INS (element, insert from GPR)
// ARM ARM C7.2.152: INS (general)
//
//   0 | 1 | 0 | 01110000 | imm5(5) | 0 | 0011 | 1 | Rn(5) | Rd(5)
//
// imm5 encodes the destination lane + element size.
// ---------------------------------------------------------------------------

/// Encode INS (general) - insert a GPR value into a vector lane.
///
/// ARM ARM C7.2.152
pub fn encode_ins_general(
    elem: ElementSize,
    lane: u8,
    rn: u8,
    rd: u8,
) -> Result<u32, NeonEncodeError> {
    check_reg(rn)?;
    check_reg(rd)?;

    let imm5 = match elem {
        ElementSize::B => ((lane as u32) << 1) | 0b1,
        ElementSize::H => ((lane as u32) << 2) | 0b10,
        ElementSize::S => ((lane as u32) << 3) | 0b100,
        ElementSize::D => ((lane as u32) << 4) | 0b1000,
    };

    if imm5 > 0b11111 {
        return Err(NeonEncodeError::InvalidLane { lane });
    }

    // INS (general): 0|1|0|01110000|imm5|0|0011|1|Rn|Rd
    // Q=1 always for INS (operates on full 128-bit register)
    Ok(((1 << 30)
        | (0b01110000 << 21)
        | (imm5 << 16))
        | (0b0011 << 11)
        | (1 << 10)
        | ((rn as u32) << 5)
        | (rd as u32))
}

// ---------------------------------------------------------------------------
// MOVI (move immediate to vector)
// ARM ARM C7.2.207: MOVI
//
// Simplified 8-bit immediate form (cmode = 1110, op = 0):
//   0 | Q | op | 0111100000 | abc(3) | cmode(4) | 01 | defgh(5) | Rd(5)
//
// The 8-bit immediate is split: abc = imm8[7:5], defgh = imm8[4:0]
// Full byte-replication: MOVI Vd.{8B,16B}, #imm8
// ---------------------------------------------------------------------------

/// Encode MOVI (vector, 8-bit immediate replicated to all byte lanes).
///
/// This encodes the simplest MOVI form: `MOVI Vd.{8B,16B}, #imm8`
/// using cmode=1110, op=0 (byte mask / 8-bit immediate to all bytes).
///
/// ARM ARM C7.2.207
pub fn encode_movi_byte(
    q: u32,
    imm8: u8,
    rd: u8,
) -> Result<u32, NeonEncodeError> {
    check_reg(rd)?;

    let abc = ((imm8 as u32) >> 5) & 0b111;
    let defgh = (imm8 as u32) & 0b11111;

    // MOVI (byte): 0|Q|op=0|0111100000|abc|cmode=1110|o2=0|1|defgh|Rd
    Ok((q << 30)        // op = 0
        | (0b0111100000 << 19)
        | (abc << 16)
        | (0b1110 << 12)     // cmode = 1110 (byte mask)
        | (0b01 << 10)
        | (defgh << 5)
        | (rd as u32))
}

// ---------------------------------------------------------------------------
// LD1 / ST1 (single structure, post-index)
// ARM ARM C7.2.167: LD1 (single structure)
// ARM ARM C7.2.282: ST1 (single structure)
//
// No-offset (single, 1 register):
//   0 | Q | 0011000 | L | 0 | 00000 | opcode(4) | size(2) | Rn(5) | Rt(5)
//   L=1 for LD1, L=0 for ST1
//
// Post-index (single, 1 register, immediate):
//   0 | Q | 0011001 | L | 0 | 11111 | opcode(4) | size(2) | Rn(5) | Rt(5)
//   The post-index immediate = 8 (for 8B) or 16 (for 16B), etc.
//
// For multiple structures, the encoding differs. We use the simple
// single-register form: LD1 {Vt.T}, [Xn], #bytes
// ---------------------------------------------------------------------------

/// Encode LD1 (single structure, 1 register, post-index by immediate).
///
/// `arr` determines the vector arrangement and post-index amount.
///
/// ARM ARM C7.2.167
pub fn encode_ld1_post_imm(
    arr: VectorArrangement,
    rn: u8,
    rt: u8,
) -> Result<u32, NeonEncodeError> {
    check_reg(rn)?;
    check_reg(rt)?;

    let (q, size) = arr.q_size();

    // LD1 single register, post-index immediate:
    // 0|Q|0011001|L=1|0|11111|opcode=0111|size|Rn|Rt
    Ok(((q << 30)
        | (0b0011001 << 23)
        | (1 << 22))
        | (0b11111 << 16)   // Rm = 11111 (immediate post-index)
        | (0b0111 << 12)    // opcode = 0111 (1 register)
        | (size << 10)
        | ((rn as u32) << 5)
        | (rt as u32))
}

/// Encode ST1 (single structure, 1 register, post-index by immediate).
///
/// ARM ARM C7.2.282
pub fn encode_st1_post_imm(
    arr: VectorArrangement,
    rn: u8,
    rt: u8,
) -> Result<u32, NeonEncodeError> {
    check_reg(rn)?;
    check_reg(rt)?;

    let (q, size) = arr.q_size();

    // ST1 single register, post-index immediate:
    // 0|Q|0011001|L=0|0|11111|opcode=0111|size|Rn|Rt
    Ok(((q << 30)
        | (0b0011001 << 23))
        | (0b11111 << 16)   // Rm = 11111 (immediate post-index)
        | (0b0111 << 12)    // opcode = 0111 (1 register)
        | (size << 10)
        | ((rn as u32) << 5)
        | (rt as u32))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // === Integer vector arithmetic ===

    #[test]
    fn test_add_vec_4s() {
        // ADD V0.4S, V1.4S, V2.4S
        // Q=1, U=0, size=10, opcode=10000
        // 0|1|0|01110|10|1|00010|10000|1|00001|00000
        let enc = encode_int_vec3_same(VectorArrangement::S4, IntVec3Op::Add, 2, 1, 0).unwrap();
        assert_eq!(enc & (0b111 << 29), 0b010 << 29, "Q=1,U=0");
        assert_eq!((enc >> 24) & 0b11111, 0b01110);
        assert_eq!((enc >> 22) & 0b11, 0b10, "size=10");
        assert_eq!((enc >> 21) & 1, 1);
        assert_eq!((enc >> 16) & 0b11111, 2, "Rm=2");
        assert_eq!((enc >> 11) & 0b11111, 0b10000, "opcode=10000");
        assert_eq!((enc >> 10) & 1, 1);
        assert_eq!((enc >> 5) & 0b11111, 1, "Rn=1");
        assert_eq!(enc & 0b11111, 0, "Rd=0");
    }

    #[test]
    fn test_sub_vec_8h() {
        // SUB V3.8H, V4.8H, V5.8H
        let enc = encode_int_vec3_same(VectorArrangement::H8, IntVec3Op::Sub, 5, 4, 3).unwrap();
        assert_eq!((enc >> 30) & 1, 1, "Q=1");
        assert_eq!((enc >> 29) & 1, 1, "U=1 for SUB");
        assert_eq!((enc >> 22) & 0b11, 0b01, "size=01 for H");
        assert_eq!((enc >> 11) & 0b11111, 0b10000, "opcode=10000");
    }

    #[test]
    fn test_mul_vec_4s() {
        // MUL V0.4S, V1.4S, V2.4S
        let enc = encode_int_vec3_same(VectorArrangement::S4, IntVec3Op::Mul, 2, 1, 0).unwrap();
        assert_eq!((enc >> 29) & 1, 0, "U=0 for MUL");
        assert_eq!((enc >> 11) & 0b11111, 0b10011, "opcode=10011 for MUL");
    }

    #[test]
    fn test_add_vec_16b() {
        // ADD V0.16B, V1.16B, V2.16B
        let enc = encode_int_vec3_same(VectorArrangement::B16, IntVec3Op::Add, 2, 1, 0).unwrap();
        assert_eq!((enc >> 30) & 1, 1, "Q=1 for 16B");
        assert_eq!((enc >> 22) & 0b11, 0b00, "size=00 for B");
    }

    #[test]
    fn test_sub_vec_8b() {
        // SUB V0.8B, V1.8B, V2.8B
        let enc = encode_int_vec3_same(VectorArrangement::B8, IntVec3Op::Sub, 2, 1, 0).unwrap();
        assert_eq!((enc >> 30) & 1, 0, "Q=0 for 8B");
        assert_eq!((enc >> 29) & 1, 1, "U=1 for SUB");
    }

    #[test]
    fn test_add_vec_2d() {
        // ADD V10.2D, V11.2D, V12.2D
        let enc = encode_int_vec3_same(VectorArrangement::D2, IntVec3Op::Add, 12, 11, 10).unwrap();
        assert_eq!((enc >> 30) & 1, 1, "Q=1 for 2D");
        assert_eq!((enc >> 22) & 0b11, 0b11, "size=11 for D");
        assert_eq!((enc >> 16) & 0b11111, 12);
        assert_eq!((enc >> 5) & 0b11111, 11);
        assert_eq!(enc & 0b11111, 10);
    }

    // === Compare ===

    #[test]
    fn test_cmeq_vec_4s() {
        // CMEQ V0.4S, V1.4S, V2.4S
        let enc = encode_int_vec3_same(VectorArrangement::S4, IntVec3Op::Cmeq, 2, 1, 0).unwrap();
        assert_eq!((enc >> 29) & 1, 1, "U=1 for CMEQ");
        assert_eq!((enc >> 11) & 0b11111, 0b10001, "opcode for CMEQ");
    }

    #[test]
    fn test_cmgt_vec_4s() {
        let enc = encode_int_vec3_same(VectorArrangement::S4, IntVec3Op::Cmgt, 2, 1, 0).unwrap();
        assert_eq!((enc >> 29) & 1, 0, "U=0 for CMGT");
        assert_eq!((enc >> 11) & 0b11111, 0b00110, "opcode for CMGT");
    }

    #[test]
    fn test_cmge_vec_4s() {
        let enc = encode_int_vec3_same(VectorArrangement::S4, IntVec3Op::Cmge, 2, 1, 0).unwrap();
        assert_eq!((enc >> 29) & 1, 0, "U=0 for CMGE");
        assert_eq!((enc >> 11) & 0b11111, 0b00111, "opcode for CMGE");
    }

    // === Vector logic ===

    #[test]
    fn test_and_vec() {
        // AND V0.16B, V1.16B, V2.16B
        let enc = encode_vec_logic(1, VecLogicOp::And, 2, 1, 0).unwrap();
        assert_eq!((enc >> 29) & 1, 0, "U=0 for AND");
        assert_eq!((enc >> 22) & 0b11, 0b00, "size=00 for AND");
        assert_eq!((enc >> 11) & 0b11111, 0b00011, "opcode=00011");
    }

    #[test]
    fn test_orr_vec() {
        let enc = encode_vec_logic(1, VecLogicOp::Orr, 2, 1, 0).unwrap();
        assert_eq!((enc >> 22) & 0b11, 0b10, "size=10 for ORR");
    }

    #[test]
    fn test_eor_vec() {
        let enc = encode_vec_logic(1, VecLogicOp::Eor, 2, 1, 0).unwrap();
        assert_eq!((enc >> 29) & 1, 1, "U=1 for EOR");
        assert_eq!((enc >> 22) & 0b11, 0b00, "size=00 for EOR");
    }

    #[test]
    fn test_bic_vec() {
        let enc = encode_vec_logic(1, VecLogicOp::Bic, 2, 1, 0).unwrap();
        assert_eq!((enc >> 22) & 0b11, 0b01, "size=01 for BIC");
    }

    // === NOT ===

    #[test]
    fn test_not_vec_16b() {
        // NOT V0.16B, V1.16B
        let enc = encode_vec_not(1, 1, 0).unwrap();
        assert_eq!((enc >> 30) & 1, 1, "Q=1");
        assert_eq!((enc >> 29) & 1, 1, "U=1");
        assert_eq!((enc >> 12) & 0b11111, 0b00101, "opcode=00101");
        assert_eq!((enc >> 5) & 0b11111, 1, "Rn=1");
        assert_eq!(enc & 0b11111, 0, "Rd=0");
    }

    // === FP vector arithmetic ===

    #[test]
    fn test_fadd_vec_4s() {
        // FADD V0.4S, V1.4S, V2.4S
        let enc = encode_fp_vec3_same(FpVectorArrangement::S4, FpVec3Op::Fadd, 2, 1, 0).unwrap();
        assert_eq!((enc >> 30) & 1, 1, "Q=1 for 4S");
        assert_eq!((enc >> 29) & 1, 0, "U=0 for FADD");
        assert_eq!((enc >> 22) & 1, 0, "sz=0 for single");
        assert_eq!((enc >> 10) & 0b111111, 0b110101, "opcode for FADD");
    }

    #[test]
    fn test_fsub_vec_4s() {
        let enc = encode_fp_vec3_same(FpVectorArrangement::S4, FpVec3Op::Fsub, 2, 1, 0).unwrap();
        assert_eq!((enc >> 23) & 1, 1, "bit23=1 for FSUB (distinguishes from FADD)");
        assert_eq!((enc >> 10) & 0b111111, 0b110101, "opcode same as FADD");
    }

    #[test]
    fn test_fmul_vec_2d() {
        // FMUL V0.2D, V1.2D, V2.2D
        let enc = encode_fp_vec3_same(FpVectorArrangement::D2, FpVec3Op::Fmul, 2, 1, 0).unwrap();
        assert_eq!((enc >> 30) & 1, 1, "Q=1 for 2D");
        assert_eq!((enc >> 29) & 1, 1, "U=1 for FMUL");
        assert_eq!((enc >> 22) & 1, 1, "sz=1 for double");
        assert_eq!((enc >> 10) & 0b111111, 0b110111, "opcode for FMUL");
    }

    #[test]
    fn test_fdiv_vec_4s() {
        let enc = encode_fp_vec3_same(FpVectorArrangement::S4, FpVec3Op::Fdiv, 2, 1, 0).unwrap();
        assert_eq!((enc >> 29) & 1, 1, "U=1 for FDIV");
        assert_eq!((enc >> 10) & 0b111111, 0b111111, "opcode for FDIV");
    }

    #[test]
    fn test_fadd_vec_2s() {
        // FADD V0.2S, V1.2S, V2.2S (64-bit register)
        let enc = encode_fp_vec3_same(FpVectorArrangement::S2, FpVec3Op::Fadd, 2, 1, 0).unwrap();
        assert_eq!((enc >> 30) & 1, 0, "Q=0 for 2S");
        assert_eq!((enc >> 22) & 1, 0, "sz=0 for single");
    }

    // === DUP ===

    #[test]
    fn test_dup_element_s() {
        // DUP V0.4S, V1.S[2]
        let enc = encode_dup_element(1, ElementSize::S, 2, 1, 0).unwrap();
        assert_eq!((enc >> 30) & 1, 1, "Q=1");
        // imm5 = (2 << 3) | 0b100 = 0b10100 = 20
        assert_eq!((enc >> 16) & 0b11111, 0b10100, "imm5 for S lane 2");
        assert_eq!((enc >> 11) & 0b1111, 0b0000, "opcode for DUP element");
    }

    #[test]
    fn test_dup_general_s() {
        // DUP V0.4S, W1
        let enc = encode_dup_general(1, ElementSize::S, 1, 0).unwrap();
        assert_eq!((enc >> 16) & 0b11111, 0b00100, "imm5 for S general");
        assert_eq!((enc >> 11) & 0b1111, 0b0011, "opcode for DUP general");
    }

    // === INS ===

    #[test]
    fn test_ins_general_s() {
        // INS V0.S[1], W2
        let enc = encode_ins_general(ElementSize::S, 1, 2, 0).unwrap();
        assert_eq!((enc >> 30) & 1, 1, "Q=1 for INS");
        // imm5 = (1 << 3) | 0b100 = 0b01100 = 12
        assert_eq!((enc >> 16) & 0b11111, 0b01100, "imm5 for S lane 1");
    }

    // === MOVI ===

    #[test]
    fn test_movi_byte_16b() {
        // MOVI V0.16B, #0xAB
        let enc = encode_movi_byte(1, 0xAB, 0).unwrap();
        assert_eq!((enc >> 30) & 1, 1, "Q=1 for 16B");
        let abc = (enc >> 16) & 0b111;
        let defgh = (enc >> 5) & 0b11111;
        let reconstructed = (abc << 5) | defgh;
        assert_eq!(reconstructed, 0xAB_u32, "immediate round-trips");
    }

    #[test]
    fn test_movi_byte_8b() {
        // MOVI V5.8B, #0x42
        let enc = encode_movi_byte(0, 0x42, 5).unwrap();
        assert_eq!((enc >> 30) & 1, 0, "Q=0 for 8B");
        assert_eq!(enc & 0b11111, 5, "Rd=5");
    }

    // === LD1 / ST1 ===

    #[test]
    fn test_ld1_post_4s() {
        // LD1 {V0.4S}, [X1], #16
        let enc = encode_ld1_post_imm(VectorArrangement::S4, 1, 0).unwrap();
        assert_eq!((enc >> 30) & 1, 1, "Q=1 for 4S");
        assert_eq!((enc >> 22) & 1, 1, "L=1 for load");
        assert_eq!((enc >> 16) & 0b11111, 0b11111, "Rm=11111 for imm post-index");
        assert_eq!((enc >> 12) & 0b1111, 0b0111, "opcode=0111 for 1 register");
    }

    #[test]
    fn test_st1_post_4s() {
        // ST1 {V0.4S}, [X1], #16
        let enc = encode_st1_post_imm(VectorArrangement::S4, 1, 0).unwrap();
        assert_eq!((enc >> 22) & 1, 0, "L=0 for store");
    }

    #[test]
    fn test_ld1_post_16b() {
        let enc = encode_ld1_post_imm(VectorArrangement::B16, 3, 7).unwrap();
        assert_eq!((enc >> 30) & 1, 1, "Q=1");
        assert_eq!((enc >> 10) & 0b11, 0b00, "size=00 for B");
        assert_eq!((enc >> 5) & 0b11111, 3, "Rn=3");
        assert_eq!(enc & 0b11111, 7, "Rt=7");
    }

    // === Register validation ===

    #[test]
    fn test_register_out_of_range() {
        assert!(encode_int_vec3_same(VectorArrangement::S4, IntVec3Op::Add, 32, 0, 0).is_err());
        assert!(encode_vec_logic(1, VecLogicOp::And, 0, 32, 0).is_err());
        assert!(encode_vec_not(1, 32, 0).is_err());
        assert!(encode_fp_vec3_same(FpVectorArrangement::S4, FpVec3Op::Fadd, 0, 0, 32).is_err());
    }

    // === Full bit-pattern verification against ARM ARM ===

    #[test]
    fn test_add_vec_4s_exact_bits() {
        // ADD V0.4S, V1.4S, V2.4S
        // 0|1|0|01110|10|1|00010|10000|1|00001|00000
        // = 0100_1110_1010_0010_1000_0100_0010_0000
        // = 0x4EA28420
        let enc = encode_int_vec3_same(VectorArrangement::S4, IntVec3Op::Add, 2, 1, 0).unwrap();
        assert_eq!(enc, 0x4EA28420, "ADD V0.4S, V1.4S, V2.4S = {enc:#010X}");
    }

    #[test]
    fn test_fadd_vec_4s_exact_bits() {
        // FADD V0.4S, V1.4S, V2.4S
        // Expected: 0|1|0|01110|0|0|1|00010|110101|00001|00000
        //         = 0x4E22D420
        let enc = encode_fp_vec3_same(FpVectorArrangement::S4, FpVec3Op::Fadd, 2, 1, 0).unwrap();
        assert_eq!(enc, 0x4E22D420, "FADD V0.4S, V1.4S, V2.4S = {enc:#010X}");
    }

    #[test]
    fn test_and_vec_16b_exact_bits() {
        // AND V0.16B, V1.16B, V2.16B
        // 0|1|0|01110|00|1|00010|00011|1|00001|00000
        // = 0x4E221C20
        let enc = encode_vec_logic(1, VecLogicOp::And, 2, 1, 0).unwrap();
        assert_eq!(enc, 0x4E221C20, "AND V0.16B, V1.16B, V2.16B = {enc:#010X}");
    }

    #[test]
    fn test_not_vec_16b_exact_bits() {
        // NOT V0.16B, V1.16B
        // 0|1|1|01110|00|10000|00101|10|00001|00000
        // = 0x6E205820
        let enc = encode_vec_not(1, 1, 0).unwrap();
        assert_eq!(enc, 0x6E205820, "NOT V0.16B, V1.16B = {enc:#010X}");
    }
}
