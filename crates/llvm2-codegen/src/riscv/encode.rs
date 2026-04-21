// llvm2-codegen/riscv/encode.rs - RISC-V instruction binary encoder
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: RISC-V Unprivileged ISA Specification (Volume 1, Version 20191213)
// Reference: RISC-V "M" Standard Extension for Integer Multiply/Divide
// Reference: RISC-V "D" Standard Extension for Double-Precision FP

//! RISC-V RV64GC instruction binary encoder.
//!
//! Encodes `RiscVOpcode` instructions into fixed 32-bit machine code words.
//! RISC-V uses six core instruction formats (R, I, S, B, U, J) with
//! standard field positions:
//!
//! ```text
//! R-type: [funct7(7)][rs2(5)][rs1(5)][funct3(3)][rd(5)][opcode(7)]
//! I-type: [imm[11:0](12)][rs1(5)][funct3(3)][rd(5)][opcode(7)]
//! S-type: [imm[11:5](7)][rs2(5)][rs1(5)][funct3(3)][imm[4:0](5)][opcode(7)]
//! B-type: [imm[12|10:5](7)][rs2(5)][rs1(5)][funct3(3)][imm[4:1|11](5)][opcode(7)]
//! U-type: [imm[31:12](20)][rd(5)][opcode(7)]
//! J-type: [imm[20|10:1|11|19:12](20)][rd(5)][opcode(7)]
//! ```
//!
//! All instructions are 32 bits wide. The encoder produces a `u32` in native
//! RISC-V bit order (little-endian when stored to memory).

use llvm2_ir::riscv_ops::RiscVOpcode;
use llvm2_ir::riscv_regs::RiscVPReg;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error type for RISC-V encoding failures.
#[derive(Debug, Clone)]
pub enum RiscVEncodeError {
    /// Missing destination register.
    MissingRd(RiscVOpcode),
    /// Missing first source register.
    MissingRs1(RiscVOpcode),
    /// Missing second source register.
    MissingRs2(RiscVOpcode),
    /// Immediate value out of range for the instruction format.
    ImmediateOutOfRange {
        opcode: RiscVOpcode,
        value: i32,
        bits: u32,
    },
}

impl core::fmt::Display for RiscVEncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingRd(op) => write!(f, "{:?}: missing rd register", op),
            Self::MissingRs1(op) => write!(f, "{:?}: missing rs1 register", op),
            Self::MissingRs2(op) => write!(f, "{:?}: missing rs2 register", op),
            Self::ImmediateOutOfRange { opcode, value, bits } => {
                write!(f, "{:?}: immediate {} out of range for {}-bit field", opcode, value, bits)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Operand container
// ---------------------------------------------------------------------------

/// Operands for a RISC-V instruction to be encoded.
///
/// The ISel or lowering pass populates this before calling the encoder.
/// For R-type: rd, rs1, rs2 are all set.
/// For I-type: rd, rs1 are set, imm holds the 12-bit immediate.
/// For S-type: rs1 (base), rs2 (source), imm holds the 12-bit offset.
/// For B-type: rs1, rs2 are set, imm holds the branch offset.
/// For U-type: rd is set, imm holds the 20-bit upper immediate.
/// For J-type: rd is set, imm holds the 21-bit jump offset.
#[derive(Debug, Clone)]
pub struct RiscVInstOperands {
    /// Destination register (rd).
    pub rd: Option<RiscVPReg>,
    /// First source register (rs1).
    pub rs1: Option<RiscVPReg>,
    /// Second source register (rs2).
    pub rs2: Option<RiscVPReg>,
    /// Immediate value (sign-extended to 32 bits).
    pub imm: i32,
}

impl RiscVInstOperands {
    /// Create empty operands.
    pub fn none() -> Self {
        Self {
            rd: None,
            rs1: None,
            rs2: None,
            imm: 0,
        }
    }

    /// Create operands for R-type: rd, rs1, rs2.
    pub fn rrr(rd: RiscVPReg, rs1: RiscVPReg, rs2: RiscVPReg) -> Self {
        Self {
            rd: Some(rd),
            rs1: Some(rs1),
            rs2: Some(rs2),
            imm: 0,
        }
    }

    /// Create operands for I-type: rd, rs1, imm12.
    pub fn rri(rd: RiscVPReg, rs1: RiscVPReg, imm: i32) -> Self {
        Self {
            rd: Some(rd),
            rs1: Some(rs1),
            rs2: None,
            imm,
        }
    }

    /// Create operands for S-type: rs1 (base), rs2 (source), imm12 (offset).
    pub fn store(rs1: RiscVPReg, rs2: RiscVPReg, imm: i32) -> Self {
        Self {
            rd: None,
            rs1: Some(rs1),
            rs2: Some(rs2),
            imm,
        }
    }

    /// Create operands for B-type: rs1, rs2, offset.
    pub fn branch(rs1: RiscVPReg, rs2: RiscVPReg, offset: i32) -> Self {
        Self {
            rd: None,
            rs1: Some(rs1),
            rs2: Some(rs2),
            imm: offset,
        }
    }

    /// Create operands for U-type: rd, imm20.
    pub fn ri(rd: RiscVPReg, imm: i32) -> Self {
        Self {
            rd: Some(rd),
            rs1: None,
            rs2: None,
            imm,
        }
    }

    /// Create operands for J-type: rd, offset.
    pub fn jump(rd: RiscVPReg, offset: i32) -> Self {
        Self {
            rd: Some(rd),
            rs1: None,
            rs2: None,
            imm: offset,
        }
    }
}

// ---------------------------------------------------------------------------
// Low-level format encoders
// ---------------------------------------------------------------------------

/// Encode an R-type instruction.
///
/// Layout: `[funct7(7)][rs2(5)][rs1(5)][funct3(3)][rd(5)][opcode(7)]`
#[inline]
pub fn encode_r_type(opcode: u32, rd: u32, funct3: u32, rs1: u32, rs2: u32, funct7: u32) -> u32 {
    (funct7 & 0x7F) << 25
        | (rs2 & 0x1F) << 20
        | (rs1 & 0x1F) << 15
        | (funct3 & 0x7) << 12
        | (rd & 0x1F) << 7
        | (opcode & 0x7F)
}

/// Encode an I-type instruction.
///
/// Layout: `[imm[11:0](12)][rs1(5)][funct3(3)][rd(5)][opcode(7)]`
#[inline]
pub fn encode_i_type(opcode: u32, rd: u32, funct3: u32, rs1: u32, imm12: i32) -> u32 {
    let imm = (imm12 as u32) & 0xFFF;
    imm << 20
        | (rs1 & 0x1F) << 15
        | (funct3 & 0x7) << 12
        | (rd & 0x1F) << 7
        | (opcode & 0x7F)
}

/// Encode an S-type instruction.
///
/// Layout: `[imm[11:5](7)][rs2(5)][rs1(5)][funct3(3)][imm[4:0](5)][opcode(7)]`
#[inline]
pub fn encode_s_type(opcode: u32, funct3: u32, rs1: u32, rs2: u32, imm12: i32) -> u32 {
    let imm = (imm12 as u32) & 0xFFF;
    let imm_11_5 = (imm >> 5) & 0x7F;
    let imm_4_0 = imm & 0x1F;
    imm_11_5 << 25
        | (rs2 & 0x1F) << 20
        | (rs1 & 0x1F) << 15
        | (funct3 & 0x7) << 12
        | imm_4_0 << 7
        | (opcode & 0x7F)
}

/// Encode a B-type instruction.
///
/// The immediate encodes a signed offset in multiples of 2 bytes.
/// Layout: `[imm[12|10:5](7)][rs2(5)][rs1(5)][funct3(3)][imm[4:1|11](5)][opcode(7)]`
#[inline]
pub fn encode_b_type(opcode: u32, funct3: u32, rs1: u32, rs2: u32, offset: i32) -> u32 {
    // B-type immediate encoding:
    // bit 31: imm[12]
    // bits 30:25: imm[10:5]
    // bits 11:8: imm[4:1]
    // bit 7: imm[11]
    // The offset is a signed value where bit 0 is implicitly 0.
    let off = offset as u32;
    let b12 = (off >> 12) & 1;
    let b11 = (off >> 11) & 1;
    let b10_5 = (off >> 5) & 0x3F;
    let b4_1 = (off >> 1) & 0xF;
    b12 << 31
        | b10_5 << 25
        | (rs2 & 0x1F) << 20
        | (rs1 & 0x1F) << 15
        | (funct3 & 0x7) << 12
        | b4_1 << 8
        | b11 << 7
        | (opcode & 0x7F)
}

/// Encode a U-type instruction.
///
/// Layout: `[imm[31:12](20)][rd(5)][opcode(7)]`
/// The immediate is the upper 20 bits (already shifted by the assembler).
#[inline]
pub fn encode_u_type(opcode: u32, rd: u32, imm20: i32) -> u32 {
    let imm = (imm20 as u32) & 0xFFFFF;
    imm << 12
        | (rd & 0x1F) << 7
        | (opcode & 0x7F)
}

/// Encode a J-type instruction (JAL).
///
/// The immediate encodes a signed offset in multiples of 2 bytes.
/// Layout: `[imm[20|10:1|11|19:12](20)][rd(5)][opcode(7)]`
#[inline]
pub fn encode_j_type(opcode: u32, rd: u32, offset: i32) -> u32 {
    let off = offset as u32;
    let b20 = (off >> 20) & 1;
    let b10_1 = (off >> 1) & 0x3FF;
    let b11 = (off >> 11) & 1;
    let b19_12 = (off >> 12) & 0xFF;
    b20 << 31
        | b10_1 << 21
        | b11 << 20
        | b19_12 << 12
        | (rd & 0x1F) << 7
        | (opcode & 0x7F)
}

// ---------------------------------------------------------------------------
// RISC-V base opcode field values (bits [6:0])
// ---------------------------------------------------------------------------

const OP: u32 = 0b0110011;       // R-type integer (ADD, SUB, etc.)
const OP_32: u32 = 0b0111011;    // R-type word (ADDW, SUBW, etc.)
const OP_IMM: u32 = 0b0010011;   // I-type integer immediate (ADDI, etc.)
const OP_IMM_32: u32 = 0b0011011; // I-type word immediate (ADDIW, etc.)
const LOAD: u32 = 0b0000011;     // I-type loads (LB, LH, LW, LD)
const LOAD_FP: u32 = 0b0000111;  // I-type FP loads (FLD)
const STORE: u32 = 0b0100011;    // S-type stores (SB, SH, SW, SD)
const STORE_FP: u32 = 0b0100111; // S-type FP stores (FSD)
const BRANCH: u32 = 0b1100011;   // B-type branches
const LUI_OP: u32 = 0b0110111;   // U-type (LUI)
const AUIPC_OP: u32 = 0b0010111; // U-type (AUIPC)
const JAL_OP: u32 = 0b1101111;   // J-type (JAL)
const JALR_OP: u32 = 0b1100111;  // I-type (JALR)
const OP_FP: u32 = 0b1010011;    // R-type FP operations

/// RISC-V NOP encoding: ADDI x0, x0, 0.
const RISCV_NOP: u32 = 0x00000013;

// ---------------------------------------------------------------------------
// Main encoder
// ---------------------------------------------------------------------------

/// Encode a single RISC-V instruction into a 32-bit machine code word.
///
/// Returns the instruction word on success. Pseudo-instructions (Phi,
/// StackAlloc, Nop) encode as NOP (ADDI x0, x0, 0 = 0x00000013).
pub fn encode_instruction(
    opcode: RiscVOpcode,
    ops: &RiscVInstOperands,
) -> Result<u32, RiscVEncodeError> {
    use RiscVOpcode::*;

    match opcode {
        // =================================================================
        // Pseudo-instructions -> NOP
        // =================================================================
        Phi | StackAlloc | Nop => Ok(RISCV_NOP),

        // =================================================================
        // R-type: RV64I integer register-register
        // =================================================================
        Add => encode_r(ops, opcode, OP, 0b000, 0b0000000),
        Sub => encode_r(ops, opcode, OP, 0b000, 0b0100000),
        And => encode_r(ops, opcode, OP, 0b111, 0b0000000),
        Or  => encode_r(ops, opcode, OP, 0b110, 0b0000000),
        Xor => encode_r(ops, opcode, OP, 0b100, 0b0000000),
        Sll => encode_r(ops, opcode, OP, 0b001, 0b0000000),
        Srl => encode_r(ops, opcode, OP, 0b101, 0b0000000),
        Sra => encode_r(ops, opcode, OP, 0b101, 0b0100000),
        Slt => encode_r(ops, opcode, OP, 0b010, 0b0000000),
        Sltu => encode_r(ops, opcode, OP, 0b011, 0b0000000),

        // =================================================================
        // R-type: RV64I word operations
        // =================================================================
        Addw => encode_r(ops, opcode, OP_32, 0b000, 0b0000000),
        Subw => encode_r(ops, opcode, OP_32, 0b000, 0b0100000),
        Sllw => encode_r(ops, opcode, OP_32, 0b001, 0b0000000),
        Srlw => encode_r(ops, opcode, OP_32, 0b101, 0b0000000),
        Sraw => encode_r(ops, opcode, OP_32, 0b101, 0b0100000),

        // =================================================================
        // R-type: M extension (multiply/divide)
        // =================================================================
        Mul    => encode_r(ops, opcode, OP, 0b000, 0b0000001),
        Mulh   => encode_r(ops, opcode, OP, 0b001, 0b0000001),
        Mulhsu => encode_r(ops, opcode, OP, 0b010, 0b0000001),
        Mulhu  => encode_r(ops, opcode, OP, 0b011, 0b0000001),
        Div    => encode_r(ops, opcode, OP, 0b100, 0b0000001),
        Divu   => encode_r(ops, opcode, OP, 0b101, 0b0000001),
        Rem    => encode_r(ops, opcode, OP, 0b110, 0b0000001),
        Remu   => encode_r(ops, opcode, OP, 0b111, 0b0000001),

        // M extension word operations
        Mulw  => encode_r(ops, opcode, OP_32, 0b000, 0b0000001),
        Divw  => encode_r(ops, opcode, OP_32, 0b100, 0b0000001),
        Divuw => encode_r(ops, opcode, OP_32, 0b101, 0b0000001),
        Remw  => encode_r(ops, opcode, OP_32, 0b110, 0b0000001),
        Remuw => encode_r(ops, opcode, OP_32, 0b111, 0b0000001),

        // =================================================================
        // I-type: integer immediate
        // =================================================================
        Addi  => encode_i(ops, opcode, OP_IMM, 0b000),
        Andi  => encode_i(ops, opcode, OP_IMM, 0b111),
        Ori   => encode_i(ops, opcode, OP_IMM, 0b110),
        Xori  => encode_i(ops, opcode, OP_IMM, 0b100),
        Slti  => encode_i(ops, opcode, OP_IMM, 0b010),
        Sltiu => encode_i(ops, opcode, OP_IMM, 0b011),

        // Shift immediates (RV64: 6-bit shamt in imm[5:0], imm[11:6] encodes variant)
        Slli => encode_shift_imm(ops, opcode, OP_IMM, 0b001, 0b000000),
        Srli => encode_shift_imm(ops, opcode, OP_IMM, 0b101, 0b000000),
        Srai => encode_shift_imm(ops, opcode, OP_IMM, 0b101, 0b010000),

        // I-type: word immediate
        Addiw => encode_i(ops, opcode, OP_IMM_32, 0b000),

        // Shift immediates (RV64W: 5-bit shamt in imm[4:0], imm[11:5] encodes variant)
        Slliw => encode_shift_imm_w(ops, opcode, OP_IMM_32, 0b001, 0b0000000),
        Srliw => encode_shift_imm_w(ops, opcode, OP_IMM_32, 0b101, 0b0000000),
        Sraiw => encode_shift_imm_w(ops, opcode, OP_IMM_32, 0b101, 0b0100000),

        // =================================================================
        // I-type: loads
        // =================================================================
        Lb  => encode_i(ops, opcode, LOAD, 0b000),
        Lh  => encode_i(ops, opcode, LOAD, 0b001),
        Lw  => encode_i(ops, opcode, LOAD, 0b010),
        Ld  => encode_i(ops, opcode, LOAD, 0b011),
        Lbu => encode_i(ops, opcode, LOAD, 0b100),
        Lhu => encode_i(ops, opcode, LOAD, 0b101),
        Lwu => encode_i(ops, opcode, LOAD, 0b110),

        // FP load
        Fld => encode_i(ops, opcode, LOAD_FP, 0b011),

        // =================================================================
        // I-type: JALR
        // =================================================================
        Jalr => encode_i(ops, opcode, JALR_OP, 0b000),

        // =================================================================
        // S-type: stores
        // =================================================================
        Sb => encode_s(ops, opcode, STORE, 0b000),
        Sh => encode_s(ops, opcode, STORE, 0b001),
        Sw => encode_s(ops, opcode, STORE, 0b010),
        Sd => encode_s(ops, opcode, STORE, 0b011),

        // FP store
        Fsd => encode_s(ops, opcode, STORE_FP, 0b011),

        // =================================================================
        // B-type: branches
        // =================================================================
        Beq  => encode_b(ops, opcode, 0b000),
        Bne  => encode_b(ops, opcode, 0b001),
        Blt  => encode_b(ops, opcode, 0b100),
        Bge  => encode_b(ops, opcode, 0b101),
        Bltu => encode_b(ops, opcode, 0b110),
        Bgeu => encode_b(ops, opcode, 0b111),

        // =================================================================
        // U-type
        // =================================================================
        Lui   => encode_u(ops, opcode, LUI_OP),
        Auipc => encode_u(ops, opcode, AUIPC_OP),

        // =================================================================
        // J-type
        // =================================================================
        Jal => encode_j(ops, opcode),

        // =================================================================
        // R-type FP: RV64D double-precision
        // =================================================================
        // FP arithmetic: funct7 encodes operation + fmt (01 = double)
        // rm field (funct3) = 000 (RNE = round to nearest, ties to even)
        FaddD => encode_r_fp(ops, opcode, 0b0000001, 0b000),
        FsubD => encode_r_fp(ops, opcode, 0b0000101, 0b000),
        FmulD => encode_r_fp(ops, opcode, 0b0001001, 0b000),
        FdivD => encode_r_fp(ops, opcode, 0b0001101, 0b000),

        // FSQRT.D: rs2 = 00000
        FsqrtD => encode_r_fp_unary(ops, opcode, 0b0101101, 0b000, 0b00000),

        // FP comparisons: result goes to GPR
        FeqD => encode_r_fp(ops, opcode, 0b1010001, 0b010),
        FltD => encode_r_fp(ops, opcode, 0b1010001, 0b001),
        FleD => encode_r_fp(ops, opcode, 0b1010001, 0b000),

        // FP conversions: rs2 encodes source format
        FcvtDW => encode_r_fp_unary(ops, opcode, 0b1101001, 0b000, 0b00000),
        FcvtWD => encode_r_fp_unary(ops, opcode, 0b1100001, 0b000, 0b00000),
        FcvtDL => encode_r_fp_unary(ops, opcode, 0b1101001, 0b000, 0b00010),
        FcvtLD => encode_r_fp_unary(ops, opcode, 0b1100001, 0b000, 0b00010),

        // FP moves
        FmvXD => encode_r_fp_unary(ops, opcode, 0b1110001, 0b000, 0b00000),
        FmvDX => encode_r_fp_unary(ops, opcode, 0b1111001, 0b000, 0b00000),
    }
}

// ---------------------------------------------------------------------------
// Encoding helpers (extract operands and dispatch to format encoders)
// ---------------------------------------------------------------------------

/// Encode an R-type instruction from operands.
fn encode_r(
    ops: &RiscVInstOperands,
    opcode: RiscVOpcode,
    op: u32,
    funct3: u32,
    funct7: u32,
) -> Result<u32, RiscVEncodeError> {
    let rd = ops.rd.ok_or(RiscVEncodeError::MissingRd(opcode))?.hw_enc() as u32;
    let rs1 = ops.rs1.ok_or(RiscVEncodeError::MissingRs1(opcode))?.hw_enc() as u32;
    let rs2 = ops.rs2.ok_or(RiscVEncodeError::MissingRs2(opcode))?.hw_enc() as u32;
    Ok(encode_r_type(op, rd, funct3, rs1, rs2, funct7))
}

/// Encode an I-type instruction from operands.
fn encode_i(
    ops: &RiscVInstOperands,
    opcode: RiscVOpcode,
    op: u32,
    funct3: u32,
) -> Result<u32, RiscVEncodeError> {
    let rd = ops.rd.ok_or(RiscVEncodeError::MissingRd(opcode))?.hw_enc() as u32;
    let rs1 = ops.rs1.ok_or(RiscVEncodeError::MissingRs1(opcode))?.hw_enc() as u32;
    Ok(encode_i_type(op, rd, funct3, rs1, ops.imm))
}

/// Encode a shift-immediate (RV64: 6-bit shamt).
/// The imm[11:6] field is set to `hi6` to distinguish SLLI/SRLI/SRAI.
fn encode_shift_imm(
    ops: &RiscVInstOperands,
    opcode: RiscVOpcode,
    op: u32,
    funct3: u32,
    hi6: u32,
) -> Result<u32, RiscVEncodeError> {
    let rd = ops.rd.ok_or(RiscVEncodeError::MissingRd(opcode))?.hw_enc() as u32;
    let rs1 = ops.rs1.ok_or(RiscVEncodeError::MissingRs1(opcode))?.hw_enc() as u32;
    let shamt = (ops.imm as u32) & 0x3F; // 6-bit shift amount
    let imm12 = (hi6 << 6) | shamt;
    Ok(encode_i_type(op, rd, funct3, rs1, imm12 as i32))
}

/// Encode a shift-immediate for W-variant (5-bit shamt).
/// The imm[11:5] field is set to `hi7` to distinguish SLLIW/SRLIW/SRAIW.
fn encode_shift_imm_w(
    ops: &RiscVInstOperands,
    opcode: RiscVOpcode,
    op: u32,
    funct3: u32,
    hi7: u32,
) -> Result<u32, RiscVEncodeError> {
    let rd = ops.rd.ok_or(RiscVEncodeError::MissingRd(opcode))?.hw_enc() as u32;
    let rs1 = ops.rs1.ok_or(RiscVEncodeError::MissingRs1(opcode))?.hw_enc() as u32;
    let shamt = (ops.imm as u32) & 0x1F; // 5-bit shift amount
    let imm12 = (hi7 << 5) | shamt;
    Ok(encode_i_type(op, rd, funct3, rs1, imm12 as i32))
}

/// Encode an S-type instruction from operands.
fn encode_s(
    ops: &RiscVInstOperands,
    opcode: RiscVOpcode,
    op: u32,
    funct3: u32,
) -> Result<u32, RiscVEncodeError> {
    let rs1 = ops.rs1.ok_or(RiscVEncodeError::MissingRs1(opcode))?.hw_enc() as u32;
    let rs2 = ops.rs2.ok_or(RiscVEncodeError::MissingRs2(opcode))?.hw_enc() as u32;
    Ok(encode_s_type(op, funct3, rs1, rs2, ops.imm))
}

/// Encode a B-type instruction from operands.
fn encode_b(
    ops: &RiscVInstOperands,
    opcode: RiscVOpcode,
    funct3: u32,
) -> Result<u32, RiscVEncodeError> {
    let rs1 = ops.rs1.ok_or(RiscVEncodeError::MissingRs1(opcode))?.hw_enc() as u32;
    let rs2 = ops.rs2.ok_or(RiscVEncodeError::MissingRs2(opcode))?.hw_enc() as u32;
    Ok(encode_b_type(BRANCH, funct3, rs1, rs2, ops.imm))
}

/// Encode a U-type instruction from operands.
fn encode_u(
    ops: &RiscVInstOperands,
    opcode: RiscVOpcode,
    op: u32,
) -> Result<u32, RiscVEncodeError> {
    let rd = ops.rd.ok_or(RiscVEncodeError::MissingRd(opcode))?.hw_enc() as u32;
    Ok(encode_u_type(op, rd, ops.imm))
}

/// Encode a J-type instruction from operands.
fn encode_j(
    ops: &RiscVInstOperands,
    opcode: RiscVOpcode,
) -> Result<u32, RiscVEncodeError> {
    let rd = ops.rd.ok_or(RiscVEncodeError::MissingRd(opcode))?.hw_enc() as u32;
    Ok(encode_j_type(JAL_OP, rd, ops.imm))
}

/// Encode an R-type FP instruction (two source registers).
fn encode_r_fp(
    ops: &RiscVInstOperands,
    opcode: RiscVOpcode,
    funct7: u32,
    funct3: u32,
) -> Result<u32, RiscVEncodeError> {
    let rd = ops.rd.ok_or(RiscVEncodeError::MissingRd(opcode))?.hw_enc() as u32;
    let rs1 = ops.rs1.ok_or(RiscVEncodeError::MissingRs1(opcode))?.hw_enc() as u32;
    let rs2 = ops.rs2.ok_or(RiscVEncodeError::MissingRs2(opcode))?.hw_enc() as u32;
    Ok(encode_r_type(OP_FP, rd, funct3, rs1, rs2, funct7))
}

/// Encode an R-type FP unary instruction (one source, rs2 is fixed).
fn encode_r_fp_unary(
    ops: &RiscVInstOperands,
    opcode: RiscVOpcode,
    funct7: u32,
    funct3: u32,
    rs2_fixed: u32,
) -> Result<u32, RiscVEncodeError> {
    let rd = ops.rd.ok_or(RiscVEncodeError::MissingRd(opcode))?.hw_enc() as u32;
    let rs1 = ops.rs1.ok_or(RiscVEncodeError::MissingRs1(opcode))?.hw_enc() as u32;
    Ok(encode_r_type(OP_FP, rd, funct3, rs1, rs2_fixed, funct7))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::riscv_regs::{X0, X1, X2, X5, X10, X11, X12, X15, X28, X31};
    use llvm2_ir::riscv_regs::{F0, F1, F10, F11};

    // Helper to encode and unwrap.
    fn enc(opcode: RiscVOpcode, ops: &RiscVInstOperands) -> u32 {
        encode_instruction(opcode, ops).unwrap()
    }

    // -----------------------------------------------------------------------
    // Pseudo-instructions
    // -----------------------------------------------------------------------

    #[test]
    fn test_nop_encoding() {
        // NOP = ADDI x0, x0, 0 = 0x00000013
        assert_eq!(enc(RiscVOpcode::Nop, &RiscVInstOperands::none()), 0x00000013);
        assert_eq!(enc(RiscVOpcode::Phi, &RiscVInstOperands::none()), 0x00000013);
        assert_eq!(enc(RiscVOpcode::StackAlloc, &RiscVInstOperands::none()), 0x00000013);
    }

    // -----------------------------------------------------------------------
    // R-type tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_x10_x11_x12() {
        // ADD x10, x11, x12
        // funct7=0000000, rs2=12, rs1=11, funct3=000, rd=10, opcode=0110011
        // = 0000000 01100 01011 000 01010 0110011
        // = 0x00C58533
        let word = enc(RiscVOpcode::Add, &RiscVInstOperands::rrr(X10, X11, X12));
        assert_eq!(word, 0x00C58533);
    }

    #[test]
    fn test_sub_x10_x11_x12() {
        // SUB x10, x11, x12
        // funct7=0100000, rs2=12, rs1=11, funct3=000, rd=10, opcode=0110011
        // = 0100000 01100 01011 000 01010 0110011
        // = 0x40C58533
        let word = enc(RiscVOpcode::Sub, &RiscVInstOperands::rrr(X10, X11, X12));
        assert_eq!(word, 0x40C58533);
    }

    #[test]
    fn test_and_x10_x11_x12() {
        // AND x10, x11, x12
        // funct7=0000000, rs2=12, rs1=11, funct3=111, rd=10, opcode=0110011
        // = 0x00C5F533
        let word = enc(RiscVOpcode::And, &RiscVInstOperands::rrr(X10, X11, X12));
        assert_eq!(word, 0x00C5F533);
    }

    #[test]
    fn test_or_x10_x11_x12() {
        // OR x10, x11, x12
        // funct7=0000000, rs2=12, rs1=11, funct3=110, rd=10, opcode=0110011
        // = 0x00C5E533
        let word = enc(RiscVOpcode::Or, &RiscVInstOperands::rrr(X10, X11, X12));
        assert_eq!(word, 0x00C5E533);
    }

    #[test]
    fn test_xor_x10_x11_x12() {
        // XOR x10, x11, x12
        // funct7=0000000, rs2=12, rs1=11, funct3=100, rd=10, opcode=0110011
        // = 0x00C5C533
        let word = enc(RiscVOpcode::Xor, &RiscVInstOperands::rrr(X10, X11, X12));
        assert_eq!(word, 0x00C5C533);
    }

    #[test]
    fn test_sll_x10_x11_x12() {
        // SLL x10, x11, x12
        // funct7=0000000, rs2=12, rs1=11, funct3=001, rd=10, opcode=0110011
        // = 0x00C59533
        let word = enc(RiscVOpcode::Sll, &RiscVInstOperands::rrr(X10, X11, X12));
        assert_eq!(word, 0x00C59533);
    }

    #[test]
    fn test_slt_x10_x11_x12() {
        // SLT x10, x11, x12
        // funct7=0000000, rs2=12, rs1=11, funct3=010, rd=10, opcode=0110011
        // = 0x00C5A533
        let word = enc(RiscVOpcode::Slt, &RiscVInstOperands::rrr(X10, X11, X12));
        assert_eq!(word, 0x00C5A533);
    }

    #[test]
    fn test_mul_x10_x11_x12() {
        // MUL x10, x11, x12
        // funct7=0000001, rs2=12, rs1=11, funct3=000, rd=10, opcode=0110011
        // = 0x02C58533
        let word = enc(RiscVOpcode::Mul, &RiscVInstOperands::rrr(X10, X11, X12));
        assert_eq!(word, 0x02C58533);
    }

    #[test]
    fn test_div_x10_x11_x12() {
        // DIV x10, x11, x12
        // funct7=0000001, rs2=12, rs1=11, funct3=100, rd=10, opcode=0110011
        // = 0x02C5C533
        let word = enc(RiscVOpcode::Div, &RiscVInstOperands::rrr(X10, X11, X12));
        assert_eq!(word, 0x02C5C533);
    }

    // -----------------------------------------------------------------------
    // I-type tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_addi_x10_x11_42() {
        // ADDI x10, x11, 42
        // imm=42=0x02A, rs1=11, funct3=000, rd=10, opcode=0010011
        // = 000000101010 01011 000 01010 0010011
        // = 0x02A58513
        let word = enc(RiscVOpcode::Addi, &RiscVInstOperands::rri(X10, X11, 42));
        assert_eq!(word, 0x02A58513);
    }

    #[test]
    fn test_addi_x0_x0_0_is_nop() {
        // ADDI x0, x0, 0 = NOP
        let word = enc(RiscVOpcode::Addi, &RiscVInstOperands::rri(X0, X0, 0));
        assert_eq!(word, 0x00000013);
    }

    #[test]
    fn test_addi_negative_imm() {
        // ADDI x10, x11, -1
        // imm = -1 = 0xFFF (12-bit), rs1=11, funct3=000, rd=10, opcode=0010011
        // = 0xFFF58513
        let word = enc(RiscVOpcode::Addi, &RiscVInstOperands::rri(X10, X11, -1));
        assert_eq!(word, 0xFFF58513);
    }

    #[test]
    fn test_andi_x10_x11_0xff() {
        // ANDI x10, x11, 0xFF
        // imm=255=0x0FF, rs1=11, funct3=111, rd=10, opcode=0010011
        // = 0x0FF5F513
        let word = enc(RiscVOpcode::Andi, &RiscVInstOperands::rri(X10, X11, 0xFF));
        assert_eq!(word, 0x0FF5F513);
    }

    #[test]
    fn test_slli_x10_x11_3() {
        // SLLI x10, x11, 3 (RV64: 6-bit shamt)
        // imm[11:6]=000000, shamt=3, rs1=11, funct3=001, rd=10, opcode=0010011
        // imm12 = 0b000000_000011 = 3
        // = 0x00359513
        let word = enc(RiscVOpcode::Slli, &RiscVInstOperands::rri(X10, X11, 3));
        assert_eq!(word, 0x00359513);
    }

    #[test]
    fn test_srai_x10_x11_4() {
        // SRAI x10, x11, 4 (RV64: 6-bit shamt)
        // imm[11:6]=010000, shamt=4, rs1=11, funct3=101, rd=10, opcode=0010011
        // imm12 = 0b010000_000100 = 0x404
        // = 0x4045D513
        let word = enc(RiscVOpcode::Srai, &RiscVInstOperands::rri(X10, X11, 4));
        assert_eq!(word, 0x4045D513);
    }

    // -----------------------------------------------------------------------
    // Load tests (I-type)
    // -----------------------------------------------------------------------

    #[test]
    fn test_ld_x10_x11_offset8() {
        // LD x10, 8(x11)
        // imm=8, rs1=11, funct3=011, rd=10, opcode=0000011
        // = 0x0085B503
        let word = enc(RiscVOpcode::Ld, &RiscVInstOperands::rri(X10, X11, 8));
        assert_eq!(word, 0x0085B503);
    }

    #[test]
    fn test_lw_x10_x2_neg4() {
        // LW x10, -4(x2/sp)
        // imm=-4=0xFFC, rs1=2, funct3=010, rd=10, opcode=0000011
        // = 0xFFC12503
        let word = enc(RiscVOpcode::Lw, &RiscVInstOperands::rri(X10, X2, -4));
        assert_eq!(word, 0xFFC12503);
    }

    // -----------------------------------------------------------------------
    // S-type tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sd_x2_x10_offset8() {
        // SD x10, 8(x2)
        // rs1=2(base), rs2=10(source), imm=8
        // imm[11:5]=0000000, imm[4:0]=01000 = 8
        // = 0x00A13423
        let word = enc(RiscVOpcode::Sd, &RiscVInstOperands::store(X2, X10, 8));
        assert_eq!(word, 0x00A13423);
    }

    #[test]
    fn test_sw_x2_x11_neg4() {
        // SW x11, -4(x2)
        // rs1=2(base), rs2=11(source), imm=-4
        // imm = -4 = 0xFFC = 111111111100
        // imm[11:5] = 1111111 = 0x7F
        // imm[4:0] = 11100 = 0x1C
        // = 0xFEB12E23
        let word = enc(RiscVOpcode::Sw, &RiscVInstOperands::store(X2, X11, -4));
        assert_eq!(word, 0xFEB12E23);
    }

    // -----------------------------------------------------------------------
    // B-type tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_beq_x10_x11_offset8() {
        // BEQ x10, x11, 8
        // rs1=10, rs2=11, offset=8
        // offset bits: b12=0, b11=0, b10:5=000000, b4:1=0100
        // bit 31: 0, bits 30:25: 000000, bits 11:8: 0100, bit 7: 0
        // = 0x00B50463
        let word = enc(RiscVOpcode::Beq, &RiscVInstOperands::branch(X10, X11, 8));
        assert_eq!(word, 0x00B50463);
    }

    #[test]
    fn test_bne_x10_x0_neg8() {
        // BNE x10, x0, -8
        // rs1=10, rs2=0, offset=-8
        // offset = -8 = 0xFFFFFFF8
        // b12 = 1, b11 = 1, b10:5 = 111111, b4:1 = 1100
        // bit 31: 1, bits 30:25: 111111, bits 11:8: 1100, bit 7: 1
        // = 0xFE051CE3
        let word = enc(RiscVOpcode::Bne, &RiscVInstOperands::branch(X10, X0, -8));
        assert_eq!(word, 0xFE051CE3);
    }

    // -----------------------------------------------------------------------
    // U-type tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lui_x10_0x12345() {
        // LUI x10, 0x12345
        // imm20=0x12345, rd=10, opcode=0110111
        // = 0x12345537
        let word = enc(RiscVOpcode::Lui, &RiscVInstOperands::ri(X10, 0x12345));
        assert_eq!(word, 0x12345537);
    }

    #[test]
    fn test_auipc_x10_1() {
        // AUIPC x10, 1
        // imm20=1, rd=10, opcode=0010111
        // = 0x00001517
        let word = enc(RiscVOpcode::Auipc, &RiscVInstOperands::ri(X10, 1));
        assert_eq!(word, 0x00001517);
    }

    // -----------------------------------------------------------------------
    // J-type tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_jal_x1_offset0() {
        // JAL x1, 0
        // rd=1, offset=0
        // = 0x000000EF
        let word = enc(RiscVOpcode::Jal, &RiscVInstOperands::jump(X1, 0));
        assert_eq!(word, 0x000000EF);
    }

    #[test]
    fn test_jal_x0_offset8() {
        // JAL x0, 8 (equivalent to J 8, unconditional jump)
        // rd=0, offset=8
        // b20=0, b10:1=0000000100, b11=0, b19:12=00000000
        // = 0x0080006F
        let word = enc(RiscVOpcode::Jal, &RiscVInstOperands::jump(X0, 8));
        assert_eq!(word, 0x0080006F);
    }

    // -----------------------------------------------------------------------
    // JALR test (I-type)
    // -----------------------------------------------------------------------

    #[test]
    fn test_jalr_x1_x10_0() {
        // JALR x1, x10, 0
        // imm=0, rs1=10, funct3=000, rd=1, opcode=1100111
        // = 0x000500E7
        let word = enc(RiscVOpcode::Jalr, &RiscVInstOperands::rri(X1, X10, 0));
        assert_eq!(word, 0x000500E7);
    }

    // -----------------------------------------------------------------------
    // Word operations (RV64)
    // -----------------------------------------------------------------------

    #[test]
    fn test_addw_x10_x11_x12() {
        // ADDW x10, x11, x12
        // funct7=0000000, rs2=12, rs1=11, funct3=000, rd=10, opcode=0111011
        // = 0x00C5853B
        let word = enc(RiscVOpcode::Addw, &RiscVInstOperands::rrr(X10, X11, X12));
        assert_eq!(word, 0x00C5853B);
    }

    #[test]
    fn test_addiw_x10_x11_10() {
        // ADDIW x10, x11, 10
        // imm=10, rs1=11, funct3=000, rd=10, opcode=0011011
        // = 0x00A5851B
        let word = enc(RiscVOpcode::Addiw, &RiscVInstOperands::rri(X10, X11, 10));
        assert_eq!(word, 0x00A5851B);
    }

    // -----------------------------------------------------------------------
    // FP tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fadd_d_f10_f0_f1() {
        // FADD.D f10, f0, f1
        // funct7=0000001, rs2=1, rs1=0, funct3(rm)=000, rd=10, opcode=1010011
        // = 0x02100553
        let word = enc(RiscVOpcode::FaddD, &RiscVInstOperands::rrr(F10, F0, F1));
        assert_eq!(word, 0x02100553);
    }

    #[test]
    fn test_fsqrt_d_f10_f11() {
        // FSQRT.D f10, f11
        // funct7=0101101, rs2=00000, rs1=11, funct3=000, rd=10, opcode=1010011
        // = 0x5A058553
        let word = enc(RiscVOpcode::FsqrtD, &RiscVInstOperands::rri(F10, F11, 0));
        assert_eq!(word, 0x5A058553);
    }

    #[test]
    fn test_fld_f10_x11_offset16() {
        // FLD f10, 16(x11)
        // imm=16, rs1=11, funct3=011, rd=10, opcode=0000111
        // = 0x0105B507
        let word = enc(RiscVOpcode::Fld, &RiscVInstOperands::rri(F10, X11, 16));
        assert_eq!(word, 0x0105B507);
    }

    #[test]
    fn test_fsd_x11_f10_offset16() {
        // FSD f10, 16(x11)
        // rs1=11(base), rs2=10(FP source), imm=16
        // S-type: imm[11:5]=0000000, rs2=10, rs1=11, funct3=011, imm[4:0]=10000, opcode=0100111
        // = 0x00A5B827
        let word = enc(RiscVOpcode::Fsd, &RiscVInstOperands::store(X11, F10, 16));
        assert_eq!(word, 0x00A5B827);
    }

    // -----------------------------------------------------------------------
    // Error tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_missing_rd_error() {
        let result = encode_instruction(
            RiscVOpcode::Add,
            &RiscVInstOperands { rd: None, rs1: Some(X10), rs2: Some(X11), imm: 0 },
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_rs1_error() {
        let result = encode_instruction(
            RiscVOpcode::Add,
            &RiscVInstOperands { rd: Some(X10), rs1: None, rs2: Some(X11), imm: 0 },
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_rs2_error() {
        let result = encode_instruction(
            RiscVOpcode::Add,
            &RiscVInstOperands { rd: Some(X10), rs1: Some(X11), rs2: None, imm: 0 },
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_error_display() {
        let e1 = RiscVEncodeError::MissingRd(RiscVOpcode::Add);
        assert!(format!("{}", e1).contains("Add"));
        assert!(format!("{}", e1).contains("rd"));

        let e2 = RiscVEncodeError::ImmediateOutOfRange {
            opcode: RiscVOpcode::Addi,
            value: 9999,
            bits: 12,
        };
        assert!(format!("{}", e2).contains("9999"));
    }

    // -----------------------------------------------------------------------
    // Format-level encoding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_r_type_raw() {
        // ADD x0, x0, x0 = all zeros except opcode
        let word = encode_r_type(0b0110011, 0, 0, 0, 0, 0);
        assert_eq!(word, 0b0110011);
    }

    #[test]
    fn test_encode_i_type_raw() {
        // ADDI x0, x0, 0
        let word = encode_i_type(0b0010011, 0, 0, 0, 0);
        assert_eq!(word, 0x00000013);
    }

    #[test]
    fn test_encode_u_type_raw() {
        // LUI x1, 1
        let word = encode_u_type(0b0110111, 1, 1);
        // rd=1 -> bits[11:7] = 00001, opcode = 0110111
        // imm=1 -> bits[31:12] = 00000000000000000001
        assert_eq!(word, 0x000010B7);
    }

    #[test]
    fn test_encode_j_type_raw() {
        // JAL x0, 0
        let word = encode_j_type(0b1101111, 0, 0);
        assert_eq!(word, 0b1101111); // 0x6F
    }

    // -----------------------------------------------------------------------
    // Comprehensive format coverage
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_r_type_regs_x31() {
        // ADD x31, x31, x31
        // funct7=0000000, rs2=31, rs1=31, funct3=000, rd=31, opcode=0110011
        let word = enc(RiscVOpcode::Add, &RiscVInstOperands::rrr(X31, X31, X31));
        let expected = encode_r_type(OP, 31, 0b000, 31, 31, 0b0000000);
        assert_eq!(word, expected);
    }

    #[test]
    fn test_sltu_x5_x10_x15() {
        // SLTU x5, x10, x15
        // funct7=0000000, rs2=15, rs1=10, funct3=011, rd=5, opcode=0110011
        let word = enc(RiscVOpcode::Sltu, &RiscVInstOperands::rrr(X5, X10, X15));
        let expected = encode_r_type(OP, 5, 0b011, 10, 15, 0b0000000);
        assert_eq!(word, expected);
    }

    #[test]
    fn test_srl_and_sra_differ() {
        // SRL and SRA differ only in funct7 bit 30
        let srl = enc(RiscVOpcode::Srl, &RiscVInstOperands::rrr(X10, X11, X12));
        let sra = enc(RiscVOpcode::Sra, &RiscVInstOperands::rrr(X10, X11, X12));
        assert_ne!(srl, sra);
        // They differ by exactly bit 30 (0x40000000)
        assert_eq!(sra - srl, 0x40000000);
    }

    #[test]
    fn test_blt_x10_x11_offset_neg16() {
        // BLT x10, x11, -16
        let word = enc(RiscVOpcode::Blt, &RiscVInstOperands::branch(X10, X11, -16));
        // Verify it decodes to the right format by checking opcode field
        assert_eq!(word & 0x7F, BRANCH);
        // funct3 for BLT = 100
        assert_eq!((word >> 12) & 0x7, 0b100);
    }

    #[test]
    fn test_bgeu_x28_x0_offset4() {
        // BGEU x28, x0, 4
        let word = enc(RiscVOpcode::Bgeu, &RiscVInstOperands::branch(X28, X0, 4));
        // Verify opcode = BRANCH and funct3 = 111
        assert_eq!(word & 0x7F, BRANCH);
        assert_eq!((word >> 12) & 0x7, 0b111);
    }

    #[test]
    fn test_sb_and_sd_differ_in_funct3() {
        let sb = enc(RiscVOpcode::Sb, &RiscVInstOperands::store(X2, X10, 0));
        let sd = enc(RiscVOpcode::Sd, &RiscVInstOperands::store(X2, X10, 0));
        // Both have same opcode, differ in funct3
        assert_eq!(sb & 0x7F, STORE);
        assert_eq!(sd & 0x7F, STORE);
        assert_eq!((sb >> 12) & 0x7, 0b000); // SB funct3
        assert_eq!((sd >> 12) & 0x7, 0b011); // SD funct3
    }

    #[test]
    fn test_remu_x10_x11_x12() {
        // REMU x10, x11, x12
        // funct7=0000001, rs2=12, rs1=11, funct3=111, rd=10, opcode=0110011
        let word = enc(RiscVOpcode::Remu, &RiscVInstOperands::rrr(X10, X11, X12));
        let expected = encode_r_type(OP, 10, 0b111, 11, 12, 0b0000001);
        assert_eq!(word, expected);
    }

    #[test]
    fn test_remuw_x10_x11_x12() {
        // REMUW x10, x11, x12
        // funct7=0000001, rs2=12, rs1=11, funct3=111, rd=10, opcode=0111011
        let word = enc(RiscVOpcode::Remuw, &RiscVInstOperands::rrr(X10, X11, X12));
        let expected = encode_r_type(OP_32, 10, 0b111, 11, 12, 0b0000001);
        assert_eq!(word, expected);
    }

    #[test]
    fn test_slliw_x10_x11_5() {
        // SLLIW x10, x11, 5
        // imm[11:5]=0000000, shamt=5, rs1=11, funct3=001, rd=10, opcode=0011011
        // imm12 = 0b0000000_00101 = 5
        let word = enc(RiscVOpcode::Slliw, &RiscVInstOperands::rri(X10, X11, 5));
        let expected = encode_i_type(OP_IMM_32, 10, 0b001, 11, 5);
        assert_eq!(word, expected);
    }

    #[test]
    fn test_sraiw_x10_x11_3() {
        // SRAIW x10, x11, 3
        // imm[11:5]=0100000, shamt=3, rs1=11, funct3=101, rd=10, opcode=0011011
        // imm12 = 0b0100000_00011 = 0x403
        let word = enc(RiscVOpcode::Sraiw, &RiscVInstOperands::rri(X10, X11, 3));
        let expected = encode_i_type(OP_IMM_32, 10, 0b101, 11, 0x403);
        assert_eq!(word, expected);
    }

    #[test]
    fn test_feq_d_x10_f0_f1() {
        // FEQ.D x10, f0, f1 (result in GPR)
        // funct7=1010001, rs2=1, rs1=0, funct3=010, rd=10, opcode=1010011
        let word = enc(RiscVOpcode::FeqD, &RiscVInstOperands::rrr(X10, F0, F1));
        let expected = encode_r_type(OP_FP, 10, 0b010, 0, 1, 0b1010001);
        assert_eq!(word, expected);
    }

    #[test]
    fn test_fmv_x_d_x10_f11() {
        // FMV.X.D x10, f11
        // funct7=1110001, rs2=00000, rs1=11, funct3=000, rd=10, opcode=1010011
        let word = enc(RiscVOpcode::FmvXD, &RiscVInstOperands::rri(X10, F11, 0));
        let expected = encode_r_type(OP_FP, 10, 0b000, 11, 0b00000, 0b1110001);
        assert_eq!(word, expected);
    }
}
