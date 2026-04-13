// llvm2-codegen/aarch64/encode.rs - Unified AArch64 instruction encoder
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Unified AArch64 instruction encoder.
//!
//! Dispatches each `AArch64Opcode` to the correct low-level encoding function
//! from [`encoding`], [`encoding_mem`], or [`encoding_fp`]. This is the single
//! entry point used by the pipeline — no inline bit manipulation elsewhere.
//!
//! Design: the encoder takes an `IrMachInst` (post-regalloc, physical registers
//! only) and returns a 32-bit encoded instruction word.

use llvm2_ir::inst::{AArch64Opcode, MachInst};
use llvm2_ir::operand::MachOperand;
use llvm2_ir::regs::SpecialReg;

use super::encoding;
use super::encoding_fp::{self, FpArithOp, FpCmpOp, FpConvOp, FpSize};
use super::encoding_mem;

use thiserror::Error;

/// AArch64 NOP encoding (used as fallback for pseudos).
const NOP: u32 = 0xD503201F;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from unified instruction encoding.
#[derive(Debug, Error)]
pub enum EncodeError {
    #[error("unsupported opcode: {0:?}")]
    UnsupportedOpcode(AArch64Opcode),
    #[error("pseudo-instruction should not reach encoder: {0:?}")]
    PseudoInstruction(AArch64Opcode),
    #[error("operand {index} missing (opcode={opcode:?}, expected at least {expected})")]
    MissingOperand {
        opcode: AArch64Opcode,
        index: usize,
        expected: usize,
    },
    #[error("memory encoding error: {0}")]
    MemEncode(#[from] encoding_mem::EncodeError),
    #[error("FP encoding error: {0}")]
    FpEncode(#[from] encoding_fp::FpEncodeError),
}

// ---------------------------------------------------------------------------
// Operand extraction helpers
// ---------------------------------------------------------------------------

/// Extract the hardware register number from an operand (PReg or Special).
/// Defaults to 31 (XZR/SP) for non-register operands.
fn preg_hw(inst: &MachInst, idx: usize) -> u32 {
    match inst.operands.get(idx) {
        Some(MachOperand::PReg(p)) => p.hw_enc() as u32,
        Some(MachOperand::Special(s)) => match s {
            SpecialReg::SP | SpecialReg::XZR | SpecialReg::WZR => 31,
        },
        _ => 31,
    }
}

/// Extract an immediate value from an operand. Returns 0 for non-immediates.
fn imm_val(inst: &MachInst, idx: usize) -> i64 {
    match inst.operands.get(idx) {
        Some(MachOperand::Imm(v)) => *v,
        _ => 0,
    }
}

/// Determine sf (size flag): 1 for 64-bit GPR, 0 for 32-bit.
/// FPRs are handled separately. Defaults to 1 (64-bit).
fn sf_from_operand(inst: &MachInst, idx: usize) -> u32 {
    match inst.operands.get(idx) {
        Some(MachOperand::PReg(p)) => {
            if p.is_gpr() && p.encoding() < 32 {
                1 // All GPRs default to 64-bit in this model
            } else {
                1
            }
        }
        _ => 1,
    }
}

/// Check if an operand is an FPR (V-register).
fn is_fpr(inst: &MachInst, idx: usize) -> bool {
    match inst.operands.get(idx) {
        Some(MachOperand::PReg(p)) => p.is_fpr(),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Unified encoder
// ---------------------------------------------------------------------------

/// Encode a single `MachInst` into a 32-bit AArch64 instruction word.
///
/// This is the single dispatch point for all AArch64 opcodes. The instruction
/// must be post-regalloc (all operands are `PReg`, `Special`, `Imm`, etc.).
///
/// Pseudo-instructions (`Phi`, `StackAlloc`, `Nop`) emit NOP as a safe
/// fallback — the caller should skip pseudos before reaching this function.
pub fn encode_instruction(inst: &MachInst) -> Result<u32, EncodeError> {
    match inst.opcode {
        // =================================================================
        // Arithmetic (data-processing)
        // =================================================================

        // ADD Rd, Rn, Rm (shifted register)
        AArch64Opcode::AddRR => {
            let sf = sf_from_operand(inst, 0);
            Ok(encoding::encode_add_sub_shifted_reg(
                sf, 0, 0, 0, preg_hw(inst, 2), 0, preg_hw(inst, 1), preg_hw(inst, 0),
            ))
        }

        // ADD Rd, Rn, #imm12
        AArch64Opcode::AddRI => {
            let sf = sf_from_operand(inst, 0);
            let imm = imm_val(inst, 2) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf, 0, 0, 0, imm, preg_hw(inst, 1), preg_hw(inst, 0),
            ))
        }

        // SUB Rd, Rn, Rm (shifted register)
        AArch64Opcode::SubRR => {
            let sf = sf_from_operand(inst, 0);
            Ok(encoding::encode_add_sub_shifted_reg(
                sf, 1, 0, 0, preg_hw(inst, 2), 0, preg_hw(inst, 1), preg_hw(inst, 0),
            ))
        }

        // SUB Rd, Rn, #imm12
        AArch64Opcode::SubRI => {
            let sf = sf_from_operand(inst, 0);
            let imm = imm_val(inst, 2) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf, 1, 0, 0, imm, preg_hw(inst, 1), preg_hw(inst, 0),
            ))
        }

        // MUL Rd, Rn, Rm — encoded as MADD Rd, Rn, Rm, XZR
        // ARM ARM: Data-processing (3 source)
        // 31 30:29 28:24 23:21 20:16 15 14:10 9:5 4:0
        // sf 00    11011 000   Rm    0  Ra     Rn  Rd
        // MADD with Ra=XZR(31) = MUL
        AArch64Opcode::MulRR => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            let rm = preg_hw(inst, 2);
            let ra = 31u32; // XZR — MADD Rd, Rn, Rm, XZR = MUL
            Ok((sf << 31)
                | (0b00 << 29)
                | (0b11011 << 24)
                | (0b000 << 21)
                | (rm << 16)
                | (0 << 15) // o0 = 0 for MADD
                | (ra << 10)
                | (rn << 5)
                | rd)
        }

        // MSUB Rd, Rn, Rm, Ra — multiply-subtract: Rd = Ra - Rn * Rm
        // ARM ARM: Data-processing (3 source), o0=1
        // 31 30:29 28:24 23:21 20:16 15 14:10 9:5 4:0
        // sf 00    11011 000   Rm    1  Ra     Rn  Rd
        // When Ra=XZR (31), this is MNEG Rd, Rn, Rm.
        // Operands: [Rd, Rn, Rm, Ra] — 4 operands. If only 3, Ra defaults to XZR (MNEG).
        AArch64Opcode::Msub => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            let rm = preg_hw(inst, 2);
            let ra = if inst.operands.len() > 3 { preg_hw(inst, 3) } else { 31 };
            Ok((sf << 31)
                | (0b00 << 29)
                | (0b11011 << 24)
                | (0b000 << 21)
                | (rm << 16)
                | (1 << 15) // o0 = 1 for MSUB
                | (ra << 10)
                | (rn << 5)
                | rd)
        }

        // SMULL Xd, Wn, Wm — signed multiply long (alias for SMADDL Xd, Wn, Wm, XZR)
        // ARM ARM: Data-processing (3 source)
        // 31 30:29 28:24 23:21 20:16 15 14:10 9:5 4:0
        //  1  00    11011 001   Rm    0  Ra     Rn  Rd
        // sf=1 (always 64-bit result), U=0 (signed), o0=0 (add), Ra=XZR(31)
        AArch64Opcode::Smull => {
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            let rm = preg_hw(inst, 2);
            let ra = 31u32; // XZR for SMULL alias
            Ok((1u32 << 31) // sf = 1 (64-bit result)
                | (0b00 << 29)
                | (0b11011 << 24)
                | (0b001 << 21) // op54=00, op31=1 (long multiply)
                | (rm << 16)
                | (0 << 15) // o0 = 0 (SMADDL)
                | (ra << 10)
                | (rn << 5)
                | rd)
        }

        // UMULL Xd, Wn, Wm — unsigned multiply long (alias for UMADDL Xd, Wn, Wm, XZR)
        // ARM ARM: Data-processing (3 source)
        // 31 30:29 28:24 23:21 20:16 15 14:10 9:5 4:0
        //  1  00    11011 101   Rm    0  Ra     Rn  Rd
        // sf=1 (always 64-bit result), U=1 (unsigned), o0=0 (add), Ra=XZR(31)
        AArch64Opcode::Umull => {
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            let rm = preg_hw(inst, 2);
            let ra = 31u32; // XZR for UMULL alias
            Ok((1u32 << 31) // sf = 1 (64-bit result)
                | (0b00 << 29)
                | (0b11011 << 24)
                | (0b101 << 21) // op54=00, op31=1, U=1 (unsigned long multiply)
                | (rm << 16)
                | (0 << 15) // o0 = 0 (UMADDL)
                | (ra << 10)
                | (rn << 5)
                | rd)
        }

        // SDIV Rd, Rn, Rm — Data-processing (2 source)
        // 31 30 28:21      20:16 15:10  9:5  4:0
        // sf  0 0011010110  Rm   000011  Rn   Rd
        AArch64Opcode::SDiv => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            let rm = preg_hw(inst, 2);
            Ok((sf << 31)
                | (0b0_0011010110u32 << 21)
                | (rm << 16)
                | (0b000011 << 10)
                | (rn << 5)
                | rd)
        }

        // UDIV Rd, Rn, Rm
        // Same as SDIV but opcode field = 000010
        AArch64Opcode::UDiv => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            let rm = preg_hw(inst, 2);
            Ok((sf << 31)
                | (0b0_0011010110u32 << 21)
                | (rm << 16)
                | (0b000010 << 10)
                | (rn << 5)
                | rd)
        }

        // NEG Rd, Rm — alias for SUB Rd, XZR, Rm
        AArch64Opcode::Neg => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0);
            let rm = preg_hw(inst, 1);
            Ok(encoding::encode_add_sub_shifted_reg(
                sf, 1, 0, 0, rm, 0, 31, rd,
            ))
        }

        // =================================================================
        // Logical (shifted register)
        // =================================================================

        AArch64Opcode::AndRR => {
            let sf = sf_from_operand(inst, 0);
            Ok(encoding::encode_logical_shifted_reg(
                sf, 0b00, 0, 0, preg_hw(inst, 2), 0, preg_hw(inst, 1), preg_hw(inst, 0),
            ))
        }

        AArch64Opcode::OrrRR => {
            let sf = sf_from_operand(inst, 0);
            Ok(encoding::encode_logical_shifted_reg(
                sf, 0b01, 0, 0, preg_hw(inst, 2), 0, preg_hw(inst, 1), preg_hw(inst, 0),
            ))
        }

        AArch64Opcode::EorRR => {
            let sf = sf_from_operand(inst, 0);
            Ok(encoding::encode_logical_shifted_reg(
                sf, 0b10, 0, 0, preg_hw(inst, 2), 0, preg_hw(inst, 1), preg_hw(inst, 0),
            ))
        }

        // =================================================================
        // Shifts — Data-processing (2 source)
        // Variable shifts: LSL/LSR/ASR Rd, Rn, Rm
        // sf 0 0011010110 Rm opcode2 Rn Rd
        // opcode2: LSLV=001000, LSRV=001001, ASRV=001010
        // =================================================================

        AArch64Opcode::LslRR => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            let rm = preg_hw(inst, 2);
            Ok((sf << 31)
                | (0b0_0011010110u32 << 21)
                | (rm << 16)
                | (0b001000 << 10) // LSLV
                | (rn << 5)
                | rd)
        }

        AArch64Opcode::LsrRR => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            let rm = preg_hw(inst, 2);
            Ok((sf << 31)
                | (0b0_0011010110u32 << 21)
                | (rm << 16)
                | (0b001001 << 10) // LSRV
                | (rn << 5)
                | rd)
        }

        AArch64Opcode::AsrRR => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            let rm = preg_hw(inst, 2);
            Ok((sf << 31)
                | (0b0_0011010110u32 << 21)
                | (rm << 16)
                | (0b001010 << 10) // ASRV
                | (rn << 5)
                | rd)
        }

        // Immediate shifts — encoded via UBFM/SBFM
        // LSL Rd, Rn, #shift  = UBFM Rd, Rn, #(-shift MOD regsize), #(regsize-1-shift)
        // LSR Rd, Rn, #shift  = UBFM Rd, Rn, #shift, #(regsize-1)
        // ASR Rd, Rn, #shift  = SBFM Rd, Rn, #shift, #(regsize-1)
        // Bitfield format:
        // sf opc(2) 100110 N immr(6) imms(6) Rn Rd
        // UBFM: opc=10, SBFM: opc=00

        AArch64Opcode::LslRI => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            let shift = imm_val(inst, 2) as u32;
            let regsize = if sf == 1 { 64u32 } else { 32u32 };
            let n = sf; // N = sf for bitfield
            let immr = regsize.wrapping_sub(shift) & (regsize - 1);
            let imms = regsize - 1 - shift;
            // UBFM: sf opc=10 100110 N immr imms Rn Rd
            Ok((sf << 31)
                | (0b10 << 29)
                | (0b100110 << 23)
                | (n << 22)
                | (immr << 16)
                | (imms << 10)
                | (rn << 5)
                | rd)
        }

        AArch64Opcode::LsrRI => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            let shift = imm_val(inst, 2) as u32;
            let regsize = if sf == 1 { 64u32 } else { 32u32 };
            let n = sf;
            let immr = shift;
            let imms = regsize - 1;
            // UBFM
            Ok((sf << 31)
                | (0b10 << 29)
                | (0b100110 << 23)
                | (n << 22)
                | (immr << 16)
                | (imms << 10)
                | (rn << 5)
                | rd)
        }

        AArch64Opcode::AsrRI => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            let shift = imm_val(inst, 2) as u32;
            let regsize = if sf == 1 { 64u32 } else { 32u32 };
            let n = sf;
            let immr = shift;
            let imms = regsize - 1;
            // SBFM: opc=00
            Ok((sf << 31)
                | (0b00 << 29)
                | (0b100110 << 23)
                | (n << 22)
                | (immr << 16)
                | (imms << 10)
                | (rn << 5)
                | rd)
        }

        // =================================================================
        // Compare
        // =================================================================

        // CMP Rn, Rm = SUBS XZR, Rn, Rm
        AArch64Opcode::CmpRR => {
            let sf = 1u32; // default 64-bit
            Ok(encoding::encode_add_sub_shifted_reg(
                sf, 1, 1, 0, preg_hw(inst, 1), 0, preg_hw(inst, 0), 31,
            ))
        }

        // CMP Rn, #imm = SUBS XZR, Rn, #imm
        AArch64Opcode::CmpRI => {
            let sf = 1u32;
            let imm = imm_val(inst, 1) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf, 1, 1, 0, imm, preg_hw(inst, 0), 31,
            ))
        }

        // TST Rn, Rm = ANDS XZR, Rn, Rm
        AArch64Opcode::Tst => {
            let sf = 1u32;
            Ok(encoding::encode_logical_shifted_reg(
                sf, 0b11, 0, 0, preg_hw(inst, 1), 0, preg_hw(inst, 0), 31,
            ))
        }

        // =================================================================
        // Move
        // =================================================================

        // MOV Rd, Rm
        // When the source is SP (Special::SP), we must use ADD Rd, SP, #0
        // because register 31 in logical instructions (ORR) is XZR, not SP.
        // In ADD/SUB (immediate) context, register 31 is SP.
        // For all other registers, use ORR Rd, XZR, Rm.
        AArch64Opcode::MovR => {
            let sf = sf_from_operand(inst, 0);
            let is_sp_source = matches!(
                inst.operands.get(1),
                Some(MachOperand::Special(SpecialReg::SP))
            );
            if is_sp_source {
                // ADD Rd, SP, #0
                Ok(encoding::encode_add_sub_imm(
                    sf, 0, 0, 0, 0, 31, preg_hw(inst, 0),
                ))
            } else {
                // ORR Rd, XZR, Rm
                Ok(encoding::encode_logical_shifted_reg(
                    sf, 0b01, 0, 0, preg_hw(inst, 1), 0, 31, preg_hw(inst, 0),
                ))
            }
        }

        // MOVZ Rd, #imm16 (and MovI treated as MOVZ)
        AArch64Opcode::MovI | AArch64Opcode::Movz => {
            let sf = sf_from_operand(inst, 0);
            let imm16 = imm_val(inst, 1) as u32 & 0xFFFF;
            Ok(encoding::encode_move_wide(sf, 0b10, 0, imm16, preg_hw(inst, 0)))
        }

        // MOVK Rd, #imm16
        AArch64Opcode::Movk => {
            let sf = sf_from_operand(inst, 0);
            let imm16 = imm_val(inst, 1) as u32 & 0xFFFF;
            Ok(encoding::encode_move_wide(sf, 0b11, 0, imm16, preg_hw(inst, 0)))
        }

        // =================================================================
        // Load/Store
        // =================================================================

        // LDR Rt, [Rn, #offset]
        AArch64Opcode::LdrRI => {
            if is_fpr(inst, 0) {
                // FP load: V=1
                let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
                let scaled = (offset / 8) as u32 & 0xFFF;
                Ok(encoding::encode_load_store_ui(0b11, 1, 0b01, scaled, preg_hw(inst, 1), preg_hw(inst, 0)))
            } else {
                // Integer load: V=0, size=11 (64-bit)
                let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
                let scaled = (offset / 8) as u32 & 0xFFF;
                Ok(encoding::encode_load_store_ui(0b11, 0, 0b01, scaled, preg_hw(inst, 1), preg_hw(inst, 0)))
            }
        }

        // STR Rt, [Rn, #offset]
        AArch64Opcode::StrRI => {
            if is_fpr(inst, 0) {
                let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
                let scaled = (offset / 8) as u32 & 0xFFF;
                Ok(encoding::encode_load_store_ui(0b11, 1, 0b00, scaled, preg_hw(inst, 1), preg_hw(inst, 0)))
            } else {
                let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
                let scaled = (offset / 8) as u32 & 0xFFF;
                Ok(encoding::encode_load_store_ui(0b11, 0, 0b00, scaled, preg_hw(inst, 1), preg_hw(inst, 0)))
            }
        }

        // LDR literal (PC-relative) — uses the same base encoding but with the
        // literal addressing mode. For now encode as LDR unsigned offset with
        // an immediate operand interpreted as the literal pool offset.
        AArch64Opcode::LdrLiteral => {
            // LDR (literal): opc(2)=01 | 011 | V | 00 | imm19 | Rt
            // We encode 64-bit literal load: opc=01, V=0
            let imm19 = imm_val(inst, 1) as u32 & 0x7FFFF;
            let rt = preg_hw(inst, 0);
            Ok((0b01 << 30)
                | (0b011 << 27)
                | (0b00 << 24)
                | (imm19 << 5)
                | rt)
        }

        // STP Rt, Rt2, [Rn, #offset] (signed offset)
        AArch64Opcode::StpRI => {
            let offset = if inst.operands.len() > 3 { imm_val(inst, 3) } else { 0 };
            let scaled_imm7 = ((offset / 8) as i32 as u32) & 0x7F;
            Ok(encoding::encode_load_store_pair(
                0b10, 0, 0, scaled_imm7, preg_hw(inst, 1), preg_hw(inst, 2), preg_hw(inst, 0),
            ))
        }

        // STP Rt, Rt2, [Rn, #offset]! (pre-index: base updated before store)
        AArch64Opcode::StpPreIndex => {
            let offset = if inst.operands.len() > 3 { imm_val(inst, 3) } else { 0 };
            let scaled_imm7 = (offset / 8) as i8;
            Ok(encoding_mem::encode_ldp_stp_pre_index(
                encoding_mem::PairSize::X64,
                false,
                encoding_mem::PairOp::StorePair,
                scaled_imm7,
                preg_hw(inst, 1) as u8,
                preg_hw(inst, 2) as u8,
                preg_hw(inst, 0) as u8,
            )?)
        }

        // LDP Rt, Rt2, [Rn, #offset] (signed offset)
        AArch64Opcode::LdpRI => {
            let offset = if inst.operands.len() > 3 { imm_val(inst, 3) } else { 0 };
            let scaled_imm7 = ((offset / 8) as i32 as u32) & 0x7F;
            Ok(encoding::encode_load_store_pair(
                0b10, 0, 1, scaled_imm7, preg_hw(inst, 1), preg_hw(inst, 2), preg_hw(inst, 0),
            ))
        }

        // LDP Rt, Rt2, [Rn], #offset (post-index: base updated after load)
        AArch64Opcode::LdpPostIndex => {
            let offset = if inst.operands.len() > 3 { imm_val(inst, 3) } else { 0 };
            let scaled_imm7 = (offset / 8) as i8;
            Ok(encoding_mem::encode_ldp_stp_post_index(
                encoding_mem::PairSize::X64,
                false,
                encoding_mem::PairOp::LoadPair,
                scaled_imm7,
                preg_hw(inst, 1) as u8,
                preg_hw(inst, 2) as u8,
                preg_hw(inst, 0) as u8,
            )?)
        }

        // =================================================================
        // Branches
        // =================================================================

        // B <offset>
        AArch64Opcode::B => {
            let offset = imm_val(inst, 0) as u32 & 0x3FFFFFF;
            Ok(encoding::encode_uncond_branch(0, offset))
        }

        // B.cond <offset>
        AArch64Opcode::BCond => {
            let cond = imm_val(inst, 0) as u32 & 0xF;
            let offset = if inst.operands.len() > 1 {
                imm_val(inst, 1) as u32 & 0x7FFFF
            } else {
                0
            };
            Ok(encoding::encode_cond_branch(offset, cond))
        }

        // CBZ Rt, <offset>
        AArch64Opcode::Cbz => {
            let sf = sf_from_operand(inst, 0);
            let rt = preg_hw(inst, 0);
            let imm19 = if inst.operands.len() > 1 {
                imm_val(inst, 1) as u32 & 0x7FFFF
            } else {
                0
            };
            Ok(encoding::encode_cmp_branch(sf, 0, imm19, rt))
        }

        // CBNZ Rt, <offset>
        AArch64Opcode::Cbnz => {
            let sf = sf_from_operand(inst, 0);
            let rt = preg_hw(inst, 0);
            let imm19 = if inst.operands.len() > 1 {
                imm_val(inst, 1) as u32 & 0x7FFFF
            } else {
                0
            };
            Ok(encoding::encode_cmp_branch(sf, 1, imm19, rt))
        }

        // TBZ Rt, #bit, <offset>
        // 31 30:25  24  23:19  18:5    4:0
        // b5 011011  op  b40   imm14   Rt
        // TBZ: op=0, TBNZ: op=1
        AArch64Opcode::Tbz => {
            let rt = preg_hw(inst, 0);
            let bit = imm_val(inst, 1) as u32;
            let imm14 = if inst.operands.len() > 2 {
                imm_val(inst, 2) as u32 & 0x3FFF
            } else {
                0
            };
            let b5 = (bit >> 5) & 1;
            let b40 = bit & 0x1F;
            Ok((b5 << 31)
                | (0b011011 << 25)
                | (0 << 24) // op=0 for TBZ
                | (b40 << 19)
                | (imm14 << 5)
                | rt)
        }

        // TBNZ Rt, #bit, <offset>
        AArch64Opcode::Tbnz => {
            let rt = preg_hw(inst, 0);
            let bit = imm_val(inst, 1) as u32;
            let imm14 = if inst.operands.len() > 2 {
                imm_val(inst, 2) as u32 & 0x3FFF
            } else {
                0
            };
            let b5 = (bit >> 5) & 1;
            let b40 = bit & 0x1F;
            Ok((b5 << 31)
                | (0b011011 << 25)
                | (1 << 24) // op=1 for TBNZ
                | (b40 << 19)
                | (imm14 << 5)
                | rt)
        }

        // BR Rn
        AArch64Opcode::Br => {
            Ok(encoding::encode_branch_reg(0b0000, preg_hw(inst, 0)))
        }

        // BL <offset>
        AArch64Opcode::Bl => {
            let offset = imm_val(inst, 0) as u32 & 0x3FFFFFF;
            Ok(encoding::encode_uncond_branch(1, offset))
        }

        // BLR Rn
        AArch64Opcode::Blr => {
            Ok(encoding::encode_branch_reg(0b0001, preg_hw(inst, 0)))
        }

        // RET (X30)
        AArch64Opcode::Ret => {
            Ok(encoding::encode_branch_reg(0b0010, 30))
        }

        // =================================================================
        // Extension instructions — encoded via SBFM/UBFM
        // =================================================================

        // SXTW Rd, Rn = SBFM Xd, Xn, #0, #31
        AArch64Opcode::Sxtw => {
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            // sf=1, opc=00(SBFM), N=1, immr=0, imms=31
            Ok((1u32 << 31)
                | (0b00 << 29)
                | (0b100110 << 23)
                | (1 << 22) // N=1
                | (0 << 16) // immr=0
                | (31 << 10) // imms=31
                | (rn << 5)
                | rd)
        }

        // UXTW is a no-op on AArch64 (upper 32 bits of W-register write are
        // zeroed automatically). Encode as MOV Wd, Wn via ORR.
        AArch64Opcode::Uxtw => {
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            // MOV Wd, Wn = ORR Wd, WZR, Wn (sf=0)
            Ok(encoding::encode_logical_shifted_reg(
                0, 0b01, 0, 0, rn, 0, 31, rd,
            ))
        }

        // SXTB Rd, Rn = SBFM Xd, Xn, #0, #7
        AArch64Opcode::Sxtb => {
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            Ok((1u32 << 31)
                | (0b00 << 29)
                | (0b100110 << 23)
                | (1 << 22)
                | (0 << 16)
                | (7 << 10) // imms=7
                | (rn << 5)
                | rd)
        }

        // SXTH Rd, Rn = SBFM Xd, Xn, #0, #15
        AArch64Opcode::Sxth => {
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            Ok((1u32 << 31)
                | (0b00 << 29)
                | (0b100110 << 23)
                | (1 << 22)
                | (0 << 16)
                | (15 << 10) // imms=15
                | (rn << 5)
                | rd)
        }

        // =================================================================
        // Address generation
        // =================================================================

        // ADRP Rd, #imm21
        AArch64Opcode::Adrp => {
            let rd = preg_hw(inst, 0);
            let imm21 = imm_val(inst, 1) as i32;
            let enc = encoding_mem::encode_adrp(imm21, rd as u8)?;
            Ok(enc)
        }

        // ADD Rd, Rn, #imm (PC-relative page offset addition)
        AArch64Opcode::AddPCRel => {
            let sf = sf_from_operand(inst, 0);
            let imm = imm_val(inst, 2) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf, 0, 0, 0, imm, preg_hw(inst, 1), preg_hw(inst, 0),
            ))
        }

        // =================================================================
        // Floating-point arithmetic
        // =================================================================

        AArch64Opcode::FaddRR => {
            let fp_size = fp_size_from_inst(inst);
            let enc = encoding_fp::encode_fp_arith(
                fp_size, FpArithOp::Add,
                preg_hw(inst, 2) as u8, preg_hw(inst, 1) as u8, preg_hw(inst, 0) as u8,
            )?;
            Ok(enc)
        }

        AArch64Opcode::FsubRR => {
            let fp_size = fp_size_from_inst(inst);
            let enc = encoding_fp::encode_fp_arith(
                fp_size, FpArithOp::Sub,
                preg_hw(inst, 2) as u8, preg_hw(inst, 1) as u8, preg_hw(inst, 0) as u8,
            )?;
            Ok(enc)
        }

        AArch64Opcode::FmulRR => {
            let fp_size = fp_size_from_inst(inst);
            let enc = encoding_fp::encode_fp_arith(
                fp_size, FpArithOp::Mul,
                preg_hw(inst, 2) as u8, preg_hw(inst, 1) as u8, preg_hw(inst, 0) as u8,
            )?;
            Ok(enc)
        }

        AArch64Opcode::FdivRR => {
            let fp_size = fp_size_from_inst(inst);
            let enc = encoding_fp::encode_fp_arith(
                fp_size, FpArithOp::Div,
                preg_hw(inst, 2) as u8, preg_hw(inst, 1) as u8, preg_hw(inst, 0) as u8,
            )?;
            Ok(enc)
        }

        // FCMP Rn, Rm
        AArch64Opcode::Fcmp => {
            let fp_size = fp_size_from_cmp_inst(inst);
            let enc = encoding_fp::encode_fcmp(
                fp_size, FpCmpOp::Cmp,
                preg_hw(inst, 1) as u8, preg_hw(inst, 0) as u8,
            )?;
            Ok(enc)
        }

        // FCVTZS Rd, Rn (FP to signed integer, round toward zero)
        AArch64Opcode::FcvtzsRR => {
            // Source is FP register (operand 1), dest is GPR (operand 0)
            let sf_64 = true; // default to 64-bit integer
            let fp_size = fp_size_from_source(inst, 1);
            let enc = encoding_fp::encode_fp_int_conv(
                sf_64, fp_size, FpConvOp::FcvtzsToInt,
                preg_hw(inst, 1) as u8, preg_hw(inst, 0) as u8,
            )?;
            Ok(enc)
        }

        // SCVTF Rd, Rn (signed integer to FP)
        AArch64Opcode::ScvtfRR => {
            let sf_64 = true;
            let fp_size = fp_size_from_source(inst, 0);
            let enc = encoding_fp::encode_fp_int_conv(
                sf_64, fp_size, FpConvOp::ScvtfToFp,
                preg_hw(inst, 1) as u8, preg_hw(inst, 0) as u8,
            )?;
            Ok(enc)
        }

        // =================================================================
        // Checked arithmetic (flag-setting variants)
        // =================================================================

        // ADDS Rd, Rn, Rm (shifted register, flag-setting)
        AArch64Opcode::AddsRR => {
            let sf = sf_from_operand(inst, 0);
            Ok(encoding::encode_add_sub_shifted_reg(
                sf, 0, 1, 0, preg_hw(inst, 2), 0, preg_hw(inst, 1), preg_hw(inst, 0),
            ))
        }

        // ADDS Rd, Rn, #imm12 (flag-setting)
        AArch64Opcode::AddsRI => {
            let sf = sf_from_operand(inst, 0);
            let imm = imm_val(inst, 2) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf, 0, 1, 0, imm, preg_hw(inst, 1), preg_hw(inst, 0),
            ))
        }

        // SUBS Rd, Rn, Rm (shifted register, flag-setting)
        AArch64Opcode::SubsRR => {
            let sf = sf_from_operand(inst, 0);
            Ok(encoding::encode_add_sub_shifted_reg(
                sf, 1, 1, 0, preg_hw(inst, 2), 0, preg_hw(inst, 1), preg_hw(inst, 0),
            ))
        }

        // SUBS Rd, Rn, #imm12 (flag-setting)
        AArch64Opcode::SubsRI => {
            let sf = sf_from_operand(inst, 0);
            let imm = imm_val(inst, 2) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf, 1, 1, 0, imm, preg_hw(inst, 1), preg_hw(inst, 0),
            ))
        }

        // =================================================================
        // Pseudo-instructions — emit NOP as safe fallback
        // =================================================================

        // Trap pseudo-instructions — emit BRK #1 (debug breakpoint).
        AArch64Opcode::TrapOverflow
        | AArch64Opcode::TrapBoundsCheck
        | AArch64Opcode::TrapNull
        | AArch64Opcode::TrapDivZero
        | AArch64Opcode::TrapShiftRange => {
            Ok(0xD4200020) // BRK #1
        }

        AArch64Opcode::Phi
        | AArch64Opcode::StackAlloc
        | AArch64Opcode::Nop
        | AArch64Opcode::Retain
        | AArch64Opcode::Release => Ok(NOP),
    }
}

// ---------------------------------------------------------------------------
// FP size helpers
// ---------------------------------------------------------------------------

/// Determine FP precision from the destination register for FP arithmetic.
/// FPR encoding: 32-47 = S/D0-D15, etc. We default to Double (64-bit).
/// A more precise approach would track register class through ISel.
fn fp_size_from_inst(_inst: &MachInst) -> FpSize {
    // Default to double precision — the ISel would have attached size info
    // in a real implementation. For now, all FP ops are double.
    FpSize::Double
}

/// FP size for compare instructions (uses source operands).
fn fp_size_from_cmp_inst(_inst: &MachInst) -> FpSize {
    FpSize::Double
}

/// FP size derived from a specific source operand index.
fn fp_size_from_source(_inst: &MachInst, _idx: usize) -> FpSize {
    FpSize::Double
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::inst::{AArch64Opcode, MachInst};
    use llvm2_ir::operand::MachOperand;
    use llvm2_ir::regs::{PReg, SpecialReg, X0, X1, X2, X9, X30, V0, V1, V2};

    /// Helper to build a MachInst with given opcode and operands.
    fn mk(opcode: AArch64Opcode, ops: Vec<MachOperand>) -> MachInst {
        MachInst::new(opcode, ops)
    }

    fn preg(r: PReg) -> MachOperand {
        MachOperand::PReg(r)
    }

    fn imm(v: i64) -> MachOperand {
        MachOperand::Imm(v)
    }

    fn sp() -> MachOperand {
        MachOperand::Special(SpecialReg::SP)
    }

    // --- Verify unified encoder produces same output as direct encoding calls ---

    #[test]
    fn test_add_rr() {
        // ADD X0, X1, X2
        let inst = mk(AArch64Opcode::AddRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_add_sub_shifted_reg(1, 0, 0, 0, 2, 0, 1, 0);
        assert_eq!(enc, direct, "ADD X0, X1, X2: unified={enc:#010X}, direct={direct:#010X}");
    }

    #[test]
    fn test_add_ri() {
        // ADD X0, X1, #42
        let inst = mk(AArch64Opcode::AddRI, vec![preg(X0), preg(X1), imm(42)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_add_sub_imm(1, 0, 0, 0, 42, 1, 0);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_sub_rr() {
        let inst = mk(AArch64Opcode::SubRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_add_sub_shifted_reg(1, 1, 0, 0, 2, 0, 1, 0);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_sub_ri() {
        let inst = mk(AArch64Opcode::SubRI, vec![preg(X0), preg(X1), imm(42)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_add_sub_imm(1, 1, 0, 0, 42, 1, 0);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_mul_rr() {
        // MUL X0, X1, X2 = MADD X0, X1, X2, XZR
        let inst = mk(AArch64Opcode::MulRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        // Expected: sf=1, 00 11011 000 Rm=2 0 Ra=31 Rn=1 Rd=0
        let expected = (1u32 << 31) | (0b11011 << 24) | (2 << 16) | (31 << 10) | (1 << 5) | 0;
        assert_eq!(enc, expected, "MUL X0, X1, X2 = {enc:#010X}");
    }

    #[test]
    fn test_msub() {
        // MSUB X0, X1, X2, X9 — Rd = X9 - X1 * X2
        // ARM ARM: sf=1 00 11011 000 Rm=2 1 Ra=9 Rn=1 Rd=0
        let inst = mk(
            AArch64Opcode::Msub,
            vec![preg(X0), preg(X1), preg(X2), preg(X9)],
        );
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31)
            | (0b11011 << 24)
            | (2 << 16)
            | (1 << 15) // o0 = 1 for MSUB
            | (9 << 10) // Ra = X9
            | (1 << 5)
            | 0;
        assert_eq!(enc, expected, "MSUB X0, X1, X2, X9 = {enc:#010X}");
    }

    #[test]
    fn test_msub_mneg_alias() {
        // MNEG X0, X1, X2 = MSUB X0, X1, X2, XZR (3 operands, Ra defaults to XZR)
        let inst = mk(
            AArch64Opcode::Msub,
            vec![preg(X0), preg(X1), preg(X2)],
        );
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31)
            | (0b11011 << 24)
            | (2 << 16)
            | (1 << 15) // o0 = 1 for MSUB
            | (31 << 10) // Ra = XZR (31)
            | (1 << 5)
            | 0;
        assert_eq!(enc, expected, "MNEG X0, X1, X2 = {enc:#010X}");
    }

    #[test]
    fn test_smull() {
        // SMULL X0, W1, W2 = SMADDL X0, W1, W2, XZR
        // ARM ARM: sf=1 00 11011 001 Rm=2 0 Ra=31 Rn=1 Rd=0
        let inst = mk(
            AArch64Opcode::Smull,
            vec![preg(X0), preg(X1), preg(X2)],
        );
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31)
            | (0b11011 << 24)
            | (0b001 << 21) // signed long multiply
            | (2 << 16)
            | (0 << 15) // o0 = 0
            | (31 << 10) // Ra = XZR
            | (1 << 5)
            | 0;
        assert_eq!(enc, expected, "SMULL X0, W1, W2 = {enc:#010X}");
    }

    #[test]
    fn test_umull() {
        // UMULL X0, W1, W2 = UMADDL X0, W1, W2, XZR
        // ARM ARM: sf=1 00 11011 101 Rm=2 0 Ra=31 Rn=1 Rd=0
        let inst = mk(
            AArch64Opcode::Umull,
            vec![preg(X0), preg(X1), preg(X2)],
        );
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31)
            | (0b11011 << 24)
            | (0b101 << 21) // unsigned long multiply
            | (2 << 16)
            | (0 << 15) // o0 = 0
            | (31 << 10) // Ra = XZR
            | (1 << 5)
            | 0;
        assert_eq!(enc, expected, "UMULL X0, W1, W2 = {enc:#010X}");
    }

    #[test]
    fn test_adds_rr() {
        // ADDS X0, X1, X2
        let inst = mk(AArch64Opcode::AddsRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_add_sub_shifted_reg(1, 0, 1, 0, 2, 0, 1, 0);
        assert_eq!(enc, direct, "ADDS X0, X1, X2: unified={enc:#010X}, direct={direct:#010X}");
    }

    #[test]
    fn test_subs_rr() {
        // SUBS X0, X1, X2
        let inst = mk(AArch64Opcode::SubsRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_add_sub_shifted_reg(1, 1, 1, 0, 2, 0, 1, 0);
        assert_eq!(enc, direct, "SUBS X0, X1, X2: unified={enc:#010X}, direct={direct:#010X}");
    }

    #[test]
    fn test_sdiv() {
        // SDIV X0, X1, X2
        let inst = mk(AArch64Opcode::SDiv, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        // sf=1 0 0011010110 Rm=2 000011 Rn=1 Rd=0
        let expected = (1u32 << 31) | (0b0_0011010110 << 21) | (2 << 16) | (0b000011 << 10) | (1 << 5) | 0;
        assert_eq!(enc, expected, "SDIV X0, X1, X2 = {enc:#010X}");
    }

    #[test]
    fn test_udiv() {
        let inst = mk(AArch64Opcode::UDiv, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31) | (0b0_0011010110 << 21) | (2 << 16) | (0b000010 << 10) | (1 << 5) | 0;
        assert_eq!(enc, expected, "UDIV X0, X1, X2 = {enc:#010X}");
    }

    #[test]
    fn test_neg() {
        // NEG X0, X1 = SUB X0, XZR, X1
        let inst = mk(AArch64Opcode::Neg, vec![preg(X0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_add_sub_shifted_reg(1, 1, 0, 0, 1, 0, 31, 0);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_and_rr() {
        let inst = mk(AArch64Opcode::AndRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_logical_shifted_reg(1, 0b00, 0, 0, 2, 0, 1, 0);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_orr_rr() {
        let inst = mk(AArch64Opcode::OrrRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_logical_shifted_reg(1, 0b01, 0, 0, 2, 0, 1, 0);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_eor_rr() {
        let inst = mk(AArch64Opcode::EorRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_logical_shifted_reg(1, 0b10, 0, 0, 2, 0, 1, 0);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_mov_r() {
        // MOV X0, X1 = ORR X0, XZR, X1
        let inst = mk(AArch64Opcode::MovR, vec![preg(X0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_logical_shifted_reg(1, 0b01, 0, 0, 1, 0, 31, 0);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_movz() {
        let inst = mk(AArch64Opcode::Movz, vec![preg(X0), imm(0x1234)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_move_wide(1, 0b10, 0, 0x1234, 0);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_movk() {
        let inst = mk(AArch64Opcode::Movk, vec![preg(X0), imm(0x5678)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_move_wide(1, 0b11, 0, 0x5678, 0);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_ldr_ri() {
        // LDR X0, [X1, #8]
        let inst = mk(AArch64Opcode::LdrRI, vec![preg(X0), preg(X1), imm(8)]);
        let enc = encode_instruction(&inst).unwrap();
        // offset 8 / 8 = 1 (scaled)
        let direct = encoding::encode_load_store_ui(0b11, 0, 0b01, 1, 1, 0);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_str_ri() {
        // STR X0, [X1]
        let inst = mk(AArch64Opcode::StrRI, vec![preg(X0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_load_store_ui(0b11, 0, 0b00, 0, 1, 0);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_stp() {
        // STP X0, X1, [SP, #16]
        let inst = mk(AArch64Opcode::StpRI, vec![sp(), preg(X0), preg(X1), imm(16)]);
        let enc = encode_instruction(&inst).unwrap();
        // offset 16 / 8 = 2, scaled imm7 = 2
        let direct = encoding::encode_load_store_pair(0b10, 0, 0, 2, 0, 1, 31);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_ldp() {
        // LDP X0, X1, [SP]
        let inst = mk(AArch64Opcode::LdpRI, vec![sp(), preg(X0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_load_store_pair(0b10, 0, 1, 0, 0, 1, 31);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_b() {
        let inst = mk(AArch64Opcode::B, vec![imm(2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_uncond_branch(0, 2);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_bcond() {
        // B.EQ <+8>: cond=0, offset=2
        let inst = mk(AArch64Opcode::BCond, vec![imm(0), imm(2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_cond_branch(2, 0);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_bl() {
        let inst = mk(AArch64Opcode::Bl, vec![imm(2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_uncond_branch(1, 2);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_blr() {
        let inst = mk(AArch64Opcode::Blr, vec![preg(X0)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_branch_reg(0b0001, 0);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_ret() {
        let inst = mk(AArch64Opcode::Ret, vec![]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_branch_reg(0b0010, 30);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_cbz() {
        // CBZ X0, <+8>
        let inst = mk(AArch64Opcode::Cbz, vec![preg(X0), imm(2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_cmp_branch(1, 0, 2, 0);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_cbnz() {
        // CBNZ X0, <+8>
        let inst = mk(AArch64Opcode::Cbnz, vec![preg(X0), imm(2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_cmp_branch(1, 1, 2, 0);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_cmp_rr() {
        let inst = mk(AArch64Opcode::CmpRR, vec![preg(X0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_add_sub_shifted_reg(1, 1, 1, 0, 1, 0, 0, 31);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_cmp_ri() {
        let inst = mk(AArch64Opcode::CmpRI, vec![preg(X0), imm(42)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_add_sub_imm(1, 1, 1, 0, 42, 0, 31);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_tst() {
        let inst = mk(AArch64Opcode::Tst, vec![preg(X0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_logical_shifted_reg(1, 0b11, 0, 0, 1, 0, 0, 31);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_adrp() {
        let inst = mk(AArch64Opcode::Adrp, vec![preg(X0), imm(1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_mem::encode_adrp(1, 0).unwrap();
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_fadd() {
        let inst = mk(AArch64Opcode::FaddRR, vec![preg(V0), preg(V1), preg(V2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_fp::encode_fp_arith(FpSize::Double, FpArithOp::Add, 2, 1, 0).unwrap();
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_fsub() {
        let inst = mk(AArch64Opcode::FsubRR, vec![preg(V0), preg(V1), preg(V2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_fp::encode_fp_arith(FpSize::Double, FpArithOp::Sub, 2, 1, 0).unwrap();
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_fmul() {
        let inst = mk(AArch64Opcode::FmulRR, vec![preg(V0), preg(V1), preg(V2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_fp::encode_fp_arith(FpSize::Double, FpArithOp::Mul, 2, 1, 0).unwrap();
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_fdiv() {
        let inst = mk(AArch64Opcode::FdivRR, vec![preg(V0), preg(V1), preg(V2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_fp::encode_fp_arith(FpSize::Double, FpArithOp::Div, 2, 1, 0).unwrap();
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_fcmp() {
        let inst = mk(AArch64Opcode::Fcmp, vec![preg(V0), preg(V1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_fp::encode_fcmp(FpSize::Double, FpCmpOp::Cmp, 1, 0).unwrap();
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_fcvtzs() {
        // FCVTZS X0, D1
        let inst = mk(AArch64Opcode::FcvtzsRR, vec![preg(X0), preg(V1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_fp::encode_fp_int_conv(
            true, FpSize::Double, FpConvOp::FcvtzsToInt, 1, 0,
        ).unwrap();
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_scvtf() {
        // SCVTF D0, X1
        let inst = mk(AArch64Opcode::ScvtfRR, vec![preg(V0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_fp::encode_fp_int_conv(
            true, FpSize::Double, FpConvOp::ScvtfToFp, 1, 0,
        ).unwrap();
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_sxtw() {
        // SXTW X0, X1 = SBFM X0, X1, #0, #31
        let inst = mk(AArch64Opcode::Sxtw, vec![preg(X0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31) | (0b00 << 29) | (0b100110 << 23)
            | (1 << 22) | (31 << 10) | (1 << 5) | 0;
        assert_eq!(enc, expected, "SXTW X0, X1 = {enc:#010X}");
    }

    #[test]
    fn test_lsl_rr() {
        // LSLV X0, X1, X2
        let inst = mk(AArch64Opcode::LslRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31) | (0b0_0011010110 << 21) | (2 << 16)
            | (0b001000 << 10) | (1 << 5) | 0;
        assert_eq!(enc, expected, "LSLV X0, X1, X2 = {enc:#010X}");
    }

    #[test]
    fn test_lsr_rr() {
        let inst = mk(AArch64Opcode::LsrRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31) | (0b0_0011010110 << 21) | (2 << 16)
            | (0b001001 << 10) | (1 << 5) | 0;
        assert_eq!(enc, expected);
    }

    #[test]
    fn test_asr_rr() {
        let inst = mk(AArch64Opcode::AsrRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31) | (0b0_0011010110 << 21) | (2 << 16)
            | (0b001010 << 10) | (1 << 5) | 0;
        assert_eq!(enc, expected);
    }

    #[test]
    fn test_br() {
        let inst = mk(AArch64Opcode::Br, vec![preg(X30)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_branch_reg(0b0000, 30);
        assert_eq!(enc, direct);
    }

    #[test]
    fn test_pseudos_emit_nop() {
        for opcode in [AArch64Opcode::Phi, AArch64Opcode::StackAlloc, AArch64Opcode::Nop] {
            let inst = mk(opcode, vec![]);
            let enc = encode_instruction(&inst).unwrap();
            assert_eq!(enc, NOP, "{opcode:?} should emit NOP");
        }
    }

    #[test]
    fn test_all_opcodes_handled() {
        // Verify that every opcode in the enum produces a result (not an error).
        // We use minimal operands — the point is to verify dispatch coverage,
        // not encoding correctness (that's tested per-opcode above).
        let test_cases: Vec<(AArch64Opcode, Vec<MachOperand>)> = vec![
            (AArch64Opcode::AddRR, vec![preg(X0), preg(X1), preg(X2)]),
            (AArch64Opcode::AddRI, vec![preg(X0), preg(X1), imm(0)]),
            (AArch64Opcode::SubRR, vec![preg(X0), preg(X1), preg(X2)]),
            (AArch64Opcode::SubRI, vec![preg(X0), preg(X1), imm(0)]),
            (AArch64Opcode::MulRR, vec![preg(X0), preg(X1), preg(X2)]),
            (AArch64Opcode::Msub, vec![preg(X0), preg(X1), preg(X2), preg(X9)]),
            (AArch64Opcode::Smull, vec![preg(X0), preg(X1), preg(X2)]),
            (AArch64Opcode::Umull, vec![preg(X0), preg(X1), preg(X2)]),
            (AArch64Opcode::SDiv, vec![preg(X0), preg(X1), preg(X2)]),
            (AArch64Opcode::UDiv, vec![preg(X0), preg(X1), preg(X2)]),
            (AArch64Opcode::Neg, vec![preg(X0), preg(X1)]),
            (AArch64Opcode::AndRR, vec![preg(X0), preg(X1), preg(X2)]),
            (AArch64Opcode::OrrRR, vec![preg(X0), preg(X1), preg(X2)]),
            (AArch64Opcode::EorRR, vec![preg(X0), preg(X1), preg(X2)]),
            (AArch64Opcode::LslRR, vec![preg(X0), preg(X1), preg(X2)]),
            (AArch64Opcode::LsrRR, vec![preg(X0), preg(X1), preg(X2)]),
            (AArch64Opcode::AsrRR, vec![preg(X0), preg(X1), preg(X2)]),
            (AArch64Opcode::LslRI, vec![preg(X0), preg(X1), imm(3)]),
            (AArch64Opcode::LsrRI, vec![preg(X0), preg(X1), imm(3)]),
            (AArch64Opcode::AsrRI, vec![preg(X0), preg(X1), imm(3)]),
            (AArch64Opcode::CmpRR, vec![preg(X0), preg(X1)]),
            (AArch64Opcode::CmpRI, vec![preg(X0), imm(0)]),
            (AArch64Opcode::Tst, vec![preg(X0), preg(X1)]),
            (AArch64Opcode::MovR, vec![preg(X0), preg(X1)]),
            (AArch64Opcode::MovI, vec![preg(X0), imm(0)]),
            (AArch64Opcode::Movz, vec![preg(X0), imm(0)]),
            (AArch64Opcode::Movk, vec![preg(X0), imm(0)]),
            (AArch64Opcode::LdrRI, vec![preg(X0), preg(X1)]),
            (AArch64Opcode::StrRI, vec![preg(X0), preg(X1)]),
            (AArch64Opcode::LdrLiteral, vec![preg(X0), imm(0)]),
            (AArch64Opcode::LdpRI, vec![sp(), preg(X0), preg(X1)]),
            (AArch64Opcode::StpRI, vec![sp(), preg(X0), preg(X1)]),
            (AArch64Opcode::B, vec![imm(0)]),
            (AArch64Opcode::BCond, vec![imm(0)]),
            (AArch64Opcode::Cbz, vec![preg(X0)]),
            (AArch64Opcode::Cbnz, vec![preg(X0)]),
            (AArch64Opcode::Tbz, vec![preg(X0), imm(0)]),
            (AArch64Opcode::Tbnz, vec![preg(X0), imm(0)]),
            (AArch64Opcode::Br, vec![preg(X0)]),
            (AArch64Opcode::Bl, vec![imm(0)]),
            (AArch64Opcode::Blr, vec![preg(X0)]),
            (AArch64Opcode::Ret, vec![]),
            (AArch64Opcode::Sxtw, vec![preg(X0), preg(X1)]),
            (AArch64Opcode::Uxtw, vec![preg(X0), preg(X1)]),
            (AArch64Opcode::Sxtb, vec![preg(X0), preg(X1)]),
            (AArch64Opcode::Sxth, vec![preg(X0), preg(X1)]),
            (AArch64Opcode::FaddRR, vec![preg(V0), preg(V1), preg(V2)]),
            (AArch64Opcode::FsubRR, vec![preg(V0), preg(V1), preg(V2)]),
            (AArch64Opcode::FmulRR, vec![preg(V0), preg(V1), preg(V2)]),
            (AArch64Opcode::FdivRR, vec![preg(V0), preg(V1), preg(V2)]),
            (AArch64Opcode::Fcmp, vec![preg(V0), preg(V1)]),
            (AArch64Opcode::FcvtzsRR, vec![preg(X0), preg(V1)]),
            (AArch64Opcode::ScvtfRR, vec![preg(V0), preg(X1)]),
            (AArch64Opcode::Adrp, vec![preg(X0), imm(0)]),
            (AArch64Opcode::AddPCRel, vec![preg(X0), preg(X1), imm(0)]),
            (AArch64Opcode::AddsRR, vec![preg(X0), preg(X1), preg(X2)]),
            (AArch64Opcode::AddsRI, vec![preg(X0), preg(X1), imm(0)]),
            (AArch64Opcode::SubsRR, vec![preg(X0), preg(X1), preg(X2)]),
            (AArch64Opcode::SubsRI, vec![preg(X0), preg(X1), imm(0)]),
            (AArch64Opcode::TrapOverflow, vec![]),
            (AArch64Opcode::TrapBoundsCheck, vec![]),
            (AArch64Opcode::TrapNull, vec![]),
            (AArch64Opcode::TrapDivZero, vec![]),
            (AArch64Opcode::TrapShiftRange, vec![]),
            (AArch64Opcode::Retain, vec![]),
            (AArch64Opcode::Release, vec![]),
            (AArch64Opcode::Phi, vec![]),
            (AArch64Opcode::StackAlloc, vec![]),
            (AArch64Opcode::Nop, vec![]),
        ];

        for (opcode, ops) in test_cases {
            let inst = mk(opcode, ops);
            let result = encode_instruction(&inst);
            assert!(result.is_ok(), "Opcode {opcode:?} should encode successfully, got: {result:?}");
        }
    }
}
