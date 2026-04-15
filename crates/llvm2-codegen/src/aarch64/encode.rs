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
use llvm2_ir::regs::{preg_class, RegClass, SpecialReg};

use super::encoding;
use super::encoding_fp::{self, FpArithOp, FpCmpOp, FpConvOp, FpSize};
use super::encoding_mem;
use super::encoding_neon;

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
    #[error("NEON encoding error: {0}")]
    NeonEncode(#[from] encoding_neon::NeonEncodeError),
    #[error("unsupported FP size {size:?} for opcode {opcode:?}")]
    UnsupportedFpSize {
        opcode: AArch64Opcode,
        size: FpSize,
    },
    #[error("invalid operand at index {index} for {opcode:?}: expected register, got {desc}")]
    InvalidOperand {
        opcode: AArch64Opcode,
        index: usize,
        desc: String,
    },
}

// ---------------------------------------------------------------------------
// Operand extraction helpers
// ---------------------------------------------------------------------------

/// Extract the hardware register number from an operand (PReg or Special).
/// Returns an error for non-register operands instead of silently defaulting
/// to XZR (reg 31), which would produce wrong code (#174).
fn preg_hw(inst: &MachInst, idx: usize) -> Result<u32, EncodeError> {
    match inst.operands.get(idx) {
        Some(MachOperand::PReg(p)) => Ok(p.hw_enc() as u32),
        Some(MachOperand::Special(s)) => match s {
            SpecialReg::SP | SpecialReg::XZR | SpecialReg::WZR => Ok(31),
        },
        Some(other) => Err(EncodeError::InvalidOperand {
            opcode: inst.opcode,
            index: idx,
            desc: format!("{:?}", other),
        }),
        None => Err(EncodeError::MissingOperand {
            opcode: inst.opcode,
            index: idx,
            expected: idx + 1,
        }),
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
/// Uses the register's class to determine size:
///   - Gpr32 (W registers) → sf=0
///   - Gpr64 (X registers) → sf=1
/// FPRs are handled separately. Defaults to 1 (64-bit).
fn sf_from_operand(inst: &MachInst, idx: usize) -> u32 {
    match inst.operands.get(idx) {
        Some(MachOperand::PReg(p)) => {
            match preg_class(*p) {
                RegClass::Gpr32 => 0,
                RegClass::Gpr64 => 1,
                // FPRs and system regs: sf not applicable, default to 1
                _ => 1,
            }
        }
        Some(MachOperand::Special(s)) => match s {
            // SP and XZR are 64-bit, WZR is 32-bit
            SpecialReg::SP | SpecialReg::XZR => 1,
            SpecialReg::WZR => 0,
        },
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
                sf, 0, 0, 0, preg_hw(inst, 2)?, 0, preg_hw(inst, 1)?, preg_hw(inst, 0)?,
            ))
        }

        // ADD Rd, Rn, #imm12
        AArch64Opcode::AddRI => {
            let sf = sf_from_operand(inst, 0);
            let imm = imm_val(inst, 2) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf, 0, 0, 0, imm, preg_hw(inst, 1)?, preg_hw(inst, 0)?,
            ))
        }

        // SUB Rd, Rn, Rm (shifted register)
        AArch64Opcode::SubRR => {
            let sf = sf_from_operand(inst, 0);
            Ok(encoding::encode_add_sub_shifted_reg(
                sf, 1, 0, 0, preg_hw(inst, 2)?, 0, preg_hw(inst, 1)?, preg_hw(inst, 0)?,
            ))
        }

        // SUB Rd, Rn, #imm12
        AArch64Opcode::SubRI => {
            let sf = sf_from_operand(inst, 0);
            let imm = imm_val(inst, 2) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf, 1, 0, 0, imm, preg_hw(inst, 1)?, preg_hw(inst, 0)?,
            ))
        }

        // MUL Rd, Rn, Rm — encoded as MADD Rd, Rn, Rm, XZR
        // ARM ARM: Data-processing (3 source)
        // 31 30:29 28:24 23:21 20:16 15 14:10 9:5 4:0
        // sf 00    11011 000   Rm    0  Ra     Rn  Rd
        // MADD with Ra=XZR(31) = MUL
        AArch64Opcode::MulRR => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let rm = preg_hw(inst, 2)?;
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
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let rm = preg_hw(inst, 2)?;
            let ra = if inst.operands.len() > 3 { preg_hw(inst, 3)? } else { 31 };
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
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let rm = preg_hw(inst, 2)?;
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
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let rm = preg_hw(inst, 2)?;
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
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let rm = preg_hw(inst, 2)?;
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
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let rm = preg_hw(inst, 2)?;
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
            let rd = preg_hw(inst, 0)?;
            let rm = preg_hw(inst, 1)?;
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
                sf, 0b00, 0, 0, preg_hw(inst, 2)?, 0, preg_hw(inst, 1)?, preg_hw(inst, 0)?,
            ))
        }

        AArch64Opcode::OrrRR => {
            let sf = sf_from_operand(inst, 0);
            Ok(encoding::encode_logical_shifted_reg(
                sf, 0b01, 0, 0, preg_hw(inst, 2)?, 0, preg_hw(inst, 1)?, preg_hw(inst, 0)?,
            ))
        }

        AArch64Opcode::EorRR => {
            let sf = sf_from_operand(inst, 0);
            Ok(encoding::encode_logical_shifted_reg(
                sf, 0b10, 0, 0, preg_hw(inst, 2)?, 0, preg_hw(inst, 1)?, preg_hw(inst, 0)?,
            ))
        }

        // ORN Rd, Rn, Rm — bitwise OR-NOT (MVN when Rn=XZR)
        // Logical shifted register with N=1, opc=01
        AArch64Opcode::OrnRR => {
            let sf = sf_from_operand(inst, 0);
            Ok(encoding::encode_logical_shifted_reg(
                sf, 0b01, 1, 0, preg_hw(inst, 2)?, 0, preg_hw(inst, 1)?, preg_hw(inst, 0)?,
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
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let rm = preg_hw(inst, 2)?;
            Ok((sf << 31)
                | (0b0_0011010110u32 << 21)
                | (rm << 16)
                | (0b001000 << 10) // LSLV
                | (rn << 5)
                | rd)
        }

        AArch64Opcode::LsrRR => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let rm = preg_hw(inst, 2)?;
            Ok((sf << 31)
                | (0b0_0011010110u32 << 21)
                | (rm << 16)
                | (0b001001 << 10) // LSRV
                | (rn << 5)
                | rd)
        }

        AArch64Opcode::AsrRR => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let rm = preg_hw(inst, 2)?;
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
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
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
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
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
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
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
            let sf = sf_from_operand(inst, 0);
            Ok(encoding::encode_add_sub_shifted_reg(
                sf, 1, 1, 0, preg_hw(inst, 1)?, 0, preg_hw(inst, 0)?, 31,
            ))
        }

        // CMP Rn, #imm = SUBS XZR, Rn, #imm
        AArch64Opcode::CmpRI => {
            let sf = sf_from_operand(inst, 0);
            let imm = imm_val(inst, 1) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf, 1, 1, 0, imm, preg_hw(inst, 0)?, 31,
            ))
        }

        // TST Rn, Rm = ANDS XZR, Rn, Rm
        AArch64Opcode::Tst => {
            let sf = sf_from_operand(inst, 0);
            Ok(encoding::encode_logical_shifted_reg(
                sf, 0b11, 0, 0, preg_hw(inst, 1)?, 0, preg_hw(inst, 0)?, 31,
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
                    sf, 0, 0, 0, 0, 31, preg_hw(inst, 0)?,
                ))
            } else {
                // ORR Rd, XZR, Rm
                Ok(encoding::encode_logical_shifted_reg(
                    sf, 0b01, 0, 0, preg_hw(inst, 1)?, 0, 31, preg_hw(inst, 0)?,
                ))
            }
        }

        // CSET Xd/Wd, cond — encoded as CSINC Xd, XZR, XZR, invert(cond)
        // ARM ARM C6.2.70: sf | 0 | 0 | 11010100 | Rm(=XZR) | inv_cond | 0 | 1 | Rn(=XZR) | Rd
        // Operands: [PReg(Rd), Imm(cond_encoding)]
        AArch64Opcode::CSet => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0)?;
            let cond = imm_val(inst, 1) as u32 & 0xF;
            // Invert condition code: flip bit 0 (ARM ARM C6.2.70)
            let inv_cond = cond ^ 1;
            let rn = 31u32; // XZR/WZR
            let rm = 31u32; // XZR/WZR
            Ok((sf << 31)
                | (0b00 << 29)
                | (0b11010100 << 21)
                | (rm << 16)
                | (inv_cond << 12)
                | (0b01 << 10)     // op2 = 01 (CSINC)
                | (rn << 5)
                | rd)
        }

        // MOVZ Rd, #imm16 (and MovI treated as MOVZ)
        AArch64Opcode::MovI | AArch64Opcode::Movz => {
            let sf = sf_from_operand(inst, 0);
            let imm16 = imm_val(inst, 1) as u32 & 0xFFFF;
            Ok(encoding::encode_move_wide(sf, 0b10, 0, imm16, preg_hw(inst, 0)?))
        }

        // MOVK Rd, #imm16
        AArch64Opcode::Movk => {
            let sf = sf_from_operand(inst, 0);
            let imm16 = imm_val(inst, 1) as u32 & 0xFFFF;
            Ok(encoding::encode_move_wide(sf, 0b11, 0, imm16, preg_hw(inst, 0)?))
        }

        // =================================================================
        // Load/Store
        // =================================================================

        // LDR Rt, [Rn, #offset]
        AArch64Opcode::LdrRI => {
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            if is_fpr(inst, 0) {
                // FP load: V=1, size from FP register class
                let fp_sz = fp_size_from_inst(inst);
                let (size, scale) = match fp_sz {
                    FpSize::Half => (0b01, 2i64),     // 16-bit FP (H registers)
                    FpSize::Single => (0b10, 4i64),   // 32-bit FP (S registers)
                    FpSize::Double => (0b11, 8i64),   // 64-bit FP (D registers)
                };
                let scaled = (offset / scale) as u32 & 0xFFF;
                Ok(encoding::encode_load_store_ui(size, 1, 0b01, scaled, preg_hw(inst, 1)?, preg_hw(inst, 0)?))
            } else {
                // Integer load: V=0, size from register class
                let sf = sf_from_operand(inst, 0);
                let (size, scale) = if sf == 1 {
                    (0b11, 8i64) // 64-bit (X registers)
                } else {
                    (0b10, 4i64) // 32-bit (W registers)
                };
                let scaled = (offset / scale) as u32 & 0xFFF;
                Ok(encoding::encode_load_store_ui(size, 0, 0b01, scaled, preg_hw(inst, 1)?, preg_hw(inst, 0)?))
            }
        }

        // STR Rt, [Rn, #offset]
        AArch64Opcode::StrRI => {
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            if is_fpr(inst, 0) {
                let fp_sz = fp_size_from_inst(inst);
                let (size, scale) = match fp_sz {
                    FpSize::Half => (0b01, 2i64),     // 16-bit FP (H registers)
                    FpSize::Single => (0b10, 4i64),   // 32-bit FP (S registers)
                    FpSize::Double => (0b11, 8i64),   // 64-bit FP (D registers)
                };
                let scaled = (offset / scale) as u32 & 0xFFF;
                Ok(encoding::encode_load_store_ui(size, 1, 0b00, scaled, preg_hw(inst, 1)?, preg_hw(inst, 0)?))
            } else {
                let sf = sf_from_operand(inst, 0);
                let (size, scale) = if sf == 1 {
                    (0b11, 8i64)
                } else {
                    (0b10, 4i64)
                };
                let scaled = (offset / scale) as u32 & 0xFFF;
                Ok(encoding::encode_load_store_ui(size, 0, 0b00, scaled, preg_hw(inst, 1)?, preg_hw(inst, 0)?))
            }
        }

        // LDRB Wt, [Xn, #offset] — load byte, zero-extend to 32-bit
        // size=00, V=0, opc=01. Offset scaled by 1 byte.
        AArch64Opcode::LdrbRI => {
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            let scaled = offset as u32 & 0xFFF; // byte-scaled (scale=1)
            Ok(encoding::encode_load_store_ui(0b00, 0, 0b01, scaled, preg_hw(inst, 1)?, preg_hw(inst, 0)?))
        }

        // LDRH Wt, [Xn, #offset] — load halfword, zero-extend to 32-bit
        // size=01, V=0, opc=01. Offset scaled by 2 bytes.
        AArch64Opcode::LdrhRI => {
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            let scaled = (offset / 2) as u32 & 0xFFF; // halfword-scaled (scale=2)
            Ok(encoding::encode_load_store_ui(0b01, 0, 0b01, scaled, preg_hw(inst, 1)?, preg_hw(inst, 0)?))
        }

        // LDRSB Wt, [Xn, #offset] — load byte, sign-extend to 32-bit
        // size=00, V=0, opc=11 (sign-extend to 32-bit). Offset scaled by 1.
        AArch64Opcode::LdrsbRI => {
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            let scaled = offset as u32 & 0xFFF;
            Ok(encoding::encode_load_store_ui(0b00, 0, 0b11, scaled, preg_hw(inst, 1)?, preg_hw(inst, 0)?))
        }

        // LDRSH Wt, [Xn, #offset] — load halfword, sign-extend to 32-bit
        // size=01, V=0, opc=11 (sign-extend to 32-bit). Offset scaled by 2.
        AArch64Opcode::LdrshRI => {
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            let scaled = (offset / 2) as u32 & 0xFFF;
            Ok(encoding::encode_load_store_ui(0b01, 0, 0b11, scaled, preg_hw(inst, 1)?, preg_hw(inst, 0)?))
        }

        // STRB Wt, [Xn, #offset] — store byte (truncating)
        // size=00, V=0, opc=00. Offset scaled by 1.
        AArch64Opcode::StrbRI => {
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            let scaled = offset as u32 & 0xFFF;
            Ok(encoding::encode_load_store_ui(0b00, 0, 0b00, scaled, preg_hw(inst, 1)?, preg_hw(inst, 0)?))
        }

        // STRH Wt, [Xn, #offset] — store halfword (truncating)
        // size=01, V=0, opc=00. Offset scaled by 2.
        AArch64Opcode::StrhRI => {
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            let scaled = (offset / 2) as u32 & 0xFFF;
            Ok(encoding::encode_load_store_ui(0b01, 0, 0b00, scaled, preg_hw(inst, 1)?, preg_hw(inst, 0)?))
        }

        // LDR literal (PC-relative) — uses the same base encoding but with the
        // literal addressing mode. For now encode as LDR unsigned offset with
        // an immediate operand interpreted as the literal pool offset.
        AArch64Opcode::LdrLiteral => {
            // LDR (literal): opc(2)=01 | 011 | V | 00 | imm19 | Rt
            // We encode 64-bit literal load: opc=01, V=0
            let imm19 = imm_val(inst, 1) as u32 & 0x7FFFF;
            let rt = preg_hw(inst, 0)?;
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
                0b10, 0, 0, scaled_imm7, preg_hw(inst, 1)?, preg_hw(inst, 2)?, preg_hw(inst, 0)?,
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
                preg_hw(inst, 1)? as u8,
                preg_hw(inst, 2)? as u8,
                preg_hw(inst, 0)? as u8,
            )?)
        }

        // LDP Rt, Rt2, [Rn, #offset] (signed offset)
        AArch64Opcode::LdpRI => {
            let offset = if inst.operands.len() > 3 { imm_val(inst, 3) } else { 0 };
            let scaled_imm7 = ((offset / 8) as i32 as u32) & 0x7F;
            Ok(encoding::encode_load_store_pair(
                0b10, 0, 1, scaled_imm7, preg_hw(inst, 1)?, preg_hw(inst, 2)?, preg_hw(inst, 0)?,
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
                preg_hw(inst, 1)? as u8,
                preg_hw(inst, 2)? as u8,
                preg_hw(inst, 0)? as u8,
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
            let rt = preg_hw(inst, 0)?;
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
            let rt = preg_hw(inst, 0)?;
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
            let rt = preg_hw(inst, 0)?;
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
            let rt = preg_hw(inst, 0)?;
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
            Ok(encoding::encode_branch_reg(0b0000, preg_hw(inst, 0)?))
        }

        // BL <offset>
        AArch64Opcode::Bl => {
            let offset = imm_val(inst, 0) as u32 & 0x3FFFFFF;
            Ok(encoding::encode_uncond_branch(1, offset))
        }

        // BLR Rn
        AArch64Opcode::Blr => {
            Ok(encoding::encode_branch_reg(0b0001, preg_hw(inst, 0)?))
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
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
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
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            // MOV Wd, Wn = ORR Wd, WZR, Wn (sf=0)
            Ok(encoding::encode_logical_shifted_reg(
                0, 0b01, 0, 0, rn, 0, 31, rd,
            ))
        }

        // SXTB Rd, Rn = SBFM Xd, Xn, #0, #7
        AArch64Opcode::Sxtb => {
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
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
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            Ok((1u32 << 31)
                | (0b00 << 29)
                | (0b100110 << 23)
                | (1 << 22)
                | (0 << 16)
                | (15 << 10) // imms=15
                | (rn << 5)
                | rd)
        }

        // UXTB Wd, Wn = UBFM Wd, Wn, #0, #7
        // Zero-extend byte: clear bits [31:8], keep bits [7:0].
        AArch64Opcode::Uxtb => {
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            // sf=0, opc=10(UBFM), N=0, immr=0, imms=7
            Ok((0u32 << 31)
                | (0b10 << 29)
                | (0b100110 << 23)
                | (0 << 22) // N=0 (32-bit)
                | (0 << 16) // immr=0
                | (7 << 10) // imms=7
                | (rn << 5)
                | rd)
        }

        // UXTH Wd, Wn = UBFM Wd, Wn, #0, #15
        // Zero-extend halfword: clear bits [31:16], keep bits [15:0].
        AArch64Opcode::Uxth => {
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            // sf=0, opc=10(UBFM), N=0, immr=0, imms=15
            Ok((0u32 << 31)
                | (0b10 << 29)
                | (0b100110 << 23)
                | (0 << 22) // N=0 (32-bit)
                | (0 << 16) // immr=0
                | (15 << 10) // imms=15
                | (rn << 5)
                | rd)
        }

        // =================================================================
        // Address generation
        // =================================================================

        // ADRP Rd, #imm21
        AArch64Opcode::Adrp => {
            let rd = preg_hw(inst, 0)?;
            let imm21 = imm_val(inst, 1) as i32;
            let enc = encoding_mem::encode_adrp(imm21, rd as u8)?;
            Ok(enc)
        }

        // ADD Rd, Rn, #imm (PC-relative page offset addition)
        AArch64Opcode::AddPCRel => {
            let sf = sf_from_operand(inst, 0);
            let imm = imm_val(inst, 2) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf, 0, 0, 0, imm, preg_hw(inst, 1)?, preg_hw(inst, 0)?,
            ))
        }

        // =================================================================
        // Floating-point arithmetic
        // =================================================================

        AArch64Opcode::FaddRR => {
            let fp_size = fp_size_from_inst(inst);
            let enc = encoding_fp::encode_fp_arith(
                fp_size, FpArithOp::Add,
                preg_hw(inst, 2)? as u8, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        AArch64Opcode::FsubRR => {
            let fp_size = fp_size_from_inst(inst);
            let enc = encoding_fp::encode_fp_arith(
                fp_size, FpArithOp::Sub,
                preg_hw(inst, 2)? as u8, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        AArch64Opcode::FmulRR => {
            let fp_size = fp_size_from_inst(inst);
            let enc = encoding_fp::encode_fp_arith(
                fp_size, FpArithOp::Mul,
                preg_hw(inst, 2)? as u8, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        AArch64Opcode::FdivRR => {
            let fp_size = fp_size_from_inst(inst);
            let enc = encoding_fp::encode_fp_arith(
                fp_size, FpArithOp::Div,
                preg_hw(inst, 2)? as u8, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // FNEG Dd, Dn — floating-point negate (1-source FP)
        AArch64Opcode::FnegRR => {
            let fp_size = fp_size_from_inst(inst);
            let enc = encoding_fp::encode_fp_unary(
                fp_size, encoding_fp::FpUnaryOp::Fneg,
                preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // FABS Dd, Dn — floating-point absolute value (1-source FP)
        AArch64Opcode::FabsRR => {
            let fp_size = fp_size_from_inst(inst);
            let enc = encoding_fp::encode_fp_unary(
                fp_size, encoding_fp::FpUnaryOp::Fabs,
                preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // FSQRT Dd, Dn — floating-point square root (1-source FP)
        AArch64Opcode::FsqrtRR => {
            let fp_size = fp_size_from_inst(inst);
            let enc = encoding_fp::encode_fp_unary(
                fp_size, encoding_fp::FpUnaryOp::Fsqrt,
                preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // FCMP Rn, Rm
        AArch64Opcode::Fcmp => {
            let fp_size = fp_size_from_cmp_inst(inst);
            let enc = encoding_fp::encode_fcmp(
                fp_size, FpCmpOp::Cmp,
                preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
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
                preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // SCVTF Rd, Rn (signed integer to FP)
        AArch64Opcode::ScvtfRR => {
            let sf_64 = true;
            let fp_size = fp_size_from_source(inst, 0);
            let enc = encoding_fp::encode_fp_int_conv(
                sf_64, fp_size, FpConvOp::ScvtfToFp,
                preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // FCVTZU Rd, Rn (FP to unsigned integer, round toward zero)
        AArch64Opcode::FcvtzuRR => {
            let sf_64 = true;
            let fp_size = fp_size_from_source(inst, 1);
            let enc = encoding_fp::encode_fp_int_conv(
                sf_64, fp_size, FpConvOp::FcvtzuToInt,
                preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // UCVTF Rd, Rn (unsigned integer to FP)
        AArch64Opcode::UcvtfRR => {
            let sf_64 = true;
            let fp_size = fp_size_from_source(inst, 0);
            let enc = encoding_fp::encode_fp_int_conv(
                sf_64, fp_size, FpConvOp::UcvtfToFp,
                preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // FCVT Dd, Sn (float precision widen: f32 -> f64)
        AArch64Opcode::FcvtSD => {
            let enc = encoding_fp::encode_fp_precision_cvt(
                FpSize::Single, FpSize::Double,
                preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // FCVT Ss, Dn (float precision narrow: f64 -> f32)
        AArch64Opcode::FcvtDS => {
            let enc = encoding_fp::encode_fp_precision_cvt(
                FpSize::Double, FpSize::Single,
                preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // FMOV Sd, Wn / FMOV Dd, Xn (GPR to FPR bitcast)
        AArch64Opcode::FmovGprFpr => {
            let sf_64 = true; // conservative default
            let fp_size = fp_size_from_source(inst, 0);
            let enc = encoding_fp::encode_fp_int_conv(
                sf_64, fp_size, FpConvOp::FmovToFp,
                preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // FMOV Wn, Sd / FMOV Xn, Dd (FPR to GPR bitcast)
        AArch64Opcode::FmovFprGpr => {
            let sf_64 = true;
            let fp_size = fp_size_from_source(inst, 1);
            let enc = encoding_fp::encode_fp_int_conv(
                sf_64, fp_size, FpConvOp::FmovToGp,
                preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
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
                sf, 0, 1, 0, preg_hw(inst, 2)?, 0, preg_hw(inst, 1)?, preg_hw(inst, 0)?,
            ))
        }

        // ADDS Rd, Rn, #imm12 (flag-setting)
        AArch64Opcode::AddsRI => {
            let sf = sf_from_operand(inst, 0);
            let imm = imm_val(inst, 2) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf, 0, 1, 0, imm, preg_hw(inst, 1)?, preg_hw(inst, 0)?,
            ))
        }

        // SUBS Rd, Rn, Rm (shifted register, flag-setting)
        AArch64Opcode::SubsRR => {
            let sf = sf_from_operand(inst, 0);
            Ok(encoding::encode_add_sub_shifted_reg(
                sf, 1, 1, 0, preg_hw(inst, 2)?, 0, preg_hw(inst, 1)?, preg_hw(inst, 0)?,
            ))
        }

        // SUBS Rd, Rn, #imm12 (flag-setting)
        AArch64Opcode::SubsRI => {
            let sf = sf_from_operand(inst, 0);
            let imm = imm_val(inst, 2) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf, 1, 1, 0, imm, preg_hw(inst, 1)?, preg_hw(inst, 0)?,
            ))
        }

        // =================================================================
        // Conditional select / set
        // =================================================================

        // CSEL Xd, Xn, Xm, cond
        // ARM ARM: C6.2.76  Conditional Select
        // 31 30 29 28:21    20:16 15:12 11 10 9:5  4:0
        // sf  0  0  11010100 Rm    cond   0  0  Rn   Rd
        // Operands: [dst, true_src, false_src, Imm(cond_code_encoding)]
        AArch64Opcode::Csel => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let rm = preg_hw(inst, 2)?;
            let cond = imm_val(inst, 3) as u32 & 0xF;
            Ok((sf << 31)
                | (0b00 << 29)
                | (0b11010100 << 21)
                | (rm << 16)
                | (cond << 12)
                | (0b00 << 10)
                | (rn << 5)
                | rd)
        }

        // =================================================================
        // NEON SIMD (vector) instructions
        // =================================================================

        // Integer vector arithmetic: ADD, SUB, MUL
        AArch64Opcode::NeonAddV => {
            let arr = neon_arrangement(inst);
            let enc = encoding_neon::encode_int_vec3_same(
                arr, encoding_neon::IntVec3Op::Add,
                preg_hw(inst, 2)? as u8, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        AArch64Opcode::NeonSubV => {
            let arr = neon_arrangement(inst);
            let enc = encoding_neon::encode_int_vec3_same(
                arr, encoding_neon::IntVec3Op::Sub,
                preg_hw(inst, 2)? as u8, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        AArch64Opcode::NeonMulV => {
            let arr = neon_arrangement(inst);
            let enc = encoding_neon::encode_int_vec3_same(
                arr, encoding_neon::IntVec3Op::Mul,
                preg_hw(inst, 2)? as u8, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // FP vector arithmetic: FADD, FSUB, FMUL, FDIV
        AArch64Opcode::NeonFaddV => {
            let fp_arr = neon_fp_arrangement(inst);
            let enc = encoding_neon::encode_fp_vec3_same(
                fp_arr, encoding_neon::FpVec3Op::Fadd,
                preg_hw(inst, 2)? as u8, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        AArch64Opcode::NeonFsubV => {
            let fp_arr = neon_fp_arrangement(inst);
            let enc = encoding_neon::encode_fp_vec3_same(
                fp_arr, encoding_neon::FpVec3Op::Fsub,
                preg_hw(inst, 2)? as u8, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        AArch64Opcode::NeonFmulV => {
            let fp_arr = neon_fp_arrangement(inst);
            let enc = encoding_neon::encode_fp_vec3_same(
                fp_arr, encoding_neon::FpVec3Op::Fmul,
                preg_hw(inst, 2)? as u8, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        AArch64Opcode::NeonFdivV => {
            let fp_arr = neon_fp_arrangement(inst);
            let enc = encoding_neon::encode_fp_vec3_same(
                fp_arr, encoding_neon::FpVec3Op::Fdiv,
                preg_hw(inst, 2)? as u8, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // Vector logic: AND, ORR, EOR, BIC
        AArch64Opcode::NeonAndV => {
            let q = neon_q_bit(inst);
            let enc = encoding_neon::encode_vec_logic(
                q, encoding_neon::VecLogicOp::And,
                preg_hw(inst, 2)? as u8, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        AArch64Opcode::NeonOrrV => {
            let q = neon_q_bit(inst);
            let enc = encoding_neon::encode_vec_logic(
                q, encoding_neon::VecLogicOp::Orr,
                preg_hw(inst, 2)? as u8, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        AArch64Opcode::NeonEorV => {
            let q = neon_q_bit(inst);
            let enc = encoding_neon::encode_vec_logic(
                q, encoding_neon::VecLogicOp::Eor,
                preg_hw(inst, 2)? as u8, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        AArch64Opcode::NeonBicV => {
            let q = neon_q_bit(inst);
            let enc = encoding_neon::encode_vec_logic(
                q, encoding_neon::VecLogicOp::Bic,
                preg_hw(inst, 2)? as u8, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // Vector NOT
        AArch64Opcode::NeonNotV => {
            let q = neon_q_bit(inst);
            let enc = encoding_neon::encode_vec_not(
                q, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // Vector compare: CMEQ, CMGT, CMGE
        AArch64Opcode::NeonCmeqV => {
            let arr = neon_arrangement(inst);
            let enc = encoding_neon::encode_int_vec3_same(
                arr, encoding_neon::IntVec3Op::Cmeq,
                preg_hw(inst, 2)? as u8, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        AArch64Opcode::NeonCmgtV => {
            let arr = neon_arrangement(inst);
            let enc = encoding_neon::encode_int_vec3_same(
                arr, encoding_neon::IntVec3Op::Cmgt,
                preg_hw(inst, 2)? as u8, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        AArch64Opcode::NeonCmgeV => {
            let arr = neon_arrangement(inst);
            let enc = encoding_neon::encode_int_vec3_same(
                arr, encoding_neon::IntVec3Op::Cmge,
                preg_hw(inst, 2)? as u8, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // DUP (element): duplicate one vector lane to all lanes
        // Operands: [Vd, Vn, Imm(lane), Imm(element_size)]
        AArch64Opcode::NeonDupElem => {
            let q = neon_q_bit(inst);
            let lane = imm_val(inst, 2) as u8;
            let elem = neon_element_size(inst, 3);
            let enc = encoding_neon::encode_dup_element(
                q, elem, lane, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // DUP (general): duplicate GPR to all vector lanes
        // Operands: [Vd, Rn, Imm(element_size)]
        AArch64Opcode::NeonDupGen => {
            let q = neon_q_bit(inst);
            let elem = neon_element_size(inst, 2);
            let enc = encoding_neon::encode_dup_general(
                q, elem, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // INS (general): insert GPR into vector lane
        // Operands: [Vd, Rn, Imm(lane), Imm(element_size)]
        AArch64Opcode::NeonInsGen => {
            let lane = imm_val(inst, 2) as u8;
            let elem = neon_element_size(inst, 3);
            let enc = encoding_neon::encode_ins_general(
                elem, lane, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // MOVI: move immediate to vector (byte form)
        // Operands: [Vd, Imm(imm8)]
        AArch64Opcode::NeonMovi => {
            let q = neon_q_bit(inst);
            let imm8 = imm_val(inst, 1) as u8;
            let enc = encoding_neon::encode_movi_byte(
                q, imm8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // LD1 post-index: SIMD load 1 register
        // Operands: [Vt, Xn, Imm(arrangement)]
        AArch64Opcode::NeonLd1Post => {
            let arr = neon_arrangement(inst);
            let enc = encoding_neon::encode_ld1_post_imm(
                arr, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
        }

        // ST1 post-index: SIMD store 1 register
        // Operands: [Vt, Xn, Imm(arrangement)]
        AArch64Opcode::NeonSt1Post => {
            let arr = neon_arrangement(inst);
            let enc = encoding_neon::encode_st1_post_imm(
                arr, preg_hw(inst, 1)? as u8, preg_hw(inst, 0)? as u8,
            )?;
            Ok(enc)
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
        | AArch64Opcode::Copy
        | AArch64Opcode::Nop
        | AArch64Opcode::Retain
        | AArch64Opcode::Release => Ok(NOP),

        // =================================================================
        // Bitfield move instructions (ARM ARM C6.2)
        // =================================================================

        // UBFM Rd, Rn, #immr, #imms — unsigned bitfield move
        // sf | 10 | 100110 | N | immr(6) | imms(6) | Rn(5) | Rd(5)
        // Aliases: LSL/LSR (imm), UBFX, UXTB, UXTH
        // Operands: [Rd, Rn, Imm(immr), Imm(imms)]
        AArch64Opcode::Ubfm => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let immr = imm_val(inst, 2) as u32 & 0x3F;
            let imms = imm_val(inst, 3) as u32 & 0x3F;
            let n = sf; // N = sf for 64-bit, 0 for 32-bit
            Ok((sf << 31)
                | (0b10 << 29) // opc = 10 (UBFM)
                | (0b100110 << 23)
                | (n << 22)
                | (immr << 16)
                | (imms << 10)
                | (rn << 5)
                | rd)
        }

        // SBFM Rd, Rn, #immr, #imms — signed bitfield move
        // sf | 00 | 100110 | N | immr(6) | imms(6) | Rn(5) | Rd(5)
        // Aliases: ASR (imm), SBFX, SXTB, SXTH, SXTW
        // Operands: [Rd, Rn, Imm(immr), Imm(imms)]
        AArch64Opcode::Sbfm => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let immr = imm_val(inst, 2) as u32 & 0x3F;
            let imms = imm_val(inst, 3) as u32 & 0x3F;
            let n = sf;
            Ok((sf << 31)
                | (0b00 << 29) // opc = 00 (SBFM)
                | (0b100110 << 23)
                | (n << 22)
                | (immr << 16)
                | (imms << 10)
                | (rn << 5)
                | rd)
        }

        // BFM Rd, Rn, #immr, #imms — bitfield move (insert)
        // sf | 01 | 100110 | N | immr(6) | imms(6) | Rn(5) | Rd(5)
        // Aliases: BFI, BFXIL
        // Operands: [Rd, Rn, Imm(immr), Imm(imms)]
        AArch64Opcode::Bfm => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let immr = imm_val(inst, 2) as u32 & 0x3F;
            let imms = imm_val(inst, 3) as u32 & 0x3F;
            let n = sf;
            Ok((sf << 31)
                | (0b01 << 29) // opc = 01 (BFM)
                | (0b100110 << 23)
                | (n << 22)
                | (immr << 16)
                | (imms << 10)
                | (rn << 5)
                | rd)
        }

        // =================================================================
        // Load/Store register offset (ARM ARM C6.2)
        // =================================================================

        // LDR Rt, [Rn, Rm{, extend {#amount}}] — register offset load
        // size | 111 | V | 00 | opc | 1 | Rm(5) | option(3) | S(1) | 10 | Rn(5) | Rt(5)
        // Operands: [Rt, Rn, Rm] — default LSL, no shift (S=0)
        // Optional 4th operand: Imm(extend_option_and_shift) packed as (option<<1)|S
        AArch64Opcode::LdrRO => {
            let sf = sf_from_operand(inst, 0);
            let rt = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let rm = preg_hw(inst, 2)?;
            // Default: LSL extend (option=011), no shift (S=0)
            let (option, s) = if inst.operands.len() > 3 {
                let packed = imm_val(inst, 3) as u32;
                ((packed >> 1) & 0b111, packed & 1)
            } else {
                (0b011, 0)
            };
            if is_fpr(inst, 0) {
                // FP register-offset load: V=1
                let fp_sz = fp_size_from_inst(inst);
                let mem_size = match fp_sz {
                    FpSize::Half => encoding_mem::LoadStoreSize::Half,       // 16-bit (H registers)
                    FpSize::Single => encoding_mem::LoadStoreSize::Word,     // 32-bit (S registers)
                    FpSize::Double => encoding_mem::LoadStoreSize::Double,   // 64-bit (D registers)
                };
                Ok(encoding_mem::encode_ldr_str_register(
                    mem_size,
                    true,
                    encoding_mem::LoadStoreOp::Load,
                    rm as u8,
                    match option { 0b010 => encoding_mem::RegExtend::Uxtw, 0b110 => encoding_mem::RegExtend::Sxtw, 0b111 => encoding_mem::RegExtend::Sxtx, _ => encoding_mem::RegExtend::Lsl },
                    s != 0,
                    rn as u8,
                    rt as u8,
                )?)
            } else {
                // Integer register-offset load: V=0
                let size = if sf == 1 { encoding_mem::LoadStoreSize::Double } else { encoding_mem::LoadStoreSize::Word };
                Ok(encoding_mem::encode_ldr_str_register(
                    size,
                    false,
                    encoding_mem::LoadStoreOp::Load,
                    rm as u8,
                    match option { 0b010 => encoding_mem::RegExtend::Uxtw, 0b110 => encoding_mem::RegExtend::Sxtw, 0b111 => encoding_mem::RegExtend::Sxtx, _ => encoding_mem::RegExtend::Lsl },
                    s != 0,
                    rn as u8,
                    rt as u8,
                )?)
            }
        }

        // STR Rt, [Rn, Rm{, extend {#amount}}] — register offset store
        // Same encoding format as LdrRO but with opc=00 (store)
        // Operands: [Rt, Rn, Rm] — default LSL, no shift (S=0)
        AArch64Opcode::StrRO => {
            let sf = sf_from_operand(inst, 0);
            let rt = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let rm = preg_hw(inst, 2)?;
            let (option, s) = if inst.operands.len() > 3 {
                let packed = imm_val(inst, 3) as u32;
                ((packed >> 1) & 0b111, packed & 1)
            } else {
                (0b011, 0)
            };
            if is_fpr(inst, 0) {
                let fp_sz = fp_size_from_inst(inst);
                let mem_size = match fp_sz {
                    FpSize::Half => encoding_mem::LoadStoreSize::Half,       // 16-bit (H registers)
                    FpSize::Single => encoding_mem::LoadStoreSize::Word,     // 32-bit (S registers)
                    FpSize::Double => encoding_mem::LoadStoreSize::Double,   // 64-bit (D registers)
                };
                Ok(encoding_mem::encode_ldr_str_register(
                    mem_size,
                    true,
                    encoding_mem::LoadStoreOp::Store,
                    rm as u8,
                    match option { 0b010 => encoding_mem::RegExtend::Uxtw, 0b110 => encoding_mem::RegExtend::Sxtw, 0b111 => encoding_mem::RegExtend::Sxtx, _ => encoding_mem::RegExtend::Lsl },
                    s != 0,
                    rn as u8,
                    rt as u8,
                )?)
            } else {
                let size = if sf == 1 { encoding_mem::LoadStoreSize::Double } else { encoding_mem::LoadStoreSize::Word };
                Ok(encoding_mem::encode_ldr_str_register(
                    size,
                    false,
                    encoding_mem::LoadStoreOp::Store,
                    rm as u8,
                    match option { 0b010 => encoding_mem::RegExtend::Uxtw, 0b110 => encoding_mem::RegExtend::Sxtw, 0b111 => encoding_mem::RegExtend::Sxtx, _ => encoding_mem::RegExtend::Lsl },
                    s != 0,
                    rn as u8,
                    rt as u8,
                )?)
            }
        }

        // =================================================================
        // GOT and TLV loads
        // =================================================================

        // LdrGot — LDR Xd, [Xn, #offset] from GOT slot
        // Encoded as a standard 64-bit unsigned-offset load. The relocation
        // for the GOT page offset is handled by the relocation layer; the
        // encoder just emits: LDR Xd, [Xn, #imm12] (size=11, V=0, opc=01).
        // Operands: [Rd, Rn, Imm(scaled_offset)]
        AArch64Opcode::LdrGot => {
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            let scaled = (offset / 8) as u32 & 0xFFF;
            // Always 64-bit load (GOT entries are pointer-sized)
            Ok(encoding::encode_load_store_ui(0b11, 0, 0b01, scaled, rn, rd))
        }

        // LdrTlvp — LDR Xd, [Xn, #offset] from TLV descriptor
        // Same encoding as LdrGot: standard 64-bit unsigned-offset load.
        // The TLV page offset relocation is handled separately.
        // Operands: [Rd, Rn, Imm(scaled_offset)]
        AArch64Opcode::LdrTlvp => {
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            let scaled = (offset / 8) as u32 & 0xFFF;
            // Always 64-bit load (TLV descriptors are pointer-sized)
            Ok(encoding::encode_load_store_ui(0b11, 0, 0b01, scaled, rn, rd))
        }

        // =================================================================
        // Logical immediate (ARM ARM C4.1.4 — Logical (immediate))
        // sf | opc(2) | 100100 | N | immr(6) | imms(6) | Rn(5) | Rd(5)
        // AND=00, ORR=01, EOR=10
        // Operands: [Rd, Rn, Imm(N), Imm(immr), Imm(imms)]
        // =================================================================

        AArch64Opcode::AndRI => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let n = imm_val(inst, 2) as u32 & 1;
            let immr = imm_val(inst, 3) as u32 & 0x3F;
            let imms = imm_val(inst, 4) as u32 & 0x3F;
            Ok((sf << 31)
                | (0b00 << 29) // opc = 00 (AND)
                | (0b100100 << 23)
                | (n << 22)
                | (immr << 16)
                | (imms << 10)
                | (rn << 5)
                | rd)
        }

        AArch64Opcode::OrrRI => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let n = imm_val(inst, 2) as u32 & 1;
            let immr = imm_val(inst, 3) as u32 & 0x3F;
            let imms = imm_val(inst, 4) as u32 & 0x3F;
            Ok((sf << 31)
                | (0b01 << 29) // opc = 01 (ORR)
                | (0b100100 << 23)
                | (n << 22)
                | (immr << 16)
                | (imms << 10)
                | (rn << 5)
                | rd)
        }

        AArch64Opcode::EorRI => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let n = imm_val(inst, 2) as u32 & 1;
            let immr = imm_val(inst, 3) as u32 & 0x3F;
            let imms = imm_val(inst, 4) as u32 & 0x3F;
            Ok((sf << 31)
                | (0b10 << 29) // opc = 10 (EOR)
                | (0b100100 << 23)
                | (n << 22)
                | (immr << 16)
                | (imms << 10)
                | (rn << 5)
                | rd)
        }

        // =================================================================
        // BIC — Bitwise AND-NOT (bit clear)
        // Logical shifted register: opc=00, N=1
        // =================================================================

        AArch64Opcode::BicRR => {
            let sf = sf_from_operand(inst, 0);
            Ok(encoding::encode_logical_shifted_reg(
                sf, 0b00, 0, 1, preg_hw(inst, 2)?, 0, preg_hw(inst, 1)?, preg_hw(inst, 0)?,
            ))
        }

        // =================================================================
        // Conditional select variants (ARM ARM C6.2)
        // =================================================================

        // CSINC Xd, Xn, Xm, cond — conditional select increment
        // ARM ARM: sf | 0 | 0 | 11010100 | Rm | cond | 0 | 1 | Rn | Rd
        // Operands: [Rd, Rn, Rm, Imm(cond)]
        AArch64Opcode::Csinc => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let rm = preg_hw(inst, 2)?;
            let cond = imm_val(inst, 3) as u32 & 0xF;
            Ok((sf << 31)
                | (0b00 << 29)
                | (0b11010100 << 21)
                | (rm << 16)
                | (cond << 12)
                | (0b01 << 10) // op2 = 01 (CSINC)
                | (rn << 5)
                | rd)
        }

        // CSINV Xd, Xn, Xm, cond — conditional select invert
        // ARM ARM: sf | 1 | 0 | 11010100 | Rm | cond | 0 | 0 | Rn | Rd
        // Operands: [Rd, Rn, Rm, Imm(cond)]
        AArch64Opcode::Csinv => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let rm = preg_hw(inst, 2)?;
            let cond = imm_val(inst, 3) as u32 & 0xF;
            Ok((sf << 31)
                | (0b10 << 29) // op = 1, S = 0
                | (0b11010100 << 21)
                | (rm << 16)
                | (cond << 12)
                | (0b00 << 10) // op2 = 00 (CSINV)
                | (rn << 5)
                | rd)
        }

        // CSNEG Xd, Xn, Xm, cond — conditional select negate
        // ARM ARM: sf | 1 | 0 | 11010100 | Rm | cond | 0 | 1 | Rn | Rd
        // Operands: [Rd, Rn, Rm, Imm(cond)]
        AArch64Opcode::Csneg => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0)?;
            let rn = preg_hw(inst, 1)?;
            let rm = preg_hw(inst, 2)?;
            let cond = imm_val(inst, 3) as u32 & 0xF;
            Ok((sf << 31)
                | (0b10 << 29) // op = 1, S = 0
                | (0b11010100 << 21)
                | (rm << 16)
                | (cond << 12)
                | (0b01 << 10) // op2 = 01 (CSNEG)
                | (rn << 5)
                | rd)
        }

        // =================================================================
        // MOVN — Move Wide with NOT
        // ARM ARM: sf | 00 | 100101 | hw | imm16 | Rd
        // opc = 00 for MOVN
        // Operands: [Rd, Imm(imm16)] or [Rd, Imm(imm16), Imm(hw_shift)]
        // =================================================================

        AArch64Opcode::Movn => {
            let sf = sf_from_operand(inst, 0);
            let imm16 = imm_val(inst, 1) as u32 & 0xFFFF;
            // Optional hw (shift) from operand 2 (used by const_materialize)
            let hw = if inst.operands.len() > 2 {
                (imm_val(inst, 2) as u32 / 16) & 0b11
            } else {
                0
            };
            Ok(encoding::encode_move_wide(sf, 0b00, hw, imm16, preg_hw(inst, 0)?))
        }

        // =================================================================
        // FMOV immediate — move FP immediate to FPR
        // ARM ARM C7.2.132: 0 | 0 | 0 | 11110 | ftype(2) | 1 | imm8 | 100 | 00000 | Rd
        // ftype: 00=single(S), 01=double(D), 11=half(H)
        // Operands: [PReg(Rd), FImm(value)]
        // =================================================================

        AArch64Opcode::FmovImm => {
            let rd = preg_hw(inst, 0)?;
            let fp_size = fp_size_from_inst(inst);
            let ftype = match fp_size {
                FpSize::Single => 0b00u32,
                FpSize::Double => 0b01u32,
                FpSize::Half => 0b11u32,
            };
            // Extract the FImm value and encode as 8-bit immediate
            let imm8 = match inst.operands.get(1) {
                Some(MachOperand::FImm(v)) => encode_fmov_imm8(*v),
                Some(MachOperand::Imm(v)) => (*v as u32) & 0xFF,
                _ => 0,
            };
            Ok((0b00011110u32 << 24)
                | (ftype << 22)
                | (1 << 21)
                | (imm8 << 13)
                | (0b100 << 10)
                | (0b00000 << 5)
                | rd)
        }

        // =================================================================
        // LLVM-style typed aliases — delegate to generic encoders
        // =================================================================

        // MOVWrr → MOV Wd, Wn (32-bit); MOVXrr → MOV Xd, Xn (64-bit)
        AArch64Opcode::MOVWrr | AArch64Opcode::MOVXrr => {
            let sf = sf_from_operand(inst, 0);
            let is_sp_source = matches!(
                inst.operands.get(1),
                Some(MachOperand::Special(SpecialReg::SP))
            );
            if is_sp_source {
                Ok(encoding::encode_add_sub_imm(
                    sf, 0, 0, 0, 0, 31, preg_hw(inst, 0)?,
                ))
            } else {
                Ok(encoding::encode_logical_shifted_reg(
                    sf, 0b01, 0, 0, preg_hw(inst, 1)?, 0, 31, preg_hw(inst, 0)?,
                ))
            }
        }

        // STR typed aliases — delegate to StrRI encoding
        AArch64Opcode::STRWui => {
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            let scaled = (offset / 4) as u32 & 0xFFF;
            Ok(encoding::encode_load_store_ui(0b10, 0, 0b00, scaled, preg_hw(inst, 1)?, preg_hw(inst, 0)?))
        }

        AArch64Opcode::STRXui => {
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            let scaled = (offset / 8) as u32 & 0xFFF;
            Ok(encoding::encode_load_store_ui(0b11, 0, 0b00, scaled, preg_hw(inst, 1)?, preg_hw(inst, 0)?))
        }

        AArch64Opcode::STRSui => {
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            let scaled = (offset / 4) as u32 & 0xFFF;
            Ok(encoding::encode_load_store_ui(0b10, 1, 0b00, scaled, preg_hw(inst, 1)?, preg_hw(inst, 0)?))
        }

        AArch64Opcode::STRDui => {
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            let scaled = (offset / 8) as u32 & 0xFFF;
            Ok(encoding::encode_load_store_ui(0b11, 1, 0b00, scaled, preg_hw(inst, 1)?, preg_hw(inst, 0)?))
        }

        // BL (LLVM alias) → Bl encoding
        AArch64Opcode::BL => {
            let offset = imm_val(inst, 0) as u32 & 0x3FFFFFF;
            Ok(encoding::encode_uncond_branch(1, offset))
        }

        // BLR (LLVM alias) → Blr encoding
        AArch64Opcode::BLR => {
            Ok(encoding::encode_branch_reg(0b0001, preg_hw(inst, 0)?))
        }

        // CMP register aliases
        AArch64Opcode::CMPWrr | AArch64Opcode::CMPXrr => {
            let sf = sf_from_operand(inst, 0);
            Ok(encoding::encode_add_sub_shifted_reg(
                sf, 1, 1, 0, preg_hw(inst, 1)?, 0, preg_hw(inst, 0)?, 31,
            ))
        }

        // CMP immediate aliases
        AArch64Opcode::CMPWri | AArch64Opcode::CMPXri => {
            let sf = sf_from_operand(inst, 0);
            let imm = imm_val(inst, 1) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf, 1, 1, 0, imm, preg_hw(inst, 0)?, 31,
            ))
        }

        // MOVZ typed aliases
        AArch64Opcode::MOVZWi | AArch64Opcode::MOVZXi => {
            let sf = sf_from_operand(inst, 0);
            let imm16 = imm_val(inst, 1) as u32 & 0xFFFF;
            Ok(encoding::encode_move_wide(sf, 0b10, 0, imm16, preg_hw(inst, 0)?))
        }

        // Bcc (LLVM alias) → BCond encoding
        AArch64Opcode::Bcc => {
            let cond = imm_val(inst, 0) as u32 & 0xF;
            let offset = if inst.operands.len() > 1 {
                imm_val(inst, 1) as u32 & 0x7FFFF
            } else {
                0
            };
            Ok(encoding::encode_cond_branch(offset, cond))
        }
    }
}

// ---------------------------------------------------------------------------
// FMOV immediate encoding helper
// ---------------------------------------------------------------------------

/// Encode an f64 value into the 8-bit FMOV immediate format.
///
/// ARM ARM C5.6.5: The 8-bit immediate `abcdefgh` encodes:
///   value = (-1)^a * 2^(NOT(b).ccc - 3) * 1.defgh
///
/// where the mantissa `1.defgh` uses 4 fraction bits.
///
/// Only a small subset of FP values are representable. If the value is not
/// exactly representable, returns 0 (which encodes +2.0).
fn encode_fmov_imm8(value: f64) -> u32 {
    let bits = value.to_bits();

    // Extract sign, exponent, mantissa from f64
    let sign = ((bits >> 63) & 1) as u32;
    let exp = ((bits >> 52) & 0x7FF) as i32;
    let frac = bits & 0x000F_FFFF_FFFF_FFFF;

    // Only the top 4 bits of the mantissa can be non-zero
    if frac & 0x0000_FFFF_FFFF_FFFF != 0 {
        return 0;
    }

    let top4 = ((frac >> 48) & 0xF) as u32;

    // Exponent must be in range: biased [1020, 1027] for f64 (bias=1023)
    // This maps to unbiased [-3, +4]
    if exp < 1020 || exp > 1027 {
        return 0;
    }

    // The 3-bit exponent encoding: biased_3 = exp - 1020, range [0, 7]
    let biased_3 = (exp - 1020) as u32;
    // bit 6 = NOT(biased_3[2]), bits 5:4 = biased_3[1:0]
    let not_b = ((biased_3 >> 2) ^ 1) & 1;

    (sign << 7) | (not_b << 6) | ((biased_3 & 0b11) << 4) | top4
}

// ---------------------------------------------------------------------------
// FP size helpers
// ---------------------------------------------------------------------------

/// Determine FP precision from a register operand's class.
fn fp_size_from_preg_class(class: RegClass) -> FpSize {
    match class {
        RegClass::Fpr32 => FpSize::Single,
        RegClass::Fpr64 => FpSize::Double,
        RegClass::Fpr16 => FpSize::Half,
        // Fpr128, Fpr8, or GPR/System: default to Double
        _ => FpSize::Double,
    }
}

/// Determine FP precision from the destination register for FP arithmetic.
/// Uses register class: Fpr32 (S registers) → Single, Fpr64 (D registers) → Double.
/// Defaults to Double if register class is ambiguous (e.g., V/Q registers).
fn fp_size_from_inst(inst: &MachInst) -> FpSize {
    match inst.operands.first() {
        Some(MachOperand::PReg(p)) => fp_size_from_preg_class(preg_class(*p)),
        _ => FpSize::Double,
    }
}

/// FP size for compare instructions (uses first source operand).
fn fp_size_from_cmp_inst(inst: &MachInst) -> FpSize {
    match inst.operands.first() {
        Some(MachOperand::PReg(p)) => fp_size_from_preg_class(preg_class(*p)),
        _ => FpSize::Double,
    }
}

/// FP size derived from a specific source operand index.
fn fp_size_from_source(inst: &MachInst, idx: usize) -> FpSize {
    match inst.operands.get(idx) {
        Some(MachOperand::PReg(p)) => fp_size_from_preg_class(preg_class(*p)),
        _ => FpSize::Double,
    }
}

// ---------------------------------------------------------------------------
// NEON helpers
// ---------------------------------------------------------------------------

/// Arrangement encoding convention for NEON instructions:
///
/// The last `Imm` operand of three-register NEON instructions encodes
/// the vector arrangement as a small integer:
///   0 = 8B, 1 = 16B, 2 = 4H, 3 = 8H, 4 = 2S, 5 = 4S, 6 = 2D
///
/// For instructions with only register operands (logic/NOT), the arrangement
/// is inferred from the register class (V registers = Q=1, default 128-bit).
fn neon_arrangement(inst: &MachInst) -> encoding_neon::VectorArrangement {
    // Check last operand for arrangement encoding
    let arr_idx = inst.operands.len().saturating_sub(1);
    let arr_val = imm_val(inst, arr_idx) as u32;
    match arr_val {
        0 => encoding_neon::VectorArrangement::B8,
        1 => encoding_neon::VectorArrangement::B16,
        2 => encoding_neon::VectorArrangement::H4,
        3 => encoding_neon::VectorArrangement::H8,
        4 => encoding_neon::VectorArrangement::S2,
        5 => encoding_neon::VectorArrangement::S4,
        6 => encoding_neon::VectorArrangement::D2,
        _ => encoding_neon::VectorArrangement::S4, // default: 4S
    }
}

/// FP arrangement for NEON FP instructions.
///
/// Convention: last Imm operand encodes arrangement:
///   0 = 2S, 1 = 4S, 2 = 2D
fn neon_fp_arrangement(inst: &MachInst) -> encoding_neon::FpVectorArrangement {
    let arr_idx = inst.operands.len().saturating_sub(1);
    let arr_val = imm_val(inst, arr_idx) as u32;
    match arr_val {
        0 => encoding_neon::FpVectorArrangement::S2,
        1 => encoding_neon::FpVectorArrangement::S4,
        2 => encoding_neon::FpVectorArrangement::D2,
        _ => encoding_neon::FpVectorArrangement::S4, // default: 4S
    }
}

/// Extract the Q bit for NEON logic/move instructions.
///
/// For logic instructions that don't carry arrangement in their operands,
/// infer from the register class: V registers (Fpr128) = Q=1 (128-bit).
fn neon_q_bit(inst: &MachInst) -> u32 {
    // Check if last operand is an Imm encoding Q directly
    let last_idx = inst.operands.len().saturating_sub(1);
    match inst.operands.get(last_idx) {
        Some(MachOperand::Imm(v)) => {
            // For logic: 0 = 8B(Q=0), 1 = 16B(Q=1)
            if *v == 0 { 0 } else { 1 }
        }
        _ => {
            // Infer from register class: V registers are 128-bit (Q=1)
            match inst.operands.first() {
                Some(MachOperand::PReg(p)) => {
                    if preg_class(*p) == RegClass::Fpr128 { 1 } else { 0 }
                }
                _ => 1, // default to Q=1 (128-bit)
            }
        }
    }
}

/// Extract element size from an Imm operand at `idx`.
///
/// Convention: 1=B, 2=H, 4=S, 8=D
fn neon_element_size(inst: &MachInst, idx: usize) -> encoding_neon::ElementSize {
    let val = imm_val(inst, idx) as u32;
    match val {
        1 => encoding_neon::ElementSize::B,
        2 => encoding_neon::ElementSize::H,
        4 => encoding_neon::ElementSize::S,
        8 => encoding_neon::ElementSize::D,
        _ => encoding_neon::ElementSize::S, // default: 32-bit
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::inst::{AArch64Opcode, MachInst};
    use llvm2_ir::operand::MachOperand;
    use llvm2_ir::regs::{PReg, SpecialReg, X0, X1, X2, X9, X30, V0, V1, V2, W0, W1, W2, W3, W4, W5, S0, S1, S2, D0, H0};

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
    fn test_tbz_bit3() {
        // TBZ X0, #3, +2
        let inst = mk(AArch64Opcode::Tbz, vec![preg(X0), imm(3), imm(2)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (0u32 << 31)
            | (0b011011 << 25)
            | (0 << 24)       // op=0 (TBZ)
            | (3 << 19)       // b40=3
            | (2 << 5)        // imm14=2
            | 0;              // Rt=X0=0
        assert_eq!(enc, expected, "TBZ X0, #3, +2 = {enc:#010X}");
        assert_ne!(enc, NOP, "TBZ must not emit NOP");
    }

    #[test]
    fn test_tbnz_bit3() {
        let inst = mk(AArch64Opcode::Tbnz, vec![preg(X0), imm(3), imm(2)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (0u32 << 31)
            | (0b011011 << 25)
            | (1 << 24)       // op=1 (TBNZ)
            | (3 << 19)
            | (2 << 5)
            | 0;
        assert_eq!(enc, expected, "TBNZ X0, #3, +2 = {enc:#010X}");
        assert_ne!(enc, NOP, "TBNZ must not emit NOP");
    }

    #[test]
    fn test_tbz_high_bit() {
        // TBZ X0, #32, +5  (bit 32: b5=1, b40=0)
        let inst = mk(AArch64Opcode::Tbz, vec![preg(X0), imm(32), imm(5)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31)
            | (0b011011 << 25)
            | (0 << 24)
            | (0 << 19)       // b40 = 32 & 0x1F = 0
            | (5 << 5)
            | 0;
        assert_eq!(enc, expected, "TBZ X0, #32, +5 = {enc:#010X}");
    }

    #[test]
    fn test_tbnz_bit63() {
        // TBNZ X1, #63, +10  (bit 63: b5=1, b40=31)
        let inst = mk(AArch64Opcode::Tbnz, vec![preg(X1), imm(63), imm(10)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31)
            | (0b011011 << 25)
            | (1 << 24)
            | (31 << 19)      // b40 = 63 & 0x1F = 31
            | (10 << 5)
            | 1;              // Rt=X1=1
        assert_eq!(enc, expected, "TBNZ X1, #63, +10 = {enc:#010X}");
    }

    #[test]
    fn test_tbz_known_encoding() {
        // TBZ W0, #0, +0 should encode as 0x36000000
        let inst = mk(AArch64Opcode::Tbz, vec![preg(X0), imm(0), imm(0)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x36000000, "TBZ X0, #0, +0 = {enc:#010X}");
    }

    #[test]
    fn test_tbnz_known_encoding() {
        // TBNZ W0, #0, +0 should encode as 0x37000000
        let inst = mk(AArch64Opcode::Tbnz, vec![preg(X0), imm(0), imm(0)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x37000000, "TBNZ X0, #0, +0 = {enc:#010X}");
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
            (AArch64Opcode::OrnRR, vec![preg(X0), preg(X1), preg(X2)]),
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
            (AArch64Opcode::CSet, vec![preg(X0), imm(1)]), // CSET W0, NE
            (AArch64Opcode::MovI, vec![preg(X0), imm(0)]),
            (AArch64Opcode::Movz, vec![preg(X0), imm(0)]),
            (AArch64Opcode::Movk, vec![preg(X0), imm(0)]),
            (AArch64Opcode::LdrRI, vec![preg(X0), preg(X1)]),
            (AArch64Opcode::StrRI, vec![preg(X0), preg(X1)]),
            (AArch64Opcode::LdrbRI, vec![preg(W0), preg(X1)]),
            (AArch64Opcode::LdrhRI, vec![preg(W0), preg(X1)]),
            (AArch64Opcode::LdrsbRI, vec![preg(W0), preg(X1)]),
            (AArch64Opcode::LdrshRI, vec![preg(W0), preg(X1)]),
            (AArch64Opcode::StrbRI, vec![preg(W0), preg(X1)]),
            (AArch64Opcode::StrhRI, vec![preg(W0), preg(X1)]),
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
            (AArch64Opcode::Uxtb, vec![preg(W0), preg(W1)]),
            (AArch64Opcode::Uxth, vec![preg(W0), preg(W1)]),
            (AArch64Opcode::Ubfm, vec![preg(X0), preg(X1), imm(0), imm(7)]),
            (AArch64Opcode::Sbfm, vec![preg(X0), preg(X1), imm(0), imm(7)]),
            (AArch64Opcode::Bfm, vec![preg(X0), preg(X1), imm(0), imm(7)]),
            (AArch64Opcode::LdrRO, vec![preg(X0), preg(X1), preg(X2)]),
            (AArch64Opcode::StrRO, vec![preg(X0), preg(X1), preg(X2)]),
            (AArch64Opcode::LdrGot, vec![preg(X0), preg(X1)]),
            (AArch64Opcode::LdrTlvp, vec![preg(X0), preg(X1)]),
            (AArch64Opcode::FaddRR, vec![preg(V0), preg(V1), preg(V2)]),
            (AArch64Opcode::FsubRR, vec![preg(V0), preg(V1), preg(V2)]),
            (AArch64Opcode::FmulRR, vec![preg(V0), preg(V1), preg(V2)]),
            (AArch64Opcode::FdivRR, vec![preg(V0), preg(V1), preg(V2)]),
            (AArch64Opcode::FnegRR, vec![preg(V0), preg(V1)]),
            (AArch64Opcode::FabsRR, vec![preg(V0), preg(V1)]),
            (AArch64Opcode::FsqrtRR, vec![preg(V0), preg(V1)]),
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

    // =========================================================================
    // 32-bit encoding tests — verify sf=0 for W-register operands
    // =========================================================================

    #[test]
    fn test_sf_from_operand_w_register() {
        // W0 (encoding 32) should produce sf=0
        let inst = mk(AArch64Opcode::AddRR, vec![preg(W0), preg(W1), preg(W2)]);
        assert_eq!(sf_from_operand(&inst, 0), 0, "W0 should produce sf=0");

        // X0 (encoding 0) should produce sf=1
        let inst = mk(AArch64Opcode::AddRR, vec![preg(X0), preg(X1), preg(X2)]);
        assert_eq!(sf_from_operand(&inst, 0), 1, "X0 should produce sf=1");
    }

    #[test]
    fn test_add_rr_32bit() {
        // ADD W0, W1, W2 — must have sf=0 (bit 31 = 0)
        let inst = mk(AArch64Opcode::AddRR, vec![preg(W0), preg(W1), preg(W2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_add_sub_shifted_reg(0, 0, 0, 0, 2, 0, 1, 0);
        assert_eq!(enc, direct, "ADD W0, W1, W2: unified={enc:#010X}, direct={direct:#010X}");
        // Verify bit 31 (sf) is 0
        assert_eq!(enc >> 31, 0, "ADD W0, W1, W2 must have sf=0 (bit 31 = 0)");
    }

    #[test]
    fn test_add_rr_64bit_has_sf_1() {
        // ADD X0, X1, X2 — must have sf=1 (bit 31 = 1)
        let inst = mk(AArch64Opcode::AddRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc >> 31, 1, "ADD X0, X1, X2 must have sf=1 (bit 31 = 1)");
    }

    #[test]
    fn test_add_ri_32bit() {
        // ADD W0, W1, #42 — sf=0
        let inst = mk(AArch64Opcode::AddRI, vec![preg(W0), preg(W1), imm(42)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_add_sub_imm(0, 0, 0, 0, 42, 1, 0);
        assert_eq!(enc, direct, "ADD W0, W1, #42");
        assert_eq!(enc >> 31, 0, "ADD W0, W1, #42 must have sf=0");
    }

    #[test]
    fn test_sub_rr_32bit() {
        // SUB W0, W1, W2 — sf=0
        let inst = mk(AArch64Opcode::SubRR, vec![preg(W0), preg(W1), preg(W2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_add_sub_shifted_reg(0, 1, 0, 0, 2, 0, 1, 0);
        assert_eq!(enc, direct, "SUB W0, W1, W2");
        assert_eq!(enc >> 31, 0, "SUB W0, W1, W2 must have sf=0");
    }

    #[test]
    fn test_sub_ri_32bit() {
        // SUB W0, W1, #10 — sf=0
        let inst = mk(AArch64Opcode::SubRI, vec![preg(W0), preg(W1), imm(10)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_add_sub_imm(0, 1, 0, 0, 10, 1, 0);
        assert_eq!(enc, direct, "SUB W0, W1, #10");
        assert_eq!(enc >> 31, 0, "SUB W0, W1, #10 must have sf=0");
    }

    #[test]
    fn test_and_rr_32bit() {
        // AND W0, W1, W2 — sf=0
        let inst = mk(AArch64Opcode::AndRR, vec![preg(W0), preg(W1), preg(W2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_logical_shifted_reg(0, 0b00, 0, 0, 2, 0, 1, 0);
        assert_eq!(enc, direct, "AND W0, W1, W2");
        assert_eq!(enc >> 31, 0, "AND W0, W1, W2 must have sf=0");
    }

    #[test]
    fn test_orr_rr_32bit() {
        // ORR W0, W1, W2 — sf=0
        let inst = mk(AArch64Opcode::OrrRR, vec![preg(W0), preg(W1), preg(W2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_logical_shifted_reg(0, 0b01, 0, 0, 2, 0, 1, 0);
        assert_eq!(enc, direct, "ORR W0, W1, W2");
        assert_eq!(enc >> 31, 0, "ORR W0, W1, W2 must have sf=0");
    }

    #[test]
    fn test_eor_rr_32bit() {
        // EOR W0, W1, W2 — sf=0
        let inst = mk(AArch64Opcode::EorRR, vec![preg(W0), preg(W1), preg(W2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_logical_shifted_reg(0, 0b10, 0, 0, 2, 0, 1, 0);
        assert_eq!(enc, direct, "EOR W0, W1, W2");
        assert_eq!(enc >> 31, 0, "EOR W0, W1, W2 must have sf=0");
    }

    #[test]
    fn test_mul_rr_32bit() {
        // MUL W0, W1, W2 = MADD W0, W1, W2, WZR — sf=0
        let inst = mk(AArch64Opcode::MulRR, vec![preg(W0), preg(W1), preg(W2)]);
        let enc = encode_instruction(&inst).unwrap();
        // Expected: sf=0, 00 11011 000 Rm=2 0 Ra=31 Rn=1 Rd=0
        let expected = (0u32 << 31) | (0b11011 << 24) | (2 << 16) | (31 << 10) | (1 << 5) | 0;
        assert_eq!(enc, expected, "MUL W0, W1, W2 = {enc:#010X}");
        assert_eq!(enc >> 31, 0, "MUL W0, W1, W2 must have sf=0");
    }

    #[test]
    fn test_sdiv_32bit() {
        // SDIV W0, W1, W2 — sf=0
        let inst = mk(AArch64Opcode::SDiv, vec![preg(W0), preg(W1), preg(W2)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (0u32 << 31) | (0b0_0011010110 << 21) | (2 << 16) | (0b000011 << 10) | (1 << 5) | 0;
        assert_eq!(enc, expected, "SDIV W0, W1, W2 = {enc:#010X}");
        assert_eq!(enc >> 31, 0, "SDIV W0, W1, W2 must have sf=0");
    }

    #[test]
    fn test_neg_32bit() {
        // NEG W0, W1 = SUB W0, WZR, W1 — sf=0
        let inst = mk(AArch64Opcode::Neg, vec![preg(W0), preg(W1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_add_sub_shifted_reg(0, 1, 0, 0, 1, 0, 31, 0);
        assert_eq!(enc, direct, "NEG W0, W1");
        assert_eq!(enc >> 31, 0, "NEG W0, W1 must have sf=0");
    }

    #[test]
    fn test_mov_r_32bit() {
        // MOV W0, W1 = ORR W0, WZR, W1 — sf=0
        let inst = mk(AArch64Opcode::MovR, vec![preg(W0), preg(W1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_logical_shifted_reg(0, 0b01, 0, 0, 1, 0, 31, 0);
        assert_eq!(enc, direct, "MOV W0, W1");
        assert_eq!(enc >> 31, 0, "MOV W0, W1 must have sf=0");
    }

    #[test]
    fn test_movz_32bit() {
        // MOVZ W0, #0x1234 — sf=0
        let inst = mk(AArch64Opcode::Movz, vec![preg(W0), imm(0x1234)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_move_wide(0, 0b10, 0, 0x1234, 0);
        assert_eq!(enc, direct, "MOVZ W0, #0x1234");
        assert_eq!(enc >> 31, 0, "MOVZ W0, #0x1234 must have sf=0");
    }

    #[test]
    fn test_cmp_rr_32bit() {
        // CMP W0, W1 = SUBS WZR, W0, W1 — sf=0
        let inst = mk(AArch64Opcode::CmpRR, vec![preg(W0), preg(W1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_add_sub_shifted_reg(0, 1, 1, 0, 1, 0, 0, 31);
        assert_eq!(enc, direct, "CMP W0, W1");
        assert_eq!(enc >> 31, 0, "CMP W0, W1 must have sf=0");
    }

    #[test]
    fn test_cmp_ri_32bit() {
        // CMP W0, #42 = SUBS WZR, W0, #42 — sf=0
        let inst = mk(AArch64Opcode::CmpRI, vec![preg(W0), imm(42)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_add_sub_imm(0, 1, 1, 0, 42, 0, 31);
        assert_eq!(enc, direct, "CMP W0, #42");
        assert_eq!(enc >> 31, 0, "CMP W0, #42 must have sf=0");
    }

    #[test]
    fn test_tst_32bit() {
        // TST W0, W1 = ANDS WZR, W0, W1 — sf=0
        let inst = mk(AArch64Opcode::Tst, vec![preg(W0), preg(W1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_logical_shifted_reg(0, 0b11, 0, 0, 1, 0, 0, 31);
        assert_eq!(enc, direct, "TST W0, W1");
        assert_eq!(enc >> 31, 0, "TST W0, W1 must have sf=0");
    }

    #[test]
    fn test_adds_rr_32bit() {
        // ADDS W0, W1, W2 — sf=0
        let inst = mk(AArch64Opcode::AddsRR, vec![preg(W0), preg(W1), preg(W2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_add_sub_shifted_reg(0, 0, 1, 0, 2, 0, 1, 0);
        assert_eq!(enc, direct, "ADDS W0, W1, W2");
        assert_eq!(enc >> 31, 0, "ADDS W0, W1, W2 must have sf=0");
    }

    #[test]
    fn test_subs_rr_32bit() {
        // SUBS W0, W1, W2 — sf=0
        let inst = mk(AArch64Opcode::SubsRR, vec![preg(W0), preg(W1), preg(W2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_add_sub_shifted_reg(0, 1, 1, 0, 2, 0, 1, 0);
        assert_eq!(enc, direct, "SUBS W0, W1, W2");
        assert_eq!(enc >> 31, 0, "SUBS W0, W1, W2 must have sf=0");
    }

    #[test]
    fn test_lsl_rr_32bit() {
        // LSLV W0, W1, W2 — sf=0
        let inst = mk(AArch64Opcode::LslRR, vec![preg(W0), preg(W1), preg(W2)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (0u32 << 31) | (0b0_0011010110 << 21) | (2 << 16)
            | (0b001000 << 10) | (1 << 5) | 0;
        assert_eq!(enc, expected, "LSLV W0, W1, W2 = {enc:#010X}");
        assert_eq!(enc >> 31, 0, "LSLV W0, W1, W2 must have sf=0");
    }

    #[test]
    fn test_cbz_32bit() {
        // CBZ W0, <+8> — sf=0
        let inst = mk(AArch64Opcode::Cbz, vec![preg(W0), imm(2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_cmp_branch(0, 0, 2, 0);
        assert_eq!(enc, direct, "CBZ W0, <+8>");
        assert_eq!(enc >> 31, 0, "CBZ W0, <+8> must have sf=0");
    }

    #[test]
    fn test_cbnz_32bit() {
        // CBNZ W0, <+8> — sf=0
        let inst = mk(AArch64Opcode::Cbnz, vec![preg(W0), imm(2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_cmp_branch(0, 1, 2, 0);
        assert_eq!(enc, direct, "CBNZ W0, <+8>");
        assert_eq!(enc >> 31, 0, "CBNZ W0, <+8> must have sf=0");
    }

    #[test]
    fn test_lsl_ri_32bit() {
        // LSL W0, W1, #3 — sf=0, regsize=32
        // UBFM W0, W1, #(-3 MOD 32)=29, #(32-1-3)=28
        let inst = mk(AArch64Opcode::LslRI, vec![preg(W0), preg(W1), imm(3)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc >> 31, 0, "LSL W0, W1, #3 must have sf=0");
        // N bit (bit 22) must be 0 for 32-bit
        assert_eq!((enc >> 22) & 1, 0, "LSL W0 must have N=0 for 32-bit");
    }

    #[test]
    fn test_fp_size_from_s_register() {
        // FADD S0, S1, S2 should use Single precision
        let inst = mk(AArch64Opcode::FaddRR, vec![preg(S0), preg(S1), preg(S2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_fp::encode_fp_arith(FpSize::Single, FpArithOp::Add, 2, 1, 0).unwrap();
        assert_eq!(enc, direct, "FADD S0, S1, S2 should use Single precision");
    }

    // =========================================================================
    // Byte/halfword load/store encoding tests
    // =========================================================================

    #[test]
    fn test_ldrb_ri() {
        // LDRB W0, [X1, #0] — size=00, V=0, opc=01
        let inst = mk(AArch64Opcode::LdrbRI, vec![preg(W0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_load_store_ui(0b00, 0, 0b01, 0, 1, 0);
        assert_eq!(enc, direct, "LDRB W0, [X1] = {enc:#010X}");
        // Verify size field (bits 31:30) = 00
        assert_eq!((enc >> 30) & 0b11, 0b00, "LDRB must have size=00");
        // Verify opc field (bits 23:22) = 01 (load)
        assert_eq!((enc >> 22) & 0b11, 0b01, "LDRB must have opc=01");
    }

    #[test]
    fn test_ldrb_ri_with_offset() {
        // LDRB W0, [X1, #5] — byte-aligned offset, no scaling needed
        let inst = mk(AArch64Opcode::LdrbRI, vec![preg(W0), preg(X1), imm(5)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_load_store_ui(0b00, 0, 0b01, 5, 1, 0);
        assert_eq!(enc, direct, "LDRB W0, [X1, #5] = {enc:#010X}");
    }

    #[test]
    fn test_ldrh_ri() {
        // LDRH W0, [X1, #0] — size=01, V=0, opc=01
        let inst = mk(AArch64Opcode::LdrhRI, vec![preg(W0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_load_store_ui(0b01, 0, 0b01, 0, 1, 0);
        assert_eq!(enc, direct, "LDRH W0, [X1] = {enc:#010X}");
        assert_eq!((enc >> 30) & 0b11, 0b01, "LDRH must have size=01");
        assert_eq!((enc >> 22) & 0b11, 0b01, "LDRH must have opc=01");
    }

    #[test]
    fn test_ldrh_ri_with_offset() {
        // LDRH W0, [X1, #4] — halfword offset 4 / 2 = 2 (scaled)
        let inst = mk(AArch64Opcode::LdrhRI, vec![preg(W0), preg(X1), imm(4)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_load_store_ui(0b01, 0, 0b01, 2, 1, 0);
        assert_eq!(enc, direct, "LDRH W0, [X1, #4] = {enc:#010X}");
    }

    #[test]
    fn test_ldrsb_ri() {
        // LDRSB W0, [X1, #0] — size=00, V=0, opc=11 (sign-extend to 32-bit)
        let inst = mk(AArch64Opcode::LdrsbRI, vec![preg(W0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_load_store_ui(0b00, 0, 0b11, 0, 1, 0);
        assert_eq!(enc, direct, "LDRSB W0, [X1] = {enc:#010X}");
        assert_eq!((enc >> 30) & 0b11, 0b00, "LDRSB must have size=00");
        assert_eq!((enc >> 22) & 0b11, 0b11, "LDRSB to W must have opc=11");
    }

    #[test]
    fn test_ldrsh_ri() {
        // LDRSH W0, [X1, #0] — size=01, V=0, opc=11 (sign-extend to 32-bit)
        let inst = mk(AArch64Opcode::LdrshRI, vec![preg(W0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_load_store_ui(0b01, 0, 0b11, 0, 1, 0);
        assert_eq!(enc, direct, "LDRSH W0, [X1] = {enc:#010X}");
        assert_eq!((enc >> 30) & 0b11, 0b01, "LDRSH must have size=01");
        assert_eq!((enc >> 22) & 0b11, 0b11, "LDRSH to W must have opc=11");
    }

    #[test]
    fn test_strb_ri() {
        // STRB W0, [X1, #0] — size=00, V=0, opc=00
        let inst = mk(AArch64Opcode::StrbRI, vec![preg(W0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_load_store_ui(0b00, 0, 0b00, 0, 1, 0);
        assert_eq!(enc, direct, "STRB W0, [X1] = {enc:#010X}");
        assert_eq!((enc >> 30) & 0b11, 0b00, "STRB must have size=00");
        assert_eq!((enc >> 22) & 0b11, 0b00, "STRB must have opc=00");
    }

    #[test]
    fn test_strb_ri_with_offset() {
        // STRB W0, [X1, #3] — byte-aligned offset
        let inst = mk(AArch64Opcode::StrbRI, vec![preg(W0), preg(X1), imm(3)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_load_store_ui(0b00, 0, 0b00, 3, 1, 0);
        assert_eq!(enc, direct, "STRB W0, [X1, #3] = {enc:#010X}");
    }

    #[test]
    fn test_strh_ri() {
        // STRH W0, [X1, #0] — size=01, V=0, opc=00
        let inst = mk(AArch64Opcode::StrhRI, vec![preg(W0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_load_store_ui(0b01, 0, 0b00, 0, 1, 0);
        assert_eq!(enc, direct, "STRH W0, [X1] = {enc:#010X}");
        assert_eq!((enc >> 30) & 0b11, 0b01, "STRH must have size=01");
        assert_eq!((enc >> 22) & 0b11, 0b00, "STRH must have opc=00");
    }

    #[test]
    fn test_strh_ri_with_offset() {
        // STRH W0, [X1, #6] — halfword offset 6 / 2 = 3 (scaled)
        let inst = mk(AArch64Opcode::StrhRI, vec![preg(W0), preg(X1), imm(6)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_load_store_ui(0b01, 0, 0b00, 3, 1, 0);
        assert_eq!(enc, direct, "STRH W0, [X1, #6] = {enc:#010X}");
    }

    // =========================================================================
    // Immediate shift encoding tests — verified against system assembler
    // (xcrun as -arch arm64) ground truth. Fixes #134.
    //
    // LslRI/LsrRI/AsrRI are encoded via UBFM/SBFM bitfield instructions:
    //   LSL Rd, Rn, #shift  = UBFM Rd, Rn, #(-shift MOD regsize), #(regsize-1-shift)
    //   LSR Rd, Rn, #shift  = UBFM Rd, Rn, #shift, #(regsize-1)
    //   ASR Rd, Rn, #shift  = SBFM Rd, Rn, #shift, #(regsize-1)
    // =========================================================================

    #[test]
    fn test_lsl_ri_64bit_ground_truth() {
        // LSL X0, X1, #2 — verified: 0xD37EF420 (xcrun as)
        // = UBFM X0, X1, #62, #61
        let inst = mk(AArch64Opcode::LslRI, vec![preg(X0), preg(X1), imm(2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xD37EF420, "LSL X0, X1, #2 = {enc:#010X}");
        // Must NOT be NOP
        assert_ne!(enc, NOP, "LSL X0, X1, #2 must not emit NOP");
    }

    #[test]
    fn test_lsr_ri_64bit_ground_truth() {
        // LSR X0, X1, #2 — verified: 0xD342FC20 (xcrun as)
        // = UBFM X0, X1, #2, #63
        let inst = mk(AArch64Opcode::LsrRI, vec![preg(X0), preg(X1), imm(2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xD342FC20, "LSR X0, X1, #2 = {enc:#010X}");
        assert_ne!(enc, NOP, "LSR X0, X1, #2 must not emit NOP");
    }

    #[test]
    fn test_asr_ri_64bit_ground_truth() {
        // ASR X0, X1, #2 — verified: 0x9342FC20 (xcrun as)
        // = SBFM X0, X1, #2, #63
        let inst = mk(AArch64Opcode::AsrRI, vec![preg(X0), preg(X1), imm(2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x9342FC20, "ASR X0, X1, #2 = {enc:#010X}");
        assert_ne!(enc, NOP, "ASR X0, X1, #2 must not emit NOP");
    }

    #[test]
    fn test_lsl_ri_32bit_ground_truth() {
        // LSL W0, W1, #3 — verified: 0x531D7020 (xcrun as)
        // = UBFM W0, W1, #29, #28
        let inst = mk(AArch64Opcode::LslRI, vec![preg(W0), preg(W1), imm(3)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x531D7020, "LSL W0, W1, #3 = {enc:#010X}");
        assert_ne!(enc, NOP, "LSL W0, W1, #3 must not emit NOP");
        assert_eq!(enc >> 31, 0, "LSL W0 must have sf=0");
        assert_eq!((enc >> 22) & 1, 0, "LSL W0 must have N=0 for 32-bit");
    }

    #[test]
    fn test_lsr_ri_32bit_ground_truth() {
        // LSR W0, W1, #3 — verified: 0x53037C20 (xcrun as)
        // = UBFM W0, W1, #3, #31
        let inst = mk(AArch64Opcode::LsrRI, vec![preg(W0), preg(W1), imm(3)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x53037C20, "LSR W0, W1, #3 = {enc:#010X}");
        assert_ne!(enc, NOP, "LSR W0, W1, #3 must not emit NOP");
        assert_eq!(enc >> 31, 0, "LSR W0 must have sf=0");
    }

    #[test]
    fn test_asr_ri_32bit_ground_truth() {
        // ASR W0, W1, #3 — verified: 0x13037C20 (xcrun as)
        // = SBFM W0, W1, #3, #31
        let inst = mk(AArch64Opcode::AsrRI, vec![preg(W0), preg(W1), imm(3)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x13037C20, "ASR W0, W1, #3 = {enc:#010X}");
        assert_ne!(enc, NOP, "ASR W0, W1, #3 must not emit NOP");
        assert_eq!(enc >> 31, 0, "ASR W0 must have sf=0");
    }

    #[test]
    fn test_lsl_ri_ubfm_field_decomposition_64bit() {
        // Verify UBFM field placement for LSL X0, X1, #2
        let inst = mk(AArch64Opcode::LslRI, vec![preg(X0), preg(X1), imm(2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!((enc >> 31) & 1, 1, "sf=1 for 64-bit");
        assert_eq!((enc >> 29) & 0b11, 0b10, "opc=10 for UBFM");
        assert_eq!((enc >> 23) & 0b111111, 0b100110, "fixed bits");
        assert_eq!((enc >> 22) & 1, 1, "N=1 for 64-bit");
        assert_eq!((enc >> 16) & 0x3F, 62, "immr=(-2 MOD 64)=62");
        assert_eq!((enc >> 10) & 0x3F, 61, "imms=(63-2)=61");
        assert_eq!((enc >> 5) & 0x1F, 1, "Rn=X1");
        assert_eq!(enc & 0x1F, 0, "Rd=X0");
    }

    #[test]
    fn test_lsr_ri_ubfm_field_decomposition_64bit() {
        // Verify UBFM field placement for LSR X0, X1, #2
        let inst = mk(AArch64Opcode::LsrRI, vec![preg(X0), preg(X1), imm(2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!((enc >> 31) & 1, 1, "sf=1 for 64-bit");
        assert_eq!((enc >> 29) & 0b11, 0b10, "opc=10 for UBFM");
        assert_eq!((enc >> 22) & 1, 1, "N=1 for 64-bit");
        assert_eq!((enc >> 16) & 0x3F, 2, "immr=shift=2");
        assert_eq!((enc >> 10) & 0x3F, 63, "imms=63 for 64-bit");
        assert_eq!((enc >> 5) & 0x1F, 1, "Rn=X1");
        assert_eq!(enc & 0x1F, 0, "Rd=X0");
    }

    #[test]
    fn test_asr_ri_sbfm_field_decomposition_64bit() {
        // Verify SBFM field placement for ASR X0, X1, #2
        let inst = mk(AArch64Opcode::AsrRI, vec![preg(X0), preg(X1), imm(2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!((enc >> 31) & 1, 1, "sf=1 for 64-bit");
        assert_eq!((enc >> 29) & 0b11, 0b00, "opc=00 for SBFM");
        assert_eq!((enc >> 22) & 1, 1, "N=1 for 64-bit");
        assert_eq!((enc >> 16) & 0x3F, 2, "immr=shift=2");
        assert_eq!((enc >> 10) & 0x3F, 63, "imms=63 for 64-bit");
        assert_eq!((enc >> 5) & 0x1F, 1, "Rn=X1");
        assert_eq!(enc & 0x1F, 0, "Rd=X0");
    }

    #[test]
    fn test_shift_ri_boundary_values() {
        // LSL X0, X1, #0 (no shift) — should still encode as UBFM, not NOP
        let inst = mk(AArch64Opcode::LslRI, vec![preg(X0), preg(X1), imm(0)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_ne!(enc, NOP, "LSL X0, X1, #0 must not emit NOP");

        // LSL X0, X1, #63 (max shift for 64-bit)
        let inst = mk(AArch64Opcode::LslRI, vec![preg(X0), preg(X1), imm(63)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_ne!(enc, NOP, "LSL X0, X1, #63 must not emit NOP");
        assert_eq!((enc >> 16) & 0x3F, 1, "immr=(-63 MOD 64)=1");
        assert_eq!((enc >> 10) & 0x3F, 0, "imms=(63-63)=0");

        // LSR X0, X1, #63 (max shift)
        let inst = mk(AArch64Opcode::LsrRI, vec![preg(X0), preg(X1), imm(63)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_ne!(enc, NOP, "LSR X0, X1, #63 must not emit NOP");
        assert_eq!((enc >> 16) & 0x3F, 63, "immr=63");
        assert_eq!((enc >> 10) & 0x3F, 63, "imms=63");
    }

    // =========================================================================
    // Bitfield move instruction tests (UBFM, SBFM, BFM) — issue #137
    // =========================================================================

    #[test]
    fn test_ubfm_64bit() {
        // UBFM X0, X1, #0, #7 — extract bits [7:0] (alias: UXTB X0, X1)
        // ARM ARM: sf=1 opc=10 100110 N=1 immr=0 imms=7 Rn=1 Rd=0
        let inst = mk(AArch64Opcode::Ubfm, vec![preg(X0), preg(X1), imm(0), imm(7)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31)
            | (0b10 << 29)
            | (0b100110 << 23)
            | (1 << 22) // N=1
            | (0 << 16) // immr=0
            | (7 << 10) // imms=7
            | (1 << 5)
            | 0;
        assert_eq!(enc, expected, "UBFM X0, X1, #0, #7 = {enc:#010X}");
        // Verify fixed field: bits [28:23] = 100110
        assert_eq!((enc >> 23) & 0x3F, 0b100110);
    }

    #[test]
    fn test_ubfm_32bit() {
        // UBFM W0, W1, #0, #7 — sf=0, N=0
        let inst = mk(AArch64Opcode::Ubfm, vec![preg(W0), preg(W1), imm(0), imm(7)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc >> 31, 0, "UBFM W0 must have sf=0");
        assert_eq!((enc >> 22) & 1, 0, "UBFM W0 must have N=0 for 32-bit");
        assert_eq!((enc >> 29) & 0b11, 0b10, "UBFM opc must be 10");
    }

    #[test]
    fn test_ubfm_lsr_alias() {
        // LSR X0, X1, #3 is encoded as UBFM X0, X1, #3, #63
        // Verify that UBFM with immr=3, imms=63 matches the LSR encoding.
        let ubfm = mk(AArch64Opcode::Ubfm, vec![preg(X0), preg(X1), imm(3), imm(63)]);
        let lsr = mk(AArch64Opcode::LsrRI, vec![preg(X0), preg(X1), imm(3)]);
        let ubfm_enc = encode_instruction(&ubfm).unwrap();
        let lsr_enc = encode_instruction(&lsr).unwrap();
        assert_eq!(ubfm_enc, lsr_enc, "UBFM X0, X1, #3, #63 must match LSR X0, X1, #3");
    }

    #[test]
    fn test_sbfm_64bit() {
        // SBFM X0, X1, #0, #31 — sign-extend bits [31:0] (alias: SXTW X0, X1)
        // ARM ARM: sf=1 opc=00 100110 N=1 immr=0 imms=31 Rn=1 Rd=0
        let inst = mk(AArch64Opcode::Sbfm, vec![preg(X0), preg(X1), imm(0), imm(31)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31)
            | (0b00 << 29)
            | (0b100110 << 23)
            | (1 << 22) // N=1
            | (0 << 16) // immr=0
            | (31 << 10) // imms=31
            | (1 << 5)
            | 0;
        assert_eq!(enc, expected, "SBFM X0, X1, #0, #31 = {enc:#010X}");
    }

    #[test]
    fn test_sbfm_sxtw_matches() {
        // SBFM X0, X1, #0, #31 must match the existing SXTW encoding
        let sbfm = mk(AArch64Opcode::Sbfm, vec![preg(X0), preg(X1), imm(0), imm(31)]);
        let sxtw = mk(AArch64Opcode::Sxtw, vec![preg(X0), preg(X1)]);
        let sbfm_enc = encode_instruction(&sbfm).unwrap();
        let sxtw_enc = encode_instruction(&sxtw).unwrap();
        assert_eq!(sbfm_enc, sxtw_enc, "SBFM X0, X1, #0, #31 must match SXTW X0, X1");
    }

    #[test]
    fn test_sbfm_asr_alias() {
        // ASR X0, X1, #3 is encoded as SBFM X0, X1, #3, #63
        let sbfm = mk(AArch64Opcode::Sbfm, vec![preg(X0), preg(X1), imm(3), imm(63)]);
        let asr = mk(AArch64Opcode::AsrRI, vec![preg(X0), preg(X1), imm(3)]);
        let sbfm_enc = encode_instruction(&sbfm).unwrap();
        let asr_enc = encode_instruction(&asr).unwrap();
        assert_eq!(sbfm_enc, asr_enc, "SBFM X0, X1, #3, #63 must match ASR X0, X1, #3");
    }

    #[test]
    fn test_sbfm_32bit() {
        // SBFM W0, W1, #0, #7 — sf=0, N=0, opc=00
        let inst = mk(AArch64Opcode::Sbfm, vec![preg(W0), preg(W1), imm(0), imm(7)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc >> 31, 0, "SBFM W0 must have sf=0");
        assert_eq!((enc >> 22) & 1, 0, "SBFM W0 must have N=0 for 32-bit");
        assert_eq!((enc >> 29) & 0b11, 0b00, "SBFM opc must be 00");
    }

    #[test]
    fn test_bfm_64bit() {
        // BFM X0, X1, #4, #11 — bitfield insert bits [11:4] of Rn into Rd
        // ARM ARM: sf=1 opc=01 100110 N=1 immr=4 imms=11 Rn=1 Rd=0
        let inst = mk(AArch64Opcode::Bfm, vec![preg(X0), preg(X1), imm(4), imm(11)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31)
            | (0b01 << 29)
            | (0b100110 << 23)
            | (1 << 22) // N=1
            | (4 << 16) // immr=4
            | (11 << 10) // imms=11
            | (1 << 5)
            | 0;
        assert_eq!(enc, expected, "BFM X0, X1, #4, #11 = {enc:#010X}");
        assert_eq!((enc >> 29) & 0b11, 0b01, "BFM opc must be 01");
    }

    #[test]
    fn test_bfm_32bit() {
        // BFM W0, W1, #4, #11 — sf=0, N=0, opc=01
        let inst = mk(AArch64Opcode::Bfm, vec![preg(W0), preg(W1), imm(4), imm(11)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc >> 31, 0, "BFM W0 must have sf=0");
        assert_eq!((enc >> 22) & 1, 0, "BFM W0 must have N=0 for 32-bit");
        assert_eq!((enc >> 29) & 0b11, 0b01, "BFM opc must be 01");
    }

    #[test]
    fn test_bitfield_opc_encoding() {
        // Verify the opc field distinguishes UBFM/SBFM/BFM correctly
        let ubfm = mk(AArch64Opcode::Ubfm, vec![preg(X0), preg(X1), imm(0), imm(0)]);
        let sbfm = mk(AArch64Opcode::Sbfm, vec![preg(X0), preg(X1), imm(0), imm(0)]);
        let bfm = mk(AArch64Opcode::Bfm, vec![preg(X0), preg(X1), imm(0), imm(0)]);
        let ubfm_enc = encode_instruction(&ubfm).unwrap();
        let sbfm_enc = encode_instruction(&sbfm).unwrap();
        let bfm_enc = encode_instruction(&bfm).unwrap();
        assert_eq!((ubfm_enc >> 29) & 0b11, 0b10, "UBFM opc=10");
        assert_eq!((sbfm_enc >> 29) & 0b11, 0b00, "SBFM opc=00");
        assert_eq!((bfm_enc >> 29) & 0b11, 0b01, "BFM opc=01");
        // All three share the same fixed field at bits [28:23]
        assert_eq!((ubfm_enc >> 23) & 0x3F, 0b100110);
        assert_eq!((sbfm_enc >> 23) & 0x3F, 0b100110);
        assert_eq!((bfm_enc >> 23) & 0x3F, 0b100110);
    }

    // =========================================================================
    // Register-offset load/store tests (LdrRO, StrRO) — issue #137
    // =========================================================================

    #[test]
    fn test_ldr_ro_64bit() {
        // LDR X0, [X1, X2] — 64-bit register-offset load, default LSL, S=0
        // size=11, V=0, opc=01, Rm=2, option=011(LSL), S=0, Rn=1, Rt=0
        let inst = mk(AArch64Opcode::LdrRO, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_mem::encode_ldr_str_register(
            encoding_mem::LoadStoreSize::Double, false, encoding_mem::LoadStoreOp::Load,
            2, encoding_mem::RegExtend::Lsl, false, 1, 0,
        ).unwrap();
        assert_eq!(enc, direct, "LDR X0, [X1, X2] = {enc:#010X}");
    }

    #[test]
    fn test_ldr_ro_32bit() {
        // LDR W0, [X1, X2] — 32-bit register-offset load
        // size=10, V=0, opc=01, Rm=2, option=011(LSL), S=0, Rn=1, Rt=0
        let inst = mk(AArch64Opcode::LdrRO, vec![preg(W0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_mem::encode_ldr_str_register(
            encoding_mem::LoadStoreSize::Word, false, encoding_mem::LoadStoreOp::Load,
            2, encoding_mem::RegExtend::Lsl, false, 1, 0,
        ).unwrap();
        assert_eq!(enc, direct, "LDR W0, [X1, X2] = {enc:#010X}");
        // Verify size field (bits 31:30)
        assert_eq!((enc >> 30) & 0b11, 0b10, "LDR W must have size=10");
    }

    #[test]
    fn test_ldr_ro_with_shift() {
        // LDR X0, [X1, X2, LSL #3] — option=011, S=1
        // 4th operand packs extend info: (option << 1) | S = (0b011 << 1) | 1 = 7
        let inst = mk(AArch64Opcode::LdrRO, vec![preg(X0), preg(X1), preg(X2), imm(7)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_mem::encode_ldr_str_register(
            encoding_mem::LoadStoreSize::Double, false, encoding_mem::LoadStoreOp::Load,
            2, encoding_mem::RegExtend::Lsl, true, 1, 0,
        ).unwrap();
        assert_eq!(enc, direct, "LDR X0, [X1, X2, LSL #3] = {enc:#010X}");
    }

    #[test]
    fn test_ldr_ro_sxtw() {
        // LDR X0, [X1, W2, SXTW] — option=110, S=0
        // packed = (0b110 << 1) | 0 = 12
        let inst = mk(AArch64Opcode::LdrRO, vec![preg(X0), preg(X1), preg(X2), imm(12)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_mem::encode_ldr_str_register(
            encoding_mem::LoadStoreSize::Double, false, encoding_mem::LoadStoreOp::Load,
            2, encoding_mem::RegExtend::Sxtw, false, 1, 0,
        ).unwrap();
        assert_eq!(enc, direct, "LDR X0, [X1, W2, SXTW] = {enc:#010X}");
    }

    #[test]
    fn test_str_ro_64bit() {
        // STR X0, [X1, X2] — 64-bit register-offset store
        // size=11, V=0, opc=00, Rm=2, option=011(LSL), S=0, Rn=1, Rt=0
        let inst = mk(AArch64Opcode::StrRO, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_mem::encode_ldr_str_register(
            encoding_mem::LoadStoreSize::Double, false, encoding_mem::LoadStoreOp::Store,
            2, encoding_mem::RegExtend::Lsl, false, 1, 0,
        ).unwrap();
        assert_eq!(enc, direct, "STR X0, [X1, X2] = {enc:#010X}");
    }

    #[test]
    fn test_str_ro_32bit() {
        // STR W0, [X1, X2] — 32-bit register-offset store
        let inst = mk(AArch64Opcode::StrRO, vec![preg(W0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_mem::encode_ldr_str_register(
            encoding_mem::LoadStoreSize::Word, false, encoding_mem::LoadStoreOp::Store,
            2, encoding_mem::RegExtend::Lsl, false, 1, 0,
        ).unwrap();
        assert_eq!(enc, direct, "STR W0, [X1, X2] = {enc:#010X}");
    }

    #[test]
    fn test_str_ro_with_shift() {
        // STR X0, [X1, X2, LSL #3] — option=011, S=1 (packed=7)
        let inst = mk(AArch64Opcode::StrRO, vec![preg(X0), preg(X1), preg(X2), imm(7)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_mem::encode_ldr_str_register(
            encoding_mem::LoadStoreSize::Double, false, encoding_mem::LoadStoreOp::Store,
            2, encoding_mem::RegExtend::Lsl, true, 1, 0,
        ).unwrap();
        assert_eq!(enc, direct, "STR X0, [X1, X2, LSL #3] = {enc:#010X}");
    }

    #[test]
    fn test_ldr_ro_str_ro_differ_by_opc() {
        // LDR and STR with same operands should differ only in opc field
        let ldr = mk(AArch64Opcode::LdrRO, vec![preg(X0), preg(X1), preg(X2)]);
        let str_inst = mk(AArch64Opcode::StrRO, vec![preg(X0), preg(X1), preg(X2)]);
        let ldr_enc = encode_instruction(&ldr).unwrap();
        let str_enc = encode_instruction(&str_inst).unwrap();
        // opc is at bits [23:22]
        assert_eq!((ldr_enc >> 22) & 0b11, 0b01, "LDR opc=01");
        assert_eq!((str_enc >> 22) & 0b11, 0b00, "STR opc=00");
        // Everything else should be the same (mask out opc bits)
        let mask = !(0b11u32 << 22);
        assert_eq!(ldr_enc & mask, str_enc & mask, "LDR and STR should match except opc");
    }

    // =========================================================================
    // FP register-offset load/store tests (LdrRO, StrRO) — issue #155
    // =========================================================================

    #[test]
    fn test_ldr_ro_fp_double() {
        // LDR D0, [X1, X2] — 64-bit FP register-offset load
        // size=11, V=1, opc=01
        let inst = mk(AArch64Opcode::LdrRO, vec![preg(D0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_mem::encode_ldr_str_register(
            encoding_mem::LoadStoreSize::Double, true, encoding_mem::LoadStoreOp::Load,
            2, encoding_mem::RegExtend::Lsl, false, 1, 0,
        ).unwrap();
        assert_eq!(enc, direct, "LDR D0, [X1, X2] = {enc:#010X}");
        assert_eq!((enc >> 30) & 0b11, 0b11, "size must be 11 for Double");
        assert_eq!((enc >> 26) & 1, 1, "V must be 1 for FP");
    }

    #[test]
    fn test_ldr_ro_fp_single() {
        // LDR S0, [X1, X2] — 32-bit FP register-offset load
        // size=10, V=1, opc=01
        let inst = mk(AArch64Opcode::LdrRO, vec![preg(S0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_mem::encode_ldr_str_register(
            encoding_mem::LoadStoreSize::Word, true, encoding_mem::LoadStoreOp::Load,
            2, encoding_mem::RegExtend::Lsl, false, 1, 0,
        ).unwrap();
        assert_eq!(enc, direct, "LDR S0, [X1, X2] = {enc:#010X}");
        assert_eq!((enc >> 30) & 0b11, 0b10, "size must be 10 for Single");
        assert_eq!((enc >> 26) & 1, 1, "V must be 1 for FP");
    }

    #[test]
    fn test_ldr_ro_fp_half() {
        // LDR H0, [X1, X2] — 16-bit FP register-offset load
        // size=01, V=1, opc=01
        let inst = mk(AArch64Opcode::LdrRO, vec![preg(H0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_mem::encode_ldr_str_register(
            encoding_mem::LoadStoreSize::Half, true, encoding_mem::LoadStoreOp::Load,
            2, encoding_mem::RegExtend::Lsl, false, 1, 0,
        ).unwrap();
        assert_eq!(enc, direct, "LDR H0, [X1, X2] = {enc:#010X}");
        assert_eq!((enc >> 30) & 0b11, 0b01, "size must be 01 for Half");
        assert_eq!((enc >> 26) & 1, 1, "V must be 1 for FP");
    }

    #[test]
    fn test_str_ro_fp_double() {
        // STR D0, [X1, X2] — 64-bit FP register-offset store
        // size=11, V=1, opc=00
        let inst = mk(AArch64Opcode::StrRO, vec![preg(D0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_mem::encode_ldr_str_register(
            encoding_mem::LoadStoreSize::Double, true, encoding_mem::LoadStoreOp::Store,
            2, encoding_mem::RegExtend::Lsl, false, 1, 0,
        ).unwrap();
        assert_eq!(enc, direct, "STR D0, [X1, X2] = {enc:#010X}");
        assert_eq!((enc >> 30) & 0b11, 0b11, "size must be 11 for Double");
        assert_eq!((enc >> 26) & 1, 1, "V must be 1 for FP");
    }

    #[test]
    fn test_str_ro_fp_single() {
        // STR S0, [X1, X2] — 32-bit FP register-offset store
        // size=10, V=1, opc=00
        let inst = mk(AArch64Opcode::StrRO, vec![preg(S0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_mem::encode_ldr_str_register(
            encoding_mem::LoadStoreSize::Word, true, encoding_mem::LoadStoreOp::Store,
            2, encoding_mem::RegExtend::Lsl, false, 1, 0,
        ).unwrap();
        assert_eq!(enc, direct, "STR S0, [X1, X2] = {enc:#010X}");
        assert_eq!((enc >> 30) & 0b11, 0b10, "size must be 10 for Single");
        assert_eq!((enc >> 26) & 1, 1, "V must be 1 for FP");
    }

    #[test]
    fn test_str_ro_fp_half() {
        // STR H0, [X1, X2] — 16-bit FP register-offset store
        // size=01, V=1, opc=00
        let inst = mk(AArch64Opcode::StrRO, vec![preg(H0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding_mem::encode_ldr_str_register(
            encoding_mem::LoadStoreSize::Half, true, encoding_mem::LoadStoreOp::Store,
            2, encoding_mem::RegExtend::Lsl, false, 1, 0,
        ).unwrap();
        assert_eq!(enc, direct, "STR H0, [X1, X2] = {enc:#010X}");
        assert_eq!((enc >> 30) & 0b11, 0b01, "size must be 01 for Half");
        assert_eq!((enc >> 26) & 1, 1, "V must be 1 for FP");
    }

    #[test]
    fn test_ldr_ro_fp_sizes_differ() {
        // Verify that Half, Single, and Double produce different size fields
        let half = mk(AArch64Opcode::LdrRO, vec![preg(H0), preg(X1), preg(X2)]);
        let single = mk(AArch64Opcode::LdrRO, vec![preg(S0), preg(X1), preg(X2)]);
        let double = mk(AArch64Opcode::LdrRO, vec![preg(D0), preg(X1), preg(X2)]);
        let h_enc = encode_instruction(&half).unwrap();
        let s_enc = encode_instruction(&single).unwrap();
        let d_enc = encode_instruction(&double).unwrap();
        let h_size = (h_enc >> 30) & 0b11;
        let s_size = (s_enc >> 30) & 0b11;
        let d_size = (d_enc >> 30) & 0b11;
        assert_eq!(h_size, 0b01, "Half size=01");
        assert_eq!(s_size, 0b10, "Single size=10");
        assert_eq!(d_size, 0b11, "Double size=11");
        // All must have V=1
        assert_eq!((h_enc >> 26) & 1, 1, "Half V=1");
        assert_eq!((s_enc >> 26) & 1, 1, "Single V=1");
        assert_eq!((d_enc >> 26) & 1, 1, "Double V=1");
    }

    // =========================================================================
    // GOT/TLV load tests (LdrGot, LdrTlvp) — issue #137
    // =========================================================================

    #[test]
    fn test_ldr_got_zero_offset() {
        // LDR X0, [X1, #0] (GOT load with zero offset)
        let inst = mk(AArch64Opcode::LdrGot, vec![preg(X0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        // 64-bit load: size=11, V=0, opc=01, imm12=0, Rn=1, Rt=0
        let direct = encoding::encode_load_store_ui(0b11, 0, 0b01, 0, 1, 0);
        assert_eq!(enc, direct, "LDR X0, [X1] (GOT) = {enc:#010X}");
    }

    #[test]
    fn test_ldr_got_with_offset() {
        // LDR X0, [X1, #8] (GOT load, offset=8 -> scaled=1)
        let inst = mk(AArch64Opcode::LdrGot, vec![preg(X0), preg(X1), imm(8)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_load_store_ui(0b11, 0, 0b01, 1, 1, 0);
        assert_eq!(enc, direct, "LDR X0, [X1, #8] (GOT) = {enc:#010X}");
    }

    #[test]
    fn test_ldr_tlvp_zero_offset() {
        // LDR X0, [X1, #0] (TLV descriptor load)
        let inst = mk(AArch64Opcode::LdrTlvp, vec![preg(X0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_load_store_ui(0b11, 0, 0b01, 0, 1, 0);
        assert_eq!(enc, direct, "LDR X0, [X1] (TLV) = {enc:#010X}");
    }

    #[test]
    fn test_ldr_tlvp_with_offset() {
        // LDR X0, [X1, #16] (TLV load, offset=16 -> scaled=2)
        let inst = mk(AArch64Opcode::LdrTlvp, vec![preg(X0), preg(X1), imm(16)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_load_store_ui(0b11, 0, 0b01, 2, 1, 0);
        assert_eq!(enc, direct, "LDR X0, [X1, #16] (TLV) = {enc:#010X}");
    }

    #[test]
    fn test_ldr_got_and_tlvp_same_encoding() {
        // LdrGot and LdrTlvp with the same operands should produce identical encodings.
        // They differ only in relocation semantics, not in instruction encoding.
        let got = mk(AArch64Opcode::LdrGot, vec![preg(X0), preg(X1), imm(8)]);
        let tlvp = mk(AArch64Opcode::LdrTlvp, vec![preg(X0), preg(X1), imm(8)]);
        let got_enc = encode_instruction(&got).unwrap();
        let tlvp_enc = encode_instruction(&tlvp).unwrap();
        assert_eq!(got_enc, tlvp_enc, "LdrGot and LdrTlvp must produce identical encoding");
    }

    // =========================================================================
    // ARM ARM bit-pattern verification against known encodings
    // =========================================================================

    #[test]
    fn test_ubfm_known_encoding() {
        // UBFM X0, X1, #16, #31 — extract unsigned bitfield [31:16]
        // ARM ARM: 1_10_100110_1_010000_011111_00001_00000
        let inst = mk(AArch64Opcode::Ubfm, vec![preg(X0), preg(X1), imm(16), imm(31)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31) // sf=1
            | (0b10 << 29)         // opc=10
            | (0b100110 << 23)
            | (1 << 22)            // N=1
            | (16 << 16)           // immr=16
            | (31 << 10)           // imms=31
            | (1 << 5)            // Rn=1
            | 0;                   // Rd=0
        assert_eq!(enc, expected, "UBFM X0, X1, #16, #31 = {enc:#010X}");
    }

    #[test]
    fn test_sbfm_sxtb_alias() {
        // SXTB X0, X1 = SBFM X0, X1, #0, #7
        // Both paths should produce identical encoding
        let sbfm = mk(AArch64Opcode::Sbfm, vec![preg(X0), preg(X1), imm(0), imm(7)]);
        let sxtb = mk(AArch64Opcode::Sxtb, vec![preg(X0), preg(X1)]);
        let sbfm_enc = encode_instruction(&sbfm).unwrap();
        let sxtb_enc = encode_instruction(&sxtb).unwrap();
        assert_eq!(sbfm_enc, sxtb_enc, "SBFM X0, X1, #0, #7 must match SXTB X0, X1");
    }

    #[test]
    fn test_sbfm_sxth_alias() {
        // SXTH X0, X1 = SBFM X0, X1, #0, #15
        let sbfm = mk(AArch64Opcode::Sbfm, vec![preg(X0), preg(X1), imm(0), imm(15)]);
        let sxth = mk(AArch64Opcode::Sxth, vec![preg(X0), preg(X1)]);
        let sbfm_enc = encode_instruction(&sbfm).unwrap();
        let sxth_enc = encode_instruction(&sxth).unwrap();
        assert_eq!(sbfm_enc, sxth_enc, "SBFM X0, X1, #0, #15 must match SXTH X0, X1");
    }

    // --- preg_hw returns Result (#174) ---

    #[test]
    fn test_preg_hw_valid_preg() {
        // Valid PReg operand should return Ok(hw_encoding)
        let inst = mk(AArch64Opcode::AddRR, vec![preg(X0), preg(X1), preg(X2)]);
        assert_eq!(preg_hw(&inst, 0).unwrap(), 0);
        assert_eq!(preg_hw(&inst, 1).unwrap(), 1);
        assert_eq!(preg_hw(&inst, 2).unwrap(), 2);
    }

    #[test]
    fn test_preg_hw_valid_special_sp() {
        let inst = mk(AArch64Opcode::MovR, vec![preg(X0), sp()]);
        assert_eq!(preg_hw(&inst, 1).unwrap(), 31);
    }

    #[test]
    fn test_preg_hw_valid_special_xzr() {
        let inst = mk(
            AArch64Opcode::CmpRR,
            vec![preg(X0), MachOperand::Special(SpecialReg::XZR)],
        );
        assert_eq!(preg_hw(&inst, 1).unwrap(), 31);
    }

    #[test]
    fn test_preg_hw_valid_special_wzr() {
        let inst = mk(
            AArch64Opcode::CmpRR,
            vec![preg(W0), MachOperand::Special(SpecialReg::WZR)],
        );
        assert_eq!(preg_hw(&inst, 1).unwrap(), 31);
    }

    #[test]
    fn test_preg_hw_rejects_imm() {
        // Imm operand where register expected should return Err(InvalidOperand)
        let inst = mk(AArch64Opcode::AddRR, vec![preg(X0), preg(X1), imm(42)]);
        let err = preg_hw(&inst, 2);
        assert!(err.is_err(), "Imm where register expected should error");
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("invalid operand"), "Expected InvalidOperand error, got: {msg}");
    }

    #[test]
    fn test_preg_hw_rejects_missing_operand() {
        // Missing operand (index out of bounds) should return Err(MissingOperand)
        let inst = mk(AArch64Opcode::AddRR, vec![preg(X0), preg(X1)]);
        let err = preg_hw(&inst, 2);
        assert!(err.is_err(), "Missing operand should error");
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("missing"), "Expected MissingOperand error, got: {msg}");
    }

    #[test]
    fn test_preg_hw_rejects_block_operand() {
        // Block operand where register expected should return Err(InvalidOperand)
        let inst = mk(
            AArch64Opcode::AddRR,
            vec![preg(X0), preg(X1), MachOperand::Block(llvm2_ir::types::BlockId(0))],
        );
        let err = preg_hw(&inst, 2);
        assert!(err.is_err(), "Block operand where register expected should error");
    }

    #[test]
    fn test_encode_add_rr_with_imm_operand_errors() {
        // Full encode_instruction call with wrong operand type should propagate error
        let inst = mk(AArch64Opcode::AddRR, vec![preg(X0), preg(X1), imm(42)]);
        let result = encode_instruction(&inst);
        assert!(result.is_err(), "AddRR with Imm where Rm expected must error, not default to XZR");
    }

    #[test]
    fn test_encode_sub_rr_with_imm_operand_errors() {
        let inst = mk(AArch64Opcode::SubRR, vec![preg(X0), imm(10), preg(X2)]);
        let result = encode_instruction(&inst);
        assert!(result.is_err(), "SubRR with Imm where Rn expected must error");
    }

    #[test]
    fn test_encode_mul_with_missing_operand_errors() {
        let inst = mk(AArch64Opcode::MulRR, vec![preg(X0), preg(X1)]);
        let result = encode_instruction(&inst);
        assert!(result.is_err(), "MulRR with missing Rm must error");
    }

    // =========================================================================
    // ARM Architecture Reference Manual (DDI 0487) encoding verification
    //
    // Each test verifies the full 32-bit instruction word against the expected
    // bit pattern derived from the ARM ARM encoding tables. These are not
    // cross-checked against internal helpers — they are independent ground
    // truth assertions.
    //
    // Encoding derivations follow the format:
    //   bit[31] | bit[30] | ... | bit[4:0]
    // with the ARM ARM section referenced for each instruction class.
    // =========================================================================

    // --- ADD/SUB register (ARM ARM C6.2.5/C6.2.294) ---
    // Add/Subtract (shifted register): sf|op|S|01011|shift|0|Rm|imm6|Rn|Rd

    #[test]
    fn test_arm_arm_add_x0_x1_x2() {
        // ADD X0, X1, X2
        // ARM ARM C6.2.5: sf=1 op=0 S=0 01011 shift=00 0 Rm=00010 imm6=000000 Rn=00001 Rd=00000
        // = 1_0_0_01011_00_0_00010_000000_00001_00000
        // = 0x8B020020
        let inst = mk(AArch64Opcode::AddRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x8B020020, "ADD X0, X1, X2 = {enc:#010X}");
    }

    #[test]
    fn test_arm_arm_add_w3_w4_w5() {
        // ADD W3, W4, W5
        // ARM ARM: sf=0 op=0 S=0 01011 shift=00 0 Rm=00101 imm6=000000 Rn=00100 Rd=00011
        // = 0_0_0_01011_00_0_00101_000000_00100_00011
        // = 0x0B050083
        let inst = mk(AArch64Opcode::AddRR, vec![preg(W3), preg(W4), preg(W5)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x0B050083, "ADD W3, W4, W5 = {enc:#010X}");
    }

    #[test]
    fn test_arm_arm_sub_x0_x1_x2() {
        // SUB X0, X1, X2
        // ARM ARM C6.2.294: sf=1 op=1 S=0 01011 00 0 Rm=00010 000000 Rn=00001 Rd=00000
        // = 1_1_0_01011_00_0_00010_000000_00001_00000
        // = 0xCB020020
        let inst = mk(AArch64Opcode::SubRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xCB020020, "SUB X0, X1, X2 = {enc:#010X}");
    }

    // --- ADD/SUB immediate (ARM ARM C6.2.4/C6.2.293) ---
    // sf|op|S|100010|sh|imm12|Rn|Rd

    #[test]
    fn test_arm_arm_add_x0_x1_imm100() {
        // ADD X0, X1, #100
        // ARM ARM C6.2.4: sf=1 op=0 S=0 100010 sh=0 imm12=000001100100 Rn=00001 Rd=00000
        // = 1_0_0_100010_0_000001100100_00001_00000
        // = 0x91019020
        let inst = mk(AArch64Opcode::AddRI, vec![preg(X0), preg(X1), imm(100)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x91019020, "ADD X0, X1, #100 = {enc:#010X}");
    }

    #[test]
    fn test_arm_arm_sub_x0_x1_imm100() {
        // SUB X0, X1, #100
        // ARM ARM C6.2.293: sf=1 op=1 S=0 100010 sh=0 imm12=000001100100 Rn=00001 Rd=00000
        // = 1_1_0_100010_0_000001100100_00001_00000
        // = 0xD1019020
        let inst = mk(AArch64Opcode::SubRI, vec![preg(X0), preg(X1), imm(100)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xD1019020, "SUB X0, X1, #100 = {enc:#010X}");
    }

    // --- MOV register (ARM ARM C6.2.186) ---
    // MOV Xd, Xm is alias for ORR Xd, XZR, Xm
    // Logical shifted reg: sf|opc|01010|shift|N|Rm|imm6|Rn|Rd

    #[test]
    fn test_arm_arm_mov_x0_x1() {
        // MOV X0, X1 = ORR X0, XZR, X1
        // ARM ARM C6.2.186/C6.2.215: sf=1 opc=01 01010 shift=00 N=0 Rm=00001 imm6=000000 Rn=11111 Rd=00000
        // = 1_01_01010_00_0_00001_000000_11111_00000
        // = 0xAA0103E0
        let inst = mk(AArch64Opcode::MovR, vec![preg(X0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xAA0103E0, "MOV X0, X1 = {enc:#010X}");
    }

    // --- MOVZ (ARM ARM C6.2.191) ---
    // sf|opc=10|100101|hw|imm16|Rd

    #[test]
    fn test_arm_arm_movz_x0_0x1234() {
        // MOVZ X0, #0x1234
        // ARM ARM C6.2.191: sf=1 opc=10 100101 hw=00 imm16=0001001000110100 Rd=00000
        // Encoding: (1<<31)|(0b10<<29)|(0b100101<<23)|(0<<21)|(0x1234<<5)|0
        //         = 0x80000000|0x40000000|0x12800000|0x00024680
        //         = 0xD2824680
        let inst = mk(AArch64Opcode::Movz, vec![preg(X0), imm(0x1234)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xD2824680, "MOVZ X0, #0x1234 = {enc:#010X}");
    }

    #[test]
    fn test_arm_arm_movz_w0_0x_ffff() {
        // MOVZ W0, #0xFFFF
        // ARM ARM: sf=0 opc=10 100101 hw=00 imm16=1111111111111111 Rd=00000
        // Encoding: (0<<31)|(0b10<<29)|(0b100101<<23)|(0<<21)|(0xFFFF<<5)|0
        //         = 0x00000000|0x40000000|0x12800000|0x001FFFE0
        //         = 0x529FFFE0
        let inst = mk(AArch64Opcode::Movz, vec![preg(W0), imm(0xFFFF)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x529FFFE0, "MOVZ W0, #0xFFFF = {enc:#010X}");
    }

    // --- MOVK (ARM ARM C6.2.189) ---
    // sf|opc=11|100101|hw|imm16|Rd

    #[test]
    fn test_arm_arm_movk_x0_0x5678() {
        // MOVK X0, #0x5678
        // ARM ARM C6.2.189: sf=1 opc=11 100101 hw=00 imm16=0101011001111000 Rd=00000
        // Encoding: (1<<31)|(0b11<<29)|(0b100101<<23)|(0<<21)|(0x5678<<5)|0
        //         = 0x80000000|0x60000000|0x12800000|0x000ACF00
        //         = 0xF28ACF00
        let inst = mk(AArch64Opcode::Movk, vec![preg(X0), imm(0x5678)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xF28ACF00, "MOVK X0, #0x5678 = {enc:#010X}");
    }

    // --- LDR unsigned immediate (ARM ARM C6.2.132) ---
    // size|111|V|01|opc|imm12|Rn|Rt

    #[test]
    fn test_arm_arm_ldr_x0_x1_imm16() {
        // LDR X0, [X1, #16]
        // ARM ARM C6.2.132: size=11 111 V=0 01 opc=01 imm12=000000000010 Rn=00001 Rt=00000
        // imm12 = byte_offset / 8 = 16 / 8 = 2
        // = 11_111_0_01_01_000000000010_00001_00000
        // = 0xF9400820
        let inst = mk(AArch64Opcode::LdrRI, vec![preg(X0), preg(X1), imm(16)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xF9400820, "LDR X0, [X1, #16] = {enc:#010X}");
    }

    #[test]
    fn test_arm_arm_ldr_w0_w1_imm0() {
        // LDR W0, [X1]  (no offset)
        // ARM ARM: size=10 111 V=0 01 opc=01 imm12=000000000000 Rn=00001 Rt=00000
        // = 10_111_0_01_01_000000000000_00001_00000
        // = 0xB9400020
        let inst = mk(AArch64Opcode::LdrRI, vec![preg(W0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xB9400020, "LDR W0, [X1] = {enc:#010X}");
    }

    // --- STR unsigned immediate (ARM ARM C6.2.280) ---
    // size|111|V|01|opc|imm12|Rn|Rt

    #[test]
    fn test_arm_arm_str_x0_x1_imm0() {
        // STR X0, [X1]
        // ARM ARM C6.2.280: size=11 111 V=0 01 opc=00 imm12=000000000000 Rn=00001 Rt=00000
        // = 11_111_0_01_00_000000000000_00001_00000
        // = 0xF9000020
        let inst = mk(AArch64Opcode::StrRI, vec![preg(X0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xF9000020, "STR X0, [X1] = {enc:#010X}");
    }

    #[test]
    fn test_arm_arm_str_x0_x1_imm8() {
        // STR X0, [X1, #8]
        // ARM ARM: size=11 111 V=0 01 opc=00 imm12=000000000001 Rn=00001 Rt=00000
        // imm12 = 8 / 8 = 1
        // = 11_111_0_01_00_000000000001_00001_00000
        // = 0xF9000420
        let inst = mk(AArch64Opcode::StrRI, vec![preg(X0), preg(X1), imm(8)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xF9000420, "STR X0, [X1, #8] = {enc:#010X}");
    }

    // --- B unconditional (ARM ARM C6.2.33) ---
    // op=0|00101|imm26

    #[test]
    fn test_arm_arm_b_offset4() {
        // B +16 (imm26 = 4, since offset is in units of 4 bytes)
        // ARM ARM C6.2.33: op=0 00101 imm26=00_0000_0000_0000_0000_0000_0100
        // = 0_00101_00000000000000000000000100
        // = 0x14000004
        let inst = mk(AArch64Opcode::B, vec![imm(4)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x14000004, "B +16 = {enc:#010X}");
    }

    // --- BL (ARM ARM C6.2.35) ---
    // op=1|00101|imm26

    #[test]
    fn test_arm_arm_bl_offset4() {
        // BL +16 (imm26 = 4)
        // ARM ARM C6.2.35: op=1 00101 imm26=00000000000000000000000100
        // = 1_00101_00000000000000000000000100
        // = 0x94000004
        let inst = mk(AArch64Opcode::Bl, vec![imm(4)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x94000004, "BL +16 = {enc:#010X}");
    }

    // --- CMP register (ARM ARM C6.2.68) ---
    // CMP Xn, Xm = SUBS XZR, Xn, Xm
    // sf|op=1|S=1|01011|shift|0|Rm|imm6|Rn|Rd=11111

    #[test]
    fn test_arm_arm_cmp_x1_x2() {
        // CMP X1, X2 = SUBS XZR, X1, X2
        // ARM ARM: sf=1 op=1 S=1 01011 00 0 Rm=00010 000000 Rn=00001 Rd=11111
        // = 1_1_1_01011_00_0_00010_000000_00001_11111
        // = 0xEB02003F
        let inst = mk(AArch64Opcode::CmpRR, vec![preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xEB02003F, "CMP X1, X2 = {enc:#010X}");
    }

    // --- CMP immediate (ARM ARM C6.2.69) ---
    // CMP Xn, #imm = SUBS XZR, Xn, #imm
    // sf|op=1|S=1|100010|sh|imm12|Rn|Rd=11111

    #[test]
    fn test_arm_arm_cmp_x1_imm10() {
        // CMP X1, #10 = SUBS XZR, X1, #10
        // ARM ARM: sf=1 op=1 S=1 100010 sh=0 imm12=000000001010 Rn=00001 Rd=11111
        // = 1_1_1_100010_0_000000001010_00001_11111
        // = 0xF100283F
        let inst = mk(AArch64Opcode::CmpRI, vec![preg(X1), imm(10)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xF100283F, "CMP X1, #10 = {enc:#010X}");
    }

    // --- AND register (ARM ARM C6.2.12) ---
    // Logical shifted register: sf|opc=00|01010|shift|N=0|Rm|imm6|Rn|Rd

    #[test]
    fn test_arm_arm_and_x0_x1_x2() {
        // AND X0, X1, X2
        // ARM ARM C6.2.12: sf=1 opc=00 01010 shift=00 N=0 Rm=00010 imm6=000000 Rn=00001 Rd=00000
        // = 1_00_01010_00_0_00010_000000_00001_00000
        // = 0x8A020020
        let inst = mk(AArch64Opcode::AndRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x8A020020, "AND X0, X1, X2 = {enc:#010X}");
    }

    // --- ORR register (ARM ARM C6.2.215) ---
    // sf|opc=01|01010|shift|N=0|Rm|imm6|Rn|Rd

    #[test]
    fn test_arm_arm_orr_x0_x1_x2() {
        // ORR X0, X1, X2
        // ARM ARM C6.2.215: sf=1 opc=01 01010 shift=00 N=0 Rm=00010 imm6=000000 Rn=00001 Rd=00000
        // = 1_01_01010_00_0_00010_000000_00001_00000
        // = 0xAA020020
        let inst = mk(AArch64Opcode::OrrRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xAA020020, "ORR X0, X1, X2 = {enc:#010X}");
    }

    // --- EOR register (ARM ARM C6.2.92) ---
    // sf|opc=10|01010|shift|N=0|Rm|imm6|Rn|Rd

    #[test]
    fn test_arm_arm_eor_x0_x1_x2() {
        // EOR X0, X1, X2
        // ARM ARM C6.2.92: sf=1 opc=10 01010 shift=00 N=0 Rm=00010 imm6=000000 Rn=00001 Rd=00000
        // = 1_10_01010_00_0_00010_000000_00001_00000
        // = 0xCA020020
        let inst = mk(AArch64Opcode::EorRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xCA020020, "EOR X0, X1, X2 = {enc:#010X}");
    }

    // --- RET (ARM ARM C6.2.241) ---
    // 1101011|opc=0010|11111|000000|Rn|00000

    #[test]
    fn test_arm_arm_ret() {
        // RET (X30)
        // ARM ARM C6.2.241: 1101011 0010 11111 000000 Rn=11110 00000
        // = 1101011_0_0010_11111_000000_11110_00000
        // = 0xD65F03C0
        let inst = mk(AArch64Opcode::Ret, vec![]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xD65F03C0, "RET = {enc:#010X}");
    }

    // --- BLR (ARM ARM C6.2.36) ---
    // 1101011|opc=0001|11111|000000|Rn|00000

    #[test]
    fn test_arm_arm_blr_x0() {
        // BLR X0
        // ARM ARM C6.2.36: 1101011 0001 11111 000000 Rn=00000 00000
        // = 1101011_0_0001_11111_000000_00000_00000
        // = 0xD63F0000
        let inst = mk(AArch64Opcode::Blr, vec![preg(X0)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xD63F0000, "BLR X0 = {enc:#010X}");
    }

    // --- NOP (ARM ARM C6.2.202) ---

    #[test]
    fn test_arm_arm_nop() {
        // NOP = 0xD503201F (system hint instruction)
        let inst = mk(AArch64Opcode::Nop, vec![]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xD503201F, "NOP = {enc:#010X}");
    }

    // --- CSEL (ARM ARM C6.2.76) ---
    // sf|op=0|S=0|11010100|Rm|cond|op2=00|Rn|Rd

    #[test]
    fn test_arm_arm_csel_x0_x1_x2_eq() {
        // CSEL X0, X1, X2, EQ
        // ARM ARM C6.2.76: sf=1 op=0 S=0 11010100 Rm=00010 cond=0000 op2=00 Rn=00001 Rd=00000
        // = 1_0_0_11010100_00010_0000_00_00001_00000
        // = 0x9A820020
        let inst = mk(AArch64Opcode::Csel, vec![preg(X0), preg(X1), preg(X2), imm(0)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x9A820020, "CSEL X0, X1, X2, EQ = {enc:#010X}");
    }

    // --- CSET (ARM ARM C6.2.70) ---
    // CSET Xd, cond = CSINC Xd, XZR, XZR, invert(cond)
    // sf|op=0|S=0|11010100|Rm=11111|inv_cond|op2=01|Rn=11111|Rd

    #[test]
    fn test_arm_arm_cset_x0_eq() {
        // CSET X0, EQ = CSINC X0, XZR, XZR, NE (inv of EQ=0000 is 0001)
        // ARM ARM: sf=1 0 0 11010100 Rm=11111 cond=0001 01 Rn=11111 Rd=00000
        // = 1_0_0_11010100_11111_0001_01_11111_00000
        // = 0x9A9F17E0
        let inst = mk(AArch64Opcode::CSet, vec![preg(X0), imm(0)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x9A9F17E0, "CSET X0, EQ = {enc:#010X}");
    }

    // --- B.cond (ARM ARM C6.2.34) ---
    // 01010100|imm19|0|cond

    #[test]
    fn test_arm_arm_b_eq_offset2() {
        // B.EQ +8 (imm19=2, in instruction units)
        // ARM ARM C6.2.34: 01010100 imm19=0000000000000000010 0 cond=0000
        // = 01010100_0000000000000000010_0_0000
        // = 0x54000040
        let inst = mk(AArch64Opcode::BCond, vec![imm(0), imm(2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x54000040, "B.EQ +8 = {enc:#010X}");
    }

    // --- CBZ (ARM ARM C6.2.44) ---
    // sf|011010|op=0|imm19|Rt

    #[test]
    fn test_arm_arm_cbz_x1_offset2() {
        // CBZ X1, +8 (imm19=2)
        // ARM ARM: sf=1 011010 op=0 imm19=0000000000000000010 Rt=00001
        // = 1_011010_0_0000000000000000010_00001
        // = 0xB4000041
        let inst = mk(AArch64Opcode::Cbz, vec![preg(X1), imm(2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xB4000041, "CBZ X1, +8 = {enc:#010X}");
    }

    // --- SXTW (ARM ARM C6.2.300) ---
    // SXTW Xd, Wn = SBFM Xd, Xn, #0, #31
    // sf=1|opc=00|100110|N=1|immr=000000|imms=011111|Rn|Rd

    #[test]
    fn test_arm_arm_sxtw_x0_x1() {
        // SXTW X0, X1 = SBFM X0, X1, #0, #31
        // ARM ARM: 1_00_100110_1_000000_011111_00001_00000
        // = 0x93407C20
        let inst = mk(AArch64Opcode::Sxtw, vec![preg(X0), preg(X1)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x93407C20, "SXTW X0, X1 = {enc:#010X}");
    }

    // --- MUL (ARM ARM C6.2.199) ---
    // MUL Xd, Xn, Xm = MADD Xd, Xn, Xm, XZR
    // sf|00|11011|000|Rm|o0=0|Ra=11111|Rn|Rd

    #[test]
    fn test_arm_arm_mul_x0_x1_x2() {
        // MUL X0, X1, X2 = MADD X0, X1, X2, XZR
        // ARM ARM: sf=1 00 11011 000 Rm=00010 o0=0 Ra=11111 Rn=00001 Rd=00000
        // = 1_00_11011_000_00010_0_11111_00001_00000
        // = 0x9B027C20
        let inst = mk(AArch64Opcode::MulRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x9B027C20, "MUL X0, X1, X2 = {enc:#010X}");
    }

    // --- SDIV (ARM ARM C6.2.253) ---
    // sf|0|0011010110|Rm|000011|Rn|Rd

    #[test]
    fn test_arm_arm_sdiv_x0_x1_x2() {
        // SDIV X0, X1, X2
        // ARM ARM: sf=1 0 0011010110 Rm=00010 000011 Rn=00001 Rd=00000
        // = 1_0_0011010110_00010_000011_00001_00000
        // = 0x9AC20C20
        let inst = mk(AArch64Opcode::SDiv, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x9AC20C20, "SDIV X0, X1, X2 = {enc:#010X}");
    }

    // --- TST register (ARM ARM C6.2.311) ---
    // TST Xn, Xm = ANDS XZR, Xn, Xm
    // sf|opc=11|01010|shift=00|N=0|Rm|imm6=000000|Rn|Rd=11111

    #[test]
    fn test_arm_arm_tst_x1_x2() {
        // TST X1, X2 = ANDS XZR, X1, X2
        // ARM ARM: sf=1 opc=11 01010 00 0 Rm=00010 000000 Rn=00001 Rd=11111
        // = 1_11_01010_00_0_00010_000000_00001_11111
        // = 0xEA02003F
        let inst = mk(AArch64Opcode::Tst, vec![preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xEA02003F, "TST X1, X2 = {enc:#010X}");
    }

    // --- BR (ARM ARM C6.2.38) ---
    // 1101011|opc=0000|11111|000000|Rn|00000

    #[test]
    fn test_arm_arm_br_x0() {
        // BR X0
        // ARM ARM: 1101011 0000 11111 000000 Rn=00000 00000
        // = 1101011_0_0000_11111_000000_00000_00000
        // = 0xD61F0000
        let inst = mk(AArch64Opcode::Br, vec![preg(X0)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xD61F0000, "BR X0 = {enc:#010X}");
    }

    // --- ADDS flag-setting (ARM ARM C6.2.6) ---
    // sf|op=0|S=1|01011|shift|0|Rm|imm6|Rn|Rd

    #[test]
    fn test_arm_arm_adds_x0_x1_x2() {
        // ADDS X0, X1, X2
        // ARM ARM: sf=1 op=0 S=1 01011 00 0 Rm=00010 000000 Rn=00001 Rd=00000
        // = 1_0_1_01011_00_0_00010_000000_00001_00000
        // = 0xAB020020
        let inst = mk(AArch64Opcode::AddsRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xAB020020, "ADDS X0, X1, X2 = {enc:#010X}");
    }

    // --- SUBS flag-setting (ARM ARM C6.2.297) ---
    // sf|op=1|S=1|01011|shift|0|Rm|imm6|Rn|Rd

    #[test]
    fn test_arm_arm_subs_x0_x1_x2() {
        // SUBS X0, X1, X2
        // ARM ARM: sf=1 op=1 S=1 01011 00 0 Rm=00010 000000 Rn=00001 Rd=00000
        // = 1_1_1_01011_00_0_00010_000000_00001_00000
        // = 0xEB020020
        let inst = mk(AArch64Opcode::SubsRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0xEB020020, "SUBS X0, X1, X2 = {enc:#010X}");
    }

    // ===================================================================
    // Tests for newly-implemented opcodes (issue #187)
    // ===================================================================

    // --- BIC (Bitwise AND-NOT) ---
    // ARM ARM C6.2.38: sf|00|01010|shift(2)|1|Rm|imm6|Rn|Rd
    // BIC = AND with N=1 (inverted Rm)

    #[test]
    fn test_bic_x0_x1_x2() {
        // BIC X0, X1, X2
        // sf=1, opc=00, shift=00, N=1, Rm=2, imm6=0, Rn=1, Rd=0
        // 1_00_01010_00_1_00010_000000_00001_00000
        // = 0x8A220020
        let inst = mk(AArch64Opcode::BicRR, vec![preg(X0), preg(X1), preg(X2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_logical_shifted_reg(1, 0b00, 0, 1, 2, 0, 1, 0);
        assert_eq!(enc, direct, "BIC X0, X1, X2: unified={enc:#010X}, direct={direct:#010X}");
        assert_eq!(enc, 0x8A220020, "BIC X0, X1, X2 ARM ARM = {enc:#010X}");
    }

    #[test]
    fn test_bic_w0_w1_w2() {
        // BIC W0, W1, W2 (32-bit)
        let inst = mk(AArch64Opcode::BicRR, vec![preg(W0), preg(W1), preg(W2)]);
        let enc = encode_instruction(&inst).unwrap();
        let direct = encoding::encode_logical_shifted_reg(0, 0b00, 0, 1, 2, 0, 1, 0);
        assert_eq!(enc, direct, "BIC W0, W1, W2 = {enc:#010X}");
    }

    // --- CSINC (Conditional Select Increment) ---
    // ARM ARM C6.2.78: sf|00|11010100|Rm|cond|01|Rn|Rd

    #[test]
    fn test_csinc_x0_x1_x2_eq() {
        // CSINC X0, X1, X2, EQ
        // sf=1, op=0, S=0, 11010100, Rm=2, cond=0000(EQ), op2=01, Rn=1, Rd=0
        // 1_00_11010100_00010_0000_01_00001_00000
        let inst = mk(AArch64Opcode::Csinc, vec![preg(X0), preg(X1), preg(X2), imm(0)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31)
            | (0b11010100 << 21)
            | (2 << 16)
            | (0b0000 << 12) // EQ
            | (0b01 << 10)
            | (1 << 5);
        assert_eq!(enc, expected, "CSINC X0, X1, X2, EQ = {enc:#010X}");
    }

    #[test]
    fn test_csinc_w0_w1_w2_ne() {
        // CSINC W0, W1, W2, NE (32-bit)
        // sf=0, cond=0001(NE), op2=01
        let inst = mk(AArch64Opcode::Csinc, vec![preg(W0), preg(W1), preg(W2), imm(1)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (0b11010100u32 << 21)
            | (2 << 16)
            | (0b0001 << 12) // NE
            | (0b01 << 10)
            | (1 << 5);
        assert_eq!(enc, expected, "CSINC W0, W1, W2, NE = {enc:#010X}");
    }

    // --- CSINV (Conditional Select Invert) ---
    // ARM ARM C6.2.79: sf|10|11010100|Rm|cond|00|Rn|Rd

    #[test]
    fn test_csinv_x0_x1_x2_eq() {
        // CSINV X0, X1, X2, EQ
        // sf=1, op=1, S=0, 11010100, Rm=2, cond=0000(EQ), op2=00, Rn=1, Rd=0
        let inst = mk(AArch64Opcode::Csinv, vec![preg(X0), preg(X1), preg(X2), imm(0)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31)
            | (0b10 << 29) // op=1, S=0
            | (0b11010100 << 21)
            | (2 << 16)
            | (0b0000 << 12) // EQ
            | (0b00 << 10) // op2=00
            | (1 << 5);
        assert_eq!(enc, expected, "CSINV X0, X1, X2, EQ = {enc:#010X}");
    }

    #[test]
    fn test_csinv_w0_w1_w2_lt() {
        // CSINV W0, W1, W2, LT (32-bit)
        // sf=0, cond=1011(LT)
        let inst = mk(AArch64Opcode::Csinv, vec![preg(W0), preg(W1), preg(W2), imm(0b1011)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (0b10u32 << 29)
            | (0b11010100 << 21)
            | (2 << 16)
            | (0b1011 << 12) // LT
            | (0b00 << 10)
            | (1 << 5);
        assert_eq!(enc, expected, "CSINV W0, W1, W2, LT = {enc:#010X}");
    }

    // --- CSNEG (Conditional Select Negate) ---
    // ARM ARM C6.2.81: sf|10|11010100|Rm|cond|01|Rn|Rd

    #[test]
    fn test_csneg_x0_x1_x2_eq() {
        // CSNEG X0, X1, X2, EQ
        // sf=1, op=1, S=0, 11010100, Rm=2, cond=0000(EQ), op2=01, Rn=1, Rd=0
        let inst = mk(AArch64Opcode::Csneg, vec![preg(X0), preg(X1), preg(X2), imm(0)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31)
            | (0b10 << 29) // op=1, S=0
            | (0b11010100 << 21)
            | (2 << 16)
            | (0b0000 << 12) // EQ
            | (0b01 << 10) // op2=01
            | (1 << 5);
        assert_eq!(enc, expected, "CSNEG X0, X1, X2, EQ = {enc:#010X}");
    }

    #[test]
    fn test_csneg_w0_w1_w2_ge() {
        // CSNEG W0, W1, W2, GE
        // sf=0, cond=1010(GE)
        let inst = mk(AArch64Opcode::Csneg, vec![preg(W0), preg(W1), preg(W2), imm(0b1010)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (0b10u32 << 29)
            | (0b11010100 << 21)
            | (2 << 16)
            | (0b1010 << 12) // GE
            | (0b01 << 10)
            | (1 << 5);
        assert_eq!(enc, expected, "CSNEG W0, W1, W2, GE = {enc:#010X}");
    }

    // --- MOVN (Move Wide with NOT) ---
    // ARM ARM C6.2.208: sf|00|100101|hw|imm16|Rd

    #[test]
    fn test_movn_x0_0() {
        // MOVN X0, #0 — encodes -1 (all ones)
        // sf=1, opc=00, hw=0, imm16=0, Rd=0
        // 1_00_100101_00_0000000000000000_00000
        // = 0x92800000
        let inst = mk(AArch64Opcode::Movn, vec![preg(X0), imm(0)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!(enc, 0x92800000, "MOVN X0, #0 = {enc:#010X}");
    }

    #[test]
    fn test_movn_x0_0xffff() {
        // MOVN X0, #0xFFFF — encodes ~0xFFFF = 0xFFFFFFFFFFFF0000
        let inst = mk(AArch64Opcode::Movn, vec![preg(X0), imm(0xFFFF)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = encoding::encode_move_wide(1, 0b00, 0, 0xFFFF, 0);
        assert_eq!(enc, expected, "MOVN X0, #0xFFFF = {enc:#010X}");
    }

    #[test]
    fn test_movn_w0_42() {
        // MOVN W0, #42
        // sf=0, opc=00, hw=0, imm16=42, Rd=0
        let inst = mk(AArch64Opcode::Movn, vec![preg(W0), imm(42)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = encoding::encode_move_wide(0, 0b00, 0, 42, 0);
        assert_eq!(enc, expected, "MOVN W0, #42 = {enc:#010X}");
    }

    // --- Logical immediate (AND/ORR/EOR) ---
    // ARM ARM C4.1.4: sf|opc(2)|100100|N|immr(6)|imms(6)|Rn(5)|Rd(5)
    // AND=00, ORR=01, EOR=10

    #[test]
    fn test_and_ri_x0_x1() {
        // AND X0, X1, #<bitmask>
        // Operands: [Rd=X0, Rn=X1, N=1, immr=0, imms=7]
        // This encodes AND X0, X1, #0xFF (byte mask)
        // sf=1, opc=00, 100100, N=1, immr=000000, imms=000111, Rn=1, Rd=0
        // 1_00_100100_1_000000_000111_00001_00000
        let inst = mk(AArch64Opcode::AndRI, vec![preg(X0), preg(X1), imm(1), imm(0), imm(7)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31)
            | (0b00 << 29)
            | (0b100100 << 23)
            | (1 << 22) // N=1
            | (0 << 16) // immr=0
            | (7 << 10) // imms=7
            | (1 << 5)  // Rn=1
            | 0;         // Rd=0
        assert_eq!(enc, expected, "AND X0, X1, #0xFF = {enc:#010X}");
    }

    #[test]
    fn test_orr_ri_x0_xzr() {
        // ORR X0, XZR, #<bitmask> — materialize bitmask constant
        // Operands: [Rd=X0, Rn=XZR, N=0, immr=0, imms=0]
        // sf=1, opc=01, 100100, N=0, immr=0, imms=0, Rn=31, Rd=0
        let inst = mk(
            AArch64Opcode::OrrRI,
            vec![preg(X0), MachOperand::Special(SpecialReg::XZR), imm(0), imm(0), imm(0)],
        );
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31)
            | (0b01 << 29) // ORR
            | (0b100100 << 23)
            | (0 << 22) // N=0
            | (0 << 16) // immr=0
            | (0 << 10) // imms=0
            | (31 << 5) // Rn=XZR
            | 0;         // Rd=0
        assert_eq!(enc, expected, "ORR X0, XZR, #imm = {enc:#010X}");
    }

    #[test]
    fn test_eor_ri_x0_x1() {
        // EOR X0, X1, #<bitmask>
        // Operands: [Rd=X0, Rn=X1, N=1, immr=16, imms=31]
        // sf=1, opc=10, 100100, N=1, immr=010000, imms=011111, Rn=1, Rd=0
        let inst = mk(AArch64Opcode::EorRI, vec![preg(X0), preg(X1), imm(1), imm(16), imm(31)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (1u32 << 31)
            | (0b10 << 29) // EOR
            | (0b100100 << 23)
            | (1 << 22) // N=1
            | (16 << 16) // immr=16
            | (31 << 10) // imms=31
            | (1 << 5)  // Rn=1
            | 0;         // Rd=0
        assert_eq!(enc, expected, "EOR X0, X1, #imm = {enc:#010X}");
    }

    #[test]
    fn test_and_ri_w0_w1_32bit() {
        // AND W0, W1, #0xF (32-bit)
        // sf=0, opc=00, N=0 (must be 0 for 32-bit), immr=0, imms=3
        let inst = mk(AArch64Opcode::AndRI, vec![preg(W0), preg(W1), imm(0), imm(0), imm(3)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (0b00u32 << 29)
            | (0b100100 << 23)
            | (0 << 22) // N=0
            | (0 << 16) // immr=0
            | (3 << 10) // imms=3
            | (1 << 5)  // Rn=1
            | 0;
        assert_eq!(enc, expected, "AND W0, W1, #0xF = {enc:#010X}");
    }

    // --- FMOV immediate ---
    // ARM ARM C7.2.132: 0|0|0|11110|ftype(2)|1|imm8|100|00000|Rd

    #[test]
    fn test_fmov_imm_2_0_single() {
        // FMOV S0, #2.0
        // ftype=00 (single), imm8 encodes 2.0
        // 2.0 = sign=0, exp=128(biased)=1024(f64), mantissa=0
        // For f64: exp=1024 -> biased_3 = 1024-1020 = 4 = 0b100
        // NOT(b) = NOT(1) = 0, ccc[1:0] = 00
        // imm8 = 0_0_00_0000 = 0x00
        let inst = mk(AArch64Opcode::FmovImm, vec![preg(S0), MachOperand::FImm(2.0)]);
        let enc = encode_instruction(&inst).unwrap();
        // ftype=00, imm8=0x00
        let expected = (0b00011110u32 << 24)
            | (0b00 << 22) // ftype=single
            | (1 << 21)
            | (0x00 << 13) // imm8
            | (0b100 << 10)
            | 0; // Rd=S0
        assert_eq!(enc, expected, "FMOV S0, #2.0 = {enc:#010X}");
    }

    #[test]
    fn test_fmov_imm_1_0_double() {
        // FMOV D0, #1.0
        // 1.0f64: sign=0, exp=1023(biased), frac=0
        // biased_3 = 1023-1020 = 3 = 0b011
        // NOT(b) = NOT(0) = 1, ccc[1:0] = 11
        // imm8 = 0_1_11_0000 = 0x70
        let inst = mk(AArch64Opcode::FmovImm, vec![preg(D0), MachOperand::FImm(1.0)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (0b00011110u32 << 24)
            | (0b01 << 22) // ftype=double
            | (1 << 21)
            | (0x70u32 << 13) // imm8 = 0b01110000
            | (0b100 << 10)
            | 0; // Rd=D0
        assert_eq!(enc, expected, "FMOV D0, #1.0 = {enc:#010X}");
    }

    #[test]
    fn test_fmov_imm_neg_1_0() {
        // FMOV D0, #-1.0
        // -1.0f64: sign=1, exp=1023, frac=0
        // biased_3 = 3 => NOT(b)=1, cc=11
        // imm8 = 1_1_11_0000 = 0xF0
        let inst = mk(AArch64Opcode::FmovImm, vec![preg(D0), MachOperand::FImm(-1.0)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (0b00011110u32 << 24)
            | (0b01 << 22) // ftype=double
            | (1 << 21)
            | (0xF0u32 << 13) // imm8 = 0b11110000
            | (0b100 << 10)
            | 0;
        assert_eq!(enc, expected, "FMOV D0, #-1.0 = {enc:#010X}");
    }

    #[test]
    fn test_fmov_imm_0_5() {
        // FMOV D0, #0.5
        // 0.5f64: sign=0, exp=1022, frac=0
        // biased_3 = 1022-1020 = 2 = 0b010
        // NOT(b) = NOT(0) = 1, cc = 10
        // imm8 = 0_1_10_0000 = 0x60
        let inst = mk(AArch64Opcode::FmovImm, vec![preg(D0), MachOperand::FImm(0.5)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = (0b00011110u32 << 24)
            | (0b01 << 22)
            | (1 << 21)
            | (0x60u32 << 13)
            | (0b100 << 10)
            | 0;
        assert_eq!(enc, expected, "FMOV D0, #0.5 = {enc:#010X}");
    }

    // --- MOVN with hw_shift (from const_materialize) ---

    #[test]
    fn test_movn_with_hw_shift() {
        // MOVN X0, #0x1234, LSL #16
        // Operands: [Rd=X0, Imm(0x1234), Imm(16)]
        // hw = 16/16 = 1
        let inst = mk(AArch64Opcode::Movn, vec![preg(X0), imm(0x1234), imm(16)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = encoding::encode_move_wide(1, 0b00, 1, 0x1234, 0);
        assert_eq!(enc, expected, "MOVN X0, #0x1234, LSL#16 = {enc:#010X}");
    }

    // --- CSEL already tested above, but verify relationship with CSINC ---

    #[test]
    fn test_csel_vs_csinc_encoding_diff() {
        // CSEL and CSINC differ only in op2 bits [11:10]
        // CSEL: op2=00, CSINC: op2=01
        let csel = mk(AArch64Opcode::Csel, vec![preg(X0), preg(X1), preg(X2), imm(0)]);
        let csinc = mk(AArch64Opcode::Csinc, vec![preg(X0), preg(X1), preg(X2), imm(0)]);
        let enc_sel = encode_instruction(&csel).unwrap();
        let enc_inc = encode_instruction(&csinc).unwrap();
        // The only difference should be in bits 11:10
        assert_eq!(enc_sel & !0xC00, enc_inc & !0xC00,
            "CSEL and CSINC should differ only in op2 bits");
        assert_eq!((enc_sel >> 10) & 0b11, 0b00, "CSEL op2 = 00");
        assert_eq!((enc_inc >> 10) & 0b11, 0b01, "CSINC op2 = 01");
    }

    // --- CSINV vs CSNEG encoding difference ---

    #[test]
    fn test_csinv_vs_csneg_encoding_diff() {
        // CSINV and CSNEG share sf|10|11010100 prefix and differ in op2
        // CSINV: op2=00, CSNEG: op2=01
        let csinv = mk(AArch64Opcode::Csinv, vec![preg(X0), preg(X1), preg(X2), imm(0)]);
        let csneg = mk(AArch64Opcode::Csneg, vec![preg(X0), preg(X1), preg(X2), imm(0)]);
        let enc_inv = encode_instruction(&csinv).unwrap();
        let enc_neg = encode_instruction(&csneg).unwrap();
        assert_eq!(enc_inv & !0xC00, enc_neg & !0xC00,
            "CSINV and CSNEG should differ only in op2 bits");
        assert_eq!((enc_inv >> 10) & 0b11, 0b00, "CSINV op2 = 00");
        assert_eq!((enc_neg >> 10) & 0b11, 0b01, "CSNEG op2 = 01");
    }

    // --- LLVM-style typed alias tests ---

    #[test]
    fn test_movwrr_alias() {
        // MOVWrr W0, W1 — should encode same as MovR with 32-bit regs
        let alias = mk(AArch64Opcode::MOVWrr, vec![preg(W0), preg(W1)]);
        let generic = mk(AArch64Opcode::MovR, vec![preg(W0), preg(W1)]);
        assert_eq!(
            encode_instruction(&alias).unwrap(),
            encode_instruction(&generic).unwrap(),
            "MOVWrr should match MovR for W registers"
        );
    }

    #[test]
    fn test_movxrr_alias() {
        // MOVXrr X0, X1
        let alias = mk(AArch64Opcode::MOVXrr, vec![preg(X0), preg(X1)]);
        let generic = mk(AArch64Opcode::MovR, vec![preg(X0), preg(X1)]);
        assert_eq!(
            encode_instruction(&alias).unwrap(),
            encode_instruction(&generic).unwrap(),
            "MOVXrr should match MovR for X registers"
        );
    }

    #[test]
    fn test_bl_alias() {
        // BL (typed) should match Bl
        let alias = mk(AArch64Opcode::BL, vec![imm(100)]);
        let generic = mk(AArch64Opcode::Bl, vec![imm(100)]);
        assert_eq!(
            encode_instruction(&alias).unwrap(),
            encode_instruction(&generic).unwrap(),
            "BL alias should match Bl"
        );
    }

    #[test]
    fn test_blr_alias() {
        // BLR (typed) should match Blr
        let alias = mk(AArch64Opcode::BLR, vec![preg(X0)]);
        let generic = mk(AArch64Opcode::Blr, vec![preg(X0)]);
        assert_eq!(
            encode_instruction(&alias).unwrap(),
            encode_instruction(&generic).unwrap(),
            "BLR alias should match Blr"
        );
    }

    #[test]
    fn test_cmpwrr_alias() {
        // CMPWrr W0, W1 should match CmpRR with 32-bit regs
        let alias = mk(AArch64Opcode::CMPWrr, vec![preg(W0), preg(W1)]);
        let generic = mk(AArch64Opcode::CmpRR, vec![preg(W0), preg(W1)]);
        assert_eq!(
            encode_instruction(&alias).unwrap(),
            encode_instruction(&generic).unwrap(),
            "CMPWrr should match CmpRR"
        );
    }

    #[test]
    fn test_cmpxrr_alias() {
        let alias = mk(AArch64Opcode::CMPXrr, vec![preg(X0), preg(X1)]);
        let generic = mk(AArch64Opcode::CmpRR, vec![preg(X0), preg(X1)]);
        assert_eq!(
            encode_instruction(&alias).unwrap(),
            encode_instruction(&generic).unwrap(),
            "CMPXrr should match CmpRR"
        );
    }

    #[test]
    fn test_cmpwri_alias() {
        let alias = mk(AArch64Opcode::CMPWri, vec![preg(W0), imm(42)]);
        let generic = mk(AArch64Opcode::CmpRI, vec![preg(W0), imm(42)]);
        assert_eq!(
            encode_instruction(&alias).unwrap(),
            encode_instruction(&generic).unwrap(),
            "CMPWri should match CmpRI"
        );
    }

    #[test]
    fn test_cmpxri_alias() {
        let alias = mk(AArch64Opcode::CMPXri, vec![preg(X0), imm(42)]);
        let generic = mk(AArch64Opcode::CmpRI, vec![preg(X0), imm(42)]);
        assert_eq!(
            encode_instruction(&alias).unwrap(),
            encode_instruction(&generic).unwrap(),
            "CMPXri should match CmpRI"
        );
    }

    #[test]
    fn test_movzwi_alias() {
        let alias = mk(AArch64Opcode::MOVZWi, vec![preg(W0), imm(0x1234)]);
        let generic = mk(AArch64Opcode::Movz, vec![preg(W0), imm(0x1234)]);
        assert_eq!(
            encode_instruction(&alias).unwrap(),
            encode_instruction(&generic).unwrap(),
            "MOVZWi should match Movz"
        );
    }

    #[test]
    fn test_movzxi_alias() {
        let alias = mk(AArch64Opcode::MOVZXi, vec![preg(X0), imm(0x5678)]);
        let generic = mk(AArch64Opcode::Movz, vec![preg(X0), imm(0x5678)]);
        assert_eq!(
            encode_instruction(&alias).unwrap(),
            encode_instruction(&generic).unwrap(),
            "MOVZXi should match Movz"
        );
    }

    #[test]
    fn test_bcc_alias() {
        // Bcc with cond=EQ, offset=2
        let alias = mk(AArch64Opcode::Bcc, vec![imm(0), imm(2)]);
        let generic = mk(AArch64Opcode::BCond, vec![imm(0), imm(2)]);
        assert_eq!(
            encode_instruction(&alias).unwrap(),
            encode_instruction(&generic).unwrap(),
            "Bcc should match BCond"
        );
    }

    #[test]
    fn test_strwui_alias() {
        // STRWui W0, [X1, #4] — 32-bit store unsigned offset
        // size=10, V=0, opc=00, imm12=4/4=1
        let inst = mk(AArch64Opcode::STRWui, vec![preg(W0), preg(X1), imm(4)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = encoding::encode_load_store_ui(0b10, 0, 0b00, 1, 1, 0);
        assert_eq!(enc, expected, "STRWui W0, [X1, #4] = {enc:#010X}");
    }

    #[test]
    fn test_strxui_alias() {
        // STRXui X0, [X1, #8] — 64-bit store unsigned offset
        // size=11, V=0, opc=00, imm12=8/8=1
        let inst = mk(AArch64Opcode::STRXui, vec![preg(X0), preg(X1), imm(8)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = encoding::encode_load_store_ui(0b11, 0, 0b00, 1, 1, 0);
        assert_eq!(enc, expected, "STRXui X0, [X1, #8] = {enc:#010X}");
    }

    #[test]
    fn test_strsui_alias() {
        // STRSui S0, [X1, #4] — 32-bit FP store unsigned offset
        // size=10, V=1, opc=00, imm12=4/4=1
        let inst = mk(AArch64Opcode::STRSui, vec![preg(S0), preg(X1), imm(4)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = encoding::encode_load_store_ui(0b10, 1, 0b00, 1, 1, 0);
        assert_eq!(enc, expected, "STRSui S0, [X1, #4] = {enc:#010X}");
    }

    #[test]
    fn test_strdui_alias() {
        // STRDui D0, [X1, #8] — 64-bit FP store unsigned offset
        // size=11, V=1, opc=00, imm12=8/8=1
        let inst = mk(AArch64Opcode::STRDui, vec![preg(D0), preg(X1), imm(8)]);
        let enc = encode_instruction(&inst).unwrap();
        let expected = encoding::encode_load_store_ui(0b11, 1, 0b00, 1, 1, 0);
        assert_eq!(enc, expected, "STRDui D0, [X1, #8] = {enc:#010X}");
    }

    // --- encode_fmov_imm8 unit tests ---

    #[test]
    fn test_encode_fmov_imm8_1_0() {
        // 1.0 = 0x3FF0_0000_0000_0000
        // exp=1023, biased_3=3=0b011, NOT(b)=1, cc=11, frac_top4=0
        // imm8 = 0_1_11_0000 = 0x70
        assert_eq!(encode_fmov_imm8(1.0), 0x70);
    }

    #[test]
    fn test_encode_fmov_imm8_2_0() {
        // 2.0 = 0x4000_0000_0000_0000
        // exp=1024, biased_3=4=0b100, NOT(b)=0, cc=00, frac_top4=0
        // imm8 = 0_0_00_0000 = 0x00
        assert_eq!(encode_fmov_imm8(2.0), 0x00);
    }

    #[test]
    fn test_encode_fmov_imm8_0_5() {
        // 0.5 = 0x3FE0_0000_0000_0000
        // exp=1022, biased_3=2=0b010, NOT(b)=1, cc=10, frac_top4=0
        // imm8 = 0_1_10_0000 = 0x60
        assert_eq!(encode_fmov_imm8(0.5), 0x60);
    }

    #[test]
    fn test_encode_fmov_imm8_neg_1_0() {
        // -1.0: sign=1
        // imm8 = 1_1_11_0000 = 0xF0
        assert_eq!(encode_fmov_imm8(-1.0), 0xF0);
    }

    #[test]
    fn test_encode_fmov_imm8_1_5() {
        // 1.5 = 0x3FF8_0000_0000_0000
        // exp=1023, biased_3=3, NOT(b)=1, cc=11, frac_top4=0b1000=8
        // imm8 = 0_1_11_1000 = 0x78
        assert_eq!(encode_fmov_imm8(1.5), 0x78);
    }

    // --- ARM ARM cross-check: logical immediate fixed bits ---

    #[test]
    fn test_logical_imm_fixed_bits() {
        // Bits 28:23 must be 100100 for logical immediate
        let inst = mk(AArch64Opcode::AndRI, vec![preg(X0), preg(X1), imm(0), imm(0), imm(0)]);
        let enc = encode_instruction(&inst).unwrap();
        assert_eq!((enc >> 23) & 0x3F, 0b100100, "Logical imm fixed bits 28:23 = 100100");
    }

    #[test]
    fn test_logical_imm_opc_differentiation() {
        // AND=00, ORR=01, EOR=10 in bits 30:29
        let and_inst = mk(AArch64Opcode::AndRI, vec![preg(X0), preg(X1), imm(0), imm(0), imm(0)]);
        let orr_inst = mk(AArch64Opcode::OrrRI, vec![preg(X0), preg(X1), imm(0), imm(0), imm(0)]);
        let eor_inst = mk(AArch64Opcode::EorRI, vec![preg(X0), preg(X1), imm(0), imm(0), imm(0)]);
        let enc_and = encode_instruction(&and_inst).unwrap();
        let enc_orr = encode_instruction(&orr_inst).unwrap();
        let enc_eor = encode_instruction(&eor_inst).unwrap();
        assert_eq!((enc_and >> 29) & 0b11, 0b00, "AND opc = 00");
        assert_eq!((enc_orr >> 29) & 0b11, 0b01, "ORR opc = 01");
        assert_eq!((enc_eor >> 29) & 0b11, 0b10, "EOR opc = 10");
    }
}
