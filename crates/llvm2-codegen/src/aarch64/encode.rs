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

        // ORN Rd, Rn, Rm — bitwise OR-NOT (MVN when Rn=XZR)
        // Logical shifted register with N=1, opc=01
        AArch64Opcode::OrnRR => {
            let sf = sf_from_operand(inst, 0);
            Ok(encoding::encode_logical_shifted_reg(
                sf, 0b01, 1, 0, preg_hw(inst, 2), 0, preg_hw(inst, 1), preg_hw(inst, 0),
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
            let sf = sf_from_operand(inst, 0);
            Ok(encoding::encode_add_sub_shifted_reg(
                sf, 1, 1, 0, preg_hw(inst, 1), 0, preg_hw(inst, 0), 31,
            ))
        }

        // CMP Rn, #imm = SUBS XZR, Rn, #imm
        AArch64Opcode::CmpRI => {
            let sf = sf_from_operand(inst, 0);
            let imm = imm_val(inst, 1) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf, 1, 1, 0, imm, preg_hw(inst, 0), 31,
            ))
        }

        // TST Rn, Rm = ANDS XZR, Rn, Rm
        AArch64Opcode::Tst => {
            let sf = sf_from_operand(inst, 0);
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

        // CSET Xd/Wd, cond — encoded as CSINC Xd, XZR, XZR, invert(cond)
        // ARM ARM C6.2.70: sf | 0 | 0 | 11010100 | Rm(=XZR) | inv_cond | 0 | 1 | Rn(=XZR) | Rd
        // Operands: [PReg(Rd), Imm(cond_encoding)]
        AArch64Opcode::CSet => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0);
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
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            if is_fpr(inst, 0) {
                // FP load: V=1, size from FP register class
                let fp_sz = fp_size_from_inst(inst);
                let (size, scale) = match fp_sz {
                    FpSize::Single => (0b10, 4i64), // 32-bit FP
                    FpSize::Double | _ => (0b11, 8i64), // 64-bit FP (and 128-bit Q)
                };
                let scaled = (offset / scale) as u32 & 0xFFF;
                Ok(encoding::encode_load_store_ui(size, 1, 0b01, scaled, preg_hw(inst, 1), preg_hw(inst, 0)))
            } else {
                // Integer load: V=0, size from register class
                let sf = sf_from_operand(inst, 0);
                let (size, scale) = if sf == 1 {
                    (0b11, 8i64) // 64-bit (X registers)
                } else {
                    (0b10, 4i64) // 32-bit (W registers)
                };
                let scaled = (offset / scale) as u32 & 0xFFF;
                Ok(encoding::encode_load_store_ui(size, 0, 0b01, scaled, preg_hw(inst, 1), preg_hw(inst, 0)))
            }
        }

        // STR Rt, [Rn, #offset]
        AArch64Opcode::StrRI => {
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            if is_fpr(inst, 0) {
                let fp_sz = fp_size_from_inst(inst);
                let (size, scale) = match fp_sz {
                    FpSize::Single => (0b10, 4i64),
                    FpSize::Double | _ => (0b11, 8i64),
                };
                let scaled = (offset / scale) as u32 & 0xFFF;
                Ok(encoding::encode_load_store_ui(size, 1, 0b00, scaled, preg_hw(inst, 1), preg_hw(inst, 0)))
            } else {
                let sf = sf_from_operand(inst, 0);
                let (size, scale) = if sf == 1 {
                    (0b11, 8i64)
                } else {
                    (0b10, 4i64)
                };
                let scaled = (offset / scale) as u32 & 0xFFF;
                Ok(encoding::encode_load_store_ui(size, 0, 0b00, scaled, preg_hw(inst, 1), preg_hw(inst, 0)))
            }
        }

        // LDRB Wt, [Xn, #offset] — load byte, zero-extend to 32-bit
        // size=00, V=0, opc=01. Offset scaled by 1 byte.
        AArch64Opcode::LdrbRI => {
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            let scaled = offset as u32 & 0xFFF; // byte-scaled (scale=1)
            Ok(encoding::encode_load_store_ui(0b00, 0, 0b01, scaled, preg_hw(inst, 1), preg_hw(inst, 0)))
        }

        // LDRH Wt, [Xn, #offset] — load halfword, zero-extend to 32-bit
        // size=01, V=0, opc=01. Offset scaled by 2 bytes.
        AArch64Opcode::LdrhRI => {
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            let scaled = (offset / 2) as u32 & 0xFFF; // halfword-scaled (scale=2)
            Ok(encoding::encode_load_store_ui(0b01, 0, 0b01, scaled, preg_hw(inst, 1), preg_hw(inst, 0)))
        }

        // LDRSB Wt, [Xn, #offset] — load byte, sign-extend to 32-bit
        // size=00, V=0, opc=11 (sign-extend to 32-bit). Offset scaled by 1.
        AArch64Opcode::LdrsbRI => {
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            let scaled = offset as u32 & 0xFFF;
            Ok(encoding::encode_load_store_ui(0b00, 0, 0b11, scaled, preg_hw(inst, 1), preg_hw(inst, 0)))
        }

        // LDRSH Wt, [Xn, #offset] — load halfword, sign-extend to 32-bit
        // size=01, V=0, opc=11 (sign-extend to 32-bit). Offset scaled by 2.
        AArch64Opcode::LdrshRI => {
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            let scaled = (offset / 2) as u32 & 0xFFF;
            Ok(encoding::encode_load_store_ui(0b01, 0, 0b11, scaled, preg_hw(inst, 1), preg_hw(inst, 0)))
        }

        // STRB Wt, [Xn, #offset] — store byte (truncating)
        // size=00, V=0, opc=00. Offset scaled by 1.
        AArch64Opcode::StrbRI => {
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            let scaled = offset as u32 & 0xFFF;
            Ok(encoding::encode_load_store_ui(0b00, 0, 0b00, scaled, preg_hw(inst, 1), preg_hw(inst, 0)))
        }

        // STRH Wt, [Xn, #offset] — store halfword (truncating)
        // size=01, V=0, opc=00. Offset scaled by 2.
        AArch64Opcode::StrhRI => {
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            let scaled = (offset / 2) as u32 & 0xFFF;
            Ok(encoding::encode_load_store_ui(0b01, 0, 0b00, scaled, preg_hw(inst, 1), preg_hw(inst, 0)))
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

        // FNEG Dd, Dn — floating-point negate (1-source FP)
        AArch64Opcode::FnegRR => {
            let fp_size = fp_size_from_inst(inst);
            let enc = encoding_fp::encode_fp_unary(
                fp_size, encoding_fp::FpUnaryOp::Fneg,
                preg_hw(inst, 1) as u8, preg_hw(inst, 0) as u8,
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

        // FCVTZU Rd, Rn (FP to unsigned integer, round toward zero)
        AArch64Opcode::FcvtzuRR => {
            let sf_64 = true;
            let fp_size = fp_size_from_source(inst, 1);
            let enc = encoding_fp::encode_fp_int_conv(
                sf_64, fp_size, FpConvOp::FcvtzuToInt,
                preg_hw(inst, 1) as u8, preg_hw(inst, 0) as u8,
            )?;
            Ok(enc)
        }

        // UCVTF Rd, Rn (unsigned integer to FP)
        AArch64Opcode::UcvtfRR => {
            let sf_64 = true;
            let fp_size = fp_size_from_source(inst, 0);
            let enc = encoding_fp::encode_fp_int_conv(
                sf_64, fp_size, FpConvOp::UcvtfToFp,
                preg_hw(inst, 1) as u8, preg_hw(inst, 0) as u8,
            )?;
            Ok(enc)
        }

        // FCVT Dd, Sn (float precision widen: f32 -> f64)
        AArch64Opcode::FcvtSD => {
            let enc = encoding_fp::encode_fp_precision_cvt(
                FpSize::Single, FpSize::Double,
                preg_hw(inst, 1) as u8, preg_hw(inst, 0) as u8,
            )?;
            Ok(enc)
        }

        // FCVT Ss, Dn (float precision narrow: f64 -> f32)
        AArch64Opcode::FcvtDS => {
            let enc = encoding_fp::encode_fp_precision_cvt(
                FpSize::Double, FpSize::Single,
                preg_hw(inst, 1) as u8, preg_hw(inst, 0) as u8,
            )?;
            Ok(enc)
        }

        // FMOV Sd, Wn / FMOV Dd, Xn (GPR to FPR bitcast)
        AArch64Opcode::FmovGprFpr => {
            let sf_64 = true; // conservative default
            let fp_size = fp_size_from_source(inst, 0);
            let enc = encoding_fp::encode_fp_int_conv(
                sf_64, fp_size, FpConvOp::FmovToFp,
                preg_hw(inst, 1) as u8, preg_hw(inst, 0) as u8,
            )?;
            Ok(enc)
        }

        // FMOV Wn, Sd / FMOV Xn, Dd (FPR to GPR bitcast)
        AArch64Opcode::FmovFprGpr => {
            let sf_64 = true;
            let fp_size = fp_size_from_source(inst, 1);
            let enc = encoding_fp::encode_fp_int_conv(
                sf_64, fp_size, FpConvOp::FmovToGp,
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
        // Conditional select / set
        // =================================================================

        // CSEL Xd, Xn, Xm, cond
        // ARM ARM: C6.2.76  Conditional Select
        // 31 30 29 28:21    20:16 15:12 11 10 9:5  4:0
        // sf  0  0  11010100 Rm    cond   0  0  Rn   Rd
        // Operands: [dst, true_src, false_src, Imm(cond_code_encoding)]
        AArch64Opcode::Csel => {
            let sf = sf_from_operand(inst, 0);
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            let rm = preg_hw(inst, 2);
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
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
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
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
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
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
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
            let rt = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            let rm = preg_hw(inst, 2);
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
                let size = match fp_sz {
                    FpSize::Single => 0b10,
                    FpSize::Double | _ => 0b11,
                };
                Ok(encoding_mem::encode_ldr_str_register(
                    match size { 0b10 => encoding_mem::LoadStoreSize::Word, _ => encoding_mem::LoadStoreSize::Double },
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
            let rt = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            let rm = preg_hw(inst, 2);
            let (option, s) = if inst.operands.len() > 3 {
                let packed = imm_val(inst, 3) as u32;
                ((packed >> 1) & 0b111, packed & 1)
            } else {
                (0b011, 0)
            };
            if is_fpr(inst, 0) {
                let fp_sz = fp_size_from_inst(inst);
                let size = match fp_sz {
                    FpSize::Single => encoding_mem::LoadStoreSize::Word,
                    FpSize::Double | _ => encoding_mem::LoadStoreSize::Double,
                };
                Ok(encoding_mem::encode_ldr_str_register(
                    size,
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
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
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
            let rd = preg_hw(inst, 0);
            let rn = preg_hw(inst, 1);
            let offset = if inst.operands.len() > 2 { imm_val(inst, 2) } else { 0 };
            let scaled = (offset / 8) as u32 & 0xFFF;
            // Always 64-bit load (TLV descriptors are pointer-sized)
            Ok(encoding::encode_load_store_ui(0b11, 0, 0b01, scaled, rn, rd))
        }

        // Still-unimplemented opcodes from ISel unification (issue #73).
        AArch64Opcode::AndRI
        | AArch64Opcode::OrrRI
        | AArch64Opcode::EorRI
        | AArch64Opcode::BicRR
        | AArch64Opcode::Csinc
        | AArch64Opcode::Csinv
        | AArch64Opcode::Csneg
        | AArch64Opcode::Movn
        | AArch64Opcode::FmovImm => {
            // TODO(#73): Implement proper encoding for these opcodes.
            Err(EncodeError::UnsupportedOpcode(inst.opcode))
        }

        // LLVM-style typed aliases (used by llvm2-lower isel)
        AArch64Opcode::MOVWrr
        | AArch64Opcode::MOVXrr
        | AArch64Opcode::STRWui
        | AArch64Opcode::STRXui
        | AArch64Opcode::STRSui
        | AArch64Opcode::STRDui
        | AArch64Opcode::BL
        | AArch64Opcode::BLR
        | AArch64Opcode::CMPWrr
        | AArch64Opcode::CMPXrr
        | AArch64Opcode::CMPWri
        | AArch64Opcode::CMPXri
        | AArch64Opcode::MOVZWi
        | AArch64Opcode::MOVZXi
        | AArch64Opcode::Bcc => {
            // TODO: Map typed aliases to their generic counterparts for encoding.
            Err(EncodeError::UnsupportedOpcode(inst.opcode))
        }
    }
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::inst::{AArch64Opcode, MachInst};
    use llvm2_ir::operand::MachOperand;
    use llvm2_ir::regs::{PReg, SpecialReg, X0, X1, X2, X9, X30, V0, V1, V2, W0, W1, W2, S0, S1, S2};

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
}
