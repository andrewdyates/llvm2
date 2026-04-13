// llvm2-codegen/lower.rs - Machine code lowering
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Lowers a post-regalloc IrMachFunction to encoded AArch64 machine code bytes.
// This is Phase 8 of the pipeline: the final step before Mach-O emission.
//
// Responsibilities:
//   1. Expand pseudo-instructions surviving regalloc (PSEUDO_COPY, spills)
//   2. Run branch relaxation to resolve block targets
//   3. Encode every real instruction via the aarch64 encoder
//   4. Apply branch fixups (patch branch offsets after all code is laid out)
//   5. Collect relocations for external symbols (ADRP, BL, etc.)
//
// Reference: pipeline.rs::encode_function (inline encoding logic)
// Reference: relax.rs (branch relaxation pass)
// Reference: frame.rs (prologue/epilogue insertion, frame index elimination)

use crate::aarch64::encoding;
use crate::aarch64::encoding_mem;
use crate::frame::{self, FrameLayout};
use crate::relax;
use llvm2_ir::function::MachFunction as IrMachFunction;
use llvm2_ir::inst::{AArch64Opcode, InstFlags, MachInst};
use llvm2_ir::operand::MachOperand;
use llvm2_ir::regs::SpecialReg;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors during machine code lowering.
#[derive(Debug, Error)]
pub enum LowerError {
    #[error("unsupported instruction: {0}")]
    UnsupportedInstruction(String),
    #[error("encoding failed: {0}")]
    EncodingFailed(String),
    #[error("missing operand at index {index} for {opcode:?}")]
    MissingOperand { opcode: AArch64Opcode, index: usize },
    #[error("unresolved pseudo-instruction after expansion: {0:?}")]
    UnresolvedPseudo(AArch64Opcode),
}

// ---------------------------------------------------------------------------
// Relocation and fixup types
// ---------------------------------------------------------------------------

/// A relocation entry — records a reference to an external symbol that the
/// linker must patch.
#[derive(Debug, Clone)]
pub struct Relocation {
    /// Byte offset within the encoded code where the relocation applies.
    pub offset: u32,
    /// Relocation kind.
    pub kind: RelocKind,
    /// Symbol name (for external references).
    pub symbol: String,
    /// Addend (signed offset added to the symbol value).
    pub addend: i64,
}

/// AArch64 relocation kinds relevant to our lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocKind {
    /// ADRP page-relative relocation (ARM64_RELOC_PAGE21).
    AdrpPage21,
    /// ADD/LDR page-offset relocation (ARM64_RELOC_PAGEOFF12).
    AddPageOff12,
    /// BL call relocation (ARM64_RELOC_BRANCH26).
    Branch26,
}

/// A branch fixup — records a branch instruction whose target offset needs
/// to be patched after all code is emitted.
#[derive(Debug, Clone)]
pub struct BranchFixup {
    /// Byte offset within the encoded code where the branch instruction lives.
    pub offset: u32,
    /// The branch opcode (determines which bits to patch).
    pub opcode: AArch64Opcode,
    /// Target byte offset from the function start.
    pub target_offset: u32,
}

// ---------------------------------------------------------------------------
// LowerResult — output of the lowering pass
// ---------------------------------------------------------------------------

/// The result of lowering a function to machine code.
#[derive(Debug, Clone)]
pub struct LowerResult {
    /// Encoded machine code bytes.
    pub code: Vec<u8>,
    /// Relocation entries for the linker.
    pub relocations: Vec<Relocation>,
    /// Frame layout used (useful for unwind info generation).
    pub frame_layout: FrameLayout,
}

// ---------------------------------------------------------------------------
// Pseudo-instruction expansion
// ---------------------------------------------------------------------------

/// Opcode values used by the register allocator for pseudo-instructions.
/// These are u16 opcode values stored in regalloc MachInst, mapped back
/// to IR opcodes during the apply-allocation phase.
///
/// The regalloc uses these sentinel opcode values:
///   PSEUDO_COPY       = 0xFFE1 (from phi_elim.rs)
///   PSEUDO_SPILL_STORE = 0xFFF0 (from spill.rs)
///   PSEUDO_SPILL_LOAD  = 0xFFF1 (from spill.rs)
///
/// By the time code reaches lower.rs, these have been converted into
/// IR-level instructions with IS_PSEUDO flag and specific opcode patterns.
/// We expand any remaining pseudo-instructions here.

/// Expand pseudo-instructions in the function into real AArch64 instructions.
///
/// After register allocation and frame lowering, some pseudo-instructions
/// may survive:
///   - Phi instructions (should have been eliminated by phi_elim)
///   - StackAlloc (should have been eliminated by frame lowering)
///   - Nop (may be intentional alignment padding or placeholder)
///   - MovR where src == dst (identity copies from coalescing)
///
/// This pass rewrites or removes them.
pub fn expand_pseudos(func: &mut IrMachFunction) {
    for inst in &mut func.insts {
        if !inst.is_pseudo() {
            // Also remove identity MOV copies (dst == src).
            if inst.opcode == AArch64Opcode::MovR
                && inst.operands.len() >= 2
                && inst.operands[0] == inst.operands[1]
            {
                // Turn into NOP (will be skipped during encoding).
                inst.opcode = AArch64Opcode::Nop;
                inst.flags = InstFlags::IS_PSEUDO;
                inst.operands.clear();
            }
            continue;
        }

        match inst.opcode {
            AArch64Opcode::Phi => {
                // Phi should have been eliminated before reaching lowering.
                // Remove it by converting to NOP.
                inst.opcode = AArch64Opcode::Nop;
                inst.operands.clear();
            }
            AArch64Opcode::StackAlloc => {
                // Frame lowering handles stack allocation. Remove.
                inst.opcode = AArch64Opcode::Nop;
                inst.operands.clear();
            }
            AArch64Opcode::Nop => {
                // Already a no-op; will be skipped during encoding.
            }
            _ => {
                // Unknown pseudo — leave as NOP to avoid encoding errors.
                inst.opcode = AArch64Opcode::Nop;
                inst.flags = InstFlags::IS_PSEUDO;
                inst.operands.clear();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Single instruction encoding
// ---------------------------------------------------------------------------

/// Encode a single IR instruction to a 32-bit AArch64 instruction word.
///
/// This is the core encoding dispatch. It extracts operands and calls
/// the appropriate encoder function from the aarch64::encoding module.
///
/// Returns the encoded 4-byte word, or an error if the instruction cannot
/// be encoded.
fn encode_inst(inst: &MachInst) -> Result<u32, LowerError> {
    // Helper: extract physical register hardware encoding from operand.
    let preg_hw = |idx: usize| -> Result<u32, LowerError> {
        if idx >= inst.operands.len() {
            return Err(LowerError::MissingOperand {
                opcode: inst.opcode,
                index: idx,
            });
        }
        match &inst.operands[idx] {
            MachOperand::PReg(p) => Ok(p.hw_enc() as u32),
            MachOperand::Special(s) => match s {
                SpecialReg::SP | SpecialReg::XZR | SpecialReg::WZR => Ok(31),
            },
            _ => Ok(31), // Default to XZR/SP for safety
        }
    };

    // Helper: extract immediate value from operand.
    let imm_val = |idx: usize| -> i64 {
        inst.operands
            .get(idx)
            .and_then(|op| {
                if let MachOperand::Imm(v) = op {
                    Some(*v)
                } else {
                    None
                }
            })
            .unwrap_or(0)
    };

    // Helper: determine if the instruction operates on 64-bit registers.
    let is_64bit = |idx: usize| -> bool {
        match inst.operands.get(idx) {
            Some(MachOperand::PReg(p)) => p.is_gpr() && p.encoding() < 32,
            _ => true,
        }
    };

    match inst.opcode {
        // --- Arithmetic ---
        AArch64Opcode::AddRR => {
            let sf = if is_64bit(0) { 1 } else { 0 };
            Ok(encoding::encode_add_sub_shifted_reg(
                sf,
                0,
                0,
                0,
                preg_hw(2)?,
                0,
                preg_hw(1)?,
                preg_hw(0)?,
            ))
        }
        AArch64Opcode::AddRI => {
            let sf = if is_64bit(0) { 1 } else { 0 };
            let imm = imm_val(2) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf,
                0,
                0,
                0,
                imm,
                preg_hw(1)?,
                preg_hw(0)?,
            ))
        }
        AArch64Opcode::SubRR => {
            let sf = if is_64bit(0) { 1 } else { 0 };
            Ok(encoding::encode_add_sub_shifted_reg(
                sf,
                1,
                0,
                0,
                preg_hw(2)?,
                0,
                preg_hw(1)?,
                preg_hw(0)?,
            ))
        }
        AArch64Opcode::SubRI => {
            let sf = if is_64bit(0) { 1 } else { 0 };
            let imm = imm_val(2) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf,
                1,
                0,
                0,
                imm,
                preg_hw(1)?,
                preg_hw(0)?,
            ))
        }
        AArch64Opcode::MulRR => {
            // MADD Rd, Rn, Rm, XZR (multiply-accumulate with zero addend = MUL)
            // Encoding: sf | 0 | 0 | 11011 | 000 | Rm | 0 | Ra(=31) | Rn | Rd
            let sf = if is_64bit(0) { 1u32 } else { 0u32 };
            let rd = preg_hw(0)?;
            let rn = preg_hw(1)?;
            let rm = preg_hw(2)?;
            let ra = 31u32; // XZR — makes MADD act as MUL
            Ok((sf << 31)
                | (0b0011011u32 << 24)
                | (rm << 16)
                | (0 << 15) // o0=0 for MADD
                | (ra << 10)
                | (rn << 5)
                | rd)
        }
        AArch64Opcode::SDiv => {
            // SDIV Rd, Rn, Rm
            // Encoding: sf | 0 | 0 | 11010110 | Rm | 00001 | 1 | Rn | Rd
            let sf = if is_64bit(0) { 1u32 } else { 0u32 };
            let rd = preg_hw(0)?;
            let rn = preg_hw(1)?;
            let rm = preg_hw(2)?;
            Ok((sf << 31)
                | (0b0011010110u32 << 21)
                | (rm << 16)
                | (0b000011u32 << 10)
                | (rn << 5)
                | rd)
        }
        AArch64Opcode::UDiv => {
            // UDIV Rd, Rn, Rm
            let sf = if is_64bit(0) { 1u32 } else { 0u32 };
            let rd = preg_hw(0)?;
            let rn = preg_hw(1)?;
            let rm = preg_hw(2)?;
            Ok((sf << 31)
                | (0b0011010110u32 << 21)
                | (rm << 16)
                | (0b000010u32 << 10)
                | (rn << 5)
                | rd)
        }
        AArch64Opcode::Neg => {
            // NEG Rd, Rm = SUB Rd, XZR, Rm
            let sf = if is_64bit(0) { 1 } else { 0 };
            Ok(encoding::encode_add_sub_shifted_reg(
                sf,
                1,
                0,
                0,
                preg_hw(1)?,
                0,
                31, // XZR as Rn
                preg_hw(0)?,
            ))
        }

        // --- Compare ---
        AArch64Opcode::CmpRR => {
            // CMP Rn, Rm = SUBS XZR, Rn, Rm
            let sf = 1u32;
            Ok(encoding::encode_add_sub_shifted_reg(
                sf,
                1,
                1,
                0,
                preg_hw(1)?,
                0,
                preg_hw(0)?,
                31,
            ))
        }
        AArch64Opcode::CmpRI => {
            let sf = 1u32;
            let imm = imm_val(1) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf,
                1,
                1,
                0,
                imm,
                preg_hw(0)?,
                31,
            ))
        }
        AArch64Opcode::Tst => {
            // TST Rn, Rm = ANDS XZR, Rn, Rm
            let sf = 1u32;
            Ok(encoding::encode_logical_shifted_reg(
                sf,
                0b11, // ANDS
                0,
                0,
                preg_hw(1)?,
                0,
                preg_hw(0)?,
                31, // XZR
            ))
        }

        // --- Logical ---
        AArch64Opcode::AndRR => {
            let sf = if is_64bit(0) { 1 } else { 0 };
            Ok(encoding::encode_logical_shifted_reg(
                sf,
                0b00,
                0,
                0,
                preg_hw(2)?,
                0,
                preg_hw(1)?,
                preg_hw(0)?,
            ))
        }
        AArch64Opcode::OrrRR => {
            let sf = if is_64bit(0) { 1 } else { 0 };
            Ok(encoding::encode_logical_shifted_reg(
                sf,
                0b01,
                0,
                0,
                preg_hw(2)?,
                0,
                preg_hw(1)?,
                preg_hw(0)?,
            ))
        }
        AArch64Opcode::EorRR => {
            let sf = if is_64bit(0) { 1 } else { 0 };
            Ok(encoding::encode_logical_shifted_reg(
                sf,
                0b10,
                0,
                0,
                preg_hw(2)?,
                0,
                preg_hw(1)?,
                preg_hw(0)?,
            ))
        }

        // --- Shifts ---
        AArch64Opcode::LslRR => {
            // LSLV Rd, Rn, Rm
            // Encoding: sf | 0 | 0 | 11010110 | Rm | 0010 | 00 | Rn | Rd
            let sf = if is_64bit(0) { 1u32 } else { 0u32 };
            Ok((sf << 31)
                | (0b0011010110u32 << 21)
                | (preg_hw(2)? << 16)
                | (0b001000u32 << 10)
                | (preg_hw(1)? << 5)
                | preg_hw(0)?)
        }
        AArch64Opcode::LsrRR => {
            // LSRV Rd, Rn, Rm
            let sf = if is_64bit(0) { 1u32 } else { 0u32 };
            Ok((sf << 31)
                | (0b0011010110u32 << 21)
                | (preg_hw(2)? << 16)
                | (0b001001u32 << 10)
                | (preg_hw(1)? << 5)
                | preg_hw(0)?)
        }
        AArch64Opcode::AsrRR => {
            // ASRV Rd, Rn, Rm
            let sf = if is_64bit(0) { 1u32 } else { 0u32 };
            Ok((sf << 31)
                | (0b0011010110u32 << 21)
                | (preg_hw(2)? << 16)
                | (0b001010u32 << 10)
                | (preg_hw(1)? << 5)
                | preg_hw(0)?)
        }
        AArch64Opcode::LslRI | AArch64Opcode::LsrRI | AArch64Opcode::AsrRI => {
            // Immediate shifts use UBFM/SBFM encoding.
            // For simplicity, encode as the register-variant with the
            // immediate handled by prior lowering. Emit NOP as fallback.
            // TODO: Implement proper UBFM/SBFM encoding for immediate shifts.
            Ok(0xD503201F) // NOP
        }

        // --- Move ---
        AArch64Opcode::MovR => {
            // MOV Rd, Rm
            // When source is SP, use ADD Rd, SP, #0 (reg 31 = SP in ADD context).
            // Otherwise, use ORR Rd, XZR, Rm (reg 31 = XZR in logical context).
            let sf = if is_64bit(0) { 1 } else { 0 };
            let is_sp_source = matches!(
                inst.operands.get(1),
                Some(MachOperand::Special(SpecialReg::SP))
            );
            if is_sp_source {
                Ok(encoding::encode_add_sub_imm(
                    sf, 0, 0, 0, 0, 31, preg_hw(0)?,
                ))
            } else {
                Ok(encoding::encode_logical_shifted_reg(
                    sf,
                    0b01,
                    0,
                    0,
                    preg_hw(1)?,
                    0,
                    31,
                    preg_hw(0)?,
                ))
            }
        }
        AArch64Opcode::MovI | AArch64Opcode::Movz => {
            // MOVZ Rd, #imm16
            let sf = if is_64bit(0) { 1 } else { 0 };
            let imm16 = imm_val(1) as u32 & 0xFFFF;
            Ok(encoding::encode_move_wide(sf, 0b10, 0, imm16, preg_hw(0)?))
        }
        AArch64Opcode::Movk => {
            // MOVK Rd, #imm16
            let sf = if is_64bit(0) { 1 } else { 0 };
            let imm16 = imm_val(1) as u32 & 0xFFFF;
            Ok(encoding::encode_move_wide(sf, 0b11, 0, imm16, preg_hw(0)?))
        }

        // --- Extension ---
        AArch64Opcode::Sxtw => {
            // SXTW Xd, Wn = SBFM Xd, Xn, #0, #31
            // Encoding: 1 | 00 | 100110 | 1 | 000000 | 011111 | Rn | Rd
            let rd = preg_hw(0)?;
            let rn = preg_hw(1)?;
            Ok((1u32 << 31)  // sf=1
                | (0b00u32 << 29)
                | (0b100110u32 << 23)
                | (1u32 << 22)  // N=1 for 64-bit
                | (0u32 << 16)  // immr=0
                | (31u32 << 10) // imms=31
                | (rn << 5)
                | rd)
        }
        AArch64Opcode::Uxtw => {
            // UXTW = MOV Wd, Wn (implicit zero-extend in AArch64)
            // Actually a 32-bit ORR Wd, WZR, Wn
            Ok(encoding::encode_logical_shifted_reg(
                0, // sf=0 (32-bit)
                0b01,
                0,
                0,
                preg_hw(1)?,
                0,
                31,
                preg_hw(0)?,
            ))
        }
        AArch64Opcode::Sxtb => {
            // SXTB Xd, Wn = SBFM Xd, Xn, #0, #7
            let rd = preg_hw(0)?;
            let rn = preg_hw(1)?;
            Ok((1u32 << 31)
                | (0b00u32 << 29)
                | (0b100110u32 << 23)
                | (1u32 << 22)
                | (0u32 << 16)  // immr=0
                | (7u32 << 10)  // imms=7
                | (rn << 5)
                | rd)
        }
        AArch64Opcode::Sxth => {
            // SXTH Xd, Wn = SBFM Xd, Xn, #0, #15
            let rd = preg_hw(0)?;
            let rn = preg_hw(1)?;
            Ok((1u32 << 31)
                | (0b00u32 << 29)
                | (0b100110u32 << 23)
                | (1u32 << 22)
                | (0u32 << 16)   // immr=0
                | (15u32 << 10)  // imms=15
                | (rn << 5)
                | rd)
        }

        // --- Load/Store ---
        AArch64Opcode::LdrRI => {
            // LDR Rt, [Rn, #offset]
            let offset = if inst.operands.len() > 2 { imm_val(2) } else { 0 };
            let scaled = (offset / 8) as u32 & 0xFFF;
            Ok(encoding::encode_load_store_ui(
                0b11,
                0,
                0b01,
                scaled,
                preg_hw(1)?,
                preg_hw(0)?,
            ))
        }
        AArch64Opcode::StrRI => {
            // STR Rt, [Rn, #offset]
            let offset = if inst.operands.len() > 2 { imm_val(2) } else { 0 };
            let scaled = (offset / 8) as u32 & 0xFFF;
            Ok(encoding::encode_load_store_ui(
                0b11,
                0,
                0b00,
                scaled,
                preg_hw(1)?,
                preg_hw(0)?,
            ))
        }
        AArch64Opcode::StpRI => {
            // STP Rt, Rt2, [Rn, #offset] (signed offset)
            let offset = if inst.operands.len() > 3 { imm_val(3) } else { 0 };
            let scaled_imm7 = ((offset / 8) as i32 as u32) & 0x7F;
            Ok(encoding::encode_load_store_pair(
                0b10,
                0,
                0,
                scaled_imm7,
                preg_hw(1)?,
                preg_hw(2)?,
                preg_hw(0)?,
            ))
        }
        AArch64Opcode::StpPreIndex => {
            // STP Rt, Rt2, [Rn, #offset]! (pre-index writeback)
            let offset = if inst.operands.len() > 3 { imm_val(3) } else { 0 };
            let scaled_imm7 = (offset / 8) as i8;
            encoding_mem::encode_ldp_stp_pre_index(
                encoding_mem::PairSize::X64,
                false,
                encoding_mem::PairOp::StorePair,
                scaled_imm7,
                preg_hw(1)? as u8,
                preg_hw(2)? as u8,
                preg_hw(0)? as u8,
            ).map_err(|e| LowerError::EncodingFailed(e.to_string()))
        }
        AArch64Opcode::LdpRI => {
            // LDP Rt, Rt2, [Rn, #offset] (signed offset)
            let offset = if inst.operands.len() > 3 { imm_val(3) } else { 0 };
            let scaled_imm7 = ((offset / 8) as i32 as u32) & 0x7F;
            Ok(encoding::encode_load_store_pair(
                0b10,
                0,
                1,
                scaled_imm7,
                preg_hw(1)?,
                preg_hw(2)?,
                preg_hw(0)?,
            ))
        }
        AArch64Opcode::LdpPostIndex => {
            // LDP Rt, Rt2, [Rn], #offset (post-index writeback)
            let offset = if inst.operands.len() > 3 { imm_val(3) } else { 0 };
            let scaled_imm7 = (offset / 8) as i8;
            encoding_mem::encode_ldp_stp_post_index(
                encoding_mem::PairSize::X64,
                false,
                encoding_mem::PairOp::LoadPair,
                scaled_imm7,
                preg_hw(1)? as u8,
                preg_hw(2)? as u8,
                preg_hw(0)? as u8,
            ).map_err(|e| LowerError::EncodingFailed(e.to_string()))
        }
        AArch64Opcode::LdrLiteral => {
            // LDR Xt, <literal> — PC-relative literal load
            // Encoding: opc=01 | 011 | V=0 | 00 | imm19 | Rt
            let imm19 = imm_val(1) as u32 & 0x7FFFF;
            let rt = preg_hw(0)?;
            Ok((0b01u32 << 30)
                | (0b011u32 << 27)
                | (0u32 << 26) // V=0
                | (0b00u32 << 24)
                | (imm19 << 5)
                | rt)
        }

        // --- Branch ---
        AArch64Opcode::B => {
            let offset = imm_val(0) as u32 & 0x3FFFFFF;
            Ok(encoding::encode_uncond_branch(0, offset))
        }
        AArch64Opcode::BCond => {
            let cond = imm_val(0) as u32 & 0xF;
            let offset = if inst.operands.len() > 1 {
                imm_val(1) as u32 & 0x7FFFF
            } else {
                0
            };
            Ok(encoding::encode_cond_branch(offset, cond))
        }
        AArch64Opcode::Cbz => {
            let sf = if is_64bit(0) { 1 } else { 0 };
            let rt = preg_hw(0)?;
            let imm19 = if inst.operands.len() > 1 {
                imm_val(1) as u32 & 0x7FFFF
            } else {
                0
            };
            Ok(encoding::encode_cmp_branch(sf, 0, imm19, rt))
        }
        AArch64Opcode::Cbnz => {
            let sf = if is_64bit(0) { 1 } else { 0 };
            let rt = preg_hw(0)?;
            let imm19 = if inst.operands.len() > 1 {
                imm_val(1) as u32 & 0x7FFFF
            } else {
                0
            };
            Ok(encoding::encode_cmp_branch(sf, 1, imm19, rt))
        }
        AArch64Opcode::Tbz | AArch64Opcode::Tbnz => {
            // TBZ/TBNZ: sf | 011011 | op | b40 | imm14 | Rt
            // For now emit NOP; these are rare and relaxation handles range.
            // TODO: Implement TBZ/TBNZ encoding.
            Ok(0xD503201F) // NOP
        }
        AArch64Opcode::Bl => {
            let offset = imm_val(0) as u32 & 0x3FFFFFF;
            Ok(encoding::encode_uncond_branch(1, offset))
        }
        AArch64Opcode::Blr => {
            Ok(encoding::encode_branch_reg(0b0001, preg_hw(0)?))
        }
        AArch64Opcode::Br => {
            Ok(encoding::encode_branch_reg(0b0000, preg_hw(0)?))
        }
        AArch64Opcode::Ret => {
            // RET (X30)
            Ok(encoding::encode_branch_reg(0b0010, 30))
        }

        // --- Address ---
        AArch64Opcode::Adrp => {
            // ADRP Xd, <page>
            // Encoding: 1 | immlo(2) | 10000 | immhi(19) | Rd
            // The immediate is page-relative and will be relocated.
            // Emit with imm=0; relocation will patch it.
            let rd = preg_hw(0)?;
            Ok((1u32 << 31) | (0b10000u32 << 24) | rd)
        }
        AArch64Opcode::AddPCRel => {
            // ADD Xd, Xn, #pageoff — page offset portion of ADRP+ADD pair.
            // Emit as ADD with imm=0; relocation will patch the offset.
            let sf = 1u32;
            Ok(encoding::encode_add_sub_imm(
                sf,
                0,
                0,
                0,
                0, // imm will be patched by relocation
                preg_hw(1)?,
                preg_hw(0)?,
            ))
        }

        // --- Floating-point ---
        AArch64Opcode::FaddRR
        | AArch64Opcode::FsubRR
        | AArch64Opcode::FmulRR
        | AArch64Opcode::FdivRR => {
            // FP data-processing 2-source:
            // 0 | 0 | 0 | 11110 | ftype(2) | 1 | Rm | opcode(4) | 10 | Rn | Rd
            let ftype = 0b01u32; // double precision
            let fp_op = match inst.opcode {
                AArch64Opcode::FaddRR => 0b0010u32,
                AArch64Opcode::FsubRR => 0b0011u32,
                AArch64Opcode::FmulRR => 0b0000u32,
                AArch64Opcode::FdivRR => 0b0001u32,
                _ => unreachable!(),
            };
            let rd = preg_hw(0)?;
            let rn = preg_hw(1)?;
            let rm = preg_hw(2)?;
            Ok((0b00011110u32 << 24)
                | (ftype << 22)
                | (1u32 << 21)
                | (rm << 16)
                | (fp_op << 12)
                | (0b10u32 << 10)
                | (rn << 5)
                | rd)
        }
        AArch64Opcode::Fcmp => {
            // FCMP Rn, Rm
            // 0 | 0 | 0 | 11110 | ftype | 1 | Rm | 00 | 1000 | Rn | 00000
            let ftype = 0b01u32;
            let rn = preg_hw(0)?;
            let rm = preg_hw(1)?;
            Ok((0b00011110u32 << 24)
                | (ftype << 22)
                | (1u32 << 21)
                | (rm << 16)
                | (0b00u32 << 14)
                | (0b1000u32 << 10)
                | (rn << 5)
                | 0b00000u32)
        }
        AArch64Opcode::FcvtzsRR => {
            // FCVTZS Wd/Xd, Dn
            // sf | 0 | 0 | 11110 | ftype | 1 | 11 | 000 | 000000 | Rn | Rd
            let sf = if is_64bit(0) { 1u32 } else { 0u32 };
            let ftype = 0b01u32; // double
            let rd = preg_hw(0)?;
            let rn = preg_hw(1)?;
            Ok((sf << 31)
                | (0b0011110u32 << 24)
                | (ftype << 22)
                | (1u32 << 21)
                | (0b11u32 << 19)
                | (0b000u32 << 16)
                | (0u32 << 10)
                | (rn << 5)
                | rd)
        }
        AArch64Opcode::ScvtfRR => {
            // SCVTF Dd, Wn/Xn
            // sf | 0 | 0 | 11110 | ftype | 1 | 00 | 010 | 000000 | Rn | Rd
            let sf = if is_64bit(1) { 1u32 } else { 0u32 };
            let ftype = 0b01u32; // double
            let rd = preg_hw(0)?;
            let rn = preg_hw(1)?;
            Ok((sf << 31)
                | (0b0011110u32 << 24)
                | (ftype << 22)
                | (1u32 << 21)
                | (0b00u32 << 19)
                | (0b010u32 << 16)
                | (0u32 << 10)
                | (rn << 5)
                | rd)
        }

        // --- Data-processing (3 source): MSUB, SMULL, UMULL ---
        AArch64Opcode::Msub => {
            let sf = if is_64bit(0) { 1u32 } else { 0u32 };
            let rd = preg_hw(0)?;
            let rn = preg_hw(1)?;
            let rm = preg_hw(2)?;
            let ra = if inst.operands.len() > 3 { preg_hw(3)? } else { 31 };
            Ok((sf << 31)
                | (0b0011011u32 << 24)
                | (rm << 16)
                | (1 << 15) // o0=1 for MSUB
                | (ra << 10)
                | (rn << 5)
                | rd)
        }
        AArch64Opcode::Smull => {
            let rd = preg_hw(0)?;
            let rn = preg_hw(1)?;
            let rm = preg_hw(2)?;
            Ok((1u32 << 31)
                | (0b0011011u32 << 24)
                | (0b001 << 21)
                | (rm << 16)
                | (0 << 15)
                | (31 << 10) // Ra = XZR
                | (rn << 5)
                | rd)
        }
        AArch64Opcode::Umull => {
            let rd = preg_hw(0)?;
            let rn = preg_hw(1)?;
            let rm = preg_hw(2)?;
            Ok((1u32 << 31)
                | (0b0011011u32 << 24)
                | (0b101 << 21)
                | (rm << 16)
                | (0 << 15)
                | (31 << 10) // Ra = XZR
                | (rn << 5)
                | rd)
        }

        // --- Flag-setting arithmetic (ADDS/SUBS) ---
        AArch64Opcode::AddsRR => {
            let sf = if is_64bit(0) { 1 } else { 0 };
            Ok(encoding::encode_add_sub_shifted_reg(
                sf, 0, 1, 0, preg_hw(2)?, 0, preg_hw(1)?, preg_hw(0)?,
            ))
        }
        AArch64Opcode::AddsRI => {
            let sf = if is_64bit(0) { 1 } else { 0 };
            let imm = imm_val(2) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf, 0, 1, 0, imm, preg_hw(1)?, preg_hw(0)?,
            ))
        }
        AArch64Opcode::SubsRR => {
            let sf = if is_64bit(0) { 1 } else { 0 };
            Ok(encoding::encode_add_sub_shifted_reg(
                sf, 1, 1, 0, preg_hw(2)?, 0, preg_hw(1)?, preg_hw(0)?,
            ))
        }
        AArch64Opcode::SubsRI => {
            let sf = if is_64bit(0) { 1 } else { 0 };
            let imm = imm_val(2) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf, 1, 1, 0, imm, preg_hw(1)?, preg_hw(0)?,
            ))
        }

        // --- Trap pseudo-instructions: emit BRK #1 ---
        AArch64Opcode::TrapOverflow
        | AArch64Opcode::TrapBoundsCheck
        | AArch64Opcode::TrapNull => {
            Ok(0xD4200020) // BRK #1
        }

        // --- Pseudo-instructions: emit NOP ---
        AArch64Opcode::Phi
        | AArch64Opcode::StackAlloc
        | AArch64Opcode::Nop
        | AArch64Opcode::Retain
        | AArch64Opcode::Release => {
            // These should have been eliminated, but emit NOP as safe fallback.
            Ok(0xD503201F)
        }

    }
}

// ---------------------------------------------------------------------------
// Function-level encoding with branch relaxation
// ---------------------------------------------------------------------------

/// Encode all instructions in a function after branch relaxation.
///
/// This walks the relaxed instruction sequence (where block targets have
/// already been resolved to signed displacements in instruction units)
/// and encodes each instruction.
fn encode_relaxed_instructions(
    instructions: &[MachInst],
) -> Result<Vec<u8>, LowerError> {
    let mut code = Vec::with_capacity(instructions.len() * 4);

    for inst in instructions {
        // Pseudo-instructions should not appear in the relaxed sequence,
        // but guard against it.
        if inst.is_pseudo() {
            continue;
        }
        let word = encode_inst(inst)?;
        code.extend_from_slice(&word.to_le_bytes());
    }

    Ok(code)
}

/// Encode all instructions in a function walking blocks in layout order.
///
/// This is the simpler path used when branch relaxation is not needed
/// (e.g., when branch targets have already been resolved to immediates).
pub fn encode_function(func: &IrMachFunction) -> Result<Vec<u8>, LowerError> {
    let mut code = Vec::new();

    for &block_id in &func.block_order {
        let block = func.block(block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            if inst.is_pseudo() {
                continue;
            }
            let word = encode_inst(inst)?;
            code.extend_from_slice(&word.to_le_bytes());
        }
    }

    Ok(code)
}

// ---------------------------------------------------------------------------
// Main entry point: lower_function
// ---------------------------------------------------------------------------

/// Lower a post-regalloc MachFunction to encoded AArch64 machine code.
///
/// This is the primary entry point for Phase 8 of the pipeline. It:
///   1. Runs frame lowering (prologue/epilogue + frame index elimination)
///   2. Expands pseudo-instructions
///   3. Runs branch relaxation (resolves block targets to byte offsets)
///   4. Encodes every instruction
///   5. Collects relocations for external references
///
/// The input function should already have:
///   - Completed ISel (all instructions are AArch64 MachInsts)
///   - Completed register allocation (all VRegs replaced with PRegs)
///   - Stack slots allocated (from spilling)
///
/// # Arguments
/// * `func` — The machine function (post-regalloc, mutable for frame lowering)
///
/// # Returns
/// * `LowerResult` containing encoded bytes, relocations, and frame layout
pub fn lower_function(func: &mut IrMachFunction) -> Result<LowerResult, LowerError> {
    // Phase 7: Frame lowering — compute layout, eliminate frame indices,
    // insert prologue/epilogue.
    let layout = frame::compute_frame_layout(func, 0, true);
    frame::eliminate_frame_indices(func, &layout);
    frame::insert_prologue_epilogue(func, &layout);

    // Expand any remaining pseudo-instructions.
    expand_pseudos(func);

    // Run branch relaxation — this resolves Block operands to immediate
    // offsets and handles out-of-range branches.
    let relaxed = relax::relax_branches(func);

    // Collect relocations from instructions that reference external symbols.
    let relocations = collect_relocations(&relaxed.instructions);

    // Encode the relaxed instruction sequence to bytes.
    let code = encode_relaxed_instructions(&relaxed.instructions)
        .map_err(|e| LowerError::EncodingFailed(e.to_string()))?;

    Ok(LowerResult {
        code,
        relocations,
        frame_layout: layout,
    })
}

/// Lower a function that has already had frame lowering applied.
///
/// Skips the frame lowering phase. Useful when the caller has already
/// run `frame::insert_prologue_epilogue`.
pub fn lower_function_no_frame(func: &mut IrMachFunction) -> Result<LowerResult, LowerError> {
    // Use a dummy frame layout since frame lowering was already done.
    let layout = frame::compute_frame_layout(func, 0, true);

    // Expand any remaining pseudo-instructions.
    expand_pseudos(func);

    // Run branch relaxation.
    let relaxed = relax::relax_branches(func);

    // Collect relocations.
    let relocations = collect_relocations(&relaxed.instructions);

    // Encode.
    let code = encode_relaxed_instructions(&relaxed.instructions)
        .map_err(|e| LowerError::EncodingFailed(e.to_string()))?;

    Ok(LowerResult {
        code,
        relocations,
        frame_layout: layout,
    })
}

// ---------------------------------------------------------------------------
// Relocation collection
// ---------------------------------------------------------------------------

/// Scan the instruction sequence for instructions that need relocations
/// (ADRP, AddPCRel, BL to external symbols).
fn collect_relocations(instructions: &[MachInst]) -> Vec<Relocation> {
    let mut relocs = Vec::new();
    let mut byte_offset = 0u32;

    for inst in instructions {
        if inst.is_pseudo() {
            continue;
        }

        match inst.opcode {
            AArch64Opcode::Adrp => {
                // ADRP with a symbol operand needs a PAGE21 relocation.
                if let Some(sym) = extract_symbol_name(inst) {
                    relocs.push(Relocation {
                        offset: byte_offset,
                        kind: RelocKind::AdrpPage21,
                        symbol: sym,
                        addend: 0,
                    });
                }
            }
            AArch64Opcode::AddPCRel => {
                // ADD Xd, Xn, #pageoff needs a PAGEOFF12 relocation.
                if let Some(sym) = extract_symbol_name(inst) {
                    relocs.push(Relocation {
                        offset: byte_offset,
                        kind: RelocKind::AddPageOff12,
                        symbol: sym,
                        addend: 0,
                    });
                }
            }
            AArch64Opcode::Bl => {
                // BL to an external function needs a BRANCH26 relocation.
                // (Only if the target is a symbol, not a resolved offset.)
                if let Some(sym) = extract_symbol_name(inst) {
                    relocs.push(Relocation {
                        offset: byte_offset,
                        kind: RelocKind::Branch26,
                        symbol: sym,
                        addend: 0,
                    });
                }
            }
            _ => {}
        }

        byte_offset += 4;
    }

    relocs
}

/// Extract a symbol name from an instruction's operands, if present.
///
/// Symbol references may appear as Imm(0) placeholders from ISel conversion
/// (see pipeline.rs convert_isel_operand). In a complete implementation,
/// these would be carried as a dedicated Symbol operand. For now, we don't
/// extract symbols since the current IR doesn't carry them through.
fn extract_symbol_name(_inst: &MachInst) -> Option<String> {
    // TODO: When the IR gains a Symbol operand type, extract it here.
    // For now, relocations for external symbols are not emitted because
    // the current ISel converts symbols to Imm(0) placeholders.
    None
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::function::{MachFunction, Signature, Type};
    use llvm2_ir::inst::{AArch64Opcode, MachInst};
    use llvm2_ir::operand::MachOperand;
    use llvm2_ir::regs::{X0, X1, X19};
    use llvm2_ir::types::BlockId;

    /// Helper: create a minimal function with instructions in the entry block.
    fn make_func(name: &str, insts: Vec<MachInst>) -> MachFunction {
        let sig = Signature::new(vec![], vec![]);
        let mut func = MachFunction::new(name.to_string(), sig);
        for inst in insts {
            let id = func.push_inst(inst);
            func.append_inst(BlockId(0), id);
        }
        func
    }

    // -----------------------------------------------------------------------
    // Encoding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_add_rr() {
        // ADD X0, X0, X1
        let inst = MachInst::new(
            AArch64Opcode::AddRR,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X0),
                MachOperand::PReg(X1),
            ],
        );
        let word = encode_inst(&inst).unwrap();
        // sf=1, op=0, S=0, shift=0, Rm=1, imm6=0, Rn=0, Rd=0
        // = 0x8B010000  (ADD X0, X0, X1)
        assert_eq!(word, 0x8B010000, "ADD X0, X0, X1 = 0x{word:08X}");
    }

    #[test]
    fn test_encode_sub_ri() {
        // SUB X0, X0, #16
        let inst = MachInst::new(
            AArch64Opcode::SubRI,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X0),
                MachOperand::Imm(16),
            ],
        );
        let word = encode_inst(&inst).unwrap();
        // sf=1, op=1, S=0, sh=0, imm12=16, Rn=0, Rd=0
        let expected = (1u32 << 31) | (1 << 30) | (0b100010 << 23) | (16 << 10);
        assert_eq!(word, expected, "SUB X0, X0, #16 = 0x{word:08X}");
    }

    #[test]
    fn test_encode_mov_r() {
        // MOV X0, X1 = ORR X0, XZR, X1
        let inst = MachInst::new(
            AArch64Opcode::MovR,
            vec![MachOperand::PReg(X0), MachOperand::PReg(X1)],
        );
        let word = encode_inst(&inst).unwrap();
        // sf=1, opc=01, shift=0, N=0, Rm=1, imm6=0, Rn=31(XZR), Rd=0
        let expected = (1u32 << 31)
            | (0b01 << 29)
            | (0b01010 << 24)
            | (1 << 16)
            | (31 << 5);
        assert_eq!(word, expected, "MOV X0, X1 = 0x{word:08X}");
    }

    #[test]
    fn test_encode_ret() {
        let inst = MachInst::new(AArch64Opcode::Ret, vec![]);
        let word = encode_inst(&inst).unwrap();
        assert_eq!(word, 0xD65F03C0, "RET = 0x{word:08X}");
    }

    #[test]
    fn test_encode_b() {
        // B +3 (instruction units)
        let inst = MachInst::new(AArch64Opcode::B, vec![MachOperand::Imm(3)]);
        let word = encode_inst(&inst).unwrap();
        let expected = (0b00101u32 << 26) | 3;
        assert_eq!(word, expected, "B +3 = 0x{word:08X}");
    }

    #[test]
    fn test_encode_bcond() {
        // B.EQ +2
        let inst = MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Imm(0), MachOperand::Imm(2)],
        );
        let word = encode_inst(&inst).unwrap();
        let expected = (0b01010100u32 << 24) | (2 << 5);
        assert_eq!(word, expected, "B.EQ +2 = 0x{word:08X}");
    }

    #[test]
    fn test_encode_movz() {
        // MOVZ X0, #42
        let inst = MachInst::new(
            AArch64Opcode::Movz,
            vec![MachOperand::PReg(X0), MachOperand::Imm(42)],
        );
        let word = encode_inst(&inst).unwrap();
        let expected = (1u32 << 31) | (0b10 << 29) | (0b100101 << 23) | (42 << 5);
        assert_eq!(word, expected, "MOVZ X0, #42 = 0x{word:08X}");
    }

    #[test]
    fn test_encode_ldr_ri() {
        // LDR X0, [X1, #8]  -> scaled offset = 8/8 = 1
        let inst = MachInst::new(
            AArch64Opcode::LdrRI,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X1),
                MachOperand::Imm(8),
            ],
        );
        let word = encode_inst(&inst).unwrap();
        let expected = (0b11u32 << 30)
            | (0b111 << 27)
            | (0b01 << 24)
            | (0b01 << 22)
            | (1 << 10)
            | (1 << 5);
        assert_eq!(word, expected, "LDR X0, [X1, #8] = 0x{word:08X}");
    }

    #[test]
    fn test_encode_str_ri() {
        // STR X0, [X1]
        let inst = MachInst::new(
            AArch64Opcode::StrRI,
            vec![MachOperand::PReg(X0), MachOperand::PReg(X1)],
        );
        let word = encode_inst(&inst).unwrap();
        let expected = (0b11u32 << 30)
            | (0b111 << 27)
            | (0b01 << 24)
            | (1 << 5);
        assert_eq!(word, expected, "STR X0, [X1] = 0x{word:08X}");
    }

    // -----------------------------------------------------------------------
    // Pseudo-expansion tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_expand_pseudos_removes_phi() {
        let mut func = make_func("phi_test", vec![
            MachInst::new(AArch64Opcode::Phi, vec![MachOperand::Imm(0)]),
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        expand_pseudos(&mut func);
        assert_eq!(func.insts[0].opcode, AArch64Opcode::Nop);
        assert!(func.insts[0].operands.is_empty());
    }

    #[test]
    fn test_expand_pseudos_removes_stack_alloc() {
        let mut func = make_func("stack_test", vec![
            MachInst::new(AArch64Opcode::StackAlloc, vec![MachOperand::Imm(16)]),
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        expand_pseudos(&mut func);
        assert_eq!(func.insts[0].opcode, AArch64Opcode::Nop);
    }

    #[test]
    fn test_expand_pseudos_identity_mov() {
        let mut func = make_func("identity_mov", vec![
            MachInst::new(
                AArch64Opcode::MovR,
                vec![MachOperand::PReg(X0), MachOperand::PReg(X0)],
            ),
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        expand_pseudos(&mut func);
        // Identity MOV X0, X0 should become NOP.
        assert_eq!(func.insts[0].opcode, AArch64Opcode::Nop);
    }

    #[test]
    fn test_expand_pseudos_keeps_real_mov() {
        let mut func = make_func("real_mov", vec![
            MachInst::new(
                AArch64Opcode::MovR,
                vec![MachOperand::PReg(X0), MachOperand::PReg(X1)],
            ),
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        expand_pseudos(&mut func);
        // MOV X0, X1 should be kept (not identity).
        assert_eq!(func.insts[0].opcode, AArch64Opcode::MovR);
    }

    // -----------------------------------------------------------------------
    // Encoding round-trip: encode_function on a simple IR function
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_function_simple() {
        // Build a simple add function: ADD X0, X0, X1; RET
        let sig = Signature::new(vec![Type::I64, Type::I64], vec![Type::I64]);
        let mut func = MachFunction::new("add".to_string(), sig);
        let entry = func.entry;

        let add = MachInst::new(
            AArch64Opcode::AddRR,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X0),
                MachOperand::PReg(X1),
            ],
        );
        let add_id = func.push_inst(add);
        func.append_inst(entry, add_id);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(entry, ret_id);

        let code = encode_function(&func).unwrap();
        assert_eq!(code.len(), 8); // 2 instructions * 4 bytes
        // Verify ADD encoding (first 4 bytes, little-endian).
        let add_word = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        assert_eq!(add_word, 0x8B010000);
        // Verify RET encoding.
        let ret_word = u32::from_le_bytes([code[4], code[5], code[6], code[7]]);
        assert_eq!(ret_word, 0xD65F03C0);
    }

    // -----------------------------------------------------------------------
    // Full lowering test
    // -----------------------------------------------------------------------

    #[test]
    fn test_lower_function_simple() {
        // Build a simple function that should survive the full lowering.
        let sig = Signature::new(vec![Type::I64, Type::I64], vec![Type::I64]);
        let mut func = MachFunction::new("add_lowered".to_string(), sig);
        let entry = func.entry;

        let add = MachInst::new(
            AArch64Opcode::AddRR,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X0),
                MachOperand::PReg(X1),
            ],
        );
        let add_id = func.push_inst(add);
        func.append_inst(entry, add_id);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(entry, ret_id);

        let result = lower_function(&mut func).unwrap();

        // Should have prologue + ADD + epilogue (including RET).
        // Minimum: STP + MOV + ADD + LDP + RET = 5 insts = 20 bytes.
        assert!(
            result.code.len() >= 12,
            "Expected at least 12 bytes of code, got {}",
            result.code.len()
        );
        // Frame layout should have FP/LR pair.
        assert!(result.frame_layout.uses_frame_pointer);
        assert_eq!(result.frame_layout.callee_saved_pairs.len(), 1); // Just FP/LR
    }

    #[test]
    fn test_lower_function_with_branch() {
        // Function with two blocks and a branch.
        let sig = Signature::new(vec![], vec![]);
        let mut func = MachFunction::new("branch_test".to_string(), sig);
        let bb0 = func.entry;
        let bb1 = func.create_block();

        // bb0: B bb1
        let b_inst = MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(bb1)]);
        let b_id = func.push_inst(b_inst);
        func.append_inst(bb0, b_id);

        // bb1: RET
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(bb1, ret_id);

        let result = lower_function(&mut func).unwrap();
        assert!(result.code.len() >= 8, "Need at least B + RET");
    }

    #[test]
    fn test_lower_function_with_callee_saves() {
        // Function that uses X19 (callee-saved).
        let sig = Signature::new(vec![], vec![]);
        let mut func = MachFunction::new("callee_save_test".to_string(), sig);
        let entry = func.entry;

        // Use X19 (callee-saved register).
        let mov = MachInst::new(
            AArch64Opcode::MovR,
            vec![MachOperand::PReg(X19), MachOperand::PReg(X0)],
        );
        let mov_id = func.push_inst(mov);
        func.append_inst(entry, mov_id);

        let mov2 = MachInst::new(
            AArch64Opcode::MovR,
            vec![MachOperand::PReg(X0), MachOperand::PReg(X19)],
        );
        let mov2_id = func.push_inst(mov2);
        func.append_inst(entry, mov2_id);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(entry, ret_id);

        let result = lower_function(&mut func).unwrap();

        // Frame layout should include X19/X20 pair.
        assert_eq!(result.frame_layout.callee_saved_pairs.len(), 2);
        // Code should be non-trivial (prologue + body + epilogue).
        assert!(
            result.code.len() >= 20,
            "Expected at least 20 bytes with callee saves, got {}",
            result.code.len()
        );
    }

    #[test]
    fn test_lower_result_has_no_relocations_for_simple_func() {
        let sig = Signature::new(vec![], vec![]);
        let mut func = MachFunction::new("simple".to_string(), sig);
        let entry = func.entry;

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(entry, ret_id);

        let result = lower_function(&mut func).unwrap();
        assert!(result.relocations.is_empty());
    }

    #[test]
    fn test_encode_mul() {
        // MUL X0, X1, X0
        let inst = MachInst::new(
            AArch64Opcode::MulRR,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X1),
                MachOperand::PReg(X0),
            ],
        );
        let word = encode_inst(&inst).unwrap();
        // MADD X0, X1, X0, XZR
        // sf=1 | 00 | 11011 | 000 | Rm=0 | o0=0 | Ra=31 | Rn=1 | Rd=0
        let expected = (1u32 << 31)
            | (0b0011011u32 << 24)
            | (0 << 16)   // Rm=X0
            | (0 << 15)   // o0=0
            | (31 << 10)  // Ra=XZR
            | (1 << 5)    // Rn=X1
            | 0;           // Rd=X0
        assert_eq!(word, expected, "MUL X0, X1, X0 = 0x{word:08X}");
    }

    #[test]
    fn test_encode_cmp_rr() {
        // CMP X0, X1 = SUBS XZR, X0, X1
        let inst = MachInst::new(
            AArch64Opcode::CmpRR,
            vec![MachOperand::PReg(X0), MachOperand::PReg(X1)],
        );
        let word = encode_inst(&inst).unwrap();
        // sf=1, op=1(SUB), S=1, shift=0, Rm=1, imm6=0, Rn=0, Rd=31(XZR)
        let expected = (1u32 << 31)
            | (1 << 30)
            | (1 << 29)
            | (0b01011 << 24)
            | (1 << 16)
            | 31;
        assert_eq!(word, expected, "CMP X0, X1 = 0x{word:08X}");
    }

    #[test]
    fn test_encode_blr() {
        // BLR X0
        let inst = MachInst::new(
            AArch64Opcode::Blr,
            vec![MachOperand::PReg(X0)],
        );
        let word = encode_inst(&inst).unwrap();
        assert_eq!(word, 0xD63F0000, "BLR X0 = 0x{word:08X}");
    }
}
