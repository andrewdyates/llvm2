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

use crate::aarch64::encode::{encode_instruction, EncodeError};
use crate::aarch64::encoding;
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
    #[error("branch relaxation failed: {0}")]
    RelaxationFailed(#[from] relax::RelaxError),
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
            // Trap pseudo-instructions survive into lowering intentionally.
            // They are encoded as BRK #1 by the encoder — do not convert to NOP.
            AArch64Opcode::TrapOverflow
            | AArch64Opcode::TrapBoundsCheck
            | AArch64Opcode::TrapNull
            | AArch64Opcode::TrapDivZero
            | AArch64Opcode::TrapShiftRange => {
                // Leave as-is; the encoder handles these directly.
            }
            // Reference counting pseudo-instructions should have been lowered
            // to actual call sequences before reaching this point. Leave as-is
            // for the encoder, which emits NOP (they are effectively eliminated).
            AArch64Opcode::Retain | AArch64Opcode::Release => {
                // Leave as-is; the encoder handles these directly.
            }
            other => {
                // Unknown pseudo-instruction — this is a bug. An unrecognized
                // pseudo reaching expansion means either ISel emitted something
                // we don't handle, or an earlier pass failed to lower it.
                // Log a warning with the opcode for debugging.
                eprintln!(
                    "WARNING: unrecognized pseudo-instruction {:?} in expand_pseudos, \
                     converting to NOP. This may indicate a missing expansion rule.",
                    other
                );
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
/// Delegates to the unified encoder (`encode_instruction` from `aarch64::encode`)
/// for all opcodes it supports. Handles pipeline-specific opcodes locally:
///
/// - **Adrp / AddPCRel**: Emit with imm=0; relocations will patch the offset.
///   The unified encoder uses the actual immediate, which is wrong for the
///   pipeline's relocation-based patching model.
/// - **BicRR, Csinc, Csinv, Csneg, Movn**: Not yet implemented in the unified
///   encoder (encode.rs lists them as UnsupportedOpcode). Implemented here
///   until encode.rs catches up.
/// - **LdrRO, StrRO**: Integer-only register-offset encoding. The unified
///   encoder handles FPR paths via encoding_mem; this uses inline encoding
///   for integer loads/stores which matches the pipeline's needs.
/// - **LdrGot, LdrTlvp**: Always 64-bit GOT/TLV loads, pipeline-specific.
///
/// Pre-validates operands (#98, #105) before delegation to catch malformed
/// instructions early rather than silently encoding wrong values.
fn encode_inst(inst: &MachInst) -> Result<u32, LowerError> {
    // Helper: extract physical register hardware encoding from operand.
    // Returns an error for non-register operands instead of silently
    // defaulting to XZR (reg 31), which would produce wrong code (#98).
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
            other => Err(LowerError::EncodingFailed(format!(
                "expected register operand at index {} for {:?}, got {:?}",
                idx, inst.opcode, other
            ))),
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

    // Helper: determine AArch64 ftype field from an FPR operand's register class.
    // Returns 0b00 for single (S-regs), 0b01 for double (D-regs).
    // Returns an error if the operand is not an FPR, instead of silently
    // defaulting to Double precision which masks encoding bugs (#105).
    let fp_size = |idx: usize| -> Result<u32, LowerError> {
        match inst.operands.get(idx) {
            Some(MachOperand::PReg(p)) => {
                let class = llvm2_ir::regs::preg_class(*p);
                match class {
                    llvm2_ir::regs::RegClass::Fpr32 => Ok(0b00),  // single
                    llvm2_ir::regs::RegClass::Fpr64 => Ok(0b01),  // double
                    llvm2_ir::regs::RegClass::Fpr128 => Ok(0b01), // treat as double for data-processing
                    _ => Err(LowerError::EncodingFailed(format!(
                        "expected FPR operand at index {} for {:?}, got GPR/other class {:?}",
                        idx, inst.opcode, class
                    ))),
                }
            }
            Some(other) => Err(LowerError::EncodingFailed(format!(
                "expected FPR register operand at index {} for {:?}, got {:?}",
                idx, inst.opcode, other
            ))),
            None => Err(LowerError::MissingOperand {
                opcode: inst.opcode,
                index: idx,
            }),
        }
    };

    match inst.opcode {
        // =================================================================
        // Pipeline-specific: ADRP and AddPCRel emit with imm=0 for
        // relocation patching. The unified encoder uses the actual
        // immediate, which doesn't work in the relocation model.
        // =================================================================
        AArch64Opcode::Adrp => {
            // ADRP Xd, <page>
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

        // =================================================================
        // Opcodes not yet in the unified encoder (encode.rs returns
        // UnsupportedOpcode for these). Keep local implementations
        // until encode.rs is updated.
        // =================================================================
        AArch64Opcode::BicRR => {
            // BIC Rd, Rn, Rm = AND Rd, Rn, NOT(Rm)
            let sf = if is_64bit(0) { 1 } else { 0 };
            Ok(encoding::encode_logical_shifted_reg(
                sf, 0b00, 1, 0, preg_hw(2)?, 0, preg_hw(1)?, preg_hw(0)?,
            ))
        }
        AArch64Opcode::Csinc => {
            // CSINC Xd, Xn, Xm, cond
            let sf = if is_64bit(0) { 1u32 } else { 0u32 };
            let rd = preg_hw(0)?;
            let rn = preg_hw(1)?;
            let rm = preg_hw(2)?;
            let cond = imm_val(3) as u32 & 0xF;
            Ok((sf << 31)
                | (0b00 << 29)
                | (0b11010100 << 21)
                | (rm << 16)
                | (cond << 12)
                | (0b01 << 10)
                | (rn << 5)
                | rd)
        }
        AArch64Opcode::Csinv => {
            // CSINV Xd, Xn, Xm, cond
            let sf = if is_64bit(0) { 1u32 } else { 0u32 };
            let rd = preg_hw(0)?;
            let rn = preg_hw(1)?;
            let rm = preg_hw(2)?;
            let cond = imm_val(3) as u32 & 0xF;
            Ok((sf << 31)
                | (0b10 << 29)
                | (0b11010100 << 21)
                | (rm << 16)
                | (cond << 12)
                | (0b00 << 10)
                | (rn << 5)
                | rd)
        }
        AArch64Opcode::Csneg => {
            // CSNEG Xd, Xn, Xm, cond
            let sf = if is_64bit(0) { 1u32 } else { 0u32 };
            let rd = preg_hw(0)?;
            let rn = preg_hw(1)?;
            let rm = preg_hw(2)?;
            let cond = imm_val(3) as u32 & 0xF;
            Ok((sf << 31)
                | (0b10 << 29)
                | (0b11010100 << 21)
                | (rm << 16)
                | (cond << 12)
                | (0b01 << 10)
                | (rn << 5)
                | rd)
        }
        AArch64Opcode::Movn => {
            // MOVN Rd, #imm16
            let sf = if is_64bit(0) { 1 } else { 0 };
            let imm16 = imm_val(1) as u32 & 0xFFFF;
            Ok(encoding::encode_move_wide(sf, 0b00, 0, imm16, preg_hw(0)?))
        }

        // =================================================================
        // Pipeline-specific: integer-only register-offset loads/stores.
        // The unified encoder handles FPR paths differently via encoding_mem.
        // Keep local for integer pipeline consistency.
        // =================================================================
        AArch64Opcode::LdrRO => {
            let sf = if is_64bit(0) { 1u32 } else { 0u32 };
            let rt = preg_hw(0)?;
            let rn = preg_hw(1)?;
            let rm = preg_hw(2)?;
            let size = if sf == 1 { 0b11u32 } else { 0b10u32 };
            let (option, s) = if inst.operands.len() > 3 {
                let packed = imm_val(3) as u32;
                ((packed >> 1) & 0b111, packed & 1)
            } else {
                (0b011u32, 0u32)
            };
            Ok((size << 30)
                | (0b111 << 27)
                | (0 << 26)
                | (0b00 << 24)
                | (0b01 << 22)
                | (1 << 21)
                | (rm << 16)
                | (option << 13)
                | (s << 12)
                | (0b10 << 10)
                | (rn << 5)
                | rt)
        }
        AArch64Opcode::StrRO => {
            let sf = if is_64bit(0) { 1u32 } else { 0u32 };
            let rt = preg_hw(0)?;
            let rn = preg_hw(1)?;
            let rm = preg_hw(2)?;
            let size = if sf == 1 { 0b11u32 } else { 0b10u32 };
            let (option, s) = if inst.operands.len() > 3 {
                let packed = imm_val(3) as u32;
                ((packed >> 1) & 0b111, packed & 1)
            } else {
                (0b011u32, 0u32)
            };
            Ok((size << 30)
                | (0b111 << 27)
                | (0 << 26)
                | (0b00 << 24)
                | (0b00 << 22)
                | (1 << 21)
                | (rm << 16)
                | (option << 13)
                | (s << 12)
                | (0b10 << 10)
                | (rn << 5)
                | rt)
        }

        // =================================================================
        // Pipeline-specific: GOT/TLV loads always use 64-bit unsigned offset.
        // =================================================================
        AArch64Opcode::LdrGot => {
            let rd = preg_hw(0)?;
            let rn = preg_hw(1)?;
            let offset = if inst.operands.len() > 2 { imm_val(2) } else { 0 };
            let scaled = (offset / 8) as u32 & 0xFFF;
            Ok(encoding::encode_load_store_ui(0b11, 0, 0b01, scaled, rn, rd))
        }
        AArch64Opcode::LdrTlvp => {
            let rd = preg_hw(0)?;
            let rn = preg_hw(1)?;
            let offset = if inst.operands.len() > 2 { imm_val(2) } else { 0 };
            let scaled = (offset / 8) as u32 & 0xFFF;
            Ok(encoding::encode_load_store_ui(0b11, 0, 0b01, scaled, rn, rd))
        }

        // =================================================================
        // FP instructions: pre-validate operand types (#105) then delegate.
        // The unified encoder silently defaults non-FPR operands to Double
        // precision, masking bugs. We validate first to preserve stricter
        // error checking, then delegate to encode_instruction.
        // =================================================================
        AArch64Opcode::FaddRR
        | AArch64Opcode::FsubRR
        | AArch64Opcode::FmulRR
        | AArch64Opcode::FdivRR => {
            // Validate that operand 0 is an FPR before delegating.
            fp_size(0)?;
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::FnegRR => {
            fp_size(0)?;
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::Fcmp => {
            fp_size(0)?;
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::FcvtzsRR | AArch64Opcode::FcvtzuRR => {
            // Source is FPR (operand 1)
            fp_size(1)?;
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::ScvtfRR | AArch64Opcode::UcvtfRR => {
            // Destination is FPR (operand 0)
            fp_size(0)?;
            encode_instruction(inst).map_err(map_encode_error)
        }

        // =================================================================
        // Integer instructions with register operands: pre-validate that
        // register operands are actual registers (#98), then delegate.
        // The unified encoder defaults non-register operands to XZR (31),
        // which silently produces wrong code.
        // =================================================================
        AArch64Opcode::AddRR | AArch64Opcode::SubRR => {
            preg_hw(0)?; preg_hw(1)?; preg_hw(2)?;
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::MulRR | AArch64Opcode::SDiv | AArch64Opcode::UDiv => {
            preg_hw(0)?; preg_hw(1)?; preg_hw(2)?;
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::AndRR | AArch64Opcode::OrrRR | AArch64Opcode::EorRR
        | AArch64Opcode::OrnRR => {
            preg_hw(0)?; preg_hw(1)?; preg_hw(2)?;
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::LslRR | AArch64Opcode::LsrRR | AArch64Opcode::AsrRR => {
            preg_hw(0)?; preg_hw(1)?; preg_hw(2)?;
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::Neg => {
            preg_hw(0)?; preg_hw(1)?;
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::CmpRR | AArch64Opcode::Tst => {
            preg_hw(0)?; preg_hw(1)?;
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::MovR => {
            preg_hw(0)?;
            // Operand 1 may be Special(SP) — that's fine, preg_hw handles it
            if inst.operands.len() >= 2 { preg_hw(1)?; }
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::Sxtw | AArch64Opcode::Uxtw
        | AArch64Opcode::Sxtb | AArch64Opcode::Sxth => {
            preg_hw(0)?; preg_hw(1)?;
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::AddsRR | AArch64Opcode::SubsRR => {
            preg_hw(0)?; preg_hw(1)?; preg_hw(2)?;
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::Csel => {
            preg_hw(0)?; preg_hw(1)?; preg_hw(2)?;
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::Msub => {
            preg_hw(0)?; preg_hw(1)?; preg_hw(2)?;
            if inst.operands.len() > 3 { preg_hw(3)?; }
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::Smull | AArch64Opcode::Umull => {
            preg_hw(0)?; preg_hw(1)?; preg_hw(2)?;
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::Blr | AArch64Opcode::Br => {
            preg_hw(0)?;
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::Cbz | AArch64Opcode::Cbnz => {
            preg_hw(0)?;
            encode_instruction(inst).map_err(map_encode_error)
        }

        // LDR/STR with immediate offset — validate base register (#98)
        AArch64Opcode::LdrRI | AArch64Opcode::StrRI => {
            preg_hw(0)?; preg_hw(1)?;
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::LdrbRI | AArch64Opcode::LdrhRI
        | AArch64Opcode::LdrsbRI | AArch64Opcode::LdrshRI
        | AArch64Opcode::StrbRI | AArch64Opcode::StrhRI => {
            preg_hw(0)?; preg_hw(1)?;
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::StpRI | AArch64Opcode::LdpRI => {
            preg_hw(0)?; preg_hw(1)?; preg_hw(2)?;
            encode_instruction(inst).map_err(map_encode_error)
        }
        AArch64Opcode::StpPreIndex | AArch64Opcode::LdpPostIndex => {
            preg_hw(0)?; preg_hw(1)?; preg_hw(2)?;
            encode_instruction(inst).map_err(map_encode_error)
        }

        // =================================================================
        // All remaining opcodes: delegate directly to unified encoder.
        // No pre-validation needed (either no register operands to
        // validate, or the unified encoder handles them correctly).
        // =================================================================
        _ => encode_instruction(inst).map_err(map_encode_error),
    }
}

/// Map an `EncodeError` from the unified encoder to a `LowerError`.
fn map_encode_error(e: EncodeError) -> LowerError {
    match e {
        EncodeError::UnsupportedOpcode(op) => {
            LowerError::UnsupportedInstruction(format!("{:?}", op))
        }
        EncodeError::PseudoInstruction(op) => {
            LowerError::UnresolvedPseudo(op)
        }
        EncodeError::MissingOperand { opcode, index, .. } => {
            LowerError::MissingOperand { opcode, index }
        }
        other => {
            LowerError::EncodingFailed(other.to_string())
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
    let relaxed = relax::relax_branches(func)?;

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
    let relaxed = relax::relax_branches(func)?;

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
/// Walks the operand list and returns the first `Symbol(name)` found.
/// Symbol operands are created by ISel for instructions that reference
/// external names (BL for calls, ADRP/ADD for globals, etc.) and are
/// preserved through the IR pipeline by `convert_isel_operand`.
fn extract_symbol_name(inst: &MachInst) -> Option<String> {
    inst.operands.iter().find_map(|op| op.as_symbol().map(|s| s.to_string()))
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
    use llvm2_ir::regs::{X0, X1, X2, X19, W0, W1, W2};
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
    fn test_encode_tbz_bit3() {
        // TBZ X0, #3, +2  (bit 3, offset 2)
        let inst = MachInst::new(
            AArch64Opcode::Tbz,
            vec![MachOperand::PReg(X0), MachOperand::Imm(3), MachOperand::Imm(2)],
        );
        let word = encode_inst(&inst).unwrap();
        let expected = (0u32 << 31)
            | (0b011011 << 25)
            | (0 << 24)
            | (3 << 19)
            | (2 << 5)
            | 0;
        assert_eq!(word, expected, "TBZ X0, #3, +2 = 0x{word:08X}");
        assert_ne!(word, 0xD503201F, "TBZ must not emit NOP");
    }

    #[test]
    fn test_encode_tbnz_bit3() {
        let inst = MachInst::new(
            AArch64Opcode::Tbnz,
            vec![MachOperand::PReg(X0), MachOperand::Imm(3), MachOperand::Imm(2)],
        );
        let word = encode_inst(&inst).unwrap();
        let expected = (0u32 << 31)
            | (0b011011 << 25)
            | (1 << 24)
            | (3 << 19)
            | (2 << 5)
            | 0;
        assert_eq!(word, expected, "TBNZ X0, #3, +2 = 0x{word:08X}");
        assert_ne!(word, 0xD503201F, "TBNZ must not emit NOP");
    }

    #[test]
    fn test_encode_tbz_high_bit() {
        // TBZ X0, #32, +5  (bit 32 means b5=1, b40=0)
        let inst = MachInst::new(
            AArch64Opcode::Tbz,
            vec![MachOperand::PReg(X0), MachOperand::Imm(32), MachOperand::Imm(5)],
        );
        let word = encode_inst(&inst).unwrap();
        let expected = (1u32 << 31)
            | (0b011011 << 25)
            | (0 << 24)
            | (0 << 19)
            | (5 << 5)
            | 0;
        assert_eq!(word, expected, "TBZ X0, #32, +5 = 0x{word:08X}");
    }

    #[test]
    fn test_encode_tbnz_bit63() {
        // TBNZ X1, #63, +10  (bit 63: b5=1, b40=31)
        let inst = MachInst::new(
            AArch64Opcode::Tbnz,
            vec![MachOperand::PReg(X1), MachOperand::Imm(63), MachOperand::Imm(10)],
        );
        let word = encode_inst(&inst).unwrap();
        let expected = (1u32 << 31)
            | (0b011011 << 25)
            | (1 << 24)
            | (31 << 19)
            | (10 << 5)
            | 1;
        assert_eq!(word, expected, "TBNZ X1, #63, +10 = 0x{word:08X}");
    }

    #[test]
    fn test_encode_tbz_matches_unified_encoder() {
        let inst = MachInst::new(
            AArch64Opcode::Tbz,
            vec![MachOperand::PReg(X2), MachOperand::Imm(7), MachOperand::Imm(100)],
        );
        let lower_word = encode_inst(&inst).unwrap();
        let unified_word = crate::aarch64::encode::encode_instruction(&inst).unwrap();
        assert_eq!(lower_word, unified_word,
            "lower encode_inst and unified encode_instruction must agree for TBZ: lower=0x{lower_word:08X}, unified=0x{unified_word:08X}");
    }

    #[test]
    fn test_encode_tbnz_matches_unified_encoder() {
        let inst = MachInst::new(
            AArch64Opcode::Tbnz,
            vec![MachOperand::PReg(X2), MachOperand::Imm(15), MachOperand::Imm(50)],
        );
        let lower_word = encode_inst(&inst).unwrap();
        let unified_word = crate::aarch64::encode::encode_instruction(&inst).unwrap();
        assert_eq!(lower_word, unified_word,
            "lower encode_inst and unified encode_instruction must agree for TBNZ: lower=0x{lower_word:08X}, unified=0x{unified_word:08X}");
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

    #[test]
    fn test_expand_pseudos_preserves_trap_overflow() {
        // TrapOverflow is a real pseudo that the encoder handles as BRK #1.
        // expand_pseudos must NOT convert it to NOP.
        let mut inst = MachInst::new(AArch64Opcode::TrapOverflow, vec![MachOperand::Imm(0)]);
        inst.flags = InstFlags::IS_PSEUDO;
        let mut func = make_func("trap_test", vec![
            inst,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        expand_pseudos(&mut func);
        assert_eq!(func.insts[0].opcode, AArch64Opcode::TrapOverflow,
            "TrapOverflow must survive expand_pseudos, not become NOP");
    }

    #[test]
    fn test_expand_pseudos_preserves_trap_bounds_check() {
        let mut inst = MachInst::new(AArch64Opcode::TrapBoundsCheck, vec![]);
        inst.flags = InstFlags::IS_PSEUDO;
        let mut func = make_func("trap_bounds", vec![
            inst,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        expand_pseudos(&mut func);
        assert_eq!(func.insts[0].opcode, AArch64Opcode::TrapBoundsCheck,
            "TrapBoundsCheck must survive expand_pseudos, not become NOP");
    }

    #[test]
    fn test_expand_pseudos_preserves_trap_null() {
        let mut inst = MachInst::new(AArch64Opcode::TrapNull, vec![]);
        inst.flags = InstFlags::IS_PSEUDO;
        let mut func = make_func("trap_null", vec![
            inst,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        expand_pseudos(&mut func);
        assert_eq!(func.insts[0].opcode, AArch64Opcode::TrapNull,
            "TrapNull must survive expand_pseudos, not become NOP");
    }

    #[test]
    fn test_expand_pseudos_preserves_trap_div_zero() {
        let mut inst = MachInst::new(AArch64Opcode::TrapDivZero, vec![]);
        inst.flags = InstFlags::IS_PSEUDO;
        let mut func = make_func("trap_div", vec![
            inst,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        expand_pseudos(&mut func);
        assert_eq!(func.insts[0].opcode, AArch64Opcode::TrapDivZero,
            "TrapDivZero must survive expand_pseudos, not become NOP");
    }

    #[test]
    fn test_expand_pseudos_preserves_trap_shift_range() {
        let mut inst = MachInst::new(AArch64Opcode::TrapShiftRange, vec![]);
        inst.flags = InstFlags::IS_PSEUDO;
        let mut func = make_func("trap_shift", vec![
            inst,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        expand_pseudos(&mut func);
        assert_eq!(func.insts[0].opcode, AArch64Opcode::TrapShiftRange,
            "TrapShiftRange must survive expand_pseudos, not become NOP");
    }

    #[test]
    fn test_expand_pseudos_preserves_retain() {
        let mut inst = MachInst::new(AArch64Opcode::Retain, vec![MachOperand::PReg(X0)]);
        inst.flags = InstFlags::IS_PSEUDO;
        let mut func = make_func("retain_test", vec![
            inst,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        expand_pseudos(&mut func);
        assert_eq!(func.insts[0].opcode, AArch64Opcode::Retain,
            "Retain must survive expand_pseudos, not become NOP");
    }

    #[test]
    fn test_expand_pseudos_preserves_release() {
        let mut inst = MachInst::new(AArch64Opcode::Release, vec![MachOperand::PReg(X0)]);
        inst.flags = InstFlags::IS_PSEUDO;
        let mut func = make_func("release_test", vec![
            inst,
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ]);
        expand_pseudos(&mut func);
        assert_eq!(func.insts[0].opcode, AArch64Opcode::Release,
            "Release must survive expand_pseudos, not become NOP");
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

    // -----------------------------------------------------------------------
    // Symbol extraction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_symbol_from_bl() {
        let inst = MachInst::new(
            AArch64Opcode::Bl,
            vec![MachOperand::Symbol("_printf".to_string())],
        );
        assert_eq!(extract_symbol_name(&inst), Some("_printf".to_string()));
    }

    #[test]
    fn test_extract_symbol_from_adrp() {
        let inst = MachInst::new(
            AArch64Opcode::Adrp,
            vec![
                MachOperand::PReg(X0),
                MachOperand::Symbol("_my_global".to_string()),
            ],
        );
        assert_eq!(extract_symbol_name(&inst), Some("_my_global".to_string()));
    }

    #[test]
    fn test_extract_symbol_from_add_pcrel() {
        let inst = MachInst::new(
            AArch64Opcode::AddPCRel,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X0),
                MachOperand::Symbol("_my_global".to_string()),
            ],
        );
        assert_eq!(extract_symbol_name(&inst), Some("_my_global".to_string()));
    }

    #[test]
    fn test_extract_symbol_none_for_plain_add() {
        let inst = MachInst::new(
            AArch64Opcode::AddRR,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X0),
                MachOperand::PReg(X1),
            ],
        );
        assert_eq!(extract_symbol_name(&inst), None);
    }

    #[test]
    fn test_extract_symbol_none_for_ret() {
        let inst = MachInst::new(AArch64Opcode::Ret, vec![]);
        assert_eq!(extract_symbol_name(&inst), None);
    }

    // -----------------------------------------------------------------------
    // Relocation collection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_collect_relocations_bl_branch26() {
        let instructions = vec![
            MachInst::new(
                AArch64Opcode::Bl,
                vec![MachOperand::Symbol("_callee".to_string())],
            ),
        ];
        let relocs = collect_relocations(&instructions);
        assert_eq!(relocs.len(), 1);
        assert_eq!(relocs[0].offset, 0);
        assert_eq!(relocs[0].kind, RelocKind::Branch26);
        assert_eq!(relocs[0].symbol, "_callee");
        assert_eq!(relocs[0].addend, 0);
    }

    #[test]
    fn test_collect_relocations_adrp_page21() {
        let instructions = vec![
            MachInst::new(
                AArch64Opcode::Adrp,
                vec![
                    MachOperand::PReg(X0),
                    MachOperand::Symbol("_global_var".to_string()),
                ],
            ),
        ];
        let relocs = collect_relocations(&instructions);
        assert_eq!(relocs.len(), 1);
        assert_eq!(relocs[0].offset, 0);
        assert_eq!(relocs[0].kind, RelocKind::AdrpPage21);
        assert_eq!(relocs[0].symbol, "_global_var");
    }

    #[test]
    fn test_collect_relocations_add_pcrel_pageoff12() {
        let instructions = vec![
            MachInst::new(
                AArch64Opcode::AddPCRel,
                vec![
                    MachOperand::PReg(X0),
                    MachOperand::PReg(X0),
                    MachOperand::Symbol("_global_var".to_string()),
                ],
            ),
        ];
        let relocs = collect_relocations(&instructions);
        assert_eq!(relocs.len(), 1);
        assert_eq!(relocs[0].offset, 0);
        assert_eq!(relocs[0].kind, RelocKind::AddPageOff12);
        assert_eq!(relocs[0].symbol, "_global_var");
    }

    #[test]
    fn test_collect_relocations_adrp_add_pair() {
        let instructions = vec![
            MachInst::new(
                AArch64Opcode::Adrp,
                vec![
                    MachOperand::PReg(X0),
                    MachOperand::Symbol("_data".to_string()),
                ],
            ),
            MachInst::new(
                AArch64Opcode::AddPCRel,
                vec![
                    MachOperand::PReg(X0),
                    MachOperand::PReg(X0),
                    MachOperand::Symbol("_data".to_string()),
                ],
            ),
        ];
        let relocs = collect_relocations(&instructions);
        assert_eq!(relocs.len(), 2);
        assert_eq!(relocs[0].offset, 0);
        assert_eq!(relocs[0].kind, RelocKind::AdrpPage21);
        assert_eq!(relocs[0].symbol, "_data");
        assert_eq!(relocs[1].offset, 4);
        assert_eq!(relocs[1].kind, RelocKind::AddPageOff12);
        assert_eq!(relocs[1].symbol, "_data");
    }

    #[test]
    fn test_collect_relocations_mixed_with_no_symbol_instrs() {
        let instructions = vec![
            MachInst::new(
                AArch64Opcode::AddRR,
                vec![
                    MachOperand::PReg(X0),
                    MachOperand::PReg(X0),
                    MachOperand::PReg(X1),
                ],
            ),
            MachInst::new(
                AArch64Opcode::Bl,
                vec![MachOperand::Symbol("_callee".to_string())],
            ),
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ];
        let relocs = collect_relocations(&instructions);
        assert_eq!(relocs.len(), 1);
        assert_eq!(relocs[0].offset, 4);
        assert_eq!(relocs[0].kind, RelocKind::Branch26);
        assert_eq!(relocs[0].symbol, "_callee");
    }

    #[test]
    fn test_collect_relocations_no_symbol_no_relocs() {
        let instructions = vec![
            MachInst::new(
                AArch64Opcode::AddRR,
                vec![
                    MachOperand::PReg(X0),
                    MachOperand::PReg(X0),
                    MachOperand::PReg(X1),
                ],
            ),
            MachInst::new(AArch64Opcode::Ret, vec![]),
        ];
        let relocs = collect_relocations(&instructions);
        assert!(relocs.is_empty());
    }

    #[test]
    fn test_collect_relocations_bl_without_symbol_no_reloc() {
        let inst = MachInst::new(
            AArch64Opcode::Bl,
            vec![MachOperand::Imm(42)],
        );
        let relocs = collect_relocations(&[inst]);
        assert!(relocs.is_empty());
    }

    // -----------------------------------------------------------------------
    // Bug #98: preg_hw returns error for non-register operands
    // -----------------------------------------------------------------------

    #[test]
    fn test_preg_hw_rejects_imm_operand() {
        let inst = MachInst::new(
            AArch64Opcode::AddRR,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X0),
                MachOperand::Imm(42),
            ],
        );
        let result = encode_inst(&inst);
        assert!(result.is_err(), "Imm operand where register expected should error, not silently encode as XZR");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("expected register operand"), "Error should mention expected register, got: {}", err_msg);
    }

    #[test]
    fn test_preg_hw_rejects_frame_index() {
        use llvm2_ir::types::FrameIdx;
        let inst = MachInst::new(
            AArch64Opcode::LdrRI,
            vec![
                MachOperand::PReg(X0),
                MachOperand::FrameIndex(FrameIdx(-8)),
            ],
        );
        let result = encode_inst(&inst);
        assert!(result.is_err(), "FrameIndex where register expected should error");
    }

    #[test]
    fn test_preg_hw_rejects_stack_slot() {
        use llvm2_ir::types::StackSlotId;
        let inst = MachInst::new(
            AArch64Opcode::StrRI,
            vec![
                MachOperand::PReg(X0),
                MachOperand::StackSlot(StackSlotId(0)),
            ],
        );
        let result = encode_inst(&inst);
        assert!(result.is_err(), "StackSlot where register expected should error");
    }

    // -----------------------------------------------------------------------
    // Bug #105: FP size derived from register class, not hardcoded
    // -----------------------------------------------------------------------

    #[test]
    fn test_fp_add_double_precision() {
        use llvm2_ir::regs::{D0, D1};
        let inst = MachInst::new(
            AArch64Opcode::FaddRR,
            vec![
                MachOperand::PReg(D0),
                MachOperand::PReg(D0),
                MachOperand::PReg(D1),
            ],
        );
        let word = encode_inst(&inst).unwrap();
        let ftype = (word >> 22) & 0b11;
        assert_eq!(ftype, 0b01, "FADD with D-regs should use ftype=01 (double), got {}", ftype);
    }

    #[test]
    fn test_fp_add_single_precision() {
        use llvm2_ir::regs::{S0, S1};
        let inst = MachInst::new(
            AArch64Opcode::FaddRR,
            vec![
                MachOperand::PReg(S0),
                MachOperand::PReg(S0),
                MachOperand::PReg(S1),
            ],
        );
        let word = encode_inst(&inst).unwrap();
        let ftype = (word >> 22) & 0b11;
        assert_eq!(ftype, 0b00, "FADD with S-regs should use ftype=00 (single), got {}", ftype);
    }

    #[test]
    fn test_fp_add_rejects_gpr_operand() {
        let inst = MachInst::new(
            AArch64Opcode::FaddRR,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X0),
                MachOperand::PReg(X1),
            ],
        );
        let result = encode_inst(&inst);
        assert!(result.is_err(), "FADD with GPR operands should error, not silently default to double");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("expected FPR"), "Error should mention expected FPR, got: {}", err_msg);
    }

    #[test]
    fn test_fp_neg_single_precision() {
        use llvm2_ir::regs::{S0, S1};
        let inst = MachInst::new(
            AArch64Opcode::FnegRR,
            vec![MachOperand::PReg(S0), MachOperand::PReg(S1)],
        );
        let word = encode_inst(&inst).unwrap();
        let ftype = (word >> 22) & 0b11;
        assert_eq!(ftype, 0b00, "FNEG with S-regs should use ftype=00 (single), got {}", ftype);
    }

    #[test]
    fn test_fcmp_single_precision() {
        use llvm2_ir::regs::{S0, S1};
        let inst = MachInst::new(
            AArch64Opcode::Fcmp,
            vec![MachOperand::PReg(S0), MachOperand::PReg(S1)],
        );
        let word = encode_inst(&inst).unwrap();
        let ftype = (word >> 22) & 0b11;
        assert_eq!(ftype, 0b00, "FCMP with S-regs should use ftype=00 (single), got {}", ftype);
    }

    #[test]
    fn test_fcvtzs_single_precision() {
        use llvm2_ir::regs::S0;
        let inst = MachInst::new(
            AArch64Opcode::FcvtzsRR,
            vec![
                MachOperand::PReg(llvm2_ir::regs::PReg::new(32)),
                MachOperand::PReg(S0),
            ],
        );
        let word = encode_inst(&inst).unwrap();
        let ftype = (word >> 22) & 0b11;
        assert_eq!(ftype, 0b00, "FCVTZS from S-reg should use ftype=00 (single), got {}", ftype);
    }

    #[test]
    fn test_scvtf_single_precision() {
        use llvm2_ir::regs::S0;
        let inst = MachInst::new(
            AArch64Opcode::ScvtfRR,
            vec![
                MachOperand::PReg(S0),
                MachOperand::PReg(X0),
            ],
        );
        let word = encode_inst(&inst).unwrap();
        let ftype = (word >> 22) & 0b11;
        assert_eq!(ftype, 0b00, "SCVTF to S-reg should use ftype=00 (single), got {}", ftype);
    }

    // -----------------------------------------------------------------------
    // Immediate shift encoding tests (previously emitted NOP — fixed in #134)
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_lsl_ri_not_nop() {
        // LSL X0, X1, #2 must NOT emit NOP
        let inst = MachInst::new(
            AArch64Opcode::LslRI,
            vec![MachOperand::PReg(X0), MachOperand::PReg(X1), MachOperand::Imm(2)],
        );
        let word = encode_inst(&inst).unwrap();
        assert_ne!(word, 0xD503201F, "LSL X0, X1, #2 must not emit NOP");
        // Verify it matches the unified encoder
        let unified = crate::aarch64::encode::encode_instruction(&inst).unwrap();
        assert_eq!(word, unified,
            "lower and unified must agree for LslRI: lower=0x{word:08X}, unified=0x{unified:08X}");
    }

    #[test]
    fn test_encode_lsr_ri_not_nop() {
        let inst = MachInst::new(
            AArch64Opcode::LsrRI,
            vec![MachOperand::PReg(X0), MachOperand::PReg(X1), MachOperand::Imm(2)],
        );
        let word = encode_inst(&inst).unwrap();
        assert_ne!(word, 0xD503201F, "LSR X0, X1, #2 must not emit NOP");
        let unified = crate::aarch64::encode::encode_instruction(&inst).unwrap();
        assert_eq!(word, unified,
            "lower and unified must agree for LsrRI: lower=0x{word:08X}, unified=0x{unified:08X}");
    }

    #[test]
    fn test_encode_asr_ri_not_nop() {
        let inst = MachInst::new(
            AArch64Opcode::AsrRI,
            vec![MachOperand::PReg(X0), MachOperand::PReg(X1), MachOperand::Imm(2)],
        );
        let word = encode_inst(&inst).unwrap();
        assert_ne!(word, 0xD503201F, "ASR X0, X1, #2 must not emit NOP");
        let unified = crate::aarch64::encode::encode_instruction(&inst).unwrap();
        assert_eq!(word, unified,
            "lower and unified must agree for AsrRI: lower=0x{word:08X}, unified=0x{unified:08X}");
    }

    #[test]
    fn test_encode_lsl_ri_known_value() {
        // LSL X0, X1, #2 = UBFM X0, X1, #62, #61
        // Expected: 0xD37EF420 (from ARM ARM)
        let inst = MachInst::new(
            AArch64Opcode::LslRI,
            vec![MachOperand::PReg(X0), MachOperand::PReg(X1), MachOperand::Imm(2)],
        );
        let word = encode_inst(&inst).unwrap();
        assert_eq!(word, 0xD37EF420, "LSL X0, X1, #2 = 0x{word:08X}");
    }

    #[test]
    fn test_encode_lsr_ri_known_value() {
        // LSR X0, X1, #2 = UBFM X0, X1, #2, #63
        // Expected: 0xD342FC20
        let inst = MachInst::new(
            AArch64Opcode::LsrRI,
            vec![MachOperand::PReg(X0), MachOperand::PReg(X1), MachOperand::Imm(2)],
        );
        let word = encode_inst(&inst).unwrap();
        assert_eq!(word, 0xD342FC20, "LSR X0, X1, #2 = 0x{word:08X}");
    }

    #[test]
    fn test_encode_asr_ri_known_value() {
        // ASR X0, X1, #2 = SBFM X0, X1, #2, #63
        // Expected: 0x9342FC20
        let inst = MachInst::new(
            AArch64Opcode::AsrRI,
            vec![MachOperand::PReg(X0), MachOperand::PReg(X1), MachOperand::Imm(2)],
        );
        let word = encode_inst(&inst).unwrap();
        assert_eq!(word, 0x9342FC20, "ASR X0, X1, #2 = 0x{word:08X}");
    }

    // -----------------------------------------------------------------------
    // UBFM/SBFM/BFM encoding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_ubfm() {
        // UBFM X0, X1, #2, #5
        let inst = MachInst::new(
            AArch64Opcode::Ubfm,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X1),
                MachOperand::Imm(2),
                MachOperand::Imm(5),
            ],
        );
        let word = encode_inst(&inst).unwrap();
        let unified = crate::aarch64::encode::encode_instruction(&inst).unwrap();
        assert_eq!(word, unified,
            "lower and unified must agree for UBFM: lower=0x{word:08X}, unified=0x{unified:08X}");
        // Check opc field = 10 (UBFM)
        assert_eq!((word >> 29) & 0b11, 0b10, "UBFM opc should be 10");
    }

    #[test]
    fn test_encode_sbfm() {
        // SBFM X0, X1, #0, #31 (= SXTW alias)
        let inst = MachInst::new(
            AArch64Opcode::Sbfm,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X1),
                MachOperand::Imm(0),
                MachOperand::Imm(31),
            ],
        );
        let word = encode_inst(&inst).unwrap();
        let unified = crate::aarch64::encode::encode_instruction(&inst).unwrap();
        assert_eq!(word, unified,
            "lower and unified must agree for SBFM: lower=0x{word:08X}, unified=0x{unified:08X}");
        // Check opc field = 00 (SBFM)
        assert_eq!((word >> 29) & 0b11, 0b00, "SBFM opc should be 00");
    }

    #[test]
    fn test_encode_bfm() {
        // BFM X0, X1, #4, #7
        let inst = MachInst::new(
            AArch64Opcode::Bfm,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X1),
                MachOperand::Imm(4),
                MachOperand::Imm(7),
            ],
        );
        let word = encode_inst(&inst).unwrap();
        let unified = crate::aarch64::encode::encode_instruction(&inst).unwrap();
        assert_eq!(word, unified,
            "lower and unified must agree for BFM: lower=0x{word:08X}, unified=0x{unified:08X}");
        // Check opc field = 01 (BFM)
        assert_eq!((word >> 29) & 0b11, 0b01, "BFM opc should be 01");
    }

    // -----------------------------------------------------------------------
    // LDR/STR register offset encoding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_ldr_ro() {
        // LDR X0, [X1, X2]
        let inst = MachInst::new(
            AArch64Opcode::LdrRO,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X1),
                MachOperand::PReg(X2),
            ],
        );
        let word = encode_inst(&inst).unwrap();
        assert_ne!(word, 0xD503201F, "LdrRO must not emit NOP");
        // Check it's a load (opc=01) and register offset (bit 21=1)
        assert_eq!((word >> 22) & 0b11, 0b01, "LdrRO should have opc=01 (load)");
        assert_eq!((word >> 21) & 1, 1, "LdrRO should have bit 21=1 (register offset)");
    }

    #[test]
    fn test_encode_str_ro() {
        // STR X0, [X1, X2]
        let inst = MachInst::new(
            AArch64Opcode::StrRO,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X1),
                MachOperand::PReg(X2),
            ],
        );
        let word = encode_inst(&inst).unwrap();
        assert_ne!(word, 0xD503201F, "StrRO must not emit NOP");
        // Check it's a store (opc=00) and register offset (bit 21=1)
        assert_eq!((word >> 22) & 0b11, 0b00, "StrRO should have opc=00 (store)");
        assert_eq!((word >> 21) & 1, 1, "StrRO should have bit 21=1 (register offset)");
    }

    // -----------------------------------------------------------------------
    // GOT/TLV load encoding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_ldr_got() {
        // LDR X0, [X1, #0] (GOT load)
        let inst = MachInst::new(
            AArch64Opcode::LdrGot,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X1),
                MachOperand::Imm(0),
            ],
        );
        let word = encode_inst(&inst).unwrap();
        assert_ne!(word, 0xD503201F, "LdrGot must not emit NOP");
        // Verify it's a 64-bit load (size=11) with unsigned offset
        assert_eq!((word >> 30) & 0b11, 0b11, "LdrGot size should be 11 (64-bit)");
        assert_eq!((word >> 22) & 0b11, 0b01, "LdrGot opc should be 01 (load)");
    }

    #[test]
    fn test_encode_ldr_tlvp() {
        // LDR X0, [X1, #0] (TLV load)
        let inst = MachInst::new(
            AArch64Opcode::LdrTlvp,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X1),
                MachOperand::Imm(0),
            ],
        );
        let word = encode_inst(&inst).unwrap();
        assert_ne!(word, 0xD503201F, "LdrTlvp must not emit NOP");
        assert_eq!((word >> 30) & 0b11, 0b11, "LdrTlvp size should be 11 (64-bit)");
    }

    // -----------------------------------------------------------------------
    // BIC, CSINC, CSINV, CSNEG, MOVN encoding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_bic_rr() {
        // BIC X0, X1, X2 = AND X0, X1, NOT(X2)
        // Uses same convention as ORN: N-bit encoded via shift param in
        // encode_logical_shifted_reg (see note in ORN — pre-existing convention).
        let inst = MachInst::new(
            AArch64Opcode::BicRR,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X1),
                MachOperand::PReg(X2),
            ],
        );
        let word = encode_inst(&inst).unwrap();
        // BIC uses opc=00 (AND family)
        assert_eq!((word >> 29) & 0b11, 0b00, "BIC opc should be 00 (AND family)");
        // Verify encoding matches encode_logical_shifted_reg with same convention as ORN
        let expected = encoding::encode_logical_shifted_reg(1, 0b00, 1, 0, 2, 0, 1, 0);
        assert_eq!(word, expected, "BIC X0, X1, X2 = 0x{word:08X}");
    }

    #[test]
    fn test_encode_csinc() {
        // CSINC X0, X1, X2, EQ (cond=0)
        let inst = MachInst::new(
            AArch64Opcode::Csinc,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X1),
                MachOperand::PReg(X2),
                MachOperand::Imm(0), // EQ condition
            ],
        );
        let word = encode_inst(&inst).unwrap();
        // Check op2 field = 01 (CSINC)
        assert_eq!((word >> 10) & 0b11, 0b01, "CSINC op2 should be 01");
        // Check opc field is 00 (not CSINV/CSNEG which use 10)
        assert_eq!((word >> 29) & 0b11, 0b00, "CSINC should have bits 30:29 = 00");
    }

    #[test]
    fn test_encode_csinv() {
        // CSINV X0, X1, X2, NE (cond=1) — 64-bit form
        let inst = MachInst::new(
            AArch64Opcode::Csinv,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X1),
                MachOperand::PReg(X2),
                MachOperand::Imm(1), // NE condition
            ],
        );
        let word = encode_inst(&inst).unwrap();
        // Check op2 field = 00 (CSINV)
        assert_eq!((word >> 10) & 0b11, 0b00, "CSINV op2 should be 00");
        // Check bit 30 = 1 (CSINV uses op=1)
        assert_eq!((word >> 30) & 1, 1, "CSINV should have bit 30 = 1");
        // S-field (bit 29) must be 0 — CSINV is non-flag-setting (#165)
        assert_eq!((word >> 29) & 1, 0, "CSINV S-field (bit 29) must be 0");
        // sf (bit 31) = 1 for 64-bit
        assert_eq!((word >> 31) & 1, 1, "64-bit CSINV should have sf=1");
    }

    #[test]
    fn test_encode_csinv_32bit() {
        // CSINV W0, W1, W2, NE (cond=1) — 32-bit form (#165)
        let inst = MachInst::new(
            AArch64Opcode::Csinv,
            vec![
                MachOperand::PReg(W0),
                MachOperand::PReg(W1),
                MachOperand::PReg(W2),
                MachOperand::Imm(1), // NE condition
            ],
        );
        let word = encode_inst(&inst).unwrap();
        // Check op2 field = 00 (CSINV)
        assert_eq!((word >> 10) & 0b11, 0b00, "32-bit CSINV op2 should be 00");
        // Check bit 30 = 1 (CSINV uses op=1)
        assert_eq!((word >> 30) & 1, 1, "32-bit CSINV should have bit 30 = 1");
        // S-field (bit 29) must be 0 — CSINV is non-flag-setting (#165)
        assert_eq!((word >> 29) & 1, 0, "32-bit CSINV S-field (bit 29) must be 0");
        // sf (bit 31) = 0 for 32-bit
        assert_eq!((word >> 31) & 1, 0, "32-bit CSINV should have sf=0");
    }

    #[test]
    fn test_encode_csneg() {
        // CSNEG X0, X1, X2, EQ (cond=0) — 64-bit form
        let inst = MachInst::new(
            AArch64Opcode::Csneg,
            vec![
                MachOperand::PReg(X0),
                MachOperand::PReg(X1),
                MachOperand::PReg(X2),
                MachOperand::Imm(0),
            ],
        );
        let word = encode_inst(&inst).unwrap();
        // Check op2 field = 01 (CSNEG)
        assert_eq!((word >> 10) & 0b11, 0b01, "CSNEG op2 should be 01");
        // Check bit 30 = 1 (CSNEG uses op=1)
        assert_eq!((word >> 30) & 1, 1, "CSNEG should have bit 30 = 1");
        // S-field (bit 29) must be 0 — CSNEG is non-flag-setting (#165)
        assert_eq!((word >> 29) & 1, 0, "CSNEG S-field (bit 29) must be 0");
        // sf (bit 31) = 1 for 64-bit
        assert_eq!((word >> 31) & 1, 1, "64-bit CSNEG should have sf=1");
    }

    #[test]
    fn test_encode_csneg_32bit() {
        // CSNEG W0, W1, W2, EQ (cond=0) — 32-bit form (#165)
        let inst = MachInst::new(
            AArch64Opcode::Csneg,
            vec![
                MachOperand::PReg(W0),
                MachOperand::PReg(W1),
                MachOperand::PReg(W2),
                MachOperand::Imm(0),
            ],
        );
        let word = encode_inst(&inst).unwrap();
        // Check op2 field = 01 (CSNEG)
        assert_eq!((word >> 10) & 0b11, 0b01, "32-bit CSNEG op2 should be 01");
        // Check bit 30 = 1 (CSNEG uses op=1)
        assert_eq!((word >> 30) & 1, 1, "32-bit CSNEG should have bit 30 = 1");
        // S-field (bit 29) must be 0 — CSNEG is non-flag-setting (#165)
        assert_eq!((word >> 29) & 1, 0, "32-bit CSNEG S-field (bit 29) must be 0");
        // sf (bit 31) = 0 for 32-bit
        assert_eq!((word >> 31) & 1, 0, "32-bit CSNEG should have sf=0");
    }

    #[test]
    fn test_encode_movn() {
        // MOVN X0, #42
        let inst = MachInst::new(
            AArch64Opcode::Movn,
            vec![MachOperand::PReg(X0), MachOperand::Imm(42)],
        );
        let word = encode_inst(&inst).unwrap();
        // MOVN uses opc=00 in the move-wide encoding
        let opc = (word >> 29) & 0b11;
        assert_eq!(opc, 0b00, "MOVN opc should be 00");
        // Check the immediate is embedded
        let imm_field = (word >> 5) & 0xFFFF;
        assert_eq!(imm_field, 42, "MOVN imm16 should be 42");
    }
}
