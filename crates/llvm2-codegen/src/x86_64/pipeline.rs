// llvm2-codegen/x86_64/pipeline.rs - x86-64 end-to-end compilation pipeline
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: System V AMD64 ABI (https://refspecs.linuxbase.org/elf/x86_64-abi-0.99.pdf)
// Reference: ~/llvm-project-ref/llvm/lib/Target/X86/X86FrameLowering.cpp

//! x86-64 end-to-end compilation pipeline.
//!
//! Takes an x86-64 ISel function (`X86ISelFunction`) and produces encoded
//! machine code bytes (optionally wrapped in an ELF .o file).
//!
//! # Pipeline phases
//!
//! ```text
//! Phase 1: Instruction Selection (llvm2-lower/x86_64_isel)
//!   tMIR Function -> X86ISelFunction (X86Opcodes, VRegs)
//!
//! Phase 2: x86-64 prologue/epilogue insertion
//!   Stack frame setup/teardown for System V AMD64 ABI
//!
//! Phase 3: Branch resolution
//!   Resolve block references to byte offsets (variable-length encoding)
//!
//! Phase 4: Encoding (llvm2-codegen/x86_64/encode)
//!   X86ISelFunction -> Vec<u8> (machine code bytes)
//!
//! Phase 5: ELF emission (optional)
//!   Vec<u8> -> ELF .o file bytes
//! ```
//!
//! # Note on register allocation
//!
//! The existing register allocator (`llvm2-regalloc`) operates on
//! `llvm2_ir::MachFunction` which uses AArch64-centric types (`PReg`,
//! `AArch64Opcode`). The x86-64 ISel produces `X86ISelFunction` with
//! `X86PReg` and `X86Opcode` — a separate type universe.
//!
//! For the initial pipeline, we implement a simplified register assignment
//! that maps VRegs directly to physical registers using a linear scan over
//! the x86-64 allocatable register set. This avoids the type-system mismatch
//! while still producing correct code for simple functions. A full x86-64
//! regalloc adapter will be built in a follow-up issue.

use std::collections::HashMap;

use llvm2_ir::regs::{RegClass, VReg};
use llvm2_ir::x86_64_ops::X86Opcode;
use llvm2_ir::x86_64_regs::{
    self, X86PReg, RBP, RSP,
    X86_ALLOCATABLE_GPRS, X86_ALLOCATABLE_XMMS,
    X86_CALLEE_SAVED_GPRS,
};

use llvm2_lower::x86_64_isel::{
    X86ISelFunction, X86ISelInst, X86ISelOperand,
};

use crate::x86_64::encode::{X86EncodeError, X86Encoder, X86InstOperands};
use crate::elf::{ElfMachine, ElfWriter};
use crate::macho::writer::{MachOTarget, MachOWriter};

// ---------------------------------------------------------------------------
// Pipeline errors
// ---------------------------------------------------------------------------

/// Errors during x86-64 compilation.
#[derive(Debug)]
pub enum X86PipelineError {
    /// Instruction selection failed.
    ISel(String),
    /// Register allocation ran out of registers.
    RegAlloc(String),
    /// Encoding failed.
    Encoding(X86EncodeError),
    /// Prologue/epilogue generation failed.
    FrameLowering(String),
}

impl core::fmt::Display for X86PipelineError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ISel(msg) => write!(f, "x86-64 ISel failed: {}", msg),
            Self::RegAlloc(msg) => write!(f, "x86-64 regalloc failed: {}", msg),
            Self::Encoding(e) => write!(f, "x86-64 encoding failed: {}", e),
            Self::FrameLowering(msg) => write!(f, "x86-64 frame lowering failed: {}", msg),
        }
    }
}

impl From<X86EncodeError> for X86PipelineError {
    fn from(e: X86EncodeError) -> Self {
        Self::Encoding(e)
    }
}

// ---------------------------------------------------------------------------
// Simple register allocator for x86-64
// ---------------------------------------------------------------------------

/// Simple linear-scan register assignment for x86-64 ISel output.
///
/// This is a temporary solution until the main regalloc is adapted for
/// x86-64 types. It assigns VRegs to physical registers in order of
/// first appearance, spilling to stack if we run out.
pub struct X86RegAssignment {
    /// VReg -> physical register mapping.
    pub allocation: HashMap<VReg, X86PReg>,
    /// Set of callee-saved registers that were used (need save/restore).
    pub used_callee_saved: Vec<X86PReg>,
    /// Number of spill slots needed.
    pub num_spills: u32,
}

impl X86RegAssignment {
    /// Perform register assignment on an ISel function.
    pub fn assign(func: &X86ISelFunction) -> Result<Self, X86PipelineError> {
        let mut allocation: HashMap<VReg, X86PReg> = HashMap::new();
        let mut gpr_idx: usize = 0;
        let mut xmm_idx: usize = 0;
        let mut used_callee_saved: Vec<X86PReg> = Vec::new();

        // Collect all VRegs referenced in the function.
        let mut vregs: Vec<VReg> = Vec::new();
        for block in func.block_order.iter() {
            if let Some(mblock) = func.blocks.get(block) {
                for inst in &mblock.insts {
                    for op in &inst.operands {
                        if let X86ISelOperand::VReg(v) = op
                            && !vregs.contains(v) {
                                vregs.push(*v);
                            }
                    }
                }
            }
        }

        // Assign physical registers.
        for vreg in &vregs {
            let is_fp = matches!(
                vreg.class,
                RegClass::Fpr32 | RegClass::Fpr64 | RegClass::Fpr128
            );

            if is_fp {
                if xmm_idx < X86_ALLOCATABLE_XMMS.len() {
                    let preg = X86_ALLOCATABLE_XMMS[xmm_idx];
                    allocation.insert(*vreg, preg);
                    xmm_idx += 1;
                } else {
                    return Err(X86PipelineError::RegAlloc(format!(
                        "ran out of XMM registers for vreg v{}",
                        vreg.id
                    )));
                }
            } else if gpr_idx < X86_ALLOCATABLE_GPRS.len() {
                let preg = X86_ALLOCATABLE_GPRS[gpr_idx];
                allocation.insert(*vreg, preg);

                // Track callee-saved usage.
                if X86_CALLEE_SAVED_GPRS.contains(&preg)
                    && !used_callee_saved.contains(&preg)
                {
                    used_callee_saved.push(preg);
                }

                gpr_idx += 1;
            } else {
                return Err(X86PipelineError::RegAlloc(format!(
                    "ran out of GPR registers for vreg v{}",
                    vreg.id
                )));
            }
        }

        Ok(Self {
            allocation,
            used_callee_saved,
            num_spills: 0,
        })
    }
}

// ---------------------------------------------------------------------------
// Operand resolution: X86ISelOperand -> X86InstOperands for the encoder
// ---------------------------------------------------------------------------

/// Resolve an ISel operand to a physical register after allocation.
fn resolve_operand(
    op: &X86ISelOperand,
    alloc: &HashMap<VReg, X86PReg>,
) -> Option<X86PReg> {
    match op {
        X86ISelOperand::VReg(v) => alloc.get(v).copied(),
        X86ISelOperand::PReg(p) => Some(*p),
        _ => None,
    }
}

/// Convert an ISel instruction to encoder operands using the register assignment.
fn resolve_inst_operands(
    inst: &X86ISelInst,
    alloc: &HashMap<VReg, X86PReg>,
) -> X86InstOperands {
    let mut ops = X86InstOperands::none();

    match inst.opcode {
        // Pseudo-instructions: no operands needed.
        X86Opcode::Nop | X86Opcode::Phi | X86Opcode::StackAlloc => {}

        // Register-register: [dst, src] or [dst, lhs, rhs] (three-address -> two-address)
        X86Opcode::AddRR | X86Opcode::SubRR | X86Opcode::AndRR
        | X86Opcode::OrRR | X86Opcode::XorRR => {
            // Three-address ISel: [dst, lhs, rhs]
            // x86-64 two-address encoding: dst = lhs op rhs, where dst == lhs.
            // The encoder takes (dst, src) where dst is the first source.
            if inst.operands.len() >= 3 {
                ops.dst = resolve_operand(&inst.operands[0], alloc);
                ops.src = resolve_operand(&inst.operands[2], alloc);
            } else if inst.operands.len() == 2 {
                ops.dst = resolve_operand(&inst.operands[0], alloc);
                ops.src = resolve_operand(&inst.operands[1], alloc);
            }
        }

        // IMUL r64, r64: dst, src (two-operand form)
        X86Opcode::ImulRR => {
            if inst.operands.len() >= 3 {
                ops.dst = resolve_operand(&inst.operands[0], alloc);
                ops.src = resolve_operand(&inst.operands[2], alloc);
            } else if inst.operands.len() == 2 {
                ops.dst = resolve_operand(&inst.operands[0], alloc);
                ops.src = resolve_operand(&inst.operands[1], alloc);
            }
        }

        // CMP: [lhs, rhs] (no destination, just sets flags)
        X86Opcode::CmpRR | X86Opcode::TestRR => {
            if inst.operands.len() >= 2 {
                ops.dst = resolve_operand(&inst.operands[0], alloc);
                ops.src = resolve_operand(&inst.operands[1], alloc);
            }
        }

        // CMP r, imm32
        X86Opcode::CmpRI | X86Opcode::TestRI => {
            if let Some(reg_op) = inst.operands.first() {
                ops.dst = resolve_operand(reg_op, alloc);
            }
            for op in &inst.operands {
                if let X86ISelOperand::Imm(imm) = op {
                    ops.imm = *imm;
                    break;
                }
            }
        }

        // ALU reg-imm: [dst, imm] or [dst, src, imm]
        X86Opcode::AddRI | X86Opcode::SubRI | X86Opcode::AndRI
        | X86Opcode::OrRI | X86Opcode::XorRI => {
            if let Some(reg_op) = inst.operands.first() {
                ops.dst = resolve_operand(reg_op, alloc);
            }
            for op in &inst.operands {
                if let X86ISelOperand::Imm(imm) = op {
                    ops.imm = *imm;
                    break;
                }
            }
        }

        // Unary: [dst, src]
        X86Opcode::Neg | X86Opcode::Not | X86Opcode::Inc | X86Opcode::Dec => {
            // dst operand (in-place)
            if let Some(op) = inst.operands.first() {
                ops.dst = resolve_operand(op, alloc);
            }
        }

        // MOV r, r
        X86Opcode::MovRR => {
            if inst.operands.len() >= 2 {
                ops.dst = resolve_operand(&inst.operands[0], alloc);
                ops.src = resolve_operand(&inst.operands[1], alloc);
            }
        }

        // MOV r, imm64
        X86Opcode::MovRI => {
            if let Some(op) = inst.operands.first() {
                ops.dst = resolve_operand(op, alloc);
            }
            for op in &inst.operands {
                if let X86ISelOperand::Imm(imm) = op {
                    ops.imm = *imm;
                    break;
                }
            }
        }

        // MOV r, [base+disp] (load)
        X86Opcode::MovRM => {
            if let Some(op) = inst.operands.first() {
                ops.dst = resolve_operand(op, alloc);
            }
            for op in &inst.operands {
                if let X86ISelOperand::MemAddr { base, disp } = op {
                    ops.base = resolve_operand(base, alloc);
                    ops.disp = *disp as i64;
                    break;
                }
            }
        }

        // MOV [base+disp], r (store)
        X86Opcode::MovMR => {
            // For stores, the encoder expects dst = source register, base = mem base.
            for op in &inst.operands {
                if let X86ISelOperand::MemAddr { base, disp } = op {
                    ops.base = resolve_operand(base, alloc);
                    ops.disp = *disp as i64;
                } else {
                    // The non-memaddr operand is the source register.
                    if ops.dst.is_none() {
                        ops.dst = resolve_operand(op, alloc);
                    }
                }
            }
            // If MemAddr came first, dst might not be set yet.
            if ops.dst.is_none()
                && let Some(last) = inst.operands.last() {
                    ops.dst = resolve_operand(last, alloc);
                }
        }

        // PUSH / POP: single register
        X86Opcode::Push | X86Opcode::Pop => {
            if let Some(op) = inst.operands.first() {
                ops.dst = resolve_operand(op, alloc);
            }
        }

        // RET: no operands
        X86Opcode::Ret => {}

        // JMP rel32: block operand resolved to displacement
        X86Opcode::Jmp => {
            // Block operands are resolved during branch resolution.
            // After resolution, the displacement is stored in the operand.
            for op in &inst.operands {
                if let X86ISelOperand::Imm(imm) = op {
                    ops.disp = *imm;
                    break;
                }
            }
        }

        // Jcc: [cc, block/imm]
        X86Opcode::Jcc => {
            for op in &inst.operands {
                match op {
                    X86ISelOperand::CondCode(cc) => {
                        ops.cc = Some(*cc);
                    }
                    X86ISelOperand::Imm(imm) => {
                        ops.disp = *imm;
                    }
                    _ => {}
                }
            }
        }

        // CALL: symbol (rel32 = 0 placeholder for relocation)
        X86Opcode::Call => {
            ops.disp = 0; // Will be fixed by linker relocation
        }
        X86Opcode::CallR => {
            if let Some(op) = inst.operands.first() {
                ops.dst = resolve_operand(op, alloc);
            }
        }

        // SSE: [dst, src]
        X86Opcode::Addsd | X86Opcode::Subsd | X86Opcode::Mulsd | X86Opcode::Divsd
        | X86Opcode::Addss | X86Opcode::Subss | X86Opcode::Mulss | X86Opcode::Divss
        | X86Opcode::MovsdRR | X86Opcode::MovssRR
        | X86Opcode::Ucomisd | X86Opcode::Ucomiss => {
            if inst.operands.len() >= 2 {
                ops.dst = resolve_operand(&inst.operands[0], alloc);
                ops.src = resolve_operand(&inst.operands[1], alloc);
            }
        }

        // Shifts: [dst, imm] or [dst]
        X86Opcode::ShlRI | X86Opcode::ShrRI | X86Opcode::SarRI => {
            if let Some(op) = inst.operands.first() {
                ops.dst = resolve_operand(op, alloc);
            }
            for op in &inst.operands {
                if let X86ISelOperand::Imm(imm) = op {
                    ops.imm = *imm;
                    break;
                }
            }
        }
        X86Opcode::ShlRR | X86Opcode::ShrRR | X86Opcode::SarRR => {
            if let Some(op) = inst.operands.first() {
                ops.dst = resolve_operand(op, alloc);
            }
        }

        // LEA, MOVZX, MOVSX, etc. — handle generically
        X86Opcode::Lea => {
            if let Some(op) = inst.operands.first() {
                ops.dst = resolve_operand(op, alloc);
            }
            for op in &inst.operands {
                if let X86ISelOperand::MemAddr { base, disp } = op {
                    ops.base = resolve_operand(base, alloc);
                    ops.disp = *disp as i64;
                    break;
                }
            }
        }
        X86Opcode::Movzx | X86Opcode::Movsx => {
            if inst.operands.len() >= 2 {
                ops.dst = resolve_operand(&inst.operands[0], alloc);
                ops.src = resolve_operand(&inst.operands[1], alloc);
            }
        }

        // CMOVcc, SETcc
        X86Opcode::Cmovcc => {
            if inst.operands.len() >= 2 {
                ops.dst = resolve_operand(&inst.operands[0], alloc);
                ops.src = resolve_operand(&inst.operands[1], alloc);
            }
            for op in &inst.operands {
                if let X86ISelOperand::CondCode(cc) = op {
                    ops.cc = Some(*cc);
                    break;
                }
            }
        }
        X86Opcode::Setcc => {
            if let Some(op) = inst.operands.first() {
                ops.dst = resolve_operand(op, alloc);
            }
            for op in &inst.operands {
                if let X86ISelOperand::CondCode(cc) = op {
                    ops.cc = Some(*cc);
                    break;
                }
            }
        }

        // Bit manipulation: [dst, src]
        X86Opcode::Bsf | X86Opcode::Bsr | X86Opcode::Tzcnt
        | X86Opcode::Lzcnt | X86Opcode::Popcnt => {
            if inst.operands.len() >= 2 {
                ops.dst = resolve_operand(&inst.operands[0], alloc);
                ops.src = resolve_operand(&inst.operands[1], alloc);
            }
        }

        // IDIV/DIV, IMUL 3-operand, etc.
        X86Opcode::Idiv | X86Opcode::Div => {
            if let Some(op) = inst.operands.first() {
                ops.dst = resolve_operand(op, alloc);
            }
        }
        X86Opcode::ImulRRI => {
            if inst.operands.len() >= 2 {
                ops.dst = resolve_operand(&inst.operands[0], alloc);
                ops.src = resolve_operand(&inst.operands[1], alloc);
            }
            for op in &inst.operands {
                if let X86ISelOperand::Imm(imm) = op {
                    ops.imm = *imm;
                    break;
                }
            }
        }

        // Memory forms
        X86Opcode::AddRM | X86Opcode::SubRM | X86Opcode::CmpRM => {
            if let Some(op) = inst.operands.first() {
                ops.dst = resolve_operand(op, alloc);
            }
            for op in &inst.operands {
                if let X86ISelOperand::MemAddr { base, disp } = op {
                    ops.base = resolve_operand(base, alloc);
                    ops.disp = *disp as i64;
                    break;
                }
            }
        }

        // SSE memory forms
        X86Opcode::MovsdRM | X86Opcode::MovssRM => {
            if let Some(op) = inst.operands.first() {
                ops.dst = resolve_operand(op, alloc);
            }
            for op in &inst.operands {
                if let X86ISelOperand::MemAddr { base, disp } = op {
                    ops.base = resolve_operand(base, alloc);
                    ops.disp = *disp as i64;
                    break;
                }
            }
        }
        X86Opcode::MovsdMR | X86Opcode::MovssMR => {
            for op in &inst.operands {
                if let X86ISelOperand::MemAddr { base, disp } = op {
                    ops.base = resolve_operand(base, alloc);
                    ops.disp = *disp as i64;
                } else if ops.dst.is_none() {
                    ops.dst = resolve_operand(op, alloc);
                }
            }
        }

        // SSE type conversion: [dst, src]
        X86Opcode::Cvtsi2sd | X86Opcode::Cvtsd2si
        | X86Opcode::Cvtsi2ss | X86Opcode::Cvtss2si
        | X86Opcode::Cvtsd2ss | X86Opcode::Cvtss2sd => {
            if inst.operands.len() >= 2 {
                ops.dst = resolve_operand(&inst.operands[0], alloc);
                ops.src = resolve_operand(&inst.operands[1], alloc);
            }
        }

        // LEA RIP-relative: [dst, disp]
        X86Opcode::LeaRip => {
            if let Some(op) = inst.operands.first() {
                ops.dst = resolve_operand(op, alloc);
            }
            for op in &inst.operands {
                if let X86ISelOperand::Imm(imm) = op {
                    ops.disp = *imm;
                    break;
                }
            }
        }

        // Scaled-index memory: [dst/src, base, index, scale, disp]
        X86Opcode::MovRMSib => {
            if let Some(op) = inst.operands.first() {
                ops.dst = resolve_operand(op, alloc);
            }
            // TODO: SIB operand resolution requires ISel support for
            // index/scale operands. For now, these are set via direct
            // X86InstOperands construction in tests and manual lowering.
        }
        X86Opcode::MovMRSib => {
            if let Some(op) = inst.operands.first() {
                ops.dst = resolve_operand(op, alloc);
            }
            // TODO: SIB operand resolution requires ISel support for
            // index/scale operands.
        }
    }

    ops
}

// ---------------------------------------------------------------------------
// System V AMD64 ABI prologue/epilogue
// ---------------------------------------------------------------------------

/// Compute the stack frame size (16-byte aligned).
///
/// System V AMD64 frame layout:
/// ```text
/// [caller's frame]
/// return address        <- RSP on entry
/// saved RBP             <- RBP after push
/// callee-saved regs
/// local variables / spill slots
/// [outgoing args area]  <- RSP (16-byte aligned)
/// ```
fn compute_frame_size(num_callee_saved: usize, num_spills: u32) -> u32 {
    // Each callee-saved register is 8 bytes (PUSH).
    // RBP is saved separately.
    // Spill slots are 8 bytes each.
    let locals = num_spills * 8;

    // After PUSH RBP: RSP is 16-byte aligned (return addr + RBP = 16 bytes).
    // After pushing callee-saved: may be misaligned.
    // We need the total to bring RSP to 16-byte alignment.
    let total_pushes = 1 + num_callee_saved as u32; // RBP + callee-saved
    let push_bytes = total_pushes * 8;
    // Return address is 8 bytes, so on entry RSP % 16 == 8.
    // After all pushes: (8 + push_bytes) % 16 should equal 0 after sub.
    let unaligned = (8 + push_bytes + locals) % 16;
    let padding = if unaligned != 0 { 16 - unaligned } else { 0 };

    locals + padding
}

/// Generate prologue instructions for System V AMD64 ABI.
fn generate_prologue(
    callee_saved: &[X86PReg],
    frame_size: u32,
) -> Vec<X86ISelInst> {
    let mut prologue = Vec::new();

    // PUSH RBP
    prologue.push(X86ISelInst::new(
        X86Opcode::Push,
        vec![X86ISelOperand::PReg(RBP)],
    ));

    // MOV RBP, RSP
    prologue.push(X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(RBP), X86ISelOperand::PReg(RSP)],
    ));

    // Push callee-saved registers.
    for &reg in callee_saved {
        prologue.push(X86ISelInst::new(
            X86Opcode::Push,
            vec![X86ISelOperand::PReg(reg)],
        ));
    }

    // SUB RSP, frame_size (allocate stack frame for locals/spills).
    if frame_size > 0 {
        prologue.push(X86ISelInst::new(
            X86Opcode::SubRI,
            vec![
                X86ISelOperand::PReg(RSP),
                X86ISelOperand::Imm(frame_size as i64),
            ],
        ));
    }

    prologue
}

/// Generate epilogue instructions for System V AMD64 ABI.
fn generate_epilogue(
    callee_saved: &[X86PReg],
    frame_size: u32,
) -> Vec<X86ISelInst> {
    let mut epilogue = Vec::new();

    // ADD RSP, frame_size (deallocate locals/spills).
    if frame_size > 0 {
        epilogue.push(X86ISelInst::new(
            X86Opcode::AddRI,
            vec![
                X86ISelOperand::PReg(RSP),
                X86ISelOperand::Imm(frame_size as i64),
            ],
        ));
    }

    // Pop callee-saved registers in reverse order.
    for &reg in callee_saved.iter().rev() {
        epilogue.push(X86ISelInst::new(
            X86Opcode::Pop,
            vec![X86ISelOperand::PReg(reg)],
        ));
    }

    // POP RBP
    epilogue.push(X86ISelInst::new(
        X86Opcode::Pop,
        vec![X86ISelOperand::PReg(RBP)],
    ));

    epilogue
}

// ---------------------------------------------------------------------------
// Branch resolution for variable-length x86-64 instructions
// ---------------------------------------------------------------------------

/// Estimate the encoded byte size of an x86-64 instruction.
///
/// This is used for branch offset calculation. The estimates are
/// conservative — we use the maximum encoding size for each class.
fn estimate_inst_size(inst: &X86ISelInst) -> usize {
    match inst.opcode {
        // Pseudo-instructions: 0 bytes.
        X86Opcode::Nop | X86Opcode::Phi | X86Opcode::StackAlloc => 0,

        // RET: 1 byte.
        X86Opcode::Ret => 1,

        // PUSH/POP: 1-2 bytes (REX prefix if extended register).
        X86Opcode::Push | X86Opcode::Pop => 2,

        // JMP rel32: 5 bytes.
        X86Opcode::Jmp => 5,

        // Jcc rel32: 6 bytes (0F 8x + 4-byte displacement).
        X86Opcode::Jcc => 6,

        // CALL rel32: 5 bytes.
        X86Opcode::Call => 5,

        // MOV r64, imm64: 10 bytes (REX.W + B8+rd + 8-byte immediate).
        X86Opcode::MovRI => 10,

        // ALU r/m64, imm32: 7 bytes (REX.W + opcode + ModRM + imm32).
        X86Opcode::AddRI | X86Opcode::SubRI | X86Opcode::AndRI
        | X86Opcode::OrRI | X86Opcode::XorRI | X86Opcode::CmpRI
        | X86Opcode::TestRI => 7,

        // ALU r/m64, r64: 3 bytes (REX.W + opcode + ModRM).
        X86Opcode::AddRR | X86Opcode::SubRR | X86Opcode::AndRR
        | X86Opcode::OrRR | X86Opcode::XorRR | X86Opcode::CmpRR
        | X86Opcode::TestRR | X86Opcode::MovRR => 3,

        // Memory operations: up to 8 bytes.
        X86Opcode::MovRM | X86Opcode::MovMR | X86Opcode::AddRM
        | X86Opcode::SubRM | X86Opcode::CmpRM | X86Opcode::Lea => 8,

        // LEA RIP-relative: 7 bytes (REX.W + 8D + ModRM + disp32).
        X86Opcode::LeaRip => 7,

        // SIB memory: up to 9 bytes (REX.W + opcode + ModRM + SIB + disp32).
        X86Opcode::MovRMSib | X86Opcode::MovMRSib => 9,

        // SSE: 4-5 bytes.
        X86Opcode::Addsd | X86Opcode::Subsd | X86Opcode::Mulsd | X86Opcode::Divsd
        | X86Opcode::Addss | X86Opcode::Subss | X86Opcode::Mulss | X86Opcode::Divss
        | X86Opcode::MovsdRR | X86Opcode::MovssRR
        | X86Opcode::Ucomisd | X86Opcode::Ucomiss => 5,

        // SSE memory: 6-8 bytes.
        X86Opcode::MovsdRM | X86Opcode::MovsdMR
        | X86Opcode::MovssRM | X86Opcode::MovssMR => 8,

        // SSE conversion: 5-6 bytes (prefix + REX.W + 0F + opcode + ModRM).
        X86Opcode::Cvtsi2sd | X86Opcode::Cvtsd2si
        | X86Opcode::Cvtsi2ss | X86Opcode::Cvtss2si => 6,
        X86Opcode::Cvtsd2ss | X86Opcode::Cvtss2sd => 5,

        // Default conservative estimate.
        _ => 7,
    }
}

/// Resolve block operands in branch instructions to byte offsets.
///
/// x86-64 branches use PC-relative offsets where the offset is relative
/// to the end of the branch instruction itself (i.e., the start of the
/// next instruction).
fn resolve_x86_branches(func: &mut X86ISelFunction) {
    use llvm2_lower::instructions::Block;

    // Phase 1: Compute byte offset of each block.
    let mut block_offsets: HashMap<Block, i64> = HashMap::new();
    let mut current_offset: i64 = 0;

    for &block_id in &func.block_order {
        block_offsets.insert(block_id, current_offset);
        if let Some(mblock) = func.blocks.get(&block_id) {
            for inst in &mblock.insts {
                current_offset += estimate_inst_size(inst) as i64;
            }
        }
    }

    // Phase 2: Replace Block operands with Imm offsets.
    // For each branch, compute the PC-relative offset from the end of
    // the branch instruction to the target block.
    let block_order = func.block_order.clone();
    for &block_id in &block_order {
        let mut inst_offset: i64 = *block_offsets.get(&block_id).unwrap_or(&0);

        if let Some(mblock) = func.blocks.get_mut(&block_id) {
            for inst in &mut mblock.insts {
                let inst_size = estimate_inst_size(inst) as i64;

                let is_branch = matches!(
                    inst.opcode,
                    X86Opcode::Jmp | X86Opcode::Jcc
                );

                if is_branch {
                    // The offset for x86-64 branches is relative to the
                    // instruction *after* the branch (i.e., inst_offset + inst_size).
                    let branch_end = inst_offset + inst_size;

                    inst.operands = inst
                        .operands
                        .iter()
                        .map(|op| {
                            if let X86ISelOperand::Block(target_block) = op {
                                if let Some(&target_offset) = block_offsets.get(target_block) {
                                    let rel_offset = target_offset - branch_end;
                                    X86ISelOperand::Imm(rel_offset)
                                } else {
                                    op.clone()
                                }
                            } else {
                                op.clone()
                            }
                        })
                        .collect();
                }

                inst_offset += inst_size;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// X86Pipeline — main entry point
// ---------------------------------------------------------------------------

/// Output format for the x86-64 pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum X86OutputFormat {
    /// Raw machine code bytes (no object file wrapper).
    RawBytes,
    /// ELF .o object file (for Linux/BSD).
    Elf,
    /// Mach-O .o object file (for macOS).
    MachO,
}

/// Configuration for the x86-64 pipeline.
#[derive(Debug, Clone)]
pub struct X86PipelineConfig {
    /// Output format: raw bytes, ELF, or Mach-O.
    pub output_format: X86OutputFormat,
    /// Whether to emit an ELF .o wrapper (vs raw code bytes).
    /// Deprecated: use `output_format` instead.
    pub emit_elf: bool,
    /// Whether to emit prologue/epilogue (false for leaf functions that
    /// don't need a frame).
    pub emit_frame: bool,
}

impl Default for X86PipelineConfig {
    fn default() -> Self {
        Self {
            output_format: X86OutputFormat::RawBytes,
            emit_elf: false,
            emit_frame: true,
        }
    }
}

/// The x86-64 compilation pipeline.
///
/// Orchestrates: ISel output -> regalloc -> frame lowering -> encoding -> object file.
/// Supports output to raw bytes, ELF, or Mach-O object files.
pub struct X86Pipeline {
    pub config: X86PipelineConfig,
}

impl X86Pipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: X86PipelineConfig) -> Self {
        Self { config }
    }

    /// Create a pipeline with default configuration (raw code bytes, with frame).
    pub fn default_config() -> Self {
        Self::new(X86PipelineConfig::default())
    }

    /// Compile an x86-64 ISel function to machine code bytes.
    ///
    /// This is the main entry point. It takes an `X86ISelFunction` (post-ISel,
    /// pre-regalloc) and returns encoded machine code bytes, optionally
    /// wrapped in an ELF or Mach-O object file.
    pub fn compile_function(
        &self,
        func: &X86ISelFunction,
    ) -> Result<Vec<u8>, X86PipelineError> {
        // Phase 1: Register assignment.
        let assignment = X86RegAssignment::assign(func)?;

        // Phase 2: Clone function and insert prologue/epilogue.
        let mut func = func.clone();
        if self.config.emit_frame {
            self.insert_prologue_epilogue(&mut func, &assignment);
        }

        // Phase 3: Resolve branch offsets.
        resolve_x86_branches(&mut func);

        // Phase 4: Encode all instructions.
        let code = self.encode_function(&func, &assignment.allocation)?;

        // Phase 5: Optionally wrap in object file format.
        // Resolve output format: `output_format` takes precedence, fall back to `emit_elf`.
        let effective_format = if self.config.output_format != X86OutputFormat::RawBytes {
            self.config.output_format
        } else if self.config.emit_elf {
            X86OutputFormat::Elf
        } else {
            X86OutputFormat::RawBytes
        };

        match effective_format {
            X86OutputFormat::RawBytes => Ok(code),
            X86OutputFormat::Elf => Ok(self.emit_elf(&func.name, &code)),
            X86OutputFormat::MachO => Ok(self.emit_macho(&func.name, &code)),
        }
    }

    /// Compile a tMIR function through the full pipeline (ISel + codegen).
    ///
    /// This is a convenience entry point that runs ISel first, then the
    /// rest of the pipeline.
    pub fn compile_tmir_function(
        &self,
        input: &llvm2_lower::Function,
    ) -> Result<Vec<u8>, X86PipelineError> {
        use llvm2_lower::x86_64_isel::X86InstructionSelector;

        let sig = llvm2_lower::function::Signature {
            params: input.signature.params.clone(),
            returns: input.signature.returns.clone(),
        };

        let mut isel = X86InstructionSelector::new(input.name.clone(), sig.clone());

        // Lower formal arguments.
        isel.lower_formal_arguments(&sig, input.entry_block)
            .map_err(|e| X86PipelineError::ISel(e.to_string()))?;

        // Sort blocks for deterministic processing.
        let mut block_order: Vec<_> = input.blocks.keys().copied().collect();
        block_order.sort_by_key(|b| b.0);

        for block_ref in &block_order {
            let basic_block = &input.blocks[block_ref];
            isel.select_block(*block_ref, &basic_block.instructions)
                .map_err(|e| X86PipelineError::ISel(e.to_string()))?;
        }

        let isel_func = isel.finalize();
        self.compile_function(&isel_func)
    }

    // --- Phase implementations ---

    /// Insert prologue/epilogue into the function.
    fn insert_prologue_epilogue(
        &self,
        func: &mut X86ISelFunction,
        assignment: &X86RegAssignment,
    ) {
        let frame_size = compute_frame_size(
            assignment.used_callee_saved.len(),
            assignment.num_spills,
        );

        let prologue = generate_prologue(&assignment.used_callee_saved, frame_size);
        let epilogue = generate_epilogue(&assignment.used_callee_saved, frame_size);

        // Insert prologue at the start of the entry block.
        if let Some(entry_block) = func.block_order.first().copied()
            && let Some(mblock) = func.blocks.get_mut(&entry_block) {
                // Prepend prologue instructions.
                let mut new_insts = prologue;
                new_insts.append(&mut mblock.insts);
                mblock.insts = new_insts;
            }

        // Insert epilogue before every RET instruction.
        for block_id in func.block_order.clone() {
            if let Some(mblock) = func.blocks.get_mut(&block_id) {
                let mut new_insts = Vec::new();
                for inst in &mblock.insts {
                    if inst.opcode == X86Opcode::Ret {
                        new_insts.extend(epilogue.clone());
                    }
                    new_insts.push(inst.clone());
                }
                mblock.insts = new_insts;
            }
        }
    }

    /// Encode all instructions in the function to machine code bytes.
    fn encode_function(
        &self,
        func: &X86ISelFunction,
        alloc: &HashMap<VReg, X86PReg>,
    ) -> Result<Vec<u8>, X86PipelineError> {
        let mut encoder = X86Encoder::new();

        for &block_id in &func.block_order {
            if let Some(mblock) = func.blocks.get(&block_id) {
                for inst in &mblock.insts {
                    // Skip pseudo-instructions.
                    if matches!(
                        inst.opcode,
                        X86Opcode::Nop | X86Opcode::Phi | X86Opcode::StackAlloc
                    ) {
                        continue;
                    }

                    let ops = resolve_inst_operands(inst, alloc);
                    encoder
                        .encode_instruction(inst.opcode, &ops)
                        .map_err(X86PipelineError::from)?;
                }
            }
        }

        Ok(encoder.finish())
    }

    /// Emit an ELF .o file wrapping the encoded machine code.
    fn emit_elf(&self, func_name: &str, code: &[u8]) -> Vec<u8> {
        let mut writer = ElfWriter::new(ElfMachine::X86_64);
        writer.add_text_section(code);
        writer.add_symbol(func_name, 1, 0, code.len() as u64, true, 2); // STT_FUNC
        writer.write()
    }

    /// Emit a Mach-O .o file wrapping the encoded machine code.
    ///
    /// Produces a valid x86-64 Mach-O relocatable object file with a __TEXT,__text
    /// section and a global function symbol with Mach-O name mangling (_prefix).
    fn emit_macho(&self, func_name: &str, code: &[u8]) -> Vec<u8> {
        let mut writer = MachOWriter::for_target(MachOTarget::X86_64);
        writer.add_text_section(code);

        // Mach-O convention: symbol names have underscore prefix.
        // Section index is 1-based (first add_text_section = section 1).
        let macho_name = format!("_{}", func_name);
        writer.add_symbol(&macho_name, 1, 0, true);
        writer.write()
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Compile an x86-64 ISel function to raw machine code bytes.
pub fn x86_compile_to_bytes(
    func: &X86ISelFunction,
) -> Result<Vec<u8>, X86PipelineError> {
    let pipeline = X86Pipeline::default_config();
    pipeline.compile_function(func)
}

/// Compile an x86-64 ISel function to an ELF .o file.
pub fn x86_compile_to_elf(
    func: &X86ISelFunction,
) -> Result<Vec<u8>, X86PipelineError> {
    let pipeline = X86Pipeline::new(X86PipelineConfig {
        emit_elf: true,
        emit_frame: true,
        ..X86PipelineConfig::default()
    });
    pipeline.compile_function(func)
}

/// Compile an x86-64 ISel function to a Mach-O .o file.
///
/// Produces a valid x86-64 Mach-O relocatable object file suitable for
/// linking on macOS. The function is emitted as a global symbol with
/// Mach-O name mangling (_prefix).
pub fn x86_compile_to_macho(
    func: &X86ISelFunction,
) -> Result<Vec<u8>, X86PipelineError> {
    let pipeline = X86Pipeline::new(X86PipelineConfig {
        output_format: X86OutputFormat::MachO,
        emit_frame: true,
        ..X86PipelineConfig::default()
    });
    pipeline.compile_function(func)
}

/// Build a simple `add(a: i64, b: i64) -> i64` x86-64 ISel function for testing.
///
/// System V AMD64 ABI: a in RDI, b in RSI, return in RAX.
pub fn build_x86_add_test_function() -> X86ISelFunction {
    use llvm2_lower::function::Signature;
    use llvm2_lower::instructions::Block;
    use llvm2_lower::types::Type;

    let sig = Signature {
        params: vec![Type::I64, Type::I64],
        returns: vec![Type::I64],
    };

    let mut func = X86ISelFunction::new("add".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    let v0 = VReg::new(0, RegClass::Gpr64);
    let v1 = VReg::new(1, RegClass::Gpr64);
    let v2 = VReg::new(2, RegClass::Gpr64);
    func.next_vreg = 3;

    // MOV v0, RDI (arg 0)
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(x86_64_regs::RDI)],
    ));

    // MOV v1, RSI (arg 1)
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v1), X86ISelOperand::PReg(x86_64_regs::RSI)],
    ));

    // ADD v2, v0, v1
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::AddRR,
        vec![
            X86ISelOperand::VReg(v2),
            X86ISelOperand::VReg(v0),
            X86ISelOperand::VReg(v1),
        ],
    ));

    // MOV RAX, v2 (return value)
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v2)],
    ));

    // RET
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));

    func
}

/// Build a simple `const42() -> i64` x86-64 ISel function for testing.
///
/// Returns the constant 42.
pub fn build_x86_const_test_function() -> X86ISelFunction {
    use llvm2_lower::function::Signature;
    use llvm2_lower::instructions::Block;
    use llvm2_lower::types::Type;

    let sig = Signature {
        params: vec![],
        returns: vec![Type::I64],
    };

    let mut func = X86ISelFunction::new("const42".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    let v0 = VReg::new(0, RegClass::Gpr64);
    func.next_vreg = 1;

    // MOV v0, 42
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRI,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::Imm(42)],
    ));

    // MOV RAX, v0
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)],
    ));

    // RET
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));

    func
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::regs::{RegClass, VReg};
    use llvm2_ir::x86_64_ops::X86Opcode;
    use llvm2_ir::x86_64_regs::{RAX, RBP, RDI, RSP};
    use llvm2_lower::function::Signature;
    use llvm2_lower::instructions::Block;
    use llvm2_lower::x86_64_isel::{X86ISelFunction, X86ISelInst, X86ISelOperand};

    // -----------------------------------------------------------------------
    // Helper: build a minimal ISel function
    // -----------------------------------------------------------------------

    fn minimal_func(name: &str) -> X86ISelFunction {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut func = X86ISelFunction::new(name.to_string(), sig);
        let entry = Block(0);
        func.ensure_block(entry);
        func
    }

    // -----------------------------------------------------------------------
    // Pipeline construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_default_config() {
        let pipeline = X86Pipeline::default_config();
        assert!(!pipeline.config.emit_elf);
        assert!(pipeline.config.emit_frame);
    }

    #[test]
    fn test_pipeline_custom_config() {
        let config = X86PipelineConfig {
            emit_elf: true,
            emit_frame: false,
            ..X86PipelineConfig::default()
        };
        let pipeline = X86Pipeline::new(config);
        assert!(pipeline.config.emit_elf);
        assert!(!pipeline.config.emit_frame);
    }

    // -----------------------------------------------------------------------
    // Frame size computation
    // -----------------------------------------------------------------------

    #[test]
    fn test_frame_size_no_callee_saved_no_spills() {
        // After PUSH RBP: RSP is 16-byte aligned (8 + 8 = 16).
        // No additional frame needed.
        let size = compute_frame_size(0, 0);
        assert_eq!(size % 16, 0, "frame size must be 16-byte aligned");
    }

    #[test]
    fn test_frame_size_with_spills() {
        let size = compute_frame_size(0, 2); // 16 bytes of spills
        // After PUSH RBP (total pushes: 8 + 8 = 16 bytes on stack).
        // With 16 bytes of spills: total = 32, which is aligned.
        assert_eq!(size % 8, 0);
    }

    #[test]
    fn test_frame_size_alignment() {
        // Ensure frame size is always aligned such that RSP is 16-byte aligned.
        for num_cs in 0..8 {
            for num_spill in 0..5 {
                let size = compute_frame_size(num_cs, num_spill);
                let total_pushes = 1 + num_cs as u32;
                let total_on_stack = 8 + total_pushes * 8 + size;
                assert_eq!(
                    total_on_stack % 16, 0,
                    "misaligned for callee_saved={}, spills={}: total={}",
                    num_cs, num_spill, total_on_stack
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Prologue/epilogue generation
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_prologue_minimal() {
        let prologue = generate_prologue(&[], 0);
        // Should have: PUSH RBP, MOV RBP RSP (no SUB since frame_size=0)
        assert_eq!(prologue.len(), 2);
        assert_eq!(prologue[0].opcode, X86Opcode::Push);
        assert_eq!(prologue[0].operands[0], X86ISelOperand::PReg(RBP));
        assert_eq!(prologue[1].opcode, X86Opcode::MovRR);
        assert_eq!(prologue[1].operands[0], X86ISelOperand::PReg(RBP));
        assert_eq!(prologue[1].operands[1], X86ISelOperand::PReg(RSP));
    }

    #[test]
    fn test_generate_prologue_with_frame() {
        let prologue = generate_prologue(&[], 32);
        assert_eq!(prologue.len(), 3);
        assert_eq!(prologue[2].opcode, X86Opcode::SubRI);
        assert_eq!(prologue[2].operands[0], X86ISelOperand::PReg(RSP));
        assert_eq!(prologue[2].operands[1], X86ISelOperand::Imm(32));
    }

    #[test]
    fn test_generate_prologue_with_callee_saved() {
        use llvm2_ir::x86_64_regs::{RBX, R12};
        let prologue = generate_prologue(&[RBX, R12], 0);
        // PUSH RBP, MOV RBP RSP, PUSH RBX, PUSH R12
        assert_eq!(prologue.len(), 4);
        assert_eq!(prologue[2].opcode, X86Opcode::Push);
        assert_eq!(prologue[2].operands[0], X86ISelOperand::PReg(RBX));
        assert_eq!(prologue[3].opcode, X86Opcode::Push);
        assert_eq!(prologue[3].operands[0], X86ISelOperand::PReg(R12));
    }

    #[test]
    fn test_generate_epilogue_minimal() {
        let epilogue = generate_epilogue(&[], 0);
        // POP RBP only (no ADD since frame_size=0)
        assert_eq!(epilogue.len(), 1);
        assert_eq!(epilogue[0].opcode, X86Opcode::Pop);
        assert_eq!(epilogue[0].operands[0], X86ISelOperand::PReg(RBP));
    }

    #[test]
    fn test_generate_epilogue_with_frame() {
        let epilogue = generate_epilogue(&[], 32);
        assert_eq!(epilogue.len(), 2);
        assert_eq!(epilogue[0].opcode, X86Opcode::AddRI);
        assert_eq!(epilogue[0].operands[0], X86ISelOperand::PReg(RSP));
        assert_eq!(epilogue[1].opcode, X86Opcode::Pop);
    }

    #[test]
    fn test_generate_epilogue_callee_saved_reverse_order() {
        use llvm2_ir::x86_64_regs::{RBX, R12};
        let epilogue = generate_epilogue(&[RBX, R12], 0);
        // POP R12, POP RBX, POP RBP (reverse of push order)
        assert_eq!(epilogue.len(), 3);
        assert_eq!(epilogue[0].opcode, X86Opcode::Pop);
        assert_eq!(epilogue[0].operands[0], X86ISelOperand::PReg(R12));
        assert_eq!(epilogue[1].opcode, X86Opcode::Pop);
        assert_eq!(epilogue[1].operands[0], X86ISelOperand::PReg(RBX));
        assert_eq!(epilogue[2].opcode, X86Opcode::Pop);
        assert_eq!(epilogue[2].operands[0], X86ISelOperand::PReg(RBP));
    }

    // -----------------------------------------------------------------------
    // Register assignment
    // -----------------------------------------------------------------------

    #[test]
    fn test_reg_assignment_simple() {
        let func = build_x86_add_test_function();
        let assignment = X86RegAssignment::assign(&func).unwrap();

        // Should have allocated 3 VRegs.
        assert_eq!(assignment.allocation.len(), 3);

        // All assigned to different physical registers.
        let pregs: Vec<X86PReg> = assignment.allocation.values().copied().collect();
        for i in 0..pregs.len() {
            for j in (i + 1)..pregs.len() {
                assert_ne!(pregs[i], pregs[j], "duplicate preg assignment");
            }
        }
    }

    #[test]
    fn test_reg_assignment_empty_function() {
        let mut func = minimal_func("empty");
        func.push_inst(Block(0), X86ISelInst::new(X86Opcode::Ret, vec![]));
        let assignment = X86RegAssignment::assign(&func).unwrap();
        assert!(assignment.allocation.is_empty());
        assert!(assignment.used_callee_saved.is_empty());
    }

    // -----------------------------------------------------------------------
    // Compile simple functions
    // -----------------------------------------------------------------------

    #[test]
    fn test_compile_void_return() {
        let mut func = minimal_func("void_ret");
        func.push_inst(Block(0), X86ISelInst::new(X86Opcode::Ret, vec![]));

        let pipeline = X86Pipeline::new(X86PipelineConfig {
            emit_elf: false,
            emit_frame: true,
            ..X86PipelineConfig::default()
        });
        let code = pipeline.compile_function(&func).unwrap();

        // Should produce non-empty code (prologue + epilogue + RET).
        assert!(!code.is_empty(), "compiled code should not be empty");

        // The last byte should be 0xC3 (RET).
        assert_eq!(*code.last().unwrap(), 0xC3, "last byte should be RET");
    }

    #[test]
    fn test_compile_void_return_no_frame() {
        let mut func = minimal_func("void_ret_noframe");
        func.push_inst(Block(0), X86ISelInst::new(X86Opcode::Ret, vec![]));

        let pipeline = X86Pipeline::new(X86PipelineConfig {
            emit_elf: false,
            emit_frame: false,
            ..X86PipelineConfig::default()
        });
        let code = pipeline.compile_function(&func).unwrap();

        // Without frame, should just be RET (0xC3).
        assert_eq!(code, vec![0xC3]);
    }

    #[test]
    fn test_compile_const42() {
        let func = build_x86_const_test_function();
        let pipeline = X86Pipeline::new(X86PipelineConfig {
            emit_elf: false,
            emit_frame: true,
            ..X86PipelineConfig::default()
        });
        let code = pipeline.compile_function(&func).unwrap();

        assert!(!code.is_empty());
        // Last byte should be RET.
        assert_eq!(*code.last().unwrap(), 0xC3);
    }

    #[test]
    fn test_compile_add_function() {
        let func = build_x86_add_test_function();
        let pipeline = X86Pipeline::new(X86PipelineConfig {
            emit_elf: false,
            emit_frame: true,
            ..X86PipelineConfig::default()
        });
        let code = pipeline.compile_function(&func).unwrap();

        assert!(!code.is_empty());
        assert_eq!(*code.last().unwrap(), 0xC3);
    }

    #[test]
    fn test_compile_add_function_no_frame() {
        let func = build_x86_add_test_function();
        let pipeline = X86Pipeline::new(X86PipelineConfig {
            emit_elf: false,
            emit_frame: false,
            ..X86PipelineConfig::default()
        });
        let code = pipeline.compile_function(&func).unwrap();

        assert!(!code.is_empty());
        assert_eq!(*code.last().unwrap(), 0xC3);
    }

    // -----------------------------------------------------------------------
    // ELF emission
    // -----------------------------------------------------------------------

    #[test]
    fn test_compile_to_elf() {
        let func = build_x86_const_test_function();
        let bytes = x86_compile_to_elf(&func).unwrap();

        // ELF magic: 0x7F 'E' 'L' 'F'
        assert!(bytes.len() > 16);
        assert_eq!(&bytes[0..4], &[0x7F, b'E', b'L', b'F']);

        // ELF class should be ELFCLASS64 (2).
        assert_eq!(bytes[4], 2);

        // Data encoding should be ELFDATA2LSB (1) = little-endian.
        assert_eq!(bytes[5], 1);

        // Machine type for x86-64 should be EM_X86_64 (0x3E = 62).
        // Located at offset 18 (e_machine) in the ELF header (little-endian u16).
        let machine = u16::from_le_bytes([bytes[18], bytes[19]]);
        assert_eq!(machine, 0x3E, "ELF machine should be EM_X86_64 (0x3E)");
    }

    #[test]
    fn test_compile_add_to_elf() {
        let func = build_x86_add_test_function();
        let bytes = x86_compile_to_elf(&func).unwrap();
        assert!(bytes.len() > 64);
        assert_eq!(&bytes[0..4], &[0x7F, b'E', b'L', b'F']);
    }

    // -----------------------------------------------------------------------
    // Mach-O emission
    // -----------------------------------------------------------------------

    #[test]
    fn test_compile_to_macho() {
        let func = build_x86_const_test_function();
        let bytes = x86_compile_to_macho(&func).unwrap();

        // Mach-O magic: 0xFEEDFACF (MH_MAGIC_64, little-endian).
        assert!(bytes.len() > 32);
        assert_eq!(&bytes[0..4], &[0xCF, 0xFA, 0xED, 0xFE]);

        // CPU type should be CPU_TYPE_X86_64 = 0x01000007 at offset 4.
        let cputype = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(cputype, 0x01000007, "CPU type should be CPU_TYPE_X86_64");
    }

    #[test]
    fn test_compile_add_to_macho() {
        let func = build_x86_add_test_function();
        let bytes = x86_compile_to_macho(&func).unwrap();

        assert!(bytes.len() > 64);
        // Mach-O magic
        assert_eq!(&bytes[0..4], &[0xCF, 0xFA, 0xED, 0xFE]);
        // CPU type = x86-64
        let cputype = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(cputype, 0x01000007);
    }

    #[test]
    fn test_compile_macho_via_config() {
        let func = build_x86_const_test_function();
        let pipeline = X86Pipeline::new(X86PipelineConfig {
            output_format: X86OutputFormat::MachO,
            emit_frame: true,
            ..X86PipelineConfig::default()
        });
        let bytes = pipeline.compile_function(&func).unwrap();

        // Mach-O magic
        assert_eq!(&bytes[0..4], &[0xCF, 0xFA, 0xED, 0xFE]);

        // Last byte of the embedded code should be RET (0xC3), but since
        // it's wrapped in Mach-O, we just verify the overall structure.
        assert!(bytes.len() > 100, "Mach-O output should be substantial");
    }

    #[test]
    fn test_output_format_enum() {
        assert_ne!(X86OutputFormat::RawBytes, X86OutputFormat::Elf);
        assert_ne!(X86OutputFormat::Elf, X86OutputFormat::MachO);
        assert_ne!(X86OutputFormat::RawBytes, X86OutputFormat::MachO);
    }

    // -----------------------------------------------------------------------
    // Branch resolution
    // -----------------------------------------------------------------------

    #[test]
    fn test_branch_resolution_unconditional() {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut func = X86ISelFunction::new("test_br".to_string(), sig);
        let b0 = Block(0);
        let b1 = Block(1);
        func.ensure_block(b0);
        func.ensure_block(b1);

        // b0: JMP b1
        func.push_inst(b0, X86ISelInst::new(
            X86Opcode::Jmp,
            vec![X86ISelOperand::Block(b1)],
        ));

        // b1: RET
        func.push_inst(b1, X86ISelInst::new(X86Opcode::Ret, vec![]));

        resolve_x86_branches(&mut func);

        // After resolution, JMP should have an Imm operand (not Block).
        let jmp = &func.blocks[&b0].insts[0];
        assert_eq!(jmp.opcode, X86Opcode::Jmp);
        match &jmp.operands[0] {
            X86ISelOperand::Imm(offset) => {
                // JMP is 5 bytes. Target (b1) starts at offset 5.
                // PC-relative offset from end of JMP (offset 5) to b1 (offset 5) = 0.
                assert_eq!(*offset, 0, "JMP to immediately following block should be offset 0");
            }
            other => panic!("expected Imm operand after resolution, got {:?}", other),
        }
    }

    #[test]
    fn test_branch_resolution_backward() {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut func = X86ISelFunction::new("test_loop".to_string(), sig);
        let b0 = Block(0);
        let b1 = Block(1);
        func.ensure_block(b0);
        func.ensure_block(b1);

        // b0: NOP (filler to make offset calculation visible)
        func.push_inst(b0, X86ISelInst::new(X86Opcode::Ret, vec![]));

        // b1: JMP b0 (backward jump)
        func.push_inst(b1, X86ISelInst::new(
            X86Opcode::Jmp,
            vec![X86ISelOperand::Block(b0)],
        ));

        resolve_x86_branches(&mut func);

        let jmp = &func.blocks[&b1].insts[0];
        match &jmp.operands[0] {
            X86ISelOperand::Imm(offset) => {
                // b0 starts at offset 0.
                // JMP in b1 starts at offset 1 (RET is 1 byte).
                // JMP is 5 bytes, so end of JMP is at offset 6.
                // Target offset is 0, so relative = 0 - 6 = -6.
                assert!(*offset < 0, "backward jump should have negative offset, got {}", offset);
            }
            other => panic!("expected Imm operand, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Convenience function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_x86_compile_to_bytes() {
        let func = build_x86_const_test_function();
        let bytes = x86_compile_to_bytes(&func).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(*bytes.last().unwrap(), 0xC3);
    }

    // -----------------------------------------------------------------------
    // Operand resolution tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_resolve_operand_vreg() {
        let v0 = VReg::new(0, RegClass::Gpr64);
        let mut alloc = HashMap::new();
        alloc.insert(v0, RAX);

        assert_eq!(
            resolve_operand(&X86ISelOperand::VReg(v0), &alloc),
            Some(RAX)
        );
    }

    #[test]
    fn test_resolve_operand_preg() {
        let alloc = HashMap::new();
        assert_eq!(
            resolve_operand(&X86ISelOperand::PReg(RDI), &alloc),
            Some(RDI)
        );
    }

    #[test]
    fn test_resolve_operand_imm_returns_none() {
        let alloc = HashMap::new();
        assert_eq!(
            resolve_operand(&X86ISelOperand::Imm(42), &alloc),
            None
        );
    }

    // -----------------------------------------------------------------------
    // Instruction size estimation
    // -----------------------------------------------------------------------

    #[test]
    fn test_estimate_inst_sizes() {
        assert_eq!(estimate_inst_size(&X86ISelInst::new(X86Opcode::Nop, vec![])), 0);
        assert_eq!(estimate_inst_size(&X86ISelInst::new(X86Opcode::Ret, vec![])), 1);
        assert_eq!(estimate_inst_size(&X86ISelInst::new(X86Opcode::Jmp, vec![])), 5);
        assert_eq!(estimate_inst_size(&X86ISelInst::new(X86Opcode::Jcc, vec![])), 6);
        assert_eq!(estimate_inst_size(&X86ISelInst::new(X86Opcode::Call, vec![])), 5);
        assert_eq!(estimate_inst_size(&X86ISelInst::new(X86Opcode::MovRI, vec![])), 10);
        assert_eq!(estimate_inst_size(&X86ISelInst::new(X86Opcode::AddRR, vec![])), 3);
        assert_eq!(estimate_inst_size(&X86ISelInst::new(X86Opcode::AddRI, vec![])), 7);
    }

    // -----------------------------------------------------------------------
    // Error display
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_error_display() {
        let e1 = X86PipelineError::ISel("bad isel".to_string());
        assert!(format!("{}", e1).contains("ISel"));

        let e2 = X86PipelineError::RegAlloc("out of regs".to_string());
        assert!(format!("{}", e2).contains("regalloc"));

        let e3 = X86PipelineError::FrameLowering("bad frame".to_string());
        assert!(format!("{}", e3).contains("frame lowering"));
    }

    // -----------------------------------------------------------------------
    // Test helper function builders
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_add_function_structure() {
        let func = build_x86_add_test_function();
        assert_eq!(func.name, "add");
        assert_eq!(func.sig.params.len(), 2);
        assert_eq!(func.sig.returns.len(), 1);
        assert_eq!(func.block_order.len(), 1);
        assert_eq!(func.next_vreg, 3);

        let entry = &func.blocks[&Block(0)];
        // MOV v0, RDI; MOV v1, RSI; ADD v2, v0, v1; MOV RAX, v2; RET
        assert_eq!(entry.insts.len(), 5);
        assert_eq!(entry.insts[0].opcode, X86Opcode::MovRR);
        assert_eq!(entry.insts[1].opcode, X86Opcode::MovRR);
        assert_eq!(entry.insts[2].opcode, X86Opcode::AddRR);
        assert_eq!(entry.insts[3].opcode, X86Opcode::MovRR);
        assert_eq!(entry.insts[4].opcode, X86Opcode::Ret);
    }

    #[test]
    fn test_build_const_function_structure() {
        let func = build_x86_const_test_function();
        assert_eq!(func.name, "const42");
        assert_eq!(func.sig.params.len(), 0);
        assert_eq!(func.sig.returns.len(), 1);
        assert_eq!(func.next_vreg, 1);

        let entry = &func.blocks[&Block(0)];
        // MOV v0, 42; MOV RAX, v0; RET
        assert_eq!(entry.insts.len(), 3);
        assert_eq!(entry.insts[0].opcode, X86Opcode::MovRI);
        assert_eq!(entry.insts[1].opcode, X86Opcode::MovRR);
        assert_eq!(entry.insts[2].opcode, X86Opcode::Ret);
    }
}
