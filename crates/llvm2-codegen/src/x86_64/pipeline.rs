// llvm2-codegen/x86_64/pipeline.rs - x86-64 end-to-end compilation pipeline
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
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
//! Phase 2: Register allocation (llvm2-regalloc, greedy by default)
//!   X86ISelFunction -> RegAllocFunction -> AllocationResult (VReg -> X86PReg)
//!   Full pipeline: phi elimination, liveness, coalescing, allocation, spills
//!
//! Phase 3: Two-address fixup (post-regalloc)
//!   Insert MOV copies for x86-64 two-address constraint (dst != lhs)
//!
//! Phase 4: x86-64 prologue/epilogue insertion
//!   Stack frame setup/teardown for System V AMD64 ABI
//!
//! Phase 5: Branch resolution
//!   Resolve block references to byte offsets (variable-length encoding)
//!
//! Phase 6: Encoding (llvm2-codegen/x86_64/encode)
//!   X86ISelFunction -> Vec<u8> (machine code bytes)
//!
//! Phase 7: Object file emission (optional)
//!   Vec<u8> -> ELF or Mach-O .o file bytes
//! ```
//!
//! # Register allocation
//!
//! The pipeline uses the full `llvm2-regalloc` allocator (greedy by default)
//! via the x86-64 adapter (`llvm2_regalloc::x86_adapter`). The ISel function
//! is converted to a `RegAllocFunction`, processed through the full pipeline
//! (phi elimination, liveness analysis, copy coalescing, allocation, spill
//! code generation), and the results are translated back to x86-64 physical
//! registers.
//!
//! A simplified first-appearance allocator (`X86RegAllocMode::Simplified`)
//! is retained for O0/testing scenarios where full regalloc overhead is
//! unnecessary.

use std::collections::HashMap;

use llvm2_ir::regs::{RegClass, VReg};
use llvm2_ir::x86_64_ops::X86Opcode;
use llvm2_ir::x86_64_regs::{
    self, X86PReg, RBP, RSP, R11, XMM15,
    X86_ARG_GPRS, X86_ARG_XMMS,
    X86_ALLOCATABLE_GPRS, X86_ALLOCATABLE_XMMS,
    X86_CALLEE_SAVED_GPRS,
};

use llvm2_lower::x86_64_isel::{
    X86ISelFunction, X86ISelInst, X86ISelOperand,
};

use crate::dwarf_cfi::{DwarfCfiSection, x86_64_fde_from_prologue};
use crate::x86_64::encode::{X86EncodeError, X86Encoder, X86InstOperands};
use crate::elf::{ElfMachine, ElfWriter};
use crate::macho::writer::{MachOTarget, MachOWriter};

// Full regalloc integration (greedy allocator via x86_adapter).
use llvm2_regalloc::{
    AllocStrategy,
    x86_64_greedy_alloc_config, x86_64_alloc_config,
    translate_allocation, x86_to_preg,
    RegAllocFunction, RegAllocBlock, RegAllocInst, RegAllocOperand,
    BlockId, InstId, StackSlotId,
};

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Cross-function call fixup
// ---------------------------------------------------------------------------

/// A pending relocation for an x86-64 `CALL rel32` instruction whose target
/// is a named symbol (typically another function in the same module).
///
/// Emitted by [`X86Pipeline::encode_function_with_all_fixups`]. During
/// multi-function object emission, the caller concatenates function code
/// into a single `__text` / `.text` section, resolves callee names to
/// symbol-table indices, and converts each fixup into a Mach-O
/// `X86_64_RELOC_BRANCH` or ELF `R_X86_64_PLT32` relocation.
///
/// See #464: multi-function module support in the x86-64 dispatcher.
#[derive(Debug, Clone)]
pub struct X86CallFixup {
    /// Byte offset of the 4-byte `disp32` field within the encoded function
    /// code (i.e., position immediately after the `E8` opcode byte).
    pub offset: usize,
    /// Name of the callee symbol (unmangled; the Mach-O / ELF emitter adds
    /// any platform-specific prefix such as `_`).
    pub callee: String,
}

/// Per-function bookkeeping produced by [`X86Pipeline::compile_module`].
///
/// Records where each function lives inside the combined `__text` / `.text`
/// section and what CALL fixups it emitted, so the module-level object
/// emitter can resolve cross-function calls into section-scoped
/// relocations.
#[derive(Debug, Clone)]
struct X86EncodedFunc {
    /// Unmangled function name (as declared in the tMIR module).
    name: String,
    /// Length of the function's code bytes in the combined section.
    #[allow(dead_code)]
    code_len: usize,
    /// Length of the function's per-function constant pool appended
    /// immediately after its code bytes.
    #[allow(dead_code)]
    const_pool_len: usize,
    /// CALL-site fixups with offsets local to this function's code range.
    call_fixups: Vec<X86CallFixup>,
    /// Byte offset where this function's code begins in the combined
    /// `__text` / `.text` section.
    text_offset: u64,
}

// ---------------------------------------------------------------------------
// Register allocation mode
// ---------------------------------------------------------------------------

/// Register allocation mode for the x86-64 pipeline.
///
/// Controls whether the pipeline uses the simplified first-appearance allocator
/// or the full `llvm2-regalloc` pipeline (with liveness analysis, coalescing,
/// spill code generation, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum X86RegAllocMode {
    /// Simplified linear-scan assignment (no liveness analysis, no spill code).
    /// Fast but limited: assigns VRegs to physical registers in order of first
    /// appearance. Suitable for O0 or simple test functions.
    Simplified,
    /// Full register allocation via llvm2-regalloc (linear scan or greedy).
    /// Requires converting X86ISelFunction to RegAllocFunction, running the
    /// allocator, and translating results back.
    Full(AllocStrategy),
}

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
/// This is a fast fallback used when `X86RegAllocMode::Simplified` is selected
/// (e.g., O0 builds). The default pipeline uses the full regalloc instead.
/// Assigns VRegs to physical registers in order of first appearance, with no
/// liveness analysis or spill code generation.
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
    ///
    /// This simplified allocator assigns VRegs to physical registers in order
    /// of first appearance. It handles register hints from `MOV vreg, PReg(X)`
    /// patterns: when a VReg is first defined as a copy from a physical register,
    /// it is assigned to that same physical register. This prevents clobbering
    /// arg registers (RDI, RSI, RDX, RCX, R8, R9) before they are read.
    ///
    /// Without hints, `MOV v1, RSI` might assign v1 -> RCX, clobbering the
    /// arg3 register before `MOV v3, RCX` reads it. With hints, v1 -> RSI
    /// so the MOV becomes a nop and RCX is preserved.
    pub fn assign(func: &X86ISelFunction) -> Result<Self, X86PipelineError> {
        let mut allocation: HashMap<VReg, X86PReg> = HashMap::new();
        let mut used_gprs: HashSet<X86PReg> = HashSet::new();
        let mut used_xmms: HashSet<X86PReg> = HashSet::new();
        let mut used_callee_saved: Vec<X86PReg> = Vec::new();

        // Pass 1: Collect register hints from `MOV vreg, PReg(X)` instructions.
        // These are typically argument loads from physical registers.
        let mut hints: HashMap<VReg, X86PReg> = HashMap::new();
        for block in func.block_order.iter() {
            if let Some(mblock) = func.blocks.get(block) {
                for inst in &mblock.insts {
                    let is_mov_rr = inst.opcode == X86Opcode::MovRR;
                    let is_movsd_rr = inst.opcode == X86Opcode::MovsdRR;
                    if (is_mov_rr || is_movsd_rr) && inst.operands.len() >= 2 {
                        if let (X86ISelOperand::VReg(dst), X86ISelOperand::PReg(src)) =
                            (&inst.operands[0], &inst.operands[1])
                        {
                            // Hint: assign dst vreg to src physical register.
                            hints.entry(*dst).or_insert(*src);
                        }
                    }
                }
            }
        }

        // Pass 2: Collect all VRegs in order of first appearance.
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

        // Pass 3: Assign hinted vregs first (to claim their preferred registers).
        for vreg in &vregs {
            if let Some(&hint_preg) = hints.get(vreg) {
                let is_allocatable_gpr = X86_ALLOCATABLE_GPRS.contains(&hint_preg);
                let is_allocatable_xmm = X86_ALLOCATABLE_XMMS.contains(&hint_preg);

                if is_allocatable_gpr && !used_gprs.contains(&hint_preg) {
                    allocation.insert(*vreg, hint_preg);
                    used_gprs.insert(hint_preg);
                    if X86_CALLEE_SAVED_GPRS.contains(&hint_preg)
                        && !used_callee_saved.contains(&hint_preg)
                    {
                        used_callee_saved.push(hint_preg);
                    }
                } else if is_allocatable_xmm && !used_xmms.contains(&hint_preg) {
                    allocation.insert(*vreg, hint_preg);
                    used_xmms.insert(hint_preg);
                }
                // If the hint can't be satisfied (already taken or not allocatable),
                // the vreg will be assigned in Pass 4.
            }
        }

        // Pass 4: Assign remaining vregs (no hint or hint unsatisfied).
        let mut gpr_idx: usize = 0;
        let mut xmm_idx: usize = 0;
        for vreg in &vregs {
            if allocation.contains_key(vreg) {
                continue; // Already assigned via hint.
            }

            let is_fp = matches!(
                vreg.class,
                RegClass::Fpr32 | RegClass::Fpr64 | RegClass::Fpr128
            );

            if is_fp {
                // Find next available XMM.
                while xmm_idx < X86_ALLOCATABLE_XMMS.len()
                    && used_xmms.contains(&X86_ALLOCATABLE_XMMS[xmm_idx])
                {
                    xmm_idx += 1;
                }
                if xmm_idx < X86_ALLOCATABLE_XMMS.len() {
                    let preg = X86_ALLOCATABLE_XMMS[xmm_idx];
                    allocation.insert(*vreg, preg);
                    used_xmms.insert(preg);
                    xmm_idx += 1;
                } else {
                    return Err(X86PipelineError::RegAlloc(format!(
                        "ran out of XMM registers for vreg v{}",
                        vreg.id
                    )));
                }
            } else {
                // Find next available GPR.
                while gpr_idx < X86_ALLOCATABLE_GPRS.len()
                    && used_gprs.contains(&X86_ALLOCATABLE_GPRS[gpr_idx])
                {
                    gpr_idx += 1;
                }
                if gpr_idx < X86_ALLOCATABLE_GPRS.len() {
                    let preg = X86_ALLOCATABLE_GPRS[gpr_idx];
                    allocation.insert(*vreg, preg);
                    used_gprs.insert(preg);

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

        // CDQ/CQO: no register operands (implicit RAX -> RDX:RAX)
        X86Opcode::Cdq | X86Opcode::Cqo => {}

        // NopMulti: imm controls NOP size (2-9 bytes)
        X86Opcode::NopMulti => {
            for op in &inst.operands {
                if let X86ISelOperand::Imm(imm) = op {
                    ops.imm = *imm;
                    break;
                }
            }
        }

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

        // CMP r, imm32 / CMP r, imm8
        X86Opcode::CmpRI | X86Opcode::CmpRI8 | X86Opcode::TestRI => {
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
        X86Opcode::CallM => {
            for op in &inst.operands {
                if let X86ISelOperand::MemAddr { base, disp } = op {
                    ops.base = resolve_operand(base, alloc);
                    ops.disp = *disp as i64;
                    break;
                }
            }
        }

        // SSE arithmetic: three-address ISel [dst, lhs, rhs] or two-address [dst, src]
        X86Opcode::Addsd | X86Opcode::Subsd | X86Opcode::Mulsd | X86Opcode::Divsd
        | X86Opcode::Addss | X86Opcode::Subss | X86Opcode::Mulss | X86Opcode::Divss => {
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

        // SSE moves and compares: always [dst, src]
        X86Opcode::MovsdRR | X86Opcode::MovssRR
        | X86Opcode::Ucomisd | X86Opcode::Ucomiss => {
            if inst.operands.len() >= 2 {
                ops.dst = resolve_operand(&inst.operands[0], alloc);
                ops.src = resolve_operand(&inst.operands[1], alloc);
            }
        }

        // SSE RIP-relative constant pool load: [dst, ConstPoolEntry(idx)]
        // The disp field stores the constant pool entry index as a placeholder.
        // The actual RIP-relative displacement is fixed up after encoding.
        X86Opcode::MovssRipRel | X86Opcode::MovsdRipRel => {
            if let Some(op) = inst.operands.first() {
                ops.dst = resolve_operand(op, alloc);
            }
            // Extract constant pool entry index from operands.
            for op in &inst.operands {
                if let X86ISelOperand::ConstPoolEntry(idx) = op {
                    // Store the entry index in imm for now; the pipeline will
                    // convert this to a proper RIP-relative displacement during
                    // the constant pool fixup phase.
                    ops.imm = *idx as i64;
                    break;
                }
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
        X86Opcode::Movzx | X86Opcode::MovzxW | X86Opcode::MovsxB
        | X86Opcode::MovsxW | X86Opcode::Movsx => {
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

        // BT r, imm8: [reg, imm]
        X86Opcode::BtRI => {
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

        // BSWAP: single register [dst]
        X86Opcode::Bswap => {
            if let Some(op) = inst.operands.first() {
                ops.dst = resolve_operand(op, alloc);
            }
        }

        // XCHG, CMPXCHG: register-register [dst, src]
        X86Opcode::Xchg | X86Opcode::Cmpxchg => {
            if inst.operands.len() >= 2 {
                ops.dst = resolve_operand(&inst.operands[0], alloc);
                ops.src = resolve_operand(&inst.operands[1], alloc);
            }
        }

        // GPR <-> XMM transfers: [dst, src]
        X86Opcode::MovdToXmm | X86Opcode::MovdFromXmm
        | X86Opcode::MovqToXmm | X86Opcode::MovqFromXmm => {
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
        X86Opcode::AddRM | X86Opcode::SubRM | X86Opcode::CmpRM
        | X86Opcode::ImulRM | X86Opcode::TestRM => {
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
        X86Opcode::MovRMSib | X86Opcode::LeaSib => {
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
// Two-address fixup pass
// ---------------------------------------------------------------------------

/// Returns `true` if the given opcode is an x86-64 register-register ALU
/// instruction that uses two-address form (dst is both read and written).
///
/// When ISel emits three-address form `OP v2, v0, v1` (where v2 != v0), we
/// must insert `MOV v2, v0` before the instruction so that the destination
/// register contains the correct first-source value.
///
/// This only covers RR (register-register) opcodes that can appear in
/// three-address form from ISel. RI, unary, and shift forms are already
/// in-place (2-operand) and do not need this fixup.
fn is_x86_two_address_rr(opcode: X86Opcode) -> bool {
    matches!(
        opcode,
        X86Opcode::AddRR
            | X86Opcode::SubRR
            | X86Opcode::AndRR
            | X86Opcode::OrRR
            | X86Opcode::XorRR
            | X86Opcode::ImulRR
            // SSE scalar arithmetic (may appear in three-address form)
            | X86Opcode::Addsd
            | X86Opcode::Subsd
            | X86Opcode::Mulsd
            | X86Opcode::Divsd
            | X86Opcode::Addss
            | X86Opcode::Subss
            | X86Opcode::Mulss
            | X86Opcode::Divss
    )
}

/// Two-address fixup pass: insert MOV copies before two-address instructions
/// when `dst != first_src`.
///
/// x86-64 ALU instructions use two-address form: `ADD dst, src` means
/// `dst = dst + src`. When ISel emits three-address `ADD v2, v0, v1`,
/// the register allocator may assign v2 and v0 to different physical
/// registers. Without this pass, `ADD preg(v2), preg(v1)` would compute
/// `preg(v2) = preg(v2) + preg(v1)` — using whatever garbage was in
/// preg(v2) instead of v0's value.
///
/// The fix: for each three-address two-address instruction where
/// `preg(dst) != preg(lhs)`, insert `MOV dst, lhs` to initialize the
/// destination register before the ALU operation.
///
/// Reference: x86_adapter.rs `is_two_address_opcode()` and `TiedOperand`.
fn fixup_two_address(func: &mut X86ISelFunction, alloc: &HashMap<VReg, X86PReg>) {
    for block_id in func.block_order.clone() {
        if let Some(block) = func.blocks.get_mut(&block_id) {
            let mut new_insts = Vec::with_capacity(block.insts.len() + block.insts.len() / 4);

            for inst in &block.insts {
                // Only fixup three-address two-address RR instructions.
                // RI, unary, and shift forms use 1 or 2 operands — already in-place.
                let needs_fixup = is_x86_two_address_rr(inst.opcode)
                    && inst.operands.len() >= 3;

                if needs_fixup {
                    let dst_op = &inst.operands[0];
                    let lhs_op = &inst.operands[1];

                    if let (Some(dst_preg), Some(lhs_preg)) = (
                        resolve_operand(dst_op, alloc),
                        resolve_operand(lhs_op, alloc),
                    ) {
                        if dst_preg != lhs_preg {
                            // Select the correct MOV opcode based on whether
                            // this is an SSE (float) or GPR (integer) operation.
                            let mov_opcode = match inst.opcode {
                                X86Opcode::Addss | X86Opcode::Subss
                                | X86Opcode::Mulss | X86Opcode::Divss => X86Opcode::MovssRR,
                                X86Opcode::Addsd | X86Opcode::Subsd
                                | X86Opcode::Mulsd | X86Opcode::Divsd => X86Opcode::MovsdRR,
                                _ => X86Opcode::MovRR,
                            };
                            // Insert MOV dst, lhs to copy the first source into dst.
                            new_insts.push(X86ISelInst::new(
                                mov_opcode,
                                vec![dst_op.clone(), lhs_op.clone()],
                            ));
                        }
                    }
                }

                new_insts.push(inst.clone());
            }

            block.insts = new_insts;
        }
    }
}

// ---------------------------------------------------------------------------
// Formal-argument parallel-copy fixup (#497, #499)
// ---------------------------------------------------------------------------

/// Return `true` if `preg` is a System V AMD64 integer-argument register
/// (RDI, RSI, RDX, RCX, R8, R9).
#[inline]
fn is_arg_gpr(preg: X86PReg) -> bool {
    X86_ARG_GPRS.contains(&preg)
}

/// Return `true` if `preg` is a System V AMD64 FP-argument register
/// (XMM0..XMM7).
#[inline]
fn is_arg_xmm(preg: X86PReg) -> bool {
    X86_ARG_XMMS.contains(&preg)
}

/// Classify a formal-arg MOV pseudo-instruction.
///
/// Returns `Some((dst_preg, src_preg, opcode))` if `inst` is
/// `MOV vreg, PReg(arg_reg)` or `MOVSD vreg, PReg(xmm_arg)` where:
/// * the second operand is an argument register for the System V AMD64 ABI,
/// * the first operand is a VReg that the allocator has assigned to a physical
///   register.
///
/// Otherwise returns `None`.
///
/// The returned `opcode` preserves the width of the original move so the
/// resolved PReg-to-PReg copy encodes correctly (MovRR for GPR, MovsdRR for
/// XMM). `MovssRR` is lumped in with `MovsdRR` because both encode as an
/// XMM-to-XMM copy of the full 128-bit register (the high bits of the source
/// are the f32-padding established by `lower_formal_arguments` anyway).
fn classify_formal_arg_mov(
    inst: &X86ISelInst,
    alloc: &HashMap<VReg, X86PReg>,
) -> Option<(X86PReg, X86PReg, X86Opcode)> {
    match inst.opcode {
        X86Opcode::MovRR | X86Opcode::MovsdRR | X86Opcode::MovssRR => {}
        _ => return None,
    }
    if inst.operands.len() < 2 {
        return None;
    }
    let X86ISelOperand::VReg(dst_vreg) = &inst.operands[0] else {
        return None;
    };
    let X86ISelOperand::PReg(src_preg) = &inst.operands[1] else {
        return None;
    };
    let src_preg = *src_preg;
    let is_gpr = inst.opcode == X86Opcode::MovRR;
    let is_xmm = matches!(inst.opcode, X86Opcode::MovsdRR | X86Opcode::MovssRR);
    if is_gpr && !is_arg_gpr(src_preg) {
        return None;
    }
    if is_xmm && !is_arg_xmm(src_preg) {
        return None;
    }
    let dst_preg = alloc.get(dst_vreg).copied()?;
    Some((dst_preg, src_preg, inst.opcode))
}

/// Resolve a set of `dst <- src` moves on physical registers into a valid
/// sequential schedule. Breaks cycles using `scratch`.
///
/// Input invariants:
/// * All moves have distinct destinations (by construction: regalloc produced
///   them and each writes a different VReg, which maps to a different PReg;
///   duplicates would indicate an allocator bug).
/// * `scratch` must not be any destination nor any source.
/// * All moves share a single `opcode` (the caller batches GPR moves separately
///   from XMM moves so the same `opcode` applies to every entry).
///
/// Output: a sequence of `(dst, src)` moves that, when emitted in order as
/// ordinary `MOV`/`MOVSD`, produces the same simultaneous effect as the input
/// parallel copy. Self-copies (`dst == src`) are dropped.
///
/// Algorithm: topological sort — repeatedly emit any move whose destination is
/// not used as a source by any remaining move. When no such move exists, all
/// remaining moves participate in cycles; break the first cycle by saving its
/// destination into `scratch` and rewriting every reader of that destination
/// to read from `scratch` instead. Repeat until empty.
///
/// Reference: Hack et al., "Register Allocation for Programs in SSA Form",
/// chapter on parallel copies; also see `phi_elim.rs::resolve_parallel_copies`
/// which implements the same algorithm on VRegs.
fn resolve_physreg_parallel_copy(
    copies: &[(X86PReg, X86PReg)],
    scratch: X86PReg,
) -> Vec<(X86PReg, X86PReg)> {
    // Drop self-copies up front; they are no-ops in parallel-copy semantics.
    let mut remaining: Vec<(X86PReg, X86PReg)> =
        copies.iter().copied().filter(|(d, s)| d != s).collect();
    let mut result: Vec<(X86PReg, X86PReg)> = Vec::with_capacity(remaining.len() + 2);

    // Interleave topological emit with cycle breaking. Each outer iteration
    // drains all moves whose destination is not read by any remaining move
    // (safe to emit directly), then breaks one cycle if any remain.
    //
    // Breaking a cycle by copying `src -> scratch` (exposing the value for
    // later reads) and rewriting any remaining move that reads `src` to
    // read `scratch` instead transforms the graph so `src` is no longer a
    // source in `remaining`. A subsequent topological pass will then emit
    // the move that writes `src` safely — which in turn unblocks more moves.
    loop {
        // Phase 1: topological emit — repeat until no more progress.
        let mut progress = true;
        while progress {
            progress = false;
            let mut i = 0;
            while i < remaining.len() {
                let (dst, _) = remaining[i];
                let is_source =
                    remaining.iter().enumerate().any(|(j, &(_, s))| j != i && s == dst);
                if !is_source {
                    result.push(remaining.remove(i));
                    progress = true;
                } else {
                    i += 1;
                }
            }
        }
        if remaining.is_empty() {
            break;
        }

        // Phase 2: break one cycle. Pick any remaining move (pop the first)
        // and copy its SOURCE into scratch; rewrite every remaining read of
        // that source to read from scratch instead. This makes `src` no
        // longer appear as a source in `remaining`, so the move that writes
        // `src` can be safely topologically emitted in the next phase-1 pass.
        let (_d0, s0) = remaining[0];
        result.push((scratch, s0));
        for copy in &mut remaining {
            if copy.1 == s0 {
                copy.1 = scratch;
            }
        }
    }

    result
}

/// Post-regalloc fixup for the formal-argument parallel copy (#497, #499).
///
/// `lower_formal_arguments` emits one `MOV vreg, PReg(arg_reg)` per integer
/// parameter and one `MOVSD vreg, PReg(xmm_arg)` per FP parameter, in
/// parameter order. After register allocation maps each `vreg` to some
/// physical register, these sequential moves can clobber each other:
///
/// ```text
/// # tMIR: fn sum8(i0..i7: i64). i3 arrives in RCX.
/// # Regalloc picked v0 -> RCX (fine on its own), v3 -> some callee-saved.
/// MOV RCX, RDI    # v0 <- RDI     *** OVERWRITES incoming i3 (still in RCX) ***
/// MOV RSI, RSI    # (elided self-copy)
/// MOV RDX, RDX    # ...
/// MOV <v3>, RCX   # v3 <- RCX, but RCX now holds i0, not i3 -> WRONG.
/// ```
///
/// The register allocator is not aware that the incoming arg PRegs are live
/// across the MOV sequence, so it freely reassigns VRegs to any arg PReg.
/// The fix is to treat the entire MOV prologue as a parallel copy on physical
/// registers and resolve it using a topological schedule with cycle-breaking
/// via a scratch register.
///
/// This pass runs AFTER regalloc assigns VRegs to PRegs and BEFORE the
/// two-address fixup (so downstream passes see a sequence of ordinary MOVs
/// that happens to be correct).
///
/// GPR and XMM copies are resolved independently (disjoint register files).
/// The scratch register for GPR cycle-breaking is `R11` (System V caller-saved,
/// never an arg register — see `X86_ARG_GPRS = [RDI, RSI, RDX, RCX, R8, R9]`).
/// For XMM cycle-breaking, the scratch is `XMM15` (above the XMM0..XMM7 arg
/// range; XMM8-XMM15 are never used for incoming args in System V).
///
/// Limitations:
/// * Stack-argument loads (`MOV vreg, [RBP+disp]`) are not parallel copies —
///   they read memory, not registers — and are left untouched.
/// * If regalloc ever assigns an incoming value to R11 or XMM15, using them as
///   scratch would clobber it. Scanning the allocation to pick a non-destination
///   scratch would be more robust; in practice R11 is rarely allocated because
///   the allocator exhausts callee-saved registers last and arg-live ranges are
///   short. See the `scratch_collision` follow-up note in #497.
///
/// # Arguments
/// * `func` — the ISel function to rewrite in place.
/// * `alloc` — the VReg-to-PReg allocation produced by regalloc.
fn fixup_formal_arg_parallel_copy(
    func: &mut X86ISelFunction,
    alloc: &HashMap<VReg, X86PReg>,
) {
    let entry = match func.block_order.first().copied() {
        Some(b) => b,
        None => return,
    };
    let block = match func.blocks.get_mut(&entry) {
        Some(b) => b,
        None => return,
    };

    // 1. Identify the leading prefix of formal-arg MOV instructions.
    //    Partition them into GPR copies and XMM copies as we go.
    let mut prefix_len: usize = 0;
    let mut gpr_copies: Vec<(X86PReg, X86PReg)> = Vec::new();
    let mut xmm_copies: Vec<(X86PReg, X86PReg)> = Vec::new();
    for inst in &block.insts {
        match classify_formal_arg_mov(inst, alloc) {
            Some((dst, src, X86Opcode::MovRR)) => {
                gpr_copies.push((dst, src));
                prefix_len += 1;
            }
            Some((dst, src, X86Opcode::MovsdRR)) | Some((dst, src, X86Opcode::MovssRR)) => {
                xmm_copies.push((dst, src));
                prefix_len += 1;
            }
            Some(_) => break, // unreachable: classify_formal_arg_mov filters opcodes
            None => break,
        }
    }

    if prefix_len == 0 {
        return;
    }

    // 2. Resolve each partition independently. GPR and XMM share no physical
    //    storage, so they cannot form cross-class cycles.
    let resolved_gpr = resolve_physreg_parallel_copy(&gpr_copies, R11);
    let resolved_xmm = resolve_physreg_parallel_copy(&xmm_copies, XMM15);

    // 3. Splice the resolved sequence in place of the original prefix.
    //
    //    The ordering between the GPR and XMM blocks is irrelevant: they
    //    commute (no shared storage). We emit GPR first for readability;
    //    downstream disassembly conventionally shows integer-arg-shuffle
    //    before FP-arg-shuffle.
    let tail: Vec<X86ISelInst> = block.insts.drain(prefix_len..).collect();
    block.insts.clear();
    for (dst, src) in resolved_gpr {
        block.insts.push(X86ISelInst::new(
            X86Opcode::MovRR,
            vec![X86ISelOperand::PReg(dst), X86ISelOperand::PReg(src)],
        ));
    }
    for (dst, src) in resolved_xmm {
        block.insts.push(X86ISelInst::new(
            X86Opcode::MovsdRR,
            vec![X86ISelOperand::PReg(dst), X86ISelOperand::PReg(src)],
        ));
    }
    block.insts.extend(tail);
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

/// Generate the `__eh_frame` section bytes for an x86-64 function.
///
/// Creates a DWARF CIE + FDE describing the System V AMD64 ABI prologue:
/// PUSH RBP; MOV RBP, RSP; PUSH callee-saved regs; SUB RSP, frame_size.
///
/// The resulting bytes are suitable for the `__TEXT,__eh_frame` Mach-O section.
///
/// # Arguments
/// - `callee_saved`: Callee-saved registers pushed after RBP (in push order).
/// - `frame_size`: Bytes allocated via SUB RSP for locals/spills (may be 0).
/// - `code_len`: Total encoded code length in bytes.
/// - `symbol_index`: Symbol table index for the function (for relocations).
///
/// NOTE: Currently unused for Mach-O output because the __eh_frame section
/// requires x86-64 relocations that are not yet implemented. Retained for
/// future use when relocation support is added.
#[allow(dead_code)]
fn generate_eh_frame_section(
    callee_saved: &[X86PReg],
    frame_size: u32,
    code_len: u32,
    symbol_index: u32,
) -> Vec<u8> {
    let mut section = DwarfCfiSection::new_x86_64();
    let fde = x86_64_fde_from_prologue(
        callee_saved,
        frame_size,
        0,         // function_offset (placeholder, relocated by linker)
        code_len,
        symbol_index,
    );
    section.add_fde(fde);
    section.to_bytes()
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

        // PUSH/POP: 1 byte (50+rd) without REX, 2 bytes (REX.B + 50+rd) with REX.
        X86Opcode::Push | X86Opcode::Pop => {
            if let Some(X86ISelOperand::PReg(preg)) = inst.operands.first() {
                if preg.needs_rex() { 2 } else { 1 }
            } else {
                2 // Conservative fallback for unresolved operands
            }
        }

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

        // SSE RIP-relative: 8 bytes (prefix + [REX] + 0F + opcode + ModRM + disp32).
        X86Opcode::MovssRipRel | X86Opcode::MovsdRipRel => 8,

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
    /// Register allocation mode (default: Full(Greedy)).
    ///
    /// - `Simplified`: fast first-appearance allocation, no spill support.
    ///   Use for O0 or trivial test functions.
    /// - `Full(LinearScan)`: full pipeline with linear scan allocator.
    /// - `Full(Greedy)`: full pipeline with LLVM-style greedy allocator
    ///   (default — same as AArch64 pipeline).
    pub regalloc_mode: X86RegAllocMode,
}

impl Default for X86PipelineConfig {
    fn default() -> Self {
        Self {
            output_format: X86OutputFormat::RawBytes,
            emit_elf: false,
            emit_frame: true,
            regalloc_mode: X86RegAllocMode::Full(AllocStrategy::Greedy),
        }
    }
}

// ---------------------------------------------------------------------------
// X86ISelFunction -> RegAllocFunction conversion
// ---------------------------------------------------------------------------

/// Classify whether an x86-64 opcode's first operand is a definition.
///
/// Returns `true` if the first operand is a def (output), `false` if all
/// operands are uses (e.g., branches, returns, compares, stores).
fn opcode_has_def(opcode: X86Opcode) -> bool {
    use X86Opcode::*;
    match opcode {
        // Control flow: no defs (all operands are uses or implicit).
        Jmp | Jcc | Ret | Call | CallR | CallM => false,

        // Compares/tests: set flags only, no register def.
        CmpRR | CmpRI | CmpRI8 | TestRR | TestRI | TestRM
        | Ucomisd | Ucomiss | CmpRM => false,

        // Memory stores: write to memory, no register def.
        MovMR | MovsdMR | MovssMR | MovMRSib => false,

        // Stack manipulation: implicit RSP modification, no register def.
        Push => false,

        // CDQ/CQO: implicit operands only (writes RDX).
        Cdq | Cqo => false,

        // IDIV/DIV: implicit RDX:RAX operands.
        Idiv | Div => false,

        // Pseudo-instructions with no meaningful def.
        Nop | NopMulti | StackAlloc => false,

        // PHI: def is first operand.
        Phi => true,

        // Everything else: first operand is a def.
        _ => true,
    }
}

/// Returns `true` if the instruction's first operand is both read and written.
///
/// This covers:
/// - Unary in-place ops like `NEG dst`
/// - RI/RM/shift forms like `ADD dst, imm` where operand 0 is always live-in
///
/// RR/SSE three-address forms do not need special handling here because a
/// two-address-safe instruction already carries the live-in value explicitly
/// as operand 1 (`ADD v0, v0, v1`).
fn first_operand_is_def_and_use(inst: &X86ISelInst) -> bool {
    use X86Opcode::*;

    match inst.opcode {
        Neg | Not | Inc | Dec
        | AddRI | SubRI | AndRI | OrRI | XorRI
        | AddRM | SubRM | ImulRM
        | ShlRI | ShrRI | SarRI
        | ShlRR | ShrRR | SarRR => true,
        _ => false,
    }
}

/// Convert an [`X86ISelFunction`] to a [`RegAllocFunction`] for the full
/// register allocation pipeline.
///
/// This bridges the x86-64 ISel type universe to the target-independent
/// regalloc type universe:
/// - Flattens per-block instruction lists into a single `Vec<RegAllocInst>`.
/// - Classifies operands into defs and uses based on opcode semantics.
/// - Converts `Block(u32)` to `BlockId(u32)`, `X86PReg` to `PReg`.
/// - Computes predecessor lists from successor information.
fn x86_isel_to_regalloc(func: &X86ISelFunction) -> Result<RegAllocFunction, X86PipelineError> {
    let mut ra_insts: Vec<RegAllocInst> = Vec::new();
    let num_blocks = func.block_order.len();

    // Pre-allocate block vector. We'll fill them as we iterate.
    let mut ra_blocks: Vec<RegAllocBlock> = Vec::with_capacity(num_blocks);
    // Map Block(u32) -> index in ra_blocks for successor/predecessor lookup.
    let mut block_index: HashMap<u32, usize> = HashMap::new();
    for (idx, blk) in func.block_order.iter().enumerate() {
        block_index.insert(blk.0, idx);
    }

    // First pass: convert blocks and instructions.
    for blk in &func.block_order {
        let isel_block = func.blocks.get(blk).ok_or_else(|| {
            X86PipelineError::RegAlloc(format!("block {} not found in function", blk.0))
        })?;

        let mut inst_ids: Vec<InstId> = Vec::new();

        for isel_inst in &isel_block.insts {
            let inst_id = InstId(ra_insts.len() as u32);
            inst_ids.push(inst_id);

            let opcode = isel_inst.opcode;
            let flags = opcode.default_flags();
            let has_def = opcode_has_def(opcode);
            let first_operand_def_and_use = first_operand_is_def_and_use(isel_inst);

            let mut defs: Vec<RegAllocOperand> = Vec::new();
            let mut uses: Vec<RegAllocOperand> = Vec::new();
            let mut implicit_defs: Vec<llvm2_regalloc::PReg> = Vec::new();
            let mut implicit_uses: Vec<llvm2_regalloc::PReg> = Vec::new();

            for (i, op) in isel_inst.operands.iter().enumerate() {
                match op {
                    X86ISelOperand::VReg(v) => {
                        let ra_op = RegAllocOperand::VReg(*v);
                        if i == 0 && has_def {
                            defs.push(ra_op.clone());
                            // x86 in-place ops read and write operand 0.
                            if first_operand_def_and_use {
                                uses.push(ra_op);
                            }
                        } else {
                            uses.push(ra_op);
                        }
                    }
                    X86ISelOperand::PReg(x86_preg) => {
                        // Physical registers go to implicit defs/uses.
                        if let Some(preg) = x86_to_preg(*x86_preg) {
                            if i == 0 && has_def {
                                implicit_defs.push(preg);
                            } else {
                                implicit_uses.push(preg);
                            }
                        }
                    }
                    X86ISelOperand::Imm(imm) => {
                        uses.push(RegAllocOperand::Imm(*imm));
                    }
                    X86ISelOperand::FImm(fimm) => {
                        uses.push(RegAllocOperand::FImm(*fimm));
                    }
                    X86ISelOperand::Block(b) => {
                        uses.push(RegAllocOperand::Block(BlockId(b.0)));
                    }
                    X86ISelOperand::CondCode(_cc) => {
                        // Condition codes are encoded in the opcode for the
                        // regalloc's purposes — store as Imm(0) placeholder.
                        uses.push(RegAllocOperand::Imm(0));
                    }
                    X86ISelOperand::Symbol(_s) => {
                        // Symbols are relocation targets, not register operands.
                        uses.push(RegAllocOperand::Imm(0));
                    }
                    X86ISelOperand::StackSlot(slot) => {
                        uses.push(RegAllocOperand::StackSlot(
                            StackSlotId(*slot),
                        ));
                    }
                    X86ISelOperand::ConstPoolEntry(idx) => {
                        // Constant pool entry index — treated as an immediate
                        // for register allocation purposes.
                        uses.push(RegAllocOperand::Imm(*idx as i64));
                    }
                    X86ISelOperand::MemAddr { base, disp } => {
                        // Decompose memory address: base register is a use,
                        // displacement is an immediate use.
                        match base.as_ref() {
                            X86ISelOperand::VReg(v) => {
                                uses.push(RegAllocOperand::VReg(*v));
                            }
                            X86ISelOperand::PReg(p) => {
                                if let Some(preg) = x86_to_preg(*p) {
                                    implicit_uses.push(preg);
                                }
                            }
                            _ => {} // Other base types are unusual.
                        }
                        uses.push(RegAllocOperand::Imm(*disp as i64));
                    }
                }
            }

            // Add call clobbers as implicit defs for call instructions.
            if flags.is_call() {
                let clobbers = llvm2_regalloc::x86_64_caller_saved_regs();
                for preg in clobbers {
                    if !implicit_defs.contains(&preg) {
                        implicit_defs.push(preg);
                    }
                }
            }

            ra_insts.push(RegAllocInst {
                opcode: opcode as u16,
                defs,
                uses,
                implicit_defs,
                implicit_uses,
                flags,
            });
        }

        // Convert successors to BlockId.
        let succs: Vec<BlockId> = isel_block
            .successors
            .iter()
            .map(|b| BlockId(b.0))
            .collect();

        ra_blocks.push(RegAllocBlock {
            insts: inst_ids,
            preds: Vec::new(), // Filled in second pass.
            succs,
            loop_depth: 0, // No loop analysis for x86 ISel output yet.
        });
    }

    // Second pass: compute predecessors from successor info.
    for src_idx in 0..ra_blocks.len() {
        let succs: Vec<BlockId> = ra_blocks[src_idx].succs.clone();
        for succ in &succs {
            if let Some(&dst_idx) = block_index.get(&succ.0) {
                let pred_bid = BlockId(src_idx as u32);
                if !ra_blocks[dst_idx].preds.contains(&pred_bid) {
                    ra_blocks[dst_idx].preds.push(pred_bid);
                }
            }
        }
    }

    let block_order: Vec<BlockId> = (0..num_blocks).map(|i| BlockId(i as u32)).collect();
    let entry_block = BlockId(0);

    Ok(RegAllocFunction {
        name: func.name.clone(),
        insts: ra_insts,
        blocks: ra_blocks,
        block_order,
        entry_block,
        next_vreg: func.next_vreg,
        next_stack_slot: 0,
        stack_slots: HashMap::new(),
    })
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

    /// Run the full register allocator (linear scan or greedy) on an x86-64
    /// ISel function, returning an `X86RegAssignment` compatible with the rest
    /// of the pipeline.
    ///
    /// This converts the ISel function to the target-independent
    /// `RegAllocFunction`, runs the full allocator pipeline (liveness, phi
    /// elimination, coalescing, allocation, spill code), and translates the
    /// result back to x86-64 physical registers.
    fn run_full_regalloc(
        &self,
        func: &X86ISelFunction,
        strategy: AllocStrategy,
    ) -> Result<X86RegAssignment, X86PipelineError> {
        // 1. Convert X86ISelFunction -> RegAllocFunction.
        let mut ra_func = x86_isel_to_regalloc(func)?;

        // 2. Collect arg register hints from MOV vreg, PReg(X) patterns.
        //
        // When ISel emits `MOV v0, RDI` (load arg from physical register),
        // we hint the allocator to assign v0 -> RDI. This prevents the
        // allocator from assigning other vregs to arg registers (RDI, RSI,
        // RDX, RCX, R8, R9) before they are read, which would clobber the
        // argument values. (Fixes #300: arg register interference bug.)
        let mut hints: HashMap<VReg, Vec<llvm2_regalloc::PReg>> = HashMap::new();
        for block in func.block_order.iter() {
            if let Some(mblock) = func.blocks.get(block) {
                for inst in &mblock.insts {
                    let is_mov = matches!(
                        inst.opcode,
                        X86Opcode::MovRR | X86Opcode::MovsdRR
                    );
                    if is_mov && inst.operands.len() >= 2 {
                        if let (
                            X86ISelOperand::VReg(dst),
                            X86ISelOperand::PReg(src_preg),
                        ) = (&inst.operands[0], &inst.operands[1])
                        {
                            if let Some(preg) = x86_to_preg(*src_preg) {
                                hints.entry(*dst).or_default().push(preg);
                            }
                        }
                    }
                }
            }
        }

        // 3. Build config based on requested strategy, with hints.
        let mut config = match strategy {
            AllocStrategy::LinearScan => x86_64_alloc_config(),
            AllocStrategy::Greedy => x86_64_greedy_alloc_config(),
        };
        config.hints = hints;

        // 4. Run the full allocator pipeline.
        let result = llvm2_regalloc::allocate(&mut ra_func, &config)
            .map_err(|e| X86PipelineError::RegAlloc(format!("{:?}", e)))?;

        // 5. Translate PReg -> X86PReg.
        let allocation = translate_allocation(&result.allocation);

        // 6. Determine which callee-saved registers were used.
        let callee_saved_set = llvm2_regalloc::x86_64_callee_saved_regs();
        let mut used_callee_saved_set: HashSet<X86PReg> = HashSet::new();
        for x86_preg in allocation.values() {
            if let Some(preg) = x86_to_preg(*x86_preg) {
                if callee_saved_set.contains(&preg) {
                    used_callee_saved_set.insert(*x86_preg);
                }
            }
        }
        let used_callee_saved: Vec<X86PReg> = used_callee_saved_set.into_iter().collect();

        Ok(X86RegAssignment {
            allocation,
            used_callee_saved,
            num_spills: result.spills.len() as u32,
        })
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
        // Phase 1: Register allocation (full greedy/linear-scan, or simplified).
        let assignment = match self.config.regalloc_mode {
            X86RegAllocMode::Simplified => X86RegAssignment::assign(func)?,
            X86RegAllocMode::Full(strategy) => self.run_full_regalloc(func, strategy)?,
        };

        // Phase 1b: Resolve the formal-argument parallel copy (#497, #499).
        // Must run BEFORE fixup_two_address so the latter sees correct PRegs.
        let mut func = func.clone();
        fixup_formal_arg_parallel_copy(&mut func, &assignment.allocation);

        // Phase 2: Two-address fixup — insert MOV copies for dst != lhs.
        fixup_two_address(&mut func, &assignment.allocation);

        // Phase 3: Insert prologue/epilogue.
        if self.config.emit_frame {
            self.insert_prologue_epilogue(&mut func, &assignment);
        }

        // Phase 4: Resolve branch offsets.
        resolve_x86_branches(&mut func);

        // Phase 5: Encode all instructions (with constant pool fixup tracking).
        let (mut code, cp_fixups) = self.encode_function_with_fixups(&func, &assignment.allocation)?;

        // Phase 5b: Build constant pool and fix up RIP-relative displacements.
        let const_pool_data = if !func.const_pool_entries.is_empty() {
            let mut pool = crate::constant_pool::ConstantPool::new();
            // Add entries in order (indices must match ISel's ConstPoolEntry indices).
            for entry in &func.const_pool_entries {
                if entry.data.len() == 4 {
                    let val = f32::from_le_bytes([
                        entry.data[0], entry.data[1], entry.data[2], entry.data[3],
                    ]);
                    pool.add_f32(val);
                } else {
                    let val = f64::from_le_bytes([
                        entry.data[0], entry.data[1], entry.data[2], entry.data[3],
                        entry.data[4], entry.data[5], entry.data[6], entry.data[7],
                    ]);
                    pool.add_f64(val);
                }
            }
            let _total_size = pool.layout();
            let cp_bytes = pool.emit();

            // The constant pool is placed immediately after the code.
            let cp_start = code.len() as i64;

            // Fix up each RIP-relative reference.
            // Each fixup records: (offset_of_disp32_in_code, entry_index)
            // RIP-relative displacement = (cp_start + entry_offset) - fixup_offset - 4
            // (The -4 is because RIP points to the end of the instruction, i.e., past the disp32.)
            for (disp32_offset, entry_idx) in &cp_fixups {
                let entry_offset = pool.entry_offset(*entry_idx) as i64;
                let target = cp_start + entry_offset;
                let rip_disp = target - (*disp32_offset as i64) - 4;
                let disp_bytes = (rip_disp as i32).to_le_bytes();
                code[*disp32_offset..*disp32_offset + 4].copy_from_slice(&disp_bytes);
            }

            cp_bytes
        } else {
            Vec::new()
        };

        // Phase 6: Optionally wrap in object file format.
        // Resolve output format: `output_format` takes precedence, fall back to `emit_elf`.
        let effective_format = if self.config.output_format != X86OutputFormat::RawBytes {
            self.config.output_format
        } else if self.config.emit_elf {
            X86OutputFormat::Elf
        } else {
            X86OutputFormat::RawBytes
        };

        match effective_format {
            X86OutputFormat::RawBytes => {
                // For raw bytes, append the constant pool after the code.
                if !const_pool_data.is_empty() {
                    code.extend_from_slice(&const_pool_data);
                }
                Ok(code)
            }
            X86OutputFormat::Elf => Ok(self.emit_elf_with_rodata(&func.name, &code, &const_pool_data)),
            X86OutputFormat::MachO => {
                // NOTE: __eh_frame emission is disabled for x86-64 Mach-O.
                //
                // The __eh_frame section requires x86-64 relocations
                // (X86_64_RELOC_SUBTRACTOR + X86_64_RELOC_UNSIGNED pairs)
                // for the FDE's CIE pointer and PC-begin fields. Without
                // these relocations, ld64 cannot resolve the CIE pointer
                // and reports "CIE ID is not zero" because it reads the
                // wrong bytes as the CIE.
                //
                // The AArch64 path uses __LD,__compact_unwind instead,
                // which does not need embedded relocations. Adding proper
                // __eh_frame relocations for x86-64 is a follow-up task.
                // For now, x86-64 functions link and run correctly without
                // unwind info -- the linker synthesizes compact unwind
                // entries for leaf functions and simple frame-pointer
                // prologues.
                Ok(self.emit_macho_with_rodata(&func.name, &code, &const_pool_data, None))
            }
        }
    }

    /// Compile a single function to raw code bytes + constant-pool data +
    /// call fixups, without wrapping in any object file format.
    ///
    /// Used by [`Self::compile_module`] to assemble a multi-function
    /// Mach-O / ELF object with cross-function call relocations (#464).
    ///
    /// Returns `(code, const_pool_data, call_fixups)`. The `call_fixups`
    /// reference offsets *within this function's local code*; the caller
    /// is responsible for adjusting them to the combined section offset.
    pub fn compile_function_with_fixups(
        &self,
        func: &X86ISelFunction,
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<X86CallFixup>), X86PipelineError> {
        // Phase 1: Register allocation (full greedy/linear-scan, or simplified).
        let assignment = match self.config.regalloc_mode {
            X86RegAllocMode::Simplified => X86RegAssignment::assign(func)?,
            X86RegAllocMode::Full(strategy) => self.run_full_regalloc(func, strategy)?,
        };

        // Phase 1b: Resolve the formal-argument parallel copy (#497, #499).
        // Must run BEFORE fixup_two_address so the latter sees correct PRegs.
        let mut func = func.clone();
        fixup_formal_arg_parallel_copy(&mut func, &assignment.allocation);

        // Phase 2: Two-address fixup — insert MOV copies for dst != lhs.
        fixup_two_address(&mut func, &assignment.allocation);

        // Phase 3: Insert prologue/epilogue.
        if self.config.emit_frame {
            self.insert_prologue_epilogue(&mut func, &assignment);
        }

        // Phase 4: Resolve branch offsets.
        resolve_x86_branches(&mut func);

        // Phase 5: Encode all instructions (track constant pool + call fixups).
        let (mut code, cp_fixups, call_fixups) =
            self.encode_function_with_all_fixups(&func, &assignment.allocation)?;

        // Phase 5b: Build per-function constant pool and fix up RIP-relative
        // displacements relative to the END of this function's code.
        //
        // NOTE: In a multi-function module each function carries its own
        // constant pool concatenated immediately after its own code, so the
        // RIP-relative fixup math stays correct regardless of the function's
        // position in the combined section. This is a simple per-function
        // layout; a future optimization could deduplicate pools across the
        // module into a single __const section.
        let const_pool_data = if !func.const_pool_entries.is_empty() {
            let mut pool = crate::constant_pool::ConstantPool::new();
            for entry in &func.const_pool_entries {
                if entry.data.len() == 4 {
                    let val = f32::from_le_bytes([
                        entry.data[0], entry.data[1], entry.data[2], entry.data[3],
                    ]);
                    pool.add_f32(val);
                } else {
                    let val = f64::from_le_bytes([
                        entry.data[0], entry.data[1], entry.data[2], entry.data[3],
                        entry.data[4], entry.data[5], entry.data[6], entry.data[7],
                    ]);
                    pool.add_f64(val);
                }
            }
            let _total_size = pool.layout();
            let cp_bytes = pool.emit();

            // The constant pool is placed immediately after this function's code.
            let cp_start = code.len() as i64;

            for (disp32_offset, entry_idx) in &cp_fixups {
                let entry_offset = pool.entry_offset(*entry_idx) as i64;
                let target = cp_start + entry_offset;
                let rip_disp = target - (*disp32_offset as i64) - 4;
                let disp_bytes = (rip_disp as i32).to_le_bytes();
                code[*disp32_offset..*disp32_offset + 4].copy_from_slice(&disp_bytes);
            }

            cp_bytes
        } else {
            Vec::new()
        };

        Ok((code, const_pool_data, call_fixups))
    }

    /// Compile multiple pre-ISel'd functions into a single Mach-O / ELF object
    /// with cross-function CALL relocations (#464).
    ///
    /// Mirrors the AArch64 `compile_module` path in
    /// [`crate::pipeline::Pipeline::compile_module`]:
    /// 1. Per-function code is encoded independently and concatenated into a
    ///    single `__text` / `.text` section.
    /// 2. Each function gets a global symbol at its start offset.
    /// 3. Cross-function `CALL rel32` sites become Mach-O
    ///    `X86_64_RELOC_BRANCH` or ELF `R_X86_64_PLT32` relocations against
    ///    those symbols. The linker (or a follow-up static link step)
    ///    patches the displacements.
    ///
    /// The per-function output format (via `self.config.output_format`) is
    /// honored: `MachO`, `Elf`, or `RawBytes` (latter simply concatenates
    /// the code bytes without any object wrapper — CALL displacements are
    /// left as zero placeholders).
    pub fn compile_module(
        &self,
        functions: &[X86ISelFunction],
    ) -> Result<Vec<u8>, X86PipelineError> {
        if functions.is_empty() {
            // Nothing to emit. Mirror the AArch64 path: compile_module on an
            // empty module produces an empty object. Callers (the dispatcher)
            // guard against empty modules up front, so this path is a defensive
            // backstop rather than a common case.
            return Ok(Vec::new());
        }

        // Phase 1: Encode each function independently. Track per-function code
        // bytes, constant pool data, call fixups (local offsets), and the
        // function's start offset in the combined __text section.
        let mut encoded: Vec<X86EncodedFunc> = Vec::with_capacity(functions.len());
        let mut combined_code: Vec<u8> = Vec::new();

        for func in functions {
            let (code, const_pool, call_fixups) = self.compile_function_with_fixups(func)?;

            let text_offset = combined_code.len() as u64;
            // Append this function's code (and its immediate per-function
            // constant pool) to the combined __text section. Keeping the
            // constant pool inline preserves the RIP-relative displacements
            // already patched in compile_function_with_fixups. A future
            // optimization could lift pools into a shared __const section,
            // but that requires post-layout RIP-relative repatching.
            combined_code.extend_from_slice(&code);
            combined_code.extend_from_slice(&const_pool);

            encoded.push(X86EncodedFunc {
                name: func.name.clone(),
                code_len: code.len(),
                const_pool_len: const_pool.len(),
                call_fixups,
                text_offset,
            });
        }

        // Resolve effective output format (matches compile_function semantics).
        let effective_format = if self.config.output_format != X86OutputFormat::RawBytes {
            self.config.output_format
        } else if self.config.emit_elf {
            X86OutputFormat::Elf
        } else {
            X86OutputFormat::RawBytes
        };

        match effective_format {
            X86OutputFormat::RawBytes => {
                // No object wrapper — return the raw concatenated code. CALL
                // displacements stay zero; this mode is only useful for tests
                // that pre-validate encoding without linking.
                Ok(combined_code)
            }
            X86OutputFormat::MachO => {
                Ok(self.emit_module_macho(&encoded, &combined_code))
            }
            X86OutputFormat::Elf => {
                Ok(self.emit_module_elf(&encoded, &combined_code))
            }
        }
    }

    /// Emit a multi-function Mach-O object with x86-64 CALL relocations (#464).
    fn emit_module_macho(
        &self,
        encoded: &[X86EncodedFunc],
        combined_code: &[u8],
    ) -> Vec<u8> {
        use crate::macho::x86_64_reloc::X86_64Relocation;

        let mut writer = MachOWriter::for_target(MachOTarget::X86_64);
        writer.add_text_section(combined_code);

        // Build symbol table: one global symbol per function at its text offset.
        // Mach-O convention: prefix external symbol names with '_'.
        //
        // symbol_map maps both the unmangled and mangled names to the symbol
        // table index, so call-site fixups can resolve by either spelling.
        let mut symbol_map: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        for (i, ef) in encoded.iter().enumerate() {
            let mangled = format!("_{}", ef.name);
            writer.add_symbol(&mangled, 1, ef.text_offset, true);
            symbol_map.insert(ef.name.clone(), i as u32);
            symbol_map.insert(mangled, i as u32);
        }

        // Convert each call fixup into an X86_64_RELOC_BRANCH relocation.
        // The fixup.offset is local to its function; add the function's
        // text_offset to get the offset within the combined __text section.
        for ef in encoded {
            for cf in &ef.call_fixups {
                let section_offset = ef.text_offset + cf.offset as u64;
                // If the callee name is not in our symbol map, skip with a
                // zero placeholder rather than panicking. External calls
                // (e.g., to libc) would normally surface here; for now the
                // dispatcher only supports intra-module calls, so a missing
                // name means the caller is referencing an undefined function
                // and the resulting object will have a dangling CALL that the
                // linker will reject — exactly the right error behavior.
                if let Some(&sym_idx) = symbol_map
                    .get(&cf.callee)
                    .or_else(|| symbol_map.get(&format!("_{}", cf.callee)))
                {
                    let reloc = X86_64Relocation::branch(section_offset as u32, sym_idx);
                    writer.add_x86_64_relocation(0, reloc);
                }
            }
        }

        writer.write()
    }

    /// Emit a multi-function ELF object with x86-64 PLT32 CALL relocations (#464).
    fn emit_module_elf(
        &self,
        encoded: &[X86EncodedFunc],
        combined_code: &[u8],
    ) -> Vec<u8> {
        use crate::elf::reloc::{Elf64Rela, X86_64RelocType};

        let mut writer = ElfWriter::new(ElfMachine::X86_64);
        writer.add_text_section(combined_code);

        // Build symbol table: one STT_FUNC per function at its text offset.
        //
        // ELF symbol indices start at 1 (index 0 is the null STN_UNDEF symbol,
        // automatically emitted by the writer). ElfWriter::add_symbol appends
        // to the symbol list in insertion order; our first add_symbol call
        // therefore corresponds to symbol index 1.
        let mut symbol_map: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        for (i, ef) in encoded.iter().enumerate() {
            writer.add_symbol(&ef.name, 1, ef.text_offset, 0, true, 2); // STT_FUNC=2
            // ELF symbol index: +1 because the null symbol occupies slot 0.
            symbol_map.insert(ef.name.clone(), (i as u32) + 1);
        }

        // Convert each call fixup into an R_X86_64_PLT32 relocation.
        // Addend is -4 (standard for x86-64 PC-relative: the displacement is
        // computed from the END of the 4-byte displacement field, but the
        // relocation offset points to the START of it, so an implicit -4
        // addend adjusts for the 4-byte field itself).
        for ef in encoded {
            for cf in &ef.call_fixups {
                let section_offset = ef.text_offset + cf.offset as u64;
                if let Some(&sym_idx) = symbol_map.get(&cf.callee) {
                    let rela = Elf64Rela::x86_64(
                        section_offset,
                        sym_idx,
                        X86_64RelocType::Plt32,
                        -4,
                    );
                    writer.add_relocation(0, rela);
                }
            }
        }

        writer.write()
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

        // Seed Value->Type hints from the adapter (Call/CallIndirect result
        // types, see #381). Must happen BEFORE select_block.
        isel.seed_value_types(&input.value_types);

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
    ///
    /// Also returns fixup information for constant pool RIP-relative references.
    /// Each fixup is a `(disp32_byte_offset_in_code, const_pool_entry_index)`.
    fn encode_function_with_fixups(
        &self,
        func: &X86ISelFunction,
        alloc: &HashMap<VReg, X86PReg>,
    ) -> Result<(Vec<u8>, Vec<(usize, usize)>), X86PipelineError> {
        let (code, cp_fixups, _call_fixups) = self.encode_function_with_all_fixups(func, alloc)?;
        Ok((code, cp_fixups))
    }

    /// Encode a function, tracking both constant-pool fixups and call-site
    /// fixups for cross-function CALL relocations (#464).
    ///
    /// `call_fixups` records, for each `X86Opcode::Call` with a
    /// `X86ISelOperand::Symbol(name)` target, the byte offset of the `disp32`
    /// field within the encoded code and the target callee name. The caller
    /// (multi-function module emitter) converts these to proper Mach-O /
    /// ELF relocations against the combined `.text` section.
    ///
    /// This is a single-function primitive; it does not know about other
    /// functions' offsets in the combined section.
    fn encode_function_with_all_fixups(
        &self,
        func: &X86ISelFunction,
        alloc: &HashMap<VReg, X86PReg>,
    ) -> Result<(Vec<u8>, Vec<(usize, usize)>, Vec<X86CallFixup>), X86PipelineError> {
        let mut encoder = X86Encoder::new();
        let mut cp_fixups: Vec<(usize, usize)> = Vec::new();
        let mut call_fixups: Vec<X86CallFixup> = Vec::new();

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

                    let mut ops = resolve_inst_operands(inst, alloc);

                    // For RIP-relative constant pool loads, the imm field holds
                    // the entry index. Record fixup and set disp=0 as placeholder.
                    if matches!(inst.opcode, X86Opcode::MovssRipRel | X86Opcode::MovsdRipRel) {
                        let entry_idx = ops.imm as usize;
                        // The disp32 is the last 4 bytes of the encoded instruction.
                        // We set disp=0 for now; the fixup pass patches it later.
                        ops.disp = 0;
                        ops.imm = 0;

                        encoder
                            .encode_instruction(inst.opcode, &ops)
                            .map_err(X86PipelineError::from)?;
                        let after = encoder.position();

                        // The disp32 occupies the last 4 bytes of the instruction.
                        let disp32_offset = after - 4;
                        cp_fixups.push((disp32_offset, entry_idx));
                    } else if matches!(inst.opcode, X86Opcode::Call) {
                        // CALL rel32: locate the symbol operand (callee name) and
                        // record a call fixup for later relocation emission.
                        // The encoder emits `E8 dd dd dd dd` (5 bytes) with
                        // disp=0; the disp32 field is the last 4 bytes.
                        let callee_name = inst.operands.iter().find_map(|op| {
                            if let X86ISelOperand::Symbol(s) = op {
                                Some(s.clone())
                            } else {
                                None
                            }
                        });

                        encoder
                            .encode_instruction(inst.opcode, &ops)
                            .map_err(X86PipelineError::from)?;
                        let after = encoder.position();

                        if let Some(name) = callee_name {
                            // disp32 occupies the last 4 bytes of the CALL rel32
                            // encoding; the linker adds an implicit -4 addend for
                            // PC-relative displacement (x86 RIP points to the
                            // instruction *after* the 4-byte displacement, which
                            // is already handled by Mach-O X86_64_RELOC_BRANCH
                            // and ELF R_X86_64_PLT32/PC32 semantics).
                            let disp32_offset = after - 4;
                            call_fixups.push(X86CallFixup {
                                offset: disp32_offset,
                                callee: name,
                            });
                        }
                        // Indirect CALLs with no Symbol operand (CallR / CallM,
                        // or rare direct CALLs with immediate targets) generate
                        // no relocation.
                    } else {
                        encoder
                            .encode_instruction(inst.opcode, &ops)
                            .map_err(X86PipelineError::from)?;
                    }
                }
            }
        }

        Ok((encoder.finish(), cp_fixups, call_fixups))
    }

    /// Encode all instructions (legacy, no constant pool fixup tracking).
    #[allow(dead_code)]
    fn encode_function(
        &self,
        func: &X86ISelFunction,
        alloc: &HashMap<VReg, X86PReg>,
    ) -> Result<Vec<u8>, X86PipelineError> {
        let (code, _fixups) = self.encode_function_with_fixups(func, alloc)?;
        Ok(code)
    }

    /// Emit an ELF .o file wrapping the encoded machine code.
    #[allow(dead_code)]
    fn emit_elf(&self, func_name: &str, code: &[u8]) -> Vec<u8> {
        self.emit_elf_with_rodata(func_name, code, &[])
    }

    /// Emit an ELF .o file with optional .rodata section for constant pool.
    fn emit_elf_with_rodata(&self, func_name: &str, code: &[u8], rodata: &[u8]) -> Vec<u8> {
        use crate::elf::constants::{SHF_ALLOC, SHT_PROGBITS};

        let mut writer = ElfWriter::new(ElfMachine::X86_64);
        writer.add_text_section(code);
        if !rodata.is_empty() {
            writer.add_section(".rodata", rodata, SHT_PROGBITS, SHF_ALLOC, 8);
        }
        writer.add_symbol(func_name, 1, 0, code.len() as u64, true, 2); // STT_FUNC
        writer.write()
    }

    /// Emit a Mach-O .o file wrapping the encoded machine code.
    #[allow(dead_code)]
    fn emit_macho(&self, func_name: &str, code: &[u8]) -> Vec<u8> {
        self.emit_macho_with_rodata(func_name, code, &[], None)
    }

    /// Emit a Mach-O .o file with optional __TEXT,__const and __TEXT,__eh_frame sections.
    fn emit_macho_with_rodata(&self, func_name: &str, code: &[u8], rodata: &[u8], eh_frame: Option<&[u8]>) -> Vec<u8> {
        use crate::macho::constants::S_REGULAR;

        let mut writer = MachOWriter::for_target(MachOTarget::X86_64);
        writer.add_text_section(code);
        if !rodata.is_empty() {
            // Use __TEXT,__const for read-only constant data (float literals).
            // Alignment = 3 (2^3 = 8 bytes, sufficient for f64 alignment).
            writer.add_custom_section(b"__const", b"__TEXT", rodata, 3, S_REGULAR);
        }

        // Mach-O convention: symbol names have underscore prefix.
        let macho_name = format!("_{}", func_name);
        writer.add_symbol(&macho_name, 1, 0, true);

        // Add __eh_frame section for DWARF CFI (stack unwinding).
        // macOS requires this for proper backtraces and exception handling.
        //
        // Section flags: S_COALESCED (0x0B) | S_ATTR_LIVE_SUPPORT (0x08000000)
        // S_COALESCED: linker may coalesce identical CIEs from multiple .o files.
        // S_ATTR_LIVE_SUPPORT: keep section if any referencing code is live.
        //
        // Reference: ~/llvm-project-ref/llvm/lib/MC/MCMachOStreamer.cpp
        if let Some(eh_frame_data) = eh_frame {
            if !eh_frame_data.is_empty() {
                // S_COALESCED | S_ATTR_LIVE_SUPPORT | S_ATTR_NO_TOC | S_ATTR_STRIP_STATIC_SYMS
                // Standard macOS __eh_frame flags: 0x6800000B
                const EH_FRAME_FLAGS: u32 = 0x6800_000B;
                writer.add_custom_section(
                    b"__eh_frame\0\0\0\0\0\0",           // sectname (16 bytes, null-padded)
                    b"__TEXT\0\0\0\0\0\0\0\0\0\0",       // segname (16 bytes, null-padded)
                    eh_frame_data,
                    3,              // alignment: 2^3 = 8 bytes (pointer-size aligned)
                    EH_FRAME_FLAGS,
                );
            }
        }

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

/// Compile an x86-64 ISel function using the full greedy register allocator.
///
/// Equivalent to `x86_compile_to_bytes` since the default pipeline now uses
/// `Full(Greedy)`. Retained for backward compatibility and explicitness.
pub fn x86_compile_with_greedy_regalloc(
    func: &X86ISelFunction,
) -> Result<Vec<u8>, X86PipelineError> {
    let pipeline = X86Pipeline::new(X86PipelineConfig {
        regalloc_mode: X86RegAllocMode::Full(AllocStrategy::Greedy),
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

    #[test]
    fn test_macho_x86_64_full_header_validation() {
        // Validate the complete Mach-O header structure for x86-64 output,
        // including CPU subtype, file type, number of load commands, and flags.
        let func = build_x86_const_test_function();
        let bytes = x86_compile_to_macho(&func).unwrap();

        // Helper to read little-endian u32.
        fn read_u32(bytes: &[u8], offset: usize) -> u32 {
            u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        }

        // mach_header_64 fields:
        // offset  0: magic     = MH_MAGIC_64 (0xFEEDFACF)
        // offset  4: cputype   = CPU_TYPE_X86_64 (0x01000007)
        // offset  8: cpusubtype = CPU_SUBTYPE_X86_64_ALL (3)
        // offset 12: filetype  = MH_OBJECT (1)
        // offset 16: ncmds     = 4 (segment, build_version, symtab, dysymtab)
        // offset 20: sizeofcmds
        // offset 24: flags     = MH_SUBSECTIONS_VIA_SYMBOLS (0x2000)
        // offset 28: reserved  = 0

        assert_eq!(read_u32(&bytes, 0), 0xFEEDFACF, "magic");
        assert_eq!(read_u32(&bytes, 4), 0x01000007, "cputype = CPU_TYPE_X86_64");
        assert_eq!(read_u32(&bytes, 8), 3, "cpusubtype = CPU_SUBTYPE_X86_64_ALL");
        assert_eq!(read_u32(&bytes, 12), 1, "filetype = MH_OBJECT");
        assert_eq!(read_u32(&bytes, 16), 4, "ncmds = 4");
        assert!(read_u32(&bytes, 20) > 0, "sizeofcmds > 0");
        assert_eq!(read_u32(&bytes, 24) & 0x2000, 0x2000, "MH_SUBSECTIONS_VIA_SYMBOLS");
        assert_eq!(read_u32(&bytes, 28), 0, "reserved = 0");
    }

    #[test]
    fn test_macho_x86_64_contains_function_code() {
        // Verify that the Mach-O output contains the actual machine code
        // by compiling a frameless function and searching for the RET byte.
        let func = build_x86_const_test_function();

        // First compile to raw bytes to know what to look for.
        let pipeline_raw = X86Pipeline::new(X86PipelineConfig {
            output_format: X86OutputFormat::RawBytes,
            emit_frame: false,
            ..X86PipelineConfig::default()
        });
        let raw_code = pipeline_raw.compile_function(&func).unwrap();

        // Now compile to Mach-O.
        let pipeline_macho = X86Pipeline::new(X86PipelineConfig {
            output_format: X86OutputFormat::MachO,
            emit_frame: false,
            ..X86PipelineConfig::default()
        });
        let macho_bytes = pipeline_macho.compile_function(&func).unwrap();

        // The raw code should appear somewhere in the Mach-O output.
        let found = macho_bytes
            .windows(raw_code.len())
            .any(|window| window == raw_code.as_slice());
        assert!(
            found,
            "Mach-O output should contain the raw machine code ({} bytes)",
            raw_code.len()
        );
    }

    #[test]
    fn test_macho_x86_64_symbol_name_mangling() {
        // Verify that the emitted symbol name has the _ prefix per Mach-O convention.
        let func = build_x86_add_test_function();
        let bytes = x86_compile_to_macho(&func).unwrap();

        // The string table should contain "_add" (underscore-prefixed function name).
        let strtab_bytes = String::from_utf8_lossy(&bytes);
        assert!(
            strtab_bytes.contains("_add"),
            "Mach-O should contain symbol name '_add' (underscore-prefixed)"
        );
    }

    #[test]
    fn test_macho_x86_64_not_aarch64() {
        // Ensure the x86-64 pipeline does NOT emit AArch64 CPU type.
        let func = build_x86_const_test_function();
        let bytes = x86_compile_to_macho(&func).unwrap();

        let cputype = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        // CPU_TYPE_ARM64 = 0x0100000C -- must NOT be this.
        assert_ne!(cputype, 0x0100000C, "x86-64 Mach-O must not have ARM64 CPU type");
    }

    #[test]
    fn test_macho_x86_64_section_alignment() {
        // x86-64 Mach-O text sections should be 16-byte aligned (2^4),
        // per System V ABI convention. Check the section header's align field.
        let func = build_x86_const_test_function();
        let bytes = x86_compile_to_macho(&func).unwrap();

        // section_64 header starts after mach_header_64(32) + segment_command_64(72).
        // section_64 layout at offset 104:
        //   sectname(16) + segname(16) + addr(8) + size(8) + offset(4) + align(4)
        //   = align field is at offset 104 + 52 = 156
        let section_hdr_start = 32 + 72; // 104
        let align_offset = section_hdr_start + 16 + 16 + 8 + 8 + 4; // 52 bytes in
        let align = u32::from_le_bytes([
            bytes[align_offset],
            bytes[align_offset + 1],
            bytes[align_offset + 2],
            bytes[align_offset + 3],
        ]);
        assert_eq!(align, 4, "x86-64 text section should have alignment 2^4 = 16 bytes");
    }

    #[test]
    fn test_macho_x86_64_build_version_platform() {
        // Verify the LC_BUILD_VERSION command specifies macOS platform.
        let func = build_x86_const_test_function();
        let bytes = x86_compile_to_macho(&func).unwrap();

        fn read_u32(bytes: &[u8], offset: usize) -> u32 {
            u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        }

        // LC_BUILD_VERSION follows the LC_SEGMENT_64 (which includes section headers).
        // Read nsects from segment command at offset 32+64 to compute correct offset.
        let nsects = read_u32(&bytes, 32 + 64);
        let bv_offset = 32 + 72 + (nsects as usize) * 80; // header + segment cmd + nsects sections
        let bv_cmd = read_u32(&bytes, bv_offset);
        assert_eq!(bv_cmd, 0x32, "expected LC_BUILD_VERSION (0x32)");

        let platform = read_u32(&bytes, bv_offset + 8);
        assert_eq!(platform, 1, "platform should be PLATFORM_MACOS (1)");
    }

    // -----------------------------------------------------------------------
    // __eh_frame (DWARF CFI) in Mach-O output
    // -----------------------------------------------------------------------

    #[test]
    fn test_macho_x86_64_no_eh_frame_section() {
        // __eh_frame is disabled for x86-64 Mach-O because it requires
        // relocations (X86_64_RELOC_SUBTRACTOR pairs) that are not yet
        // implemented. Without relocations, ld64 reports "CIE ID is not zero".
        // Verify that the compiled Mach-O does NOT contain __eh_frame.
        let func = build_x86_add_test_function();
        let bytes = x86_compile_to_macho(&func).unwrap();

        fn read_u32(bytes: &[u8], offset: usize) -> u32 {
            u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        }

        let nsects = read_u32(&bytes, 32 + 64);
        for i in 0..nsects as usize {
            let sec_offset = 32 + 72 + i * 80;
            let sectname = &bytes[sec_offset..sec_offset + 16];
            assert!(
                !sectname.starts_with(b"__eh_frame"),
                "x86-64 Mach-O should not contain __eh_frame (no relocation support yet)"
            );
        }
    }

    #[test]
    fn test_macho_x86_64_no_eh_frame_when_disabled() {
        // When emit_frame is false, no __eh_frame section should be emitted.
        let func = build_x86_const_test_function();
        let pipeline = X86Pipeline::new(X86PipelineConfig {
            output_format: X86OutputFormat::MachO,
            emit_frame: false,
            ..X86PipelineConfig::default()
        });
        let bytes = pipeline.compile_function(&func).unwrap();

        fn read_u32(bytes: &[u8], offset: usize) -> u32 {
            u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        }

        let nsects = read_u32(&bytes, 32 + 64);
        for i in 0..nsects as usize {
            let sec_off = 32 + 72 + i * 80;
            let sectname = &bytes[sec_off..sec_off + 16];
            assert!(!sectname.starts_with(b"__eh_frame"),
                "should not have __eh_frame when emit_frame is false");
        }
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

    // -----------------------------------------------------------------------
    // Full regalloc integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_regalloc_mode_default_is_full_greedy() {
        let config = X86PipelineConfig::default();
        assert_eq!(config.regalloc_mode, X86RegAllocMode::Full(AllocStrategy::Greedy));
    }

    #[test]
    fn test_isel_to_regalloc_conversion() {
        let func = build_x86_add_test_function();
        let ra_func = x86_isel_to_regalloc(&func).unwrap();
        assert_eq!(ra_func.name, "add");
        assert!(!ra_func.blocks.is_empty(), "should have at least one block");
        assert!(!ra_func.insts.is_empty(), "should have instructions");
        // add function: MOV, MOV, ADD, MOV, RET = 5 instructions
        assert_eq!(ra_func.insts.len(), 5);
        assert_eq!(ra_func.blocks.len(), 1);
    }

    #[test]
    fn test_isel_to_regalloc_addri_marks_operand0_live_in() {
        let sig = Signature {
            params: vec![llvm2_lower::types::Type::I64, llvm2_lower::types::Type::I64],
            returns: vec![llvm2_lower::types::Type::I64],
        };
        let mut func = X86ISelFunction::new("add_imm_inplace".to_string(), sig);
        let entry = Block(0);
        func.ensure_block(entry);

        let v0 = VReg::new(0, RegClass::Gpr64);
        func.next_vreg = 1;

        func.push_inst(entry, X86ISelInst::new(
            X86Opcode::MovRR,
            vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(x86_64_regs::RDI)],
        ));
        func.push_inst(entry, X86ISelInst::new(
            X86Opcode::AddRI,
            vec![X86ISelOperand::VReg(v0), X86ISelOperand::Imm(7)],
        ));
        func.push_inst(entry, X86ISelInst::new(
            X86Opcode::MovRR,
            vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)],
        ));
        func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));

        let ra_func = x86_isel_to_regalloc(&func).unwrap();
        let add = &ra_func.insts[1];
        assert_eq!(add.defs, vec![RegAllocOperand::VReg(VReg::new(0, RegClass::Gpr64))]);
        assert_eq!(
            add.uses,
            vec![
                RegAllocOperand::VReg(VReg::new(0, RegClass::Gpr64)),
                RegAllocOperand::Imm(7),
            ]
        );
    }

    #[test]
    fn test_isel_to_regalloc_const_function() {
        let func = build_x86_const_test_function();
        let ra_func = x86_isel_to_regalloc(&func).unwrap();
        assert_eq!(ra_func.name, "const42");
        // const42: MOV v0 42, MOV RAX v0, RET = 3 instructions
        assert_eq!(ra_func.insts.len(), 3);
    }

    #[test]
    fn test_greedy_regalloc_const_function() {
        let func = build_x86_const_test_function();
        let code = x86_compile_with_greedy_regalloc(&func).unwrap();
        assert!(!code.is_empty(), "greedy regalloc should produce code");
        assert_eq!(*code.last().unwrap(), 0xC3, "last byte should be RET");
    }

    #[test]
    fn test_greedy_regalloc_add_function() {
        let func = build_x86_add_test_function();
        let pipeline = X86Pipeline::new(X86PipelineConfig {
            regalloc_mode: X86RegAllocMode::Full(AllocStrategy::Greedy),
            ..X86PipelineConfig::default()
        });
        let code = pipeline.compile_function(&func).unwrap();
        assert!(!code.is_empty(), "greedy regalloc should produce code");
        assert_eq!(*code.last().unwrap(), 0xC3, "last byte should be RET");
    }

    #[test]
    fn test_linear_scan_full_regalloc_add_function() {
        let func = build_x86_add_test_function();
        let pipeline = X86Pipeline::new(X86PipelineConfig {
            regalloc_mode: X86RegAllocMode::Full(AllocStrategy::LinearScan),
            ..X86PipelineConfig::default()
        });
        let code = pipeline.compile_function(&func).unwrap();
        assert!(!code.is_empty(), "linear scan regalloc should produce code");
        assert_eq!(*code.last().unwrap(), 0xC3, "last byte should be RET");
    }

    #[test]
    fn test_full_regalloc_no_frame() {
        let func = build_x86_const_test_function();
        let pipeline = X86Pipeline::new(X86PipelineConfig {
            regalloc_mode: X86RegAllocMode::Full(AllocStrategy::Greedy),
            emit_frame: false,
            ..X86PipelineConfig::default()
        });
        let code = pipeline.compile_function(&func).unwrap();
        assert!(!code.is_empty());
        assert_eq!(*code.last().unwrap(), 0xC3);
    }

    #[test]
    fn test_simplified_and_full_both_produce_valid_code() {
        // Both regalloc modes should produce code ending in RET for the same function.
        let func = build_x86_const_test_function();

        let simplified = X86Pipeline::new(X86PipelineConfig {
            regalloc_mode: X86RegAllocMode::Simplified,
            ..X86PipelineConfig::default()
        });
        let full = X86Pipeline::new(X86PipelineConfig {
            regalloc_mode: X86RegAllocMode::Full(AllocStrategy::Greedy),
            ..X86PipelineConfig::default()
        });

        let code_s = simplified.compile_function(&func).unwrap();
        let code_f = full.compile_function(&func).unwrap();

        assert_eq!(*code_s.last().unwrap(), 0xC3);
        assert_eq!(*code_f.last().unwrap(), 0xC3);
    }

    // -----------------------------------------------------------------------
    // Two-address fixup tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_x86_two_address_rr() {
        assert!(is_x86_two_address_rr(X86Opcode::AddRR));
        assert!(is_x86_two_address_rr(X86Opcode::SubRR));
        assert!(is_x86_two_address_rr(X86Opcode::AndRR));
        assert!(is_x86_two_address_rr(X86Opcode::OrRR));
        assert!(is_x86_two_address_rr(X86Opcode::XorRR));
        assert!(is_x86_two_address_rr(X86Opcode::ImulRR));
        assert!(is_x86_two_address_rr(X86Opcode::Addsd));
        assert!(is_x86_two_address_rr(X86Opcode::Subsd));

        // These are NOT two-address RR:
        assert!(!is_x86_two_address_rr(X86Opcode::MovRR));
        assert!(!is_x86_two_address_rr(X86Opcode::CmpRR));
        assert!(!is_x86_two_address_rr(X86Opcode::Lea));
        assert!(!is_x86_two_address_rr(X86Opcode::Ret));
        assert!(!is_x86_two_address_rr(X86Opcode::AddRI)); // RI is in-place, not RR
        assert!(!is_x86_two_address_rr(X86Opcode::Neg));    // unary, not RR
    }

    #[test]
    fn test_fixup_two_address_inserts_mov_when_dst_ne_lhs() {
        // Build: ADD v2, v0, v1 (three-address, v2 != v0)
        // After assignment: v0->RAX, v1->RCX, v2->RDX
        // Fixup should insert: MOV v2, v0 before ADD v2, v0, v1
        use llvm2_ir::x86_64_regs::{RAX, RCX, RDX};

        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut func = X86ISelFunction::new("fixup_test".to_string(), sig);
        let entry = Block(0);
        func.ensure_block(entry);

        let v0 = VReg::new(0, RegClass::Gpr64);
        let v1 = VReg::new(1, RegClass::Gpr64);
        let v2 = VReg::new(2, RegClass::Gpr64);
        func.next_vreg = 3;

        func.push_inst(entry, X86ISelInst::new(
            X86Opcode::AddRR,
            vec![
                X86ISelOperand::VReg(v2),
                X86ISelOperand::VReg(v0),
                X86ISelOperand::VReg(v1),
            ],
        ));

        let mut alloc = HashMap::new();
        alloc.insert(v0, RAX);
        alloc.insert(v1, RCX);
        alloc.insert(v2, RDX);

        // Before fixup: 1 instruction (ADD)
        assert_eq!(func.blocks[&entry].insts.len(), 1);

        fixup_two_address(&mut func, &alloc);

        // After fixup: 2 instructions (MOV + ADD)
        let insts = &func.blocks[&entry].insts;
        assert_eq!(insts.len(), 2, "should insert MOV before ADD");
        assert_eq!(insts[0].opcode, X86Opcode::MovRR, "first should be MOV");
        assert_eq!(insts[1].opcode, X86Opcode::AddRR, "second should be ADD");

        // MOV operands should be [dst(v2), lhs(v0)]
        assert_eq!(insts[0].operands[0], X86ISelOperand::VReg(v2));
        assert_eq!(insts[0].operands[1], X86ISelOperand::VReg(v0));
    }

    #[test]
    fn test_fixup_two_address_no_insert_when_dst_eq_lhs() {
        // Build: ADD v0, v0, v1 (three-address, but dst == lhs)
        // After assignment: v0->RAX, v1->RCX
        // Fixup should NOT insert anything.
        use llvm2_ir::x86_64_regs::{RAX, RCX};

        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut func = X86ISelFunction::new("no_fixup_test".to_string(), sig);
        let entry = Block(0);
        func.ensure_block(entry);

        let v0 = VReg::new(0, RegClass::Gpr64);
        let v1 = VReg::new(1, RegClass::Gpr64);
        func.next_vreg = 2;

        func.push_inst(entry, X86ISelInst::new(
            X86Opcode::AddRR,
            vec![
                X86ISelOperand::VReg(v0),
                X86ISelOperand::VReg(v0),
                X86ISelOperand::VReg(v1),
            ],
        ));

        let mut alloc = HashMap::new();
        alloc.insert(v0, RAX);
        alloc.insert(v1, RCX);

        fixup_two_address(&mut func, &alloc);

        // No MOV inserted — dst == lhs
        let insts = &func.blocks[&entry].insts;
        assert_eq!(insts.len(), 1, "should NOT insert MOV when dst == lhs");
        assert_eq!(insts[0].opcode, X86Opcode::AddRR);
    }

    #[test]
    fn test_fixup_two_address_skips_two_operand_form() {
        // Build: ADD v0, v1 (two-operand form, already in-place)
        // Fixup should NOT insert anything (only 2 operands, not >= 3).
        use llvm2_ir::x86_64_regs::{RAX, RCX};

        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut func = X86ISelFunction::new("two_op_test".to_string(), sig);
        let entry = Block(0);
        func.ensure_block(entry);

        let v0 = VReg::new(0, RegClass::Gpr64);
        let v1 = VReg::new(1, RegClass::Gpr64);
        func.next_vreg = 2;

        func.push_inst(entry, X86ISelInst::new(
            X86Opcode::AddRR,
            vec![
                X86ISelOperand::VReg(v0),
                X86ISelOperand::VReg(v1),
            ],
        ));

        let mut alloc = HashMap::new();
        alloc.insert(v0, RAX);
        alloc.insert(v1, RCX);

        fixup_two_address(&mut func, &alloc);

        // No MOV inserted — only 2 operands
        assert_eq!(func.blocks[&entry].insts.len(), 1);
    }

    #[test]
    fn test_three_address_add_compiles_correctly() {
        // The original three-address add function (v2 = v0 + v1, v2 != v0)
        // should now compile correctly thanks to the two-address fixup.
        let func = build_x86_add_test_function();
        let pipeline = X86Pipeline::new(X86PipelineConfig {
            emit_frame: false,
            ..X86PipelineConfig::default()
        });
        let code = pipeline.compile_function(&func).unwrap();
        assert!(!code.is_empty());
        assert_eq!(*code.last().unwrap(), 0xC3, "last byte should be RET");
    }

    // -----------------------------------------------------------------------
    // Formal-argument parallel-copy resolver tests (#497, #499)
    // -----------------------------------------------------------------------

    #[test]
    fn test_resolve_physreg_parallel_copy_no_cycle() {
        // v0 <- RDI, v1 <- RSI, v2 <- RDX with allocation v0->RAX, v1->RBX, v2->RCX.
        // No cycle: each destination is fresh.
        use llvm2_ir::x86_64_regs::{
            RAX, RBX, RCX, RDI, RDX, RSI, R11,
        };
        let copies = vec![
            (RAX, RDI),
            (RBX, RSI),
            (RCX, RDX),
        ];
        let resolved = resolve_physreg_parallel_copy(&copies, R11);
        // All 3 moves preserved, no scratch used.
        assert_eq!(resolved.len(), 3);
        // The ordering is whatever topological sort produced; all must appear.
        let set: std::collections::HashSet<_> = resolved.iter().copied().collect();
        assert!(set.contains(&(RAX, RDI)));
        assert!(set.contains(&(RBX, RSI)));
        assert!(set.contains(&(RCX, RDX)));
        // R11 must not appear as either a dst or src.
        for (d, s) in &resolved {
            assert_ne!(*d, R11, "scratch should not be written");
            assert_ne!(*s, R11, "scratch should not be read");
        }
    }

    #[test]
    fn test_resolve_physreg_parallel_copy_drops_self_copy() {
        use llvm2_ir::x86_64_regs::{RDI, RSI, R11};
        // RSI <- RDI, RSI <- RSI (self-copy, should be elided).
        let copies = vec![(RSI, RDI), (RSI, RSI)];
        let resolved = resolve_physreg_parallel_copy(&copies, R11);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0], (RSI, RDI));
    }

    #[test]
    fn test_resolve_physreg_parallel_copy_two_cycle_uses_scratch() {
        // RCX <- RBX, RBX <- RCX: 2-cycle that must be broken via scratch.
        // Expected output: save one of the registers into R11, then sequence
        // the moves reading from R11 where appropriate.
        use llvm2_ir::x86_64_regs::{RBX, RCX, R11};
        let copies = vec![(RCX, RBX), (RBX, RCX)];
        let resolved = resolve_physreg_parallel_copy(&copies, R11);

        // Simulate execution on a concrete register file to verify semantics.
        // Initial state: RBX = 0xBBBB, RCX = 0xCCCC.
        let mut bx: u64 = 0xBBBB;
        let mut cx: u64 = 0xCCCC;
        let mut r11: u64 = 0;
        for (dst, src) in &resolved {
            let v = if *src == RBX { bx }
                    else if *src == RCX { cx }
                    else if *src == R11 { r11 }
                    else { panic!("unexpected src {:?}", src) };
            if *dst == RBX { bx = v; }
            else if *dst == RCX { cx = v; }
            else if *dst == R11 { r11 = v; }
            else { panic!("unexpected dst {:?}", dst); }
        }
        // After parallel swap semantics: RBX should hold the old RCX, RCX the old RBX.
        assert_eq!(bx, 0xCCCC, "RBX should now hold old RCX value");
        assert_eq!(cx, 0xBBBB, "RCX should now hold old RBX value");
    }

    #[test]
    fn test_resolve_physreg_parallel_copy_rdi_rcx_rbx_chain() {
        // The bug pattern from #497:
        //   v0 (i0) <- RDI, v3 (i3) <- RCX  after regalloc:
        //     RCX <- RDI (v0 now in RCX)
        //     RBX <- RCX (v3 now in RBX) -- but this reads RCX which was clobbered.
        // Parallel-copy resolution must reorder or use scratch.
        use llvm2_ir::x86_64_regs::{RBX, RCX, RDI, R11};
        let copies = vec![(RCX, RDI), (RBX, RCX)];
        let resolved = resolve_physreg_parallel_copy(&copies, R11);

        // Simulate: initial RDI=0xD, RCX=0xC, RBX=0xB0.
        let mut di: u64 = 0xD;
        let mut cx: u64 = 0xC;
        let mut bx: u64 = 0xB0;
        let mut r11: u64 = 0;
        for (dst, src) in &resolved {
            let v = match src {
                r if *r == RDI => di,
                r if *r == RCX => cx,
                r if *r == RBX => bx,
                r if *r == R11 => r11,
                _ => panic!("unexpected src"),
            };
            match dst {
                d if *d == RDI => di = v,
                d if *d == RCX => cx = v,
                d if *d == RBX => bx = v,
                d if *d == R11 => r11 = v,
                _ => panic!("unexpected dst"),
            }
        }
        // Parallel semantics: RCX gets old RDI (=0xD), RBX gets old RCX (=0xC).
        assert_eq!(cx, 0xD, "RCX should hold old RDI");
        assert_eq!(bx, 0xC, "RBX should hold old RCX (the incoming i3)");
    }

    #[test]
    fn test_fixup_formal_arg_parallel_copy_rdi_rcx_rbx_cycle() {
        // End-to-end on an X86ISelFunction: emit the exact pattern
        // lower_formal_arguments would emit, then verify the fixup reorders
        // them into a correct sequence.
        use llvm2_ir::x86_64_regs::{RBX, RCX, RDI, R11};
        let mut func = minimal_func("fixup_formal_arg_test");
        let entry = Block(0);

        let v0 = VReg::new(0, RegClass::Gpr64);
        let v1 = VReg::new(1, RegClass::Gpr64);
        func.next_vreg = 2;

        // MOV v0, RDI ; MOV v1, RCX  — formal-arg prologue
        func.push_inst(entry, X86ISelInst::new(
            X86Opcode::MovRR,
            vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(RDI)],
        ));
        func.push_inst(entry, X86ISelInst::new(
            X86Opcode::MovRR,
            vec![X86ISelOperand::VReg(v1), X86ISelOperand::PReg(RCX)],
        ));
        // Tail: a RET to prove we don't disturb what follows the prefix.
        func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));

        // Allocation: v0 -> RCX (clash!), v1 -> RBX.
        // If emitted sequentially as MOV RCX,RDI; MOV RBX,RCX, the second
        // move reads a clobbered RCX. The fixup must fix this.
        let mut alloc = HashMap::new();
        alloc.insert(v0, RCX);
        alloc.insert(v1, RBX);

        fixup_formal_arg_parallel_copy(&mut func, &alloc);

        let insts = &func.blocks[&entry].insts;
        // Last instruction should still be the RET.
        assert_eq!(insts.last().unwrap().opcode, X86Opcode::Ret);

        // Simulate execution starting from incoming register values.
        // Initial: RDI=0xAAA (i0), RCX=0xCCC (i1). RBX scratch.
        let mut di: u64 = 0xAAA;
        let mut cx: u64 = 0xCCC;
        let mut bx: u64 = 0;
        let mut r11: u64 = 0;
        for inst in &insts[..insts.len() - 1] {
            assert_eq!(inst.opcode, X86Opcode::MovRR, "prefix must be all MovRR");
            let dst = match &inst.operands[0] {
                X86ISelOperand::PReg(p) => *p,
                other => panic!("expected PReg dst after fixup, got {:?}", other),
            };
            let src = match &inst.operands[1] {
                X86ISelOperand::PReg(p) => *p,
                other => panic!("expected PReg src after fixup, got {:?}", other),
            };
            let v = match src {
                r if r == RDI => di,
                r if r == RCX => cx,
                r if r == RBX => bx,
                r if r == R11 => r11,
                _ => panic!("unexpected src {:?}", src),
            };
            match dst {
                d if d == RDI => di = v,
                d if d == RCX => cx = v,
                d if d == RBX => bx = v,
                d if d == R11 => r11 = v,
                _ => panic!("unexpected dst {:?}", dst),
            }
        }
        // Parallel semantics:
        //   v0 in RCX = old RDI = 0xAAA (i0)
        //   v1 in RBX = old RCX = 0xCCC (i1)
        assert_eq!(cx, 0xAAA, "v0/RCX must receive incoming i0 (RDI)");
        assert_eq!(bx, 0xCCC, "v1/RBX must receive incoming i1 (old RCX)");
    }
}
