// llvm2-codegen/riscv/pipeline.rs - RISC-V end-to-end compilation pipeline
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: RISC-V Unprivileged ISA Specification (Volume 1, Version 20191213)
// Reference: RISC-V ELF psABI Specification (calling convention)

//! RISC-V RV64GC end-to-end compilation pipeline.
//!
//! Takes a RISC-V ISel function (`RiscVISelFunction`) and produces encoded
//! machine code bytes (optionally wrapped in an ELF .o file).
//!
//! # Pipeline phases
//!
//! ```text
//! Phase 1: Instruction Selection
//!   tMIR Function -> RiscVISelFunction (RiscVOpcodes, VRegs)
//!
//! Phase 2: RISC-V prologue/epilogue insertion
//!   Stack frame setup/teardown for RISC-V LP64D ABI
//!
//! Phase 3: Branch resolution
//!   Resolve block references to byte offsets (fixed 4-byte encoding)
//!
//! Phase 4: Encoding (llvm2-codegen/riscv/encode)
//!   RiscVISelFunction -> Vec<u8> (machine code bytes)
//!
//! Phase 5: ELF emission (optional)
//!   Vec<u8> -> ELF .o file bytes
//! ```
//!
//! # Note on register allocation
//!
//! The existing register allocator (`llvm2-regalloc`) operates on
//! `llvm2_ir::MachFunction` which uses AArch64-centric types. The RISC-V
//! ISel produces `RiscVISelFunction` with `RiscVPReg` and `RiscVOpcode` --
//! a separate type universe. For the initial pipeline, we implement a
//! simplified register assignment that maps VRegs directly to physical
//! registers. A full RISC-V regalloc adapter will be built in a follow-up.

use std::collections::HashMap;

use llvm2_ir::regs::{RegClass, VReg};
use llvm2_ir::riscv_ops::RiscVOpcode;
use llvm2_ir::riscv_regs::{
    self, RiscVPReg,
    RISCV_ALLOCATABLE_GPRS, RISCV_ALLOCATABLE_FPRS,
    RISCV_CALLEE_SAVED_GPRS,
    RA, SP, S0, A0, A1,
};

use crate::riscv::encode::{
    RiscVEncodeError, RiscVInstOperands,
    encode_instruction,
};
use crate::elf::{ElfMachine, ElfWriter};

// ---------------------------------------------------------------------------
// ISel types for RISC-V (parallel to x86_64_isel types)
// ---------------------------------------------------------------------------

/// Operand in a RISC-V ISel instruction.
#[derive(Debug, Clone, PartialEq)]
pub enum RiscVISelOperand {
    /// Virtual register.
    VReg(VReg),
    /// Physical register (for ABI constraints).
    PReg(RiscVPReg),
    /// Immediate integer.
    Imm(i64),
    /// Basic block target (for branches).
    Block(llvm2_lower::instructions::Block),
    /// Global symbol name (for call relocations).
    Symbol(String),
    /// Stack slot index (resolved during frame lowering).
    StackSlot(u32),
}

/// A RISC-V ISel instruction: opcode + operands.
#[derive(Debug, Clone)]
pub struct RiscVISelInst {
    pub opcode: RiscVOpcode,
    pub operands: Vec<RiscVISelOperand>,
}

impl RiscVISelInst {
    pub fn new(opcode: RiscVOpcode, operands: Vec<RiscVISelOperand>) -> Self {
        Self { opcode, operands }
    }
}

/// A RISC-V ISel basic block.
#[derive(Debug, Clone, Default)]
pub struct RiscVISelBlock {
    pub insts: Vec<RiscVISelInst>,
    pub successors: Vec<llvm2_lower::instructions::Block>,
}

/// A RISC-V ISel function containing RiscVISelInsts with VRegs.
#[derive(Debug, Clone)]
pub struct RiscVISelFunction {
    pub name: String,
    pub sig: llvm2_lower::function::Signature,
    pub blocks: HashMap<llvm2_lower::instructions::Block, RiscVISelBlock>,
    pub block_order: Vec<llvm2_lower::instructions::Block>,
    pub next_vreg: u32,
}

impl RiscVISelFunction {
    pub fn new(name: String, sig: llvm2_lower::function::Signature) -> Self {
        Self {
            name,
            sig,
            blocks: HashMap::new(),
            block_order: Vec::new(),
            next_vreg: 0,
        }
    }

    /// Emit a machine instruction into the given block.
    pub fn push_inst(&mut self, block: llvm2_lower::instructions::Block, inst: RiscVISelInst) {
        self.blocks.entry(block).or_default().insts.push(inst);
    }

    /// Add a block to the function (if not already present).
    pub fn ensure_block(&mut self, block: llvm2_lower::instructions::Block) {
        if let std::collections::hash_map::Entry::Vacant(e) = self.blocks.entry(block) {
            e.insert(RiscVISelBlock::default());
            self.block_order.push(block);
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline errors
// ---------------------------------------------------------------------------

/// Errors during RISC-V compilation.
#[derive(Debug)]
pub enum RiscVPipelineError {
    /// Instruction selection failed.
    ISel(String),
    /// Register allocation ran out of registers.
    RegAlloc(String),
    /// Encoding failed.
    Encoding(RiscVEncodeError),
    /// Prologue/epilogue generation failed.
    FrameLowering(String),
}

impl core::fmt::Display for RiscVPipelineError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ISel(msg) => write!(f, "RISC-V ISel failed: {}", msg),
            Self::RegAlloc(msg) => write!(f, "RISC-V regalloc failed: {}", msg),
            Self::Encoding(e) => write!(f, "RISC-V encoding failed: {}", e),
            Self::FrameLowering(msg) => write!(f, "RISC-V frame lowering failed: {}", msg),
        }
    }
}

impl From<RiscVEncodeError> for RiscVPipelineError {
    fn from(e: RiscVEncodeError) -> Self {
        Self::Encoding(e)
    }
}

// ---------------------------------------------------------------------------
// Simple register allocator for RISC-V
// ---------------------------------------------------------------------------

/// Simple linear-scan register assignment for RISC-V ISel output.
///
/// Assigns VRegs to physical registers in order of first appearance.
/// This is a temporary solution until the main regalloc is adapted for
/// RISC-V types.
pub struct RiscVRegAssignment {
    /// VReg -> physical register mapping.
    pub allocation: HashMap<VReg, RiscVPReg>,
    /// Set of callee-saved registers that were used (need save/restore).
    pub used_callee_saved: Vec<RiscVPReg>,
    /// Number of spill slots needed.
    pub num_spills: u32,
}

impl RiscVRegAssignment {
    /// Perform register assignment on an ISel function.
    pub fn assign(func: &RiscVISelFunction) -> Result<Self, RiscVPipelineError> {
        let mut allocation: HashMap<VReg, RiscVPReg> = HashMap::new();
        let mut gpr_idx: usize = 0;
        let mut fpr_idx: usize = 0;
        let mut used_callee_saved: Vec<RiscVPReg> = Vec::new();

        // Collect all VRegs referenced in the function.
        let mut vregs: Vec<VReg> = Vec::new();
        for block in func.block_order.iter() {
            if let Some(mblock) = func.blocks.get(block) {
                for inst in &mblock.insts {
                    for op in &inst.operands {
                        if let RiscVISelOperand::VReg(v) = op
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
                if fpr_idx < RISCV_ALLOCATABLE_FPRS.len() {
                    let preg = RISCV_ALLOCATABLE_FPRS[fpr_idx];
                    allocation.insert(*vreg, preg);
                    fpr_idx += 1;
                } else {
                    return Err(RiscVPipelineError::RegAlloc(format!(
                        "ran out of FPR registers for vreg v{}",
                        vreg.id
                    )));
                }
            } else if gpr_idx < RISCV_ALLOCATABLE_GPRS.len() {
                let preg = RISCV_ALLOCATABLE_GPRS[gpr_idx];
                allocation.insert(*vreg, preg);

                // Track callee-saved usage.
                if RISCV_CALLEE_SAVED_GPRS.contains(&preg)
                    && !used_callee_saved.contains(&preg)
                {
                    used_callee_saved.push(preg);
                }

                gpr_idx += 1;
            } else {
                return Err(RiscVPipelineError::RegAlloc(format!(
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
// Operand resolution: RiscVISelOperand -> RiscVInstOperands for the encoder
// ---------------------------------------------------------------------------

/// Resolve an ISel operand to a physical register after allocation.
fn resolve_operand(
    op: &RiscVISelOperand,
    alloc: &HashMap<VReg, RiscVPReg>,
) -> Option<RiscVPReg> {
    match op {
        RiscVISelOperand::VReg(v) => alloc.get(v).copied(),
        RiscVISelOperand::PReg(p) => Some(*p),
        _ => None,
    }
}

/// Convert an ISel instruction to encoder operands using the register assignment.
///
/// RISC-V is a 3-operand ISA: most instructions are `rd, rs1, rs2` or
/// `rd, rs1, imm12`. This is simpler than x86-64's 2-address form.
fn resolve_inst_operands(
    inst: &RiscVISelInst,
    alloc: &HashMap<VReg, RiscVPReg>,
) -> RiscVInstOperands {
    let mut ops = RiscVInstOperands::none();

    match inst.opcode {
        // =================================================================
        // Pseudo-instructions: no operands needed.
        // =================================================================
        RiscVOpcode::Nop | RiscVOpcode::Phi | RiscVOpcode::StackAlloc => {}

        // =================================================================
        // R-type: [rd, rs1, rs2] (three-register)
        // =================================================================
        RiscVOpcode::Add | RiscVOpcode::Sub | RiscVOpcode::And
        | RiscVOpcode::Or | RiscVOpcode::Xor | RiscVOpcode::Sll
        | RiscVOpcode::Srl | RiscVOpcode::Sra | RiscVOpcode::Slt
        | RiscVOpcode::Sltu
        | RiscVOpcode::Addw | RiscVOpcode::Subw | RiscVOpcode::Sllw
        | RiscVOpcode::Srlw | RiscVOpcode::Sraw
        | RiscVOpcode::Mul | RiscVOpcode::Mulh | RiscVOpcode::Mulhsu
        | RiscVOpcode::Mulhu | RiscVOpcode::Div | RiscVOpcode::Divu
        | RiscVOpcode::Rem | RiscVOpcode::Remu
        | RiscVOpcode::Mulw | RiscVOpcode::Divw | RiscVOpcode::Divuw
        | RiscVOpcode::Remw | RiscVOpcode::Remuw
        | RiscVOpcode::FaddD | RiscVOpcode::FsubD | RiscVOpcode::FmulD
        | RiscVOpcode::FdivD
        | RiscVOpcode::FeqD | RiscVOpcode::FltD | RiscVOpcode::FleD => {
            if inst.operands.len() >= 3 {
                ops.rd = resolve_operand(&inst.operands[0], alloc);
                ops.rs1 = resolve_operand(&inst.operands[1], alloc);
                ops.rs2 = resolve_operand(&inst.operands[2], alloc);
            }
        }

        // =================================================================
        // I-type: [rd, rs1, imm12] (register-immediate)
        // =================================================================
        RiscVOpcode::Addi | RiscVOpcode::Andi | RiscVOpcode::Ori
        | RiscVOpcode::Xori | RiscVOpcode::Slti | RiscVOpcode::Sltiu
        | RiscVOpcode::Slli | RiscVOpcode::Srli | RiscVOpcode::Srai
        | RiscVOpcode::Addiw | RiscVOpcode::Slliw | RiscVOpcode::Srliw
        | RiscVOpcode::Sraiw => {
            if inst.operands.len() >= 2 {
                ops.rd = resolve_operand(&inst.operands[0], alloc);
                ops.rs1 = resolve_operand(&inst.operands[1], alloc);
            }
            for op in &inst.operands {
                if let RiscVISelOperand::Imm(imm) = op {
                    ops.imm = *imm as i32;
                    break;
                }
            }
        }

        // =================================================================
        // I-type: loads [rd, rs1(base), imm12(offset)]
        // =================================================================
        RiscVOpcode::Lb | RiscVOpcode::Lh | RiscVOpcode::Lw
        | RiscVOpcode::Ld | RiscVOpcode::Lbu | RiscVOpcode::Lhu
        | RiscVOpcode::Lwu | RiscVOpcode::Fld => {
            if inst.operands.len() >= 2 {
                ops.rd = resolve_operand(&inst.operands[0], alloc);
                ops.rs1 = resolve_operand(&inst.operands[1], alloc);
            }
            for op in &inst.operands {
                if let RiscVISelOperand::Imm(imm) = op {
                    ops.imm = *imm as i32;
                    break;
                }
            }
        }

        // =================================================================
        // S-type: stores [rs2(source), rs1(base), imm12(offset)]
        // =================================================================
        RiscVOpcode::Sb | RiscVOpcode::Sh | RiscVOpcode::Sw
        | RiscVOpcode::Sd | RiscVOpcode::Fsd => {
            if inst.operands.len() >= 2 {
                // ISel format: [src_reg, base_reg, offset]
                ops.rs2 = resolve_operand(&inst.operands[0], alloc);
                ops.rs1 = resolve_operand(&inst.operands[1], alloc);
            }
            for op in &inst.operands {
                if let RiscVISelOperand::Imm(imm) = op {
                    ops.imm = *imm as i32;
                    break;
                }
            }
        }

        // =================================================================
        // B-type: branches [rs1, rs2, offset]
        // =================================================================
        RiscVOpcode::Beq | RiscVOpcode::Bne | RiscVOpcode::Blt
        | RiscVOpcode::Bge | RiscVOpcode::Bltu | RiscVOpcode::Bgeu => {
            if inst.operands.len() >= 2 {
                ops.rs1 = resolve_operand(&inst.operands[0], alloc);
                ops.rs2 = resolve_operand(&inst.operands[1], alloc);
            }
            for op in &inst.operands {
                if let RiscVISelOperand::Imm(imm) = op {
                    ops.imm = *imm as i32;
                    break;
                }
            }
        }

        // =================================================================
        // U-type: [rd, imm20]
        // =================================================================
        RiscVOpcode::Lui | RiscVOpcode::Auipc => {
            if let Some(op) = inst.operands.first() {
                ops.rd = resolve_operand(op, alloc);
            }
            for op in &inst.operands {
                if let RiscVISelOperand::Imm(imm) = op {
                    ops.imm = *imm as i32;
                    break;
                }
            }
        }

        // =================================================================
        // J-type: JAL [rd, offset]
        // =================================================================
        RiscVOpcode::Jal => {
            if let Some(op) = inst.operands.first() {
                ops.rd = resolve_operand(op, alloc);
            }
            for op in &inst.operands {
                if let RiscVISelOperand::Imm(imm) = op {
                    ops.imm = *imm as i32;
                    break;
                }
            }
        }

        // =================================================================
        // JALR: [rd, rs1, imm12]
        // =================================================================
        RiscVOpcode::Jalr => {
            if inst.operands.len() >= 2 {
                ops.rd = resolve_operand(&inst.operands[0], alloc);
                ops.rs1 = resolve_operand(&inst.operands[1], alloc);
            }
            for op in &inst.operands {
                if let RiscVISelOperand::Imm(imm) = op {
                    ops.imm = *imm as i32;
                    break;
                }
            }
        }

        // =================================================================
        // Unary FP: [rd, rs1] (FSQRT.D, conversions, moves)
        // =================================================================
        RiscVOpcode::FsqrtD
        | RiscVOpcode::FcvtDW | RiscVOpcode::FcvtWD
        | RiscVOpcode::FcvtDL | RiscVOpcode::FcvtLD
        | RiscVOpcode::FmvXD | RiscVOpcode::FmvDX => {
            if inst.operands.len() >= 2 {
                ops.rd = resolve_operand(&inst.operands[0], alloc);
                ops.rs1 = resolve_operand(&inst.operands[1], alloc);
            }
        }
    }

    ops
}

// ---------------------------------------------------------------------------
// RISC-V LP64D ABI prologue/epilogue
// ---------------------------------------------------------------------------

/// Compute the stack frame size (16-byte aligned).
///
/// RISC-V LP64D frame layout:
/// ```text
/// [caller's frame]
/// return address (saved ra)
/// saved s0/fp
/// callee-saved regs
/// local variables / spill slots
/// [outgoing args area]  <- SP (16-byte aligned)
/// ```
fn compute_frame_size(num_callee_saved: usize, num_spills: u32) -> u32 {
    // RA + S0/FP + callee-saved registers + spill slots
    let total_slots = 2 + num_callee_saved as u32 + num_spills;
    let total_bytes = total_slots * 8;

    // Round up to 16-byte alignment.
    (total_bytes + 15) & !15
}

/// Generate prologue instructions for RISC-V LP64D ABI.
fn generate_prologue(
    callee_saved: &[RiscVPReg],
    frame_size: u32,
) -> Vec<RiscVISelInst> {
    let mut prologue = Vec::new();

    // ADDI SP, SP, -frame_size (allocate stack frame)
    if frame_size > 0 {
        prologue.push(RiscVISelInst::new(
            RiscVOpcode::Addi,
            vec![
                RiscVISelOperand::PReg(SP),
                RiscVISelOperand::PReg(SP),
                RiscVISelOperand::Imm(-(frame_size as i64)),
            ],
        ));
    }

    // SD RA, frame_size-8(SP) (save return address)
    let ra_offset = frame_size as i64 - 8;
    prologue.push(RiscVISelInst::new(
        RiscVOpcode::Sd,
        vec![
            RiscVISelOperand::PReg(RA),
            RiscVISelOperand::PReg(SP),
            RiscVISelOperand::Imm(ra_offset),
        ],
    ));

    // SD S0, frame_size-16(SP) (save frame pointer)
    let fp_offset = frame_size as i64 - 16;
    prologue.push(RiscVISelInst::new(
        RiscVOpcode::Sd,
        vec![
            RiscVISelOperand::PReg(S0),
            RiscVISelOperand::PReg(SP),
            RiscVISelOperand::Imm(fp_offset),
        ],
    ));

    // ADDI S0, SP, frame_size (set frame pointer)
    prologue.push(RiscVISelInst::new(
        RiscVOpcode::Addi,
        vec![
            RiscVISelOperand::PReg(S0),
            RiscVISelOperand::PReg(SP),
            RiscVISelOperand::Imm(frame_size as i64),
        ],
    ));

    // Save callee-saved registers.
    for (i, &reg) in callee_saved.iter().enumerate() {
        let offset = frame_size as i64 - 24 - (i as i64 * 8);
        prologue.push(RiscVISelInst::new(
            RiscVOpcode::Sd,
            vec![
                RiscVISelOperand::PReg(reg),
                RiscVISelOperand::PReg(SP),
                RiscVISelOperand::Imm(offset),
            ],
        ));
    }

    prologue
}

/// Generate epilogue instructions for RISC-V LP64D ABI.
fn generate_epilogue(
    callee_saved: &[RiscVPReg],
    frame_size: u32,
) -> Vec<RiscVISelInst> {
    let mut epilogue = Vec::new();

    // Restore callee-saved registers in reverse order.
    for (i, &reg) in callee_saved.iter().enumerate().rev() {
        let offset = frame_size as i64 - 24 - (i as i64 * 8);
        epilogue.push(RiscVISelInst::new(
            RiscVOpcode::Ld,
            vec![
                RiscVISelOperand::PReg(reg),
                RiscVISelOperand::PReg(SP),
                RiscVISelOperand::Imm(offset),
            ],
        ));
    }

    // LD RA, frame_size-8(SP) (restore return address)
    let ra_offset = frame_size as i64 - 8;
    epilogue.push(RiscVISelInst::new(
        RiscVOpcode::Ld,
        vec![
            RiscVISelOperand::PReg(RA),
            RiscVISelOperand::PReg(SP),
            RiscVISelOperand::Imm(ra_offset),
        ],
    ));

    // LD S0, frame_size-16(SP) (restore frame pointer)
    let fp_offset = frame_size as i64 - 16;
    epilogue.push(RiscVISelInst::new(
        RiscVOpcode::Ld,
        vec![
            RiscVISelOperand::PReg(S0),
            RiscVISelOperand::PReg(SP),
            RiscVISelOperand::Imm(fp_offset),
        ],
    ));

    // ADDI SP, SP, frame_size (deallocate stack frame)
    if frame_size > 0 {
        epilogue.push(RiscVISelInst::new(
            RiscVOpcode::Addi,
            vec![
                RiscVISelOperand::PReg(SP),
                RiscVISelOperand::PReg(SP),
                RiscVISelOperand::Imm(frame_size as i64),
            ],
        ));
    }

    epilogue
}

// ---------------------------------------------------------------------------
// Branch resolution for fixed-length RISC-V instructions
// ---------------------------------------------------------------------------

/// Return the encoded byte size of a RISC-V instruction.
///
/// All RISC-V base ISA instructions are exactly 4 bytes (32 bits).
/// Pseudo-instructions (Phi, StackAlloc, Nop-as-pseudo) encode as 4 bytes
/// since NOP is ADDI x0, x0, 0.
fn inst_size(inst: &RiscVISelInst) -> usize {
    match inst.opcode {
        // Pseudo-instructions that we skip during encoding.
        RiscVOpcode::Phi | RiscVOpcode::StackAlloc => 0,
        // Everything else is 4 bytes.
        _ => 4,
    }
}

/// Resolve block operands in branch instructions to byte offsets.
///
/// RISC-V branches use PC-relative offsets where the offset is relative
/// to the address of the branch instruction itself (not the next instruction
/// as in x86-64).
fn resolve_riscv_branches(func: &mut RiscVISelFunction) {
    use llvm2_lower::instructions::Block;

    // Phase 1: Compute byte offset of each block.
    let mut block_offsets: HashMap<Block, i64> = HashMap::new();
    let mut current_offset: i64 = 0;

    for &block_id in &func.block_order {
        block_offsets.insert(block_id, current_offset);
        if let Some(mblock) = func.blocks.get(&block_id) {
            for inst in &mblock.insts {
                current_offset += inst_size(inst) as i64;
            }
        }
    }

    // Phase 2: Replace Block operands with Imm offsets.
    let block_order = func.block_order.clone();
    for &block_id in &block_order {
        let mut inst_offset: i64 = *block_offsets.get(&block_id).unwrap_or(&0);

        if let Some(mblock) = func.blocks.get_mut(&block_id) {
            for inst in &mut mblock.insts {
                let isize_val = inst_size(inst) as i64;

                let is_branch = matches!(
                    inst.opcode,
                    RiscVOpcode::Beq | RiscVOpcode::Bne | RiscVOpcode::Blt
                    | RiscVOpcode::Bge | RiscVOpcode::Bltu | RiscVOpcode::Bgeu
                    | RiscVOpcode::Jal
                );

                if is_branch {
                    // RISC-V: offset is relative to the branch instruction itself.
                    let branch_addr = inst_offset;

                    inst.operands = inst
                        .operands
                        .iter()
                        .map(|op| {
                            if let RiscVISelOperand::Block(target_block) = op {
                                if let Some(&target_offset) = block_offsets.get(target_block) {
                                    let rel_offset = target_offset - branch_addr;
                                    RiscVISelOperand::Imm(rel_offset)
                                } else {
                                    op.clone()
                                }
                            } else {
                                op.clone()
                            }
                        })
                        .collect();
                }

                inst_offset += isize_val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RiscVPipeline -- main entry point
// ---------------------------------------------------------------------------

/// Configuration for the RISC-V pipeline.
#[derive(Debug, Clone)]
pub struct RiscVPipelineConfig {
    /// Whether to emit an ELF .o wrapper (vs raw code bytes).
    pub emit_elf: bool,
    /// Whether to emit prologue/epilogue (false for leaf functions that
    /// don't need a frame).
    pub emit_frame: bool,
}

impl Default for RiscVPipelineConfig {
    fn default() -> Self {
        Self {
            emit_elf: false,
            emit_frame: true,
        }
    }
}

/// The RISC-V compilation pipeline.
///
/// Orchestrates: ISel output -> regalloc -> frame lowering -> encoding -> ELF.
pub struct RiscVPipeline {
    pub config: RiscVPipelineConfig,
}

impl RiscVPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: RiscVPipelineConfig) -> Self {
        Self { config }
    }

    /// Create a pipeline with default configuration (raw code bytes, with frame).
    pub fn default_config() -> Self {
        Self::new(RiscVPipelineConfig::default())
    }

    /// Compile a RISC-V ISel function to machine code bytes.
    ///
    /// This is the main entry point. It takes a `RiscVISelFunction` (post-ISel,
    /// pre-regalloc) and returns encoded machine code bytes.
    pub fn compile_function(
        &self,
        func: &RiscVISelFunction,
    ) -> Result<Vec<u8>, RiscVPipelineError> {
        // Phase 1: Register assignment.
        let assignment = RiscVRegAssignment::assign(func)?;

        // Phase 2: Clone function and insert prologue/epilogue.
        let mut func = func.clone();
        if self.config.emit_frame {
            self.insert_prologue_epilogue(&mut func, &assignment);
        }

        // Phase 3: Resolve branch offsets.
        resolve_riscv_branches(&mut func);

        // Phase 4: Encode all instructions.
        let code = self.encode_function(&func, &assignment.allocation)?;

        // Phase 5: Optionally wrap in ELF.
        if self.config.emit_elf {
            Ok(self.emit_elf(&func.name, &code))
        } else {
            Ok(code)
        }
    }

    // --- Phase implementations ---

    /// Insert prologue/epilogue into the function.
    fn insert_prologue_epilogue(
        &self,
        func: &mut RiscVISelFunction,
        assignment: &RiscVRegAssignment,
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
                let mut new_insts = prologue;
                new_insts.append(&mut mblock.insts);
                mblock.insts = new_insts;
            }

        // Insert epilogue before every JALR that acts as RET.
        // Convention: JALR x0, ra, 0 is the return instruction.
        for block_id in func.block_order.clone() {
            if let Some(mblock) = func.blocks.get_mut(&block_id) {
                let mut new_insts = Vec::new();
                for inst in &mblock.insts {
                    if is_ret_inst(inst) {
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
        func: &RiscVISelFunction,
        alloc: &HashMap<VReg, RiscVPReg>,
    ) -> Result<Vec<u8>, RiscVPipelineError> {
        let mut bytes: Vec<u8> = Vec::new();

        for &block_id in &func.block_order {
            if let Some(mblock) = func.blocks.get(&block_id) {
                for inst in &mblock.insts {
                    // Skip pseudo-instructions that produce no code.
                    if matches!(
                        inst.opcode,
                        RiscVOpcode::Phi | RiscVOpcode::StackAlloc
                    ) {
                        continue;
                    }

                    let ops = resolve_inst_operands(inst, alloc);
                    let word = encode_instruction(inst.opcode, &ops)
                        .map_err(RiscVPipelineError::from)?;

                    // RISC-V is little-endian: emit 4 bytes in LE order.
                    bytes.extend_from_slice(&word.to_le_bytes());
                }
            }
        }

        Ok(bytes)
    }

    /// Emit an ELF .o file wrapping the encoded machine code.
    fn emit_elf(&self, func_name: &str, code: &[u8]) -> Vec<u8> {
        let mut writer = ElfWriter::new(ElfMachine::Riscv64);
        writer.add_text_section(code);
        writer.add_symbol(func_name, 1, 0, code.len() as u64, true, 2); // STT_FUNC
        writer.write()
    }
}

/// Check whether an ISel instruction is a return (JALR x0, ra, 0).
fn is_ret_inst(inst: &RiscVISelInst) -> bool {
    if inst.opcode != RiscVOpcode::Jalr {
        return false;
    }
    // Check for JALR x0, ra, 0 pattern (rd=x0, rs1=ra, imm=0).
    if inst.operands.len() >= 2 {
        let rd_is_zero = matches!(&inst.operands[0],
            RiscVISelOperand::PReg(p) if *p == riscv_regs::ZERO
        );
        let rs1_is_ra = matches!(&inst.operands[1],
            RiscVISelOperand::PReg(p) if *p == riscv_regs::RA
        );
        return rd_is_zero && rs1_is_ra;
    }
    false
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Compile a RISC-V ISel function to raw machine code bytes.
pub fn riscv_compile_to_bytes(
    func: &RiscVISelFunction,
) -> Result<Vec<u8>, RiscVPipelineError> {
    let pipeline = RiscVPipeline::default_config();
    pipeline.compile_function(func)
}

/// Compile a RISC-V ISel function to an ELF .o file.
pub fn riscv_compile_to_elf(
    func: &RiscVISelFunction,
) -> Result<Vec<u8>, RiscVPipelineError> {
    let pipeline = RiscVPipeline::new(RiscVPipelineConfig {
        emit_elf: true,
        emit_frame: true,
    });
    pipeline.compile_function(func)
}

/// Build a simple `add(a: i64, b: i64) -> i64` RISC-V ISel function for testing.
///
/// RISC-V LP64D ABI: a in a0, b in a1, return in a0.
pub fn build_riscv_add_test_function() -> RiscVISelFunction {
    use llvm2_lower::function::Signature;
    use llvm2_lower::instructions::Block;
    use llvm2_lower::types::Type;

    let sig = Signature {
        params: vec![Type::I64, Type::I64],
        returns: vec![Type::I64],
    };

    let mut func = RiscVISelFunction::new("add".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    let v0 = VReg::new(0, RegClass::Gpr64);
    let v1 = VReg::new(1, RegClass::Gpr64);
    let v2 = VReg::new(2, RegClass::Gpr64);
    func.next_vreg = 3;

    // ADDI v0, a0, 0 (move arg 0)
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Addi,
        vec![
            RiscVISelOperand::VReg(v0),
            RiscVISelOperand::PReg(A0),
            RiscVISelOperand::Imm(0),
        ],
    ));

    // ADDI v1, a1, 0 (move arg 1)
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Addi,
        vec![
            RiscVISelOperand::VReg(v1),
            RiscVISelOperand::PReg(A1),
            RiscVISelOperand::Imm(0),
        ],
    ));

    // ADD v2, v0, v1
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Add,
        vec![
            RiscVISelOperand::VReg(v2),
            RiscVISelOperand::VReg(v0),
            RiscVISelOperand::VReg(v1),
        ],
    ));

    // ADDI a0, v2, 0 (move result to return register)
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Addi,
        vec![
            RiscVISelOperand::PReg(A0),
            RiscVISelOperand::VReg(v2),
            RiscVISelOperand::Imm(0),
        ],
    ));

    // JALR x0, ra, 0 (return)
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Jalr,
        vec![
            RiscVISelOperand::PReg(riscv_regs::ZERO),
            RiscVISelOperand::PReg(RA),
            RiscVISelOperand::Imm(0),
        ],
    ));

    func
}

/// Build a simple `const42() -> i64` RISC-V ISel function for testing.
///
/// Returns the constant 42.
pub fn build_riscv_const_test_function() -> RiscVISelFunction {
    use llvm2_lower::function::Signature;
    use llvm2_lower::instructions::Block;
    use llvm2_lower::types::Type;

    let sig = Signature {
        params: vec![],
        returns: vec![Type::I64],
    };

    let mut func = RiscVISelFunction::new("const42".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    // LI a0, 42 (pseudo: ADDI a0, x0, 42)
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Addi,
        vec![
            RiscVISelOperand::PReg(A0),
            RiscVISelOperand::PReg(riscv_regs::ZERO),
            RiscVISelOperand::Imm(42),
        ],
    ));

    // JALR x0, ra, 0 (return)
    func.push_inst(entry, RiscVISelInst::new(
        RiscVOpcode::Jalr,
        vec![
            RiscVISelOperand::PReg(riscv_regs::ZERO),
            RiscVISelOperand::PReg(RA),
            RiscVISelOperand::Imm(0),
        ],
    ));

    func
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::regs::{RegClass, VReg};
    use llvm2_ir::riscv_ops::RiscVOpcode;
    use llvm2_ir::riscv_regs::{A0, RA, ZERO};
    use llvm2_lower::function::Signature;
    use llvm2_lower::instructions::Block;

    // -----------------------------------------------------------------------
    // Helper: build a minimal ISel function
    // -----------------------------------------------------------------------

    fn minimal_func(name: &str) -> RiscVISelFunction {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut func = RiscVISelFunction::new(name.to_string(), sig);
        let entry = Block(0);
        func.ensure_block(entry);
        func
    }

    // -----------------------------------------------------------------------
    // Pipeline construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_default_config() {
        let pipeline = RiscVPipeline::default_config();
        assert!(!pipeline.config.emit_elf);
        assert!(pipeline.config.emit_frame);
    }

    #[test]
    fn test_pipeline_custom_config() {
        let config = RiscVPipelineConfig {
            emit_elf: true,
            emit_frame: false,
        };
        let pipeline = RiscVPipeline::new(config);
        assert!(pipeline.config.emit_elf);
        assert!(!pipeline.config.emit_frame);
    }

    // -----------------------------------------------------------------------
    // Frame size computation
    // -----------------------------------------------------------------------

    #[test]
    fn test_frame_size_no_callee_saved_no_spills() {
        // RA + S0 = 2 slots = 16 bytes (already aligned)
        let size = compute_frame_size(0, 0);
        assert_eq!(size, 16);
        assert_eq!(size % 16, 0);
    }

    #[test]
    fn test_frame_size_with_spills() {
        let size = compute_frame_size(0, 2);
        // RA + S0 + 2 spills = 4 slots = 32 bytes
        assert_eq!(size, 32);
        assert_eq!(size % 16, 0);
    }

    #[test]
    fn test_frame_size_alignment() {
        for num_cs in 0..8 {
            for num_spill in 0..5 {
                let size = compute_frame_size(num_cs, num_spill);
                assert_eq!(
                    size % 16, 0,
                    "misaligned for callee_saved={}, spills={}: size={}",
                    num_cs, num_spill, size
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Prologue/epilogue generation
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_prologue_minimal() {
        let prologue = generate_prologue(&[], 16);
        // ADDI SP, -16; SD RA, 8(SP); SD S0, 0(SP); ADDI S0, SP, 16
        assert_eq!(prologue.len(), 4);
        assert_eq!(prologue[0].opcode, RiscVOpcode::Addi); // SP -= 16
        assert_eq!(prologue[1].opcode, RiscVOpcode::Sd);   // save RA
        assert_eq!(prologue[2].opcode, RiscVOpcode::Sd);   // save S0
        assert_eq!(prologue[3].opcode, RiscVOpcode::Addi); // set FP
    }

    #[test]
    fn test_generate_epilogue_minimal() {
        let epilogue = generate_epilogue(&[], 16);
        // LD RA, 8(SP); LD S0, 0(SP); ADDI SP, 16
        assert_eq!(epilogue.len(), 3);
        assert_eq!(epilogue[0].opcode, RiscVOpcode::Ld);   // restore RA
        assert_eq!(epilogue[1].opcode, RiscVOpcode::Ld);   // restore S0
        assert_eq!(epilogue[2].opcode, RiscVOpcode::Addi); // SP += 16
    }

    #[test]
    fn test_generate_prologue_with_callee_saved() {
        use llvm2_ir::riscv_regs::{S2, S3};
        let prologue = generate_prologue(&[S2, S3], 48);
        // ADDI SP, -48; SD RA; SD S0; ADDI S0; SD S2; SD S3
        assert_eq!(prologue.len(), 6);
        assert_eq!(prologue[4].opcode, RiscVOpcode::Sd); // save S2
        assert_eq!(prologue[5].opcode, RiscVOpcode::Sd); // save S3
    }

    // -----------------------------------------------------------------------
    // Register assignment
    // -----------------------------------------------------------------------

    #[test]
    fn test_reg_assignment_simple() {
        let func = build_riscv_add_test_function();
        let assignment = RiscVRegAssignment::assign(&func).unwrap();

        // Should have allocated 3 VRegs.
        assert_eq!(assignment.allocation.len(), 3);

        // All assigned to different physical registers.
        let pregs: Vec<RiscVPReg> = assignment.allocation.values().copied().collect();
        for i in 0..pregs.len() {
            for j in (i + 1)..pregs.len() {
                assert_ne!(pregs[i], pregs[j], "duplicate preg assignment");
            }
        }
    }

    #[test]
    fn test_reg_assignment_empty_function() {
        let mut func = minimal_func("empty");
        func.push_inst(Block(0), RiscVISelInst::new(
            RiscVOpcode::Jalr,
            vec![
                RiscVISelOperand::PReg(ZERO),
                RiscVISelOperand::PReg(RA),
                RiscVISelOperand::Imm(0),
            ],
        ));
        let assignment = RiscVRegAssignment::assign(&func).unwrap();
        assert!(assignment.allocation.is_empty());
        assert!(assignment.used_callee_saved.is_empty());
    }

    // -----------------------------------------------------------------------
    // Compile simple functions
    // -----------------------------------------------------------------------

    #[test]
    fn test_compile_void_return() {
        let mut func = minimal_func("void_ret");
        // JALR x0, ra, 0 (return)
        func.push_inst(Block(0), RiscVISelInst::new(
            RiscVOpcode::Jalr,
            vec![
                RiscVISelOperand::PReg(ZERO),
                RiscVISelOperand::PReg(RA),
                RiscVISelOperand::Imm(0),
            ],
        ));

        let pipeline = RiscVPipeline::new(RiscVPipelineConfig {
            emit_elf: false,
            emit_frame: true,
        });
        let code = pipeline.compile_function(&func).unwrap();

        // Should produce non-empty code (prologue + epilogue + RET).
        assert!(!code.is_empty(), "compiled code should not be empty");

        // All instructions are 4 bytes, so total must be multiple of 4.
        assert_eq!(code.len() % 4, 0, "RISC-V code must be 4-byte aligned");
    }

    #[test]
    fn test_compile_void_return_no_frame() {
        let mut func = minimal_func("void_ret_noframe");
        // JALR x0, ra, 0 (return)
        func.push_inst(Block(0), RiscVISelInst::new(
            RiscVOpcode::Jalr,
            vec![
                RiscVISelOperand::PReg(ZERO),
                RiscVISelOperand::PReg(RA),
                RiscVISelOperand::Imm(0),
            ],
        ));

        let pipeline = RiscVPipeline::new(RiscVPipelineConfig {
            emit_elf: false,
            emit_frame: false,
        });
        let code = pipeline.compile_function(&func).unwrap();

        // Without frame, should just be JALR x0, ra, 0.
        // JALR: I-type, opcode=1100111, funct3=000, rd=0, rs1=1, imm=0
        // = 0x00008067
        assert_eq!(code.len(), 4);
        let word = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        assert_eq!(word, 0x00008067, "expected JALR x0, ra, 0 = 0x00008067");
    }

    #[test]
    fn test_compile_const42() {
        let func = build_riscv_const_test_function();
        let pipeline = RiscVPipeline::new(RiscVPipelineConfig {
            emit_elf: false,
            emit_frame: true,
        });
        let code = pipeline.compile_function(&func).unwrap();

        assert!(!code.is_empty());
        assert_eq!(code.len() % 4, 0);
    }

    #[test]
    fn test_compile_add_function() {
        let func = build_riscv_add_test_function();
        let pipeline = RiscVPipeline::new(RiscVPipelineConfig {
            emit_elf: false,
            emit_frame: true,
        });
        let code = pipeline.compile_function(&func).unwrap();

        assert!(!code.is_empty());
        assert_eq!(code.len() % 4, 0);
    }

    #[test]
    fn test_compile_add_function_no_frame() {
        let func = build_riscv_add_test_function();
        let pipeline = RiscVPipeline::new(RiscVPipelineConfig {
            emit_elf: false,
            emit_frame: false,
        });
        let code = pipeline.compile_function(&func).unwrap();

        assert!(!code.is_empty());
        assert_eq!(code.len() % 4, 0);
    }

    // -----------------------------------------------------------------------
    // ELF emission
    // -----------------------------------------------------------------------

    #[test]
    fn test_compile_to_elf() {
        let func = build_riscv_const_test_function();
        let bytes = riscv_compile_to_elf(&func).unwrap();

        // ELF magic: 0x7F 'E' 'L' 'F'
        assert!(bytes.len() > 16);
        assert_eq!(&bytes[0..4], &[0x7F, b'E', b'L', b'F']);

        // ELF class should be ELFCLASS64 (2).
        assert_eq!(bytes[4], 2);

        // Data encoding should be ELFDATA2LSB (1) = little-endian.
        assert_eq!(bytes[5], 1);

        // Machine type for RISC-V should be EM_RISCV (0xF3 = 243).
        let machine = u16::from_le_bytes([bytes[18], bytes[19]]);
        assert_eq!(machine, 0xF3, "ELF machine should be EM_RISCV (0xF3)");
    }

    #[test]
    fn test_compile_add_to_elf() {
        let func = build_riscv_add_test_function();
        let bytes = riscv_compile_to_elf(&func).unwrap();
        assert!(bytes.len() > 64);
        assert_eq!(&bytes[0..4], &[0x7F, b'E', b'L', b'F']);
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
        let mut func = RiscVISelFunction::new("test_br".to_string(), sig);
        let b0 = Block(0);
        let b1 = Block(1);
        func.ensure_block(b0);
        func.ensure_block(b1);

        // b0: JAL x0, b1 (unconditional jump)
        func.push_inst(b0, RiscVISelInst::new(
            RiscVOpcode::Jal,
            vec![
                RiscVISelOperand::PReg(ZERO),
                RiscVISelOperand::Block(b1),
            ],
        ));

        // b1: NOP
        func.push_inst(b1, RiscVISelInst::new(RiscVOpcode::Nop, vec![]));

        resolve_riscv_branches(&mut func);

        // After resolution, JAL should have an Imm operand (not Block).
        let jal = &func.blocks[&b0].insts[0];
        assert_eq!(jal.opcode, RiscVOpcode::Jal);
        // JAL is 4 bytes. Target (b1) starts at offset 4.
        // RISC-V: offset is relative to the branch instruction at offset 0.
        // So offset = 4 - 0 = 4.
        let has_imm = jal.operands.iter().any(|op| {
            matches!(op, RiscVISelOperand::Imm(4))
        });
        assert!(has_imm, "JAL to next block should have offset 4, got {:?}", jal.operands);
    }

    #[test]
    fn test_branch_resolution_backward() {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut func = RiscVISelFunction::new("test_loop".to_string(), sig);
        let b0 = Block(0);
        let b1 = Block(1);
        func.ensure_block(b0);
        func.ensure_block(b1);

        // b0: NOP (4 bytes)
        func.push_inst(b0, RiscVISelInst::new(RiscVOpcode::Nop, vec![]));

        // b1: JAL x0, b0 (backward jump)
        func.push_inst(b1, RiscVISelInst::new(
            RiscVOpcode::Jal,
            vec![
                RiscVISelOperand::PReg(ZERO),
                RiscVISelOperand::Block(b0),
            ],
        ));

        resolve_riscv_branches(&mut func);

        let jal = &func.blocks[&b1].insts[0];
        let has_neg_imm = jal.operands.iter().any(|op| {
            if let RiscVISelOperand::Imm(v) = op { *v < 0 } else { false }
        });
        assert!(has_neg_imm, "backward jump should have negative offset, got {:?}", jal.operands);
    }

    // -----------------------------------------------------------------------
    // Convenience function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_riscv_compile_to_bytes() {
        let func = build_riscv_const_test_function();
        let bytes = riscv_compile_to_bytes(&func).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes.len() % 4, 0);
    }

    // -----------------------------------------------------------------------
    // Operand resolution tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_resolve_operand_vreg() {
        let v0 = VReg::new(0, RegClass::Gpr64);
        let mut alloc = HashMap::new();
        alloc.insert(v0, A0);

        assert_eq!(
            resolve_operand(&RiscVISelOperand::VReg(v0), &alloc),
            Some(A0)
        );
    }

    #[test]
    fn test_resolve_operand_preg() {
        let alloc = HashMap::new();
        assert_eq!(
            resolve_operand(&RiscVISelOperand::PReg(A0), &alloc),
            Some(A0)
        );
    }

    #[test]
    fn test_resolve_operand_imm_returns_none() {
        let alloc = HashMap::new();
        assert_eq!(
            resolve_operand(&RiscVISelOperand::Imm(42), &alloc),
            None
        );
    }

    // -----------------------------------------------------------------------
    // Instruction size
    // -----------------------------------------------------------------------

    #[test]
    fn test_inst_sizes() {
        assert_eq!(inst_size(&RiscVISelInst::new(RiscVOpcode::Phi, vec![])), 0);
        assert_eq!(inst_size(&RiscVISelInst::new(RiscVOpcode::StackAlloc, vec![])), 0);
        assert_eq!(inst_size(&RiscVISelInst::new(RiscVOpcode::Nop, vec![])), 4);
        assert_eq!(inst_size(&RiscVISelInst::new(RiscVOpcode::Add, vec![])), 4);
        assert_eq!(inst_size(&RiscVISelInst::new(RiscVOpcode::Jal, vec![])), 4);
        assert_eq!(inst_size(&RiscVISelInst::new(RiscVOpcode::Sd, vec![])), 4);
    }

    // -----------------------------------------------------------------------
    // Error display
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_error_display() {
        let e1 = RiscVPipelineError::ISel("bad isel".to_string());
        assert!(format!("{}", e1).contains("ISel"));

        let e2 = RiscVPipelineError::RegAlloc("out of regs".to_string());
        assert!(format!("{}", e2).contains("regalloc"));

        let e3 = RiscVPipelineError::FrameLowering("bad frame".to_string());
        assert!(format!("{}", e3).contains("frame lowering"));
    }

    // -----------------------------------------------------------------------
    // Test helper function builders
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_add_function_structure() {
        let func = build_riscv_add_test_function();
        assert_eq!(func.name, "add");
        assert_eq!(func.sig.params.len(), 2);
        assert_eq!(func.sig.returns.len(), 1);
        assert_eq!(func.block_order.len(), 1);
        assert_eq!(func.next_vreg, 3);

        let entry = &func.blocks[&Block(0)];
        // ADDI v0, a0, 0; ADDI v1, a1, 0; ADD v2, v0, v1; ADDI a0, v2, 0; JALR x0, ra, 0
        assert_eq!(entry.insts.len(), 5);
        assert_eq!(entry.insts[0].opcode, RiscVOpcode::Addi);
        assert_eq!(entry.insts[1].opcode, RiscVOpcode::Addi);
        assert_eq!(entry.insts[2].opcode, RiscVOpcode::Add);
        assert_eq!(entry.insts[3].opcode, RiscVOpcode::Addi);
        assert_eq!(entry.insts[4].opcode, RiscVOpcode::Jalr);
    }

    #[test]
    fn test_build_const_function_structure() {
        let func = build_riscv_const_test_function();
        assert_eq!(func.name, "const42");
        assert_eq!(func.sig.params.len(), 0);
        assert_eq!(func.sig.returns.len(), 1);

        let entry = &func.blocks[&Block(0)];
        // ADDI a0, x0, 42; JALR x0, ra, 0
        assert_eq!(entry.insts.len(), 2);
        assert_eq!(entry.insts[0].opcode, RiscVOpcode::Addi);
        assert_eq!(entry.insts[1].opcode, RiscVOpcode::Jalr);
    }

    // -----------------------------------------------------------------------
    // is_ret_inst
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_ret_inst() {
        // Valid RET: JALR x0, ra, 0
        let ret_inst = RiscVISelInst::new(
            RiscVOpcode::Jalr,
            vec![
                RiscVISelOperand::PReg(ZERO),
                RiscVISelOperand::PReg(RA),
                RiscVISelOperand::Imm(0),
            ],
        );
        assert!(is_ret_inst(&ret_inst));

        // Not RET: JALR ra, a0, 0 (function call)
        let call_inst = RiscVISelInst::new(
            RiscVOpcode::Jalr,
            vec![
                RiscVISelOperand::PReg(RA),
                RiscVISelOperand::PReg(A0),
                RiscVISelOperand::Imm(0),
            ],
        );
        assert!(!is_ret_inst(&call_inst));

        // Not RET: ADD instruction
        let add_inst = RiscVISelInst::new(RiscVOpcode::Add, vec![]);
        assert!(!is_ret_inst(&add_inst));
    }
}
