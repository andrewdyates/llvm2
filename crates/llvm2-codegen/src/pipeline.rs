// llvm2-codegen/pipeline.rs - End-to-end compilation pipeline
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Wires together all LLVM2 crates into a single entry point:
//   tMIR -> ISel -> Optimization -> RegAlloc -> Frame Lowering -> Encoding -> Mach-O

//! End-to-end compilation pipeline for LLVM2.
//!
//! Takes a tMIR function (represented as `llvm2_lower::Function`) and produces
//! a valid AArch64 Mach-O .o file.
//!
//! # Pipeline phases
//!
//! ```text
//! Phase 1: ISel (llvm2-lower)
//!   llvm2_lower::Function -> llvm2_lower::isel::MachFunction (VRegs, ISel opcodes)
//!
//! Phase 2: Adapt ISel -> IR
//!   llvm2_lower::isel::MachFunction -> llvm2_ir::MachFunction
//!
//! Phase 3: Optimization (llvm2-opt)
//!   llvm2_ir::MachFunction -> llvm2_ir::MachFunction (optimized)
//!
//! Phase 4: Adapt IR -> RegAlloc
//!   llvm2_ir::MachFunction -> llvm2_regalloc::MachFunction
//!
//! Phase 5: Register Allocation (llvm2-regalloc)
//!   llvm2_regalloc::MachFunction -> AllocationResult (VReg -> PReg map)
//!
//! Phase 6: Apply allocation to IR
//!   llvm2_ir::MachFunction + AllocationResult -> llvm2_ir::MachFunction (PRegs only)
//!
//! Phase 7: Frame Lowering (llvm2-codegen/frame)
//!   Insert prologue/epilogue, eliminate frame indices
//!
//! Phase 8: Encoding (llvm2-codegen/aarch64)
//!   llvm2_ir::MachFunction -> Vec<u8> (machine code bytes)
//!
//! Phase 9: Mach-O Emission (llvm2-codegen/macho)
//!   Vec<u8> -> Mach-O .o file bytes
//! ```
//!
//! # Type mismatches
//!
//! Each crate has its own MachFunction/MachInst types. The adapter functions
//! in this module convert between them. TODO: Unify these types in a future
//! phase so adapters are unnecessary.

use std::collections::HashMap;
use thiserror::Error;

use llvm2_ir::function::{MachFunction as IrMachFunction, Signature as IrSignature, StackSlot as IrStackSlot};
use llvm2_ir::inst::{AArch64Opcode as IrOpcode, MachInst as IrMachInst};
use llvm2_ir::operand::MachOperand as IrOperand;
use llvm2_ir::regs::{PReg, VReg};
use llvm2_ir::types::BlockId;

// ---------------------------------------------------------------------------
// Pipeline errors
// ---------------------------------------------------------------------------

/// Errors during compilation.
#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("instruction selection failed: {0}")]
    ISel(String),
    #[error("register allocation failed: {0}")]
    RegAlloc(String),
    #[error("encoding failed: {0}")]
    Encoding(String),
    #[error("unsupported opcode during encoding: {0:?}")]
    UnsupportedOpcode(IrOpcode),
}

// ---------------------------------------------------------------------------
// Pipeline configuration
// ---------------------------------------------------------------------------

/// Optimization level for the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    O0,
    O1,
    O2,
    O3,
}

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Optimization level.
    pub opt_level: OptLevel,
    /// Whether to emit debug info (placeholder for future).
    pub emit_debug: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            opt_level: OptLevel::O2,
            emit_debug: false,
        }
    }
}

// ---------------------------------------------------------------------------
// CompilationUnit — holds all state across compilation phases
// ---------------------------------------------------------------------------

/// Holds the state of a single function being compiled through the pipeline.
pub struct CompilationUnit {
    /// The function name.
    pub name: String,
    /// The IR-level function (shared model used by opt and frame lowering).
    pub ir_func: IrMachFunction,
    /// Encoded machine code bytes (populated after encoding phase).
    pub code: Vec<u8>,
    /// Pipeline configuration.
    pub config: PipelineConfig,
}

impl CompilationUnit {
    /// Create a new compilation unit from an IR function.
    pub fn new(ir_func: IrMachFunction, config: PipelineConfig) -> Self {
        let name = ir_func.name.clone();
        Self {
            name,
            ir_func,
            code: Vec::new(),
            config,
        }
    }
}

// ---------------------------------------------------------------------------
// Adapter: llvm2_lower::isel::MachFunction -> llvm2_ir::MachFunction
// ---------------------------------------------------------------------------

/// Map ISel opcodes to llvm2-ir opcodes.
///
/// The ISel has its own fine-grained opcode enum (e.g., ADDWrr, ADDXrr).
/// We map these to the coarser llvm2-ir opcodes which are what the encoder
/// and optimizer understand.
///
/// TODO: Unify opcode enums across crates so this mapping is unnecessary.
fn map_isel_opcode(isel_op: &llvm2_lower::isel::AArch64Opcode) -> IrOpcode {
    use llvm2_lower::isel::AArch64Opcode as IsOp;
    match isel_op {
        IsOp::ADDWrr | IsOp::ADDXrr => IrOpcode::AddRR,
        IsOp::ADDWri | IsOp::ADDXri => IrOpcode::AddRI,
        IsOp::SUBWrr | IsOp::SUBXrr => IrOpcode::SubRR,
        IsOp::SUBWri | IsOp::SUBXri => IrOpcode::SubRI,
        IsOp::MULWrrr | IsOp::MULXrrr => IrOpcode::MulRR,
        IsOp::SDIVWrr | IsOp::SDIVXrr => IrOpcode::SDiv,
        IsOp::UDIVWrr | IsOp::UDIVXrr => IrOpcode::UDiv,
        IsOp::CMPWrr | IsOp::CMPXrr => IrOpcode::CmpRR,
        IsOp::CMPWri | IsOp::CMPXri => IrOpcode::CmpRI,
        IsOp::CSETWcc => IrOpcode::MovI, // CSET is a special MOV-like
        IsOp::MOVWrr | IsOp::MOVXrr => IrOpcode::MovR,
        IsOp::MOVZWi | IsOp::MOVZXi => IrOpcode::Movz,
        IsOp::MOVNWi | IsOp::MOVNXi => IrOpcode::MovI, // MOVN is a variant
        IsOp::MOVKXi => IrOpcode::Movk,
        IsOp::FMOVSri | IsOp::FMOVDri => IrOpcode::MovI,
        IsOp::LDRWui | IsOp::LDRXui | IsOp::LDRSui | IsOp::LDRDui => IrOpcode::LdrRI,
        IsOp::STRWui | IsOp::STRXui | IsOp::STRSui | IsOp::STRDui => IrOpcode::StrRI,
        IsOp::B => IrOpcode::B,
        IsOp::Bcc => IrOpcode::BCond,
        IsOp::BL => IrOpcode::Bl,
        IsOp::BLR => IrOpcode::Blr,
        IsOp::RET => IrOpcode::Ret,
        IsOp::ANDWrr | IsOp::ANDXrr | IsOp::ANDWri | IsOp::ANDXri => IrOpcode::AndRR,
        IsOp::ORRWrr | IsOp::ORRXrr | IsOp::ORRWri | IsOp::ORRXri => IrOpcode::OrrRR,
        IsOp::EORWrr | IsOp::EORXrr | IsOp::EORWri | IsOp::EORXri => IrOpcode::EorRR,
        IsOp::LSLVWr | IsOp::LSLVXr | IsOp::LSLWi | IsOp::LSLXi => IrOpcode::LslRR,
        IsOp::LSRVWr | IsOp::LSRVXr | IsOp::LSRWi | IsOp::LSRXi => IrOpcode::LsrRR,
        IsOp::ASRVWr | IsOp::ASRVXr | IsOp::ASRWi | IsOp::ASRXi => IrOpcode::AsrRR,
        IsOp::ADRP => IrOpcode::Adrp,
        IsOp::ADDXriPCRel | IsOp::ADDXriSP => IrOpcode::AddPCRel,
        IsOp::FADDSrr | IsOp::FADDDrr => IrOpcode::FaddRR,
        IsOp::FSUBSrr | IsOp::FSUBDrr => IrOpcode::FsubRR,
        IsOp::FMULSrr | IsOp::FMULDrr => IrOpcode::FmulRR,
        IsOp::FDIVSrr | IsOp::FDIVDrr => IrOpcode::FdivRR,
        IsOp::FCMPSrr | IsOp::FCMPDrr => IrOpcode::Fcmp,
        IsOp::FCVTZSWr | IsOp::FCVTZSXr => IrOpcode::FcvtzsRR,
        IsOp::SCVTFSWr | IsOp::SCVTFDWr | IsOp::SCVTFSXr | IsOp::SCVTFDXr => IrOpcode::ScvtfRR,
        IsOp::SXTBWr | IsOp::SXTBXr | IsOp::SXTHWr | IsOp::SXTHXr | IsOp::SXTWXr => IrOpcode::Sxtw,
        IsOp::UXTBWr | IsOp::UXTHWr => IrOpcode::Uxtw,
        IsOp::COPY => IrOpcode::MovR,
        IsOp::PHI => IrOpcode::Phi,
        // Bitfield ops, conditional selects, BIC/ORN, etc. map to Nop as placeholder
        // TODO: Add proper IR opcodes for these
        _ => IrOpcode::Nop,
    }
}

/// Convert an ISel MachOperand to an IR MachOperand.
fn convert_isel_operand(op: &llvm2_lower::isel::MachOperand) -> IrOperand {
    use llvm2_lower::isel::MachOperand as IsOp;
    match op {
        IsOp::VReg(v) => IrOperand::VReg(*v),
        IsOp::PReg(p) => IrOperand::PReg(*p),
        IsOp::Imm(v) => IrOperand::Imm(*v),
        IsOp::FImm(v) => IrOperand::FImm(*v),
        IsOp::Block(b) => IrOperand::Block(BlockId(b.0)),
        IsOp::CondCode(_cc) => {
            // Condition codes are encoded as immediates in the IR model.
            // TODO: Add a proper CondCode operand to IR.
            IrOperand::Imm(0)
        }
        IsOp::Symbol(_name) => {
            // Symbol references become immediates (relocated later).
            // TODO: Add a proper Symbol operand to IR.
            IrOperand::Imm(0)
        }
        IsOp::StackSlot(idx) => {
            IrOperand::StackSlot(llvm2_ir::types::StackSlotId(*idx))
        }
    }
}

/// Convert ISel's MachFunction to the shared IR MachFunction.
///
/// This is the Phase 1->2 adapter. The ISel uses HashMap-based blocks and
/// its own opcode enum. The IR uses Vec-based arena storage and a unified
/// opcode enum.
pub fn isel_to_ir(
    isel_func: &llvm2_lower::isel::MachFunction,
    param_types: &[llvm2_ir::function::Type],
    return_types: &[llvm2_ir::function::Type],
) -> IrMachFunction {
    let sig = IrSignature::new(param_types.to_vec(), return_types.to_vec());
    let mut ir_func = IrMachFunction::new(isel_func.name.clone(), sig);
    ir_func.next_vreg = isel_func.next_vreg;

    // Clear the default entry block that MachFunction::new creates.
    ir_func.blocks.clear();
    ir_func.block_order.clear();

    // Create all blocks first.
    for &block_ref in &isel_func.block_order {
        let block_id = BlockId(block_ref.0);
        while ir_func.blocks.len() <= block_id.0 as usize {
            ir_func.blocks.push(llvm2_ir::function::MachBlock::new());
        }
        ir_func.block_order.push(block_id);
    }
    ir_func.entry = if isel_func.block_order.is_empty() {
        BlockId(0)
    } else {
        BlockId(isel_func.block_order[0].0)
    };

    // Convert instructions block by block.
    for &block_ref in &isel_func.block_order {
        let block_id = BlockId(block_ref.0);
        if let Some(isel_block) = isel_func.blocks.get(&block_ref) {
            for isel_inst in &isel_block.insts {
                let ir_opcode = map_isel_opcode(&isel_inst.opcode);
                let ir_operands: Vec<IrOperand> = isel_inst
                    .operands
                    .iter()
                    .map(convert_isel_operand)
                    .collect();
                let ir_inst = IrMachInst::new(ir_opcode, ir_operands);
                let inst_id = ir_func.push_inst(ir_inst);
                ir_func.append_inst(block_id, inst_id);
            }
        }
    }

    ir_func
}

// ---------------------------------------------------------------------------
// Adapter: llvm2_ir::MachFunction -> llvm2_regalloc::MachFunction
// ---------------------------------------------------------------------------

/// Convert an IR MachFunction to the regalloc MachFunction format.
///
/// The regalloc MachFunction separates defs from uses for liveness analysis.
/// We use a simple heuristic: first operand is def, rest are uses
/// (for most instructions). Special cases: branches have no defs, etc.
pub fn ir_to_regalloc(ir_func: &IrMachFunction) -> llvm2_regalloc::MachFunction {
    use llvm2_regalloc::machine_types as ra;

    let mut ra_func = ra::MachFunction {
        name: ir_func.name.clone(),
        insts: Vec::new(),
        blocks: Vec::new(),
        block_order: ir_func.block_order.clone(),
        entry_block: ir_func.entry,
        next_vreg: ir_func.next_vreg,
        next_stack_slot: ir_func.stack_slots.len() as u32,
        stack_slots: HashMap::new(),
    };

    // Convert stack slots.
    for (i, slot) in ir_func.stack_slots.iter().enumerate() {
        ra_func.stack_slots.insert(
            llvm2_ir::types::StackSlotId(i as u32),
            ra::StackSlot {
                size: slot.size,
                align: slot.align,
            },
        );
    }

    // Convert instructions.
    for ir_inst in &ir_func.insts {
        let flags = convert_ir_flags_to_regalloc(ir_inst);

        let (defs, uses) = classify_def_use(ir_inst);
        let implicit_defs: Vec<PReg> = ir_inst.implicit_defs.to_vec();
        let implicit_uses: Vec<PReg> = ir_inst.implicit_uses.to_vec();

        ra_func.insts.push(ra::MachInst {
            opcode: ir_inst.opcode as u16,
            defs,
            uses,
            implicit_defs,
            implicit_uses,
            flags,
        });
    }

    // Convert blocks.
    for ir_block in &ir_func.blocks {
        ra_func.blocks.push(ra::MachBlock {
            insts: ir_block.insts.clone(),
            preds: ir_block.preds.clone(),
            succs: ir_block.succs.clone(),
            loop_depth: 0,
        });
    }

    ra_func
}

/// Classify operands into defs and uses for regalloc.
///
/// Convention:
/// - For most ALU instructions: operand[0] is def, rest are uses
/// - For stores: all operands are uses (no register def)
/// - For branches/returns: all operands are uses
/// - For loads: operand[0] is def, rest are uses
fn classify_def_use(
    inst: &IrMachInst,
) -> (Vec<llvm2_regalloc::MachOperand>, Vec<llvm2_regalloc::MachOperand>) {
    use llvm2_regalloc::machine_types::MachOperand as RaOp;

    let convert_op = |op: &IrOperand| -> RaOp {
        match op {
            IrOperand::VReg(v) => RaOp::VReg(*v),
            IrOperand::PReg(p) => RaOp::PReg(*p),
            IrOperand::Imm(i) => RaOp::Imm(*i),
            IrOperand::FImm(f) => RaOp::FImm(*f),
            IrOperand::Block(b) => RaOp::Block(*b),
            IrOperand::StackSlot(s) => RaOp::StackSlot(*s),
            // FrameIndex, MemOp, Special are not used pre-regalloc in this adapter
            _ => RaOp::Imm(0),
        }
    };

    let is_store = inst.flags.contains(llvm2_ir::inst::InstFlags::WRITES_MEMORY);
    let is_branch = inst.flags.contains(llvm2_ir::inst::InstFlags::IS_BRANCH);
    let is_return = inst.flags.contains(llvm2_ir::inst::InstFlags::IS_RETURN);
    let is_cmp = matches!(inst.opcode, IrOpcode::CmpRR | IrOpcode::CmpRI | IrOpcode::Fcmp);

    if is_store || is_branch || is_return || is_cmp || inst.operands.is_empty() {
        // All uses, no defs.
        let uses: Vec<RaOp> = inst.operands.iter().map(convert_op).collect();
        (Vec::new(), uses)
    } else {
        // First operand is def, rest are uses.
        let defs = vec![convert_op(&inst.operands[0])];
        let uses: Vec<RaOp> = inst.operands[1..].iter().map(convert_op).collect();
        (defs, uses)
    }
}

/// Convert IR InstFlags to regalloc InstFlags.
fn convert_ir_flags_to_regalloc(inst: &IrMachInst) -> llvm2_regalloc::InstFlags {
    use llvm2_regalloc::machine_types::InstFlags as RaFlags;

    let ir_flags = inst.flags;
    let mut ra_bits: u16 = 0;

    if ir_flags.contains(llvm2_ir::inst::InstFlags::IS_CALL) {
        ra_bits |= RaFlags::IS_CALL;
    }
    if ir_flags.contains(llvm2_ir::inst::InstFlags::IS_BRANCH) {
        ra_bits |= RaFlags::IS_BRANCH;
    }
    if ir_flags.contains(llvm2_ir::inst::InstFlags::IS_RETURN) {
        ra_bits |= RaFlags::IS_RETURN;
    }
    if ir_flags.contains(llvm2_ir::inst::InstFlags::IS_TERMINATOR) {
        ra_bits |= RaFlags::IS_TERMINATOR;
    }
    if ir_flags.contains(llvm2_ir::inst::InstFlags::HAS_SIDE_EFFECTS) {
        ra_bits |= RaFlags::HAS_SIDE_EFFECTS;
    }
    if ir_flags.contains(llvm2_ir::inst::InstFlags::IS_PSEUDO) {
        ra_bits |= RaFlags::IS_PSEUDO;
    }
    if ir_flags.contains(llvm2_ir::inst::InstFlags::READS_MEMORY) {
        ra_bits |= RaFlags::READS_MEMORY;
    }
    if ir_flags.contains(llvm2_ir::inst::InstFlags::WRITES_MEMORY) {
        ra_bits |= RaFlags::WRITES_MEMORY;
    }
    if ir_flags.contains(llvm2_ir::inst::InstFlags::IS_PHI) {
        ra_bits |= RaFlags::IS_PHI;
    }

    RaFlags(ra_bits)
}

// ---------------------------------------------------------------------------
// Apply register allocation results back to IR
// ---------------------------------------------------------------------------

/// Rewrite all VReg operands in the IR function with their allocated PRegs.
pub fn apply_regalloc(
    ir_func: &mut IrMachFunction,
    allocation: &HashMap<VReg, PReg>,
) {
    for inst in &mut ir_func.insts {
        for operand in &mut inst.operands {
            if let IrOperand::VReg(vreg) = operand {
                if let Some(&preg) = allocation.get(vreg) {
                    *operand = IrOperand::PReg(preg);
                }
                // If not found in allocation, the vreg was spilled.
                // Spill code insertion should have already handled this.
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Encoding: IR MachFunction -> machine code bytes
// ---------------------------------------------------------------------------

/// Encode a single IR instruction into AArch64 machine code.
///
/// Returns the 4-byte little-endian encoding, or an error for pseudo/unsupported
/// instructions.
fn encode_ir_inst(inst: &IrMachInst) -> Result<u32, PipelineError> {
    use crate::aarch64::encoding;

    // Helper to extract physical register number from operand.
    let preg_hw = |idx: usize| -> u32 {
        match &inst.operands[idx] {
            IrOperand::PReg(p) => p.hw_enc() as u32,
            IrOperand::Special(s) => {
                match s {
                    llvm2_ir::regs::SpecialReg::SP => 31,
                    llvm2_ir::regs::SpecialReg::XZR => 31,
                    llvm2_ir::regs::SpecialReg::WZR => 31,
                }
            }
            _ => 31, // Default to XZR/SP for safety
        }
    };

    let imm_val = |idx: usize| -> i64 {
        match &inst.operands[idx] {
            IrOperand::Imm(v) => *v,
            _ => 0,
        }
    };

    // Determine if the instruction is 64-bit based on register class.
    let is_64bit = |idx: usize| -> bool {
        match &inst.operands.get(idx) {
            Some(IrOperand::PReg(p)) => p.is_gpr() && p.0 < 32, // All GPRs default to 64-bit
            _ => true,
        }
    };

    match inst.opcode {
        // --- Arithmetic ---
        IrOpcode::AddRR => {
            // ADD Rd, Rn, Rm
            let sf = if is_64bit(0) { 1 } else { 0 };
            Ok(encoding::encode_add_sub_shifted_reg(
                sf, 0, 0, 0, preg_hw(2), 0, preg_hw(1), preg_hw(0),
            ))
        }
        IrOpcode::AddRI => {
            // ADD Rd, Rn, #imm
            let sf = if is_64bit(0) { 1 } else { 0 };
            let imm = imm_val(2) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf, 0, 0, 0, imm, preg_hw(1), preg_hw(0),
            ))
        }
        IrOpcode::SubRR => {
            let sf = if is_64bit(0) { 1 } else { 0 };
            Ok(encoding::encode_add_sub_shifted_reg(
                sf, 1, 0, 0, preg_hw(2), 0, preg_hw(1), preg_hw(0),
            ))
        }
        IrOpcode::SubRI => {
            let sf = if is_64bit(0) { 1 } else { 0 };
            let imm = imm_val(2) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf, 1, 0, 0, imm, preg_hw(1), preg_hw(0),
            ))
        }

        // --- Compare ---
        IrOpcode::CmpRR => {
            // CMP = SUBS XZR, Rn, Rm
            let sf = 1u32; // default 64-bit
            Ok(encoding::encode_add_sub_shifted_reg(
                sf, 1, 1, 0, preg_hw(1), 0, preg_hw(0), 31,
            ))
        }
        IrOpcode::CmpRI => {
            let sf = 1u32;
            let imm = imm_val(1) as u32 & 0xFFF;
            Ok(encoding::encode_add_sub_imm(
                sf, 1, 1, 0, imm, preg_hw(0), 31,
            ))
        }

        // --- Logical ---
        IrOpcode::AndRR => {
            let sf = if is_64bit(0) { 1 } else { 0 };
            Ok(encoding::encode_logical_shifted_reg(
                sf, 0b00, 0, 0, preg_hw(2), 0, preg_hw(1), preg_hw(0),
            ))
        }
        IrOpcode::OrrRR => {
            let sf = if is_64bit(0) { 1 } else { 0 };
            Ok(encoding::encode_logical_shifted_reg(
                sf, 0b01, 0, 0, preg_hw(2), 0, preg_hw(1), preg_hw(0),
            ))
        }
        IrOpcode::EorRR => {
            let sf = if is_64bit(0) { 1 } else { 0 };
            Ok(encoding::encode_logical_shifted_reg(
                sf, 0b10, 0, 0, preg_hw(2), 0, preg_hw(1), preg_hw(0),
            ))
        }

        // --- Move ---
        IrOpcode::MovR => {
            // MOV Rd, Rm = ORR Rd, XZR, Rm
            let sf = if is_64bit(0) { 1 } else { 0 };
            Ok(encoding::encode_logical_shifted_reg(
                sf, 0b01, 0, 0, preg_hw(1), 0, 31, preg_hw(0),
            ))
        }
        IrOpcode::MovI | IrOpcode::Movz => {
            // MOVZ Rd, #imm16
            let sf = if is_64bit(0) { 1 } else { 0 };
            let imm16 = imm_val(1) as u32 & 0xFFFF;
            Ok(encoding::encode_move_wide(sf, 0b10, 0, imm16, preg_hw(0)))
        }
        IrOpcode::Movk => {
            // MOVK Rd, #imm16
            let sf = if is_64bit(0) { 1 } else { 0 };
            let imm16 = imm_val(1) as u32 & 0xFFFF;
            Ok(encoding::encode_move_wide(sf, 0b11, 0, imm16, preg_hw(0)))
        }

        // --- Load/Store ---
        IrOpcode::LdrRI => {
            // LDR Rt, [Rn, #offset]
            // Size=11 (64-bit), V=0, opc=01 (LDR)
            let offset = if inst.operands.len() > 2 { imm_val(2) } else { 0 };
            let scaled = (offset / 8) as u32 & 0xFFF;
            Ok(encoding::encode_load_store_ui(0b11, 0, 0b01, scaled, preg_hw(1), preg_hw(0)))
        }
        IrOpcode::StrRI => {
            // STR Rt, [Rn, #offset]
            let offset = if inst.operands.len() > 2 { imm_val(2) } else { 0 };
            let scaled = (offset / 8) as u32 & 0xFFF;
            Ok(encoding::encode_load_store_ui(0b11, 0, 0b00, scaled, preg_hw(1), preg_hw(0)))
        }
        IrOpcode::StpRI => {
            // STP Rt, Rt2, [Rn, #offset]
            let offset = if inst.operands.len() > 3 { imm_val(3) } else { 0 };
            let scaled_imm7 = ((offset / 8) as i32 as u32) & 0x7F;
            Ok(encoding::encode_load_store_pair(
                0b10, 0, 0, scaled_imm7, preg_hw(1), preg_hw(2), preg_hw(0),
            ))
        }
        IrOpcode::LdpRI => {
            // LDP Rt, Rt2, [Rn, #offset]
            let offset = if inst.operands.len() > 3 { imm_val(3) } else { 0 };
            let scaled_imm7 = ((offset / 8) as i32 as u32) & 0x7F;
            Ok(encoding::encode_load_store_pair(
                0b10, 0, 1, scaled_imm7, preg_hw(1), preg_hw(2), preg_hw(0),
            ))
        }

        // --- Branch ---
        IrOpcode::B => {
            // Unconditional branch: B <offset>
            // offset in operand is instruction count
            let offset = imm_val(0) as u32 & 0x3FFFFFF;
            Ok(encoding::encode_uncond_branch(0, offset))
        }
        IrOpcode::BCond => {
            // Conditional branch: B.cond <offset>
            let cond = imm_val(0) as u32 & 0xF;
            let offset = if inst.operands.len() > 1 { imm_val(1) as u32 & 0x7FFFF } else { 0 };
            Ok(encoding::encode_cond_branch(offset, cond))
        }
        IrOpcode::Bl => {
            // BL <offset>
            let offset = imm_val(0) as u32 & 0x3FFFFFF;
            Ok(encoding::encode_uncond_branch(1, offset))
        }
        IrOpcode::Blr => {
            // BLR Rn
            Ok(encoding::encode_branch_reg(0b0001, preg_hw(0)))
        }
        IrOpcode::Ret => {
            // RET (X30)
            Ok(encoding::encode_branch_reg(0b0010, 30))
        }

        // --- Pseudo-instructions: skip encoding ---
        IrOpcode::Phi | IrOpcode::StackAlloc | IrOpcode::Nop => {
            // Pseudos should have been eliminated before encoding.
            // Emit NOP (0xD503201F) as a safe fallback.
            Ok(0xD503201F)
        }

        // For unimplemented opcodes, emit NOP.
        // TODO: Implement encoding for all remaining opcodes.
        _ => Ok(0xD503201F),
    }
}

/// Encode all instructions in a function to machine code bytes.
///
/// Walks blocks in layout order, encoding each non-pseudo instruction.
pub fn encode_function(func: &IrMachFunction) -> Result<Vec<u8>, PipelineError> {
    let mut code = Vec::new();

    for &block_id in &func.block_order {
        let block = func.block(block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            // Skip pseudo-instructions that should have been eliminated.
            if inst.is_pseudo() {
                continue;
            }
            let word = encode_ir_inst(inst)?;
            code.extend_from_slice(&word.to_le_bytes());
        }
    }

    Ok(code)
}

// ---------------------------------------------------------------------------
// Pipeline — main entry point
// ---------------------------------------------------------------------------

/// The LLVM2 compilation pipeline.
///
/// Orchestrates all phases from tMIR input through to Mach-O .o emission.
pub struct Pipeline {
    pub config: PipelineConfig,
}

impl Pipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// Create a pipeline with default O2 configuration.
    pub fn default_o2() -> Self {
        Self::new(PipelineConfig::default())
    }

    /// Compile a tMIR function through the full pipeline, producing Mach-O bytes.
    ///
    /// This is the main entry point. It takes a `llvm2_lower::Function` (tMIR)
    /// and returns the bytes of a valid .o file.
    pub fn compile_function(
        &self,
        input: &llvm2_lower::Function,
    ) -> Result<Vec<u8>, PipelineError> {
        // Phase 1: Instruction Selection
        let isel_func = self.run_isel(input)?;

        // Phase 2: Convert ISel output to shared IR
        let param_types = self.convert_lower_types_to_ir(&input.signature.params);
        let return_types = self.convert_lower_types_to_ir(&input.signature.returns);
        let mut ir_func = isel_to_ir(&isel_func, &param_types, &return_types);

        // Phase 3: Optimization
        self.run_optimization(&mut ir_func);

        // Phase 4-6: Register Allocation
        self.run_regalloc(&mut ir_func)?;

        // Phase 7: Frame Lowering
        self.run_frame_lowering(&mut ir_func);

        // Phase 8: Encoding
        let code = encode_function(&ir_func)?;

        // Phase 9: Mach-O Emission
        let obj_bytes = self.emit_macho(&ir_func.name, &code);

        Ok(obj_bytes)
    }

    /// Compile a pre-built IR function (skipping ISel).
    ///
    /// Useful when you already have an `IrMachFunction` and want to run
    /// the remaining phases (opt, regalloc, frame, encode, emit).
    pub fn compile_ir_function(
        &self,
        ir_func: &mut IrMachFunction,
    ) -> Result<Vec<u8>, PipelineError> {
        // Phase 3: Optimization
        self.run_optimization(ir_func);

        // Phase 4-6: Register Allocation
        self.run_regalloc(ir_func)?;

        // Phase 7: Frame Lowering
        self.run_frame_lowering(ir_func);

        // Phase 8: Encoding
        let code = encode_function(ir_func)?;

        // Phase 9: Mach-O Emission
        let obj_bytes = self.emit_macho(&ir_func.name, &code);

        Ok(obj_bytes)
    }

    /// Encode an already-allocated IR function to Mach-O.
    ///
    /// Skips ISel, optimization, and regalloc. Useful for testing the
    /// encoding and emission phases in isolation.
    pub fn encode_and_emit(
        &self,
        ir_func: &mut IrMachFunction,
    ) -> Result<Vec<u8>, PipelineError> {
        // Phase 7: Frame Lowering
        self.run_frame_lowering(ir_func);

        // Phase 8: Encoding
        let code = encode_function(ir_func)?;

        // Phase 9: Mach-O Emission
        let obj_bytes = self.emit_macho(&ir_func.name, &code);

        Ok(obj_bytes)
    }

    // --- Phase implementations ---

    /// Phase 1: Run instruction selection on the input function.
    fn run_isel(
        &self,
        input: &llvm2_lower::Function,
    ) -> Result<llvm2_lower::isel::MachFunction, PipelineError> {
        use llvm2_lower::isel::InstructionSelector;

        let sig = llvm2_lower::function::Signature {
            params: input.signature.params.clone(),
            returns: input.signature.returns.clone(),
        };

        let mut isel = InstructionSelector::new(input.name.clone(), sig);

        // Process blocks in order.
        for (&block_ref, basic_block) in &input.blocks {
            isel.select_block(block_ref, &basic_block.instructions);
        }

        Ok(isel.finalize())
    }

    /// Phase 3: Run optimization passes.
    fn run_optimization(&self, func: &mut IrMachFunction) {
        use llvm2_opt::pipeline::{OptLevel as OptOptLevel, OptimizationPipeline};

        let opt_level = match self.config.opt_level {
            OptLevel::O0 => OptOptLevel::O0,
            OptLevel::O1 => OptOptLevel::O1,
            OptLevel::O2 => OptOptLevel::O2,
            OptLevel::O3 => OptOptLevel::O3,
        };

        let pipeline = OptimizationPipeline::new(opt_level);
        let _stats = pipeline.run(func);
    }

    /// Phase 4-6: Run register allocation and apply results.
    fn run_regalloc(
        &self,
        ir_func: &mut IrMachFunction,
    ) -> Result<(), PipelineError> {
        // Phase 4: Convert IR to regalloc format
        let mut ra_func = ir_to_regalloc(ir_func);

        // Phase 5: Run the allocator
        let ra_config = llvm2_regalloc::AllocConfig::default_aarch64();
        let result = llvm2_regalloc::allocate(&mut ra_func, &ra_config)
            .map_err(|e| PipelineError::RegAlloc(e.to_string()))?;

        // Phase 6: Apply allocation back to IR
        apply_regalloc(ir_func, &result.allocation);

        // Add spill stack slots to the IR function.
        for _spill in &result.spills {
            // Each spill gets an 8-byte slot (register-sized).
            ir_func.alloc_stack_slot(IrStackSlot::new(8, 8));
        }

        Ok(())
    }

    /// Phase 7: Frame lowering — prologue/epilogue + frame index elimination.
    fn run_frame_lowering(&self, func: &mut IrMachFunction) {
        use crate::frame;

        let layout = frame::compute_frame_layout(func, 0, true);
        frame::eliminate_frame_indices(func, &layout);
        frame::insert_prologue_epilogue(func, &layout);
    }

    /// Phase 9: Emit a Mach-O .o file from encoded machine code.
    fn emit_macho(&self, func_name: &str, code: &[u8]) -> Vec<u8> {
        use crate::macho::MachOWriter;

        let mut writer = MachOWriter::new();
        writer.add_text_section(code);

        // Add the function symbol with Mach-O name mangling (_prefix).
        let symbol_name = format!("_{}", func_name);
        writer.add_symbol(&symbol_name, 1, 0, true);

        writer.write()
    }

    /// Convert llvm2-lower types to llvm2-ir types.
    fn convert_lower_types_to_ir(
        &self,
        types: &[llvm2_lower::types::Type],
    ) -> Vec<llvm2_ir::function::Type> {
        types
            .iter()
            .map(|t| match t {
                llvm2_lower::types::Type::I8 => llvm2_ir::function::Type::I8,
                llvm2_lower::types::Type::I16 => llvm2_ir::function::Type::I16,
                llvm2_lower::types::Type::I32 => llvm2_ir::function::Type::I32,
                llvm2_lower::types::Type::I64 => llvm2_ir::function::Type::I64,
                llvm2_lower::types::Type::I128 => llvm2_ir::function::Type::I128,
                llvm2_lower::types::Type::F32 => llvm2_ir::function::Type::F32,
                llvm2_lower::types::Type::F64 => llvm2_ir::function::Type::F64,
                llvm2_lower::types::Type::B1 => llvm2_ir::function::Type::B1,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Convenience: compile_to_object — one-shot compilation
// ---------------------------------------------------------------------------

/// Compile a tMIR function to a Mach-O .o file in one call.
///
/// This is the highest-level API for LLVM2 compilation.
pub fn compile_to_object(
    input: &llvm2_lower::Function,
    opt_level: OptLevel,
) -> Result<Vec<u8>, PipelineError> {
    let config = PipelineConfig {
        opt_level,
        emit_debug: false,
    };
    let pipeline = Pipeline::new(config);
    pipeline.compile_function(input)
}

/// Build an `add(a: i32, b: i32) -> i32` function directly in IR.
///
/// This is a convenience for testing the pipeline without needing the full
/// tMIR input infrastructure. It builds the function directly in llvm2-ir
/// types with physical registers (simulating post-ISel + post-regalloc state).
pub fn build_add_test_function() -> IrMachFunction {
    use llvm2_ir::function::Type;
    use llvm2_ir::regs::{X0, X1};

    let sig = IrSignature::new(
        vec![Type::I32, Type::I32],
        vec![Type::I32],
    );
    let mut func = IrMachFunction::new("add".to_string(), sig);
    let entry = func.entry;

    // ADD X0, X0, X1 (result in X0 per ABI)
    let add = IrMachInst::new(
        IrOpcode::AddRR,
        vec![
            IrOperand::PReg(X0),
            IrOperand::PReg(X0),
            IrOperand::PReg(X1),
        ],
    );
    let add_id = func.push_inst(add);
    func.append_inst(entry, add_id);

    // RET
    let ret = IrMachInst::new(IrOpcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}
