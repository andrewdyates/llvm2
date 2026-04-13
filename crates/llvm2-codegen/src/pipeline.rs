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
//!   llvm2_lower::Function -> llvm2_lower::isel::MachFunction
//!   (VRegs, canonical llvm2_ir::AArch64Opcode — unified in issue #73)
//!
//! Phase 2: Adapt ISel -> IR (structural only — opcodes are already unified)
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
//! # Type adapters and unification status (issue #73)
//!
//! **Unified (no adapter needed):**
//! - `AArch64Opcode`: ISel uses `llvm2_ir::AArch64Opcode` directly.
//! - `InstFlags`: regalloc re-exports `llvm2_ir::InstFlags` directly.
//!   The `convert_ir_flags_to_regalloc()` function has been eliminated.
//! - Primitive types: `VReg`, `PReg`, `RegClass`, `BlockId`, `InstId`,
//!   `StackSlotId` — all re-exported from `llvm2_ir` by regalloc.
//!
//! **Remaining structural adapters (separate types with good reason):**
//! - `MachOperand`: ISel has `CondCode`/`Symbol`/`StackSlot(u32)` variants;
//!   regalloc omits `MemOp`/`FrameIndex`/`Special`. Unification requires
//!   a superset enum with phase-dependent validation.
//! - `MachInst`: regalloc separates defs/uses for liveness analysis;
//!   ISel has no flags/implicit-defs/proofs. Unification requires adding
//!   def/use classification to `llvm2_ir::MachInst`.
//! - `MachBlock`: regalloc adds `loop_depth`; ISel uses inline `Vec<MachInst>`.
//! - `MachFunction`: ISel uses `HashMap<Block, MachBlock>` (construction-friendly);
//!   regalloc uses `HashMap<StackSlotId, StackSlot>` + `next_stack_slot`.
//!
//! See issue #73 for the remaining unification plan.

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
    #[error("branch relaxation failed: {0}")]
    Relaxation(#[from] crate::relax::RelaxError),
    #[error("invalid operand in regalloc adapter: {0}")]
    InvalidOperand(String),
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

// NOTE: `map_isel_opcode()` has been ELIMINATED as of issue #73.
//
// The ISel now uses `llvm2_ir::AArch64Opcode` directly (re-exported via
// `llvm2_lower::isel::AArch64Opcode`). There is no longer a separate ISel
// opcode enum, so no mapping is needed. The `isel_to_ir` adapter below
// uses `isel_inst.opcode` directly as an `IrOpcode`.

/// Convert an ISel MachOperand to an IR MachOperand.
fn convert_isel_operand(op: &llvm2_lower::isel::MachOperand) -> IrOperand {
    use llvm2_lower::isel::MachOperand as IsOp;
    match op {
        IsOp::VReg(v) => IrOperand::VReg(*v),
        IsOp::PReg(p) => IrOperand::PReg(*p),
        IsOp::Imm(v) => IrOperand::Imm(*v),
        IsOp::FImm(v) => IrOperand::FImm(*v),
        IsOp::Block(b) => IrOperand::Block(BlockId(b.0)),
        IsOp::CondCode(cc) => {
            // Condition codes are encoded as 4-bit immediates per ARM ARM.
            use llvm2_lower::isel::AArch64CC;
            let encoding = match cc {
                AArch64CC::EQ => 0,
                AArch64CC::NE => 1,
                AArch64CC::HS => 2,
                AArch64CC::LO => 3,
                AArch64CC::MI => 4,
                AArch64CC::PL => 5,
                AArch64CC::VS => 6,
                AArch64CC::VC => 7,
                AArch64CC::HI => 8,
                AArch64CC::LS => 9,
                AArch64CC::GE => 10,
                AArch64CC::LT => 11,
                AArch64CC::GT => 12,
                AArch64CC::LE => 13,
            };
            IrOperand::Imm(encoding)
        }
        IsOp::Symbol(name) => {
            // Symbol references are preserved through the pipeline so the
            // relocation collector can emit proper linker entries.
            IrOperand::Symbol(name.clone())
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

    // Convert instructions block by block and copy successor edges.
    for &block_ref in &isel_func.block_order {
        let block_id = BlockId(block_ref.0);
        if let Some(isel_block) = isel_func.blocks.get(&block_ref) {
            for isel_inst in &isel_block.insts {
                // ISel now uses canonical llvm2_ir::AArch64Opcode directly
                // (issue #73 unification). No opcode mapping needed.
                let ir_operands: Vec<IrOperand> = isel_inst
                    .operands
                    .iter()
                    .map(convert_isel_operand)
                    .collect();
                let ir_inst = IrMachInst::new(isel_inst.opcode, ir_operands);
                let inst_id = ir_func.push_inst(ir_inst);
                ir_func.append_inst(block_id, inst_id);
            }

            // Copy successor edges from ISel blocks to IR blocks.
            for &succ in &isel_block.successors {
                let succ_id = BlockId(succ.0);
                ir_func.blocks[block_id.0 as usize].succs.push(succ_id);
            }
        }
    }

    // Compute predecessor edges from successors.
    let num_blocks = ir_func.blocks.len();
    let mut preds_map: Vec<Vec<BlockId>> = vec![Vec::new(); num_blocks];
    for &block_ref in &isel_func.block_order {
        let block_id = BlockId(block_ref.0);
        for &succ_id in &ir_func.blocks[block_id.0 as usize].succs {
            if (succ_id.0 as usize) < num_blocks {
                preds_map[succ_id.0 as usize].push(block_id);
            }
        }
    }
    for (i, preds) in preds_map.into_iter().enumerate() {
        ir_func.blocks[i].preds = preds;
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
pub fn ir_to_regalloc(ir_func: &IrMachFunction) -> Result<llvm2_regalloc::MachFunction, PipelineError> {
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
    // InstFlags are unified (issue #73) — same type in llvm2_ir and llvm2_regalloc,
    // so flags pass through directly without conversion.
    for ir_inst in &ir_func.insts {
        let (defs, uses) = classify_def_use(ir_inst)?;
        let implicit_defs: Vec<PReg> = ir_inst.implicit_defs.to_vec();
        let implicit_uses: Vec<PReg> = ir_inst.implicit_uses.to_vec();

        ra_func.insts.push(ra::MachInst {
            opcode: ir_inst.opcode as u16,
            defs,
            uses,
            implicit_defs,
            implicit_uses,
            flags: ir_inst.flags,
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

    Ok(ra_func)
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
) -> Result<(Vec<llvm2_regalloc::MachOperand>, Vec<llvm2_regalloc::MachOperand>), PipelineError> {
    use llvm2_regalloc::machine_types::MachOperand as RaOp;

    let convert_op = |op: &IrOperand| -> Result<RaOp, PipelineError> {
        match op {
            IrOperand::VReg(v) => Ok(RaOp::VReg(*v)),
            IrOperand::PReg(p) => Ok(RaOp::PReg(*p)),
            IrOperand::Imm(i) => Ok(RaOp::Imm(*i)),
            IrOperand::FImm(f) => Ok(RaOp::FImm(*f)),
            IrOperand::Block(b) => Ok(RaOp::Block(*b)),
            IrOperand::StackSlot(s) => Ok(RaOp::StackSlot(*s)),
            // FrameIndex, MemOp, and Special should not appear pre-regalloc.
            // Silently mapping them to Imm(0) produces wrong code (#94).
            IrOperand::FrameIndex(fi) => Err(PipelineError::InvalidOperand(format!(
                "FrameIndex({:?}) operand reached regalloc adapter; \
                 frame indices must be eliminated before register allocation",
                fi
            ))),
            IrOperand::MemOp { base, offset } => Err(PipelineError::InvalidOperand(format!(
                "MemOp(base={:?}, offset={}) operand reached regalloc adapter; \
                 memory operands must be lowered before register allocation",
                base, offset
            ))),
            IrOperand::Special(s) => Err(PipelineError::InvalidOperand(format!(
                "Special({:?}) operand reached regalloc adapter; \
                 special registers must be lowered to PReg before register allocation",
                s
            ))),
            // Symbol operands are pure metadata (relocation targets) — no register involvement.
            // Map to Imm(0) since regalloc doesn't need to track them.
            IrOperand::Symbol(_) => Ok(RaOp::Imm(0)),
        }
    };

    let is_store = inst.flags.contains(llvm2_ir::inst::InstFlags::WRITES_MEMORY);
    let is_branch = inst.flags.contains(llvm2_ir::inst::InstFlags::IS_BRANCH);
    let is_return = inst.flags.contains(llvm2_ir::inst::InstFlags::IS_RETURN);
    let is_cmp = matches!(inst.opcode, IrOpcode::CmpRR | IrOpcode::CmpRI | IrOpcode::Fcmp);

    if is_store || is_branch || is_return || is_cmp || inst.operands.is_empty() {
        // All uses, no defs.
        let uses: Vec<RaOp> = inst.operands.iter()
            .map(|op| convert_op(op))
            .collect::<Result<_, _>>()?;
        Ok((Vec::new(), uses))
    } else {
        // First operand is def, rest are uses.
        let defs = vec![convert_op(&inst.operands[0])?];
        let uses: Vec<RaOp> = inst.operands[1..].iter()
            .map(|op| convert_op(op))
            .collect::<Result<_, _>>()?;
        Ok((defs, uses))
    }
}

// NOTE: `convert_ir_flags_to_regalloc()` has been ELIMINATED as of issue #73.
//
// InstFlags are now unified: llvm2_regalloc re-exports llvm2_ir::InstFlags
// directly. The bit-for-bit flag conversion function is no longer needed —
// flags pass through directly in ir_to_regalloc().

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

/// Resolve symbolic block references in branch instructions to byte offsets.
///
/// After register allocation and frame lowering, branch instructions may still
/// contain `Block(BlockId)` operands from the ISel. This pass computes the byte
/// offset of each block in the final layout and replaces Block operands with
/// immediate offsets (PC-relative, in instruction units).
///
/// Must be called after frame lowering (which may add instructions) and before
/// encoding (which needs concrete offsets).
pub fn resolve_branches(func: &mut IrMachFunction) {
    // Phase 1: Compute the instruction-index offset of each block.
    // Walk blocks in layout order, counting non-pseudo instructions.
    let mut block_offsets: HashMap<BlockId, i64> = HashMap::new();
    let mut current_offset: i64 = 0;

    for &block_id in &func.block_order {
        block_offsets.insert(block_id, current_offset);
        let block = &func.blocks[block_id.0 as usize];
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.0 as usize];
            if !inst.is_pseudo() {
                current_offset += 1; // Each instruction is 4 bytes = 1 instruction unit
            }
        }
    }

    // Phase 2: Walk all instructions and replace Block operands with offsets.
    // For each branch instruction, find the Block operand, look up its offset,
    // and replace it with the PC-relative offset (in instruction units).
    let mut inst_offset: i64 = 0;
    for &block_id in &func.block_order.clone() {
        let block = &func.blocks[block_id.0 as usize];
        let inst_ids: Vec<_> = block.insts.clone();
        for &inst_id in &inst_ids {
            let inst = &func.insts[inst_id.0 as usize];
            if inst.is_pseudo() {
                continue;
            }

            let is_branch = matches!(
                inst.opcode,
                IrOpcode::B | IrOpcode::BCond | IrOpcode::Cbz | IrOpcode::Cbnz
                | IrOpcode::Tbz | IrOpcode::Tbnz
            );

            if is_branch {
                // Find and resolve Block operands to PC-relative immediates.
                let new_operands: Vec<IrOperand> = inst
                    .operands
                    .iter()
                    .map(|op| {
                        if let IrOperand::Block(target_block) = op {
                            if let Some(&target_offset) = block_offsets.get(target_block) {
                                // PC-relative offset in instruction units
                                let rel_offset = target_offset - inst_offset;
                                IrOperand::Imm(rel_offset)
                            } else {
                                op.clone()
                            }
                        } else {
                            op.clone()
                        }
                    })
                    .collect();
                func.insts[inst_id.0 as usize].operands = new_operands;
            }

            inst_offset += 1;
        }
    }
}

/// Encode a single IR instruction into AArch64 machine code.
///
/// Delegates to the unified encoder in [`crate::aarch64::encode`] which
/// dispatches to the correct format-specific encoding function.
fn encode_ir_inst(inst: &IrMachInst) -> Result<u32, PipelineError> {
    crate::aarch64::encode::encode_instruction(inst)
        .map_err(|e| PipelineError::Encoding(e.to_string()))
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

        // Debug: verify succs/preds are populated for multi-block functions
        #[cfg(debug_assertions)]
        {
            let has_branches = ir_func.blocks.len() > 1;
            if has_branches {
                let total_succs: usize = ir_func.blocks.iter().map(|b| b.succs.len()).sum();
                let total_preds: usize = ir_func.blocks.iter().map(|b| b.preds.len()).sum();
                debug_assert!(
                    total_succs > 0,
                    "Multi-block function '{}' has no successor edges! Blocks: {}",
                    ir_func.name, ir_func.blocks.len()
                );
                debug_assert!(
                    total_preds > 0,
                    "Multi-block function '{}' has no predecessor edges! Blocks: {}",
                    ir_func.name, ir_func.blocks.len()
                );
            }
        }

        // Phase 3: Optimization
        self.run_optimization(&mut ir_func);

        // Phase 4-6: Register Allocation
        self.run_regalloc(&mut ir_func)?;

        // Phase 7: Frame Lowering
        let frame_layout = self.run_frame_lowering(&mut ir_func);

        // Phase 7.5: Branch Resolution — resolve symbolic Block operands to
        // PC-relative byte offsets before encoding.
        resolve_branches(&mut ir_func);

        // Phase 8: Encoding
        let code = encode_function(&ir_func)?;

        // Phase 9: Mach-O Emission (with compact unwind from frame layout)
        let obj_bytes = self.emit_macho(&ir_func.name, &code, Some(&frame_layout));

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
        let frame_layout = self.run_frame_lowering(ir_func);

        // Phase 7.5: Branch Resolution
        resolve_branches(ir_func);

        // Phase 8: Encoding
        let code = encode_function(ir_func)?;

        // Phase 9: Mach-O Emission (with compact unwind from frame layout)
        let obj_bytes = self.emit_macho(&ir_func.name, &code, Some(&frame_layout));

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
        let frame_layout = self.run_frame_lowering(ir_func);

        // Phase 8: Encoding
        let code = encode_function(ir_func)?;

        // Phase 9: Mach-O Emission (with compact unwind from frame layout)
        let obj_bytes = self.emit_macho(&ir_func.name, &code, Some(&frame_layout));

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

        let mut isel = InstructionSelector::new(input.name.clone(), sig.clone());

        // Lower formal arguments at entry: copies ABI physical registers into
        // VRegs and populates the value_map for function parameters.
        isel.lower_formal_arguments(&sig, input.entry_block)
            .map_err(|e| PipelineError::ISel(e.to_string()))?;

        // Sort blocks by ID to ensure deterministic processing order. The entry
        // block is processed first (it should have the lowest ID), then remaining
        // blocks in layout order. This ensures SSA values defined in dominating
        // blocks are available when referenced by later blocks.
        let mut block_order: Vec<_> = input.blocks.keys().copied().collect();
        block_order.sort_by_key(|b| b.0);

        // Process blocks in sorted order. For non-entry blocks, define block
        // parameter Values before processing instructions.
        for block_ref in &block_order {
            let basic_block = &input.blocks[block_ref];
            if *block_ref != input.entry_block && !basic_block.params.is_empty() {
                isel.define_block_params(&basic_block.params);
            }
            isel.select_block(*block_ref, &basic_block.instructions)
                .map_err(|e| PipelineError::ISel(e.to_string()))?;
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
        let mut ra_func = ir_to_regalloc(ir_func)?;

        // Debug: dump block structure before regalloc
        #[cfg(debug_assertions)]
        if std::env::var("LLVM2_DEBUG_REGALLOC").is_ok() {
            eprintln!("=== REGALLOC DEBUG: {} ===", ra_func.name);
            for (bi, block) in ra_func.blocks.iter().enumerate() {
                eprintln!("  block {}: insts={:?} succs={:?} preds={:?}",
                    bi, block.insts.len(), block.succs, block.preds);
            }
            for (ii, inst) in ra_func.insts.iter().enumerate() {
                eprintln!("  inst {}: opc={} defs={:?} uses={:?}",
                    ii, inst.opcode,
                    inst.defs.iter().filter_map(|o| o.as_vreg()).collect::<Vec<_>>(),
                    inst.uses.iter().filter_map(|o| o.as_vreg()).collect::<Vec<_>>());
            }
        }

        // Phase 5: Run the allocator
        let ra_config = llvm2_regalloc::AllocConfig::default_aarch64();
        let result = llvm2_regalloc::allocate(&mut ra_func, &ra_config)
            .map_err(|e| PipelineError::RegAlloc(e.to_string()))?;

        // Debug: dump allocation
        #[cfg(debug_assertions)]
        if std::env::var("LLVM2_DEBUG_REGALLOC").is_ok() {
            eprintln!("  allocation:");
            let mut allocs: Vec<_> = result.allocation.iter().collect();
            allocs.sort_by_key(|(v, _)| v.id);
            for (vreg, preg) in allocs {
                eprintln!("    v{} -> {:?}", vreg.id, preg);
            }
            eprintln!("=== END REGALLOC DEBUG ===");
        }

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
    ///
    /// Returns the computed [`FrameLayout`] so downstream phases (compact unwind
    /// emission) can use it without recomputing.
    fn run_frame_lowering(
        &self,
        func: &mut IrMachFunction,
    ) -> crate::frame::FrameLayout {
        use crate::frame;

        let layout = frame::compute_frame_layout(func, 0, true);
        frame::eliminate_frame_indices(func, &layout);
        frame::insert_prologue_epilogue(func, &layout);
        layout
    }

    /// Phase 9: Emit a Mach-O .o file from encoded machine code.
    ///
    /// When `frame_layout` is provided, a `__LD,__compact_unwind` section is
    /// emitted containing a single compact unwind entry for the function.
    /// This is required by macOS for backtraces, profilers, and exception handling.
    ///
    /// When `self.config.emit_debug` is true, DWARF debug sections are also
    /// emitted (`__DWARF,__debug_info`, `__debug_abbrev`, `__debug_str`,
    /// `__debug_line`).
    fn emit_macho(
        &self,
        func_name: &str,
        code: &[u8],
        frame_layout: Option<&crate::frame::FrameLayout>,
    ) -> Vec<u8> {
        use crate::macho::MachOWriter;
        use crate::unwind::{CompactUnwindEntry, CompactUnwindSection, add_compact_unwind_to_writer};

        let mut writer = MachOWriter::new();
        writer.add_text_section(code);

        // Add the function symbol with Mach-O name mangling (_prefix).
        let symbol_name = format!("_{}", func_name);
        // Section index 1 = __text (first section, 1-based).
        writer.add_symbol(&symbol_name, 1, 0, true);

        // Emit compact unwind section if we have frame layout information.
        if let Some(layout) = frame_layout {
            let mut cu_section = CompactUnwindSection::new();
            // Symbol index 0 = the function symbol we just added.
            let entry = CompactUnwindEntry::from_layout(
                layout,
                0, // function_offset (relocated via ARM64_RELOC_UNSIGNED)
                code.len() as u32,
                0, // symbol_index for the function
            );
            cu_section.add_entry(entry);
            add_compact_unwind_to_writer(&mut writer, &cu_section);
        }

        // Emit DWARF debug sections when debug info is requested.
        if self.config.emit_debug {
            use crate::dwarf_info::{
                DwarfDebugInfo, FunctionDebugInfo, SourceLanguage,
                add_debug_info_to_writer,
            };

            let mut debug_info = DwarfDebugInfo::new(
                &format!("{}.rs", func_name),
                ".",
                SourceLanguage::Rust,
            );
            debug_info.add_standard_types();
            debug_info.add_function(FunctionDebugInfo {
                name: func_name.to_string(),
                low_pc: 0,
                size: code.len() as u32,
                params: vec![],
            });
            add_debug_info_to_writer(&mut writer, &debug_info);
        }

        writer.write()
    }

    /// Convert llvm2-lower types to llvm2-ir types.
    fn convert_lower_types_to_ir(
        &self,
        types: &[llvm2_lower::types::Type],
    ) -> Vec<llvm2_ir::function::Type> {
        types
            .iter()
            .map(|t| Self::convert_lower_type_to_ir(t))
            .collect()
    }

    /// Convert a single llvm2-lower type to an llvm2-ir type.
    fn convert_lower_type_to_ir(t: &llvm2_lower::types::Type) -> llvm2_ir::function::Type {
        match t {
            llvm2_lower::types::Type::I8 => llvm2_ir::function::Type::I8,
            llvm2_lower::types::Type::I16 => llvm2_ir::function::Type::I16,
            llvm2_lower::types::Type::I32 => llvm2_ir::function::Type::I32,
            llvm2_lower::types::Type::I64 => llvm2_ir::function::Type::I64,
            llvm2_lower::types::Type::I128 => llvm2_ir::function::Type::I128,
            llvm2_lower::types::Type::F32 => llvm2_ir::function::Type::F32,
            llvm2_lower::types::Type::F64 => llvm2_ir::function::Type::F64,
            llvm2_lower::types::Type::B1 => llvm2_ir::function::Type::B1,
            llvm2_lower::types::Type::Struct(fields) => {
                let ir_fields: Vec<llvm2_ir::function::Type> =
                    fields.iter().map(|f| Self::convert_lower_type_to_ir(f)).collect();
                llvm2_ir::function::Type::Struct(ir_fields)
            }
            llvm2_lower::types::Type::Array(elem, count) => {
                let ir_elem = Self::convert_lower_type_to_ir(elem);
                llvm2_ir::function::Type::Array(Box::new(ir_elem), *count)
            }
        }
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
