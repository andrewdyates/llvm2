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
//!   llvm2_lower::Function -> llvm2_lower::isel::ISelFunction
//!   (VRegs, canonical llvm2_ir::AArch64Opcode — unified in issue #73)
//!
//! Phase 2: Adapt ISel -> IR (structural only — opcodes are already unified)
//!   llvm2_lower::isel::ISelFunction -> llvm2_ir::MachFunction
//!
//! Phase 3: Optimization (llvm2-opt)
//!   llvm2_ir::MachFunction -> llvm2_ir::MachFunction (optimized)
//!
//! Phase 4: Adapt IR -> RegAlloc
//!   llvm2_ir::MachFunction -> llvm2_regalloc::RegAllocFunction
//!
//! Phase 5: Register Allocation (llvm2-regalloc)
//!   llvm2_regalloc::RegAllocFunction -> AllocationResult (VReg -> PReg map)
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
//! - `MachBlock` / `RegAllocBlock`: structurally identical after `loop_depth`
//!   was added to `llvm2_ir::MachBlock`. Converted via `From<&MachBlock>`.
//! - `StackSlot` / `RegAllocStackSlot`: converted via `From<&StackSlot>`.
//! - Primitive types: `VReg`, `PReg`, `RegClass`, `BlockId`, `InstId`,
//!   `StackSlotId` — all re-exported from `llvm2_ir` by regalloc.
//!
//! **Remaining structural adapters (separate types, with From/TryFrom impls):**
//! - `ISelOperand` / `RegAllocOperand`: ISel has `CondCode`/`Symbol`/`StackSlot(u32)`
//!   variants; regalloc omits `MemOp`/`FrameIndex`/`Special`.
//!   `TryFrom<&MachOperand>` for `RegAllocOperand` handles the conversion.
//! - `ISelInst` / `RegAllocInst`: regalloc separates defs/uses for liveness analysis;
//!   ISel has no flags/implicit-defs/proofs. Unification requires adding
//!   def/use classification to `llvm2_ir::MachInst`.
//! - `ISelBlock`: ISel uses inline `Vec<ISelInst>` (not arena-indexed).
//! - `ISelFunction` / `RegAllocFunction`: ISel uses `HashMap<Block, ISelBlock>`
//!   (construction-friendly); regalloc uses `HashMap<StackSlotId, RegAllocStackSlot>`
//!   + `next_stack_slot`.
//!
//! The canonical types are in `llvm2_ir`: `MachFunction`, `MachInst`, `MachBlock`,
//! `MachOperand`. ISel and regalloc types have been renamed to avoid confusion.
//! See `llvm2_ir::type_hierarchy` for complete documentation.
//! See issue #73 for the remaining unification plan.

use std::collections::HashMap;
use thiserror::Error;

use llvm2_ir::function::{MachFunction as IrMachFunction, Signature as IrSignature, StackSlot as IrStackSlot};
use llvm2_ir::inst::{AArch64Opcode as IrOpcode, MachInst as IrMachInst};
use llvm2_ir::operand::MachOperand as IrOperand;
use llvm2_ir::regs::{PReg, VReg};
use llvm2_ir::types::BlockId;

use llvm2_lower::compute_graph::ComputeGraph;
use llvm2_lower::dispatch::{
    DispatchPlan, VerificationReport,
    generate_dispatch_plan, verify_dispatch_plan_properties,
};
use llvm2_lower::TargetRecommendation;

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
    #[error("dispatch verification failed ({violations} violations): {summary}")]
    DispatchVerificationFailed {
        violations: usize,
        summary: String,
        report: VerificationReport,
    },
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

/// Policy for handling dispatch verification failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchVerifyMode {
    /// Do not verify dispatch plans (skip verification entirely).
    Off,
    /// Verify dispatch plans. On failure, fall back to a CPU-only plan.
    FallbackOnFailure,
    /// Verify dispatch plans. On failure, return an error.
    ErrorOnFailure,
}

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Optimization level.
    pub opt_level: OptLevel,
    /// Whether to emit debug info (placeholder for future).
    pub emit_debug: bool,
    /// Dispatch plan verification mode.
    ///
    /// When set to [`DispatchVerifyMode::FallbackOnFailure`], any dispatch plan
    /// that fails property verification is replaced with a CPU-only plan. When
    /// set to [`DispatchVerifyMode::ErrorOnFailure`], verification failures
    /// produce a [`PipelineError::DispatchVerificationFailed`].
    pub verify_dispatch: DispatchVerifyMode,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            opt_level: OptLevel::O2,
            emit_debug: false,
            verify_dispatch: DispatchVerifyMode::FallbackOnFailure,
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
// Adapter: llvm2_lower::isel::ISelFunction -> llvm2_ir::MachFunction
// ---------------------------------------------------------------------------

// NOTE: `map_isel_opcode()` has been ELIMINATED as of issue #73.
//
// The ISel now uses `llvm2_ir::AArch64Opcode` directly (re-exported via
// `llvm2_lower::isel::AArch64Opcode`). There is no longer a separate ISel
// opcode enum, so no mapping is needed. The `isel_to_ir` adapter below
// uses `isel_inst.opcode` directly as an `IrOpcode`.

/// Convert ISel's `ISelFunction` to the canonical `llvm2_ir::MachFunction`.
///
/// Delegates to `ISelFunction::to_ir_func()` which owns the conversion logic
/// (issue #73 consolidation). This thin wrapper is preserved for backward
/// compatibility — new code should call `isel_func.to_ir_func()` directly.
pub fn isel_to_ir(
    isel_func: &llvm2_lower::isel::ISelFunction,
    _param_types: &[llvm2_ir::function::Type],
    _return_types: &[llvm2_ir::function::Type],
) -> IrMachFunction {
    // The ISelFunction::to_ir_func() method derives the IR signature from
    // the ISel's own signature using From<Signature>. The param_types and
    // return_types arguments are now unused — they were only needed when
    // the conversion was done externally.
    isel_func.to_ir_func()
}

// ---------------------------------------------------------------------------
// Adapter: llvm2_ir::MachFunction -> llvm2_regalloc::RegAllocFunction
// ---------------------------------------------------------------------------

/// Convert the canonical `llvm2_ir::MachFunction` to `RegAllocFunction` format.
///
/// The `RegAllocFunction` separates defs from uses for liveness analysis.
/// We use a simple heuristic: first operand is def, rest are uses
/// (for most instructions). Special cases: branches have no defs, etc.
///
/// Uses `From`/`TryFrom` impls from `llvm2_regalloc::machine_types` for
/// individual block and operand conversions (issue #73 consolidation).
pub fn ir_to_regalloc(ir_func: &IrMachFunction) -> Result<llvm2_regalloc::RegAllocFunction, PipelineError> {
    use llvm2_regalloc::machine_types as ra;

    let mut ra_func = ra::RegAllocFunction {
        name: ir_func.name.clone(),
        insts: Vec::new(),
        blocks: Vec::new(),
        block_order: ir_func.block_order.clone(),
        entry_block: ir_func.entry,
        next_vreg: ir_func.next_vreg,
        next_stack_slot: ir_func.stack_slots.len() as u32,
        stack_slots: HashMap::new(),
    };

    // Convert stack slots using From impl (issue #73).
    for (i, slot) in ir_func.stack_slots.iter().enumerate() {
        ra_func.stack_slots.insert(
            llvm2_ir::types::StackSlotId(i as u32),
            ra::RegAllocStackSlot::from(slot),
        );
    }

    // Convert instructions.
    // InstFlags are unified (issue #73) — same type in llvm2_ir and llvm2_regalloc,
    // so flags pass through directly without conversion.
    for ir_inst in &ir_func.insts {
        let (defs, uses) = classify_def_use(ir_inst)?;
        let implicit_defs: Vec<PReg> = ir_inst.implicit_defs.to_vec();
        let implicit_uses: Vec<PReg> = ir_inst.implicit_uses.to_vec();

        ra_func.insts.push(ra::RegAllocInst {
            opcode: ir_inst.opcode as u16,
            defs,
            uses,
            implicit_defs,
            implicit_uses,
            flags: ir_inst.flags,
        });
    }

    // Convert blocks using From impl (issue #73).
    for ir_block in &ir_func.blocks {
        ra_func.blocks.push(ra::RegAllocBlock::from(ir_block));
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
///
/// Uses `TryFrom<&MachOperand>` for `RegAllocOperand` (issue #73) for the
/// individual operand conversions. The TryFrom impl rejects IR-only variants
/// (FrameIndex, MemOp, Special) that must be lowered before register
/// allocation.
fn classify_def_use(
    inst: &IrMachInst,
) -> Result<(Vec<llvm2_regalloc::RegAllocOperand>, Vec<llvm2_regalloc::RegAllocOperand>), PipelineError> {
    use llvm2_regalloc::machine_types::RegAllocOperand as RaOp;

    let convert_op = |op: &IrOperand| -> Result<RaOp, PipelineError> {
        RaOp::try_from(op).map_err(|e| PipelineError::InvalidOperand(e.message))
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
// Copy lowering: pseudo Copy -> real MovR
// ---------------------------------------------------------------------------

/// Lower all pseudo `Copy` instructions to real `MovR` instructions.
///
/// After register allocation, the allocator inserts `Copy` pseudo-instructions
/// to move data between physical registers. These must be lowered to real
/// `MovR` (encoded as `ORR Rd, XZR, Rm`) before encoding, since the encoder
/// skips pseudo-instructions.
///
/// Copies where source == destination are eliminated entirely (converted to Nop).
pub fn lower_copies(ir_func: &mut IrMachFunction) {
    for inst in &mut ir_func.insts {
        if inst.opcode == IrOpcode::Copy && inst.operands.len() >= 2 {
            // Copy operands: [dst, src]
            // Check if src == dst (redundant copy).
            let is_redundant = match (&inst.operands[0], &inst.operands[1]) {
                (IrOperand::PReg(a), IrOperand::PReg(b)) => a == b,
                _ => false,
            };
            if is_redundant {
                inst.opcode = IrOpcode::Nop;
                // Nop is still pseudo, flags stay IS_PSEUDO — that's correct.
            } else {
                inst.opcode = IrOpcode::MovR;
                // Clear the IS_PSEUDO flag so the encoder processes this as a
                // real instruction. MovR encodes as ORR Rd, XZR, Rm.
                inst.flags.remove(llvm2_ir::inst::InstFlags::IS_PSEUDO);
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

        // Phase 2: Convert ISel output to shared IR (issue #73 consolidation:
        // conversion logic now lives in ISelFunction::to_ir_func()).
        let mut ir_func = isel_func.to_ir_func();

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

        // Phase 6.5: Lower pseudo Copy instructions to real MovR.
        lower_copies(&mut ir_func);

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

        // Phase 6.5: Lower pseudo Copy instructions to real MovR.
        lower_copies(ir_func);

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
    ) -> Result<llvm2_lower::isel::ISelFunction, PipelineError> {
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

    // --- Dispatch verification ---

    /// Verify a dispatch plan against a compute graph, applying the configured
    /// [`DispatchVerifyMode`] policy.
    ///
    /// Returns:
    /// - `Ok(plan)` if verification passes or is disabled
    /// - `Ok(cpu_only_plan)` if verification fails under
    ///   [`DispatchVerifyMode::FallbackOnFailure`]
    /// - `Err(DispatchVerificationFailed)` if verification fails under
    ///   [`DispatchVerifyMode::ErrorOnFailure`]
    pub fn verify_dispatch_plan(
        &self,
        graph: &ComputeGraph,
        plan: DispatchPlan,
        recommendations: &[TargetRecommendation],
    ) -> Result<DispatchPlan, PipelineError> {
        match self.config.verify_dispatch {
            DispatchVerifyMode::Off => Ok(plan),
            DispatchVerifyMode::FallbackOnFailure => {
                let report = verify_dispatch_plan_properties(graph, &plan);
                if report.all_ok() {
                    Ok(plan)
                } else {
                    // Fall back to a CPU-only plan.
                    Ok(generate_cpu_only_plan(graph, recommendations))
                }
            }
            DispatchVerifyMode::ErrorOnFailure => {
                let report = verify_dispatch_plan_properties(graph, &plan);
                if report.all_ok() {
                    Ok(plan)
                } else {
                    let violations = report.total_violations();
                    let summary = format!("{}", report);
                    Err(PipelineError::DispatchVerificationFailed {
                        violations,
                        summary,
                        report,
                    })
                }
            }
        }
    }

    /// Generate a dispatch plan, verify it, and return the verified (or
    /// fallback) plan. This is the primary integration point for dispatch
    /// verification in the compilation pipeline.
    ///
    /// Combines plan generation with verification in a single call.
    pub fn generate_and_verify_dispatch(
        &self,
        graph: &ComputeGraph,
        recommendations: &[TargetRecommendation],
    ) -> Result<DispatchPlan, PipelineError> {
        let plan = generate_dispatch_plan(graph, recommendations);
        self.verify_dispatch_plan(graph, plan, recommendations)
    }
}

// ---------------------------------------------------------------------------
// CPU-only dispatch plan fallback
// ---------------------------------------------------------------------------

/// Generate a CPU-only dispatch plan as a safe fallback.
///
/// When dispatch verification fails, this function generates a conservative
/// plan that runs every node on `CpuScalar`. No data transfers or
/// synchronizations are needed because all execution stays on the CPU.
pub fn generate_cpu_only_plan(
    graph: &ComputeGraph,
    _recommendations: &[TargetRecommendation],
) -> DispatchPlan {
    use llvm2_lower::dispatch::DispatchOp;
    use llvm2_lower::target_analysis::ComputeTarget;

    let mut assignment = HashMap::new();
    let mut ops = Vec::new();
    let mut total_cycles: u64 = 0;

    // Walk nodes in ID order. Since all nodes are on the same target
    // (CpuScalar), no transfers or syncs are needed.
    let mut node_ids: Vec<_> = graph.nodes.iter().map(|n| n.id).collect();
    node_ids.sort_by_key(|id| id.0);

    for node_id in &node_ids {
        assignment.insert(*node_id, ComputeTarget::CpuScalar);

        let node = graph.node(*node_id);
        let estimated_cycles = node
            .and_then(|n| n.costs.get(&ComputeTarget::CpuScalar))
            .map(|c| c.latency_cycles)
            .unwrap_or(1);

        ops.push(DispatchOp::CpuFallback {
            node_id: *node_id,
            reason: "dispatch verification fallback".to_string(),
        });
        total_cycles = total_cycles.saturating_add(estimated_cycles);
    }

    DispatchPlan {
        ops,
        assignment,
        estimated_total_cycles: total_cycles,
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
        verify_dispatch: DispatchVerifyMode::FallbackOnFailure,
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

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::function::{MachFunction as IrMachFunction, Signature as IrSignature, Type};
    use llvm2_ir::inst::{AArch64Opcode as IrOpcode, InstFlags, MachInst as IrMachInst};
    use llvm2_ir::operand::MachOperand as IrOperand;
    use llvm2_ir::regs::{RegClass, VReg, X0, X1, X2};
    use llvm2_ir::types::BlockId;

    // -- Helper: create an instruction with specific flags --

    fn make_inst(opcode: IrOpcode, operands: Vec<IrOperand>) -> IrMachInst {
        IrMachInst::new(opcode, operands)
    }

    fn make_inst_with_flags(opcode: IrOpcode, operands: Vec<IrOperand>, flags: InstFlags) -> IrMachInst {
        IrMachInst::with_flags(opcode, operands, flags)
    }

    // =========================================================================
    // classify_def_use tests
    // =========================================================================

    #[test]
    fn classify_def_use_alu_instruction_first_operand_is_def() {
        // AddRR: [dst, src1, src2] -> defs=[dst], uses=[src1, src2]
        let v0 = VReg::new(0, RegClass::Gpr64);
        let v1 = VReg::new(1, RegClass::Gpr64);
        let v2 = VReg::new(2, RegClass::Gpr64);
        let inst = make_inst(
            IrOpcode::AddRR,
            vec![IrOperand::VReg(v0), IrOperand::VReg(v1), IrOperand::VReg(v2)],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(uses.len(), 2);
        assert_eq!(defs[0].as_vreg(), Some(v0));
        assert_eq!(uses[0].as_vreg(), Some(v1));
        assert_eq!(uses[1].as_vreg(), Some(v2));
    }

    #[test]
    fn classify_def_use_add_immediate() {
        // AddRI: [dst, src, imm] -> defs=[dst], uses=[src, imm]
        let v0 = VReg::new(0, RegClass::Gpr64);
        let v1 = VReg::new(1, RegClass::Gpr64);
        let inst = make_inst(
            IrOpcode::AddRI,
            vec![IrOperand::VReg(v0), IrOperand::VReg(v1), IrOperand::Imm(42)],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(uses.len(), 2);
        assert_eq!(defs[0].as_vreg(), Some(v0));
    }

    #[test]
    fn classify_def_use_store_all_uses_no_defs() {
        // StrRI has WRITES_MEMORY flag -> all operands are uses
        let inst = make_inst(
            IrOpcode::StrRI,
            vec![
                IrOperand::PReg(X0),
                IrOperand::PReg(X1),
                IrOperand::Imm(8),
            ],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty(), "store should have no defs");
        assert_eq!(uses.len(), 3);
    }

    #[test]
    fn classify_def_use_branch_all_uses_no_defs() {
        // B has IS_BRANCH flag -> all operands are uses
        let inst = make_inst(
            IrOpcode::B,
            vec![IrOperand::Block(BlockId(1))],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty(), "branch should have no defs");
        assert_eq!(uses.len(), 1);
    }

    #[test]
    fn classify_def_use_conditional_branch() {
        // BCond has IS_BRANCH flag -> all uses
        let inst = make_inst(
            IrOpcode::BCond,
            vec![IrOperand::Imm(0), IrOperand::Block(BlockId(2))],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty());
        assert_eq!(uses.len(), 2);
    }

    #[test]
    fn classify_def_use_return_all_uses_no_defs() {
        // Ret has IS_RETURN flag -> all uses
        let inst = make_inst(IrOpcode::Ret, vec![]);

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty());
        assert!(uses.is_empty());
    }

    #[test]
    fn classify_def_use_compare_all_uses_no_defs() {
        // CmpRR -> classified as compare, all uses
        let inst = make_inst(
            IrOpcode::CmpRR,
            vec![IrOperand::PReg(X0), IrOperand::PReg(X1)],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty(), "compare should have no defs");
        assert_eq!(uses.len(), 2);
    }

    #[test]
    fn classify_def_use_compare_immediate() {
        let inst = make_inst(
            IrOpcode::CmpRI,
            vec![IrOperand::PReg(X0), IrOperand::Imm(0)],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty());
        assert_eq!(uses.len(), 2);
    }

    #[test]
    fn classify_def_use_fcmp_all_uses() {
        let v0 = VReg::new(0, RegClass::Fpr64);
        let v1 = VReg::new(1, RegClass::Fpr64);
        let inst = make_inst(
            IrOpcode::Fcmp,
            vec![IrOperand::VReg(v0), IrOperand::VReg(v1)],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty(), "fcmp should have no defs");
        assert_eq!(uses.len(), 2);
    }

    #[test]
    fn classify_def_use_load_first_operand_is_def() {
        // LdrRI: [dst, base, offset] -> defs=[dst], uses=[base, offset]
        let v0 = VReg::new(0, RegClass::Gpr64);
        let inst = make_inst(
            IrOpcode::LdrRI,
            vec![IrOperand::VReg(v0), IrOperand::PReg(X1), IrOperand::Imm(0)],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(uses.len(), 2);
        assert_eq!(defs[0].as_vreg(), Some(v0));
    }

    #[test]
    fn classify_def_use_empty_operands_no_defs_no_uses() {
        // An instruction with no operands -> empty defs and uses regardless of flags
        let inst = make_inst(IrOpcode::Nop, vec![]);

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty());
        assert!(uses.is_empty());
    }

    #[test]
    fn classify_def_use_mov_immediate() {
        // MovI: [dst, imm] -> defs=[dst], uses=[imm]
        let v0 = VReg::new(0, RegClass::Gpr64);
        let inst = make_inst(
            IrOpcode::MovI,
            vec![IrOperand::VReg(v0), IrOperand::Imm(100)],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(uses.len(), 1);
        assert_eq!(defs[0].as_vreg(), Some(v0));
    }

    #[test]
    fn classify_def_use_physical_register_operands() {
        // Post-regalloc instruction with PRegs
        let inst = make_inst(
            IrOpcode::AddRR,
            vec![IrOperand::PReg(X0), IrOperand::PReg(X1), IrOperand::PReg(X2)],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(uses.len(), 2);
        assert_eq!(defs[0].as_preg(), Some(X0));
        assert_eq!(uses[0].as_preg(), Some(X1));
    }

    #[test]
    fn classify_def_use_rejects_frame_index_operand() {
        // FrameIndex should be lowered before regalloc; classify_def_use should error
        let inst = make_inst(
            IrOpcode::AddRR,
            vec![
                IrOperand::FrameIndex(llvm2_ir::types::FrameIdx(-8)),
                IrOperand::PReg(X0),
            ],
        );

        let result = classify_def_use(&inst);
        assert!(result.is_err());
        match result.unwrap_err() {
            PipelineError::InvalidOperand(msg) => {
                assert!(msg.contains("FrameIndex"));
            }
            other => panic!("expected InvalidOperand, got {:?}", other),
        }
    }

    #[test]
    fn classify_def_use_rejects_memop_operand() {
        let inst = make_inst(
            IrOpcode::AddRR,
            vec![
                IrOperand::MemOp { base: X0, offset: 16 },
                IrOperand::PReg(X1),
            ],
        );

        let result = classify_def_use(&inst);
        assert!(result.is_err());
    }

    #[test]
    fn classify_def_use_rejects_special_operand() {
        let inst = make_inst(
            IrOpcode::AddRR,
            vec![
                IrOperand::Special(llvm2_ir::regs::SpecialReg::SP),
                IrOperand::PReg(X0),
            ],
        );

        let result = classify_def_use(&inst);
        assert!(result.is_err());
    }

    #[test]
    fn classify_def_use_store_with_writes_memory_flag() {
        // Verify the WRITES_MEMORY flag drives the "all uses" classification
        // even with a non-standard opcode that has WRITES_MEMORY set manually
        let inst = make_inst_with_flags(
            IrOpcode::StpRI,
            vec![IrOperand::PReg(X0), IrOperand::PReg(X1), IrOperand::PReg(X2), IrOperand::Imm(0)],
            InstFlags::WRITES_MEMORY | InstFlags::HAS_SIDE_EFFECTS,
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty());
        assert_eq!(uses.len(), 4);
    }

    #[test]
    fn classify_def_use_cbz_branch() {
        let inst = make_inst(
            IrOpcode::Cbz,
            vec![IrOperand::PReg(X0), IrOperand::Block(BlockId(3))],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty(), "cbz is a branch, no defs");
        assert_eq!(uses.len(), 2);
    }

    // =========================================================================
    // apply_regalloc tests
    // =========================================================================

    #[test]
    fn apply_regalloc_rewrites_vregs_to_pregs() {
        let sig = IrSignature::new(vec![Type::I64, Type::I64], vec![Type::I64]);
        let mut func = IrMachFunction::new("test".to_string(), sig);

        let v0 = VReg::new(0, RegClass::Gpr64);
        let v1 = VReg::new(1, RegClass::Gpr64);
        let v2 = VReg::new(2, RegClass::Gpr64);

        let inst = IrMachInst::new(
            IrOpcode::AddRR,
            vec![IrOperand::VReg(v0), IrOperand::VReg(v1), IrOperand::VReg(v2)],
        );
        func.push_inst(inst);

        let mut allocation = HashMap::new();
        allocation.insert(v0, X0);
        allocation.insert(v1, X1);
        allocation.insert(v2, X2);

        apply_regalloc(&mut func, &allocation);

        assert_eq!(func.insts[0].operands[0], IrOperand::PReg(X0));
        assert_eq!(func.insts[0].operands[1], IrOperand::PReg(X1));
        assert_eq!(func.insts[0].operands[2], IrOperand::PReg(X2));
    }

    #[test]
    fn apply_regalloc_leaves_pregs_unchanged() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);

        let inst = IrMachInst::new(
            IrOpcode::AddRR,
            vec![IrOperand::PReg(X0), IrOperand::PReg(X1), IrOperand::PReg(X2)],
        );
        func.push_inst(inst);

        let allocation = HashMap::new();
        apply_regalloc(&mut func, &allocation);

        assert_eq!(func.insts[0].operands[0], IrOperand::PReg(X0));
        assert_eq!(func.insts[0].operands[1], IrOperand::PReg(X1));
        assert_eq!(func.insts[0].operands[2], IrOperand::PReg(X2));
    }

    #[test]
    fn apply_regalloc_leaves_immediates_unchanged() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);

        let v0 = VReg::new(0, RegClass::Gpr64);
        let inst = IrMachInst::new(
            IrOpcode::AddRI,
            vec![IrOperand::VReg(v0), IrOperand::PReg(X1), IrOperand::Imm(42)],
        );
        func.push_inst(inst);

        let mut allocation = HashMap::new();
        allocation.insert(v0, X0);

        apply_regalloc(&mut func, &allocation);

        assert_eq!(func.insts[0].operands[0], IrOperand::PReg(X0));
        assert_eq!(func.insts[0].operands[2], IrOperand::Imm(42));
    }

    #[test]
    fn apply_regalloc_unallocated_vreg_stays_vreg() {
        // VRegs not in the allocation map are left as-is (spilled)
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);

        let v0 = VReg::new(0, RegClass::Gpr64);
        let inst = IrMachInst::new(IrOpcode::MovI, vec![IrOperand::VReg(v0), IrOperand::Imm(0)]);
        func.push_inst(inst);

        let allocation = HashMap::new(); // empty allocation
        apply_regalloc(&mut func, &allocation);

        assert_eq!(func.insts[0].operands[0], IrOperand::VReg(v0));
    }

    // =========================================================================
    // lower_copies tests
    // =========================================================================

    #[test]
    fn lower_copies_converts_copy_to_movr() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);

        let copy = IrMachInst::new(
            IrOpcode::Copy,
            vec![IrOperand::PReg(X0), IrOperand::PReg(X1)],
        );
        // Copy starts as pseudo
        assert!(copy.flags.contains(InstFlags::IS_PSEUDO));
        func.push_inst(copy);

        lower_copies(&mut func);

        assert_eq!(func.insts[0].opcode, IrOpcode::MovR);
        assert!(!func.insts[0].flags.contains(InstFlags::IS_PSEUDO),
            "MovR should not be pseudo after lowering");
    }

    #[test]
    fn lower_copies_eliminates_redundant_copy() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);

        // Copy X0 -> X0 (redundant)
        let copy = IrMachInst::new(
            IrOpcode::Copy,
            vec![IrOperand::PReg(X0), IrOperand::PReg(X0)],
        );
        func.push_inst(copy);

        lower_copies(&mut func);

        assert_eq!(func.insts[0].opcode, IrOpcode::Nop,
            "redundant copy should become Nop");
    }

    #[test]
    fn lower_copies_leaves_non_copy_instructions() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);

        let add = IrMachInst::new(
            IrOpcode::AddRR,
            vec![IrOperand::PReg(X0), IrOperand::PReg(X1), IrOperand::PReg(X2)],
        );
        func.push_inst(add);

        lower_copies(&mut func);

        assert_eq!(func.insts[0].opcode, IrOpcode::AddRR);
    }

    #[test]
    fn lower_copies_handles_mixed_instructions() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);

        // Non-redundant copy
        func.push_inst(IrMachInst::new(
            IrOpcode::Copy,
            vec![IrOperand::PReg(X0), IrOperand::PReg(X1)],
        ));
        // Regular instruction
        func.push_inst(IrMachInst::new(
            IrOpcode::AddRR,
            vec![IrOperand::PReg(X0), IrOperand::PReg(X0), IrOperand::PReg(X2)],
        ));
        // Redundant copy
        func.push_inst(IrMachInst::new(
            IrOpcode::Copy,
            vec![IrOperand::PReg(X2), IrOperand::PReg(X2)],
        ));

        lower_copies(&mut func);

        assert_eq!(func.insts[0].opcode, IrOpcode::MovR);
        assert_eq!(func.insts[1].opcode, IrOpcode::AddRR);
        assert_eq!(func.insts[2].opcode, IrOpcode::Nop);
    }

    // =========================================================================
    // resolve_branches tests
    // =========================================================================

    #[test]
    fn resolve_branches_single_block_no_branches() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);
        let entry = func.entry;

        let add = IrMachInst::new(
            IrOpcode::AddRR,
            vec![IrOperand::PReg(X0), IrOperand::PReg(X0), IrOperand::PReg(X1)],
        );
        let add_id = func.push_inst(add);
        func.append_inst(entry, add_id);

        let ret = IrMachInst::new(IrOpcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(entry, ret_id);

        resolve_branches(&mut func);

        // No branches, nothing should change
        assert_eq!(func.insts[add_id.0 as usize].operands.len(), 3);
    }

    #[test]
    fn resolve_branches_replaces_block_operand_with_offset() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);
        let entry = func.entry;
        let bb1 = func.create_block();

        // bb0: B bb1
        let br = IrMachInst::new(IrOpcode::B, vec![IrOperand::Block(bb1)]);
        let br_id = func.push_inst(br);
        func.append_inst(entry, br_id);

        // bb1: RET
        let ret = IrMachInst::new(IrOpcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(bb1, ret_id);

        resolve_branches(&mut func);

        // Branch at offset 0 targets bb1 at offset 1 -> relative offset = 1
        let resolved_operand = &func.insts[br_id.0 as usize].operands[0];
        assert!(
            matches!(resolved_operand, IrOperand::Imm(1)),
            "expected Imm(1), got {:?}", resolved_operand
        );
    }

    // =========================================================================
    // PipelineConfig / OptLevel tests
    // =========================================================================

    #[test]
    fn pipeline_config_default_is_o2() {
        let config = PipelineConfig::default();
        assert_eq!(config.opt_level, OptLevel::O2);
        assert!(!config.emit_debug);
    }

    #[test]
    fn opt_level_equality() {
        assert_eq!(OptLevel::O0, OptLevel::O0);
        assert_eq!(OptLevel::O1, OptLevel::O1);
        assert_eq!(OptLevel::O2, OptLevel::O2);
        assert_eq!(OptLevel::O3, OptLevel::O3);
        assert_ne!(OptLevel::O0, OptLevel::O3);
        assert_ne!(OptLevel::O1, OptLevel::O2);
    }

    #[test]
    fn pipeline_config_custom() {
        let config = PipelineConfig {
            opt_level: OptLevel::O0,
            emit_debug: true,
        };
        assert_eq!(config.opt_level, OptLevel::O0);
        assert!(config.emit_debug);
    }

    #[test]
    fn pipeline_config_clone() {
        let config = PipelineConfig {
            opt_level: OptLevel::O3,
            emit_debug: true,
        };
        let cloned = config.clone();
        assert_eq!(cloned.opt_level, OptLevel::O3);
        assert!(cloned.emit_debug);
    }

    // =========================================================================
    // Pipeline constructor tests
    // =========================================================================

    #[test]
    fn pipeline_new_with_config() {
        let config = PipelineConfig {
            opt_level: OptLevel::O1,
            emit_debug: false,
        };
        let pipeline = Pipeline::new(config);
        assert_eq!(pipeline.config.opt_level, OptLevel::O1);
    }

    #[test]
    fn pipeline_default_o2() {
        let pipeline = Pipeline::default_o2();
        assert_eq!(pipeline.config.opt_level, OptLevel::O2);
        assert!(!pipeline.config.emit_debug);
    }

    // =========================================================================
    // CompilationUnit tests
    // =========================================================================

    #[test]
    fn compilation_unit_new() {
        let sig = IrSignature::new(vec![Type::I64], vec![Type::I64]);
        let func = IrMachFunction::new("my_func".to_string(), sig);
        let config = PipelineConfig::default();
        let unit = CompilationUnit::new(func, config);

        assert_eq!(unit.name, "my_func");
        assert!(unit.code.is_empty());
        assert_eq!(unit.config.opt_level, OptLevel::O2);
    }

    // =========================================================================
    // build_add_test_function tests
    // =========================================================================

    #[test]
    fn build_add_test_function_structure() {
        let func = build_add_test_function();

        assert_eq!(func.name, "add");
        assert_eq!(func.signature.params.len(), 2);
        assert_eq!(func.signature.returns.len(), 1);
        assert_eq!(func.num_blocks(), 1);
        // Should have 2 instructions: ADD + RET
        assert_eq!(func.block(func.entry).len(), 2);
    }

    #[test]
    fn build_add_test_function_instructions() {
        let func = build_add_test_function();
        let entry_block = func.block(func.entry);

        let add_inst = func.inst(entry_block.insts[0]);
        assert_eq!(add_inst.opcode, IrOpcode::AddRR);
        assert_eq!(add_inst.operands.len(), 3);
        assert_eq!(add_inst.operands[0], IrOperand::PReg(X0));
        assert_eq!(add_inst.operands[1], IrOperand::PReg(X0));
        assert_eq!(add_inst.operands[2], IrOperand::PReg(X1));

        let ret_inst = func.inst(entry_block.insts[1]);
        assert_eq!(ret_inst.opcode, IrOpcode::Ret);
        assert!(ret_inst.operands.is_empty());
    }

    // =========================================================================
    // ir_to_regalloc adapter tests
    // =========================================================================

    #[test]
    fn ir_to_regalloc_preserves_function_name() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("my_function".to_string(), sig);
        let entry = func.entry;

        let ret = IrMachInst::new(IrOpcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(entry, ret_id);

        let ra_func = ir_to_regalloc(&func).unwrap();
        assert_eq!(ra_func.name, "my_function");
    }

    #[test]
    fn ir_to_regalloc_converts_stack_slots() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);
        func.alloc_stack_slot(llvm2_ir::function::StackSlot::new(8, 8));
        func.alloc_stack_slot(llvm2_ir::function::StackSlot::new(16, 16));

        let entry = func.entry;
        let ret = IrMachInst::new(IrOpcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(entry, ret_id);

        let ra_func = ir_to_regalloc(&func).unwrap();
        assert_eq!(ra_func.stack_slots.len(), 2);
        assert_eq!(ra_func.next_stack_slot, 2);
    }

    #[test]
    fn ir_to_regalloc_preserves_block_order() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);
        let entry = func.entry;
        let bb1 = func.create_block();

        // Add instructions to both blocks
        let nop1 = IrMachInst::new(IrOpcode::Nop, vec![]);
        let nop1_id = func.push_inst(nop1);
        func.append_inst(entry, nop1_id);

        let nop2 = IrMachInst::new(IrOpcode::Nop, vec![]);
        let nop2_id = func.push_inst(nop2);
        func.append_inst(bb1, nop2_id);

        let ra_func = ir_to_regalloc(&func).unwrap();
        assert_eq!(ra_func.block_order.len(), 2);
        assert_eq!(ra_func.block_order[0], BlockId(0));
        assert_eq!(ra_func.block_order[1], BlockId(1));
        assert_eq!(ra_func.entry_block, BlockId(0));
    }

    // =========================================================================
    // PipelineError Display tests
    // =========================================================================

    #[test]
    fn pipeline_error_display_isel() {
        let err = PipelineError::ISel("something went wrong".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("instruction selection failed"));
        assert!(msg.contains("something went wrong"));
    }

    #[test]
    fn pipeline_error_display_regalloc() {
        let err = PipelineError::RegAlloc("out of registers".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("register allocation failed"));
    }

    #[test]
    fn pipeline_error_display_encoding() {
        let err = PipelineError::Encoding("bad instruction".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("encoding failed"));
    }

    #[test]
    fn pipeline_error_display_unsupported_opcode() {
        let err = PipelineError::UnsupportedOpcode(IrOpcode::Phi);
        let msg = format!("{}", err);
        assert!(msg.contains("unsupported opcode"));
    }

    #[test]
    fn pipeline_error_display_invalid_operand() {
        let err = PipelineError::InvalidOperand("FrameIndex must be eliminated".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("invalid operand"));
    }
}

// ---------------------------------------------------------------------------
// Tests — dispatch verification integration
// ---------------------------------------------------------------------------

#[cfg(test)]
mod dispatch_verification_tests {
    use super::*;
    use llvm2_lower::compute_graph::{
        ComputeCost, ComputeNode, ComputeNodeId, DataEdge, NodeKind, TransferCost,
    };
    use llvm2_lower::target_analysis::ComputeTarget;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Build a ComputeGraph from nodes and edges (avoids pub(crate) field issues).
    fn make_graph(nodes: Vec<ComputeNode>, edges: Vec<DataEdge>) -> ComputeGraph {
        let mut graph = ComputeGraph::new();
        graph.nodes = nodes;
        graph.edges = edges;
        graph
    }

    /// Build a simple single-node scalar graph.
    fn scalar_graph() -> (ComputeGraph, Vec<TargetRecommendation>) {
        let mut costs = HashMap::new();
        costs.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 10,
            throughput_ops_per_kcycle: 1000,
        });

        let node = ComputeNode {
            id: ComputeNodeId(0),
            instructions: vec![],
            costs,
            legal_targets: vec![ComputeTarget::CpuScalar],
            kind: NodeKind::Scalar,
            data_size_bytes: 8,
            produced_values: vec![],
            consumed_values: vec![],
            dominant_op: "ADD".to_string(),
            target_legality: None,
        };

        let graph = make_graph(vec![node], vec![]);

        let recs = vec![TargetRecommendation {
            node_id: ComputeNodeId(0),
            recommended_target: ComputeTarget::CpuScalar,
            legal_targets: vec![ComputeTarget::CpuScalar],
            reason: "scalar op".to_string(),
            parallel_reduction_legal: false,
        }];

        (graph, recs)
    }

    /// Build a two-node GPU graph: CPU producer -> GPU consumer.
    fn gpu_graph() -> (ComputeGraph, Vec<TargetRecommendation>) {
        let mut cpu_costs = HashMap::new();
        cpu_costs.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 20,
            throughput_ops_per_kcycle: 1000,
        });

        let mut gpu_costs = HashMap::new();
        gpu_costs.insert(ComputeTarget::Gpu, ComputeCost {
            latency_cycles: 5,
            throughput_ops_per_kcycle: 100_000,
        });
        gpu_costs.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 500,
            throughput_ops_per_kcycle: 1000,
        });

        let producer = ComputeNode {
            id: ComputeNodeId(0),
            instructions: vec![],
            costs: cpu_costs,
            legal_targets: vec![ComputeTarget::CpuScalar],
            kind: NodeKind::Scalar,
            data_size_bytes: 8,
            produced_values: vec![],
            consumed_values: vec![],
            dominant_op: "ADD".to_string(),
            target_legality: None,
        };

        let consumer = ComputeNode {
            id: ComputeNodeId(1),
            instructions: vec![],
            costs: gpu_costs,
            legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
            kind: NodeKind::DataParallel,
            data_size_bytes: 4096,
            produced_values: vec![],
            consumed_values: vec![],
            dominant_op: "ADD".to_string(),
            target_legality: None,
        };

        let edge = DataEdge {
            from: ComputeNodeId(0),
            to: ComputeNodeId(1),
            transfer_bytes: 4096,
            transfer_cost: TransferCost::zero(),
        };

        let graph = make_graph(vec![producer, consumer], vec![edge]);

        let recs = vec![
            TargetRecommendation {
                node_id: ComputeNodeId(0),
                recommended_target: ComputeTarget::CpuScalar,
                legal_targets: vec![ComputeTarget::CpuScalar],
                reason: "scalar input".to_string(),
                parallel_reduction_legal: false,
            },
            TargetRecommendation {
                node_id: ComputeNodeId(1),
                recommended_target: ComputeTarget::Gpu,
                legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
                reason: "data-parallel on GPU".to_string(),
                parallel_reduction_legal: true,
            },
        ];

        (graph, recs)
    }

    /// Build a graph with a GPU-only node (missing CpuScalar from legal_targets).
    fn bad_fallback_graph() -> (ComputeGraph, Vec<TargetRecommendation>) {
        let mut gpu_costs = HashMap::new();
        gpu_costs.insert(ComputeTarget::Gpu, ComputeCost {
            latency_cycles: 5,
            throughput_ops_per_kcycle: 100_000,
        });

        let node = ComputeNode {
            id: ComputeNodeId(0),
            instructions: vec![],
            costs: gpu_costs,
            legal_targets: vec![ComputeTarget::Gpu], // Missing CpuScalar!
            kind: NodeKind::DataParallel,
            data_size_bytes: 4096,
            produced_values: vec![],
            consumed_values: vec![],
            dominant_op: "ADD".to_string(),
            target_legality: None,
        };

        let graph = make_graph(vec![node], vec![]);

        let recs = vec![TargetRecommendation {
            node_id: ComputeNodeId(0),
            recommended_target: ComputeTarget::Gpu,
            legal_targets: vec![ComputeTarget::Gpu],
            reason: "GPU only".to_string(),
            parallel_reduction_legal: false,
        }];

        (graph, recs)
    }

    // -----------------------------------------------------------------------
    // Test 1: Valid plan passes verification in all modes
    // -----------------------------------------------------------------------

    #[test]
    fn test_valid_plan_passes_all_modes() {
        let (graph, recs) = scalar_graph();

        for mode in [
            DispatchVerifyMode::Off,
            DispatchVerifyMode::FallbackOnFailure,
            DispatchVerifyMode::ErrorOnFailure,
        ] {
            let pipeline = Pipeline::new(PipelineConfig {
                verify_dispatch: mode,
                ..Default::default()
            });

            let result = pipeline.generate_and_verify_dispatch(&graph, &recs);
            assert!(result.is_ok(), "mode {:?} should accept a valid plan", mode);
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: Off mode skips verification entirely
    // -----------------------------------------------------------------------

    #[test]
    fn test_off_mode_skips_verification() {
        let (graph, recs) = bad_fallback_graph();

        let pipeline = Pipeline::new(PipelineConfig {
            verify_dispatch: DispatchVerifyMode::Off,
            ..Default::default()
        });

        // Even a plan with issues should pass when verification is off.
        let result = pipeline.generate_and_verify_dispatch(&graph, &recs);
        assert!(result.is_ok(), "Off mode should skip verification");

        // The plan should be the original (GPU launch, not CPU fallback).
        let plan = result.unwrap();
        let has_gpu_launch = plan.ops.iter().any(|op| {
            matches!(op, llvm2_lower::dispatch::DispatchOp::KernelLaunch {
                target: ComputeTarget::Gpu, ..
            })
        });
        assert!(has_gpu_launch, "Off mode should preserve original GPU plan");
    }

    // -----------------------------------------------------------------------
    // Test 3: FallbackOnFailure mode produces CPU-only plan on bad graph
    // -----------------------------------------------------------------------

    #[test]
    fn test_fallback_mode_produces_cpu_plan_on_failure() {
        let (graph, recs) = bad_fallback_graph();

        let pipeline = Pipeline::new(PipelineConfig {
            verify_dispatch: DispatchVerifyMode::FallbackOnFailure,
            ..Default::default()
        });

        let result = pipeline.generate_and_verify_dispatch(&graph, &recs);
        assert!(result.is_ok(), "FallbackOnFailure should not error");

        let plan = result.unwrap();
        // The fallback plan should contain only CpuFallback ops.
        assert_eq!(plan.count_fallbacks(), 1, "should have 1 CPU fallback");
        assert_eq!(plan.count_launches(), 0, "no kernel launches in fallback");
        assert_eq!(plan.count_transfers(), 0, "no transfers in fallback");
        assert_eq!(plan.count_syncs(), 0, "no syncs in fallback");
    }

    // -----------------------------------------------------------------------
    // Test 4: ErrorOnFailure mode returns error on bad graph
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_mode_returns_error_on_failure() {
        let (graph, recs) = bad_fallback_graph();

        let pipeline = Pipeline::new(PipelineConfig {
            verify_dispatch: DispatchVerifyMode::ErrorOnFailure,
            ..Default::default()
        });

        let result = pipeline.generate_and_verify_dispatch(&graph, &recs);
        assert!(result.is_err(), "ErrorOnFailure should return error");

        match result.unwrap_err() {
            PipelineError::DispatchVerificationFailed { violations, summary, report } => {
                assert!(violations > 0, "should report violations");
                assert!(!summary.is_empty(), "should have summary text");
                assert!(!report.cpu_fallback_ok, "should flag CPU fallback issue");
            }
            other => panic!("Expected DispatchVerificationFailed, got: {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: GPU graph produces valid plan that passes verification
    // -----------------------------------------------------------------------

    #[test]
    fn test_gpu_graph_passes_verification() {
        let (graph, recs) = gpu_graph();

        let pipeline = Pipeline::new(PipelineConfig {
            verify_dispatch: DispatchVerifyMode::ErrorOnFailure,
            ..Default::default()
        });

        let result = pipeline.generate_and_verify_dispatch(&graph, &recs);
        assert!(result.is_ok(), "GPU graph should produce a valid plan");

        let plan = result.unwrap();
        assert!(plan.count_launches() >= 2, "should have CPU + GPU launches");
        assert!(plan.count_transfers() >= 1, "should have at least one transfer");
    }

    // -----------------------------------------------------------------------
    // Test 6: generate_cpu_only_plan produces all-CPU plan
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_cpu_only_plan() {
        let (graph, recs) = gpu_graph();
        let plan = generate_cpu_only_plan(&graph, &recs);

        assert_eq!(plan.ops.len(), 2, "one fallback per node");
        assert_eq!(plan.count_fallbacks(), 2, "all ops should be CpuFallback");
        assert_eq!(plan.count_launches(), 0, "no kernel launches");
        assert_eq!(plan.count_transfers(), 0, "no transfers");
        assert_eq!(plan.count_syncs(), 0, "no syncs");

        // All assignments should be CpuScalar.
        for (_, target) in &plan.assignment {
            assert_eq!(*target, ComputeTarget::CpuScalar);
        }
    }

    // -----------------------------------------------------------------------
    // Test 7: CPU-only plan has correct estimated cycles
    // -----------------------------------------------------------------------

    #[test]
    fn test_cpu_only_plan_estimated_cycles() {
        let (graph, recs) = gpu_graph();
        let plan = generate_cpu_only_plan(&graph, &recs);

        // Node 0: CPU cost 20 cycles, Node 1: CPU cost 500 cycles.
        assert_eq!(plan.estimated_total_cycles, 520,
            "CPU-only plan should sum CPU costs: 20 + 500 = 520");
    }

    // -----------------------------------------------------------------------
    // Test 8: verify_dispatch_plan with explicit plan validates correctly
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_dispatch_plan_explicit() {
        let (graph, recs) = gpu_graph();
        let plan = generate_dispatch_plan(&graph, &recs);

        let pipeline = Pipeline::new(PipelineConfig {
            verify_dispatch: DispatchVerifyMode::ErrorOnFailure,
            ..Default::default()
        });

        let result = pipeline.verify_dispatch_plan(&graph, plan, &recs);
        assert!(result.is_ok(), "Explicitly generated plan should pass");
    }

    // -----------------------------------------------------------------------
    // Test 9: Empty graph produces valid empty plan in all modes
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_graph_all_modes() {
        let graph = ComputeGraph::new();
        let recs = vec![];

        for mode in [
            DispatchVerifyMode::Off,
            DispatchVerifyMode::FallbackOnFailure,
            DispatchVerifyMode::ErrorOnFailure,
        ] {
            let pipeline = Pipeline::new(PipelineConfig {
                verify_dispatch: mode,
                ..Default::default()
            });

            let result = pipeline.generate_and_verify_dispatch(&graph, &recs);
            assert!(result.is_ok(), "Empty graph should pass in mode {:?}", mode);
            let plan = result.unwrap();
            assert!(plan.is_empty(), "Empty graph should produce empty plan");
        }
    }

    // -----------------------------------------------------------------------
    // Test 10: Default config uses FallbackOnFailure
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_config_uses_fallback_mode() {
        let config = PipelineConfig::default();
        assert_eq!(
            config.verify_dispatch,
            DispatchVerifyMode::FallbackOnFailure,
            "Default should be FallbackOnFailure"
        );
    }

    // -----------------------------------------------------------------------
    // Test 11: CPU-only fallback plan passes verification itself
    // -----------------------------------------------------------------------

    #[test]
    fn test_cpu_only_fallback_passes_verification() {
        let (graph, recs) = gpu_graph();
        let cpu_plan = generate_cpu_only_plan(&graph, &recs);

        let report = verify_dispatch_plan_properties(&graph, &cpu_plan);
        assert!(report.all_ok(),
            "CPU-only fallback plan should pass all verification properties:\n{}",
            report);
    }

    // -----------------------------------------------------------------------
    // Test 12: DispatchVerificationFailed error has meaningful Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_dispatch_verification_error_display() {
        let (graph, recs) = bad_fallback_graph();

        let pipeline = Pipeline::new(PipelineConfig {
            verify_dispatch: DispatchVerifyMode::ErrorOnFailure,
            ..Default::default()
        });

        let err = pipeline.generate_and_verify_dispatch(&graph, &recs).unwrap_err();
        let display = format!("{}", err);
        assert!(display.contains("dispatch verification failed"),
            "Error display should contain 'dispatch verification failed', got: {}",
            display);
        assert!(display.contains("violation"),
            "Error display should contain 'violation', got: {}",
            display);
    }
}
