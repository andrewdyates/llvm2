// llvm2-lower/isel.rs - AArch64 instruction selection (Phase 1: tMIR -> MachIR)
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// *** This is instruction selection, not machine code emission. ***
// Despite naming similarity, this module is unrelated to `llvm2-codegen/lower.rs`.
// The two modules operate at opposite ends of the compilation pipeline:
//
//   llvm2-lower/isel.rs   (Phase 1): tMIR SSA IR  -> AArch64 MachIR (VRegs)
//   llvm2-codegen/lower.rs (Phase 8): AArch64 MachIR (PRegs) -> binary bytes
//
// This module matches tMIR opcodes to AArch64 instructions with virtual registers.
// `lower.rs` encodes already-allocated physical-register instructions into bytes.
// There is no code overlap between them.
//
// Reference: LLVM AArch64ISelLowering.cpp (tree-pattern matching + late combines)
// Reference: designs/2026-04-12-aarch64-backend.md (two-phase ISel architecture)

//! AArch64 instruction selection: tMIR SSA IR -> AArch64 MachIR with virtual registers.
//!
//! Phase 1 (this module): Walk tMIR blocks in reverse postorder, match each
//! instruction bottom-up, emit AArch64 ISelInst with VRegs. This covers
//! arithmetic, comparisons, branches, calls, returns, loads, stores, and
//! constants.
//!
//! The ISel types (`ISelFunction`, `ISelInst`, `ISelBlock`, `ISelOperand`) are
//! ISel-specific intermediates, distinct from the canonical `llvm2_ir::MachFunction`
//! / `MachInst` / `MachBlock` / `MachOperand` types. The pipeline adapter
//! (`isel_to_ir` in `llvm2-codegen/pipeline.rs`) converts between them.
//!
//! Phase 2 (llvm2-opt late combines): Address-mode formation, cmp+branch
//! fusing, csel/cset formation. Those depend on one-use analysis not available
//! during tree matching.

use std::collections::{HashMap, HashSet};

use crate::abi::{AppleAArch64ABI, ArgLocation, HfaBaseType, PReg, gpr};
use crate::function::{Signature, StackSlotInfo};
use crate::instructions::{AtomicOrdering, AtomicRmwOp, Block, FloatCC, Instruction, IntCC, Opcode, Value};
use crate::overflow_idiom::{detect_overflow_idioms, OverflowAnalysis, OverflowKind};
use crate::smulh_idiom::{detect_smulh_idioms, SmulhAnalysis};
use crate::types::Type;
use thiserror::Error;

// Import canonical register types from llvm2-ir.
use llvm2_ir::TlsModel;
use llvm2_ir::regs::{FP, RegClass, SP, VReg, WZR, XZR};

// ---------------------------------------------------------------------------
// Register model helpers
// ---------------------------------------------------------------------------

/// Derive the register class for a given LIR type.
fn reg_class_for_type(ty: &Type) -> RegClass {
    match ty {
        Type::B1 | Type::I8 | Type::I16 | Type::I32 => RegClass::Gpr32,
        Type::I64 | Type::I128 => RegClass::Gpr64,
        Type::F32 => RegClass::Fpr32,
        Type::F64 => RegClass::Fpr64,
        Type::V128 => RegClass::Fpr128,
        // Aggregates are handled via pointers at the machine level.
        Type::Struct(_) | Type::Array(_, _) => RegClass::Gpr64,
    }
}

/// 16-bit system register encoding for TPIDR_EL0.
const TPIDR_EL0_SYSREG: i64 = 0xDE82;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors during instruction selection.
#[derive(Debug, Error)]
pub enum ISelError {
    #[error("value {0:?} not defined before use")]
    UndefinedValue(Value),
    #[error("unsupported type for Fconst: {0:?} (expected F32 or F64)")]
    UnsupportedFconstType(Type),
    #[error("unsupported ABI location for return value: stack-passed returns are not supported")]
    UnsupportedReturnLocation,
    #[error(
        "Return arity mismatch: function signature declares {expected} return value(s) but Return instruction has {actual} arg(s); every non-void function must pass its return value(s) in Return.args"
    )]
    ReturnArityMismatch { expected: usize, actual: usize },
    #[error(
        "Return type mismatch at index {index}: signature declares {expected} but Return arg has type {actual}"
    )]
    ReturnTypeMismatch {
        index: usize,
        expected: String,
        actual: String,
    },
    #[error("unsupported ABI location for argument: RegSequence on non-aggregate type")]
    UnsupportedArgLocation,
    #[error(
        "malformed instruction: expected at least {expected} args, got {actual} (opcode context: {context})"
    )]
    InsufficientArgs {
        expected: usize,
        actual: usize,
        context: &'static str,
    },
    #[error("malformed instruction: expected at least 1 result (opcode context: {0})")]
    MissingResult(&'static str),
    #[error("StructGep on non-struct type: {0:?}")]
    StructGepNonStruct(Type),
    #[error("StructGep field index {index} out of range for struct with {field_count} fields")]
    StructGepOutOfRange { index: u32, field_count: usize },
    #[error("ArrayGep with zero-sized element type: {0:?}")]
    ArrayGepZeroSized(Type),
    #[error("aggregate type too large for inline return: {0} bytes")]
    AggregateReturnTooLarge(u32),
    #[error("LocalExec TlsRef requires local_exec_offset")]
    LocalExecTlsRefMissingOffset,
    #[error("LocalExec TlsRef offset too large for imm12 hi/lo pair")]
    LocalExecTlsRefOffsetTooLarge,
    #[error("internal error: block not found in ISel function")]
    BlockNotFound,
}

// ---------------------------------------------------------------------------
// AArch64 opcode — re-exported from llvm2-ir (unified, issue #73)
// ---------------------------------------------------------------------------
//
// ISel now uses the canonical `llvm2_ir::AArch64Opcode` directly instead of
// its own per-width opcode enum. Width information (32-bit vs 64-bit) is
// conveyed by the register class of the operands, not the opcode name.
//
// The pipeline no longer needs `map_isel_opcode()` to translate between
// ISel opcodes and IR opcodes — they are the same type.
// ---------------------------------------------------------------------------

pub use llvm2_ir::AArch64Opcode;

// ---------------------------------------------------------------------------
// Machine operand
// ---------------------------------------------------------------------------

/// Operand of an ISel-output machine instruction (post-isel, pre-regalloc).
///
/// Separate from `llvm2_ir::MachOperand`: includes ISel-specific variants
/// (CondCode, Symbol) not present in the canonical IR, and uses `Block`
/// (LIR block ID) instead of `BlockId` (MachIR block ID).
///
/// Named `ISelOperand` (issue #73) to avoid confusion with the canonical
/// `llvm2_ir::MachOperand`. The pipeline adapter (`isel_to_ir`) converts
/// these to `llvm2_ir::MachOperand`.
#[derive(Debug, Clone, PartialEq)]
pub enum ISelOperand {
    /// Virtual register.
    VReg(VReg),
    /// Physical register (for ABI constraints, e.g. X0 for return).
    PReg(PReg),
    /// Immediate integer.
    Imm(i64),
    /// Immediate float.
    FImm(f64),
    /// Basic block target (for branches).
    Block(Block),
    /// Condition code (for B.cond, CSET).
    CondCode(AArch64CC),
    /// Global symbol name (for ADRP/ADD relocations).
    Symbol(String),
    /// Stack slot index (resolved during frame lowering).
    StackSlot(u32),
    /// Jump table data for switch lowering (legacy, pre-registration).
    /// Contains the minimum case value and a vector of target blocks indexed
    /// from 0, where `targets[i]` is the block for case `min_val + i`.
    /// Holes (values without explicit cases) map to the default block.
    ///
    /// During `to_ir_func()` conversion, any remaining `JumpTable` operand
    /// is registered on `MachFunction::jump_tables` on the fly; but the
    /// preferred form is `JumpTableIndex(u32)` pointing into an already-
    /// registered table on `ISelFunction::jump_tables`.
    JumpTable { min_val: i64, targets: Vec<Block> },
    /// Jump table reference. Indexes into
    /// [`ISelFunction::jump_tables`](ISelFunction::jump_tables). Emitted by
    /// `switch::emit_jump_table`. Converted to
    /// [`llvm2_ir::MachOperand::JumpTableIndex`] during `to_ir_func()`.
    JumpTableIndex(u32),
    /// Incoming stack argument offset from the caller's SP. Resolved by
    /// frame lowering to `[FP, #callee_saved_area_size + offset]`.
    ///
    /// Converted to `llvm2_ir::MachOperand::IncomingArg(i64)` during
    /// `to_ir_func()`.
    IncomingArg(i64),
}

/// Backward-compatible alias (deprecated). Use `ISelOperand` directly.
pub type MachOperand = ISelOperand;

/// AArch64 condition codes (NZCV-based) for ISel output.
///
/// Separate from `llvm2_ir::cc::AArch64CC` (which aliases `CondCode`):
/// this version includes `from_intcc`/`from_floatcc` conversion methods
/// for tMIR comparison conditions. The canonical `CondCode` in llvm2-ir
/// is the hardware-level encoding without tMIR conversion logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AArch64CC {
    EQ, // Equal (Z=1)
    NE, // Not equal (Z=0)
    HS, // Unsigned higher or same (C=1) (aka CS)
    LO, // Unsigned lower (C=0) (aka CC)
    MI, // Negative (N=1)
    PL, // Positive or zero (N=0)
    VS, // Overflow (V=1)
    VC, // No overflow (V=0)
    HI, // Unsigned higher (C=1 & Z=0)
    LS, // Unsigned lower or same (C=0 | Z=1)
    GE, // Signed greater or equal (N=V)
    LT, // Signed less than (N!=V)
    GT, // Signed greater than (Z=0 & N=V)
    LE, // Signed less or equal (Z=1 | N!=V)
}

impl AArch64CC {
    /// Map tMIR integer comparison condition to AArch64 condition code.
    pub fn from_intcc(cc: IntCC) -> Self {
        match cc {
            IntCC::Equal => AArch64CC::EQ,
            IntCC::NotEqual => AArch64CC::NE,
            IntCC::SignedLessThan => AArch64CC::LT,
            IntCC::SignedGreaterThanOrEqual => AArch64CC::GE,
            IntCC::SignedGreaterThan => AArch64CC::GT,
            IntCC::SignedLessThanOrEqual => AArch64CC::LE,
            IntCC::UnsignedLessThan => AArch64CC::LO,
            IntCC::UnsignedGreaterThanOrEqual => AArch64CC::HS,
            IntCC::UnsignedGreaterThan => AArch64CC::HI,
            IntCC::UnsignedLessThanOrEqual => AArch64CC::LS,
        }
    }

    /// Map tMIR floating-point comparison condition to AArch64 condition code.
    ///
    /// AArch64 FCMP sets NZCV as follows:
    ///   Equal:     NZCV = 0110 (Z=1, C=1)
    ///   LessThan:  NZCV = 1000 (N=1)
    ///   GreaterThan: NZCV = 0010 (C=1)
    ///   Unordered (NaN): NZCV = 0011 (C=1, V=1)
    ///
    /// Ordered predicates return false for NaN. Unordered predicates return
    /// true for NaN, implemented by inverting the complementary ordered CC:
    ///   UnorderedNotEqual = !Equal => NE  (NaN: Z=0, NE true)
    ///   UnorderedLessThan = !GreaterThanOrEqual => LT  (NaN: N!=V, LT true)
    ///   UnorderedLessThanOrEqual = !GreaterThan => LE  (NaN: !(Z=0&&N=V), LE true)
    ///   UnorderedGreaterThan = !LessThanOrEqual => HI  (NaN: C=1&&Z=0, HI true)
    ///   UnorderedGreaterThanOrEqual = !LessThan => PL  (NaN: N=0, PL true)
    ///
    /// Note: UnorderedEqual (EQ||VS) cannot be a single CC. The ISel emits
    /// a CSET+CSINC sequence for this case (see `select_fcmp`).
    /// The CC returned here (VS) is only used as the second condition in that
    /// sequence — the first condition (EQ) is handled directly by select_fcmp.
    pub fn from_floatcc(cc: FloatCC) -> Self {
        match cc {
            // Ordered (false for NaN)
            FloatCC::Equal => AArch64CC::EQ,
            FloatCC::NotEqual => AArch64CC::NE,
            FloatCC::LessThan => AArch64CC::MI,
            FloatCC::LessThanOrEqual => AArch64CC::LS,
            FloatCC::GreaterThan => AArch64CC::GT,
            FloatCC::GreaterThanOrEqual => AArch64CC::GE,
            FloatCC::Ordered => AArch64CC::VC,
            FloatCC::Unordered => AArch64CC::VS,
            // Unordered (true for NaN) — invert complementary ordered CC
            FloatCC::UnorderedNotEqual => AArch64CC::NE, // !OEQ
            FloatCC::UnorderedLessThan => AArch64CC::LT, // !OGE
            FloatCC::UnorderedLessThanOrEqual => AArch64CC::LE, // !OGT
            FloatCC::UnorderedGreaterThan => AArch64CC::HI, // !OLE
            FloatCC::UnorderedGreaterThanOrEqual => AArch64CC::PL, // !OLT
            // UnorderedEqual needs two CCs (EQ || VS); select_fcmp handles
            // this specially. Return VS here as the secondary condition.
            FloatCC::UnorderedEqual => AArch64CC::VS,
        }
    }

    /// Return the inverted condition code.
    pub fn invert(self) -> Self {
        match self {
            AArch64CC::EQ => AArch64CC::NE,
            AArch64CC::NE => AArch64CC::EQ,
            AArch64CC::HS => AArch64CC::LO,
            AArch64CC::LO => AArch64CC::HS,
            AArch64CC::MI => AArch64CC::PL,
            AArch64CC::PL => AArch64CC::MI,
            AArch64CC::VS => AArch64CC::VC,
            AArch64CC::VC => AArch64CC::VS,
            AArch64CC::HI => AArch64CC::LS,
            AArch64CC::LS => AArch64CC::HI,
            AArch64CC::GE => AArch64CC::LT,
            AArch64CC::LT => AArch64CC::GE,
            AArch64CC::GT => AArch64CC::LE,
            AArch64CC::LE => AArch64CC::GT,
        }
    }
}

// ---------------------------------------------------------------------------
// ISel-level machine instruction
// ---------------------------------------------------------------------------

/// A single ISel-output machine instruction (pre-regalloc).
///
/// Simpler than the canonical `llvm2_ir::MachInst`: no flags, no implicit
/// defs/uses, no proof annotations. Those are added during the ISel-to-IR
/// translation in the pipeline adapter.
///
/// Named `ISelInst` (issue #73) to avoid confusion with `llvm2_ir::MachInst`.
#[derive(Debug, Clone)]
pub struct ISelInst {
    pub opcode: AArch64Opcode,
    pub operands: Vec<ISelOperand>,
}

impl ISelInst {
    pub fn new(opcode: AArch64Opcode, operands: Vec<ISelOperand>) -> Self {
        Self { opcode, operands }
    }
}

/// Backward-compatible alias (deprecated). Use `ISelInst` directly.
pub type MachInst = ISelInst;

// ---------------------------------------------------------------------------
// ISel-level machine basic block
// ---------------------------------------------------------------------------

/// An ISel-output basic block of machine instructions.
///
/// Uses `Vec<ISelInst>` inline (not arena-indexed), and `successors` (not
/// `succs`/`preds`). The canonical `llvm2_ir::MachBlock` uses arena-indexed
/// `Vec<InstId>` and explicit predecessor tracking.
///
/// Named `ISelBlock` (issue #73) to avoid confusion with `llvm2_ir::MachBlock`.
#[derive(Debug, Clone, Default)]
pub struct ISelBlock {
    pub insts: Vec<ISelInst>,
    pub successors: Vec<Block>,
    /// Source locations parallel to `insts` for DWARF line number propagation.
    ///
    /// Each entry corresponds to the same-index instruction in `insts`.
    /// Populated during ISel from tMIR source spans carried on LIR instructions.
    pub source_locs: Vec<Option<llvm2_ir::SourceLoc>>,
    /// Proof annotations parallel to `insts`, copied onto `MachInst.proof`
    /// during `to_ir_func()`.
    ///
    /// Each entry corresponds to the same-index instruction in `insts`.
    /// Populated during ISel for instructions whose semantics carry a
    /// statically-provable property the rest of the pipeline can consume
    /// (e.g. `Bl` calls whose callee was tagged `ProofAnnotation::Pure` in
    /// the tMIR source — SROA partial-escape, #456).
    ///
    /// Shorter than `insts` is tolerated by `to_ir_func`: missing entries
    /// are treated as `None`.
    pub proofs: Vec<Option<llvm2_ir::ProofAnnotation>>,
}

/// Backward-compatible alias (deprecated). Use `ISelBlock` directly.
pub type MachBlock = ISelBlock;

// ---------------------------------------------------------------------------
// ISel-level machine function
// ---------------------------------------------------------------------------

/// An ISel-output function containing ISelInsts with VRegs.
///
/// Uses `HashMap<Block, ISelBlock>` for blocks (convenient for ISel
/// construction), while the canonical `llvm2_ir::MachFunction` uses
/// `Vec<MachBlock>` indexed by `BlockId` (cache-friendly for later passes).
///
/// Named `ISelFunction` (issue #73) to avoid confusion with `llvm2_ir::MachFunction`,
/// which is the canonical machine function type for the pipeline.
#[derive(Debug, Clone)]
pub struct ISelFunction {
    pub name: String,
    pub sig: Signature,
    pub blocks: HashMap<Block, ISelBlock>,
    pub block_order: Vec<Block>,
    pub next_vreg: u32,
    /// Stack slot metadata from the adapter.
    ///
    /// Indexed by slot number (matching `ISelOperand::StackSlot(n)`).
    /// Propagated to `MachFunction::stack_slots` via `to_ir_func()`.
    pub stack_slots: Vec<StackSlotInfo>,
    /// Jump tables for switch lowering.
    ///
    /// Each entry is registered by `switch::emit_jump_table` when a dense
    /// switch is lowered. The ADR instruction references the table via
    /// `ISelOperand::JumpTableIndex(idx)`, and the codegen pipeline patches
    /// the final PC-relative byte offset.
    pub jump_tables: Vec<ISelJumpTable>,
}

/// Jump table data carried on an [`ISelFunction`]. Mirrors
/// [`llvm2_ir::function::JumpTableData`] but uses ISel-level [`Block`] ids
/// instead of [`llvm2_ir::types::BlockId`]. Converted in `to_ir_func()`.
#[derive(Debug, Clone)]
pub struct ISelJumpTable {
    pub min_val: i64,
    pub targets: Vec<Block>,
}

/// Backward-compatible alias (deprecated). Use `ISelFunction` directly.
pub type MachFunction = ISelFunction;

impl ISelFunction {
    pub fn new(name: String, sig: Signature) -> Self {
        Self {
            name,
            sig,
            blocks: HashMap::new(),
            block_order: Vec::new(),
            next_vreg: 0,
            stack_slots: Vec::new(),
            jump_tables: Vec::new(),
        }
    }

    /// Register a jump table and return its index. The returned index is
    /// stored in `ISelOperand::JumpTableIndex` on the `Adr` instruction
    /// that loads the table base.
    pub fn add_jump_table(&mut self, min_val: i64, targets: Vec<Block>) -> u32 {
        let idx = self.jump_tables.len() as u32;
        self.jump_tables.push(ISelJumpTable { min_val, targets });
        idx
    }

    /// Emit a machine instruction into the given block.
    pub fn push_inst(&mut self, block: Block, inst: ISelInst) {
        let blk = self.blocks.entry(block).or_default();
        blk.insts.push(inst);
        blk.source_locs.push(None);
        blk.proofs.push(None);
    }

    /// Emit a machine instruction with an associated source location.
    pub fn push_inst_with_loc(
        &mut self,
        block: Block,
        inst: ISelInst,
        loc: Option<llvm2_ir::SourceLoc>,
    ) {
        let blk = self.blocks.entry(block).or_default();
        blk.insts.push(inst);
        blk.source_locs.push(loc);
        blk.proofs.push(None);
    }

    /// Emit a machine instruction with an associated source location AND
    /// proof annotation. Used by call lowering to stamp `ProofAnnotation::Pure`
    /// onto the emitted `Bl` when the callee is known pure (#456).
    pub fn push_inst_with_proof(
        &mut self,
        block: Block,
        inst: ISelInst,
        loc: Option<llvm2_ir::SourceLoc>,
        proof: Option<llvm2_ir::ProofAnnotation>,
    ) {
        let blk = self.blocks.entry(block).or_default();
        blk.insts.push(inst);
        blk.source_locs.push(loc);
        blk.proofs.push(proof);
    }

    /// Add a block to the function (if not already present).
    pub fn ensure_block(&mut self, block: Block) {
        if let std::collections::hash_map::Entry::Vacant(e) = self.blocks.entry(block) {
            e.insert(ISelBlock::default());
            self.block_order.push(block);
        }
    }

    /// Convert this ISel function to the canonical `llvm2_ir::MachFunction`.
    ///
    /// This is the Phase 1->2 adapter (previously `isel_to_ir()` in pipeline.rs).
    /// The ISel uses HashMap-based blocks and ISel-specific operand types.
    /// The IR uses Vec-based arena storage and canonical types.
    ///
    /// Successor edges are copied directly; predecessor edges are computed from
    /// successors after construction.
    pub fn to_ir_func(&self) -> llvm2_ir::MachFunction {
        use llvm2_ir::function::{JumpTableData, MachBlock as IrBlock, MachFunction as IrFunc};
        use llvm2_ir::inst::MachInst as IrInst;
        use llvm2_ir::types::BlockId;

        let ir_sig: llvm2_ir::function::Signature = (&self.sig).into();
        let mut ir_func = IrFunc::new(self.name.clone(), ir_sig);
        ir_func.next_vreg = self.next_vreg;

        // Clear the default entry block that MachFunction::new creates.
        ir_func.blocks.clear();
        ir_func.block_order.clear();

        // Create all blocks first.
        for &block_ref in &self.block_order {
            let block_id = BlockId(block_ref.0);
            while ir_func.blocks.len() <= block_id.0 as usize {
                ir_func.blocks.push(IrBlock::new());
            }
            ir_func.block_order.push(block_id);
        }
        ir_func.entry = if self.block_order.is_empty() {
            BlockId(0)
        } else {
            BlockId(self.block_order[0].0)
        };

        // Propagate jump tables from ISel to MachIR. Each ISel jump table
        // translates directly to an `IrOp::JumpTableIndex(idx)` reference
        // on the `Adr` instruction (handled in convert_isel_operand_to_ir).
        for isel_jt in &self.jump_tables {
            ir_func.jump_tables.push(JumpTableData {
                min_val: isel_jt.min_val,
                targets: isel_jt.targets.iter().map(|b| BlockId(b.0)).collect(),
            });
        }

        // Convert instructions block by block and copy successor edges.
        // For legacy `ISelOperand::JumpTable { .. }` operands (pre-registration
        // path used by unit tests), we register them on the fly here.
        for &block_ref in &self.block_order {
            let block_id = BlockId(block_ref.0);
            if let Some(isel_block) = self.blocks.get(&block_ref) {
                for (idx, isel_inst) in isel_block.insts.iter().enumerate() {
                    let ir_operands: Vec<llvm2_ir::MachOperand> = isel_inst
                        .operands
                        .iter()
                        .map(|op| {
                            if let ISelOperand::JumpTable { min_val, targets } = op {
                                // On-the-fly registration for legacy callers
                                // that did not go through `add_jump_table`.
                                let new_idx = ir_func.jump_tables.len() as u32;
                                ir_func.jump_tables.push(JumpTableData {
                                    min_val: *min_val,
                                    targets: targets.iter().map(|b| BlockId(b.0)).collect(),
                                });
                                llvm2_ir::MachOperand::JumpTableIndex(new_idx)
                            } else {
                                convert_isel_operand_to_ir(op)
                            }
                        })
                        .collect();
                    let mut ir_inst = IrInst::new(isel_inst.opcode, ir_operands);
                    // Propagate source location from ISel to MachInst for DWARF.
                    if let Some(loc) = isel_block.source_locs.get(idx).copied().flatten() {
                        ir_inst.source_loc = Some(loc);
                    }
                    // Propagate proof annotation from ISel to MachInst. SROA
                    // partial-escape (#456) reads `Bl.proof == Some(Pure)` to
                    // keep call-arg slot LDRs scalarisable.
                    if let Some(proof) = isel_block.proofs.get(idx).copied().flatten() {
                        ir_inst.proof = Some(proof);
                    }
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
        for &block_ref in &self.block_order {
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

        // Propagate stack slot metadata from the ISel function to the IR function.
        for slot_info in &self.stack_slots {
            ir_func.alloc_stack_slot(llvm2_ir::function::StackSlot::new(
                slot_info.size,
                slot_info.align,
            ));
        }

        ir_func
    }
}

/// Convert an ISel `ISelOperand` to a canonical `llvm2_ir::MachOperand`.
///
/// This is the operand-level conversion used by `ISelFunction::to_ir_func()`.
/// Condition codes are encoded as 4-bit immediates per ARM ARM.
/// Symbols are preserved as `MachOperand::Symbol` for relocation emission.
pub fn convert_isel_operand_to_ir(op: &ISelOperand) -> llvm2_ir::MachOperand {
    use llvm2_ir::MachOperand as IrOp;
    use llvm2_ir::types::BlockId;

    match op {
        ISelOperand::VReg(v) => IrOp::VReg(*v),
        ISelOperand::PReg(p) => IrOp::PReg(*p),
        ISelOperand::Imm(v) => IrOp::Imm(*v),
        ISelOperand::FImm(v) => IrOp::FImm(*v),
        ISelOperand::Block(b) => IrOp::Block(BlockId(b.0)),
        ISelOperand::CondCode(cc) => {
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
            IrOp::Imm(encoding)
        }
        ISelOperand::Symbol(name) => IrOp::Symbol(name.clone()),
        ISelOperand::StackSlot(idx) => IrOp::StackSlot(llvm2_ir::types::StackSlotId(*idx)),
        ISelOperand::JumpTable { .. } => {
            // Legacy path: when `JumpTable` reaches this standalone
            // converter (not via to_ir_func), we have no access to the
            // MachFunction side-table to register the data. Emit Imm(0)
            // as a placeholder. New code should use
            // `ISelFunction::add_jump_table` + `JumpTableIndex` instead.
            IrOp::Imm(0)
        }
        ISelOperand::JumpTableIndex(idx) => IrOp::JumpTableIndex(*idx),
        ISelOperand::IncomingArg(off) => IrOp::IncomingArg(*off),
    }
}

// ---------------------------------------------------------------------------
// Instruction selector
// ---------------------------------------------------------------------------

/// AArch64 instruction selector.
///
/// Walks tMIR blocks in order, selects each instruction into one or more
/// AArch64 ISelInsts, tracking value -> VReg mappings. After selection,
/// `finalize()` returns the completed ISelFunction.
pub struct InstructionSelector {
    func: ISelFunction,
    /// tMIR Value -> machine operand mapping.
    value_map: HashMap<Value, ISelOperand>,
    /// Type of each value, tracked for selecting correct instruction width.
    value_types: HashMap<Value, Type>,
    /// High-half register for i128 values.
    ///
    /// When a Value has type I128, `value_map` contains the low 64-bit half
    /// and this map contains the high 64-bit half. Both are `RegClass::Gpr64`.
    i128_high_map: HashMap<Value, VReg>,
    /// Next available block ID for synthesized intermediate blocks.
    ///
    /// Used by binary search tree switch lowering to allocate fresh blocks
    /// for BST internal nodes. Updated when `ensure_block` or `select_block`
    /// is called to stay above all existing block IDs.
    next_block_id: u32,
    /// Current source location being propagated to emitted instructions.
    ///
    /// Set before each instruction selection from the LIR instruction's
    /// source_loc. All ISelInsts emitted during that selection inherit this
    /// location for DWARF line number program generation.
    current_source_loc: Option<llvm2_ir::SourceLoc>,
    /// Per-block signed-overflow idiom analysis (see `overflow_idiom.rs`).
    ///
    /// Populated at the start of `select_block_with_source_locs`. Consulted by
    /// `select_instruction` to (a) skip idiom intermediates, (b) emit the
    /// flag-setting `AddsRR`/`SubsRR` in place of the plain `Add`/`Sub` for
    /// the narrow arithmetic op, and (c) fuse the overflow-branch into a
    /// single `B.VS`. Issue #430.
    overflow_analysis: OverflowAnalysis,
    /// Per-block signed multiply-high idiom analysis (see `smulh_idiom.rs`).
    ///
    /// Populated at the start of `select_block_with_source_locs`. Consulted by
    /// `select_instruction` to (a) skip the widened i128 intermediates and
    /// (b) replace the final `Trunc(I128 -> I64)` with a single `SMULH`.
    /// Issue #429.
    pub(crate) smulh_analysis: SmulhAnalysis,
    /// Current instruction index inside the block being selected.
    ///
    /// Parallel to `overflow_analysis.skip_indices` and
    /// `smulh_analysis.skip_indices`; updated by
    /// `select_block_with_source_locs` on each iteration so that
    /// `select_instruction` can check whether the current op is part of a
    /// recognised idiom.
    current_inst_idx: usize,
    /// Values whose current meaning is "the V flag is set iff overflow
    /// occurred in the most recent `AddsRR`/`SubsRR`."
    ///
    /// Populated when an overflow idiom's narrow arithmetic is emitted as
    /// flag-setting. Cleared once the flag is consumed (by Brif fuse, or by
    /// a CSET fallback materialising the bool into a vreg). Issue #430.
    pending_v_flag: std::collections::HashSet<Value>,
    /// Set of direct-call callee names known to be pure (from tMIR
    /// `ProofAnnotation::Pure`, surfaced via `Function::pure_callees`).
    ///
    /// When `select_call` emits `AArch64Opcode::Bl` and the `callee_name` is
    /// in this set, the Bl is stamped with `Some(ProofAnnotation::Pure)` so
    /// SROA partial-escape (#456) can treat the call-arg slot address as
    /// non-escaping for read elimination.
    pure_callees: HashSet<String>,
}

impl InstructionSelector {
    /// Create a new instruction selector for the given function.
    pub fn new(name: String, sig: Signature) -> Self {
        Self {
            func: ISelFunction::new(name, sig),
            value_map: HashMap::new(),
            value_types: HashMap::new(),
            i128_high_map: HashMap::new(),
            next_block_id: 0,
            current_source_loc: None,
            overflow_analysis: OverflowAnalysis::default(),
            smulh_analysis: SmulhAnalysis::default(),
            current_inst_idx: 0,
            pending_v_flag: std::collections::HashSet::new(),
            pure_callees: HashSet::new(),
        }
    }

    /// Set the stack slot metadata from the LIR function.
    ///
    /// This must be called before `finalize()` so that `to_ir_func()` can
    /// propagate the slots to the canonical `MachFunction::stack_slots`.
    pub fn set_stack_slots(&mut self, slots: Vec<StackSlotInfo>) {
        self.func.stack_slots = slots;
    }

    /// Seed the `value_types` map with type hints from the adapter.
    ///
    /// Used to convey types of `Value`s whose producing opcode does not
    /// carry enough information for the selector to infer on its own
    /// (e.g., `Opcode::Call` result values — the callee's return type is
    /// only known from the tMIR function signature, which the adapter
    /// preserves in `Function::value_types`). See #381.
    ///
    /// Call this before `select_block*` so that `value_type()` lookups
    /// during selection return the correct type instead of the I64
    /// fallback.
    pub fn seed_value_types(&mut self, types: &HashMap<Value, Type>) {
        for (val, ty) in types {
            self.value_types.insert(*val, ty.clone());
        }
    }

    /// Seed the set of direct-call callee names known to be pure.
    ///
    /// Populated from `Function::pure_callees` (in turn set by the tMIR
    /// adapter when the source function carried `ProofAnnotation::Pure`).
    /// When `select_call` emits `Bl` with a symbol in this set, the `Bl`
    /// carries `Some(ProofAnnotation::Pure)` through to the canonical
    /// `MachInst.proof` so SROA can apply partial-escape (#456).
    pub fn seed_pure_callees(&mut self, names: &HashSet<String>) {
        for name in names {
            self.pure_callees.insert(name.clone());
        }
    }

    /// Emit an ISelInst into a block, attaching the current source location.
    ///
    /// This is the primary emission method used by select_instruction and its
    /// helpers. It stamps the instruction with `current_source_loc` which was
    /// set by `select_block_with_source_locs` from the LIR instruction's span.
    fn emit(&mut self, block: Block, inst: ISelInst) {
        self.func
            .push_inst_with_loc(block, inst, self.current_source_loc);
    }

    /// Emit a Bl ISelInst carrying a proof annotation (call lowering #456).
    ///
    /// Uses the same source-location stamping as `emit`, plus threads the
    /// proof through to the parallel `ISelBlock.proofs` vector. The only
    /// current caller is `select_call` when the callee is known pure.
    fn emit_with_proof(
        &mut self,
        block: Block,
        inst: ISelInst,
        proof: Option<llvm2_ir::ProofAnnotation>,
    ) {
        self.func
            .push_inst_with_proof(block, inst, self.current_source_loc, proof);
    }

    /// Require that `inst` has at least `n` arguments.
    fn require_args(inst: &Instruction, n: usize, context: &'static str) -> Result<(), ISelError> {
        if inst.args.len() < n {
            return Err(ISelError::InsufficientArgs {
                expected: n,
                actual: inst.args.len(),
                context,
            });
        }
        Ok(())
    }

    /// Require that `inst` has at least one result.
    fn require_result(inst: &Instruction, context: &'static str) -> Result<(), ISelError> {
        if inst.results.is_empty() {
            return Err(ISelError::MissingResult(context));
        }
        Ok(())
    }

    /// Allocate a fresh virtual register.
    fn new_vreg(&mut self, class: RegClass) -> VReg {
        let id = self.func.next_vreg;
        self.func.next_vreg += 1;
        VReg { id, class }
    }

    /// Record a mapping from tMIR Value to machine operand.
    fn define_value(&mut self, val: Value, operand: ISelOperand, ty: Type) {
        self.value_map.insert(val, operand);
        self.value_types.insert(val, ty);
    }

    /// Look up the machine operand for a tMIR Value.
    fn use_value(&self, val: &Value) -> Result<ISelOperand, ISelError> {
        self.value_map
            .get(val)
            .cloned()
            .ok_or(ISelError::UndefinedValue(*val))
    }

    /// Get the type of a tMIR Value.
    fn value_type(&self, val: &Value) -> Type {
        self.value_types.get(val).cloned().unwrap_or(Type::I64) // default to i64 if unknown
    }

    /// Determine if a type is 32-bit (uses W registers).
    fn is_32bit(ty: &Type) -> bool {
        matches!(ty, Type::B1 | Type::I8 | Type::I16 | Type::I32 | Type::F32)
    }

    // -----------------------------------------------------------------------
    // Top-level selection
    // -----------------------------------------------------------------------

    /// Select all instructions in a block.
    pub fn select_block(
        &mut self,
        block: Block,
        instructions: &[Instruction],
    ) -> Result<(), ISelError> {
        self.select_block_with_source_locs(block, instructions, &[])
    }

    /// Select all instructions in a block with associated source locations.
    ///
    /// `source_locs` is parallel to `instructions`: `source_locs[i]` is the
    /// source location for `instructions[i]`. If shorter than `instructions`,
    /// missing entries are treated as None.
    ///
    /// Source locations are propagated to the emitted ISelInsts for eventual
    /// DWARF line number program generation via `to_ir_func()`.
    pub fn select_block_with_source_locs(
        &mut self,
        block: Block,
        instructions: &[Instruction],
        source_locs: &[Option<llvm2_ir::SourceLoc>],
    ) -> Result<(), ISelError> {
        self.func.ensure_block(block);
        // Keep next_block_id above all known blocks.
        if block.0 >= self.next_block_id {
            self.next_block_id = block.0 + 1;
        }

        // Pre-scan this block for the canonical i128-widened signed-overflow
        // idiom (issue #430). Detection is SSA-exact and in-block-only; see
        // `overflow_idiom.rs` for the pattern. When a match fires,
        // `select_instruction` skips the idiom's i128 intermediates and
        // substitutes the flag-setting ADDS/SUBS for the narrow arithmetic.
        self.overflow_analysis = detect_overflow_idioms(instructions);
        // Pre-scan this block for the canonical i128-widened signed
        // multiply-high idiom (issue #429). Detection is SSA-exact and
        // in-block-only; see `smulh_idiom.rs`. When a match fires,
        // `select_instruction` skips the widened intermediates and replaces
        // the final `Trunc(I128 -> I64)` with a single `SMULH`.
        self.smulh_analysis = detect_smulh_idioms(instructions);
        // Clear any stale pending V-flag state from a previous block.
        self.pending_v_flag.clear();

        for (i, inst) in instructions.iter().enumerate() {
            self.current_source_loc = source_locs.get(i).copied().flatten();
            self.current_inst_idx = i;
            // Skip idiom intermediates entirely — their LIR results are never
            // referenced by non-idiom ops (checked during detection), so no
            // vreg needs to be defined.
            if self.overflow_analysis.is_skipped(i) || self.smulh_analysis.is_skipped(i) {
                continue;
            }
            self.select_instruction(inst, block)?;
        }
        // Drop analysis state so it isn't reused across blocks.
        self.overflow_analysis = OverflowAnalysis::default();
        self.smulh_analysis = SmulhAnalysis::default();
        self.pending_v_flag.clear();
        self.current_source_loc = None;
        Ok(())
    }

    /// Select a single tMIR instruction, emitting ISelInsts into the block.
    fn select_instruction(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        match &inst.opcode {
            // Constants
            Opcode::Iconst {
                ty: Type::I128,
                imm,
            } => self.select_i128_iconst(*imm, inst, block)?,
            Opcode::Iconst { ty, imm } => self.select_iconst(ty.clone(), *imm, inst, block)?,
            Opcode::Fconst { ty, imm } => self.select_fconst(ty.clone(), *imm, inst, block)?,

            // Copy pseudo — register-to-register move (block-arg passing,
            // tMIR Copy lowering, borrow lowering). See #417.
            Opcode::Copy => self.select_copy(inst, block)?,
            // Arithmetic
            // i128 multi-register arithmetic: intercept before normal binop dispatch.
            Opcode::Iadd
                if !inst.args.is_empty() && self.value_type(&inst.args[0]) == Type::I128 =>
            {
                self.select_i128_add(inst, block)?
            }
            // Guard against regressions: single-arg Iadd used to be the COPY
            // placeholder; after #417 it must not appear in well-formed LIR.
            // Use Opcode::Copy for moves. Zero-arg falls through to select_binop
            // which returns a proper InsufficientArgs error.
            Opcode::Iadd if inst.args.len() == 1 => {
                return Err(ISelError::InsufficientArgs {
                    expected: 2,
                    actual: 1,
                    context: "Iadd (use Opcode::Copy for single-arg moves; see #417)",
                });
            }
            Opcode::Isub
                if !inst.args.is_empty() && self.value_type(&inst.args[0]) == Type::I128 =>
            {
                self.select_i128_sub(inst, block)?
            }
            Opcode::Imul
                if !inst.args.is_empty() && self.value_type(&inst.args[0]) == Type::I128 =>
            {
                self.select_i128_mul(inst, block)?
            }
            Opcode::Sdiv
                if !inst.args.is_empty() && self.value_type(&inst.args[0]) == Type::I128 =>
            {
                self.select_i128_sdiv(inst, block)?
            }
            Opcode::Udiv
                if !inst.args.is_empty() && self.value_type(&inst.args[0]) == Type::I128 =>
            {
                self.select_i128_udiv(inst, block)?
            }
            // i128 signed-overflow idiom fast path (#430): narrow ADD/SUB that
            // feeds an i128-widened overflow check must be emitted with the
            // flag-setting variant (AddsRR/SubsRR) so a subsequent B.VS /
            // CSET VS can read the V flag. See `overflow_idiom.rs`.
            Opcode::Iadd
                if !inst.results.is_empty()
                    && self.overflow_analysis.sum_idiom(&inst.results[0]).is_some() =>
            {
                self.select_overflow_narrow(OverflowKind::SignedAdd, inst, block)?
            }
            Opcode::Isub
                if !inst.results.is_empty()
                    && self.overflow_analysis.sum_idiom(&inst.results[0]).is_some() =>
            {
                self.select_overflow_narrow(OverflowKind::SignedSub, inst, block)?
            }
            Opcode::Iadd => self.select_binop(AArch64BinOp::Add, inst, block)?,
            Opcode::Isub => self.select_binop(AArch64BinOp::Sub, inst, block)?,
            Opcode::Imul => self.select_binop(AArch64BinOp::Mul, inst, block)?,
            Opcode::Sdiv => self.select_binop(AArch64BinOp::Sdiv, inst, block)?,
            Opcode::Udiv => self.select_binop(AArch64BinOp::Udiv, inst, block)?,
            Opcode::Srem => self.select_remainder(/*signed=*/ true, inst, block)?,
            Opcode::Urem => self.select_remainder(/*signed=*/ false, inst, block)?,

            // Checked signed arithmetic (#474): lower directly to the
            // AArch64 native flag-setting idiom. See
            // `select_checked_sadd_ssub` / `select_checked_smul`.
            Opcode::CheckedSadd => self.select_checked_sadd_ssub(CheckedArith::Sadd, inst, block)?,
            Opcode::CheckedSsub => self.select_checked_sadd_ssub(CheckedArith::Ssub, inst, block)?,
            Opcode::CheckedSmul => self.select_checked_smul(inst, block)?,

            // Unary operations
            Opcode::Ineg => self.select_int_unaryop(AArch64IntUnaryOp::Neg, inst, block)?,
            Opcode::Bnot => self.select_int_unaryop(AArch64IntUnaryOp::Mvn, inst, block)?,
            Opcode::Fneg => self.select_fp_unaryop(AArch64Opcode::FnegRR, inst, block)?,
            Opcode::Fabs => self.select_fp_unaryop(AArch64Opcode::FabsRR, inst, block)?,
            Opcode::Fsqrt => self.select_fp_unaryop(AArch64Opcode::FsqrtRR, inst, block)?,

            // Shift operations
            // i128 multi-register shifts: intercept before normal shift dispatch
            Opcode::Ishl
                if !inst.args.is_empty() && self.value_type(&inst.args[0]) == Type::I128 =>
            {
                self.select_i128_shl(inst, block)?
            }
            Opcode::Ushr
                if !inst.args.is_empty() && self.value_type(&inst.args[0]) == Type::I128 =>
            {
                self.select_i128_lshr(inst, block)?
            }
            Opcode::Sshr
                if !inst.args.is_empty() && self.value_type(&inst.args[0]) == Type::I128 =>
            {
                self.select_i128_ashr(inst, block)?
            }
            Opcode::Ishl => self.select_shift(AArch64ShiftOp::Lsl, inst, block)?,
            Opcode::Ushr => self.select_shift(AArch64ShiftOp::Lsr, inst, block)?,
            Opcode::Sshr => self.select_shift(AArch64ShiftOp::Asr, inst, block)?,

            // Logical operations
            Opcode::Band => self.select_logic(AArch64LogicOp::And, inst, block)?,
            Opcode::Bor => self.select_logic(AArch64LogicOp::Orr, inst, block)?,
            Opcode::Bxor => self.select_logic(AArch64LogicOp::Eor, inst, block)?,
            Opcode::BandNot => self.select_logic(AArch64LogicOp::Bic, inst, block)?,
            Opcode::BorNot => self.select_logic(AArch64LogicOp::Orn, inst, block)?,

            // Extensions
            Opcode::Sextend { from_ty, to_ty } => {
                self.select_extend(true, from_ty, to_ty, inst, block)?;
            }
            Opcode::Uextend { from_ty, to_ty } => {
                self.select_extend(false, from_ty, to_ty, inst, block)?;
            }

            // Bitfield operations
            Opcode::ExtractBits { lsb, width } => {
                self.select_bitfield_extract(false, *lsb, *width, inst, block)?;
            }
            Opcode::SextractBits { lsb, width } => {
                self.select_bitfield_extract(true, *lsb, *width, inst, block)?;
            }
            Opcode::InsertBits { lsb, width } => {
                self.select_bitfield_insert(*lsb, *width, inst, block)?;
            }

            // Conditional select
            Opcode::Select { cond } => self.select_csel(*cond, inst, block)?,

            // Comparison
            Opcode::Icmp { cond }
                if !inst.args.is_empty() && self.value_type(&inst.args[0]) == Type::I128 =>
            {
                self.select_i128_cmp(*cond, inst, block)?
            }
            Opcode::Icmp { cond } => self.select_cmp(*cond, inst, block)?,

            // Floating-point arithmetic
            Opcode::Fadd => self.select_fp_binop(AArch64FpBinOp::Fadd, inst, block)?,
            Opcode::Fsub => self.select_fp_binop(AArch64FpBinOp::Fsub, inst, block)?,
            Opcode::Fmul => self.select_fp_binop(AArch64FpBinOp::Fmul, inst, block)?,
            Opcode::Fdiv => self.select_fp_binop(AArch64FpBinOp::Fdiv, inst, block)?,
            Opcode::Fcmp { cond } => self.select_fcmp(*cond, inst, block)?,
            Opcode::FcvtToInt { dst_ty } => {
                self.select_fcvt_to_int(dst_ty.clone(), inst, block)?;
            }
            Opcode::FcvtFromInt { src_ty } => {
                self.select_fcvt_from_int(src_ty.clone(), inst, block)?;
            }

            // Addressing
            Opcode::GlobalRef { name } => {
                self.select_global_ref(name, inst, block)?;
            }
            Opcode::ExternRef { name } => {
                self.select_extern_ref(name, inst, block)?;
            }
            Opcode::TlsRef {
                name,
                model,
                local_exec_offset,
            } => {
                self.select_tls_ref(name, *model, *local_exec_offset, inst, block)?;
            }
            Opcode::StackAddr { slot } => {
                self.select_stack_addr(*slot, inst, block)?;
            }

            // Control flow
            Opcode::Jump { dest } => self.select_jump(*dest, block)?,
            Opcode::Brif {
                then_dest,
                else_dest,
                ..
            } => {
                self.select_brif(inst, *then_dest, *else_dest, block)?;
            }
            Opcode::Return => self.select_return(inst, block)?,
            Opcode::Call { name } => {
                self.select_call_from_lir(name, inst, block)?;
            }
            Opcode::CallIndirect => {
                self.select_call_indirect_from_lir(inst, block)?;
            }
            Opcode::CallVariadic { name, fixed_args } => {
                self.select_variadic_call_from_lir(name, *fixed_args, inst, block)?;
            }
            Opcode::Switch { cases, default } => {
                self.select_switch(cases, *default, inst, block)?;
            }
            Opcode::Invoke {
                name,
                normal_dest,
                unwind_dest,
            } => {
                self.select_invoke(name, *normal_dest, *unwind_dest, inst, block)?;
            }
            Opcode::LandingPad {
                is_cleanup,
                catch_type_indices,
            } => {
                self.select_landing_pad(*is_cleanup, catch_type_indices, inst, block)?;
            }
            Opcode::Resume => {
                self.select_resume(inst, block)?;
            }

            // Type conversions (unsigned FP <-> int)
            Opcode::FcvtToUint { dst_ty } => {
                self.select_fcvt_to_uint(dst_ty.clone(), inst, block)?;
            }
            Opcode::FcvtFromUint { src_ty } => {
                self.select_fcvt_from_uint(src_ty.clone(), inst, block)?;
            }
            // Float precision conversion
            Opcode::FPExt => self.select_fp_ext(inst, block)?,
            Opcode::FPTrunc => self.select_fp_trunc(inst, block)?,
            // Integer truncation and bitcast
            // SMULH fast path (#429): when the Trunc closes the canonical
            // i128-widened signed-multiply-high idiom, emit a single SMULH
            // instead of the full widened MUL+UMULH+MADD+MADD sequence.
            Opcode::Trunc { to_ty: Type::I64 }
                if self.smulh_analysis.smulh_for_trunc(self.current_inst_idx).is_some() =>
            {
                self.select_smulh_idiom(inst, block)?
            }
            Opcode::Trunc { to_ty } => self.select_trunc(&to_ty.clone(), inst, block)?,
            Opcode::Bitcast { to_ty } => self.select_bitcast(&to_ty.clone(), inst, block)?,

            // Memory
            Opcode::Load { ty } => self.select_load(ty.clone(), inst, block)?,
            Opcode::Store => self.select_store(inst, block)?,

            // Atomic memory operations
            Opcode::AtomicLoad { ty, ordering } => {
                self.select_atomic_load(ty.clone(), *ordering, inst, block)?;
            }
            Opcode::AtomicStore { ordering } => {
                self.select_atomic_store(*ordering, inst, block)?;
            }
            Opcode::AtomicRmw { op, ty, ordering } => {
                self.select_atomic_rmw(*op, ty.clone(), *ordering, inst, block)?;
            }
            Opcode::CmpXchg { ty, ordering } => {
                self.select_cmpxchg(ty.clone(), *ordering, inst, block)?;
            }
            Opcode::Fence { ordering } => {
                self.select_fence(*ordering, block)?;
            }

            // Aggregate operations
            Opcode::StructGep {
                struct_ty,
                field_index,
            } => {
                self.select_struct_gep(struct_ty.clone(), *field_index, inst, block)?;
            }
            Opcode::ArrayGep { elem_ty } => {
                self.select_array_gep(elem_ty.clone(), inst, block)?;
            }

            // Memory intrinsics
            Opcode::Memcpy => self.select_memcpy(inst, block)?,
            Opcode::Memmove => self.select_memmove(inst, block)?,
            Opcode::Memset => self.select_memset(inst, block)?,
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    /// Select integer constant materialization.
    /// Small values (0..65535): MOVZ
    /// Negative small values: MOVN
    /// Large values: MOVZ + MOVK sequence (up to 4 chunks for 64-bit)
    fn select_iconst(
        &mut self,
        ty: Type,
        imm: i64,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_result(inst, "Iconst")?;
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);
        let result = &inst.results[0];

        if (0..=0xFFFF).contains(&imm) {
            // Simple MOVZ
            let opc = if Self::is_32bit(&ty) {
                AArch64Opcode::Movz
            } else {
                AArch64Opcode::Movz
            };
            self.emit(
                block,
                ISelInst::new(opc, vec![ISelOperand::VReg(dst), ISelOperand::Imm(imm)]),
            );
        } else if (-0x10000..0).contains(&imm) {
            // MOVN for small negative values
            let opc = if Self::is_32bit(&ty) {
                AArch64Opcode::Movn
            } else {
                AArch64Opcode::Movn
            };
            // MOVN encodes ~(imm16 << shift), so for -1 we encode MOVN Xd, #0
            let encoded = (!imm) as u64 & 0xFFFF;
            self.emit(
                block,
                ISelInst::new(
                    opc,
                    vec![ISelOperand::VReg(dst), ISelOperand::Imm(encoded as i64)],
                ),
            );
        } else {
            // Large immediate: MOVZ + MOVK sequence
            // Start with lowest 16 bits via MOVZ
            let opc_z = if Self::is_32bit(&ty) {
                AArch64Opcode::Movz
            } else {
                AArch64Opcode::Movz
            };
            let low16 = (imm as u64) & 0xFFFF;
            self.emit(
                block,
                ISelInst::new(
                    opc_z,
                    vec![ISelOperand::VReg(dst), ISelOperand::Imm(low16 as i64)],
                ),
            );

            // Insert remaining 16-bit chunks via MOVK
            let chunks = if Self::is_32bit(&ty) { 2 } else { 4 };
            for shift in 1..chunks {
                let chunk = ((imm as u64) >> (shift * 16)) & 0xFFFF;
                if chunk != 0 {
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::Movk,
                            vec![
                                ISelOperand::VReg(dst),
                                ISelOperand::Imm(chunk as i64),
                                ISelOperand::Imm(shift as i64 * 16), // shift amount
                            ],
                        ),
                    );
                }
            }
        }

        self.define_value(*result, ISelOperand::VReg(dst), ty);
        Ok(())
    }

    /// Select float constant materialization.
    fn select_fconst(
        &mut self,
        ty: Type,
        imm: f64,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_result(inst, "Fconst")?;
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);
        let result = &inst.results[0];

        let opc = match ty {
            Type::F32 => AArch64Opcode::FmovImm,
            Type::F64 => AArch64Opcode::FmovImm,
            _ => return Err(ISelError::UnsupportedFconstType(ty)),
        };

        // FMOV immediate encoding is limited to a small set of values.
        // For the scaffold, emit FMOV for all; legalization will handle
        // values outside the 8-bit FP immediate range via constant pool.
        self.emit(
            block,
            ISelInst::new(opc, vec![ISelOperand::VReg(dst), ISelOperand::FImm(imm)]),
        );

        self.define_value(*result, ISelOperand::VReg(dst), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Copy (single-arg Iadd used as COPY by adapter)
    // -----------------------------------------------------------------------

    /// Select a COPY instruction (emitted for single-arg Iadd from the adapter).
    ///
    /// The adapter uses `Iadd` with a single argument as a placeholder for
    /// register copies (block argument passing, phi resolution). We emit a
    /// MOV (register-to-register copy) instruction.
    fn select_copy(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "Copy")?;
        Self::require_result(inst, "Copy")?;

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let ty = self.value_type(&src_val);
        let src = self.use_value(&src_val)?;

        // If the result value already has a vreg (e.g., a block parameter
        // that was defined by an earlier predecessor's copy), reuse it.
        // This ensures all predecessors write to the same vreg for block
        // parameters, which is essential for correct register allocation
        // across loop back-edges.
        let dst = if let Some(existing) = self.value_map.get(&result_val) {
            if let ISelOperand::VReg(v) = existing {
                *v
            } else {
                let class = reg_class_for_type(&ty);
                self.new_vreg(class)
            }
        } else {
            let class = reg_class_for_type(&ty);
            self.new_vreg(class)
        };

        let opc = if Self::is_32bit(&ty) {
            AArch64Opcode::MovR
        } else {
            AArch64Opcode::MovR
        };

        self.emit(block, ISelInst::new(opc, vec![ISelOperand::VReg(dst), src]));

        self.define_value(result_val, ISelOperand::VReg(dst), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    /// Internal enum for selecting the right opcode variant.
    #[allow(dead_code)]
    fn select_binop(
        &mut self,
        op: AArch64BinOp,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "BinOp")?;
        Self::require_result(inst, "BinOp")?;

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let result_val = inst.results[0];

        // Determine type from LHS
        let ty = self.value_type(&lhs_val);
        let is_32 = Self::is_32bit(&ty);

        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        let opc = match (op, is_32) {
            (AArch64BinOp::Add, true) => AArch64Opcode::AddRR,
            (AArch64BinOp::Add, false) => AArch64Opcode::AddRR,
            (AArch64BinOp::Sub, true) => AArch64Opcode::SubRR,
            (AArch64BinOp::Sub, false) => AArch64Opcode::SubRR,
            (AArch64BinOp::Mul, true) => AArch64Opcode::MulRR,
            (AArch64BinOp::Mul, false) => AArch64Opcode::MulRR,
            (AArch64BinOp::Sdiv, true) => AArch64Opcode::SDiv,
            (AArch64BinOp::Sdiv, false) => AArch64Opcode::SDiv,
            (AArch64BinOp::Udiv, true) => AArch64Opcode::UDiv,
            (AArch64BinOp::Udiv, false) => AArch64Opcode::UDiv,
        };

        self.emit(
            block,
            ISelInst::new(opc, vec![ISelOperand::VReg(dst), lhs, rhs]),
        );

        self.define_value(result_val, ISelOperand::VReg(dst), ty);
        Ok(())
    }

    /// Select integer remainder: SDIV/UDIV + MSUB.
    ///
    /// AArch64 has no hardware remainder instruction. We compute:
    ///   tmp = a / b         (SDIV or UDIV)
    ///   result = a - tmp*b  (MSUB: Rd = Ra - Rn * Rm)
    ///
    /// MSUB operand order: MSUB Rd, Rn, Rm, Ra  =>  Rd = Ra - Rn * Rm
    /// So: MSUB result, tmp, b, a  =>  result = a - tmp * b
    fn select_remainder(
        &mut self,
        signed: bool,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "Remainder")?;
        Self::require_result(inst, "Remainder")?;

        let lhs_val = inst.args[0]; // dividend (a)
        let rhs_val = inst.args[1]; // divisor (b)
        let result_val = inst.results[0];

        let ty = self.value_type(&lhs_val);
        let is_32 = Self::is_32bit(&ty);
        let class = reg_class_for_type(&ty);

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        // Step 1: tmp = a / b
        let tmp = self.new_vreg(class);
        let div_opc = match (signed, is_32) {
            (true, true) => AArch64Opcode::SDiv,
            (true, false) => AArch64Opcode::SDiv,
            (false, true) => AArch64Opcode::UDiv,
            (false, false) => AArch64Opcode::UDiv,
        };
        self.emit(
            block,
            ISelInst::new(
                div_opc,
                vec![ISelOperand::VReg(tmp), lhs.clone(), rhs.clone()],
            ),
        );

        // Step 2: result = a - tmp * b  (MSUB Rd, Rn, Rm, Ra)
        // Operands: [dst, Rn=tmp, Rm=b, Ra=a]
        let dst = self.new_vreg(class);
        let msub_opc = if is_32 {
            AArch64Opcode::Msub
        } else {
            AArch64Opcode::Msub
        };
        self.emit(
            block,
            ISelInst::new(
                msub_opc,
                vec![ISelOperand::VReg(dst), ISelOperand::VReg(tmp), rhs, lhs],
            ),
        );

        self.define_value(result_val, ISelOperand::VReg(dst), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Unary operations
    // -----------------------------------------------------------------------

    /// Select integer unary operation: NEG, MVN (bitwise NOT).
    ///
    /// AArch64 lowering:
    ///   `Ineg(x)` → `SUB Xd, XZR, Xn` (alias: NEG)
    ///   `Bnot(x)` → `ORN Xd, XZR, Xn` (alias: MVN)
    fn select_int_unaryop(
        &mut self,
        op: AArch64IntUnaryOp,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "IntUnaryOp")?;
        Self::require_result(inst, "IntUnaryOp")?;

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let ty = self.value_type(&src_val);
        let is_32 = Self::is_32bit(&ty);
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let src = self.use_value(&src_val)?;

        match op {
            AArch64IntUnaryOp::Neg => {
                // NEG Rd, Rm — encoder hardcodes XZR for Rn, 2-operand form.
                self.emit(
                    block,
                    ISelInst::new(AArch64Opcode::Neg, vec![ISelOperand::VReg(dst), src]),
                );
            }
            AArch64IntUnaryOp::Mvn => {
                // MVN Rd, Rm = ORN Rd, XZR, Rm — 3-operand form required by encoder.
                // Use XZR for 64-bit, WZR for 32-bit to match sf determination.
                let zr = if is_32 {
                    ISelOperand::PReg(WZR)
                } else {
                    ISelOperand::PReg(XZR)
                };
                self.emit(
                    block,
                    ISelInst::new(AArch64Opcode::OrnRR, vec![ISelOperand::VReg(dst), zr, src]),
                );
            }
        }

        self.define_value(result_val, ISelOperand::VReg(dst), ty);
        Ok(())
    }

    /// Select floating-point unary operation: FNEG/FABS/FSQRT Sd/Dd, Sn/Dn.
    ///
    /// The `opc` parameter selects the concrete AArch64 opcode (FnegRR, FabsRR,
    /// FsqrtRR). All three share the same operand shape: one FPR source, one
    /// FPR destination, precision determined by register class.
    fn select_fp_unaryop(
        &mut self,
        opc: AArch64Opcode,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "FpUnaryOp")?;
        Self::require_result(inst, "FpUnaryOp")?;

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let ty = self.value_type(&src_val);
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let src = self.use_value(&src_val)?;

        self.emit(block, ISelInst::new(opc, vec![ISelOperand::VReg(dst), src]));

        self.define_value(result_val, ISelOperand::VReg(dst), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Comparison
    // -----------------------------------------------------------------------

    /// Select integer comparison: CMP + CSET.
    ///
    /// tMIR `Icmp(cond, lhs, rhs) -> bool_result` becomes:
    ///   CMP Wn/Xn, Wm/Xm        (sets NZCV flags)
    ///   CSET Wd, <cond_code>     (materializes flag into register)
    fn select_cmp(
        &mut self,
        cond: IntCC,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "Icmp")?;
        Self::require_result(inst, "Icmp")?;

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let result_val = inst.results[0];

        let ty = self.value_type(&lhs_val);
        let is_32 = Self::is_32bit(&ty);

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        // CMP (subtract setting flags, discard result)
        let cmp_opc = if is_32 {
            AArch64Opcode::CmpRR
        } else {
            AArch64Opcode::CmpRR
        };
        self.emit(block, ISelInst::new(cmp_opc, vec![lhs, rhs]));

        // CSET: materialize condition code into a register.
        // We use Gpr64 for the destination to avoid register aliasing issues
        // with the linear scan allocator, which treats W<n> and X<n> as
        // independent physical registers despite them sharing hardware.
        // CSET writes a 32-bit result (0 or 1) that zero-extends to 64 bits,
        // so using an X register is semantically correct.
        let cc = AArch64CC::from_intcc(cond);
        let dst = self.new_vreg(RegClass::Gpr64);
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::CSet,
                vec![ISelOperand::VReg(dst), ISelOperand::CondCode(cc)],
            ),
        );

        self.define_value(result_val, ISelOperand::VReg(dst), Type::B1);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Control flow
    // -----------------------------------------------------------------------

    /// Select unconditional jump.
    fn select_jump(&mut self, dest: Block, block: Block) -> Result<(), ISelError> {
        self.emit(
            block,
            ISelInst::new(AArch64Opcode::B, vec![ISelOperand::Block(dest)]),
        );
        self.func
            .blocks
            .entry(block)
            .or_default()
            .successors
            .push(dest);
        Ok(())
    }

    /// Select conditional branch: tMIR `Brif(cond, then, else)`.
    ///
    /// The condition value should already be a comparison result. We emit:
    ///   CMP Wcond, #0          (test boolean value)
    ///   B.NE then_block        (branch if true)
    ///   B else_block           (fallthrough to else)
    fn select_brif(
        &mut self,
        inst: &Instruction,
        then_dest: Block,
        else_dest: Block,
        block: Block,
    ) -> Result<(), ISelError> {
        // The condition is the first argument
        let cond_val = inst.args[0];

        // i128 signed-overflow idiom fast path (#430): if the condition is
        // the overflow boolean of a recognised idiom and the V flag is still
        // live from the preceding ADDS/SUBS, emit B.VS directly (no CMP, no
        // CSET).
        if self.pending_v_flag.contains(&cond_val) {
            // Consume the pending flag.
            self.pending_v_flag.remove(&cond_val);

            // B.VS then_dest — overflow occurred.
            self.emit(
                block,
                ISelInst::new(
                    AArch64Opcode::BCond,
                    vec![
                        ISelOperand::CondCode(AArch64CC::VS),
                        ISelOperand::Block(then_dest),
                    ],
                ),
            );
            // B else_dest — unconditional fall-through.
            self.emit(
                block,
                ISelInst::new(AArch64Opcode::B, vec![ISelOperand::Block(else_dest)]),
            );
            let mblock = self.func.blocks.entry(block).or_default();
            mblock.successors.push(then_dest);
            mblock.successors.push(else_dest);
            return Ok(());
        }

        let cond_op = self.use_value(&cond_val)?;

        // CMP cond, #0 to set NZCV from boolean.
        // Use 64-bit CMP to match the Gpr64 class of the CSET result.
        self.emit(
            block,
            ISelInst::new(AArch64Opcode::CmpRI, vec![cond_op, ISelOperand::Imm(0)]),
        );

        // B.NE then_block (condition was nonzero = true)
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::BCond,
                vec![
                    ISelOperand::CondCode(AArch64CC::NE),
                    ISelOperand::Block(then_dest),
                ],
            ),
        );

        // B else_block (unconditional fallthrough)
        self.emit(
            block,
            ISelInst::new(AArch64Opcode::B, vec![ISelOperand::Block(else_dest)]),
        );

        let mblock = self.func.blocks.entry(block).or_default();
        mblock.successors.push(then_dest);
        mblock.successors.push(else_dest);
        Ok(())
    }

    /// Select the narrow (I64) arithmetic half of a detected signed-overflow
    /// idiom. Emits the flag-setting variant (`AddsRR`/`SubsRR`) so that a
    /// subsequent `B.VS` or `CSET VS` can read the V flag.
    ///
    /// The idiom detection (see [`crate::overflow_idiom`]) guarantees that no
    /// flag-clobbering instruction fires between this emission and the
    /// overflow consumer: all idiom intermediates are marked for skipping in
    /// `select_block_with_source_locs`. Issue #430.
    fn select_overflow_narrow(
        &mut self,
        kind: OverflowKind,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "overflow narrow arith")?;
        Self::require_result(inst, "overflow narrow arith")?;

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let result_val = inst.results[0];

        let ty = self.value_type(&lhs_val);
        // Idiom is defined only on I64; guard anyway.
        if ty != Type::I64 {
            // Fallback to plain binop — the idiom guard should never fire
            // here because detection requires I64 inputs, but we stay
            // defensive.
            return match kind {
                OverflowKind::SignedAdd => self.select_binop(AArch64BinOp::Add, inst, block),
                OverflowKind::SignedSub => self.select_binop(AArch64BinOp::Sub, inst, block),
            };
        }

        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        let opc = match kind {
            OverflowKind::SignedAdd => AArch64Opcode::AddsRR,
            OverflowKind::SignedSub => AArch64Opcode::SubsRR,
        };

        self.emit(
            block,
            ISelInst::new(opc, vec![ISelOperand::VReg(dst), lhs, rhs]),
        );

        self.define_value(result_val, ISelOperand::VReg(dst), ty);

        // Dispatch the paired overflow Value: either defer it to the
        // terminator Brif (fast path, B.VS) or materialise it here via CSET.
        if let Some(idiom) = self.overflow_analysis.sum_idiom(&result_val).cloned() {
            if idiom.needs_cset_fallback {
                // CSET dst_b1, VS — materialise V flag into a bool vreg.
                // Must be emitted immediately after the flag-setter so no
                // other instruction clobbers NZCV between them. All idiom
                // intermediates are skip-marked, so we are safe.
                let bool_dst = self.new_vreg(RegClass::Gpr64);
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::CSet,
                        vec![
                            ISelOperand::VReg(bool_dst),
                            ISelOperand::CondCode(AArch64CC::VS),
                        ],
                    ),
                );
                self.define_value(idiom.overflow, ISelOperand::VReg(bool_dst), Type::B1);
            } else {
                // Fast path: the block terminator is Brif(overflow, ...).
                // Defer consumption to `select_brif`, which will emit B.VS.
                self.pending_v_flag.insert(idiom.overflow);
            }
        }
        Ok(())
    }

    /// Select the final `Trunc(I128 -> I64)` of a detected SMULH idiom.
    ///
    /// The detector (see [`crate::smulh_idiom`]) guarantees that the
    /// truncation's source is the high-half extraction of a signed widened
    /// multiply, so the full i128 chain collapses to a single
    /// `SMULH Xd, Xn, Xm`. Issue #429.
    fn select_smulh_idiom(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "SMULH idiom trunc")?;
        Self::require_result(inst, "SMULH idiom trunc")?;

        let idx = self.current_inst_idx;
        let Some(idiom) = self.smulh_analysis.by_trunc.get(&idx).copied() else {
            // Defensive: the dispatch guard promised this is Some. If not,
            // fall back to plain Trunc lowering rather than panicking.
            return self.select_trunc(&Type::I64, inst, block);
        };

        let a = self.use_value(&idiom.a)?;
        let b = self.use_value(&idiom.b)?;

        let dst = self.new_vreg(RegClass::Gpr64);
        self.emit(
            block,
            ISelInst::new(AArch64Opcode::Smulh, vec![ISelOperand::VReg(dst), a, b]),
        );

        self.define_value(inst.results[0], ISelOperand::VReg(dst), Type::I64);
        Ok(())
    }

    /// Select a `CheckedSadd` / `CheckedSsub` (signed add/sub with overflow,
    /// issue #474).
    ///
    /// Lowers to the canonical two-instruction AArch64 idiom:
    ///
    /// ```text
    ///     ADDS  Xd, Xn, Xm        ; or SUBS — sets NZCV, V = signed overflow
    ///     CSET  Xov, VS           ; Xov = (V == 1) ? 1 : 0
    /// ```
    ///
    /// `args`:    `[lhs, rhs]`
    /// `results`: `[value, overflow_b1]`
    ///
    /// Supported widths: I32 (W-register form) and I64 (X-register form). The
    /// AddsRR/SubsRR opcodes dispatch on operand register class. Narrower
    /// widths (I8, I16) still go through the adapter's bit-pattern fallback.
    ///
    /// No instruction may be emitted between the ADDS/SUBS and CSET that
    /// clobbers NZCV; both are emitted here back-to-back within the same
    /// block for that reason. See `select_overflow_narrow` for the analogous
    /// i128-widened idiom path.
    fn select_checked_sadd_ssub(
        &mut self,
        kind: CheckedArith,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "CheckedSadd/Ssub")?;
        if inst.results.len() < 2 {
            return Err(ISelError::InsufficientArgs {
                expected: 2,
                actual: inst.results.len(),
                context: "CheckedSadd/Ssub (need [value, overflow_b1] results)",
            });
        }

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let value_result = inst.results[0];
        let overflow_result = inst.results[1];

        let ty = self.value_type(&lhs_val);
        // The adapter currently only produces these opcodes for I64; guard
        // anyway so a malformed LIR surfaces immediately rather than silently
        // mislowering (e.g. emitting a 32-bit ADDS on an I16 pair).
        if !matches!(ty, Type::I32 | Type::I64) {
            return Err(ISelError::InsufficientArgs {
                expected: 64,
                actual: 0,
                context: "CheckedSadd/Ssub requires I32 or I64 operands",
            });
        }

        let class = reg_class_for_type(&ty);
        let value_dst = self.new_vreg(class);
        let overflow_dst = self.new_vreg(RegClass::Gpr64);

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        let arith_opc = match kind {
            CheckedArith::Sadd => AArch64Opcode::AddsRR,
            CheckedArith::Ssub => AArch64Opcode::SubsRR,
        };

        // ADDS/SUBS — sets NZCV. V = 1 iff signed overflow.
        self.emit(
            block,
            ISelInst::new(
                arith_opc,
                vec![ISelOperand::VReg(value_dst), lhs, rhs],
            ),
        );
        // CSET Xov, VS — materialise V into a bool vreg. Must be emitted
        // immediately after ADDS/SUBS with no flag-clobbering op between.
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::CSet,
                vec![
                    ISelOperand::VReg(overflow_dst),
                    ISelOperand::CondCode(AArch64CC::VS),
                ],
            ),
        );

        self.define_value(value_result, ISelOperand::VReg(value_dst), ty);
        self.define_value(overflow_result, ISelOperand::VReg(overflow_dst), Type::B1);
        Ok(())
    }

    /// Select a `CheckedSmul` (signed multiply with overflow, issue #474).
    ///
    /// For I64 operands, AArch64 has no flag-setting signed multiply. The
    /// canonical overflow-safe idiom uses the upper 64 bits of the signed
    /// 64x64 -> 128 product (SMULH, #429) and checks that the sign-extension
    /// of the low half matches:
    ///
    /// ```text
    ///     MUL   Xlo,  Xa, Xb                 ; low  64 bits of signed product
    ///     SMULH Xhi,  Xa, Xb                 ; high 64 bits of signed product
    ///     ASR   Xsign, Xlo, #63              ; sign-extension of Xlo
    ///     CMP   Xhi,   Xsign                 ; overflow iff Xhi != Xsign
    ///     CSET  Xov,   NE
    /// ```
    ///
    /// Correctness: the true signed product is representable in 64 bits iff
    /// its full 128-bit form equals the sign-extension of its low half. That
    /// is, the upper 64 bits must all be copies of bit 63 of the low half.
    ///
    /// Supported widths: I64 only (SMULH has no 32-bit variant). I32 and
    /// smaller fall back to the bit-pattern sequence in the adapter. The
    /// adapter never emits CheckedSmul for non-I64, but we guard defensively.
    fn select_checked_smul(
        &mut self,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "CheckedSmul")?;
        if inst.results.len() < 2 {
            return Err(ISelError::InsufficientArgs {
                expected: 2,
                actual: inst.results.len(),
                context: "CheckedSmul (need [value, overflow_b1] results)",
            });
        }

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let value_result = inst.results[0];
        let overflow_result = inst.results[1];

        let ty = self.value_type(&lhs_val);
        if ty != Type::I64 {
            return Err(ISelError::InsufficientArgs {
                expected: 64,
                actual: 0,
                context: "CheckedSmul requires I64 operands (SMULH is 64-bit only)",
            });
        }

        let value_dst = self.new_vreg(RegClass::Gpr64);
        let hi_dst = self.new_vreg(RegClass::Gpr64);
        let sign_dst = self.new_vreg(RegClass::Gpr64);
        let overflow_dst = self.new_vreg(RegClass::Gpr64);

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        // Xlo = MUL Xa, Xb — low 64 bits of signed product.
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::MulRR,
                vec![ISelOperand::VReg(value_dst), lhs.clone(), rhs.clone()],
            ),
        );
        // Xhi = SMULH Xa, Xb — high 64 bits of signed product.
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Smulh,
                vec![ISelOperand::VReg(hi_dst), lhs, rhs],
            ),
        );
        // Xsign = ASR Xlo, #63 — sign-extension of the low half.
        // AsrRI operand order: [dst, src, Imm(shift_amt)].
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::AsrRI,
                vec![
                    ISelOperand::VReg(sign_dst),
                    ISelOperand::VReg(value_dst),
                    ISelOperand::Imm(63),
                ],
            ),
        );
        // CMP Xhi, Xsign — sets NZCV. Z=1 iff Xhi == Xsign (no overflow).
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::CmpRR,
                vec![ISelOperand::VReg(hi_dst), ISelOperand::VReg(sign_dst)],
            ),
        );
        // CSET Xov, NE — Xov = (Xhi != Xsign) ? 1 : 0.
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::CSet,
                vec![
                    ISelOperand::VReg(overflow_dst),
                    ISelOperand::CondCode(AArch64CC::NE),
                ],
            ),
        );

        self.define_value(value_result, ISelOperand::VReg(value_dst), Type::I64);
        self.define_value(overflow_result, ISelOperand::VReg(overflow_dst), Type::B1);
        Ok(())
    }

    /// Select return. Moves return values into ABI-specified physical registers
    /// (X0/V0 etc.) and emits RET.
    fn select_return(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        // Validate arity against function signature. The bridge (tRust, tSwift, tC)
        // must populate `Return.args` with the function's return value(s); a void
        // function returns with `args: vec![]`. A silent empty args on a non-void
        // function would leave the return register (X0/V0) holding garbage. See #307.
        let sig_returns = self.func.sig.returns.clone();
        let expected = sig_returns.len();
        let actual = inst.args.len();
        if expected != actual {
            return Err(ISelError::ReturnArityMismatch { expected, actual });
        }

        for (i, arg_val) in inst.args.iter().enumerate() {
            let sig_ty = &sig_returns[i];
            let arg_ty = self.value_type(arg_val);
            let sig_class = reg_class_for_type(sig_ty);
            let arg_class = reg_class_for_type(&arg_ty);
            if &arg_ty != sig_ty || arg_class != sig_class {
                return Err(ISelError::ReturnTypeMismatch {
                    index: i,
                    expected: format!("{:?}", sig_ty),
                    actual: format!("{:?}", arg_ty),
                });
            }
        }

        // Classify return types to know which physical registers to use
        let ret_locs = AppleAArch64ABI::classify_returns(&sig_returns);

        // Move each return value into its designated physical register
        for (i, (val, loc)) in inst.args.iter().zip(ret_locs.iter()).enumerate() {
            let src = self.use_value(val)?;
            let ty = sig_returns[i].clone();

            // Aggregate types (structs/arrays) need special handling even when
            // the ABI says Reg: the source operand is a pointer to the aggregate
            // in memory, so we must load the value(s) into the target register(s).
            // The select_aggregate_return helper handles all size categories.
            if ty.is_aggregate() {
                self.select_aggregate_return(src, &ty, block)?;
                continue;
            }

            match loc {
                ArgLocation::Reg(preg) => {
                    // Use Copy pseudo for ALL register moves (GPR and FPR).
                    // lower_copies() dispatches Copy to MovR (GPR) or
                    // FmovFprFpr (FPR) based on destination register class.
                    // Using MovR directly for FP types is wrong — MovR encodes
                    // as ORR Xd, XZR, Xm which operates on GPR only.
                    self.emit(
                        block,
                        ISelInst::new(AArch64Opcode::Copy, vec![ISelOperand::PReg(*preg), src]),
                    );
                }
                ArgLocation::Indirect { .. } => {
                    // Large aggregate indirect return via sret pointer.
                    self.select_aggregate_return(src, &ty, block)?;
                }
                ArgLocation::RegPair(lo_preg, hi_preg) => {
                    // i128 register pair return: copy lo/hi halves to
                    // the designated physical register pair.
                    let (lo, hi) = self.use_i128_value(val)?;
                    self.emit(
                        block,
                        ISelInst::new(AArch64Opcode::Copy, vec![ISelOperand::PReg(*lo_preg), lo]),
                    );
                    self.emit(
                        block,
                        ISelInst::new(AArch64Opcode::Copy, vec![ISelOperand::PReg(*hi_preg), hi]),
                    );
                }
                ArgLocation::RegSequence(_) => {
                    // HFA return for non-aggregate scalar types should not occur.
                    // Aggregate HFA returns are dispatched above via select_aggregate_return.
                    return Err(ISelError::UnsupportedReturnLocation);
                }
                ArgLocation::Stack { .. } => {
                    return Err(ISelError::UnsupportedReturnLocation);
                }
            }
        }

        // Emit RET (branches to LR)
        self.emit(
            block,
            ISelInst::new(AArch64Opcode::Ret, vec![ISelOperand::PReg(gpr::LR)]),
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Call
    // -----------------------------------------------------------------------

    /// Select a function call.
    ///
    /// 1. Move arguments to physical registers (X0-X7, V0-V7) or stack
    /// 2. Emit BL (direct call)
    /// 3. Copy results from physical return registers to vregs
    pub fn select_call(
        &mut self,
        callee_name: &str,
        arg_vals: &[Value],
        result_vals: &[Value],
        result_types: &[Type],
        block: Block,
    ) -> Result<(), ISelError> {
        // Classify argument locations
        let arg_types: Vec<Type> = arg_vals.iter().map(|v| self.value_type(v)).collect();
        let arg_locs = AppleAArch64ABI::classify_params(&arg_types);

        // Move arguments to ABI locations
        for (i, (val, loc)) in arg_vals.iter().zip(arg_locs.iter()).enumerate() {
            let src = self.use_value(val)?;
            let ty = arg_types[i].clone();

            // Dispatch aggregate arguments to specialized handler
            if ty.is_aggregate() {
                self.select_aggregate_arg(src, &ty, loc, block)?;
                continue;
            }

            match loc {
                ArgLocation::Reg(preg) => {
                    // Use Copy pseudo for ALL register moves (GPR and FPR).
                    // lower_copies() dispatches Copy to MovR (GPR) or
                    // FmovFprFpr (FPR) based on destination register class.
                    self.emit(
                        block,
                        ISelInst::new(AArch64Opcode::Copy, vec![ISelOperand::PReg(*preg), src]),
                    );
                }
                ArgLocation::Stack { offset, size: _ } => {
                    // STR to [SP + offset]
                    // AArch64 StrRI handles all widths; register class determines size.
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::StrRI,
                            vec![src, ISelOperand::PReg(SP), ISelOperand::Imm(*offset)],
                        ),
                    );
                }
                ArgLocation::Indirect { ptr_reg } => {
                    // Large aggregate indirect: pass pointer in register
                    self.emit(
                        block,
                        ISelInst::new(AArch64Opcode::MovR, vec![ISelOperand::PReg(*ptr_reg), src]),
                    );
                }
                ArgLocation::RegPair(lo_preg, hi_preg) => {
                    // i128 register pair argument: copy both halves
                    let (lo, hi) = self.use_i128_value(val)?;
                    self.emit(
                        block,
                        ISelInst::new(AArch64Opcode::Copy, vec![ISelOperand::PReg(*lo_preg), lo]),
                    );
                    self.emit(
                        block,
                        ISelInst::new(AArch64Opcode::Copy, vec![ISelOperand::PReg(*hi_preg), hi]),
                    );
                }
                ArgLocation::RegSequence(_) => {
                    // HFA: aggregate types are dispatched above via select_aggregate_arg.
                    // Scalar types should never receive RegSequence classification.
                    return Err(ISelError::UnsupportedArgLocation);
                }
            }
        }

        // Emit BL (direct call) with the callee symbol for relocation.
        // If the callee was tagged `ProofAnnotation::Pure` in the tMIR source
        // (propagated via `Function::pure_callees` → `seed_pure_callees`),
        // stamp the Bl with `Some(Pure)` so SROA partial-escape (#456) can
        // see that the call does not escape its pointer arguments.
        let bl_proof = if self.pure_callees.contains(callee_name) {
            Some(llvm2_ir::ProofAnnotation::Pure)
        } else {
            None
        };
        self.emit_with_proof(
            block,
            ISelInst::new(
                AArch64Opcode::Bl,
                vec![ISelOperand::Symbol(callee_name.to_string())],
            ),
            bl_proof,
        );

        // Copy results from ABI return registers to vregs
        let ret_locs = AppleAArch64ABI::classify_returns(result_types);
        for (i, (val, loc)) in result_vals.iter().zip(ret_locs.iter()).enumerate() {
            match loc {
                ArgLocation::Reg(preg) => {
                    let ty = result_types[i].clone();
                    let class = reg_class_for_type(&ty);
                    let dst = self.new_vreg(class);
                    // Use Copy pseudo for ALL register moves (GPR and FPR).
                    // lower_copies() dispatches to MovR (GPR) or FmovFprFpr (FPR).
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::Copy,
                            vec![ISelOperand::VReg(dst), ISelOperand::PReg(*preg)],
                        ),
                    );
                    self.define_value(*val, ISelOperand::VReg(dst), ty);
                }
                ArgLocation::RegSequence(regs) => {
                    // HFA return: the callee placed each FP member in consecutive
                    // typed FPR registers. For the caller, the aggregate is
                    // reconstituted from these registers. We define the result
                    // as the pointer to a stack-allocated buffer where we store
                    // the HFA members.
                    let ty = result_types[i].clone();
                    let dst = self.new_vreg(RegClass::Gpr64);
                    // For scaffold: copy the first register as a representative.
                    // Full implementation stores each HFA member to memory.
                    if let Some(first) = regs.first() {
                        self.emit(
                            block,
                            ISelInst::new(
                                AArch64Opcode::MovR,
                                vec![ISelOperand::VReg(dst), ISelOperand::PReg(*first)],
                            ),
                        );
                    }
                    self.define_value(*val, ISelOperand::VReg(dst), ty);
                }
                ArgLocation::Indirect { ptr_reg } => {
                    // Aggregate returned via sret pointer (X8).
                    // The caller passed X8 as the sret buffer; after the call
                    // the data is at [X8]. Define the result as the sret
                    // pointer itself so downstream code can access fields.
                    let ty = result_types[i].clone();
                    let dst = self.new_vreg(RegClass::Gpr64);
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::MovR,
                            vec![ISelOperand::VReg(dst), ISelOperand::PReg(*ptr_reg)],
                        ),
                    );
                    self.define_value(*val, ISelOperand::VReg(dst), ty);
                }
                ArgLocation::RegPair(lo_preg, hi_preg) => {
                    // i128 register pair return: copy both halves from
                    // physical registers to virtual registers.
                    let dst_lo = self.new_vreg(RegClass::Gpr64);
                    let dst_hi = self.new_vreg(RegClass::Gpr64);
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::Copy,
                            vec![ISelOperand::VReg(dst_lo), ISelOperand::PReg(*lo_preg)],
                        ),
                    );
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::Copy,
                            vec![ISelOperand::VReg(dst_hi), ISelOperand::PReg(*hi_preg)],
                        ),
                    );
                    self.define_i128_value(*val, dst_lo, dst_hi);
                }
                ArgLocation::Stack { .. } => {
                    return Err(ISelError::UnsupportedReturnLocation);
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Invoke (call that may throw)
    // -----------------------------------------------------------------------

    /// Select an `Invoke` instruction (call with exception handling).
    ///
    /// An invoke is lowered identically to a Call (BL instruction) at the
    /// machine level. The difference is purely metadata: the call site is
    /// recorded for the LSDA call site table, with the landing pad offset
    /// pointing to the unwind_dest block.
    ///
    /// At the MachIR level, the invoke generates:
    /// 1. The same BL as a normal call
    /// 2. A fallthrough to normal_dest (B instruction)
    /// 3. The unwind_dest block is recorded as a successor (for CFG)
    ///
    /// The LSDA generation pass later uses the EH metadata to build the
    /// call site table mapping the BL instruction range to the landing pad.
    ///
    /// Reference: LLVM SelectionDAGBuilder::visitInvoke
    fn select_invoke(
        &mut self,
        name: &str,
        normal_dest: Block,
        unwind_dest: Block,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        // Lower the call exactly like a normal Call.
        let result_types: Vec<Type> = inst.results.iter().map(|v| self.value_type(v)).collect();
        self.select_call(name, &inst.args, &inst.results, &result_types, block)?;

        // Add a branch to normal_dest (the non-exception path).
        self.emit(
            block,
            ISelInst::new(AArch64Opcode::B, vec![ISelOperand::Block(normal_dest)]),
        );

        // Record both successors in the block (normal + unwind).
        let isel_block = self
            .func
            .blocks
            .get_mut(&block)
            .ok_or(ISelError::BlockNotFound)?;
        if !isel_block.successors.contains(&normal_dest) {
            isel_block.successors.push(normal_dest);
        }
        if !isel_block.successors.contains(&unwind_dest) {
            isel_block.successors.push(unwind_dest);
        }

        Ok(())
    }

    /// Select a `LandingPad` instruction.
    ///
    /// A landing pad marks the beginning of an exception handler block.
    /// The unwinder has already set up the exception state before transferring
    /// control here. On AArch64, the unwinder places:
    /// - The exception object pointer in X0
    /// - The type selector in X1
    ///
    /// This method defines the two result values (exception pointer and
    /// type selector) from X0 and X1 respectively.
    ///
    /// The `is_cleanup` and `catch_type_indices` metadata are stored on the
    /// instruction for later use by the LSDA generation pass. At the
    /// machine instruction level, the landing pad is just a block entry
    /// that reads X0 and X1.
    ///
    /// Reference: Itanium C++ ABI sec. 2.5.1 — exception object in X0
    /// Reference: LLVM AArch64ISelLowering.cpp — EH_RETURN lowering
    fn select_landing_pad(
        &mut self,
        _is_cleanup: bool,
        _catch_type_indices: &[u32],
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        // The landing pad produces two results:
        //   results[0] = exception object pointer (I64, in X0)
        //   results[1] = type selector value (I32, in X1)
        if inst.results.len() >= 1 {
            // Exception pointer: copy from X0.
            let exc_vreg = self.new_vreg(RegClass::Gpr64);
            self.emit(
                block,
                ISelInst::new(
                    AArch64Opcode::MovR,
                    vec![ISelOperand::VReg(exc_vreg), ISelOperand::PReg(gpr::X0)],
                ),
            );
            self.define_value(inst.results[0], ISelOperand::VReg(exc_vreg), Type::I64);
        }

        if inst.results.len() >= 2 {
            // Type selector: copy from X1.
            let sel_vreg = self.new_vreg(RegClass::Gpr32);
            self.emit(
                block,
                ISelInst::new(
                    AArch64Opcode::MovR,
                    vec![ISelOperand::VReg(sel_vreg), ISelOperand::PReg(gpr::X1)],
                ),
            );
            self.define_value(inst.results[1], ISelOperand::VReg(sel_vreg), Type::I32);
        }

        Ok(())
    }

    /// Select a `Resume` instruction (re-throw / continue unwinding).
    ///
    /// Resume transfers the exception object back to the unwinder to continue
    /// stack unwinding after a cleanup handler has executed.
    ///
    /// Lowered to a call to `_Unwind_Resume(exception_ptr)`:
    /// - Move args[0] (exception pointer) to X0
    /// - BL _Unwind_Resume
    ///
    /// `_Unwind_Resume` does not return (it transfers control to the next
    /// frame's landing pad or terminates).
    ///
    /// Reference: Itanium C++ ABI sec. 1.3 — _Unwind_Resume
    fn select_resume(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "Resume")?;

        // Move exception pointer to X0.
        let exc_ptr = self.use_value(&inst.args[0])?;
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::MovR,
                vec![ISelOperand::PReg(gpr::X0), exc_ptr],
            ),
        );

        // Call _Unwind_Resume (does not return).
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Bl,
                vec![ISelOperand::Symbol("_Unwind_Resume".to_string())],
            ),
        );

        Ok(())
    }

    /// Select a call instruction from the LIR `Opcode::Call { name }`.
    ///
    /// This bridges the adapter's `Call` opcode to the existing `select_call`
    /// method by extracting the callee name, argument values, result values,
    /// and result types from the LIR instruction.
    fn select_call_from_lir(
        &mut self,
        name: &str,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        let result_types: Vec<Type> = inst.results.iter().map(|v| self.value_type(v)).collect();
        self.select_call(name, &inst.args, &inst.results, &result_types, block)
    }

    // -----------------------------------------------------------------------
    // Variadic calls (Apple AArch64 ABI)
    // -----------------------------------------------------------------------

    /// Select a variadic function call (e.g., printf, NSLog).
    ///
    /// Apple AArch64 ABI variadic convention:
    /// - Fixed args (0..fixed_count) use normal register/stack classification
    /// - ALL variadic args (fixed_count..) go on the stack, 8-byte aligned
    ///
    /// This differs from standard AAPCS64, where variadic args can use registers.
    pub fn select_variadic_call(
        &mut self,
        callee_name: &str,
        fixed_count: usize,
        arg_vals: &[Value],
        result_vals: &[Value],
        result_types: &[Type],
        block: Block,
    ) -> Result<(), ISelError> {
        // Classify argument locations using variadic ABI rules
        let arg_types: Vec<Type> = arg_vals.iter().map(|v| self.value_type(v)).collect();
        let arg_locs = AppleAArch64ABI::classify_params_variadic(fixed_count, &arg_types);

        // Move arguments to ABI locations
        for (i, (val, loc)) in arg_vals.iter().zip(arg_locs.iter()).enumerate() {
            let src = self.use_value(val)?;
            let ty = arg_types[i].clone();

            // Dispatch aggregate arguments to specialized handler
            if ty.is_aggregate() {
                self.select_aggregate_arg(src, &ty, loc, block)?;
                continue;
            }

            match loc {
                ArgLocation::Reg(preg) => {
                    // Use Copy pseudo for ALL register moves (GPR and FPR).
                    // lower_copies() dispatches to MovR (GPR) or FmovFprFpr (FPR).
                    self.emit(
                        block,
                        ISelInst::new(AArch64Opcode::Copy, vec![ISelOperand::PReg(*preg), src]),
                    );
                }
                ArgLocation::Stack { offset, size: _ } => {
                    // STR to [SP + offset] — used for both fixed overflow and
                    // variadic arguments (Apple ABI puts all varargs on stack).
                    let opc = if matches!(ty, Type::F32) {
                        AArch64Opcode::StrRI
                    } else if matches!(ty, Type::F64) {
                        AArch64Opcode::StrRI
                    } else if Self::is_32bit(&ty) {
                        AArch64Opcode::StrRI
                    } else {
                        AArch64Opcode::StrRI
                    };
                    self.emit(
                        block,
                        ISelInst::new(
                            opc,
                            vec![src, ISelOperand::PReg(SP), ISelOperand::Imm(*offset)],
                        ),
                    );
                }
                ArgLocation::Indirect { ptr_reg } => {
                    // Large aggregate indirect: pass pointer in register
                    self.emit(
                        block,
                        ISelInst::new(AArch64Opcode::MovR, vec![ISelOperand::PReg(*ptr_reg), src]),
                    );
                }
                ArgLocation::RegPair(lo_preg, hi_preg) => {
                    // i128 register pair argument: copy both halves
                    let (lo, hi) = self.use_i128_value(val)?;
                    self.emit(
                        block,
                        ISelInst::new(AArch64Opcode::Copy, vec![ISelOperand::PReg(*lo_preg), lo]),
                    );
                    self.emit(
                        block,
                        ISelInst::new(AArch64Opcode::Copy, vec![ISelOperand::PReg(*hi_preg), hi]),
                    );
                }
                ArgLocation::RegSequence(_) => {
                    // HFA: aggregate types dispatched above via select_aggregate_arg
                    return Err(ISelError::UnsupportedArgLocation);
                }
            }
        }

        // Emit BL (direct call) with the callee symbol for relocation.
        // If the callee was tagged `ProofAnnotation::Pure` in the tMIR source
        // (propagated via `Function::pure_callees` → `seed_pure_callees`),
        // stamp the Bl with `Some(Pure)` so SROA partial-escape (#456) can
        // see that the call does not escape its pointer arguments.
        let bl_proof = if self.pure_callees.contains(callee_name) {
            Some(llvm2_ir::ProofAnnotation::Pure)
        } else {
            None
        };
        self.emit_with_proof(
            block,
            ISelInst::new(
                AArch64Opcode::Bl,
                vec![ISelOperand::Symbol(callee_name.to_string())],
            ),
            bl_proof,
        );

        // Copy results from ABI return registers to vregs
        let ret_locs = AppleAArch64ABI::classify_returns(result_types);
        for (i, (val, loc)) in result_vals.iter().zip(ret_locs.iter()).enumerate() {
            match loc {
                ArgLocation::Reg(preg) => {
                    let ty = result_types[i].clone();
                    let class = reg_class_for_type(&ty);
                    let dst = self.new_vreg(class);
                    // Use Copy pseudo for ALL register moves (GPR and FPR).
                    // lower_copies() dispatches to MovR (GPR) or FmovFprFpr (FPR).
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::Copy,
                            vec![ISelOperand::VReg(dst), ISelOperand::PReg(*preg)],
                        ),
                    );
                    self.define_value(*val, ISelOperand::VReg(dst), ty);
                }
                ArgLocation::RegSequence(regs) => {
                    // HFA return from variadic call: copy first FPR register.
                    let ty = result_types[i].clone();
                    let dst = self.new_vreg(RegClass::Gpr64);
                    if let Some(first) = regs.first() {
                        self.emit(
                            block,
                            ISelInst::new(
                                AArch64Opcode::MovR,
                                vec![ISelOperand::VReg(dst), ISelOperand::PReg(*first)],
                            ),
                        );
                    }
                    self.define_value(*val, ISelOperand::VReg(dst), ty);
                }
                ArgLocation::Indirect { ptr_reg } => {
                    let ty = result_types[i].clone();
                    let dst = self.new_vreg(RegClass::Gpr64);
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::MovR,
                            vec![ISelOperand::VReg(dst), ISelOperand::PReg(*ptr_reg)],
                        ),
                    );
                    self.define_value(*val, ISelOperand::VReg(dst), ty);
                }
                ArgLocation::RegPair(lo_preg, hi_preg) => {
                    // i128 register pair return from variadic call
                    let dst_lo = self.new_vreg(RegClass::Gpr64);
                    let dst_hi = self.new_vreg(RegClass::Gpr64);
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::Copy,
                            vec![ISelOperand::VReg(dst_lo), ISelOperand::PReg(*lo_preg)],
                        ),
                    );
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::Copy,
                            vec![ISelOperand::VReg(dst_hi), ISelOperand::PReg(*hi_preg)],
                        ),
                    );
                    self.define_i128_value(*val, dst_lo, dst_hi);
                }
                ArgLocation::Stack { .. } => {
                    return Err(ISelError::UnsupportedReturnLocation);
                }
            }
        }
        Ok(())
    }

    /// Select a variadic call from LIR `Opcode::CallVariadic`.
    fn select_variadic_call_from_lir(
        &mut self,
        name: &str,
        fixed_args: u32,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        let result_types: Vec<Type> = inst.results.iter().map(|v| self.value_type(v)).collect();
        self.select_variadic_call(
            name,
            fixed_args as usize,
            &inst.args,
            &inst.results,
            &result_types,
            block,
        )
    }

    // -----------------------------------------------------------------------
    // Indirect call (CallIndirect)
    // -----------------------------------------------------------------------

    /// Select an indirect function call (BLR).
    ///
    /// `CallIndirect` is like `Call` but the target is a register (function
    /// pointer) instead of a symbol name.
    ///
    /// args[0] = function pointer (I64)
    /// args[1..] = call arguments (classified per ABI)
    /// results = return values
    ///
    /// Lowering:
    /// 1. Move call arguments to ABI registers/stack (same as direct call)
    /// 2. MOV function pointer to a scratch register (X16, intra-procedure-call
    ///    scratch per AArch64 ABI)
    /// 3. BLR scratch_reg
    /// 4. Copy return values from ABI registers to vregs
    pub fn select_call_indirect(
        &mut self,
        fn_ptr_val: &Value,
        arg_vals: &[Value],
        result_vals: &[Value],
        result_types: &[Type],
        block: Block,
    ) -> Result<(), ISelError> {
        // Classify argument locations (excluding the function pointer)
        let arg_types: Vec<Type> = arg_vals.iter().map(|v| self.value_type(v)).collect();
        let arg_locs = AppleAArch64ABI::classify_params(&arg_types);

        // Move arguments to ABI locations (same as direct call)
        for (i, (val, loc)) in arg_vals.iter().zip(arg_locs.iter()).enumerate() {
            let src = self.use_value(val)?;
            let ty = arg_types[i].clone();

            if ty.is_aggregate() {
                self.select_aggregate_arg(src, &ty, loc, block)?;
                continue;
            }

            match loc {
                ArgLocation::Reg(preg) => {
                    // Use Copy pseudo for ALL register moves (GPR and FPR).
                    // lower_copies() dispatches to MovR (GPR) or FmovFprFpr (FPR).
                    self.emit(
                        block,
                        ISelInst::new(AArch64Opcode::Copy, vec![ISelOperand::PReg(*preg), src]),
                    );
                }
                ArgLocation::Stack { offset, size: _ } => {
                    // AArch64 StrRI handles all widths; register class determines size.
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::StrRI,
                            vec![src, ISelOperand::PReg(SP), ISelOperand::Imm(*offset)],
                        ),
                    );
                }
                ArgLocation::Indirect { ptr_reg } => {
                    self.emit(
                        block,
                        ISelInst::new(AArch64Opcode::MovR, vec![ISelOperand::PReg(*ptr_reg), src]),
                    );
                }
                ArgLocation::RegPair(lo_preg, hi_preg) => {
                    // i128 register pair argument for indirect call
                    let (lo, hi) = self.use_i128_value(val)?;
                    self.emit(
                        block,
                        ISelInst::new(AArch64Opcode::Copy, vec![ISelOperand::PReg(*lo_preg), lo]),
                    );
                    self.emit(
                        block,
                        ISelInst::new(AArch64Opcode::Copy, vec![ISelOperand::PReg(*hi_preg), hi]),
                    );
                }
                ArgLocation::RegSequence(_) => {
                    // HFA: aggregate types dispatched above via select_aggregate_arg
                    return Err(ISelError::UnsupportedArgLocation);
                }
            }
        }

        // Move function pointer to X16 (intra-procedure-call scratch register).
        // X16/X17 are designated as IP0/IP1 in AArch64 ABI — used by linker
        // veneers and safe to clobber across calls.
        let fn_ptr = self.use_value(fn_ptr_val)?;
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::MovR,
                vec![ISelOperand::PReg(gpr::X16), fn_ptr],
            ),
        );

        // Emit BLR X16 (indirect call via register)
        self.emit(
            block,
            ISelInst::new(AArch64Opcode::Blr, vec![ISelOperand::PReg(gpr::X16)]),
        );

        // Copy results from ABI return registers to vregs
        let ret_locs = AppleAArch64ABI::classify_returns(result_types);
        for (i, (val, loc)) in result_vals.iter().zip(ret_locs.iter()).enumerate() {
            match loc {
                ArgLocation::Reg(preg) => {
                    let ty = result_types[i].clone();
                    let class = reg_class_for_type(&ty);
                    let dst = self.new_vreg(class);
                    // Use Copy pseudo for ALL register moves (GPR and FPR).
                    // lower_copies() dispatches to MovR (GPR) or FmovFprFpr (FPR).
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::Copy,
                            vec![ISelOperand::VReg(dst), ISelOperand::PReg(*preg)],
                        ),
                    );
                    self.define_value(*val, ISelOperand::VReg(dst), ty);
                }
                ArgLocation::RegSequence(regs) => {
                    // HFA return from indirect call
                    let ty = result_types[i].clone();
                    let dst = self.new_vreg(RegClass::Gpr64);
                    if let Some(first) = regs.first() {
                        self.emit(
                            block,
                            ISelInst::new(
                                AArch64Opcode::MovR,
                                vec![ISelOperand::VReg(dst), ISelOperand::PReg(*first)],
                            ),
                        );
                    }
                    self.define_value(*val, ISelOperand::VReg(dst), ty);
                }
                ArgLocation::Indirect { ptr_reg } => {
                    let ty = result_types[i].clone();
                    let dst = self.new_vreg(RegClass::Gpr64);
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::MovR,
                            vec![ISelOperand::VReg(dst), ISelOperand::PReg(*ptr_reg)],
                        ),
                    );
                    self.define_value(*val, ISelOperand::VReg(dst), ty);
                }
                ArgLocation::RegPair(lo_preg, hi_preg) => {
                    // i128 register pair return from indirect call
                    let dst_lo = self.new_vreg(RegClass::Gpr64);
                    let dst_hi = self.new_vreg(RegClass::Gpr64);
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::Copy,
                            vec![ISelOperand::VReg(dst_lo), ISelOperand::PReg(*lo_preg)],
                        ),
                    );
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::Copy,
                            vec![ISelOperand::VReg(dst_hi), ISelOperand::PReg(*hi_preg)],
                        ),
                    );
                    self.define_i128_value(*val, dst_lo, dst_hi);
                }
                ArgLocation::Stack { .. } => {
                    return Err(ISelError::UnsupportedReturnLocation);
                }
            }
        }
        Ok(())
    }

    /// Select a `CallIndirect` from LIR.
    ///
    /// `inst.args[0]` is the function pointer, `inst.args[1..]` are call args.
    fn select_call_indirect_from_lir(
        &mut self,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "CallIndirect")?;

        let fn_ptr_val = inst.args[0];
        let call_args = &inst.args[1..];
        let result_types: Vec<Type> = inst.results.iter().map(|v| self.value_type(v)).collect();
        self.select_call_indirect(&fn_ptr_val, call_args, &inst.results, &result_types, block)
    }

    // -----------------------------------------------------------------------
    // Switch (jump table / binary search / linear scan)
    // -----------------------------------------------------------------------

    /// Select a switch statement, choosing between jump table, binary search,
    /// and linear scan based on case count and density.
    ///
    /// - N <= 3: linear scan (sequential CMP+B.EQ chain, O(n))
    /// - N >= 4, density > 0.4: jump table (O(1) dispatch)
    /// - N > 3, sparse: binary search tree (O(log n) dispatch)
    fn select_switch(
        &mut self,
        cases: &[(i64, Block)],
        default: Block,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        use crate::switch::SwitchStrategy;

        match crate::switch::choose_strategy(cases) {
            SwitchStrategy::JumpTable => self.select_switch_jump_table(cases, default, inst, block),
            SwitchStrategy::BinarySearch => {
                self.select_switch_binary_search(cases, default, inst, block)
            }
            SwitchStrategy::LinearScan => self.select_switch_cascade(cases, default, inst, block),
        }
    }

    /// Select a sparse switch using a binary search tree.
    ///
    /// Creates a balanced BST of compare-and-branch blocks with O(log n)
    /// worst-case dispatch. Each internal BST node compares the selector
    /// against a pivot, branching to the target on equality or to left/right
    /// subtrees. Leaf groups (1-3 cases) use linear scan.
    fn select_switch_binary_search(
        &mut self,
        cases: &[(i64, Block)],
        default: Block,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "Switch")?;

        let selector_val = inst.args[0];
        let selector = self.use_value(&selector_val)?;
        let sel_ty = self.value_type(&selector_val);
        let is_32 = Self::is_32bit(&sel_ty);

        // Ensure next_block_id is above all case target blocks and default.
        for &(_, target) in cases {
            if target.0 >= self.next_block_id {
                self.next_block_id = target.0 + 1;
            }
        }
        if default.0 >= self.next_block_id {
            self.next_block_id = default.0 + 1;
        }
        if block.0 >= self.next_block_id {
            self.next_block_id = block.0 + 1;
        }

        crate::switch::emit_binary_search(
            &mut self.func,
            &mut self.next_block_id,
            &selector,
            is_32,
            cases,
            default,
            block,
        );

        Ok(())
    }

    /// Select a dense switch statement using a jump table.
    ///
    /// Delegates to `switch::emit_jump_table` which emits the AArch64 sequence:
    /// SUB (normalize), CMP+B.HI (range check), ADR+LDRSW+ADD+BR (table dispatch).
    fn select_switch_jump_table(
        &mut self,
        cases: &[(i64, Block)],
        default: Block,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "Switch")?;

        let selector_val = inst.args[0];
        let selector = self.use_value(&selector_val)?;
        let sel_ty = self.value_type(&selector_val);
        let is_32 = Self::is_32bit(&sel_ty);

        crate::switch::emit_jump_table(
            &mut self.func,
            &mut self.next_block_id,
            &selector,
            is_32,
            cases,
            default,
            block,
        );

        Ok(())
    }

    /// Select a switch statement as a cascading CMP+B.EQ chain (sparse path).
    ///
    /// For each case `(value, target_block)`:
    ///   CMP selector, #value
    ///   B.EQ target_block
    /// After all cases:
    ///   B default_block
    fn select_switch_cascade(
        &mut self,
        cases: &[(i64, Block)],
        default: Block,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "Switch")?;

        let selector_val = inst.args[0];
        let selector = self.use_value(&selector_val)?;
        let sel_ty = self.value_type(&selector_val);
        let is_32 = Self::is_32bit(&sel_ty);

        // For each case: CMP selector, #case_val then B.EQ target
        for (case_val, target) in cases {
            // Materialize the case value as an immediate comparison.
            // AArch64 CMP immediate (CMPri) supports 12-bit unsigned values.
            // For values outside that range, materialize into a register first.
            let case_fits_imm12 = *case_val >= 0 && *case_val <= 0xFFF;

            if case_fits_imm12 {
                let cmp_opc = if is_32 {
                    AArch64Opcode::CmpRI
                } else {
                    AArch64Opcode::CmpRI
                };
                self.emit(
                    block,
                    ISelInst::new(cmp_opc, vec![selector.clone(), ISelOperand::Imm(*case_val)]),
                );
            } else {
                // Materialize case value into a register, then CMP reg, reg.
                let class = if is_32 {
                    RegClass::Gpr32
                } else {
                    RegClass::Gpr64
                };
                let case_vreg = self.new_vreg(class);
                let mov_opc = if is_32 {
                    AArch64Opcode::Movz
                } else {
                    AArch64Opcode::Movz
                };
                // For simplicity, use MOVZi for non-negative values that fit
                // in 16 bits, and the full materialization sequence for larger.
                // TODO: Full 64-bit materialization for very large constants.
                self.emit(
                    block,
                    ISelInst::new(
                        mov_opc,
                        vec![ISelOperand::VReg(case_vreg), ISelOperand::Imm(*case_val)],
                    ),
                );

                let cmp_opc = if is_32 {
                    AArch64Opcode::CmpRR
                } else {
                    AArch64Opcode::CmpRR
                };
                self.emit(
                    block,
                    ISelInst::new(
                        cmp_opc,
                        vec![selector.clone(), ISelOperand::VReg(case_vreg)],
                    ),
                );
            }

            // B.EQ target_block
            self.emit(
                block,
                ISelInst::new(
                    AArch64Opcode::BCond,
                    vec![
                        ISelOperand::CondCode(AArch64CC::EQ),
                        ISelOperand::Block(*target),
                    ],
                ),
            );

            // Record successor
            self.func
                .blocks
                .entry(block)
                .or_default()
                .successors
                .push(*target);
        }

        // Fall through to default block
        self.emit(
            block,
            ISelInst::new(AArch64Opcode::B, vec![ISelOperand::Block(default)]),
        );
        self.func
            .blocks
            .entry(block)
            .or_default()
            .successors
            .push(default);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Load / Store
    // -----------------------------------------------------------------------

    /// Select a memory load.
    fn select_load(&mut self, ty: Type, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "Load")?;
        Self::require_result(inst, "Load")?;

        let addr_val = inst.args[0];
        let result_val = inst.results[0];
        let addr = self.use_value(&addr_val)?;

        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        // Select opcode based on load type. For I8/I16, we default to
        // zero-extending loads (LDRB/LDRH). Sign-extending loads (LDRSB/LDRSH)
        // would be selected when the load feeds into an Sextend, but that
        // optimization is deferred to late combines (Phase 2).
        let opc = match ty {
            Type::I8 => AArch64Opcode::LdrbRI,
            Type::I16 => AArch64Opcode::LdrhRI,
            Type::I32 => AArch64Opcode::LdrRI,
            Type::I64 => AArch64Opcode::LdrRI,
            Type::F32 => AArch64Opcode::LdrRI,
            Type::F64 => AArch64Opcode::LdrRI,
            _ => AArch64Opcode::LdrRI,
        };

        self.emit(
            block,
            ISelInst::new(opc, vec![ISelOperand::VReg(dst), addr, ISelOperand::Imm(0)]),
        );

        self.define_value(result_val, ISelOperand::VReg(dst), ty);
        Ok(())
    }

    /// Select a memory store.
    fn select_store(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "Store")?;

        let value_val = inst.args[0];
        let addr_val = inst.args[1];

        let src = self.use_value(&value_val)?;
        let addr = self.use_value(&addr_val)?;
        let ty = self.value_type(&value_val);

        let opc = match ty {
            Type::I8 => AArch64Opcode::StrbRI,
            Type::I16 => AArch64Opcode::StrhRI,
            Type::I32 => AArch64Opcode::StrRI,
            Type::I64 => AArch64Opcode::StrRI,
            Type::F32 => AArch64Opcode::StrRI,
            Type::F64 => AArch64Opcode::StrRI,
            _ => AArch64Opcode::StrRI,
        };

        self.emit(
            block,
            ISelInst::new(opc, vec![src, addr, ISelOperand::Imm(0)]),
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Atomic memory operations
    // -----------------------------------------------------------------------

    /// Select an atomic load: LDAR Rt, [Rn].
    ///
    /// Atomic loads use LDAR (load-acquire) which provides acquire semantics.
    /// For byte/halfword sizes, LDARB/LDARH are used.
    /// Operands: [Rt (dest), Rn (address)].
    fn select_atomic_load(
        &mut self,
        ty: Type,
        _ordering: AtomicOrdering,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "AtomicLoad")?;
        Self::require_result(inst, "AtomicLoad")?;

        let addr_val = inst.args[0];
        let result_val = inst.results[0];
        let addr = self.use_value(&addr_val)?;

        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        // Select LDAR variant based on type size.
        // All orderings map to LDAR (acquire) since AArch64 LDAR provides
        // at least acquire semantics (load-acquire). For SeqCst, the
        // LDAR/STLR pair provides sequential consistency per the
        // ARMv8 memory model.
        let opc = match ty {
            Type::I8 => AArch64Opcode::Ldarb,
            Type::I16 => AArch64Opcode::Ldarh,
            _ => AArch64Opcode::Ldar, // I32, I64
        };

        self.emit(
            block,
            ISelInst::new(opc, vec![ISelOperand::VReg(dst), addr]),
        );

        self.define_value(result_val, ISelOperand::VReg(dst), ty);
        Ok(())
    }

    /// Select an atomic store: STLR Rt, [Rn].
    ///
    /// Atomic stores use STLR (store-release) which provides release semantics.
    /// For byte/halfword sizes, STLRB/STLRH are used.
    /// Operands: [Rt (value), Rn (address)].
    fn select_atomic_store(
        &mut self,
        _ordering: AtomicOrdering,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "AtomicStore")?;

        let value_val = inst.args[0];
        let addr_val = inst.args[1];

        let src = self.use_value(&value_val)?;
        let addr = self.use_value(&addr_val)?;
        let ty = self.value_type(&value_val);

        let opc = match ty {
            Type::I8 => AArch64Opcode::Stlrb,
            Type::I16 => AArch64Opcode::Stlrh,
            _ => AArch64Opcode::Stlr, // I32, I64
        };

        self.emit(block, ISelInst::new(opc, vec![src, addr]));
        Ok(())
    }

    /// Select an atomic read-modify-write operation.
    ///
    /// LSE path (ARMv8.1-a): Uses LDADD/LDCLR/LDEOR/LDSET/SWP.
    /// All LSE atomics have the form: Xs (operand), Xt (old value), [Xn] (address).
    /// Operands: [Rs, Rt, Rn].
    fn select_atomic_rmw(
        &mut self,
        op: AtomicRmwOp,
        ty: Type,
        ordering: AtomicOrdering,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "AtomicRmw")?;
        Self::require_result(inst, "AtomicRmw")?;

        let val_val = inst.args[0];
        let addr_val = inst.args[1];
        let result_val = inst.results[0];

        let val = self.use_value(&val_val)?;
        let addr = self.use_value(&addr_val)?;

        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        // For AND and SUB, we need a preparation step. Track the actual
        // operand vreg to pass to the LSE instruction.
        let mut actual_val = val;

        // Select LSE opcode. For SeqCst/AcqRel, use *AL variants (acquire+release).
        // For Acquire, use *A variants. Otherwise use base (relaxed).
        let opc = match op {
            AtomicRmwOp::Add => match ordering {
                AtomicOrdering::SeqCst | AtomicOrdering::AcqRel => AArch64Opcode::Ldaddal,
                AtomicOrdering::Acquire => AArch64Opcode::Ldadda,
                _ => AArch64Opcode::Ldadd,
            },
            AtomicRmwOp::And => {
                // AND is lowered as LDCLR with inverted operand.
                // LDCLR clears bits: new = old AND NOT(Rs).
                // To achieve AND(val): new = old AND val = old AND NOT(NOT(val)).
                // So we invert val first: emit MVN (ORN Xd, XZR, Xn alias).
                let inv = self.new_vreg(class);
                let zr = if Self::is_32bit(&ty) {
                    ISelOperand::PReg(WZR)
                } else {
                    ISelOperand::PReg(XZR)
                };
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::OrnRR,
                        vec![ISelOperand::VReg(inv), zr, actual_val],
                    ),
                );
                actual_val = ISelOperand::VReg(inv);
                match ordering {
                    AtomicOrdering::SeqCst | AtomicOrdering::AcqRel => AArch64Opcode::Ldclral,
                    _ => AArch64Opcode::Ldclr,
                }
            }
            AtomicRmwOp::Or => match ordering {
                AtomicOrdering::SeqCst | AtomicOrdering::AcqRel => AArch64Opcode::Ldsetal,
                _ => AArch64Opcode::Ldset,
            },
            AtomicRmwOp::Xor => match ordering {
                AtomicOrdering::SeqCst | AtomicOrdering::AcqRel => AArch64Opcode::Ldeoral,
                _ => AArch64Opcode::Ldeor,
            },
            AtomicRmwOp::Sub => {
                // SUB is lowered as LDADD with negated value.
                // NEG Xd, Xn = SUB Xd, XZR, Xn (2-operand ISel form).
                let neg = self.new_vreg(class);
                self.emit(
                    block,
                    ISelInst::new(AArch64Opcode::Neg, vec![ISelOperand::VReg(neg), actual_val]),
                );
                actual_val = ISelOperand::VReg(neg);
                match ordering {
                    AtomicOrdering::SeqCst | AtomicOrdering::AcqRel => AArch64Opcode::Ldaddal,
                    AtomicOrdering::Acquire => AArch64Opcode::Ldadda,
                    _ => AArch64Opcode::Ldadd,
                }
            }
            AtomicRmwOp::Xchg => match ordering {
                AtomicOrdering::SeqCst | AtomicOrdering::AcqRel => AArch64Opcode::Swpal,
                _ => AArch64Opcode::Swp,
            },
        };

        self.emit(
            block,
            ISelInst::new(opc, vec![actual_val, ISelOperand::VReg(dst), addr]),
        );

        self.define_value(result_val, ISelOperand::VReg(dst), ty);
        Ok(())
    }

    /// Select a compare-and-swap: CAS Rs, Rt, [Xn].
    ///
    /// LSE path: CAS/CASA/CASAL.
    /// Rs = expected value (updated to old value on return).
    /// Rt = desired value.
    /// [Xn] = memory address.
    ///
    /// After CAS: Rs contains old value. Success = (Rs == original expected).
    fn select_cmpxchg(
        &mut self,
        ty: Type,
        ordering: AtomicOrdering,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 3, "CmpXchg")?;
        Self::require_result(inst, "CmpXchg")?;

        let expected_val = inst.args[0];
        let desired_val = inst.args[1];
        let addr_val = inst.args[2];

        let expected = self.use_value(&expected_val)?;
        let desired = self.use_value(&desired_val)?;
        let addr = self.use_value(&addr_val)?;

        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        // Copy expected to dst (CAS reads and writes Rs in-place).
        self.emit(
            block,
            ISelInst::new(AArch64Opcode::MovR, vec![ISelOperand::VReg(dst), expected]),
        );

        let opc = match ordering {
            AtomicOrdering::SeqCst | AtomicOrdering::AcqRel => AArch64Opcode::Casal,
            AtomicOrdering::Acquire => AArch64Opcode::Casa,
            _ => AArch64Opcode::Cas,
        };

        // CAS Rs, Rt, [Xn]: Rs (expected/result), Rt (desired), Xn (address)
        self.emit(
            block,
            ISelInst::new(opc, vec![ISelOperand::VReg(dst), desired, addr]),
        );

        // Result is the old value in dst.
        self.define_value(inst.results[0], ISelOperand::VReg(dst), ty);
        Ok(())
    }

    /// Select a memory fence: DMB.
    ///
    /// DMB with option field:
    ///   - SeqCst/AcqRel: DMB ISH (0xB) — inner-shareable full barrier
    ///   - Acquire: DMB ISHLD (0x9) — inner-shareable load barrier
    ///   - Release: DMB ISHST (0xA) — inner-shareable store barrier
    ///   - Relaxed: NOP (no barrier needed)
    fn select_fence(&mut self, ordering: AtomicOrdering, block: Block) -> Result<(), ISelError> {
        let option = match ordering {
            AtomicOrdering::SeqCst | AtomicOrdering::AcqRel => 0x0B, // ISH
            AtomicOrdering::Acquire => 0x09,                         // ISHLD
            AtomicOrdering::Release => 0x0A,                         // ISHST
            AtomicOrdering::Relaxed => return Ok(()),                // No barrier needed
        };

        self.emit(
            block,
            ISelInst::new(AArch64Opcode::Dmb, vec![ISelOperand::Imm(option)]),
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Aggregate operations
    // -----------------------------------------------------------------------

    /// Select a struct field address computation (GEP-like).
    ///
    /// Given a pointer to a struct and a field index, compute:
    ///   result = base + offset_of(struct_ty, field_index)
    ///
    /// For zero offset (field 0 with no padding), emits a MOV.
    /// Otherwise emits ADD Xd, base, #offset.
    fn select_struct_gep(
        &mut self,
        struct_ty: Type,
        field_index: u32,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "StructGep")?;
        Self::require_result(inst, "StructGep")?;

        // Validate the struct type
        let field_count = match &struct_ty {
            Type::Struct(fields) => fields.len(),
            _ => return Err(ISelError::StructGepNonStruct(struct_ty)),
        };
        if field_index as usize >= field_count {
            return Err(ISelError::StructGepOutOfRange {
                index: field_index,
                field_count,
            });
        }

        let base_val = inst.args[0];
        let result_val = inst.results[0];
        let base = self.use_value(&base_val)?;

        let offset = struct_ty.offset_of(field_index as usize).unwrap_or(0);

        let dst = self.new_vreg(RegClass::Gpr64);

        if offset == 0 {
            // Field is at base address, just move the pointer
            self.emit(
                block,
                ISelInst::new(AArch64Opcode::MovR, vec![ISelOperand::VReg(dst), base]),
            );
        } else {
            // ADD Xd, base, #offset
            self.emit(
                block,
                ISelInst::new(
                    AArch64Opcode::AddRI,
                    vec![
                        ISelOperand::VReg(dst),
                        base,
                        ISelOperand::Imm(offset as i64),
                    ],
                ),
            );
        }

        // Result is a pointer (I64 on AArch64)
        self.define_value(result_val, ISelOperand::VReg(dst), Type::I64);
        Ok(())
    }

    /// Select an array element address computation (GEP-like).
    ///
    /// Given a base pointer and an integer index, compute:
    ///   result = base + index * sizeof(elem_ty)
    ///
    /// For power-of-two element sizes, emits `LslRI + AddRR`. For unit
    /// stride (elem_size == 1), emits a single `AddRR`. Otherwise emits
    /// `Movz` (materialise size), `MulRR`, `AddRR`.
    ///
    /// Phase 1 aggregate lowering — see
    /// `designs/2026-04-18-aggregate-lowering.md`.
    fn select_array_gep(
        &mut self,
        elem_ty: Type,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "ArrayGep")?;
        Self::require_result(inst, "ArrayGep")?;

        let elem_size = elem_ty.bytes();
        if elem_size == 0 {
            return Err(ISelError::ArrayGepZeroSized(elem_ty));
        }

        let base_val = inst.args[0];
        let index_val = inst.args[1];
        let result_val = inst.results[0];
        let base = self.use_value(&base_val)?;
        let index = self.use_value(&index_val)?;
        let dst = self.new_vreg(RegClass::Gpr64);

        if elem_size == 1 {
            // result = base + index
            self.emit(
                block,
                ISelInst::new(
                    AArch64Opcode::AddRR,
                    vec![ISelOperand::VReg(dst), base, index],
                ),
            );
        } else if elem_size.is_power_of_two() {
            // shifted = index << log2(elem_size); result = base + shifted
            let shift = elem_size.trailing_zeros() as i64;
            let shifted = self.new_vreg(RegClass::Gpr64);
            self.emit(
                block,
                ISelInst::new(
                    AArch64Opcode::LslRI,
                    vec![ISelOperand::VReg(shifted), index, ISelOperand::Imm(shift)],
                ),
            );
            self.emit(
                block,
                ISelInst::new(
                    AArch64Opcode::AddRR,
                    vec![ISelOperand::VReg(dst), base, ISelOperand::VReg(shifted)],
                ),
            );
        } else {
            // size_reg = MOVZ #elem_size; scaled = index * size_reg; result = base + scaled
            let size_reg = self.new_vreg(RegClass::Gpr64);
            self.emit(
                block,
                ISelInst::new(
                    AArch64Opcode::Movz,
                    vec![
                        ISelOperand::VReg(size_reg),
                        ISelOperand::Imm(elem_size as i64),
                    ],
                ),
            );
            let scaled = self.new_vreg(RegClass::Gpr64);
            self.emit(
                block,
                ISelInst::new(
                    AArch64Opcode::MulRR,
                    vec![
                        ISelOperand::VReg(scaled),
                        index,
                        ISelOperand::VReg(size_reg),
                    ],
                ),
            );
            self.emit(
                block,
                ISelInst::new(
                    AArch64Opcode::AddRR,
                    vec![ISelOperand::VReg(dst), base, ISelOperand::VReg(scaled)],
                ),
            );
        }

        self.define_value(result_val, ISelOperand::VReg(dst), Type::I64);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Memory intrinsics (memcpy, memmove, memset)
    // -----------------------------------------------------------------------

    /// Select a memcpy intrinsic.
    ///
    /// Lowered to a regular libc call: `memcpy(dest, src, len)`.
    /// args[0] = dest ptr (I64), args[1] = src ptr (I64), args[2] = length (I64).
    /// No results (void).
    ///
    /// Future optimization: inline with LDP/STP for small known sizes.
    fn select_memcpy(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 3, "Memcpy")?;
        // Lower as a regular call to libc memcpy. Void return, no results.
        self.select_call("memcpy", &inst.args, &[], &[], block)
    }

    /// Select a memmove intrinsic.
    ///
    /// Lowered to a regular libc call: `memmove(dest, src, len)`.
    /// args[0] = dest ptr (I64), args[1] = src ptr (I64), args[2] = length (I64).
    /// No results (void).
    fn select_memmove(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 3, "Memmove")?;
        self.select_call("memmove", &inst.args, &[], &[], block)
    }

    /// Select a memset intrinsic.
    ///
    /// Lowered to a regular libc call: `memset(dest, val, len)`.
    /// args[0] = dest ptr (I64), args[1] = fill value (I32), args[2] = length (I64).
    /// No results (void).
    fn select_memset(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 3, "Memset")?;
        self.select_call("memset", &inst.args, &[], &[], block)
    }

    /// Select aggregate return value lowering.
    ///
    /// Apple AArch64 ABI:
    /// - HFA (1-4 same FP fields): load members into consecutive S/D registers
    /// - Small (<=8 bytes): pack fields into X0
    /// - Medium (<=16 bytes): pack fields into X0 + X1
    /// - Large (>16 bytes): store to memory pointed to by X8 (sret)
    ///
    /// This is called from `select_return` when it detects an aggregate type.
    fn select_aggregate_return(
        &mut self,
        src: ISelOperand,
        agg_ty: &Type,
        block: Block,
    ) -> Result<(), ISelError> {
        let size = agg_ty.bytes();

        // Check for HFA first: return in consecutive typed FPR registers.
        if let Some((hfa_base, count)) = AppleAArch64ABI::classify_hfa(agg_ty) {
            let elem_size = match hfa_base {
                HfaBaseType::F32 => 4u32,
                HfaBaseType::F64 => 8u32,
            };
            let typed_regs: &[PReg] = match hfa_base {
                HfaBaseType::F32 => AppleAArch64ABI::s_arg_regs(),
                HfaBaseType::F64 => AppleAArch64ABI::d_arg_regs(),
            };
            // Load each HFA member from memory into the corresponding FPR.
            for i in 0..count {
                if i < typed_regs.len() {
                    let byte_offset = i as u32 * elem_size;
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::LdrRI,
                            vec![
                                ISelOperand::PReg(typed_regs[i]),
                                src.clone(),
                                ISelOperand::Imm(byte_offset as i64),
                            ],
                        ),
                    );
                }
            }
            return Ok(());
        }

        if size <= 8 {
            // Small aggregate: load entire struct as a single X0 value.
            // src is a pointer to the aggregate in memory.
            // Emit: LDR X0, [src]
            self.emit(
                block,
                ISelInst::new(
                    AArch64Opcode::LdrRI,
                    vec![ISelOperand::PReg(gpr::X0), src, ISelOperand::Imm(0)],
                ),
            );
        } else if size <= 16 {
            // Medium aggregate: load into X0 + X1.
            // First 8 bytes -> X0
            self.emit(
                block,
                ISelInst::new(
                    AArch64Opcode::LdrRI,
                    vec![ISelOperand::PReg(gpr::X0), src.clone(), ISelOperand::Imm(0)],
                ),
            );
            // Next bytes -> X1
            self.emit(
                block,
                ISelInst::new(
                    AArch64Opcode::LdrRI,
                    vec![ISelOperand::PReg(gpr::X1), src, ISelOperand::Imm(8)],
                ),
            );
        } else {
            // Large aggregate: store to [X8] (sret pointer).
            // Emit a sequence of stores from the source to the sret buffer.
            // For now, emit word-at-a-time copies. A real implementation would
            // use memcpy or unrolled LDP/STP for large sizes.
            let mut offset: u32 = 0;
            while offset + 8 <= size {
                // Load 8 bytes from source
                let tmp = self.new_vreg(RegClass::Gpr64);
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::LdrRI,
                        vec![
                            ISelOperand::VReg(tmp),
                            src.clone(),
                            ISelOperand::Imm(offset as i64),
                        ],
                    ),
                );
                // Store 8 bytes to sret destination
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::StrRI,
                        vec![
                            ISelOperand::VReg(tmp),
                            ISelOperand::PReg(gpr::X8),
                            ISelOperand::Imm(offset as i64),
                        ],
                    ),
                );
                offset += 8;
            }
            // Handle trailing 4-byte chunk
            if offset + 4 <= size {
                let tmp = self.new_vreg(RegClass::Gpr32);
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::LdrRI,
                        vec![
                            ISelOperand::VReg(tmp),
                            src.clone(),
                            ISelOperand::Imm(offset as i64),
                        ],
                    ),
                );
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::StrRI,
                        vec![
                            ISelOperand::VReg(tmp),
                            ISelOperand::PReg(gpr::X8),
                            ISelOperand::Imm(offset as i64),
                        ],
                    ),
                );
                offset += 4;
            }
            // Handle trailing 2-byte chunk
            if offset + 2 <= size {
                let tmp = self.new_vreg(RegClass::Gpr32);
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::LdrhRI,
                        vec![
                            ISelOperand::VReg(tmp),
                            src.clone(),
                            ISelOperand::Imm(offset as i64),
                        ],
                    ),
                );
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::StrhRI,
                        vec![
                            ISelOperand::VReg(tmp),
                            ISelOperand::PReg(gpr::X8),
                            ISelOperand::Imm(offset as i64),
                        ],
                    ),
                );
                offset += 2;
            }
            // Handle trailing 1-byte chunk
            if offset < size {
                let tmp = self.new_vreg(RegClass::Gpr32);
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::LdrbRI,
                        vec![
                            ISelOperand::VReg(tmp),
                            src.clone(),
                            ISelOperand::Imm(offset as i64),
                        ],
                    ),
                );
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::StrbRI,
                        vec![
                            ISelOperand::VReg(tmp),
                            ISelOperand::PReg(gpr::X8),
                            ISelOperand::Imm(offset as i64),
                        ],
                    ),
                );
            }
        }

        Ok(())
    }

    /// Select aggregate argument passing for a call.
    ///
    /// Apple AArch64 ABI:
    /// - Small (<=8 bytes): load into a single GPR
    /// - Medium (<=16 bytes): load into two consecutive GPRs
    /// - Large (>16 bytes): allocate stack space, store aggregate, pass pointer
    ///
    /// `src` is the pointer to the aggregate in memory.
    /// `preg` is the physical register assigned by the ABI classifier.
    fn select_aggregate_arg(
        &mut self,
        src: ISelOperand,
        agg_ty: &Type,
        loc: &ArgLocation,
        block: Block,
    ) -> Result<(), ISelError> {
        let size = agg_ty.bytes();

        match loc {
            ArgLocation::Reg(preg) => {
                // Small aggregate (<=8 bytes): load as single value into register
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::LdrRI,
                        vec![ISelOperand::PReg(*preg), src, ISelOperand::Imm(0)],
                    ),
                );
            }
            ArgLocation::RegSequence(regs) => {
                // HFA (Homogeneous Floating-point Aggregate): load each member
                // from memory into the designated typed FPR register.
                //
                // The register list uses the element type's register class:
                //   F32 HFA: S0, S1, S2, ... (each member is 4 bytes)
                //   F64 HFA: D0, D1, D2, ... (each member is 8 bytes)
                //
                // `src` is a pointer to the aggregate in memory. Each member is
                // loaded from `[src + i * element_size]`.
                let elem_size = if regs.is_empty() {
                    4 // fallback; shouldn't happen
                } else {
                    // Determine element size from HFA classification.
                    if let Some((hfa_base, _)) = AppleAArch64ABI::classify_hfa(agg_ty) {
                        match hfa_base {
                            HfaBaseType::F32 => 4,
                            HfaBaseType::F64 => 8,
                        }
                    } else {
                        size / regs.len() as u32
                    }
                };
                for (i, preg) in regs.iter().enumerate() {
                    let byte_offset = i as u32 * elem_size;
                    // LDR Sn/Dn, [src, #offset]
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::LdrRI,
                            vec![
                                ISelOperand::PReg(*preg),
                                src.clone(),
                                ISelOperand::Imm(byte_offset as i64),
                            ],
                        ),
                    );
                }
            }
            ArgLocation::Indirect { ptr_reg } => {
                if size <= 16 {
                    // Medium aggregate passed as register pair.
                    // The ABI classifier returns Indirect for medium aggregates,
                    // meaning the first register of the pair. Load first 8 bytes.
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::LdrRI,
                            vec![
                                ISelOperand::PReg(*ptr_reg),
                                src.clone(),
                                ISelOperand::Imm(0),
                            ],
                        ),
                    );
                    // Load next bytes into the following register.
                    // ptr_reg is XN, next register is X(N+1).
                    let next_reg = PReg::new(ptr_reg.hw_enc() as u16 + 1);
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::LdrRI,
                            vec![ISelOperand::PReg(next_reg), src, ISelOperand::Imm(8)],
                        ),
                    );
                } else {
                    // Large aggregate: pass pointer to the aggregate.
                    // The caller has already placed the aggregate in memory;
                    // just pass the pointer in the designated register.
                    self.emit(
                        block,
                        ISelInst::new(AArch64Opcode::MovR, vec![ISelOperand::PReg(*ptr_reg), src]),
                    );
                }
            }
            ArgLocation::Stack {
                offset,
                size: slot_size,
            } => {
                // Aggregate on stack: store field-by-field.
                let mut byte_offset: u32 = 0;
                while byte_offset + 8 <= *slot_size {
                    let tmp = self.new_vreg(RegClass::Gpr64);
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::LdrRI,
                            vec![
                                ISelOperand::VReg(tmp),
                                src.clone(),
                                ISelOperand::Imm(byte_offset as i64),
                            ],
                        ),
                    );
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::StrRI,
                            vec![
                                ISelOperand::VReg(tmp),
                                ISelOperand::PReg(SP),
                                ISelOperand::Imm(*offset + byte_offset as i64),
                            ],
                        ),
                    );
                    byte_offset += 8;
                }
                if byte_offset + 4 <= *slot_size {
                    let tmp = self.new_vreg(RegClass::Gpr32);
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::LdrRI,
                            vec![
                                ISelOperand::VReg(tmp),
                                src.clone(),
                                ISelOperand::Imm(byte_offset as i64),
                            ],
                        ),
                    );
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::StrRI,
                            vec![
                                ISelOperand::VReg(tmp),
                                ISelOperand::PReg(SP),
                                ISelOperand::Imm(*offset + byte_offset as i64),
                            ],
                        ),
                    );
                    byte_offset += 4;
                }
                // Handle trailing 2-byte chunk
                if byte_offset + 2 <= *slot_size {
                    let tmp = self.new_vreg(RegClass::Gpr32);
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::LdrhRI,
                            vec![
                                ISelOperand::VReg(tmp),
                                src.clone(),
                                ISelOperand::Imm(byte_offset as i64),
                            ],
                        ),
                    );
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::StrhRI,
                            vec![
                                ISelOperand::VReg(tmp),
                                ISelOperand::PReg(SP),
                                ISelOperand::Imm(*offset + byte_offset as i64),
                            ],
                        ),
                    );
                    byte_offset += 2;
                }
                // Handle trailing 1-byte chunk
                if byte_offset < *slot_size {
                    let tmp = self.new_vreg(RegClass::Gpr32);
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::LdrbRI,
                            vec![
                                ISelOperand::VReg(tmp),
                                src.clone(),
                                ISelOperand::Imm(byte_offset as i64),
                            ],
                        ),
                    );
                    self.emit(
                        block,
                        ISelInst::new(
                            AArch64Opcode::StrbRI,
                            vec![
                                ISelOperand::VReg(tmp),
                                ISelOperand::PReg(SP),
                                ISelOperand::Imm(*offset + byte_offset as i64),
                            ],
                        ),
                    );
                }
            }
            ArgLocation::RegPair(_, _) => {
                // RegPair is only used for i128 scalar values, never for aggregates.
                return Err(ISelError::UnsupportedArgLocation);
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Function entry: lower formal arguments
    // -----------------------------------------------------------------------

    /// Lower formal arguments at function entry.
    ///
    /// For each parameter, emit a COPY from the physical register (or stack
    /// location) designated by the ABI to a fresh virtual register. This
    /// establishes the initial value_map for the function body.
    pub fn lower_formal_arguments(
        &mut self,
        sig: &Signature,
        entry_block: Block,
    ) -> Result<(), ISelError> {
        self.func.ensure_block(entry_block);

        let param_locs = AppleAArch64ABI::classify_params(&sig.params);

        // We need Value ids for the formal arguments. By convention, formal
        // args are Value(0), Value(1), ..., Value(n-1).
        for (i, (ty, loc)) in sig.params.iter().zip(param_locs.iter()).enumerate() {
            let val = Value(i as u32);
            let class = reg_class_for_type(ty);
            let vreg = self.new_vreg(class);

            match loc {
                ArgLocation::Reg(preg) => {
                    // COPY from physical to virtual register
                    self.emit(
                        entry_block,
                        ISelInst::new(
                            AArch64Opcode::Copy,
                            vec![ISelOperand::VReg(vreg), ISelOperand::PReg(*preg)],
                        ),
                    );
                }
                ArgLocation::Stack { offset, size: _ } => {
                    // Load from stack using an IncomingArg operand that frame
                    // lowering resolves to `[FP, #callee_saved_area_size + offset]`.
                    //
                    // Incoming stack arguments are located above the callee's
                    // saved FP/LR and callee-saved registers. FP points at the
                    // saved X29/X30, and the prologue stores callee-saved
                    // registers above that (at positive offsets from FP).
                    // The caller's SP (where incoming args start) is at
                    // `FP + callee_saved_area_size`, so incoming arg N is at
                    // `FP + callee_saved_area_size + N`.
                    //
                    // We cannot compute that offset here because ISel runs
                    // before register allocation decides which callee-saved
                    // registers to preserve; the CSA size is only known after
                    // regalloc, during frame lowering. We therefore emit an
                    // abstract IncomingArg(offset) operand and let frame
                    // lowering fill in the concrete FP-relative offset.
                    //
                    // Reference: frame.rs FrameLayout.callee_saved_area_size
                    // Reference: LLVM AArch64FrameLowering::resolveFrameIndexReference
                    let opc = if Self::is_32bit(ty) {
                        AArch64Opcode::LdrRI
                    } else {
                        AArch64Opcode::LdrRI
                    };
                    self.emit(
                        entry_block,
                        ISelInst::new(
                            opc,
                            vec![
                                ISelOperand::VReg(vreg),
                                ISelOperand::PReg(FP),
                                ISelOperand::IncomingArg(*offset),
                            ],
                        ),
                    );
                }
                ArgLocation::RegSequence(regs) => {
                    // HFA formal argument: the caller passed each FP member in
                    // consecutive typed FPR registers. For the callee, we copy
                    // the first register as a representative. Full HFA lowering
                    // would allocate stack space and store all members.
                    if let Some(first) = regs.first() {
                        self.emit(
                            entry_block,
                            ISelInst::new(
                                AArch64Opcode::Copy,
                                vec![ISelOperand::VReg(vreg), ISelOperand::PReg(*first)],
                            ),
                        );
                    }
                }
                ArgLocation::Indirect { ptr_reg } => {
                    // Large aggregate: load pointer from register, then load data
                    // For scaffold: just copy the pointer register
                    self.emit(
                        entry_block,
                        ISelInst::new(
                            AArch64Opcode::Copy,
                            vec![ISelOperand::VReg(vreg), ISelOperand::PReg(*ptr_reg)],
                        ),
                    );
                }
                ArgLocation::RegPair(lo_preg, hi_preg) => {
                    // i128 register pair: copy both halves from physical regs
                    // to virtual regs and define as i128 value pair.
                    let vreg_lo = self.new_vreg(RegClass::Gpr64);
                    let vreg_hi = self.new_vreg(RegClass::Gpr64);
                    self.emit(
                        entry_block,
                        ISelInst::new(
                            AArch64Opcode::Copy,
                            vec![ISelOperand::VReg(vreg_lo), ISelOperand::PReg(*lo_preg)],
                        ),
                    );
                    self.emit(
                        entry_block,
                        ISelInst::new(
                            AArch64Opcode::Copy,
                            vec![ISelOperand::VReg(vreg_hi), ISelOperand::PReg(*hi_preg)],
                        ),
                    );
                    self.define_i128_value(val, vreg_lo, vreg_hi);
                    continue; // skip the generic define_value below
                }
            }

            self.define_value(val, ISelOperand::VReg(vreg), ty.clone());
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Shift operations
    // -----------------------------------------------------------------------

    /// Select shift operation: LSL, LSR, ASR.
    ///
    /// If the shift amount is a constant (tracked as an Iconst in the value
    /// map), we emit the immediate form; otherwise, the register form.
    fn select_shift(
        &mut self,
        op: AArch64ShiftOp,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "Shift")?;
        Self::require_result(inst, "Shift")?;

        let src_val = inst.args[0];
        let amt_val = inst.args[1];
        let result_val = inst.results[0];

        let ty = self.value_type(&src_val);
        let is_32 = Self::is_32bit(&ty);
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let src = self.use_value(&src_val)?;
        let amt = self.use_value(&amt_val)?;

        // Check if shift amount is an immediate
        let is_imm = matches!(amt, ISelOperand::Imm(_));

        if is_imm {
            // Immediate shift form
            let opc = match (op, is_32) {
                (AArch64ShiftOp::Lsl, true) => AArch64Opcode::LslRI,
                (AArch64ShiftOp::Lsl, false) => AArch64Opcode::LslRI,
                (AArch64ShiftOp::Lsr, true) => AArch64Opcode::LsrRI,
                (AArch64ShiftOp::Lsr, false) => AArch64Opcode::LsrRI,
                (AArch64ShiftOp::Asr, true) => AArch64Opcode::AsrRI,
                (AArch64ShiftOp::Asr, false) => AArch64Opcode::AsrRI,
            };
            self.emit(
                block,
                ISelInst::new(opc, vec![ISelOperand::VReg(dst), src, amt]),
            );
        } else {
            // Register shift form
            let opc = match (op, is_32) {
                (AArch64ShiftOp::Lsl, true) => AArch64Opcode::LslRR,
                (AArch64ShiftOp::Lsl, false) => AArch64Opcode::LslRR,
                (AArch64ShiftOp::Lsr, true) => AArch64Opcode::LsrRR,
                (AArch64ShiftOp::Lsr, false) => AArch64Opcode::LsrRR,
                (AArch64ShiftOp::Asr, true) => AArch64Opcode::AsrRR,
                (AArch64ShiftOp::Asr, false) => AArch64Opcode::AsrRR,
            };
            self.emit(
                block,
                ISelInst::new(opc, vec![ISelOperand::VReg(dst), src, amt]),
            );
        }

        self.define_value(result_val, ISelOperand::VReg(dst), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Logical operations
    // -----------------------------------------------------------------------

    /// Select logical operation: AND, ORR, EOR, BIC, ORN.
    fn select_logic(
        &mut self,
        op: AArch64LogicOp,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "LogicOp")?;
        Self::require_result(inst, "LogicOp")?;

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let result_val = inst.results[0];

        let ty = self.value_type(&lhs_val);
        let is_32 = Self::is_32bit(&ty);
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        let opc = match (op, is_32) {
            (AArch64LogicOp::And, true) => AArch64Opcode::AndRR,
            (AArch64LogicOp::And, false) => AArch64Opcode::AndRR,
            (AArch64LogicOp::Orr, true) => AArch64Opcode::OrrRR,
            (AArch64LogicOp::Orr, false) => AArch64Opcode::OrrRR,
            (AArch64LogicOp::Eor, true) => AArch64Opcode::EorRR,
            (AArch64LogicOp::Eor, false) => AArch64Opcode::EorRR,
            (AArch64LogicOp::Bic, true) => AArch64Opcode::BicRR,
            (AArch64LogicOp::Bic, false) => AArch64Opcode::BicRR,
            (AArch64LogicOp::Orn, true) => AArch64Opcode::OrnRR,
            (AArch64LogicOp::Orn, false) => AArch64Opcode::OrnRR,
        };

        self.emit(
            block,
            ISelInst::new(opc, vec![ISelOperand::VReg(dst), lhs, rhs]),
        );

        self.define_value(result_val, ISelOperand::VReg(dst), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Extension operations
    // -----------------------------------------------------------------------

    /// Select zero/sign extension.
    ///
    /// Maps tMIR Sextend/Uextend to the appropriate AArch64 extension
    /// instruction. The AArch64 extension instructions are actually aliases
    /// of SBFM (sign) or UBFM (zero) with specific bit patterns.
    ///
    /// AArch64 extension instruction mapping:
    ///
    /// | From | To  | Signed | Instruction | Encoding |
    /// |------|-----|--------|-------------|----------|
    /// | I8   | I32 | yes    | SXTB Wd,Wn | SBFM Wd,Wn,#0,#7  |
    /// | I8   | I64 | yes    | SXTB Xd,Wn | SBFM Xd,Wn,#0,#7  |
    /// | I16  | I32 | yes    | SXTH Wd,Wn | SBFM Wd,Wn,#0,#15 |
    /// | I16  | I64 | yes    | SXTH Xd,Wn | SBFM Xd,Wn,#0,#15 |
    /// | I32  | I64 | yes    | SXTW Xd,Wn | SBFM Xd,Wn,#0,#31 |
    /// | I8   | I32 | no     | UXTB Wd,Wn | UBFM Wd,Wn,#0,#7  |
    /// | I8   | I64 | no     | UXTB Wd,Wn | UBFM Wd,Wn,#0,#7  |
    /// | I16  | I32 | no     | UXTH Wd,Wn | UBFM Wd,Wn,#0,#15 |
    /// | I16  | I64 | no     | UXTH Wd,Wn | UBFM Wd,Wn,#0,#15 |
    /// | I32  | I64 | no     | MOV Wd,Wn  | W-write zeroes top 32 |
    fn select_extend(
        &mut self,
        signed: bool,
        from_ty: &Type,
        to_ty: &Type,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "Extend")?;
        Self::require_result(inst, "Extend")?;

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let dst_class = reg_class_for_type(to_ty);
        let dst = self.new_vreg(dst_class);
        let src = self.use_value(&src_val)?;

        let to_64 = !Self::is_32bit(to_ty);
        let to_ty_owned = to_ty.clone();

        let opc = if signed {
            match (from_ty, to_64) {
                (Type::I8, false) => AArch64Opcode::Sxtb,
                (Type::I8, true) => AArch64Opcode::Sxtb,
                (Type::I16, false) => AArch64Opcode::Sxth,
                (Type::I16, true) => AArch64Opcode::Sxth,
                (Type::I32, true) => AArch64Opcode::Sxtw,
                _ => {
                    // Same-width or unsupported: just copy
                    self.emit(
                        block,
                        ISelInst::new(AArch64Opcode::Copy, vec![ISelOperand::VReg(dst), src]),
                    );
                    self.define_value(result_val, ISelOperand::VReg(dst), to_ty_owned);
                    return Ok(());
                }
            }
        } else {
            // Unsigned extension
            match from_ty {
                Type::I8 => AArch64Opcode::Uxtb, // AND Wd, Wn, #0xFF (UBFM Wd, Wn, #0, #7)
                Type::I16 => AArch64Opcode::Uxth, // AND Wd, Wn, #0xFFFF (UBFM Wd, Wn, #0, #15)
                Type::I32 if to_64 => {
                    // UXTW: zero-extend W to X. On AArch64, writing a W register
                    // implicitly zero-extends to X, so a MOV Wd, Wn suffices.
                    // But we need an explicit instruction for tracking.
                    self.emit(
                        block,
                        ISelInst::new(AArch64Opcode::MovR, vec![ISelOperand::VReg(dst), src]),
                    );
                    self.define_value(result_val, ISelOperand::VReg(dst), to_ty_owned);
                    return Ok(());
                }
                _ => {
                    self.emit(
                        block,
                        ISelInst::new(AArch64Opcode::Copy, vec![ISelOperand::VReg(dst), src]),
                    );
                    self.define_value(result_val, ISelOperand::VReg(dst), to_ty_owned);
                    return Ok(());
                }
            }
        };

        self.emit(block, ISelInst::new(opc, vec![ISelOperand::VReg(dst), src]));
        self.define_value(result_val, ISelOperand::VReg(dst), to_ty_owned);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Bitfield operations
    // -----------------------------------------------------------------------

    /// Select unsigned or signed bitfield extract.
    ///
    /// UBFM Wd, Wn, #immr, #imms  -- unsigned bitfield extract
    /// SBFM Wd, Wn, #immr, #imms  -- signed bitfield extract
    ///
    /// For extract at position `lsb` with `width` bits:
    ///   immr = lsb, imms = lsb + width - 1
    fn select_bitfield_extract(
        &mut self,
        signed: bool,
        lsb: u8,
        width: u8,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "BitfieldExtract")?;
        Self::require_result(inst, "BitfieldExtract")?;

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let ty = self.value_type(&src_val);
        let is_32 = Self::is_32bit(&ty);
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);
        let src = self.use_value(&src_val)?;

        let immr = lsb as i64;
        let imms = (lsb + width - 1) as i64;

        let opc = match (signed, is_32) {
            (false, true) => AArch64Opcode::Ubfm,
            (false, false) => AArch64Opcode::Ubfm,
            (true, true) => AArch64Opcode::Sbfm,
            (true, false) => AArch64Opcode::Sbfm,
        };

        self.emit(
            block,
            ISelInst::new(
                opc,
                vec![
                    ISelOperand::VReg(dst),
                    src,
                    ISelOperand::Imm(immr),
                    ISelOperand::Imm(imms),
                ],
            ),
        );

        self.define_value(result_val, ISelOperand::VReg(dst), ty);
        Ok(())
    }

    /// Select bitfield insert.
    ///
    /// BFM Wd, Wn, #immr, #imms -- insert `width` bits from Wn starting
    /// at bit `lsb` into Wd (which is also read).
    fn select_bitfield_insert(
        &mut self,
        lsb: u8,
        width: u8,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "BitfieldInsert")?;
        Self::require_result(inst, "BitfieldInsert")?;

        let dst_val = inst.args[0]; // destination (also source of unmodified bits)
        let src_val = inst.args[1]; // source of bits to insert

        let ty = self.value_type(&dst_val);
        let is_32 = Self::is_32bit(&ty);
        let class = reg_class_for_type(&ty);
        let result_val = inst.results[0];

        // BFM reads and writes the destination register. We model this by
        // first copying dst to a fresh vreg, then BFM modifying it in place.
        let result = self.new_vreg(class);
        let dst_op = self.use_value(&dst_val)?;
        let src_op = self.use_value(&src_val)?;

        // Copy dst to result (BFM operates on Wd as both read and write)
        self.emit(
            block,
            ISelInst::new(AArch64Opcode::Copy, vec![ISelOperand::VReg(result), dst_op]),
        );

        // BFM encoding for insert at position lsb with width:
        //   immr = (reg_size - lsb) % reg_size
        //   imms = width - 1
        let reg_size = if is_32 { 32 } else { 64 };
        let immr = (reg_size - lsb as i64) % reg_size;
        let imms = (width - 1) as i64;

        let opc = if is_32 {
            AArch64Opcode::Bfm
        } else {
            AArch64Opcode::Bfm
        };

        self.emit(
            block,
            ISelInst::new(
                opc,
                vec![
                    ISelOperand::VReg(result),
                    src_op,
                    ISelOperand::Imm(immr),
                    ISelOperand::Imm(imms),
                ],
            ),
        );

        self.define_value(result_val, ISelOperand::VReg(result), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Conditional select
    // -----------------------------------------------------------------------

    /// Select conditional select (CSEL).
    ///
    /// tMIR: Select { cond } (cc_val, true_val, false_val) -> result
    /// The cc_val is a comparison result (must have set NZCV flags first).
    /// We emit: CMP cc_val, #0; CSEL Wd/Xd, Wn/Xn, Wm/Xm, NE
    fn select_csel(
        &mut self,
        cond: IntCC,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 3, "Select")?;
        Self::require_result(inst, "Select")?;

        let cond_val = inst.args[0];
        let true_val = inst.args[1];
        let false_val = inst.args[2];
        let result_val = inst.results[0];

        let ty = self.value_type(&true_val);
        let is_32 = Self::is_32bit(&ty);
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let cond_op = self.use_value(&cond_val)?;
        let true_op = self.use_value(&true_val)?;
        let false_op = self.use_value(&false_val)?;

        // First, test the condition value (CMP cond, #0)
        self.emit(
            block,
            ISelInst::new(AArch64Opcode::CmpRI, vec![cond_op, ISelOperand::Imm(0)]),
        );

        // CSEL based on the IntCC condition
        let cc = AArch64CC::from_intcc(cond);
        let opc = if is_32 {
            AArch64Opcode::Csel
        } else {
            AArch64Opcode::Csel
        };

        self.emit(
            block,
            ISelInst::new(
                opc,
                vec![
                    ISelOperand::VReg(dst),
                    true_op,
                    false_op,
                    ISelOperand::CondCode(cc),
                ],
            ),
        );

        self.define_value(result_val, ISelOperand::VReg(dst), ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Floating-point arithmetic
    // -----------------------------------------------------------------------

    /// Select floating-point binary operation.
    fn select_fp_binop(
        &mut self,
        op: AArch64FpBinOp,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "FpBinOp")?;
        Self::require_result(inst, "FpBinOp")?;

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let result_val = inst.results[0];

        let ty = self.value_type(&lhs_val);
        let is_f32 = matches!(ty, Type::F32);
        let class = reg_class_for_type(&ty);
        let dst = self.new_vreg(class);

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        let opc = match (op, is_f32) {
            (AArch64FpBinOp::Fadd, true) => AArch64Opcode::FaddRR,
            (AArch64FpBinOp::Fadd, false) => AArch64Opcode::FaddRR,
            (AArch64FpBinOp::Fsub, true) => AArch64Opcode::FsubRR,
            (AArch64FpBinOp::Fsub, false) => AArch64Opcode::FsubRR,
            (AArch64FpBinOp::Fmul, true) => AArch64Opcode::FmulRR,
            (AArch64FpBinOp::Fmul, false) => AArch64Opcode::FmulRR,
            (AArch64FpBinOp::Fdiv, true) => AArch64Opcode::FdivRR,
            (AArch64FpBinOp::Fdiv, false) => AArch64Opcode::FdivRR,
        };

        self.emit(
            block,
            ISelInst::new(opc, vec![ISelOperand::VReg(dst), lhs, rhs]),
        );

        self.define_value(result_val, ISelOperand::VReg(dst), ty);
        Ok(())
    }

    /// Select floating-point comparison: FCMP + CSET (+ CSINC for two-CC cases).
    ///
    /// Most float conditions map to a single AArch64 condition code after FCMP.
    /// `UnorderedEqual` (EQ || VS) needs a two-instruction materialization:
    ///   CSET tmp, EQ; CSINC dst, tmp, WZR, VS
    /// When VS (NaN): dst = WZR+1 = 1. When !VS: dst = tmp (1 if EQ, 0 otherwise).
    fn select_fcmp(
        &mut self,
        cond: FloatCC,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "Fcmp")?;
        Self::require_result(inst, "Fcmp")?;

        let lhs_val = inst.args[0];
        let rhs_val = inst.args[1];
        let result_val = inst.results[0];

        let ty = self.value_type(&lhs_val);
        let is_f32 = matches!(ty, Type::F32);

        let lhs = self.use_value(&lhs_val)?;
        let rhs = self.use_value(&rhs_val)?;

        // FCMP sets NZCV
        let cmp_opc = if is_f32 {
            AArch64Opcode::Fcmp
        } else {
            AArch64Opcode::Fcmp
        };
        self.emit(block, ISelInst::new(cmp_opc, vec![lhs, rhs]));

        // UnorderedEqual needs two condition codes: EQ || VS.
        // Emit: CSET tmp, EQ; CSINC dst, tmp, WZR, VS
        // Use Gpr64 to match select_icmp and avoid W/X aliasing (issue #335).
        if cond == FloatCC::UnorderedEqual {
            let tmp = self.new_vreg(RegClass::Gpr64);
            self.emit(
                block,
                ISelInst::new(
                    AArch64Opcode::CSet,
                    vec![ISelOperand::VReg(tmp), ISelOperand::CondCode(AArch64CC::EQ)],
                ),
            );
            let dst = self.new_vreg(RegClass::Gpr64);
            self.emit(
                block,
                ISelInst::new(
                    AArch64Opcode::Csinc,
                    vec![
                        ISelOperand::VReg(dst),
                        ISelOperand::VReg(tmp),
                        ISelOperand::Imm(0), // WZR
                        ISelOperand::CondCode(AArch64CC::VS),
                    ],
                ),
            );
            self.define_value(result_val, ISelOperand::VReg(dst), Type::B1);
            return Ok(());
        }

        // Single-CC case: CSET to materialize result.
        // Use Gpr64 to match select_icmp and avoid W/X register aliasing
        // issues with the linear scan allocator (issue #335).
        let cc = AArch64CC::from_floatcc(cond);
        let dst = self.new_vreg(RegClass::Gpr64);
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::CSet,
                vec![ISelOperand::VReg(dst), ISelOperand::CondCode(cc)],
            ),
        );

        self.define_value(result_val, ISelOperand::VReg(dst), Type::B1);
        Ok(())
    }

    /// Select float-to-integer conversion (FCVTZS).
    ///
    /// FCVTZS rounds toward zero (truncation), matching C cast semantics.
    fn select_fcvt_to_int(
        &mut self,
        dst_ty: Type,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "FcvtToInt")?;
        Self::require_result(inst, "FcvtToInt")?;

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let src_ty = self.value_type(&src_val);
        let src = self.use_value(&src_val)?;

        let dst_class = reg_class_for_type(&dst_ty);
        let dst = self.new_vreg(dst_class);

        // Select based on destination integer width. The source float
        // precision is encoded in the operand register class.
        let opc = if Self::is_32bit(&dst_ty) {
            AArch64Opcode::FcvtzsRR
        } else {
            AArch64Opcode::FcvtzsRR
        };

        // Operands: [dst, src, src_type_hint]
        // The src_type_hint helps the encoder know if source is S or D reg.
        let src_hint = if matches!(src_ty, Type::F32) {
            32i64
        } else {
            64
        };
        self.emit(
            block,
            ISelInst::new(
                opc,
                vec![ISelOperand::VReg(dst), src, ISelOperand::Imm(src_hint)],
            ),
        );

        self.define_value(result_val, ISelOperand::VReg(dst), dst_ty);
        Ok(())
    }

    /// Select integer-to-float conversion (SCVTF).
    fn select_fcvt_from_int(
        &mut self,
        src_ty: Type,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 1, "FcvtFromInt")?;
        Self::require_result(inst, "FcvtFromInt")?;

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        // Destination type is the result's FP type; we infer from context.
        // The src_ty tells us the integer width of the source.
        let src = self.use_value(&src_val)?;

        // We need to know the destination float type. We default to F64 unless
        // the result type is tracked. For now, use the convention that
        // FcvtFromInt produces a value of the same "size class" as src_ty:
        //   I32 -> F32, I64 -> F64
        let dst_ty = if Self::is_32bit(&src_ty) {
            Type::F32
        } else {
            Type::F64
        };

        let dst_class = reg_class_for_type(&dst_ty);
        let dst = self.new_vreg(dst_class);

        let opc = match (Self::is_32bit(&src_ty), matches!(dst_ty, Type::F32)) {
            (true, true) => AArch64Opcode::ScvtfRR,   // SCVTF Sd, Wn
            (true, false) => AArch64Opcode::ScvtfRR,  // SCVTF Dd, Wn
            (false, true) => AArch64Opcode::ScvtfRR,  // SCVTF Sd, Xn
            (false, false) => AArch64Opcode::ScvtfRR, // SCVTF Dd, Xn
        };

        self.emit(block, ISelInst::new(opc, vec![ISelOperand::VReg(dst), src]));

        self.define_value(result_val, ISelOperand::VReg(dst), dst_ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Unsigned FP conversions
    // -----------------------------------------------------------------------

    /// Select float-to-unsigned-integer conversion (FCVTZU).
    ///
    /// FCVTZU rounds toward zero (truncation), like FCVTZS but for unsigned.
    fn select_fcvt_to_uint(
        &mut self,
        dst_ty: Type,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        assert!(!inst.args.is_empty(), "FcvtToUint must have 1 arg");
        assert!(!inst.results.is_empty(), "FcvtToUint must have a result");

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let src_ty = self.value_type(&src_val);
        let src = self.use_value(&src_val)?;

        let dst_class = reg_class_for_type(&dst_ty);
        let dst = self.new_vreg(dst_class);

        let opc = if Self::is_32bit(&dst_ty) {
            AArch64Opcode::FcvtzuRR
        } else {
            AArch64Opcode::FcvtzuRR
        };

        let src_hint = if matches!(src_ty, Type::F32) {
            32i64
        } else {
            64
        };
        self.emit(
            block,
            ISelInst::new(
                opc,
                vec![ISelOperand::VReg(dst), src, ISelOperand::Imm(src_hint)],
            ),
        );

        self.define_value(result_val, ISelOperand::VReg(dst), dst_ty);
        Ok(())
    }

    /// Select unsigned-integer-to-float conversion (UCVTF).
    fn select_fcvt_from_uint(
        &mut self,
        src_ty: Type,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        assert!(!inst.args.is_empty(), "FcvtFromUint must have 1 arg");
        assert!(!inst.results.is_empty(), "FcvtFromUint must have a result");

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let src = self.use_value(&src_val)?;

        // Same convention as signed: I32 -> F32, I64 -> F64
        let dst_ty = if Self::is_32bit(&src_ty) {
            Type::F32
        } else {
            Type::F64
        };

        let dst_class = reg_class_for_type(&dst_ty);
        let dst = self.new_vreg(dst_class);

        let opc = match (Self::is_32bit(&src_ty), matches!(dst_ty, Type::F32)) {
            (true, true) => AArch64Opcode::UcvtfRR,   // UCVTF Sd, Wn
            (true, false) => AArch64Opcode::UcvtfRR,  // UCVTF Dd, Wn
            (false, true) => AArch64Opcode::UcvtfRR,  // UCVTF Sd, Xn
            (false, false) => AArch64Opcode::UcvtfRR, // UCVTF Dd, Xn
        };

        self.emit(block, ISelInst::new(opc, vec![ISelOperand::VReg(dst), src]));

        self.define_value(result_val, ISelOperand::VReg(dst), dst_ty);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Float precision conversion
    // -----------------------------------------------------------------------

    /// Select float precision widening: f32 -> f64 (FCVT Dd, Sn).
    fn select_fp_ext(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        assert!(!inst.args.is_empty(), "FPExt must have 1 arg");
        assert!(!inst.results.is_empty(), "FPExt must have a result");

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let src = self.use_value(&src_val)?;

        let dst = self.new_vreg(RegClass::Fpr64);

        self.emit(
            block,
            ISelInst::new(AArch64Opcode::FcvtSD, vec![ISelOperand::VReg(dst), src]),
        );

        self.define_value(result_val, ISelOperand::VReg(dst), Type::F64);
        Ok(())
    }

    /// Select float precision narrowing: f64 -> f32 (FCVT Ss, Dn).
    fn select_fp_trunc(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        assert!(!inst.args.is_empty(), "FPTrunc must have 1 arg");
        assert!(!inst.results.is_empty(), "FPTrunc must have a result");

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let src = self.use_value(&src_val)?;

        let dst = self.new_vreg(RegClass::Fpr32);

        self.emit(
            block,
            ISelInst::new(AArch64Opcode::FcvtDS, vec![ISelOperand::VReg(dst), src]),
        );

        self.define_value(result_val, ISelOperand::VReg(dst), Type::F32);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Integer truncation
    // -----------------------------------------------------------------------

    /// Select integer truncation (narrow: i64->i32, i32->i16, etc.).
    ///
    /// On AArch64, truncation strategy depends on target width:
    ///
    /// - i64 -> i32: MOV Wd, Wn (upper 32 bits ignored, W-register write
    ///   implicitly zero-extends)
    /// - i32/i64 -> i16: AND Wd, Wn, #0xFFFF (mask to 16 bits)
    /// - i32/i64 -> i8:  AND Wd, Wn, #0xFF   (mask to 8 bits)
    ///
    /// Sub-32-bit truncation requires an explicit AND to ensure the upper
    /// bits are cleared. A plain MOV would leave stale bits that could
    /// affect subsequent operations expecting a clean narrow value.
    fn select_trunc(
        &mut self,
        to_ty: &Type,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        assert!(!inst.args.is_empty(), "Trunc must have 1 arg");
        assert!(!inst.results.is_empty(), "Trunc must have a result");

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let src = self.use_value(&src_val)?;

        let dst_class = reg_class_for_type(to_ty);
        let dst = self.new_vreg(dst_class);

        match to_ty {
            Type::I8 => {
                // Trunc to i8: AND Wd, Wn, #0xFF
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::AndRI,
                        vec![ISelOperand::VReg(dst), src, ISelOperand::Imm(0xFF)],
                    ),
                );
            }
            Type::I16 => {
                // Trunc to i16: AND Wd, Wn, #0xFFFF
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::AndRI,
                        vec![ISelOperand::VReg(dst), src, ISelOperand::Imm(0xFFFF)],
                    ),
                );
            }
            _ => {
                // Trunc to i32 (from i64): MOV Wd, Wn — upper bits ignored
                self.emit(
                    block,
                    ISelInst::new(AArch64Opcode::MovR, vec![ISelOperand::VReg(dst), src]),
                );
            }
        }

        self.define_value(result_val, ISelOperand::VReg(dst), to_ty.clone());
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Bitcast
    // -----------------------------------------------------------------------

    /// Select bitcast: reinterpret bits between same-size types.
    ///
    /// On AArch64, bitcast between GPR and FPR uses FMOV:
    /// - GPR -> FPR: FMOV Sd, Wn (32-bit) or FMOV Dd, Xn (64-bit)
    /// - FPR -> GPR: FMOV Wd, Sn (32-bit) or FMOV Xd, Dn (64-bit)
    /// - Same register class: plain MOV (e.g., i32 -> u32, no-op in our IR)
    fn select_bitcast(
        &mut self,
        to_ty: &Type,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        assert!(!inst.args.is_empty(), "Bitcast must have 1 arg");
        assert!(!inst.results.is_empty(), "Bitcast must have a result");

        let src_val = inst.args[0];
        let result_val = inst.results[0];

        let src_ty = self.value_type(&src_val);
        let src = self.use_value(&src_val)?;

        let src_is_fp = matches!(src_ty, Type::F32 | Type::F64);
        let dst_is_fp = matches!(to_ty, Type::F32 | Type::F64);

        let dst_class = reg_class_for_type(to_ty);
        let dst = self.new_vreg(dst_class);

        let opc = match (src_is_fp, dst_is_fp) {
            // GPR -> FPR (integer bits reinterpreted as float)
            (false, true) => {
                if Self::is_32bit(to_ty) {
                    AArch64Opcode::FmovGprFpr // FMOV Sd, Wn
                } else {
                    AArch64Opcode::FmovGprFpr // FMOV Dd, Xn
                }
            }
            // FPR -> GPR (float bits reinterpreted as integer)
            (true, false) => {
                if Self::is_32bit(&src_ty) {
                    AArch64Opcode::FmovFprGpr // FMOV Wd, Sn
                } else {
                    AArch64Opcode::FmovFprGpr // FMOV Xd, Dn
                }
            }
            // Same class: plain MOV
            _ => {
                if Self::is_32bit(to_ty) {
                    AArch64Opcode::MovR
                } else {
                    AArch64Opcode::MovR
                }
            }
        };

        self.emit(block, ISelInst::new(opc, vec![ISelOperand::VReg(dst), src]));

        self.define_value(result_val, ISelOperand::VReg(dst), to_ty.clone());
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Addressing
    // -----------------------------------------------------------------------

    /// Select global symbol reference.
    ///
    /// On AArch64, global references use a two-instruction sequence:
    ///   ADRP Xd, #page(symbol)      -- load 4KB-aligned page address
    ///   ADD  Xd, Xd, #pageoff(symbol) -- add 12-bit page offset
    fn select_global_ref(
        &mut self,
        name: &str,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_result(inst, "GlobalRef")?;

        let result_val = inst.results[0];
        let dst = self.new_vreg(RegClass::Gpr64);

        // ADRP Xd, symbol@PAGE
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Adrp,
                vec![
                    ISelOperand::VReg(dst),
                    ISelOperand::Symbol(name.to_string()),
                ],
            ),
        );

        // ADD Xd, Xd, symbol@PAGEOFF
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::AddPCRel,
                vec![
                    ISelOperand::VReg(dst),
                    ISelOperand::VReg(dst),
                    ISelOperand::Symbol(name.to_string()),
                ],
            ),
        );

        self.define_value(result_val, ISelOperand::VReg(dst), Type::I64);
        Ok(())
    }

    /// Select external symbol reference via GOT (Global Offset Table).
    ///
    /// External symbols (from shared libraries or other translation units)
    /// require GOT-indirect access on Darwin/AArch64:
    ///   ADRP  Xd, symbol@GOTPAGE       -- load GOT page address
    ///   LDR   Xd, [Xd, symbol@GOTPAGEOFF]  -- load pointer from GOT slot
    ///
    /// This is different from local GlobalRef which uses ADRP+ADD (direct).
    /// The GOT slot contains the actual address of the symbol, resolved by
    /// the dynamic linker at load time.
    fn select_extern_ref(
        &mut self,
        name: &str,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        assert!(!inst.results.is_empty(), "ExternRef must have a result");

        let result_val = inst.results[0];
        let dst = self.new_vreg(RegClass::Gpr64);

        // ADRP Xd, symbol@GOTPAGE
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Adrp,
                vec![
                    ISelOperand::VReg(dst),
                    ISelOperand::Symbol(name.to_string()),
                ],
            ),
        );

        // LDR Xd, [Xd, symbol@GOTPAGEOFF]
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::LdrGot,
                vec![
                    ISelOperand::VReg(dst),
                    ISelOperand::VReg(dst),
                    ISelOperand::Symbol(name.to_string()),
                ],
            ),
        );

        self.define_value(result_val, ISelOperand::VReg(dst), Type::I64);
        Ok(())
    }

    /// Select thread-local variable reference via the configured TLS model.
    ///
    /// `LocalExec` materializes the address from TPIDR_EL0 plus a pre-resolved
    /// TPREL offset. Other models preserve the legacy Darwin TLV descriptor
    /// sequence until those paths are wired end-to-end.
    fn select_tls_ref(
        &mut self,
        name: &str,
        model: TlsModel,
        local_exec_offset: Option<u32>,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_result(inst, "TlsRef")?;

        let result_val = inst.results[0];
        let dst = self.new_vreg(RegClass::Gpr64);

        if model == TlsModel::LocalExec {
            let off = local_exec_offset.ok_or(ISelError::LocalExecTlsRefMissingOffset)?;
            if off > 0xFF_FFFF {
                return Err(ISelError::LocalExecTlsRefOffsetTooLarge);
            }

            self.emit(
                block,
                ISelInst::new(
                    AArch64Opcode::Mrs,
                    vec![ISelOperand::VReg(dst), ISelOperand::Imm(TPIDR_EL0_SYSREG)],
                ),
            );

            let hi12 = ((off >> 12) & 0xFFF) as i64;
            let lo12 = (off & 0xFFF) as i64;

            if hi12 != 0 {
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::AddRIShift12,
                        vec![
                            ISelOperand::VReg(dst),
                            ISelOperand::VReg(dst),
                            ISelOperand::Imm(hi12),
                        ],
                    ),
                );
            }

            if lo12 != 0 {
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::AddRI,
                        vec![
                            ISelOperand::VReg(dst),
                            ISelOperand::VReg(dst),
                            ISelOperand::Imm(lo12),
                        ],
                    ),
                );
            }
        } else {
            // ADRP Xd, symbol@TLVPPAGE
            self.emit(
                block,
                ISelInst::new(
                    AArch64Opcode::Adrp,
                    vec![
                        ISelOperand::VReg(dst),
                        ISelOperand::Symbol(name.to_string()),
                    ],
                ),
            );

            // LDR Xd, [Xd, symbol@TLVPPAGEOFF]
            self.emit(
                block,
                ISelInst::new(
                    AArch64Opcode::LdrTlvp,
                    vec![
                        ISelOperand::VReg(dst),
                        ISelOperand::VReg(dst),
                        ISelOperand::Symbol(name.to_string()),
                    ],
                ),
            );
        }

        self.define_value(result_val, ISelOperand::VReg(dst), Type::I64);
        Ok(())
    }

    /// Select stack slot address computation.
    ///
    /// Emits ADD Xd, SP, #offset (the actual offset is resolved during
    /// frame lowering; we emit a placeholder StackSlot operand).
    fn select_stack_addr(
        &mut self,
        slot: u32,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_result(inst, "StackAddr")?;

        let result_val = inst.results[0];
        let dst = self.new_vreg(RegClass::Gpr64);

        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::AddPCRel,
                vec![
                    ISelOperand::VReg(dst),
                    ISelOperand::PReg(SP), // SP
                    ISelOperand::StackSlot(slot),
                ],
            ),
        );

        self.define_value(result_val, ISelOperand::VReg(dst), Type::I64);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Finalize
    // -----------------------------------------------------------------------

    /// Define block parameter Values for a non-entry block.
    ///
    /// tMIR uses block parameters for SSA phi semantics. Before processing
    /// instructions in a block that has parameters, the ISel must have VRegs
    /// allocated for those parameter Values. This method creates a fresh VReg
    /// for each block parameter and records the mapping.
    ///
    /// Entry block parameters should be handled by `lower_formal_arguments`
    /// instead, which also emits COPY instructions from physical registers.
    pub fn define_block_params(&mut self, params: &[(Value, Type)]) {
        for (val, ty) in params {
            // Skip values already defined by copies in predecessor blocks.
            // The adapter emits single-arg Iadd (copy) instructions in
            // predecessor blocks that define block parameter values before
            // the target block is processed.
            if self.value_map.contains_key(val) {
                continue;
            }
            let class = reg_class_for_type(ty);
            let vreg = self.new_vreg(class);
            self.define_value(*val, ISelOperand::VReg(vreg), ty.clone());
        }
    }

    /// Consume the selector and return the completed ISelFunction.
    pub fn finalize(self) -> ISelFunction {
        self.func
    }

    /// Consume the selector and return both the completed ISelFunction
    /// and the value-to-operand mapping.
    ///
    /// The value map is needed by the pipeline to propagate proof
    /// annotations from the adapter's [`ProofContext`](crate::adapter::ProofContext)
    /// onto the machine instructions that define each value.
    pub fn finalize_with_value_map(self) -> (ISelFunction, HashMap<Value, ISelOperand>) {
        (self.func, self.value_map)
    }

    // -------------------------------------------------------------------
    // i128 multi-register arithmetic lowering
    // -------------------------------------------------------------------

    /// Record an i128 value as a pair of GPR64 virtual registers.
    ///
    /// The low half is tracked in `value_map` (via `define_value`).
    /// The high half is tracked separately in `i128_high_map`.
    fn define_i128_value(&mut self, val: Value, lo: VReg, hi: VReg) {
        self.define_value(val, ISelOperand::VReg(lo), Type::I128);
        self.i128_high_map.insert(val, hi);
    }

    /// Look up both halves of an i128 value.
    ///
    /// Returns `(lo_operand, hi_operand)` where both are GPR64 VRegs.
    fn use_i128_value(&self, val: &Value) -> Result<(ISelOperand, ISelOperand), ISelError> {
        let lo = self.use_value(val)?;
        let hi = self
            .i128_high_map
            .get(val)
            .copied()
            .ok_or(ISelError::UndefinedValue(*val))?;
        Ok((lo, ISelOperand::VReg(hi)))
    }

    /// Select i128 addition: ADDS on low half (sets carry), ADC on high half (reads carry).
    ///
    /// ```text
    /// dst_lo = ADDS lhs_lo, rhs_lo   // sets C flag
    /// dst_hi = ADC  lhs_hi, rhs_hi   // reads C flag
    /// ```
    fn select_i128_add(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "i128 add")?;
        Self::require_result(inst, "i128 add")?;

        let (lhs_lo, lhs_hi) = self.use_i128_value(&inst.args[0])?;
        let (rhs_lo, rhs_hi) = self.use_i128_value(&inst.args[1])?;

        let dst_lo = self.new_vreg(RegClass::Gpr64);
        let dst_hi = self.new_vreg(RegClass::Gpr64);

        // ADDS: low half with flag-setting
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::AddsRR,
                vec![ISelOperand::VReg(dst_lo), lhs_lo, rhs_lo],
            ),
        );
        // ADC: high half reads carry from ADDS
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Adc,
                vec![ISelOperand::VReg(dst_hi), lhs_hi, rhs_hi],
            ),
        );

        self.define_i128_value(inst.results[0], dst_lo, dst_hi);
        Ok(())
    }

    /// Select i128 subtraction: SUBS on low half (sets borrow), SBC on high half (reads borrow).
    ///
    /// ```text
    /// dst_lo = SUBS lhs_lo, rhs_lo   // sets C flag (inverted borrow)
    /// dst_hi = SBC  lhs_hi, rhs_hi   // reads C flag
    /// ```
    fn select_i128_sub(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "i128 sub")?;
        Self::require_result(inst, "i128 sub")?;

        let (lhs_lo, lhs_hi) = self.use_i128_value(&inst.args[0])?;
        let (rhs_lo, rhs_hi) = self.use_i128_value(&inst.args[1])?;

        let dst_lo = self.new_vreg(RegClass::Gpr64);
        let dst_hi = self.new_vreg(RegClass::Gpr64);

        // SUBS: low half with flag-setting
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::SubsRR,
                vec![ISelOperand::VReg(dst_lo), lhs_lo, rhs_lo],
            ),
        );
        // SBC: high half reads borrow from SUBS
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Sbc,
                vec![ISelOperand::VReg(dst_hi), lhs_hi, rhs_hi],
            ),
        );

        self.define_i128_value(inst.results[0], dst_lo, dst_hi);
        Ok(())
    }

    /// Select i128 multiplication using MUL, UMULH, and cross-term MADDs.
    ///
    /// For `a = (a_hi : a_lo)` and `b = (b_hi : b_lo)`, the low 128 bits of the product:
    /// ```text
    /// dst_lo = MUL   a_lo, b_lo                   // low 64 bits of a_lo * b_lo
    /// t0     = UMULH  a_lo, b_lo                   // high 64 bits of a_lo * b_lo
    /// t1     = MADD   t1, a_lo, b_hi, t0           // t0 + a_lo * b_hi
    /// dst_hi = MADD   dst_hi, a_hi, b_lo, t1       // t1 + a_hi * b_lo
    /// ```
    fn select_i128_mul(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "i128 mul")?;
        Self::require_result(inst, "i128 mul")?;

        let (lhs_lo, lhs_hi) = self.use_i128_value(&inst.args[0])?;
        let (rhs_lo, rhs_hi) = self.use_i128_value(&inst.args[1])?;

        let dst_lo = self.new_vreg(RegClass::Gpr64);
        let t0 = self.new_vreg(RegClass::Gpr64);
        let t1 = self.new_vreg(RegClass::Gpr64);
        let dst_hi = self.new_vreg(RegClass::Gpr64);

        // dst_lo = MUL a_lo, b_lo
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::MulRR,
                vec![ISelOperand::VReg(dst_lo), lhs_lo.clone(), rhs_lo.clone()],
            ),
        );
        // t0 = UMULH a_lo, b_lo
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Umulh,
                vec![ISelOperand::VReg(t0), lhs_lo.clone(), rhs_lo.clone()],
            ),
        );
        // t1 = MADD t1, a_lo, b_hi, t0  =>  t0 + a_lo * b_hi
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Madd,
                vec![ISelOperand::VReg(t1), lhs_lo, rhs_hi, ISelOperand::VReg(t0)],
            ),
        );
        // dst_hi = MADD dst_hi, a_hi, b_lo, t1  =>  t1 + a_hi * b_lo
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Madd,
                vec![
                    ISelOperand::VReg(dst_hi),
                    lhs_hi,
                    rhs_lo,
                    ISelOperand::VReg(t1),
                ],
            ),
        );

        self.define_i128_value(inst.results[0], dst_lo, dst_hi);
        Ok(())
    }

    /// Select i128 comparison by comparing high halves first, then low halves.
    ///
    /// For EQ/NE: compare both halves and combine with AND/ORR.
    /// For ordered comparisons: if high halves differ, the result is determined
    /// by the high comparison. If equal, the low comparison (always unsigned)
    /// determines the result.
    fn select_i128_cmp(
        &mut self,
        cond: IntCC,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "i128 cmp")?;
        Self::require_result(inst, "i128 cmp")?;

        let (lhs_lo, lhs_hi) = self.use_i128_value(&inst.args[0])?;
        let (rhs_lo, rhs_hi) = self.use_i128_value(&inst.args[1])?;
        let result_val = inst.results[0];

        match cond {
            IntCC::Equal => {
                // CMP hi; CSET hi_eq, EQ; CMP lo; CSET lo_eq, EQ; AND result, hi_eq, lo_eq
                self.emit(
                    block,
                    ISelInst::new(AArch64Opcode::CmpRR, vec![lhs_hi, rhs_hi]),
                );
                let hi_eq = self.new_vreg(RegClass::Gpr64);
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::CSet,
                        vec![
                            ISelOperand::VReg(hi_eq),
                            ISelOperand::CondCode(AArch64CC::EQ),
                        ],
                    ),
                );
                self.emit(
                    block,
                    ISelInst::new(AArch64Opcode::CmpRR, vec![lhs_lo, rhs_lo]),
                );
                let lo_eq = self.new_vreg(RegClass::Gpr64);
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::CSet,
                        vec![
                            ISelOperand::VReg(lo_eq),
                            ISelOperand::CondCode(AArch64CC::EQ),
                        ],
                    ),
                );
                let dst = self.new_vreg(RegClass::Gpr64);
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::AndRR,
                        vec![
                            ISelOperand::VReg(dst),
                            ISelOperand::VReg(hi_eq),
                            ISelOperand::VReg(lo_eq),
                        ],
                    ),
                );
                self.define_value(result_val, ISelOperand::VReg(dst), Type::B1);
            }
            IntCC::NotEqual => {
                // CMP hi; CSET hi_ne, NE; CMP lo; CSET lo_ne, NE; ORR result, hi_ne, lo_ne
                self.emit(
                    block,
                    ISelInst::new(AArch64Opcode::CmpRR, vec![lhs_hi, rhs_hi]),
                );
                let hi_ne = self.new_vreg(RegClass::Gpr64);
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::CSet,
                        vec![
                            ISelOperand::VReg(hi_ne),
                            ISelOperand::CondCode(AArch64CC::NE),
                        ],
                    ),
                );
                self.emit(
                    block,
                    ISelInst::new(AArch64Opcode::CmpRR, vec![lhs_lo, rhs_lo]),
                );
                let lo_ne = self.new_vreg(RegClass::Gpr64);
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::CSet,
                        vec![
                            ISelOperand::VReg(lo_ne),
                            ISelOperand::CondCode(AArch64CC::NE),
                        ],
                    ),
                );
                let dst = self.new_vreg(RegClass::Gpr64);
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::OrrRR,
                        vec![
                            ISelOperand::VReg(dst),
                            ISelOperand::VReg(hi_ne),
                            ISelOperand::VReg(lo_ne),
                        ],
                    ),
                );
                self.define_value(result_val, ISelOperand::VReg(dst), Type::B1);
            }
            _ => {
                // Ordered comparison: hi decides when strictly ordered; low decides when hi equal.
                // High comparison uses the original signedness; low comparison is always unsigned.
                let hi_cc = match cond {
                    IntCC::SignedLessThan | IntCC::SignedLessThanOrEqual => AArch64CC::LT,
                    IntCC::SignedGreaterThan | IntCC::SignedGreaterThanOrEqual => AArch64CC::GT,
                    IntCC::UnsignedLessThan | IntCC::UnsignedLessThanOrEqual => AArch64CC::LO,
                    IntCC::UnsignedGreaterThan | IntCC::UnsignedGreaterThanOrEqual => AArch64CC::HI,
                    _ => unreachable!(),
                };
                let lo_cc = match cond {
                    IntCC::SignedLessThan | IntCC::UnsignedLessThan => AArch64CC::LO,
                    IntCC::SignedGreaterThan | IntCC::UnsignedGreaterThan => AArch64CC::HI,
                    IntCC::SignedLessThanOrEqual | IntCC::UnsignedLessThanOrEqual => AArch64CC::LS,
                    IntCC::SignedGreaterThanOrEqual | IntCC::UnsignedGreaterThanOrEqual => {
                        AArch64CC::HS
                    }
                    _ => unreachable!(),
                };

                // CMP hi_l, hi_r
                self.emit(
                    block,
                    ISelInst::new(AArch64Opcode::CmpRR, vec![lhs_hi, rhs_hi]),
                );
                let hi_cond = self.new_vreg(RegClass::Gpr64);
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::CSet,
                        vec![ISelOperand::VReg(hi_cond), ISelOperand::CondCode(hi_cc)],
                    ),
                );
                let hi_eq = self.new_vreg(RegClass::Gpr64);
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::CSet,
                        vec![
                            ISelOperand::VReg(hi_eq),
                            ISelOperand::CondCode(AArch64CC::EQ),
                        ],
                    ),
                );

                // CMP lo_l, lo_r (unsigned)
                self.emit(
                    block,
                    ISelInst::new(AArch64Opcode::CmpRR, vec![lhs_lo, rhs_lo]),
                );
                let lo_cond = self.new_vreg(RegClass::Gpr64);
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::CSet,
                        vec![ISelOperand::VReg(lo_cond), ISelOperand::CondCode(lo_cc)],
                    ),
                );

                // result = hi_cond | (hi_eq & lo_cond)
                let tmp = self.new_vreg(RegClass::Gpr64);
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::AndRR,
                        vec![
                            ISelOperand::VReg(tmp),
                            ISelOperand::VReg(hi_eq),
                            ISelOperand::VReg(lo_cond),
                        ],
                    ),
                );
                let dst = self.new_vreg(RegClass::Gpr64);
                self.emit(
                    block,
                    ISelInst::new(
                        AArch64Opcode::OrrRR,
                        vec![
                            ISelOperand::VReg(dst),
                            ISelOperand::VReg(hi_cond),
                            ISelOperand::VReg(tmp),
                        ],
                    ),
                );
                self.define_value(result_val, ISelOperand::VReg(dst), Type::B1);
            }
        }

        Ok(())
    }

    // -------------------------------------------------------------------
    // i128 constant materialization
    // -------------------------------------------------------------------

    /// Select i128 constant: split into two 64-bit halves and materialize each
    /// with a MOVZ+MOVK sequence.
    ///
    /// Since `imm` is i64, the high half is the sign extension:
    /// - negative: hi = 0xFFFF_FFFF_FFFF_FFFF
    /// - non-negative: hi = 0
    fn select_i128_iconst(
        &mut self,
        imm: i64,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_result(inst, "i128 iconst")?;

        let lo_val = imm as u64;
        let hi_val: u64 = if imm < 0 { u64::MAX } else { 0 };

        let dst_lo = self.new_vreg(RegClass::Gpr64);
        let dst_hi = self.new_vreg(RegClass::Gpr64);

        // Materialize low half
        Self::emit_movz_movk_sequence(&mut self.func, block, dst_lo, lo_val);
        // Materialize high half
        Self::emit_movz_movk_sequence(&mut self.func, block, dst_hi, hi_val);

        self.define_i128_value(inst.results[0], dst_lo, dst_hi);
        Ok(())
    }

    /// Emit a MOVZ + MOVK sequence to materialize a 64-bit value into a vreg.
    fn emit_movz_movk_sequence(func: &mut ISelFunction, block: Block, dst: VReg, val: u64) {
        let low16 = val & 0xFFFF;
        func.push_inst(
            block,
            ISelInst::new(
                AArch64Opcode::Movz,
                vec![ISelOperand::VReg(dst), ISelOperand::Imm(low16 as i64)],
            ),
        );
        for shift in 1..4u64 {
            let chunk = (val >> (shift * 16)) & 0xFFFF;
            if chunk != 0 {
                func.push_inst(
                    block,
                    ISelInst::new(
                        AArch64Opcode::Movk,
                        vec![
                            ISelOperand::VReg(dst),
                            ISelOperand::Imm(chunk as i64),
                            ISelOperand::Imm((shift * 16) as i64),
                        ],
                    ),
                );
            }
        }
    }

    // -------------------------------------------------------------------
    // i128 shift operations
    // -------------------------------------------------------------------

    /// Select i128 left shift using a general register-based decomposition.
    ///
    /// ```text
    /// c64       = MOVZ #64
    /// neg_shift = SUB c64, shift          // 64 - shift
    /// lo_spill  = LSR src_lo, neg_shift   // bits from lo that shift into hi
    /// hi_shifted= LSL src_hi, shift       // hi << shift
    /// hi_normal = ORR hi_shifted, lo_spill// combined hi for shift < 64
    /// lo_normal = LSL src_lo, shift       // lo << shift
    /// big_shift = SUB shift, c64          // shift - 64
    /// hi_big    = LSL src_lo, big_shift   // lo << (shift-64), for shift >= 64
    /// zero      = MOVZ #0
    /// CMP shift, #64
    /// dst_lo    = CSEL zero, lo_normal, HS      // shift >= 64 ? 0 : lo_normal
    /// dst_hi    = CSEL hi_big, hi_normal, HS    // shift >= 64 ? hi_big : hi_normal
    /// ```
    fn select_i128_shl(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "i128 shl")?;
        Self::require_result(inst, "i128 shl")?;

        let (src_lo, src_hi) = self.use_i128_value(&inst.args[0])?;
        let shift = self.use_value(&inst.args[1])?;

        let c64 = self.new_vreg(RegClass::Gpr64);
        let neg_shift = self.new_vreg(RegClass::Gpr64);
        let lo_spill = self.new_vreg(RegClass::Gpr64);
        let hi_shifted = self.new_vreg(RegClass::Gpr64);
        let hi_normal = self.new_vreg(RegClass::Gpr64);
        let lo_normal = self.new_vreg(RegClass::Gpr64);
        let big_shift = self.new_vreg(RegClass::Gpr64);
        let hi_big = self.new_vreg(RegClass::Gpr64);
        let zero = self.new_vreg(RegClass::Gpr64);
        let dst_lo = self.new_vreg(RegClass::Gpr64);
        let dst_hi = self.new_vreg(RegClass::Gpr64);

        // c64 = 64
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Movz,
                vec![ISelOperand::VReg(c64), ISelOperand::Imm(64)],
            ),
        );
        // neg_shift = 64 - shift
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::SubRR,
                vec![
                    ISelOperand::VReg(neg_shift),
                    ISelOperand::VReg(c64),
                    shift.clone(),
                ],
            ),
        );
        // lo_spill = src_lo >> (64 - shift)
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::LsrRR,
                vec![
                    ISelOperand::VReg(lo_spill),
                    src_lo.clone(),
                    ISelOperand::VReg(neg_shift),
                ],
            ),
        );
        // hi_shifted = src_hi << shift
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::LslRR,
                vec![ISelOperand::VReg(hi_shifted), src_hi, shift.clone()],
            ),
        );
        // hi_normal = hi_shifted | lo_spill
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::OrrRR,
                vec![
                    ISelOperand::VReg(hi_normal),
                    ISelOperand::VReg(hi_shifted),
                    ISelOperand::VReg(lo_spill),
                ],
            ),
        );
        // lo_normal = src_lo << shift
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::LslRR,
                vec![ISelOperand::VReg(lo_normal), src_lo.clone(), shift.clone()],
            ),
        );
        // big_shift = shift - 64
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::SubRR,
                vec![
                    ISelOperand::VReg(big_shift),
                    shift.clone(),
                    ISelOperand::VReg(c64),
                ],
            ),
        );
        // hi_big = src_lo << (shift - 64)
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::LslRR,
                vec![
                    ISelOperand::VReg(hi_big),
                    src_lo,
                    ISelOperand::VReg(big_shift),
                ],
            ),
        );
        // zero = 0
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Movz,
                vec![ISelOperand::VReg(zero), ISelOperand::Imm(0)],
            ),
        );
        // CMP shift, #64
        self.emit(
            block,
            ISelInst::new(AArch64Opcode::CmpRI, vec![shift, ISelOperand::Imm(64)]),
        );
        // dst_lo = shift >= 64 ? zero : lo_normal
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Csel,
                vec![
                    ISelOperand::VReg(dst_lo),
                    ISelOperand::VReg(zero),
                    ISelOperand::VReg(lo_normal),
                    ISelOperand::CondCode(AArch64CC::HS),
                ],
            ),
        );
        // dst_hi = shift >= 64 ? hi_big : hi_normal
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Csel,
                vec![
                    ISelOperand::VReg(dst_hi),
                    ISelOperand::VReg(hi_big),
                    ISelOperand::VReg(hi_normal),
                    ISelOperand::CondCode(AArch64CC::HS),
                ],
            ),
        );

        self.define_i128_value(inst.results[0], dst_lo, dst_hi);
        Ok(())
    }

    /// Select i128 logical shift right (unsigned).
    ///
    /// Mirror of left shift: high half shifts right, low half receives spill bits.
    /// ```text
    /// c64       = MOVZ #64
    /// neg_shift = SUB c64, shift
    /// hi_spill  = LSL src_hi, neg_shift   // bits from hi that shift into lo
    /// lo_shifted= LSR src_lo, shift
    /// lo_normal = ORR lo_shifted, hi_spill
    /// hi_normal = LSR src_hi, shift
    /// big_shift = SUB shift, c64
    /// lo_big    = LSR src_hi, big_shift   // for shift >= 64
    /// zero      = MOVZ #0
    /// CMP shift, #64
    /// dst_hi    = CSEL zero, hi_normal, HS
    /// dst_lo    = CSEL lo_big, lo_normal, HS
    /// ```
    fn select_i128_lshr(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "i128 lshr")?;
        Self::require_result(inst, "i128 lshr")?;

        let (src_lo, src_hi) = self.use_i128_value(&inst.args[0])?;
        let shift = self.use_value(&inst.args[1])?;

        let c64 = self.new_vreg(RegClass::Gpr64);
        let neg_shift = self.new_vreg(RegClass::Gpr64);
        let hi_spill = self.new_vreg(RegClass::Gpr64);
        let lo_shifted = self.new_vreg(RegClass::Gpr64);
        let lo_normal = self.new_vreg(RegClass::Gpr64);
        let hi_normal = self.new_vreg(RegClass::Gpr64);
        let big_shift = self.new_vreg(RegClass::Gpr64);
        let lo_big = self.new_vreg(RegClass::Gpr64);
        let zero = self.new_vreg(RegClass::Gpr64);
        let dst_lo = self.new_vreg(RegClass::Gpr64);
        let dst_hi = self.new_vreg(RegClass::Gpr64);

        // c64 = 64
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Movz,
                vec![ISelOperand::VReg(c64), ISelOperand::Imm(64)],
            ),
        );
        // neg_shift = 64 - shift
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::SubRR,
                vec![
                    ISelOperand::VReg(neg_shift),
                    ISelOperand::VReg(c64),
                    shift.clone(),
                ],
            ),
        );
        // hi_spill = src_hi << (64 - shift)
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::LslRR,
                vec![
                    ISelOperand::VReg(hi_spill),
                    src_hi.clone(),
                    ISelOperand::VReg(neg_shift),
                ],
            ),
        );
        // lo_shifted = src_lo >> shift
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::LsrRR,
                vec![ISelOperand::VReg(lo_shifted), src_lo, shift.clone()],
            ),
        );
        // lo_normal = lo_shifted | hi_spill
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::OrrRR,
                vec![
                    ISelOperand::VReg(lo_normal),
                    ISelOperand::VReg(lo_shifted),
                    ISelOperand::VReg(hi_spill),
                ],
            ),
        );
        // hi_normal = src_hi >> shift
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::LsrRR,
                vec![ISelOperand::VReg(hi_normal), src_hi.clone(), shift.clone()],
            ),
        );
        // big_shift = shift - 64
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::SubRR,
                vec![
                    ISelOperand::VReg(big_shift),
                    shift.clone(),
                    ISelOperand::VReg(c64),
                ],
            ),
        );
        // lo_big = src_hi >> (shift - 64)
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::LsrRR,
                vec![
                    ISelOperand::VReg(lo_big),
                    src_hi,
                    ISelOperand::VReg(big_shift),
                ],
            ),
        );
        // zero = 0
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Movz,
                vec![ISelOperand::VReg(zero), ISelOperand::Imm(0)],
            ),
        );
        // CMP shift, #64
        self.emit(
            block,
            ISelInst::new(AArch64Opcode::CmpRI, vec![shift, ISelOperand::Imm(64)]),
        );
        // dst_hi = shift >= 64 ? zero : hi_normal
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Csel,
                vec![
                    ISelOperand::VReg(dst_hi),
                    ISelOperand::VReg(zero),
                    ISelOperand::VReg(hi_normal),
                    ISelOperand::CondCode(AArch64CC::HS),
                ],
            ),
        );
        // dst_lo = shift >= 64 ? lo_big : lo_normal
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Csel,
                vec![
                    ISelOperand::VReg(dst_lo),
                    ISelOperand::VReg(lo_big),
                    ISelOperand::VReg(lo_normal),
                    ISelOperand::CondCode(AArch64CC::HS),
                ],
            ),
        );

        self.define_i128_value(inst.results[0], dst_lo, dst_hi);
        Ok(())
    }

    /// Select i128 arithmetic shift right (signed).
    ///
    /// Like lshr but the high half uses ASR and the shift >= 64 case
    /// sign-fills the high half (ASR by 63 = all 0s or all 1s).
    /// ```text
    /// c64       = MOVZ #64
    /// c63       = MOVZ #63
    /// neg_shift = SUB c64, shift
    /// hi_spill  = LSL src_hi, neg_shift
    /// lo_shifted= LSR src_lo, shift
    /// lo_normal = ORR lo_shifted, hi_spill
    /// hi_normal = ASR src_hi, shift
    /// big_shift = SUB shift, c64
    /// lo_big    = ASR src_hi, big_shift
    /// hi_sign   = ASR src_hi, c63           // sign-fill
    /// CMP shift, #64
    /// dst_hi    = CSEL hi_sign, hi_normal, HS
    /// dst_lo    = CSEL lo_big, lo_normal, HS
    /// ```
    fn select_i128_ashr(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "i128 ashr")?;
        Self::require_result(inst, "i128 ashr")?;

        let (src_lo, src_hi) = self.use_i128_value(&inst.args[0])?;
        let shift = self.use_value(&inst.args[1])?;

        let c64 = self.new_vreg(RegClass::Gpr64);
        let c63 = self.new_vreg(RegClass::Gpr64);
        let neg_shift = self.new_vreg(RegClass::Gpr64);
        let hi_spill = self.new_vreg(RegClass::Gpr64);
        let lo_shifted = self.new_vreg(RegClass::Gpr64);
        let lo_normal = self.new_vreg(RegClass::Gpr64);
        let hi_normal = self.new_vreg(RegClass::Gpr64);
        let big_shift = self.new_vreg(RegClass::Gpr64);
        let lo_big = self.new_vreg(RegClass::Gpr64);
        let hi_sign = self.new_vreg(RegClass::Gpr64);
        let dst_lo = self.new_vreg(RegClass::Gpr64);
        let dst_hi = self.new_vreg(RegClass::Gpr64);

        // c64 = 64
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Movz,
                vec![ISelOperand::VReg(c64), ISelOperand::Imm(64)],
            ),
        );
        // c63 = 63
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Movz,
                vec![ISelOperand::VReg(c63), ISelOperand::Imm(63)],
            ),
        );
        // neg_shift = 64 - shift
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::SubRR,
                vec![
                    ISelOperand::VReg(neg_shift),
                    ISelOperand::VReg(c64),
                    shift.clone(),
                ],
            ),
        );
        // hi_spill = src_hi << (64 - shift)
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::LslRR,
                vec![
                    ISelOperand::VReg(hi_spill),
                    src_hi.clone(),
                    ISelOperand::VReg(neg_shift),
                ],
            ),
        );
        // lo_shifted = src_lo >> shift (logical)
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::LsrRR,
                vec![ISelOperand::VReg(lo_shifted), src_lo, shift.clone()],
            ),
        );
        // lo_normal = lo_shifted | hi_spill
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::OrrRR,
                vec![
                    ISelOperand::VReg(lo_normal),
                    ISelOperand::VReg(lo_shifted),
                    ISelOperand::VReg(hi_spill),
                ],
            ),
        );
        // hi_normal = src_hi >>> shift (arithmetic)
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::AsrRR,
                vec![ISelOperand::VReg(hi_normal), src_hi.clone(), shift.clone()],
            ),
        );
        // big_shift = shift - 64
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::SubRR,
                vec![
                    ISelOperand::VReg(big_shift),
                    shift.clone(),
                    ISelOperand::VReg(c64),
                ],
            ),
        );
        // lo_big = src_hi >>> (shift - 64) (arithmetic)
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::AsrRR,
                vec![
                    ISelOperand::VReg(lo_big),
                    src_hi.clone(),
                    ISelOperand::VReg(big_shift),
                ],
            ),
        );
        // hi_sign = src_hi >>> 63 (sign-fill: all 0s or all 1s)
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::AsrRR,
                vec![ISelOperand::VReg(hi_sign), src_hi, ISelOperand::VReg(c63)],
            ),
        );
        // CMP shift, #64
        self.emit(
            block,
            ISelInst::new(AArch64Opcode::CmpRI, vec![shift, ISelOperand::Imm(64)]),
        );
        // dst_hi = shift >= 64 ? hi_sign : hi_normal
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Csel,
                vec![
                    ISelOperand::VReg(dst_hi),
                    ISelOperand::VReg(hi_sign),
                    ISelOperand::VReg(hi_normal),
                    ISelOperand::CondCode(AArch64CC::HS),
                ],
            ),
        );
        // dst_lo = shift >= 64 ? lo_big : lo_normal
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Csel,
                vec![
                    ISelOperand::VReg(dst_lo),
                    ISelOperand::VReg(lo_big),
                    ISelOperand::VReg(lo_normal),
                    ISelOperand::CondCode(AArch64CC::HS),
                ],
            ),
        );

        self.define_i128_value(inst.results[0], dst_lo, dst_hi);
        Ok(())
    }

    // -------------------------------------------------------------------
    // i128 division (libcall)
    // -------------------------------------------------------------------

    /// Select i128 unsigned division via libcall to `__udivti3`.
    ///
    /// ABI: i128 arguments passed as register pairs:
    /// - dividend: X0 (lo), X1 (hi)
    /// - divisor:  X2 (lo), X3 (hi)
    /// - result:   X0 (lo), X1 (hi)
    fn select_i128_udiv(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        self.select_i128_div_libcall("__udivti3", inst, block)
    }

    /// Select i128 signed division via libcall to `__divti3`.
    fn select_i128_sdiv(&mut self, inst: &Instruction, block: Block) -> Result<(), ISelError> {
        self.select_i128_div_libcall("__divti3", inst, block)
    }

    /// Common implementation for i128 division libcalls.
    fn select_i128_div_libcall(
        &mut self,
        name: &str,
        inst: &Instruction,
        block: Block,
    ) -> Result<(), ISelError> {
        Self::require_args(inst, 2, "i128 div")?;
        Self::require_result(inst, "i128 div")?;

        let (a_lo, a_hi) = self.use_i128_value(&inst.args[0])?;
        let (b_lo, b_hi) = self.use_i128_value(&inst.args[1])?;

        // Move dividend to X0:X1 via Copy pseudo
        self.emit(
            block,
            ISelInst::new(AArch64Opcode::Copy, vec![ISelOperand::PReg(gpr::X0), a_lo]),
        );
        self.emit(
            block,
            ISelInst::new(AArch64Opcode::Copy, vec![ISelOperand::PReg(gpr::X1), a_hi]),
        );
        // Move divisor to X2:X3 via Copy pseudo
        self.emit(
            block,
            ISelInst::new(AArch64Opcode::Copy, vec![ISelOperand::PReg(gpr::X2), b_lo]),
        );
        self.emit(
            block,
            ISelInst::new(AArch64Opcode::Copy, vec![ISelOperand::PReg(gpr::X3), b_hi]),
        );

        // BL to libcall
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Bl,
                vec![ISelOperand::Symbol(name.to_string())],
            ),
        );

        // Copy results from X0:X1 via Copy pseudo
        let dst_lo = self.new_vreg(RegClass::Gpr64);
        let dst_hi = self.new_vreg(RegClass::Gpr64);

        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Copy,
                vec![ISelOperand::VReg(dst_lo), ISelOperand::PReg(gpr::X0)],
            ),
        );
        self.emit(
            block,
            ISelInst::new(
                AArch64Opcode::Copy,
                vec![ISelOperand::VReg(dst_hi), ISelOperand::PReg(gpr::X1)],
            ),
        );

        self.define_i128_value(inst.results[0], dst_lo, dst_hi);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Internal binop classification for opcode selection.
#[derive(Debug, Clone, Copy)]
enum AArch64BinOp {
    Add,
    Sub,
    Mul,
    Sdiv,
    Udiv,
}

/// Internal classification for `CheckedSadd` / `CheckedSsub` ISel lowering
/// (issue #474). `CheckedSmul` uses a separate selector because its idiom
/// differs (SMULH + ASR + CMP rather than ADDS/SUBS + CSET VS).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckedArith {
    Sadd,
    Ssub,
}

/// Shift operation classification.
#[derive(Debug, Clone, Copy)]
enum AArch64ShiftOp {
    Lsl,
    Lsr,
    Asr,
}

/// Logical operation classification.
#[derive(Debug, Clone, Copy)]
enum AArch64LogicOp {
    And,
    Orr,
    Eor,
    Bic,
    Orn,
}

/// Integer unary operation classification.
#[derive(Debug, Clone, Copy)]
enum AArch64IntUnaryOp {
    Neg,
    Mvn,
}

/// FP binop classification.
#[derive(Debug, Clone, Copy)]
enum AArch64FpBinOp {
    Fadd,
    Fsub,
    Fmul,
    Fdiv,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function::Signature;
    use crate::instructions::{
        AtomicOrdering, AtomicRmwOp, Block, FloatCC, Instruction, IntCC, Opcode, Value,
    };

    /// Helper: create a simple isel for fn(i32, i32) -> i32.
    fn make_add_isel() -> (InstructionSelector, Block) {
        let sig = Signature {
            params: vec![Type::I32, Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("add".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();
        (isel, entry)
    }

    /// Helper: create isel for fn(i64, i64) -> i64.
    fn make_i64_isel() -> (InstructionSelector, Block) {
        let sig = Signature {
            params: vec![Type::I64, Type::I64],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("op64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();
        (isel, entry)
    }

    /// Helper: create isel for fn(f64, f64) -> f64.
    fn make_f64_isel() -> (InstructionSelector, Block) {
        let sig = Signature {
            params: vec![Type::F64, Type::F64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("fpop".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();
        (isel, entry)
    }

    /// Helper: create isel for fn(f32, f32) -> f32.
    fn make_f32_isel() -> (InstructionSelector, Block) {
        let sig = Signature {
            params: vec![Type::F32, Type::F32],
            returns: vec![Type::F32],
        };
        let mut isel = InstructionSelector::new("fpop32".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();
        (isel, entry)
    }

    /// Helper: create isel with no args, manually defining values.
    fn make_empty_isel() -> (InstructionSelector, Block) {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut isel = InstructionSelector::new("test".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);
        (isel, entry)
    }

    // =======================================================================
    // Original tests (preserved from scaffold)
    // =======================================================================

    #[test]
    fn lower_formal_arguments_two_i32() {
        let (isel, entry) = make_add_isel();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[0].operands[1], ISelOperand::PReg(PReg::new(0))); // X0
        assert_eq!(mblock.insts[1].operands[1], ISelOperand::PReg(PReg::new(1))); // X1
    }

    #[test]
    fn select_iadd_i32() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 3);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::AddRR);
    }

    #[test]
    fn select_isub_i64() {
        let (mut isel, entry) = make_i64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Isub,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::SubRR);
    }

    #[test]
    fn select_icmp_and_brif() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Icmp { cond: IntCC::Equal },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Brif {
                    cond: Value(2),
                    then_dest: Block(1),
                    else_dest: Block(2),
                },
                args: vec![Value(2)],
                results: vec![],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 7);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::CmpRR);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::CSet);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::CmpRI);
        assert_eq!(mblock.insts[5].opcode, AArch64Opcode::BCond);
        assert_eq!(mblock.insts[6].opcode, AArch64Opcode::B);
        assert_eq!(mblock.successors.len(), 2);
    }

    #[test]
    fn select_return_i32() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 4);
        assert_eq!(mblock.insts[2].operands[0], ISelOperand::PReg(PReg::new(0))); // X0
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn select_return_void_ok() {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut isel = InstructionSelector::new("ret_void".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 1);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn select_return_rejects_empty_args_for_non_void() {
        let sig = Signature {
            params: vec![],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("ret_i32".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        let err = isel
            .select_instruction(
                &Instruction {
                    opcode: Opcode::Return,
                    args: vec![],
                    results: vec![],
                },
                entry,
            )
            .unwrap_err();

        assert!(matches!(
            err,
            ISelError::ReturnArityMismatch {
                expected: 1,
                actual: 0
            }
        ));
    }

    #[test]
    fn select_return_rejects_extra_args_for_void() {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut isel = InstructionSelector::new("ret_void_bad".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        let err = isel
            .select_instruction(
                &Instruction {
                    opcode: Opcode::Return,
                    args: vec![Value(0)],
                    results: vec![],
                },
                entry,
            )
            .unwrap_err();

        assert!(matches!(
            err,
            ISelError::ReturnArityMismatch {
                expected: 0,
                actual: 1
            }
        ));
    }

    #[test]
    fn select_return_rejects_type_mismatch_for_non_void() {
        let sig = Signature {
            params: vec![],
            returns: vec![Type::F32],
        };
        let mut isel = InstructionSelector::new("ret_f32".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        let vreg = isel.new_vreg(reg_class_for_type(&Type::I32));
        isel.define_value(Value(0), ISelOperand::VReg(vreg), Type::I32);

        let err = isel
            .select_instruction(
                &Instruction {
                    opcode: Opcode::Return,
                    args: vec![Value(0)],
                    results: vec![],
                },
                entry,
            )
            .unwrap_err();

        assert!(matches!(
            err,
            ISelError::ReturnTypeMismatch {
                index: 0,
                expected,
                actual,
            } if expected == "F32" && actual == "I32"
        ));
    }

    #[test]
    fn select_iconst_small() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I32,
                    imm: 42,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 1);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Movz);
    }

    #[test]
    fn select_iconst_negative() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I64,
                    imm: -1,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[0].opcode, AArch64Opcode::Movn);
    }

    #[test]
    fn condition_code_mapping() {
        assert_eq!(AArch64CC::from_intcc(IntCC::Equal), AArch64CC::EQ);
        assert_eq!(AArch64CC::from_intcc(IntCC::NotEqual), AArch64CC::NE);
        assert_eq!(AArch64CC::from_intcc(IntCC::SignedLessThan), AArch64CC::LT);
        assert_eq!(
            AArch64CC::from_intcc(IntCC::UnsignedGreaterThan),
            AArch64CC::HI
        );
    }

    #[test]
    fn full_add_function() {
        let sig = Signature {
            params: vec![Type::I32, Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("add".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(2)],
                results: vec![],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 5);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::AddRR);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Copy); // return value move
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::Ret);
        assert_eq!(mfunc.name, "add");
        assert_eq!(mfunc.next_vreg, 3);
    }

    // =======================================================================
    // New tests: Shift operations
    // =======================================================================

    #[test]
    fn select_lsl_i32_register() {
        let (mut isel, entry) = make_add_isel();
        // LSL: Value(2) = Value(0) << Value(1)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Ishl,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 2 COPY + 1 LslRR
        assert_eq!(mblock.insts.len(), 3);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::LslRR);
    }

    #[test]
    fn select_lsr_i64_register() {
        let (mut isel, entry) = make_i64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Ushr,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::LsrRR);
    }

    #[test]
    fn select_asr_i32_register() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Sshr,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::AsrRR);
    }

    #[test]
    fn select_lsl_i32_immediate() {
        // Test immediate shift: define a constant shift amount, then shift
        let (mut isel, entry) = make_empty_isel();

        // Value(0) = iconst i32, 100
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I32,
                    imm: 100,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        // Value(1) = iconst i32, 4 (shift amount)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I32,
                    imm: 4,
                },
                args: vec![],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        // Value(2) = ishl Value(0), Value(1)
        // Note: the shift amount is tracked as a VReg, not an immediate,
        // so this will use the register form. Immediate detection would
        // require peephole optimization (late combine phase).
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Ishl,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 1 Movz + 1 Movz + 1 LslRR (register form because value mapped as vreg)
        assert_eq!(mblock.insts.len(), 3);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::LslRR);
    }

    // =======================================================================
    // New tests: Logical operations
    // =======================================================================

    #[test]
    fn select_and_i32() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Band,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::AndRR);
    }

    #[test]
    fn select_orr_i64() {
        let (mut isel, entry) = make_i64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bor,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::OrrRR);
    }

    #[test]
    fn select_eor_i32() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bxor,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::EorRR);
    }

    #[test]
    fn select_bic_i64() {
        let (mut isel, entry) = make_i64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::BandNot,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::BicRR);
    }

    #[test]
    fn select_orn_i32() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::BorNot,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::OrnRR);
    }

    // =======================================================================
    // New tests: Extensions
    // =======================================================================

    #[test]
    fn select_sextb_to_i32() {
        // fn(i8) - sign extend byte to i32
        let sig = Signature {
            params: vec![Type::I8],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("sextb".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Sextend {
                    from_ty: Type::I8,
                    to_ty: Type::I32,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 1 COPY (formal arg) + 1 SXTBWr
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Sxtb);
    }

    #[test]
    fn select_sexth_to_i64() {
        let sig = Signature {
            params: vec![Type::I16],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("sexth".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Sextend {
                    from_ty: Type::I16,
                    to_ty: Type::I64,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::Sxth);
    }

    #[test]
    fn select_sxtw_to_i64() {
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("sxtw".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Sextend {
                    from_ty: Type::I32,
                    to_ty: Type::I64,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::Sxtw);
    }

    #[test]
    fn select_uxtb_to_i32() {
        let sig = Signature {
            params: vec![Type::I8],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("uxtb".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Uextend {
                    from_ty: Type::I8,
                    to_ty: Type::I32,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::Uxtb);
    }

    #[test]
    fn select_uxth_to_i32() {
        let sig = Signature {
            params: vec![Type::I16],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("uxth".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Uextend {
                    from_ty: Type::I16,
                    to_ty: Type::I32,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::Uxth);
    }

    #[test]
    fn select_uextend_i32_to_i64() {
        // Zero-extend i32 to i64: on AArch64, writing W reg zero-extends to X,
        // so we emit MovR.
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("uxtw".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Uextend {
                    from_ty: Type::I32,
                    to_ty: Type::I64,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::MovR);
    }

    // =======================================================================
    // New tests: Bitfield operations
    // =======================================================================

    #[test]
    fn select_ubfm_extract_i32() {
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("ubfm".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        // Extract 8 bits starting at bit 4: UBFM Wd, Wn, #4, #11
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::ExtractBits { lsb: 4, width: 8 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[1];
        assert_eq!(inst.opcode, AArch64Opcode::Ubfm);
        // immr = 4, imms = 11
        assert_eq!(inst.operands[2], ISelOperand::Imm(4));
        assert_eq!(inst.operands[3], ISelOperand::Imm(11));
    }

    #[test]
    fn select_sbfm_extract_i64() {
        let sig = Signature {
            params: vec![Type::I64],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("sbfm".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        // Signed extract 16 bits from bit 0: SBFM Xd, Xn, #0, #15
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::SextractBits { lsb: 0, width: 16 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[1];
        assert_eq!(inst.opcode, AArch64Opcode::Sbfm);
        assert_eq!(inst.operands[2], ISelOperand::Imm(0));
        assert_eq!(inst.operands[3], ISelOperand::Imm(15));
    }

    #[test]
    fn select_bfm_insert_i32() {
        let (mut isel, entry) = make_add_isel();

        // Insert 8 bits from Value(1) into Value(0) at bit 4
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::InsertBits { lsb: 4, width: 8 },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 2 COPY (args) + 1 COPY (dst to result) + 1 BFMWri
        assert_eq!(mblock.insts.len(), 4);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Bfm);
        // immr = (32 - 4) % 32 = 28, imms = 7
        assert_eq!(mblock.insts[3].operands[2], ISelOperand::Imm(28));
        assert_eq!(mblock.insts[3].operands[3], ISelOperand::Imm(7));
    }

    // =======================================================================
    // New tests: Conditional select
    // =======================================================================

    #[test]
    fn select_csel_i32() {
        let sig = Signature {
            params: vec![Type::I32, Type::I32, Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("csel".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        // Value(3) = select(NE, Value(0), Value(1), Value(2))
        // cond_val=Value(0), true=Value(1), false=Value(2)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Select {
                    cond: IntCC::NotEqual,
                },
                args: vec![Value(0), Value(1), Value(2)],
                results: vec![Value(3)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 3 COPY (args) + 1 CmpRI + 1 Csel
        assert_eq!(mblock.insts.len(), 5);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::CmpRI);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::Csel);
        assert_eq!(
            mblock.insts[4].operands[3],
            ISelOperand::CondCode(AArch64CC::NE)
        );
    }

    #[test]
    fn select_csel_i64() {
        let sig = Signature {
            params: vec![Type::I32, Type::I64, Type::I64],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("csel64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Select {
                    cond: IntCC::SignedGreaterThan,
                },
                args: vec![Value(0), Value(1), Value(2)],
                results: vec![Value(3)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 3 COPY + 1 CmpRI + 1 Csel (64-bit because true_val is I64)
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::Csel);
        assert_eq!(
            mblock.insts[4].operands[3],
            ISelOperand::CondCode(AArch64CC::GT)
        );
    }

    // =======================================================================
    // New tests: Move wide (large immediate materialization)
    // =======================================================================

    #[test]
    fn select_iconst_large_i64() {
        let (mut isel, entry) = make_empty_isel();
        // 0x0001_0002_0003_0004: needs MOVZ + 3 MOVK
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I64,
                    imm: 0x0001_0002_0003_0004i64,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // MOVZ (low 16 bits) + 3 MOVK (remaining non-zero chunks)
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Movz);
        assert_eq!(mblock.insts[0].operands[1], ISelOperand::Imm(0x0004));

        // Should have MOVK for bits [16:31], [32:47], [48:63]
        let movk_count = mblock
            .insts
            .iter()
            .filter(|i| i.opcode == AArch64Opcode::Movk)
            .count();
        assert_eq!(movk_count, 3);
    }

    #[test]
    fn select_iconst_i32_large() {
        let (mut isel, entry) = make_empty_isel();
        // 0x00010002: needs MOVZ + 1 MOVK (2 chunks for 32-bit)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I32,
                    imm: 0x00010002i64,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Movz);
        assert_eq!(mblock.insts[0].operands[1], ISelOperand::Imm(0x0002));
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Movk);
        assert_eq!(mblock.insts[1].operands[1], ISelOperand::Imm(0x0001));
        assert_eq!(mblock.insts[1].operands[2], ISelOperand::Imm(16)); // shift=16
    }

    // =======================================================================
    // New tests: FP operations
    // =======================================================================

    #[test]
    fn select_fadd_f64() {
        let (mut isel, entry) = make_f64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::FaddRR);
    }

    #[test]
    fn select_fsub_f32() {
        let (mut isel, entry) = make_f32_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fsub,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::FsubRR);
    }

    #[test]
    fn select_fmul_f64() {
        let (mut isel, entry) = make_f64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fmul,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::FmulRR);
    }

    #[test]
    fn select_fdiv_f32() {
        let (mut isel, entry) = make_f32_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fdiv,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::FdivRR);
    }

    #[test]
    fn select_fcmp_f64() {
        let (mut isel, entry) = make_f64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fcmp {
                    cond: FloatCC::LessThan,
                },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 2 COPY + 1 FCMPDrr + 1 CSETWcc
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::Fcmp);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::CSet);
        assert_eq!(
            mblock.insts[3].operands[1],
            ISelOperand::CondCode(AArch64CC::MI)
        );
    }

    #[test]
    fn select_fcvt_to_int_i32() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("fcvt2i".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FcvtToInt { dst_ty: Type::I32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(
            mfunc.blocks[&entry].insts[1].opcode,
            AArch64Opcode::FcvtzsRR
        );
    }

    #[test]
    fn select_fcvt_from_int_i32_to_f32() {
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::F32],
        };
        let mut isel = InstructionSelector::new("i2f".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FcvtFromInt { src_ty: Type::I32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::ScvtfRR);
    }

    #[test]
    fn select_fcvt_from_int_i64_to_f64() {
        let sig = Signature {
            params: vec![Type::I64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("i642f64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FcvtFromInt { src_ty: Type::I64 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::ScvtfRR);
    }

    // =======================================================================
    // New tests: Addressing
    // =======================================================================

    #[test]
    fn select_global_ref() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::GlobalRef {
                    name: "my_global".to_string(),
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // ADRP + ADD
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Adrp);
        assert_eq!(
            mblock.insts[0].operands[1],
            ISelOperand::Symbol("my_global".to_string())
        );
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::AddPCRel);
        assert_eq!(
            mblock.insts[1].operands[2],
            ISelOperand::Symbol("my_global".to_string())
        );
    }

    #[test]
    fn select_extern_ref_got_indirect() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::ExternRef {
                    name: "printf".to_string(),
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // ADRP + LDR (GOT-indirect, not ADD)
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Adrp);
        assert_eq!(
            mblock.insts[0].operands[1],
            ISelOperand::Symbol("printf".to_string())
        );
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::LdrGot);
        assert_eq!(
            mblock.insts[1].operands[2],
            ISelOperand::Symbol("printf".to_string())
        );
    }

    #[test]
    fn select_tls_ref_tlv_indirect() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::TlsRef {
                    name: "thread_local_var".to_string(),
                    model: TlsModel::Tlv,
                    local_exec_offset: None,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // ADRP + LDR (TLV-indirect)
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Adrp);
        assert_eq!(
            mblock.insts[0].operands[1],
            ISelOperand::Symbol("thread_local_var".to_string())
        );
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::LdrTlvp);
        assert_eq!(
            mblock.insts[1].operands[2],
            ISelOperand::Symbol("thread_local_var".to_string())
        );
    }

    #[test]
    fn test_select_tls_ref_local_exec_zero_offset() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::TlsRef {
                    name: "thread_local_var".to_string(),
                    model: TlsModel::LocalExec,
                    local_exec_offset: Some(0),
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 1);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Mrs);
        assert_eq!(
            mblock.insts[0].operands[1],
            ISelOperand::Imm(TPIDR_EL0_SYSREG)
        );
    }

    #[test]
    fn test_select_tls_ref_local_exec_small_offset() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::TlsRef {
                    name: "thread_local_var".to_string(),
                    model: TlsModel::LocalExec,
                    local_exec_offset: Some(0x40),
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Mrs);
        assert_eq!(
            mblock.insts[0].operands[1],
            ISelOperand::Imm(TPIDR_EL0_SYSREG)
        );
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::AddRI);
        assert_eq!(mblock.insts[1].operands[2], ISelOperand::Imm(0x40));
    }

    #[test]
    fn test_select_tls_ref_local_exec_large_offset() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::TlsRef {
                    name: "thread_local_var".to_string(),
                    model: TlsModel::LocalExec,
                    local_exec_offset: Some(0xAB123),
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 3);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Mrs);
        assert_eq!(
            mblock.insts[0].operands[1],
            ISelOperand::Imm(TPIDR_EL0_SYSREG)
        );
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::AddRIShift12);
        assert_eq!(mblock.insts[1].operands[2], ISelOperand::Imm(0xAB));
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::AddRI);
        assert_eq!(mblock.insts[2].operands[2], ISelOperand::Imm(0x123));
    }

    #[test]
    fn test_select_tls_ref_local_exec_missing_offset_errors() {
        let (mut isel, entry) = make_empty_isel();
        let err = isel
            .select_instruction(
                &Instruction {
                    opcode: Opcode::TlsRef {
                        name: "thread_local_var".to_string(),
                        model: TlsModel::LocalExec,
                        local_exec_offset: None,
                    },
                    args: vec![],
                    results: vec![Value(0)],
                },
                entry,
            )
            .expect_err("LocalExec TlsRef without offset must return an error");

        assert!(matches!(err, ISelError::LocalExecTlsRefMissingOffset));
    }

    #[test]
    fn select_stack_addr() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::StackAddr { slot: 3 },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 1);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::AddPCRel);
        // Should reference SP (PReg(31)) and StackSlot(3)
        assert_eq!(mblock.insts[0].operands[1], ISelOperand::PReg(SP));
        assert_eq!(mblock.insts[0].operands[2], ISelOperand::StackSlot(3));
    }

    // =======================================================================
    // New tests: FloatCC mapping
    // =======================================================================

    #[test]
    fn float_condition_code_mapping() {
        assert_eq!(AArch64CC::from_floatcc(FloatCC::Equal), AArch64CC::EQ);
        assert_eq!(AArch64CC::from_floatcc(FloatCC::NotEqual), AArch64CC::NE);
        assert_eq!(AArch64CC::from_floatcc(FloatCC::LessThan), AArch64CC::MI);
        assert_eq!(
            AArch64CC::from_floatcc(FloatCC::LessThanOrEqual),
            AArch64CC::LS
        );
        assert_eq!(AArch64CC::from_floatcc(FloatCC::GreaterThan), AArch64CC::GT);
        assert_eq!(
            AArch64CC::from_floatcc(FloatCC::GreaterThanOrEqual),
            AArch64CC::GE
        );
        assert_eq!(AArch64CC::from_floatcc(FloatCC::Ordered), AArch64CC::VC);
        assert_eq!(AArch64CC::from_floatcc(FloatCC::Unordered), AArch64CC::VS);
    }

    #[test]
    fn condition_code_inversion() {
        assert_eq!(AArch64CC::EQ.invert(), AArch64CC::NE);
        assert_eq!(AArch64CC::NE.invert(), AArch64CC::EQ);
        assert_eq!(AArch64CC::LT.invert(), AArch64CC::GE);
        assert_eq!(AArch64CC::GE.invert(), AArch64CC::LT);
        assert_eq!(AArch64CC::GT.invert(), AArch64CC::LE);
        assert_eq!(AArch64CC::LE.invert(), AArch64CC::GT);
        assert_eq!(AArch64CC::HI.invert(), AArch64CC::LS);
        assert_eq!(AArch64CC::LS.invert(), AArch64CC::HI);
        assert_eq!(AArch64CC::VS.invert(), AArch64CC::VC);
        assert_eq!(AArch64CC::VC.invert(), AArch64CC::VS);
    }

    // =======================================================================
    // End-to-end: compound pattern
    // =======================================================================

    #[test]
    fn full_shift_and_mask() {
        // fn shift_mask(i32, i32) -> i32 { return (a << b) & 0xFF; }
        // This tests shift followed by logical AND.
        let (mut isel, entry) = make_add_isel();

        // Value(2) = ishl Value(0), Value(1)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Ishl,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        // Value(3) = iconst i32, 0xFF
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I32,
                    imm: 0xFF,
                },
                args: vec![],
                results: vec![Value(3)],
            },
            entry,
        )
        .unwrap();

        // Value(4) = band Value(2), Value(3)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Band,
                args: vec![Value(2), Value(3)],
                results: vec![Value(4)],
            },
            entry,
        )
        .unwrap();

        // Return Value(4)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(4)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Expected:
        // 0: COPY vreg0 <- X0
        // 1: COPY vreg1 <- X1
        // 2: LslRR vreg2 <- vreg0, vreg1
        // 3: Movz vreg3 <- 0xFF
        // 4: ANDWrr vreg4 <- vreg2, vreg3
        // 5: Copy X0 <- vreg4 (return value)
        // 6: RET
        assert_eq!(mblock.insts.len(), 7);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::LslRR);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Movz);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::AndRR);
        assert_eq!(mblock.insts[5].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[6].opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn full_fp_add_function() {
        // fn fadd(f64, f64) -> f64 { return a + b; }
        let sig = Signature {
            params: vec![Type::F64, Type::F64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("fadd64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(2)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 2 COPY (formal args) + 1 FaddRR + 1 Copy (to V0) + 1 RET
        assert_eq!(mblock.insts.len(), 5);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::FaddRR);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::Ret);
    }

    // =======================================================================
    // Call with Symbol operand (issue #69)
    // =======================================================================

    #[test]
    fn select_call_emits_symbol_operand() {
        // Test that Call { name } from the adapter produces a BL with Symbol operand.
        let (mut isel, entry) = make_empty_isel();

        // Define an arg value (simulating a function parameter).
        let vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), Type::I32);

        // Select a Call instruction.
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Call {
                    name: "my_callee".to_string(),
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Should contain at least: MOV (arg to X0) + BL + MOV (X0 to result)
        // Find the BL instruction and verify it has a Symbol operand.
        let bl_inst = mblock.insts.iter().find(|i| i.opcode == AArch64Opcode::Bl);
        assert!(
            bl_inst.is_some(),
            "Expected BL instruction for function call"
        );
        let bl = bl_inst.unwrap();
        assert_eq!(
            bl.operands[0],
            ISelOperand::Symbol("my_callee".to_string()),
            "BL should have Symbol operand with callee name"
        );
    }

    // =======================================================================
    // CondCode operand preserved through Brif (issue #69)
    // =======================================================================

    #[test]
    fn select_brif_emits_condcode_operand() {
        // Test that Brif from the adapter produces B.cond with CondCode operand.
        let (mut isel, entry) = make_empty_isel();

        // Define a boolean condition value.
        let vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), Type::B1);

        let then_block = Block(1);
        let else_block = Block(2);
        isel.func.ensure_block(then_block);
        isel.func.ensure_block(else_block);

        // Select a Brif instruction.
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Brif {
                    cond: Value(0),
                    then_dest: then_block,
                    else_dest: else_block,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Expected: CMP W, #0 + B.NE then + B else
        assert!(mblock.insts.len() >= 3, "Brif should emit CMP + B.cond + B");

        // Find B.cond and verify CondCode operand.
        let bcc_inst = mblock
            .insts
            .iter()
            .find(|i| i.opcode == AArch64Opcode::BCond);
        assert!(bcc_inst.is_some(), "Expected B.cond instruction");
        let bcc = bcc_inst.unwrap();
        assert_eq!(
            bcc.operands[0],
            ISelOperand::CondCode(AArch64CC::NE),
            "B.cond should have NE condition code (branch if nonzero)"
        );
        assert_eq!(
            bcc.operands[1],
            ISelOperand::Block(then_block),
            "B.cond should target the then block"
        );
    }

    // =======================================================================
    // Integer remainder (SRem/URem) tests
    // =======================================================================

    #[test]
    fn select_urem_i32() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Urem,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 2 COPYs for args + UDIV + MSUB = 4 instructions
        assert_eq!(mblock.insts.len(), 4);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::UDiv);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Msub);
        // MSUB operands: [dst, tmp(quotient), divisor, dividend]
        assert_eq!(mblock.insts[3].operands.len(), 4);
    }

    #[test]
    fn select_srem_i32() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Srem,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 4);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::SDiv);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Msub);
    }

    #[test]
    fn select_urem_i64() {
        let (mut isel, entry) = make_i64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Urem,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 4);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::UDiv);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Msub);
    }

    #[test]
    fn select_srem_i64() {
        let (mut isel, entry) = make_i64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Srem,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 4);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::SDiv);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Msub);
    }

    #[test]
    fn select_srem_msub_operand_order() {
        // Verify MSUB operand order: MSUB dst, tmp(quotient), divisor, dividend
        // This ensures result = dividend - quotient * divisor
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Srem,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        let div_inst = &mblock.insts[2];
        let msub_inst = &mblock.insts[3];

        // SDIV dst=tmp, lhs=dividend, rhs=divisor
        assert_eq!(div_inst.opcode, AArch64Opcode::SDiv);
        let div_dst = &div_inst.operands[0]; // tmp vreg
        let dividend_op = &div_inst.operands[1]; // dividend
        let divisor_op = &div_inst.operands[2]; // divisor

        // MSUB dst, Rn=tmp, Rm=divisor, Ra=dividend
        assert_eq!(msub_inst.opcode, AArch64Opcode::Msub);
        assert_eq!(
            &msub_inst.operands[1], div_dst,
            "MSUB Rn should be quotient from SDIV"
        );
        assert_eq!(
            &msub_inst.operands[2], divisor_op,
            "MSUB Rm should be divisor"
        );
        assert_eq!(
            &msub_inst.operands[3], dividend_op,
            "MSUB Ra should be dividend"
        );
    }

    // =======================================================================
    // Unary operations: Neg, Not (Bnot), FNeg
    // =======================================================================

    #[test]
    fn select_ineg_i32() {
        // fn(i32) -> i32: return -arg
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("neg32".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Ineg,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 1 COPY (formal arg) + 1 Neg
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Neg);
        assert_eq!(mblock.insts[1].operands.len(), 2);
    }

    #[test]
    fn select_ineg_i64() {
        let sig = Signature {
            params: vec![Type::I64],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("neg64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Ineg,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::Neg);
    }

    #[test]
    fn select_bnot_i32() {
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("not32".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bnot,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 1 COPY (formal arg) + 1 ORN (MVN alias)
        assert_eq!(mblock.insts.len(), 2);
        let orn = &mblock.insts[1];
        assert_eq!(orn.opcode, AArch64Opcode::OrnRR);
        // ORN Rd, WZR, Rm — 3 operands (issue #334)
        assert_eq!(
            orn.operands.len(),
            3,
            "OrnRR (MVN i32) must have 3 operands [dst, WZR, src]"
        );
        assert_eq!(
            orn.operands[1],
            ISelOperand::PReg(WZR),
            "OrnRR (MVN i32) operand[1] must be WZR"
        );
    }

    #[test]
    fn select_bnot_i64() {
        let sig = Signature {
            params: vec![Type::I64],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("not64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bnot,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let orn = &mfunc.blocks[&entry].insts[1];
        assert_eq!(orn.opcode, AArch64Opcode::OrnRR);
        // ORN Rd, XZR, Rm — 3 operands (issue #334)
        assert_eq!(
            orn.operands.len(),
            3,
            "OrnRR (MVN i64) must have 3 operands [dst, XZR, src]"
        );
        assert_eq!(
            orn.operands[1],
            ISelOperand::PReg(XZR),
            "OrnRR (MVN i64) operand[1] must be XZR"
        );
    }

    #[test]
    fn select_fneg_f64() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("fneg64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fneg,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 1 COPY (formal arg) + 1 FNEGDr
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::FnegRR);
    }

    #[test]
    fn select_fneg_f32() {
        let sig = Signature {
            params: vec![Type::F32],
            returns: vec![Type::F32],
        };
        let mut isel = InstructionSelector::new("fneg32".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fneg,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::FnegRR);
    }

    #[test]
    fn full_negate_function() {
        // fn negate(i32) -> i32 { return -arg; }
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("negate".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        // Value(1) = Ineg(Value(0))
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Ineg,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        // Return Value(1)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(1)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // COPY (arg) + Neg + Copy (to X0) + RET
        assert_eq!(mblock.insts.len(), 4);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Neg);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn full_fneg_function() {
        // fn fneg(f64) -> f64 { return -arg; }
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("fneg_fn".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fneg,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(1)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // COPY (arg) + FNEGDr + MOV (to V0) + RET
        assert_eq!(mblock.insts.len(), 4);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::FnegRR);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Ret);
    }

    // =======================================================================
    // Type conversion: unsigned FP <-> int
    // =======================================================================

    #[test]
    fn select_fcvt_to_uint_i32() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("fcvt2u".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FcvtToUint { dst_ty: Type::I32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(
            mfunc.blocks[&entry].insts[1].opcode,
            AArch64Opcode::FcvtzuRR
        );
    }

    #[test]
    fn select_fcvt_to_uint_i64() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("fcvt2u64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FcvtToUint { dst_ty: Type::I64 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(
            mfunc.blocks[&entry].insts[1].opcode,
            AArch64Opcode::FcvtzuRR
        );
    }

    #[test]
    fn select_fcvt_from_uint_i32_to_f32() {
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::F32],
        };
        let mut isel = InstructionSelector::new("u2f".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FcvtFromUint { src_ty: Type::I32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::UcvtfRR);
    }

    #[test]
    fn select_fcvt_from_uint_i64_to_f64() {
        let sig = Signature {
            params: vec![Type::I64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("u642f64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FcvtFromUint { src_ty: Type::I64 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::UcvtfRR);
    }

    // =======================================================================
    // Type conversion: FP precision (FPExt, FPTrunc)
    // =======================================================================

    #[test]
    fn select_fp_ext_f32_to_f64() {
        let sig = Signature {
            params: vec![Type::F32],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("fpext".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FPExt,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::FcvtSD);
    }

    #[test]
    fn select_fp_trunc_f64_to_f32() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::F32],
        };
        let mut isel = InstructionSelector::new("fptrunc".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FPTrunc,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::FcvtDS);
    }

    // =======================================================================
    // Type conversion: integer truncation
    // =======================================================================

    #[test]
    fn select_trunc_i64_to_i32() {
        let sig = Signature {
            params: vec![Type::I64],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("trunc64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Trunc { to_ty: Type::I32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        // On AArch64, truncation is a MOV to narrower register class
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::MovR);
    }

    #[test]
    fn select_trunc_i32_to_i16() {
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::I16],
        };
        let mut isel = InstructionSelector::new("trunc16".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Trunc { to_ty: Type::I16 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        // Trunc to i16 uses AND with mask 0xFFFF
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::AndRI);
        assert_eq!(
            mfunc.blocks[&entry].insts[1].operands[2],
            ISelOperand::Imm(0xFFFF)
        );
    }

    #[test]
    fn select_trunc_i64_to_i8() {
        let sig = Signature {
            params: vec![Type::I64],
            returns: vec![Type::I8],
        };
        let mut isel = InstructionSelector::new("trunc8".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Trunc { to_ty: Type::I8 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        // Trunc to i8 uses AND with mask 0xFF
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::AndRI);
        assert_eq!(
            mfunc.blocks[&entry].insts[1].operands[2],
            ISelOperand::Imm(0xFF)
        );
    }

    // =======================================================================
    // Extension/truncation: additional coverage
    // =======================================================================

    #[test]
    fn select_uxtb_to_i64() {
        let sig = Signature {
            params: vec![Type::I8],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("select_uxtb_to_i64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Uextend {
                    from_ty: Type::I8,
                    to_ty: Type::I64,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::Uxtb);
    }

    #[test]
    fn select_uxth_to_i64() {
        let sig = Signature {
            params: vec![Type::I16],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("select_uxth_to_i64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Uextend {
                    from_ty: Type::I16,
                    to_ty: Type::I64,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::Uxth);
    }

    #[test]
    fn select_sextb_to_i64() {
        let sig = Signature {
            params: vec![Type::I8],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("select_sextb_to_i64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Sextend {
                    from_ty: Type::I8,
                    to_ty: Type::I64,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::Sxtb);
    }

    #[test]
    fn select_sexth_to_i32() {
        let sig = Signature {
            params: vec![Type::I16],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("select_sexth_to_i32".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Sextend {
                    from_ty: Type::I16,
                    to_ty: Type::I32,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::Sxth);
    }

    #[test]
    fn select_trunc_i64_to_i16() {
        let sig = Signature {
            params: vec![Type::I64],
            returns: vec![Type::I16],
        };
        let mut isel = InstructionSelector::new("select_trunc_i64_to_i16".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Trunc { to_ty: Type::I16 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::AndRI);
        assert_eq!(
            mfunc.blocks[&entry].insts[1].operands[2],
            ISelOperand::Imm(0xFFFF)
        );
    }

    #[test]
    fn select_trunc_i32_to_i8() {
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::I8],
        };
        let mut isel = InstructionSelector::new("select_trunc_i32_to_i8".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Trunc { to_ty: Type::I8 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::AndRI);
        assert_eq!(
            mfunc.blocks[&entry].insts[1].operands[2],
            ISelOperand::Imm(0xFF)
        );
    }

    #[test]
    fn select_same_width_uext_copy() {
        // Unsigned extension with same source and target type falls through to Copy.
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("select_same_width_copy".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Uextend {
                    from_ty: Type::I32,
                    to_ty: Type::I32,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::Copy);
    }

    // =======================================================================
    // Type conversion: bitcast
    // =======================================================================

    #[test]
    fn select_bitcast_i32_to_f32() {
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::F32],
        };
        let mut isel = InstructionSelector::new("bc_i2f".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bitcast { to_ty: Type::F32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        // GPR -> FPR: FMOV Sd, Wn
        assert_eq!(
            mfunc.blocks[&entry].insts[1].opcode,
            AArch64Opcode::FmovGprFpr
        );
    }

    #[test]
    fn select_bitcast_i64_to_f64() {
        let sig = Signature {
            params: vec![Type::I64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("bc_i642f64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bitcast { to_ty: Type::F64 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        // GPR -> FPR: FMOV Dd, Xn
        assert_eq!(
            mfunc.blocks[&entry].insts[1].opcode,
            AArch64Opcode::FmovGprFpr
        );
    }

    #[test]
    fn select_bitcast_f32_to_i32() {
        let sig = Signature {
            params: vec![Type::F32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("bc_f2i".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bitcast { to_ty: Type::I32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        // FPR -> GPR: FMOV Wd, Sn
        assert_eq!(
            mfunc.blocks[&entry].insts[1].opcode,
            AArch64Opcode::FmovFprGpr
        );
    }

    #[test]
    fn select_bitcast_f64_to_i64() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("bc_f642i64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bitcast { to_ty: Type::I64 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        // FPR -> GPR: FMOV Xd, Dn
        assert_eq!(
            mfunc.blocks[&entry].insts[1].opcode,
            AArch64Opcode::FmovFprGpr
        );
    }

    #[test]
    fn select_bitcast_same_class_i32_to_i32() {
        // Same class bitcast (e.g., type punning between integer types)
        // should emit a plain MOV
        let (mut isel, entry) = make_empty_isel();

        let vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), Type::I32);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Bitcast { to_ty: Type::I32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[0].opcode, AArch64Opcode::MovR);
    }

    // =======================================================================
    // Aggregate ISel tests
    // =======================================================================

    #[test]
    fn select_struct_gep_field_0() {
        // struct { I32, I32 } -> field 0 is at offset 0, should emit MOV
        let (mut isel, entry) = make_empty_isel();

        // Define a pointer value
        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), Type::I64);

        let struct_ty = Type::Struct(vec![Type::I32, Type::I32]);
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::StructGep {
                    struct_ty,
                    field_index: 0,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // Field 0 at offset 0 -> MovR
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::MovR);
    }

    #[test]
    fn select_struct_gep_field_1_with_offset() {
        // struct { I32, I32 } -> field 1 is at offset 4, should emit ADD
        let (mut isel, entry) = make_empty_isel();

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), Type::I64);

        let struct_ty = Type::Struct(vec![Type::I32, Type::I32]);
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::StructGep {
                    struct_ty,
                    field_index: 1,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // Field 1 at offset 4 -> ADDXri
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::AddRI);
        // Offset should be 4
        assert_eq!(mblock.insts[0].operands[2], ISelOperand::Imm(4));
    }

    #[test]
    fn select_struct_gep_with_padding() {
        // struct { I8, I32 } -> field 1 is at offset 4 (3 bytes padding)
        let (mut isel, entry) = make_empty_isel();

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), Type::I64);

        let struct_ty = Type::Struct(vec![Type::I8, Type::I32]);
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::StructGep {
                    struct_ty,
                    field_index: 1,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::AddRI);
        assert_eq!(mblock.insts[0].operands[2], ISelOperand::Imm(4));
    }

    #[test]
    fn select_struct_gep_out_of_range() {
        let (mut isel, entry) = make_empty_isel();

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), Type::I64);

        let struct_ty = Type::Struct(vec![Type::I32]);
        let result = isel.select_instruction(
            &Instruction {
                opcode: Opcode::StructGep {
                    struct_ty,
                    field_index: 5,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        );
        assert!(result.is_err());
    }

    #[test]
    fn select_struct_gep_non_struct_type() {
        let (mut isel, entry) = make_empty_isel();

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), Type::I64);

        // Using I32 as the struct_ty should fail
        let result = isel.select_instruction(
            &Instruction {
                opcode: Opcode::StructGep {
                    struct_ty: Type::I32,
                    field_index: 0,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        );
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // ArrayGep tests — Phase 1 aggregate lowering (#391)
    // -----------------------------------------------------------------------

    #[test]
    fn select_array_gep_i32_power_of_two() {
        // i32 has bytes()==4 (power of two) → LslRI (#2) + AddRR
        let (mut isel, entry) = make_empty_isel();
        let base = isel.new_vreg(RegClass::Gpr64);
        let index = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(base), Type::I64);
        isel.define_value(Value(1), ISelOperand::VReg(index), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::ArrayGep { elem_ty: Type::I32 },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 2, "pow2 → LslRI + AddRR");
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::LslRI);
        assert_eq!(mblock.insts[0].operands[2], ISelOperand::Imm(2));
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::AddRR);
    }

    #[test]
    fn select_array_gep_i8_unit_stride() {
        // i8 has bytes()==1 → single AddRR (no shift)
        let (mut isel, entry) = make_empty_isel();
        let base = isel.new_vreg(RegClass::Gpr64);
        let index = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(base), Type::I64);
        isel.define_value(Value(1), ISelOperand::VReg(index), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::ArrayGep { elem_ty: Type::I8 },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 1, "unit stride → one AddRR");
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::AddRR);
    }

    #[test]
    fn select_array_gep_i64_power_of_two() {
        // i64 has bytes()==8 (power of two) → LslRI (#3) + AddRR
        let (mut isel, entry) = make_empty_isel();
        let base = isel.new_vreg(RegClass::Gpr64);
        let index = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(base), Type::I64);
        isel.define_value(Value(1), ISelOperand::VReg(index), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::ArrayGep { elem_ty: Type::I64 },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::LslRI);
        assert_eq!(mblock.insts[0].operands[2], ISelOperand::Imm(3));
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::AddRR);
    }

    #[test]
    fn select_array_gep_non_power_of_two_size() {
        // struct { I8, I8, I8 } has bytes()==3 (not power of two)
        //   → Movz + MulRR + AddRR
        let (mut isel, entry) = make_empty_isel();
        let base = isel.new_vreg(RegClass::Gpr64);
        let index = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(base), Type::I64);
        isel.define_value(Value(1), ISelOperand::VReg(index), Type::I64);

        let elem_ty = Type::Struct(vec![Type::I8, Type::I8, Type::I8]);
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::ArrayGep { elem_ty },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 3, "non-pow2 → Movz + MulRR + AddRR");
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Movz);
        assert_eq!(mblock.insts[0].operands[1], ISelOperand::Imm(3));
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::MulRR);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::AddRR);
    }

    #[test]
    fn select_array_gep_result_type_is_pointer() {
        // The result of ArrayGep is always I64 (pointer), regardless of elem_ty.
        let (mut isel, entry) = make_empty_isel();
        let base = isel.new_vreg(RegClass::Gpr64);
        let index = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(base), Type::I64);
        isel.define_value(Value(1), ISelOperand::VReg(index), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::ArrayGep { elem_ty: Type::F64 },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        // The value map should record Value(2) as I64.
        assert_eq!(isel.value_type(&Value(2)), Type::I64);
    }

    #[test]
    fn select_small_struct_return() {
        // Return a small struct (8 bytes, two i32 fields) packed in X0.
        // The return value is a pointer to the struct; select_return should
        // load it into X0.
        let struct_ty = Type::Struct(vec![Type::I32, Type::I32]);
        let sig = Signature {
            params: vec![],
            returns: vec![struct_ty.clone()],
        };
        let mut isel = InstructionSelector::new("ret_small".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        // Define a pointer to the struct as a vreg
        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), struct_ty);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // ABI classifies 8-byte Struct return as Reg(X0). The isel layer
        // detects aggregate types and uses select_aggregate_return which
        // loads the struct from memory: single LDR X0, [src].
        let insts = &mblock.insts;
        // Find the LDR X0 instruction
        let ldr_inst = insts.iter().find(|i| {
            i.opcode == AArch64Opcode::LdrRI
                && i.operands.first() == Some(&ISelOperand::PReg(gpr::X0))
        });
        assert!(
            ldr_inst.is_some(),
            "Expected LDR X0 for small struct return"
        );
        // Must end with RET
        assert_eq!(insts.last().unwrap().opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn select_medium_struct_return_16bytes() {
        // Return a 16-byte struct -> X0 + X1
        let struct_ty = Type::Struct(vec![Type::I64, Type::I64]);
        assert_eq!(struct_ty.bytes(), 16);

        let sig = Signature {
            params: vec![],
            returns: vec![struct_ty.clone()],
        };
        let mut isel = InstructionSelector::new("ret_med".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), struct_ty);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        let insts = &mblock.insts;

        // Should have LDR X0 at offset 0, LDR X1 at offset 8, then RET
        let ldr_x0 = insts.iter().find(|i| {
            i.opcode == AArch64Opcode::LdrRI
                && i.operands.first() == Some(&ISelOperand::PReg(gpr::X0))
        });
        assert!(ldr_x0.is_some(), "Expected LDR X0 for medium struct return");

        let ldr_x1 = insts.iter().find(|i| {
            i.opcode == AArch64Opcode::LdrRI
                && i.operands.first() == Some(&ISelOperand::PReg(gpr::X1))
        });
        assert!(ldr_x1.is_some(), "Expected LDR X1 for medium struct return");
        assert_eq!(insts.last().unwrap().opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn select_large_struct_return_via_sret() {
        // Return a large struct (24 bytes) -> store to [X8]
        let struct_ty = Type::Struct(vec![Type::I64, Type::I64, Type::I64]);
        assert_eq!(struct_ty.bytes(), 24);

        let sig = Signature {
            params: vec![],
            returns: vec![struct_ty.clone()],
        };
        let mut isel = InstructionSelector::new("ret_large".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), struct_ty);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        let insts = &mblock.insts;

        // Should have 3 LDR+STR pairs (24 bytes / 8 = 3 chunks) then RET
        let str_to_x8: Vec<_> = insts
            .iter()
            .filter(|i| {
                i.opcode == AArch64Opcode::StrRI
                    && i.operands.get(1) == Some(&ISelOperand::PReg(gpr::X8))
            })
            .collect();
        assert_eq!(
            str_to_x8.len(),
            3,
            "Expected 3 stores to X8 for 24-byte struct"
        );
        assert_eq!(insts.last().unwrap().opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn select_call_with_small_aggregate_arg() {
        // Call a function that takes a small struct (8 bytes).
        // The ABI should pass it in a single register.
        let (mut isel, entry) = make_empty_isel();

        let struct_ty = Type::Struct(vec![Type::I32, Type::I32]);
        // Define a pointer to the struct
        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), struct_ty);

        // Define a result value
        let result_types = vec![Type::I32];
        isel.select_call("callee", &[Value(0)], &[Value(1)], &result_types, entry)
            .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Should have: LDR X0 (aggregate arg), BL, MOV (result)
        let has_bl = mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::Bl);
        assert!(has_bl, "Expected BL instruction for call");

        // The aggregate arg should be loaded into X0 (small struct -> Reg(X0))
        let ldr_x0 = mblock.insts.iter().find(|i| {
            i.opcode == AArch64Opcode::LdrRI
                && i.operands.first() == Some(&ISelOperand::PReg(gpr::X0))
        });
        assert!(ldr_x0.is_some(), "Expected LDR X0 for small aggregate arg");
    }

    #[test]
    fn select_call_with_large_aggregate_arg() {
        // Call a function that takes a large struct (24 bytes).
        // The ABI should pass it indirectly via pointer.
        let (mut isel, entry) = make_empty_isel();

        let struct_ty = Type::Struct(vec![Type::I64, Type::I64, Type::I64]);
        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), struct_ty);

        let result_types = vec![Type::I32];
        isel.select_call("callee_big", &[Value(0)], &[Value(1)], &result_types, entry)
            .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Large aggregate -> Indirect{X0}, so should emit MovR to X0
        let mov_x0 = mblock.insts.iter().find(|i| {
            i.opcode == AArch64Opcode::MovR
                && i.operands.first() == Some(&ISelOperand::PReg(gpr::X0))
        });
        assert!(
            mov_x0.is_some(),
            "Expected MOV X0 for large aggregate indirect pass"
        );
    }

    #[test]
    fn select_call_returns_aggregate_via_sret() {
        // Call a function that returns an aggregate (via X8 sret)
        let (mut isel, entry) = make_empty_isel();

        let struct_ty = Type::Struct(vec![Type::I64, Type::I64, Type::I64]);

        let result_types = vec![struct_ty];
        isel.select_call("returns_struct", &[], &[Value(0)], &result_types, entry)
            .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // After BL, should copy X8 to a vreg for the result
        let has_bl = mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::Bl);
        assert!(has_bl);

        // Result should be defined (the MovR from X8)
        let mov_from_x8 = mblock.insts.iter().find(|i| {
            i.opcode == AArch64Opcode::MovR
                && i.operands.get(1) == Some(&ISelOperand::PReg(gpr::X8))
        });
        assert!(
            mov_from_x8.is_some(),
            "Expected MOV from X8 for sret result"
        );
    }

    #[test]
    fn struct_gep_three_field_struct() {
        // struct { I8, I32, I64 } -> test all field offsets
        let (mut isel, entry) = make_empty_isel();

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), Type::I64);

        let struct_ty = Type::Struct(vec![Type::I8, Type::I32, Type::I64]);
        // Verify expected offsets from the type system
        assert_eq!(struct_ty.offset_of(0), Some(0));
        assert_eq!(struct_ty.offset_of(1), Some(4)); // 1 byte + 3 padding
        assert_eq!(struct_ty.offset_of(2), Some(8)); // 4+4=8, already aligned to 8

        // Field 2 at offset 8
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::StructGep {
                    struct_ty,
                    field_index: 2,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::AddRI);
        assert_eq!(mblock.insts[0].operands[2], ISelOperand::Imm(8));
    }

    // =======================================================================
    // Extending loads and truncating stores (byte/halfword)
    // =======================================================================

    #[test]
    fn select_load_i8_emits_ldrb() {
        let (mut isel, entry) = make_empty_isel();
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Load { ty: Type::I8 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::LdrbRI,
            "Load I8 should emit LDRBui (zero-extending byte load)"
        );
    }

    #[test]
    fn select_load_i16_emits_ldrh() {
        let (mut isel, entry) = make_empty_isel();
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Load { ty: Type::I16 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::LdrhRI,
            "Load I16 should emit LDRHui (zero-extending halfword load)"
        );
    }

    #[test]
    fn select_store_i8_emits_strb() {
        let (mut isel, entry) = make_empty_isel();
        let val_vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), ISelOperand::VReg(val_vreg), Type::I8);
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Store,
                args: vec![Value(0), Value(1)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::StrbRI,
            "Store I8 should emit StrbRI (truncating byte store)"
        );
    }

    #[test]
    fn select_store_i16_emits_strh() {
        let (mut isel, entry) = make_empty_isel();
        let val_vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), ISelOperand::VReg(val_vreg), Type::I16);
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Store,
                args: vec![Value(0), Value(1)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::StrhRI,
            "Store I16 should emit StrhRI (truncating halfword store)"
        );
    }

    #[test]
    fn select_load_i8_add_store_i8_roundtrip() {
        let (mut isel, entry) = make_empty_isel();
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(addr_vreg), Type::I64);
        let addr_vreg2 = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), ISelOperand::VReg(addr_vreg2), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Load { ty: Type::I8 },
                args: vec![Value(0)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Load { ty: Type::I8 },
                args: vec![Value(1)],
                results: vec![Value(3)],
            },
            entry,
        )
        .unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(2), Value(3)],
                results: vec![Value(4)],
            },
            entry,
        )
        .unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Store,
                args: vec![Value(4), Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;
        assert_eq!(
            insts[0].opcode,
            AArch64Opcode::LdrbRI,
            "First instruction should be byte load"
        );
        assert_eq!(
            insts[1].opcode,
            AArch64Opcode::LdrbRI,
            "Second instruction should be byte load"
        );
        assert_eq!(
            insts[2].opcode,
            AArch64Opcode::AddRR,
            "Third instruction should be 32-bit add (I8 uses Gpr32)"
        );
        assert_eq!(
            insts[3].opcode,
            AArch64Opcode::StrbRI,
            "Fourth instruction should be byte store"
        );
    }

    #[test]
    fn select_large_struct_return_with_trailing_bytes() {
        // Return a struct with 17 I8 fields (17 bytes, align 1).
        // sret path: 2x 8-byte LDR/STR + 1x 1-byte LDRB/STRB tail.
        let struct_ty = Type::Struct(vec![Type::I8; 17]);
        assert_eq!(struct_ty.bytes(), 17);

        let sig = Signature {
            params: vec![],
            returns: vec![struct_ty.clone()],
        };
        let mut isel = InstructionSelector::new("ret_tail".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), struct_ty);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        let insts = &mblock.insts;

        // Count stores to X8
        let str_to_x8: Vec<_> = insts
            .iter()
            .filter(|i| {
                (i.opcode == AArch64Opcode::StrRI || i.opcode == AArch64Opcode::StrbRI)
                    && i.operands.get(1) == Some(&ISelOperand::PReg(gpr::X8))
            })
            .collect();
        // 2x StrRI (8-byte each) + 1x StrbRI (1-byte tail) = 3 stores
        assert_eq!(
            str_to_x8.len(),
            3,
            "Expected 3 stores for 17-byte struct sret: 2x8 + 1x1"
        );

        // Verify the tail store is a byte store
        let tail_store = str_to_x8.last().unwrap();
        assert_eq!(
            tail_store.opcode,
            AArch64Opcode::StrbRI,
            "Trailing byte should use STRB"
        );
        assert_eq!(
            tail_store.operands.get(2),
            Some(&ISelOperand::Imm(16)),
            "Trailing byte store should be at offset 16"
        );

        assert_eq!(insts.last().unwrap().opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn select_large_struct_return_with_trailing_halfword() {
        // Return a struct with a trailing 2-byte field: 3x I64 + 1x I16 = 26 bytes
        // (padded to 32 by align_to(26, 8) = 32).
        // sret: 4x 8-byte = 32 bytes copied. No tails needed.
        //
        // Better: 9x I16 = 18 bytes, align 2, align_to(18, 2) = 18.
        let struct_ty = Type::Struct(vec![Type::I16; 9]);
        assert_eq!(struct_ty.bytes(), 18);

        let sig = Signature {
            params: vec![],
            returns: vec![struct_ty.clone()],
        };
        let mut isel = InstructionSelector::new("ret_hw_tail".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), struct_ty);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        let insts = &mblock.insts;

        // 18 bytes: 2x 8-byte StrRI (16), then 1x 2-byte StrhRI (18)
        // No 4-byte or 1-byte tails.
        let str_to_x8: Vec<_> = insts
            .iter()
            .filter(|i| {
                (i.opcode == AArch64Opcode::StrRI || i.opcode == AArch64Opcode::StrhRI)
                    && i.operands.get(1) == Some(&ISelOperand::PReg(gpr::X8))
            })
            .collect();
        assert_eq!(
            str_to_x8.len(),
            3,
            "Expected 3 stores for 18-byte struct: 2x8 + 1x2"
        );

        let tail_store = str_to_x8.last().unwrap();
        assert_eq!(
            tail_store.opcode,
            AArch64Opcode::StrhRI,
            "Trailing halfword should use STRH"
        );
        assert_eq!(
            tail_store.operands.get(2),
            Some(&ISelOperand::Imm(16)),
            "Trailing halfword store should be at offset 16"
        );

        assert_eq!(insts.last().unwrap().opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn select_large_struct_return_with_4_2_1_tail() {
        // 23 bytes: 2x 8-byte (16) + 1x 4-byte (20) + 1x 2-byte (22) + 1x 1-byte (23)
        // Use 23x I8 -> bytes=23, align=1
        let struct_ty = Type::Struct(vec![Type::I8; 23]);
        assert_eq!(struct_ty.bytes(), 23);

        let sig = Signature {
            params: vec![],
            returns: vec![struct_ty.clone()],
        };
        let mut isel = InstructionSelector::new("ret_421_tail".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        let vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), struct_ty);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        let insts = &mblock.insts;

        // Count all stores to X8
        let str_to_x8: Vec<_> = insts
            .iter()
            .filter(|i| {
                matches!(
                    i.opcode,
                    AArch64Opcode::StrRI | AArch64Opcode::StrhRI | AArch64Opcode::StrbRI
                ) && i.operands.get(1) == Some(&ISelOperand::PReg(gpr::X8))
            })
            .collect();
        // 2x StrRI + 1x StrRI + 1x StrhRI + 1x StrbRI = 5 stores
        assert_eq!(
            str_to_x8.len(),
            5,
            "Expected 5 stores for 23-byte struct: 2x8 + 1x4 + 1x2 + 1x1"
        );

        // Verify tail opcodes in order
        assert_eq!(str_to_x8[0].opcode, AArch64Opcode::StrRI);
        assert_eq!(str_to_x8[1].opcode, AArch64Opcode::StrRI);
        assert_eq!(str_to_x8[2].opcode, AArch64Opcode::StrRI);
        assert_eq!(str_to_x8[3].opcode, AArch64Opcode::StrhRI);
        assert_eq!(str_to_x8[4].opcode, AArch64Opcode::StrbRI);

        // Verify offsets
        assert_eq!(str_to_x8[0].operands.get(2), Some(&ISelOperand::Imm(0)));
        assert_eq!(str_to_x8[1].operands.get(2), Some(&ISelOperand::Imm(8)));
        assert_eq!(str_to_x8[2].operands.get(2), Some(&ISelOperand::Imm(16)));
        assert_eq!(str_to_x8[3].operands.get(2), Some(&ISelOperand::Imm(20)));
        assert_eq!(str_to_x8[4].operands.get(2), Some(&ISelOperand::Imm(22)));

        assert_eq!(insts.last().unwrap().opcode, AArch64Opcode::Ret);
    }

    // =======================================================================
    // Variadic call ISel tests (Apple AArch64 ABI, issue #79)
    // =======================================================================

    #[test]
    fn select_variadic_call_printf_like() {
        // printf(const char* fmt, ...) called as printf(fmt, 42, 100)
        // fixed_count=1: fmt -> X0 (register)
        // variadic: 42(I32) -> stack[0], 100(I64) -> stack[8]
        let (mut isel, entry) = make_empty_isel();

        // Define arg values
        let v0_reg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(v0_reg), Type::I64); // fmt
        let v1_reg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(1), ISelOperand::VReg(v1_reg), Type::I32); // 42
        let v2_reg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(2), ISelOperand::VReg(v2_reg), Type::I64); // 100

        let result_types = vec![Type::I32]; // printf returns int
        isel.select_variadic_call(
            "printf",
            1, // 1 fixed arg (fmt)
            &[Value(0), Value(1), Value(2)],
            &[Value(3)],
            &result_types,
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        let insts = &mblock.insts;

        // Should have: Copy X0 (fmt), STR [SP+0] (42), STR [SP+8] (100), BL, Copy (result)
        // First: Copy to X0 for fixed arg
        let mov_x0 = insts.iter().find(|i| {
            i.opcode == AArch64Opcode::Copy
                && i.operands.first() == Some(&ISelOperand::PReg(gpr::X0))
        });
        assert!(mov_x0.is_some(), "Fixed arg should go to X0");

        // Variadic args should be STR to stack
        let str_sp: Vec<_> = insts
            .iter()
            .filter(|i| {
                matches!(i.opcode, AArch64Opcode::StrRI)
                    && i.operands.get(1) == Some(&ISelOperand::PReg(SP))
            })
            .collect();
        assert_eq!(
            str_sp.len(),
            2,
            "Two variadic args should be stored to stack"
        );

        // Verify offsets: first at 0, second at 8
        assert_eq!(str_sp[0].operands.get(2), Some(&ISelOperand::Imm(0)));
        assert_eq!(str_sp[1].operands.get(2), Some(&ISelOperand::Imm(8)));

        // BL should be present
        let bl_inst = insts.iter().find(|i| i.opcode == AArch64Opcode::Bl);
        assert!(bl_inst.is_some(), "Expected BL instruction");
        assert_eq!(
            bl_inst.unwrap().operands[0],
            ISelOperand::Symbol("printf".to_string()),
        );
    }

    #[test]
    fn select_variadic_call_float_on_stack() {
        // Apple ABI: variadic floats go on stack, NOT in FPR.
        // fn(i64, ...) called with (ptr, 1.0f64)
        let (mut isel, entry) = make_empty_isel();

        let v0_reg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(v0_reg), Type::I64);
        let v1_reg = isel.new_vreg(RegClass::Fpr64);
        isel.define_value(Value(1), ISelOperand::VReg(v1_reg), Type::F64);

        let result_types = vec![Type::I32];
        isel.select_variadic_call(
            "my_varfn",
            1,
            &[Value(0), Value(1)],
            &[Value(2)],
            &result_types,
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        let insts = &mblock.insts;

        // The variadic F64 should be stored to stack using StrRI (not MOV to V0)
        let str_d_sp = insts.iter().find(|i| {
            i.opcode == AArch64Opcode::StrRI && i.operands.get(1) == Some(&ISelOperand::PReg(SP))
        });
        assert!(str_d_sp.is_some(), "Variadic f64 should use StrRI to stack");
        assert_eq!(
            str_d_sp.unwrap().operands.get(2),
            Some(&ISelOperand::Imm(0)),
            "Variadic f64 at stack offset 0"
        );

        // V0 should NOT be used (no MOV to V0 for variadic float)
        let mov_v0 = insts
            .iter()
            .find(|i| i.operands.first() == Some(&ISelOperand::PReg(gpr::V0)));
        assert!(
            mov_v0.is_none(),
            "Variadic float should NOT go in V0 (Apple ABI)"
        );
    }

    #[test]
    fn select_variadic_call_via_opcode() {
        // Test the CallVariadic opcode dispatch through select_instruction.
        let (mut isel, entry) = make_empty_isel();

        let v0_reg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(v0_reg), Type::I64); // fmt
        let v1_reg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(1), ISelOperand::VReg(v1_reg), Type::I32); // vararg

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::CallVariadic {
                    name: "NSLog".to_string(),
                    fixed_args: 1,
                },
                args: vec![Value(0), Value(1)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // MOV X0 (fixed), STR [SP+0] (variadic), BL NSLog
        let bl_inst = mblock.insts.iter().find(|i| i.opcode == AArch64Opcode::Bl);
        assert!(bl_inst.is_some());
        assert_eq!(
            bl_inst.unwrap().operands[0],
            ISelOperand::Symbol("NSLog".to_string()),
        );

        let str_sp = mblock.insts.iter().find(|i| {
            matches!(i.opcode, AArch64Opcode::StrRI)
                && i.operands.get(1) == Some(&ISelOperand::PReg(SP))
        });
        assert!(str_sp.is_some(), "Variadic i32 arg should be on stack");
    }

    #[test]
    fn select_variadic_call_no_varargs() {
        // Variadic function called with only fixed args (no varargs passed).
        // Should behave identically to a normal call.
        let (mut isel, entry) = make_empty_isel();

        let v0_reg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(v0_reg), Type::I64);

        let result_types = vec![Type::I32];
        isel.select_variadic_call("printf", 1, &[Value(0)], &[Value(1)], &result_types, entry)
            .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Fixed arg in X0, BL, result from X0
        let mov_x0 = mblock.insts.iter().find(|i| {
            i.opcode == AArch64Opcode::Copy
                && i.operands.first() == Some(&ISelOperand::PReg(gpr::X0))
        });
        assert!(mov_x0.is_some(), "Fixed arg should go to X0");

        // No stack stores (no varargs)
        let str_sp = mblock.insts.iter().any(|i| {
            i.operands.get(1) == Some(&ISelOperand::PReg(SP))
                && matches!(
                    i.opcode,
                    AArch64Opcode::StrRI
                        | AArch64Opcode::StrbRI
                        | AArch64Opcode::StrhRI
                        | AArch64Opcode::StpRI
                )
        });
        assert!(!str_sp, "No stack stores when no varargs passed");
    }

    // =======================================================================
    // CallIndirect (indirect call via function pointer)
    // =======================================================================

    #[test]
    fn select_call_indirect_basic() {
        // Test: call_indirect(fn_ptr, arg) -> result
        // fn_ptr is I64, arg is I32, returns I32
        let (mut isel, entry) = make_empty_isel();

        // Define the function pointer value
        let fp_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(fp_vreg), Type::I64);

        // Define an argument value
        let arg_vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(1), ISelOperand::VReg(arg_vreg), Type::I32);

        // Select CallIndirect: args[0]=fn_ptr, args[1]=arg, results=[retval]
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::CallIndirect,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Should contain: MOV (arg to X0), MOV (fn_ptr to X16), BLR X16, MOV (X0 to result)
        // Find the BLR instruction
        let blr_inst = mblock.insts.iter().find(|i| i.opcode == AArch64Opcode::Blr);
        assert!(
            blr_inst.is_some(),
            "Expected BLR instruction for indirect call"
        );
        let blr = blr_inst.unwrap();
        assert_eq!(
            blr.operands[0],
            ISelOperand::PReg(gpr::X16),
            "BLR should target X16 (IP0 scratch register)"
        );

        // Verify arg was moved to X0 via Copy pseudo
        let mov_x0 = mblock.insts.iter().find(|i| {
            i.opcode == AArch64Opcode::Copy
                && i.operands.first() == Some(&ISelOperand::PReg(gpr::X0))
        });
        assert!(mov_x0.is_some(), "Arg should be copied to X0");

        // Verify fn_ptr was moved to X16 (always GPR MovR)
        let mov_x16 = mblock.insts.iter().find(|i| {
            i.opcode == AArch64Opcode::MovR
                && i.operands.first() == Some(&ISelOperand::PReg(gpr::X16))
        });
        assert!(mov_x16.is_some(), "Function pointer should be moved to X16");
    }

    #[test]
    fn select_call_indirect_no_args_no_results() {
        // Test: call_indirect(fn_ptr) with no args and no return
        let (mut isel, entry) = make_empty_isel();

        let fp_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(fp_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::CallIndirect,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Should have: MOV fn_ptr -> X16, BLR X16
        assert!(mblock.insts.len() >= 2, "At least MOV + BLR");

        let blr_inst = mblock.insts.iter().find(|i| i.opcode == AArch64Opcode::Blr);
        assert!(blr_inst.is_some(), "Expected BLR for indirect call");
    }

    #[test]
    fn select_call_indirect_multiple_args() {
        // Test: call_indirect(fn_ptr, a, b, c) -> result
        let (mut isel, entry) = make_empty_isel();

        let fp_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(fp_vreg), Type::I64);

        let a_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), ISelOperand::VReg(a_vreg), Type::I64);
        let b_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(2), ISelOperand::VReg(b_vreg), Type::I64);
        let c_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(3), ISelOperand::VReg(c_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::CallIndirect,
                args: vec![Value(0), Value(1), Value(2), Value(3)],
                results: vec![Value(4)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Verify args go to X0, X1, X2 via Copy pseudo
        let mov_x0 = mblock.insts.iter().any(|i| {
            i.opcode == AArch64Opcode::Copy
                && i.operands.first() == Some(&ISelOperand::PReg(gpr::X0))
        });
        let mov_x1 = mblock.insts.iter().any(|i| {
            i.opcode == AArch64Opcode::Copy
                && i.operands.first() == Some(&ISelOperand::PReg(gpr::X1))
        });
        let mov_x2 = mblock.insts.iter().any(|i| {
            i.opcode == AArch64Opcode::Copy
                && i.operands.first() == Some(&ISelOperand::PReg(gpr::X2))
        });
        assert!(mov_x0, "First arg should be copied to X0");
        assert!(mov_x1, "Second arg should be copied to X1");
        assert!(mov_x2, "Third arg should be copied to X2");

        // BLR should be present
        let blr = mblock.insts.iter().find(|i| i.opcode == AArch64Opcode::Blr);
        assert!(blr.is_some(), "Expected BLR");
    }

    // =======================================================================
    // Switch (cascading CMP+B.EQ chain)
    // =======================================================================

    #[test]
    fn select_switch_two_cases() {
        // switch(selector) { 0 => block1, 1 => block2, default => block3 }
        let (mut isel, entry) = make_empty_isel();

        let sel_vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), ISelOperand::VReg(sel_vreg), Type::I32);

        let block1 = Block(1);
        let block2 = Block(2);
        let block3 = Block(3);
        isel.func.ensure_block(block1);
        isel.func.ensure_block(block2);
        isel.func.ensure_block(block3);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Switch {
                    cases: vec![(0, block1), (1, block2)],
                    default: block3,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Expected pattern:
        //   CMP sel, #0; B.EQ block1
        //   CMP sel, #1; B.EQ block2
        //   B block3 (default)
        //
        // That's 2*(CMP+B.cond) + 1*B = 5 instructions

        // Count B.cond instructions
        let bcc_count = mblock
            .insts
            .iter()
            .filter(|i| i.opcode == AArch64Opcode::BCond)
            .count();
        assert_eq!(
            bcc_count, 2,
            "Should have 2 conditional branches (one per case)"
        );

        // Each B.cond should use EQ condition
        for bcc in mblock
            .insts
            .iter()
            .filter(|i| i.opcode == AArch64Opcode::BCond)
        {
            assert_eq!(
                bcc.operands[0],
                ISelOperand::CondCode(AArch64CC::EQ),
                "Switch cases use B.EQ"
            );
        }

        // The last instruction should be unconditional B to default
        let last = mblock.insts.last().unwrap();
        assert_eq!(
            last.opcode,
            AArch64Opcode::B,
            "Last inst should be B (default fallthrough)"
        );
        assert_eq!(
            last.operands[0],
            ISelOperand::Block(block3),
            "Default should be block3"
        );

        // Verify successors
        let succs = &mfunc.blocks[&entry].successors;
        assert!(succs.contains(&block1), "block1 should be a successor");
        assert!(succs.contains(&block2), "block2 should be a successor");
        assert!(
            succs.contains(&block3),
            "block3 (default) should be a successor"
        );
    }

    #[test]
    fn select_switch_large_case_value() {
        // Test case value > 4095 (doesn't fit in CMP immediate)
        let (mut isel, entry) = make_empty_isel();

        let sel_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(sel_vreg), Type::I64);

        let block1 = Block(1);
        let block2 = Block(2);
        isel.func.ensure_block(block1);
        isel.func.ensure_block(block2);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Switch {
                    cases: vec![(0x10000, block1)],
                    default: block2,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // For large case value: MOVZ + CMP reg,reg + B.EQ + B
        // MOVZ materializes the constant, then CMP is register-register
        let movz_inst = mblock
            .insts
            .iter()
            .find(|i| i.opcode == AArch64Opcode::Movz);
        assert!(
            movz_inst.is_some(),
            "Large case value should be materialized via Movz"
        );

        let cmp_rr = mblock
            .insts
            .iter()
            .find(|i| i.opcode == AArch64Opcode::CmpRR);
        assert!(cmp_rr.is_some(), "Large case value should use CMP reg,reg");
    }

    #[test]
    fn select_switch_single_case() {
        // Degenerate switch with one case
        let (mut isel, entry) = make_empty_isel();

        let sel_vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), ISelOperand::VReg(sel_vreg), Type::I32);

        let block1 = Block(1);
        let block_default = Block(2);
        isel.func.ensure_block(block1);
        isel.func.ensure_block(block_default);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Switch {
                    cases: vec![(42, block1)],
                    default: block_default,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // CMP + B.EQ + B = 3 instructions
        let bcc_count = mblock
            .insts
            .iter()
            .filter(|i| i.opcode == AArch64Opcode::BCond)
            .count();
        assert_eq!(bcc_count, 1, "Single case should produce 1 B.EQ");

        let last = mblock.insts.last().unwrap();
        assert_eq!(last.opcode, AArch64Opcode::B);
        assert_eq!(last.operands[0], ISelOperand::Block(block_default));
    }

    #[test]
    fn select_switch_zero_cases() {
        // Switch with no cases = just jump to default
        let (mut isel, entry) = make_empty_isel();

        let sel_vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), ISelOperand::VReg(sel_vreg), Type::I32);

        let block_default = Block(1);
        isel.func.ensure_block(block_default);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Switch {
                    cases: vec![],
                    default: block_default,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Just one B to default
        assert_eq!(mblock.insts.len(), 1, "Empty switch = just B to default");
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::B);
        assert_eq!(
            mblock.insts[0].operands[0],
            ISelOperand::Block(block_default)
        );
    }

    // =======================================================================
    // Switch jump table lowering
    // =======================================================================

    #[test]
    fn select_switch_dense_jump_table() {
        // 8 consecutive cases (0..7) -> jump table
        let (mut isel, entry) = make_empty_isel();

        let sel_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(sel_vreg), Type::I64);

        let default_block = Block(9);
        isel.func.ensure_block(default_block);

        let mut cases = Vec::new();
        for i in 0..8 {
            let blk = Block(i + 1);
            isel.func.ensure_block(blk);
            cases.push((i as i64, blk));
        }

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Switch {
                    cases,
                    default: default_block,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Jump table path: should have Adr, LdrswRO, Br and NO BCond with EQ
        let has_adr = mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::Adr);
        let has_ldrsw = mblock
            .insts
            .iter()
            .any(|i| i.opcode == AArch64Opcode::LdrswRO);
        let has_br = mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::Br);
        let eq_bcond_count = mblock
            .insts
            .iter()
            .filter(|i| {
                i.opcode == AArch64Opcode::BCond
                    && i.operands[0] == ISelOperand::CondCode(AArch64CC::EQ)
            })
            .count();

        assert!(has_adr, "Dense switch should emit ADR for jump table base");
        assert!(
            has_ldrsw,
            "Dense switch should emit LDRSW for jump table load"
        );
        assert!(has_br, "Dense switch should emit BR for indirect branch");
        assert_eq!(
            eq_bcond_count, 0,
            "Dense switch should NOT use B.EQ cascade"
        );

        // Verify B.HI for the range check
        let hi_bcond = mblock.insts.iter().find(|i| {
            i.opcode == AArch64Opcode::BCond
                && i.operands[0] == ISelOperand::CondCode(AArch64CC::HI)
        });
        assert!(
            hi_bcond.is_some(),
            "Jump table should have B.HI for range check"
        );
    }

    #[test]
    fn select_switch_sparse_cascade() {
        // 3 sparse cases (0, 100, 200) -> cascade (too few + sparse)
        let (mut isel, entry) = make_empty_isel();

        let sel_vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), ISelOperand::VReg(sel_vreg), Type::I32);

        let block1 = Block(1);
        let block2 = Block(2);
        let block3 = Block(3);
        let block_default = Block(4);
        isel.func.ensure_block(block1);
        isel.func.ensure_block(block2);
        isel.func.ensure_block(block3);
        isel.func.ensure_block(block_default);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Switch {
                    cases: vec![(0, block1), (100, block2), (200, block3)],
                    default: block_default,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Cascade path: should have B.EQ conditional branches, NO Adr
        let eq_bcond_count = mblock
            .insts
            .iter()
            .filter(|i| {
                i.opcode == AArch64Opcode::BCond
                    && i.operands[0] == ISelOperand::CondCode(AArch64CC::EQ)
            })
            .count();
        let has_adr = mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::Adr);

        assert_eq!(eq_bcond_count, 3, "Sparse switch should use B.EQ cascade");
        assert!(!has_adr, "Sparse switch should NOT use jump table (ADR)");
    }

    #[test]
    fn select_switch_dense_with_holes() {
        // Cases [0,1,2,4,5,6,7] with hole at 3, density = 7/8 = 0.875
        let (mut isel, entry) = make_empty_isel();

        let sel_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(sel_vreg), Type::I64);

        let default_block = Block(9);
        isel.func.ensure_block(default_block);

        let case_vals: Vec<i64> = vec![0, 1, 2, 4, 5, 6, 7];
        let mut cases = Vec::new();
        for (idx, &val) in case_vals.iter().enumerate() {
            let blk = Block(idx as u32 + 1);
            isel.func.ensure_block(blk);
            cases.push((val, blk));
        }

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Switch {
                    cases,
                    default: default_block,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Dense path: should use jump table
        let has_adr = mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::Adr);
        let has_ldrsw = mblock
            .insts
            .iter()
            .any(|i| i.opcode == AArch64Opcode::LdrswRO);

        assert!(
            has_adr,
            "Dense-with-holes switch should use jump table (ADR)"
        );
        assert!(has_ldrsw, "Dense-with-holes switch should use LDRSW");

        // The jump table should have 8 entries (0..7), with index 3 -> default.
        // Look up via `jump_tables` side-table keyed by the JumpTableIndex operand.
        let adr_inst = mblock
            .insts
            .iter()
            .find(|i| i.opcode == AArch64Opcode::Adr)
            .unwrap();
        let jt_idx = if let ISelOperand::JumpTableIndex(idx) = &adr_inst.operands[1] {
            *idx
        } else {
            panic!(
                "ADR operand[1] should be JumpTableIndex, got {:?}",
                adr_inst.operands[1]
            );
        };
        let jt = &mfunc.jump_tables[jt_idx as usize];
        assert_eq!(
            jt.targets.len(),
            8,
            "Jump table should have 8 entries for range 0..7"
        );
        assert_eq!(
            jt.targets[3], default_block,
            "Hole at index 3 should map to default"
        );
    }

    #[test]
    fn select_switch_density_threshold_exact() {
        // 4 cases spanning range 10: density = 4/10 = 0.4 (not > 0.4) -> binary search
        let (mut isel, entry) = make_empty_isel();

        let sel_vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), ISelOperand::VReg(sel_vreg), Type::I32);

        let default_block = Block(5);
        isel.func.ensure_block(default_block);

        let case_vals: Vec<i64> = vec![0, 3, 6, 9]; // range = 10, density = 4/10 = 0.4
        let mut cases = Vec::new();
        for (idx, &val) in case_vals.iter().enumerate() {
            let blk = Block(idx as u32 + 1);
            isel.func.ensure_block(blk);
            cases.push((val, blk));
        }

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Switch {
                    cases,
                    default: default_block,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        let has_adr = mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::Adr);
        assert!(
            !has_adr,
            "Density exactly 0.4 should NOT use jump table (need > 0.4)"
        );

        // With 4 sparse cases (density = 0.4), binary search is used.
        // Root BST node: CMP against pivot, B.EQ, B.LT, B (to right subtree).
        let has_blt = mblock.insts.iter().any(|i| {
            i.opcode == AArch64Opcode::BCond
                && i.operands[0] == ISelOperand::CondCode(AArch64CC::LT)
        });
        assert!(has_blt, "Sparse 4-case switch should use BST with B.LT");
    }

    #[test]
    fn select_switch_density_above_threshold() {
        // 5 cases spanning range 8: density = 5/8 = 0.625 -> jump table
        let (mut isel, entry) = make_empty_isel();

        let sel_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(sel_vreg), Type::I64);

        let default_block = Block(6);
        isel.func.ensure_block(default_block);

        let case_vals: Vec<i64> = vec![0, 1, 3, 5, 7]; // range = 8, density = 5/8 = 0.625
        let mut cases = Vec::new();
        for (idx, &val) in case_vals.iter().enumerate() {
            let blk = Block(idx as u32 + 1);
            isel.func.ensure_block(blk);
            cases.push((val, blk));
        }

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Switch {
                    cases,
                    default: default_block,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        let has_adr = mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::Adr);
        let has_br = mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::Br);

        assert!(has_adr, "Density 0.625 should use jump table");
        assert!(has_br, "Jump table should end with BR");
    }

    #[test]
    fn select_switch_negative_cases() {
        // Cases [-3,-2,-1,0,1,2], range = 6, density = 6/6 = 1.0 -> jump table
        let (mut isel, entry) = make_empty_isel();

        let sel_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(sel_vreg), Type::I64);

        let default_block = Block(7);
        isel.func.ensure_block(default_block);

        let case_vals: Vec<i64> = vec![-3, -2, -1, 0, 1, 2];
        let mut cases = Vec::new();
        for (idx, &val) in case_vals.iter().enumerate() {
            let blk = Block(idx as u32 + 1);
            isel.func.ensure_block(blk);
            cases.push((val, blk));
        }

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Switch {
                    cases,
                    default: default_block,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Should use jump table even with negative case values
        let has_adr = mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::Adr);
        assert!(has_adr, "Negative case values should still use jump table");

        // min_val = -3, which doesn't fit in imm12, so should use Movz+SubRR
        // (or the SubRI path won't work since min_val < 0)
        let has_movz = mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::Movz);
        let has_sub_rr = mblock
            .insts
            .iter()
            .any(|i| i.opcode == AArch64Opcode::SubRR);
        assert!(has_movz, "Negative min_val should be materialized via Movz");
        assert!(has_sub_rr, "Negative min_val should use SubRR");

        // Verify the jump table has 6 entries (lookup via side-table).
        let adr_inst = mblock
            .insts
            .iter()
            .find(|i| i.opcode == AArch64Opcode::Adr)
            .unwrap();
        let jt_idx = if let ISelOperand::JumpTableIndex(idx) = &adr_inst.operands[1] {
            *idx
        } else {
            panic!(
                "ADR operand[1] should be JumpTableIndex, got {:?}",
                adr_inst.operands[1]
            );
        };
        let jt = &mfunc.jump_tables[jt_idx as usize];
        assert_eq!(jt.min_val, -3, "min_val should be -3");
        assert_eq!(jt.targets.len(), 6, "Jump table should have 6 entries");
    }

    #[test]
    fn select_switch_three_cases_no_jump_table() {
        // 3 consecutive cases (too few for jump table, minimum is 4)
        let (mut isel, entry) = make_empty_isel();

        let sel_vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), ISelOperand::VReg(sel_vreg), Type::I32);

        let block1 = Block(1);
        let block2 = Block(2);
        let block3 = Block(3);
        let block_default = Block(4);
        isel.func.ensure_block(block1);
        isel.func.ensure_block(block2);
        isel.func.ensure_block(block3);
        isel.func.ensure_block(block_default);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Switch {
                    cases: vec![(0, block1), (1, block2), (2, block3)],
                    default: block_default,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Even though density = 1.0, 3 cases < 4 minimum -> cascade
        let has_adr = mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::Adr);
        let eq_bcond_count = mblock
            .insts
            .iter()
            .filter(|i| {
                i.opcode == AArch64Opcode::BCond
                    && i.operands[0] == ISelOperand::CondCode(AArch64CC::EQ)
            })
            .count();

        assert!(!has_adr, "3 cases should NOT use jump table (minimum 4)");
        assert_eq!(eq_bcond_count, 3, "3 cases should use B.EQ cascade");
    }

    // =======================================================================
    // Switch binary search tree lowering (via select_instruction)
    // =======================================================================

    #[test]
    fn select_switch_binary_search_sparse() {
        // 6 sparse cases with density = 6/600 = 0.01 -> binary search
        let (mut isel, entry) = make_empty_isel();

        let sel_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(sel_vreg), Type::I64);

        let default_block = Block(7);
        isel.func.ensure_block(default_block);

        let case_vals: Vec<i64> = vec![0, 100, 200, 300, 400, 500];
        let mut cases = Vec::new();
        for (idx, &val) in case_vals.iter().enumerate() {
            let blk = Block(idx as u32 + 1);
            isel.func.ensure_block(blk);
            cases.push((val, blk));
        }

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Switch {
                    cases: cases.clone(),
                    default: default_block,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // BST path: should have B.LT (BST branching), NO ADR (no jump table)
        let has_adr = mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::Adr);
        assert!(!has_adr, "Sparse 6-case switch should NOT use jump table");

        // Root BST node should CMP against pivot (median) and use B.LT
        let has_blt = mblock.insts.iter().any(|i| {
            i.opcode == AArch64Opcode::BCond
                && i.operands[0] == ISelOperand::CondCode(AArch64CC::LT)
        });
        assert!(has_blt, "BST root should have B.LT for left subtree");

        // Root should also have B.EQ for exact match
        let has_beq = mblock.insts.iter().any(|i| {
            i.opcode == AArch64Opcode::BCond
                && i.operands[0] == ISelOperand::CondCode(AArch64CC::EQ)
        });
        assert!(has_beq, "BST root should have B.EQ for pivot match");

        // Should create intermediate blocks (more than original 8)
        assert!(
            mfunc.blocks.len() > 8,
            "BST should create intermediate blocks: got {}",
            mfunc.blocks.len()
        );
    }

    #[test]
    fn select_switch_binary_search_all_targets_reachable() {
        // 8 sparse cases -> binary search; verify all targets appear as successors
        let (mut isel, entry) = make_empty_isel();

        let sel_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(sel_vreg), Type::I64);

        let default_block = Block(9);
        isel.func.ensure_block(default_block);

        let mut cases = Vec::new();
        for i in 0..8 {
            let blk = Block(i + 1);
            isel.func.ensure_block(blk);
            cases.push((i as i64 * 1000, blk)); // values: 0, 1000, 2000, ..., 7000
        }

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Switch {
                    cases: cases.clone(),
                    default: default_block,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();

        // Collect all successor blocks reachable from any block
        let all_succs: Vec<Block> = mfunc
            .blocks
            .values()
            .flat_map(|b| b.successors.iter().copied())
            .collect();

        for i in 1..=8 {
            assert!(
                all_succs.contains(&Block(i)),
                "Block({}) should be reachable in BST",
                i
            );
        }
        assert!(
            all_succs.contains(&default_block),
            "Default block should be reachable"
        );
    }

    #[test]
    fn select_switch_density_half_uses_jump_table() {
        // 4 cases, range = 8, density = 4/8 = 0.5 > 0.4 -> jump table (threshold is 0.4)
        let (mut isel, entry) = make_empty_isel();

        let sel_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(sel_vreg), Type::I64);

        let default_block = Block(5);
        isel.func.ensure_block(default_block);

        let case_vals: Vec<i64> = vec![0, 2, 5, 7]; // range = 8, density = 4/8 = 0.5
        let mut cases = Vec::new();
        for (idx, &val) in case_vals.iter().enumerate() {
            let blk = Block(idx as u32 + 1);
            isel.func.ensure_block(blk);
            cases.push((val, blk));
        }

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Switch {
                    cases,
                    default: default_block,
                },
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        let has_adr = mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::Adr);
        assert!(has_adr, "Density 0.5 > 0.4 threshold should use jump table");
    }

    // =======================================================================
    // Coverage expansion: multiplication and division ISel
    // =======================================================================

    #[test]
    fn select_imul_i32() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Imul,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::MulRR);
    }

    #[test]
    fn select_imul_i64() {
        let (mut isel, entry) = make_i64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Imul,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::MulRR);
    }

    #[test]
    fn select_sdiv_i32() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Sdiv,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::SDiv);
    }

    #[test]
    fn select_sdiv_i64() {
        let (mut isel, entry) = make_i64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Sdiv,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::SDiv);
    }

    #[test]
    fn select_udiv_i32() {
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Udiv,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::UDiv);
    }

    #[test]
    fn select_udiv_i64() {
        let (mut isel, entry) = make_i64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Udiv,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::UDiv);
    }

    // =======================================================================
    // Coverage expansion: unconditional jump
    // =======================================================================

    #[test]
    fn select_jump_unconditional() {
        let (mut isel, entry) = make_empty_isel();
        let target = Block(1);
        isel.func.ensure_block(target);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Jump { dest: target },
                args: vec![],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 1);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::B);
        assert_eq!(mblock.insts[0].operands[0], ISelOperand::Block(target));
        assert_eq!(mblock.successors.len(), 1);
        assert_eq!(mblock.successors[0], target);
    }

    // =======================================================================
    // Coverage expansion: load/store for 32-bit, 64-bit, and float types
    // =======================================================================

    #[test]
    fn select_load_i32_emits_ldr() {
        let (mut isel, entry) = make_empty_isel();
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Load { ty: Type::I32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::LdrRI,
            "Load I32 should emit LDR (word load)"
        );
    }

    #[test]
    fn select_load_i64_emits_ldr() {
        let (mut isel, entry) = make_empty_isel();
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Load { ty: Type::I64 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::LdrRI,
            "Load I64 should emit LDR (doubleword load)"
        );
    }

    #[test]
    fn select_load_f32_emits_ldr_s() {
        let (mut isel, entry) = make_empty_isel();
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Load { ty: Type::F32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        // F32 loads use LdrRI with FPR destination register class
        assert_eq!(
            inst.opcode,
            AArch64Opcode::LdrRI,
            "Load F32 should emit LDR (float single load)"
        );
    }

    #[test]
    fn select_load_f64_emits_ldr_d() {
        let (mut isel, entry) = make_empty_isel();
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Load { ty: Type::F64 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::LdrRI,
            "Load F64 should emit LDR (float double load)"
        );
    }

    #[test]
    fn select_store_i32_emits_str() {
        let (mut isel, entry) = make_empty_isel();
        let val_vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), ISelOperand::VReg(val_vreg), Type::I32);
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Store,
                args: vec![Value(0), Value(1)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::StrRI,
            "Store I32 should emit STR (word store)"
        );
    }

    #[test]
    fn select_store_i64_emits_str() {
        let (mut isel, entry) = make_empty_isel();
        let val_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(val_vreg), Type::I64);
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Store,
                args: vec![Value(0), Value(1)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::StrRI,
            "Store I64 should emit STR (doubleword store)"
        );
    }

    // =======================================================================
    // Coverage expansion: fconst materialization
    // =======================================================================

    #[test]
    fn select_fconst_f32() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fconst {
                    ty: Type::F32,
                    imm: 1.0,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, AArch64Opcode::FmovImm);
        assert_eq!(inst.operands[1], ISelOperand::FImm(1.0));
    }

    #[test]
    fn select_fconst_f64() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fconst {
                    ty: Type::F64,
                    imm: 2.78,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, AArch64Opcode::FmovImm);
    }

    #[test]
    fn select_fconst_invalid_type_errors() {
        let (mut isel, entry) = make_empty_isel();
        let result = isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fconst {
                    ty: Type::I32,
                    imm: 1.0,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        );
        assert!(result.is_err(), "Fconst with integer type should error");
    }

    // =======================================================================
    // Coverage expansion: Copy pseudo (formerly single-arg Iadd; see #417)
    // =======================================================================

    #[test]
    fn select_copy_pseudo_emits_mov() {
        let (mut isel, entry) = make_empty_isel();
        // Define a source value
        let vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), Type::I32);

        // Opcode::Copy lowers to MOV (register copy).
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Copy,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 1);
        assert_eq!(
            mblock.insts[0].opcode,
            AArch64Opcode::MovR,
            "Opcode::Copy should emit MOV (register copy)"
        );
    }

    #[test]
    fn single_arg_iadd_is_rejected() {
        // Regression guard for #417: single-arg Iadd was previously used as a
        // COPY placeholder. It must now be rejected explicitly.
        let (mut isel, entry) = make_empty_isel();
        let vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), Type::I32);

        let result = isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        );
        assert!(
            result.is_err(),
            "single-arg Iadd must be rejected; use Opcode::Copy"
        );
    }

    // =======================================================================
    // Coverage expansion: select_block batch processing
    // =======================================================================

    #[test]
    fn select_block_multiple_instructions() {
        let sig = Signature {
            params: vec![Type::I32, Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("batch".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        // Select an entire block of instructions at once
        let instructions = vec![
            Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            Instruction {
                opcode: Opcode::Return,
                args: vec![Value(2)],
                results: vec![],
            },
        ];
        isel.select_block(entry, &instructions).unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 2 COPY (formal args) + 1 AddRR + 1 Copy (to X0) + 1 Ret = 5
        assert_eq!(mblock.insts.len(), 5);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::AddRR);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::Ret);
    }

    // =======================================================================
    // Coverage expansion: to_ir_func conversion
    // =======================================================================

    #[test]
    fn isel_to_ir_func_conversion() {
        let sig = Signature {
            params: vec![Type::I32, Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("add".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(2)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let ir_func = mfunc.to_ir_func();

        // Verify basic properties of the converted IR function
        assert_eq!(ir_func.name, "add");
        assert_eq!(ir_func.block_order.len(), 1);
        assert_eq!(ir_func.next_vreg, 3);
        // Verify instructions were transferred
        let entry_block = &ir_func.blocks[ir_func.entry.0 as usize];
        assert_eq!(entry_block.insts.len(), 5); // 2 COPY + AddRR + Copy (ret) + Ret
    }

    // =======================================================================
    // Coverage expansion: error handling edge cases
    // =======================================================================

    #[test]
    fn select_binop_missing_args_errors() {
        let (mut isel, entry) = make_empty_isel();
        let vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), Type::I32);

        // Isub with only 1 arg — binop path requires 2 args.
        // (Note: after #417, single-arg Iadd is a rejected error case;
        //  use Opcode::Copy for single-source moves.)
        let result = isel.select_instruction(
            &Instruction {
                opcode: Opcode::Isub, // requires 2 args
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        );
        assert!(result.is_err(), "Isub with 1 arg should error");
    }

    #[test]
    fn select_binop_missing_result_errors() {
        let (mut isel, entry) = make_add_isel();
        let result = isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0), Value(1)],
                results: vec![], // no result
            },
            entry,
        );
        assert!(result.is_err(), "Iadd with no result should error");
    }

    #[test]
    fn select_undefined_value_errors() {
        let (mut isel, entry) = make_empty_isel();
        let result = isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(99), Value(100)], // never defined
                results: vec![Value(101)],
            },
            entry,
        );
        assert!(result.is_err(), "Using undefined values should error");
    }

    // =======================================================================
    // Coverage expansion: all IntCC condition code mappings
    // =======================================================================

    #[test]
    fn condition_code_mapping_all_variants() {
        // Exhaustively test all IntCC -> AArch64CC mappings
        assert_eq!(AArch64CC::from_intcc(IntCC::Equal), AArch64CC::EQ);
        assert_eq!(AArch64CC::from_intcc(IntCC::NotEqual), AArch64CC::NE);
        assert_eq!(AArch64CC::from_intcc(IntCC::SignedLessThan), AArch64CC::LT);
        assert_eq!(
            AArch64CC::from_intcc(IntCC::SignedGreaterThanOrEqual),
            AArch64CC::GE
        );
        assert_eq!(
            AArch64CC::from_intcc(IntCC::SignedGreaterThan),
            AArch64CC::GT
        );
        assert_eq!(
            AArch64CC::from_intcc(IntCC::SignedLessThanOrEqual),
            AArch64CC::LE
        );
        assert_eq!(
            AArch64CC::from_intcc(IntCC::UnsignedLessThan),
            AArch64CC::LO
        );
        assert_eq!(
            AArch64CC::from_intcc(IntCC::UnsignedGreaterThanOrEqual),
            AArch64CC::HS
        );
        assert_eq!(
            AArch64CC::from_intcc(IntCC::UnsignedGreaterThan),
            AArch64CC::HI
        );
        assert_eq!(
            AArch64CC::from_intcc(IntCC::UnsignedLessThanOrEqual),
            AArch64CC::LS
        );
    }

    #[test]
    fn condition_code_inversion_round_trip() {
        // Inverting twice should return to original
        let codes = [
            AArch64CC::EQ,
            AArch64CC::NE,
            AArch64CC::HS,
            AArch64CC::LO,
            AArch64CC::MI,
            AArch64CC::PL,
            AArch64CC::VS,
            AArch64CC::VC,
            AArch64CC::HI,
            AArch64CC::LS,
            AArch64CC::GE,
            AArch64CC::LT,
            AArch64CC::GT,
            AArch64CC::LE,
        ];
        for cc in &codes {
            assert_eq!(
                cc.invert().invert(),
                *cc,
                "Double inversion should be identity for {:?}",
                cc
            );
        }
    }

    // =======================================================================
    // Coverage expansion: comparison selection for all IntCC variants
    // =======================================================================

    #[test]
    fn select_icmp_all_signed_conditions() {
        for (cc, expected_aarch64_cc) in &[
            (IntCC::SignedLessThan, AArch64CC::LT),
            (IntCC::SignedGreaterThan, AArch64CC::GT),
            (IntCC::SignedLessThanOrEqual, AArch64CC::LE),
            (IntCC::SignedGreaterThanOrEqual, AArch64CC::GE),
        ] {
            let (mut isel, entry) = make_add_isel();
            isel.select_instruction(
                &Instruction {
                    opcode: Opcode::Icmp { cond: *cc },
                    args: vec![Value(0), Value(1)],
                    results: vec![Value(2)],
                },
                entry,
            )
            .unwrap();
            let mfunc = isel.finalize();
            let mblock = &mfunc.blocks[&entry];
            // Find the CSET instruction and verify condition code
            let cset = mblock
                .insts
                .iter()
                .find(|i| i.opcode == AArch64Opcode::CSet)
                .unwrap();
            assert_eq!(
                cset.operands[1],
                ISelOperand::CondCode(*expected_aarch64_cc),
                "IntCC::{:?} should map to AArch64CC::{:?}",
                cc,
                expected_aarch64_cc
            );
        }
    }

    #[test]
    fn select_icmp_all_unsigned_conditions() {
        for (cc, expected_aarch64_cc) in &[
            (IntCC::UnsignedLessThan, AArch64CC::LO),
            (IntCC::UnsignedGreaterThan, AArch64CC::HI),
            (IntCC::UnsignedLessThanOrEqual, AArch64CC::LS),
            (IntCC::UnsignedGreaterThanOrEqual, AArch64CC::HS),
        ] {
            let (mut isel, entry) = make_add_isel();
            isel.select_instruction(
                &Instruction {
                    opcode: Opcode::Icmp { cond: *cc },
                    args: vec![Value(0), Value(1)],
                    results: vec![Value(2)],
                },
                entry,
            )
            .unwrap();
            let mfunc = isel.finalize();
            let mblock = &mfunc.blocks[&entry];
            let cset = mblock
                .insts
                .iter()
                .find(|i| i.opcode == AArch64Opcode::CSet)
                .unwrap();
            assert_eq!(
                cset.operands[1],
                ISelOperand::CondCode(*expected_aarch64_cc),
                "IntCC::{:?} should map to AArch64CC::{:?}",
                cc,
                expected_aarch64_cc
            );
        }
    }

    // =======================================================================
    // Coverage expansion: iconst edge cases
    // =======================================================================

    #[test]
    fn select_iconst_zero() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I32,
                    imm: 0,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, AArch64Opcode::Movz);
        assert_eq!(inst.operands[1], ISelOperand::Imm(0));
    }

    #[test]
    fn select_iconst_max_movz_range() {
        // 0xFFFF is the max value for a single MOVZ
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I32,
                    imm: 0xFFFF,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // Should be a single MOVZ (no MOVK needed)
        assert_eq!(mblock.insts.len(), 1);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Movz);
        assert_eq!(mblock.insts[0].operands[1], ISelOperand::Imm(0xFFFF));
    }

    #[test]
    fn select_iconst_just_above_movz_range() {
        // 0x10000 requires MOVZ + MOVK
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I32,
                    imm: 0x10000,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // MOVZ low16=0x0000 + MOVK high16=0x0001
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Movz);
        assert_eq!(mblock.insts[0].operands[1], ISelOperand::Imm(0));
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Movk);
        assert_eq!(mblock.insts[1].operands[1], ISelOperand::Imm(1));
    }

    #[test]
    fn select_iconst_negative_boundary() {
        // -0x10000 is the boundary of MOVN range
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I64,
                    imm: -0x10000,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Movn);
        // MOVN encodes ~(-0x10000) = 0xFFFF
        assert_eq!(mblock.insts[0].operands[1], ISelOperand::Imm(0xFFFF));
    }

    // =======================================================================
    // Coverage expansion: reg_class_for_type helper
    // =======================================================================

    #[test]
    fn reg_class_for_all_types() {
        assert_eq!(reg_class_for_type(&Type::B1), RegClass::Gpr32);
        assert_eq!(reg_class_for_type(&Type::I8), RegClass::Gpr32);
        assert_eq!(reg_class_for_type(&Type::I16), RegClass::Gpr32);
        assert_eq!(reg_class_for_type(&Type::I32), RegClass::Gpr32);
        assert_eq!(reg_class_for_type(&Type::I64), RegClass::Gpr64);
        assert_eq!(reg_class_for_type(&Type::I128), RegClass::Gpr64);
        assert_eq!(reg_class_for_type(&Type::F32), RegClass::Fpr32);
        assert_eq!(reg_class_for_type(&Type::F64), RegClass::Fpr64);
        // Aggregates are pointers at machine level
        assert_eq!(
            reg_class_for_type(&Type::Struct(vec![Type::I32])),
            RegClass::Gpr64
        );
        assert_eq!(
            reg_class_for_type(&Type::Array(Box::new(Type::I32), 4)),
            RegClass::Gpr64
        );
    }

    // =======================================================================
    // Coverage expansion: is_32bit helper
    // =======================================================================

    #[test]
    fn is_32bit_classification() {
        assert!(InstructionSelector::is_32bit(&Type::B1));
        assert!(InstructionSelector::is_32bit(&Type::I8));
        assert!(InstructionSelector::is_32bit(&Type::I16));
        assert!(InstructionSelector::is_32bit(&Type::I32));
        assert!(InstructionSelector::is_32bit(&Type::F32));
        assert!(!InstructionSelector::is_32bit(&Type::I64));
        assert!(!InstructionSelector::is_32bit(&Type::I128));
        assert!(!InstructionSelector::is_32bit(&Type::F64));
    }

    // =======================================================================
    // Floating-point instruction selection tests (issue #211)
    // =======================================================================

    // -- FABS tests --

    #[test]
    fn select_fabs_f64() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("fabs64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fabs,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // 1 COPY (formal arg) + 1 FABS
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::FabsRR);
    }

    #[test]
    fn select_fabs_f32() {
        let sig = Signature {
            params: vec![Type::F32],
            returns: vec![Type::F32],
        };
        let mut isel = InstructionSelector::new("fabs32".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fabs,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::FabsRR);
    }

    #[test]
    fn fabs_result_is_defined_with_correct_type() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("fabs_ty".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fabs,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        // The result value should be usable in subsequent instructions
        // (verifies it was properly defined in the value map).
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::FabsRR);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::FaddRR);
    }

    // -- FSQRT tests --

    #[test]
    fn select_fsqrt_f64() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("fsqrt64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fsqrt,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::FsqrtRR);
    }

    #[test]
    fn select_fsqrt_f32() {
        let sig = Signature {
            params: vec![Type::F32],
            returns: vec![Type::F32],
        };
        let mut isel = InstructionSelector::new("fsqrt32".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fsqrt,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::FsqrtRR);
    }

    #[test]
    fn fsqrt_result_chains_to_fadd() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("sqrtadd".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        // sqrt(x) + x
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fsqrt,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fadd,
                args: vec![Value(1), Value(0)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::FsqrtRR);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::FaddRR);
    }

    // -- FP arithmetic with both precision variants --

    #[test]
    fn select_fadd_f32() {
        let (mut isel, entry) = make_f32_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::FaddRR);
    }

    #[test]
    fn select_fsub_f64() {
        let (mut isel, entry) = make_f64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fsub,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::FsubRR);
    }

    #[test]
    fn select_fmul_f32() {
        let (mut isel, entry) = make_f32_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fmul,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::FmulRR);
    }

    #[test]
    fn select_fdiv_f64() {
        let (mut isel, entry) = make_f64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fdiv,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[2].opcode, AArch64Opcode::FdivRR);
    }

    // -- FP comparison with various condition codes --

    #[test]
    fn select_fcmp_f32_equal() {
        let (mut isel, entry) = make_f32_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fcmp {
                    cond: FloatCC::Equal,
                },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::Fcmp);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::CSet);
        assert_eq!(
            mblock.insts[3].operands[1],
            ISelOperand::CondCode(AArch64CC::EQ)
        );
    }

    #[test]
    fn select_fcmp_f64_greater_than() {
        let (mut isel, entry) = make_f64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fcmp {
                    cond: FloatCC::GreaterThan,
                },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::Fcmp);
        assert_eq!(
            mblock.insts[3].operands[1],
            ISelOperand::CondCode(AArch64CC::GT)
        );
    }

    #[test]
    fn select_fcmp_f64_unordered() {
        let (mut isel, entry) = make_f64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fcmp {
                    cond: FloatCC::Unordered,
                },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(
            mblock.insts[3].operands[1],
            ISelOperand::CondCode(AArch64CC::VS)
        );
    }

    // -- FP-to-int conversions --

    #[test]
    fn select_fcvt_to_int_i64_from_f64() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("fcvt_i64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FcvtToInt { dst_ty: Type::I64 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(
            mfunc.blocks[&entry].insts[1].opcode,
            AArch64Opcode::FcvtzsRR
        );
    }

    #[test]
    fn select_fcvt_to_uint_from_f32() {
        let sig = Signature {
            params: vec![Type::F32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("f32_to_u32".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FcvtToUint { dst_ty: Type::I32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(
            mfunc.blocks[&entry].insts[1].opcode,
            AArch64Opcode::FcvtzuRR
        );
    }

    // -- Int-to-FP conversions --

    #[test]
    fn select_scvtf_i64_to_f64() {
        let sig = Signature {
            params: vec![Type::I64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("i64_to_f64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FcvtFromInt { src_ty: Type::I64 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::ScvtfRR);
    }

    #[test]
    fn select_ucvtf_i32_to_f32() {
        let sig = Signature {
            params: vec![Type::I32],
            returns: vec![Type::F32],
        };
        let mut isel = InstructionSelector::new("u32_to_f32".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FcvtFromUint { src_ty: Type::I32 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::UcvtfRR);
    }

    #[test]
    fn select_ucvtf_i64_to_f64() {
        let sig = Signature {
            params: vec![Type::I64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("u64_to_f64".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FcvtFromUint { src_ty: Type::I64 },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[1].opcode, AArch64Opcode::UcvtfRR);
    }

    // -- FP precision conversion --

    #[test]
    fn select_fpext_f32_to_f64_operand_types() {
        let sig = Signature {
            params: vec![Type::F32],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("fpext".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FPExt,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[1];
        assert_eq!(inst.opcode, AArch64Opcode::FcvtSD);
        // dst is Fpr64, src is Fpr32
        if let ISelOperand::VReg(dst) = &inst.operands[0] {
            assert_eq!(dst.class, RegClass::Fpr64);
        } else {
            panic!("expected VReg dst");
        }
    }

    #[test]
    fn select_fptrunc_f64_to_f32_operand_types() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::F32],
        };
        let mut isel = InstructionSelector::new("fptrunc".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::FPTrunc,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[1];
        assert_eq!(inst.opcode, AArch64Opcode::FcvtDS);
        if let ISelOperand::VReg(dst) = &inst.operands[0] {
            assert_eq!(dst.class, RegClass::Fpr32);
        } else {
            panic!("expected VReg dst");
        }
    }

    // -- FP constants --

    #[test]
    fn select_fconst_f32_one() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fconst {
                    ty: Type::F32,
                    imm: 1.0,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, AArch64Opcode::FmovImm);
        assert_eq!(inst.operands[1], ISelOperand::FImm(1.0));
    }

    #[test]
    fn select_fconst_f64_negative() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fconst {
                    ty: Type::F64,
                    imm: -3.0,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(inst.opcode, AArch64Opcode::FmovImm);
        assert_eq!(inst.operands[1], ISelOperand::FImm(-3.0));
    }

    #[test]
    fn select_fconst_f64_zero() {
        let (mut isel, entry) = make_empty_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fconst {
                    ty: Type::F64,
                    imm: 0.0,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        assert_eq!(mfunc.blocks[&entry].insts[0].opcode, AArch64Opcode::FmovImm);
    }

    // -- Combined FP operations (chaining) --

    #[test]
    fn fp_chain_fabs_then_fsqrt() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("abs_sqrt".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        // result = sqrt(|x|)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fabs,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fsqrt,
                args: vec![Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::FabsRR);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::FsqrtRR);
    }

    #[test]
    fn fp_chain_fmul_then_fabs() {
        let (mut isel, entry) = make_f64_isel();

        // result = |x * y|
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fmul,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fabs,
                args: vec![Value(2)],
                results: vec![Value(3)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::FmulRR);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::FabsRR);
    }

    // -- Error handling --

    #[test]
    fn fabs_missing_arg_errors() {
        let (mut isel, entry) = make_empty_isel();
        let result = isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fabs,
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        );
        assert!(result.is_err());
    }

    #[test]
    fn fsqrt_missing_result_errors() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![],
        };
        let mut isel = InstructionSelector::new("err".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        let result = isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fsqrt,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        );
        assert!(result.is_err());
    }

    // -- to_ir_func conversion for FP ops --

    #[test]
    fn fp_isel_to_ir_func_preserves_fabs_opcode() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("ir_fabs".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fabs,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let isel_func = isel.finalize();
        let ir_func = isel_func.to_ir_func();

        // Verify the opcode survives the ISel->IR conversion
        let ir_insts = &ir_func.blocks[0].insts;
        let fabs_inst_id = ir_insts[1]; // after the COPY for formal arg
        let fabs_inst = &ir_func.insts[fabs_inst_id.0 as usize];
        assert_eq!(fabs_inst.opcode, AArch64Opcode::FabsRR);
    }

    #[test]
    fn fp_isel_to_ir_func_preserves_fsqrt_opcode() {
        let sig = Signature {
            params: vec![Type::F64],
            returns: vec![Type::F64],
        };
        let mut isel = InstructionSelector::new("ir_fsqrt".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fsqrt,
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let isel_func = isel.finalize();
        let ir_func = isel_func.to_ir_func();

        let ir_insts = &ir_func.blocks[0].insts;
        let fsqrt_inst_id = ir_insts[1];
        let fsqrt_inst = &ir_func.insts[fsqrt_inst_id.0 as usize];
        assert_eq!(fsqrt_inst.opcode, AArch64Opcode::FsqrtRR);
    }

    // =======================================================================
    // Atomic memory operation tests
    // =======================================================================

    #[test]
    fn select_atomic_load_i32_emits_ldar() {
        let (mut isel, entry) = make_empty_isel();
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::AtomicLoad {
                    ty: Type::I32,
                    ordering: AtomicOrdering::Acquire,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::Ldar,
            "AtomicLoad I32 with Acquire ordering should emit LDAR"
        );
    }

    #[test]
    fn select_atomic_load_i8_emits_ldarb() {
        let (mut isel, entry) = make_empty_isel();
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::AtomicLoad {
                    ty: Type::I8,
                    ordering: AtomicOrdering::SeqCst,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::Ldarb,
            "AtomicLoad I8 should emit LDARB"
        );
    }

    #[test]
    fn select_atomic_load_i16_emits_ldarh() {
        let (mut isel, entry) = make_empty_isel();
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::AtomicLoad {
                    ty: Type::I16,
                    ordering: AtomicOrdering::Acquire,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::Ldarh,
            "AtomicLoad I16 should emit LDARH"
        );
    }

    #[test]
    fn select_atomic_store_i32_emits_stlr() {
        let (mut isel, entry) = make_empty_isel();
        let val_vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), ISelOperand::VReg(val_vreg), Type::I32);
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::AtomicStore {
                    ordering: AtomicOrdering::Release,
                },
                args: vec![Value(0), Value(1)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::Stlr,
            "AtomicStore I32 with Release ordering should emit STLR"
        );
    }

    #[test]
    fn select_atomic_store_i8_emits_stlrb() {
        let (mut isel, entry) = make_empty_isel();
        let val_vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), ISelOperand::VReg(val_vreg), Type::I8);
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::AtomicStore {
                    ordering: AtomicOrdering::SeqCst,
                },
                args: vec![Value(0), Value(1)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::Stlrb,
            "AtomicStore I8 should emit STLRB"
        );
    }

    #[test]
    fn select_atomic_rmw_add_seqcst_emits_ldaddal() {
        let (mut isel, entry) = make_empty_isel();
        let val_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(val_vreg), Type::I64);
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::AtomicRmw {
                    op: AtomicRmwOp::Add,
                    ty: Type::I64,
                    ordering: AtomicOrdering::SeqCst,
                },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::Ldaddal,
            "AtomicRmw Add with SeqCst should emit LDADDAL"
        );
    }

    #[test]
    fn select_atomic_rmw_add_relaxed_emits_ldadd() {
        let (mut isel, entry) = make_empty_isel();
        let val_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(val_vreg), Type::I64);
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::AtomicRmw {
                    op: AtomicRmwOp::Add,
                    ty: Type::I64,
                    ordering: AtomicOrdering::Relaxed,
                },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::Ldadd,
            "AtomicRmw Add with Relaxed should emit LDADD (no ordering suffix)"
        );
    }

    #[test]
    fn select_atomic_rmw_or_seqcst_emits_ldsetal() {
        let (mut isel, entry) = make_empty_isel();
        let val_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(val_vreg), Type::I64);
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::AtomicRmw {
                    op: AtomicRmwOp::Or,
                    ty: Type::I64,
                    ordering: AtomicOrdering::SeqCst,
                },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::Ldsetal,
            "AtomicRmw Or with SeqCst should emit LDSETAL"
        );
    }

    #[test]
    fn select_atomic_rmw_xchg_seqcst_emits_swpal() {
        let (mut isel, entry) = make_empty_isel();
        let val_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(val_vreg), Type::I64);
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::AtomicRmw {
                    op: AtomicRmwOp::Xchg,
                    ty: Type::I64,
                    ordering: AtomicOrdering::SeqCst,
                },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::Swpal,
            "AtomicRmw Xchg with SeqCst should emit SWPAL"
        );
    }

    #[test]
    fn select_atomic_rmw_and_emits_mvn_then_ldclral() {
        let (mut isel, entry) = make_empty_isel();
        let val_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(val_vreg), Type::I64);
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::AtomicRmw {
                    op: AtomicRmwOp::And,
                    ty: Type::I64,
                    ordering: AtomicOrdering::SeqCst,
                },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;
        // AND is lowered as MVN (ORN) + LDCLRAL
        assert_eq!(
            insts[0].opcode,
            AArch64Opcode::OrnRR,
            "AtomicRmw And should first emit MVN (ORN) to invert the operand"
        );
        // ORN must have 3 operands: [dst, XZR, src] (issue #334)
        assert_eq!(
            insts[0].operands.len(),
            3,
            "OrnRR (atomic MVN i64) must have 3 operands [dst, XZR, src]"
        );
        assert_eq!(
            insts[0].operands[1],
            ISelOperand::PReg(XZR),
            "OrnRR (atomic MVN i64) operand[1] must be XZR"
        );
        assert_eq!(
            insts[1].opcode,
            AArch64Opcode::Ldclral,
            "AtomicRmw And with SeqCst should emit LDCLRAL after MVN"
        );
    }

    #[test]
    fn select_atomic_rmw_sub_emits_neg_then_ldaddal() {
        let (mut isel, entry) = make_empty_isel();
        let val_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(val_vreg), Type::I64);
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::AtomicRmw {
                    op: AtomicRmwOp::Sub,
                    ty: Type::I64,
                    ordering: AtomicOrdering::SeqCst,
                },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;
        // SUB is lowered as NEG + LDADDAL
        assert_eq!(
            insts[0].opcode,
            AArch64Opcode::Neg,
            "AtomicRmw Sub should first emit NEG to negate the operand"
        );
        assert_eq!(
            insts[1].opcode,
            AArch64Opcode::Ldaddal,
            "AtomicRmw Sub with SeqCst should emit LDADDAL after NEG"
        );
    }

    #[test]
    fn select_cmpxchg_seqcst_emits_mov_then_casal() {
        let (mut isel, entry) = make_empty_isel();
        let exp_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(exp_vreg), Type::I64);
        let des_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), ISelOperand::VReg(des_vreg), Type::I64);
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(2), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::CmpXchg {
                    ty: Type::I64,
                    ordering: AtomicOrdering::SeqCst,
                },
                args: vec![Value(0), Value(1), Value(2)],
                results: vec![Value(3)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;
        // CmpXchg emits MOV (copy expected) then CASAL
        assert_eq!(
            insts[0].opcode,
            AArch64Opcode::MovR,
            "CmpXchg should first copy expected value with MOV"
        );
        assert_eq!(
            insts[1].opcode,
            AArch64Opcode::Casal,
            "CmpXchg with SeqCst should emit CASAL"
        );
    }

    #[test]
    fn select_cmpxchg_acquire_emits_casa() {
        let (mut isel, entry) = make_empty_isel();
        let exp_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(exp_vreg), Type::I64);
        let des_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), ISelOperand::VReg(des_vreg), Type::I64);
        let addr_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(2), ISelOperand::VReg(addr_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::CmpXchg {
                    ty: Type::I64,
                    ordering: AtomicOrdering::Acquire,
                },
                args: vec![Value(0), Value(1), Value(2)],
                results: vec![Value(3)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;
        assert_eq!(
            insts[1].opcode,
            AArch64Opcode::Casa,
            "CmpXchg with Acquire should emit CASA"
        );
    }

    #[test]
    fn select_fence_seqcst_emits_dmb_ish() {
        let (mut isel, entry) = make_empty_isel();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fence {
                    ordering: AtomicOrdering::SeqCst,
                },
                args: vec![],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::Dmb,
            "Fence with SeqCst should emit DMB"
        );
        // The immediate operand should be ISH (0xB)
        assert_eq!(
            inst.operands[0],
            ISelOperand::Imm(0x0B),
            "SeqCst fence should use DMB ISH (0xB)"
        );
    }

    #[test]
    fn select_fence_acquire_emits_dmb_ishld() {
        let (mut isel, entry) = make_empty_isel();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fence {
                    ordering: AtomicOrdering::Acquire,
                },
                args: vec![],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let inst = &mfunc.blocks[&entry].insts[0];
        assert_eq!(
            inst.opcode,
            AArch64Opcode::Dmb,
            "Fence with Acquire should emit DMB"
        );
        assert_eq!(
            inst.operands[0],
            ISelOperand::Imm(0x09),
            "Acquire fence should use DMB ISHLD (0x9)"
        );
    }

    #[test]
    fn select_fence_relaxed_emits_nothing() {
        let (mut isel, entry) = make_empty_isel();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Fence {
                    ordering: AtomicOrdering::Relaxed,
                },
                args: vec![],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;
        assert!(
            insts.is_empty(),
            "Fence with Relaxed ordering should emit no instructions (NOP)"
        );
    }

    // =======================================================================
    // Invoke (call that may throw)
    // =======================================================================

    #[test]
    fn select_invoke_emits_bl_and_branch() {
        let (mut isel, entry) = make_empty_isel();

        // Create normal and unwind destination blocks.
        let normal_dest = Block(1);
        let unwind_dest = Block(2);
        isel.func.ensure_block(normal_dest);
        isel.func.ensure_block(unwind_dest);

        // Define an arg value.
        let vreg = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(0), ISelOperand::VReg(vreg), Type::I32);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Invoke {
                    name: "may_throw".to_string(),
                    normal_dest,
                    unwind_dest,
                },
                args: vec![Value(0)],
                results: vec![Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;

        // Should contain at least a BL and a B to normal_dest.
        let has_bl = insts.iter().any(|i| i.opcode == AArch64Opcode::Bl);
        let has_b = insts.iter().any(|i| i.opcode == AArch64Opcode::B);
        assert!(has_bl, "Invoke should emit BL instruction");
        assert!(has_b, "Invoke should emit B to normal_dest");

        // Check the BL targets "may_throw".
        let bl_inst = insts
            .iter()
            .find(|i| i.opcode == AArch64Opcode::Bl)
            .unwrap();
        assert!(
            bl_inst
                .operands
                .iter()
                .any(|op| matches!(op, ISelOperand::Symbol(s) if s == "may_throw")),
            "BL should target 'may_throw'"
        );

        // Check successors include both normal and unwind blocks.
        let block = &mfunc.blocks[&entry];
        assert!(
            block.successors.contains(&normal_dest),
            "Should have normal_dest as successor"
        );
        assert!(
            block.successors.contains(&unwind_dest),
            "Should have unwind_dest as successor"
        );
    }

    #[test]
    fn select_invoke_no_args_no_results() {
        let (mut isel, entry) = make_empty_isel();

        let normal_dest = Block(1);
        let unwind_dest = Block(2);
        isel.func.ensure_block(normal_dest);
        isel.func.ensure_block(unwind_dest);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Invoke {
                    name: "void_throwing".to_string(),
                    normal_dest,
                    unwind_dest,
                },
                args: vec![],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;

        let has_bl = insts.iter().any(|i| i.opcode == AArch64Opcode::Bl);
        assert!(has_bl, "Invoke should emit BL even with no args/results");
    }

    // =======================================================================
    // LandingPad (exception handler entry)
    // =======================================================================

    #[test]
    fn select_landing_pad_defines_exception_ptr_and_selector() {
        let (mut isel, entry) = make_empty_isel();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::LandingPad {
                    is_cleanup: false,
                    catch_type_indices: vec![1],
                },
                args: vec![],
                results: vec![Value(0), Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;

        // Should have 2 MOV instructions: X0 -> exc_ptr, X1 -> selector.
        assert_eq!(insts.len(), 2, "LandingPad should emit 2 MOV instructions");
        assert_eq!(insts[0].opcode, AArch64Opcode::MovR);
        assert_eq!(insts[1].opcode, AArch64Opcode::MovR);

        // First MOV should source from X0.
        assert!(
            insts[0]
                .operands
                .iter()
                .any(|op| matches!(op, ISelOperand::PReg(p) if *p == gpr::X0)),
            "Exception pointer should come from X0"
        );

        // Second MOV should source from X1.
        assert!(
            insts[1]
                .operands
                .iter()
                .any(|op| matches!(op, ISelOperand::PReg(p) if *p == gpr::X1)),
            "Type selector should come from X1"
        );
    }

    #[test]
    fn select_landing_pad_cleanup_only() {
        let (mut isel, entry) = make_empty_isel();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::LandingPad {
                    is_cleanup: true,
                    catch_type_indices: vec![],
                },
                args: vec![],
                results: vec![Value(0), Value(1)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;

        // Cleanup landing pads still define the exception ptr and selector.
        assert_eq!(insts.len(), 2);
    }

    #[test]
    fn select_landing_pad_single_result() {
        let (mut isel, entry) = make_empty_isel();

        // Only one result (exception pointer, no selector).
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::LandingPad {
                    is_cleanup: true,
                    catch_type_indices: vec![],
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;

        // Only one MOV (exception pointer from X0).
        assert_eq!(insts.len(), 1);
        assert_eq!(insts[0].opcode, AArch64Opcode::MovR);
    }

    // =======================================================================
    // Resume (continue unwinding)
    // =======================================================================

    #[test]
    fn select_resume_calls_unwind_resume() {
        let (mut isel, entry) = make_empty_isel();

        // Define exception pointer value.
        let exc_vreg = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(exc_vreg), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Resume,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let insts = &mfunc.blocks[&entry].insts;

        // Should emit: MOV X0, exc_ptr; BL _Unwind_Resume
        assert!(
            insts.len() >= 2,
            "Resume should emit at least 2 instructions"
        );

        // Last instruction should be BL to _Unwind_Resume.
        let bl_inst = insts
            .iter()
            .find(|i| i.opcode == AArch64Opcode::Bl)
            .unwrap();
        assert!(
            bl_inst
                .operands
                .iter()
                .any(|op| matches!(op, ISelOperand::Symbol(s) if s == "_Unwind_Resume")),
            "Resume should call _Unwind_Resume"
        );

        // Should have a MOV to X0.
        let mov_inst = insts.iter().find(|i| {
            i.opcode == AArch64Opcode::MovR
                && i.operands
                    .iter()
                    .any(|op| matches!(op, ISelOperand::PReg(p) if *p == gpr::X0))
        });
        assert!(mov_inst.is_some(), "Resume should move exception ptr to X0");
    }

    #[test]
    fn select_resume_requires_one_arg() {
        let (mut isel, entry) = make_empty_isel();

        let result = isel.select_instruction(
            &Instruction {
                opcode: Opcode::Resume,
                args: vec![], // missing the exception pointer
                results: vec![],
            },
            entry,
        );

        assert!(result.is_err(), "Resume with no args should fail");
    }

    // =======================================================================
    // Memory intrinsics (memcpy, memmove, memset) — issue #327
    // =======================================================================

    #[test]
    fn select_memcpy_emits_bl_memcpy() {
        let (mut isel, entry) = make_empty_isel();

        // Define 3 I64 values: dest, src, len
        let v0 = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(v0), Type::I64);
        let v1 = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), ISelOperand::VReg(v1), Type::I64);
        let v2 = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(2), ISelOperand::VReg(v2), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Memcpy,
                args: vec![Value(0), Value(1), Value(2)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Should contain BL memcpy
        let bl_inst = mblock.insts.iter().find(|i| i.opcode == AArch64Opcode::Bl);
        assert!(bl_inst.is_some(), "Expected BL instruction for memcpy");
        assert_eq!(
            bl_inst.unwrap().operands[0],
            ISelOperand::Symbol("memcpy".to_string()),
            "BL should target 'memcpy'"
        );

        // Should have Copies to set up arguments in X0, X1, X2
        let copy_count = mblock
            .insts
            .iter()
            .filter(|i| i.opcode == AArch64Opcode::Copy)
            .count();
        assert!(
            copy_count >= 3,
            "Should have at least 3 Copies for memcpy args (dest, src, len)"
        );
    }

    #[test]
    fn select_memmove_emits_bl_memmove() {
        let (mut isel, entry) = make_empty_isel();

        let v0 = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(v0), Type::I64);
        let v1 = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), ISelOperand::VReg(v1), Type::I64);
        let v2 = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(2), ISelOperand::VReg(v2), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Memmove,
                args: vec![Value(0), Value(1), Value(2)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        let bl_inst = mblock.insts.iter().find(|i| i.opcode == AArch64Opcode::Bl);
        assert!(bl_inst.is_some(), "Expected BL instruction for memmove");
        assert_eq!(
            bl_inst.unwrap().operands[0],
            ISelOperand::Symbol("memmove".to_string()),
            "BL should target 'memmove'"
        );
    }

    #[test]
    fn select_memset_emits_bl_memset() {
        let (mut isel, entry) = make_empty_isel();

        // memset(dest, val, len): dest=I64, val=I32, len=I64
        let v0 = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(v0), Type::I64);
        let v1 = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(1), ISelOperand::VReg(v1), Type::I32);
        let v2 = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(2), ISelOperand::VReg(v2), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Memset,
                args: vec![Value(0), Value(1), Value(2)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        let bl_inst = mblock.insts.iter().find(|i| i.opcode == AArch64Opcode::Bl);
        assert!(bl_inst.is_some(), "Expected BL instruction for memset");
        assert_eq!(
            bl_inst.unwrap().operands[0],
            ISelOperand::Symbol("memset".to_string()),
            "BL should target 'memset'"
        );
    }

    #[test]
    fn select_memcpy_requires_three_args() {
        let (mut isel, entry) = make_empty_isel();

        let v0 = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(v0), Type::I64);

        let result = isel.select_instruction(
            &Instruction {
                opcode: Opcode::Memcpy,
                args: vec![Value(0)], // only 1 arg, needs 3
                results: vec![],
            },
            entry,
        );

        assert!(result.is_err(), "Memcpy with < 3 args should fail");
    }

    #[test]
    fn select_memset_no_results() {
        // memset produces no results (void return)
        let (mut isel, entry) = make_empty_isel();

        let v0 = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(0), ISelOperand::VReg(v0), Type::I64);
        let v1 = isel.new_vreg(RegClass::Gpr32);
        isel.define_value(Value(1), ISelOperand::VReg(v1), Type::I32);
        let v2 = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(2), ISelOperand::VReg(v2), Type::I64);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Memset,
                args: vec![Value(0), Value(1), Value(2)],
                results: vec![], // void
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // After BL, there should be no result copy (no return value)
        let bl_idx = mblock
            .insts
            .iter()
            .position(|i| i.opcode == AArch64Opcode::Bl)
            .unwrap();
        // No Copy/MovR after the BL (no return value to copy)
        let post_bl_copy = mblock.insts[bl_idx + 1..]
            .iter()
            .any(|i| i.opcode == AArch64Opcode::MovR || i.opcode == AArch64Opcode::Copy);
        assert!(!post_bl_copy, "memset is void — no result copy after BL");
    }

    // =======================================================================
    // i128 multi-register arithmetic tests
    // =======================================================================

    /// Helper: create isel with two manually-defined i128 values (lo:hi pairs).
    fn make_i128_isel() -> (InstructionSelector, Block) {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut isel = InstructionSelector::new("i128op".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        let v0_lo = isel.new_vreg(RegClass::Gpr64);
        let v0_hi = isel.new_vreg(RegClass::Gpr64);
        isel.define_i128_value(Value(0), v0_lo, v0_hi);

        let v1_lo = isel.new_vreg(RegClass::Gpr64);
        let v1_hi = isel.new_vreg(RegClass::Gpr64);
        isel.define_i128_value(Value(1), v1_lo, v1_hi);

        (isel, entry)
    }

    #[test]
    fn select_i128_add_emits_adds_adc() {
        let (mut isel, entry) = make_i128_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::AddsRR);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Adc);
    }

    #[test]
    fn select_i128_sub_emits_subs_sbc() {
        let (mut isel, entry) = make_i128_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Isub,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::SubsRR);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Sbc);
    }

    #[test]
    fn select_i128_mul_emits_mul_umulh_madd() {
        let (mut isel, entry) = make_i128_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Imul,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        assert_eq!(mblock.insts.len(), 4);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::MulRR);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Umulh);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::Madd);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Madd);
    }

    #[test]
    fn select_i128_cmp_eq() {
        let (mut isel, entry) = make_i128_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Icmp { cond: IntCC::Equal },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // CMP hi + CSET EQ + CMP lo + CSET EQ + AND
        assert_eq!(mblock.insts.len(), 5);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::CmpRR);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::CSet);
        assert_eq!(
            mblock.insts[1].operands[1],
            ISelOperand::CondCode(AArch64CC::EQ)
        );
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::CmpRR);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::CSet);
        assert_eq!(
            mblock.insts[3].operands[1],
            ISelOperand::CondCode(AArch64CC::EQ)
        );
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::AndRR);
    }

    #[test]
    fn select_i128_cmp_ne() {
        let (mut isel, entry) = make_i128_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Icmp {
                    cond: IntCC::NotEqual,
                },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // CMP hi + CSET NE + CMP lo + CSET NE + ORR
        assert_eq!(mblock.insts.len(), 5);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::CmpRR);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::CSet);
        assert_eq!(
            mblock.insts[1].operands[1],
            ISelOperand::CondCode(AArch64CC::NE)
        );
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::CmpRR);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::CSet);
        assert_eq!(
            mblock.insts[3].operands[1],
            ISelOperand::CondCode(AArch64CC::NE)
        );
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::OrrRR);
    }

    #[test]
    fn select_i128_cmp_slt() {
        let (mut isel, entry) = make_i128_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Icmp {
                    cond: IntCC::SignedLessThan,
                },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // CMP hi + CSET LT + CSET EQ + CMP lo + CSET LO + AND + ORR
        assert_eq!(mblock.insts.len(), 7);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::CmpRR); // CMP hi
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::CSet); // CSET LT (signed)
        assert_eq!(
            mblock.insts[1].operands[1],
            ISelOperand::CondCode(AArch64CC::LT)
        );
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::CSet); // CSET EQ
        assert_eq!(
            mblock.insts[2].operands[1],
            ISelOperand::CondCode(AArch64CC::EQ)
        );
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::CmpRR); // CMP lo
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::CSet); // CSET LO (unsigned)
        assert_eq!(
            mblock.insts[4].operands[1],
            ISelOperand::CondCode(AArch64CC::LO)
        );
        assert_eq!(mblock.insts[5].opcode, AArch64Opcode::AndRR); // AND (hi_eq & lo_cond)
        assert_eq!(mblock.insts[6].opcode, AArch64Opcode::OrrRR); // ORR (hi_cond | tmp)
    }

    #[test]
    fn select_i128_cmp_uge() {
        let (mut isel, entry) = make_i128_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Icmp {
                    cond: IntCC::UnsignedGreaterThanOrEqual,
                },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // CMP hi + CSET HI + CSET EQ + CMP lo + CSET HS + AND + ORR
        assert_eq!(mblock.insts.len(), 7);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::CmpRR); // CMP hi
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::CSet); // CSET HI (unsigned >)
        assert_eq!(
            mblock.insts[1].operands[1],
            ISelOperand::CondCode(AArch64CC::HI)
        );
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::CSet); // CSET EQ
        assert_eq!(
            mblock.insts[2].operands[1],
            ISelOperand::CondCode(AArch64CC::EQ)
        );
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::CmpRR); // CMP lo
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::CSet); // CSET HS (unsigned >=)
        assert_eq!(
            mblock.insts[4].operands[1],
            ISelOperand::CondCode(AArch64CC::HS)
        );
        assert_eq!(mblock.insts[5].opcode, AArch64Opcode::AndRR); // AND (hi_eq & lo_cond)
        assert_eq!(mblock.insts[6].opcode, AArch64Opcode::OrrRR); // ORR (hi_cond | tmp)
    }

    // =======================================================================
    // i128 constant materialization tests
    // =======================================================================

    #[test]
    fn select_i128_iconst_zero() {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut isel = InstructionSelector::new("const128".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I128,
                    imm: 0,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Two MOVZ instructions (lo=0, hi=0)
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Movz);
        assert_eq!(mblock.insts[0].operands[1], ISelOperand::Imm(0));
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Movz);
        assert_eq!(mblock.insts[1].operands[1], ISelOperand::Imm(0));
    }

    #[test]
    fn select_i128_iconst_small_positive() {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut isel = InstructionSelector::new("const128".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I128,
                    imm: 42,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // MOVZ lo(42) + MOVZ hi(0) = 2 instructions
        assert_eq!(mblock.insts.len(), 2);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Movz);
        assert_eq!(mblock.insts[0].operands[1], ISelOperand::Imm(42));
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Movz);
        assert_eq!(mblock.insts[1].operands[1], ISelOperand::Imm(0)); // hi = 0
    }

    #[test]
    fn select_i128_iconst_negative() {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut isel = InstructionSelector::new("const128".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I128,
                    imm: -1,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // lo = 0xFFFF_FFFF_FFFF_FFFF: MOVZ + 3 MOVK = 4
        // hi = 0xFFFF_FFFF_FFFF_FFFF: MOVZ + 3 MOVK = 4
        // Total: 8 instructions
        assert_eq!(mblock.insts.len(), 8);
        // lo half: MOVZ 0xFFFF, MOVK 0xFFFF<<16, MOVK 0xFFFF<<32, MOVK 0xFFFF<<48
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Movz);
        assert_eq!(mblock.insts[0].operands[1], ISelOperand::Imm(0xFFFF));
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Movk);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::Movk);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Movk);
        // hi half: same pattern
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::Movz);
        assert_eq!(mblock.insts[4].operands[1], ISelOperand::Imm(0xFFFF));
        assert_eq!(mblock.insts[5].opcode, AArch64Opcode::Movk);
        assert_eq!(mblock.insts[6].opcode, AArch64Opcode::Movk);
        assert_eq!(mblock.insts[7].opcode, AArch64Opcode::Movk);
    }

    #[test]
    fn select_i128_iconst_large() {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut isel = InstructionSelector::new("const128".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        // 0x1_0000 (65536) — needs MOVZ + 1 MOVK for lo, MOVZ for hi
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I128,
                    imm: 0x1_0000,
                },
                args: vec![],
                results: vec![Value(0)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // lo = 0x10000: MOVZ 0 + MOVK 1<<16 = 2 instructions
        // hi = 0: MOVZ 0 = 1 instruction
        // Total: 3
        assert_eq!(mblock.insts.len(), 3);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Movz); // lo: MOVZ #0
        assert_eq!(mblock.insts[0].operands[1], ISelOperand::Imm(0));
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Movk); // lo: MOVK #1, LSL#16
        assert_eq!(mblock.insts[1].operands[1], ISelOperand::Imm(1));
        assert_eq!(mblock.insts[1].operands[2], ISelOperand::Imm(16));
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::Movz); // hi: MOVZ #0
    }

    // =======================================================================
    // i128 shift tests
    // =======================================================================

    /// Helper: create isel with one i128 value and one scalar shift amount.
    fn make_i128_shift_isel() -> (InstructionSelector, Block) {
        let sig = Signature {
            params: vec![],
            returns: vec![],
        };
        let mut isel = InstructionSelector::new("i128shift".to_string(), sig);
        let entry = Block(0);
        isel.func.ensure_block(entry);

        // i128 value as (lo, hi) pair
        let v0_lo = isel.new_vreg(RegClass::Gpr64);
        let v0_hi = isel.new_vreg(RegClass::Gpr64);
        isel.define_i128_value(Value(0), v0_lo, v0_hi);

        // scalar shift amount
        let v1 = isel.new_vreg(RegClass::Gpr64);
        isel.define_value(Value(1), ISelOperand::VReg(v1), Type::I64);

        (isel, entry)
    }

    #[test]
    fn select_i128_shl_emits_correct_sequence() {
        let (mut isel, entry) = make_i128_shift_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Ishl,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Expected: MOVZ(64) + SubRR + LsrRR + LslRR + OrrRR + LslRR + SubRR + LslRR + MOVZ(0) + CmpRI + Csel + Csel
        assert_eq!(mblock.insts.len(), 12);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Movz); // c64
        assert_eq!(mblock.insts[0].operands[1], ISelOperand::Imm(64));
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::SubRR); // neg_shift
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::LsrRR); // lo_spill
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::LslRR); // hi_shifted
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::OrrRR); // hi_normal
        assert_eq!(mblock.insts[5].opcode, AArch64Opcode::LslRR); // lo_normal
        assert_eq!(mblock.insts[6].opcode, AArch64Opcode::SubRR); // big_shift
        assert_eq!(mblock.insts[7].opcode, AArch64Opcode::LslRR); // hi_big
        assert_eq!(mblock.insts[8].opcode, AArch64Opcode::Movz); // zero
        assert_eq!(mblock.insts[8].operands[1], ISelOperand::Imm(0));
        assert_eq!(mblock.insts[9].opcode, AArch64Opcode::CmpRI); // cmp shift, 64
        assert_eq!(mblock.insts[10].opcode, AArch64Opcode::Csel); // dst_lo
        assert_eq!(
            mblock.insts[10].operands[3],
            ISelOperand::CondCode(AArch64CC::HS)
        );
        assert_eq!(mblock.insts[11].opcode, AArch64Opcode::Csel); // dst_hi
        assert_eq!(
            mblock.insts[11].operands[3],
            ISelOperand::CondCode(AArch64CC::HS)
        );
    }

    #[test]
    fn select_i128_lshr_emits_correct_sequence() {
        let (mut isel, entry) = make_i128_shift_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Ushr,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Expected: MOVZ(64) + SubRR + LslRR + LsrRR + OrrRR + LsrRR + SubRR + LsrRR + MOVZ(0) + CmpRI + Csel + Csel
        assert_eq!(mblock.insts.len(), 12);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Movz); // c64
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::SubRR); // neg_shift
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::LslRR); // hi_spill
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::LsrRR); // lo_shifted
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::OrrRR); // lo_normal
        assert_eq!(mblock.insts[5].opcode, AArch64Opcode::LsrRR); // hi_normal
        assert_eq!(mblock.insts[6].opcode, AArch64Opcode::SubRR); // big_shift
        assert_eq!(mblock.insts[7].opcode, AArch64Opcode::LsrRR); // lo_big
        assert_eq!(mblock.insts[8].opcode, AArch64Opcode::Movz); // zero
        assert_eq!(mblock.insts[9].opcode, AArch64Opcode::CmpRI); // cmp shift, 64
        assert_eq!(mblock.insts[10].opcode, AArch64Opcode::Csel); // dst_hi
        assert_eq!(mblock.insts[11].opcode, AArch64Opcode::Csel); // dst_lo
    }

    #[test]
    fn select_i128_ashr_emits_correct_sequence() {
        let (mut isel, entry) = make_i128_shift_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Sshr,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Expected: MOVZ(64) + MOVZ(63) + SubRR + LslRR + LsrRR + OrrRR + AsrRR + SubRR + AsrRR + AsrRR + CmpRI + Csel + Csel
        assert_eq!(mblock.insts.len(), 13);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Movz); // c64
        assert_eq!(mblock.insts[0].operands[1], ISelOperand::Imm(64));
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Movz); // c63
        assert_eq!(mblock.insts[1].operands[1], ISelOperand::Imm(63));
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::SubRR); // neg_shift
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::LslRR); // hi_spill
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::LsrRR); // lo_shifted
        assert_eq!(mblock.insts[5].opcode, AArch64Opcode::OrrRR); // lo_normal
        assert_eq!(mblock.insts[6].opcode, AArch64Opcode::AsrRR); // hi_normal
        assert_eq!(mblock.insts[7].opcode, AArch64Opcode::SubRR); // big_shift
        assert_eq!(mblock.insts[8].opcode, AArch64Opcode::AsrRR); // lo_big
        assert_eq!(mblock.insts[9].opcode, AArch64Opcode::AsrRR); // hi_sign (ASR by 63)
        assert_eq!(mblock.insts[10].opcode, AArch64Opcode::CmpRI); // cmp shift, 64
        assert_eq!(mblock.insts[11].opcode, AArch64Opcode::Csel); // dst_hi
        assert_eq!(
            mblock.insts[11].operands[3],
            ISelOperand::CondCode(AArch64CC::HS)
        );
        assert_eq!(mblock.insts[12].opcode, AArch64Opcode::Csel); // dst_lo
    }

    // =======================================================================
    // i128 division tests
    // =======================================================================

    #[test]
    fn select_i128_udiv_emits_libcall() {
        let (mut isel, entry) = make_i128_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Udiv,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Copy X0, a_lo + Copy X1, a_hi + Copy X2, b_lo + Copy X3, b_hi + BL + Copy dst_lo, X0 + Copy dst_hi, X1
        assert_eq!(mblock.insts.len(), 7);
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Copy); // X0 <- a_lo
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Copy); // X1 <- a_hi
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::Copy); // X2 <- b_lo
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Copy); // X3 <- b_hi
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::Bl); // BL __udivti3
        assert_eq!(
            mblock.insts[4].operands[0],
            ISelOperand::Symbol("__udivti3".to_string())
        );
        assert_eq!(mblock.insts[5].opcode, AArch64Opcode::Copy); // dst_lo <- X0
        assert_eq!(mblock.insts[6].opcode, AArch64Opcode::Copy); // dst_hi <- X1
    }

    #[test]
    fn select_i128_sdiv_emits_libcall() {
        let (mut isel, entry) = make_i128_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Sdiv,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        assert_eq!(mblock.insts.len(), 7);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::Bl);
        assert_eq!(
            mblock.insts[4].operands[0],
            ISelOperand::Symbol("__divti3".to_string())
        );
    }

    // =======================================================================
    // i128 ABI register pair tests
    // =======================================================================

    /// Helper: create isel for fn(i128, i128) -> i128 with formal args lowered.
    fn make_i128_param_isel() -> (InstructionSelector, Block) {
        let sig = Signature {
            params: vec![Type::I128, Type::I128],
            returns: vec![Type::I128],
        };
        let mut isel = InstructionSelector::new("i128_abi".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();
        (isel, entry)
    }

    #[test]
    fn lower_formal_arguments_i128_params() {
        // fn(i128, i128): should produce 4 Copy instructions
        // X0 -> v_lo_0, X1 -> v_hi_0, X2 -> v_lo_1, X3 -> v_hi_1
        let (isel, entry) = make_i128_param_isel();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        assert_eq!(mblock.insts.len(), 4);
        // First i128 param: X0 (low), X1 (high)
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[0].operands[1], ISelOperand::PReg(gpr::X0));
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[1].operands[1], ISelOperand::PReg(gpr::X1));
        // Second i128 param: X2 (low), X3 (high)
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[2].operands[1], ISelOperand::PReg(gpr::X2));
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[3].operands[1], ISelOperand::PReg(gpr::X3));
    }

    #[test]
    fn lower_formal_arguments_i128_defines_pair() {
        // Verify that i128 formal args are properly tracked as lo:hi pairs
        let (isel, _entry) = make_i128_param_isel();

        // Value(0) should be defined as i128 pair
        let (lo0, hi0) = isel.use_i128_value(&Value(0)).unwrap();
        assert!(matches!(lo0, ISelOperand::VReg(_)));
        assert!(matches!(hi0, ISelOperand::VReg(_)));
        // lo and hi should be different vregs
        assert_ne!(lo0, hi0);

        // Value(1) should be a different i128 pair
        let (lo1, hi1) = isel.use_i128_value(&Value(1)).unwrap();
        assert!(matches!(lo1, ISelOperand::VReg(_)));
        assert!(matches!(hi1, ISelOperand::VReg(_)));
        assert_ne!(lo1, hi1);
        // Different from first pair
        assert_ne!(lo0, lo1);
        assert_ne!(hi0, hi1);
    }

    #[test]
    fn select_return_i128_pair() {
        // fn(i128) -> i128: return the i128 argument
        let sig = Signature {
            params: vec![Type::I128],
            returns: vec![Type::I128],
        };
        let mut isel = InstructionSelector::new("ret128".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        // Return Value(0)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(0)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // 2 formal arg copies + 2 return copies + 1 RET = 5 instructions
        assert_eq!(mblock.insts.len(), 5);
        // Formal arg copies: X0->v0, X1->v1
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Copy);
        // Return lo -> X0
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[2].operands[0], ISelOperand::PReg(gpr::X0));
        // Return hi -> X1
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[3].operands[0], ISelOperand::PReg(gpr::X1));
        // RET
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn select_call_i128_args_and_return() {
        // Call a function with two i128 args and one i128 return
        let (mut isel, entry) = make_i128_isel(); // gives Value(0), Value(1) as i128

        // Call "add128" with two i128 args, producing one i128 result
        isel.select_call(
            "add128",
            &[Value(0), Value(1)],
            &[Value(2)],
            &[Type::I128],
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Arg copies: lo0->X0, hi0->X1, lo1->X2, hi1->X3
        // BL add128
        // Result copies: X0->dst_lo, X1->dst_hi
        // Total: 4 arg + 1 BL + 2 result = 7
        assert_eq!(mblock.insts.len(), 7);

        // Arg0 lo -> X0
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[0].operands[0], ISelOperand::PReg(gpr::X0));
        // Arg0 hi -> X1
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[1].operands[0], ISelOperand::PReg(gpr::X1));
        // Arg1 lo -> X2
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[2].operands[0], ISelOperand::PReg(gpr::X2));
        // Arg1 hi -> X3
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[3].operands[0], ISelOperand::PReg(gpr::X3));
        // BL
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::Bl);
        assert_eq!(
            mblock.insts[4].operands[0],
            ISelOperand::Symbol("add128".to_string())
        );
        // Result lo <- X0
        assert_eq!(mblock.insts[5].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[5].operands[1], ISelOperand::PReg(gpr::X0));
        // Result hi <- X1
        assert_eq!(mblock.insts[6].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[6].operands[1], ISelOperand::PReg(gpr::X1));
    }

    #[test]
    fn i128_add_through_abi_e2e() {
        // End-to-end: fn(i128, i128) -> i128 that adds and returns
        let (mut isel, entry) = make_i128_param_isel();

        // Add: Value(2) = Value(0) + Value(1)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        // Return Value(2)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(2)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // 4 formal arg copies
        // 2 arithmetic (ADDS + ADC)
        // 2 return copies (lo -> X0, hi -> X1)
        // 1 RET
        // Total: 9
        assert_eq!(mblock.insts.len(), 9);

        // Formal args: X0, X1, X2, X3
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[3].opcode, AArch64Opcode::Copy);

        // ADDS + ADC (i128 add)
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::AddsRR);
        assert_eq!(mblock.insts[5].opcode, AArch64Opcode::Adc);

        // Return: lo -> X0, hi -> X1
        assert_eq!(mblock.insts[6].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[6].operands[0], ISelOperand::PReg(gpr::X0));
        assert_eq!(mblock.insts[7].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[7].operands[0], ISelOperand::PReg(gpr::X1));

        // RET
        assert_eq!(mblock.insts[8].opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn i128_sub_through_abi_e2e() {
        // End-to-end: fn(i128, i128) -> i128 that subtracts and returns
        let (mut isel, entry) = make_i128_param_isel();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Isub,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(2)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // 4 args + SUBS + SBC + 2 return + RET = 9
        assert_eq!(mblock.insts.len(), 9);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::SubsRR);
        assert_eq!(mblock.insts[5].opcode, AArch64Opcode::Sbc);
        assert_eq!(mblock.insts[8].opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn i128_mul_through_abi_e2e() {
        // End-to-end: fn(i128, i128) -> i128 that multiplies and returns
        let (mut isel, entry) = make_i128_param_isel();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Imul,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(2)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // 4 args + MUL + UMULH + MADD + MADD + 2 return + RET = 11
        assert_eq!(mblock.insts.len(), 11);
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::MulRR);
        assert_eq!(mblock.insts[5].opcode, AArch64Opcode::Umulh);
        assert_eq!(mblock.insts[6].opcode, AArch64Opcode::Madd);
        assert_eq!(mblock.insts[7].opcode, AArch64Opcode::Madd);
        assert_eq!(mblock.insts[10].opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn i128_cmp_through_abi_e2e() {
        // fn(i128, i128) -> b1 : compare equal and return
        let sig = Signature {
            params: vec![Type::I128, Type::I128],
            returns: vec![Type::B1],
        };
        let mut isel = InstructionSelector::new("cmp128".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        // Value(2) = icmp eq Value(0), Value(1)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Icmp { cond: IntCC::Equal },
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        // Return Value(2)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(2)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // 4 formal arg copies + 5 cmp (CMP+CSET+CMP+CSET+AND) + 1 ret copy + 1 RET = 11
        assert_eq!(mblock.insts.len(), 11);
        // Formal args
        for i in 0..4 {
            assert_eq!(mblock.insts[i].opcode, AArch64Opcode::Copy);
        }
        // CMP sequence for i128 equal
        assert_eq!(mblock.insts[4].opcode, AArch64Opcode::CmpRR);
        assert_eq!(mblock.insts[5].opcode, AArch64Opcode::CSet);
        assert_eq!(mblock.insts[6].opcode, AArch64Opcode::CmpRR);
        assert_eq!(mblock.insts[7].opcode, AArch64Opcode::CSet);
        assert_eq!(mblock.insts[8].opcode, AArch64Opcode::AndRR);
        // Return + RET
        assert_eq!(mblock.insts[9].opcode, AArch64Opcode::Copy); // result -> X0
        assert_eq!(mblock.insts[9].operands[0], ISelOperand::PReg(gpr::X0));
        assert_eq!(mblock.insts[10].opcode, AArch64Opcode::Ret);
    }

    #[test]
    fn i128_shift_through_abi_e2e() {
        // fn(i128, i64) -> i128: left shift i128 by i64 and return
        let sig = Signature {
            params: vec![Type::I128, Type::I64],
            returns: vec![Type::I128],
        };
        let mut isel = InstructionSelector::new("shl128".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        // Value(2) = shl Value(0), Value(1)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Ishl,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();

        // Return Value(2)
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Return,
                args: vec![Value(2)],
                results: vec![],
            },
            entry,
        )
        .unwrap();

        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Formal args: 2 copies for i128 (X0, X1) + 1 copy for i64 (X2)
        assert_eq!(mblock.insts[0].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[0].operands[1], ISelOperand::PReg(gpr::X0));
        assert_eq!(mblock.insts[1].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[1].operands[1], ISelOperand::PReg(gpr::X1));
        assert_eq!(mblock.insts[2].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[2].operands[1], ISelOperand::PReg(gpr::X2));

        // Last 3 instructions: return lo -> X0, return hi -> X1, RET
        let n = mblock.insts.len();
        assert_eq!(mblock.insts[n - 3].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[n - 3].operands[0], ISelOperand::PReg(gpr::X0));
        assert_eq!(mblock.insts[n - 2].opcode, AArch64Opcode::Copy);
        assert_eq!(mblock.insts[n - 2].operands[0], ISelOperand::PReg(gpr::X1));
        assert_eq!(mblock.insts[n - 1].opcode, AArch64Opcode::Ret);
    }

    // -----------------------------------------------------------------------
    // Source location propagation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_source_loc_propagated_through_isel() {
        // Verify that source locations from LIR instructions are propagated
        // to ISelInsts during instruction selection.
        let sig = Signature {
            params: vec![Type::I32, Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("add_with_loc".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        let instructions = vec![
            Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            Instruction {
                opcode: Opcode::Return,
                args: vec![Value(2)],
                results: vec![],
            },
        ];

        let source_locs = vec![
            Some(llvm2_ir::SourceLoc {
                file: 0,
                line: 10,
                col: 5,
            }),
            Some(llvm2_ir::SourceLoc {
                file: 0,
                line: 11,
                col: 3,
            }),
        ];

        isel.select_block_with_source_locs(entry, &instructions, &source_locs)
            .unwrap();
        let func = isel.finalize();

        // The ISel block should have source_locs populated.
        let block = func.blocks.get(&entry).unwrap();
        assert!(!block.source_locs.is_empty());
        // The add instruction generates at least one ISelInst with the add's loc.
        // Find the first non-None source loc and verify it matches.
        let add_loc = block
            .source_locs
            .iter()
            .find(|loc| loc.is_some())
            .expect("should have at least one source loc");
        assert_eq!(
            *add_loc,
            Some(llvm2_ir::SourceLoc {
                file: 0,
                line: 10,
                col: 5
            })
        );
    }

    #[test]
    fn test_source_loc_propagated_to_ir_func() {
        // Verify that source locations survive the ISel -> MachIR conversion.
        let sig = Signature {
            params: vec![Type::I64, Type::I64],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("sub_with_loc".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        let instructions = vec![
            Instruction {
                opcode: Opcode::Isub,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            Instruction {
                opcode: Opcode::Return,
                args: vec![Value(2)],
                results: vec![],
            },
        ];

        let source_locs = vec![
            Some(llvm2_ir::SourceLoc {
                file: 1,
                line: 42,
                col: 8,
            }),
            Some(llvm2_ir::SourceLoc {
                file: 1,
                line: 43,
                col: 1,
            }),
        ];

        isel.select_block_with_source_locs(entry, &instructions, &source_locs)
            .unwrap();
        let func = isel.finalize();
        let ir_func = func.to_ir_func();

        // Verify that at least one MachInst in the IR function carries
        // a source_loc from our ISel input.
        let has_source_loc = ir_func.insts.iter().any(|inst| inst.source_loc.is_some());
        assert!(has_source_loc, "MachInsts should carry source locations");

        // Find the sub instruction and verify its loc.
        let sub_inst = ir_func.insts.iter().find(|inst| {
            inst.source_loc
                == Some(llvm2_ir::SourceLoc {
                    file: 1,
                    line: 42,
                    col: 8,
                })
        });
        assert!(
            sub_inst.is_some(),
            "sub instruction should have line 42 source loc"
        );
    }

    #[test]
    fn test_source_loc_none_when_no_locs_provided() {
        // Verify that select_block (without locs) results in None source_locs.
        let (mut isel, entry) = make_add_isel();
        let instructions = vec![
            Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            Instruction {
                opcode: Opcode::Return,
                args: vec![Value(2)],
                results: vec![],
            },
        ];

        isel.select_block(entry, &instructions).unwrap();
        let func = isel.finalize();

        // All source_locs should be None when not provided.
        let block = func.blocks.get(&entry).unwrap();
        assert!(block.source_locs.iter().all(|loc| loc.is_none()));
    }

    #[test]
    fn test_source_loc_partial_locs_handled() {
        // Verify that a shorter source_locs slice is handled gracefully
        // (missing entries treated as None).
        let sig = Signature {
            params: vec![Type::I32, Type::I32],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("partial".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        let instructions = vec![
            Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            Instruction {
                opcode: Opcode::Return,
                args: vec![Value(2)],
                results: vec![],
            },
        ];

        // Only provide one source loc (shorter than instructions).
        let source_locs = vec![Some(llvm2_ir::SourceLoc {
            file: 0,
            line: 99,
            col: 0,
        })];

        isel.select_block_with_source_locs(entry, &instructions, &source_locs)
            .unwrap();
        let func = isel.finalize();

        let block = func.blocks.get(&entry).unwrap();
        // Should have source_locs for all emitted instructions.
        assert_eq!(block.source_locs.len(), block.insts.len());
        // First instruction (from add) should have the provided loc.
        // The return instruction should have None (index 1 is beyond source_locs).
        let has_line_99 = block.source_locs.iter().any(|loc| {
            *loc == Some(llvm2_ir::SourceLoc {
                file: 0,
                line: 99,
                col: 0,
            })
        });
        assert!(has_line_99, "first instruction should carry line 99");
    }

    // =======================================================================
    // Stack-passed argument tests (9+ args, issue #337)
    // =======================================================================

    #[test]
    fn lower_formal_arguments_nine_i64_stack_arg_uses_fp() {
        // fn(i64 x 9) -> i64: 9th argument overflows to stack.
        // After prologue, stack arg should be loaded relative to FP.
        // ISel emits an IncomingArg(offset) operand; frame lowering later
        // resolves that to `FP + callee_saved_area_size + offset`, because
        // ISel does not yet know the CSA size (chosen by regalloc).
        let sig = Signature {
            params: vec![
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64, // 9th arg -> stack
            ],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("many_args".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // First 8 args: Copy from X0-X7
        for i in 0..8 {
            assert_eq!(mblock.insts[i].opcode, AArch64Opcode::Copy);
            assert_eq!(
                mblock.insts[i].operands[1],
                ISelOperand::PReg(PReg::new(i as u16))
            );
        }

        // 9th arg: LDR from [FP, IncomingArg(0)]
        assert_eq!(mblock.insts[8].opcode, AArch64Opcode::LdrRI);
        // Base register should be FP (X29), not SP
        assert_eq!(mblock.insts[8].operands[1], ISelOperand::PReg(FP));
        // Offset operand should be IncomingArg(0) (ABI-classified offset; frame
        // lowering adds callee_saved_area_size to get the final FP offset).
        assert_eq!(mblock.insts[8].operands[2], ISelOperand::IncomingArg(0));
    }

    #[test]
    fn lower_formal_arguments_ten_i64_stack_offsets() {
        // fn(i64 x 10) -> i64: 9th and 10th overflow to stack.
        // 9th -> IncomingArg(0), 10th -> IncomingArg(8) (8-byte aligned slots).
        let sig = Signature {
            params: vec![
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64, // 9th arg -> stack offset 0
                Type::I64, // 10th arg -> stack offset 8
            ],
            returns: vec![Type::I64],
        };
        let mut isel = InstructionSelector::new("ten_args".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // 9th arg: LDR from [FP, IncomingArg(0)]
        assert_eq!(mblock.insts[8].opcode, AArch64Opcode::LdrRI);
        assert_eq!(mblock.insts[8].operands[1], ISelOperand::PReg(FP));
        assert_eq!(mblock.insts[8].operands[2], ISelOperand::IncomingArg(0));

        // 10th arg: LDR from [FP, IncomingArg(8)]
        assert_eq!(mblock.insts[9].opcode, AArch64Opcode::LdrRI);
        assert_eq!(mblock.insts[9].operands[1], ISelOperand::PReg(FP));
        assert_eq!(mblock.insts[9].operands[2], ISelOperand::IncomingArg(8));
    }

    #[test]
    fn lower_formal_arguments_mixed_types_stack_overflow() {
        // fn(i64, i64, i64, i64, i64, i64, i64, i64, i32) -> i32
        // The i32 overflows to stack (even though it's 32-bit, it occupies
        // an 8-byte slot per ABI). Should emit IncomingArg(0).
        let sig = Signature {
            params: vec![
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I32, // 9th arg -> stack
            ],
            returns: vec![Type::I32],
        };
        let mut isel = InstructionSelector::new("mixed_stack".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // 9th arg (i32): LDR from [FP, IncomingArg(0)]
        assert_eq!(mblock.insts[8].opcode, AArch64Opcode::LdrRI);
        assert_eq!(mblock.insts[8].operands[1], ISelOperand::PReg(FP));
        assert_eq!(mblock.insts[8].operands[2], ISelOperand::IncomingArg(0));
    }

    // =======================================================================
    // i128-widened signed-overflow idiom fast path (issue #430)
    // =======================================================================

    /// Build the canonical overflow-idiom instruction list.
    ///
    /// Formal args are Value(0) = a, Value(1) = b. The idiom produces
    /// - Value(2) = Iadd/Isub(a, b)              (narrow)
    /// - Value(3) = SExt I64->I128(a)
    /// - Value(4) = SExt I64->I128(b)
    /// - Value(5) = Iadd/Isub(Value(3), Value(4)) (wide)
    /// - Value(6) = SExt I64->I128(Value(2))
    /// - Value(7) = Icmp Ne(Value(6), Value(5))
    fn idiom_insts(narrow: Opcode, wide: Opcode) -> Vec<Instruction> {
        vec![
            Instruction {
                opcode: narrow,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            Instruction {
                opcode: Opcode::Sextend {
                    from_ty: Type::I64,
                    to_ty: Type::I128,
                },
                args: vec![Value(0)],
                results: vec![Value(3)],
            },
            Instruction {
                opcode: Opcode::Sextend {
                    from_ty: Type::I64,
                    to_ty: Type::I128,
                },
                args: vec![Value(1)],
                results: vec![Value(4)],
            },
            Instruction {
                opcode: wide,
                args: vec![Value(3), Value(4)],
                results: vec![Value(5)],
            },
            Instruction {
                opcode: Opcode::Sextend {
                    from_ty: Type::I64,
                    to_ty: Type::I128,
                },
                args: vec![Value(2)],
                results: vec![Value(6)],
            },
            Instruction {
                opcode: Opcode::Icmp {
                    cond: IntCC::NotEqual,
                },
                args: vec![Value(6), Value(5)],
                results: vec![Value(7)],
            },
        ]
    }

    #[test]
    fn overflow_idiom_add_with_brif_emits_adds_bvs_b() {
        // fn(a: i64, b: i64) with Brif(overflow, then, else) as terminator.
        let (mut isel, entry) = make_i64_isel();
        let then_block = Block(1);
        let else_block = Block(2);
        isel.func.ensure_block(then_block);
        isel.func.ensure_block(else_block);

        let mut insts = idiom_insts(Opcode::Iadd, Opcode::Iadd);
        // Terminator: Brif(overflow=Value(7), then, else).
        insts.push(Instruction {
            opcode: Opcode::Brif {
                cond: Value(7),
                then_dest: then_block,
                else_dest: else_block,
            },
            args: vec![Value(7)],
            results: vec![],
        });

        isel.select_block(entry, &insts).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Expected trailing sequence: ADDS, B.VS then, B else.
        // (Formal-arg COPYs precede, so we look for the tail.)
        let n = mblock.insts.len();
        assert!(n >= 4, "expected at least ADDS + B.VS + B; got {n}");
        assert_eq!(
            mblock.insts[n - 3].opcode,
            AArch64Opcode::AddsRR,
            "narrow add must be flag-setting ADDS"
        );
        assert_eq!(
            mblock.insts[n - 2].opcode,
            AArch64Opcode::BCond,
            "second-to-last must be B.cond"
        );
        assert_eq!(
            mblock.insts[n - 2].operands[0],
            ISelOperand::CondCode(AArch64CC::VS),
            "B.cond condition must be VS (overflow)"
        );
        assert_eq!(
            mblock.insts[n - 2].operands[1],
            ISelOperand::Block(then_block)
        );
        assert_eq!(
            mblock.insts[n - 1].opcode,
            AArch64Opcode::B,
            "last must be unconditional B to else"
        );
        assert_eq!(
            mblock.insts[n - 1].operands[0],
            ISelOperand::Block(else_block)
        );

        // No CMP (comparing overflow bool to 0) should appear — B.VS reads
        // the V flag directly.
        assert!(
            !mblock
                .insts
                .iter()
                .any(|i| i.opcode == AArch64Opcode::CmpRI),
            "fast path must not emit CMP #0 for the overflow boolean"
        );
        // No CSET — the boolean is never materialised.
        assert!(
            !mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::CSet),
            "fast path must not emit CSET for the overflow boolean"
        );
        // None of the wide-i128 lowering instructions should appear.
        // The i128 ADDS+ADC pair from select_i128_add would produce an Adc;
        // verify none is emitted.
        assert!(
            !mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::Adc),
            "idiom fast path must not emit the ADC from i128 lowering"
        );
    }

    #[test]
    fn overflow_idiom_sub_with_brif_emits_subs_bvs_b() {
        let (mut isel, entry) = make_i64_isel();
        let then_block = Block(1);
        let else_block = Block(2);
        isel.func.ensure_block(then_block);
        isel.func.ensure_block(else_block);

        let mut insts = idiom_insts(Opcode::Isub, Opcode::Isub);
        insts.push(Instruction {
            opcode: Opcode::Brif {
                cond: Value(7),
                then_dest: then_block,
                else_dest: else_block,
            },
            args: vec![Value(7)],
            results: vec![],
        });

        isel.select_block(entry, &insts).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        let n = mblock.insts.len();
        assert!(n >= 4, "expected at least SUBS + B.VS + B; got {n}");
        assert_eq!(
            mblock.insts[n - 3].opcode,
            AArch64Opcode::SubsRR,
            "narrow sub must be flag-setting SUBS"
        );
        assert_eq!(mblock.insts[n - 2].opcode, AArch64Opcode::BCond);
        assert_eq!(
            mblock.insts[n - 2].operands[0],
            ISelOperand::CondCode(AArch64CC::VS)
        );
        assert_eq!(mblock.insts[n - 1].opcode, AArch64Opcode::B);
        assert!(
            !mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::Sbc),
            "idiom fast path must not emit the SBC from i128 lowering"
        );
    }

    #[test]
    fn overflow_idiom_value_consumer_emits_cset_vs() {
        // When the overflow bool is consumed by a non-Brif op (e.g. Return),
        // the selector must materialise the V flag via CSET VS immediately
        // after ADDS so the bool is available as a vreg.
        let sig = Signature {
            params: vec![Type::I64, Type::I64],
            returns: vec![Type::B1],
        };
        let mut isel = InstructionSelector::new("ret_overflow".to_string(), sig.clone());
        let entry = Block(0);
        isel.lower_formal_arguments(&sig, entry).unwrap();

        let mut insts = idiom_insts(Opcode::Iadd, Opcode::Iadd);
        // Terminator: Return(overflow).
        insts.push(Instruction {
            opcode: Opcode::Return,
            args: vec![Value(7)],
            results: vec![],
        });

        isel.select_block(entry, &insts).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Locate ADDS; next instruction must be CSET VS.
        let adds_idx = mblock
            .insts
            .iter()
            .position(|i| i.opcode == AArch64Opcode::AddsRR)
            .expect("expected AddsRR for narrow add");
        assert!(adds_idx + 1 < mblock.insts.len(), "missing follow-up op");
        let cset = &mblock.insts[adds_idx + 1];
        assert_eq!(
            cset.opcode,
            AArch64Opcode::CSet,
            "non-Brif consumer must get CSET VS"
        );
        assert_eq!(
            cset.operands[1],
            ISelOperand::CondCode(AArch64CC::VS),
            "CSET condition must be VS"
        );
        // Idiom must NOT emit an ADC (would have been part of the full i128
        // lowering path).
        assert!(
            !mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::Adc),
            "idiom fast path must not emit ADC"
        );
    }

    #[test]
    fn overflow_idiom_without_pattern_falls_back_to_full_i128() {
        // Sanity check: a plain i64 Add with no idiom around it still goes
        // through select_binop → normal Add. This guards against the guarded
        // match arm firing when it shouldn't.
        let (mut isel, entry) = make_i64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            entry,
        )
        .unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        // No ADDS (flag-setting) — just plain ADD.
        assert!(
            !mblock
                .insts
                .iter()
                .any(|i| i.opcode == AArch64Opcode::AddsRR),
            "plain i64 Add with no overflow idiom must not emit AddsRR"
        );
        assert!(
            mblock
                .insts
                .iter()
                .any(|i| i.opcode == AArch64Opcode::AddRR),
            "plain i64 Add should emit a non-flag-setting AddRR"
        );
    }

    // =======================================================================
    // SMULH fast-path tests (issue #429)
    // =======================================================================

    mod smulh_idiom_tests {
        use super::*;

        /// Build the canonical i128-widened signed-mulhi idiom with a custom
        /// shift-amount immediate. When `shift_imm == 64` this should collapse
        /// to a single SMULH; any other amount must NOT take the fast path.
        fn canonical_smulh_idiom(shift_imm: i64) -> Vec<Instruction> {
            vec![
                Instruction {
                    opcode: Opcode::Sextend { from_ty: Type::I64, to_ty: Type::I128 },
                    args: vec![Value(0)],
                    results: vec![Value(2)],
                },
                Instruction {
                    opcode: Opcode::Sextend { from_ty: Type::I64, to_ty: Type::I128 },
                    args: vec![Value(1)],
                    results: vec![Value(3)],
                },
                Instruction {
                    opcode: Opcode::Imul,
                    args: vec![Value(2), Value(3)],
                    results: vec![Value(4)],
                },
                Instruction {
                    opcode: Opcode::Iconst { ty: Type::I128, imm: shift_imm },
                    args: vec![],
                    results: vec![Value(5)],
                },
                Instruction {
                    opcode: Opcode::Sshr,
                    args: vec![Value(4), Value(5)],
                    results: vec![Value(6)],
                },
                Instruction {
                    opcode: Opcode::Trunc { to_ty: Type::I64 },
                    args: vec![Value(6)],
                    results: vec![Value(7)],
                },
                Instruction {
                    opcode: Opcode::Return,
                    args: vec![Value(7)],
                    results: vec![],
                },
            ]
        }

        #[test]
        fn canonical_idiom_emits_single_smulh() {
            let (mut isel, entry) = make_i64_isel();
            let insts = canonical_smulh_idiom(64);

            isel.select_block(entry, &insts).unwrap();
            let mfunc = isel.finalize();
            let mblock = &mfunc.blocks[&entry];

            let smulh_count = mblock
                .insts
                .iter()
                .filter(|inst| inst.opcode == AArch64Opcode::Smulh)
                .count();
            assert_eq!(
                smulh_count, 1,
                "canonical idiom must collapse to exactly one SMULH; got {smulh_count}"
            );
            // The i128 widened multiply sequence (MUL + UMULH + MADD + MADD)
            // must NOT appear — the SMULH fast path supplants it entirely.
            assert!(
                !mblock.insts.iter().any(|inst| inst.opcode == AArch64Opcode::MulRR),
                "SMULH fast path must not emit the widened low-half MUL"
            );
            assert!(
                !mblock
                    .insts
                    .iter()
                    .any(|inst| matches!(inst.opcode, AArch64Opcode::Umulh | AArch64Opcode::Madd)),
                "SMULH fast path must not emit the widened UMULH/MADD sequence"
            );
        }

        #[test]
        fn shift_by_63_does_not_match_detector() {
            // Exercise the detector directly rather than driving a full
            // `select_block`. The fallback i128-Sshr-by-i128-constant path is
            // unrelated to SMULH and has a separate lowering bug; what we
            // care about here is that the SMULH detector *rejects* any shift
            // amount other than 64.
            let insts = canonical_smulh_idiom(63);
            let analysis = crate::smulh_idiom::detect_smulh_idioms(&insts);

            assert!(
                analysis.by_hi.is_empty(),
                "shift != 64 must not be recognised as an SMULH idiom"
            );
            assert!(
                analysis.by_trunc.is_empty(),
                "shift != 64 must not populate by_trunc"
            );
            assert!(
                analysis.skip_indices.is_empty(),
                "shift != 64 must not mark any instructions as skipped"
            );
        }

        #[test]
        fn shift_by_64_matches_detector() {
            // Positive control for the detector: the canonical idiom with
            // shift=64 must match and mark the four i128 intermediates as
            // skippable.
            let insts = canonical_smulh_idiom(64);
            let analysis = crate::smulh_idiom::detect_smulh_idioms(&insts);

            assert_eq!(
                analysis.by_hi.len(),
                1,
                "canonical idiom with shift=64 must match exactly once"
            );
            assert_eq!(
                analysis.by_trunc.len(),
                1,
                "canonical idiom with shift=64 must register one trunc dispatch"
            );
        }
    }

    // =======================================================================
    // CheckedSadd / CheckedSsub / CheckedSmul (issue #474)
    //
    // These opcodes are emitted by the tMIR adapter for
    // `Inst::Overflow { AddOverflow | SubOverflow | MulOverflow, I64 }` and
    // are lowered directly to the canonical AArch64 flag-setting idiom:
    //     sadd -> ADDS + CSET VS
    //     ssub -> SUBS + CSET VS
    //     smul -> MUL + SMULH + ASR + CMP + CSET NE
    // =======================================================================

    #[test]
    fn select_checked_sadd_i64_emits_adds_cset_vs() {
        let (mut isel, entry) = make_i64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::CheckedSadd,
                args: vec![Value(0), Value(1)],
                // Results are [value_i64, overflow_b1].
                results: vec![Value(2), Value(3)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Find the ADDS and assert the next op is CSET VS.
        let adds_idx = mblock.insts.iter().position(
            |i| i.opcode == AArch64Opcode::AddsRR
        ).expect("CheckedSadd must emit AArch64 ADDS");
        assert!(adds_idx + 1 < mblock.insts.len(), "missing CSET after ADDS");
        let cset = &mblock.insts[adds_idx + 1];
        assert_eq!(cset.opcode, AArch64Opcode::CSet);
        assert_eq!(cset.operands[1], ISelOperand::CondCode(AArch64CC::VS),
            "CSET condition must be VS (signed overflow)");

        // CSET dst must be a vreg (the materialised overflow boolean).
        assert!(
            matches!(cset.operands[0], ISelOperand::VReg(_)),
            "CSET dst must be a vreg"
        );

        // Must NOT emit a plain AddRR (that would be the non-flag path).
        assert!(
            !mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::AddRR),
            "CheckedSadd must not emit non-flag AddRR"
        );
    }

    #[test]
    fn select_checked_ssub_i64_emits_subs_cset_vs() {
        let (mut isel, entry) = make_i64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::CheckedSsub,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2), Value(3)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        let subs_idx = mblock.insts.iter().position(
            |i| i.opcode == AArch64Opcode::SubsRR
        ).expect("CheckedSsub must emit AArch64 SUBS");
        assert!(subs_idx + 1 < mblock.insts.len(), "missing CSET after SUBS");
        let cset = &mblock.insts[subs_idx + 1];
        assert_eq!(cset.opcode, AArch64Opcode::CSet);
        assert_eq!(cset.operands[1], ISelOperand::CondCode(AArch64CC::VS));

        assert!(
            !mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::SubRR),
            "CheckedSsub must not emit non-flag SubRR"
        );
    }

    #[test]
    fn select_checked_smul_i64_emits_mul_smulh_asr_cmp_cset_ne() {
        let (mut isel, entry) = make_i64_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::CheckedSmul,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2), Value(3)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];

        // Expected idiom order: MUL, SMULH, ASR #63, CMP, CSET NE.
        // Other instructions (arg copies) may precede MUL, so search and
        // verify relative positions rather than absolute indices.
        let mul_idx = mblock.insts.iter().position(
            |i| i.opcode == AArch64Opcode::MulRR
        ).expect("CheckedSmul must emit MUL");
        let smulh_idx = mblock.insts.iter().position(
            |i| i.opcode == AArch64Opcode::Smulh
        ).expect("CheckedSmul must emit SMULH (#429)");
        let asr_idx = mblock.insts.iter().position(
            |i| i.opcode == AArch64Opcode::AsrRI
        ).expect("CheckedSmul must emit ASR (sign-extract low 63)");
        let cmp_idx = mblock.insts.iter().position(
            |i| i.opcode == AArch64Opcode::CmpRR
        ).expect("CheckedSmul must emit CMP Xhi, Xsign");
        let cset_idx = mblock.insts.iter().position(
            |i| i.opcode == AArch64Opcode::CSet
        ).expect("CheckedSmul must emit CSET NE");

        assert!(mul_idx < smulh_idx, "MUL must precede SMULH");
        assert!(smulh_idx < asr_idx, "SMULH must precede ASR");
        assert!(asr_idx < cmp_idx, "ASR must precede CMP");
        assert!(cmp_idx < cset_idx, "CMP must precede CSET");

        // ASR shift amount must be 63 (sign-extract bit 63 to all positions).
        let asr = &mblock.insts[asr_idx];
        assert_eq!(asr.operands[2], ISelOperand::Imm(63),
            "ASR shift must be #63 to sign-extend the low half");

        // CSET condition must be NE (overflow iff Xhi != Xsign).
        let cset = &mblock.insts[cset_idx];
        assert_eq!(cset.operands[1], ISelOperand::CondCode(AArch64CC::NE),
            "CSET condition must be NE (overflow iff high != sign-extension of low)");
    }

    #[test]
    fn select_checked_sadd_i32_emits_adds_w() {
        // I32 is also supported by the flag-setting selector because
        // AddsRR/SubsRR dispatch on the operand register class (GPR32 vs
        // GPR64). This sanity-checks that we don't accidentally restrict
        // the path to I64 only.
        let (mut isel, entry) = make_add_isel();
        isel.select_instruction(
            &Instruction {
                opcode: Opcode::CheckedSadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2), Value(3)],
            },
            entry,
        ).unwrap();
        let mfunc = isel.finalize();
        let mblock = &mfunc.blocks[&entry];
        assert!(
            mblock.insts.iter().any(|i| i.opcode == AArch64Opcode::AddsRR),
            "CheckedSadd on I32 must emit AArch64 ADDS (flag-setting)"
        );
        assert!(
            mblock.insts.iter().any(|i|
                i.opcode == AArch64Opcode::CSet
                && i.operands.get(1) == Some(&ISelOperand::CondCode(AArch64CC::VS))
            ),
            "CheckedSadd on I32 must emit CSET VS"
        );
    }
}
