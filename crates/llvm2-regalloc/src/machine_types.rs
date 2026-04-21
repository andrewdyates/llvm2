// llvm2-regalloc/machine_types.rs - Machine-level types for register allocation
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Machine-level type definitions used by the register allocator.
//!
//! See [`llvm2_ir::type_hierarchy`] for the full type hierarchy documentation.
//!
//! ## Unified types (re-exported from `llvm2-ir`)
//!
//! These types are shared with `llvm2-ir` via re-export (no adapter needed):
//!
//! | Type | Source |
//! |------|--------|
//! | `VReg`, `PReg`, `RegClass` | `llvm2_ir::regs` |
//! | `BlockId`, `InstId`, `StackSlotId` | `llvm2_ir::types` |
//! | `InstFlags` | `llvm2_ir::inst` (unified in issue #73) |
//!
//! ## Compound types (regalloc-specific, renamed in issue #73)
//!
//! The compound types are regalloc-specific versions with structural differences
//! required for register allocation. They have been renamed (issue #73) to avoid
//! shadowing the canonical `llvm2-ir` types:
//!
//! | Canonical (llvm2-ir) | RegAlloc name | Why separate | Conversion |
//! |---------------------|---------------|--------------|------------|
//! | `MachInst` | `RegAllocInst` | Separates defs/uses for liveness; opcode is `u16` not enum | `classify_def_use()` in pipeline |
//! | `MachOperand` | `RegAllocOperand` | Subset of variants (no MemOp/FrameIndex/Special) | `TryFrom<&MachOperand>` |
//! | `MachBlock` | `RegAllocBlock` | Structurally identical (loop_depth added to IR, issue #73) | `From<&MachBlock>` |
//! | `StackSlot` | `RegAllocStackSlot` | No alignment assertion | `From<&StackSlot>` |
//! | `MachFunction` | `RegAllocFunction` | `HashMap` stack slots, `next_stack_slot` counter | `ir_to_regalloc()` in pipeline |
//!
//! Backward-compatible type aliases (`MachInst`, `MachBlock`, etc.) are provided
//! but deprecated. New code should use the `RegAlloc*` names.
//!
//! The remaining unification target is `MachInst` / `RegAllocInst`: enriching
//! `llvm2_ir::MachInst` with def/use classification would eliminate the need
//! for `RegAllocInst`. See issue #73 for tracking.
//!
//! Reference: `~/llvm-project-ref/llvm/include/llvm/CodeGen/MachineInstr.h`

use std::collections::HashMap;

// Re-export canonical primitive types from llvm2-ir.
pub use llvm2_ir::regs::{PReg, RegClass, VReg};
pub use llvm2_ir::types::{BlockId, InstId, StackSlotId};

// ---------------------------------------------------------------------------
// InstFlags — unified, re-exported from llvm2-ir (issue #73)
// ---------------------------------------------------------------------------
//
// Previously this module had its own `InstFlags` struct with the same bit
// encoding but a different API (`pub u16` inner field, `u16` constants).
// As of issue #73, regalloc uses the canonical `llvm2_ir::InstFlags` directly.
//
// Migration note for existing code:
//   Old: `InstFlags(InstFlags::IS_CALL | InstFlags::IS_BRANCH)`
//   New: `InstFlags::IS_CALL.union(InstFlags::IS_BRANCH)`
//   Or:  `InstFlags::from_bits(0x01 | 0x02)`
//
// Query methods (`is_call()`, `is_branch()`, etc.) now live on `InstFlags`
// itself (in llvm2-ir), so `inst.flags.is_call()` works the same as before.
// ---------------------------------------------------------------------------
pub use llvm2_ir::inst::InstFlags;

/// Operand of a regalloc-level machine instruction.
///
/// Subset of the canonical `llvm2_ir::MachOperand` — omits `MemOp`,
/// `FrameIndex`, and `Special` variants which are not needed for register
/// allocation. Will be unified with `llvm2_ir::MachOperand` in a future phase.
///
/// Named `RegAllocOperand` (issue #73) to avoid confusion with
/// `llvm2_ir::MachOperand`, the canonical operand type.
#[derive(Debug, Clone, PartialEq)]
pub enum RegAllocOperand {
    VReg(VReg),
    PReg(PReg),
    Imm(i64),
    FImm(f64),
    Block(BlockId),
    StackSlot(StackSlotId),
}

/// Backward-compatible alias (deprecated). Use `RegAllocOperand` directly.
pub type MachOperand = RegAllocOperand;

impl RegAllocOperand {
    /// Returns the VReg if this operand is a virtual register.
    pub fn as_vreg(&self) -> Option<VReg> {
        match self {
            RegAllocOperand::VReg(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the PReg if this operand is a physical register.
    pub fn as_preg(&self) -> Option<PReg> {
        match self {
            RegAllocOperand::PReg(p) => Some(*p),
            _ => None,
        }
    }
}

/// A machine instruction for register allocation.
///
/// Unlike the canonical `llvm2_ir::MachInst` which stores all operands in a
/// single list, this struct separates defs (outputs) from uses (inputs) for
/// efficient liveness analysis. The opcode is stored as u16 to be
/// target-independent.
///
/// Named `RegAllocInst` (issue #73) to avoid confusion with `llvm2_ir::MachInst`.
/// A future unification pass will reconcile this with the llvm2-ir model,
/// likely by adding def/use classification to `llvm2_ir::MachInst`.
#[derive(Debug, Clone)]
pub struct RegAllocInst {
    /// Target-specific opcode.
    pub opcode: u16,
    /// Defined (output) operands.
    pub defs: Vec<RegAllocOperand>,
    /// Used (input) operands.
    pub uses: Vec<RegAllocOperand>,
    /// Physical registers implicitly defined (e.g., call clobbers).
    pub implicit_defs: Vec<PReg>,
    /// Physical registers implicitly used.
    pub implicit_uses: Vec<PReg>,
    /// Instruction flags.
    pub flags: InstFlags,
}

/// Backward-compatible alias (deprecated). Use `RegAllocInst` directly.
pub type MachInst = RegAllocInst;

impl RegAllocInst {
    /// Returns all VRegs defined by this instruction.
    pub fn vreg_defs(&self) -> impl Iterator<Item = VReg> + '_ {
        self.defs.iter().filter_map(|op| op.as_vreg())
    }

    /// Returns all VRegs used by this instruction.
    pub fn vreg_uses(&self) -> impl Iterator<Item = VReg> + '_ {
        self.uses.iter().filter_map(|op| op.as_vreg())
    }
}

/// A regalloc-level machine basic block.
///
/// Structurally identical to `llvm2_ir::MachBlock` after the `loop_depth`
/// field was added to the canonical type (issue #73). The `From<&MachBlock>`
/// impl provides lossless conversion.
///
/// Named `RegAllocBlock` (issue #73) to avoid confusion with `llvm2_ir::MachBlock`.
#[derive(Debug, Clone)]
pub struct RegAllocBlock {
    /// Instructions in this block, in order.
    pub insts: Vec<InstId>,
    /// Predecessor blocks.
    pub preds: Vec<BlockId>,
    /// Successor blocks.
    pub succs: Vec<BlockId>,
    /// Loop depth (0 = not in a loop). Used for spill weight computation.
    pub loop_depth: u32,
}

/// Backward-compatible alias (deprecated). Use `RegAllocBlock` directly.
pub type MachBlock = RegAllocBlock;

/// A stack slot for spilled values.
///
/// Same fields as `llvm2_ir::function::StackSlot` but without the
/// `debug_assert` alignment check. Convertible via `From<&StackSlot>`
/// (issue #73).
///
/// Named `RegAllocStackSlot` (issue #73) to avoid confusion with
/// `llvm2_ir::function::StackSlot`.
#[derive(Debug, Clone)]
pub struct RegAllocStackSlot {
    pub size: u32,
    pub align: u32,
}

/// Backward-compatible alias (deprecated). Use `RegAllocStackSlot` directly.
pub type StackSlot = RegAllocStackSlot;

/// A regalloc-level machine function -- the unit of register allocation.
///
/// Differs from the canonical `llvm2_ir::MachFunction` in several ways:
/// - Uses regalloc-specific `RegAllocInst` with separated defs/uses
/// - Uses `HashMap<StackSlotId, RegAllocStackSlot>` instead of `Vec<StackSlot>`
/// - Includes `next_stack_slot` counter for spill allocation
///
/// Named `RegAllocFunction` (issue #73) to avoid confusion with
/// `llvm2_ir::MachFunction`, the canonical machine function type.
/// Will be unified in a future phase.
#[derive(Debug, Clone)]
pub struct RegAllocFunction {
    pub name: String,
    /// All instructions, indexed by InstId.
    pub insts: Vec<RegAllocInst>,
    /// All blocks, indexed by BlockId.
    pub blocks: Vec<RegAllocBlock>,
    /// Block ordering (RPO or linear).
    pub block_order: Vec<BlockId>,
    /// Entry block.
    pub entry_block: BlockId,
    /// Next available VReg id.
    pub next_vreg: u32,
    /// Next available stack slot id.
    pub next_stack_slot: u32,
    /// Stack slots allocated (for spills and locals).
    pub stack_slots: HashMap<StackSlotId, RegAllocStackSlot>,
}

/// Backward-compatible alias (deprecated). Use `RegAllocFunction` directly.
pub type MachFunction = RegAllocFunction;

// ---------------------------------------------------------------------------
// From / TryFrom impls — canonical llvm2-ir types -> regalloc types (issue #73)
// ---------------------------------------------------------------------------
//
// These conversions formalize the adapter code that was previously scattered
// in `llvm2-codegen/pipeline.rs`. The pipeline adapter (`ir_to_regalloc`)
// now delegates to these impls for individual type conversions.

/// Error type for operand conversions that encounter IR-only variants.
#[derive(Debug, Clone)]
pub struct OperandConversionError {
    pub message: String,
}

impl std::fmt::Display for OperandConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "operand conversion error: {}", self.message)
    }
}

impl std::error::Error for OperandConversionError {}

/// Convert a canonical `llvm2_ir::MachOperand` to a `RegAllocOperand`.
///
/// Succeeds for the subset of operand variants that regalloc handles
/// (VReg, PReg, Imm, FImm, Block, StackSlot). Fails for IR-only variants
/// (FrameIndex, MemOp, Special) that must be lowered before register
/// allocation. Symbol operands are mapped to Imm(0) since regalloc does
/// not track relocation metadata; likewise `JumpTableIndex` is mapped to
/// Imm(0) because the table index is opaque to liveness analysis (the
/// codegen pipeline reads the original IR operand post-regalloc to patch
/// ADR instructions and append the table bytes).
impl TryFrom<&llvm2_ir::MachOperand> for RegAllocOperand {
    type Error = OperandConversionError;

    fn try_from(op: &llvm2_ir::MachOperand) -> Result<Self, Self::Error> {
        use llvm2_ir::MachOperand as IrOp;
        match op {
            IrOp::VReg(v) => Ok(RegAllocOperand::VReg(*v)),
            IrOp::PReg(p) => Ok(RegAllocOperand::PReg(*p)),
            IrOp::Imm(i) => Ok(RegAllocOperand::Imm(*i)),
            IrOp::FImm(f) => Ok(RegAllocOperand::FImm(*f)),
            IrOp::Block(b) => Ok(RegAllocOperand::Block(*b)),
            IrOp::StackSlot(s) => Ok(RegAllocOperand::StackSlot(*s)),
            IrOp::Symbol(_) => Ok(RegAllocOperand::Imm(0)),
            IrOp::JumpTableIndex(_) => Ok(RegAllocOperand::Imm(0)),
            IrOp::FrameIndex(fi) => Err(OperandConversionError {
                message: format!(
                    "FrameIndex({:?}) must be eliminated before register allocation",
                    fi
                ),
            }),
            IrOp::MemOp { base, offset } => Err(OperandConversionError {
                message: format!(
                    "MemOp(base={:?}, offset={}) must be lowered before register allocation",
                    base, offset
                ),
            }),
            IrOp::Special(s) => Err(OperandConversionError {
                message: format!(
                    "Special({:?}) must be lowered to PReg before register allocation",
                    s
                ),
            }),
            // IncomingArg is an abstract offset operand that frame lowering
            // resolves to a concrete FP-relative immediate *after* register
            // allocation. It carries no register reference, so regalloc sees
            // it as an opaque constant — same treatment as Symbol / JumpTableIndex.
            // The original IR operand is preserved in the MachFunction (regalloc
            // only rewrites VReg operands), so frame lowering can still find
            // and rewrite it.
            IrOp::IncomingArg(_) => Ok(RegAllocOperand::Imm(0)),
        }
    }
}

/// Convert a canonical `llvm2_ir::MachBlock` to a `RegAllocBlock`.
///
/// Copies insts, preds, succs, and loop_depth directly. The loop_depth
/// field was added to `llvm2_ir::MachBlock` in issue #73 to enable this
/// lossless conversion.
impl From<&llvm2_ir::function::MachBlock> for RegAllocBlock {
    fn from(block: &llvm2_ir::function::MachBlock) -> Self {
        RegAllocBlock {
            insts: block.insts.clone(),
            preds: block.preds.clone(),
            succs: block.succs.clone(),
            loop_depth: block.loop_depth,
        }
    }
}

/// Convert a canonical `llvm2_ir::function::StackSlot` to a `RegAllocStackSlot`.
impl From<&llvm2_ir::function::StackSlot> for RegAllocStackSlot {
    fn from(slot: &llvm2_ir::function::StackSlot) -> Self {
        RegAllocStackSlot {
            size: slot.size,
            align: slot.align,
        }
    }
}

impl RegAllocFunction {
    /// Allocate a new stack slot for a spill.
    pub fn alloc_stack_slot(&mut self, size: u32, align: u32) -> StackSlotId {
        let id = StackSlotId(self.next_stack_slot);
        self.next_stack_slot += 1;
        self.stack_slots.insert(id, RegAllocStackSlot { size, align });
        id
    }

    /// Allocate a new virtual register.
    pub fn alloc_vreg(&mut self, class: RegClass) -> VReg {
        let id = self.next_vreg;
        self.next_vreg += 1;
        VReg { id, class }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn reg_alloc_operand_as_vreg_returns_some_for_vreg_variant() {
        let vreg = VReg::new(7, RegClass::Gpr64);
        let operand = RegAllocOperand::VReg(vreg);

        assert_eq!(operand.as_vreg(), Some(vreg));
    }

    #[test]
    fn reg_alloc_operand_as_vreg_returns_none_for_non_vreg_variants() {
        let operands = vec![
            RegAllocOperand::PReg(PReg::new(3)),
            RegAllocOperand::Imm(42),
            RegAllocOperand::FImm(3.25),
            RegAllocOperand::Block(BlockId(2)),
            RegAllocOperand::StackSlot(StackSlotId(7)),
        ];

        for operand in operands {
            assert_eq!(operand.as_vreg(), None);
        }
    }

    #[test]
    fn reg_alloc_operand_as_preg_returns_some_for_preg_variant() {
        let preg = PReg::new(11);
        let operand = RegAllocOperand::PReg(preg);

        assert_eq!(operand.as_preg(), Some(preg));
    }

    #[test]
    fn reg_alloc_operand_as_preg_returns_none_for_non_preg_variants() {
        let operands = vec![
            RegAllocOperand::VReg(VReg::new(1, RegClass::Gpr64)),
            RegAllocOperand::Imm(99),
            RegAllocOperand::FImm(1.5),
            RegAllocOperand::Block(BlockId(4)),
            RegAllocOperand::StackSlot(StackSlotId(9)),
        ];

        for operand in operands {
            assert_eq!(operand.as_preg(), None);
        }
    }

    #[test]
    fn reg_alloc_inst_vreg_defs_extracts_only_vreg_operands_from_defs() {
        let vreg0 = VReg::new(1, RegClass::Gpr64);
        let vreg1 = VReg::new(2, RegClass::Fpr64);
        let inst = RegAllocInst {
            opcode: 123,
            defs: vec![
                RegAllocOperand::Imm(10),
                RegAllocOperand::VReg(vreg0),
                RegAllocOperand::PReg(PReg::new(5)),
                RegAllocOperand::Block(BlockId(1)),
                RegAllocOperand::FImm(2.0),
                RegAllocOperand::VReg(vreg1),
                RegAllocOperand::StackSlot(StackSlotId(3)),
            ],
            uses: vec![],
            implicit_defs: vec![],
            implicit_uses: vec![],
            flags: InstFlags::default(),
        };

        let defs: Vec<_> = inst.vreg_defs().collect();

        assert_eq!(defs, vec![vreg0, vreg1]);
    }

    #[test]
    fn reg_alloc_inst_vreg_uses_extracts_only_vreg_operands_from_uses() {
        let vreg0 = VReg::new(8, RegClass::Gpr32);
        let vreg1 = VReg::new(9, RegClass::Fpr32);
        let inst = RegAllocInst {
            opcode: 456,
            defs: vec![],
            uses: vec![
                RegAllocOperand::PReg(PReg::new(1)),
                RegAllocOperand::VReg(vreg0),
                RegAllocOperand::Imm(7),
                RegAllocOperand::StackSlot(StackSlotId(5)),
                RegAllocOperand::VReg(vreg1),
                RegAllocOperand::Block(BlockId(2)),
                RegAllocOperand::FImm(4.5),
            ],
            implicit_defs: vec![],
            implicit_uses: vec![],
            flags: InstFlags::default(),
        };

        let uses: Vec<_> = inst.vreg_uses().collect();

        assert_eq!(uses, vec![vreg0, vreg1]);
    }

    #[test]
    fn reg_alloc_inst_vreg_defs_returns_empty_when_defs_only_contain_preg_and_imm() {
        let inst = RegAllocInst {
            opcode: 789,
            defs: vec![
                RegAllocOperand::PReg(PReg::new(2)),
                RegAllocOperand::Imm(17),
                RegAllocOperand::PReg(PReg::new(6)),
                RegAllocOperand::Imm(-1),
            ],
            uses: vec![],
            implicit_defs: vec![],
            implicit_uses: vec![],
            flags: InstFlags::default(),
        };

        assert!(inst.vreg_defs().collect::<Vec<_>>().is_empty());
    }

    #[test]
    fn reg_alloc_function_alloc_stack_slot_increments_next_stack_slot() {
        let mut func = RegAllocFunction {
            name: "test".into(),
            insts: vec![],
            blocks: vec![],
            block_order: vec![],
            entry_block: BlockId(0),
            next_vreg: 0,
            next_stack_slot: 3,
            stack_slots: HashMap::new(),
        };

        let first = func.alloc_stack_slot(8, 8);
        let second = func.alloc_stack_slot(4, 4);

        assert_eq!(first, StackSlotId(3));
        assert_eq!(second, StackSlotId(4));
        assert_eq!(func.next_stack_slot, 5);
    }

    #[test]
    fn reg_alloc_function_alloc_vreg_increments_next_vreg() {
        let mut func = RegAllocFunction {
            name: "test".into(),
            insts: vec![],
            blocks: vec![],
            block_order: vec![],
            entry_block: BlockId(0),
            next_vreg: 11,
            next_stack_slot: 0,
            stack_slots: HashMap::new(),
        };

        let first = func.alloc_vreg(RegClass::Gpr32);
        let second = func.alloc_vreg(RegClass::Fpr64);

        assert_eq!(first, VReg::new(11, RegClass::Gpr32));
        assert_eq!(second, VReg::new(12, RegClass::Fpr64));
        assert_eq!(func.next_vreg, 13);
    }

    #[test]
    fn reg_alloc_function_alloc_stack_slot_inserts_into_stack_slots_map() {
        let mut func = RegAllocFunction {
            name: "test".into(),
            insts: vec![],
            blocks: vec![],
            block_order: vec![],
            entry_block: BlockId(0),
            next_vreg: 0,
            next_stack_slot: 0,
            stack_slots: HashMap::new(),
        };

        let slot_id = func.alloc_stack_slot(16, 8);

        assert_eq!(func.stack_slots.len(), 1);
        assert_eq!(
            func.stack_slots.get(&slot_id).map(|slot| (slot.size, slot.align)),
            Some((16, 8))
        );
    }

    // ---- From / TryFrom conversion tests (issue #73) ----

    #[test]
    fn try_from_ir_operand_vreg() {
        let vreg = VReg::new(5, RegClass::Gpr64);
        let ir_op = llvm2_ir::MachOperand::VReg(vreg);
        let ra_op = RegAllocOperand::try_from(&ir_op).unwrap();
        assert_eq!(ra_op, RegAllocOperand::VReg(vreg));
    }

    #[test]
    fn try_from_ir_operand_preg() {
        let preg = PReg::new(10);
        let ir_op = llvm2_ir::MachOperand::PReg(preg);
        let ra_op = RegAllocOperand::try_from(&ir_op).unwrap();
        assert_eq!(ra_op, RegAllocOperand::PReg(preg));
    }

    #[test]
    fn try_from_ir_operand_imm() {
        let ir_op = llvm2_ir::MachOperand::Imm(42);
        let ra_op = RegAllocOperand::try_from(&ir_op).unwrap();
        assert_eq!(ra_op, RegAllocOperand::Imm(42));
    }

    #[test]
    fn try_from_ir_operand_fimm() {
        let ir_op = llvm2_ir::MachOperand::FImm(2.78);
        let ra_op = RegAllocOperand::try_from(&ir_op).unwrap();
        assert_eq!(ra_op, RegAllocOperand::FImm(2.78));
    }

    #[test]
    fn try_from_ir_operand_block() {
        let ir_op = llvm2_ir::MachOperand::Block(BlockId(7));
        let ra_op = RegAllocOperand::try_from(&ir_op).unwrap();
        assert_eq!(ra_op, RegAllocOperand::Block(BlockId(7)));
    }

    #[test]
    fn try_from_ir_operand_stack_slot() {
        let ir_op = llvm2_ir::MachOperand::StackSlot(StackSlotId(3));
        let ra_op = RegAllocOperand::try_from(&ir_op).unwrap();
        assert_eq!(ra_op, RegAllocOperand::StackSlot(StackSlotId(3)));
    }

    #[test]
    fn try_from_ir_operand_symbol_maps_to_imm_zero() {
        let ir_op = llvm2_ir::MachOperand::Symbol("_printf".to_string());
        let ra_op = RegAllocOperand::try_from(&ir_op).unwrap();
        assert_eq!(ra_op, RegAllocOperand::Imm(0));
    }

    #[test]
    fn try_from_ir_operand_frame_index_fails() {
        let ir_op = llvm2_ir::MachOperand::FrameIndex(llvm2_ir::types::FrameIdx(-8));
        let result = RegAllocOperand::try_from(&ir_op);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("FrameIndex"));
    }

    #[test]
    fn try_from_ir_operand_memop_fails() {
        let ir_op = llvm2_ir::MachOperand::MemOp { base: PReg::new(0), offset: 16 };
        let result = RegAllocOperand::try_from(&ir_op);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("MemOp"));
    }

    #[test]
    fn try_from_ir_operand_special_fails() {
        let ir_op = llvm2_ir::MachOperand::Special(llvm2_ir::regs::SpecialReg::SP);
        let result = RegAllocOperand::try_from(&ir_op);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("Special"));
    }

    #[test]
    fn from_ir_block_copies_all_fields() {
        let ir_block = llvm2_ir::function::MachBlock {
            insts: vec![InstId(0), InstId(1), InstId(2)],
            preds: vec![BlockId(5)],
            succs: vec![BlockId(6), BlockId(7)],
            loop_depth: 3,
        };
        let ra_block = RegAllocBlock::from(&ir_block);
        assert_eq!(ra_block.insts, vec![InstId(0), InstId(1), InstId(2)]);
        assert_eq!(ra_block.preds, vec![BlockId(5)]);
        assert_eq!(ra_block.succs, vec![BlockId(6), BlockId(7)]);
        assert_eq!(ra_block.loop_depth, 3);
    }

    #[test]
    fn from_ir_block_empty() {
        let ir_block = llvm2_ir::function::MachBlock::new();
        let ra_block = RegAllocBlock::from(&ir_block);
        assert!(ra_block.insts.is_empty());
        assert!(ra_block.preds.is_empty());
        assert!(ra_block.succs.is_empty());
        assert_eq!(ra_block.loop_depth, 0);
    }

    #[test]
    fn from_ir_stack_slot() {
        let ir_slot = llvm2_ir::function::StackSlot::new(16, 8);
        let ra_slot = RegAllocStackSlot::from(&ir_slot);
        assert_eq!(ra_slot.size, 16);
        assert_eq!(ra_slot.align, 8);
    }
}
