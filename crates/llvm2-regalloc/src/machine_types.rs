// llvm2-regalloc/machine_types.rs - Machine-level types for register allocation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Machine-level type definitions used by the register allocator.
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
//! | Canonical (llvm2-ir) | RegAlloc name | Why separate |
//! |---------------------|---------------|--------------|
//! | `MachInst` | `RegAllocInst` | Separates defs/uses for liveness; opcode is `u16` not enum |
//! | `MachOperand` | `RegAllocOperand` | Subset of variants (no MemOp/FrameIndex/Special) |
//! | `MachBlock` | `RegAllocBlock` | Adds `loop_depth` for spill weight computation |
//! | `StackSlot` | `RegAllocStackSlot` | No alignment assertion |
//! | `MachFunction` | `RegAllocFunction` | `HashMap` stack slots, `next_stack_slot` counter |
//!
//! Backward-compatible type aliases (`MachInst`, `MachBlock`, etc.) are provided
//! but deprecated. New code should use the `RegAlloc*` names.
//!
//! These will be unified with the `llvm2-ir` versions in a future phase,
//! likely by enriching `llvm2_ir::MachInst` with def/use classification.
//! See issue #73 for tracking.
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
/// Extends `llvm2_ir::MachBlock` with `loop_depth` for spill weight
/// computation. Will be unified in a future phase.
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
/// `debug_assert` alignment check. Will be unified.
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
