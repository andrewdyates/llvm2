// llvm2-regalloc/machine_types.rs - Machine-level types for register allocation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Machine-level type definitions used by the register allocator.
//!
//! ## Primitive types (re-exported from `llvm2-ir`)
//!
//! `VReg`, `PReg`, `RegClass`, `BlockId`, `InstId`, `StackSlotId` are
//! imported from `llvm2-ir`, the canonical source of truth for machine IR types.
//!
//! ## Compound types (regalloc-specific, intentionally separate)
//!
//! The compound types (`MachInst`, `MachBlock`, `MachFunction`, `MachOperand`,
//! `InstFlags`, `StackSlot`) are regalloc-specific versions that shadow the
//! `llvm2-ir` types by name. They are NOT stale stubs -- they have structural
//! differences required for register allocation:
//!
//! | Type | Why separate |
//! |------|--------------|
//! | `MachInst` | Separates defs/uses for liveness; opcode is `u16` not enum |
//! | `MachOperand` | Subset of variants (no MemOp/FrameIndex/Special) |
//! | `InstFlags` | Same encoding, different API (`pub u16` vs typed constants) |
//! | `MachBlock` | Adds `loop_depth` for spill weight computation |
//! | `StackSlot` | No alignment assertion |
//! | `MachFunction` | `HashMap` stack slots, `next_stack_slot` counter |
//!
//! These will be unified with the `llvm2-ir` versions in a future phase,
//! likely by enriching `llvm2_ir::MachInst` with def/use classification.
//! See issue #37 for tracking.
//!
//! Reference: `~/llvm-project-ref/llvm/include/llvm/CodeGen/MachineInstr.h`

use std::collections::HashMap;

// Re-export canonical primitive types from llvm2-ir.
pub use llvm2_ir::regs::{PReg, RegClass, VReg};
pub use llvm2_ir::types::{BlockId, InstId, StackSlotId};

/// Operand of a regalloc-level machine instruction.
///
/// Subset of `llvm2_ir::MachOperand` — omits `MemOp`, `FrameIndex`, and
/// `Special` variants which are not needed for register allocation.
/// Will be unified with `llvm2_ir::MachOperand` in a future phase. (#37)
#[derive(Debug, Clone, PartialEq)]
pub enum MachOperand {
    VReg(VReg),
    PReg(PReg),
    Imm(i64),
    FImm(f64),
    Block(BlockId),
    StackSlot(StackSlotId),
}

impl MachOperand {
    /// Returns the VReg if this operand is a virtual register.
    pub fn as_vreg(&self) -> Option<VReg> {
        match self {
            MachOperand::VReg(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the PReg if this operand is a physical register.
    pub fn as_preg(&self) -> Option<PReg> {
        match self {
            MachOperand::PReg(p) => Some(*p),
            _ => None,
        }
    }
}

/// Instruction flags describing side effects and control flow.
///
/// Duplicates `llvm2_ir::InstFlags` with the same bit encoding but a
/// different API: exposes constants as `u16` values (not `Self` constants)
/// and uses `pub` inner field to allow direct construction via bitwise OR
/// (e.g., `InstFlags(IS_CALL | IS_BRANCH)`). The `llvm2_ir::InstFlags`
/// version uses `const fn union()` and typed constants instead.
/// Will be unified with `llvm2_ir::InstFlags` in a future phase. (#37)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InstFlags(pub u16);

impl InstFlags {
    pub const IS_CALL: u16 = 0x01;
    pub const IS_BRANCH: u16 = 0x02;
    pub const IS_RETURN: u16 = 0x04;
    pub const IS_TERMINATOR: u16 = 0x08;
    pub const HAS_SIDE_EFFECTS: u16 = 0x10;
    pub const IS_PSEUDO: u16 = 0x20;
    pub const READS_MEMORY: u16 = 0x40;
    pub const WRITES_MEMORY: u16 = 0x80;
    pub const IS_PHI: u16 = 0x100;

    pub fn is_call(self) -> bool {
        self.0 & Self::IS_CALL != 0
    }
    pub fn is_branch(self) -> bool {
        self.0 & Self::IS_BRANCH != 0
    }
    pub fn is_return(self) -> bool {
        self.0 & Self::IS_RETURN != 0
    }
    pub fn is_terminator(self) -> bool {
        self.0 & Self::IS_TERMINATOR != 0
    }
    pub fn is_phi(self) -> bool {
        self.0 & Self::IS_PHI != 0
    }
}

/// A machine instruction for register allocation.
///
/// Unlike `llvm2_ir::MachInst` which stores all operands in a single list,
/// this struct separates defs (outputs) from uses (inputs) for efficient
/// liveness analysis. The opcode is stored as u16 to be target-independent.
///
/// A future unification pass will reconcile this with the llvm2-ir model,
/// likely by adding def/use classification to `llvm2_ir::MachInst`.
#[derive(Debug, Clone)]
pub struct MachInst {
    /// Target-specific opcode.
    pub opcode: u16,
    /// Defined (output) operands.
    pub defs: Vec<MachOperand>,
    /// Used (input) operands.
    pub uses: Vec<MachOperand>,
    /// Physical registers implicitly defined (e.g., call clobbers).
    pub implicit_defs: Vec<PReg>,
    /// Physical registers implicitly used.
    pub implicit_uses: Vec<PReg>,
    /// Instruction flags.
    pub flags: InstFlags,
}

impl MachInst {
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
/// computation. Will be unified in a future phase. (#37)
#[derive(Debug, Clone)]
pub struct MachBlock {
    /// Instructions in this block, in order.
    pub insts: Vec<InstId>,
    /// Predecessor blocks.
    pub preds: Vec<BlockId>,
    /// Successor blocks.
    pub succs: Vec<BlockId>,
    /// Loop depth (0 = not in a loop). Used for spill weight computation.
    pub loop_depth: u32,
}

/// A stack slot for spilled values.
///
/// Same fields as `llvm2_ir::function::StackSlot` but without the
/// `debug_assert` alignment check. Will be unified. (#37)
#[derive(Debug, Clone)]
pub struct StackSlot {
    pub size: u32,
    pub align: u32,
}

/// A regalloc-level machine function -- the unit of register allocation.
///
/// Differs from `llvm2_ir::MachFunction` in several ways:
/// - Uses regalloc-specific `MachInst` with separated defs/uses
/// - Uses `HashMap<StackSlotId, StackSlot>` instead of `Vec<StackSlot>`
/// - Includes `next_stack_slot` counter for spill allocation
/// Will be unified in a future phase. (#37)
#[derive(Debug, Clone)]
pub struct MachFunction {
    pub name: String,
    /// All instructions, indexed by InstId.
    pub insts: Vec<MachInst>,
    /// All blocks, indexed by BlockId.
    pub blocks: Vec<MachBlock>,
    /// Block ordering (RPO or linear).
    pub block_order: Vec<BlockId>,
    /// Entry block.
    pub entry_block: BlockId,
    /// Next available VReg id.
    pub next_vreg: u32,
    /// Next available stack slot id.
    pub next_stack_slot: u32,
    /// Stack slots allocated (for spills and locals).
    pub stack_slots: HashMap<StackSlotId, StackSlot>,
}

impl MachFunction {
    /// Allocate a new stack slot for a spill.
    pub fn alloc_stack_slot(&mut self, size: u32, align: u32) -> StackSlotId {
        let id = StackSlotId(self.next_stack_slot);
        self.next_stack_slot += 1;
        self.stack_slots.insert(id, StackSlot { size, align });
        id
    }

    /// Allocate a new virtual register.
    pub fn alloc_vreg(&mut self, class: RegClass) -> VReg {
        let id = self.next_vreg;
        self.next_vreg += 1;
        VReg { id, class }
    }
}
