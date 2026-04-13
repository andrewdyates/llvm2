// llvm2-regalloc/machine_types.rs - Machine-level types for register allocation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Machine-level type definitions used by the register allocator.
//!
//! Primitive types (VReg, PReg, RegClass, BlockId, InstId, StackSlotId) are
//! imported from `llvm2-ir`, the canonical source of truth for machine IR types.
//!
//! The compound types (MachInst, MachBlock, MachFunction, MachOperand) have
//! regalloc-specific structure: MachInst separates defs from uses for
//! efficient liveness computation, and MachFunction uses arena-based storage
//! with stack slot tracking. These are intentionally different from the
//! llvm2-ir versions and will be unified in a future phase.
//!
//! Reference: `~/llvm-project-ref/llvm/include/llvm/CodeGen/MachineInstr.h`

use std::collections::HashMap;

// Re-export canonical primitive types from llvm2-ir.
pub use llvm2_ir::regs::{PReg, RegClass, VReg};
pub use llvm2_ir::types::{BlockId, InstId, StackSlotId};

/// Operand of a machine instruction.
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
/// Uses the same flag encoding as `llvm2_ir::InstFlags` with additional
/// regalloc-specific flags (IS_PHI).
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

/// A machine basic block.
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
#[derive(Debug, Clone)]
pub struct StackSlot {
    pub size: u32,
    pub align: u32,
}

/// A machine function -- the unit of register allocation.
///
/// Arena-based storage: instructions are stored in a flat Vec indexed by
/// InstId, blocks in a Vec indexed by BlockId.
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
