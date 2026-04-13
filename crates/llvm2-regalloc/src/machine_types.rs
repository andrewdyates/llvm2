// llvm2-regalloc/machine_types.rs - Machine-level types for register allocation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Machine-level type definitions used by the register allocator.
//!
//! These types represent the machine-level IR that register allocation
//! operates on. When `llvm2-ir` lands as the shared machine model crate,
//! these types should be re-exported from there.
//!
//! Reference: `~/llvm-project-ref/llvm/include/llvm/CodeGen/MachineInstr.h`

use std::collections::HashMap;
use std::fmt;

/// Register class — determines which physical registers a virtual register
/// can be allocated to.
///
/// Reference: designs/2026-04-12-aarch64-backend.md
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegClass {
    /// 32-bit general purpose (W0-W30)
    Gpr32,
    /// 64-bit general purpose (X0-X30)
    Gpr64,
    /// 32-bit floating point (S0-S31)
    Fpr32,
    /// 64-bit floating point (D0-D31)
    Fpr64,
    /// 128-bit SIMD (V0-V31)
    Vec128,
}

/// Virtual register — SSA value that needs a physical register assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VReg {
    pub id: u32,
    pub class: RegClass,
}

impl fmt::Display for VReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.id)
    }
}

/// Physical register — an actual hardware register.
///
/// Encoding: 0-30 = GPR (X0-X30), 32-63 = FPR (V0-V31).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PReg(pub u8);

impl PReg {
    /// Returns true if this is a GPR.
    pub fn is_gpr(self) -> bool {
        self.0 <= 30
    }

    /// Returns true if this is an FPR/SIMD register.
    pub fn is_fpr(self) -> bool {
        self.0 >= 32 && self.0 <= 63
    }

    /// Returns the register number within its class.
    pub fn hw_index(self) -> u8 {
        if self.is_fpr() {
            self.0 - 32
        } else {
            self.0
        }
    }
}

impl fmt::Display for PReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_fpr() {
            write!(f, "v{}", self.hw_index())
        } else {
            write!(f, "x{}", self.hw_index())
        }
    }
}

/// Stack slot identifier for spilled values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StackSlotId(pub u32);

/// Block identifier in the machine function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

/// Instruction identifier (index into the function's instruction list).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstId(pub u32);

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

/// A machine instruction.
///
/// Generic container with target-specific opcode (stored as u16 for now;
/// will be an enum when `llvm2-ir` provides target definitions).
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

/// A machine function — the unit of register allocation.
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

/// A stack slot for spilled values.
#[derive(Debug, Clone)]
pub struct StackSlot {
    pub size: u32,
    pub align: u32,
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
