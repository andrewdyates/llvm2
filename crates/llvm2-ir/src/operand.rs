// llvm2-ir - Shared machine IR model
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Machine operand types for AArch64 instructions.

use crate::regs::{PReg, SpecialReg, VReg};
use crate::types::{BlockId, FrameIdx, StackSlotId};

/// A machine-level operand. Instructions reference operands by position
/// in their operand list.
#[derive(Debug, Clone, PartialEq)]
pub enum MachOperand {
    /// Virtual register (pre-regalloc).
    VReg(VReg),
    /// Physical register (post-regalloc or fixed constraints).
    PReg(PReg),
    /// Immediate integer value.
    Imm(i64),
    /// Immediate floating-point value.
    FImm(f64),
    /// Basic block target (for branches).
    Block(BlockId),
    /// Stack slot reference.
    StackSlot(StackSlotId),
    /// Frame index (for stack frame layout).
    FrameIndex(FrameIdx),
    /// Memory operand: base register + signed offset.
    MemOp {
        base: PReg,
        offset: i64,
    },
    /// Special register (SP, XZR, WZR).
    Special(SpecialReg),
}

impl MachOperand {
    /// Returns true if this operand is a virtual register.
    pub fn is_vreg(&self) -> bool {
        matches!(self, Self::VReg(_))
    }

    /// Returns true if this operand is a physical register.
    pub fn is_preg(&self) -> bool {
        matches!(self, Self::PReg(_))
    }

    /// Returns true if this operand is an immediate.
    pub fn is_imm(&self) -> bool {
        matches!(self, Self::Imm(_))
    }

    /// Returns true if this operand is a memory operand.
    pub fn is_mem(&self) -> bool {
        matches!(self, Self::MemOp { .. })
    }

    /// Returns true if this operand is a block target.
    pub fn is_block(&self) -> bool {
        matches!(self, Self::Block(_))
    }

    /// Returns the virtual register, if this is a VReg operand.
    pub fn as_vreg(&self) -> Option<VReg> {
        match self {
            Self::VReg(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the physical register, if this is a PReg operand.
    pub fn as_preg(&self) -> Option<PReg> {
        match self {
            Self::PReg(p) => Some(*p),
            _ => None,
        }
    }

    /// Returns the immediate value, if this is an Imm operand.
    pub fn as_imm(&self) -> Option<i64> {
        match self {
            Self::Imm(v) => Some(*v),
            _ => None,
        }
    }
}
