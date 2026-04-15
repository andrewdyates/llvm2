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
    /// Symbol reference (for relocations: function calls, globals, TLS).
    /// Carries the symbol name through the pipeline so the relocation
    /// collector can emit proper linker entries.
    Symbol(String),
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

    /// Returns true if this operand is a symbol reference.
    pub fn is_symbol(&self) -> bool {
        matches!(self, Self::Symbol(_))
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

    /// Returns the symbol name, if this is a Symbol operand.
    pub fn as_symbol(&self) -> Option<&str> {
        match self {
            Self::Symbol(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regs::{PReg, RegClass, VReg, X0, X1};
    use crate::types::{BlockId, FrameIdx, StackSlotId};

    // ---- Construction tests ----

    #[test]
    fn construct_vreg_operand() {
        let v = VReg::new(0, RegClass::Gpr64);
        let op = MachOperand::VReg(v);
        assert!(op.is_vreg());
        assert!(!op.is_preg());
        assert!(!op.is_imm());
        assert!(!op.is_mem());
        assert!(!op.is_block());
    }

    #[test]
    fn construct_preg_operand() {
        let op = MachOperand::PReg(X0);
        assert!(op.is_preg());
        assert!(!op.is_vreg());
        assert!(!op.is_imm());
    }

    #[test]
    fn construct_imm_operand() {
        let op = MachOperand::Imm(42);
        assert!(op.is_imm());
        assert!(!op.is_vreg());
        assert!(!op.is_preg());
    }

    #[test]
    fn construct_imm_negative() {
        let op = MachOperand::Imm(-100);
        assert!(op.is_imm());
        assert_eq!(op.as_imm(), Some(-100));
    }

    #[test]
    fn construct_imm_extremes() {
        let op_max = MachOperand::Imm(i64::MAX);
        assert_eq!(op_max.as_imm(), Some(i64::MAX));

        let op_min = MachOperand::Imm(i64::MIN);
        assert_eq!(op_min.as_imm(), Some(i64::MIN));
    }

    #[test]
    fn construct_fimm_operand() {
        let op = MachOperand::FImm(2.78);
        assert!(!op.is_vreg());
        assert!(!op.is_preg());
        assert!(!op.is_imm());
    }

    #[test]
    fn construct_block_operand() {
        let op = MachOperand::Block(BlockId(5));
        assert!(op.is_block());
        assert!(!op.is_vreg());
        assert!(!op.is_mem());
    }

    #[test]
    fn construct_stack_slot_operand() {
        let op = MachOperand::StackSlot(StackSlotId(0));
        assert!(!op.is_vreg());
        assert!(!op.is_imm());
        assert!(!op.is_mem());
    }

    #[test]
    fn construct_frame_index_operand() {
        let op = MachOperand::FrameIndex(FrameIdx(-8));
        assert!(!op.is_vreg());
        assert!(!op.is_imm());
    }

    #[test]
    fn construct_memop() {
        let op = MachOperand::MemOp { base: X0, offset: 16 };
        assert!(op.is_mem());
        assert!(!op.is_vreg());
        assert!(!op.is_preg());
        assert!(!op.is_imm());
    }

    #[test]
    fn construct_memop_negative_offset() {
        let op = MachOperand::MemOp { base: X1, offset: -256 };
        assert!(op.is_mem());
    }

    #[test]
    fn construct_special_operand() {
        let op = MachOperand::Special(SpecialReg::SP);
        assert!(!op.is_vreg());
        assert!(!op.is_preg());
    }

    // ---- Accessor tests ----

    #[test]
    fn as_vreg_some() {
        let v = VReg::new(7, RegClass::Fpr64);
        let op = MachOperand::VReg(v);
        assert_eq!(op.as_vreg(), Some(v));
    }

    #[test]
    fn as_vreg_none() {
        assert_eq!(MachOperand::Imm(0).as_vreg(), None);
        assert_eq!(MachOperand::PReg(X0).as_vreg(), None);
    }

    #[test]
    fn as_preg_some() {
        let op = MachOperand::PReg(X0);
        assert_eq!(op.as_preg(), Some(X0));
    }

    #[test]
    fn as_preg_none() {
        assert_eq!(MachOperand::Imm(0).as_preg(), None);
        assert_eq!(MachOperand::VReg(VReg::new(0, RegClass::Gpr64)).as_preg(), None);
    }

    #[test]
    fn as_imm_some() {
        let op = MachOperand::Imm(999);
        assert_eq!(op.as_imm(), Some(999));
    }

    #[test]
    fn as_imm_none() {
        assert_eq!(MachOperand::PReg(X0).as_imm(), None);
        assert_eq!(MachOperand::FImm(1.0).as_imm(), None);
    }

    // ---- Equality tests ----

    #[test]
    fn operand_equality_vreg() {
        let a = MachOperand::VReg(VReg::new(0, RegClass::Gpr64));
        let b = MachOperand::VReg(VReg::new(0, RegClass::Gpr64));
        let c = MachOperand::VReg(VReg::new(1, RegClass::Gpr64));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn operand_equality_preg() {
        assert_eq!(MachOperand::PReg(X0), MachOperand::PReg(X0));
        assert_ne!(MachOperand::PReg(X0), MachOperand::PReg(X1));
    }

    #[test]
    fn operand_equality_imm() {
        assert_eq!(MachOperand::Imm(42), MachOperand::Imm(42));
        assert_ne!(MachOperand::Imm(42), MachOperand::Imm(43));
    }

    #[test]
    fn operand_equality_memop() {
        let a = MachOperand::MemOp { base: X0, offset: 8 };
        let b = MachOperand::MemOp { base: X0, offset: 8 };
        let c = MachOperand::MemOp { base: X0, offset: 16 };
        let d = MachOperand::MemOp { base: X1, offset: 8 };
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }

    #[test]
    fn operand_equality_different_variants() {
        // Different variants with "similar" values should not be equal
        assert_ne!(MachOperand::Imm(0), MachOperand::PReg(PReg::new(0)));
        assert_ne!(
            MachOperand::VReg(VReg::new(0, RegClass::Gpr64)),
            MachOperand::PReg(X0)
        );
    }

    #[test]
    fn operand_equality_block() {
        assert_eq!(MachOperand::Block(BlockId(0)), MachOperand::Block(BlockId(0)));
        assert_ne!(MachOperand::Block(BlockId(0)), MachOperand::Block(BlockId(1)));
    }

    #[test]
    fn operand_equality_special() {
        assert_eq!(
            MachOperand::Special(SpecialReg::SP),
            MachOperand::Special(SpecialReg::SP)
        );
        assert_ne!(
            MachOperand::Special(SpecialReg::SP),
            MachOperand::Special(SpecialReg::XZR)
        );
    }

    // ---- Clone test ----

    #[test]
    fn operand_clone() {
        let op = MachOperand::MemOp { base: X0, offset: 64 };
        let op2 = op.clone();
        assert_eq!(op, op2);

        let op3 = MachOperand::VReg(VReg::new(5, RegClass::Gpr64));
        let op4 = op3.clone();
        assert_eq!(op3, op4);
    }

    // ---- Debug test ----

    #[test]
    fn operand_debug_doesnt_panic() {
        let ops = [
            MachOperand::VReg(VReg::new(0, RegClass::Gpr64)),
            MachOperand::PReg(X0),
            MachOperand::Imm(42),
            MachOperand::FImm(2.78),
            MachOperand::Block(BlockId(0)),
            MachOperand::StackSlot(StackSlotId(0)),
            MachOperand::FrameIndex(FrameIdx(-8)),
            MachOperand::MemOp { base: X0, offset: 16 },
            MachOperand::Special(SpecialReg::SP),
            MachOperand::Symbol("_my_func".to_string()),
        ];
        for op in &ops {
            let _ = format!("{:?}", op);
        }
    }

    // ---- Symbol operand tests ----

    #[test]
    fn construct_symbol_operand() {
        let op = MachOperand::Symbol("_printf".to_string());
        assert!(op.is_symbol());
        assert!(!op.is_vreg());
        assert!(!op.is_preg());
        assert!(!op.is_imm());
        assert!(!op.is_mem());
        assert!(!op.is_block());
    }

    #[test]
    fn as_symbol_some() {
        let op = MachOperand::Symbol("_my_global".to_string());
        assert_eq!(op.as_symbol(), Some("_my_global"));
    }

    #[test]
    fn as_symbol_none() {
        assert_eq!(MachOperand::Imm(0).as_symbol(), None);
        assert_eq!(MachOperand::PReg(X0).as_symbol(), None);
    }

    #[test]
    fn operand_equality_symbol() {
        assert_eq!(
            MachOperand::Symbol("foo".to_string()),
            MachOperand::Symbol("foo".to_string())
        );
        assert_ne!(
            MachOperand::Symbol("foo".to_string()),
            MachOperand::Symbol("bar".to_string())
        );
        // Symbol is distinct from Imm even though symbols can carry Imm(0) placeholders
        assert_ne!(
            MachOperand::Symbol("foo".to_string()),
            MachOperand::Imm(0)
        );
    }
}
