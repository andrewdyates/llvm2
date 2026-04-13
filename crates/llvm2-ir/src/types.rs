// llvm2-ir - Shared machine IR model
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Typed index wrappers for arena-based storage.
//!
//! All IR storage uses Vec + index patterns. These newtypes provide
//! type safety so you can't accidentally use an InstId where a BlockId
//! is expected.

/// Index into the instruction arena (Vec<MachInst>).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstId(pub u32);

/// Index into the block arena (Vec<MachBlock>).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

/// Virtual register identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VRegId(pub u32);

/// Index into the stack slot arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StackSlotId(pub u32);

/// Frame index — signed offset for stack frame layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FrameIdx(pub i32);

// Display implementations for readable output.

impl core::fmt::Display for InstId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "inst{}", self.0)
    }
}

impl core::fmt::Display for BlockId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

impl core::fmt::Display for VRegId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl core::fmt::Display for StackSlotId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ss{}", self.0)
    }
}

impl core::fmt::Display for FrameIdx {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "fi{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- InstId tests ----

    #[test]
    fn inst_id_creation() {
        let id = InstId(0);
        assert_eq!(id.0, 0);
        let id = InstId(42);
        assert_eq!(id.0, 42);
        let id = InstId(u32::MAX);
        assert_eq!(id.0, u32::MAX);
    }

    #[test]
    fn inst_id_display() {
        assert_eq!(format!("{}", InstId(0)), "inst0");
        assert_eq!(format!("{}", InstId(42)), "inst42");
        assert_eq!(format!("{}", InstId(999)), "inst999");
    }

    #[test]
    fn inst_id_equality() {
        assert_eq!(InstId(0), InstId(0));
        assert_ne!(InstId(0), InstId(1));
    }

    #[test]
    fn inst_id_ordering() {
        assert!(InstId(0) < InstId(1));
        assert!(InstId(1) > InstId(0));
        assert!(InstId(5) <= InstId(5));
        assert!(InstId(5) >= InstId(5));
        assert!(InstId(3) <= InstId(4));
    }

    #[test]
    fn inst_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(InstId(0));
        set.insert(InstId(0)); // duplicate
        set.insert(InstId(1));
        assert_eq!(set.len(), 2);
    }

    // ---- BlockId tests ----

    #[test]
    fn block_id_creation() {
        let id = BlockId(0);
        assert_eq!(id.0, 0);
        let id = BlockId(100);
        assert_eq!(id.0, 100);
    }

    #[test]
    fn block_id_display() {
        assert_eq!(format!("{}", BlockId(0)), "bb0");
        assert_eq!(format!("{}", BlockId(7)), "bb7");
        assert_eq!(format!("{}", BlockId(255)), "bb255");
    }

    #[test]
    fn block_id_equality_and_ordering() {
        assert_eq!(BlockId(0), BlockId(0));
        assert_ne!(BlockId(0), BlockId(1));
        assert!(BlockId(0) < BlockId(1));
        assert!(BlockId(10) > BlockId(5));
    }

    // ---- VRegId tests ----

    #[test]
    fn vreg_id_creation() {
        let id = VRegId(0);
        assert_eq!(id.0, 0);
    }

    #[test]
    fn vreg_id_display() {
        assert_eq!(format!("{}", VRegId(0)), "v0");
        assert_eq!(format!("{}", VRegId(99)), "v99");
    }

    #[test]
    fn vreg_id_equality_and_ordering() {
        assert_eq!(VRegId(5), VRegId(5));
        assert_ne!(VRegId(5), VRegId(6));
        assert!(VRegId(0) < VRegId(1));
    }

    // ---- StackSlotId tests ----

    #[test]
    fn stack_slot_id_creation() {
        let id = StackSlotId(0);
        assert_eq!(id.0, 0);
    }

    #[test]
    fn stack_slot_id_display() {
        assert_eq!(format!("{}", StackSlotId(0)), "ss0");
        assert_eq!(format!("{}", StackSlotId(3)), "ss3");
    }

    #[test]
    fn stack_slot_id_equality_and_ordering() {
        assert_eq!(StackSlotId(1), StackSlotId(1));
        assert_ne!(StackSlotId(1), StackSlotId(2));
        assert!(StackSlotId(0) < StackSlotId(1));
    }

    // ---- FrameIdx tests ----

    #[test]
    fn frame_idx_creation() {
        let fi = FrameIdx(0);
        assert_eq!(fi.0, 0);
        let fi = FrameIdx(-1);
        assert_eq!(fi.0, -1);
        let fi = FrameIdx(i32::MAX);
        assert_eq!(fi.0, i32::MAX);
        let fi = FrameIdx(i32::MIN);
        assert_eq!(fi.0, i32::MIN);
    }

    #[test]
    fn frame_idx_display() {
        assert_eq!(format!("{}", FrameIdx(0)), "fi0");
        assert_eq!(format!("{}", FrameIdx(-1)), "fi-1");
        assert_eq!(format!("{}", FrameIdx(42)), "fi42");
    }

    #[test]
    fn frame_idx_equality_and_ordering() {
        assert_eq!(FrameIdx(0), FrameIdx(0));
        assert_ne!(FrameIdx(0), FrameIdx(1));
        assert!(FrameIdx(-1) < FrameIdx(0));
        assert!(FrameIdx(0) < FrameIdx(1));
    }

    // ---- Cross-type safety tests ----

    #[test]
    fn type_safety_different_id_types() {
        // Verify that different ID types with same inner value are distinct types
        // (compile-time type safety, but we verify they can coexist)
        let inst = InstId(5);
        let block = BlockId(5);
        let vreg = VRegId(5);
        let ss = StackSlotId(5);
        let fi = FrameIdx(5);

        // All have inner value 5 but are different types
        assert_eq!(inst.0, 5u32);
        assert_eq!(block.0, 5u32);
        assert_eq!(vreg.0, 5u32);
        assert_eq!(ss.0, 5u32);
        assert_eq!(fi.0, 5i32);
    }

    #[test]
    fn copy_clone_all_types() {
        let inst = InstId(1);
        let inst2 = inst; // Copy
        let inst3 = inst.clone(); // Clone
        assert_eq!(inst, inst2);
        assert_eq!(inst, inst3);

        let block = BlockId(1);
        let block2 = block;
        assert_eq!(block, block2);

        let vreg = VRegId(1);
        let vreg2 = vreg;
        assert_eq!(vreg, vreg2);

        let ss = StackSlotId(1);
        let ss2 = ss;
        assert_eq!(ss, ss2);

        let fi = FrameIdx(1);
        let fi2 = fi;
        assert_eq!(fi, fi2);
    }

    #[test]
    fn debug_impls() {
        // Ensure Debug is derived and doesn't panic
        let _ = format!("{:?}", InstId(0));
        let _ = format!("{:?}", BlockId(0));
        let _ = format!("{:?}", VRegId(0));
        let _ = format!("{:?}", StackSlotId(0));
        let _ = format!("{:?}", FrameIdx(0));
    }
}
