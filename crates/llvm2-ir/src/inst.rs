// llvm2-ir - Shared machine IR model
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Machine instruction types: AArch64Opcode, InstFlags, MachInst.

use crate::operand::MachOperand;
use crate::regs::PReg;

// ---------------------------------------------------------------------------
// AArch64Opcode
// ---------------------------------------------------------------------------

/// AArch64 instruction opcodes.
///
/// Naming convention: `<mnemonic><operand_kinds>` where RR = register-register,
/// RI = register-immediate. Pseudo-instructions have no hardware encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AArch64Opcode {
    // -- Arithmetic --
    AddRR,
    AddRI,
    SubRR,
    SubRI,
    MulRR,
    SDiv,
    UDiv,
    Neg,

    // -- Logical --
    AndRR,
    OrrRR,
    EorRR,

    // -- Shifts --
    LslRR,
    LsrRR,
    AsrRR,
    LslRI,
    LsrRI,
    AsrRI,

    // -- Compare --
    CmpRR,
    CmpRI,
    Tst,

    // -- Move --
    MovR,
    MovI,
    Movz,
    Movk,

    // -- Memory --
    LdrRI,
    StrRI,
    LdrLiteral,
    LdpRI,
    StpRI,

    // -- Branch --
    B,
    BCond,
    Cbz,
    Cbnz,
    Tbz,
    Tbnz,
    Br,
    Bl,
    Blr,
    Ret,

    // -- Extension --
    Sxtw,
    Uxtw,
    Sxtb,
    Sxth,

    // -- Floating-point --
    FaddRR,
    FsubRR,
    FmulRR,
    FdivRR,
    Fcmp,
    FcvtzsRR,
    ScvtfRR,

    // -- Address --
    Adrp,
    AddPCRel,

    // -- Checked arithmetic (set flags for overflow detection) --
    /// ADDS: add and set flags (used for overflow-checked addition).
    AddsRR,
    /// ADDS immediate: add immediate and set flags.
    AddsRI,
    /// SUBS: subtract and set flags (used for overflow-checked subtraction).
    SubsRR,
    /// SUBS immediate: subtract immediate and set flags.
    SubsRI,

    // -- Trap / panic pseudo-instructions --
    /// Trap on overflow: conditional branch to trap block after ADDS/SUBS.
    /// Operands: [condition_code_imm, Block(trap_target)].
    TrapOverflow,
    /// Trap on bounds check failure: branch to panic if index >= length.
    /// Operands: [Block(panic_target)].
    TrapBoundsCheck,
    /// Trap on null pointer.
    /// Operands: [Block(panic_target)].
    TrapNull,

    // -- Reference counting pseudo-instructions --
    /// Retain (increment reference count). Operands: [ptr].
    Retain,
    /// Release (decrement reference count). Operands: [ptr].
    Release,

    // -- Pseudo-instructions (no hardware encoding) --
    Phi,
    StackAlloc,
    Nop,
}

impl AArch64Opcode {
    /// Returns the default instruction flags for this opcode.
    pub fn default_flags(self) -> InstFlags {
        use AArch64Opcode::*;
        match self {
            // Branches
            B => InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),
            BCond => InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),
            Cbz => InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),
            Cbnz => InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),
            Tbz => InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),
            Tbnz => InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),
            Br => InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),

            // Calls
            Bl => InstFlags::IS_CALL.union(InstFlags::HAS_SIDE_EFFECTS),
            Blr => InstFlags::IS_CALL.union(InstFlags::HAS_SIDE_EFFECTS),

            // Return
            Ret => InstFlags::IS_RETURN.union(InstFlags::IS_TERMINATOR),

            // Memory loads
            LdrRI => InstFlags::READS_MEMORY,
            LdrLiteral => InstFlags::READS_MEMORY,
            LdpRI => InstFlags::READS_MEMORY,

            // Memory stores
            StrRI => InstFlags::WRITES_MEMORY.union(InstFlags::HAS_SIDE_EFFECTS),
            StpRI => InstFlags::WRITES_MEMORY.union(InstFlags::HAS_SIDE_EFFECTS),

            // Pseudo-instructions
            Phi => InstFlags::IS_PSEUDO,
            StackAlloc => InstFlags::IS_PSEUDO.union(InstFlags::HAS_SIDE_EFFECTS),
            Nop => InstFlags::IS_PSEUDO,

            // Compare/test (set condition flags = side effect)
            CmpRR | CmpRI | Tst | Fcmp => InstFlags::HAS_SIDE_EFFECTS,

            // Checked arithmetic: produce a result AND set flags (side effect)
            AddsRR | AddsRI | SubsRR | SubsRI => InstFlags::HAS_SIDE_EFFECTS,

            // Trap pseudo-instructions: conditional branches to panic blocks
            TrapOverflow => InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR).union(InstFlags::HAS_SIDE_EFFECTS),
            TrapBoundsCheck => InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR).union(InstFlags::HAS_SIDE_EFFECTS),
            TrapNull => InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR).union(InstFlags::HAS_SIDE_EFFECTS),

            // Reference counting: side effects (modify refcount in memory)
            Retain => InstFlags::HAS_SIDE_EFFECTS.union(InstFlags::READS_MEMORY).union(InstFlags::WRITES_MEMORY),
            Release => InstFlags::HAS_SIDE_EFFECTS.union(InstFlags::READS_MEMORY).union(InstFlags::WRITES_MEMORY),

            // Everything else: pure computation, no flags
            _ => InstFlags::EMPTY,
        }
    }

    /// Returns true if this is a pseudo-instruction with no hardware encoding.
    pub fn is_pseudo(self) -> bool {
        matches!(
            self,
            Self::Phi
                | Self::StackAlloc
                | Self::Nop
                | Self::TrapOverflow
                | Self::TrapBoundsCheck
                | Self::TrapNull
                | Self::Retain
                | Self::Release
        )
    }

    /// Returns true if this is a phi instruction.
    pub fn is_phi(self) -> bool {
        matches!(self, Self::Phi)
    }
}

// ---------------------------------------------------------------------------
// InstFlags (manual bitflags, no external crate)
// ---------------------------------------------------------------------------

/// Instruction property flags, packed as a u16 bitfield.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstFlags(u16);

impl InstFlags {
    pub const EMPTY: Self = Self(0);
    pub const IS_CALL: Self = Self(0x01);
    pub const IS_BRANCH: Self = Self(0x02);
    pub const IS_RETURN: Self = Self(0x04);
    pub const IS_TERMINATOR: Self = Self(0x08);
    pub const HAS_SIDE_EFFECTS: Self = Self(0x10);
    pub const IS_PSEUDO: Self = Self(0x20);
    pub const READS_MEMORY: Self = Self(0x40);
    pub const WRITES_MEMORY: Self = Self(0x80);
    pub const IS_PHI: Self = Self(0x100);

    /// Returns true if all bits in `other` are set in `self`.
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Set all bits in `other`.
    #[inline]
    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    /// Clear all bits in `other`.
    #[inline]
    pub fn remove(&mut self, other: Self) {
        self.0 &= !other.0;
    }

    /// Union of two flag sets.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Intersection of two flag sets.
    #[inline]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Returns true if no flags are set.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Raw bits.
    #[inline]
    pub const fn bits(self) -> u16 {
        self.0
    }
}

impl Default for InstFlags {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl core::ops::BitOr for InstFlags {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitAnd for InstFlags {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl core::ops::BitOrAssign for InstFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl core::fmt::Debug for InstFlags {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let flags = [
            (Self::IS_CALL, "IS_CALL"),
            (Self::IS_BRANCH, "IS_BRANCH"),
            (Self::IS_RETURN, "IS_RETURN"),
            (Self::IS_TERMINATOR, "IS_TERMINATOR"),
            (Self::HAS_SIDE_EFFECTS, "HAS_SIDE_EFFECTS"),
            (Self::IS_PSEUDO, "IS_PSEUDO"),
            (Self::READS_MEMORY, "READS_MEMORY"),
            (Self::WRITES_MEMORY, "WRITES_MEMORY"),
            (Self::IS_PHI, "IS_PHI"),
        ];
        write!(f, "InstFlags(")?;
        for (flag, name) in &flags {
            if self.contains(*flag) {
                if !first {
                    write!(f, " | ")?;
                }
                write!(f, "{}", name)?;
                first = false;
            }
        }
        if first {
            write!(f, "EMPTY")?;
        }
        write!(f, ")")
    }
}

// ---------------------------------------------------------------------------
// ProofAnnotation
// ---------------------------------------------------------------------------

/// Proof annotations from tMIR that enable optimizations no other compiler can do.
///
/// These annotations represent formally verified preconditions that the tMIR
/// frontend has proven about program values. The LLVM2 backend can consume
/// these proofs to eliminate runtime checks that would otherwise be required.
///
/// Each annotation corresponds to a specific optimization opportunity:
/// - `NoOverflow` → eliminate overflow checks, use unchecked arithmetic
/// - `InBounds` → eliminate array bounds checks
/// - `NotNull` → eliminate null pointer checks
/// - `ValidBorrow` → enable load/store reordering (refined alias analysis)
/// - `PositiveRefCount` → eliminate redundant retain/release pairs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProofAnnotation {
    /// tMIR has proven this arithmetic operation cannot overflow.
    /// Enables: ADDS/SUBS → ADD/SUB, remove TrapOverflow.
    NoOverflow,

    /// tMIR has proven this array access index is within bounds.
    /// Enables: remove CMP+B.HS bounds check guard.
    InBounds,

    /// tMIR has proven this pointer is not null.
    /// Enables: remove CBZ/CBNZ null check guard.
    NotNull,

    /// tMIR has proven this borrow/reference is valid (no aliasing violations).
    /// Enables: load/store reordering past other memory operations.
    ValidBorrow,

    /// tMIR has proven the reference count is positive (object is live).
    /// Enables: eliminate redundant retain/release pairs.
    PositiveRefCount,
}

// ---------------------------------------------------------------------------
// MachInst
// ---------------------------------------------------------------------------

/// A single machine instruction.
///
/// Operands are stored inline in a Vec. Implicit defs/uses are static slices
/// (e.g., call instructions implicitly clobber caller-saved registers).
///
/// The `proof` field carries optional tMIR proof annotations that enable
/// proof-consuming optimizations unique to LLVM2.
#[derive(Debug, Clone)]
pub struct MachInst {
    pub opcode: AArch64Opcode,
    pub operands: Vec<MachOperand>,
    pub implicit_defs: &'static [PReg],
    pub implicit_uses: &'static [PReg],
    pub flags: InstFlags,
    /// Optional proof annotation from tMIR. When present, indicates that
    /// the tMIR frontend has formally verified a property about this
    /// instruction's operands, enabling proof-consuming optimizations.
    pub proof: Option<ProofAnnotation>,
}

impl MachInst {
    /// Create a new instruction with default flags for the opcode.
    pub fn new(opcode: AArch64Opcode, operands: Vec<MachOperand>) -> Self {
        Self {
            flags: opcode.default_flags(),
            opcode,
            operands,
            implicit_defs: &[],
            implicit_uses: &[],
            proof: None,
        }
    }

    /// Create a new instruction with explicit flags.
    pub fn with_flags(
        opcode: AArch64Opcode,
        operands: Vec<MachOperand>,
        flags: InstFlags,
    ) -> Self {
        Self {
            opcode,
            operands,
            implicit_defs: &[],
            implicit_uses: &[],
            flags,
            proof: None,
        }
    }

    /// Attach a proof annotation to this instruction.
    pub fn with_proof(mut self, proof: ProofAnnotation) -> Self {
        self.proof = Some(proof);
        self
    }

    /// Set implicit register definitions (clobbers).
    pub fn with_implicit_defs(mut self, defs: &'static [PReg]) -> Self {
        self.implicit_defs = defs;
        self
    }

    /// Set implicit register uses.
    pub fn with_implicit_uses(mut self, uses: &'static [PReg]) -> Self {
        self.implicit_uses = uses;
        self
    }

    // -- Flag query convenience methods --

    #[inline]
    pub fn is_call(&self) -> bool {
        self.flags.contains(InstFlags::IS_CALL)
    }

    #[inline]
    pub fn is_branch(&self) -> bool {
        self.flags.contains(InstFlags::IS_BRANCH)
    }

    #[inline]
    pub fn is_return(&self) -> bool {
        self.flags.contains(InstFlags::IS_RETURN)
    }

    #[inline]
    pub fn is_terminator(&self) -> bool {
        self.flags.contains(InstFlags::IS_TERMINATOR)
    }

    #[inline]
    pub fn has_side_effects(&self) -> bool {
        self.flags.contains(InstFlags::HAS_SIDE_EFFECTS)
    }

    #[inline]
    pub fn is_pseudo(&self) -> bool {
        self.flags.contains(InstFlags::IS_PSEUDO)
    }

    #[inline]
    pub fn reads_memory(&self) -> bool {
        self.flags.contains(InstFlags::READS_MEMORY)
    }

    #[inline]
    pub fn writes_memory(&self) -> bool {
        self.flags.contains(InstFlags::WRITES_MEMORY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operand::MachOperand;
    use crate::regs::{PReg, RegClass, VReg, X0, X1, X30};
    use crate::types::BlockId;

    // ---- AArch64Opcode flag tests ----

    #[test]
    fn branch_opcodes_have_branch_and_terminator_flags() {
        let branch_ops = [
            AArch64Opcode::B,
            AArch64Opcode::BCond,
            AArch64Opcode::Cbz,
            AArch64Opcode::Cbnz,
            AArch64Opcode::Tbz,
            AArch64Opcode::Tbnz,
            AArch64Opcode::Br,
        ];
        for op in &branch_ops {
            let flags = op.default_flags();
            assert!(
                flags.contains(InstFlags::IS_BRANCH),
                "{:?} should have IS_BRANCH", op
            );
            assert!(
                flags.contains(InstFlags::IS_TERMINATOR),
                "{:?} should have IS_TERMINATOR", op
            );
        }
    }

    #[test]
    fn call_opcodes_have_call_and_side_effect_flags() {
        let call_ops = [AArch64Opcode::Bl, AArch64Opcode::Blr];
        for op in &call_ops {
            let flags = op.default_flags();
            assert!(
                flags.contains(InstFlags::IS_CALL),
                "{:?} should have IS_CALL", op
            );
            assert!(
                flags.contains(InstFlags::HAS_SIDE_EFFECTS),
                "{:?} should have HAS_SIDE_EFFECTS", op
            );
            assert!(
                !flags.contains(InstFlags::IS_BRANCH),
                "{:?} should NOT have IS_BRANCH", op
            );
        }
    }

    #[test]
    fn ret_opcode_has_return_and_terminator_flags() {
        let flags = AArch64Opcode::Ret.default_flags();
        assert!(flags.contains(InstFlags::IS_RETURN));
        assert!(flags.contains(InstFlags::IS_TERMINATOR));
        assert!(!flags.contains(InstFlags::IS_CALL));
        assert!(!flags.contains(InstFlags::IS_BRANCH));
    }

    #[test]
    fn load_opcodes_have_reads_memory() {
        let load_ops = [
            AArch64Opcode::LdrRI,
            AArch64Opcode::LdrLiteral,
            AArch64Opcode::LdpRI,
        ];
        for op in &load_ops {
            let flags = op.default_flags();
            assert!(
                flags.contains(InstFlags::READS_MEMORY),
                "{:?} should have READS_MEMORY", op
            );
            assert!(
                !flags.contains(InstFlags::WRITES_MEMORY),
                "{:?} should NOT have WRITES_MEMORY", op
            );
        }
    }

    #[test]
    fn store_opcodes_have_writes_memory_and_side_effects() {
        let store_ops = [AArch64Opcode::StrRI, AArch64Opcode::StpRI];
        for op in &store_ops {
            let flags = op.default_flags();
            assert!(
                flags.contains(InstFlags::WRITES_MEMORY),
                "{:?} should have WRITES_MEMORY", op
            );
            assert!(
                flags.contains(InstFlags::HAS_SIDE_EFFECTS),
                "{:?} should have HAS_SIDE_EFFECTS", op
            );
        }
    }

    #[test]
    fn pseudo_opcodes_have_pseudo_flag() {
        let pseudo_ops = [
            AArch64Opcode::Phi,
            AArch64Opcode::StackAlloc,
            AArch64Opcode::Nop,
        ];
        for op in &pseudo_ops {
            let flags = op.default_flags();
            assert!(
                flags.contains(InstFlags::IS_PSEUDO),
                "{:?} should have IS_PSEUDO", op
            );
        }
    }

    #[test]
    fn is_pseudo_method() {
        assert!(AArch64Opcode::Phi.is_pseudo());
        assert!(AArch64Opcode::StackAlloc.is_pseudo());
        assert!(AArch64Opcode::Nop.is_pseudo());
        assert!(!AArch64Opcode::AddRR.is_pseudo());
        assert!(!AArch64Opcode::B.is_pseudo());
        assert!(!AArch64Opcode::Ret.is_pseudo());
    }

    #[test]
    fn is_phi_method() {
        assert!(AArch64Opcode::Phi.is_phi());
        assert!(!AArch64Opcode::Nop.is_phi());
        assert!(!AArch64Opcode::AddRR.is_phi());
    }

    #[test]
    fn pure_arithmetic_has_empty_flags() {
        let pure_ops = [
            AArch64Opcode::AddRR,
            AArch64Opcode::AddRI,
            AArch64Opcode::SubRR,
            AArch64Opcode::SubRI,
            AArch64Opcode::MulRR,
            AArch64Opcode::SDiv,
            AArch64Opcode::UDiv,
            AArch64Opcode::Neg,
            AArch64Opcode::AndRR,
            AArch64Opcode::OrrRR,
            AArch64Opcode::EorRR,
            AArch64Opcode::MovR,
            AArch64Opcode::MovI,
        ];
        for op in &pure_ops {
            let flags = op.default_flags();
            assert!(
                flags.is_empty(),
                "{:?} should have EMPTY flags but has {:?}", op, flags
            );
        }
    }

    #[test]
    fn compare_opcodes_have_side_effects() {
        let cmp_ops = [
            AArch64Opcode::CmpRR,
            AArch64Opcode::CmpRI,
            AArch64Opcode::Tst,
            AArch64Opcode::Fcmp,
        ];
        for op in &cmp_ops {
            let flags = op.default_flags();
            assert!(
                flags.contains(InstFlags::HAS_SIDE_EFFECTS),
                "{:?} should have HAS_SIDE_EFFECTS", op
            );
        }
    }

    // ---- InstFlags bitwise operation tests ----

    #[test]
    fn instflags_empty() {
        let f = InstFlags::EMPTY;
        assert!(f.is_empty());
        assert_eq!(f.bits(), 0);
    }

    #[test]
    fn instflags_single_flag() {
        let f = InstFlags::IS_CALL;
        assert!(!f.is_empty());
        assert!(f.contains(InstFlags::IS_CALL));
        assert!(!f.contains(InstFlags::IS_BRANCH));
    }

    #[test]
    fn instflags_union() {
        let f = InstFlags::IS_CALL.union(InstFlags::HAS_SIDE_EFFECTS);
        assert!(f.contains(InstFlags::IS_CALL));
        assert!(f.contains(InstFlags::HAS_SIDE_EFFECTS));
        assert!(!f.contains(InstFlags::IS_BRANCH));
    }

    #[test]
    fn instflags_intersection() {
        let a = InstFlags::IS_CALL.union(InstFlags::HAS_SIDE_EFFECTS);
        let b = InstFlags::IS_CALL.union(InstFlags::IS_BRANCH);
        let c = a.intersection(b);
        assert!(c.contains(InstFlags::IS_CALL));
        assert!(!c.contains(InstFlags::HAS_SIDE_EFFECTS));
        assert!(!c.contains(InstFlags::IS_BRANCH));
    }

    #[test]
    fn instflags_insert() {
        let mut f = InstFlags::EMPTY;
        assert!(f.is_empty());
        f.insert(InstFlags::IS_CALL);
        assert!(f.contains(InstFlags::IS_CALL));
        f.insert(InstFlags::IS_BRANCH);
        assert!(f.contains(InstFlags::IS_CALL));
        assert!(f.contains(InstFlags::IS_BRANCH));
    }

    #[test]
    fn instflags_remove() {
        let mut f = InstFlags::IS_CALL.union(InstFlags::IS_BRANCH);
        f.remove(InstFlags::IS_CALL);
        assert!(!f.contains(InstFlags::IS_CALL));
        assert!(f.contains(InstFlags::IS_BRANCH));
    }

    #[test]
    fn instflags_bitor_operator() {
        let f = InstFlags::IS_CALL | InstFlags::IS_BRANCH;
        assert!(f.contains(InstFlags::IS_CALL));
        assert!(f.contains(InstFlags::IS_BRANCH));
    }

    #[test]
    fn instflags_bitand_operator() {
        let a = InstFlags::IS_CALL | InstFlags::IS_BRANCH;
        let b = InstFlags::IS_CALL | InstFlags::IS_RETURN;
        let c = a & b;
        assert!(c.contains(InstFlags::IS_CALL));
        assert!(!c.contains(InstFlags::IS_BRANCH));
        assert!(!c.contains(InstFlags::IS_RETURN));
    }

    #[test]
    fn instflags_bitor_assign() {
        let mut f = InstFlags::IS_CALL;
        f |= InstFlags::IS_BRANCH;
        assert!(f.contains(InstFlags::IS_CALL));
        assert!(f.contains(InstFlags::IS_BRANCH));
    }

    #[test]
    fn instflags_default_is_empty() {
        let f = InstFlags::default();
        assert!(f.is_empty());
        assert_eq!(f, InstFlags::EMPTY);
    }

    #[test]
    fn instflags_contains_self() {
        let flags = [
            InstFlags::IS_CALL,
            InstFlags::IS_BRANCH,
            InstFlags::IS_RETURN,
            InstFlags::IS_TERMINATOR,
            InstFlags::HAS_SIDE_EFFECTS,
            InstFlags::IS_PSEUDO,
            InstFlags::READS_MEMORY,
            InstFlags::WRITES_MEMORY,
            InstFlags::IS_PHI,
        ];
        for f in &flags {
            assert!(f.contains(*f), "{:?} should contain itself", f);
        }
    }

    #[test]
    fn instflags_empty_contains_nothing() {
        let flags = [
            InstFlags::IS_CALL,
            InstFlags::IS_BRANCH,
            InstFlags::IS_RETURN,
            InstFlags::IS_TERMINATOR,
            InstFlags::HAS_SIDE_EFFECTS,
            InstFlags::IS_PSEUDO,
            InstFlags::READS_MEMORY,
            InstFlags::WRITES_MEMORY,
            InstFlags::IS_PHI,
        ];
        for f in &flags {
            assert!(!InstFlags::EMPTY.contains(*f));
        }
    }

    #[test]
    fn instflags_bit_values_are_distinct() {
        let flags = [
            InstFlags::IS_CALL,
            InstFlags::IS_BRANCH,
            InstFlags::IS_RETURN,
            InstFlags::IS_TERMINATOR,
            InstFlags::HAS_SIDE_EFFECTS,
            InstFlags::IS_PSEUDO,
            InstFlags::READS_MEMORY,
            InstFlags::WRITES_MEMORY,
            InstFlags::IS_PHI,
        ];
        for i in 0..flags.len() {
            for j in (i + 1)..flags.len() {
                assert_ne!(
                    flags[i].bits(), flags[j].bits(),
                    "flags {:?} and {:?} have same bits", flags[i], flags[j]
                );
            }
        }
    }

    #[test]
    fn instflags_debug_empty() {
        let f = InstFlags::EMPTY;
        let s = format!("{:?}", f);
        assert!(s.contains("EMPTY"));
    }

    #[test]
    fn instflags_debug_single() {
        let f = InstFlags::IS_CALL;
        let s = format!("{:?}", f);
        assert!(s.contains("IS_CALL"));
        assert!(!s.contains("IS_BRANCH"));
    }

    #[test]
    fn instflags_debug_multiple() {
        let f = InstFlags::IS_CALL | InstFlags::HAS_SIDE_EFFECTS;
        let s = format!("{:?}", f);
        assert!(s.contains("IS_CALL"));
        assert!(s.contains("HAS_SIDE_EFFECTS"));
    }

    // ---- MachInst construction tests ----

    #[test]
    fn machinst_new_uses_default_flags() {
        let inst = MachInst::new(AArch64Opcode::AddRR, vec![]);
        assert_eq!(inst.opcode, AArch64Opcode::AddRR);
        assert!(inst.flags.is_empty()); // AddRR has empty default flags
        assert!(inst.operands.is_empty());
        assert!(inst.implicit_defs.is_empty());
        assert!(inst.implicit_uses.is_empty());
    }

    #[test]
    fn machinst_new_branch_has_correct_flags() {
        let inst = MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(BlockId(1))],
        );
        assert!(inst.is_branch());
        assert!(inst.is_terminator());
        assert!(!inst.is_call());
        assert!(!inst.is_return());
    }

    #[test]
    fn machinst_new_ret_has_correct_flags() {
        let inst = MachInst::new(AArch64Opcode::Ret, vec![]);
        assert!(inst.is_return());
        assert!(inst.is_terminator());
        assert!(!inst.is_branch());
        assert!(!inst.is_call());
    }

    #[test]
    fn machinst_with_flags_overrides_defaults() {
        let inst = MachInst::with_flags(
            AArch64Opcode::AddRR,
            vec![],
            InstFlags::HAS_SIDE_EFFECTS,
        );
        assert!(inst.has_side_effects());
        assert!(!inst.is_call());
    }

    #[test]
    fn machinst_with_implicit_defs() {
        static DEFS: &[PReg] = &[X0, X1];
        let inst = MachInst::new(AArch64Opcode::Bl, vec![])
            .with_implicit_defs(DEFS);
        assert_eq!(inst.implicit_defs, DEFS);
        assert!(inst.implicit_uses.is_empty());
    }

    #[test]
    fn machinst_with_implicit_uses() {
        static USES: &[PReg] = &[X0];
        let inst = MachInst::new(AArch64Opcode::Ret, vec![])
            .with_implicit_uses(USES);
        assert_eq!(inst.implicit_uses, USES);
        assert!(inst.implicit_defs.is_empty());
    }

    #[test]
    fn machinst_builder_chain() {
        static DEFS: &[PReg] = &[X0, X1];
        static USES: &[PReg] = &[X30];
        let inst = MachInst::new(AArch64Opcode::Blr, vec![MachOperand::PReg(X30)])
            .with_implicit_defs(DEFS)
            .with_implicit_uses(USES);
        assert!(inst.is_call());
        assert!(inst.has_side_effects());
        assert_eq!(inst.implicit_defs.len(), 2);
        assert_eq!(inst.implicit_uses.len(), 1);
        assert_eq!(inst.operands.len(), 1);
    }

    #[test]
    fn machinst_with_operands() {
        let v0 = VReg::new(0, RegClass::Gpr64);
        let v1 = VReg::new(1, RegClass::Gpr64);
        let inst = MachInst::new(
            AArch64Opcode::AddRR,
            vec![
                MachOperand::VReg(v0),
                MachOperand::VReg(v1),
                MachOperand::VReg(v0),
            ],
        );
        assert_eq!(inst.operands.len(), 3);
        assert_eq!(inst.operands[0].as_vreg(), Some(v0));
        assert_eq!(inst.operands[1].as_vreg(), Some(v1));
    }

    // ---- MachInst flag query convenience methods ----

    #[test]
    fn machinst_flag_queries_match_flags() {
        let inst_call = MachInst::new(AArch64Opcode::Bl, vec![]);
        assert!(inst_call.is_call());
        assert!(inst_call.has_side_effects());
        assert!(!inst_call.is_branch());
        assert!(!inst_call.is_return());
        assert!(!inst_call.is_terminator());
        assert!(!inst_call.is_pseudo());
        assert!(!inst_call.reads_memory());
        assert!(!inst_call.writes_memory());

        let inst_load = MachInst::new(AArch64Opcode::LdrRI, vec![]);
        assert!(inst_load.reads_memory());
        assert!(!inst_load.writes_memory());

        let inst_store = MachInst::new(AArch64Opcode::StrRI, vec![]);
        assert!(inst_store.writes_memory());
        assert!(inst_store.has_side_effects());
        assert!(!inst_store.reads_memory());

        let inst_phi = MachInst::new(AArch64Opcode::Phi, vec![]);
        assert!(inst_phi.is_pseudo());
    }

    #[test]
    fn machinst_clone() {
        let inst = MachInst::new(
            AArch64Opcode::AddRR,
            vec![MachOperand::Imm(42)],
        );
        let inst2 = inst.clone();
        assert_eq!(inst2.opcode, inst.opcode);
        assert_eq!(inst2.operands.len(), inst.operands.len());
        assert_eq!(inst2.flags, inst.flags);
    }
}
