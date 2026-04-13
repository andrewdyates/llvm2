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

            // Everything else: pure computation, no flags
            _ => InstFlags::EMPTY,
        }
    }

    /// Returns true if this is a pseudo-instruction with no hardware encoding.
    pub fn is_pseudo(self) -> bool {
        matches!(self, Self::Phi | Self::StackAlloc | Self::Nop)
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
// MachInst
// ---------------------------------------------------------------------------

/// A single machine instruction.
///
/// Operands are stored inline in a Vec. Implicit defs/uses are static slices
/// (e.g., call instructions implicitly clobber caller-saved registers).
#[derive(Debug, Clone)]
pub struct MachInst {
    pub opcode: AArch64Opcode,
    pub operands: Vec<MachOperand>,
    pub implicit_defs: &'static [PReg],
    pub implicit_uses: &'static [PReg],
    pub flags: InstFlags,
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
        }
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
