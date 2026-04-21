// llvm2-ir - Multi-target opcode categorization
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Target-independent opcode categories and target abstraction trait.
//!
//! Optimization passes that need to reason about instruction semantics
//! (constant folding, peephole, CSE key hashing) can use [`OpcodeCategory`]
//! instead of matching target-specific opcode enums. Each target provides
//! a `categorize()` method mapping its opcodes to categories.
//!
//! # Design
//!
//! The category enum covers the *semantic* operations that optimization
//! passes care about (add, sub, shift-left, move-register, etc.). Opcodes
//! that don't map to any generic optimization pattern get [`OpcodeCategory::Other`].
//!
//! The [`TargetInfo`] trait collects per-target queries that optimization
//! passes need: categorization, value production, flag access, commutativity,
//! and canonical opcode accessors for mov/shl replacements in peephole.

use crate::inst::AArch64Opcode;
use crate::x86_64_ops::X86Opcode;

// ---------------------------------------------------------------------------
// OpcodeCategory
// ---------------------------------------------------------------------------

/// Target-independent opcode category for use by optimization passes.
///
/// Passes can match on categories to apply generic transformations
/// (e.g., "add with zero immediate is identity") without knowing the
/// target-specific opcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpcodeCategory {
    // -- Arithmetic --
    /// Register-register addition.
    AddRR,
    /// Register-immediate addition.
    AddRI,
    /// Register-register subtraction.
    SubRR,
    /// Register-immediate subtraction.
    SubRI,
    /// Register-register multiplication.
    MulRR,
    /// Unary negate.
    Neg,

    // -- Logical --
    /// Bitwise AND register-register.
    AndRR,
    /// Bitwise AND register-immediate.
    AndRI,
    /// Bitwise OR register-register.
    OrRR,
    /// Bitwise OR register-immediate.
    OrRI,
    /// Bitwise XOR register-register.
    XorRR,
    /// Bitwise XOR register-immediate.
    XorRI,

    // -- Shifts --
    /// Shift left by register.
    ShlRR,
    /// Shift left by immediate.
    ShlRI,
    /// Logical shift right by register.
    ShrRR,
    /// Logical shift right by immediate.
    ShrRI,
    /// Arithmetic shift right by register.
    SarRR,
    /// Arithmetic shift right by immediate.
    SarRI,

    // -- Move --
    /// Register-to-register move (copy).
    MovRR,
    /// Immediate-to-register move.
    MovRI,

    // -- Compare --
    /// Register-register compare (sets flags, no value produced).
    CmpRR,
    /// Register-immediate compare.
    CmpRI,

    // -- Control flow --
    /// No-operation.
    Nop,
    /// Function return.
    Ret,
    /// Function call.
    Call,
    /// Unconditional branch.
    Branch,
    /// Conditional branch.
    CondBranch,

    // -- Memory --
    /// Load from memory.
    Load,
    /// Store to memory.
    Store,

    // -- SSA --
    /// Phi node (SSA merge point).
    Phi,

    // -- Catch-all --
    /// Target-specific opcode with no generic category.
    Other,
}

impl OpcodeCategory {
    /// Returns true if this category is a binary arithmetic operation
    /// that optimization passes commonly fold or simplify.
    #[inline]
    pub fn is_arithmetic(self) -> bool {
        matches!(
            self,
            Self::AddRR | Self::AddRI | Self::SubRR | Self::SubRI | Self::MulRR | Self::Neg
        )
    }

    /// Returns true if this category is a binary logical operation.
    #[inline]
    pub fn is_logical(self) -> bool {
        matches!(
            self,
            Self::AndRR | Self::AndRI | Self::OrRR | Self::OrRI | Self::XorRR | Self::XorRI
        )
    }

    /// Returns true if this category is a shift operation.
    #[inline]
    pub fn is_shift(self) -> bool {
        matches!(
            self,
            Self::ShlRR | Self::ShlRI | Self::ShrRR | Self::ShrRI | Self::SarRR | Self::SarRI
        )
    }

    /// Returns true if this category is a move (register or immediate).
    #[inline]
    pub fn is_move(self) -> bool {
        matches!(self, Self::MovRR | Self::MovRI)
    }

    /// Returns true if this is a register-immediate form (has an immediate
    /// operand that optimization passes can inspect).
    #[inline]
    pub fn is_reg_imm(self) -> bool {
        matches!(
            self,
            Self::AddRI
                | Self::SubRI
                | Self::AndRI
                | Self::OrRI
                | Self::XorRI
                | Self::ShlRI
                | Self::ShrRI
                | Self::SarRI
                | Self::CmpRI
                | Self::MovRI
        )
    }

    /// Returns true if this is a register-register form where both source
    /// operands being the same register enables special simplifications
    /// (e.g., sub x,x = 0, or x,x = x, xor x,x = 0).
    #[inline]
    pub fn is_reg_reg_binary(self) -> bool {
        matches!(
            self,
            Self::AddRR
                | Self::SubRR
                | Self::MulRR
                | Self::AndRR
                | Self::OrRR
                | Self::XorRR
                | Self::ShlRR
                | Self::ShrRR
                | Self::SarRR
        )
    }
}

// ---------------------------------------------------------------------------
// AArch64Opcode::categorize
// ---------------------------------------------------------------------------

impl AArch64Opcode {
    /// Classify this AArch64 opcode into a target-independent category.
    pub fn categorize(self) -> OpcodeCategory {
        use AArch64Opcode::*;
        match self {
            // Arithmetic
            AddRR => OpcodeCategory::AddRR,
            AddRI | AddRIShift12 => OpcodeCategory::AddRI,
            SubRR => OpcodeCategory::SubRR,
            SubRI => OpcodeCategory::SubRI,
            MulRR => OpcodeCategory::MulRR,
            Neg => OpcodeCategory::Neg,

            // Logical (AArch64 uses Orr/Eor naming)
            AndRR => OpcodeCategory::AndRR,
            AndRI => OpcodeCategory::AndRI,
            OrrRR => OpcodeCategory::OrRR,
            OrrRI => OpcodeCategory::OrRI,
            EorRR => OpcodeCategory::XorRR,
            EorRI => OpcodeCategory::XorRI,

            // Shifts (AArch64 uses Lsl/Lsr/Asr naming)
            LslRR => OpcodeCategory::ShlRR,
            LslRI => OpcodeCategory::ShlRI,
            LsrRR => OpcodeCategory::ShrRR,
            LsrRI => OpcodeCategory::ShrRI,
            AsrRR => OpcodeCategory::SarRR,
            AsrRI => OpcodeCategory::SarRI,

            // Moves
            MovR | Copy | MOVWrr | MOVXrr => OpcodeCategory::MovRR,
            MovI | Movz | Movn | MOVZWi | MOVZXi => OpcodeCategory::MovRI,

            // Compare
            CmpRR | CMPWrr | CMPXrr => OpcodeCategory::CmpRR,
            CmpRI | CMPWri | CMPXri => OpcodeCategory::CmpRI,

            // Control flow
            B | Br => OpcodeCategory::Branch,
            BCond | Bcc | Cbz | Cbnz | Tbz | Tbnz => OpcodeCategory::CondBranch,
            Bl | Blr | BL | BLR => OpcodeCategory::Call,
            Ret => OpcodeCategory::Ret,

            // Memory loads
            LdrRI | LdrbRI | LdrhRI | LdrsbRI | LdrshRI | LdrLiteral | LdpRI | LdpPostIndex
            | LdrRO | LdrswRO | LdrGot | LdrTlvp | NeonLd1Post | Ldar | Ldarb | Ldarh | Ldaxr => {
                OpcodeCategory::Load
            }

            // Memory stores
            StrRI | StrbRI | StrhRI | StpRI | StpPreIndex | StrRO | STRWui | STRXui | STRSui
            | STRDui | NeonSt1Post | Stlr | Stlrb | Stlrh | Stlxr => OpcodeCategory::Store,

            // Pseudo
            Phi => OpcodeCategory::Phi,
            Nop => OpcodeCategory::Nop,

            // Everything else: target-specific without a generic category
            _ => OpcodeCategory::Other,
        }
    }
}

// ---------------------------------------------------------------------------
// X86Opcode::categorize
// ---------------------------------------------------------------------------

impl X86Opcode {
    /// Classify this x86-64 opcode into a target-independent category.
    pub fn categorize(self) -> OpcodeCategory {
        use X86Opcode::*;
        match self {
            // Arithmetic
            AddRR => OpcodeCategory::AddRR,
            AddRI => OpcodeCategory::AddRI,
            SubRR => OpcodeCategory::SubRR,
            SubRI => OpcodeCategory::SubRI,
            ImulRR => OpcodeCategory::MulRR,
            Neg => OpcodeCategory::Neg,

            // Logical (x86 uses Or/Xor naming)
            AndRR => OpcodeCategory::AndRR,
            AndRI => OpcodeCategory::AndRI,
            OrRR => OpcodeCategory::OrRR,
            OrRI => OpcodeCategory::OrRI,
            XorRR => OpcodeCategory::XorRR,
            XorRI => OpcodeCategory::XorRI,

            // Shifts (x86 uses Shl/Shr/Sar naming)
            ShlRR => OpcodeCategory::ShlRR,
            ShlRI => OpcodeCategory::ShlRI,
            ShrRR => OpcodeCategory::ShrRR,
            ShrRI => OpcodeCategory::ShrRI,
            SarRR => OpcodeCategory::SarRR,
            SarRI => OpcodeCategory::SarRI,

            // Moves
            MovRR | MovsdRR | MovssRR => OpcodeCategory::MovRR,
            MovRI => OpcodeCategory::MovRI,

            // Compare
            CmpRR => OpcodeCategory::CmpRR,
            CmpRI | CmpRI8 => OpcodeCategory::CmpRI,

            // Control flow
            Jmp => OpcodeCategory::Branch,
            Jcc => OpcodeCategory::CondBranch,
            Call | CallR | CallM => OpcodeCategory::Call,
            Ret => OpcodeCategory::Ret,

            // Memory loads
            MovRM | MovsdRM | MovssRM | MovRMSib | AddRM | SubRM | CmpRM | ImulRM | TestRM
            | MovssRipRel | MovsdRipRel | Pop => OpcodeCategory::Load,

            // Memory stores
            MovMR | MovsdMR | MovssMR | MovMRSib | Push => OpcodeCategory::Store,

            // Pseudo
            Phi => OpcodeCategory::Phi,
            Nop => OpcodeCategory::Nop,

            // Everything else: target-specific
            _ => OpcodeCategory::Other,
        }
    }
}

// ---------------------------------------------------------------------------
// TargetInfo trait
// ---------------------------------------------------------------------------

/// Target abstraction for optimization passes.
///
/// Provides target-independent queries needed by passes like peephole,
/// constant folding, CSE, and DCE. Each target (AArch64, x86-64)
/// implements this trait.
///
/// Note: memory effects (`MemoryEffect`) are NOT included here because
/// that type is defined in `llvm2-opt`, not `llvm2-ir`. Memory effect
/// queries remain in `llvm2-opt/src/effects.rs`.
pub trait TargetInfo {
    /// The target-specific opcode type.
    type Opcode: Copy + Eq + core::hash::Hash + core::fmt::Debug;

    /// Classify an opcode into a target-independent category.
    fn categorize(opcode: Self::Opcode) -> OpcodeCategory;

    /// Does this opcode produce a value (operand[0] is a def)?
    fn produces_value(opcode: Self::Opcode) -> bool;

    /// Does this opcode write implicit condition flags?
    fn writes_flags(opcode: Self::Opcode) -> bool;

    /// Does this opcode read implicit condition flags?
    fn reads_flags(opcode: Self::Opcode) -> bool;

    /// Is this opcode commutative (operand order doesn't affect result)?
    fn is_commutative(opcode: Self::Opcode) -> bool;

    /// Return the register-to-register move opcode for this target.
    fn mov_rr() -> Self::Opcode;

    /// Return the immediate-to-register move opcode for this target.
    fn mov_ri() -> Self::Opcode;

    /// Return the shift-left-by-immediate opcode for this target.
    fn shl_ri() -> Self::Opcode;

    /// Return the register-register subtraction opcode for this target.
    fn sub_rr() -> Self::Opcode;

    /// Return the register-register addition opcode for this target.
    fn add_rr() -> Self::Opcode;

    /// Return the negate opcode for this target.
    fn neg() -> Self::Opcode;

    /// Return the register-immediate subtraction opcode for this target.
    fn sub_ri() -> Self::Opcode;

    /// Return the register-immediate addition opcode for this target.
    fn add_ri() -> Self::Opcode;
}

// ---------------------------------------------------------------------------
// AArch64Target
// ---------------------------------------------------------------------------

/// AArch64 target implementation.
pub struct AArch64Target;

impl TargetInfo for AArch64Target {
    type Opcode = AArch64Opcode;

    #[inline]
    fn categorize(opcode: AArch64Opcode) -> OpcodeCategory {
        opcode.categorize()
    }

    #[inline]
    fn produces_value(opcode: AArch64Opcode) -> bool {
        opcode.produces_value()
    }

    fn writes_flags(opcode: AArch64Opcode) -> bool {
        use AArch64Opcode::*;
        matches!(
            opcode,
            CmpRR
                | CmpRI
                | CMPWrr
                | CMPXrr
                | CMPWri
                | CMPXri
                | Tst
                | Fcmp
                | AddsRR
                | AddsRI
                | SubsRR
                | SubsRI
        )
    }

    fn reads_flags(opcode: AArch64Opcode) -> bool {
        use AArch64Opcode::*;
        // Must stay in sync with `llvm2_opt::effects::reads_flags`.
        // - CSEL-family: test NZCV against a condition code immediate.
        // - ADC/SBC: consume the carry flag for i128 multi-precision
        //   arithmetic. Classifying them here keeps any TargetInfo-based
        //   consumer from reordering/CSE'ing ADC/SBC across a flag writer.
        //   See issue #409.
        matches!(opcode, CSet | Csel | Csinc | Csinv | Csneg | Adc | Sbc)
    }

    fn is_commutative(opcode: AArch64Opcode) -> bool {
        opcode.is_commutative()
    }

    #[inline]
    fn mov_rr() -> AArch64Opcode {
        AArch64Opcode::MovR
    }

    #[inline]
    fn mov_ri() -> AArch64Opcode {
        AArch64Opcode::MovI
    }

    #[inline]
    fn shl_ri() -> AArch64Opcode {
        AArch64Opcode::LslRI
    }

    #[inline]
    fn sub_rr() -> AArch64Opcode {
        AArch64Opcode::SubRR
    }

    #[inline]
    fn add_rr() -> AArch64Opcode {
        AArch64Opcode::AddRR
    }

    #[inline]
    fn neg() -> AArch64Opcode {
        AArch64Opcode::Neg
    }

    #[inline]
    fn sub_ri() -> AArch64Opcode {
        AArch64Opcode::SubRI
    }

    #[inline]
    fn add_ri() -> AArch64Opcode {
        AArch64Opcode::AddRI
    }
}

// ---------------------------------------------------------------------------
// X86_64Target
// ---------------------------------------------------------------------------

/// x86-64 target implementation.
pub struct X86_64Target;

impl TargetInfo for X86_64Target {
    type Opcode = X86Opcode;

    #[inline]
    fn categorize(opcode: X86Opcode) -> OpcodeCategory {
        opcode.categorize()
    }

    fn produces_value(opcode: X86Opcode) -> bool {
        use X86Opcode::*;
        // Instructions that do NOT produce a value:
        !matches!(
            opcode,
            // Compare/test: only set flags
            CmpRR | CmpRI | CmpRI8 | CmpRM | TestRR | TestRI | TestRM
            | Ucomisd | Ucomiss | BtRI
            // Stores
            | MovMR | MovsdMR | MovssMR | MovMRSib
            // Branches and control flow
            | Jmp | Jcc | Call | CallR | CallM | Ret
            // Stack store
            | Push
            // Pseudo with no value
            | Nop | NopMulti | StackAlloc
            // Atomic exchange (complex implicit operands)
            | Cmpxchg
            // Sign-extend implicit writes
            | Cdq | Cqo
        )
    }

    fn writes_flags(opcode: X86Opcode) -> bool {
        use X86Opcode::*;
        // On x86, almost ALL arithmetic/logical/shift instructions set RFLAGS.
        // Only moves, LEA, and pseudo-instructions do NOT set flags.
        matches!(
            opcode,
            // Arithmetic
            AddRR | AddRI | AddRM | SubRR | SubRI | SubRM
            | ImulRR | ImulRRI | ImulRM | Idiv | Div
            | Neg | Inc | Dec
            // Logical
            | AndRR | AndRI | OrRR | OrRI | XorRR | XorRI | Not
            // Shifts
            | ShlRR | ShlRI | ShrRR | ShrRI | SarRR | SarRI
            // Compare/test
            | CmpRR | CmpRI | CmpRI8 | CmpRM | TestRR | TestRI | TestRM
            // FP compare
            | Ucomisd | Ucomiss
            // Bit manipulation that sets flags
            | Bsf | Bsr | Tzcnt | Lzcnt | Popcnt | BtRI
            // Atomic
            | Cmpxchg
        )
    }

    fn reads_flags(opcode: X86Opcode) -> bool {
        use X86Opcode::*;
        matches!(opcode, Cmovcc | Setcc | Jcc)
    }

    fn is_commutative(opcode: X86Opcode) -> bool {
        use X86Opcode::*;
        matches!(
            opcode,
            AddRR | ImulRR | AndRR | OrRR | XorRR | Addsd | Mulsd | Addss | Mulss | Xchg
        )
    }

    #[inline]
    fn mov_rr() -> X86Opcode {
        X86Opcode::MovRR
    }

    #[inline]
    fn mov_ri() -> X86Opcode {
        X86Opcode::MovRI
    }

    #[inline]
    fn shl_ri() -> X86Opcode {
        X86Opcode::ShlRI
    }

    #[inline]
    fn sub_rr() -> X86Opcode {
        X86Opcode::SubRR
    }

    #[inline]
    fn add_rr() -> X86Opcode {
        X86Opcode::AddRR
    }

    #[inline]
    fn neg() -> X86Opcode {
        X86Opcode::Neg
    }

    #[inline]
    fn sub_ri() -> X86Opcode {
        X86Opcode::SubRI
    }

    #[inline]
    fn add_ri() -> X86Opcode {
        X86Opcode::AddRI
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- OpcodeCategory helper tests --

    #[test]
    fn category_is_arithmetic() {
        assert!(OpcodeCategory::AddRR.is_arithmetic());
        assert!(OpcodeCategory::SubRI.is_arithmetic());
        assert!(OpcodeCategory::Neg.is_arithmetic());
        assert!(!OpcodeCategory::AndRR.is_arithmetic());
        assert!(!OpcodeCategory::ShlRI.is_arithmetic());
        assert!(!OpcodeCategory::Other.is_arithmetic());
    }

    #[test]
    fn category_is_logical() {
        assert!(OpcodeCategory::AndRR.is_logical());
        assert!(OpcodeCategory::OrRI.is_logical());
        assert!(OpcodeCategory::XorRR.is_logical());
        assert!(!OpcodeCategory::AddRR.is_logical());
    }

    #[test]
    fn category_is_shift() {
        assert!(OpcodeCategory::ShlRR.is_shift());
        assert!(OpcodeCategory::SarRI.is_shift());
        assert!(!OpcodeCategory::AddRR.is_shift());
    }

    #[test]
    fn category_is_move() {
        assert!(OpcodeCategory::MovRR.is_move());
        assert!(OpcodeCategory::MovRI.is_move());
        assert!(!OpcodeCategory::AddRR.is_move());
    }

    // -- AArch64 categorize tests --

    #[test]
    fn aarch64_arithmetic_categories() {
        assert_eq!(AArch64Opcode::AddRR.categorize(), OpcodeCategory::AddRR);
        assert_eq!(AArch64Opcode::AddRI.categorize(), OpcodeCategory::AddRI);
        assert_eq!(
            AArch64Opcode::AddRIShift12.categorize(),
            OpcodeCategory::AddRI
        );
        assert_eq!(AArch64Opcode::SubRR.categorize(), OpcodeCategory::SubRR);
        assert_eq!(AArch64Opcode::SubRI.categorize(), OpcodeCategory::SubRI);
        assert_eq!(AArch64Opcode::MulRR.categorize(), OpcodeCategory::MulRR);
        assert_eq!(AArch64Opcode::Neg.categorize(), OpcodeCategory::Neg);
    }

    #[test]
    fn aarch64_logical_categories() {
        assert_eq!(AArch64Opcode::AndRR.categorize(), OpcodeCategory::AndRR);
        assert_eq!(AArch64Opcode::AndRI.categorize(), OpcodeCategory::AndRI);
        assert_eq!(AArch64Opcode::OrrRR.categorize(), OpcodeCategory::OrRR);
        assert_eq!(AArch64Opcode::OrrRI.categorize(), OpcodeCategory::OrRI);
        assert_eq!(AArch64Opcode::EorRR.categorize(), OpcodeCategory::XorRR);
        assert_eq!(AArch64Opcode::EorRI.categorize(), OpcodeCategory::XorRI);
    }

    #[test]
    fn aarch64_shift_categories() {
        assert_eq!(AArch64Opcode::LslRR.categorize(), OpcodeCategory::ShlRR);
        assert_eq!(AArch64Opcode::LslRI.categorize(), OpcodeCategory::ShlRI);
        assert_eq!(AArch64Opcode::LsrRR.categorize(), OpcodeCategory::ShrRR);
        assert_eq!(AArch64Opcode::LsrRI.categorize(), OpcodeCategory::ShrRI);
        assert_eq!(AArch64Opcode::AsrRR.categorize(), OpcodeCategory::SarRR);
        assert_eq!(AArch64Opcode::AsrRI.categorize(), OpcodeCategory::SarRI);
    }

    #[test]
    fn aarch64_move_categories() {
        assert_eq!(AArch64Opcode::MovR.categorize(), OpcodeCategory::MovRR);
        assert_eq!(AArch64Opcode::Copy.categorize(), OpcodeCategory::MovRR);
        assert_eq!(AArch64Opcode::MOVWrr.categorize(), OpcodeCategory::MovRR);
        assert_eq!(AArch64Opcode::MovI.categorize(), OpcodeCategory::MovRI);
        assert_eq!(AArch64Opcode::Movz.categorize(), OpcodeCategory::MovRI);
        assert_eq!(AArch64Opcode::Movn.categorize(), OpcodeCategory::MovRI);
    }

    #[test]
    fn aarch64_compare_categories() {
        assert_eq!(AArch64Opcode::CmpRR.categorize(), OpcodeCategory::CmpRR);
        assert_eq!(AArch64Opcode::CMPWrr.categorize(), OpcodeCategory::CmpRR);
        assert_eq!(AArch64Opcode::CmpRI.categorize(), OpcodeCategory::CmpRI);
        assert_eq!(AArch64Opcode::CMPXri.categorize(), OpcodeCategory::CmpRI);
    }

    #[test]
    fn aarch64_control_flow_categories() {
        assert_eq!(AArch64Opcode::B.categorize(), OpcodeCategory::Branch);
        assert_eq!(
            AArch64Opcode::BCond.categorize(),
            OpcodeCategory::CondBranch
        );
        assert_eq!(AArch64Opcode::Cbz.categorize(), OpcodeCategory::CondBranch);
        assert_eq!(AArch64Opcode::Bl.categorize(), OpcodeCategory::Call);
        assert_eq!(AArch64Opcode::BL.categorize(), OpcodeCategory::Call);
        assert_eq!(AArch64Opcode::Ret.categorize(), OpcodeCategory::Ret);
    }

    #[test]
    fn aarch64_memory_categories() {
        assert_eq!(AArch64Opcode::LdrRI.categorize(), OpcodeCategory::Load);
        assert_eq!(AArch64Opcode::LdpRI.categorize(), OpcodeCategory::Load);
        assert_eq!(AArch64Opcode::StrRI.categorize(), OpcodeCategory::Store);
        assert_eq!(AArch64Opcode::StpRI.categorize(), OpcodeCategory::Store);
    }

    #[test]
    fn aarch64_pseudo_categories() {
        assert_eq!(AArch64Opcode::Phi.categorize(), OpcodeCategory::Phi);
        assert_eq!(AArch64Opcode::Nop.categorize(), OpcodeCategory::Nop);
    }

    #[test]
    fn aarch64_other_categories() {
        // Target-specific opcodes that don't have generic categories
        assert_eq!(AArch64Opcode::Csel.categorize(), OpcodeCategory::Other);
        assert_eq!(AArch64Opcode::CSet.categorize(), OpcodeCategory::Other);
        assert_eq!(AArch64Opcode::FaddRR.categorize(), OpcodeCategory::Other);
        assert_eq!(AArch64Opcode::NeonAddV.categorize(), OpcodeCategory::Other);
        assert_eq!(AArch64Opcode::Movk.categorize(), OpcodeCategory::Other);
        assert_eq!(AArch64Opcode::Adrp.categorize(), OpcodeCategory::Other);
    }

    // -- X86 categorize tests --

    #[test]
    fn x86_arithmetic_categories() {
        assert_eq!(X86Opcode::AddRR.categorize(), OpcodeCategory::AddRR);
        assert_eq!(X86Opcode::AddRI.categorize(), OpcodeCategory::AddRI);
        assert_eq!(X86Opcode::SubRR.categorize(), OpcodeCategory::SubRR);
        assert_eq!(X86Opcode::SubRI.categorize(), OpcodeCategory::SubRI);
        assert_eq!(X86Opcode::ImulRR.categorize(), OpcodeCategory::MulRR);
        assert_eq!(X86Opcode::Neg.categorize(), OpcodeCategory::Neg);
    }

    #[test]
    fn x86_logical_categories() {
        assert_eq!(X86Opcode::AndRR.categorize(), OpcodeCategory::AndRR);
        assert_eq!(X86Opcode::AndRI.categorize(), OpcodeCategory::AndRI);
        assert_eq!(X86Opcode::OrRR.categorize(), OpcodeCategory::OrRR);
        assert_eq!(X86Opcode::OrRI.categorize(), OpcodeCategory::OrRI);
        assert_eq!(X86Opcode::XorRR.categorize(), OpcodeCategory::XorRR);
        assert_eq!(X86Opcode::XorRI.categorize(), OpcodeCategory::XorRI);
    }

    #[test]
    fn x86_shift_categories() {
        assert_eq!(X86Opcode::ShlRR.categorize(), OpcodeCategory::ShlRR);
        assert_eq!(X86Opcode::ShlRI.categorize(), OpcodeCategory::ShlRI);
        assert_eq!(X86Opcode::ShrRR.categorize(), OpcodeCategory::ShrRR);
        assert_eq!(X86Opcode::ShrRI.categorize(), OpcodeCategory::ShrRI);
        assert_eq!(X86Opcode::SarRR.categorize(), OpcodeCategory::SarRR);
        assert_eq!(X86Opcode::SarRI.categorize(), OpcodeCategory::SarRI);
    }

    #[test]
    fn x86_move_categories() {
        assert_eq!(X86Opcode::MovRR.categorize(), OpcodeCategory::MovRR);
        assert_eq!(X86Opcode::MovsdRR.categorize(), OpcodeCategory::MovRR);
        assert_eq!(X86Opcode::MovssRR.categorize(), OpcodeCategory::MovRR);
        assert_eq!(X86Opcode::MovRI.categorize(), OpcodeCategory::MovRI);
    }

    #[test]
    fn x86_compare_categories() {
        assert_eq!(X86Opcode::CmpRR.categorize(), OpcodeCategory::CmpRR);
        assert_eq!(X86Opcode::CmpRI.categorize(), OpcodeCategory::CmpRI);
        assert_eq!(X86Opcode::CmpRI8.categorize(), OpcodeCategory::CmpRI);
    }

    #[test]
    fn x86_control_flow_categories() {
        assert_eq!(X86Opcode::Jmp.categorize(), OpcodeCategory::Branch);
        assert_eq!(X86Opcode::Jcc.categorize(), OpcodeCategory::CondBranch);
        assert_eq!(X86Opcode::Call.categorize(), OpcodeCategory::Call);
        assert_eq!(X86Opcode::CallR.categorize(), OpcodeCategory::Call);
        assert_eq!(X86Opcode::Ret.categorize(), OpcodeCategory::Ret);
    }

    #[test]
    fn x86_memory_categories() {
        assert_eq!(X86Opcode::MovRM.categorize(), OpcodeCategory::Load);
        assert_eq!(X86Opcode::MovsdRM.categorize(), OpcodeCategory::Load);
        assert_eq!(X86Opcode::Pop.categorize(), OpcodeCategory::Load);
        assert_eq!(X86Opcode::MovMR.categorize(), OpcodeCategory::Store);
        assert_eq!(X86Opcode::MovsdMR.categorize(), OpcodeCategory::Store);
        assert_eq!(X86Opcode::Push.categorize(), OpcodeCategory::Store);
    }

    #[test]
    fn x86_pseudo_categories() {
        assert_eq!(X86Opcode::Phi.categorize(), OpcodeCategory::Phi);
        assert_eq!(X86Opcode::Nop.categorize(), OpcodeCategory::Nop);
    }

    #[test]
    fn x86_other_categories() {
        assert_eq!(X86Opcode::Cmovcc.categorize(), OpcodeCategory::Other);
        assert_eq!(X86Opcode::Setcc.categorize(), OpcodeCategory::Other);
        assert_eq!(X86Opcode::Lea.categorize(), OpcodeCategory::Other);
        assert_eq!(X86Opcode::Bswap.categorize(), OpcodeCategory::Other);
        assert_eq!(X86Opcode::Xchg.categorize(), OpcodeCategory::Other);
    }

    // -- TargetInfo trait tests --

    #[test]
    fn aarch64_target_info() {
        assert_eq!(
            AArch64Target::categorize(AArch64Opcode::AddRR),
            OpcodeCategory::AddRR
        );
        assert!(AArch64Target::produces_value(AArch64Opcode::AddRR));
        assert!(!AArch64Target::produces_value(AArch64Opcode::CmpRR));
        assert!(AArch64Target::writes_flags(AArch64Opcode::CmpRR));
        assert!(!AArch64Target::writes_flags(AArch64Opcode::AddRR));
        assert!(AArch64Target::reads_flags(AArch64Opcode::CSet));
        assert!(!AArch64Target::reads_flags(AArch64Opcode::AddRR));
        assert!(AArch64Target::is_commutative(AArch64Opcode::AddRR));
        assert!(!AArch64Target::is_commutative(AArch64Opcode::SubRR));
        assert_eq!(AArch64Target::mov_rr(), AArch64Opcode::MovR);
        assert_eq!(AArch64Target::mov_ri(), AArch64Opcode::MovI);
        assert_eq!(AArch64Target::shl_ri(), AArch64Opcode::LslRI);
        assert_eq!(AArch64Target::sub_rr(), AArch64Opcode::SubRR);
        assert_eq!(AArch64Target::add_rr(), AArch64Opcode::AddRR);
        assert_eq!(AArch64Target::neg(), AArch64Opcode::Neg);
        assert_eq!(AArch64Target::sub_ri(), AArch64Opcode::SubRI);
        assert_eq!(AArch64Target::add_ri(), AArch64Opcode::AddRI);
    }

    #[test]
    fn x86_target_info() {
        assert_eq!(
            X86_64Target::categorize(X86Opcode::AddRR),
            OpcodeCategory::AddRR
        );
        assert!(X86_64Target::produces_value(X86Opcode::AddRR));
        assert!(!X86_64Target::produces_value(X86Opcode::CmpRR));
        // x86: ADD sets flags, unlike AArch64 ADD
        assert!(X86_64Target::writes_flags(X86Opcode::AddRR));
        assert!(X86_64Target::writes_flags(X86Opcode::CmpRR));
        assert!(!X86_64Target::writes_flags(X86Opcode::MovRR));
        assert!(X86_64Target::reads_flags(X86Opcode::Cmovcc));
        assert!(X86_64Target::reads_flags(X86Opcode::Setcc));
        assert!(X86_64Target::reads_flags(X86Opcode::Jcc));
        assert!(!X86_64Target::reads_flags(X86Opcode::AddRR));
        assert!(X86_64Target::is_commutative(X86Opcode::AddRR));
        assert!(!X86_64Target::is_commutative(X86Opcode::SubRR));
        assert_eq!(X86_64Target::mov_rr(), X86Opcode::MovRR);
        assert_eq!(X86_64Target::mov_ri(), X86Opcode::MovRI);
        assert_eq!(X86_64Target::shl_ri(), X86Opcode::ShlRI);
        assert_eq!(X86_64Target::sub_rr(), X86Opcode::SubRR);
        assert_eq!(X86_64Target::add_rr(), X86Opcode::AddRR);
        assert_eq!(X86_64Target::neg(), X86Opcode::Neg);
        assert_eq!(X86_64Target::sub_ri(), X86Opcode::SubRI);
        assert_eq!(X86_64Target::add_ri(), X86Opcode::AddRI);
    }

    // -- Cross-target consistency tests --

    #[test]
    fn both_targets_agree_on_add_category() {
        assert_eq!(
            AArch64Target::categorize(AArch64Opcode::AddRR),
            X86_64Target::categorize(X86Opcode::AddRR),
        );
        assert_eq!(
            AArch64Target::categorize(AArch64Opcode::AddRI),
            X86_64Target::categorize(X86Opcode::AddRI),
        );
    }

    #[test]
    fn both_targets_agree_on_sub_category() {
        assert_eq!(
            AArch64Target::categorize(AArch64Opcode::SubRR),
            X86_64Target::categorize(X86Opcode::SubRR),
        );
    }

    #[test]
    fn both_targets_agree_on_move_category() {
        assert_eq!(
            AArch64Target::categorize(AArch64Opcode::MovR),
            X86_64Target::categorize(X86Opcode::MovRR),
        );
        assert_eq!(
            AArch64Target::categorize(AArch64Opcode::MovI),
            X86_64Target::categorize(X86Opcode::MovRI),
        );
    }

    #[test]
    fn both_targets_mov_rr_categorizes_as_mov_rr() {
        assert_eq!(
            AArch64Target::categorize(AArch64Target::mov_rr()),
            OpcodeCategory::MovRR,
        );
        assert_eq!(
            X86_64Target::categorize(X86_64Target::mov_rr()),
            OpcodeCategory::MovRR,
        );
    }

    #[test]
    fn both_targets_shl_ri_categorizes_as_shl_ri() {
        assert_eq!(
            AArch64Target::categorize(AArch64Target::shl_ri()),
            OpcodeCategory::ShlRI,
        );
        assert_eq!(
            X86_64Target::categorize(X86_64Target::shl_ri()),
            OpcodeCategory::ShlRI,
        );
    }

    #[test]
    fn both_targets_sub_rr_categorizes_as_sub_rr() {
        assert_eq!(
            AArch64Target::categorize(AArch64Target::sub_rr()),
            OpcodeCategory::SubRR,
        );
        assert_eq!(
            X86_64Target::categorize(X86_64Target::sub_rr()),
            OpcodeCategory::SubRR,
        );
    }

    #[test]
    fn both_targets_add_rr_categorizes_as_add_rr() {
        assert_eq!(
            AArch64Target::categorize(AArch64Target::add_rr()),
            OpcodeCategory::AddRR,
        );
        assert_eq!(
            X86_64Target::categorize(X86_64Target::add_rr()),
            OpcodeCategory::AddRR,
        );
    }

    #[test]
    fn both_targets_neg_categorizes_as_neg() {
        assert_eq!(
            AArch64Target::categorize(AArch64Target::neg()),
            OpcodeCategory::Neg,
        );
        assert_eq!(
            X86_64Target::categorize(X86_64Target::neg()),
            OpcodeCategory::Neg,
        );
    }

    #[test]
    fn both_targets_sub_ri_categorizes_as_sub_ri() {
        assert_eq!(
            AArch64Target::categorize(AArch64Target::sub_ri()),
            OpcodeCategory::SubRI,
        );
        assert_eq!(
            X86_64Target::categorize(X86_64Target::sub_ri()),
            OpcodeCategory::SubRI,
        );
    }

    #[test]
    fn both_targets_add_ri_categorizes_as_add_ri() {
        assert_eq!(
            AArch64Target::categorize(AArch64Target::add_ri()),
            OpcodeCategory::AddRI,
        );
        assert_eq!(
            X86_64Target::categorize(X86_64Target::add_ri()),
            OpcodeCategory::AddRI,
        );
    }
}
