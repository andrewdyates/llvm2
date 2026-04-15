// llvm2-opt - Memory effects model
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Memory-effects model for machine instructions.
//!
//! Required for safe CSE, LICM, and DCE. Each opcode is classified as
//! Pure, Load, Store, or Call, which determines whether the instruction
//! can be reordered, eliminated, or hoisted.
//!
//! Reference: designs/2026-04-12-aarch64-backend.md, "Memory-Effects Model"
//!
//! | Effect | Meaning |
//! |--------|---------|
//! | Pure   | No memory access, no side effects. Safe to reorder, CSE, DCE. |
//! | Load   | Reads memory. Can be CSE'd with identical loads if no intervening store. |
//! | Store  | Writes memory. Barrier for loads and other stores. |
//! | Call   | Clobbers everything (conservative default). |

use llvm2_ir::AArch64Opcode;

/// Memory effect classification for an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryEffect {
    /// No memory access, no observable side effects.
    /// Safe to reorder, eliminate if unused, or CSE.
    Pure,
    /// Reads memory. May depend on prior stores.
    /// Can be CSE'd if no intervening store to the same location.
    Load,
    /// Writes memory. Acts as a barrier for loads and other stores.
    Store,
    /// Full memory barrier. Clobbers all registers and memory
    /// (conservative assumption for function calls).
    Call,
}

impl MemoryEffect {
    /// Returns true if this instruction has no memory side effects.
    /// Pure instructions can be safely eliminated if their result is unused.
    #[inline]
    pub fn is_pure(self) -> bool {
        self == Self::Pure
    }

    /// Returns true if this instruction reads memory.
    #[inline]
    pub fn reads_memory(self) -> bool {
        matches!(self, Self::Load | Self::Call)
    }

    /// Returns true if this instruction writes memory.
    #[inline]
    pub fn writes_memory(self) -> bool {
        matches!(self, Self::Store | Self::Call)
    }

    /// Returns true if this instruction is a memory barrier
    /// (prevents reordering of loads and stores across it).
    #[inline]
    pub fn is_barrier(self) -> bool {
        matches!(self, Self::Call)
    }
}

/// Classify the memory effect of an AArch64 opcode.
///
/// This is the authoritative source for memory effect information.
/// The classification is conservative: if in doubt, we classify as
/// having more effects rather than fewer (correctness over optimization).
pub fn opcode_effect(opcode: AArch64Opcode) -> MemoryEffect {
    use AArch64Opcode::*;
    match opcode {
        // -- Loads: read memory --
        LdrRI | LdrbRI | LdrhRI | LdrsbRI | LdrshRI | LdrLiteral | LdpRI | LdpPostIndex
        | LdrRO | LdrGot | LdrTlvp | NeonLd1Post => MemoryEffect::Load,

        // -- Stores: write memory --
        StrRI | StrbRI | StrhRI | StpRI | StpPreIndex | StrRO
        | STRWui | STRXui | STRSui | STRDui | NeonSt1Post => MemoryEffect::Store,

        // -- Calls: full barrier --
        Bl | Blr | BL | BLR => MemoryEffect::Call,

        // -- Stack allocation: side effect (modifies SP) --
        StackAlloc => MemoryEffect::Store,

        // -- Everything else: pure computation --
        // Arithmetic
        AddRR | AddRI | SubRR | SubRI | MulRR | Msub | Smull | Umull | SDiv | UDiv | Neg => MemoryEffect::Pure,

        // Logical
        AndRR | AndRI | OrrRR | OrrRI | EorRR | EorRI | BicRR => MemoryEffect::Pure,

        // Shifts
        LslRR | LsrRR | AsrRR | LslRI | LsrRI | AsrRI => MemoryEffect::Pure,

        // Compare/test: these set flags but don't access memory.
        // Note: CMP/TST have HAS_SIDE_EFFECTS in InstFlags because they
        // write condition flags. DCE handles this via InstFlags, not
        // MemoryEffect. For memory-effect purposes, these are pure.
        CmpRR | CmpRI | CMPWrr | CMPXrr | CMPWri | CMPXri | Tst | Fcmp => MemoryEffect::Pure,

        // Conditional select/set: pure computation, no memory access.
        Csel | CSet | Csinc | Csinv | Csneg => MemoryEffect::Pure,

        // Move (including LLVM-style typed aliases)
        MovR | MovI | Movz | Movn | Movk | FmovImm
        | MOVWrr | MOVXrr | MOVZWi | MOVZXi => MemoryEffect::Pure,

        // Extension
        Sxtw | Uxtw | Sxtb | Sxth | Uxtb | Uxth => MemoryEffect::Pure,

        // Bitfield operations
        Ubfm | Sbfm | Bfm => MemoryEffect::Pure,

        // Logical (OR-NOT)
        OrnRR => MemoryEffect::Pure,

        // Floating-point arithmetic
        FaddRR | FsubRR | FmulRR | FdivRR | FnegRR | FabsRR | FsqrtRR => MemoryEffect::Pure,

        // NEON SIMD: pure computation (no memory access except LD1/ST1 above)
        NeonAddV | NeonSubV | NeonMulV
        | NeonFaddV | NeonFsubV | NeonFmulV | NeonFdivV
        | NeonAndV | NeonOrrV | NeonEorV | NeonBicV | NeonNotV
        | NeonCmeqV | NeonCmgtV | NeonCmgeV
        | NeonDupElem | NeonDupGen | NeonInsGen | NeonMovi => MemoryEffect::Pure,

        // FP conversion
        FcvtzsRR | FcvtzuRR | ScvtfRR | UcvtfRR => MemoryEffect::Pure,

        // Float precision conversion
        FcvtSD | FcvtDS => MemoryEffect::Pure,

        // Bitcast (FMOV between GPR/FPR)
        FmovGprFpr | FmovFprGpr => MemoryEffect::Pure,

        // Address computation (no memory access)
        Adrp | AddPCRel => MemoryEffect::Pure,

        // Checked arithmetic: set flags but no memory access
        AddsRR | AddsRI | SubsRR | SubsRI => MemoryEffect::Pure,

        // Trap pseudo-instructions: control flow, not memory ops
        TrapOverflow | TrapBoundsCheck | TrapNull | TrapDivZero | TrapShiftRange => MemoryEffect::Pure,

        // Reference counting: read and write memory (refcount field)
        Retain | Release => MemoryEffect::Store,

        // Atomic loads (load-acquire): memory read with ordering
        Ldar | Ldarb | Ldarh | Ldaxr => MemoryEffect::Load,

        // Atomic stores (store-release): memory write with ordering
        Stlr | Stlrb | Stlrh | Stlxr => MemoryEffect::Store,

        // Atomic RMW (LSE): both read and write — classify as Store (conservative)
        Ldadd | Ldadda | Ldaddal
        | Ldclr | Ldclral
        | Ldeor | Ldeoral
        | Ldset | Ldsetal
        | Swp | Swpal
        | Cas | Casa | Casal => MemoryEffect::Store,

        // Barriers: full memory barrier (acts like a call for ordering purposes)
        Dmb | Dsb | Isb => MemoryEffect::Call,

        // Branches: not memory ops. DCE handles branches via InstFlags.
        B | BCond | Bcc | Cbz | Cbnz | Tbz | Tbnz | Br | Ret => MemoryEffect::Pure,

        // Pseudo-instructions
        Phi => MemoryEffect::Pure,
        Copy => MemoryEffect::Pure,
        Nop => MemoryEffect::Pure,
    }
}

/// Returns true if this opcode produces a value (operand[0] is a def).
///
/// Instructions that don't produce values: CMP, TST, STR, STP, branches,
/// returns, NOP, calls, traps, and reference counting ops.
///
/// This is the **authoritative, single definition** — all passes (DCE, CSE,
/// LICM, addr-mode, etc.) must call this rather than maintaining their own
/// copy. See issue #96.
pub fn produces_value(opcode: AArch64Opcode) -> bool {
    use AArch64Opcode::*;
    match opcode {
        // Compare/test: set flags, no register def
        CmpRR | CmpRI | Tst | Fcmp => false,
        // Stores: write to memory, no register def
        StrRI | StrbRI | StrhRI | StpRI | StpPreIndex | StrRO => false,
        // Branches and returns: control flow, no register def
        B | BCond | Cbz | Cbnz | Tbz | Tbnz | Br | Ret => false,
        // Trap pseudo-instructions: control flow, no register def
        TrapOverflow | TrapBoundsCheck | TrapNull | TrapDivZero | TrapShiftRange => false,
        // Reference counting: side effects, no register def
        Retain | Release => false,
        // Nop: no def
        Nop => false,
        // Calls: they DO produce a result (in X0 typically), but that's
        // handled via implicit defs. For our simple model, calls have
        // side effects and won't be DCE'd anyway.
        Bl | Blr => false,
        // Everything else produces a value in operand[0]
        // (including AddsRR/SubsRR which produce a result plus set flags)
        _ => true,
    }
}

/// Returns true if this instruction produces a value.
///
/// Convenience wrapper around [`produces_value`] that takes a `MachInst`
/// reference instead of a bare opcode.
pub fn inst_produces_value(inst: &llvm2_ir::MachInst) -> bool {
    produces_value(inst.opcode)
}

/// Returns true if an instruction with the given opcode can be safely
/// eliminated if its result is unused and it has no other side effects.
///
/// This combines the memory-effect model with the knowledge that
/// compare/test instructions set condition flags (a side effect tracked
/// by InstFlags, not MemoryEffect).
pub fn is_removable(opcode: AArch64Opcode) -> bool {
    let effect = opcode_effect(opcode);
    if !effect.is_pure() {
        return false;
    }

    use AArch64Opcode::*;
    // Compare/test and checked arithmetic set NZCV flags — not removable
    // even though they don't access memory.
    !matches!(opcode, CmpRR | CmpRI | Tst | Fcmp | AddsRR | AddsRI | SubsRR | SubsRI)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::AArch64Opcode;

    #[test]
    fn test_arithmetic_is_pure() {
        assert_eq!(opcode_effect(AArch64Opcode::AddRR), MemoryEffect::Pure);
        assert_eq!(opcode_effect(AArch64Opcode::SubRI), MemoryEffect::Pure);
        assert_eq!(opcode_effect(AArch64Opcode::MulRR), MemoryEffect::Pure);
        assert_eq!(opcode_effect(AArch64Opcode::SDiv), MemoryEffect::Pure);
    }

    #[test]
    fn test_loads_are_load() {
        assert_eq!(opcode_effect(AArch64Opcode::LdrRI), MemoryEffect::Load);
        assert_eq!(opcode_effect(AArch64Opcode::LdpRI), MemoryEffect::Load);
        assert_eq!(opcode_effect(AArch64Opcode::LdrLiteral), MemoryEffect::Load);
        // Byte/halfword extending loads
        assert_eq!(opcode_effect(AArch64Opcode::LdrbRI), MemoryEffect::Load);
        assert_eq!(opcode_effect(AArch64Opcode::LdrhRI), MemoryEffect::Load);
        assert_eq!(opcode_effect(AArch64Opcode::LdrsbRI), MemoryEffect::Load);
        assert_eq!(opcode_effect(AArch64Opcode::LdrshRI), MemoryEffect::Load);
    }

    #[test]
    fn test_stores_are_store() {
        assert_eq!(opcode_effect(AArch64Opcode::StrRI), MemoryEffect::Store);
        assert_eq!(opcode_effect(AArch64Opcode::StpRI), MemoryEffect::Store);
        // Byte/halfword truncating stores
        assert_eq!(opcode_effect(AArch64Opcode::StrbRI), MemoryEffect::Store);
        assert_eq!(opcode_effect(AArch64Opcode::StrhRI), MemoryEffect::Store);
    }

    #[test]
    fn test_calls_are_call() {
        assert_eq!(opcode_effect(AArch64Opcode::Bl), MemoryEffect::Call);
        assert_eq!(opcode_effect(AArch64Opcode::Blr), MemoryEffect::Call);
    }

    #[test]
    fn test_branches_are_pure() {
        assert_eq!(opcode_effect(AArch64Opcode::B), MemoryEffect::Pure);
        assert_eq!(opcode_effect(AArch64Opcode::BCond), MemoryEffect::Pure);
        assert_eq!(opcode_effect(AArch64Opcode::Ret), MemoryEffect::Pure);
    }

    #[test]
    fn test_removable() {
        assert!(is_removable(AArch64Opcode::AddRR));
        assert!(is_removable(AArch64Opcode::MovR));
        assert!(!is_removable(AArch64Opcode::CmpRR));
        assert!(!is_removable(AArch64Opcode::LdrRI));
        assert!(!is_removable(AArch64Opcode::StrRI));
        assert!(!is_removable(AArch64Opcode::Bl));
    }

    #[test]
    fn test_memory_effect_queries() {
        assert!(MemoryEffect::Pure.is_pure());
        assert!(!MemoryEffect::Load.is_pure());

        assert!(MemoryEffect::Load.reads_memory());
        assert!(MemoryEffect::Call.reads_memory());
        assert!(!MemoryEffect::Pure.reads_memory());

        assert!(MemoryEffect::Store.writes_memory());
        assert!(MemoryEffect::Call.writes_memory());
        assert!(!MemoryEffect::Load.writes_memory());

        assert!(MemoryEffect::Call.is_barrier());
        assert!(!MemoryEffect::Store.is_barrier());
    }
}
