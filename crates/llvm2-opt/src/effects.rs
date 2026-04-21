// llvm2-opt - Memory effects model
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
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
use llvm2_ir::OpcodeCategory;
use llvm2_ir::X86Opcode;

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
        LdrRI | LdrbRI | LdrhRI | LdrsbRI | LdrshRI | LdrLiteral | LdpRI | LdpPostIndex | LdrRO
        | LdrswRO | LdrGot | LdrTlvp | NeonLd1Post => MemoryEffect::Load,

        // -- Stores: write memory --
        StrRI | StrbRI | StrhRI | StpRI | StpPreIndex | StrRO | STRWui | STRXui | STRSui
        | STRDui | NeonSt1Post => MemoryEffect::Store,

        // -- Calls: full barrier --
        Bl | Blr | BL | BLR => MemoryEffect::Call,

        // -- Stack allocation: side effect (modifies SP) --
        StackAlloc => MemoryEffect::Store,

        // -- Everything else: pure computation --
        // Arithmetic
        AddRR | AddRI | AddRIShift12 | SubRR | SubRI | MulRR | Msub | Smull | Umull | SDiv
        | UDiv | Neg => MemoryEffect::Pure,

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
        MovR | MovI | Movz | Movn | Movk | FmovImm | MOVWrr | MOVXrr | MOVZWi | MOVZXi => {
            MemoryEffect::Pure
        }

        // Extension
        Sxtw | Uxtw | Sxtb | Sxth | Uxtb | Uxth => MemoryEffect::Pure,

        // Bitfield operations
        Ubfm | Sbfm | Bfm => MemoryEffect::Pure,

        // Logical (OR-NOT)
        OrnRR => MemoryEffect::Pure,

        // Floating-point arithmetic
        FaddRR | FsubRR | FmulRR | FdivRR | FnegRR | FabsRR | FsqrtRR => MemoryEffect::Pure,

        // NEON SIMD: pure computation (no memory access except LD1/ST1 above)
        NeonAddV | NeonSubV | NeonMulV | NeonFaddV | NeonFsubV | NeonFmulV | NeonFdivV
        | NeonAndV | NeonOrrV | NeonEorV | NeonBicV | NeonNotV | NeonCmeqV | NeonCmgtV
        | NeonCmgeV | NeonDupElem | NeonDupGen | NeonInsGen | NeonMovi => MemoryEffect::Pure,

        // FP conversion
        FcvtzsRR | FcvtzuRR | ScvtfRR | UcvtfRR => MemoryEffect::Pure,

        // Float precision conversion
        FcvtSD | FcvtDS => MemoryEffect::Pure,

        // Bitcast (FMOV between GPR/FPR)
        FmovGprFpr | FmovFprGpr | FmovFprFpr => MemoryEffect::Pure,

        // Address computation (no memory access)
        Adrp | Adr | AddPCRel => MemoryEffect::Pure,

        // Checked arithmetic: set flags but no memory access
        AddsRR | AddsRI | SubsRR | SubsRI => MemoryEffect::Pure,

        // i128 multi-register arithmetic: pure computation (ADC/SBC read flags, no memory)
        Adc | Sbc | Umulh | Smulh | Madd => MemoryEffect::Pure,

        // Trap pseudo-instructions: control flow, not memory ops
        TrapOverflow | TrapBoundsCheck | TrapNull | TrapDivZero | TrapShiftRange => {
            MemoryEffect::Pure
        }

        // Reference counting: read and write memory (refcount field)
        Retain | Release => MemoryEffect::Store,

        // Atomic loads (load-acquire): memory read with ordering
        Ldar | Ldarb | Ldarh | Ldaxr => MemoryEffect::Load,

        // Atomic stores (store-release): memory write with ordering
        Stlr | Stlrb | Stlrh | Stlxr => MemoryEffect::Store,

        // Atomic RMW (LSE): both read and write — classify as Store (conservative)
        Ldadd | Ldadda | Ldaddal | Ldclr | Ldclral | Ldeor | Ldeoral | Ldset | Ldsetal | Swp
        | Swpal | Cas | Casa | Casal => MemoryEffect::Store,

        // Barriers: full memory barrier (acts like a call for ordering purposes)
        Dmb | Dsb | Isb => MemoryEffect::Call,

        // System register read (MRS): model as a Call for alias-analysis
        // purposes. This keeps MRS from being hoisted out of loops, sunk
        // past memory ops, or CSE'd across writes to the same sysreg.
        // TPIDR_EL0 specifically is thread-stable, but this opcode is the
        // umbrella for all sysregs (including performance counters), so the
        // conservative choice is a barrier-style classification.
        Mrs => MemoryEffect::Call,

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
/// Delegates to [`AArch64Opcode::produces_value`] — the single source of
/// truth. This wrapper preserves the existing function signature for callers.
/// See issue #96.
pub fn produces_value(opcode: AArch64Opcode) -> bool {
    opcode.produces_value()
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
    !matches!(
        opcode,
        CmpRR | CmpRI | Tst | Fcmp | AddsRR | AddsRI | SubsRR | SubsRI
    )
}

/// Returns true if the opcode writes implicit NZCV condition flags.
///
/// These instructions set the processor flags (N, Z, C, V) as a side effect.
/// Any subsequent flag-reading instruction (CSet, Csel, BCond, etc.) depends
/// on the most recent flag-writing instruction, but this dependency is NOT
/// captured in the explicit operand list.
///
/// **The instruction scheduler must order flag-writers before flag-readers.**
/// Without explicit edges, the scheduler may freely reorder a CSet before
/// the CMP that sets the flags it consumes.
pub fn writes_flags(opcode: AArch64Opcode) -> bool {
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

/// Returns true if the opcode has a tied def-use operand (operand[0] is
/// both the destination AND an implicit source — a read-modify-write).
///
/// MOVK (`MOVK Rd, #imm16, LSL #shift`) inserts a 16-bit immediate into
/// the destination register while preserving the other bits. This means
/// the "current value" of Rd is an implicit input to the instruction,
/// but it does not appear in the operand list (which only contains
/// `[def_reg, imm, shift]`).
///
/// BFM (`BFM Rd, Rn, #immr, #imms`, alias `BFI`/`BFXIL`) is a bitfield
/// *insert*: it writes a contiguous bitfield of `Rn` into `Rd` while
/// preserving the bits of `Rd` outside that field. Like MOVK, the prior
/// value of `Rd` is an implicit input that is not visible in the operand
/// list. Its siblings `UBFM` and `SBFM` do NOT have this property — they
/// zero or sign-extend the uncovered bits and therefore fully redefine
/// `Rd`, so only `BFM` belongs here.
///
/// **Why this matters:**
/// - **GVN/CSE**: two BFMs (or two MOVKs) with the same explicit operands
///   are NOT the same expression unless their prior destination values
///   match. Treating them as identical silently drops the second write
///   onto a different register and corrupts the destination. See issues
///   #366 (MOVK) and #408 (BFM).
/// - **Instruction scheduler**: a tied-def-use instruction must be ordered
///   AFTER the instruction that established its prior destination value.
///   Without an explicit RAW edge on operand[0], the scheduler may move
///   it before its preceding setter in the materialize chain, or move
///   readers between the setter and a trailing MOVK/BFM. See issue #382.
pub fn has_tied_def_use(opcode: AArch64Opcode) -> bool {
    matches!(opcode, AArch64Opcode::Movk | AArch64Opcode::Bfm)
}

/// Returns true if the opcode reads implicit NZCV condition flags.
///
/// These instructions consume flag state set by a prior CMP/TST/ADDS/SUBS
/// but do NOT capture that dependency in their explicit operands. The
/// condition code immediate (e.g., LE=13) is just a selector for which
/// flags to test, NOT the flag values themselves. Therefore, two flag-reading
/// instructions with the same explicit operands may produce different values
/// if different comparisons set the flags.
///
/// This set covers:
/// - CSEL-family conditional selects (`CSEL`, `CSET`, `CSINC`, `CSINV`,
///   `CSNEG`) which test NZCV against a condition code immediate.
/// - `ADC`/`SBC` multi-precision arithmetic, which adds/subtracts with
///   the carry flag `C` set by a prior `ADDS`/`SUBS`. The carry bit is
///   an implicit input; two ADCs with identical explicit operands but
///   reached after different flag writers produce different results.
///   See issue #409 for the i128 miscompile class.
///
/// **CSE and GVN must skip these instructions.** Treating them as pure
/// functions of their explicit operands is unsound because the implicit
/// flag input is not visible in the operand list.
///
/// **LICM must not hoist these instructions out of loops** — the carry
/// flag changes every iteration based on the loop body's `ADDS`/`SUBS`.
///
/// **The instruction scheduler must add edges from the most recent
/// flag-writing instruction to each flag-reading instruction.**
pub fn reads_flags(opcode: AArch64Opcode) -> bool {
    use AArch64Opcode::*;
    matches!(opcode, CSet | Csel | Csinc | Csinv | Csneg | Adc | Sbc)
}

// ===========================================================================
// x86-64 memory-effects model
// ===========================================================================

/// Classify the memory effect of an x86-64 opcode.
///
/// Mirrors [`opcode_effect`] for the x86-64 target.
pub fn x86_opcode_effect(opcode: X86Opcode) -> MemoryEffect {
    use X86Opcode::*;
    match opcode {
        // -- Loads: read memory --
        MovRM | MovsdRM | MovssRM | MovRMSib | AddRM | SubRM | CmpRM | ImulRM | TestRM
        | MovssRipRel | MovsdRipRel | Pop => MemoryEffect::Load,

        // -- Stores: write memory --
        MovMR | MovsdMR | MovssMR | MovMRSib | Push => MemoryEffect::Store,

        // -- Calls: full barrier --
        Call | CallR | CallM => MemoryEffect::Call,

        // -- Everything else: pure computation --
        // Arithmetic
        AddRR | AddRI | SubRR | SubRI | ImulRR | ImulRRI | Neg | Inc | Dec => MemoryEffect::Pure,

        // Division (has side effects but no memory access)
        Idiv | Div => MemoryEffect::Pure,

        // Sign-extend implicit (CDQ/CQO)
        Cdq | Cqo => MemoryEffect::Pure,

        // Logical
        AndRR | AndRI | OrRR | OrRI | XorRR | XorRI | Not => MemoryEffect::Pure,

        // Shifts
        ShlRR | ShlRI | ShrRR | ShrRI | SarRR | SarRI => MemoryEffect::Pure,

        // Compare/test (set flags, no memory access)
        CmpRR | CmpRI | CmpRI8 | TestRR | TestRI | Ucomisd | Ucomiss | BtRI => MemoryEffect::Pure,

        // Moves (register-register and register-immediate)
        MovRR | MovRI | Movzx | MovzxW | MovsxB | MovsxW | Movsx | MovsdRR | MovssRR => {
            MemoryEffect::Pure
        }

        // LEA (address computation, no memory access)
        Lea | LeaSib | LeaRip => MemoryEffect::Pure,

        // Conditional move/set
        Cmovcc | Setcc => MemoryEffect::Pure,

        // SSE register-register arithmetic
        Addsd | Subsd | Mulsd | Divsd | Addss | Subss | Mulss | Divss => MemoryEffect::Pure,

        // SSE type conversions
        Cvtsi2sd | Cvtsd2si | Cvtsi2ss | Cvtss2si | Cvtsd2ss | Cvtss2sd => MemoryEffect::Pure,

        // GPR <-> XMM transfers
        MovdToXmm | MovdFromXmm | MovqToXmm | MovqFromXmm => MemoryEffect::Pure,

        // Bit manipulation
        Bsf | Bsr | Tzcnt | Lzcnt | Popcnt | Bswap => MemoryEffect::Pure,

        // Atomic: conservative (read + write memory)
        Xchg => MemoryEffect::Store,
        Cmpxchg => MemoryEffect::Store,

        // Branches / control flow (no memory ops; DCE uses InstFlags)
        Jmp | Jcc | Ret => MemoryEffect::Pure,

        // Pseudo-instructions
        Phi => MemoryEffect::Pure,
        StackAlloc => MemoryEffect::Store,
        Nop | NopMulti => MemoryEffect::Pure,
    }
}

/// Returns true if this x86-64 opcode produces a value (operand[0] is a def).
pub fn x86_produces_value(opcode: X86Opcode) -> bool {
    use X86Opcode::*;
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

/// Returns true if this x86-64 opcode can be safely eliminated if
/// its result is unused.
///
/// x86 is more conservative than AArch64: most arithmetic instructions
/// set RFLAGS as a side effect. However, if a pass can prove the flags
/// are not consumed, the instruction is removable. This function returns
/// the conservative answer (assuming flags may be live).
pub fn x86_is_removable(opcode: X86Opcode) -> bool {
    let effect = x86_opcode_effect(opcode);
    if !effect.is_pure() {
        return false;
    }

    use X86Opcode::*;
    // On x86, arithmetic/logical/shift instructions set RFLAGS.
    // Only register moves, LEA, SSE moves, extensions, conversions,
    // GPR<->XMM transfers, and pseudo-instructions are truly removable.
    matches!(
        opcode,
        MovRR
            | MovRI
            | Movzx
            | MovzxW
            | MovsxB
            | MovsxW
            | Movsx
            | MovsdRR
            | MovssRR
            | Lea
            | LeaSib
            | LeaRip
            | Cvtsi2sd
            | Cvtsd2si
            | Cvtsi2ss
            | Cvtss2si
            | Cvtsd2ss
            | Cvtss2sd
            | MovdToXmm
            | MovdFromXmm
            | MovqToXmm
            | MovqFromXmm
            | Bswap
            | Phi
            | Nop
    )
}

/// Returns true if this x86-64 opcode writes RFLAGS.
///
/// On x86, nearly ALL arithmetic, logical, and shift instructions
/// modify condition flags. This is a fundamental difference from AArch64
/// where only explicit flag-setting instructions (CMP, TST, ADDS, SUBS)
/// modify NZCV.
pub fn x86_writes_flags(opcode: X86Opcode) -> bool {
    use X86Opcode::*;
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

/// Returns true if this x86-64 opcode reads RFLAGS.
///
/// CMOVcc, SETcc, and Jcc all read condition flags to decide behavior.
pub fn x86_reads_flags(opcode: X86Opcode) -> bool {
    use X86Opcode::*;
    matches!(opcode, Cmovcc | Setcc | Jcc)
}

// ===========================================================================
// Target-independent category-based queries
// ===========================================================================

/// Target-independent memory effect classification based on [`OpcodeCategory`].
///
/// This provides a conservative but correct classification for any target.
/// Target-specific functions ([`opcode_effect`], [`x86_opcode_effect`]) provide
/// more precise per-opcode classification but are limited to their target.
///
/// Used by passes that want to reason about opcodes via category alone,
/// enabling multi-target optimization without per-target match arms.
pub fn category_memory_effect(cat: OpcodeCategory) -> MemoryEffect {
    use OpcodeCategory::*;
    match cat {
        Load => MemoryEffect::Load,
        Store => MemoryEffect::Store,
        Call => MemoryEffect::Call,
        // All other categories are pure computation.
        AddRR | AddRI | SubRR | SubRI | MulRR | Neg | AndRR | AndRI | OrRR | OrRI | XorRR
        | XorRI | ShlRR | ShlRI | ShrRR | ShrRI | SarRR | SarRI | MovRR | MovRI | CmpRR | CmpRI
        | Nop | Ret | Branch | CondBranch | Phi | Other => MemoryEffect::Pure,
    }
}

/// Target-independent removability check based on [`OpcodeCategory`].
///
/// An instruction is removable if:
/// 1. Its category has no memory effects (pure).
/// 2. It does not write implicit flags (caller must supply this from
///    [`TargetInfo::writes_flags`]).
/// 3. It is not control flow.
///
/// This is conservative: target-specific functions ([`is_removable`],
/// [`x86_is_removable`]) may be more precise for their respective targets.
pub fn category_is_removable(cat: OpcodeCategory, target_writes_flags: bool) -> bool {
    if !category_memory_effect(cat).is_pure() {
        return false;
    }
    // Compare instructions always set flags — not removable.
    if matches!(cat, OpcodeCategory::CmpRR | OpcodeCategory::CmpRI) {
        return false;
    }
    // If the target says this opcode writes flags, not removable.
    if target_writes_flags {
        return false;
    }
    // Control flow is not removable.
    if matches!(
        cat,
        OpcodeCategory::Branch
            | OpcodeCategory::CondBranch
            | OpcodeCategory::Ret
            | OpcodeCategory::Call
    ) {
        return false;
    }
    true
}

/// Target-independent flag-reading check based on [`OpcodeCategory`].
///
/// Returns `true` for categories that are *known* to read flags.
/// Returns `false` conservatively — actual flag reading depends on the
/// target-specific opcode (e.g., `CSet` on AArch64 reads flags but has
/// category [`OpcodeCategory::Other`]).
pub fn category_reads_flags(cat: OpcodeCategory) -> bool {
    matches!(cat, OpcodeCategory::CondBranch)
}

/// Target-independent flag-writing check based on [`OpcodeCategory`].
///
/// Returns `true` for categories that are *known* to write flags.
/// Compare instructions always write flags on all targets.
pub fn category_writes_flags(cat: OpcodeCategory) -> bool {
    matches!(cat, OpcodeCategory::CmpRR | OpcodeCategory::CmpRI)
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

    #[test]
    fn test_writes_flags() {
        assert!(writes_flags(AArch64Opcode::CmpRR));
        assert!(writes_flags(AArch64Opcode::CmpRI));
        assert!(writes_flags(AArch64Opcode::Tst));
        assert!(writes_flags(AArch64Opcode::Fcmp));
        assert!(writes_flags(AArch64Opcode::AddsRR));
        assert!(writes_flags(AArch64Opcode::SubsRI));
        // Arithmetic and moves should not write flags
        assert!(!writes_flags(AArch64Opcode::AddRR));
        assert!(!writes_flags(AArch64Opcode::MovR));
        assert!(!writes_flags(AArch64Opcode::CSet));
    }

    #[test]
    fn test_reads_flags() {
        assert!(reads_flags(AArch64Opcode::CSet));
        assert!(reads_flags(AArch64Opcode::Csel));
        assert!(reads_flags(AArch64Opcode::Csinc));
        assert!(reads_flags(AArch64Opcode::Csinv));
        assert!(reads_flags(AArch64Opcode::Csneg));
        // CMP writes but does not read flags
        assert!(!reads_flags(AArch64Opcode::CmpRR));
        assert!(!reads_flags(AArch64Opcode::AddRR));
    }

    // Regression for #409: ADC/SBC consume the carry flag implicitly for
    // multi-precision (i128) arithmetic. They must be classified as
    // flag-readers so that CSE/GVN/LICM/scheduler treat them as depending
    // on the most recent flag writer rather than as a free-to-reorder
    // pure function of their explicit operands.
    #[test]
    fn test_reads_flags_adc_sbc() {
        assert!(
            reads_flags(AArch64Opcode::Adc),
            "ADC reads carry implicitly — must be a flag-reader"
        );
        assert!(
            reads_flags(AArch64Opcode::Sbc),
            "SBC reads carry/borrow implicitly — must be a flag-reader"
        );
        // Sanity: the pure-arithmetic siblings do NOT read flags.
        assert!(!reads_flags(AArch64Opcode::AddRR));
        assert!(!reads_flags(AArch64Opcode::SubRR));
        assert!(!reads_flags(AArch64Opcode::Umulh));
        assert!(!reads_flags(AArch64Opcode::Smulh));
        assert!(!reads_flags(AArch64Opcode::Madd));
    }

    // Regression for #408: BFM is a bitfield *insert* that preserves the
    // bits of Rd outside the inserted field. Its prior destination value
    // is an implicit input — same shape as MOVK. Classifying it as tied
    // def-use stops GVN/CSE from folding two BFMs with identical explicit
    // operands but different prior Rd values into a single op.
    #[test]
    fn test_has_tied_def_use_bfm() {
        assert!(
            has_tied_def_use(AArch64Opcode::Bfm),
            "BFM preserves uncovered bits of Rd — must be tied def-use"
        );
        // MOVK remains tied (existing guarantee).
        assert!(has_tied_def_use(AArch64Opcode::Movk));
        // UBFM / SBFM fully redefine Rd (uncovered bits become 0 / sign-ext)
        // so they are NOT tied def-use.
        assert!(!has_tied_def_use(AArch64Opcode::Ubfm));
        assert!(!has_tied_def_use(AArch64Opcode::Sbfm));
        // Sanity: other arithmetic / moves are not tied.
        assert!(!has_tied_def_use(AArch64Opcode::AddRR));
        assert!(!has_tied_def_use(AArch64Opcode::MovR));
        assert!(!has_tied_def_use(AArch64Opcode::Adc));
    }
}

// ===========================================================================
// x86-64 tests
// ===========================================================================

#[cfg(test)]
mod tests_x86 {
    use super::*;
    use llvm2_ir::X86Opcode;

    #[test]
    fn test_x86_arithmetic_is_pure() {
        assert_eq!(x86_opcode_effect(X86Opcode::AddRR), MemoryEffect::Pure);
        assert_eq!(x86_opcode_effect(X86Opcode::SubRI), MemoryEffect::Pure);
        assert_eq!(x86_opcode_effect(X86Opcode::ImulRR), MemoryEffect::Pure);
        assert_eq!(x86_opcode_effect(X86Opcode::Neg), MemoryEffect::Pure);
        assert_eq!(x86_opcode_effect(X86Opcode::Inc), MemoryEffect::Pure);
        assert_eq!(x86_opcode_effect(X86Opcode::Dec), MemoryEffect::Pure);
    }

    #[test]
    fn test_x86_loads_are_load() {
        assert_eq!(x86_opcode_effect(X86Opcode::MovRM), MemoryEffect::Load);
        assert_eq!(x86_opcode_effect(X86Opcode::MovsdRM), MemoryEffect::Load);
        assert_eq!(x86_opcode_effect(X86Opcode::MovssRM), MemoryEffect::Load);
        assert_eq!(x86_opcode_effect(X86Opcode::Pop), MemoryEffect::Load);
        assert_eq!(x86_opcode_effect(X86Opcode::MovRMSib), MemoryEffect::Load);
    }

    #[test]
    fn test_x86_stores_are_store() {
        assert_eq!(x86_opcode_effect(X86Opcode::MovMR), MemoryEffect::Store);
        assert_eq!(x86_opcode_effect(X86Opcode::MovsdMR), MemoryEffect::Store);
        assert_eq!(x86_opcode_effect(X86Opcode::Push), MemoryEffect::Store);
    }

    #[test]
    fn test_x86_calls_are_call() {
        assert_eq!(x86_opcode_effect(X86Opcode::Call), MemoryEffect::Call);
        assert_eq!(x86_opcode_effect(X86Opcode::CallR), MemoryEffect::Call);
        assert_eq!(x86_opcode_effect(X86Opcode::CallM), MemoryEffect::Call);
    }

    #[test]
    fn test_x86_moves_are_pure() {
        assert_eq!(x86_opcode_effect(X86Opcode::MovRR), MemoryEffect::Pure);
        assert_eq!(x86_opcode_effect(X86Opcode::MovRI), MemoryEffect::Pure);
        assert_eq!(x86_opcode_effect(X86Opcode::Lea), MemoryEffect::Pure);
    }

    #[test]
    fn test_x86_produces_value() {
        assert!(x86_produces_value(X86Opcode::AddRR));
        assert!(x86_produces_value(X86Opcode::MovRR));
        assert!(x86_produces_value(X86Opcode::MovRI));
        assert!(x86_produces_value(X86Opcode::Lea));
        assert!(x86_produces_value(X86Opcode::Cmovcc));
        assert!(x86_produces_value(X86Opcode::Setcc));
        assert!(x86_produces_value(X86Opcode::Pop));

        assert!(!x86_produces_value(X86Opcode::CmpRR));
        assert!(!x86_produces_value(X86Opcode::CmpRI));
        assert!(!x86_produces_value(X86Opcode::TestRR));
        assert!(!x86_produces_value(X86Opcode::MovMR));
        assert!(!x86_produces_value(X86Opcode::Push));
        assert!(!x86_produces_value(X86Opcode::Jmp));
        assert!(!x86_produces_value(X86Opcode::Ret));
        assert!(!x86_produces_value(X86Opcode::Call));
        assert!(!x86_produces_value(X86Opcode::Nop));
    }

    #[test]
    fn test_x86_removable() {
        // Moves and LEA are removable (no flag side effects)
        assert!(x86_is_removable(X86Opcode::MovRR));
        assert!(x86_is_removable(X86Opcode::MovRI));
        assert!(x86_is_removable(X86Opcode::Lea));
        assert!(x86_is_removable(X86Opcode::Movzx));

        // Arithmetic is NOT removable (sets RFLAGS)
        assert!(!x86_is_removable(X86Opcode::AddRR));
        assert!(!x86_is_removable(X86Opcode::SubRR));
        assert!(!x86_is_removable(X86Opcode::ImulRR));

        // Memory ops are not removable
        assert!(!x86_is_removable(X86Opcode::MovRM));
        assert!(!x86_is_removable(X86Opcode::MovMR));
        assert!(!x86_is_removable(X86Opcode::Call));
    }

    #[test]
    fn test_x86_writes_flags() {
        // x86 arithmetic sets flags (unlike AArch64)
        assert!(x86_writes_flags(X86Opcode::AddRR));
        assert!(x86_writes_flags(X86Opcode::SubRR));
        assert!(x86_writes_flags(X86Opcode::ImulRR));
        assert!(x86_writes_flags(X86Opcode::Neg));
        assert!(x86_writes_flags(X86Opcode::AndRR));
        assert!(x86_writes_flags(X86Opcode::ShlRI));
        assert!(x86_writes_flags(X86Opcode::CmpRR));
        assert!(x86_writes_flags(X86Opcode::TestRR));

        // Moves and LEA do NOT set flags
        assert!(!x86_writes_flags(X86Opcode::MovRR));
        assert!(!x86_writes_flags(X86Opcode::MovRI));
        assert!(!x86_writes_flags(X86Opcode::Lea));
        assert!(!x86_writes_flags(X86Opcode::Cmovcc));
    }

    #[test]
    fn test_x86_reads_flags() {
        assert!(x86_reads_flags(X86Opcode::Cmovcc));
        assert!(x86_reads_flags(X86Opcode::Setcc));
        assert!(x86_reads_flags(X86Opcode::Jcc));

        assert!(!x86_reads_flags(X86Opcode::AddRR));
        assert!(!x86_reads_flags(X86Opcode::CmpRR));
        assert!(!x86_reads_flags(X86Opcode::MovRR));
    }

    #[test]
    fn test_x86_sse_is_pure() {
        assert_eq!(x86_opcode_effect(X86Opcode::Addsd), MemoryEffect::Pure);
        assert_eq!(x86_opcode_effect(X86Opcode::Mulsd), MemoryEffect::Pure);
        assert_eq!(x86_opcode_effect(X86Opcode::Addss), MemoryEffect::Pure);
        assert_eq!(x86_opcode_effect(X86Opcode::MovsdRR), MemoryEffect::Pure);
        assert_eq!(x86_opcode_effect(X86Opcode::MovssRR), MemoryEffect::Pure);
    }

    #[test]
    fn test_x86_conversion_is_pure() {
        assert_eq!(x86_opcode_effect(X86Opcode::Cvtsi2sd), MemoryEffect::Pure);
        assert_eq!(x86_opcode_effect(X86Opcode::Cvtsd2si), MemoryEffect::Pure);
        assert_eq!(x86_opcode_effect(X86Opcode::Cvtsd2ss), MemoryEffect::Pure);
        assert_eq!(x86_opcode_effect(X86Opcode::Cvtss2sd), MemoryEffect::Pure);
    }
}

// ===========================================================================
// Category-based tests
// ===========================================================================

#[cfg(test)]
mod tests_category {
    use super::*;
    use llvm2_ir::OpcodeCategory;

    #[test]
    fn test_category_memory_effect_load() {
        assert_eq!(
            category_memory_effect(OpcodeCategory::Load),
            MemoryEffect::Load
        );
    }

    #[test]
    fn test_category_memory_effect_store() {
        assert_eq!(
            category_memory_effect(OpcodeCategory::Store),
            MemoryEffect::Store
        );
    }

    #[test]
    fn test_category_memory_effect_call() {
        assert_eq!(
            category_memory_effect(OpcodeCategory::Call),
            MemoryEffect::Call
        );
    }

    #[test]
    fn test_category_memory_effect_pure_arithmetic() {
        assert_eq!(
            category_memory_effect(OpcodeCategory::AddRR),
            MemoryEffect::Pure
        );
        assert_eq!(
            category_memory_effect(OpcodeCategory::SubRI),
            MemoryEffect::Pure
        );
        assert_eq!(
            category_memory_effect(OpcodeCategory::MulRR),
            MemoryEffect::Pure
        );
        assert_eq!(
            category_memory_effect(OpcodeCategory::Neg),
            MemoryEffect::Pure
        );
    }

    #[test]
    fn test_category_memory_effect_pure_logical() {
        assert_eq!(
            category_memory_effect(OpcodeCategory::AndRR),
            MemoryEffect::Pure
        );
        assert_eq!(
            category_memory_effect(OpcodeCategory::OrRI),
            MemoryEffect::Pure
        );
        assert_eq!(
            category_memory_effect(OpcodeCategory::XorRR),
            MemoryEffect::Pure
        );
    }

    #[test]
    fn test_category_memory_effect_pure_moves() {
        assert_eq!(
            category_memory_effect(OpcodeCategory::MovRR),
            MemoryEffect::Pure
        );
        assert_eq!(
            category_memory_effect(OpcodeCategory::MovRI),
            MemoryEffect::Pure
        );
    }

    #[test]
    fn test_category_memory_effect_other_is_pure() {
        // Other is conservatively pure (target-specific, but no known memory effect)
        assert_eq!(
            category_memory_effect(OpcodeCategory::Other),
            MemoryEffect::Pure
        );
    }

    #[test]
    fn test_category_is_removable_pure_no_flags() {
        assert!(category_is_removable(OpcodeCategory::AddRR, false));
        assert!(category_is_removable(OpcodeCategory::MovRR, false));
        assert!(category_is_removable(OpcodeCategory::ShlRI, false));
    }

    #[test]
    fn test_category_is_removable_memory_ops() {
        assert!(!category_is_removable(OpcodeCategory::Load, false));
        assert!(!category_is_removable(OpcodeCategory::Store, false));
        assert!(!category_is_removable(OpcodeCategory::Call, false));
    }

    #[test]
    fn test_category_is_removable_compare_not_removable() {
        assert!(!category_is_removable(OpcodeCategory::CmpRR, false));
        assert!(!category_is_removable(OpcodeCategory::CmpRI, false));
    }

    #[test]
    fn test_category_is_removable_flag_writing_not_removable() {
        // Even if category is pure, if the target says it writes flags, not removable
        assert!(!category_is_removable(OpcodeCategory::AddRR, true));
    }

    #[test]
    fn test_category_is_removable_control_flow() {
        assert!(!category_is_removable(OpcodeCategory::Branch, false));
        assert!(!category_is_removable(OpcodeCategory::CondBranch, false));
        assert!(!category_is_removable(OpcodeCategory::Ret, false));
    }

    #[test]
    fn test_category_reads_flags() {
        assert!(category_reads_flags(OpcodeCategory::CondBranch));
        assert!(!category_reads_flags(OpcodeCategory::AddRR));
        assert!(!category_reads_flags(OpcodeCategory::CmpRR));
        assert!(!category_reads_flags(OpcodeCategory::Other));
    }

    #[test]
    fn test_category_writes_flags() {
        assert!(category_writes_flags(OpcodeCategory::CmpRR));
        assert!(category_writes_flags(OpcodeCategory::CmpRI));
        assert!(!category_writes_flags(OpcodeCategory::AddRR));
        assert!(!category_writes_flags(OpcodeCategory::MovRR));
        assert!(!category_writes_flags(OpcodeCategory::Other));
    }

    // Cross-check: verify category-based classification is consistent
    // with the per-opcode AArch64 classification for categorized opcodes.
    #[test]
    fn test_category_consistent_with_aarch64_for_loads() {
        use llvm2_ir::AArch64Opcode;
        let op = AArch64Opcode::LdrRI;
        let cat = op.categorize();
        assert_eq!(category_memory_effect(cat), opcode_effect(op));
    }

    #[test]
    fn test_category_consistent_with_aarch64_for_stores() {
        use llvm2_ir::AArch64Opcode;
        let op = AArch64Opcode::StrRI;
        let cat = op.categorize();
        assert_eq!(category_memory_effect(cat), opcode_effect(op));
    }

    #[test]
    fn test_category_consistent_with_aarch64_for_calls() {
        use llvm2_ir::AArch64Opcode;
        let op = AArch64Opcode::Bl;
        let cat = op.categorize();
        assert_eq!(category_memory_effect(cat), opcode_effect(op));
    }

    #[test]
    fn test_category_consistent_with_aarch64_for_arithmetic() {
        use llvm2_ir::AArch64Opcode;
        let op = AArch64Opcode::AddRR;
        let cat = op.categorize();
        assert_eq!(category_memory_effect(cat), opcode_effect(op));
    }
}
