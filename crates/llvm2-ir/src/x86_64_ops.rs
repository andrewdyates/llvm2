// llvm2-ir - x86-64 opcode definitions
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: ~/llvm-project-ref/llvm/lib/Target/X86/X86InstrInfo.td
// Reference: Intel 64 and IA-32 Architectures Software Developer's Manual

//! x86-64 instruction opcode enum.
//!
//! Naming convention follows the AArch64 pattern: `<mnemonic><operand_kinds>`
//! where RR = register-register, RI = register-immediate, RM = register-memory,
//! MR = memory-register.

use crate::inst::InstFlags;

// ---------------------------------------------------------------------------
// X86Opcode
// ---------------------------------------------------------------------------

/// x86-64 instruction opcodes.
///
/// This covers the core integer, logical, move, compare, branch, and
/// SSE scalar double instructions needed for tMIR lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum X86Opcode {
    // =====================================================================
    // Arithmetic
    // =====================================================================

    /// ADD r64, r64
    AddRR,
    /// ADD r64, imm32 (sign-extended)
    AddRI,
    /// ADD r64, [mem]
    AddRM,
    /// SUB r64, r64
    SubRR,
    /// SUB r64, imm32 (sign-extended)
    SubRI,
    /// SUB r64, [mem]
    SubRM,
    /// IMUL r64, r64 (signed multiply, two-operand form)
    ImulRR,
    /// IMUL r64, r64, imm32 (signed multiply, three-operand form)
    ImulRRI,
    /// IDIV r64 (signed divide RDX:RAX by r64, quotient in RAX, remainder in RDX)
    Idiv,
    /// NEG r64 (two's complement negate)
    Neg,
    /// INC r64
    Inc,
    /// DEC r64
    Dec,

    // =====================================================================
    // Logical / bitwise
    // =====================================================================

    /// AND r64, r64
    AndRR,
    /// AND r64, imm32
    AndRI,
    /// OR r64, r64
    OrRR,
    /// OR r64, imm32
    OrRI,
    /// XOR r64, r64
    XorRR,
    /// XOR r64, imm32
    XorRI,
    /// NOT r64 (bitwise complement)
    Not,

    // =====================================================================
    // Shifts
    // =====================================================================

    /// SHL r64, CL (shift left by CL register)
    ShlRR,
    /// SHL r64, imm8
    ShlRI,
    /// SHR r64, CL (logical shift right by CL)
    ShrRR,
    /// SHR r64, imm8
    ShrRI,
    /// SAR r64, CL (arithmetic shift right by CL)
    SarRR,
    /// SAR r64, imm8
    SarRI,

    // =====================================================================
    // Move
    // =====================================================================

    /// MOV r64, r64
    MovRR,
    /// MOV r64, imm64 (movabs for 64-bit immediates)
    MovRI,
    /// MOV r64, [mem]
    MovRM,
    /// MOV [mem], r64
    MovMR,
    /// MOVZX r64, r8/r16 (zero-extend)
    Movzx,
    /// MOVSX r64, r8/r16/r32 (sign-extend, aka MOVSXD for 32->64)
    Movsx,
    /// LEA r64, [mem] (load effective address)
    Lea,

    // =====================================================================
    // Compare / test
    // =====================================================================

    /// CMP r64, r64 (sets RFLAGS)
    CmpRR,
    /// CMP r64, imm32 (sets RFLAGS)
    CmpRI,
    /// CMP r64, [mem] (sets RFLAGS)
    CmpRM,
    /// TEST r64, r64 (AND without storing result, sets RFLAGS)
    TestRR,
    /// TEST r64, imm32
    TestRI,

    // =====================================================================
    // Branch / control flow
    // =====================================================================

    /// JMP rel32 (unconditional near jump)
    Jmp,
    /// Jcc rel32 (conditional near jump based on RFLAGS)
    Jcc,
    /// CALL rel32 (near call)
    Call,
    /// CALL r64 (indirect call)
    CallR,
    /// RET (near return)
    Ret,

    // =====================================================================
    // SSE scalar double-precision
    // =====================================================================

    /// ADDSD xmm, xmm (scalar double add)
    Addsd,
    /// SUBSD xmm, xmm (scalar double subtract)
    Subsd,
    /// MULSD xmm, xmm (scalar double multiply)
    Mulsd,
    /// DIVSD xmm, xmm (scalar double divide)
    Divsd,
    /// MOVSD xmm, xmm (scalar double move)
    MovsdRR,
    /// MOVSD xmm, [mem] (scalar double load)
    MovsdRM,
    /// MOVSD [mem], xmm (scalar double store)
    MovsdMR,
    /// UCOMISD xmm, xmm (unordered compare scalar double, sets RFLAGS)
    Ucomisd,

    // =====================================================================
    // SSE scalar single-precision
    // =====================================================================

    /// ADDSS xmm, xmm (scalar single add)
    Addss,
    /// SUBSS xmm, xmm (scalar single subtract)
    Subss,
    /// MULSS xmm, xmm (scalar single multiply)
    Mulss,
    /// DIVSS xmm, xmm (scalar single divide)
    Divss,
    /// MOVSS xmm, xmm (scalar single move)
    MovssRR,
    /// MOVSS xmm, [mem] (scalar single load)
    MovssRM,
    /// MOVSS [mem], xmm (scalar single store)
    MovssMR,
    /// UCOMISS xmm, xmm (unordered compare scalar single, sets RFLAGS)
    Ucomiss,

    // =====================================================================
    // Conditional move / set
    // =====================================================================

    /// CMOVcc r64, r64 (conditional move based on RFLAGS)
    Cmovcc,
    /// SETcc r8 (set byte based on RFLAGS condition)
    Setcc,

    // =====================================================================
    // Bit manipulation
    // =====================================================================

    /// BSF r64, r64 (bit scan forward — find lowest set bit)
    Bsf,
    /// BSR r64, r64 (bit scan reverse — find highest set bit)
    Bsr,
    /// TZCNT r64, r64 (trailing zero count — BMI1)
    Tzcnt,
    /// LZCNT r64, r64 (leading zero count — ABM/LZCNT)
    Lzcnt,
    /// POPCNT r64, r64 (population count — POPCNT)
    Popcnt,

    // =====================================================================
    // Stack
    // =====================================================================

    /// PUSH r64
    Push,
    /// POP r64
    Pop,

    // =====================================================================
    // Pseudo-instructions (no hardware encoding)
    // =====================================================================

    /// PHI node (SSA merge point).
    Phi,
    /// Stack allocation pseudo (allocates local stack space).
    StackAlloc,
    /// No-op.
    Nop,
}

impl X86Opcode {
    /// Returns the default instruction flags for this opcode.
    pub fn default_flags(self) -> InstFlags {
        use X86Opcode::*;
        match self {
            // Unconditional branch
            Jmp => InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),
            // Conditional branch
            Jcc => InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),

            // Calls
            Call => InstFlags::IS_CALL.union(InstFlags::HAS_SIDE_EFFECTS),
            CallR => InstFlags::IS_CALL.union(InstFlags::HAS_SIDE_EFFECTS),

            // Return
            Ret => InstFlags::IS_RETURN.union(InstFlags::IS_TERMINATOR),

            // Memory loads
            MovRM | MovsdRM | MovssRM | AddRM | SubRM | CmpRM => InstFlags::READS_MEMORY,

            // Memory stores
            MovMR | MovsdMR | MovssMR => InstFlags::WRITES_MEMORY.union(InstFlags::HAS_SIDE_EFFECTS),

            // Compare/test (set RFLAGS = side effect)
            CmpRR | CmpRI | TestRR | TestRI | Ucomisd | Ucomiss => InstFlags::HAS_SIDE_EFFECTS,

            // SETcc sets RFLAGS-dependent byte (reads RFLAGS)
            Setcc => InstFlags::EMPTY,

            // IDIV has implicit operands (RDX:RAX) and can trap on division by zero
            Idiv => InstFlags::HAS_SIDE_EFFECTS,

            // Stack manipulation (modifies RSP)
            Push => InstFlags::WRITES_MEMORY.union(InstFlags::HAS_SIDE_EFFECTS),
            Pop => InstFlags::READS_MEMORY.union(InstFlags::HAS_SIDE_EFFECTS),

            // Pseudo-instructions
            Phi => InstFlags::IS_PSEUDO,
            StackAlloc => InstFlags::IS_PSEUDO.union(InstFlags::HAS_SIDE_EFFECTS),
            Nop => InstFlags::IS_PSEUDO,

            // Everything else: pure computation
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
// x86-64 Condition Codes (for Jcc, SETcc, CMOVcc)
// ---------------------------------------------------------------------------
// Reference: Intel SDM Vol 2A, Appendix B (Jcc encoding)

/// x86-64 condition code for conditional jumps, sets, and moves.
///
/// The 4-bit encoding matches the hardware encoding used in Jcc/SETcc/CMOVcc
/// opcode bytes (0F 80+cc through 0F 8F+cc).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum X86CondCode {
    /// Overflow (OF=1)
    O = 0x0,
    /// No overflow (OF=0)
    NO = 0x1,
    /// Below / carry (CF=1) — unsigned less than
    B = 0x2,
    /// Above or equal / no carry (CF=0) — unsigned greater or equal
    AE = 0x3,
    /// Equal / zero (ZF=1)
    E = 0x4,
    /// Not equal / not zero (ZF=0)
    NE = 0x5,
    /// Below or equal (CF=1 or ZF=1) — unsigned less or equal
    BE = 0x6,
    /// Above (CF=0 and ZF=0) — unsigned greater
    A = 0x7,
    /// Sign / negative (SF=1)
    S = 0x8,
    /// No sign / positive (SF=0)
    NS = 0x9,
    /// Parity even (PF=1)
    P = 0xA,
    /// Parity odd (PF=0)
    NP = 0xB,
    /// Less than (SF!=OF) — signed less than
    L = 0xC,
    /// Greater or equal (SF=OF) — signed greater or equal
    GE = 0xD,
    /// Less or equal (ZF=1 or SF!=OF) — signed less or equal
    LE = 0xE,
    /// Greater (ZF=0 and SF=OF) — signed greater than
    G = 0xF,
}

impl X86CondCode {
    /// Return the 4-bit hardware encoding.
    #[inline]
    pub const fn encoding(self) -> u8 {
        self as u8
    }

    /// Invert the condition (logical negation).
    ///
    /// Flipping bit 0 of the encoding inverts the condition.
    #[inline]
    pub const fn invert(self) -> Self {
        let inv = (self as u8) ^ 1;
        // SAFETY: all 16 values 0x0..=0xF are valid X86CondCode variants.
        unsafe { core::mem::transmute::<u8, X86CondCode>(inv) }
    }

    /// Return the assembly mnemonic suffix.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::O => "o",
            Self::NO => "no",
            Self::B => "b",
            Self::AE => "ae",
            Self::E => "e",
            Self::NE => "ne",
            Self::BE => "be",
            Self::A => "a",
            Self::S => "s",
            Self::NS => "ns",
            Self::P => "p",
            Self::NP => "np",
            Self::L => "l",
            Self::GE => "ge",
            Self::LE => "le",
            Self::G => "g",
        }
    }

    /// Return `true` if this is a signed comparison condition.
    #[inline]
    pub const fn is_signed(self) -> bool {
        matches!(self, Self::L | Self::GE | Self::LE | Self::G)
    }

    /// Return `true` if this is an unsigned comparison condition.
    #[inline]
    pub const fn is_unsigned(self) -> bool {
        matches!(self, Self::B | Self::AE | Self::BE | Self::A)
    }
}

impl core::fmt::Display for X86CondCode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branch_opcodes_have_branch_and_terminator_flags() {
        let flags = X86Opcode::Jmp.default_flags();
        assert!(flags.contains(InstFlags::IS_BRANCH));
        assert!(flags.contains(InstFlags::IS_TERMINATOR));

        let flags = X86Opcode::Jcc.default_flags();
        assert!(flags.contains(InstFlags::IS_BRANCH));
        assert!(flags.contains(InstFlags::IS_TERMINATOR));
    }

    #[test]
    fn call_opcodes_have_call_and_side_effect_flags() {
        for op in &[X86Opcode::Call, X86Opcode::CallR] {
            let flags = op.default_flags();
            assert!(flags.contains(InstFlags::IS_CALL), "{:?}", op);
            assert!(flags.contains(InstFlags::HAS_SIDE_EFFECTS), "{:?}", op);
        }
    }

    #[test]
    fn ret_has_return_and_terminator() {
        let flags = X86Opcode::Ret.default_flags();
        assert!(flags.contains(InstFlags::IS_RETURN));
        assert!(flags.contains(InstFlags::IS_TERMINATOR));
    }

    #[test]
    fn memory_load_opcodes() {
        for op in &[X86Opcode::MovRM, X86Opcode::MovsdRM, X86Opcode::MovssRM] {
            let flags = op.default_flags();
            assert!(flags.contains(InstFlags::READS_MEMORY), "{:?}", op);
            assert!(!flags.contains(InstFlags::WRITES_MEMORY), "{:?}", op);
        }
    }

    #[test]
    fn memory_store_opcodes() {
        for op in &[X86Opcode::MovMR, X86Opcode::MovsdMR, X86Opcode::MovssMR] {
            let flags = op.default_flags();
            assert!(flags.contains(InstFlags::WRITES_MEMORY), "{:?}", op);
            assert!(flags.contains(InstFlags::HAS_SIDE_EFFECTS), "{:?}", op);
        }
    }

    #[test]
    fn compare_opcodes_have_side_effects() {
        for op in &[X86Opcode::CmpRR, X86Opcode::CmpRI, X86Opcode::TestRR, X86Opcode::TestRI, X86Opcode::Ucomisd, X86Opcode::Ucomiss] {
            let flags = op.default_flags();
            assert!(flags.contains(InstFlags::HAS_SIDE_EFFECTS), "{:?}", op);
        }
    }

    #[test]
    fn pure_arithmetic_has_empty_flags() {
        let pure_ops = [
            X86Opcode::AddRR, X86Opcode::AddRI,
            X86Opcode::SubRR, X86Opcode::SubRI,
            X86Opcode::ImulRR, X86Opcode::ImulRRI,
            X86Opcode::Neg, X86Opcode::Inc, X86Opcode::Dec,
            X86Opcode::AndRR, X86Opcode::AndRI,
            X86Opcode::OrRR, X86Opcode::OrRI,
            X86Opcode::XorRR, X86Opcode::XorRI,
            X86Opcode::Not,
            X86Opcode::ShlRR, X86Opcode::ShlRI,
            X86Opcode::ShrRR, X86Opcode::ShrRI,
            X86Opcode::SarRR, X86Opcode::SarRI,
            X86Opcode::MovRR, X86Opcode::MovRI,
            X86Opcode::Movzx, X86Opcode::Movsx, X86Opcode::Lea,
            X86Opcode::Addsd, X86Opcode::Subsd,
            X86Opcode::Mulsd, X86Opcode::Divsd,
            X86Opcode::MovsdRR,
            // SSE single-precision
            X86Opcode::Addss, X86Opcode::Subss,
            X86Opcode::Mulss, X86Opcode::Divss,
            X86Opcode::MovssRR,
            // Conditional move, SETcc, bit manipulation
            X86Opcode::Cmovcc, X86Opcode::Setcc,
            X86Opcode::Bsf, X86Opcode::Bsr,
            X86Opcode::Tzcnt, X86Opcode::Lzcnt, X86Opcode::Popcnt,
        ];
        for op in &pure_ops {
            assert!(op.default_flags().is_empty(), "{:?} should have EMPTY flags", op);
        }
    }

    #[test]
    fn pseudo_opcodes() {
        assert!(X86Opcode::Phi.is_pseudo());
        assert!(X86Opcode::StackAlloc.is_pseudo());
        assert!(X86Opcode::Nop.is_pseudo());
        assert!(!X86Opcode::AddRR.is_pseudo());
    }

    #[test]
    fn is_phi_method() {
        assert!(X86Opcode::Phi.is_phi());
        assert!(!X86Opcode::Nop.is_phi());
    }

    // ---- X86CondCode tests ----

    #[test]
    fn cond_code_encoding() {
        assert_eq!(X86CondCode::O.encoding(), 0x0);
        assert_eq!(X86CondCode::NO.encoding(), 0x1);
        assert_eq!(X86CondCode::B.encoding(), 0x2);
        assert_eq!(X86CondCode::AE.encoding(), 0x3);
        assert_eq!(X86CondCode::E.encoding(), 0x4);
        assert_eq!(X86CondCode::NE.encoding(), 0x5);
        assert_eq!(X86CondCode::BE.encoding(), 0x6);
        assert_eq!(X86CondCode::A.encoding(), 0x7);
        assert_eq!(X86CondCode::S.encoding(), 0x8);
        assert_eq!(X86CondCode::NS.encoding(), 0x9);
        assert_eq!(X86CondCode::P.encoding(), 0xA);
        assert_eq!(X86CondCode::NP.encoding(), 0xB);
        assert_eq!(X86CondCode::L.encoding(), 0xC);
        assert_eq!(X86CondCode::GE.encoding(), 0xD);
        assert_eq!(X86CondCode::LE.encoding(), 0xE);
        assert_eq!(X86CondCode::G.encoding(), 0xF);
    }

    #[test]
    fn cond_code_invert() {
        assert_eq!(X86CondCode::O.invert(), X86CondCode::NO);
        assert_eq!(X86CondCode::NO.invert(), X86CondCode::O);
        assert_eq!(X86CondCode::B.invert(), X86CondCode::AE);
        assert_eq!(X86CondCode::AE.invert(), X86CondCode::B);
        assert_eq!(X86CondCode::E.invert(), X86CondCode::NE);
        assert_eq!(X86CondCode::NE.invert(), X86CondCode::E);
        assert_eq!(X86CondCode::L.invert(), X86CondCode::GE);
        assert_eq!(X86CondCode::GE.invert(), X86CondCode::L);
        assert_eq!(X86CondCode::LE.invert(), X86CondCode::G);
        assert_eq!(X86CondCode::G.invert(), X86CondCode::LE);
    }

    #[test]
    fn cond_code_double_invert_is_identity() {
        let all = [
            X86CondCode::O, X86CondCode::NO, X86CondCode::B, X86CondCode::AE,
            X86CondCode::E, X86CondCode::NE, X86CondCode::BE, X86CondCode::A,
            X86CondCode::S, X86CondCode::NS, X86CondCode::P, X86CondCode::NP,
            X86CondCode::L, X86CondCode::GE, X86CondCode::LE, X86CondCode::G,
        ];
        for cc in &all {
            assert_eq!(cc.invert().invert(), *cc, "double invert identity for {:?}", cc);
        }
    }

    #[test]
    fn cond_code_signed_unsigned() {
        assert!(X86CondCode::L.is_signed());
        assert!(X86CondCode::GE.is_signed());
        assert!(X86CondCode::LE.is_signed());
        assert!(X86CondCode::G.is_signed());
        assert!(!X86CondCode::E.is_signed());
        assert!(!X86CondCode::B.is_signed());

        assert!(X86CondCode::B.is_unsigned());
        assert!(X86CondCode::AE.is_unsigned());
        assert!(X86CondCode::BE.is_unsigned());
        assert!(X86CondCode::A.is_unsigned());
        assert!(!X86CondCode::L.is_unsigned());
        assert!(!X86CondCode::E.is_unsigned());
    }

    #[test]
    fn cond_code_display() {
        assert_eq!(format!("{}", X86CondCode::E), "e");
        assert_eq!(format!("{}", X86CondCode::NE), "ne");
        assert_eq!(format!("{}", X86CondCode::G), "g");
    }
}
