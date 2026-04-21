// llvm2-ir - RISC-V opcode definitions
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: RISC-V Unprivileged ISA Specification (Volume 1, Version 20191213)
// Reference: RISC-V "M" Standard Extension for Integer Multiply/Divide
// Reference: RISC-V "F"/"D" Standard Extensions for Single/Double FP

//! RISC-V RV64GC instruction opcode enum.
//!
//! Covers RV64I base integer instructions, M extension (multiply/divide),
//! and D extension (double-precision floating-point). Naming follows the
//! RISC-V ISA mnemonic directly (no operand-kind suffixes like x86) since
//! RISC-V instructions have fixed formats.

use crate::inst::InstFlags;

// ---------------------------------------------------------------------------
// RiscVOpcode
// ---------------------------------------------------------------------------

/// RISC-V instruction opcodes for RV64GC.
///
/// Organized by ISA extension and functional group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RiscVOpcode {
    // =====================================================================
    // RV64I: Integer Register-Register
    // =====================================================================

    /// ADD rd, rs1, rs2
    Add,
    /// SUB rd, rs1, rs2
    Sub,
    /// AND rd, rs1, rs2
    And,
    /// OR rd, rs1, rs2
    Or,
    /// XOR rd, rs1, rs2
    Xor,
    /// SLL rd, rs1, rs2 (shift left logical)
    Sll,
    /// SRL rd, rs1, rs2 (shift right logical)
    Srl,
    /// SRA rd, rs1, rs2 (shift right arithmetic)
    Sra,
    /// SLT rd, rs1, rs2 (set less than, signed)
    Slt,
    /// SLTU rd, rs1, rs2 (set less than, unsigned)
    Sltu,

    // =====================================================================
    // RV64I: Integer Register-Immediate
    // =====================================================================

    /// ADDI rd, rs1, imm12
    Addi,
    /// ANDI rd, rs1, imm12
    Andi,
    /// ORI rd, rs1, imm12
    Ori,
    /// XORI rd, rs1, imm12
    Xori,
    /// SLTI rd, rs1, imm12 (set less than immediate, signed)
    Slti,
    /// SLTIU rd, rs1, imm12 (set less than immediate, unsigned)
    Sltiu,
    /// SLLI rd, rs1, shamt (shift left logical immediate)
    Slli,
    /// SRLI rd, rs1, shamt (shift right logical immediate)
    Srli,
    /// SRAI rd, rs1, shamt (shift right arithmetic immediate)
    Srai,

    // =====================================================================
    // RV64I: Upper Immediate
    // =====================================================================

    /// LUI rd, imm20 (load upper immediate)
    Lui,
    /// AUIPC rd, imm20 (add upper immediate to PC)
    Auipc,

    // =====================================================================
    // RV64I: Word (32-bit) operations on RV64
    // =====================================================================

    /// ADDW rd, rs1, rs2 (32-bit add, sign-extend result to 64 bits)
    Addw,
    /// SUBW rd, rs1, rs2 (32-bit subtract, sign-extend result)
    Subw,
    /// SLLW rd, rs1, rs2 (32-bit shift left logical)
    Sllw,
    /// SRLW rd, rs1, rs2 (32-bit shift right logical)
    Srlw,
    /// SRAW rd, rs1, rs2 (32-bit shift right arithmetic)
    Sraw,
    /// ADDIW rd, rs1, imm12 (32-bit add immediate, sign-extend result)
    Addiw,
    /// SLLIW rd, rs1, shamt (32-bit shift left logical immediate)
    Slliw,
    /// SRLIW rd, rs1, shamt (32-bit shift right logical immediate)
    Srliw,
    /// SRAIW rd, rs1, shamt (32-bit shift right arithmetic immediate)
    Sraiw,

    // =====================================================================
    // RV64I: Load
    // =====================================================================

    /// LB rd, offset(rs1) (load byte, sign-extend)
    Lb,
    /// LH rd, offset(rs1) (load halfword, sign-extend)
    Lh,
    /// LW rd, offset(rs1) (load word, sign-extend)
    Lw,
    /// LD rd, offset(rs1) (load doubleword)
    Ld,
    /// LBU rd, offset(rs1) (load byte, zero-extend)
    Lbu,
    /// LHU rd, offset(rs1) (load halfword, zero-extend)
    Lhu,
    /// LWU rd, offset(rs1) (load word, zero-extend)
    Lwu,

    // =====================================================================
    // RV64I: Store
    // =====================================================================

    /// SB rs2, offset(rs1) (store byte)
    Sb,
    /// SH rs2, offset(rs1) (store halfword)
    Sh,
    /// SW rs2, offset(rs1) (store word)
    Sw,
    /// SD rs2, offset(rs1) (store doubleword)
    Sd,

    // =====================================================================
    // RV64I: Branch
    // =====================================================================

    /// BEQ rs1, rs2, offset (branch if equal)
    Beq,
    /// BNE rs1, rs2, offset (branch if not equal)
    Bne,
    /// BLT rs1, rs2, offset (branch if less than, signed)
    Blt,
    /// BGE rs1, rs2, offset (branch if greater or equal, signed)
    Bge,
    /// BLTU rs1, rs2, offset (branch if less than, unsigned)
    Bltu,
    /// BGEU rs1, rs2, offset (branch if greater or equal, unsigned)
    Bgeu,

    // =====================================================================
    // RV64I: Jump
    // =====================================================================

    /// JAL rd, offset (jump and link)
    Jal,
    /// JALR rd, rs1, offset (jump and link register)
    Jalr,

    // =====================================================================
    // RV64M: Multiply / Divide
    // =====================================================================

    /// MUL rd, rs1, rs2 (multiply, low 64 bits)
    Mul,
    /// MULH rd, rs1, rs2 (multiply high, signed x signed)
    Mulh,
    /// MULHSU rd, rs1, rs2 (multiply high, signed x unsigned)
    Mulhsu,
    /// MULHU rd, rs1, rs2 (multiply high, unsigned x unsigned)
    Mulhu,
    /// DIV rd, rs1, rs2 (signed divide)
    Div,
    /// DIVU rd, rs1, rs2 (unsigned divide)
    Divu,
    /// REM rd, rs1, rs2 (signed remainder)
    Rem,
    /// REMU rd, rs1, rs2 (unsigned remainder)
    Remu,

    // RV64M: Word (32-bit) multiply/divide
    /// MULW rd, rs1, rs2 (32-bit multiply, sign-extend result)
    Mulw,
    /// DIVW rd, rs1, rs2 (32-bit signed divide, sign-extend result)
    Divw,
    /// DIVUW rd, rs1, rs2 (32-bit unsigned divide, sign-extend result)
    Divuw,
    /// REMW rd, rs1, rs2 (32-bit signed remainder, sign-extend result)
    Remw,
    /// REMUW rd, rs1, rs2 (32-bit unsigned remainder, sign-extend result)
    Remuw,

    // =====================================================================
    // RV64D: Double-Precision Floating-Point
    // =====================================================================

    /// FADD.D rd, rs1, rs2 (double add)
    FaddD,
    /// FSUB.D rd, rs1, rs2 (double subtract)
    FsubD,
    /// FMUL.D rd, rs1, rs2 (double multiply)
    FmulD,
    /// FDIV.D rd, rs1, rs2 (double divide)
    FdivD,
    /// FSQRT.D rd, rs1 (double square root)
    FsqrtD,

    /// FLD rd, offset(rs1) (load double from memory)
    Fld,
    /// FSD rs2, offset(rs1) (store double to memory)
    Fsd,

    /// FEQ.D rd, rs1, rs2 (double equal comparison, result in GPR)
    FeqD,
    /// FLT.D rd, rs1, rs2 (double less-than comparison, result in GPR)
    FltD,
    /// FLE.D rd, rs1, rs2 (double less-or-equal comparison, result in GPR)
    FleD,

    /// FCVT.D.W rd, rs1 (convert signed 32-bit int to double)
    FcvtDW,
    /// FCVT.W.D rd, rs1 (convert double to signed 32-bit int)
    FcvtWD,
    /// FCVT.D.L rd, rs1 (convert signed 64-bit int to double)
    FcvtDL,
    /// FCVT.L.D rd, rs1 (convert double to signed 64-bit int)
    FcvtLD,

    /// FMV.X.D rd, rs1 (bitwise move FPR to GPR)
    FmvXD,
    /// FMV.D.X rd, rs1 (bitwise move GPR to FPR)
    FmvDX,

    // =====================================================================
    // Pseudo-instructions (no hardware encoding)
    // =====================================================================

    /// PHI node (SSA merge point).
    Phi,
    /// Stack allocation pseudo (allocates local stack space).
    StackAlloc,
    /// No-op (ADDI x0, x0, 0 in hardware but tracked separately).
    Nop,
}

impl RiscVOpcode {
    /// Returns the default instruction flags for this opcode.
    pub fn default_flags(self) -> InstFlags {
        use RiscVOpcode::*;
        match self {
            // Conditional branches (all B-type)
            Beq | Bne | Blt | Bge | Bltu | Bgeu => {
                InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR)
            }

            // Unconditional jump: JAL with rd=x0 is a plain jump (terminator),
            // JAL with rd!=x0 is a call. We model JAL as branch+terminator;
            // the call variant is distinguished by operand analysis.
            Jal => InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),

            // JALR is used for both indirect calls and returns.
            // When rd=ra it is a call; when rd=x0,rs1=ra it is a return.
            // We model it as a call by default since that is the common case.
            Jalr => InstFlags::IS_CALL.union(InstFlags::HAS_SIDE_EFFECTS),

            // Integer loads (I-type)
            Lb | Lh | Lw | Ld | Lbu | Lhu | Lwu => InstFlags::READS_MEMORY,

            // FP load
            Fld => InstFlags::READS_MEMORY,

            // Integer stores (S-type)
            Sb | Sh | Sw | Sd => InstFlags::WRITES_MEMORY.union(InstFlags::HAS_SIDE_EFFECTS),

            // FP store
            Fsd => InstFlags::WRITES_MEMORY.union(InstFlags::HAS_SIDE_EFFECTS),

            // Division can trap on division by zero (implementation-defined on RISC-V,
            // but we mark as side-effecting for safety)
            Div | Divu | Rem | Remu | Divw | Divuw | Remw | Remuw => {
                InstFlags::HAS_SIDE_EFFECTS
            }

            // Pseudo-instructions
            Phi => InstFlags::IS_PSEUDO,
            StackAlloc => InstFlags::IS_PSEUDO.union(InstFlags::HAS_SIDE_EFFECTS),
            Nop => InstFlags::IS_PSEUDO,

            // Everything else: pure computation (arithmetic, logical, shifts,
            // multiply, FP arithmetic, FP comparisons, conversions, moves)
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

    /// Returns true if this is a branch instruction (conditional or unconditional).
    pub fn is_branch(self) -> bool {
        use RiscVOpcode::*;
        matches!(self, Beq | Bne | Blt | Bge | Bltu | Bgeu | Jal)
    }

    /// Returns true if this is a load instruction.
    pub fn is_load(self) -> bool {
        use RiscVOpcode::*;
        matches!(self, Lb | Lh | Lw | Ld | Lbu | Lhu | Lwu | Fld)
    }

    /// Returns true if this is a store instruction.
    pub fn is_store(self) -> bool {
        use RiscVOpcode::*;
        matches!(self, Sb | Sh | Sw | Sd | Fsd)
    }

    /// Returns true if this is a 32-bit word operation (W-suffix).
    pub fn is_word_op(self) -> bool {
        use RiscVOpcode::*;
        matches!(
            self,
            Addw | Subw | Sllw | Srlw | Sraw | Addiw | Slliw | Srliw | Sraiw
                | Mulw | Divw | Divuw | Remw | Remuw
        )
    }

    /// Returns true if this is a floating-point instruction.
    pub fn is_fp(self) -> bool {
        use RiscVOpcode::*;
        matches!(
            self,
            FaddD | FsubD | FmulD | FdivD | FsqrtD | Fld | Fsd
                | FeqD | FltD | FleD
                | FcvtDW | FcvtWD | FcvtDL | FcvtLD
                | FmvXD | FmvDX
        )
    }

    /// Returns true if this is from the M extension (multiply/divide).
    pub fn is_m_extension(self) -> bool {
        use RiscVOpcode::*;
        matches!(
            self,
            Mul | Mulh | Mulhsu | Mulhu | Div | Divu | Rem | Remu
                | Mulw | Divw | Divuw | Remw | Remuw
        )
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
        let branches = [
            RiscVOpcode::Beq,
            RiscVOpcode::Bne,
            RiscVOpcode::Blt,
            RiscVOpcode::Bge,
            RiscVOpcode::Bltu,
            RiscVOpcode::Bgeu,
            RiscVOpcode::Jal,
        ];
        for op in &branches {
            let flags = op.default_flags();
            assert!(flags.contains(InstFlags::IS_BRANCH), "{:?}", op);
            assert!(flags.contains(InstFlags::IS_TERMINATOR), "{:?}", op);
        }
    }

    #[test]
    fn jalr_has_call_and_side_effect_flags() {
        let flags = RiscVOpcode::Jalr.default_flags();
        assert!(flags.contains(InstFlags::IS_CALL));
        assert!(flags.contains(InstFlags::HAS_SIDE_EFFECTS));
    }

    #[test]
    fn integer_load_opcodes() {
        let loads = [
            RiscVOpcode::Lb,
            RiscVOpcode::Lh,
            RiscVOpcode::Lw,
            RiscVOpcode::Ld,
            RiscVOpcode::Lbu,
            RiscVOpcode::Lhu,
            RiscVOpcode::Lwu,
        ];
        for op in &loads {
            let flags = op.default_flags();
            assert!(flags.contains(InstFlags::READS_MEMORY), "{:?}", op);
            assert!(!flags.contains(InstFlags::WRITES_MEMORY), "{:?}", op);
        }
    }

    #[test]
    fn fp_load_opcode() {
        let flags = RiscVOpcode::Fld.default_flags();
        assert!(flags.contains(InstFlags::READS_MEMORY));
        assert!(!flags.contains(InstFlags::WRITES_MEMORY));
    }

    #[test]
    fn integer_store_opcodes() {
        let stores = [
            RiscVOpcode::Sb,
            RiscVOpcode::Sh,
            RiscVOpcode::Sw,
            RiscVOpcode::Sd,
        ];
        for op in &stores {
            let flags = op.default_flags();
            assert!(flags.contains(InstFlags::WRITES_MEMORY), "{:?}", op);
            assert!(flags.contains(InstFlags::HAS_SIDE_EFFECTS), "{:?}", op);
        }
    }

    #[test]
    fn fp_store_opcode() {
        let flags = RiscVOpcode::Fsd.default_flags();
        assert!(flags.contains(InstFlags::WRITES_MEMORY));
        assert!(flags.contains(InstFlags::HAS_SIDE_EFFECTS));
    }

    #[test]
    fn division_opcodes_have_side_effects() {
        let divs = [
            RiscVOpcode::Div,
            RiscVOpcode::Divu,
            RiscVOpcode::Rem,
            RiscVOpcode::Remu,
            RiscVOpcode::Divw,
            RiscVOpcode::Divuw,
            RiscVOpcode::Remw,
            RiscVOpcode::Remuw,
        ];
        for op in &divs {
            let flags = op.default_flags();
            assert!(flags.contains(InstFlags::HAS_SIDE_EFFECTS), "{:?}", op);
        }
    }

    #[test]
    fn pure_arithmetic_has_empty_flags() {
        let pure_ops = [
            // R-type integer
            RiscVOpcode::Add,
            RiscVOpcode::Sub,
            RiscVOpcode::And,
            RiscVOpcode::Or,
            RiscVOpcode::Xor,
            RiscVOpcode::Sll,
            RiscVOpcode::Srl,
            RiscVOpcode::Sra,
            RiscVOpcode::Slt,
            RiscVOpcode::Sltu,
            // I-type integer
            RiscVOpcode::Addi,
            RiscVOpcode::Andi,
            RiscVOpcode::Ori,
            RiscVOpcode::Xori,
            RiscVOpcode::Slti,
            RiscVOpcode::Sltiu,
            RiscVOpcode::Slli,
            RiscVOpcode::Srli,
            RiscVOpcode::Srai,
            // Upper immediate
            RiscVOpcode::Lui,
            RiscVOpcode::Auipc,
            // W-type integer
            RiscVOpcode::Addw,
            RiscVOpcode::Subw,
            RiscVOpcode::Sllw,
            RiscVOpcode::Srlw,
            RiscVOpcode::Sraw,
            RiscVOpcode::Addiw,
            RiscVOpcode::Slliw,
            RiscVOpcode::Srliw,
            RiscVOpcode::Sraiw,
            // Multiply (non-division)
            RiscVOpcode::Mul,
            RiscVOpcode::Mulh,
            RiscVOpcode::Mulhsu,
            RiscVOpcode::Mulhu,
            RiscVOpcode::Mulw,
            // FP arithmetic
            RiscVOpcode::FaddD,
            RiscVOpcode::FsubD,
            RiscVOpcode::FmulD,
            RiscVOpcode::FdivD,
            RiscVOpcode::FsqrtD,
            // FP comparisons (write to GPR, no side effects)
            RiscVOpcode::FeqD,
            RiscVOpcode::FltD,
            RiscVOpcode::FleD,
            // FP conversions
            RiscVOpcode::FcvtDW,
            RiscVOpcode::FcvtWD,
            RiscVOpcode::FcvtDL,
            RiscVOpcode::FcvtLD,
            // FP moves
            RiscVOpcode::FmvXD,
            RiscVOpcode::FmvDX,
        ];
        for op in &pure_ops {
            assert!(
                op.default_flags().is_empty(),
                "{:?} should have EMPTY flags but has {:?}",
                op,
                op.default_flags()
            );
        }
    }

    #[test]
    fn pseudo_opcodes() {
        assert!(RiscVOpcode::Phi.is_pseudo());
        assert!(RiscVOpcode::StackAlloc.is_pseudo());
        assert!(RiscVOpcode::Nop.is_pseudo());
        assert!(!RiscVOpcode::Add.is_pseudo());
    }

    #[test]
    fn is_phi_method() {
        assert!(RiscVOpcode::Phi.is_phi());
        assert!(!RiscVOpcode::Nop.is_phi());
    }

    #[test]
    fn is_branch_method() {
        assert!(RiscVOpcode::Beq.is_branch());
        assert!(RiscVOpcode::Bne.is_branch());
        assert!(RiscVOpcode::Jal.is_branch());
        assert!(!RiscVOpcode::Jalr.is_branch());
        assert!(!RiscVOpcode::Add.is_branch());
    }

    #[test]
    fn is_load_method() {
        assert!(RiscVOpcode::Lb.is_load());
        assert!(RiscVOpcode::Ld.is_load());
        assert!(RiscVOpcode::Lwu.is_load());
        assert!(RiscVOpcode::Fld.is_load());
        assert!(!RiscVOpcode::Sd.is_load());
    }

    #[test]
    fn is_store_method() {
        assert!(RiscVOpcode::Sb.is_store());
        assert!(RiscVOpcode::Sd.is_store());
        assert!(RiscVOpcode::Fsd.is_store());
        assert!(!RiscVOpcode::Ld.is_store());
    }

    #[test]
    fn is_word_op_method() {
        assert!(RiscVOpcode::Addw.is_word_op());
        assert!(RiscVOpcode::Subw.is_word_op());
        assert!(RiscVOpcode::Addiw.is_word_op());
        assert!(RiscVOpcode::Mulw.is_word_op());
        assert!(RiscVOpcode::Divw.is_word_op());
        assert!(!RiscVOpcode::Add.is_word_op());
        assert!(!RiscVOpcode::Mul.is_word_op());
    }

    #[test]
    fn is_fp_method() {
        assert!(RiscVOpcode::FaddD.is_fp());
        assert!(RiscVOpcode::Fld.is_fp());
        assert!(RiscVOpcode::Fsd.is_fp());
        assert!(RiscVOpcode::FeqD.is_fp());
        assert!(RiscVOpcode::FcvtDW.is_fp());
        assert!(RiscVOpcode::FmvXD.is_fp());
        assert!(!RiscVOpcode::Add.is_fp());
        assert!(!RiscVOpcode::Mul.is_fp());
    }

    #[test]
    fn is_m_extension_method() {
        assert!(RiscVOpcode::Mul.is_m_extension());
        assert!(RiscVOpcode::Mulh.is_m_extension());
        assert!(RiscVOpcode::Div.is_m_extension());
        assert!(RiscVOpcode::Remu.is_m_extension());
        assert!(RiscVOpcode::Mulw.is_m_extension());
        assert!(RiscVOpcode::Remuw.is_m_extension());
        assert!(!RiscVOpcode::Add.is_m_extension());
        assert!(!RiscVOpcode::FaddD.is_m_extension());
    }

    #[test]
    fn opcode_count() {
        // Verify we have the expected number of opcodes by counting variants.
        // RV64I R-type: 10, I-type: 9, U-type: 2, W-type: 9 = 30
        // Load: 7, Store: 4 = 11
        // Branch: 6, Jump: 2 = 8
        // M ext: 8 + 5 word = 13
        // D ext: 5 arith + 2 load/store + 3 compare + 4 convert + 2 move = 16
        // Pseudo: 3
        // Total: 30 + 11 + 8 + 13 + 16 + 3 = 81
        let all_opcodes: Vec<RiscVOpcode> = vec![
            RiscVOpcode::Add, RiscVOpcode::Sub, RiscVOpcode::And,
            RiscVOpcode::Or, RiscVOpcode::Xor, RiscVOpcode::Sll,
            RiscVOpcode::Srl, RiscVOpcode::Sra, RiscVOpcode::Slt,
            RiscVOpcode::Sltu,
            RiscVOpcode::Addi, RiscVOpcode::Andi, RiscVOpcode::Ori,
            RiscVOpcode::Xori, RiscVOpcode::Slti, RiscVOpcode::Sltiu,
            RiscVOpcode::Slli, RiscVOpcode::Srli, RiscVOpcode::Srai,
            RiscVOpcode::Lui, RiscVOpcode::Auipc,
            RiscVOpcode::Addw, RiscVOpcode::Subw, RiscVOpcode::Sllw,
            RiscVOpcode::Srlw, RiscVOpcode::Sraw, RiscVOpcode::Addiw,
            RiscVOpcode::Slliw, RiscVOpcode::Srliw, RiscVOpcode::Sraiw,
            RiscVOpcode::Lb, RiscVOpcode::Lh, RiscVOpcode::Lw,
            RiscVOpcode::Ld, RiscVOpcode::Lbu, RiscVOpcode::Lhu,
            RiscVOpcode::Lwu,
            RiscVOpcode::Sb, RiscVOpcode::Sh, RiscVOpcode::Sw,
            RiscVOpcode::Sd,
            RiscVOpcode::Beq, RiscVOpcode::Bne, RiscVOpcode::Blt,
            RiscVOpcode::Bge, RiscVOpcode::Bltu, RiscVOpcode::Bgeu,
            RiscVOpcode::Jal, RiscVOpcode::Jalr,
            RiscVOpcode::Mul, RiscVOpcode::Mulh, RiscVOpcode::Mulhsu,
            RiscVOpcode::Mulhu, RiscVOpcode::Div, RiscVOpcode::Divu,
            RiscVOpcode::Rem, RiscVOpcode::Remu,
            RiscVOpcode::Mulw, RiscVOpcode::Divw, RiscVOpcode::Divuw,
            RiscVOpcode::Remw, RiscVOpcode::Remuw,
            RiscVOpcode::FaddD, RiscVOpcode::FsubD, RiscVOpcode::FmulD,
            RiscVOpcode::FdivD, RiscVOpcode::FsqrtD,
            RiscVOpcode::Fld, RiscVOpcode::Fsd,
            RiscVOpcode::FeqD, RiscVOpcode::FltD, RiscVOpcode::FleD,
            RiscVOpcode::FcvtDW, RiscVOpcode::FcvtWD, RiscVOpcode::FcvtDL,
            RiscVOpcode::FcvtLD,
            RiscVOpcode::FmvXD, RiscVOpcode::FmvDX,
            RiscVOpcode::Phi, RiscVOpcode::StackAlloc, RiscVOpcode::Nop,
        ];
        assert_eq!(all_opcodes.len(), 81);
    }
}
