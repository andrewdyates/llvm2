// llvm2-ir - Shared machine IR model
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

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
    /// ADD (immediate, shift by 12) — `Xd = Xn + (imm12 << 12)`.
    /// Operands: `[PReg(Rd), PReg(Rn), Imm(imm12)]`.
    /// Used by the TLS local-exec sequence for the hi12 offset.
    AddRIShift12,
    SubRR,
    SubRI,
    MulRR,
    /// MSUB Rd, Rn, Rm, Ra — multiply-subtract: Rd = Ra - Rn * Rm.
    /// When Ra=XZR, this is MNEG Rd, Rn, Rm.
    Msub,
    /// SMULL Xd, Wn, Wm — signed multiply long: Xd = sext(Wn) * sext(Wm).
    Smull,
    /// UMULL Xd, Wn, Wm — unsigned multiply long: Xd = zext(Wn) * zext(Wm).
    Umull,
    SDiv,
    UDiv,
    Neg,

    // -- Logical --
    AndRR,
    AndRI,
    OrrRR,
    OrrRI,
    EorRR,
    EorRI,
    /// ORN Rd, Rn, Rm — bitwise OR-NOT.
    /// When Rn=XZR, this is MVN (bitwise NOT).
    OrnRR,
    /// BIC Rd, Rn, Rm — bitwise AND-NOT (bit clear).
    BicRR,

    // -- Shifts --
    LslRR,
    LsrRR,
    AsrRR,
    LslRI,
    LsrRI,
    AsrRI,

    // -- Compare / conditional select --
    CmpRR,
    CmpRI,
    Tst,
    /// CSEL Xd, Xn, Xm, cond — conditional select.
    /// Operands: [dst, true_src, false_src, Imm(cond_code_encoding)].
    Csel,
    /// CSINC Xd, Xn, Xm, cond — conditional select increment.
    /// Semantically: Xd = cond ? Xn : (Xm + 1).
    Csinc,
    /// CSINV Xd, Xn, Xm, cond — conditional select invert.
    /// Semantically: Xd = cond ? Xn : ~Xm.
    Csinv,
    /// CSNEG Xd, Xn, Xm, cond — conditional select negate.
    /// Semantically: Xd = cond ? Xn : -Xm.
    Csneg,

    // -- Move --
    MovR,
    MovI,
    Movz,
    /// MOVN: move wide with NOT (for small negative constants).
    Movn,
    Movk,
    /// FMOV immediate to FPR (e.g., FMOV Sd, #imm8 or FMOV Dd, #imm8).
    FmovImm,

    // -- Memory (immediate offset) --
    LdrRI,
    StrRI,
    /// LDRB (unsigned offset): load byte, zero-extend to 32-bit.
    /// Operands: [PReg(Rt), PReg(Rn)|Special(SP), Imm(offset)]
    LdrbRI,
    /// LDRH (unsigned offset): load halfword, zero-extend to 32-bit.
    /// Operands: [PReg(Rt), PReg(Rn)|Special(SP), Imm(offset)]
    LdrhRI,
    /// LDRSB (unsigned offset): load byte, sign-extend to 32-bit.
    /// Operands: [PReg(Rt), PReg(Rn)|Special(SP), Imm(offset)]
    LdrsbRI,
    /// LDRSH (unsigned offset): load halfword, sign-extend to 32-bit.
    /// Operands: [PReg(Rt), PReg(Rn)|Special(SP), Imm(offset)]
    LdrshRI,
    /// STRB (unsigned offset): store byte (truncating).
    /// Operands: [PReg(Rt), PReg(Rn)|Special(SP), Imm(offset)]
    StrbRI,
    /// STRH (unsigned offset): store halfword (truncating).
    /// Operands: [PReg(Rt), PReg(Rn)|Special(SP), Imm(offset)]
    StrhRI,
    LdrLiteral,
    LdpRI,
    StpRI,
    /// STP with pre-index writeback: STP Rt, Rt2, [Rn, #imm]!
    /// The base register is updated before the store.
    /// Operands: [PReg(Rt), PReg(Rt2), Special(SP)|PReg(Rn), Imm(offset)]
    StpPreIndex,
    /// LDP with post-index writeback: LDP Rt, Rt2, [Rn], #imm
    /// The base register is updated after the load.
    /// Operands: [PReg(Rt), PReg(Rt2), Special(SP)|PReg(Rn), Imm(offset)]
    LdpPostIndex,

    // -- Memory (register offset) --
    /// LDR Wt, [Xn, Xm] — load 32-bit, base + register offset.
    LdrRO,
    /// STR Wt, [Xn, Xm] — store 32-bit/64-bit, base + register offset.
    StrRO,

    // -- Memory (GOT / TLV) --
    /// LDR Xd, [Xn, #got_pageoff] — load from GOT slot.
    LdrGot,
    /// LDR Xd, [Xn, #tlvp_pageoff] — load from TLV descriptor.
    LdrTlvp,

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

    // -- Conditional --
    /// CSET Xd/Wd, cond — conditional set (materialize condition as 0/1).
    /// Encoded as CSINC Xd, XZR, XZR, invert(cond) per ARM ARM C6.2.70.
    /// Operands: [dst, Imm(cond_code_encoding)].
    /// Semantically: Xd = (cond) ? 1 : 0.
    CSet,

    // -- Extension --
    Sxtw,
    Uxtw,
    Sxtb,
    Sxth,
    /// UXTB Wd, Wn — zero-extend byte to 32-bit (alias: AND Wd, Wn, #0xFF).
    /// Encoded as UBFM Wd, Wn, #0, #7.
    Uxtb,
    /// UXTH Wd, Wn — zero-extend halfword to 32-bit (alias: AND Wd, Wn, #0xFFFF).
    /// Encoded as UBFM Wd, Wn, #0, #15.
    Uxth,

    // -- Bitfield operations --
    /// UBFM Rd, Rn, #immr, #imms — unsigned bitfield move.
    /// Aliases: LSL/LSR (imm), UBFX, UXTB, UXTH.
    Ubfm,
    /// SBFM Rd, Rn, #immr, #imms — signed bitfield move.
    /// Aliases: ASR (imm), SBFX, SXTB, SXTH, SXTW.
    Sbfm,
    /// BFM Rd, Rn, #immr, #imms — bitfield move (insert).
    /// Aliases: BFI, BFXIL.
    Bfm,

    // -- Floating-point --
    FaddRR,
    FsubRR,
    FmulRR,
    FdivRR,
    /// FNEG Dd, Dn — floating-point negate.
    FnegRR,
    /// FABS Dd, Dn — floating-point absolute value.
    FabsRR,
    /// FSQRT Dd, Dn — floating-point square root.
    FsqrtRR,
    Fcmp,
    FcvtzsRR,
    /// FCVTZU: float-to-unsigned-integer conversion (round toward zero).
    FcvtzuRR,
    ScvtfRR,
    /// UCVTF: unsigned-integer-to-float conversion.
    UcvtfRR,
    /// FCVT Dd, Sn: float precision widen (f32 -> f64).
    FcvtSD,
    /// FCVT Ss, Dn: float precision narrow (f64 -> f32).
    FcvtDS,
    /// FMOV between GPR and FPR (e.g., FMOV Sd, Wn or FMOV Dd, Xn).
    FmovGprFpr,
    /// FMOV between FPR and GPR (e.g., FMOV Wn, Sd or FMOV Xn, Dd).
    FmovFprGpr,
    /// FMOV between FPR registers (e.g., FMOV Dd, Dn or FMOV Ss, Sn).
    /// Encoded as FP data-processing 1-source with opcode=00 (FmovReg).
    /// Operands: [Rd, Rn] where both are FPR.
    FmovFprFpr,

    // -- NEON SIMD (vector) --
    /// ADD Vd.T, Vn.T, Vm.T — integer vector add.
    /// Operands: [Vd, Vn, Vm, Imm(arrangement)]
    NeonAddV,
    /// SUB Vd.T, Vn.T, Vm.T — integer vector subtract.
    NeonSubV,
    /// MUL Vd.T, Vn.T, Vm.T — integer vector multiply.
    NeonMulV,
    /// FADD Vd.T, Vn.T, Vm.T — FP vector add.
    NeonFaddV,
    /// FSUB Vd.T, Vn.T, Vm.T — FP vector subtract.
    NeonFsubV,
    /// FMUL Vd.T, Vn.T, Vm.T — FP vector multiply.
    NeonFmulV,
    /// FDIV Vd.T, Vn.T, Vm.T — FP vector divide.
    NeonFdivV,
    /// AND Vd.T, Vn.T, Vm.T — vector bitwise AND.
    NeonAndV,
    /// ORR Vd.T, Vn.T, Vm.T — vector bitwise OR.
    NeonOrrV,
    /// EOR Vd.T, Vn.T, Vm.T — vector bitwise XOR.
    NeonEorV,
    /// BIC Vd.T, Vn.T, Vm.T — vector bitwise AND-NOT.
    NeonBicV,
    /// NOT Vd.T, Vn.T — vector bitwise NOT.
    NeonNotV,
    /// CMEQ Vd.T, Vn.T, Vm.T — vector compare equal.
    NeonCmeqV,
    /// CMGT Vd.T, Vn.T, Vm.T — vector compare greater than (signed).
    NeonCmgtV,
    /// CMGE Vd.T, Vn.T, Vm.T — vector compare greater or equal (signed).
    NeonCmgeV,
    /// DUP Vd.T, Vn.Ts[lane] — duplicate element to all lanes.
    /// Operands: [Vd, Vn, Imm(lane), Imm(element_size)]
    NeonDupElem,
    /// DUP Vd.T, Xn/Wn — duplicate GPR to all vector lanes.
    /// Operands: [Vd, Rn, Imm(element_size)]
    NeonDupGen,
    /// INS Vd.Ts[lane], Xn/Wn — insert GPR into vector lane.
    /// Operands: [Vd, Rn, Imm(lane), Imm(element_size)]
    NeonInsGen,
    /// MOVI Vd.T, #imm8 — move immediate to vector (byte form).
    /// Operands: [Vd, Imm(imm8)]
    NeonMovi,
    /// LD1 {Vt.T}, [Xn], #imm — SIMD load 1 register, post-index.
    /// Operands: [Vt, Xn, Imm(arrangement)]
    NeonLd1Post,
    /// ST1 {Vt.T}, [Xn], #imm — SIMD store 1 register, post-index.
    /// Operands: [Vt, Xn, Imm(arrangement)]
    NeonSt1Post,

    // -- Atomic memory operations (ARMv8.1-a LSE + legacy LL/SC) --
    /// LDAR Xt, [Xn] — load-acquire (sequential consistency load).
    /// size: 32-bit (Wt) or 64-bit (Xt) from register class.
    /// Operands: [Rt, Rn]
    Ldar,
    /// LDARB Wt, [Xn] — load-acquire byte.
    /// Operands: [Rt, Rn]
    Ldarb,
    /// LDARH Wt, [Xn] — load-acquire halfword.
    /// Operands: [Rt, Rn]
    Ldarh,
    /// STLR Xt, [Xn] — store-release (sequential consistency store).
    /// size: 32-bit (Wt) or 64-bit (Xt) from register class.
    /// Operands: [Rt, Rn]
    Stlr,
    /// STLRB Wt, [Xn] — store-release byte.
    /// Operands: [Rt, Rn]
    Stlrb,
    /// STLRH Wt, [Xn] — store-release halfword.
    /// Operands: [Rt, Rn]
    Stlrh,

    /// LDADD Xs, Xt, [Xn] — atomic add (ARMv8.1-a LSE).
    /// Atomically: Xt = *Xn; *Xn = Xt + Xs.
    /// Operands: [Rs (addend), Rt (old value dest), Rn (address)]
    Ldadd,
    /// LDADDA — load-acquire variant.
    Ldadda,
    /// LDADDAL — load-acquire + store-release (full barrier).
    Ldaddal,

    /// LDCLR Xs, Xt, [Xn] — atomic bit clear (AND NOT) (ARMv8.1-a LSE).
    /// Atomically: Xt = *Xn; *Xn = Xt AND NOT Xs.
    /// Operands: [Rs, Rt, Rn]
    Ldclr,
    /// LDCLRAL — full barrier variant.
    Ldclral,

    /// LDEOR Xs, Xt, [Xn] — atomic exclusive OR (ARMv8.1-a LSE).
    /// Atomically: Xt = *Xn; *Xn = Xt XOR Xs.
    /// Operands: [Rs, Rt, Rn]
    Ldeor,
    /// LDEORAL — full barrier variant.
    Ldeoral,

    /// LDSET Xs, Xt, [Xn] — atomic bit set (OR) (ARMv8.1-a LSE).
    /// Atomically: Xt = *Xn; *Xn = Xt OR Xs.
    /// Operands: [Rs, Rt, Rn]
    Ldset,
    /// LDSETAL — full barrier variant.
    Ldsetal,

    /// SWP Xs, Xt, [Xn] — atomic swap (ARMv8.1-a LSE).
    /// Atomically: Xt = *Xn; *Xn = Xs.
    /// Operands: [Rs, Rt, Rn]
    Swp,
    /// SWPAL — full barrier variant.
    Swpal,

    /// CAS Xs, Xt, [Xn] — compare and swap (ARMv8.1-a LSE).
    /// Atomically: if *Xn == Xs then *Xn = Xt; Xs = old *Xn.
    /// Operands: [Rs (expected/result), Rt (desired), Rn (address)]
    Cas,
    /// CASA — load-acquire variant.
    Casa,
    /// CASAL — full barrier (acquire + release).
    Casal,

    /// LDAXR Xt, [Xn] — load-acquire exclusive register (LL/SC legacy path).
    /// Operands: [Rt, Rn]
    Ldaxr,
    /// STLXR Ws, Xt, [Xn] — store-release exclusive register (LL/SC legacy path).
    /// Ws receives 0 on success, 1 on failure.
    /// Operands: [Ws (status), Rt (value), Rn (address)]
    Stlxr,

    /// DMB — data memory barrier.
    /// Operands: [Imm(option)] where option is CRm field (e.g., 0xF = SY, 0xB = ISH).
    Dmb,
    /// DSB — data synchronization barrier.
    /// Operands: [Imm(option)]
    Dsb,
    /// ISB — instruction synchronization barrier.
    /// Operands: [Imm(option)] (typically 0xF = SY).
    Isb,

    // -- Address --
    Adrp,
    /// ADR Xd, label — form PC-relative address (used for jump table base).
    /// Operands (at ISel level): [Xd, JumpTable{...}]
    /// Operands (at codegen level): [Xd, Imm(offset)]
    Adr,
    AddPCRel,

    // -- Jump table support --
    /// LDRSW Xt, [Xn, Xm, LSL #2] — load signed word with register offset.
    /// Used to load 32-bit relative offsets from jump tables.
    /// Operands: [Xt (dst), Xn (base), Xm (index)]
    LdrswRO,

    // -- Checked arithmetic (set flags for overflow detection) --
    /// ADDS: add and set flags (used for overflow-checked addition).
    AddsRR,
    /// ADDS immediate: add immediate and set flags.
    AddsRI,
    /// SUBS: subtract and set flags (used for overflow-checked subtraction).
    SubsRR,
    /// SUBS immediate: subtract immediate and set flags.
    SubsRI,

    // -- i128 multi-register arithmetic --
    /// ADC Xd, Xn, Xm — add with carry (for i128 high-half addition).
    /// Reads carry flag from previous ADDS. Always 64-bit.
    Adc,
    /// SBC Xd, Xn, Xm — subtract with carry/borrow (for i128 high-half subtraction).
    /// Reads carry/borrow flag from previous SUBS. Always 64-bit.
    Sbc,
    /// UMULH Xd, Xn, Xm — unsigned multiply high (upper 64 bits of 64x64->128 product).
    /// Always 64-bit (no 32-bit variant).
    Umulh,
    /// SMULH Xd, Xn, Xm — signed multiply high (upper 64 bits of signed 64x64->128 product).
    /// Always 64-bit (no 32-bit variant). Used for the aarch64 overflow-safe
    /// signed-mul idiom: `MUL lo; SMULH hi; CMP hi, lo, ASR #63; B.NE overflow`.
    Smulh,
    /// MADD Xd, Xn, Xm, Xa — multiply-add: Xd = Xa + Xn * Xm.
    /// Used for i128 multiplication middle-term accumulation.
    Madd,

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
    /// Trap on division by zero: branch to trap block if divisor is zero.
    /// Operands: [Block(panic_target)].
    TrapDivZero,
    /// Trap on out-of-range shift amount: branch to trap block if shift >= bitwidth.
    /// Operands: [Block(panic_target)].
    TrapShiftRange,

    // -- Reference counting pseudo-instructions --
    /// Retain (increment reference count). Operands: [ptr].
    Retain,
    /// Release (decrement reference count). Operands: [ptr].
    Release,

    // -- LLVM-style typed aliases (used by llvm2-lower isel) --
    /// MOV Wd, Wn — 32-bit register move.
    MOVWrr,
    /// MOV Xd, Xn — 64-bit register move.
    MOVXrr,
    /// STR Wt, [Xn, #imm] — store 32-bit integer, unsigned immediate offset.
    STRWui,
    /// STR Xt, [Xn, #imm] — store 64-bit integer, unsigned immediate offset.
    STRXui,
    /// STR St, [Xn, #imm] — store 32-bit FP, unsigned immediate offset.
    STRSui,
    /// STR Dt, [Xn, #imm] — store 64-bit FP, unsigned immediate offset.
    STRDui,
    /// BL label — branch with link (LLVM-style alias for Bl).
    BL,
    /// BLR Xn — branch with link to register (LLVM-style alias for Blr).
    BLR,
    /// CMP Wn, Wm — 32-bit compare register.
    CMPWrr,
    /// CMP Xn, Xm — 64-bit compare register.
    CMPXrr,
    /// CMP Wn, #imm — 32-bit compare immediate.
    CMPWri,
    /// CMP Xn, #imm — 64-bit compare immediate.
    CMPXri,
    /// MOVZ Wd, #imm — 32-bit move zero immediate.
    MOVZWi,
    /// MOVZ Xd, #imm — 64-bit move zero immediate.
    MOVZXi,
    /// B.cond label — conditional branch (LLVM-style alias for BCond).
    Bcc,

    // -- System register access --
    /// MRS Xd, (sysreg) — move from system register.
    ///
    /// Reads an AArch64 system register into a GPR. Used by the thread-local
    /// storage local-exec sequence (MRS Xd, TPIDR_EL0) and other system-level
    /// accesses. The destination is always 64-bit.
    ///
    /// Operands: `[PReg(Xd), Imm(sysreg_encoding)]`
    ///
    /// `sysreg_encoding` packs op0/op1/CRn/CRm/op2 into the 16-bit
    /// "systemreg" field used by the A64 instruction encoding (bits[20:5]):
    ///   bits [15:14] = op0  (always 0b11 for EL0/EL1 sysregs MRS/MSR can access)
    ///   bits [13:11] = op1
    ///   bits [10:7]  = CRn
    ///   bits [6:3]   = CRm
    ///   bits [2:0]   = op2
    /// For TPIDR_EL0 (op0=11, op1=011, CRn=1101, CRm=0000, op2=010) the
    /// packed value is `0xDE82`, and the full instruction word is
    /// `0xD53BD040 | Rd`.
    ///
    /// See the MRS encoder in `aarch64/encode.rs`, ARM ARM C6.2.169, and
    /// LLVM `AArch64SystemOperands.td` `class SysReg`.
    Mrs,

    // -- Pseudo-instructions (no hardware encoding) --
    Phi,
    StackAlloc,
    /// COPY: register-to-register copy pseudo (resolved by regalloc).
    Copy,
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

            // Conditional branch aliases
            Bcc => InstFlags::IS_BRANCH.union(InstFlags::IS_TERMINATOR),

            // Calls
            Bl | BL => InstFlags::IS_CALL.union(InstFlags::HAS_SIDE_EFFECTS),
            Blr | BLR => InstFlags::IS_CALL.union(InstFlags::HAS_SIDE_EFFECTS),

            // Return
            Ret => InstFlags::IS_RETURN.union(InstFlags::IS_TERMINATOR),

            // Memory loads
            LdrRI | LdrbRI | LdrhRI | LdrsbRI | LdrshRI | LdrRO | LdrswRO => {
                InstFlags::READS_MEMORY
            }
            LdrLiteral | LdrGot | LdrTlvp => InstFlags::READS_MEMORY,
            LdpRI | LdpPostIndex => InstFlags::READS_MEMORY,
            NeonLd1Post => InstFlags::READS_MEMORY,

            // Memory stores
            StrRI | StrbRI | StrhRI | StrRO => {
                InstFlags::WRITES_MEMORY.union(InstFlags::HAS_SIDE_EFFECTS)
            }
            STRWui | STRXui | STRSui | STRDui => {
                InstFlags::WRITES_MEMORY.union(InstFlags::HAS_SIDE_EFFECTS)
            }
            StpRI | StpPreIndex => InstFlags::WRITES_MEMORY.union(InstFlags::HAS_SIDE_EFFECTS),
            NeonSt1Post => InstFlags::WRITES_MEMORY.union(InstFlags::HAS_SIDE_EFFECTS),

            // Atomic loads (load-acquire): read memory with ordering side effect
            Ldar | Ldarb | Ldarh => InstFlags::READS_MEMORY.union(InstFlags::HAS_SIDE_EFFECTS),

            // Atomic stores (store-release): write memory with ordering side effect
            Stlr | Stlrb | Stlrh => InstFlags::WRITES_MEMORY.union(InstFlags::HAS_SIDE_EFFECTS),

            // Atomic read-modify-write (LSE): read AND write memory, always side-effecting
            Ldadd | Ldadda | Ldaddal | Ldclr | Ldclral | Ldeor | Ldeoral | Ldset | Ldsetal
            | Swp | Swpal => InstFlags::READS_MEMORY
                .union(InstFlags::WRITES_MEMORY)
                .union(InstFlags::HAS_SIDE_EFFECTS),

            // Compare-and-swap (LSE): read AND write memory, always side-effecting
            Cas | Casa | Casal => InstFlags::READS_MEMORY
                .union(InstFlags::WRITES_MEMORY)
                .union(InstFlags::HAS_SIDE_EFFECTS),

            // Exclusive load/store (LL/SC legacy path)
            Ldaxr => InstFlags::READS_MEMORY.union(InstFlags::HAS_SIDE_EFFECTS),
            Stlxr => InstFlags::WRITES_MEMORY.union(InstFlags::HAS_SIDE_EFFECTS),

            // Memory barriers: pure side effects (enforce ordering)
            Dmb | Dsb | Isb => InstFlags::HAS_SIDE_EFFECTS,

            // System register read: treat as side-effecting so optimization
            // passes never reorder it across memory ops or speculate it. The
            // opcode covers all sysregs; only some (e.g. TPIDR_EL0) are
            // thread-stable, and the optimizer has no way to know which.
            Mrs => InstFlags::HAS_SIDE_EFFECTS,

            // Shifted immediate arithmetic: same default semantics as AddRI.
            AddRIShift12 => InstFlags::EMPTY,

            // Pseudo-instructions
            Phi => InstFlags::IS_PSEUDO,
            StackAlloc => InstFlags::IS_PSEUDO.union(InstFlags::HAS_SIDE_EFFECTS),
            Copy => InstFlags::IS_PSEUDO,
            Nop => InstFlags::IS_PSEUDO,

            // Compare/test (set condition flags = side effect)
            CmpRR | CmpRI | CMPWrr | CMPXrr | CMPWri | CMPXri | Tst | Fcmp => {
                InstFlags::HAS_SIDE_EFFECTS
            }

            // Checked arithmetic: produce a result AND set flags (side effect)
            AddsRR | AddsRI | SubsRR | SubsRI => InstFlags::HAS_SIDE_EFFECTS,

            // i128 multi-register: ADC/SBC read NZCV flags from preceding ADDS/SUBS
            Adc | Sbc => InstFlags::HAS_SIDE_EFFECTS,

            // Trap pseudo-instructions: conditional branches to panic blocks
            TrapOverflow => InstFlags::IS_BRANCH
                .union(InstFlags::IS_TERMINATOR)
                .union(InstFlags::HAS_SIDE_EFFECTS),
            TrapBoundsCheck => InstFlags::IS_BRANCH
                .union(InstFlags::IS_TERMINATOR)
                .union(InstFlags::HAS_SIDE_EFFECTS),
            TrapNull => InstFlags::IS_BRANCH
                .union(InstFlags::IS_TERMINATOR)
                .union(InstFlags::HAS_SIDE_EFFECTS),
            TrapDivZero => InstFlags::IS_BRANCH
                .union(InstFlags::IS_TERMINATOR)
                .union(InstFlags::HAS_SIDE_EFFECTS),
            TrapShiftRange => InstFlags::IS_BRANCH
                .union(InstFlags::IS_TERMINATOR)
                .union(InstFlags::HAS_SIDE_EFFECTS),

            // Reference counting: side effects (modify refcount in memory)
            Retain => InstFlags::HAS_SIDE_EFFECTS
                .union(InstFlags::READS_MEMORY)
                .union(InstFlags::WRITES_MEMORY),
            Release => InstFlags::HAS_SIDE_EFFECTS
                .union(InstFlags::READS_MEMORY)
                .union(InstFlags::WRITES_MEMORY),

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
                | Self::Copy
                | Self::Nop
                | Self::TrapOverflow
                | Self::TrapBoundsCheck
                | Self::TrapNull
                | Self::TrapDivZero
                | Self::TrapShiftRange
                | Self::Retain
                | Self::Release
        )
    }

    /// Returns true if this is a phi instruction.
    pub fn is_phi(self) -> bool {
        matches!(self, Self::Phi)
    }

    // -- Generic instruction property queries --
    //
    // These enable optimization passes to operate on generic instruction
    // properties rather than matching target-specific opcode variants.
    // This is the foundation for multi-target optimization support.

    /// Returns true if this is a no-op instruction (can be deleted without
    /// affecting program semantics).
    pub fn is_nop(self) -> bool {
        matches!(self, Self::Nop)
    }

    /// Returns true if this is a register-to-register move (copy).
    ///
    /// Move instructions transfer a value from one register to another
    /// without modifying it. Includes both generic pseudo-moves and
    /// target-specific move variants.
    pub fn is_move(self) -> bool {
        matches!(
            self,
            Self::MovR | Self::Copy | Self::MOVWrr | Self::MOVXrr | Self::FmovFprFpr
        )
    }

    /// Returns true if this is a move-immediate instruction (loads a
    /// constant value into a register).
    pub fn is_move_imm(self) -> bool {
        matches!(
            self,
            Self::MovI | Self::Movz | Self::Movn | Self::MOVZWi | Self::MOVZXi
        )
    }

    /// Returns true if this is an unconditional branch (always transfers
    /// control, no fallthrough).
    pub fn is_unconditional_branch(self) -> bool {
        matches!(self, Self::B | Self::Br)
    }

    /// Returns true if this is a conditional branch (may or may not transfer
    /// control; has a fallthrough path).
    pub fn is_conditional_branch(self) -> bool {
        matches!(
            self,
            Self::BCond | Self::Bcc | Self::Cbz | Self::Cbnz | Self::Tbz | Self::Tbnz
        )
    }

    /// Returns true if this is a compare-and-branch-if-zero instruction.
    pub fn is_cbz(self) -> bool {
        matches!(self, Self::Cbz)
    }

    /// Returns true if this is a compare-and-branch-if-not-zero instruction.
    pub fn is_cbnz(self) -> bool {
        matches!(self, Self::Cbnz)
    }

    /// Returns true if this is a commutative operation (operand order does
    /// not affect the result).
    pub fn is_commutative(self) -> bool {
        matches!(
            self,
            Self::AddRR
                | Self::MulRR
                | Self::AndRR
                | Self::OrrRR
                | Self::EorRR
                | Self::FaddRR
                | Self::FmulRR
                | Self::NeonAddV
                | Self::NeonMulV
                | Self::NeonFaddV
                | Self::NeonFmulV
                | Self::NeonAndV
                | Self::NeonOrrV
                | Self::NeonEorV
        )
    }

    /// Returns true if this opcode produces a value (operand[0] is a def).
    ///
    /// Instructions that don't produce values: CMP, TST, STR, STP, branches,
    /// returns, NOP, calls, traps, and reference counting ops.
    pub fn produces_value(self) -> bool {
        use AArch64Opcode::*;
        match self {
            // Compare/test: set flags, no register def
            CmpRR | CmpRI | Tst | Fcmp | CMPWrr | CMPXrr | CMPWri | CMPXri => false,
            // Stores: write to memory, no register def
            StrRI | StrbRI | StrhRI | StpRI | StpPreIndex | StrRO | STRWui | STRXui | STRSui
            | STRDui | NeonSt1Post => false,
            // Branches and returns: control flow, no register def
            B | BCond | Bcc | Cbz | Cbnz | Tbz | Tbnz | Br | Ret => false,
            // Trap pseudo-instructions: control flow, no register def
            TrapOverflow | TrapBoundsCheck | TrapNull | TrapDivZero | TrapShiftRange => false,
            // Reference counting: side effects, no register def
            Retain | Release => false,
            // Nop: no def
            Nop => false,
            // Calls: produce result via implicit defs; for simple model, not a value producer
            Bl | Blr | BL | BLR => false,
            // Memory barriers: no register def
            Dmb | Dsb | Isb => false,
            // Atomic stores: no register def for the store itself
            Stlr | Stlrb | Stlrh | Stlxr => false,
            // Everything else produces a value in operand[0]
            _ => true,
        }
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
    /// Proof-guided: this memory instruction has been proven safe to reorder
    /// past other memory operations. Set by the ValidBorrow proof optimization.
    pub const PROOF_REORDERABLE: Self = Self(0x200);

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

    /// Create from raw bits (e.g., `InstFlags::from_bits(IS_CALL | IS_BRANCH)`).
    ///
    /// Used by crates that construct InstFlags from u16 constants
    /// (e.g., regalloc test helpers, pipeline adapters).
    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    // -- Convenience query methods --
    //
    // These duplicate the methods on MachInst but operate on InstFlags directly.
    // Used by the register allocator which stores InstFlags separately from
    // opcode/operands and needs to query flags without a MachInst wrapper.

    #[inline]
    pub const fn is_call(self) -> bool {
        self.contains(Self::IS_CALL)
    }

    #[inline]
    pub const fn is_branch(self) -> bool {
        self.contains(Self::IS_BRANCH)
    }

    #[inline]
    pub const fn is_return(self) -> bool {
        self.contains(Self::IS_RETURN)
    }

    #[inline]
    pub const fn is_terminator(self) -> bool {
        self.contains(Self::IS_TERMINATOR)
    }

    #[inline]
    pub const fn has_side_effects(self) -> bool {
        self.contains(Self::HAS_SIDE_EFFECTS)
    }

    #[inline]
    pub const fn is_pseudo(self) -> bool {
        self.contains(Self::IS_PSEUDO)
    }

    #[inline]
    pub const fn reads_memory(self) -> bool {
        self.contains(Self::READS_MEMORY)
    }

    #[inline]
    pub const fn writes_memory(self) -> bool {
        self.contains(Self::WRITES_MEMORY)
    }

    #[inline]
    pub const fn is_phi(self) -> bool {
        self.contains(Self::IS_PHI)
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
            (Self::PROOF_REORDERABLE, "PROOF_REORDERABLE"),
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
/// - `NonZeroDivisor` → eliminate division-by-zero checks
/// - `ValidShift` → eliminate shift-amount range checks
/// - `Pure` → aggressive CSE/LICM of proven-pure memory operations
/// - `Associative` → parallel reduction trees, operation reordering
/// - `Commutative` → operand canonicalization, parallel reduction
/// - `Idempotent` → redundant application elimination
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

    /// tMIR has proven the divisor is non-zero.
    /// Enables: remove CBZ divisor / TrapDivZero guard before UDIV/SDIV.
    NonZeroDivisor,

    /// tMIR has proven the shift amount is in [0, bitwidth).
    /// Enables: remove CMP+B.GE range check before LSL/LSR/ASR.
    ValidShift,

    /// tMIR has proven this operation is pure (no observable side effects).
    /// Enables: aggressive CSE of loads, LICM of memory operations.
    /// A load with Pure proof can be treated as a pure computation for
    /// CSE purposes: if two loads from the same address exist and the
    /// address is proven pure (immutable), the second load is redundant.
    Pure,

    /// tMIR has proven this operation is associative: (a op b) op c = a op (b op c).
    /// Enables: parallel reduction trees, operation reordering for vectorization.
    Associative,

    /// tMIR has proven this operation is commutative: a op b = b op a.
    /// Enables: operand canonicalization, parallel reduction, vectorization.
    Commutative,

    /// tMIR has proven this operation is idempotent: f(f(x)) = f(x).
    /// Enables: redundant application elimination.
    Idempotent,
}

impl ProofAnnotation {
    /// Conservatively merge two optional proof annotations.
    ///
    /// Used by optimization passes to combine proof annotations when
    /// instructions are replaced or eliminated:
    /// - If both are `None`, returns `None`.
    /// - If one is `Some` and the other is `None`, returns the `Some`.
    /// - If both are `Some` and equal, returns that annotation.
    /// - If both are `Some` but different, returns `None` (conservative:
    ///   we cannot combine proofs of different properties).
    pub fn merge(
        a: Option<ProofAnnotation>,
        b: Option<ProofAnnotation>,
    ) -> Option<ProofAnnotation> {
        match (a, b) {
            (None, None) => None,
            (Some(proof), None) | (None, Some(proof)) => Some(proof),
            (Some(x), Some(y)) if x == y => Some(x),
            (Some(_), Some(_)) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// SourceLoc — source location for debug info
// ---------------------------------------------------------------------------

/// Source location for DWARF debug info.
///
/// Tracks the original source file, line, and column for a machine instruction.
/// Populated from tMIR `SourceSpan` during instruction selection and preserved
/// through optimization/regalloc for DWARF line number program emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SourceLoc {
    /// Source file index (0-based, matches tMIR SourceSpan.file).
    pub file: u32,
    /// Source line number (1-based).
    pub line: u32,
    /// Source column number (0 = unknown).
    pub col: u32,
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
    /// Optional source location from tMIR for DWARF debug info.
    /// Populated from tMIR `SourceSpan` during ISel, preserved through
    /// optimization and register allocation for line number program emission.
    pub source_loc: Option<SourceLoc>,
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
            source_loc: None,
        }
    }

    /// Create a new instruction with explicit flags.
    pub fn with_flags(opcode: AArch64Opcode, operands: Vec<MachOperand>, flags: InstFlags) -> Self {
        Self {
            opcode,
            operands,
            implicit_defs: &[],
            implicit_uses: &[],
            flags,
            proof: None,
            source_loc: None,
        }
    }

    /// Attach a proof annotation to this instruction.
    pub fn with_proof(mut self, proof: ProofAnnotation) -> Self {
        self.proof = Some(proof);
        self
    }

    /// Attach a source location for DWARF debug info.
    pub fn with_source_loc(mut self, loc: SourceLoc) -> Self {
        self.source_loc = Some(loc);
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

    // -- Generic instruction property queries (delegates to opcode) --

    /// Returns true if this is a no-op instruction.
    #[inline]
    pub fn is_nop(&self) -> bool {
        self.opcode.is_nop()
    }

    /// Returns true if this is a register-to-register move/copy.
    #[inline]
    pub fn is_move(&self) -> bool {
        self.opcode.is_move()
    }

    /// Returns true if this is a move-immediate instruction.
    #[inline]
    pub fn is_move_imm(&self) -> bool {
        self.opcode.is_move_imm()
    }

    /// Returns true if this is an unconditional branch.
    #[inline]
    pub fn is_unconditional_branch(&self) -> bool {
        self.opcode.is_unconditional_branch()
    }

    /// Returns true if this is a conditional branch.
    #[inline]
    pub fn is_conditional_branch(&self) -> bool {
        self.opcode.is_conditional_branch()
    }

    /// Returns true if this is a commutative operation.
    #[inline]
    pub fn is_commutative(&self) -> bool {
        self.opcode.is_commutative()
    }

    /// Returns true if this instruction produces a value (operand[0] is a def).
    #[inline]
    pub fn produces_value(&self) -> bool {
        self.opcode.produces_value()
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
                "{:?} should have IS_BRANCH",
                op
            );
            assert!(
                flags.contains(InstFlags::IS_TERMINATOR),
                "{:?} should have IS_TERMINATOR",
                op
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
                "{:?} should have IS_CALL",
                op
            );
            assert!(
                flags.contains(InstFlags::HAS_SIDE_EFFECTS),
                "{:?} should have HAS_SIDE_EFFECTS",
                op
            );
            assert!(
                !flags.contains(InstFlags::IS_BRANCH),
                "{:?} should NOT have IS_BRANCH",
                op
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
            AArch64Opcode::LdpPostIndex,
        ];
        for op in &load_ops {
            let flags = op.default_flags();
            assert!(
                flags.contains(InstFlags::READS_MEMORY),
                "{:?} should have READS_MEMORY",
                op
            );
            assert!(
                !flags.contains(InstFlags::WRITES_MEMORY),
                "{:?} should NOT have WRITES_MEMORY",
                op
            );
        }
    }

    #[test]
    fn store_opcodes_have_writes_memory_and_side_effects() {
        let store_ops = [
            AArch64Opcode::StrRI,
            AArch64Opcode::StpRI,
            AArch64Opcode::StpPreIndex,
        ];
        for op in &store_ops {
            let flags = op.default_flags();
            assert!(
                flags.contains(InstFlags::WRITES_MEMORY),
                "{:?} should have WRITES_MEMORY",
                op
            );
            assert!(
                flags.contains(InstFlags::HAS_SIDE_EFFECTS),
                "{:?} should have HAS_SIDE_EFFECTS",
                op
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
                "{:?} should have IS_PSEUDO",
                op
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
            AArch64Opcode::AddRIShift12,
            AArch64Opcode::SubRR,
            AArch64Opcode::SubRI,
            AArch64Opcode::MulRR,
            AArch64Opcode::SDiv,
            AArch64Opcode::UDiv,
            AArch64Opcode::Neg,
            AArch64Opcode::AndRR,
            AArch64Opcode::OrrRR,
            AArch64Opcode::EorRR,
            AArch64Opcode::OrnRR,
            AArch64Opcode::FnegRR,
            AArch64Opcode::MovR,
            AArch64Opcode::MovI,
        ];
        for op in &pure_ops {
            let flags = op.default_flags();
            assert!(
                flags.is_empty(),
                "{:?} should have EMPTY flags but has {:?}",
                op,
                flags
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
                "{:?} should have HAS_SIDE_EFFECTS",
                op
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
                    flags[i].bits(),
                    flags[j].bits(),
                    "flags {:?} and {:?} have same bits",
                    flags[i],
                    flags[j]
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
        let inst = MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(BlockId(1))]);
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
        let inst = MachInst::with_flags(AArch64Opcode::AddRR, vec![], InstFlags::HAS_SIDE_EFFECTS);
        assert!(inst.has_side_effects());
        assert!(!inst.is_call());
    }

    #[test]
    fn machinst_with_implicit_defs() {
        static DEFS: &[PReg] = &[X0, X1];
        let inst = MachInst::new(AArch64Opcode::Bl, vec![]).with_implicit_defs(DEFS);
        assert_eq!(inst.implicit_defs, DEFS);
        assert!(inst.implicit_uses.is_empty());
    }

    #[test]
    fn machinst_with_implicit_uses() {
        static USES: &[PReg] = &[X0];
        let inst = MachInst::new(AArch64Opcode::Ret, vec![]).with_implicit_uses(USES);
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
        let inst = MachInst::new(AArch64Opcode::AddRR, vec![MachOperand::Imm(42)]);
        let inst2 = inst.clone();
        assert_eq!(inst2.opcode, inst.opcode);
        assert_eq!(inst2.operands.len(), inst.operands.len());
        assert_eq!(inst2.flags, inst.flags);
    }

    // ---- ProofAnnotation::merge tests ----

    #[test]
    fn proof_merge_none_none() {
        assert_eq!(ProofAnnotation::merge(None, None), None);
    }

    #[test]
    fn proof_merge_some_and_none() {
        assert_eq!(
            ProofAnnotation::merge(Some(ProofAnnotation::NoOverflow), None),
            Some(ProofAnnotation::NoOverflow),
        );
        assert_eq!(
            ProofAnnotation::merge(None, Some(ProofAnnotation::InBounds)),
            Some(ProofAnnotation::InBounds),
        );
    }

    #[test]
    fn proof_merge_equal() {
        assert_eq!(
            ProofAnnotation::merge(
                Some(ProofAnnotation::NotNull),
                Some(ProofAnnotation::NotNull),
            ),
            Some(ProofAnnotation::NotNull),
        );
    }

    #[test]
    fn proof_merge_different_returns_none() {
        assert_eq!(
            ProofAnnotation::merge(
                Some(ProofAnnotation::ValidBorrow),
                Some(ProofAnnotation::PositiveRefCount),
            ),
            None,
        );
    }

    #[test]
    fn proof_merge_all_variants_with_self() {
        let variants = [
            ProofAnnotation::NoOverflow,
            ProofAnnotation::InBounds,
            ProofAnnotation::NotNull,
            ProofAnnotation::ValidBorrow,
            ProofAnnotation::PositiveRefCount,
            ProofAnnotation::NonZeroDivisor,
            ProofAnnotation::ValidShift,
            ProofAnnotation::Pure,
            ProofAnnotation::Associative,
            ProofAnnotation::Commutative,
            ProofAnnotation::Idempotent,
        ];
        for v in &variants {
            assert_eq!(
                ProofAnnotation::merge(Some(*v), Some(*v)),
                Some(*v),
                "{:?} merged with itself should be Some({:?})",
                v,
                v,
            );
        }
    }

    // --- SourceLoc tests ---

    #[test]
    fn test_source_loc_on_mach_inst() {
        let inst = MachInst::new(AArch64Opcode::AddRR, vec![]).with_source_loc(SourceLoc {
            file: 0,
            line: 42,
            col: 5,
        });
        assert!(inst.source_loc.is_some());
        let loc = inst.source_loc.unwrap();
        assert_eq!(loc.file, 0);
        assert_eq!(loc.line, 42);
        assert_eq!(loc.col, 5);
    }

    #[test]
    fn test_source_loc_default_none() {
        let inst = MachInst::new(AArch64Opcode::Ret, vec![]);
        assert!(inst.source_loc.is_none());
    }

    #[test]
    fn test_source_loc_preserved_through_clone() {
        let inst = MachInst::new(AArch64Opcode::SubRR, vec![]).with_source_loc(SourceLoc {
            file: 1,
            line: 100,
            col: 0,
        });
        let cloned = inst.clone();
        assert_eq!(
            cloned.source_loc,
            Some(SourceLoc {
                file: 1,
                line: 100,
                col: 0
            })
        );
    }
}
