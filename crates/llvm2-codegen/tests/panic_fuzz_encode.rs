// llvm2-codegen/tests/panic_fuzz_encode.rs
// Property-based panic-fuzz harness for `encode_instruction` (AArch64).
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Part of #387 (proptest panic-fuzz) / Part of #372 (Crash-free codegen) /
// Part of #450 (widen opcode coverage >= 80%).
//
// Reference: `designs/2026-04-18-crash-free-codegen-plan.md` §5 (proptest
// as primary defense) and §6 (per-crate harness).
//
// Contract under test: for *any* `MachInst` value, `encode_instruction`
// must either return `Ok(u32)` or `Err(EncodeError)` — it must NEVER
// panic, abort, overflow (in debug), or trigger a slice-index panic. This
// is the empirical half of the Phase-1 boundary conversion: Phase 1
// replaced the in-function `unwrap()` / `unreachable!()` sites with
// typed `EncodeError` returns; this harness proves the replacement is
// exhaustive over random-but-valid and random-malformed inputs.
//
// Run:
//   cargo test -p llvm2-codegen --test panic_fuzz_encode
// Increase case count via env:
//   PROPTEST_CASES=100000 cargo test -p llvm2-codegen --test panic_fuzz_encode
//
// ---------------------------------------------------------------------------
// Coverage (#450)
// ---------------------------------------------------------------------------
//
// `opcode_strategy()` below enumerates **every** variant of
// `AArch64Opcode` (178 as of 2026-04-19), giving 100% static opcode
// coverage. This widens the earlier 47% cross-section to the full enum so
// that newly added variants are automatically fuzzed.
//
// `valid_machinst_strategy()` below composes per-category "well-shaped"
// strategies (arithmetic, logical, shift, compare/select, move, memory
// immediate + register-offset + pair, branch, FP scalar + conversion,
// NEON SIMD, atomic LSE, LL/SC, barrier, bitfield, extension, trap /
// RC / sysreg pseudos, LLVM-style typed aliases). These exercise the
// happy path for every dispatched arm in `encode_instruction`.

use std::panic;

use llvm2_codegen::aarch64::encode::{encode_instruction, EncodeError};
use llvm2_ir::inst::{AArch64Opcode, InstFlags, MachInst};
use llvm2_ir::operand::MachOperand;
use llvm2_ir::regs::{PReg, SpecialReg};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Opcode strategy — full enumeration of every `AArch64Opcode` variant.
// ---------------------------------------------------------------------------
//
// The list is intentionally exhaustive (#450). `encode_instruction`
// dispatches by opcode; any variant we miss is a coverage gap, not a
// useful exclusion. Pseudo opcodes (`Phi`, `StackAlloc`, `Copy`, `Nop`,
// `Retain`, `Release`, `Trap*`) are included deliberately: they must
// return `Err(EncodeError::PseudoInstruction(..))` or
// `Err(EncodeError::UnsupportedOpcode(..))` rather than panic when they
// leak into the encoder.
fn opcode_strategy() -> impl Strategy<Value = AArch64Opcode> {
    use AArch64Opcode::*;
    // Split across multiple `prop_oneof!` calls because the macro has a
    // hard arity limit well below 178. Each sub-strategy is uniformly
    // weighted; we then compose them with another `prop_oneof!`.
    let arith_logical = prop_oneof![
        Just(AddRR), Just(AddRI), Just(SubRR), Just(SubRI),
        Just(MulRR), Just(Msub), Just(Smull), Just(Umull),
        Just(SDiv), Just(UDiv), Just(Neg),
        Just(AndRR), Just(AndRI), Just(OrrRR), Just(OrrRI),
        Just(EorRR), Just(EorRI), Just(OrnRR), Just(BicRR),
    ];
    let shifts_cmp_sel = prop_oneof![
        Just(LslRR), Just(LsrRR), Just(AsrRR),
        Just(LslRI), Just(LsrRI), Just(AsrRI),
        Just(CmpRR), Just(CmpRI), Just(Tst),
        Just(Csel), Just(Csinc), Just(Csinv), Just(Csneg), Just(CSet),
    ];
    let moves = prop_oneof![
        Just(MovR), Just(MovI), Just(Movz), Just(Movn), Just(Movk),
        Just(FmovImm),
    ];
    let mem_imm = prop_oneof![
        Just(LdrRI), Just(StrRI),
        Just(LdrbRI), Just(LdrhRI), Just(LdrsbRI), Just(LdrshRI),
        Just(StrbRI), Just(StrhRI),
        Just(LdrLiteral), Just(LdpRI), Just(StpRI),
        Just(StpPreIndex), Just(LdpPostIndex),
    ];
    let mem_reg_got = prop_oneof![
        Just(LdrRO), Just(StrRO),
        Just(LdrGot), Just(LdrTlvp),
        Just(LdrswRO),
    ];
    let branches = prop_oneof![
        Just(B), Just(BCond), Just(Cbz), Just(Cbnz),
        Just(Tbz), Just(Tbnz), Just(Br), Just(Bl),
        Just(Blr), Just(Ret),
    ];
    let extend_bitfield = prop_oneof![
        Just(Sxtw), Just(Uxtw), Just(Sxtb), Just(Sxth),
        Just(Uxtb), Just(Uxth),
        Just(Ubfm), Just(Sbfm), Just(Bfm),
    ];
    let fp_scalar = prop_oneof![
        Just(FaddRR), Just(FsubRR), Just(FmulRR), Just(FdivRR),
        Just(FnegRR), Just(FabsRR), Just(FsqrtRR), Just(Fcmp),
        Just(FcvtzsRR), Just(FcvtzuRR),
        Just(ScvtfRR), Just(UcvtfRR),
        Just(FcvtSD), Just(FcvtDS),
        Just(FmovGprFpr), Just(FmovFprGpr), Just(FmovFprFpr),
    ];
    let neon = prop_oneof![
        Just(NeonAddV), Just(NeonSubV), Just(NeonMulV),
        Just(NeonFaddV), Just(NeonFsubV), Just(NeonFmulV), Just(NeonFdivV),
        Just(NeonAndV), Just(NeonOrrV), Just(NeonEorV), Just(NeonBicV),
        Just(NeonNotV),
        Just(NeonCmeqV), Just(NeonCmgtV), Just(NeonCmgeV),
        Just(NeonDupElem), Just(NeonDupGen), Just(NeonInsGen),
        Just(NeonMovi),
        Just(NeonLd1Post), Just(NeonSt1Post),
    ];
    let atomic_lse = prop_oneof![
        Just(Ldar), Just(Ldarb), Just(Ldarh),
        Just(Stlr), Just(Stlrb), Just(Stlrh),
        Just(Ldadd), Just(Ldadda), Just(Ldaddal),
        Just(Ldclr), Just(Ldclral),
        Just(Ldeor), Just(Ldeoral),
        Just(Ldset), Just(Ldsetal),
        Just(Swp), Just(Swpal),
        Just(Cas), Just(Casa), Just(Casal),
        Just(Ldaxr), Just(Stlxr),
        Just(Dmb), Just(Dsb), Just(Isb),
    ];
    let addr_checked = prop_oneof![
        Just(Adrp), Just(Adr), Just(AddPCRel),
        Just(AddsRR), Just(AddsRI), Just(SubsRR), Just(SubsRI),
        Just(Adc), Just(Sbc),
        Just(Umulh), Just(Smulh), Just(Madd),
    ];
    let trap_rc_sysreg = prop_oneof![
        Just(TrapOverflow), Just(TrapBoundsCheck), Just(TrapNull),
        Just(TrapDivZero), Just(TrapShiftRange),
        Just(Retain), Just(Release),
        Just(Mrs),
    ];
    let llvm_aliases = prop_oneof![
        Just(MOVWrr), Just(MOVXrr),
        Just(STRWui), Just(STRXui), Just(STRSui), Just(STRDui),
        Just(BL), Just(BLR),
        Just(CMPWrr), Just(CMPXrr), Just(CMPWri), Just(CMPXri),
        Just(MOVZWi), Just(MOVZXi),
        Just(Bcc),
    ];
    let pseudos = prop_oneof![
        Just(Phi), Just(StackAlloc), Just(Copy), Just(Nop),
    ];

    prop_oneof![
        arith_logical,
        shifts_cmp_sel,
        moves,
        mem_imm,
        mem_reg_got,
        branches,
        extend_bitfield,
        fp_scalar,
        neon,
        atomic_lse,
        addr_checked,
        trap_rc_sysreg,
        llvm_aliases,
        pseudos,
    ]
}

// ---------------------------------------------------------------------------
// Operand strategy
// ---------------------------------------------------------------------------

fn preg_strategy() -> impl Strategy<Value = PReg> {
    // Cover the full legitimate PReg encoding range (0..=228) plus a small
    // beyond-the-end zone (229..=255) so we also exercise "unknown" regs
    // that may slip through from a buggy regalloc.
    (0u16..=255u16).prop_map(PReg::new)
}

/// Restricted GPR64 range (0..=30) — matches the canonical `X0..X30`
/// encodings used by the well-shaped category strategies below.
fn gpr_strategy() -> impl Strategy<Value = PReg> {
    (0u16..=30u16).prop_map(PReg::new)
}

/// Restricted FPR64 range (96..=126) — matches `D0..D30`. Used for FP
/// well-shaped instructions.
fn fpr_strategy() -> impl Strategy<Value = PReg> {
    (96u16..=126u16).prop_map(PReg::new)
}

/// Restricted FPR128 range (64..=94) — matches `V0..V30`. Used for NEON
/// well-shaped instructions.
fn vreg_strategy() -> impl Strategy<Value = PReg> {
    (64u16..=94u16).prop_map(PReg::new)
}

fn special_reg_strategy() -> impl Strategy<Value = SpecialReg> {
    prop_oneof![
        Just(SpecialReg::SP),
        Just(SpecialReg::XZR),
        Just(SpecialReg::WZR),
    ]
}

fn operand_strategy() -> impl Strategy<Value = MachOperand> {
    prop_oneof![
        preg_strategy().prop_map(MachOperand::PReg),
        // VRegs must not reach the encoder (it's post-regalloc) — but if
        // they do, we want Err, not panic.
        (0u32..=64u32).prop_map(|id| MachOperand::VReg(llvm2_ir::regs::VReg::new(
            id, llvm2_ir::regs::RegClass::Gpr64
        ))),
        any::<i64>().prop_map(MachOperand::Imm),
        // FImm — exclude NaN/inf is unnecessary: encoder must tolerate any f64.
        any::<u64>().prop_map(|bits| MachOperand::FImm(f64::from_bits(bits))),
        special_reg_strategy().prop_map(MachOperand::Special),
        (0u32..=64u32).prop_map(|id| MachOperand::Block(llvm2_ir::types::BlockId(id))),
        (preg_strategy(), any::<i64>()).prop_map(|(base, offset)| {
            MachOperand::MemOp { base, offset }
        }),
    ]
}

// ---------------------------------------------------------------------------
// MachInst strategy (malformed path — any opcode + any operand shape)
// ---------------------------------------------------------------------------

fn machinst_strategy() -> impl Strategy<Value = MachInst> {
    (
        opcode_strategy(),
        // 0..=8 operands: covers the realistic arity range. Too-few operands
        // must produce `EncodeError::MissingOperand`, not panic.
        prop::collection::vec(operand_strategy(), 0..=8),
    )
        // #447 closed: the filter that excluded {Lsl,Lsr,Asr}RI / Movk /
        // {Ubfm,Sbfm,Bfm} is no longer required — those dispatch arms now
        // return `Err(EncodeError::InvalidOperand)` instead of debug-
        // overflowing on bad shifts, and `encode_move_wide` no longer trips
        // `debug_assert!(hw <= 0b11)` on oversized `Movk` shift operands.
        .prop_map(|(opcode, operands)| MachInst {
            opcode,
            operands,
            implicit_defs: &[],
            implicit_uses: &[],
            flags: InstFlags::EMPTY,
            proof: None,
            source_loc: None,
        })
}

// ---------------------------------------------------------------------------
// Well-shaped MachInst strategy (happy path — per-category shape fns)
// ---------------------------------------------------------------------------
//
// Each category strategy yields `(opcode, Vec<MachOperand>)` with operand
// shapes that plausibly match the opcode. We keep each function flat
// (no shared state) so proptest's `Map` clone-bound is trivially met.
//
// The overall `valid_machinst_strategy` composes them via `prop_oneof!`.

type OpPair = (AArch64Opcode, Vec<MachOperand>);

fn strat_arith_rr() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    (
        prop_oneof![
            Just(AddRR), Just(SubRR), Just(MulRR),
            Just(AndRR), Just(OrrRR), Just(EorRR),
            Just(OrnRR), Just(BicRR),
            Just(LslRR), Just(LsrRR), Just(AsrRR),
            Just(AddsRR), Just(SubsRR),
            Just(Adc), Just(Sbc),
            Just(Umulh), Just(Smulh),
            Just(Smull), Just(Umull),
            Just(SDiv), Just(UDiv),
        ],
        gpr_strategy(), gpr_strategy(), gpr_strategy(),
    ).prop_map(|(op, a, b, c)| (op, vec![
        MachOperand::PReg(a), MachOperand::PReg(b), MachOperand::PReg(c),
    ]))
}

fn strat_arith_ri() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    (
        prop_oneof![
            Just(AddRI), Just(SubRI),
            Just(AndRI), Just(OrrRI), Just(EorRI),
            Just(AddsRI), Just(SubsRI),
        ],
        gpr_strategy(), gpr_strategy(), -4096i64..=4095i64,
    ).prop_map(|(op, a, b, imm)| (op, vec![
        MachOperand::PReg(a), MachOperand::PReg(b), MachOperand::Imm(imm),
    ]))
}

fn strat_arith_rrr_madd() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // MADD/MSUB Rd, Rn, Rm, Ra — 4-register shapes.
    (
        prop_oneof![Just(Madd), Just(Msub)],
        gpr_strategy(), gpr_strategy(), gpr_strategy(), gpr_strategy(),
    ).prop_map(|(op, a, b, c, d)| (op, vec![
        MachOperand::PReg(a), MachOperand::PReg(b),
        MachOperand::PReg(c), MachOperand::PReg(d),
    ]))
}

fn strat_shift_ri() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // Shift-immediate: 2 GPR + arbitrary i64 shift (mixes legal 0..=63
    // values with out-of-range ones — encoder must return Err for the
    // latter without panicking, #447).
    (
        prop_oneof![Just(LslRI), Just(LsrRI), Just(AsrRI)],
        gpr_strategy(), gpr_strategy(), any::<i64>(),
    ).prop_map(|(op, a, b, imm)| (op, vec![
        MachOperand::PReg(a), MachOperand::PReg(b), MachOperand::Imm(imm),
    ]))
}

fn strat_bitfield() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // UBFM/SBFM/BFM Rd, Rn, #immr, #imms — 2 GPR + 2 shift immediates.
    (
        prop_oneof![Just(Ubfm), Just(Sbfm), Just(Bfm)],
        gpr_strategy(), gpr_strategy(), 0i64..=63i64, 0i64..=63i64,
    ).prop_map(|(op, a, b, immr, imms)| (op, vec![
        MachOperand::PReg(a), MachOperand::PReg(b),
        MachOperand::Imm(immr), MachOperand::Imm(imms),
    ]))
}

fn strat_extend() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // SXTB/SXTH/SXTW/UXTB/UXTH/UXTW Rd, Rn — 2 GPR.
    (
        prop_oneof![
            Just(Sxtw), Just(Uxtw), Just(Sxtb), Just(Sxth),
            Just(Uxtb), Just(Uxth), Just(Neg),
        ],
        gpr_strategy(), gpr_strategy(),
    ).prop_map(|(op, a, b)| (op, vec![
        MachOperand::PReg(a), MachOperand::PReg(b),
    ]))
}

fn strat_cmp_rr() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    (
        prop_oneof![
            Just(CmpRR), Just(Tst),
            Just(CMPWrr), Just(CMPXrr),
        ],
        gpr_strategy(), gpr_strategy(),
    ).prop_map(|(op, a, b)| (op, vec![
        MachOperand::PReg(a), MachOperand::PReg(b),
    ]))
}

fn strat_cmp_ri() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    (
        prop_oneof![
            Just(CmpRI), Just(CMPWri), Just(CMPXri),
        ],
        gpr_strategy(), -4096i64..=4095i64,
    ).prop_map(|(op, a, imm)| (op, vec![
        MachOperand::PReg(a), MachOperand::Imm(imm),
    ]))
}

fn strat_csel() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // CSEL/CSINC/CSINV/CSNEG Xd, Xn, Xm, #cond (cond 0..=15).
    (
        prop_oneof![Just(Csel), Just(Csinc), Just(Csinv), Just(Csneg)],
        gpr_strategy(), gpr_strategy(), gpr_strategy(), 0i64..=15i64,
    ).prop_map(|(op, a, b, c, cc)| (op, vec![
        MachOperand::PReg(a), MachOperand::PReg(b),
        MachOperand::PReg(c), MachOperand::Imm(cc),
    ]))
}

fn strat_cset() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // CSET Xd, #cond.
    (
        gpr_strategy(), 0i64..=15i64,
    ).prop_map(|(a, cc)| (CSet, vec![
        MachOperand::PReg(a), MachOperand::Imm(cc),
    ]))
}

fn strat_mov_reg() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    (
        prop_oneof![Just(MovR), Just(MOVWrr), Just(MOVXrr)],
        gpr_strategy(), gpr_strategy(),
    ).prop_map(|(op, a, b)| (op, vec![
        MachOperand::PReg(a), MachOperand::PReg(b),
    ]))
}

fn strat_mov_imm() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // Move-immediate (16-bit): MOVZ/MOVN/MOVI + LLVM aliases, without shift.
    (
        prop_oneof![
            Just(Movz), Just(Movn), Just(MovI),
            Just(MOVZWi), Just(MOVZXi),
        ],
        gpr_strategy(), 0i64..=0xFFFFi64,
    ).prop_map(|(op, a, imm)| (op, vec![
        MachOperand::PReg(a), MachOperand::Imm(imm),
    ]))
}

fn strat_movk() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // MOVK with full-range shift operand. Valid shifts are {0,16,32,48};
    // anything else must yield `Err(EncodeError::InvalidOperand)`
    // rather than panic (#447).
    (
        gpr_strategy(), 0i64..=0xFFFFi64, any::<i64>(),
    ).prop_map(|(a, imm, shift)| (Movk, vec![
        MachOperand::PReg(a), MachOperand::Imm(imm), MachOperand::Imm(shift),
    ]))
}

fn strat_mem_imm() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // Loads/stores with immediate offset: [Rt, Rn, Imm(offset)].
    (
        prop_oneof![
            Just(LdrRI), Just(StrRI),
            Just(LdrbRI), Just(LdrhRI), Just(LdrsbRI), Just(LdrshRI),
            Just(StrbRI), Just(StrhRI),
            Just(STRWui), Just(STRXui),
        ],
        gpr_strategy(), gpr_strategy(), -256i64..=4095i64,
    ).prop_map(|(op, rt, rn, off)| (op, vec![
        MachOperand::PReg(rt), MachOperand::PReg(rn), MachOperand::Imm(off),
    ]))
}

fn strat_mem_fpr_imm() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // FP stores with immediate offset: [St/Dt, Rn, Imm(offset)].
    (
        prop_oneof![Just(STRSui), Just(STRDui)],
        fpr_strategy(), gpr_strategy(), -256i64..=4095i64,
    ).prop_map(|(op, rt, rn, off)| (op, vec![
        MachOperand::PReg(rt), MachOperand::PReg(rn), MachOperand::Imm(off),
    ]))
}

fn strat_mem_pair() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // LDP/STP pair: [Rt, Rt2, Rn, Imm(offset)].
    (
        prop_oneof![Just(LdpRI), Just(StpRI), Just(StpPreIndex), Just(LdpPostIndex)],
        gpr_strategy(), gpr_strategy(), gpr_strategy(), -256i64..=252i64,
    ).prop_map(|(op, rt, rt2, rn, off)| (op, vec![
        MachOperand::PReg(rt), MachOperand::PReg(rt2),
        MachOperand::PReg(rn), MachOperand::Imm(off),
    ]))
}

fn strat_mem_reg_off() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // LDR/STR with register offset: [Rt, Rn, Rm].
    (
        prop_oneof![Just(LdrRO), Just(StrRO), Just(LdrswRO)],
        gpr_strategy(), gpr_strategy(), gpr_strategy(),
    ).prop_map(|(op, rt, rn, rm)| (op, vec![
        MachOperand::PReg(rt), MachOperand::PReg(rn), MachOperand::PReg(rm),
    ]))
}

fn strat_mem_got_literal() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // Loads with PC-relative or GOT/TLV offset: [Rt, Imm(offset)].
    (
        prop_oneof![Just(LdrLiteral), Just(LdrGot), Just(LdrTlvp)],
        gpr_strategy(), any::<i64>(),
    ).prop_map(|(op, rt, imm)| (op, vec![
        MachOperand::PReg(rt), MachOperand::Imm(imm),
    ]))
}

fn strat_branch_block() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // Unconditional / linked branches to a block: [Block(id)].
    (
        prop_oneof![Just(Br), Just(B), Just(Bl), Just(BL), Just(Blr), Just(BLR)],
        0u32..=16u32,
    ).prop_map(|(op, id)| (op, vec![
        MachOperand::Block(llvm2_ir::types::BlockId(id)),
    ]))
}

fn strat_branch_cond() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // B.cond: [Imm(cond), Block].
    (
        prop_oneof![Just(BCond), Just(Bcc)],
        0i64..=15i64, 0u32..=16u32,
    ).prop_map(|(op, cc, id)| (op, vec![
        MachOperand::Imm(cc),
        MachOperand::Block(llvm2_ir::types::BlockId(id)),
    ]))
}

fn strat_branch_cbz() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // CBZ / CBNZ: [Rt, Block].
    (
        prop_oneof![Just(Cbz), Just(Cbnz)],
        gpr_strategy(), 0u32..=16u32,
    ).prop_map(|(op, rt, id)| (op, vec![
        MachOperand::PReg(rt),
        MachOperand::Block(llvm2_ir::types::BlockId(id)),
    ]))
}

fn strat_branch_tbz() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // TBZ / TBNZ: [Rt, Imm(bit), Block].
    (
        prop_oneof![Just(Tbz), Just(Tbnz)],
        gpr_strategy(), 0i64..=63i64, 0u32..=16u32,
    ).prop_map(|(op, rt, bit, id)| (op, vec![
        MachOperand::PReg(rt),
        MachOperand::Imm(bit),
        MachOperand::Block(llvm2_ir::types::BlockId(id)),
    ]))
}

fn strat_ret() -> impl Strategy<Value = OpPair> {
    Just((AArch64Opcode::Ret, Vec::<MachOperand>::new())).boxed()
}

fn strat_fp_rr() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // FP binary: Dd = Dn op Dm.
    (
        prop_oneof![Just(FaddRR), Just(FsubRR), Just(FmulRR), Just(FdivRR)],
        fpr_strategy(), fpr_strategy(), fpr_strategy(),
    ).prop_map(|(op, d, n, m)| (op, vec![
        MachOperand::PReg(d), MachOperand::PReg(n), MachOperand::PReg(m),
    ]))
}

fn strat_fp_unary() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    (
        prop_oneof![Just(FnegRR), Just(FabsRR), Just(FsqrtRR)],
        fpr_strategy(), fpr_strategy(),
    ).prop_map(|(op, d, n)| (op, vec![
        MachOperand::PReg(d), MachOperand::PReg(n),
    ]))
}

fn strat_fcmp() -> impl Strategy<Value = OpPair> {
    (
        fpr_strategy(), fpr_strategy(),
    ).prop_map(|(n, m)| (AArch64Opcode::Fcmp, vec![
        MachOperand::PReg(n), MachOperand::PReg(m),
    ]))
}

fn strat_fp_cvt_f2i() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // FP → int: Xd/Wd = cvt(Dn/Sn).
    (
        prop_oneof![Just(FcvtzsRR), Just(FcvtzuRR)],
        gpr_strategy(), fpr_strategy(),
    ).prop_map(|(op, d, n)| (op, vec![
        MachOperand::PReg(d), MachOperand::PReg(n),
    ]))
}

fn strat_fp_cvt_i2f() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // int → FP: Dd = cvt(Xn/Wn).
    (
        prop_oneof![Just(ScvtfRR), Just(UcvtfRR)],
        fpr_strategy(), gpr_strategy(),
    ).prop_map(|(op, d, n)| (op, vec![
        MachOperand::PReg(d), MachOperand::PReg(n),
    ]))
}

fn strat_fcvt_precision() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // FP precision widen/narrow: Dd/Sd = cvt(Sn/Dn).
    (
        prop_oneof![Just(FcvtSD), Just(FcvtDS)],
        fpr_strategy(), fpr_strategy(),
    ).prop_map(|(op, d, n)| (op, vec![
        MachOperand::PReg(d), MachOperand::PReg(n),
    ]))
}

fn strat_fmov_cross() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // FMOV between GPR and FPR. Shape: [dst, src] — which class is GPR
    // vs. FPR depends on the variant, but for fuzz purposes either
    // assignment is an interesting coverage point.
    (
        prop_oneof![Just(FmovGprFpr), Just(FmovFprGpr)],
        gpr_strategy(), fpr_strategy(),
    ).prop_map(|(op, g, f)| match op {
        // FmovGprFpr: dst=FPR, src=GPR  (e.g., FMOV Dd, Xn).
        FmovGprFpr => (op, vec![MachOperand::PReg(f), MachOperand::PReg(g)]),
        // FmovFprGpr: dst=GPR, src=FPR  (e.g., FMOV Xn, Dd).
        _          => (op, vec![MachOperand::PReg(g), MachOperand::PReg(f)]),
    })
}

fn strat_fmov_fpr_fpr() -> impl Strategy<Value = OpPair> {
    (
        fpr_strategy(), fpr_strategy(),
    ).prop_map(|(d, n)| (AArch64Opcode::FmovFprFpr, vec![
        MachOperand::PReg(d), MachOperand::PReg(n),
    ]))
}

fn strat_fmov_imm() -> impl Strategy<Value = OpPair> {
    (
        fpr_strategy(), any::<u64>(),
    ).prop_map(|(d, bits)| (AArch64Opcode::FmovImm, vec![
        MachOperand::PReg(d),
        MachOperand::FImm(f64::from_bits(bits)),
    ]))
}

fn strat_neon_rrr() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // NEON 3-vector: [Vd, Vn, Vm, Imm(arrangement)].
    // Arrangement encoding 0..=7: 8B,16B,4H,8H,2S,4S,2D (0..=6 are
    // commonly used; we fuzz 0..=7 to exercise boundary handling).
    (
        prop_oneof![
            Just(NeonAddV), Just(NeonSubV), Just(NeonMulV),
            Just(NeonFaddV), Just(NeonFsubV), Just(NeonFmulV), Just(NeonFdivV),
            Just(NeonAndV), Just(NeonOrrV), Just(NeonEorV), Just(NeonBicV),
            Just(NeonCmeqV), Just(NeonCmgtV), Just(NeonCmgeV),
        ],
        vreg_strategy(), vreg_strategy(), vreg_strategy(), 0i64..=7i64,
    ).prop_map(|(op, d, n, m, arr)| (op, vec![
        MachOperand::PReg(d), MachOperand::PReg(n), MachOperand::PReg(m),
        MachOperand::Imm(arr),
    ]))
}

fn strat_neon_not() -> impl Strategy<Value = OpPair> {
    (
        vreg_strategy(), vreg_strategy(), 0i64..=7i64,
    ).prop_map(|(d, n, arr)| (AArch64Opcode::NeonNotV, vec![
        MachOperand::PReg(d), MachOperand::PReg(n), MachOperand::Imm(arr),
    ]))
}

fn strat_neon_dup_gen() -> impl Strategy<Value = OpPair> {
    (
        vreg_strategy(), gpr_strategy(), 0i64..=3i64,
    ).prop_map(|(d, n, esize)| (AArch64Opcode::NeonDupGen, vec![
        MachOperand::PReg(d), MachOperand::PReg(n), MachOperand::Imm(esize),
    ]))
}

fn strat_neon_dup_elem() -> impl Strategy<Value = OpPair> {
    (
        vreg_strategy(), vreg_strategy(), 0i64..=15i64, 0i64..=3i64,
    ).prop_map(|(d, n, lane, esize)| (AArch64Opcode::NeonDupElem, vec![
        MachOperand::PReg(d), MachOperand::PReg(n),
        MachOperand::Imm(lane), MachOperand::Imm(esize),
    ]))
}

fn strat_neon_ins_gen() -> impl Strategy<Value = OpPair> {
    (
        vreg_strategy(), gpr_strategy(), 0i64..=15i64, 0i64..=3i64,
    ).prop_map(|(d, n, lane, esize)| (AArch64Opcode::NeonInsGen, vec![
        MachOperand::PReg(d), MachOperand::PReg(n),
        MachOperand::Imm(lane), MachOperand::Imm(esize),
    ]))
}

fn strat_neon_movi() -> impl Strategy<Value = OpPair> {
    (
        vreg_strategy(), 0i64..=255i64,
    ).prop_map(|(d, imm)| (AArch64Opcode::NeonMovi, vec![
        MachOperand::PReg(d), MachOperand::Imm(imm),
    ]))
}

fn strat_neon_ld1_st1_post() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // LD1/ST1 post-indexed: [Vt, Xn, Imm(arrangement)].
    (
        prop_oneof![Just(NeonLd1Post), Just(NeonSt1Post)],
        vreg_strategy(), gpr_strategy(), 0i64..=7i64,
    ).prop_map(|(op, vt, rn, arr)| (op, vec![
        MachOperand::PReg(vt), MachOperand::PReg(rn), MachOperand::Imm(arr),
    ]))
}

fn strat_atomic_load_acquire() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // LDAR/LDARB/LDARH/LDAXR: [Rt, Rn].
    (
        prop_oneof![Just(Ldar), Just(Ldarb), Just(Ldarh), Just(Ldaxr)],
        gpr_strategy(), gpr_strategy(),
    ).prop_map(|(op, rt, rn)| (op, vec![
        MachOperand::PReg(rt), MachOperand::PReg(rn),
    ]))
}

fn strat_atomic_store_release() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // STLR/STLRB/STLRH: [Rt, Rn].
    (
        prop_oneof![Just(Stlr), Just(Stlrb), Just(Stlrh)],
        gpr_strategy(), gpr_strategy(),
    ).prop_map(|(op, rt, rn)| (op, vec![
        MachOperand::PReg(rt), MachOperand::PReg(rn),
    ]))
}

fn strat_atomic_stlxr() -> impl Strategy<Value = OpPair> {
    // STLXR Ws, Xt, [Xn]: [Ws, Rt, Rn].
    (
        gpr_strategy(), gpr_strategy(), gpr_strategy(),
    ).prop_map(|(ws, rt, rn)| (AArch64Opcode::Stlxr, vec![
        MachOperand::PReg(ws), MachOperand::PReg(rt), MachOperand::PReg(rn),
    ]))
}

fn strat_atomic_lse_rmw() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // LDADD/LDCLR/LDEOR/LDSET/SWP/CAS family: [Rs, Rt, Rn].
    (
        prop_oneof![
            Just(Ldadd), Just(Ldadda), Just(Ldaddal),
            Just(Ldclr), Just(Ldclral),
            Just(Ldeor), Just(Ldeoral),
            Just(Ldset), Just(Ldsetal),
            Just(Swp), Just(Swpal),
            Just(Cas), Just(Casa), Just(Casal),
        ],
        gpr_strategy(), gpr_strategy(), gpr_strategy(),
    ).prop_map(|(op, rs, rt, rn)| (op, vec![
        MachOperand::PReg(rs), MachOperand::PReg(rt), MachOperand::PReg(rn),
    ]))
}

fn strat_barrier() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // DMB/DSB/ISB: [Imm(option)].
    (
        prop_oneof![Just(Dmb), Just(Dsb), Just(Isb)],
        0i64..=0xFi64,
    ).prop_map(|(op, opt)| (op, vec![
        MachOperand::Imm(opt),
    ]))
}

fn strat_address() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // ADRP/ADR/ADD_PCREL: [Rd, Imm(offset)].
    (
        prop_oneof![Just(Adrp), Just(Adr), Just(AddPCRel)],
        gpr_strategy(), any::<i64>(),
    ).prop_map(|(op, rd, off)| (op, vec![
        MachOperand::PReg(rd), MachOperand::Imm(off),
    ]))
}

fn strat_mrs() -> impl Strategy<Value = OpPair> {
    (
        gpr_strategy(), 0i64..=0xFFFFi64,
    ).prop_map(|(rd, enc)| (AArch64Opcode::Mrs, vec![
        MachOperand::PReg(rd), MachOperand::Imm(enc),
    ]))
}

fn strat_trap_cond() -> impl Strategy<Value = OpPair> {
    // TrapOverflow: [Imm(cond), Block].
    (
        0i64..=15i64, 0u32..=16u32,
    ).prop_map(|(cc, id)| (AArch64Opcode::TrapOverflow, vec![
        MachOperand::Imm(cc),
        MachOperand::Block(llvm2_ir::types::BlockId(id)),
    ]))
}

fn strat_trap_block() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // TrapBoundsCheck/Null/DivZero/ShiftRange: [Block].
    (
        prop_oneof![
            Just(TrapBoundsCheck), Just(TrapNull),
            Just(TrapDivZero), Just(TrapShiftRange),
        ],
        0u32..=16u32,
    ).prop_map(|(op, id)| (op, vec![
        MachOperand::Block(llvm2_ir::types::BlockId(id)),
    ]))
}

fn strat_rc_pseudo() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // Retain/Release: [ptr].
    (
        prop_oneof![Just(Retain), Just(Release)],
        gpr_strategy(),
    ).prop_map(|(op, p)| (op, vec![
        MachOperand::PReg(p),
    ]))
}

fn strat_basic_pseudo() -> impl Strategy<Value = OpPair> {
    use AArch64Opcode::*;
    // Pseudo opcodes that must cleanly return Err (not panic).
    prop_oneof![
        Just((Phi, Vec::<MachOperand>::new())),
        Just((StackAlloc, Vec::<MachOperand>::new())),
        Just((Copy, Vec::<MachOperand>::new())),
        Just((Nop, Vec::<MachOperand>::new())),
    ]
}

/// "Well-shaped" generator: produces a MachInst with an operand count that
/// at least plausibly matches the opcode. Exercises the happy path so any
/// panic here is a definite P1 bug (the corresponding malformed-input path
/// below catches latent panics in error-handling code).
///
/// #447 closed: shift immediates for `{Lsl,Lsr,Asr}RI` and `Movk` are now
/// validated at the encoder boundary. The arbitrary-`i64` shift ranges
/// below are intentional fuzz input — the encoder converts out-of-range
/// values to `Err(EncodeError::InvalidOperand)`, which `assert_no_panic`
/// accepts as a non-panic outcome.
fn valid_machinst_strategy() -> impl Strategy<Value = MachInst> {
    // Proptest requires uniform branch types for `prop_oneof!`. Boxing
    // each per-category strategy to `BoxedStrategy<OpPair>` unifies them.
    let cats: Vec<BoxedStrategy<OpPair>> = vec![
        strat_arith_rr().boxed(),
        strat_arith_ri().boxed(),
        strat_arith_rrr_madd().boxed(),
        strat_shift_ri().boxed(),
        strat_bitfield().boxed(),
        strat_extend().boxed(),
        strat_cmp_rr().boxed(),
        strat_cmp_ri().boxed(),
        strat_csel().boxed(),
        strat_cset().boxed(),
        strat_mov_reg().boxed(),
        strat_mov_imm().boxed(),
        strat_movk().boxed(),
        strat_mem_imm().boxed(),
        strat_mem_fpr_imm().boxed(),
        strat_mem_pair().boxed(),
        strat_mem_reg_off().boxed(),
        strat_mem_got_literal().boxed(),
        strat_branch_block().boxed(),
        strat_branch_cond().boxed(),
        strat_branch_cbz().boxed(),
        strat_branch_tbz().boxed(),
        strat_ret().boxed(),
        strat_fp_rr().boxed(),
        strat_fp_unary().boxed(),
        strat_fcmp().boxed(),
        strat_fp_cvt_f2i().boxed(),
        strat_fp_cvt_i2f().boxed(),
        strat_fcvt_precision().boxed(),
        strat_fmov_cross().boxed(),
        strat_fmov_fpr_fpr().boxed(),
        strat_fmov_imm().boxed(),
        strat_neon_rrr().boxed(),
        strat_neon_not().boxed(),
        strat_neon_dup_gen().boxed(),
        strat_neon_dup_elem().boxed(),
        strat_neon_ins_gen().boxed(),
        strat_neon_movi().boxed(),
        strat_neon_ld1_st1_post().boxed(),
        strat_atomic_load_acquire().boxed(),
        strat_atomic_store_release().boxed(),
        strat_atomic_stlxr().boxed(),
        strat_atomic_lse_rmw().boxed(),
        strat_barrier().boxed(),
        strat_address().boxed(),
        strat_mrs().boxed(),
        strat_trap_cond().boxed(),
        strat_trap_block().boxed(),
        strat_rc_pseudo().boxed(),
        strat_basic_pseudo().boxed(),
    ];
    proptest::strategy::Union::new(cats)
        .prop_map(|(opcode, operands)| MachInst {
            opcode,
            operands,
            implicit_defs: &[],
            implicit_uses: &[],
            flags: InstFlags::EMPTY,
            proof: None,
            source_loc: None,
        })
}

// ---------------------------------------------------------------------------
// Property
// ---------------------------------------------------------------------------

/// Run `encode_instruction` inside `catch_unwind` and assert no panic.
///
/// The returned `Result<u32, EncodeError>` is discarded; the *only*
/// failure mode tracked here is "panic reached the caller".
fn assert_no_panic(inst: &MachInst) {
    // Clone inst into the closure so `catch_unwind`'s UnwindSafe bound is
    // trivially satisfied for this POD-ish type.
    let inst_copy = inst.clone();
    let result = panic::catch_unwind(panic::AssertUnwindSafe(move || {
        let _ = encode_instruction(&inst_copy);
    }));
    if let Err(payload) = result {
        let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".to_string()
        };
        panic!(
            "encode_instruction panicked on input {:?}: {}",
            inst, msg
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256),
        max_shrink_iters: 500,
        .. ProptestConfig::default()
    })]

    /// Random-but-plausible inputs — the happy path.
    #[test]
    fn encode_never_panics_on_valid(inst in valid_machinst_strategy()) {
        assert_no_panic(&inst);
    }

    /// Random-malformed inputs — arbitrary opcode paired with arbitrary
    /// operand shapes. This is the main totality property: the encoder
    /// should convert *any* malformed shape into `Err`, never a panic.
    #[test]
    fn encode_never_panics_on_malformed(inst in machinst_strategy()) {
        assert_no_panic(&inst);
    }
}

// ---------------------------------------------------------------------------
// Static coverage sanity check (#450 acceptance criterion)
// ---------------------------------------------------------------------------
//
// Guardrail test: if a future refactor narrows `opcode_strategy()` below
// 80% of the 178 `AArch64Opcode` variants, this test fails loudly.
// Implemented by exhaustively sampling the strategy until a fixed cap
// and counting distinct opcodes seen.
#[test]
fn opcode_strategy_covers_at_least_80_percent() {
    use std::collections::HashSet;
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;

    let mut runner = TestRunner::deterministic();
    let strat = opcode_strategy();
    let mut seen: HashSet<AArch64Opcode> = HashSet::new();
    // 10k draws is plenty to cover 178 uniformly-weighted variants under
    // the current sub-strategy composition.
    for _ in 0..10_000 {
        let tree = strat.new_tree(&mut runner).expect("strategy tree");
        seen.insert(tree.current());
        if seen.len() >= 178 {
            break;
        }
    }
    // 142 / 178 = 79.78%; require ≥ 142 for the 80% floor (#450).
    assert!(
        seen.len() >= 142,
        "opcode_strategy only samples {} distinct variants; need ≥ 142 (80% of 178)",
        seen.len()
    );
}

// ---------------------------------------------------------------------------
// Regression reproducers for known panics found by this harness
// ---------------------------------------------------------------------------
//
// These pin-down tests are hand-reduced shrinks of failing proptest cases.
// They previously pinned buggy behavior with `#[should_panic]`; they now
// assert the post-fix behavior (a typed `Err(EncodeError::..)`). See #447.

/// `LslRI` with a negative shift immediate previously triggered a debug
/// `attempt to subtract with overflow` at `aarch64/encode.rs`. Fixed under
/// #447: the dispatch arm now returns
/// `Err(EncodeError::InvalidOperand { .. })` for any shift outside
/// `[0, regsize)`.
#[test]
fn regression_lslri_negative_shift_panics() {
    let inst = MachInst {
        opcode: AArch64Opcode::LslRI,
        operands: vec![
            MachOperand::PReg(PReg::new(0)),   // X0
            MachOperand::PReg(PReg::new(1)),   // X1
            MachOperand::Imm(-1),
        ],
        implicit_defs: &[],
        implicit_uses: &[],
        flags: InstFlags::EMPTY,
        proof: None,
        source_loc: None,
    };
    let err = encode_instruction(&inst).expect_err("negative shift must be rejected");
    assert!(
        matches!(
            err,
            EncodeError::InvalidOperand { opcode: AArch64Opcode::LslRI, index: 2, .. }
        ),
        "expected InvalidOperand on operand 2, got {err:?}"
    );
}

/// `LslRI` with shift ≥ regsize previously had the same overflow
/// characteristic. Fixed under #447.
#[test]
fn regression_lslri_shift_ge_regsize_panics() {
    let inst = MachInst {
        opcode: AArch64Opcode::LslRI,
        operands: vec![
            MachOperand::PReg(PReg::new(0)),
            MachOperand::PReg(PReg::new(1)),
            MachOperand::Imm(64),
        ],
        implicit_defs: &[],
        implicit_uses: &[],
        flags: InstFlags::EMPTY,
        proof: None,
        source_loc: None,
    };
    let err = encode_instruction(&inst).expect_err("shift >= regsize must be rejected");
    assert!(
        matches!(
            err,
            EncodeError::InvalidOperand { opcode: AArch64Opcode::LslRI, index: 2, .. }
        ),
        "expected InvalidOperand on operand 2, got {err:?}"
    );
}

/// `Movk` with a shift operand whose `shift / 16` value exceeds 3 previously
/// tripped `debug_assert!(hw <= 0b11)` inside `encode_move_wide`. Fixed
/// under #447: `Movk`'s dispatch arm validates that the shift operand is a
/// non-negative multiple of 16 with `shift/16 ∈ 0..=3`, and
/// `encode_move_wide` no longer asserts on `hw` (masks defensively).
#[test]
fn regression_movk_oversized_shift_panics() {
    let inst = MachInst {
        opcode: AArch64Opcode::Movk,
        operands: vec![
            MachOperand::PReg(PReg::new(0)),
            MachOperand::Imm(0x1234),
            MachOperand::Imm(128), // shift/16 = 8, overflows hw field
        ],
        implicit_defs: &[],
        implicit_uses: &[],
        flags: InstFlags::EMPTY,
        proof: None,
        source_loc: None,
    };
    let err = encode_instruction(&inst).expect_err("oversized MOVK shift must be rejected");
    assert!(
        matches!(
            err,
            EncodeError::InvalidOperand { opcode: AArch64Opcode::Movk, index: 2, .. }
        ),
        "expected InvalidOperand on operand 2, got {err:?}"
    );
}
