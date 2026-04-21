// llvm2-codegen/tests/panic_fuzz_encode_x86_64.rs
// Property-based panic-fuzz harness for `X86Encoder::encode_instruction` (x86-64).
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
//! Part of #473 (x86-64 encoder panic-fuzz) / Part of #445 (per-target
//! harness gap) / Lineage: #387 (proptest panic-fuzz), #447 (panic-fix
//! hardening), #450 (widen opcode coverage >= 80%).
//!
//! This is the x86-64 sibling of `panic_fuzz_encode.rs`.
//
// Reference: `designs/2026-04-18-crash-free-codegen-plan.md` §5 (proptest
// as primary defense) and §6 (per-crate harness).
//
// Contract under test: for *any* `(X86Opcode, X86InstOperands)` value,
// `X86Encoder::encode_instruction` must either return `Ok(usize)` or
// `Err(X86EncodeError)` — it must NEVER panic, abort, overflow (in
// debug), or trigger a slice-index panic. This is the empirical half of
// the Phase-1 boundary conversion: Phase 1 replaced in-function
// `unwrap()` / `unreachable!()` sites with typed `X86EncodeError`
// returns; this harness proves the replacement is exhaustive over
// random-but-valid and random-malformed inputs.
//
// Run:
//   cargo test -p llvm2-codegen --test panic_fuzz_encode_x86_64
// Increase case count via env:
//   PROPTEST_CASES=100000 cargo test -p llvm2-codegen --test panic_fuzz_encode_x86_64
//
// ---------------------------------------------------------------------------
// Coverage (#473 / #450)
// ---------------------------------------------------------------------------
//
// `opcode_strategy()` below enumerates **every** variant of `X86Opcode`
// (101 as of 2026-04-20), giving 100% static opcode coverage. This
// closes the x86-64 sibling gap tracked under #445 while preserving the
// same ≥80% guardrail pattern introduced under #450.
//
// `valid_operands_strategy()` below composes per-category "well-shaped"
// strategies (arithmetic, logical, shift, move, memory base + SIB +
// RIP-relative, compare/test, branch, SSE scalar + conversion, CMOV/SET,
// bit-manip, atomic, GPR↔XMM transfers, stack, pseudos, hardware NOP).
// These exercise the happy path for every dispatched arm in
// `encode_instruction`.

use std::panic;

use llvm2_codegen::x86_64::encode::*;
use llvm2_ir::x86_64_ops::*;
use llvm2_ir::x86_64_regs::*;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Opcode strategy — full enumeration of every `X86Opcode` variant.
// ---------------------------------------------------------------------------
//
// The list is intentionally exhaustive (#450). `encode_instruction`
// dispatches by opcode; any variant we miss is a coverage gap, not a
// useful exclusion. Pseudo opcodes (`Phi`, `StackAlloc`, `Nop`) are
// included deliberately: they must return `Ok(0)` or
// `Err(X86EncodeError::UnsupportedOpcode(..))` rather than panic when
// they leak into the encoder.
fn opcode_strategy() -> impl Strategy<Value = X86Opcode> {
    use X86Opcode::*;
    // Split across multiple `prop_oneof!` calls because the macro has a
    // hard arity limit well below 101. Each sub-strategy is uniformly
    // weighted; we then compose them with another `prop_oneof!`.
    let arith = prop_oneof![
        Just(AddRR),
        Just(AddRI),
        Just(AddRM),
        Just(SubRR),
        Just(SubRI),
        Just(SubRM),
        Just(ImulRR),
        Just(ImulRRI),
        Just(ImulRM),
        Just(Idiv),
        Just(Div),
        Just(Neg),
        Just(Inc),
        Just(Dec),
        Just(Cdq),
        Just(Cqo),
    ];
    let logical = prop_oneof![
        Just(AndRR),
        Just(AndRI),
        Just(OrRR),
        Just(OrRI),
        Just(XorRR),
        Just(XorRI),
        Just(Not),
    ];
    let shifts = prop_oneof![
        Just(ShlRR),
        Just(ShlRI),
        Just(ShrRR),
        Just(ShrRI),
        Just(SarRR),
        Just(SarRI),
    ];
    let moves = prop_oneof![
        Just(MovRR),
        Just(MovRI),
        Just(MovRM),
        Just(MovMR),
        Just(Movzx),
        Just(MovzxW),
        Just(MovsxB),
        Just(MovsxW),
        Just(Movsx),
        Just(Lea),
        Just(LeaSib),
        Just(MovRMSib),
        Just(MovMRSib),
        Just(LeaRip),
    ];
    let compare = prop_oneof![
        Just(CmpRR),
        Just(CmpRI),
        Just(CmpRI8),
        Just(CmpRM),
        Just(TestRR),
        Just(TestRI),
        Just(TestRM),
    ];
    let branch = prop_oneof![
        Just(Jmp),
        Just(Jcc),
        Just(Call),
        Just(CallR),
        Just(CallM),
        Just(Ret),
    ];
    let sse_scalar = prop_oneof![
        Just(Addsd),
        Just(Subsd),
        Just(Mulsd),
        Just(Divsd),
        Just(MovsdRR),
        Just(MovsdRM),
        Just(MovsdMR),
        Just(Ucomisd),
        Just(Addss),
        Just(Subss),
        Just(Mulss),
        Just(Divss),
        Just(MovssRR),
        Just(MovssRM),
        Just(MovssMR),
        Just(Ucomiss),
    ];
    let sse_riprel = prop_oneof![Just(MovssRipRel), Just(MovsdRipRel),];
    let cmov_setcc = prop_oneof![Just(Cmovcc), Just(Setcc),];
    let cvt = prop_oneof![
        Just(Cvtsi2sd),
        Just(Cvtsd2si),
        Just(Cvtsi2ss),
        Just(Cvtss2si),
        Just(Cvtsd2ss),
        Just(Cvtss2sd),
    ];
    let bitmanip = prop_oneof![
        Just(Bsf),
        Just(Bsr),
        Just(Tzcnt),
        Just(Lzcnt),
        Just(Popcnt),
        Just(BtRI),
        Just(Bswap),
    ];
    let atomic = prop_oneof![Just(Xchg), Just(Cmpxchg),];
    let gpr_xmm = prop_oneof![
        Just(MovdToXmm),
        Just(MovdFromXmm),
        Just(MovqToXmm),
        Just(MovqFromXmm),
    ];
    let stack = prop_oneof![Just(Push), Just(Pop),];
    let pseudos = prop_oneof![Just(Phi), Just(StackAlloc), Just(Nop),];
    let hw_nop = prop_oneof![Just(NopMulti),];

    prop_oneof![
        arith, logical, shifts, moves, compare, branch, sse_scalar, sse_riprel, cmov_setcc, cvt,
        bitmanip, atomic, gpr_xmm, stack, pseudos, hw_nop,
    ]
}

// ---------------------------------------------------------------------------
// Operand strategy
// ---------------------------------------------------------------------------

fn x86_preg_strategy() -> impl Strategy<Value = X86PReg> {
    // Cover the full legitimate data-register range (0..=79) plus a
    // beyond-the-end zone (80..=255) so we also exercise "unknown" regs
    // that may slip through from a buggy regalloc and hit `hw_enc()`
    // fallback logic.
    (0u16..=255u16).prop_map(X86PReg::new)
}

/// Restricted GPR64 range (0..=15) — matches the canonical `RAX..R15`
/// encodings used by the well-shaped category strategies below.
fn gpr64_strategy() -> impl Strategy<Value = X86PReg> {
    (0u16..=15u16).prop_map(X86PReg::new)
}

/// Restricted GPR32 range (16..=31) — matches `EAX..R15D`. Used for
/// the MOVD transfer and MOVSXD categories.
fn gpr32_strategy() -> impl Strategy<Value = X86PReg> {
    (16u16..=31u16).prop_map(X86PReg::new)
}

/// Restricted GPR16 range (32..=47) — matches `AX..R15W`. Used for the
/// `MovzxW` / `MovsxW` happy-path shapes.
fn gpr16_strategy() -> impl Strategy<Value = X86PReg> {
    (32u16..=47u16).prop_map(X86PReg::new)
}

/// Restricted GPR8 range (48..=63) — matches `AL..R15B`. Used for
/// `Setcc`, `Movzx`, and `MovsxB`.
fn gpr8_strategy() -> impl Strategy<Value = X86PReg> {
    (48u16..=63u16).prop_map(X86PReg::new)
}

/// Restricted XMM range (64..=79) — matches `XMM0..XMM15`. Used for SSE
/// scalar, conversion, and GPR↔XMM transfer categories.
fn xmm_strategy() -> impl Strategy<Value = X86PReg> {
    (64u16..=79u16).prop_map(X86PReg::new)
}

/// SIB index register range — excludes encodings whose low 3 bits are 4
/// (`RSP` / `R12`), since x86 SIB uses that encoding for "no index".
fn sib_index_strategy() -> impl Strategy<Value = X86PReg> {
    prop_oneof![0u16..=3u16, 5u16..=11u16, 13u16..=15u16].prop_map(X86PReg::new)
}

fn x86_cond_code_strategy() -> impl Strategy<Value = X86CondCode> {
    use X86CondCode::*;
    prop_oneof![
        Just(O),
        Just(NO),
        Just(B),
        Just(AE),
        Just(E),
        Just(NE),
        Just(BE),
        Just(A),
        Just(S),
        Just(NS),
        Just(P),
        Just(NP),
        Just(L),
        Just(GE),
        Just(LE),
        Just(G),
    ]
}

fn operands_strategy() -> impl Strategy<Value = X86InstOperands> {
    (
        prop::option::of(x86_preg_strategy()),
        prop::option::of(x86_preg_strategy()),
        prop::option::of(x86_preg_strategy()),
        prop::option::of(x86_preg_strategy()),
        // Arbitrary scale intentionally exercises the encoder's
        // silent-fallback branch in `Sib::scaled`.
        any::<u8>(),
        any::<i64>(),
        any::<i64>(),
        prop::option::of(x86_cond_code_strategy()),
    )
        .prop_map(
            |(dst, src, base, index, scale, disp, imm, cc)| X86InstOperands {
                dst,
                src,
                base,
                index,
                scale,
                disp,
                imm,
                cc,
            },
        )
}

// ---------------------------------------------------------------------------
// Well-shaped operand strategy (happy path — per-category shape fns)
// ---------------------------------------------------------------------------
//
// Each category strategy yields `(X86Opcode, X86InstOperands)` with
// operand shapes that plausibly match the opcode. We keep each function
// flat (no shared state) so proptest's `Map` clone-bound is trivially
// met.
//
// The overall `valid_operands_strategy` composes them via
// `proptest::strategy::Union::new`.

type OpPair = (X86Opcode, X86InstOperands);

fn strat_arith_rr() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    (
        prop_oneof![
            Just(AddRR),
            Just(SubRR),
            Just(AndRR),
            Just(OrRR),
            Just(XorRR),
            Just(CmpRR),
            Just(TestRR),
            Just(MovRR),
            Just(ImulRR),
        ],
        gpr64_strategy(),
        gpr64_strategy(),
    )
        .prop_map(|(op, dst, src)| (op, X86InstOperands::rr(dst, src)))
}

fn strat_arith_ri() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    (
        prop_oneof![
            Just(AddRI),
            Just(SubRI),
            Just(AndRI),
            Just(OrRI),
            Just(XorRI),
            Just(CmpRI),
            Just(TestRI),
        ],
        gpr64_strategy(),
        any::<i32>().prop_map(i64::from),
    )
        .prop_map(|(op, dst, imm)| (op, X86InstOperands::ri(dst, imm)))
}

fn strat_cmp_ri8() -> impl Strategy<Value = OpPair> {
    (gpr64_strategy(), any::<i8>().prop_map(i64::from))
        .prop_map(|(dst, imm)| (X86Opcode::CmpRI8, X86InstOperands::ri(dst, imm)))
}

fn strat_imul_rri() -> impl Strategy<Value = OpPair> {
    (
        gpr64_strategy(),
        gpr64_strategy(),
        any::<i32>().prop_map(i64::from),
    )
        .prop_map(|(dst, src, imm)| (X86Opcode::ImulRRI, X86InstOperands::rri(dst, src, imm)))
}

fn strat_mem_rm() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    (
        prop_oneof![
            Just(AddRM),
            Just(SubRM),
            Just(CmpRM),
            Just(MovRM),
            Just(MovMR),
            Just(TestRM),
            Just(ImulRM),
        ],
        gpr64_strategy(),
        gpr64_strategy(),
        any::<i64>(),
    )
        .prop_map(|(op, reg, base, disp)| (op, X86InstOperands::rm(reg, base, disp)))
}

fn strat_mem_rm_sib() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    (
        prop_oneof![Just(MovRMSib), Just(MovMRSib), Just(LeaSib)],
        gpr64_strategy(),
        gpr64_strategy(),
        sib_index_strategy(),
        prop_oneof![Just(1u8), Just(2u8), Just(4u8), Just(8u8)],
        any::<i64>(),
    )
        .prop_map(|(op, reg, base, index, scale, disp)| {
            (op, X86InstOperands::rm_sib(reg, base, index, scale, disp))
        })
}

fn strat_rip_rel() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    prop_oneof![
        (gpr64_strategy(), any::<i64>())
            .prop_map(|(dst, disp)| { (LeaRip, X86InstOperands::rip_rel(dst, disp)) }),
        (xmm_strategy(), any::<i64>())
            .prop_map(|(dst, disp)| { (MovssRipRel, X86InstOperands::rip_rel(dst, disp)) }),
        (xmm_strategy(), any::<i64>())
            .prop_map(|(dst, disp)| { (MovsdRipRel, X86InstOperands::rip_rel(dst, disp)) }),
    ]
}

fn strat_shift_rr() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    (
        prop_oneof![Just(ShlRR), Just(ShrRR), Just(SarRR)],
        gpr64_strategy(),
    )
        .prop_map(|(op, dst)| (op, X86InstOperands::r(dst)))
}

fn strat_shift_ri() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    // Shift-immediate: arbitrary `i64` is intentional. The encoder
    // truncates to `i8`; the only property under test here is "never
    // panic" under the full signed range.
    (
        prop_oneof![Just(ShlRI), Just(ShrRI), Just(SarRI)],
        gpr64_strategy(),
        any::<i64>(),
    )
        .prop_map(|(op, dst, imm)| (op, X86InstOperands::ri(dst, imm)))
}

fn strat_mov_r() -> impl Strategy<Value = OpPair> {
    // x86-64 has no dedicated unary `MovR`; keep the sibling harness
    // category by using the degenerate `MovRR r, r` shape.
    gpr64_strategy().prop_map(|dst| (X86Opcode::MovRR, X86InstOperands::rr(dst, dst)))
}

fn strat_mov_ri() -> impl Strategy<Value = OpPair> {
    (gpr64_strategy(), any::<i64>())
        .prop_map(|(dst, imm)| (X86Opcode::MovRI, X86InstOperands::ri(dst, imm)))
}

fn strat_mov_ext() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    prop_oneof![
        (gpr64_strategy(), gpr8_strategy())
            .prop_map(|(dst, src)| { (Movzx, X86InstOperands::rr(dst, src)) }),
        (gpr64_strategy(), gpr16_strategy())
            .prop_map(|(dst, src)| { (MovzxW, X86InstOperands::rr(dst, src)) }),
        (gpr64_strategy(), gpr8_strategy())
            .prop_map(|(dst, src)| { (MovsxB, X86InstOperands::rr(dst, src)) }),
        (gpr64_strategy(), gpr16_strategy())
            .prop_map(|(dst, src)| { (MovsxW, X86InstOperands::rr(dst, src)) }),
        (gpr64_strategy(), gpr32_strategy())
            .prop_map(|(dst, src)| { (Movsx, X86InstOperands::rr(dst, src)) }),
    ]
}

fn strat_lea() -> impl Strategy<Value = OpPair> {
    (gpr64_strategy(), gpr64_strategy(), any::<i64>())
        .prop_map(|(dst, base, disp)| (X86Opcode::Lea, X86InstOperands::rm(dst, base, disp)))
}

fn strat_stack() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    (prop_oneof![Just(Push), Just(Pop)], gpr64_strategy())
        .prop_map(|(op, reg)| (op, X86InstOperands::r(reg)))
}

fn strat_nullary() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    prop_oneof![
        Just((Ret, X86InstOperands::none())),
        Just((Cdq, X86InstOperands::none())),
        Just((Cqo, X86InstOperands::none())),
    ]
}

fn strat_unary_r() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    (
        prop_oneof![Just(Neg), Just(Inc), Just(Dec), Just(Not), Just(Bswap)],
        gpr64_strategy(),
    )
        .prop_map(|(op, reg)| (op, X86InstOperands::r(reg)))
}

fn strat_div() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    (prop_oneof![Just(Div), Just(Idiv)], gpr64_strategy())
        .prop_map(|(op, reg)| (op, X86InstOperands::r(reg)))
}

fn strat_bitmanip() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    (
        prop_oneof![Just(Bsf), Just(Bsr), Just(Tzcnt), Just(Lzcnt), Just(Popcnt)],
        gpr64_strategy(),
        gpr64_strategy(),
    )
        .prop_map(|(op, dst, src)| (op, X86InstOperands::rr(dst, src)))
}

fn strat_bt_ri() -> impl Strategy<Value = OpPair> {
    (gpr64_strategy(), any::<i8>().prop_map(i64::from))
        .prop_map(|(dst, imm)| (X86Opcode::BtRI, X86InstOperands::ri(dst, imm)))
}

fn strat_jmp() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    (
        prop_oneof![Just(Jmp), Just(Call)],
        any::<i32>().prop_map(i64::from),
    )
        .prop_map(|(op, disp)| (op, X86InstOperands::rel(disp)))
}

fn strat_jcc() -> impl Strategy<Value = OpPair> {
    (x86_cond_code_strategy(), any::<i32>().prop_map(i64::from))
        .prop_map(|(cc, disp)| (X86Opcode::Jcc, X86InstOperands::jcc(cc, disp)))
}

fn strat_call_indirect() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    prop_oneof![
        gpr64_strategy().prop_map(|reg| (CallR, X86InstOperands::r(reg))),
        (gpr64_strategy(), any::<i64>()).prop_map(|(base, disp)| {
            (
                CallM,
                X86InstOperands {
                    base: Some(base),
                    disp,
                    ..X86InstOperands::none()
                },
            )
        }),
    ]
}

fn strat_cmovcc() -> impl Strategy<Value = OpPair> {
    (x86_cond_code_strategy(), gpr64_strategy(), gpr64_strategy()).prop_map(|(cc, dst, src)| {
        (
            X86Opcode::Cmovcc,
            X86InstOperands {
                dst: Some(dst),
                src: Some(src),
                cc: Some(cc),
                ..X86InstOperands::none()
            },
        )
    })
}

fn strat_setcc() -> impl Strategy<Value = OpPair> {
    (x86_cond_code_strategy(), gpr8_strategy()).prop_map(|(cc, dst)| {
        (
            X86Opcode::Setcc,
            X86InstOperands {
                dst: Some(dst),
                cc: Some(cc),
                ..X86InstOperands::none()
            },
        )
    })
}

fn strat_sse_rr() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    (
        prop_oneof![
            Just(Addsd),
            Just(Subsd),
            Just(Mulsd),
            Just(Divsd),
            Just(MovsdRR),
            Just(Ucomisd),
            Just(Addss),
            Just(Subss),
            Just(Mulss),
            Just(Divss),
            Just(MovssRR),
            Just(Ucomiss),
        ],
        xmm_strategy(),
        xmm_strategy(),
    )
        .prop_map(|(op, dst, src)| (op, X86InstOperands::rr(dst, src)))
}

fn strat_sse_mem() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    (
        prop_oneof![Just(MovsdRM), Just(MovsdMR), Just(MovssRM), Just(MovssMR)],
        xmm_strategy(),
        gpr64_strategy(),
        any::<i64>(),
    )
        .prop_map(|(op, reg, base, disp)| (op, X86InstOperands::rm(reg, base, disp)))
}

fn strat_cvt() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    prop_oneof![
        (xmm_strategy(), gpr64_strategy())
            .prop_map(|(dst, src)| { (Cvtsi2sd, X86InstOperands::rr(dst, src)) }),
        (gpr64_strategy(), xmm_strategy())
            .prop_map(|(dst, src)| { (Cvtsd2si, X86InstOperands::rr(dst, src)) }),
        (xmm_strategy(), gpr64_strategy())
            .prop_map(|(dst, src)| { (Cvtsi2ss, X86InstOperands::rr(dst, src)) }),
        (gpr64_strategy(), xmm_strategy())
            .prop_map(|(dst, src)| { (Cvtss2si, X86InstOperands::rr(dst, src)) }),
        (xmm_strategy(), xmm_strategy())
            .prop_map(|(dst, src)| { (Cvtsd2ss, X86InstOperands::rr(dst, src)) }),
        (xmm_strategy(), xmm_strategy())
            .prop_map(|(dst, src)| { (Cvtss2sd, X86InstOperands::rr(dst, src)) }),
    ]
}

fn strat_gpr_xmm() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    prop_oneof![
        (xmm_strategy(), gpr32_strategy())
            .prop_map(|(dst, src)| { (MovdToXmm, X86InstOperands::rr(dst, src)) }),
        (gpr32_strategy(), xmm_strategy())
            .prop_map(|(dst, src)| { (MovdFromXmm, X86InstOperands::rr(dst, src)) }),
        (xmm_strategy(), gpr64_strategy())
            .prop_map(|(dst, src)| { (MovqToXmm, X86InstOperands::rr(dst, src)) }),
        (gpr64_strategy(), xmm_strategy())
            .prop_map(|(dst, src)| { (MovqFromXmm, X86InstOperands::rr(dst, src)) }),
    ]
}

fn strat_atomic() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    (
        prop_oneof![Just(Xchg), Just(Cmpxchg)],
        gpr64_strategy(),
        gpr64_strategy(),
    )
        .prop_map(|(op, dst, src)| (op, X86InstOperands::rr(dst, src)))
}

fn strat_nop_multi() -> impl Strategy<Value = OpPair> {
    (1i64..=9i64).prop_map(|imm| {
        (
            X86Opcode::NopMulti,
            X86InstOperands {
                imm,
                ..X86InstOperands::none()
            },
        )
    })
}

fn strat_pseudos() -> impl Strategy<Value = OpPair> {
    use X86Opcode::*;
    // Pseudo opcodes that must cleanly return `Ok(0)` or `Err`, never
    // panic.
    prop_oneof![
        Just((Phi, X86InstOperands::none())),
        Just((StackAlloc, X86InstOperands::none())),
        Just((Nop, X86InstOperands::none())),
    ]
}

/// "Well-shaped" generator: produces an `(X86Opcode, X86InstOperands)`
/// pair with a field population that at least plausibly matches the
/// opcode. Exercises the happy path so any panic here is a definite P1
/// bug (the corresponding malformed-input path below catches latent
/// panics in error-handling code).
fn valid_operands_strategy() -> impl Strategy<Value = OpPair> {
    // Proptest requires uniform branch types for `Union::new`. Boxing
    // each per-category strategy to `BoxedStrategy<OpPair>` unifies them.
    let cats: Vec<BoxedStrategy<OpPair>> = vec![
        strat_arith_rr().boxed(),
        strat_arith_ri().boxed(),
        strat_cmp_ri8().boxed(),
        strat_imul_rri().boxed(),
        strat_mem_rm().boxed(),
        strat_mem_rm_sib().boxed(),
        strat_rip_rel().boxed(),
        strat_shift_rr().boxed(),
        strat_shift_ri().boxed(),
        strat_mov_r().boxed(),
        strat_mov_ri().boxed(),
        strat_mov_ext().boxed(),
        strat_lea().boxed(),
        strat_stack().boxed(),
        strat_nullary().boxed(),
        strat_unary_r().boxed(),
        strat_div().boxed(),
        strat_bitmanip().boxed(),
        strat_bt_ri().boxed(),
        strat_jmp().boxed(),
        strat_jcc().boxed(),
        strat_call_indirect().boxed(),
        strat_cmovcc().boxed(),
        strat_setcc().boxed(),
        strat_sse_rr().boxed(),
        strat_sse_mem().boxed(),
        strat_cvt().boxed(),
        strat_gpr_xmm().boxed(),
        strat_atomic().boxed(),
        strat_nop_multi().boxed(),
        strat_pseudos().boxed(),
    ];
    proptest::strategy::Union::new(cats)
}

// ---------------------------------------------------------------------------
// Property
// ---------------------------------------------------------------------------

/// Run `encode_instruction` inside `catch_unwind` and assert no panic.
///
/// The returned `Result<usize, X86EncodeError>` is discarded; the *only*
/// failure mode tracked here is "panic reached the caller".
fn assert_no_panic(opcode: X86Opcode, ops: &X86InstOperands) {
    // Clone inputs into the closure so `catch_unwind`'s UnwindSafe bound
    // is trivially satisfied for these POD-ish types.
    let opcode_copy = opcode.clone();
    let ops_copy = ops.clone();
    let result = panic::catch_unwind(panic::AssertUnwindSafe(move || {
        let mut enc = X86Encoder::new();
        let _ = enc.encode_instruction(opcode_copy, &ops_copy);
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
            (opcode, ops),
            msg
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
    fn encode_never_panics_on_valid((opcode, ops) in valid_operands_strategy()) {
        assert_no_panic(opcode, &ops);
    }

    /// Random-malformed inputs — arbitrary opcode paired with arbitrary
    /// operand shapes. This is the main totality property: the encoder
    /// should convert *any* malformed shape into `Err`, never a panic.
    #[test]
    fn encode_never_panics_on_malformed(
        opcode in opcode_strategy(),
        ops in operands_strategy(),
    ) {
        assert_no_panic(opcode, &ops);
    }
}

// ---------------------------------------------------------------------------
// Static coverage sanity check (#450 acceptance criterion)
// ---------------------------------------------------------------------------
//
// Guardrail test: if a future refactor narrows `opcode_strategy()` below
// 80% of the 101 `X86Opcode` variants, this test fails loudly.
// Implemented by exhaustively sampling the strategy until a fixed cap
// and counting distinct opcodes seen.
#[test]
fn opcode_strategy_covers_at_least_80_percent() {
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;
    use std::collections::HashSet;

    let mut runner = TestRunner::deterministic();
    let strat = opcode_strategy();
    let mut seen: HashSet<X86Opcode> = HashSet::new();
    // 10k draws is plenty to cover 101 uniformly-weighted variants under
    // the current sub-strategy composition.
    for _ in 0..10_000 {
        let tree = strat.new_tree(&mut runner).expect("strategy tree");
        seen.insert(tree.current());
        if seen.len() >= 101 {
            break;
        }
    }
    // 80 / 101 = 79.21%; require ≥ 80 for the 80% floor (#450).
    assert!(
        seen.len() >= 80,
        "opcode_strategy only samples {} distinct variants; need ≥ 80 (80% of 101)",
        seen.len()
    );
}

// ---------------------------------------------------------------------------
// Regression reproducers for known panics found by this harness
// ---------------------------------------------------------------------------
//
// These pin-down tests are hand-reduced shrinks of failing proptest cases.
// Each asserts the post-fix behavior (a typed `Err(X86EncodeError::..)`)
// rather than the original crash.

/// `NopMulti` with a large positive `imm` previously recursed via
/// `encode_multibyte_nop(size - 9)` without a bound, overflowing the stack
/// on the x86-64 encoder and aborting the process with SIGABRT. Surfaced
/// by the malformed-input proptest harness on 2026-04-20 under #473.
///
/// Fixed two ways: (1) `encode_multibyte_nop` converted to iteration so
/// stack depth is O(1) regardless of `size`; (2) dispatch site in
/// `encode_instruction` clamps `ops.imm` to `[1, 15]` and returns
/// `Err(X86EncodeError::InvalidOperands)` for anything outside that range.
#[test]
fn regression_nopmulti_large_imm_stack_overflow() {
    let mut enc = X86Encoder::new();
    let ops = X86InstOperands {
        imm: i64::MAX,
        ..X86InstOperands::none()
    };
    let err = enc
        .encode_instruction(X86Opcode::NopMulti, &ops)
        .expect_err("i64::MAX imm must be rejected, not recursed on");
    assert!(
        matches!(err, X86EncodeError::InvalidOperands(_)),
        "expected InvalidOperands on oversized NopMulti imm, got {err:?}"
    );
}

/// Negative `NopMulti` `imm` previously collapsed to the default size (3)
/// via `if ops.imm > 0 { ops.imm as usize } else { 3 }`. That branch is
/// still exercised deliberately as the happy-path fallback — this test
/// just pins the documented contract: a negative imm must NOT be treated
/// as a giant `usize` via wrap-around, and must emit a valid 3-byte NOP.
#[test]
fn regression_nopmulti_negative_imm_falls_back_to_default() {
    let mut enc = X86Encoder::new();
    let ops = X86InstOperands {
        imm: -1,
        ..X86InstOperands::none()
    };
    let n = enc
        .encode_instruction(X86Opcode::NopMulti, &ops)
        .expect("negative imm must fall back to default 3-byte NOP");
    assert_eq!(n, 3, "default NopMulti size must be 3 bytes");
    assert_eq!(enc.bytes, vec![0x0F, 0x1F, 0x00]);
}

/// `NopMulti` with size in the accepted alignment-padding range (1..=15)
/// must emit bit-identical output to the previous recursive implementation.
/// Sizes 10..=15 exercise the post-refactor iteration path.
#[test]
fn regression_nopmulti_alignment_range_emits_expected_bytes() {
    // Size 10: one 9-byte NOP followed by one 1-byte NOP.
    let mut enc10 = X86Encoder::new();
    let ops10 = X86InstOperands {
        imm: 10,
        ..X86InstOperands::none()
    };
    enc10
        .encode_instruction(X86Opcode::NopMulti, &ops10)
        .expect("size=10 must encode");
    assert_eq!(enc10.bytes.len(), 10, "size=10 must emit 10 bytes");
    assert_eq!(
        &enc10.bytes[0..9],
        &[0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
        "first 9 bytes must be the canonical 9-byte NOP"
    );
    assert_eq!(enc10.bytes[9], 0x90, "trailing byte must be 1-byte NOP");

    // Size 15: one 9-byte NOP followed by one 6-byte NOP.
    let mut enc15 = X86Encoder::new();
    let ops15 = X86InstOperands {
        imm: 15,
        ..X86InstOperands::none()
    };
    enc15
        .encode_instruction(X86Opcode::NopMulti, &ops15)
        .expect("size=15 must encode");
    assert_eq!(enc15.bytes.len(), 15, "size=15 must emit 15 bytes");
    assert_eq!(
        &enc15.bytes[0..9],
        &[0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
        "first 9 bytes must be the canonical 9-byte NOP"
    );
    assert_eq!(
        &enc15.bytes[9..15],
        &[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00],
        "trailing 6 bytes must be the canonical 6-byte NOP"
    );

    // Size 16 (just out of range): rejected with typed error.
    let mut enc16 = X86Encoder::new();
    let ops16 = X86InstOperands {
        imm: 16,
        ..X86InstOperands::none()
    };
    let err = enc16
        .encode_instruction(X86Opcode::NopMulti, &ops16)
        .expect_err("size=16 must be rejected");
    assert!(
        matches!(err, X86EncodeError::InvalidOperands(_)),
        "size=16 must yield InvalidOperands, got {err:?}"
    );
}
