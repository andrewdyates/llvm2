// llvm2-codegen/tests/jit_block_counters.rs — Per-basic-block JIT counters.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Covers the public API landed for issue #364, Phase 2 `BlockCounts`:
// - `ProfileHookMode::BlockCounts` enables one `AtomicU64` per basic block.
// - `ExecutableBuffer::block_count(name, block_id)` reads a specific block's
//   counter.
// - `ExecutableBuffer::block_counts(name)` enumerates every block counter
//   for a function.
// - `ExecutableBuffer::get_profile(name)` / `entry_count(name)` continue to
//   return the FUNCTION-ENTRY count (= entry block's count) so the stable
//   #478 API is preserved.
//
// The canonical diamond fixture is an if/else on `X0 == 0`:
//
//     entry (block 0): CMP X0,#0 ; B.EQ then
//     else  (block 1): MOVZ X0,#77 ; B join
//     then  (block 2): MOVZ X0,#11 ; (falls through)
//     join  (block 3): RET
//
// Block layout order is [entry, else, then, join]. The entry block has a
// BCond to `then` (the third block in layout) and falls through to `else`.
// The `else` block branches unconditionally to `join`. The `then` block
// falls through to `join`. Calling with argument `0` executes blocks
// {entry, then, join}; calling with non-zero executes {entry, else, join}.
//
// Running the function N=100 times with alternating arguments yields
// exactly these per-block counts:
// - entry: 100
// - else:  50  (odd calls, non-zero argument)
// - then:  50  (even calls, argument 0)
// - join:  100
//
// Part of #364

#[cfg(target_arch = "aarch64")]
use llvm2_codegen::jit::{JitCompiler, JitConfig, ProfileHookMode};
#[cfg(target_arch = "aarch64")]
use llvm2_codegen::pipeline::resolve_branches;
#[cfg(target_arch = "aarch64")]
use llvm2_ir::function::{JumpTableData, MachFunction, Signature, Type};
#[cfg(target_arch = "aarch64")]
use llvm2_ir::inst::{AArch64Opcode, MachInst};
#[cfg(target_arch = "aarch64")]
use llvm2_ir::operand::MachOperand;
#[cfg(target_arch = "aarch64")]
use llvm2_ir::regs::{X0, X1, X2, X3};
#[cfg(target_arch = "aarch64")]
use llvm2_ir::types::BlockId;
#[cfg(target_arch = "aarch64")]
use std::collections::HashMap;

/// Build the if/else diamond described in the module comment.
///
/// Returns a [`MachFunction`] with `resolve_branches` already applied so the
/// per-block PC-relative immediates are baked in and `compile_raw` can
/// encode the function directly.
#[cfg(target_arch = "aarch64")]
fn build_diamond() -> MachFunction {
    let sig = Signature::new(vec![Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("diamond".to_string(), sig);

    let entry = func.entry;
    let else_b = func.create_block();
    let then_b = func.create_block();
    let join_b = func.create_block();

    // entry: CMP X0, #0 ; B.EQ then
    let cmp = MachInst::new(
        AArch64Opcode::CmpRI,
        vec![MachOperand::PReg(X0), MachOperand::Imm(0)],
    );
    let cmp_id = func.push_inst(cmp);
    func.append_inst(entry, cmp_id);

    let beq = MachInst::new(
        AArch64Opcode::BCond,
        vec![
            MachOperand::Imm(0x0), // EQ
            MachOperand::Block(then_b),
        ],
    );
    let beq_id = func.push_inst(beq);
    func.append_inst(entry, beq_id);

    // else: MOVZ X0, #77 ; B join
    let mov_77 = MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::PReg(X0), MachOperand::Imm(77)],
    );
    let mov_77_id = func.push_inst(mov_77);
    func.append_inst(else_b, mov_77_id);

    let b_join = MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(join_b)]);
    let b_join_id = func.push_inst(b_join);
    func.append_inst(else_b, b_join_id);

    // then: MOVZ X0, #11 (falls through to join)
    let mov_11 = MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::PReg(X0), MachOperand::Imm(11)],
    );
    let mov_11_id = func.push_inst(mov_11);
    func.append_inst(then_b, mov_11_id);

    // join: RET
    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(join_b, ret_id);

    // Resolve MachOperand::Block(...) to PC-relative MachOperand::Imm.
    resolve_branches(&mut func);
    func
}

#[cfg(target_arch = "aarch64")]
#[test]
fn block_counts_diamond_alternating_inputs() {
    let jit = JitCompiler::new(JitConfig {
        profile_hooks: ProfileHookMode::BlockCounts,
        ..JitConfig::default()
    });
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_diamond()], &ext)
        .expect("compile_raw must succeed with BlockCounts on AArch64");

    let diamond: extern "C" fn(u64) -> u64 = unsafe {
        buf.get_fn_bound("diamond")
            .expect("typed function pointer must exist")
            .into_inner()
    };

    // Run 100 alternating calls: odd => non-zero argument => `else` path
    // returns 77. Even => argument 0 => `then` path returns 11.
    const N: u64 = 100;
    let mut even_calls = 0u64;
    let mut odd_calls = 0u64;
    for i in 0..N {
        if i % 2 == 0 {
            even_calls += 1;
            assert_eq!(diamond(0), 11, "even call {} (arg=0) must hit `then`", i);
        } else {
            odd_calls += 1;
            assert_eq!(
                diamond(i),
                77,
                "odd call {} (arg={}) must hit `else`",
                i,
                i
            );
        }
    }
    assert_eq!(even_calls + odd_calls, N);
    assert_eq!(even_calls, 50);
    assert_eq!(odd_calls, 50);

    // Expected per-block counts:
    //   block 0 (entry) = N
    //   block 1 (else)  = odd_calls
    //   block 2 (then)  = even_calls
    //   block 3 (join)  = N
    assert_eq!(
        buf.block_count("diamond", BlockId(0)),
        Some(N),
        "entry block must be entered once per call"
    );
    assert_eq!(
        buf.block_count("diamond", BlockId(1)),
        Some(odd_calls),
        "else block must be entered only on non-zero-argument calls"
    );
    assert_eq!(
        buf.block_count("diamond", BlockId(2)),
        Some(even_calls),
        "then block must be entered only on zero-argument calls"
    );
    assert_eq!(
        buf.block_count("diamond", BlockId(3)),
        Some(N),
        "join block must be entered once per call (every path reaches it)"
    );

    // The block-counts iterator returns all four block counters.
    let mut all: Vec<(u32, u64)> = buf.block_counts("diamond");
    all.sort_by_key(|&(bid, _)| bid);
    assert_eq!(
        all,
        vec![(0, N), (1, odd_calls), (2, even_calls), (3, N)]
    );

    // And the stable function-entry alias surface (#478) must still work:
    // `get_profile` / `entry_count` return the entry block's counter.
    assert_eq!(buf.entry_count("diamond"), Some(N));
    assert_eq!(
        buf.get_profile("diamond").expect("entry profile").call_count,
        N
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
fn block_counts_unknown_block_returns_none() {
    let jit = JitCompiler::new(JitConfig {
        profile_hooks: ProfileHookMode::BlockCounts,
        ..JitConfig::default()
    });
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_diamond()], &ext)
        .expect("compile_raw must succeed");

    // Block id 99 does not exist in `diamond` — accessor must return None
    // rather than fabricate a zero.
    assert_eq!(buf.block_count("diamond", BlockId(99)), None);
    // And unknown function names must likewise return None.
    assert_eq!(buf.block_count("not_a_function", BlockId(0)), None);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn block_counts_disabled_mode_yields_no_block_counters() {
    // With the default (CallCounts-less) profile mode, `block_count` must
    // return `None` for every block — the BlockCounts splice path was not
    // taken, so no per-block counters were allocated.
    let jit = JitCompiler::new(JitConfig::default());
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_diamond()], &ext)
        .expect("compile_raw must succeed with default JitConfig");

    for bid in [BlockId(0), BlockId(1), BlockId(2), BlockId(3)] {
        assert_eq!(
            buf.block_count("diamond", bid),
            None,
            "BlockCounts disabled => no per-block counters allocated"
        );
    }
    assert!(buf.block_counts("diamond").is_empty());
}

#[cfg(not(target_arch = "aarch64"))]
#[test]
fn block_counts_aarch64_only_placeholder() {
    // Intentionally empty: the splice path is AArch64-only in the initial
    // landing (issue #364). A follow-up will cover x86-64. Keeping this
    // file compiling on non-aarch64 targets makes the `tests/` directory
    // cross-architecture clean.
}

// ---------------------------------------------------------------------------
// Issue #490 — jump-table-dispatched switch under `BlockCounts`.
//
// `splice_block_trampolines_aarch64` used to silently miscompile functions
// whose lowering emits a jump table (`Adr` + trailing i32 entries) because
// neither the `Adr` imm21 nor the table entries were re-patched against
// the post-splice block layout. This test builds a hand-rolled
// switch-lowered function with THREE case targets plus a default block,
// compiles it with `ProfileHookMode::BlockCounts`, invokes it with a mix
// of in-range and out-of-range selectors, and asserts that:
//   1. Each call returns the correct per-case constant (i.e. the jump
//      table + ADR were re-patched correctly so the indirect branch
//      reaches the right block's trampoline).
//   2. The per-block counter for the case that was actually entered
//      increments the expected number of times.
//
// Layout of the function `switch3`:
//
//   entry (bb0):
//     CMP X0, #2
//     B.HI default           ; selector > 2 -> default
//     ADR X1, <jumptable>
//     LDRSW X2, [X1, X0, LSL #2]
//     ADD X3, X1, X2
//     BR X3
//
//   case_0 (bb1): MOVZ X0, #10 ; B end
//   case_1 (bb2): MOVZ X0, #20 ; B end
//   case_2 (bb3): MOVZ X0, #30 ; B end
//   default (bb4): MOVZ X0, #99 ; B end
//   end   (bb5): RET
//
// The jump table has three entries `[case_0, case_1, case_2]`. Block
// layout order is `[entry, case_0, case_1, case_2, default, end]`.
// Selector values 0/1/2 go through the table; anything else falls to
// default via the B.HI bounds check.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn build_switch3() -> MachFunction {
    let sig = Signature::new(vec![Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("switch3".to_string(), sig);

    let entry = func.entry;
    let case_0 = func.create_block();
    let case_1 = func.create_block();
    let case_2 = func.create_block();
    let default_b = func.create_block();
    let end_b = func.create_block();

    // Register the jump table BEFORE emitting the ADR. Index 0 selects
    // case_0, 1 selects case_1, 2 selects case_2.
    let jt_idx = func.jump_tables.len() as u32;
    func.jump_tables.push(JumpTableData {
        min_val: 0,
        targets: vec![case_0, case_1, case_2],
    });

    // entry: CMP X0, #2 ; B.HI default ; ADR X1, jt ; LDRSW X2, [X1,X0,LSL#2] ;
    //        ADD X3, X1, X2 ; BR X3
    //
    // CMP against `num_cases - 1` (= 2) rather than `num_cases` (= 3): B.HI
    // means "unsigned higher", so selectors > 2 go to default, selectors in
    // {0, 1, 2} fall through and index the jump table. Using CMP #3 would
    // let selector=3 fall through and index out-of-bounds entry 3, which
    // produces the classic out-of-range jump-table miscompile (SIGILL).
    let cmp = MachInst::new(
        AArch64Opcode::CmpRI,
        vec![MachOperand::PReg(X0), MachOperand::Imm(2)],
    );
    let cmp_id = func.push_inst(cmp);
    func.append_inst(entry, cmp_id);

    // B.HI default — condition code HI (unsigned higher) = 0x8.
    let bhi = MachInst::new(
        AArch64Opcode::BCond,
        vec![
            MachOperand::Imm(0x8), // HI
            MachOperand::Block(default_b),
        ],
    );
    let bhi_id = func.push_inst(bhi);
    func.append_inst(entry, bhi_id);

    // ADR X1, jumptable_idx. The pipeline patches imm21 once block
    // layout is known.
    let adr = MachInst::new(
        AArch64Opcode::Adr,
        vec![MachOperand::PReg(X1), MachOperand::JumpTableIndex(jt_idx)],
    );
    let adr_id = func.push_inst(adr);
    func.append_inst(entry, adr_id);

    // LDRSW X2, [X1, X0, LSL #2].
    let ldrsw = MachInst::new(
        AArch64Opcode::LdrswRO,
        vec![
            MachOperand::PReg(X2),
            MachOperand::PReg(X1),
            MachOperand::PReg(X0),
        ],
    );
    let ldrsw_id = func.push_inst(ldrsw);
    func.append_inst(entry, ldrsw_id);

    // ADD X3, X1, X2.
    let add = MachInst::new(
        AArch64Opcode::AddRR,
        vec![
            MachOperand::PReg(X3),
            MachOperand::PReg(X1),
            MachOperand::PReg(X2),
        ],
    );
    let add_id = func.push_inst(add);
    func.append_inst(entry, add_id);

    // BR X3.
    let br = MachInst::new(AArch64Opcode::Br, vec![MachOperand::PReg(X3)]);
    let br_id = func.push_inst(br);
    func.append_inst(entry, br_id);

    // case_0: MOVZ X0, #10 ; B end
    let m10 = MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::PReg(X0), MachOperand::Imm(10)],
    );
    let m10_id = func.push_inst(m10);
    func.append_inst(case_0, m10_id);
    let b0 = MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(end_b)]);
    let b0_id = func.push_inst(b0);
    func.append_inst(case_0, b0_id);

    // case_1: MOVZ X0, #20 ; B end
    let m20 = MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::PReg(X0), MachOperand::Imm(20)],
    );
    let m20_id = func.push_inst(m20);
    func.append_inst(case_1, m20_id);
    let b1 = MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(end_b)]);
    let b1_id = func.push_inst(b1);
    func.append_inst(case_1, b1_id);

    // case_2: MOVZ X0, #30 ; B end
    let m30 = MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::PReg(X0), MachOperand::Imm(30)],
    );
    let m30_id = func.push_inst(m30);
    func.append_inst(case_2, m30_id);
    let b2 = MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(end_b)]);
    let b2_id = func.push_inst(b2);
    func.append_inst(case_2, b2_id);

    // default: MOVZ X0, #99 ; B end
    let m99 = MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::PReg(X0), MachOperand::Imm(99)],
    );
    let m99_id = func.push_inst(m99);
    func.append_inst(default_b, m99_id);
    let bd = MachInst::new(AArch64Opcode::B, vec![MachOperand::Block(end_b)]);
    let bd_id = func.push_inst(bd);
    func.append_inst(default_b, bd_id);

    // end: RET
    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(end_b, ret_id);

    // Resolve B/BCond Block operands to PC-relative immediates.
    resolve_branches(&mut func);
    func
}

#[cfg(target_arch = "aarch64")]
#[test]
fn switch3_without_profile_hooks_is_correct() {
    // Sanity check: without any profile hooks the switch must already
    // work. If this test fails, the bug is in the base switch lowering
    // encoding (not the splice), and the BlockCounts test below can't
    // tell us anything about #490.
    let jit = JitCompiler::new(JitConfig::default());
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_switch3()], &ext)
        .expect("compile_raw must succeed for switch3 without profile hooks");

    let switch3: extern "C" fn(u64) -> u64 = unsafe {
        buf.get_fn_bound("switch3")
            .expect("typed function pointer must exist")
            .into_inner()
    };

    assert_eq!(switch3(0), 10);
    assert_eq!(switch3(1), 20);
    assert_eq!(switch3(2), 30);
    assert_eq!(switch3(3), 99);
    assert_eq!(switch3(100), 99);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn block_counts_jump_table_switch_three_cases_and_default() {
    let jit = JitCompiler::new(JitConfig {
        profile_hooks: ProfileHookMode::BlockCounts,
        ..JitConfig::default()
    });
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_switch3()], &ext)
        .expect("compile_raw must succeed for switch3 with BlockCounts");

    let switch3: extern "C" fn(u64) -> u64 = unsafe {
        buf.get_fn_bound("switch3")
            .expect("typed function pointer must exist")
            .into_inner()
    };

    // Correctness: each case returns its own constant. If the ADR imm21
    // or the jump-table entries are stale post-splice, the indirect
    // branch either lands in the middle of a trampoline (producing
    // nonsense) or crashes outright. Asserting exact return values
    // catches either failure mode.
    assert_eq!(switch3(0), 10, "case_0 must return 10");
    assert_eq!(switch3(1), 20, "case_1 must return 20");
    assert_eq!(switch3(2), 30, "case_2 must return 30");
    assert_eq!(switch3(3), 99, "selector=3 must fall to default");
    assert_eq!(switch3(100), 99, "far-out selector must fall to default");

    // Drive a deterministic mix and check per-block counters.
    //   selector=0 -> case_0 (bb1)
    //   selector=1 -> case_1 (bb2)
    //   selector=2 -> case_2 (bb3)
    //   selector=4 -> default (bb4)
    // Run selector=1 exactly 7 times so we can assert the block counter
    // exactly matches, not merely "non-zero".
    for _ in 0..7 {
        assert_eq!(switch3(1), 20);
    }
    for _ in 0..3 {
        assert_eq!(switch3(0), 10);
    }
    for _ in 0..2 {
        assert_eq!(switch3(4), 99);
    }

    // Total per-case (includes the 1 invocation from the correctness
    // block above for cases 0, 1, 2 and 2 default invocations from
    // selectors 3 and 100):
    //   case_0 (bb1): 1 + 3 = 4
    //   case_1 (bb2): 1 + 7 = 8
    //   case_2 (bb3): 1
    //   default (bb4): 2 + 2 = 4
    //   entry (bb0): 1 + 1 + 1 + 1 + 1 + 7 + 3 + 2 = 17
    //   end   (bb5): 17 (every path reaches end)
    assert_eq!(buf.block_count("switch3", BlockId(0)), Some(17), "entry");
    assert_eq!(buf.block_count("switch3", BlockId(1)), Some(4), "case_0");
    assert_eq!(buf.block_count("switch3", BlockId(2)), Some(8), "case_1");
    assert_eq!(buf.block_count("switch3", BlockId(3)), Some(1), "case_2");
    assert_eq!(buf.block_count("switch3", BlockId(4)), Some(4), "default");
    assert_eq!(buf.block_count("switch3", BlockId(5)), Some(17), "end");

    // Function-entry alias (`#478`) keeps working.
    assert_eq!(buf.entry_count("switch3"), Some(17));
}
