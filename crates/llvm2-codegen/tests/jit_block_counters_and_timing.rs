// llvm2-codegen/tests/jit_block_counters_and_timing.rs
// Per-basic-block JIT counters + cycle timing (AArch64, issue #364 Phase 3).
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Covers the public API landed for issue #364, Phase 3
// `BlockCountsAndTiming`:
// - `ProfileHookMode::BlockCountsAndTiming` enables one
//   `{count, total_cycles}` cell per basic block plus a single buffer-wide
//   `TimingState`.
// - `ExecutableBuffer::block_timing(name, BlockId)` returns the
//   `(count, total_cycles)` tuple for a specific block.
// - `ExecutableBuffer::block_timings(name)` enumerates every
//   `(block_id, count, total_cycles)` tuple for a function.
// - `ExecutableBuffer::block_count(name, BlockId)` / `block_counts(name)`
//   keep working in parallel with the timing-aware storage — the
//   read-side API does not care which mode compiled the function.
// - `ExecutableBuffer::get_profile(name)` / `entry_count(name)` continue to
//   return the entry block's counter (so the stable #478 API is
//   preserved even under timing mode).
//
// Fixture (same diamond as `jit_block_counters.rs`): if/else on X0 == 0,
// layout order [entry, else, then, join]. Calling with alternating
// arguments (non-zero / zero) produces these per-block counts for N=100
// iterations: entry=100, else=50, then=50, join=100.
//
// For cycle timing, the attribution model (see `TimingState` docs) is
// "first block entered under buffer contributes 0 cycles; each subsequent
// block attributes the delta from its entry back to the previous block's
// total_cycles". On a diamond run N times that means the `entry` block,
// `else` block, and `then` block each accumulate nonzero cycles (each has
// a next-block attribution chain). We assert >0 rather than pinning a
// specific value because CNTVCT_EL0 rates vary and scheduling jitter
// dominates small deltas.
//
// Part of #364

#[cfg(target_arch = "aarch64")]
use llvm2_codegen::jit::{JitCompiler, JitConfig, ProfileHookMode};
#[cfg(target_arch = "aarch64")]
use llvm2_codegen::pipeline::resolve_branches;
#[cfg(target_arch = "aarch64")]
use llvm2_ir::function::{MachFunction, Signature, Type};
#[cfg(target_arch = "aarch64")]
use llvm2_ir::inst::{AArch64Opcode, MachInst};
#[cfg(target_arch = "aarch64")]
use llvm2_ir::operand::MachOperand;
#[cfg(target_arch = "aarch64")]
use llvm2_ir::regs::X0;
#[cfg(target_arch = "aarch64")]
use llvm2_ir::types::BlockId;
#[cfg(target_arch = "aarch64")]
use std::collections::HashMap;

/// Build the canonical if/else diamond — same fixture shape as
/// `jit_block_counters.rs` but lifted here so the two suites can evolve
/// independently.
#[cfg(target_arch = "aarch64")]
fn build_diamond() -> MachFunction {
    let sig = Signature::new(vec![Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("diamond_t".to_string(), sig);

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

    resolve_branches(&mut func);
    func
}

#[cfg(target_arch = "aarch64")]
#[test]
fn block_timing_diamond_alternating_inputs() {
    let jit = JitCompiler::new(JitConfig {
        profile_hooks: ProfileHookMode::BlockCountsAndTiming,
        ..JitConfig::default()
    });
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_diamond()], &ext)
        .expect("compile_raw must succeed with BlockCountsAndTiming on AArch64");

    let diamond: extern "C" fn(u64) -> u64 = unsafe {
        buf.get_fn_bound("diamond_t")
            .expect("typed function pointer must exist")
            .into_inner()
    };

    // Alternate inputs so both arms of the diamond fire.
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

    // --- Count assertions: identical to the BlockCounts suite.
    //   block 0 (entry) = N
    //   block 1 (else)  = odd_calls
    //   block 2 (then)  = even_calls
    //   block 3 (join)  = N
    assert_eq!(
        buf.block_count("diamond_t", BlockId(0)),
        Some(N),
        "entry block must be entered once per call"
    );
    assert_eq!(
        buf.block_count("diamond_t", BlockId(1)),
        Some(odd_calls),
        "else block must be entered only on non-zero-argument calls"
    );
    assert_eq!(
        buf.block_count("diamond_t", BlockId(2)),
        Some(even_calls),
        "then block must be entered only on zero-argument calls"
    );
    assert_eq!(
        buf.block_count("diamond_t", BlockId(3)),
        Some(N),
        "join block must be entered once per call (every path reaches it)"
    );

    // `block_counts` walks the timing-cells map in Phase 3.
    let mut all_counts: Vec<(u32, u64)> = buf.block_counts("diamond_t");
    all_counts.sort_by_key(|&(bid, _)| bid);
    assert_eq!(
        all_counts,
        vec![(0, N), (1, odd_calls), (2, even_calls), (3, N)]
    );

    // Stable #478 alias surface must still report the entry count.
    assert_eq!(buf.entry_count("diamond_t"), Some(N));
    assert_eq!(
        buf.get_profile("diamond_t").expect("entry profile").call_count,
        N
    );

    // --- Timing assertions.
    //
    // The attribution chain for the diamond (alternating calls):
    //   entry -> {then,else} -> join -> entry -> ... -> entry -> {then,else} -> join
    // First block entered under the buffer contributes 0 cycles (CBZ
    // skips attribution when prev_ts=0). Every subsequent block entry
    // attributes `now - prev_ts` back to the PREVIOUSLY-ENTERED cell.
    //
    // For each of the four blocks, total_cycles must therefore be
    // nonzero after N=100 alternating calls: every block is both
    // entered-after-another-block and exited-before-another-block
    // multiple times.
    //
    // We deliberately assert `> 0` rather than pinning a specific cycle
    // count: `CNTVCT_EL0` frequency differs across cores (e.g. Apple
    // Silicon exposes 24 MHz virtual counter) and scheduling jitter
    // dominates small deltas. The structural "each cell accumulates
    // something" invariant is what matters for correctness.
    let entry_tim = buf
        .block_timing("diamond_t", BlockId(0))
        .expect("entry block timing must be present in Phase 3");
    assert_eq!(entry_tim.0, N, "timing cell count must match block_count");
    assert!(
        entry_tim.1 > 0,
        "entry block must accumulate cycles across N={} alternating calls (got {})",
        N,
        entry_tim.1
    );

    let else_tim = buf
        .block_timing("diamond_t", BlockId(1))
        .expect("else block timing must be present");
    assert_eq!(else_tim.0, odd_calls);
    assert!(
        else_tim.1 > 0,
        "else block must accumulate cycles (got {})",
        else_tim.1
    );

    let then_tim = buf
        .block_timing("diamond_t", BlockId(2))
        .expect("then block timing must be present");
    assert_eq!(then_tim.0, even_calls);
    assert!(
        then_tim.1 > 0,
        "then block must accumulate cycles (got {})",
        then_tim.1
    );

    // `join` is always followed by the next call's `entry`, so on every
    // call but the very first it gets attributed cycles. After N=100
    // calls that's 99 attributions => must be >0.
    let join_tim = buf
        .block_timing("diamond_t", BlockId(3))
        .expect("join block timing must be present");
    assert_eq!(join_tim.0, N);
    assert!(
        join_tim.1 > 0,
        "join block must accumulate cycles across inter-call attributions (got {})",
        join_tim.1
    );

    // block_timings iterator yields all four cells.
    let mut all_tim: Vec<(u32, u64, u64)> = buf.block_timings("diamond_t");
    all_tim.sort_by_key(|&(bid, _, _)| bid);
    assert_eq!(all_tim.len(), 4);
    assert_eq!(all_tim[0].0, 0);
    assert_eq!(all_tim[0].1, N);
    assert_eq!(all_tim[1].0, 1);
    assert_eq!(all_tim[1].1, odd_calls);
    assert_eq!(all_tim[2].0, 2);
    assert_eq!(all_tim[2].1, even_calls);
    assert_eq!(all_tim[3].0, 3);
    assert_eq!(all_tim[3].1, N);
    for (bid, _, cyc) in &all_tim {
        assert!(
            *cyc > 0,
            "block {} must have >0 cycles after {} calls",
            bid,
            N
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn block_timing_unknown_block_returns_none() {
    let jit = JitCompiler::new(JitConfig {
        profile_hooks: ProfileHookMode::BlockCountsAndTiming,
        ..JitConfig::default()
    });
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_diamond()], &ext)
        .expect("compile_raw must succeed");

    // Unknown block id within a known function must return None.
    assert_eq!(buf.block_timing("diamond_t", BlockId(99)), None);
    // Unknown function name must likewise return None.
    assert_eq!(buf.block_timing("not_a_function", BlockId(0)), None);
    assert!(buf.block_timings("not_a_function").is_empty());
}

#[cfg(target_arch = "aarch64")]
#[test]
fn block_timing_disabled_mode_yields_no_timing_cells() {
    // With a non-timing profile mode, `block_timing` must return `None`
    // for every block.
    let jit = JitCompiler::new(JitConfig {
        profile_hooks: ProfileHookMode::BlockCounts,
        ..JitConfig::default()
    });
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_diamond()], &ext)
        .expect("compile_raw must succeed with BlockCounts");

    for bid in [BlockId(0), BlockId(1), BlockId(2), BlockId(3)] {
        assert_eq!(
            buf.block_timing("diamond_t", bid),
            None,
            "BlockCounts mode must NOT allocate timing cells"
        );
    }
    assert!(buf.block_timings("diamond_t").is_empty());
}

#[cfg(not(target_arch = "aarch64"))]
#[test]
fn block_timing_aarch64_only_placeholder() {
    // Intentionally empty: the timing splice path is AArch64-only in
    // the Phase 3 landing (issue #364). A follow-up will cover x86-64
    // via RDTSC. Keeping this file compiling on non-aarch64 targets
    // makes the `tests/` directory cross-architecture clean.
}
