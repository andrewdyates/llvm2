// llvm2-codegen/tests/jit_block_counter_lifetime.rs
// BlockCounts counter lifetime invariant (issue #494).
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Covers the lifetime contract documented in `jit.rs` under the
// "Profile counter & timing-cell lifetime" module section for issue #494:
//
// - Each per-block `AtomicU64` is `Box`-pinned and owned by the
//   `ExecutableBuffer`. The baked-in raw pointer stays valid for the
//   entire lifetime of the executable mapping.
// - Dropping the `ExecutableBuffer` unmaps `memory` BEFORE the counter
//   boxes are freed (field declaration order in `ExecutableBuffer`),
//   so the trampolines cannot execute against a dangling counter.
// - The `debug_assert!` added at compile-time in `compile_raw_inner`
//   validates every patch-site pointer lands in a `Box` the buffer is
//   about to own (belt-and-braces: the code path is structural already,
//   but the assert catches regressions if construction is refactored).
//
// Part of #494 / #364.

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

/// Straight-line function: entry: RET. One block, one counter — enough to
/// exercise the counter patch-site path without pulling in branch logic.
#[cfg(target_arch = "aarch64")]
fn build_noop(name: &str) -> MachFunction {
    let sig = Signature::new(vec![], vec![]);
    let mut func = MachFunction::new(name.to_string(), sig);
    let entry = func.entry;
    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);
    resolve_branches(&mut func);
    func
}

/// Diamond with X0 == 0 arm, identical shape to `jit_block_counters.rs`.
#[cfg(target_arch = "aarch64")]
fn build_diamond() -> MachFunction {
    let sig = Signature::new(vec![Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("diamond_lt".to_string(), sig);

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
        vec![MachOperand::Imm(0x0), MachOperand::Block(then_b)],
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

/// Counter reads succeed repeatedly across many calls, confirming that
/// the Box-pinned counter addresses remain valid for the full lifetime of
/// the `ExecutableBuffer`. This exercises the positive case of the #494
/// lifetime contract: the trampolines write into counters the buffer owns,
/// and the reader accessors see consistent values afterward.
#[cfg(target_arch = "aarch64")]
#[test]
fn block_counts_survive_many_calls_under_same_buffer() {
    let jit = JitCompiler::new(JitConfig {
        profile_hooks: ProfileHookMode::BlockCounts,
        ..JitConfig::default()
    });
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_diamond()], &ext)
        .expect("compile_raw must succeed with BlockCounts on AArch64");

    let diamond: extern "C" fn(u64) -> u64 = unsafe {
        buf.get_fn_bound("diamond_lt")
            .expect("typed function pointer must exist")
            .into_inner()
    };

    // 1000 alternating calls — enough to shake out any pointer corruption
    // from `Box` address drift (which would be a Rust soundness bug, not a
    // regression here, but we verify structurally).
    const N: u64 = 1000;
    for i in 0..N {
        if i % 2 == 0 {
            assert_eq!(diamond(0), 11);
        } else {
            assert_eq!(diamond(i), 77);
        }
    }

    // Every call ran the entry and join blocks once. Odd/even arms split.
    assert_eq!(buf.block_count("diamond_lt", BlockId(0)), Some(N));
    assert_eq!(buf.block_count("diamond_lt", BlockId(1)), Some(N / 2));
    assert_eq!(buf.block_count("diamond_lt", BlockId(2)), Some(N / 2));
    assert_eq!(buf.block_count("diamond_lt", BlockId(3)), Some(N));
}

/// Dropping the `ExecutableBuffer` after the caller has finished using
/// the JIT function must not trigger any use-after-free or assertion
/// failure, even though the counter storage is freed shortly after the
/// executable pages are unmapped. This exercises the field drop order
/// documented in `ExecutableBuffer`'s struct-level doc comment: `memory`
/// unmaps first, `counters` / `timing_cells` / `timing_state` boxes
/// drop after, so there is no window in which a live trampoline could
/// observe a freed counter.
///
/// This is a "positive" safety test: we set up the scenario that WOULD
/// regress if `ExecutableBuffer`'s field order ever changed, and verify
/// the drop is clean.
#[cfg(target_arch = "aarch64")]
#[test]
fn drop_after_use_is_clean_under_block_counts() {
    let jit = JitCompiler::new(JitConfig {
        profile_hooks: ProfileHookMode::BlockCounts,
        ..JitConfig::default()
    });
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_diamond(), build_noop("lonely")], &ext)
        .expect("compile_raw must succeed with BlockCounts on AArch64");

    // Invoke the functions via lifetime-bound pointers — these borrow
    // `buf` so `buf` cannot be dropped until the returned values go out
    // of scope.
    {
        let diamond: extern "C" fn(u64) -> u64 = unsafe {
            buf.get_fn_bound("diamond_lt")
                .expect("diamond_lt symbol")
                .into_inner()
        };
        let lonely: extern "C" fn() = unsafe {
            buf.get_fn_bound("lonely")
                .expect("lonely symbol")
                .into_inner()
        };

        for _ in 0..10 {
            let _ = diamond(0);
            let _ = diamond(1);
            lonely();
        }
    }
    // Borrowed function pointers are now out of scope — safe to drop.

    // Snapshot counters before drop.
    let entry_count = buf.block_count("diamond_lt", BlockId(0));
    let lonely_count = buf.entry_count("lonely");
    assert_eq!(entry_count, Some(10 + 10)); // 20 calls total.
    assert!(lonely_count.is_some());

    // Explicit drop: `munmap` runs first (in `Drop::drop`), THEN Rust
    // field drops run in declaration order — `counters` after `memory`.
    // A panic or segfault here indicates the lifetime contract is broken.
    drop(buf);
}

/// The `BlockCountsAndTiming` path wires in an additional `Box<TimingState>`
/// allocation plus `Box<BlockTimingCell>` cells keyed by block. Dropping
/// that buffer must also be clean. Exercise the full Phase 3 setup.
#[cfg(target_arch = "aarch64")]
#[test]
fn drop_after_use_is_clean_under_block_counts_and_timing() {
    let jit = JitCompiler::new(JitConfig {
        profile_hooks: ProfileHookMode::BlockCountsAndTiming,
        ..JitConfig::default()
    });
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_diamond()], &ext)
        .expect("compile_raw must succeed with BlockCountsAndTiming on AArch64");

    {
        let diamond: extern "C" fn(u64) -> u64 = unsafe {
            buf.get_fn_bound("diamond_lt")
                .expect("diamond_lt symbol")
                .into_inner()
        };
        for i in 0..50 {
            let _ = diamond(i);
        }
    }

    // All four blocks have been entered; timing tuples should exist.
    for bid in [BlockId(0), BlockId(1), BlockId(2), BlockId(3)] {
        let timing = buf.block_timing("diamond_lt", bid);
        assert!(
            timing.is_some(),
            "block {} should have a timing cell under BlockCountsAndTiming",
            bid.0
        );
    }

    // Drop `buf` — this unmaps `memory` first, then drops the
    // `timing_state` and `timing_cells` boxes. Any UAF here would
    // surface as a panic, segfault, or assertion failure.
    drop(buf);
}

/// Repeated compile / drop cycles must each land cleanly. This stresses
/// the construction-time `debug_assert!` added in `compile_raw_inner`
/// (#494): every iteration re-runs the patch-site validation that each
/// baked-in counter pointer lands inside the `Box` the new buffer is
/// about to own.
#[cfg(target_arch = "aarch64")]
#[test]
fn compile_drop_cycles_preserve_invariant() {
    for iter in 0..16u64 {
        let jit = JitCompiler::new(JitConfig {
            profile_hooks: ProfileHookMode::BlockCounts,
            ..JitConfig::default()
        });
        let ext: HashMap<String, *const u8> = HashMap::new();
        let buf = jit
            .compile_raw(&[build_diamond()], &ext)
            .unwrap_or_else(|e| panic!("iter {}: compile_raw failed: {:?}", iter, e));

        let diamond: extern "C" fn(u64) -> u64 = unsafe {
            buf.get_fn_bound("diamond_lt")
                .expect("diamond_lt symbol")
                .into_inner()
        };
        assert_eq!(diamond(iter), if iter == 0 { 11 } else { 77 });
        assert_eq!(buf.block_count("diamond_lt", BlockId(0)), Some(1));
        drop(buf);
    }
}

/// Non-AArch64 placeholder — the counter-splice path is AArch64-only in
/// the initial #364 landing, so this test file would otherwise refuse to
/// compile on other hosts.
#[cfg(not(target_arch = "aarch64"))]
#[test]
fn block_counter_lifetime_aarch64_only_placeholder() {
    // Intentionally empty. A companion follow-up will extend BlockCounts
    // to x86-64; when that lands, this file should grow tests under
    // `#[cfg(target_arch = "x86_64")]`.
}
