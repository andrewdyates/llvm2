// llvm2-codegen/tests/jit_profiling.rs - End-to-end JIT profiling hook tests
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Exercises the JIT call-count profiling hook from the public API surface.
// Covers:
// - Per-function atomic call counters increment correctly under N calls
// - Counters for distinct functions stay independent
// - ProfileHookMode::None produces no profile data (zero-overhead default)
// - `profiles()` iterator lists every instrumented function
//
// Part of #364 — JIT profiling hooks (execution counters).

// `get_fn_bound` is the non-deprecated API; suppress only if necessary in the
// future. No `#![allow(deprecated)]` needed here.

use std::collections::HashMap;
use std::time::Instant;

use llvm2_codegen::Compiler;
#[cfg(target_arch = "aarch64")]
use llvm2_codegen::jit::{JitCompiler, JitConfig, ProfileHookMode};
#[cfg(target_arch = "aarch64")]
use llvm2_ir::function::{MachFunction, Signature, Type};
#[cfg(target_arch = "aarch64")]
use llvm2_ir::inst::{AArch64Opcode, MachInst};
#[cfg(target_arch = "aarch64")]
use llvm2_ir::operand::MachOperand;
#[cfg(target_arch = "aarch64")]
use llvm2_ir::regs::X0;

/// Build `fn <name>() -> i64 { <value> }` — MOVZ X0, #<value> ; RET
#[cfg(target_arch = "aarch64")]
fn build_return_const(name: &str, value: u16) -> MachFunction {
    let sig = Signature::new(vec![], vec![Type::I64]);
    let mut func = MachFunction::new(name.to_string(), sig);
    let entry = func.entry;

    let mov = MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::PReg(X0), MachOperand::Imm(value as i64)],
    );
    let mov_id = func.push_inst(mov);
    func.append_inst(entry, mov_id);

    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}

fn build_two_function_metrics_module() -> tmir::Module {
    use tmir::Ty;
    use tmir_build::ModuleBuilder;

    let mut mb = ModuleBuilder::new("jit_metrics");
    {
        let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
        let mut fb = mb.function("add_fn", ty);
        let entry = fb.create_block();
        let a = fb.add_block_param(entry, Ty::I64);
        let b = fb.add_block_param(entry, Ty::I64);
        fb.switch_to_block(entry);
        let sum = fb.add(Ty::I64, a, b);
        let doubled = fb.add(Ty::I64, sum, sum);
        fb.ret(vec![doubled]);
        fb.build();
    }
    {
        let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
        let mut fb = mb.function("mul_fn", ty);
        let entry = fb.create_block();
        let a = fb.add_block_param(entry, Ty::I64);
        let b = fb.add_block_param(entry, Ty::I64);
        fb.switch_to_block(entry);
        let prod = fb.mul(Ty::I64, a, b);
        let adjusted = fb.sub(Ty::I64, prod, a);
        fb.ret(vec![adjusted]);
        fb.build();
    }
    mb.build()
}

fn build_single_function_metrics_module() -> tmir::Module {
    use tmir::Ty;
    use tmir_build::ModuleBuilder;

    let mut mb = ModuleBuilder::new("jit_metrics_single");
    let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
    let mut fb = mb.function("work_fn", ty);
    let entry = fb.create_block();
    let a = fb.add_block_param(entry, Ty::I64);
    let b = fb.add_block_param(entry, Ty::I64);
    fb.switch_to_block(entry);
    let sum = fb.add(Ty::I64, a, b);
    let prod = fb.mul(Ty::I64, sum, b);
    let adjusted = fb.sub(Ty::I64, prod, a);
    fb.ret(vec![adjusted]);
    fb.build();
    mb.build()
}

/// Compile the function with profiling mode `mode`, call it `n` times, and
/// assert the returned value matches `expected`. Returns the executable buffer
/// so callers can inspect the profile snapshot.
#[cfg(target_arch = "aarch64")]
#[test]
fn profile_hook_counts_single_function_exact_n() {
    let jit = JitCompiler::new(JitConfig {
        profile_hooks: ProfileHookMode::CallCounts,
        ..JitConfig::default()
    });
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_return_const("answer", 42)], &ext)
        .expect("compile_raw succeeds with profiling enabled");

    let f: extern "C" fn() -> u64 = unsafe {
        buf.get_fn_bound("answer")
            .expect("typed function pointer must exist")
            .into_inner()
    };

    const N: u64 = 250;
    let mut observed_sum: u64 = 0;
    for _ in 0..N {
        observed_sum = observed_sum.wrapping_add(f());
    }
    assert_eq!(
        observed_sum,
        42u64.wrapping_mul(N),
        "function must still produce correct values under profiling"
    );

    let stats = buf
        .get_profile("answer")
        .expect("profile should be recorded for instrumented function");
    assert_eq!(
        stats.call_count, N,
        "counter must equal number of invocations"
    );

    let collected: Vec<(&str, _)> = buf.profiles().collect();
    assert_eq!(collected.len(), 1, "exactly one profiled function");
    assert_eq!(collected[0].0, "answer");
    assert_eq!(collected[0].1.call_count, N);
}

/// Two independently-compiled functions must have independent counters.
#[cfg(target_arch = "aarch64")]
#[test]
fn profile_hook_counts_multiple_functions_independently() {
    let jit = JitCompiler::new(JitConfig {
        profile_hooks: ProfileHookMode::CallCounts,
        ..JitConfig::default()
    });
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(
            &[
                build_return_const("alpha", 7),
                build_return_const("beta", 11),
            ],
            &ext,
        )
        .expect("compile_raw succeeds for two profiled functions");

    let alpha: extern "C" fn() -> u64 = unsafe {
        buf.get_fn_bound("alpha")
            .expect("alpha function pointer")
            .into_inner()
    };
    let beta: extern "C" fn() -> u64 = unsafe {
        buf.get_fn_bound("beta")
            .expect("beta function pointer")
            .into_inner()
    };

    const N_ALPHA: u64 = 17;
    const N_BETA: u64 = 41;
    for _ in 0..N_ALPHA {
        assert_eq!(alpha(), 7);
    }
    for _ in 0..N_BETA {
        assert_eq!(beta(), 11);
    }

    let a = buf.get_profile("alpha").expect("alpha profile");
    let b = buf.get_profile("beta").expect("beta profile");
    assert_eq!(a.call_count, N_ALPHA);
    assert_eq!(b.call_count, N_BETA);

    // Iterator surface must enumerate exactly both functions.
    let mut names: Vec<&str> = buf.profiles().map(|(n, _)| n).collect();
    names.sort();
    assert_eq!(names, vec!["alpha", "beta"]);
}

/// With the default profiling mode (None) no counters are allocated and the
/// JIT path stays zero-overhead.
#[cfg(target_arch = "aarch64")]
#[test]
fn profile_hook_disabled_produces_no_profiles() {
    let jit = JitCompiler::new(JitConfig::default());
    assert_eq!(
        JitConfig::default().profile_hooks,
        ProfileHookMode::None,
        "default profile mode must remain None for zero-overhead default"
    );
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_return_const("quiet", 5)], &ext)
        .expect("compile_raw succeeds without profiling");

    let f: extern "C" fn() -> u64 = unsafe {
        buf.get_fn_bound("quiet")
            .expect("quiet function pointer")
            .into_inner()
    };
    for _ in 0..10 {
        assert_eq!(f(), 5);
    }

    assert!(
        buf.get_profile("quiet").is_none(),
        "no profile entry when hooks disabled"
    );
    assert_eq!(
        buf.profiles().count(),
        0,
        "profiles iterator must be empty when hooks disabled"
    );
}

/// `CallCountsAndTiming` mode must at minimum record call counts. (Timing
/// aggregation is scaffolded for follow-up per the #364 progress comment; this
/// test pins the current behaviour so future timing work doesn't regress the
/// counter path.)
#[cfg(target_arch = "aarch64")]
#[test]
fn profile_hook_counts_and_timing_mode_also_increments_counter() {
    let jit = JitCompiler::new(JitConfig {
        profile_hooks: ProfileHookMode::CallCountsAndTiming,
        ..JitConfig::default()
    });
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_return_const("timed", 3)], &ext)
        .expect("compile_raw succeeds in CallCountsAndTiming mode");

    let f: extern "C" fn() -> u64 = unsafe {
        buf.get_fn_bound("timed")
            .expect("timed function pointer")
            .into_inner()
    };
    const N: u64 = 64;
    for _ in 0..N {
        assert_eq!(f(), 3);
    }
    let stats = buf.get_profile("timed").expect("profile recorded");
    assert_eq!(stats.call_count, N);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_reset_profile_zeroes_counter_and_returns_true() {
    let jit = JitCompiler::new(JitConfig {
        profile_hooks: ProfileHookMode::CallCounts,
        ..JitConfig::default()
    });
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_return_const("answer", 42)], &ext)
        .expect("compile_raw succeeds with profiling enabled");

    let f: extern "C" fn() -> u64 = unsafe {
        buf.get_fn_bound("answer")
            .expect("typed function pointer must exist")
            .into_inner()
    };

    for _ in 0..5 {
        assert_eq!(f(), 42);
    }
    assert_eq!(
        buf.get_profile("answer")
            .expect("profile recorded")
            .call_count,
        5
    );

    assert!(
        buf.reset_profile("answer"),
        "reset_profile should return true for known counters"
    );

    for _ in 0..3 {
        assert_eq!(f(), 42);
    }
    assert_eq!(
        buf.get_profile("answer")
            .expect("profile recorded")
            .call_count,
        3
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_reset_profile_unknown_name_returns_false() {
    let jit = JitCompiler::new(JitConfig {
        profile_hooks: ProfileHookMode::CallCounts,
        ..JitConfig::default()
    });
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_return_const("a", 7)], &ext)
        .expect("compile_raw succeeds with profiling enabled");

    let f: extern "C" fn() -> u64 = unsafe {
        buf.get_fn_bound("a")
            .expect("typed function pointer must exist")
            .into_inner()
    };

    for _ in 0..4 {
        assert_eq!(f(), 7);
    }

    assert!(
        !buf.reset_profile("nonexistent"),
        "reset_profile should return false for unknown counters"
    );
    assert_eq!(
        buf.get_profile("a").expect("profile recorded").call_count,
        4
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_reset_all_profiles_returns_counter_count() {
    let jit = JitCompiler::new(JitConfig {
        profile_hooks: ProfileHookMode::CallCounts,
        ..JitConfig::default()
    });
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(
            &[
                build_return_const("alpha", 7),
                build_return_const("beta", 11),
            ],
            &ext,
        )
        .expect("compile_raw succeeds for two profiled functions");

    let alpha: extern "C" fn() -> u64 = unsafe {
        buf.get_fn_bound("alpha")
            .expect("alpha function pointer")
            .into_inner()
    };
    let beta: extern "C" fn() -> u64 = unsafe {
        buf.get_fn_bound("beta")
            .expect("beta function pointer")
            .into_inner()
    };

    for _ in 0..6 {
        assert_eq!(alpha(), 7);
    }
    for _ in 0..2 {
        assert_eq!(beta(), 11);
    }

    assert_eq!(buf.reset_all_profiles(), 2);
    assert_eq!(
        buf.get_profile("alpha").expect("alpha profile").call_count,
        0
    );
    assert_eq!(buf.get_profile("beta").expect("beta profile").call_count, 0);
}

#[test]
fn test_per_function_metrics_populated() {
    let module = build_two_function_metrics_module();
    let compiler = Compiler::default_o2();
    let result = compiler
        .compile_module_to_jit(&module, &HashMap::new())
        .expect("compile_module_to_jit should succeed");

    assert_eq!(result.per_function_metrics.len(), 2);
    for metrics in &result.per_function_metrics {
        assert!(
            !metrics.name.is_empty(),
            "function name should be populated"
        );
        assert!(
            metrics.instruction_count > 0,
            "instruction_count should be populated for {}",
            metrics.name
        );
        assert!(
            metrics
                .phase_timings
                .isel
                .expect("isel timing should exist")
                > std::time::Duration::ZERO
        );
        assert!(
            metrics
                .phase_timings
                .optimization
                .expect("optimization timing should exist")
                > std::time::Duration::ZERO
        );
        assert!(
            metrics
                .phase_timings
                .regalloc
                .expect("regalloc timing should exist")
                > std::time::Duration::ZERO
        );
        assert!(
            metrics
                .phase_timings
                .frame_lowering
                .expect("frame lowering timing should exist")
                > std::time::Duration::ZERO
        );
        assert!(
            metrics
                .phase_timings
                .branch_resolution
                .expect("branch resolution timing should exist")
                > std::time::Duration::ZERO
        );
    }
}

#[test]
fn test_phase_timings_total_is_monotonic() {
    let module = build_single_function_metrics_module();
    let compiler = Compiler::default_o2();
    let start = Instant::now();
    let result = compiler
        .compile_module_to_jit(&module, &HashMap::new())
        .expect("compile_module_to_jit should succeed");
    let elapsed = start.elapsed();

    assert_eq!(result.per_function_metrics.len(), 1);
    let total = result.per_function_metrics[0].phase_timings.total();
    assert!(total > std::time::Duration::ZERO);
    assert!(
        total <= elapsed,
        "sum of per-phase timings ({total:?}) should not exceed external elapsed time ({elapsed:?})"
    );
}
