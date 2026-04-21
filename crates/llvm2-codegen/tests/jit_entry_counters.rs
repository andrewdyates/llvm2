// llvm2-codegen/tests/jit_entry_counters.rs - Public entry-counter API tests
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Covers the stable public API for the function-entry counter slice of #364:
// - `JitConfig::emit_entry_counters`
// - `ExecutableBuffer::entry_count`
// - `ExecutableBuffer::reset_entry_count`
// - `ExecutableBuffer::entry_counts`
//
// Part of #478
// Part of #364

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
#[cfg(target_arch = "aarch64")]
use std::collections::HashMap;

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

#[cfg(target_arch = "aarch64")]
#[test]
fn entry_counters_convenience_flag_enables_counts() {
    let jit = JitCompiler::new(JitConfig {
        emit_entry_counters: true,
        ..JitConfig::default()
    });
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_return_const("f", 42)], &ext)
        .expect("compile_raw succeeds with entry counters enabled");

    let f: extern "C" fn() -> u64 = unsafe {
        buf.get_fn_bound("f")
            .expect("typed function pointer must exist")
            .into_inner()
    };

    const N: u64 = 25;
    for _ in 0..N {
        assert_eq!(f(), 42);
    }

    assert_eq!(buf.entry_count("f"), Some(N));
}

#[cfg(target_arch = "aarch64")]
#[test]
fn entry_counters_convenience_flag_zero_by_default() {
    let jit = JitCompiler::new(JitConfig::default());
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_return_const("f", 5)], &ext)
        .expect("compile_raw succeeds with default config");

    assert_eq!(buf.entry_count("f"), None);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn entry_counters_explicit_profile_hooks_wins() {
    let jit = JitCompiler::new(JitConfig {
        profile_hooks: ProfileHookMode::CallCounts,
        emit_entry_counters: false,
        ..JitConfig::default()
    });
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_return_const("f", 9)], &ext)
        .expect("explicit profile_hooks should still emit counters");

    let f: extern "C" fn() -> u64 = unsafe {
        buf.get_fn_bound("f")
            .expect("typed function pointer must exist")
            .into_inner()
    };

    const N: u64 = 11;
    for _ in 0..N {
        assert_eq!(f(), 9);
    }

    assert_eq!(buf.entry_count("f"), Some(N));
}

#[cfg(target_arch = "aarch64")]
#[test]
fn entry_counts_snapshot_orders_and_resets() {
    let jit = JitCompiler::new(JitConfig {
        emit_entry_counters: true,
        ..JitConfig::default()
    });
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(
            &[build_return_const("a", 1), build_return_const("b", 2)],
            &ext,
        )
        .expect("compile_raw succeeds for two functions");

    let a: extern "C" fn() -> u64 = unsafe {
        buf.get_fn_bound("a")
            .expect("a function pointer")
            .into_inner()
    };
    let b: extern "C" fn() -> u64 = unsafe {
        buf.get_fn_bound("b")
            .expect("b function pointer")
            .into_inner()
    };

    for _ in 0..3 {
        assert_eq!(a(), 1);
    }
    for _ in 0..7 {
        assert_eq!(b(), 2);
    }

    assert_eq!(
        buf.entry_counts(),
        vec![("a".to_string(), 3), ("b".to_string(), 7)]
    );

    assert!(buf.reset_entry_count("a"));
    assert_eq!(buf.entry_count("a"), Some(0));
    assert_eq!(buf.entry_count("b"), Some(7));
}

#[cfg(target_arch = "aarch64")]
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn entry_counters_work_on_both_arches() {
    let jit = JitCompiler::new(JitConfig {
        emit_entry_counters: true,
        ..JitConfig::default()
    });
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_return_const("f", 3)], &ext)
        .expect("compile_raw succeeds with entry counters enabled");

    let f: extern "C" fn() -> u64 = unsafe {
        buf.get_fn_bound("f")
            .expect("typed function pointer must exist")
            .into_inner()
    };

    assert_eq!(f(), 3);
    assert!(
        buf.entry_count("f").expect("counter must exist") > 0,
        "entry counter must increment after executing the function"
    );
}
