// llvm2-codegen/tests/jit_tls.rs - TLS-via-pointer JIT convention test
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Part of #361 — JIT needs thread-local storage (TLS) access for tla2 model checker
//
// Verifies the Option A convention: callers resolve thread-local addresses
// in Rust and pass them as extern "C" pointer arguments. JIT code treats
// them as regular pointers.

#![cfg(target_arch = "aarch64")]

use std::cell::UnsafeCell;
use std::collections::HashMap;

use llvm2_codegen::jit::{JitCompiler, JitConfig};
use llvm2_ir::function::{MachFunction, Signature, Type};
use llvm2_ir::inst::{AArch64Opcode, MachInst};
use llvm2_ir::operand::MachOperand;
use llvm2_ir::regs::X0;

thread_local! {
    static SCRATCH: UnsafeCell<u64> = UnsafeCell::new(0);
}

fn build_load_tls_u64() -> MachFunction {
    let sig = Signature::new(vec![Type::Ptr], vec![Type::I64]);
    let mut func = MachFunction::new("load_tls_u64".to_string(), sig);
    let entry = func.entry;

    let load = MachInst::new(
        AArch64Opcode::LdrRI,
        vec![
            MachOperand::PReg(X0),
            MachOperand::PReg(X0),
            MachOperand::Imm(0),
        ],
    );
    let load_id = func.push_inst(load);
    func.append_inst(entry, load_id);

    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}

#[test]
fn test_jit_tls_via_pointer_argument() {
    const EXPECTED: u64 = 0xDEAD_BEEF_CAFE_BABE;
    const STACK_VALUE: u64 = 0x0123_4567_89AB_CDEF;

    let jit = JitCompiler::new(JitConfig::default());
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_load_tls_u64()], &ext)
        .expect("compile_raw should succeed");

    // SAFETY: `F` (`extern "C" fn(*const u64) -> u64`) matches the ABI of the
    // compiled `load_tls_u64` function. The returned `JitFn` is bound to
    // `buf`'s lifetime via `get_fn_bound`, preventing use-after-free.
    let jfn = unsafe {
        buf.get_fn_bound::<extern "C" fn(*const u64) -> u64>("load_tls_u64")
            .expect("should find 'load_tls_u64' symbol")
    };
    let f = *jfn.as_ref();

    SCRATCH.with(|cell| {
        unsafe {
            *cell.get() = EXPECTED;
        }
        let ptr = cell.get() as *const u64;
        assert_eq!(f(ptr), EXPECTED);
    });

    let stack_value = STACK_VALUE;
    assert_eq!(f(&stack_value), STACK_VALUE);
}
