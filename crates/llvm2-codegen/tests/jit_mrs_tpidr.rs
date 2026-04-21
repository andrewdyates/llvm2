// llvm2-codegen/tests/jit_mrs_tpidr.rs
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Part of #383 — first end-to-end exercise of the AArch64 TLS local-exec
// primitive (MRS Xd, TPIDR_EL0). Compiles a MachFunction that reads
// TPIDR_EL0 and returns it, then compares against the value read from
// the host process via inline asm.

#![cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]

use std::collections::HashMap;

use llvm2_codegen::jit::{JitCompiler, JitConfig};
use llvm2_ir::function::{MachFunction, Signature, Type};
use llvm2_ir::inst::{AArch64Opcode, MachInst};
use llvm2_ir::operand::MachOperand;
use llvm2_ir::regs::X0;

/// 16-bit systemreg encoding for TPIDR_EL0 (op0=11 op1=011 CRn=1101 CRm=0000 op2=010).
const TPIDR_EL0_SYSREG: i64 = 0xDE82;

/// Read TPIDR_EL0 from the calling Rust thread via inline asm.
#[inline(never)]
fn host_read_tpidr_el0() -> u64 {
    let v: u64;
    // SAFETY: `mrs` only reads a system register; no memory access.
    unsafe {
        core::arch::asm!(
            "mrs {v}, TPIDR_EL0",
            v = out(reg) v,
            options(nostack, nomem, preserves_flags),
        );
    }
    v
}

fn build_read_tpidr_el0() -> MachFunction {
    let sig = Signature::new(vec![], vec![Type::I64]);
    let mut func = MachFunction::new("read_tpidr_el0".to_string(), sig);
    let entry = func.entry;

    // MRS X0, TPIDR_EL0
    let mrs = MachInst::new(
        AArch64Opcode::Mrs,
        vec![MachOperand::PReg(X0), MachOperand::Imm(TPIDR_EL0_SYSREG)],
    );
    let mrs_id = func.push_inst(mrs);
    func.append_inst(entry, mrs_id);

    // RET
    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}

#[test]
fn test_jit_mrs_tpidr_el0_matches_host() {
    let jit = JitCompiler::new(JitConfig::default());
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_read_tpidr_el0()], &ext)
        .expect("compile_raw should succeed for a two-instruction function");

    // SAFETY: signature `extern "C" fn() -> u64` matches the tMIR signature
    // `() -> i64` under Apple DarwinPCS / AAPCS64. Lifetime is bound to `buf`.
    let jfn = unsafe {
        buf.get_fn_bound::<extern "C" fn() -> u64>("read_tpidr_el0")
            .expect("symbol 'read_tpidr_el0' should exist in the JIT buffer")
    };
    let f = *jfn.as_ref();

    // Invoking the JIT'd function stays on the same Rust thread, so the
    // TPIDR_EL0 value observed by the JIT'd code must match the value
    // observed by the host inline-asm read: the kernel gives every thread
    // exactly one TPIDR_EL0 value for its entire lifetime on the supported
    // hosts (Apple Silicon macOS and Linux).
    let host_tp = host_read_tpidr_el0();
    let jit_tp = f();
    assert_eq!(
        jit_tp, host_tp,
        "JIT MRS TPIDR_EL0 = {jit_tp:#x} must equal host MRS TPIDR_EL0 = {host_tp:#x} on the same thread"
    );
    // Linux AArch64 exposes the thread pointer in TPIDR_EL0. Some Darwin
    // environments return zero here even though the MRS encoding itself is
    // correct, so cross-host coverage relies on the host-vs-JIT equality
    // check above and only keeps the non-zero assertion on Linux.
    if cfg!(target_os = "linux") {
        assert_ne!(
            jit_tp, 0,
            "TPIDR_EL0 should not be zero on a running thread"
        );
    }
}
