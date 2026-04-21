// llvm2-codegen/tests/jit_calling_convention_contract.rs
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//! Regression test that pins the #431 calling-convention contract. JIT-compiled
//! tMIR functions must follow the host C ABI so that `std::mem::transmute` to
//! `extern "C" fn(...)` is sound; the contract is defined in `crates/llvm2-codegen/src/jit.rs` module docs.

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use llvm2_codegen::jit::{JitCompiler, JitConfig};
    use llvm2_ir::function::{MachFunction, Signature, Type};
    use llvm2_ir::inst::{AArch64Opcode, MachInst};
    use llvm2_ir::operand::MachOperand;
    use llvm2_ir::regs::{X0, X1};
    use std::collections::HashMap;

    fn build_add() -> MachFunction {
        let mut f = MachFunction::new("add".to_string(), Signature::new(vec![Type::I64, Type::I64], vec![Type::I64]));
        let e = f.entry;
        let add = MachInst::new(AArch64Opcode::AddRR, vec![MachOperand::PReg(X0), MachOperand::PReg(X0), MachOperand::PReg(X1)]);
        let id = f.push_inst(add);
        f.append_inst(e, id);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let id = f.push_inst(ret);
        f.append_inst(e, id);
        f
    }

    #[test]
    fn aarch64_aapcs64_contract_extern_c_fn_i64_i64_i64() {
        let jit = JitCompiler::new(JitConfig::default());
        let ext: HashMap<String, *const u8> = HashMap::new();
        let buf = jit.compile_raw(&[build_add()], &ext).expect("compile_raw ok");
        let f = unsafe { buf.get_fn_bound::<extern "C" fn(i64, i64) -> i64>("add").expect("symbol 'add'") }.into_inner();
        assert_eq!(f(3, 4), 7);
        assert_eq!(f(-1, 1), 0);
        assert_eq!(f(i64::MAX, 0), i64::MAX);
        assert_eq!(f(-100, -200), -300);
    }
}

#[cfg(target_arch = "x86_64")]
// x86-64 contract coverage lives in tests/jit_integration_x86_64.rs (#467); this file adds only the aarch64 contract-named regression.
const _: () = ();
