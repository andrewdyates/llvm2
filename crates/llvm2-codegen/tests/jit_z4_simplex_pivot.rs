// llvm2-codegen/tests/jit_z4_simplex_pivot.rs
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// z4 simplex pivot hot-path JIT smoke test.
//
// Part of #485 — z4 consumes LLVM2 as a JIT backend for its simplex solver
// (see `~/z4/crates/z4-jit/src/simplex_jit.rs` for the i64 fast-path
// analogue). This test targets the f64 pivot-row-normalization step, which
// is what drives the general LP path: given a pointer to a row of doubles
// and a runtime pivot-column index, divide every column by the pivot value
// so that `row[pivot_col] == 1.0` afterwards.
//
// Exercises:
// - FP register-offset load (LdrRO with D-register destination + X-register
//   base + X-register index, packed extend = LSL #3 for 8-byte stride)
// - FP compile-time-offset load/store (LdrRI / StrRI with D-register)
// - FP divide (FdivRR)
// - End-to-end JIT compile + call via extern "C" fn-ptr

#![cfg(target_arch = "aarch64")]

use std::collections::HashMap;

use llvm2_codegen::jit::{JitCompiler, JitConfig};
use llvm2_ir::function::{MachFunction, Signature, Type};
use llvm2_ir::inst::{AArch64Opcode, MachInst};
use llvm2_ir::operand::MachOperand;
use llvm2_ir::regs::{D0, D1, X0, X1};

fn build_pivot_normalize_4col() -> MachFunction {
    let sig = Signature::new(vec![Type::Ptr, Type::I64], vec![]);
    let mut func = MachFunction::new("pivot_normalize_4col".to_string(), sig);
    let entry = func.entry;

    // LDR D0, [X0, X1, LSL #3]  — D0 = row[pivot_col]
    let ldr_pv = MachInst::new(
        AArch64Opcode::LdrRO,
        vec![
            MachOperand::PReg(D0),
            MachOperand::PReg(X0),
            MachOperand::PReg(X1),
            MachOperand::Imm(7), // (option=0b011 LSL) << 1 | S=1 -> shift by 3 bits (*8)
        ],
    );
    let id = func.push_inst(ldr_pv);
    func.append_inst(entry, id);

    for j in 0..4i64 {
        let ldr = MachInst::new(
            AArch64Opcode::LdrRI,
            vec![
                MachOperand::PReg(D1),
                MachOperand::PReg(X0),
                MachOperand::Imm(j * 8),
            ],
        );
        let id = func.push_inst(ldr);
        func.append_inst(entry, id);

        let fdiv = MachInst::new(
            AArch64Opcode::FdivRR,
            vec![
                MachOperand::PReg(D1),
                MachOperand::PReg(D1),
                MachOperand::PReg(D0),
            ],
        );
        let id = func.push_inst(fdiv);
        func.append_inst(entry, id);

        let str_ = MachInst::new(
            AArch64Opcode::StrRI,
            vec![
                MachOperand::PReg(D1),
                MachOperand::PReg(X0),
                MachOperand::Imm(j * 8),
            ],
        );
        let id = func.push_inst(str_);
        func.append_inst(entry, id);
    }

    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let id = func.push_inst(ret);
    func.append_inst(entry, id);

    func
}

#[test]
fn test_jit_simplex_pivot_normalize_basic() {
    let jit = JitCompiler::new(JitConfig::default());
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_pivot_normalize_4col()], &ext)
        .expect("compile_raw should succeed for pivot_normalize_4col");

    let mut row: [f64; 4] = [2.0, 4.0, 8.0, 16.0];
    let pivot_col: i64 = 0;

    let f: unsafe extern "C" fn(*mut f64, i64) = unsafe {
        buf.get_fn_bound::<unsafe extern "C" fn(*mut f64, i64)>("pivot_normalize_4col")
            .expect("should find symbol")
    }
    .into_inner();

    unsafe {
        f(row.as_mut_ptr(), pivot_col);
    }

    assert_eq!(row[0], 1.0);
    assert_eq!(row[1], 2.0);
    assert_eq!(row[2], 4.0);
    assert_eq!(row[3], 8.0);
}

#[test]
fn test_jit_simplex_pivot_normalize_different_pivot_col() {
    let jit = JitCompiler::new(JitConfig::default());
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_pivot_normalize_4col()], &ext)
        .expect("compile_raw should succeed for pivot_normalize_4col");

    let mut row: [f64; 4] = [10.0, 5.0, 20.0, 40.0];
    let pivot_col: i64 = 1;

    let f: unsafe extern "C" fn(*mut f64, i64) = unsafe {
        buf.get_fn_bound::<unsafe extern "C" fn(*mut f64, i64)>("pivot_normalize_4col")
            .expect("should find symbol")
    }
    .into_inner();

    unsafe {
        f(row.as_mut_ptr(), pivot_col);
    }

    assert_eq!(row[0], 2.0);
    assert_eq!(row[1], 1.0);
    assert_eq!(row[2], 4.0);
    assert_eq!(row[3], 8.0);
}

#[test]
fn test_jit_simplex_pivot_matches_host_semantics() {
    let jit = JitCompiler::new(JitConfig::default());
    let ext: HashMap<String, *const u8> = HashMap::new();
    let buf = jit
        .compile_raw(&[build_pivot_normalize_4col()], &ext)
        .expect("compile_raw should succeed for pivot_normalize_4col");

    let f: unsafe extern "C" fn(*mut f64, i64) = unsafe {
        buf.get_fn_bound::<unsafe extern "C" fn(*mut f64, i64)>("pivot_normalize_4col")
            .expect("should find symbol")
    }
    .into_inner();

    let mut row: [f64; 4] = [1.0, 3.0, 7.0, 13.0];
    let pivot_col: i64 = 0;
    unsafe {
        f(row.as_mut_ptr(), pivot_col);
    }
    assert_eq!(row, [1.0, 3.0, 7.0, 13.0]);

    let expected: [f64; 4] = [3.0 / 3.0, 9.0 / 3.0, 27.0 / 3.0, 81.0 / 3.0];
    let mut row: [f64; 4] = [3.0, 9.0, 27.0, 81.0];
    let pivot_col: i64 = 0;
    unsafe {
        f(row.as_mut_ptr(), pivot_col);
    }
    assert_eq!(row, expected);
}
