// llvm2-codegen/tests/abi_many_args_e2e.rs - AAPCS64 >8-arg ABI E2E test
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Verifies the stack-slot overflow path in crates/llvm2-lower/src/abi.rs::classify_params.
// Apple AArch64 ABI (AAPCS64):
//   - GPR args 0-7 in X0-X7; arg 8+ spills to SP+0, SP+8, ... (8-byte aligned).
//   - FPR args 0-7 in V0-V7; arg 8+ spills to stack (8-byte aligned for F64).
//
// If the ABI is broken (e.g., overflow args read from the wrong slot), the
// compiled function returns a wrong sum and the `cc`-linked driver exits
// non-zero. If the ABI is correct, the driver prints "OK".
//
// Part of #489

#![cfg(target_arch = "aarch64")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use llvm2_codegen::pipeline::{OptLevel, Pipeline, PipelineConfig};

use tmir::{
    Block as TmirBlock, CastOp, Constant, FuncTy, Function as TmirFunction, Module as TmirModule,
};
use tmir::{BinOp, Inst, InstrNode};
use tmir::{BlockId, FuncId, Ty, ValueId};

// ---------------------------------------------------------------------------
// Helpers (mirror crates/llvm2-codegen/tests/tla2_bfs_minimal.rs)
// ---------------------------------------------------------------------------

fn v(n: u32) -> ValueId {
    ValueId::new(n)
}

fn b(n: u32) -> BlockId {
    BlockId::new(n)
}

fn is_aarch64() -> bool {
    cfg!(target_arch = "aarch64")
}

fn has_cc() -> bool {
    Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn make_test_dir(test_name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("llvm2_e2e_{}", test_name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create test dir");
    dir
}

fn write_object_file(dir: &Path, filename: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(filename);
    fs::write(&path, bytes).expect("write .o file");
    path
}

fn write_c_driver(dir: &Path, filename: &str, source: &str) -> PathBuf {
    let path = dir.join(filename);
    fs::write(&path, source).expect("write C driver");
    path
}

fn link_with_cc(dir: &Path, driver_c: &Path, obj: &Path, output_name: &str) -> PathBuf {
    let binary = dir.join(output_name);
    let result = Command::new("cc")
        .arg("-o")
        .arg(&binary)
        .arg(driver_c)
        .arg(obj)
        .arg("-Wl,-no_pie")
        .output()
        .expect("run cc");
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        panic!("Linking failed: {}", stderr);
    }
    binary
}

fn run_binary(binary: &Path) -> (i32, String) {
    let result = Command::new(binary).output().expect("run binary");
    let stdout = String::from_utf8_lossy(&result.stdout).to_string();
    (result.status.code().unwrap_or(-1), stdout)
}

fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

fn compile_tmir(
    tmir_func: &TmirFunction,
    module: &TmirModule,
    opt_level: OptLevel,
) -> Result<Vec<u8>, String> {
    let (lir_func, _proof_ctx) =
        llvm2_lower::translate_function(tmir_func, module).map_err(|e| format!("adapter: {}", e))?;
    let config = PipelineConfig {
        opt_level,
        emit_debug: false,
        ..Default::default()
    };
    let pipeline = Pipeline::new(config);
    pipeline
        .compile_function(&lir_func)
        .map_err(|e| format!("pipeline: {}", e))
}

fn assert_valid_macho(bytes: &[u8], ctx: &str) {
    assert!(bytes.len() >= 4, "{}: too small ({} bytes)", ctx, bytes.len());
    assert_eq!(
        &bytes[0..4],
        &[0xCF, 0xFA, 0xED, 0xFE],
        "{}: invalid Mach-O magic",
        ctx
    );
}

struct VidCounter(u32);

impl VidCounter {
    fn new(start: u32) -> Self {
        Self(start)
    }

    fn next(&mut self) -> ValueId {
        let v = ValueId::new(self.0);
        self.0 += 1;
        v
    }
}

#[allow(dead_code)]
fn const_i64(vid: &mut VidCounter, val: i64) -> (ValueId, InstrNode) {
    let r = vid.next();
    let node = InstrNode::new(Inst::Const {
        ty: Ty::I64,
        value: Constant::Int(val as i128),
    })
    .with_result(r);
    (r, node)
}

fn binop_i64(
    vid: &mut VidCounter,
    op: BinOp,
    lhs: ValueId,
    rhs: ValueId,
) -> (ValueId, InstrNode) {
    let r = vid.next();
    let node = InstrNode::new(Inst::BinOp {
        op,
        ty: Ty::I64,
        lhs,
        rhs,
    })
    .with_result(r);
    (r, node)
}

fn add_i64(vid: &mut VidCounter, lhs: ValueId, rhs: ValueId) -> (ValueId, InstrNode) {
    binop_i64(vid, BinOp::Add, lhs, rhs)
}

fn fp_to_si_i64(vid: &mut VidCounter, operand: ValueId) -> (ValueId, InstrNode) {
    let r = vid.next();
    let node = InstrNode::new(Inst::Cast {
        op: CastOp::FPToSI,
        src_ty: Ty::F64,
        dst_ty: Ty::I64,
        operand,
    })
    .with_result(r);
    (r, node)
}

// ---------------------------------------------------------------------------
// Test 1: sum10_i64 (10 i64 args, last 2 spill to stack)
// ---------------------------------------------------------------------------
//
// AAPCS64 placement:
//   v(0)..v(7) -> X0..X7  (registers)
//   v(8)       -> [SP + 0]  (stack overflow)
//   v(9)       -> [SP + 8]  (stack overflow)
//
// Body: iterative add chain v(0)+v(1)+...+v(9).
// Expected result for (1,2,3,4,5,6,7,8,9,10): 55.

fn build_sum10_i64() -> (TmirFunction, TmirModule) {
    let mut module = TmirModule::new("abi_sum10_i64");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64; 10],
        returns: vec![Ty::I64],
        is_vararg: false,
    });

    let entry = b(0);
    let mut func = TmirFunction::new(FuncId::new(0), "sum10_i64", ft_id, entry);

    // Block params: v(0)..v(9) all i64.
    let params: Vec<(ValueId, Ty)> = (0..10).map(|i| (v(i), Ty::I64)).collect();

    // Result ids start at 10 to avoid colliding with block params.
    let mut vid = VidCounter::new(10);
    let mut body = Vec::new();

    // Iteratively accumulate: acc = v(0) + v(1); acc = acc + v(2); ...
    let mut acc: ValueId = v(0);
    for i in 1..10u32 {
        let (next, node) = add_i64(&mut vid, acc, v(i));
        body.push(node);
        acc = next;
    }

    body.push(InstrNode::new(Inst::Return {
        values: vec![acc],
    }));

    func.blocks = vec![TmirBlock {
        id: entry,
        params,
        body,
    }];

    module.add_function(func.clone());
    (func, module)
}

const C_DRIVER_SUM10: &str = r#"
#include <stdint.h>
#include <stdio.h>

extern int64_t sum10_i64(int64_t, int64_t, int64_t, int64_t, int64_t,
                         int64_t, int64_t, int64_t, int64_t, int64_t);

int main(void) {
    int64_t r = sum10_i64(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    if (r != 55) {
        printf("sum10_i64 r=%lld expected=55\n", (long long)r);
        return 1;
    }
    printf("OK\n");
    return 0;
}
"#;

fn run_sum10_i64_e2e(opt_level: OptLevel, test_name: &str) {
    if !is_aarch64() {
        eprintln!("skipping {}: requires aarch64", test_name);
        return;
    }
    if !has_cc() {
        eprintln!("skipping {}: cc not available", test_name);
        return;
    }

    let dir = make_test_dir(test_name);
    let (func, module) = build_sum10_i64();
    let obj_bytes = compile_tmir(&func, &module, opt_level)
        .expect("sum10_i64 compilation should succeed");
    assert_valid_macho(&obj_bytes, test_name);

    let obj_path = write_object_file(&dir, "sum10_i64.o", &obj_bytes);
    let driver_path = write_c_driver(&dir, "driver.c", C_DRIVER_SUM10);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "sum10_i64_test");

    let (exit_code, stdout) = run_binary(&binary);
    assert_eq!(
        exit_code, 0,
        "binary should exit cleanly; stdout: {}",
        stdout
    );
    assert_eq!(stdout.trim(), "OK", "unexpected driver output: {}", stdout);

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test 2: mixed_4i64_10f64 (4 i64 + 10 f64, last 2 f64 spill to stack)
// ---------------------------------------------------------------------------
//
// AAPCS64 placement:
//   v(0)..v(3)   -> X0..X3   (integer regs; X4..X7 unused)
//   v(4)..v(11)  -> V0..V7   (FP regs)
//   v(12)        -> [SP + 0] (FP overflow, 8-byte aligned)
//   v(13)        -> [SP + 8] (FP overflow)
//
// Body: FPToSI-cast each f64 to i64, then sum all 14 i64 values.
// Expected result for (1,2,3,4, 1.0..10.0): (1+2+3+4) + (1+...+10) = 10 + 55 = 65.

fn build_mixed_4i64_10f64() -> (TmirFunction, TmirModule) {
    let mut module = TmirModule::new("abi_mixed_4i64_10f64");
    let mut params_ty: Vec<Ty> = vec![Ty::I64; 4];
    params_ty.extend(std::iter::repeat(Ty::F64).take(10));
    let ft_id = module.add_func_type(FuncTy {
        params: params_ty,
        returns: vec![Ty::I64],
        is_vararg: false,
    });

    let entry = b(0);
    let mut func = TmirFunction::new(FuncId::new(0), "mixed_4i64_10f64", ft_id, entry);

    // Block params: v(0)..v(3) i64, v(4)..v(13) f64.
    let mut params: Vec<(ValueId, Ty)> = (0..4).map(|i| (v(i), Ty::I64)).collect();
    params.extend((4..14).map(|i| (v(i), Ty::F64)));

    // Result ids start at 14.
    let mut vid = VidCounter::new(14);
    let mut body = Vec::new();

    // Cast each f64 parameter to i64.
    let mut casts: Vec<ValueId> = Vec::with_capacity(10);
    for i in 4..14u32 {
        let (r, node) = fp_to_si_i64(&mut vid, v(i));
        body.push(node);
        casts.push(r);
    }

    // Accumulate: start with the 4 i64 block params, then add each cast result.
    let mut acc: ValueId = v(0);
    for i in 1..4u32 {
        let (next, node) = add_i64(&mut vid, acc, v(i));
        body.push(node);
        acc = next;
    }
    for cast_r in &casts {
        let (next, node) = add_i64(&mut vid, acc, *cast_r);
        body.push(node);
        acc = next;
    }

    body.push(InstrNode::new(Inst::Return {
        values: vec![acc],
    }));

    func.blocks = vec![TmirBlock {
        id: entry,
        params,
        body,
    }];

    module.add_function(func.clone());
    (func, module)
}

const C_DRIVER_MIXED: &str = r#"
#include <stdint.h>
#include <stdio.h>

extern int64_t mixed_4i64_10f64(int64_t, int64_t, int64_t, int64_t,
                                double, double, double, double, double,
                                double, double, double, double, double);

int main(void) {
    int64_t r = mixed_4i64_10f64(1, 2, 3, 4,
                                 1.0, 2.0, 3.0, 4.0, 5.0,
                                 6.0, 7.0, 8.0, 9.0, 10.0);
    // (1+2+3+4) + (1+2+3+4+5+6+7+8+9+10) = 10 + 55 = 65
    if (r != 65) {
        printf("mixed_4i64_10f64 r=%lld expected=65\n", (long long)r);
        return 1;
    }
    printf("OK\n");
    return 0;
}
"#;

fn run_mixed_4i64_10f64_e2e(opt_level: OptLevel, test_name: &str) {
    if !is_aarch64() {
        eprintln!("skipping {}: requires aarch64", test_name);
        return;
    }
    if !has_cc() {
        eprintln!("skipping {}: cc not available", test_name);
        return;
    }

    let dir = make_test_dir(test_name);
    let (func, module) = build_mixed_4i64_10f64();
    let obj_bytes = compile_tmir(&func, &module, opt_level)
        .expect("mixed_4i64_10f64 compilation should succeed");
    assert_valid_macho(&obj_bytes, test_name);

    let obj_path = write_object_file(&dir, "mixed_4i64_10f64.o", &obj_bytes);
    let driver_path = write_c_driver(&dir, "driver.c", C_DRIVER_MIXED);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "mixed_4i64_10f64_test");

    let (exit_code, stdout) = run_binary(&binary);
    assert_eq!(
        exit_code, 0,
        "binary should exit cleanly; stdout: {}",
        stdout
    );
    assert_eq!(stdout.trim(), "OK", "unexpected driver output: {}", stdout);

    cleanup(&dir);
}

// ===========================================================================
// Tests — compile-only + O0/O2 divergence + end-to-end
// ===========================================================================

#[test]
fn test_sum10_i64_compiles_o0() {
    let (func, module) = build_sum10_i64();
    let obj_bytes = compile_tmir(&func, &module, OptLevel::O0)
        .expect("sum10_i64 should compile at O0");
    assert_valid_macho(&obj_bytes, "sum10_i64 O0");
}

#[test]
fn test_sum10_i64_compiles_o2() {
    let (func, module) = build_sum10_i64();
    let obj_bytes = compile_tmir(&func, &module, OptLevel::O2)
        .expect("sum10_i64 should compile at O2");
    assert_valid_macho(&obj_bytes, "sum10_i64 O2");
}

#[test]
fn test_sum10_i64_o0_vs_o2_differ() {
    let (func, module) = build_sum10_i64();
    let obj_o0 =
        compile_tmir(&func, &module, OptLevel::O0).expect("sum10_i64 should compile at O0");
    let obj_o2 =
        compile_tmir(&func, &module, OptLevel::O2).expect("sum10_i64 should compile at O2");
    assert_ne!(
        obj_o0, obj_o2,
        "O0 and O2 should produce different object files"
    );
}

#[test]
fn test_sum10_i64_e2e_correctness() {
    run_sum10_i64_e2e(OptLevel::O0, "sum10_i64_e2e_o0");
}

#[test]
fn test_sum10_i64_e2e_correctness_o2() {
    run_sum10_i64_e2e(OptLevel::O2, "sum10_i64_e2e_o2");
}

#[test]
fn test_mixed_4i64_10f64_compiles_o0() {
    let (func, module) = build_mixed_4i64_10f64();
    let obj_bytes = compile_tmir(&func, &module, OptLevel::O0)
        .expect("mixed_4i64_10f64 should compile at O0");
    assert_valid_macho(&obj_bytes, "mixed_4i64_10f64 O0");
}

#[test]
fn test_mixed_4i64_10f64_compiles_o2() {
    let (func, module) = build_mixed_4i64_10f64();
    let obj_bytes = compile_tmir(&func, &module, OptLevel::O2)
        .expect("mixed_4i64_10f64 should compile at O2");
    assert_valid_macho(&obj_bytes, "mixed_4i64_10f64 O2");
}

#[test]
fn test_mixed_4i64_10f64_o0_vs_o2_differ() {
    let (func, module) = build_mixed_4i64_10f64();
    let obj_o0 = compile_tmir(&func, &module, OptLevel::O0)
        .expect("mixed_4i64_10f64 should compile at O0");
    let obj_o2 = compile_tmir(&func, &module, OptLevel::O2)
        .expect("mixed_4i64_10f64 should compile at O2");
    assert_ne!(
        obj_o0, obj_o2,
        "O0 and O2 should produce different object files"
    );
}

#[test]
fn test_mixed_4i64_10f64_e2e_correctness() {
    run_mixed_4i64_10f64_e2e(OptLevel::O0, "mixed_4i64_10f64_e2e_o0");
}

#[test]
fn test_mixed_4i64_10f64_e2e_correctness_o2() {
    run_mixed_4i64_10f64_e2e(OptLevel::O2, "mixed_4i64_10f64_e2e_o2");
}
