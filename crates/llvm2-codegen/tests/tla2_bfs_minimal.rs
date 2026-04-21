// llvm2-codegen/tests/tla2_bfs_minimal.rs - minimal tla2 BFS kernel as tMIR
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Minimal end-to-end integration test for the tla2 model-checker BFS backend.
// Verifies compile-time-offset i64 array load/store, runtime-index i64 GEP,
// pointer+i64 calling convention, and O0/O2 compilation behavior.
//
// Part of #270

#![cfg(target_arch = "aarch64")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use llvm2_codegen::pipeline::{OptLevel, Pipeline, PipelineConfig};

use tmir::{
    Block as TmirBlock, Constant, FuncTy, Function as TmirFunction, Module as TmirModule,
};
use tmir::{BinOp, Inst, InstrNode};
use tmir::{BlockId, FuncId, Ty, ValueId};

// ---------------------------------------------------------------------------
// Helpers
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

// ---------------------------------------------------------------------------
// Compilation helper
// ---------------------------------------------------------------------------

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

/// ValueId counter for building tMIR functions without collisions.
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

fn const_i64(vid: &mut VidCounter, val: i64) -> (ValueId, InstrNode) {
    let r = vid.next();
    let node = InstrNode::new(Inst::Const {
        ty: Ty::I64,
        value: Constant::Int(val as i128),
    })
    .with_result(r);
    (r, node)
}

fn gep_i64(vid: &mut VidCounter, base: ValueId, index: ValueId) -> (ValueId, InstrNode) {
    let r = vid.next();
    let node = InstrNode::new(Inst::GEP {
        pointee_ty: Ty::I64,
        base,
        indices: vec![index],
    })
    .with_result(r);
    (r, node)
}

fn load_i64(vid: &mut VidCounter, ptr: ValueId) -> (ValueId, InstrNode) {
    let r = vid.next();
    let node = InstrNode::new(Inst::Load {
        ty: Ty::I64,
        ptr,
        volatile: false,
        align: None,
    })
    .with_result(r);
    (r, node)
}

fn store_i64(ptr: ValueId, value: ValueId) -> InstrNode {
    InstrNode::new(Inst::Store {
        ty: Ty::I64,
        ptr,
        value,
        volatile: false,
        align: None,
    })
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

fn xor_i64(vid: &mut VidCounter, lhs: ValueId, rhs: ValueId) -> (ValueId, InstrNode) {
    binop_i64(vid, BinOp::Xor, lhs, rhs)
}

fn mul_i64(vid: &mut VidCounter, lhs: ValueId, rhs: ValueId) -> (ValueId, InstrNode) {
    binop_i64(vid, BinOp::Mul, lhs, rhs)
}

// ---------------------------------------------------------------------------
// Build the tla2_bfs_kernel tMIR function
// ---------------------------------------------------------------------------

fn build_tla2_bfs_kernel() -> (TmirFunction, TmirModule) {
    let mut module = TmirModule::new("tla2_bfs");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::Ptr, Ty::Ptr, Ty::Ptr, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });

    let entry = b(0);
    let mut func = TmirFunction::new(FuncId::new(0), "tla2_bfs_kernel", ft_id, entry);

    // v(0) = state: Ptr
    // v(1) = succ: Ptr
    // v(2) = idx_ptr: Ptr
    // v(3) = len: I64
    let state = v(0);
    let succ = v(1);
    let idx_ptr = v(2);
    let _len = v(3);
    let mut vid = VidCounter::new(4);

    let mut body = Vec::new();

    let (c3, n) = const_i64(&mut vid, 3);
    body.push(n);
    let (c5, n) = const_i64(&mut vid, 5);
    body.push(n);
    let (c0, n) = const_i64(&mut vid, 0);
    body.push(n);
    let (c2, n) = const_i64(&mut vid, 2);
    body.push(n);

    // a = state[3]
    let (state_3_ptr, n) = gep_i64(&mut vid, state, c3);
    body.push(n);
    let (a, n) = load_i64(&mut vid, state_3_ptr);
    body.push(n);

    // b = state[5]
    let (state_5_ptr, n) = gep_i64(&mut vid, state, c5);
    body.push(n);
    let (b_val, n) = load_i64(&mut vid, state_5_ptr);
    body.push(n);

    // succ[3] = a + b
    let (sum_ab, n) = add_i64(&mut vid, a, b_val);
    body.push(n);
    let (succ_3_ptr, n) = gep_i64(&mut vid, succ, c3);
    body.push(n);
    body.push(store_i64(succ_3_ptr, sum_ab));

    // succ[5] = a ^ b
    let (xor_ab, n) = xor_i64(&mut vid, a, b_val);
    body.push(n);
    let (succ_5_ptr, n) = gep_i64(&mut vid, succ, c5);
    body.push(n);
    body.push(store_i64(succ_5_ptr, xor_ab));

    // x = idx_ptr[0]
    let (idx_ptr_0, n) = gep_i64(&mut vid, idx_ptr, c0);
    body.push(n);
    let (x, n) = load_i64(&mut vid, idx_ptr_0);
    body.push(n);

    // y = state[x]
    let (state_x_ptr, n) = gep_i64(&mut vid, state, x);
    body.push(n);
    let (y, n) = load_i64(&mut vid, state_x_ptr);
    body.push(n);

    // succ[x] = y * 2
    let (succ_x_ptr, n) = gep_i64(&mut vid, succ, x);
    body.push(n);
    let (y_times_2, n) = mul_i64(&mut vid, y, c2);
    body.push(n);
    body.push(store_i64(succ_x_ptr, y_times_2));

    // return a + b + y
    let (checksum, n) = add_i64(&mut vid, sum_ab, y);
    body.push(n);
    body.push(InstrNode::new(Inst::Return {
        values: vec![checksum],
    }));

    func.blocks = vec![TmirBlock {
        id: entry,
        params: vec![
            (state, Ty::Ptr),
            (succ, Ty::Ptr),
            (idx_ptr, Ty::Ptr),
            (v(3), Ty::I64),
        ],
        body,
    }];

    module.add_function(func.clone());
    (func, module)
}

const C_DRIVER: &str = r#"
#include <stdint.h>
#include <stdio.h>

extern int64_t tla2_bfs_kernel(int64_t* state, int64_t* succ, int64_t* idx, int64_t len);

int main(void) {
    int64_t state[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    int64_t succ[8] = {0};
    int64_t idx = 7;

    int64_t r = tla2_bfs_kernel(state, succ, &idx, 8);

    if (succ[3] != 100) {
        printf("succ[3]=%lld expected=100\n", (long long)succ[3]);
        return 1;
    }

    if (succ[5] != (40 ^ 60)) {
        printf("succ[5]=%lld expected=%lld\n",
               (long long)succ[5],
               (long long)(40 ^ 60));
        return 2;
    }

    if (succ[7] != 160) {
        printf("succ[7]=%lld expected=160\n", (long long)succ[7]);
        return 3;
    }

    if (r != 180) {
        printf("r=%lld expected=180\n", (long long)r);
        return 4;
    }

    printf("OK\n");
    return 0;
}
"#;

fn run_tla2_bfs_e2e(opt_level: OptLevel, test_name: &str) {
    if !is_aarch64() {
        eprintln!("skipping {}: requires aarch64", test_name);
        return;
    }
    if !has_cc() {
        eprintln!("skipping {}: cc not available", test_name);
        return;
    }

    let dir = make_test_dir(test_name);
    let (func, module) = build_tla2_bfs_kernel();
    let obj_bytes = compile_tmir(&func, &module, opt_level)
        .expect("tla2_bfs_kernel compilation should succeed");
    assert_valid_macho(&obj_bytes, test_name);

    let obj_path = write_object_file(&dir, "tla2_bfs_kernel.o", &obj_bytes);
    let driver_path = write_c_driver(&dir, "driver.c", C_DRIVER);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "tla2_bfs_test");

    let (exit_code, stdout) = run_binary(&binary);
    assert_eq!(exit_code, 0, "binary should exit cleanly; stdout: {}", stdout);
    assert_eq!(stdout.trim(), "OK", "unexpected driver output: {}", stdout);

    cleanup(&dir);
}

// ===========================================================================
// Tests
// ===========================================================================

#[test]
fn test_tla2_bfs_compiles_o0() {
    let (func, module) = build_tla2_bfs_kernel();
    let obj_bytes = compile_tmir(&func, &module, OptLevel::O0)
        .expect("tla2_bfs_kernel should compile at O0");
    assert_valid_macho(&obj_bytes, "tla2_bfs O0");
}

#[test]
fn test_tla2_bfs_compiles_o2() {
    let (func, module) = build_tla2_bfs_kernel();
    let obj_bytes = compile_tmir(&func, &module, OptLevel::O2)
        .expect("tla2_bfs_kernel should compile at O2");
    assert_valid_macho(&obj_bytes, "tla2_bfs O2");
}

#[test]
fn test_tla2_bfs_o0_vs_o2_differ() {
    let (func, module) = build_tla2_bfs_kernel();
    let obj_o0 = compile_tmir(&func, &module, OptLevel::O0)
        .expect("tla2_bfs_kernel should compile at O0");
    let obj_o2 = compile_tmir(&func, &module, OptLevel::O2)
        .expect("tla2_bfs_kernel should compile at O2");
    assert_ne!(
        obj_o0, obj_o2,
        "O0 and O2 should produce different object files"
    );
}

#[test]
fn test_tla2_bfs_e2e_correctness() {
    run_tla2_bfs_e2e(OptLevel::O0, "tla2_bfs_e2e_o0");
}

#[test]
fn test_tla2_bfs_e2e_correctness_o2() {
    run_tla2_bfs_e2e(OptLevel::O2, "tla2_bfs_e2e_o2");
}
