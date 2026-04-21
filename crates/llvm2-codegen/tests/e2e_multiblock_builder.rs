// llvm2-codegen/tests/e2e_multiblock_builder.rs
//
// Multi-block E2E tests using the tMIR builder API.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// These tests exercise multi-block control flow (if/else, loops, diamond CFG)
// through the FULL compilation pipeline using the tMIR builder API:
//   ModuleBuilder -> Compiler::compile() -> Mach-O .o -> cc link -> run
//
// Unlike e2e_aarch64_link.rs which constructs tMIR with raw struct literals,
// these tests use the ergonomic builder API from builder, proving
// that the builder correctly produces multi-block programs that compile and
// run correctly end-to-end.
//
// Part of #242 -- Multi-block E2E tests for control flow

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use llvm2_codegen::compiler::{Compiler, CompilerConfig, CompilerTraceLevel};
use llvm2_codegen::pipeline::OptLevel;
use tmir::{Module as TmirModule, Ty};
use tmir::{BinOp, ICmpOp};
use tmir_build::ModuleBuilder;


// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

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
    let dir = std::env::temp_dir().join(format!("llvm2_e2e_mb_{}", test_name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create test dir");
    dir
}

fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

/// Compile a tMIR module through the full Compiler::compile() pipeline.
fn compile_module(module: &TmirModule) -> Vec<u8> {
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        trace_level: CompilerTraceLevel::Full,
        ..CompilerConfig::default()
    });
    let result = compiler
        .compile(module)
        .expect("compilation should succeed");
    assert!(
        !result.object_code.is_empty(),
        "compiled object code must be non-empty"
    );
    // Verify valid Mach-O magic
    let obj = &result.object_code;
    assert!(obj.len() >= 4, "object code too small for Mach-O header");
    let magic = u32::from_le_bytes([obj[0], obj[1], obj[2], obj[3]]);
    assert_eq!(
        magic, 0xFEED_FACF,
        "expected Mach-O 64-bit magic, got {:#010X}",
        magic
    );
    result.object_code
}

/// Link an object file with a C driver and run the resulting binary.
/// Returns (exit_code, stdout).
fn link_and_run(dir: &Path, obj_bytes: &[u8], obj_name: &str, driver_src: &str) -> (i32, String) {
    let obj_path = dir.join(format!("{}.o", obj_name));
    fs::write(&obj_path, obj_bytes).expect("write .o");

    let driver_path = dir.join("driver.c");
    fs::write(&driver_path, driver_src).expect("write driver.c");

    let binary_path = dir.join(format!("test_{}", obj_name));

    let link_out = Command::new("cc")
        .args([
            driver_path.to_str().unwrap(),
            obj_path.to_str().unwrap(),
            "-o",
            binary_path.to_str().unwrap(),
        ])
        .output()
        .expect("cc should be available");

    if !link_out.status.success() {
        let stderr = String::from_utf8_lossy(&link_out.stderr);

        // Debug: disassemble and show symbols
        let otool = Command::new("otool")
            .args(["-tv", obj_path.to_str().unwrap()])
            .output()
            .ok();
        let nm = Command::new("nm")
            .args([obj_path.to_str().unwrap()])
            .output()
            .ok();

        let disasm = otool
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();
        let symbols = nm
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();

        panic!(
            "Linking failed for {}!\nstderr: {}\notool:\n{}\nnm:\n{}",
            obj_name, stderr, disasm, symbols
        );
    }

    let run_out = Command::new(binary_path.to_str().unwrap())
        .output()
        .expect("run binary");

    let stdout = String::from_utf8_lossy(&run_out.stdout).to_string();
    let exit_code = run_out.status.code().unwrap_or(-1);

    (exit_code, stdout)
}

// ---------------------------------------------------------------------------
// Test 1: if/else -- max(a, b) using builder API
//
// fn _max_builder(a: i64, b: i64) -> i64 {
//     if a > b { return a } else { return b }
// }
//
// bb0 (entry): cmp a > b, condbr -> bb1 (return a), bb2 (return b)
// bb1: return a
// bb2: return b
// ---------------------------------------------------------------------------

fn build_max_module() -> TmirModule {
    let mut mb = ModuleBuilder::new("e2e_max_builder_test");
    let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
    let mut fb = mb.function("_max_builder", ty);

    let entry = fb.create_block();
    let a = fb.add_block_param(entry, Ty::I64);
    let b = fb.add_block_param(entry, Ty::I64);
    let bb_then = fb.create_block();
    let bb_else = fb.create_block();

    fb.switch_to_block(entry);
    let cmp_result = fb.icmp(ICmpOp::Sgt, Ty::I64, a, b);
    fb.condbr(cmp_result, bb_then, vec![], bb_else, vec![]);

    fb.switch_to_block(bb_then);
    fb.ret(vec![a]);

    fb.switch_to_block(bb_else);
    fb.ret(vec![b]);

    fb.build();
    mb.build()
}

#[test]
fn e2e_multiblock_max_compile() {
    let module = build_max_module();
    let obj_bytes = compile_module(&module);

    // Multi-block: should produce more than a trivial single-block function
    assert!(
        obj_bytes.len() > 100,
        "multi-block max function should produce substantial object code"
    );
}

#[test]
fn e2e_multiblock_max_link_and_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("max_builder");
    let module = build_max_module();
    let obj_bytes = compile_module(&module);

    let driver = r#"
#include <stdio.h>

extern long _max_builder(long a, long b);

int main(void) {
    long r1 = _max_builder(10, 20);
    long r2 = _max_builder(20, 10);
    long r3 = _max_builder(5, 5);
    long r4 = _max_builder(-3, -7);
    long r5 = _max_builder(-1, 1);
    printf("max(10,20)=%ld max(20,10)=%ld max(5,5)=%ld max(-3,-7)=%ld max(-1,1)=%ld\n",
           r1, r2, r3, r4, r5);
    if (r1 != 20) return 1;
    if (r2 != 20) return 2;
    if (r3 != 5)  return 3;
    if (r4 != -3) return 4;
    if (r5 != 1)  return 5;
    return 0;
}
"#;

    let (exit_code, stdout) = link_and_run(&dir, &obj_bytes, "max_builder", driver);
    eprintln!("max_builder stdout: {}", stdout.trim());
    assert_eq!(
        exit_code, 0,
        "max_builder failed (exit {}). \
         1=max(10,20)!=20, 2=max(20,10)!=20, 3=max(5,5)!=5, 4=max(-3,-7)!=-3, 5=max(-1,1)!=1. \
         stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test 2: loop -- sum_to(n) using builder API
//
// fn _sum_to_builder(n: i64) -> i64 {
//     sum = 0; i = 1
//     while i <= n { sum += i; i += 1 }
//     return sum
// }
//
// bb0 (entry): sum=0, i=1, br -> bb1
// bb1 (loop header): params(sum, i), cmp i <= n, condbr -> bb2 (body), bb3 (exit)
// bb2 (body): new_sum = sum + i, new_i = i + 1, br -> bb1(new_sum, new_i)
// bb3 (exit): return sum
// ---------------------------------------------------------------------------

fn build_sum_to_module() -> TmirModule {
    let mut mb = ModuleBuilder::new("e2e_sum_to_builder_test");
    let ty = mb.add_func_type(vec![Ty::I64], vec![Ty::I64]);
    let mut fb = mb.function("_sum_to_builder", ty);

    let entry = fb.create_block();
    let n = fb.add_block_param(entry, Ty::I64);
    let bb_loop = fb.create_block();
    let loop_sum = fb.add_block_param(bb_loop, Ty::I64);
    let loop_i = fb.add_block_param(bb_loop, Ty::I64);
    let bb_body = fb.create_block();
    let bb_exit = fb.create_block();

    // bb0 (entry): init sum=0, i=1, jump to loop header
    fb.switch_to_block(entry);
    let sum_init = fb.iconst(Ty::I64, 0);
    let i_init = fb.iconst(Ty::I64, 1);
    fb.br(bb_loop, vec![sum_init, i_init]);

    // bb1 (loop header): check i <= n
    fb.switch_to_block(bb_loop);
    let cmp_val = fb.icmp(ICmpOp::Sle, Ty::I64, loop_i, n);
    fb.condbr(cmp_val, bb_body, vec![], bb_exit, vec![]);

    // bb2 (body): sum += i, i += 1, back to loop
    fb.switch_to_block(bb_body);
    let new_sum = fb.binop(BinOp::Add, Ty::I64, loop_sum, loop_i);
    let one_val = fb.iconst(Ty::I64, 1);
    let new_i = fb.binop(BinOp::Add, Ty::I64, loop_i, one_val);
    fb.br(bb_loop, vec![new_sum, new_i]);

    // bb3 (exit): return sum
    fb.switch_to_block(bb_exit);
    fb.ret(vec![loop_sum]);

    fb.build();
    mb.build()
}

#[test]
fn e2e_multiblock_sum_to_compile() {
    let module = build_sum_to_module();
    let obj_bytes = compile_module(&module);

    assert!(
        obj_bytes.len() > 100,
        "multi-block sum_to function should produce substantial object code"
    );
}

#[test]
fn e2e_multiblock_sum_to_link_and_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("sum_to_builder");
    let module = build_sum_to_module();
    let obj_bytes = compile_module(&module);

    // sum(1..=n) = n*(n+1)/2
    let driver = r#"
#include <stdio.h>

extern long _sum_to_builder(long n);

int main(void) {
    long r0  = _sum_to_builder(0);    /* 0 */
    long r1  = _sum_to_builder(1);    /* 1 */
    long r5  = _sum_to_builder(5);    /* 15 */
    long r10 = _sum_to_builder(10);   /* 55 */
    long r100 = _sum_to_builder(100); /* 5050 */
    printf("sum(0)=%ld sum(1)=%ld sum(5)=%ld sum(10)=%ld sum(100)=%ld\n",
           r0, r1, r5, r10, r100);
    if (r0 != 0)     return 1;
    if (r1 != 1)     return 2;
    if (r5 != 15)    return 3;
    if (r10 != 55)   return 4;
    if (r100 != 5050) return 5;
    return 0;
}
"#;

    let (exit_code, stdout) = link_and_run(&dir, &obj_bytes, "sum_to_builder", driver);
    eprintln!("sum_to_builder stdout: {}", stdout.trim());
    assert_eq!(
        exit_code, 0,
        "sum_to_builder failed (exit {}). \
         1=sum(0)!=0, 2=sum(1)!=1, 3=sum(5)!=15, 4=sum(10)!=55, 5=sum(100)!=5050. \
         stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test 3: diamond CFG -- clamp(x, lo, hi) using builder API
//
// fn _clamp_builder(x: i64, lo: i64, hi: i64) -> i64 {
//     if x < lo { return lo }
//     else if x > hi { return hi }
//     else { return x }
// }
//
// bb0 (entry): cmp x < lo, condbr -> bb1 (ret lo), bb2 (check hi)
// bb1: return lo
// bb2: cmp x > hi, condbr -> bb3 (ret hi), bb4 (ret x)
// bb3: return hi
// bb4: return x
// ---------------------------------------------------------------------------

fn build_clamp_module() -> TmirModule {
    let mut mb = ModuleBuilder::new("e2e_clamp_builder_test");
    let ty = mb.add_func_type(vec![Ty::I64, Ty::I64, Ty::I64], vec![Ty::I64]);
    let mut fb = mb.function("_clamp_builder", ty);

    let entry = fb.create_block();
    let x = fb.add_block_param(entry, Ty::I64);
    let lo = fb.add_block_param(entry, Ty::I64);
    let hi = fb.add_block_param(entry, Ty::I64);
    let bb_ret_lo = fb.create_block();
    let bb_check_hi = fb.create_block();
    let bb_ret_hi = fb.create_block();
    let bb_ret_x = fb.create_block();

    // bb0 (entry): check x < lo
    fb.switch_to_block(entry);
    let cmp_lo = fb.icmp(ICmpOp::Slt, Ty::I64, x, lo);
    fb.condbr(cmp_lo, bb_ret_lo, vec![], bb_check_hi, vec![]);

    // bb1: return lo
    fb.switch_to_block(bb_ret_lo);
    fb.ret(vec![lo]);

    // bb2: check x > hi
    fb.switch_to_block(bb_check_hi);
    let cmp_hi = fb.icmp(ICmpOp::Sgt, Ty::I64, x, hi);
    fb.condbr(cmp_hi, bb_ret_hi, vec![], bb_ret_x, vec![]);

    // bb3: return hi
    fb.switch_to_block(bb_ret_hi);
    fb.ret(vec![hi]);

    // bb4: return x
    fb.switch_to_block(bb_ret_x);
    fb.ret(vec![x]);

    fb.build();
    mb.build()
}

#[test]
fn e2e_multiblock_clamp_compile() {
    let module = build_clamp_module();
    let obj_bytes = compile_module(&module);

    assert!(
        obj_bytes.len() > 100,
        "multi-block clamp function should produce substantial object code"
    );
}

#[test]
fn e2e_multiblock_clamp_link_and_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("clamp_builder");
    let module = build_clamp_module();
    let obj_bytes = compile_module(&module);

    let driver = r#"
#include <stdio.h>

extern long _clamp_builder(long x, long lo, long hi);

int main(void) {
    long r1 = _clamp_builder(5, 0, 10);    /* 5 -- in range */
    long r2 = _clamp_builder(-5, 0, 10);   /* 0 -- below lo */
    long r3 = _clamp_builder(15, 0, 10);   /* 10 -- above hi */
    long r4 = _clamp_builder(0, 0, 10);    /* 0 -- at lo boundary */
    long r5 = _clamp_builder(10, 0, 10);   /* 10 -- at hi boundary */
    long r6 = _clamp_builder(-100, -50, 50); /* -50 -- below lo */
    long r7 = _clamp_builder(100, -50, 50);  /* 50 -- above hi */
    printf("clamp(5,0,10)=%ld clamp(-5,0,10)=%ld clamp(15,0,10)=%ld "
           "clamp(0,0,10)=%ld clamp(10,0,10)=%ld "
           "clamp(-100,-50,50)=%ld clamp(100,-50,50)=%ld\n",
           r1, r2, r3, r4, r5, r6, r7);
    if (r1 != 5)   return 1;
    if (r2 != 0)   return 2;
    if (r3 != 10)  return 3;
    if (r4 != 0)   return 4;
    if (r5 != 10)  return 5;
    if (r6 != -50) return 6;
    if (r7 != 50)  return 7;
    return 0;
}
"#;

    let (exit_code, stdout) = link_and_run(&dir, &obj_bytes, "clamp_builder", driver);
    eprintln!("clamp_builder stdout: {}", stdout.trim());
    assert_eq!(
        exit_code, 0,
        "clamp_builder failed (exit {}). \
         1=in-range, 2=below-lo, 3=above-hi, 4=at-lo, 5=at-hi, 6=neg-below, 7=neg-above. \
         stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test 4: all multi-block builder tests at multiple optimization levels
// ---------------------------------------------------------------------------

#[test]
fn e2e_multiblock_builder_all_opt_levels() {
    let modules: &[(&str, TmirModule)] = &[
        ("max_builder", build_max_module()),
        ("sum_to_builder", build_sum_to_module()),
        ("clamp_builder", build_clamp_module()),
    ];

    for (name, module) in modules {
        for opt in &[OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3] {
            let compiler = Compiler::new(CompilerConfig {
                opt_level: *opt,
                ..CompilerConfig::default()
            });
            let result = compiler.compile(module).unwrap_or_else(|e| {
                panic!("{} at {:?} failed: {}", name, opt, e)
            });

            assert!(
                !result.object_code.is_empty(),
                "{} at {:?} produced empty object",
                name, opt
            );

            let obj = &result.object_code;
            let magic = u32::from_le_bytes([obj[0], obj[1], obj[2], obj[3]]);
            assert_eq!(
                magic, 0xFEED_FACF,
                "{} at {:?} produced invalid Mach-O",
                name, opt
            );

            eprintln!(
                "  {} {:?}: {} bytes, {} instructions",
                name, opt, result.metrics.code_size_bytes, result.metrics.instruction_count
            );
        }
    }
}
