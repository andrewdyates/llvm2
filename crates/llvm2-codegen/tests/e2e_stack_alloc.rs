// llvm2-codegen/tests/e2e_stack_alloc.rs - E2E tests for stack allocation
// through the full tMIR pipeline (Alloc/Store/Load)
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// These tests exercise stack allocation (Alloc/Store/Load) through the
// complete tMIR-to-binary pipeline:
//   1. Build a tMIR module using the builder API with Alloc/Store/Load instrs
//   2. Compile via Compiler::compile()
//   3. Write .o file, link with a C driver, run, and verify output
//
// This catches bugs in the adapter (translate_alloc/translate_store/
// translate_load), ISel (select_stack_addr/select_load/select_store),
// register allocation, frame lowering (stack slot offset resolution),
// and binary encoding of memory instructions.
//
// KNOWN LIMITATION: The adapter's translate_alloc() maps ALL Alloc
// instructions to StackAddr { slot: 0 }, and the ISel resolves slot 0
// to SP+0 which overlaps with the FP/LR save area in the prologue.
// This means:
//   1. Multiple allocs produce the same pointer (slot 0 collision).
//   2. Stores to the slot clobber the saved frame pointer (x29).
// For tests that use alloc+store, the C driver must call the function
// only ONCE and use the return value immediately — making multiple
// calls from the same function will crash due to corrupted FP chain.
// The link+run tests below are structured to work within this limitation.
//
// Part of #243 — Stack allocation E2E tests through full tMIR pipeline.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use llvm2_codegen::compiler::{Compiler, CompilerConfig, CompilerTraceLevel};
use llvm2_codegen::pipeline::OptLevel;
use tmir::{Module as TmirModule, Ty};
use tmir::BinOp;
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
    let dir = std::env::temp_dir().join(format!("llvm2_e2e_stack_alloc_{}", test_name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("Failed to create test directory");
    dir
}

fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

/// Compile a tMIR module through Compiler::compile(), link with a C driver,
/// run, and return (exit_code, stdout).
fn compile_link_run(
    module: &TmirModule,
    test_name: &str,
    driver_src: &str,
) -> (i32, String) {
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        trace_level: CompilerTraceLevel::Full,
        ..CompilerConfig::default()
    });

    let result = compiler
        .compile(module)
        .unwrap_or_else(|e| panic!("{}: compilation failed: {}", test_name, e));

    assert!(
        !result.object_code.is_empty(),
        "{}: compilation must produce non-empty Mach-O bytes",
        test_name
    );

    let test_dir = make_test_dir(test_name);
    let obj_path = test_dir.join(format!("{}.o", test_name));
    fs::write(&obj_path, &result.object_code).expect("write .o file");

    // Disassemble for debugging.
    let otool = Command::new("otool")
        .args(["-tv", obj_path.to_str().unwrap()])
        .output()
        .expect("otool");
    let disasm = String::from_utf8_lossy(&otool.stdout);
    eprintln!("{} disassembly:\n{}", test_name, disasm);

    // Check symbols.
    let nm = Command::new("nm")
        .arg(obj_path.to_str().unwrap())
        .output()
        .expect("nm");
    let nm_stdout = String::from_utf8_lossy(&nm.stdout);
    eprintln!("{} symbols:\n{}", test_name, nm_stdout);

    let driver_path = test_dir.join("driver.c");
    fs::write(&driver_path, driver_src).expect("write C driver");

    let binary_path = test_dir.join(format!("test_{}", test_name));
    let link_output = Command::new("cc")
        .args([
            driver_path.to_str().unwrap(),
            obj_path.to_str().unwrap(),
            "-o",
            binary_path.to_str().unwrap(),
        ])
        .output()
        .expect("cc should be available");

    if !link_output.status.success() {
        let link_stderr = String::from_utf8_lossy(&link_output.stderr);
        cleanup(&test_dir);
        panic!(
            "{}: linking failed.\nLinker stderr: {}\nSymbols:\n{}",
            test_name, link_stderr, nm_stdout
        );
    }

    eprintln!("{}: link succeeded", test_name);

    let run_output = Command::new(binary_path.to_str().unwrap())
        .output()
        .expect("should be able to run the binary");

    let run_stdout = String::from_utf8_lossy(&run_output.stdout).to_string();
    let exit_code = run_output.status.code().unwrap_or(-1);

    eprintln!("{}: stdout={} exit_code={}", test_name, run_stdout.trim(), exit_code);

    cleanup(&test_dir);

    (exit_code, run_stdout)
}

// ---------------------------------------------------------------------------
// Test 1: Simple alloc + store + load (constant)
//
// fn _stack_simple() -> i64 {
//     let slot: *mut i64 = alloc(i64);
//     store(slot, 42);
//     let val = load(slot);
//     return val;
// }
//
// Expected: returns 42.
// Exercises: alloc -> store constant -> load -> return.
// ---------------------------------------------------------------------------

#[test]
fn e2e_stack_alloc_store_load_simple() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e_stack_alloc_store_load_simple: requires AArch64 + cc");
        return;
    }

    let mut mb = ModuleBuilder::new("stack_alloc_simple_test");
    let ty = mb.add_func_type(vec![], vec![Ty::I64]);
    let mut fb = mb.function("_stack_simple", ty);
    let entry = fb.create_block();
    fb.switch_to_block(entry);

    let ptr_val = fb.alloca(Ty::I64);
    let const_42 = fb.iconst(Ty::I64, 42);
    fb.store(Ty::I64, ptr_val, const_42);
    let loaded_val = fb.load(Ty::I64, ptr_val);
    fb.ret(vec![loaded_val]);

    fb.build();
    let module = mb.build();

    // NOTE: Single call from main, result used immediately then exit.
    // See KNOWN LIMITATION in file header re: FP corruption.
    let driver_src = r#"
#include <stdio.h>

extern long _stack_simple(void);

int main() {
    long result = _stack_simple();
    printf("%ld\n", result);
    return (result == 42) ? 0 : 1;
}
"#;

    let (exit_code, stdout) = compile_link_run(&module, "stack_simple", driver_src);

    assert!(
        exit_code == 0,
        "stack_simple() should return 42. Got exit code {}. stdout: {}",
        exit_code,
        stdout
    );
    assert_eq!(
        stdout.trim(),
        "42",
        "stack_simple() should print 42"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Store argument to stack, load it back
//
// fn _stack_arg(a: i64) -> i64 {
//     let slot: *mut i64 = alloc(i64);
//     store(slot, a);
//     let val = load(slot);
//     return val;
// }
//
// Expected: _stack_arg(42) returns 42.
// Exercises: alloc -> store function argument -> load -> return.
// This tests that the store/load path correctly handles values from
// function parameters (not just constants).
// ---------------------------------------------------------------------------

#[test]
fn e2e_stack_alloc_store_load_argument() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e_stack_alloc_store_load_argument: requires AArch64 + cc");
        return;
    }

    let mut mb = ModuleBuilder::new("stack_alloc_arg_test");
    let ty = mb.add_func_type(vec![Ty::I64], vec![Ty::I64]);
    let mut fb = mb.function("_stack_arg", ty);
    let entry = fb.create_block();
    let param0 = fb.add_block_param(entry, Ty::I64);
    fb.switch_to_block(entry);

    let ptr = fb.alloca(Ty::I64);
    fb.store(Ty::I64, ptr, param0);
    let loaded = fb.load(Ty::I64, ptr);
    fb.ret(vec![loaded]);

    fb.build();
    let module = mb.build();

    // Single call, immediate use. See KNOWN LIMITATION re: FP corruption.
    let driver_src = r#"
#include <stdio.h>

extern long _stack_arg(long a);

int main() {
    long result = _stack_arg(42);
    printf("%ld\n", result);
    return (result == 42) ? 0 : 1;
}
"#;

    let (exit_code, stdout) = compile_link_run(&module, "stack_arg", driver_src);

    assert!(
        exit_code == 0,
        "_stack_arg(42) should return 42. Got exit code {}. stdout: {}",
        exit_code,
        stdout
    );
    assert_eq!(
        stdout.trim(),
        "42",
        "_stack_arg(42) should print 42"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Store to stack, compute, load back, combine
//
// fn _stack_add_reload(a: i64, b: i64) -> i64 {
//     let slot: *mut i64 = alloc(i64);
//     let sum = a + b;
//     store(slot, sum);
//     let reloaded = load(slot);
//     return reloaded;
// }
//
// Expected: _stack_add_reload(30, 12) returns 42.
// Exercises: compute -> store result -> load result -> return.
// Tests that computed values survive the stack round-trip.
// ---------------------------------------------------------------------------

#[test]
fn e2e_stack_alloc_compute_store_reload() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e_stack_alloc_compute_store_reload: requires AArch64 + cc");
        return;
    }

    let mut mb = ModuleBuilder::new("stack_alloc_compute_test");
    let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
    let mut fb = mb.function("_stack_add_reload", ty);
    let entry = fb.create_block();
    let param0 = fb.add_block_param(entry, Ty::I64);
    let param1 = fb.add_block_param(entry, Ty::I64);
    fb.switch_to_block(entry);

    let ptr = fb.alloca(Ty::I64);
    let sum = fb.binop(BinOp::Add, Ty::I64, param0, param1);
    fb.store(Ty::I64, ptr, sum);
    let reloaded = fb.load(Ty::I64, ptr);
    fb.ret(vec![reloaded]);

    fb.build();
    let module = mb.build();

    // Single call. See KNOWN LIMITATION re: FP corruption.
    let driver_src = r#"
#include <stdio.h>

extern long _stack_add_reload(long a, long b);

int main() {
    long result = _stack_add_reload(30, 12);
    printf("%ld\n", result);
    return (result == 42) ? 0 : 1;
}
"#;

    let (exit_code, stdout) = compile_link_run(&module, "stack_add_reload", driver_src);

    assert!(
        exit_code == 0,
        "_stack_add_reload(30, 12) should return 42. Got exit code {}. stdout: {}",
        exit_code,
        stdout
    );
    assert_eq!(
        stdout.trim(),
        "42",
        "_stack_add_reload(30, 12) should print 42"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Compilation-only — verifies Alloc/Store/Load compiles at all
// optimization levels. Does not require AArch64 or cc.
// ---------------------------------------------------------------------------

#[test]
fn e2e_stack_alloc_compiles_all_opt_levels() {
    let mut mb = ModuleBuilder::new("stack_alloc_opt_test");
    let ty = mb.add_func_type(vec![], vec![Ty::I64]);
    let mut fb = mb.function("_stack_opt", ty);
    let entry = fb.create_block();
    fb.switch_to_block(entry);

    let ptr = fb.alloca(Ty::I64);
    let const_val = fb.iconst(Ty::I64, 99);
    fb.store(Ty::I64, ptr, const_val);
    let loaded = fb.load(Ty::I64, ptr);
    fb.ret(vec![loaded]);

    fb.build();
    let module = mb.build();

    for opt in &[OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3] {
        let compiler = Compiler::new(CompilerConfig {
            opt_level: *opt,
            ..CompilerConfig::default()
        });

        let result = compiler
            .compile(&module)
            .unwrap_or_else(|e| panic!("compilation at {:?} failed: {}", opt, e));

        assert!(
            !result.object_code.is_empty(),
            "{:?}: stack alloc should produce non-empty object code",
            opt
        );

        // Verify valid Mach-O magic.
        let obj = &result.object_code;
        assert!(
            obj.len() >= 4,
            "{:?}: object code too small ({})",
            opt,
            obj.len()
        );
        let magic = u32::from_le_bytes([obj[0], obj[1], obj[2], obj[3]]);
        assert_eq!(
            magic, 0xFEED_FACF,
            "{:?}: invalid Mach-O magic 0x{:08X}",
            opt, magic
        );

        eprintln!(
            "  {:?}: stack alloc function compiled to {} bytes",
            opt,
            result.metrics.code_size_bytes
        );
    }
}

// ---------------------------------------------------------------------------
// Test 5: Compilation-only — verifies store-overwrite-load pattern
// compiles correctly, exercising multiple stores to the same slot.
// Does not require AArch64 or cc.
// ---------------------------------------------------------------------------

#[test]
fn e2e_stack_alloc_overwrite_compiles() {
    let mut mb = ModuleBuilder::new("stack_overwrite_compile_test");
    let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
    let mut fb = mb.function("_stack_ow", ty);
    let entry = fb.create_block();
    let param0 = fb.add_block_param(entry, Ty::I64);
    let param1 = fb.add_block_param(entry, Ty::I64);
    fb.switch_to_block(entry);

    let ptr = fb.alloca(Ty::I64);
    fb.store(Ty::I64, ptr, param0);
    let va = fb.load(Ty::I64, ptr);
    fb.store(Ty::I64, ptr, param1);
    let vb = fb.load(Ty::I64, ptr);
    let sum = fb.binop(BinOp::Add, Ty::I64, va, vb);
    fb.ret(vec![sum]);

    fb.build();
    let module = mb.build();

    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        trace_level: CompilerTraceLevel::Full,
        ..CompilerConfig::default()
    });

    let result = compiler
        .compile(&module)
        .expect("store-overwrite-load pattern should compile");

    assert!(
        !result.object_code.is_empty(),
        "should produce non-empty object code"
    );
    assert_eq!(result.metrics.function_count, 1);
    assert!(result.metrics.code_size_bytes > 0);

    eprintln!(
        "  store-overwrite-load compiled to {} bytes, {} instructions",
        result.metrics.code_size_bytes, result.metrics.instruction_count
    );
}
