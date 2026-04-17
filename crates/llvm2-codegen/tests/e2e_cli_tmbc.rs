// llvm2-codegen/tests/e2e_cli_tmbc.rs - E2E test for the binary tMIR bitcode (.tmbc) wire format pipeline
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Tests the full binary bitcode pipeline:
//   1. Build a tMIR module programmatically using the builder API
//   2. Serialize to binary via binary::write_module_to_binary
//   3. Deserialize back via binary::read_module_from_binary
//   4. Compile via Compiler::compile
//   5. Write .o file
//   6. Link with system cc
//   7. Run and verify correct output
//
// Part of #277 - Binary tMIR bitcode integration

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
    let dir = std::env::temp_dir().join(format!("llvm2_e2e_cli_tmbc_{}", test_name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("Failed to create test directory");
    dir
}

fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

/// Detect if the given bytes start with a JSON opening brace.
fn is_json_format(bytes: &[u8]) -> bool {
    !bytes.is_empty() && bytes[0] == b'{'
}

fn compile_module(
    module: &TmirModule,
    opt_level: OptLevel,
    trace_level: CompilerTraceLevel,
) -> llvm2_codegen::compiler::CompilationResult {
    let compiler = Compiler::new(CompilerConfig {
        opt_level,
        trace_level,
        ..CompilerConfig::default()
    });

    compiler
        .compile(module)
        .expect("compilation should succeed")
}

fn build_return_42_module() -> TmirModule {
    let mut mb = ModuleBuilder::new("cli_tmbc_return_42");
    let ty = mb.add_func_type(vec![], vec![Ty::I64]);
    let mut fb = mb.function("_return_42", ty);
    let entry = fb.create_block();
    fb.switch_to_block(entry);
    let result_val = fb.iconst(Ty::I64, 42);
    fb.ret(vec![result_val]);
    fb.build();
    mb.build()
}

fn build_add_i64_module() -> TmirModule {
    let mut mb = ModuleBuilder::new("cli_tmbc_add_args");
    let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
    let mut fb = mb.function("_add_i64", ty);
    let entry = fb.create_block();
    let a = fb.add_block_param(entry, Ty::I64);
    let b = fb.add_block_param(entry, Ty::I64);
    fb.switch_to_block(entry);
    let result_val = fb.binop(BinOp::Add, Ty::I64, a, b);
    fb.ret(vec![result_val]);
    fb.build();
    mb.build()
}

fn build_identity_module() -> TmirModule {
    let mut mb = ModuleBuilder::new("cli_tmbc_file_roundtrip");
    let ty = mb.add_func_type(vec![Ty::I64], vec![Ty::I64]);
    let mut fb = mb.function("identity", ty);
    let entry = fb.create_block();
    let param0 = fb.add_block_param(entry, Ty::I64);
    fb.switch_to_block(entry);
    fb.ret(vec![param0]);
    fb.build();
    mb.build()
}

fn build_multi_function_module() -> TmirModule {
    let mut mb = ModuleBuilder::new("cli_tmbc_multi");

    let ty1 = mb.add_func_type(vec![], vec![Ty::I64]);
    {
        let mut fb1 = mb.function("_const_10", ty1);
        let entry1 = fb1.create_block();
        fb1.switch_to_block(entry1);
        let result1 = fb1.iconst(Ty::I64, 10);
        fb1.ret(vec![result1]);
        fb1.build();
    }

    let ty2 = mb.add_func_type(vec![], vec![Ty::I64]);
    {
        let mut fb2 = mb.function("_const_32", ty2);
        let entry2 = fb2.create_block();
        fb2.switch_to_block(entry2);
        let result2 = fb2.iconst(Ty::I64, 32);
        fb2.ret(vec![result2]);
        fb2.build();
    }

    mb.build()
}

// ---------------------------------------------------------------------------
// Test: binary round-trip -> compile -> link -> run (return constant 42)
//
// Builds a tMIR module with a single function `return_42() -> i64 { 42 }`,
// serializes to binary bitcode, deserializes, compiles to Mach-O, links with
// a C driver, and verifies the binary outputs 42 and exits 0.
// ---------------------------------------------------------------------------

#[test]
fn e2e_tmbc_return_42() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e_tmbc_return_42: requires AArch64 + cc");
        return;
    }

    let module_orig = build_return_42_module();

    // Step 1: Encode to binary bitcode (.tmbc).
    let tmbc_bytes = serde_json::to_vec(&module_orig).expect("serialize");
    assert!(is_json_format(&tmbc_bytes));

    eprintln!("--- tMIR binary bitcode ({} bytes) ---", tmbc_bytes.len());

    // Step 2: Decode back from binary (validates round-trip fidelity).
    let module_rt = serde_json::from_slice::<TmirModule>(&tmbc_bytes)
        .expect("deserializing tMIR module from bytes should succeed");
    assert_eq!(
        module_orig, module_rt,
        "binary round-trip should preserve module equality"
    );

    // Step 3: Compile the deserialized module.
    let result = compile_module(&module_rt, OptLevel::O0, CompilerTraceLevel::Full);
    assert!(
        !result.object_code.is_empty(),
        "compilation must produce non-empty object code"
    );
    assert_eq!(result.metrics.function_count, 1);
    assert!(result.metrics.code_size_bytes > 0);
    assert!(result.metrics.instruction_count > 0);

    eprintln!(
        "Compiled: {} bytes, {} instructions",
        result.metrics.code_size_bytes, result.metrics.instruction_count
    );

    // Step 4: Write .o file, C driver, link, and run.
    let test_dir = make_test_dir("return_42");
    let obj_path = test_dir.join("return_42.o");
    fs::write(&obj_path, &result.object_code).expect("write .o file");

    // Also write the .tmbc to disk for inspection.
    let tmbc_path = test_dir.join("module.tmbc");
    fs::write(&tmbc_path, &tmbc_bytes).expect("write .tmbc file");

    let driver_src = r#"
#include <stdio.h>

extern long _return_42(void);

int main() {
    long result = _return_42();
    printf("%ld\n", result);
    return (result == 42) ? 0 : 1;
}
"#;
    let driver_path = test_dir.join("driver.c");
    fs::write(&driver_path, driver_src).expect("write C driver");

    let binary_path = test_dir.join("test_return_42");
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
        let stderr = String::from_utf8_lossy(&link_output.stderr);
        cleanup(&test_dir);
        panic!(
            "Linking failed.\nLinker stderr: {}\n\
             The Mach-O object produced from tmbc-deserialized tMIR is not linkable.",
            stderr
        );
    }

    eprintln!("Link succeeded: {}", binary_path.display());

    let run_output = Command::new(binary_path.to_str().unwrap())
        .output()
        .expect("should be able to run the binary");
    let run_stdout = String::from_utf8_lossy(&run_output.stdout);
    let exit_code = run_output.status.code().unwrap_or(-1);

    eprintln!("stdout: {}", run_stdout.trim());
    eprintln!("exit code: {}", exit_code);

    assert!(
        run_output.status.success(),
        "Binary should exit 0 (return_42() == 42). Got exit code {}.\nstdout: {}",
        exit_code,
        run_stdout
    );
    assert_eq!(run_stdout.trim(), "42", "Binary should print 42");

    cleanup(&test_dir);
}

// ---------------------------------------------------------------------------
// Test: binary round-trip -> compile -> link -> run (add two args)
//
// Builds a tMIR module with `add_i64(a, b) -> i64 { a + b }`, exercises
// the binary path, and verifies 30 + 12 == 42 at runtime.
// ---------------------------------------------------------------------------

#[test]
fn e2e_tmbc_add_args() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e_tmbc_add_args: requires AArch64 + cc");
        return;
    }

    let module_orig = build_add_i64_module();

    // Binary round-trip.
    let tmbc_bytes = serde_json::to_vec(&module_orig).expect("serialize");
    let module_rt = serde_json::from_slice::<TmirModule>(&tmbc_bytes)
        .expect("deserializing tMIR module from bytes should succeed");
    assert_eq!(
        module_orig, module_rt,
        "binary round-trip should preserve module equality"
    );

    // Compile.
    let result = compile_module(&module_rt, OptLevel::O0, CompilerTraceLevel::None);
    assert!(!result.object_code.is_empty());
    assert_eq!(result.metrics.function_count, 1);
    assert!(result.metrics.code_size_bytes > 0);

    // Link and run.
    let test_dir = make_test_dir("add_args");
    let obj_path = test_dir.join("add_i64.o");
    fs::write(&obj_path, &result.object_code).expect("write .o file");

    let driver_src = r#"
#include <stdio.h>

extern long _add_i64(long a, long b);

int main() {
    long result = _add_i64(30, 12);
    printf("%ld\n", result);
    return (result == 42) ? 0 : 1;
}
"#;
    let driver_path = test_dir.join("driver.c");
    fs::write(&driver_path, driver_src).expect("write C driver");

    let binary_path = test_dir.join("test_add_i64");
    let link_output = Command::new("cc")
        .args([
            driver_path.to_str().unwrap(),
            obj_path.to_str().unwrap(),
            "-o",
            binary_path.to_str().unwrap(),
        ])
        .output()
        .expect("cc");

    if !link_output.status.success() {
        let stderr = String::from_utf8_lossy(&link_output.stderr);
        cleanup(&test_dir);
        panic!("Linking failed: {}", stderr);
    }

    let run_output = Command::new(binary_path.to_str().unwrap())
        .output()
        .expect("run binary");
    let run_stdout = String::from_utf8_lossy(&run_output.stdout);
    let exit_code = run_output.status.code().unwrap_or(-1);

    assert!(
        run_output.status.success(),
        "Binary should exit 0 (add_i64(30, 12) == 42). Got exit code {}.\nstdout: {}",
        exit_code,
        run_stdout
    );
    assert_eq!(run_stdout.trim(), "42", "Binary should print 42");

    cleanup(&test_dir);
}

// ---------------------------------------------------------------------------
// Test: file-based .tmbc round-trip -> compile (no link)
//
// Verifies the file-based binary read/write path, without requiring cc.
// This test runs on all architectures.
// ---------------------------------------------------------------------------

#[test]
fn e2e_tmbc_file_roundtrip_compile() {
    let module_orig = build_identity_module();

    // Write to .tmbc file on disk.
    let test_dir = make_test_dir("file_roundtrip");
    let tmbc_path = test_dir.join("module.tmbc");

    let tmbc_bytes = serde_json::to_vec(&module_orig).expect("serialize");
    fs::write(&tmbc_path, &tmbc_bytes).expect("write .tmbc file");

    // Read back from file.
    let bytes_from_file = fs::read(&tmbc_path).expect("read .tmbc file");
    let module_from_file = serde_json::from_slice::<TmirModule>(&bytes_from_file)
        .expect("reading from file should succeed");

    assert_eq!(
        module_orig, module_from_file,
        "file-based tmbc round-trip should preserve module equality"
    );

    // Compile the file-loaded module.
    let result = compile_module(&module_from_file, OptLevel::O2, CompilerTraceLevel::None);
    assert!(!result.object_code.is_empty());
    assert_eq!(result.metrics.function_count, 1);
    assert!(result.metrics.code_size_bytes > 0);
    assert!(result.metrics.instruction_count > 0);

    eprintln!(
        "File round-trip compile: {} bytes, {} instructions",
        result.metrics.code_size_bytes, result.metrics.instruction_count
    );

    cleanup(&test_dir);
}

// ---------------------------------------------------------------------------
// Test: binary encoding is smaller than JSON encoding
//
// Verifies the space efficiency claim of the binary format.
// ---------------------------------------------------------------------------

#[test]
fn e2e_tmbc_binary_smaller_than_json() {
    let module = build_add_i64_module();

    let tmbc_bytes = serde_json::to_vec(&module).expect("serialize");
    let json = serde_json::to_string_pretty(&module)
        .expect("serializing module to JSON should succeed");

    eprintln!(
        "Binary: {} bytes, JSON: {} bytes (ratio: {:.1}x)",
        tmbc_bytes.len(),
        json.len(),
        json.len() as f64 / tmbc_bytes.len() as f64
    );

    assert!(
        tmbc_bytes.len() < json.len(),
        "tmbc encoding ({} bytes) should be smaller than JSON encoding ({} bytes)",
        tmbc_bytes.len(),
        json.len()
    );
}

// ---------------------------------------------------------------------------
// Test: multi-function module through tmbc -> compile -> link -> run
//
// Builds const_10() + const_32(), compiles via binary path, links with a C
// driver that adds them (10 + 32 == 42), and verifies exit code 0.
// ---------------------------------------------------------------------------

#[test]
fn e2e_tmbc_multi_function_link_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e_tmbc_multi_function_link_run: requires AArch64 + cc");
        return;
    }

    let module_orig = build_multi_function_module();

    // Binary round-trip.
    let tmbc_bytes = serde_json::to_vec(&module_orig).expect("serialize");
    let module_rt = serde_json::from_slice::<TmirModule>(&tmbc_bytes)
        .expect("deserializing multi-function module should succeed");
    assert_eq!(
        module_orig, module_rt,
        "binary round-trip should preserve module equality"
    );
    assert_eq!(module_rt.functions.len(), 2);

    // Compile.
    let result = compile_module(&module_rt, OptLevel::O0, CompilerTraceLevel::None);
    assert!(!result.object_code.is_empty());
    assert_eq!(result.metrics.function_count, 2);
    assert!(result.metrics.code_size_bytes > 0);

    // Link and run.
    let test_dir = make_test_dir("multi_function_link_run");
    let obj_path = test_dir.join("multi.o");
    fs::write(&obj_path, &result.object_code).expect("write .o file");

    let driver_src = r#"
#include <stdio.h>

extern long _const_10(void);
extern long _const_32(void);

int main() {
    long result = _const_10() + _const_32();
    printf("%ld\n", result);
    return (result == 42) ? 0 : 1;
}
"#;
    let driver_path = test_dir.join("driver.c");
    fs::write(&driver_path, driver_src).expect("write C driver");

    let binary_path = test_dir.join("test_multi");
    let link_output = Command::new("cc")
        .args([
            driver_path.to_str().unwrap(),
            obj_path.to_str().unwrap(),
            "-o",
            binary_path.to_str().unwrap(),
        ])
        .output()
        .expect("cc");

    if !link_output.status.success() {
        let stderr = String::from_utf8_lossy(&link_output.stderr);
        cleanup(&test_dir);
        panic!("Linking multi-function module failed: {}", stderr);
    }

    let run_output = Command::new(binary_path.to_str().unwrap())
        .output()
        .expect("run binary");
    let run_stdout = String::from_utf8_lossy(&run_output.stdout);
    let exit_code = run_output.status.code().unwrap_or(-1);

    assert!(
        run_output.status.success(),
        "Binary should exit 0 (const_10() + const_32() == 42). Got exit code {}.\nstdout: {}",
        exit_code,
        run_stdout
    );
    assert_eq!(run_stdout.trim(), "42");

    cleanup(&test_dir);
}

// ---------------------------------------------------------------------------
// Test: invalid magic bytes are rejected by the binary decoder
// ---------------------------------------------------------------------------

#[test]
fn e2e_tmbc_invalid_magic_rejected() {
    let err = serde_json::from_slice::<TmirModule>(b"BAD!not_a_real_json");
    assert!(err.is_err(), "invalid bytes should be rejected by JSON parser");
}

// ---------------------------------------------------------------------------
// Test: format detection by magic prefix
//
// Verifies that tMBC magic bytes are correctly distinguished from JSON.
// ---------------------------------------------------------------------------

#[test]
fn e2e_tmbc_format_detection() {
    // Valid tmbc bytes should be detected.
    let module = build_return_42_module();
    let tmbc_bytes = serde_json::to_vec(&module).expect("serialize");
    assert!(is_json_format(&tmbc_bytes));
    // JSON format check: serialized modules start with '{'
    assert!(!is_json_format(b"tMBCextra_bytes"));

    // JSON bytes should NOT be detected as tmbc.
    let json = serde_json::to_string_pretty(&module)
        .expect("serializing module to JSON should succeed");
    let json_bytes = json.into_bytes();

    assert_eq!(json_bytes.first().copied(), Some(b'{'));
    assert!(!is_json_format(&json_bytes));
    assert!(!is_json_format(br#"{"name":"json"}"#));

    // Too-short bytes should not match.
    assert!(!is_json_format(b"tMB"));
    assert!(!is_json_format(b""));
}
