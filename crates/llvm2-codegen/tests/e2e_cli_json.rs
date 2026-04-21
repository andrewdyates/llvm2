// llvm2-codegen/tests/e2e_cli_json.rs - E2E test for the CLI tMIR-to-binary JSON wire format
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Tests the full JSON wire format pipeline:
//   1. Build a tMIR module programmatically using the builder API
//   2. Serialize to JSON via reader::write_module_to_string()
//   3. Deserialize back via reader::read_module_from_str()
//   4. Compile via Compiler::compile()
//   5. Write .o file
//   6. Link with system cc
//   7. Run and verify correct output
//
// This exercises the same path the CLI uses when invoked with --input-json,
// ensuring the JSON wire format is a faithful representation of tMIR modules.
//
// Part of #256 — E2E test for the CLI tMIR-to-binary pipeline.

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
    let dir = std::env::temp_dir().join(format!("llvm2_e2e_cli_json_{}", test_name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("Failed to create test directory");
    dir
}

fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------------------
// Test: JSON round-trip -> compile -> link -> run (return constant 42)
//
// Builds a tMIR module with a single function `return_42() -> i64 { 42 }`,
// serializes to JSON, deserializes, compiles to Mach-O, links with a C
// driver, and verifies the binary outputs 42 and exits 0.
// ---------------------------------------------------------------------------

#[test]
fn e2e_json_wire_format_return_42() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e_json_wire_format_return_42: requires AArch64 + cc");
        return;
    }

    // Step 1: Build a tMIR module using the builder API.
    let mut mb = ModuleBuilder::new("cli_e2e_test");
    // Function names use leading underscore per Mach-O convention:
    // tMIR name "_return_42" -> Mach-O symbol "__return_42" -> C sees "_return_42"
    let ty = mb.add_func_type(vec![], vec![Ty::I64]);
    let mut fb = mb.function("_return_42", ty);
    let entry = fb.create_block();
    fb.switch_to_block(entry);
    let result_val = fb.iconst(Ty::I64, 42);
    fb.ret(vec![result_val]);
    fb.build();
    let module_orig = mb.build();

    // Step 2: Serialize to JSON (the wire format).
    let json_str = serde_json::to_string_pretty(&module_orig)
        .expect("serializing tMIR module to JSON should succeed");

    eprintln!("--- tMIR JSON wire format ({} bytes) ---", json_str.len());
    eprintln!("{}", &json_str[..json_str.len().min(500)]);
    if json_str.len() > 500 {
        eprintln!("... (truncated)");
    }
    eprintln!("--- end JSON ---");

    // Step 3: Deserialize back from JSON (validates round-trip fidelity).
    let module_rt = serde_json::from_str::<TmirModule>(&json_str)
        .expect("deserializing tMIR module from JSON should succeed");
    assert_eq!(
        module_orig, module_rt,
        "JSON round-trip should preserve module equality"
    );

    // Step 4: Compile the deserialized module via the Compiler API.
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        trace_level: CompilerTraceLevel::Full,
        ..CompilerConfig::default()
    });
    let result = compiler
        .compile(&module_rt)
        .expect("compiling JSON-deserialized tMIR module should succeed");

    assert!(
        !result.object_code.is_empty(),
        "compilation must produce non-empty Mach-O bytes"
    );
    assert_eq!(result.metrics.function_count, 1);
    assert!(result.metrics.code_size_bytes > 0);

    eprintln!(
        "Compiled: {} bytes, {} instructions",
        result.metrics.code_size_bytes, result.metrics.instruction_count
    );

    // Step 5: Write .o file, C driver, link, and run.
    let test_dir = make_test_dir("return_42");
    let obj_path = test_dir.join("return_42.o");
    fs::write(&obj_path, &result.object_code).expect("write .o file");

    // Also write the JSON to disk (as the CLI would read it).
    let json_path = test_dir.join("module.json");
    fs::write(&json_path, &json_str).expect("write JSON file");

    // Verify JSON file can be read back from disk too.
    let json_from_file = fs::read_to_string(&json_path).expect("read JSON file");
    let module_from_file: TmirModule = serde_json::from_str(&json_from_file)
        .expect("reading JSON from file should succeed");
    assert_eq!(module_orig, module_from_file);

    // C driver that calls our compiled function and checks the return value.
    let driver_src = r#"
#include <stdio.h>

// LLVM2 emits Mach-O symbols with leading underscore: __return_42
// C sees this as _return_42 (the C ABI adds one underscore).
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

    let link_stderr = String::from_utf8_lossy(&link_output.stderr);
    if !link_output.status.success() {
        // Diagnostic: dump symbols and disassembly on link failure
        let nm_out = Command::new("nm")
            .arg(obj_path.to_str().unwrap())
            .output()
            .expect("nm");
        let nm_stdout = String::from_utf8_lossy(&nm_out.stdout);
        eprintln!("nm symbols:\n{}", nm_stdout);

        let otool_out = Command::new("otool")
            .args(["-tv", obj_path.to_str().unwrap()])
            .output()
            .expect("otool");
        let otool_stdout = String::from_utf8_lossy(&otool_out.stdout);
        eprintln!("otool disassembly:\n{}", otool_stdout);

        cleanup(&test_dir);
        panic!(
            "Linking failed.\nLinker stderr: {}\n\
             The Mach-O object produced from JSON-deserialized tMIR is not linkable.",
            link_stderr
        );
    }

    eprintln!("Link succeeded: {}", binary_path.display());

    // Run the binary.
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
    assert_eq!(
        run_stdout.trim(),
        "42",
        "Binary should print 42"
    );

    cleanup(&test_dir);
}

// ---------------------------------------------------------------------------
// Test: JSON round-trip -> compile -> link -> run (add two args)
//
// Builds a tMIR module with `add_i64(a, b) -> i64 { a + b }`, exercises
// the JSON path, and verifies 30 + 12 == 42 at runtime.
// ---------------------------------------------------------------------------

#[test]
fn e2e_json_wire_format_add_args() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e_json_wire_format_add_args: requires AArch64 + cc");
        return;
    }

    // Build module: fn _add_i64(a: i64, b: i64) -> i64 { a + b }
    // Leading underscore for Mach-O convention.
    let mut mb = ModuleBuilder::new("cli_e2e_add");
    let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
    let mut fb = mb.function("_add_i64", ty);
    let entry = fb.create_block();
    let a = fb.add_block_param(entry, Ty::I64);
    let b = fb.add_block_param(entry, Ty::I64);
    fb.switch_to_block(entry);
    let result_val = fb.binop(BinOp::Add, Ty::I64, a, b);
    fb.ret(vec![result_val]);
    fb.build();
    let module_orig = mb.build();

    // JSON round-trip.
    let json_str = serde_json::to_string_pretty(&module_orig)
        .expect("JSON serialization should succeed");
    let module_rt = serde_json::from_str::<TmirModule>(&json_str)
        .expect("JSON deserialization should succeed");
    assert_eq!(module_orig, module_rt, "JSON round-trip must preserve equality");

    // Compile.
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        ..CompilerConfig::default()
    });
    let result = compiler
        .compile(&module_rt)
        .expect("compilation should succeed");
    assert!(!result.object_code.is_empty());

    // Link and run.
    let test_dir = make_test_dir("add_args");
    let obj_path = test_dir.join("add_i64.o");
    fs::write(&obj_path, &result.object_code).expect("write .o");

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
// Test: JSON file I/O round-trip -> compile (no link)
//
// Verifies the file-based JSON read/write path that the CLI uses,
// without requiring cc. This test runs on all architectures.
// ---------------------------------------------------------------------------

#[test]
fn e2e_json_file_roundtrip_compile() {
    // Build a simple tMIR module.
    let mut mb = ModuleBuilder::new("file_roundtrip_test");
    let ty = mb.add_func_type(vec![Ty::I64], vec![Ty::I64]);
    let mut fb = mb.function("identity", ty);
    let entry = fb.create_block();
    let param0 = fb.add_block_param(entry, Ty::I64);
    fb.switch_to_block(entry);
    fb.ret(vec![param0]);
    fb.build();
    let module_orig = mb.build();

    // Write to JSON file on disk (as external tools would produce).
    let test_dir = make_test_dir("file_roundtrip");
    let json_path = test_dir.join("module.json");
    let json_str_file = serde_json::to_string_pretty(&module_orig)
        .expect("serialize JSON should succeed");
    fs::write(&json_path, &json_str_file).expect("write JSON to file should succeed");

    // Read back from file (as the CLI does with --input-json).
    let json_from_file2 = fs::read_to_string(&json_path).expect("read JSON file");
    let module_from_file: TmirModule = serde_json::from_str(&json_from_file2)
        .expect("read JSON from file should succeed");
    assert_eq!(
        module_orig, module_from_file,
        "file-based JSON round-trip must preserve module equality"
    );

    // Verify the JSON is valid serde_json (raw deserialization without validation).
    let json_bytes = fs::read(&json_path).expect("read JSON bytes");
    let module_raw: TmirModule =
        serde_json::from_slice(&json_bytes).expect("raw serde_json deserialization should work");
    assert_eq!(module_orig, module_raw);

    // Compile the file-loaded module.
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O2,
        ..CompilerConfig::default()
    });
    let result = compiler
        .compile(&module_from_file)
        .expect("compilation should succeed");

    assert!(!result.object_code.is_empty());
    assert_eq!(result.metrics.function_count, 1);

    eprintln!(
        "File round-trip compile: {} bytes, {} instructions",
        result.metrics.code_size_bytes, result.metrics.instruction_count
    );

    cleanup(&test_dir);
}

// ---------------------------------------------------------------------------
// Test: Multi-function module through JSON wire format
//
// Builds a module with two functions, serializes/deserializes via JSON,
// compiles both, and verifies the object contains symbols for each.
// ---------------------------------------------------------------------------

#[test]
fn e2e_json_multi_function_module() {
    // Build module with two functions:
    //   fn const_10() -> i64 { 10 }
    //   fn const_32() -> i64 { 32 }
    let mut mb = ModuleBuilder::new("multi_func_json");

    // Leading underscore for Mach-O convention.
    let ty1 = mb.add_func_type(vec![], vec![Ty::I64]);
    {
        let mut fb1 = mb.function("_const_10", ty1);
        let entry1 = fb1.create_block();
        fb1.switch_to_block(entry1);
        let r1 = fb1.iconst(Ty::I64, 10);
        fb1.ret(vec![r1]);
        fb1.build();
    }

    let ty2 = mb.add_func_type(vec![], vec![Ty::I64]);
    {
        let mut fb2 = mb.function("_const_32", ty2);
        let entry2 = fb2.create_block();
        fb2.switch_to_block(entry2);
        let r2 = fb2.iconst(Ty::I64, 32);
        fb2.ret(vec![r2]);
        fb2.build();
    }

    let module_orig = mb.build();

    // JSON round-trip.
    let json_str = serde_json::to_string_pretty(&module_orig)
        .expect("JSON serialization should succeed");
    let module_rt: TmirModule = serde_json::from_str(&json_str)
        .expect("JSON deserialization should succeed");
    assert_eq!(module_orig, module_rt);
    assert_eq!(module_rt.functions.len(), 2);

    // Compile.
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        ..CompilerConfig::default()
    });
    let result = compiler
        .compile(&module_rt)
        .expect("multi-function module compilation should succeed");

    assert!(!result.object_code.is_empty());
    assert_eq!(result.metrics.function_count, 2);

    // On AArch64, verify both symbols appear via nm.
    if is_aarch64() {
        let test_dir = make_test_dir("multi_func");
        let obj_path = test_dir.join("multi.o");
        fs::write(&obj_path, &result.object_code).expect("write .o");

        let nm_output = Command::new("nm")
            .arg(obj_path.to_str().unwrap())
            .output()
            .expect("nm should be available");
        let nm_stdout = String::from_utf8_lossy(&nm_output.stdout);

        eprintln!("nm output:\n{}", nm_stdout);

        assert!(
            nm_stdout.contains("const_10"),
            "nm should show const_10 symbol. Got:\n{}",
            nm_stdout
        );
        assert!(
            nm_stdout.contains("const_32"),
            "nm should show const_32 symbol. Got:\n{}",
            nm_stdout
        );

        cleanup(&test_dir);
    }

    eprintln!(
        "Multi-function JSON E2E: {} functions, {} bytes",
        result.metrics.function_count, result.metrics.code_size_bytes
    );
}

// ---------------------------------------------------------------------------
// Test: Multi-function module through JSON -> link -> run
//
// Builds const_10() + const_32(), compiles via JSON path, links with a C
// driver that adds them (10 + 32 == 42), and verifies exit code 0.
// ---------------------------------------------------------------------------

#[test]
fn e2e_json_multi_function_link_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e_json_multi_function_link_run: requires AArch64 + cc");
        return;
    }

    let mut mb = ModuleBuilder::new("multi_link");

    // Leading underscore for Mach-O convention.
    let ty1 = mb.add_func_type(vec![], vec![Ty::I64]);
    {
        let mut fb1 = mb.function("_const_10", ty1);
        let entry1 = fb1.create_block();
        fb1.switch_to_block(entry1);
        let r1 = fb1.iconst(Ty::I64, 10);
        fb1.ret(vec![r1]);
        fb1.build();
    }

    let ty2 = mb.add_func_type(vec![], vec![Ty::I64]);
    {
        let mut fb2 = mb.function("_const_32", ty2);
        let entry2 = fb2.create_block();
        fb2.switch_to_block(entry2);
        let r2 = fb2.iconst(Ty::I64, 32);
        fb2.ret(vec![r2]);
        fb2.build();
    }

    let module_orig = mb.build();

    // JSON round-trip.
    let json_str = serde_json::to_string_pretty(&module_orig).unwrap();
    let module_rt: TmirModule = serde_json::from_str(&json_str).unwrap();

    // Compile.
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        ..CompilerConfig::default()
    });
    let result = compiler.compile(&module_rt).expect("compilation should succeed");

    // Link and run.
    let test_dir = make_test_dir("multi_link_run");
    let obj_path = test_dir.join("multi.o");
    fs::write(&obj_path, &result.object_code).expect("write .o");

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
