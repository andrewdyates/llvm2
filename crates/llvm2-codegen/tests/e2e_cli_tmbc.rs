// llvm2-codegen/tests/e2e_cli_tmbc.rs - E2E test for the tMIR serialization wire format pipeline
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Tests the full serialization pipeline:
//   1. Build a tMIR module programmatically using the builder API
//   2. Serialize to JSON via serde_json
//   3. Deserialize back via serde_json
//   4. Compile via Compiler::compile
//   5. Write .o file
//   6. Link with system cc
//   7. Run and verify correct output
//
// Originally tested binary tMBC format from tmir_func::binary. Since migration
// to the real tmir crate (which uses serde derives), these tests now exercise
// the serde_json serialization path which serves the same purpose: verifying
// that tMIR modules survive serialization round-trips faithfully.
//
// Part of #277 - Serialization integration

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use llvm2_codegen::compiler::{Compiler, CompilerConfig, CompilerTraceLevel};
use llvm2_codegen::pipeline::OptLevel;

use tmir::Ty;
use tmir_build::builder::ModuleBuilder;

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

fn compile_module(
    module: &tmir::Module,
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

fn build_return_42_module() -> tmir::Module {
    let mut mb = ModuleBuilder::new("cli_tmbc_return_42");
    let ft = mb.add_func_type(vec![], vec![Ty::I64]);
    let mut fb = mb.function("_return_42", ft);

    let entry = fb.create_block();
    fb.set_entry(entry);
    fb.switch_to_block(entry);
    let result_val = fb.iconst(Ty::I64, 42);
    fb.ret(vec![result_val]);

    fb.build();
    mb.build()
}

fn build_add_i64_module() -> tmir::Module {
    let mut mb = ModuleBuilder::new("cli_tmbc_add_args");
    let ft = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
    let mut fb = mb.function("_add_i64", ft);

    let entry = fb.create_block();
    let a = fb.add_block_param(entry, Ty::I64);
    let b = fb.add_block_param(entry, Ty::I64);
    fb.set_entry(entry);

    fb.switch_to_block(entry);
    let result_val = fb.add(Ty::I64, a, b);
    fb.ret(vec![result_val]);

    fb.build();
    mb.build()
}

fn build_identity_module() -> tmir::Module {
    let mut mb = ModuleBuilder::new("cli_tmbc_file_roundtrip");
    let ft = mb.add_func_type(vec![Ty::I64], vec![Ty::I64]);
    let mut fb = mb.function("identity", ft);

    let entry = fb.create_block();
    let a = fb.add_block_param(entry, Ty::I64);
    fb.set_entry(entry);

    fb.switch_to_block(entry);
    fb.ret(vec![a]);

    fb.build();
    mb.build()
}

fn build_multi_function_module() -> tmir::Module {
    let mut mb = ModuleBuilder::new("cli_tmbc_multi");

    let ft = mb.add_func_type(vec![], vec![Ty::I64]);

    let mut fb1 = mb.function("_const_10", ft);
    let entry1 = fb1.create_block();
    fb1.set_entry(entry1);
    fb1.switch_to_block(entry1);
    let result1 = fb1.iconst(Ty::I64, 10);
    fb1.ret(vec![result1]);
    fb1.build();

    let mut fb2 = mb.function("_const_32", ft);
    let entry2 = fb2.create_block();
    fb2.set_entry(entry2);
    fb2.switch_to_block(entry2);
    let result2 = fb2.iconst(Ty::I64, 32);
    fb2.ret(vec![result2]);
    fb2.build();

    mb.build()
}

// ---------------------------------------------------------------------------
// Test: serialization round-trip -> compile -> link -> run (return constant 42)
// ---------------------------------------------------------------------------

#[test]
fn e2e_tmbc_return_42() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e_tmbc_return_42: requires AArch64 + cc");
        return;
    }

    let module_orig = build_return_42_module();

    // Step 1: Serialize to JSON.
    let json_bytes = serde_json::to_vec(&module_orig)
        .expect("serializing tMIR module should succeed");

    eprintln!("--- tMIR serialized ({} bytes) ---", json_bytes.len());

    // Step 2: Deserialize back (validates round-trip fidelity).
    let module_rt: tmir::Module = serde_json::from_slice(&json_bytes)
        .expect("deserializing tMIR module should succeed");
    assert_eq!(
        module_orig, module_rt,
        "serialization round-trip should preserve module equality"
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
             The Mach-O object produced from deserialized tMIR is not linkable.",
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
// Test: serialization round-trip -> compile -> link -> run (add two args)
// ---------------------------------------------------------------------------

#[test]
fn e2e_tmbc_add_args() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e_tmbc_add_args: requires AArch64 + cc");
        return;
    }

    let module_orig = build_add_i64_module();

    // Serialization round-trip.
    let json_bytes = serde_json::to_vec(&module_orig)
        .expect("serialization should succeed");
    let module_rt: tmir::Module = serde_json::from_slice(&json_bytes)
        .expect("deserialization should succeed");
    assert_eq!(
        module_orig, module_rt,
        "round-trip should preserve module equality"
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
// Test: file-based serialization round-trip -> compile (no link)
// ---------------------------------------------------------------------------

#[test]
fn e2e_tmbc_file_roundtrip_compile() {
    let module_orig = build_identity_module();

    // Write to JSON file on disk.
    let test_dir = make_test_dir("file_roundtrip");
    let json_path = test_dir.join("module.json");

    let json_bytes = serde_json::to_vec_pretty(&module_orig)
        .expect("serialization should succeed");
    fs::write(&json_path, &json_bytes).expect("write JSON file");

    // Read back from file.
    let bytes_from_file = fs::read(&json_path).expect("read JSON file");
    let module_from_file: tmir::Module = serde_json::from_slice(&bytes_from_file)
        .expect("reading JSON from file should succeed");

    assert_eq!(
        module_orig, module_from_file,
        "file-based round-trip should preserve module equality"
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
// Test: compact vs pretty JSON encoding size difference
// ---------------------------------------------------------------------------

#[test]
fn e2e_tmbc_compact_smaller_than_pretty() {
    let module = build_add_i64_module();

    let compact = serde_json::to_vec(&module)
        .expect("compact serialization should succeed");
    let pretty = serde_json::to_string_pretty(&module)
        .expect("pretty serialization should succeed");

    eprintln!(
        "Compact: {} bytes, Pretty: {} bytes (ratio: {:.1}x)",
        compact.len(),
        pretty.len(),
        pretty.len() as f64 / compact.len() as f64
    );

    assert!(
        compact.len() < pretty.len(),
        "compact encoding ({} bytes) should be smaller than pretty encoding ({} bytes)",
        compact.len(),
        pretty.len()
    );
}

// ---------------------------------------------------------------------------
// Test: multi-function module through serialization -> compile -> link -> run
// ---------------------------------------------------------------------------

#[test]
fn e2e_tmbc_multi_function_link_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e_tmbc_multi_function_link_run: requires AArch64 + cc");
        return;
    }

    let module_orig = build_multi_function_module();

    // Serialization round-trip.
    let json_bytes = serde_json::to_vec(&module_orig)
        .expect("serialization should succeed");
    let module_rt: tmir::Module = serde_json::from_slice(&json_bytes)
        .expect("deserialization should succeed");
    assert_eq!(
        module_orig, module_rt,
        "round-trip should preserve module equality"
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
// Test: invalid JSON is rejected by deserialization
// ---------------------------------------------------------------------------

#[test]
fn e2e_tmbc_invalid_data_rejected() {
    let err = serde_json::from_slice::<tmir::Module>(b"not_valid_json")
        .expect_err("invalid data should be rejected");

    eprintln!("Expected rejection: {}", err);
}

// ---------------------------------------------------------------------------
// Test: format detection by content
//
// Verifies that JSON content is correctly identified.
// ---------------------------------------------------------------------------

#[test]
fn e2e_tmbc_format_detection() {
    // Valid JSON should parse.
    let module = build_return_42_module();
    let json_bytes = serde_json::to_vec(&module)
        .expect("serialization should succeed");
    assert!(json_bytes.first().copied() == Some(b'{'));

    // Invalid JSON should not parse.
    assert!(serde_json::from_slice::<tmir::Module>(b"").is_err());
    assert!(serde_json::from_slice::<tmir::Module>(b"tMBCfake").is_err());
}
