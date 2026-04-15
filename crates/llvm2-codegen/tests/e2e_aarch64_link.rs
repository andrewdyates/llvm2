// E2E integration tests: tMIR -> LLVM2 pipeline -> Mach-O .o -> link -> run
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// These tests verify the complete compilation pipeline produces runnable
// AArch64 binaries on macOS (Apple Silicon). This is the most important
// milestone for LLVM2: proving it can generate real executables.

use std::fs;
use std::io::Write;
use std::process::Command;

use llvm2_codegen::compiler::{Compiler, CompilerConfig, CompilerTraceLevel};
use llvm2_codegen::pipeline::{self, OptLevel};

// =============================================================================
// Helper: write bytes to a temp file and return the path
// =============================================================================

fn write_temp_file(name: &str, suffix: &str, contents: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("llvm2_e2e_tests");
    fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join(format!("{}{}", name, suffix));
    let mut f = fs::File::create(&path).expect("create temp file");
    f.write_all(contents).expect("write temp file");
    path
}

// =============================================================================
// Helper: compile a tMIR module through the Compiler API
// =============================================================================

/// Build a minimal tMIR module: `fn add(a: i32, b: i32) -> i32 { a + b }`
fn build_tmir_add_module() -> tmir_func::Module {
    use tmir_instrs::{BinOp, Instr, InstrNode};
    use tmir_types::{BlockId, FuncId, FuncTy, Ty, ValueId};

    let func = tmir_func::Function {
        id: FuncId(0),
        name: "_add_two".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(32), Ty::Int(32)],
            returns: vec![Ty::Int(32)],
        },
        entry: BlockId(0),
        blocks: vec![tmir_func::Block {
            id: BlockId(0),
            params: vec![
                (ValueId(0), Ty::Int(32)), // param a
                (ValueId(1), Ty::Int(32)), // param b
            ],
            body: vec![
                InstrNode {
                    instr: Instr::BinOp {
                        op: BinOp::Add,
                        ty: Ty::Int(32),
                        lhs: ValueId(0),
                        rhs: ValueId(1),
                    },
                    results: vec![ValueId(2)],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(2)],
                    },
                    results: vec![],
                    proofs: vec![],
                },
            ],
        }],
        proofs: vec![],
    };

    tmir_func::Module {
        name: "e2e_test".to_string(),
        functions: vec![func],
        structs: vec![],
    }
}

/// Build a tMIR module: `fn const_42() -> i32 { 42 }`
fn build_tmir_const_module() -> tmir_func::Module {
    use tmir_instrs::{Instr, InstrNode};
    use tmir_types::{BlockId, FuncId, FuncTy, Ty, ValueId};

    let func = tmir_func::Function {
        id: FuncId(0),
        name: "_const_42".to_string(),
        ty: FuncTy {
            params: vec![],
            returns: vec![Ty::Int(32)],
        },
        entry: BlockId(0),
        blocks: vec![tmir_func::Block {
            id: BlockId(0),
            params: vec![],
            body: vec![
                InstrNode {
                    instr: Instr::Const {
                        ty: Ty::Int(32),
                        value: 42,
                    },
                    results: vec![ValueId(0)],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(0)],
                    },
                    results: vec![],
                    proofs: vec![],
                },
            ],
        }],
        proofs: vec![],
    };

    tmir_func::Module {
        name: "e2e_const_test".to_string(),
        functions: vec![func],
        structs: vec![],
    }
}

/// Build a tMIR module: `fn sub_vals(a: i64, b: i64) -> i64 { a - b }`
fn build_tmir_sub_module() -> tmir_func::Module {
    use tmir_instrs::{BinOp, Instr, InstrNode};
    use tmir_types::{BlockId, FuncId, FuncTy, Ty, ValueId};

    let func = tmir_func::Function {
        id: FuncId(0),
        name: "_sub_vals".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(64), Ty::Int(64)],
            returns: vec![Ty::Int(64)],
        },
        entry: BlockId(0),
        blocks: vec![tmir_func::Block {
            id: BlockId(0),
            params: vec![
                (ValueId(0), Ty::Int(64)),
                (ValueId(1), Ty::Int(64)),
            ],
            body: vec![
                InstrNode {
                    instr: Instr::BinOp {
                        op: BinOp::Sub,
                        ty: Ty::Int(64),
                        lhs: ValueId(0),
                        rhs: ValueId(1),
                    },
                    results: vec![ValueId(2)],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(2)],
                    },
                    results: vec![],
                    proofs: vec![],
                },
            ],
        }],
        proofs: vec![],
    };

    tmir_func::Module {
        name: "e2e_sub_test".to_string(),
        functions: vec![func],
        structs: vec![],
    }
}

// =============================================================================
// Test 1: Compiler API produces non-empty Mach-O from tMIR module
// =============================================================================

#[test]
fn e2e_aarch64_tmir_module_to_object_code() {
    let module = build_tmir_add_module();
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        trace_level: CompilerTraceLevel::Full,
        ..CompilerConfig::default()
    });

    let result = compiler.compile(&module).expect("compilation should succeed");

    // Object code must be non-empty
    assert!(
        !result.object_code.is_empty(),
        "Compiler::compile() must produce non-empty Mach-O bytes"
    );

    // Metrics should reflect one function
    assert_eq!(result.metrics.function_count, 1);
    assert!(result.metrics.code_size_bytes > 0);

    // Trace should be populated (we asked for Full)
    let trace = result.trace.expect("trace should be present with Full level");
    assert!(!trace.entries.is_empty());

    eprintln!(
        "tMIR->object: {} bytes, {} instructions, trace entries: {}",
        result.metrics.code_size_bytes,
        result.metrics.instruction_count,
        trace.entries.len()
    );
}

// =============================================================================
// Test 2: Mach-O file has valid magic number and structure
// =============================================================================

#[test]
fn e2e_aarch64_macho_magic_number() {
    let module = build_tmir_add_module();
    let compiler = Compiler::default_o2();
    let result = compiler.compile(&module).expect("compilation should succeed");
    let obj = &result.object_code;

    // Mach-O 64-bit magic: 0xFEEDFACF
    assert!(obj.len() >= 4, "object too small for Mach-O header");
    let magic = u32::from_le_bytes([obj[0], obj[1], obj[2], obj[3]]);
    assert_eq!(
        magic, 0xFEED_FACF,
        "expected Mach-O 64-bit magic 0xFEEDFACF, got {:#010X}",
        magic
    );

    // CPU type for ARM64: 0x0100000C (CPU_TYPE_ARM64)
    let cputype = u32::from_le_bytes([obj[4], obj[5], obj[6], obj[7]]);
    assert_eq!(
        cputype, 0x0100_000C,
        "expected CPU_TYPE_ARM64 (0x0100000C), got {:#010X}",
        cputype
    );
}

// =============================================================================
// Test 3: Object file disassembles with otool (validates encoding)
// =============================================================================

#[test]
fn e2e_aarch64_otool_disassembly() {
    let module = build_tmir_add_module();
    let compiler = Compiler::default_o2();
    let result = compiler.compile(&module).expect("compilation should succeed");

    let obj_path = write_temp_file("add_two", ".o", &result.object_code);

    // Run otool -tv to disassemble
    let output = Command::new("otool")
        .args(["-tv", obj_path.to_str().unwrap()])
        .output()
        .expect("otool should be available on macOS");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "otool -tv failed: stderr={}",
        stderr
    );

    // Should contain the function symbol
    assert!(
        stdout.contains("__add_two") || stdout.contains("_add_two"),
        "otool output should contain the function symbol. Got:\n{}",
        stdout
    );

    // Should contain AArch64 instructions (add, ret are expected)
    let has_instructions = stdout.contains("add")
        || stdout.contains("ret")
        || stdout.contains("mov");
    assert!(
        has_instructions,
        "otool output should show AArch64 instructions. Got:\n{}",
        stdout
    );

    eprintln!("otool disassembly:\n{}", stdout);
}

// =============================================================================
// Test 4: Link and run -- the big one
//
// We compile add_two(i32, i32) -> i32 via LLVM2, write a C main() that calls
// it, link them together, and verify the output.
// =============================================================================

#[test]
fn e2e_aarch64_link_and_run() {
    let module = build_tmir_add_module();
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        ..CompilerConfig::default()
    });
    let result = compiler.compile(&module).expect("compilation should succeed");

    let test_dir = std::env::temp_dir().join("llvm2_e2e_tests");
    fs::create_dir_all(&test_dir).expect("create temp dir");

    let obj_path = test_dir.join("add_two.o");
    fs::write(&obj_path, &result.object_code).expect("write object file");

    // Write a C driver that calls our compiled function
    let driver_path = test_dir.join("driver_add.c");
    let driver_src = r#"
#include <stdio.h>

// Declare the function compiled by LLVM2
// Mach-O symbol: __add_two (C sees it as _add_two without extra underscore)
extern int _add_two(int a, int b);

int main() {
    int result = _add_two(30, 12);
    printf("%d\n", result);
    return (result == 42) ? 0 : 1;
}
"#;
    fs::write(&driver_path, driver_src).expect("write driver source");

    let binary_path = test_dir.join("test_add");

    // Compile and link: cc driver.c add_two.o -o test_add
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
        // If linking fails, fall back to verifying the .o with otool
        eprintln!("Linking failed (may be expected if ABI is off): {}", link_stderr);

        // Verify the object at least disassembles
        let otool_out = Command::new("otool")
            .args(["-tv", obj_path.to_str().unwrap()])
            .output()
            .expect("otool");
        let otool_stdout = String::from_utf8_lossy(&otool_out.stdout);
        assert!(
            otool_out.status.success(),
            "otool should at least work on the .o"
        );
        eprintln!("Object disassembly (link failed):\n{}", otool_stdout);

        // Also check symbols
        let nm_out = Command::new("nm")
            .args([obj_path.to_str().unwrap()])
            .output()
            .expect("nm");
        let nm_stdout = String::from_utf8_lossy(&nm_out.stdout);
        eprintln!("Symbols:\n{}", nm_stdout);

        panic!(
            "Linking failed. This means the Mach-O structure or ABI is wrong.\n\
             Linker stderr: {}\n\
             Fix the pipeline to produce a linkable object.",
            link_stderr
        );
    }

    eprintln!("Link succeeded: {}", binary_path.display());

    // Run the binary
    let run_output = Command::new(binary_path.to_str().unwrap())
        .output()
        .expect("should be able to run the binary");

    let run_stdout = String::from_utf8_lossy(&run_output.stdout);
    let run_stderr = String::from_utf8_lossy(&run_output.stderr);

    eprintln!("Binary stdout: {}", run_stdout.trim());
    eprintln!("Binary stderr: {}", run_stderr);
    eprintln!("Exit code: {:?}", run_output.status.code());

    assert!(
        run_output.status.success(),
        "Binary should exit 0 (add(30, 12) == 42). Got exit code {:?}\nstdout: {}\nstderr: {}",
        run_output.status.code(),
        run_stdout,
        run_stderr
    );

    assert_eq!(
        run_stdout.trim(),
        "42",
        "Binary should print 42 (30 + 12)"
    );
}

// =============================================================================
// Test 5: build_add_test_function() IR path (bypasses tMIR adapter + ISel)
// =============================================================================

#[test]
fn e2e_aarch64_ir_function_compile() {
    let mut ir_func = pipeline::build_add_test_function();
    let compiler = Compiler::default_o2();
    let result = compiler
        .compile_ir_function(&mut ir_func)
        .expect("compile_ir_function should succeed");

    assert!(!result.object_code.is_empty());
    assert!(result.metrics.code_size_bytes > 0);
    assert_eq!(result.metrics.function_count, 1);

    // Verify Mach-O magic
    let obj = &result.object_code;
    let magic = u32::from_le_bytes([obj[0], obj[1], obj[2], obj[3]]);
    assert_eq!(magic, 0xFEED_FACF, "should be valid Mach-O");

    eprintln!(
        "IR path: {} bytes, {} estimated instructions",
        result.metrics.code_size_bytes,
        result.metrics.instruction_count
    );
}

// =============================================================================
// Test 6: IR path link-and-run with the pre-built add function
// =============================================================================

#[test]
fn e2e_aarch64_ir_link_and_run() {
    let mut ir_func = pipeline::build_add_test_function();
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        ..CompilerConfig::default()
    });
    let result = compiler
        .compile_ir_function(&mut ir_func)
        .expect("compile_ir_function should succeed");

    let test_dir = std::env::temp_dir().join("llvm2_e2e_tests");
    fs::create_dir_all(&test_dir).expect("create temp dir");

    let obj_path = test_dir.join("ir_add.o");
    fs::write(&obj_path, &result.object_code).expect("write object file");

    // The IR build_add_test_function creates a function named "add",
    // which becomes symbol "_add" in Mach-O
    let driver_path = test_dir.join("driver_ir_add.c");
    let driver_src = r#"
#include <stdio.h>

extern int add(int a, int b);

int main() {
    int result = add(17, 25);
    printf("%d\n", result);
    return (result == 42) ? 0 : 1;
}
"#;
    fs::write(&driver_path, driver_src).expect("write driver source");

    let binary_path = test_dir.join("test_ir_add");

    let link_output = Command::new("cc")
        .args([
            driver_path.to_str().unwrap(),
            obj_path.to_str().unwrap(),
            "-o",
            binary_path.to_str().unwrap(),
        ])
        .output()
        .expect("cc");

    let link_stderr = String::from_utf8_lossy(&link_output.stderr);
    if !link_output.status.success() {
        eprintln!("IR path linking failed: {}", link_stderr);

        let otool_out = Command::new("otool")
            .args(["-tv", obj_path.to_str().unwrap()])
            .output()
            .expect("otool");
        eprintln!(
            "Disassembly:\n{}",
            String::from_utf8_lossy(&otool_out.stdout)
        );

        let nm_out = Command::new("nm")
            .args([obj_path.to_str().unwrap()])
            .output()
            .expect("nm");
        eprintln!("Symbols:\n{}", String::from_utf8_lossy(&nm_out.stdout));

        panic!("IR path linking failed: {}", link_stderr);
    }

    let run_output = Command::new(binary_path.to_str().unwrap())
        .output()
        .expect("run binary");

    let stdout = String::from_utf8_lossy(&run_output.stdout);
    eprintln!("IR path binary stdout: {}", stdout.trim());

    assert!(
        run_output.status.success(),
        "IR add(17, 25) should produce 42 and exit 0. Got {:?}\n{}",
        run_output.status.code(),
        stdout
    );
    assert_eq!(stdout.trim(), "42");
}

// =============================================================================
// Test 7: tMIR constant function through full pipeline
// =============================================================================

#[test]
fn e2e_aarch64_tmir_const_to_object() {
    let module = build_tmir_const_module();
    let compiler = Compiler::default_o2();
    let result = compiler.compile(&module).expect("const compilation should succeed");

    assert!(!result.object_code.is_empty());
    assert_eq!(result.metrics.function_count, 1);

    // Verify valid Mach-O
    let obj = &result.object_code;
    let magic = u32::from_le_bytes([obj[0], obj[1], obj[2], obj[3]]);
    assert_eq!(magic, 0xFEED_FACF);

    eprintln!(
        "tMIR const->object: {} bytes",
        result.metrics.code_size_bytes
    );
}

// =============================================================================
// Test 8: tMIR subtraction (i64) through full pipeline
// =============================================================================

#[test]
fn e2e_aarch64_tmir_sub_to_object() {
    let module = build_tmir_sub_module();
    let compiler = Compiler::default_o2();
    let result = compiler.compile(&module).expect("sub compilation should succeed");

    assert!(!result.object_code.is_empty());
    assert_eq!(result.metrics.function_count, 1);

    // Verify valid Mach-O
    let obj = &result.object_code;
    let magic = u32::from_le_bytes([obj[0], obj[1], obj[2], obj[3]]);
    assert_eq!(magic, 0xFEED_FACF);

    eprintln!(
        "tMIR sub->object: {} bytes",
        result.metrics.code_size_bytes
    );
}

// =============================================================================
// Test 9: All optimization levels produce valid output
// =============================================================================

#[test]
fn e2e_aarch64_all_opt_levels() {
    let module = build_tmir_add_module();

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
            "{:?} produced empty object code",
            opt
        );

        let obj = &result.object_code;
        let magic = u32::from_le_bytes([obj[0], obj[1], obj[2], obj[3]]);
        assert_eq!(magic, 0xFEED_FACF, "{:?} produced invalid Mach-O", opt);

        eprintln!(
            "  {:?}: {} bytes, {} instructions",
            opt, result.metrics.code_size_bytes, result.metrics.instruction_count
        );
    }
}

// =============================================================================
// Test 10: nm shows expected symbol
// =============================================================================

#[test]
fn e2e_aarch64_nm_symbol_check() {
    let module = build_tmir_add_module();
    let compiler = Compiler::default_o2();
    let result = compiler.compile(&module).expect("compilation should succeed");

    let obj_path = write_temp_file("nm_test", ".o", &result.object_code);

    let output = Command::new("nm")
        .args([obj_path.to_str().unwrap()])
        .output()
        .expect("nm should be available");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "nm failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // The function should appear as an external text symbol
    // nm format: "<addr> T __add_two" (with Mach-O double underscore)
    assert!(
        stdout.contains("__add_two"),
        "nm should show __add_two symbol. Got:\n{}",
        stdout
    );

    eprintln!("nm output:\n{}", stdout);
}
