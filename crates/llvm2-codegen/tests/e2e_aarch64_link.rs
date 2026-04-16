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

// Multi-block test imports
use tmir_func::{Block as TmirBlock, Function as TmirFunction};
use tmir_instrs::{CmpOp, UnOp, Operand};
use tmir_types::{BlockId, FuncId, FuncTy};

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
                        lhs: Operand::Value(ValueId(0)),
                        rhs: Operand::Value(ValueId(1)),
                    },
                    results: vec![ValueId(2)],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Return {
                        values: vec![Operand::Value(ValueId(2))],
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
                        values: vec![Operand::Value(ValueId(0))],
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
                        lhs: Operand::Value(ValueId(0)),
                        rhs: Operand::Value(ValueId(1)),
                    },
                    results: vec![ValueId(2)],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Return {
                        values: vec![Operand::Value(ValueId(2))],
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

// =============================================================================
// Multi-block tMIR builders and E2E tests
//
// These tests exercise control flow (branches, loops) through the FULL pipeline:
//   tMIR -> adapter -> ISel -> opt -> regalloc -> frame -> encode -> Mach-O -> link -> run
//
// Part of #242 -- Multi-block E2E tests for control flow
// =============================================================================

// ---------------------------------------------------------------------------
// Helpers for multi-block link-and-run tests
// ---------------------------------------------------------------------------

fn make_test_dir(test_name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("llvm2_e2e_multiblock_{}", test_name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create test dir");
    dir
}

fn compile_tmir_module_to_obj(module: &tmir_func::Module) -> Vec<u8> {
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        ..CompilerConfig::default()
    });
    let result = compiler
        .compile(module)
        .expect("compilation should succeed");
    assert!(
        !result.object_code.is_empty(),
        "compiled object code must be non-empty"
    );
    result.object_code
}

fn link_and_run(
    test_name: &str,
    func_name: &str,
    obj_bytes: &[u8],
    driver_src: &str,
) -> (i32, String) {
    let dir = make_test_dir(test_name);
    let obj_path = dir.join(format!("{}.o", func_name));
    fs::write(&obj_path, obj_bytes).expect("write .o file");

    let driver_path = dir.join("driver.c");
    fs::write(&driver_path, driver_src).expect("write driver.c");

    let binary_path = dir.join(format!("test_{}", func_name));

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
        // Debug: show otool and nm output
        let otool_out = Command::new("otool")
            .args(["-tv", obj_path.to_str().unwrap()])
            .output()
            .ok();
        let nm_out = Command::new("nm")
            .args([obj_path.to_str().unwrap()])
            .output()
            .ok();

        let disasm = otool_out
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();
        let symbols = nm_out
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();

        panic!(
            "Linking failed for {}!\nstderr: {}\notool:\n{}\nnm:\n{}",
            func_name, stderr, disasm, symbols
        );
    }

    let run_output = Command::new(binary_path.to_str().unwrap())
        .output()
        .expect("should be able to run the binary");

    let stdout = String::from_utf8_lossy(&run_output.stdout).to_string();
    let exit_code = run_output.status.code().unwrap_or(-1);

    let _ = fs::remove_dir_all(&dir);

    (exit_code, stdout)
}

// ---------------------------------------------------------------------------
// Builder: max(a, b) -- conditional branch (if-then-else diamond)
//
// fn max_val(a: i64, b: i64) -> i64 {
//     if a > b { a } else { b }
// }
//
// bb0 (entry): cmp a > b, condbr -> bb1 (return a), bb2 (return b)
// bb1: return a
// bb2: return b
// ---------------------------------------------------------------------------

fn build_tmir_max_module() -> tmir_func::Module {
    use tmir_instrs::{Instr, InstrNode};
    use tmir_types::{Ty, ValueId};

    let func = TmirFunction {
        id: FuncId(0),
        name: "_max_val".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(64), Ty::Int(64)],
            returns: vec![Ty::Int(64)],
        },
        entry: BlockId(0),
        blocks: vec![
            // bb0 (entry): compare and branch
            TmirBlock {
                id: BlockId(0),
                params: vec![
                    (ValueId(0), Ty::Int(64)), // a
                    (ValueId(1), Ty::Int(64)), // b
                ],
                body: vec![
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Sgt,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(0)), // a
                            rhs: Operand::Value(ValueId(1)), // b
                        },
                        results: vec![ValueId(2)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: Operand::Value(ValueId(2)),
                            then_target: BlockId(1), // a > b => return a
                            then_args: vec![],
                            else_target: BlockId(2), // else return b
                            else_args: vec![],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb1: return a
            TmirBlock {
                id: BlockId(1),
                params: vec![],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![Operand::Value(ValueId(0))],
                    },
                    results: vec![],
                    proofs: vec![],
                }],
            },
            // bb2: return b
            TmirBlock {
                id: BlockId(2),
                params: vec![],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![Operand::Value(ValueId(1))],
                    },
                    results: vec![],
                    proofs: vec![],
                }],
            },
        ],
        proofs: vec![],
    };

    tmir_func::Module {
        name: "e2e_max_test".to_string(),
        functions: vec![func],
        structs: vec![],
    }
}

// ---------------------------------------------------------------------------
// Builder: abs(x) -- comparison + conditional negate
//
// fn abs_val(x: i64) -> i64 {
//     if x < 0 { 0 - x } else { x }
// }
//
// bb0 (entry): const 0, cmp x < 0, condbr -> bb1 (negate), bb2 (return x)
// bb1: neg = 0 - x, return neg
// bb2: return x
// ---------------------------------------------------------------------------

fn build_tmir_abs_module() -> tmir_func::Module {
    use tmir_instrs::{BinOp, Instr, InstrNode};
    use tmir_types::{Ty, ValueId};

    let func = TmirFunction {
        id: FuncId(0),
        name: "_abs_val".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(64)],
            returns: vec![Ty::Int(64)],
        },
        entry: BlockId(0),
        blocks: vec![
            // bb0 (entry): compare x < 0
            TmirBlock {
                id: BlockId(0),
                params: vec![(ValueId(0), Ty::Int(64))], // x
                body: vec![
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 0,
                        },
                        results: vec![ValueId(1)], // zero
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Slt,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(0)), // x
                            rhs: Operand::Value(ValueId(1)), // 0
                        },
                        results: vec![ValueId(2)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: Operand::Value(ValueId(2)),
                            then_target: BlockId(1), // negate
                            then_args: vec![],
                            else_target: BlockId(2), // return x
                            else_args: vec![],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb1: negate -- return 0 - x
            TmirBlock {
                id: BlockId(1),
                params: vec![],
                body: vec![
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Sub,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(1)), // 0 from bb0
                            rhs: Operand::Value(ValueId(0)), // x from bb0
                        },
                        results: vec![ValueId(3)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Return {
                            values: vec![Operand::Value(ValueId(3))],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb2: return x
            TmirBlock {
                id: BlockId(2),
                params: vec![],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![Operand::Value(ValueId(0))],
                    },
                    results: vec![],
                    proofs: vec![],
                }],
            },
        ],
        proofs: vec![],
    };

    tmir_func::Module {
        name: "e2e_abs_test".to_string(),
        functions: vec![func],
        structs: vec![],
    }
}

// ---------------------------------------------------------------------------
// Builder: fibonacci(n) -- loop with accumulator and block parameters
//
// fn fibonacci(n: i64) -> i64 {
//     if n <= 1 { return n }
//     a = 0; b = 1; i = 2
//     loop { tmp = a + b; a = b; b = tmp; i += 1; if i > n: return b }
// }
//
// bb0 (entry): const 1, cmp n <= 1, condbr -> bb1 (ret n), bb2 (loop_init)
// bb1: return n
// bb2: a=0, b=1, i=2, br -> bb3
// bb3 (loop): params(a, b, i)
//   tmp = a + b, new_i = i + 1, cmp new_i <= n
//   condbr -> bb3(b, tmp, new_i), bb4(tmp)
// bb4 (exit): params(result), return result
// ---------------------------------------------------------------------------

fn build_tmir_fibonacci_module() -> tmir_func::Module {
    use tmir_instrs::{BinOp, Instr, InstrNode};
    use tmir_types::{Ty, ValueId};

    let func = TmirFunction {
        id: FuncId(0),
        name: "_fibonacci".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(64)],
            returns: vec![Ty::Int(64)],
        },
        entry: BlockId(0),
        blocks: vec![
            // bb0 (entry): check n <= 1
            TmirBlock {
                id: BlockId(0),
                params: vec![(ValueId(0), Ty::Int(64))], // n
                body: vec![
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 1,
                        },
                        results: vec![ValueId(1)], // const_1
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Sle,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(0)), // n
                            rhs: Operand::Value(ValueId(1)), // 1
                        },
                        results: vec![ValueId(2)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: Operand::Value(ValueId(2)),
                            then_target: BlockId(1), // ret_n
                            then_args: vec![],
                            else_target: BlockId(2), // loop_init
                            else_args: vec![],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb1 (ret_n): return n
            TmirBlock {
                id: BlockId(1),
                params: vec![],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![Operand::Value(ValueId(0))],
                    },
                    results: vec![],
                    proofs: vec![],
                }],
            },
            // bb2 (loop_init): a=0, b=1, i=2, jump to loop
            TmirBlock {
                id: BlockId(2),
                params: vec![],
                body: vec![
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 0,
                        },
                        results: vec![ValueId(10)], // a = 0
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 1,
                        },
                        results: vec![ValueId(11)], // b = 1
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 2,
                        },
                        results: vec![ValueId(12)], // i = 2
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Br {
                            target: BlockId(3),
                            args: vec![Operand::Value(ValueId(10)), Operand::Value(ValueId(11)), Operand::Value(ValueId(12))],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb3 (loop body): params(a, b, i)
            TmirBlock {
                id: BlockId(3),
                params: vec![
                    (ValueId(20), Ty::Int(64)), // a
                    (ValueId(21), Ty::Int(64)), // b
                    (ValueId(22), Ty::Int(64)), // i
                ],
                body: vec![
                    // tmp = a + b
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Add,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(20)), // a
                            rhs: Operand::Value(ValueId(21)), // b
                        },
                        results: vec![ValueId(23)], // tmp
                        proofs: vec![],
                    },
                    // new_i = i + 1
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 1,
                        },
                        results: vec![ValueId(24)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Add,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(22)), // i
                            rhs: Operand::Value(ValueId(24)), // 1
                        },
                        results: vec![ValueId(25)], // new_i
                        proofs: vec![],
                    },
                    // cmp new_i <= n
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Sle,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(25)), // new_i
                            rhs: Operand::Value(ValueId(0)),  // n (from entry)
                        },
                        results: vec![ValueId(26)],
                        proofs: vec![],
                    },
                    // condbr: loop back or exit
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: Operand::Value(ValueId(26)),
                            then_target: BlockId(3), // loop (b, tmp, new_i)
                            then_args: vec![Operand::Value(ValueId(21)), Operand::Value(ValueId(23)), Operand::Value(ValueId(25))],
                            else_target: BlockId(4), // exit (tmp)
                            else_args: vec![Operand::Value(ValueId(23))],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb4 (exit): return result
            TmirBlock {
                id: BlockId(4),
                params: vec![(ValueId(30), Ty::Int(64))],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![Operand::Value(ValueId(30))],
                    },
                    results: vec![],
                    proofs: vec![],
                }],
            },
        ],
        proofs: vec![],
    };

    tmir_func::Module {
        name: "e2e_fibonacci_test".to_string(),
        functions: vec![func],
        structs: vec![],
    }
}

// ---------------------------------------------------------------------------
// Builder: sum_1_to_n(n) -- simple counting loop
//
// fn sum_1_to_n(n: i64) -> i64 {
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

fn build_tmir_sum_1_to_n_module() -> tmir_func::Module {
    use tmir_instrs::{BinOp, Instr, InstrNode};
    use tmir_types::{Ty, ValueId};

    let func = TmirFunction {
        id: FuncId(0),
        name: "_sum_1_to_n".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(64)],
            returns: vec![Ty::Int(64)],
        },
        entry: BlockId(0),
        blocks: vec![
            // bb0 (entry): init sum=0, i=1, jump to loop
            TmirBlock {
                id: BlockId(0),
                params: vec![(ValueId(0), Ty::Int(64))], // n
                body: vec![
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 0,
                        },
                        results: vec![ValueId(1)], // sum_init = 0
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 1,
                        },
                        results: vec![ValueId(2)], // i_init = 1
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Br {
                            target: BlockId(1),
                            args: vec![Operand::Value(ValueId(1)), Operand::Value(ValueId(2))],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb1 (loop header): params(sum, i), check i <= n
            TmirBlock {
                id: BlockId(1),
                params: vec![
                    (ValueId(10), Ty::Int(64)), // sum
                    (ValueId(11), Ty::Int(64)), // i
                ],
                body: vec![
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Sle,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(11)), // i
                            rhs: Operand::Value(ValueId(0)),  // n
                        },
                        results: vec![ValueId(12)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: Operand::Value(ValueId(12)),
                            then_target: BlockId(2), // body
                            then_args: vec![],
                            else_target: BlockId(3), // exit
                            else_args: vec![],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb2 (body): sum += i, i += 1, back to loop
            TmirBlock {
                id: BlockId(2),
                params: vec![],
                body: vec![
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Add,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(10)), // sum
                            rhs: Operand::Value(ValueId(11)), // i
                        },
                        results: vec![ValueId(20)], // new_sum
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 1,
                        },
                        results: vec![ValueId(21)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Add,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(11)), // i
                            rhs: Operand::Value(ValueId(21)), // 1
                        },
                        results: vec![ValueId(22)], // new_i
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Br {
                            target: BlockId(1),
                            args: vec![Operand::Value(ValueId(20)), Operand::Value(ValueId(22))],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb3 (exit): return sum
            TmirBlock {
                id: BlockId(3),
                params: vec![],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![Operand::Value(ValueId(10))], // sum from bb1
                    },
                    results: vec![],
                    proofs: vec![],
                }],
            },
        ],
        proofs: vec![],
    };

    tmir_func::Module {
        name: "e2e_sum_1_to_n_test".to_string(),
        functions: vec![func],
        structs: vec![],
    }
}

// =============================================================================
// Test 11: max(a, b) -- conditional branch, compile to valid Mach-O
// =============================================================================

#[test]
fn e2e_aarch64_max_val_compile() {
    let module = build_tmir_max_module();
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        trace_level: CompilerTraceLevel::Full,
        ..CompilerConfig::default()
    });

    let result = compiler.compile(&module).expect("max_val compilation should succeed");

    assert!(
        !result.object_code.is_empty(),
        "max_val must produce non-empty object code"
    );
    assert_eq!(result.metrics.function_count, 1);

    // Valid Mach-O magic
    let obj = &result.object_code;
    let magic = u32::from_le_bytes([obj[0], obj[1], obj[2], obj[3]]);
    assert_eq!(magic, 0xFEED_FACF, "must be valid Mach-O");

    eprintln!(
        "max_val: {} bytes, {} instructions",
        result.metrics.code_size_bytes,
        result.metrics.instruction_count
    );
}

// =============================================================================
// Test 12: max(a, b) -- link and run (if-then-else diamond CFG)
// =============================================================================

#[test]
fn e2e_aarch64_max_val_link_and_run() {
    let module = build_tmir_max_module();
    let obj_bytes = compile_tmir_module_to_obj(&module);

    let driver = r#"
#include <stdio.h>

extern long _max_val(long a, long b);

int main(void) {
    long r1 = _max_val(10, 20);
    long r2 = _max_val(20, 10);
    long r3 = _max_val(5, 5);
    long r4 = _max_val(-3, -7);
    long r5 = _max_val(-1, 1);
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

    let (exit_code, stdout) = link_and_run("max_val", "max_val", &obj_bytes, driver);
    eprintln!("max_val link+run stdout: {}", stdout.trim());
    assert_eq!(
        exit_code, 0,
        "max_val link+run failed (exit {}). \
         1=max(10,20)!=20, 2=max(20,10)!=20, 3=max(5,5)!=5, 4=max(-3,-7)!=-3, 5=max(-1,1)!=1. \
         stdout: {}",
        exit_code, stdout
    );
}

// =============================================================================
// Test 13: abs(x) -- comparison + conditional negate, compile to valid Mach-O
// =============================================================================

#[test]
fn e2e_aarch64_abs_val_compile() {
    let module = build_tmir_abs_module();
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        ..CompilerConfig::default()
    });

    let result = compiler.compile(&module).expect("abs_val compilation should succeed");

    assert!(!result.object_code.is_empty());
    assert_eq!(result.metrics.function_count, 1);

    let obj = &result.object_code;
    let magic = u32::from_le_bytes([obj[0], obj[1], obj[2], obj[3]]);
    assert_eq!(magic, 0xFEED_FACF);

    eprintln!(
        "abs_val: {} bytes, {} instructions",
        result.metrics.code_size_bytes,
        result.metrics.instruction_count
    );
}

// =============================================================================
// Test 14: abs(x) -- link and run
// =============================================================================

#[test]
fn e2e_aarch64_abs_val_link_and_run() {
    let module = build_tmir_abs_module();
    let obj_bytes = compile_tmir_module_to_obj(&module);

    let driver = r#"
#include <stdio.h>

extern long _abs_val(long x);

int main(void) {
    long r1 = _abs_val(42);
    long r2 = _abs_val(-42);
    long r3 = _abs_val(0);
    long r4 = _abs_val(-1);
    long r5 = _abs_val(1);
    long r6 = _abs_val(-9999);
    printf("abs(42)=%ld abs(-42)=%ld abs(0)=%ld abs(-1)=%ld abs(1)=%ld abs(-9999)=%ld\n",
           r1, r2, r3, r4, r5, r6);
    if (r1 != 42)   return 1;
    if (r2 != 42)   return 2;
    if (r3 != 0)    return 3;
    if (r4 != 1)    return 4;
    if (r5 != 1)    return 5;
    if (r6 != 9999) return 6;
    return 0;
}
"#;

    let (exit_code, stdout) = link_and_run("abs_val", "abs_val", &obj_bytes, driver);
    eprintln!("abs_val link+run stdout: {}", stdout.trim());
    assert_eq!(
        exit_code, 0,
        "abs_val link+run failed (exit {}). \
         1=abs(42)!=42, 2=abs(-42)!=42, 3=abs(0)!=0, 4=abs(-1)!=1, 5=abs(1)!=1, 6=abs(-9999)!=9999. \
         stdout: {}",
        exit_code, stdout
    );
}

// =============================================================================
// Test 15: fibonacci(n) -- loop with accumulator, compile to valid Mach-O
// =============================================================================

#[test]
fn e2e_aarch64_fibonacci_compile() {
    let module = build_tmir_fibonacci_module();
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        trace_level: CompilerTraceLevel::Full,
        ..CompilerConfig::default()
    });

    let result = compiler.compile(&module).expect("fibonacci compilation should succeed");

    assert!(!result.object_code.is_empty());
    assert_eq!(result.metrics.function_count, 1);

    let obj = &result.object_code;
    let magic = u32::from_le_bytes([obj[0], obj[1], obj[2], obj[3]]);
    assert_eq!(magic, 0xFEED_FACF);

    // Multi-block function should produce more code than simple single-block
    assert!(
        result.metrics.code_size_bytes > 50,
        "fibonacci (5 blocks, loop) should produce substantial code, got {} bytes",
        result.metrics.code_size_bytes
    );

    let trace = result.trace.expect("trace should be present");
    eprintln!(
        "fibonacci: {} bytes, {} instructions, {} trace entries",
        result.metrics.code_size_bytes,
        result.metrics.instruction_count,
        trace.entries.len()
    );
}

// =============================================================================
// Test 16: fibonacci(n) -- link and run
// =============================================================================

#[test]
fn e2e_aarch64_fibonacci_link_and_run() {
    let module = build_tmir_fibonacci_module();
    let obj_bytes = compile_tmir_module_to_obj(&module);

    // Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55
    let driver = r#"
#include <stdio.h>

extern long _fibonacci(long n);

int main(void) {
    long r0 = _fibonacci(0);   /* 0 */
    long r1 = _fibonacci(1);   /* 1 */
    long r2 = _fibonacci(2);   /* 1 */
    long r5 = _fibonacci(5);   /* 5 */
    long r10 = _fibonacci(10); /* 55 */
    long r20 = _fibonacci(20); /* 6765 */
    printf("fib(0)=%ld fib(1)=%ld fib(2)=%ld fib(5)=%ld fib(10)=%ld fib(20)=%ld\n",
           r0, r1, r2, r5, r10, r20);
    if (r0 != 0)    return 1;
    if (r1 != 1)    return 2;
    if (r2 != 1)    return 3;
    if (r5 != 5)    return 4;
    if (r10 != 55)  return 5;
    if (r20 != 6765) return 6;
    return 0;
}
"#;

    let (exit_code, stdout) = link_and_run("fibonacci", "fibonacci", &obj_bytes, driver);
    eprintln!("fibonacci link+run stdout: {}", stdout.trim());
    assert_eq!(
        exit_code, 0,
        "fibonacci link+run failed (exit {}). \
         1=fib(0)!=0, 2=fib(1)!=1, 3=fib(2)!=1, 4=fib(5)!=5, 5=fib(10)!=55, 6=fib(20)!=6765. \
         stdout: {}",
        exit_code, stdout
    );
}

// =============================================================================
// Test 17: sum_1_to_n(n) -- simple counting loop, compile to valid Mach-O
// =============================================================================

#[test]
fn e2e_aarch64_sum_1_to_n_compile() {
    let module = build_tmir_sum_1_to_n_module();
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        ..CompilerConfig::default()
    });

    let result = compiler
        .compile(&module)
        .expect("sum_1_to_n compilation should succeed");

    assert!(!result.object_code.is_empty());
    assert_eq!(result.metrics.function_count, 1);

    let obj = &result.object_code;
    let magic = u32::from_le_bytes([obj[0], obj[1], obj[2], obj[3]]);
    assert_eq!(magic, 0xFEED_FACF);

    eprintln!(
        "sum_1_to_n: {} bytes, {} instructions",
        result.metrics.code_size_bytes,
        result.metrics.instruction_count
    );
}

// =============================================================================
// Test 18: sum_1_to_n(n) -- link and run
// =============================================================================

#[test]
fn e2e_aarch64_sum_1_to_n_link_and_run() {
    let module = build_tmir_sum_1_to_n_module();
    let obj_bytes = compile_tmir_module_to_obj(&module);

    // sum(1..=n) = n*(n+1)/2
    let driver = r#"
#include <stdio.h>

extern long _sum_1_to_n(long n);

int main(void) {
    long r0  = _sum_1_to_n(0);    /* 0 */
    long r1  = _sum_1_to_n(1);    /* 1 */
    long r5  = _sum_1_to_n(5);    /* 15 */
    long r10 = _sum_1_to_n(10);   /* 55 */
    long r100 = _sum_1_to_n(100); /* 5050 */
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

    let (exit_code, stdout) = link_and_run("sum_1_to_n", "sum_1_to_n", &obj_bytes, driver);
    eprintln!("sum_1_to_n link+run stdout: {}", stdout.trim());
    assert_eq!(
        exit_code, 0,
        "sum_1_to_n link+run failed (exit {}). \
         1=sum(0)!=0, 2=sum(1)!=1, 3=sum(5)!=15, 4=sum(10)!=55, 5=sum(100)!=5050. \
         stdout: {}",
        exit_code, stdout
    );
}

// =============================================================================
// Test 19: multi-block functions at multiple optimization levels
// =============================================================================

#[test]
fn e2e_aarch64_multiblock_all_opt_levels() {
    // Verify that multi-block functions compile at all opt levels
    let modules: &[(&str, tmir_func::Module)] = &[
        ("max_val", build_tmir_max_module()),
        ("abs_val", build_tmir_abs_module()),
        ("fibonacci", build_tmir_fibonacci_module()),
        ("sum_1_to_n", build_tmir_sum_1_to_n_module()),
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
                "{} at {:?} produced empty object code",
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
                "  {} {:?}: {} bytes",
                name, opt, result.metrics.code_size_bytes
            );
        }
    }
}

// =============================================================================
// Cross-function call tests (BRANCH26 relocation)
//
// Part of #241 -- BL relocation for cross-function calls
// =============================================================================

// ---------------------------------------------------------------------------
// Builder: two functions where caller BLs to callee
//
// fn _callee(x: i32) -> i32 { x + 10 }
// fn _caller(x: i32) -> i32 { _callee(x) }
//
// The key: _caller uses a BL instruction with a Symbol operand targeting
// _callee. When compiled into one .o via compile_module(), this BL must
// get an ARM64_RELOC_BRANCH26 relocation so the linker patches the offset.
// ---------------------------------------------------------------------------

fn build_tmir_cross_call_module() -> tmir_func::Module {
    use tmir_instrs::{BinOp, Instr, InstrNode, Operand};
    use tmir_types::{Ty, ValueId};

    // fn _callee(x: i32) -> i32 { x + 10 }
    let callee = TmirFunction {
        id: FuncId(0),
        name: "_callee".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(32)],
            returns: vec![Ty::Int(32)],
        },
        entry: BlockId(0),
        blocks: vec![TmirBlock {
            id: BlockId(0),
            params: vec![(ValueId(0), Ty::Int(32))],
            body: vec![
                InstrNode {
                    instr: Instr::Const {
                        ty: Ty::Int(32),
                        value: 10,
                    },
                    results: vec![ValueId(1)],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::BinOp {
                        op: BinOp::Add,
                        ty: Ty::Int(32),
                        lhs: Operand::Value(ValueId(0)),
                        rhs: Operand::Value(ValueId(1)),
                    },
                    results: vec![ValueId(2)],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Return {
                        values: vec![Operand::Value(ValueId(2))],
                    },
                    results: vec![],
                    proofs: vec![],
                },
            ],
        }],
        proofs: vec![],
    };

    // fn _caller(x: i32) -> i32 { _callee(x) }
    let caller = TmirFunction {
        id: FuncId(1),
        name: "_caller".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(32)],
            returns: vec![Ty::Int(32)],
        },
        entry: BlockId(0),
        blocks: vec![TmirBlock {
            id: BlockId(0),
            params: vec![(ValueId(0), Ty::Int(32))],
            body: vec![
                InstrNode {
                    instr: Instr::Call {
                        func: FuncId(0), // calls _callee
                        args: vec![Operand::Value(ValueId(0))],
                        ret_ty: vec![Ty::Int(32)],
                    },
                    results: vec![ValueId(1)],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Return {
                        values: vec![Operand::Value(ValueId(1))],
                    },
                    results: vec![],
                    proofs: vec![],
                },
            ],
        }],
        proofs: vec![],
    };

    tmir_func::Module {
        name: "e2e_cross_call_test".to_string(),
        functions: vec![callee, caller],
        structs: vec![],
    }
}

// =============================================================================
// Test 20: Cross-function call -- compile to valid Mach-O with both symbols
// =============================================================================

#[test]
fn e2e_aarch64_cross_call_compile() {
    let module = build_tmir_cross_call_module();
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        trace_level: CompilerTraceLevel::Full,
        ..CompilerConfig::default()
    });

    let result = compiler.compile(&module).expect("cross-call compilation should succeed");

    assert!(
        !result.object_code.is_empty(),
        "cross-call must produce non-empty object code"
    );
    // Module has 2 functions
    assert_eq!(result.metrics.function_count, 2);

    // Valid Mach-O magic
    let obj = &result.object_code;
    let magic = u32::from_le_bytes([obj[0], obj[1], obj[2], obj[3]]);
    assert_eq!(magic, 0xFEED_FACF, "must be valid Mach-O");

    // Write to temp and verify both symbols are present via nm
    let obj_path = write_temp_file("cross_call", ".o", obj);
    let nm_out = Command::new("nm")
        .args([obj_path.to_str().unwrap()])
        .output()
        .expect("nm should be available");
    let nm_stdout = String::from_utf8_lossy(&nm_out.stdout);

    assert!(
        nm_stdout.contains("__callee"),
        "nm should show __callee symbol. Got:\n{}",
        nm_stdout
    );
    assert!(
        nm_stdout.contains("__caller"),
        "nm should show __caller symbol. Got:\n{}",
        nm_stdout
    );

    // Verify relocations via otool -r (should show BRANCH26)
    let reloc_out = Command::new("otool")
        .args(["-r", obj_path.to_str().unwrap()])
        .output()
        .expect("otool -r should be available");
    let reloc_stdout = String::from_utf8_lossy(&reloc_out.stdout);

    eprintln!("nm output:\n{}", nm_stdout);
    eprintln!("otool -r output:\n{}", reloc_stdout);
    eprintln!(
        "cross-call: {} bytes, {} functions",
        result.metrics.code_size_bytes,
        result.metrics.function_count
    );

    // The relocation output should contain a BRANCH26 entry for the BL.
    // otool -r shows ARM64_RELOC_BRANCH26 as type value "2" in numeric output.
    // Check for both the symbolic name and the numeric type.
    let has_branch26_reloc = reloc_stdout.contains("ARM64_RELOC_BRANCH26")
        || reloc_stdout.contains("BRANCH26")
        // otool -r numeric format: columns are
        //   address pcrel length extern type scattered symbolnum
        // For BRANCH26: pcrel=1, length=2, extern=1, type=2
        || (reloc_stdout.contains("Relocation information")
            && reloc_stdout.contains("1     2      1      2"));
    assert!(
        has_branch26_reloc,
        "otool -r should show ARM64_RELOC_BRANCH26 (type 2) for the cross-function BL. Got:\n{}",
        reloc_stdout
    );
}

// =============================================================================
// Test 21: Cross-function call -- link and run
//
// fn _callee(x: i32) -> i32 { x + 10 }
// fn _caller(x: i32) -> i32 { _callee(x) }
//
// C driver calls _caller(32), expects 42 (32 + 10).
// =============================================================================

#[test]
fn e2e_aarch64_cross_call_link_and_run() {
    let module = build_tmir_cross_call_module();
    let obj_bytes = compile_tmir_module_to_obj(&module);

    let driver = r#"
#include <stdio.h>

extern int _caller(int x);

int main(void) {
    int r1 = _caller(32);
    int r2 = _caller(0);
    int r3 = _caller(-10);
    printf("caller(32)=%d caller(0)=%d caller(-10)=%d\n", r1, r2, r3);
    if (r1 != 42) return 1;
    if (r2 != 10) return 2;
    if (r3 != 0) return 3;
    return 0;
}
"#;

    let (exit_code, stdout) = link_and_run("cross_call", "cross_call", &obj_bytes, driver);
    eprintln!("cross_call link+run stdout: {}", stdout.trim());
    assert_eq!(
        exit_code, 0,
        "cross_call link+run failed (exit {}). \
         1=caller(32)!=42, 2=caller(0)!=10, 3=caller(-10)!=0. \
         stdout: {}",
        exit_code, stdout
    );
}

// =============================================================================
// Multi-block E2E tests adapted to the Operand model
//
// Part of #242 -- if/else, loop, nested conditional, factorial
// =============================================================================

// ---------------------------------------------------------------------------
// Builder: classify(x) -- 3-way if/else (sign classification)
//
// fn classify(x: i64) -> i64 {
//     if x < 0 { return -1 }
//     if x == 0 { return 0 }
//     return 1
// }
//
// bb0 (entry): cmp x < 0, condbr -> bb1 (neg), bb2 (check_zero)
// bb1 (neg): return -1
// bb2 (check_zero): cmp x == 0, condbr -> bb3 (zero), bb4 (pos)
// bb3 (zero): return 0
// bb4 (pos): return 1
// ---------------------------------------------------------------------------

fn build_tmir_classify_module() -> tmir_func::Module {
    use tmir_instrs::{Instr, InstrNode};
    use tmir_types::{Ty, ValueId};

    let func = TmirFunction {
        id: FuncId(0),
        name: "_classify".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(64)],
            returns: vec![Ty::Int(64)],
        },
        entry: BlockId(0),
        blocks: vec![
            // bb0 (entry): check x < 0
            TmirBlock {
                id: BlockId(0),
                params: vec![(ValueId(0), Ty::Int(64))], // x
                body: vec![
                    // const 0 for comparison
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 0,
                        },
                        results: vec![ValueId(1)], // zero
                        proofs: vec![],
                    },
                    // cmp x < 0
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Slt,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(0)),
                            rhs: Operand::Value(ValueId(1)),
                        },
                        results: vec![ValueId(2)],
                        proofs: vec![],
                    },
                    // if x < 0 -> bb1 (neg), else -> bb2 (check_zero)
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: Operand::Value(ValueId(2)),
                            then_target: BlockId(1),
                            then_args: vec![],
                            else_target: BlockId(2),
                            else_args: vec![],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb1 (neg): return -1
            TmirBlock {
                id: BlockId(1),
                params: vec![],
                body: vec![
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: -1_i64 as i64,
                        },
                        results: vec![ValueId(10)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Return {
                            values: vec![Operand::Value(ValueId(10))],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb2 (check_zero): cmp x == 0
            TmirBlock {
                id: BlockId(2),
                params: vec![],
                body: vec![
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Eq,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(0)),
                            rhs: Operand::Value(ValueId(1)), // zero from bb0
                        },
                        results: vec![ValueId(20)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: Operand::Value(ValueId(20)),
                            then_target: BlockId(3), // zero
                            then_args: vec![],
                            else_target: BlockId(4), // pos
                            else_args: vec![],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb3 (zero): return 0
            TmirBlock {
                id: BlockId(3),
                params: vec![],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![Operand::Value(ValueId(1))], // reuse zero from bb0
                    },
                    results: vec![],
                    proofs: vec![],
                }],
            },
            // bb4 (pos): return 1
            TmirBlock {
                id: BlockId(4),
                params: vec![],
                body: vec![
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 1,
                        },
                        results: vec![ValueId(30)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Return {
                            values: vec![Operand::Value(ValueId(30))],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
        ],
        proofs: vec![],
    };

    tmir_func::Module {
        name: "e2e_classify_test".to_string(),
        functions: vec![func],
        structs: vec![],
    }
}

// ---------------------------------------------------------------------------
// Builder: sum_to_n(n) -- loop with accumulator
//
// fn sum_to_n(n: i64) -> i64 {
//     let mut acc = 0;
//     let mut i = 1;
//     while i <= n { acc += i; i += 1; }
//     return acc;
// }
//
// bb0 (entry): acc=0, i=1, br -> bb1
// bb1 (loop header): params(acc, i), cmp i <= n, condbr -> bb2 (body), bb3 (exit)
// bb2 (body): new_acc = acc + i, new_i = i + 1, br -> bb1(new_acc, new_i)
// bb3 (exit): return acc
// ---------------------------------------------------------------------------

fn build_tmir_sum_to_n_module() -> tmir_func::Module {
    use tmir_instrs::{BinOp, Instr, InstrNode};
    use tmir_types::{Ty, ValueId};

    let func = TmirFunction {
        id: FuncId(0),
        name: "_sum_to_n".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(64)],
            returns: vec![Ty::Int(64)],
        },
        entry: BlockId(0),
        blocks: vec![
            // bb0 (entry): acc=0, i=1, jump to loop
            TmirBlock {
                id: BlockId(0),
                params: vec![(ValueId(0), Ty::Int(64))], // n
                body: vec![
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 0,
                        },
                        results: vec![ValueId(1)], // acc_init = 0
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 1,
                        },
                        results: vec![ValueId(2)], // i_init = 1
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Br {
                            target: BlockId(1),
                            args: vec![
                                Operand::Value(ValueId(1)),
                                Operand::Value(ValueId(2)),
                            ],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb1 (loop header): params(acc, i), check i <= n
            TmirBlock {
                id: BlockId(1),
                params: vec![
                    (ValueId(10), Ty::Int(64)), // acc
                    (ValueId(11), Ty::Int(64)), // i
                ],
                body: vec![
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Sle,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(11)), // i
                            rhs: Operand::Value(ValueId(0)),  // n
                        },
                        results: vec![ValueId(12)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: Operand::Value(ValueId(12)),
                            then_target: BlockId(2), // body
                            then_args: vec![],
                            else_target: BlockId(3), // exit
                            else_args: vec![],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb2 (body): acc += i, i += 1, back to loop header
            TmirBlock {
                id: BlockId(2),
                params: vec![],
                body: vec![
                    // new_acc = acc + i
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Add,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(10)), // acc
                            rhs: Operand::Value(ValueId(11)), // i
                        },
                        results: vec![ValueId(20)], // new_acc
                        proofs: vec![],
                    },
                    // new_i = i + 1
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 1,
                        },
                        results: vec![ValueId(21)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Add,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(11)), // i
                            rhs: Operand::Value(ValueId(21)), // 1
                        },
                        results: vec![ValueId(22)], // new_i
                        proofs: vec![],
                    },
                    // br -> bb1(new_acc, new_i)
                    InstrNode {
                        instr: Instr::Br {
                            target: BlockId(1),
                            args: vec![
                                Operand::Value(ValueId(20)),
                                Operand::Value(ValueId(22)),
                            ],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb3 (exit): return acc
            TmirBlock {
                id: BlockId(3),
                params: vec![],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![Operand::Value(ValueId(10))], // acc from bb1
                    },
                    results: vec![],
                    proofs: vec![],
                }],
            },
        ],
        proofs: vec![],
    };

    tmir_func::Module {
        name: "e2e_sum_to_n_test".to_string(),
        functions: vec![func],
        structs: vec![],
    }
}

// ---------------------------------------------------------------------------
// Builder: clamp(x, lo, hi) -- nested conditional (5+ blocks)
//
// fn clamp(x: i64, lo: i64, hi: i64) -> i64 {
//     if x < lo { return lo }
//     if x > hi { return hi }
//     return x
// }
//
// bb0 (entry): cmp x < lo, condbr -> bb1 (ret_lo), bb2 (check_hi)
// bb1 (ret_lo): return lo
// bb2 (check_hi): cmp x > hi, condbr -> bb3 (ret_hi), bb4 (ret_x)
// bb3 (ret_hi): return hi
// bb4 (ret_x): return x
// ---------------------------------------------------------------------------

fn build_tmir_clamp_module() -> tmir_func::Module {
    use tmir_instrs::{Instr, InstrNode};
    use tmir_types::{Ty, ValueId};

    let func = TmirFunction {
        id: FuncId(0),
        name: "_clamp".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(64), Ty::Int(64), Ty::Int(64)],
            returns: vec![Ty::Int(64)],
        },
        entry: BlockId(0),
        blocks: vec![
            // bb0 (entry): cmp x < lo
            TmirBlock {
                id: BlockId(0),
                params: vec![
                    (ValueId(0), Ty::Int(64)), // x
                    (ValueId(1), Ty::Int(64)), // lo
                    (ValueId(2), Ty::Int(64)), // hi
                ],
                body: vec![
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Slt,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(0)), // x
                            rhs: Operand::Value(ValueId(1)), // lo
                        },
                        results: vec![ValueId(3)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: Operand::Value(ValueId(3)),
                            then_target: BlockId(1), // ret_lo
                            then_args: vec![],
                            else_target: BlockId(2), // check_hi
                            else_args: vec![],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb1 (ret_lo): return lo
            TmirBlock {
                id: BlockId(1),
                params: vec![],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![Operand::Value(ValueId(1))], // lo
                    },
                    results: vec![],
                    proofs: vec![],
                }],
            },
            // bb2 (check_hi): cmp x > hi
            TmirBlock {
                id: BlockId(2),
                params: vec![],
                body: vec![
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Sgt,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(0)), // x
                            rhs: Operand::Value(ValueId(2)), // hi
                        },
                        results: vec![ValueId(10)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: Operand::Value(ValueId(10)),
                            then_target: BlockId(3), // ret_hi
                            then_args: vec![],
                            else_target: BlockId(4), // ret_x
                            else_args: vec![],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb3 (ret_hi): return hi
            TmirBlock {
                id: BlockId(3),
                params: vec![],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![Operand::Value(ValueId(2))], // hi
                    },
                    results: vec![],
                    proofs: vec![],
                }],
            },
            // bb4 (ret_x): return x
            TmirBlock {
                id: BlockId(4),
                params: vec![],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![Operand::Value(ValueId(0))], // x
                    },
                    results: vec![],
                    proofs: vec![],
                }],
            },
        ],
        proofs: vec![],
    };

    tmir_func::Module {
        name: "e2e_clamp_test".to_string(),
        functions: vec![func],
        structs: vec![],
    }
}

// ---------------------------------------------------------------------------
// Builder: factorial(n) -- loop with multiply
//
// fn factorial(n: i64) -> i64 {
//     if n <= 1 { return 1 }
//     let mut acc = 1;
//     let mut i = 2;
//     while i <= n { acc *= i; i += 1; }
//     return acc;
// }
//
// bb0 (entry): cmp n <= 1, condbr -> bb1 (ret_1), bb2 (loop_init)
// bb1 (ret_1): return 1
// bb2 (loop_init): acc=1, i=2, br -> bb3
// bb3 (loop): params(acc, i), cmp i <= n, condbr -> bb4 (body), bb5 (exit)
// bb4 (body): new_acc = acc * i, new_i = i + 1, br -> bb3(new_acc, new_i)
// bb5 (exit): return acc
// ---------------------------------------------------------------------------

fn build_tmir_factorial_module() -> tmir_func::Module {
    use tmir_instrs::{BinOp, Instr, InstrNode};
    use tmir_types::{Ty, ValueId};

    let func = TmirFunction {
        id: FuncId(0),
        name: "_factorial".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(64)],
            returns: vec![Ty::Int(64)],
        },
        entry: BlockId(0),
        blocks: vec![
            // bb0 (entry): check n <= 1
            TmirBlock {
                id: BlockId(0),
                params: vec![(ValueId(0), Ty::Int(64))], // n
                body: vec![
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 1,
                        },
                        results: vec![ValueId(1)], // const_1
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Sle,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(0)), // n
                            rhs: Operand::Value(ValueId(1)), // 1
                        },
                        results: vec![ValueId(2)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: Operand::Value(ValueId(2)),
                            then_target: BlockId(1), // ret_1
                            then_args: vec![],
                            else_target: BlockId(2), // loop_init
                            else_args: vec![],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb1 (ret_1): return 1
            TmirBlock {
                id: BlockId(1),
                params: vec![],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![Operand::Value(ValueId(1))], // const_1 from bb0
                    },
                    results: vec![],
                    proofs: vec![],
                }],
            },
            // bb2 (loop_init): acc=1, i=2, jump to loop
            TmirBlock {
                id: BlockId(2),
                params: vec![],
                body: vec![
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 1,
                        },
                        results: vec![ValueId(10)], // acc_init = 1
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 2,
                        },
                        results: vec![ValueId(11)], // i_init = 2
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Br {
                            target: BlockId(3),
                            args: vec![
                                Operand::Value(ValueId(10)),
                                Operand::Value(ValueId(11)),
                            ],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb3 (loop header): params(acc, i), check i <= n
            TmirBlock {
                id: BlockId(3),
                params: vec![
                    (ValueId(20), Ty::Int(64)), // acc
                    (ValueId(21), Ty::Int(64)), // i
                ],
                body: vec![
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Sle,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(21)), // i
                            rhs: Operand::Value(ValueId(0)),  // n
                        },
                        results: vec![ValueId(22)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: Operand::Value(ValueId(22)),
                            then_target: BlockId(4), // body
                            then_args: vec![],
                            else_target: BlockId(5), // exit
                            else_args: vec![],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb4 (body): acc *= i, i += 1, back to loop header
            TmirBlock {
                id: BlockId(4),
                params: vec![],
                body: vec![
                    // new_acc = acc * i
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Mul,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(20)), // acc
                            rhs: Operand::Value(ValueId(21)), // i
                        },
                        results: vec![ValueId(30)], // new_acc
                        proofs: vec![],
                    },
                    // new_i = i + 1
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 1,
                        },
                        results: vec![ValueId(31)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Add,
                            ty: Ty::Int(64),
                            lhs: Operand::Value(ValueId(21)), // i
                            rhs: Operand::Value(ValueId(31)), // 1
                        },
                        results: vec![ValueId(32)], // new_i
                        proofs: vec![],
                    },
                    // br -> bb3(new_acc, new_i)
                    InstrNode {
                        instr: Instr::Br {
                            target: BlockId(3),
                            args: vec![
                                Operand::Value(ValueId(30)),
                                Operand::Value(ValueId(32)),
                            ],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb5 (exit): return acc
            TmirBlock {
                id: BlockId(5),
                params: vec![],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![Operand::Value(ValueId(20))], // acc from bb3
                    },
                    results: vec![],
                    proofs: vec![],
                }],
            },
        ],
        proofs: vec![],
    };

    tmir_func::Module {
        name: "e2e_factorial_test".to_string(),
        functions: vec![func],
        structs: vec![],
    }
}

// =============================================================================
// Test 22: classify(x) -- if/else 3-way branch, compile to valid Mach-O
// =============================================================================

#[test]
fn e2e_aarch64_classify_compile() {
    let module = build_tmir_classify_module();
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        trace_level: CompilerTraceLevel::Full,
        ..CompilerConfig::default()
    });

    let result = compiler
        .compile(&module)
        .expect("classify compilation should succeed");

    assert!(
        !result.object_code.is_empty(),
        "classify must produce non-empty object code"
    );
    assert_eq!(result.metrics.function_count, 1);

    let obj = &result.object_code;
    let magic = u32::from_le_bytes([obj[0], obj[1], obj[2], obj[3]]);
    assert_eq!(magic, 0xFEED_FACF, "must be valid Mach-O");

    eprintln!(
        "classify: {} bytes, {} instructions",
        result.metrics.code_size_bytes,
        result.metrics.instruction_count
    );
}

// =============================================================================
// Test 23: classify(x) -- link and run
// =============================================================================

#[test]
fn e2e_aarch64_classify_link_and_run() {
    let module = build_tmir_classify_module();
    let obj_bytes = compile_tmir_module_to_obj(&module);

    let driver = r#"
#include <stdio.h>

extern long _classify(long x);

int main(void) {
    long r1 = _classify(-100);
    long r2 = _classify(-1);
    long r3 = _classify(0);
    long r4 = _classify(1);
    long r5 = _classify(999);
    printf("classify(-100)=%ld classify(-1)=%ld classify(0)=%ld classify(1)=%ld classify(999)=%ld\n",
           r1, r2, r3, r4, r5);
    if (r1 != -1) return 1;
    if (r2 != -1) return 2;
    if (r3 != 0)  return 3;
    if (r4 != 1)  return 4;
    if (r5 != 1)  return 5;
    return 0;
}
"#;

    let (exit_code, stdout) = link_and_run("classify", "classify", &obj_bytes, driver);
    eprintln!("classify link+run stdout: {}", stdout.trim());
    assert_eq!(
        exit_code, 0,
        "classify link+run failed (exit {}). \
         1=classify(-100)!=-1, 2=classify(-1)!=-1, 3=classify(0)!=0, \
         4=classify(1)!=1, 5=classify(999)!=1. stdout: {}",
        exit_code, stdout
    );
}

// =============================================================================
// Test 24: sum_to_n(n) -- loop with accumulator, compile to valid Mach-O
// =============================================================================

#[test]
fn e2e_aarch64_sum_to_n_compile() {
    let module = build_tmir_sum_to_n_module();
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        ..CompilerConfig::default()
    });

    let result = compiler
        .compile(&module)
        .expect("sum_to_n compilation should succeed");

    assert!(!result.object_code.is_empty());
    assert_eq!(result.metrics.function_count, 1);

    let obj = &result.object_code;
    let magic = u32::from_le_bytes([obj[0], obj[1], obj[2], obj[3]]);
    assert_eq!(magic, 0xFEED_FACF);

    eprintln!(
        "sum_to_n: {} bytes, {} instructions",
        result.metrics.code_size_bytes,
        result.metrics.instruction_count
    );
}

// =============================================================================
// Test 25: sum_to_n(n) -- link and run
// =============================================================================

#[test]
fn e2e_aarch64_sum_to_n_link_and_run() {
    let module = build_tmir_sum_to_n_module();
    let obj_bytes = compile_tmir_module_to_obj(&module);

    // sum(1..=n) = n*(n+1)/2
    let driver = r#"
#include <stdio.h>

extern long _sum_to_n(long n);

int main(void) {
    long r0  = _sum_to_n(0);    /* 0 */
    long r1  = _sum_to_n(1);    /* 1 */
    long r5  = _sum_to_n(5);    /* 15 */
    long r10 = _sum_to_n(10);   /* 55 */
    long r100 = _sum_to_n(100); /* 5050 */
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

    let (exit_code, stdout) = link_and_run("sum_to_n", "sum_to_n", &obj_bytes, driver);
    eprintln!("sum_to_n link+run stdout: {}", stdout.trim());
    assert_eq!(
        exit_code, 0,
        "sum_to_n link+run failed (exit {}). \
         1=sum(0)!=0, 2=sum(1)!=1, 3=sum(5)!=15, 4=sum(10)!=55, 5=sum(100)!=5050. \
         stdout: {}",
        exit_code, stdout
    );
}

// =============================================================================
// Test 26: clamp(x, lo, hi) -- nested conditional (5 blocks), compile
// =============================================================================

#[test]
fn e2e_aarch64_clamp_compile() {
    let module = build_tmir_clamp_module();
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        trace_level: CompilerTraceLevel::Full,
        ..CompilerConfig::default()
    });

    let result = compiler
        .compile(&module)
        .expect("clamp compilation should succeed");

    assert!(
        !result.object_code.is_empty(),
        "clamp must produce non-empty object code"
    );
    assert_eq!(result.metrics.function_count, 1);

    let obj = &result.object_code;
    let magic = u32::from_le_bytes([obj[0], obj[1], obj[2], obj[3]]);
    assert_eq!(magic, 0xFEED_FACF, "must be valid Mach-O");

    eprintln!(
        "clamp: {} bytes, {} instructions",
        result.metrics.code_size_bytes,
        result.metrics.instruction_count
    );
}

// =============================================================================
// Test 27: clamp(x, lo, hi) -- link and run
// =============================================================================

#[test]
fn e2e_aarch64_clamp_link_and_run() {
    let module = build_tmir_clamp_module();
    let obj_bytes = compile_tmir_module_to_obj(&module);

    let driver = r#"
#include <stdio.h>

extern long _clamp(long x, long lo, long hi);

int main(void) {
    long r1 = _clamp(5, 0, 10);     /* 5 (in range) */
    long r2 = _clamp(-5, 0, 10);    /* 0 (below lo) */
    long r3 = _clamp(15, 0, 10);    /* 10 (above hi) */
    long r4 = _clamp(0, 0, 10);     /* 0 (at lo boundary) */
    long r5 = _clamp(10, 0, 10);    /* 10 (at hi boundary) */
    long r6 = _clamp(-100, -50, 50); /* -50 (below negative lo) */
    long r7 = _clamp(100, -50, 50);  /* 50 (above positive hi) */
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

    let (exit_code, stdout) = link_and_run("clamp", "clamp", &obj_bytes, driver);
    eprintln!("clamp link+run stdout: {}", stdout.trim());
    assert_eq!(
        exit_code, 0,
        "clamp link+run failed (exit {}). \
         1=clamp(5,0,10)!=5, 2=clamp(-5,0,10)!=0, 3=clamp(15,0,10)!=10, \
         4=clamp(0,0,10)!=0, 5=clamp(10,0,10)!=10, \
         6=clamp(-100,-50,50)!=-50, 7=clamp(100,-50,50)!=50. stdout: {}",
        exit_code, stdout
    );
}

// =============================================================================
// Test 28: factorial(n) -- loop with multiply, compile to valid Mach-O
// =============================================================================

#[test]
fn e2e_aarch64_factorial_compile() {
    let module = build_tmir_factorial_module();
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        trace_level: CompilerTraceLevel::Full,
        ..CompilerConfig::default()
    });

    let result = compiler
        .compile(&module)
        .expect("factorial compilation should succeed");

    assert!(
        !result.object_code.is_empty(),
        "factorial must produce non-empty object code"
    );
    assert_eq!(result.metrics.function_count, 1);

    let obj = &result.object_code;
    let magic = u32::from_le_bytes([obj[0], obj[1], obj[2], obj[3]]);
    assert_eq!(magic, 0xFEED_FACF, "must be valid Mach-O");

    // Multi-block function with loop should produce substantial code
    assert!(
        result.metrics.code_size_bytes > 50,
        "factorial (6 blocks, loop) should produce substantial code, got {} bytes",
        result.metrics.code_size_bytes
    );

    let trace = result.trace.expect("trace should be present");
    eprintln!(
        "factorial: {} bytes, {} instructions, {} trace entries",
        result.metrics.code_size_bytes,
        result.metrics.instruction_count,
        trace.entries.len()
    );
}

// =============================================================================
// Test 29: factorial(n) -- link and run
// =============================================================================

#[test]
fn e2e_aarch64_factorial_link_and_run() {
    let module = build_tmir_factorial_module();
    let obj_bytes = compile_tmir_module_to_obj(&module);

    // 0! = 1, 1! = 1, 5! = 120, 10! = 3628800, 12! = 479001600, 20! = 2432902008176640000
    let driver = r#"
#include <stdio.h>

extern long _factorial(long n);

int main(void) {
    long r0  = _factorial(0);   /* 1 */
    long r1  = _factorial(1);   /* 1 */
    long r5  = _factorial(5);   /* 120 */
    long r10 = _factorial(10);  /* 3628800 */
    long r12 = _factorial(12);  /* 479001600 */
    long r20 = _factorial(20);  /* 2432902008176640000 */
    printf("fact(0)=%ld fact(1)=%ld fact(5)=%ld fact(10)=%ld fact(12)=%ld fact(20)=%ld\n",
           r0, r1, r5, r10, r12, r20);
    if (r0 != 1)          return 1;
    if (r1 != 1)          return 2;
    if (r5 != 120)        return 3;
    if (r10 != 3628800)   return 4;
    if (r12 != 479001600) return 5;
    if (r20 != 2432902008176640000L) return 6;
    return 0;
}
"#;

    let (exit_code, stdout) = link_and_run("factorial", "factorial", &obj_bytes, driver);
    eprintln!("factorial link+run stdout: {}", stdout.trim());
    assert_eq!(
        exit_code, 0,
        "factorial link+run failed (exit {}). \
         1=fact(0)!=1, 2=fact(1)!=1, 3=fact(5)!=120, 4=fact(10)!=3628800, \
         5=fact(12)!=479001600, 6=fact(20)!=2432902008176640000. stdout: {}",
        exit_code, stdout
    );
}

// =============================================================================
// Test 30: new multi-block functions at all optimization levels
// =============================================================================

#[test]
fn e2e_aarch64_new_multiblock_all_opt_levels() {
    let modules: &[(&str, tmir_func::Module)] = &[
        ("classify", build_tmir_classify_module()),
        ("sum_to_n", build_tmir_sum_to_n_module()),
        ("clamp", build_tmir_clamp_module()),
        ("factorial", build_tmir_factorial_module()),
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
                "{} at {:?} produced empty object code",
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
                "  {} {:?}: {} bytes",
                name, opt, result.metrics.code_size_bytes
            );
        }
    }
}
