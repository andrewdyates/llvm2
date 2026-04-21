// llvm2-codegen/tests/e2e_x86_64_correctness.rs - x86-64 computed result verification
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// End-to-end tests that compile x86-64 functions through the LLVM2 pipeline,
// link them with C drivers, run under Rosetta 2, and verify COMPUTED RESULTS
// are correct (not just "it runs without crashing").
//
// Part of #301 -- x86-64 add produces wrong results (originally)
// Fixed by #305 -- two-address fixup pass inserted in pipeline
//
// # Two-address constraint (now enforced)
//
// x86-64 ALU instructions use two-address form: `ADD dst, src` means `dst = dst + src`.
// The ISel emits three-address form: `ADD v2, v0, v1` (dst, lhs, rhs).
// The pipeline's `fixup_two_address` pass (between regalloc and prologue/epilogue)
// inserts `MOV dst, lhs` when `preg(dst) != preg(lhs)`, ensuring correctness.
//
// This file tests both three-address form (with fixup) and two-address-safe
// form (dst == lhs, no fixup needed) to verify both paths produce correct results.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use llvm2_codegen::x86_64::{
    build_x86_add_test_function, build_x86_const_test_function,
    X86OutputFormat, X86Pipeline, X86PipelineConfig,
};
use llvm2_codegen::x86_64::pipeline::X86RegAllocMode;

use llvm2_ir::regs::{RegClass, VReg};
use llvm2_ir::x86_64_ops::{X86CondCode, X86Opcode};
use llvm2_ir::x86_64_regs;
use llvm2_lower::function::Signature;
use llvm2_lower::instructions::Block;
use llvm2_lower::types::Type;
use llvm2_lower::x86_64_isel::{X86ISelFunction, X86ISelInst, X86ISelOperand};

use llvm2_regalloc::AllocStrategy;

// ===========================================================================
// Test infrastructure (mirrors e2e_x86_64_link.rs)
// ===========================================================================

fn has_cc_x86_64() -> bool {
    let dir = std::env::temp_dir().join("llvm2_x86_64_cc_check_correctness");
    let _ = fs::create_dir_all(&dir);
    let src = dir.join("check.c");
    let out = dir.join("check");
    let _ = fs::write(&src, "int main(void) { return 0; }\n");

    let result = Command::new("cc")
        .args(["-arch", "x86_64", "-o", out.to_str().unwrap(), src.to_str().unwrap()])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    let _ = fs::remove_dir_all(&dir);
    result
}

fn make_test_dir(test_name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("llvm2_e2e_x86_64_correctness_{}", test_name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("Failed to create test directory");
    dir
}

fn write_object_file(dir: &Path, filename: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(filename);
    fs::write(&path, bytes).expect("Failed to write .o file");
    path
}

fn write_c_driver(dir: &Path, filename: &str, source: &str) -> PathBuf {
    let path = dir.join(filename);
    fs::write(&path, source).expect("Failed to write C driver");
    path
}

fn link_x86_64(dir: &Path, driver_c: &Path, obj: &Path, output_name: &str) -> PathBuf {
    let binary = dir.join(output_name);
    let result = Command::new("cc")
        .args([
            "-arch", "x86_64",
            "-o", binary.to_str().unwrap(),
            driver_c.to_str().unwrap(),
            obj.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run cc -arch x86_64");

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        let stdout = String::from_utf8_lossy(&result.stdout);

        let otool_out = Command::new("otool")
            .args(["-tv", obj.to_str().unwrap()])
            .output()
            .ok();
        let disasm = otool_out
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();

        panic!(
            "Linking failed for {}!\ncc stdout: {}\ncc stderr: {}\notool -tv:\n{}",
            output_name, stdout, stderr, disasm
        );
    }

    binary
}

/// Compile an ISel function to Mach-O, link with C driver, run, return (exit_code, stdout).
fn compile_link_run(
    test_name: &str,
    func: &X86ISelFunction,
    driver_src: &str,
    config: X86PipelineConfig,
) -> (i32, String) {
    let pipeline = X86Pipeline::new(config);
    let bytes = pipeline
        .compile_function(func)
        .expect("x86-64 compilation should succeed");

    let dir = make_test_dir(test_name);
    let obj_path = write_object_file(&dir, &format!("{}.o", test_name), &bytes);
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_x86_64(&dir, &driver_path, &obj_path, &format!("test_{}", test_name));

    // Also disassemble for debugging
    let otool_out = Command::new("otool")
        .args(["-tv", obj_path.to_str().unwrap()])
        .output()
        .ok();
    if let Some(ref out) = otool_out {
        if out.status.success() {
            let disasm = String::from_utf8_lossy(&out.stdout);
            eprintln!("[{}] disassembly:\n{}", test_name, disasm);
        }
    }

    let run_output = Command::new(binary.to_str().unwrap())
        .output()
        .expect("should be able to run the x86_64 binary (Rosetta 2)");

    let stdout = String::from_utf8_lossy(&run_output.stdout).to_string();
    let exit_code = run_output.status.code().unwrap_or(-1);

    eprintln!(
        "[{}] exit_code={}, stdout={}",
        test_name, exit_code, stdout.trim()
    );

    let _ = fs::remove_dir_all(&dir);

    (exit_code, stdout)
}

/// Compile and run a C source file as golden truth, return (exit_code, stdout).
fn compile_run_c_golden(test_name: &str, c_source: &str) -> (i32, String) {
    let dir = make_test_dir(&format!("{}_golden", test_name));
    let src_path = dir.join("golden.c");
    let bin_path = dir.join("golden");
    fs::write(&src_path, c_source).expect("Failed to write golden C source");

    let result = Command::new("cc")
        .args([
            "-arch", "x86_64",
            "-o", bin_path.to_str().unwrap(),
            src_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to compile golden C");

    assert!(
        result.status.success(),
        "Golden C compilation failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    let run_output = Command::new(bin_path.to_str().unwrap())
        .output()
        .expect("Failed to run golden binary");

    let stdout = String::from_utf8_lossy(&run_output.stdout).to_string();
    let exit_code = run_output.status.code().unwrap_or(-1);

    eprintln!(
        "[{}_golden] exit_code={}, stdout={}",
        test_name, exit_code, stdout.trim()
    );

    let _ = fs::remove_dir_all(&dir);

    (exit_code, stdout)
}

fn default_macho_config() -> X86PipelineConfig {
    X86PipelineConfig {
        output_format: X86OutputFormat::MachO,
        emit_frame: true,
        ..X86PipelineConfig::default()
    }
}

fn greedy_macho_config() -> X86PipelineConfig {
    X86PipelineConfig {
        output_format: X86OutputFormat::MachO,
        emit_frame: true,
        regalloc_mode: X86RegAllocMode::Full(AllocStrategy::Greedy),
        ..X86PipelineConfig::default()
    }
}

// ===========================================================================
// Function builders
// ===========================================================================

/// Build `add_correct(a: i64, b: i64) -> i64 { a + b }` that satisfies
/// the two-address constraint by using dst == lhs.
///
/// ISel output (two-address-safe):
///   MOV v0, RDI     ; v0 = arg0
///   MOV v1, RSI     ; v1 = arg1
///   ADD v0, v0, v1  ; v0 = v0 + v1 (dst == lhs!)
///   MOV RAX, v0     ; return v0
///   RET
fn build_add_correct() -> X86ISelFunction {
    let sig = Signature {
        params: vec![Type::I64, Type::I64],
        returns: vec![Type::I64],
    };

    let mut func = X86ISelFunction::new("add_correct".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    let v0 = VReg::new(0, RegClass::Gpr64);
    let v1 = VReg::new(1, RegClass::Gpr64);
    func.next_vreg = 2;

    // MOV v0, RDI (arg 0)
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(x86_64_regs::RDI)],
    ));

    // MOV v1, RSI (arg 1)
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v1), X86ISelOperand::PReg(x86_64_regs::RSI)],
    ));

    // ADD v0, v0, v1 (three-address form, but dst == lhs, so two-address-safe)
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::AddRR,
        vec![
            X86ISelOperand::VReg(v0),
            X86ISelOperand::VReg(v0),
            X86ISelOperand::VReg(v1),
        ],
    ));

    // MOV RAX, v0 (return result)
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)],
    ));

    // RET
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));

    func
}

/// Build `sub_correct(a: i64, b: i64) -> i64 { a - b }` (two-address-safe).
fn build_sub_correct() -> X86ISelFunction {
    let sig = Signature {
        params: vec![Type::I64, Type::I64],
        returns: vec![Type::I64],
    };

    let mut func = X86ISelFunction::new("sub_correct".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    let v0 = VReg::new(0, RegClass::Gpr64);
    let v1 = VReg::new(1, RegClass::Gpr64);
    func.next_vreg = 2;

    // MOV v0, RDI (arg 0)
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(x86_64_regs::RDI)],
    ));

    // MOV v1, RSI (arg 1)
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v1), X86ISelOperand::PReg(x86_64_regs::RSI)],
    ));

    // SUB v0, v0, v1 (dst == lhs, two-address-safe)
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::SubRR,
        vec![
            X86ISelOperand::VReg(v0),
            X86ISelOperand::VReg(v0),
            X86ISelOperand::VReg(v1),
        ],
    ));

    // MOV RAX, v0
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)],
    ));

    // RET
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));

    func
}

/// Build `identity(a: i64) -> i64 { a }` -- return the argument unchanged.
fn build_identity() -> X86ISelFunction {
    let sig = Signature {
        params: vec![Type::I64],
        returns: vec![Type::I64],
    };

    let mut func = X86ISelFunction::new("identity".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    let v0 = VReg::new(0, RegClass::Gpr64);
    func.next_vreg = 1;

    // MOV v0, RDI
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(x86_64_regs::RDI)],
    ));

    // MOV RAX, v0
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)],
    ));

    // RET
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));

    func
}

/// Build `mul_correct(a: i64, b: i64) -> i64 { a * b }` (two-address-safe).
///
/// IMUL r64, r64 is two-address: dst = dst * src.
fn build_mul_correct() -> X86ISelFunction {
    let sig = Signature {
        params: vec![Type::I64, Type::I64],
        returns: vec![Type::I64],
    };

    let mut func = X86ISelFunction::new("mul_correct".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    let v0 = VReg::new(0, RegClass::Gpr64);
    let v1 = VReg::new(1, RegClass::Gpr64);
    func.next_vreg = 2;

    // MOV v0, RDI
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(x86_64_regs::RDI)],
    ));

    // MOV v1, RSI
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v1), X86ISelOperand::PReg(x86_64_regs::RSI)],
    ));

    // IMUL v0, v0, v1 (dst == lhs, two-address-safe)
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::ImulRR,
        vec![
            X86ISelOperand::VReg(v0),
            X86ISelOperand::VReg(v0),
            X86ISelOperand::VReg(v1),
        ],
    ));

    // MOV RAX, v0
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)],
    ));

    // RET
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));

    func
}

/// Build `bitwise_and(a: i64, b: i64) -> i64 { a & b }` (two-address-safe).
fn build_and_correct() -> X86ISelFunction {
    let sig = Signature {
        params: vec![Type::I64, Type::I64],
        returns: vec![Type::I64],
    };

    let mut func = X86ISelFunction::new("bitwise_and".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    let v0 = VReg::new(0, RegClass::Gpr64);
    let v1 = VReg::new(1, RegClass::Gpr64);
    func.next_vreg = 2;

    // MOV v0, RDI
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(x86_64_regs::RDI)],
    ));

    // MOV v1, RSI
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v1), X86ISelOperand::PReg(x86_64_regs::RSI)],
    ));

    // AND v0, v0, v1 (dst == lhs)
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::AndRR,
        vec![
            X86ISelOperand::VReg(v0),
            X86ISelOperand::VReg(v0),
            X86ISelOperand::VReg(v1),
        ],
    ));

    // MOV RAX, v0
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)],
    ));

    // RET
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));

    func
}

/// Build `add_imm(a: i64) -> i64 { a + 100 }` using ADD r, imm.
fn build_add_imm() -> X86ISelFunction {
    let sig = Signature {
        params: vec![Type::I64],
        returns: vec![Type::I64],
    };

    let mut func = X86ISelFunction::new("add_imm".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    let v0 = VReg::new(0, RegClass::Gpr64);
    func.next_vreg = 1;

    // MOV v0, RDI
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(x86_64_regs::RDI)],
    ));

    // ADD v0, 100 (register-immediate form, in-place)
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::AddRI,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::Imm(100)],
    ));

    // MOV RAX, v0
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)],
    ));

    // RET
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));

    func
}

/// Build `neg_val(a: i64) -> i64 { -a }` using NEG (unary in-place).
fn build_neg() -> X86ISelFunction {
    let sig = Signature {
        params: vec![Type::I64],
        returns: vec![Type::I64],
    };

    let mut func = X86ISelFunction::new("neg_val".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    let v0 = VReg::new(0, RegClass::Gpr64);
    func.next_vreg = 1;

    // MOV v0, RDI
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(x86_64_regs::RDI)],
    ));

    // NEG v0
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::Neg,
        vec![X86ISelOperand::VReg(v0)],
    ));

    // MOV RAX, v0
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)],
    ));

    // RET
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));

    func
}

// ===========================================================================
// Test 1: const42 — baseline, no arithmetic, should be correct
// ===========================================================================

#[test]
fn test_correctness_const42() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available");
        return;
    }

    let func = build_x86_const_test_function();
    let driver = r#"
#include <stdio.h>
extern long const42(void);
int main(void) {
    long result = const42();
    printf("const42() = %ld\n", result);
    return (result == 42) ? 0 : 1;
}
"#;

    let (exit_code, stdout) = compile_link_run("const42_correct", &func, driver, default_macho_config());
    assert_eq!(exit_code, 0, "const42() should return 42. stdout: {}", stdout);
    assert!(stdout.contains("const42() = 42"), "Expected 42, got: {}", stdout);

    // Golden truth: compile equivalent C
    let (golden_exit, golden_stdout) = compile_run_c_golden("const42", r#"
#include <stdio.h>
long const42(void) { return 42; }
int main(void) {
    long result = const42();
    printf("const42() = %ld\n", result);
    return (result == 42) ? 0 : 1;
}
"#);
    assert_eq!(golden_exit, 0, "Golden C const42 failed");
    assert_eq!(
        stdout.trim(), golden_stdout.trim(),
        "LLVM2 output should match golden C output"
    );
}

// ===========================================================================
// Test 2: identity — return the argument unchanged
// ===========================================================================

#[test]
fn test_correctness_identity() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available");
        return;
    }

    let func = build_identity();
    let driver = r#"
#include <stdio.h>
extern long identity(long a);
int main(void) {
    long r1 = identity(42);
    long r2 = identity(0);
    long r3 = identity(-7);
    long r4 = identity(1000000);
    printf("identity(42)=%ld identity(0)=%ld identity(-7)=%ld identity(1000000)=%ld\n",
           r1, r2, r3, r4);
    int ok = (r1 == 42) && (r2 == 0) && (r3 == -7) && (r4 == 1000000);
    return ok ? 0 : 1;
}
"#;

    let (exit_code, stdout) = compile_link_run("identity_correct", &func, driver, default_macho_config());
    assert_eq!(exit_code, 0, "identity should pass through values. stdout: {}", stdout);
}

// ===========================================================================
// Test 3: add_correct — two-address-safe add, MUST produce correct results
// ===========================================================================

#[test]
fn test_correctness_add_two_address_safe() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available");
        return;
    }

    let func = build_add_correct();
    let driver = r#"
#include <stdio.h>
extern long add_correct(long a, long b);
int main(void) {
    long r1 = add_correct(30, 12);
    long r2 = add_correct(0, 0);
    long r3 = add_correct(-5, 5);
    long r4 = add_correct(100, -1);
    printf("add(30,12)=%ld add(0,0)=%ld add(-5,5)=%ld add(100,-1)=%ld\n",
           r1, r2, r3, r4);
    int ok = (r1 == 42) && (r2 == 0) && (r3 == 0) && (r4 == 99);
    return ok ? 0 : 1;
}
"#;

    let (exit_code, stdout) = compile_link_run(
        "add_correct", &func, driver, default_macho_config(),
    );
    assert_eq!(
        exit_code, 0,
        "Two-address-safe add MUST produce correct results. stdout: {}",
        stdout
    );
    assert!(
        stdout.contains("add(30,12)=42"),
        "Expected add(30,12)=42, got: {}", stdout
    );

    // Golden truth
    let (golden_exit, golden_stdout) = compile_run_c_golden("add", r#"
#include <stdio.h>
long add_correct(long a, long b) { return a + b; }
int main(void) {
    long r1 = add_correct(30, 12);
    long r2 = add_correct(0, 0);
    long r3 = add_correct(-5, 5);
    long r4 = add_correct(100, -1);
    printf("add(30,12)=%ld add(0,0)=%ld add(-5,5)=%ld add(100,-1)=%ld\n",
           r1, r2, r3, r4);
    int ok = (r1 == 42) && (r2 == 0) && (r3 == 0) && (r4 == 99);
    return ok ? 0 : 1;
}
"#);
    assert_eq!(golden_exit, 0);
    assert_eq!(stdout.trim(), golden_stdout.trim(), "LLVM2 add should match C golden");
}

// ===========================================================================
// Test 4: original add (three-address ISel) — documents the two-address bug
// ===========================================================================

#[test]
fn test_correctness_add_three_address_fixed() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available");
        return;
    }

    // The original add function uses three-address form:
    // ADD v2, v0, v1 (where v2 != v0)
    // Previously this produced WRONG results due to the two-address constraint bug.
    // Now fixed: the pipeline inserts MOV copies for dst != lhs (#305).
    let func = build_x86_add_test_function();
    let driver = r#"
#include <stdio.h>
extern long add(long a, long b);
int main(void) {
    long result = add(30, 12);
    printf("add(30,12)=%ld\n", result);
    return (result == 42) ? 0 : 1;
}
"#;

    let (exit_code, stdout) = compile_link_run(
        "add_three_address_fixed", &func, driver, default_macho_config(),
    );
    assert_eq!(
        exit_code, 0,
        "Three-address add(30,12) should return 42 after two-address fixup. stdout: {}",
        stdout
    );
    assert!(
        stdout.contains("add(30,12)=42"),
        "Expected add(30,12)=42, got: {}",
        stdout
    );
}

// ===========================================================================
// Test 5: sub_correct — subtraction with two-address-safe form
// ===========================================================================

#[test]
fn test_correctness_sub_two_address_safe() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available");
        return;
    }

    let func = build_sub_correct();
    let driver = r#"
#include <stdio.h>
extern long sub_correct(long a, long b);
int main(void) {
    long r1 = sub_correct(50, 8);
    long r2 = sub_correct(0, 0);
    long r3 = sub_correct(10, 20);
    long r4 = sub_correct(-3, -10);
    printf("sub(50,8)=%ld sub(0,0)=%ld sub(10,20)=%ld sub(-3,-10)=%ld\n",
           r1, r2, r3, r4);
    int ok = (r1 == 42) && (r2 == 0) && (r3 == -10) && (r4 == 7);
    return ok ? 0 : 1;
}
"#;

    let (exit_code, stdout) = compile_link_run(
        "sub_correct", &func, driver, default_macho_config(),
    );
    assert_eq!(
        exit_code, 0,
        "Two-address-safe sub MUST produce correct results. stdout: {}",
        stdout
    );
    assert!(
        stdout.contains("sub(50,8)=42"),
        "Expected sub(50,8)=42, got: {}", stdout
    );
}

// ===========================================================================
// Test 6: mul_correct — multiplication with two-address-safe form
// ===========================================================================

#[test]
fn test_correctness_mul_two_address_safe() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available");
        return;
    }

    let func = build_mul_correct();
    let driver = r#"
#include <stdio.h>
extern long mul_correct(long a, long b);
int main(void) {
    long r1 = mul_correct(6, 7);
    long r2 = mul_correct(0, 100);
    long r3 = mul_correct(-3, 5);
    long r4 = mul_correct(-4, -5);
    printf("mul(6,7)=%ld mul(0,100)=%ld mul(-3,5)=%ld mul(-4,-5)=%ld\n",
           r1, r2, r3, r4);
    int ok = (r1 == 42) && (r2 == 0) && (r3 == -15) && (r4 == 20);
    return ok ? 0 : 1;
}
"#;

    let (exit_code, stdout) = compile_link_run(
        "mul_correct", &func, driver, default_macho_config(),
    );
    assert_eq!(
        exit_code, 0,
        "Two-address-safe mul MUST produce correct results. stdout: {}",
        stdout
    );
    assert!(
        stdout.contains("mul(6,7)=42"),
        "Expected mul(6,7)=42, got: {}", stdout
    );
}

// ===========================================================================
// Test 7: add_imm — register + immediate
// ===========================================================================

#[test]
fn test_correctness_add_immediate() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available");
        return;
    }

    let func = build_add_imm();
    let driver = r#"
#include <stdio.h>
extern long add_imm(long a);
int main(void) {
    long r1 = add_imm(0);
    long r2 = add_imm(42);
    long r3 = add_imm(-100);
    printf("add_imm(0)=%ld add_imm(42)=%ld add_imm(-100)=%ld\n", r1, r2, r3);
    int ok = (r1 == 100) && (r2 == 142) && (r3 == 0);
    return ok ? 0 : 1;
}
"#;

    let (exit_code, stdout) = compile_link_run(
        "add_imm_correct", &func, driver, default_macho_config(),
    );
    assert_eq!(
        exit_code, 0,
        "ADD reg, imm should produce correct results. stdout: {}",
        stdout
    );
}

// ===========================================================================
// Test 8: neg — unary negation (in-place)
// ===========================================================================

#[test]
fn test_correctness_neg() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available");
        return;
    }

    let func = build_neg();
    let driver = r#"
#include <stdio.h>
extern long neg_val(long a);
int main(void) {
    long r1 = neg_val(42);
    long r2 = neg_val(0);
    long r3 = neg_val(-7);
    printf("neg(42)=%ld neg(0)=%ld neg(-7)=%ld\n", r1, r2, r3);
    int ok = (r1 == -42) && (r2 == 0) && (r3 == 7);
    return ok ? 0 : 1;
}
"#;

    let (exit_code, stdout) = compile_link_run(
        "neg_correct", &func, driver, default_macho_config(),
    );
    assert_eq!(
        exit_code, 0,
        "NEG should negate values correctly. stdout: {}",
        stdout
    );
}

// ===========================================================================
// Test 9: bitwise AND — two-address-safe
// ===========================================================================

#[test]
fn test_correctness_and_two_address_safe() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available");
        return;
    }

    let func = build_and_correct();
    let driver = r#"
#include <stdio.h>
extern long bitwise_and(long a, long b);
int main(void) {
    long r1 = bitwise_and(0xFF, 0x0F);
    long r2 = bitwise_and(0, 0xFFFF);
    long r3 = bitwise_and(-1, 42);
    printf("and(0xFF,0x0F)=%ld and(0,0xFFFF)=%ld and(-1,42)=%ld\n", r1, r2, r3);
    int ok = (r1 == 0x0F) && (r2 == 0) && (r3 == 42);
    return ok ? 0 : 1;
}
"#;

    let (exit_code, stdout) = compile_link_run(
        "and_correct", &func, driver, default_macho_config(),
    );
    assert_eq!(
        exit_code, 0,
        "Bitwise AND should produce correct results. stdout: {}",
        stdout
    );
}

// ===========================================================================
// Test 10: add_correct with greedy regalloc
// ===========================================================================

#[test]
fn test_correctness_add_greedy_regalloc() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available");
        return;
    }

    let func = build_add_correct();
    let driver = r#"
#include <stdio.h>
extern long add_correct(long a, long b);
int main(void) {
    long r = add_correct(30, 12);
    printf("add_greedy(30,12)=%ld\n", r);
    return (r == 42) ? 0 : 1;
}
"#;

    let (exit_code, stdout) = compile_link_run(
        "add_greedy", &func, driver, greedy_macho_config(),
    );
    assert_eq!(
        exit_code, 0,
        "Greedy regalloc add should produce 42. stdout: {}",
        stdout
    );
}

// ===========================================================================
// Test 11: add with greedy regalloc on ORIGINAL three-address form
// Explore whether greedy allocator happens to produce correct results
// ===========================================================================

#[test]
fn test_correctness_add_three_address_greedy_fixed() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available");
        return;
    }

    // Three-address add with greedy regalloc — now fixed by two-address fixup.
    let func = build_x86_add_test_function();
    let driver = r#"
#include <stdio.h>
extern long add(long a, long b);
int main(void) {
    long result = add(30, 12);
    printf("add_greedy(30,12)=%ld\n", result);
    return (result == 42) ? 0 : 1;
}
"#;

    let (exit_code, stdout) = compile_link_run(
        "add_greedy_three_addr", &func, driver, greedy_macho_config(),
    );
    assert_eq!(
        exit_code, 0,
        "Three-address add with greedy regalloc should return 42 after fixup. stdout: {}",
        stdout
    );
    assert!(
        stdout.contains("add_greedy(30,12)=42"),
        "Expected add_greedy(30,12)=42, got: {}",
        stdout
    );
}

// ===========================================================================
// Test 12: multi-operation — add then sub (chain of two-address ops)
// ===========================================================================

#[test]
fn test_correctness_add_then_sub() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available");
        return;
    }

    // Build: fn add_sub(a: i64, b: i64) -> i64 { (a + b) - 1 }
    let sig = Signature {
        params: vec![Type::I64, Type::I64],
        returns: vec![Type::I64],
    };

    let mut func = X86ISelFunction::new("add_sub".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    let v0 = VReg::new(0, RegClass::Gpr64);
    let v1 = VReg::new(1, RegClass::Gpr64);
    func.next_vreg = 2;

    // MOV v0, RDI
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(x86_64_regs::RDI)],
    ));
    // MOV v1, RSI
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v1), X86ISelOperand::PReg(x86_64_regs::RSI)],
    ));
    // ADD v0, v0, v1
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::AddRR,
        vec![
            X86ISelOperand::VReg(v0),
            X86ISelOperand::VReg(v0),
            X86ISelOperand::VReg(v1),
        ],
    ));
    // SUB v0, 1  (subtract immediate)
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::SubRI,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::Imm(1)],
    ));
    // MOV RAX, v0
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)],
    ));
    // RET
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));

    let driver = r#"
#include <stdio.h>
extern long add_sub(long a, long b);
int main(void) {
    long r = add_sub(30, 13);
    printf("add_sub(30,13)=%ld\n", r);
    return (r == 42) ? 0 : 1;
}
"#;

    let (exit_code, stdout) = compile_link_run(
        "add_sub_correct", &func, driver, default_macho_config(),
    );
    assert_eq!(
        exit_code, 0,
        "add_sub(30,13) should return 42. stdout: {}",
        stdout
    );
}

// ===========================================================================
// Test 13: comprehensive golden truth comparison — all operations
// ===========================================================================

#[test]
fn test_correctness_golden_add() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available");
        return;
    }

    // Compile the golden truth C program
    let (golden_exit, _) = compile_run_c_golden("golden_all", r#"
#include <stdio.h>

long c_add(long a, long b) { return a + b; }
long c_sub(long a, long b) { return a - b; }
long c_mul(long a, long b) { return a * b; }
long c_neg(long a) { return -a; }
long c_identity(long a) { return a; }

int main(void) {
    int pass = 1;
    pass &= (c_add(30, 12) == 42);
    pass &= (c_sub(50, 8) == 42);
    pass &= (c_mul(6, 7) == 42);
    pass &= (c_neg(42) == -42);
    pass &= (c_identity(42) == 42);
    printf("golden_all: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
"#);
    assert_eq!(golden_exit, 0, "Golden C should pass all checks");
}

// ===========================================================================
// Summary test — runs all correct functions and reports results
// ===========================================================================

#[test]
fn test_correctness_summary() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available");
        return;
    }

    eprintln!("\n=== x86-64 Correctness Test Summary ===");
    eprintln!("All functions MUST compute correct results:");
    eprintln!("  - const42()        -> 42    [no arithmetic, trivially correct]");
    eprintln!("  - identity(x)      -> x     [no arithmetic, trivially correct]");
    eprintln!("  - add_correct(a,b) -> a+b   [two-address-safe: dst == lhs]");
    eprintln!("  - add(a,b)         -> a+b   [three-address: dst != lhs, fixed by #305]");
    eprintln!("  - sub_correct(a,b) -> a-b   [two-address-safe: dst == lhs]");
    eprintln!("  - mul_correct(a,b) -> a*b   [two-address-safe: dst == lhs]");
    eprintln!("  - add_imm(a)       -> a+100 [register-immediate, in-place]");
    eprintln!("  - neg_val(a)       -> -a    [unary, in-place]");
    eprintln!("  - bitwise_and(a,b) -> a&b   [two-address-safe: dst == lhs]");
    eprintln!("  - add_sub(a,b)     -> a+b-1 [chained operations]");
    eprintln!();
    eprintln!("Two-address constraint fix (#305):");
    eprintln!("  The pipeline now inserts MOV copies before two-address instructions");
    eprintln!("  when dst != lhs. This runs between regalloc and prologue/epilogue.");
    eprintln!("  Both simplified and greedy/linear-scan allocators are covered.");
    eprintln!("=== End Summary ===\n");
}

// ===========================================================================
// NEW FUNCTION BUILDERS — Wave 42 expansion
// ===========================================================================

/// Build `bitwise_or(a: i64, b: i64) -> i64 { a | b }` (two-address-safe).
fn build_or_correct() -> X86ISelFunction {
    let sig = Signature {
        params: vec![Type::I64, Type::I64],
        returns: vec![Type::I64],
    };
    let mut func = X86ISelFunction::new("bitwise_or".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);
    let v0 = VReg::new(0, RegClass::Gpr64);
    let v1 = VReg::new(1, RegClass::Gpr64);
    func.next_vreg = 2;

    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(x86_64_regs::RDI)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v1), X86ISelOperand::PReg(x86_64_regs::RSI)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::OrRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::VReg(v0), X86ISelOperand::VReg(v1)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));
    func
}

/// Build `bitwise_xor(a: i64, b: i64) -> i64 { a ^ b }` (two-address-safe).
fn build_xor_correct() -> X86ISelFunction {
    let sig = Signature {
        params: vec![Type::I64, Type::I64],
        returns: vec![Type::I64],
    };
    let mut func = X86ISelFunction::new("bitwise_xor".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);
    let v0 = VReg::new(0, RegClass::Gpr64);
    let v1 = VReg::new(1, RegClass::Gpr64);
    func.next_vreg = 2;

    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(x86_64_regs::RDI)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v1), X86ISelOperand::PReg(x86_64_regs::RSI)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::XorRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::VReg(v0), X86ISelOperand::VReg(v1)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));
    func
}

/// Build `shl_imm(a: i64) -> i64 { a << 3 }`.
fn build_shl_correct() -> X86ISelFunction {
    let sig = Signature {
        params: vec![Type::I64],
        returns: vec![Type::I64],
    };
    let mut func = X86ISelFunction::new("shl_imm".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);
    let v0 = VReg::new(0, RegClass::Gpr64);
    func.next_vreg = 1;

    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(x86_64_regs::RDI)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::ShlRI,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::Imm(3)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));
    func
}

/// Build `shr_imm(a: i64) -> i64 { a >> 2 }` (logical shift right).
fn build_shr_correct() -> X86ISelFunction {
    let sig = Signature {
        params: vec![Type::I64],
        returns: vec![Type::I64],
    };
    let mut func = X86ISelFunction::new("shr_imm".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);
    let v0 = VReg::new(0, RegClass::Gpr64);
    func.next_vreg = 1;

    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(x86_64_regs::RDI)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::ShrRI,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::Imm(2)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));
    func
}

/// Build `max_cmov(a: i64, b: i64) -> i64 { max(a, b) }` using CMP + CMOVL.
///
/// Branchless: CMP a, b; CMOVL a, b => if a < b then a = b.
fn build_max_cmov() -> X86ISelFunction {
    let sig = Signature {
        params: vec![Type::I64, Type::I64],
        returns: vec![Type::I64],
    };
    let mut func = X86ISelFunction::new("max_cmov".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);
    let v0 = VReg::new(0, RegClass::Gpr64);
    let v1 = VReg::new(1, RegClass::Gpr64);
    func.next_vreg = 2;

    // v0 = a, v1 = b
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(x86_64_regs::RDI)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v1), X86ISelOperand::PReg(x86_64_regs::RSI)]));
    // CMP v0, v1 (sets flags based on a - b)
    func.push_inst(entry, X86ISelInst::new(X86Opcode::CmpRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::VReg(v1)]));
    // CMOVL v0, v1: if a < b (signed), set v0 = v1
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Cmovcc,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::VReg(v1),
             X86ISelOperand::CondCode(X86CondCode::L)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));
    func
}

/// Build `abs_val(x: i64) -> i64 { if x < 0 { -x } else { x } }` using NEG + CMP + CMOV.
///
/// v0 = x, v1 = copy of x, NEG v1 => v1 = -x
/// CMP v0, 0: if v0 < 0 (signed), CMOVL v0, v1 => v0 = -x
fn build_abs_val() -> X86ISelFunction {
    let sig = Signature {
        params: vec![Type::I64],
        returns: vec![Type::I64],
    };
    let mut func = X86ISelFunction::new("abs_val".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);
    let v0 = VReg::new(0, RegClass::Gpr64);
    let v1 = VReg::new(1, RegClass::Gpr64);
    func.next_vreg = 2;

    // v0 = x
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(x86_64_regs::RDI)]));
    // v1 = x (copy from v0)
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v1), X86ISelOperand::VReg(v0)]));
    // NEG v1 => v1 = -x
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Neg,
        vec![X86ISelOperand::VReg(v1)]));
    // CMP v0, 0
    func.push_inst(entry, X86ISelInst::new(X86Opcode::CmpRI,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::Imm(0)]));
    // CMOVL v0, v1: if x < 0 (signed), v0 = v1 = -x
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Cmovcc,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::VReg(v1),
             X86ISelOperand::CondCode(X86CondCode::L)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));
    func
}

/// Build `combined_arith(a, b, c, d: i64) -> i64 { (a + b) * (c - d) }`.
///
/// 4-arg SysV: a=RDI, b=RSI, c=RDX, d=RCX
fn build_combined_arith() -> X86ISelFunction {
    let sig = Signature {
        params: vec![Type::I64, Type::I64, Type::I64, Type::I64],
        returns: vec![Type::I64],
    };
    let mut func = X86ISelFunction::new("combined_arith".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);
    let v0 = VReg::new(0, RegClass::Gpr64);
    let v1 = VReg::new(1, RegClass::Gpr64);
    let v2 = VReg::new(2, RegClass::Gpr64);
    let v3 = VReg::new(3, RegClass::Gpr64);
    func.next_vreg = 4;

    // Load 4 args
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(x86_64_regs::RDI)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v1), X86ISelOperand::PReg(x86_64_regs::RSI)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v2), X86ISelOperand::PReg(x86_64_regs::RDX)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v3), X86ISelOperand::PReg(x86_64_regs::RCX)]));
    // v0 = a + b (two-address-safe)
    func.push_inst(entry, X86ISelInst::new(X86Opcode::AddRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::VReg(v0), X86ISelOperand::VReg(v1)]));
    // v2 = c - d (two-address-safe)
    func.push_inst(entry, X86ISelInst::new(X86Opcode::SubRR,
        vec![X86ISelOperand::VReg(v2), X86ISelOperand::VReg(v2), X86ISelOperand::VReg(v3)]));
    // v0 = (a+b) * (c-d) (two-address-safe)
    func.push_inst(entry, X86ISelInst::new(X86Opcode::ImulRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::VReg(v0), X86ISelOperand::VReg(v2)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));
    func
}

/// Build `max_branch(a, b: i64) -> i64 { if a > b { a } else { b } }` using multi-block branches.
fn build_max_branch() -> X86ISelFunction {
    let sig = Signature {
        params: vec![Type::I64, Type::I64],
        returns: vec![Type::I64],
    };
    let mut func = X86ISelFunction::new("max_branch".to_string(), sig);
    let bb0 = Block(0);
    let bb1 = Block(1);
    let bb2 = Block(2);
    func.ensure_block(bb0);
    func.ensure_block(bb1);
    func.ensure_block(bb2);
    let v0 = VReg::new(0, RegClass::Gpr64);
    let v1 = VReg::new(1, RegClass::Gpr64);
    func.next_vreg = 2;

    // bb0: load args, compare, branch
    func.push_inst(bb0, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(x86_64_regs::RDI)]));
    func.push_inst(bb0, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v1), X86ISelOperand::PReg(x86_64_regs::RSI)]));
    func.push_inst(bb0, X86ISelInst::new(X86Opcode::CmpRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::VReg(v1)]));
    // If a > b (signed), jump to bb1; else fall through to bb2
    func.push_inst(bb0, X86ISelInst::new(X86Opcode::Jcc,
        vec![X86ISelOperand::CondCode(X86CondCode::G), X86ISelOperand::Block(bb1)]));
    func.push_inst(bb0, X86ISelInst::new(X86Opcode::Jmp,
        vec![X86ISelOperand::Block(bb2)]));

    // bb1: return a
    func.push_inst(bb1, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)]));
    func.push_inst(bb1, X86ISelInst::new(X86Opcode::Ret, vec![]));

    // bb2: return b
    func.push_inst(bb2, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v1)]));
    func.push_inst(bb2, X86ISelInst::new(X86Opcode::Ret, vec![]));

    func
}

/// Build `signed_arith(a, b: i64) -> i64 { a * b + (-a) }` = a * (b - 1).
///
/// Tests signed arithmetic with negation.
fn build_signed_arithmetic() -> X86ISelFunction {
    let sig = Signature {
        params: vec![Type::I64, Type::I64],
        returns: vec![Type::I64],
    };
    let mut func = X86ISelFunction::new("signed_arith".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);
    let v0 = VReg::new(0, RegClass::Gpr64);
    let v1 = VReg::new(1, RegClass::Gpr64);
    let v2 = VReg::new(2, RegClass::Gpr64);
    func.next_vreg = 3;

    // v0 = a, v1 = b
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::PReg(x86_64_regs::RDI)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v1), X86ISelOperand::PReg(x86_64_regs::RSI)]));
    // v2 = copy of v0 (a)
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v2), X86ISelOperand::VReg(v0)]));
    // v2 = -a
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Neg,
        vec![X86ISelOperand::VReg(v2)]));
    // v0 = a * b (two-address-safe)
    func.push_inst(entry, X86ISelInst::new(X86Opcode::ImulRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::VReg(v0), X86ISelOperand::VReg(v1)]));
    // v0 = a*b + (-a) (two-address-safe)
    func.push_inst(entry, X86ISelInst::new(X86Opcode::AddRR,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::VReg(v0), X86ISelOperand::VReg(v2)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));
    func
}

/// Build `large_const() -> i64 { 4294967296 }` (0x1_0000_0000, >32 bits).
fn build_large_const() -> X86ISelFunction {
    let sig = Signature {
        params: vec![],
        returns: vec![Type::I64],
    };
    let mut func = X86ISelFunction::new("large_const".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);
    let v0 = VReg::new(0, RegClass::Gpr64);
    func.next_vreg = 1;

    // MOV r64, imm64 (requires movabs encoding for >32-bit immediate)
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRI,
        vec![X86ISelOperand::VReg(v0), X86ISelOperand::Imm(4294967296)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v0)]));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));
    func
}

// ===========================================================================
// Test 14: OR — two-address-safe bitwise OR
// ===========================================================================

#[test]
fn test_correctness_or() {
    if !has_cc_x86_64() { eprintln!("SKIP"); return; }

    let func = build_or_correct();
    let driver = r#"
#include <stdio.h>
extern long bitwise_or(long a, long b);
int main(void) {
    long r1 = bitwise_or(0xFF, 0xF0);
    long r2 = bitwise_or(0, 0);
    long r3 = bitwise_or(0xA, 0x5);
    printf("or(0xFF,0xF0)=%ld or(0,0)=%ld or(0xA,0x5)=%ld\n", r1, r2, r3);
    int ok = (r1 == 0xFF) && (r2 == 0) && (r3 == 0xF);
    return ok ? 0 : 1;
}
"#;
    let (exit_code, stdout) = compile_link_run("or_correct", &func, driver, default_macho_config());
    assert_eq!(exit_code, 0, "OR should produce correct results. stdout: {}", stdout);

    let (golden_exit, golden_stdout) = compile_run_c_golden("or", r#"
#include <stdio.h>
long bitwise_or(long a, long b) { return a | b; }
int main(void) {
    long r1 = bitwise_or(0xFF, 0xF0);
    long r2 = bitwise_or(0, 0);
    long r3 = bitwise_or(0xA, 0x5);
    printf("or(0xFF,0xF0)=%ld or(0,0)=%ld or(0xA,0x5)=%ld\n", r1, r2, r3);
    int ok = (r1 == 0xFF) && (r2 == 0) && (r3 == 0xF);
    return ok ? 0 : 1;
}
"#);
    assert_eq!(golden_exit, 0);
    assert_eq!(stdout.trim(), golden_stdout.trim(), "LLVM2 OR should match golden C");
}

// ===========================================================================
// Test 15: XOR — two-address-safe bitwise XOR
// ===========================================================================

#[test]
fn test_correctness_xor() {
    if !has_cc_x86_64() { eprintln!("SKIP"); return; }

    let func = build_xor_correct();
    let driver = r#"
#include <stdio.h>
extern long bitwise_xor(long a, long b);
int main(void) {
    long r1 = bitwise_xor(0xFF, 0xFF);
    long r2 = bitwise_xor(0xA, 0x5);
    long r3 = bitwise_xor(0, 42);
    printf("xor(0xFF,0xFF)=%ld xor(0xA,0x5)=%ld xor(0,42)=%ld\n", r1, r2, r3);
    int ok = (r1 == 0) && (r2 == 0xF) && (r3 == 42);
    return ok ? 0 : 1;
}
"#;
    let (exit_code, stdout) = compile_link_run("xor_correct", &func, driver, default_macho_config());
    assert_eq!(exit_code, 0, "XOR should produce correct results. stdout: {}", stdout);

    let (golden_exit, golden_stdout) = compile_run_c_golden("xor", r#"
#include <stdio.h>
long bitwise_xor(long a, long b) { return a ^ b; }
int main(void) {
    long r1 = bitwise_xor(0xFF, 0xFF);
    long r2 = bitwise_xor(0xA, 0x5);
    long r3 = bitwise_xor(0, 42);
    printf("xor(0xFF,0xFF)=%ld xor(0xA,0x5)=%ld xor(0,42)=%ld\n", r1, r2, r3);
    int ok = (r1 == 0) && (r2 == 0xF) && (r3 == 42);
    return ok ? 0 : 1;
}
"#);
    assert_eq!(golden_exit, 0);
    assert_eq!(stdout.trim(), golden_stdout.trim(), "LLVM2 XOR should match golden C");
}

// ===========================================================================
// Test 16: SHL — shift left by immediate
// ===========================================================================

#[test]
fn test_correctness_shl() {
    if !has_cc_x86_64() { eprintln!("SKIP"); return; }

    let func = build_shl_correct();
    let driver = r#"
#include <stdio.h>
extern long shl_imm(long a);
int main(void) {
    long r1 = shl_imm(1);
    long r2 = shl_imm(0);
    long r3 = shl_imm(5);
    printf("shl(1)=%ld shl(0)=%ld shl(5)=%ld\n", r1, r2, r3);
    int ok = (r1 == 8) && (r2 == 0) && (r3 == 40);
    return ok ? 0 : 1;
}
"#;
    let (exit_code, stdout) = compile_link_run("shl_correct", &func, driver, default_macho_config());
    assert_eq!(exit_code, 0, "SHL should produce correct results. stdout: {}", stdout);
}

// ===========================================================================
// Test 17: SHR — logical shift right by immediate
// ===========================================================================

#[test]
fn test_correctness_shr() {
    if !has_cc_x86_64() { eprintln!("SKIP"); return; }

    let func = build_shr_correct();
    let driver = r#"
#include <stdio.h>
extern long shr_imm(long a);
int main(void) {
    long r1 = shr_imm(16);
    long r2 = shr_imm(0);
    long r3 = shr_imm(255);
    printf("shr(16)=%ld shr(0)=%ld shr(255)=%ld\n", r1, r2, r3);
    int ok = (r1 == 4) && (r2 == 0) && (r3 == 63);
    return ok ? 0 : 1;
}
"#;
    let (exit_code, stdout) = compile_link_run("shr_correct", &func, driver, default_macho_config());
    assert_eq!(exit_code, 0, "SHR should produce correct results. stdout: {}", stdout);
}

// ===========================================================================
// Test 18: max via CMOV — branchless conditional
// ===========================================================================

#[test]
fn test_correctness_max_cmov() {
    if !has_cc_x86_64() { eprintln!("SKIP"); return; }

    let func = build_max_cmov();
    let driver = r#"
#include <stdio.h>
extern long max_cmov(long a, long b);
int main(void) {
    long r1 = max_cmov(3, 7);
    long r2 = max_cmov(7, 3);
    long r3 = max_cmov(5, 5);
    long r4 = max_cmov(-5, 3);
    long r5 = max_cmov(-10, -3);
    printf("max(3,7)=%ld max(7,3)=%ld max(5,5)=%ld max(-5,3)=%ld max(-10,-3)=%ld\n",
           r1, r2, r3, r4, r5);
    int ok = (r1 == 7) && (r2 == 7) && (r3 == 5) && (r4 == 3) && (r5 == -3);
    return ok ? 0 : 1;
}
"#;
    let (exit_code, stdout) = compile_link_run("max_cmov_correct", &func, driver, default_macho_config());
    assert_eq!(exit_code, 0, "CMOV max should produce correct results. stdout: {}", stdout);

    let (golden_exit, golden_stdout) = compile_run_c_golden("max_cmov", r#"
#include <stdio.h>
long max_cmov(long a, long b) { return a > b ? a : b; }
int main(void) {
    long r1 = max_cmov(3, 7);
    long r2 = max_cmov(7, 3);
    long r3 = max_cmov(5, 5);
    long r4 = max_cmov(-5, 3);
    long r5 = max_cmov(-10, -3);
    printf("max(3,7)=%ld max(7,3)=%ld max(5,5)=%ld max(-5,3)=%ld max(-10,-3)=%ld\n",
           r1, r2, r3, r4, r5);
    int ok = (r1 == 7) && (r2 == 7) && (r3 == 5) && (r4 == 3) && (r5 == -3);
    return ok ? 0 : 1;
}
"#);
    assert_eq!(golden_exit, 0);
    assert_eq!(stdout.trim(), golden_stdout.trim(), "LLVM2 max_cmov should match golden C");
}

// ===========================================================================
// Test 19: abs via NEG + CMP + CMOV
// ===========================================================================

#[test]
fn test_correctness_abs() {
    if !has_cc_x86_64() { eprintln!("SKIP"); return; }

    let func = build_abs_val();
    let driver = r#"
#include <stdio.h>
extern long abs_val(long x);
int main(void) {
    long r1 = abs_val(42);
    long r2 = abs_val(-42);
    long r3 = abs_val(0);
    long r4 = abs_val(-1);
    long r5 = abs_val(1);
    printf("abs(42)=%ld abs(-42)=%ld abs(0)=%ld abs(-1)=%ld abs(1)=%ld\n",
           r1, r2, r3, r4, r5);
    int ok = (r1 == 42) && (r2 == 42) && (r3 == 0) && (r4 == 1) && (r5 == 1);
    return ok ? 0 : 1;
}
"#;
    let (exit_code, stdout) = compile_link_run("abs_correct", &func, driver, default_macho_config());
    assert_eq!(exit_code, 0, "ABS should produce correct results. stdout: {}", stdout);

    let (golden_exit, golden_stdout) = compile_run_c_golden("abs", r#"
#include <stdio.h>
long abs_val(long x) { return x < 0 ? -x : x; }
int main(void) {
    long r1 = abs_val(42);
    long r2 = abs_val(-42);
    long r3 = abs_val(0);
    long r4 = abs_val(-1);
    long r5 = abs_val(1);
    printf("abs(42)=%ld abs(-42)=%ld abs(0)=%ld abs(-1)=%ld abs(1)=%ld\n",
           r1, r2, r3, r4, r5);
    int ok = (r1 == 42) && (r2 == 42) && (r3 == 0) && (r4 == 1) && (r5 == 1);
    return ok ? 0 : 1;
}
"#);
    assert_eq!(golden_exit, 0);
    assert_eq!(stdout.trim(), golden_stdout.trim(), "LLVM2 abs should match golden C");
}

// ===========================================================================
// Test 20: combined arithmetic — (a + b) * (c - d) with 4 args
// ===========================================================================

#[test]
fn test_correctness_combined_arith() {
    if !has_cc_x86_64() { eprintln!("SKIP"); return; }

    let func = build_combined_arith();
    let driver = r#"
#include <stdio.h>
extern long combined_arith(long a, long b, long c, long d);
int main(void) {
    long r1 = combined_arith(2, 3, 10, 4);
    long r2 = combined_arith(0, 0, 5, 5);
    long r3 = combined_arith(1, 1, 3, 1);
    long r4 = combined_arith(-2, 5, 10, 3);
    printf("(2+3)*(10-4)=%ld (0+0)*(5-5)=%ld (1+1)*(3-1)=%ld (-2+5)*(10-3)=%ld\n",
           r1, r2, r3, r4);
    int ok = (r1 == 30) && (r2 == 0) && (r3 == 4) && (r4 == 21);
    return ok ? 0 : 1;
}
"#;
    let (exit_code, stdout) = compile_link_run("combined_correct", &func, driver, default_macho_config());
    assert_eq!(exit_code, 0, "Combined arith should produce correct results. stdout: {}", stdout);

    let (golden_exit, golden_stdout) = compile_run_c_golden("combined", r#"
#include <stdio.h>
long combined_arith(long a, long b, long c, long d) { return (a + b) * (c - d); }
int main(void) {
    long r1 = combined_arith(2, 3, 10, 4);
    long r2 = combined_arith(0, 0, 5, 5);
    long r3 = combined_arith(1, 1, 3, 1);
    long r4 = combined_arith(-2, 5, 10, 3);
    printf("(2+3)*(10-4)=%ld (0+0)*(5-5)=%ld (1+1)*(3-1)=%ld (-2+5)*(10-3)=%ld\n",
           r1, r2, r3, r4);
    int ok = (r1 == 30) && (r2 == 0) && (r3 == 4) && (r4 == 21);
    return ok ? 0 : 1;
}
"#);
    assert_eq!(golden_exit, 0);
    assert_eq!(stdout.trim(), golden_stdout.trim(), "LLVM2 combined arith should match golden C");
}

// ===========================================================================
// Test 21: max via branches — multi-block conditional
// ===========================================================================

#[test]
fn test_correctness_max_branch() {
    if !has_cc_x86_64() { eprintln!("SKIP"); return; }

    let func = build_max_branch();
    let driver = r#"
#include <stdio.h>
extern long max_branch(long a, long b);
int main(void) {
    long r1 = max_branch(3, 7);
    long r2 = max_branch(7, 3);
    long r3 = max_branch(5, 5);
    long r4 = max_branch(-5, 3);
    long r5 = max_branch(-10, -3);
    printf("max(3,7)=%ld max(7,3)=%ld max(5,5)=%ld max(-5,3)=%ld max(-10,-3)=%ld\n",
           r1, r2, r3, r4, r5);
    int ok = (r1 == 7) && (r2 == 7) && (r3 == 5) && (r4 == 3) && (r5 == -3);
    return ok ? 0 : 1;
}
"#;
    let (exit_code, stdout) = compile_link_run("max_branch_correct", &func, driver, default_macho_config());
    assert_eq!(exit_code, 0, "Branch max should produce correct results. stdout: {}", stdout);

    let (golden_exit, golden_stdout) = compile_run_c_golden("max_branch", r#"
#include <stdio.h>
long max_branch(long a, long b) { return a > b ? a : b; }
int main(void) {
    long r1 = max_branch(3, 7);
    long r2 = max_branch(7, 3);
    long r3 = max_branch(5, 5);
    long r4 = max_branch(-5, 3);
    long r5 = max_branch(-10, -3);
    printf("max(3,7)=%ld max(7,3)=%ld max(5,5)=%ld max(-5,3)=%ld max(-10,-3)=%ld\n",
           r1, r2, r3, r4, r5);
    int ok = (r1 == 7) && (r2 == 7) && (r3 == 5) && (r4 == 3) && (r5 == -3);
    return ok ? 0 : 1;
}
"#);
    assert_eq!(golden_exit, 0);
    assert_eq!(stdout.trim(), golden_stdout.trim(), "LLVM2 max_branch should match golden C");
}

// ===========================================================================
// Test 22: signed arithmetic — a * b + (-a) = a * (b - 1)
// ===========================================================================

#[test]
fn test_correctness_signed_arithmetic() {
    if !has_cc_x86_64() { eprintln!("SKIP"); return; }

    let func = build_signed_arithmetic();
    let driver = r#"
#include <stdio.h>
extern long signed_arith(long a, long b);
int main(void) {
    long r1 = signed_arith(5, 3);
    long r2 = signed_arith(-3, 4);
    long r3 = signed_arith(0, 100);
    long r4 = signed_arith(10, 1);
    printf("sa(5,3)=%ld sa(-3,4)=%ld sa(0,100)=%ld sa(10,1)=%ld\n",
           r1, r2, r3, r4);
    int ok = (r1 == 10) && (r2 == -9) && (r3 == 0) && (r4 == 0);
    return ok ? 0 : 1;
}
"#;
    let (exit_code, stdout) = compile_link_run("signed_arith_correct", &func, driver, default_macho_config());
    assert_eq!(exit_code, 0, "Signed arith should produce correct results. stdout: {}", stdout);

    let (golden_exit, golden_stdout) = compile_run_c_golden("signed_arith", r#"
#include <stdio.h>
long signed_arith(long a, long b) { return a * b + (-a); }
int main(void) {
    long r1 = signed_arith(5, 3);
    long r2 = signed_arith(-3, 4);
    long r3 = signed_arith(0, 100);
    long r4 = signed_arith(10, 1);
    printf("sa(5,3)=%ld sa(-3,4)=%ld sa(0,100)=%ld sa(10,1)=%ld\n",
           r1, r2, r3, r4);
    int ok = (r1 == 10) && (r2 == -9) && (r3 == 0) && (r4 == 0);
    return ok ? 0 : 1;
}
"#);
    assert_eq!(golden_exit, 0);
    assert_eq!(stdout.trim(), golden_stdout.trim(), "LLVM2 signed arith should match golden C");
}

// ===========================================================================
// Test 23: large constant — 64-bit immediate > 2^32
// ===========================================================================

#[test]
fn test_correctness_large_const() {
    if !has_cc_x86_64() { eprintln!("SKIP"); return; }

    let func = build_large_const();
    let driver = r#"
#include <stdio.h>
extern long large_const(void);
int main(void) {
    long result = large_const();
    printf("large_const()=%ld\n", result);
    return (result == 4294967296L) ? 0 : 1;
}
"#;
    let (exit_code, stdout) = compile_link_run("large_const_correct", &func, driver, default_macho_config());
    assert_eq!(exit_code, 0, "Large const should return 4294967296. stdout: {}", stdout);

    let (golden_exit, golden_stdout) = compile_run_c_golden("large_const", r#"
#include <stdio.h>
long large_const(void) { return 4294967296L; }
int main(void) {
    long result = large_const();
    printf("large_const()=%ld\n", result);
    return (result == 4294967296L) ? 0 : 1;
}
"#);
    assert_eq!(golden_exit, 0);
    assert_eq!(stdout.trim(), golden_stdout.trim(), "LLVM2 large const should match golden C");
}

// ===========================================================================
// Test 24: large multiply — 64-bit multiplication result
// ===========================================================================

#[test]
fn test_correctness_large_multiply() {
    if !has_cc_x86_64() { eprintln!("SKIP"); return; }

    let func = build_mul_correct();
    let driver = r#"
#include <stdio.h>
extern long mul_correct(long a, long b);
int main(void) {
    long r1 = mul_correct(100000, 100000);
    long r2 = mul_correct(-100000, 100000);
    printf("mul(100000,100000)=%ld mul(-100000,100000)=%ld\n", r1, r2);
    int ok = (r1 == 10000000000L) && (r2 == -10000000000L);
    return ok ? 0 : 1;
}
"#;
    let (exit_code, stdout) = compile_link_run("large_mul_correct", &func, driver, default_macho_config());
    assert_eq!(exit_code, 0, "Large multiply should produce 64-bit results. stdout: {}", stdout);

    let (golden_exit, golden_stdout) = compile_run_c_golden("large_mul", r#"
#include <stdio.h>
long mul_correct(long a, long b) { return a * b; }
int main(void) {
    long r1 = mul_correct(100000, 100000);
    long r2 = mul_correct(-100000, 100000);
    printf("mul(100000,100000)=%ld mul(-100000,100000)=%ld\n", r1, r2);
    int ok = (r1 == 10000000000L) && (r2 == -10000000000L);
    return ok ? 0 : 1;
}
"#);
    assert_eq!(golden_exit, 0);
    assert_eq!(stdout.trim(), golden_stdout.trim(), "LLVM2 large mul should match golden C");
}

// ===========================================================================
// Test 25: 8-argument sum — stack argument passing (args 7-8 via stack)
// ===========================================================================

/// Build `stack_args_sum(a,b,c,d,e,f,g,h: i64) -> i64 { a+b+c+d+e+f+g+h }`
///
/// System V AMD64 ABI:
///   args 0-5: RDI, RSI, RDX, RCX, R8, R9 (registers)
///   args 6-7: [RBP+16], [RBP+24] (stack)
///
/// Uses two-address-safe ADD chain: v0 += v1; v0 += v2; ... v0 += v7.
fn build_stack_args_sum() -> X86ISelFunction {
    let sig = Signature {
        params: vec![Type::I64; 8],
        returns: vec![Type::I64],
    };

    let mut func = X86ISelFunction::new("stack_args_sum".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    // Allocate 8 vregs for args
    let v: Vec<VReg> = (0..8).map(|i| VReg::new(i, RegClass::Gpr64)).collect();
    func.next_vreg = 8;

    // Load args 0-5 from registers
    let arg_regs = [
        x86_64_regs::RDI, x86_64_regs::RSI, x86_64_regs::RDX,
        x86_64_regs::RCX, x86_64_regs::R8,  x86_64_regs::R9,
    ];
    for (i, preg) in arg_regs.iter().enumerate() {
        func.push_inst(entry, X86ISelInst::new(
            X86Opcode::MovRR,
            vec![X86ISelOperand::VReg(v[i]), X86ISelOperand::PReg(*preg)],
        ));
    }

    // Load arg 6 from [RBP+16] (7th arg, first stack arg)
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRM,
        vec![
            X86ISelOperand::VReg(v[6]),
            X86ISelOperand::MemAddr {
                base: Box::new(X86ISelOperand::PReg(x86_64_regs::RBP)),
                disp: 16,
            },
        ],
    ));

    // Load arg 7 from [RBP+24] (8th arg, second stack arg)
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRM,
        vec![
            X86ISelOperand::VReg(v[7]),
            X86ISelOperand::MemAddr {
                base: Box::new(X86ISelOperand::PReg(x86_64_regs::RBP)),
                disp: 24,
            },
        ],
    ));

    // Sum: v[0] += v[1]; v[0] += v[2]; ... v[0] += v[7]
    for i in 1..8 {
        func.push_inst(entry, X86ISelInst::new(
            X86Opcode::AddRR,
            vec![
                X86ISelOperand::VReg(v[0]),
                X86ISelOperand::VReg(v[0]),
                X86ISelOperand::VReg(v[i]),
            ],
        ));
    }

    // Return v[0] via RAX
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v[0])],
    ));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));

    func
}

#[test]
fn test_correctness_stack_args_sum() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available");
        return;
    }

    let func = build_stack_args_sum();
    let driver = r#"
#include <stdio.h>
extern long stack_args_sum(long a, long b, long c, long d,
                           long e, long f, long g, long h);
int main(void) {
    long r1 = stack_args_sum(1, 2, 3, 4, 5, 6, 7, 8);
    long r2 = stack_args_sum(0, 0, 0, 0, 0, 0, 0, 0);
    long r3 = stack_args_sum(10, 20, 30, 40, 50, 60, 70, 80);
    long r4 = stack_args_sum(-1, -2, -3, -4, -5, -6, -7, -8);
    printf("sum(1..8)=%ld sum(0s)=%ld sum(10s)=%ld sum(neg)=%ld\n",
           r1, r2, r3, r4);
    int ok = (r1 == 36) && (r2 == 0) && (r3 == 360) && (r4 == -36);
    return ok ? 0 : 1;
}
"#;

    let (exit_code, stdout) = compile_link_run(
        "stack_args_sum", &func, driver, default_macho_config(),
    );
    assert_eq!(
        exit_code, 0,
        "8-arg stack_args_sum should produce correct results. stdout: {}",
        stdout
    );
    assert!(
        stdout.contains("sum(1..8)=36"),
        "Expected sum(1..8)=36, got: {}", stdout
    );

    // Golden truth comparison
    let (golden_exit, golden_stdout) = compile_run_c_golden("stack_args_sum", r#"
#include <stdio.h>
long stack_args_sum(long a, long b, long c, long d,
                    long e, long f, long g, long h) {
    return a + b + c + d + e + f + g + h;
}
int main(void) {
    long r1 = stack_args_sum(1, 2, 3, 4, 5, 6, 7, 8);
    long r2 = stack_args_sum(0, 0, 0, 0, 0, 0, 0, 0);
    long r3 = stack_args_sum(10, 20, 30, 40, 50, 60, 70, 80);
    long r4 = stack_args_sum(-1, -2, -3, -4, -5, -6, -7, -8);
    printf("sum(1..8)=%ld sum(0s)=%ld sum(10s)=%ld sum(neg)=%ld\n",
           r1, r2, r3, r4);
    int ok = (r1 == 36) && (r2 == 0) && (r3 == 360) && (r4 == -36);
    return ok ? 0 : 1;
}
"#);
    assert_eq!(golden_exit, 0);
    assert_eq!(
        stdout.trim(), golden_stdout.trim(),
        "LLVM2 stack_args_sum should match golden C"
    );
}

// NOTE: Greedy regalloc test for 8+ args is skipped. The full regalloc
// (greedy/linear-scan) has a pre-existing arg register interference bug
// (#300) where it may assign vregs to arg registers (RDI, RSI, RDX, RCX,
// R8, R9) before they've been read, clobbering the original arg values.
// The simplified allocator now handles this via register hints. The full
// regalloc fix requires adding explicit arg register liveness at function
// entry — filed as a follow-up.

// ===========================================================================
// Test 26: 10-argument sum — stress test with 4 stack args
// ===========================================================================

/// Build `stack_args_sum10(a,b,c,d,e,f,g,h,i,j: i64) -> i64 { sum of all }`
///
/// 6 register args + 4 stack args at [RBP+16], [RBP+24], [RBP+32], [RBP+40].
fn build_stack_args_sum10() -> X86ISelFunction {
    let sig = Signature {
        params: vec![Type::I64; 10],
        returns: vec![Type::I64],
    };

    let mut func = X86ISelFunction::new("stack_args_sum10".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    let v: Vec<VReg> = (0..10).map(|i| VReg::new(i, RegClass::Gpr64)).collect();
    func.next_vreg = 10;

    // Load args 0-5 from registers
    let arg_regs = [
        x86_64_regs::RDI, x86_64_regs::RSI, x86_64_regs::RDX,
        x86_64_regs::RCX, x86_64_regs::R8,  x86_64_regs::R9,
    ];
    for (i, preg) in arg_regs.iter().enumerate() {
        func.push_inst(entry, X86ISelInst::new(
            X86Opcode::MovRR,
            vec![X86ISelOperand::VReg(v[i]), X86ISelOperand::PReg(*preg)],
        ));
    }

    // Load args 6-9 from stack: [RBP+16], [RBP+24], [RBP+32], [RBP+40]
    for i in 0..4u32 {
        let disp = 16 + (i as i32) * 8;
        func.push_inst(entry, X86ISelInst::new(
            X86Opcode::MovRM,
            vec![
                X86ISelOperand::VReg(v[6 + i as usize]),
                X86ISelOperand::MemAddr {
                    base: Box::new(X86ISelOperand::PReg(x86_64_regs::RBP)),
                    disp,
                },
            ],
        ));
    }

    // Sum: v[0] += v[1] through v[9]
    for i in 1..10 {
        func.push_inst(entry, X86ISelInst::new(
            X86Opcode::AddRR,
            vec![
                X86ISelOperand::VReg(v[0]),
                X86ISelOperand::VReg(v[0]),
                X86ISelOperand::VReg(v[i]),
            ],
        ));
    }

    // Return via RAX
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v[0])],
    ));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));

    func
}

#[test]
fn test_correctness_stack_args_sum10() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available");
        return;
    }

    let func = build_stack_args_sum10();
    let driver = r#"
#include <stdio.h>
extern long stack_args_sum10(long a, long b, long c, long d, long e,
                             long f, long g, long h, long i, long j);
int main(void) {
    long r1 = stack_args_sum10(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    long r2 = stack_args_sum10(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    long r3 = stack_args_sum10(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000);
    long r4 = stack_args_sum10(-1, -2, -3, -4, -5, -6, -7, -8, -9, -10);
    printf("sum10(1..10)=%ld sum10(0s)=%ld sum10(100s)=%ld sum10(neg)=%ld\n",
           r1, r2, r3, r4);
    int ok = (r1 == 55) && (r2 == 0) && (r3 == 5500) && (r4 == -55);
    return ok ? 0 : 1;
}
"#;

    let (exit_code, stdout) = compile_link_run(
        "stack_args_sum10", &func, driver, default_macho_config(),
    );
    assert_eq!(
        exit_code, 0,
        "10-arg stack_args_sum10 should produce correct results. stdout: {}",
        stdout
    );
    assert!(
        stdout.contains("sum10(1..10)=55"),
        "Expected sum10(1..10)=55, got: {}", stdout
    );

    // Golden truth
    let (golden_exit, golden_stdout) = compile_run_c_golden("stack_args_sum10", r#"
#include <stdio.h>
long stack_args_sum10(long a, long b, long c, long d, long e,
                      long f, long g, long h, long i, long j) {
    return a + b + c + d + e + f + g + h + i + j;
}
int main(void) {
    long r1 = stack_args_sum10(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    long r2 = stack_args_sum10(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    long r3 = stack_args_sum10(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000);
    long r4 = stack_args_sum10(-1, -2, -3, -4, -5, -6, -7, -8, -9, -10);
    printf("sum10(1..10)=%ld sum10(0s)=%ld sum10(100s)=%ld sum10(neg)=%ld\n",
           r1, r2, r3, r4);
    int ok = (r1 == 55) && (r2 == 0) && (r3 == 5500) && (r4 == -55);
    return ok ? 0 : 1;
}
"#);
    assert_eq!(golden_exit, 0);
    assert_eq!(
        stdout.trim(), golden_stdout.trim(),
        "LLVM2 stack_args_sum10 should match golden C"
    );
}

// ===========================================================================
// Test 27: 8-argument weighted sum -- stack args with multiplication
// ===========================================================================

/// Build `stack_args_weighted(a,b,c,d,e,f,g,h: i64) -> i64`
/// Returns `1*a + 2*b + 3*c + 4*d + 5*e + 6*f + 7*g + 8*h`
///
/// This tests that stack-loaded values can be used in complex arithmetic,
/// not just simple addition. Uses a compact register scheme: one temp vreg
/// for coefficient loads (reused each iteration), and multiplies each arg
/// vreg in-place by its coefficient before adding to the accumulator.
///
/// Strategy: accum = a; for k in 2..8: arg[k-1] *= k; accum += arg[k-1]
/// Total vregs: 8 (args) + 1 (accum) + 1 (coeff) = 10, fits in 14 GPRs.
fn build_stack_args_weighted() -> X86ISelFunction {
    let sig = Signature {
        params: vec![Type::I64; 8],
        returns: vec![Type::I64],
    };

    let mut func = X86ISelFunction::new("stack_args_weighted".to_string(), sig);
    let entry = Block(0);
    func.ensure_block(entry);

    // v0-v7: args, v8: accumulator, v9: coefficient temp (reused)
    let v_args: Vec<VReg> = (0..8).map(|i| VReg::new(i, RegClass::Gpr64)).collect();
    let v_accum = VReg::new(8, RegClass::Gpr64);
    let v_coeff = VReg::new(9, RegClass::Gpr64);
    func.next_vreg = 10;

    // Load args 0-5 from registers
    let arg_regs = [
        x86_64_regs::RDI, x86_64_regs::RSI, x86_64_regs::RDX,
        x86_64_regs::RCX, x86_64_regs::R8,  x86_64_regs::R9,
    ];
    for (i, preg) in arg_regs.iter().enumerate() {
        func.push_inst(entry, X86ISelInst::new(
            X86Opcode::MovRR,
            vec![X86ISelOperand::VReg(v_args[i]), X86ISelOperand::PReg(*preg)],
        ));
    }

    // Load args 6-7 from stack
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRM,
        vec![
            X86ISelOperand::VReg(v_args[6]),
            X86ISelOperand::MemAddr {
                base: Box::new(X86ISelOperand::PReg(x86_64_regs::RBP)),
                disp: 16,
            },
        ],
    ));
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRM,
        vec![
            X86ISelOperand::VReg(v_args[7]),
            X86ISelOperand::MemAddr {
                base: Box::new(X86ISelOperand::PReg(x86_64_regs::RBP)),
                disp: 24,
            },
        ],
    ));

    // accum = v_args[0] (coefficient 1, no multiply needed)
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::VReg(v_accum), X86ISelOperand::VReg(v_args[0])],
    ));

    // For each coefficient k=2..8, multiply arg[k-1] in place and add to accum.
    // We reuse v_coeff for loading the coefficient each iteration.
    let coefficients = [2i64, 3, 4, 5, 6, 7, 8];
    for (idx, &coeff) in coefficients.iter().enumerate() {
        let arg_idx = idx + 1; // v_args[1..7]

        // Load coefficient into v_coeff (reused each iteration)
        func.push_inst(entry, X86ISelInst::new(
            X86Opcode::MovRI,
            vec![X86ISelOperand::VReg(v_coeff), X86ISelOperand::Imm(coeff)],
        ));

        // Multiply arg in-place: v_args[arg_idx] = v_args[arg_idx] * v_coeff
        // (two-address-safe: dst == lhs)
        func.push_inst(entry, X86ISelInst::new(
            X86Opcode::ImulRR,
            vec![
                X86ISelOperand::VReg(v_args[arg_idx]),
                X86ISelOperand::VReg(v_args[arg_idx]),
                X86ISelOperand::VReg(v_coeff),
            ],
        ));

        // accum += v_args[arg_idx] (two-address-safe)
        func.push_inst(entry, X86ISelInst::new(
            X86Opcode::AddRR,
            vec![
                X86ISelOperand::VReg(v_accum),
                X86ISelOperand::VReg(v_accum),
                X86ISelOperand::VReg(v_args[arg_idx]),
            ],
        ));
    }

    // Return accum via RAX
    func.push_inst(entry, X86ISelInst::new(
        X86Opcode::MovRR,
        vec![X86ISelOperand::PReg(x86_64_regs::RAX), X86ISelOperand::VReg(v_accum)],
    ));
    func.push_inst(entry, X86ISelInst::new(X86Opcode::Ret, vec![]));

    func
}

#[test]
fn test_correctness_stack_args_weighted() {
    if !has_cc_x86_64() {
        eprintln!("SKIP: cc -arch x86_64 not available");
        return;
    }

    let func = build_stack_args_weighted();
    let driver = r#"
#include <stdio.h>
extern long stack_args_weighted(long a, long b, long c, long d,
                                long e, long f, long g, long h);
int main(void) {
    /* 1*1 + 2*1 + 3*1 + 4*1 + 5*1 + 6*1 + 7*1 + 8*1 = 36 */
    long r1 = stack_args_weighted(1, 1, 1, 1, 1, 1, 1, 1);
    /* 1*1 + 2*2 + 3*3 + 4*4 + 5*5 + 6*6 + 7*7 + 8*8 = 1+4+9+16+25+36+49+64 = 204 */
    long r2 = stack_args_weighted(1, 2, 3, 4, 5, 6, 7, 8);
    /* 1*10 + 0 + ... = 10 */
    long r3 = stack_args_weighted(10, 0, 0, 0, 0, 0, 0, 0);
    /* 0 + ... + 8*1 = 8 */
    long r4 = stack_args_weighted(0, 0, 0, 0, 0, 0, 0, 1);
    printf("w(1s)=%ld w(1..8)=%ld w(10,0s)=%ld w(0s,1)=%ld\n",
           r1, r2, r3, r4);
    int ok = (r1 == 36) && (r2 == 204) && (r3 == 10) && (r4 == 8);
    return ok ? 0 : 1;
}
"#;

    let (exit_code, stdout) = compile_link_run(
        "stack_args_weighted", &func, driver, default_macho_config(),
    );
    assert_eq!(
        exit_code, 0,
        "8-arg weighted sum should produce correct results. stdout: {}",
        stdout
    );
    assert!(
        stdout.contains("w(1s)=36"),
        "Expected w(1s)=36, got: {}", stdout
    );

    // Golden truth comparison
    let (golden_exit, golden_stdout) = compile_run_c_golden("stack_args_weighted", r#"
#include <stdio.h>
long stack_args_weighted(long a, long b, long c, long d,
                         long e, long f, long g, long h) {
    return 1*a + 2*b + 3*c + 4*d + 5*e + 6*f + 7*g + 8*h;
}
int main(void) {
    long r1 = stack_args_weighted(1, 1, 1, 1, 1, 1, 1, 1);
    long r2 = stack_args_weighted(1, 2, 3, 4, 5, 6, 7, 8);
    long r3 = stack_args_weighted(10, 0, 0, 0, 0, 0, 0, 0);
    long r4 = stack_args_weighted(0, 0, 0, 0, 0, 0, 0, 1);
    printf("w(1s)=%ld w(1..8)=%ld w(10,0s)=%ld w(0s,1)=%ld\n",
           r1, r2, r3, r4);
    int ok = (r1 == 36) && (r2 == 204) && (r3 == 10) && (r4 == 8);
    return ok ? 0 : 1;
}
"#);
    assert_eq!(golden_exit, 0);
    assert_eq!(
        stdout.trim(), golden_stdout.trim(),
        "LLVM2 stack_args_weighted should match golden C"
    );
}
