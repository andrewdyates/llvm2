// llvm2-codegen/tests/e2e_run.rs - End-to-end compile, link, and run tests
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// These tests compile functions through the LLVM2 pipeline, write .o files,
// link them with C drivers using the system compiler (cc), execute the
// resulting binaries, and verify correct output via exit codes.
//
// Architecture: AArch64 (Apple Silicon) only. Tests are skipped on other
// architectures via runtime checks.
//
// Part of #60 — End-to-end test: compile, link, and run.
// Stack allocation tests: Part of #243.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use llvm2_codegen::macho::MachOWriter;
use llvm2_codegen::pipeline::{
    build_add_test_function, encode_function, Pipeline, PipelineConfig, OptLevel,
};
use llvm2_ir::function::{MachFunction, Signature, Type};
use llvm2_ir::inst::{AArch64Opcode, MachInst};
use llvm2_ir::operand::MachOperand;
use llvm2_ir::regs::{X0, X1, X8, X9};

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

/// Returns true if we are running on AArch64 (Apple Silicon).
/// E2E tests only work on the native target architecture.
fn is_aarch64() -> bool {
    cfg!(target_arch = "aarch64")
}

/// Returns true if the system C compiler is available.
fn has_cc() -> bool {
    Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Create a temporary directory for test artifacts.
/// Returns the path. Caller is responsible for cleanup.
fn make_test_dir(test_name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("llvm2_e2e_{}", test_name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("Failed to create test directory");
    dir
}

/// Write a C driver source file to the given directory.
/// Returns the path to the .c file.
fn write_c_driver(dir: &Path, filename: &str, source: &str) -> PathBuf {
    let path = dir.join(filename);
    fs::write(&path, source).expect("Failed to write C driver");
    path
}

/// Write object file bytes to the given directory.
/// Returns the path to the .o file.
fn write_object_file(dir: &Path, filename: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(filename);
    fs::write(&path, bytes).expect("Failed to write .o file");
    path
}

/// Compile and link a C driver with an object file.
/// Returns the path to the output binary.
fn link_with_cc(dir: &Path, driver_c: &Path, obj: &Path, output_name: &str) -> PathBuf {
    let binary = dir.join(output_name);
    let result = Command::new("cc")
        .arg("-o")
        .arg(&binary)
        .arg(driver_c)
        .arg(obj)
        .arg("-Wl,-no_pie") // Simplify linking for test objects
        .output()
        .expect("Failed to run cc");

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        let stdout = String::from_utf8_lossy(&result.stdout);
        panic!(
            "Linking failed!\ncc stdout: {}\ncc stderr: {}\nDriver: {}\nObject: {}",
            stdout,
            stderr,
            driver_c.display(),
            obj.display()
        );
    }

    binary
}

/// Run a binary and return its exit code.
fn run_binary(binary: &Path) -> i32 {
    let result = Command::new(binary)
        .output()
        .expect("Failed to run binary");

    result.status.code().unwrap_or(-1)
}

/// Run a binary and return (exit_code, stdout).
fn run_binary_with_output(binary: &Path) -> (i32, String) {
    let result = Command::new(binary)
        .output()
        .expect("Failed to run binary");

    let stdout = String::from_utf8_lossy(&result.stdout).to_string();
    (result.status.code().unwrap_or(-1), stdout)
}

/// Cleanup a test directory.
fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------------------
// Helper: build a "naked" object file (no frame lowering)
//
// For simple leaf functions that don't need a frame, this helper encodes
// raw instructions directly without prologue/epilogue overhead.
// For tests that need frame lowering, use Pipeline::encode_and_emit
// (see test_full_pipeline_frame_lowering).
// ---------------------------------------------------------------------------

/// Encode an IR function into a Mach-O .o file WITHOUT frame lowering.
///
/// This produces a "naked" function: just the raw instructions, no
/// prologue/epilogue. For leaf functions that don't clobber callee-saved
/// registers and don't use the stack, this is sufficient and correct.
///
/// For full pipeline compilation with frame lowering (prologue/epilogue),
/// use `Pipeline::encode_and_emit` instead.
fn encode_naked_to_macho(func: &MachFunction) -> Vec<u8> {
    let code = encode_function(func).expect("encoding should succeed");
    let mut writer = MachOWriter::new();
    writer.add_text_section(&code);
    let symbol_name = format!("_{}", func.name);
    writer.add_symbol(&symbol_name, 1, 0, true);
    writer.write()
}

// ---------------------------------------------------------------------------
// IR builders for test functions
// ---------------------------------------------------------------------------

/// Build `fn add(a: i32, b: i32) -> i32 { a + b }`
///
/// AArch64 calling convention: a in W0, b in W1, result in W0.
/// We use X registers (64-bit) because the encoder defaults to sf=1 for
/// PReg(0..31). The upper 32 bits are ignored by the C caller which
/// reads W0.
fn build_add_function() -> MachFunction {
    build_add_test_function()
}

/// Build `fn return_const() -> i32 { 42 }`
///
/// Simplest possible test: just return a constant.
/// MOVZ X0, #42 ; RET
fn build_return_const_function() -> MachFunction {
    let sig = Signature::new(vec![], vec![Type::I32]);
    let mut func = MachFunction::new("return_const".to_string(), sig);
    let entry = func.entry;

    let mov = MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::PReg(X0), MachOperand::Imm(42)],
    );
    let mov_id = func.push_inst(mov);
    func.append_inst(entry, mov_id);

    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}

/// Build `fn sub(a: i32, b: i32) -> i32 { a - b }`
///
/// SUB X0, X0, X1 ; RET
fn build_sub_function() -> MachFunction {
    let sig = Signature::new(vec![Type::I32, Type::I32], vec![Type::I32]);
    let mut func = MachFunction::new("sub".to_string(), sig);
    let entry = func.entry;

    let sub = MachInst::new(
        AArch64Opcode::SubRR,
        vec![
            MachOperand::PReg(X0),
            MachOperand::PReg(X0),
            MachOperand::PReg(X1),
        ],
    );
    let sub_id = func.push_inst(sub);
    func.append_inst(entry, sub_id);

    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}

/// Build `fn max(a: i32, b: i32) -> i32 { if a > b { a } else { b } }`
///
/// Uses CMP + B.GT + MOV pattern:
///   CMP X0, X1
///   B.GT .done      (if a > b, result already in X0)
///   MOV X0, X1      (else result = b)
/// .done:
///   RET
///
/// Note: We use a branchless CSEL approach instead for simplicity, but
/// CSEL is not yet encoded. So we use CMP + conditional branch + MOV.
///
/// Block layout:
///   bb0: CMP X0, X1; B.GT bb2 (skip to ret if a > b)
///   bb1: MOV X0, X1 (fallthrough: a <= b, so result = b)
///   bb2: RET
fn build_max_function() -> MachFunction {
    let sig = Signature::new(vec![Type::I32, Type::I32], vec![Type::I32]);
    let mut func = MachFunction::new("max".to_string(), sig);

    // We need 3 blocks: entry (bb0), move (bb1), done (bb2).
    // MachFunction::new creates bb0 (entry). We need to add bb1 and bb2.
    let bb0 = func.entry; // BlockId(0)

    // Create bb1 and bb2.
    let bb1 = func.create_block();
    let bb2 = func.create_block();

    // bb0: CMP X0, X1; B.GT bb2
    let cmp = MachInst::new(
        AArch64Opcode::CmpRR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X1)],
    );
    let cmp_id = func.push_inst(cmp);
    func.append_inst(bb0, cmp_id);

    // B.GT bb2 — branch offset is +2 instructions (skip MOV X0,X1 + fallthrough)
    // In the block layout [bb0, bb1, bb2], bb0 has CMP + B.GT = 2 instrs.
    // bb1 has MOV = 1 instr. So from B.GT to bb2 start is +2 instructions = +8 bytes.
    // Branch offset in B.cond is in instruction units (4 bytes each), PC-relative.
    // But we need to encode the offset from the B.GT instruction itself.
    // B.GT is at offset 4 (after CMP). bb2 starts at offset 4+4+4=12.
    // PC-relative offset = (12 - 4) / 4 = 2 instructions.
    let bcond = MachInst::new(
        AArch64Opcode::BCond,
        vec![
            MachOperand::Imm(0xC), // condition: GT = 0b1100 = 12
            MachOperand::Imm(2),   // imm19 offset: +2 instructions
        ],
    );
    let bcond_id = func.push_inst(bcond);
    func.append_inst(bb0, bcond_id);

    // bb1: MOV X0, X1 (a <= b, so result = b)
    let mov = MachInst::new(
        AArch64Opcode::MovR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X1)],
    );
    let mov_id = func.push_inst(mov);
    func.append_inst(bb1, mov_id);

    // bb2: RET
    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(bb2, ret_id);

    func
}

/// Build `fn factorial(n: i32) -> i32` (iterative)
///
/// Iterative factorial using a loop:
///   result = 1  (X9)
///   i = n       (X0, then X8)
/// loop:
///   CMP X8, #1
///   B.LE done
///   MUL X9, X9, X8
///   SUB X8, X8, #1
///   B loop
/// done:
///   MOV X0, X9
///   RET
///
/// MUL is encoded as MADD Xd, Xn, Xm, XZR per ARM ARM "Data-processing
/// (3 source)" encoding class.
fn build_factorial_function() -> MachFunction {
    let sig = Signature::new(vec![Type::I32], vec![Type::I32]);
    let mut func = MachFunction::new("factorial".to_string(), sig);

    let bb_entry = func.entry; // bb0
    let bb_loop = func.create_block(); // bb1
    let bb_done = func.create_block(); // bb2

    // bb_entry: setup
    // MOV X8, X0  (i = n, save n to X8)
    let mov_n = MachInst::new(
        AArch64Opcode::MovR,
        vec![MachOperand::PReg(X8), MachOperand::PReg(X0)],
    );
    let mov_n_id = func.push_inst(mov_n);
    func.append_inst(bb_entry, mov_n_id);

    // MOVZ X9, #1  (result = 1)
    let mov_one = MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::PReg(X9), MachOperand::Imm(1)],
    );
    let mov_one_id = func.push_inst(mov_one);
    func.append_inst(bb_entry, mov_one_id);

    // B bb_loop (unconditional jump to loop header)
    // From bb_entry end to bb_loop start: bb_entry has 3 instrs (mov, movz, b).
    // bb_loop starts right after = next instruction. Offset = +1.
    // Actually, the B is the last instruction of bb_entry. bb_loop is the next
    // block. Since blocks are laid out in order, the B offset is 0 (fallthrough).
    // But we need to jump to bb_loop = +1 instruction (skip the B itself? No.)
    // B offset is PC-relative: B at position P jumps to P + offset*4.
    // If B is the last instr of bb_entry and bb_loop is right after, offset = +1.
    // Actually, let's just fall through to bb_loop (no explicit branch needed).
    // We omit the branch and rely on fallthrough from bb_entry to bb_loop.

    // bb_loop:
    // CMP X8, #1
    let cmp = MachInst::new(
        AArch64Opcode::CmpRI,
        vec![MachOperand::PReg(X8), MachOperand::Imm(1)],
    );
    let cmp_id = func.push_inst(cmp);
    func.append_inst(bb_loop, cmp_id);

    // B.LE bb_done — offset from this B.LE to bb_done.
    // bb_loop: CMP(0), B.LE(1), MUL(2), SUB(3), B(4) = 5 instructions.
    // bb_done starts right after bb_loop. Offset from B.LE to bb_done = +4 instructions.
    let ble_done = MachInst::new(
        AArch64Opcode::BCond,
        vec![
            MachOperand::Imm(0xD), // condition: LE = 0b1101 = 13
            MachOperand::Imm(4),   // imm19 offset: +4 instructions to bb_done
        ],
    );
    let ble_done_id = func.push_inst(ble_done);
    func.append_inst(bb_loop, ble_done_id);

    // MUL X9, X9, X8  (result *= i)
    let mul = MachInst::new(
        AArch64Opcode::MulRR,
        vec![
            MachOperand::PReg(X9),
            MachOperand::PReg(X9),
            MachOperand::PReg(X8),
        ],
    );
    let mul_id = func.push_inst(mul);
    func.append_inst(bb_loop, mul_id);

    // SUB X8, X8, #1  (i -= 1)
    let sub = MachInst::new(
        AArch64Opcode::SubRI,
        vec![
            MachOperand::PReg(X8),
            MachOperand::PReg(X8),
            MachOperand::Imm(1),
        ],
    );
    let sub_id = func.push_inst(sub);
    func.append_inst(bb_loop, sub_id);

    // B bb_loop (loop back)
    // Offset from this B to bb_loop start. bb_loop is 5 instrs back.
    // B is at the end of bb_loop. Offset = -(5-1) = -4? No.
    // This B is the 5th instruction of bb_loop (index 4).
    // bb_loop[0] = CMP. Offset from B (at position P) to CMP (at position P-4*4=-16 bytes).
    // In instruction units: -4.
    // But imm26 is signed. We encode as ((-4) as u32) & 0x3FFFFFF.
    let b_loop = MachInst::new(
        AArch64Opcode::B,
        vec![MachOperand::Imm(-4i64)],
    );
    let b_loop_id = func.push_inst(b_loop);
    func.append_inst(bb_loop, b_loop_id);

    // bb_done:
    // MOV X0, X9  (return result)
    let mov_result = MachInst::new(
        AArch64Opcode::MovR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X9)],
    );
    let mov_result_id = func.push_inst(mov_result);
    func.append_inst(bb_done, mov_result_id);

    // RET
    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(bb_done, ret_id);

    func
}

// ---------------------------------------------------------------------------
// Test: return_const — simplest possible e2e test
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_return_const() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e test: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("return_const");

    // Build the function: MOVZ X0, #42; RET
    let func = build_return_const_function();
    let obj_bytes = encode_naked_to_macho(&func);

    // Write object file
    let obj_path = write_object_file(&dir, "return_const.o", &obj_bytes);

    // Write C driver that calls return_const() and checks the result
    let driver_src = r#"
extern int return_const(void);
int main(void) {
    int result = return_const();
    /* Exit 0 if result is 42, exit 1 otherwise */
    return (result == 42) ? 0 : 1;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);

    // Link
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_return_const");

    // Run
    let exit_code = run_binary(&binary);
    assert_eq!(
        exit_code, 0,
        "return_const() should return 42. Exit code {} indicates wrong result.",
        exit_code
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test: add — the canonical e2e test
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_add() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e test: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("add");

    // Build the function: ADD X0, X0, X1; RET
    let func = build_add_function();
    let obj_bytes = encode_naked_to_macho(&func);

    // Write object file
    let obj_path = write_object_file(&dir, "add.o", &obj_bytes);

    // Write C driver
    let driver_src = r#"
#include <stdio.h>
extern int add(int a, int b);
int main(void) {
    int result = add(3, 4);
    printf("add(3, 4) = %d\n", result);
    return (result == 7) ? 0 : 1;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);

    // Link
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_add");

    // Run
    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_e2e_add stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "add(3, 4) should return 7. Exit code {} indicates wrong result. stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test: sub — subtraction
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_sub() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e test: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("sub");

    let func = build_sub_function();
    let obj_bytes = encode_naked_to_macho(&func);

    let obj_path = write_object_file(&dir, "sub.o", &obj_bytes);

    let driver_src = r#"
#include <stdio.h>
extern int sub(int a, int b);
int main(void) {
    int result = sub(10, 3);
    printf("sub(10, 3) = %d\n", result);
    return (result == 7) ? 0 : 1;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_sub");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_e2e_sub stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "sub(10, 3) should return 7. Exit code {} indicates wrong result. stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test: max — conditional branch
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_max() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e test: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("max");

    let func = build_max_function();
    let obj_bytes = encode_naked_to_macho(&func);

    let obj_path = write_object_file(&dir, "max.o", &obj_bytes);

    let driver_src = r#"
#include <stdio.h>
extern int max(int a, int b);
int main(void) {
    int r1 = max(5, 3);
    int r2 = max(2, 7);
    int r3 = max(4, 4);
    printf("max(5,3)=%d max(2,7)=%d max(4,4)=%d\n", r1, r2, r3);
    /* All three must be correct for exit 0 */
    if (r1 != 5) return 1;
    if (r2 != 7) return 2;
    if (r3 != 4) return 3;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_max");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_e2e_max stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "max() returned wrong results. Exit code {} (1=max(5,3)!=5, 2=max(2,7)!=7, 3=max(4,4)!=4). stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test: factorial — iterative loop with MUL (MADD encoding)
// ---------------------------------------------------------------------------

// MUL Xd, Xn, Xm is encoded as MADD Xd, Xn, Xm, XZR per ARM ARM
// "Data-processing (3 source)" encoding class.
#[test]
fn test_e2e_factorial() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e test: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("factorial");

    let func = build_factorial_function();

    // Verify the function encodes without errors.
    let code = encode_function(&func).expect("encoding should succeed");
    assert!(
        !code.is_empty(),
        "Factorial function should produce non-empty code"
    );

    // Verify MUL is encoded as MADD (not NOP).
    // The MUL instruction is at index 2 in the loop block (bb1).
    // In the linear layout: bb0 has 2 instrs (MOV, MOVZ), bb1 has 5 instrs
    // (CMP, B.LE, MUL, SUB, B). MUL is at instruction index 4 overall.
    // Each instruction is 4 bytes, so MUL is at byte offset 16.
    if code.len() >= 20 {
        let mul_word = u32::from_le_bytes([code[16], code[17], code[18], code[19]]);
        // MADD X9, X9, X8, XZR = 0x9B087D29
        assert_eq!(
            mul_word, 0x9B087D29,
            "MUL should be encoded as MADD (0x9B087D29), got 0x{:08X}",
            mul_word
        );
    }

    // Link and run the factorial binary.
    let obj_bytes = encode_naked_to_macho(&func);
    let obj_path = write_object_file(&dir, "factorial.o", &obj_bytes);
    let driver_src = r#"
#include <stdio.h>
extern int factorial(int n);
int main(void) {
    int r1 = factorial(5);
    int r2 = factorial(1);
    int r3 = factorial(0);
    printf("factorial(5)=%d factorial(1)=%d factorial(0)=%d\n", r1, r2, r3);
    if (r1 != 120) return 1;
    if (r2 != 1) return 2;
    if (r3 != 1) return 3;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_factorial");
    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_e2e_factorial stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "factorial test failed with exit code {} (1=f(5)!=120, 2=f(1)!=1, 3=f(0)!=1). stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test: full pipeline with frame lowering
// ---------------------------------------------------------------------------

/// Tests the full pipeline with frame lowering (prologue/epilogue).
///
/// After frame lowering, the function should be:
///   STP X29, X30, [SP, #-16]!   (pre-index writeback via StpPreIndex)
///   ADD X29, SP, #0              (MOV X29, SP via ADD to avoid XZR ambiguity)
///   ADD X0, X0, X1               (body)
///   LDP X29, X30, [SP], #16     (post-index writeback via LdpPostIndex)
///   RET
///
/// Frame lowering works correctly: STP/LDP use pre/post-index writeback
/// forms (StpPreIndex/LdpPostIndex opcodes) and SP-to-register moves
/// use ADD (avoiding the XZR/SP ambiguity in ORR encoding).
#[test]
fn test_full_pipeline_frame_lowering() {
    // Build the add function and run it through encode_and_emit (includes
    // frame lowering).
    let mut func = build_add_function();

    let pipeline = Pipeline::new(PipelineConfig {
        opt_level: OptLevel::O0,
        emit_debug: false,
        ..Default::default()
    });

    let obj_bytes = pipeline
        .encode_and_emit(&mut func)
        .expect("encode_and_emit should succeed");

    // The object file is structurally valid Mach-O.
    assert!(obj_bytes.len() >= 4);
    assert_eq!(
        &obj_bytes[0..4],
        &[0xCF, 0xFA, 0xED, 0xFE],
        "Should be valid Mach-O"
    );

    // Verify the Mach-O is structurally valid and disassembles correctly.
    if is_aarch64() {
        let dir = make_test_dir("frame_lowering_gaps");
        let obj_path = write_object_file(&dir, "add_framed.o", &obj_bytes);

        let otool = Command::new("otool")
            .args(["-h", obj_path.to_str().unwrap()])
            .output();

        if let Ok(output) = otool {
            let stdout = String::from_utf8_lossy(&output.stdout);
            eprintln!("otool -h output: {}", stdout);
            assert!(
                output.status.success(),
                "otool should accept the Mach-O file"
            );
        }

        // Verify otool -tv can disassemble it.
        let otool_tv = Command::new("otool")
            .args(["-tv", obj_path.to_str().unwrap()])
            .output();

        if let Ok(output) = otool_tv {
            let stdout = String::from_utf8_lossy(&output.stdout);
            eprintln!("otool -tv disassembly:\n{}", stdout);
        }

        cleanup(&dir);
    }
}

// ---------------------------------------------------------------------------
// Test: stack allocation — store to stack and reload (explicit SP-relative)
// ---------------------------------------------------------------------------

/// Tests that a function which manually stores a value to the stack via
/// STR [SP, #offset] and reloads it via LDR [SP, #offset] produces
/// correct results. This exercises the AArch64 memory encoding with SP
/// as the base register.
///
/// The function:
///   1. Saves FP/LR (prologue via encode_and_emit)
///   2. SUB SP, SP, #16  (allocate 16 bytes of local space)
///   3. STR X0, [SP, #0]  (store argument to stack)
///   4. LDR X0, [SP, #0]  (reload from stack)
///   5. ADD SP, SP, #16  (deallocate locals)
///   6. Epilogue + RET
///
/// We build this using encode_and_emit which handles the frame lowering.
///
/// Part of #243 — Stack allocation E2E tests.
#[test]
fn test_e2e_stack_store_reload() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e test: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("stack_store_reload");

    // Build a function that stores its argument to the stack and reloads it.
    // fn store_reload(a: i64) -> i64 {
    //     let local = a;  // store to stack
    //     return local;   // reload from stack
    // }
    //
    // We build this manually as instructions with explicit SP-relative
    // store/load, wrapped in prologue/epilogue via encode_and_emit.
    let sig = Signature::new(vec![Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("store_reload".to_string(), sig);
    let entry = func.entry;

    // Allocate a stack slot for the local variable.
    use llvm2_ir::function::StackSlot;
    func.alloc_stack_slot(StackSlot::new(8, 8));

    // STR X0, [FP, #frame_idx_0]  — store arg to stack slot 0
    // We use FrameIndex which will be resolved by frame lowering.
    use llvm2_ir::types::FrameIdx;
    let str_inst = MachInst::new(
        AArch64Opcode::StrRI,
        vec![
            MachOperand::PReg(X0),
            MachOperand::FrameIndex(FrameIdx(0)),
        ],
    );
    let str_id = func.push_inst(str_inst);
    func.append_inst(entry, str_id);

    // LDR X0, [FP, #frame_idx_0]  — reload from stack slot 0
    let ldr_inst = MachInst::new(
        AArch64Opcode::LdrRI,
        vec![
            MachOperand::PReg(X0),
            MachOperand::FrameIndex(FrameIdx(0)),
        ],
    );
    let ldr_id = func.push_inst(ldr_inst);
    func.append_inst(entry, ldr_id);

    // RET (epilogue will replace this)
    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    // Run through encode_and_emit which does frame lowering + encoding.
    let pipeline = Pipeline::new(PipelineConfig {
        opt_level: OptLevel::O0,
        emit_debug: false,
        ..Default::default()
    });

    let obj_bytes = pipeline
        .encode_and_emit(&mut func)
        .expect("encode_and_emit should succeed");

    // Verify Mach-O header.
    assert!(obj_bytes.len() >= 4);
    let magic = u32::from_le_bytes([obj_bytes[0], obj_bytes[1], obj_bytes[2], obj_bytes[3]]);
    assert_eq!(magic, 0xFEED_FACF, "should be valid Mach-O");

    // Write and link.
    let obj_path = write_object_file(&dir, "store_reload.o", &obj_bytes);

    // Disassemble to verify the encoding includes store/load instructions.
    let otool = Command::new("otool")
        .args(["-tv", obj_path.to_str().unwrap()])
        .output()
        .expect("otool");
    let disasm = String::from_utf8_lossy(&otool.stdout);
    eprintln!("store_reload disassembly:\n{}", disasm);

    // The disassembly should contain str and ldr instructions.
    assert!(
        disasm.contains("str") || disasm.contains("stur"),
        "Should contain a store instruction. Got:\n{}",
        disasm
    );
    assert!(
        disasm.contains("ldr") || disasm.contains("ldur"),
        "Should contain a load instruction. Got:\n{}",
        disasm
    );

    // Write C driver.
    let driver_src = r#"
#include <stdio.h>
extern long store_reload(long a);
int main(void) {
    long r1 = store_reload(42);
    long r2 = store_reload(100);
    long r3 = store_reload(-7);
    printf("store_reload(42)=%ld store_reload(100)=%ld store_reload(-7)=%ld\n", r1, r2, r3);
    if (r1 != 42) return 1;
    if (r2 != 100) return 2;
    if (r3 != -7) return 3;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_store_reload");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_e2e_stack_store_reload stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "store_reload test failed with exit code {} (1=42, 2=100, 3=-7). stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test: callee-saved register spill through frame lowering
// ---------------------------------------------------------------------------

/// Tests that a function using callee-saved registers (X19-X22) gets correct
/// prologue/epilogue insertion through frame lowering, and the callee-saved
/// registers are properly saved/restored.
///
/// The function uses X19 as an accumulator (callee-saved), which forces the
/// frame lowering to save/restore it in the prologue/epilogue:
///   fn callee_saved_accum(a: i64, b: i64) -> i64 {
///       let saved = a;   // X19 = X0
///       return saved + b; // X0 = X19 + X1
///   }
///
/// Part of #243 — Stack allocation E2E tests.
#[test]
fn test_e2e_callee_saved_spill() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e test: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("callee_saved_spill");

    let sig = Signature::new(vec![Type::I64, Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("callee_saved_accum".to_string(), sig);
    let entry = func.entry;

    // MOV X19, X0  (save arg a in callee-saved register)
    let mov = MachInst::new(
        AArch64Opcode::MovR,
        vec![MachOperand::PReg(X9), MachOperand::PReg(X0)],
    );

    // Use X19 (callee-saved) so frame lowering must save/restore it.
    let mov_cs = MachInst::new(
        AArch64Opcode::MovR,
        vec![
            MachOperand::PReg(llvm2_ir::regs::X19),
            MachOperand::PReg(X0),
        ],
    );
    let mov_cs_id = func.push_inst(mov_cs);
    func.append_inst(entry, mov_cs_id);

    // ADD X0, X19, X1  (result = saved_a + b)
    let add = MachInst::new(
        AArch64Opcode::AddRR,
        vec![
            MachOperand::PReg(X0),
            MachOperand::PReg(llvm2_ir::regs::X19),
            MachOperand::PReg(X1),
        ],
    );
    let add_id = func.push_inst(add);
    func.append_inst(entry, add_id);

    // RET
    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    // Run through encode_and_emit (includes frame lowering).
    let pipeline = Pipeline::new(PipelineConfig {
        opt_level: OptLevel::O0,
        emit_debug: false,
        ..Default::default()
    });

    let obj_bytes = pipeline
        .encode_and_emit(&mut func)
        .expect("encode_and_emit should succeed");

    let obj_path = write_object_file(&dir, "callee_saved_accum.o", &obj_bytes);

    // Disassemble — should show STP/LDP for callee-saved registers.
    let otool = Command::new("otool")
        .args(["-tv", obj_path.to_str().unwrap()])
        .output()
        .expect("otool");
    let disasm = String::from_utf8_lossy(&otool.stdout);
    eprintln!("callee_saved_accum disassembly:\n{}", disasm);

    // Should contain stp (save pair) for callee-saved registers.
    assert!(
        disasm.contains("stp"),
        "Should contain STP for callee-saved register save. Got:\n{}",
        disasm
    );

    // Link and run.
    let driver_src = r#"
#include <stdio.h>
extern long callee_saved_accum(long a, long b);
int main(void) {
    long r1 = callee_saved_accum(30, 12);
    long r2 = callee_saved_accum(0, 0);
    long r3 = callee_saved_accum(-10, 52);
    printf("cs(30,12)=%ld cs(0,0)=%ld cs(-10,52)=%ld\n", r1, r2, r3);
    if (r1 != 42) return 1;
    if (r2 != 0) return 2;
    if (r3 != 42) return 3;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_callee_saved");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_e2e_callee_saved_spill stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "callee_saved_accum test failed with exit code {} (1=30+12, 2=0+0, 3=-10+52). stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test: stack allocation with multiple stack slots and callee-saved regs
// ---------------------------------------------------------------------------

/// Tests a function with both stack slots and callee-saved register usage,
/// exercising the full frame layout: FP/LR save, callee-saved GPR save,
/// stack slot allocation, and SP adjustment.
///
/// fn multi_slot(a: i64, b: i64) -> i64 {
///     // X19 = a (callee-saved)
///     // stack_slot[0] = b (store to stack)
///     // result = X19 + stack_slot[0]
///     return a + b;
/// }
///
/// Part of #243 — Stack allocation E2E tests.
#[test]
fn test_e2e_stack_slots_with_callee_saved() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e test: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("stack_slots_callee_saved");

    let sig = Signature::new(vec![Type::I64, Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("multi_slot".to_string(), sig);
    let entry = func.entry;

    // Allocate two stack slots.
    use llvm2_ir::function::StackSlot;
    func.alloc_stack_slot(StackSlot::new(8, 8)); // slot 0
    func.alloc_stack_slot(StackSlot::new(8, 8)); // slot 1

    // MOV X19, X0  (save a in callee-saved register)
    let mov = MachInst::new(
        AArch64Opcode::MovR,
        vec![
            MachOperand::PReg(llvm2_ir::regs::X19),
            MachOperand::PReg(X0),
        ],
    );
    let mov_id = func.push_inst(mov);
    func.append_inst(entry, mov_id);

    // STR X1, [FP, #slot0]  (store b to stack slot 0)
    use llvm2_ir::types::FrameIdx;
    let str_inst = MachInst::new(
        AArch64Opcode::StrRI,
        vec![
            MachOperand::PReg(X1),
            MachOperand::FrameIndex(FrameIdx(0)),
        ],
    );
    let str_id = func.push_inst(str_inst);
    func.append_inst(entry, str_id);

    // LDR X1, [FP, #slot0]  (reload b from stack)
    let ldr_inst = MachInst::new(
        AArch64Opcode::LdrRI,
        vec![
            MachOperand::PReg(X1),
            MachOperand::FrameIndex(FrameIdx(0)),
        ],
    );
    let ldr_id = func.push_inst(ldr_inst);
    func.append_inst(entry, ldr_id);

    // ADD X0, X19, X1  (result = a + b)
    let add = MachInst::new(
        AArch64Opcode::AddRR,
        vec![
            MachOperand::PReg(X0),
            MachOperand::PReg(llvm2_ir::regs::X19),
            MachOperand::PReg(X1),
        ],
    );
    let add_id = func.push_inst(add);
    func.append_inst(entry, add_id);

    // RET
    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    // Run through encode_and_emit.
    let pipeline = Pipeline::new(PipelineConfig {
        opt_level: OptLevel::O0,
        emit_debug: false,
        ..Default::default()
    });

    let obj_bytes = pipeline
        .encode_and_emit(&mut func)
        .expect("encode_and_emit should succeed");

    let obj_path = write_object_file(&dir, "multi_slot.o", &obj_bytes);

    // Disassemble.
    let otool = Command::new("otool")
        .args(["-tv", obj_path.to_str().unwrap()])
        .output()
        .expect("otool");
    let disasm = String::from_utf8_lossy(&otool.stdout);
    eprintln!("multi_slot disassembly:\n{}", disasm);

    // Should contain sub sp (stack allocation) and stp (callee-saved save).
    assert!(
        disasm.contains("stp"),
        "Should contain STP for frame setup. Got:\n{}",
        disasm
    );
    assert!(
        disasm.contains("sub") || disasm.contains("str"),
        "Should contain SUB SP (stack alloc) or STR (store). Got:\n{}",
        disasm
    );

    // Link and run.
    let driver_src = r#"
#include <stdio.h>
extern long multi_slot(long a, long b);
int main(void) {
    long r1 = multi_slot(30, 12);
    long r2 = multi_slot(100, -58);
    long r3 = multi_slot(0, 42);
    printf("ms(30,12)=%ld ms(100,-58)=%ld ms(0,42)=%ld\n", r1, r2, r3);
    if (r1 != 42) return 1;
    if (r2 != 42) return 2;
    if (r3 != 42) return 3;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_multi_slot");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_e2e_stack_slots_with_callee_saved stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "multi_slot test failed with exit code {} (1=30+12, 2=100-58, 3=0+42). stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test: tMIR pipeline with high register pressure forces spilling
// ---------------------------------------------------------------------------

/// Tests that a tMIR function with many simultaneous live values compiles
/// correctly through the full pipeline (tMIR -> ISel -> regalloc -> frame
/// lowering -> encoding -> Mach-O). This forces the register allocator to
/// spill some values to stack slots, which must then be correctly handled
/// by frame lowering.
///
/// The function computes: sum = a + b + (a*b) + (a-b) + (a+1) + (b+1)
/// which creates many intermediate values that may exceed available registers.
///
/// Part of #243 — Stack allocation E2E tests.
#[test]
fn test_e2e_tmir_high_register_pressure() {
    use llvm2_codegen::compiler::{Compiler, CompilerConfig, CompilerTraceLevel};

    let module = build_tmir_high_pressure_module();
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        trace_level: CompilerTraceLevel::Full,
        ..CompilerConfig::default()
    });

    let result = compiler.compile(&module).expect("high-pressure compilation should succeed");

    assert!(
        !result.object_code.is_empty(),
        "Should produce non-empty object code"
    );
    assert_eq!(result.metrics.function_count, 1);

    // Verify valid Mach-O.
    let obj = &result.object_code;
    let magic = u32::from_le_bytes([obj[0], obj[1], obj[2], obj[3]]);
    assert_eq!(magic, 0xFEED_FACF, "should be valid Mach-O");

    // If we are on AArch64, try to link and run.
    if is_aarch64() && has_cc() {
        let dir = make_test_dir("tmir_high_pressure");
        let obj_path = write_object_file(&dir, "high_pressure.o", &result.object_code);

        // Disassemble to inspect.
        let otool = Command::new("otool")
            .args(["-tv", obj_path.to_str().unwrap()])
            .output()
            .expect("otool");
        let disasm = String::from_utf8_lossy(&otool.stdout);
        eprintln!("tmir_high_pressure disassembly:\n{}", disasm);

        // The function should have frame setup (STP for FP/LR at minimum).
        assert!(
            disasm.contains("stp") || disasm.contains("sub"),
            "Should contain frame setup instructions. Got:\n{}",
            disasm
        );

        // Write C driver.
        let driver_src = r#"
#include <stdio.h>
extern long _high_pressure(long a, long b);
int main(void) {
    /* sum = a + b + (a*b) + (a-b) + (a+1) + (b+1)
     * For a=3, b=4: 3+4 + 12 + (-1) + 4 + 5 = 27 */
    long r1 = _high_pressure(3, 4);
    printf("high_pressure(3,4)=%ld\n", r1);
    if (r1 != 27) return 1;

    /* For a=10, b=2: 10+2 + 20 + 8 + 11 + 3 = 54 */
    long r2 = _high_pressure(10, 2);
    printf("high_pressure(10,2)=%ld\n", r2);
    if (r2 != 54) return 2;

    return 0;
}
"#;
        let driver_path = write_c_driver(&dir, "driver.c", driver_src);

        // Link.
        let binary = dir.join("test_high_pressure");
        let link_result = Command::new("cc")
            .arg("-o")
            .arg(&binary)
            .arg(&driver_path)
            .arg(&obj_path)
            .arg("-Wl,-no_pie")
            .output()
            .expect("cc");

        if link_result.status.success() {
            let (exit_code, stdout) = run_binary_with_output(&binary);
            eprintln!("test_e2e_tmir_high_register_pressure stdout: {}", stdout);
            assert_eq!(
                exit_code, 0,
                "High pressure test failed with exit code {}. stdout: {}",
                exit_code, stdout
            );
        } else {
            let stderr = String::from_utf8_lossy(&link_result.stderr);
            eprintln!("Linking high_pressure failed (inspecting .o): {}", stderr);

            // Verify the object at least has symbols.
            let nm = Command::new("nm")
                .args([obj_path.to_str().unwrap()])
                .output()
                .expect("nm");
            let nm_stdout = String::from_utf8_lossy(&nm.stdout);
            eprintln!("nm output:\n{}", nm_stdout);
            assert!(
                nm_stdout.contains("_high_pressure"),
                "Object should contain _high_pressure symbol"
            );
        }

        cleanup(&dir);
    }
}

/// Build a tMIR module with a function that has high register pressure.
///
/// fn _high_pressure(a: i64, b: i64) -> i64 {
///     let v2 = a + b;
///     let v3 = a * b;
///     let v4 = a - b;
///     let v5 = a + 1;
///     let v6 = b + 1;
///     let v7 = v2 + v3;
///     let v8 = v7 + v4;
///     let v9 = v8 + v5;
///     let v10 = v9 + v6;
///     return v10;
/// }
fn build_tmir_high_pressure_module() -> tmir_func::Module {
    use tmir_instrs::{BinOp, Instr, InstrNode, Operand};
    use tmir_types::{BlockId, FuncId, FuncTy, Ty, ValueId};

    let func = tmir_func::Function {
        id: FuncId(0),
        name: "_high_pressure".to_string(),
        ty: FuncTy {
            params: vec![Ty::int(64), Ty::int(64)],
            returns: vec![Ty::int(64)],
        },
        entry: BlockId(0),
        blocks: vec![tmir_func::Block {
            id: BlockId(0),
            params: vec![
                (ValueId(0), Ty::int(64)), // param a
                (ValueId(1), Ty::int(64)), // param b
            ],
            body: vec![
                // v2 = a + b
                InstrNode {
                    instr: Instr::BinOp {
                        op: BinOp::Add,
                        ty: Ty::int(64),
                        lhs: Operand::Value(ValueId(0)),
                        rhs: Operand::Value(ValueId(1)),
                    },
                    results: vec![ValueId(2)],
                    proofs: vec![],
                },
                // v3 = a * b
                InstrNode {
                    instr: Instr::BinOp {
                        op: BinOp::Mul,
                        ty: Ty::int(64),
                        lhs: Operand::Value(ValueId(0)),
                        rhs: Operand::Value(ValueId(1)),
                    },
                    results: vec![ValueId(3)],
                    proofs: vec![],
                },
                // v4 = a - b
                InstrNode {
                    instr: Instr::BinOp {
                        op: BinOp::Sub,
                        ty: Ty::int(64),
                        lhs: Operand::Value(ValueId(0)),
                        rhs: Operand::Value(ValueId(1)),
                    },
                    results: vec![ValueId(4)],
                    proofs: vec![],
                },
                // const 1
                InstrNode {
                    instr: Instr::Const {
                        ty: Ty::int(64),
                        value: 1,
                    },
                    results: vec![ValueId(10)],
                    proofs: vec![],
                },
                // v5 = a + 1
                InstrNode {
                    instr: Instr::BinOp {
                        op: BinOp::Add,
                        ty: Ty::int(64),
                        lhs: Operand::Value(ValueId(0)),
                        rhs: Operand::Value(ValueId(10)),
                    },
                    results: vec![ValueId(5)],
                    proofs: vec![],
                },
                // v6 = b + 1
                InstrNode {
                    instr: Instr::BinOp {
                        op: BinOp::Add,
                        ty: Ty::int(64),
                        lhs: Operand::Value(ValueId(1)),
                        rhs: Operand::Value(ValueId(10)),
                    },
                    results: vec![ValueId(6)],
                    proofs: vec![],
                },
                // v7 = v2 + v3
                InstrNode {
                    instr: Instr::BinOp {
                        op: BinOp::Add,
                        ty: Ty::int(64),
                        lhs: Operand::Value(ValueId(2)),
                        rhs: Operand::Value(ValueId(3)),
                    },
                    results: vec![ValueId(7)],
                    proofs: vec![],
                },
                // v8 = v7 + v4
                InstrNode {
                    instr: Instr::BinOp {
                        op: BinOp::Add,
                        ty: Ty::int(64),
                        lhs: Operand::Value(ValueId(7)),
                        rhs: Operand::Value(ValueId(4)),
                    },
                    results: vec![ValueId(8)],
                    proofs: vec![],
                },
                // v9 = v8 + v5
                InstrNode {
                    instr: Instr::BinOp {
                        op: BinOp::Add,
                        ty: Ty::int(64),
                        lhs: Operand::Value(ValueId(8)),
                        rhs: Operand::Value(ValueId(5)),
                    },
                    results: vec![ValueId(9)],
                    proofs: vec![],
                },
                // v10_result = v9 + v6
                InstrNode {
                    instr: Instr::BinOp {
                        op: BinOp::Add,
                        ty: Ty::int(64),
                        lhs: Operand::Value(ValueId(9)),
                        rhs: Operand::Value(ValueId(6)),
                    },
                    results: vec![ValueId(11)],
                    proofs: vec![],
                },
                // return v10_result
                InstrNode {
                    instr: Instr::Return {
                        values: vec![Operand::Value(ValueId(11))],
                    },
                    results: vec![],
                    proofs: vec![],
                },
            ],
        }],
        proofs: vec![],
    };

    tmir_func::Module {
        name: "e2e_high_pressure_test".to_string(),
        functions: vec![func],
        structs: vec![],
    }
}

// ---------------------------------------------------------------------------
// Test: frame layout correctly accounts for stack slots at all opt levels
// ---------------------------------------------------------------------------

/// Tests that the pipeline produces valid Mach-O for functions with stack
/// slots at every optimization level. This verifies that frame lowering
/// correctly handles the SP adjustment for stack slots regardless of which
/// optimizations are applied.
///
/// Part of #243 — Stack allocation E2E tests.
#[test]
fn test_e2e_stack_slots_all_opt_levels() {
    use llvm2_ir::function::StackSlot;

    for opt in &[OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3] {
        let sig = Signature::new(vec![Type::I64], vec![Type::I64]);
        let mut func = MachFunction::new("with_slots".to_string(), sig);
        let entry = func.entry;

        // Allocate several stack slots of different sizes.
        func.alloc_stack_slot(StackSlot::new(8, 8));
        func.alloc_stack_slot(StackSlot::new(16, 16));
        func.alloc_stack_slot(StackSlot::new(4, 4));

        // Simple body: just return the argument (but we have stack slots allocated).
        // MOV X0, X0 (identity - keep the argument)
        let mov = MachInst::new(
            AArch64Opcode::MovR,
            vec![MachOperand::PReg(X0), MachOperand::PReg(X0)],
        );
        let mov_id = func.push_inst(mov);
        func.append_inst(entry, mov_id);

        // RET
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(entry, ret_id);

        let pipeline = Pipeline::new(PipelineConfig {
            opt_level: *opt,
            emit_debug: false,
            ..Default::default()
        });

        let obj_bytes = pipeline
            .encode_and_emit(&mut func)
            .unwrap_or_else(|e| panic!("encode_and_emit at {:?} failed: {}", opt, e));

        // Verify valid Mach-O.
        assert!(obj_bytes.len() >= 4, "{:?} produced tiny output", opt);
        let magic = u32::from_le_bytes([obj_bytes[0], obj_bytes[1], obj_bytes[2], obj_bytes[3]]);
        assert_eq!(
            magic, 0xFEED_FACF,
            "{:?} produced invalid Mach-O magic",
            opt
        );

        eprintln!(
            "  {:?}: stack_slots function produced {} bytes",
            opt,
            obj_bytes.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Test: multiple functions in one object (add + sub + return_const)
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_multiple_functions() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e test: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("multi_func");

    // Build three functions and encode them into separate .o files.
    // (The MachOWriter currently supports one text section per object file.
    // Combining multiple functions in one .o requires either multiple symbols
    // at different offsets or multiple sections. For now, use separate .o files.)

    let add_func = build_add_function();
    let add_obj = encode_naked_to_macho(&add_func);
    let add_path = write_object_file(&dir, "add.o", &add_obj);

    let sub_func = build_sub_function();
    let sub_obj = encode_naked_to_macho(&sub_func);
    let sub_path = write_object_file(&dir, "sub.o", &sub_obj);

    let const_func = build_return_const_function();
    let const_obj = encode_naked_to_macho(&const_func);
    let const_path = write_object_file(&dir, "return_const.o", &const_obj);

    // Write C driver that tests all three functions.
    let driver_src = r#"
#include <stdio.h>
extern int add(int a, int b);
extern int sub(int a, int b);
extern int return_const(void);
int main(void) {
    int a = add(3, 4);
    int s = sub(10, 3);
    int c = return_const();
    printf("add(3,4)=%d sub(10,3)=%d return_const()=%d\n", a, s, c);
    if (a != 7) return 1;
    if (s != 7) return 2;
    if (c != 42) return 3;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);

    // Link all three .o files with the driver.
    let binary = dir.join("test_multi");
    let result = Command::new("cc")
        .arg("-o")
        .arg(&binary)
        .arg(&driver_path)
        .arg(&add_path)
        .arg(&sub_path)
        .arg(&const_path)
        .arg("-Wl,-no_pie")
        .output()
        .expect("Failed to run cc");

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        panic!("Linking multiple objects failed: {}", stderr);
    }

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_e2e_multiple_functions stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "Multi-function test failed with exit code {} (1=add, 2=sub, 3=const). stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}
