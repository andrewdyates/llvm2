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
// Frame lowering currently has encoding bugs with STP/LDP pre/post-index
// forms and MOV X29, SP (see TODO comments below). For truly runnable code,
// we bypass frame lowering and emit minimal naked functions.
// ---------------------------------------------------------------------------

/// Encode an IR function into a Mach-O .o file WITHOUT frame lowering.
///
/// This produces a "naked" function: just the raw instructions, no
/// prologue/epilogue. For leaf functions that don't clobber callee-saved
/// registers and don't use the stack, this is sufficient and correct.
///
/// TODO: Once the frame lowering encoding bugs are fixed (see
/// test_full_pipeline_add_with_frame_lowering), switch to using
/// Pipeline::encode_and_emit for all tests.
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
/// Note: MUL encoding is not yet implemented in encode_ir_inst (falls through
/// to NOP). This test documents that MUL support is needed for factorial.
///
/// TODO: Implement MUL encoding in pipeline.rs encode_ir_inst to make
/// this test pass.
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
    // TODO: MUL is not encoded yet (emits NOP). See encode_ir_inst in pipeline.rs.
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
        vec![MachOperand::Imm((-4i64) as i64)],
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
// Test: factorial — iterative loop (blocked on MUL encoding)
// ---------------------------------------------------------------------------

// TODO: MUL (MADD) encoding is not yet implemented in encode_ir_inst.
// The MulRR opcode falls through to the default NOP case. Once MUL encoding
// is added, this test should pass.
//
// MUL Xd, Xn, Xm is encoded as MADD Xd, Xn, Xm, XZR:
//   sf=1, 0011011000, Rm, 0, Ra=11111(XZR), Rn, Rd
//   = (sf<<31) | (0b00011011000<<21) | (Rm<<16) | (0<<15) | (31<<10) | (Rn<<5) | Rd
//
// See ARM ARM: "Data-processing (3 source)" encoding class.
#[test]
fn test_e2e_factorial_blocked_on_mul() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e test: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("factorial");

    let func = build_factorial_function();

    // Verify the function encodes without errors (MUL becomes NOP).
    let code = encode_function(&func).expect("encoding should succeed (MUL -> NOP)");
    assert!(
        code.len() > 0,
        "Factorial function should produce non-empty code"
    );

    // Verify MUL is currently encoded as NOP (documenting the gap).
    // The MUL instruction is at index 2 in the loop block (bb1).
    // In the linear layout: bb0 has 2 instrs (MOV, MOVZ), bb1 has 5 instrs
    // (CMP, B.LE, MUL, SUB, B). MUL is at instruction index 4 overall.
    // Each instruction is 4 bytes, so MUL is at byte offset 16.
    if code.len() >= 20 {
        let mul_word = u32::from_le_bytes([code[16], code[17], code[18], code[19]]);
        // NOP = 0xD503201F
        assert_eq!(
            mul_word, 0xD503201F,
            "MUL should currently be encoded as NOP (0xD503201F), got 0x{:08X}. \
             If this assertion fails, MUL encoding has been implemented — update this test!",
            mul_word
        );
    }

    // We intentionally do NOT link and run the factorial binary, because
    // the MUL instruction is encoded as NOP, which would produce wrong results.
    // Once MUL encoding is implemented, uncomment the link-and-run test below.

    /*
    // TODO: Uncomment when MUL encoding is implemented.
    let obj_bytes = encode_naked_to_macho(&func);
    let obj_path = write_object_file(&dir, "factorial.o", &obj_bytes);
    let driver_src = r#"
#include <stdio.h>
extern int factorial(int n);
int main(void) {
    int result = factorial(5);
    printf("factorial(5) = %d\n", result);
    return (result == 120) ? 0 : 1;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_factorial");
    let (exit_code, stdout) = run_binary_with_output(&binary);
    assert_eq!(exit_code, 0,
        "factorial(5) should return 120. stdout: {}", stdout);
    */

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test: full pipeline with frame lowering (documents encoding bugs)
// ---------------------------------------------------------------------------

/// This test documents the current state of the full pipeline (with frame
/// lowering). The frame lowering inserts prologue/epilogue instructions, but
/// the encoding has bugs that prevent the output from being runnable:
///
/// BUG 1: STP/LDP pre-index and post-index forms.
///   Frame lowering emits STP X29, X30, [SP, #-16]! (pre-indexed) for the
///   prologue and LDP X29, X30, [SP], #16 (post-indexed) for the epilogue.
///   However, encode_ir_inst encodes StpRI/LdpRI as the "signed offset" form
///   (bits 25:23 = 010), not the pre-indexed form (011) or post-indexed form
///   (001). This means SP is never actually adjusted, and the epilogue reads
///   from the wrong stack location.
///
/// BUG 2: MOV X29, SP encoding.
///   Frame lowering emits MOV X29, SP. The encoder translates MovR as
///   ORR Rd, XZR, Rm. But register 31 in logical instructions (ORR) is XZR,
///   not SP. So MOV X29, SP encodes as ORR X29, XZR, XZR = X29 = 0.
///   The correct encoding is ADD X29, SP, #0.
///
/// These bugs make the frame lowering output non-functional. The naked
/// function tests above bypass frame lowering to produce correct code.
///
/// TODO: Fix the encoding of STP/LDP pre/post-index and MOV-from-SP,
/// then convert this test to an actual link-and-run test.
#[test]
fn test_full_pipeline_frame_lowering_encoding_gaps() {
    // Build the add function and run it through encode_and_emit (includes
    // frame lowering).
    let mut func = build_add_function();

    let pipeline = Pipeline::new(PipelineConfig {
        opt_level: OptLevel::O0,
        emit_debug: false,
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

    // Document the prologue/epilogue instruction sequence.
    // After frame lowering, the function should be:
    //   STP X29, X30, [SP, #-16]!   <-- prologue
    //   MOV X29, SP                  <-- prologue
    //   ADD X0, X0, X1              <-- body
    //   LDP X29, X30, [SP], #16     <-- epilogue
    //   RET                          <-- epilogue
    //
    // But due to encoding bugs, the actual machine code is:
    //   STP X29, X30, [SP, #-16]  (signed-offset, SP not modified)
    //   ORR X29, XZR, XZR        (X29 = 0, not SP)
    //   ADD X0, X0, X1           (correct)
    //   LDP X29, X30, [SP, #16]  (signed-offset, reads wrong location)
    //   RET                       (returns with corrupted LR if SP was modified)
    //
    // Verify the file passes otool -h (structural validity, not semantic correctness).
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

        // Also verify otool -tv can disassemble it.
        let otool_tv = Command::new("otool")
            .args(["-tv", obj_path.to_str().unwrap()])
            .output();

        if let Ok(output) = otool_tv {
            let stdout = String::from_utf8_lossy(&output.stdout);
            eprintln!("otool -tv disassembly:\n{}", stdout);
            // The disassembly should show recognizable instructions.
            // It won't crash otool even though the semantics are wrong.
        }

        cleanup(&dir);
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
