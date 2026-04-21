// llvm2-codegen/tests/e2e_correctness.rs - Correctness validation tests
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// These tests verify that LLVM2-compiled functions produce CORRECT computed
// results, not just valid binaries. Each test checks multiple inputs and
// compares against known-good values.
//
// Two approaches are tested:
//   1. Hand-coded MachIR (naked encoding, no frame lowering)
//   2. Full tMIR pipeline through the Compiler API (ISel -> opt -> regalloc
//      -> frame lowering -> encoding -> Mach-O)
//
// Both are compared against clang-compiled golden truth when possible.
//
// Part of #301 — AArch64 correctness validation: computed results

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use llvm2_codegen::compiler::{Compiler, CompilerConfig, CompilerTraceLevel};
use llvm2_codegen::macho::MachOWriter;
use llvm2_codegen::pipeline::{encode_function, OptLevel};
use llvm2_ir::function::{MachFunction, Signature, Type};
use llvm2_ir::inst::{AArch64Opcode, MachInst};
use llvm2_ir::operand::MachOperand;
use llvm2_ir::regs::{X0, X8, X9, X10, X11};

use tmir::{
    Block as TmirBlock, Function as TmirFunction, Module as TmirModule,
    BinOp, BlockId, Constant, FuncId, FuncTy, ICmpOp, Inst, InstrNode, Ty, ValueId,
};

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
    let dir = std::env::temp_dir().join(format!("llvm2_correctness_{}", test_name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create test dir");
    dir
}

fn write_c_driver(dir: &Path, filename: &str, source: &str) -> PathBuf {
    let path = dir.join(filename);
    fs::write(&path, source).expect("write C driver");
    path
}

fn write_object_file(dir: &Path, filename: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(filename);
    fs::write(&path, bytes).expect("write .o file");
    path
}

fn link_with_cc(dir: &Path, driver_c: &Path, obj: &Path, output_name: &str) -> PathBuf {
    let binary = dir.join(output_name);
    let result = Command::new("cc")
        .arg("-o")
        .arg(&binary)
        .arg(driver_c)
        .arg(obj)
        .arg("-Wl,-no_pie")
        .output()
        .expect("run cc");

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        let stdout = String::from_utf8_lossy(&result.stdout);
        panic!(
            "Linking failed!\ncc stdout: {}\ncc stderr: {}\nDriver: {}\nObject: {}",
            stdout, stderr,
            driver_c.display(), obj.display()
        );
    }

    binary
}

fn run_binary_with_output(binary: &Path) -> (i32, String) {
    use std::time::Duration;

    let mut child = Command::new(binary)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn binary");

    // Timeout after 10 seconds to catch infinite loops in generated code.
    let timeout = Duration::from_secs(10);
    let start = std::time::Instant::now();

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stdout = {
                    let mut s = String::new();
                    if let Some(mut out) = child.stdout.take() {
                        use std::io::Read;
                        let _ = out.read_to_string(&mut s);
                    }
                    s
                };
                return (status.code().unwrap_or(-1), stdout);
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!(
                        "Binary {} timed out after {:?} — likely infinite loop",
                        binary.display(), timeout
                    );
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => panic!("Error waiting for binary: {}", e),
        }
    }
}

fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

/// Encode a MachFunction into Mach-O WITHOUT frame lowering (naked).
fn encode_naked_to_macho(func: &MachFunction) -> Vec<u8> {
    let code = encode_function(func).expect("encoding should succeed");
    let mut writer = MachOWriter::new();
    writer.add_text_section(&code);
    let symbol_name = format!("_{}", func.name);
    writer.add_symbol(&symbol_name, 1, 0, true);
    writer.write()
}

// ===========================================================================
// TYPE 1: Hand-coded MachIR fibonacci (naked encoding)
// ===========================================================================

/// Build `fn fibonacci(n: i64) -> i64` using hand-coded AArch64 MachIR.
///
/// Iterative fibonacci:
///   a = 0 (X9)
///   b = 1 (X10)
///   counter = n (X8, from X0)
/// loop:
///   if counter <= 0: goto done
///   t = a + b (X11)
///   a = b
///   b = t
///   counter -= 1
///   goto loop
/// done:
///   return a (X0 = X9)
///
/// Block layout:
///   bb0 (entry): MOV X8,X0; MOVZ X9,#0; MOVZ X10,#1 [fallthrough to bb1]
///   bb1 (loop):  CMP X8,#0; B.LE bb2; ADD X11,X9,X10; MOV X9,X10; MOV X10,X11; SUB X8,X8,#1; B bb1
///   bb2 (done):  MOV X0,X9; RET
fn build_fibonacci_machir() -> MachFunction {
    let sig = Signature::new(vec![Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("fibonacci".to_string(), sig);

    let bb_entry = func.entry; // bb0
    let bb_loop = func.create_block(); // bb1
    let bb_done = func.create_block(); // bb2

    // --- bb_entry ---
    // MOV X8, X0 (counter = n)
    let mov_n = MachInst::new(
        AArch64Opcode::MovR,
        vec![MachOperand::PReg(X8), MachOperand::PReg(X0)],
    );
    let id = func.push_inst(mov_n);
    func.append_inst(bb_entry, id);

    // MOVZ X9, #0 (a = 0)
    let movz_a = MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::PReg(X9), MachOperand::Imm(0)],
    );
    let id = func.push_inst(movz_a);
    func.append_inst(bb_entry, id);

    // MOVZ X10, #1 (b = 1)
    let movz_b = MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::PReg(X10), MachOperand::Imm(1)],
    );
    let id = func.push_inst(movz_b);
    func.append_inst(bb_entry, id);

    // Fallthrough to bb_loop (no explicit branch needed)

    // --- bb_loop ---
    // CMP X8, #0
    let cmp = MachInst::new(
        AArch64Opcode::CmpRI,
        vec![MachOperand::PReg(X8), MachOperand::Imm(0)],
    );
    let id = func.push_inst(cmp);
    func.append_inst(bb_loop, id);

    // B.LE bb_done
    // bb_loop has 7 instructions: CMP, B.LE, ADD, MOV, MOV, SUB, B
    // B.LE is at index 1. bb_done starts after bb_loop.
    // Offset from B.LE to bb_done = +6 instructions (skip remaining 5 + next block).
    // Actually: offset = number of instructions between B.LE (exclusive) and bb_done start.
    // After B.LE: ADD(2), MOV(3), MOV(4), SUB(5), B(6) = 5 instrs.
    // bb_done starts at index 7 from bb_loop[0]. B.LE is at index 1.
    // PC-relative offset = (bb_done_start - B.LE_position) in instruction units.
    // B.LE at position P, bb_done at position P+6 instructions -> offset = +6.
    // Wait: the offset in B.cond imm19 is from the B.cond instruction itself.
    // B.LE is instruction 1 in bb_loop. bb_done starts at instruction 7.
    // Offset = 7 - 1 = 6.
    // Actually the B.LE is the 4th instruction overall (3 in entry + 1). But for
    // the branch offset we just count from B.LE to target. After B.LE we have
    // ADD, MOV, MOV, SUB, B = 5 instructions, then bb_done. So offset = +6.
    let ble_done = MachInst::new(
        AArch64Opcode::BCond,
        vec![
            MachOperand::Imm(0xD), // LE = 0b1101 = 13
            MachOperand::Imm(6),   // +6 instructions to bb_done
        ],
    );
    let id = func.push_inst(ble_done);
    func.append_inst(bb_loop, id);

    // ADD X11, X9, X10 (t = a + b)
    let add = MachInst::new(
        AArch64Opcode::AddRR,
        vec![
            MachOperand::PReg(X11),
            MachOperand::PReg(X9),
            MachOperand::PReg(X10),
        ],
    );
    let id = func.push_inst(add);
    func.append_inst(bb_loop, id);

    // MOV X9, X10 (a = b)
    let mov_ab = MachInst::new(
        AArch64Opcode::MovR,
        vec![MachOperand::PReg(X9), MachOperand::PReg(X10)],
    );
    let id = func.push_inst(mov_ab);
    func.append_inst(bb_loop, id);

    // MOV X10, X11 (b = t)
    let mov_bt = MachInst::new(
        AArch64Opcode::MovR,
        vec![MachOperand::PReg(X10), MachOperand::PReg(X11)],
    );
    let id = func.push_inst(mov_bt);
    func.append_inst(bb_loop, id);

    // SUB X8, X8, #1 (counter -= 1)
    let sub = MachInst::new(
        AArch64Opcode::SubRI,
        vec![
            MachOperand::PReg(X8),
            MachOperand::PReg(X8),
            MachOperand::Imm(1),
        ],
    );
    let id = func.push_inst(sub);
    func.append_inst(bb_loop, id);

    // B bb_loop (loop back)
    // This B is the 7th instruction of bb_loop (index 6).
    // bb_loop[0] = CMP. Offset from B to CMP = -(7-1) = -6.
    let b_loop = MachInst::new(
        AArch64Opcode::B,
        vec![MachOperand::Imm(-6i64)],
    );
    let id = func.push_inst(b_loop);
    func.append_inst(bb_loop, id);

    // --- bb_done ---
    // MOV X0, X9 (return a)
    let mov_result = MachInst::new(
        AArch64Opcode::MovR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X9)],
    );
    let id = func.push_inst(mov_result);
    func.append_inst(bb_done, id);

    // RET
    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let id = func.push_inst(ret);
    func.append_inst(bb_done, id);

    func
}

// ---------------------------------------------------------------------------
// Test: Hand-coded MachIR fibonacci — naked encoding, multiple test values
// ---------------------------------------------------------------------------

/// Verifies that hand-coded fibonacci MachIR produces correct results for
/// a comprehensive set of inputs including edge cases and larger values.
///
/// This exercises: loops (back-edge), conditional branches, multiple live
/// registers (X8/X9/X10/X11), and integer addition in a tight loop.
#[test]
fn test_correctness_fibonacci_machir() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e test: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("fibonacci_machir");

    let func = build_fibonacci_machir();
    let obj_bytes = encode_naked_to_macho(&func);
    let obj_path = write_object_file(&dir, "fibonacci.o", &obj_bytes);

    // C driver that checks all fibonacci values and prints diagnostics.
    let driver_src = r#"
#include <stdio.h>
extern long fibonacci(long n);
int main(void) {
    /* Test fibonacci across a range of inputs */
    struct { long n; long expected; } tests[] = {
        {0, 0},
        {1, 1},
        {2, 1},
        {3, 2},
        {4, 3},
        {5, 5},
        {6, 8},
        {7, 13},
        {10, 55},
        {15, 610},
        {20, 6765},
    };
    int n_tests = sizeof(tests) / sizeof(tests[0]);

    for (int i = 0; i < n_tests; i++) {
        long result = fibonacci(tests[i].n);
        printf("fibonacci(%ld) = %ld (expected %ld) %s\n",
               tests[i].n, result, tests[i].expected,
               result == tests[i].expected ? "OK" : "FAIL");
        if (result != tests[i].expected) return i + 1;
    }

    printf("All %d fibonacci tests passed.\n", n_tests);
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_fibonacci_machir");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_correctness_fibonacci_machir stdout:\n{}", stdout);
    assert_eq!(
        exit_code, 0,
        "Fibonacci MachIR correctness test failed at test case {} (1-indexed). stdout:\n{}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ===========================================================================
// TYPE 2: tMIR fibonacci through the Compiler API (full pipeline)
// ===========================================================================

/// Build a tMIR module containing `fn _fibonacci_tmir(n: i64) -> i64`.
///
/// Iterative fibonacci using SSA block parameters:
///   bb0 (entry): param n, check n <= 0 -> return 0; else check n == 1 -> return 1; else loop
///   bb1 (loop_init): const 0, const 1, br loop_check(n, 0, 1)
///   bb2 (loop_check): params (counter, a, b), const 1, icmp sle counter 1, condbr done(b) loop_body
///   bb3 (loop_body): params (counter, a, b), add a+b -> t, const 1, sub counter-1 -> new_counter, br loop_check(new_counter, b, t)
///   bb4 (done): param result, return result
///
/// Simplified for robustness: just use the simpler pattern from e2e_full_pipeline.
fn build_fibonacci_tmir_module() -> TmirModule {
    let mut module = TmirModule::new("correctness_test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });

    let mut func = TmirFunction::new(
        FuncId::new(0),
        "_fibonacci_tmir",
        ft_id,
        BlockId::new(0),
    );

    func.blocks = vec![
        // bb0 (entry): check if n <= 1 -> return n directly
        TmirBlock {
            id: BlockId::new(0),
            params: vec![(ValueId::new(0), Ty::I64)], // n
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(1)),
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sle,
                    ty: Ty::I64,
                    lhs: ValueId::new(0), // n
                    rhs: ValueId::new(1), // 1
                })
                .with_result(ValueId::new(2)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(2),
                    then_target: BlockId::new(1),  // ret_n
                    then_args: vec![],
                    else_target: BlockId::new(2),  // loop_init
                    else_args: vec![],
                }),
            ],
        },
        // bb1 (ret_n): return n directly (handles n=0 and n=1)
        TmirBlock {
            id: BlockId::new(1),
            params: vec![],
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(0)], // n from entry
            })],
        },
        // bb2 (loop_init): a=0, b=1, i=2, jump to loop
        TmirBlock {
            id: BlockId::new(2),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(10)),
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(11)),
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(2),
                })
                .with_result(ValueId::new(12)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(3),
                    args: vec![
                        ValueId::new(10), // a = 0
                        ValueId::new(11), // b = 1
                        ValueId::new(12), // i = 2
                    ],
                }),
            ],
        },
        // bb3 (loop): block params (a, b, i), compute next step
        TmirBlock {
            id: BlockId::new(3),
            params: vec![
                (ValueId::new(20), Ty::I64), // a
                (ValueId::new(21), Ty::I64), // b
                (ValueId::new(22), Ty::I64), // i
            ],
            body: vec![
                // tmp = a + b
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(20), // a
                    rhs: ValueId::new(21), // b
                })
                .with_result(ValueId::new(23)),
                // new_i = i + 1
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(24)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(22), // i
                    rhs: ValueId::new(24), // 1
                })
                .with_result(ValueId::new(25)),
                // cmp new_i <= n
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sle,
                    ty: Ty::I64,
                    lhs: ValueId::new(25), // new_i
                    rhs: ValueId::new(0),  // n (from entry)
                })
                .with_result(ValueId::new(26)),
                // condbr: if new_i <= n, loop; else exit with result
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(26),
                    then_target: BlockId::new(3), // loop back
                    then_args: vec![
                        ValueId::new(21), // new_a = old b
                        ValueId::new(23), // new_b = tmp (a+b)
                        ValueId::new(25), // new_i
                    ],
                    else_target: BlockId::new(4), // done
                    else_args: vec![ValueId::new(23)], // result = tmp
                }),
            ],
        },
        // bb4 (done): return the computed fibonacci value
        TmirBlock {
            id: BlockId::new(4),
            params: vec![(ValueId::new(30), Ty::I64)], // result
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(30)],
            })],
        },
    ];

    module.add_function(func);
    module
}

// ---------------------------------------------------------------------------
// Test: tMIR fibonacci through Compiler API — full pipeline correctness
// ---------------------------------------------------------------------------

/// Compiles fibonacci through the FULL pipeline via the Compiler API:
///   tMIR -> adapter -> ISel -> optimization -> regalloc -> frame lowering
///   -> encoding -> Mach-O
///
/// Tests a comprehensive set of fibonacci values including fib(20)=6765.
/// This exercises the Compiler public API (not just the internal Pipeline).
#[test]
fn test_correctness_fibonacci_tmir_compiler() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e test: not AArch64 or cc not available");
        return;
    }

    let module = build_fibonacci_tmir_module();
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        trace_level: CompilerTraceLevel::Full,
        ..CompilerConfig::default()
    });

    let result = compiler.compile(&module).expect("compilation should succeed");

    // Verify valid Mach-O.
    assert!(!result.object_code.is_empty(), "non-empty object code");
    let magic = u32::from_le_bytes([
        result.object_code[0],
        result.object_code[1],
        result.object_code[2],
        result.object_code[3],
    ]);
    assert_eq!(magic, 0xFEED_FACF, "valid Mach-O magic");
    assert_eq!(result.metrics.function_count, 1);

    eprintln!(
        "Compiler metrics: {} instrs, {} bytes, {} opt passes",
        result.metrics.instruction_count,
        result.metrics.code_size_bytes,
        result.metrics.optimization_passes_run,
    );

    let dir = make_test_dir("fibonacci_tmir_compiler");
    let obj_path = write_object_file(&dir, "fibonacci_tmir.o", &result.object_code);

    // Disassemble for inspection.
    let otool = Command::new("otool")
        .args(["-tv", obj_path.to_str().unwrap()])
        .output()
        .expect("otool");
    eprintln!(
        "fibonacci_tmir disassembly:\n{}",
        String::from_utf8_lossy(&otool.stdout)
    );

    // C driver with comprehensive fibonacci checks.
    let driver_src = r#"
#include <stdio.h>
extern long _fibonacci_tmir(long n);
int main(void) {
    struct { long n; long expected; } tests[] = {
        {0, 0},
        {1, 1},
        {2, 1},
        {3, 2},
        {5, 5},
        {10, 55},
        {15, 610},
        {20, 6765},
    };
    int n_tests = sizeof(tests) / sizeof(tests[0]);

    for (int i = 0; i < n_tests; i++) {
        long result = _fibonacci_tmir(tests[i].n);
        printf("fibonacci_tmir(%ld) = %ld (expected %ld) %s\n",
               tests[i].n, result, tests[i].expected,
               result == tests[i].expected ? "OK" : "FAIL");
        if (result != tests[i].expected) return i + 1;
    }

    printf("All %d fibonacci_tmir tests passed.\n", n_tests);
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_fibonacci_tmir");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_correctness_fibonacci_tmir_compiler stdout:\n{}", stdout);
    assert_eq!(
        exit_code, 0,
        "Fibonacci tMIR Compiler correctness test failed at test {} (1-indexed). stdout:\n{}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ===========================================================================
// Golden truth comparison: LLVM2 vs clang
// ===========================================================================

/// Compiles the same fibonacci function with clang and LLVM2, then runs both
/// and verifies they produce identical results for all test inputs.
///
/// This is the strongest correctness test: if our compiler disagrees with
/// clang on ANY input, something is wrong.
#[test]
fn test_correctness_fibonacci_golden_truth() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e test: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("fibonacci_golden");

    // Step 1: Build the clang golden-truth fibonacci.
    let clang_src = r#"
long fibonacci_clang(long n) {
    if (n <= 1) return n;
    long a = 0, b = 1;
    for (long i = 2; i <= n; i++) {
        long t = a + b;
        a = b;
        b = t;
    }
    return b;
}
"#;
    let clang_c = write_c_driver(&dir, "fib_clang.c", clang_src);
    let clang_obj = dir.join("fib_clang.o");
    let cc_result = Command::new("cc")
        .args(["-c", "-o"])
        .arg(&clang_obj)
        .arg(&clang_c)
        .output()
        .expect("run cc -c");
    assert!(
        cc_result.status.success(),
        "clang compilation failed: {}",
        String::from_utf8_lossy(&cc_result.stderr)
    );

    // Step 2: Build the LLVM2-compiled fibonacci (hand-coded MachIR).
    let func = build_fibonacci_machir();
    let llvm2_obj_bytes = encode_naked_to_macho(&func);
    let llvm2_obj = write_object_file(&dir, "fib_llvm2.o", &llvm2_obj_bytes);

    // Step 3: Write a comparator driver that calls both and checks agreement.
    let compare_driver = r#"
#include <stdio.h>
extern long fibonacci(long n);       /* LLVM2 */
extern long fibonacci_clang(long n); /* clang golden truth */
int main(void) {
    long test_inputs[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25};
    int n_tests = sizeof(test_inputs) / sizeof(test_inputs[0]);

    for (int i = 0; i < n_tests; i++) {
        long n = test_inputs[i];
        long llvm2_result = fibonacci(n);
        long clang_result = fibonacci_clang(n);
        printf("fib(%ld): llvm2=%ld clang=%ld %s\n",
               n, llvm2_result, clang_result,
               llvm2_result == clang_result ? "MATCH" : "MISMATCH");
        if (llvm2_result != clang_result) return i + 1;
    }

    printf("All %d golden truth comparisons passed.\n", n_tests);
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "compare.c", compare_driver);

    // Link: compare.c + fib_llvm2.o + fib_clang.o
    let binary = dir.join("test_golden");
    let link_result = Command::new("cc")
        .arg("-o")
        .arg(&binary)
        .arg(&driver_path)
        .arg(&llvm2_obj)
        .arg(&clang_obj)
        .arg("-Wl,-no_pie")
        .output()
        .expect("run cc for golden truth link");

    if !link_result.status.success() {
        let stderr = String::from_utf8_lossy(&link_result.stderr);
        panic!("Golden truth linking failed: {}", stderr);
    }

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_correctness_fibonacci_golden_truth stdout:\n{}", stdout);
    assert_eq!(
        exit_code, 0,
        "Golden truth comparison failed at test {} (1-indexed). LLVM2 disagrees with clang. stdout:\n{}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ===========================================================================
// Bonus: sum_1_to_n — simpler loop function as baseline
// ===========================================================================

/// Build `fn sum_1_to_n(n: i64) -> i64` using hand-coded MachIR.
///
/// Computes 1 + 2 + ... + n = n*(n+1)/2.
/// Uses a counting loop:
///   sum = 0 (X9)
///   i = 1 (X10)
/// loop:
///   if i > n: goto done
///   sum += i
///   i += 1
///   goto loop
/// done:
///   return sum
fn build_sum_1_to_n_machir() -> MachFunction {
    let sig = Signature::new(vec![Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("sum_1_to_n".to_string(), sig);

    let bb_entry = func.entry;
    let bb_loop = func.create_block();
    let bb_done = func.create_block();

    // --- bb_entry ---
    // MOV X8, X0 (n)
    let inst = MachInst::new(
        AArch64Opcode::MovR,
        vec![MachOperand::PReg(X8), MachOperand::PReg(X0)],
    );
    let id = func.push_inst(inst);
    func.append_inst(bb_entry, id);

    // MOVZ X9, #0 (sum = 0)
    let inst = MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::PReg(X9), MachOperand::Imm(0)],
    );
    let id = func.push_inst(inst);
    func.append_inst(bb_entry, id);

    // MOVZ X10, #1 (i = 1)
    let inst = MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::PReg(X10), MachOperand::Imm(1)],
    );
    let id = func.push_inst(inst);
    func.append_inst(bb_entry, id);

    // --- bb_loop ---
    // CMP X10, X8 (compare i with n)
    let inst = MachInst::new(
        AArch64Opcode::CmpRR,
        vec![MachOperand::PReg(X10), MachOperand::PReg(X8)],
    );
    let id = func.push_inst(inst);
    func.append_inst(bb_loop, id);

    // B.GT bb_done (if i > n, done)
    // bb_loop: CMP(0), B.GT(1), ADD(2), ADD(3), B(4) = 5 instructions
    // offset from B.GT to bb_done = +4
    let inst = MachInst::new(
        AArch64Opcode::BCond,
        vec![
            MachOperand::Imm(0xC), // GT = 0b1100 = 12
            MachOperand::Imm(4),   // +4 instructions
        ],
    );
    let id = func.push_inst(inst);
    func.append_inst(bb_loop, id);

    // ADD X9, X9, X10 (sum += i)
    let inst = MachInst::new(
        AArch64Opcode::AddRR,
        vec![
            MachOperand::PReg(X9),
            MachOperand::PReg(X9),
            MachOperand::PReg(X10),
        ],
    );
    let id = func.push_inst(inst);
    func.append_inst(bb_loop, id);

    // ADD X10, X10, #1 (i += 1)
    let inst = MachInst::new(
        AArch64Opcode::AddRI,
        vec![
            MachOperand::PReg(X10),
            MachOperand::PReg(X10),
            MachOperand::Imm(1),
        ],
    );
    let id = func.push_inst(inst);
    func.append_inst(bb_loop, id);

    // B bb_loop
    // 5th instruction (index 4), back to index 0: offset = -4
    let inst = MachInst::new(
        AArch64Opcode::B,
        vec![MachOperand::Imm(-4i64)],
    );
    let id = func.push_inst(inst);
    func.append_inst(bb_loop, id);

    // --- bb_done ---
    // MOV X0, X9 (return sum)
    let inst = MachInst::new(
        AArch64Opcode::MovR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X9)],
    );
    let id = func.push_inst(inst);
    func.append_inst(bb_done, id);

    // RET
    let inst = MachInst::new(AArch64Opcode::Ret, vec![]);
    let id = func.push_inst(inst);
    func.append_inst(bb_done, id);

    func
}

/// Verifies that sum_1_to_n produces correct results using the closed-form
/// formula n*(n+1)/2 as ground truth.
#[test]
fn test_correctness_sum_1_to_n() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping e2e test: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("sum_1_to_n");

    let func = build_sum_1_to_n_machir();
    let obj_bytes = encode_naked_to_macho(&func);
    let obj_path = write_object_file(&dir, "sum_1_to_n.o", &obj_bytes);

    let driver_src = r#"
#include <stdio.h>
extern long sum_1_to_n(long n);
int main(void) {
    struct { long n; long expected; } tests[] = {
        {0, 0},
        {1, 1},
        {5, 15},
        {10, 55},
        {100, 5050},
        {1000, 500500},
    };
    int n_tests = sizeof(tests) / sizeof(tests[0]);

    for (int i = 0; i < n_tests; i++) {
        long result = sum_1_to_n(tests[i].n);
        printf("sum_1_to_n(%ld) = %ld (expected %ld) %s\n",
               tests[i].n, result, tests[i].expected,
               result == tests[i].expected ? "OK" : "FAIL");
        if (result != tests[i].expected) return i + 1;
    }

    printf("All %d sum_1_to_n tests passed.\n", n_tests);
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_sum_1_to_n");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_correctness_sum_1_to_n stdout:\n{}", stdout);
    assert_eq!(
        exit_code, 0,
        "sum_1_to_n correctness test failed at test {} (1-indexed). stdout:\n{}",
        exit_code, stdout
    );

    cleanup(&dir);
}
