// llvm2-codegen/tests/e2e_full_pipeline.rs - Full pipeline e2e tests from tMIR
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// These tests exercise the COMPLETE compilation pipeline starting from tMIR-like
// IR, through the adapter, ISel, optimization, register allocation, frame
// lowering, encoding, and Mach-O emission. On AArch64 with cc, they link and
// run the binaries to verify correct output.
//
// Unlike e2e_run.rs which builds IR directly (bypassing ISel), these tests
// start from tMIR and exercise every pipeline phase. This catches adapter and
// ISel bugs that direct-IR tests miss.
//
// Part of #24 — Full pipeline e2e tests

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use llvm2_codegen::pipeline::{Pipeline, PipelineConfig, OptLevel};

use tmir_func::{Block as TmirBlock, Function as TmirFunction};
use tmir_instrs::{BinOp, CmpOp, Instr, InstrNode};
use tmir_types::{BlockId, FuncId, FuncTy, Ty, ValueId};

// ---------------------------------------------------------------------------
// Test infrastructure (shared with e2e_run.rs patterns)
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
    let dir = std::env::temp_dir().join(format!("llvm2_e2e_full_{}", test_name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("Failed to create test directory");
    dir
}

fn write_c_driver(dir: &Path, filename: &str, source: &str) -> PathBuf {
    let path = dir.join(filename);
    fs::write(&path, source).expect("Failed to write C driver");
    path
}

fn write_object_file(dir: &Path, filename: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(filename);
    fs::write(&path, bytes).expect("Failed to write .o file");
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

fn run_binary_with_output(binary: &Path) -> (i32, String) {
    use std::time::Duration;

    let mut child = Command::new(binary)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("Failed to spawn binary");

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
                        "Binary {} timed out after {:?} — likely infinite loop in generated code",
                        binary.display(),
                        timeout
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

// ---------------------------------------------------------------------------
// Helper: compile a tMIR function through the full pipeline
// ---------------------------------------------------------------------------

/// Translate a tMIR function through the adapter, then compile through the
/// full pipeline (ISel -> opt -> regalloc -> frame -> encode -> Mach-O).
///
/// Returns the Mach-O .o file bytes.
fn compile_tmir_function(
    tmir_func: &TmirFunction,
    opt_level: OptLevel,
) -> Result<Vec<u8>, String> {
    // Phase 0: Translate tMIR -> LIR (adapter)
    let (lir_func, _proof_ctx) =
        llvm2_lower::translate_function(tmir_func, &[])
            .map_err(|e| format!("adapter error: {}", e))?;

    // Phase 1-9: Compile LIR through full pipeline
    let config = PipelineConfig {
        opt_level,
        emit_debug: false,
        ..Default::default()
    };
    let pipeline = Pipeline::new(config);
    let obj_bytes = pipeline
        .compile_function(&lir_func)
        .map_err(|e| format!("pipeline error: {}", e))?;

    Ok(obj_bytes)
}

// ---------------------------------------------------------------------------
// tMIR function builders
// ---------------------------------------------------------------------------

/// Build tMIR for: fn fibonacci(n: i64) -> i64
///
/// Iterative Fibonacci:
///   if n <= 1: return n
///   a = 0, b = 1
///   for i in 2..=n: (a, b) = (b, a + b)
///   return b
///
/// tMIR blocks:
///   bb0 (entry): n = param, const 1, cmp n <= 1, condbr -> bb_ret_n, bb_loop_init
///   bb1 (ret_n): return n
///   bb2 (loop_init): a=0, b=1, i=2, br -> bb_loop
///   bb3 (loop): params(a, b, i), tmp = a+b, new_a = b, new_b = tmp,
///               new_i = i+1, cmp new_i <= n, condbr -> bb_loop(new_a, new_b, new_i), bb_ret_b
///   bb4 (ret_b): params(result), return result
fn build_fibonacci_tmir() -> TmirFunction {
    TmirFunction {
        id: FuncId(0),
        name: "fibonacci".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(64)],
            returns: vec![Ty::Int(64)],
        },
        entry: BlockId(0),
        blocks: vec![
            // bb0 (entry): check if n <= 1
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
                            lhs: ValueId(0), // n
                            rhs: ValueId(1), // 1
                        },
                        results: vec![ValueId(2)], // cmp_result
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: ValueId(2),
                            then_target: BlockId(1),  // ret_n
                            then_args: vec![],
                            else_target: BlockId(2),  // loop_init
                            else_args: vec![],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb1 (ret_n): return n directly
            TmirBlock {
                id: BlockId(1),
                params: vec![],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(0)], // n from bb0
                    },
                    results: vec![],
                    proofs: vec![],
                }],
            },
            // bb2 (loop_init): setup a=0, b=1, i=2, jump to loop
            TmirBlock {
                id: BlockId(2),
                params: vec![],
                body: vec![
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 0,
                        },
                        results: vec![ValueId(10)], // a_init = 0
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 1,
                        },
                        results: vec![ValueId(11)], // b_init = 1
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(64),
                            value: 2,
                        },
                        results: vec![ValueId(12)], // i_init = 2
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Br {
                            target: BlockId(3),
                            args: vec![ValueId(10), ValueId(11), ValueId(12)],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb3 (loop): loop body with block parameters
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
                            lhs: ValueId(20), // a
                            rhs: ValueId(21), // b
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
                        results: vec![ValueId(24)], // const_1
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Add,
                            ty: Ty::Int(64),
                            lhs: ValueId(22), // i
                            rhs: ValueId(24), // 1
                        },
                        results: vec![ValueId(25)], // new_i
                        proofs: vec![],
                    },
                    // cmp new_i <= n
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Sle,
                            ty: Ty::Int(64),
                            lhs: ValueId(25), // new_i
                            rhs: ValueId(0),  // n (from entry)
                        },
                        results: vec![ValueId(26)], // loop_cond
                        proofs: vec![],
                    },
                    // condbr: loop back or exit
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: ValueId(26),
                            then_target: BlockId(3), // loop back
                            then_args: vec![ValueId(21), ValueId(23), ValueId(25)], // new_a=b, new_b=tmp, new_i
                            else_target: BlockId(4), // exit
                            else_args: vec![ValueId(23)], // result = tmp (= a+b = new b)
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb4 (ret_b): return the result
            TmirBlock {
                id: BlockId(4),
                params: vec![(ValueId(30), Ty::Int(64))], // result
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(30)],
                    },
                    results: vec![],
                    proofs: vec![],
                }],
            },
        ],
        proofs: vec![],
    }
}

/// Build tMIR for: fn is_prime(n: i64) -> i64
///
/// Simple trial division:
///   if n <= 1: return 0
///   if n <= 3: return 1
///   i = 2
///   loop:
///     if i * i > n: return 1 (prime)
///     // Use div + mul + sub to check n % i == 0
///     q = n / i
///     r = n - q * i
///     if r == 0: return 0 (not prime)
///     i = i + 1
///     goto loop
///
/// Returns 1 for prime, 0 for not prime (used as exit code offset).
fn build_is_prime_tmir() -> TmirFunction {
    TmirFunction {
        id: FuncId(0),
        name: "is_prime".to_string(),
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
                        instr: Instr::Const { ty: Ty::Int(64), value: 1 },
                        results: vec![ValueId(1)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Sle,
                            ty: Ty::Int(64),
                            lhs: ValueId(0),
                            rhs: ValueId(1),
                        },
                        results: vec![ValueId(2)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: ValueId(2),
                            then_target: BlockId(1), // return 0
                            then_args: vec![],
                            else_target: BlockId(2), // check n <= 3
                            else_args: vec![],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb1: return 0 (not prime)
            TmirBlock {
                id: BlockId(1),
                params: vec![],
                body: vec![
                    InstrNode {
                        instr: Instr::Const { ty: Ty::Int(64), value: 0 },
                        results: vec![ValueId(3)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Return { values: vec![ValueId(3)] },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb2: check n <= 3
            TmirBlock {
                id: BlockId(2),
                params: vec![],
                body: vec![
                    InstrNode {
                        instr: Instr::Const { ty: Ty::Int(64), value: 3 },
                        results: vec![ValueId(4)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Sle,
                            ty: Ty::Int(64),
                            lhs: ValueId(0),
                            rhs: ValueId(4),
                        },
                        results: vec![ValueId(5)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: ValueId(5),
                            then_target: BlockId(3), // return 1 (2 and 3 are prime)
                            then_args: vec![],
                            else_target: BlockId(4), // loop init
                            else_args: vec![],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb3: return 1 (prime)
            TmirBlock {
                id: BlockId(3),
                params: vec![],
                body: vec![
                    InstrNode {
                        instr: Instr::Const { ty: Ty::Int(64), value: 1 },
                        results: vec![ValueId(6)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Return { values: vec![ValueId(6)] },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb4: loop init (i = 2)
            TmirBlock {
                id: BlockId(4),
                params: vec![],
                body: vec![
                    InstrNode {
                        instr: Instr::Const { ty: Ty::Int(64), value: 2 },
                        results: vec![ValueId(7)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Br {
                            target: BlockId(5),
                            args: vec![ValueId(7)],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb5: loop header (with param i)
            TmirBlock {
                id: BlockId(5),
                params: vec![(ValueId(10), Ty::Int(64))], // i
                body: vec![
                    // i_sq = i * i
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Mul,
                            ty: Ty::Int(64),
                            lhs: ValueId(10),
                            rhs: ValueId(10),
                        },
                        results: vec![ValueId(11)],
                        proofs: vec![],
                    },
                    // cmp i_sq > n
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Sgt,
                            ty: Ty::Int(64),
                            lhs: ValueId(11),
                            rhs: ValueId(0),
                        },
                        results: vec![ValueId(12)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: ValueId(12),
                            then_target: BlockId(3), // return 1 (prime)
                            then_args: vec![],
                            else_target: BlockId(6), // check divisibility
                            else_args: vec![],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb6: check n % i == 0 using div+mul+sub
            TmirBlock {
                id: BlockId(6),
                params: vec![],
                body: vec![
                    // q = n / i (signed div)
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::SDiv,
                            ty: Ty::Int(64),
                            lhs: ValueId(0),
                            rhs: ValueId(10),
                        },
                        results: vec![ValueId(13)],
                        proofs: vec![],
                    },
                    // qi = q * i
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Mul,
                            ty: Ty::Int(64),
                            lhs: ValueId(13),
                            rhs: ValueId(10),
                        },
                        results: vec![ValueId(14)],
                        proofs: vec![],
                    },
                    // r = n - qi
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Sub,
                            ty: Ty::Int(64),
                            lhs: ValueId(0),
                            rhs: ValueId(14),
                        },
                        results: vec![ValueId(15)],
                        proofs: vec![],
                    },
                    // cmp r == 0
                    InstrNode {
                        instr: Instr::Const { ty: Ty::Int(64), value: 0 },
                        results: vec![ValueId(16)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Eq,
                            ty: Ty::Int(64),
                            lhs: ValueId(15),
                            rhs: ValueId(16),
                        },
                        results: vec![ValueId(17)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: ValueId(17),
                            then_target: BlockId(1), // return 0 (not prime)
                            then_args: vec![],
                            else_target: BlockId(7), // increment i
                            else_args: vec![],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb7: increment i, loop back
            TmirBlock {
                id: BlockId(7),
                params: vec![],
                body: vec![
                    InstrNode {
                        instr: Instr::Const { ty: Ty::Int(64), value: 1 },
                        results: vec![ValueId(18)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Add,
                            ty: Ty::Int(64),
                            lhs: ValueId(10),
                            rhs: ValueId(18),
                        },
                        results: vec![ValueId(19)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Br {
                            target: BlockId(5),
                            args: vec![ValueId(19)],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
        ],
        proofs: vec![],
    }
}

/// Build tMIR for: fn sum_array(arr: *i64, len: i64) -> i64
///
/// Sums an array of i64 values:
///   sum = 0
///   i = 0
///   loop:
///     if i >= len: return sum
///     // ptr = arr + i * 8 (byte offset)
///     offset = i * 8
///     ptr = arr + offset  (pointer arithmetic as integer add)
///     val = load(ptr)
///     sum = sum + val
///     i = i + 1
///     goto loop
fn build_sum_array_tmir() -> TmirFunction {
    TmirFunction {
        id: FuncId(0),
        name: "sum_array".to_string(),
        ty: FuncTy {
            params: vec![
                Ty::Ptr(Box::new(Ty::Int(64))),  // arr
                Ty::Int(64),                       // len
            ],
            returns: vec![Ty::Int(64)],
        },
        entry: BlockId(0),
        blocks: vec![
            // bb0 (entry): init sum=0, i=0, jump to loop
            TmirBlock {
                id: BlockId(0),
                params: vec![
                    (ValueId(0), Ty::Ptr(Box::new(Ty::Int(64)))), // arr
                    (ValueId(1), Ty::Int(64)),                     // len
                ],
                body: vec![
                    InstrNode {
                        instr: Instr::Const { ty: Ty::Int(64), value: 0 },
                        results: vec![ValueId(2)], // sum_init = 0
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Const { ty: Ty::Int(64), value: 0 },
                        results: vec![ValueId(3)], // i_init = 0
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Br {
                            target: BlockId(1),
                            args: vec![ValueId(2), ValueId(3)],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb1 (loop header): params(sum, i)
            TmirBlock {
                id: BlockId(1),
                params: vec![
                    (ValueId(10), Ty::Int(64)), // sum
                    (ValueId(11), Ty::Int(64)), // i
                ],
                body: vec![
                    // cmp i >= len
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Sge,
                            ty: Ty::Int(64),
                            lhs: ValueId(11), // i
                            rhs: ValueId(1),  // len
                        },
                        results: vec![ValueId(12)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: ValueId(12),
                            then_target: BlockId(2), // return sum
                            then_args: vec![ValueId(10)],
                            else_target: BlockId(3), // loop body
                            else_args: vec![],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // bb2 (exit): return sum
            TmirBlock {
                id: BlockId(2),
                params: vec![(ValueId(20), Ty::Int(64))],
                body: vec![InstrNode {
                    instr: Instr::Return { values: vec![ValueId(20)] },
                    results: vec![],
                    proofs: vec![],
                }],
            },
            // bb3 (loop body): load arr[i], accumulate, increment
            TmirBlock {
                id: BlockId(3),
                params: vec![],
                body: vec![
                    // offset = i * 8
                    InstrNode {
                        instr: Instr::Const { ty: Ty::Int(64), value: 8 },
                        results: vec![ValueId(13)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Mul,
                            ty: Ty::Int(64),
                            lhs: ValueId(11), // i
                            rhs: ValueId(13), // 8
                        },
                        results: vec![ValueId(14)], // offset
                        proofs: vec![],
                    },
                    // ptr = arr + offset (pointer arithmetic as integer add on I64)
                    // We cast arr (Ptr) to I64 first, then add, then cast back
                    InstrNode {
                        instr: Instr::Cast {
                            op: tmir_instrs::CastOp::PtrToInt,
                            src_ty: Ty::Ptr(Box::new(Ty::Int(64))),
                            dst_ty: Ty::Int(64),
                            operand: ValueId(0), // arr
                        },
                        results: vec![ValueId(15)], // arr_int
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Add,
                            ty: Ty::Int(64),
                            lhs: ValueId(15), // arr_int
                            rhs: ValueId(14), // offset
                        },
                        results: vec![ValueId(16)], // elem_ptr_int
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Cast {
                            op: tmir_instrs::CastOp::IntToPtr,
                            src_ty: Ty::Int(64),
                            dst_ty: Ty::Ptr(Box::new(Ty::Int(64))),
                            operand: ValueId(16),
                        },
                        results: vec![ValueId(17)], // elem_ptr
                        proofs: vec![],
                    },
                    // val = load(elem_ptr)
                    InstrNode {
                        instr: Instr::Load {
                            ty: Ty::Int(64),
                            ptr: ValueId(17),
                        },
                        results: vec![ValueId(18)], // val
                        proofs: vec![],
                    },
                    // new_sum = sum + val
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Add,
                            ty: Ty::Int(64),
                            lhs: ValueId(10), // sum
                            rhs: ValueId(18), // val
                        },
                        results: vec![ValueId(19)], // new_sum
                        proofs: vec![],
                    },
                    // new_i = i + 1
                    InstrNode {
                        instr: Instr::Const { ty: Ty::Int(64), value: 1 },
                        results: vec![ValueId(21)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Add,
                            ty: Ty::Int(64),
                            lhs: ValueId(11),
                            rhs: ValueId(21),
                        },
                        results: vec![ValueId(22)], // new_i
                        proofs: vec![],
                    },
                    // loop back
                    InstrNode {
                        instr: Instr::Br {
                            target: BlockId(1),
                            args: vec![ValueId(19), ValueId(22)],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
        ],
        proofs: vec![],
    }
}

/// Build tMIR for: fn simple_add(a: i64, b: i64) -> i64 { a + b }
///
/// Minimal full-pipeline test: just add two arguments and return.
fn build_simple_add_tmir() -> TmirFunction {
    TmirFunction {
        id: FuncId(0),
        name: "simple_add".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(64), Ty::Int(64)],
            returns: vec![Ty::Int(64)],
        },
        entry: BlockId(0),
        blocks: vec![TmirBlock {
            id: BlockId(0),
            params: vec![
                (ValueId(0), Ty::Int(64)),
                (ValueId(1), Ty::Int(64)),
            ],
            body: vec![
                InstrNode {
                    instr: Instr::BinOp {
                        op: BinOp::Add,
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
    }
}

// ---------------------------------------------------------------------------
// Test: simple_add — minimal full-pipeline sanity check
// ---------------------------------------------------------------------------

/// Verifies the full pipeline works for the simplest possible case:
/// tMIR -> adapter -> ISel -> opt -> regalloc -> frame -> encode -> Mach-O.
#[test]
fn test_full_pipeline_simple_add_encoding() {
    let tmir_func = build_simple_add_tmir();

    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile simple_add");

    // Verify it produced a valid Mach-O file.
    assert!(obj_bytes.len() >= 4, "object file should not be empty");
    assert_eq!(
        &obj_bytes[0..4],
        &[0xCF, 0xFA, 0xED, 0xFE],
        "should be valid Mach-O magic"
    );
}

/// Full pipeline: simple_add compiled, linked, and executed.
#[test]
fn test_full_pipeline_simple_add_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("full_simple_add");

    let tmir_func = build_simple_add_tmir();
    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile simple_add");

    let obj_path = write_object_file(&dir, "simple_add.o", &obj_bytes);

    let driver_src = r#"
#include <stdio.h>
extern long simple_add(long a, long b);
int main(void) {
    long r1 = simple_add(3, 4);
    long r2 = simple_add(0, 0);
    long r3 = simple_add(-5, 10);
    printf("simple_add(3,4)=%ld simple_add(0,0)=%ld simple_add(-5,10)=%ld\n", r1, r2, r3);
    if (r1 != 7) return 1;
    if (r2 != 0) return 2;
    if (r3 != 5) return 3;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_full_simple_add");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_full_pipeline_simple_add stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "simple_add full pipeline test failed with exit code {} (1=add(3,4)!=7, 2=add(0,0)!=0, 3=add(-5,10)!=5). stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test: fibonacci — loops, conditional branches, multiple registers
// ---------------------------------------------------------------------------

/// Verifies that the fibonacci function compiles through the full pipeline
/// without errors (encoding check only, no linking).
#[test]
fn test_full_pipeline_fibonacci_encoding() {
    let tmir_func = build_fibonacci_tmir();

    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile fibonacci");

    assert!(obj_bytes.len() >= 4, "object file should not be empty");
    assert_eq!(
        &obj_bytes[0..4],
        &[0xCF, 0xFA, 0xED, 0xFE],
        "should be valid Mach-O magic"
    );
}

/// Full pipeline: fibonacci compiled, linked, and executed.
#[test]
fn test_full_pipeline_fibonacci_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("full_fibonacci");

    let tmir_func = build_fibonacci_tmir();
    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile fibonacci");

    let obj_path = write_object_file(&dir, "fibonacci.o", &obj_bytes);

    let driver_src = r#"
#include <stdio.h>
extern long fibonacci(long n);
int main(void) {
    long r0 = fibonacci(0);
    long r1 = fibonacci(1);
    long r5 = fibonacci(5);
    long r10 = fibonacci(10);
    printf("fib(0)=%ld fib(1)=%ld fib(5)=%ld fib(10)=%ld\n", r0, r1, r5, r10);
    if (r0 != 0) return 1;
    if (r1 != 1) return 2;
    if (r5 != 5) return 3;
    if (r10 != 55) return 4;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_full_fibonacci");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_full_pipeline_fibonacci stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "fibonacci full pipeline test failed with exit code {} \
         (1=fib(0)!=0, 2=fib(1)!=1, 3=fib(5)!=5, 4=fib(10)!=55). stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test: is_prime — division, modulo (via div+mul+sub), multi-exit loops
// ---------------------------------------------------------------------------

/// Verifies that is_prime compiles through the full pipeline (encoding check).
#[test]
fn test_full_pipeline_is_prime_encoding() {
    let tmir_func = build_is_prime_tmir();

    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile is_prime");

    assert!(obj_bytes.len() >= 4, "object file should not be empty");
    assert_eq!(
        &obj_bytes[0..4],
        &[0xCF, 0xFA, 0xED, 0xFE],
        "should be valid Mach-O magic"
    );
}

/// Full pipeline: is_prime compiled, linked, and executed.
#[test]
fn test_full_pipeline_is_prime_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("full_is_prime");

    let tmir_func = build_is_prime_tmir();
    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile is_prime");

    let obj_path = write_object_file(&dir, "is_prime.o", &obj_bytes);

    let driver_src = r#"
#include <stdio.h>
extern long is_prime(long n);
int main(void) {
    long r0 = is_prime(0);
    long r1 = is_prime(1);
    long r2 = is_prime(2);
    long r4 = is_prime(4);
    long r7 = is_prime(7);
    long r9 = is_prime(9);
    long r13 = is_prime(13);
    printf("prime(0)=%ld prime(1)=%ld prime(2)=%ld prime(4)=%ld prime(7)=%ld prime(9)=%ld prime(13)=%ld\n",
           r0, r1, r2, r4, r7, r9, r13);
    /* 0 and 1 are not prime */
    if (r0 != 0) return 1;
    if (r1 != 0) return 2;
    /* 2 is prime */
    if (r2 != 1) return 3;
    /* 4 is not prime */
    if (r4 != 0) return 4;
    /* 7 is prime */
    if (r7 != 1) return 5;
    /* 9 = 3*3 is not prime */
    if (r9 != 0) return 6;
    /* 13 is prime */
    if (r13 != 1) return 7;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_full_is_prime");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_full_pipeline_is_prime stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "is_prime full pipeline test failed with exit code {} \
         (1=p(0)!=0, 2=p(1)!=0, 3=p(2)!=1, 4=p(4)!=0, 5=p(7)!=1, 6=p(9)!=0, 7=p(13)!=1). stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test: sum_array — pointer arithmetic, loads, array traversal
// ---------------------------------------------------------------------------

/// Verifies that sum_array compiles through the full pipeline (encoding check).
#[test]
fn test_full_pipeline_sum_array_encoding() {
    let tmir_func = build_sum_array_tmir();

    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile sum_array");

    assert!(obj_bytes.len() >= 4, "object file should not be empty");
    assert_eq!(
        &obj_bytes[0..4],
        &[0xCF, 0xFA, 0xED, 0xFE],
        "should be valid Mach-O magic"
    );
}

/// Full pipeline: sum_array compiled, linked, and executed.
#[test]
fn test_full_pipeline_sum_array_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("full_sum_array");

    let tmir_func = build_sum_array_tmir();
    let obj_bytes = compile_tmir_function(&tmir_func, OptLevel::O0)
        .expect("full pipeline should compile sum_array");

    let obj_path = write_object_file(&dir, "sum_array.o", &obj_bytes);

    let driver_src = r#"
#include <stdio.h>
extern long sum_array(long *arr, long len);
int main(void) {
    long a1[] = {1, 2, 3, 4, 5};
    long a2[] = {10, 20, 30};
    long a3[] = {-1, 1};

    long r1 = sum_array(a1, 5);
    long r2 = sum_array(a2, 3);
    long r3 = sum_array(a3, 2);
    long r4 = sum_array(a1, 0);  /* empty sum */

    printf("sum([1..5])=%ld sum([10,20,30])=%ld sum([-1,1])=%ld sum([],0)=%ld\n",
           r1, r2, r3, r4);

    if (r1 != 15) return 1;
    if (r2 != 60) return 2;
    if (r3 != 0) return 3;
    if (r4 != 0) return 4;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_full_sum_array");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_full_pipeline_sum_array stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "sum_array full pipeline test failed with exit code {} \
         (1=sum(1..5)!=15, 2=sum(10,20,30)!=60, 3=sum(-1,1)!=0, 4=sum([],0)!=0). stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test: optimization levels — verify pipeline works at O0, O1, O2
// ---------------------------------------------------------------------------

/// Verify that the simple_add function compiles at all optimization levels.
#[test]
fn test_full_pipeline_opt_levels() {
    let tmir_func = build_simple_add_tmir();

    for opt_level in &[OptLevel::O0, OptLevel::O1, OptLevel::O2] {
        let obj_bytes = compile_tmir_function(&tmir_func, *opt_level)
            .unwrap_or_else(|e| panic!("full pipeline at {:?} failed: {}", opt_level, e));

        assert!(
            obj_bytes.len() >= 4,
            "object file at {:?} should not be empty",
            opt_level
        );
        assert_eq!(
            &obj_bytes[0..4],
            &[0xCF, 0xFA, 0xED, 0xFE],
            "should be valid Mach-O at {:?}",
            opt_level
        );
    }
}

// ---------------------------------------------------------------------------
// Test: adapter integration — verify tMIR->LIR->ISel chain preserves semantics
// ---------------------------------------------------------------------------

/// Verify the adapter correctly translates tMIR fibonacci to LIR with expected
/// block count and instruction types.
#[test]
fn test_adapter_fibonacci_structure() {
    let tmir_func = build_fibonacci_tmir();

    let (lir_func, _proof_ctx) =
        llvm2_lower::translate_function(&tmir_func, &[])
            .expect("adapter should translate fibonacci");

    assert_eq!(lir_func.name, "fibonacci");
    assert_eq!(
        lir_func.signature.params,
        vec![llvm2_lower::types::Type::I64]
    );
    assert_eq!(
        lir_func.signature.returns,
        vec![llvm2_lower::types::Type::I64]
    );
    // Should have 5 blocks (entry, ret_n, loop_init, loop, ret_b).
    assert_eq!(
        lir_func.blocks.len(),
        5,
        "fibonacci should have 5 blocks, got {}",
        lir_func.blocks.len()
    );
}

/// Verify the adapter correctly translates is_prime to LIR.
#[test]
fn test_adapter_is_prime_structure() {
    let tmir_func = build_is_prime_tmir();

    let (lir_func, _proof_ctx) =
        llvm2_lower::translate_function(&tmir_func, &[])
            .expect("adapter should translate is_prime");

    assert_eq!(lir_func.name, "is_prime");
    // Should have 8 blocks.
    assert_eq!(
        lir_func.blocks.len(),
        8,
        "is_prime should have 8 blocks, got {}",
        lir_func.blocks.len()
    );
}
