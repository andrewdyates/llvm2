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

use llvm2_codegen::pipeline::{OptLevel, Pipeline, PipelineConfig};

use tmir::{Block as TmirBlock, BlockId, FuncId, FuncTyId, Function as TmirFunction, Module, Ty, ValueId};
use tmir::constant::Constant;
use tmir::inst::{BinOp, CastOp, ICmpOp, Inst};
use tmir::node::InstrNode;
use tmir::ty::FuncTy;

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

fn wrap_function_in_module(mut func: TmirFunction, func_ty: FuncTy) -> Module {
    let mut module = Module::new("test");
    let ft_id = module.add_func_type(func_ty);
    func.ty = ft_id;
    module.add_function(func);
    module
}

fn test_function(module: &Module) -> &TmirFunction {
    module
        .functions
        .first()
        .expect("test module should contain one function")
}

// ---------------------------------------------------------------------------
// Helper: compile a tMIR function through the full pipeline
// ---------------------------------------------------------------------------

/// Translate a tMIR function through the adapter, then compile through the
/// full pipeline (ISel -> opt -> regalloc -> frame -> encode -> Mach-O).
///
/// Returns the Mach-O .o file bytes.
fn compile_tmir_function(module: &Module, opt_level: OptLevel) -> Result<Vec<u8>, String> {
    let tmir_func = test_function(module);

    // Phase 0: Translate tMIR -> LIR (adapter)
    let (lir_func, _proof_ctx) =
        llvm2_lower::translate_function(tmir_func, module).map_err(|e| format!("adapter error: {}", e))?;

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
fn build_fibonacci_tmir() -> Module {
    let func_ty = FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    };

    let func = TmirFunction {
        id: FuncId::new(0),
        name: "fibonacci".to_string(),
        ty: FuncTyId::new(0),
        entry: BlockId::new(0),
        blocks: vec![
            TmirBlock {
                id: BlockId::new(0),
                params: vec![(ValueId::new(0), Ty::I64)],
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(1),
                    })
                    .with_result(ValueId::new(1)),
                    InstrNode::new(Inst::ICmp {
                        op: ICmpOp::Sle,
                        ty: Ty::I64,
                        lhs: ValueId::new(0),
                        rhs: ValueId::new(1),
                    })
                    .with_result(ValueId::new(2)),
                    InstrNode::new(Inst::CondBr {
                        cond: ValueId::new(2),
                        then_target: BlockId::new(1),
                        then_args: vec![],
                        else_target: BlockId::new(2),
                        else_args: vec![],
                    }),
                ],
            },
            TmirBlock {
                id: BlockId::new(1),
                params: vec![],
                body: vec![InstrNode::new(Inst::Return {
                    values: vec![ValueId::new(0)],
                })],
            },
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
                        args: vec![ValueId::new(10), ValueId::new(11), ValueId::new(12)],
                    }),
                ],
            },
            TmirBlock {
                id: BlockId::new(3),
                params: vec![
                    (ValueId::new(20), Ty::I64),
                    (ValueId::new(21), Ty::I64),
                    (ValueId::new(22), Ty::I64),
                ],
                body: vec![
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: ValueId::new(20),
                        rhs: ValueId::new(21),
                    })
                    .with_result(ValueId::new(23)),
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(1),
                    })
                    .with_result(ValueId::new(24)),
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: ValueId::new(22),
                        rhs: ValueId::new(24),
                    })
                    .with_result(ValueId::new(25)),
                    InstrNode::new(Inst::ICmp {
                        op: ICmpOp::Sle,
                        ty: Ty::I64,
                        lhs: ValueId::new(25),
                        rhs: ValueId::new(0),
                    })
                    .with_result(ValueId::new(26)),
                    InstrNode::new(Inst::CondBr {
                        cond: ValueId::new(26),
                        then_target: BlockId::new(3),
                        then_args: vec![ValueId::new(21), ValueId::new(23), ValueId::new(25)],
                        else_target: BlockId::new(4),
                        else_args: vec![ValueId::new(23)],
                    }),
                ],
            },
            TmirBlock {
                id: BlockId::new(4),
                params: vec![(ValueId::new(30), Ty::I64)],
                body: vec![InstrNode::new(Inst::Return {
                    values: vec![ValueId::new(30)],
                })],
            },
        ],
        proofs: vec![],
    };

    wrap_function_in_module(func, func_ty)
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
fn build_is_prime_tmir() -> Module {
    let func_ty = FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    };

    let func = TmirFunction {
        id: FuncId::new(0),
        name: "is_prime".to_string(),
        ty: FuncTyId::new(0),
        entry: BlockId::new(0),
        blocks: vec![
            TmirBlock {
                id: BlockId::new(0),
                params: vec![(ValueId::new(0), Ty::I64)],
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(1),
                    })
                    .with_result(ValueId::new(1)),
                    InstrNode::new(Inst::ICmp {
                        op: ICmpOp::Sle,
                        ty: Ty::I64,
                        lhs: ValueId::new(0),
                        rhs: ValueId::new(1),
                    })
                    .with_result(ValueId::new(2)),
                    InstrNode::new(Inst::CondBr {
                        cond: ValueId::new(2),
                        then_target: BlockId::new(1),
                        then_args: vec![],
                        else_target: BlockId::new(2),
                        else_args: vec![],
                    }),
                ],
            },
            TmirBlock {
                id: BlockId::new(1),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(0),
                    })
                    .with_result(ValueId::new(3)),
                    InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(3)],
                    }),
                ],
            },
            TmirBlock {
                id: BlockId::new(2),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(3),
                    })
                    .with_result(ValueId::new(4)),
                    InstrNode::new(Inst::ICmp {
                        op: ICmpOp::Sle,
                        ty: Ty::I64,
                        lhs: ValueId::new(0),
                        rhs: ValueId::new(4),
                    })
                    .with_result(ValueId::new(5)),
                    InstrNode::new(Inst::CondBr {
                        cond: ValueId::new(5),
                        then_target: BlockId::new(3),
                        then_args: vec![],
                        else_target: BlockId::new(4),
                        else_args: vec![],
                    }),
                ],
            },
            TmirBlock {
                id: BlockId::new(3),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(1),
                    })
                    .with_result(ValueId::new(6)),
                    InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(6)],
                    }),
                ],
            },
            TmirBlock {
                id: BlockId::new(4),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(2),
                    })
                    .with_result(ValueId::new(7)),
                    InstrNode::new(Inst::Br {
                        target: BlockId::new(5),
                        args: vec![ValueId::new(7)],
                    }),
                ],
            },
            TmirBlock {
                id: BlockId::new(5),
                params: vec![(ValueId::new(10), Ty::I64)],
                body: vec![
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Mul,
                        ty: Ty::I64,
                        lhs: ValueId::new(10),
                        rhs: ValueId::new(10),
                    })
                    .with_result(ValueId::new(11)),
                    InstrNode::new(Inst::ICmp {
                        op: ICmpOp::Sgt,
                        ty: Ty::I64,
                        lhs: ValueId::new(11),
                        rhs: ValueId::new(0),
                    })
                    .with_result(ValueId::new(12)),
                    InstrNode::new(Inst::CondBr {
                        cond: ValueId::new(12),
                        then_target: BlockId::new(3),
                        then_args: vec![],
                        else_target: BlockId::new(6),
                        else_args: vec![],
                    }),
                ],
            },
            TmirBlock {
                id: BlockId::new(6),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::SDiv,
                        ty: Ty::I64,
                        lhs: ValueId::new(0),
                        rhs: ValueId::new(10),
                    })
                    .with_result(ValueId::new(13)),
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Mul,
                        ty: Ty::I64,
                        lhs: ValueId::new(13),
                        rhs: ValueId::new(10),
                    })
                    .with_result(ValueId::new(14)),
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Sub,
                        ty: Ty::I64,
                        lhs: ValueId::new(0),
                        rhs: ValueId::new(14),
                    })
                    .with_result(ValueId::new(15)),
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(0),
                    })
                    .with_result(ValueId::new(16)),
                    InstrNode::new(Inst::ICmp {
                        op: ICmpOp::Eq,
                        ty: Ty::I64,
                        lhs: ValueId::new(15),
                        rhs: ValueId::new(16),
                    })
                    .with_result(ValueId::new(17)),
                    InstrNode::new(Inst::CondBr {
                        cond: ValueId::new(17),
                        then_target: BlockId::new(1),
                        then_args: vec![],
                        else_target: BlockId::new(7),
                        else_args: vec![],
                    }),
                ],
            },
            TmirBlock {
                id: BlockId::new(7),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(1),
                    })
                    .with_result(ValueId::new(18)),
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: ValueId::new(10),
                        rhs: ValueId::new(18),
                    })
                    .with_result(ValueId::new(19)),
                    InstrNode::new(Inst::Br {
                        target: BlockId::new(5),
                        args: vec![ValueId::new(19)],
                    }),
                ],
            },
        ],
        proofs: vec![],
    };

    wrap_function_in_module(func, func_ty)
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
fn build_sum_array_tmir() -> Module {
    let func_ty = FuncTy {
        params: vec![Ty::Ptr, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    };

    let func = TmirFunction {
        id: FuncId::new(0),
        name: "sum_array".to_string(),
        ty: FuncTyId::new(0),
        entry: BlockId::new(0),
        blocks: vec![
            TmirBlock {
                id: BlockId::new(0),
                params: vec![(ValueId::new(0), Ty::Ptr), (ValueId::new(1), Ty::I64)],
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(0),
                    })
                    .with_result(ValueId::new(2)),
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(0),
                    })
                    .with_result(ValueId::new(3)),
                    InstrNode::new(Inst::Br {
                        target: BlockId::new(1),
                        args: vec![ValueId::new(2), ValueId::new(3)],
                    }),
                ],
            },
            TmirBlock {
                id: BlockId::new(1),
                params: vec![(ValueId::new(10), Ty::I64), (ValueId::new(11), Ty::I64)],
                body: vec![
                    InstrNode::new(Inst::ICmp {
                        op: ICmpOp::Sge,
                        ty: Ty::I64,
                        lhs: ValueId::new(11),
                        rhs: ValueId::new(1),
                    })
                    .with_result(ValueId::new(12)),
                    InstrNode::new(Inst::CondBr {
                        cond: ValueId::new(12),
                        then_target: BlockId::new(2),
                        then_args: vec![ValueId::new(10)],
                        else_target: BlockId::new(3),
                        else_args: vec![],
                    }),
                ],
            },
            TmirBlock {
                id: BlockId::new(2),
                params: vec![(ValueId::new(20), Ty::I64)],
                body: vec![InstrNode::new(Inst::Return {
                    values: vec![ValueId::new(20)],
                })],
            },
            TmirBlock {
                id: BlockId::new(3),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(8),
                    })
                    .with_result(ValueId::new(13)),
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Mul,
                        ty: Ty::I64,
                        lhs: ValueId::new(11),
                        rhs: ValueId::new(13),
                    })
                    .with_result(ValueId::new(14)),
                    InstrNode::new(Inst::Cast {
                        op: CastOp::PtrToInt,
                        src_ty: Ty::Ptr,
                        dst_ty: Ty::I64,
                        operand: ValueId::new(0),
                    })
                    .with_result(ValueId::new(15)),
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: ValueId::new(15),
                        rhs: ValueId::new(14),
                    })
                    .with_result(ValueId::new(16)),
                    InstrNode::new(Inst::Cast {
                        op: CastOp::IntToPtr,
                        src_ty: Ty::I64,
                        dst_ty: Ty::Ptr,
                        operand: ValueId::new(16),
                    })
                    .with_result(ValueId::new(17)),
                    InstrNode::new(Inst::Load {
                        ty: Ty::I64,
                        ptr: ValueId::new(17),
                    })
                    .with_result(ValueId::new(18)),
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: ValueId::new(10),
                        rhs: ValueId::new(18),
                    })
                    .with_result(ValueId::new(19)),
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(1),
                    })
                    .with_result(ValueId::new(21)),
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: ValueId::new(11),
                        rhs: ValueId::new(21),
                    })
                    .with_result(ValueId::new(22)),
                    InstrNode::new(Inst::Br {
                        target: BlockId::new(1),
                        args: vec![ValueId::new(19), ValueId::new(22)],
                    }),
                ],
            },
        ],
        proofs: vec![],
    };

    wrap_function_in_module(func, func_ty)
}

/// Build tMIR for: fn simple_add(a: i64, b: i64) -> i64 { a + b }
///
/// Minimal full-pipeline test: just add two arguments and return.
fn build_simple_add_tmir() -> Module {
    let func_ty = FuncTy {
        params: vec![Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    };

    let func = TmirFunction {
        id: FuncId::new(0),
        name: "simple_add".to_string(),
        ty: FuncTyId::new(0),
        entry: BlockId::new(0),
        blocks: vec![TmirBlock {
            id: BlockId::new(0),
            params: vec![(ValueId::new(0), Ty::I64), (ValueId::new(1), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(0),
                    rhs: ValueId::new(1),
                })
                .with_result(ValueId::new(2)),
                InstrNode::new(Inst::Return {
                    values: vec![ValueId::new(2)],
                }),
            ],
        }],
        proofs: vec![],
    };

    wrap_function_in_module(func, func_ty)
}

// ---------------------------------------------------------------------------
// Test: simple_add — minimal full-pipeline sanity check
// ---------------------------------------------------------------------------

/// Verifies the full pipeline works for the simplest possible case:
/// tMIR -> adapter -> ISel -> opt -> regalloc -> frame -> encode -> Mach-O.
#[test]
fn test_full_pipeline_simple_add_encoding() {
    let module = build_simple_add_tmir();

    let obj_bytes = compile_tmir_function(&module, OptLevel::O0).expect("full pipeline should compile simple_add");

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

    let module = build_simple_add_tmir();
    let obj_bytes = compile_tmir_function(&module, OptLevel::O0).expect("full pipeline should compile simple_add");

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
    let module = build_fibonacci_tmir();

    let obj_bytes = compile_tmir_function(&module, OptLevel::O0).expect("full pipeline should compile fibonacci");

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

    let module = build_fibonacci_tmir();
    let obj_bytes = compile_tmir_function(&module, OptLevel::O0).expect("full pipeline should compile fibonacci");

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
    let module = build_is_prime_tmir();

    let obj_bytes = compile_tmir_function(&module, OptLevel::O0).expect("full pipeline should compile is_prime");

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

    let module = build_is_prime_tmir();
    let obj_bytes = compile_tmir_function(&module, OptLevel::O0).expect("full pipeline should compile is_prime");

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
    let module = build_sum_array_tmir();

    let obj_bytes = compile_tmir_function(&module, OptLevel::O0).expect("full pipeline should compile sum_array");

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

    let module = build_sum_array_tmir();
    let obj_bytes = compile_tmir_function(&module, OptLevel::O0).expect("full pipeline should compile sum_array");

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
    let module = build_simple_add_tmir();

    for opt_level in &[OptLevel::O0, OptLevel::O1, OptLevel::O2] {
        let obj_bytes = compile_tmir_function(&module, *opt_level)
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
    let module = build_fibonacci_tmir();

    let (lir_func, _proof_ctx) =
        llvm2_lower::translate_function(test_function(&module), &module).expect("adapter should translate fibonacci");

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
    let module = build_is_prime_tmir();

    let (lir_func, _proof_ctx) =
        llvm2_lower::translate_function(test_function(&module), &module).expect("adapter should translate is_prime");

    assert_eq!(lir_func.name, "is_prime");
    // Should have 8 blocks.
    assert_eq!(
        lir_func.blocks.len(),
        8,
        "is_prime should have 8 blocks, got {}",
        lir_func.blocks.len()
    );
}
