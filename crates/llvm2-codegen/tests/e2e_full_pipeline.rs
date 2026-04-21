// llvm2-codegen/tests/e2e_full_pipeline.rs - Full pipeline e2e tests from tMIR
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
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

use tmir::{Block as TmirBlock, Function as TmirFunction, Module as TmirModule, FuncTy, Ty, Constant};
use tmir::{Inst, InstrNode, BinOp, ICmpOp, CastOp};
use tmir::{BlockId, FuncId, ValueId};

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
    module: &TmirModule,
    opt_level: OptLevel,
) -> Result<Vec<u8>, String> {
    // Phase 0: Translate tMIR -> LIR (adapter)
    let (lir_func, _proof_ctx) =
        llvm2_lower::translate_function(tmir_func, module)
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
fn build_fibonacci_tmir() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "fibonacci", ft_id, BlockId::new(0));
    func.blocks = vec![
            // bb0 (entry): check if n <= 1
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
            // bb1 (ret_n): return n directly
            TmirBlock {
                id: BlockId::new(1),
                params: vec![],
                body: vec![InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(0)], // n from bb0
                    })],
            },
            // bb2 (loop_init): setup a=0, b=1, i=2, jump to loop
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
            // bb3 (loop): loop body with block parameters
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
                    // condbr: loop back or exit
                    InstrNode::new(Inst::CondBr {
                            cond: ValueId::new(26),
                            then_target: BlockId::new(3), // loop back
                            then_args: vec![ValueId::new(21), ValueId::new(23), ValueId::new(25)], // new_a=b, new_b=tmp, new_i
                            else_target: BlockId::new(4), // exit
                            else_args: vec![ValueId::new(23)], // result = tmp (= a+b = new b)
                        }),
                ],
            },
            // bb4 (ret_b): return the result
            TmirBlock {
                id: BlockId::new(4),
                params: vec![(ValueId::new(30), Ty::I64)], // result
                body: vec![InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(30)],
                    })],
            },
        ];
    module.add_function(func.clone());
    (func, module)
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
fn build_is_prime_tmir() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "is_prime", ft_id, BlockId::new(0));
    func.blocks = vec![
            // bb0 (entry): check n <= 1
            TmirBlock {
                id: BlockId::new(0),
                params: vec![(ValueId::new(0), Ty::I64)], // n
                body: vec![
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(1) })

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
                            then_target: BlockId::new(1), // return 0
                            then_args: vec![],
                            else_target: BlockId::new(2), // check n <= 3
                            else_args: vec![],
                        }),
                ],
            },
            // bb1: return 0 (not prime)
            TmirBlock {
                id: BlockId::new(1),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(0) })

                        .with_result(ValueId::new(3)),
                    InstrNode::new(Inst::Return { values: vec![ValueId::new(3)] }),
                ],
            },
            // bb2: check n <= 3
            TmirBlock {
                id: BlockId::new(2),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(3) })

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
                            then_target: BlockId::new(3), // return 1 (2 and 3 are prime)
                            then_args: vec![],
                            else_target: BlockId::new(4), // loop init
                            else_args: vec![],
                        }),
                ],
            },
            // bb3: return 1 (prime)
            TmirBlock {
                id: BlockId::new(3),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(1) })

                        .with_result(ValueId::new(6)),
                    InstrNode::new(Inst::Return { values: vec![ValueId::new(6)] }),
                ],
            },
            // bb4: loop init (i = 2)
            TmirBlock {
                id: BlockId::new(4),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(2) })

                        .with_result(ValueId::new(7)),
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(5),
                            args: vec![ValueId::new(7)],
                        }),
                ],
            },
            // bb5: loop header (with param i)
            TmirBlock {
                id: BlockId::new(5),
                params: vec![(ValueId::new(10), Ty::I64)], // i
                body: vec![
                    // i_sq = i * i
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Mul,
                            ty: Ty::I64,
                            lhs: ValueId::new(10),
                            rhs: ValueId::new(10),
                        })

                        .with_result(ValueId::new(11)),
                    // cmp i_sq > n
                    InstrNode::new(Inst::ICmp {
                            op: ICmpOp::Sgt,
                            ty: Ty::I64,
                            lhs: ValueId::new(11),
                            rhs: ValueId::new(0),
                        })

                        .with_result(ValueId::new(12)),
                    InstrNode::new(Inst::CondBr {
                            cond: ValueId::new(12),
                            then_target: BlockId::new(3), // return 1 (prime)
                            then_args: vec![],
                            else_target: BlockId::new(6), // check divisibility
                            else_args: vec![],
                        }),
                ],
            },
            // bb6: check n % i == 0 using div+mul+sub
            TmirBlock {
                id: BlockId::new(6),
                params: vec![],
                body: vec![
                    // q = n / i (signed div)
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::SDiv,
                            ty: Ty::I64,
                            lhs: ValueId::new(0),
                            rhs: ValueId::new(10),
                        })

                        .with_result(ValueId::new(13)),
                    // qi = q * i
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Mul,
                            ty: Ty::I64,
                            lhs: ValueId::new(13),
                            rhs: ValueId::new(10),
                        })

                        .with_result(ValueId::new(14)),
                    // r = n - qi
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Sub,
                            ty: Ty::I64,
                            lhs: ValueId::new(0),
                            rhs: ValueId::new(14),
                        })

                        .with_result(ValueId::new(15)),
                    // cmp r == 0
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(0) })

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
                            then_target: BlockId::new(1), // return 0 (not prime)
                            then_args: vec![],
                            else_target: BlockId::new(7), // increment i
                            else_args: vec![],
                        }),
                ],
            },
            // bb7: increment i, loop back
            TmirBlock {
                id: BlockId::new(7),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(1) })

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
        ];
    module.add_function(func.clone());
    (func, module)
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
fn build_sum_array_tmir() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![
                Ty::Ptr,  // arr
                Ty::I64,                       // len
            ],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "sum_array", ft_id, BlockId::new(0));
    func.blocks = vec![
            // bb0 (entry): init sum=0, i=0, jump to loop
            TmirBlock {
                id: BlockId::new(0),
                params: vec![
                    (ValueId::new(0), Ty::Ptr), // arr
                    (ValueId::new(1), Ty::I64),                     // len
                ],
                body: vec![
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(0) })
                        .with_result(ValueId::new(2)),
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(0) })
                        .with_result(ValueId::new(3)),
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(1),
                            args: vec![ValueId::new(2), ValueId::new(3)],
                        }),
                ],
            },
            // bb1 (loop header): params(sum, i)
            TmirBlock {
                id: BlockId::new(1),
                params: vec![
                    (ValueId::new(10), Ty::I64), // sum
                    (ValueId::new(11), Ty::I64), // i
                ],
                body: vec![
                    // cmp i >= len
                    InstrNode::new(Inst::ICmp {
                            op: ICmpOp::Sge,
                            ty: Ty::I64,
                            lhs: ValueId::new(11), // i
                            rhs: ValueId::new(1),  // len
                        })

                        .with_result(ValueId::new(12)),
                    InstrNode::new(Inst::CondBr {
                            cond: ValueId::new(12),
                            then_target: BlockId::new(2), // return sum
                            then_args: vec![ValueId::new(10)],
                            else_target: BlockId::new(3), // loop body
                            else_args: vec![],
                        }),
                ],
            },
            // bb2 (exit): return sum
            TmirBlock {
                id: BlockId::new(2),
                params: vec![(ValueId::new(20), Ty::I64)],
                body: vec![InstrNode::new(Inst::Return { values: vec![ValueId::new(20)] })],
            },
            // bb3 (loop body): load arr[i], accumulate, increment
            TmirBlock {
                id: BlockId::new(3),
                params: vec![],
                body: vec![
                    // offset = i * 8
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(8) })

                        .with_result(ValueId::new(13)),
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Mul,
                            ty: Ty::I64,
                            lhs: ValueId::new(11), // i
                            rhs: ValueId::new(13), // 8
                        })
                        .with_result(ValueId::new(14)),
                    // ptr = arr + offset (pointer arithmetic as integer add on I64)
                    // We cast arr (Ptr) to I64 first, then add, then cast back
                    InstrNode::new(Inst::Cast {
                            op: CastOp::PtrToInt,
                            src_ty: Ty::Ptr,
                            dst_ty: Ty::I64,
                            operand: ValueId::new(0), // arr
                        })
                        .with_result(ValueId::new(15)),
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Add,
                            ty: Ty::I64,
                            lhs: ValueId::new(15), // arr_int
                            rhs: ValueId::new(14), // offset
                        })
                        .with_result(ValueId::new(16)),
                    InstrNode::new(Inst::Cast {
                            op: CastOp::IntToPtr,
                            src_ty: Ty::I64,
                            dst_ty: Ty::Ptr,
                            operand: ValueId::new(16),
                        })
                        .with_result(ValueId::new(17)),
                    // val = load(elem_ptr)
                    InstrNode::new(Inst::Load {
                            ty: Ty::I64,
                            ptr: ValueId::new(17),
                            volatile: false,
                            align: None,
                        })
                        .with_result(ValueId::new(18)),
                    // new_sum = sum + val
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Add,
                            ty: Ty::I64,
                            lhs: ValueId::new(10), // sum
                            rhs: ValueId::new(18), // val
                        })
                        .with_result(ValueId::new(19)),
                    // new_i = i + 1
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(1) })

                        .with_result(ValueId::new(21)),
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Add,
                            ty: Ty::I64,
                            lhs: ValueId::new(11),
                            rhs: ValueId::new(21),
                        })
                        .with_result(ValueId::new(22)),
                    // loop back
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(1),
                            args: vec![ValueId::new(19), ValueId::new(22)],
                        }),
                ],
            },
        ];
    module.add_function(func.clone());
    (func, module)
}

/// Build tMIR for: fn simple_add(a: i64, b: i64) -> i64 { a + b }
///
/// Minimal full-pipeline test: just add two arguments and return.
fn build_simple_add_tmir() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64, Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "simple_add", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
            id: BlockId::new(0),
            params: vec![
                (ValueId::new(0), Ty::I64),
                (ValueId::new(1), Ty::I64),
            ],
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
        }];
    module.add_function(func.clone());
    (func, module)
}

// ---------------------------------------------------------------------------
// Test: simple_add — minimal full-pipeline sanity check
// ---------------------------------------------------------------------------

/// Verifies the full pipeline works for the simplest possible case:
/// tMIR -> adapter -> ISel -> opt -> regalloc -> frame -> encode -> Mach-O.
#[test]
fn test_full_pipeline_simple_add_encoding() {
    let (tmir_func, module) = build_simple_add_tmir();

    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
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

    let (tmir_func, module) = build_simple_add_tmir();
    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
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
    let (tmir_func, module) = build_fibonacci_tmir();

    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
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

    let (tmir_func, module) = build_fibonacci_tmir();
    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
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
    let (tmir_func, module) = build_is_prime_tmir();

    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
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

    let (tmir_func, module) = build_is_prime_tmir();
    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
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
    let (tmir_func, module) = build_sum_array_tmir();

    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
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

    let (tmir_func, module) = build_sum_array_tmir();
    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
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
    let (tmir_func, module) = build_simple_add_tmir();

    for opt_level in &[OptLevel::O0, OptLevel::O1, OptLevel::O2] {
        let obj_bytes = compile_tmir_function(&tmir_func, &module, *opt_level)
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
    let (tmir_func, module) = build_fibonacci_tmir();

    let (lir_func, _proof_ctx) =
        llvm2_lower::translate_function(&tmir_func, &module)
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
    // 5 original blocks + copy blocks for phi elimination at branch targets.
    assert!(
        lir_func.blocks.len() >= 5,
        "fibonacci should have at least 5 blocks, got {}",
        lir_func.blocks.len()
    );
}

/// Verify the adapter correctly translates is_prime to LIR.
#[test]
fn test_adapter_is_prime_structure() {
    let (tmir_func, module) = build_is_prime_tmir();

    let (lir_func, _proof_ctx) =
        llvm2_lower::translate_function(&tmir_func, &module)
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

// ---------------------------------------------------------------------------
// tMIR function builders: gcd and collatz_steps
// ---------------------------------------------------------------------------

/// Build tMIR for: fn gcd(a: i64, b: i64) -> i64
///
/// Euclidean algorithm using SRem:
///   bb0 (entry): params a, b. Jump to bb1 (loop) with args (a, b)
///   bb1 (loop): params (a, b). Check b != 0 -> condbr bb2 (body) or bb3 (exit with a)
///   bb2 (body): t = a SRem b. br -> bb1(b, t)
///   bb3 (exit): param (result). return result
fn build_gcd_tmir() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64, Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "gcd", ft_id, BlockId::new(0));
    func.blocks = vec![
            // bb0 (entry): params a, b. Jump to loop.
            TmirBlock {
                id: BlockId::new(0),
                params: vec![
                    (ValueId::new(0), Ty::I64), // a
                    (ValueId::new(1), Ty::I64), // b
                ],
                body: vec![
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(1),
                            args: vec![ValueId::new(0), ValueId::new(1)],
                        }),
                ],
            },
            // bb1 (loop): params (a, b). Check b != 0.
            TmirBlock {
                id: BlockId::new(1),
                params: vec![
                    (ValueId::new(10), Ty::I64), // a
                    (ValueId::new(11), Ty::I64), // b
                ],
                body: vec![
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(0) })
                        .with_result(ValueId::new(12)),
                    InstrNode::new(Inst::ICmp {
                            op: ICmpOp::Ne,
                            ty: Ty::I64,
                            lhs: ValueId::new(11), // b
                            rhs: ValueId::new(12), // 0
                        })
                        .with_result(ValueId::new(13)),
                    InstrNode::new(Inst::CondBr {
                            cond: ValueId::new(13),
                            then_target: BlockId::new(2), // body (b != 0)
                            then_args: vec![],
                            else_target: BlockId::new(3), // exit (b == 0, return a)
                            else_args: vec![ValueId::new(10)],
                        }),
                ],
            },
            // bb2 (body): t = a % b, loop back with (b, t)
            TmirBlock {
                id: BlockId::new(2),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::SRem,
                            ty: Ty::I64,
                            lhs: ValueId::new(10), // a
                            rhs: ValueId::new(11), // b
                        })
                        .with_result(ValueId::new(20)),
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(1),
                            args: vec![ValueId::new(11), ValueId::new(20)], // (b, t)
                        }),
                ],
            },
            // bb3 (exit): return result
            TmirBlock {
                id: BlockId::new(3),
                params: vec![(ValueId::new(30), Ty::I64)], // result
                body: vec![InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(30)],
                    })],
            },
        ];
    module.add_function(func.clone());
    (func, module)
}

/// Build tMIR for: fn collatz_steps(n: i64) -> i64
///
/// Count steps to reach 1:
///   bb0 (entry): param n, jump to loop with (n, 0)
///   bb1 (loop): params (n, steps). Check n != 1 -> condbr bb2 or bb3
///   bb2 (check even/odd): n % 2 == 0? -> condbr bb4 (even) or bb5 (odd)
///   bb3 (exit): param (result). return result
///   bb4 (even): new_n = n / 2, new_steps = steps + 1, br bb1
///   bb5 (odd): tmp = n * 3, new_n = tmp + 1, new_steps = steps + 1, br bb1
fn build_collatz_steps_tmir() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "collatz_steps", ft_id, BlockId::new(0));
    func.blocks = vec![
            // bb0 (entry): param n, jump to loop with (n, 0)
            TmirBlock {
                id: BlockId::new(0),
                params: vec![(ValueId::new(0), Ty::I64)], // n
                body: vec![
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(0) })
                        .with_result(ValueId::new(1)),
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(1),
                            args: vec![ValueId::new(0), ValueId::new(1)], // (n, 0)
                        }),
                ],
            },
            // bb1 (loop): params (n, steps). Check n != 1.
            TmirBlock {
                id: BlockId::new(1),
                params: vec![
                    (ValueId::new(10), Ty::I64), // n
                    (ValueId::new(11), Ty::I64), // steps
                ],
                body: vec![
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(1) })
                        .with_result(ValueId::new(12)),
                    InstrNode::new(Inst::ICmp {
                            op: ICmpOp::Ne,
                            ty: Ty::I64,
                            lhs: ValueId::new(10), // n
                            rhs: ValueId::new(12), // 1
                        })
                        .with_result(ValueId::new(13)),
                    InstrNode::new(Inst::CondBr {
                            cond: ValueId::new(13),
                            then_target: BlockId::new(2), // check even/odd (n != 1)
                            then_args: vec![],
                            else_target: BlockId::new(3), // exit (n == 1, return steps)
                            else_args: vec![ValueId::new(11)],
                        }),
                ],
            },
            // bb2 (check even/odd): n % 2 == 0?
            TmirBlock {
                id: BlockId::new(2),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(2) })
                        .with_result(ValueId::new(20)),
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::SRem,
                            ty: Ty::I64,
                            lhs: ValueId::new(10), // n
                            rhs: ValueId::new(20), // 2
                        })
                        .with_result(ValueId::new(21)),
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(0) })
                        .with_result(ValueId::new(22)),
                    InstrNode::new(Inst::ICmp {
                            op: ICmpOp::Eq,
                            ty: Ty::I64,
                            lhs: ValueId::new(21), // n % 2
                            rhs: ValueId::new(22), // 0
                        })
                        .with_result(ValueId::new(23)),
                    InstrNode::new(Inst::CondBr {
                            cond: ValueId::new(23),
                            then_target: BlockId::new(4), // even
                            then_args: vec![],
                            else_target: BlockId::new(5), // odd
                            else_args: vec![],
                        }),
                ],
            },
            // bb3 (exit): return steps
            TmirBlock {
                id: BlockId::new(3),
                params: vec![(ValueId::new(30), Ty::I64)], // result
                body: vec![InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(30)],
                    })],
            },
            // bb4 (even): new_n = n / 2, new_steps = steps + 1, br bb1
            TmirBlock {
                id: BlockId::new(4),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(2) })
                        .with_result(ValueId::new(40)),
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::SDiv,
                            ty: Ty::I64,
                            lhs: ValueId::new(10), // n
                            rhs: ValueId::new(40), // 2
                        })
                        .with_result(ValueId::new(41)),
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(1) })
                        .with_result(ValueId::new(42)),
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Add,
                            ty: Ty::I64,
                            lhs: ValueId::new(11), // steps
                            rhs: ValueId::new(42), // 1
                        })
                        .with_result(ValueId::new(43)),
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(1),
                            args: vec![ValueId::new(41), ValueId::new(43)], // (n/2, steps+1)
                        }),
                ],
            },
            // bb5 (odd): tmp = n * 3, new_n = tmp + 1, new_steps = steps + 1, br bb1
            TmirBlock {
                id: BlockId::new(5),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(3) })
                        .with_result(ValueId::new(50)),
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Mul,
                            ty: Ty::I64,
                            lhs: ValueId::new(10), // n
                            rhs: ValueId::new(50), // 3
                        })
                        .with_result(ValueId::new(51)),
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(1) })
                        .with_result(ValueId::new(52)),
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Add,
                            ty: Ty::I64,
                            lhs: ValueId::new(51), // n*3
                            rhs: ValueId::new(52), // 1
                        })
                        .with_result(ValueId::new(53)),
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(1) })
                        .with_result(ValueId::new(54)),
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Add,
                            ty: Ty::I64,
                            lhs: ValueId::new(11), // steps
                            rhs: ValueId::new(54), // 1
                        })
                        .with_result(ValueId::new(55)),
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(1),
                            args: vec![ValueId::new(53), ValueId::new(55)], // (3n+1, steps+1)
                        }),
                ],
            },
        ];
    module.add_function(func.clone());
    (func, module)
}

// ---------------------------------------------------------------------------
// Test: gcd — SRem, Euclidean algorithm, loop with block params
// ---------------------------------------------------------------------------

/// Verifies that gcd compiles through the full pipeline (encoding check).
#[test]
fn test_full_pipeline_gcd_encoding() {
    let (tmir_func, module) = build_gcd_tmir();

    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
        .expect("full pipeline should compile gcd");

    assert!(obj_bytes.len() >= 4, "object file should not be empty");
    assert_eq!(
        &obj_bytes[0..4],
        &[0xCF, 0xFA, 0xED, 0xFE],
        "should be valid Mach-O magic"
    );
}

/// Full pipeline: gcd compiled, linked, and executed.
#[test]
fn test_full_pipeline_gcd_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("full_gcd");

    let (tmir_func, module) = build_gcd_tmir();
    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
        .expect("full pipeline should compile gcd");

    let obj_path = write_object_file(&dir, "gcd.o", &obj_bytes);

    let driver_src = r#"
#include <stdio.h>
extern long gcd(long a, long b);
int main(void) {
    long r1 = gcd(12, 8);
    long r2 = gcd(100, 75);
    long r3 = gcd(17, 13);
    long r4 = gcd(0, 5);
    long r5 = gcd(7, 0);
    printf("gcd(12,8)=%ld gcd(100,75)=%ld gcd(17,13)=%ld gcd(0,5)=%ld gcd(7,0)=%ld\n",
           r1, r2, r3, r4, r5);
    if (r1 != 4) return 1;
    if (r2 != 25) return 2;
    if (r3 != 1) return 3;
    if (r4 != 5) return 4;
    if (r5 != 7) return 5;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_full_gcd");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_full_pipeline_gcd stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "gcd full pipeline test failed with exit code {} \
         (1=gcd(12,8)!=4, 2=gcd(100,75)!=25, 3=gcd(17,13)!=1, 4=gcd(0,5)!=5, 5=gcd(7,0)!=7). stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test: collatz_steps — SRem, SDiv, Mul, multi-path loop
// ---------------------------------------------------------------------------

/// Verifies that collatz_steps compiles through the full pipeline (encoding check).
#[test]
fn test_full_pipeline_collatz_encoding() {
    let (tmir_func, module) = build_collatz_steps_tmir();

    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
        .expect("full pipeline should compile collatz_steps");

    assert!(obj_bytes.len() >= 4, "object file should not be empty");
    assert_eq!(
        &obj_bytes[0..4],
        &[0xCF, 0xFA, 0xED, 0xFE],
        "should be valid Mach-O magic"
    );
}

/// Full pipeline: collatz_steps compiled, linked, and executed.
#[test]
fn test_full_pipeline_collatz_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("full_collatz");

    let (tmir_func, module) = build_collatz_steps_tmir();
    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
        .expect("full pipeline should compile collatz_steps");

    let obj_path = write_object_file(&dir, "collatz_steps.o", &obj_bytes);

    let driver_src = r#"
#include <stdio.h>
extern long collatz_steps(long n);
int main(void) {
    long r1 = collatz_steps(1);
    long r2 = collatz_steps(2);
    long r6 = collatz_steps(6);
    long r27 = collatz_steps(27);
    printf("collatz(1)=%ld collatz(2)=%ld collatz(6)=%ld collatz(27)=%ld\n",
           r1, r2, r6, r27);
    if (r1 != 0) return 1;
    if (r2 != 1) return 2;
    if (r6 != 8) return 3;
    if (r27 != 111) return 4;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_full_collatz");

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_full_pipeline_collatz stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "collatz full pipeline test failed with exit code {} \
         (1=collatz(1)!=0, 2=collatz(2)!=1, 3=collatz(6)!=8, 4=collatz(27)!=111). stdout: {}",
        exit_code, stdout
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test: adapter structure checks for gcd and collatz
// ---------------------------------------------------------------------------

/// Verify the adapter correctly translates gcd to LIR.
#[test]
fn test_adapter_gcd_structure() {
    let (tmir_func, module) = build_gcd_tmir();

    let (lir_func, _proof_ctx) =
        llvm2_lower::translate_function(&tmir_func, &module)
            .expect("adapter should translate gcd");

    assert_eq!(lir_func.name, "gcd");
    assert_eq!(
        lir_func.signature.params,
        vec![llvm2_lower::types::Type::I64, llvm2_lower::types::Type::I64]
    );
    assert_eq!(
        lir_func.signature.returns,
        vec![llvm2_lower::types::Type::I64]
    );
    // 4 original blocks + copy blocks for phi elimination at branch targets.
    assert!(
        lir_func.blocks.len() >= 4,
        "gcd should have at least 4 blocks, got {}",
        lir_func.blocks.len()
    );
}

/// Verify the adapter correctly translates collatz_steps to LIR.
#[test]
fn test_adapter_collatz_structure() {
    let (tmir_func, module) = build_collatz_steps_tmir();

    let (lir_func, _proof_ctx) =
        llvm2_lower::translate_function(&tmir_func, &module)
            .expect("adapter should translate collatz_steps");

    assert_eq!(lir_func.name, "collatz_steps");
    assert_eq!(
        lir_func.signature.params,
        vec![llvm2_lower::types::Type::I64]
    );
    assert_eq!(
        lir_func.signature.returns,
        vec![llvm2_lower::types::Type::I64]
    );
    // 6 original blocks + copy blocks for phi elimination at branch targets.
    assert!(
        lir_func.blocks.len() >= 6,
        "collatz_steps should have at least 6 blocks, got {}",
        lir_func.blocks.len()
    );
}

// ===========================================================================
// HARDER PROGRAMS: Wave 42 — stress multiple compiler subsystems
// ===========================================================================
//
// These tests go beyond Wave 41 (fibonacci, gcd, collatz, sum, factorial)
// by combining multiple operations within tighter loops, using nested loops,
// and exercising division + multiplication + comparison in combination.
//
// Part of #301 — AArch64 correctness validation: harder programs

// ---------------------------------------------------------------------------
// tMIR builders for harder programs
// ---------------------------------------------------------------------------

/// Build tMIR for: fn power(base: i64, exp: i64) -> i64
///
/// Iterative exponentiation:
///   if exp <= 0: return 1
///   result = 1
///   for i in 0..exp: result *= base
///   return result
///
/// Stresses: loop with multiplication accumulator, two-parameter function,
///           early exit for exp <= 0, Mul in tight loop.
fn build_power_tmir() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64, Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "power", ft_id, BlockId::new(0));
    func.blocks = vec![
            // bb0 (entry): params(base, exp), if exp <= 0 return 1, else init loop
            TmirBlock {
                id: BlockId::new(0),
                params: vec![
                    (ValueId::new(0), Ty::I64), // base
                    (ValueId::new(1), Ty::I64), // exp
                ],
                body: vec![
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(0),
                        })
                        .with_result(ValueId::new(2)),
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(1),
                        })
                        .with_result(ValueId::new(3)),
                    InstrNode::new(Inst::ICmp {
                            op: ICmpOp::Sle,
                            ty: Ty::I64,
                            lhs: ValueId::new(1), // exp
                            rhs: ValueId::new(2), // 0
                        })
                        .with_result(ValueId::new(4)),
                    InstrNode::new(Inst::CondBr {
                            cond: ValueId::new(4),
                            then_target: BlockId::new(3), // exit with 1
                            then_args: vec![ValueId::new(3)],
                            else_target: BlockId::new(1), // loop_init
                            else_args: vec![],
                        }),
                ],
            },
            // bb1 (loop_init): result=1, i=0, br -> bb2
            TmirBlock {
                id: BlockId::new(1),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(1),
                        })
                        .with_result(ValueId::new(10)),
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(0),
                        })
                        .with_result(ValueId::new(11)),
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(2),
                            args: vec![ValueId::new(10), ValueId::new(11)],
                        }),
                ],
            },
            // bb2 (loop): params(result, i), multiply and advance
            TmirBlock {
                id: BlockId::new(2),
                params: vec![
                    (ValueId::new(20), Ty::I64), // result
                    (ValueId::new(21), Ty::I64), // i
                ],
                body: vec![
                    // new_result = result * base
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Mul,
                            ty: Ty::I64,
                            lhs: ValueId::new(20), // result
                            rhs: ValueId::new(0),  // base
                        })
                        .with_result(ValueId::new(22)),
                    // new_i = i + 1
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(1),
                        })
                        .with_result(ValueId::new(23)),
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Add,
                            ty: Ty::I64,
                            lhs: ValueId::new(21), // i
                            rhs: ValueId::new(23), // 1
                        })
                        .with_result(ValueId::new(24)),
                    // cmp new_i < exp
                    InstrNode::new(Inst::ICmp {
                            op: ICmpOp::Slt,
                            ty: Ty::I64,
                            lhs: ValueId::new(24), // new_i
                            rhs: ValueId::new(1),  // exp
                        })
                        .with_result(ValueId::new(25)),
                    InstrNode::new(Inst::CondBr {
                            cond: ValueId::new(25),
                            then_target: BlockId::new(2), // loop
                            then_args: vec![ValueId::new(22), ValueId::new(24)],
                            else_target: BlockId::new(3), // exit
                            else_args: vec![ValueId::new(22)],
                        }),
                ],
            },
            // bb3 (exit): return result
            TmirBlock {
                id: BlockId::new(3),
                params: vec![(ValueId::new(30), Ty::I64)],
                body: vec![InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(30)],
                    })],
            },
        ];
    module.add_function(func.clone());
    (func, module)
}

/// Build tMIR for: fn count_digits(n: i64) -> i64
///
/// Counts decimal digits: count=0, while n>0 { n/=10; count+=1 }
/// count_digits(0) = 0.
///
/// Stresses: SDiv (division by 10) in a loop, counter pattern,
///           zero-trip loop for n=0.
fn build_count_digits_tmir() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "count_digits", ft_id, BlockId::new(0));
    func.blocks = vec![
            // bb0 (entry): param n, br -> bb1
            TmirBlock {
                id: BlockId::new(0),
                params: vec![(ValueId::new(0), Ty::I64)], // n
                body: vec![
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(1),
                            args: vec![],
                        }),
                ],
            },
            // bb1 (loop_init): count=0, br -> bb2(count, n)
            TmirBlock {
                id: BlockId::new(1),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(0),
                        })
                        .with_result(ValueId::new(10)),
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(2),
                            args: vec![ValueId::new(10), ValueId::new(0)],
                        }),
                ],
            },
            // bb2 (loop): params(count, n_cur), exit if n_cur <= 0
            TmirBlock {
                id: BlockId::new(2),
                params: vec![
                    (ValueId::new(20), Ty::I64), // count
                    (ValueId::new(21), Ty::I64), // n_cur
                ],
                body: vec![
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(0),
                        })
                        .with_result(ValueId::new(22)),
                    InstrNode::new(Inst::ICmp {
                            op: ICmpOp::Sle,
                            ty: Ty::I64,
                            lhs: ValueId::new(21), // n_cur
                            rhs: ValueId::new(22), // 0
                        })
                        .with_result(ValueId::new(23)),
                    InstrNode::new(Inst::CondBr {
                            cond: ValueId::new(23),
                            then_target: BlockId::new(4), // exit
                            then_args: vec![ValueId::new(20)],
                            else_target: BlockId::new(3), // body
                            else_args: vec![],
                        }),
                ],
            },
            // bb3 (body): new_n = n_cur / 10, new_count = count + 1, br -> bb2
            TmirBlock {
                id: BlockId::new(3),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(10),
                        })
                        .with_result(ValueId::new(30)),
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::SDiv,
                            ty: Ty::I64,
                            lhs: ValueId::new(21), // n_cur
                            rhs: ValueId::new(30), // 10
                        })
                        .with_result(ValueId::new(31)),
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(1),
                        })
                        .with_result(ValueId::new(32)),
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Add,
                            ty: Ty::I64,
                            lhs: ValueId::new(20), // count
                            rhs: ValueId::new(32), // 1
                        })
                        .with_result(ValueId::new(33)),
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(2),
                            args: vec![ValueId::new(33), ValueId::new(31)],
                        }),
                ],
            },
            // bb4 (exit): return result
            TmirBlock {
                id: BlockId::new(4),
                params: vec![(ValueId::new(40), Ty::I64)],
                body: vec![InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(40)],
                    })],
            },
        ];
    module.add_function(func.clone());
    (func, module)
}

/// Build tMIR for: fn nested_loop_sum(n: i64) -> i64
///
/// Double nested loop: sum=0, for i in 0..n { for j in 0..n { sum += i*j } }
///
/// This is the HARDEST test so far: two nested loops, O(n^2) iterations,
/// high register pressure (n, i, j, sum, product all live simultaneously),
/// Mul inside inner loop, cross-block value references (i from outer loop
/// used in inner loop body).
fn build_nested_loop_sum_tmir() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "nested_loop_sum", ft_id, BlockId::new(0));
    func.blocks = vec![
            // bb0 (entry): param n, br -> bb1
            TmirBlock {
                id: BlockId::new(0),
                params: vec![(ValueId::new(0), Ty::I64)], // n
                body: vec![
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(1),
                            args: vec![],
                        }),
                ],
            },
            // bb1 (outer_init): sum=0, i=0, br -> bb2
            TmirBlock {
                id: BlockId::new(1),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(0),
                        })
                        .with_result(ValueId::new(10)),
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(0),
                        })
                        .with_result(ValueId::new(11)),
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(2),
                            args: vec![ValueId::new(10), ValueId::new(11)],
                        }),
                ],
            },
            // bb2 (outer_check): params(sum, i), if i >= n exit else enter inner loop
            TmirBlock {
                id: BlockId::new(2),
                params: vec![
                    (ValueId::new(20), Ty::I64), // sum
                    (ValueId::new(21), Ty::I64), // i
                ],
                body: vec![
                    InstrNode::new(Inst::ICmp {
                            op: ICmpOp::Sge,
                            ty: Ty::I64,
                            lhs: ValueId::new(21), // i
                            rhs: ValueId::new(0),  // n
                        })
                        .with_result(ValueId::new(22)),
                    InstrNode::new(Inst::CondBr {
                            cond: ValueId::new(22),
                            then_target: BlockId::new(7), // exit
                            then_args: vec![ValueId::new(20)],
                            else_target: BlockId::new(3), // inner_init
                            else_args: vec![],
                        }),
                ],
            },
            // bb3 (inner_init): j=0, br -> bb4(sum, j)
            TmirBlock {
                id: BlockId::new(3),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(0),
                        })
                        .with_result(ValueId::new(30)),
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(4),
                            args: vec![ValueId::new(20), ValueId::new(30)],
                        }),
                ],
            },
            // bb4 (inner_check): params(sum2, j), if j >= n go to outer_next else inner_body
            TmirBlock {
                id: BlockId::new(4),
                params: vec![
                    (ValueId::new(40), Ty::I64), // sum2
                    (ValueId::new(41), Ty::I64), // j
                ],
                body: vec![
                    InstrNode::new(Inst::ICmp {
                            op: ICmpOp::Sge,
                            ty: Ty::I64,
                            lhs: ValueId::new(41), // j
                            rhs: ValueId::new(0),  // n
                        })
                        .with_result(ValueId::new(42)),
                    InstrNode::new(Inst::CondBr {
                            cond: ValueId::new(42),
                            then_target: BlockId::new(6), // outer_next
                            then_args: vec![ValueId::new(40)],
                            else_target: BlockId::new(5), // inner_body
                            else_args: vec![],
                        }),
                ],
            },
            // bb5 (inner_body): sum2 += i * j, j += 1, br -> bb4
            TmirBlock {
                id: BlockId::new(5),
                params: vec![],
                body: vec![
                    // prod = i * j
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Mul,
                            ty: Ty::I64,
                            lhs: ValueId::new(21), // i (from outer loop)
                            rhs: ValueId::new(41), // j
                        })
                        .with_result(ValueId::new(50)),
                    // new_sum = sum2 + prod
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Add,
                            ty: Ty::I64,
                            lhs: ValueId::new(40), // sum2
                            rhs: ValueId::new(50), // prod
                        })
                        .with_result(ValueId::new(51)),
                    // new_j = j + 1
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(1),
                        })
                        .with_result(ValueId::new(52)),
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Add,
                            ty: Ty::I64,
                            lhs: ValueId::new(41), // j
                            rhs: ValueId::new(52), // 1
                        })
                        .with_result(ValueId::new(53)),
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(4),
                            args: vec![ValueId::new(51), ValueId::new(53)],
                        }),
                ],
            },
            // bb6 (outer_next): new_i = i + 1, br -> bb2(new_outer_sum, new_i)
            TmirBlock {
                id: BlockId::new(6),
                params: vec![(ValueId::new(60), Ty::I64)], // new_outer_sum
                body: vec![
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(1),
                        })
                        .with_result(ValueId::new(61)),
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Add,
                            ty: Ty::I64,
                            lhs: ValueId::new(21), // i (from outer loop)
                            rhs: ValueId::new(61), // 1
                        })
                        .with_result(ValueId::new(62)),
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(2),
                            args: vec![ValueId::new(60), ValueId::new(62)],
                        }),
                ],
            },
            // bb7 (exit): return result
            TmirBlock {
                id: BlockId::new(7),
                params: vec![(ValueId::new(70), Ty::I64)],
                body: vec![InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(70)],
                    })],
            },
        ];
    module.add_function(func.clone());
    (func, module)
}

// ---------------------------------------------------------------------------
// Test: power — loop with multiplication accumulator, early exit
// ---------------------------------------------------------------------------

/// Verifies that power compiles through the full pipeline (encoding check).
#[test]
fn test_full_pipeline_power_encoding() {
    let (tmir_func, module) = build_power_tmir();

    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
        .expect("full pipeline should compile power");

    assert!(obj_bytes.len() >= 4, "object file should not be empty");
    assert_eq!(
        &obj_bytes[0..4],
        &[0xCF, 0xFA, 0xED, 0xFE],
        "should be valid Mach-O magic"
    );
}

/// Full pipeline: power compiled, linked, and executed with multiple test cases.
#[test]
fn test_full_pipeline_power_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("full_power");
    let (tmir_func, module) = build_power_tmir();
    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
        .expect("full pipeline should compile power");
    let obj_path = write_object_file(&dir, "power.o", &obj_bytes);

    let driver_src = r#"
#include <stdio.h>
extern long power(long base, long exp);
int main(void) {
    long r1 = power(2, 10);
    long r2 = power(3, 5);
    long r3 = power(5, 0);
    long r4 = power(7, 1);
    long r5 = power(1, 100);
    printf("power(2,10)=%ld power(3,5)=%ld power(5,0)=%ld power(7,1)=%ld power(1,100)=%ld\n",
           r1, r2, r3, r4, r5);
    if (r1 != 1024) return 1;
    if (r2 != 243) return 2;
    if (r3 != 1) return 3;
    if (r4 != 7) return 4;
    if (r5 != 1) return 5;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_full_power");
    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_full_pipeline_power stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "power full pipeline test failed with exit code {} \
         (1=power(2,10)!=1024, 2=power(3,5)!=243, 3=power(5,0)!=1, \
          4=power(7,1)!=7, 5=power(1,100)!=1). stdout: {}",
        exit_code, stdout
    );
    cleanup(&dir);
}

/// Golden truth: LLVM2 power vs clang power.
#[test]
fn test_full_pipeline_power_golden_truth() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("golden_power");

    let (tmir_func, module) = build_power_tmir();
    let obj_bytes = compile_tmir_function(&tmir_func, &module, OptLevel::O0)
        .expect("full pipeline should compile power");
    let llvm2_obj = write_object_file(&dir, "power_llvm2.o", &obj_bytes);

    let clang_src = r#"
long power_clang(long base, long exp) {
    long r = 1;
    for (long i = 0; i < exp; i++) r *= base;
    return r;
}
"#;
    let clang_c = write_c_driver(&dir, "power_clang.c", clang_src);
    let clang_obj = dir.join("power_clang.o");
    let cc_result = Command::new("cc")
        .args(["-c", "-o"])
        .arg(&clang_obj)
        .arg(&clang_c)
        .output()
        .expect("cc");
    assert!(cc_result.status.success(), "clang compilation failed: {}",
        String::from_utf8_lossy(&cc_result.stderr));

    let compare_src = r#"
#include <stdio.h>
extern long power(long base, long exp);
extern long power_clang(long base, long exp);
int main(void) {
    struct { long base; long exp; } tests[] = {
        {2, 10}, {3, 5}, {5, 0}, {7, 1}, {1, 100}, {2, 0}, {10, 3},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    for (int i = 0; i < n; i++) {
        long llvm2 = power(tests[i].base, tests[i].exp);
        long clang = power_clang(tests[i].base, tests[i].exp);
        printf("power(%ld,%ld): llvm2=%ld clang=%ld %s\n",
               tests[i].base, tests[i].exp, llvm2, clang,
               llvm2 == clang ? "MATCH" : "MISMATCH");
        if (llvm2 != clang) return i + 1;
    }
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "compare.c", compare_src);
    let binary = dir.join("test_golden_power");
    let link = Command::new("cc")
        .arg("-o").arg(&binary)
        .arg(&driver_path).arg(&llvm2_obj).arg(&clang_obj)
        .arg("-Wl,-no_pie")
        .output().expect("cc");
    assert!(link.status.success(), "golden truth link failed: {}",
        String::from_utf8_lossy(&link.stderr));

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_full_pipeline_power_golden_truth stdout: {}", stdout);
    assert_eq!(exit_code, 0,
        "power golden truth failed at test {} (1-indexed). stdout: {}",
        exit_code, stdout);
    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test: count_digits — division loop, counter pattern
// ---------------------------------------------------------------------------

/// Verifies that count_digits compiles through the full pipeline (encoding check).
#[test]
fn test_full_pipeline_count_digits_encoding() {
    let (tmir_func, module) = build_count_digits_tmir();

    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
        .expect("full pipeline should compile count_digits");

    assert!(obj_bytes.len() >= 4, "object file should not be empty");
    assert_eq!(
        &obj_bytes[0..4],
        &[0xCF, 0xFA, 0xED, 0xFE],
        "should be valid Mach-O magic"
    );
}

/// Full pipeline: count_digits compiled, linked, and executed.
#[test]
fn test_full_pipeline_count_digits_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("full_count_digits");
    let (tmir_func, module) = build_count_digits_tmir();
    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
        .expect("full pipeline should compile count_digits");
    let obj_path = write_object_file(&dir, "count_digits.o", &obj_bytes);

    let driver_src = r#"
#include <stdio.h>
extern long count_digits(long n);
int main(void) {
    long r1 = count_digits(0);
    long r2 = count_digits(5);
    long r3 = count_digits(99);
    long r4 = count_digits(12345);
    long r5 = count_digits(1000000);
    printf("count_digits(0)=%ld count_digits(5)=%ld count_digits(99)=%ld count_digits(12345)=%ld count_digits(1000000)=%ld\n",
           r1, r2, r3, r4, r5);
    if (r1 != 0) return 1;
    if (r2 != 1) return 2;
    if (r3 != 2) return 3;
    if (r4 != 5) return 4;
    if (r5 != 7) return 5;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_full_count_digits");
    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_full_pipeline_count_digits stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "count_digits full pipeline test failed with exit code {} \
         (1=count_digits(0)!=0, 2=count_digits(5)!=1, 3=count_digits(99)!=2, \
          4=count_digits(12345)!=5, 5=count_digits(1000000)!=7). stdout: {}",
        exit_code, stdout
    );
    cleanup(&dir);
}

/// Golden truth: LLVM2 count_digits vs clang count_digits.
#[test]
fn test_full_pipeline_count_digits_golden_truth() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("golden_count_digits");

    let (tmir_func, module) = build_count_digits_tmir();
    let obj_bytes = compile_tmir_function(&tmir_func, &module, OptLevel::O0)
        .expect("full pipeline should compile count_digits");
    let llvm2_obj = write_object_file(&dir, "count_digits_llvm2.o", &obj_bytes);

    let clang_src = r#"
long count_digits_clang(long n) {
    long count = 0;
    while (n > 0) { n /= 10; count += 1; }
    return count;
}
"#;
    let clang_c = write_c_driver(&dir, "count_digits_clang.c", clang_src);
    let clang_obj = dir.join("count_digits_clang.o");
    let cc_result = Command::new("cc")
        .args(["-c", "-o"])
        .arg(&clang_obj)
        .arg(&clang_c)
        .output()
        .expect("cc");
    assert!(cc_result.status.success(), "clang compilation failed: {}",
        String::from_utf8_lossy(&cc_result.stderr));

    let compare_src = r#"
#include <stdio.h>
extern long count_digits(long n);
extern long count_digits_clang(long n);
int main(void) {
    long inputs[] = {0, 1, 5, 99, 100, 12345, 999999, 1000000};
    int n = sizeof(inputs) / sizeof(inputs[0]);
    for (int i = 0; i < n; i++) {
        long llvm2 = count_digits(inputs[i]);
        long clang = count_digits_clang(inputs[i]);
        printf("count_digits(%ld): llvm2=%ld clang=%ld %s\n",
               inputs[i], llvm2, clang,
               llvm2 == clang ? "MATCH" : "MISMATCH");
        if (llvm2 != clang) return i + 1;
    }
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "compare.c", compare_src);
    let binary = dir.join("test_golden_count_digits");
    let link = Command::new("cc")
        .arg("-o").arg(&binary)
        .arg(&driver_path).arg(&llvm2_obj).arg(&clang_obj)
        .arg("-Wl,-no_pie")
        .output().expect("cc");
    assert!(link.status.success(), "golden truth link failed: {}",
        String::from_utf8_lossy(&link.stderr));

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_full_pipeline_count_digits_golden_truth stdout: {}", stdout);
    assert_eq!(exit_code, 0,
        "count_digits golden truth failed at test {} (1-indexed). stdout: {}",
        exit_code, stdout);
    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Test: nested_loop_sum — O(n^2) nested loops, high register pressure
// ---------------------------------------------------------------------------

/// Verifies that nested_loop_sum compiles through the full pipeline (encoding check).
#[test]
fn test_full_pipeline_nested_loop_sum_encoding() {
    let (tmir_func, module) = build_nested_loop_sum_tmir();

    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
        .expect("full pipeline should compile nested_loop_sum");

    assert!(obj_bytes.len() >= 4, "object file should not be empty");
    assert_eq!(
        &obj_bytes[0..4],
        &[0xCF, 0xFA, 0xED, 0xFE],
        "should be valid Mach-O magic"
    );
}

/// Full pipeline: nested_loop_sum compiled, linked, and executed.
/// This is the hardest test: TWO nested loops with Mul in the inner loop,
/// cross-block references, and O(n^2) iterations.
#[test]
fn test_full_pipeline_nested_loop_sum_run() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("full_nested_loop_sum");
    let (tmir_func, module) = build_nested_loop_sum_tmir();
    let obj_bytes = compile_tmir_function(
        &tmir_func,
        &module,
        OptLevel::O0)
        .expect("full pipeline should compile nested_loop_sum");
    let obj_path = write_object_file(&dir, "nested_loop_sum.o", &obj_bytes);

    let driver_src = r#"
#include <stdio.h>
extern long nested_loop_sum(long n);
int main(void) {
    long r1 = nested_loop_sum(0);
    long r2 = nested_loop_sum(1);
    long r3 = nested_loop_sum(3);
    long r4 = nested_loop_sum(5);
    long r5 = nested_loop_sum(10);
    printf("nested_loop_sum(0)=%ld nested_loop_sum(1)=%ld nested_loop_sum(3)=%ld nested_loop_sum(5)=%ld nested_loop_sum(10)=%ld\n",
           r1, r2, r3, r4, r5);
    if (r1 != 0) return 1;
    if (r2 != 0) return 2;
    if (r3 != 9) return 3;
    if (r4 != 100) return 4;
    if (r5 != 2025) return 5;
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "driver.c", driver_src);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "test_full_nested_loop_sum");
    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_full_pipeline_nested_loop_sum stdout: {}", stdout);
    assert_eq!(
        exit_code, 0,
        "nested_loop_sum full pipeline test failed with exit code {} \
         (1=nested_loop_sum(0)!=0, 2=nested_loop_sum(1)!=0, 3=nested_loop_sum(3)!=9, \
          4=nested_loop_sum(5)!=100, 5=nested_loop_sum(10)!=2025). stdout: {}",
        exit_code, stdout
    );
    cleanup(&dir);
}

/// Golden truth: LLVM2 nested_loop_sum vs clang nested_loop_sum.
#[test]
fn test_full_pipeline_nested_loop_sum_golden_truth() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping: not AArch64 or cc not available");
        return;
    }

    let dir = make_test_dir("golden_nested_loop_sum");

    let (tmir_func, module) = build_nested_loop_sum_tmir();
    let obj_bytes = compile_tmir_function(&tmir_func, &module, OptLevel::O0)
        .expect("full pipeline should compile nested_loop_sum");
    let llvm2_obj = write_object_file(&dir, "nested_loop_sum_llvm2.o", &obj_bytes);

    let clang_src = r#"
long nested_loop_sum_clang(long n) {
    long sum = 0;
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            sum += i * j;
        }
    }
    return sum;
}
"#;
    let clang_c = write_c_driver(&dir, "nested_loop_sum_clang.c", clang_src);
    let clang_obj = dir.join("nested_loop_sum_clang.o");
    let cc_result = Command::new("cc")
        .args(["-c", "-o"])
        .arg(&clang_obj)
        .arg(&clang_c)
        .output()
        .expect("cc");
    assert!(cc_result.status.success(), "clang compilation failed: {}",
        String::from_utf8_lossy(&cc_result.stderr));

    let compare_src = r#"
#include <stdio.h>
extern long nested_loop_sum(long n);
extern long nested_loop_sum_clang(long n);
int main(void) {
    long inputs[] = {0, 1, 2, 3, 5, 10, 20};
    int n = sizeof(inputs) / sizeof(inputs[0]);
    for (int i = 0; i < n; i++) {
        long llvm2 = nested_loop_sum(inputs[i]);
        long clang = nested_loop_sum_clang(inputs[i]);
        printf("nested_loop_sum(%ld): llvm2=%ld clang=%ld %s\n",
               inputs[i], llvm2, clang,
               llvm2 == clang ? "MATCH" : "MISMATCH");
        if (llvm2 != clang) return i + 1;
    }
    return 0;
}
"#;
    let driver_path = write_c_driver(&dir, "compare.c", compare_src);
    let binary = dir.join("test_golden_nested_loop_sum");
    let link = Command::new("cc")
        .arg("-o").arg(&binary)
        .arg(&driver_path).arg(&llvm2_obj).arg(&clang_obj)
        .arg("-Wl,-no_pie")
        .output().expect("cc");
    assert!(link.status.success(), "golden truth link failed: {}",
        String::from_utf8_lossy(&link.stderr));

    let (exit_code, stdout) = run_binary_with_output(&binary);
    eprintln!("test_full_pipeline_nested_loop_sum_golden_truth stdout: {}", stdout);
    assert_eq!(exit_code, 0,
        "nested_loop_sum golden truth failed at test {} (1-indexed). stdout: {}",
        exit_code, stdout);
    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Adapter structure tests for harder programs
// ---------------------------------------------------------------------------

/// Verify the adapter correctly translates power to LIR.
#[test]
fn test_adapter_power_structure() {
    let (tmir_func, module) = build_power_tmir();

    let (lir_func, _proof_ctx) =
        llvm2_lower::translate_function(&tmir_func, &module)
            .expect("adapter should translate power");

    assert_eq!(lir_func.name, "power");
    assert_eq!(
        lir_func.signature.params,
        vec![llvm2_lower::types::Type::I64, llvm2_lower::types::Type::I64]
    );
    assert_eq!(
        lir_func.signature.returns,
        vec![llvm2_lower::types::Type::I64]
    );
    // 4 blocks: entry, loop_init, loop, exit
    assert!(
        lir_func.blocks.len() >= 4,
        "power should have >= 4 blocks, got {}",
        lir_func.blocks.len()
    );
}

/// Verify the adapter correctly translates count_digits to LIR.
#[test]
fn test_adapter_count_digits_structure() {
    let (tmir_func, module) = build_count_digits_tmir();

    let (lir_func, _proof_ctx) =
        llvm2_lower::translate_function(&tmir_func, &module)
            .expect("adapter should translate count_digits");

    assert_eq!(lir_func.name, "count_digits");
    assert_eq!(
        lir_func.signature.params,
        vec![llvm2_lower::types::Type::I64]
    );
    // 5 blocks: entry, loop_init, loop, body, exit
    assert!(
        lir_func.blocks.len() >= 5,
        "count_digits should have >= 5 blocks, got {}",
        lir_func.blocks.len()
    );
}

/// Verify the adapter correctly translates nested_loop_sum to LIR.
#[test]
fn test_adapter_nested_loop_sum_structure() {
    let (tmir_func, module) = build_nested_loop_sum_tmir();

    let (lir_func, _proof_ctx) =
        llvm2_lower::translate_function(&tmir_func, &module)
            .expect("adapter should translate nested_loop_sum");

    assert_eq!(lir_func.name, "nested_loop_sum");
    assert_eq!(
        lir_func.signature.params,
        vec![llvm2_lower::types::Type::I64]
    );
    // 8 blocks: entry, outer_init, outer_check, inner_init, inner_check,
    //           inner_body, outer_next, exit
    assert!(
        lir_func.blocks.len() >= 8,
        "nested_loop_sum should have >= 8 blocks, got {}",
        lir_func.blocks.len()
    );
}
