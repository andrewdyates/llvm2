// llvm2-codegen/tests/e2e_differential.rs - Differential testing: LLVM2 vs clang
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// These tests compile the SAME function through BOTH the LLVM2 pipeline and
// the system C compiler (clang/cc), run both binaries, and assert their output
// is identical. Clang is treated as the golden reference: if LLVM2 produces
// different results, that is a bug in LLVM2.
//
// Architecture: AArch64 (Apple Silicon) only. Skipped on other targets.
//
// Part of #301 — Differential test harness: LLVM2 vs clang comparison.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use llvm2_codegen::compiler::{Compiler, CompilerConfig};
use llvm2_codegen::pipeline::OptLevel;

// tMIR imports
use tmir::{Block as TmirBlock, Function as TmirFunction, Module as TmirModule, FuncTy, Ty, Constant};
use tmir::{Inst, InstrNode, BinOp, ICmpOp, CastOp, FCmpOp, UnOp};
use tmir::{BlockId, FuncId, ValueId};
use tmir::SwitchCase;

// =============================================================================
// Test infrastructure
// =============================================================================

/// Returns true if we are running on AArch64 (Apple Silicon).
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
fn make_test_dir(test_name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("llvm2_differential_{}", test_name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("Failed to create test directory");
    dir
}

/// Cleanup a test directory.
fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

/// Compile a tMIR module through the LLVM2 pipeline, returning raw .o bytes.
fn compile_tmir_module(module: &TmirModule) -> Vec<u8> {
    let compiler = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        ..CompilerConfig::default()
    });
    let result = compiler
        .compile(module)
        .expect("LLVM2 compilation should succeed");
    assert!(
        !result.object_code.is_empty(),
        "LLVM2 must produce non-empty object code"
    );
    result.object_code
}

// =============================================================================
// Core differential test harness
// =============================================================================

/// Run a differential test: compile the same function through LLVM2 and clang,
/// link each with a common test driver, run both, and assert stdout matches.
///
/// # Arguments
/// * `test_name` - Unique name for this test (used for temp directory)
/// * `tmir_module` - tMIR module to compile through LLVM2
/// * `c_reference` - Equivalent C source code implementing the same function(s)
/// * `driver_src` - C driver (main) that calls the function and prints results
///
/// The C reference source must define functions with the SAME names as the tMIR
/// module's functions (with leading underscore, e.g., `int _add_two(int a, int b)`).
/// This way both the LLVM2 .o and the clang .o export the same symbol, and the
/// same driver can link against either one.
fn differential_test(
    test_name: &str,
    tmir_module: &TmirModule,
    c_reference: &str,
    driver_src: &str,
) -> Result<(), String> {
    let dir = make_test_dir(test_name);

    // --- Step 1: LLVM2 path ---
    // Compile tMIR -> .o via LLVM2 pipeline
    let llvm2_obj_bytes = compile_tmir_module(tmir_module);
    let llvm2_obj_path = dir.join("llvm2_func.o");
    fs::write(&llvm2_obj_path, &llvm2_obj_bytes)
        .map_err(|e| format!("write llvm2 .o: {}", e))?;

    // --- Step 2: Clang path ---
    // Write reference C source and compile with cc -c
    let ref_c_path = dir.join("reference.c");
    fs::write(&ref_c_path, c_reference)
        .map_err(|e| format!("write reference.c: {}", e))?;

    let clang_obj_path = dir.join("clang_func.o");
    let cc_compile = Command::new("cc")
        .args([
            "-c",
            "-O0",
            "-o",
            clang_obj_path.to_str().unwrap(),
            ref_c_path.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| format!("cc -c failed to run: {}", e))?;

    if !cc_compile.status.success() {
        let stderr = String::from_utf8_lossy(&cc_compile.stderr);
        return Err(format!("cc -c reference.c failed: {}", stderr));
    }

    // --- Step 3: Write the shared test driver ---
    let driver_path = dir.join("driver.c");
    fs::write(&driver_path, driver_src)
        .map_err(|e| format!("write driver.c: {}", e))?;

    // --- Step 4: Link and run LLVM2 version ---
    let llvm2_binary = dir.join("test_llvm2");
    let llvm2_link = Command::new("cc")
        .args([
            "-o",
            llvm2_binary.to_str().unwrap(),
            driver_path.to_str().unwrap(),
            llvm2_obj_path.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| format!("link llvm2 binary: {}", e))?;

    if !llvm2_link.status.success() {
        let stderr = String::from_utf8_lossy(&llvm2_link.stderr);
        // Debug: show symbols and disassembly
        let nm = Command::new("nm")
            .args([llvm2_obj_path.to_str().unwrap()])
            .output()
            .ok();
        let nm_out = nm
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();
        let otool = Command::new("otool")
            .args(["-tv", llvm2_obj_path.to_str().unwrap()])
            .output()
            .ok();
        let disasm = otool
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();
        return Err(format!(
            "Linking LLVM2 binary failed!\nstderr: {}\nnm:\n{}\notool:\n{}",
            stderr, nm_out, disasm
        ));
    }

    let llvm2_run = Command::new(&llvm2_binary)
        .output()
        .map_err(|e| format!("run llvm2 binary: {}", e))?;
    let llvm2_stdout = String::from_utf8_lossy(&llvm2_run.stdout).to_string();
    let llvm2_exit = llvm2_run.status.code().unwrap_or(-1);

    // --- Step 5: Link and run clang version ---
    let clang_binary = dir.join("test_clang");
    let clang_link = Command::new("cc")
        .args([
            "-o",
            clang_binary.to_str().unwrap(),
            driver_path.to_str().unwrap(),
            clang_obj_path.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| format!("link clang binary: {}", e))?;

    if !clang_link.status.success() {
        let stderr = String::from_utf8_lossy(&clang_link.stderr);
        return Err(format!("Linking clang binary failed: {}", stderr));
    }

    let clang_run = Command::new(&clang_binary)
        .output()
        .map_err(|e| format!("run clang binary: {}", e))?;
    let clang_stdout = String::from_utf8_lossy(&clang_run.stdout).to_string();
    let clang_exit = clang_run.status.code().unwrap_or(-1);

    // --- Step 6: Compare ---
    eprintln!("=== Differential test: {} ===", test_name);
    eprintln!("  LLVM2 stdout: {}", llvm2_stdout.trim());
    eprintln!("  Clang stdout: {}", clang_stdout.trim());
    eprintln!("  LLVM2 exit:   {}", llvm2_exit);
    eprintln!("  Clang exit:   {}", clang_exit);

    if llvm2_stdout != clang_stdout {
        // Debug: show disassembly of LLVM2 object
        let otool = Command::new("otool")
            .args(["-tv", llvm2_obj_path.to_str().unwrap()])
            .output()
            .ok();
        let disasm = otool
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();
        eprintln!("  LLVM2 disassembly:\n{}", disasm);
        cleanup(&dir);
        return Err(format!(
            "OUTPUT MISMATCH!\n  LLVM2: {}\n  Clang: {}",
            llvm2_stdout.trim(),
            clang_stdout.trim()
        ));
    }

    if llvm2_exit != clang_exit {
        cleanup(&dir);
        return Err(format!(
            "EXIT CODE MISMATCH!\n  LLVM2: {}\n  Clang: {}",
            llvm2_exit, clang_exit
        ));
    }

    // Verify both actually succeeded (not just matching failures).
    if clang_exit != 0 {
        cleanup(&dir);
        return Err(format!(
            "Both binaries exited with non-zero code {}. \
             The C reference itself has a bug or the driver is wrong.",
            clang_exit
        ));
    }

    cleanup(&dir);
    Ok(())
}

// =============================================================================
// tMIR module builders (duplicated from e2e_aarch64_link.rs — these are
// integration tests in separate binaries, so they cannot share code)
// =============================================================================

/// `fn _add_two(a: i32, b: i32) -> i32 { a + b }`
fn build_tmir_add_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32, Ty::I32],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_add_two", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I32),
            (ValueId::new(1), Ty::I32),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

/// `fn _max_val(a: i64, b: i64) -> i64 { if a > b { a } else { b } }`
fn build_tmir_max_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_max_val", ft_id, BlockId::new(0));
    func.blocks = vec![
        TmirBlock {
            id: BlockId::new(0),
            params: vec![
                (ValueId::new(0), Ty::I64),
                (ValueId::new(1), Ty::I64),
            ],
            body: vec![
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sgt,
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
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(1)],
            })],
        },
    ];
    module.add_function(func);
    module
}

/// `fn _abs_val(x: i64) -> i64 { if x < 0 { 0 - x } else { x } }`
fn build_tmir_abs_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_abs_val", ft_id, BlockId::new(0));
    func.blocks = vec![
        TmirBlock {
            id: BlockId::new(0),
            params: vec![(ValueId::new(0), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(1)),
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Slt,
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
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Sub,
                    ty: Ty::I64,
                    lhs: ValueId::new(1),
                    rhs: ValueId::new(0),
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
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(0)],
            })],
        },
    ];
    module.add_function(func);
    module
}

/// `fn _factorial(n: i64) -> i64 { iterative factorial }`
///
/// bb0: cmp n <= 1, condbr -> bb1 (ret 1), bb2 (loop init)
/// bb1: return 1
/// bb2: acc=1, i=2, br -> bb3
/// bb3 (loop): params(acc, i), cmp i <= n, condbr -> bb4 (body), bb5 (exit)
/// bb4: new_acc = acc * i, new_i = i + 1, br -> bb3
/// bb5: return acc
fn build_tmir_factorial_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_factorial", ft_id, BlockId::new(0));
    func.blocks = vec![
        // bb0: check n <= 1
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
        // bb1: return 1
        TmirBlock {
            id: BlockId::new(1),
            params: vec![],
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(1)],
            })],
        },
        // bb2: loop init
        TmirBlock {
            id: BlockId::new(2),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(10)),
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(2),
                })
                .with_result(ValueId::new(11)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(3),
                    args: vec![ValueId::new(10), ValueId::new(11)],
                }),
            ],
        },
        // bb3: loop header
        TmirBlock {
            id: BlockId::new(3),
            params: vec![
                (ValueId::new(20), Ty::I64), // acc
                (ValueId::new(21), Ty::I64), // i
            ],
            body: vec![
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sle,
                    ty: Ty::I64,
                    lhs: ValueId::new(21),
                    rhs: ValueId::new(0),
                })
                .with_result(ValueId::new(22)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(22),
                    then_target: BlockId::new(4),
                    then_args: vec![],
                    else_target: BlockId::new(5),
                    else_args: vec![],
                }),
            ],
        },
        // bb4: loop body
        TmirBlock {
            id: BlockId::new(4),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Mul,
                    ty: Ty::I64,
                    lhs: ValueId::new(20),
                    rhs: ValueId::new(21),
                })
                .with_result(ValueId::new(30)),
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(31)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(21),
                    rhs: ValueId::new(31),
                })
                .with_result(ValueId::new(32)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(3),
                    args: vec![ValueId::new(30), ValueId::new(32)],
                }),
            ],
        },
        // bb5: exit
        TmirBlock {
            id: BlockId::new(5),
            params: vec![],
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(20)],
            })],
        },
    ];
    module.add_function(func);
    module
}

/// `fn _sum_1_to_n(n: i64) -> i64 { sum from 1 to n }`
///
/// bb0: sum=0, i=1, br -> bb1
/// bb1 (loop): params(sum, i), cmp i <= n, condbr -> bb2 (body), bb3 (exit)
/// bb2: new_sum = sum + i, new_i = i + 1, br -> bb1
/// bb3: return sum
fn build_tmir_sum_1_to_n_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_sum_1_to_n", ft_id, BlockId::new(0));
    func.blocks = vec![
        // bb0: init
        TmirBlock {
            id: BlockId::new(0),
            params: vec![(ValueId::new(0), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(1)),
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(2)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(1),
                    args: vec![ValueId::new(1), ValueId::new(2)],
                }),
            ],
        },
        // bb1: loop header
        TmirBlock {
            id: BlockId::new(1),
            params: vec![
                (ValueId::new(10), Ty::I64), // sum
                (ValueId::new(11), Ty::I64), // i
            ],
            body: vec![
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sle,
                    ty: Ty::I64,
                    lhs: ValueId::new(11),
                    rhs: ValueId::new(0),
                })
                .with_result(ValueId::new(12)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(12),
                    then_target: BlockId::new(2),
                    then_args: vec![],
                    else_target: BlockId::new(3),
                    else_args: vec![],
                }),
            ],
        },
        // bb2: loop body
        TmirBlock {
            id: BlockId::new(2),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(10),
                    rhs: ValueId::new(11),
                })
                .with_result(ValueId::new(20)),
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
                    args: vec![ValueId::new(20), ValueId::new(22)],
                }),
            ],
        },
        // bb3: exit
        TmirBlock {
            id: BlockId::new(3),
            params: vec![],
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(10)],
            })],
        },
    ];
    module.add_function(func);
    module
}

// =============================================================================
// Test 1: add(a, b) -> a + b
//
// The simplest differential test. Both LLVM2 and clang must agree on addition.
// =============================================================================

#[test]
fn test_differential_add() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_add_module();

    let c_reference = r#"
int _add_two(int a, int b) {
    return a + b;
}
"#;

    let driver = r#"
#include <stdio.h>
extern int _add_two(int a, int b);
int main(void) {
    printf("add(3,4)=%d\n", _add_two(3, 4));
    printf("add(100,-50)=%d\n", _add_two(100, -50));
    printf("add(0,0)=%d\n", _add_two(0, 0));
    printf("add(-1,-1)=%d\n", _add_two(-1, -1));
    printf("add(2147483647,0)=%d\n", _add_two(2147483647, 0));
    return 0;
}
"#;

    let result = differential_test("add", &module, c_reference, driver);
    assert!(result.is_ok(), "Differential add test failed: {}", result.unwrap_err());
}

// =============================================================================
// Test 2: max(a, b) -> if a > b { a } else { b }
//
// Tests conditional branching. Both compilers must agree on all sign cases.
// =============================================================================

#[test]
fn test_differential_max() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_max_module();

    let c_reference = r#"
long _max_val(long a, long b) {
    return (a > b) ? a : b;
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _max_val(long a, long b);
int main(void) {
    printf("max(10,20)=%ld\n", _max_val(10, 20));
    printf("max(20,10)=%ld\n", _max_val(20, 10));
    printf("max(5,5)=%ld\n", _max_val(5, 5));
    printf("max(-3,-7)=%ld\n", _max_val(-3, -7));
    printf("max(-1,1)=%ld\n", _max_val(-1, 1));
    printf("max(0,0)=%ld\n", _max_val(0, 0));
    printf("max(-1000000,1000000)=%ld\n", _max_val(-1000000, 1000000));
    return 0;
}
"#;

    let result = differential_test("max", &module, c_reference, driver);
    assert!(result.is_ok(), "Differential max test failed: {}", result.unwrap_err());
}

// =============================================================================
// Test 3: abs(x) -> if x < 0 { -x } else { x }
//
// Tests comparison + conditional negate. Edge cases: 0, -1, 1.
// =============================================================================

#[test]
fn test_differential_abs() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_abs_module();

    let c_reference = r#"
long _abs_val(long x) {
    return (x < 0) ? (0 - x) : x;
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _abs_val(long x);
int main(void) {
    printf("abs(42)=%ld\n", _abs_val(42));
    printf("abs(-42)=%ld\n", _abs_val(-42));
    printf("abs(0)=%ld\n", _abs_val(0));
    printf("abs(-1)=%ld\n", _abs_val(-1));
    printf("abs(1)=%ld\n", _abs_val(1));
    printf("abs(-9999)=%ld\n", _abs_val(-9999));
    return 0;
}
"#;

    let result = differential_test("abs", &module, c_reference, driver);
    assert!(result.is_ok(), "Differential abs test failed: {}", result.unwrap_err());
}

// =============================================================================
// Test 4: factorial(n) -> iterative factorial with multiply loop
//
// Tests loop with accumulator and multiplication. This exercises control flow,
// block parameters, and the MUL (MADD) encoding.
// =============================================================================

#[test]
fn test_differential_factorial() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_factorial_module();

    let c_reference = r#"
long _factorial(long n) {
    if (n <= 1) return 1;
    long acc = 1;
    for (long i = 2; i <= n; i++) {
        acc *= i;
    }
    return acc;
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _factorial(long n);
int main(void) {
    printf("fact(0)=%ld\n", _factorial(0));
    printf("fact(1)=%ld\n", _factorial(1));
    printf("fact(5)=%ld\n", _factorial(5));
    printf("fact(10)=%ld\n", _factorial(10));
    printf("fact(12)=%ld\n", _factorial(12));
    printf("fact(20)=%ld\n", _factorial(20));
    return 0;
}
"#;

    let result = differential_test("factorial", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential factorial test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 5: sum_1_to_n(n) -> sum of 1..=n
//
// Tests a counting loop with addition. Verifiable via n*(n+1)/2 formula.
// =============================================================================

#[test]
fn test_differential_sum_1_to_n() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_sum_1_to_n_module();

    let c_reference = r#"
long _sum_1_to_n(long n) {
    long sum = 0;
    for (long i = 1; i <= n; i++) {
        sum += i;
    }
    return sum;
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _sum_1_to_n(long n);
int main(void) {
    printf("sum(0)=%ld\n", _sum_1_to_n(0));
    printf("sum(1)=%ld\n", _sum_1_to_n(1));
    printf("sum(5)=%ld\n", _sum_1_to_n(5));
    printf("sum(10)=%ld\n", _sum_1_to_n(10));
    printf("sum(100)=%ld\n", _sum_1_to_n(100));
    printf("sum(1000)=%ld\n", _sum_1_to_n(1000));
    return 0;
}
"#;

    let result = differential_test("sum_1_to_n", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential sum_1_to_n test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 6: bitwise_and(a, b) -> a & b
//
// Tests bitwise AND across zero, one, sign-bit, and pattern cases.
// =============================================================================

/// `fn _bitwise_and(a: i32, b: i32) -> i32 { a & b }`
fn build_tmir_bitwise_and_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32, Ty::I32],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_bitwise_and", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I32),
            (ValueId::new(1), Ty::I32),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_bitwise_and() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_bitwise_and_module();

    let c_reference = r#"
int _bitwise_and(int a, int b) {
    return a & b;
}
"#;

    let driver = r#"
#include <stdio.h>
extern int _bitwise_and(int a, int b);
int main(void) {
    printf("and(0,0)=%d\n", _bitwise_and(0, 0));
    printf("and(1,1)=%d\n", _bitwise_and(1, 1));
    printf("and(-1,0)=%d\n", _bitwise_and(-1, 0));
    printf("and(-1,1)=%d\n", _bitwise_and(-1, 1));
    printf("and(2147483647,1)=%d\n", _bitwise_and(2147483647, 1));
    printf("and(-2147483648,2147483647)=%d\n", _bitwise_and((-2147483647 - 1), 2147483647));
    printf("and(0x55555555,0x33333333)=%d\n", _bitwise_and(0x55555555, 0x33333333));
    printf("and(-123456789,123456789)=%d\n", _bitwise_and(-123456789, 123456789));
    return 0;
}
"#;

    let result = differential_test("bitwise_and", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential bitwise_and test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 7: bitwise_or(a, b) -> a | b
//
// Tests bitwise OR across zero, one, sign-bit, and pattern cases.
// =============================================================================

/// `fn _bitwise_or(a: i32, b: i32) -> i32 { a | b }`
fn build_tmir_bitwise_or_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32, Ty::I32],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_bitwise_or", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I32),
            (ValueId::new(1), Ty::I32),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::Or,
                ty: Ty::I32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_bitwise_or() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_bitwise_or_module();

    let c_reference = r#"
int _bitwise_or(int a, int b) {
    return a | b;
}
"#;

    let driver = r#"
#include <stdio.h>
extern int _bitwise_or(int a, int b);
int main(void) {
    printf("or(0,0)=%d\n", _bitwise_or(0, 0));
    printf("or(1,0)=%d\n", _bitwise_or(1, 0));
    printf("or(1,1)=%d\n", _bitwise_or(1, 1));
    printf("or(-1,0)=%d\n", _bitwise_or(-1, 0));
    printf("or(-1,1)=%d\n", _bitwise_or(-1, 1));
    printf("or(2147483647,0)=%d\n", _bitwise_or(2147483647, 0));
    printf("or(-2147483648,1)=%d\n", _bitwise_or((-2147483647 - 1), 1));
    printf("or(0x55555555,0x33333333)=%d\n", _bitwise_or(0x55555555, 0x33333333));
    printf("or(-123456789,123456789)=%d\n", _bitwise_or(-123456789, 123456789));
    return 0;
}
"#;

    let result = differential_test("bitwise_or", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential bitwise_or test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 8: bitwise_xor(a, b) -> a ^ b
//
// Tests bitwise XOR across zero, one, sign-bit, and pattern cases.
// =============================================================================

/// `fn _bitwise_xor(a: i32, b: i32) -> i32 { a ^ b }`
fn build_tmir_bitwise_xor_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32, Ty::I32],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_bitwise_xor", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I32),
            (ValueId::new(1), Ty::I32),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::Xor,
                ty: Ty::I32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_bitwise_xor() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_bitwise_xor_module();

    let c_reference = r#"
int _bitwise_xor(int a, int b) {
    return a ^ b;
}
"#;

    let driver = r#"
#include <stdio.h>
extern int _bitwise_xor(int a, int b);
int main(void) {
    printf("xor(0,0)=%d\n", _bitwise_xor(0, 0));
    printf("xor(1,0)=%d\n", _bitwise_xor(1, 0));
    printf("xor(1,1)=%d\n", _bitwise_xor(1, 1));
    printf("xor(-1,0)=%d\n", _bitwise_xor(-1, 0));
    printf("xor(-1,-1)=%d\n", _bitwise_xor(-1, -1));
    printf("xor(2147483647,2147483647)=%d\n", _bitwise_xor(2147483647, 2147483647));
    printf("xor(-2147483648,2147483647)=%d\n", _bitwise_xor((-2147483647 - 1), 2147483647));
    printf("xor(0x55555555,0x33333333)=%d\n", _bitwise_xor(0x55555555, 0x33333333));
    printf("xor(-123456789,123456789)=%d\n", _bitwise_xor(-123456789, 123456789));
    return 0;
}
"#;

    let result = differential_test("bitwise_xor", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential bitwise_xor test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 9: urem(a, b) -> a % b (unsigned semantics)
//
// Tests unsigned remainder on zero, powers of two, and max-width values.
// =============================================================================

/// `fn _urem(a: i32, b: i32) -> i32 { a %u b }`
fn build_tmir_urem_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32, Ty::I32],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_urem", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I32),
            (ValueId::new(1), Ty::I32),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::URem,
                ty: Ty::I32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_urem() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_urem_module();

    let c_reference = r#"
unsigned int _urem(unsigned int a, unsigned int b) {
    return a % b;
}
"#;

    let driver = r#"
#include <stdio.h>
extern unsigned int _urem(unsigned int a, unsigned int b);
int main(void) {
    printf("urem(0,1)=%u\n", _urem(0u, 1u));
    printf("urem(1,1)=%u\n", _urem(1u, 1u));
    printf("urem(10,3)=%u\n", _urem(10u, 3u));
    printf("urem(16,8)=%u\n", _urem(16u, 8u));
    printf("urem(2147483648,3)=%u\n", _urem(2147483648u, 3u));
    printf("urem(2147483648,1024)=%u\n", _urem(2147483648u, 1024u));
    printf("urem(4294967295,2)=%u\n", _urem(4294967295u, 2u));
    printf("urem(4294967295,2147483647)=%u\n", _urem(4294967295u, 2147483647u));
    return 0;
}
"#;

    let result = differential_test("urem", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential urem test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 10: srem(a, b) -> a % b
//
// Tests signed remainder across positive and negative dividend/divisor cases.
// =============================================================================

/// `fn _srem(a: i32, b: i32) -> i32 { a % b }`
fn build_tmir_srem_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32, Ty::I32],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_srem", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I32),
            (ValueId::new(1), Ty::I32),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::SRem,
                ty: Ty::I32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_srem() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_srem_module();

    let c_reference = r#"
int _srem(int a, int b) {
    return a % b;
}
"#;

    let driver = r#"
#include <stdio.h>
extern int _srem(int a, int b);
int main(void) {
    printf("srem(0,1)=%d\n", _srem(0, 1));
    printf("srem(1,1)=%d\n", _srem(1, 1));
    printf("srem(-1,2)=%d\n", _srem(-1, 2));
    printf("srem(17,5)=%d\n", _srem(17, 5));
    printf("srem(17,-5)=%d\n", _srem(17, -5));
    printf("srem(-17,5)=%d\n", _srem(-17, 5));
    printf("srem(-17,-5)=%d\n", _srem(-17, -5));
    printf("srem(2147483647,2)=%d\n", _srem(2147483647, 2));
    printf("srem(-2147483648,2)=%d\n", _srem((-2147483647 - 1), 2));
    printf("srem(-2147483648,2147483647)=%d\n", _srem((-2147483647 - 1), 2147483647));
    return 0;
}
"#;

    let result = differential_test("srem", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential srem test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 11: shl(a, b) -> a << (b & 31)
//
// Tests left shift with masked shift counts and only defined-value inputs.
// =============================================================================

/// `fn _shl(a: i32, b: i32) -> i32 { a << (b & 31) }`
fn build_tmir_shl_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32, Ty::I32],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_shl", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I32),
            (ValueId::new(1), Ty::I32),
        ],
        body: vec![
            InstrNode::new(Inst::Const {
                ty: Ty::I32,
                value: Constant::Int(31),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I32,
                lhs: ValueId::new(1),
                rhs: ValueId::new(2),
            })
            .with_result(ValueId::new(3)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Shl,
                ty: Ty::I32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(3),
            })
            .with_result(ValueId::new(4)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(4)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_shl() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_shl_module();

    let c_reference = r#"
int _shl(int a, int b) {
    unsigned int amt = ((unsigned int)b) & 31u;
    return a << amt;
}
"#;

    let driver = r#"
#include <stdio.h>
extern int _shl(int a, int b);
int main(void) {
    printf("shl(0,0)=%d\n", _shl(0, 0));
    printf("shl(1,0)=%d\n", _shl(1, 0));
    printf("shl(1,1)=%d\n", _shl(1, 1));
    printf("shl(1,30)=%d\n", _shl(1, 30));
    printf("shl(1024,5)=%d\n", _shl(1024, 5));
    printf("shl(1073741824,0)=%d\n", _shl(1073741824, 0));
    printf("shl(1,32)=%d\n", _shl(1, 32));
    printf("shl(2,33)=%d\n", _shl(2, 33));
    printf("shl(0,-1)=%d\n", _shl(0, -1));
    printf("shl(1,-2)=%d\n", _shl(1, -2));
    return 0;
}
"#;

    let result = differential_test("shl", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential shl test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 12: lshr(a, b) -> a >> (b & 31) with unsigned semantics
//
// Tests logical right shift using unsigned C values and masked shift counts.
// =============================================================================

/// `fn _lshr(a: i32, b: i32) -> i32 { ((unsigned)a) >> (b & 31) }`
fn build_tmir_lshr_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32, Ty::I32],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_lshr", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I32),
            (ValueId::new(1), Ty::I32),
        ],
        body: vec![
            InstrNode::new(Inst::Const {
                ty: Ty::I32,
                value: Constant::Int(31),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I32,
                lhs: ValueId::new(1),
                rhs: ValueId::new(2),
            })
            .with_result(ValueId::new(3)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::LShr,
                ty: Ty::I32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(3),
            })
            .with_result(ValueId::new(4)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(4)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_lshr() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_lshr_module();

    let c_reference = r#"
unsigned int _lshr(unsigned int a, int b) {
    unsigned int amt = ((unsigned int)b) & 31u;
    return a >> amt;
}
"#;

    let driver = r#"
#include <stdio.h>
extern unsigned int _lshr(unsigned int a, int b);
int main(void) {
    printf("lshr(0,0)=%u\n", _lshr(0u, 0));
    printf("lshr(1,0)=%u\n", _lshr(1u, 0));
    printf("lshr(1,1)=%u\n", _lshr(1u, 1));
    printf("lshr(4294967295,1)=%u\n", _lshr(4294967295u, 1));
    printf("lshr(4294967295,31)=%u\n", _lshr(4294967295u, 31));
    printf("lshr(2147483648,31)=%u\n", _lshr(2147483648u, 31));
    printf("lshr(2147483648,32)=%u\n", _lshr(2147483648u, 32));
    printf("lshr(2147483648,33)=%u\n", _lshr(2147483648u, 33));
    printf("lshr(4294967295,-1)=%u\n", _lshr(4294967295u, -1));
    printf("lshr(2147483647,-32)=%u\n", _lshr(2147483647u, -32));
    return 0;
}
"#;

    let result = differential_test("lshr", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential lshr test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 13: ashr(a, b) -> a >> (b & 31) with signed semantics
//
// Tests arithmetic right shift across positive, negative, and masked counts.
// =============================================================================

/// `fn _ashr(a: i32, b: i32) -> i32 { a >> (b & 31) }`
fn build_tmir_ashr_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32, Ty::I32],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_ashr", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I32),
            (ValueId::new(1), Ty::I32),
        ],
        body: vec![
            InstrNode::new(Inst::Const {
                ty: Ty::I32,
                value: Constant::Int(31),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I32,
                lhs: ValueId::new(1),
                rhs: ValueId::new(2),
            })
            .with_result(ValueId::new(3)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::AShr,
                ty: Ty::I32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(3),
            })
            .with_result(ValueId::new(4)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(4)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_ashr() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_ashr_module();

    let c_reference = r#"
int _ashr(int a, int b) {
    unsigned int amt = ((unsigned int)b) & 31u;
    return a >> amt;
}
"#;

    let driver = r#"
#include <stdio.h>
extern int _ashr(int a, int b);
int main(void) {
    printf("ashr(0,0)=%d\n", _ashr(0, 0));
    printf("ashr(1,1)=%d\n", _ashr(1, 1));
    printf("ashr(-1,1)=%d\n", _ashr(-1, 1));
    printf("ashr(-2147483648,1)=%d\n", _ashr((-2147483647 - 1), 1));
    printf("ashr(-2147483648,31)=%d\n", _ashr((-2147483647 - 1), 31));
    printf("ashr(2147483647,31)=%d\n", _ashr(2147483647, 31));
    printf("ashr(-123456789,4)=%d\n", _ashr(-123456789, 4));
    printf("ashr(1024,32)=%d\n", _ashr(1024, 32));
    printf("ashr(-1,-1)=%d\n", _ashr(-1, -1));
    printf("ashr(-2147483648,33)=%d\n", _ashr((-2147483647 - 1), 33));
    return 0;
}
"#;

    let result = differential_test("ashr", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential ashr test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 14: sext_i32_to_i64(x) -> sign-extend i32 to i64
//
// Tests sign extension across zero, one, negative one, and i32 extremes.
// =============================================================================

/// `fn _sext_i32_to_i64(x: i32) -> i64 { x as i64 }`
fn build_tmir_sext_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_sext_i32_to_i64", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::I32)],
        body: vec![
            InstrNode::new(Inst::Cast {
                op: CastOp::SExt,
                src_ty: Ty::I32,
                dst_ty: Ty::I64,
                operand: ValueId::new(0),
            })
            .with_result(ValueId::new(1)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(1)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_sext_i32_to_i64() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_sext_module();

    let c_reference = r#"
long _sext_i32_to_i64(int a) {
    return (long)a;
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _sext_i32_to_i64(int a);
int main(void) {
    printf("sext(0)=%ld\n", _sext_i32_to_i64(0));
    printf("sext(1)=%ld\n", _sext_i32_to_i64(1));
    printf("sext(-1)=%ld\n", _sext_i32_to_i64(-1));
    printf("sext(2147483647)=%ld\n", _sext_i32_to_i64(2147483647));
    printf("sext(-2147483648)=%ld\n", _sext_i32_to_i64((-2147483647 - 1)));
    printf("sext(123456789)=%ld\n", _sext_i32_to_i64(123456789));
    printf("sext(-123456789)=%ld\n", _sext_i32_to_i64(-123456789));
    printf("sext(1073741824)=%ld\n", _sext_i32_to_i64(1073741824));
    return 0;
}
"#;

    let result = differential_test("sext_i32_to_i64", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential sext_i32_to_i64 test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 15: trunc_i64_to_i32(x) -> truncate i64 to i32
//
// Tests truncation at in-range values and around 32-bit wrap boundaries.
// =============================================================================

/// `fn _trunc_i64_to_i32(x: i64) -> i32 { x as i32 }`
fn build_tmir_trunc_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_trunc_i64_to_i32", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::I64)],
        body: vec![
            InstrNode::new(Inst::Cast {
                op: CastOp::Trunc,
                src_ty: Ty::I64,
                dst_ty: Ty::I32,
                operand: ValueId::new(0),
            })
            .with_result(ValueId::new(1)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(1)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_trunc_i64_to_i32() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_trunc_module();

    let c_reference = r#"
int _trunc_i64_to_i32(long a) {
    return (int)a;
}
"#;

    let driver = r#"
#include <stdio.h>
extern int _trunc_i64_to_i32(long a);
int main(void) {
    printf("trunc(0)=%d\n", _trunc_i64_to_i32(0L));
    printf("trunc(1)=%d\n", _trunc_i64_to_i32(1L));
    printf("trunc(-1)=%d\n", _trunc_i64_to_i32(-1L));
    printf("trunc(2147483647)=%d\n", _trunc_i64_to_i32(2147483647L));
    printf("trunc(-2147483648)=%d\n", _trunc_i64_to_i32((-2147483647L - 1L)));
    printf("trunc(2147483648)=%d\n", _trunc_i64_to_i32(2147483648L));
    printf("trunc(-2147483649)=%d\n", _trunc_i64_to_i32(-2147483649L));
    printf("trunc(4294967296)=%d\n", _trunc_i64_to_i32(4294967296L));
    printf("trunc(4294967297)=%d\n", _trunc_i64_to_i32(4294967297L));
    printf("trunc(-4294967297)=%d\n", _trunc_i64_to_i32(-4294967297L));
    return 0;
}
"#;

    let result = differential_test("trunc_i64_to_i32", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential trunc_i64_to_i32 test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 16: combined_bitops(a, b) -> (a & b) | ((a ^ b) << 1)
//
// Tests a combined expression that exercises AND, OR, XOR, and SHL together.
// =============================================================================

/// `fn _combined_bitops(a: i32, b: i32) -> i32 { (a & b) | ((a ^ b) << 1) }`
fn build_tmir_combined_bitops_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32, Ty::I32],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_combined_bitops", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I32),
            (ValueId::new(1), Ty::I32),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Xor,
                ty: Ty::I32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(3)),
            InstrNode::new(Inst::Const {
                ty: Ty::I32,
                value: Constant::Int(1),
            })
            .with_result(ValueId::new(4)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Shl,
                ty: Ty::I32,
                lhs: ValueId::new(3),
                rhs: ValueId::new(4),
            })
            .with_result(ValueId::new(5)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Or,
                ty: Ty::I32,
                lhs: ValueId::new(2),
                rhs: ValueId::new(5),
            })
            .with_result(ValueId::new(6)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(6)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_combined_bitops() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_combined_bitops_module();

    let c_reference = r#"
int _combined_bitops(int a, int b) {
    return (a & b) | ((a ^ b) << 1);
}
"#;

    let driver = r#"
#include <stdio.h>
extern int _combined_bitops(int a, int b);
int main(void) {
    printf("combined(0,0)=%d\n", _combined_bitops(0, 0));
    printf("combined(1,1)=%d\n", _combined_bitops(1, 1));
    printf("combined(1,2)=%d\n", _combined_bitops(1, 2));
    printf("combined(3,5)=%d\n", _combined_bitops(3, 5));
    printf("combined(0x0f0f0f0f,0x33333333)=%d\n", _combined_bitops(0x0f0f0f0f, 0x33333333));
    printf("combined(536870912,268435456)=%d\n", _combined_bitops(536870912, 268435456));
    printf("combined(1073741823,0)=%d\n", _combined_bitops(1073741823, 0));
    printf("combined(0,1073741823)=%d\n", _combined_bitops(0, 1073741823));
    return 0;
}
"#;

    let result = differential_test("combined_bitops", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential combined_bitops test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 17: fp_add(a, b) -> a + b
//
// Tests F64 floating-point addition.
// =============================================================================

/// `fn _fp_add(a: f64, b: f64) -> f64 { a + b }`
fn build_tmir_fp_add_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::F64, Ty::F64],
        returns: vec![Ty::F64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_fp_add", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::F64),
            (ValueId::new(1), Ty::F64),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::FAdd,
                ty: Ty::F64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_fp_add() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_fp_add_module();

    let c_reference = r#"
double _fp_add(double a, double b) {
    return a + b;
}
"#;

    let driver = r#"
#include <stdio.h>
extern double _fp_add(double a, double b);
int main(void) {
    printf("add(1.5,2.25)=%.10f\n", _fp_add(1.5, 2.25));
    printf("add(-3.75,1.25)=%.10f\n", _fp_add(-3.75, 1.25));
    printf("add(0.0,42.5)=%.10f\n", _fp_add(0.0, 42.5));
    printf("add(1000000.125,0.875)=%.10f\n", _fp_add(1000000.125, 0.875));
    return 0;
}
"#;

    let result = differential_test("fp_add", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential fp_add test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 18: fp_sub(a, b) -> a - b
//
// Tests F64 floating-point subtraction.
// =============================================================================

/// `fn _fp_sub(a: f64, b: f64) -> f64 { a - b }`
fn build_tmir_fp_sub_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::F64, Ty::F64],
        returns: vec![Ty::F64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_fp_sub", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::F64),
            (ValueId::new(1), Ty::F64),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::FSub,
                ty: Ty::F64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_fp_sub() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_fp_sub_module();

    let c_reference = r#"
double _fp_sub(double a, double b) {
    return a - b;
}
"#;

    let driver = r#"
#include <stdio.h>
extern double _fp_sub(double a, double b);
int main(void) {
    printf("sub(5.5,2.25)=%.10f\n", _fp_sub(5.5, 2.25));
    printf("sub(-3.0,-4.5)=%.10f\n", _fp_sub(-3.0, -4.5));
    printf("sub(0.0,7.125)=%.10f\n", _fp_sub(0.0, 7.125));
    printf("sub(100.0,0.0625)=%.10f\n", _fp_sub(100.0, 0.0625));
    return 0;
}
"#;

    let result = differential_test("fp_sub", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential fp_sub test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 19: fp_mul(a, b) -> a * b
//
// Tests F64 floating-point multiplication.
// =============================================================================

/// `fn _fp_mul(a: f64, b: f64) -> f64 { a * b }`
fn build_tmir_fp_mul_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::F64, Ty::F64],
        returns: vec![Ty::F64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_fp_mul", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::F64),
            (ValueId::new(1), Ty::F64),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::FMul,
                ty: Ty::F64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_fp_mul() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_fp_mul_module();

    let c_reference = r#"
double _fp_mul(double a, double b) {
    return a * b;
}
"#;

    let driver = r#"
#include <stdio.h>
extern double _fp_mul(double a, double b);
int main(void) {
    printf("mul(3.0,2.5)=%.10f\n", _fp_mul(3.0, 2.5));
    printf("mul(-1.5,4.0)=%.10f\n", _fp_mul(-1.5, 4.0));
    printf("mul(0.0,99.0)=%.10f\n", _fp_mul(0.0, 99.0));
    printf("mul(0.125,8.0)=%.10f\n", _fp_mul(0.125, 8.0));
    return 0;
}
"#;

    let result = differential_test("fp_mul", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential fp_mul test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 20: fp_div(a, b) -> a / b
//
// Tests F64 floating-point division with non-zero divisors.
// =============================================================================

/// `fn _fp_div(a: f64, b: f64) -> f64 { a / b }`
fn build_tmir_fp_div_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::F64, Ty::F64],
        returns: vec![Ty::F64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_fp_div", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::F64),
            (ValueId::new(1), Ty::F64),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::FDiv,
                ty: Ty::F64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_fp_div() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_fp_div_module();

    let c_reference = r#"
double _fp_div(double a, double b) {
    return a / b;
}
"#;

    let driver = r#"
#include <stdio.h>
extern double _fp_div(double a, double b);
int main(void) {
    printf("div(7.5,2.5)=%.10f\n", _fp_div(7.5, 2.5));
    printf("div(-9.0,2.0)=%.10f\n", _fp_div(-9.0, 2.0));
    printf("div(1.0,8.0)=%.10f\n", _fp_div(1.0, 8.0));
    printf("div(100.0,0.5)=%.10f\n", _fp_div(100.0, 0.5));
    return 0;
}
"#;

    let result = differential_test("fp_div", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential fp_div test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 21: fp_neg(x) -> -x
//
// Tests F64 floating-point negation.
// =============================================================================

/// `fn _fp_neg(x: f64) -> f64 { -x }`
fn build_tmir_fp_neg_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::F64],
        returns: vec![Ty::F64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_fp_neg", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::F64)],
        body: vec![
            InstrNode::new(Inst::UnOp {
                op: UnOp::FNeg,
                ty: Ty::F64,
                operand: ValueId::new(0),
            })
            .with_result(ValueId::new(1)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(1)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_fp_neg() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_fp_neg_module();

    let c_reference = r#"
double _fp_neg(double x) {
    return -x;
}
"#;

    let driver = r#"
#include <stdio.h>
extern double _fp_neg(double x);
int main(void) {
    printf("neg(3.5)=%.10f\n", _fp_neg(3.5));
    printf("neg(-2.25)=%.10f\n", _fp_neg(-2.25));
    printf("neg(1024.125)=%.10f\n", _fp_neg(1024.125));
    printf("neg(-0.5)=%.10f\n", _fp_neg(-0.5));
    return 0;
}
"#;

    let result = differential_test("fp_neg", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential fp_neg test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 22: fp_cmp_lt(a, b) -> if a < b { 1 } else { 0 }
//
// Tests ordered F64 less-than via FCmp + CondBr.
// =============================================================================

/// `fn _fp_cmp_lt(a: f64, b: f64) -> i32 { if a < b { 1 } else { 0 } }`
fn build_tmir_fp_cmp_lt_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::F64, Ty::F64],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_fp_cmp_lt", ft_id, BlockId::new(0));
    func.blocks = vec![
        TmirBlock {
            id: BlockId::new(0),
            params: vec![
                (ValueId::new(0), Ty::F64),
                (ValueId::new(1), Ty::F64),
            ],
            body: vec![
                InstrNode::new(Inst::FCmp {
                    op: FCmpOp::OLt,
                    ty: Ty::F64,
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
                    ty: Ty::I32,
                    value: Constant::Int(1),
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
                    ty: Ty::I32,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(4)),
                InstrNode::new(Inst::Return {
                    values: vec![ValueId::new(4)],
                }),
            ],
        },
    ];
    module.add_function(func);
    module
}

#[test]
fn test_differential_fp_cmp_lt() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_fp_cmp_lt_module();

    let c_reference = r#"
int _fp_cmp_lt(double a, double b) {
    return a < b ? 1 : 0;
}
"#;

    let driver = r#"
#include <stdio.h>
extern int _fp_cmp_lt(double a, double b);
int main(void) {
    printf("cmp(1.0,2.0)=%d\n", _fp_cmp_lt(1.0, 2.0));
    printf("cmp(2.0,1.0)=%d\n", _fp_cmp_lt(2.0, 1.0));
    printf("cmp(3.5,3.5)=%d\n", _fp_cmp_lt(3.5, 3.5));
    printf("cmp(-4.0,-1.0)=%d\n", _fp_cmp_lt(-4.0, -1.0));
    printf("cmp(10.0,-10.0)=%d\n", _fp_cmp_lt(10.0, -10.0));
    return 0;
}
"#;

    let result = differential_test("fp_cmp_lt", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential fp_cmp_lt test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 23: fp_discriminant(a, b, c) -> b*b - 4.0*a*c
//
// Tests a multi-step F64 expression with float constants.
// =============================================================================

/// `fn _fp_discriminant(a: f64, b: f64, c: f64) -> f64 { b*b - 4.0*a*c }`
fn build_tmir_fp_discriminant_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::F64, Ty::F64, Ty::F64],
        returns: vec![Ty::F64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_fp_discriminant", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::F64),
            (ValueId::new(1), Ty::F64),
            (ValueId::new(2), Ty::F64),
        ],
        body: vec![
            InstrNode::new(Inst::Const {
                ty: Ty::F64,
                value: Constant::Float(4.0),
            })
            .with_result(ValueId::new(3)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::FMul,
                ty: Ty::F64,
                lhs: ValueId::new(1),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(4)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::FMul,
                ty: Ty::F64,
                lhs: ValueId::new(3),
                rhs: ValueId::new(0),
            })
            .with_result(ValueId::new(5)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::FMul,
                ty: Ty::F64,
                lhs: ValueId::new(5),
                rhs: ValueId::new(2),
            })
            .with_result(ValueId::new(6)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::FSub,
                ty: Ty::F64,
                lhs: ValueId::new(4),
                rhs: ValueId::new(6),
            })
            .with_result(ValueId::new(7)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(7)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_fp_discriminant() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_fp_discriminant_module();

    let c_reference = r#"
double _fp_discriminant(double a, double b, double c) {
    return b * b - 4.0 * a * c;
}
"#;

    let driver = r#"
#include <stdio.h>
extern double _fp_discriminant(double a, double b, double c);
int main(void) {
    printf("disc(1.0,5.0,6.0)=%.10f\n", _fp_discriminant(1.0, 5.0, 6.0));
    printf("disc(1.0,2.0,1.0)=%.10f\n", _fp_discriminant(1.0, 2.0, 1.0));
    printf("disc(1.0,1.0,1.0)=%.10f\n", _fp_discriminant(1.0, 1.0, 1.0));
    printf("disc(2.0,4.0,1.0)=%.10f\n", _fp_discriminant(2.0, 4.0, 1.0));
    printf("disc(0.5,1.5,0.25)=%.10f\n", _fp_discriminant(0.5, 1.5, 0.25));
    return 0;
}
"#;

    let result = differential_test("fp_discriminant", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential fp_discriminant test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 24: fp32_add(a, b) -> a + b
//
// Tests the F32 floating-point path.
// =============================================================================

/// `fn _fp32_add(a: f32, b: f32) -> f32 { a + b }`
fn build_tmir_fp32_add_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::F32, Ty::F32],
        returns: vec![Ty::F32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_fp32_add", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::F32),
            (ValueId::new(1), Ty::F32),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::FAdd,
                ty: Ty::F32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_fp32_add() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_fp32_add_module();

    let c_reference = r#"
float _fp32_add(float a, float b) {
    return a + b;
}
"#;

    let driver = r#"
#include <stdio.h>
extern float _fp32_add(float a, float b);
int main(void) {
    printf("fp32_add(1.5,2.25)=%.10f\n", (double)_fp32_add(1.5f, 2.25f));
    printf("fp32_add(-3.75,1.25)=%.10f\n", (double)_fp32_add(-3.75f, 1.25f));
    printf("fp32_add(0.1,0.2)=%.10f\n", (double)_fp32_add(0.1f, 0.2f));
    printf("fp32_add(1000.5,0.25)=%.10f\n", (double)_fp32_add(1000.5f, 0.25f));
    return 0;
}
"#;

    let result = differential_test("fp32_add", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential fp32_add test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 25: int_neg(x) -> -x
//
// Tests integer negation via UnOp::Neg.
// =============================================================================

/// `fn _int_neg(x: i64) -> i64 { -x }`
fn build_tmir_int_neg_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_int_neg", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::I64)],
        body: vec![
            InstrNode::new(Inst::UnOp {
                op: UnOp::Neg,
                ty: Ty::I64,
                operand: ValueId::new(0),
            })
            .with_result(ValueId::new(1)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(1)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_int_neg() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_int_neg_module();

    let c_reference = r#"
long _int_neg(long x) {
    return -x;
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _int_neg(long x);
int main(void) {
    printf("int_neg(0)=%ld\n", _int_neg(0));
    printf("int_neg(1)=%ld\n", _int_neg(1));
    printf("int_neg(-1)=%ld\n", _int_neg(-1));
    printf("int_neg(42)=%ld\n", _int_neg(42));
    printf("int_neg(-42)=%ld\n", _int_neg(-42));
    printf("int_neg(4000000000)=%ld\n", _int_neg(4000000000L));
    printf("int_neg(-4000000000)=%ld\n", _int_neg(-4000000000L));
    return 0;
}
"#;

    let result = differential_test("int_neg", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential int_neg test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 26: bitwise_not(x) -> ~x
//
// Tests UnOp::Not lowered via ORN Rd, XZR, Rm (MVN alias).
// Previously broken (issue #334): ISel emitted OrnRR with 2 operands
// but encoder expects 3 (Rd, Rn=XZR, Rm). Fixed by emitting XZR as
// the explicit second operand.
// =============================================================================

/// `fn _bitwise_not(x: i64) -> i64 { !x }`
fn build_tmir_bitwise_not_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_bitwise_not", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::I64)],
        body: vec![
            InstrNode::new(Inst::UnOp {
                op: UnOp::Not,
                ty: Ty::I64,
                operand: ValueId::new(0),
            })
            .with_result(ValueId::new(1)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(1)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_bitwise_not() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_bitwise_not_module();

    let c_reference = r#"
long _bitwise_not(long x) {
    return ~x;
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _bitwise_not(long x);
int main(void) {
    printf("not(0)=%ld\n", _bitwise_not(0));
    printf("not(1)=%ld\n", _bitwise_not(1));
    printf("not(-1)=%ld\n", _bitwise_not(-1));
    printf("not(255)=%ld\n", _bitwise_not(255));
    printf("not(0x7FFFFFFFFFFFFFFF)=%ld\n", _bitwise_not(0x7FFFFFFFFFFFFFFFL));
    return 0;
}
"#;

    let result = differential_test("bitwise_not", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential bitwise_not test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 27: icmp_eq(a, b) -> if a == b { 1 } else { 0 }
//
// Tests integer equality comparison via ICmp::Eq + CondBr.
// =============================================================================

/// `fn _icmp_eq(a: i64, b: i64) -> i64 { if a == b { 1 } else { 0 } }`
fn build_tmir_icmp_eq_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_icmp_eq", ft_id, BlockId::new(0));
    func.blocks = vec![
        TmirBlock {
            id: BlockId::new(0),
            params: vec![
                (ValueId::new(0), Ty::I64),
                (ValueId::new(1), Ty::I64),
            ],
            body: vec![
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Eq,
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
                    value: Constant::Int(1),
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
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(4)),
                InstrNode::new(Inst::Return {
                    values: vec![ValueId::new(4)],
                }),
            ],
        },
    ];
    module.add_function(func);
    module
}

#[test]
fn test_differential_icmp_eq() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_icmp_eq_module();

    let c_reference = r#"
long _icmp_eq(long a, long b) {
    return a == b ? 1L : 0L;
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _icmp_eq(long a, long b);
int main(void) {
    printf("icmp_eq(0,0)=%ld\n", _icmp_eq(0, 0));
    printf("icmp_eq(1,1)=%ld\n", _icmp_eq(1, 1));
    printf("icmp_eq(1,2)=%ld\n", _icmp_eq(1, 2));
    printf("icmp_eq(-1,-1)=%ld\n", _icmp_eq(-1, -1));
    printf("icmp_eq(-1,1)=%ld\n", _icmp_eq(-1, 1));
    printf("icmp_eq(123456789,123456789)=%ld\n", _icmp_eq(123456789, 123456789));
    printf("icmp_eq(4000000000,4000000000)=%ld\n", _icmp_eq(4000000000L, 4000000000L));
    printf("icmp_eq(4000000000,-4000000000)=%ld\n", _icmp_eq(4000000000L, -4000000000L));
    return 0;
}
"#;

    let result = differential_test("icmp_eq", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential icmp_eq test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 28: icmp_ne(a, b) -> if a != b { 1 } else { 0 }
//
// Tests integer inequality comparison via ICmp::Ne + CondBr.
// =============================================================================

/// `fn _icmp_ne(a: i64, b: i64) -> i64 { if a != b { 1 } else { 0 } }`
fn build_tmir_icmp_ne_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_icmp_ne", ft_id, BlockId::new(0));
    func.blocks = vec![
        TmirBlock {
            id: BlockId::new(0),
            params: vec![
                (ValueId::new(0), Ty::I64),
                (ValueId::new(1), Ty::I64),
            ],
            body: vec![
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Ne,
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
                    value: Constant::Int(1),
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
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(4)),
                InstrNode::new(Inst::Return {
                    values: vec![ValueId::new(4)],
                }),
            ],
        },
    ];
    module.add_function(func);
    module
}

#[test]
fn test_differential_icmp_ne() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_icmp_ne_module();

    let c_reference = r#"
long _icmp_ne(long a, long b) {
    return a != b ? 1L : 0L;
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _icmp_ne(long a, long b);
int main(void) {
    printf("icmp_ne(0,0)=%ld\n", _icmp_ne(0, 0));
    printf("icmp_ne(1,1)=%ld\n", _icmp_ne(1, 1));
    printf("icmp_ne(1,2)=%ld\n", _icmp_ne(1, 2));
    printf("icmp_ne(-1,-1)=%ld\n", _icmp_ne(-1, -1));
    printf("icmp_ne(-1,1)=%ld\n", _icmp_ne(-1, 1));
    printf("icmp_ne(123456789,123456789)=%ld\n", _icmp_ne(123456789, 123456789));
    printf("icmp_ne(4000000000,4000000000)=%ld\n", _icmp_ne(4000000000L, 4000000000L));
    printf("icmp_ne(4000000000,-4000000000)=%ld\n", _icmp_ne(4000000000L, -4000000000L));
    return 0;
}
"#;

    let result = differential_test("icmp_ne", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential icmp_ne test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 29: icmp_ult(a, b) -> if a <u b { 1 } else { 0 }
//
// Tests unsigned i64 less-than comparison via ICmp::Ult + CondBr.
// =============================================================================

/// `fn _icmp_ult(a: i64, b: i64) -> i64 { if a <u b { 1 } else { 0 } }`
fn build_tmir_icmp_ult_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_icmp_ult", ft_id, BlockId::new(0));
    func.blocks = vec![
        TmirBlock {
            id: BlockId::new(0),
            params: vec![
                (ValueId::new(0), Ty::I64),
                (ValueId::new(1), Ty::I64),
            ],
            body: vec![
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Ult,
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
                    value: Constant::Int(1),
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
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(4)),
                InstrNode::new(Inst::Return {
                    values: vec![ValueId::new(4)],
                }),
            ],
        },
    ];
    module.add_function(func);
    module
}

#[test]
fn test_differential_icmp_ult() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_icmp_ult_module();

    let c_reference = r#"
long _icmp_ult(long a, long b) {
    return ((unsigned long)a < (unsigned long)b) ? 1L : 0L;
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _icmp_ult(long a, long b);
int main(void) {
    printf("icmp_ult(0,0)=%ld\n", _icmp_ult(0, 0));
    printf("icmp_ult(0,1)=%ld\n", _icmp_ult(0, 1));
    printf("icmp_ult(1,0)=%ld\n", _icmp_ult(1, 0));
    printf("icmp_ult(-1,0)=%ld\n", _icmp_ult(-1, 0));
    printf("icmp_ult(0,-1)=%ld\n", _icmp_ult(0, -1));
    printf("icmp_ult(-2,-1)=%ld\n", _icmp_ult(-2, -1));
    printf("icmp_ult(-1,-2)=%ld\n", _icmp_ult(-1, -2));
    printf("icmp_ult(9223372036854775807,-1)=%ld\n", _icmp_ult(9223372036854775807L, -1L));
    return 0;
}
"#;

    let result = differential_test("icmp_ult", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential icmp_ult test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 30: sdiv(a, b) -> a / b
//
// Tests signed i32 division with non-zero divisors.
// =============================================================================

/// `fn _sdiv(a: i32, b: i32) -> i32 { a / b }`
fn build_tmir_sdiv_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32, Ty::I32],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_sdiv", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I32),
            (ValueId::new(1), Ty::I32),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::SDiv,
                ty: Ty::I32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_sdiv() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_sdiv_module();

    let c_reference = r#"
int _sdiv(int a, int b) {
    return a / b;
}
"#;

    let driver = r#"
#include <stdio.h>
extern int _sdiv(int a, int b);
int main(void) {
    printf("sdiv(0,1)=%d\n", _sdiv(0, 1));
    printf("sdiv(1,1)=%d\n", _sdiv(1, 1));
    printf("sdiv(-1,1)=%d\n", _sdiv(-1, 1));
    printf("sdiv(-1,-1)=%d\n", _sdiv(-1, -1));
    printf("sdiv(17,5)=%d\n", _sdiv(17, 5));
    printf("sdiv(17,-5)=%d\n", _sdiv(17, -5));
    printf("sdiv(-17,5)=%d\n", _sdiv(-17, 5));
    printf("sdiv(-17,-5)=%d\n", _sdiv(-17, -5));
    printf("sdiv(2147483647,2)=%d\n", _sdiv(2147483647, 2));
    printf("sdiv(-2147483648,2)=%d\n", _sdiv((-2147483647 - 1), 2));
    printf("sdiv(-2147483648,2147483647)=%d\n", _sdiv((-2147483647 - 1), 2147483647));
    return 0;
}
"#;

    let result = differential_test("sdiv", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential sdiv test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 31: udiv(a, b) -> a /u b
//
// Tests unsigned i32 division with non-zero divisors.
// =============================================================================

/// `fn _udiv(a: i32, b: i32) -> i32 { a /u b }`
fn build_tmir_udiv_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32, Ty::I32],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_udiv", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I32),
            (ValueId::new(1), Ty::I32),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::UDiv,
                ty: Ty::I32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_udiv() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_udiv_module();

    let c_reference = r#"
unsigned int _udiv(unsigned int a, unsigned int b) {
    return a / b;
}
"#;

    let driver = r#"
#include <stdio.h>
extern unsigned int _udiv(unsigned int a, unsigned int b);
int main(void) {
    printf("udiv(0,1)=%u\n", _udiv(0u, 1u));
    printf("udiv(1,1)=%u\n", _udiv(1u, 1u));
    printf("udiv(10,3)=%u\n", _udiv(10u, 3u));
    printf("udiv(16,4)=%u\n", _udiv(16u, 4u));
    printf("udiv(2147483648,2)=%u\n", _udiv(2147483648u, 2u));
    printf("udiv(2147483648,3)=%u\n", _udiv(2147483648u, 3u));
    printf("udiv(4294967295,2)=%u\n", _udiv(4294967295u, 2u));
    printf("udiv(4294967295,2147483647)=%u\n", _udiv(4294967295u, 2147483647u));
    return 0;
}
"#;

    let result = differential_test("udiv", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential udiv test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 32: zext_i32_to_i64(x) -> zero-extend i32 to i64
//
// Tests zero-extension preserving the low 32 bits into a 64-bit result.
// =============================================================================

/// `fn _zext_i32_to_i64(x: i32) -> i64 { x as u64 }`
fn build_tmir_zext_i32_to_i64_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func =
        TmirFunction::new(FuncId::new(0), "_zext_i32_to_i64", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::I32)],
        body: vec![
            InstrNode::new(Inst::Cast {
                op: CastOp::ZExt,
                src_ty: Ty::I32,
                dst_ty: Ty::I64,
                operand: ValueId::new(0),
            })
            .with_result(ValueId::new(1)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(1)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_zext_i32_to_i64() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_zext_i32_to_i64_module();

    let c_reference = r#"
unsigned long _zext_i32_to_i64(int x) {
    return (unsigned long)(unsigned int)x;
}
"#;

    let driver = r#"
#include <stdio.h>
extern unsigned long _zext_i32_to_i64(int x);
int main(void) {
    printf("zext(0)=%lu\n", _zext_i32_to_i64(0));
    printf("zext(1)=%lu\n", _zext_i32_to_i64(1));
    printf("zext(-1)=%lu\n", _zext_i32_to_i64(-1));
    printf("zext(2147483647)=%lu\n", _zext_i32_to_i64(2147483647));
    printf("zext(-2147483648)=%lu\n", _zext_i32_to_i64((-2147483647 - 1)));
    printf("zext(123456789)=%lu\n", _zext_i32_to_i64(123456789));
    printf("zext(-123456789)=%lu\n", _zext_i32_to_i64(-123456789));
    printf("zext(1073741824)=%lu\n", _zext_i32_to_i64(1073741824));
    return 0;
}
"#;

    let result = differential_test("zext_i32_to_i64", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential zext_i32_to_i64 test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 33: select_val(cond_a, cond_b, x, y) -> if cond_a > cond_b { x } else { y }
//
// Tests ICmp + Select for value selection without a final control-flow split.
// =============================================================================

/// `fn _select_val(cond_a: i64, cond_b: i64, x: i64, y: i64) -> i64 { if cond_a > cond_b { x } else { y } }`
fn build_tmir_select_val_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64, Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_select_val", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I64),
            (ValueId::new(1), Ty::I64),
            (ValueId::new(2), Ty::I64),
            (ValueId::new(3), Ty::I64),
        ],
        body: vec![
            InstrNode::new(Inst::ICmp {
                op: ICmpOp::Sgt,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(4)),
            InstrNode::new(Inst::Select {
                ty: Ty::I64,
                cond: ValueId::new(4),
                then_val: ValueId::new(2),
                else_val: ValueId::new(3),
            })
            .with_result(ValueId::new(5)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(5)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_select_val() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_select_val_module();

    let c_reference = r#"
long _select_val(long cond_a, long cond_b, long x, long y) {
    return cond_a > cond_b ? x : y;
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _select_val(long cond_a, long cond_b, long x, long y);
int main(void) {
    printf("select(5,3,10,20)=%ld\n", _select_val(5, 3, 10, 20));
    printf("select(3,5,10,20)=%ld\n", _select_val(3, 5, 10, 20));
    printf("select(7,7,10,20)=%ld\n", _select_val(7, 7, 10, 20));
    printf("select(-1,-2,111,222)=%ld\n", _select_val(-1, -2, 111, 222));
    printf("select(-2,-1,111,222)=%ld\n", _select_val(-2, -1, 111, 222));
    printf("select(100,0,4000000000,-4000000000)=%ld\n", _select_val(100, 0, 4000000000L, -4000000000L));
    return 0;
}
"#;

    let result = differential_test("select_val", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential select_val test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 34: clamp(x, lo, hi) -> if x < lo { lo } else if x > hi { hi } else { x }
//
// Tests multi-level control flow with 4 basic blocks and block arguments.
// =============================================================================

/// `fn _clamp(x: i64, lo: i64, hi: i64) -> i64 { if x < lo { lo } else if x > hi { hi } else { x } }`
fn build_tmir_clamp_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_clamp", ft_id, BlockId::new(0));
    func.blocks = vec![
        TmirBlock {
            id: BlockId::new(0),
            params: vec![
                (ValueId::new(0), Ty::I64),
                (ValueId::new(1), Ty::I64),
                (ValueId::new(2), Ty::I64),
            ],
            body: vec![
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Slt,
                    ty: Ty::I64,
                    lhs: ValueId::new(0),
                    rhs: ValueId::new(1),
                })
                .with_result(ValueId::new(3)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(3),
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
                values: vec![ValueId::new(1)],
            })],
        },
        TmirBlock {
            id: BlockId::new(2),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sgt,
                    ty: Ty::I64,
                    lhs: ValueId::new(0),
                    rhs: ValueId::new(2),
                })
                .with_result(ValueId::new(10)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(10),
                    then_target: BlockId::new(3),
                    then_args: vec![ValueId::new(2)],
                    else_target: BlockId::new(3),
                    else_args: vec![ValueId::new(0)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(3),
            params: vec![(ValueId::new(20), Ty::I64)],
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(20)],
            })],
        },
    ];
    module.add_function(func);
    module
}

#[test]
fn test_differential_clamp() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_clamp_module();

    let c_reference = r#"
long _clamp(long x, long lo, long hi) {
    if (x < lo) {
        return lo;
    } else if (x > hi) {
        return hi;
    } else {
        return x;
    }
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _clamp(long x, long lo, long hi);
int main(void) {
    printf("clamp(5,0,10)=%ld\n", _clamp(5, 0, 10));
    printf("clamp(-5,0,10)=%ld\n", _clamp(-5, 0, 10));
    printf("clamp(15,0,10)=%ld\n", _clamp(15, 0, 10));
    printf("clamp(0,0,10)=%ld\n", _clamp(0, 0, 10));
    printf("clamp(10,0,10)=%ld\n", _clamp(10, 0, 10));
    printf("clamp(-100,-50,50)=%ld\n", _clamp(-100, -50, 50));
    printf("clamp(100,-50,50)=%ld\n", _clamp(100, -50, 50));
    printf("clamp(4000000000,0,3000000000)=%ld\n", _clamp(4000000000L, 0L, 3000000000L));
    return 0;
}
"#;

    let result = differential_test("clamp", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential clamp test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 35: fp_cmp_gt(a, b) -> if a > b { 1 } else { 0 }
//
// Tests ordered F64 greater-than via FCmp + CondBr.
// Previously broken (issue #335): FCmpOp::OGt always evaluated to false.
// Fixed by using Gpr64 for CSET destination in select_fcmp (matching
// select_icmp) to avoid W/X register aliasing issues in regalloc.
// =============================================================================

/// `fn _fp_cmp_gt(a: f64, b: f64) -> i32 { if a > b { 1 } else { 0 } }`
fn build_tmir_fp_cmp_gt_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::F64, Ty::F64],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_fp_cmp_gt", ft_id, BlockId::new(0));
    func.blocks = vec![
        TmirBlock {
            id: BlockId::new(0),
            params: vec![
                (ValueId::new(0), Ty::F64),
                (ValueId::new(1), Ty::F64),
            ],
            body: vec![
                InstrNode::new(Inst::FCmp {
                    op: FCmpOp::OGt,
                    ty: Ty::F64,
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
                    ty: Ty::I32,
                    value: Constant::Int(1),
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
                    ty: Ty::I32,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(4)),
                InstrNode::new(Inst::Return {
                    values: vec![ValueId::new(4)],
                }),
            ],
        },
    ];
    module.add_function(func);
    module
}

#[test]
fn test_differential_fp_cmp_gt() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_fp_cmp_gt_module();

    let c_reference = r#"
int _fp_cmp_gt(double a, double b) {
    return a > b ? 1 : 0;
}
"#;

    let driver = r#"
#include <stdio.h>
extern int _fp_cmp_gt(double a, double b);
int main(void) {
    printf("cmp(2.0,1.0)=%d\n", _fp_cmp_gt(2.0, 1.0));
    printf("cmp(1.0,2.0)=%d\n", _fp_cmp_gt(1.0, 2.0));
    printf("cmp(3.5,3.5)=%d\n", _fp_cmp_gt(3.5, 3.5));
    printf("cmp(-1.0,-4.0)=%d\n", _fp_cmp_gt(-1.0, -4.0));
    printf("cmp(-10.0,10.0)=%d\n", _fp_cmp_gt(-10.0, 10.0));
    return 0;
}
"#;

    let result = differential_test("fp_cmp_gt", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential fp_cmp_gt test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 36: sub64(a, b) -> a - b
//
// Tests standalone i64 subtraction.
// =============================================================================

/// `fn _sub64(a: i64, b: i64) -> i64 { a - b }`
fn build_tmir_sub64_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_sub64", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I64),
            (ValueId::new(1), Ty::I64),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::Sub,
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
    module.add_function(func);
    module
}

#[test]
fn test_differential_sub64() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_sub64_module();

    let c_reference = r#"
long _sub64(long a, long b) {
    return a - b;
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _sub64(long a, long b);
int main(void) {
    printf("sub64(0,0)=%ld\n", _sub64(0, 0));
    printf("sub64(1,1)=%ld\n", _sub64(1, 1));
    printf("sub64(10,3)=%ld\n", _sub64(10, 3));
    printf("sub64(-10,3)=%ld\n", _sub64(-10, 3));
    printf("sub64(10,-3)=%ld\n", _sub64(10, -3));
    printf("sub64(-10,-3)=%ld\n", _sub64(-10, -3));
    printf("sub64(4000000000,1)=%ld\n", _sub64(4000000000L, 1L));
    printf("sub64(-4000000000,1)=%ld\n", _sub64(-4000000000L, 1L));
    printf("sub64(1234567890123,987654321)=%ld\n", _sub64(1234567890123L, 987654321L));
    return 0;
}
"#;

    let result = differential_test("sub64", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential sub64 test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 37: mul32(a, b) -> a * b
//
// Tests standalone i32 multiplication with non-overflowing inputs.
// =============================================================================

/// `fn _mul32(a: i32, b: i32) -> i32 { a * b }`
fn build_tmir_mul32_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32, Ty::I32],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_mul32", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I32),
            (ValueId::new(1), Ty::I32),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_mul32() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_mul32_module();

    let c_reference = r#"
int _mul32(int a, int b) {
    return a * b;
}
"#;

    let driver = r#"
#include <stdio.h>
extern int _mul32(int a, int b);
int main(void) {
    printf("mul32(0,0)=%d\n", _mul32(0, 0));
    printf("mul32(1,1)=%d\n", _mul32(1, 1));
    printf("mul32(-1,1)=%d\n", _mul32(-1, 1));
    printf("mul32(-1,-1)=%d\n", _mul32(-1, -1));
    printf("mul32(3,7)=%d\n", _mul32(3, 7));
    printf("mul32(46340,46340)=%d\n", _mul32(46340, 46340));
    printf("mul32(-46340,46340)=%d\n", _mul32(-46340, 46340));
    printf("mul32(12345,67)=%d\n", _mul32(12345, 67));
    printf("mul32(-12345,67)=%d\n", _mul32(-12345, 67));
    return 0;
}
"#;

    let result = differential_test("mul32", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential mul32 test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 38: polynomial(x) -> x*x*x - 2*x*x + 3*x - 4
//
// Tests a multi-step i64 arithmetic expression with dependent operations.
// =============================================================================

/// `fn _polynomial(x: i64) -> i64 { x*x*x - 2*x*x + 3*x - 4 }`
fn build_tmir_polynomial_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_polynomial", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::I64)],
        body: vec![
            // x*x -> v1
            InstrNode::new(Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(0),
            })
            .with_result(ValueId::new(1)),
            // x*x*x -> v2
            InstrNode::new(Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: ValueId::new(1),
                rhs: ValueId::new(0),
            })
            .with_result(ValueId::new(2)),
            // 2 -> v3
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(2),
            })
            .with_result(ValueId::new(3)),
            // 2*x*x -> v4
            InstrNode::new(Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: ValueId::new(3),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(4)),
            // x^3 - 2*x^2 -> v5
            InstrNode::new(Inst::BinOp {
                op: BinOp::Sub,
                ty: Ty::I64,
                lhs: ValueId::new(2),
                rhs: ValueId::new(4),
            })
            .with_result(ValueId::new(5)),
            // 3 -> v6
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(3),
            })
            .with_result(ValueId::new(6)),
            // 3*x -> v7
            InstrNode::new(Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: ValueId::new(6),
                rhs: ValueId::new(0),
            })
            .with_result(ValueId::new(7)),
            // (x^3 - 2*x^2) + 3*x -> v8
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(5),
                rhs: ValueId::new(7),
            })
            .with_result(ValueId::new(8)),
            // 4 -> v9
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(4),
            })
            .with_result(ValueId::new(9)),
            // result - 4 -> v10
            InstrNode::new(Inst::BinOp {
                op: BinOp::Sub,
                ty: Ty::I64,
                lhs: ValueId::new(8),
                rhs: ValueId::new(9),
            })
            .with_result(ValueId::new(10)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(10)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_polynomial() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_polynomial_module();

    let c_reference = r#"
long _polynomial(long x) {
    return x * x * x - 2L * x * x + 3L * x - 4L;
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _polynomial(long x);
int main(void) {
    printf("poly(0)=%ld\n", _polynomial(0));
    printf("poly(1)=%ld\n", _polynomial(1));
    printf("poly(2)=%ld\n", _polynomial(2));
    printf("poly(-1)=%ld\n", _polynomial(-1));
    printf("poly(-2)=%ld\n", _polynomial(-2));
    printf("poly(3)=%ld\n", _polynomial(3));
    printf("poly(10)=%ld\n", _polynomial(10));
    return 0;
}
"#;

    let result = differential_test("polynomial", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential polynomial test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Tests 39-50: Loop patterns, nested control flow, register pressure,
// mixed-width arithmetic, select chains, and nested loops.
//
// Added in Wave 47 for #301 — E2E correctness validation expansion.
// =============================================================================

// Test 39: while_countdown
fn build_tmir_while_countdown_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_while_countdown", ft_id, BlockId::new(0));
    func.blocks = vec![
        TmirBlock {
            id: BlockId::new(0),
            params: vec![(ValueId::new(0), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(1)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(1),
                    args: vec![ValueId::new(0), ValueId::new(1)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(1),
            params: vec![(ValueId::new(2), Ty::I64), (ValueId::new(3), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(4)),
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sgt,
                    ty: Ty::I64,
                    lhs: ValueId::new(2),
                    rhs: ValueId::new(4),
                })
                .with_result(ValueId::new(5)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(5),
                    then_target: BlockId::new(2),
                    then_args: vec![ValueId::new(2), ValueId::new(3)],
                    else_target: BlockId::new(3),
                    else_args: vec![ValueId::new(3)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(2),
            params: vec![(ValueId::new(6), Ty::I64), (ValueId::new(7), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(8)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Sub,
                    ty: Ty::I64,
                    lhs: ValueId::new(6),
                    rhs: ValueId::new(8),
                })
                .with_result(ValueId::new(9)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(7),
                    rhs: ValueId::new(8),
                })
                .with_result(ValueId::new(10)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(1),
                    args: vec![ValueId::new(9), ValueId::new(10)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(3),
            params: vec![(ValueId::new(11), Ty::I64)],
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(11)],
            })],
        },
    ];
    module.add_function(func);
    module
}

#[test]
fn test_differential_while_countdown() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }
    let module = build_tmir_while_countdown_module();
    let c_reference = r#"
long _while_countdown(long n) {
    long count = 0;
    while (n > 0) {
        n = n - 1;
        count = count + 1;
    }
    return count;
}
"#;
    let driver = r#"
#include <stdio.h>
extern long _while_countdown(long n);
int main(void) {
    printf("while_countdown(0)=%ld\n", _while_countdown(0));
    printf("while_countdown(1)=%ld\n", _while_countdown(1));
    printf("while_countdown(5)=%ld\n", _while_countdown(5));
    printf("while_countdown(-3)=%ld\n", _while_countdown(-3));
    return 0;
}
"#;
    let result = differential_test("while_countdown", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential while_countdown test failed: {}",
        result.unwrap_err()
    );
}

// Test 40: nested_if_in_loop
fn build_tmir_nested_if_in_loop_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_nested_if_in_loop", ft_id, BlockId::new(0));
    func.blocks = vec![
        TmirBlock {
            id: BlockId::new(0),
            params: vec![(ValueId::new(0), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(1)),
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(2)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(1),
                    args: vec![ValueId::new(1), ValueId::new(2)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(1),
            params: vec![(ValueId::new(3), Ty::I64), (ValueId::new(4), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sle,
                    ty: Ty::I64,
                    lhs: ValueId::new(3),
                    rhs: ValueId::new(0),
                })
                .with_result(ValueId::new(5)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(5),
                    then_target: BlockId::new(2),
                    then_args: vec![ValueId::new(3), ValueId::new(4)],
                    else_target: BlockId::new(5),
                    else_args: vec![ValueId::new(4)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(2),
            params: vec![(ValueId::new(6), Ty::I64), (ValueId::new(7), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(2),
                })
                .with_result(ValueId::new(8)),
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(9)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::SRem,
                    ty: Ty::I64,
                    lhs: ValueId::new(6),
                    rhs: ValueId::new(8),
                })
                .with_result(ValueId::new(10)),
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Eq,
                    ty: Ty::I64,
                    lhs: ValueId::new(10),
                    rhs: ValueId::new(9),
                })
                .with_result(ValueId::new(11)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(11),
                    then_target: BlockId::new(3),
                    then_args: vec![ValueId::new(6), ValueId::new(7)],
                    else_target: BlockId::new(4),
                    else_args: vec![ValueId::new(6), ValueId::new(7)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(3),
            params: vec![(ValueId::new(12), Ty::I64), (ValueId::new(13), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(13),
                    rhs: ValueId::new(12),
                })
                .with_result(ValueId::new(14)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(4),
                    args: vec![ValueId::new(12), ValueId::new(14)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(4),
            params: vec![(ValueId::new(15), Ty::I64), (ValueId::new(16), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(17)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(15),
                    rhs: ValueId::new(17),
                })
                .with_result(ValueId::new(18)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(1),
                    args: vec![ValueId::new(18), ValueId::new(16)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(5),
            params: vec![(ValueId::new(19), Ty::I64)],
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(19)],
            })],
        },
    ];
    module.add_function(func);
    module
}

#[test]
fn test_differential_nested_if_in_loop() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }
    let module = build_tmir_nested_if_in_loop_module();
    let c_reference = r#"
long _nested_if_in_loop(long n) {
    long i = 1;
    long sum = 0;
    while (i <= n) {
        if ((i % 2) == 0) {
            sum = sum + i;
        }
        i = i + 1;
    }
    return sum;
}
"#;
    let driver = r#"
#include <stdio.h>
extern long _nested_if_in_loop(long n);
int main(void) {
    printf("nested_if_in_loop(0)=%ld\n", _nested_if_in_loop(0));
    printf("nested_if_in_loop(1)=%ld\n", _nested_if_in_loop(1));
    printf("nested_if_in_loop(6)=%ld\n", _nested_if_in_loop(6));
    printf("nested_if_in_loop(9)=%ld\n", _nested_if_in_loop(9));
    return 0;
}
"#;
    let result = differential_test("nested_if_in_loop", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential nested_if_in_loop test failed: {}",
        result.unwrap_err()
    );
}

// Test 41: loop_in_if
fn build_tmir_loop_in_if_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_loop_in_if", ft_id, BlockId::new(0));
    func.blocks = vec![
        TmirBlock {
            id: BlockId::new(0),
            params: vec![(ValueId::new(0), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(1)),
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sgt,
                    ty: Ty::I64,
                    lhs: ValueId::new(0),
                    rhs: ValueId::new(1),
                })
                .with_result(ValueId::new(2)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(2),
                    then_target: BlockId::new(1),
                    then_args: vec![],
                    else_target: BlockId::new(4),
                    else_args: vec![ValueId::new(1)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(1),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(3)),
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(4)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(2),
                    args: vec![ValueId::new(3), ValueId::new(4)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(2),
            params: vec![(ValueId::new(5), Ty::I64), (ValueId::new(6), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sle,
                    ty: Ty::I64,
                    lhs: ValueId::new(5),
                    rhs: ValueId::new(0),
                })
                .with_result(ValueId::new(7)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(7),
                    then_target: BlockId::new(3),
                    then_args: vec![ValueId::new(5), ValueId::new(6)],
                    else_target: BlockId::new(4),
                    else_args: vec![ValueId::new(6)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(3),
            params: vec![(ValueId::new(8), Ty::I64), (ValueId::new(9), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(10)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(9),
                    rhs: ValueId::new(8),
                })
                .with_result(ValueId::new(11)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(8),
                    rhs: ValueId::new(10),
                })
                .with_result(ValueId::new(12)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(2),
                    args: vec![ValueId::new(12), ValueId::new(11)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(4),
            params: vec![(ValueId::new(13), Ty::I64)],
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(13)],
            })],
        },
    ];
    module.add_function(func);
    module
}

#[test]
fn test_differential_loop_in_if() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }
    let module = build_tmir_loop_in_if_module();
    let c_reference = r#"
long _loop_in_if(long n) {
    if (n > 0) {
        long i = 1;
        long sum = 0;
        while (i <= n) {
            sum = sum + i;
            i = i + 1;
        }
        return sum;
    }
    return 0;
}
"#;
    let driver = r#"
#include <stdio.h>
extern long _loop_in_if(long n);
int main(void) {
    printf("loop_in_if(-2)=%ld\n", _loop_in_if(-2));
    printf("loop_in_if(0)=%ld\n", _loop_in_if(0));
    printf("loop_in_if(1)=%ld\n", _loop_in_if(1));
    printf("loop_in_if(5)=%ld\n", _loop_in_if(5));
    return 0;
}
"#;
    let result = differential_test("loop_in_if", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential loop_in_if test failed: {}",
        result.unwrap_err()
    );
}

// Test 42: collatz_steps
fn build_tmir_collatz_steps_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_collatz_steps", ft_id, BlockId::new(0));
    func.blocks = vec![
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
                    then_target: BlockId::new(5),
                    then_args: vec![],
                    else_target: BlockId::new(1),
                    else_args: vec![ValueId::new(0)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(1),
            params: vec![(ValueId::new(3), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(4)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(2),
                    args: vec![ValueId::new(3), ValueId::new(4)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(2),
            params: vec![(ValueId::new(5), Ty::I64), (ValueId::new(6), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(7)),
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Ne,
                    ty: Ty::I64,
                    lhs: ValueId::new(5),
                    rhs: ValueId::new(7),
                })
                .with_result(ValueId::new(8)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(8),
                    then_target: BlockId::new(3),
                    then_args: vec![ValueId::new(5), ValueId::new(6)],
                    else_target: BlockId::new(6),
                    else_args: vec![ValueId::new(6)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(3),
            params: vec![(ValueId::new(9), Ty::I64), (ValueId::new(10), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(2),
                })
                .with_result(ValueId::new(11)),
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(12)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::SRem,
                    ty: Ty::I64,
                    lhs: ValueId::new(9),
                    rhs: ValueId::new(11),
                })
                .with_result(ValueId::new(13)),
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Eq,
                    ty: Ty::I64,
                    lhs: ValueId::new(13),
                    rhs: ValueId::new(12),
                })
                .with_result(ValueId::new(14)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(14),
                    then_target: BlockId::new(4),
                    then_args: vec![ValueId::new(9), ValueId::new(10)],
                    else_target: BlockId::new(7),
                    else_args: vec![ValueId::new(9), ValueId::new(10)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(4),
            params: vec![(ValueId::new(15), Ty::I64), (ValueId::new(16), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(2),
                })
                .with_result(ValueId::new(17)),
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(18)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::SDiv,
                    ty: Ty::I64,
                    lhs: ValueId::new(15),
                    rhs: ValueId::new(17),
                })
                .with_result(ValueId::new(19)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(16),
                    rhs: ValueId::new(18),
                })
                .with_result(ValueId::new(20)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(2),
                    args: vec![ValueId::new(19), ValueId::new(20)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(5),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(21)),
                InstrNode::new(Inst::Return {
                    values: vec![ValueId::new(21)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(6),
            params: vec![(ValueId::new(22), Ty::I64)],
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(22)],
            })],
        },
        TmirBlock {
            id: BlockId::new(7),
            params: vec![(ValueId::new(23), Ty::I64), (ValueId::new(24), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(3),
                })
                .with_result(ValueId::new(25)),
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(26)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Mul,
                    ty: Ty::I64,
                    lhs: ValueId::new(23),
                    rhs: ValueId::new(25),
                })
                .with_result(ValueId::new(27)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(27),
                    rhs: ValueId::new(26),
                })
                .with_result(ValueId::new(28)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(24),
                    rhs: ValueId::new(26),
                })
                .with_result(ValueId::new(29)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(2),
                    args: vec![ValueId::new(28), ValueId::new(29)],
                }),
            ],
        },
    ];
    module.add_function(func);
    module
}

#[test]
fn test_differential_collatz_steps() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }
    let module = build_tmir_collatz_steps_module();
    let c_reference = r#"
long _collatz_steps(long n) {
    if (n <= 1) {
        return 0;
    }
    long steps = 0;
    while (n != 1) {
        if ((n % 2) == 0) {
            n = n / 2;
        } else {
            n = 3 * n + 1;
        }
        steps = steps + 1;
    }
    return steps;
}
"#;
    let driver = r#"
#include <stdio.h>
extern long _collatz_steps(long n);
int main(void) {
    printf("collatz_steps(1)=%ld\n", _collatz_steps(1));
    printf("collatz_steps(2)=%ld\n", _collatz_steps(2));
    printf("collatz_steps(3)=%ld\n", _collatz_steps(3));
    printf("collatz_steps(7)=%ld\n", _collatz_steps(7));
    return 0;
}
"#;
    let result = differential_test("collatz_steps", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential collatz_steps test failed: {}",
        result.unwrap_err()
    );
}

// Test 43: multi_call_sequence
fn build_tmir_multi_call_sequence_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func =
        TmirFunction::new(FuncId::new(0), "_multi_call_sequence", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::I64)],
        body: vec![
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(1),
            })
            .with_result(ValueId::new(1)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(2),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(3)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(3)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_multi_call_sequence() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }
    let module = build_tmir_multi_call_sequence_module();
    let c_reference = r#"
static long helper(long x) {
    return x + 1;
}

long _multi_call_sequence(long x) {
    return helper(helper(x));
}
"#;
    let driver = r#"
#include <stdio.h>
extern long _multi_call_sequence(long x);
int main(void) {
    printf("multi_call_sequence(0)=%ld\n", _multi_call_sequence(0));
    printf("multi_call_sequence(5)=%ld\n", _multi_call_sequence(5));
    printf("multi_call_sequence(-4)=%ld\n", _multi_call_sequence(-4));
    return 0;
}
"#;
    let result = differential_test("multi_call_sequence", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential multi_call_sequence test failed: {}",
        result.unwrap_err()
    );
}

// Test 44: gcd_euclidean
fn build_tmir_gcd_euclidean_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_gcd_euclidean", ft_id, BlockId::new(0));
    func.blocks = vec![
        TmirBlock {
            id: BlockId::new(0),
            params: vec![(ValueId::new(0), Ty::I64), (ValueId::new(1), Ty::I64)],
            body: vec![InstrNode::new(Inst::Br {
                target: BlockId::new(1),
                args: vec![ValueId::new(0), ValueId::new(1)],
            })],
        },
        TmirBlock {
            id: BlockId::new(1),
            params: vec![(ValueId::new(2), Ty::I64), (ValueId::new(3), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(4)),
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Ne,
                    ty: Ty::I64,
                    lhs: ValueId::new(3),
                    rhs: ValueId::new(4),
                })
                .with_result(ValueId::new(5)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(5),
                    then_target: BlockId::new(2),
                    then_args: vec![ValueId::new(2), ValueId::new(3)],
                    else_target: BlockId::new(3),
                    else_args: vec![ValueId::new(2)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(2),
            params: vec![(ValueId::new(6), Ty::I64), (ValueId::new(7), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::BinOp {
                    op: BinOp::SRem,
                    ty: Ty::I64,
                    lhs: ValueId::new(6),
                    rhs: ValueId::new(7),
                })
                .with_result(ValueId::new(8)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(1),
                    args: vec![ValueId::new(7), ValueId::new(8)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(3),
            params: vec![(ValueId::new(9), Ty::I64)],
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(9)],
            })],
        },
    ];
    module.add_function(func);
    module
}

#[test]
fn test_differential_gcd_euclidean() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }
    let module = build_tmir_gcd_euclidean_module();
    let c_reference = r#"
long _gcd_euclidean(long a, long b) {
    while (b != 0) {
        long t = b;
        b = a % b;
        a = t;
    }
    return a;
}
"#;
    let driver = r#"
#include <stdio.h>
extern long _gcd_euclidean(long a, long b);
int main(void) {
    printf("gcd_euclidean(48,18)=%ld\n", _gcd_euclidean(48, 18));
    printf("gcd_euclidean(1071,462)=%ld\n", _gcd_euclidean(1071, 462));
    printf("gcd_euclidean(7,3)=%ld\n", _gcd_euclidean(7, 3));
    return 0;
}
"#;
    let result = differential_test("gcd_euclidean", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential gcd_euclidean test failed: {}",
        result.unwrap_err()
    );
}

// Test 45: many_live_vars
fn build_tmir_many_live_vars_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_many_live_vars", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::I64), (ValueId::new(1), Ty::I64)],
        body: vec![
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(2),
            })
            .with_result(ValueId::new(3)),
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(3),
            })
            .with_result(ValueId::new(4)),
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(4),
            })
            .with_result(ValueId::new(5)),
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(5),
            })
            .with_result(ValueId::new(6)),
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(6),
            })
            .with_result(ValueId::new(7)),
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(7),
            })
            .with_result(ValueId::new(8)),
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(8),
            })
            .with_result(ValueId::new(9)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(2),
            })
            .with_result(ValueId::new(10)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Sub,
                ty: Ty::I64,
                lhs: ValueId::new(1),
                rhs: ValueId::new(3),
            })
            .with_result(ValueId::new(11)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(4),
            })
            .with_result(ValueId::new(12)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(1),
                rhs: ValueId::new(5),
            })
            .with_result(ValueId::new(13)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Xor,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(6),
            })
            .with_result(ValueId::new(14)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Or,
                ty: Ty::I64,
                lhs: ValueId::new(1),
                rhs: ValueId::new(7),
            })
            .with_result(ValueId::new(15)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(8),
            })
            .with_result(ValueId::new(16)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(1),
                rhs: ValueId::new(9),
            })
            .with_result(ValueId::new(17)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(10),
                rhs: ValueId::new(11),
            })
            .with_result(ValueId::new(18)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(12),
                rhs: ValueId::new(13),
            })
            .with_result(ValueId::new(19)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(14),
                rhs: ValueId::new(15),
            })
            .with_result(ValueId::new(20)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(16),
                rhs: ValueId::new(17),
            })
            .with_result(ValueId::new(21)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(18),
                rhs: ValueId::new(19),
            })
            .with_result(ValueId::new(22)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(20),
                rhs: ValueId::new(21),
            })
            .with_result(ValueId::new(23)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(22),
                rhs: ValueId::new(23),
            })
            .with_result(ValueId::new(24)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(24)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_many_live_vars() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }
    let module = build_tmir_many_live_vars_module();
    let c_reference = r#"
long _many_live_vars(long a, long b) {
    long v1 = a + 1;
    long v2 = b - 2;
    long v3 = a * 3;
    long v4 = b + 4;
    long v5 = a ^ 5;
    long v6 = b | 6;
    long v7 = a & 7;
    long v8 = b + 8;
    return (v1 + v2) + (v3 + v4) + (v5 + v6) + (v7 + v8);
}
"#;
    let driver = r#"
#include <stdio.h>
extern long _many_live_vars(long a, long b);
int main(void) {
    printf("many_live_vars(1,2)=%ld\n", _many_live_vars(1, 2));
    printf("many_live_vars(10,20)=%ld\n", _many_live_vars(10, 20));
    printf("many_live_vars(-3,7)=%ld\n", _many_live_vars(-3, 7));
    return 0;
}
"#;
    let result = differential_test("many_live_vars", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential many_live_vars test failed: {}",
        result.unwrap_err()
    );
}

// Test 46: mixed_width_sext
//
// Tests sign extension of an i32 parameter to i64, then arithmetic on the
// extended value. Uses a single i32 param to avoid the known mixed-width
// ABI bug (i32+i64 params clobber due to w-reg/x-reg aliasing in regalloc).
// Also avoids trunc+sext chains where the original i64 value is still live
// (same regalloc aliasing bug). Both bugs are tracked separately.
fn build_tmir_mixed_width_sext_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_mixed_width_sext", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::I32)],
        body: vec![
            // sext i32 -> i64
            InstrNode::new(Inst::Cast {
                op: CastOp::SExt,
                src_ty: Ty::I32,
                dst_ty: Ty::I64,
                operand: ValueId::new(0),
            })
            .with_result(ValueId::new(1)),
            // multiply by constant 3
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(3),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: ValueId::new(1),
                rhs: ValueId::new(2),
            })
            .with_result(ValueId::new(3)),
            // add constant 100
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(100),
            })
            .with_result(ValueId::new(4)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(3),
                rhs: ValueId::new(4),
            })
            .with_result(ValueId::new(5)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(5)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_mixed_width_sext() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }
    let module = build_tmir_mixed_width_sext_module();
    let c_reference = r#"
long _mixed_width_sext(int x) {
    long extended = (long)x;
    return extended * 3 + 100;
}
"#;
    let driver = r#"
#include <stdio.h>
extern long _mixed_width_sext(int x);
int main(void) {
    printf("mixed_width_sext(7)=%ld\n", _mixed_width_sext(7));
    printf("mixed_width_sext(-3)=%ld\n", _mixed_width_sext(-3));
    printf("mixed_width_sext(0)=%ld\n", _mixed_width_sext(0));
    printf("mixed_width_sext(2147483647)=%ld\n", _mixed_width_sext(2147483647));
    printf("mixed_width_sext(-2147483647)=%ld\n", _mixed_width_sext(-2147483647));
    return 0;
}
"#;
    let result = differential_test("mixed_width_sext", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential mixed_width_sext test failed: {}",
        result.unwrap_err()
    );
}

// Test 47: mixed_width_zext
fn build_tmir_mixed_width_zext_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_mixed_width_zext", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::I32)],
        body: vec![
            InstrNode::new(Inst::Cast {
                op: CastOp::ZExt,
                src_ty: Ty::I32,
                dst_ty: Ty::I64,
                operand: ValueId::new(0),
            })
            .with_result(ValueId::new(1)),
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(1000),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(1),
                rhs: ValueId::new(2),
            })
            .with_result(ValueId::new(3)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(3)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_mixed_width_zext() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }
    let module = build_tmir_mixed_width_zext_module();
    let c_reference = r#"
long _mixed_width_zext(int x) {
    return (long)(unsigned int)x + 1000;
}
"#;
    let driver = r#"
#include <stdio.h>
extern long _mixed_width_zext(int x);
int main(void) {
    printf("mixed_width_zext(0)=%ld\n", _mixed_width_zext(0));
    printf("mixed_width_zext(42)=%ld\n", _mixed_width_zext(42));
    printf("mixed_width_zext(-1)=%ld\n", _mixed_width_zext(-1));
    return 0;
}
"#;
    let result = differential_test("mixed_width_zext", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential mixed_width_zext test failed: {}",
        result.unwrap_err()
    );
}

// Test 48: chain_of_selects
fn build_tmir_chain_of_selects_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_chain_of_selects", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I64),
            (ValueId::new(1), Ty::I64),
            (ValueId::new(2), Ty::I64),
        ],
        body: vec![
            InstrNode::new(Inst::ICmp {
                op: ICmpOp::Sgt,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(3)),
            InstrNode::new(Inst::Select {
                ty: Ty::I64,
                cond: ValueId::new(3),
                then_val: ValueId::new(0),
                else_val: ValueId::new(1),
            })
            .with_result(ValueId::new(4)),
            InstrNode::new(Inst::ICmp {
                op: ICmpOp::Sgt,
                ty: Ty::I64,
                lhs: ValueId::new(4),
                rhs: ValueId::new(2),
            })
            .with_result(ValueId::new(5)),
            InstrNode::new(Inst::Select {
                ty: Ty::I64,
                cond: ValueId::new(5),
                then_val: ValueId::new(4),
                else_val: ValueId::new(2),
            })
            .with_result(ValueId::new(6)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(6)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_chain_of_selects() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }
    let module = build_tmir_chain_of_selects_module();
    let c_reference = r#"
long _chain_of_selects(long a, long b, long c) {
    long ab = (a > b) ? a : b;
    return (ab > c) ? ab : c;
}
"#;
    let driver = r#"
#include <stdio.h>
extern long _chain_of_selects(long a, long b, long c);
int main(void) {
    printf("chain_of_selects(1,2,3)=%ld\n", _chain_of_selects(1, 2, 3));
    printf("chain_of_selects(9,4,7)=%ld\n", _chain_of_selects(9, 4, 7));
    printf("chain_of_selects(-1,-5,-3)=%ld\n", _chain_of_selects(-1, -5, -3));
    return 0;
}
"#;
    let result = differential_test("chain_of_selects", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential chain_of_selects test failed: {}",
        result.unwrap_err()
    );
}

// Test 49: diamond_merge
fn build_tmir_diamond_merge_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_diamond_merge", ft_id, BlockId::new(0));
    func.blocks = vec![
        TmirBlock {
            id: BlockId::new(0),
            params: vec![(ValueId::new(0), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(1)),
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sgt,
                    ty: Ty::I64,
                    lhs: ValueId::new(0),
                    rhs: ValueId::new(1),
                })
                .with_result(ValueId::new(2)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(2),
                    then_target: BlockId::new(1),
                    then_args: vec![ValueId::new(0)],
                    else_target: BlockId::new(2),
                    else_args: vec![ValueId::new(0)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(1),
            params: vec![(ValueId::new(3), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(4)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(3),
                    rhs: ValueId::new(4),
                })
                .with_result(ValueId::new(5)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(3),
                    args: vec![ValueId::new(5)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(2),
            params: vec![(ValueId::new(6), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(7)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Sub,
                    ty: Ty::I64,
                    lhs: ValueId::new(6),
                    rhs: ValueId::new(7),
                })
                .with_result(ValueId::new(8)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(3),
                    args: vec![ValueId::new(8)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(3),
            params: vec![(ValueId::new(9), Ty::I64)],
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(9)],
            })],
        },
    ];
    module.add_function(func);
    module
}

#[test]
fn test_differential_diamond_merge() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }
    let module = build_tmir_diamond_merge_module();
    let c_reference = r#"
long _diamond_merge(long x) {
    long y;
    if (x > 0) {
        y = x + 1;
    } else {
        y = x - 1;
    }
    return y;
}
"#;
    let driver = r#"
#include <stdio.h>
extern long _diamond_merge(long x);
int main(void) {
    printf("diamond_merge(5)=%ld\n", _diamond_merge(5));
    printf("diamond_merge(0)=%ld\n", _diamond_merge(0));
    printf("diamond_merge(-7)=%ld\n", _diamond_merge(-7));
    return 0;
}
"#;
    let result = differential_test("diamond_merge", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential diamond_merge test failed: {}",
        result.unwrap_err()
    );
}

// Test 50: nested_loops
fn build_tmir_nested_loops_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_nested_loops", ft_id, BlockId::new(0));
    func.blocks = vec![
        TmirBlock {
            id: BlockId::new(0),
            params: vec![(ValueId::new(0), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(1)),
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(2)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(1),
                    args: vec![ValueId::new(1), ValueId::new(2)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(1),
            params: vec![(ValueId::new(3), Ty::I64), (ValueId::new(4), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sle,
                    ty: Ty::I64,
                    lhs: ValueId::new(3),
                    rhs: ValueId::new(0),
                })
                .with_result(ValueId::new(5)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(5),
                    then_target: BlockId::new(2),
                    then_args: vec![ValueId::new(3), ValueId::new(4)],
                    else_target: BlockId::new(6),
                    else_args: vec![ValueId::new(4)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(2),
            params: vec![(ValueId::new(6), Ty::I64), (ValueId::new(7), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(8)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(3),
                    args: vec![ValueId::new(6), ValueId::new(8), ValueId::new(7)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(3),
            params: vec![
                (ValueId::new(9), Ty::I64),
                (ValueId::new(10), Ty::I64),
                (ValueId::new(11), Ty::I64),
            ],
            body: vec![
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sle,
                    ty: Ty::I64,
                    lhs: ValueId::new(10),
                    rhs: ValueId::new(9),
                })
                .with_result(ValueId::new(12)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(12),
                    then_target: BlockId::new(4),
                    then_args: vec![ValueId::new(9), ValueId::new(10), ValueId::new(11)],
                    else_target: BlockId::new(5),
                    else_args: vec![ValueId::new(9), ValueId::new(11)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(4),
            params: vec![
                (ValueId::new(13), Ty::I64),
                (ValueId::new(14), Ty::I64),
                (ValueId::new(15), Ty::I64),
            ],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(16)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(15),
                    rhs: ValueId::new(14),
                })
                .with_result(ValueId::new(17)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(14),
                    rhs: ValueId::new(16),
                })
                .with_result(ValueId::new(18)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(3),
                    args: vec![ValueId::new(13), ValueId::new(18), ValueId::new(17)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(5),
            params: vec![(ValueId::new(19), Ty::I64), (ValueId::new(20), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(21)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(19),
                    rhs: ValueId::new(21),
                })
                .with_result(ValueId::new(22)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(1),
                    args: vec![ValueId::new(22), ValueId::new(20)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(6),
            params: vec![(ValueId::new(23), Ty::I64)],
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(23)],
            })],
        },
    ];
    module.add_function(func);
    module
}

#[test]
fn test_differential_nested_loops() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }
    let module = build_tmir_nested_loops_module();
    let c_reference = r#"
long _nested_loops(long n) {
    long outer = 1;
    long sum = 0;
    while (outer <= n) {
        long inner = 1;
        while (inner <= outer) {
            sum = sum + inner;
            inner = inner + 1;
        }
        outer = outer + 1;
    }
    return sum;
}
"#;
    let driver = r#"
#include <stdio.h>
extern long _nested_loops(long n);
int main(void) {
    printf("nested_loops(0)=%ld\n", _nested_loops(0));
    printf("nested_loops(1)=%ld\n", _nested_loops(1));
    printf("nested_loops(3)=%ld\n", _nested_loops(3));
    printf("nested_loops(5)=%ld\n", _nested_loops(5));
    return 0;
}
"#;
    let result = differential_test("nested_loops", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential nested_loops test failed: {}",
        result.unwrap_err()
    );
}
// =============================================================================
// Multi-function differential tests
// =============================================================================

// =============================================================================
// Test 51: Caller-callee basic cross-function call
// =============================================================================

fn build_tmir_caller_callee_basic_module() -> TmirModule {
    let mut module = TmirModule::new("test");

    // Function 0: callee
    let ft_id_0 = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut callee = TmirFunction::new(FuncId::new(0), "_callee", ft_id_0, BlockId::new(0));
    callee.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::I64)],
        body: vec![
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(2),
            })
            .with_result(ValueId::new(1)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(1),
            })
            .with_result(ValueId::new(3)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(2),
                rhs: ValueId::new(3),
            })
            .with_result(ValueId::new(4)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(4)],
            }),
        ],
    }];

    // Function 1: caller
    let ft_id_1 = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut caller = TmirFunction::new(FuncId::new(1), "_multi_caller", ft_id_1, BlockId::new(0));
    caller.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::I64), (ValueId::new(1), Ty::I64)],
        body: vec![
            InstrNode::new(Inst::Call {
                callee: FuncId::new(0),
                args: vec![ValueId::new(0)],
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Call {
                callee: FuncId::new(0),
                args: vec![ValueId::new(1)],
            })
            .with_result(ValueId::new(3)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(2),
                rhs: ValueId::new(3),
            })
            .with_result(ValueId::new(4)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(4)],
            }),
        ],
    }];

    module.add_function(callee);
    module.add_function(caller);
    module
}

#[test]
fn test_differential_caller_callee_basic() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_caller_callee_basic_module();

    let c_reference = r#"
long _callee(long x) {
    return x * 2 + 1;
}

long _multi_caller(long a, long b) {
    return _callee(a) + _callee(b);
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _multi_caller(long a, long b);
int main(void) {
    printf("multi_caller(0,0)=%ld\n", _multi_caller(0, 0));
    printf("multi_caller(1,2)=%ld\n", _multi_caller(1, 2));
    printf("multi_caller(-3,4)=%ld\n", _multi_caller(-3, 4));
    printf("multi_caller(10,20)=%ld\n", _multi_caller(10, 20));
    printf("multi_caller(-7,-8)=%ld\n", _multi_caller(-7, -8));
    return 0;
}
"#;

    let result = differential_test(
        "caller_callee_basic",
        &module,
        c_reference,
        driver,
    );
    assert!(
        result.is_ok(),
        "Differential caller_callee_basic test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 52: Recursive/chained function calls
// =============================================================================

fn build_tmir_recursive_fact_double_module() -> TmirModule {
    let mut module = TmirModule::new("test");

    // Function 0: recursive helper
    let ft_id_0 = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut fact_helper =
        TmirFunction::new(FuncId::new(0), "_fact_helper", ft_id_0, BlockId::new(0));
    fact_helper.blocks = vec![
        TmirBlock {
            id: BlockId::new(0),
            params: vec![(ValueId::new(0), Ty::I64), (ValueId::new(1), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(2)),
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sle,
                    ty: Ty::I64,
                    lhs: ValueId::new(0),
                    rhs: ValueId::new(2),
                })
                .with_result(ValueId::new(3)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(3),
                    then_target: BlockId::new(1),
                    then_args: vec![ValueId::new(1)],
                    else_target: BlockId::new(2),
                    else_args: vec![ValueId::new(0), ValueId::new(1)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(1),
            params: vec![(ValueId::new(8), Ty::I64)],
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(8)],
            })],
        },
        TmirBlock {
            id: BlockId::new(2),
            params: vec![(ValueId::new(9), Ty::I64), (ValueId::new(10), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(11)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Sub,
                    ty: Ty::I64,
                    lhs: ValueId::new(9),
                    rhs: ValueId::new(11),
                })
                .with_result(ValueId::new(12)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Mul,
                    ty: Ty::I64,
                    lhs: ValueId::new(10),
                    rhs: ValueId::new(9),
                })
                .with_result(ValueId::new(13)),
                InstrNode::new(Inst::Call {
                    callee: FuncId::new(0),
                    args: vec![ValueId::new(12), ValueId::new(13)],
                })
                .with_result(ValueId::new(14)),
                InstrNode::new(Inst::Return {
                    values: vec![ValueId::new(14)],
                }),
            ],
        },
    ];

    // Function 1: chained caller
    let ft_id_1 = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut fact_double =
        TmirFunction::new(FuncId::new(1), "_fact_double", ft_id_1, BlockId::new(0));
    fact_double.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::I64)],
        body: vec![
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(1),
            })
            .with_result(ValueId::new(1)),
            InstrNode::new(Inst::Call {
                callee: FuncId::new(0),
                args: vec![ValueId::new(0), ValueId::new(1)],
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(2),
            })
            .with_result(ValueId::new(3)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: ValueId::new(2),
                rhs: ValueId::new(3),
            })
            .with_result(ValueId::new(4)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(4)],
            }),
        ],
    }];

    module.add_function(fact_helper);
    module.add_function(fact_double);
    module
}

#[test]
fn test_differential_recursive_fact_double() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_recursive_fact_double_module();

    let c_reference = r#"
long _fact_helper(long n, long acc) {
    if (n <= 1) {
        return acc;
    } else {
        return _fact_helper(n - 1, acc * n);
    }
}

long _fact_double(long n) {
    return _fact_helper(n, 1) * 2;
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _fact_double(long n);
int main(void) {
    printf("fact_double(0)=%ld\n", _fact_double(0));
    printf("fact_double(1)=%ld\n", _fact_double(1));
    printf("fact_double(2)=%ld\n", _fact_double(2));
    printf("fact_double(3)=%ld\n", _fact_double(3));
    printf("fact_double(5)=%ld\n", _fact_double(5));
    printf("fact_double(8)=%ld\n", _fact_double(8));
    return 0;
}
"#;

    let result = differential_test(
        "recursive_fact_double",
        &module,
        c_reference,
        driver,
    );
    assert!(
        result.is_ok(),
        "Differential recursive_fact_double test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 53: Function with multiple return paths called by another function
// =============================================================================

fn build_tmir_multi_return_module() -> TmirModule {
    let mut module = TmirModule::new("test");

    // Function 0: multi-return callee
    let ft_id_0 = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut multi_return =
        TmirFunction::new(FuncId::new(0), "_multi_return", ft_id_0, BlockId::new(0));
    multi_return.blocks = vec![
        TmirBlock {
            id: BlockId::new(0),
            params: vec![(ValueId::new(0), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(1)),
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Slt,
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
                    else_args: vec![ValueId::new(0)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(1),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(-1),
                })
                .with_result(ValueId::new(3)),
                InstrNode::new(Inst::Return {
                    values: vec![ValueId::new(3)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(2),
            params: vec![(ValueId::new(10), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(11)),
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Eq,
                    ty: Ty::I64,
                    lhs: ValueId::new(10),
                    rhs: ValueId::new(11),
                })
                .with_result(ValueId::new(12)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(12),
                    then_target: BlockId::new(3),
                    then_args: vec![],
                    else_target: BlockId::new(4),
                    else_args: vec![ValueId::new(10)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(3),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(13)),
                InstrNode::new(Inst::Return {
                    values: vec![ValueId::new(13)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(4),
            params: vec![(ValueId::new(20), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(100),
                })
                .with_result(ValueId::new(21)),
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sgt,
                    ty: Ty::I64,
                    lhs: ValueId::new(20),
                    rhs: ValueId::new(21),
                })
                .with_result(ValueId::new(22)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(22),
                    then_target: BlockId::new(5),
                    then_args: vec![],
                    else_target: BlockId::new(6),
                    else_args: vec![ValueId::new(20)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(5),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(100),
                })
                .with_result(ValueId::new(23)),
                InstrNode::new(Inst::Return {
                    values: vec![ValueId::new(23)],
                }),
            ],
        },
        TmirBlock {
            id: BlockId::new(6),
            params: vec![(ValueId::new(30), Ty::I64)],
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(30)],
            })],
        },
    ];

    // Function 1: caller
    let ft_id_1 = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut classify = TmirFunction::new(FuncId::new(1), "_classify", ft_id_1, BlockId::new(0));
    classify.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::I64)],
        body: vec![
            InstrNode::new(Inst::Call {
                callee: FuncId::new(0),
                args: vec![ValueId::new(0)],
            })
            .with_result(ValueId::new(1)),
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(1),
                rhs: ValueId::new(2),
            })
            .with_result(ValueId::new(3)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(3)],
            }),
        ],
    }];

    module.add_function(multi_return);
    module.add_function(classify);
    module
}

#[test]
fn test_differential_multi_return() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_multi_return_module();

    let c_reference = r#"
long _multi_return(long x) {
    if (x < 0) {
        return -1;
    }
    if (x == 0) {
        return 0;
    }
    if (x > 100) {
        return 100;
    }
    return x;
}

long _classify(long x) {
    return _multi_return(x) + 1;
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _classify(long x);
int main(void) {
    printf("classify(-5)=%ld\n", _classify(-5));
    printf("classify(0)=%ld\n", _classify(0));
    printf("classify(7)=%ld\n", _classify(7));
    printf("classify(100)=%ld\n", _classify(100));
    printf("classify(101)=%ld\n", _classify(101));
    printf("classify(500)=%ld\n", _classify(500));
    return 0;
}
"#;

    let result = differential_test("multi_return", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential multi_return test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 54: Multiple helper functions combined
// =============================================================================

fn build_tmir_pair_combine_module() -> TmirModule {
    let mut module = TmirModule::new("test");

    // Function 0: sum helper
    let ft_id_0 = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut pair_sum =
        TmirFunction::new(FuncId::new(0), "_compute_pair_sum", ft_id_0, BlockId::new(0));
    pair_sum.blocks = vec![TmirBlock {
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
    }];

    // Function 1: diff helper
    let ft_id_1 = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut pair_diff =
        TmirFunction::new(FuncId::new(1), "_compute_pair_diff", ft_id_1, BlockId::new(0));
    pair_diff.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::I64), (ValueId::new(1), Ty::I64)],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::Sub,
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

    // Function 2: combine helpers
    let ft_id_2 = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut pair_combine =
        TmirFunction::new(FuncId::new(2), "_pair_combine", ft_id_2, BlockId::new(0));
    pair_combine.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::I64), (ValueId::new(1), Ty::I64)],
        body: vec![
            InstrNode::new(Inst::Call {
                callee: FuncId::new(0),
                args: vec![ValueId::new(0), ValueId::new(1)],
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Call {
                callee: FuncId::new(1),
                args: vec![ValueId::new(0), ValueId::new(1)],
            })
            .with_result(ValueId::new(3)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(2),
                rhs: ValueId::new(3),
            })
            .with_result(ValueId::new(4)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(4)],
            }),
        ],
    }];

    module.add_function(pair_sum);
    module.add_function(pair_diff);
    module.add_function(pair_combine);
    module
}

#[test]
fn test_differential_pair_combine() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_pair_combine_module();

    let c_reference = r#"
long _compute_pair_sum(long a, long b) {
    return a + b;
}

long _compute_pair_diff(long a, long b) {
    return a - b;
}

long _pair_combine(long a, long b) {
    return _compute_pair_sum(a, b) + _compute_pair_diff(a, b);
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _pair_combine(long a, long b);
int main(void) {
    long r0 = _pair_combine(0, 0);
    printf("pair_combine(0,0)=%ld expected=%ld\n", r0, 0L);
    if (r0 != 0L) return 1;

    long r1 = _pair_combine(3, 10);
    printf("pair_combine(3,10)=%ld expected=%ld\n", r1, 6L);
    if (r1 != 6L) return 1;

    long r2 = _pair_combine(-4, 7);
    printf("pair_combine(-4,7)=%ld expected=%ld\n", r2, -8L);
    if (r2 != -8L) return 1;

    long r3 = _pair_combine(123456, -999);
    printf("pair_combine(123456,-999)=%ld expected=%ld\n", r3, 246912L);
    if (r3 != 246912L) return 1;

    return 0;
}
"#;

    let result = differential_test("pair_combine", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential pair_combine test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 55: Function with 8 arguments (full register ABI on AArch64)
//
// On AArch64, x0-x7 are the first 8 integer argument registers.
// This test exercises all 8 register arguments in a cross-function call.
// (9+ args would require stack spill, which is a known limitation tracked
// separately.)
// =============================================================================

fn build_tmir_sum_eight_module() -> TmirModule {
    let mut module = TmirModule::new("test");

    // Function 0: 8-argument callee
    let ft_id_0 = module.add_func_type(FuncTy {
        params: vec![
            Ty::I64, Ty::I64, Ty::I64, Ty::I64,
            Ty::I64, Ty::I64, Ty::I64, Ty::I64,
        ],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut sum_eight = TmirFunction::new(FuncId::new(0), "_sum_eight", ft_id_0, BlockId::new(0));
    sum_eight.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I64),
            (ValueId::new(1), Ty::I64),
            (ValueId::new(2), Ty::I64),
            (ValueId::new(3), Ty::I64),
            (ValueId::new(4), Ty::I64),
            (ValueId::new(5), Ty::I64),
            (ValueId::new(6), Ty::I64),
            (ValueId::new(7), Ty::I64),
        ],
        body: vec![
            // a + b
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(8)),
            // (a+b) + c
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(8),
                rhs: ValueId::new(2),
            })
            .with_result(ValueId::new(9)),
            // (a+b+c) + d
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(9),
                rhs: ValueId::new(3),
            })
            .with_result(ValueId::new(10)),
            // (a+b+c+d) + e
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(10),
                rhs: ValueId::new(4),
            })
            .with_result(ValueId::new(11)),
            // (a+b+c+d+e) + f
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(11),
                rhs: ValueId::new(5),
            })
            .with_result(ValueId::new(12)),
            // (a+b+c+d+e+f) + g
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(12),
                rhs: ValueId::new(6),
            })
            .with_result(ValueId::new(13)),
            // (a+b+c+d+e+f+g) + h
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(13),
                rhs: ValueId::new(7),
            })
            .with_result(ValueId::new(14)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(14)],
            }),
        ],
    }];

    // Function 1: caller that prepares 8 arguments and calls sum_eight
    let ft_id_1 = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut call_sum_eight =
        TmirFunction::new(FuncId::new(1), "_call_sum_eight", ft_id_1, BlockId::new(0));
    call_sum_eight.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::I64)],
        body: vec![
            // x+1
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(1),
            })
            .with_result(ValueId::new(1)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            // x+2
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(2),
            })
            .with_result(ValueId::new(3)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(3),
            })
            .with_result(ValueId::new(4)),
            // x+3
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(3),
            })
            .with_result(ValueId::new(5)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(5),
            })
            .with_result(ValueId::new(6)),
            // x+4
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(4),
            })
            .with_result(ValueId::new(7)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(7),
            })
            .with_result(ValueId::new(8)),
            // x+5
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(5),
            })
            .with_result(ValueId::new(9)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(9),
            })
            .with_result(ValueId::new(10)),
            // x+6
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(6),
            })
            .with_result(ValueId::new(11)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(11),
            })
            .with_result(ValueId::new(12)),
            // x+7
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(7),
            })
            .with_result(ValueId::new(13)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(13),
            })
            .with_result(ValueId::new(14)),
            // call _sum_eight(x, x+1, x+2, x+3, x+4, x+5, x+6, x+7)
            InstrNode::new(Inst::Call {
                callee: FuncId::new(0),
                args: vec![
                    ValueId::new(0),
                    ValueId::new(2),
                    ValueId::new(4),
                    ValueId::new(6),
                    ValueId::new(8),
                    ValueId::new(10),
                    ValueId::new(12),
                    ValueId::new(14),
                ],
            })
            .with_result(ValueId::new(15)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(15)],
            }),
        ],
    }];

    module.add_function(sum_eight);
    module.add_function(call_sum_eight);
    module
}

#[test]
fn test_differential_sum_eight() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_sum_eight_module();

    let c_reference = r#"
long _sum_eight(long a, long b, long c, long d, long e, long f, long g, long h) {
    return a + b + c + d + e + f + g + h;
}

long _call_sum_eight(long x) {
    return _sum_eight(x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7);
}
"#;

    // _call_sum_eight(x) = 8*x + (0+1+2+3+4+5+6+7) = 8*x + 28
    let driver = r#"
#include <stdio.h>
extern long _call_sum_eight(long x);
int main(void) {
    printf("call_sum_eight(0)=%ld\n", _call_sum_eight(0));
    printf("call_sum_eight(1)=%ld\n", _call_sum_eight(1));
    printf("call_sum_eight(5)=%ld\n", _call_sum_eight(5));
    printf("call_sum_eight(-3)=%ld\n", _call_sum_eight(-3));
    printf("call_sum_eight(10)=%ld\n", _call_sum_eight(10));
    printf("call_sum_eight(100)=%ld\n", _call_sum_eight(100));
    return 0;
}
"#;

    let result = differential_test("sum_eight", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential sum_eight test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 56: overflow_add_u32 — unsigned addition wrapping at UINT32_MAX
//
// Tests that LLVM2 and clang agree on unsigned addition overflow behavior.
// =============================================================================

/// `fn _overflow_add(a: u32, b: u32) -> u32 { a + b }`
fn build_tmir_overflow_add_u32_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32, Ty::I32],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_overflow_add", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I32),
            (ValueId::new(1), Ty::I32),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_overflow_add_u32() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_overflow_add_u32_module();

    let c_reference = r#"
unsigned int _overflow_add(unsigned int a, unsigned int b) {
    return a + b;
}
"#;

    let driver = r#"
#include <stdio.h>
extern unsigned int _overflow_add(unsigned int a, unsigned int b);
int main(void) {
    printf("add(4294967295,1)=%u\n", _overflow_add(4294967295u, 1u));
    printf("add(2147483648,2147483648)=%u\n", _overflow_add(2147483648u, 2147483648u));
    printf("add(0,0)=%u\n", _overflow_add(0u, 0u));
    printf("add(100,200)=%u\n", _overflow_add(100u, 200u));
    printf("add(4294967295,4294967295)=%u\n", _overflow_add(4294967295u, 4294967295u));
    printf("add(1,4294967294)=%u\n", _overflow_add(1u, 4294967294u));
    return 0;
}
"#;

    let result = differential_test("overflow_add_u32", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential overflow_add_u32 test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 57: overflow_mul_u32 — unsigned multiplication overflow wrapping
//
// Tests multiply overflow with unsigned semantics (defined behavior).
// =============================================================================

/// `fn _overflow_mul(a: u32, b: u32) -> u32 { a * b }`
fn build_tmir_overflow_mul_u32_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32, Ty::I32],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_overflow_mul", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I32),
            (ValueId::new(1), Ty::I32),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_overflow_mul_u32() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_overflow_mul_u32_module();

    let c_reference = r#"
unsigned int _overflow_mul(unsigned int a, unsigned int b) {
    return a * b;
}
"#;

    let driver = r#"
#include <stdio.h>
extern unsigned int _overflow_mul(unsigned int a, unsigned int b);
int main(void) {
    printf("mul(65536,65536)=%u\n", _overflow_mul(65536u, 65536u));
    printf("mul(4294967295,2)=%u\n", _overflow_mul(4294967295u, 2u));
    printf("mul(100000,100000)=%u\n", _overflow_mul(100000u, 100000u));
    printf("mul(1,4294967295)=%u\n", _overflow_mul(1u, 4294967295u));
    printf("mul(2147483648,2)=%u\n", _overflow_mul(2147483648u, 2u));
    printf("mul(0,4294967295)=%u\n", _overflow_mul(0u, 4294967295u));
    return 0;
}
"#;

    let result = differential_test("overflow_mul_u32", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential overflow_mul_u32 test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 58: overflow_sub_u32 — unsigned subtraction wrapping (borrow)
//
// Tests unsigned subtraction underflow where a < b wraps around.
// =============================================================================

/// `fn _overflow_sub(a: u32, b: u32) -> u32 { a - b }`
fn build_tmir_overflow_sub_u32_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32, Ty::I32],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_overflow_sub", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I32),
            (ValueId::new(1), Ty::I32),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::Sub,
                ty: Ty::I32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_overflow_sub_u32() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_overflow_sub_u32_module();

    let c_reference = r#"
unsigned int _overflow_sub(unsigned int a, unsigned int b) {
    return a - b;
}
"#;

    let driver = r#"
#include <stdio.h>
extern unsigned int _overflow_sub(unsigned int a, unsigned int b);
int main(void) {
    printf("sub(0,1)=%u\n", _overflow_sub(0u, 1u));
    printf("sub(1,2)=%u\n", _overflow_sub(1u, 2u));
    printf("sub(100,50)=%u\n", _overflow_sub(100u, 50u));
    printf("sub(4294967295,4294967295)=%u\n", _overflow_sub(4294967295u, 4294967295u));
    printf("sub(0,4294967295)=%u\n", _overflow_sub(0u, 4294967295u));
    printf("sub(2147483648,1)=%u\n", _overflow_sub(2147483648u, 1u));
    return 0;
}
"#;

    let result = differential_test("overflow_sub_u32", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential overflow_sub_u32 test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 59: switch_basic — 4-case switch statement
//
// Tests the Switch instruction with 3 case arms and a default.
// =============================================================================

/// `fn _switch_basic(x: i32) -> i32 { switch x { 0=>10, 1=>20, 2=>30, default=>99 } }`
fn build_tmir_switch_basic_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_switch_basic", ft_id, BlockId::new(0));
    func.blocks = vec![
        // bb0: entry with switch
        TmirBlock {
            id: BlockId::new(0),
            params: vec![(ValueId::new(0), Ty::I32)],
            body: vec![InstrNode::new(Inst::Switch {
                value: ValueId::new(0),
                default: BlockId::new(4),
                default_args: vec![],
                cases: vec![
                    SwitchCase {
                        value: Constant::Int(0),
                        target: BlockId::new(1),
                        args: vec![],
                    },
                    SwitchCase {
                        value: Constant::Int(1),
                        target: BlockId::new(2),
                        args: vec![],
                    },
                    SwitchCase {
                        value: Constant::Int(2),
                        target: BlockId::new(3),
                        args: vec![],
                    },
                ],
            })],
        },
        // bb1: case 0 -> return 10
        TmirBlock {
            id: BlockId::new(1),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I32,
                    value: Constant::Int(10),
                })
                .with_result(ValueId::new(10)),
                InstrNode::new(Inst::Return {
                    values: vec![ValueId::new(10)],
                }),
            ],
        },
        // bb2: case 1 -> return 20
        TmirBlock {
            id: BlockId::new(2),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I32,
                    value: Constant::Int(20),
                })
                .with_result(ValueId::new(20)),
                InstrNode::new(Inst::Return {
                    values: vec![ValueId::new(20)],
                }),
            ],
        },
        // bb3: case 2 -> return 30
        TmirBlock {
            id: BlockId::new(3),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I32,
                    value: Constant::Int(30),
                })
                .with_result(ValueId::new(30)),
                InstrNode::new(Inst::Return {
                    values: vec![ValueId::new(30)],
                }),
            ],
        },
        // bb4: default -> return 99
        TmirBlock {
            id: BlockId::new(4),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I32,
                    value: Constant::Int(99),
                })
                .with_result(ValueId::new(40)),
                InstrNode::new(Inst::Return {
                    values: vec![ValueId::new(40)],
                }),
            ],
        },
    ];
    module.add_function(func);
    module
}

#[test]
fn test_differential_switch_basic() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_switch_basic_module();

    let c_reference = r#"
int _switch_basic(int x) {
    switch (x) {
        case 0: return 10;
        case 1: return 20;
        case 2: return 30;
        default: return 99;
    }
}
"#;

    let driver = r#"
#include <stdio.h>
extern int _switch_basic(int x);
int main(void) {
    printf("switch(0)=%d\n", _switch_basic(0));
    printf("switch(1)=%d\n", _switch_basic(1));
    printf("switch(2)=%d\n", _switch_basic(2));
    printf("switch(3)=%d\n", _switch_basic(3));
    printf("switch(-1)=%d\n", _switch_basic(-1));
    printf("switch(100)=%d\n", _switch_basic(100));
    printf("switch(2147483647)=%d\n", _switch_basic(2147483647));
    return 0;
}
"#;

    let result = differential_test("switch_basic", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential switch_basic test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 60: nested_select — chained select instructions
//
// Tests two dependent Select instructions computing:
// r1 = (x > 0) ? y : z; result = (y > 0) ? r1 : 0
// =============================================================================

/// `fn _nested_select2(x: i64, y: i64, z: i64) -> i64`
fn build_tmir_nested_select2_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_nested_select2", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I64),
            (ValueId::new(1), Ty::I64),
            (ValueId::new(2), Ty::I64),
        ],
        body: vec![
            // zero constant
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(0),
            })
            .with_result(ValueId::new(3)),
            // cmp1 = x > 0
            InstrNode::new(Inst::ICmp {
                op: ICmpOp::Sgt,
                ty: Ty::I64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(3),
            })
            .with_result(ValueId::new(4)),
            // r1 = cmp1 ? y : z
            InstrNode::new(Inst::Select {
                ty: Ty::I64,
                cond: ValueId::new(4),
                then_val: ValueId::new(1),
                else_val: ValueId::new(2),
            })
            .with_result(ValueId::new(5)),
            // cmp2 = y > 0
            InstrNode::new(Inst::ICmp {
                op: ICmpOp::Sgt,
                ty: Ty::I64,
                lhs: ValueId::new(1),
                rhs: ValueId::new(3),
            })
            .with_result(ValueId::new(6)),
            // result = cmp2 ? r1 : 0
            InstrNode::new(Inst::Select {
                ty: Ty::I64,
                cond: ValueId::new(6),
                then_val: ValueId::new(5),
                else_val: ValueId::new(3),
            })
            .with_result(ValueId::new(7)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(7)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_nested_select2() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_nested_select2_module();

    let c_reference = r#"
long _nested_select2(long x, long y, long z) {
    long r1 = (x > 0) ? y : z;
    return (y > 0) ? r1 : 0;
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _nested_select2(long x, long y, long z);
int main(void) {
    printf("ns2(1,2,3)=%ld\n", _nested_select2(1, 2, 3));
    printf("ns2(-1,2,3)=%ld\n", _nested_select2(-1, 2, 3));
    printf("ns2(1,-2,3)=%ld\n", _nested_select2(1, -2, 3));
    printf("ns2(-1,-2,3)=%ld\n", _nested_select2(-1, -2, 3));
    printf("ns2(0,0,0)=%ld\n", _nested_select2(0, 0, 0));
    printf("ns2(100,-100,50)=%ld\n", _nested_select2(100, -100, 50));
    printf("ns2(1,1,-999)=%ld\n", _nested_select2(1, 1, -999));
    return 0;
}
"#;

    let result = differential_test("nested_select2", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential nested_select2 test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 61: deeply_nested_if — 3 levels of nested conditionals (5 blocks)
//
// Tests deeply nested control flow:
// if a>0 { if b>0 { if c>0 { d } else { c } } else { b } } else { a }
// =============================================================================

/// `fn _deeply_nested(a: i64, b: i64, c: i64, d: i64) -> i64`
fn build_tmir_deeply_nested_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64, Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_deeply_nested", ft_id, BlockId::new(0));
    func.blocks = vec![
        // bb0: if a > 0
        TmirBlock {
            id: BlockId::new(0),
            params: vec![
                (ValueId::new(0), Ty::I64),
                (ValueId::new(1), Ty::I64),
                (ValueId::new(2), Ty::I64),
                (ValueId::new(3), Ty::I64),
            ],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(10)),
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sgt,
                    ty: Ty::I64,
                    lhs: ValueId::new(0),
                    rhs: ValueId::new(10),
                })
                .with_result(ValueId::new(11)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(11),
                    then_target: BlockId::new(1),
                    then_args: vec![],
                    else_target: BlockId::new(4),
                    else_args: vec![ValueId::new(0)],
                }),
            ],
        },
        // bb1: if b > 0
        TmirBlock {
            id: BlockId::new(1),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sgt,
                    ty: Ty::I64,
                    lhs: ValueId::new(1),
                    rhs: ValueId::new(10),
                })
                .with_result(ValueId::new(12)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(12),
                    then_target: BlockId::new(2),
                    then_args: vec![],
                    else_target: BlockId::new(4),
                    else_args: vec![ValueId::new(1)],
                }),
            ],
        },
        // bb2: if c > 0
        TmirBlock {
            id: BlockId::new(2),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sgt,
                    ty: Ty::I64,
                    lhs: ValueId::new(2),
                    rhs: ValueId::new(10),
                })
                .with_result(ValueId::new(13)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(13),
                    then_target: BlockId::new(4),
                    then_args: vec![ValueId::new(3)],
                    else_target: BlockId::new(4),
                    else_args: vec![ValueId::new(2)],
                }),
            ],
        },
        // bb3: unused (numbering gap)
        TmirBlock {
            id: BlockId::new(3),
            params: vec![],
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(10)],
            })],
        },
        // bb4: merge — return result
        TmirBlock {
            id: BlockId::new(4),
            params: vec![(ValueId::new(20), Ty::I64)],
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(20)],
            })],
        },
    ];
    module.add_function(func);
    module
}

#[test]
fn test_differential_deeply_nested_if_v2() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_deeply_nested_module();

    let c_reference = r#"
long _deeply_nested(long a, long b, long c, long d) {
    if (a > 0) {
        if (b > 0) {
            if (c > 0) {
                return d;
            } else {
                return c;
            }
        } else {
            return b;
        }
    } else {
        return a;
    }
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _deeply_nested(long a, long b, long c, long d);
int main(void) {
    printf("dn(1,1,1,42)=%ld\n", _deeply_nested(1, 1, 1, 42));
    printf("dn(-1,5,5,5)=%ld\n", _deeply_nested(-1, 5, 5, 5));
    printf("dn(1,-1,5,5)=%ld\n", _deeply_nested(1, -1, 5, 5));
    printf("dn(1,1,-1,5)=%ld\n", _deeply_nested(1, 1, -1, 5));
    printf("dn(0,0,0,0)=%ld\n", _deeply_nested(0, 0, 0, 0));
    printf("dn(100,200,300,400)=%ld\n", _deeply_nested(100, 200, 300, 400));
    printf("dn(1,1,0,-999)=%ld\n", _deeply_nested(1, 1, 0, -999));
    return 0;
}
"#;

    let result = differential_test("deeply_nested_if_v2", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential deeply_nested_if_v2 test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 62: loop_early_break — loop with conditional early exit
//
// sum=0; for i=1..=n { sum += i; if sum > limit { return sum; } } return sum;
// =============================================================================

/// `fn _loop_early_break(n: i64, limit: i64) -> i64`
fn build_tmir_loop_early_break_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_loop_early_break", ft_id, BlockId::new(0));
    func.blocks = vec![
        // bb0: init sum=0, i=1, br -> bb1
        TmirBlock {
            id: BlockId::new(0),
            params: vec![
                (ValueId::new(0), Ty::I64), // n
                (ValueId::new(1), Ty::I64), // limit
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
                InstrNode::new(Inst::Br {
                    target: BlockId::new(1),
                    args: vec![ValueId::new(2), ValueId::new(3)],
                }),
            ],
        },
        // bb1: loop header — params(sum, i), cmp i <= n
        TmirBlock {
            id: BlockId::new(1),
            params: vec![
                (ValueId::new(10), Ty::I64), // sum
                (ValueId::new(11), Ty::I64), // i
            ],
            body: vec![
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sle,
                    ty: Ty::I64,
                    lhs: ValueId::new(11),
                    rhs: ValueId::new(0),
                })
                .with_result(ValueId::new(12)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(12),
                    then_target: BlockId::new(2),
                    then_args: vec![],
                    else_target: BlockId::new(4),
                    else_args: vec![],
                }),
            ],
        },
        // bb2: loop body — new_sum = sum + i, check if > limit
        TmirBlock {
            id: BlockId::new(2),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(10),
                    rhs: ValueId::new(11),
                })
                .with_result(ValueId::new(20)),
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sgt,
                    ty: Ty::I64,
                    lhs: ValueId::new(20),
                    rhs: ValueId::new(1),
                })
                .with_result(ValueId::new(21)),
                InstrNode::new(Inst::CondBr {
                    cond: ValueId::new(21),
                    then_target: BlockId::new(3),
                    then_args: vec![],
                    else_target: BlockId::new(5),
                    else_args: vec![],
                }),
            ],
        },
        // bb3: early exit — return sum that exceeded limit
        TmirBlock {
            id: BlockId::new(3),
            params: vec![],
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(20)],
            })],
        },
        // bb4: normal exit — return sum (loop completed)
        TmirBlock {
            id: BlockId::new(4),
            params: vec![],
            body: vec![InstrNode::new(Inst::Return {
                values: vec![ValueId::new(10)],
            })],
        },
        // bb5: increment i, loop back
        TmirBlock {
            id: BlockId::new(5),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(1),
                })
                .with_result(ValueId::new(30)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: ValueId::new(11),
                    rhs: ValueId::new(30),
                })
                .with_result(ValueId::new(31)),
                InstrNode::new(Inst::Br {
                    target: BlockId::new(1),
                    args: vec![ValueId::new(20), ValueId::new(31)],
                }),
            ],
        },
    ];
    module.add_function(func);
    module
}

#[test]
fn test_differential_loop_early_break() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_loop_early_break_module();

    let c_reference = r#"
long _loop_early_break(long n, long limit) {
    long sum = 0;
    for (long i = 1; i <= n; i++) {
        sum += i;
        if (sum > limit) return sum;
    }
    return sum;
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _loop_early_break(long n, long limit);
int main(void) {
    printf("leb(10,1000)=%ld\n", _loop_early_break(10, 1000));
    printf("leb(10,15)=%ld\n", _loop_early_break(10, 15));
    printf("leb(100,50)=%ld\n", _loop_early_break(100, 50));
    printf("leb(0,100)=%ld\n", _loop_early_break(0, 100));
    printf("leb(5,100)=%ld\n", _loop_early_break(5, 100));
    printf("leb(1000,0)=%ld\n", _loop_early_break(1000, 0));
    return 0;
}
"#;

    let result = differential_test("loop_early_break", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential loop_early_break test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 63: sum_ten — 10-argument function (stack-passed args)
//
// AArch64 ABI passes first 8 integer args in x0-x7; args 9 and 10 go on stack.
// =============================================================================

/// `fn _sum_ten(a..j: i64) -> i64 { a+b+c+d+e+f+g+h+i+j }`
fn build_tmir_sum_ten_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64; 10],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_sum_ten", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: (0..10u32).map(|i| (ValueId::new(i), Ty::I64)).collect(),
        body: vec![
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(0), rhs: ValueId::new(1) }).with_result(ValueId::new(10)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(10), rhs: ValueId::new(2) }).with_result(ValueId::new(11)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(11), rhs: ValueId::new(3) }).with_result(ValueId::new(12)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(12), rhs: ValueId::new(4) }).with_result(ValueId::new(13)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(13), rhs: ValueId::new(5) }).with_result(ValueId::new(14)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(14), rhs: ValueId::new(6) }).with_result(ValueId::new(15)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(15), rhs: ValueId::new(7) }).with_result(ValueId::new(16)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(16), rhs: ValueId::new(8) }).with_result(ValueId::new(17)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(17), rhs: ValueId::new(9) }).with_result(ValueId::new(18)),
            InstrNode::new(Inst::Return { values: vec![ValueId::new(18)] }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_sum_ten() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_sum_ten_module();

    let c_reference = r#"
long _sum_ten(long a, long b, long c, long d, long e, long f, long g, long h, long i, long j) {
    return a + b + c + d + e + f + g + h + i + j;
}
"#;

    let driver = r#"
#include <stdio.h>
extern long _sum_ten(long a, long b, long c, long d, long e, long f, long g, long h, long i, long j);
int main(void) {
    printf("sum10(1..10)=%ld\n", _sum_ten(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
    printf("sum10(0s)=%ld\n", _sum_ten(0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
    printf("sum10(-1s)=%ld\n", _sum_ten(-1, -2, -3, -4, -5, -6, -7, -8, -9, -10));
    printf("sum10(1_only_j)=%ld\n", _sum_ten(0, 0, 0, 0, 0, 0, 0, 0, 0, 1));
    printf("sum10(large)=%ld\n", _sum_ten(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000));
    printf("sum10(mixed)=%ld\n", _sum_ten(1000000, -1000000, 2000000, -2000000, 3000000, -3000000, 4000000, -4000000, 5000000, -5000000));
    return 0;
}
"#;

    let result = differential_test("sum_ten", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential sum_ten test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 64: fp_special_values — f64 addition with infinity and NaN
//
// Tests IEEE 754 special value handling in floating-point addition.
// =============================================================================

/// `fn _fp_special(a: f64, b: f64) -> f64 { a + b }`
fn build_tmir_fp_special_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::F64, Ty::F64],
        returns: vec![Ty::F64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_fp_special", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::F64),
            (ValueId::new(1), Ty::F64),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::FAdd,
                ty: Ty::F64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_fp_special_values() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_fp_special_module();

    let c_reference = r#"
double _fp_special(double a, double b) {
    return a + b;
}
"#;

    let driver = r#"
#include <stdio.h>
#include <math.h>
extern double _fp_special(double a, double b);
int main(void) {
    double inf = __builtin_inf();
    double nan_val = __builtin_nan("");
    printf("fp(1.0,2.0)=%a\n", _fp_special(1.0, 2.0));
    printf("fp(inf,1.0)=%a\n", _fp_special(inf, 1.0));
    printf("fp(-inf,1.0)=%a\n", _fp_special(-inf, 1.0));
    printf("fp(inf,-inf)=%a\n", _fp_special(inf, -inf));
    printf("fp(nan,1.0)=%a\n", _fp_special(nan_val, 1.0));
    printf("fp(nan,nan)=%a\n", _fp_special(nan_val, nan_val));
    printf("fp(1e308,1e308)=%a\n", _fp_special(1e308, 1e308));
    printf("fp(-0.0,0.0)=%a\n", _fp_special(-0.0, 0.0));
    return 0;
}
"#;

    let result = differential_test("fp_special_values", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential fp_special_values test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 65: fp_mul_special — f64 multiply with denormals and extremes
//
// Tests IEEE 754 edge cases in multiplication.
// =============================================================================

/// `fn _fp_mul_special(a: f64, b: f64) -> f64 { a * b }`
fn build_tmir_fp_mul_special_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::F64, Ty::F64],
        returns: vec![Ty::F64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_fp_mul_special", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::F64),
            (ValueId::new(1), Ty::F64),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::FMul,
                ty: Ty::F64,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_fp_mul_special() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_fp_mul_special_module();

    let c_reference = r#"
double _fp_mul_special(double a, double b) {
    return a * b;
}
"#;

    let driver = r#"
#include <stdio.h>
#include <math.h>
#include <float.h>
extern double _fp_mul_special(double a, double b);
int main(void) {
    double inf = __builtin_inf();
    double nan_val = __builtin_nan("");
    printf("fmul(2.0,3.0)=%a\n", _fp_mul_special(2.0, 3.0));
    printf("fmul(inf,0.0)=%a\n", _fp_mul_special(inf, 0.0));
    printf("fmul(inf,2.0)=%a\n", _fp_mul_special(inf, 2.0));
    printf("fmul(-inf,0.5)=%a\n", _fp_mul_special(-inf, 0.5));
    printf("fmul(nan,1.0)=%a\n", _fp_mul_special(nan_val, 1.0));
    printf("fmul(1e308,2.0)=%a\n", _fp_mul_special(1e308, 2.0));
    printf("fmul(5e-324,0.5)=%a\n", _fp_mul_special(5e-324, 0.5));
    printf("fmul(-0.0,-0.0)=%a\n", _fp_mul_special(-0.0, -0.0));
    printf("fmul(DBL_MIN,DBL_MIN)=%a\n", _fp_mul_special(DBL_MIN, DBL_MIN));
    return 0;
}
"#;

    let result = differential_test("fp_mul_special", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential fp_mul_special test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 66: switch_many_cases — 8-case switch statement
//
// Tests switch with many arms to stress the lowering.
// =============================================================================

/// `fn _switch_many(x: i32) -> i32 { switch x { 0=>0, 1=>11, ..., 7=>77, default=>-1 } }`
fn build_tmir_switch_many_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_switch_many", ft_id, BlockId::new(0));

    let mut blocks = vec![
        TmirBlock {
            id: BlockId::new(0),
            params: vec![(ValueId::new(0), Ty::I32)],
            body: vec![InstrNode::new(Inst::Switch {
                value: ValueId::new(0),
                default: BlockId::new(9),
                default_args: vec![],
                cases: (0..8i128)
                    .map(|i| SwitchCase {
                        value: Constant::Int(i),
                        target: BlockId::new((i + 1) as u32),
                        args: vec![],
                    })
                    .collect(),
            })],
        },
    ];

    // bb1..bb8: return i*10+i (0, 11, 22, 33, 44, 55, 66, 77)
    for i in 0..8i128 {
        let result_val = i * 10 + i;
        blocks.push(TmirBlock {
            id: BlockId::new((i + 1) as u32),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I32,
                    value: Constant::Int(result_val),
                })
                .with_result(ValueId::new((100 + i) as u32)),
                InstrNode::new(Inst::Return {
                    values: vec![ValueId::new((100 + i) as u32)],
                }),
            ],
        });
    }

    // bb9: default -> return -1
    blocks.push(TmirBlock {
        id: BlockId::new(9),
        params: vec![],
        body: vec![
            InstrNode::new(Inst::Const {
                ty: Ty::I32,
                value: Constant::Int(-1),
            })
            .with_result(ValueId::new(200)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(200)],
            }),
        ],
    });

    func.blocks = blocks;
    module.add_function(func);
    module
}

#[test]
fn test_differential_switch_many_cases() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_switch_many_module();

    let c_reference = r#"
int _switch_many(int x) {
    switch (x) {
        case 0: return 0;
        case 1: return 11;
        case 2: return 22;
        case 3: return 33;
        case 4: return 44;
        case 5: return 55;
        case 6: return 66;
        case 7: return 77;
        default: return -1;
    }
}
"#;

    let driver = r#"
#include <stdio.h>
extern int _switch_many(int x);
int main(void) {
    printf("sw(0)=%d\n", _switch_many(0));
    printf("sw(1)=%d\n", _switch_many(1));
    printf("sw(2)=%d\n", _switch_many(2));
    printf("sw(3)=%d\n", _switch_many(3));
    printf("sw(4)=%d\n", _switch_many(4));
    printf("sw(5)=%d\n", _switch_many(5));
    printf("sw(6)=%d\n", _switch_many(6));
    printf("sw(7)=%d\n", _switch_many(7));
    printf("sw(8)=%d\n", _switch_many(8));
    printf("sw(-1)=%d\n", _switch_many(-1));
    printf("sw(100)=%d\n", _switch_many(100));
    return 0;
}
"#;

    let result = differential_test("switch_many_cases", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential switch_many_cases test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 67: icmp_boundary — integer comparison at boundary values
//
// Tests comparisons at INT64_MIN, INT64_MAX, and zero boundaries.
// Returns encoded flags: 1=lt, 2=eq, 4=gt.
// =============================================================================

/// `fn _icmp_boundary(a: i64, b: i64) -> i64 { lt + eq*2 + gt*4 }`
fn build_tmir_icmp_boundary_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_icmp_boundary", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I64),
            (ValueId::new(1), Ty::I64),
        ],
        body: vec![
            InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(0) }).with_result(ValueId::new(2)),
            InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(1) }).with_result(ValueId::new(3)),
            InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(2) }).with_result(ValueId::new(4)),
            InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(4) }).with_result(ValueId::new(5)),
            // lt
            InstrNode::new(Inst::ICmp { op: ICmpOp::Slt, ty: Ty::I64, lhs: ValueId::new(0), rhs: ValueId::new(1) }).with_result(ValueId::new(10)),
            InstrNode::new(Inst::Select { ty: Ty::I64, cond: ValueId::new(10), then_val: ValueId::new(3), else_val: ValueId::new(2) }).with_result(ValueId::new(11)),
            // eq
            InstrNode::new(Inst::ICmp { op: ICmpOp::Eq, ty: Ty::I64, lhs: ValueId::new(0), rhs: ValueId::new(1) }).with_result(ValueId::new(12)),
            InstrNode::new(Inst::Select { ty: Ty::I64, cond: ValueId::new(12), then_val: ValueId::new(4), else_val: ValueId::new(2) }).with_result(ValueId::new(13)),
            // gt
            InstrNode::new(Inst::ICmp { op: ICmpOp::Sgt, ty: Ty::I64, lhs: ValueId::new(0), rhs: ValueId::new(1) }).with_result(ValueId::new(14)),
            InstrNode::new(Inst::Select { ty: Ty::I64, cond: ValueId::new(14), then_val: ValueId::new(5), else_val: ValueId::new(2) }).with_result(ValueId::new(15)),
            // sum
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(11), rhs: ValueId::new(13) }).with_result(ValueId::new(16)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(16), rhs: ValueId::new(15) }).with_result(ValueId::new(17)),
            InstrNode::new(Inst::Return { values: vec![ValueId::new(17)] }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_icmp_boundary() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_icmp_boundary_module();

    let c_reference = r#"
long _icmp_boundary(long a, long b) {
    long lt = (a < b) ? 1 : 0;
    long eq = (a == b) ? 2 : 0;
    long gt = (a > b) ? 4 : 0;
    return lt + eq + gt;
}
"#;

    let driver = r#"
#include <stdio.h>
#include <limits.h>
extern long _icmp_boundary(long a, long b);
int main(void) {
    printf("cmp(0,0)=%ld\n", _icmp_boundary(0, 0));
    printf("cmp(1,0)=%ld\n", _icmp_boundary(1, 0));
    printf("cmp(0,1)=%ld\n", _icmp_boundary(0, 1));
    printf("cmp(-1,0)=%ld\n", _icmp_boundary(-1, 0));
    printf("cmp(0,-1)=%ld\n", _icmp_boundary(0, -1));
    printf("cmp(LONG_MAX,LONG_MAX)=%ld\n", _icmp_boundary(LONG_MAX, LONG_MAX));
    printf("cmp(LONG_MIN,LONG_MIN)=%ld\n", _icmp_boundary(LONG_MIN, LONG_MIN));
    printf("cmp(LONG_MIN,LONG_MAX)=%ld\n", _icmp_boundary(LONG_MIN, LONG_MAX));
    printf("cmp(LONG_MAX,LONG_MIN)=%ld\n", _icmp_boundary(LONG_MAX, LONG_MIN));
    printf("cmp(-1,1)=%ld\n", _icmp_boundary(-1, 1));
    return 0;
}
"#;

    let result = differential_test("icmp_boundary", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential icmp_boundary test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 68: unsigned_comparisons — Ugt/Ult at sign boundary
//
// Tests unsigned comparisons where signed/unsigned interpretations differ.
// =============================================================================

/// `fn _ucmp_classify(a: i64, b: i64) -> i64` (1=ult, 2=eq, 4=ugt)
fn build_tmir_ucmp_classify_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I64, Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_ucmp_classify", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::I64),
            (ValueId::new(1), Ty::I64),
        ],
        body: vec![
            InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(0) }).with_result(ValueId::new(2)),
            InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(1) }).with_result(ValueId::new(3)),
            InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(2) }).with_result(ValueId::new(4)),
            InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(4) }).with_result(ValueId::new(5)),
            // ult
            InstrNode::new(Inst::ICmp { op: ICmpOp::Ult, ty: Ty::I64, lhs: ValueId::new(0), rhs: ValueId::new(1) }).with_result(ValueId::new(10)),
            InstrNode::new(Inst::Select { ty: Ty::I64, cond: ValueId::new(10), then_val: ValueId::new(3), else_val: ValueId::new(2) }).with_result(ValueId::new(11)),
            // eq
            InstrNode::new(Inst::ICmp { op: ICmpOp::Eq, ty: Ty::I64, lhs: ValueId::new(0), rhs: ValueId::new(1) }).with_result(ValueId::new(12)),
            InstrNode::new(Inst::Select { ty: Ty::I64, cond: ValueId::new(12), then_val: ValueId::new(4), else_val: ValueId::new(2) }).with_result(ValueId::new(13)),
            // ugt
            InstrNode::new(Inst::ICmp { op: ICmpOp::Ugt, ty: Ty::I64, lhs: ValueId::new(0), rhs: ValueId::new(1) }).with_result(ValueId::new(14)),
            InstrNode::new(Inst::Select { ty: Ty::I64, cond: ValueId::new(14), then_val: ValueId::new(5), else_val: ValueId::new(2) }).with_result(ValueId::new(15)),
            // sum
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(11), rhs: ValueId::new(13) }).with_result(ValueId::new(16)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(16), rhs: ValueId::new(15) }).with_result(ValueId::new(17)),
            InstrNode::new(Inst::Return { values: vec![ValueId::new(17)] }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_unsigned_comparisons() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_ucmp_classify_module();

    let c_reference = r#"
long _ucmp_classify(long a, long b) {
    unsigned long ua = (unsigned long)a;
    unsigned long ub = (unsigned long)b;
    long lt = (ua < ub) ? 1 : 0;
    long eq = (ua == ub) ? 2 : 0;
    long gt = (ua > ub) ? 4 : 0;
    return lt + eq + gt;
}
"#;

    let driver = r#"
#include <stdio.h>
#include <limits.h>
extern long _ucmp_classify(long a, long b);
int main(void) {
    printf("ucmp(0,0)=%ld\n", _ucmp_classify(0, 0));
    printf("ucmp(1,0)=%ld\n", _ucmp_classify(1, 0));
    printf("ucmp(0,1)=%ld\n", _ucmp_classify(0, 1));
    printf("ucmp(-1,0)=%ld\n", _ucmp_classify(-1, 0));
    printf("ucmp(0,-1)=%ld\n", _ucmp_classify(0, -1));
    printf("ucmp(-1,-1)=%ld\n", _ucmp_classify(-1, -1));
    printf("ucmp(-1,1)=%ld\n", _ucmp_classify(-1, 1));
    printf("ucmp(LONG_MAX,LONG_MIN)=%ld\n", _ucmp_classify(LONG_MAX, LONG_MIN));
    printf("ucmp(LONG_MIN,LONG_MAX)=%ld\n", _ucmp_classify(LONG_MIN, LONG_MAX));
    return 0;
}
"#;

    let result = differential_test("unsigned_comparisons", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential unsigned_comparisons test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 69: call_with_stack_args — caller passes 10 args (2 on stack)
//
// Tests that function CALLS with >8 args correctly pass stack arguments.
// =============================================================================

/// Module with _sum_ten_impl(10 args) and _call_ten(x) that calls it with x..x+9
fn build_tmir_call_with_stack_args_module() -> TmirModule {
    let mut module = TmirModule::new("test");

    let ft_sum = module.add_func_type(FuncTy {
        params: vec![Ty::I64; 10],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let ft_call = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });

    // _sum_ten_impl: sum all 10 args
    let mut sum_fn = TmirFunction::new(FuncId::new(0), "_sum_ten_impl", ft_sum, BlockId::new(0));
    sum_fn.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: (0..10u32).map(|i| (ValueId::new(i), Ty::I64)).collect(),
        body: vec![
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(0), rhs: ValueId::new(1) }).with_result(ValueId::new(10)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(10), rhs: ValueId::new(2) }).with_result(ValueId::new(11)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(11), rhs: ValueId::new(3) }).with_result(ValueId::new(12)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(12), rhs: ValueId::new(4) }).with_result(ValueId::new(13)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(13), rhs: ValueId::new(5) }).with_result(ValueId::new(14)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(14), rhs: ValueId::new(6) }).with_result(ValueId::new(15)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(15), rhs: ValueId::new(7) }).with_result(ValueId::new(16)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(16), rhs: ValueId::new(8) }).with_result(ValueId::new(17)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(17), rhs: ValueId::new(9) }).with_result(ValueId::new(18)),
            InstrNode::new(Inst::Return { values: vec![ValueId::new(18)] }),
        ],
    }];

    // _call_ten: compute x+0..x+9 and call _sum_ten_impl
    let mut call_fn = TmirFunction::new(FuncId::new(1), "_call_ten", ft_call, BlockId::new(0));
    call_fn.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::I64)],
        body: vec![
            InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(1) }).with_result(ValueId::new(50)),
            InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(2) }).with_result(ValueId::new(51)),
            InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(3) }).with_result(ValueId::new(52)),
            InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(4) }).with_result(ValueId::new(53)),
            InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(5) }).with_result(ValueId::new(54)),
            InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(6) }).with_result(ValueId::new(55)),
            InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(7) }).with_result(ValueId::new(56)),
            InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(8) }).with_result(ValueId::new(57)),
            InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(9) }).with_result(ValueId::new(58)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(0), rhs: ValueId::new(50) }).with_result(ValueId::new(60)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(0), rhs: ValueId::new(51) }).with_result(ValueId::new(61)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(0), rhs: ValueId::new(52) }).with_result(ValueId::new(62)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(0), rhs: ValueId::new(53) }).with_result(ValueId::new(63)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(0), rhs: ValueId::new(54) }).with_result(ValueId::new(64)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(0), rhs: ValueId::new(55) }).with_result(ValueId::new(65)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(0), rhs: ValueId::new(56) }).with_result(ValueId::new(66)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(0), rhs: ValueId::new(57) }).with_result(ValueId::new(67)),
            InstrNode::new(Inst::BinOp { op: BinOp::Add, ty: Ty::I64, lhs: ValueId::new(0), rhs: ValueId::new(58) }).with_result(ValueId::new(68)),
            // Call with 10 args: x, x+1, ..., x+9
            InstrNode::new(Inst::Call {
                callee: FuncId::new(0),
                args: vec![
                    ValueId::new(0), ValueId::new(60), ValueId::new(61),
                    ValueId::new(62), ValueId::new(63), ValueId::new(64),
                    ValueId::new(65), ValueId::new(66), ValueId::new(67),
                    ValueId::new(68),
                ],
            })
            .with_result(ValueId::new(70)),
            InstrNode::new(Inst::Return { values: vec![ValueId::new(70)] }),
        ],
    }];

    module.add_function(sum_fn);
    module.add_function(call_fn);
    module
}

#[test]
fn test_differential_call_with_stack_args() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_call_with_stack_args_module();

    let c_reference = r#"
long _sum_ten_impl(long a, long b, long c, long d, long e, long f, long g, long h, long i, long j) {
    return a + b + c + d + e + f + g + h + i + j;
}

long _call_ten(long x) {
    return _sum_ten_impl(x, x+1, x+2, x+3, x+4, x+5, x+6, x+7, x+8, x+9);
}
"#;

    // _call_ten(x) = 10*x + (0+1+2+3+4+5+6+7+8+9) = 10*x + 45
    let driver = r#"
#include <stdio.h>
extern long _call_ten(long x);
int main(void) {
    printf("call_ten(0)=%ld\n", _call_ten(0));
    printf("call_ten(1)=%ld\n", _call_ten(1));
    printf("call_ten(10)=%ld\n", _call_ten(10));
    printf("call_ten(-5)=%ld\n", _call_ten(-5));
    printf("call_ten(100)=%ld\n", _call_ten(100));
    printf("call_ten(-1000000)=%ld\n", _call_ten(-1000000));
    return 0;
}
"#;

    let result = differential_test("call_with_stack_args", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential call_with_stack_args test failed: {}",
        result.unwrap_err()
    );
}

// =============================================================================
// Test 70: fp32_mul_add — f32 multiply-add sequence
//
// Tests f32 arithmetic path: (a * b) + c.
// =============================================================================

/// `fn _fp32_mul_add(a: f32, b: f32, c: f32) -> f32 { a * b + c }`
fn build_tmir_fp32_mul_add_module() -> TmirModule {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::F32, Ty::F32, Ty::F32],
        returns: vec![Ty::F32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_fp32_mul_add", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![
            (ValueId::new(0), Ty::F32),
            (ValueId::new(1), Ty::F32),
            (ValueId::new(2), Ty::F32),
        ],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::FMul,
                ty: Ty::F32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(3)),
            InstrNode::new(Inst::BinOp {
                op: BinOp::FAdd,
                ty: Ty::F32,
                lhs: ValueId::new(3),
                rhs: ValueId::new(2),
            })
            .with_result(ValueId::new(4)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(4)],
            }),
        ],
    }];
    module.add_function(func);
    module
}

#[test]
fn test_differential_fp32_mul_add() {
    if !is_aarch64() || !has_cc() {
        eprintln!("Skipping differential test: not AArch64 or cc not available");
        return;
    }

    let module = build_tmir_fp32_mul_add_module();

    let c_reference = r#"
float _fp32_mul_add(float a, float b, float c) {
    return a * b + c;
}
"#;

    let driver = r#"
#include <stdio.h>
extern float _fp32_mul_add(float a, float b, float c);
int main(void) {
    printf("fma32(2,3,4)=%a\n", (double)_fp32_mul_add(2.0f, 3.0f, 4.0f));
    printf("fma32(0,1,1)=%a\n", (double)_fp32_mul_add(0.0f, 1.0f, 1.0f));
    printf("fma32(-1,2,3)=%a\n", (double)_fp32_mul_add(-1.0f, 2.0f, 3.0f));
    printf("fma32(1e38,2,0)=%a\n", (double)_fp32_mul_add(1e38f, 2.0f, 0.0f));
    printf("fma32(1e-38,1e-38,0)=%a\n", (double)_fp32_mul_add(1e-38f, 1e-38f, 0.0f));
    printf("fma32(0.1,0.2,0.3)=%a\n", (double)_fp32_mul_add(0.1f, 0.2f, 0.3f));
    return 0;
}
"#;

    let result = differential_test("fp32_mul_add", &module, c_reference, driver);
    assert!(
        result.is_ok(),
        "Differential fp32_mul_add test failed: {}",
        result.unwrap_err()
    );
}

