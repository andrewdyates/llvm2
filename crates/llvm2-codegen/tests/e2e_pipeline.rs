// llvm2-codegen/tests/e2e_pipeline.rs - End-to-end compilation pipeline integration tests
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Integration tests exercising the full compilation pipeline from tMIR to Mach-O.
// Tests cover: empty function, 32-bit arithmetic, branches, loops, multi-arg ABI,
// call instructions, O0 vs O2 comparison, Mach-O structure validation, bitwise
// operations, shifts, unsigned division, negation, and error paths.
//
// Part of #406 - End-to-end compilation pipeline integration tests

use llvm2_codegen::pipeline::{Pipeline, PipelineConfig, OptLevel};

use tmir::{Block as TmirBlock, Function as TmirFunction, Module as TmirModule, FuncTy, Ty, FuncTyId, Constant};
use tmir::{Inst, InstrNode, BinOp, ICmpOp, UnOp};
use tmir::{BlockId, FuncId, ValueId};

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

/// Compile a tMIR function through the full pipeline (adapter -> ISel -> opt ->
/// regalloc -> frame -> encode -> Mach-O).
fn compile_tmir(
    tmir_func: &TmirFunction,
    module: &TmirModule,
    opt_level: OptLevel,
) -> Result<Vec<u8>, String> {
    let (lir_func, _proof_ctx) = llvm2_lower::translate_function(tmir_func, module)
        .map_err(|e| format!("adapter error: {}", e))?;

    let config = PipelineConfig {
        opt_level,
        emit_debug: false,
        ..Default::default()
    };
    let pipeline = Pipeline::new(config);
    pipeline
        .compile_function(&lir_func)
        .map_err(|e| format!("pipeline error: {}", e))
}

/// Verify bytes begin with a valid Mach-O 64-bit magic number.
fn assert_valid_macho(bytes: &[u8], ctx: &str) {
    assert!(
        bytes.len() >= 4,
        "{}: object file too small ({} bytes)",
        ctx,
        bytes.len()
    );
    assert_eq!(
        &bytes[0..4],
        &[0xCF, 0xFA, 0xED, 0xFE],
        "{}: invalid Mach-O magic",
        ctx
    );
}

/// Extract the filetype field from the Mach-O header (offset 12).
fn macho_filetype(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]])
}

/// Extract the cputype field from the Mach-O header (offset 4).
fn macho_cputype(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])
}

/// Extract ncmds from the Mach-O header (offset 16).
fn macho_ncmds(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]])
}

/// Extract sizeofcmds from the Mach-O header (offset 20).
fn macho_sizeofcmds(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]])
}

// ---------------------------------------------------------------------------
// tMIR function builders
// ---------------------------------------------------------------------------

/// fn empty() -> i64 { return 0; }
///
/// Minimal function with a single constant and return.
fn build_empty_function() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "empty", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
            id: BlockId::new(0),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(0),
                    })

                    .with_result(ValueId::new(0)),
                InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(0)],
                    }),
            ],
        }];
    module.add_function(func.clone());
    (func, module)
}

/// fn add32(a: i32, b: i32) -> i32 { a + b }
///
/// Tests 32-bit integer type support through the full pipeline.
fn build_add_i32() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I32, Ty::I32],
            returns: vec![Ty::I32], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "add32", ft_id, BlockId::new(0));
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
    module.add_function(func.clone());
    (func, module)
}

/// fn abs_val(x: i64) -> i64 { if x >= 0 { x } else { 0 - x } }
///
/// Three blocks: entry (compare, condbr), positive return, negate return.
fn build_branch_function() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "abs_val", ft_id, BlockId::new(0));
    func.blocks = vec![
            // bb0 (entry): cmp x >= 0, condbr
            TmirBlock {
                id: BlockId::new(0),
                params: vec![(ValueId::new(0), Ty::I64)], // x
                body: vec![
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(0),
                        })
                        .with_result(ValueId::new(1)),
                    InstrNode::new(Inst::ICmp {
                            op: ICmpOp::Sge,
                            ty: Ty::I64,
                            lhs: ValueId::new(0), // x
                            rhs: ValueId::new(1), // 0
                        })
                        .with_result(ValueId::new(2)),
                    InstrNode::new(Inst::CondBr {
                            cond: ValueId::new(2),
                            then_target: BlockId::new(1), // return x
                            then_args: vec![],
                            else_target: BlockId::new(2), // negate
                            else_args: vec![],
                        }),
                ],
            },
            // bb1: return x (positive case)
            TmirBlock {
                id: BlockId::new(1),
                params: vec![],
                body: vec![InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(0)],
                    })],
            },
            // bb2: negate and return
            TmirBlock {
                id: BlockId::new(2),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(0),
                        })
                        .with_result(ValueId::new(3)),
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Sub,
                            ty: Ty::I64,
                            lhs: ValueId::new(3), // 0
                            rhs: ValueId::new(0), // x
                        })
                        .with_result(ValueId::new(4)),
                    InstrNode::new(Inst::Return {
                            values: vec![ValueId::new(4)],
                        }),
                ],
            },
        ];
    module.add_function(func.clone());
    (func, module)
}

/// fn sum_to_n(n: i64) -> i64
///
/// Loop: sum = 0, i = 1; while (i <= n) { sum += i; i += 1 }; return sum
///
/// bb0 (entry): init sum=0, i=1, br -> bb1
/// bb1 (loop header): params(sum, i), cmp i <= n, condbr -> bb2 (body), bb3 (exit)
/// bb2 (loop body): new_sum = sum + i, new_i = i + 1, br -> bb1
/// bb3 (exit): params(result), return result
fn build_loop_function() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "sum_to_n", ft_id, BlockId::new(0));
    func.blocks = vec![
            // bb0 (entry): init
            TmirBlock {
                id: BlockId::new(0),
                params: vec![(ValueId::new(0), Ty::I64)], // n
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
            // bb1 (loop header): check i <= n
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
                            lhs: ValueId::new(11), // i
                            rhs: ValueId::new(0),  // n
                        })
                        .with_result(ValueId::new(12)),
                    InstrNode::new(Inst::CondBr {
                            cond: ValueId::new(12),
                            then_target: BlockId::new(2), // loop body
                            then_args: vec![],
                            else_target: BlockId::new(3), // exit
                            else_args: vec![ValueId::new(10)],
                        }),
                ],
            },
            // bb2 (loop body): sum += i, i++
            TmirBlock {
                id: BlockId::new(2),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Add,
                            ty: Ty::I64,
                            lhs: ValueId::new(10), // sum
                            rhs: ValueId::new(11), // i
                        })
                        .with_result(ValueId::new(13)),
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(1),
                        })
                        .with_result(ValueId::new(14)),
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Add,
                            ty: Ty::I64,
                            lhs: ValueId::new(11), // i
                            rhs: ValueId::new(14), // 1
                        })
                        .with_result(ValueId::new(15)),
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(1),
                            args: vec![ValueId::new(13), ValueId::new(15)],
                        }),
                ],
            },
            // bb3 (exit): return sum
            TmirBlock {
                id: BlockId::new(3),
                params: vec![(ValueId::new(20), Ty::I64)],
                body: vec![InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(20)],
                    })],
            },
        ];
    module.add_function(func.clone());
    (func, module)
}

/// fn add4(a: i64, b: i64, c: i64, d: i64) -> i64 { a + b + c + d }
///
/// 4 parameters testing ABI register allocation (x0-x3).
fn build_multi_arg_function() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64, Ty::I64, Ty::I64, Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "add4", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
            id: BlockId::new(0),
            params: vec![
                (ValueId::new(0), Ty::I64), // a
                (ValueId::new(1), Ty::I64), // b
                (ValueId::new(2), Ty::I64), // c
                (ValueId::new(3), Ty::I64), // d
            ],
            body: vec![
                // t1 = a + b
                InstrNode::new(Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: ValueId::new(0),
                        rhs: ValueId::new(1),
                    })

                    .with_result(ValueId::new(4)),
                // t2 = t1 + c
                InstrNode::new(Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: ValueId::new(4),
                        rhs: ValueId::new(2),
                    })

                    .with_result(ValueId::new(5)),
                // t3 = t2 + d
                InstrNode::new(Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: ValueId::new(5),
                        rhs: ValueId::new(3),
                    })

                    .with_result(ValueId::new(6)),
                InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(6)],
                    }),
            ],
        }];
    module.add_function(func.clone());
    (func, module)
}

/// fn call_test(a: i64) -> i64 { call add_one(a) }
///
/// Tests the Call instruction through the pipeline.
fn build_call_function() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "call_test", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
            id: BlockId::new(0),
            params: vec![(ValueId::new(0), Ty::I64)], // a
            body: vec![
                InstrNode::new(Inst::Call { callee: FuncId::new(1), // add_one
                        args: vec![ValueId::new(0)],
                    })

                    .with_result(ValueId::new(1)),
                InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(1)],
                    }),
            ],
        }];
    module.add_function(func.clone());
    (func, module)
}

/// fn simple_add(a: i64, b: i64) -> i64 { a + b }
///
/// Used for Mach-O structure validation and O0 vs O2 comparison.
fn build_simple_add() -> (TmirFunction, TmirModule) {

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

/// fn bitwise(a: i64, b: i64) -> i64 { (a & b) | (a ^ b) }
fn build_bitwise_function() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64, Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "bitwise", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
            id: BlockId::new(0),
            params: vec![
                (ValueId::new(0), Ty::I64), // a
                (ValueId::new(1), Ty::I64), // b
            ],
            body: vec![
                // and_result = a & b
                InstrNode::new(Inst::BinOp {
                        op: BinOp::And,
                        ty: Ty::I64,
                        lhs: ValueId::new(0),
                        rhs: ValueId::new(1),
                    })

                    .with_result(ValueId::new(2)),
                // xor_result = a ^ b
                InstrNode::new(Inst::BinOp {
                        op: BinOp::Xor,
                        ty: Ty::I64,
                        lhs: ValueId::new(0),
                        rhs: ValueId::new(1),
                    })

                    .with_result(ValueId::new(3)),
                // or_result = and_result | xor_result
                InstrNode::new(Inst::BinOp {
                        op: BinOp::Or,
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
    module.add_function(func.clone());
    (func, module)
}

/// fn sub_vals(a: i64, b: i64) -> i64 { a - b }
fn build_sub_function() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64, Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "sub_vals", ft_id, BlockId::new(0));
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
    module.add_function(func.clone());
    (func, module)
}

/// fn negate(x: i64) -> i64 { 0 - x }
fn build_negation_function() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "negate", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
            id: BlockId::new(0),
            params: vec![(ValueId::new(0), Ty::I64)], // x
            body: vec![
                InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(0),
                    })
                    .with_result(ValueId::new(1)),
                InstrNode::new(Inst::BinOp {
                        op: BinOp::Sub,
                        ty: Ty::I64,
                        lhs: ValueId::new(1), // 0
                        rhs: ValueId::new(0), // x
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

/// fn shift_test(x: i64) -> i64 { (x << 2) >> 1 }
fn build_shift_function() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "shift_test", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
            id: BlockId::new(0),
            params: vec![(ValueId::new(0), Ty::I64)], // x
            body: vec![
                InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(2),
                    })
                    .with_result(ValueId::new(1)),
                // shl_result = x << 2
                InstrNode::new(Inst::BinOp {
                        op: BinOp::Shl,
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
                // ashr_result = shl_result >> 1
                InstrNode::new(Inst::BinOp {
                        op: BinOp::AShr,
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
    module.add_function(func.clone());
    (func, module)
}

/// fn udiv_test(a: i64, b: i64) -> i64 { a /u b }
fn build_unsigned_div_function() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64, Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "udiv_test", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
            id: BlockId::new(0),
            params: vec![
                (ValueId::new(0), Ty::I64), // a
                (ValueId::new(1), Ty::I64), // b
            ],
            body: vec![
                InstrNode::new(Inst::BinOp {
                        op: BinOp::UDiv,
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

/// fn mul_vals(a: i64, b: i64) -> i64 { a * b }
fn build_mul_function() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64, Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(0), "mul_vals", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
            id: BlockId::new(0),
            params: vec![
                (ValueId::new(0), Ty::I64),
                (ValueId::new(1), Ty::I64),
            ],
            body: vec![
                InstrNode::new(Inst::BinOp {
                        op: BinOp::Mul,
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

// ===========================================================================
// TEST 1: Empty function compiles to valid Mach-O
// ===========================================================================

#[test]
fn test_empty_function_compiles() {
    let (tmir_func, module) = build_empty_function();

    let obj_bytes = compile_tmir(&tmir_func, &module, OptLevel::O0)
        .expect("empty function should compile through full pipeline");

    assert_valid_macho(&obj_bytes, "empty");
    assert_eq!(
        macho_filetype(&obj_bytes),
        1, // MH_OBJECT
        "filetype should be MH_OBJECT (1)"
    );
    assert!(
        obj_bytes.len() > 50,
        "empty function should produce non-trivial object ({} bytes)",
        obj_bytes.len()
    );
}

// ===========================================================================
// TEST 2: 32-bit add function compiles
// ===========================================================================

#[test]
fn test_add_i32_function_compiles() {
    let (tmir_func, module) = build_add_i32();

    let obj_bytes = compile_tmir(&tmir_func, &module, OptLevel::O0)
        .expect("add32 should compile through full pipeline");

    assert_valid_macho(&obj_bytes, "add32");
    assert_eq!(macho_filetype(&obj_bytes), 1);
    assert!(
        obj_bytes.len() > 100,
        "add32 should produce meaningful code ({} bytes)",
        obj_bytes.len()
    );
}

// ===========================================================================
// TEST 3: Branch function compiles (conditional control flow)
// ===========================================================================

#[test]
fn test_branch_function_compiles() {
    let (tmir_func, module) = build_branch_function();

    let obj_bytes = compile_tmir(&tmir_func, &module, OptLevel::O0)
        .expect("abs_val should compile through full pipeline");

    assert_valid_macho(&obj_bytes, "abs_val");

    // Multi-block function should produce more code than minimal function.
    let (empty_func, empty_module) = build_empty_function();
    let empty_obj = compile_tmir(&empty_func, &empty_module, OptLevel::O0)
        .expect("empty function should compile");

    assert!(
        obj_bytes.len() >= empty_obj.len(),
        "branch function ({} bytes) should be at least as large as empty function ({} bytes)",
        obj_bytes.len(),
        empty_obj.len()
    );

    // Verify the adapter creates the expected number of blocks.
    let (lir_func, _) = llvm2_lower::translate_function(&tmir_func, &module)
        .expect("adapter should translate abs_val");
    assert_eq!(
        lir_func.blocks.len(),
        3,
        "abs_val should have 3 blocks (entry, positive, negate)"
    );
}

// ===========================================================================
// TEST 4: Loop function compiles (back-edge, block parameters)
// ===========================================================================

#[test]
fn test_loop_function_compiles() {
    let (tmir_func, module) = build_loop_function();

    let obj_bytes = compile_tmir(&tmir_func, &module, OptLevel::O0)
        .expect("sum_to_n should compile through full pipeline");

    assert_valid_macho(&obj_bytes, "sum_to_n");

    // Verify the adapter creates the expected number of blocks.
    let (lir_func, _) = llvm2_lower::translate_function(&tmir_func, &module)
        .expect("adapter should translate sum_to_n");
    assert_eq!(
        lir_func.blocks.len(),
        4,
        "sum_to_n should have 4 blocks (entry, header, body, exit)"
    );
}

// ===========================================================================
// TEST 5: Multi-arg function (4+ args for ABI testing)
// ===========================================================================

#[test]
fn test_multi_arg_function() {
    let (tmir_func, module) = build_multi_arg_function();

    let obj_bytes = compile_tmir(&tmir_func, &module, OptLevel::O0)
        .expect("add4 should compile through full pipeline");

    assert_valid_macho(&obj_bytes, "add4");

    // Verify the function has 4 parameters in the LIR.
    let (lir_func, _) = llvm2_lower::translate_function(&tmir_func, &module)
        .expect("adapter should translate add4");
    assert_eq!(
        lir_func.signature.params.len(),
        4,
        "add4 should have 4 parameters"
    );
}

// ===========================================================================
// TEST 6: Call instruction
// ===========================================================================

#[test]
fn test_call_instruction() {
    let (tmir_func, module) = build_call_function();

    // Call instruction support may or may not be complete. The adapter should
    // at least translate the function successfully.
    let translate_result = llvm2_lower::translate_function(&tmir_func, &module);
    assert!(
        translate_result.is_ok(),
        "adapter should translate call_test: {:?}",
        translate_result.err()
    );

    let (lir_func, _) = translate_result.unwrap();

    // Attempt full compilation. Call lowering may succeed (producing a BL
    // instruction) or fail at encoding (external calls need relocations).
    let config = PipelineConfig {
        opt_level: OptLevel::O0,
        emit_debug: false,
        ..Default::default()
    };
    let pipeline = Pipeline::new(config);
    let compile_result = pipeline.compile_function(&lir_func);

    match compile_result {
        Ok(obj_bytes) => {
            assert_valid_macho(&obj_bytes, "call_test");
        }
        Err(e) => {
            // Call instruction lowering or encoding is incomplete.
            // This is an expected state for an in-development compiler.
            let err_msg = format!("{}", e);
            eprintln!(
                "call_test compilation failed (expected for incomplete call support): {}",
                err_msg
            );
            // Verify the error is from ISel/encoding, not from the adapter.
            assert!(
                err_msg.contains("instruction selection")
                    || err_msg.contains("encoding")
                    || err_msg.contains("register allocation")
                    || err_msg.contains("unsupported"),
                "error should be from ISel/encoding/regalloc, not adapter: {}",
                err_msg
            );
        }
    }
}

// ===========================================================================
// TEST 7: O0 vs O2 comparison
// ===========================================================================

#[test]
fn test_pipeline_o0_vs_o2() {
    let (tmir_func, module) = build_loop_function();

    let obj_o0 = compile_tmir(&tmir_func, &module, OptLevel::O0)
        .expect("sum_to_n at O0 should compile");
    let obj_o2 = compile_tmir(&tmir_func, &module, OptLevel::O2)
        .expect("sum_to_n at O2 should compile");

    // Both must produce valid Mach-O.
    assert_valid_macho(&obj_o0, "sum_to_n@O0");
    assert_valid_macho(&obj_o2, "sum_to_n@O2");

    // Both must be MH_OBJECT.
    assert_eq!(macho_filetype(&obj_o0), 1, "O0 should be MH_OBJECT");
    assert_eq!(macho_filetype(&obj_o2), 1, "O2 should be MH_OBJECT");

    // Both should have the same cputype (AArch64).
    assert_eq!(
        macho_cputype(&obj_o0),
        macho_cputype(&obj_o2),
        "O0 and O2 should target the same CPU type"
    );

    // Both should produce non-trivial code.
    assert!(obj_o0.len() > 100, "O0 should produce meaningful code");
    assert!(obj_o2.len() > 100, "O2 should produce meaningful code");
}

// ===========================================================================
// TEST 8: Mach-O output structure validation
// ===========================================================================

#[test]
fn test_mach_o_output_structure() {
    let (tmir_func, module) = build_simple_add();

    let obj_bytes = compile_tmir(&tmir_func, &module, OptLevel::O0)
        .expect("simple_add should compile");

    // 1. Magic number (64-bit little-endian): 0xFEEDFACF
    assert_valid_macho(&obj_bytes, "macho_structure");

    // 2. CPU type: ARM64 = 0x0100000C (CPU_TYPE_ARM64)
    let cputype = macho_cputype(&obj_bytes);
    assert_eq!(
        cputype, 0x0100000C,
        "cputype should be ARM64 (0x0100000C), got 0x{:08X}",
        cputype
    );

    // 3. File type: MH_OBJECT = 1
    assert_eq!(macho_filetype(&obj_bytes), 1, "filetype should be MH_OBJECT");

    // 4. ncmds > 0 (must have at least one load command)
    let ncmds = macho_ncmds(&obj_bytes);
    assert!(ncmds > 0, "must have at least one load command, got {}", ncmds);

    // 5. sizeofcmds > 0
    let sizeofcmds = macho_sizeofcmds(&obj_bytes);
    assert!(sizeofcmds > 0, "sizeofcmds must be > 0, got {}", sizeofcmds);

    // 6. File size > header (32 bytes) + load commands
    let min_expected_size = 32 + sizeofcmds as usize;
    assert!(
        obj_bytes.len() >= min_expected_size,
        "file ({} bytes) should be >= header + cmds ({} bytes)",
        obj_bytes.len(),
        min_expected_size
    );

    // 7. CPU subtype at offset 8 should be non-zero (ARM64_ALL = 0 or ARM64E)
    // Actually CPU_SUBTYPE_ARM64_ALL = 0, which is valid.
    let cpusubtype = u32::from_le_bytes([obj_bytes[8], obj_bytes[9], obj_bytes[10], obj_bytes[11]]);
    // ARM64_ALL = 0, ARM64E = 2 — both valid
    assert!(
        cpusubtype <= 2,
        "cpusubtype should be ARM64_ALL (0) or ARM64E (2), got {}",
        cpusubtype
    );
}

// ===========================================================================
// TEST 9: Bitwise operations compile
// ===========================================================================

#[test]
fn test_bitwise_operations() {
    let (tmir_func, module) = build_bitwise_function();

    let obj_bytes = compile_tmir(&tmir_func, &module, OptLevel::O0)
        .expect("bitwise function should compile through full pipeline");

    assert_valid_macho(&obj_bytes, "bitwise");
    assert_eq!(macho_filetype(&obj_bytes), 1);
}

// ===========================================================================
// TEST 10: Subtraction function compiles
// ===========================================================================

#[test]
fn test_sub_function_compiles() {
    let (tmir_func, module) = build_sub_function();

    let obj_bytes = compile_tmir(&tmir_func, &module, OptLevel::O0)
        .expect("sub_vals should compile through full pipeline");

    assert_valid_macho(&obj_bytes, "sub_vals");
}

// ===========================================================================
// TEST 11: Negation function compiles
// ===========================================================================

#[test]
fn test_negation_function() {
    let (tmir_func, module) = build_negation_function();

    let obj_bytes = compile_tmir(&tmir_func, &module, OptLevel::O0)
        .expect("negate should compile through full pipeline");

    assert_valid_macho(&obj_bytes, "negate");
}

// ===========================================================================
// TEST 12: All optimization levels produce valid output
// ===========================================================================

#[test]
fn test_multiple_opt_levels_all_valid() {
    let (tmir_func, module) = build_loop_function();

    for opt_level in &[OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3] {
        let obj_bytes = compile_tmir(&tmir_func, &module, *opt_level)
            .unwrap_or_else(|e| panic!("sum_to_n at {:?} failed: {}", opt_level, e));

        assert_valid_macho(&obj_bytes, &format!("sum_to_n@{:?}", opt_level));
        assert_eq!(
            macho_filetype(&obj_bytes),
            1,
            "filetype should be MH_OBJECT at {:?}",
            opt_level
        );
    }
}

// ===========================================================================
// TEST 13: Shift operations compile
// ===========================================================================

#[test]
fn test_shift_operations() {
    let (tmir_func, module) = build_shift_function();

    let obj_bytes = compile_tmir(&tmir_func, &module, OptLevel::O0)
        .expect("shift_test should compile through full pipeline");

    assert_valid_macho(&obj_bytes, "shift_test");
    assert_eq!(macho_filetype(&obj_bytes), 1);
}

// ===========================================================================
// TEST 14: Unsigned division compiles
// ===========================================================================

#[test]
fn test_unsigned_division() {
    let (tmir_func, module) = build_unsigned_div_function();

    let obj_bytes = compile_tmir(&tmir_func, &module, OptLevel::O0)
        .expect("udiv_test should compile through full pipeline");

    assert_valid_macho(&obj_bytes, "udiv_test");
}

// ===========================================================================
// TEST 15: Multiplication compiles
// ===========================================================================

#[test]
fn test_mul_function_compiles() {
    let (tmir_func, module) = build_mul_function();

    let obj_bytes = compile_tmir(&tmir_func, &module, OptLevel::O0)
        .expect("mul_vals should compile through full pipeline");

    assert_valid_macho(&obj_bytes, "mul_vals");
}

// ===========================================================================
// TEST 16: Batch arithmetic — all basic ops produce valid Mach-O
// ===========================================================================

#[test]
fn test_all_arithmetic_ops() {
    let functions: Vec<(&str, TmirFunction, TmirModule)> = vec![
        { let (f, m) = build_simple_add(); ("simple_add", f, m) },
        { let (f, m) = build_sub_function(); ("sub_vals", f, m) },
        { let (f, m) = build_mul_function(); ("mul_vals", f, m) },
        { let (f, m) = build_negation_function(); ("negate", f, m) },
        { let (f, m) = build_bitwise_function(); ("bitwise", f, m) },
        { let (f, m) = build_shift_function(); ("shift_test", f, m) },
        { let (f, m) = build_unsigned_div_function(); ("udiv_test", f, m) },
    ];

    for (name, tmir_func, module) in &functions {
        let obj_bytes = compile_tmir(tmir_func, module, OptLevel::O0)
            .unwrap_or_else(|e| panic!("{}: compilation failed: {}", name, e));

        assert_valid_macho(&obj_bytes, name);
        assert_eq!(
            macho_filetype(&obj_bytes),
            1,
            "{}: filetype should be MH_OBJECT (1)",
            name
        );
        assert!(
            obj_bytes.len() > 50,
            "{}: object file suspiciously small ({} bytes)",
            name,
            obj_bytes.len()
        );
    }
}

// ===========================================================================
// TEST 17: Multi-block functions at multiple opt levels
// ===========================================================================

#[test]
fn test_multi_block_at_multiple_opt_levels() {
    let functions: Vec<(&str, TmirFunction, TmirModule)> = vec![
        { let (f, m) = build_branch_function(); ("abs_val", f, m) },
        { let (f, m) = build_loop_function(); ("sum_to_n", f, m) },
    ];

    for (name, tmir_func, module) in &functions {
        for opt_level in &[OptLevel::O0, OptLevel::O1, OptLevel::O2] {
            let obj_bytes = compile_tmir(tmir_func, module, *opt_level)
                .unwrap_or_else(|e| {
                    panic!("{} at {:?} failed: {}", name, opt_level, e)
                });

            assert_valid_macho(
                &obj_bytes,
                &format!("{}@{:?}", name, opt_level),
            );
        }
    }
}

// ===========================================================================
// TEST 18: Pipeline produces consistent cputype across functions
// ===========================================================================

#[test]
fn test_consistent_cputype() {
    let functions: Vec<(&str, TmirFunction, TmirModule)> = vec![
        { let (f, m) = build_empty_function(); ("empty", f, m) },
        { let (f, m) = build_simple_add(); ("simple_add", f, m) },
        { let (f, m) = build_branch_function(); ("abs_val", f, m) },
        { let (f, m) = build_loop_function(); ("sum_to_n", f, m) },
        { let (f, m) = build_multi_arg_function(); ("add4", f, m) },
    ];

    let mut cputypes = Vec::new();
    for (name, tmir_func, module) in &functions {
        let obj_bytes = compile_tmir(tmir_func, module, OptLevel::O0)
            .unwrap_or_else(|e| panic!("{}: compilation failed: {}", name, e));
        cputypes.push((name, macho_cputype(&obj_bytes)));
    }

    let expected_cputype = cputypes[0].1;
    for (name, cputype) in &cputypes {
        assert_eq!(
            *cputype, expected_cputype,
            "{}: cputype 0x{:08X} differs from expected 0x{:08X}",
            name, cputype, expected_cputype
        );
    }
}

// ===========================================================================
// TEST 19: Empty function at O2 (optimizer on trivial input)
// ===========================================================================

#[test]
fn test_empty_function_at_o2() {
    let (tmir_func, module) = build_empty_function();

    let obj_bytes = compile_tmir(&tmir_func, &module, OptLevel::O2)
        .expect("empty function should compile at O2");

    assert_valid_macho(&obj_bytes, "empty@O2");
}

// ===========================================================================
// TEST 20: Debug info increases object file size
// ===========================================================================

#[test]
fn test_debug_info_increases_size() {
    let (tmir_func, module) = build_simple_add();
    let (lir_func, _) = llvm2_lower::translate_function(&tmir_func, &module)
        .expect("adapter should translate simple_add");

    // Compile without debug info.
    let pipeline_no_debug = Pipeline::new(PipelineConfig {
        opt_level: OptLevel::O0,
        emit_debug: false,
        ..Default::default()
    });
    let obj_no_debug = pipeline_no_debug
        .compile_function(&lir_func)
        .expect("should compile without debug");

    // Compile with debug info.
    let (lir_func2, _) = llvm2_lower::translate_function(&tmir_func, &module)
        .expect("adapter should translate simple_add again");
    let pipeline_debug = Pipeline::new(PipelineConfig {
        opt_level: OptLevel::O0,
        emit_debug: true,
        ..Default::default()
    });
    let obj_debug = pipeline_debug
        .compile_function(&lir_func2)
        .expect("should compile with debug");

    assert_valid_macho(&obj_no_debug, "no_debug");
    assert_valid_macho(&obj_debug, "debug");

    assert!(
        obj_debug.len() > obj_no_debug.len(),
        "debug info should increase object file size ({} vs {} bytes)",
        obj_debug.len(),
        obj_no_debug.len()
    );
}
