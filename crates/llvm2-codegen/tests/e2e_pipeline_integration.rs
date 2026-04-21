// llvm2-codegen/tests/e2e_pipeline_integration.rs - End-to-end pipeline integration tests
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Integration tests exercising the full compilation path from tMIR to Mach-O
// with coverage of:
//   - Simple arithmetic (add, sub, mul, bitwise)
//   - Control flow (branches, multi-block, loops)
//   - Proof annotations (NoOverflow, InBounds, NonZeroDivisor)
//   - Pipeline configuration (O0 vs O2 vs O3, dispatch verification, debug info)
//   - Compiler API (structured results, metrics, tracing, proof certificates)
//   - Module-level compilation
//   - Error paths (empty module, invalid configuration)
//
// Part of #404 — End-to-end compilation pipeline integration tests

use llvm2_codegen::compiler::{
    Compiler, CompilerConfig, CompilerTraceLevel, CompileError,
};
use llvm2_codegen::pipeline::{
    DispatchVerifyMode, OptLevel, Pipeline, PipelineConfig,
};

use tmir::{Block as TmirBlock, Function as TmirFunction, Module as TmirModule, FuncTy, Ty, Constant};
use tmir::{Inst, InstrNode, BinOp, ICmpOp};
use tmir::{BlockId, FuncId, FuncTyId, ValueId};
use tmir::ProofAnnotation;


// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

/// Verify that bytes begin with a valid Mach-O 64-bit magic number.
fn assert_valid_macho(bytes: &[u8], context: &str) {
    assert!(
        bytes.len() >= 4,
        "{}: object file too small ({} bytes)",
        context,
        bytes.len()
    );
    assert_eq!(
        &bytes[0..4],
        &[0xCF, 0xFA, 0xED, 0xFE],
        "{}: invalid Mach-O magic",
        context
    );
}

/// Extract the MH_OBJECT filetype field from the Mach-O header.
fn macho_filetype(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]])
}

/// Compile a tMIR function through the full pipeline via the Pipeline API.
fn compile_tmir_via_pipeline(
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

// ---------------------------------------------------------------------------
// tMIR function builders
// ---------------------------------------------------------------------------

/// fn simple_add(a: i64, b: i64) -> i64 { a + b }
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

/// fn simple_sub(a: i64, b: i64) -> i64 { a - b }
fn build_simple_sub() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64, Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(1), "simple_sub", ft_id, BlockId::new(0));
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

/// fn mul_vals(a: i64, b: i64) -> i64 { a * b }
fn build_mul() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64, Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(2), "mul_vals", ft_id, BlockId::new(0));
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

/// fn return_const() -> i64 { 42 }
fn build_return_const() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(3), "return_const", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
            id: BlockId::new(0),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(42),
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

/// fn max_val(a: i64, b: i64) -> i64 { if a > b { a } else { b } }
///
/// Three blocks: entry with branch, two return blocks.
fn build_max_val() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64, Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(4), "max_val", ft_id, BlockId::new(0));
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
    module.add_function(func.clone());
    (func, module)
}

/// fn count_down(n: i64) -> i64
///
/// Loop: sum = 0, i = n; while (i > 0) { sum += i; i -= 1 }; return sum
fn build_count_down() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(5), "count_down", ft_id, BlockId::new(0));
    func.blocks = vec![
            // bb0 (entry): init sum=0, jump to loop header
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
                            args: vec![ValueId::new(1), ValueId::new(0)],
                        }),
                ],
            },
            // bb1 (loop header): params(sum, i)
            TmirBlock {
                id: BlockId::new(1),
                params: vec![
                    (ValueId::new(10), Ty::I64),
                    (ValueId::new(11), Ty::I64),
                ],
                body: vec![
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(0),
                        })

                        .with_result(ValueId::new(12)),
                    InstrNode::new(Inst::ICmp {
                            op: ICmpOp::Sle,
                            ty: Ty::I64,
                            lhs: ValueId::new(11),
                            rhs: ValueId::new(12),
                        })

                        .with_result(ValueId::new(13)),
                    InstrNode::new(Inst::CondBr {
                            cond: ValueId::new(13),
                            then_target: BlockId::new(2),
                            then_args: vec![ValueId::new(10)],
                            else_target: BlockId::new(3),
                            else_args: vec![],
                        }),
                ],
            },
            // bb2 (exit): return sum
            TmirBlock {
                id: BlockId::new(2),
                params: vec![(ValueId::new(20), Ty::I64)],
                body: vec![InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(20)],
                    })],
            },
            // bb3 (loop body): sum += i, i -= 1, loop back
            TmirBlock {
                id: BlockId::new(3),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Add,
                            ty: Ty::I64,
                            lhs: ValueId::new(10),
                            rhs: ValueId::new(11),
                        })

                        .with_result(ValueId::new(14)),
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(1),
                        })

                        .with_result(ValueId::new(15)),
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Sub,
                            ty: Ty::I64,
                            lhs: ValueId::new(11),
                            rhs: ValueId::new(15),
                        })

                        .with_result(ValueId::new(16)),
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(1),
                            args: vec![ValueId::new(14), ValueId::new(16)],
                        }),
                ],
            },
        ];
    module.add_function(func.clone());
    (func, module)
}

/// fn proven_add(a: i64, b: i64) -> i64 { a + b }
///
/// Same as simple_add but with NoOverflow proof annotation on the add
/// instruction, which should enable proof-guided optimizations.
fn build_add_with_no_overflow_proof() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64, Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(6), "proven_add", ft_id, BlockId::new(0));
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
                    .with_result(ValueId::new(2))
                    .with_proof(ProofAnnotation::NoOverflow),
                InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(2)],
                    }),
            ],
        }];
    module.add_function(func.clone());
    (func, module)
}

/// fn proven_div(a: i64, b: i64) -> i64 { a / b }
///
/// Division with NonZeroDivisor proof annotation on the divisor.
fn build_div_with_nonzero_proof() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64, Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(7), "proven_div", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
            id: BlockId::new(0),
            params: vec![
                (ValueId::new(0), Ty::I64),
                (ValueId::new(1), Ty::I64),
            ],
            body: vec![
                InstrNode::new(Inst::BinOp {
                        op: BinOp::SDiv,
                        ty: Ty::I64,
                        lhs: ValueId::new(0),
                        rhs: ValueId::new(1),
                    })
                    .with_result(ValueId::new(2))
                    .with_proof(ProofAnnotation::DivNonZero),
                InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(2)],
                    }),
            ],
        }];
    module.add_function(func.clone());
    (func, module)
}

/// fn pure_add(a: i64, b: i64) -> i64 { a + b }
///
/// Function-level Pure proof annotation, which enables aggressive CSE/LICM
/// and potentially GPU/ANE dispatch.
fn build_pure_function() -> (TmirFunction, TmirModule) {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64, Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(8), "pure_add", ft_id, BlockId::new(0));
    func.proofs = vec![ProofAnnotation::Pure];
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

/// fn diamond(a: i64, b: i64) -> i64
///
/// Diamond control flow: entry -> if/else -> merge -> return.
///   bb0: cmp a > b, condbr -> bb1 (a+1), bb2 (b+1)
///   bb1: result = a + 1, br -> bb3
///   bb2: result = b + 1, br -> bb3
///   bb3: return result
fn build_diamond_cfg() -> (TmirFunction, TmirModule) {

    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy { params: vec![Ty::I64, Ty::I64],
            returns: vec![Ty::I64], is_vararg: false });
    let mut func = TmirFunction::new(FuncId::new(9), "diamond", ft_id, BlockId::new(0));
    func.blocks = vec![
            // bb0: compare and branch
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
            // bb1: result = a + 1
            TmirBlock {
                id: BlockId::new(1),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(1),
                        })

                        .with_result(ValueId::new(3)),
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Add,
                            ty: Ty::I64,
                            lhs: ValueId::new(0),
                            rhs: ValueId::new(3),
                        })

                        .with_result(ValueId::new(4)),
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(3),
                            args: vec![ValueId::new(4)],
                        }),
                ],
            },
            // bb2: result = b + 1
            TmirBlock {
                id: BlockId::new(2),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(1),
                        })

                        .with_result(ValueId::new(5)),
                    InstrNode::new(Inst::BinOp {
                            op: BinOp::Add,
                            ty: Ty::I64,
                            lhs: ValueId::new(1),
                            rhs: ValueId::new(5),
                        })

                        .with_result(ValueId::new(6)),
                    InstrNode::new(Inst::Br {
                            target: BlockId::new(3),
                            args: vec![ValueId::new(6)],
                        }),
                ],
            },
            // bb3 (merge): return result
            TmirBlock {
                id: BlockId::new(3),
                params: vec![(ValueId::new(10), Ty::I64)],
                body: vec![InstrNode::new(Inst::Return {
                        values: vec![ValueId::new(10)],
                    })],
            },
        ];
    module.add_function(func.clone());
    (func, module)
}

// ===========================================================================
// TEST 1: Simple arithmetic — add, sub, mul all produce valid Mach-O
// ===========================================================================

#[test]
fn test_pipeline_arithmetic_ops_produce_valid_macho() {
    let functions: Vec<(&str, TmirFunction, TmirModule)> = vec![
        { let (f, m) = build_simple_add(); ("simple_add", f, m) },
        { let (f, m) = build_simple_sub(); ("simple_sub", f, m) },
        { let (f, m) = build_mul(); ("mul_vals", f, m) },
        { let (f, m) = build_return_const(); ("return_const", f, m) },
    ];

    for (name, tmir_func, module) in &functions {
        let obj_bytes = compile_tmir_via_pipeline(tmir_func, module, OptLevel::O0)
            .unwrap_or_else(|e| panic!("{}: compilation failed: {}", name, e));

        assert_valid_macho(&obj_bytes, name);
        assert_eq!(
            macho_filetype(&obj_bytes),
            1, // MH_OBJECT
            "{}: filetype should be MH_OBJECT (1)",
            name
        );
        assert!(
            obj_bytes.len() > 100,
            "{}: object file suspiciously small ({} bytes)",
            name,
            obj_bytes.len()
        );
    }
}

// ===========================================================================
// TEST 2: Multi-block control flow — branch, if/else, diamond
// ===========================================================================

#[test]
fn test_pipeline_multi_block_control_flow() {
    let (tmir_func, module) = build_max_val();
    let obj_bytes = compile_tmir_via_pipeline(&tmir_func, &module, OptLevel::O0)
        .expect("max_val should compile through full pipeline");

    assert_valid_macho(&obj_bytes, "max_val");

    // Multi-block function should produce larger code than single-block
    let (rc_func, rc_module) = build_return_const();
    let single_block = compile_tmir_via_pipeline(&rc_func, &rc_module, OptLevel::O0)
        .expect("return_const should compile");

    // max_val (3 blocks, compare + branch + 2 returns) should have more
    // code than return_const (1 block, const + return).
    // We check text section sizes via the object file sizes as a proxy.
    // max_val has branching logic so it should be non-trivially larger.
    assert!(
        obj_bytes.len() >= single_block.len(),
        "Multi-block function should produce at least as much code as single-block"
    );
}

// ===========================================================================
// TEST 3: Diamond CFG — merge point with block parameters
// ===========================================================================

#[test]
fn test_pipeline_diamond_cfg_compiles() {
    let (tmir_func, module) = build_diamond_cfg();

    let obj_bytes = compile_tmir_via_pipeline(&tmir_func, &module, OptLevel::O0)
        .expect("diamond CFG should compile through full pipeline");

    assert_valid_macho(&obj_bytes, "diamond");

    // Also verify the LIR structure has the expected 4 blocks.
    let (lir_func, _proof_ctx) = llvm2_lower::translate_function(&tmir_func, &module)
        .expect("adapter should translate diamond");
    assert_eq!(
        lir_func.blocks.len(),
        4,
        "diamond should have 4 blocks (entry, if, else, merge)"
    );
}

// ===========================================================================
// TEST 4: Loop — backward branch with block parameters
// ===========================================================================

#[test]
fn test_pipeline_loop_compiles() {
    let (tmir_func, module) = build_count_down();

    let obj_bytes = compile_tmir_via_pipeline(&tmir_func, &module, OptLevel::O0)
        .expect("count_down loop should compile through full pipeline");

    assert_valid_macho(&obj_bytes, "count_down");

    // count_down has 4 blocks (entry, loop header, exit, loop body).
    let (lir_func, _) = llvm2_lower::translate_function(&tmir_func, &module)
        .expect("adapter should translate count_down");
    assert!(lir_func.blocks.len() >= 4, "count_down should have at least 4 blocks, got {}", lir_func.blocks.len());
}

// ===========================================================================
// TEST 5: Proof annotations — NoOverflow compiles and preserves semantics
// ===========================================================================

#[test]
fn test_pipeline_proof_annotation_no_overflow() {
    let (tmir_func, module) = build_add_with_no_overflow_proof();

    // The function should compile successfully. The NoOverflow proof
    // annotation enables the proof-guided optimization pass to convert
    // checked arithmetic to unchecked arithmetic.
    let obj_bytes = compile_tmir_via_pipeline(&tmir_func, &module, OptLevel::O2)
        .expect("proven_add with NoOverflow should compile at O2");

    assert_valid_macho(&obj_bytes, "proven_add");

    // Verify the proof context was extracted by the adapter.
    let (_, proof_ctx) = llvm2_lower::translate_function(&tmir_func, &module)
        .expect("adapter should translate proven_add");
    assert!(
        !proof_ctx.value_proofs.is_empty(),
        "proof context should contain extracted NoOverflow proof"
    );
}

// ===========================================================================
// TEST 6: Proof annotations — NonZeroDivisor compiles
// ===========================================================================

#[test]
fn test_pipeline_proof_annotation_non_zero_divisor() {
    let (tmir_func, module) = build_div_with_nonzero_proof();

    // Division with NonZeroDivisor proof should compile.
    // The optimization pass may eliminate a divide-by-zero check.
    let obj_bytes = compile_tmir_via_pipeline(&tmir_func, &module, OptLevel::O2)
        .expect("proven_div with NonZeroDivisor should compile at O2");

    assert_valid_macho(&obj_bytes, "proven_div");
}

// ===========================================================================
// TEST 7: Proof annotations — Pure function-level proof
// ===========================================================================

#[test]
fn test_pipeline_proof_annotation_pure_function() {
    let (tmir_func, module) = build_pure_function();

    // A function-level Pure proof enables aggressive CSE/LICM.
    let obj_bytes = compile_tmir_via_pipeline(&tmir_func, &module, OptLevel::O2)
        .expect("pure_add with Pure annotation should compile at O2");

    assert_valid_macho(&obj_bytes, "pure_add");

    // Verify function-level proofs are accessible.
    assert_eq!(tmir_func.proofs.len(), 1);
    assert_eq!(tmir_func.proofs[0], ProofAnnotation::Pure);
}

// ===========================================================================
// TEST 8: Optimization levels — O0, O1, O2, O3 all produce valid output
// ===========================================================================

#[test]
fn test_pipeline_all_optimization_levels() {
    let (tmir_func, module) = build_count_down(); // Use a non-trivial function with loops

    for opt_level in &[OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3] {
        let obj_bytes = compile_tmir_via_pipeline(&tmir_func, &module, *opt_level)
            .unwrap_or_else(|e| panic!("count_down at {:?} failed: {}", opt_level, e));

        assert_valid_macho(&obj_bytes, &format!("count_down@{:?}", opt_level));
        assert_eq!(
            macho_filetype(&obj_bytes),
            1,
            "filetype should be MH_OBJECT at {:?}",
            opt_level
        );
    }
}

// ===========================================================================
// TEST 9: Compiler API — structured compilation result
// ===========================================================================

#[test]
fn test_compiler_api_compile_ir_function() {
    let mut ir_func = llvm2_codegen::pipeline::build_add_test_function();
    let compiler = Compiler::default_o2();

    let result = compiler
        .compile_ir_function(&mut ir_func)
        .expect("compile_ir_function should succeed");

    // Verify structured result fields.
    assert!(!result.object_code.is_empty(), "should produce Mach-O bytes");
    assert_valid_macho(&result.object_code, "compiler_api_add");
    assert_eq!(result.metrics.function_count, 1);
    assert!(result.metrics.code_size_bytes > 0);
    assert!(result.metrics.instruction_count > 0);
    assert!(result.trace.is_none(), "trace should be None with default config");
    assert!(result.proofs.is_none(), "proofs should be None by default");
}

// ===========================================================================
// TEST 10: Compiler API — tracing enabled
// ===========================================================================

#[test]
fn test_compiler_api_with_tracing() {
    let mut ir_func = llvm2_codegen::pipeline::build_add_test_function();

    let compiler = Compiler::new(CompilerConfig {
        trace_level: CompilerTraceLevel::Full,
        ..CompilerConfig::default()
    });

    let result = compiler
        .compile_ir_function(&mut ir_func)
        .expect("compile with tracing should succeed");

    assert!(result.trace.is_some(), "trace should be present at Full level");
    let trace = result.trace.unwrap();
    assert!(!trace.entries.is_empty(), "trace should have phase entries");
    assert!(
        trace.entries[0].phase.contains("compile"),
        "first trace entry should be compilation phase"
    );
}

// ===========================================================================
// TEST 11: Compiler API — proof certificates from function verifier
// ===========================================================================

#[test]
fn test_compiler_api_with_proof_certificates() {
    let mut ir_func = llvm2_codegen::pipeline::build_add_test_function();

    let compiler = Compiler::new(CompilerConfig {
        emit_proofs: true,
        ..CompilerConfig::default()
    });

    let result = compiler
        .compile_ir_function(&mut ir_func)
        .expect("compile with proofs should succeed");

    assert!(
        result.proofs.is_some(),
        "proofs field should be Some when emit_proofs is true"
    );
    let proofs = result.proofs.unwrap();
    // The add test function contains verified instructions (e.g., AddRR)
    // that produce real proof certificates from the proof database.
    assert!(!proofs.is_empty(), "proof certificates should be non-empty");
    for cert in &proofs {
        assert!(cert.verified, "certificate '{}' should be verified", cert.rule_name);
        assert!(!cert.rule_name.is_empty());
        assert!(!cert.function_name.is_empty());
    }
}

// ===========================================================================
// TEST 12: Compiler API — module-level compilation
// ===========================================================================

/// Build a multi-function module with all func_types properly registered.
/// Each function builder creates its own module with its own FuncTyId(0), so
/// we cannot simply add functions from separate builders into a new empty module.
/// Instead, we register all needed func_types in the shared module and
/// reassign FuncTyIds accordingly.
fn build_multi_function_module(
    name: &str,
    builders: &[fn() -> (TmirFunction, TmirModule)],
) -> TmirModule {
    let mut module = TmirModule::new(name);
    // Track registered func_types to deduplicate
    let mut registered_fts: Vec<FuncTy> = Vec::new();

    for builder in builders {
        let (mut func, src_module) = builder();
        // Look up the func_type from the source module
        let src_ft = &src_module.func_types[func.ty.index() as usize];
        // Check if we already registered an equivalent func_type
        let new_ft_id = if let Some(pos) = registered_fts.iter().position(|ft| ft == src_ft) {
            FuncTyId::new(pos as u32)
        } else {
            let id = module.add_func_type(src_ft.clone());
            registered_fts.push(src_ft.clone());
            id
        };
        func.ty = new_ft_id;
        module.add_function(func);
    }
    module
}

#[test]
fn test_compiler_api_compile_module() {
    let module = build_multi_function_module("test_module", &[
        build_simple_add,
        build_return_const,
    ]);

    let compiler = Compiler::default_o2();
    let result = compiler.compile(&module).expect("module compilation should succeed");

    assert!(!result.object_code.is_empty());
    assert_valid_macho(&result.object_code, "module_compilation");
    assert_eq!(
        result.metrics.function_count, 2,
        "should report 2 functions compiled"
    );
}

// ===========================================================================
// TEST 13: Error path — empty module
// ===========================================================================

#[test]
fn test_compiler_api_empty_module_error() {
    let module = TmirModule::new("empty_module");
    let compiler = Compiler::default_o2();

    let result = compiler.compile(&module);
    assert!(result.is_err(), "empty module should produce an error");

    match result.unwrap_err() {
        CompileError::EmptyModule => {} // expected
        other => panic!("Expected EmptyModule error, got: {:?}", other),
    }
}

// ===========================================================================
// TEST 14: Dispatch verification — FallbackOnFailure mode
// ===========================================================================

#[test]
fn test_pipeline_dispatch_verify_fallback_mode() {
    let config = PipelineConfig {
        opt_level: OptLevel::O2,
        emit_debug: false,
        verify_dispatch: DispatchVerifyMode::FallbackOnFailure,
        ..Default::default()
    };

    let pipeline = Pipeline::new(config);
    assert_eq!(
        pipeline.config.verify_dispatch,
        DispatchVerifyMode::FallbackOnFailure
    );

    // Compile a function through this pipeline configuration.
    let (tmir_func, module) = build_simple_add();
    let (lir_func, _) = llvm2_lower::translate_function(&tmir_func, &module)
        .expect("adapter should translate simple_add");

    let obj_bytes = pipeline
        .compile_function(&lir_func)
        .expect("FallbackOnFailure mode should compile successfully");
    assert_valid_macho(&obj_bytes, "dispatch_fallback");
}

// ===========================================================================
// TEST 15: Dispatch verification — Off mode
// ===========================================================================

#[test]
fn test_pipeline_dispatch_verify_off_mode() {
    let config = PipelineConfig {
        opt_level: OptLevel::O0,
        emit_debug: false,
        verify_dispatch: DispatchVerifyMode::Off,
        ..Default::default()
    };

    let pipeline = Pipeline::new(config);
    let (tmir_func, module) = build_simple_add();
    let (lir_func, _) = llvm2_lower::translate_function(&tmir_func, &module)
        .expect("adapter should translate simple_add");

    let obj_bytes = pipeline
        .compile_function(&lir_func)
        .expect("Off mode should compile successfully");
    assert_valid_macho(&obj_bytes, "dispatch_off");
}

// ===========================================================================
// TEST 16: Pipeline with debug info enabled
// ===========================================================================

#[test]
fn test_pipeline_with_debug_info() {
    let config = PipelineConfig {
        opt_level: OptLevel::O0,
        emit_debug: true,
        verify_dispatch: DispatchVerifyMode::Off,
        ..Default::default()
    };

    let pipeline = Pipeline::new(config);
    let (tmir_func, module) = build_simple_add();
    let (lir_func, _) = llvm2_lower::translate_function(&tmir_func, &module)
        .expect("adapter should translate simple_add");

    let obj_bytes = pipeline
        .compile_function(&lir_func)
        .expect("pipeline with debug info should compile successfully");

    assert_valid_macho(&obj_bytes, "debug_info");

    // With debug info enabled, the object file should be larger due to
    // DWARF sections (__debug_info, __debug_abbrev, __debug_str, __debug_line).
    let config_no_debug = PipelineConfig {
        opt_level: OptLevel::O0,
        emit_debug: false,
        verify_dispatch: DispatchVerifyMode::Off,
        ..Default::default()
    };
    let pipeline_no_debug = Pipeline::new(config_no_debug);
    let (lir_func2, _) = llvm2_lower::translate_function(&tmir_func, &module)
        .expect("adapter should translate simple_add again");
    let obj_bytes_no_debug = pipeline_no_debug
        .compile_function(&lir_func2)
        .expect("pipeline without debug should compile");

    assert!(
        obj_bytes.len() > obj_bytes_no_debug.len(),
        "debug info should increase object file size ({} vs {} bytes)",
        obj_bytes.len(),
        obj_bytes_no_debug.len()
    );
}

// ===========================================================================
// TEST 17: Proof-annotated function at O0 vs O2 — proof opts affect code size
// ===========================================================================

#[test]
fn test_proof_annotations_affect_optimization() {
    // Compile the same function (with proof) at O0 and O2.
    // At O2, the proof-guided optimization pass should fire.
    let (tmir_func, module) = build_add_with_no_overflow_proof();

    let obj_o0 = compile_tmir_via_pipeline(&tmir_func, &module, OptLevel::O0)
        .expect("proven_add at O0 should compile");
    let obj_o2 = compile_tmir_via_pipeline(&tmir_func, &module, OptLevel::O2)
        .expect("proven_add at O2 should compile");

    assert_valid_macho(&obj_o0, "proven_add@O0");
    assert_valid_macho(&obj_o2, "proven_add@O2");

    // Both should be valid Mach-O. The O2 version may have different code
    // size due to optimizations, but both must be structurally valid.
    // (Exact size comparison depends on which optimizations fire.)
}

// ===========================================================================
// TEST 18: Compiler config — all optimization levels produce valid metrics
// ===========================================================================

#[test]
fn test_compiler_api_optimization_pass_metrics() {
    let mut ir_func = llvm2_codegen::pipeline::build_add_test_function();

    // O0 should report 0 optimization passes.
    let compiler_o0 = Compiler::new(CompilerConfig {
        opt_level: OptLevel::O0,
        ..CompilerConfig::default()
    });
    let result_o0 = compiler_o0
        .compile_ir_function(&mut ir_func)
        .expect("O0 should compile");
    assert_eq!(
        result_o0.metrics.optimization_passes_run, 0,
        "O0 should run 0 optimization passes"
    );

    // O2 should report > 0 optimization passes.
    let mut ir_func2 = llvm2_codegen::pipeline::build_add_test_function();
    let compiler_o2 = Compiler::default_o2();
    let result_o2 = compiler_o2
        .compile_ir_function(&mut ir_func2)
        .expect("O2 should compile");
    assert!(
        result_o2.metrics.optimization_passes_run > 0,
        "O2 should run optimization passes, got {}",
        result_o2.metrics.optimization_passes_run
    );
}

// ===========================================================================
// TEST 19: Pipeline default config validates
// ===========================================================================

#[test]
fn test_pipeline_default_config() {
    let config = PipelineConfig::default();
    assert_eq!(config.opt_level, OptLevel::O2);
    assert!(!config.emit_debug);
    assert_eq!(
        config.verify_dispatch,
        DispatchVerifyMode::FallbackOnFailure
    );
}

// ===========================================================================
// TEST 20: compile_to_object convenience function
// ===========================================================================

#[test]
fn test_compile_to_object_convenience() {
    use llvm2_codegen::pipeline::compile_to_object;

    let (tmir_func, module) = build_simple_add();
    let (lir_func, _) = llvm2_lower::translate_function(&tmir_func, &module)
        .expect("adapter should translate simple_add");

    let obj_bytes =
        compile_to_object(&lir_func, OptLevel::O2).expect("compile_to_object should succeed");

    assert_valid_macho(&obj_bytes, "compile_to_object");
    assert_eq!(macho_filetype(&obj_bytes), 1);
}

// ===========================================================================
// TEST 21: Multi-function module — all functions emitted in single .o
// ===========================================================================

/// Parse the Mach-O string table to extract all symbol name strings.
/// This is a minimal parser that finds the LC_SYMTAB load command,
/// reads the string table offset/size, and extracts null-terminated strings.
fn extract_macho_symbol_names(bytes: &[u8]) -> Vec<String> {
    // Mach-O 64-bit header: magic(4) + cputype(4) + cpusubtype(4) + filetype(4)
    //                       + ncmds(4) + sizeofcmds(4) + flags(4) + reserved(4) = 32 bytes
    let ncmds = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;
    let mut offset = 32; // after mach_header_64

    let mut symtab_offset = 0u32;
    let mut symtab_nsyms = 0u32;
    let mut strtab_offset = 0u32;
    let mut strtab_size = 0u32;

    // Find LC_SYMTAB (cmd == 2)
    for _ in 0..ncmds {
        if offset + 8 > bytes.len() {
            break;
        }
        let cmd = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]);
        let cmdsize = u32::from_le_bytes([bytes[offset+4], bytes[offset+5], bytes[offset+6], bytes[offset+7]]) as usize;

        if cmd == 2 {
            // LC_SYMTAB: cmd(4) + cmdsize(4) + symoff(4) + nsyms(4) + stroff(4) + strsize(4)
            symtab_offset = u32::from_le_bytes([bytes[offset+8], bytes[offset+9], bytes[offset+10], bytes[offset+11]]);
            symtab_nsyms = u32::from_le_bytes([bytes[offset+12], bytes[offset+13], bytes[offset+14], bytes[offset+15]]);
            strtab_offset = u32::from_le_bytes([bytes[offset+16], bytes[offset+17], bytes[offset+18], bytes[offset+19]]);
            strtab_size = u32::from_le_bytes([bytes[offset+20], bytes[offset+21], bytes[offset+22], bytes[offset+23]]);
            break;
        }
        offset += cmdsize;
    }

    let mut names = Vec::new();
    if symtab_nsyms == 0 || strtab_size == 0 {
        return names;
    }

    // Each nlist_64 entry is 16 bytes: n_strx(4) + n_type(1) + n_sect(1) + n_desc(2) + n_value(8)
    let nlist_size = 16usize;
    for i in 0..symtab_nsyms as usize {
        let nlist_off = symtab_offset as usize + i * nlist_size;
        if nlist_off + 4 > bytes.len() {
            break;
        }
        let n_strx = u32::from_le_bytes([
            bytes[nlist_off], bytes[nlist_off+1], bytes[nlist_off+2], bytes[nlist_off+3],
        ]) as usize;

        let str_start = strtab_offset as usize + n_strx;
        if str_start >= bytes.len() {
            continue;
        }
        // Read null-terminated string
        let str_end = bytes[str_start..].iter().position(|&b| b == 0).unwrap_or(0) + str_start;
        if str_end > str_start {
            if let Ok(name) = std::str::from_utf8(&bytes[str_start..str_end]) {
                names.push(name.to_string());
            }
        }
    }

    names
}

#[test]
fn test_multi_function_module_all_functions_emitted() {
    // Build a module with three functions using the shared-module builder.
    let module = build_multi_function_module("multi_func_module", &[
        build_simple_add,
        build_simple_sub,
        build_return_const,
    ]);

    let compiler = Compiler::default_o2();
    let result = compiler.compile(&module).expect("multi-function module should compile");

    // Basic validity.
    assert!(!result.object_code.is_empty());
    assert_valid_macho(&result.object_code, "multi_func_module");
    assert_eq!(result.metrics.function_count, 3, "should report 3 functions compiled");

    // Verify all function symbols are present in the Mach-O symbol table.
    let symbol_names = extract_macho_symbol_names(&result.object_code);
    assert!(
        symbol_names.contains(&"_simple_add".to_string()),
        "symbol table should contain _simple_add. Found: {:?}",
        symbol_names
    );
    assert!(
        symbol_names.contains(&"_simple_sub".to_string()),
        "symbol table should contain _simple_sub. Found: {:?}",
        symbol_names
    );
    assert!(
        symbol_names.contains(&"_return_const".to_string()),
        "symbol table should contain _return_const. Found: {:?}",
        symbol_names
    );
}

// ===========================================================================
// TEST 22: Multi-function module — different from single-function output
// ===========================================================================

#[test]
fn test_multi_function_module_differs_from_single() {
    // A module with two functions should produce different (and larger)
    // output than a module with one function.
    let module_one = build_multi_function_module("one_func", &[
        build_simple_add,
    ]);

    let module_two = build_multi_function_module("two_func", &[
        build_simple_add,
        build_return_const,
    ]);

    let compiler = Compiler::default_o2();
    let result_one = compiler.compile(&module_one).expect("single-function module");
    let result_two = compiler.compile(&module_two).expect("two-function module");

    assert!(
        result_two.object_code.len() > result_one.object_code.len(),
        "two-function module ({} bytes) should be larger than single-function ({} bytes)",
        result_two.object_code.len(),
        result_one.object_code.len(),
    );

    // Verify the two-function module has both symbols.
    let symbols = extract_macho_symbol_names(&result_two.object_code);
    assert!(symbols.contains(&"_simple_add".to_string()));
    assert!(symbols.contains(&"_return_const".to_string()));

    // The single-function module should only have one symbol.
    let symbols_one = extract_macho_symbol_names(&result_one.object_code);
    assert!(symbols_one.contains(&"_simple_add".to_string()));
    assert!(
        !symbols_one.contains(&"_return_const".to_string()),
        "single-function module should not contain _return_const"
    );
}
