// tmir_integration.rs - End-to-end tMIR -> adapter -> ISel integration tests
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

use llvm2_lower::adapter::{translate_function, translate_module, AdapterError, ProofContext};
use llvm2_lower::function::Function;
use llvm2_lower::instructions::{Block, Opcode};
use llvm2_lower::isel::{AArch64Opcode, ISelFunction, InstructionSelector};
use llvm2_lower::types::Type;

use tmir::{
    BinOp, Block as TmirBlock, BlockId, CastOp, Constant, FuncId, FuncTy, FuncTyId,
    Function as TmirFunction, ICmpOp, Inst, InstrNode, Module as TmirModule,
    ProofAnnotation, SwitchCase, Ty, UnOp, ValueId,
};

fn v(n: u32) -> ValueId {
    ValueId::new(n)
}

fn b(n: u32) -> BlockId {
    BlockId::new(n)
}

fn f(n: u32) -> FuncId {
    FuncId::new(n)
}

fn func_ty(params: Vec<Ty>, returns: Vec<Ty>) -> FuncTy {
    FuncTy {
        params,
        returns,
        is_vararg: false,
    }
}

fn single_function_module(
    func_id: u32,
    func_name: &str,
    ty: FuncTy,
    blocks: Vec<TmirBlock>,
    proofs: Vec<ProofAnnotation>,
) -> TmirModule {
    let entry = blocks.first().expect("module must have a block").id;
    let mut module = TmirModule::new(func_name);
    let func_ty_id: FuncTyId = module.add_func_type(ty);

    let mut func = TmirFunction::new(f(func_id), func_name, func_ty_id, entry);
    func.blocks = blocks;
    func.proofs = proofs;

    module.add_function(func);
    module
}

fn single_function(module: &TmirModule) -> &TmirFunction {
    module
        .functions
        .first()
        .expect("expected a single-function module")
}

// ---------------------------------------------------------------------------
// Helper: run a tMIR function through adapter + ISel
// ---------------------------------------------------------------------------

fn compile_tmir_function(module: &TmirModule) -> ISelFunction {
    let func = single_function(module);

    let (lir_func, _proof_ctx) =
        translate_function(func, module).expect("adapter translation failed");

    let mut isel = InstructionSelector::new(lir_func.name.clone(), lir_func.signature.clone());

    isel.lower_formal_arguments(&lir_func.signature, lir_func.entry_block)
        .unwrap();

    let mut block_order: Vec<Block> = lir_func.blocks.keys().copied().collect();
    block_order.sort_by_key(|b| if *b == lir_func.entry_block { 0 } else { b.0 + 1 });

    for block_id in &block_order {
        let bb = &lir_func.blocks[block_id];
        isel.select_block(*block_id, &bb.instructions).unwrap();
    }

    isel.finalize()
}

fn translate_only(module: &TmirModule) -> Result<(Function, ProofContext), AdapterError> {
    translate_function(single_function(module), module)
}

fn count_opcode(mfunc: &ISelFunction, opcode: AArch64Opcode) -> usize {
    mfunc
        .blocks
        .values()
        .flat_map(|b| &b.insts)
        .filter(|inst| inst.opcode == opcode)
        .count()
}

fn has_opcode(mfunc: &ISelFunction, opcode: AArch64Opcode) -> bool {
    count_opcode(mfunc, opcode) > 0
}

fn total_insts(mfunc: &ISelFunction) -> usize {
    mfunc.blocks.values().map(|b| b.insts.len()).sum()
}

// ===========================================================================
// Test 1: identity(x: i32) -> i32 { x }
// ===========================================================================

fn build_identity() -> TmirModule {
    single_function_module(
        0,
        "identity",
        func_ty(vec![Ty::I32], vec![Ty::I32]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I32)],
            body: vec![InstrNode::new(Inst::Return { values: vec![v(0)] })],
        }],
        vec![],
    )
}

#[test]
fn test_identity_adapter() {
    let module = build_identity();
    let (lir_func, proof_ctx) = translate_only(&module).unwrap();

    assert_eq!(lir_func.name, "identity");
    assert_eq!(lir_func.signature.params, vec![Type::I32]);
    assert_eq!(lir_func.signature.returns, vec![Type::I32]);
    assert_eq!(lir_func.blocks.len(), 1);

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.params.len(), 1);
    assert_eq!(entry.instructions.len(), 1);

    assert!(proof_ctx.value_proofs.is_empty());
}

#[test]
fn test_identity_isel() {
    let module = build_identity();
    let mfunc = compile_tmir_function(&module);

    assert_eq!(mfunc.name, "identity");
    assert!(!mfunc.blocks.is_empty());
    assert!(has_opcode(&mfunc, AArch64Opcode::Copy));
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test 2: add(a: i32, b: i32) -> i32 { a + b }
// ===========================================================================

fn build_add() -> TmirModule {
    single_function_module(
        1,
        "add",
        func_ty(vec![Ty::I32, Ty::I32], vec![Ty::I32]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I32), (v(1), Ty::I32)],
            body: vec![
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I32,
                    lhs: v(0),
                    rhs: v(1),
                })
                .with_result(v(2)),
                InstrNode::new(Inst::Return { values: vec![v(2)] }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_add_adapter() {
    let module = build_add();
    let (lir_func, _) = translate_only(&module).unwrap();

    assert_eq!(lir_func.name, "add");
    assert_eq!(lir_func.signature.params, vec![Type::I32, Type::I32]);

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.params.len(), 2);
    assert_eq!(entry.instructions.len(), 2);
}

#[test]
fn test_add_isel() {
    let module = build_add();
    let mfunc = compile_tmir_function(&module);

    assert_eq!(mfunc.name, "add");
    assert!(
        has_opcode(&mfunc, AArch64Opcode::AddRR),
        "Expected ADDWrr for i32 addition"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test 3: negate(x: i32) -> i32 { -x }
// ===========================================================================

fn build_negate() -> TmirModule {
    single_function_module(
        2,
        "negate",
        func_ty(vec![Ty::I32], vec![Ty::I32]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I32)],
            body: vec![
                InstrNode::new(Inst::UnOp {
                    op: UnOp::Neg,
                    ty: Ty::I32,
                    operand: v(0),
                })
                .with_result(v(1)),
                InstrNode::new(Inst::Return { values: vec![v(1)] }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_negate_adapter() {
    let module = build_negate();
    let (lir_func, _) = translate_only(&module).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 2);
}

#[test]
fn test_negate_isel() {
    let module = build_negate();
    let mfunc = compile_tmir_function(&module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::Neg),
        "Expected NEGWr for negation (-x)"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test 4: max(a: i32, b: i32) -> i32 { if a > b { a } else { b } }
// ===========================================================================

fn build_max() -> TmirModule {
    single_function_module(
        3,
        "max",
        func_ty(vec![Ty::I32, Ty::I32], vec![Ty::I32]),
        vec![
            TmirBlock {
                id: b(0),
                params: vec![(v(0), Ty::I32), (v(1), Ty::I32)],
                body: vec![
                    InstrNode::new(Inst::ICmp {
                        op: ICmpOp::Sgt,
                        ty: Ty::I32,
                        lhs: v(0),
                        rhs: v(1),
                    })
                    .with_result(v(2)),
                    InstrNode::new(Inst::CondBr {
                        cond: v(2),
                        then_target: b(1),
                        then_args: vec![],
                        else_target: b(2),
                        else_args: vec![],
                    }),
                ],
            },
            TmirBlock {
                id: b(1),
                params: vec![],
                body: vec![InstrNode::new(Inst::Return { values: vec![v(0)] })],
            },
            TmirBlock {
                id: b(2),
                params: vec![],
                body: vec![InstrNode::new(Inst::Return { values: vec![v(1)] })],
            },
        ],
        vec![],
    )
}

#[test]
fn test_max_adapter() {
    let module = build_max();
    let (lir_func, _) = translate_only(&module).unwrap();

    assert_eq!(lir_func.blocks.len(), 3);

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 2);
}

#[test]
fn test_max_isel() {
    let module = build_max();
    let mfunc = compile_tmir_function(&module);

    assert_eq!(mfunc.blocks.len(), 3);
    assert!(
        has_opcode(&mfunc, AArch64Opcode::CmpRR),
        "Expected CMPWrr for signed comparison"
    );
    assert!(
        has_opcode(&mfunc, AArch64Opcode::BCond),
        "Expected Bcc for conditional branch"
    );
    assert!(
        count_opcode(&mfunc, AArch64Opcode::Ret) >= 2,
        "Expected at least 2 RET instructions (then + else)"
    );
}

// ===========================================================================
// Test 5: sum(n: i32) -> i32
// ===========================================================================

fn build_sum() -> TmirModule {
    single_function_module(
        4,
        "sum",
        func_ty(vec![Ty::I32], vec![Ty::I32]),
        vec![
            TmirBlock {
                id: b(0),
                params: vec![(v(0), Ty::I32)],
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I32,
                        value: Constant::Int(0),
                    })
                    .with_result(v(1)),
                    InstrNode::new(Inst::Br {
                        target: b(1),
                        args: vec![v(0), v(1)],
                    }),
                ],
            },
            TmirBlock {
                id: b(1),
                params: vec![(v(2), Ty::I32), (v(3), Ty::I32)],
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I32,
                        value: Constant::Int(0),
                    })
                    .with_result(v(4)),
                    InstrNode::new(Inst::ICmp {
                        op: ICmpOp::Sgt,
                        ty: Ty::I32,
                        lhs: v(2),
                        rhs: v(4),
                    })
                    .with_result(v(5)),
                    InstrNode::new(Inst::CondBr {
                        cond: v(5),
                        then_target: b(2),
                        then_args: vec![],
                        else_target: b(3),
                        else_args: vec![],
                    }),
                ],
            },
            TmirBlock {
                id: b(2),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I32,
                        lhs: v(3),
                        rhs: v(2),
                    })
                    .with_result(v(6)),
                    InstrNode::new(Inst::Const {
                        ty: Ty::I32,
                        value: Constant::Int(1),
                    })
                    .with_result(v(7)),
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Sub,
                        ty: Ty::I32,
                        lhs: v(2),
                        rhs: v(7),
                    })
                    .with_result(v(8)),
                    InstrNode::new(Inst::Br {
                        target: b(1),
                        args: vec![v(8), v(6)],
                    }),
                ],
            },
            TmirBlock {
                id: b(3),
                params: vec![],
                body: vec![InstrNode::new(Inst::Return { values: vec![v(3)] })],
            },
        ],
        vec![],
    )
}

#[test]
fn test_sum_adapter() {
    let module = build_sum();
    let (lir_func, _) = translate_only(&module).unwrap();

    assert_eq!(lir_func.name, "sum");
    assert_eq!(lir_func.blocks.len(), 4);

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert!(
        entry.instructions.len() >= 2,
        "Entry should have at least Iconst + Jump, got {}",
        entry.instructions.len()
    );
}

#[test]
fn test_sum_isel_body_instructions() {
    use llvm2_lower::function::Signature;
    use llvm2_lower::instructions::{Instruction, Value};

    let sig = Signature {
        params: vec![Type::I32, Type::I32],
        returns: vec![Type::I32],
    };
    let mut isel = InstructionSelector::new("sum_body".to_string(), sig.clone());
    let entry = Block(0);
    isel.lower_formal_arguments(&sig, entry).unwrap();

    isel.select_block(
        entry,
        &[
            Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I32,
                    imm: 1,
                },
                args: vec![],
                results: vec![Value(3)],
            },
            Instruction {
                opcode: Opcode::Isub,
                args: vec![Value(1), Value(3)],
                results: vec![Value(4)],
            },
            Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I32,
                    imm: 0,
                },
                args: vec![],
                results: vec![Value(5)],
            },
            Instruction {
                opcode: Opcode::Icmp {
                    cond: llvm2_lower::instructions::IntCC::SignedGreaterThan,
                },
                args: vec![Value(4), Value(5)],
                results: vec![Value(6)],
            },
            Instruction {
                opcode: Opcode::Return,
                args: vec![Value(2)],
                results: vec![],
            },
        ],
    )
    .unwrap();

    let mfunc = isel.finalize();
    let mblock = &mfunc.blocks[&entry];

    assert!(has_opcode(&mfunc, AArch64Opcode::AddRR));
    assert!(has_opcode(&mfunc, AArch64Opcode::SubRR));
    assert!(has_opcode(&mfunc, AArch64Opcode::CmpRR));
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
    assert!(mblock.insts.len() >= 6, "Expected at least 6 instructions");
}

// ===========================================================================
// Test 6: load_store(p: *mut i32, v: i32) { *p = v; }
// ===========================================================================

fn build_load_store() -> TmirModule {
    single_function_module(
        5,
        "load_store",
        func_ty(vec![Ty::Ptr, Ty::I32], vec![]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::Ptr), (v(1), Ty::I32)],
            body: vec![
                InstrNode::new(Inst::Store {
                    ty: Ty::I32,
                    ptr: v(0),
                    value: v(1),
                }),
                InstrNode::new(Inst::Return { values: vec![] }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_load_store_adapter() {
    let module = build_load_store();
    let (lir_func, _) = translate_only(&module).unwrap();

    assert_eq!(lir_func.name, "load_store");
    assert_eq!(lir_func.signature.params, vec![Type::I64, Type::I32]);
    assert_eq!(lir_func.signature.returns, vec![]);

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 2);
}

#[test]
fn test_load_store_isel() {
    let module = build_load_store();
    let mfunc = compile_tmir_function(&module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::StrRI),
        "Expected STRWui for 32-bit store"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Extended test: load then store
// ===========================================================================

fn build_load_then_store() -> TmirModule {
    single_function_module(
        6,
        "load_then_store",
        func_ty(vec![Ty::Ptr], vec![]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::Ptr)],
            body: vec![
                InstrNode::new(Inst::Load {
                    ty: Ty::I32,
                    ptr: v(0),
                })
                .with_result(v(1)),
                InstrNode::new(Inst::Store {
                    ty: Ty::I32,
                    ptr: v(0),
                    value: v(1),
                }),
                InstrNode::new(Inst::Return { values: vec![] }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_load_then_store_isel() {
    let module = build_load_then_store();
    let mfunc = compile_tmir_function(&module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::LdrRI),
        "Expected LDRWui for 32-bit load"
    );
    assert!(
        has_opcode(&mfunc, AArch64Opcode::StrRI),
        "Expected STRWui for 32-bit store"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// End-to-end pipeline test
// ===========================================================================

#[test]
fn test_all_single_block_programs_compile_without_panic() {
    let programs: Vec<(&str, TmirModule)> = vec![
        ("identity", build_identity()),
        ("add", build_add()),
        ("negate", build_negate()),
        ("max", build_max()),
        ("load_store", build_load_store()),
    ];

    for (name, module) in &programs {
        let mfunc = compile_tmir_function(module);
        assert!(
            !mfunc.blocks.is_empty(),
            "{}: produced empty ISelFunction",
            name
        );
        assert!(
            total_insts(&mfunc) > 0,
            "{}: produced no machine instructions",
            name
        );
        assert!(
            has_opcode(&mfunc, AArch64Opcode::Ret),
            "{}: missing RET instruction",
            name
        );
        assert!(
            has_opcode(&mfunc, AArch64Opcode::Copy),
            "{}: missing COPY for formal arguments",
            name
        );
    }
}

#[test]
fn test_all_programs_adapter_succeeds() {
    let programs: Vec<(&str, TmirModule)> = vec![
        ("identity", build_identity()),
        ("add", build_add()),
        ("negate", build_negate()),
        ("max", build_max()),
        ("sum", build_sum()),
        ("load_store", build_load_store()),
    ];

    for (name, module) in &programs {
        let result = translate_only(module);
        assert!(
            result.is_ok(),
            "{}: adapter translation failed: {:?}",
            name,
            result.err()
        );
        let (lir_func, _) = result.unwrap();
        assert_eq!(lir_func.name, *name);
        assert!(!lir_func.blocks.is_empty());
    }
}

// ===========================================================================
// Adapter-level regression: verify type translation preserves semantics
// ===========================================================================

#[test]
fn test_adapter_type_translation_in_context() {
    let module = build_load_store();
    let (lir_func, _) = translate_only(&module).unwrap();

    assert_eq!(lir_func.signature.params[0], Type::I64);
    assert_eq!(lir_func.signature.params[1], Type::I32);
}

// ===========================================================================
// Adapter-level: verify block parameter passing
// ===========================================================================

#[test]
fn test_adapter_block_params_for_loop() {
    let module = build_sum();
    let (lir_func, _) = translate_only(&module).unwrap();

    let loop_header_block = lir_func
        .blocks
        .iter()
        .find(|(_, bb)| bb.params.len() == 2)
        .expect("Should have a block with 2 params (loop header)");

    assert_eq!(loop_header_block.1.params.len(), 2);
    assert_eq!(loop_header_block.1.params[0].1, Type::I32);
    assert_eq!(loop_header_block.1.params[1].1, Type::I32);
}

// ===========================================================================
// Test 7: Direct function call
// ===========================================================================

fn build_call_module() -> TmirModule {
    let mut module = TmirModule::new("call_test");
    let shared_sig: FuncTyId = module.add_func_type(func_ty(vec![Ty::I32], vec![Ty::I32]));

    let mut callee = TmirFunction::new(f(0), "callee", shared_sig, b(0));
    callee.blocks.push(TmirBlock {
        id: b(0),
        params: vec![(v(0), Ty::I32)],
        body: vec![InstrNode::new(Inst::Return { values: vec![v(0)] })],
    });

    let mut caller = TmirFunction::new(f(1), "caller", shared_sig, b(0));
    caller.blocks.push(TmirBlock {
        id: b(0),
        params: vec![(v(0), Ty::I32)],
        body: vec![
            InstrNode::new(Inst::Call {
                callee: f(0),
                args: vec![v(0)],
            })
            .with_result(v(1)),
            InstrNode::new(Inst::Return { values: vec![v(1)] }),
        ],
    });

    module.add_function(callee);
    module.add_function(caller);
    module
}

#[test]
fn test_call_adapter() {
    let module = build_call_module();
    let results = translate_module(&module).unwrap();
    assert_eq!(results.len(), 2);

    let (callee_func, _) = &results[0];
    assert_eq!(callee_func.name, "callee");

    let (caller_func, _) = &results[1];
    assert_eq!(caller_func.name, "caller");

    let entry = &caller_func.blocks[&caller_func.entry_block];
    assert_eq!(entry.instructions.len(), 2);
    assert!(
        matches!(&entry.instructions[0].opcode, Opcode::Call { name } if name == "callee"),
        "Expected Call to 'callee', got {:?}",
        entry.instructions[0].opcode
    );
}

#[test]
fn test_call_isel() {
    let module = build_call_module();
    let results = translate_module(&module).unwrap();

    let (caller_lir, _) = &results[1];
    let mut isel =
        InstructionSelector::new(caller_lir.name.clone(), caller_lir.signature.clone());
    isel.lower_formal_arguments(&caller_lir.signature, caller_lir.entry_block)
        .unwrap();

    let mut block_order: Vec<Block> = caller_lir.blocks.keys().copied().collect();
    block_order.sort_by_key(|b| if *b == caller_lir.entry_block { 0 } else { b.0 + 1 });

    for block_id in &block_order {
        let bb = &caller_lir.blocks[block_id];
        isel.select_block(*block_id, &bb.instructions).unwrap();
    }

    let mfunc = isel.finalize();
    assert!(
        has_opcode(&mfunc, AArch64Opcode::Bl),
        "Expected BL instruction for direct call"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test 8: Switch
// ===========================================================================

fn build_switch() -> TmirModule {
    single_function_module(
        10,
        "dispatch",
        func_ty(vec![Ty::I32], vec![Ty::I32]),
        vec![
            TmirBlock {
                id: b(0),
                params: vec![(v(0), Ty::I32)],
                body: vec![InstrNode::new(Inst::Switch {
                    value: v(0),
                    default: b(3),
                    default_args: vec![],
                    cases: vec![
                        SwitchCase {
                            value: Constant::Int(0),
                            target: b(1),
                            args: vec![],
                        },
                        SwitchCase {
                            value: Constant::Int(1),
                            target: b(2),
                            args: vec![],
                        },
                    ],
                })],
            },
            TmirBlock {
                id: b(1),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I32,
                        value: Constant::Int(10),
                    })
                    .with_result(v(1)),
                    InstrNode::new(Inst::Return { values: vec![v(1)] }),
                ],
            },
            TmirBlock {
                id: b(2),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I32,
                        value: Constant::Int(20),
                    })
                    .with_result(v(2)),
                    InstrNode::new(Inst::Return { values: vec![v(2)] }),
                ],
            },
            TmirBlock {
                id: b(3),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I32,
                        value: Constant::Int(30),
                    })
                    .with_result(v(3)),
                    InstrNode::new(Inst::Return { values: vec![v(3)] }),
                ],
            },
        ],
        vec![],
    )
}

#[test]
fn test_switch_adapter() {
    let module = build_switch();
    let (lir_func, _) = translate_only(&module).unwrap();

    assert_eq!(lir_func.name, "dispatch");
    assert_eq!(lir_func.blocks.len(), 4);

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 1);
    match &entry.instructions[0].opcode {
        Opcode::Switch { cases, default: _ } => {
            assert_eq!(cases.len(), 2, "Expected 2 switch cases");
            assert_eq!(cases[0].0, 0);
            assert_eq!(cases[1].0, 1);
        }
        other => panic!("Expected Switch opcode, got {:?}", other),
    }
}

#[test]
fn test_switch_isel() {
    let module = build_switch();
    let mfunc = compile_tmir_function(&module);

    assert_eq!(mfunc.blocks.len(), 4);
    assert!(
        has_opcode(&mfunc, AArch64Opcode::CmpRR) || has_opcode(&mfunc, AArch64Opcode::CmpRI),
        "Expected CMP instruction for switch case comparison"
    );
    assert!(
        has_opcode(&mfunc, AArch64Opcode::BCond),
        "Expected B.EQ for switch case branch"
    );
    assert!(
        count_opcode(&mfunc, AArch64Opcode::Ret) >= 3,
        "Expected at least 3 RET instructions (case 0, case 1, default)"
    );
}

// ===========================================================================
// Test 9: CallIndirect
// ===========================================================================

fn build_call_indirect() -> TmirModule {
    let mut module = TmirModule::new("call_through_ptr");
    let callee_sig: FuncTyId = module.add_func_type(func_ty(vec![Ty::I32], vec![Ty::I32]));
    let func_sig: FuncTyId = module.add_func_type(func_ty(vec![Ty::Ptr, Ty::I32], vec![Ty::I32]));

    let mut func = TmirFunction::new(f(11), "call_through_ptr", func_sig, b(0));
    func.blocks.push(TmirBlock {
        id: b(0),
        params: vec![(v(0), Ty::Ptr), (v(1), Ty::I32)],
        body: vec![
            InstrNode::new(Inst::CallIndirect {
                callee: v(0),
                sig: callee_sig,
                args: vec![v(1)],
            })
            .with_result(v(2)),
            InstrNode::new(Inst::Return { values: vec![v(2)] }),
        ],
    });

    module.add_function(func);
    module
}

#[test]
fn test_call_indirect_adapter() {
    let module = build_call_indirect();
    let (lir_func, _) = translate_only(&module).unwrap();

    assert_eq!(lir_func.name, "call_through_ptr");

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 2);
    assert!(
        matches!(&entry.instructions[0].opcode, Opcode::CallIndirect),
        "Expected CallIndirect opcode, got {:?}",
        entry.instructions[0].opcode
    );
    assert_eq!(
        entry.instructions[0].args.len(),
        2,
        "CallIndirect should have fn_ptr + 1 arg = 2 total args"
    );
}

#[test]
fn test_call_indirect_isel() {
    let module = build_call_indirect();
    let mfunc = compile_tmir_function(&module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::Blr),
        "Expected BLR instruction for indirect call"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test 10: Select
// ===========================================================================

fn build_select() -> TmirModule {
    single_function_module(
        12,
        "abs_select",
        func_ty(vec![Ty::I32], vec![Ty::I32]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I32)],
            body: vec![
                InstrNode::new(Inst::UnOp {
                    op: UnOp::Neg,
                    ty: Ty::I32,
                    operand: v(0),
                })
                .with_result(v(1)),
                InstrNode::new(Inst::Const {
                    ty: Ty::I32,
                    value: Constant::Int(0),
                })
                .with_result(v(2)),
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Slt,
                    ty: Ty::I32,
                    lhs: v(0),
                    rhs: v(2),
                })
                .with_result(v(3)),
                InstrNode::new(Inst::Select {
                    ty: Ty::I32,
                    cond: v(3),
                    then_val: v(1),
                    else_val: v(0),
                })
                .with_result(v(4)),
                InstrNode::new(Inst::Return { values: vec![v(4)] }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_select_adapter() {
    let module = build_select();
    let (lir_func, _) = translate_only(&module).unwrap();

    assert_eq!(lir_func.name, "abs_select");
    let entry = &lir_func.blocks[&lir_func.entry_block];

    assert!(
        entry.instructions.len() >= 4,
        "Expected at least 4 instructions for select pattern, got {}",
        entry.instructions.len()
    );

    let has_select = entry
        .instructions
        .iter()
        .any(|inst| matches!(&inst.opcode, Opcode::Select { .. }));
    assert!(has_select, "Expected Select opcode in instruction stream");
}

#[test]
fn test_select_isel() {
    let module = build_select();
    let mfunc = compile_tmir_function(&module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::Csel),
        "Expected CSEL instruction for Select"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test 11: GEP
// ===========================================================================

fn build_gep() -> TmirModule {
    single_function_module(
        13,
        "array_access",
        func_ty(vec![Ty::Ptr, Ty::I64], vec![Ty::I32]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::Ptr), (v(1), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::GEP {
                    pointee_ty: Ty::I32,
                    base: v(0),
                    indices: vec![v(1)],
                })
                .with_result(v(2)),
                InstrNode::new(Inst::Load {
                    ty: Ty::I32,
                    ptr: v(2),
                })
                .with_result(v(3)),
                InstrNode::new(Inst::Return { values: vec![v(3)] }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_gep_adapter() {
    let module = build_gep();
    let (lir_func, _) = translate_only(&module).unwrap();

    assert_eq!(lir_func.name, "array_access");
    let entry = &lir_func.blocks[&lir_func.entry_block];

    assert!(
        entry.instructions.len() >= 4,
        "Expected at least 4 instructions for GEP+Load+Return, got {}",
        entry.instructions.len()
    );

    let has_load = entry
        .instructions
        .iter()
        .any(|inst| matches!(&inst.opcode, Opcode::Load { .. }));
    assert!(has_load, "Expected Load opcode after GEP");
}

#[test]
fn test_gep_isel() {
    let module = build_gep();
    let mfunc = compile_tmir_function(&module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::LdrRI) || has_opcode(&mfunc, AArch64Opcode::LdrRO),
        "Expected LDR for load after GEP"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test 12: GEP with explicit byte offset
// ===========================================================================

fn build_gep_with_offset() -> TmirModule {
    single_function_module(
        14,
        "struct_field_in_array",
        func_ty(vec![Ty::Ptr, Ty::I64], vec![Ty::I32]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::Ptr), (v(1), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::GEP {
                    pointee_ty: Ty::I64,
                    base: v(0),
                    indices: vec![v(1)],
                })
                .with_result(v(2)),
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(4),
                })
                .with_result(v(3)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::Ptr,
                    lhs: v(2),
                    rhs: v(3),
                })
                .with_result(v(4)),
                InstrNode::new(Inst::Load {
                    ty: Ty::I32,
                    ptr: v(4),
                })
                .with_result(v(5)),
                InstrNode::new(Inst::Return { values: vec![v(5)] }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_gep_with_offset_adapter() {
    let module = build_gep_with_offset();
    let (lir_func, _) = translate_only(&module).unwrap();

    assert_eq!(lir_func.name, "struct_field_in_array");
    let entry = &lir_func.blocks[&lir_func.entry_block];

    assert!(
        entry.instructions.len() >= 6,
        "Expected at least 6 instructions for GEP with offset, got {}",
        entry.instructions.len()
    );
}

#[test]
fn test_gep_with_offset_isel() {
    let module = build_gep_with_offset();
    let mfunc = compile_tmir_function(&module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::MulRR),
        "Expected MUL for index scaling in GEP"
    );
    assert!(
        has_opcode(&mfunc, AArch64Opcode::LdrRI) || has_opcode(&mfunc, AArch64Opcode::LdrRO),
        "Expected LDR for load after GEP"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test 13: Type casts
// ===========================================================================

fn build_cast_chain() -> TmirModule {
    single_function_module(
        15,
        "widen_and_narrow",
        func_ty(vec![Ty::I8], vec![Ty::I8]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I8)],
            body: vec![
                InstrNode::new(Inst::Cast {
                    op: CastOp::SExt,
                    src_ty: Ty::I8,
                    dst_ty: Ty::I32,
                    operand: v(0),
                })
                .with_result(v(1)),
                InstrNode::new(Inst::Cast {
                    op: CastOp::Trunc,
                    src_ty: Ty::I32,
                    dst_ty: Ty::I8,
                    operand: v(1),
                })
                .with_result(v(2)),
                InstrNode::new(Inst::Return { values: vec![v(2)] }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_cast_chain_adapter() {
    let module = build_cast_chain();
    let (lir_func, _) = translate_only(&module).unwrap();

    assert_eq!(lir_func.name, "widen_and_narrow");
    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 3);
    assert!(matches!(
        &entry.instructions[0].opcode,
        Opcode::Sextend {
            from_ty: Type::I8,
            to_ty: Type::I32
        }
    ));
    assert!(matches!(
        &entry.instructions[1].opcode,
        Opcode::Trunc { to_ty: Type::I8 }
    ));
}

#[test]
fn test_cast_chain_isel() {
    let module = build_cast_chain();
    let mfunc = compile_tmir_function(&module);

    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
    assert!(
        total_insts(&mfunc) >= 3,
        "Expected at least 3 machine instructions (COPY, SXTB, AND/TRUNC, RET)"
    );
}

// ===========================================================================
// Test 14: Float arithmetic + FP cast
// ===========================================================================

fn build_float_to_int() -> TmirModule {
    single_function_module(
        16,
        "float_to_int",
        func_ty(vec![Ty::F64, Ty::F64], vec![Ty::I32]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::F64), (v(1), Ty::F64)],
            body: vec![
                InstrNode::new(Inst::BinOp {
                    op: BinOp::FAdd,
                    ty: Ty::F64,
                    lhs: v(0),
                    rhs: v(1),
                })
                .with_result(v(2)),
                InstrNode::new(Inst::Cast {
                    op: CastOp::FPToSI,
                    src_ty: Ty::F64,
                    dst_ty: Ty::I32,
                    operand: v(2),
                })
                .with_result(v(3)),
                InstrNode::new(Inst::Return { values: vec![v(3)] }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_float_to_int_adapter() {
    let module = build_float_to_int();
    let (lir_func, _) = translate_only(&module).unwrap();

    assert_eq!(lir_func.name, "float_to_int");
    assert_eq!(lir_func.signature.params, vec![Type::F64, Type::F64]);
    assert_eq!(lir_func.signature.returns, vec![Type::I32]);

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 3);
}

#[test]
fn test_float_to_int_isel() {
    let module = build_float_to_int();
    let mfunc = compile_tmir_function(&module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::FaddRR),
        "Expected FADD for f64 addition"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Proof annotations survive adapter translation
// ===========================================================================

#[test]
fn test_proof_annotations_survive() {
    let module = single_function_module(
        22,
        "proven_add",
        func_ty(vec![Ty::I32, Ty::I32], vec![Ty::I32]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I32), (v(1), Ty::I32)],
            body: vec![
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I32,
                    lhs: v(0),
                    rhs: v(1),
                })
                .with_result(v(2))
                .with_proof(ProofAnnotation::NoOverflow),
                InstrNode::new(Inst::Return { values: vec![v(2)] }),
            ],
        }],
        vec![ProofAnnotation::Pure],
    );

    let (_, proof_ctx) = translate_only(&module).unwrap();

    assert!(
        proof_ctx.is_function_pure(),
        "Function-level Pure proof should survive adapter translation"
    );

    let has_no_overflow = proof_ctx.value_proofs.values().any(|proofs| {
        proofs
            .iter()
            .any(|p| matches!(p, llvm2_lower::adapter::Proof::NoOverflow { signed: true }))
    });
    assert!(
        has_no_overflow,
        "NoOverflow proof should be attached to the add result"
    );
}

// ===========================================================================
// Comprehensive: all new programs translate through adapter without errors
// ===========================================================================

#[test]
fn test_all_new_programs_adapter_succeeds() {
    let programs: Vec<(&str, TmirModule)> = vec![
        ("dispatch", build_switch()),
        ("call_through_ptr", build_call_indirect()),
        ("abs_select", build_select()),
        ("array_access", build_gep()),
        ("struct_field_in_array", build_gep_with_offset()),
        ("widen_and_narrow", build_cast_chain()),
        ("float_to_int", build_float_to_int()),
    ];

    for (name, module) in &programs {
        let result = translate_only(module);
        assert!(
            result.is_ok(),
            "{}: adapter translation failed: {:?}",
            name,
            result.err()
        );
        let (lir_func, _) = result.unwrap();
        assert_eq!(lir_func.name, *name);
        assert!(!lir_func.blocks.is_empty());
    }
}

// ===========================================================================
// Comprehensive: all new single-function programs compile through ISel
// ===========================================================================

#[test]
fn test_all_new_single_block_programs_compile() {
    let programs: Vec<(&str, TmirModule)> = vec![
        ("call_through_ptr", build_call_indirect()),
        ("abs_select", build_select()),
        ("array_access", build_gep()),
        ("struct_field_in_array", build_gep_with_offset()),
        ("widen_and_narrow", build_cast_chain()),
        ("float_to_int", build_float_to_int()),
    ];

    for (name, module) in &programs {
        let mfunc = compile_tmir_function(module);
        assert!(
            !mfunc.blocks.is_empty(),
            "{}: produced empty ISelFunction",
            name
        );
        assert!(
            total_insts(&mfunc) > 0,
            "{}: produced no machine instructions",
            name
        );
        assert!(
            has_opcode(&mfunc, AArch64Opcode::Ret),
            "{}: missing RET instruction",
            name
        );
    }
}
