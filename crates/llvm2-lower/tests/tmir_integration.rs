// tmir_integration.rs - End-to-end tMIR -> adapter -> ISel integration tests
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

use llvm2_lower::adapter::{translate_function, translate_module, AdapterError, ProofContext};
use llvm2_lower::function::Function;
use llvm2_lower::instructions::{Block, Instruction, IntCC, Opcode};
use llvm2_lower::isel::{AArch64Opcode, ISelFunction, InstructionSelector};
use llvm2_lower::types::Type;

use tmir::{
    BinOp, Block as TmirBlock, BlockId, CastOp, Constant, FuncId, FuncTy, FuncTyId,
    Function as TmirFunction, ICmpOp, Inst, InstrNode, Module as TmirModule, OverflowOp,
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

    // Seed Value->Type hints (Call/CallIndirect result types, #381).
    isel.seed_value_types(&lir_func.value_types);

    isel.lower_formal_arguments(&lir_func.signature, lir_func.entry_block)
        .unwrap();

    let mut block_order: Vec<Block> = lir_func.blocks.keys().copied().collect();
    block_order.sort_by_key(|b| if *b == lir_func.entry_block { 0 } else { b.0 + 1 });

    for block_id in &block_order {
        let bb = &lir_func.blocks[block_id];
        isel.select_block_with_source_locs(*block_id, &bb.instructions, &bb.source_locs)
            .unwrap();
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
                    volatile: false,
                    align: None,
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
                    volatile: false,
                    align: None,
                })
                .with_result(v(1)),
                InstrNode::new(Inst::Store {
                    ty: Ty::I32,
                    ptr: v(0),
                    value: v(1),
                    volatile: false,
                    align: None,
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
    // After #381, the adapter propagates the callee signature's return type
    // onto the Call result Value, so I32 end-to-end is type-correct through
    // ISel's #307 Return check. (Historic note: this fixture was forced to
    // I64 in #380 as a workaround for the missing type propagation.)
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
    // Seed Value->Type hints (Call result types, #381).
    isel.seed_value_types(&caller_lir.value_types);
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
    // #381: adapter now propagates the callee signature's return types into
    // ISel's value_types map, so I32 call results are tracked correctly and
    // no longer fall back to I64.
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
                    volatile: false,
                    align: None,
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
                    volatile: false,
                    align: None,
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
// #456: ProofAnnotation::Pure propagates through call lowering to the Bl
// ===========================================================================

/// Build a two-function module whose callee carries `ProofAnnotation::Pure`
/// and whose caller invokes it via `Inst::Call`. Returns the module.
fn build_pure_call_module() -> TmirModule {
    let mut module = TmirModule::new("pure_call_test");
    let sig: FuncTyId = module.add_func_type(func_ty(vec![Ty::I32], vec![Ty::I32]));

    // Pure callee: return arg unchanged.
    let mut callee = TmirFunction::new(f(0), "pure_callee", sig, b(0));
    callee.blocks.push(TmirBlock {
        id: b(0),
        params: vec![(v(0), Ty::I32)],
        body: vec![InstrNode::new(Inst::Return { values: vec![v(0)] })],
    });
    callee.proofs = vec![ProofAnnotation::Pure];

    // Caller: call the pure callee.
    let mut caller = TmirFunction::new(f(1), "caller", sig, b(0));
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
fn test_pure_callee_surfaces_in_lir_function() {
    let module = build_pure_call_module();
    let results = translate_module(&module).unwrap();
    assert_eq!(results.len(), 2);

    let (caller_lir, _) = results
        .iter()
        .find(|(f, _)| f.name == "caller")
        .expect("caller must be present");

    assert!(
        caller_lir.pure_callees.contains("pure_callee"),
        "adapter should record 'pure_callee' in Function::pure_callees, got {:?}",
        caller_lir.pure_callees
    );
}

#[test]
fn test_pure_callee_bl_carries_proof_pure() {
    use llvm2_ir::ProofAnnotation as IrProof;

    let module = build_pure_call_module();
    let results = translate_module(&module).unwrap();

    let (caller_lir, _) = results
        .iter()
        .find(|(f, _)| f.name == "caller")
        .expect("caller must be present");

    let mut isel =
        InstructionSelector::new(caller_lir.name.clone(), caller_lir.signature.clone());
    isel.seed_value_types(&caller_lir.value_types);
    isel.seed_pure_callees(&caller_lir.pure_callees);
    isel.lower_formal_arguments(&caller_lir.signature, caller_lir.entry_block)
        .unwrap();

    let mut block_order: Vec<Block> = caller_lir.blocks.keys().copied().collect();
    block_order.sort_by_key(|b| if *b == caller_lir.entry_block { 0 } else { b.0 + 1 });
    for block_id in &block_order {
        let bb = &caller_lir.blocks[block_id];
        isel.select_block_with_source_locs(*block_id, &bb.instructions, &bb.source_locs)
            .unwrap();
    }

    // Convert to canonical MachFunction so we can read MachInst.proof.
    let isel_func = isel.finalize();
    let mfunc = isel_func.to_ir_func();

    // Find the Bl to `pure_callee` and confirm it carries Some(Pure).
    let mut found_pure_bl = false;
    for block in &mfunc.blocks {
        for inst_id in &block.insts {
            let inst = &mfunc.insts[inst_id.0 as usize];
            if inst.opcode == AArch64Opcode::Bl {
                let is_pure_callee = inst.operands.iter().any(|op| {
                    matches!(op, llvm2_ir::MachOperand::Symbol(s) if s == "pure_callee")
                });
                if is_pure_callee {
                    assert_eq!(
                        inst.proof,
                        Some(IrProof::Pure),
                        "Bl to pure_callee must carry ProofAnnotation::Pure for SROA \
                         partial-escape (#456); got {:?}",
                        inst.proof,
                    );
                    found_pure_bl = true;
                }
            }
        }
    }
    assert!(
        found_pure_bl,
        "expected a Bl to 'pure_callee' in the caller's MachFunction"
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

// ===========================================================================
// Test: Source location propagation through full pipeline
// ===========================================================================

use tmir::SourceSpan;

/// Build a tMIR function with source spans on instructions, compile through
/// the full pipeline, and verify the spans appear on MachInsts.
#[test]
fn test_source_loc_end_to_end() {
    let module = single_function_module(
        0,
        "add_with_locs",
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
                .with_span(SourceSpan { file: 0, line: 10, col: 5 }),
                InstrNode::new(Inst::Return { values: vec![v(2)] })
                    .with_span(SourceSpan { file: 0, line: 11, col: 1 }),
            ],
        }],
        vec![],
    );

    let isel_func = compile_tmir_function(&module);

    // Verify source_locs are present on the ISelBlock.
    let entry_block = isel_func.blocks.values().next().unwrap();
    let has_line_10 = entry_block.source_locs.iter().any(|loc| {
        *loc == Some(llvm2_ir::SourceLoc { file: 0, line: 10, col: 5 })
    });
    assert!(has_line_10, "ISel output should carry line 10 source loc from tMIR span");

    let has_line_11 = entry_block.source_locs.iter().any(|loc| {
        *loc == Some(llvm2_ir::SourceLoc { file: 0, line: 11, col: 1 })
    });
    assert!(has_line_11, "ISel output should carry line 11 source loc from tMIR span");

    // Verify propagation through to_ir_func().
    let ir_func = isel_func.to_ir_func();
    let ir_has_line_10 = ir_func.insts.iter().any(|inst| {
        inst.source_loc == Some(llvm2_ir::SourceLoc { file: 0, line: 10, col: 5 })
    });
    assert!(ir_has_line_10, "MachInsts should carry line 10 source loc after to_ir_func()");

    let ir_has_line_11 = ir_func.insts.iter().any(|inst| {
        inst.source_loc == Some(llvm2_ir::SourceLoc { file: 0, line: 11, col: 1 })
    });
    assert!(ir_has_line_11, "MachInsts should carry line 11 source loc after to_ir_func()");
}

/// Verify that instructions without spans get None source_loc.
#[test]
fn test_source_loc_none_for_spanless_instrs() {
    let module = single_function_module(
        0,
        "no_spans",
        func_ty(vec![Ty::I32], vec![Ty::I32]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I32)],
            body: vec![
                // No .with_span() — span is None.
                InstrNode::new(Inst::Return { values: vec![v(0)] }),
            ],
        }],
        vec![],
    );

    let isel_func = compile_tmir_function(&module);
    let entry_block = isel_func.blocks.values().next().unwrap();

    // All source_locs should be None.
    assert!(
        entry_block.source_locs.iter().all(|loc| loc.is_none()),
        "ISel output should have None source_locs for instructions without spans"
    );
}

// ===========================================================================
// Test: Diamond CFG with merge via block parameters (if-then-else pattern)
//
// This exercises the pattern from tla2's BigToSmall/SmallToBig actions:
//   abs_diff(x: i32, y: i32) -> i32 {
//     if x > y { x - y } else { y - x }
//   }
//
// tMIR form:
//   entry(x: i32, y: i32):
//     cond = icmp sgt x, y
//     condbr cond -> then_block(), else_block()
//   then_block():
//     diff = isub x, y
//     br merge(diff)
//   else_block():
//     diff2 = isub y, x
//     br merge(diff2)
//   merge(result: i32):
//     return result
// ===========================================================================

fn build_diamond_merge() -> TmirModule {
    single_function_module(
        50,
        "abs_diff",
        func_ty(vec![Ty::I32, Ty::I32], vec![Ty::I32]),
        vec![
            // entry block: compare and branch
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
            // then block: x - y, branch to merge with result
            TmirBlock {
                id: b(1),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Sub,
                        ty: Ty::I32,
                        lhs: v(0),
                        rhs: v(1),
                    })
                    .with_result(v(3)),
                    InstrNode::new(Inst::Br {
                        target: b(3),
                        args: vec![v(3)],
                    }),
                ],
            },
            // else block: y - x, branch to merge with result
            TmirBlock {
                id: b(2),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Sub,
                        ty: Ty::I32,
                        lhs: v(1),
                        rhs: v(0),
                    })
                    .with_result(v(4)),
                    InstrNode::new(Inst::Br {
                        target: b(3),
                        args: vec![v(4)],
                    }),
                ],
            },
            // merge block: receive result via block parameter, return it
            TmirBlock {
                id: b(3),
                params: vec![(v(5), Ty::I32)],
                body: vec![InstrNode::new(Inst::Return { values: vec![v(5)] })],
            },
        ],
        vec![],
    )
}

#[test]
fn test_diamond_merge_adapter() {
    let module = build_diamond_merge();
    let (lir_func, _) = translate_only(&module).unwrap();

    // 4 blocks: entry, then, else, merge
    assert_eq!(lir_func.blocks.len(), 4);

    let entry = &lir_func.blocks[&lir_func.entry_block];
    // entry: ICmp + Brif = 2 instructions
    assert_eq!(entry.instructions.len(), 2);
    assert!(matches!(entry.instructions[0].opcode, Opcode::Icmp { .. }));
    assert!(matches!(entry.instructions[1].opcode, Opcode::Brif { .. }));

    // Verify merge block has exactly 1 parameter.
    // The entry block also has params (function args), so filter it out.
    let merge_block = lir_func
        .blocks
        .iter()
        .filter(|(id, _)| **id != lir_func.entry_block)
        .map(|(_, bb)| bb)
        .find(|bb| !bb.params.is_empty())
        .expect("merge block should have block parameters");
    assert_eq!(merge_block.params.len(), 1, "merge block should have exactly 1 param");

    // Verify then and else blocks each have: sub + copy + jump = 3 instructions.
    // Filter out entry (has function params) and merge (has block params).
    let branch_blocks: Vec<_> = lir_func
        .blocks
        .iter()
        .filter(|(id, bb)| **id != lir_func.entry_block && bb.params.is_empty())
        .map(|(_, bb)| bb)
        .collect();
    assert_eq!(branch_blocks.len(), 2, "should have 2 branch blocks");

    for bb in &branch_blocks {
        assert_eq!(
            bb.instructions.len(),
            3,
            "branch block should have sub + copy + jump, got {:?}",
            bb.instructions.iter().map(|i| &i.opcode).collect::<Vec<_>>()
        );
        assert!(matches!(bb.instructions[0].opcode, Opcode::Isub));
        // Copy pseudo (block-arg passing; see #417).
        assert!(matches!(bb.instructions[1].opcode, Opcode::Copy));
        assert_eq!(bb.instructions[1].args.len(), 1, "copy should have 1 arg");
        assert!(matches!(bb.instructions[2].opcode, Opcode::Jump { .. }));
    }

    // Both copies should target the same destination (the merge block param)
    let then_copy_dst = branch_blocks[0].instructions[1].results[0];
    let else_copy_dst = branch_blocks[1].instructions[1].results[0];
    assert_eq!(
        then_copy_dst, else_copy_dst,
        "both branch copies should target the same merge block parameter"
    );

    // The copy destinations should be different from the copy sources
    let then_copy_src = branch_blocks[0].instructions[1].args[0];
    let else_copy_src = branch_blocks[1].instructions[1].args[0];
    assert_ne!(
        then_copy_src, else_copy_src,
        "branch copies should use different source values"
    );
}

#[test]
fn test_diamond_merge_isel() {
    let module = build_diamond_merge();
    let mfunc = compile_tmir_function(&module);

    // Should have 4 blocks
    assert_eq!(mfunc.blocks.len(), 4, "diamond merge should produce 4 ISel blocks");

    // Should have comparison, conditional branch, 2 subs, and return
    assert!(
        has_opcode(&mfunc, AArch64Opcode::CmpRR),
        "Expected CMP for the comparison"
    );
    assert!(
        has_opcode(&mfunc, AArch64Opcode::BCond),
        "Expected B.cond for the conditional branch"
    );
    assert!(
        count_opcode(&mfunc, AArch64Opcode::SubRR) >= 2,
        "Expected at least 2 SUB instructions (then + else branches)"
    );
    assert!(
        has_opcode(&mfunc, AArch64Opcode::Ret),
        "Expected RET instruction"
    );
}

// ===========================================================================
// Test: Diamond CFG with CondBr block args (direct arg passing on branch)
//
// This tests the pattern where CondBr passes different values to the SAME
// merge block parameter on each edge, going through the full ISel pipeline.
// Exercises the #302 copy-block fix end-to-end.
//
// tMIR form:
//   entry(cond: bool, x: i32, y: i32):
//     condbr cond -> merge(x), merge(y)
//   merge(result: i32):
//     return result
// ===========================================================================

fn build_condbr_merge() -> TmirModule {
    let mut module = TmirModule::new("condbr_merge");
    let ft_id = module.add_func_type(func_ty(vec![Ty::Bool, Ty::I32, Ty::I32], vec![Ty::I32]));

    let mut func = TmirFunction::new(f(51), "select_via_branch", ft_id, b(0));
    func.blocks = vec![
        // entry: condbr with args to same merge block
        TmirBlock {
            id: b(0),
            params: vec![
                (v(0), Ty::Bool),  // cond
                (v(1), Ty::I32),   // x
                (v(2), Ty::I32),   // y
            ],
            body: vec![InstrNode::new(Inst::CondBr {
                cond: v(0),
                then_target: b(1),
                then_args: vec![v(1)],   // pass x to merge
                else_target: b(1),
                else_args: vec![v(2)],   // pass y to merge
            })],
        },
        // merge block: receives result, returns it
        TmirBlock {
            id: b(1),
            params: vec![(v(3), Ty::I32)],
            body: vec![InstrNode::new(Inst::Return { values: vec![v(3)] })],
        },
    ];

    module.add_function(func);
    module
}

#[test]
fn test_condbr_merge_adapter() {
    let module = build_condbr_merge();
    let (lir_func, _) = translate_only(&module).unwrap();

    // 4 blocks: entry + merge + 2 copy blocks
    assert_eq!(lir_func.blocks.len(), 4,
        "expected 4 blocks (entry + merge + 2 copy blocks), got {}",
        lir_func.blocks.len()
    );
}

#[test]
fn test_condbr_merge_isel() {
    let module = build_condbr_merge();
    let mfunc = compile_tmir_function(&module);

    // Should have 4 blocks: entry, merge, 2 copy blocks
    assert_eq!(mfunc.blocks.len(), 4,
        "condbr merge should produce 4 ISel blocks, got {}",
        mfunc.blocks.len()
    );

    assert!(
        has_opcode(&mfunc, AArch64Opcode::BCond),
        "Expected conditional branch"
    );
    assert!(
        has_opcode(&mfunc, AArch64Opcode::Ret),
        "Expected RET instruction"
    );
    // Copy blocks should have MOV instructions for the phi copies
    assert!(
        count_opcode(&mfunc, AArch64Opcode::MovR) >= 2,
        "Expected at least 2 MOV instructions for copy blocks"
    );
}

// ===========================================================================
// tla-tmir integration tests (issue #339)
//
// These tests verify that llvm2-lower handles all tMIR instruction types
// emitted by tla-tmir (the TLA+ bytecode -> tMIR lowering crate).
// tla2 uses i64 as its primary integer type throughout.
// ===========================================================================

// ---------------------------------------------------------------------------
// Category 1: Scalar arithmetic on i64 (Mul, SDiv, SRem)
// Add and Sub already tested above; these cover the remaining ops tla2 uses.
// ---------------------------------------------------------------------------

fn build_i64_mul() -> TmirModule {
    single_function_module(
        100,
        "i64_mul",
        func_ty(vec![Ty::I64, Ty::I64], vec![Ty::I64]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I64), (v(1), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Mul,
                    ty: Ty::I64,
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
fn test_i64_mul_adapter() {
    let module = build_i64_mul();
    let (lir_func, _) = translate_only(&module).unwrap();

    assert_eq!(lir_func.signature.params, vec![Type::I64, Type::I64]);
    assert_eq!(lir_func.signature.returns, vec![Type::I64]);

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 2);
    assert!(matches!(entry.instructions[0].opcode, Opcode::Imul));
}

#[test]
fn test_i64_mul_isel() {
    let module = build_i64_mul();
    let mfunc = compile_tmir_function(&module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::MulRR),
        "Expected MUL instruction for i64 multiplication"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

fn build_i64_sdiv() -> TmirModule {
    single_function_module(
        101,
        "i64_sdiv",
        func_ty(vec![Ty::I64, Ty::I64], vec![Ty::I64]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I64), (v(1), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::BinOp {
                    op: BinOp::SDiv,
                    ty: Ty::I64,
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
fn test_i64_sdiv_adapter() {
    let module = build_i64_sdiv();
    let (lir_func, _) = translate_only(&module).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 2);
    assert!(matches!(entry.instructions[0].opcode, Opcode::Sdiv));
}

#[test]
fn test_i64_sdiv_isel() {
    let module = build_i64_sdiv();
    let mfunc = compile_tmir_function(&module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::SDiv),
        "Expected SDIV instruction for signed i64 division"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

fn build_i64_srem() -> TmirModule {
    single_function_module(
        102,
        "i64_srem",
        func_ty(vec![Ty::I64, Ty::I64], vec![Ty::I64]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I64), (v(1), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::BinOp {
                    op: BinOp::SRem,
                    ty: Ty::I64,
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
fn test_i64_srem_adapter() {
    let module = build_i64_srem();
    let (lir_func, _) = translate_only(&module).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 2);
    assert!(matches!(entry.instructions[0].opcode, Opcode::Srem));
}

#[test]
fn test_i64_srem_isel() {
    let module = build_i64_srem();
    let mfunc = compile_tmir_function(&module);

    // SREM is lowered as: SDIV tmp, a, b; MSUB result, tmp, b, a
    assert!(
        has_opcode(&mfunc, AArch64Opcode::SDiv),
        "Expected SDIV as part of SREM lowering"
    );
    assert!(
        has_opcode(&mfunc, AArch64Opcode::Msub),
        "Expected MSUB as part of SREM lowering (result = a - (a/b)*b)"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ---------------------------------------------------------------------------
// Category 2: ICmp with all comparison operators used by tla2
// ---------------------------------------------------------------------------

fn build_icmp_variant(op: ICmpOp, name: &str, func_id: u32) -> TmirModule {
    single_function_module(
        func_id,
        name,
        func_ty(vec![Ty::I64, Ty::I64], vec![Ty::I64]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I64), (v(1), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::ICmp {
                    op,
                    ty: Ty::I64,
                    lhs: v(0),
                    rhs: v(1),
                })
                .with_result(v(2)),
                // ZExt i1->i64 to return the boolean as an integer (tla2 pattern)
                InstrNode::new(Inst::Cast {
                    op: CastOp::ZExt,
                    src_ty: Ty::Bool,
                    dst_ty: Ty::I64,
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
fn test_icmp_eq_adapter_and_isel() {
    let module = build_icmp_variant(ICmpOp::Eq, "icmp_eq", 110);
    let (lir_func, _) = translate_only(&module).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert!(entry.instructions.iter().any(|i| matches!(
        &i.opcode,
        Opcode::Icmp { cond: IntCC::Equal }
    )));

    let mfunc = compile_tmir_function(&module);
    assert!(has_opcode(&mfunc, AArch64Opcode::CmpRR));
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

#[test]
fn test_icmp_ne_adapter_and_isel() {
    let module = build_icmp_variant(ICmpOp::Ne, "icmp_ne", 111);
    let (lir_func, _) = translate_only(&module).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert!(entry.instructions.iter().any(|i| matches!(
        &i.opcode,
        Opcode::Icmp { cond: IntCC::NotEqual }
    )));

    let mfunc = compile_tmir_function(&module);
    assert!(has_opcode(&mfunc, AArch64Opcode::CmpRR));
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

#[test]
fn test_icmp_slt_adapter_and_isel() {
    let module = build_icmp_variant(ICmpOp::Slt, "icmp_slt", 112);
    let (lir_func, _) = translate_only(&module).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert!(entry.instructions.iter().any(|i| matches!(
        &i.opcode,
        Opcode::Icmp { cond: IntCC::SignedLessThan }
    )));

    let mfunc = compile_tmir_function(&module);
    assert!(has_opcode(&mfunc, AArch64Opcode::CmpRR));
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

#[test]
fn test_icmp_sle_adapter_and_isel() {
    let module = build_icmp_variant(ICmpOp::Sle, "icmp_sle", 113);
    let (lir_func, _) = translate_only(&module).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert!(entry.instructions.iter().any(|i| matches!(
        &i.opcode,
        Opcode::Icmp {
            cond: IntCC::SignedLessThanOrEqual
        }
    )));

    let mfunc = compile_tmir_function(&module);
    assert!(has_opcode(&mfunc, AArch64Opcode::CmpRR));
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

#[test]
fn test_icmp_sge_adapter_and_isel() {
    let module = build_icmp_variant(ICmpOp::Sge, "icmp_sge", 114);
    let (lir_func, _) = translate_only(&module).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert!(entry.instructions.iter().any(|i| matches!(
        &i.opcode,
        Opcode::Icmp {
            cond: IntCC::SignedGreaterThanOrEqual
        }
    )));

    let mfunc = compile_tmir_function(&module);
    assert!(has_opcode(&mfunc, AArch64Opcode::CmpRR));
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ---------------------------------------------------------------------------
// Category 3: Boolean logic (And, Or, Xor, Not)
// tla2 uses these for TLA+ logical operators: /\, \/, ~
// ---------------------------------------------------------------------------

fn build_bool_and() -> TmirModule {
    single_function_module(
        120,
        "bool_and",
        func_ty(vec![Ty::I64, Ty::I64], vec![Ty::I64]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I64), (v(1), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::BinOp {
                    op: BinOp::And,
                    ty: Ty::I64,
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
fn test_bool_and_adapter() {
    let module = build_bool_and();
    let (lir_func, _) = translate_only(&module).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 2);
    assert!(matches!(entry.instructions[0].opcode, Opcode::Band));
}

#[test]
fn test_bool_and_isel() {
    let module = build_bool_and();
    let mfunc = compile_tmir_function(&module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::AndRR),
        "Expected AND instruction for bitwise AND"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

fn build_bool_or() -> TmirModule {
    single_function_module(
        121,
        "bool_or",
        func_ty(vec![Ty::I64, Ty::I64], vec![Ty::I64]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I64), (v(1), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Or,
                    ty: Ty::I64,
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
fn test_bool_or_adapter() {
    let module = build_bool_or();
    let (lir_func, _) = translate_only(&module).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 2);
    assert!(matches!(entry.instructions[0].opcode, Opcode::Bor));
}

#[test]
fn test_bool_or_isel() {
    let module = build_bool_or();
    let mfunc = compile_tmir_function(&module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::OrrRR),
        "Expected ORR instruction for bitwise OR"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

fn build_bool_xor() -> TmirModule {
    single_function_module(
        122,
        "bool_xor",
        func_ty(vec![Ty::I64, Ty::I64], vec![Ty::I64]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I64), (v(1), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Xor,
                    ty: Ty::I64,
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
fn test_bool_xor_adapter() {
    let module = build_bool_xor();
    let (lir_func, _) = translate_only(&module).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 2);
    assert!(matches!(entry.instructions[0].opcode, Opcode::Bxor));
}

#[test]
fn test_bool_xor_isel() {
    let module = build_bool_xor();
    let mfunc = compile_tmir_function(&module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::EorRR),
        "Expected EOR instruction for bitwise XOR"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

/// TLA+ negation (~) is lowered as UnOp::Not which maps to Bnot (MVN on AArch64)
fn build_bool_not() -> TmirModule {
    single_function_module(
        123,
        "bool_not",
        func_ty(vec![Ty::I64], vec![Ty::I64]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::UnOp {
                    op: UnOp::Not,
                    ty: Ty::I64,
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
fn test_bool_not_adapter() {
    let module = build_bool_not();
    let (lir_func, _) = translate_only(&module).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 2);
    assert!(matches!(entry.instructions[0].opcode, Opcode::Bnot));
}

#[test]
fn test_bool_not_isel() {
    let module = build_bool_not();
    let mfunc = compile_tmir_function(&module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::OrnRR),
        "Expected ORN (MVN) instruction for bitwise NOT"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ---------------------------------------------------------------------------
// Category 4: Casts - ZExt (i1->i64) for boolean promotion
// tla2 compares i64 values producing i1, then ZExts back to i64
// ---------------------------------------------------------------------------

fn build_zext_i1_to_i64() -> TmirModule {
    single_function_module(
        130,
        "zext_bool_to_i64",
        func_ty(vec![Ty::I64, Ty::I64], vec![Ty::I64]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I64), (v(1), Ty::I64)],
            body: vec![
                // Compare: result is i1
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Eq,
                    ty: Ty::I64,
                    lhs: v(0),
                    rhs: v(1),
                })
                .with_result(v(2)),
                // ZExt i1 -> i64 (boolean to integer promotion)
                InstrNode::new(Inst::Cast {
                    op: CastOp::ZExt,
                    src_ty: Ty::Bool,
                    dst_ty: Ty::I64,
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
fn test_zext_i1_to_i64_adapter() {
    let module = build_zext_i1_to_i64();
    let (lir_func, _) = translate_only(&module).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert!(entry.instructions.len() >= 3);
    assert!(entry.instructions.iter().any(|i| matches!(
        &i.opcode,
        Opcode::Icmp { cond: IntCC::Equal }
    )));
    assert!(entry.instructions.iter().any(|i| matches!(
        &i.opcode,
        Opcode::Uextend {
            from_ty: Type::B1,
            to_ty: Type::I64
        }
    )));
}

#[test]
fn test_zext_i1_to_i64_isel() {
    let module = build_zext_i1_to_i64();
    let mfunc = compile_tmir_function(&module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::CmpRR),
        "Expected CMP for comparison"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

/// Trunc i64 -> i1 (integer to boolean demotion, for feeding CondBr)
fn build_trunc_i64_to_i1() -> TmirModule {
    single_function_module(
        131,
        "trunc_i64_to_bool",
        func_ty(vec![Ty::I64], vec![Ty::I64]),
        vec![
            TmirBlock {
                id: b(0),
                params: vec![(v(0), Ty::I64)],
                body: vec![
                    // Trunc i64 -> Bool (take low bit)
                    InstrNode::new(Inst::Cast {
                        op: CastOp::Trunc,
                        src_ty: Ty::I64,
                        dst_ty: Ty::Bool,
                        operand: v(0),
                    })
                    .with_result(v(1)),
                    // Use the bool in a CondBr
                    InstrNode::new(Inst::CondBr {
                        cond: v(1),
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
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(1),
                    })
                    .with_result(v(2)),
                    InstrNode::new(Inst::Return { values: vec![v(2)] }),
                ],
            },
            TmirBlock {
                id: b(2),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(0),
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
fn test_trunc_i64_to_i1_adapter() {
    let module = build_trunc_i64_to_i1();
    let (lir_func, _) = translate_only(&module).unwrap();

    assert_eq!(lir_func.blocks.len(), 3);
    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert!(entry.instructions.iter().any(|i| matches!(
        &i.opcode,
        Opcode::Trunc { to_ty: Type::B1 }
    )));
}

#[test]
fn test_trunc_i64_to_i1_isel() {
    let module = build_trunc_i64_to_i1();
    let mfunc = compile_tmir_function(&module);

    assert!(has_opcode(&mfunc, AArch64Opcode::BCond));
    assert!(
        count_opcode(&mfunc, AArch64Opcode::Ret) >= 2,
        "Expected 2 RET instructions (then + else)"
    );
}

// ---------------------------------------------------------------------------
// Category 5: Large i64 constants
// tla2 needs large constants for set cardinalities, model checking bounds, etc.
// ---------------------------------------------------------------------------

fn build_large_i64_const() -> TmirModule {
    single_function_module(
        140,
        "large_const",
        func_ty(vec![], vec![Ty::I64]),
        vec![TmirBlock {
            id: b(0),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0x7FFF_FFFF_FFFF_FFFF), // i64::MAX
                })
                .with_result(v(0)),
                InstrNode::new(Inst::Return { values: vec![v(0)] }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_large_i64_const_adapter() {
    let module = build_large_i64_const();
    let (lir_func, _) = translate_only(&module).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 2);
    match &entry.instructions[0].opcode {
        Opcode::Iconst { ty, imm } => {
            assert_eq!(*ty, Type::I64);
            assert_eq!(*imm, 0x7FFF_FFFF_FFFF_FFFF_i64);
        }
        other => panic!("Expected Iconst, got {:?}", other),
    }
}

#[test]
fn test_large_i64_const_isel() {
    let module = build_large_i64_const();
    let mfunc = compile_tmir_function(&module);

    // Large constants may need multiple MOV instructions (MOVZ + MOVK)
    assert!(
        has_opcode(&mfunc, AArch64Opcode::Movz) || has_opcode(&mfunc, AArch64Opcode::MovI),
        "Expected MOVZ or MOV immediate for large constant materialization"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

fn build_negative_i64_const() -> TmirModule {
    single_function_module(
        141,
        "neg_const",
        func_ty(vec![], vec![Ty::I64]),
        vec![TmirBlock {
            id: b(0),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(-1),
                })
                .with_result(v(0)),
                InstrNode::new(Inst::Return { values: vec![v(0)] }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_negative_i64_const_adapter() {
    let module = build_negative_i64_const();
    let (lir_func, _) = translate_only(&module).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];
    match &entry.instructions[0].opcode {
        Opcode::Iconst { ty, imm } => {
            assert_eq!(*ty, Type::I64);
            assert_eq!(*imm, -1_i64);
        }
        other => panic!("Expected Iconst, got {:?}", other),
    }
}

#[test]
fn test_negative_i64_const_isel() {
    let module = build_negative_i64_const();
    let mfunc = compile_tmir_function(&module);

    // -1 can be materialized via MOVN
    assert!(
        has_opcode(&mfunc, AArch64Opcode::Movn)
            || has_opcode(&mfunc, AArch64Opcode::MovI)
            || has_opcode(&mfunc, AArch64Opcode::Movz),
        "Expected MOV variant for negative constant"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ---------------------------------------------------------------------------
// Category 6: Memory operations on i64 arrays (tla2's primary data structure)
// TLA+ functions/sequences are typically i64 arrays accessed via GEP
// ---------------------------------------------------------------------------

/// Load an i64 element from an array: arr[idx]
fn build_i64_array_load() -> TmirModule {
    single_function_module(
        150,
        "i64_array_load",
        func_ty(vec![Ty::Ptr, Ty::I64], vec![Ty::I64]),
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
                InstrNode::new(Inst::Load {
                    ty: Ty::I64,
                    ptr: v(2),
                    volatile: false,
                    align: None,
                })
                .with_result(v(3)),
                InstrNode::new(Inst::Return { values: vec![v(3)] }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_i64_array_load_adapter() {
    let module = build_i64_array_load();
    let (lir_func, _) = translate_only(&module).unwrap();

    assert_eq!(lir_func.signature.params, vec![Type::I64, Type::I64]);
    assert_eq!(lir_func.signature.returns, vec![Type::I64]);

    let entry = &lir_func.blocks[&lir_func.entry_block];
    // GEP expands to: Iconst(8) + Imul(idx, 8) + Iadd(base, offset)
    // Then: Load + Return
    let has_load = entry
        .instructions
        .iter()
        .any(|i| matches!(&i.opcode, Opcode::Load { ty: Type::I64 }));
    assert!(has_load, "Expected Load of type I64 after GEP");
}

#[test]
fn test_i64_array_load_isel() {
    let module = build_i64_array_load();
    let mfunc = compile_tmir_function(&module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::LdrRI) || has_opcode(&mfunc, AArch64Opcode::LdrRO),
        "Expected LDR for i64 array load"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

/// Store an i64 element into an array: arr[idx] = val
fn build_i64_array_store() -> TmirModule {
    single_function_module(
        151,
        "i64_array_store",
        func_ty(vec![Ty::Ptr, Ty::I64, Ty::I64], vec![]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::Ptr), (v(1), Ty::I64), (v(2), Ty::I64)],
            body: vec![
                InstrNode::new(Inst::GEP {
                    pointee_ty: Ty::I64,
                    base: v(0),
                    indices: vec![v(1)],
                })
                .with_result(v(3)),
                InstrNode::new(Inst::Store {
                    ty: Ty::I64,
                    ptr: v(3),
                    value: v(2),
                    volatile: false,
                    align: None,
                }),
                InstrNode::new(Inst::Return { values: vec![] }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_i64_array_store_adapter() {
    let module = build_i64_array_store();
    let (lir_func, _) = translate_only(&module).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];
    let has_store = entry
        .instructions
        .iter()
        .any(|i| matches!(&i.opcode, Opcode::Store));
    assert!(has_store, "Expected Store opcode for array write");
}

#[test]
fn test_i64_array_store_isel() {
    let module = build_i64_array_store();
    let mfunc = compile_tmir_function(&module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::StrRI) || has_opcode(&mfunc, AArch64Opcode::StrRO),
        "Expected STR for i64 array store"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ---------------------------------------------------------------------------
// Category 7: Function calls to extern "C" runtime helpers
// tla2 calls into runtime helpers for set operations, string handling, etc.
// ---------------------------------------------------------------------------

fn build_extern_call() -> TmirModule {
    let mut module = TmirModule::new("extern_call_test");
    let helper_sig = module.add_func_type(func_ty(vec![Ty::Ptr, Ty::I64], vec![Ty::I64]));
    let main_sig = module.add_func_type(func_ty(vec![Ty::Ptr, Ty::I64], vec![Ty::I64]));

    // External function (minimal stub -- just returns its second argument)
    let mut helper = TmirFunction::new(f(0), "tla2_runtime_set_card", helper_sig, b(0));
    helper.blocks.push(TmirBlock {
        id: b(0),
        params: vec![(v(0), Ty::Ptr), (v(1), Ty::I64)],
        body: vec![InstrNode::new(Inst::Return { values: vec![v(1)] })],
    });

    // Main function that calls the helper
    let mut main_func = TmirFunction::new(f(1), "count_elements", main_sig, b(0));
    main_func.blocks.push(TmirBlock {
        id: b(0),
        params: vec![(v(0), Ty::Ptr), (v(1), Ty::I64)],
        body: vec![
            InstrNode::new(Inst::Call {
                callee: f(0),
                args: vec![v(0), v(1)],
            })
            .with_result(v(2)),
            InstrNode::new(Inst::Return { values: vec![v(2)] }),
        ],
    });

    module.add_function(helper);
    module.add_function(main_func);
    module
}

#[test]
fn test_extern_call_adapter() {
    let module = build_extern_call();
    let results = translate_module(&module).unwrap();

    // Should have 2 functions (helper declaration + caller)
    assert!(results.len() >= 1, "Should translate at least the caller");

    // Find the caller function
    let caller = results.iter().find(|(f, _)| f.name == "count_elements");
    assert!(caller.is_some(), "Should have translated count_elements");

    let (caller_func, _) = caller.unwrap();
    let entry = &caller_func.blocks[&caller_func.entry_block];
    let has_call = entry.instructions.iter().any(|i| matches!(
        &i.opcode,
        Opcode::Call { name } if name == "tla2_runtime_set_card"
    ));
    assert!(has_call, "Expected Call to tla2_runtime_set_card");
}

// ---------------------------------------------------------------------------
// Category 8: Select (conditional value without branching)
// Already tested above (test_select), but add i64 variant for tla2
// ---------------------------------------------------------------------------

fn build_i64_select() -> TmirModule {
    single_function_module(
        160,
        "i64_select",
        func_ty(vec![Ty::I64, Ty::I64, Ty::I64], vec![Ty::I64]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I64), (v(1), Ty::I64), (v(2), Ty::I64)],
            body: vec![
                // Compare first two args
                InstrNode::new(Inst::ICmp {
                    op: ICmpOp::Sgt,
                    ty: Ty::I64,
                    lhs: v(0),
                    rhs: v(1),
                })
                .with_result(v(3)),
                // Select between them based on comparison
                InstrNode::new(Inst::Select {
                    ty: Ty::I64,
                    cond: v(3),
                    then_val: v(0),
                    else_val: v(1),
                })
                .with_result(v(4)),
                InstrNode::new(Inst::Return { values: vec![v(4)] }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_i64_select_adapter() {
    let module = build_i64_select();
    let (lir_func, _) = translate_only(&module).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];
    let has_select = entry
        .instructions
        .iter()
        .any(|i| matches!(&i.opcode, Opcode::Select { .. }));
    assert!(has_select, "Expected Select opcode for i64 conditional select");
}

#[test]
fn test_i64_select_isel() {
    let module = build_i64_select();
    let mfunc = compile_tmir_function(&module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::CmpRR),
        "Expected CMP for the comparison"
    );
    assert!(
        has_opcode(&mfunc, AArch64Opcode::Csel),
        "Expected CSEL for i64 conditional select"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ---------------------------------------------------------------------------
// Category 9: Alloca for local stack variables
// tla2 uses stack allocation for local variables in action functions
// ---------------------------------------------------------------------------

fn build_alloca_usage() -> TmirModule {
    single_function_module(
        170,
        "alloca_local",
        func_ty(vec![Ty::I64], vec![Ty::I64]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I64)],
            body: vec![
                // Alloca for an i64 local
                InstrNode::new(Inst::Alloca {
                    ty: Ty::I64,
                    count: None,
                    align: None,
                })
                .with_result(v(1)),
                // Store into it
                InstrNode::new(Inst::Store {
                    ty: Ty::I64,
                    ptr: v(1),
                    value: v(0),
                    volatile: false,
                    align: None,
                }),
                // Load back
                InstrNode::new(Inst::Load {
                    ty: Ty::I64,
                    ptr: v(1),
                    volatile: false,
                    align: None,
                })
                .with_result(v(2)),
                InstrNode::new(Inst::Return { values: vec![v(2)] }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_alloca_adapter() {
    let module = build_alloca_usage();
    let (lir_func, _) = translate_only(&module).unwrap();

    // Alloca should create a stack slot
    assert!(
        !lir_func.stack_slots.is_empty(),
        "Expected at least one stack slot from Alloca"
    );

    let entry = &lir_func.blocks[&lir_func.entry_block];
    // Should have: StackAddr + Store + Load + Return
    let has_stack_addr = entry
        .instructions
        .iter()
        .any(|i| matches!(&i.opcode, Opcode::StackAddr { .. }));
    assert!(has_stack_addr, "Expected StackAddr for alloca result");
}

#[test]
fn test_alloca_isel() {
    let module = build_alloca_usage();
    let mfunc = compile_tmir_function(&module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::LdrRI) || has_opcode(&mfunc, AArch64Opcode::LdrRO),
        "Expected LDR for loading from alloca"
    );
    assert!(
        has_opcode(&mfunc, AArch64Opcode::StrRI) || has_opcode(&mfunc, AArch64Opcode::StrRO),
        "Expected STR for storing to alloca"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ---------------------------------------------------------------------------
// Comprehensive: all tla-tmir test programs compile through full pipeline
// ---------------------------------------------------------------------------

#[test]
fn test_all_tla_tmir_programs_adapter_succeeds() {
    let programs: Vec<(&str, TmirModule)> = vec![
        ("i64_mul", build_i64_mul()),
        ("i64_sdiv", build_i64_sdiv()),
        ("i64_srem", build_i64_srem()),
        ("bool_and", build_bool_and()),
        ("bool_or", build_bool_or()),
        ("bool_xor", build_bool_xor()),
        ("bool_not", build_bool_not()),
        ("zext_bool_to_i64", build_zext_i1_to_i64()),
        ("trunc_i64_to_bool", build_trunc_i64_to_i1()),
        ("large_const", build_large_i64_const()),
        ("neg_const", build_negative_i64_const()),
        ("i64_array_load", build_i64_array_load()),
        ("i64_array_store", build_i64_array_store()),
        ("i64_select", build_i64_select()),
        ("alloca_local", build_alloca_usage()),
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

#[test]
fn test_all_tla_tmir_programs_isel_succeeds() {
    let programs: Vec<(&str, TmirModule)> = vec![
        ("i64_mul", build_i64_mul()),
        ("i64_sdiv", build_i64_sdiv()),
        ("i64_srem", build_i64_srem()),
        ("bool_and", build_bool_and()),
        ("bool_or", build_bool_or()),
        ("bool_xor", build_bool_xor()),
        ("bool_not", build_bool_not()),
        ("zext_bool_to_i64", build_zext_i1_to_i64()),
        ("trunc_i64_to_bool", build_trunc_i64_to_i1()),
        ("large_const", build_large_i64_const()),
        ("neg_const", build_negative_i64_const()),
        ("i64_array_load", build_i64_array_load()),
        ("i64_array_store", build_i64_array_store()),
        ("i64_select", build_i64_select()),
        ("alloca_local", build_alloca_usage()),
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

// ===========================================================================
// Test: Multi-value diamond merge (tla2 DieHard-like pattern)
//
// Exercises the pattern where MULTIPLE output values flow through a diamond
// CFG and merge via MULTIPLE block parameters. This is the pattern tla2's
// DieHard actions produce when an IF-THEN-ELSE updates several state
// variables at once.
//
// tMIR form:
//   entry(x: i32, y: i32, cond: bool):
//     condbr cond -> then_block(), else_block()
//   then_block():
//     a = iadd x, y
//     b = isub x, y
//     br merge(a, b)
//   else_block():
//     c = isub y, x
//     d = iadd y, x
//     br merge(c, d)
//   merge(p1: i32, p2: i32):
//     result = iadd p1, p2
//     return result
// ===========================================================================

fn build_multi_value_diamond() -> TmirModule {
    single_function_module(
        60,
        "multi_val_diamond",
        func_ty(vec![Ty::I32, Ty::I32, Ty::Bool], vec![Ty::I32]),
        vec![
            // entry block: compare and branch
            TmirBlock {
                id: b(0),
                params: vec![
                    (v(0), Ty::I32),  // x
                    (v(1), Ty::I32),  // y
                    (v(2), Ty::Bool), // cond
                ],
                body: vec![
                    InstrNode::new(Inst::CondBr {
                        cond: v(2),
                        then_target: b(1),
                        then_args: vec![],
                        else_target: b(2),
                        else_args: vec![],
                    }),
                ],
            },
            // then block: compute two values, branch to merge
            TmirBlock {
                id: b(1),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I32,
                        lhs: v(0),
                        rhs: v(1),
                    }).with_result(v(10)),
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Sub,
                        ty: Ty::I32,
                        lhs: v(0),
                        rhs: v(1),
                    }).with_result(v(11)),
                    InstrNode::new(Inst::Br {
                        target: b(3),
                        args: vec![v(10), v(11)],
                    }),
                ],
            },
            // else block: compute two different values, branch to merge
            TmirBlock {
                id: b(2),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Sub,
                        ty: Ty::I32,
                        lhs: v(1),
                        rhs: v(0),
                    }).with_result(v(20)),
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I32,
                        lhs: v(1),
                        rhs: v(0),
                    }).with_result(v(21)),
                    InstrNode::new(Inst::Br {
                        target: b(3),
                        args: vec![v(20), v(21)],
                    }),
                ],
            },
            // merge block: use both merged values
            TmirBlock {
                id: b(3),
                params: vec![(v(30), Ty::I32), (v(31), Ty::I32)],
                body: vec![
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I32,
                        lhs: v(30),
                        rhs: v(31),
                    }).with_result(v(32)),
                    InstrNode::new(Inst::Return { values: vec![v(32)] }),
                ],
            },
        ],
        vec![],
    )
}

#[test]
fn test_multi_value_diamond_adapter() {
    let module = build_multi_value_diamond();
    let (lir_func, _) = translate_only(&module).unwrap();

    // 4 blocks: entry, then, else, merge (Br copies are inline)
    assert_eq!(lir_func.blocks.len(), 4,
        "expected 4 blocks, got {}", lir_func.blocks.len());

    // Verify merge block has exactly 2 parameters
    let merge_block = lir_func
        .blocks
        .iter()
        .filter(|(id, _)| **id != lir_func.entry_block)
        .map(|(_, bb)| bb)
        .find(|bb| bb.params.len() == 2)
        .expect("merge block should have 2 block parameters");
    assert_eq!(merge_block.params.len(), 2);

    // Each branch block: 2 ops + 2 copies + 1 jump = 5 instructions
    let branch_blocks: Vec<_> = lir_func
        .blocks
        .iter()
        .filter(|(id, bb)| **id != lir_func.entry_block && bb.params.is_empty())
        .map(|(_, bb)| bb)
        .collect();
    assert_eq!(branch_blocks.len(), 2, "should have 2 branch blocks");

    for bb in &branch_blocks {
        assert_eq!(
            bb.instructions.len(), 5,
            "branch block should have 2 ops + 2 copies + jump, got {:?}",
            bb.instructions.iter().map(|i| &i.opcode).collect::<Vec<_>>()
        );
        // First two are BinOps (Iadd or Isub)
        // Next two are copies (single-arg Iadd)
        assert_eq!(bb.instructions[2].args.len(), 1, "copy should have 1 arg");
        assert_eq!(bb.instructions[3].args.len(), 1, "copy should have 1 arg");
        // Last is Jump
        assert!(matches!(bb.instructions[4].opcode, Opcode::Jump { .. }));
    }

    // Both branch blocks' copies should target the same merge block params
    let then_dst0 = branch_blocks[0].instructions[2].results[0];
    let then_dst1 = branch_blocks[0].instructions[3].results[0];
    let else_dst0 = branch_blocks[1].instructions[2].results[0];
    let else_dst1 = branch_blocks[1].instructions[3].results[0];
    assert_eq!(then_dst0, else_dst0, "first param copies should target same Value");
    assert_eq!(then_dst1, else_dst1, "second param copies should target same Value");
    assert_ne!(then_dst0, then_dst1, "the two params should be different Values");
}

#[test]
fn test_multi_value_diamond_isel() {
    let module = build_multi_value_diamond();
    let mfunc = compile_tmir_function(&module);

    // Should have 4 blocks
    assert_eq!(mfunc.blocks.len(), 4, "multi-value diamond should produce 4 ISel blocks");

    // Should have conditional branch, arithmetic, and return
    assert!(has_opcode(&mfunc, AArch64Opcode::BCond), "Expected B.cond");
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret), "Expected RET");
    assert!(
        count_opcode(&mfunc, AArch64Opcode::AddRR) >= 2,
        "Expected at least 2 ADD instructions"
    );
    assert!(
        count_opcode(&mfunc, AArch64Opcode::SubRR) >= 2,
        "Expected at least 2 SUB instructions"
    );
    // Copy instructions for block params
    assert!(
        count_opcode(&mfunc, AArch64Opcode::MovR) >= 4,
        "Expected at least 4 MOV instructions for 2 params x 2 branches"
    );
}

// ===========================================================================
// Test: CondBr with multi-value args to same merge block
//
// Like the #302 test but with MULTIPLE block parameters, exercising the
// pattern where CondBr directly passes multiple different values to the
// same merge block from each edge.
//
// tMIR form:
//   entry(cond: bool, a: i32, b: i32, c: i32, d: i32):
//     condbr cond -> merge(a, b), merge(c, d)
//   merge(p1: i32, p2: i32):
//     result = iadd p1, p2
//     return result
// ===========================================================================

fn build_condbr_multi_value_merge() -> TmirModule {
    let mut module = TmirModule::new("condbr_multi_val");
    let ft_id = module.add_func_type(func_ty(
        vec![Ty::Bool, Ty::I32, Ty::I32, Ty::I32, Ty::I32],
        vec![Ty::I32],
    ));

    let mut func = TmirFunction::new(f(61), "select_pair", ft_id, b(0));
    func.blocks = vec![
        // entry: condbr with multi-value args to same merge block
        TmirBlock {
            id: b(0),
            params: vec![
                (v(0), Ty::Bool),  // cond
                (v(1), Ty::I32),   // a
                (v(2), Ty::I32),   // b
                (v(3), Ty::I32),   // c
                (v(4), Ty::I32),   // d
            ],
            body: vec![InstrNode::new(Inst::CondBr {
                cond: v(0),
                then_target: b(1),
                then_args: vec![v(1), v(2)],   // pass (a, b) to merge
                else_target: b(1),
                else_args: vec![v(3), v(4)],   // pass (c, d) to merge
            })],
        },
        // merge block: receives two params, combines them
        TmirBlock {
            id: b(1),
            params: vec![(v(10), Ty::I32), (v(11), Ty::I32)],
            body: vec![
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I32,
                    lhs: v(10),
                    rhs: v(11),
                }).with_result(v(12)),
                InstrNode::new(Inst::Return { values: vec![v(12)] }),
            ],
        },
    ];

    module.add_function(func);
    module
}

#[test]
fn test_condbr_multi_value_merge_adapter() {
    let module = build_condbr_multi_value_merge();
    let (lir_func, _) = translate_only(&module).unwrap();

    // 4 blocks: entry + merge + 2 copy blocks
    assert_eq!(lir_func.blocks.len(), 4,
        "expected 4 blocks (entry + merge + 2 copy blocks), got {}",
        lir_func.blocks.len()
    );

    let entry = &lir_func.blocks[&lir_func.entry_block];
    // Entry should have only the Brif
    assert_eq!(entry.instructions.len(), 1);
    match &entry.instructions[0].opcode {
        Opcode::Brif { then_dest, else_dest, .. } => {
            assert_ne!(then_dest, else_dest, "copy blocks should be different");

            let then_block = &lir_func.blocks[then_dest];
            // 2 copies + 1 jump = 3 instructions
            assert_eq!(then_block.instructions.len(), 3,
                "then-copy block should have 2 copies + jump");

            let else_block = &lir_func.blocks[else_dest];
            assert_eq!(else_block.instructions.len(), 3,
                "else-copy block should have 2 copies + jump");

            // Both copy blocks jump to the same merge block
            let then_jump = match &then_block.instructions[2].opcode {
                Opcode::Jump { dest } => *dest,
                _ => panic!("expected Jump"),
            };
            let else_jump = match &else_block.instructions[2].opcode {
                Opcode::Jump { dest } => *dest,
                _ => panic!("expected Jump"),
            };
            assert_eq!(then_jump, else_jump, "both should jump to merge");

            // Copy destinations should match: then[0].dst == else[0].dst (param p1)
            // and then[1].dst == else[1].dst (param p2)
            assert_eq!(
                then_block.instructions[0].results[0],
                else_block.instructions[0].results[0],
                "first param copies should target same Value (p1)"
            );
            assert_eq!(
                then_block.instructions[1].results[0],
                else_block.instructions[1].results[0],
                "second param copies should target same Value (p2)"
            );

            // Copy sources should differ (a,b vs c,d)
            assert_ne!(
                then_block.instructions[0].args[0],
                else_block.instructions[0].args[0],
                "first copies should use different sources"
            );
            assert_ne!(
                then_block.instructions[1].args[0],
                else_block.instructions[1].args[0],
                "second copies should use different sources"
            );
        }
        other => panic!("expected Brif, got {:?}", other),
    }
}

#[test]
fn test_condbr_multi_value_merge_isel() {
    let module = build_condbr_multi_value_merge();
    let mfunc = compile_tmir_function(&module);

    // Should have 4 blocks
    assert_eq!(mfunc.blocks.len(), 4,
        "condbr multi-value merge should produce 4 ISel blocks, got {}",
        mfunc.blocks.len()
    );

    assert!(has_opcode(&mfunc, AArch64Opcode::BCond), "Expected B.cond");
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret), "Expected RET");
    // 4 MOV instructions total: 2 per copy block
    assert!(
        count_opcode(&mfunc, AArch64Opcode::MovR) >= 4,
        "Expected at least 4 MOV for 2 params x 2 edges, got {}",
        count_opcode(&mfunc, AArch64Opcode::MovR)
    );
}

// ===========================================================================
// Test: tla2 DieHard-style i64 IF-THEN-ELSE with multi-variable state writes
//
// This exercises the EXACT pattern tla2's BigToSmall / SmallToBig actions
// produce: an i64 IF-THEN-ELSE where BOTH branches write MULTIPLE state
// variables, and the merge block consumes all of them (via block parameters).
//
// Pattern:
//   pour_big_to_small(big: i64, small: i64, big_cap: i64, small_cap: i64)
//                                               -> (new_big, new_small)
//
//   IF big >= (small_cap - small) THEN
//     -- pouring saturates small jug
//     new_big   = big - (small_cap - small)
//     new_small = small_cap
//   ELSE
//     -- all of big fits in small
//     new_big   = 0
//     new_small = small + big
//
// We return new_big + new_small as a proxy for successful multi-value merge.
//
// tMIR form (i64 everywhere, mirrors tla2 usage):
//   entry(big, small, big_cap, small_cap: i64):
//     room     = isub small_cap, small
//     cond     = icmp sge big, room
//     condbr cond -> saturate(...), fits(...)
//   saturate():
//     new_big   = isub big, room
//     new_small = small_cap
//     br merge(new_big, new_small)
//   fits():
//     new_big   = 0
//     new_small = iadd small, big
//     br merge(new_big, new_small)
//   merge(final_big: i64, final_small: i64):
//     result = iadd final_big, final_small
//     return result
// ===========================================================================

fn build_diehard_big_to_small() -> TmirModule {
    single_function_module(
        70,
        "pour_big_to_small",
        func_ty(vec![Ty::I64, Ty::I64, Ty::I64, Ty::I64], vec![Ty::I64]),
        vec![
            // entry: compute room, compare, branch
            TmirBlock {
                id: b(0),
                params: vec![
                    (v(0), Ty::I64), // big
                    (v(1), Ty::I64), // small
                    (v(2), Ty::I64), // big_cap (unused, mirrors tla2 signature)
                    (v(3), Ty::I64), // small_cap
                ],
                body: vec![
                    // room = small_cap - small
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Sub,
                        ty: Ty::I64,
                        lhs: v(3),
                        rhs: v(1),
                    })
                    .with_result(v(4)),
                    // cond = big >= room
                    InstrNode::new(Inst::ICmp {
                        op: ICmpOp::Sge,
                        ty: Ty::I64,
                        lhs: v(0),
                        rhs: v(4),
                    })
                    .with_result(v(5)),
                    // branch
                    InstrNode::new(Inst::CondBr {
                        cond: v(5),
                        then_target: b(1),
                        then_args: vec![],
                        else_target: b(2),
                        else_args: vec![],
                    }),
                ],
            },
            // saturate: new_big = big - room; new_small = small_cap
            TmirBlock {
                id: b(1),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Sub,
                        ty: Ty::I64,
                        lhs: v(0),
                        rhs: v(4),
                    })
                    .with_result(v(10)),
                    // new_small = small_cap -> use a MoveValue proxy via
                    // Const(small_cap) is not possible; pass small_cap value
                    // directly as merge arg since no additional computation.
                    InstrNode::new(Inst::Br {
                        target: b(3),
                        args: vec![v(10), v(3)], // (new_big, small_cap)
                    }),
                ],
            },
            // fits: new_big = 0; new_small = small + big
            TmirBlock {
                id: b(2),
                params: vec![],
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(0),
                    })
                    .with_result(v(20)),
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: v(1),
                        rhs: v(0),
                    })
                    .with_result(v(21)),
                    InstrNode::new(Inst::Br {
                        target: b(3),
                        args: vec![v(20), v(21)], // (0, small+big)
                    }),
                ],
            },
            // merge: sum the two new state values and return
            TmirBlock {
                id: b(3),
                params: vec![(v(30), Ty::I64), (v(31), Ty::I64)],
                body: vec![
                    InstrNode::new(Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: v(30),
                        rhs: v(31),
                    })
                    .with_result(v(32)),
                    InstrNode::new(Inst::Return { values: vec![v(32)] }),
                ],
            },
        ],
        vec![],
    )
}

#[test]
fn test_diehard_big_to_small_adapter() {
    let module = build_diehard_big_to_small();
    let (lir_func, _) = translate_only(&module).unwrap();

    // 4 blocks: entry, saturate, fits, merge
    assert_eq!(
        lir_func.blocks.len(),
        4,
        "DieHard i64 IF-THEN-ELSE should produce 4 LIR blocks, got {}",
        lir_func.blocks.len()
    );

    // Signature should propagate i64 parameters and return
    assert_eq!(
        lir_func.signature.params,
        vec![Type::I64, Type::I64, Type::I64, Type::I64]
    );
    assert_eq!(lir_func.signature.returns, vec![Type::I64]);

    // Each branch arm must have its own copy sequence (no unified copies
    // hoisted out to the entry block). Verify by looking at the non-entry,
    // non-merge blocks' instruction layout.
    let merge_block = lir_func
        .blocks
        .iter()
        .filter(|(id, _)| **id != lir_func.entry_block)
        .map(|(_, bb)| bb)
        .find(|bb| bb.params.len() == 2)
        .expect("merge block should have 2 i64 block parameters");
    assert_eq!(merge_block.params[0].1, Type::I64);
    assert_eq!(merge_block.params[1].1, Type::I64);

    // Post-SSA (block-parameter-form) correctness invariants:
    //
    // 1. Every non-block-parameter Value is defined at most once across the
    //    function. Block parameters may be the destination of COPY
    //    instructions in multiple predecessor blocks — that is the standard
    //    block-parameter SSA deconstruction form and is NOT an SSA violation.
    // 2. A merge-block parameter Value must never be the result of an
    //    instruction INSIDE the merge block itself (block params can only be
    //    written by predecessor copies).
    // 3. Within each single predecessor edge's copy sequence, each block
    //    parameter must be written at most ONCE. Writing the same merge-param
    //    twice from one edge is the bug reported in #365 ("invalid SSA
    //    references when both branches write to the same output variable").

    let merge_param_values: std::collections::HashSet<_> =
        merge_block.params.iter().map(|(v, _)| *v).collect();

    // Invariant 1: non-param defs unique across the function.
    let mut non_param_defs: std::collections::HashMap<llvm2_lower::instructions::Value, usize> =
        std::collections::HashMap::new();
    for (_, bb) in &lir_func.blocks {
        for inst in &bb.instructions {
            for r in &inst.results {
                if !merge_param_values.contains(r) {
                    let n = non_param_defs.entry(*r).or_insert(0);
                    *n += 1;
                    assert!(
                        *n == 1,
                        "SSA violation: non-param Value {:?} defined {} times (opcode {:?})",
                        r,
                        n,
                        inst.opcode
                    );
                }
            }
        }
    }

    // Invariant 2: merge-block body does not write merge-param Values.
    for inst in &merge_block.instructions {
        for r in &inst.results {
            assert!(
                !merge_param_values.contains(r),
                "merge block param {:?} written by instruction in merge block (opcode {:?})",
                r,
                inst.opcode
            );
        }
    }

    // Invariant 3: each predecessor edge writes each merge-param at most once.
    for (bid, bb) in &lir_func.blocks {
        let mut per_edge_writes: std::collections::HashMap<
            llvm2_lower::instructions::Value,
            usize,
        > = std::collections::HashMap::new();
        for inst in &bb.instructions {
            for r in &inst.results {
                if merge_param_values.contains(r) {
                    *per_edge_writes.entry(*r).or_insert(0) += 1;
                }
            }
        }
        for (v, n) in per_edge_writes {
            assert!(
                n == 1,
                "merge param {:?} written {} times on edge from block {:?} (expected <=1)",
                v,
                n,
                bid
            );
        }
    }
}

#[test]
fn test_diehard_big_to_small_isel() {
    let module = build_diehard_big_to_small();
    let mfunc = compile_tmir_function(&module);

    assert_eq!(
        mfunc.blocks.len(),
        4,
        "DieHard ISel should produce 4 machine blocks, got {}",
        mfunc.blocks.len()
    );

    assert!(has_opcode(&mfunc, AArch64Opcode::BCond), "Expected B.cond");
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret), "Expected RET");
    // At least one SUB (room, new_big in saturate) and one ADD (small+big in fits)
    assert!(
        count_opcode(&mfunc, AArch64Opcode::SubRR) >= 2,
        "Expected at least 2 SUB instructions"
    );
    assert!(
        count_opcode(&mfunc, AArch64Opcode::AddRR) >= 2,
        "Expected at least 2 ADD instructions (final merge + fits)"
    );
    // Copies for the 2 params x 2 edges
    assert!(
        count_opcode(&mfunc, AArch64Opcode::MovR) >= 4,
        "Expected at least 4 MOV for 2 params x 2 edges, got {}",
        count_opcode(&mfunc, AArch64Opcode::MovR)
    );
}

// ===========================================================================
// Test: Parallel-copy swap detection via back-edge argument permutation
//
// This is the classic parallel-copy correctness test. A loop back-edge
// passes its own header's parameters in SWAPPED order:
//
//   header(a, b):
//     ... use a, b ...
//     cond = ...
//     condbr cond -> header(b, a) {back}, exit(a, b)
//
// Under SEQUENTIAL copies (a = b; b = a), the second copy reads `a` AFTER
// it has been overwritten to the old `b`, yielding b = b instead of b = a.
//
// The correct lowering is a parallel copy: detect the cycle and insert a
// temp, e.g.:
//     tmp = a; a = b; b = tmp;
//
// This test asserts the structural property that the emitted copy sequence
// for a swap edge does NOT expose the sequential-overwrite miscompile. We
// check that either:
//   (a) a fresh temporary Value is introduced (cycle-break insertion), or
//   (b) the copy order is such that no destination of an earlier copy is
//       read as source by a later copy in the SAME edge.
// ===========================================================================

fn build_back_edge_swap() -> TmirModule {
    single_function_module(
        71,
        "swap_back_edge",
        func_ty(vec![Ty::I64, Ty::I64], vec![Ty::I64]),
        vec![
            // entry: br header(x, y)
            TmirBlock {
                id: b(0),
                params: vec![(v(0), Ty::I64), (v(1), Ty::I64)],
                body: vec![InstrNode::new(Inst::Br {
                    target: b(1),
                    args: vec![v(0), v(1)],
                })],
            },
            // header(a, b): compute cond, condbr header(b, a) | exit
            TmirBlock {
                id: b(1),
                params: vec![(v(2), Ty::I64), (v(3), Ty::I64)],
                body: vec![
                    // cond = a > 0  (to ensure loop eventually terminates
                    // in an abstract sense — we only care about structure)
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(0),
                    })
                    .with_result(v(4)),
                    InstrNode::new(Inst::ICmp {
                        op: ICmpOp::Sgt,
                        ty: Ty::I64,
                        lhs: v(2),
                        rhs: v(4),
                    })
                    .with_result(v(5)),
                    InstrNode::new(Inst::CondBr {
                        cond: v(5),
                        then_target: b(1),       // back-edge
                        then_args: vec![v(3), v(2)], // SWAP: pass (b, a)
                        else_target: b(2),
                        else_args: vec![v(2)],
                    }),
                ],
            },
            // exit(result): return result
            TmirBlock {
                id: b(2),
                params: vec![(v(6), Ty::I64)],
                body: vec![InstrNode::new(Inst::Return { values: vec![v(6)] })],
            },
        ],
        vec![],
    )
}

#[test]
fn test_back_edge_swap_parallel_copy_correctness() {
    let module = build_back_edge_swap();
    let (lir_func, _) = translate_only(&module).unwrap();

    // Find the back-edge copy block (jumps back to header, which is the only
    // non-entry block with 2 params).
    let header_block_id_and_params: Vec<_> = lir_func
        .blocks
        .iter()
        .filter(|(id, bb)| **id != lir_func.entry_block && bb.params.len() == 2)
        .map(|(id, bb)| (*id, bb.params.iter().map(|(v, _)| *v).collect::<Vec<_>>()))
        .collect();
    assert_eq!(header_block_id_and_params.len(), 1, "expected exactly one loop header");
    let (header_id, header_params) = &header_block_id_and_params[0];
    let header_param_a = header_params[0];
    let header_param_b = header_params[1];

    // The back-edge copy block is the one that:
    //   - has no block params
    //   - ends in Jump { dest: header_id }
    //   - contains copies whose DESTINATIONS are the header's parameters
    let back_edge_copy_block = lir_func
        .blocks
        .values()
        .find(|bb| {
            bb.params.is_empty()
                && bb.instructions.iter().any(|i| {
                    matches!(i.opcode, Opcode::Jump { dest } if dest == *header_id)
                })
                && bb.instructions.iter().any(|i| {
                    !i.results.is_empty()
                        && (i.results[0] == header_param_a || i.results[0] == header_param_b)
                })
        })
        .expect("should find a back-edge copy block that jumps to header");

    // Collect the in-order (source, dest) copy pairs before the terminator.
    let copies: Vec<(
        llvm2_lower::instructions::Value,
        llvm2_lower::instructions::Value,
    )> = back_edge_copy_block
        .instructions
        .iter()
        .filter(|i| !matches!(i.opcode, Opcode::Jump { .. }))
        .map(|i| (i.args[0], i.results[0]))
        .collect();

    assert!(
        copies.len() >= 2,
        "expected at least 2 copies on the swap back-edge, got {}",
        copies.len()
    );

    // Parallel-copy correctness invariant (semantic):
    //
    // A correct lowering of `br header(b, a)` from `header(a, b)` must
    // ensure that, AFTER the full copy sequence, the header's first
    // parameter holds the ORIGINAL value of `b` and the second parameter
    // holds the ORIGINAL value of `a`. Under the previous naive sequential
    // emission, the swap was silently miscompiled (both params ended up
    // holding the same value).
    //
    // We simulate the copy sequence with a symbolic register file:
    //   - Each original Value starts holding its own symbolic value.
    //   - Each copy (src, dst) sets the file's "contents at dst" to the
    //     CURRENT contents at src.
    //   - After all copies, read the merge-param Values and check they
    //     contain the correct swapped originals.
    use std::collections::HashMap;
    let mut reg_file: HashMap<llvm2_lower::instructions::Value, llvm2_lower::instructions::Value> =
        HashMap::new();
    for (src, dst) in &copies {
        // Ensure src and dst have an entry (initialised to their own Value
        // if not yet seen — that represents the "original" symbolic value).
        reg_file
            .entry(*src)
            .or_insert(*src);
        reg_file
            .entry(*dst)
            .or_insert(*dst);
        let val = *reg_file.get(src).unwrap();
        reg_file.insert(*dst, val);
    }

    // Expected post-state: header_param_a holds the ORIGINAL source that the
    // tMIR back-edge passed as the first arg, which is the entry block's
    // second param (header_param_b is the ORIGINAL second param mapped from
    // v(3), i.e. Value(3)). But map_value may produce arbitrary Values, so
    // we deduce the expectation from the tMIR: the first arg of the back
    // edge's CondBr.then_args is ValueId(3), which is header's second
    // parameter. So after the copies, header_param_a (the first merge param)
    // must hold the value that was originally at header_param_b.
    assert_eq!(
        reg_file.get(&header_param_a).copied(),
        Some(header_param_b),
        "swap miscompile: header's first param should hold the original second param's value, got {:?}",
        reg_file.get(&header_param_a)
    );
    assert_eq!(
        reg_file.get(&header_param_b).copied(),
        Some(header_param_a),
        "swap miscompile: header's second param should hold the original first param's value, got {:?}",
        reg_file.get(&header_param_b)
    );
}

// ===========================================================================
// Test: 3-way rotation parallel copy (a<-b, b<-c, c<-a)
//
// This is the generalised version of the back-edge swap test. A loop
// back-edge passes (b, c, a) to header(a, b, c), forcing a 3-cycle in the
// parallel-copy graph. The scheduler must insert exactly one temp to break
// the cycle, then emit the remaining copies in a safe order.
//
// tMIR form:
//   entry(x, y, z: i64):
//     br header(x, y, z)
//   header(a, b, c: i64):
//     cond = a > 0
//     condbr cond -> header(b, c, a), exit(a)
//   exit(result):
//     return result
// ===========================================================================

fn build_three_way_rotation() -> TmirModule {
    single_function_module(
        72,
        "rotate_back_edge",
        func_ty(vec![Ty::I64, Ty::I64, Ty::I64], vec![Ty::I64]),
        vec![
            // entry: br header(x, y, z)
            TmirBlock {
                id: b(0),
                params: vec![(v(0), Ty::I64), (v(1), Ty::I64), (v(2), Ty::I64)],
                body: vec![InstrNode::new(Inst::Br {
                    target: b(1),
                    args: vec![v(0), v(1), v(2)],
                })],
            },
            // header(a, b, c): condbr back with rotation or exit
            TmirBlock {
                id: b(1),
                params: vec![
                    (v(3), Ty::I64),
                    (v(4), Ty::I64),
                    (v(5), Ty::I64),
                ],
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(0),
                    })
                    .with_result(v(6)),
                    InstrNode::new(Inst::ICmp {
                        op: ICmpOp::Sgt,
                        ty: Ty::I64,
                        lhs: v(3),
                        rhs: v(6),
                    })
                    .with_result(v(7)),
                    InstrNode::new(Inst::CondBr {
                        cond: v(7),
                        then_target: b(1),
                        then_args: vec![v(4), v(5), v(3)], // ROTATE: (b, c, a)
                        else_target: b(2),
                        else_args: vec![v(3)],
                    }),
                ],
            },
            // exit(result)
            TmirBlock {
                id: b(2),
                params: vec![(v(8), Ty::I64)],
                body: vec![InstrNode::new(Inst::Return { values: vec![v(8)] })],
            },
        ],
        vec![],
    )
}

#[test]
fn test_three_way_rotation_parallel_copy_correctness() {
    let module = build_three_way_rotation();
    let (lir_func, _) = translate_only(&module).unwrap();

    // Find the header block (non-entry, 3 params).
    let header_info: Vec<_> = lir_func
        .blocks
        .iter()
        .filter(|(id, bb)| **id != lir_func.entry_block && bb.params.len() == 3)
        .map(|(id, bb)| (*id, bb.params.iter().map(|(v, _)| *v).collect::<Vec<_>>()))
        .collect();
    assert_eq!(header_info.len(), 1);
    let (header_id, header_params) = &header_info[0];
    let pa = header_params[0];
    let pb = header_params[1];
    let pc = header_params[2];

    // Find the back-edge copy block: jumps to header AND writes at least
    // one of the header params.
    let back_edge_copy_block = lir_func
        .blocks
        .values()
        .find(|bb| {
            bb.params.is_empty()
                && bb.instructions.iter().any(|i| {
                    matches!(i.opcode, Opcode::Jump { dest } if dest == *header_id)
                })
                && bb.instructions.iter().any(|i| {
                    !i.results.is_empty()
                        && (i.results[0] == pa || i.results[0] == pb || i.results[0] == pc)
                })
        })
        .expect("should find back-edge copy block");

    let copies: Vec<(
        llvm2_lower::instructions::Value,
        llvm2_lower::instructions::Value,
    )> = back_edge_copy_block
        .instructions
        .iter()
        .filter(|i| !matches!(i.opcode, Opcode::Jump { .. }))
        .map(|i| (i.args[0], i.results[0]))
        .collect();

    // Symbolically execute the copy sequence and verify the post-state
    // matches the intended rotation.
    use std::collections::HashMap;
    let mut reg_file: HashMap<llvm2_lower::instructions::Value, llvm2_lower::instructions::Value> =
        HashMap::new();
    for (src, dst) in &copies {
        reg_file.entry(*src).or_insert(*src);
        reg_file.entry(*dst).or_insert(*dst);
        let val = *reg_file.get(src).unwrap();
        reg_file.insert(*dst, val);
    }

    // Expected: pa <- original pb, pb <- original pc, pc <- original pa.
    assert_eq!(
        reg_file.get(&pa).copied(),
        Some(pb),
        "rotation miscompile: pa should hold original pb, got {:?}",
        reg_file.get(&pa)
    );
    assert_eq!(
        reg_file.get(&pb).copied(),
        Some(pc),
        "rotation miscompile: pb should hold original pc, got {:?}",
        reg_file.get(&pb)
    );
    assert_eq!(
        reg_file.get(&pc).copied(),
        Some(pa),
        "rotation miscompile: pc should hold original pa, got {:?}",
        reg_file.get(&pc)
    );
}

// ===========================================================================
// Test: Non-cyclic dependent copies require topological reordering
//
// This is distinct from the full 3-cycle rotation test above because it
// stresses ONLY the scheduler's ready-list ordering logic (no cycle, so
// no temp-variable insertion). A naive sequential emitter would miscompile
// this — the scheduler must reorder the copies.
//
// Pattern: loop back-edge carrying `(a, a, b)` into `header(a, b, c)`.
// This produces copy pairs:
//   (map_value(a_src) -> map_value(a_param))    -- self-copy of `a`, dropped
//   (map_value(a_src) -> map_value(b_param))    -- b <- a
//   (map_value(b_src) -> map_value(c_param))    -- c <- b
// After the self-copy drop, we have pending pairs [(a, b), (b, c)].
//   * Naive sequential order would emit `b = a` first, destroying the
//     original `b` value, then emit `c = b` which reads the NEW b (=a).
//     WRONG: c should receive the ORIGINAL b.
//   * Correct parallel scheduler detects that `b` is a source of another
//     pending copy, defers (a -> b), emits `c = b` first, then `b = a`.
//
// No cycle exists (the dependency graph is a DAG: a depends on nothing,
// b depends on a, c depends on b). So this test fails before any
// cycle-breaking code runs — it exercises the ready-list reordering.
//
// tMIR form:
//   entry(x, y, z):
//     br header(x, y, z)
//   header(a, b, c):
//     if a > 0 {
//       br header(a, a, b)     -- chain copies: (a, a, b) into (a, b, c)
//     } else {
//       br exit(a)
//     }
//   exit(r):
//     return r
// ===========================================================================

fn build_chain_reorder() -> TmirModule {
    single_function_module(
        73,
        "chain_reorder",
        func_ty(vec![Ty::I64, Ty::I64, Ty::I64], vec![Ty::I64]),
        vec![
            TmirBlock {
                id: b(0),
                params: vec![(v(0), Ty::I64), (v(1), Ty::I64), (v(2), Ty::I64)],
                body: vec![InstrNode::new(Inst::Br {
                    target: b(1),
                    args: vec![v(0), v(1), v(2)],
                })],
            },
            // header(a, b, c): cond-branch back with chain-copy, or exit
            TmirBlock {
                id: b(1),
                params: vec![
                    (v(3), Ty::I64), // a
                    (v(4), Ty::I64), // b
                    (v(5), Ty::I64), // c
                ],
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(0),
                    })
                    .with_result(v(6)),
                    InstrNode::new(Inst::ICmp {
                        op: ICmpOp::Sgt,
                        ty: Ty::I64,
                        lhs: v(3),
                        rhs: v(6),
                    })
                    .with_result(v(7)),
                    InstrNode::new(Inst::CondBr {
                        cond: v(7),
                        then_target: b(1),
                        then_args: vec![v(3), v(3), v(4)], // CHAIN: (a, a, b)
                        else_target: b(2),
                        else_args: vec![v(3)],
                    }),
                ],
            },
            // exit(r)
            TmirBlock {
                id: b(2),
                params: vec![(v(8), Ty::I64)],
                body: vec![InstrNode::new(Inst::Return { values: vec![v(8)] })],
            },
        ],
        vec![],
    )
}

#[test]
fn test_chain_reorder_parallel_copy() {
    let module = build_chain_reorder();
    let (lir_func, _) = translate_only(&module).unwrap();

    // Find the header block (non-entry, 3 params).
    let header_info: Vec<_> = lir_func
        .blocks
        .iter()
        .filter(|(id, bb)| **id != lir_func.entry_block && bb.params.len() == 3)
        .map(|(id, bb)| (*id, bb.params.iter().map(|(v, _)| *v).collect::<Vec<_>>()))
        .collect();
    assert_eq!(header_info.len(), 1);
    let (header_id, header_params) = &header_info[0];
    let pa = header_params[0];
    let pb = header_params[1];
    let pc = header_params[2];

    // Find the back-edge copy block (jumps to header AND writes at least
    // one header param).
    let back_edge_copy_block = lir_func
        .blocks
        .values()
        .find(|bb| {
            bb.params.is_empty()
                && bb.instructions.iter().any(|i| {
                    matches!(i.opcode, Opcode::Jump { dest } if dest == *header_id)
                })
                && bb.instructions.iter().any(|i| {
                    !i.results.is_empty()
                        && (i.results[0] == pa || i.results[0] == pb || i.results[0] == pc)
                })
        })
        .expect("should find back-edge copy block");

    let copies: Vec<(
        llvm2_lower::instructions::Value,
        llvm2_lower::instructions::Value,
    )> = back_edge_copy_block
        .instructions
        .iter()
        .filter(|i| !matches!(i.opcode, Opcode::Jump { .. }))
        .map(|i| (i.args[0], i.results[0]))
        .collect();

    // Non-triviality guard: at least one emitted copy's dst must equal
    // another emitted copy's src. Otherwise the test is vacuous (a naive
    // sequential emitter would also pass) and should not be trusted.
    let srcs: std::collections::HashSet<_> = copies.iter().map(|(s, _)| *s).collect();
    let has_overlap = copies.iter().any(|(_, d)| srcs.contains(d));
    assert!(
        has_overlap,
        "test is vacuous — no copy dst appears as another copy's src. \
         copies = {:?}",
        copies
    );

    // Symbolically execute the copy sequence and verify the post-state
    // matches the intended chain: (a, b, c) <- (a, a, b), i.e.
    //   new_pa = orig_pa (self-copy elided)
    //   new_pb = orig_pa
    //   new_pc = orig_pb
    use std::collections::HashMap;
    let mut reg_file: HashMap<llvm2_lower::instructions::Value, llvm2_lower::instructions::Value> =
        HashMap::new();
    for (src, dst) in &copies {
        reg_file.entry(*src).or_insert(*src);
        reg_file.entry(*dst).or_insert(*dst);
        let val = *reg_file.get(src).unwrap();
        reg_file.insert(*dst, val);
    }

    assert_eq!(
        reg_file.get(&pa).copied().unwrap_or(pa),
        pa,
        "chain miscompile: pa (self-copy) should preserve original pa"
    );
    assert_eq!(
        reg_file.get(&pb).copied(),
        Some(pa),
        "chain miscompile: pb should hold original pa, got {:?}",
        reg_file.get(&pb)
    );
    assert_eq!(
        reg_file.get(&pc).copied(),
        Some(pb),
        "chain miscompile: pc should hold original pb (NOT the new pb=pa), \
         got {:?}",
        reg_file.get(&pc)
    );
}

// ===========================================================================
// Category 10: tla-tmir coverage — Inst::Overflow with checked-arithmetic idiom
//
// tla-tmir (see ~/tla2/crates/tla-tmir/src/lower/arithmetic.rs:16-117) lowers
// every TLA+ integer Add/Sub/Mul/Neg as:
//     (value, overflow_flag) = Inst::Overflow { op, ty, lhs, rhs }
//     CondBr overflow_flag -> overflow_error_block, continue_block
// The correctness of TLA+ runtime overflow detection depends on the adapter
// producing a real overflow flag from the hardware.
//
// Issue #339 and reports/2026-04-18-tla-tmir-coverage.md document that the
// current adapter hardcodes the flag to Iconst { imm: 0 } (silent miscompile).
// The tests below are PINNING TESTS — they assert current adapter behavior
// verbatim so that any future fix to the overflow flag requires updating the
// assertion in the same diff, preventing a silent regression and making the
// fix visible.
//
// When the overflow-flag bug is fixed, these tests will fail by design and
// must be updated alongside the fix.
// ===========================================================================

fn build_tla_checked_add() -> TmirModule {
    // Mirror lower_checked_binary_overflow from tla-tmir:
    //   entry(lhs: i64, rhs: i64):
    //     (result, flag) = Inst::Overflow { AddOverflow, lhs, rhs }
    //     CondBr flag -> overflow_block, continue_block
    //   overflow_block: Return 0
    //   continue_block: Return result
    single_function_module(
        200,
        "tla_checked_add",
        func_ty(vec![Ty::I64, Ty::I64], vec![Ty::I64]),
        vec![
            TmirBlock {
                id: b(0),
                params: vec![(v(0), Ty::I64), (v(1), Ty::I64)],
                body: vec![
                    InstrNode::new(Inst::Overflow {
                        op: OverflowOp::AddOverflow,
                        ty: Ty::I64,
                        lhs: v(0),
                        rhs: v(1),
                    })
                    .with_result(v(2)) // value
                    .with_result(v(3)), // overflow flag
                    InstrNode::new(Inst::CondBr {
                        cond: v(3),
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
                body: vec![
                    InstrNode::new(Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(0),
                    })
                    .with_result(v(4)),
                    InstrNode::new(Inst::Return {
                        values: vec![v(4)],
                    }),
                ],
            },
            TmirBlock {
                id: b(2),
                params: vec![],
                body: vec![InstrNode::new(Inst::Return {
                    values: vec![v(2)],
                })],
            },
        ],
        vec![],
    )
}

#[test]
fn test_tla_checked_add_adapter_translates() {
    // Documents that the adapter at least translates the Inst::Overflow
    // instruction without raising UnsupportedInstruction.
    let module = build_tla_checked_add();
    let (lir_func, _) = translate_only(&module).expect("adapter must translate Inst::Overflow");
    assert_eq!(lir_func.name, "tla_checked_add");
    assert!(
        lir_func.blocks.len() >= 3,
        "expected at least 3 blocks (entry, overflow, continue), got {}",
        lir_func.blocks.len()
    );
}

#[test]
fn test_tla_checked_add_overflow_flag_is_real() {
    // Regression test for issue #339 Finding 1 and #474.
    //
    // History:
    //   * Originally (#339 Finding 1): adapter emitted `Iconst B1 imm 0` as
    //     the overflow flag — a silent miscompile.
    //   * First fix: adapter expanded `Inst::Overflow { AddOverflow }` to
    //     a bit-pattern overflow check using Bxor/Bnot/Band + Icmp Slt.
    //   * Second fix (#474): for I64, the adapter now emits a single
    //     `Opcode::CheckedSadd` LIR op that ISel lowers directly to the
    //     canonical AArch64 ADDS+CSET VS idiom. The bit-pattern path is
    //     retained only for I8/I16/I32 where the flag-setting idiom is
    //     not yet wired up.
    //
    // This test asserts the post-#474 I64 behavior:
    //   * Exactly one `Opcode::CheckedSadd` instruction is emitted.
    //   * It has two args (lhs, rhs) and two results (value, overflow_b1).
    //   * No `Iadd`, `Bxor`, or `Icmp { cond: SignedLessThan }` remains —
    //     those would indicate regression back to the bit-pattern lowering.
    //   * The bogus `Iconst B1 imm 0` is still absent (guards against the
    //     original #339 miscompile pattern).
    let module = build_tla_checked_add();
    let (lir_func, _) = translate_only(&module).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];

    // Exactly one CheckedSadd must appear for this I64 fixture.
    let checked_sadds: Vec<&Instruction> = entry
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::CheckedSadd))
        .collect();
    assert_eq!(
        checked_sadds.len(),
        1,
        "expected exactly one CheckedSadd for an I64 overflow add, got {}. \
         Entry instructions: {:#?}",
        checked_sadds.len(),
        entry.instructions,
    );
    let checked = checked_sadds[0];
    assert_eq!(
        checked.args.len(),
        2,
        "CheckedSadd must take [lhs, rhs] args; got {} args",
        checked.args.len()
    );
    assert_eq!(
        checked.results.len(),
        2,
        "CheckedSadd must produce [value, overflow_b1] results; got {} results",
        checked.results.len()
    );

    // Bit-pattern leftovers from the pre-#474 lowering must NOT be present.
    let has_iadd = entry
        .instructions
        .iter()
        .any(|i| matches!(i.opcode, Opcode::Iadd));
    assert!(
        !has_iadd,
        "I64 overflow add must not produce a bare Iadd after #474 — \
         the native CheckedSadd op subsumes it. Regression to bit-pattern lowering?"
    );
    let has_bxor = entry
        .instructions
        .iter()
        .any(|i| matches!(i.opcode, Opcode::Bxor));
    assert!(
        !has_bxor,
        "I64 overflow add must not produce Bxor after #474 — native flag \
         idiom doesn't need the XOR chain. Regression to bit-pattern lowering?"
    );
    let has_icmp_slt = entry.instructions.iter().any(|i| {
        matches!(&i.opcode, Opcode::Icmp { cond: IntCC::SignedLessThan })
    });
    assert!(
        !has_icmp_slt,
        "I64 overflow add must not produce Icmp SignedLessThan after #474 — \
         overflow is derived from NZCV.V via CSET VS, not a signed compare."
    );

    // The bogus Iconst { ty: B1, imm: 0 } from the original #339 miscompile
    // must remain absent. CheckedSadd is a single op that can't be constant-
    // folded away at adapter time, so this is a belt-and-suspenders check.
    let has_bogus_zero_flag = entry.instructions.iter().any(|i| {
        matches!(
            &i.opcode,
            Opcode::Iconst { ty: Type::B1, imm: 0 }
        )
    });
    assert!(
        !has_bogus_zero_flag,
        "Regression: adapter re-introduced Iconst B1 imm 0 as the overflow flag."
    );
}

// ===========================================================================
// Category 11: tla-tmir coverage — Alloca { count: Some(..) } for aggregates
//
// tla-tmir allocates sets/sequences/tuples/records via
//     let n = Const { ty: I32, value: Constant::Int(count) };
//     alloc = Alloca { ty: I64, count: Some(n), align: None }
// (see ~/tla2/crates/tla-tmir/src/lower/mod.rs:920-936 and set_ops.rs).
//
// Issue #339 and reports/2026-04-18-tla-tmir-coverage.md document that the
// current adapter ignores `count` and always produces an 8-byte stack slot.
// For a 10-element set literal this produces silent buffer overrun when the
// 2nd+ elements are stored.
//
// The test below is a PINNING TEST — it captures the current slot size so
// any future fix that honors `count` will require updating the assertion.
// ===========================================================================

fn build_tla_aggregate_alloca() -> TmirModule {
    // Allocate 10 i64 slots, store the parameter into slot 5, load it back.
    // Mirrors the set/sequence/tuple literal pattern in tla-tmir.
    single_function_module(
        210,
        "tla_aggregate_alloca",
        func_ty(vec![Ty::I64], vec![Ty::I64]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I64)],
            body: vec![
                // count = 10
                InstrNode::new(Inst::Const {
                    ty: Ty::I32,
                    value: Constant::Int(10),
                })
                .with_result(v(1)),
                // alloca 10 * i64
                InstrNode::new(Inst::Alloca {
                    ty: Ty::I64,
                    count: Some(v(1)),
                    align: None,
                })
                .with_result(v(2)),
                // idx = 5
                InstrNode::new(Inst::Const {
                    ty: Ty::I32,
                    value: Constant::Int(5),
                })
                .with_result(v(3)),
                // ptr = GEP v2[5]
                InstrNode::new(Inst::GEP {
                    pointee_ty: Ty::I64,
                    base: v(2),
                    indices: vec![v(3)],
                })
                .with_result(v(4)),
                InstrNode::new(Inst::Store {
                    ty: Ty::I64,
                    ptr: v(4),
                    value: v(0),
                    align: None,
                    volatile: false,
                }),
                InstrNode::new(Inst::Load {
                    ty: Ty::I64,
                    ptr: v(4),
                    align: None,
                    volatile: false,
                })
                .with_result(v(5)),
                InstrNode::new(Inst::Return {
                    values: vec![v(5)],
                }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_tla_aggregate_alloca_adapter_translates() {
    let module = build_tla_aggregate_alloca();
    let (lir_func, _) =
        translate_only(&module).expect("adapter must translate Alloca { count: Some(..) }");
    assert!(
        !lir_func.stack_slots.is_empty(),
        "expected at least one stack slot from Alloca"
    );
}

#[test]
fn test_tla_aggregate_alloca_slot_size_is_count_times_element_bytes() {
    // Regression test for issue #339 Finding 2 — formerly a PINNING test
    // that asserted the adapter ignored the `count` field and allocated
    // only 8 bytes. The fix constant-folds a `Some(vid)` count whose
    // producer is an `Inst::Const`, multiplies by element size, and
    // requests that many bytes from the stack slot allocator.
    //
    // The tMIR fixture allocates `count = 10` I64 elements, so the slot
    // must be 10 * 8 = 80 bytes, with 8-byte alignment.
    let module = build_tla_aggregate_alloca();
    let (lir_func, _) = translate_only(&module).unwrap();

    assert_eq!(
        lir_func.stack_slots.len(),
        1,
        "expected exactly one stack slot from the Alloca"
    );
    let slot = &lir_func.stack_slots[0];
    assert_eq!(
        slot.size, 80,
        "expected size = 10 (count) * 8 (sizeof I64) = 80 bytes. \
         If this assertion fails, the `count` folding regressed."
    );
    assert_eq!(
        slot.align, 8,
        "I64 alloca must retain 8-byte alignment regardless of count"
    );
}

// ===========================================================================
// Category 12: tla-tmir coverage — Constant::Int i128 range checks
//
// Issue #339 Finding 3: the adapter previously did `*v as i64` on a
// Constant::Int(i128) value, silently truncating any constant whose value
// did not fit in i64. Latent bug because current tla-tmir only emits
// i64-sourced literals, but explicit in the issue body ("large constants
// for set membership encoding"). Fix rejects out-of-range integer literals
// with a clear AdapterError::UnsupportedInstruction.
// ===========================================================================

fn build_i64_constant_module(value: i128) -> TmirModule {
    single_function_module(
        220,
        "i64_const",
        func_ty(vec![], vec![Ty::I64]),
        vec![TmirBlock {
            id: b(0),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(value),
                })
                .with_result(v(0)),
                InstrNode::new(Inst::Return { values: vec![v(0)] }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_i64_constant_in_range_is_accepted() {
    // Values inside i64 range must still translate successfully — do not
    // regress on normal integer literals while closing the silent-truncation
    // hole.
    for value in [0i128, 1, -1, i64::MIN as i128, i64::MAX as i128] {
        let module = build_i64_constant_module(value);
        translate_only(&module).unwrap_or_else(|e| {
            panic!("in-range i64 constant {} must translate, got {:?}", value, e)
        });
    }
}

#[test]
fn test_i64_constant_above_range_is_rejected() {
    // A Constant::Int beyond the 64-bit bit-pattern range (above u64::MAX)
    // must NOT be silently truncated. The adapter accepts values in
    // [i64::MIN, u64::MAX] for Ty::I64 (unsigned bit-patterns like hash
    // primes are common in tMIR frontends), but anything wider must
    // produce an explicit UnsupportedInstruction error.
    let module = build_i64_constant_module(u64::MAX as i128 + 1);
    let err = translate_only(&module).expect_err(
        "Constant::Int above u64::MAX must be rejected, not silently truncated",
    );
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("Constant::Int") && msg.contains("does not fit"),
        "expected UnsupportedInstruction about Constant::Int not fitting, got: {:?}",
        err
    );
}

#[test]
fn test_i64_constant_below_range_is_rejected() {
    // Symmetric check for the negative bound.
    let module = build_i64_constant_module(i64::MIN as i128 - 1);
    let err = translate_only(&module)
        .expect_err("Constant::Int below i64::MIN must be rejected");
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("Constant::Int") && msg.contains("does not fit"),
        "expected UnsupportedInstruction about Constant::Int not fitting, got: {:?}",
        err
    );
}

#[test]
fn test_i32_constant_overflow_is_rejected() {
    // Narrower target types must also reject out-of-range literals. A
    // Constant::Int with value 1 << 33 cannot fit in I32 and must error.
    let module = single_function_module(
        221,
        "i32_overflow",
        func_ty(vec![], vec![Ty::I32]),
        vec![TmirBlock {
            id: b(0),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: Ty::I32,
                    value: Constant::Int(1i128 << 33),
                })
                .with_result(v(0)),
                InstrNode::new(Inst::Return { values: vec![v(0)] }),
            ],
        }],
        vec![],
    );
    let err = translate_only(&module)
        .expect_err("Constant::Int(1<<33) in I32 context must be rejected");
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("does not fit"),
        "expected range-check error message, got: {:?}",
        err
    );
}

// ===========================================================================
// Category 13: End-to-end pinning tests for the #339 adapter fixes (#384).
//
// Issue #384 is the end-to-end follow-up to #339 and asks for integration
// tests that exercise the 3 silent-miscompile fixes (e00554a, 67292df,
// b2e2705 + 19455a8) against the full LLVM2 pipeline (adapter -> ISel ->
// MachFunction) rather than just the adapter. The pre-existing pinning
// tests earlier in this file assert adapter-level correctness
// (`translate_only`); these tests run the same canonical tMIR fixtures
// through `compile_tmir_function` and assert that the post-fix behavior
// survives the ISel lowering that a TLA+ spec would hit in tla-llvm2.
//
// Cross-repo (tla2) validation against tla-jit state graphs remains open
// on #384; those tests live in ~/tla2/crates/tla-check/tests/.
// ===========================================================================

#[test]
fn test_tla_checked_add_overflow_flag_e2e() {
    // Finding 1 regression (e00554a) + #474 native-idiom lowering:
    //
    // The adapter must emit a real overflow check for
    // `Inst::Overflow { AddOverflow }`, not `Iconst B1 imm=0`.
    //
    // After #474, for I64 fixtures the end-to-end lowering produces the
    // canonical AArch64 flag-setting idiom:
    //     ADDS  Xd, Xa, Xb       ; flag-setting add (V = signed overflow)
    //     CSET  Xov, VS          ; materialize V into a bool register
    //
    // Pre-#474 this path went through the bit-pattern expansion
    // (`~(lhs ^ rhs) & (lhs ^ sum)`, MSB=1 iff overflow) which surfaced as
    // AArch64 EOR family opcodes in the MachFunction. That idiom is now
    // gone for I64 and has been replaced by ADDS+CSET VS.
    //
    // This test pins the NEW post-#474 shape: ADDS must appear, CSET must
    // appear, and the bit-pattern EOR chain must be absent.
    let module = build_tla_checked_add();
    let mfunc = compile_tmir_function(&module);

    // ADDS must be present — it is both the wrapping value result AND the
    // NZCV.V flag source. The plain AddRR/AddRI path is no longer emitted
    // for I64 overflow adds (the adapter emits `Opcode::CheckedSadd` which
    // ISel lowers to `AArch64Opcode::AddsRR`).
    let adds_flag = count_opcode(&mfunc, AArch64Opcode::AddsRR);
    assert!(
        adds_flag >= 1,
        "expected at least one AArch64 ADDS (flag-setting add) from #474 \
         CheckedSadd lowering, got 0. MachFunction: {:#?}",
        mfunc.blocks,
    );

    // CSET must be present — materialises NZCV.V into a register. The
    // condition-code operand (VS) rides on the instruction operand list,
    // not on the opcode, so we assert the opcode count here.
    let csets = count_opcode(&mfunc, AArch64Opcode::CSet);
    assert!(
        csets >= 1,
        "expected at least one AArch64 CSET (VS) to materialise the V flag \
         produced by ADDS. Regression would mean the overflow bool is \
         unreachable or constant-folded. MachFunction: {:#?}",
        mfunc.blocks,
    );

    // The bit-pattern XOR chain must be gone for I64 — presence would mean
    // the adapter fell back to `Iadd + Bxor + Bnot + Band + Icmp` instead
    // of emitting the native CheckedSadd op.
    let xors = count_opcode(&mfunc, AArch64Opcode::EorRR)
        + count_opcode(&mfunc, AArch64Opcode::EorRI);
    assert_eq!(
        xors, 0,
        "I64 overflow add must not emit EOR after #474 — the native ADDS+CSET \
         idiom does not need a bit-pattern check. Regression to bit-pattern \
         lowering? MachFunction: {:#?}",
        mfunc.blocks,
    );

    assert!(
        has_opcode(&mfunc, AArch64Opcode::Ret),
        "expected terminating RET in the overflow-add function"
    );
}

/// Local variant of `compile_tmir_function` that also propagates
/// adapter-produced `stack_slots` onto the resulting `ISelFunction` via
/// `set_stack_slots`. Used by the #384 e2e tests that assert the slot
/// metadata survives ISel. The shared helper above deliberately omits
/// slot propagation; replicating it here keeps existing tests unchanged.
fn compile_tmir_function_with_stack_slots(module: &TmirModule) -> ISelFunction {
    let func = single_function(module);
    let (lir_func, _proof_ctx) =
        translate_function(func, module).expect("adapter translation failed");

    let mut isel = InstructionSelector::new(lir_func.name.clone(), lir_func.signature.clone());
    isel.seed_value_types(&lir_func.value_types);
    isel.set_stack_slots(lir_func.stack_slots.clone());
    isel.lower_formal_arguments(&lir_func.signature, lir_func.entry_block)
        .unwrap();

    let mut block_order: Vec<Block> = lir_func.blocks.keys().copied().collect();
    block_order.sort_by_key(|b| if *b == lir_func.entry_block { 0 } else { b.0 + 1 });
    for block_id in &block_order {
        let bb = &lir_func.blocks[block_id];
        isel.select_block_with_source_locs(*block_id, &bb.instructions, &bb.source_locs)
            .unwrap();
    }
    isel.finalize()
}

#[test]
fn test_tla_aggregate_alloca_slot_size_e2e() {
    // Finding 2 regression (67292df): Alloca { count: Some(const(10)) } on
    // an I64 element type must allocate 10 * 8 = 80 bytes of stack. End-to-
    // end this is visible on the MachFunction's stack_slots — ISel carries
    // the adapter's slot size through unchanged (via `set_stack_slots`), so
    // the post-fix value must appear in `mfunc.stack_slots`. A regression
    // that silently drops the count would show size=8 here.
    let module = build_tla_aggregate_alloca();
    let mfunc = compile_tmir_function_with_stack_slots(&module);

    assert_eq!(
        mfunc.stack_slots.len(),
        1,
        "expected exactly one stack slot from the Alloca, got {}",
        mfunc.stack_slots.len()
    );
    assert_eq!(
        mfunc.stack_slots[0].size, 80,
        "expected 10 * sizeof(I64) = 80 bytes post-fix; a size of 8 means \
         the adapter silently dropped the `count` field again."
    );
    assert_eq!(
        mfunc.stack_slots[0].align, 8,
        "I64 alloca must retain 8-byte alignment end-to-end"
    );

    // A store+load round-trip must survive ISel.
    assert!(
        has_opcode(&mfunc, AArch64Opcode::StrRI) || has_opcode(&mfunc, AArch64Opcode::StrRO),
        "expected STR for the aggregate store"
    );
    assert!(
        has_opcode(&mfunc, AArch64Opcode::LdrRI) || has_opcode(&mfunc, AArch64Opcode::LdrRO),
        "expected LDR for the aggregate load"
    );
}

#[test]
fn test_tla_i64_constant_range_e2e() {
    // Finding 3 regression (b2e2705 + 19455a8): Constant::Int(i128) must
    // be range-checked against the target type, NOT silently truncated via
    // `*v as i64`. End-to-end the post-fix contract is two-sided:
    //
    // 1. An in-range I64 constant compiles cleanly through ISel (we use
    //    i64::MAX as the canonical representative) and reaches a MOV family
    //    opcode in the MachFunction.
    // 2. An out-of-range wide-imm (u64::MAX + 1, which exceeds both the
    //    signed and u64 bit-pattern windows) is rejected at the adapter
    //    layer — no MachFunction is produced. A regression to
    //    truncation-by-cast would allow the wide-imm to silently pass and
    //    produce a bogus MachFunction with a truncated `imm`.
    let ok_module = build_i64_constant_module(i64::MAX as i128);
    let ok_mfunc = compile_tmir_function(&ok_module);
    let has_mov = has_opcode(&ok_mfunc, AArch64Opcode::Movz)
        || has_opcode(&ok_mfunc, AArch64Opcode::MovI)
        || has_opcode(&ok_mfunc, AArch64Opcode::MovR)
        || has_opcode(&ok_mfunc, AArch64Opcode::Movn)
        || has_opcode(&ok_mfunc, AArch64Opcode::Movk)
        || has_opcode(&ok_mfunc, AArch64Opcode::LdrLiteral);
    assert!(
        has_mov,
        "expected a MOV-family or LDR-literal opcode for the i64 constant \
         materialization; got MachFunction: {:#?}",
        ok_mfunc.blocks,
    );
    assert!(
        has_opcode(&ok_mfunc, AArch64Opcode::Ret),
        "expected RET in the i64 constant function"
    );

    // Out-of-range wide immediate must error at the adapter — ISel never
    // runs because translate_only returns Err. u64::MAX as i128 + 1 is
    // above the u64-bit-pattern acceptance window, so it still fails even
    // after the 19455a8 u64-bit-pattern widening.
    let too_wide = (u64::MAX as i128) + 1;
    let bad_module = build_i64_constant_module(too_wide);
    let err = translate_only(&bad_module)
        .expect_err("Constant::Int above u64::MAX must be rejected end-to-end");
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("does not fit"),
        "expected adapter range-check error, got: {:?}",
        err
    );
}

// ===========================================================================
// GEP stride contract (#475)
//
// Pins the ABI-critical stride convention: GEP { pointee_ty: I64, base, [idx] }
// must lower to `base + idx * 8`. External consumers (z4, tla2) rely on
// `stride = sizeof(pointee_ty)`; see the `llvm2-ir` crate docs.
// ===========================================================================

fn build_gep_stride_contract_i64() -> TmirModule {
    single_function_module(
        9999,
        "gep_stride_contract_i64",
        func_ty(vec![Ty::Ptr, Ty::I64], vec![Ty::Ptr]),
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
                InstrNode::new(Inst::Return { values: vec![v(2)] }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_gep_stride_contract_i64() {
    // Contract (#475): for pointee_ty = I64 (sizeof = 8), a single-index GEP
    // must lower to `base + index * 8`, implemented as:
    //   Iconst { ty: I64, imm: 8 }  (stride materialization)
    //   Imul                         (scaled index)
    //   Iadd                         (base + scaled)
    let module = build_gep_stride_contract_i64();
    let (lir_func, _) = translate_only(&module).expect("adapter translation failed");
    let entry = &lir_func.blocks[&lir_func.entry_block];

    let has_stride_8 = entry.instructions.iter().any(|inst| {
        matches!(
            inst.opcode,
            Opcode::Iconst {
                ty: Type::I64,
                imm: 8,
            }
        )
    });
    assert!(
        has_stride_8,
        "GEP stride contract (#475): expected Iconst {{ ty: I64, imm: 8 }} \
         (sizeof(I64) stride) but got {:?}",
        entry.instructions.iter().map(|i| &i.opcode).collect::<Vec<_>>()
    );

    let has_imul = entry
        .instructions
        .iter()
        .any(|inst| matches!(inst.opcode, Opcode::Imul));
    assert!(
        has_imul,
        "GEP stride contract (#475): expected Imul for index * stride"
    );

    let has_iadd = entry
        .instructions
        .iter()
        .any(|inst| matches!(inst.opcode, Opcode::Iadd));
    assert!(
        has_iadd,
        "GEP stride contract (#475): expected Iadd for base + scaled_index"
    );
}

// ===========================================================================
// Category 14: EWD998Small-shaped tla-tmir Record/Sequence aggregates (#384)
//
// Issue #384 tracks end-to-end validation for the #339 adapter fixes. A real
// cross-repo tla-check harness (AddTwoTest / DieHardTLA / bcastFolklore_small
// vs tla-jit) is filed as a separate tla2-side tracker so tla2 owns the
// driver; see `reports/2026-04-20-384-ewd998-status.md`.
//
// The LLVM2-side deliverable is a pinning regression that captures the exact
// nested Record / Sequence aggregate shape a TLA+ spec emits through
// tla-tmir. The canonical pattern (from
// `~/tla2/crates/tla-tmir/src/lower/mod.rs:1322-1431`,
// `~/tla2/crates/tla-tmir/src/lower/sequences.rs:70-108`,
// `~/tla2/crates/tla-tmir/src/lower/set_ops.rs:25-60`) is:
//
//     %cnt  = Const I32 N
//     %agg  = Alloca I64, count: Some(%cnt)                 ; N-slot aggregate
//     for i in 0..N {
//         %idx = Const I32 i
//         %ptr = GEP i64, %agg, [%idx]
//         Store i64, %ptr, %vi                              ; or Load
//     }
//
// That is the shape EWD998Small would hit at every `RecordNew`/`SetEnum`/
// `Append` operation. The pre-existing `test_tla_aggregate_alloca_*` tests
// exercise *one* offset (slot 5); the EWD998-shaped regressions below stress
// the full multi-offset store+load pattern, which is what tla-check actually
// observes between states. If the adapter or ISel ever regresses on any
// single slot, these tests fail loudly.
//
// Substitute specs (per #384 body and `reports/2026-04-18-339-ewd998-plan.md`):
//   - Record: 4-field record, like a `DieHard` state `[big |-> i, small |-> j]`
//   - Sequence: length-header + 3 elements, like a small `Append` step
//
// The cross-repo tla-jit diff work tracked by the tla2-side tracker remains
// the closure gate for #384; these pins are the LLVM2 half.
// ===========================================================================

/// EWD998-shape: a 4-field Record — `Alloca(count=4)` + 4 × `Const+GEP+Store`
/// then 4 × `Const+GEP+Load`, finally summing the first two loaded fields.
///
/// Mirrors `tla-tmir::lower_record_new`
/// (`~/tla2/crates/tla-tmir/src/lower/sequences.rs:77-93`), which is what a
/// `[a |-> x, b |-> y, c |-> z, d |-> w]` record literal expands to. The
/// summed-fields return keeps the aggregate observably live so DCE can't
/// erase the stores.
fn build_tla_record_new_4fields() -> TmirModule {
    // (v0, v1, v2, v3) are the 4 incoming i64 field values.
    // Locals:
    //   v10..v13 = Const I32 {0,1,2,3} for field indices
    //   v20      = Const I32 4           (aggregate slot count)
    //   v21      = Alloca I64, count v20
    //   v30..v33 = GEP v21 [v10..v13]    (per-field slot pointers)
    //   v40..v43 = Load from v30..v33    (read back for verification)
    //   v50      = Iadd v40, v41         (observably uses the loads)
    single_function_module(
        310,
        "tla_record_new_4fields",
        func_ty(vec![Ty::I64, Ty::I64, Ty::I64, Ty::I64], vec![Ty::I64]),
        vec![TmirBlock {
            id: b(0),
            params: vec![
                (v(0), Ty::I64),
                (v(1), Ty::I64),
                (v(2), Ty::I64),
                (v(3), Ty::I64),
            ],
            body: vec![
                // Field indices (tla-tmir uses I32 constants for offsets)
                InstrNode::new(Inst::Const { ty: Ty::I32, value: Constant::Int(0) })
                    .with_result(v(10)),
                InstrNode::new(Inst::Const { ty: Ty::I32, value: Constant::Int(1) })
                    .with_result(v(11)),
                InstrNode::new(Inst::Const { ty: Ty::I32, value: Constant::Int(2) })
                    .with_result(v(12)),
                InstrNode::new(Inst::Const { ty: Ty::I32, value: Constant::Int(3) })
                    .with_result(v(13)),
                // Aggregate count + allocation (canonical alloc_aggregate pattern)
                InstrNode::new(Inst::Const { ty: Ty::I32, value: Constant::Int(4) })
                    .with_result(v(20)),
                InstrNode::new(Inst::Alloca {
                    ty: Ty::I64,
                    count: Some(v(20)),
                    align: None,
                })
                .with_result(v(21)),
                // Per-field stores: store_at_offset(agg, i, v_i)
                InstrNode::new(Inst::GEP {
                    pointee_ty: Ty::I64,
                    base: v(21),
                    indices: vec![v(10)],
                })
                .with_result(v(30)),
                InstrNode::new(Inst::Store {
                    ty: Ty::I64,
                    ptr: v(30),
                    value: v(0),
                    align: None,
                    volatile: false,
                }),
                InstrNode::new(Inst::GEP {
                    pointee_ty: Ty::I64,
                    base: v(21),
                    indices: vec![v(11)],
                })
                .with_result(v(31)),
                InstrNode::new(Inst::Store {
                    ty: Ty::I64,
                    ptr: v(31),
                    value: v(1),
                    align: None,
                    volatile: false,
                }),
                InstrNode::new(Inst::GEP {
                    pointee_ty: Ty::I64,
                    base: v(21),
                    indices: vec![v(12)],
                })
                .with_result(v(32)),
                InstrNode::new(Inst::Store {
                    ty: Ty::I64,
                    ptr: v(32),
                    value: v(2),
                    align: None,
                    volatile: false,
                }),
                InstrNode::new(Inst::GEP {
                    pointee_ty: Ty::I64,
                    base: v(21),
                    indices: vec![v(13)],
                })
                .with_result(v(33)),
                InstrNode::new(Inst::Store {
                    ty: Ty::I64,
                    ptr: v(33),
                    value: v(3),
                    align: None,
                    volatile: false,
                }),
                // Read slot[0] and slot[1] back for observability
                // (fresh GEPs — tla-tmir always re-emits the GEP per load/store)
                InstrNode::new(Inst::GEP {
                    pointee_ty: Ty::I64,
                    base: v(21),
                    indices: vec![v(10)],
                })
                .with_result(v(40)),
                InstrNode::new(Inst::Load {
                    ty: Ty::I64,
                    ptr: v(40),
                    align: None,
                    volatile: false,
                })
                .with_result(v(41)),
                InstrNode::new(Inst::GEP {
                    pointee_ty: Ty::I64,
                    base: v(21),
                    indices: vec![v(11)],
                })
                .with_result(v(42)),
                InstrNode::new(Inst::Load {
                    ty: Ty::I64,
                    ptr: v(42),
                    align: None,
                    volatile: false,
                })
                .with_result(v(43)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: v(41),
                    rhs: v(43),
                })
                .with_result(v(50)),
                InstrNode::new(Inst::Return {
                    values: vec![v(50)],
                }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_tla_record_new_4fields_adapter() {
    // Adapter layer must translate the full record-new pattern:
    // Alloca(count=4) must fold into a single 32-byte slot, and all 4 stores
    // plus 2 loads must reach the LIR entry block.
    let module = build_tla_record_new_4fields();
    let (lir_func, _) =
        translate_only(&module).expect("adapter must translate 4-field record pattern");

    assert_eq!(
        lir_func.stack_slots.len(),
        1,
        "4-field record must produce exactly one Alloca-backed slot, got {}",
        lir_func.stack_slots.len()
    );
    assert_eq!(
        lir_func.stack_slots[0].size, 32,
        "4 * sizeof(I64) = 32 bytes expected; got {}. A size of 8 would mean \
         the adapter regressed on the #339 Finding 2 `count` fold.",
        lir_func.stack_slots[0].size
    );
    assert_eq!(
        lir_func.stack_slots[0].align, 8,
        "I64 record aggregate must retain 8-byte alignment"
    );

    let entry = &lir_func.blocks[&lir_func.entry_block];
    let store_count = entry
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::Store))
        .count();
    assert_eq!(
        store_count, 4,
        "expected exactly 4 Store opcodes (one per record field), got {}",
        store_count
    );
    let load_count = entry
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::Load { ty: Type::I64 }))
        .count();
    assert_eq!(
        load_count, 2,
        "expected exactly 2 Load opcodes (for the summed-back field reads), got {}",
        load_count
    );
}

#[test]
fn test_tla_record_new_4fields_isel() {
    // End-to-end: after adapter + ISel, the AArch64 MachFunction must contain
    // 4 STR (per-field store) and 2 LDR (summed-back reads). The parallel
    // field operations also stress register allocation — a regression where
    // GEP+Store gets folded into a single base-only STR would drop per-field
    // offset correctness (the DieHardTLA test would diverge immediately).
    let module = build_tla_record_new_4fields();
    let mfunc = compile_tmir_function_with_stack_slots(&module);

    let strs = count_opcode(&mfunc, AArch64Opcode::StrRI)
        + count_opcode(&mfunc, AArch64Opcode::StrRO);
    assert!(
        strs >= 4,
        "expected >=4 STR opcodes for the 4 record-field stores, got {}. \
         MachFunction blocks: {:#?}",
        strs,
        mfunc.blocks,
    );

    let ldrs = count_opcode(&mfunc, AArch64Opcode::LdrRI)
        + count_opcode(&mfunc, AArch64Opcode::LdrRO);
    assert!(
        ldrs >= 2,
        "expected >=2 LDR opcodes for the slot-0 + slot-1 read-back, got {}. \
         MachFunction blocks: {:#?}",
        ldrs,
        mfunc.blocks,
    );

    assert_eq!(
        mfunc.stack_slots.len(),
        1,
        "ISel must carry the single 32-byte slot through unchanged"
    );
    assert_eq!(
        mfunc.stack_slots[0].size, 32,
        "record slot must remain 32 bytes end-to-end"
    );

    assert!(
        has_opcode(&mfunc, AArch64Opcode::Ret),
        "record return must terminate with RET"
    );
}

/// EWD998-shape: a length-prefixed Sequence aggregate with 3 elements —
/// `Alloca(count=4)` (1 header + 3 data), store length at slot[0], stores
/// at slot[1..4], then `Len(seq)` reads slot[0] and `Head(seq)` reads
/// slot[1]. Mirrors `tla-tmir::lower_seq_len`/`lower_seq_head` calls on a
/// sequence built by `lower_seq_enum` (`~/tla2/crates/tla-tmir/src/lower/
/// constants.rs:91-125` and `sequences.rs:127-152`).
///
/// The returned value is `Len(seq) + Head(seq)`, which keeps both loads
/// observably live.
fn build_tla_sequence_len_plus_head() -> TmirModule {
    single_function_module(
        311,
        "tla_sequence_len_plus_head",
        func_ty(vec![Ty::I64, Ty::I64, Ty::I64], vec![Ty::I64]),
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I64), (v(1), Ty::I64), (v(2), Ty::I64)],
            body: vec![
                // Offset constants: 0 (length header), 1/2/3 (elements)
                InstrNode::new(Inst::Const { ty: Ty::I32, value: Constant::Int(0) })
                    .with_result(v(10)),
                InstrNode::new(Inst::Const { ty: Ty::I32, value: Constant::Int(1) })
                    .with_result(v(11)),
                InstrNode::new(Inst::Const { ty: Ty::I32, value: Constant::Int(2) })
                    .with_result(v(12)),
                InstrNode::new(Inst::Const { ty: Ty::I32, value: Constant::Int(3) })
                    .with_result(v(13)),
                // Aggregate total slot count: 1 (length) + 3 (elements) = 4
                InstrNode::new(Inst::Const { ty: Ty::I32, value: Constant::Int(4) })
                    .with_result(v(14)),
                // Length value (number of elements, stored at slot[0])
                InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(3) })
                    .with_result(v(15)),
                // Alloca: total slot count (1 header + 3 data)
                InstrNode::new(Inst::Alloca {
                    ty: Ty::I64,
                    count: Some(v(14)),
                    align: None,
                })
                .with_result(v(20)),
                // Store length at slot[0]
                InstrNode::new(Inst::GEP {
                    pointee_ty: Ty::I64,
                    base: v(20),
                    indices: vec![v(10)],
                })
                .with_result(v(30)),
                InstrNode::new(Inst::Store {
                    ty: Ty::I64,
                    ptr: v(30),
                    value: v(15),
                    align: None,
                    volatile: false,
                }),
                // Store elements at slots [1], [2], [3]
                InstrNode::new(Inst::GEP {
                    pointee_ty: Ty::I64,
                    base: v(20),
                    indices: vec![v(11)],
                })
                .with_result(v(31)),
                InstrNode::new(Inst::Store {
                    ty: Ty::I64,
                    ptr: v(31),
                    value: v(0),
                    align: None,
                    volatile: false,
                }),
                InstrNode::new(Inst::GEP {
                    pointee_ty: Ty::I64,
                    base: v(20),
                    indices: vec![v(12)],
                })
                .with_result(v(32)),
                InstrNode::new(Inst::Store {
                    ty: Ty::I64,
                    ptr: v(32),
                    value: v(1),
                    align: None,
                    volatile: false,
                }),
                InstrNode::new(Inst::GEP {
                    pointee_ty: Ty::I64,
                    base: v(20),
                    indices: vec![v(13)],
                })
                .with_result(v(33)),
                InstrNode::new(Inst::Store {
                    ty: Ty::I64,
                    ptr: v(33),
                    value: v(2),
                    align: None,
                    volatile: false,
                }),
                // Len(seq): reload slot[0]
                InstrNode::new(Inst::GEP {
                    pointee_ty: Ty::I64,
                    base: v(20),
                    indices: vec![v(10)],
                })
                .with_result(v(40)),
                InstrNode::new(Inst::Load {
                    ty: Ty::I64,
                    ptr: v(40),
                    align: None,
                    volatile: false,
                })
                .with_result(v(41)),
                // Head(seq): reload slot[1]
                InstrNode::new(Inst::GEP {
                    pointee_ty: Ty::I64,
                    base: v(20),
                    indices: vec![v(11)],
                })
                .with_result(v(42)),
                InstrNode::new(Inst::Load {
                    ty: Ty::I64,
                    ptr: v(42),
                    align: None,
                    volatile: false,
                })
                .with_result(v(43)),
                InstrNode::new(Inst::BinOp {
                    op: BinOp::Add,
                    ty: Ty::I64,
                    lhs: v(41),
                    rhs: v(43),
                })
                .with_result(v(50)),
                InstrNode::new(Inst::Return {
                    values: vec![v(50)],
                }),
            ],
        }],
        vec![],
    )
}

#[test]
fn test_tla_sequence_len_plus_head_adapter() {
    // Adapter must allocate 4 * 8 = 32 bytes for the 1-header + 3-data layout
    // and lower all 4 stores (length + 3 elements) plus 2 loads (Len + Head).
    let module = build_tla_sequence_len_plus_head();
    let (lir_func, _) = translate_only(&module)
        .expect("adapter must translate length-prefixed sequence pattern");

    assert_eq!(
        lir_func.stack_slots.len(),
        1,
        "sequence aggregate must produce exactly one Alloca-backed slot"
    );
    assert_eq!(
        lir_func.stack_slots[0].size, 32,
        "expected 4 * 8 = 32 bytes (1 length + 3 elements); \
         got {}. A size of 8 means the adapter regressed on Alloca `count` fold.",
        lir_func.stack_slots[0].size
    );

    let entry = &lir_func.blocks[&lir_func.entry_block];
    let store_count = entry
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::Store))
        .count();
    assert_eq!(
        store_count, 4,
        "expected 4 Stores (length + 3 elements); got {}",
        store_count
    );
    let load_count = entry
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::Load { ty: Type::I64 }))
        .count();
    assert_eq!(
        load_count, 2,
        "expected 2 Loads (Len + Head readbacks); got {}",
        load_count
    );
}

#[test]
fn test_tla_sequence_len_plus_head_isel() {
    // End-to-end ISel pin: ensure the length-prefixed sequence shape survives
    // register allocation and produces the expected STR/LDR counts. Stride
    // must be 8 (i64 contract, #475): a GEP-miscompile would put slot[1]'s
    // store at the wrong byte offset, which EWD998Small-style state hashing
    // would surface as state-graph divergence vs tla-jit.
    let module = build_tla_sequence_len_plus_head();
    let mfunc = compile_tmir_function_with_stack_slots(&module);

    let strs = count_opcode(&mfunc, AArch64Opcode::StrRI)
        + count_opcode(&mfunc, AArch64Opcode::StrRO);
    assert!(
        strs >= 4,
        "expected >=4 STR opcodes for header + 3 elements, got {}. \
         MachFunction: {:#?}",
        strs,
        mfunc.blocks,
    );

    let ldrs = count_opcode(&mfunc, AArch64Opcode::LdrRI)
        + count_opcode(&mfunc, AArch64Opcode::LdrRO);
    assert!(
        ldrs >= 2,
        "expected >=2 LDR opcodes for Len + Head readbacks, got {}. \
         MachFunction: {:#?}",
        ldrs,
        mfunc.blocks,
    );

    assert_eq!(
        mfunc.stack_slots.len(),
        1,
        "ISel must carry the 32-byte slot through unchanged"
    );
    assert_eq!(
        mfunc.stack_slots[0].size, 32,
        "sequence slot must remain 32 bytes end-to-end"
    );

    assert!(
        has_opcode(&mfunc, AArch64Opcode::Ret),
        "sequence-return function must terminate with RET"
    );
}
