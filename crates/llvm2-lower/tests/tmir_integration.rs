// tmir_integration.rs - End-to-end tMIR -> adapter -> ISel integration tests
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Tests tMIR programs through the full adapter->ISel pipeline.

use llvm2_lower::adapter::{translate_function, translate_module, AdapterError, ProofContext};
use llvm2_lower::function::Function;
use llvm2_lower::isel::{AArch64Opcode, InstructionSelector, ISelFunction};
use llvm2_lower::instructions::{Block, Opcode};
use llvm2_lower::types::Type;

use tmir::{
    Block as TmirBlock, Function as TmirFunction, Module as TmirModule,
    BinOp, CastOp, UnOp, Inst, InstrNode, ICmpOp, Constant,
    Ty, BlockId, FuncId, ValueId, FuncTy, ProofAnnotation,
    SwitchCase,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn v(n: u32) -> ValueId { ValueId::new(n) }
fn b(n: u32) -> BlockId { BlockId::new(n) }

fn node(inst: Inst, results: Vec<ValueId>, proofs: Vec<ProofAnnotation>) -> InstrNode {
    InstrNode { inst, results, proofs, span: None }
}

/// Build a simple module with one function.
fn simple_module(
    name: &str,
    func_name: &str,
    params: Vec<Ty>,
    returns: Vec<Ty>,
    blocks: Vec<TmirBlock>,
) -> TmirModule {
    let mut module = TmirModule::new(name);
    let ft_id = module.add_func_type(FuncTy { params, returns, is_vararg: false });
    let entry = blocks.first().map(|b| b.id).unwrap_or(BlockId::new(0));
    module.add_function(TmirFunction {
        id: FuncId::new(0),
        name: func_name.to_string(),
        ty: ft_id,
        entry,
        blocks,
        proofs: vec![],
    });
    module
}

/// Translate a tMIR function through the adapter, then run ISel on all blocks.
fn compile_tmir_function(func: &TmirFunction, module: &TmirModule) -> ISelFunction {
    let (lir_func, _proof_ctx) =
        translate_function(func, module).expect("adapter translation failed");

    let mut isel = InstructionSelector::new(lir_func.name.clone(), lir_func.signature.clone());
    isel.lower_formal_arguments(&lir_func.signature, lir_func.entry_block).unwrap();

    let mut block_order: Vec<Block> = lir_func.blocks.keys().copied().collect();
    block_order.sort_by_key(|b| {
        if *b == lir_func.entry_block { 0 } else { b.0 + 1 }
    });

    for block_id in &block_order {
        let bb = &lir_func.blocks[block_id];
        isel.select_block(*block_id, &bb.instructions).unwrap();
    }

    isel.finalize()
}

/// Translate only through the adapter.
fn translate_only(func: &TmirFunction, module: &TmirModule) -> Result<(Function, ProofContext), AdapterError> {
    translate_function(func, module)
}

fn count_opcode(mfunc: &ISelFunction, opcode: AArch64Opcode) -> usize {
    mfunc.blocks.values()
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

fn build_identity_module() -> TmirModule {
    simple_module("test", "identity",
        vec![Ty::I32], vec![Ty::I32],
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I32)],
            body: vec![node(Inst::Return { values: vec![v(0)] }, vec![], vec![])],
        }],
    )
}

#[test]
fn test_identity_adapter() {
    let module = build_identity_module();
    let (lir_func, proof_ctx) = translate_only(&module.functions[0], &module).unwrap();

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
    let module = build_identity_module();
    let mfunc = compile_tmir_function(&module.functions[0], &module);

    assert_eq!(mfunc.name, "identity");
    assert!(!mfunc.blocks.is_empty());
    assert!(has_opcode(&mfunc, AArch64Opcode::Copy));
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test 2: add(a: i32, b: i32) -> i32 { a + b }
// ===========================================================================

fn build_add_module() -> TmirModule {
    simple_module("test", "add",
        vec![Ty::I32, Ty::I32], vec![Ty::I32],
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I32), (v(1), Ty::I32)],
            body: vec![
                node(Inst::BinOp { op: BinOp::Add, ty: Ty::I32, lhs: v(0), rhs: v(1) }, vec![v(2)], vec![]),
                node(Inst::Return { values: vec![v(2)] }, vec![], vec![]),
            ],
        }],
    )
}

#[test]
fn test_add_adapter() {
    let module = build_add_module();
    let (lir_func, _) = translate_only(&module.functions[0], &module).unwrap();

    assert_eq!(lir_func.name, "add");
    assert_eq!(lir_func.signature.params, vec![Type::I32, Type::I32]);

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.params.len(), 2);
    assert_eq!(entry.instructions.len(), 2);
}

#[test]
fn test_add_isel() {
    let module = build_add_module();
    let mfunc = compile_tmir_function(&module.functions[0], &module);

    assert_eq!(mfunc.name, "add");
    assert!(has_opcode(&mfunc, AArch64Opcode::AddRR), "Expected ADDWrr for i32 addition");
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test 3: negate(x: i32) -> i32 { -x }
// ===========================================================================

fn build_negate_module() -> TmirModule {
    simple_module("test", "negate",
        vec![Ty::I32], vec![Ty::I32],
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I32)],
            body: vec![
                node(Inst::UnOp { op: UnOp::Neg, ty: Ty::I32, operand: v(0) }, vec![v(1)], vec![]),
                node(Inst::Return { values: vec![v(1)] }, vec![], vec![]),
            ],
        }],
    )
}

#[test]
fn test_negate_adapter() {
    let module = build_negate_module();
    let (lir_func, _) = translate_only(&module.functions[0], &module).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 2);
}

#[test]
fn test_negate_isel() {
    let module = build_negate_module();
    let mfunc = compile_tmir_function(&module.functions[0], &module);

    assert!(has_opcode(&mfunc, AArch64Opcode::Neg), "Expected NEGWr for negation (-x)");
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test 4: max(a: i32, b: i32) -> i32 { if a > b { a } else { b } }
// ===========================================================================

fn build_max_module() -> TmirModule {
    simple_module("test", "max",
        vec![Ty::I32, Ty::I32], vec![Ty::I32],
        vec![
            TmirBlock {
                id: b(0),
                params: vec![(v(0), Ty::I32), (v(1), Ty::I32)],
                body: vec![
                    node(Inst::ICmp { op: ICmpOp::Sgt, ty: Ty::I32, lhs: v(0), rhs: v(1) }, vec![v(2)], vec![]),
                    node(Inst::CondBr {
                        cond: v(2),
                        then_target: b(1), then_args: vec![],
                        else_target: b(2), else_args: vec![],
                    }, vec![], vec![]),
                ],
            },
            TmirBlock { id: b(1), params: vec![], body: vec![
                node(Inst::Return { values: vec![v(0)] }, vec![], vec![]),
            ]},
            TmirBlock { id: b(2), params: vec![], body: vec![
                node(Inst::Return { values: vec![v(1)] }, vec![], vec![]),
            ]},
        ],
    )
}

#[test]
fn test_max_adapter() {
    let module = build_max_module();
    let (lir_func, _) = translate_only(&module.functions[0], &module).unwrap();

    assert_eq!(lir_func.blocks.len(), 3);
    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 2);
}

#[test]
fn test_max_isel() {
    let module = build_max_module();
    let mfunc = compile_tmir_function(&module.functions[0], &module);

    assert_eq!(mfunc.blocks.len(), 3);
    assert!(has_opcode(&mfunc, AArch64Opcode::CmpRR), "Expected CMPWrr");
    assert!(has_opcode(&mfunc, AArch64Opcode::BCond), "Expected Bcc");
    assert!(count_opcode(&mfunc, AArch64Opcode::Ret) >= 2);
}

// ===========================================================================
// Test 5: sum(n: i32) -> i32 { let mut s = 0; while n > 0 { s += n; n -= 1; } s }
// ===========================================================================

fn build_sum_module() -> TmirModule {
    simple_module("test", "sum",
        vec![Ty::I32], vec![Ty::I32],
        vec![
            // block0: entry
            TmirBlock {
                id: b(0),
                params: vec![(v(0), Ty::I32)],
                body: vec![
                    node(Inst::Const { ty: Ty::I32, value: Constant::Int(0) }, vec![v(1)], vec![]),
                    node(Inst::Br { target: b(1), args: vec![v(0), v(1)] }, vec![], vec![]),
                ],
            },
            // block1: loop header
            TmirBlock {
                id: b(1),
                params: vec![(v(2), Ty::I32), (v(3), Ty::I32)],
                body: vec![
                    node(Inst::Const { ty: Ty::I32, value: Constant::Int(0) }, vec![v(4)], vec![]),
                    node(Inst::ICmp { op: ICmpOp::Sgt, ty: Ty::I32, lhs: v(2), rhs: v(4) }, vec![v(5)], vec![]),
                    node(Inst::CondBr {
                        cond: v(5),
                        then_target: b(2), then_args: vec![],
                        else_target: b(3), else_args: vec![],
                    }, vec![], vec![]),
                ],
            },
            // block2: loop body
            TmirBlock {
                id: b(2),
                params: vec![],
                body: vec![
                    node(Inst::BinOp { op: BinOp::Add, ty: Ty::I32, lhs: v(3), rhs: v(2) }, vec![v(6)], vec![]),
                    node(Inst::Const { ty: Ty::I32, value: Constant::Int(1) }, vec![v(7)], vec![]),
                    node(Inst::BinOp { op: BinOp::Sub, ty: Ty::I32, lhs: v(2), rhs: v(7) }, vec![v(8)], vec![]),
                    node(Inst::Br { target: b(1), args: vec![v(8), v(6)] }, vec![], vec![]),
                ],
            },
            // block3: exit
            TmirBlock {
                id: b(3),
                params: vec![],
                body: vec![node(Inst::Return { values: vec![v(3)] }, vec![], vec![])],
            },
        ],
    )
}

#[test]
fn test_sum_adapter() {
    let module = build_sum_module();
    let (lir_func, _) = translate_only(&module.functions[0], &module).unwrap();

    assert_eq!(lir_func.name, "sum");
    assert_eq!(lir_func.blocks.len(), 4);

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert!(entry.instructions.len() >= 2, "Entry should have at least Iconst + Jump, got {}", entry.instructions.len());
}

#[test]
fn test_sum_isel_body_instructions() {
    use llvm2_lower::function::Signature;
    use llvm2_lower::instructions::{Instruction, Opcode, Value};

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
                opcode: Opcode::Iconst { ty: Type::I32, imm: 1 },
                args: vec![],
                results: vec![Value(3)],
            },
            Instruction {
                opcode: Opcode::Isub,
                args: vec![Value(1), Value(3)],
                results: vec![Value(4)],
            },
            Instruction {
                opcode: Opcode::Iconst { ty: Type::I32, imm: 0 },
                args: vec![],
                results: vec![Value(5)],
            },
            Instruction {
                opcode: Opcode::Icmp { cond: llvm2_lower::instructions::IntCC::SignedGreaterThan },
                args: vec![Value(4), Value(5)],
                results: vec![Value(6)],
            },
            Instruction {
                opcode: Opcode::Return,
                args: vec![Value(2)],
                results: vec![],
            },
        ],
    ).unwrap();

    let mfunc = isel.finalize();
    assert!(has_opcode(&mfunc, AArch64Opcode::AddRR));
    assert!(has_opcode(&mfunc, AArch64Opcode::SubRR));
    assert!(has_opcode(&mfunc, AArch64Opcode::CmpRR));
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test 6: load_store(p: *mut i32, v: i32) { *p = v; }
// ===========================================================================

fn build_load_store_module() -> TmirModule {
    simple_module("test", "load_store",
        vec![Ty::Ptr, Ty::I32], vec![],
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::Ptr), (v(1), Ty::I32)],
            body: vec![
                node(Inst::Store { ty: Ty::I32, ptr: v(0), value: v(1) }, vec![], vec![]),
                node(Inst::Return { values: vec![] }, vec![], vec![]),
            ],
        }],
    )
}

#[test]
fn test_load_store_adapter() {
    let module = build_load_store_module();
    let (lir_func, _) = translate_only(&module.functions[0], &module).unwrap();

    assert_eq!(lir_func.name, "load_store");
    assert_eq!(lir_func.signature.params, vec![Type::I64, Type::I32]);
    assert_eq!(lir_func.signature.returns, vec![]);

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 2);
}

#[test]
fn test_load_store_isel() {
    let module = build_load_store_module();
    let mfunc = compile_tmir_function(&module.functions[0], &module);

    assert!(has_opcode(&mfunc, AArch64Opcode::StrRI), "Expected STRWui for 32-bit store");
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test 7: load then store
// ===========================================================================

fn build_load_then_store_module() -> TmirModule {
    simple_module("test", "load_then_store",
        vec![Ty::Ptr], vec![],
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::Ptr)],
            body: vec![
                node(Inst::Load { ty: Ty::I32, ptr: v(0) }, vec![v(1)], vec![]),
                node(Inst::Store { ty: Ty::I32, ptr: v(0), value: v(1) }, vec![], vec![]),
                node(Inst::Return { values: vec![] }, vec![], vec![]),
            ],
        }],
    )
}

#[test]
fn test_load_then_store_isel() {
    let module = build_load_then_store_module();
    let mfunc = compile_tmir_function(&module.functions[0], &module);

    assert!(has_opcode(&mfunc, AArch64Opcode::LdrRI), "Expected LDRWui for 32-bit load");
    assert!(has_opcode(&mfunc, AArch64Opcode::StrRI), "Expected STRWui for 32-bit store");
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test: all single-block programs compile without panic
// ===========================================================================

#[test]
fn test_all_single_block_programs_compile_without_panic() {
    let modules: Vec<(&str, TmirModule)> = vec![
        ("identity", build_identity_module()),
        ("add", build_add_module()),
        ("negate", build_negate_module()),
        ("max", build_max_module()),
        ("load_store", build_load_store_module()),
    ];

    for (name, module) in &modules {
        let mfunc = compile_tmir_function(&module.functions[0], module);
        assert!(!mfunc.blocks.is_empty(), "{}: produced empty ISelFunction", name);
        assert!(total_insts(&mfunc) > 0, "{}: produced no machine instructions", name);
        assert!(has_opcode(&mfunc, AArch64Opcode::Ret), "{}: missing RET instruction", name);
        assert!(has_opcode(&mfunc, AArch64Opcode::Copy), "{}: missing COPY for formal arguments", name);
    }
}

#[test]
fn test_all_programs_adapter_succeeds() {
    let modules: Vec<(&str, TmirModule)> = vec![
        ("identity", build_identity_module()),
        ("add", build_add_module()),
        ("negate", build_negate_module()),
        ("max", build_max_module()),
        ("sum", build_sum_module()),
        ("load_store", build_load_store_module()),
    ];

    for (name, module) in &modules {
        let result = translate_only(&module.functions[0], module);
        assert!(result.is_ok(), "{}: adapter translation failed: {:?}", name, result.err());
        let (lir_func, _) = result.unwrap();
        assert_eq!(lir_func.name, *name);
        assert!(!lir_func.blocks.is_empty());
    }
}

// ===========================================================================
// Adapter-level regression: type translation
// ===========================================================================

#[test]
fn test_adapter_type_translation_in_context() {
    let module = build_load_store_module();
    let (lir_func, _) = translate_only(&module.functions[0], &module).unwrap();

    // First param is Ptr -> should become I64
    assert_eq!(lir_func.signature.params[0], Type::I64);
    // Second param is I32 -> should become I32
    assert_eq!(lir_func.signature.params[1], Type::I32);
}

#[test]
fn test_adapter_block_params_for_loop() {
    let module = build_sum_module();
    let (lir_func, _) = translate_only(&module.functions[0], &module).unwrap();

    let loop_header_block = lir_func.blocks.iter()
        .find(|(_, bb)| bb.params.len() == 2)
        .expect("Should have a block with 2 params (loop header)");

    assert_eq!(loop_header_block.1.params.len(), 2);
    assert_eq!(loop_header_block.1.params[0].1, Type::I32);
    assert_eq!(loop_header_block.1.params[1].1, Type::I32);
}

// ===========================================================================
// Test: Direct function call
// ===========================================================================

fn build_call_module() -> TmirModule {
    let mut module = TmirModule::new("call_test");
    let ft = module.add_func_type(FuncTy { params: vec![Ty::I32], returns: vec![Ty::I32], is_vararg: false });

    module.add_function(TmirFunction {
        id: FuncId::new(0), name: "callee".to_string(), ty: ft, entry: b(0),
        blocks: vec![TmirBlock { id: b(0), params: vec![(v(0), Ty::I32)], body: vec![
            node(Inst::Return { values: vec![v(0)] }, vec![], vec![]),
        ]}],
        proofs: vec![],
    });
    module.add_function(TmirFunction {
        id: FuncId::new(1), name: "caller".to_string(), ty: ft, entry: b(0),
        blocks: vec![TmirBlock { id: b(0), params: vec![(v(0), Ty::I32)], body: vec![
            node(Inst::Call { callee: FuncId::new(0), args: vec![v(0)] }, vec![v(1)], vec![]),
            node(Inst::Return { values: vec![v(1)] }, vec![], vec![]),
        ]}],
        proofs: vec![],
    });
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
        "Expected Call to 'callee', got {:?}", entry.instructions[0].opcode
    );
}

#[test]
fn test_call_isel() {
    let module = build_call_module();
    let results = translate_module(&module).unwrap();

    let (caller_lir, _) = &results[1];
    let mut isel = InstructionSelector::new(caller_lir.name.clone(), caller_lir.signature.clone());
    isel.lower_formal_arguments(&caller_lir.signature, caller_lir.entry_block).unwrap();

    let mut block_order: Vec<Block> = caller_lir.blocks.keys().copied().collect();
    block_order.sort_by_key(|b| if *b == caller_lir.entry_block { 0 } else { b.0 + 1 });

    for block_id in &block_order {
        let bb = &caller_lir.blocks[block_id];
        isel.select_block(*block_id, &bb.instructions).unwrap();
    }

    let mfunc = isel.finalize();
    assert!(has_opcode(&mfunc, AArch64Opcode::Bl), "Expected BL instruction for direct call");
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test: Switch
// ===========================================================================

fn build_switch_module() -> TmirModule {
    simple_module("test", "dispatch",
        vec![Ty::I32], vec![Ty::I32],
        vec![
            TmirBlock {
                id: b(0),
                params: vec![(v(0), Ty::I32)],
                body: vec![node(Inst::Switch {
                    value: v(0),
                    default: b(3),
                    default_args: vec![],
                    cases: vec![
                        SwitchCase { value: Constant::Int(0), target: b(1), args: vec![] },
                        SwitchCase { value: Constant::Int(1), target: b(2), args: vec![] },
                    ],
                }, vec![], vec![])],
            },
            TmirBlock { id: b(1), params: vec![], body: vec![
                node(Inst::Const { ty: Ty::I32, value: Constant::Int(10) }, vec![v(1)], vec![]),
                node(Inst::Return { values: vec![v(1)] }, vec![], vec![]),
            ]},
            TmirBlock { id: b(2), params: vec![], body: vec![
                node(Inst::Const { ty: Ty::I32, value: Constant::Int(20) }, vec![v(2)], vec![]),
                node(Inst::Return { values: vec![v(2)] }, vec![], vec![]),
            ]},
            TmirBlock { id: b(3), params: vec![], body: vec![
                node(Inst::Const { ty: Ty::I32, value: Constant::Int(30) }, vec![v(3)], vec![]),
                node(Inst::Return { values: vec![v(3)] }, vec![], vec![]),
            ]},
        ],
    )
}

#[test]
fn test_switch_adapter() {
    let module = build_switch_module();
    let (lir_func, _) = translate_only(&module.functions[0], &module).unwrap();

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
    let module = build_switch_module();
    let mfunc = compile_tmir_function(&module.functions[0], &module);

    assert_eq!(mfunc.blocks.len(), 4);
    assert!(
        has_opcode(&mfunc, AArch64Opcode::CmpRR) || has_opcode(&mfunc, AArch64Opcode::CmpRI),
        "Expected CMP instruction for switch case comparison"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::BCond));
    assert!(count_opcode(&mfunc, AArch64Opcode::Ret) >= 3);
}

// ===========================================================================
// Test: CallIndirect
// ===========================================================================

fn build_call_indirect_module() -> TmirModule {
    let mut module = TmirModule::new("indirect_call");
    let callee_ft = module.add_func_type(FuncTy { params: vec![Ty::I32], returns: vec![Ty::I32], is_vararg: false });
    // The main function: takes a fn ptr and an arg
    let main_ft = module.add_func_type(FuncTy { params: vec![Ty::Ptr, Ty::I32], returns: vec![Ty::I32], is_vararg: false });
    module.add_function(TmirFunction {
        id: FuncId::new(0), name: "call_through_ptr".to_string(), ty: main_ft, entry: b(0),
        blocks: vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::Ptr), (v(1), Ty::I32)],
            body: vec![
                node(Inst::CallIndirect { callee: v(0), sig: callee_ft, args: vec![v(1)] }, vec![v(2)], vec![]),
                node(Inst::Return { values: vec![v(2)] }, vec![], vec![]),
            ],
        }],
        proofs: vec![],
    });
    module
}

#[test]
fn test_call_indirect_adapter() {
    let module = build_call_indirect_module();
    let (lir_func, _) = translate_only(&module.functions[0], &module).unwrap();

    assert_eq!(lir_func.name, "call_through_ptr");

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 2);
    assert!(
        matches!(&entry.instructions[0].opcode, Opcode::CallIndirect),
        "Expected CallIndirect opcode, got {:?}", entry.instructions[0].opcode
    );
    assert_eq!(entry.instructions[0].args.len(), 2, "CallIndirect should have fn_ptr + 1 arg = 2 total args");
}

#[test]
fn test_call_indirect_isel() {
    let module = build_call_indirect_module();
    let mfunc = compile_tmir_function(&module.functions[0], &module);

    assert!(has_opcode(&mfunc, AArch64Opcode::Blr), "Expected BLR instruction for indirect call");
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test: Select
// ===========================================================================

fn build_select_module() -> TmirModule {
    simple_module("test", "abs_select",
        vec![Ty::I32], vec![Ty::I32],
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I32)],
            body: vec![
                node(Inst::UnOp { op: UnOp::Neg, ty: Ty::I32, operand: v(0) }, vec![v(1)], vec![]),
                node(Inst::Const { ty: Ty::I32, value: Constant::Int(0) }, vec![v(2)], vec![]),
                node(Inst::ICmp { op: ICmpOp::Slt, ty: Ty::I32, lhs: v(0), rhs: v(2) }, vec![v(3)], vec![]),
                node(Inst::Select { ty: Ty::I32, cond: v(3), then_val: v(1), else_val: v(0) }, vec![v(4)], vec![]),
                node(Inst::Return { values: vec![v(4)] }, vec![], vec![]),
            ],
        }],
    )
}

#[test]
fn test_select_adapter() {
    let module = build_select_module();
    let (lir_func, _) = translate_only(&module.functions[0], &module).unwrap();

    assert_eq!(lir_func.name, "abs_select");
    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert!(entry.instructions.len() >= 4, "Expected at least 4 instructions, got {}", entry.instructions.len());

    let has_select = entry.instructions.iter().any(|inst| matches!(&inst.opcode, Opcode::Select { .. }));
    assert!(has_select, "Expected Select opcode in instruction stream");
}

#[test]
fn test_select_isel() {
    let module = build_select_module();
    let mfunc = compile_tmir_function(&module.functions[0], &module);

    assert!(has_opcode(&mfunc, AArch64Opcode::Csel), "Expected CSEL instruction for Select");
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test: GEP
// ===========================================================================

fn build_gep_module() -> TmirModule {
    simple_module("test", "array_access",
        vec![Ty::Ptr, Ty::I64], vec![Ty::I32],
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::Ptr), (v(1), Ty::I64)],
            body: vec![
                node(Inst::GEP { pointee_ty: Ty::I32, base: v(0), indices: vec![v(1)] }, vec![v(2)], vec![]),
                node(Inst::Load { ty: Ty::I32, ptr: v(2) }, vec![v(3)], vec![]),
                node(Inst::Return { values: vec![v(3)] }, vec![], vec![]),
            ],
        }],
    )
}

#[test]
fn test_gep_adapter() {
    let module = build_gep_module();
    let (lir_func, _) = translate_only(&module.functions[0], &module).unwrap();

    assert_eq!(lir_func.name, "array_access");
    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert!(entry.instructions.len() >= 4, "Expected at least 4 instructions for GEP+Load+Return, got {}", entry.instructions.len());

    let has_load = entry.instructions.iter().any(|inst| matches!(&inst.opcode, Opcode::Load { .. }));
    assert!(has_load, "Expected Load opcode after GEP");
}

#[test]
fn test_gep_isel() {
    let module = build_gep_module();
    let mfunc = compile_tmir_function(&module.functions[0], &module);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::LdrRI) || has_opcode(&mfunc, AArch64Opcode::LdrRO),
        "Expected LDR for load after GEP"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test: Type casts (SExt, ZExt, Trunc)
// ===========================================================================

fn build_cast_chain_module() -> TmirModule {
    simple_module("test", "widen_and_narrow",
        vec![Ty::I8], vec![Ty::I8],
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I8)],
            body: vec![
                node(Inst::Cast { op: CastOp::SExt, src_ty: Ty::I8, dst_ty: Ty::I32, operand: v(0) }, vec![v(1)], vec![]),
                node(Inst::Cast { op: CastOp::Trunc, src_ty: Ty::I32, dst_ty: Ty::I8, operand: v(1) }, vec![v(2)], vec![]),
                node(Inst::Return { values: vec![v(2)] }, vec![], vec![]),
            ],
        }],
    )
}

#[test]
fn test_cast_chain_adapter() {
    let module = build_cast_chain_module();
    let (lir_func, _) = translate_only(&module.functions[0], &module).unwrap();

    assert_eq!(lir_func.name, "widen_and_narrow");
    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 3);
    assert!(matches!(&entry.instructions[0].opcode, Opcode::Sextend { from_ty: Type::I8, to_ty: Type::I32 }));
    assert!(matches!(&entry.instructions[1].opcode, Opcode::Trunc { to_ty: Type::I8 }));
}

#[test]
fn test_cast_chain_isel() {
    let module = build_cast_chain_module();
    let mfunc = compile_tmir_function(&module.functions[0], &module);

    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
    assert!(total_insts(&mfunc) >= 3);
}

// ===========================================================================
// Test: Float arithmetic + FP cast
// ===========================================================================

fn build_float_to_int_module() -> TmirModule {
    simple_module("test", "float_to_int",
        vec![Ty::F64, Ty::F64], vec![Ty::I32],
        vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::F64), (v(1), Ty::F64)],
            body: vec![
                node(Inst::BinOp { op: BinOp::FAdd, ty: Ty::F64, lhs: v(0), rhs: v(1) }, vec![v(2)], vec![]),
                node(Inst::Cast { op: CastOp::FPToSI, src_ty: Ty::F64, dst_ty: Ty::I32, operand: v(2) }, vec![v(3)], vec![]),
                node(Inst::Return { values: vec![v(3)] }, vec![], vec![]),
            ],
        }],
    )
}

#[test]
fn test_float_to_int_adapter() {
    let module = build_float_to_int_module();
    let (lir_func, _) = translate_only(&module.functions[0], &module).unwrap();

    assert_eq!(lir_func.name, "float_to_int");
    assert_eq!(lir_func.signature.params, vec![Type::F64, Type::F64]);
    assert_eq!(lir_func.signature.returns, vec![Type::I32]);

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 3);
}

#[test]
fn test_float_to_int_isel() {
    let module = build_float_to_int_module();
    let mfunc = compile_tmir_function(&module.functions[0], &module);

    assert!(has_opcode(&mfunc, AArch64Opcode::FaddRR), "Expected FADD for f64 addition");
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test: Proof annotations survive adapter translation
// ===========================================================================

#[test]
fn test_proof_annotations_survive() {
    let mut module = TmirModule::new("test");
    let ft = module.add_func_type(FuncTy { params: vec![Ty::I32, Ty::I32], returns: vec![Ty::I32], is_vararg: false });
    module.add_function(TmirFunction {
        id: FuncId::new(0), name: "proven_add".to_string(), ty: ft, entry: b(0),
        blocks: vec![TmirBlock {
            id: b(0),
            params: vec![(v(0), Ty::I32), (v(1), Ty::I32)],
            body: vec![
                node(Inst::BinOp { op: BinOp::Add, ty: Ty::I32, lhs: v(0), rhs: v(1) }, vec![v(2)],
                     vec![ProofAnnotation::NoOverflow]),
                node(Inst::Return { values: vec![v(2)] }, vec![], vec![]),
            ],
        }],
        proofs: vec![ProofAnnotation::Pure],
    });

    let (_, proof_ctx) = translate_only(&module.functions[0], &module).unwrap();

    assert!(proof_ctx.is_function_pure(), "Function-level Pure proof should survive");

    let has_no_overflow = proof_ctx.value_proofs.values()
        .any(|proofs| proofs.iter().any(|p| matches!(p, llvm2_lower::adapter::Proof::NoOverflow { signed: true })));
    assert!(has_no_overflow, "NoOverflow proof should be attached to the add result");
}

// ===========================================================================
// Comprehensive: all new programs adapter succeeds
// ===========================================================================

#[test]
fn test_all_new_programs_adapter_succeeds() {
    let modules: Vec<(&str, TmirModule)> = vec![
        ("dispatch", build_switch_module()),
        ("call_through_ptr", build_call_indirect_module()),
        ("abs_select", build_select_module()),
        ("array_access", build_gep_module()),
        ("widen_and_narrow", build_cast_chain_module()),
        ("float_to_int", build_float_to_int_module()),
    ];

    for (name, module) in &modules {
        let result = translate_only(&module.functions[0], module);
        assert!(result.is_ok(), "{}: adapter translation failed: {:?}", name, result.err());
        let (lir_func, _) = result.unwrap();
        assert_eq!(lir_func.name, *name);
        assert!(!lir_func.blocks.is_empty());
    }
}

#[test]
fn test_all_new_single_block_programs_compile() {
    let modules: Vec<(&str, TmirModule)> = vec![
        ("call_through_ptr", build_call_indirect_module()),
        ("abs_select", build_select_module()),
        ("array_access", build_gep_module()),
        ("widen_and_narrow", build_cast_chain_module()),
        ("float_to_int", build_float_to_int_module()),
    ];

    for (name, module) in &modules {
        let mfunc = compile_tmir_function(&module.functions[0], module);
        assert!(!mfunc.blocks.is_empty(), "{}: produced empty ISelFunction", name);
        assert!(total_insts(&mfunc) > 0, "{}: produced no machine instructions", name);
        assert!(has_opcode(&mfunc, AArch64Opcode::Ret), "{}: missing RET instruction", name);
    }
}
