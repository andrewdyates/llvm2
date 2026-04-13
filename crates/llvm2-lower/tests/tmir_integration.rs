// tmir_integration.rs - End-to-end tMIR -> adapter -> ISel integration tests
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Tests 6 tMIR programs through the full adapter->ISel pipeline:
//   1. identity(x: i32) -> i32   — simplest possible function
//   2. add(a: i32, b: i32) -> i32 — arithmetic
//   3. negate(x: i32) -> i32      — unary op
//   4. max(a: i32, b: i32) -> i32 — conditional (if/else)
//   5. sum(n: i32) -> i32         — loop with accumulator
//   6. load_store(p: *mut i32, v: i32) — memory operations

use llvm2_lower::adapter::{translate_function, AdapterError, ProofContext};
use llvm2_lower::function::Function;
use llvm2_lower::isel::{AArch64Opcode, InstructionSelector, MachFunction};
use llvm2_lower::instructions::Block;
use llvm2_lower::types::Type;

use tmir_func::{Block as TmirBlock, Function as TmirFunction};
use tmir_instrs::{BinOp, CmpOp, Instr, InstrNode, UnOp};
use tmir_types::{BlockId, FuncId, FuncTy, Ty, ValueId};

// ---------------------------------------------------------------------------
// Helper: run a tMIR function through adapter + ISel
// ---------------------------------------------------------------------------

/// Translate a tMIR function through the adapter, then run ISel on all blocks.
///
/// Returns the completed MachFunction. Panics on adapter errors; the ISel
/// does not return errors (it panics on undefined values, which would indicate
/// a bug in the adapter or the test program).
fn compile_tmir_function(func: &TmirFunction) -> MachFunction {
    // Step 1: adapter — tMIR -> internal LIR
    let (lir_func, _proof_ctx) =
        translate_function(func, &[]).expect("adapter translation failed");

    // Step 2: ISel — internal LIR -> AArch64 MachInst
    let mut isel = InstructionSelector::new(lir_func.name.clone(), lir_func.signature.clone());

    // Lower formal arguments into the entry block (establishes Value -> VReg mapping
    // for function parameters).
    isel.lower_formal_arguments(&lir_func.signature, lir_func.entry_block).unwrap();

    // Select each block. We need a deterministic order: entry block first, then
    // the rest. The ISel requires that values are defined before use, and block
    // params are established by the copy instructions emitted by the adapter.
    let mut block_order: Vec<Block> = lir_func.blocks.keys().copied().collect();
    block_order.sort_by_key(|b| {
        if *b == lir_func.entry_block {
            0
        } else {
            b.0 + 1
        }
    });

    for block_id in &block_order {
        let bb = &lir_func.blocks[block_id];
        isel.select_block(*block_id, &bb.instructions).unwrap();
    }

    isel.finalize()
}

/// Translate only through the adapter (for cases where ISel fails).
fn translate_only(func: &TmirFunction) -> Result<(Function, ProofContext), AdapterError> {
    translate_function(func, &[])
}

/// Count occurrences of a specific opcode in a MachFunction.
fn count_opcode(mfunc: &MachFunction, opcode: AArch64Opcode) -> usize {
    mfunc
        .blocks
        .values()
        .flat_map(|b| &b.insts)
        .filter(|inst| inst.opcode == opcode)
        .count()
}

/// Check that a MachFunction contains at least one instance of the given opcode.
fn has_opcode(mfunc: &MachFunction, opcode: AArch64Opcode) -> bool {
    count_opcode(mfunc, opcode) > 0
}

/// Get total instruction count across all blocks.
fn total_insts(mfunc: &MachFunction) -> usize {
    mfunc.blocks.values().map(|b| b.insts.len()).sum()
}

// ===========================================================================
// Test 1: identity(x: i32) -> i32 { x }
// ===========================================================================
//
// Simplest possible function: one param, return it.
// tMIR:
//   block0(v0: i32):
//     return v0

fn build_identity() -> TmirFunction {
    TmirFunction {
        id: FuncId(0),
        name: "identity".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(32)],
            returns: vec![Ty::Int(32)],
        },
        entry: BlockId(0),
        blocks: vec![TmirBlock {
            id: BlockId(0),
            params: vec![(ValueId(0), Ty::Int(32))],
            body: vec![InstrNode {
                instr: Instr::Return {
                    values: vec![ValueId(0)],
                },
                results: vec![],
            }],
        }],
    }
}

#[test]
fn test_identity_adapter() {
    let func = build_identity();
    let (lir_func, proof_ctx) = translate_only(&func).unwrap();

    assert_eq!(lir_func.name, "identity");
    assert_eq!(lir_func.signature.params, vec![Type::I32]);
    assert_eq!(lir_func.signature.returns, vec![Type::I32]);
    assert_eq!(lir_func.blocks.len(), 1);

    // Entry block: 1 param, 1 instruction (Return)
    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.params.len(), 1);
    assert_eq!(entry.instructions.len(), 1);

    // No proofs expected from stubs
    assert!(proof_ctx.value_proofs.is_empty());
}

#[test]
fn test_identity_isel() {
    let func = build_identity();
    let mfunc = compile_tmir_function(&func);

    assert_eq!(mfunc.name, "identity");
    assert!(!mfunc.blocks.is_empty());

    // Should have COPY instructions for the formal arg, plus a RET.
    assert!(has_opcode(&mfunc, AArch64Opcode::COPY));
    assert!(has_opcode(&mfunc, AArch64Opcode::RET));
}

// ===========================================================================
// Test 2: add(a: i32, b: i32) -> i32 { a + b }
// ===========================================================================
//
// tMIR:
//   block0(v0: i32, v1: i32):
//     v2 = BinOp::Add(i32, v0, v1)
//     return v2

fn build_add() -> TmirFunction {
    TmirFunction {
        id: FuncId(1),
        name: "add".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(32), Ty::Int(32)],
            returns: vec![Ty::Int(32)],
        },
        entry: BlockId(0),
        blocks: vec![TmirBlock {
            id: BlockId(0),
            params: vec![(ValueId(0), Ty::Int(32)), (ValueId(1), Ty::Int(32))],
            body: vec![
                InstrNode {
                    instr: Instr::BinOp {
                        op: BinOp::Add,
                        ty: Ty::Int(32),
                        lhs: ValueId(0),
                        rhs: ValueId(1),
                    },
                    results: vec![ValueId(2)],
                },
                InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(2)],
                    },
                    results: vec![],
                },
            ],
        }],
    }
}

#[test]
fn test_add_adapter() {
    let func = build_add();
    let (lir_func, _) = translate_only(&func).unwrap();

    assert_eq!(lir_func.name, "add");
    assert_eq!(lir_func.signature.params, vec![Type::I32, Type::I32]);

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.params.len(), 2);
    // Iadd + Return = 2 instructions
    assert_eq!(entry.instructions.len(), 2);
}

#[test]
fn test_add_isel() {
    let func = build_add();
    let mfunc = compile_tmir_function(&func);

    assert_eq!(mfunc.name, "add");
    // Must contain ADDWrr (32-bit integer add, register form)
    assert!(
        has_opcode(&mfunc, AArch64Opcode::ADDWrr),
        "Expected ADDWrr for i32 addition"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::RET));
}

// ===========================================================================
// Test 3: negate(x: i32) -> i32 { -x }
// ===========================================================================
//
// Unary negation. The adapter lowers this as: const 0; sub(0, x).
// tMIR:
//   block0(v0: i32):
//     v1 = UnOp::Neg(i32, v0)
//     return v1

fn build_negate() -> TmirFunction {
    TmirFunction {
        id: FuncId(2),
        name: "negate".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(32)],
            returns: vec![Ty::Int(32)],
        },
        entry: BlockId(0),
        blocks: vec![TmirBlock {
            id: BlockId(0),
            params: vec![(ValueId(0), Ty::Int(32))],
            body: vec![
                InstrNode {
                    instr: Instr::UnOp {
                        op: UnOp::Neg,
                        ty: Ty::Int(32),
                        operand: ValueId(0),
                    },
                    results: vec![ValueId(1)],
                },
                InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(1)],
                    },
                    results: vec![],
                },
            ],
        }],
    }
}

#[test]
fn test_negate_adapter() {
    let func = build_negate();
    let (lir_func, _) = translate_only(&func).unwrap();

    let entry = &lir_func.blocks[&lir_func.entry_block];
    // Neg is lowered directly as: Ineg, Return = 2 instructions
    assert_eq!(entry.instructions.len(), 2);
}

#[test]
fn test_negate_isel() {
    let func = build_negate();
    let mfunc = compile_tmir_function(&func);

    // Negation is lowered directly as NEGWr (SUB Wd, WZR, Wn alias).
    assert!(
        has_opcode(&mfunc, AArch64Opcode::NEGWr),
        "Expected NEGWr for negation (-x)"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::RET));
}

// ===========================================================================
// Test 4: max(a: i32, b: i32) -> i32 { if a > b { a } else { b } }
// ===========================================================================
//
// Conditional with 3 blocks: entry (compare + branch), then, else.
// tMIR:
//   block0(v0: i32, v1: i32):
//     v2 = Cmp::Sgt(i32, v0, v1)
//     condbr v2, block1(), block2()
//   block1:
//     return v0
//   block2:
//     return v1

fn build_max() -> TmirFunction {
    TmirFunction {
        id: FuncId(3),
        name: "max".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(32), Ty::Int(32)],
            returns: vec![Ty::Int(32)],
        },
        entry: BlockId(0),
        blocks: vec![
            TmirBlock {
                id: BlockId(0),
                params: vec![(ValueId(0), Ty::Int(32)), (ValueId(1), Ty::Int(32))],
                body: vec![
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Sgt,
                            ty: Ty::Int(32),
                            lhs: ValueId(0),
                            rhs: ValueId(1),
                        },
                        results: vec![ValueId(2)],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: ValueId(2),
                            then_target: BlockId(1),
                            then_args: vec![],
                            else_target: BlockId(2),
                            else_args: vec![],
                        },
                        results: vec![],
                    },
                ],
            },
            TmirBlock {
                id: BlockId(1),
                params: vec![],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(0)],
                    },
                    results: vec![],
                }],
            },
            TmirBlock {
                id: BlockId(2),
                params: vec![],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(1)],
                    },
                    results: vec![],
                }],
            },
        ],
    }
}

#[test]
fn test_max_adapter() {
    let func = build_max();
    let (lir_func, _) = translate_only(&func).unwrap();

    assert_eq!(lir_func.blocks.len(), 3);

    let entry = &lir_func.blocks[&lir_func.entry_block];
    // Icmp + Brif = 2 instructions
    assert_eq!(entry.instructions.len(), 2);
}

#[test]
fn test_max_isel() {
    let func = build_max();
    let mfunc = compile_tmir_function(&func);

    assert_eq!(mfunc.blocks.len(), 3);

    // Should have a compare (CMPWrr), conditional set (CSETWcc), conditional
    // branch (Bcc), and returns (RET).
    assert!(
        has_opcode(&mfunc, AArch64Opcode::CMPWrr),
        "Expected CMPWrr for signed comparison"
    );
    assert!(
        has_opcode(&mfunc, AArch64Opcode::Bcc),
        "Expected Bcc for conditional branch"
    );
    assert!(
        count_opcode(&mfunc, AArch64Opcode::RET) >= 2,
        "Expected at least 2 RET instructions (then + else)"
    );
}

// ===========================================================================
// Test 5: sum(n: i32) -> i32 { let mut s = 0; while n > 0 { s += n; n -= 1; } s }
// ===========================================================================
//
// Loop with accumulator. Uses block parameters for loop-carried values.
// tMIR:
//   block0(v0: i32):              // entry: n = param
//     v1 = Const(i32, 0)          // s = 0
//     br block1(v0, v1)           // goto header with (n, s)
//   block1(v2: i32, v3: i32):    // loop header: (n_cur, s_cur)
//     v4 = Const(i32, 0)
//     v5 = Cmp::Sgt(i32, v2, v4) // n_cur > 0?
//     condbr v5, block2(), block3()
//   block2:                       // loop body
//     v6 = BinOp::Add(i32, v3, v2) // s_new = s_cur + n_cur
//     v7 = Const(i32, 1)
//     v8 = BinOp::Sub(i32, v2, v7) // n_new = n_cur - 1
//     br block1(v8, v6)           // back-edge
//   block3:                       // exit
//     return v3

fn build_sum() -> TmirFunction {
    TmirFunction {
        id: FuncId(4),
        name: "sum".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(32)],
            returns: vec![Ty::Int(32)],
        },
        entry: BlockId(0),
        blocks: vec![
            // block0: entry
            TmirBlock {
                id: BlockId(0),
                params: vec![(ValueId(0), Ty::Int(32))],
                body: vec![
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(32),
                            value: 0,
                        },
                        results: vec![ValueId(1)],
                    },
                    InstrNode {
                        instr: Instr::Br {
                            target: BlockId(1),
                            args: vec![ValueId(0), ValueId(1)],
                        },
                        results: vec![],
                    },
                ],
            },
            // block1: loop header
            TmirBlock {
                id: BlockId(1),
                params: vec![(ValueId(2), Ty::Int(32)), (ValueId(3), Ty::Int(32))],
                body: vec![
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(32),
                            value: 0,
                        },
                        results: vec![ValueId(4)],
                    },
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Sgt,
                            ty: Ty::Int(32),
                            lhs: ValueId(2),
                            rhs: ValueId(4),
                        },
                        results: vec![ValueId(5)],
                    },
                    InstrNode {
                        instr: Instr::CondBr {
                            cond: ValueId(5),
                            then_target: BlockId(2),
                            then_args: vec![],
                            else_target: BlockId(3),
                            else_args: vec![],
                        },
                        results: vec![],
                    },
                ],
            },
            // block2: loop body
            TmirBlock {
                id: BlockId(2),
                params: vec![],
                body: vec![
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Add,
                            ty: Ty::Int(32),
                            lhs: ValueId(3),
                            rhs: ValueId(2),
                        },
                        results: vec![ValueId(6)],
                    },
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(32),
                            value: 1,
                        },
                        results: vec![ValueId(7)],
                    },
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Sub,
                            ty: Ty::Int(32),
                            lhs: ValueId(2),
                            rhs: ValueId(7),
                        },
                        results: vec![ValueId(8)],
                    },
                    InstrNode {
                        instr: Instr::Br {
                            target: BlockId(1),
                            args: vec![ValueId(8), ValueId(6)],
                        },
                        results: vec![],
                    },
                ],
            },
            // block3: exit
            TmirBlock {
                id: BlockId(3),
                params: vec![],
                body: vec![InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(3)],
                    },
                    results: vec![],
                }],
            },
        ],
    }
}

#[test]
fn test_sum_adapter() {
    let func = build_sum();
    let (lir_func, _) = translate_only(&func).unwrap();

    assert_eq!(lir_func.name, "sum");
    assert_eq!(lir_func.blocks.len(), 4);

    // block0 (entry): Iconst(0) + copy + copy + Jump = 4 instructions
    // (adapter emits copies for block args before the jump)
    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert!(
        entry.instructions.len() >= 2,
        "Entry should have at least Iconst + Jump, got {}",
        entry.instructions.len()
    );
}

/// The sum function's ISel currently fails because the adapter emits single-arg
/// Iadd instructions as pseudo-COPY for block argument passing (see
/// adapter.rs:resolve_block_args), but the ISel expects Iadd to have exactly
/// 2 operands. This is a known adapter/ISel incompatibility: the adapter
/// uses Iadd as a placeholder for a COPY pseudo-opcode that doesn't exist
/// in the LIR opcode set yet.
///
/// BLOCKER: adapter needs a dedicated Copy opcode in the LIR, or the ISel
/// needs to handle single-arg Iadd as a register move.
///
/// We test the loop body blocks independently through the ISel by manually
/// constructing the LIR instructions that would appear in a loop body,
/// avoiding the block-arg copy issue.
#[test]
fn test_sum_isel_body_instructions() {
    // Test that the key loop-body instructions (add, sub, cmp, const)
    // select correctly through the ISel, independent of block-arg copies.
    use llvm2_lower::function::Signature;
    use llvm2_lower::instructions::{Instruction, Opcode, Value};

    let sig = Signature {
        params: vec![Type::I32, Type::I32],
        returns: vec![Type::I32],
    };
    let mut isel = InstructionSelector::new("sum_body".to_string(), sig.clone());
    let entry = Block(0);
    isel.lower_formal_arguments(&sig, entry).unwrap();

    // Simulate: v2 = add(v0, v1)  (s += n)
    isel.select_block(
        entry,
        &[
            Instruction {
                opcode: Opcode::Iadd,
                args: vec![Value(0), Value(1)],
                results: vec![Value(2)],
            },
            // v3 = const 1
            Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I32,
                    imm: 1,
                },
                args: vec![],
                results: vec![Value(3)],
            },
            // v4 = sub(v1, v3)  (n -= 1)
            Instruction {
                opcode: Opcode::Isub,
                args: vec![Value(1), Value(3)],
                results: vec![Value(4)],
            },
            // v5 = const 0
            Instruction {
                opcode: Opcode::Iconst {
                    ty: Type::I32,
                    imm: 0,
                },
                args: vec![],
                results: vec![Value(5)],
            },
            // v6 = cmp sgt(v4, v5)  (n > 0)
            Instruction {
                opcode: Opcode::Icmp {
                    cond: llvm2_lower::instructions::IntCC::SignedGreaterThan,
                },
                args: vec![Value(4), Value(5)],
                results: vec![Value(6)],
            },
            // return v2
            Instruction {
                opcode: Opcode::Return,
                args: vec![Value(2)],
                results: vec![],
            },
        ],
    ).unwrap();

    let mfunc = isel.finalize();
    let mblock = &mfunc.blocks[&entry];

    // Should contain: 2 COPYs (formal args) + ADDWrr + MOVZWi + SUBWrr
    // + MOVZWi + CMPWrr + CSETWcc + (RET logic)
    assert!(has_opcode(&mfunc, AArch64Opcode::ADDWrr));
    assert!(has_opcode(&mfunc, AArch64Opcode::SUBWrr));
    assert!(has_opcode(&mfunc, AArch64Opcode::CMPWrr));
    assert!(has_opcode(&mfunc, AArch64Opcode::RET));
    assert!(mblock.insts.len() >= 6, "Expected at least 6 instructions");
}

// ===========================================================================
// Test 6: load_store(p: *mut i32, v: i32) { *p = v; }
// ===========================================================================
//
// Memory operations: store value through pointer.
// tMIR:
//   block0(v0: ptr(i32), v1: i32):
//     store(i32, v0, v1)
//     return

fn build_load_store() -> TmirFunction {
    TmirFunction {
        id: FuncId(5),
        name: "load_store".to_string(),
        ty: FuncTy {
            params: vec![Ty::Ptr(Box::new(Ty::Int(32))), Ty::Int(32)],
            returns: vec![],
        },
        entry: BlockId(0),
        blocks: vec![TmirBlock {
            id: BlockId(0),
            params: vec![
                (ValueId(0), Ty::Ptr(Box::new(Ty::Int(32)))),
                (ValueId(1), Ty::Int(32)),
            ],
            body: vec![
                InstrNode {
                    instr: Instr::Store {
                        ty: Ty::Int(32),
                        ptr: ValueId(0),
                        value: ValueId(1),
                    },
                    results: vec![],
                },
                InstrNode {
                    instr: Instr::Return { values: vec![] },
                    results: vec![],
                },
            ],
        }],
    }
}

#[test]
fn test_load_store_adapter() {
    let func = build_load_store();
    let (lir_func, _) = translate_only(&func).unwrap();

    assert_eq!(lir_func.name, "load_store");
    // Ptr maps to I64 in the adapter
    assert_eq!(lir_func.signature.params, vec![Type::I64, Type::I32]);
    assert_eq!(lir_func.signature.returns, vec![]);

    let entry = &lir_func.blocks[&lir_func.entry_block];
    // Store + Return = 2 instructions
    assert_eq!(entry.instructions.len(), 2);
}

#[test]
fn test_load_store_isel() {
    let func = build_load_store();
    let mfunc = compile_tmir_function(&func);

    // Should have a STRWui (store 32-bit, unsigned offset) and RET
    assert!(
        has_opcode(&mfunc, AArch64Opcode::STRWui),
        "Expected STRWui for 32-bit store"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::RET));
}

// ===========================================================================
// Extended test: load then store (read-modify pattern)
// ===========================================================================

fn build_load_then_store() -> TmirFunction {
    TmirFunction {
        id: FuncId(6),
        name: "load_then_store".to_string(),
        ty: FuncTy {
            params: vec![Ty::Ptr(Box::new(Ty::Int(32)))],
            returns: vec![],
        },
        entry: BlockId(0),
        blocks: vec![TmirBlock {
            id: BlockId(0),
            params: vec![(ValueId(0), Ty::Ptr(Box::new(Ty::Int(32))))],
            body: vec![
                InstrNode {
                    instr: Instr::Load {
                        ty: Ty::Int(32),
                        ptr: ValueId(0),
                    },
                    results: vec![ValueId(1)],
                },
                InstrNode {
                    instr: Instr::Store {
                        ty: Ty::Int(32),
                        ptr: ValueId(0),
                        value: ValueId(1),
                    },
                    results: vec![],
                },
                InstrNode {
                    instr: Instr::Return { values: vec![] },
                    results: vec![],
                },
            ],
        }],
    }
}

#[test]
fn test_load_then_store_isel() {
    let func = build_load_then_store();
    let mfunc = compile_tmir_function(&func);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::LDRWui),
        "Expected LDRWui for 32-bit load"
    );
    assert!(
        has_opcode(&mfunc, AArch64Opcode::STRWui),
        "Expected STRWui for 32-bit store"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::RET));
}

// ===========================================================================
// End-to-end pipeline test: all 6 programs compile without panic
// ===========================================================================

#[test]
fn test_all_single_block_programs_compile_without_panic() {
    // All single-block programs go through the full adapter->ISel pipeline.
    // The sum function (loop) is excluded because it uses block arguments
    // which produce single-arg Iadd (copy placeholder) that the ISel does
    // not handle yet — see test_sum_isel_body_instructions and
    // test_sum_adapter for coverage of the sum function.
    let programs: Vec<(&str, TmirFunction)> = vec![
        ("identity", build_identity()),
        ("add", build_add()),
        ("negate", build_negate()),
        ("max", build_max()),
        ("load_store", build_load_store()),
    ];

    for (name, func) in &programs {
        let mfunc = compile_tmir_function(func);
        assert!(
            !mfunc.blocks.is_empty(),
            "{}: produced empty MachFunction",
            name
        );
        assert!(
            total_insts(&mfunc) > 0,
            "{}: produced no machine instructions",
            name
        );

        // Every function must end with at least one RET
        assert!(
            has_opcode(&mfunc, AArch64Opcode::RET),
            "{}: missing RET instruction",
            name
        );

        // Every function must have COPY instructions for formal arguments
        // (except void-returning functions with no params, but all our test
        // functions have at least one param).
        assert!(
            has_opcode(&mfunc, AArch64Opcode::COPY),
            "{}: missing COPY for formal arguments",
            name
        );
    }
}

/// Verify all 6 programs translate through the adapter without errors.
#[test]
fn test_all_programs_adapter_succeeds() {
    let programs: Vec<(&str, TmirFunction)> = vec![
        ("identity", build_identity()),
        ("add", build_add()),
        ("negate", build_negate()),
        ("max", build_max()),
        ("sum", build_sum()),
        ("load_store", build_load_store()),
    ];

    for (name, func) in &programs {
        let result = translate_only(func);
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
    // Verify that pointer types in function signatures translate correctly
    // (Ptr -> I64 on AArch64)
    let func = build_load_store();
    let (lir_func, _) = translate_only(&func).unwrap();

    // First param is Ptr(Int(32)) -> should become I64
    assert_eq!(lir_func.signature.params[0], Type::I64);
    // Second param is Int(32) -> should become I32
    assert_eq!(lir_func.signature.params[1], Type::I32);
}

// ===========================================================================
// Adapter-level: verify block parameter passing
// ===========================================================================

#[test]
fn test_adapter_block_params_for_loop() {
    let func = build_sum();
    let (lir_func, _) = translate_only(&func).unwrap();

    // The loop header (block1) should have 2 parameters (n_cur, s_cur)
    // Find block1 by checking params
    let loop_header_block = lir_func
        .blocks
        .iter()
        .find(|(_, bb)| bb.params.len() == 2)
        .expect("Should have a block with 2 params (loop header)");

    assert_eq!(loop_header_block.1.params.len(), 2);
    // Both params should be I32
    assert_eq!(loop_header_block.1.params[0].1, Type::I32);
    assert_eq!(loop_header_block.1.params[1].1, Type::I32);
}
