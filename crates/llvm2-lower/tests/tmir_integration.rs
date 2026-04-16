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

use llvm2_lower::adapter::{translate_function, translate_module, AdapterError, ProofContext};
use llvm2_lower::function::Function;
use llvm2_lower::isel::{AArch64Opcode, InstructionSelector, ISelFunction};
use llvm2_lower::instructions::{Block, Opcode};
use llvm2_lower::types::Type;

use tmir_func::{Block as TmirBlock, Function as TmirFunction, Module as TmirModule};
use tmir_func::builder::{self, FunctionBuilder};
use tmir_instrs::{BinOp, CastOp, CmpOp, Instr, InstrNode, SwitchCase, UnOp};
use tmir_types::{BlockId, FuncId, FuncTy, Ty, ValueId};

// ---------------------------------------------------------------------------
// Helper: run a tMIR function through adapter + ISel
// ---------------------------------------------------------------------------

/// Translate a tMIR function through the adapter, then run ISel on all blocks.
///
/// Returns the completed ISelFunction. Panics on adapter errors; the ISel
/// does not return errors (it panics on undefined values, which would indicate
/// a bug in the adapter or the test program).
fn compile_tmir_function(func: &TmirFunction) -> ISelFunction {
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

/// Count occurrences of a specific opcode in a ISelFunction.
fn count_opcode(mfunc: &ISelFunction, opcode: AArch64Opcode) -> usize {
    mfunc
        .blocks
        .values()
        .flat_map(|b| &b.insts)
        .filter(|inst| inst.opcode == opcode)
        .count()
}

/// Check that a ISelFunction contains at least one instance of the given opcode.
fn has_opcode(mfunc: &ISelFunction, opcode: AArch64Opcode) -> bool {
    count_opcode(mfunc, opcode) > 0
}

/// Get total instruction count across all blocks.
fn total_insts(mfunc: &ISelFunction) -> usize {
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
                proofs: vec![],
            }],
        }],
        proofs: vec![],
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
    assert!(has_opcode(&mfunc, AArch64Opcode::Copy));
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
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
        has_opcode(&mfunc, AArch64Opcode::AddRR),
        "Expected ADDWrr for i32 addition"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
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
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(1)],
                    },
                    results: vec![],
                    proofs: vec![],
                },
            ],
        }],
        proofs: vec![],
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
        has_opcode(&mfunc, AArch64Opcode::Neg),
        "Expected NEGWr for negation (-x)"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
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
                        proofs: vec![],
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
                        proofs: vec![],
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
                    proofs: vec![],
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
                    proofs: vec![],
                }],
            },
        ],
        proofs: vec![],
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
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Br {
                            target: BlockId(1),
                            args: vec![ValueId(0), ValueId(1)],
                        },
                        results: vec![],
                        proofs: vec![],
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
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Cmp {
                            op: CmpOp::Sgt,
                            ty: Ty::Int(32),
                            lhs: ValueId(2),
                            rhs: ValueId(4),
                        },
                        results: vec![ValueId(5)],
                        proofs: vec![],
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
                        proofs: vec![],
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
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(32),
                            value: 1,
                        },
                        results: vec![ValueId(7)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::BinOp {
                            op: BinOp::Sub,
                            ty: Ty::Int(32),
                            lhs: ValueId(2),
                            rhs: ValueId(7),
                        },
                        results: vec![ValueId(8)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Br {
                            target: BlockId(1),
                            args: vec![ValueId(8), ValueId(6)],
                        },
                        results: vec![],
                        proofs: vec![],
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
                    proofs: vec![],
                }],
            },
        ],
        proofs: vec![],
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
    assert!(has_opcode(&mfunc, AArch64Opcode::AddRR));
    assert!(has_opcode(&mfunc, AArch64Opcode::SubRR));
    assert!(has_opcode(&mfunc, AArch64Opcode::CmpRR));
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
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
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Return { values: vec![] },
                    results: vec![],
                    proofs: vec![],
                },
            ],
        }],
        proofs: vec![],
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
        has_opcode(&mfunc, AArch64Opcode::StrRI),
        "Expected STRWui for 32-bit store"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
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
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Store {
                        ty: Ty::Int(32),
                        ptr: ValueId(0),
                        value: ValueId(1),
                    },
                    results: vec![],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Return { values: vec![] },
                    results: vec![],
                    proofs: vec![],
                },
            ],
        }],
        proofs: vec![],
    }
}

#[test]
fn test_load_then_store_isel() {
    let func = build_load_then_store();
    let mfunc = compile_tmir_function(&func);

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
            "{}: produced empty ISelFunction",
            name
        );
        assert!(
            total_insts(&mfunc) > 0,
            "{}: produced no machine instructions",
            name
        );

        // Every function must end with at least one RET
        assert!(
            has_opcode(&mfunc, AArch64Opcode::Ret),
            "{}: missing RET instruction",
            name
        );

        // Every function must have COPY instructions for formal arguments
        // (except void-returning functions with no params, but all our test
        // functions have at least one param).
        assert!(
            has_opcode(&mfunc, AArch64Opcode::Copy),
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

// ===========================================================================
// Test 7: Direct function call
// ===========================================================================
//
// fn callee(x: i32) -> i32 { x }
// fn caller(a: i32) -> i32 { callee(a) }
//
// Tests direct call lowering through the adapter and ISel pipeline.

fn build_call_module() -> TmirModule {
    let callee = TmirFunction {
        id: FuncId(0),
        name: "callee".to_string(),
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
                proofs: vec![],
            }],
        }],
        proofs: vec![],
    };

    let caller = TmirFunction {
        id: FuncId(1),
        name: "caller".to_string(),
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
                    instr: Instr::Call {
                        func: FuncId(0),
                        args: vec![ValueId(0)],
                        ret_ty: vec![Ty::Int(32)],
                    },
                    results: vec![ValueId(1)],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(1)],
                    },
                    results: vec![],
                    proofs: vec![],
                },
            ],
        }],
        proofs: vec![],
    };

    TmirModule {
        name: "call_test".to_string(),
        functions: vec![callee, caller],
        structs: vec![],
    }
}

#[test]
fn test_call_adapter() {
    let module = build_call_module();
    let results = translate_module(&module).unwrap();
    assert_eq!(results.len(), 2);

    // callee function
    let (callee_func, _) = &results[0];
    assert_eq!(callee_func.name, "callee");

    // caller function
    let (caller_func, _) = &results[1];
    assert_eq!(caller_func.name, "caller");

    let entry = &caller_func.blocks[&caller_func.entry_block];
    // Should have a Call instruction followed by Return
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

    // Compile the caller function through ISel
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
// Test 8: Switch (multi-way branch)
// ===========================================================================
//
// fn dispatch(x: i32) -> i32 {
//     match x {
//         0 => 10,
//         1 => 20,
//         _ => 30,
//     }
// }

fn build_switch() -> TmirFunction {
    TmirFunction {
        id: FuncId(10),
        name: "dispatch".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(32)],
            returns: vec![Ty::Int(32)],
        },
        entry: BlockId(0),
        blocks: vec![
            // block0: entry — switch on x
            TmirBlock {
                id: BlockId(0),
                params: vec![(ValueId(0), Ty::Int(32))],
                body: vec![InstrNode {
                    instr: Instr::Switch {
                        value: ValueId(0),
                        cases: vec![
                            SwitchCase {
                                value: 0,
                                target: BlockId(1),
                            },
                            SwitchCase {
                                value: 1,
                                target: BlockId(2),
                            },
                        ],
                        default: BlockId(3),
                    },
                    results: vec![],
                    proofs: vec![],
                }],
            },
            // block1: case 0 -> return 10
            TmirBlock {
                id: BlockId(1),
                params: vec![],
                body: vec![
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(32),
                            value: 10,
                        },
                        results: vec![ValueId(1)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Return {
                            values: vec![ValueId(1)],
                        },
                        results: vec![],
                        proofs: vec![],
                    },
                ],
            },
            // block2: case 1 -> return 20
            TmirBlock {
                id: BlockId(2),
                params: vec![],
                body: vec![
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(32),
                            value: 20,
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
            },
            // block3: default -> return 30
            TmirBlock {
                id: BlockId(3),
                params: vec![],
                body: vec![
                    InstrNode {
                        instr: Instr::Const {
                            ty: Ty::Int(32),
                            value: 30,
                        },
                        results: vec![ValueId(3)],
                        proofs: vec![],
                    },
                    InstrNode {
                        instr: Instr::Return {
                            values: vec![ValueId(3)],
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

#[test]
fn test_switch_adapter() {
    let func = build_switch();
    let (lir_func, _) = translate_only(&func).unwrap();

    assert_eq!(lir_func.name, "dispatch");
    assert_eq!(lir_func.blocks.len(), 4);

    // Entry block should have exactly 1 instruction: Switch
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
    let func = build_switch();
    let mfunc = compile_tmir_function(&func);

    assert_eq!(mfunc.blocks.len(), 4);

    // Switch is lowered as a cascading CMP+B.EQ chain.
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
// Test 9: CallIndirect (indirect function call)
// ===========================================================================
//
// fn call_through_ptr(fn_ptr: *fn(i32) -> i32, arg: i32) -> i32 {
//     fn_ptr(arg)
// }

fn build_call_indirect() -> TmirFunction {
    TmirFunction {
        id: FuncId(11),
        name: "call_through_ptr".to_string(),
        ty: FuncTy {
            params: vec![
                Ty::Ptr(Box::new(Ty::Func(FuncTy {
                    params: vec![Ty::Int(32)],
                    returns: vec![Ty::Int(32)],
                }))),
                Ty::Int(32),
            ],
            returns: vec![Ty::Int(32)],
        },
        entry: BlockId(0),
        blocks: vec![TmirBlock {
            id: BlockId(0),
            params: vec![
                (
                    ValueId(0),
                    Ty::Ptr(Box::new(Ty::Func(FuncTy {
                        params: vec![Ty::Int(32)],
                        returns: vec![Ty::Int(32)],
                    }))),
                ),
                (ValueId(1), Ty::Int(32)),
            ],
            body: vec![
                InstrNode {
                    instr: Instr::CallIndirect {
                        callee: ValueId(0),
                        args: vec![ValueId(1)],
                        ret_ty: vec![Ty::Int(32)],
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

#[test]
fn test_call_indirect_adapter() {
    let func = build_call_indirect();
    let (lir_func, _) = translate_only(&func).unwrap();

    assert_eq!(lir_func.name, "call_through_ptr");

    let entry = &lir_func.blocks[&lir_func.entry_block];
    // CallIndirect + Return = 2 instructions
    assert_eq!(entry.instructions.len(), 2);
    assert!(
        matches!(&entry.instructions[0].opcode, Opcode::CallIndirect),
        "Expected CallIndirect opcode, got {:?}",
        entry.instructions[0].opcode
    );
    // First arg should be the function pointer, second the call arg
    assert_eq!(
        entry.instructions[0].args.len(),
        2,
        "CallIndirect should have fn_ptr + 1 arg = 2 total args"
    );
}

#[test]
fn test_call_indirect_isel() {
    let func = build_call_indirect();
    let mfunc = compile_tmir_function(&func);

    // Indirect call is lowered to BLR on AArch64
    assert!(
        has_opcode(&mfunc, AArch64Opcode::Blr),
        "Expected BLR instruction for indirect call"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test 10: Select (branchless conditional value)
// ===========================================================================
//
// fn abs(x: i32) -> i32 {
//     let neg = -x;
//     let is_neg = x < 0;
//     select(is_neg, neg, x)
// }

fn build_select() -> TmirFunction {
    TmirFunction {
        id: FuncId(12),
        name: "abs_select".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(32)],
            returns: vec![Ty::Int(32)],
        },
        entry: BlockId(0),
        blocks: vec![TmirBlock {
            id: BlockId(0),
            params: vec![(ValueId(0), Ty::Int(32))],
            body: vec![
                // v1 = -x
                InstrNode {
                    instr: Instr::UnOp {
                        op: UnOp::Neg,
                        ty: Ty::Int(32),
                        operand: ValueId(0),
                    },
                    results: vec![ValueId(1)],
                    proofs: vec![],
                },
                // v2 = const 0
                InstrNode {
                    instr: Instr::Const {
                        ty: Ty::Int(32),
                        value: 0,
                    },
                    results: vec![ValueId(2)],
                    proofs: vec![],
                },
                // v3 = x < 0
                InstrNode {
                    instr: Instr::Cmp {
                        op: CmpOp::Slt,
                        ty: Ty::Int(32),
                        lhs: ValueId(0),
                        rhs: ValueId(2),
                    },
                    results: vec![ValueId(3)],
                    proofs: vec![],
                },
                // v4 = select(v3, v1, v0)  — if x<0 then -x else x
                InstrNode {
                    instr: Instr::Select {
                        ty: Ty::Int(32),
                        cond: ValueId(3),
                        true_val: ValueId(1),
                        false_val: ValueId(0),
                    },
                    results: vec![ValueId(4)],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(4)],
                    },
                    results: vec![],
                    proofs: vec![],
                },
            ],
        }],
        proofs: vec![],
    }
}

#[test]
fn test_select_adapter() {
    let func = build_select();
    let (lir_func, _) = translate_only(&func).unwrap();

    assert_eq!(lir_func.name, "abs_select");
    let entry = &lir_func.blocks[&lir_func.entry_block];

    // Should have: Ineg, Iconst(0), Icmp, Select(NE), Return
    // The Select lowering passes the boolean condition directly with NE
    assert!(
        entry.instructions.len() >= 4,
        "Expected at least 4 instructions for select pattern, got {}",
        entry.instructions.len()
    );

    // Find the Select opcode in the instruction stream
    let has_select = entry
        .instructions
        .iter()
        .any(|inst| matches!(&inst.opcode, Opcode::Select { .. }));
    assert!(has_select, "Expected Select opcode in instruction stream");
}

#[test]
fn test_select_isel() {
    let func = build_select();
    let mfunc = compile_tmir_function(&func);

    // Select is lowered to CSEL on AArch64
    assert!(
        has_opcode(&mfunc, AArch64Opcode::Csel),
        "Expected CSEL instruction for Select"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test 11: GetElementPtr (typed pointer arithmetic)
// ===========================================================================
//
// fn array_access(arr: *i32, idx: i32) -> i32 {
//     *gep(i32, arr, idx, 0)
// }

fn build_gep() -> TmirFunction {
    TmirFunction {
        id: FuncId(13),
        name: "array_access".to_string(),
        ty: FuncTy {
            params: vec![Ty::Ptr(Box::new(Ty::Int(32))), Ty::Int(64)],
            returns: vec![Ty::Int(32)],
        },
        entry: BlockId(0),
        blocks: vec![TmirBlock {
            id: BlockId(0),
            params: vec![
                (ValueId(0), Ty::Ptr(Box::new(Ty::Int(32)))),
                (ValueId(1), Ty::Int(64)),
            ],
            body: vec![
                // v2 = gep(i32, v0, v1, 0)
                InstrNode {
                    instr: Instr::GetElementPtr {
                        elem_ty: Ty::Int(32),
                        base: ValueId(0),
                        index: ValueId(1),
                        offset: 0,
                    },
                    results: vec![ValueId(2)],
                    proofs: vec![],
                },
                // v3 = load(i32, v2)
                InstrNode {
                    instr: Instr::Load {
                        ty: Ty::Int(32),
                        ptr: ValueId(2),
                    },
                    results: vec![ValueId(3)],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(3)],
                    },
                    results: vec![],
                    proofs: vec![],
                },
            ],
        }],
        proofs: vec![],
    }
}

#[test]
fn test_gep_adapter() {
    let func = build_gep();
    let (lir_func, _) = translate_only(&func).unwrap();

    assert_eq!(lir_func.name, "array_access");
    let entry = &lir_func.blocks[&lir_func.entry_block];

    // GEP for i32 (4 bytes) with offset 0:
    //   Iconst(4) + Imul(idx, 4) + Iadd(base, mul_result) + Iconst(0) + Iadd(alias) + Load + Return
    assert!(
        entry.instructions.len() >= 4,
        "Expected at least 4 instructions for GEP+Load+Return, got {}",
        entry.instructions.len()
    );

    // Should have a Load instruction
    let has_load = entry
        .instructions
        .iter()
        .any(|inst| matches!(&inst.opcode, Opcode::Load { .. }));
    assert!(has_load, "Expected Load opcode after GEP");
}

#[test]
fn test_gep_isel() {
    let func = build_gep();
    let mfunc = compile_tmir_function(&func);

    // GEP lowers to MUL + ADD, then load
    assert!(
        has_opcode(&mfunc, AArch64Opcode::LdrRI) || has_opcode(&mfunc, AArch64Opcode::LdrRO),
        "Expected LDR for load after GEP"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test 12: GEP with non-zero byte offset (struct array access)
// ===========================================================================
//
// fn struct_field_in_array(arr: *{i32, i32}, idx: i64) -> i32 {
//     // Access arr[idx].field_at_offset_4
//     let ptr = gep({i32,i32}, arr, idx, 4);
//     *ptr
// }

fn build_gep_with_offset() -> TmirFunction {
    TmirFunction {
        id: FuncId(14),
        name: "struct_field_in_array".to_string(),
        ty: FuncTy {
            params: vec![Ty::Ptr(Box::new(Ty::Int(64))), Ty::Int(64)],
            returns: vec![Ty::Int(32)],
        },
        entry: BlockId(0),
        blocks: vec![TmirBlock {
            id: BlockId(0),
            params: vec![
                (ValueId(0), Ty::Ptr(Box::new(Ty::Int(64)))),
                (ValueId(1), Ty::Int(64)),
            ],
            body: vec![
                // v2 = gep(i64, v0, v1, 4) — each element is 8 bytes (i64), offset 4 for second i32 field
                InstrNode {
                    instr: Instr::GetElementPtr {
                        elem_ty: Ty::Int(64),
                        base: ValueId(0),
                        index: ValueId(1),
                        offset: 4,
                    },
                    results: vec![ValueId(2)],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Load {
                        ty: Ty::Int(32),
                        ptr: ValueId(2),
                    },
                    results: vec![ValueId(3)],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(3)],
                    },
                    results: vec![],
                    proofs: vec![],
                },
            ],
        }],
        proofs: vec![],
    }
}

#[test]
fn test_gep_with_offset_adapter() {
    let func = build_gep_with_offset();
    let (lir_func, _) = translate_only(&func).unwrap();

    assert_eq!(lir_func.name, "struct_field_in_array");
    let entry = &lir_func.blocks[&lir_func.entry_block];

    // GEP for i64 (8 bytes) with offset 4:
    //   Iconst(8) + Imul + Iadd(base, scaled) + Iconst(4) + Iadd(offset) + Load + Return
    assert!(
        entry.instructions.len() >= 6,
        "Expected at least 6 instructions for GEP with offset, got {}",
        entry.instructions.len()
    );
}

#[test]
fn test_gep_with_offset_isel() {
    let func = build_gep_with_offset();
    let mfunc = compile_tmir_function(&func);

    // Should have multiplication (for index scaling) and load
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
// Test 13: Type casts (SExt, ZExt, Trunc)
// ===========================================================================
//
// fn widen_and_narrow(x: i8) -> i8 {
//     let wide = sext(x, i32);
//     let narrow = trunc(wide, i8);
//     narrow
// }

fn build_cast_chain() -> TmirFunction {
    TmirFunction {
        id: FuncId(15),
        name: "widen_and_narrow".to_string(),
        ty: FuncTy {
            params: vec![Ty::Int(8)],
            returns: vec![Ty::Int(8)],
        },
        entry: BlockId(0),
        blocks: vec![TmirBlock {
            id: BlockId(0),
            params: vec![(ValueId(0), Ty::Int(8))],
            body: vec![
                // v1 = sext(v0, i8 -> i32)
                InstrNode {
                    instr: Instr::Cast {
                        op: CastOp::SExt,
                        src_ty: Ty::Int(8),
                        dst_ty: Ty::Int(32),
                        operand: ValueId(0),
                    },
                    results: vec![ValueId(1)],
                    proofs: vec![],
                },
                // v2 = trunc(v1, i32 -> i8)
                InstrNode {
                    instr: Instr::Cast {
                        op: CastOp::Trunc,
                        src_ty: Ty::Int(32),
                        dst_ty: Ty::Int(8),
                        operand: ValueId(1),
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

#[test]
fn test_cast_chain_adapter() {
    let func = build_cast_chain();
    let (lir_func, _) = translate_only(&func).unwrap();

    assert_eq!(lir_func.name, "widen_and_narrow");
    let entry = &lir_func.blocks[&lir_func.entry_block];
    // Sextend + Trunc + Return = 3 instructions
    assert_eq!(entry.instructions.len(), 3);
    assert!(matches!(
        &entry.instructions[0].opcode,
        Opcode::Sextend { from_ty: Type::I8, to_ty: Type::I32 }
    ));
    assert!(matches!(
        &entry.instructions[1].opcode,
        Opcode::Trunc { to_ty: Type::I8 }
    ));
}

#[test]
fn test_cast_chain_isel() {
    let func = build_cast_chain();
    let mfunc = compile_tmir_function(&func);

    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
    assert!(
        total_insts(&mfunc) >= 3,
        "Expected at least 3 machine instructions (COPY, SXTB, AND/TRUNC, RET)"
    );
}

// ===========================================================================
// Test 14: Float arithmetic + FP cast
// ===========================================================================
//
// fn float_op(a: f64, b: f64) -> i32 {
//     let sum = a + b;
//     fcvt_to_int(sum, i32)
// }

fn build_float_to_int() -> TmirFunction {
    TmirFunction {
        id: FuncId(16),
        name: "float_to_int".to_string(),
        ty: FuncTy {
            params: vec![Ty::Float(64), Ty::Float(64)],
            returns: vec![Ty::Int(32)],
        },
        entry: BlockId(0),
        blocks: vec![TmirBlock {
            id: BlockId(0),
            params: vec![(ValueId(0), Ty::Float(64)), (ValueId(1), Ty::Float(64))],
            body: vec![
                InstrNode {
                    instr: Instr::BinOp {
                        op: BinOp::FAdd,
                        ty: Ty::Float(64),
                        lhs: ValueId(0),
                        rhs: ValueId(1),
                    },
                    results: vec![ValueId(2)],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Cast {
                        op: CastOp::FPToSI,
                        src_ty: Ty::Float(64),
                        dst_ty: Ty::Int(32),
                        operand: ValueId(2),
                    },
                    results: vec![ValueId(3)],
                    proofs: vec![],
                },
                InstrNode {
                    instr: Instr::Return {
                        values: vec![ValueId(3)],
                    },
                    results: vec![],
                    proofs: vec![],
                },
            ],
        }],
        proofs: vec![],
    }
}

#[test]
fn test_float_to_int_adapter() {
    let func = build_float_to_int();
    let (lir_func, _) = translate_only(&func).unwrap();

    assert_eq!(lir_func.name, "float_to_int");
    assert_eq!(
        lir_func.signature.params,
        vec![Type::F64, Type::F64]
    );
    assert_eq!(lir_func.signature.returns, vec![Type::I32]);

    let entry = &lir_func.blocks[&lir_func.entry_block];
    // Fadd + FcvtToInt + Return = 3 instructions
    assert_eq!(entry.instructions.len(), 3);
}

#[test]
fn test_float_to_int_isel() {
    let func = build_float_to_int();
    let mfunc = compile_tmir_function(&func);

    assert!(
        has_opcode(&mfunc, AArch64Opcode::FaddRR),
        "Expected FADD for f64 addition"
    );
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test 15: Builder API usage
// ===========================================================================
//
// Test the tmir-func builder API by constructing a function with it.
// fn builder_add(a: i32, b: i32) -> i32 { a + b }

#[test]
fn test_builder_api() {
    let mut fb = FunctionBuilder::new(
        FuncId(20),
        "builder_add".to_string(),
        vec![Ty::Int(32), Ty::Int(32)],
        vec![Ty::Int(32)],
    );

    let (entry_block, params) = fb.entry_block();
    let a = params[0];
    let b = params[1];
    let result = fb.fresh_value();

    fb.add_block(
        entry_block,
        vec![(a, Ty::Int(32)), (b, Ty::Int(32))],
        vec![
            builder::binop(BinOp::Add, Ty::Int(32), a, b, result),
            builder::ret(vec![result]),
        ],
    );

    let func = fb.build();
    let (lir_func, _) = translate_only(&func).unwrap();

    assert_eq!(lir_func.name, "builder_add");
    assert_eq!(lir_func.signature.params, vec![Type::I32, Type::I32]);
    assert_eq!(lir_func.signature.returns, vec![Type::I32]);

    let entry = &lir_func.blocks[&lir_func.entry_block];
    assert_eq!(entry.instructions.len(), 2); // Iadd + Return
}

#[test]
fn test_builder_api_isel() {
    let mut fb = FunctionBuilder::new(
        FuncId(21),
        "builder_add".to_string(),
        vec![Ty::Int(32), Ty::Int(32)],
        vec![Ty::Int(32)],
    );

    let (entry_block, params) = fb.entry_block();
    let a = params[0];
    let b = params[1];
    let result = fb.fresh_value();

    fb.add_block(
        entry_block,
        vec![(a, Ty::Int(32)), (b, Ty::Int(32))],
        vec![
            builder::binop(BinOp::Add, Ty::Int(32), a, b, result),
            builder::ret(vec![result]),
        ],
    );

    let func = fb.build();
    let mfunc = compile_tmir_function(&func);

    assert!(has_opcode(&mfunc, AArch64Opcode::AddRR));
    assert!(has_opcode(&mfunc, AArch64Opcode::Ret));
}

// ===========================================================================
// Test 16: Proof annotations survive adapter translation
// ===========================================================================

#[test]
fn test_proof_annotations_survive() {
    use tmir_types::TmirProof;

    let func = TmirFunction {
        id: FuncId(22),
        name: "proven_add".to_string(),
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
                    proofs: vec![TmirProof::NoOverflow { signed: true }],
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
        proofs: vec![TmirProof::Pure],
    };

    let (_, proof_ctx) = translate_only(&func).unwrap();

    // Function-level proofs
    assert!(
        proof_ctx.is_function_pure(),
        "Function-level Pure proof should survive adapter translation"
    );

    // Value-level proofs: the result of the add (ValueId(2) -> mapped Value) should have NoOverflow
    let has_no_overflow = proof_ctx
        .value_proofs
        .values()
        .any(|proofs| proofs.iter().any(|p| matches!(p, llvm2_lower::adapter::Proof::NoOverflow { signed: true })));
    assert!(
        has_no_overflow,
        "NoOverflow proof should be attached to the add result"
    );
}

// ===========================================================================
// Comprehensive: all new programs compile through adapter without errors
// ===========================================================================

#[test]
fn test_all_new_programs_adapter_succeeds() {
    let programs: Vec<(&str, TmirFunction)> = vec![
        ("dispatch", build_switch()),
        ("call_through_ptr", build_call_indirect()),
        ("abs_select", build_select()),
        ("array_access", build_gep()),
        ("struct_field_in_array", build_gep_with_offset()),
        ("widen_and_narrow", build_cast_chain()),
        ("float_to_int", build_float_to_int()),
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
// Comprehensive: all new single-block programs compile through ISel
// ===========================================================================

#[test]
fn test_all_new_single_block_programs_compile() {
    let programs: Vec<(&str, TmirFunction)> = vec![
        ("call_through_ptr", build_call_indirect()),
        ("abs_select", build_select()),
        ("array_access", build_gep()),
        ("struct_field_in_array", build_gep_with_offset()),
        ("widen_and_narrow", build_cast_chain()),
        ("float_to_int", build_float_to_int()),
    ];

    for (name, func) in &programs {
        let mfunc = compile_tmir_function(func);
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
