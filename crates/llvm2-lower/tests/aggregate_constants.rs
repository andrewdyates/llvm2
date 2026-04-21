// aggregate_constants.rs — Phase 2b aggregate-constant lowering (#391)
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Phase 2b aggregate-constant lowering tests.
//!
//! A `Const { value: Constant::Aggregate(..) }` of a struct / tuple / array
//! type with all-scalar fields must survive the adapter without erroring and
//! must materialise as:
//!
//! ```text
//!   StackAddr slot            -> result (pointer to aggregate)
//!   Iconst/Fconst <field 0>
//!   StructGep/ArrayGep        -> field_ptr 0
//!   Store <field 0>, field_ptr 0
//!   ... (repeat per field) ...
//! ```
//!
//! Downstream SROA (see `crates/llvm2-opt/tests/sroa_pipeline.rs`) eliminates
//! the stack traffic at O1+ when the aggregate pointer does not escape.
//!
//! Enum and function pointer aggregate constants are **intentionally** rejected
//! pending a decision on #398.

use llvm2_lower::adapter::translate_function;
use llvm2_lower::instructions::Opcode;
use tmir::{
    Block as TmirBlock, BlockId, Constant, FieldDef, FuncId, FuncTy, FuncTyId,
    Function as TmirFunction, Inst, InstrNode, Module as TmirModule, StructDef, StructId, Ty,
    TyId, ValueId,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn v(n: u32) -> ValueId {
    ValueId::new(n)
}

fn b(n: u32) -> BlockId {
    BlockId::new(n)
}

fn f(n: u32) -> FuncId {
    FuncId::new(n)
}

fn single_function_module(
    func_name: &str,
    ty: FuncTy,
    structs: Vec<StructDef>,
    types: Vec<Ty>,
    blocks: Vec<TmirBlock>,
) -> TmirModule {
    let entry = blocks.first().expect("module must have a block").id;
    let mut module = TmirModule::new(func_name);
    module.structs = structs;
    module.types = types;
    let fty_id: FuncTyId = module.add_func_type(ty);
    let mut func = TmirFunction::new(f(0), func_name, fty_id, entry);
    func.blocks = blocks;
    module.add_function(func);
    module
}

fn single_function<'m>(module: &'m TmirModule) -> &'m TmirFunction {
    module.functions.first().expect("module must have one function")
}

// ---------------------------------------------------------------------------
// Tests — structs, tuples, arrays
// ---------------------------------------------------------------------------

/// A struct aggregate const with two i64 fields lowers to StackAddr + 2×(Iconst + StructGep + Store).
#[test]
fn struct_aggregate_const_lowers() {
    let structs = vec![StructDef {
        id: StructId::new(0),
        name: "Pair".to_string(),
        fields: vec![
            FieldDef { name: "a".to_string(), ty: Ty::I64, offset: None },
            FieldDef { name: "b".to_string(), ty: Ty::I64, offset: None },
        ],
        size: None,
        align: None,
    }];
    let struct_ty = Ty::Struct(StructId::new(0));

    let module = single_function_module(
        "mk_pair",
        FuncTy { params: vec![], returns: vec![struct_ty.clone()], is_vararg: false },
        structs,
        vec![],
        vec![TmirBlock {
            id: b(0),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: struct_ty.clone(),
                    value: Constant::Aggregate(vec![Constant::Int(7), Constant::Int(42)]),
                })
                .with_result(v(0)),
                InstrNode::new(Inst::Return { values: vec![v(0)] }),
            ],
        }],
    );

    let (lir_func, _proofs) =
        translate_function(single_function(&module), &module).expect("adapter must accept struct const");

    // Inspect the entry block's instruction stream.
    let entry = lir_func.entry_block;
    let bb = &lir_func.blocks[&entry];

    let stack_addrs = bb
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::StackAddr { .. }))
        .count();
    let struct_geps = bb
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::StructGep { .. }))
        .count();
    let stores = bb
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::Store))
        .count();
    let iconsts = bb
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::Iconst { .. }))
        .count();

    assert_eq!(stack_addrs, 1, "one StackAddr for the aggregate slot");
    assert_eq!(struct_geps, 2, "one StructGep per field");
    assert_eq!(stores, 2, "one Store per field");
    assert_eq!(iconsts, 2, "one Iconst per integer field");
}

/// A tuple aggregate const lowers the same way as a struct const.
#[test]
fn tuple_aggregate_const_lowers() {
    let tuple_ty = Ty::Tuple(vec![Ty::I32, Ty::I64, Ty::Bool]);
    let module = single_function_module(
        "mk_tuple",
        FuncTy { params: vec![], returns: vec![tuple_ty.clone()], is_vararg: false },
        vec![],
        vec![],
        vec![TmirBlock {
            id: b(0),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: tuple_ty.clone(),
                    value: Constant::Aggregate(vec![
                        Constant::Int(1),
                        Constant::Int(2),
                        Constant::Bool(true),
                    ]),
                })
                .with_result(v(0)),
                InstrNode::new(Inst::Return { values: vec![v(0)] }),
            ],
        }],
    );

    let (lir_func, _proofs) =
        translate_function(single_function(&module), &module).expect("adapter must accept tuple const");
    let bb = &lir_func.blocks[&lir_func.entry_block];

    let struct_geps = bb
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::StructGep { .. }))
        .count();
    let stores = bb
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::Store))
        .count();
    assert_eq!(struct_geps, 3, "three fields → three StructGeps");
    assert_eq!(stores, 3, "three Stores");
}

/// An array aggregate const lowers to ArrayGep per element (not StructGep).
#[test]
fn array_aggregate_const_lowers_via_array_gep() {
    // types[0] = I32, array_ty = [I32; 3]
    let types = vec![Ty::I32];
    let array_ty = Ty::Array(TyId::new(0), 3);

    let module = single_function_module(
        "mk_arr",
        FuncTy { params: vec![], returns: vec![array_ty.clone()], is_vararg: false },
        vec![],
        types,
        vec![TmirBlock {
            id: b(0),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: array_ty.clone(),
                    value: Constant::Aggregate(vec![
                        Constant::Int(10),
                        Constant::Int(20),
                        Constant::Int(30),
                    ]),
                })
                .with_result(v(0)),
                InstrNode::new(Inst::Return { values: vec![v(0)] }),
            ],
        }],
    );

    let (lir_func, _proofs) =
        translate_function(single_function(&module), &module).expect("adapter must accept array const");
    let bb = &lir_func.blocks[&lir_func.entry_block];

    let array_geps = bb
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::ArrayGep { .. }))
        .count();
    let struct_geps = bb
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::StructGep { .. }))
        .count();
    let stores = bb
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::Store))
        .count();

    assert_eq!(array_geps, 3, "array const uses ArrayGep per element");
    assert_eq!(struct_geps, 0, "array const must not emit StructGep");
    assert_eq!(stores, 3, "three elements → three Stores");
}

/// Nested aggregates (struct-of-struct constants) recurse into the same
/// outer stack slot. Phase 2c of #443 lifts the earlier Phase 2b rejection:
/// the inner aggregate is written through a parent `StructGep` + per-leaf
/// `Store`, with **no nested StackAddr** emitted.
#[test]
fn nested_aggregate_const_lowers_into_one_slot() {
    let structs = vec![StructDef {
        id: StructId::new(0),
        name: "Wrapper".to_string(),
        fields: vec![FieldDef {
            name: "inner".to_string(),
            ty: Ty::I64,
            offset: None,
        }],
        size: None,
        align: None,
    }];
    let tuple_ty = Ty::Tuple(vec![Ty::Struct(StructId::new(0))]);

    let module = single_function_module(
        "mk_nested",
        FuncTy { params: vec![], returns: vec![tuple_ty.clone()], is_vararg: false },
        structs,
        vec![],
        vec![TmirBlock {
            id: b(0),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: tuple_ty.clone(),
                    value: Constant::Aggregate(vec![Constant::Aggregate(vec![Constant::Int(1)])]),
                })
                .with_result(v(0)),
                InstrNode::new(Inst::Return { values: vec![v(0)] }),
            ],
        }],
    );

    let (lir_func, _proofs) = translate_function(single_function(&module), &module)
        .expect("nested aggregate constants lower in Phase 2c");
    let bb = &lir_func.blocks[&lir_func.entry_block];

    let stack_addrs = bb
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::StackAddr { .. }))
        .count();
    let struct_geps = bb
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::StructGep { .. }))
        .count();
    let stores = bb
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::Store))
        .count();
    let iconsts = bb
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::Iconst { .. }))
        .count();

    assert_eq!(
        stack_addrs, 1,
        "nested aggregate reuses the outer StackAddr — no nested slot"
    );
    // Outer tuple field + inner struct field = 2 StructGeps.
    assert_eq!(
        struct_geps, 2,
        "one GEP to the tuple field + one GEP to the nested struct field"
    );
    assert_eq!(stores, 1, "one Store at the innermost leaf");
    assert_eq!(iconsts, 1, "one Iconst for the leaf i64 value");
}

/// Deeply nested aggregate: struct { inner: struct { leaf: i64, flag: bool } }
/// must lower into exactly one stack slot with two GEPs per leaf and per-leaf
/// scalar stores.
#[test]
fn deeply_nested_aggregate_const_lowers() {
    let structs = vec![
        StructDef {
            id: StructId::new(0),
            name: "Leaf".to_string(),
            fields: vec![
                FieldDef { name: "leaf".to_string(), ty: Ty::I64, offset: None },
                FieldDef { name: "flag".to_string(), ty: Ty::Bool, offset: None },
            ],
            size: None,
            align: None,
        },
        StructDef {
            id: StructId::new(1),
            name: "Outer".to_string(),
            fields: vec![FieldDef {
                name: "inner".to_string(),
                ty: Ty::Struct(StructId::new(0)),
                offset: None,
            }],
            size: None,
            align: None,
        },
    ];
    let outer_ty = Ty::Struct(StructId::new(1));

    let module = single_function_module(
        "mk_deep",
        FuncTy { params: vec![], returns: vec![outer_ty.clone()], is_vararg: false },
        structs,
        vec![],
        vec![TmirBlock {
            id: b(0),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: outer_ty.clone(),
                    value: Constant::Aggregate(vec![Constant::Aggregate(vec![
                        Constant::Int(7),
                        Constant::Bool(true),
                    ])]),
                })
                .with_result(v(0)),
                InstrNode::new(Inst::Return { values: vec![v(0)] }),
            ],
        }],
    );

    let (lir_func, _proofs) = translate_function(single_function(&module), &module)
        .expect("deeply nested aggregate constants lower in Phase 2c");
    let bb = &lir_func.blocks[&lir_func.entry_block];

    let stack_addrs = bb
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::StackAddr { .. }))
        .count();
    let stores = bb
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::Store))
        .count();

    assert_eq!(stack_addrs, 1, "exactly one StackAddr regardless of nesting depth");
    // Two leaf fields -> two Stores.
    assert_eq!(stores, 2, "two leaves -> two Stores");
}

/// Closure-style aggregate: a struct whose first field is a function pointer
/// (`Ty::Func`) and whose second field is a captured `i64` must lower in
/// Phase 2c — the function pointer is treated as a 64-bit scalar pointer.
#[test]
fn func_ptr_aggregate_const_lowers() {
    // Build a `FuncTy` id for the function-pointer field.
    let mut module = TmirModule::new("closure_mod");
    let callee_fty = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let closure_fields = vec![
        FieldDef { name: "fptr".to_string(), ty: Ty::Func(callee_fty), offset: None },
        FieldDef { name: "env".to_string(), ty: Ty::I64, offset: None },
    ];
    module.structs = vec![StructDef {
        id: StructId::new(0),
        name: "Closure".to_string(),
        fields: closure_fields,
        size: None,
        align: None,
    }];
    let closure_ty = Ty::Struct(StructId::new(0));
    let sig_id: FuncTyId = module.add_func_type(FuncTy {
        params: vec![],
        returns: vec![closure_ty.clone()],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(f(0), "mk_closure", sig_id, b(0));
    func.blocks = vec![TmirBlock {
        id: b(0),
        params: vec![],
        body: vec![
            InstrNode::new(Inst::Const {
                ty: closure_ty.clone(),
                // `Constant::Int` for the func-ptr field carries the function
                // symbol's address; `0x1000` is a stand-in — the adapter only
                // cares that the encoding is pointer-shaped.
                value: Constant::Aggregate(vec![
                    Constant::Int(0x1000),
                    Constant::Int(42),
                ]),
            })
            .with_result(v(0)),
            InstrNode::new(Inst::Return { values: vec![v(0)] }),
        ],
    }];
    module.add_function(func);

    let (lir_func, _proofs) =
        translate_function(single_function(&module), &module)
            .expect("func-ptr aggregate constants lower in Phase 2c");
    let bb = &lir_func.blocks[&lir_func.entry_block];

    let stack_addrs = bb
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::StackAddr { .. }))
        .count();
    let stores = bb
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::Store))
        .count();
    let iconsts = bb
        .instructions
        .iter()
        .filter(|i| matches!(i.opcode, Opcode::Iconst { .. }))
        .count();

    assert_eq!(stack_addrs, 1, "closure struct lowers into a single stack slot");
    assert_eq!(stores, 2, "one Store per field (fptr + env)");
    assert_eq!(iconsts, 2, "one Iconst per Int constant (fptr + env)");
}

/// `Ty::Enum` aggregate constants remain blocked on #398 — the adapter must
/// reject them with a clear diagnostic so front-ends know the path is not yet
/// supported.
#[test]
fn enum_aggregate_const_blocked_on_398() {
    use tmir::EnumId;
    let enum_ty = Ty::Enum(EnumId::new(0));

    let module = single_function_module(
        "mk_enum",
        FuncTy { params: vec![], returns: vec![enum_ty.clone()], is_vararg: false },
        vec![],
        vec![],
        vec![TmirBlock {
            id: b(0),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: enum_ty.clone(),
                    value: Constant::Aggregate(vec![Constant::Int(0)]),
                })
                .with_result(v(0)),
                InstrNode::new(Inst::Return { values: vec![v(0)] }),
            ],
        }],
    );

    let result = translate_function(single_function(&module), &module);
    assert!(
        result.is_err(),
        "Ty::Enum aggregate constants are blocked on #398"
    );
}

/// Arity mismatch (type says 2 fields, const supplies 1) must be caught early.
#[test]
fn aggregate_const_arity_mismatch_rejected() {
    let structs = vec![StructDef {
        id: StructId::new(0),
        name: "Pair".to_string(),
        fields: vec![
            FieldDef { name: "a".to_string(), ty: Ty::I64, offset: None },
            FieldDef { name: "b".to_string(), ty: Ty::I64, offset: None },
        ],
        size: None,
        align: None,
    }];
    let struct_ty = Ty::Struct(StructId::new(0));

    let module = single_function_module(
        "bad_arity",
        FuncTy { params: vec![], returns: vec![struct_ty.clone()], is_vararg: false },
        structs,
        vec![],
        vec![TmirBlock {
            id: b(0),
            params: vec![],
            body: vec![
                InstrNode::new(Inst::Const {
                    ty: struct_ty.clone(),
                    value: Constant::Aggregate(vec![Constant::Int(1)]),
                })
                .with_result(v(0)),
                InstrNode::new(Inst::Return { values: vec![v(0)] }),
            ],
        }],
    );

    let result = translate_function(single_function(&module), &module);
    assert!(
        result.is_err(),
        "arity mismatch between type and Constant::Aggregate must be rejected"
    );
}
