// aggregate_types.rs — Phase 2a aggregate type translation tests (#391)
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Phase 2a aggregate-type translation tests.
//!
//! These tests cover the adapter's extended type translator which now accepts
//! `Ty::Array(TyId, len)` and `Ty::Tuple(Vec<Ty>)` — types that previously
//! returned `UnsupportedType`. The adapter resolves `TyId` via the module's
//! `types` table, and flattens tuples to anonymous LIR structs.
//!
//! Design: `designs/2026-04-18-aggregate-lowering.md` (Phase 2a section).

use llvm2_lower::adapter::{
    translate_type, translate_type_with_structs, translate_type_with_tables,
    AdapterError,
};
use llvm2_lower::types::Type;
use tmir::{
    Block as TmirBlock, BlockId, CallingConv, FieldDef, FuncId, FuncTy, Function,
    Inst, InstrNode, Linkage, Module, StructDef, StructId, Ty, TyId, ValueId,
};

// ---------------------------------------------------------------------------
// translate_type_with_tables — direct unit tests
// ---------------------------------------------------------------------------

#[test]
fn array_of_i32_resolves_through_types_table() {
    // types[0] = i32, so Array(TyId(0), 8) -> Array<I32; 8>
    let types = vec![Ty::I32];
    let arr = Ty::Array(TyId::new(0), 8);
    let lir = translate_type_with_tables(&arr, &[], &types).unwrap();
    assert_eq!(lir, Type::Array(Box::new(Type::I32), 8));
    // Size/align sanity-check: 8 * 4 = 32 bytes, align 4.
    assert_eq!(lir.bytes(), 32);
    assert_eq!(lir.align(), 4);
}

#[test]
fn array_length_zero_is_valid() {
    // A zero-length array is legal in tMIR (e.g., C-style `int[0]`); adapter
    // should not reject it. Downstream size calculations handle zero cleanly
    // (`Array(elem, 0).bytes() == 0`).
    let types = vec![Ty::I64];
    let arr = Ty::Array(TyId::new(0), 0);
    let lir = translate_type_with_tables(&arr, &[], &types).unwrap();
    assert_eq!(lir, Type::Array(Box::new(Type::I64), 0));
    assert_eq!(lir.bytes(), 0);
}

#[test]
fn array_of_f64_preserves_element_type() {
    let types = vec![Ty::F64];
    let arr = Ty::Array(TyId::new(0), 4);
    let lir = translate_type_with_tables(&arr, &[], &types).unwrap();
    assert_eq!(lir, Type::Array(Box::new(Type::F64), 4));
    assert_eq!(lir.bytes(), 32); // 4 * 8
    assert_eq!(lir.align(), 8);
}

#[test]
fn array_with_out_of_range_tyid_errors_cleanly() {
    // types table is empty, so TyId(0) cannot resolve — must not panic.
    let arr = Ty::Array(TyId::new(0), 4);
    let err = translate_type_with_tables(&arr, &[], &[]).unwrap_err();
    match err {
        AdapterError::UnsupportedType(msg) => {
            assert!(msg.contains("Array"), "error msg: {}", msg);
            assert!(msg.contains("out of range"), "error msg: {}", msg);
        }
        other => panic!("expected UnsupportedType, got {:?}", other),
    }
}

#[test]
fn array_length_exceeding_u32_max_is_rejected() {
    // tMIR's `Ty::Array(TyId, u64)` allows lengths up to `u64::MAX`, but LIR
    // stores array lengths as `u32`. The adapter uses `u32::try_from` to
    // surface oversized lengths as a clean `UnsupportedType` error rather
    // than silently truncating (which a future lossy `as u32` would do).
    let types = vec![Ty::I32];
    let arr = Ty::Array(TyId::new(0), u32::MAX as u64 + 1);
    let err = translate_type_with_tables(&arr, &[], &types).unwrap_err();
    match err {
        AdapterError::UnsupportedType(msg) => {
            assert!(msg.contains("exceeds u32::MAX"), "got: {}", msg);
        }
        other => panic!("expected UnsupportedType, got {:?}", other),
    }
}

#[test]
fn array_of_struct_resolves_recursively() {
    // types[0] = Struct(Point{x:F64, y:F64}); array of 3 points.
    let structs = vec![StructDef {
        id: StructId::new(0),
        name: "Point".to_string(),
        fields: vec![
            FieldDef { name: "x".to_string(), ty: Ty::F64, offset: None },
            FieldDef { name: "y".to_string(), ty: Ty::F64, offset: None },
        ],
        size: None,
        align: None,
    }];
    let types = vec![Ty::Struct(StructId::new(0))];
    let arr = Ty::Array(TyId::new(0), 3);
    let lir = translate_type_with_tables(&arr, &structs, &types).unwrap();
    let point = Type::Struct(vec![Type::F64, Type::F64]);
    assert_eq!(lir, Type::Array(Box::new(point), 3));
    // Each Point is 16 bytes (2 × F64), so 3 × 16 = 48.
    assert_eq!(lir.bytes(), 48);
}

#[test]
fn array_of_array_via_nested_tyids() {
    // types[0] = I32 (inner element),
    // types[1] = Array(TyId(0), 4)  (inner Array<I32; 4>).
    // outer = Array(TyId(1), 2) -> Array<Array<I32; 4>; 2>.
    let types = vec![Ty::I32, Ty::Array(TyId::new(0), 4)];
    let outer = Ty::Array(TyId::new(1), 2);
    let lir = translate_type_with_tables(&outer, &[], &types).unwrap();
    let inner = Type::Array(Box::new(Type::I32), 4);
    assert_eq!(lir, Type::Array(Box::new(inner), 2));
    assert_eq!(lir.bytes(), 32); // 2 × (4 × 4)
}

#[test]
fn empty_tuple_maps_to_empty_struct() {
    let t = Ty::Tuple(vec![]);
    let lir = translate_type(&t).unwrap();
    assert_eq!(lir, Type::Struct(vec![]));
    assert_eq!(lir.bytes(), 0);
}

#[test]
fn tuple_of_scalars_maps_to_struct() {
    // (i32, bool, f64) — bool is B1 in LIR.
    let t = Ty::Tuple(vec![Ty::I32, Ty::Bool, Ty::F64]);
    let lir = translate_type(&t).unwrap();
    assert_eq!(lir, Type::Struct(vec![Type::I32, Type::B1, Type::F64]));
}

#[test]
fn tuple_of_mixed_sizes_has_c_padding() {
    // (i8, i64) — expect padding between i8 and i64 (7 bytes), total 16, align 8.
    let t = Ty::Tuple(vec![Ty::I8, Ty::I64]);
    let lir = translate_type(&t).unwrap();
    assert_eq!(lir, Type::Struct(vec![Type::I8, Type::I64]));
    assert_eq!(lir.bytes(), 16);
    assert_eq!(lir.align(), 8);
    // offset_of must reflect alignment.
    assert_eq!(lir.offset_of(0), Some(0));
    assert_eq!(lir.offset_of(1), Some(8));
}

#[test]
fn tuple_of_tuples_nests_correctly() {
    let t = Ty::Tuple(vec![
        Ty::Tuple(vec![Ty::I32, Ty::I32]),
        Ty::F64,
    ]);
    let lir = translate_type(&t).unwrap();
    assert_eq!(
        lir,
        Type::Struct(vec![
            Type::Struct(vec![Type::I32, Type::I32]),
            Type::F64,
        ])
    );
}

#[test]
fn tuple_of_array_resolves_through_types_table() {
    // Tuple elements are inline Tys, but an Array element inside the tuple
    // still needs the types table.
    let types = vec![Ty::U8];
    let t = Ty::Tuple(vec![Ty::I32, Ty::Array(TyId::new(0), 4)]);
    let lir = translate_type_with_tables(&t, &[], &types).unwrap();
    assert_eq!(
        lir,
        Type::Struct(vec![
            Type::I32,
            Type::Array(Box::new(Type::I8), 4), // U8 normalises to I8
        ])
    );
}

#[test]
fn struct_containing_array_field_resolves() {
    // Regression for design-doc Gap B: struct fields that contain arrays
    // previously failed to lower because the adapter rejected Ty::Array.
    let structs = vec![StructDef {
        id: StructId::new(0),
        name: "Buf".to_string(),
        fields: vec![
            FieldDef { name: "len".to_string(), ty: Ty::I64, offset: None },
            FieldDef {
                name: "data".to_string(),
                ty: Ty::Array(TyId::new(0), 16),
                offset: None,
            },
        ],
        size: None,
        align: None,
    }];
    let types = vec![Ty::U8];
    let lir = translate_type_with_tables(
        &Ty::Struct(StructId::new(0)),
        &structs,
        &types,
    )
    .unwrap();
    assert_eq!(
        lir,
        Type::Struct(vec![
            Type::I64,
            Type::Array(Box::new(Type::I8), 16),
        ])
    );
    // Layout: i64 (8) + [u8; 16] (16), aligned to 8. 8 + 16 = 24.
    assert_eq!(lir.bytes(), 24);
}

#[test]
fn enum_type_remains_unsupported_phase_2a() {
    // Enums are deferred to Phase 2b (#398). Verify adapter still rejects.
    use tmir::EnumId;
    let e = Ty::Enum(EnumId::new(0));
    assert!(translate_type(&e).is_err());
}

#[test]
fn translate_type_with_structs_still_rejects_array() {
    // The back-compat wrapper passes an empty types slice, so Array must
    // continue to error (matches pre-Phase-2a semantics).
    let structs: Vec<StructDef> = vec![];
    let arr = Ty::Array(TyId::new(0), 4);
    assert!(translate_type_with_structs(&arr, &structs).is_err());
}

// ---------------------------------------------------------------------------
// Integration tests — translate a full function whose signature contains
// Array / Tuple types, exercising the adapter end-to-end.
// ---------------------------------------------------------------------------

/// Build a tMIR function `fn id(x: Ptr) -> Ptr` so we can verify aggregate
/// parameter/return types flow through `translate_signature` without
/// failing at signature translation.
fn make_identity_function(params: Vec<Ty>, returns: Vec<Ty>) -> Module {
    let mut module = Module::new("aggregate_sig_test");
    let ft = FuncTy { params: params.clone(), returns, is_vararg: false };
    let ft_id = module.add_func_type(ft);

    // One block with params matching the function params, returning the first
    // param unchanged. We use `Ty::Ptr` for inputs to avoid needing an aggregate
    // value in the body — the signature types themselves are what we're testing.
    let entry = BlockId::new(0);
    let v0 = ValueId::new(0);
    let block = TmirBlock {
        id: entry,
        params: params
            .iter()
            .enumerate()
            .map(|(i, t)| (ValueId::new(i as u32), t.clone()))
            .collect(),
        body: vec![InstrNode {
            inst: Inst::Return { values: vec![v0] },
            results: vec![],
            proofs: vec![],
            span: None,
        }],
    };

    let mut func = Function::new(FuncId::new(0), "id", ft_id, entry);
    func.linkage = Linkage::External;
    func.calling_conv = CallingConv::C;
    func.blocks = vec![block];
    module.add_function(func);
    module
}

#[test]
fn function_with_array_param_translates_through_signature() {
    // fn f(a: [i32; 4]) -> [i32; 4] — types[0] = i32, param/return = Array(TyId(0), 4).
    let mut module = make_identity_function(
        vec![Ty::Array(TyId::new(0), 4)],
        vec![Ty::Array(TyId::new(0), 4)],
    );
    // Register the element type in the module's types table.
    module.types.push(Ty::I32);

    let results = llvm2_lower::adapter::translate_module(&module).unwrap();
    assert_eq!(results.len(), 1);
    let (func, _proofs) = &results[0];
    let expected = Type::Array(Box::new(Type::I32), 4);
    assert_eq!(func.signature.params, vec![expected.clone()]);
    assert_eq!(func.signature.returns, vec![expected]);
}

#[test]
fn function_with_tuple_param_translates_through_signature() {
    // fn f((i32, f64)) -> (i32, f64).
    let tup = Ty::Tuple(vec![Ty::I32, Ty::F64]);
    let module = make_identity_function(vec![tup.clone()], vec![tup]);
    let results = llvm2_lower::adapter::translate_module(&module).unwrap();
    let (func, _proofs) = &results[0];
    let expected = Type::Struct(vec![Type::I32, Type::F64]);
    assert_eq!(func.signature.params, vec![expected.clone()]);
    assert_eq!(func.signature.returns, vec![expected]);
}

#[test]
fn function_with_tuple_of_array_param_translates() {
    // fn f((i32, [u8; 8])) -> (i32, [u8; 8]).
    let mut module = make_identity_function(
        vec![Ty::Tuple(vec![Ty::I32, Ty::Array(TyId::new(0), 8)])],
        vec![Ty::Tuple(vec![Ty::I32, Ty::Array(TyId::new(0), 8)])],
    );
    module.types.push(Ty::U8);

    let results = llvm2_lower::adapter::translate_module(&module).unwrap();
    let (func, _proofs) = &results[0];
    let expected = Type::Struct(vec![
        Type::I32,
        Type::Array(Box::new(Type::I8), 8),
    ]);
    assert_eq!(func.signature.params, vec![expected]);
}

#[test]
fn function_with_array_returns_error_when_types_table_missing() {
    // Adapter must surface a clean error (not panic) when the types table
    // is missing an entry for an Array(TyId).
    let module = make_identity_function(
        vec![Ty::Array(TyId::new(0), 4)],
        vec![Ty::Array(TyId::new(0), 4)],
    );
    // Intentionally don't push to module.types — leave it empty.
    let result = llvm2_lower::adapter::translate_module(&module);
    assert!(result.is_err(), "expected error for unresolvable TyId");
    match result.unwrap_err() {
        AdapterError::UnsupportedType(msg) => {
            assert!(msg.contains("Array"), "msg={}", msg);
        }
        other => panic!("expected UnsupportedType, got {:?}", other),
    }
}
