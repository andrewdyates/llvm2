// llvm2-dialect - Type-constraint vocabulary tests
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Tests for the `TypeConstraint` vocabulary added per the coordination spec
//! §5 (`designs/2026-04-18-aggregates-dialects-coordination.md`).
//!
//! The R9 snapshot-6 audit caught that `OpDef` silently dropped
//! `operand_types` / `result_types`. These tests lock in the restored
//! behaviour so #393's declarative rewrite engine has a stable contract to
//! match against.

use std::collections::HashMap;

use llvm2_dialect::dialect::TypeConstraint;
use llvm2_dialect::dialects::conversions::register_all;
use llvm2_dialect::dialects::{machir, tmir, verif};
use llvm2_dialect::id::{DialectOpId, ValueId};
use llvm2_dialect::module::{DialectFunction, DialectModule};
use llvm2_dialect::op::DialectOp;
use llvm2_dialect::pass::{
    validate_legality, validate_type_constraints, validate_type_constraints_with_env,
    Legality,
};
use llvm2_dialect::registry::DialectRegistry;
use llvm2_ir::Type;

// ---------------------------------------------------------------------------
// Unit-level `TypeConstraint::accepts` coverage.
// ---------------------------------------------------------------------------

#[test]
fn type_constraint_any_accepts_anything() {
    let c = TypeConstraint::Any;
    assert!(c.accepts(&Type::I32, &[]));
    assert!(c.accepts(&Type::F64, &[]));
    assert!(c.accepts(&Type::Struct(vec![Type::I8]), &[]));
}

#[test]
fn type_constraint_anyint_accepts_only_integers() {
    let c = TypeConstraint::AnyInt;
    assert!(c.accepts(&Type::I8, &[]));
    assert!(c.accepts(&Type::I64, &[]));
    assert!(c.accepts(&Type::I128, &[]));
    assert!(!c.accepts(&Type::F32, &[]));
    assert!(!c.accepts(&Type::B1, &[]));
    assert!(!c.accepts(&Type::Ptr, &[]));
    assert!(!c.accepts(&Type::Struct(vec![Type::I32]), &[]));
}

#[test]
fn type_constraint_anyfloat_accepts_only_floats() {
    let c = TypeConstraint::AnyFloat;
    assert!(c.accepts(&Type::F32, &[]));
    assert!(c.accepts(&Type::F64, &[]));
    assert!(!c.accepts(&Type::I32, &[]));
    assert!(!c.accepts(&Type::B1, &[]));
    assert!(!c.accepts(&Type::Ptr, &[]));
}

#[test]
fn type_constraint_anyscalar_rejects_aggregates() {
    let c = TypeConstraint::AnyScalar;
    assert!(c.accepts(&Type::I32, &[]));
    assert!(c.accepts(&Type::Ptr, &[]));
    assert!(c.accepts(&Type::B1, &[]));
    assert!(!c.accepts(&Type::Struct(vec![Type::I32]), &[]));
    assert!(!c.accepts(&Type::Array(Box::new(Type::I8), 4), &[]));
}

#[test]
fn type_constraint_aggregate_accepts_only_aggregates() {
    let c = TypeConstraint::Aggregate;
    assert!(c.accepts(&Type::Struct(vec![Type::I64, Type::Ptr]), &[]));
    assert!(c.accepts(&Type::Array(Box::new(Type::F32), 8), &[]));
    assert!(!c.accepts(&Type::I32, &[]));
    assert!(!c.accepts(&Type::Ptr, &[]));
}

#[test]
fn type_constraint_specific_requires_exact_match() {
    let c = TypeConstraint::Specific(Type::I64);
    assert!(c.accepts(&Type::I64, &[]));
    assert!(!c.accepts(&Type::I32, &[]));
    assert!(!c.accepts(&Type::F64, &[]));

    let struct_ty = Type::Struct(vec![Type::I64, Type::Ptr]);
    let c2 = TypeConstraint::Specific(struct_ty.clone());
    assert!(c2.accepts(&struct_ty, &[]));
    assert!(!c2.accepts(&Type::Struct(vec![Type::I32, Type::Ptr]), &[]));
}

#[test]
fn type_constraint_sameas_resolves_against_operand_list() {
    // operand_tys = [I32, I32] — SameAs(0) accepts I32 and rejects others.
    let operand_tys = vec![Some(Type::I32), Some(Type::I32)];
    let c = TypeConstraint::SameAs(0);
    assert!(c.accepts(&Type::I32, &operand_tys));
    assert!(!c.accepts(&Type::I64, &operand_tys));
}

#[test]
fn type_constraint_sameas_out_of_range_fails_closed() {
    // No panic; constraint simply doesn't match.
    let c = TypeConstraint::SameAs(5);
    assert!(!c.accepts(&Type::I32, &[Some(Type::I32)]));
}

#[test]
fn type_constraint_sameas_unresolved_slot_fails_closed() {
    // Regression for #410: a `SameAs(idx)` reference to an unresolved
    // operand slot must fail closed. Previously the caller collapsed `None`
    // entries out of the operand list before handing it to `accepts`, which
    // silently shifted indices — e.g. `SameAs(1)` would resolve against
    // operand[2] if operand[1] was unresolved.
    let operand_tys = vec![Some(Type::I32), None, Some(Type::I32)];

    // SameAs(1) points at the unresolved middle slot → must NOT silently
    // accept against operand[2]'s I32.
    let c_mid = TypeConstraint::SameAs(1);
    assert!(
        !c_mid.accepts(&Type::I32, &operand_tys),
        "SameAs(1) with operand[1] unresolved must fail closed, not resolve against operand[2]"
    );

    // SameAs(2) still resolves correctly when operand[2] is present.
    let c_tail = TypeConstraint::SameAs(2);
    assert!(
        c_tail.accepts(&Type::I32, &operand_tys),
        "SameAs(2) with operand[2] resolved as I32 should accept I32"
    );
    assert!(
        !c_tail.accepts(&Type::I64, &operand_tys),
        "SameAs(2) with operand[2] resolved as I32 should reject I64"
    );
}

// ---------------------------------------------------------------------------
// `OpDef` sample dialect coverage — every sample op now carries non-empty
// constraint slices (the regression R9 caught).
// ---------------------------------------------------------------------------

#[test]
fn sample_dialects_populate_type_constraints() {
    let mut registry = DialectRegistry::new();
    let (verif_id, tmir_id, machir_id) = register_all(&mut registry);

    let verif_stub = DialectOpId::new(verif_id, verif::FINGERPRINT_BATCH_STUB);
    let def = registry.op_def(verif_stub).unwrap();
    assert_eq!(def.operand_types.len(), 2, "verif stub operand constraints");
    assert_eq!(def.result_types.len(), 1, "verif stub result constraints");

    for op in &[tmir::TMIR_ADD, tmir::TMIR_XOR] {
        let def = registry
            .op_def(DialectOpId::new(tmir_id, *op))
            .expect("tmir op def");
        assert_eq!(def.operand_types.len(), 2);
        assert_eq!(def.result_types.len(), 1);
        // SameAs(0) on result ties result type to operand[0].
        assert_eq!(def.result_types[0], TypeConstraint::SameAs(0));
    }

    let tmir_const = registry
        .op_def(DialectOpId::new(tmir_id, tmir::TMIR_CONST))
        .unwrap();
    assert!(tmir_const.operand_types.is_empty(), "const takes no operands");
    assert_eq!(tmir_const.result_types, &[TypeConstraint::AnyInt]);

    // Every machir op declares at least one constraint side.
    for op in &[
        machir::MACHIR_MOVZ_I64,
        machir::MACHIR_ADD_RR,
        machir::MACHIR_EOR_RR,
    ] {
        let def = registry
            .op_def(DialectOpId::new(machir_id, *op))
            .expect("machir op def");
        assert!(
            !def.result_types.is_empty(),
            "machir op {:?} should declare result type",
            op
        );
    }
}

// ---------------------------------------------------------------------------
// `validate_type_constraints` + `validate_type_constraints_with_env`
// ---------------------------------------------------------------------------

fn make_tmir_add_op(operands: Vec<ValueId>, result_ty: Type) -> DialectOp {
    // Stand-alone DialectOp isolated from a function arena. The arena id is
    // irrelevant for the type-constraint check.
    DialectOp {
        id: llvm2_dialect::id::OpId(0),
        op: DialectOpId::new(llvm2_dialect::id::DialectId(1), tmir::TMIR_ADD),
        results: vec![(ValueId(99), result_ty)],
        operands,
        attrs: vec![],
        source: None,
    }
}

#[test]
fn validate_type_constraints_rejects_type_mismatch() {
    let mut registry = DialectRegistry::new();
    let (_verif_id, tmir_id, _machir_id) = register_all(&mut registry);

    let op = make_tmir_add_op(vec![ValueId(0), ValueId(1)], Type::F64);
    let def = registry
        .op_def(DialectOpId::new(tmir_id, tmir::TMIR_ADD))
        .unwrap();

    // `tmir.add` result is `SameAs(0)` which must match operand[0]'s type.
    // With env [v0 -> I32, v1 -> I32], an F64 result violates the constraint.
    let mut env = HashMap::new();
    env.insert(ValueId(0), Type::I32);
    env.insert(ValueId(1), Type::I32);

    assert!(!validate_type_constraints(&op, def),
        "env-less check skips operand side but the result still fails AnyInt",
    );
    let err = validate_type_constraints_with_env(&op, def, &env)
        .expect_err("F64 result violates SameAs(0) when operand[0] is I32");
    assert!(format!("{}", err).contains("tmir.add result 0"));
}

#[test]
fn validate_type_constraints_accepts_well_typed_op() {
    let mut registry = DialectRegistry::new();
    let (_verif_id, tmir_id, _machir_id) = register_all(&mut registry);

    let op = make_tmir_add_op(vec![ValueId(0), ValueId(1)], Type::I64);
    let def = registry
        .op_def(DialectOpId::new(tmir_id, tmir::TMIR_ADD))
        .unwrap();

    let mut env = HashMap::new();
    env.insert(ValueId(0), Type::I64);
    env.insert(ValueId(1), Type::I64);

    validate_type_constraints_with_env(&op, def, &env)
        .expect("I64 add with I64 operands should validate");
}

#[test]
fn validate_type_constraints_rejects_non_integer_operands_for_tmir_xor() {
    let mut registry = DialectRegistry::new();
    let (_verif_id, tmir_id, _machir_id) = register_all(&mut registry);

    let op = DialectOp {
        id: llvm2_dialect::id::OpId(0),
        op: DialectOpId::new(llvm2_dialect::id::DialectId(1), tmir::TMIR_XOR),
        results: vec![(ValueId(99), Type::F64)],
        operands: vec![ValueId(0), ValueId(1)],
        attrs: vec![],
        source: None,
    };
    let def = registry
        .op_def(DialectOpId::new(tmir_id, tmir::TMIR_XOR))
        .unwrap();

    let mut env = HashMap::new();
    env.insert(ValueId(0), Type::F32);
    env.insert(ValueId(1), Type::F32);

    let err = validate_type_constraints_with_env(&op, def, &env)
        .expect_err("F32 operands should fail AnyInt");
    let msg = format!("{}", err);
    assert!(
        msg.contains("operand 0") && msg.contains("AnyInt"),
        "expected AnyInt error, got: {}",
        msg
    );
}

#[test]
fn validate_type_constraints_arity_mismatch_is_error() {
    let mut registry = DialectRegistry::new();
    let (_verif_id, tmir_id, _machir_id) = register_all(&mut registry);

    // `tmir.add` is fixed arity 2; passing only one operand is illegal.
    let op = DialectOp {
        id: llvm2_dialect::id::OpId(0),
        op: DialectOpId::new(llvm2_dialect::id::DialectId(1), tmir::TMIR_ADD),
        results: vec![(ValueId(99), Type::I64)],
        operands: vec![ValueId(0)],
        attrs: vec![],
        source: None,
    };
    let def = registry
        .op_def(DialectOpId::new(tmir_id, tmir::TMIR_ADD))
        .unwrap();
    let env = HashMap::new();
    let err = validate_type_constraints_with_env(&op, def, &env)
        .expect_err("arity mismatch should trip");
    let msg = format!("{}", err);
    assert!(msg.contains("arity"), "expected arity error, got: {}", msg);
}

#[test]
fn validate_legality_now_checks_types_end_to_end() {
    // Build a minimal module with a `tmir.add` of two I64 params returning
    // I64. `validate_legality` should accept it now that per-position type
    // constraints are woven into the check.
    let mut registry = DialectRegistry::new();
    let (_verif_id, tmir_id, _machir_id) = register_all(&mut registry);

    let mut func = DialectFunction::new(
        "add_i64",
        vec![Type::I64, Type::I64],
        vec![Type::I64],
    );
    let entry = func.entry_block().unwrap();
    let (a, b) = (func.params[0].0, func.params[1].0);
    let sum = func.alloc_value();
    func.append_op(
        entry,
        DialectOpId::new(tmir_id, tmir::TMIR_ADD),
        vec![(sum, Type::I64)],
        vec![a, b],
        vec![],
        None,
    );
    func.append_op(
        entry,
        DialectOpId::new(tmir_id, tmir::TMIR_RET),
        vec![],
        vec![sum],
        vec![],
        None,
    );

    let mut module = DialectModule::new("m", registry);
    module.push_function(func);

    validate_legality(&module, &Legality::new().produces(tmir_id))
        .expect("well-typed tmir module should pass legality");
}

#[test]
fn validate_type_constraints_with_partial_env_fails_closed_on_unresolved_sameas() {
    // Regression for #410. Construct an op whose `SameAs(idx)` target
    // references an unresolved operand slot. The old implementation
    // collapsed `None`s out of the operand list before handing it to
    // `TypeConstraint::accepts`, so `SameAs(1)` would silently resolve
    // against operand[2]'s type when operand[1] was missing. With the
    // positional fix the constraint must fail closed with a
    // LegalityViolation.
    //
    // This mirrors the partial-env use case #393's declarative rewriter
    // needs mid-pattern-match, and any incremental validator (linters,
    // conversion drivers).
    use llvm2_dialect::dialect::{Arity, Capabilities, OpDef};
    use llvm2_dialect::id::OpCode;

    let mut registry = DialectRegistry::new();
    let (_verif_id, tmir_id, _machir_id) = register_all(&mut registry);

    // Build a synthetic 3-operand op whose result must match operand[1]
    // via `SameAs(1)`. `tmir.add` is fixed-arity 2, so we fake a def
    // inline rather than muck with the registered dialect. We only need
    // the type-constraint machinery, and `validate_type_constraints_with_env`
    // takes the OpDef by reference.
    static OPERAND_TYS: &[TypeConstraint] = &[
        TypeConstraint::AnyInt,
        TypeConstraint::AnyInt,
        TypeConstraint::AnyInt,
    ];
    static RESULT_TYS: &[TypeConstraint] = &[TypeConstraint::SameAs(1)];
    let def = OpDef {
        op: OpCode(9999),
        name: "test.triop_sameas1",
        capabilities: Capabilities::EMPTY,
        num_operands: Arity::Fixed(3),
        num_results: Arity::Fixed(1),
        operand_types: OPERAND_TYS,
        result_types: RESULT_TYS,
    };

    let op = DialectOp {
        id: llvm2_dialect::id::OpId(0),
        op: DialectOpId::new(tmir_id, OpCode(9999)),
        results: vec![(ValueId(99), Type::I32)],
        operands: vec![ValueId(0), ValueId(1), ValueId(2)],
        attrs: vec![],
        source: None,
    };

    // Partial env: operand[1] (ValueId(1)) is deliberately missing so its
    // type is unresolved. Old code would silently let SameAs(1) match
    // against operand[2] (I32) and accept the I32 result.
    let mut env = HashMap::new();
    env.insert(ValueId(0), Type::I32);
    env.insert(ValueId(2), Type::I32);

    let err = validate_type_constraints_with_env(&op, &def, &env)
        .expect_err("SameAs(1) with operand[1] unresolved must fail closed");
    let msg = format!("{}", err);
    assert!(
        msg.contains("result 0") && msg.contains("SameAs(1)"),
        "expected SameAs(1) result violation, got: {}",
        msg
    );

    // Positive control: when the same env fully resolves operand[1] to
    // I32, the result (I32) should validate.
    let mut full_env = env.clone();
    full_env.insert(ValueId(1), Type::I32);
    validate_type_constraints_with_env(&op, &def, &full_env)
        .expect("SameAs(1) with operand[1] == I32 and result I32 should accept");

    // Negative control: operand[1] resolved to I64 should reject an I32
    // result (proves the constraint is actually consulting operand[1],
    // not operand[2]).
    let mut mismatched_env = env.clone();
    mismatched_env.insert(ValueId(1), Type::I64);
    let err = validate_type_constraints_with_env(&op, &def, &mismatched_env)
        .expect_err("SameAs(1) with operand[1] == I64 must reject I32 result");
    assert!(format!("{}", err).contains("SameAs(1)"));
}

#[test]
fn validate_legality_catches_type_mismatch_in_module() {
    // Same shape as above, but an F32 slot is passed as a parameter and fed
    // into a `tmir.add`. The operand-side AnyInt constraint should now trip
    // `validate_legality` since type checking is integrated.
    let mut registry = DialectRegistry::new();
    let (_verif_id, tmir_id, _machir_id) = register_all(&mut registry);

    let mut func = DialectFunction::new(
        "bad_add",
        vec![Type::F32, Type::F32],
        vec![Type::F32],
    );
    let entry = func.entry_block().unwrap();
    let (a, b) = (func.params[0].0, func.params[1].0);
    let sum = func.alloc_value();
    // Mismatched op: operands are F32 but `tmir.add` demands AnyInt.
    func.append_op(
        entry,
        DialectOpId::new(tmir_id, tmir::TMIR_ADD),
        vec![(sum, Type::F32)],
        vec![a, b],
        vec![],
        None,
    );

    let mut module = DialectModule::new("m_bad", registry);
    module.push_function(func);

    let err = validate_legality(&module, &Legality::new().produces(tmir_id))
        .expect_err("F32 operands should violate tmir.add's AnyInt constraint");
    let msg = format!("{}", err);
    assert!(
        msg.contains("tmir.add") && msg.contains("AnyInt"),
        "expected AnyInt violation, got: {}",
        msg
    );
}
