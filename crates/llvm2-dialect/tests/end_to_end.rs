// llvm2-dialect - End-to-end progressive lowering test
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Proof-of-concept end-to-end test: build a `verif.*` function, lower it
//! through `tmir.*` and `machir.*`, and emit a `MachFunction`.

use llvm2_ir::{AArch64Opcode, Type};

use llvm2_dialect::conversion::ConversionDriver;
use llvm2_dialect::dialects::conversions::{
    register_all, tmir_to_machir_driver, verif_to_tmir_driver, FINGERPRINT_STUB_MAGIC,
};
use llvm2_dialect::dialects::{machir, tmir, verif};
use llvm2_dialect::emit_mach_function;
use llvm2_dialect::id::DialectOpId;
use llvm2_dialect::module::{DialectFunction, DialectModule};
use llvm2_dialect::pass::{validate_legality, Legality};
use llvm2_dialect::registry::DialectRegistry;

fn build_verif_module() -> (DialectModule, llvm2_dialect::id::DialectId, llvm2_dialect::id::DialectId, llvm2_dialect::id::DialectId) {
    let mut registry = DialectRegistry::new();
    let (verif_id, tmir_id, machir_id) = register_all(&mut registry);

    // fn fingerprint_of(ptr: i64, len: i64) -> i64 {
    //     return verif.fingerprint_batch_stub(ptr, len)
    // }
    let mut func = DialectFunction::new(
        "fingerprint_of",
        vec![Type::I64, Type::I64],
        vec![Type::I64],
    );
    let entry = func.entry_block().unwrap();
    let ptr = func.params[0].0;
    let len = func.params[1].0;
    let result = func.alloc_value();
    func.append_op(
        entry,
        DialectOpId::new(verif_id, verif::FINGERPRINT_BATCH_STUB),
        vec![(result, Type::I64)],
        vec![ptr, len],
        vec![],
        None,
    );
    // Mirror tmir.ret so after verif->tmir lowering the module returns.
    func.append_op(
        entry,
        DialectOpId::new(tmir_id, tmir::TMIR_RET),
        vec![],
        vec![result],
        vec![],
        None,
    );

    let mut module = DialectModule::new("fingerprint", registry);
    module.push_function(func);
    (module, verif_id, tmir_id, machir_id)
}

#[test]
fn progressive_lowering_verif_to_machine() {
    let (mut module, verif_id, tmir_id, machir_id) = build_verif_module();

    // --- Stage 1: verif -> tmir ---
    let stage1 = verif_to_tmir_driver(verif_id, tmir_id);
    stage1.run(&mut module).expect("verif->tmir conversion succeeded");

    // After stage 1 no verif.* ops should remain. We assert legality against a
    // producer set of {tmir}. Note: the tmir.ret we inserted up front is in
    // tmir, so it's legal.
    let stage1_legality = Legality::new().produces(tmir_id);
    validate_legality(&module, &stage1_legality)
        .expect("stage1 output contains only tmir.* ops");

    // --- Stage 2: tmir -> machir ---
    let stage2 = tmir_to_machir_driver(tmir_id, machir_id);
    stage2.run(&mut module).expect("tmir->machir conversion succeeded");

    let stage2_legality = Legality::new().produces(machir_id);
    validate_legality(&module, &stage2_legality)
        .expect("stage2 output contains only machir.* ops");

    // --- Stage 3: machir -> MachFunction ---
    let mf = emit_mach_function(&module, 0).expect("mach function emit");

    // Assertions on the resulting MachFunction ---------------------------
    // Signature preserved.
    assert_eq!(mf.name, "fingerprint_of");
    assert_eq!(mf.signature.params, vec![Type::I64, Type::I64]);
    assert_eq!(mf.signature.returns, vec![Type::I64]);

    // We expect the following ops in order (ignoring flags, exact registers):
    //   Movz  (the magic constant)
    //   EorRR (ptr XOR len)
    //   EorRR (XOR magic)
    //   Ret
    let opcodes: Vec<AArch64Opcode> = mf.insts.iter().map(|i| i.opcode).collect();
    assert_eq!(
        opcodes,
        vec![
            AArch64Opcode::Movz,
            AArch64Opcode::EorRR,
            AArch64Opcode::EorRR,
            AArch64Opcode::Ret,
        ],
        "unexpected MachInst sequence: {:?}",
        opcodes
    );

    // The Movz instruction should carry the FINGERPRINT_STUB_MAGIC value.
    let movz = &mf.insts[0];
    let imm = movz
        .operands
        .iter()
        .find_map(|o| o.as_imm())
        .expect("Movz has immediate operand");
    assert_eq!(imm as u64, FINGERPRINT_STUB_MAGIC);

    // Every instruction should be in the entry block.
    let entry = &mf.blocks[mf.entry.0 as usize];
    assert_eq!(entry.insts.len(), mf.insts.len());
}

#[test]
fn legality_violation_detected() {
    // Build a module that contains both verif.* and tmir.* ops. Verify the
    // legality checker catches both "op dialect not in produces set" and the
    // explicit-forbid path.
    let (module, verif_id, tmir_id, _machir_id) = build_verif_module();

    // The module mixes verif + tmir, so producing-only-verif fails on the
    // tmir.ret op.
    let legality = Legality::new().produces(verif_id);
    let err = validate_legality(&module, &legality)
        .expect_err("mixed verif+tmir module should fail a verif-only produces set");
    let msg = format!("{}", err);
    assert!(msg.contains("not in produces set"), "got: {}", msg);

    // Accepting both verif and tmir passes.
    let legality = Legality::new().produces(verif_id).produces(tmir_id);
    validate_legality(&module, &legality).expect("both dialects allowed");

    // Explicit forbid on the verif fingerprint op trips legality.
    let forbidden = DialectOpId::new(verif_id, verif::FINGERPRINT_BATCH_STUB);
    let legality = Legality::new()
        .produces(verif_id)
        .produces(tmir_id)
        .forbid(forbidden);
    let err = validate_legality(&module, &legality)
        .expect_err("explicitly forbidden op should trip legality");
    let msg = format!("{}", err);
    assert!(msg.contains("explicitly forbidden"), "got: {}", msg);
}

#[test]
fn legality_accepts_side_rejects_unexpected_source_dialect() {
    // Pass's declared accepts set must be enforced symmetrically with produces.
    // A pass that accepts only tmir must reject a module containing verif ops,
    // even though produces is empty (= "no output constraint").
    let (module, _verif_id, tmir_id, _machir_id) = build_verif_module();

    let legality = Legality::new().accepts(tmir_id);
    let err = validate_legality(&module, &legality)
        .expect_err("verif op should not pass an accepts: [tmir] gate");
    let msg = format!("{}", err);
    assert!(
        msg.contains("not in accepts set"),
        "expected accepts-side error, got: {}",
        msg
    );
}

#[test]
fn registry_roundtrip_and_ops_lookup() {
    let mut registry = DialectRegistry::new();
    let (verif_id, tmir_id, machir_id) = register_all(&mut registry);

    assert_eq!(registry.by_name("verif"), Some(verif_id));
    assert_eq!(registry.by_name("tmir"), Some(tmir_id));
    assert_eq!(registry.by_name("machir"), Some(machir_id));
    assert_eq!(registry.by_name("nope"), None);

    let stub = DialectOpId::new(verif_id, verif::FINGERPRINT_BATCH_STUB);
    let def = registry.op_def(stub).expect("op def lookup");
    assert_eq!(def.name, "verif.fingerprint_batch_stub");
    assert!(def.capabilities.is_pure());

    let ret = DialectOpId::new(tmir_id, tmir::TMIR_RET);
    let def = registry.op_def(ret).expect("ret def lookup");
    assert!(def.capabilities.is_terminator());

    let add_rr = DialectOpId::new(machir_id, machir::MACHIR_ADD_RR);
    assert_eq!(registry.op_def(add_rr).map(|d| d.name), Some("machir.add.rr"));
}

#[test]
fn unknown_ops_pass_through_driver() {
    // If a driver has no registered pattern for an op, the conversion driver
    // should copy it through verbatim. This enables mixed-dialect modules
    // (e.g. partial conversions where some ops are already in the destination
    // dialect).
    let (mut module, verif_id, tmir_id, _machir_id) = build_verif_module();

    // An empty driver leaves everything untouched.
    let empty = ConversionDriver::new();
    empty.run(&mut module).unwrap();

    // Verify we still see the original verif op present.
    let func = &module.functions[0];
    let has_verif = func.iter_ops().any(|o| o.op.dialect == verif_id);
    assert!(has_verif, "empty driver should pass verif op through");

    // And the tmir.ret we seeded should still be there.
    let has_ret = func
        .iter_ops()
        .any(|o| o.op == DialectOpId::new(tmir_id, tmir::TMIR_RET));
    assert!(has_ret, "empty driver should pass tmir.ret through");
}
