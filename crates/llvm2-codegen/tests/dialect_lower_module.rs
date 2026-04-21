// llvm2-codegen/tests/dialect_lower_module.rs - lower_module pipeline hook
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Integration test for LLVM2#433 / tMIR#428. Confirms that
// `pipeline::dialect_lower_module` drives the full verif.* -> tmir.* ->
// machir.* -> llvm2_ir::MachFunction pipeline with at least one dialect
// registered. Unhandled DialectOps must fail in the legality gate rather
// than leak into the resulting MachFunction.

use llvm2_dialect::dialects::conversions::{register_all, FINGERPRINT_STUB_MAGIC};
use llvm2_dialect::dialects::{tmir, verif};
use llvm2_dialect::id::DialectOpId;
use llvm2_dialect::module::{DialectFunction, DialectModule};
use llvm2_dialect::registry::DialectRegistry;
use llvm2_dialect::LowerModuleError;
use llvm2_ir::{AArch64Opcode, Type};
use llvm2_codegen::pipeline::dialect_lower_module;

fn build_fingerprint_module() -> DialectModule {
    let mut registry = DialectRegistry::new();
    let (verif_id, tmir_id, _machir_id) = register_all(&mut registry);

    // fn fingerprint_of(ptr: i64, len: i64) -> i64 {
    //   return verif.fingerprint_batch_stub(ptr, len)
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
    // Seed a tmir.ret so the module returns after verif->tmir lowering.
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
    module
}

#[test]
fn pipeline_dialect_lower_module_emits_mach_function() {
    let mut module = build_fingerprint_module();
    let mach_fns =
        dialect_lower_module(&mut module).expect("dialect_lower_module succeeded");

    assert_eq!(mach_fns.len(), 1, "one function in, one MachFunction out");
    let mf = &mach_fns[0];

    // Signature preserved through the full pipeline.
    assert_eq!(mf.name, "fingerprint_of");
    assert_eq!(mf.signature.params, vec![Type::I64, Type::I64]);
    assert_eq!(mf.signature.returns, vec![Type::I64]);

    // Expected post-lowering sequence: Movz magic, Eor ptr^len, Eor ^magic, Ret.
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

    // Magic constant survives the verif -> tmir -> machir -> MachInst chain.
    let movz = &mf.insts[0];
    let imm = movz
        .operands
        .iter()
        .find_map(|o| o.as_imm())
        .expect("Movz has immediate operand");
    assert_eq!(imm as u64, FINGERPRINT_STUB_MAGIC);
}

#[test]
fn pipeline_dialect_lower_module_rejects_missing_dialect() {
    // Build a module whose registry has none of the required dialects.
    let registry = DialectRegistry::new();
    let mut module = DialectModule::new("empty", registry);
    // Push an empty function so the module isn't trivially empty on another axis.
    module.push_function(DialectFunction::new(
        "noop",
        vec![],
        vec![],
    ));
    let err = dialect_lower_module(&mut module)
        .expect_err("missing dialects must be a hard error");
    assert!(
        matches!(err, LowerModuleError::MissingDialect(_)),
        "expected MissingDialect, got: {:?}",
        err
    );
}
