// llvm2-codegen/tests/e2e_cegis_superopt.rs - CEGIS superopt pipeline wiring tests
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Regression coverage for issue #395: wire the CEGIS superopt pass into the
// pipeline and expose direct observability for integration tests.

use llvm2_codegen::pipeline::{OptLevel, Pipeline, PipelineConfig, build_add_test_function};

use tmir::{
    BinOp, Block as TmirBlock, BlockId, FuncId, FuncTy, Function as TmirFunction,
    Inst, InstrNode, Module as TmirModule, Ty, ValueId,
};

fn make_cegis_pipeline() -> Pipeline {
    Pipeline::new(PipelineConfig {
        opt_level: OptLevel::O0,
        cegis_superopt_budget_sec: Some(1),
        target_triple: "aarch64-unknown-unknown".to_string(),
        ..Default::default()
    })
}

fn build_add_tmir() -> (TmirFunction, TmirModule) {
    let mut module = TmirModule::new("test");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![Ty::I32, Ty::I32],
        returns: vec![Ty::I32],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "add32", ft_id, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![(ValueId::new(0), Ty::I32), (ValueId::new(1), Ty::I32)],
        body: vec![
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I32,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            })
            .with_result(ValueId::new(2)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(2)],
            }),
        ],
    }];
    module.add_function(func.clone());
    (func, module)
}

#[test]
fn test_cegis_flag_is_noop_when_disabled() {
    let pipeline = Pipeline::new(PipelineConfig::default());
    let mut func = build_add_test_function();

    assert!(pipeline.run_cegis_superopt(&mut func).is_none());
}

#[cfg(feature = "verify")]
#[test]
fn test_cegis_flag_runs_pass() {
    let pipeline = make_cegis_pipeline();
    let mut func = build_add_test_function();

    let stats = pipeline
        .run_cegis_superopt(&mut func)
        .expect("CEGIS pass should run when budget is enabled");

    assert_eq!(stats.functions_seen, 1);
    assert_eq!(stats.cache_misses, 1);
    assert_eq!(stats.cache_puts, 1);
}

#[cfg(feature = "verify")]
#[test]
fn test_cegis_cache_hit_on_repeat() {
    let pipeline = make_cegis_pipeline();

    let mut func1 = build_add_test_function();
    let first = pipeline
        .run_cegis_superopt(&mut func1)
        .expect("first CEGIS run should execute");
    assert_eq!(first.cache_misses, 1);
    assert_eq!(first.cache_puts, 1);

    let mut func2 = build_add_test_function();
    let second = pipeline
        .run_cegis_superopt(&mut func2)
        .expect("second CEGIS run should execute");
    assert_eq!(second.functions_seen, 1);
    assert_eq!(second.cache_hits, 1);
    assert_eq!(second.cache_misses, 0);
}

#[test]
fn test_full_pipeline_with_cegis_flag() {
    let pipeline = make_cegis_pipeline();
    let (tmir_func, module) = build_add_tmir();
    let (lir_func, _) = llvm2_lower::translate_function(&tmir_func, &module)
        .expect("tMIR add function should translate");

    let obj_bytes = pipeline
        .compile_function(&lir_func)
        .expect("full pipeline should compile with the CEGIS flag enabled");

    assert!(!obj_bytes.is_empty(), "pipeline should produce non-empty object bytes");
}
