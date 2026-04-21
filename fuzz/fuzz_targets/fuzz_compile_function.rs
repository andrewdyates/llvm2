// fuzz/fuzz_targets/fuzz_compile_function.rs
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// libFuzzer target shadowing `panic_fuzz_compile.rs`. Derives a tMIR
// function from `data`, lowers it, and compiles it via the full pipeline
// at a fuzzer-selected opt level. The contract is: the pipeline either
// returns `Ok(Vec<u8>)` or `Err(..)` — never panics.

#![no_main]

use libfuzzer_sys::fuzz_target;

use llvm2_codegen::pipeline::{OptLevel, Pipeline, PipelineConfig};
use tmir::{
    BinOp, Block as TmirBlock, BlockId, Constant, FuncId, FuncTy, Function as TmirFunction,
    Inst, InstrNode, Module as TmirModule, Ty, ValueId,
};

fn pick_opt_level(byte: u8) -> OptLevel {
    match byte % 4 {
        0 => OptLevel::O0,
        1 => OptLevel::O1,
        2 => OptLevel::O2,
        _ => OptLevel::O3,
    }
}

fn pick_binop(byte: u8) -> BinOp {
    match byte % 9 {
        0 => BinOp::Add,
        1 => BinOp::Sub,
        2 => BinOp::Mul,
        3 => BinOp::And,
        4 => BinOp::Or,
        5 => BinOp::Xor,
        6 => BinOp::Shl,
        7 => BinOp::LShr,
        _ => BinOp::AShr,
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let mut module = TmirModule::new("_fuzz_compile");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_fuzz_fn", ft_id, BlockId::new(0));
    let mut next_vid: u32 = 1000;
    let mut alloc_vid = || {
        let v = ValueId::new(next_vid);
        next_vid += 1;
        v
    };

    let mut body: Vec<InstrNode> = Vec::new();
    let mut defined: Vec<ValueId> = Vec::new();

    // Seed with a constant.
    let zero = alloc_vid();
    body.push(
        InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(0) })
            .with_result(zero),
    );
    defined.push(zero);

    let nsteps = data.len().saturating_sub(1).min(8);
    for i in 0..nsteps {
        let b = data[i + 1];
        if b & 1 == 0 {
            let v = alloc_vid();
            body.push(
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(b as i128),
                })
                .with_result(v),
            );
            defined.push(v);
        } else {
            let lhs = *defined.last().unwrap();
            let rhs = defined[defined.len().saturating_sub(2).max(0)];
            let v = alloc_vid();
            body.push(
                InstrNode::new(Inst::BinOp { op: pick_binop(b), ty: Ty::I64, lhs, rhs })
                    .with_result(v),
            );
            defined.push(v);
        }
    }

    let ret = *defined.last().unwrap_or(&zero);
    body.push(InstrNode::new(Inst::Return { values: vec![ret] }));
    func.blocks.push(TmirBlock {
        id: BlockId::new(0),
        params: vec![],
        body,
    });

    let lir = match llvm2_lower::translate_function(&func, &module) {
        Ok((l, _)) => l,
        Err(_) => return,
    };
    let config = PipelineConfig {
        opt_level: pick_opt_level(data[0]),
        emit_debug: false,
        ..Default::default()
    };
    let pipeline = Pipeline::new(config);
    let _ = pipeline.compile_function(&lir);
});
