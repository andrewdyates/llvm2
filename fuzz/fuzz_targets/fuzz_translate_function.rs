// fuzz/fuzz_targets/fuzz_translate_function.rs
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// libFuzzer target shadowing `panic_fuzz_lower.rs`. Derives a minimal
// well-formed tMIR function from the fuzzer's byte stream (any shape
// survives because the tMIR types do not implement `Arbitrary` directly;
// we use byte bits to pick instruction variants). Feeds the result to
// `llvm2_lower::translate_function`. The contract is identical to the
// proptest harness: translate must return `Ok((Function, ProofContext))`
// or `Err(AdapterError)` — never panic.

#![no_main]

use libfuzzer_sys::fuzz_target;

use tmir::{
    BinOp, Block as TmirBlock, BlockId, Constant, FuncId, FuncTy, Function as TmirFunction,
    ICmpOp, Inst, InstrNode, Module as TmirModule, Ty, ValueId,
};

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

fn pick_icmp(byte: u8) -> ICmpOp {
    match byte % 6 {
        0 => ICmpOp::Eq,
        1 => ICmpOp::Ne,
        2 => ICmpOp::Slt,
        3 => ICmpOp::Sle,
        4 => ICmpOp::Ult,
        _ => ICmpOp::Ule,
    }
}

fuzz_target!(|data: &[u8]| {
    // Build a tMIR function whose shape is driven by `data`. We deliberately
    // stay on the well-formed side: a single block, 0..=2 params, a handful
    // of insts drawn from `data`. The "malformed" axis is already covered
    // by the proptest harness; here we want corpus-driven coverage of the
    // lowering passes, so `Err(..)` from the adapter short-circuits the
    // interesting signal anyway.
    let mut module = TmirModule::new("_fuzz_lower");
    let num_params = (data.first().copied().unwrap_or(0) % 3) as usize;
    let params = vec![Ty::I64; num_params];
    let ft_id = module.add_func_type(FuncTy {
        params: params.clone(),
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

    // Entry block params.
    let mut defined: Vec<ValueId> = Vec::new();
    let mut body: Vec<InstrNode> = Vec::new();
    let mut block_params: Vec<(ValueId, Ty)> = Vec::new();
    for _ in 0..num_params {
        let v = alloc_vid();
        block_params.push((v, Ty::I64));
        defined.push(v);
    }

    // Seed a constant so every ValueId is eventually defined.
    let zero = alloc_vid();
    body.push(
        InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(0) })
            .with_result(zero),
    );
    defined.push(zero);

    let nsteps = data.len().min(6);
    for i in 0..nsteps {
        let b = data[i];
        let kind = b % 4;
        match kind {
            0 => {
                // Constant
                let val = i as i64 - (b as i64);
                let v = alloc_vid();
                body.push(
                    InstrNode::new(Inst::Const { ty: Ty::I64, value: Constant::Int(val as i128) })
                        .with_result(v),
                );
                defined.push(v);
            }
            1 => {
                let lhs = *defined.last().unwrap();
                let rhs = defined[defined.len().saturating_sub(2).max(0)];
                let v = alloc_vid();
                body.push(
                    InstrNode::new(Inst::BinOp {
                        op: pick_binop(b),
                        ty: Ty::I64,
                        lhs,
                        rhs,
                    })
                    .with_result(v),
                );
                defined.push(v);
            }
            2 => {
                let lhs = *defined.last().unwrap();
                let rhs = defined[defined.len().saturating_sub(2).max(0)];
                let v = alloc_vid();
                body.push(
                    InstrNode::new(Inst::ICmp { op: pick_icmp(b), ty: Ty::I64, lhs, rhs })
                        .with_result(v),
                );
            }
            _ => {}
        }
    }

    let ret = *defined.last().unwrap_or(&zero);
    body.push(InstrNode::new(Inst::Return { values: vec![ret] }));
    func.blocks.push(TmirBlock {
        id: BlockId::new(0),
        params: block_params,
        body,
    });

    let _ = llvm2_lower::translate_function(&func, &module);
});
