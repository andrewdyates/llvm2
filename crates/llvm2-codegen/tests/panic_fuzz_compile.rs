// llvm2-codegen/tests/panic_fuzz_compile.rs
// Property-based panic-fuzz harness for `Pipeline::compile_function`.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Part of #387 (proptest panic-fuzz) / Part of #372 (Crash-free codegen).
//
// Reference: `designs/2026-04-18-crash-free-codegen-plan.md` §5 (proptest
// as primary defense) and §6 (per-crate harness).
//
// Contract under test: for *any* tMIR function, the full end-to-end
// compilation pipeline (`translate_function` -> `Pipeline::compile_function`)
// must either return `Ok(Vec<u8>)` or `Err(..)` — it must NEVER panic,
// abort, or debug-overflow. This is the integration-level totality test;
// the per-stage harnesses (`panic_fuzz_lower`, `panic_fuzz_encode`) cover
// each boundary individually.
//
// Run:
//   cargo test -p llvm2-codegen --test panic_fuzz_compile
// Increase case count via env:
//   PROPTEST_CASES=100000 cargo test -p llvm2-codegen --test panic_fuzz_compile

use std::panic;

use llvm2_codegen::pipeline::{OptLevel, Pipeline, PipelineConfig};
use proptest::prelude::*;
use tmir::{
    BinOp, Block as TmirBlock, BlockId, Constant, FuncId, FuncTy,
    Function as TmirFunction, ICmpOp, Inst, InstrNode, Module as TmirModule, Ty, ValueId,
};

// ---------------------------------------------------------------------------
// Shape spec (mirrors the structure used by panic_fuzz_lower so the two
// harnesses share a generator shape and shrink to the same minimal cases).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum InstKind {
    Const(i64),
    BinOp(BinOp),
    ICmp(ICmpOp),
    Return,
    Br(u32),
    CondBr(u32, u32),
}

#[derive(Debug, Clone)]
struct BlockSpec {
    insts: Vec<InstKind>,
}

#[derive(Debug, Clone)]
struct FuncSpec {
    num_params: u8,
    blocks: Vec<BlockSpec>,
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

fn binop_strategy() -> impl Strategy<Value = BinOp> {
    prop_oneof![
        Just(BinOp::Add), Just(BinOp::Sub), Just(BinOp::Mul),
        Just(BinOp::And), Just(BinOp::Or), Just(BinOp::Xor),
        Just(BinOp::Shl), Just(BinOp::LShr), Just(BinOp::AShr),
    ]
}

fn icmp_strategy() -> impl Strategy<Value = ICmpOp> {
    prop_oneof![
        Just(ICmpOp::Eq), Just(ICmpOp::Ne),
        Just(ICmpOp::Slt), Just(ICmpOp::Sle),
        Just(ICmpOp::Ult), Just(ICmpOp::Ule),
    ]
}

fn inst_kind_strategy(block_count: u32) -> impl Strategy<Value = InstKind> {
    prop_oneof![
        (-1000i64..=1000i64).prop_map(InstKind::Const),
        binop_strategy().prop_map(InstKind::BinOp),
        icmp_strategy().prop_map(InstKind::ICmp),
        Just(InstKind::Return),
        (0u32..block_count.max(1)).prop_map(InstKind::Br),
        (0u32..block_count.max(1), 0u32..block_count.max(1))
            .prop_map(|(a, b)| InstKind::CondBr(a, b)),
    ]
}

fn block_spec_strategy(block_count: u32) -> impl Strategy<Value = BlockSpec> {
    prop::collection::vec(inst_kind_strategy(block_count), 0..=5)
        .prop_map(|insts| BlockSpec { insts })
}

fn func_spec_strategy() -> impl Strategy<Value = FuncSpec> {
    // #447 closed: the pipeline's post-ISel debug-only connectivity check
    // no longer panics on legal multi-block shapes with zero inter-block
    // edges (e.g. all blocks terminating with `Return`). The generator is
    // widened back to 1..=3 blocks now that those inputs are tolerated.
    (1usize..=3usize)
        .prop_flat_map(|nb| {
            let bc = nb as u32;
            (
                0u8..=2,
                prop::collection::vec(block_spec_strategy(bc), nb..=nb),
            )
                .prop_map(|(num_params, blocks)| FuncSpec {
                    num_params,
                    blocks,
                })
        })
}

fn opt_level_strategy() -> impl Strategy<Value = OptLevel> {
    prop_oneof![
        Just(OptLevel::O0),
        Just(OptLevel::O1),
        Just(OptLevel::O2),
        Just(OptLevel::O3),
    ]
}

// ---------------------------------------------------------------------------
// Materialise a well-formed tMIR function.
//
// Unlike the lower harness, we intentionally stay on the well-formed side
// here because (a) the adapter-level malformed totality is already covered
// by `panic_fuzz_lower`, and (b) the full pipeline is a deep stack of
// passes and we want the signal here to isolate *pipeline-internal* panics
// rather than adapter-rejected-input re-panics. "Well-formed" still
// exercises a broad shape: 1..=3 blocks, 0..=2 params, 0..=5 insts/block,
// random br/condbr targets within range, random binop/icmp chains.
// ---------------------------------------------------------------------------

fn materialise(spec: &FuncSpec) -> (TmirFunction, TmirModule) {
    let mut module = TmirModule::new("_panic_fuzz_compile");
    let params: Vec<Ty> = vec![Ty::I64; spec.num_params as usize];
    let ft_id = module.add_func_type(FuncTy {
        params: params.clone(),
        returns: vec![Ty::I64],
        is_vararg: false,
    });

    let block_count = spec.blocks.len();
    let mut func = TmirFunction::new(FuncId::new(0), "_fuzz_fn", ft_id, BlockId::new(0));

    let mut next_vid: u32 = 1000;
    let mut alloc_vid = || -> ValueId {
        let v = ValueId::new(next_vid);
        next_vid += 1;
        v
    };

    func.blocks = Vec::with_capacity(block_count);
    for (bi, bspec) in spec.blocks.iter().enumerate() {
        let block_id = BlockId::new(bi as u32);
        let mut block_params: Vec<(ValueId, Ty)> = Vec::new();
        if bi == 0 {
            for _ in 0..spec.num_params {
                block_params.push((alloc_vid(), Ty::I64));
            }
        }

        let mut body: Vec<InstrNode> = Vec::new();
        let mut defined_i64: Vec<ValueId> = block_params.iter().map(|(v, _)| *v).collect();

        for inst in &bspec.insts {
            match inst {
                InstKind::Const(v) => {
                    let vid = alloc_vid();
                    body.push(
                        InstrNode::new(Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(*v as i128),
                        })
                        .with_result(vid),
                    );
                    defined_i64.push(vid);
                }
                InstKind::BinOp(op) => {
                    let (lhs, rhs) = pick_two(&mut defined_i64, &mut body, &mut alloc_vid);
                    let vid = alloc_vid();
                    body.push(
                        InstrNode::new(Inst::BinOp {
                            op: *op,
                            ty: Ty::I64,
                            lhs,
                            rhs,
                        })
                        .with_result(vid),
                    );
                    defined_i64.push(vid);
                }
                InstKind::ICmp(op) => {
                    let (lhs, rhs) = pick_two(&mut defined_i64, &mut body, &mut alloc_vid);
                    let vid = alloc_vid();
                    body.push(
                        InstrNode::new(Inst::ICmp {
                            op: *op,
                            ty: Ty::I64,
                            lhs,
                            rhs,
                        })
                        .with_result(vid),
                    );
                    let _ = vid; // bool value retained only by presence in body
                }
                InstKind::Return => {
                    let ret_v = pick_one(&mut defined_i64, &mut body, &mut alloc_vid);
                    body.push(InstrNode::new(Inst::Return { values: vec![ret_v] }));
                }
                InstKind::Br(tgt) => {
                    let tgt = BlockId::new(*tgt % block_count.max(1) as u32);
                    body.push(InstrNode::new(Inst::Br {
                        target: tgt,
                        args: vec![],
                    }));
                }
                InstKind::CondBr(t, e) => {
                    let cond = pick_one(&mut defined_i64, &mut body, &mut alloc_vid);
                    let t = BlockId::new(*t % block_count.max(1) as u32);
                    let e = BlockId::new(*e % block_count.max(1) as u32);
                    body.push(InstrNode::new(Inst::CondBr {
                        cond,
                        then_target: t,
                        then_args: vec![],
                        else_target: e,
                        else_args: vec![],
                    }));
                }
            }
        }

        // Ensure the block has at least one terminator.
        let terminates = body
            .last()
            .map(|n| {
                matches!(
                    n.inst,
                    Inst::Return { .. } | Inst::Br { .. } | Inst::CondBr { .. }
                )
            })
            .unwrap_or(false);
        if !terminates {
            let zero_vid = alloc_vid();
            body.push(
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(zero_vid),
            );
            body.push(InstrNode::new(Inst::Return {
                values: vec![zero_vid],
            }));
        }

        func.blocks.push(TmirBlock {
            id: block_id,
            params: block_params,
            body,
        });
    }

    (func, module)
}

fn pick_one<F: FnMut() -> ValueId>(
    defined: &mut Vec<ValueId>,
    body: &mut Vec<InstrNode>,
    alloc_vid: &mut F,
) -> ValueId {
    if let Some(v) = defined.last() {
        *v
    } else {
        let vid = alloc_vid();
        body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(0),
            })
            .with_result(vid),
        );
        defined.push(vid);
        vid
    }
}

fn pick_two<F: FnMut() -> ValueId>(
    defined: &mut Vec<ValueId>,
    body: &mut Vec<InstrNode>,
    alloc_vid: &mut F,
) -> (ValueId, ValueId) {
    let a = pick_one(defined, body, alloc_vid);
    let b = pick_one(defined, body, alloc_vid);
    (a, b)
}

// ---------------------------------------------------------------------------
// Property
// ---------------------------------------------------------------------------

fn assert_no_panic(func: &TmirFunction, module: &TmirModule, opt_level: OptLevel) {
    let f = func.clone();
    let m = module.clone();
    let result = panic::catch_unwind(panic::AssertUnwindSafe(move || {
        // Stage 1: tMIR -> LIR
        let lir = match llvm2_lower::translate_function(&f, &m) {
            Ok((lir_func, _)) => lir_func,
            Err(_) => return, // adapter rejected — that's fine, it didn't panic
        };
        // Stage 2: LIR -> object
        let config = PipelineConfig {
            opt_level,
            emit_debug: false,
            ..Default::default()
        };
        let pipeline = Pipeline::new(config);
        let _ = pipeline.compile_function(&lir);
    }));
    if let Err(payload) = result {
        let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".to_string()
        };
        panic!(
            "pipeline panicked on function '{}' ({} blocks, opt={:?}): {}",
            func.name,
            func.blocks.len(),
            opt_level,
            msg,
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        // The full pipeline is materially slower than the per-stage
        // harnesses (every case runs ISel, opt, regalloc, frame, encode,
        // emit). 256 cases at O3 on a 3-block function is still <30s
        // locally; reduce via PROPTEST_CASES=32 while iterating.
        cases: std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64),
        max_shrink_iters: 200,
        .. ProptestConfig::default()
    })]

    /// Random well-formed tMIR compiled at a random opt level. The
    /// pipeline must either produce a Mach-O object or return a typed
    /// error — never panic.
    #[test]
    fn compile_never_panics(
        spec in func_spec_strategy(),
        opt_level in opt_level_strategy(),
    ) {
        let (func, module) = materialise(&spec);
        assert_no_panic(&func, &module, opt_level);
    }

    /// Same generator, but force O0 to exercise the low-opt dispatcher
    /// path explicitly. Kept as a separate property so a regression in
    /// just-the-O0 path is easy to read from the failure message.
    #[test]
    fn compile_never_panics_o0(spec in func_spec_strategy()) {
        let (func, module) = materialise(&spec);
        assert_no_panic(&func, &module, OptLevel::O0);
    }
}

// ---------------------------------------------------------------------------
// Regression reproducers for known panics found by this harness
// ---------------------------------------------------------------------------
//
// These pin-down tests are hand-reduced shrinks of failing proptest cases.
// They previously pinned buggy behavior with `#[should_panic]`; they now
// assert the post-fix behavior (no panic; Ok or typed Err). See #447.

/// A multi-block function in which each block independently terminates
/// with `Return` (no `Br`/`CondBr` edges between them) previously tripped
/// a post-ISel `debug_assert!` in `pipeline.rs` ("Multi-block function
/// '…' has no successor edges"). Fixed under #447: the assertion is
/// relaxed — unreachable non-entry blocks are legal tMIR and the pipeline
/// simply encodes them (they are harmless after DCE).
#[test]
fn regression_multi_block_no_edges_panics() {
    use llvm2_codegen::pipeline::{OptLevel, Pipeline, PipelineConfig};

    // Build a 2-block function where both blocks end in Return.
    let mut module = TmirModule::new("_panic_fuzz_compile");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let mut func = TmirFunction::new(FuncId::new(0), "_fuzz_fn", ft_id, BlockId::new(0));

    let mk_block = |id: u32, vid: u32| TmirBlock {
        id: BlockId::new(id),
        params: vec![],
        body: vec![
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(0),
            })
            .with_result(ValueId::new(vid)),
            InstrNode::new(Inst::Return {
                values: vec![ValueId::new(vid)],
            }),
        ],
    };
    func.blocks = vec![mk_block(0, 1000), mk_block(1, 1001)];

    // Drive the full pipeline — must not panic. Any return value (Ok or
    // typed Err) is acceptable; only a panic would regress this fix.
    let (lir, _) = llvm2_lower::translate_function(&func, &module)
        .expect("adapter should accept multi-block all-Return");
    let pipeline = Pipeline::new(PipelineConfig {
        opt_level: OptLevel::O0,
        emit_debug: false,
        ..Default::default()
    });
    let _ = pipeline.compile_function(&lir);
}
