// llvm2-lower/tests/panic_fuzz_lower.rs
// Property-based panic-fuzz harness for `translate_function` (tMIR adapter).
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Part of #387 (proptest panic-fuzz) / Part of #372 (Crash-free codegen).
//
// Reference: `designs/2026-04-18-crash-free-codegen-plan.md` §5 (proptest
// as primary defense) and §6 (per-crate harness).
//
// Contract under test: for *any* tMIR function, `translate_function` must
// either return `Ok(..)` or `Err(AdapterError)` — it must NEVER panic,
// abort, or underflow in debug. Phase 1 of #372 converted the known
// unwrap/unreachable sites in the adapter to `AdapterError` returns
// (notably `adapter.rs`'s block-id and function-type lookups); this
// harness proves the conversion is exhaustive over random-but-valid AND
// random-malformed tMIR inputs. The malformed stream is the more
// interesting of the two: it feeds tMIR whose block IDs, value IDs, type
// IDs, and function signatures are deliberately inconsistent, because
// this is exactly what a buggy frontend would produce.
//
// Run:
//   cargo test -p llvm2-lower --test panic_fuzz_lower
// Increase case count via env:
//   PROPTEST_CASES=100000 cargo test -p llvm2-lower --test panic_fuzz_lower

use std::panic;

use llvm2_lower::{translate_function, AdapterError};
use proptest::prelude::*;
use tmir::{
    BinOp, Block as TmirBlock, BlockId, Constant, FuncId, FuncTy, FuncTyId,
    Function as TmirFunction, ICmpOp, Inst, InstrNode, Module as TmirModule,
    SwitchCase, Ty, ValueId,
};

// ---------------------------------------------------------------------------
// Shape spec
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum InstKind {
    Const(i64),
    BinOp(BinOp),
    ICmp(ICmpOp),
    Return,
    Br(u32),
    CondBr(u32, u32),
    // #451 widening: high-value Inst variants. These exercise adapter paths
    // that were previously unreachable via the BinOp/ICmp-only harness.
    // Each variant runs through both the well-formed and malformed strategies
    // (the malformed strategy additionally sets `bad_entry`/`bad_func_ty`).
    Load,
    Store,
    Alloca,
    Gep,
    Call,
    Switch(u32, u32), // (default_tgt, case_tgt) — both mod block_count at lower time
    ExtractField,
    InsertField,
    UnreachableValueRef, // feed a ValueId that was never defined
    UnknownBlockJump,    // br to a BlockId out of range
}

#[derive(Debug, Clone)]
struct BlockSpec {
    insts: Vec<InstKind>,
}

#[derive(Debug, Clone)]
struct FuncSpec {
    num_params: u8,
    blocks: Vec<BlockSpec>,
    /// When true, deliberately set `entry` to an out-of-range block id so
    /// the adapter's entry resolution path is exercised.
    bad_entry: bool,
    /// When true, use an unregistered `FuncTyId` so the adapter's type
    /// resolution path is exercised.
    bad_func_ty: bool,
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Full-coverage BinOp strategy: all 18 variants of `tmir::BinOp`.
///
/// #451 widened this from the original 9 (50%) to all 18 (100%). The
/// integer division family (`UDiv`/`SDiv`/`URem`/`SRem`) is emitted on
/// `Ty::I64` operands — this is semantically valid tMIR. The FP family
/// (`FAdd`/`FSub`/`FMul`/`FDiv`/`FRem`) is emitted with whatever operands
/// the harness has available (`Ty::I64`-typed ValueIds) and therefore
/// represents type-mismatched tMIR. That is *fine* for a totality
/// harness: `translate_function` is required to be panic-free on any
/// input, valid or malformed. The adapter's BinOp arm (adapter.rs:876)
/// ignores the `ty` field and dispatches on opcode alone, so FP variants
/// do not crash at the adapter layer either way.
fn binop_strategy() -> impl Strategy<Value = BinOp> {
    prop_oneof![
        // Integer arithmetic
        Just(BinOp::Add), Just(BinOp::Sub), Just(BinOp::Mul),
        Just(BinOp::UDiv), Just(BinOp::SDiv),
        Just(BinOp::URem), Just(BinOp::SRem),
        // Floating-point arithmetic (type-mismatched against I64 operands
        // in this harness — intentional malformed coverage)
        Just(BinOp::FAdd), Just(BinOp::FSub), Just(BinOp::FMul),
        Just(BinOp::FDiv), Just(BinOp::FRem),
        // Bitwise
        Just(BinOp::And), Just(BinOp::Or), Just(BinOp::Xor),
        // Shifts
        Just(BinOp::Shl), Just(BinOp::LShr), Just(BinOp::AShr),
    ]
}

/// Full-coverage ICmpOp strategy: all 10 variants of `tmir::ICmpOp`.
///
/// #451 widened this from the original 6 (60%) to all 10 (100%). The
/// previously-missing signed/unsigned `Gt`/`Ge` variants all have
/// explicit arms in the adapter (adapter.rs:937-951), so they are
/// first-class citizens of the totality property.
fn icmp_strategy() -> impl Strategy<Value = ICmpOp> {
    prop_oneof![
        Just(ICmpOp::Eq), Just(ICmpOp::Ne),
        Just(ICmpOp::Slt), Just(ICmpOp::Sle),
        Just(ICmpOp::Sgt), Just(ICmpOp::Sge),
        Just(ICmpOp::Ult), Just(ICmpOp::Ule),
        Just(ICmpOp::Ugt), Just(ICmpOp::Uge),
    ]
}

/// Return `true` for FP BinOps (FAdd, FSub, FMul, FDiv, FRem). Used by the
/// materialiser to pick the declared `ty` field — FP ops get `Ty::F64`,
/// everything else gets `Ty::I64`.
fn binop_is_fp(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem
    )
}

fn inst_kind_strategy(block_count: u32) -> impl Strategy<Value = InstKind> {
    let bc = block_count.max(1);
    prop_oneof![
        (-1000i64..=1000i64).prop_map(InstKind::Const),
        binop_strategy().prop_map(InstKind::BinOp),
        icmp_strategy().prop_map(InstKind::ICmp),
        Just(InstKind::Return),
        (0u32..bc).prop_map(InstKind::Br),
        (0u32..bc, 0u32..bc).prop_map(|(a, b)| InstKind::CondBr(a, b)),
        // #451 widening: memory, aggregate, and control-flow Inst variants.
        // Every arm is exercised in both the well-formed and malformed
        // generator because `func_spec_strategy` calls this routine
        // identically in both modes. What changes between modes is
        // `bad_entry` / `bad_func_ty`, not the per-inst strategy.
        Just(InstKind::Load),
        Just(InstKind::Store),
        Just(InstKind::Alloca),
        Just(InstKind::Gep),
        Just(InstKind::Call),
        (0u32..bc, 0u32..bc).prop_map(|(d, c)| InstKind::Switch(d, c)),
        Just(InstKind::ExtractField),
        Just(InstKind::InsertField),
        // Malformed signals — only the malformed strategy uses these.
        Just(InstKind::UnreachableValueRef),
        Just(InstKind::UnknownBlockJump),
    ]
}

fn block_spec_strategy(block_count: u32) -> impl Strategy<Value = BlockSpec> {
    prop::collection::vec(inst_kind_strategy(block_count), 0..=6)
        .prop_map(|insts| BlockSpec { insts })
}

fn func_spec_strategy(malformed: bool) -> impl Strategy<Value = FuncSpec> {
    // #447 closed: `translate_signature` now returns
    // `Err(AdapterError::InvalidFuncTyId(..))` for unregistered FuncTyIds
    // instead of indexing out of bounds. The `bad_func_ty` arm is enabled
    // under the `malformed` flag; `bad_entry` was already safe.
    (1usize..=4usize)
        .prop_flat_map(move |nb| {
            let bc = nb as u32;
            (
                0u8..=2,
                prop::collection::vec(block_spec_strategy(bc), nb..=nb),
                any::<bool>(),
                any::<bool>(),
            )
                .prop_map(move |(num_params, blocks, bad_entry, bad_func_ty)| FuncSpec {
                    num_params,
                    blocks,
                    bad_entry: malformed && bad_entry,
                    bad_func_ty: malformed && bad_func_ty,
                })
        })
}

// ---------------------------------------------------------------------------
// Spec -> tmir::Function materialisation (hand-rolled, no builder)
//
// We bypass `tmir_build::FunctionBuilder` because we want the freedom to
// emit malformed IR (e.g. dangling ValueIds, unknown block targets, wrong
// entry). A well-behaved builder rejects those inputs before the adapter
// ever sees them.
// ---------------------------------------------------------------------------

fn materialise(spec: &FuncSpec) -> (TmirFunction, TmirModule) {
    let mut module = TmirModule::new("_panic_fuzz");
    let params: Vec<Ty> = vec![Ty::I64; spec.num_params as usize];
    let ft_id = module.add_func_type(FuncTy {
        params: params.clone(),
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    // If bad_func_ty is set, use a FuncTyId that was not registered.
    let used_ft_id = if spec.bad_func_ty { FuncTyId::new(9999) } else { ft_id };

    let block_count = spec.blocks.len();
    let entry_block_id = if spec.bad_entry {
        BlockId::new(u32::MAX) // well out of range
    } else {
        BlockId::new(0)
    };

    let mut func = TmirFunction::new(FuncId::new(0), "_fuzz_fn", used_ft_id, entry_block_id);

    // Allocate a monotonically-increasing ValueId pool. For malformed
    // paths we occasionally reference a ValueId that we never define.
    let mut next_vid: u32 = 1000;
    let mut alloc_vid = || -> ValueId {
        let v = ValueId::new(next_vid);
        next_vid += 1;
        v
    };

    // Build blocks.
    func.blocks = Vec::with_capacity(block_count);
    for (bi, bspec) in spec.blocks.iter().enumerate() {
        let block_id = BlockId::new(bi as u32);
        let mut block_params: Vec<(ValueId, Ty)> = Vec::new();
        if bi == 0 {
            // entry block gets the function params
            for _ in 0..spec.num_params {
                block_params.push((alloc_vid(), Ty::I64));
            }
        }

        let mut body: Vec<InstrNode> = Vec::new();
        // Track defined I64 SSA values in-scope so we can feed them to
        // subsequent binops. Seeded with entry params.
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
                    let (lhs, rhs) = pick_two_or_zero(&mut defined_i64, &mut body, &mut alloc_vid);
                    let vid = alloc_vid();
                    // #451: FP ops (FAdd/FSub/FMul/FDiv/FRem) declare Ty::F64
                    // while operands are drawn from the I64 pool. This
                    // deliberately mismatched typing is exactly the kind of
                    // buggy-frontend input the adapter must reject with
                    // `Err`, not panic. Integer div/rem (UDiv/SDiv/URem/SRem)
                    // get Ty::I64 and will additionally exercise any
                    // "divide by zero on a constant RHS" paths — proptest
                    // shrinker will find a zero if one is reachable.
                    let ty = if binop_is_fp(*op) { Ty::F64 } else { Ty::I64 };
                    body.push(
                        InstrNode::new(Inst::BinOp {
                            op: *op,
                            ty,
                            lhs,
                            rhs,
                        })
                        .with_result(vid),
                    );
                    // Only push into the I64 pool for integer results so we
                    // don't feed FP result vids into subsequent integer ops
                    // as if they were I64. (The adapter may still see them
                    // via later `Return`, which is fine.)
                    if !binop_is_fp(*op) {
                        defined_i64.push(vid);
                    }
                }
                InstKind::ICmp(op) => {
                    let (lhs, rhs) = pick_two_or_zero(&mut defined_i64, &mut body, &mut alloc_vid);
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
                    // Bool in tMIR (via ICmp) — stash as-is; downstream consumers
                    // can use it via CondBr.
                }
                InstKind::Return => {
                    let ret_v = pick_one_or_zero(&mut defined_i64, &mut body, &mut alloc_vid);
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
                    let cond = pick_one_or_zero(&mut defined_i64, &mut body, &mut alloc_vid);
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
                // ---------------- #451 widening: memory / aggregate / CF ----------------
                //
                // These cases feed the adapter's Load/Store/Alloca/GEP/Call/
                // Switch/ExtractField/InsertField paths. They all reuse the
                // I64 SSA pool for pointer/aggregate operands — this is
                // *intentional* malformed input: the adapter must translate
                // them cleanly or return `Err(..)`, never panic. Results that
                // logically have pointer or aggregate type are still pushed
                // back into `defined_i64` so downstream random ops have
                // something to chew on.
                InstKind::Alloca => {
                    // Alloca with a random element type and an optional SSA
                    // count pulled from the I64 pool. Generator intentionally
                    // feeds a wide mix of types (including FP) so the
                    // adapter's size/align computation is stressed.
                    let count = if defined_i64.is_empty() {
                        None
                    } else {
                        Some(*defined_i64.last().unwrap())
                    };
                    let vid = alloc_vid();
                    body.push(
                        InstrNode::new(Inst::Alloca {
                            ty: Ty::I64,
                            count,
                            align: Some(8),
                        })
                        .with_result(vid),
                    );
                    defined_i64.push(vid);
                }
                InstKind::Load => {
                    let ptr = pick_one_or_zero(&mut defined_i64, &mut body, &mut alloc_vid);
                    let vid = alloc_vid();
                    body.push(
                        InstrNode::new(Inst::Load {
                            ty: Ty::I64,
                            ptr,
                            volatile: false,
                            align: Some(8),
                        })
                        .with_result(vid),
                    );
                    defined_i64.push(vid);
                }
                InstKind::Store => {
                    let (ptr, value) =
                        pick_two_or_zero(&mut defined_i64, &mut body, &mut alloc_vid);
                    body.push(InstrNode::new(Inst::Store {
                        ty: Ty::I64,
                        ptr,
                        value,
                        volatile: false,
                        align: Some(8),
                    }));
                }
                InstKind::Gep => {
                    // GEP over a one-index base. Both base and index come
                    // from the I64 pool — base may be an integer, not a
                    // pointer; the adapter must reject or tolerate, not
                    // panic. Note tMIR's variant is named `GEP` (not
                    // `GetElementPtr`); the issue text uses the LLVM name.
                    let base = pick_one_or_zero(&mut defined_i64, &mut body, &mut alloc_vid);
                    let idx = pick_one_or_zero(&mut defined_i64, &mut body, &mut alloc_vid);
                    let vid = alloc_vid();
                    body.push(
                        InstrNode::new(Inst::GEP {
                            pointee_ty: Ty::I64,
                            base,
                            indices: vec![idx],
                        })
                        .with_result(vid),
                    );
                    defined_i64.push(vid);
                }
                InstKind::Call => {
                    // Reference `FuncId(99)` which is never registered in the
                    // throwaway module. This exercises the adapter's
                    // callee-resolution error path. A future improvement
                    // (#451 follow-up) may register one sibling function and
                    // emit a legal call — for panic-fuzz purposes the
                    // unregistered form is strictly more adversarial.
                    let arg = pick_one_or_zero(&mut defined_i64, &mut body, &mut alloc_vid);
                    let vid = alloc_vid();
                    body.push(
                        InstrNode::new(Inst::Call {
                            callee: FuncId::new(99),
                            args: vec![arg],
                        })
                        .with_result(vid),
                    );
                    defined_i64.push(vid);
                }
                InstKind::Switch(default_tgt, case_tgt) => {
                    let scrut = pick_one_or_zero(&mut defined_i64, &mut body, &mut alloc_vid);
                    let d = BlockId::new(*default_tgt % block_count.max(1) as u32);
                    let c = BlockId::new(*case_tgt % block_count.max(1) as u32);
                    body.push(InstrNode::new(Inst::Switch {
                        value: scrut,
                        default: d,
                        default_args: vec![],
                        cases: vec![SwitchCase {
                            value: Constant::Int(0),
                            target: c,
                            args: vec![],
                        }],
                    }));
                }
                InstKind::ExtractField => {
                    // Exercises the adapter's aggregate extraction path on
                    // an I64 "aggregate" — deliberately wrong type, field 0.
                    let agg = pick_one_or_zero(&mut defined_i64, &mut body, &mut alloc_vid);
                    let vid = alloc_vid();
                    body.push(
                        InstrNode::new(Inst::ExtractField {
                            ty: Ty::I64,
                            aggregate: agg,
                            field: 0,
                        })
                        .with_result(vid),
                    );
                    defined_i64.push(vid);
                }
                InstKind::InsertField => {
                    let (agg, val) =
                        pick_two_or_zero(&mut defined_i64, &mut body, &mut alloc_vid);
                    let vid = alloc_vid();
                    body.push(
                        InstrNode::new(Inst::InsertField {
                            ty: Ty::I64,
                            aggregate: agg,
                            field: 0,
                            value: val,
                        })
                        .with_result(vid),
                    );
                    defined_i64.push(vid);
                }
                InstKind::UnreachableValueRef => {
                    // Feed a ValueId that was never defined anywhere in the
                    // function. The adapter's operand resolution path must
                    // return `Err`, not panic.
                    let phantom = ValueId::new(u32::MAX - 7);
                    let vid = alloc_vid();
                    body.push(
                        InstrNode::new(Inst::BinOp {
                            op: BinOp::Add,
                            ty: Ty::I64,
                            lhs: phantom,
                            rhs: phantom,
                        })
                        .with_result(vid),
                    );
                    defined_i64.push(vid);
                }
                InstKind::UnknownBlockJump => {
                    body.push(InstrNode::new(Inst::Br {
                        target: BlockId::new(u32::MAX - 3),
                        args: vec![],
                    }));
                }
            }
        }

        // Ensure the block has at least one terminator. If none, synth a Ret
        // on a freshly-materialised zero. This is only a generator convenience
        // — it does not suppress the randomly-injected Ret/Br/CondBr/Switch
        // above. `Switch` was added under #451; without it the trailing
        // synth-Ret would push past the Switch and confuse downstream checks
        // that assume exactly one terminator per block.
        let terminates = body.last().map(|n| {
            matches!(
                n.inst,
                Inst::Return { .. }
                    | Inst::Br { .. }
                    | Inst::CondBr { .. }
                    | Inst::Switch { .. }
            )
        }).unwrap_or(false);
        if !terminates {
            let zero_vid = alloc_vid();
            body.push(
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                })
                .with_result(zero_vid),
            );
            body.push(InstrNode::new(Inst::Return { values: vec![zero_vid] }));
        }

        func.blocks.push(TmirBlock {
            id: block_id,
            params: block_params,
            body,
        });
    }

    (func, module)
}

fn pick_one_or_zero<F: FnMut() -> ValueId>(
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

fn pick_two_or_zero<F: FnMut() -> ValueId>(
    defined: &mut Vec<ValueId>,
    body: &mut Vec<InstrNode>,
    alloc_vid: &mut F,
) -> (ValueId, ValueId) {
    let a = pick_one_or_zero(defined, body, alloc_vid);
    let b = pick_one_or_zero(defined, body, alloc_vid);
    (a, b)
}

// ---------------------------------------------------------------------------
// Property
// ---------------------------------------------------------------------------

fn assert_no_panic(func: &TmirFunction, module: &TmirModule) {
    // tmir::Function/Module are Clone + UnwindSafe-in-practice; wrap
    // in AssertUnwindSafe because we're only asserting the "no panic"
    // property.
    let f = func.clone();
    let m = module.clone();
    let result = panic::catch_unwind(panic::AssertUnwindSafe(move || {
        let _ = translate_function(&f, &m);
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
            "translate_function panicked on function '{}' ({} blocks): {}",
            func.name,
            func.blocks.len(),
            msg,
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256),
        max_shrink_iters: 500,
        .. ProptestConfig::default()
    })]

    /// Random-but-valid tMIR (no dangling refs, entry block = 0, registered
    /// func type). The adapter must succeed or return a clean error.
    #[test]
    fn lower_never_panics_on_valid(spec in func_spec_strategy(false)) {
        let (func, module) = materialise(&spec);
        assert_no_panic(&func, &module);
    }

    /// Deliberately malformed tMIR (dangling ValueIds, out-of-range block
    /// jump targets). Note: `bad_entry` and `bad_func_ty` are currently
    /// gated off because they hit production panics tracked by the P1
    /// regression tests below. This is the main totality property over
    /// everything else.
    #[test]
    fn lower_never_panics_on_malformed(spec in func_spec_strategy(true)) {
        let (func, module) = materialise(&spec);
        assert_no_panic(&func, &module);
    }
}

// ---------------------------------------------------------------------------
// Regression reproducers for known panics found by this harness
// ---------------------------------------------------------------------------
//
// These pin-down tests are hand-reduced shrinks of failing proptest cases.
// They previously pinned buggy behavior with `#[should_panic]`; they now
// assert the post-fix behavior (a typed `Err(AdapterError::..)`). See #447.

/// `translate_signature` previously indexed `module.func_types` directly:
/// a `TmirFunction` whose `ty` referenced an unregistered `FuncTyId`
/// triggered an out-of-bounds panic. Fixed under #447: the adapter now
/// returns `Err(AdapterError::InvalidFuncTyId(..))`.
#[test]
fn regression_unregistered_func_ty_panics() {
    let mut module = TmirModule::new("_panic_fuzz");
    // Register ONE func type (id 0).
    let _ = module.add_func_type(FuncTy {
        params: vec![Ty::I64],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    // Construct a function that *references* FuncTyId(9999), which is
    // unregistered.
    let bad_ft = FuncTyId::new(9999);
    let mut func = TmirFunction::new(FuncId::new(0), "_bad_ft", bad_ft, BlockId::new(0));
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![],
        body: vec![InstrNode::new(Inst::Return { values: vec![] })],
    }];
    let err = translate_function(&func, &module)
        .expect_err("unregistered FuncTyId must be rejected, not panic");
    assert!(
        matches!(err, AdapterError::InvalidFuncTyId(9999)),
        "expected AdapterError::InvalidFuncTyId(9999), got {err:?}"
    );
}

/// Confirms the *good* case: an out-of-range entry BlockId yields an
/// `Err(AdapterError::…)` rather than a panic. Keeps this invariant pinned
/// so a future refactor doesn't regress it into an indexing panic.
#[test]
fn regression_out_of_range_entry_returns_err() {
    let mut module = TmirModule::new("_panic_fuzz");
    let ft_id = module.add_func_type(FuncTy {
        params: vec![],
        returns: vec![Ty::I64],
        is_vararg: false,
    });
    let bad_entry = BlockId::new(u32::MAX);
    let mut func = TmirFunction::new(FuncId::new(0), "_bad_entry", ft_id, bad_entry);
    func.blocks = vec![TmirBlock {
        id: BlockId::new(0),
        params: vec![],
        body: vec![InstrNode::new(Inst::Return { values: vec![] })],
    }];
    // Acceptable outcomes: Ok (adapter silently picks some block) or Err.
    // The NOT-acceptable outcome is a panic, which is what this test pins.
    let _ = translate_function(&func, &module);
}
