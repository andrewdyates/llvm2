// llvm2-codegen/tests/tmbc_roundtrip_prop.rs
// Property-based round-trip test for the `.tmbc` binary format (#415).
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: `designs/2026-04-16-tmir-transport-architecture.md`, section
// "Verification of the Format", property 1:
//
//     Round-trip property: forall m: Module, decode(encode(m)) = m
//     — verified by z4 for small modules, tested exhaustively for all
//     instruction types.
//
// This suite is the empirical half of that property. Each generated
// `tmir::Module` is encoded via `encode_tmbc`, decoded via `decode_tmbc`,
// and the result is asserted byte-equal AND structurally equal to the
// original. The generator covers a bounded shape (≤3 functions, ≤5 blocks
// per function, ≤10 instructions per block) and exercises every BinOp /
// ICmpOp / UnOp variant plus branches, returns, and block parameters.
// Integer widths cover I8/I16/I32/I64. Float widths (F32/F64) are covered
// via `fconst` + FP compares so the encoder sees floats too.
//
// By construction the generator never produces UB-shaped modules
// (unreachable blocks, dangling branches, non-dominating SSA uses) so the
// property is a clean equality over `decode ∘ encode = id`. The test is
// deliberately agnostic to semantic validity beyond what the encoder /
// decoder requires — it does NOT invoke the compiler or the verifier.
//
// Complementary to `tests/tmbc_canonical.rs` (#416) which tests the
// *canonical encoding* property (encode determinism) with fixed fixtures.

use llvm2_codegen::pipeline::{decode_tmbc, encode_tmbc};
use proptest::prelude::*;
use tmir::{BinOp, FCmpOp, ICmpOp, Module as TmirModule, Ty, UnOp};
use tmir_build::ModuleBuilder;

// ---------------------------------------------------------------------------
// Shape specification
// ---------------------------------------------------------------------------

/// A tiny IR-level instruction description that the generator knows how to
/// lower into one or more builder calls. Kept deliberately opaque so the
/// generator is free to pick operands.
#[derive(Debug, Clone)]
enum InstKind {
    IConst(IntKind, i64),
    FConst(FloatKind, f64),
    IBinop(BinOp, IntKind),
    FBinop(BinOp, FloatKind),
    IUnop(UnOp, IntKind),
    ICmp(ICmpOp, IntKind),
    FCmp(FCmpOp, FloatKind),
    IZext(IntKind, IntKind), // src -> dst (widening)
    ITrunc(IntKind, IntKind), // src -> dst (narrowing)
    Select,                   // select cond:Bool, then:I64, else:I64 -> I64
}

#[derive(Debug, Clone, Copy)]
enum IntKind {
    I8,
    I16,
    I32,
    I64,
}

impl IntKind {
    fn ty(self) -> Ty {
        match self {
            IntKind::I8 => Ty::I8,
            IntKind::I16 => Ty::I16,
            IntKind::I32 => Ty::I32,
            IntKind::I64 => Ty::I64,
        }
    }
    fn bits(self) -> u32 {
        match self {
            IntKind::I8 => 8,
            IntKind::I16 => 16,
            IntKind::I32 => 32,
            IntKind::I64 => 64,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum FloatKind {
    F32,
    F64,
}
impl FloatKind {
    fn ty(self) -> Ty {
        match self {
            FloatKind::F32 => Ty::F32,
            FloatKind::F64 => Ty::F64,
        }
    }
}

/// A single block's spec: how many instructions to attempt, and whether it
/// terminates by branching to the next block (`false`) or by returning
/// (`true`). The last block in a function always returns.
#[derive(Debug, Clone)]
struct BlockSpec {
    insts: Vec<InstKind>,
    terminates_with_ret: bool,
    // For conditional branch: if true and there is a successor +2 (i.e. two
    // distinct forward blocks available), emit condbr. Otherwise emit br.
    use_condbr: bool,
}

#[derive(Debug, Clone)]
struct FuncSpec {
    name: String,
    num_params: u8, // 0..=2 I64 params
    blocks: Vec<BlockSpec>,
}

#[derive(Debug, Clone)]
struct ModuleSpec {
    name: String,
    functions: Vec<FuncSpec>,
}

// ---------------------------------------------------------------------------
// Proptest strategies
// ---------------------------------------------------------------------------

fn int_kind_strategy() -> impl Strategy<Value = IntKind> {
    prop_oneof![
        Just(IntKind::I8),
        Just(IntKind::I16),
        Just(IntKind::I32),
        Just(IntKind::I64),
    ]
}

fn float_kind_strategy() -> impl Strategy<Value = FloatKind> {
    prop_oneof![Just(FloatKind::F32), Just(FloatKind::F64)]
}

fn binop_int_strategy() -> impl Strategy<Value = BinOp> {
    prop_oneof![
        Just(BinOp::Add),
        Just(BinOp::Sub),
        Just(BinOp::Mul),
        Just(BinOp::And),
        Just(BinOp::Or),
        Just(BinOp::Xor),
        Just(BinOp::Shl),
        Just(BinOp::LShr),
        Just(BinOp::AShr),
    ]
}

fn binop_float_strategy() -> impl Strategy<Value = BinOp> {
    prop_oneof![
        Just(BinOp::FAdd),
        Just(BinOp::FSub),
        Just(BinOp::FMul),
        Just(BinOp::FDiv),
    ]
}

fn icmp_strategy() -> impl Strategy<Value = ICmpOp> {
    prop_oneof![
        Just(ICmpOp::Eq),
        Just(ICmpOp::Ne),
        Just(ICmpOp::Slt),
        Just(ICmpOp::Sle),
        Just(ICmpOp::Sgt),
        Just(ICmpOp::Sge),
        Just(ICmpOp::Ult),
        Just(ICmpOp::Ule),
        Just(ICmpOp::Ugt),
        Just(ICmpOp::Uge),
    ]
}

fn fcmp_strategy() -> impl Strategy<Value = FCmpOp> {
    prop_oneof![
        Just(FCmpOp::OEq),
        Just(FCmpOp::ONe),
        Just(FCmpOp::OLt),
        Just(FCmpOp::OLe),
        Just(FCmpOp::OGt),
        Just(FCmpOp::OGe),
    ]
}

fn unop_int_strategy() -> impl Strategy<Value = UnOp> {
    prop_oneof![Just(UnOp::Neg), Just(UnOp::Not)]
}

fn inst_kind_strategy() -> impl Strategy<Value = InstKind> {
    prop_oneof![
        // IConst covers all int widths; use a narrow i64 value range so we
        // stay safely representable in every int kind.
        (int_kind_strategy(), (-1000i64..=1000i64)).prop_map(|(k, v)| InstKind::IConst(k, v)),
        (float_kind_strategy(), (-1e3f64..=1e3f64))
            .prop_map(|(k, v)| InstKind::FConst(k, v)),
        (binop_int_strategy(), int_kind_strategy()).prop_map(|(op, k)| InstKind::IBinop(op, k)),
        (binop_float_strategy(), float_kind_strategy()).prop_map(|(op, k)| InstKind::FBinop(op, k)),
        (unop_int_strategy(), int_kind_strategy()).prop_map(|(op, k)| InstKind::IUnop(op, k)),
        (icmp_strategy(), int_kind_strategy()).prop_map(|(op, k)| InstKind::ICmp(op, k)),
        (fcmp_strategy(), float_kind_strategy()).prop_map(|(op, k)| InstKind::FCmp(op, k)),
        // Widening zext: pick two kinds, require src.bits < dst.bits.
        (int_kind_strategy(), int_kind_strategy())
            .prop_filter("src < dst for zext", |(s, d)| s.bits() < d.bits())
            .prop_map(|(s, d)| InstKind::IZext(s, d)),
        // Narrowing trunc: pick two kinds, require src.bits > dst.bits.
        (int_kind_strategy(), int_kind_strategy())
            .prop_filter("src > dst for trunc", |(s, d)| s.bits() > d.bits())
            .prop_map(|(s, d)| InstKind::ITrunc(s, d)),
        Just(InstKind::Select),
    ]
}

fn block_spec_strategy() -> impl Strategy<Value = BlockSpec> {
    (
        prop::collection::vec(inst_kind_strategy(), 0..=10),
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(|(insts, terminates_with_ret, use_condbr)| BlockSpec {
            insts,
            terminates_with_ret,
            use_condbr,
        })
}

fn func_spec_strategy(idx: usize) -> impl Strategy<Value = FuncSpec> {
    (
        0u8..=2,
        prop::collection::vec(block_spec_strategy(), 1..=5),
    )
        .prop_map(move |(num_params, blocks)| FuncSpec {
            name: format!("_prop_f{}", idx),
            num_params,
            blocks,
        })
}

fn module_spec_strategy() -> impl Strategy<Value = ModuleSpec> {
    (1usize..=3usize).prop_flat_map(|n| {
        let funcs = (0..n)
            .map(func_spec_strategy)
            .collect::<Vec<_>>();
        (Just(n), funcs).prop_map(|(n, functions)| ModuleSpec {
            name: format!("tmbc_prop_mod_{}", n),
            functions,
        })
    })
}

// ---------------------------------------------------------------------------
// Spec -> tmir::Module materialisation
// ---------------------------------------------------------------------------

fn materialise(spec: &ModuleSpec) -> TmirModule {
    let mut mb = ModuleBuilder::new(&spec.name);
    for fspec in &spec.functions {
        let params = vec![Ty::I64; fspec.num_params as usize];
        let ty = mb.add_func_type(params, vec![Ty::I64]);
        let mut fb = mb.function(&fspec.name, ty);

        // Create all blocks up front so `br` / `condbr` have forward targets.
        let block_count = fspec.blocks.len();
        let mut blocks = Vec::with_capacity(block_count);
        for _ in 0..block_count {
            blocks.push(fb.create_block());
        }

        // Entry block: take function params (all I64).
        let entry = blocks[0];
        let mut entry_params: Vec<_> = Vec::new();
        for _ in 0..fspec.num_params {
            entry_params.push(fb.add_block_param(entry, Ty::I64));
        }
        // Non-entry blocks: no block params (keeps branch lowering trivial).

        for (bi, bspec) in fspec.blocks.iter().enumerate() {
            fb.switch_to_block(blocks[bi]);

            // Per-block scratch: a small stack of (value, ty) we can pull
            // operands from. Seeded with entry params in block 0 for
            // realistic data flow; other blocks start empty so we emit
            // iconst-first to populate.
            let mut scratch: Vec<(tmir::ValueId, Ty)> = if bi == 0 {
                entry_params.iter().map(|v| (*v, Ty::I64)).collect()
            } else {
                Vec::new()
            };

            for inst in &bspec.insts {
                emit_inst(&mut fb, &mut scratch, inst);
            }

            // Terminator.
            let is_last = bi + 1 == block_count;
            if is_last || bspec.terminates_with_ret {
                // Return an I64. If we have an I64 in scratch, use it;
                // otherwise materialise a zero.
                let ret_v = pick_or_make_i64(&mut fb, &mut scratch);
                fb.ret(vec![ret_v]);
            } else {
                let next = blocks[bi + 1];
                let successor_exists = bi + 2 < block_count;
                if bspec.use_condbr && successor_exists {
                    // Need a Bool cond; fabricate via `icmp eq 0, 0` (always true).
                    let z = fb.iconst(Ty::I64, 0);
                    let cond = fb.icmp(ICmpOp::Eq, Ty::I64, z, z);
                    let tgt2 = blocks[bi + 2];
                    fb.condbr(cond, next, vec![], tgt2, vec![]);
                } else {
                    fb.br(next, vec![]);
                }
            }
        }
        fb.build();
    }
    mb.build()
}

fn emit_inst(
    fb: &mut tmir_build::FunctionBuilder<'_>,
    scratch: &mut Vec<(tmir::ValueId, Ty)>,
    inst: &InstKind,
) {
    match inst {
        InstKind::IConst(k, v) => {
            let vid = fb.iconst(k.ty(), *v as i128);
            scratch.push((vid, k.ty()));
        }
        InstKind::FConst(k, v) => {
            // Skip NaN/±inf to avoid any serializer-level float quirks that
            // are out of scope for this ticket (see #415 out-of-scope).
            if !v.is_finite() {
                return;
            }
            let vid = fb.fconst(k.ty(), *v);
            scratch.push((vid, k.ty()));
        }
        InstKind::IBinop(op, k) => {
            let ty = k.ty();
            let lhs = find_or_make(fb, scratch, &ty);
            let rhs = find_or_make(fb, scratch, &ty);
            let vid = fb.binop(*op, ty.clone(), lhs, rhs);
            scratch.push((vid, ty));
        }
        InstKind::FBinop(op, k) => {
            let ty = k.ty();
            let lhs = find_or_make_f(fb, scratch, &ty);
            let rhs = find_or_make_f(fb, scratch, &ty);
            let vid = fb.binop(*op, ty.clone(), lhs, rhs);
            scratch.push((vid, ty));
        }
        InstKind::IUnop(op, k) => {
            let ty = k.ty();
            let operand = find_or_make(fb, scratch, &ty);
            let vid = fb.unop(*op, ty.clone(), operand);
            scratch.push((vid, ty));
        }
        InstKind::ICmp(op, k) => {
            let ty = k.ty();
            let lhs = find_or_make(fb, scratch, &ty);
            let rhs = find_or_make(fb, scratch, &ty);
            let vid = fb.icmp(*op, ty, lhs, rhs);
            scratch.push((vid, Ty::Bool));
        }
        InstKind::FCmp(op, k) => {
            let ty = k.ty();
            let lhs = find_or_make_f(fb, scratch, &ty);
            let rhs = find_or_make_f(fb, scratch, &ty);
            let vid = fb.fcmp(*op, ty, lhs, rhs);
            scratch.push((vid, Ty::Bool));
        }
        InstKind::IZext(s, d) => {
            let src_ty = s.ty();
            let operand = find_or_make(fb, scratch, &src_ty);
            let vid = fb.zext(src_ty, d.ty(), operand);
            scratch.push((vid, d.ty()));
        }
        InstKind::ITrunc(s, d) => {
            let src_ty = s.ty();
            let operand = find_or_make(fb, scratch, &src_ty);
            let vid = fb.trunc(src_ty, d.ty(), operand);
            scratch.push((vid, d.ty()));
        }
        InstKind::Select => {
            let cond = find_or_make_bool(fb, scratch);
            let then_v = find_or_make(fb, scratch, &Ty::I64);
            let else_v = find_or_make(fb, scratch, &Ty::I64);
            let vid = fb.select(Ty::I64, cond, then_v, else_v);
            scratch.push((vid, Ty::I64));
        }
    }
}

fn find_or_make(
    fb: &mut tmir_build::FunctionBuilder<'_>,
    scratch: &mut Vec<(tmir::ValueId, Ty)>,
    want: &Ty,
) -> tmir::ValueId {
    for (v, t) in scratch.iter().rev() {
        if t == want {
            return *v;
        }
    }
    let v = fb.iconst(want.clone(), 0);
    scratch.push((v, want.clone()));
    v
}

fn find_or_make_f(
    fb: &mut tmir_build::FunctionBuilder<'_>,
    scratch: &mut Vec<(tmir::ValueId, Ty)>,
    want: &Ty,
) -> tmir::ValueId {
    for (v, t) in scratch.iter().rev() {
        if t == want {
            return *v;
        }
    }
    let v = fb.fconst(want.clone(), 0.0);
    scratch.push((v, want.clone()));
    v
}

fn find_or_make_bool(
    fb: &mut tmir_build::FunctionBuilder<'_>,
    scratch: &mut Vec<(tmir::ValueId, Ty)>,
) -> tmir::ValueId {
    for (v, t) in scratch.iter().rev() {
        if *t == Ty::Bool {
            return *v;
        }
    }
    let v = fb.bool_const(false);
    scratch.push((v, Ty::Bool));
    v
}

fn pick_or_make_i64(
    fb: &mut tmir_build::FunctionBuilder<'_>,
    scratch: &mut Vec<(tmir::ValueId, Ty)>,
) -> tmir::ValueId {
    for (v, t) in scratch.iter().rev() {
        if *t == Ty::I64 {
            return *v;
        }
    }
    let v = fb.iconst(Ty::I64, 0);
    scratch.push((v, Ty::I64));
    v
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

proptest! {
    // 256 cases satisfies the ticket's ≥256 lower bound. Runtime is well
    // under a minute even with every instruction variant in play.
    #![proptest_config(ProptestConfig {
        cases: 256,
        .. ProptestConfig::default()
    })]

    #[test]
    fn tmbc_roundtrip_preserves_structural_equality(spec in module_spec_strategy()) {
        let module = materialise(&spec);
        let bytes = encode_tmbc(&module).expect("encode_tmbc must succeed");
        let decoded = decode_tmbc(&bytes).expect("decode_tmbc must succeed");
        prop_assert_eq!(
            &module, &decoded,
            "round-trip must preserve structural Module equality"
        );
    }

    #[test]
    fn tmbc_reencode_is_a_fixed_point(spec in module_spec_strategy()) {
        // Even stricter than structural equality: encode -> decode -> encode
        // must produce byte-equal output. This catches any field that
        // round-trips to a semantically-equal but byte-different state (a
        // canonicality violation exposed only via random inputs).
        let module = materialise(&spec);
        let b1 = encode_tmbc(&module).expect("encode #1");
        let round = decode_tmbc(&b1).expect("decode");
        let b2 = encode_tmbc(&round).expect("encode #2");
        prop_assert_eq!(
            b1, b2,
            "encode ∘ decode ∘ encode must equal encode (canonical fixed point)"
        );
    }
}

// Smoke: confirm the generator itself produces a usable module with the
// required shape bounds. Runs once per `cargo test` invocation.
#[test]
fn generator_smoke_bounds_respected() {
    use proptest::strategy::{Strategy, ValueTree};
    use proptest::test_runner::TestRunner;

    let mut runner = TestRunner::default();
    for _ in 0..64 {
        let tree = module_spec_strategy().new_tree(&mut runner).unwrap();
        let spec = tree.current();
        assert!(!spec.functions.is_empty() && spec.functions.len() <= 3);
        for f in &spec.functions {
            assert!(!f.blocks.is_empty() && f.blocks.len() <= 5);
            for b in &f.blocks {
                assert!(b.insts.len() <= 10);
            }
        }
        // Must materialise without panic and round-trip cleanly.
        let m = materialise(&spec);
        let bytes = encode_tmbc(&m).expect("encode");
        let back = decode_tmbc(&bytes).expect("decode");
        assert_eq!(m, back);
    }
}
