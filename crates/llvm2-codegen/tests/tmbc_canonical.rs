// llvm2-codegen/tests/tmbc_canonical.rs
// Canonical encoding determinism tests for the `.tmbc` binary format (#416).
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: `designs/2026-04-16-tmir-transport-architecture.md`, section
// "Verification of the Format", property 2:
//
//     Canonical encoding: forall m: Module, encode(m) produces a unique
//     byte sequence — no padding, no alignment-dependent output, no
//     hash-map iteration order dependency.
//
// These tests assert byte-for-byte equality of `encode_tmbc` output across:
//   (1) two modules built via the exact same builder sequence (baseline
//       determinism — same process, same run);
//   (2) two modules built via sequences that introduce extra intermediate
//       state churn (ordering-sensitivity probe);
//   (3) an encode -> decode -> encode round-trip (encoder is a fixed point
//       of the decode composition).
//
// Audit note (2026-04-19): inspection of upstream `tmir` crate (rev
// 67f1fdcffa3ff0c020f3f89bcb47496b7ece90d8) confirms that `Module`,
// `Function`, `Block`, and `InstrNode` store all collections as `Vec<_>`.
// No `HashMap`/`HashSet` appear in any `#[derive(Serialize)]` struct field
// (the only `HashSet` uses live inside `#[cfg(test)]` blocks in
// `tmir/src/inst.rs` and `tmir/src/proof.rs`). Therefore rmp-serde output
// is expected to be byte-deterministic. This test enforces that invariant
// as a regression guard so a future upstream change cannot silently
// reintroduce iteration-order dependency.

use llvm2_codegen::pipeline::{decode_tmbc, encode_tmbc};
use tmir::{Global, Linkage, Module as TmirModule, Ty};
use tmir_build::ModuleBuilder;

/// Build `fn return_42() -> i64 { 42 }`.
fn make_const_module(name: &str) -> TmirModule {
    let mut mb = ModuleBuilder::new(name);
    let ty = mb.add_func_type(vec![], vec![Ty::I64]);
    let mut fb = mb.function("_return_42", ty);
    let entry = fb.create_block();
    fb.switch_to_block(entry);
    let r = fb.iconst(Ty::I64, 42);
    fb.ret(vec![r]);
    fb.build();
    mb.build()
}

/// Build `fn add(a, b) -> i64 { a + b }` with two globals `G0`, `G1`
/// appended in positional order.
fn make_module_with_globals(name: &str) -> TmirModule {
    let mut mb = ModuleBuilder::new(name);
    let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
    let mut fb = mb.function("_add", ty);
    let entry = fb.create_block();
    let a = fb.add_block_param(entry, Ty::I64);
    let b = fb.add_block_param(entry, Ty::I64);
    fb.switch_to_block(entry);
    let sum = fb.add(Ty::I64, a, b);
    fb.ret(vec![sum]);
    fb.build();
    mb.add_global(Global {
        name: "G0".into(),
        ty: Ty::I64,
        mutable: false,
        initializer: None,
        linkage: Linkage::External,
    });
    mb.add_global(Global {
        name: "G1".into(),
        ty: Ty::I64,
        mutable: true,
        initializer: None,
        linkage: Linkage::External,
    });
    mb.build()
}

// ---------------------------------------------------------------------------
// (1) Baseline: same builder sequence, two constructions => identical bytes.
// ---------------------------------------------------------------------------

#[test]
fn encode_tmbc_is_deterministic_const_module() {
    let m1 = make_const_module("tmbc_canon_const");
    let m2 = make_const_module("tmbc_canon_const");
    let b1 = encode_tmbc(&m1).expect("encode m1");
    let b2 = encode_tmbc(&m2).expect("encode m2");
    assert_eq!(
        b1, b2,
        "encode_tmbc must be byte-deterministic for structurally-equal modules"
    );
}

#[test]
fn encode_tmbc_is_deterministic_module_with_globals() {
    let m1 = make_module_with_globals("tmbc_canon_globals");
    let m2 = make_module_with_globals("tmbc_canon_globals");
    let b1 = encode_tmbc(&m1).expect("encode m1");
    let b2 = encode_tmbc(&m2).expect("encode m2");
    assert_eq!(
        b1, b2,
        "encode_tmbc must be byte-deterministic for modules with globals"
    );
}

// ---------------------------------------------------------------------------
// (2) Churn probe: build one module, encode, drop, rebuild in a fresh
//     process-local allocator context, re-encode. Any HashMap-backed
//     container with a random seed (ASLR-randomized DefaultHasher) would
//     diverge across constructions inside the same process run — rmp-serde
//     iterates containers in `Serialize` order, so Vec<_> is safe but
//     HashMap<_,_> is not.
// ---------------------------------------------------------------------------

#[test]
fn encode_tmbc_stable_across_many_reconstructions() {
    let reference = encode_tmbc(&make_module_with_globals("tmbc_canon_churn"))
        .expect("encode reference");
    for i in 0..16 {
        // Allocate and drop noise to jitter allocator state between builds.
        let _scratch: Vec<String> = (0..i + 1).map(|k| format!("noise-{}-{}", i, k)).collect();
        let m = make_module_with_globals("tmbc_canon_churn");
        let b = encode_tmbc(&m).expect("encode iteration");
        assert_eq!(
            reference, b,
            "iteration {i}: encode_tmbc diverged from reference encoding"
        );
    }
}

// ---------------------------------------------------------------------------
// (3) Round-trip canonicality: encode -> decode -> encode is a fixed point.
//     If any field round-trips to a semantically-equal but
//     byte-non-equivalent state (e.g. reordering), this catches it.
// ---------------------------------------------------------------------------

#[test]
fn encode_decode_encode_is_fixed_point() {
    let m = make_module_with_globals("tmbc_canon_fixed_point");
    let b1 = encode_tmbc(&m).expect("encode #1");
    let round = decode_tmbc(&b1).expect("decode");
    let b2 = encode_tmbc(&round).expect("encode #2");
    assert_eq!(
        b1, b2,
        "encode . decode . encode must equal encode (canonical fixed point)"
    );
}
