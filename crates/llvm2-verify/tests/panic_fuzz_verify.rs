// llvm2-verify/tests/panic_fuzz_verify.rs
// Property-based panic-fuzz harness for the function-level verifier.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Part of #448 (#387 follow-up) / Part of #372 (Crash-free codegen).
//
// Reference: `designs/2026-04-18-crash-free-codegen-plan.md` §5 (proptest
// as primary defense) and §6 (per-crate harness).
//
// Contract under test: `FunctionVerifier::verify` / `verify_function` must
// either return a `FunctionVerificationReport` or raise a typed error — it
// must NEVER panic, abort, or overflow on any `MachFunction` shape. The
// verifier takes untrusted MachIR as input (post-ISel / post-regalloc),
// and any panic here is a correctness bug: a malformed function should
// surface as a skipped / unverified entry in the report, not kill the
// pipeline.
//
// Strategies here mirror the shape used by `panic_fuzz_compile` /
// `panic_fuzz_encode` so the generators share structure across crates.
//
// Run:
//   cargo test -p llvm2-verify --test panic_fuzz_verify
// Increase case count via env:
//   PROPTEST_CASES=100000 cargo test -p llvm2-verify --test panic_fuzz_verify

use llvm2_ir::function::{MachBlock, MachFunction, Signature, Type};
use llvm2_ir::inst::{AArch64Opcode, MachInst};
use llvm2_ir::operand::MachOperand;
use llvm2_ir::regs::{
    PReg, RegClass, SpecialReg, VReg, D0, D1, D2, D3, S0, S1, S2, S3, SP, W0, W1, W2, W3,
    W4, W5, W6, W7, X0, X1, X2, X3, X4, X5, X6, X7,
};
use llvm2_ir::types::{BlockId, InstId};
use llvm2_verify::function_verifier::{verify_function, FunctionVerifier};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Opcode strategy — covers every proof-category branch in
// `FunctionVerifier::opcode_to_proof_query` plus a handful of opcodes that
// map to `None` (unverified).
// ---------------------------------------------------------------------------

fn opcode_strategy() -> impl Strategy<Value = AArch64Opcode> {
    use AArch64Opcode::*;
    prop_oneof![
        // Arithmetic
        Just(AddRR), Just(AddRI), Just(SubRR), Just(SubRI),
        Just(MulRR), Just(Neg),
        // Division
        Just(SDiv), Just(UDiv),
        // Compare / NZCV
        Just(CmpRR), Just(CmpRI), Just(Tst),
        Just(CMPWrr), Just(CMPXrr), Just(CMPWri), Just(CMPXri),
        // Branch
        Just(BCond), Just(Bcc),
        // Memory
        Just(LdrRI), Just(StrRI), Just(LdrbRI), Just(StrbRI),
        Just(STRWui), Just(STRXui),
        // Floating-point
        Just(FaddRR), Just(FsubRR), Just(FmulRR), Just(FnegRR),
        // Peephole logical / shifts
        Just(AndRR), Just(AndRI), Just(OrrRR), Just(OrrRI),
        Just(LslRR), Just(LslRI), Just(LsrRR), Just(AsrRR),
        // Conditional select
        Just(CSet),
        // Pseudos (must be classified as Skipped — not a panic)
        Just(Phi), Just(Copy), Just(StackAlloc), Just(Nop),
        // Unmapped opcodes — exercise the `None` branch
        Just(Ret), Just(Br), Just(Bl), Just(Blr),
    ]
}

fn preg_strategy() -> impl Strategy<Value = PReg> {
    // Use known-valid physical registers (x0..x7, w0..w7, d0..d3, s0..s3).
    // PReg is an opaque encoding; we use the published constants to
    // guarantee every sample is a legal register.
    prop_oneof![
        Just(X0), Just(X1), Just(X2), Just(X3),
        Just(X4), Just(X5), Just(X6), Just(X7),
        Just(W0), Just(W1), Just(W2), Just(W3),
        Just(W4), Just(W5), Just(W6), Just(W7),
        Just(D0), Just(D1), Just(D2), Just(D3),
        Just(S0), Just(S1), Just(S2), Just(S3),
        Just(SP),
    ]
}

fn vreg_strategy() -> impl Strategy<Value = VReg> {
    (0u32..=32, 0u8..=3).prop_map(|(id, klass)| {
        let class = match klass {
            0 => RegClass::Gpr32,
            1 => RegClass::Gpr64,
            2 => RegClass::Fpr32,
            _ => RegClass::Fpr64,
        };
        VReg::new(id, class)
    })
}

fn operand_strategy() -> impl Strategy<Value = MachOperand> {
    prop_oneof![
        vreg_strategy().prop_map(MachOperand::VReg),
        preg_strategy().prop_map(MachOperand::PReg),
        any::<i64>().prop_map(MachOperand::Imm),
        any::<f64>().prop_map(MachOperand::FImm),
        (0u32..=8).prop_map(|b| MachOperand::Block(BlockId(b))),
        prop_oneof![
            Just(SpecialReg::SP),
            Just(SpecialReg::XZR),
            Just(SpecialReg::WZR),
        ]
        .prop_map(MachOperand::Special),
        "[_a-zA-Z][_a-zA-Z0-9]{0,16}".prop_map(MachOperand::Symbol),
        (preg_strategy(), any::<i64>())
            .prop_map(|(base, offset)| MachOperand::MemOp { base, offset }),
    ]
}

fn mach_inst_strategy() -> impl Strategy<Value = MachInst> {
    (
        opcode_strategy(),
        prop::collection::vec(operand_strategy(), 0..=4),
    )
        .prop_map(|(opcode, operands)| MachInst::new(opcode, operands))
}

// ---------------------------------------------------------------------------
// MachFunction construction
//
// We generate two shape axes:
//  1. A flat list of instructions, to drive the inst-id walk.
//  2. Per-block selections (indices into the flat list), to drive the
//     block walk. We intentionally allow out-of-range inst indices in the
//     per-block list — the verifier's inner loop has an explicit
//     `if idx >= func.insts.len() { continue; }` guard, and this harness
//     validates that guard stays in place.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct FuncSpec {
    insts: Vec<MachInst>,
    blocks: Vec<Vec<u32>>, // per-block raw inst indices (may be OOB)
    name: String,
    num_params: u8,
    num_returns: u8,
}

fn func_spec_strategy() -> impl Strategy<Value = FuncSpec> {
    (
        prop::collection::vec(mach_inst_strategy(), 0..=12),
        1usize..=4,
        "[_a-zA-Z][_a-zA-Z0-9]{0,16}",
        0u8..=3,
        0u8..=2,
    )
        .prop_flat_map(|(insts, num_blocks, name, num_params, num_returns)| {
            let max_idx = insts.len().max(1) as u32;
            // Allow block entries to reference indices up to 2x the inst
            // list length so we also fuzz the "OOB inst index" branch.
            let per_block = prop::collection::vec(0u32..(max_idx * 2 + 4), 0..=6);
            prop::collection::vec(per_block, num_blocks..=num_blocks).prop_map(
                move |blocks| FuncSpec {
                    insts: insts.clone(),
                    blocks,
                    name: name.clone(),
                    num_params,
                    num_returns,
                },
            )
        })
}

fn materialise(spec: &FuncSpec) -> MachFunction {
    let param_types = vec![Type::I64; spec.num_params as usize];
    let return_types = vec![Type::I64; spec.num_returns as usize];
    let mut func = MachFunction::new(spec.name.clone(), Signature::new(param_types, return_types));
    // Reset blocks (MachFunction::new pre-populates bb0).
    func.blocks.clear();
    func.block_order.clear();
    // Push raw instructions.
    for inst in &spec.insts {
        func.insts.push(inst.clone());
    }
    // Build blocks from the spec. Raw inst indices (even OOB ones) are
    // written directly; the verifier must tolerate them.
    for (bi, raw_idxs) in spec.blocks.iter().enumerate() {
        let mut block = MachBlock::new();
        for &raw in raw_idxs {
            block.insts.push(InstId(raw));
        }
        func.blocks.push(block);
        func.block_order.push(BlockId(bi as u32));
    }
    // Ensure at least one block exists — the verifier tolerates zero, but
    // several downstream consumers assume >= 1.
    if func.blocks.is_empty() {
        func.blocks.push(MachBlock::new());
        func.block_order.push(BlockId(0));
    }
    func.entry = BlockId(0);
    func
}

// ---------------------------------------------------------------------------
// Panic-catching wrapper
// ---------------------------------------------------------------------------

/// Run `verify_function` on a function in a scoped thread with a large
/// stack. The verifier's `ProofDatabase::new` path accumulates a very
/// large debug-build stack frame (see `proof_database.rs` §"issue #205"),
/// and proptest's many iterations can trip the default 8 MB test-thread
/// stack. The 64 MB stack here is purely an artefact of running in debug
/// under proptest; release builds fit easily in the default.
fn run_verify_in_big_stack<F: FnOnce() + Send + 'static>(f: F) -> std::thread::Result<()> {
    let handle = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            f();
        })
        .expect("spawn big-stack thread");
    handle.join()
}

fn assert_no_panic(func: MachFunction) {
    let name = func.name.clone();
    let nblocks = func.blocks.len();
    let ninsts = func.insts.len();
    let f = func;
    let result = run_verify_in_big_stack(move || {
        let _ = verify_function(&f);
    });
    if let Err(payload) = result {
        let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".to_string()
        };
        panic!(
            "verify_function panicked on '{}' ({} blocks, {} insts): {}",
            name, nblocks, ninsts, msg,
        );
    }
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig {
        // Verification walks the proof database on every instruction
        // with a proof mapping. 128 cases at ~10 insts × 4 blocks is
        // <30s locally; reduce via PROPTEST_CASES=32 while iterating.
        cases: std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64),
        max_shrink_iters: 200,
        .. ProptestConfig::default()
    })]

    /// Random MachFunction shapes, including malformed ones (OOB inst
    /// indices in block lists, unpaired operands, etc.). The verifier
    /// must produce a report or a typed error — never panic.
    #[test]
    fn verify_function_never_panics(spec in func_spec_strategy()) {
        let func = materialise(&spec);
        assert_no_panic(func);
    }

    /// Same generator, but drives `FunctionVerifier::verify` directly so
    /// a regression in the configurable path is distinguishable from a
    /// regression in the convenience wrapper.
    #[test]
    fn verifier_verify_never_panics(spec in func_spec_strategy()) {
        let func = materialise(&spec);
        let name = func.name.clone();
        let nblocks = func.blocks.len();
        let ninsts = func.insts.len();
        let f = func;
        let result = run_verify_in_big_stack(move || {
            let verifier = FunctionVerifier::new();
            let _ = verifier.verify(&f);
        });
        if let Err(payload) = result {
            let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<non-string panic payload>".to_string()
            };
            panic!(
                "FunctionVerifier::verify panicked on '{}' ({} blocks, {} insts): {}",
                name, nblocks, ninsts, msg,
            );
        }
    }

    /// Opcode-to-proof-query mapping must be total: every opcode either
    /// returns `Some((_, category))` or `None` — never panics. This path
    /// doesn't touch the proof database, so the default stack is fine.
    #[test]
    fn opcode_to_proof_query_never_panics(opcode in opcode_strategy()) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _ = FunctionVerifier::opcode_to_proof_query(opcode);
        }));
        prop_assert!(result.is_ok(), "opcode_to_proof_query panicked on {:?}", opcode);
    }
}

// ---------------------------------------------------------------------------
// Hand-written pin-downs
// ---------------------------------------------------------------------------

/// Empty function: no blocks, no insts. Must produce an empty report, not
/// a panic.
#[test]
fn verify_empty_function_ok() {
    let func = MachFunction::new("_empty".into(), Signature::new(vec![], vec![]));
    let report = verify_function(&func);
    assert_eq!(report.function_name, "_empty");
    assert_eq!(report.total(), 0);
}

/// Block that references out-of-range inst indices. The verifier's
/// `if idx >= func.insts.len() { continue; }` guard must keep this
/// non-panicking; out-of-range entries are silently skipped.
#[test]
fn verify_function_oob_inst_indices_ok() {
    let mut func = MachFunction::new("_oob".into(), Signature::new(vec![], vec![]));
    func.blocks.clear();
    func.block_order.clear();
    let mut block = MachBlock::new();
    // Reference three non-existent instructions.
    block.insts.push(InstId(0));
    block.insts.push(InstId(42));
    block.insts.push(InstId(u32::MAX));
    func.blocks.push(block);
    func.block_order.push(BlockId(0));
    func.entry = BlockId(0);
    let report = verify_function(&func);
    // All three OOB references are skipped; the report walks zero
    // instructions without panicking.
    assert_eq!(report.total(), 0);
}

/// Pseudo opcodes must be classified as `Skipped`, not panic and not map
/// through `opcode_to_proof_query`.
#[test]
fn verify_pseudo_inst_is_skipped() {
    let mut func = MachFunction::new("_pseudo".into(), Signature::new(vec![], vec![]));
    func.blocks.clear();
    func.block_order.clear();
    let mut block = MachBlock::new();
    func.insts.push(MachInst::new(AArch64Opcode::Nop, vec![]));
    block.insts.push(InstId(0));
    func.blocks.push(block);
    func.block_order.push(BlockId(0));
    func.entry = BlockId(0);
    let report = verify_function(&func);
    assert_eq!(report.total(), 1);
    assert_eq!(report.skipped_count(), 1);
}
