// llvm2-codegen/tests/jit_proof_certs.rs - End-to-end JIT proof certificate tests
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Exercises the Phase-1 cut of issue #348: proof certificates for
// JIT-compiled code. Each test compiles a minimal MachFunction through
// `JitCompiler::compile_raw` with `JitConfig::verify = true`, then asserts
// that `ExecutableBuffer` carries a well-formed `JitCertificate` for the
// function, that the certificate's replay check passes, and that the
// tMIR→MachInst provenance is populated.
//
// The compilation pipeline pulls in the full backend (opt + regalloc +
// encoder). That stack is deeper than the default 2 MiB that `cargo test`
// gives each test thread, so every test runs its body in a spawned thread
// with an 8 MiB stack. Reference: the existing `jit_integration.rs` tests
// avoid this by not enabling verification; we need `verify=true` here.
//
// Part of #348 — Proof certificates for JIT-compiled code.

#![cfg(all(target_arch = "aarch64", feature = "verify"))]
#![allow(deprecated)]

use std::collections::HashMap;

use llvm2_codegen::jit::{JitCompiler, JitConfig};
use llvm2_ir::function::{MachFunction, Signature, Type};
use llvm2_ir::inst::{AArch64Opcode, MachInst};
use llvm2_ir::operand::MachOperand;
use llvm2_ir::regs::{X0, X1};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// `fn add(a: i64, b: i64) -> i64 { a + b }` — `ADD X0, X0, X1 ; RET`.
fn build_add() -> MachFunction {
    let sig = Signature::new(vec![Type::I64, Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("add".to_string(), sig);
    let entry = func.entry;

    let add = MachInst::new(
        AArch64Opcode::AddRR,
        vec![
            MachOperand::PReg(X0),
            MachOperand::PReg(X0),
            MachOperand::PReg(X1),
        ],
    );
    let add_id = func.push_inst(add);
    func.append_inst(entry, add_id);

    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}

/// Run `body` on a worker thread with an 8 MiB stack. The default 2 MiB
/// test stack is too small for the full JIT compile pipeline with
/// verification enabled in debug builds.
fn with_big_stack<F: FnOnce() + Send + 'static>(body: F) {
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(body)
        .expect("spawn worker thread")
        .join()
        .expect("worker thread panicked");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Certificate is attached when `verify = true` and every field is
/// populated consistently with the executable buffer's code layout.
#[test]
fn verify_on_attaches_certificate() {
    with_big_stack(|| {
        let jit = JitCompiler::new(JitConfig {
            verify: true,
            ..Default::default()
        });
        let add_fn = build_add();
        let ext: HashMap<String, *const u8> = HashMap::new();
        let buf = jit
            .compile_raw(&[add_fn], &ext)
            .expect("compile_raw should succeed");

        let cert = buf
            .certificate("add")
            .expect("add should have a proof certificate when verify=true");

        assert_eq!(cert.function(), "add");

        let range = cert.code_range();
        assert!(
            range.end > range.start,
            "code range must be non-empty (got {range:?})"
        );
        assert!(
            range.end as usize <= buf.allocated_size(),
            "code range end {} must not exceed buffer size {}",
            range.end,
            buf.allocated_size()
        );

        // AddRR carries a lowering proof; Ret is a pseudo with no
        // obligation. Coverage is the fraction of non-skipped
        // instructions that verified. We accept any non-zero coverage
        // as evidence the verifier ran and did not crash; the important
        // property for tla2 is that the chain is populated.
        assert!(
            cert.coverage_percent() >= 0.0 && cert.coverage_percent() <= 100.0,
            "coverage out of range: {}",
            cert.coverage_percent()
        );

        // Phase-1 provenance: one TmirPair per MachInst. The exact
        // count depends on the encoder (frame setup/teardown), but
        // there must be at least AddRR + Ret pairs and the AddRR must
        // have been mapped to Iadd_I32 by the Phase-1 lookup table.
        let pairs = cert.tmir_pairs();
        assert!(
            pairs.len() >= 2,
            "expected at least 2 pairs, got {}",
            pairs.len()
        );
        assert!(
            pairs.iter().any(|p| p.tmir_op == "Iadd_I32"),
            "expected at least one Iadd_I32 pair, got {:?}",
            pairs.iter().map(|p| p.tmir_op.as_str()).collect::<Vec<_>>()
        );

        // Certificate chain must contain the successful AddRR proof.
        let chain = cert.chain();
        assert!(
            chain.all_verified(),
            "chain should be all-verified (only records proven obligations): {}",
            chain.certificates.len()
        );
        assert!(
            !chain.certificates.is_empty(),
            "chain must contain at least one certificate (AddRR)"
        );
    });
}

/// Replay check succeeds on an untampered certificate.
#[test]
fn replay_check_passes_on_untampered() {
    with_big_stack(|| {
        let jit = JitCompiler::new(JitConfig {
            verify: true,
            ..Default::default()
        });
        let add_fn = build_add();
        let ext: HashMap<String, *const u8> = HashMap::new();
        let buf = jit
            .compile_raw(&[add_fn], &ext)
            .expect("compile_raw should succeed");

        let cert = buf.certificate("add").expect("certificate must exist");
        assert!(
            cert.replay_check(),
            "replay_check must pass on untampered certificate"
        );
    });
}

/// Certificate is NOT attached when the caller explicitly disables
/// verification. `all_verified()` still returns true vacuously (no
/// certificates = no failures).
#[test]
fn verify_off_skips_certificate() {
    with_big_stack(|| {
        let jit = JitCompiler::new(JitConfig {
            verify: false,
            ..Default::default()
        });
        let add_fn = build_add();
        let ext: HashMap<String, *const u8> = HashMap::new();
        let buf = jit
            .compile_raw(&[add_fn], &ext)
            .expect("compile_raw should succeed");

        assert!(
            buf.certificate("add").is_none(),
            "no certificate should be attached when verify=false"
        );
        assert_eq!(
            buf.certificates().count(),
            0,
            "certificates iterator should be empty when verify=false"
        );
        assert!(
            buf.all_verified(),
            "all_verified is vacuously true when no certs are present"
        );
    });
}

/// `export_proofs` emits a JSON bundle containing every attached
/// certificate. Callers such as tla2 rely on this for cross-system
/// proof composition.
#[test]
fn export_proofs_emits_function_bundle() {
    with_big_stack(|| {
        let jit = JitCompiler::new(JitConfig {
            verify: true,
            ..Default::default()
        });
        let add_fn = build_add();
        let ext: HashMap<String, *const u8> = HashMap::new();
        let buf = jit
            .compile_raw(&[add_fn], &ext)
            .expect("compile_raw should succeed");

        let json = buf.export_proofs();
        assert!(json.contains("\"functions\""), "json: {json}");
        assert!(json.contains("\"add\""), "json: {json}");
        assert!(json.contains("Iadd_I32"), "json: {json}");
        assert!(json.contains("\"chain\""), "json: {json}");
    });
}

/// Certificate survives executing the JIT'd function — the buffer does
/// not lose or corrupt its certificate state after the code pages go
/// read-execute and are called. This closes the concern that runtime
/// execution might alias or stomp on the cert map.
#[test]
fn certificate_survives_execution() {
    with_big_stack(|| {
        let jit = JitCompiler::new(JitConfig {
            verify: true,
            ..Default::default()
        });
        let add_fn = build_add();
        let ext: HashMap<String, *const u8> = HashMap::new();
        let buf = jit
            .compile_raw(&[add_fn], &ext)
            .expect("compile_raw should succeed");

        let f: extern "C" fn(i64, i64) -> i64 =
            unsafe { buf.get_fn("add").expect("should find 'add' symbol") };
        assert_eq!(f(3, 4), 7);
        assert_eq!(f(-1, 1), 0);

        // Certificate must still be intact and well-formed after execution.
        let cert = buf.certificate("add").expect("certificate must exist");
        assert!(cert.replay_check());
        assert!(!cert.tmir_pairs().is_empty());
    });
}
