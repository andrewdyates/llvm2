// llvm2-cli/tests/emit_proofs_flag.rs - Integration tests for --emit-proofs=<dir> (#421)
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Exercises the CLI binary end-to-end to verify that:
//   1. Passing `--emit-proofs=<dir>` writes at least one `.smt2` + `.cert` pair.
//   2. Omitting the flag writes no proof files.
//   3. `--help` documents the flag.
//
// Layout under <dir> is `<ProofCategory>/<proof_name>.{smt2,cert}` per
// issue #421 / epic #407, task 6.

use std::path::PathBuf;
use std::process::Command;

use llvm2_codegen::pipeline::encode_tmbc;
use tmir::{Module as TmirModule, Ty};
use tmir_build::ModuleBuilder;

/// Build `fn add_two(a: i32, b: i32) -> i32 { a + b }` so the compiler
/// emits at least one verified Arithmetic-category lowering (IADD_I32 ->
/// ADDWrr) plus ReturnValue / ControlFlow proofs.
fn make_test_module() -> TmirModule {
    let mut mb = ModuleBuilder::new("emit_proofs_flag_test");
    let ty = mb.add_func_type(vec![Ty::I32, Ty::I32], vec![Ty::I32]);
    let mut fb = mb.function("_add_two", ty);
    let entry = fb.create_block();
    let a = fb.add_block_param(entry, Ty::I32);
    let b = fb.add_block_param(entry, Ty::I32);
    fb.switch_to_block(entry);
    let r = fb.add(Ty::I32, a, b);
    fb.ret(vec![r]);
    fb.build();
    mb.build()
}

fn scratch_dir(test_name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "llvm2_cli_emit_proofs_{}_{}",
        test_name,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn llvm2_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_llvm2"))
}

/// Recursively collect files under `root` that end with `ext`.
fn find_files_with_ext(root: &std::path::Path, ext: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(p) = stack.pop() {
        let entries = match std::fs::read_dir(&p) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|s| s.to_str()) == Some(ext) {
                out.push(path);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Case 1: --emit-proofs=<dir> writes at least one smt2+cert pair.
// ---------------------------------------------------------------------------

#[test]
fn cli_emit_proofs_writes_smt2_and_cert_pairs() {
    let dir = scratch_dir("writes_pairs");
    let tmbc_path = dir.join("module.tmbc");
    let out_path = dir.join("module.o");
    let proofs_dir = dir.join("proofs");

    let module = make_test_module();
    let tmbc = encode_tmbc(&module).expect("encode tMBC");
    std::fs::write(&tmbc_path, &tmbc).expect("write tmbc");

    let output = Command::new(llvm2_bin())
        .arg("-c")
        .arg("-o")
        .arg(&out_path)
        .arg(format!("--emit-proofs={}", proofs_dir.display()))
        .arg(&tmbc_path)
        .output()
        .expect("run llvm2");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "compile with --emit-proofs should succeed. stderr: {}",
        stderr
    );
    assert!(proofs_dir.exists(), "proofs directory must be created");

    let smt2_files = find_files_with_ext(&proofs_dir, "smt2");
    let cert_files = find_files_with_ext(&proofs_dir, "cert");

    assert!(
        !smt2_files.is_empty(),
        "at least one .smt2 file should be written. stderr: {}",
        stderr
    );
    assert!(
        !cert_files.is_empty(),
        "at least one .cert file should be written. stderr: {}",
        stderr
    );

    // Verify layout: each file lives under <dir>/<Category>/<name>.ext
    for smt2 in &smt2_files {
        let rel = smt2.strip_prefix(&proofs_dir).expect("rel path");
        let components: Vec<_> = rel.components().collect();
        assert_eq!(
            components.len(),
            2,
            ".smt2 files must live under <dir>/<Category>/<name>.smt2, found {:?}",
            rel
        );
    }

    // Verify the .smt2 content looks like SMT-LIB2.
    let sample_smt2 = std::fs::read_to_string(&smt2_files[0]).expect("read smt2");
    assert!(
        sample_smt2.contains("(set-logic") && sample_smt2.contains("(check-sat)"),
        "smt2 file should be a complete SMT-LIB2 query. content:\n{}",
        sample_smt2
    );

    // Verify the .cert content is valid-looking JSON with the required keys.
    let sample_cert = std::fs::read_to_string(&cert_files[0]).expect("read cert");
    for key in &[
        "\"result\"",
        "\"solver\"",
        "\"timestamp\"",
        "\"hash\"",
        "\"proof_name\"",
    ] {
        assert!(
            sample_cert.contains(key),
            "cert file missing key {}. content:\n{}",
            key,
            sample_cert
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Case 2: no flag => no proof files written.
// ---------------------------------------------------------------------------

#[test]
fn cli_no_flag_writes_no_proof_files() {
    let dir = scratch_dir("no_flag");
    let tmbc_path = dir.join("module.tmbc");
    let out_path = dir.join("module.o");
    let proofs_dir = dir.join("proofs_unused");

    let module = make_test_module();
    let tmbc = encode_tmbc(&module).expect("encode tMBC");
    std::fs::write(&tmbc_path, &tmbc).expect("write tmbc");

    let output = Command::new(llvm2_bin())
        .arg("-c")
        .arg("-o")
        .arg(&out_path)
        .arg(&tmbc_path)
        .output()
        .expect("run llvm2");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "plain compile should succeed. stderr: {}",
        stderr
    );
    assert!(
        !proofs_dir.exists(),
        "no proofs directory should be created without --emit-proofs"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Case 3: --help documents the flag.
// ---------------------------------------------------------------------------

#[test]
fn cli_help_documents_emit_proofs_flag() {
    let output = Command::new(llvm2_bin())
        .arg("--help")
        .output()
        .expect("run llvm2 --help");
    assert!(output.status.success(), "--help should succeed");
    let help = String::from_utf8_lossy(&output.stdout);
    assert!(
        help.contains("--emit-proofs"),
        "--help must mention --emit-proofs. stdout:\n{}",
        help
    );
    assert!(
        help.contains("DIR"),
        "--help must show --emit-proofs accepts a directory. stdout:\n{}",
        help
    );
}
