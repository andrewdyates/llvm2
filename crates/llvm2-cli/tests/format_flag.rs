// llvm2-cli/tests/format_flag.rs - Integration tests for the --format flag (#414)
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Per `designs/2026-04-16-tmir-transport-architecture.md` Layer 4 and
// issue #414, the CLI defaults to binary `.tmbc` input. JSON is retained
// only as `--format=json`. These tests exercise the CLI binary end-to-end
// for the three acceptance-criteria cases:
//
//   1. A `.tmbc` file with no flag compiles successfully (default).
//   2. A `.json` file without `--format=json` errors out with a message
//      naming the `--format=json` escape hatch.
//   3. A `.json` file with `--format=json` compiles successfully.
//
// A fourth test covers the legacy `--format=auto` mode for backcompat.

use std::path::PathBuf;
use std::process::Command;

use llvm2_codegen::pipeline::encode_tmbc;
use tmir::{Module as TmirModule, Ty};
use tmir_build::ModuleBuilder;

/// Build a minimal `fn return_42() -> i64 { 42 }` tMIR module.
fn make_test_module() -> TmirModule {
    let mut mb = ModuleBuilder::new("cli_format_flag_test");
    let ty = mb.add_func_type(vec![], vec![Ty::I64]);
    let mut fb = mb.function("_return_42", ty);
    let entry = fb.create_block();
    fb.switch_to_block(entry);
    let r = fb.iconst(Ty::I64, 42);
    fb.ret(vec![r]);
    fb.build();
    mb.build()
}

/// Create a fresh, empty scratch directory under the OS temp dir.
///
/// Uses `process::id` + a test-name suffix so parallel `cargo test`
/// invocations do not collide.
fn scratch_dir(test_name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "llvm2_cli_format_{}_{}",
        test_name,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Path to the compiled `llvm2` binary for this test run.
fn llvm2_bin() -> PathBuf {
    // `CARGO_BIN_EXE_<name>` is injected by cargo for integration tests
    // of packages that declare `[[bin]]`s.
    PathBuf::from(env!("CARGO_BIN_EXE_llvm2"))
}

// ---------------------------------------------------------------------------
// Case 1: binary default works with no flag.
// ---------------------------------------------------------------------------

#[test]
fn cli_binary_default_accepts_tmbc() {
    let dir = scratch_dir("binary_default");
    let tmbc_path = dir.join("module.tmbc");
    let out_path = dir.join("module.o");

    let module = make_test_module();
    let tmbc = encode_tmbc(&module).expect("encode tMBC");
    std::fs::write(&tmbc_path, &tmbc).expect("write tmbc");

    let status = Command::new(llvm2_bin())
        .arg("-c")
        .arg("-o")
        .arg(&out_path)
        .arg(&tmbc_path)
        .output()
        .expect("run llvm2");

    let stderr = String::from_utf8_lossy(&status.stderr);
    assert!(
        status.status.success(),
        "binary-default tMBC compile should succeed. stderr: {}",
        stderr
    );
    assert!(
        out_path.exists(),
        "expected object file at {} after successful compile",
        out_path.display()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Case 2: JSON file without --format=json errors clearly.
// ---------------------------------------------------------------------------

#[test]
fn cli_binary_default_rejects_json_with_hint() {
    let dir = scratch_dir("binary_rejects_json");
    let json_path = dir.join("module.json");
    let out_path = dir.join("module.o");

    let module = make_test_module();
    let json = serde_json::to_string_pretty(&module).expect("serialize JSON");
    std::fs::write(&json_path, json).expect("write json");

    let output = Command::new(llvm2_bin())
        .arg("-c")
        .arg("-o")
        .arg(&out_path)
        .arg(&json_path)
        .output()
        .expect("run llvm2");

    assert!(
        !output.status.success(),
        "JSON without --format=json must fail under the new default"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--format=json"),
        "error message must reference --format=json as the escape hatch.\n\
         actual stderr:\n{}",
        stderr
    );
    assert!(
        stderr.contains("tMBC") || stderr.contains("binary"),
        "error message must explain that binary is the new default.\n\
         actual stderr:\n{}",
        stderr
    );
    assert!(
        !out_path.exists(),
        "no object file should be produced on failed load; found {}",
        out_path.display()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Case 3: JSON file with --format=json works.
// ---------------------------------------------------------------------------

#[test]
fn cli_format_json_accepts_json_input() {
    let dir = scratch_dir("format_json");
    let json_path = dir.join("module.json");
    let out_path = dir.join("module.o");

    let module = make_test_module();
    let json = serde_json::to_string_pretty(&module).expect("serialize JSON");
    std::fs::write(&json_path, json).expect("write json");

    let output = Command::new(llvm2_bin())
        .arg("--format=json")
        .arg("-c")
        .arg("-o")
        .arg(&out_path)
        .arg(&json_path)
        .output()
        .expect("run llvm2");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "--format=json with a JSON file should succeed. stderr: {}",
        stderr
    );
    assert!(
        out_path.exists(),
        "expected object file at {} after successful JSON compile",
        out_path.display()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Case 4 (bonus): --format=auto restores legacy extension/magic sniffing.
// ---------------------------------------------------------------------------

#[test]
fn cli_format_auto_accepts_both() {
    let dir = scratch_dir("format_auto");
    let json_path = dir.join("module.json");
    let tmbc_path = dir.join("module.tmbc");
    let out_json = dir.join("json.o");
    let out_tmbc = dir.join("tmbc.o");

    let module = make_test_module();
    std::fs::write(
        &json_path,
        serde_json::to_string_pretty(&module).expect("serialize JSON"),
    )
    .expect("write json");
    std::fs::write(
        &tmbc_path,
        encode_tmbc(&module).expect("encode tMBC"),
    )
    .expect("write tmbc");

    // JSON via --format=auto.
    let out_j = Command::new(llvm2_bin())
        .arg("--format=auto")
        .arg("-c")
        .arg("-o")
        .arg(&out_json)
        .arg(&json_path)
        .output()
        .expect("run llvm2 (auto, json)");
    assert!(
        out_j.status.success(),
        "--format=auto should accept .json files. stderr: {}",
        String::from_utf8_lossy(&out_j.stderr)
    );

    // .tmbc via --format=auto.
    let out_b = Command::new(llvm2_bin())
        .arg("--format=auto")
        .arg("-c")
        .arg("-o")
        .arg(&out_tmbc)
        .arg(&tmbc_path)
        .output()
        .expect("run llvm2 (auto, tmbc)");
    assert!(
        out_b.status.success(),
        "--format=auto should accept .tmbc files. stderr: {}",
        String::from_utf8_lossy(&out_b.stderr)
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Case 5 (bonus): deprecated --input-json still works and emits a warning.
// ---------------------------------------------------------------------------

#[test]
fn cli_input_json_is_deprecated_alias() {
    let dir = scratch_dir("input_json_alias");
    let json_path = dir.join("module.json");
    let out_path = dir.join("module.o");

    let module = make_test_module();
    std::fs::write(
        &json_path,
        serde_json::to_string_pretty(&module).expect("serialize JSON"),
    )
    .expect("write json");

    let output = Command::new(llvm2_bin())
        .arg("--input-json")
        .arg(&json_path)
        .arg("-c")
        .arg("-o")
        .arg(&out_path)
        .output()
        .expect("run llvm2 (--input-json)");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "--input-json must still work as a deprecated alias. stderr: {}",
        stderr
    );
    assert!(
        stderr.contains("deprecated") && stderr.contains("--format=json"),
        "deprecation warning should mention --format=json. stderr: {}",
        stderr
    );

    let _ = std::fs::remove_dir_all(&dir);
}
