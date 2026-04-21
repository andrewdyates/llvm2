// llvm2-cli/tests/tmir_text_format.rs - Integration tests for .tmir text I/O (#413)
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Per `designs/2026-04-16-tmir-transport-architecture.md` Layer 3 and
// issue #413, LLVM2 now:
//   - accepts the human-readable `.tmir` text format as a debug input
//     (enabled via `--format=text` or the `.tmir` extension under
//     `--format=auto`).
//   - can emit `.tmir` text via `--emit-tmir <PATH>` for round-tripping.
//
// Upstream status: `tmir::Module`'s `Display` impl does NOT emit the
// `func_types` table; correspondingly, `tmir::parser::parse_module`
// does not rebuild it. This means a parsed `.tmir` module cannot yet
// be handed to LLVM2's lowering pass (see `crates/llvm2-lower/src/
// adapter.rs:450`, `module.func_types[func.ty.index()]`). The output
// path (Display / `--emit-tmir`) is fully functional today, and the
// parser/loader wiring is in place so that when upstream fixes the
// func_types round-trip, `--format=text` compilation will work with
// no LLVM2-side changes.
//
// These tests exercise the CLI binary end-to-end, concentrating on
// the output-side path that is working today:
//
//   1. `--emit-tmir <PATH>` writes parseable `.tmir` text from any input.
//   2. `--emit-tmir` rejects multi-input invocations.
//   3. Golden check: the printer output starts with the canonical
//      `; tMIR text format v1` header (catches unintentional format drift).
//   4. `--format=text` accepts a `.tmir` input through the loader
//      (parse succeeds). Full compilation is disabled until the
//      upstream func_types round-trip lands.
//   5. `--format=auto` detects `.tmir` by extension.

use std::path::PathBuf;
use std::process::Command;

use llvm2_codegen::pipeline::{encode_tmbc, encode_tmir_text, parse_tmir_text};
use tmir::{Module as TmirModule, Ty};
use tmir_build::ModuleBuilder;

/// Build a minimal `fn return_42() -> i64 { 42 }` tMIR module.
fn make_test_module() -> TmirModule {
    let mut mb = ModuleBuilder::new("cli_tmir_text_test");
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
fn scratch_dir(test_name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "llvm2_cli_tmir_{}_{}",
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

// ---------------------------------------------------------------------------
// Case 1: `.tmir` input under `--format=text` is loaded by the parser.
//
// End-to-end compilation of a parsed `.tmir` module is currently blocked
// on an upstream limitation: `tmir::Module`'s `Display` impl does not
// emit the `func_types` table, and the parser does not rebuild it, so
// `llvm2-lower::adapter::translate_signature` panics on `func_types[0]`.
// What we CAN verify today is that the CLI wires the loader correctly:
// the parse itself succeeds (we do not see the "failed to read tMIR
// module" error path), and the failure happens in the downstream
// lowering pass. Once upstream emits+parses `func_types`, this test
// can be tightened to `status.success()`.
// ---------------------------------------------------------------------------

#[test]
fn cli_format_text_reaches_lowering_pass() {
    let dir = scratch_dir("format_text");
    let tmir_path = dir.join("module.tmir");
    let out_path = dir.join("module.o");

    let module = make_test_module();
    let text = encode_tmir_text(&module);
    std::fs::write(&tmir_path, &text).expect("write .tmir");

    let output = Command::new(llvm2_bin())
        .arg("--format=text")
        .arg("-c")
        .arg("-o")
        .arg(&out_path)
        .arg(&tmir_path)
        .output()
        .expect("run llvm2");

    let stderr = String::from_utf8_lossy(&output.stderr);
    // If the parser itself rejected the file, we'd see this error from
    // `compile_one`. Its absence proves the loader path works.
    assert!(
        !stderr.contains("failed to read tMIR module"),
        "--format=text must NOT fail at the parser. stderr:\n{}",
        stderr
    );
    // Belt-and-braces: the observed downstream panic sits in adapter.rs
    // and will go away once upstream emits+parses the func_types table
    // in `.tmir` text. Verifying the message here keeps this test a
    // stable indicator of "we're at the upstream limit, not a new bug".
    // We tolerate either success (future upstream fix) or the specific
    // known panic. Any other failure mode is surfaced.
    if !output.status.success() {
        assert!(
            stderr.contains("adapter.rs") || stderr.contains("func_types") ||
            stderr.contains("index out of bounds"),
            "unexpected --format=text failure (not the upstream blocker). stderr:\n{}",
            stderr
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Case 2: `--format=auto` picks up `.tmir` by extension (same caveat).
// ---------------------------------------------------------------------------

#[test]
fn cli_format_auto_picks_up_tmir_extension() {
    let dir = scratch_dir("auto_tmir");
    let tmir_path = dir.join("module.tmir");
    let out_path = dir.join("module.o");

    let module = make_test_module();
    let text = encode_tmir_text(&module);
    std::fs::write(&tmir_path, &text).expect("write .tmir");

    let output = Command::new(llvm2_bin())
        .arg("--format=auto")
        .arg("-c")
        .arg("-o")
        .arg(&out_path)
        .arg(&tmir_path)
        .output()
        .expect("run llvm2");

    let stderr = String::from_utf8_lossy(&output.stderr);
    // The loader must NOT mis-detect `.tmir` as JSON (which would
    // produce a JSON parse error). It must be recognised as text
    // and successfully parsed.
    assert!(
        !stderr.contains("JSON error"),
        "--format=auto should recognise .tmir extension as text, not JSON. stderr:\n{}",
        stderr
    );
    assert!(
        !stderr.contains("failed to read tMIR module"),
        "--format=auto must NOT fail at the parser for .tmir input. stderr:\n{}",
        stderr
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Case 3: `--emit-tmir` dumps a parseable .tmir text module.
// ---------------------------------------------------------------------------

#[test]
fn cli_emit_tmir_round_trips_through_parser() {
    let dir = scratch_dir("emit_tmir");
    let tmbc_path = dir.join("module.tmbc");
    let emitted_tmir = dir.join("dumped.tmir");

    let module = make_test_module();
    let tmbc = encode_tmbc(&module).expect("encode tMBC");
    std::fs::write(&tmbc_path, &tmbc).expect("write tmbc");

    // Call llvm2 with --emit-tmir; don't need a compile to succeed,
    // but request -c so we don't try to link. The .tmir file should
    // be written before compilation, which is what we're checking.
    let output = Command::new(llvm2_bin())
        .arg("--emit-tmir")
        .arg(&emitted_tmir)
        .arg("-c")
        .arg("-o")
        .arg(dir.join("module.o"))
        .arg(&tmbc_path)
        .output()
        .expect("run llvm2");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "--emit-tmir should succeed. stderr: {}",
        stderr
    );
    assert!(
        emitted_tmir.exists(),
        "--emit-tmir should write {}",
        emitted_tmir.display()
    );

    // Round-trip: parse the emitted text back to a module.
    let text = std::fs::read_to_string(&emitted_tmir).expect("read emitted .tmir");
    let reparsed = parse_tmir_text(&text).expect("parse emitted .tmir");
    assert_eq!(reparsed.name, module.name, "module name round-trips");
    assert_eq!(
        reparsed.functions.len(),
        module.functions.len(),
        "function count round-trips"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Case 4 (golden): the printer output starts with the canonical header.
// ---------------------------------------------------------------------------

#[test]
fn tmir_display_golden_header_is_stable() {
    let module = make_test_module();
    let text = encode_tmir_text(&module);
    // If this line changes, it is a breaking change to the text format
    // and should be propagated to downstream debuggers intentionally.
    assert!(
        text.starts_with("; tMIR text format v1\n"),
        "expected canonical '; tMIR text format v1' header; got:\n{}",
        text.lines().take(3).collect::<Vec<_>>().join("\n")
    );
    // Sanity: module name appears in the dump.
    assert!(
        text.contains("\"cli_tmir_text_test\""),
        "expected module name in text dump. got:\n{}",
        text
    );
}

// ---------------------------------------------------------------------------
// Case 5: `--emit-tmir` rejects multi-input invocations.
// ---------------------------------------------------------------------------

#[test]
fn cli_emit_tmir_rejects_multiple_inputs() {
    let dir = scratch_dir("emit_tmir_multi");
    let a = dir.join("a.tmbc");
    let b = dir.join("b.tmbc");
    let out = dir.join("dumped.tmir");

    let module = make_test_module();
    let tmbc = encode_tmbc(&module).expect("encode tMBC");
    std::fs::write(&a, &tmbc).expect("write a");
    std::fs::write(&b, &tmbc).expect("write b");

    let output = Command::new(llvm2_bin())
        .arg("--emit-tmir")
        .arg(&out)
        .arg("-c")
        .arg(&a)
        .arg(&b)
        .output()
        .expect("run llvm2");

    assert!(
        !output.status.success(),
        "--emit-tmir with >1 input must fail"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--emit-tmir") && stderr.contains("one input"),
        "error should mention --emit-tmir and one input. stderr: {}",
        stderr
    );

    let _ = std::fs::remove_dir_all(&dir);
}
