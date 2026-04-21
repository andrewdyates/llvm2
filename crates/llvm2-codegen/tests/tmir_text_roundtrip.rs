// llvm2-codegen/tests/tmir_text_roundtrip.rs
// Round-trip tests for the `.tmir` text format (#413).
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Verifies that the pipeline helpers `encode_tmir_text` /
// `parse_tmir_text` / `save_module_to_tmir_text` / `load_module_as(..,
// FormatMode::Text)` behave correctly for representative modules,
// matching the upstream `tmir` Display/parser round-trip contract.
//
// The upstream round-trip contract is:
//     parse(display(m)) == m          (structural equality on fields)
//     display(parse(display(m))) == display(m)   (Display is canonical)
//
// See `designs/2026-04-16-tmir-transport-architecture.md` Layer 3.

use llvm2_codegen::pipeline::{
    detect_input_format, encode_tmir_text, load_module_as, load_module_from_bytes,
    parse_tmir_text, save_module_to_tmir_text, FormatMode, InputFormat,
};
use tmir::{Module as TmirModule, Ty};
use tmir_build::ModuleBuilder;

/// Build a minimal `fn return_42() -> i64 { 42 }` module.
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

/// Build a module with a trivial `fn add(a, b) -> i64 { a + b }`.
fn make_add_module() -> TmirModule {
    let mut mb = ModuleBuilder::new("text_roundtrip_add");
    let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
    let mut fb = mb.function("_add", ty);
    let entry = fb.create_block();
    let a = fb.add_block_param(entry, Ty::I64);
    let b = fb.add_block_param(entry, Ty::I64);
    fb.switch_to_block(entry);
    let sum = fb.add(Ty::I64, a, b);
    fb.ret(vec![sum]);
    fb.build();
    mb.build()
}

// ---------------------------------------------------------------------------
// Round-trip: display -> parse -> display is idempotent.
// ---------------------------------------------------------------------------

#[test]
fn display_parse_display_is_idempotent_const() {
    let m = make_const_module("text_roundtrip_const");
    let text1 = encode_tmir_text(&m);
    let parsed = parse_tmir_text(&text1).expect("parse #1");
    let text2 = encode_tmir_text(&parsed);
    assert_eq!(
        text1, text2,
        "Display output must be canonical (fixed point of parse)"
    );
}

#[test]
fn display_parse_display_is_idempotent_add() {
    let m = make_add_module();
    let text1 = encode_tmir_text(&m);
    let parsed = parse_tmir_text(&text1).expect("parse #1");
    let text2 = encode_tmir_text(&parsed);
    assert_eq!(
        text1, text2,
        "Display output must be canonical (fixed point of parse)"
    );
}

// ---------------------------------------------------------------------------
// Round-trip preserves module identity (name + function count).
// ---------------------------------------------------------------------------

#[test]
fn parse_preserves_module_identity() {
    let m = make_const_module("text_roundtrip_identity");
    let text = encode_tmir_text(&m);
    let parsed = parse_tmir_text(&text).expect("parse");
    assert_eq!(parsed.name, m.name);
    assert_eq!(parsed.functions.len(), m.functions.len());
    assert_eq!(parsed.globals.len(), m.globals.len());
}

// ---------------------------------------------------------------------------
// save/load round-trip via a scratch file.
// ---------------------------------------------------------------------------

#[test]
fn save_then_load_via_format_text_round_trips() {
    let dir = std::env::temp_dir().join(format!(
        "llvm2_tmir_save_load_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("module.tmir");

    let m = make_const_module("text_save_load");
    save_module_to_tmir_text(&m, &path).expect("save .tmir text");

    let loaded = load_module_as(&path, FormatMode::Text).expect("load as Text");
    assert_eq!(loaded.name, m.name);

    // Also verify auto-detect on .tmir extension.
    let loaded_auto = load_module_as(&path, FormatMode::Auto).expect("load as Auto");
    assert_eq!(loaded_auto.name, m.name);

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// detect_input_format recognizes .tmir extension and the magic prefix.
// ---------------------------------------------------------------------------

#[test]
fn detect_input_format_recognizes_tmir_extension() {
    let dir = std::env::temp_dir().join(format!(
        "llvm2_tmir_detect_ext_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("module.tmir");
    std::fs::write(&path, "; tMIR text format v1\nmodule \"x\"\n").expect("write");
    assert_eq!(detect_input_format(&path), InputFormat::Tmir);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn detect_input_format_sniffs_tmir_magic_comment() {
    // A file with a weird extension but the canonical `; tMIR text format`
    // header should still be detected as Tmir.
    let dir = std::env::temp_dir().join(format!(
        "llvm2_tmir_detect_magic_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("module.txt");
    std::fs::write(&path, "; tMIR text format v1\nmodule \"x\"\n").expect("write");
    assert_eq!(detect_input_format(&path), InputFormat::Tmir);
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// load_module_from_bytes auto-detects .tmir text input.
// ---------------------------------------------------------------------------

#[test]
fn load_module_from_bytes_detects_tmir_text() {
    let m = make_const_module("text_from_bytes");
    let text = encode_tmir_text(&m);
    let loaded = load_module_from_bytes(text.as_bytes()).expect("load bytes as .tmir");
    assert_eq!(loaded.name, m.name);
}

// ---------------------------------------------------------------------------
// Golden: the Display output has the canonical header.
// ---------------------------------------------------------------------------

#[test]
fn tmir_text_header_is_canonical() {
    let m = make_const_module("golden_header");
    let text = encode_tmir_text(&m);
    assert!(
        text.starts_with("; tMIR text format v1\n"),
        "expected '; tMIR text format v1' header (format drift guard). got:\n{}",
        text.lines().take(2).collect::<Vec<_>>().join("\n")
    );
    assert!(text.contains("module \"golden_header\""));
}
