// tmir-func/tests/json_wire_format.rs - Integration tests for JSON wire format
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Tests JSON serialization round-trip, fixture file reading, and compilation
// through the LLVM2 pipeline from JSON input.

use std::path::PathBuf;

use tmir_func::builder::{self, ModuleBuilder};
use tmir_func::reader::{
    self, read_module_from_json, read_module_from_str, round_trip, validate_module,
    write_module_to_string,
};
use tmir_func::{Function, Module};
use tmir_instrs::BinOp;
use tmir_types::{BlockId, FuncId, FuncTy, Ty};

// ---------------------------------------------------------------------------
// Helper: path to a test fixture
// ---------------------------------------------------------------------------

fn fixture_path(name: &str) -> PathBuf {
    // Navigate from stubs/tmir-func/ up to repo root then into tests/fixtures/
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // stubs/
    p.pop(); // repo root
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

// ---------------------------------------------------------------------------
// Test: read each fixture file
// ---------------------------------------------------------------------------

#[test]
fn test_read_fixture_add_i32() {
    let path = fixture_path("add_i32.json");
    let module = read_module_from_json(&path).expect("failed to read add_i32.json");
    assert_eq!(module.name, "add_example");
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name, "add_i32");
    assert_eq!(module.functions[0].ty.params.len(), 2);
    assert_eq!(module.functions[0].ty.returns.len(), 1);
    assert_eq!(module.functions[0].blocks.len(), 1);
}

#[test]
fn test_read_fixture_branch_example() {
    let path = fixture_path("branch_example.json");
    let module = read_module_from_json(&path).expect("failed to read branch_example.json");
    assert_eq!(module.name, "branch_example");
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name, "max_i64");
    // 3 blocks: entry, then, else
    assert_eq!(module.functions[0].blocks.len(), 3);
}

#[test]
fn test_read_fixture_multi_func_proofs() {
    let path = fixture_path("multi_func_proofs.json");
    let module = read_module_from_json(&path).expect("failed to read multi_func_proofs.json");
    assert_eq!(module.name, "multi_func_proofs");
    assert_eq!(module.functions.len(), 3);
    assert_eq!(module.functions[0].name, "pure_add");
    assert_eq!(module.functions[1].name, "negate_f64");
    assert_eq!(module.functions[2].name, "load_and_store");
    // Verify proof annotations survived deserialization
    assert!(!module.functions[0].proofs.is_empty(), "pure_add should have function proofs");
    // Instruction-level proofs
    let add_instr = &module.functions[0].blocks[0].body[0];
    assert!(
        !add_instr.proofs.is_empty(),
        "add instruction should have proof annotations"
    );
}

// ---------------------------------------------------------------------------
// Test: round-trip each fixture through serialize/deserialize
// ---------------------------------------------------------------------------

#[test]
fn test_round_trip_add_i32() {
    let path = fixture_path("add_i32.json");
    let original = read_module_from_json(&path).unwrap();
    let rt = round_trip(&original).expect("round-trip failed");
    assert_eq!(original, rt);
}

#[test]
fn test_round_trip_branch_example() {
    let path = fixture_path("branch_example.json");
    let original = read_module_from_json(&path).unwrap();
    let rt = round_trip(&original).expect("round-trip failed");
    assert_eq!(original, rt);
}

#[test]
fn test_round_trip_multi_func_proofs() {
    let path = fixture_path("multi_func_proofs.json");
    let original = read_module_from_json(&path).unwrap();
    let rt = round_trip(&original).expect("round-trip failed");
    assert_eq!(original, rt);
}

// ---------------------------------------------------------------------------
// Test: builder-constructed module round-trips through JSON
// ---------------------------------------------------------------------------

#[test]
fn test_builder_to_json_round_trip() {
    let mut mb = ModuleBuilder::new("builder_test");

    // Build a simple add function
    let mut fb = mb.function("add_i64", vec![Ty::i64(), Ty::i64()], vec![Ty::i64()]);
    let (entry_id, params) = fb.entry_block();
    let result = fb.fresh_value();
    fb.add_block(
        entry_id,
        vec![(params[0], Ty::i64()), (params[1], Ty::i64())],
        vec![
            builder::binop(BinOp::Add, Ty::i64(), params[0], params[1], result),
            builder::ret(vec![result]),
        ],
    );
    let func = fb.build();
    mb.add_function(func);

    let module = mb.build();

    // Serialize to JSON, then parse back
    let json = write_module_to_string(&module).expect("serialization failed");
    let parsed = read_module_from_str(&json).expect("deserialization failed");
    assert_eq!(module, parsed);
}

// ---------------------------------------------------------------------------
// Test: validation rejects invalid modules
// ---------------------------------------------------------------------------

#[test]
fn test_validation_rejects_empty_name() {
    let module = Module {
        name: "".to_string(),
        functions: vec![],
        structs: vec![],
        globals: vec![],
        data_layout: None,
    };
    assert!(validate_module(&module).is_err());
}

#[test]
fn test_validation_rejects_missing_entry_block() {
    let module = Module {
        name: "bad".to_string(),
        functions: vec![Function {
            id: FuncId(0),
            name: "bad_func".to_string(),
            ty: FuncTy {
                params: vec![],
                returns: vec![],
            },
            entry: BlockId(99),
            blocks: vec![],
            proofs: vec![],
        }],
        structs: vec![],
        globals: vec![],
        data_layout: None,
    };
    assert!(validate_module(&module).is_err());
}

#[test]
fn test_validation_accepts_empty_function_list() {
    let module = Module {
        name: "empty".to_string(),
        functions: vec![],
        structs: vec![],
        globals: vec![],
        data_layout: None,
    };
    assert!(validate_module(&module).is_ok());
}

// ---------------------------------------------------------------------------
// Test: malformed JSON is rejected
// ---------------------------------------------------------------------------

#[test]
fn test_malformed_json_rejected() {
    let bad_json = r#"{ "name": "test", "functions": [ { "bad_field": 1 } ] }"#;
    assert!(read_module_from_str(bad_json).is_err());
}

#[test]
fn test_completely_invalid_json_rejected() {
    assert!(read_module_from_str("not json at all").is_err());
}

// ---------------------------------------------------------------------------
// Test: JSON wire format reader from in-memory bytes (reader trait)
// ---------------------------------------------------------------------------

#[test]
fn test_read_module_from_reader() {
    let module = Module::new("reader_test");
    let json = serde_json::to_string(&module).unwrap();
    let cursor = std::io::Cursor::new(json.as_bytes());
    let parsed = reader::read_module_from_reader(cursor).expect("reader failed");
    assert_eq!(module.name, parsed.name);
}

// ---------------------------------------------------------------------------
// Test: write and read back from temp file
// ---------------------------------------------------------------------------

#[test]
fn test_write_read_tempfile() {
    let mut mb = ModuleBuilder::new("tempfile_test");
    let mut fb = mb.function("identity", vec![Ty::i32()], vec![Ty::i32()]);
    let (entry_id, params) = fb.entry_block();
    fb.add_block(
        entry_id,
        vec![(params[0], Ty::i32())],
        vec![builder::ret(vec![params[0]])],
    );
    mb.add_function(fb.build());
    let module = mb.build();

    let dir = std::env::temp_dir().join("llvm2_json_wire_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_module.json");

    reader::write_module_to_json(&module, &path).expect("write failed");
    let loaded = reader::read_module_from_json(&path).expect("read failed");
    assert_eq!(module, loaded);

    let _ = std::fs::remove_file(&path);
}
