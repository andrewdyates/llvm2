// tmir-func/reader.rs — JSON reader/writer for tMIR modules
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Provides JSON serialization/deserialization with validation for tMIR modules.
// This is the wire format used by external tools to feed tMIR into LLVM2
// without depending on the tMIR repo directly.

use crate::Module;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

/// Error type for tMIR JSON read/write operations.
#[derive(Debug)]
pub enum TmirJsonError {
    /// File system I/O error.
    Io(std::io::Error),
    /// JSON serialization or deserialization error.
    Json(serde_json::Error),
    /// Structural validation error (e.g., missing entry block).
    Validation(String),
}

impl fmt::Display for TmirJsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Json(err) => write!(f, "JSON error: {err}"),
            Self::Validation(msg) => write!(f, "validation error: {msg}"),
        }
    }
}

impl Error for TmirJsonError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Json(err) => Some(err),
            Self::Validation(_) => None,
        }
    }
}

impl From<std::io::Error> for TmirJsonError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<serde_json::Error> for TmirJsonError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

/// Read a tMIR module from a JSON file on disk.
pub fn read_module_from_json(path: &Path) -> Result<Module, TmirJsonError> {
    let json = fs::read_to_string(path)?;
    read_module_from_str(&json)
}

/// Parse a tMIR module from a JSON string.
pub fn read_module_from_str(json: &str) -> Result<Module, TmirJsonError> {
    let module: Module = serde_json::from_str(json)?;
    validate_module(&module)?;
    Ok(module)
}

/// Parse a tMIR module from any `Read` source (stdin, network, etc.).
pub fn read_module_from_reader<R: Read>(mut reader: R) -> Result<Module, TmirJsonError> {
    let mut json = String::new();
    reader.read_to_string(&mut json)?;
    read_module_from_str(&json)
}

/// Write a tMIR module to a JSON file on disk (pretty-printed).
pub fn write_module_to_json(module: &Module, path: &Path) -> Result<(), TmirJsonError> {
    let json = write_module_to_string(module)?;
    fs::write(path, json)?;
    Ok(())
}

/// Serialize a tMIR module to a pretty-printed JSON string.
pub fn write_module_to_string(module: &Module) -> Result<String, TmirJsonError> {
    validate_module(module)?;
    Ok(serde_json::to_string_pretty(module)?)
}

/// Write a tMIR module as JSON to any `Write` sink (stdout, file, network).
pub fn write_module_to_writer<W: Write>(
    module: &Module,
    mut writer: W,
) -> Result<(), TmirJsonError> {
    let json = write_module_to_string(module)?;
    writer.write_all(json.as_bytes())?;
    writer.flush()?;
    Ok(())
}

/// Validate a tMIR module for structural correctness.
///
/// Checks:
/// - Module name is non-empty
/// - Each function has a non-empty name
/// - Each function's entry block ID references an existing block
pub fn validate_module(module: &Module) -> Result<(), TmirJsonError> {
    if module.name.trim().is_empty() {
        return Err(TmirJsonError::Validation(
            "module name cannot be empty".to_string(),
        ));
    }

    for (index, function) in module.functions.iter().enumerate() {
        if function.name.trim().is_empty() {
            return Err(TmirJsonError::Validation(format!(
                "function at index {index} has an empty name"
            )));
        }

        if function.block(function.entry).is_none() {
            return Err(TmirJsonError::Validation(format!(
                "function `{}` is missing its entry block (BlockId({}))",
                function.name, function.entry.0
            )));
        }
    }

    Ok(())
}

/// Serialize a module to JSON and deserialize it back.
///
/// Useful for testing round-trip fidelity of the wire format.
pub fn round_trip(module: &Module) -> Result<Module, TmirJsonError> {
    let json = write_module_to_string(module)?;
    read_module_from_str(&json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{self, ModuleBuilder};
    use tmir_instrs::{BinOp, LandingPadClause};
    use tmir_types::{BlockId, FuncId, Ty};

    fn make_add_module() -> Module {
        let mut mb = ModuleBuilder::new("test_add");
        let mut fb = mb.function("add_i32", vec![Ty::i32(), Ty::i32()], vec![Ty::i32()]);
        let (entry_id, params) = fb.entry_block();
        let result = fb.fresh_value();
        fb.add_block(
            entry_id,
            vec![(params[0], Ty::i32()), (params[1], Ty::i32())],
            vec![
                builder::binop(BinOp::Add, Ty::i32(), params[0], params[1], result),
                builder::ret(vec![result]),
            ],
        );
        let func = fb.build();
        mb.add_function(func);
        mb.build()
    }

    #[test]
    fn test_round_trip_preserves_equality() {
        let module = make_add_module();
        let rt = round_trip(&module).expect("round-trip failed");
        assert_eq!(module, rt);
    }

    #[test]
    fn test_write_and_read_string() {
        let module = make_add_module();
        let json = write_module_to_string(&module).unwrap();
        let parsed = read_module_from_str(&json).unwrap();
        assert_eq!(module, parsed);
    }

    #[test]
    fn test_validation_empty_name() {
        let module = Module {
            name: "".to_string(),
            functions: vec![],
            structs: vec![],
            globals: vec![],
            data_layout: None,
        };
        let result = validate_module(&module);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("empty"),
            "expected empty name error"
        );
    }

    #[test]
    fn test_validation_missing_entry_block() {
        let module = Module {
            name: "bad_module".to_string(),
            functions: vec![crate::Function {
                id: tmir_types::FuncId(0),
                name: "bad_func".to_string(),
                ty: tmir_types::FuncTy {
                    params: vec![],
                    returns: vec![],
                },
                entry: BlockId(99), // does not exist
                blocks: vec![],
                proofs: vec![],
            }],
            structs: vec![],
            globals: vec![],
            data_layout: None,
        };
        let result = validate_module(&module);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("missing its entry block"),
        );
    }

    #[test]
    fn test_read_write_file() {
        let module = make_add_module();
        let dir = std::env::temp_dir().join("tmir_reader_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_module.json");
        write_module_to_json(&module, &path).unwrap();
        let loaded = read_module_from_json(&path).unwrap();
        assert_eq!(module, loaded);
        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // Exception handling tests
    // -----------------------------------------------------------------------

    /// Build a module with invoke/landingpad/resume to test the full EH flow.
    ///
    /// Structure:
    ///   entry:
    ///     %r = invoke @callee() to normal_bb unwind landing_bb
    ///   normal_bb:
    ///     return %r
    ///   landing_bb:
    ///     %exn = landingpad i64 catch i32
    ///     resume %exn
    fn make_eh_module() -> Module {
        let mut mb = ModuleBuilder::new("test_eh");
        let callee_id = FuncId(1);

        let mut fb = mb.function("eh_func", vec![], vec![Ty::i32()]);
        let (entry_id, _params) = fb.entry_block();

        let normal_bb = fb.fresh_block();
        let landing_bb = fb.fresh_block();

        let invoke_result = fb.fresh_value();
        let exn_val = fb.fresh_value();

        // Entry block: invoke @callee
        fb.add_block(
            entry_id,
            vec![],
            vec![builder::invoke(
                callee_id,
                vec![],
                vec![Ty::i32()],
                normal_bb,
                vec![],
                landing_bb,
                vec![],
                vec![invoke_result],
            )],
        );

        // Normal continuation: return the result
        fb.add_block(
            normal_bb,
            vec![],
            vec![builder::ret(vec![invoke_result])],
        );

        // Landing pad: catch i32 exceptions, then resume
        fb.add_block(
            landing_bb,
            vec![],
            vec![
                builder::landing_pad(
                    Ty::i64(),
                    vec![LandingPadClause::Catch(Ty::i32())],
                    exn_val,
                ),
                builder::resume(exn_val),
            ],
        );

        let func = fb.build();
        mb.add_function(func);
        mb.build()
    }

    #[test]
    fn test_eh_module_round_trip() {
        let module = make_eh_module();
        let rt = round_trip(&module).expect("round-trip failed");
        assert_eq!(module, rt);
    }

    #[test]
    fn test_eh_module_json_string() {
        let module = make_eh_module();
        let json = write_module_to_string(&module).unwrap();
        // Verify key instruction names appear in JSON
        assert!(json.contains("Invoke"), "JSON should contain Invoke variant");
        assert!(
            json.contains("LandingPad"),
            "JSON should contain LandingPad variant"
        );
        assert!(json.contains("Resume"), "JSON should contain Resume variant");
        assert!(
            json.contains("Catch"),
            "JSON should contain Catch clause"
        );
        // Parse back
        let parsed = read_module_from_str(&json).unwrap();
        assert_eq!(module, parsed);
    }

    #[test]
    fn test_builder_invoke() {
        let node = builder::invoke(
            FuncId(0),
            vec![tmir_types::ValueId(0)],
            vec![Ty::i32()],
            BlockId(1),
            vec![],
            BlockId(2),
            vec![],
            vec![tmir_types::ValueId(1)],
        );
        assert_eq!(node.results.len(), 1);
        assert!(matches!(node.instr, tmir_instrs::Instr::Invoke { .. }));
    }

    #[test]
    fn test_builder_landing_pad() {
        let node = builder::landing_pad(
            Ty::i64(),
            vec![
                LandingPadClause::Catch(Ty::i32()),
                LandingPadClause::Filter(vec![Ty::i32()]),
            ],
            tmir_types::ValueId(5),
        );
        assert_eq!(node.results.len(), 1);
        assert_eq!(node.results[0], tmir_types::ValueId(5));
        match &node.instr {
            tmir_instrs::Instr::LandingPad { ty, clauses } => {
                assert_eq!(*ty, Ty::i64());
                assert_eq!(clauses.len(), 2);
            }
            other => panic!("expected LandingPad, got {:?}", other),
        }
    }

    #[test]
    fn test_builder_resume() {
        let node = builder::resume(tmir_types::ValueId(3));
        assert!(node.results.is_empty());
        match &node.instr {
            tmir_instrs::Instr::Resume { value } => {
                assert_eq!(*value, tmir_types::ValueId(3));
            }
            other => panic!("expected Resume, got {:?}", other),
        }
    }

    #[test]
    fn test_eh_with_filter_clause_round_trip() {
        let mut mb = ModuleBuilder::new("test_filter");
        let mut fb = mb.function("filter_func", vec![], vec![]);
        let (entry_id, _params) = fb.entry_block();
        let exn = fb.fresh_value();
        fb.add_block(
            entry_id,
            vec![],
            vec![
                builder::landing_pad(
                    Ty::i64(),
                    vec![LandingPadClause::Filter(vec![Ty::i32(), Ty::i64()])],
                    exn,
                ),
                builder::resume(exn),
            ],
        );
        let func = fb.build();
        mb.add_function(func);
        let module = mb.build();
        let rt = round_trip(&module).expect("round-trip failed");
        assert_eq!(module, rt);
    }
}
