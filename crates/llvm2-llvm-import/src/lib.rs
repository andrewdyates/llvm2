// llvm2-llvm-import / lib.rs
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Minimal LLVM-IR text (.ll) -> tMIR importer. See crate-level README
// and `Cargo.toml` for scope and motivation (WS2, issue #439).
//
// The importer is line-oriented: each instruction in clang -O0 output
// sits on its own line. We tokenize lightly (enough to split operands
// and types) and translate directly to `tmir::Inst` variants. Every
// unsupported construct returns `Error::Unsupported(String)` so the
// driver script can classify programs as `unsupported` (not `crash`)
// in the per-run JSON.

#![forbid(unsafe_code)]

use std::path::Path;

mod parser;

pub use parser::{import_module, import_text};

/// Errors returned by the LLVM IR importer.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// I/O error reading the input file.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// A parse error: the input is syntactically malformed for the
    /// (very narrow) grammar the importer understands.
    #[error("parse error at line {line}: {message}")]
    Parse { line: usize, message: String },

    /// The input uses an LLVM IR feature that this importer does not
    /// translate. The driver should record this as `status: unsupported`
    /// rather than as a crash or failure.
    #[error("unsupported LLVM IR feature: {0}")]
    Unsupported(String),
}

/// Convenience alias for importer results.
pub type Result<T> = std::result::Result<T, Error>;

/// Re-export helper: read the file at `path` and import its LLVM IR
/// text into a `tmir::Module`.
pub fn import_path(path: &Path) -> Result<tmir::Module> {
    import_module(path)
}
