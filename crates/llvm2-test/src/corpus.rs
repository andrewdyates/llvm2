// External corpus discovery for `llvm2-test`.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Manage locations of external reference corpora.
//!
//! Each WS-specific subcommand asks this module for the canonical path
//! to its input corpus. Paths default to `~/<repo>-ref/` per the
//! org-wide convention (`andrewdyates.md` — Reference Repos).

use std::path::PathBuf;

/// A reference corpus cloned somewhere on disk.
#[derive(Clone, Debug)]
pub struct Corpus {
    /// Short label, e.g. `"llvm-test-suite"`.
    pub name: &'static str,
    /// Default path under `$HOME`.
    pub default_path: PathBuf,
}

impl Corpus {
    fn new(name: &'static str, rel: &str) -> Self {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/tmp"));
        Self {
            name,
            default_path: home.join(rel),
        }
    }

    /// Does the corpus appear to be present on disk?
    #[must_use]
    pub fn present(&self) -> bool {
        self.default_path.is_dir()
    }
}

/// Every corpus this binary knows about. Used by `doctor`.
#[must_use]
pub fn all() -> Vec<Corpus> {
    vec![
        Corpus::new("llvm-test-suite", "llvm-test-suite-ref"),
        Corpus::new("rustc", "rustc-ref"),
        Corpus::new("rustc_codegen_cranelift", "rustc_codegen_cranelift-ref"),
        Corpus::new("llvm-project", "llvm-project-ref"),
    ]
}
