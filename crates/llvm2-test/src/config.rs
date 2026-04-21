// llvm2-test configuration loader.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Typed configuration for `llvm2-test`.
//!
//! Reads `llvm2-test.toml` at the repo root (or the path given by
//! `--config`). All tunable defaults for every subcommand live here.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The resolved project root. All I/O paths in `llvm2-test` are derived
/// from this type — nothing is hard-coded.
#[derive(Clone, Debug)]
pub struct RepoRoot(pub PathBuf);

impl RepoRoot {
    /// Locate the repo root by walking up from `start` looking for the
    /// workspace `Cargo.toml`. Falls back to the current working
    /// directory if the walk fails.
    pub fn locate(start: &Path) -> anyhow::Result<Self> {
        let mut cur = start.canonicalize().unwrap_or_else(|_| start.to_path_buf());
        loop {
            if cur.join("Cargo.toml").is_file() && cur.join("crates").is_dir() {
                return Ok(Self(cur));
            }
            if !cur.pop() {
                // Fall back to cwd — keeps `llvm2-test` runnable from the
                // binary directory during development.
                return Ok(Self(std::env::current_dir()?));
            }
        }
    }

    /// Path relative to the repo root.
    #[must_use]
    pub fn join<P: AsRef<Path>>(&self, rel: P) -> PathBuf {
        self.0.join(rel)
    }
}

/// Top-level config file schema.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ConfigFile {
    /// Where to write result artifacts. Default: `evals/results/`.
    pub results_dir: Option<PathBuf>,
    /// Where the committed schemas live. Default: `evals/schema/`.
    pub schema_dir: Option<PathBuf>,
    /// Where weekly reports are written. Default: `reports/weekly/`.
    pub weekly_dir: Option<PathBuf>,
    /// Per-subcommand defaults.
    pub matrix: MatrixConfig,
    /// Fuzz defaults.
    pub fuzz: FuzzConfig,
}

/// Defaults for the `matrix` subcommand.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MatrixConfig {
    /// Default timeout in seconds per crate.
    pub per_crate_timeout_s: Option<u64>,
}

/// Defaults for the `fuzz` subcommand.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FuzzConfig {
    /// Default fuzz duration in seconds.
    pub duration_s: Option<u64>,
}

impl ConfigFile {
    /// Load from an explicit path.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        if !path.is_file() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path)?;
        let cfg: Self = toml::from_str(&text)?;
        Ok(cfg)
    }
}
