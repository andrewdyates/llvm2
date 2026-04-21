// Typed wrappers around every external tool `llvm2-test` may invoke.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Only `shell.rs` and this module may construct a subprocess. Any other
// file doing so fails `llvm2-test ratchet shell-isolation`.

//! Typed external-tool wrappers.
//!
//! Each supported tool (clang, csmith, z4, ...) is a zero-sized struct
//! with a `check()` method that verifies presence + returns a
//! `VersionInfo`. Subcommands compose these into higher-level operations.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::shell::{Spawn, which};

/// Tool presence + reported version.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VersionInfo {
    /// Short tool label, e.g. `"clang"`.
    pub name: String,
    /// Absolute path to the executable (if resolvable).
    pub path: Option<String>,
    /// Reported version string — empty if `--version` failed.
    pub version: String,
    /// Whether the tool was found on `$PATH`.
    pub present: bool,
}

impl VersionInfo {
    fn missing(name: &str) -> Self {
        Self {
            name: name.to_string(),
            path: None,
            version: String::new(),
            present: false,
        }
    }
}

fn probe(name: &str, arg: &str) -> VersionInfo {
    let Some(path) = which(name) else {
        return VersionInfo::missing(name);
    };
    let res = Spawn::new(name).arg(arg).capture();
    let version = res
        .as_ref()
        .map(|c| {
            let raw = if c.stdout.trim().is_empty() {
                c.stderr.clone()
            } else {
                c.stdout.clone()
            };
            raw.lines().next().unwrap_or("").trim().to_string()
        })
        .unwrap_or_default();
    VersionInfo {
        name: name.to_string(),
        path: Some(path.to_string_lossy().into_owned()),
        version,
        present: true,
    }
}

/// Discover tool presence + version without shelling out beyond
/// `--version`. Returns `VersionInfo` for each requested name.
///
/// # Errors
/// Currently infallible; returns `Result` for forward-compatibility.
pub fn probe_many<I, S>(names: I) -> Result<Vec<VersionInfo>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    Ok(names
        .into_iter()
        .map(|n| probe(n.as_ref(), "--version"))
        .collect())
}

macro_rules! tool {
    ($(#[$attr:meta])* $name:ident, $bin:expr, $version_arg:expr) => {
        $(#[$attr])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;
        impl $name {
            /// Short program name used on `PATH`.
            #[must_use]
            pub const fn program() -> &'static str {
                $bin
            }
            /// Probe presence + version.
            ///
            /// # Errors
            /// Currently infallible but returns `Result` for consistency.
            pub fn check(self) -> Result<VersionInfo> {
                Ok(probe($bin, $version_arg))
            }
        }
    };
}

tool!(
    /// The `clang` C/C++ compiler. Required for WS2, WS3.
    Clang, "clang", "--version"
);
tool!(
    /// The system `cc` compiler (links binaries in several WSs).
    Cc, "cc", "--version"
);
tool!(
    /// csmith — random C program generator. Required for WS3 `fuzz`.
    Csmith, "csmith", "--version"
);
tool!(
    /// YARPGen — random C/C++ program generator. Required for WS3.
    Yarpgen, "yarpgen", "--version"
);
tool!(
    /// creduce — testcase reducer for miscompiles.
    Creduce, "creduce", "--version"
);
tool!(
    /// z4 — SMT solver for WS7.
    Z4, "z4", "--version"
);
tool!(
    /// `gh` — GitHub CLI. Used to file miscompile issues from WS3.
    Gh, "gh", "--version"
);
tool!(
    /// `cargo` — Rust package manager. Used by WS1 and WS6.
    Cargo, "cargo", "--version"
);
tool!(
    /// `rustc` — Rust compiler. Used by WS4, WS5, WS6.
    Rustc, "rustc", "--version"
);

/// Called by `cmd::doctor` to tell the operator how to install a missing
/// tool. Keep in sync with the `tool!` macro calls above.
#[must_use]
pub fn install_hint(program: &str) -> &'static str {
    match program {
        "clang" | "cc" => "install Xcode Command Line Tools: `xcode-select --install`",
        "csmith" => "brew install csmith",
        "yarpgen" => "build from source: https://github.com/intel/yarpgen",
        "creduce" => "brew install creduce",
        "z4" => "see `designs/2026-04-14-z4-integration-guide.md`",
        "gh" => "brew install gh",
        "cargo" | "rustc" => "install via rustup: https://rustup.rs",
        _ => "check tool documentation",
    }
}

/// Convenience: call `probe` once for a single tool name.
#[must_use]
pub fn probe_one(name: &str) -> VersionInfo {
    probe(name, "--version")
}
