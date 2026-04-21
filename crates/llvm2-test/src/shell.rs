// Shell execution wrapper — the ONLY module allowed to import
// `std::process::Command`. Enforced by `llvm2-test ratchet shell-isolation`.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! External-process execution helpers for `llvm2-test`.
//!
//! # Invariants
//!
//! * No code outside `shell.rs` and `external.rs` may use
//!   `std::process::Command`. Verified by `ratchet shell-isolation`.
//! * `Spawn::run()` and `Spawn::capture()` are the only functions that
//!   actually fork. Every external tool goes through a typed wrapper in
//!   `external.rs` which internally uses `Spawn`.

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::{Context, Result};

/// A pending external-process invocation. Build with `Spawn::new`, chain
/// `.arg(..)` / `.env(..)`, then call `.run()` or `.capture()`.
#[derive(Debug)]
pub struct Spawn {
    program: OsString,
    args: Vec<OsString>,
    envs: Vec<(OsString, OsString)>,
    cwd: Option<PathBuf>,
    inherit_stderr: bool,
}

/// Captured stdout/stderr + exit status from an external invocation.
#[derive(Clone, Debug)]
pub struct Captured {
    /// Process exit code, or 255 if terminated by signal.
    pub code: i32,
    /// Full stdout.
    pub stdout: String,
    /// Full stderr.
    pub stderr: String,
}

impl Captured {
    /// Did the process exit with status 0?
    #[must_use]
    pub fn success(&self) -> bool {
        self.code == 0
    }
}

impl Spawn {
    /// Begin building a new invocation.
    #[must_use]
    pub fn new(program: impl Into<OsString>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            envs: Vec::new(),
            cwd: None,
            inherit_stderr: false,
        }
    }

    /// Append one argument.
    #[must_use]
    pub fn arg(mut self, arg: impl Into<OsString>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Append many arguments.
    #[must_use]
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<OsString>,
    {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Set an environment variable.
    #[must_use]
    pub fn env(mut self, key: impl Into<OsString>, val: impl Into<OsString>) -> Self {
        self.envs.push((key.into(), val.into()));
        self
    }

    /// Set the working directory for the child.
    #[must_use]
    pub fn cwd(mut self, path: impl Into<PathBuf>) -> Self {
        self.cwd = Some(path.into());
        self
    }

    /// When set, stderr streams to this process's stderr instead of being
    /// captured.
    #[must_use]
    pub fn inherit_stderr(mut self) -> Self {
        self.inherit_stderr = true;
        self
    }

    fn to_command(&self) -> Command {
        let mut c = Command::new(&self.program);
        c.args(&self.args);
        for (k, v) in &self.envs {
            c.env(k, v);
        }
        if let Some(cwd) = &self.cwd {
            c.current_dir(cwd);
        }
        c
    }

    /// Run, inherit stdio, and return the exit status.
    pub fn run(self) -> Result<i32> {
        let program = self.program.clone();
        let status = self
            .to_command()
            .status()
            .with_context(|| format!("failed to spawn {}", program.to_string_lossy()))?;
        Ok(status.code().unwrap_or(-1))
    }

    /// Run and capture stdout + stderr. Always returns `Captured` even on
    /// non-zero exit; inspect `.code` to decide how to treat the result.
    pub fn capture(self) -> Result<Captured> {
        let program = self.program.clone();
        let mut cmd = self.to_command();
        cmd.stdout(Stdio::piped());
        cmd.stderr(if self.inherit_stderr {
            Stdio::inherit()
        } else {
            Stdio::piped()
        });
        let output = cmd
            .output()
            .with_context(|| format!("failed to spawn {}", program.to_string_lossy()))?;
        Ok(Captured {
            code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

/// Check whether an executable is available on `PATH`.
#[must_use]
pub fn which(program: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let cand = dir.join(program);
        if cand.is_file() {
            return Some(cand);
        }
        // macOS: also try `program` without extension (already tried),
        // Windows: not supported here since we target macOS/Linux.
    }
    None
}
