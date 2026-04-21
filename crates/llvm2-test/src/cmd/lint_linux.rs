// `llvm2-test lint-linux` — cross-compile check for Linux targets.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Cross-compilation lint for the `#[cfg(target_os = "linux")]` paths in
//! `llvm2-codegen` (especially `src/jit.rs`).
//!
//! ## Why this exists
//!
//! The JIT module has hand-written Linux syscall and page-flag paths
//! (`SYS_MMAP` numbers, `MAP_FLAGS`, AArch64 `x8` vs macOS `x16`,
//! negative-return vs carry-flag error convention). On a macOS-only
//! developer workstation these paths are never exercised — they only
//! compile on the Linux target triple. Without a cross-check they bit-rot
//! silently until someone runs on Linux and hits a compile error.
//!
//! This subcommand runs `cargo check --target <TARGET> -p <PACKAGE>` for
//! every (target, package) pair we care about and reports the results. It
//! does **not** execute code — only checks that the compiler accepts the
//! Linux `cfg` arms. Runtime behavior testing would need QEMU or a real
//! Linux host (see issue #346 comments for the layered strategy).
//!
//! ## Toolchain discovery
//!
//! If `rustup` is not on `PATH`, or the requested target is not
//! installed (`rustup target list --installed` misses it), the target is
//! reported as `skipped` with an install hint. This keeps the default
//! `cargo test` run on macOS green when cross-toolchains aren't set up —
//! missing toolchains are environmental, not a Worker regression.
//!
//! Exit codes:
//!   * 0 (Ok)         — every installed target compiled cleanly.
//!   * 1 (Failed)     — at least one installed target failed to compile.
//!   * 2 (EnvBroken)  — `cargo` is missing (cannot run anything).
//!
//! When every target is skipped (no installed Linux targets), the exit
//! is `Ok` with a diagnostic — this is the common case on a fresh macOS
//! box and must not break `llvm2-test doctor` or CI.
//!
//! # Examples
//!
//! ```text
//! llvm2-test lint-linux
//! llvm2-test lint-linux --format json
//! llvm2-test lint-linux --target x86_64-unknown-linux-gnu
//! ```

use std::path::Path;

use anyhow::Context;
use clap::Args;
use serde::Serialize;

use super::GlobalArgs;
use crate::OutputFormat;
use crate::config::RepoRoot;
use crate::external::Cargo;
use crate::results::ResultStatus;
use crate::shell::{Spawn, which};

/// Default (target, package) pairs the lint runs.
///
/// Keep this list in lockstep with any crate that grows new
/// `#[cfg(target_os = "linux")]` blocks. Today only `llvm2-codegen` has
/// significant Linux-only code (`src/jit.rs`). When another crate adds
/// one, append to this list rather than invoking `lint-linux` N times.
const DEFAULT_TARGETS: &[&str] = &[
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
];

const DEFAULT_PACKAGE: &str = "llvm2-codegen";

/// Arguments for `llvm2-test lint-linux`.
#[derive(Args, Debug, Clone)]
#[command(
    long_about = "Cross-compile check for the Linux `#[cfg(target_os = \"linux\")]` \
                  paths in llvm2-codegen. Runs `cargo check --target <T> -p <P>` \
                  for each installed Linux target. Missing targets are skipped \
                  (they are environmental, not a failure) so this subcommand is \
                  safe to wire into CI on macOS-only developer boxes.\n\n\
                  Installed targets are discovered via `rustup target list \
                  --installed`. Install one with e.g. \
                  `rustup target add x86_64-unknown-linux-gnu`.\n\n\
                  Exit codes:\n  \
                  0 — every installed target compiled (or all targets skipped).\n  \
                  1 — at least one installed target failed to compile.\n  \
                  2 — cargo missing from PATH.\n\n\
                  # Examples\n\n  \
                  llvm2-test lint-linux\n  \
                  llvm2-test lint-linux --format json\n  \
                  llvm2-test lint-linux --target x86_64-unknown-linux-gnu\n  \
                  llvm2-test lint-linux --package llvm2-codegen --package llvm2-ir"
)]
pub struct LintLinuxArgs {
    /// Override the target triple(s) to check. Defaults to the two Linux
    /// triples we care about. Repeatable.
    #[arg(long, value_name = "TRIPLE")]
    pub target: Vec<String>,

    /// Package(s) to check. Defaults to `llvm2-codegen`. Repeatable.
    #[arg(long, value_name = "PACKAGE")]
    pub package: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum TargetStatus {
    /// Target installed and `cargo check` exited 0.
    Passed,
    /// Target installed but `cargo check` exited non-zero.
    Failed,
    /// Target not installed; compiler wasn't invoked.
    Skipped,
}

#[derive(Clone, Debug, Serialize)]
struct TargetRow {
    target: String,
    package: String,
    status: TargetStatus,
    /// Exit code of `cargo check`. `None` when skipped.
    exit_code: Option<i32>,
    /// Brief reason when `Skipped`, or diagnostic tail when `Failed`.
    note: String,
}

#[derive(Debug, Serialize)]
struct LintReport {
    command: &'static str,
    rustup_present: bool,
    installed_targets: Vec<String>,
    rows: Vec<TargetRow>,
    exit: ResultStatus,
}

/// Entry point.
pub fn run(global: &GlobalArgs, args: &LintLinuxArgs) -> anyhow::Result<ResultStatus> {
    let repo = RepoRoot::locate(Path::new("."))?;

    // Env broken: cargo is missing. Nothing else to do.
    if which(Cargo::program()).is_none() {
        eprintln!("lint-linux: cargo not found on PATH");
        return Ok(ResultStatus::EnvBroken);
    }

    let targets: Vec<String> = if args.target.is_empty() {
        DEFAULT_TARGETS.iter().map(|s| (*s).to_string()).collect()
    } else {
        args.target.clone()
    };
    let packages: Vec<String> = if args.package.is_empty() {
        vec![DEFAULT_PACKAGE.to_string()]
    } else {
        args.package.clone()
    };

    let (rustup_present, installed) = discover_installed_targets();

    let mut rows: Vec<TargetRow> = Vec::new();
    let mut any_failed = false;
    let mut any_ran = false;

    for target in &targets {
        let installed_here = installed.iter().any(|t| t == target);
        for package in &packages {
            if !installed_here {
                rows.push(TargetRow {
                    target: target.clone(),
                    package: package.clone(),
                    status: TargetStatus::Skipped,
                    exit_code: None,
                    note: skip_reason(rustup_present, target),
                });
                continue;
            }
            any_ran = true;
            let outcome = run_cargo_check(&repo, target, package, global.dry_run)?;
            if matches!(outcome.status, TargetStatus::Failed) {
                any_failed = true;
            }
            rows.push(outcome);
        }
    }

    // Exit: Failed if at least one installed target failed; otherwise Ok
    // (including the all-skipped case — a macOS box without cross targets
    // is a supported default, not an error).
    let exit = if any_failed {
        ResultStatus::Failed
    } else {
        ResultStatus::Ok
    };

    let report = LintReport {
        command: "lint-linux",
        rustup_present,
        installed_targets: installed,
        rows,
        exit,
    };

    match global.format {
        OutputFormat::Json | OutputFormat::Junit => {
            let json = serde_json::to_string_pretty(&report)?;
            println!("{json}");
        }
        OutputFormat::Human => print_human(&report, any_ran),
    }

    Ok(exit)
}

/// Return `(rustup_present, installed_targets)`. When `rustup` is missing
/// we cannot enumerate installed targets, so we return an empty list and
/// every target ends up `Skipped` with a rustup-install hint.
fn discover_installed_targets() -> (bool, Vec<String>) {
    if which("rustup").is_none() {
        return (false, Vec::new());
    }
    let captured = Spawn::new("rustup")
        .arg("target")
        .arg("list")
        .arg("--installed")
        .capture();
    let Ok(out) = captured else {
        return (true, Vec::new());
    };
    if !out.success() {
        return (true, Vec::new());
    }
    let list: Vec<String> = out
        .stdout
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    (true, list)
}

fn skip_reason(rustup_present: bool, target: &str) -> String {
    if !rustup_present {
        return "rustup missing; install: https://rustup.rs".to_string();
    }
    format!("target not installed; run: rustup target add {target}")
}

/// Run `cargo check --target <target> -p <package>` from the repo root.
///
/// Non-zero exit codes are reflected in `TargetStatus::Failed` rather
/// than propagated as a `Result::Err` — the caller wants to continue
/// across targets to produce a full report.
fn run_cargo_check(
    repo: &RepoRoot,
    target: &str,
    package: &str,
    dry_run: bool,
) -> anyhow::Result<TargetRow> {
    if dry_run {
        return Ok(TargetRow {
            target: target.to_string(),
            package: package.to_string(),
            status: TargetStatus::Passed,
            exit_code: Some(0),
            note: format!("dry-run: would invoke `cargo check --target {target} -p {package}`"),
        });
    }

    let captured = Spawn::new(Cargo::program())
        .cwd(repo.0.clone())
        .arg("check")
        .arg("--target")
        .arg(target)
        .arg("-p")
        .arg(package)
        .arg("--message-format=human")
        .arg("--color=never")
        .capture()
        .with_context(|| format!("invoking cargo check for target {target}"))?;

    let status = if captured.success() {
        TargetStatus::Passed
    } else {
        TargetStatus::Failed
    };

    // On failure, capture a short diagnostic tail so the report points at
    // the right file. Full output goes to `--format json` via the JSON
    // branch (not modeled here) or is reproducible from the repro hint.
    let note = if captured.success() {
        String::new()
    } else {
        diagnostic_tail(&captured.stderr, &captured.stdout)
    };

    Ok(TargetRow {
        target: target.to_string(),
        package: package.to_string(),
        status,
        exit_code: Some(captured.code),
        note,
    })
}

/// Keep the last ~20 lines of combined stderr+stdout for the human view.
/// Good enough to point at the offending file without blowing up JSON size.
fn diagnostic_tail(stderr: &str, stdout: &str) -> String {
    let combined = if stderr.is_empty() {
        stdout.to_string()
    } else if stdout.is_empty() {
        stderr.to_string()
    } else {
        format!("{stderr}\n{stdout}")
    };
    let lines: Vec<&str> = combined.lines().collect();
    let take = lines.len().min(20);
    let start = lines.len() - take;
    lines[start..].join("\n")
}

fn print_human(r: &LintReport, any_ran: bool) {
    println!("llvm2-test lint-linux");
    println!();
    if !r.rustup_present {
        println!("  rustup: not found on PATH — every target will be skipped.");
        println!("  install via https://rustup.rs then:");
        println!("    rustup target add x86_64-unknown-linux-gnu");
        println!("    rustup target add aarch64-unknown-linux-gnu");
        println!();
    } else if r.installed_targets.is_empty() {
        println!("  rustup: present, but `rustup target list --installed` returned empty.");
        println!();
    }

    println!(
        "  {:<36} {:<18} {:<9} {:>5}  note",
        "target", "package", "status", "exit"
    );
    println!("  {}", "-".repeat(95));
    for row in &r.rows {
        let status_str = match row.status {
            TargetStatus::Passed => "passed",
            TargetStatus::Failed => "FAILED",
            TargetStatus::Skipped => "skipped",
        };
        let exit = row
            .exit_code
            .map(|c| c.to_string())
            .unwrap_or_else(|| "-".to_string());
        let note = if row.note.lines().count() > 1 {
            // For multi-line notes (a FAILED diagnostic tail), show only
            // the first line in the table. The JSON view has the full tail.
            row.note.lines().next().unwrap_or("").to_string()
        } else {
            row.note.clone()
        };
        println!(
            "  {:<36} {:<18} {:<9} {:>5}  {}",
            row.target, row.package, status_str, exit, note
        );
    }

    println!();
    match r.exit {
        ResultStatus::Ok if any_ran => println!("  status: OK (all installed Linux targets compiled)"),
        ResultStatus::Ok => println!(
            "  status: OK (no Linux targets installed; install one with `rustup target add <triple>`)"
        ),
        ResultStatus::Failed => println!("  status: FAILED (see failing row above; re-run `cargo check --target <triple> -p <package>` for full output)"),
        ResultStatus::EnvBroken => println!("  status: ENV_BROKEN (cargo missing)"),
        ResultStatus::NotImplemented
        | ResultStatus::UsageError
        | ResultStatus::Errored => println!("  status: {:?}", r.exit),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skip_reason_without_rustup_points_to_installer() {
        let s = skip_reason(false, "x86_64-unknown-linux-gnu");
        assert!(s.contains("rustup"));
        assert!(s.contains("rustup.rs"));
    }

    #[test]
    fn skip_reason_with_rustup_points_to_target_add() {
        let s = skip_reason(true, "aarch64-unknown-linux-gnu");
        assert!(s.contains("rustup target add aarch64-unknown-linux-gnu"));
    }

    #[test]
    fn diagnostic_tail_truncates_to_last_20_lines() {
        let lines: Vec<String> = (0..50).map(|i| format!("line{i}")).collect();
        let joined = lines.join("\n");
        let tail = diagnostic_tail(&joined, "");
        let tail_lines: Vec<&str> = tail.lines().collect();
        assert_eq!(tail_lines.len(), 20);
        assert_eq!(tail_lines.first().copied(), Some("line30"));
        assert_eq!(tail_lines.last().copied(), Some("line49"));
    }

    #[test]
    fn diagnostic_tail_handles_empty_inputs() {
        assert!(diagnostic_tail("", "").is_empty());
        assert_eq!(diagnostic_tail("only stderr", ""), "only stderr");
        assert_eq!(diagnostic_tail("", "only stdout"), "only stdout");
    }

    #[test]
    fn diagnostic_tail_combines_both_streams() {
        let t = diagnostic_tail("err\n", "out\n");
        assert!(t.contains("err"));
        assert!(t.contains("out"));
    }

    #[test]
    fn default_targets_cover_both_linux_arches() {
        assert!(DEFAULT_TARGETS.contains(&"x86_64-unknown-linux-gnu"));
        assert!(DEFAULT_TARGETS.contains(&"aarch64-unknown-linux-gnu"));
    }
}
