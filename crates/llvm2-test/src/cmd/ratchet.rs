// Ratchet checks — shell isolation, schema drift, etc.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! `llvm2-test ratchet` — invariants CI enforces over the test suite.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Args, Subcommand};
use schemars::schema_for;
use serde::{Deserialize, Serialize};

use super::GlobalArgs;
use super::lint_linux::LintLinuxArgs;
use crate::OutputFormat;
use crate::config::RepoRoot;
use crate::external::Cargo;
use crate::results::{EmptyRecord, ResultStatus, RunSummary};
use crate::shell::Spawn;

/// Which ratchet to run.
#[derive(Subcommand, Debug, Clone)]
pub enum RatchetCommand {
    /// Pass-count regression check — compares the newest JSON under
    /// `evals/results/tests/` against `ratchet/tests_baseline.json`.
    Tests(TestsArgs),
    /// Warnings-zero ratchet — builds the workspace + tests and compares
    /// the diagnostic count against `ratchet/warnings_baseline.json`.
    Warnings,
    /// Monotonic-decrease ratchet on production panic-family sites
    /// (`.unwrap()` / `.expect(` / `panic!` / `unreachable!` / `todo!`).
    /// Compares per-file counts against `ratchet/unwrap_baseline.json`.
    Unwrap,
    /// Enforce `std::process::Command` is only used in `shell.rs` / `external.rs`.
    ShellIsolation,
    /// Assert committed `evals/schema/*.json` matches what schemars emits.
    Schema,
    /// Cross-compile check for `#[cfg(target_os = "linux")]` paths in
    /// `llvm2-codegen` (issue #346). Safe on macOS-only boxes — targets
    /// not installed via `rustup target add` are reported as `skipped`.
    ///
    /// Equivalent to the top-level `llvm2-test lint-linux`. Registered as
    /// a ratchet so CI can sweep every invariant with
    /// `llvm2-test ratchet <check>` under a single dispatcher (#446).
    LintLinux(LintLinuxArgs),
}

/// Arguments for `llvm2-test ratchet tests`.
#[derive(Args, Debug, Clone, Default)]
#[command(
    long_about = "Monotonic pass-count ratchet for the workspace test matrix.\n\n\
                  Compares the newest JSON in `evals/results/tests/` against\n\
                  `ratchet/tests_baseline.json`. Fails if any `(crate, shard)`\n\
                  pair in the baseline has a lower `passed` count in the\n\
                  current run, or a higher `failed` count, or has disappeared\n\
                  entirely. New shards are OK — the baseline only grows.\n\n\
                  Exit codes:\n  \
                  0 — current run is at or above baseline on every pair.\n  \
                  1 — regression detected.\n  \
                  2 — tooling error (missing files, parse error, etc.).\n\n\
                  # Examples\n\n  \
                  llvm2-test ratchet tests\n  \
                  llvm2-test ratchet tests --current evals/results/tests/2026-04-19.json\n  \
                  llvm2-test ratchet tests --format json"
)]
pub struct TestsArgs {
    /// Override the "current" test-matrix JSON. Default: newest
    /// `*.json` under `evals/results/tests/`.
    #[arg(long, value_name = "PATH")]
    pub current: Option<PathBuf>,
    /// Override the baseline JSON. Default: `ratchet/tests_baseline.json`.
    #[arg(long, value_name = "PATH")]
    pub baseline: Option<PathBuf>,
}

/// Arguments for `llvm2-test ratchet`.
#[derive(Args, Debug, Clone)]
#[command(
    long_about = "Run ratchet checks (called by CI). Every ratchet fails \
                  with a non-zero exit code when its invariant is \
                  violated.\n\n\
                  # Examples\n\n  \
                  llvm2-test ratchet shell-isolation\n  \
                  llvm2-test ratchet schema --format json\n  \
                  llvm2-test ratchet tests\n  \
                  llvm2-test ratchet warnings\n  \
                  llvm2-test ratchet unwrap\n  \
                  llvm2-test ratchet lint-linux"
)]
pub struct RatchetArgs {
    /// Which ratchet to run.
    #[command(subcommand)]
    pub cmd: RatchetCommand,
}

/// Entry point.
pub fn run(global: &GlobalArgs, args: &RatchetArgs) -> anyhow::Result<ResultStatus> {
    match &args.cmd {
        RatchetCommand::Tests(a) => tests(global, a),
        RatchetCommand::Warnings => warnings(global),
        RatchetCommand::Unwrap => unwrap(global),
        RatchetCommand::ShellIsolation => shell_isolation(global),
        RatchetCommand::Schema => schema(global),
        RatchetCommand::LintLinux(a) => super::lint_linux::run(global, a),
    }
}

// ---------------- tests (pass-count ratchet) ----------------
//
// Port of `scripts/check_test_ratchet.sh`. Compares the newest test-
// matrix JSON under `evals/results/tests/` against
// `ratchet/tests_baseline.json`. Fails (exit 1) on any of:
//   * A `(crate, shard)` pair in the baseline has a lower `passed`
//     count in the current run.
//   * A `(crate, shard)` pair in the baseline has a higher `failed`
//     count in the current run.
//   * A baseline shard is missing from the current run entirely.
// New shards in the current run are accepted silently — the baseline
// grows monotonically on refresh.
//
// Exit semantics match the shell script:
//   * `ResultStatus::Ok` (0)          — no regressions
//   * `ResultStatus::Failed` (1)      — at least one violation / missing
//   * `ResultStatus::EnvBroken` (2)   — missing / unparseable input files

/// One per-crate/per-shard entry in the test-matrix JSON. Field set
/// matches `ratchet/tests_baseline.json` today. Unknown fields are
/// ignored so future matrix runs can add metadata without breaking the
/// ratchet.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct ShardEntry {
    #[serde(default, rename = "crate")]
    krate: String,
    #[serde(default)]
    shard: String,
    #[serde(default)]
    passed: u64,
    #[serde(default)]
    failed: u64,
}

/// Top-level test-matrix JSON document. Only the fields the ratchet
/// reads are modeled.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct TestMatrixDoc {
    #[serde(default)]
    shards: Vec<ShardEntry>,
    #[serde(default)]
    totals: TotalsEntry,
}

/// `totals` subtree — only `passed` / `failed` are used for the
/// human-readable summary line.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct TotalsEntry {
    #[serde(default)]
    passed: u64,
    #[serde(default)]
    failed: u64,
}

/// Why a `(crate, shard)` pair regressed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum RegressionKind {
    /// `passed` decreased.
    PassedDecreased,
    /// `failed` increased.
    FailedIncreased,
}

impl RegressionKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::PassedDecreased => "passed decreased",
            Self::FailedIncreased => "failed increased",
        }
    }
}

/// One regression finding.
#[derive(Clone, Debug, Serialize)]
struct Regression {
    #[serde(rename = "crate")]
    krate: String,
    shard: String,
    kind: RegressionKind,
    baseline: u64,
    current: u64,
}

impl Regression {
    fn delta(&self) -> i128 {
        i128::from(self.current) - i128::from(self.baseline)
    }
}

/// Machine-readable summary emitted under `--format json` / `--format junit`.
#[derive(Clone, Debug, Serialize)]
struct TestsReport {
    command: &'static str,
    current_path: String,
    baseline_path: String,
    baseline_totals: TotalsSummary,
    current_totals: TotalsSummary,
    regressions: Vec<Regression>,
    missing_shards: Vec<ShardKey>,
    exit: ResultStatus,
}

#[derive(Clone, Debug, Serialize)]
struct TotalsSummary {
    passed: u64,
    failed: u64,
}

#[derive(Clone, Debug, Serialize)]
struct ShardKey {
    #[serde(rename = "crate")]
    krate: String,
    shard: String,
}

fn tests(global: &GlobalArgs, args: &TestsArgs) -> anyhow::Result<ResultStatus> {
    let repo = RepoRoot::locate(Path::new("."))?;
    let results_dir = repo.join("evals").join("results").join("tests");
    let default_baseline = repo.join("ratchet").join("tests_baseline.json");

    // Resolve current + baseline paths.
    let current_path = match args.current.clone() {
        Some(p) => p,
        None => match newest_json(&results_dir) {
            Some(p) => p,
            None => {
                eprintln!(
                    "ratchet tests: no current test-matrix JSON found (looked in {}/)",
                    results_dir.display()
                );
                eprintln!("Run: scripts/run_full_test_matrix.sh");
                return Ok(ResultStatus::EnvBroken);
            }
        },
    };
    let baseline_path = args.baseline.clone().unwrap_or(default_baseline);

    if !current_path.is_file() {
        eprintln!(
            "ratchet tests: current file missing: {}",
            current_path.display()
        );
        return Ok(ResultStatus::EnvBroken);
    }
    if !baseline_path.is_file() {
        eprintln!(
            "ratchet tests: baseline file missing: {}",
            baseline_path.display()
        );
        eprintln!("To seed the baseline from the current run:");
        eprintln!(
            "  cp {} {}",
            current_path.display(),
            baseline_path.display()
        );
        return Ok(ResultStatus::EnvBroken);
    }

    let current_doc = load_matrix(&current_path)?;
    let baseline_doc = load_matrix(&baseline_path)?;

    let (regressions, missing) = diff_matrices(&baseline_doc, &current_doc);

    let exit = if regressions.is_empty() && missing.is_empty() {
        ResultStatus::Ok
    } else {
        ResultStatus::Failed
    };

    let report = TestsReport {
        command: "ratchet.tests",
        current_path: current_path.display().to_string(),
        baseline_path: baseline_path.display().to_string(),
        baseline_totals: TotalsSummary {
            passed: baseline_doc.totals.passed,
            failed: baseline_doc.totals.failed,
        },
        current_totals: TotalsSummary {
            passed: current_doc.totals.passed,
            failed: current_doc.totals.failed,
        },
        regressions: regressions.clone(),
        missing_shards: missing.clone(),
        exit,
    };

    match global.format {
        OutputFormat::Json | OutputFormat::Junit => {
            let json = serde_json::to_string_pretty(&report)?;
            println!("{json}");
        }
        OutputFormat::Human => print_tests_human(&report),
    }

    Ok(exit)
}

/// Parse a test-matrix JSON doc. Returns `EnvBroken` exit on failure via
/// the outer error context so callers can surface a clean error instead
/// of a Rust panic string.
fn load_matrix(path: &Path) -> anyhow::Result<TestMatrixDoc> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let doc: TestMatrixDoc = serde_json::from_str(&text)
        .with_context(|| format!("parsing {} as test-matrix JSON", path.display()))?;
    Ok(doc)
}

/// Pick the newest `*.json` in `dir` by lexicographic filename (matches
/// the shell script's `ls -1 … | sort | tail -n 1`). Returns `None` when
/// the directory is empty or missing. Non-JSON files and the
/// `.llvm2-verify.all.log` sidecar are ignored.
fn newest_json(dir: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    let mut jsons: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    jsons.sort();
    jsons.pop()
}

/// Return `(regressions, missing_shards)` for `current` vs `baseline`.
/// `missing_shards` lists baseline `(crate, shard)` pairs absent from
/// the current run — disappearance is its own failure mode per the
/// shell script's "can't silently delete" rule.
fn diff_matrices(
    baseline: &TestMatrixDoc,
    current: &TestMatrixDoc,
) -> (Vec<Regression>, Vec<ShardKey>) {
    let current_ix: std::collections::HashMap<(String, String), &ShardEntry> = current
        .shards
        .iter()
        .map(|s| ((s.krate.clone(), s.shard.clone()), s))
        .collect();

    let mut regressions: Vec<Regression> = Vec::new();
    let mut missing: Vec<ShardKey> = Vec::new();

    for base_s in &baseline.shards {
        let key = (base_s.krate.clone(), base_s.shard.clone());
        let Some(cur_s) = current_ix.get(&key) else {
            missing.push(ShardKey {
                krate: base_s.krate.clone(),
                shard: base_s.shard.clone(),
            });
            continue;
        };
        if cur_s.passed < base_s.passed {
            regressions.push(Regression {
                krate: base_s.krate.clone(),
                shard: base_s.shard.clone(),
                kind: RegressionKind::PassedDecreased,
                baseline: base_s.passed,
                current: cur_s.passed,
            });
        }
        if cur_s.failed > base_s.failed {
            regressions.push(Regression {
                krate: base_s.krate.clone(),
                shard: base_s.shard.clone(),
                kind: RegressionKind::FailedIncreased,
                baseline: base_s.failed,
                current: cur_s.failed,
            });
        }
    }

    (regressions, missing)
}

fn print_tests_human(r: &TestsReport) {
    if !r.missing_shards.is_empty() {
        println!("test ratchet FAILED: baseline shards missing from current run:");
        for k in &r.missing_shards {
            println!("  {}/{}", k.krate, k.shard);
        }
        println!();
    }

    if !r.regressions.is_empty() {
        println!("test ratchet FAILED: regression detected.");
        println!("  current:  {}", r.current_path);
        println!("  baseline: {}", r.baseline_path);
        println!();
        println!(
            "{:<42} {:<20} {:>8} {:>8} {:>8}",
            "crate/shard", "metric", "base", "cur", "delta"
        );
        for v in &r.regressions {
            let label = format!("{}/{}", v.krate, v.shard);
            println!(
                "{:<42} {:<20} {:>8} {:>8} {:>+8}",
                label,
                v.kind.as_str(),
                v.baseline,
                v.current,
                v.delta()
            );
        }
        println!();
        println!("Fix the regression, or refresh the baseline after approval:");
        println!("  cp {} {}", r.current_path, r.baseline_path);
        return;
    }

    if r.exit == ResultStatus::Failed {
        // Only reachable when missing shards but no metric regressions.
        return;
    }

    println!(
        "test ratchet OK: passed {} >= baseline {}; failed {} <= baseline {}.",
        r.current_totals.passed,
        r.baseline_totals.passed,
        r.current_totals.failed,
        r.baseline_totals.failed,
    );
}

// ---------------- shell-isolation ----------------

#[derive(Debug, Serialize)]
struct ShellViolation {
    path: String,
    line: u32,
    text: String,
}

fn shell_isolation(global: &GlobalArgs) -> anyhow::Result<ResultStatus> {
    let repo = RepoRoot::locate(Path::new("."))?;
    let crate_src = repo
        .join("crates")
        .join("llvm2-test")
        .join("src");
    let mut violations: Vec<ShellViolation> = Vec::new();
    walk_rs(&crate_src, &mut |path| {
        scan_file(&crate_src, path, &mut violations);
    })?;

    match global.format {
        OutputFormat::Json | OutputFormat::Junit => {
            let json = serde_json::to_string_pretty(&violations)?;
            println!("{json}");
        }
        OutputFormat::Human => {
            if violations.is_empty() {
                println!("ratchet shell-isolation: OK (no offenders)");
            } else {
                println!(
                    "ratchet shell-isolation: {} violation(s):",
                    violations.len()
                );
                for v in &violations {
                    println!("  {}:{}  {}", v.path, v.line, v.text.trim());
                }
            }
        }
    }

    Ok(if violations.is_empty() {
        ResultStatus::Ok
    } else {
        ResultStatus::Failed
    })
}

fn walk_rs(dir: &Path, cb: &mut dyn FnMut(&Path)) -> anyhow::Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_rs(&path, cb)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            cb(&path);
        }
    }
    Ok(())
}

fn scan_file(base: &Path, path: &Path, out: &mut Vec<ShellViolation>) {
    // Exempt files: shell.rs, external.rs, and this ratchet itself (the
    // only place we need to refer to the offending symbol as a literal).
    let rel = path.strip_prefix(base).unwrap_or(path);
    let rel_str = rel.to_string_lossy().replace('\\', "/");
    if rel_str == "shell.rs"
        || rel_str == "external.rs"
        || rel_str == "cmd/ratchet.rs"
    {
        return;
    }
    let Ok(text) = fs::read_to_string(path) else {
        return;
    };
    for (i, line) in text.lines().enumerate() {
        // Skip comments/doc lines.
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") || trimmed.starts_with("///") || trimmed.starts_with("//!") {
            continue;
        }
        // Skip lines that are entirely inside a string literal (crude but
        // sufficient — full tokenization is overkill for this check).
        let without_strings = strip_string_literals(line);
        let banned_fragment = "std::proc";
        let banned_symbol = "ess::Command"; // split intentionally to avoid self-trigger
        let full_symbol = format!("{banned_fragment}{banned_symbol}");
        if without_strings.contains(&full_symbol) {
            out.push(ShellViolation {
                path: path.to_string_lossy().into_owned(),
                line: u32::try_from(i + 1).unwrap_or(0),
                text: line.to_string(),
            });
        }
    }
}

fn strip_string_literals(line: &str) -> String {
    // Replace anything between paired `"` with a placeholder. Naive but
    // sufficient for this ratchet's scope (one-line literals only).
    let mut out = String::with_capacity(line.len());
    let mut in_str = false;
    let mut prev_bs = false;
    for ch in line.chars() {
        match ch {
            '"' if !prev_bs => {
                in_str = !in_str;
                out.push(' ');
            }
            _ if in_str => out.push(' '),
            _ => out.push(ch),
        }
        prev_bs = ch == '\\' && !prev_bs;
    }
    out
}

// ---------------- schema ----------------

fn schema(global: &GlobalArgs) -> anyhow::Result<ResultStatus> {
    let repo = RepoRoot::locate(Path::new("."))?;
    let schema_dir = repo.join("evals").join("schema");
    fs::create_dir_all(&schema_dir)?;

    let generated = generate_schemas();

    // Compare each schema with the checked-in file.
    let mut drift: Vec<(String, String)> = Vec::new();
    let mut wrote: Vec<String> = Vec::new();
    for (name, json) in &generated {
        let target = schema_dir.join(format!("{name}.json"));
        let current = fs::read_to_string(&target).ok();
        if current.as_deref() != Some(json.as_str()) {
            if current.is_none() {
                wrote.push(name.clone());
                fs::write(&target, json)?;
            } else {
                drift.push((name.clone(), target.display().to_string()));
            }
        }
    }

    let exit = if drift.is_empty() {
        ResultStatus::Ok
    } else {
        ResultStatus::Failed
    };

    match global.format {
        OutputFormat::Json | OutputFormat::Junit => {
            let report = serde_json::json!({
                "command": "ratchet.schema",
                "drift": drift.iter().map(|(n, p)| serde_json::json!({"name": n, "path": p})).collect::<Vec<_>>(),
                "wrote": wrote,
                "exit": exit,
            });
            println!("{report:#}");
        }
        OutputFormat::Human => {
            if drift.is_empty() {
                println!("ratchet schema: OK ({} schemas checked)", generated.len());
                if !wrote.is_empty() {
                    println!(
                        "  initial schemas written: {}",
                        wrote.join(", ")
                    );
                }
            } else {
                println!("ratchet schema: drift in {} schema(s):", drift.len());
                for (name, path) in &drift {
                    println!("  {name}  ({path})");
                }
                println!(
                    "  regenerate with `llvm2-test ratchet schema --apply` (not yet wired)"
                );
            }
        }
    }

    Ok(exit)
}

fn generate_schemas() -> Vec<(String, String)> {
    let s_run = schema_for!(RunSummary<EmptyRecord>);
    let run_summary = serde_json::to_string_pretty(&s_run).unwrap_or_default();
    let s_empty = schema_for!(EmptyRecord);
    let empty = serde_json::to_string_pretty(&s_empty).unwrap_or_default();
    vec![
        ("run_summary".to_string(), run_summary),
        ("empty_record".to_string(), empty),
    ]
}

// Unused helper kept for link completeness when future ratchets are added.
#[allow(dead_code)]
fn unused_path_marker() -> PathBuf {
    PathBuf::new()
}

// ---------------- warnings ----------------
//
// Port of `scripts/check_warnings_ratchet.sh`. Builds
// `cargo build --workspace` and `cargo build --workspace --tests`, counts
// distinct `warning:` diagnostics in each output, and compares the sum
// against `ratchet/warnings_baseline.json` (default baseline: 0).
//
// Exit semantics match the shell script:
//   * `ResultStatus::Ok` (0)          — total <= baseline
//   * `ResultStatus::Failed` (1)      — total > baseline
//   * `ResultStatus::EnvBroken` (2)   — missing cargo / baseline file

/// Schema for `ratchet/warnings_baseline.json`.
#[derive(Clone, Debug, Deserialize, Serialize)]
struct WarningsBaseline {
    /// Allowed maximum number of warning diagnostics across both builds.
    warnings: u64,
}

/// Machine-readable summary emitted under `--format json`.
#[derive(Clone, Debug, Serialize)]
struct WarningsReport {
    command: &'static str,
    baseline: u64,
    prod_warnings: u64,
    tests_warnings: u64,
    total: u64,
    prod_samples: Vec<String>,
    tests_samples: Vec<String>,
    exit: ResultStatus,
}

fn warnings(global: &GlobalArgs) -> anyhow::Result<ResultStatus> {
    let repo = RepoRoot::locate(Path::new("."))?;
    let baseline_path = repo.join("ratchet").join("warnings_baseline.json");

    // Env broken: no baseline file.
    if !baseline_path.is_file() {
        eprintln!(
            "ratchet warnings: baseline file missing: {}",
            baseline_path.display()
        );
        return Ok(ResultStatus::EnvBroken);
    }
    // Env broken: no cargo on PATH.
    if crate::shell::which(Cargo::program()).is_none() {
        eprintln!("ratchet warnings: cargo not found on PATH");
        return Ok(ResultStatus::EnvBroken);
    }

    let baseline_text = fs::read_to_string(&baseline_path)
        .with_context(|| format!("reading {}", baseline_path.display()))?;
    let baseline: WarningsBaseline = serde_json::from_str(&baseline_text).with_context(|| {
        format!(
            "parsing {} as {{\"warnings\": <u64>}}",
            baseline_path.display()
        )
    })?;

    // Run the two builds. `deny(warnings)` can turn warnings into errors,
    // so non-zero exit is expected and intentionally ignored — we still
    // want to count diagnostics for the report.
    let prod_out = cargo_build_output(&repo, &["build", "--workspace"])?;
    let tests_out = cargo_build_output(&repo, &["build", "--workspace", "--tests"])?;

    let prod_diags = extract_warning_lines(&prod_out);
    let tests_diags = extract_warning_lines(&tests_out);
    let prod_count = u64::try_from(prod_diags.len()).unwrap_or(u64::MAX);
    let tests_count = u64::try_from(tests_diags.len()).unwrap_or(u64::MAX);
    let total = prod_count.saturating_add(tests_count);

    let exit = if total > baseline.warnings {
        ResultStatus::Failed
    } else {
        ResultStatus::Ok
    };

    let report = WarningsReport {
        command: "ratchet.warnings",
        baseline: baseline.warnings,
        prod_warnings: prod_count,
        tests_warnings: tests_count,
        total,
        prod_samples: prod_diags.clone(),
        tests_samples: tests_diags.clone(),
        exit,
    };

    match global.format {
        OutputFormat::Json | OutputFormat::Junit => {
            let json = serde_json::to_string_pretty(&report)?;
            println!("{json}");
        }
        OutputFormat::Human => print_warnings_human(&report),
    }

    Ok(exit)
}

/// Run `cargo <args>` at `repo` with `--message-format=human --color=never`
/// and capture combined stdout+stderr. A non-zero exit is not an error
/// here — `deny(warnings)` can turn warnings into hard errors, and the
/// ratchet's job is to count diagnostics regardless.
fn cargo_build_output(repo: &RepoRoot, args: &[&str]) -> anyhow::Result<String> {
    let mut spawn = Spawn::new(Cargo::program()).cwd(repo.0.clone());
    for a in args {
        spawn = spawn.arg(*a);
    }
    spawn = spawn.arg("--message-format=human").arg("--color=never");
    let captured = spawn.capture()?;
    let mut combined = String::with_capacity(captured.stdout.len() + captured.stderr.len());
    combined.push_str(&captured.stdout);
    combined.push_str(&captured.stderr);
    Ok(combined)
}

/// Extract the set of `warning: <...>` diagnostic lines from cargo output.
///
/// Mirrors the shell script's three-stage grep pipeline:
///   1. Keep only lines starting with `warning: `.
///   2. Drop the per-crate summary line
///      (`warning: \`<crate>\` (...) generated N warnings?`).
///   3. Drop cargo network/manifest warnings we never want to gate on.
///
/// The returned `Vec` preserves source order so the `--format human`
/// view matches the shell-script output verbatim.
fn extract_warning_lines(output: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in output.lines() {
        if !line.starts_with("warning: ") {
            continue;
        }
        if is_crate_summary(line) {
            continue;
        }
        if is_cargo_housekeeping(line) {
            continue;
        }
        out.push(line.to_string());
    }
    out
}

/// Match the shell script's `^warning: \`[^\`]+\` \(.*\) generated [0-9]+ warnings?`.
fn is_crate_summary(line: &str) -> bool {
    // Strip the leading `warning: ` prefix.
    let rest = match line.strip_prefix("warning: ") {
        Some(r) => r,
        None => return false,
    };
    // Must begin with a backtick-quoted crate name.
    let after_first_tick = match rest.strip_prefix('`') {
        Some(r) => r,
        None => return false,
    };
    let close = match after_first_tick.find('`') {
        Some(i) => i,
        None => return false,
    };
    let after_close = &after_first_tick[close + 1..];
    // Followed by ` (...) `.
    let trimmed = after_close.trim_start();
    let after_paren = match trimmed.strip_prefix('(') {
        Some(r) => r,
        None => return false,
    };
    let close_paren = match after_paren.find(')') {
        Some(i) => i,
        None => return false,
    };
    let tail = after_paren[close_paren + 1..].trim_start();
    // ...then literally `generated N warning` or `generated N warnings`.
    let rest = match tail.strip_prefix("generated ") {
        Some(r) => r,
        None => return false,
    };
    // Accept `<digits> warning(s)?...` without consuming anything after.
    let digit_end = rest
        .char_indices()
        .find(|(_, ch)| !ch.is_ascii_digit())
        .map_or(rest.len(), |(i, _)| i);
    if digit_end == 0 {
        return false;
    }
    let after_digits = rest[digit_end..].trim_start();
    after_digits.starts_with("warning")
}

/// Match the shell script's second negative filter:
/// `^warning: (spurious network error|unused manifest key)`.
fn is_cargo_housekeeping(line: &str) -> bool {
    let rest = match line.strip_prefix("warning: ") {
        Some(r) => r,
        None => return false,
    };
    rest.starts_with("spurious network error") || rest.starts_with("unused manifest key")
}

fn print_warnings_human(r: &WarningsReport) {
    println!("warnings ratchet: baseline={}", r.baseline);
    println!("  cargo build --workspace:          {} warnings", r.prod_warnings);
    println!("  cargo build --workspace --tests:  {} warnings", r.tests_warnings);
    println!("  total:                             {}", r.total);
    if r.exit == ResultStatus::Failed {
        println!();
        println!(
            "warnings ratchet FAILED: total {} > baseline {}.",
            r.total, r.baseline
        );
        println!();
        println!("Production warnings:");
        for w in &r.prod_samples {
            println!("{w}");
        }
        println!();
        println!("Test warnings:");
        for w in &r.tests_samples {
            println!("{w}");
        }
        println!();
        println!(
            "Fix the warnings, or update ratchet/warnings_baseline.json and \
             justify in commit body."
        );
    } else {
        println!(
            "warnings ratchet OK: total {} <= baseline {}.",
            r.total, r.baseline
        );
    }
}

// ---------------- unwrap (panic-family site ratchet) ----------------
//
// Port of `scripts/check_unwrap_ratchet.sh` (#385 / Part of #372). The
// shell script shells out to `scripts/generate_unwrap_audit.py --write`
// to compute fresh per-file counts and then diffs them against the
// checked-in `ratchet/unwrap_baseline.json`. This Rust port keeps the
// Python generator as the scanner (still the source of truth for
// category labels in `docs/panic_audit.md`), but moves the diff + exit
// contract into `llvm2-test` so CI can depend on a single binary.
//
// Exit semantics match the shell script:
//   * `ResultStatus::Ok` (0)          — every file is at or below baseline
//   * `ResultStatus::Failed` (1)      — at least one file exceeds baseline
//   * `ResultStatus::EnvBroken` (2)   — missing python3 / baseline / generator

/// Top-level schema for `ratchet/unwrap_baseline.json`. Only the fields
/// the ratchet reads are modeled — `schema_version`, `note`, etc. are
/// ignored so the generator can keep adding metadata.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct UnwrapBaseline {
    #[serde(default)]
    per_file: std::collections::BTreeMap<String, u64>,
    #[serde(default)]
    total: u64,
}

/// One per-file regression finding.
#[derive(Clone, Debug, Serialize)]
struct UnwrapViolation {
    file: String,
    baseline: u64,
    current: u64,
}

impl UnwrapViolation {
    fn delta(&self) -> i128 {
        i128::from(self.current) - i128::from(self.baseline)
    }
}

/// Machine-readable summary emitted under `--format json` / `--format junit`.
#[derive(Clone, Debug, Serialize)]
struct UnwrapReport {
    command: &'static str,
    baseline_path: String,
    current_path: String,
    baseline_total: u64,
    current_total: u64,
    reduced: u64,
    violations: Vec<UnwrapViolation>,
    exit: ResultStatus,
}

fn unwrap(global: &GlobalArgs) -> anyhow::Result<ResultStatus> {
    let repo = RepoRoot::locate(Path::new("."))?;
    let baseline_path = repo.join("ratchet").join("unwrap_baseline.json");
    let generator = repo.join("scripts").join("generate_unwrap_audit.py");

    // Env broken: no python3, missing baseline, missing generator.
    if crate::shell::which("python3").is_none() {
        eprintln!("ratchet unwrap: python3 not found on PATH");
        return Ok(ResultStatus::EnvBroken);
    }
    if !baseline_path.is_file() {
        eprintln!(
            "ratchet unwrap: baseline file missing: {}",
            baseline_path.display()
        );
        eprintln!("Run: python3 scripts/generate_unwrap_audit.py --write");
        return Ok(ResultStatus::EnvBroken);
    }
    if !generator.is_file() {
        eprintln!(
            "ratchet unwrap: generator missing: {}",
            generator.display()
        );
        return Ok(ResultStatus::EnvBroken);
    }

    // Write current counts to a temp file, then diff. Mirrors the shell
    // script's `mktemp` + `--baseline-out $TMP` flow.
    let tmp_dir = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp_current = tmp_dir.join(format!(
        "unwrap_current_pid{}_{}.json",
        std::process::id(),
        nanos
    ));

    let mut spawn = Spawn::new("python3").cwd(repo.0.clone());
    spawn = spawn
        .arg(generator.display().to_string())
        .arg("--write")
        .arg("--baseline-out")
        .arg(tmp_current.display().to_string())
        .arg("--audit-out")
        .arg("/dev/null");
    let captured = spawn.capture()?;
    if !captured.success() {
        eprintln!(
            "ratchet unwrap: generator failed (exit {})",
            captured.code
        );
        if !captured.stderr.is_empty() {
            eprintln!("{}", captured.stderr);
        }
        // Clean up temp file if the generator wrote it anyway.
        let _ = fs::remove_file(&tmp_current);
        return Ok(ResultStatus::EnvBroken);
    }

    let baseline_text = fs::read_to_string(&baseline_path)
        .with_context(|| format!("reading {}", baseline_path.display()))?;
    let baseline: UnwrapBaseline = serde_json::from_str(&baseline_text)
        .with_context(|| format!("parsing {} as unwrap baseline", baseline_path.display()))?;

    let current_text = fs::read_to_string(&tmp_current)
        .with_context(|| format!("reading {}", tmp_current.display()))?;
    let current: UnwrapBaseline = serde_json::from_str(&current_text)
        .with_context(|| format!("parsing {} as unwrap baseline", tmp_current.display()))?;

    // Always clean the temp file once we've loaded it.
    let _ = fs::remove_file(&tmp_current);

    let (violations, reduced) = diff_unwrap(&baseline, &current);

    let exit = if violations.is_empty() {
        ResultStatus::Ok
    } else {
        ResultStatus::Failed
    };

    let report = UnwrapReport {
        command: "ratchet.unwrap",
        baseline_path: baseline_path.display().to_string(),
        current_path: tmp_current.display().to_string(),
        baseline_total: baseline.total,
        current_total: current.total,
        reduced,
        violations: violations.clone(),
        exit,
    };

    match global.format {
        OutputFormat::Json | OutputFormat::Junit => {
            let json = serde_json::to_string_pretty(&report)?;
            println!("{json}");
        }
        OutputFormat::Human => print_unwrap_human(&report),
    }

    Ok(exit)
}

/// Return `(violations, reduced)` for `current` vs `baseline`.
/// `violations` lists files whose production panic-family count rose;
/// `reduced` is the sum of per-file decreases (informational only).
fn diff_unwrap(
    baseline: &UnwrapBaseline,
    current: &UnwrapBaseline,
) -> (Vec<UnwrapViolation>, u64) {
    let mut violations: Vec<UnwrapViolation> = Vec::new();
    let mut reduced: u64 = 0;

    let mut all_files: std::collections::BTreeSet<&String> =
        std::collections::BTreeSet::new();
    all_files.extend(baseline.per_file.keys());
    all_files.extend(current.per_file.keys());

    for f in all_files {
        let base_n = baseline.per_file.get(f).copied().unwrap_or(0);
        let cur_n = current.per_file.get(f).copied().unwrap_or(0);
        if cur_n > base_n {
            violations.push(UnwrapViolation {
                file: f.clone(),
                baseline: base_n,
                current: cur_n,
            });
        } else if cur_n < base_n {
            reduced = reduced.saturating_add(base_n - cur_n);
        }
    }

    (violations, reduced)
}

fn print_unwrap_human(r: &UnwrapReport) {
    if !r.violations.is_empty() {
        println!("unwrap ratchet FAILED: production panic-family count increased.");
        println!("  baseline: {}", r.baseline_path);
        println!("  current:  {}", r.current_path);
        println!();
        println!("{:<60} {:>6} {:>6} {:>6}", "file", "base", "cur", "delta");
        for v in &r.violations {
            println!(
                "{:<60} {:>6} {:>6} {:>+6}",
                v.file,
                v.baseline,
                v.current,
                v.delta()
            );
        }
        println!();
        println!("Every production .unwrap() / .expect( / panic! / unreachable! / todo!");
        println!("added outside #[cfg(test)] must be matched by a removal elsewhere in");
        println!("the same file, or you must update the baseline with:");
        println!("  python3 scripts/generate_unwrap_audit.py --write");
        println!("and justify the increase in the commit body.");
        return;
    }

    println!(
        "unwrap ratchet OK: total {} <= baseline {} (reduced {}).",
        r.current_total, r.baseline_total, r.reduced
    );
}

#[cfg(test)]
mod tests {
    use clap::{Command, Subcommand};

    use super::{
        RatchetCommand, RegressionKind, ShardEntry, TestMatrixDoc, TotalsEntry,
        UnwrapBaseline, diff_matrices, diff_unwrap, extract_warning_lines,
        is_cargo_housekeeping, is_crate_summary, newest_json,
    };

    /// Guard against a silent drop of a subcommand from the ratchet tree.
    /// Every ratchet port must appear here so clap can reach it; if the
    /// `RatchetCommand` enum ever loses a variant, this test breaks
    /// loudly instead of the subcommand vanishing from CI (#446).
    #[test]
    fn ratchet_subcommand_tree_is_complete() {
        let cmd = <RatchetCommand as Subcommand>::augment_subcommands(
            Command::new("ratchet-test"),
        );
        let subs: Vec<String> = cmd
            .get_subcommands()
            .map(|c| c.get_name().to_string())
            .collect();
        for expected in [
            "tests",
            "warnings",
            "unwrap",
            "shell-isolation",
            "schema",
            "lint-linux",
        ] {
            assert!(
                subs.iter().any(|s| s == expected),
                "ratchet subcommand `{expected}` missing from RatchetCommand tree; \
                 present: {subs:?}"
            );
        }
    }

    #[test]
    fn counts_source_warnings_only() {
        let sample = "\
Compiling foo v0.1.0
warning: unused variable: `x`
  --> src/lib.rs:3:9
warning: `foo` (lib) generated 1 warning
warning: unused import: `bar`
  --> src/lib.rs:1:5
warning: `foo` (lib test) generated 1 warning
    Finished dev profile
";
        let diags = extract_warning_lines(sample);
        assert_eq!(diags.len(), 2, "only source diagnostics, not summaries");
        assert_eq!(diags[0], "warning: unused variable: `x`");
        assert_eq!(diags[1], "warning: unused import: `bar`");
    }

    #[test]
    fn filters_cargo_housekeeping() {
        let sample = "\
warning: spurious network error (2 tries remaining): timeout
warning: unused manifest key: package.metadata.foo
warning: dead_code — real diagnostic
warning: `mycrate` (bin \"x\") generated 0 warnings
";
        let diags = extract_warning_lines(sample);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0], "warning: dead_code — real diagnostic");
    }

    #[test]
    fn crate_summary_matcher_is_strict() {
        assert!(is_crate_summary(
            "warning: `llvm2-opt` (lib) generated 3 warnings"
        ));
        assert!(is_crate_summary(
            "warning: `llvm2-opt` (lib test) generated 1 warning"
        ));
        // Not a summary — real diagnostic that happens to mention generated.
        assert!(!is_crate_summary(
            "warning: code generated from macro"
        ));
        // Missing trailing word.
        assert!(!is_crate_summary(
            "warning: `llvm2-opt` (lib) generated 3 issues"
        ));
    }

    #[test]
    fn housekeeping_matcher_only_hits_prefixes() {
        assert!(is_cargo_housekeeping(
            "warning: spurious network error (2 tries remaining): timeout"
        ));
        assert!(is_cargo_housekeeping(
            "warning: unused manifest key: package.metadata.foo"
        ));
        assert!(!is_cargo_housekeeping(
            "warning: unused variable: spurious_network_error"
        ));
    }

    #[test]
    fn empty_input_yields_no_warnings() {
        assert!(extract_warning_lines("").is_empty());
        assert!(extract_warning_lines("Finished dev [unoptimized]\n").is_empty());
    }

    // -------- tests (pass-count ratchet) helpers --------

    fn shard(krate: &str, shard: &str, passed: u64, failed: u64) -> ShardEntry {
        ShardEntry {
            krate: krate.to_string(),
            shard: shard.to_string(),
            passed,
            failed,
        }
    }

    fn doc(shards: Vec<ShardEntry>) -> TestMatrixDoc {
        let (tp, tf) = shards
            .iter()
            .fold((0u64, 0u64), |(p, f), s| (p + s.passed, f + s.failed));
        TestMatrixDoc {
            shards,
            totals: TotalsEntry {
                passed: tp,
                failed: tf,
            },
        }
    }

    #[test]
    fn diff_no_change_is_clean() {
        let base = doc(vec![shard("c1", "all", 10, 0), shard("c2", "lib", 5, 1)]);
        let cur = base.clone();
        let (regressions, missing) = diff_matrices(&base, &cur);
        assert!(regressions.is_empty());
        assert!(missing.is_empty());
    }

    #[test]
    fn diff_catches_passed_decrease() {
        let base = doc(vec![shard("c1", "all", 10, 0)]);
        let cur = doc(vec![shard("c1", "all", 9, 0)]);
        let (regressions, missing) = diff_matrices(&base, &cur);
        assert!(missing.is_empty());
        assert_eq!(regressions.len(), 1);
        assert_eq!(regressions[0].kind, RegressionKind::PassedDecreased);
        assert_eq!(regressions[0].baseline, 10);
        assert_eq!(regressions[0].current, 9);
        assert_eq!(regressions[0].delta(), -1);
    }

    #[test]
    fn diff_catches_failed_increase() {
        let base = doc(vec![shard("c1", "all", 10, 0)]);
        let cur = doc(vec![shard("c1", "all", 10, 3)]);
        let (regressions, missing) = diff_matrices(&base, &cur);
        assert!(missing.is_empty());
        assert_eq!(regressions.len(), 1);
        assert_eq!(regressions[0].kind, RegressionKind::FailedIncreased);
        assert_eq!(regressions[0].baseline, 0);
        assert_eq!(regressions[0].current, 3);
    }

    #[test]
    fn diff_reports_both_metric_breaks() {
        let base = doc(vec![shard("c1", "all", 10, 0)]);
        let cur = doc(vec![shard("c1", "all", 8, 2)]);
        let (regressions, missing) = diff_matrices(&base, &cur);
        assert!(missing.is_empty());
        assert_eq!(regressions.len(), 2);
        let kinds: Vec<_> = regressions.iter().map(|r| r.kind).collect();
        assert!(kinds.contains(&RegressionKind::PassedDecreased));
        assert!(kinds.contains(&RegressionKind::FailedIncreased));
    }

    #[test]
    fn diff_flags_missing_shards() {
        let base = doc(vec![shard("c1", "all", 10, 0), shard("c2", "lib", 5, 0)]);
        let cur = doc(vec![shard("c1", "all", 10, 0)]);
        let (regressions, missing) = diff_matrices(&base, &cur);
        assert!(regressions.is_empty());
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].krate, "c2");
        assert_eq!(missing[0].shard, "lib");
    }

    #[test]
    fn diff_accepts_new_shards_in_current() {
        // New `(crate, shard)` pairs in `current` that aren't in baseline
        // must not trigger a regression — the baseline grows monotonically.
        let base = doc(vec![shard("c1", "all", 10, 0)]);
        let cur = doc(vec![shard("c1", "all", 10, 0), shard("c2", "new", 3, 0)]);
        let (regressions, missing) = diff_matrices(&base, &cur);
        assert!(regressions.is_empty());
        assert!(missing.is_empty());
    }

    #[test]
    fn diff_accepts_increased_passed_and_decreased_failed() {
        let base = doc(vec![shard("c1", "all", 10, 2)]);
        let cur = doc(vec![shard("c1", "all", 12, 1)]);
        let (regressions, missing) = diff_matrices(&base, &cur);
        assert!(regressions.is_empty());
        assert!(missing.is_empty());
    }

    #[test]
    fn newest_json_picks_lexicographically_latest() {
        let tmp = tempdir_unique("ratchet_newest_json_ok");
        std::fs::write(tmp.join("2026-04-17.json"), "{}").expect("write");
        std::fs::write(tmp.join("2026-04-19.json"), "{}").expect("write");
        std::fs::write(tmp.join("2026-04-18.json"), "{}").expect("write");
        // Sidecar log file — must be ignored.
        std::fs::write(tmp.join("2026-04-19.llvm2-verify.all.log"), "x").expect("write");
        let pick = newest_json(&tmp).expect("some");
        assert_eq!(pick.file_name().unwrap(), "2026-04-19.json");
    }

    #[test]
    fn newest_json_returns_none_for_missing_or_empty_dir() {
        let missing = std::env::temp_dir().join("llvm2_ratchet_missing_XYZ_does_not_exist");
        assert!(newest_json(&missing).is_none());
        let empty = tempdir_unique("ratchet_newest_json_empty");
        assert!(newest_json(&empty).is_none());
    }

    fn tempdir_unique(tag: &str) -> std::path::PathBuf {
        // Crude but dependency-free unique tempdir. `std::env::temp_dir()`
        // + pid + nanoseconds is sufficient for this crate's test volume.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let p = std::env::temp_dir().join(format!(
            "llvm2_{}_pid{}_{}",
            tag,
            std::process::id(),
            nanos
        ));
        std::fs::create_dir_all(&p).expect("mkdir tempdir");
        p
    }

    // -------- unwrap (panic-family site ratchet) helpers --------

    fn unwrap_doc(files: &[(&str, u64)]) -> UnwrapBaseline {
        let mut per_file = std::collections::BTreeMap::new();
        let mut total: u64 = 0;
        for (f, n) in files {
            per_file.insert((*f).to_string(), *n);
            total += *n;
        }
        UnwrapBaseline { per_file, total }
    }

    #[test]
    fn unwrap_no_change_is_clean() {
        let base = unwrap_doc(&[("crates/foo/src/a.rs", 3), ("crates/bar/src/b.rs", 1)]);
        let cur = base.clone();
        let (violations, reduced) = diff_unwrap(&base, &cur);
        assert!(violations.is_empty());
        assert_eq!(reduced, 0);
    }

    #[test]
    fn unwrap_catches_per_file_increase() {
        let base = unwrap_doc(&[("crates/foo/src/a.rs", 3)]);
        let cur = unwrap_doc(&[("crates/foo/src/a.rs", 5)]);
        let (violations, reduced) = diff_unwrap(&base, &cur);
        assert_eq!(reduced, 0);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].file, "crates/foo/src/a.rs");
        assert_eq!(violations[0].baseline, 3);
        assert_eq!(violations[0].current, 5);
        assert_eq!(violations[0].delta(), 2);
    }

    #[test]
    fn unwrap_new_file_triggers_violation() {
        // A brand-new file with non-zero production sites is a regression
        // (baseline implicitly 0 via the BTreeMap miss).
        let base = unwrap_doc(&[("crates/foo/src/a.rs", 3)]);
        let cur = unwrap_doc(&[
            ("crates/foo/src/a.rs", 3),
            ("crates/foo/src/b.rs", 2),
        ]);
        let (violations, reduced) = diff_unwrap(&base, &cur);
        assert_eq!(reduced, 0);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].file, "crates/foo/src/b.rs");
        assert_eq!(violations[0].baseline, 0);
        assert_eq!(violations[0].current, 2);
    }

    #[test]
    fn unwrap_per_file_decrease_counted_as_reduced() {
        let base = unwrap_doc(&[("crates/foo/src/a.rs", 5)]);
        let cur = unwrap_doc(&[("crates/foo/src/a.rs", 2)]);
        let (violations, reduced) = diff_unwrap(&base, &cur);
        assert!(violations.is_empty());
        assert_eq!(reduced, 3);
    }

    #[test]
    fn unwrap_file_fully_removed_counts_as_reduced() {
        // A file that dropped to 0 sites (fully cleaned up) is a win:
        // no violation, and the removal counts toward `reduced`.
        let base = unwrap_doc(&[
            ("crates/foo/src/a.rs", 5),
            ("crates/foo/src/b.rs", 2),
        ]);
        let cur = unwrap_doc(&[("crates/foo/src/a.rs", 5)]);
        let (violations, reduced) = diff_unwrap(&base, &cur);
        assert!(violations.is_empty());
        assert_eq!(reduced, 2);
    }

    #[test]
    fn unwrap_reports_multiple_increases() {
        let base = unwrap_doc(&[
            ("crates/a.rs", 1),
            ("crates/b.rs", 1),
            ("crates/c.rs", 1),
        ]);
        let cur = unwrap_doc(&[
            ("crates/a.rs", 2),
            ("crates/b.rs", 4),
            ("crates/c.rs", 1),
        ]);
        let (violations, reduced) = diff_unwrap(&base, &cur);
        assert_eq!(reduced, 0);
        assert_eq!(violations.len(), 2);
        // BTreeSet iteration yields files in lexicographic order.
        assert_eq!(violations[0].file, "crates/a.rs");
        assert_eq!(violations[1].file, "crates/b.rs");
    }

    #[test]
    fn unwrap_parses_real_baseline_schema() {
        // The generator emits a superset of fields (schema_version, note,
        // excluded_crates, per_crate_totals, per_file, total). The parser
        // must tolerate the extras without error.
        let text = r#"{
  "schema_version": 1,
  "note": "...",
  "excluded_crates": ["llvm2-verify"],
  "per_crate_totals": {"llvm2-opt": 8},
  "per_file": {"crates/llvm2-opt/src/gvn.rs": 1},
  "total": 1
}"#;
        let b: UnwrapBaseline = serde_json::from_str(text).expect("parse");
        assert_eq!(b.total, 1);
        assert_eq!(
            b.per_file.get("crates/llvm2-opt/src/gvn.rs").copied(),
            Some(1u64)
        );
    }
}
