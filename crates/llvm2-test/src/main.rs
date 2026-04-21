// llvm2-test — unified test and verification CLI entry point.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// This binary is the ONLY entry point for every workstream (WS1-WS9) in the
// "Proving LLVM2 replaces LLVM for Rust" program. See
// `designs/2026-04-19-proving-llvm2-replaces-llvm.md`.
//
// Design rules enforced here:
//   * One CLI — no scripts, no Makefile, no `cargo run -p <tool>`.
//   * Every subcommand parses argv via clap-derive, writes a typed JSON
//     result under `evals/results/<cmd>/<iso-date>.json`, and prints a
//     human table under `--format human`.
//   * Exit codes map to `ResultStatus` (sysexits.h aligned). See
//     `results::ResultStatus`.
//
// Adding a subcommand? Read `docs/testing/architecture.md` first.

#![deny(missing_docs)]
#![deny(clippy::pedantic)]
#![deny(warnings)]
#![allow(clippy::module_name_repetitions)]
// Doc strings intentionally carry natural prose (subcommand help text).
// Backticking every product name (YARPGen, JUnit, SingleSource, MIR) in
// operator-facing `--help` output is worse UX than the lint suggests.
#![allow(clippy::doc_markdown)]
// `match` expressions with a single arm that binds make the intent clearer
// than `if let` in several of this crate's dispatch sites. Allow both.
#![allow(clippy::single_match_else)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::format_push_string)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::ref_option)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::manual_contains)]
#![allow(clippy::collapsible_if)]
// The skeleton exposes APIs that WS1-WS9 will call in follow-up PRs.
// They appear unused here; allow that explicitly rather than littering
// the code with per-item `#[allow]`.
#![allow(dead_code)]

//! Unified test and verification CLI for LLVM2.
//!
//! Run `llvm2-test --help` for the command tree. Every subcommand supports
//! `--format {human,json,junit}`, `-o/--out`, `--config`, `--timeout`,
//! `--parallel`, `-q/--quiet`, `-v/--verbose`, `--no-cache`, `--dry-run`.

use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use tracing_subscriber::EnvFilter;

mod cmd;
mod config;
mod corpus;
mod external;
mod progress;
mod results;
mod shell;

use results::ResultStatus;

/// Output format for result-emitting subcommands.
#[derive(Clone, Copy, Debug, ValueEnum, Default)]
#[value(rename_all = "lower")]
pub enum OutputFormat {
    /// Pretty table + narrative, targeted at humans.
    #[default]
    Human,
    /// Machine-readable JSON following the schemas in `evals/schema/`.
    Json,
    /// JUnit XML for CI consumers.
    Junit,
}

/// Top-level `llvm2-test` CLI.
#[derive(Parser, Debug)]
#[command(
    name = "llvm2-test",
    version,
    about = "Unified test and verification CLI for LLVM2",
    long_about = "The single entry point for every test, fuzz, and proof \
                  workstream in the LLVM2 program. See \
                  `designs/2026-04-19-proving-llvm2-replaces-llvm.md` and \
                  `docs/testing/README.md`. No shell scripts. No Python. \
                  Every subcommand ships `--format json`, a `# Examples` \
                  block in its `--help`, and a typed schema under \
                  `evals/schema/`.\n\n\
                  # Examples\n\n  \
                  llvm2-test matrix --format human\n  \
                  llvm2-test doctor --for fuzz\n  \
                  llvm2-test report --out reports/weekly/2026-04-19.md"
)]
pub struct Cli {
    /// Output format (human, json, junit).
    #[arg(short = 'f', long, value_enum, default_value_t = OutputFormat::Human, global = true)]
    pub format: OutputFormat,

    /// Write machine-readable result artifact here. Default:
    /// `evals/results/<cmd>/<iso-date>.json`.
    #[arg(short = 'o', long, value_name = "PATH", global = true)]
    pub out: Option<std::path::PathBuf>,

    /// Config file (TOML). Default: `llvm2-test.toml` at repo root.
    #[arg(long, value_name = "PATH", global = true)]
    pub config: Option<std::path::PathBuf>,

    /// Per-unit timeout in seconds. Subcommands may apply per-unit
    /// defaults when this is unset.
    #[arg(long, value_name = "SECS", global = true)]
    pub timeout: Option<u64>,

    /// Worker count (honors cargo-serialization lock). Default: auto.
    #[arg(long, value_name = "N", global = true)]
    pub parallel: Option<usize>,

    /// Suppress progress bars; errors still go to stderr.
    #[arg(short = 'q', long, global = true)]
    pub quiet: bool,

    /// Show debug-level events. Repeat for more: -vv, -vvv.
    #[arg(short = 'v', long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    /// Ignore proof / corpus caches.
    #[arg(long, global = true)]
    pub no_cache: bool,

    /// Print what would run without executing external tools.
    #[arg(long, global = true)]
    pub dry_run: bool,

    /// Subcommand.
    #[command(subcommand)]
    pub command: Command,
}

/// Every workstream's entry point. See `docs/testing/<name>.md` for the
/// deep dive on each subcommand.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// WS1 — Run the workspace unit/integration test matrix.
    Matrix(cmd::matrix::MatrixArgs),

    /// WS2 — Run the external llvm-test-suite SingleSource corpus.
    Suite(cmd::suite::SuiteArgs),

    /// WS3 — Run differential fuzzers: csmith, yarpgen, tmir-gen.
    Fuzz(cmd::fuzz::FuzzArgs),

    /// WS4 — Drive `rustc_codegen_llvm2` + rustc UI tests.
    Rustc(cmd::rustc::RustcArgs),

    /// WS5 — Bootstrap rustc with LLVM2.
    Bootstrap(cmd::bootstrap::BootstrapArgs),

    /// WS6 — Run `cargo test` on top-100 crates.io.
    Ecosystem(cmd::ecosystem::EcosystemArgs),

    /// WS7 — Discharge z4 lowering obligations.
    Prove(cmd::prove::ProveArgs),

    /// WS8 — Prove non-ISel stages: regalloc, scheduler, emit.
    Pipeline(cmd::pipeline::PipelineArgs),

    /// WS9 — Generate weekly report + dashboard.
    Report(cmd::report::ReportArgs),

    /// Run ratchet checks (called by CI).
    Ratchet(cmd::ratchet::RatchetArgs),

    /// Check environment for required tools.
    Doctor(cmd::doctor::DoctorArgs),

    /// Cross-compile check for Linux `#[cfg(target_os = "linux")]` paths
    /// in `llvm2-codegen` (issue #346). Safe on macOS-only boxes — targets
    /// not installed via `rustup target add` are reported as `skipped`.
    LintLinux(cmd::lint_linux::LintLinuxArgs),
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    init_tracing(cli.verbose, cli.quiet);
    match run(cli) {
        Ok(status) => ExitCode::from(status.exit_code()),
        Err(err) => {
            eprintln!("llvm2-test: error: {err:#}");
            ExitCode::from(ResultStatus::Errored.exit_code())
        }
    }
}

fn init_tracing(verbose: u8, quiet: bool) {
    let filter = match (quiet, verbose) {
        (true, _) => EnvFilter::new("error"),
        (_, 0) => EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        (_, 1) => EnvFilter::new("debug"),
        _ => EnvFilter::new("trace"),
    };
    // Subscriber init can only fail if one is already installed, which
    // cannot happen here because `main` runs exactly once.
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(filter)
        .try_init();
}

fn run(cli: Cli) -> anyhow::Result<ResultStatus> {
    let global = cmd::GlobalArgs {
        format: cli.format,
        out: cli.out,
        config: cli.config,
        timeout: cli.timeout,
        parallel: cli.parallel,
        quiet: cli.quiet,
        verbose: cli.verbose,
        no_cache: cli.no_cache,
        dry_run: cli.dry_run,
    };
    match cli.command {
        Command::Matrix(args) => cmd::matrix::run(&global, &args),
        Command::Suite(args) => cmd::suite::run(&global, &args),
        Command::Fuzz(args) => cmd::fuzz::run(&global, &args),
        Command::Rustc(args) => cmd::rustc::run(&global, &args),
        Command::Bootstrap(args) => cmd::bootstrap::run(&global, &args),
        Command::Ecosystem(args) => cmd::ecosystem::run(&global, &args),
        Command::Prove(args) => cmd::prove::run(&global, &args),
        Command::Pipeline(args) => cmd::pipeline::run(&global, &args),
        Command::Report(args) => cmd::report::run(&global, &args),
        Command::Ratchet(args) => cmd::ratchet::run(&global, &args),
        Command::Doctor(args) => cmd::doctor::run(&global, &args),
        Command::LintLinux(args) => cmd::lint_linux::run(&global, &args),
    }
}
