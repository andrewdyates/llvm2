// `llvm2-test doctor` — environment check.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Environment self-check for `llvm2-test`.
//!
//! Runs through every external tool + reference corpus the suite may
//! depend on and prints a table. Scoped to one subcommand's
//! requirements via `--for <cmd>`.

use clap::{Args, ValueEnum};
use serde::Serialize;

use super::GlobalArgs;
use crate::OutputFormat;
use crate::corpus;
use crate::external::{self, VersionInfo, install_hint};
use crate::results::ResultStatus;

/// Which subcommand's dependency set to check.
#[derive(Clone, Copy, Debug, ValueEnum, Default)]
#[value(rename_all = "lower")]
pub enum TargetCmd {
    /// Everything across every subcommand.
    #[default]
    All,
    /// Just the tools `matrix` needs.
    Matrix,
    /// Just the tools `suite` needs.
    Suite,
    /// Just the tools `fuzz` needs.
    Fuzz,
    /// Just the tools `rustc` needs.
    Rustc,
    /// Just the tools `bootstrap` needs.
    Bootstrap,
    /// Just the tools `ecosystem` needs.
    Ecosystem,
    /// Just the tools `prove` needs.
    Prove,
    /// Just the tools `pipeline` needs.
    Pipeline,
    /// Just the tools `report` needs.
    Report,
}

/// Arguments for `llvm2-test doctor`.
#[derive(Args, Debug, Clone)]
#[command(
    long_about = "Check the local environment for tools and reference \
                  corpora used by `llvm2-test`. Prints a table for \
                  `--format human`, a structured JSON report for \
                  `--format json`. Exits 0 when every *required* tool \
                  for the given `--for` target is present, 2 otherwise.\n\n\
                  # Examples\n\n  \
                  llvm2-test doctor\n  \
                  llvm2-test doctor --for fuzz --format json\n  \
                  llvm2-test doctor --for matrix"
)]
pub struct DoctorArgs {
    /// Restrict checks to tools needed by one subcommand.
    #[arg(long = "for", value_enum, default_value_t = TargetCmd::All)]
    pub for_cmd: TargetCmd,
}

#[derive(Clone, Debug, Serialize)]
struct Row {
    tool: String,
    required: bool,
    present: bool,
    version: String,
    path: Option<String>,
    install_hint: &'static str,
}

#[derive(Clone, Debug, Serialize)]
struct CorpusRow {
    name: String,
    required: bool,
    present: bool,
    default_path: String,
}

#[derive(Debug, Serialize)]
struct DoctorReport {
    target: String,
    tools: Vec<Row>,
    corpora: Vec<CorpusRow>,
    exit: ResultStatus,
}

fn required_tools(target: TargetCmd) -> Vec<&'static str> {
    match target {
        TargetCmd::All => vec!["cargo", "rustc", "cc", "clang", "gh"],
        TargetCmd::Matrix => vec!["cargo"],
        TargetCmd::Suite => vec!["clang", "cc"],
        TargetCmd::Fuzz => vec!["clang", "csmith", "creduce"],
        TargetCmd::Rustc | TargetCmd::Bootstrap | TargetCmd::Ecosystem => vec!["cargo", "rustc"],
        TargetCmd::Prove => vec!["z4"],
        TargetCmd::Pipeline => vec!["z4"],
        TargetCmd::Report => vec![],
    }
}

fn all_tools() -> Vec<&'static str> {
    vec![
        "cargo", "rustc", "cc", "clang", "csmith", "yarpgen", "creduce", "z4", "gh",
    ]
}

fn required_corpora(target: TargetCmd) -> Vec<&'static str> {
    match target {
        TargetCmd::Suite => vec!["llvm-test-suite"],
        TargetCmd::Bootstrap => vec!["rustc"],
        _ => vec![],
    }
}

fn probe_row(name: &str, required: bool) -> Row {
    let v: VersionInfo = external::probe_one(name);
    Row {
        tool: v.name,
        required,
        present: v.present,
        version: v.version,
        path: v.path,
        install_hint: install_hint(name),
    }
}

/// Entry point.
pub fn run(global: &GlobalArgs, args: &DoctorArgs) -> anyhow::Result<ResultStatus> {
    let required = required_tools(args.for_cmd);
    let rows: Vec<Row> = all_tools()
        .iter()
        .map(|n| probe_row(n, required.iter().any(|r| r == n)))
        .collect();

    let req_corpora = required_corpora(args.for_cmd);
    let corpora: Vec<CorpusRow> = corpus::all()
        .into_iter()
        .map(|c| CorpusRow {
            name: c.name.to_string(),
            required: req_corpora.iter().any(|r| *r == c.name),
            present: c.present(),
            default_path: c.default_path.display().to_string(),
        })
        .collect();

    let tool_ok = rows.iter().all(|r| !r.required || r.present);
    let corpus_ok = corpora.iter().all(|r| !r.required || r.present);
    let exit = if tool_ok && corpus_ok {
        ResultStatus::Ok
    } else {
        ResultStatus::EnvBroken
    };

    let report = DoctorReport {
        target: format!("{:?}", args.for_cmd).to_lowercase(),
        tools: rows,
        corpora,
        exit,
    };

    match global.format {
        OutputFormat::Json | OutputFormat::Junit => {
            let json = serde_json::to_string_pretty(&report)?;
            println!("{json}");
        }
        OutputFormat::Human => print_human(&report),
    }

    Ok(exit)
}

fn print_human(r: &DoctorReport) {
    println!("llvm2-test doctor  (target: {})", r.target);
    println!();
    println!(
        "  {:<9} {:<9} {:<10} {:<32} install hint",
        "tool", "required", "present", "version"
    );
    println!("  {}", "-".repeat(86));
    for row in &r.tools {
        let ver = if row.version.is_empty() {
            "-".to_string()
        } else {
            let mut v = row.version.clone();
            if v.len() > 30 {
                v.truncate(30);
                v.push('~');
            }
            v
        };
        println!(
            "  {:<9} {:<9} {:<10} {:<32} {}",
            row.tool,
            if row.required { "yes" } else { "no" },
            if row.present { "yes" } else { "no" },
            ver,
            if row.present { "" } else { row.install_hint },
        );
    }
    println!();
    println!("  reference corpora:");
    for c in &r.corpora {
        println!(
            "    {:<32} required={:<3} present={:<3}   {}",
            c.name,
            if c.required { "yes" } else { "no" },
            if c.present { "yes" } else { "no" },
            c.default_path,
        );
    }
    println!();
    match r.exit {
        ResultStatus::Ok => println!("  status: OK"),
        _ => println!("  status: ENV_BROKEN (exit {})", r.exit.exit_code()),
    }
}
