// WS9 — weekly report generator.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// The template literal in this file IS the source of truth for the weekly
// report format. Never hand-edit `reports/weekly/TEMPLATE.md` — it is
// regenerated from here.

//! `llvm2-test report` — render the weekly LLVM2 dashboard.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;
use chrono::{NaiveDate, Utc};
use clap::Args;
use serde::Deserialize;

use super::GlobalArgs;
use crate::OutputFormat;
use crate::config::RepoRoot;
use crate::results::ResultStatus;

/// Arguments for `llvm2-test report`.
#[derive(Args, Debug, Clone)]
#[command(
    long_about = "Generate the weekly LLVM2 dashboard (WS9).\n\n\
                  Reads every `evals/results/**/<latest>.json`, renders the \
                  north-star table + per-workstream sections. Missing data \
                  is rendered as `-` with a footnote. Writes to \
                  `reports/weekly/<iso-date>.md` by default.\n\n\
                  # Examples\n\n  \
                  llvm2-test report\n  \
                  llvm2-test report --week 2026-04-19 --format markdown\n  \
                  llvm2-test report --out /tmp/weekly.md"
)]
pub struct ReportArgs {
    /// ISO date for the report. Default: today (UTC).
    #[arg(long, value_name = "ISO-DATE")]
    pub week: Option<String>,

    /// Markdown target path (takes precedence over `--out`).
    #[arg(long, value_name = "PATH")]
    pub markdown_out: Option<PathBuf>,

    /// Also write the dashboard pointer (`reports/dashboard.md`).
    #[arg(long)]
    pub publish: bool,
}

#[derive(Debug, Deserialize)]
struct LatestResult {
    command: String,
    #[serde(default)]
    totals: Option<serde_json::Value>,
    #[serde(default)]
    exit: Option<serde_json::Value>,
}

fn discover_latest(root: &Path) -> BTreeMap<String, Option<LatestResult>> {
    let wanted = [
        "matrix",
        "suite",
        "fuzz",
        "rustc",
        "bootstrap",
        "ecosystem",
        "prove",
        "pipeline",
    ];
    let mut out: BTreeMap<String, Option<LatestResult>> = BTreeMap::new();
    for w in wanted {
        out.insert(w.to_string(), None);
    }
    let base = root.join("evals").join("results");
    if !base.is_dir() {
        return out;
    }
    let Ok(entries) = fs::read_dir(&base) else {
        return out;
    };
    for e in entries.flatten() {
        let path = e.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !out.contains_key(name) {
            continue;
        }
        if let Some(latest) = find_latest_json(&path) {
            if let Ok(text) = fs::read_to_string(&latest) {
                if let Ok(parsed) = serde_json::from_str::<LatestResult>(&text) {
                    out.insert(name.to_string(), Some(parsed));
                }
            }
        }
    }
    out
}

fn find_latest_json(dir: &Path) -> Option<PathBuf> {
    let mut best: Option<(String, PathBuf)> = None;
    let entries = fs::read_dir(dir).ok()?;
    for e in entries.flatten() {
        let path = e.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        match &best {
            None => best = Some((stem, path.clone())),
            Some((b, _)) if stem > *b => best = Some((stem, path.clone())),
            _ => {}
        }
    }
    best.map(|(_, p)| p)
}

fn metric_cell(record: &Option<LatestResult>) -> String {
    let Some(r) = record else {
        return "—".to_string();
    };
    match &r.totals {
        Some(t) => serde_json::to_string(t).unwrap_or_else(|_| "—".to_string()),
        None => {
            let exit = r
                .exit
                .as_ref()
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            format!("exit={exit}")
        }
    }
}

fn render_markdown(
    week_iso: &str,
    records: &BTreeMap<String, Option<LatestResult>>,
    sha: &str,
) -> String {
    let row = |ws: &str, layer: &str, metric: &str, rec: &Option<LatestResult>| -> String {
        format!(
            "| {ws} | {layer} | {metric} | {cell} |",
            cell = metric_cell(rec)
        )
    };
    let mut s = String::new();
    s.push_str(&format!("# LLVM2 weekly report — {week_iso}\n\n"));
    s.push_str(&format!("Git commit: `{sha}`\n\n"));
    s.push_str("> Rendered by `llvm2-test report`. `—` means not measured yet.\n\n");
    s.push_str("## North-star metrics\n\n");
    s.push_str("| WS | Layer | Metric | Latest |\n");
    s.push_str("|---|---|---|---|\n");
    s.push_str(&row(
        "WS1",
        "L1 Unit",
        "`cargo test` pass-count",
        records.get("matrix").unwrap_or(&None),
    ));
    s.push('\n');
    s.push_str(&row(
        "WS2",
        "L2 E2E",
        "`llvm-test-suite SingleSource` pass-rate",
        records.get("suite").unwrap_or(&None),
    ));
    s.push('\n');
    s.push_str(&row(
        "WS3",
        "L3 Fuzz",
        "miscompiles found/fixed",
        records.get("fuzz").unwrap_or(&None),
    ));
    s.push('\n');
    s.push_str(&row(
        "WS4",
        "L4 rustc",
        "rustc UI pass-rate",
        records.get("rustc").unwrap_or(&None),
    ));
    s.push('\n');
    s.push_str(&row(
        "WS5",
        "L5 Bootstrap",
        "rustc stage reached",
        records.get("bootstrap").unwrap_or(&None),
    ));
    s.push('\n');
    s.push_str(&row(
        "WS6",
        "L6 Ecosystem",
        "top-100 crates.io pass-rate",
        records.get("ecosystem").unwrap_or(&None),
    ));
    s.push('\n');
    s.push_str(&row(
        "WS7",
        "L7 Proof",
        "z4 obligations discharged",
        records.get("prove").unwrap_or(&None),
    ));
    s.push('\n');
    s.push_str(&row(
        "WS8",
        "L8 Proof infra",
        "RA/sched/emit stages proven",
        records.get("pipeline").unwrap_or(&None),
    ));
    s.push('\n');
    s.push_str("\n<sup>Cells marked `—` have no result artifact under `evals/results/<ws>/`. ");
    s.push_str("Run the matching `llvm2-test <ws>` subcommand to populate.</sup>\n\n");
    s.push_str("## Per-workstream notes\n\n");
    for ws in [
        "matrix",
        "suite",
        "fuzz",
        "rustc",
        "bootstrap",
        "ecosystem",
        "prove",
        "pipeline",
    ] {
        s.push_str(&format!(
            "### {}\n\n",
            match ws {
                "matrix" => "WS1 — matrix",
                "suite" => "WS2 — suite",
                "fuzz" => "WS3 — fuzz",
                "rustc" => "WS4 — rustc",
                "bootstrap" => "WS5 — bootstrap",
                "ecosystem" => "WS6 — ecosystem",
                "prove" => "WS7 — prove",
                "pipeline" => "WS8 — pipeline",
                _ => ws,
            }
        ));
        match records.get(ws).unwrap_or(&None) {
            Some(r) => s.push_str(&format!(
                "Command: `{}`. Totals: {}.\n\n",
                r.command,
                metric_cell(&Some(LatestResult {
                    command: r.command.clone(),
                    totals: r.totals.clone(),
                    exit: r.exit.clone(),
                }))
            )),
            None => s.push_str(&format!(
                "No result yet. See `docs/testing/{ws}.md` for how to run.\n\n"
            )),
        }
    }
    s.push_str("## Source\n\n");
    s.push_str("This report is generated by `llvm2-test report`. ");
    s.push_str("Do not hand-edit — regenerate via `llvm2-test report`.\n");
    s
}

fn today_iso() -> String {
    Utc::now().format("%Y-%m-%d").to_string()
}

fn parse_week(input: &str) -> anyhow::Result<String> {
    NaiveDate::parse_from_str(input, "%Y-%m-%d")
        .with_context(|| format!("week must be YYYY-MM-DD, got {input:?}"))?;
    Ok(input.to_string())
}

fn repo_sha(root: &Path) -> String {
    // Works for plain repos and for linked worktrees where `.git` is a
    // file containing `gitdir: <path-to-worktree-gitdir>`.
    let git = root.join(".git");
    let git_dir = if git.is_dir() {
        git
    } else if let Ok(text) = fs::read_to_string(&git) {
        text.lines()
            .find_map(|l| l.strip_prefix("gitdir: ").map(str::trim).map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from(".git"))
    } else {
        return "unknown".to_string();
    };
    let head = git_dir.join("HEAD");
    let Ok(content) = fs::read_to_string(&head) else {
        return "unknown".to_string();
    };
    let content = content.trim();
    if let Some(reference) = content.strip_prefix("ref: ") {
        // For linked worktrees, refs are in the main repo's refs dir.
        // The common dir is captured by `commondir` next to `HEAD`.
        let commondir = git_dir.join("commondir");
        let base = if commondir.is_file() {
            let rel = fs::read_to_string(&commondir).unwrap_or_default();
            let rel = rel.trim();
            git_dir.join(rel)
        } else {
            git_dir.clone()
        };
        let ref_path = base.join(reference);
        return fs::read_to_string(ref_path)
            .ok()
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string());
    }
    content.to_string()
}

/// Entry point.
pub fn run(global: &GlobalArgs, args: &ReportArgs) -> anyhow::Result<ResultStatus> {
    let repo = RepoRoot::locate(Path::new("."))?;
    let week = match &args.week {
        Some(w) => parse_week(w)?,
        None => today_iso(),
    };
    let records = discover_latest(&repo.0);
    let sha = repo_sha(&repo.0);
    let md = render_markdown(&week, &records, &sha);

    let target = if let Some(p) = args.markdown_out.clone() {
        p
    } else if let Some(p) = global.out.clone() {
        p
    } else {
        let dir = repo.join("reports").join("weekly");
        fs::create_dir_all(&dir)?;
        dir.join(format!("{week}.md"))
    };

    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&target, &md)?;

    if args.publish {
        let dashboard = repo.join("reports").join("dashboard.md");
        let pointer = format!(
            "# LLVM2 dashboard\n\nLatest weekly report: [`reports/weekly/{week}.md`](weekly/{week}.md)\n"
        );
        fs::write(dashboard, pointer)?;
    }

    match global.format {
        OutputFormat::Json | OutputFormat::Junit => {
            let json = serde_json::json!({
                "command": "report",
                "week": week,
                "output": target,
                "bytes": md.len(),
            });
            println!("{json}");
        }
        OutputFormat::Human => {
            println!("llvm2-test report");
            println!("  week:   {week}");
            println!("  output: {}", target.display());
            println!("  size:   {} bytes", md.len());
        }
    }

    Ok(ResultStatus::Ok)
}
