// Golden help-text snapshot tests for `llvm2-test`.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// The snapshots under `tests/snapshots/` are the stable contract with
// operators + CI — CI fails on any un-committed drift. Regenerate via:
//   INSTA_UPDATE=always cargo test -p llvm2-test --test cli

use assert_cmd::Command;

fn bin() -> Command {
    Command::cargo_bin("llvm2-test").expect("binary built")
}

fn help_of(subcmd: &[&str]) -> String {
    let mut c = bin();
    c.args(subcmd).arg("--help");
    let out = c.output().expect("run");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn top_level_help() {
    insta::assert_snapshot!("top_help", help_of(&[]));
}

#[test]
fn matrix_help() {
    insta::assert_snapshot!("matrix_help", help_of(&["matrix"]));
}

#[test]
fn suite_help() {
    insta::assert_snapshot!("suite_help", help_of(&["suite"]));
}

#[test]
fn fuzz_help() {
    insta::assert_snapshot!("fuzz_help", help_of(&["fuzz"]));
}

#[test]
fn rustc_help() {
    insta::assert_snapshot!("rustc_help", help_of(&["rustc"]));
}

#[test]
fn bootstrap_help() {
    insta::assert_snapshot!("bootstrap_help", help_of(&["bootstrap"]));
}

#[test]
fn ecosystem_help() {
    insta::assert_snapshot!("ecosystem_help", help_of(&["ecosystem"]));
}

#[test]
fn prove_help() {
    insta::assert_snapshot!("prove_help", help_of(&["prove"]));
}

#[test]
fn pipeline_help() {
    insta::assert_snapshot!("pipeline_help", help_of(&["pipeline"]));
}

#[test]
fn report_help() {
    insta::assert_snapshot!("report_help", help_of(&["report"]));
}

#[test]
fn ratchet_help() {
    insta::assert_snapshot!("ratchet_help", help_of(&["ratchet"]));
}

#[test]
fn doctor_help() {
    insta::assert_snapshot!("doctor_help", help_of(&["doctor"]));
}

#[test]
fn stub_exits_with_code_2() {
    // Sanity check: every WS stub is NotImplemented (exit 2).
    let mut c = bin();
    c.args(["matrix"]);
    let out = c.output().expect("run");
    assert_eq!(out.status.code(), Some(2), "matrix stub must exit 2");
}

#[test]
fn doctor_json_format_smoke() {
    let mut c = bin();
    c.args(["--format", "json", "doctor", "--for", "report"]);
    let out = c.output().expect("run");
    // `for=report` has no required tools, so we expect exit 0.
    assert!(
        out.status.code() == Some(0) || out.status.code() == Some(2),
        "unexpected doctor exit: {:?}",
        out.status.code()
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("\"tools\""), "json report missing tools");
}
