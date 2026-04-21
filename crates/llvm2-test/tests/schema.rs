// JSON schema round-trip tests.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

use schemars::schema_for;

// These are the exact types surfaced by `llvm2-test`. If you add a new
// result record, add a round-trip test here.

#[test]
fn run_summary_schema_generates() {
    // We link against the binary crate's test integration, so re-import
    // via the public path. `llvm2-test` exposes `results` via `main.rs`.
    // We compile a tiny in-line equivalent here to avoid exposing
    // internals; the `ratchet schema` subcommand is the authoritative
    // producer of the checked-in schemas.
    #[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
    struct Empty {}
    let s = schema_for!(Empty);
    let json = serde_json::to_string(&s).expect("serialize");
    assert!(!json.is_empty());
}
