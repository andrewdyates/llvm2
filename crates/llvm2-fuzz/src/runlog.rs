// llvm2-fuzz/src/runlog.rs - JSON run log schema
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Shared JSON result schema for all three drivers. The campaign script
// (scripts/fuzz_campaign.sh) and WS9 consume this format.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Repro {
    /// PRNG seed or external-tool seed that reproduces this run.
    pub seed: u64,
    /// Path to the minimized input (tMIR json for tmir-gen, .c file for
    /// csmith/yarpgen). None when we couldn't save it.
    pub minimized_input_path: Option<String>,
    /// One-line summary of the defect for humans browsing the log.
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunLog {
    /// Driver name: "tmir-gen" | "csmith-driver" | "yarpgen-driver".
    pub driver: String,
    /// "ok" | "unavailable".
    pub status: String,
    /// Human-readable reason when `status == "unavailable"` (e.g. which
    /// external tool is missing and how to install it).
    pub reason: Option<String>,
    /// Campaign duration in seconds (requested).
    pub duration_secs: u64,
    /// Number of iterations that actually ran.
    pub runs: u64,
    /// Runs that timed out (per-iteration timeout).
    pub timeouts: u64,
    /// Runs where the compiler or an external tool crashed (panic, non-zero
    /// exit from a phase that shouldn't fail).
    pub crashes: u64,
    /// Runs where the oracle and LLVM2 disagreed on the result — actual
    /// miscompiles.
    pub miscompiles: u64,
    /// Detailed repros (up to a cap, to keep JSON manageable).
    pub repros: Vec<Repro>,
    /// ISO8601 start timestamp (UTC).
    pub started_at: String,
    /// ISO8601 end timestamp (UTC).
    pub finished_at: String,
}
