// Shared JSON result schema for `llvm2-test`.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// See `designs/2026-04-19-proving-llvm2-replaces-llvm.md` section
// "Structured results schema". Every result-emitting subcommand serializes
// a `RunSummary<T>` where `T` implements `ResultRecord`.
//
// JSON schemas live under `evals/schema/` and are round-tripped by
// `tests/schema.rs`. `llvm2-test ratchet schema` verifies the checked-in
// files match what `schemars` would emit today.

//! Result types for every `llvm2-test` subcommand.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// Terminal status for a run. Mapped to process exit code in `main`.
///
/// The numeric exit codes follow `sysexits.h` where applicable:
///   * 0   — all units passed
///   * 1   — at least one unit failed (user-level failure)
///   * 2   — environment broken / tool missing
///   * 64  — argv / usage error
///   * 70  — internal bug (panic or unexpected error)
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ResultStatus {
    /// Every unit passed.
    Ok,
    /// At least one unit failed; the run completed cleanly.
    Failed,
    /// A required external tool was missing or the environment is broken.
    EnvBroken,
    /// User provided invalid arguments.
    UsageError,
    /// Internal bug — panic caught at `main`, or unexpected error.
    Errored,
    /// Subcommand is not yet implemented. Tracked per the design doc.
    NotImplemented,
}

impl ResultStatus {
    /// Convert to the numeric exit code surfaced by `main`.
    #[must_use]
    pub fn exit_code(self) -> u8 {
        match self {
            Self::Ok => 0,
            Self::Failed => 1,
            Self::EnvBroken | Self::NotImplemented => 2,
            Self::UsageError => 64,
            Self::Errored => 70,
        }
    }
}

/// Aggregate counts for a run.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct Totals {
    /// Number of units that passed.
    pub passed: u64,
    /// Number of units that failed the oracle check.
    pub failed: u64,
    /// Number of units intentionally skipped. Should be 0 in CI (rule #341).
    pub skipped: u64,
    /// Number of units that errored before reaching the oracle.
    pub errored: u64,
}

impl Totals {
    /// Total unit count.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.passed + self.failed + self.skipped + self.errored
    }
}

/// Host metadata captured at run start — useful for reproducibility.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct HostInfo {
    /// Target os string (e.g. `macos`).
    pub os: String,
    /// Target arch (e.g. `aarch64`, `x86_64`).
    pub arch: String,
    /// Optional CPU brand string if detectable.
    pub cpu: Option<String>,
    /// Logical CPU count.
    pub cores: u32,
    /// Hostname reported by the OS.
    pub hostname: Option<String>,
}

impl HostInfo {
    /// Collect host info for the running process.
    #[must_use]
    pub fn detect() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu: None,
            cores: u32::try_from(std::thread::available_parallelism().map_or(1, usize::from))
                .unwrap_or(1),
            hostname: std::env::var("HOSTNAME").ok(),
        }
    }
}

/// Tool version info discovered at run start.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct Environment {
    /// `tool_name -> version_string` as reported by `--version`.
    pub tools: BTreeMap<String, String>,
}

/// The canonical top-level result document written to disk by every
/// result-emitting subcommand. `T` is the per-unit record type and must
/// implement `Serialize + DeserializeOwned + JsonSchema`.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[schemars(rename_all = "snake_case")]
#[serde(bound(
    serialize = "T: Serialize",
    deserialize = "T: DeserializeOwned"
))]
pub struct RunSummary<T: Serialize + DeserializeOwned + JsonSchema> {
    /// Command name, e.g. `"matrix"`, `"fuzz.csmith"`.
    pub command: String,
    /// Repo version — `git rev-parse HEAD` at run time.
    pub version: String,
    /// Start time, UTC.
    pub started_at: DateTime<Utc>,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Host metadata.
    pub host: HostInfo,
    /// `blake3` digest of the resolved config + argv — used for cache
    /// invalidation and repro validation.
    pub config_digest: String,
    /// Aggregate counts.
    pub totals: Totals,
    /// Per-unit records.
    pub units: Vec<T>,
    /// Observed tool versions.
    pub environment: Environment,
    /// Terminal status.
    pub exit: ResultStatus,
}

// The `#[serde(bound)]` attribute above suppresses serde's auto-derived
// `T: Deserialize<'de>` bound on the `Vec<T>` field so we can use the
// simpler `DeserializeOwned` trait bound on the struct itself.

/// Empty record — used by `doctor` and other subcommands that don't
/// enumerate per-unit results but still emit a summary.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct EmptyRecord {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_status_exit_codes() {
        assert_eq!(ResultStatus::Ok.exit_code(), 0);
        assert_eq!(ResultStatus::Failed.exit_code(), 1);
        assert_eq!(ResultStatus::EnvBroken.exit_code(), 2);
        assert_eq!(ResultStatus::NotImplemented.exit_code(), 2);
        assert_eq!(ResultStatus::UsageError.exit_code(), 64);
        assert_eq!(ResultStatus::Errored.exit_code(), 70);
    }

    #[test]
    fn run_summary_roundtrips() {
        let summary = RunSummary::<EmptyRecord> {
            command: "doctor".to_string(),
            version: "abc123".to_string(),
            started_at: Utc::now(),
            duration_ms: 42,
            host: HostInfo::detect(),
            config_digest: "deadbeef".to_string(),
            totals: Totals::default(),
            units: vec![],
            environment: Environment::default(),
            exit: ResultStatus::Ok,
        };
        let json = serde_json::to_string(&summary).expect("serialize");
        let back: RunSummary<EmptyRecord> = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.command, "doctor");
        assert_eq!(back.exit, ResultStatus::Ok);
    }
}
