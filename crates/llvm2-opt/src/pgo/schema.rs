// llvm2-opt/pgo/schema.rs - .profdata on-disk schema (v0)
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: designs/2026-04-18-pgo-workflow.md
//
// Phase 1 MVP schema. Serialized as JSON via serde for v0; we will upgrade
// to a binary format (see the design doc byte layout) in a later phase.
// The on-disk `version` field is bumped on any incompatible change.
//
// Cross-cutting with #395: the 128-bit `module_hash` is produced by
// [`crate::cache::StableHasher`]. The reader compares the stored
// `module_hash` against the module currently being compiled to detect stale
// profiles. The `cache_key_version` field pins the hash-algorithm version
// (i.e., [`crate::cache::CACHE_KEY_VERSION`]).
//
// Reader must reject:
//   - unknown magic
//   - version > PROFDATA_VERSION (unknown schema)
//   - cache_key_version mismatch (different hasher layout)

//! On-disk schema for `.profdata` files.
//!
//! A `.profdata` file describes per-function basic-block execution counts
//! captured during a `--profile-generate` run. It is keyed by the stable
//! 128-bit module hash so that stale profiles can be detected.
//!
//! The v0 wire format is JSON (via `serde_json`). See
//! [`ProfData::write_to`](crate::pgo::ProfData::write_to) and
//! [`ProfData::read_from`](crate::pgo::ProfData::read_from).

use serde::{Deserialize, Serialize};

use crate::cache::CACHE_KEY_VERSION;

/// File magic string embedded in every `.profdata` file.
///
/// ASCII "LLVM2PGO" in the spelling used by [`ProfData::magic`].
/// Kept in the JSON so a plain text `head` on a corrupt file is informative.
pub const PROFDATA_MAGIC: &str = "LLVM2PGO";

/// Schema version. Bump on any incompatible layout change.
///
/// The writer always stamps this value. The reader rejects files whose
/// `version` is greater than its own `PROFDATA_VERSION`.
pub const PROFDATA_VERSION: u32 = 0;

/// Per-function profile record.
///
/// Blocks are identified by their [`llvm2_ir::BlockId`] value (`u32`).
/// Counts are raw `u64` hit totals and may be zero for blocks that were
/// not executed by the canary run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionProfile {
    /// Mangled function symbol name, as it appears in
    /// [`llvm2_ir::MachFunction::name`].
    pub name: String,
    /// Total call count (entry-to-the-function). Mirrors the existing
    /// `ProfileHookMode::CallCounts` trampoline data.
    #[serde(default)]
    pub call_count: u64,
    /// Per-block hit counts. `blocks[i].block_id` is the `u32` from
    /// [`llvm2_ir::BlockId`]. Order is not semantically significant; the
    /// reader indexes by `block_id`.
    #[serde(default)]
    pub blocks: Vec<BlockProfile>,
    /// Optional per-edge counts (`(from_block, to_block) -> hits`).
    ///
    /// v0 readers/writers may leave this empty. The field is reserved for
    /// Phase 3 edge-count instrumentation.
    #[serde(default)]
    pub edges: Vec<EdgeProfile>,
}

impl FunctionProfile {
    /// Create a new function profile with no counters.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            call_count: 0,
            blocks: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Look up the hit count for a block id, returning 0 if the block was
    /// not present in the profile (i.e., not executed or not instrumented).
    pub fn block_hits(&self, block_id: u32) -> u64 {
        self.blocks
            .iter()
            .find(|b| b.block_id == block_id)
            .map(|b| b.hits)
            .unwrap_or(0)
    }
}

/// Per-block counter payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockProfile {
    /// Raw `u32` value of the `BlockId`.
    pub block_id: u32,
    /// Total hits during the canary run. `0` means the block was not
    /// executed by the canary (treat as cold, not as infinitely-rare).
    pub hits: u64,
}

impl BlockProfile {
    /// Convenience constructor.
    pub fn new(block_id: u32, hits: u64) -> Self {
        Self { block_id, hits }
    }
}

/// Per-edge counter payload (reserved; v0 writers emit none).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeProfile {
    /// Source block id.
    pub from: u32,
    /// Destination block id.
    pub to: u32,
    /// Total traversals during the canary run.
    pub hits: u64,
}

impl EdgeProfile {
    /// Convenience constructor.
    pub fn new(from: u32, to: u32, hits: u64) -> Self {
        Self { from, to, hits }
    }
}

/// Top-level profile document.
///
/// Serialized in v0 as JSON. The `module_hash` is the 128-bit
/// [`crate::cache::StableHasher`] digest of the source tMIR module bytes
/// that produced this profile; the reader compares against the hash of
/// the module currently being compiled.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfData {
    /// Magic string, always [`PROFDATA_MAGIC`].
    #[serde(default = "default_magic")]
    pub magic: String,
    /// Schema version, always [`PROFDATA_VERSION`] on write.
    #[serde(default = "default_version")]
    pub version: u32,
    /// Hash-algorithm version ([`CACHE_KEY_VERSION`]). If this differs
    /// from the reader's version, the profile must be rejected because
    /// the `module_hash` is not comparable.
    #[serde(default = "default_cache_key_version")]
    pub cache_key_version: u32,
    /// 128-bit stable hash of the source module, serialized as a
    /// 32-character lowercase hex string so the file is greppable.
    pub module_hash: String,
    /// Optional target triple for diagnostic use only. The reader does
    /// not compare it; the compiler is responsible for lining up the
    /// compilation target.
    #[serde(default)]
    pub target_triple: String,
    /// Optional opt-level label for diagnostic use only.
    #[serde(default)]
    pub opt_level: String,
    /// Per-function records.
    #[serde(default)]
    pub functions: Vec<FunctionProfile>,
}

fn default_magic() -> String {
    PROFDATA_MAGIC.to_string()
}
fn default_version() -> u32 {
    PROFDATA_VERSION
}
fn default_cache_key_version() -> u32 {
    CACHE_KEY_VERSION
}

impl ProfData {
    /// Create an empty `ProfData` stamped with the canonical magic /
    /// version / cache-key-version. The caller fills in `module_hash`
    /// and the per-function records.
    pub fn new(module_hash: u128) -> Self {
        Self {
            magic: PROFDATA_MAGIC.to_string(),
            version: PROFDATA_VERSION,
            cache_key_version: CACHE_KEY_VERSION,
            module_hash: format!("{:032x}", module_hash),
            target_triple: String::new(),
            opt_level: String::new(),
            functions: Vec::new(),
        }
    }

    /// Return the module hash as the raw `u128`.
    ///
    /// Returns `None` if the stored hex string is malformed.
    pub fn module_hash_u128(&self) -> Option<u128> {
        if self.module_hash.len() != 32 {
            return None;
        }
        u128::from_str_radix(&self.module_hash, 16).ok()
    }

    /// Look up a function profile by name.
    pub fn function(&self, name: &str) -> Option<&FunctionProfile> {
        self.functions.iter().find(|f| f.name == name)
    }

    /// Mutable lookup by name; inserts a default record if missing.
    pub fn function_mut_or_insert(&mut self, name: &str) -> &mut FunctionProfile {
        let pos = self.functions.iter().position(|f| f.name == name);
        match pos {
            Some(i) => &mut self.functions[i],
            None => {
                self.functions.push(FunctionProfile::new(name));
                self.functions.last_mut().unwrap()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profdata_constructs_with_stamped_headers() {
        let p = ProfData::new(0x1234_5678_9abc_def0_1122_3344_5566_7788);
        assert_eq!(p.magic, PROFDATA_MAGIC);
        assert_eq!(p.version, PROFDATA_VERSION);
        assert_eq!(p.cache_key_version, CACHE_KEY_VERSION);
        assert_eq!(p.module_hash.len(), 32);
        assert_eq!(
            p.module_hash_u128(),
            Some(0x1234_5678_9abc_def0_1122_3344_5566_7788_u128)
        );
    }

    #[test]
    fn function_profile_block_hits_defaults_to_zero() {
        let mut f = FunctionProfile::new("foo");
        f.blocks.push(BlockProfile::new(0, 10));
        f.blocks.push(BlockProfile::new(1, 20));
        assert_eq!(f.block_hits(0), 10);
        assert_eq!(f.block_hits(1), 20);
        assert_eq!(f.block_hits(99), 0);
    }

    #[test]
    fn function_mut_or_insert_idempotent() {
        let mut p = ProfData::new(0);
        p.function_mut_or_insert("foo").call_count = 3;
        p.function_mut_or_insert("foo").call_count += 5;
        assert_eq!(p.functions.len(), 1);
        assert_eq!(p.function("foo").unwrap().call_count, 8);
    }

    #[test]
    fn module_hash_u128_rejects_malformed() {
        let mut p = ProfData::new(0);
        p.module_hash = "not-hex".to_string();
        assert!(p.module_hash_u128().is_none());
    }
}
