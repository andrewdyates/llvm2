// llvm2-verify/z4_cache.rs - Persistent result cache for the z4 bridge
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: Task 3 of epic #407 / issue #420.
//
// Caches `Z4Result` outcomes across process invocations so that default-on
// `cargo test --features z4-prove` reruns skip obligations whose SMT-LIB2
// text, solver version, and configuration are byte-identical to a prior
// Verified / Invalid / Timeout run. `Unknown`/transport errors are NOT
// cached — those are transient and a rerun may succeed.
//
// The cache key is the 128-bit `StableHasher` digest of:
//   (obligation_smt2 || "\0" || config_signature || "\0" || solver_version)
// Domain-separated with NUL bytes so prefix collisions cannot spoof a key.
//
// On-disk layout mirrors `llvm2_opt::cache::FileCache`:
//   {root}/{first_two_hex}/{full_hex}.json
// where the payload is a `Z4CacheEntry` serialised via `serde_json` (stable
// schema for forward compatibility; corrupt entries are treated as misses).

//! On-disk result cache for the z4 SMT bridge.
//!
//! The cache is keyed on a `StableHasher` digest over the obligation's
//! SMT-LIB2 text, the solver config (timeout), and the solver version so
//! that a cache hit implies the same query was posed to the same solver
//! with the same budget on a previous run.
//!
//! Cached outcomes: [`Z4Result::Verified`], [`Z4Result::CounterExample`]
//! (Invalid with counterexample), [`Z4Result::Timeout`]. The cache
//! deliberately does NOT store `Z4Result::Error` results — those are
//! transient (solver missing, parse failure, IO error) and replaying them
//! would be actively harmful.
//!
//! Location resolution order (first hit wins):
//! 1. `LLVM2_Z4_CACHE` environment variable (explicit override).
//! 2. `$CARGO_TARGET_DIR/.llvm2-z4-cache/`.
//! 3. `$CARGO_MANIFEST_DIR/../target/.llvm2-z4-cache/` (workspace target).
//! 4. `$HOME/.llvm2-z4-cache/`.
//! 5. `std::env::temp_dir()/.llvm2-z4-cache/` (last resort).
//!
//! Entries are JSON-serialised so a human can inspect the cache, and
//! corrupt entries are silently treated as misses rather than hard errors.

use crate::z4_bridge::Z4Result;
use llvm2_opt::cache::StableHasher;
use serde::{Deserialize, Serialize};
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// On-disk schema version. Bump whenever the entry layout changes; old
/// entries will then round-trip as a deserialise error and be treated as
/// cache misses (never a hard failure).
pub const Z4_CACHE_SCHEMA_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// CachedZ4Result
// ---------------------------------------------------------------------------

/// Serde-friendly mirror of [`Z4Result`] restricted to the deterministic
/// outcomes that are safe to cache. `Unknown` and transport `Error`
/// results are deliberately not representable here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CachedZ4Result {
    /// The property was proven (UNSAT of the negated equivalence).
    Verified,
    /// The property failed with a concrete counterexample.
    /// Values are stored as hex strings for schema stability across
    /// future 128-bit extensions.
    Invalid { cex: Vec<(String, u64)> },
    /// The solver timed out (or hit a resource limit) on this query.
    Timeout,
}

impl CachedZ4Result {
    /// Try to convert a [`Z4Result`] into a cacheable outcome. Returns
    /// `None` for `Error(_)` — those must never be cached.
    pub fn try_from_result(result: &Z4Result) -> Option<Self> {
        match result {
            Z4Result::Verified => Some(CachedZ4Result::Verified),
            Z4Result::CounterExample(cex) => {
                Some(CachedZ4Result::Invalid { cex: cex.clone() })
            }
            Z4Result::Timeout => Some(CachedZ4Result::Timeout),
            Z4Result::Error(_) => None,
        }
    }

    /// Lift a cached outcome back into a [`Z4Result`].
    pub fn into_result(self) -> Z4Result {
        match self {
            CachedZ4Result::Verified => Z4Result::Verified,
            CachedZ4Result::Invalid { cex } => Z4Result::CounterExample(cex),
            CachedZ4Result::Timeout => Z4Result::Timeout,
        }
    }
}

// ---------------------------------------------------------------------------
// Z4CacheEntry
// ---------------------------------------------------------------------------

/// Persisted cache entry. The 64-bit `hash` is the low half of the
/// 128-bit `StableHasher` digest used to route the entry to its on-disk
/// path; it is stored in the payload as a double-check against collision
/// or stray file writes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Z4CacheEntry {
    /// Schema version tag (see [`Z4_CACHE_SCHEMA_VERSION`]).
    pub schema: u32,
    /// Low 64 bits of the 128-bit cache key. Filename carries the full
    /// 128-bit digest; this field lets corruption checks bail quickly.
    pub hash: u64,
    /// Deterministic cached outcome.
    pub result: CachedZ4Result,
    /// Free-form solver version string (e.g. `"z4 0.3"` or `"z3 4.13.0"`).
    pub solver_version: String,
    /// Unix seconds at the time the entry was written. Advisory only —
    /// not used for invalidation in this MVP.
    pub timestamp: u64,
}

impl Z4CacheEntry {
    fn new(digest: u128, result: CachedZ4Result, solver_version: String) -> Self {
        Self {
            schema: Z4_CACHE_SCHEMA_VERSION,
            hash: digest as u64,
            result,
            solver_version,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }
}

// ---------------------------------------------------------------------------
// Z4ResultCache
// ---------------------------------------------------------------------------

/// File-per-key persistent cache for z4 results.
///
/// The cache is safe to share across threads; each `get`/`put` performs
/// its own filesystem access. There is no eviction in this MVP — the
/// directory grows until the user or an outer process prunes it.
#[derive(Debug, Clone)]
pub struct Z4ResultCache {
    root: PathBuf,
}

impl Z4ResultCache {
    /// Create a cache rooted at `root`, creating the directory if needed.
    pub fn new(root: impl Into<PathBuf>) -> io::Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// Resolve the default cache root using the documented search order.
    pub fn default_root() -> PathBuf {
        default_cache_root()
    }

    /// Open the default cache location, creating the directory on demand.
    pub fn open_default() -> io::Result<Self> {
        Self::new(default_cache_root())
    }

    /// Root directory of this cache.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Compute the 128-bit cache key for a query.
    ///
    /// `config_signature` should encode every `Z4Config` field that can
    /// influence the solver's answer (timeout, for now). `solver_version`
    /// is the string returned by e.g. `solver_info()` or the z4 crate
    /// version — the caller decides, but it MUST be the same across runs
    /// when the solver is the same, or entries will never be reused.
    pub fn cache_key(
        obligation_smt2: &str,
        config_signature: &str,
        solver_version: &str,
    ) -> u128 {
        let mut h = StableHasher::new();
        h.write_str(obligation_smt2);
        h.write_u8(0);
        h.write_str(config_signature);
        h.write_u8(0);
        h.write_str(solver_version);
        h.finish128()
    }

    /// Hex filename used on disk for a given digest.
    fn hex_for(digest: u128) -> String {
        format!("{:032x}", digest)
    }

    fn path_for(&self, digest: u128) -> PathBuf {
        let hex = Self::hex_for(digest);
        let shard = &hex[0..2];
        self.root.join(shard).join(format!("{}.json", hex))
    }

    /// Look up a cached outcome. Returns `None` on miss, IO error, or
    /// corrupt-entry (treated as miss so that retries just overwrite).
    pub fn get(&self, digest: u128) -> Option<Z4CacheEntry> {
        let path = self.path_for(digest);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => return None,
        };
        let entry: Z4CacheEntry = match serde_json::from_slice(&bytes) {
            Ok(e) => e,
            Err(_) => return None,
        };
        if entry.schema != Z4_CACHE_SCHEMA_VERSION {
            return None;
        }
        if entry.hash != digest as u64 {
            // Hash mismatch: collision on the sharded path or tampered
            // file. Treat as miss so the caller re-solves and overwrites.
            return None;
        }
        Some(entry)
    }

    /// Store an outcome under `digest`. Failures are logged (stderr) but
    /// never panic; the cache is best-effort.
    pub fn put(
        &self,
        digest: u128,
        result: CachedZ4Result,
        solver_version: impl Into<String>,
    ) {
        let entry = Z4CacheEntry::new(digest, result, solver_version.into());
        let path = self.path_for(digest);
        if let Err(e) = self.write_atomic(&path, &entry) {
            eprintln!(
                "llvm2-verify::z4_cache: warning: failed to write {}: {}",
                path.display(),
                e
            );
        }
    }

    fn write_atomic(&self, target: &Path, entry: &Z4CacheEntry) -> io::Result<()> {
        let shard_dir = target
            .parent()
            .ok_or_else(|| io::Error::other("cache target has no parent dir"))?;
        std::fs::create_dir_all(shard_dir)?;

        let bytes = serde_json::to_vec(entry)
            .map_err(|e| io::Error::other(format!("serialize entry: {}", e)))?;

        let tmp = shard_dir.join(format!(
            ".{}.{}.tmp",
            std::process::id(),
            target.file_name().and_then(|n| n.to_str()).unwrap_or("entry")
        ));
        std::fs::write(&tmp, &bytes)?;
        match std::fs::rename(&tmp, target) {
            Ok(()) => Ok(()),
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                Err(e)
            }
        }
    }

    /// Remove the entry for `digest`, if present. Returns `true` if a
    /// file was removed. Used by tests and manual invalidation tooling.
    pub fn remove(&self, digest: u128) -> bool {
        std::fs::remove_file(self.path_for(digest)).is_ok()
    }
}

// ---------------------------------------------------------------------------
// Default-root resolution
// ---------------------------------------------------------------------------

/// Resolve the default cache root. See module docs for the search order.
///
/// This function never fails — if everything else is unusable, it falls
/// back to `std::env::temp_dir()/.llvm2-z4-cache/`.
pub fn default_cache_root() -> PathBuf {
    if let Ok(raw) = std::env::var("LLVM2_Z4_CACHE") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }

    if let Ok(raw) = std::env::var("CARGO_TARGET_DIR") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed).join(".llvm2-z4-cache");
        }
    }

    if let Ok(raw) = std::env::var("CARGO_MANIFEST_DIR") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let workspace_target = PathBuf::from(trimmed)
                .join("..")
                .join("..")
                .join("target")
                .join(".llvm2-z4-cache");
            return workspace_target;
        }
    }

    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        if !home.as_os_str().is_empty() {
            return home.join(".llvm2-z4-cache");
        }
    }

    std::env::temp_dir().join(".llvm2-z4-cache")
}

/// Build the canonical config signature string for cache-key derivation.
///
/// Kept as a dedicated helper so the bridge and tests agree on the
/// encoding. Whenever a new semantically-significant field is added to
/// [`crate::z4_bridge::Z4Config`] it MUST be appended here (bumping
/// [`Z4_CACHE_SCHEMA_VERSION`] if the change would otherwise silently
/// reuse old entries).
pub fn config_signature(timeout_ms: u64, produce_models: bool) -> String {
    format!(
        "z4cfg/v{}/timeout_ms={}/produce_models={}",
        Z4_CACHE_SCHEMA_VERSION, timeout_ms, produce_models
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    /// Monotonic counter so parallel tests never share a cache directory.
    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn isolated_cache() -> (Z4ResultCache, PathBuf) {
        let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "llvm2_z4_cache_test_{}_{}_{}",
            std::process::id(),
            Instant::now().elapsed().as_nanos(),
            id
        ));
        let cache = Z4ResultCache::new(&dir).expect("create test cache");
        (cache, dir)
    }

    #[test]
    fn try_from_result_filters_errors() {
        assert_eq!(
            CachedZ4Result::try_from_result(&Z4Result::Verified),
            Some(CachedZ4Result::Verified)
        );
        assert_eq!(
            CachedZ4Result::try_from_result(&Z4Result::Timeout),
            Some(CachedZ4Result::Timeout)
        );
        let cex = vec![("x".to_string(), 0xdeadbeef)];
        assert_eq!(
            CachedZ4Result::try_from_result(&Z4Result::CounterExample(cex.clone())),
            Some(CachedZ4Result::Invalid { cex })
        );
        assert_eq!(
            CachedZ4Result::try_from_result(&Z4Result::Error("boom".into())),
            None,
            "transport errors must never be cached"
        );
    }

    #[test]
    fn into_result_roundtrip() {
        let cex = vec![("a".to_string(), 1), ("b".to_string(), 2)];
        assert_eq!(
            CachedZ4Result::Verified.into_result(),
            Z4Result::Verified
        );
        assert_eq!(
            CachedZ4Result::Timeout.into_result(),
            Z4Result::Timeout
        );
        assert_eq!(
            CachedZ4Result::Invalid { cex: cex.clone() }.into_result(),
            Z4Result::CounterExample(cex)
        );
    }

    #[test]
    fn cache_key_is_deterministic_and_sensitive() {
        let a1 = Z4ResultCache::cache_key("(assert false)", "cfg", "z4-0.3");
        let a2 = Z4ResultCache::cache_key("(assert false)", "cfg", "z4-0.3");
        assert_eq!(a1, a2, "same inputs must produce same key");

        let b = Z4ResultCache::cache_key("(assert true)", "cfg", "z4-0.3");
        assert_ne!(a1, b, "different smt2 must produce different key");

        let c = Z4ResultCache::cache_key("(assert false)", "cfg2", "z4-0.3");
        assert_ne!(a1, c, "different config must produce different key");

        let d = Z4ResultCache::cache_key("(assert false)", "cfg", "z4-0.4");
        assert_ne!(a1, d, "different solver version must produce different key");
    }

    #[test]
    fn cache_key_is_prefix_separated() {
        // Without domain separation, ("ab","c") and ("a","bc") would
        // hash to the same key. The NUL separators must prevent that.
        let k1 = Z4ResultCache::cache_key("ab", "c", "v");
        let k2 = Z4ResultCache::cache_key("a", "bc", "v");
        assert_ne!(k1, k2);
    }

    #[test]
    fn miss_then_hit_roundtrip() {
        let (cache, _dir) = isolated_cache();
        let digest = Z4ResultCache::cache_key("q1", "cfg", "z4-0.3");

        assert!(cache.get(digest).is_none(), "fresh cache should miss");

        cache.put(digest, CachedZ4Result::Verified, "z4-0.3");
        let hit = cache.get(digest).expect("second lookup should hit");
        assert_eq!(hit.result, CachedZ4Result::Verified);
        assert_eq!(hit.solver_version, "z4-0.3");
        assert_eq!(hit.schema, Z4_CACHE_SCHEMA_VERSION);
        assert_eq!(hit.hash, digest as u64);
    }

    #[test]
    fn counterexample_roundtrips() {
        let (cache, _dir) = isolated_cache();
        let digest = Z4ResultCache::cache_key("q_cex", "cfg", "v");
        let cex = vec![("x".to_string(), 0x1234), ("y".to_string(), 0)];
        cache.put(
            digest,
            CachedZ4Result::Invalid { cex: cex.clone() },
            "v",
        );
        let hit = cache.get(digest).expect("hit");
        match hit.result {
            CachedZ4Result::Invalid { cex: stored } => assert_eq!(stored, cex),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn timeout_roundtrips() {
        let (cache, _dir) = isolated_cache();
        let digest = Z4ResultCache::cache_key("q_to", "cfg", "v");
        cache.put(digest, CachedZ4Result::Timeout, "v");
        assert_eq!(cache.get(digest).unwrap().result, CachedZ4Result::Timeout);
    }

    #[test]
    fn corrupt_entry_is_treated_as_miss() {
        let (cache, _dir) = isolated_cache();
        let digest = Z4ResultCache::cache_key("q_corrupt", "cfg", "v");
        let path = cache.path_for(digest);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"this is not json {{{").unwrap();
        assert!(cache.get(digest).is_none(), "corrupt entries must miss");
    }

    #[test]
    fn schema_mismatch_is_treated_as_miss() {
        let (cache, _dir) = isolated_cache();
        let digest = Z4ResultCache::cache_key("q_schema", "cfg", "v");
        let path = cache.path_for(digest);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        // Write a syntactically-valid JSON entry with a bogus schema tag.
        let bad = serde_json::json!({
            "schema": 999u32,
            "hash": digest as u64,
            "result": {"Verified": null},
            "solver_version": "v",
            "timestamp": 0u64,
        });
        std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
        assert!(
            cache.get(digest).is_none(),
            "entries with the wrong schema must be ignored"
        );
    }

    #[test]
    fn hash_mismatch_is_treated_as_miss() {
        let (cache, _dir) = isolated_cache();
        let digest = Z4ResultCache::cache_key("q_hash", "cfg", "v");
        let path = cache.path_for(digest);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let bad = Z4CacheEntry {
            schema: Z4_CACHE_SCHEMA_VERSION,
            hash: (digest as u64).wrapping_add(1),
            result: CachedZ4Result::Verified,
            solver_version: "v".into(),
            timestamp: 0,
        };
        std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
        assert!(
            cache.get(digest).is_none(),
            "entries with a hash mismatch must be ignored"
        );
    }

    #[test]
    fn default_root_respects_env_override() {
        let want = std::env::temp_dir().join("llvm2_z4_cache_override_probe");
        // Save + restore any existing value so parallel tests don't leak.
        let prev = std::env::var("LLVM2_Z4_CACHE").ok();
        // SAFETY: single-threaded test scope; we restore the var before returning.
        unsafe { std::env::set_var("LLVM2_Z4_CACHE", &want); }
        assert_eq!(default_cache_root(), want);
        unsafe {
            match prev {
                Some(v) => std::env::set_var("LLVM2_Z4_CACHE", v),
                None => std::env::remove_var("LLVM2_Z4_CACHE"),
            }
        }
    }

    #[test]
    fn config_signature_changes_with_fields() {
        let s1 = config_signature(5000, true);
        let s2 = config_signature(5001, true);
        let s3 = config_signature(5000, false);
        assert_ne!(s1, s2);
        assert_ne!(s1, s3);
        assert_eq!(s1, config_signature(5000, true));
    }

    /// Synthetic "second run is cheaper than first" check. Uses the
    /// cache directly (no solver required) and asserts that replaying
    /// the cached result is at least 10x cheaper than a simulated
    /// 10ms solver call.
    ///
    /// Acceptance criterion 4: second run is <10% of first run.
    #[test]
    fn second_run_is_an_order_of_magnitude_cheaper_than_simulated_solver() {
        let (cache, _dir) = isolated_cache();
        let digest = Z4ResultCache::cache_key("synthetic", "cfg", "v");

        // Simulated first run: "solve" (sleep) then cache.
        let t0 = Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(10));
        cache.put(digest, CachedZ4Result::Verified, "v");
        let first_run = t0.elapsed();

        // Second run: cache hit, no solver.
        let t1 = Instant::now();
        let hit = cache.get(digest).expect("cache should hit on second run");
        let second_run = t1.elapsed();
        assert_eq!(hit.result, CachedZ4Result::Verified);

        assert!(
            second_run * 10 < first_run,
            "second run ({:?}) should be <10% of first run ({:?})",
            second_run,
            first_run
        );
    }
}
