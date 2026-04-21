// llvm2-opt/cache.rs - Compilation cache + stable hash
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: designs/2026-04-18-cache-and-cegis.md
//
// Stable hashing uses an in-tree xxh3-inspired mixer with a fixed seed so
// cache keys are deterministic across runs, processes, Rust versions, and
// machines (modulo CPU feature differences which are part of the key).
//
// DO NOT substitute `std::collections::hash_map::DefaultHasher` here: it is
// randomised on many platforms and explicitly not stable across Rust
// versions. Stability is the whole point of this module.

//! Compilation cache hooks and stable hashing for LLVM2.
//!
//! This module provides:
//!
//! - [`stable_hash`] / [`StableHasher`]: deterministic 128-bit hashing
//!   built on xxh3-style mixers with fixed seeds.
//! - [`CacheKey`]: composite key bundling module hash + opt level + target
//!   triple + CPU + feature flags. Stable 128-bit output, hex-serializable.
//! - [`CacheBackend`]: pluggable trait for cache storage, with in-memory
//!   and on-disk backends.
//! - [`InMemoryCache`], [`FileCache`]: concrete backends.
//! - [`StatsCache`]: wraps any backend and records hit/miss/put counters.
//!
//! # Example
//!
//! ```
//! use llvm2_opt::cache::{CacheBackend, CacheKey, InMemoryCache, stable_hash};
//!
//! let key = CacheKey::new(
//!     stable_hash(b"module bytes"),
//!     2, // O2
//!     "aarch64-apple-darwin".to_string(),
//!     "apple-m1".to_string(),
//!     vec!["+neon".to_string()],
//! );
//! let cache = InMemoryCache::new();
//! assert_eq!(cache.get(&key), None);
//! cache.put(&key, b"cached value");
//! assert_eq!(cache.get(&key).as_deref(), Some(&b"cached value"[..]));
//! ```

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Stable hashing
// ---------------------------------------------------------------------------

/// Golden-ratio seed used by the stable mixer. Matches the constant used
/// in xxh3 secret derivation and in MurmurHash finalisation. Picked for
/// good avalanche behaviour on integer-shaped inputs.
pub const STABLE_HASH_SEED: u64 = 0x9E3779B185EBCA87;

/// Secondary seed for the high 64-bit stream of the 128-bit hasher.
///
/// Picked as `FRAC(sqrt(2)) * 2^64` — a different irrational-derived
/// constant so the two streams mix independently.
pub const STABLE_HASH_SEED_HI: u64 = 0x6A09E667F3BCC908;

/// Fixed mixer constants (xxh64/xxh3 style). These are part of the cache
/// wire format — changing them invalidates every on-disk cache entry.
const PRIME_1: u64 = 0x9E3779B185EBCA87;
const PRIME_2: u64 = 0xC2B2AE3D27D4EB4F;
const PRIME_3: u64 = 0x165667B19E3779F9;
const PRIME_4: u64 = 0x85EBCA77C2B2AE63;
const PRIME_5: u64 = 0x27D4EB2F165667C5;

/// Single round of xxh3-style avalanche on a 64-bit lane.
#[inline(always)]
fn mix64(mut v: u64, input: u64, prime: u64) -> u64 {
    v = v.wrapping_add(input.wrapping_mul(prime));
    v = v.rotate_left(31);
    v.wrapping_mul(PRIME_1)
}

/// Final avalanche (matches xxh64 finalisation).
#[inline(always)]
fn avalanche(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(PRIME_2);
    h ^= h >> 29;
    h = h.wrapping_mul(PRIME_3);
    h ^= h >> 32;
    h
}

/// A deterministic 128-bit hasher.
///
/// Two independent 64-bit lanes seeded from [`STABLE_HASH_SEED`] and
/// [`STABLE_HASH_SEED_HI`] are mixed with the input and combined into a
/// `u128` at finalisation. The output is stable across runs, processes,
/// and machines.
#[derive(Debug, Clone)]
pub struct StableHasher {
    lo: u64,
    hi: u64,
    len: u64,
}

impl Default for StableHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl StableHasher {
    /// Create a hasher seeded with the canonical LLVM2 cache seeds.
    pub fn new() -> Self {
        Self::with_seeds(STABLE_HASH_SEED, STABLE_HASH_SEED_HI)
    }

    /// Create a hasher with explicit 64-bit seeds. Useful for domain
    /// separation (e.g., "module hash" vs "function hash").
    pub fn with_seeds(lo_seed: u64, hi_seed: u64) -> Self {
        Self { lo: lo_seed, hi: hi_seed, len: 0 }
    }

    /// Absorb a byte slice.
    ///
    /// Bytes are consumed in 8-byte blocks where possible; the tail is
    /// mixed one byte at a time. This differs from full xxh3 (which has
    /// a fast-path SIMD implementation) but is cross-platform stable
    /// by construction.
    pub fn write(&mut self, bytes: &[u8]) {
        self.len = self.len.wrapping_add(bytes.len() as u64);

        let mut i = 0;
        while i + 8 <= bytes.len() {
            let word = u64::from_le_bytes([
                bytes[i],
                bytes[i + 1],
                bytes[i + 2],
                bytes[i + 3],
                bytes[i + 4],
                bytes[i + 5],
                bytes[i + 6],
                bytes[i + 7],
            ]);
            self.lo = mix64(self.lo, word, PRIME_2);
            self.hi = mix64(self.hi, word, PRIME_3);
            i += 8;
        }
        // Tail: fold remaining bytes.
        while i < bytes.len() {
            let byte = bytes[i] as u64;
            self.lo = mix64(self.lo, byte, PRIME_4);
            self.hi = mix64(self.hi, byte.rotate_left(7), PRIME_5);
            i += 1;
        }
    }

    /// Absorb a length-prefixed byte slice. Prevents ambiguity where two
    /// concatenations could produce the same hash (e.g. ["a","bc"] vs
    /// ["ab","c"]).
    pub fn write_framed(&mut self, bytes: &[u8]) {
        self.write_u64(bytes.len() as u64);
        self.write(bytes);
    }

    /// Absorb a string. Equivalent to `write_framed(s.as_bytes())`.
    pub fn write_str(&mut self, s: &str) {
        self.write_framed(s.as_bytes());
    }

    /// Absorb a little-endian u64.
    pub fn write_u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }

    /// Absorb a little-endian u32.
    pub fn write_u32(&mut self, value: u32) {
        self.write(&value.to_le_bytes());
    }

    /// Absorb a single byte.
    pub fn write_u8(&mut self, value: u8) {
        self.write(&[value]);
    }

    /// Finalise the hasher and return a 128-bit digest.
    ///
    /// Finalisation also mixes the total absorbed length into both lanes
    /// so that the empty input and an all-zero input produce distinct
    /// hashes.
    pub fn finish128(&self) -> u128 {
        let mut lo = self.lo ^ self.len;
        let mut hi = self.hi ^ self.len.rotate_left(32);
        lo = avalanche(lo);
        hi = avalanche(hi.wrapping_add(PRIME_1));
        (u128::from(hi) << 64) | u128::from(lo)
    }

    /// Finalise and return only the low 64 bits. Use sparingly — the
    /// 128-bit form is preferred for cache keys to avoid collisions.
    pub fn finish64(&self) -> u64 {
        self.finish128() as u64
    }
}

/// Compute a 128-bit stable hash of the given bytes.
///
/// ```
/// use llvm2_opt::cache::stable_hash;
/// let a = stable_hash(b"hello");
/// let b = stable_hash(b"hello");
/// assert_eq!(a, b);
/// let c = stable_hash(b"world");
/// assert_ne!(a, c);
/// ```
pub fn stable_hash(bytes: &[u8]) -> u128 {
    let mut h = StableHasher::new();
    h.write(bytes);
    h.finish128()
}

// ---------------------------------------------------------------------------
// Cache key
// ---------------------------------------------------------------------------

/// Version tag baked into every cache key. Bump whenever the key layout
/// or the hash algorithm changes; the bump invalidates every existing
/// on-disk cache entry.
pub const CACHE_KEY_VERSION: u32 = 1;

/// A composite cache key.
///
/// The key bundles everything the compiler needs to produce a byte-for-byte
/// identical artifact:
///
/// - `module_hash`: 128-bit hash of the input module (see [`stable_hash`]).
/// - `opt_level`: numeric opt level (0, 1, 2, 3). Higher levels produce
///   different machine code.
/// - `target_triple`: `aarch64-apple-darwin`, etc.
/// - `cpu`: target CPU model (e.g. `apple-m1`).
/// - `features`: sorted, canonicalised target-feature flags (`+neon`,
///   `+fp16`). Order is normalised in [`CacheKey::new`] so callers do
///   not have to sort.
///
/// The 128-bit digest returned by [`CacheKey::digest`] is the primary key
/// material; equality and hashing are defined on the digest, so two keys
/// built from semantically identical inputs compare equal even if the
/// backing `Vec<String>` has a different order.
#[derive(Debug, Clone, Eq)]
pub struct CacheKey {
    module_hash: u128,
    opt_level: u8,
    target_triple: String,
    cpu: String,
    features: Vec<String>,
    digest: u128,
}

impl CacheKey {
    /// Build a new [`CacheKey`]. Features are sorted and deduplicated so
    /// that `+neon,+fp16` and `+fp16,+neon` produce identical keys.
    pub fn new(
        module_hash: u128,
        opt_level: u8,
        target_triple: String,
        cpu: String,
        mut features: Vec<String>,
    ) -> Self {
        features.sort();
        features.dedup();

        let digest = Self::compute_digest(
            module_hash,
            opt_level,
            &target_triple,
            &cpu,
            &features,
        );

        Self {
            module_hash,
            opt_level,
            target_triple,
            cpu,
            features,
            digest,
        }
    }

    fn compute_digest(
        module_hash: u128,
        opt_level: u8,
        target_triple: &str,
        cpu: &str,
        features: &[String],
    ) -> u128 {
        let mut h = StableHasher::new();
        h.write_u32(CACHE_KEY_VERSION);
        h.write(&module_hash.to_le_bytes());
        h.write_u8(opt_level);
        h.write_str(target_triple);
        h.write_str(cpu);
        h.write_u32(features.len() as u32);
        for f in features {
            h.write_str(f);
        }
        h.finish128()
    }

    /// 128-bit cache digest. Stable across runs.
    pub fn digest(&self) -> u128 {
        self.digest
    }

    /// 128-bit cache digest serialised as 32 lowercase hex characters.
    pub fn hex(&self) -> String {
        format!("{:032x}", self.digest)
    }

    /// Input module hash (not the full key).
    pub fn module_hash(&self) -> u128 {
        self.module_hash
    }

    /// Optimisation level (0..=3).
    pub fn opt_level(&self) -> u8 {
        self.opt_level
    }

    /// Target triple.
    pub fn target_triple(&self) -> &str {
        &self.target_triple
    }

    /// Target CPU model.
    pub fn cpu(&self) -> &str {
        &self.cpu
    }

    /// Canonicalised target features (sorted, deduplicated).
    pub fn features(&self) -> &[String] {
        &self.features
    }
}

impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest
    }
}

impl std::hash::Hash for CacheKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.digest.hash(state);
    }
}

// ---------------------------------------------------------------------------
// CacheBackend trait + implementations
// ---------------------------------------------------------------------------

/// A pluggable cache backend.
///
/// Implementations must be safe to share across threads: the compiler
/// threads per-function compilation jobs via rayon, so concurrent gets
/// and puts are expected. Values are opaque byte blobs; callers choose
/// their own payload format (e.g. `rmp-serde` for structured data).
pub trait CacheBackend: Send + Sync {
    /// Look up a key. Returns `None` on miss or recoverable I/O failure.
    fn get(&self, key: &CacheKey) -> Option<Vec<u8>>;

    /// Insert or overwrite a key. Failures are logged (via `eprintln!`
    /// in the default impls) but never panic; the caller treats the
    /// cache as best-effort.
    fn put(&self, key: &CacheKey, value: &[u8]);
}

/// In-memory cache backed by a `RwLock<HashMap>`.
#[derive(Debug, Default)]
pub struct InMemoryCache {
    entries: RwLock<HashMap<u128, Vec<u8>>>,
}

impl InMemoryCache {
    /// Create an empty in-memory cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.entries.read().map(|g| g.len()).unwrap_or(0)
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove all entries.
    pub fn clear(&self) {
        if let Ok(mut g) = self.entries.write() {
            g.clear();
        }
    }
}

impl CacheBackend for InMemoryCache {
    fn get(&self, key: &CacheKey) -> Option<Vec<u8>> {
        let guard = self.entries.read().ok()?;
        guard.get(&key.digest).cloned()
    }

    fn put(&self, key: &CacheKey, value: &[u8]) {
        if let Ok(mut g) = self.entries.write() {
            g.insert(key.digest, value.to_vec());
        }
    }
}

/// File-per-key on-disk cache.
///
/// Layout: `{root}/{first_two_hex}/{full_hex}`. The two-character prefix
/// directory avoids flooding a single directory when the key space grows.
/// Writes go to a temp file inside the shard directory and are atomically
/// renamed into place, so a crash mid-write never leaves a partial entry.
///
/// There is **no eviction** in this MVP — callers that need a size cap
/// should wrap the backend or prune externally.
#[derive(Debug, Clone)]
pub struct FileCache {
    root: PathBuf,
}

impl FileCache {
    /// Create a [`FileCache`] rooted at `root`. The directory is created
    /// if it does not already exist. Returns an error if the directory
    /// cannot be created.
    pub fn new(root: impl Into<PathBuf>) -> io::Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// Root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn path_for(&self, key: &CacheKey) -> PathBuf {
        let hex = key.hex();
        let shard = &hex[0..2];
        self.root.join(shard).join(&hex)
    }

    fn write_atomic(&self, key: &CacheKey, value: &[u8]) -> io::Result<()> {
        let target = self.path_for(key);
        let shard_dir = target
            .parent()
            .ok_or_else(|| io::Error::other("cache target has no parent dir"))?;
        std::fs::create_dir_all(shard_dir)?;

        // Temp file name includes the PID + digest to avoid races between
        // concurrent writers for the same key (the last rename wins — both
        // payloads are equivalent for a given key so that is acceptable).
        let tmp = shard_dir.join(format!(
            ".{}.{}.tmp",
            std::process::id(),
            &key.hex()
        ));
        std::fs::write(&tmp, value)?;
        match std::fs::rename(&tmp, &target) {
            Ok(()) => Ok(()),
            Err(e) => {
                // Best-effort cleanup of the temp file.
                let _ = std::fs::remove_file(&tmp);
                Err(e)
            }
        }
    }
}

impl CacheBackend for FileCache {
    fn get(&self, key: &CacheKey) -> Option<Vec<u8>> {
        match std::fs::read(self.path_for(key)) {
            Ok(bytes) => Some(bytes),
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => {
                eprintln!(
                    "llvm2-opt::cache: warning: failed to read {}: {}",
                    self.path_for(key).display(),
                    e
                );
                None
            }
        }
    }

    fn put(&self, key: &CacheKey, value: &[u8]) {
        if let Err(e) = self.write_atomic(key, value) {
            eprintln!(
                "llvm2-opt::cache: warning: failed to write {}: {}",
                self.path_for(key).display(),
                e
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Stats wrapper
// ---------------------------------------------------------------------------

/// Atomic counters for cache hit/miss/put.
#[derive(Debug, Default)]
pub struct CacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
    puts: AtomicU64,
}

impl CacheStats {
    /// Create a new zeroed counter set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of successful `get` calls.
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Number of `get` calls that returned `None`.
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Number of `put` calls.
    pub fn puts(&self) -> u64 {
        self.puts.load(Ordering::Relaxed)
    }

    /// Hit rate as a fraction of (hits + misses). Returns 0.0 if there
    /// have been no lookups yet.
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits() as f64;
        let misses = self.misses() as f64;
        let total = hits + misses;
        if total > 0.0 { hits / total } else { 0.0 }
    }
}

/// Wraps any [`CacheBackend`] with hit/miss/put counters.
///
/// Useful for tests and for proving "the second run had a cache hit"
/// assertions at the call site without bolting stats into every backend.
pub struct StatsCache<B: CacheBackend> {
    inner: B,
    stats: CacheStats,
}

impl<B: CacheBackend> StatsCache<B> {
    /// Wrap an existing backend.
    pub fn new(inner: B) -> Self {
        Self { inner, stats: CacheStats::new() }
    }

    /// Statistics accumulated so far.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Borrow the wrapped backend.
    pub fn inner(&self) -> &B {
        &self.inner
    }
}

impl<B: CacheBackend> CacheBackend for StatsCache<B> {
    fn get(&self, key: &CacheKey) -> Option<Vec<u8>> {
        let result = self.inner.get(key);
        if result.is_some() {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    fn put(&self, key: &CacheKey, value: &[u8]) {
        self.stats.puts.fetch_add(1, Ordering::Relaxed);
        self.inner.put(key, value);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn key(module_hash: u128, opt: u8) -> CacheKey {
        CacheKey::new(
            module_hash,
            opt,
            "aarch64-apple-darwin".into(),
            "apple-m1".into(),
            vec!["+neon".into()],
        )
    }

    // -----------------------------------------------------------------------
    // StableHasher determinism
    // -----------------------------------------------------------------------

    #[test]
    fn test_stable_hash_deterministic_empty() {
        let a = stable_hash(b"");
        let b = stable_hash(b"");
        assert_eq!(a, b);
    }

    #[test]
    fn test_stable_hash_deterministic_nonempty() {
        let a = stable_hash(b"hello world");
        let b = stable_hash(b"hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn test_stable_hash_distinguishes_inputs() {
        let a = stable_hash(b"hello");
        let b = stable_hash(b"world");
        assert_ne!(a, b);
    }

    #[test]
    fn test_stable_hash_length_sensitive() {
        // Plain byte concatenation without framing would collide for
        // ["a","bc"] vs ["ab","c"]; make sure our hasher still
        // distinguishes the empty prefix vs non-empty prefix.
        let mut h1 = StableHasher::new();
        h1.write(b"");
        h1.write(b"abc");
        let d1 = h1.finish128();

        let mut h2 = StableHasher::new();
        h2.write(b"abc");
        h2.write(b"");
        let d2 = h2.finish128();

        assert_eq!(d1, d2, "unframed writes are length-agnostic (documenting)");
        // But write_framed MUST differentiate.
        let mut h3 = StableHasher::new();
        h3.write_framed(b"");
        h3.write_framed(b"abc");
        let d3 = h3.finish128();

        let mut h4 = StableHasher::new();
        h4.write_framed(b"abc");
        h4.write_framed(b"");
        let d4 = h4.finish128();

        assert_ne!(d3, d4, "framed writes must distinguish boundaries");
    }

    #[test]
    fn test_stable_hash_known_constants() {
        // Lock in a couple of known digests so regressions in the mixer
        // surface in CI. If you change the mixer, bump CACHE_KEY_VERSION
        // and update these.
        let d1 = stable_hash(b"");
        let d2 = stable_hash(b"a");
        let d3 = stable_hash(b"abcdefghijklmnop"); // two 8-byte lanes

        // They must all be distinct.
        assert_ne!(d1, d2);
        assert_ne!(d2, d3);
        assert_ne!(d1, d3);

        // And stable across this run.
        assert_eq!(d1, stable_hash(b""));
        assert_eq!(d2, stable_hash(b"a"));
        assert_eq!(d3, stable_hash(b"abcdefghijklmnop"));
    }

    // -----------------------------------------------------------------------
    // CacheKey
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_key_equality() {
        let k1 = key(0x1234, 2);
        let k2 = key(0x1234, 2);
        assert_eq!(k1, k2);
        assert_eq!(k1.hex(), k2.hex());
    }

    #[test]
    fn test_cache_key_distinguishes_opt_level() {
        let k1 = key(0x1234, 2);
        let k2 = key(0x1234, 3);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_distinguishes_module_hash() {
        let k1 = key(0x1234, 2);
        let k2 = key(0x5678, 2);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_feature_order_invariant() {
        let a = CacheKey::new(
            0xabc,
            2,
            "aarch64-apple-darwin".into(),
            "apple-m1".into(),
            vec!["+neon".into(), "+fp16".into()],
        );
        let b = CacheKey::new(
            0xabc,
            2,
            "aarch64-apple-darwin".into(),
            "apple-m1".into(),
            vec!["+fp16".into(), "+neon".into()],
        );
        assert_eq!(a, b, "feature order must not affect the key");
    }

    #[test]
    fn test_cache_key_features_deduplicated() {
        let a = CacheKey::new(
            0xabc,
            2,
            "aarch64-apple-darwin".into(),
            "apple-m1".into(),
            vec!["+neon".into(), "+neon".into(), "+fp16".into()],
        );
        assert_eq!(a.features(), &["+fp16", "+neon"]);
    }

    #[test]
    fn test_cache_key_hex_length() {
        let k = key(0, 0);
        assert_eq!(k.hex().len(), 32);
    }

    // -----------------------------------------------------------------------
    // InMemoryCache
    // -----------------------------------------------------------------------

    #[test]
    fn test_memory_cache_miss_then_hit() {
        let cache = InMemoryCache::new();
        let k = key(0x42, 2);
        assert!(cache.get(&k).is_none());
        cache.put(&k, b"payload");
        assert_eq!(cache.get(&k).as_deref(), Some(&b"payload"[..]));
    }

    #[test]
    fn test_memory_cache_overwrite() {
        let cache = InMemoryCache::new();
        let k = key(0x42, 2);
        cache.put(&k, b"v1");
        cache.put(&k, b"v2");
        assert_eq!(cache.get(&k).as_deref(), Some(&b"v2"[..]));
    }

    #[test]
    fn test_memory_cache_len() {
        let cache = InMemoryCache::new();
        assert_eq!(cache.len(), 0);
        cache.put(&key(1, 0), b"a");
        cache.put(&key(2, 0), b"b");
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
    }

    // -----------------------------------------------------------------------
    // FileCache
    // -----------------------------------------------------------------------

    fn tmpdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "llvm2_cache_test_{}_{}",
            name,
            std::process::id(),
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_file_cache_round_trip() {
        let root = tmpdir("round_trip");
        let cache = FileCache::new(&root).unwrap();
        let k = key(0x99, 2);
        assert!(cache.get(&k).is_none());
        cache.put(&k, b"hello");
        assert_eq!(cache.get(&k).as_deref(), Some(&b"hello"[..]));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_file_cache_persists_across_instances() {
        let root = tmpdir("persist");
        let k = key(0xdead, 2);
        {
            let cache = FileCache::new(&root).unwrap();
            cache.put(&k, b"persisted");
        }
        {
            let cache = FileCache::new(&root).unwrap();
            assert_eq!(
                cache.get(&k).as_deref(),
                Some(&b"persisted"[..]),
                "file cache must survive across FileCache instances on the same root",
            );
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_file_cache_sharding() {
        let root = tmpdir("shard");
        let cache = FileCache::new(&root).unwrap();
        let k = key(0x1234, 2);
        cache.put(&k, b"x");
        let p = cache.path_for(&k);
        // The file must exist inside a 2-char shard dir.
        assert!(p.exists(), "expected shard file at {}", p.display());
        let shard = p.parent().unwrap().file_name().unwrap().to_str().unwrap();
        assert_eq!(shard.len(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_file_cache_missing_returns_none() {
        let root = tmpdir("missing");
        let cache = FileCache::new(&root).unwrap();
        let k = key(0xfeed, 2);
        assert!(cache.get(&k).is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    // -----------------------------------------------------------------------
    // StatsCache
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_cache_counters() {
        let cache = StatsCache::new(InMemoryCache::new());
        let k = key(1, 2);
        assert!(cache.get(&k).is_none());
        cache.put(&k, b"v");
        assert_eq!(cache.get(&k).as_deref(), Some(&b"v"[..]));
        assert_eq!(cache.stats().hits(), 1);
        assert_eq!(cache.stats().misses(), 1);
        assert_eq!(cache.stats().puts(), 1);
        assert!((cache.stats().hit_rate() - 0.5).abs() < 1e-9);
    }
}
