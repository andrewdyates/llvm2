// Integration test: stable_hash determinism across hasher constructions.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// tla2 supremacy blocker 7 (issue #395): the compilation cache must be
// deterministic across runs. If `stable_hash` on the same input returns
// different bytes twice, every repeat compilation is a cache miss — tla2's
// CEGIS-on-by-default strategy fails.
//
// This test hashes a fixed byte string twice via two freshly constructed
// `StableHasher` instances and via the `stable_hash` free function, and
// asserts all four digests are equal.

use llvm2_opt::{CacheKey, InMemoryCache, StableHasher, stable_hash};
use llvm2_opt::CacheBackend;

#[test]
fn stable_hash_is_deterministic() {
    let input = b"llvm2 cache determinism smoke input 0123456789";

    // Path 1: fresh StableHasher ➜ write ➜ finish128.
    let mut h1 = StableHasher::new();
    h1.write(input);
    let a = h1.finish128();

    // Path 2: second fresh StableHasher ➜ write ➜ finish128.
    let mut h2 = StableHasher::new();
    h2.write(input);
    let b = h2.finish128();

    // Path 3: the convenience free function.
    let c = stable_hash(input);

    // Path 4: framed variant for a different input MUST differ.
    let mut h3 = StableHasher::new();
    h3.write_framed(input);
    let d = h3.finish128();

    assert_eq!(a, b, "two fresh hashers with identical input must agree");
    assert_eq!(a, c, "stable_hash must match the fresh-hasher pipeline");
    assert_ne!(a, d, "framed length prefix must distinguish from raw write");
}

#[test]
fn cache_key_is_deterministic_across_feature_orders() {
    // The design doc calls out feature-order invariance as a cache-hit
    // correctness property. Re-order should NOT produce a new key.
    let k1 = CacheKey::new(
        0xABCD_EF01_u128,
        2,
        "aarch64-apple-darwin".to_string(),
        "apple-m1".to_string(),
        vec!["neon".to_string(), "fp-armv8".to_string(), "crc".to_string()],
    );
    let k2 = CacheKey::new(
        0xABCD_EF01_u128,
        2,
        "aarch64-apple-darwin".to_string(),
        "apple-m1".to_string(),
        vec!["crc".to_string(), "fp-armv8".to_string(), "neon".to_string()],
    );
    assert_eq!(k1, k2, "feature order must not affect the cache key");
}

#[test]
fn cache_key_round_trips_through_inmemory_backend() {
    let cache = InMemoryCache::new();
    let key = CacheKey::new(
        0x1111_2222_3333_4444_u128,
        2,
        "aarch64-apple-darwin".to_string(),
        "apple-m1".to_string(),
        vec!["neon".to_string()],
    );
    assert!(cache.get(&key).is_none(), "miss on fresh cache");
    cache.put(&key, b"payload-abc");
    assert_eq!(cache.get(&key).as_deref(), Some(b"payload-abc".as_slice()));
}
