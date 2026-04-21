// Integration test: cache backend semantics (InMemory + File).
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Verifies the miss → put → hit lifecycle on both backends, and that the
// on-disk backend survives across two FileCache constructions on the same
// root (as a crash-safety/atomic-rename smoke test).

use std::fs;
use std::path::PathBuf;

use llvm2_opt::{CacheBackend, CacheKey, FileCache, InMemoryCache};

fn tmp_root(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    p.push(format!("llvm2-cache-backends-{}-{}-{}-test", name, pid, nanos));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).expect("create tmp root");
    p
}

fn make_key(tag: u64) -> CacheKey {
    CacheKey::new(
        (tag as u128) ^ 0xA5A5_A5A5_A5A5_A5A5_u128,
        2,
        "aarch64-apple-darwin".to_string(),
        "apple-m1".to_string(),
        vec!["neon".to_string()],
    )
}

#[test]
fn inmemory_miss_put_hit() {
    let cache = InMemoryCache::new();
    let key = make_key(1);

    assert!(cache.get(&key).is_none(), "fresh cache is empty");
    cache.put(&key, b"hello");
    let got = cache.get(&key);
    assert_eq!(got.as_deref(), Some(b"hello".as_slice()));
}

#[test]
fn inmemory_distinct_keys_are_separate() {
    let cache = InMemoryCache::new();
    let k1 = make_key(1);
    let k2 = make_key(2);
    cache.put(&k1, b"one");
    cache.put(&k2, b"two");
    assert_eq!(cache.get(&k1).as_deref(), Some(b"one".as_slice()));
    assert_eq!(cache.get(&k2).as_deref(), Some(b"two".as_slice()));
}

#[test]
fn file_cache_miss_put_hit() {
    let root = tmp_root("miss_put_hit");
    let cache = FileCache::new(root.clone()).expect("create file cache");
    let key = make_key(7);

    assert!(cache.get(&key).is_none(), "fresh disk cache is empty");
    cache.put(&key, b"disk-value-0");
    let got = cache.get(&key);
    assert_eq!(got.as_deref(), Some(b"disk-value-0".as_slice()));

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn file_cache_persists_across_new_handles() {
    let root = tmp_root("persist");
    let key = make_key(42);

    {
        let a = FileCache::new(root.clone()).expect("create file cache");
        a.put(&key, b"persisted");
    }
    // Second handle on the same root must see the write from the first.
    let b = FileCache::new(root.clone()).expect("reopen file cache");
    assert_eq!(b.get(&key).as_deref(), Some(b"persisted".as_slice()));

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn file_cache_overwrite_wins() {
    let root = tmp_root("overwrite");
    let cache = FileCache::new(root.clone()).expect("create file cache");
    let key = make_key(3);
    cache.put(&key, b"first");
    cache.put(&key, b"second");
    assert_eq!(cache.get(&key).as_deref(), Some(b"second".as_slice()));
    let _ = fs::remove_dir_all(&root);
}
