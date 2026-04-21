// Integration test: CegisSuperoptPass cache hit/miss + MachinePass wiring.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// tla2 supremacy blocker 8 (issue #395): repeat compilations of the same
// (function, target, cpu, features) tuple must reuse cached CEGIS results.
// The first run is a miss + put; the second run on an identical function
// must be a cache hit (no solver work).

use std::sync::Arc;

use llvm2_ir::MachFunction;
use llvm2_ir::function::Signature;
use llvm2_opt::{CacheBackend, InMemoryCache, MachinePass};
use llvm2_verify::{CegisSuperoptConfig, CegisSuperoptPass};

fn empty_func(name: &str) -> MachFunction {
    MachFunction::new(name.to_string(), Signature::new(vec![], vec![]))
}

fn make_config(cache: Option<Arc<dyn CacheBackend>>, budget_sec: u64) -> CegisSuperoptConfig {
    CegisSuperoptConfig {
        budget_sec,
        per_query_ms: 500,
        target_triple: "aarch64-apple-darwin".to_string(),
        cpu: "apple-m1".to_string(),
        features: vec!["neon".to_string(), "fp-armv8".to_string()],
        opt_level: 2,
        cache,
        trace: None,
    }
}

#[test]
fn disabled_pass_is_noop() {
    let cfg = CegisSuperoptConfig::default();
    assert!(!cfg.is_enabled());
    let mut pass = CegisSuperoptPass::new(cfg);
    let mut func = empty_func("f");
    let mutated = pass.run(&mut func);
    assert!(!mutated, "pass must not report mutation when disabled");
    assert_eq!(pass.stats().functions_seen, 0);
}

#[test]
fn repeat_compilation_uses_cache() {
    let cache: Arc<dyn CacheBackend> = Arc::new(InMemoryCache::new());
    let cfg = make_config(Some(cache.clone()), 1);

    // First compile: miss + put.
    let mut pass1 = CegisSuperoptPass::new(cfg.clone());
    let mut f1 = empty_func("bfs_visit");
    pass1.run(&mut f1);
    assert_eq!(pass1.stats().cache_misses, 1);
    assert_eq!(pass1.stats().cache_hits, 0);
    assert_eq!(pass1.stats().cache_puts, 1);

    // Second compile with a fresh pass on an identical function: hit.
    let mut pass2 = CegisSuperoptPass::new(cfg);
    let mut f2 = empty_func("bfs_visit");
    pass2.run(&mut f2);
    assert_eq!(pass2.stats().cache_misses, 0);
    assert_eq!(pass2.stats().cache_hits, 1);
    assert_eq!(pass2.stats().cache_puts, 0);
}

#[test]
fn different_function_names_are_separate_cache_entries() {
    let cache: Arc<dyn CacheBackend> = Arc::new(InMemoryCache::new());
    let cfg = make_config(Some(cache.clone()), 1);

    let mut pass = CegisSuperoptPass::new(cfg.clone());
    let mut alpha = empty_func("alpha");
    pass.run(&mut alpha);
    let mut beta = empty_func("beta");
    pass.run(&mut beta);
    // Two distinct keys ➜ two misses.
    assert_eq!(pass.stats().cache_misses, 2);
    assert_eq!(pass.stats().cache_hits, 0);
}

#[test]
fn different_features_are_separate_cache_entries() {
    let cache: Arc<dyn CacheBackend> = Arc::new(InMemoryCache::new());

    let cfg_a = make_config(Some(cache.clone()), 1);
    let mut cfg_b = make_config(Some(cache.clone()), 1);
    cfg_b.features.push("extra-feature".to_string());

    let mut pass_a = CegisSuperoptPass::new(cfg_a);
    let mut pass_b = CegisSuperoptPass::new(cfg_b);
    let mut f = empty_func("f");
    pass_a.run(&mut f);
    let mut f2 = empty_func("f");
    pass_b.run(&mut f2);

    assert_eq!(pass_a.stats().cache_misses, 1);
    assert_eq!(pass_b.stats().cache_misses, 1);
}

#[test]
fn pass_has_stable_name() {
    let pass = CegisSuperoptPass::new(CegisSuperoptConfig::default());
    assert_eq!(pass.name(), "CegisSuperoptPass");
}
